#!/usr/bin/env python3
"""
Run a tool use task with Gemini-2.5-Pro to predict and refine actions.
"""
import os
import json
import random
import argparse
import copy
from pyGameWorld.world import loadFromDict
from pyGameWorld import PGWorld, ToolPicker
from pyGameWorld.viewer import demonstrateTPPlacement, demonstrateWorld, saveWorld, demonstrateWorld_and_save_gif
import numpy as np
import pygame as pg
from gemini_api import GeminiClient
from PIL import Image

from prompts import prompt_init, prompt_invalid, prompt_check, prompt_video
from schemas import ToolUseAction, ToolUseActionCheck, ToolUseVideoAction
from utils import (
    save_json,
)

# Fix random seed for reproducibility
random.seed(0)

# load the file
JSON_FILE = './Trials/Original/Basic.json'

def main(args):
    """
    Main entry point: sets up directories, loads tasks, initializes simulator and Gemini agent,
    then iteratively predicts actions until the task is solved or trial limit is reached.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Parse command-line arguments
    output_root = args.output_root
    exp_name = f"fold_task"
    initial_image_filename = 'initial_image.png'
    print("Experiment name:", exp_name)
    base_dir = os.path.join(output_root, exp_name)
    os.makedirs(base_dir, exist_ok=True)

    # Create a new run folder (e.g., run_001) based on existing directories
    run_id = len(os.listdir(base_dir))
    run_dir = os.path.join(base_dir, f"run_{run_id:03d}")
    os.makedirs(run_dir, exist_ok=True)

    # Prepare subdirectories for visuals and responses
    visuals_dir = os.path.join(run_dir, 'visuals')
    responses_dir = os.path.join(run_dir, 'responses')
    os.makedirs(visuals_dir, exist_ok=True)
    os.makedirs(responses_dir, exist_ok=True)

    # Load the world from a JSON file
    with open(JSON_FILE, 'r') as f:
        pgw_dict = json.load(f)

    # Store a deep copy of the original world dictionary for visualization
    original_pgw_dict = copy.deepcopy(pgw_dict)

    # Turn this into a toolpicker game
    tp = ToolPicker(pgw_dict)

    # Initialize pygame and create a screen for visualization
    pg.init()
    screen = pg.display.set_mode((600,600))
    screen.fill((255,255,255)) 

    # Load the world from the dictionary
    world = loadFromDict(pgw_dict['world'])
    tools = pgw_dict['tools']
   
    # screen the initial image of the world
    image = saveWorld(world, initial_image_filename, tools)
    
    # Configure the Gemini agent for inference
    agent = GeminiClient(upload_file=True, fps=3.0)

    # Round 0: initial prediction based on the first image
    prompt = prompt_init

    result = agent.inference_image(
        initial_image_filename,
        prompt,
        schema=ToolUseAction
    )
    print("Predicted action:", result)

    toolname = result['toolname']
    position = result['position']
    pred_action = {'toolname': toolname, 'position': position}
    print(f"Tool: {toolname}, Position: {position}")

    # Simulate the action using the ToolPicker
    path_dict, success, time_to_success = tp.observePlacementPath(
        toolname=toolname,
        position=position,
        maxtime=20.
    )
    print("Action was successful?", success)
    print("Time to success:", time_to_success)
    
    # Save visualization of the action result
    try:
        demonstrateTPPlacement(tp, toolname, position, path=os.path.join(visuals_dir, 'round_0.png'))
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")
        # Fallback: save the current world state
        current_world = loadFromDict(original_pgw_dict['world'])
        saveWorld(current_world, os.path.join(visuals_dir, 'round_0.png'), original_pgw_dict['tools'])
    
    # if path_dict is None, the action was invalid
    invalid = path_dict is None
    save_json(result, path=os.path.join(responses_dir, 'round_0.json'))

    # Iterative correction loop: up to a trial limit
    trial_limit = 5
    count = 1
    checked = False
    img = 'round_0.png'  # Use the initial image for context

    while count <= trial_limit:
        # If action was invalid, ask for a corrected action via text prompt
        if invalid:
            print("Invalid action detected. Asking Gemini to correct.")
            prompt = prompt_invalid.replace('<PREDICTED_ACTION>', str(pred_action))
            result = agent.inference_text(
                prompt,
                schema=ToolUseAction,
                history=True
            )
            toolname = result['toolname']
            position = result['position']
            pred_action = {'toolname': toolname, 'position': position}
            path_dict, success, time_to_success = tp.observePlacementPath(
                toolname=toolname,
                position=position,
                maxtime=20.
            )
            invalid = path_dict is None
            try:
                demonstrateTPPlacement(tp, toolname, position, path=os.path.join(visuals_dir, f'round_{count}.png'))
            except Exception as e:
                print(f"Warning: Could not generate visualization: {e}")
                # Fallback: save the current world state
                current_world = loadFromDict(original_pgw_dict['world'])
                saveWorld(current_world, os.path.join(visuals_dir, f'round_{count}.png'), original_pgw_dict['tools'])
            save_json(result, path=os.path.join(responses_dir, f'round_{count}.json'))

        # If valid but not yet correct, optionally check with a second image prompt
        elif not success and not invalid and not checked:
            print("Valid but unsuccessful. Asking Gemini to check.")
            # Use the first simulation image for additional context
            prompt = prompt_check.replace('<PREDICTED_ACTION>', str(pred_action))
            result = agent.inference_image(
                os.path.join(visuals_dir, f'round_{count - 1}.png'),
                prompt,
                schema=ToolUseAction,
                history=True
            )
            toolname = result['toolname']
            position = result['position']
            pred_action = {'toolname': toolname, 'position': position}
            path_dict, success, time_to_success = tp.observePlacementPath(
                toolname=toolname,
                position=position,
                maxtime=20.
            )
            invalid = path_dict is None
            checked = True
            try:
                demonstrateTPPlacement(tp, toolname, position, path=os.path.join(visuals_dir, f'round_{count}.png'))
            except Exception as e:
                print(f"Warning: Could not generate visualization: {e}")
                # Fallback: save the current world state
                current_world = loadFromDict(original_pgw_dict['world'])
                saveWorld(current_world, os.path.join(visuals_dir, f'round_{count}.png'), original_pgw_dict['tools'])
            save_json(result, path=os.path.join(responses_dir, f'round_{count}.json'))

        # If still not solved but was previously checked, try a video prompt for more context
        elif not success and checked:
            print("Still unsuccessful. Showing video context.")
            world = loadFromDict(original_pgw_dict['world'])  # reset world state from original
            gif_path = os.path.join(visuals_dir, f'round_{count}.gif')
            demonstrateWorld_and_save_gif(world, gif_filename=gif_path)
            prompt = prompt_video.replace('<PREDICTED_ACTION>', str(pred_action))
            result = agent.inference_video(
                os.path.join(visuals_dir, f'round_{count - 1}.png'),
                prompt,
                schema=ToolUseAction,
                history=True
            )
            toolname = result['toolname']
            position = result['position']
            pred_action = {'toolname': toolname, 'position': position}
            path_dict, success, time_to_success = tp.observePlacementPath(
                toolname=toolname,
                position=position,
                maxtime=20.
            )
            invalid = path_dict is None
            checked = False
            try:
                demonstrateTPPlacement(tp, toolname, position, path=os.path.join(visuals_dir, f'round_{count}.png'))
            except Exception as e:
                print(f"Warning: Could not generate visualization: {e}")
                # Fallback: save the current world state
                current_world = loadFromDict(original_pgw_dict['world'])
                saveWorld(current_world, os.path.join(visuals_dir, f'round_{count}.png'), original_pgw_dict['tools'])
            save_json(result, path=os.path.join(responses_dir, f'round_{count}.json'))

        if success:
            print("Task solved!")
            break

        count += 1

    print(f"Task completed after {count} rounds. Success: {success}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run task with Gemini-2.5-Pro to predict and refine actions."
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='./output',
        help='Root directory for saving results'
    )
    args = parser.parse_args()
    main(args)
