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
from pyGameWorld.viewer import demonstrateTPPlacement, demonstrateWorld, saveWorld, demonstrateWorld_and_save_video
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
    invalid_attempts = 0  # Track consecutive invalid attempts
    img = 'round_0.png'  # Use the initial image for context

    while count <= trial_limit:
        # If action was invalid, ask for a corrected action via text prompt
        if invalid:
            invalid_attempts += 1
            print(f"Invalid action detected (attempt {invalid_attempts}). Asking Gemini to correct.")
            
            # After 2 consecutive invalid attempts, show MP4 for better context
            if invalid_attempts >= 2 and not checked:
                print("Multiple invalid attempts. Showing video context for better understanding.")
                world = loadFromDict(original_pgw_dict['world'])  # reset world state from original
                video_path = os.path.join(visuals_dir, f'round_{count}_invalid_context.mp4')
                demonstrateWorld_and_save_video(world, video_filename=video_path, max_frames=5)
                prompt = prompt_video.replace('<PREDICTED_ACTION>', str(pred_action))
                
                # Check if the MP4 file exists before using it
                if os.path.exists(video_path):
                    print(f"Using MP4 for context: {video_path}")
                    result = agent.inference_video(
                        video_path,  
                        prompt,
                        schema=ToolUseVideoAction,  
                        history=True
                    )
                else:
                    print("Warning: MP4 file not found, using fallback text inference")
                    prompt = prompt_invalid.replace('<PREDICTED_ACTION>', str(pred_action))
                    result = agent.inference_text(
                        prompt,
                        schema=ToolUseAction,
                        history=True
                    )
                checked = True  # Mark as checked to avoid repeating this
            else:
                # Standard invalid action correction
                prompt = prompt_invalid.replace('<PREDICTED_ACTION>', str(pred_action))
                result = agent.inference_text(
                    prompt,
                    schema=ToolUseAction,
                    history=True
                )
            
            if result is not None:
                toolname = result['toolname']
                position = result['position']
                pred_action = {'toolname': toolname, 'position': position}
                path_dict, success, time_to_success = tp.observePlacementPath(
                    toolname=toolname,
                    position=position,
                    maxtime=20.
                )
                invalid = path_dict is None
                if not invalid:
                    invalid_attempts = 0  # Reset counter on valid action
                try:
                    demonstrateTPPlacement(tp, toolname, position, path=os.path.join(visuals_dir, f'round_{count}.png'))
                except Exception as e:
                    print(f"Warning: Could not generate visualization: {e}")
                    # Fallback: save the current world state
                    current_world = loadFromDict(original_pgw_dict['world'])
                    saveWorld(current_world, os.path.join(visuals_dir, f'round_{count}.png'), original_pgw_dict['tools'])
                save_json(result, path=os.path.join(responses_dir, f'round_{count}.json'))
            else:
                print("No result from agent, treating as invalid action")
                invalid = True

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
            video_path = os.path.join(visuals_dir, f'round_{count}.mp4')
            demonstrateWorld_and_save_video(world, video_filename=video_path, max_frames=5)
            prompt = prompt_video.replace('<PREDICTED_ACTION>', str(pred_action))
            
            # check if the MP4 file exists before using it
            if os.path.exists(video_path):
                result = agent.inference_video(
                    video_path,  
                    prompt,
                    schema=ToolUseVideoAction,  
                    history=True
                )
            else:
                print("Warning: MP4 file not found, using fallback image inference")
                # Fallback to the last image if MP4 is not available
                fallback_image = os.path.join(visuals_dir, f'round_{count - 1}.png')
                result = agent.inference_image(
                    fallback_image,
                    prompt,
                    schema=ToolUseAction,
                    history=True
                )
            
            if result is not None:
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
            else:
                print("Warning: No result from agent, skipping this round")
                invalid = True

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
