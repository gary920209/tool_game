#!/usr/bin/env python3
"""
Run PHyre tasks with Gemini API to predict actions and evaluate simulations.
"""
import os
import random
import argparse
from pyGameWorld import PGWorld, ToolPicker
from pyGameWorld.viewer import demonstrateTPPlacement, demonstrateWorld, saveWorld
import numpy as np
# from gemini_api import GeminiClient
import pygame as pg
from gemini_api import GeminiClient

# Local modules
from prompts import prompt_init, prompt_invalid, prompt_check, prompt_video
from schemas import ToolUseAction, ToolUseActionCheck, ToolUseVideoAction

# Fix random seed for reproducibility
random.seed(0)


def main(args):
    """
    Main entry point: sets up directories, loads tasks, initializes simulator and Gemini agent,
    then iteratively predicts actions until the task is solved or trial limit is reached.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Parse command-line arguments
    output_root = args.output_root
    image_filename ="image.png"
    # Construct experiment name and output directories
    exp_name = f"fold_task"
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

    # Make the basic world
    pgw = PGWorld(dimensions=(600,600), gravity=200)
    # Name, [left, bottom, right, top], color, density (0 is static)
    pgw.addBox('Table', [0,0,300,200],(0,0,0),0)
    # Name, points (counter-clockwise), width, color, density
    pgw.addContainer('Goal', [[330,100],[330,5],[375,5],[375,100]], 10, (0,255,0), (0,0,0), 0)
    # Name, position of center, radius, color, (density is 1 by default)
    pgw.addBall('Ball',[100,215],15,(0,0,255))
    # Sets up the condition that "Ball" must go into "Goal" and stay there for 2 seconds
    pgw.attachSpecificInGoal("Goal","Ball",2.)
    pgw_dict = pgw.toDict()
    # pgw_dict.setdefault('defaults', {'elasticity': 0.0, 'friction': 0.5, 'density': 1.0})

    tools = {
    "obj1" : [[[-30,-15],[-30,15],[30,15],[0,-15]]],
    "obj2" : [[[-20,0],[0,20],[20,0],[0,-20]]],
    "obj3" : [[[-40,-5],[-40,5],[40,5],[40,-5]]]
    }

    '''
    # Save to a file
    # Can reload with loadFromDict function in pyGameWorld

    with open('basic_trial.json','w') as jfl:
        json.dump(pgw_dict, jfl)
    '''
    # Turn this into a toolpicker game
    tp = ToolPicker({'world': pgw_dict, 'tools':tools})
    pg.init()
    screen = pg.display.set_mode((600,600))
    screen.fill((255,255,255)) 
    # screen the initial image of the world
    image = saveWorld(pgw, image_filename, tools)
    # Configure the Gemini agent for inference
    agent = GeminiClient(upload_file=True, fps=3.0)

    # Round 0: initial prediction based on the first image
    prompt = prompt_init

    result = agent.inference_image(
        image_filename,
        prompt,
        schema=ToolUseAction
    )
    print("Predicted action:", result)

    toolname = result['toolname']
    position = result['position']
    print(f"Tool: {toolname}, Position: {position}")
    path_dict, success, time_to_success = tp.observePlacementPath(
        toolname=toolname,
        position=position,
        maxtime=20.
    )
    print("Action was successful?", success)
    # print("Paths:", path_dict)
    print("Time to success:", time_to_success)
    demonstrateTPPlacement(tp, toolname, position)


#     # Log outcomes and save response
#     solved, invalid = log_simulation_results(
#         pred_action, task_index, tasks, simulation
#     )
#     save_json(result, path=os.path.join(responses_dir, 'round_0.json'))

#     # Iterative correction loop: up to a trial limit
#     trial_limit = 5
#     count = 1
#     checked = False

#     while count <= trial_limit:
#         # If action was invalid, ask for a corrected action via text prompt
#         if invalid:
#             prompt = prompt_invalid.replace('<PREDICTED_ACTION>', str(pred_action))
#             result = agent.inference_text(
#                 prompt,
#                 schema=ActionSchema,
#                 history=True
#             )
#             print("Corrected action:", result['action'])

#         # If valid but not yet correct, optionally check with a second image prompt
#         elif not solved and not invalid and not checked:
#             # Use the first simulation image for additional context
#             img = phyre.observations_to_float_rgb(simulation.images[0])
#             prompt = prompt_check.replace('<PREDICTED_ACTION>', str(pred_action))
#             save_image(img, path=os.path.join(visuals_dir, f'round_{count}.png'))
#             result = agent.inference_image(
#                 img,
#                 prompt,
#                 schema=ActionSchemaCheck,
#                 history=True
#             )
#             print("Checked action:", result['action'])
#             checked = result.get('correct', False)

#         # If still not solved but was previously checked, try a video prompt for more context
#         elif not solved and not invalid and checked:
#             prompt = prompt_video.replace('<PREDICTED_ACTION>', str(pred_action))
#             save_gif(simulation.images, path=os.path.join(visuals_dir, f'round_{count}.gif'))
#             result = agent.inference_video(
#                 convert_to_np(simulation.images),
#                 prompt,
#                 schema=VideoActionSchema,
#                 history=True
#             )
#             print("Video-based action:", result['action'])
#             checked = False

#         # Update prediction and re-simulate
#         pred_action = np.array(result['action'], dtype=np.float32)
#         simulation = simulator.simulate_action(
#             task_index,
#             pred_action,
#             need_images=True,
#             need_featurized_objects=True
#         )
#         solved, invalid = log_simulation_results(
#             pred_action, task_index, tasks, simulation
#         )
#         save_json(result, path=os.path.join(responses_dir, f'round_{count}.json'))

#         # If solved, exit loop
#         if solved:
#             print("Task solved!")
#             break

#         count += 1

#     # Save final simulation sequence as a GIF for review
#     save_gif(
#         simulation.images,
#         path=os.path.join(visuals_dir, f'round_final.gif')
#     )


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
