#!/usr/bin/env python3
"""
Run PHyre tasks with Gemini API to predict actions and evaluate simulations.
"""
import os
import random
import argparse

import numpy as np
import phyre
from gemini_api import GeminiClient

# Local modules
from gpt_api import *  # consider importing only needed functions
from prompts import prompt_init, prompt_invalid, prompt_check, prompt_video
from schemas import ActionSchema, ActionSchemaCheck, VideoActionSchema
from utils import (
    save_image,
    save_gif,
    save_json,
    convert_to_np,
    log_simulation_results,
)

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
    eval_setup = args.eval_setups
    fold_id = args.fold_id
    task_index = args.task_index
    output_root = args.output_root

    # Construct experiment name and output directories
    exp_name = f"{eval_setup}_fold_{fold_id}_task_{task_index}"
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

    # Load train/dev/test splits for the given evaluation setup and fold
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
    print(
        'Size of splits:',
        f'train={len(train_tasks)}, dev={len(dev_tasks)}, test={len(test_tasks)}'
    )

    # Determine the action tier (difficulty level) for this evaluation setup
    action_tier = phyre.eval_setup_to_action_tier(eval_setup)
    print('Action tier for', eval_setup, 'is', action_tier)

    # Limit to the first 50 dev tasks for quick debugging
    tasks = dev_tasks[:50]

    # Initialize the PHyre simulator for these tasks
    simulator = phyre.initialize_simulator(tasks, action_tier)

    # Select the specific task by index
    task_id = simulator.task_ids[task_index]
    print('Task ID:', task_id)

    # Get the initial scene observation (as RGB image)
    initial_scene = simulator.initial_scenes[task_index]
    init_img = phyre.observations_to_float_rgb(initial_scene)
    print(
        f'Initial scene: shape={initial_scene.shape}, dtype={initial_scene.dtype}'
    )

    # Configure the Gemini agent for inference
    agent = GeminiClient(upload_file=True, fps=3.0)

    # Round 0: initial prediction based on the first image
    prompt = prompt_init
    save_image(init_img, path=os.path.join(visuals_dir, 'round_0.png'))

    result = agent.inference_image(
        init_img,
        prompt,
        schema=ActionSchema
    )
    print("Predicted action:", result['action'])

    # Convert action to numpy array and simulate it
    pred_action = np.array(result['action'], dtype=np.float32)
    simulation = simulator.simulate_action(
        task_index,
        pred_action,
        need_images=True,
        need_featurized_objects=True
    )

    # Log outcomes and save response
    solved, invalid = log_simulation_results(
        pred_action, task_index, tasks, simulation
    )
    save_json(result, path=os.path.join(responses_dir, 'round_0.json'))

    # Iterative correction loop: up to a trial limit
    trial_limit = 5
    count = 1
    checked = False

    while count <= trial_limit:
        # If action was invalid, ask for a corrected action via text prompt
        if invalid:
            prompt = prompt_invalid.replace('<PREDICTED_ACTION>', str(pred_action))
            result = agent.inference_text(
                prompt,
                schema=ActionSchema,
                history=True
            )
            print("Corrected action:", result['action'])

        # If valid but not yet correct, optionally check with a second image prompt
        elif not solved and not invalid and not checked:
            # Use the first simulation image for additional context
            img = phyre.observations_to_float_rgb(simulation.images[0])
            prompt = prompt_check.replace('<PREDICTED_ACTION>', str(pred_action))
            save_image(img, path=os.path.join(visuals_dir, f'round_{count}.png'))
            result = agent.inference_image(
                img,
                prompt,
                schema=ActionSchemaCheck,
                history=True
            )
            print("Checked action:", result['action'])
            checked = result.get('correct', False)

        # If still not solved but was previously checked, try a video prompt for more context
        elif not solved and not invalid and checked:
            prompt = prompt_video.replace('<PREDICTED_ACTION>', str(pred_action))
            save_gif(simulation.images, path=os.path.join(visuals_dir, f'round_{count}.gif'))
            result = agent.inference_video(
                convert_to_np(simulation.images),
                prompt,
                schema=VideoActionSchema,
                history=True
            )
            print("Video-based action:", result['action'])
            checked = False

        # Update prediction and re-simulate
        pred_action = np.array(result['action'], dtype=np.float32)
        simulation = simulator.simulate_action(
            task_index,
            pred_action,
            need_images=True,
            need_featurized_objects=True
        )
        solved, invalid = log_simulation_results(
            pred_action, task_index, tasks, simulation
        )
        save_json(result, path=os.path.join(responses_dir, f'round_{count}.json'))

        # If solved, exit loop
        if solved:
            print("Task solved!")
            break

        count += 1

    # Save final simulation sequence as a GIF for review
    save_gif(
        simulation.images,
        path=os.path.join(visuals_dir, f'round_final.gif')
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PHyre task with Gemini-2.5-Pro to predict and refine actions."
    )
    parser.add_argument(
        '--eval_setups',
        type=str,
        default='ball_within_template',
        help='PHyre evaluation setup name'
    )
    parser.add_argument(
        '--fold_id',
        type=int,
        default=0,
        help='Fold index for cross-validation'
    )
    parser.add_argument(
        '--task_index',
        type=int,
        default=3,
        help='Index of the task within the dev split'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='./output',
        help='Root directory for saving results'
    )
    args = parser.parse_args()
    main(args)
