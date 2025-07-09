#!/usr/bin/env python3
"""
Run a tool use task with Gemini-2.5-Pro to predict and refine actions.
"""
import os
import json
import random
import argparse
from pyGameWorld.world import loadFromDict
from pyGameWorld import PGWorld, ToolPicker
from pyGameWorld.viewer import demonstrateTPPlacement, demonstrateWorld, saveWorld, makeImageArray
import numpy as np
import pygame as pg
from gemini_api import GeminiClient
from openai_api import OpenAIClient

from prompts import prompt_init, prompt_invalid, prompt_check, prompt_video, prompt_feedback, prompt_video_feedback
from schemas import ToolUseAction, ToolUseActionCheck, ToolUseVideoAction

# Fix random seed for reproducibility
random.seed(0)

# load the file
JSON_FILE = './Trials/Original/Basic.json'

def pygame_surface_to_numpy(surface):
    """Convert pygame surface to numpy array"""
    # Get the raw pixel data
    pixel_data = pg.surfarray.array3d(surface)
    # Convert from (width, height, 3) to (height, width, 3)
    pixel_data = np.transpose(pixel_data, (1, 0, 2))
    # Convert to float in [0, 1] range
    pixel_data = pixel_data.astype(np.float32) / 255.0
    return pixel_data

def save_simulation_images(images, attempt, visuals_dir):
    """Save only the final simulation image for feedback"""
    # Return the final image for feedback
    final_img = images[-1] if images else None
    if final_img:
        final_img_path = os.path.join(visuals_dir, f'attempt_{attempt:03d}_final.png')
        pg.image.save(final_img, final_img_path)
        return final_img_path
    
    return None

def create_simulation_video(images, attempt, visuals_dir, fps=80):
    """Create video from simulation images and return video path"""
    if not images:
        return None
    
    import cv2
    
    # Convert pygame surfaces to numpy arrays
    frames = []
    for img in images:
        # Convert pygame surface to numpy array
        pixel_data = pg.surfarray.array3d(img)
        # Convert from (width, height, 3) to (height, width, 3)
        pixel_data = np.transpose(pixel_data, (1, 0, 2))
        # Convert to BGR for OpenCV
        pixel_data = cv2.cvtColor(pixel_data, cv2.COLOR_RGB2BGR)
        frames.append(pixel_data)
    
    # Create video
    video_path = os.path.join(visuals_dir, f'attempt_{attempt:03d}_simulation.mp4')
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Simulation video saved to: {video_path}")
    return video_path

def main(args):
    """
    Main entry point: sets up directories, loads tasks, initializes simulator and agent,
    then iteratively predicts actions until the task is solved or trial limit is reached.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Parse command-line arguments
    output_root = args.output_root
    max_attempts = args.max_attempts
    use_video_feedback = args.use_video_feedback
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

    # Turn this into a toolpicker game
    tp = ToolPicker(pgw_dict)

    # Initialize pygame and create a screen for visualization
    pg.init()
    screen = pg.display.set_mode((600,600))
    screen.fill((255,255,255)) 

    # Load the world from the dictionary
    world = loadFromDict(pgw_dict['world'])
    tools = pgw_dict['tools']
   
    # Save the initial image of the world
    image = saveWorld(world, initial_image_filename, tools)
    
    # Configure the agent for inference
    if args.model == 'gemini':
        agent = GeminiClient(upload_file=True, fps=3.0)
    elif args.model == 'openai':
        agent = OpenAIClient(upload_file=False, fps=3.0)
    else:
        raise ValueError(f"Invalid model: {args.model}")

    # Initialize tracking variables
    attempt = 0
    solved = False
    invalid_action = False
    last_action = None
    use_video = False
    
    print(f"Starting tool use task with maximum {max_attempts} attempts")
    print("=" * 50)

    while attempt < max_attempts and not solved:
        attempt += 1
        print(f"\n--- Attempt {attempt}/{max_attempts} ---")
        
        # Determine which prompt to use based on the situation
        if attempt == 1:
            # First attempt: use initial prompt
            prompt = prompt_init
            current_image = initial_image_filename
            schema = ToolUseAction
            history = False
            use_video = False
        elif invalid_action and last_action is not None:
            # Previous action was invalid: use invalid action prompt
            prompt = prompt_invalid.replace('<PREDICTED_ACTION>', str(last_action))
            current_image = initial_image_filename
            schema = ToolUseAction
            history = True
            use_video = False
        elif last_action is not None:
            # Previous action was valid but didn't solve: use feedback prompt with simulation result
            if use_video_feedback and 'last_simulation_video' in locals():
                # Use video feedback
                prompt = prompt_video_feedback.replace('<PREDICTED_ACTION>', str(last_action))
                current_video = last_simulation_video
                schema = ToolUseVideoAction
                history = True
                use_video = True
            else:
                # Use image feedback
                prompt = prompt_feedback.replace('<PREDICTED_ACTION>', str(last_action))
                # Ensure we have a valid image file
                if 'last_simulation_final_image' in locals() and last_simulation_final_image is not None and os.path.exists(last_simulation_final_image):
                    current_image = last_simulation_final_image
                else:
                    current_image = initial_image_filename
                schema = ToolUseAction
                history = True
                use_video = False
        else:
            # Fallback for unexpected state
            prompt = prompt_init
            current_image = initial_image_filename
            schema = ToolUseAction
            history = False
            use_video = False

        # Get prediction from the agent
        print(f"Getting prediction for attempt {attempt}...")
        if use_video:
            # Use video inference
            result = agent.inference_video(
                current_video,
                prompt,
                schema=schema,
                history=history
            )
        else:
            # Use image inference
            result = agent.inference_image(
                current_image,
                prompt,
                schema=schema,
                history=history
            )
        
        if result is None:
            print("Failed to get prediction from agent")
            invalid_action = True
            continue
            
        print("Predicted action:", result)
        
        # Validate action format
        toolname = result['toolname']
        position = result['position']
        
        # Validate toolname
        if toolname not in ['obj1', 'obj2', 'obj3']:
            print(f"Invalid toolname: {toolname}. Must be one of: obj1, obj2, obj3")
            invalid_action = True
            continue
        
        # Validate position
        if not isinstance(position, list) or len(position) != 2:
            print(f"Invalid position format: {position}. Must be a list of 2 numbers")
            invalid_action = True
            continue
        
        x, y = position
        if not (0 <= x <= 600 and 0 <= y <= 600):
            print(f"Position out of bounds: {position}. Must be within [0, 600] x [0, 600]")
            invalid_action = True
            continue
        
        last_action = result
        
        print(f"Tool: {toolname}, Position: {position}")
        
        # Simulate the action using the ToolPicker
        print("Simulating action...")
        path_dict, success, time_to_success, world_dict = tp.observeFullPlacementPath(
            toolname=toolname,
            position=position,
            maxtime=5.,
            returnDict=True
        )

        
        print(f"Action was successful? {success}")
        print(f"Time to success: {time_to_success}")
        
        # Capture simulation images for feedback
        simulation_images = []
        if path_dict is not None and len(path_dict) > 0:
            # Generate image sequence from the simulation path
            simulation_images = makeImageArray(world_dict, path_dict, sample_ratio=1)
            print(f"Captured {len(simulation_images)} simulation frames")
        else:
            print("No path data available for image generation")
            
        # Save final simulation image
        last_simulation_final_image = save_simulation_images(simulation_images, attempt, visuals_dir)
        
        # Always create video from simulation images
        if simulation_images:
            print(f"Creating video for attempt {attempt}")
            last_simulation_video = create_simulation_video(simulation_images, attempt, visuals_dir)
        
        # Convert final image to numpy array for AI feedback
        if last_simulation_final_image:
            final_surface = pg.image.load(last_simulation_final_image)
            final_numpy = pygame_surface_to_numpy(final_surface)
            # Save numpy array for AI processing
            np.save(os.path.join(visuals_dir, f'attempt_{attempt:03d}_final.npy'), final_numpy)
        
        # Check if the task is solved
        solved = success
        invalid_action = False
        
        # Save attempt results
        attempt_result = {
            'attempt': attempt,
            'action': result,
            'success': success,
            'time_to_success': time_to_success,
            'path_dict': path_dict,
            'num_simulation_frames': len(simulation_images) if simulation_images else 0,
            'simulation_final_image': last_simulation_final_image if 'last_simulation_final_image' in locals() else None,
            'simulation_video': last_simulation_video if 'last_simulation_video' in locals() else None,
            'used_video_feedback': use_video_feedback
        }
        
        with open(os.path.join(responses_dir, f'attempt_{attempt:03d}.json'), 'w') as f:
            json.dump(attempt_result, f, indent=2, default=str)
        
        # Demonstrate the placement
        demonstrateTPPlacement(tp, toolname, position)
        
        if solved:
            print(f"\n🎉 Task solved in {attempt} attempts!")
            break
        else:
            print(f"Task not solved yet. Continuing...")

    # Final summary
    print("\n" + "=" * 50)
    if solved:
        print(f"✅ Task completed successfully in {attempt} attempts!")
    else:
        print(f"❌ Task failed after {max_attempts} attempts")
    
    # Save final summary
    summary = {
        'task_solved': solved,
        'total_attempts': attempt,
        'max_attempts': max_attempts,
        'model_used': args.model,
        'final_success': solved
    }
    
    with open(os.path.join(responses_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to: {run_dir}")
    print(f"Summary: {summary}")
    
    # Clean up
    pg.quit()


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
        description="Run tool use task with AI models to predict and refine actions."
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='./output',
        help='Root directory for saving results'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemini',
        choices=['openai', 'gemini'],
        help='Model to use for inference'
    )
    parser.add_argument(
        '--max_attempts',
        type=int,
        default=20,
        help='Maximum number of attempts to solve the task'
    )
    parser.add_argument(
        '--use_video_feedback',
        action='store_true',
        help='Use video feedback instead of final image for failed attempts'
    )
    args = parser.parse_args()
    main(args)
