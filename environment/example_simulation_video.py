#!/usr/bin/env python3
"""
Example usage of GeminiClient with simulation video processing
"""
import os
import json
import pygame as pg
from pyGameWorld.world import loadFromDict
from pyGameWorld import ToolPicker
from pyGameWorld.viewer import makeImageArray
from gemini_api import GeminiClient
from schemas import ToolUseVideoAction

def main():
    # Load the world from a JSON file
    JSON_FILE = './Trials/Original/Basic.json'
    
    with open(JSON_FILE, 'r') as f:
        pgw_dict = json.load(f)

    # Turn this into a toolpicker game
    tp = ToolPicker(pgw_dict)

    # Initialize pygame
    pg.init()
    
    # Create Gemini client
    client = GeminiClient(upload_file=True, fps=10.0)
    
    # Simulate an action
    toolname = "obj1"
    position = [300, 300]
    
    print("Simulating action...")
    path_dict, success, time_to_success, world_dict = tp.observeFullPlacementPath(
        toolname=toolname,
        position=position,
        maxtime=5.,
        returnDict=True
    )
    
    if path_dict is not None:
        # Generate image sequence from the simulation path
        simulation_images = makeImageArray(world_dict, path_dict, sample_ratio=1)
        print(f"Generated {len(simulation_images)} simulation frames")
        
        # Create video from simulation using GeminiClient
        visuals_dir = "./temp_visuals"
        os.makedirs(visuals_dir, exist_ok=True)
        
        attempt = 1
        video_path = client.create_simulation_video(
            simulation_images, 
            attempt, 
            visuals_dir, 
            fps=10
        )
        
        if video_path:
            print(f"Video created: {video_path}")
            
            # Example of using the video for inference
            prompt = "Analyze this simulation video and determine if the action was successful."
            
            try:
                # Use the new inference_simulation_video method
                result = client.inference_simulation_video(
                    pygame_images=simulation_images,
                    prompt=prompt,
                    schema=ToolUseVideoAction,
                    history=False,
                    fps=10
                )
                
                print("AI Analysis Result:")
                print(json.dumps(result, indent=2))
                
            except Exception as e:
                print(f"Error during inference: {e}")
        
        # Clean up
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(visuals_dir):
            import shutil
            shutil.rmtree(visuals_dir)
    
    pg.quit()

if __name__ == "__main__":
    main()
