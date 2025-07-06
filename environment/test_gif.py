#!/usr/bin/env python3
"""
Test script to validate GIF generation and playback
"""
import os
import pygame as pg
from PIL import Image
from pyGameWorld.world import loadFromDict
from pyGameWorld.viewer import demonstrateWorld_and_save_gif
import json

def test_gif_generation():
    """Test GIF generation with a simple world"""
    
    # Load a basic world configuration
    JSON_FILE = './Trials/Original/Basic.json'
    
    if not os.path.exists(JSON_FILE):
        print(f"Error: {JSON_FILE} not found")
        return
    
    with open(JSON_FILE, 'r') as f:
        pgw_dict = json.load(f)
    
    # Load the world
    world = loadFromDict(pgw_dict['world'])
    
    # Create test directory
    test_dir = './test_gif'
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate test GIF
    gif_path = os.path.join(test_dir, 'test_simulation.gif')
    
    print("Generating test GIF...")
    try:
        demonstrateWorld_and_save_gif(
            world, 
            gif_filename=gif_path, 
            hz=10,  # Lower frame rate for testing
            max_frames=50  # Shorter for testing
        )
        
        if os.path.exists(gif_path):
            file_size = os.path.getsize(gif_path)
            print(f"✅ GIF created successfully!")
            print(f"   Path: {gif_path}")
            print(f"   Size: {file_size} bytes")
            
            # Try to open with PIL to validate
            try:
                gif = Image.open(gif_path)
                print(f"   Format: {gif.format}")
                print(f"   Size: {gif.size}")
                print(f"   Frames: {gif.n_frames if hasattr(gif, 'n_frames') else 'Unknown'}")
                print("✅ GIF validation successful!")
            except Exception as e:
                print(f"❌ GIF validation failed: {e}")
        else:
            print("❌ GIF file was not created")
            
    except Exception as e:
        print(f"❌ Error generating GIF: {e}")
    
    # Clean up
    if os.path.exists(gif_path):
        # Don't delete for manual inspection
        print(f"GIF saved for manual inspection: {gif_path}")

if __name__ == "__main__":
    test_gif_generation()
