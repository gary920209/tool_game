#!/usr/bin/env python3
"""
Direct test for GIF generation
"""
import os
import json
import pygame as pg
from pyGameWorld.world import loadFromDict
from pyGameWorld.viewer import demonstrateWorld_and_save_gif

def test_gif_generation():
    """Test GIF generation directly"""
    
    # Load the world
    JSON_FILE = './Trials/Original/Basic.json'
    
    if not os.path.exists(JSON_FILE):
        print(f"Error: {JSON_FILE} not found")
        return False
    
    print("Loading world configuration...")
    with open(JSON_FILE, 'r') as f:
        pgw_dict = json.load(f)
    
    # Create world
    print("Creating world...")
    world = loadFromDict(pgw_dict['world'])
    
    # Create test directory
    test_dir = './test_gif_output'
    os.makedirs(test_dir, exist_ok=True)
    
    # Test GIF generation
    gif_path = os.path.join(test_dir, 'direct_test.gif')
    
    print(f"Generating GIF: {gif_path}")
    print("This may take a few seconds...")
    
    try:
        demonstrateWorld_and_save_gif(
            world=world,
            gif_filename=gif_path,
            hz=10,  # Lower frame rate for faster generation
            max_frames=30  # Fewer frames for testing
        )
        
        if os.path.exists(gif_path):
            file_size = os.path.getsize(gif_path)
            print(f"✅ SUCCESS: GIF created!")
            print(f"   Path: {gif_path}")
            print(f"   Size: {file_size} bytes")
            return True
        else:
            print("❌ FAILED: GIF file not found after generation")
            return False
            
    except Exception as e:
        print(f"❌ ERROR during GIF generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_world_step():
    """Test if world.step() works correctly"""
    JSON_FILE = './Trials/Original/Basic.json'
    
    try:
        with open(JSON_FILE, 'r') as f:
            pgw_dict = json.load(f)
        
        world = loadFromDict(pgw_dict['world'])
        print("✅ World loaded successfully")
        
        # Test a few steps
        for i in range(5):
            world.step(0.033)  # ~30 FPS
            print(f"Step {i+1}: OK")
        
        print("✅ World stepping works correctly")
        return True
        
    except Exception as e:
        print(f"❌ World step test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing GIF generation directly...")
    
    # First test world stepping
    print("\n--- Testing World Step Function ---")
    step_ok = test_world_step()
    
    if step_ok:
        print("\n--- Testing GIF Generation ---")
        gif_ok = test_gif_generation()
        
        if gif_ok:
            print("\n🎉 All tests passed! GIF generation is working.")
        else:
            print("\n❌ GIF generation test failed.")
    else:
        print("\n❌ World step test failed - cannot proceed with GIF test.")
