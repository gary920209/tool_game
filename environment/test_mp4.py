#!/usr/bin/env python3
"""
Test script to validate MP4 video generation functionality
"""
import os
import sys
import json
from pyGameWorld.world import loadFromDict
from pyGameWorld.viewer import demonstrateWorld_and_save_video

def test_mp4_generation():
    """Test MP4 video generation with a basic world"""
    
    # Load a basic trial
    trial_path = os.path.join(os.path.dirname(__file__), 'Trials', 'Original', 'Basic.json')
    print(f"Loading trial from: {trial_path}")
    
    if not os.path.exists(trial_path):
        print(f"Error: Trial file not found at {trial_path}")
        return False
    
    with open(trial_path, 'r') as f:
        trial_data = json.load(f)
    
    # Create world from trial data
    world = loadFromDict(trial_data['world'])
    print(f"World loaded successfully. Dimensions: {world.dims}")
    
    # Test MP4 generation
    output_path = os.path.join(os.path.dirname(__file__), 'test_output.mp4')
    
    try:
        print("🎬 Testing MP4 video generation...")
        demonstrateWorld_and_save_video(world, video_filename=output_path, hz=30, max_frames=5)
        
        # Check if file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ MP4 video created successfully!")
            print(f"   File: {output_path}")
            print(f"   Size: {file_size} bytes")
            
            # Optional: Remove test file
            # os.remove(output_path)
            
            return True
        else:
            print("❌ MP4 file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error during MP4 generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing MP4 video generation...")
    success = test_mp4_generation()
    
    if success:
        print("\n🎉 MP4 test completed successfully!")
    else:
        print("\n💥 MP4 test failed!")
        sys.exit(1)
