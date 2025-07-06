#!/usr/bin/env python3
"""
Simple MP4 generation test without pyGameWorld dependencies
"""
import os
import numpy as np
import cv2
from PIL import Image

def create_simple_mp4_test():
    """Create a simple MP4 video with basic frames"""
    
    # Create some sample frames
    frames = []
    width, height = 640, 480
    
    # Create 5 frames with different colors
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255)   # Magenta
    ]
    
    for i, color in enumerate(colors):
        # Create a frame with the specified color
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        
        # Add some text to distinguish frames
        cv2.putText(frame, f'Frame {i+1}', (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame_bgr)
    
    # Create video
    output_path = 'test_simple.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    
    # Check if file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"✅ Simple MP4 created successfully!")
        print(f"   File: {output_path}")
        print(f"   Size: {file_size} bytes")
        print(f"   Frames: {len(frames)}")
        return True
    else:
        print("❌ MP4 file was not created")
        return False

if __name__ == "__main__":
    print("Testing simple MP4 generation...")
    try:
        success = create_simple_mp4_test()
        if success:
            print("\n🎉 Simple MP4 test completed successfully!")
        else:
            print("\n💥 Simple MP4 test failed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
