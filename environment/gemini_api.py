import os
import json
import argparse
import re
import tempfile
from dotenv import load_dotenv 

import time
import cv2
import base64
from PIL import Image
from io import BytesIO

import numpy as np
import pygame as pg
from google import genai
from google.genai import types

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

class GeminiClient:
    def __init__(
        self, api_key=GOOGLE_API_KEY, upload_file=False, model="models/gemini-2.5-flash-preview-05-20", fps=30.0
    ):

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.fps = fps

        self.upload_file = upload_file

        self.conversation_history = []
        
    def clean_history(self):
        """
        Reset the conversation history and output directories.
        """
        self.conversation_history = []
    
    def pygame_surface_to_numpy(self, surface):
        """Convert pygame surface to numpy array"""
        # Get the raw pixel data
        pixel_data = pg.surfarray.array3d(surface)
        # Convert from (width, height, 3) to (height, width, 3)
        pixel_data = np.transpose(pixel_data, (1, 0, 2))
        # Convert to float in [0, 1] range
        pixel_data = pixel_data.astype(np.float32) / 255.0
        return pixel_data
    
    def save_simulation_images(self, images, attempt, visuals_dir):
        """Save only the final simulation image for feedback"""
        # Return the final image for feedback
        final_img = images[-1] if images else None
        if final_img:
            final_img_path = os.path.join(visuals_dir, f'attempt_{attempt:03d}_final.png')
            pg.image.save(final_img, final_img_path)
            return final_img_path
        
        return None
    
    def create_simulation_video(self, images, attempt, visuals_dir, fps=10):
        """Create video from simulation images and return video path"""
        if not images:
            return None
        
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
    
    def pygame_images_to_video_array(self, images, fps=10):
        """Convert pygame surface images to numpy video array for AI processing"""
        if not images:
            return None
        
        # Convert pygame surfaces to numpy arrays
        frames = []
        for img in images:
            # Convert pygame surface to numpy array
            pixel_data = pg.surfarray.array3d(img)
            # Convert from (width, height, 3) to (height, width, 3)
            pixel_data = np.transpose(pixel_data, (1, 0, 2))
            # Ensure it's in uint8 format
            if pixel_data.dtype != np.uint8:
                pixel_data = (pixel_data * 255).clip(0, 255).astype(np.uint8)
            frames.append(pixel_data)
        
        # Convert to 4D numpy array
        video_array = np.array(frames)
        return video_array
    
    def _safe_json_load(self, json_str):
        """
        Extract and parse a valid JSON object from a raw string that may contain extra text
        (e.g., markdown formatting or log prefixes).
        Raises ValueError if parsing fails.
        """
        match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}")
        else:
            raise ValueError("No valid JSON object found in the input string.")
        
    def encode_image(self, image_object):
        """
        Encode an image and get its base64 representation.
        """
        if isinstance(image_object, str) and os.path.isfile(image_object):
            image_path = image_object
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        elif isinstance(image_object, np.ndarray):
            # Ensure the values are in [0, 255] range and convert to uint8
            img_uint8 = (image_object * 255).clip(0, 255).astype(np.uint8)

            # Convert NumPy array to PIL image
            img_pil = Image.fromarray(img_uint8)

            # Save to a BytesIO buffer
            buffer = BytesIO()
            img_pil.save(buffer, format="PNG")
            buffer.seek(0)

            # Encode the image in base64
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        
        else:
            raise ValueError("Unsupported image object type. Provide a file path or a NumPy array.")
        return image_base64

    def encode_images(self, image_objects):
        """
        Encode a list of images and get their base64 representations.
        """
        image_base64_list = []
        for image_object in image_objects:
            image_base64 = self.encode_image(image_object)
            image_base64_list.append(image_base64)
        return image_base64_list
    
    def upload_image(self, image_object):
        """
        Upload an image to Gemini and get its file object.
        """
        if isinstance(image_object, str) and os.path.isfile(image_object):
            image_path = image_object
            my_file = self.client.files.upload(file=image_path)

        elif isinstance(image_object, np.ndarray):
            # Ensure the values are in [0, 255] range and convert to uint8
            img_uint8 = (image_object * 255).clip(0, 255).astype(np.uint8)

            # Convert NumPy array to PIL image
            img_pil = Image.fromarray(img_uint8)

            # Save to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            img_pil.save(tmp_path, format="PNG")

            # Upload the image to Gemini
            my_file = self.client.files.upload(file=tmp_path)
            
            # Clean up temporary file
            os.remove(tmp_path)

        else:
            raise ValueError("Unsupported image object type. Provide a file path or a NumPy array.")
        time.sleep(1)  # Wait for the upload to complete
        return my_file

    def upload_video(self, video_object, fps=None):
        """
        Upload a video file to Gemini and return the file object.
        """
        if fps is None:
            fps = self.fps

        if isinstance(video_object, str) and os.path.isfile(video_object):
            video_path = video_object
            video_file = self.client.files.upload(file=video_path)
        
        elif isinstance(video_object, np.ndarray):
            # Ensure the video is in the correct format (e.g., a 4D array or list of frames)
            if video_object.ndim == 4:
                # 4D array: (frames, height, width, channels)
                frames = video_object
            else:
                raise ValueError("Video object must be a 4D NumPy array (frames, height, width, channels).")

            # Convert each frame to an image and save to a temporary video file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                temp_video_path = tmp_file.name
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            for frame in frames:
                # Ensure frame is in uint8 format
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                out.write(frame)

            out.release()
            video_file = self.client.files.upload(file=temp_video_path)
            os.remove(temp_video_path)
        
        else:
            raise ValueError("Unsupported video object type. Provide a file path or a list of frames (NumPy arrays).")
        time.sleep(1)  # Wait for the upload to complete
        return video_file

    def encode_video(self, video_object, fps=None):
        """
        Encode a video and get its base64 representation.
        """
        if fps is None:
            fps = self.fps

        if isinstance(video_object, str) and os.path.isfile(video_object):
            video_path = video_object
            video_bytes = open(video_path, 'rb').read()

        elif isinstance(video_object, np.ndarray):
            # Ensure the video is in the correct format (e.g., a list of frames)
            if video_object.ndim == 4:
                # 4D array: (frames, height, width, channels)
                frames = video_object
            else:
                raise ValueError("Video object must be a 4D NumPy array (frames, height, width, channels).")
            
            # Convert each frame to an image and save to a temporary video file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                temp_video_path = tmp_file.name
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            for frame in frames:
                # Ensure frame is in uint8 format
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                out.write(frame)

            out.release()
            with open(temp_video_path, 'rb') as video_file:
                video_bytes = video_file.read()
            os.remove(temp_video_path)

        else:
            raise ValueError("Unsupported video object type. Provide a file path or a list of frames (NumPy arrays).")
        return video_bytes

        
    def request(self, user_inputs, schema=None):
        if schema is not None:
            config={
                'response_mime_type': 'application/json',
                'response_schema': schema,
            }
        else:
            config=None
        print("Requesting with user inputs...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_inputs,
                    config=config,
                )
                print(f"Request successful on attempt {attempt + 1}")
                return response.text
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"Request attempt {attempt + 1} failed ({error_type}): {error_msg}")
                
                # Check for specific error types
                if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    print("Rate limit or quota exceeded. Waiting longer...")
                    wait_time = (attempt + 1) * 10  # Longer wait for rate limits
                elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                    print("Network error detected. Waiting before retry...")
                    wait_time = (attempt + 1) * 5
                else:
                    wait_time = (attempt + 1) * 2
                
                if attempt < max_retries - 1:
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Max retries ({max_retries}) reached. Request failed.")
                    raise e
    
    def inference_text(self, prompt, replace_dict=None, schema=None, history=False):
        if replace_dict is not None:
            for key, value in replace_dict.items():
                prompt = prompt.replace(key, value)
        request_prompt = types.Part(text=prompt)

        
        if history:
            user_inputs = self.conversation_history + [request_prompt]
        else:
            user_inputs = [request_prompt]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.request(user_inputs, schema=schema)
                if schema is not None:
                    pred_result = self._safe_json_load(response)
                else:
                    pred_result = response
                
                self.conversation_history += [request_prompt, types.Part(text=str(pred_result))]
                return pred_result

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"Inference attempt {attempt + 1} failed ({error_type}): {error_msg}")
                
                # Handle specific error types
                if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    wait_time = (attempt + 1) * 10  # Longer wait for rate limits
                    print("Rate limit detected. Waiting longer...")
                elif attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Max inference retries ({max_retries}) reached. Returning None.")
                    return None

    def inference_image(self, image_object, prompt, replace_dict=None, schema=None, history=False):
        if replace_dict is not None:
            for key, value in replace_dict.items():
                prompt = prompt.replace(key, value)
        request_prompt = types.Part(text=prompt)
        
        if self.upload_file:
            image_file = self.upload_image(image_object)
        else:
            image_bytes = self.encode_image(image_object)
            image_file = types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            )
        
        if history:
            user_inputs = self.conversation_history + [image_file, request_prompt]
        else:
            user_inputs = [image_file, request_prompt]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.request(user_inputs, schema=schema)
                if schema is not None:
                    pred_result = self._safe_json_load(response)
                else:
                    pred_result = response
                
                self.conversation_history += [image_file, request_prompt, types.Part(text=str(pred_result))]
                return pred_result

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"Image inference attempt {attempt + 1} failed ({error_type}): {error_msg}")
                
                if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    wait_time = (attempt + 1) * 10
                    print("Rate limit detected. Waiting longer...")
                elif attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Max image inference retries ({max_retries}) reached. Returning None.")
                    return None
    
    def inference_video(self, video_object, prompt, replace_dict=None, schema=None, history=False, fps=None):
        if fps is None:
            fps = self.fps

        if replace_dict is not None:
            for key, value in replace_dict.items():
                prompt = prompt.replace(key, value)
        request_prompt = types.Part(text=prompt)
        
        if self.upload_file:
            video_file = self.upload_video(video_object)
        else:
            video_bytes = self.encode_video(video_object)
            video_file = types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'),
                video_metadata=types.VideoMetadata(fps=self.fps)
            )

        if history:
            user_inputs = self.conversation_history + [video_file, request_prompt]
        else:
            user_inputs = [video_file, request_prompt]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.request(user_inputs, schema=schema)
                if schema is not None:
                    pred_result = self._safe_json_load(response)
                else:
                    pred_result = response
                
                self.conversation_history += [video_file, request_prompt, types.Part(text=str(pred_result))]
                return pred_result

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"Video inference attempt {attempt + 1} failed ({error_type}): {error_msg}")
                
                if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    wait_time = (attempt + 1) * 10
                    print("Rate limit detected. Waiting longer...")
                elif attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Max video inference retries ({max_retries}) reached. Returning None.")
                    return None
    
    def inference_simulation_video(self, pygame_images, prompt, replace_dict=None, schema=None, history=False, fps=10):
        """
        Perform inference on a simulation video created from pygame images.
        
        Args:
            pygame_images: List of pygame surfaces representing simulation frames
            prompt: Text prompt for the AI
            replace_dict: Dictionary for prompt replacement
            schema: Response schema
            history: Whether to include conversation history
            fps: Frames per second for video
        
        Returns:
            AI response
        """
        if not pygame_images:
            raise ValueError("No pygame images provided")
        
        # Convert pygame images to video array
        video_array = self.pygame_images_to_video_array(pygame_images, fps)
        if video_array is None:
            raise ValueError("Failed to convert pygame images to video array")
        
        print(f"Created video array with shape: {video_array.shape}")
        
        # Use the existing inference_video method
        return self.inference_video(
            video_array, 
            prompt, 
            replace_dict=replace_dict, 
            schema=schema, 
            history=history, 
            fps=fps
        )
