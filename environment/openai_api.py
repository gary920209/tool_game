import os
import json
import argparse
import re
from dotenv import load_dotenv 

import time
import cv2
import base64
from PIL import Image
from io import BytesIO

import numpy as np
import openai
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

class OpenAIClient:
    def __init__(
        self, api_key=OPENAI_API_KEY, upload_file=False, model="gpt-4o-mini", fps=30.0
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fps = fps
        self.upload_file = upload_file
        self.conversation_history = []
        
    def clean_history(self):
        """
        Reset the conversation history and output directories.
        """
        self.conversation_history = []
    
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
    
    # def upload_image(self, image_object):
    #     """
    #     Upload an image to OpenAI and get its file object.
    #     Note: OpenAI doesn't have a separate upload mechanism like Gemini,
    #     so this method returns the image data for direct use in messages.
    #     """
    #     if isinstance(image_object, str) and os.path.isfile(image_object):
    #         return image_object
    #     elif isinstance(image_object, np.ndarray):
    #         # Save NumPy array to temporary file
    #         img_uint8 = (image_object * 255).clip(0, 255).astype(np.uint8)
    #         img_pil = Image.fromarray(img_uint8)
    #         temp_path = "/tmp/temp_image.png"
    #         img_pil.save(temp_path, format="PNG")
    #         return temp_path
    #     else:
    #         raise ValueError("Unsupported image object type. Provide a file path or a NumPy array.")

    def upload_video(self, video_object, fps=None):
        """
        Upload a video file to OpenAI and return the file path.
        Note: OpenAI doesn't have a separate upload mechanism like Gemini,
        so this method returns the video file path for direct use in messages.
        """
        if fps is None:
            fps = self.fps

        if isinstance(video_object, str) and os.path.isfile(video_object):
            return video_object
        
        elif isinstance(video_object, np.ndarray):
            # Ensure the video is in the correct format (e.g., a list of frames)
            assert video_object.ndim == 4, "Video object must be a 4D NumPy array (frames, height, width, channels)."

            # Convert each frame to an image and save to a temporary video file
            temp_video_path = "/tmp/temp_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width, _ = video_object[0].shape
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            for frame in video_object:
                out.write(frame)

            out.release()
            return temp_video_path
        
        else:
            raise ValueError("Unsupported video object type. Provide a file path or a list of frames (NumPy arrays).")

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
            if not isinstance(video_object, list):
                raise ValueError("Video object must be a list of frames (NumPy arrays).")
            
            # Convert each frame to an image and save to a temporary video file
            temp_video_path = "/tmp/temp_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width, _ = video_object[0].shape
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            for frame in video_object:
                out.write(frame)

            out.release()
            with open(temp_video_path, 'rb') as video_file:
                video_bytes = video_file.read()
            os.remove(temp_video_path)

        else:
            raise ValueError("Unsupported video object type. Provide a file path or a list of frames (NumPy arrays).")
        return video_bytes

    def request(self, messages, schema=None, max_retries=3):
        """
        Make a request to OpenAI API with the given messages.
        """
        print("Requesting with user inputs...")
        
        # Prepare the request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1
        }
        
        # Add response format if schema is provided
        if schema is not None:
            request_params["response_format"] = {"type": "json_object"}
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**request_params)
                return response.choices[0].message.content
            except Exception as e:
                print(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def inference_text(self, prompt, replace_dict=None, schema=None, history=False, max_retries=5):
        if replace_dict is not None:
            for key, value in replace_dict.items():
                prompt = prompt.replace(key, value)
        
        # Create message for OpenAI format
        message = {"role": "user", "content": prompt}
        
        if history:
            messages = self.conversation_history + [message]
        else:
            messages = [message]
        
        for attempt in range(max_retries):
            try:
                response = self.request(messages, schema=schema)
                if schema is not None:
                    pred_result = self._safe_json_load(response)
                else:
                    pred_result = response
                
                # Add to conversation history
                self.conversation_history += [message, {"role": "assistant", "content": str(pred_result)}]
                return pred_result

            except Exception as e:
                print(f"Error processing (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                print("Retrying...")
                time.sleep(1)

    def inference_image(self, image_object, prompt, replace_dict=None, schema=None, history=False, max_retries=5):
        if replace_dict is not None:
            for key, value in replace_dict.items():
                prompt = prompt.replace(key, value)
        
        # Prepare image content
        if self.upload_file:
            print("OpenAI doesn't support image upload, please set upload_file=False")
            return None
        else:
            # Handle both single image and list of images
            if isinstance(image_object, list):
                # Multiple images
                image_base64_list = self.encode_images(image_object)
                image_contents = []
                for image_base64 in image_base64_list:
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    })
            else:
                # Single image
                image_base64 = self.encode_image(image_object)
                image_contents = [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }]
        
        # Create message for OpenAI format
        message = {
            "role": "user",
            "content": image_contents + [{"type": "text", "text": prompt}]
        }
        
        if history:
            messages = self.conversation_history + [message]
        else:
            messages = [message]
        
        for attempt in range(max_retries):
            try:
                response = self.request(messages, schema=schema)
                if schema is not None:
                    pred_result = self._safe_json_load(response)
                else:
                    pred_result = response
                
                # Add to conversation history
                self.conversation_history += [message, {"role": "assistant", "content": str(pred_result)}]
                return pred_result

            except Exception as e:
                print(f"Error processing (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                print("Retrying...")
                time.sleep(1)
    
    def inference_video(self, video_object, prompt, replace_dict=None, schema=None, history=False, fps=None, max_retries=5):
        if fps is None:
            fps = self.fps

        if replace_dict is not None:
            for key, value in replace_dict.items():
                prompt = prompt.replace(key, value)
        
        # Prepare video content
        if self.upload_file:
            print("OpenAI doesn't support video upload, please set upload_file=False")
            return None
        else:
            video_base64 = self.encode_video(video_object)
            video_content = {
                "type": "video_url",
                "video_url": {
                    "url": f"data:video/mp4;base64,{base64.b64encode(video_base64).decode('utf-8')}"
                }
            }

        # Create message for OpenAI format
        message = {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": prompt}
            ]
        }

        if history:
            messages = self.conversation_history + [message]
        else:
            messages = [message]
        
        for attempt in range(max_retries):
            try:
                response = self.request(messages, schema=schema)
                if schema is not None:
                    pred_result = self._safe_json_load(response)
                else:
                    pred_result = response
                
                # Add to conversation history
                self.conversation_history += [message, {"role": "assistant", "content": str(pred_result)}]
                return pred_result

            except Exception as e:
                print(f"Error processing (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                print("Retrying...")
                time.sleep(1) 