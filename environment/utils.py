import os, json
# import phyre
from PIL import Image
import numpy as np
import cv2


def save_json(data, path):
    """
    Save a dictionary to a JSON file.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {path}")

def save_image(image, path):
    """
    Save a NumPy array as an image file.
    """
    img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_pil.save(path)
    print(f"Image saved to {path}")

