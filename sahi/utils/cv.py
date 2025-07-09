# Patched version of sahi.utils.cv to avoid importing cv2
from PIL import Image
import numpy as np

def read_image(path: str):
    return Image.open(path)

def resize(image, height, width):
    return image.resize((width, height))

def crop(image, x_min, y_min, x_max, y_max):
    return image.crop((x_min, y_min, x_max, y_max))
