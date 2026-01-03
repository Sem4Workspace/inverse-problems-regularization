import numpy as np
from skimage import data, color, img_as_float

def load_image():
    img = data.camera()              # standard test image
    img = img_as_float(img)
    return img
