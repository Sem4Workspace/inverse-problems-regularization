import os
import numpy as np
from skimage import io, img_as_float

def load_random_patch(bsds_path, patch_size=16):
    files = [f for f in os.listdir(bsds_path) if f.endswith(('.jpg', '.png'))]
    img_file = np.random.choice(files)
    
    img = img_as_float(io.imread(os.path.join(bsds_path, img_file), as_gray=True))
    h, w = img.shape

    i = np.random.randint(0, h - patch_size)
    j = np.random.randint(0, w - patch_size)

    return img[i:i+patch_size, j:j+patch_size]
