from data_input import load_image
from forward_model import forward_blur
from baseline_pseudoinverse import (
    build_convolution_matrix,
    pseudoinverse_reconstruction
)
from diagnostics import compute_diagnostics
import matplotlib.pyplot as plt
import numpy as np

# Load data
img = load_image()

# Select small patch
patch = img[100:116, 100:116]

# Forward model
y_img, kernel = forward_blur(patch)

# Build system
A = build_convolution_matrix(kernel, patch.shape[0])

# Baseline pseudoinverse
x_hat, S = pseudoinverse_reconstruction(A, y_img.flatten())

# Diagnostics
diag = compute_diagnostics(S)
print("Diagnostics:", diag)

# Visualize
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(patch, cmap='gray')
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(y_img, cmap='gray')
plt.title("Blurred + Noise")

plt.subplot(1,3,3)
plt.imshow(x_hat.reshape(patch.shape), cmap='gray')
plt.title("Pseudoinverse")

plt.show()
