import numpy as np
from numpy.linalg import svd

def build_convolution_matrix(kernel, patch_size):
    k = kernel.shape[0]
    p = patch_size
    A = np.zeros((p*p, p*p))

    pad = k // 2
    kernel_padded = np.zeros((p, p))
    kernel_padded[:k, :k] = kernel

    for i in range(p*p):
        basis = np.zeros((p, p))
        basis.flat[i] = 1.0
        conv = np.fft.ifft2(
            np.fft.fft2(basis) * np.fft.fft2(kernel_padded)
        ).real
        A[:, i] = conv.flatten()

    return A

def pseudoinverse_reconstruction(A, y):
    U, S, Vt = svd(A, full_matrices=False)
    S_inv = np.diag(1 / S)
    x_hat = Vt.T @ S_inv @ U.T @ y
    return x_hat, S

from data_input import load_image
from forward_model import forward_blur
from baseline_pseudoinverse import build_convolution_matrix, pseudoinverse_reconstruction

img = load_image()
patch = img[100:116, 100:116]   # 16×16 patch

y_img, kernel = forward_blur(patch)
A = build_convolution_matrix(kernel, patch.shape[0])

x_hat, singular_values = pseudoinverse_reconstruction(A, y_img.flatten())
