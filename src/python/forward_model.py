import numpy as np
from scipy.signal import convolve2d

def gaussian_kernel(size=9, sigma=2):
    ax = np.linspace(-(size//2), size//2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def forward_blur(image, noise_std=0.01):
    kernel = gaussian_kernel()
    blurred = convolve2d(image, kernel, mode='same', boundary='symm')
    noisy = blurred + noise_std * np.random.randn(*blurred.shape)
    return noisy, kernel
