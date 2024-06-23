import cv2
import numpy as np
from copy import deepcopy as dpc

Do = 3  # Radius of the notch
n = 2   # Order of the filter

def ideal_notch_reject_filter(M, N, points):
    H = np.ones((M, N), np.float32)
    for u, v in points:
        # Zero out the 3x3 neighborhood around each notch point
        for i in range(u - 1, u + 2):
            for j in range(v - 1, v + 2):
                if 0 <= i < M and 0 <= j < N:
                    H[i, j] = 0.0
    return H

def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)
    return ((img_inp - inp_min) / (inp_max - inp_min)) * 255

# Read input image
img_input = cv2.imread('two_noise.jpeg', 0)
img = dpc(img_input)

# Fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift))

# Define notch points
points = [(200, 200), (300, 300)]  # Example notch points

# Apply ideal notch reject filter
notch_filter = ideal_notch_reject_filter(img.shape[0], img.shape[1], points)

# Apply filter to magnitude spectrum
result = magnitude_spectrum * notch_filter

# Inverse Fourier transform
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(np.exp(result))))

# Normalize and display images
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)
cv2.imshow("Magnitude Spectrum", magnitude_spectrum_scaled.astype(np.uint8))
cv2.imshow("Filtered Image", img_back.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()