import cv2
import numpy as np
from copy import deepcopy as dpc
import math

D0 = 3  # Radius of the notch
n = 2   # Order of the filter

def dk(u,uk,v,vk):
    return ((u- M/2 - uk)**2 + (v- N/2 - vk)**2)**0.5

def d_k(u,uk,v,vk):
    return ((u- M/2 - uk)**2 + (v- N/2 - vk)**2)**0.5

def butterworth_notch_reject_filter(u,v):
    Huv = 1
    for i in range(centers):
        Huv *= 1/(1+ (D0/dk(u,uk,v,vk))**(2*n)) * 1/(1+ (D0/d_k(u,uk,v,vk))**(2*n))
    return Huv

def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)
    return ((img_inp - inp_min) / (inp_max - inp_min)) * 255

# Read input image
img_input = cv2.imread('/Users/rakibulnasib/Desktop/image/Lab 4/two_noise.jpeg', 0)
img = dpc(img_input)

# Fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift))

# Define notch points
points = [(200, 200), (300, 300)]  # Example notch points

# Apply Butterworth notch reject filter
notch_filter = butterworth_notch_reject_filter(img.shape[0], img.shape[1], points, Do, n)

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
