import cv2
import numpy as np

# Read input image
img_input = cv2.imread('/Users/rakibulnasib/Desktop/image/Lab 4/two_noise.jpeg', 0)
img = img_input.copy()

# Image size
M  = img.shape[0]
N = img.shape[1]

# Notch reject filter parameters
centers = 1
D0 = 5
n = 2

def dk(u, uk, v, vk):
    return np.sqrt((u - M / 2 - uk)**2 + (v - N / 2 - vk)**2)

def d_k(u, uk, v, vk):
    return np.sqrt((u - M / 2 + uk)**2 + (v - N / 2 + vk)**2)

def butterworth_notch_reject_filter(u, v):
    Huv = 1
    for i in range(centers):
        Huv *= 1 / (1 + (D0 / dk(u, M//2, v, N//2))**(2 * n)) * 1 / (1 + (D0 / d_k(u, M//2, v, N//2))**(2 * n))
    return Huv

# Fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)

# Magnitude spectrum
magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(magnitude_spectrum_ac + 1)
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Phase spectrum
ang = np.angle(ft_shift)
ang_ = cv2.normalize(ang, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Apply Butterworth notch reject filter
for u in range(M):
    for v in range(N):
        ft_shift[u, v] *= butterworth_notch_reject_filter(u, v)

# Inverse Fourier transform
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(ft_shift)))
img_back_scaled = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display images
cv2.imshow("Input Image", img_input)
cv2.imshow("Magnitude Spectrum", magnitude_spectrum)
cv2.imshow("Phase Spectrum", ang_)
cv2.imshow("Filtered Image", img_back_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
