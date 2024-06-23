import cv2
import numpy as np
import math

class NotchRejectFilter:
    def __init__(self, img_path, points, Do=3, n=2):
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.points = points
        self.Do = Do
        self.n = n

    def butterworth_notch_reject_filter(self, M, N):
        H = np.ones((M, N), np.float32)
        for u, v in self.points:
            for i in range(M):
                for j in range(N):
                    # Calculate distance from the center of the image
                    D_uv = math.sqrt((i - u)**2 + (j - v)**2)
                    # Apply Butterworth notch filter equation
                    H[i, j] = 1 / (1 + (self.Do / D_uv)**(2*self.n))
        return H

    def min_max_normalize(self, img_inp):
        inp_min = np.min(img_inp)
        inp_max = np.max(img_inp)
        return ((img_inp - inp_min) / (inp_max - inp_min)) * 255

    def apply_filter(self):
        # Fourier transform
        ft = np.fft.fft2(self.img)
        ft_shift = np.fft.fftshift(ft)
        magnitude_spectrum = 20 * np.log(np.abs(ft_shift))

        # Apply Butterworth notch reject filter
        notch_filter = self.butterworth_notch_reject_filter(self.img.shape[0], self.img.shape[1])

        # Apply filter to magnitude spectrum
        result = magnitude_spectrum * notch_filter

        # Inverse Fourier transform
        img_back = np.real(np.fft.ifft2(np.fft.ifftshift(np.exp(result))))

        # Normalize images
        magnitude_spectrum_scaled = self.min_max_normalize(magnitude_spectrum)
        img_back_normalized = self.min_max_normalize(img_back)

        return magnitude_spectrum_scaled.astype(np.uint8), img_back_normalized.astype(np.uint8)


# Define notch points
points = [(200, 200), (300, 300)]  # Example notch points

# Create NotchRejectFilter object
filter_obj = NotchRejectFilter('/Users/rakibulnasib/Desktop/image/Lab 4/two_noise.jpeg', points)

# Apply filter and get images
magnitude_spectrum_img, filtered_img = filter_obj.apply_filter()

# Display images
cv2.imshow("Magnitude Spectrum", magnitude_spectrum_img)
cv2.imshow("Filtered Image", filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
