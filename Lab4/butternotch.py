import numpy as np
import math

def butterworth_notch_reject_filter(M, N, points, D0, n):
    H = np.ones((M, N), np.float32)
    for u, v in points:
        for i in range(M):
            for j in range(N):
                # Calculate distance from the center of the image
                D_uv = math.sqrt((i - u)**2 + (j - v)**2)
                # Apply Butterworth notch filter equation
                H[i, j] = 1 / (1 + (D0 / D_uv)**(2*n))
    return H

# Example usage:
M = 100  # Number of rows
N = 100  # Number of columns
points = [(20, 30), (40, 60)]  # List of notch points
D0 = 20  # Distance parameter for notch filter
n = 2  # Order of Butterworth filter

filter_H = butterworth_notch_reject_filter(M, N, points, D0, n)
print(filter_H)
