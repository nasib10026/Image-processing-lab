import cv2
import numpy as np
import math
from tabulate import tabulate

def calculate_cosine_similarity(descriptor1, descriptor2):
    dot_product = 0
    magnitude1 = 0
    magnitude2 = 0

    for i in range(len(descriptor1)):
        dot_product += descriptor1[i] * descriptor2[i]
        magnitude1 += descriptor1[i] ** 2
        magnitude2 += descriptor2[i] ** 2

    magnitude1 = math.sqrt(magnitude1)
    magnitude2 = math.sqrt(magnitude2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0 

    return dot_product / (magnitude1 * magnitude2)

def find_max_diameter(border_points):
    x_min = float('inf')
    y_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')

    for point in border_points:
        x, y = point
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)

    max_diameter = max(x_max - x_min, y_max - y_min)

    return max_diameter

def calculate_descriptors(image_path):
    # Read the image
    img = cv2.imread(image_path, 0)

    # Perform erosion operation
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)
    border = img - eroded

    # Threshold border to binary
    _, border = cv2.threshold(border, 1, 255, cv2.THRESH_BINARY)

    # Calculate area and perimeter
    area = np.count_nonzero(img)
    perimeter = np.count_nonzero(border)
    border_points = np.argwhere(border == 255)
    max_diameter = find_max_diameter(border_points)

    form_factor = (4 * np.pi * area) / (perimeter ** 2)
    roundness = (4 * area) / (np.pi * max_diameter ** 2)
    compactness = perimeter ** 2 / area

    descriptor = [form_factor, roundness, compactness]

    return descriptor

train_paths = ['Lab5/c1.jpg', 'Lab5/t1.jpg', 'Lab5/p1.png']
test_imgs = ['Lab5/c2.jpg', 'Lab5/t2.jpg', 'Lab5/p2.png', 'Lab5/st.jpg']

train_descriptors = []
test_descriptors = []

for train_path in train_paths:
    train_descriptor = calculate_descriptors(train_path)
    train_descriptors.append(train_descriptor)

for test_img in test_imgs:
    test_descriptor = calculate_descriptors(test_img)
    test_descriptors.append(test_descriptor)

distances_matrix = []

for d1 in train_descriptors:
    row = []
    for d2 in test_descriptors:
        cosine_similarity = calculate_cosine_similarity(d1, d2)
        row.append(cosine_similarity)
    distances_matrix.append(row)

row_headers = [f'Train Image {i+1}' for i in range(len(train_paths))]
col_headers = [f'Test Image {i+1}' for i in range(len(test_imgs))]

print(tabulate(distances_matrix, headers=col_headers, showindex=row_headers, tablefmt='grid'))
