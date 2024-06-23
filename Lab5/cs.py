import cv2
import numpy as np
import math
from tabulate import tabulate

def calculate_cosine_similarity(descriptor1, descriptor2):
    dot_product = sum(x * y for x, y in zip(descriptor1, descriptor2))
    magnitude1 = math.sqrt(sum(x ** 2 for x in descriptor1))
    magnitude2 = math.sqrt(sum(y ** 2 for y in descriptor2))
    
    return dot_product / (magnitude1 * magnitude2)

def find_max_diameter(border_points):
    x_min = float('inf')
    y_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')

    for point in border_points:
        x, y = point
        #farthest point paite
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
    
    area = np.count_nonzero(img)#shada area calculate
    perimeter = np.count_nonzero(border)
    border_points = np.argwhere(border == 255)#border white
    max_diameter = find_max_diameter(border_points)

    form_factor = (4 *np.pi* area) / (perimeter ** 2)
    roundness = (4* area) / (np.pi * max_diameter ** 2)
    compactness = (perimeter ** 2) / area

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

for train_descriptor in train_descriptors:
    distances_row = []
    for test_descriptor in test_descriptors:
        cosine_similarity = calculate_cosine_similarity(train_descriptor, test_descriptor)
        distances_row.append(cosine_similarity)
    distances_matrix.append(distances_row)

row_headers = ['c1.jpg', 't1.jpg', 'p1.png']
col_headers = ['c2.jpg', 't2.jpg', 'p2.png', 'st.jpg']

print(tabulate(distances_matrix, headers=col_headers, showindex=row_headers, tablefmt='grid'))
