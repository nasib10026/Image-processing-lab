import cv2
import numpy as np
import math
from tabulate import tabulate


def calculate_euclidean_distance(descriptor1, descriptor2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(descriptor1, descriptor2)))


def find_max_diameter(border_points):
    x_min = y_min = float('inf')
    x_max = y_max = float('-inf')

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
#todo
    area = np.count_nonzero(image_path)
    perimeter = np.count_nonzero(border)
    border_points = np.argwhere(border == 255)
    max_diameter = find_max_diameter(border_points)

    form_factor = area / (perimeter ** 2)
    roundness = (4 * area) / (np.pi * max_diameter ** 2)
    compactness = perimeter ** 2 / area

    descriptor = [form_factor, roundness, compactness]

    print(descriptor)

    return descriptor, border


train_paths = ['c1.jpg', 't1.jpg', 'p1.png']
test_imgs = ['c2.jpg', 't2.jpg', 'p2.png', 'st.jpg']

train_descriptor = []
test_descriptors = []

for i, train_path in enumerate(train_paths):
    train_descriptor, _ = calculate_descriptors(train_path)
    train_descriptor.append(train_descriptor)

for i, test_img in enumerate(test_imgs):
    test_descriptor,_ = calculate_descriptors(test_img)
    test_descriptors.append(test_descriptor)

    # cv2.imshow('Input image ' + str(i + 1), cv2.imread(image_path))
    # cv2.imshow('Border ' + str(i + 1), border)

# euclidean_distance = np.array(np.len(train_paths),len(test_imgs))
i = 0
j = 0
for d1 in train_descriptor:
    for d2 in test_descriptors:

        euclidean_distance = calculate_euclidean_distance(d1, d2)
        # i += 1
        # j += 1

distances_matrix = [[euclidean_distance]]
#todo
d1 = np.array(train_paths)

row_headers = ['Image1 vs. Image2']
col_headers = ['Euclidean Distance']

distances_matrix = np.array(distances_matrix)


print(tabulate(distances_matrix, headers=col_headers, showindex=row_headers, tablefmt='grid'))

cv2.waitKey(0)
cv2.destroyAllWindows()