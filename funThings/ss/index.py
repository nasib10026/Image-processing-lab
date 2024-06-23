import cv2
import numpy as np
import math

def calculate_descriptors(image, i):
    _, binary_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(binary_img, kernel, iterations=1)
    bordered = cv2.subtract(binary_img, eroded)
    
    area_ = np.count_nonzero(binary_img)
    peri_ = np.count_nonzero(bordered)
    
    #calculate kortesi 4 ta corner of the border
    idx = calculate_idx(bordered)
    dist1 = math.sqrt((idx[0][0] - idx[3][0])**2 + (idx[0][1] - idx[3][1])**2)
    dist2 = math.sqrt((idx[1][0] - idx[2][0])**2 + (idx[1][1] - idx[2][1])**2)
    max_d = max(dist1, dist2)
    
    comp = (peri_ ** 2) / area_
    ff = (4 * math.pi * area_) / (peri_ ** 2)
    rnd = (4 * area_) / (math.pi * max_d ** 2)
    
    desc = [comp, ff, rnd]
    cv2.imshow('Border'+str(i), bordered)
    cv2.imshow('Input image'+str(i), image)

    print(desc)
    return desc

def euc_dis(desc1, desc2):
    a = desc1[0] - desc2[0]
    b = desc1[1] - desc2[1]
    c = desc1[2] - desc2[2]
    
    res = math.sqrt(a * a + b * b + c * c)
    return res

def calculate_idx(img):
    h, w = img.shape
    
    idx = [[0, 0], [0, 0], [0, 0], [0, 0]]
    
    for x in range(h):
        for y in range(w):
            if img[x][y] != 0:
                if x + y < idx[0][0] + idx[0][1]:
                    idx[0] = [x, y]
                if x - y < idx[1][0] - idx[1][1]:
                    idx[1] = [x, y]
                if x + y > idx[2][0] + idx[2][1]:
                    idx[2] = [x, y]
                if x - y > idx[3][0] - idx[3][1]:
                    idx[3] = [x, y]
                    
    return idx

image_name = ['c1.jpg','t1.jpg','p1.png','c2.jpg','t2.jpg','p2.png']

descriptors = []

for i in range(len(image_name)):
    img = cv2.imread(image_name[i], 0)
    descriptors.append(calculate_descriptors(img, i))

# Matching descriptors
# Here you would compare the descriptors of the test and train images to find a close match
# For example, calculate the Euclidean distance between descriptors and find the minimum distance for matching.

cv2.waitKey(0)
cv2.destroyAllWindows()
