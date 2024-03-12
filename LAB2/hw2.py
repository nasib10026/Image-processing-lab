import cv2 as cv
import numpy as np

img = cv.imread("/Users/rakibulnasib/Desktop/image/LAB2/Lena.jpg")

print("Enter the threshold value: ")
threshold = int(input())

cv.imshow('input', img)
cv.waitKey(0)

def segmentation(topRow, bottomRow, topCol, bottomCol):
    height = bottomRow - topRow + 1
    width = bottomCol - topCol + 1
    
    img_part = img[topRow:bottomRow + 1, topCol:bottomCol + 1]

    # Split kora image_part to R,G,B 
    img_red, img_green, img_blue = cv.split(img_part)

    std_r = np.std(img_red)
    print(std_r)
    std_g = np.std(img_green)
    print(std_g)
    std_b = np.std(img_blue)
    print(std_b)
    
    if np.all(np.array([std_r, std_g, std_b]) < threshold) or (height <= 2 and width <= 2):
     mean1 = np.mean(img_red)
     mean2 = np.mean(img_green)
     mean3 = np.mean(img_blue)
     img[topRow:bottomRow + 1, topCol:bottomCol + 1, :] = np.full_like(img_part, [mean1, mean2, mean3])
     return
 # homogeneous tai ar segmentation lagbe na

    mid_row = (topRow + bottomRow) // 2
    mid_col = (topCol + bottomCol) // 2
    for i in range(2):
     for j in range(2):
        row_start = topRow + i * (mid_row - topRow)
        row_end = mid_row + (i == 1) * (bottomRow - mid_row)
        col_start = topCol + j * (mid_col - topCol)
        col_end = mid_col + (j == 1) * (bottomCol - mid_col)
        segmentation(row_start, row_end, col_start, col_end)

row_num = img.shape[0]
col_num = img.shape[1]
segmentation(0, row_num, 0, col_num)

img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
cv.imshow('output', np.rint(img).astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()
