import cv2 as cv
import numpy as np

img = cv.imread("/Users/rakibulnasib/Desktop/image/LAB2/Lena.jpg")

threshold = 25

cv.imshow('input', img)
cv.waitKey(0)

def segmentation(topRow, bottomRow, topCol, bottomCol):
    height = bottomRow - topRow + 1
    width = bottomCol - topCol + 1
    
    img_slice = img[topRow:bottomRow + 1, topCol:bottomCol + 1]

    # Splitting the image slice into its color channels using cv.split
    img_red, img_green, img_blue = cv.split(img_slice)

    std1 = np.std(img_red)
    print(std1)
    std2 = np.std(img_green)
    print(std2)
    std3 = np.std(img_blue)
    print(std3)
    
    if np.all(np.array([std1, std2, std3]) < threshold) or (height <= 2 and width <= 2):
     mean1 = np.mean(img_red)
     mean2 = np.mean(img_green)
     mean3 = np.mean(img_blue)
     img[topRow:bottomRow + 1, topCol:bottomCol + 1, :] = np.full_like(img_slice, [mean1, mean2, mean3])
     return
 # homogeneous tai ar segmentation lagbe na

    mid_row = (topRow + bottomRow) // 2
    mid_col = (topCol + bottomCol) // 2
    segmentation(topRow, mid_row, topCol, mid_col)
    segmentation(mid_row + 1, bottomRow, topCol, mid_col)
    segmentation(topRow, mid_row, mid_col + 1, bottomCol)
    segmentation(mid_row + 1, bottomRow, mid_col + 1, bottomCol)

row_num = img.shape[0]
col_num = img.shape[1]
segmentation(0, row_num, 0, col_num)

img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
cv.imshow('output', np.rint(img).astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()