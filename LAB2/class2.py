import cv2
import numpy as np
import math

def logKernel(sigma, MUL = 7):
    n = int(sigma * MUL)
    n = n | 1
    kernel = np.zeros( (n,n) )

    center = n // 2
    part1 = -1 / (np.pi * sigma**4)
    
    for x in range(n):
        for y in range(n):
            dx = x - center
            dy = y - center
            
            part2 = (dx**2 + dy**2) / (2 * sigma**2)
            
            kernel[x][y] =  part1 * (1 - part2) * np.exp(-part2)

    
    return (kernel)



def zero_crossing_thresholding(image,t_val):
    rows = image.shape[0] 
    cols = image.shape[1]
    zero_crossing_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [image[i - 1, j], image[i + 1, j], image[i, j - 1], image[i, j + 1]]
            if any(np.sign(image[i, j]) != np.sign(neighbor) for neighbor in neighbors):
                 if image[i,j]>8:
                     zero_crossing_image[i, j] = 255
            else:
                zero_crossing_image[i,j] = 0    

    return zero_crossing_image 



def convolution(kernel, image):
    w = kernel.shape[1]//2
    h = kernel.shape[0]//2
    img_bordered = cv2.copyMakeBorder(src=image, top=h, bottom=h, left = w, right = w,  borderType=cv2.BORDER_CONSTANT)
    out = np.zeros((img_bordered.shape[0], img_bordered.shape[1]), dtype=np.float32)


    for i in range(h, img_bordered.shape[0] - h):
        for j in range(w, img_bordered.shape[1]- w):
            sum = 0 
            for x in range(-w, w + 1):
                for y in range(-h, h + 1):
                    sum += kernel[x + w, y + h] * img_bordered[i - x, j - y]
            out[i, j] = sum    

    # cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    # out = np.round(out).astype(np.uint8)
    # cv2.imshow("Output Image", out)     
    return out

img = cv2.imread("/Users/rakibulnasib/Desktop/image/LAB2/Lena.jpg", cv2.IMREAD_GRAYSCALE)
kernel = logKernel(1,7)
print(kernel)
convoluted_img = convolution(kernel,img)
cv2.imshow("output image",convoluted_img)
cv2.waitKey(0)
zero_crossed_image = zero_crossing_thresholding(convoluted_img,60)

cv2.imshow("output image",zero_crossed_image)
cv2.waitKey(0)

row = zero_crossed_image.shape[0]
col = zero_crossed_image.shape[1]
pad = kernel.shape[0]//2

# for i in range(pad,row-pad):
#     for j in range(pad,col-pad):
#         local_region = zero_crossed_image[i-pad:i+pad+1]




