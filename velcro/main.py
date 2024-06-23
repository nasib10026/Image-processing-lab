import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import pickle

 #video theke capture kora image
cap = cv2.VideoCapture('Project/Videos/Video2.mp4')
frameCounter = 0
#377,52 944,71  261,624 1058,612

cornerPoints = [[377,52],[944,71],[261,624],[1058,612]]
colorFinder = ColorFinder(False)
#green ball values
hsvVals = {'hmin': 30, 'smin': 34, 'vmin': 0, 'hmax': 41, 'smax': 255, 'vmax': 255}

#perspective function
def getPerspectiveTransform(pts1, pts2):
    A = []

    for i in range(4):
        x, y = pts1[i][0], pts1[i][1]
        u, v = pts2[i][0], pts2[i][1]
        #2 ta kore equation nicchi
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])

    A = np.array(A)
    #performing singluar value decomposition
    U, S, V = np.linalg.svd(A)
    #last row 9 elements converted to 3,3 matrix
    H = V[-1].reshape(3, 3)

    return H

def GaussianBlur(img, kernel_size, sigma):
    # Create a 1D Gaussian kernel
    kernel_1d = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    kernel_1d = np.exp(-0.5 * (kernel_1d / sigma) ** 2)
    kernel_1d /= np.sum(kernel_1d)

    # Perform horizontal convolution
    img_blur = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                for m in range(-kernel_size // 2, kernel_size // 2 + 1):
                    if 0 <= j + m < img.shape[1]:
                        img_blur[i, j, k] += img[i, j + m, k] * kernel_1d[m + kernel_size // 2]

    # Perform vertical convolution
    img_blur_final = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                for n in range(-kernel_size // 2, kernel_size // 2 + 1):
                    if 0 <= i + n < img.shape[0]:
                        img_blur_final[i, j, k] += img_blur[i + n, j, k] * kernel_1d[n + kernel_size // 2]

    return img_blur_final.astype(np.uint8)


def detectColorDarts(img): 
    imgBlur = cv2.GaussianBlur(img,(7,7), 2)
    #imgBlur = GaussianBlur(img,7, 2)
    imgColor, mask = colorFinder.update(imgBlur, hsvVals)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 9)
    mask = cv2.dilate(mask, kernel, iterations=4)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("Image Color", imgColor)
    return mask
    

def warpPerspective(img, H, width, height):
    img_out = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
    #inverse homography matrix calculate kore
    H_inv = np.linalg.inv(H)
    #iterating over each pixels o/p image e
    for i in range(height):
        for j in range(width):
            # destination pixel (i, j) back to the source image // back transformation
            src_pt = H_inv @ np.array([j, i, 1])
            src_pt = src_pt / src_pt[2] #homogenous form theke cartesian form e convert
            x, y = int(src_pt[0]), int(src_pt[1])
            
            #source image er bound check
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                img_out[i, j] = img[y, x]

    return img_out


def getBoard(img):
   #for scaling
   width ,height = int(400 * 1.5),int(380 * 1.5)
   pts1 = np.float32(cornerPoints)
   #now want to convert these points
   pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
   #matrix  = cv2.getPerspectiveTransform(pts1,pts2)
   #imgOutput= cv2.warpPerspective(img,matrix,(width,height))
   matrix  = cv2.getPerspectiveTransform(pts1,pts2)
   imgOutput= cv2.warpPerspective(img,matrix,(width,height))
   for x in range(4):
     cv2.circle(img,(cornerPoints[x][0],cornerPoints[x][1]),15,(0,255,0),cv2.FILLED)  
   return imgOutput

while True:

    frameCounter += 1
    #checking how many frames in video
    if frameCounter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
      frameCounter = 0
      #maximum frame reach korle reset.so on loop cholbe
      cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    
    
    #catching each frame
    #success, img = cap.read()
    img = cv2.imread("velcro/img.png") 
    #imgColor , mask = colorFinder.update(img)
    
    imgBoard = getBoard(img)
    mask = detectColorDarts(img)
    # cv2.imwrite('imgBoard.png',imgBoard)
    cv2.imshow("Image",img) 
    #cv2.imshow("Perspective-Transformed Image", imgBoard) 
    
    cv2.imshow("Image Mask", mask)

    cv2.waitKey(1) 

