# Import modules
import cv2
import numpy as np
import matplotlib as plt
import imutils
import glob
import sys
import os
from checkerboard import detect_checkerboard

# Define checkerboard dimensions
check_x = 8
check_y = 8
matrix = np.zeros((check_x, check_y))

# Detect voxels in simple checkerboard
filename = sys.argv[1]
img = cv2.imread(filename)
img = imutils.resize(img, width=700)
# cv2.imwrite("enlarged.jpg", img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
#print(hsv)

# Threshold of magenta in HSV space 
lower_mag = np.array([140,100,20]) # H = 130
upper_mag = np.array([170,255,255]) # H = 170

# preparing the mask to overlay 
mask = cv2.inRange(hsv, lower_mag, upper_mag) 
print(mask.shape)
      
# The black region in the mask has the value of 0, 
# so when multiplied with original image removes all non-magenta regions 
result = cv2.bitwise_and(img, img, mask = mask) 
print(result.shape)
  
cv2.imshow('frame', img) 
cv2.imshow('mask', mask) 
cv2.imshow('result', result)     

# 1. Blue then edge detection
blur = cv2.GaussianBlur(result, (5, 5), cv2.BORDER_DEFAULT)

median = cv.medianBlur(img, 3)
bilateral = cv2.bilateralFilter(img, 5, 10, 5) # retain edges
canny = cv2.Canny(blur, 125, 175)
cv2.imshow('Canny Edges', canny)

# 2. Threshold to binarize image
# Feed in filtered image to find centroid
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(mask,127,255,0)
cv2.imshow('thresh', thresh)

# find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
blank = np.zeros(img.shape[:2], dtype='uint8')
cv2.drawContours(blank, contours, -1, (255,0,0), 1)
cv2.imshow('Contours Drawn', blank)
print(len(contours))

# for c in contours:
#     cv2.drawContours(img, [c], -1, (255, 0, 0), 2)
#     # calculate moments for each contour
#     M = cv2.moments(c)

#     # calculate x,y coordinate of center
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#     cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
#     cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#     # display the image
#     cv2.imshow("Image", img)
#     cv2.waitKey(0)

# Hardcode finding centroid
voxelx_center = int((img.shape[0] / check_x) / 2)
voxely_center = int((img.shape[1] / check_y) / 2)
voxelx_size = int((img.shape[0] / check_x))
voxely_size = int((img.shape[1] / check_y))

print("voxelx_center: ", voxelx_center)
print("voxely_center: ", voxely_center)
print("voxelx_size: ", voxelx_size)
print("voxely_size: ", voxely_size)

for i in range(check_x):
    for j in range(check_y):
        # if (hsv[voxelx_center + i*voxelx_size, voxely_center + j*voxely_size][0] == 154):
        if mask[voxelx_center + i*voxelx_size, voxely_center + j*voxely_size]: 
            cv2.circle(img, tuple([voxelx_center + i*voxelx_size, voxely_center + j*voxely_size]), radius=5, color=(0, 0, 255), thickness=-1)
            matrix[i, j] = 1

cv2.imshow('img',img)
cv2.waitKey()
print(matrix)
