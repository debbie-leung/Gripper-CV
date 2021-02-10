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
cv2.imwrite("enlarged.jpg", img)

# img = cv2.GaussianBlur(img, (5, 5), 0)
# img = cv2.addWeighted(img, 1.5, img, -0.5, 0, img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
#print(hsv)

# Threshold of magenta in HSV space 
lower_mag = np.array([144,100,100]) 
upper_mag = np.array([164,255,255])

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
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 
# cap.release() 

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
