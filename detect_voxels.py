# Import modules
import cv2
import numpy as np
import matplotlib as plt
import imutils
import glob
import sys
import os
from checkerboard import detect_checkerboard
from helper_functions import *

# Detect voxels in simple checkerboard
filename = sys.argv[1]
img = cv2.imread(filename)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
print(hsv)

# Threshold of magenta in HSV space 
lower_mag = np.array([154,100,100]) 
upper_mag = np.array([154,255,255]) 

# preparing the mask to overlay 
mask = cv2.inRange(hsv, lower_mag, upper_mag) 
      
# The black region in the mask has the value of 0, 
# so when multiplied with original image removes all non-magenta regions 
result = cv2.bitwise_and(img, img, mask = mask) 
  
cv2.imshow('frame', img) 
cv2.imshow('mask', mask) 
cv2.imshow('result', result) 
      
cv2.waitKey(0) 
# cv2.destroyAllWindows() 
# cap.release() 

# Create empty matrix
print(img.shape)
matrix = np.zeros((img.shape[0], img.shape[1]))
print(matrix)

# Get checkerboard corners
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (5,7),None)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)
    print(corners2)

    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (5,7), corners2,ret)
    cv2.imshow('img',img)
    cv2.waitKey()

# cv2.destroyAllWindows()