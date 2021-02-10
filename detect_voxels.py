# This computer vision script detects and quantifies the number of voxels picked up by the gripper using OpenCV

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

# Read in image
filename = sys.argv[1]
img = cv2.imread(filename)
img = imutils.resize(img, width=700)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

# Find contours of the gripper in the thresholded image
cnts, x, y, w, h = largest_4_sided_contour(thresh)
cv2.drawContours(img, [cnts], -1, (0, 255, 0), 2)
#cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
print(type(cnts))
print(cnts.shape)
print(len(cnts))
print(cnts)

cv2.imshow('binary', thresh)
cv2.imshow("Image", img)
cv2.waitKey()
print("finished contours")

# Perform perspective correction
# 1. crop image
img = img[y:y+h,x:x+w]
# 2. find corners
cnts_ls = np.ndarray.tolist(np.squeeze(cnts))
left_top = min(cnts_ls)
print(left_top)
check = cv2.circle(img, tuple(left_top), radius=10, color=(0, 0, 255), thickness=-1)
right_bottom = max(cnts_ls)
print(right_bottom)
check = cv2.circle(check, tuple(right_bottom), radius=10, color=(0, 0, 255), thickness=-1)
cv2.imshow("check", check)
cv2.waitKey()

# Crop checkerboard

# cv2.imshow('Output', roi)
# cv2.imwrite('Cropped.jpg', roi)

# Apply threshold to detect checkerboard pattern
roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

#roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
roi_thresh = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('binary', roi_thresh)
cv2.waitKey()

nline = 31
ncol = 31
size = (ncol, nline) # size of checkerboard
corners, score = detect_checkerboard(roi, size)
print("corners")
print(corners)
print("score")
print(score)

# Threshold
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret, corners = cv2.findChessboardCorners(thresh, (nline, ncol), None)
#img_inverted = np.array(256-thresh, dtype=uint8)
print(ret, corners)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# If found, add object points, image points (after refining them)
if ret == True:
    print("checkerboard detected")
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (nline, ncol), corners2, ret)
    cv2.imshow('img',img)
    cv2.waitKey()

""" 
# Convert image to grayscale and median blur to smooth image
blur = cv2.medianBlur(gray, 5)

# Sharpen image to enhance edges
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

# Threshold
thresh = cv2.threshold(sharpen,100,255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

# Perform morphological transformations
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours and filter using minimum/maximum threshold area
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
"""

# Detect voxels