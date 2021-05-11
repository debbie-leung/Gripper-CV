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
# cnts, x, y, w, h = largest_4_sided_contour(thresh)
# check = cv2.drawContours(img, [cnts], -1, (0,255,0), 2)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4:
        # (x, y, w, h) = cv2.boundingRect(approx)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # rect = cv2.minAreaRect(approx)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(img,[box],0,(0,0,255),2)
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

#cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow('binary', thresh)
cv2.imshow("Image", img)
cv2.waitKey()
print("finished contours")

# Perform perspective correction
# 1. crop image
img = img[y-10:y+h+10,x-10:x+w+10]
gray = gray[y-10:y+h+10,x-10:x+w+10]
# check = check[y-10:y+h+10,x-10:x+w+10]
# 2. find corners
bi = cv2.bilateralFilter(gray, 5, 75, 75)
dst = cv2.cornerHarris(bi, 2, 3, 0.04)
img[dst > 0.01 * dst.max()] = [0, 0, 255]   #--- [0, 0, 255] --> Red ---
cv2.imshow('dst', img)

# blurred = cv2.GaussianBlur(check, (5, 5), 0)
# thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
# canny = cv2.Canny(thresh, 120, 255, 1)
# corners = cv2.goodFeaturesToTrack(canny,4,0.5,50)

# for corner in corners:
#     x,y = corner.ravel()
#     cv2.circle(img,(x,y),5,(0,0,255),-1)

# cv2.imshow('canny', canny)
# cv2.imshow('image', img)
cv2.waitKey()

# cnts_ls = np.ndarray.tolist(np.squeeze(cnts))
# left_top = min(cnts_ls)
# print(left_top)
# check = cv2.circle(img, tuple(left_top), radius=10, color=(0, 0, 255), thickness=-1)
# right_bottom = max(cnts_ls)
# print(right_bottom)
# check = cv2.circle(check, tuple(right_bottom), radius=10, color=(0, 0, 255), thickness=-1)
# cv2.imshow("check", check)
# cv2.waitKey()

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
