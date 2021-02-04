# This computer vision script detects and quantifies the number of voxels picked up by the gripper using OpenCV

# Import modules
import cv2
import numpy as np
import matplotlib as plt
import imutils
import glob
import sys
import os

# Helper function to find largest 4-sided square for checkboard
def largest_4_sided_contour(processed, show_contours=False):
    contours, _ = cv2.findContours(
        processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # check cv2.RETR_EXTERNAL
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:min(3, len(contours))]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            return cnt
    return None

# Detect checkboard
nline = 31
ncol = 31

filename = sys.argv[1]
img = cv2.imread(filename)
img = imutils.resize(img, width=700)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image
cnts = largest_4_sided_contour(thresh)
cv2.drawContours(img, [cnts], -1, (0, 255, 0), 2)
cv2.imshow('binary', thresh)
cv2.imshow("Image", img)
cv2.waitKey()

contours, hierarchy = cv2.findContours(thresh, 1, 2)
x,y,w,h = cv2.boundingRect(contours[0])
rect = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
print(rect)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img,[box],0,(0,0,255),2)

# Threshold
print("finished contours")

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