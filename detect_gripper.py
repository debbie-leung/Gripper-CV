# This computer vision script performs perspective correction to detect the gripper using OpenCV

# Import modules
import cv2 as cv
import numpy as np
import matplotlib as plt
import imutils
import glob
import sys
import os
from checkerboard import detect_checkerboard
import test_helper as f

# Read in image
filename = sys.argv[1]
img = cv.imread(filename)
img = imutils.resize(img, width=700)
img2 = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blurred = cv.GaussianBlur(gray, (5, 5), 0)
thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)[1]

# Find contours of the gripper in the thresholded image
cnts, x, y, w, h, contours_poly, box = f.largest_4_sided_contour(thresh)
cv.drawContours(img2, [cnts], -1, (0,255,0), 2)

# Change DST list points to SQUARE!!! (take the longer edge)
dst = np.float32([[x,y], [x+w,y], [x,y+h], [x+w,y+h]])
print(dst)
for i in range(4):
    cv.circle(img2, tuple(contours_poly[i][0]), 10, (0, 0, 255), -1)
    text = str(tuple(contours_poly[i][0]))
    cv.putText(img2, text, tuple(contours_poly[i][0]), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(img2, str(tuple(dst[i])), tuple(dst[i]), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv.LINE_AA)

cv.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
# cv.drawContours(img,[rect],0,(0,0,0),2)
# cv.drawContours(img,[box],0,(255,255,255),2)
 
# sort source points (first sort y, then x)
src_points = np.squeeze(contours_poly)
src_points = sorted(src_points, key=lambda element: element[1])
src = sorted(src_points[:2], key=lambda element: element[0])
src.extend(sorted(src_points[2:], key=lambda element: element[0]))
src = np.float32(np.array(src))
print(contours_poly)
print(dst)
print(src)
print(img.shape[:2])
print(img.shape)

h, w = img.shape[:2]
M = cv.getPerspectiveTransform(src, dst)
warped = cv.warpPerspective(img, M, (w, h), flags=cv.INTER_LINEAR)
cv.imshow("warped", warped)

# cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
# 	cv.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)

cv.imshow("Image", img)
cv.imshow("Annotated", img2)
cv.waitKey()
print("finished contours")

"""
# Perform perspective correction
# 1. crop image
img = img[y-10:y+h+10,x-10:x+w+10]
gray = gray[y-10:y+h+10,x-10:x+w+10]
# check = check[y-10:y+h+10,x-10:x+w+10]
# 2. find corners
bi = cv.bilateralFilter(gray, 5, 75, 75)
dst = cv.cornerHarris(bi, 2, 3, 0.04)
img[dst > 0.01 * dst.max()] = [0, 0, 255]   #--- [0, 0, 255] --> Red ---
cv.imshow('dst', img)

# blurred = cv.GaussianBlur(check, (5, 5), 0)
# thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)[1]
# canny = cv.Canny(thresh, 120, 255, 1)
# corners = cv.goodFeaturesToTrack(canny,4,0.5,50)

# for corner in corners:
#     x,y = corner.ravel()
#     cv.circle(img,(x,y),5,(0,0,255),-1)

# cv.imshow('canny', canny)
# cv.imshow('image', img)
cv.waitKey()

# cnts_ls = np.ndarray.tolist(np.squeeze(cnts))
# left_top = min(cnts_ls)
# print(left_top)
# check = cv.circle(img, tuple(left_top), radius=10, color=(0, 0, 255), thickness=-1)
# right_bottom = max(cnts_ls)
# print(right_bottom)
# check = cv.circle(check, tuple(right_bottom), radius=10, color=(0, 0, 255), thickness=-1)
# cv.imshow("check", check)
# cv.waitKey()

# Crop checkerboard

# cv.imshow('Output', roi)
# cv.imwrite('Cropped.jpg', roi)

# Apply threshold to detect checkerboard pattern
roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

#roi_blurred = cv.GaussianBlur(roi_gray, (5, 5), 0)
roi_thresh = cv.threshold(roi, 150, 255, cv.THRESH_BINARY)[1]
cv.imshow('binary', roi_thresh)
cv.waitKey()

nline = 31
ncol = 31
size = (ncol, nline) # size of checkerboard
corners, score = detect_checkerboard(roi, size)
print("corners")
print(corners)
print("score")
print(score)

# Threshold
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret, corners = cv.findChessboardCorners(thresh, (nline, ncol), None)
#img_inverted = np.array(256-thresh, dtype=uint8)
print(ret, corners)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

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

    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    img = cv.drawChessboardCorners(img, (nline, ncol), corners2, ret)
    cv.imshow('img',img)
    cv.waitKey()


# Convert image to grayscale and median blur to smooth image
blur = cv.medianBlur(gray, 5)

# Sharpen image to enhance edges
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv.filter2D(blur, -1, sharpen_kernel)

# Threshold
thresh = cv.threshold(sharpen,100,255, cv.THRESH_BINARY_INV)[1]
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))

# Perform morphological transformations
close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)

# Find contours and filter using minimum/maximum threshold area
cnts = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
"""
