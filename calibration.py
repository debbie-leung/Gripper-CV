# This testing script does camera calibration

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

"""Steps to remove lens distortion:
1. Perform camera calibration and get the intrinsic camera parameters. This is what we did in the previous post of this series. The intrinsic parameters also include the camera distortion parameters.
2. Refine the camera matrix to control the percentage of unwanted pixels in the undistorted image.
3. Using the refined camera matrix to undistort the image.
"""
# Read in image
filename = sys.argv[1]
img = cv.imread(filename)
img = imutils.resize(img, width=700)
img2 = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 1. Camera calibration to get camera matrix
# Defining the dimensions of checkerboard
CHECKERBOARD = (6,9)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob('./Checkerboard/*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    cv.imshow('img',img)
    cv.waitKey(0)

cv.destroyAllWindows()

h,w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

# 2. Refine camera matrix

newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# 3. Undistort image

# Method 1 to undistort the image
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# Method 2 to undistort the image
mapx,mapy=cv.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv.remap(img,mapx,mapy,cv.INTER_LINEAR)

# Displaying the undistorted image
cv.imshow("undistorted image",dst)
cv.waitKey(0)
