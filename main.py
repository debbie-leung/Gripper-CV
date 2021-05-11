# This is the main working script for detecting voxels
# Complete: watershed algorithm to assign voxels to boolean array
# Todo: perform perspective correction, draw lines across image, correct error arrays, calculate accuracy accumulated across images

# Import modules
import glob
import sys
import os
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import main_helper as f

# Parameters to be changed
# 1. number of grid in x-y space of the gripper
num_x = 32 # img.shape[1] is the x-length (width)
num_y = 32 # img.shape[0] is the y length (height)
# 2. total pixel area of each detected voxel
upper_voxel_size = 5000 # 3000 for 8x8 array (resized), 5000 for 32x32 array
lower_voxel_size = 3000 # 1500 for 8x8 array (resized), 3000 for 32x32 array
# 3. threshold of magenta/blue in HSV space 
lower_mag = np.array([140,100,100]) 
upper_mag = np.array([170,255,255]) 
lower_blue = np.array([100,100,100]) 
upper_blue = np.array([130,255,255]) 
lower = lower_blue
upper = upper_blue

# Initialization of boolean, distance and orientation error arrays
grid = np.zeros((num_y,num_x),dtype=int)
dist = np.zeros((num_y,num_x),dtype=float)
ori = np.zeros((num_y,num_x),dtype=float)
# Set blocked voxel spaces as -1
grid[1::2,::2] = -1
grid[::2,1::2] = -1
dist[1::2,::2] = -1
dist[::2,1::2] = -1
ori[1::2,::2] = -1
ori[::2,1::2] = -1

#####################################################################################
# Read image
filename = sys.argv[1] # file path = "photos/img_name.jpeg"
img = cv.imread(filename)
img_contour = img.copy()

# Calculate the number of pixels per voxel
pixelx_per_voxel = img.shape[1]/(num_x) 
pixely_per_voxel = img.shape[0]/(num_y)

# Convert image to hsv for masking
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 
mask = cv.inRange(hsv, lower, upper) 

# Perform watershed (may need to tweak the num parameters)
markers = f.watershed(img, mask, filename, num_morph=5, num_dilate=3, num_dist=0.4)

# Get the contours of each voxel to assign to boolean and error arrays
ret, m2 = cv.threshold(markers, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(m2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

for c in contours:

    # calculate moments for each contour
    M = cv.moments(c)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"]) # M["m00"] is the area of the blob
    cY = int(M["m01"] / M["m00"])

    if lower_voxel_size < M["m00"] < upper_voxel_size: 
        cv.circle(img_contour, (cX, cY), 1, (255, 255, 255), -1)
        text = str((cX, cY))
        cv.putText(img_contour, text, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv.LINE_AA)
        cv.drawContours(img_contour, c, -1, (0,255,0), 1)

        # Boolean array: find where the centroid should belong to
        x_idx = int(cX//pixelx_per_voxel)
        y_idx = int(cY//pixely_per_voxel)
        grid[y_idx, x_idx] = 1

# Write image and numpy arrays into excel
cv.imwrite(os.path.splitext(filename)[0] + "_contour.jpg", img_contour)
num_voxels = np.count_nonzero(grid == 1)
# np.savetxt(os.path.splitext(filename)[0] + "_grid_" + str(num_voxels) + ".csv", grid, fmt='%d', delimiter=',')

# Display images for troubleshooting
cv.imshow("centroids", img_contour)
cv.waitKey()