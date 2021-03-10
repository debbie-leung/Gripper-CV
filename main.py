import glob
import sys
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import helper_functions as f

# Parameters to be changed
check_x = 8 # this is in fact y (vertical column) -- needs to get fixed with image.shape
check_y = 8
# upper_voxel_size
# lower_voxel_size

# Read image
filename = sys.argv[1] # file path = "photos/img_name.jpeg"
img = cv.imread(filename)
img = imutils.resize(img, width=400)

# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 

# Threshold of magenta in HSV space 
lower_mag = np.array([140,100,20]) # H = 130
upper_mag = np.array([170,255,255]) # H = 170

# Prepare the mask to overlay 
mask = cv.inRange(hsv, lower_mag, upper_mag) 
cv.imshow("mask", mask)

markers = f.watershed(img, mask)

centroids, output, filled_contours = f.centroid_finder(img, markers)

cv.imshow("centroids", output)

# Create checkerboard for theoretical checking

# Output boolean array
pixelx_per_mm = img.shape[0]/(check_x*3) # assume width and height are perfectly square
pixely_per_mm = img.shape[1]/(check_y*3)
grid = np.zeros((check_x,check_y))
print("image size: " + str(img.shape))
print("pixel x: " + str(pixelx_per_mm))
print("pixel x: " + str(pixely_per_mm))

for i in range(check_x):
    for j in range(check_y):
        # if (hsv[voxelx_center + i*voxelx_size, voxely_center + j*voxely_size][0] == 154):
        if filled_contours[int(1.5*pixelx_per_mm + i*3*pixelx_per_mm), int(1.5*pixely_per_mm + j*3*pixely_per_mm)]: 
            cv.circle(img, tuple([int(1.5*pixelx_per_mm + i*3*pixelx_per_mm), int(1.5*pixely_per_mm + j*3*pixely_per_mm)]), radius=2, color=(0, 255, 0), thickness=-1)
            grid[i, j] = 1
print(grid)

cv.imshow("filled", filled_contours)
cv.imshow("boolean", img)

# Output array of centroids
centroid_grid = np.zeros((check_x,check_y), dtype='i,i')

centroids = set(centroids)
centroids = sorted(list(centroids), key=lambda element: element[1])
# print(centroids)
# np.savetxt("centroids.txt", np.array(centroids), fmt='%d')
# print("numpy")
centroids = np.array(centroids)
# print(centroids)

low_y = 0
# for j in range(check_y):
#     # slice centroids list to get each row for sorting
#     #  sorted(centroids[centroids[:, 1] > 50], key=lambda element: element[0])
#     high_y = 0 + (j+1)*pixely_per_mm*3
#     print(low_y)
#     print(high_y)
#     row = centroids[centroids[:, 1] >= low_y]
#     row = row[row[:, 1] <= high_y]
#     # row = [x for x in centroids_reshaped if low_y <= centroids_reshaped[:,1] <= high_y]
#     low_y = high_y
#     row = sorted(row, key=lambda element: element[0])
#     row = np.array(row)
#     row = map(tuple, row)
#     row = tuple(row)
#     idx = np.where(grid[j,:])
#     centroid_grid(idx) 
#     print(row[2])
#     print(grid[j,:])
#     print(row[:, grid[j,:]])
       
cv.waitKey()
    
