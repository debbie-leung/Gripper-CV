import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import helper_functions as f

# Parameters to be changed
check_x = 8
check_y = 8
# upper_voxel_size
# lower_voxel_size

# Read image
img = cv.imread('missing.jpeg')
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

centroids, img3, filled_contours = f.centroid_finder(img, markers)

cv.imshow("img3", img3)

# Output boolean array
pixelx_per_mm = img.shape[0]/(check_x*3) # assume width and height are perfectly square
pixely_per_mm = img.shape[1]/(check_y*3)
grid = np.zeros((check_x,check_y))
print(grid)

for i in range(check_x):
    for j in range(check_y):
        # if (hsv[voxelx_center + i*voxelx_size, voxely_center + j*voxely_size][0] == 154):
        if filled_contours[int(1.5*pixelx_per_mm + i*3*pixelx_per_mm), int(1.5*pixely_per_mm + j*3*pixely_per_mm)]: 
            cv.circle(img4, tuple([int(1.5*pixelx_per_mm + i*3*pixelx_per_mm), int(1.5*pixely_per_mm + j*3*pixely_per_mm)]), radius=2, color=(0, 255, 0), thickness=-1)
            grid[i, j] = 2
# print(grid)

voxelx_center = int((img.shape[0] / check_x) / 2)
voxely_center = int((img.shape[1] / check_y) / 2)
voxelx_size = int((img.shape[0] / check_x))
voxely_size = int((img.shape[1] / check_y))

for i in range(check_x):
    for j in range(check_y):
        # if (hsv[voxelx_center + i*voxelx_size, voxely_center + j*voxely_size][0] == 154):
        if mask[voxelx_center + i*voxelx_size, voxely_center + j*voxely_size]: 
            cv.circle(img5, tuple([voxelx_center + i*voxelx_size, voxely_center + j*voxely_size]), radius=2, color=(255, 0, 0), thickness=-1)
            grid[i, j] = 1

cv.imshow("img4", img4)
cv.imshow("filled", filled_contours)
# cv.imshow('img5',img5)
cv.waitKey()

# print(grid)

# Output array of centroids
centroid_grid = np.zeros((check_x,check_y), dtype='i,i')

centroids = set(centroids)
# print("length")
# print(len(centroids))
centroids = sorted(list(centroids), key=lambda element: element[1])
# print(centroids)
# np.savetxt("centroids.txt", np.array(centroids), fmt='%d')
# print("numpy")
centroids = np.array(centroids)
# print(centroids)

# low_y = 0
# for j in range(check_y - 1):
#     # slice centroids list to get each row for sorting
#     #  sorted(centroids[centroids[:, 1] > 50], key=lambda element: element[0])
#     high_y = 0 + (j+1)*pixely_per_mm
#     print(low_y)
#     print(high_y)
#     row = centroids[centroids[:, 1] >= low_y]
#     row = row[row[:, 1] <= high_y]
#     # row = [x for x in centroids_reshaped if low_y <= centroids_reshaped[:,1] <= high_y]
#     low_y = high_y
#     row = sorted(row, key=lambda element: element[0])
#     print(list(row))
    
    
    
