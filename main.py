import glob
import sys
import os
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import helper_functions as f

# Parameters to be changed
check_x = 8 # this is in fact y (vertical column) -- needs to get fixed with image.shape
check_y = 8
# img.shape[0] is the y length (height)
# img.shape[1] is the x-length (width)
# Need to check this value!
upper_voxel_size = 3000 # 180 low resolution
lower_voxel_size = 1500 # 120 low resolution

# Read image
filename = sys.argv[1] # file path = "photos/img_name.jpeg"
img = cv.imread(filename)
img = imutils.resize(img, width=400)
img2 = img.copy()
img3 = img.copy()

# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 

# Threshold of magenta in HSV space 
lower_mag = np.array([140,100,100]) # H = 130 (HSV = 140,100,20)
upper_mag = np.array([170,255,255]) # H = 170
lower_blue = np.array([100,100,100]) #RGB = 0, 128, 255 (HSV = 110,50,50)
upper_blue = np.array([130,255,255]) # 130,255,255
lower = lower_mag
upper = upper_mag

# Prepare the mask to overlay 
mask = cv.inRange(hsv, lower, upper) 
cv.imshow("mask", mask)

markers = f.watershed(img, mask, filename)

# centroids, output, filled_contours = f.centroid_finder(img, markers)
markers1 = markers.astype(np.uint8)
ret, m2 = cv.threshold(markers1, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(m2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

centroids = []
idx = 0

filled_contours = np.zeros((img.shape[0], img.shape[1]),np.uint8)

pixelx_per_mm = img.shape[1]/(check_x*3)
pixely_per_mm = img.shape[0]/(check_y*3)
pixelx_per_voxel = img.shape[1]/(check_x) 
pixely_per_voxel = img.shape[0]/(check_y)

# Create centroid array
centroid_grid = np.zeros((check_y,check_x),dtype='i,i')
# centroid_grid[1::2,::2] = (-1, -1)
# centroid_grid[::2,1::2] = (-1, -1)

# Create boolean array
grid = np.zeros((check_y,check_x),dtype=int)
grid[1::2,::2] = -1
grid[::2,1::2] = -1

# Create error distance array
dist = np.zeros((check_y,check_x),dtype=float)
dist[1::2,::2] = -1
dist[::2,1::2] = -1

# Create error orientation array
ori = np.zeros((check_y,check_x),dtype=float)
ori[1::2,::2] = -1
ori[::2,1::2] = -1

for c in contours:

    # calculate moments for each contour
    M = cv.moments(c)
    area = M["m00"]

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"]) # M["m00"] is the area of the blob
    cY = int(M["m01"] / M["m00"])

    if lower_voxel_size < M["m00"] < upper_voxel_size: 
        # print(M["m00"])
        centroids.append((cX, cY))
        cv.circle(img2, (cX, cY), 1, (255, 255, 255), -1)
        text = str((cX, cY))
        #cv.putText(img2, text, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv.LINE_AA)
        cv.drawContours(img2, c, -1, (0, 255, 0), 1)
        cv.drawContours(filled_contours, [c], -1, (255, 255, 255), -1)

        # Boolean array: find where the centroid should belong to
        x_idx = int(cX//pixelx_per_voxel)
        y_idx = int(cY//pixely_per_voxel)
        grid[y_idx, x_idx] = 1
        centroid_grid[y_idx, x_idx] = (cX, cY)

        # translation error array: TALK TO JONI about having perfectly bordered voxels
        # xt = 1.5*pixelx_per_mm + x_idx*3*pixelx_per_mm # thereotical centroid x-coord
        # yt = 1.5*pixely_per_mm + y_idx*3*pixely_per_mm # thereotical centroid y-coord
        # diff = math.sqrt((cX-xt)**2 + (cY-yt)**2)
        # dist[y_idx, x_idx] = diff

        # rotation error array (minAreaRect)
        # (x,y),(MA,ma),angle = cv.fitEllipse(c)
        rect = cv.minAreaRect(c)
        # print(rect)
        box = cv.boxPoints(rect) 
        box = np.int0(box)
        cv.drawContours(img2,[box],0,(0,0,255),1)
        angle = rect[2]
        ori[y_idx, x_idx] = angle
    
        # using moments (use minimum inertia)
        # u20 = (M["m20"] / M["m00"]) - cX**2
        # u02 = (M["m02"] / M["m00"]) - cY**2
        # u11 = (M["m11"] / M["m00"]) - cX*cY
        # angle = math.degrees(math.atan2(2*u11, u20-u02) / 2)
        # ori[y_idx, x_idx] = angle

        # draw angle line
        length = 10
        P2x = int(cX + length * math.cos(math.radians(angle)))
        P2y = int(cY + length * math.sin(math.radians(angle)))
        cv.line(img2,(cX, cY),(P2x,P2y),(255,255,255),3)

print(grid)
print(dist)
print(ori)

# draw lines across image
voxely_size = int((img.shape[0] / check_y))
for i in range(1, check_y):
    cv.line(img3, (0, i*voxely_size), (img.shape[1],i*voxely_size), (255,255,255), 1)

voxelx_size = int((img.shape[1] / check_x))
for j in range(1, check_x):
    cv.line(img3, (j*voxelx_size, 0), (j*voxelx_size, img.shape[0]), (255,255,255), 1)

# hough transform
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(mask,50,150,apertureSize = 3)
cv.imshow("canny", edges)
minLineLength = 10
maxLineGap = 5
lines = cv.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
for i in range(len(lines)):
    for x1,y1,x2,y2 in lines[i]:
        cv.line(img3,(x1,y1),(x2,y2),(0,0,255),2)

# lines = cv.HoughLines(edges,0.1,np.pi/360,50)
# print(len(lines))
# for i in range(len(lines)):
#     for rho,theta in lines[i]:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 50*(-b))
#         y1 = int(y0 + 50*(a))
#         x2 = int(x0 - 50*(-b))
#         y2 = int(y0 - 50*(a))
    # cv.line(img3,(x1,y1),(x2,y2),(0,0,255),1)

cv.imshow("hough", img3)

# print number of voxels
# unique, counts = np.unique(grid, return_counts=True)
# print(dict(zip(unique, counts)))
# num_voxels = np.count_nonzero(grid == 1)
# np.savetxt(os.path.splitext(filename)[0] + "_grid_" + str(num_voxels) + ".csv", grid, fmt='%d', delimiter=',')

# np.savetxt(os.path.splitext(filename)[0] + "check_sum.csv", grid[grid == 1], fmt='%d', delimiter=',')
# print(np.sum(grid[grid == 1]))
# np.savetxt(os.path.splitext(filename)[0] + "_centroid.csv", centroid_grid, fmt='%f, %f', delimiter=',')

# Save error arrays
# np.savetxt("dist.txt", dist, fmt='%d')
# np.savetxt("ori_moment.txt", ori, fmt='%d')

cv.imshow("centroids", img2)
cv.imshow("edges", img)
# cv.imwrite(os.path.splitext(filename)[0] + "_centroids.jpg", img2)

cv.waitKey()