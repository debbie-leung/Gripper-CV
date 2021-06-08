# This testing script detects and draws lines across gripper board

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
import test_helper as f

# Read in image
filename = sys.argv[1]
img = cv.imread(filename)
img = imutils.resize(img, width=700)

# Parameters for filtering hough lines
num_x = 8 # this is in fact y (vertical column) -- needs to get fixed with image.shape
num_y = 8
pixelx_per_voxel = img.shape[1]/(num_x) 
pixely_per_voxel = img.shape[0]/(num_y)

# Get the gray image and process GaussianBlur
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
kernel_size = 5
blur_gray = cv.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Process edge detection use Canny
low_threshold = 50
high_threshold = 150
edges = cv.Canny(blur_gray, low_threshold, high_threshold)
# edges = cv.Canny(mask,50,150,apertureSize = 3)

# ret, thresh = cv.threshold(cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY) , 127, 255, cv.THRESH_BINARY)
# contours, hier = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img, contours, -1, (255, 0, 0), 1)

# for cnt in contours:
#     epsilon = 0.01 * cv.arcLength(cnt, True)
#     approx = cv.approxPolyDP(cnt, epsilon, True)
# cv.imshow("contours", approx)

# Use HoughLinesP to get the lines. You can adjust the parameters for better performance.
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 100  # minimum number of votes (intersections in Hough grid cell) (or 10/15)
min_line_length = 100  # minimum number of pixels making up a line (or 10/50)
max_line_gap = 80  # maximum gap in pixels between connectable line segments (or 5/20)
line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

# Sort lines 
lines = np.squeeze(lines)
lines = sorted(lines[:], key=lambda element: element[0])

#for line in lines:
    # for x1,y1,x2,y2 in line:
for i in range(len(lines)):
    x1, y1, x2, y2 = lines[i]
    # Filter out diagonal and only keep vertical and horizontal
    slope = (y2-y1)/(x2-x1)       
    if (slope == 0) or (slope == -math.inf) or (slope == math.inf):
        print(lines[i], slope)
        # Filter out lines not spaced enough apart for voxels
        # if (i < len(lines)-1):
        #     print(lines[i+1][0]-x1, pixelx_per_voxel)
        #     if math.isclose(lines[i+1][0]-x1, pixelx_per_voxel, abs_tol=10) or math.isclose(lines[i+1][1]-y1, pixely_per_voxel, abs_tol=10):
        #         cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)
        # else:
        cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)

# Draw the lines on the  image
lines_edges = cv.addWeighted(img, 0.8, line_image, 1, 0)

cv.imshow("canny", edges)
cv.imshow("line edges", lines_edges)

# lines = cv.HoughLines(edges,0.1,np.pi/360,50)
# print(lines)
# img2 = np.copy(img) * 0  # creating a blank to draw lines on
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
#         cv.line(img2,(x1,y1),(x2,y2),(0,0,255),1)

# cv.imshow("full", img2)
cv.waitKey()
