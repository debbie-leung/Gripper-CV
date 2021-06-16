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
img2 = img.copy()
cv.imshow("image", img)

# Parameters for filtering hough lines
num_x = 8 # this is in fact y (vertical column) -- needs to get fixed with image.shape
num_y = 8
pixelx_per_voxel = img.shape[1]/(num_x) 
pixely_per_voxel = img.shape[0]/(num_y)

# # Extract mask for gold electrode and another mask for purple background
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 
# lower_gold = np.array([5,80,100]) 
# upper_gold = np.array([30,255,255]) 
# mask = cv.inRange(hsv, lower_gold, upper_gold) 
# cv.imshow("electrode", mask)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# th, im_th = cv.threshold(img, 100, 255, cv.THRESH_BINARY_INV)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow("binary", ~thresh)
inv_thresh = ~thresh
kernel = np.ones((15,15),np.uint8)
dilation = cv.dilate(inv_thresh,kernel,iterations = 1)
cv.imshow("dilation", dilation)
erosion = cv.erode(dilation,kernel,iterations = 1)
cv.imshow("erosion", erosion)
# mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
# cv.imshow("closing", mask)

# # Get the gray image and process GaussianBlur
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
kernel_size = 5
blur_gray = cv.GaussianBlur(erosion,(kernel_size, kernel_size),0)
# blur_gray = erosion

# Process edge detection use Canny
low_threshold = 50
high_threshold = 150
edges = cv.Canny(blur_gray, low_threshold, high_threshold)
# edges = cv.Canny(mask,50,150,apertureSize = 3)

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

lines = np.squeeze(lines)

# Filter out diagonal and only keep vertical and horizontal
lines = [list(t) for t in lines if ((t[3]-t[1])/(t[2]-t[0]) == 0) or ((t[3]-t[1])/(t[2]-t[0]) == -math.inf) or ((t[3]-t[1])/(t[2]-t[0]) == math.inf)]
# lines_horizontal = [list(t) for t in lines if ((t[3]-t[1])/(t[2]-t[0]) == 0)]
# lines_horizontal = sorted(lines_horizontal[:], key=lambda element: element[0])
# lines_vertical = [list(t) for t in lines if ((t[3]-t[1])/(t[2]-t[0]) == -math.inf) or ((t[3]-t[1])/(t[2]-t[0]) == math.inf)]
# lines_vertical = sorted(lines_vertical[:], key=lambda element: element[1])

# Only take unique x or y points for lines list
# set_x = set([x[0] for x in lines_horizontal])
# set_y = set([y[0] for y in lines_vertical])
# lines_horizontal = [x for x in lines_horizontal if x[0] not in set_x]

# f.draw_hough_lines(lines_horizontal, 0, 2, line_image, pixelx_per_voxel)
# f.draw_hough_lines(lines_vertical, 1, 3, line_image, pixely_per_voxel)

print(lines)

for line in lines:
    x1, y1, x2, y2 = line
    orientation = math.atan2(abs((y2-y1)), abs((x2-x1)))
    cv.line(line_image,(x1,y1), (x2,y2),(0,0,255),2)
    x_extend = int(img.shape[1] * math.cos(orientation))
    y_extend = int(img.shape[0] * math.sin(orientation))
    cv.line(line_image,(x1-x_extend,y1-y_extend),(x1+x_extend,y1+y_extend),(255,0,0),2)

cv.imshow("line image", line_image)
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

# This subpart extracts horizontal and vertical lines using structuring element: https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html

# def show_wait_destroy(winname, img):
#     cv.imshow(winname, img)
#     cv.moveWindow(winname, 500, 0)
#     cv.waitKey(0)
#     cv.destroyWindow(winname)

# image_hsv = None   # global ;(
# pixel = (20,60,80) # some stupid default

# Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
# gray = cv.bitwise_not(gray)
# bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
#                             cv.THRESH_BINARY, 15, -2)
# # Show binary image
# # show_wait_destroy("binary", bw)
# # [bin]
# # [init]
# # Create the images that will use to extract the horizontal and vertical lines
# horizontal = np.copy(mask)
# vertical = np.copy(mask)
# # [init]
# # [horiz]
# # Specify size on horizontal axis
# cols = horizontal.shape[1]
# horizontal_size = cols // num_x
# # Create structure element for extracting horizontal lines through morphology operations
# horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
# # Apply morphology operations
# horizontal = cv.erode(horizontal, horizontalStructure)
# horizontal = cv.dilate(horizontal, horizontalStructure)
# # Show extracted horizontal lines
# show_wait_destroy("horizontal", horizontal)

# # [vert]
# # Specify size on vertical axis
# rows = vertical.shape[0]
# verticalsize = rows // (num_y+2)
# # Create structure element for extracting vertical lines through morphology operations
# verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
# # Apply morphology operations
# vertical = cv.erode(vertical, verticalStructure)
# vertical = cv.dilate(vertical, verticalStructure)
# # Show extracted vertical lines
# show_wait_destroy("vertical", vertical)
