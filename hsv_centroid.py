# Import modules
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import glob
import sys
import os

filename = sys.argv[1]
img = cv2.imread(filename)
img = imutils.resize(img, width=700)
voxel = img[10:100, 10:95]
# cv2.imshow('voxel', voxel)

# create a mask for one magental voxel
blank = np.zeros(img.shape[:2], dtype='uint8')
mask = cv2.rectangle(blank.copy(), (10,10), (100,95), 255, -1)
masked = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('Masked', masked)

# RGB Color Space
plt.figure()
plt.title('RGB Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')

colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()

# HSV Color Space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s = hsv[:1].flatten()
plt.hist(s, 256)
plt.show()

h = hsv[:2].flatten()
plt.hist(h, 256)
plt.show()

# plt.figure()
# plt.title('HSV Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')

# colors = ('b', 'g', 'r')
# for i, col in enumerate(colors):
#     hist = cv2.calcHist([hsv], [i], mask, [256], [0, 256])
#     plt.plot(hist, color=col)
#     plt.xlim([0, 256])

# plt.show()
  
# hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# cv2.imshow('hist', hist)
# plt.imshow(hist,interpolation = 'nearest')
# print(hsv)

# Adaptive thresholding
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,100,255,0)
cv2.imshow('thresh', thresh)
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
cv2.imshow("Adap Mean Thresh", adaptive_thresh)
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
cv2.imshow("Adap Gaussian Thresh", adaptive_thresh)
cv2.waitKey(0)

# Filter out magenta voxels and find centroid
lowerBound = cv2.Scalar(144, 100, 20)
upperBound = cv2.Scalar(165, 255, 255)

# this gives you the mask for those in the ranges you specified
cv2.InRange(cv_input, lowerBound, upperBound, cv_output)
# This will set all bits in cv_input 
cv2.Not(cv_output, cv_inverse)

# calculate moments of binary image
M = cv2.moments(thresh)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# put text and highlight the center
cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# display the image
cv2.imshow("Image", img)
cv2.waitKey(0)