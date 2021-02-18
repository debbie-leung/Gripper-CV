import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils

img = cv.imread('cropped.jpg')
# img = imutils.resize(img, width=700)
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 

# Threshold of magenta in HSV space 
lower_mag = np.array([140,100,20]) # H = 130
upper_mag = np.array([170,255,255]) # H = 170

# preparing the mask to overlay 
mask = cv.inRange(hsv, lower_mag, upper_mag) 

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel, iterations = 5)

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
# cv.imshow('bg', sure_bg)

sure_bg_plt = cv.cvtColor(sure_bg, cv.COLOR_BGR2RGB)
plt.subplot(1, 3, 1)
plt.imshow(sure_bg_plt)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# cv.imshow('distance transform', dist_transform)
# cv.imshow('sure_fg', sure_fg)

sure_fg_plt = cv.cvtColor(sure_fg, cv.COLOR_BGR2RGB)
plt.subplot(1, 3, 2)
plt.imshow(sure_fg_plt)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]

img_plt = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.subplot(1, 3, 3)
plt.imshow(img_plt)
plt.show()
