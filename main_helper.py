# This contain helper functions that accompany the main script 

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from math import atan2, cos, sin, sqrt, pi

# Watershed function for delineating voxels
def watershed(img, mask, filename, num_morph, num_dilate, num_dist): # add parameters for tweaking
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel, iterations=num_morph) # default iterations = 5

    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=num_dilate) # default iterations = 3

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,num_dist*dist_transform.max(),255,0) # default parameter = 0.7 (used 0.4)

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
    markers = markers.astype(np.uint8)

    return markers