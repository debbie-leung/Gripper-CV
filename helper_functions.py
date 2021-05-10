import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from math import atan2, cos, sin, sqrt, pi

# Resize images
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Helper function to find largest 4-sided square for checkboard
def largest_4_sided_contour(processed, show_contours=False):
    contours, _ = cv.findContours(
        processed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # check cv.RETR_EXTERNAL
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    font = cv.FONT_HERSHEY_COMPLEX 
    for cnt in contours[:min(3, len(contours))]:
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv.boundingRect(cnt)
            return cnt, x, y, w, h
    return None
    
def unwarp(img, src, dst, testing):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

def watershed(img, mask, filename): # add parameters for tweaking
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel, iterations = 5) # default iterations = 5

    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3) # default iterations = 3
    # cv.imshow('bg', sure_bg)

    sure_bg_plt = cv.cvtColor(sure_bg, cv.COLOR_BGR2RGB)
    plt.subplot(1, 3, 1)
    plt.imshow(sure_bg_plt)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.4*dist_transform.max(),255,0) # default parameter = 0.7 (used 0.4)
    # cv.imshow('distance transform', dist_transform)
    # cv.imshow('sure_fg', sure_fg)

    sure_fg_plt = cv.cvtColor(sure_fg, cv.COLOR_BGR2RGB)
    plt.subplot(1, 3, 2)
    plt.imshow(sure_fg_plt)
    cv.imwrite(os.path.splitext(filename)[0] + "_foreground.jpg", sure_fg)

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

    return markers

def centroid_finder(img, markers):
    # Find centroids of markers (https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/)

    # Output: annotated image with centroids, image with contour fills

    markers1 = markers.astype(np.uint8)
    ret, m2 = cv.threshold(markers1, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(m2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    centroids = []
    idx = 0

    filled_contours = np.zeros((img.shape[0], img.shape[1]),np.uint8)

    img2 = img.copy()

    for i, c in enumerate(contours):

        # calculate moments for each contour
        M = cv.moments(c)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"]) # M["m00"] is the area of the blob
        cY = int(M["m01"] / M["m00"])
        # if 120 < M["m00"] < 180:
        if 1500 < M["m00"] < 3000:
            centroids.append((cX, cY))
            cv.circle(img2, (cX, cY), 1, (255, 255, 255), -1)
            text = str((cX, cY))
            cv.putText(img2, text, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv.LINE_AA)
            cv.drawContours(img2, c, -1, (0, 255, 0), 1)
            cv.drawContours(filled_contours, [c], -1, (255, 255, 255), -1)
        
    return centroids, img2, filled_contours 


def raw_moment(data, i_order, j_order):
    nrows, ncols = data.shape
    y_indices, x_indicies = np.mgrid[:nrows, :ncols]
    return (data * x_indicies**i_order * y_indices**j_order).sum()

def moments_cov(data):
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_centroid = m10 / data_sum
    y_centroid = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return cov

def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
  ## [visualization1]
 
def getOrientation(pts, img):
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  ## [visualization]
  # Draw the principal components
#   cv.circle(img, cntr, 3, (255, 0, 255), 2)
#   p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
#   p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
#   drawAxis(img, cntr, p1, (255, 255, 0), 1)
#   drawAxis(img, cntr, p2, (0, 0, 255), 5)
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
#   angle = -int(np.rad2deg(angle)) - 90
  ## [visualization]
 
  # Label with the rotation angle
#   label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
#   textbox = cv.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
#   cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
 
  return angle