import cv2
import matplotlib.pyplot as plt
import numpy as np

# Resize images
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Helper function to find largest 4-sided square for checkboard
def largest_4_sided_contour(processed, show_contours=False):
    contours, _ = cv2.findContours(
        processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # check cv2.RETR_EXTERNAL
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    font = cv2.FONT_HERSHEY_COMPLEX 
    for cnt in contours[:min(3, len(contours))]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(cnt)
            return cnt, x, y, w, h
    return None
    
def unwarp(img, src, dst, testing):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)