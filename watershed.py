import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils

# Parameters to be changed
check_x = 8
check_y = 8
# upper_voxel_size
# lower_voxel_size

# Read image
img = cv.imread('cropped.jpg')
img = imutils.resize(img, width=400)
img2 = img.copy()
img3 = img.copy()

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
opening = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel, iterations = 5) # default iterations = 3

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=2) # default iterations = 3
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
img = imutils.resize(img, width=400)
cv.imshow('img1', img)

# Find centroids of markers (https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/)
# img2 = img.copy()
markers1 = markers.astype(np.uint8)
ret, m2 = cv.threshold(markers1, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(m2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

centroids = []
idx = 0

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
        cv.putText(img3, text, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv.LINE_AA)
        cv.drawContours(img3, c, -1, (0, 255, 0), 1)

# print(len(centroids))
centroids = set(centroids)
# print("length")
print(len(centroids))
centroids = sorted(list(centroids), key=lambda element: element[1])
# np.savetxt("centroids.txt", np.array(centroids), fmt='%d')
# print("numpy")
# centroids_reshaped = np.array(centroids).reshape((8,8))
# print(np.array(centroids))
# img2 = imutils.resize(img2, width=400)
# cv.imshow('markers1', markers1)
cv.imshow('img2', img2)
# img3 = imutils.resize(img3, width=400)
cv.imshow('img3', img3)
cv.waitKey()

# Output boolean array
pixelx_per_mm = img.shape[0]/(check_x*3) # assume width and height are perfectly square
grid = np.zeros((check_x,check_y))
# Hardcode finding centroid
# voxelx_center = int((img.shape[0] / check_x) / 2)
# voxely_center = int((img.shape[1] / check_y) / 2)
# voxelx_size = int((img.shape[0] / check_x))
# voxely_size = int((img.shape[1] / check_y))

# print("voxelx_center: ", voxelx_center)
# print("voxely_center: ", voxely_center)
# print("voxelx_size: ", voxelx_size)
# print("voxely_size: ", voxely_size)

# for i in range(check_x):
#     for j in range(check_y):
#         # if (hsv[voxelx_center + i*voxelx_size, voxely_center + j*voxely_size][0] == 154):
#         if mask[voxelx_center + i*voxelx_size, voxely_center + j*voxely_size]: 
#             cv2.circle(img, tuple([voxelx_center + i*voxelx_size, voxely_center + j*voxely_size]), radius=5, color=(0, 0, 255), thickness=-1)
#             grid[i, j] = 1

# Output array of centroids
pixely_per_mm = img.shape[1]/(check_y*3)
centroid_grid = np.zeros((check_x,check_y), dtype='i,i')

low_y = 0
for j in range(voxel_num - 1):
    # slice centroids list to get each row for sorting
    #  sorted(centroids[centroids[:, 1] > 50], key=lambda element: element[0])
    high_y = 0 + (j+1)*pixely_per_mm
    row = x for x in centroids if low_y <= centroids[:, 1] <= high_y)
    low_y = high_y
    row = sorted(row, , key=lambda element: element[0])
    
    
    
