# Gripper-CV
Computer vision script for detecting voxel pickups in gripper

To dos:
V- detect checkerboard pixels
V- detect voxels
V- translate into numpy array
V- do perspective correction
- correct for voxel shading
V- draw lines across image
- fix orientation offset
- expand code to take multiple images and calculate array voxel average accuracy

Steps:
1. Detect and outline gripper square
2. Perform perspective correction on the gripper outline
3. Filter out voxel color

Challenges:
- background noise and rectangular objects in the background; filter largest 4-sided object
- openCV or python checkerboard function does not work
- gripper outline is not perfect rectangle; need to find source points for perspective transformation

New Directions:
1. Hash centroids into boolean array by flooring # of pixels/# of voxels (get # of pixels per voxel)
2. Get error array by
- calculating Euclidean distance from centroids
- calculating theta offset from orientation/minimum inertia
3. Watershed algorithm not perfectly working/robust (check area and contour, tweak parameters)

References:
1. Orientation/Image moments
- https://towardsdatascience.com/computer-vision-for-beginners-part-4-64a8d9856208
- https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/
- https://theailearner.com/tag/cv2-minarearect/
- https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
- https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
2. Finding corresponding HSV values
- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
- https://stackoverflow.com/questions/36817133/identifying-the-range-of-a-color-in-hsv-using-opencv/51686953
3. Perspective correction
- https://stackoverflow.com/questions/22656698/perspective-correction-in-opencv-using-python 
- https://stackoverflow.com/questions/22519545/automatic-perspective-correction-opencv?noredirect=1&lq=1 
- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
4. Camera lens distortion
- https://learnopencv.com/camera-calibration-using-opencv/
- https://learnopencv.com/understanding-lens-distortion/ 
- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
5. Hough line transform
- https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
- https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
- https://stackoverflow.com/questions/40638778/convert-houghlinesp-output-line-format-to-houghlines-output-line-format
- https://stackoverflow.com/questions/36452491/opencv-python-houghlinesp-strange-results

Hough Line Transform Tips:
- grow and shrink the blobs with thresholding (watershed)
- separate image based on color range (gold, purple, black holes)
https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
https://www.tutorialspoint.com/line-detection-in-python-with-opencv
- use math.isclose() for filtering vertical and horizontal lines
- how many lines to plot and how far they are spaced apart