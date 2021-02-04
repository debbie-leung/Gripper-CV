# Gripper-CV
Computer vision script for detecting voxel pickups in gripper

To dos:
- detect checkerboard pixels
- detect voxels
- translate into numpy array
- do perspective correction

Steps:
1. Detect and outline gripper square
2. Perform perspective correction on the gripper outline
3. Filter out voxel color

Challenges:
- background noise and rectangular objects in the background; filter largest 4-sided object
- openCV or python checkerboard function does not work
- gripper outline is not perfect rectangle; need to find source points for perspective transformation