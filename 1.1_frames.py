# FRAME IS JUST AN IMAGE
# PROVE THAT A FRAME = IMAGE (MATRIX OF PIXDELS)

import cv2
import matplotlib.pyplot as plt

# Read an image  (this is a "frame" in the context of video processing  but here we are just using a single image)

frame=cv2.imread("image.png")

# open cv loads in BGR=> CONVERT TO RGB
frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#  dISPLAY THE IMAGE using matplotlib
plt.imshow(frame_rgb)
plt.title("This is a Frame (just an image)")
plt.axis("off")
plt.show()

print("Shape of frame:", frame.shape)