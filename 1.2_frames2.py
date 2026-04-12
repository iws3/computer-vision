# 2. A Frame is Data (Pixels)
# Let’s inspect pixel values.?

import numpy as np
import cv2
import matplotlib.pyplot as plt


# Read an image (this is a frame)

frame=cv2.imread("image.png")

print(type(frame))
print(frame[0,0])
print(frame[100, 200])

# Insight:
# Each pixel = numbers
# Computer vision = math on these numbers