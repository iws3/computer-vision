# 4. Count Total Frames in a Video

import cv2

video=cv2.VideoCapture("video.mp4");

total_frames=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps=video.get(cv2.CAP_PROP_FPS)
print("Total Frames:", total_frames);
print("FPS: ",fps)

video.release()