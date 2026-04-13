import cv2
import os
video=cv2.VideoCapture("video.mp4")

os.makedirs("frames", exist_ok=True)

count=0

while True:
    ret, frame=video.read()
    if not ret or count==10:
        break;
    cv2.imwrite(f"frames/frame_{count}.jpg", frame)
    count+=1