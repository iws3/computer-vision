# Writng code to do first pose detection
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models

import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# LOAD MODEL

base_options=python.BaseOptions(
    model_asset_path="models/pose_landmarker_lite.task"
)

options=vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
# landmarker=vision.PoseLandmarker.create_from_options(options)
landmarker = vision.PoseLandmarker.create_from_options(options)

print("Model loaded successfully")
print("______________________________________________");

# print("Landmarker here is: ", landmarker)
# START CAMERA
cap=cv2.VideoCapture(0)
# process frames

while True:
    ret, frame=cap.read()
    if not ret:
        break
    # convert BGR ->RGB
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # convert each frame to media pipe image
    mp_image=mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )
    # timestamp [required for video mode]
    
    timestamp=int(time.time()*1000)
    # run detection
    result=landmarker.detect_for_video(mp_image, timestamp)
    # draw landmarks
    if result.pose_landmarks:
        h, w, _=frame.shape
        for lm in result.pose_landmarks[0]:
            x=int(lm.x*w)
            y=int(lm.y*h)
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
    # show output
    
    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# release resources
cap.release()
cv2.destroyAllWindows()