import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



base_options=python.BaseOptions(
    model_asset_path="models/pose_landmarker_lite.task"
)


print(f"Observing the base options: {base_options}")

# RESULT: Observing the base options: BaseOptions(model_asset_path='models/pose_landmarker_lite.task', model_asset_buffer=None, delegate=None)

options=vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

print(f"Observing the pose landmarker options: {options}")