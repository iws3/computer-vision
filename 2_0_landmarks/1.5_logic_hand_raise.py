# step3_logic_hand_raise.py
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LEFT_WRIST = 15
LEFT_SHOULDER = 11

base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker_full.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]

        if lm[LEFT_WRIST].y < lm[LEFT_SHOULDER].y:
            cv2.putText(frame, "HAND RAISED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Step 3: Hand Raise Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()