import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LEFT_WRIST = 15
LEFT_SHOULDER = 11

base_options = python.BaseOptions(model_asset_path="models/pose_landmarker_lite.task")
options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Get frame dimensions
    h, w, _ = frame.shape 

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]

        # 2. CONVERT TO PIXELS
        # Shoulder
        sh_x, sh_y = int(lm[LEFT_SHOULDER].x * w), int(lm[LEFT_SHOULDER].y * h)
        # Wrist
        wr_x, wr_y = int(lm[LEFT_WRIST].x * w), int(lm[LEFT_WRIST].y * h)

        # 3. DRAW THE VISUALS
        # Draw a Blue dot on Shoulder
        cv2.circle(frame, (sh_x, sh_y), 10, (255, 0, 0), -1) 
        # Draw a Red dot on Wrist
        cv2.circle(frame, (wr_x, wr_y), 10, (0, 0, 255), -1) 
        # Draw a White line between them
        cv2.line(frame, (sh_x, sh_y), (wr_x, wr_y), (255, 255, 255), 2)

        # LOGIC CHECK
        if lm[LEFT_WRIST].y < lm[LEFT_SHOULDER].y:
            cv2.putText(frame, "HAND RAISED", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Step 3: Hand Raise Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()