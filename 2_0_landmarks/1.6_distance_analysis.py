# # step4_distance_analysis.py

# import cv2
# import mediapipe as mp
# import time
# import math
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# def distance(a, b):
#     return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

# base_options = python.BaseOptions(
#     model_asset_path="models/pose_landmarker_lite.task"
# )

# options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.VIDEO
# )

# landmarker = vision.PoseLandmarker.create_from_options(options)

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

#     timestamp = int(time.time() * 1000)
#     result = landmarker.detect_for_video(mp_image, timestamp)

#     if result.pose_landmarks:
#         lm = result.pose_landmarks[0]

#         d = distance(lm[11], lm[12])  # shoulders

#         cv2.putText(frame, f"Shoulder Dist: {d:.2f}",
#                     (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                     1, (255, 0, 0), 2)

#     cv2.imshow("Step 4: Distance Analysis", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()







import cv2
import mediapipe as mp
import time
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- STEP 1: UTILITY FUNCTIONS ---
def get_distance(p1, p2):
    """Calculates the mathematical distance between two points (0.0 to 1.0)"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# --- STEP 2: SETUP MEDIAPIPE ---
base_options = python.BaseOptions(model_asset_path="models/pose_landmarker_lite.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# --- STEP 3: START CAMERA ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Get dimensions for drawing
    h, w, _ = frame.shape

    # Prepare Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)

    # Run AI
    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]

        # 1. Coordinate Definitions (Indices)
        L_SHOULDER = lm[11]
        R_SHOULDER = lm[12]
        L_WRIST = lm[15]

        # 2. Convert to Pixel Coordinates for Drawing
        ls_px = (int(L_SHOULDER.x * w), int(L_SHOULDER.y * h))
        rs_px = (int(R_SHOULDER.x * w), int(R_SHOULDER.y * h))
        lw_px = (int(L_WRIST.x * w), int(L_WRIST.y * h))

        # 3. Logic: Distance Calculation
        shoulder_dist = get_distance(L_SHOULDER, R_SHOULDER)

        # 4. Logic: Hand Raise Detection
        # Remember: Smaller Y means HIGHER on screen
        is_raised = L_WRIST.y < L_SHOULDER.y

        # --- DRAWING SECTION ---
        
        # Draw Shoulder Line (Blue)
        cv2.line(frame, ls_px, rs_px, (255, 0, 0), 3)
        
        # Draw Shoulder Dots
        cv2.circle(frame, ls_px, 8, (255, 255, 255), -1)
        cv2.circle(frame, rs_px, 8, (255, 255, 255), -1)

        # Draw Wrist Dot (Changes color if raised)
        wrist_color = (0, 255, 0) if is_raised else (0, 0, 255) # Green if up, Red if down
        cv2.circle(frame, lw_px, 12, wrist_color, -1)

        # Display Distance Text
        cv2.putText(frame, f"Width: {shoulder_dist:.2f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display Hand Raise Alert
        if is_raised:
            cv2.putText(frame, "HAND RAISED!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show Output
    cv2.imshow("The Full System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()