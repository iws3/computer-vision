import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "hand_landmarker.task")

print(f"Looking for model at: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

if not os.path.exists(model_path):
    print("ERROR: hand_landmarker.task model file not found!")
    print("Please download it from: https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task")
    print("Place the file in your project directory and try again.")
    exit(1)

try:
    # Create HandLandmarker options
    BaseOptions = mp.tasks.BaseOptions
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        exit(1)
    
    print("Webcam opened successfully. Press 'q' to quit.")
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        frame_counter = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from webcam")
                break
            
            frame_counter += 1
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hand landmarks
            detection_result = landmarker.detect_for_video(mp_image, frame_counter)
            
            # Draw landmarks if hands are detected
            if detection_result.hand_landmarks:
                # Create a copy of the frame for drawing
                output_frame = frame.copy()
                
                # Draw each hand's landmarks
                for hand_landmarks in detection_result.hand_landmarks:
                    vision.drawing_utils.draw_landmarks(
                        output_frame,
                        hand_landmarks,
                        vision.HandLandmarksConnections.HAND_CONNECTIONS)
            else:
                output_frame = frame
            
            cv2.imshow("Hand Detection", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()