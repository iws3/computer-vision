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

# create_from_options: This is the Power Button. It actually loads the model into your computer's memory using the rules you just defined.

options=vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

print(f"Observing the pose landmarker options: {options}")


# Observing the pose landmarker options: PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path='models/pose_landmarker_lite.task', model_asset_buffer=None, delegate=None), running_mode=<VisionTaskRunningMode.VIDEO: 'VIDEO'>, num_poses=1, min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5, output_segmentation_masks=False, result_callback=None)

landmarker=vision.PoseLandmarker.create_from_options(options)

cap=cv2.VideoCapture(0)

while True:
    # This is a loop that runs roughly 30 times per second. Every "lap" of this loop is one single frame of video.
    ret, frame=cap.read()
    
    if not ret:
        break;
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb);
    
    timestamp=int(time.time() * 1000)
    # result.pose_landmarks: This is a big list of 33 points (nose, shoulders, elbows, etc.).

    result=landmarker.detect_for_video(mp_image, timestamp)
    print(f"Observing the result object: {result}")
    h, w, _=frame.shape
    
    if result.pose_landmarks:
        lm=result.pose_landmarks[0];
        # print nose and left wrist
        print("Nose:", lm[0].x, lm[0].y)
        print("Left wrist:", lm[15].x, lm[15].y)
        
        nose_x=int(lm[0].x*w)
        nose_y=int(lm[0].y*h)
        # get shoulder pixels (left 11, Right 12)
        l_sh_x, l_sh_y=int(lm[11].x*w), int(lm[11].y*h)
        r_sh_x, r_sh_y=int(lm[12].x*w), int(lm[12].y*h)
        # drawing section
        cv2.circle(frame, (nose_x, nose_y), 10, (255, 0,0), -1)
        # draw a green line between shoulders
        cv2.line(frame, (l_sh_x, l_sh_y), (r_sh_x, r_sh_y), (0,255,0), 3)
        
        
        
    cv2.imshow("Step 2: Coordinates", frame)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()