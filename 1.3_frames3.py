# 🟢 3. Video = Sequence of Frames

import cv2

video=cv2.VideoCapture("video.mp4")

while True:
    ret, frame=video.read()
    print("Understanding ret", ret)
    print("Understanding frame", frame)
    if not ret:
        break
    
    cv2.imshow("Video Frame", frame)
    
    # Press 'q' to stop
    if cv2.waitKey(30) & 0XFF == ord('q'):
        break
    video.release()
    cv2.destroyAllWindows()
    
    