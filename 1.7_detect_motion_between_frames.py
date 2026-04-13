import cv2

video=cv2.VideoCapture("video.mp4")
ret, prev_frame=video.read()
prev_gray=cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame=video.read()
    if not ret:
        break;
    
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Differences between frames
    diff=cv2.absdiff(prev_gray, gray)
    print(diff)
    cv2.imshow("Motion Detection", diff)
    prev_gray=gray
    if cv2.waitKey(30) & 0xFF==ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()