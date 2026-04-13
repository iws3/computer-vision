import cv2
video=cv2.VideoCapture("video.mp4")

while True:
    ret, frame=video.read()
    if not ret:
        break
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    cv2.imshow("Original", frame)
    cv2.imshow("Gray Frame", gray)
    
    if  cv2.waitKey(30) & 0xFF==ord('q'):
        break
video.release()
cv2.destroyAllWindows();

# Computer vision = processing each frame
# Same operation repeated many times