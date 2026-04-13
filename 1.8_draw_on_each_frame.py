# Draw Something on Each Frame

import cv2
video=cv2.VideoCapture("video.mp4")

while True:
    ret, frame=video.read()
    if not ret:
        break;
    
    
    # Draw rectangle
    
    cv2.rectangle(frame, (300, 100), (400, 700), (0, 255, 0), 8)
    
    # Add text
    
    cv2.putText(frame, "Frame Processing", (50, 40), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 0, 0), 1)
    
    cv2.imshow("Annotated Frame",frame)
    
    if cv2.waitKey(30) & 0xFF ==ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()