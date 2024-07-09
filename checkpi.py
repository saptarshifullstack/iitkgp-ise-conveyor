import cv2

# RTSP URL
rtsp_url = "rtsp://admin:Melpl@123@192.168.0.197"

# Create a VideoCapture object
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow('Live Camera Feed', frame)

    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
