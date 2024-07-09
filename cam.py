#detect and measure

import cv2
import pandas as pd
from ultralytics import YOLO
import time
import threading
import numpy as np

# RTSP URL
rtsp_url = "rtsp://admin:Melpl@123@192.168.0.197"

# Create a VideoCapture object
def connect_rtsp(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return None
    return cap

class FrameCaptureThread(threading.Thread):
    def __init__(self, url):
        threading.Thread.__init__(self)
        self.cap = connect_rtsp(url)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print("Error: Could not read frame. Reconnecting...")
                self.cap.release()
                self.cap = None
                time.sleep(2)
                self.cap = connect_rtsp(rtsp_url)
        self.cap.release()

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False

# Load the YOLO model
model_local_path = r"C:\Users\Public\Saptarshi\best.pt"
model = YOLO(model_local_path)  # Replace with your actual model path

# Function to calculate diameter
def calculate_diameter(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Define the coordinates for the region of interest (ROI)
x_start = 750  # starting x coordinate
y_start = 0    # starting y coordinate
x_end = 1150   # ending x coordinate
y_end = 800    # ending y coordinate

# Start frame capture thread
frame_capture_thread = FrameCaptureThread(rtsp_url)
frame_capture_thread.start()

while True:
    frame = frame_capture_thread.get_frame()
    
    if frame is None:
        time.sleep(0.1)
        continue

    # Crop the frame to the defined ROI
    roi_frame = frame[y_start:y_end, x_start:x_end]

    # Run YOLO model on the cropped frame
    results = model.predict(roi_frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Count current detections
    current_count = len(px)

    # Draw bounding boxes and calculate diameter on the cropped frame
    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box

        # Calculate diameter
        diameter = calculate_diameter(x1, y1, x2, y2)
        cv2.putText(roi_frame, f'Diameter: {diameter:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the cropped frame with current detections count
    cv2.putText(roi_frame, f'Current Detections: {current_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Cropped Camera Feed with Detections", roi_frame)

    # Press 'q' on keyboard to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Stop the frame capture thread and release resources
frame_capture_thread.stop()
frame_capture_thread.join()
cv2.destroyAllWindows()

