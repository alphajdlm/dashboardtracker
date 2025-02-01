import cv2
import pandas as pd
import numpy as np
import threading
import tkinter as tk
from ultralytics import YOLO
from tracker import Tracker

# Load YOLO model
model = YOLO('yolov8s.pt')

# Define areas for entry and exit
area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]

# Open video file
cap = cv2.VideoCapture('peoplecount1.mp4')

# Load class names
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# Initialize tracking variables
tracker = Tracker()
people_entering = {}
people_exiting = {}
entering = set()
exiting = set()

# Tkinter Dashboard setup
root = tk.Tk()
root.title("People Tracking Dashboard")
root.geometry("400x300")
root.config(bg="#333")
root.attributes("-topmost", True)  # Ensure the dashboard stays on top of other windows

# Label styling
label_font = ("Arial", 16)

# Dashboard components
entering_label = tk.Label(root, text="Entering: 0", font=label_font, fg="white", bg="#333")
entering_label.pack(pady=10)

exiting_label = tk.Label(root, text="Exiting: 0", font=label_font, fg="white", bg="#333")
exiting_label.pack(pady=10)

total_label = tk.Label(root, text="Total Persons: 0", font=label_font, fg="white", bg="#333")
total_label.pack(pady=10)

# Update dashboard data
def update_dashboard():
    while True:
        entering_label.config(text=f"Entering: {len(entering)}")
        exiting_label.config(text=f"Exiting: {len(exiting)}")
        total_label.config(text=f"Total Persons: {len(entering) + len(exiting)}")
        root.update_idletasks()

# Run dashboard in a separate thread
dashboard_thread = threading.Thread(target=update_dashboard, daemon=True)
dashboard_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    frame = cv2.resize(frame, (1020, 500))

    # YOLO Detection
    results = model.predict(frame)
    boxes = results[0].boxes.data
    px = pd.DataFrame(boxes).astype("float")

    object_list = []
    for _, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row)
        if class_list[class_id] == 'person':
            object_list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(object_list)

    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox

        # Check entry area
        if cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False) >= 0:
            if obj_id not in entering and obj_id not in people_exiting:
                people_entering[obj_id] = (x4, y4)
                entering.add(obj_id)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        # Check exit area
        if obj_id in entering and obj_id not in people_exiting:
            if cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False) >= 0:
                people_exiting[obj_id] = (x4, y4)
                exiting.add(obj_id)
                entering.remove(obj_id)  # Remove from entering since they have exited
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                cv2.putText(frame, str(obj_id), (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw areas on the frame
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)

    # Display counts directly on the video frame
    cv2.putText(frame, f'Entering: {len(entering)} Exiting: {len(exiting)}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("People Tracking", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
root.quit()
