import cv2
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 
cap = cv2.VideoCapture(0)

# 1. INITIALIZED OUTSIDE: This tracks history
already_counted = False # Simple flag to avoid double-counting the same person

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model.predict(frame, conf=0.5, show=False)
    annotated_frame = results[0].plot()

    # 2. INITIALIZED INSIDE: This resets every frame for "Live" count
    live_person_count = 0 

    for box in results[0].boxes:
        if int(box.cls[0]) == 0: # 0 is the ID for 'person'
            live_person_count += 1

    # Logic to update the "Total" count
    if live_person_count > 0 and not already_counted:
        already_counted = True # We saw someone! Don't count again until they leave
    elif live_person_count == 0:
        already_counted = False # Screen is empty, ready to count the next person

    # 3. Draw the HUD
    cv2.putText(annotated_frame, f"Live: {live_person_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Counter", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()