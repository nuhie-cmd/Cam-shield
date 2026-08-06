from ultralytics import YOLO
import cv2
import os
from tampering import check_tampering
from test_videos.live_video import get_live_

# Load YOLOse model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# Create assets folder if it doesn't exist
os.makedirs("dashboard/assets", exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    tampered, message = check_tampering(frame)
    
    # Run detection
    results = model(frame)

    # Draw detections
    annotated = results[0].plot()
    if tampered:
     cv2.putText(
        annotated,
        message,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Check if a person is detected
    for box in results[0].boxes:
        cls = int(box.cls[0])

        # Class 0 = person in COCO dataset
        if cls == 0:
            cv2.imwrite("dashboard/assets/evidence.jpeg", annotated)

    cv2.imshow("CamShield Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()