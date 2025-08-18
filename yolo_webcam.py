from ultralytics import YOLO
import cv2
from deepface import DeepFace
import numpy as np

# Load YOLOv8n model
model = YOLO("yolov8n.pt")  # Make sure yolov8n.pt is in the same folder

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    male_count = 0
    female_count = 0

    # YOLO inference
    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        # Only consider person class (YOLO class 0)
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_roi = frame[y1:y2, x1:x2] # roi - Region of Interest

        h, w = person_roi.shape[:2]
        if h < 30 or w < 30:
            # skip tiny regions
            continue

        try:
            analysis = DeepFace.analyze(
                person_roi,
                actions=['age', 'gender'],
                enforce_detection=False
            )

            # Handle DeepFace list output
            if isinstance(analysis, list):
                analysis = analysis[0]

            age = int(round(analysis['age']))
            gender = analysis.get('dominant_gender', 'Unknown')
            gender_conf = analysis.get('gender', {}).get(gender, 0.0)

            # Update counts
            if gender == 'Man':
                male_count += 1
                color = (0, 255, 0)
            else:
                female_count += 1
                color = (255, 0, 255)

            label = f"{gender} ({gender_conf:.1f}%), Age: {age}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except Exception as e:
            print("DeepFace skipped:", e)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display male/female counts on top-left
    cv2.putText(frame, f"Male: {male_count}  Female: {female_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Guardian Vision", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
