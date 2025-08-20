import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Load Models
# -----------------------------
# YOLOv8n for person detection
model = YOLO(r"models/model_detection/weights/yolov8n.pt")

# Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(
    r"models/model_detection/haarcascade_frontalface_default.xml"
)

# Caffe gender classification model
gender_net = cv2.dnn.readNetFromCaffe(
    r"models/model_detection/gender_deploy.prototxt",
    r"models/model_detection/gender_net.caffemodel"
)
GENDER_LIST = ["Male", "Female"]

# -----------------------------
# Video Capture
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    male_count, female_count = 0, 0
    height, width, _ = frame.shape

    # -----------------------------
    # Step 1: Detect persons with YOLOv8
    # -----------------------------
    results = model(frame, stream=True)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # class ID
            if cls == 0:  # class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw person bounding box (optional)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # -----------------------------
                # Step 2: Detect face inside person box
                # -----------------------------
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0:
                    continue

                gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10)
                )

                for (fx, fy, fw, fh) in faces:
                    face = person_roi[fy:fy+fh, fx:fx+fw]

                    # -----------------------------
                    # Step 3: Gender classification
                    # -----------------------------
                    blob = cv2.dnn.blobFromImage(
                        face, 1.0, (227, 227),
                        (78.4263377603, 87.7689143744, 114.895847746),
                        swapRB=False
                    )
                    gender_net.setInput(blob)
                    gender_preds = gender_net.forward()
                    gender = GENDER_LIST[gender_preds[0].argmax()]
                    confidence = gender_preds[0].max() * 100

                    # Count genders
                    if gender == "Male":
                        male_count += 1
                        color = (255, 0, 0)  # Blue
                    else:
                        female_count += 1
                        color = (255, 0, 255)  # Pink

                    # Draw face box + label
                    cv2.rectangle(person_roi, (fx, fy), (fx+fw, fy+fh), color, 2)
                    cv2.putText(frame, f"{gender}: {confidence:.1f}%",
                                (x1+fx, y1+fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -----------------------------
    # Step 4: Display Counts
    # -----------------------------
    cv2.rectangle(frame, (0, 0), (250, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Male: {male_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Female: {female_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Show frame
    cv2.imshow("Guardian Vision - YOLOv8 + Gender", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
