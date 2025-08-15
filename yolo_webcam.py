from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Load gender classification model
gender_net = cv2.dnn.readNetFromCaffe(
    r"C:\Women_Safety_Analytics\models\model_detection\gender_deploy.prototxt",
    r"C:\Women_Safety_Analytics\models\model_detection\gender_net.caffemodel"
)
GENDER_LIST = ["Male", "Female"]

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CONFIDENCE_THRESHOLD = 70  # Minimum confidence to accept gender prediction
FACE_SHRINK_RATIO = 0.1    # Shrink detected face by 10% for better precision

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection for 'person' class
    results = model(frame, classes=[0], conf=0.5)

    male_count, female_count = 0, 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop person box
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue

            h, w = person_img.shape[:2]
            if h < 50 or w < 50:
                continue

            # Adjust brightness/contrast
            person_img = cv2.convertScaleAbs(person_img, alpha=1.2, beta=20)

            # Detect faces inside person box
            gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (fx, fy, fw, fh) in faces:
                # Shrink face box for precision
                shrink_w, shrink_h = int(fw * FACE_SHRINK_RATIO), int(fh * FACE_SHRINK_RATIO)
                sx1 = fx + shrink_w
                sy1 = fy + shrink_h
                sx2 = fx + fw - shrink_w
                sy2 = fy + fh - shrink_h
                face_img = person_img[sy1:sy2, sx1:sx2]

                if face_img.size == 0:
                    continue

                # Prepare blob for gender model
                blob_gender = cv2.dnn.blobFromImage(
                    face_img, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False
                )
                gender_net.setInput(blob_gender)
                gender_preds = gender_net.forward()
                confidence_gender = gender_preds[0].max() * 100

                if confidence_gender < CONFIDENCE_THRESHOLD:
                    gender = "Unknown"
                    color = (0, 255, 255)  # Yellow for uncertain
                else:
                    gender = GENDER_LIST[gender_preds[0].argmax()]
                    color = (255, 0, 0) if gender == "Male" else (255, 0, 255)

                    # Count males/females
                    if gender == "Male":
                        male_count += 1
                    else:
                        female_count += 1

                # Draw face box & label
                cv2.rectangle(frame, (x1+sx1, y1+sy1), (x1+sx2, y1+sy2), color, 2)
                label = f"{gender}: {confidence_gender:.1f}%" if gender != "Unknown" else "Unknown"
                cv2.putText(frame, label, (x1+sx1, y1+sy1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display counts
    cv2.rectangle(frame, (0, 0), (250, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Male: {male_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Female: {female_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imshow("Guardian Vision - YOLOv8n", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
