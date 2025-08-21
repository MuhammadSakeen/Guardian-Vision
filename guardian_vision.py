import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import mediapipe as mp
import time

# -----------------------------
# Load Models
# -----------------------------
# YOLOv8n for person detection
model = YOLO(r"models/weights/yolov8n.pt")

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

# Mediapipe Hands (for wave SOS)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

wave_counter = 0
last_x = None
last_time = time.time()

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

                # Draw person bounding box
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
    # Step 4: Alerts
    # -----------------------------
    alerts = []

    # Lone Woman Alert 🚨
    if female_count == 1 and male_count >= 2:
        alerts.append("⚠️ Lone Woman in Crowd!")

    # Unsafe Crowd Ratio ⚖️
    if female_count > 0 and male_count > 5 * female_count:
        alerts.append("⚠️ Female Outnumbered!")

    # Nighttime Detection 🌙
    hour = datetime.datetime.now().hour
    if female_count == 1 and male_count >= 2 and (hour >= 20 or hour < 6):
        alerts.append("🌙 High Risk: Lone Woman at Night")

    # -----------------------------
    # Step 5: Display Counts + Alerts
    # -----------------------------
    cv2.rectangle(frame, (0, 0), (250, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Male: {male_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Female: {female_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    y_pos = height - 20
    for alert_text in alerts:
        cv2.rectangle(frame, (0, y_pos-30), (width, y_pos), (0, 0, 255), -1)
        cv2.putText(frame, alert_text, (10, y_pos-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos -= 40

    # -----------------------------
    # Step 6: SOS Trigger (Keyboard)
    # -----------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        print("🚨 SOS Triggered (Keyboard)!")
        cv2.rectangle(frame, (0, height-60), (width, height), (0,0,255), -1)
        cv2.putText(frame, "🚨 SOS Triggered!", (10, height-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # -----------------------------
    # Step 7: SOS Trigger (Wave Detection)
    # -----------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist_x = hand_landmarks.landmark[0].x
            curr_time = time.time()

            if last_x is not None:
                if abs(wrist_x - last_x) > 0.05:  # hand moved significantly
                    wave_counter += 1
                    last_time = curr_time

            last_x = wrist_x

            if curr_time - last_time > 2:  # reset if idle
                wave_counter = 0

            if wave_counter >= 4:  # SOS if waved 4 times
                print("🚨 SOS Triggered (Wave)!")
                cv2.rectangle(frame, (0, height-100), (width, height-60), (0,0,255), -1)
                cv2.putText(frame, "🚨 SOS Triggered (Wave)!", (10, height-70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                wave_counter = 0

    # -----------------------------
    # Step 8: Show Frame
    # -----------------------------
    cv2.imshow("Guardian Vision - YOLOv8 + Gender + Alerts + SOS", frame)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
