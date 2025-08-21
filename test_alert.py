import cv2
import numpy as np
import glob
import datetime
from ultralytics import YOLO

# -----------------------------
# Load Models
# -----------------------------
model = YOLO(r"models/weights/yolov8n.pt")  # YOLO for person detection

face_cascade = cv2.CascadeClassifier(
    r"models/model_detection/haarcascade_frontalface_default.xml"
)

gender_net = cv2.dnn.readNetFromCaffe(
    r"models/model_detection/gender_deploy.prototxt",
    r"models/model_detection/gender_net.caffemodel"
)
GENDER_LIST = ["Male", "Female"]

# -----------------------------
# Load Test Images
# -----------------------------
image_paths = glob.glob("test_images/*.jpg") + glob.glob("test_images/*.png")

for img_path in image_paths:
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"❌ Could not load {img_path}")
        continue

    # Resize image (max width = 900px)
    max_width = 900
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    male_count, female_count = 0, 0
    height, width, _ = frame.shape

    # -----------------------------
    # Step 1: Detect persons with YOLOv8 (conf threshold = 0.5)
    # -----------------------------
    results = model(frame, stream=True, conf=0.5)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Person ROI
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0:
                    continue

                gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                if len(faces) == 0:
                    continue  # ❌ Skip this detection if no face found

                for (fx, fy, fw, fh) in faces:
                    face = person_roi[fy:fy+fh, fx:fx+fw]

                    # Gender classification
                    blob = cv2.dnn.blobFromImage(
                        face, 1.0, (227, 227),
                        (78.4263377603, 87.7689143744, 114.895847746),
                        swapRB=False
                    )
                    gender_net.setInput(blob)
                    gender_preds = gender_net.forward()
                    gender = GENDER_LIST[gender_preds[0].argmax()]
                    confidence = gender_preds[0].max() * 100

                    # Count
                    if gender == "Male":
                        male_count += 1
                        color = (255, 0, 0)
                    else:
                        female_count += 1
                        color = (255, 0, 255)

                    # Draw boxes
                    cv2.rectangle(frame, (x1+fx, y1+fy), (x1+fx+fw, y1+fy+fh), color, 2)
                    cv2.putText(frame, f"{gender}: {confidence:.1f}%",
                                (x1+fx, y1+fy-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

                # Draw person bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # -----------------------------
    # Step 2: Alerts
    # -----------------------------
    alerts = []
    if female_count == 1 and male_count >= 2:
        alerts.append("⚠️ Lone Woman in Crowd!")
    if female_count > 0 and male_count > 5 * female_count:
        alerts.append("⚠️ Female Outnumbered!")
    hour = datetime.datetime.now().hour
    if female_count == 1 and male_count >= 2 and (hour >= 20 or hour < 6):
        alerts.append("🌙 High Risk: Lone Woman at Night")

    # Show counts
    cv2.rectangle(frame, (0, 0), (250, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Male: {male_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Female: {female_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Show alerts at bottom
    y_pos = height - 20
    for alert_text in alerts:
        cv2.rectangle(frame, (0, y_pos - 30), (width, y_pos), (0, 0, 255), -1)
        cv2.putText(frame, alert_text, (10, y_pos - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos -= 40

    # -----------------------------
    # Display
    # -----------------------------
    cv2.imshow("Guardian Vision - Image Test", frame)
    print(f"✅ Processed {img_path} | Male: {male_count}, Female: {female_count}, Alerts: {alerts}")

    cv2.waitKey(0)

cv2.destroyAllWindows()
