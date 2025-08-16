from ultralytics import YOLO
import cv2
import numpy as np
import datetime

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Load gender classification model
gender_net = cv2.dnn.readNetFromCaffe(
    r"C:\\Women_Safety_Analytics\\models\\model_detection\\gender_deploy.prototxt",
    r"C:\\Women_Safety_Analytics\\models\\model_detection\\gender_net.caffemodel"
)
GENDER_LIST = ["Male", "Female"]

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CONFIDENCE_THRESHOLD = 50  # Lower threshold for smoother classification
FACE_SHRINK_RATIO = 0.05   # Smaller shrink for more face context

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0], conf=0.5)
    male_count, female_count = 0, 0

    for result in results:
        for box in result.boxes:
            coords = box.xyxy
            if hasattr(coords, 'cpu'):
                coords = coords.cpu().numpy()
            coords = np.array(coords).flatten()
            x1, y1, x2, y2 = map(int, coords[:4])

            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue

            h, w = person_img.shape[:2]
            if h < 50 or w < 50:
                continue

            # Light preprocessing
            person_img = cv2.convertScaleAbs(person_img, alpha=1.2, beta=20)
            gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            min_face_size = 40
            face_counted = False

            for (fx, fy, fw, fh) in faces:
                # Skip tiny regions
                if fw < min_face_size or fh < min_face_size:
                    continue

                shrink_w, shrink_h = int(fw * FACE_SHRINK_RATIO), int(fh * FACE_SHRINK_RATIO)
                sx1 = max(fx + shrink_w, 0)
                sy1 = max(fy + shrink_h, 0)
                sx2 = min(fx + fw - shrink_w, w)
                sy2 = min(fy + fh - shrink_h, h)
                face_img = person_img[sy1:sy2, sx1:sx2]

                if face_img.size == 0:
                    continue

                # Gender classification
                blob_gender = cv2.dnn.blobFromImage(
                    face_img, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False
                )
                gender_net.setInput(blob_gender)
                gender_preds = gender_net.forward()[0]  # [male_score, female_score]

                male_score = float(gender_preds[0])
                female_score = float(gender_preds[1])

                confidence_gender = max(male_score, female_score) * 100
                gender_idx = gender_preds.argmax()
                gender = GENDER_LIST[gender_idx]

                print(f"Raw gender prediction: Male {male_score:.2f}, Female {female_score:.2f}. Decision: {gender} ({confidence_gender:.1f}%)")

                # Decide label + color
                if confidence_gender < CONFIDENCE_THRESHOLD:
                    gender = "Unknown"
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (255, 0, 0) if gender == "Male" else (255, 0, 255)
                    if gender == "Male":
                        male_count += 1
                    elif gender == "Female":
                        female_count += 1

                # Draw bounding box + label
                cv2.rectangle(frame, (x1+sx1, y1+sy1), (x1+sx2, y1+sy2), color, 2)
                label = (f"M:{male_score*100:.1f}% | F:{female_score*100:.1f}%"
                         if gender != "Unknown"
                         else "Unknown")
                cv2.putText(frame, label, (x1+sx1, y1+sy1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow("Face Crop", face_img)

                # Only count one face per detected person
                face_counted = True
                break

    # Lone woman alert at night
    is_night = datetime.datetime.now().hour >= 20 or datetime.datetime.now().hour < 6
    if is_night and female_count == 1 and male_count == 0:
        cv2.putText(frame, "LONE WOMAN ALERT!", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Overlay male/female counts
    cv2.rectangle(frame, (0, 0), (250, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Male: {male_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Female: {female_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imshow("Guardian Vision - YOLOv8n", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('c'):
        cv2.destroyWindow("Face Crop")

cap.release()
cv2.destroyAllWindows()
