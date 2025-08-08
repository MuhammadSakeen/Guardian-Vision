import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np

# Load gender detection model
gender_net = cv2.dnn.readNetFromCaffe(
    "models/model_detection/gender_deploy (1).prototxt",
    "models/model_detection/gender_net.caffemodel"
)

gender_list = ['M', 'F']

# Start webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()
    if not status:
        break

    # Detect persons in the frame
    bbox, label, conf = cv.detect_common_objects(frame)
    
    male_count = 0
    female_count = 0

    for i, l in enumerate(label):
        if l == 'person':
            # Crop the person from frame
            x, y, x2, y2 = bbox[i]
            person_img = frame[y:y2, x:x2]

            # Detect face within the person
            face_bbox, _ = cv.detect_face(person_img)

            for fx, fy, fx2, fy2 in face_bbox:
                # Extract face image
                face_img = person_img[fy:fy2, fx:fx2]

                # Preprocess and predict gender
                try:
                    blob = cv2.dnn.blobFromImage(
                        face_img, 1.0, (227, 227),
                        (78.4263377603, 87.7689143744, 114.895847746),
                        swapRB=False
                    )
                    gender_net.setInput(blob)
                    gender_preds = gender_net.forward()
                    gender = gender_list[gender_preds[0].argmax()]
                    confidence = gender_preds[0].max()

                    # Count based on prediction
                    if confidence > 0.6:
                        if gender == 'Male':
                            male_count += 1
                        else:
                            female_count += 1

                    # Draw rectangle on face
                    cv2.rectangle(person_img, (fx, fy), (fx2, fy2), (0, 255, 255), 2)
                    cv2.putText(person_img, f"{gender} ({confidence*100:.1f}%)", (fx, fy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                except:
                    pass

    # Draw person bounding boxes
    annotated_frame = draw_bbox(frame, bbox, label, conf)

    # Overlay counts
    cv2.putText(annotated_frame, f"People: {label.count('person')}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Males: {male_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Females: {female_count}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Real-time Gender Classification", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
webcam.release()
cv2.destroyAllWindows()
