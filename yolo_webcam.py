from ultralytics import YOLO #used for object detection
import cv2 #model of openCV to open or access webcam

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0) # cap stands for capture

while True:
    ret, frame = cap.read() # ret ->  tells us if the frame was successfully captured (True/False), frame is the actual image from the webcam

    if not ret:
        break

    results = model(frame)[0]

    #annotated = results[0].plot() #plot() -> draw a box or rectangle to objects it found with confidedent score
    
    annotated = frame.copy()

    for box in results.boxes:
        cls_id = int(box.cls[0]) #built in id numebrs to say what obj it is
        conf = float(box.conf[0]) #return the considence score
        label = model.names[cls_id] #Convert the class number into a word, like 'person' or 'car'

        if label == "person" and conf > 0.6:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) #These are the box corners: (x1, y1) = top-left, (x2, y2) = bottom-right We use these to draw the box.
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2git ) # draws a green box of thickness 2
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("People Detection Only", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()