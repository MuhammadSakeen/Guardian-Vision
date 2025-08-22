# guardian_vision_with_sos_logging.py
import os
import cv2
import csv
import time
import datetime
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()

# optional external libs
try:
    import mediapipe as mp
except Exception:
    raise RuntimeError("mediapipe not installed. pip install mediapipe")

# -----------------------------
# CONFIG
# -----------------------------
ALERT_LOG_FILE = "alerts_log.csv"
CAMERA_ID = os.getenv("CAMERA_ID", "CAMERA_1")
LOCATION_NAME = os.getenv("LOCATION_NAME", "")
SOS_COOLDOWN = 60       # seconds for SOS
GESTURE_COOLDOWN = 10   # seconds for gesture
ALERT_DISPLAY_DURATION = 6  # seconds for on-screen alerts

# -----------------------------
# Helper functions
# -----------------------------
def ensure_log_exists(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "alert", "male", "female", "camera_id", "location"])

def log_alert(alert_type, male_count, female_count):
    ensure_log_exists(ALERT_LOG_FILE)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, alert_type, male_count, female_count, CAMERA_ID, LOCATION_NAME]
    with open(ALERT_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print("Logged alert:", row)

last_sos_time = 0
def trigger_sos(alert_reason, male_count, female_count, show_on_frame_callback=None):
    global last_sos_time
    now = time.time()
    if now - last_sos_time < SOS_COOLDOWN:
        print("SOS cooldown active, ignoring repeated SOS.")
        return
    last_sos_time = now
    log_alert(alert_reason, male_count, female_count)
    if show_on_frame_callback:
        show_on_frame_callback(f"🚨 {alert_reason} | male={male_count} female={female_count}")

# -----------------------------
# Load Models
# -----------------------------
model = YOLO(r"models/weights/yolov8n.pt")
face_cascade = cv2.CascadeClassifier(
    r"models/model_detection/haarcascade_frontalface_default.xml"
)
gender_net = cv2.dnn.readNetFromCaffe(
    r"models/model_detection/gender_deploy.prototxt",
    r"models/model_detection/gender_net.caffemodel"
)
GENDER_LIST = ["Male", "Female"]

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

wave_counter = 0
last_wrist_x = None
last_wrist_time = time.time()

# -----------------------------
# Video capture
# -----------------------------
cap = cv2.VideoCapture(0)
sos_banner_text = None
sos_banner_expire = 0

def show_banner(text, duration=6):
    global sos_banner_text, sos_banner_expire
    sos_banner_text = text
    sos_banner_expire = time.time() + duration

ensure_log_exists(ALERT_LOG_FILE)
print("Starting Guardian Vision... Press 'q' to quit.")

last_male_count = 0
last_female_count = 0
gesture_last_trigger = {"wave":0, "both_hands_up":0, "keyboard":0}
active_alerts = []  # (text, expire_time)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width = frame.shape[:2]
    male_count = 0
    female_count = 0

    # -----------------------------
    # YOLO person detection
    # -----------------------------
    results = model(frame, stream=True, conf=0.45)
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0: continue
            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            if len(faces)==0: continue
            for (fx, fy, fw, fh) in faces:
                face = person_roi[fy:fy+fh, fx:fx+fw]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227,227),
                                             (78.4263377603, 87.7689143744, 114.895847746),
                                             swapRB=False)
                gender_net.setInput(blob)
                preds = gender_net.forward()
                gender = GENDER_LIST[int(preds[0].argmax())]
                if gender=="Male": male_count+=1; color=(255,0,0)
                else: female_count+=1; color=(255,0,255)
                cv2.rectangle(frame,(x1+fx,y1+fy),(x1+fx+fw,y1+fy+fh),color,2)
                cv2.putText(frame,f"{gender}: {preds[0].max()*100:.1f}%",(x1+fx,y1+fy-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)

    last_male_count = male_count
    last_female_count = female_count

    # -----------------------------
    # Alerts logic
    # -----------------------------
    alerts = []
    hour = datetime.datetime.now().hour
    if female_count==1 and male_count>=2: alerts.append("⚠️ Lone Woman in Crowd!")
    if female_count>0 and male_count>5*female_count: alerts.append("⚠️ Female Outnumbered!")
    if female_count==1 and male_count>=2 and (hour>=20 or hour<6): alerts.append("🌙 High Risk: Lone Woman at Night")

    # Draw counts
    cv2.rectangle(frame,(0,0),(260,60),(0,0,0),-1)
    cv2.putText(frame,f"Male: {male_count}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
    cv2.putText(frame,f"Female: {female_count}",(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

    # -----------------------------
    # Log alerts and update active_alerts for display
    # -----------------------------
    curr_time = time.time()
    for alert_text in alerts:
        # Always log every alert detection
        log_alert(alert_text, male_count, female_count)
        # Add to active_alerts for on-screen display
        if all(alert_text != a[0] for a in active_alerts):
            active_alerts.append((alert_text, curr_time + ALERT_DISPLAY_DURATION))

    # Remove expired alerts
    active_alerts = [(text, expire) for text, expire in active_alerts if curr_time < expire]

    # Draw alerts
    y_pos = height - 20
    for alert_text, _ in active_alerts:
        cv2.rectangle(frame,(0,y_pos-30),(width,y_pos),(0,0,255),-1)
        cv2.putText(frame,alert_text,(10,y_pos-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        y_pos -= 40

    # -----------------------------
    # Gesture detection
    # -----------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(rgb)
    pose_results = pose.process(rgb)

    both_hands_up = False
    try:
        if hands_results.multi_hand_landmarks and pose_results.pose_landmarks:
            pose_lm = pose_results.pose_landmarks.landmark
            left_sh_y = pose_lm[11].y
            right_sh_y = pose_lm[12].y
            wrists_y = [h.landmark[0].y for h in hands_results.multi_hand_landmarks]
            shoulder_line = (left_sh_y+right_sh_y)/2.0
            if len(wrists_y)>=2 and wrists_y[0]<shoulder_line and wrists_y[1]<shoulder_line:
                both_hands_up = True
    except Exception:
        both_hands_up=False

    wave_detected=False
    if hands_results.multi_hand_landmarks:
        hw = hands_results.multi_hand_landmarks[0]
        wrist_x = hw.landmark[0].x
        if last_wrist_x is not None:
            dx = abs(wrist_x-last_wrist_x)
            if dx>0.06:
                wave_counter+=1
                last_wrist_time=time.time()
        last_wrist_x=wrist_x
        if time.time()-last_wrist_time>1.6: wave_counter=0
        if wave_counter>=4:
            wave_detected=True
            wave_counter=0

    # Draw mediapipe
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame,pose_results.pose_landmarks,mp_pose.POSE_CONNECTIONS)

    # -----------------------------
    # Trigger SOS gestures with cooldown
    # -----------------------------
    curr_time = time.time()
    if wave_detected and curr_time - gesture_last_trigger["wave"] > GESTURE_COOLDOWN:
        gesture_last_trigger["wave"]=curr_time
        reason="SOS Triggered (Wave Gesture)"
        show_banner(reason,duration=6)
        trigger_sos(reason,last_male_count,last_female_count,show_on_frame_callback=show_banner)

    if both_hands_up and curr_time - gesture_last_trigger["both_hands_up"] > GESTURE_COOLDOWN:
        gesture_last_trigger["both_hands_up"]=curr_time
        reason="SOS Triggered (Both Hands Up)"
        show_banner(reason,duration=6)
        trigger_sos(reason,last_male_count,last_female_count,show_on_frame_callback=show_banner)

    key = cv2.waitKey(1) & 0xFF
    if key==ord("s") and curr_time - gesture_last_trigger["keyboard"] > GESTURE_COOLDOWN:
        gesture_last_trigger["keyboard"]=curr_time
        reason="SOS Triggered (Keyboard)"
        show_banner(reason,duration=6)
        trigger_sos(reason,last_male_count,last_female_count,show_on_frame_callback=show_banner)

    # Draw SOS banner
    if sos_banner_text and time.time() < sos_banner_expire:
        cv2.rectangle(frame,(0,height-80),(width,height-40),(0,0,255),-1)
        cv2.putText(frame,sos_banner_text[:80],(10,height-55),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    else:
        sos_banner_text=None

    cv2.imshow("Guardian Vision - Alerts & SOS", frame)
    if key==ord("q"): break

# Cleanup
cap.release()
cv2.destroyAllWindows()
