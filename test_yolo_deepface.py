from ultralytics import YOLO
from deepface import DeepFace
import cv2

# 1. Test YOLO
print("\n--- Testing YOLO ---")
try:
    model = YOLO("yolov8n.pt")  # lightweight YOLO model
    print("YOLO loaded successfully ✅")
except Exception as e:
    print("YOLO error ❌:", e)

# 2. Test DeepFace
print("\n--- Testing DeepFace ---")
try:
    result = DeepFace.analyze(img_path="https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img1.jpg",
                              actions=['age', 'gender'],
                              enforce_detection=False)
    print("DeepFace analysis successful ✅")
    print("Prediction:", result)
except Exception as e:
    print("DeepFace error ❌:", e)

print("\n🎉 Both YOLO + DeepFace tested!")
