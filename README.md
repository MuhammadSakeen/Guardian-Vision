# Women Safety Analytics 🛡️

A real-time AI-based surveillance system aimed at enhancing women’s safety in public environments using deep learning and computer vision.

This project currently supports real-time **person detection** using YOLOv8 and OpenCV. It highlights detected persons with bounding boxes on a live webcam feed.

---

### Current Feature
- Live webcam detection of persons using YOLOv8n
- Filters only “person” class with confidence threshold
- Bounding boxes and labels drawn on screen

---

### Files
- `yolo_webcam.py` — Main detection script using webcam
- `README.md` — Project description

---

### Upcoming Features (Planned)
- Lone woman detection
- Alert logic for potential threats
- Web UI using Streamlit

---

### Tech Stack
- Python
- OpenCV
- YOLOv8n (Ultralytics)

---

### Project Goal
To build a smart surveillance system capable of detecting and alerting unsafe conditions for women in real time.
