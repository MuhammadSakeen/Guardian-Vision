# Guardian Vision
*AI-powered surveillance system for women’s safety*

Guardian Vision is a real-time AI-based analytics system designed to **enhance women’s safety** in public environments using deep learning and computer vision.  
It detects people, identifies gender, and provides actionable insights through live video feeds.

---

## Features
- **Real-time Person Detection** using YOLOv8n (Nano version, optimized for speed).  
- **Face Detection** with Haarcascade inside each person bounding box.  
- **Gender Classification** using Caffe model (Male/Female).  
- **Live Counts** of males & females displayed on-screen.  
- Works on **CPU/GPU** with webcam input.  

---

## Upcoming Features
- Age estimation with Caffe age model.  
- Lone woman detection in unsafe conditions.  
- Threat/suspicious behavior alerts.  
- Heatmap mapping of high-risk areas.  
- Web Dashboard for remote monitoring.  

---

## Project Structure

Guardian_Vision/
│
├── guardian_vision.py # Main script
├── requirements.txt # Dependencies
├── README.md # Project documentation
│
├── data/
│ └── coco.names # COCO class names
│
├── models/
│ └── model_detection/
│ ├── age_deploy.prototxt # Age model config
│ ├── age_net.caffemodel # Age model weights
│ ├── gender_deploy.prototxt # Gender model config
│ ├── gender_net.caffemodel # Gender model weights
│ └── haarcascade_frontalface_default.xml # Haarcascade for face detection
│
└── weights/
└── yolov8n.pt # YOLOv8n lightweight model


## Tech Stack
- **Python 3.10+**  
- **OpenCV** (Haarcascade + DNN module)  
- **YOLOv8n (Ultralytics)** for person detection  
- **Caffe Models** for gender classification  
- **NumPy** for array processing  

---

## Run the Project
Clone the repo and install dependencies:
```bash
git clone https://github.com/MuhammadSakeen/Guardian-Vision.git
cd Guardian-Vision
pip install -r requirements.txt
```

Run Guardian Vision:
```bash
python guardian_vision.py
```

---

## Download Models
Due to large file sizes, models are **not included** in this repository.  
Please download them and place inside `models/model_detection/`.

- [YOLOv8n Weights](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt)  
- [Haarcascade Frontal Face](https://github.com/opencv/opencv/raw/master/data/haarcascadeshaarcascade_frontalface_default.xml)  
- [gender_deploy.prototxt](https://github.com/spmallick/learnopencv/raw/master/AgeGender/models/gender_deploy.prototxt)  
- [gender_net.caffemodel](https://github.com/spmallick/learnopencv/raw/master/AgeGender/models/gender_net.caffemodel)  

---

## Project Goal
To build a **smart AI surveillance system** that:  
- Detects unsafe conditions for women in real-time.  
- Sends **alerts** for potential threats or misuse.  
- Provides **data-driven insights** to law enforcement for **crime prevention**.  
