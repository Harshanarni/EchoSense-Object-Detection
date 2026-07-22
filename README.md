# EchoSense: Real-Time Object Detection and Audio Feedback System

A real-time assistive system that helps visually impaired individuals understand their surroundings using object detection and voice feedback.

## Features
- Real-time object detection via webcam (default or external) using **YOLOv11**
- Photo and video upload with detection overlays
- Indian currency note recognition (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000)
- Real-time audio feedback using **Text-to-Speech (pyttsx3)**
- Manual and continuous audio feedback modes

## Tech Stack
- **Backend:** Python, Flask
- **Object Detection:** YOLOv11 (Ultralytics)
- **Computer Vision:** OpenCV
- **Text-to-Speech:** pyttsx3
- **Frontend:** HTML, CSS, JavaScript

## Setup

1. Clone the repository
```bash
git clone https://github.com/<your-username>/EchoSense-Object-Detection.git
cd EchoSense-Object-Detection
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add your YOLO model weights (`yolo11l.pt` and `best_s.pt`) to the project root — these are not included in this repo due to file size limits.

4. Run the app
```bash
python app.py
```

5. Open `http://127.0.0.1:5000` in your browser.

## Project Structure
```
EchoSense/
├── app.py
├── templates/
│   ├── index.html
│   ├── webcam.html
│   ├── externalweb.html
│   ├── money_detection.html
│   ├── play_video.html
│   └── show_photo.html
├── requirements.txt
└── README.md
```

## Team
- Pathuri Mahitha Sai
- Basina Divya Sri
- Chelluri Sai Krishna
- Narni Harsha Vardhan

**Guide:** Mr. G. Sridhar, Assistant Professor, Dept. of CSE
**Institution:** Ramachandra College of Engineering (Autonomous), Eluru

## Note
This project was developed as a B.Tech final year project (2024–2025).
