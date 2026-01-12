# 🚗 GuardianAI – Driver Monitoring System

GuardianAI is a real-time computer vision–based Driver Monitoring System (DMS) designed to enhance road safety by continuously analyzing driver behavior.  
It detects drowsiness and distraction using facial landmarks, provides instant visual alerts, and maintains a dynamic safety score to gamify safe driving.

---

## 🚀 Key Features

- Real-Time Drowsiness Detection  
  Uses Eye Aspect Ratio (EAR) to detect prolonged eye closure.

- Distraction Detection  
  Estimates 3D Head Pose (Pitch, Yaw, Roll) to detect when the driver looks away from the road.

- Dynamic Safety Score (0–100)  
  Penalizes unsafe behavior and regenerates during safe driving.

- Visual Dashboard  
  Modern on-screen UI showing safety score, head angles, alerts, and system status.

- Incident Logging  
  Automatically logs violations to a CSV file and saves image snapshots during critical events.

- Auto-Model Management  
  Automatically downloads the required MediaPipe face_landmarker.task model if missing.

---

## 🛠️ Tech Stack

- Python 3.8+  
- OpenCV (cv2) – Video capture and image processing  
- MediaPipe (Vision Tasks API) – Face landmark detection  
- NumPy – Vector math and EAR calculations  

---

## 📦 Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/GuardianAI.git
cd GuardianAI
```

### Install Dependencies
```bash
pip install opencv-python mediapipe numpy
```

---

## 📁 Project Structure

```
GuardianAI/
├── main.py                 # Entry point
├── config.py               # System settings
├── face_landmarker.task    # Downloaded automatically on first run
├── detectors/
│   ├── __init__.py
│   ├── drowsiness.py       # EAR logic
│   └── head_pose.py        # PnP & Euler angles logic
└── utils/
    ├── __init__.py
    └── logger.py           # Logging & Snapshot handling
```

---

## 🖥️ Usage

### Run the System
```bash
python main.py
```

Note: The first run may take a few seconds to download the AI model.

---

## 🎮 Controls

When the window is active:

- q : Quit the application  
- r : Reset the safety score to 100  
- s : Save a manual snapshot  
- c : Clear the log history  

---

## ⚙️ Configuration

You can fine-tune the system sensitivity in config.py.

Key parameters include:

- EAR_THRESHOLD = 0.21  
  EAR value below which eyes are considered closed.

- BLINK_CONSEC_FRAMES = 15  
  Number of consecutive frames eyes must be closed to trigger an alert.

- YAW_THRESHOLD = 60  
  Maximum head rotation (degrees) before distraction is triggered.

- SAFETY_SCORE = 100  
  Starting safety score. Drops on violations, regenerates on safe driving.

- CAMERA_ID = 0  
  Change this if using an external USB webcam.

---

## 🧠 How It Works

### 1. Drowsiness Detection (EAR)

The system tracks 6 facial landmarks per eye and computes the Eye Aspect Ratio (EAR) using the formula from Soukupová and Čech (2016):

EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2 × ||p1 - p4|| )

- Open Eye: EAR is stable (approximately 0.30)  
- Closed Eye: EAR falls rapidly toward 0.0  
- Alert: Triggered if EAR < 0.21 for 15 consecutive frames  

---

### 2. Head Pose Estimation (Distraction)

The system uses Perspective-n-Point (PnP) to map 2D facial landmarks to a generic 3D face model.

- Input: 2D image points (nose, chin, eyes, mouth)  
- Output: Rotation vector and translation vector  
- Result: Converted to Euler angles (pitch, yaw, roll)

Distraction Rule:
- If yaw angle exceeds 60°, the driver is considered distracted.

---

### 3. Safety Scoring System

- Penalties  
  - Drowsiness: −5 points  
  - Distraction: −3 points  

- Regeneration  
  - +0.5 points per frame of safe driving  

- Critical Threshold  
  - If score drops below 30, the UI turns red and snapshots are saved automatically.

---

## 📊 Outputs

- Visual Dashboard  
  Displays real-time safety score, head angles, alerts, and system status.

- Logs  
  All events are saved to logs.csv with timestamp, violation type, and score.

- Evidence  
  Image snapshots of violations are saved in the evidence/ directory.

---

## 📌 Use Cases

- Driver Monitoring Systems (DMS)  
- Fleet safety solutions  
- Autonomous vehicle research  
- Computer vision and AI safety projects  

---
Drive safe. GuardianAI is watching.
