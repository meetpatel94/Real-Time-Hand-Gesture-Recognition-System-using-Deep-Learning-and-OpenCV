# Real-Time Hand Gesture Recognition System using Deep Learning and OpenCV

## Project Overview

This project implements a **Real-Time Hand Gesture Recognition System** using **Deep Learning**, **OpenCV**, and **Python**. The system captures hand gestures from a webcam, preprocesses the hand region, and classifies gestures using a **Convolutional Neural Network (CNN)** model.

The system aims to improve communication and enable touchless interaction, especially for sign language users and gesture-controlled applications.

---

## Features

✔ Real-time hand gesture recognition using webcam
✔ CNN-based gesture classification
✔ Image preprocessing using OpenCV
✔ Background subtraction and contour extraction
✔ Custom dataset creation
✔ Live prediction system
✔ User-friendly and lightweight implementation
✔ Supports different lighting conditions

---

## Applications

- Sign Language Recognition
- Human Computer Interaction (HCI)
- Smart Home Automation
- Gesture-Controlled Robotics
- Touchless Control Systems
- Assistive Technologies

---

## Tech Stack

------------------------------------------
| Technology   | Purpose                 |
|--------------|-------------------------|
| Python       | Programming Language    |
| OpenCV       | Image Processing        |
| TensorFlow   | Deep Learning Framework |
| Keras        | CNN Model Building      |
| NumPy        | Numerical Operations    |
| Matplotlib   | Visualization           |
| Scikit-learn | Data Processing         |
------------------------------------------
---

# Project Structure

```bash
Real-Time-Hand-Gesture-Recognition/
│
├── dataset/
│   ├── A/
│   ├── B/
│   ├── C/
│   └── ...
│
├── model/
│   ├── gesture_model.h5
│
├── screenshots/
│   ├── output1.png
│   ├── output2.png
│
├── create_gesture_data.py
├── trainCNN.py
├── predict_live.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation Guide

## Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/Real-Time-Hand-Gesture-Recognition.git
```

Move inside folder:

```bash
cd Real-Time-Hand-Gesture-Recognition
```

---

## Step 2: Create Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate virtual environment:

CMD:

```bash
venv\Scripts\activate
```

PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

Linux/Mac:

```bash
source venv/bin/activate
```

---

## Step 3: Install Required Packages

Install all packages:

```bash
pip install -r requirements.txt
```

OR manually:

```bash
pip install tensorflow==2.13.0
pip install keras==2.13.1
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install matplotlib==3.7.2
pip install scikit-learn==1.3.0
pip install pillow==10.0.0
```

Verify installation:

```bash
pip list
```

---

# Python Version

Recommended:

```bash
Python 3.10
```

Check version:

```bash
python --version
```

---

# Dataset Creation

The system uses a custom dataset captured from webcam.

Run:

```bash
python create_gesture_data.py
```

### Instructions:

1. Webcam opens automatically
2. Place hand inside ROI box
3. Press:

```text
A → Save gesture A images
B → Save gesture B images
C → Save gesture C images
```

4. Capture multiple samples

Recommended:

```text
200–500 images per gesture
```

Dataset folder structure:

```bash
dataset/

├── A/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...

├── B/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...

├── C/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
```

---

# Training CNN Model

After creating dataset:

Run:

```bash
python trainCNN.py
```

Training process:

- Load dataset
- Resize images
- Normalize images
- Split train/test data
- Train CNN model
- Save trained model

Output:

```bash
gesture_model.h5
```

---

# Run Real-Time Prediction

After model training:

```bash
python predict_live.py
```

Steps:

1. Webcam opens
2. Put hand in ROI area
3. System preprocesses frame
4. CNN predicts gesture
5. Prediction displayed on screen

Example:

```text
Prediction : HELLO

Prediction : THANK YOU

Prediction : OK
```

---

# CNN Architecture

Model consists of:

Input Layer

↓

Convolution Layer + ReLU

↓

MaxPooling Layer

↓

Convolution Layer + ReLU

↓

MaxPooling Layer

↓

Flatten Layer

↓

Dense Layer

↓

Output Layer

---

# Performance Metrics

Metrics used:

- Accuracy
- Precision
- Recall
- F1 Score

Expected Performance:

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 88–95% |
| Precision | 86–92% |
| Recall    | 85–91% |
| F1 Score  | 85–90% |

Performance depends on:

- Dataset size
- Lighting condition
- Hand position
- Background quality

---

# Future Improvements

- Dynamic gesture recognition using LSTM
- Audio output generation
- Larger sign vocabulary
- Mobile application support
- Edge deployment
- MediaPipe integration
- Continuous sentence recognition

---

# Troubleshooting

### Webcam not opening

Check camera access:

Windows:

Settings

→ Privacy
→ Camera
→ Enable Camera Access

---

### Module Not Found Error

Install missing packages:

```bash
pip install package_name
```

Example:

```bash
pip install opencv-python
```

---

### TensorFlow installation issue

Use:

```bash
pip install tensorflow --upgrade
```

---

### Low prediction accuracy

Solutions:

- Increase dataset size

- Use proper lighting

- Remove noisy backgrounds

- Capture multiple hand angles

---

# Output

## Training

![Training](screenshots/output1.png)

## Prediction

![Prediction](screenshots/output2.png)

---

# Author

Meet Patel

Master of Engineering (Computer Engineering - AI & DS)

Gujarat Technological University

---

# License

This project is developed for educational and research purposes.
