# AI-Based Driver Drowsiness Detection Mini Project

This project detects driver drowsiness using a webcam.  
When the user's eyes remain closed for a few continuous frames, the system plays an alarm sound.

## Features

- Real-time webcam monitoring
- Eye Aspect Ratio (EAR) based drowsiness detection
- Alarm sound when drowsiness is detected
- Simple Python mini-project structure
- No TensorFlow required

## Project Folder Structure

```text
ai_drowsiness_detection/
│
├── main.py
├── requirements.txt
├── README.md
│
├── assets/
│   └── alarm.wav
│
└── src/
    ├── __init__.py
    ├── detector.py
    ├── alarm.py
    └── utils.py
```

## Requirements

- Python 3.10 or 3.11 recommended
- Webcam
- Windows / Linux / macOS

## Step-by-Step Setup

### 1. Create project folder

Example path on Windows:

```bash
cd C:\Users\sudarshan\OneDrive\Desktop
mkdir ai_drowsiness_detection
cd ai_drowsiness_detection
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

### 3. Activate virtual environment

For Windows PowerShell:

```bash
.venv\Scripts\activate
```

For CMD:

```bash
.venv\Scripts\activate.bat
```

### 4. Install required packages

```bash
pip install -r requirements.txt
```

### 5. Run the project

```bash
python main.py
```

## Controls

- Press `q` to quit the camera window.

## How It Works

1. Webcam captures live video.
2. MediaPipe detects face landmarks.
3. Eye landmarks are extracted.
4. Eye Aspect Ratio is calculated.
5. If EAR is below threshold for continuous frames, drowsiness is detected.
6. Alarm sound is played.

## Important Settings

Inside `main.py`:

```python
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 20
```

If alarm starts too quickly, increase `CONSECUTIVE_FRAMES`.

If eyes are not detected properly, adjust `EAR_THRESHOLD` between `0.20` and `0.30`.

## Output

Normal state:

```text
Status: ACTIVE
```

Drowsy state:

```text
Status: DROWSINESS ALERT!
```

## Mini Project Title

AI-Based Driver Drowsiness Detection System Using Computer Vision
