"""
System Configuration & Constants
AI-Based Road Accident Prevention and Traffic Management System
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "accident_prevention.db"

# Create directories if they do not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model Paths
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
YOLO_MODEL = MODELS_DIR / "yolov8n.pt"

# Camera Settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# DMS Thresholds (Driver Monitoring System)
# EAR: Eye Aspect Ratio (< threshold indicates eye closure)
EAR_THRESHOLD = 0.22
# Number of consecutive frames eyes must remain closed to trigger drowsiness alert
DROWSINESS_CONSECUTIVE_FRAMES = 24  # approx 1.0 - 1.2 seconds at 25-30 FPS

# MAR: Mouth Aspect Ratio (> threshold indicates open mouth / yawning)
MAR_THRESHOLD = 0.65
YAWN_CONSECUTIVE_FRAMES = 20  # approx 0.8 - 1.0 second

# Head Pose Distraction (Yaw / Pitch angles in degrees)
HEAD_YAW_THRESHOLD = 25.0    # Looking left or right away from road
HEAD_PITCH_DOWN_THRESHOLD = 20.0  # Looking down (e.g. at lap / mobile)
DISTRACTION_CONSECUTIVE_FRAMES = 30  # approx 1.2 - 1.5 seconds

# Phone Detection
PHONE_CLASS_ID = 67  # COCO dataset class index for 'cell phone'
PHONE_CONFIDENCE_THRESHOLD = 0.40

# Audio Alert Cooldowns (in seconds to prevent repetitive spam)
ALERT_COOLDOWNS = {
    "DROWSINESS": 5.0,
    "DISTRACTION": 4.0,
    "YAWNING": 8.0,
    "PHONE": 4.0,
    "OVERSPEED": 6.0,
    "HIGH_RISK": 8.0
}

# GPS Speed Settings (in km/h)
DEFAULT_SPEED_LIMIT = 60.0
OVERSPEED_SUSTAINED_SECONDS = 3.0  # Alert after exceeding limit for this duration
SIMULATED_GPS_DEFAULT_MODE = "CITY"  # Options: "CITY", "HIGHWAY", "OVERSPEED_TEST"
