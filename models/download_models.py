"""
Model Downloader Utility
Downloads official MediaPipe FaceLandmarker task and YOLOv8n models
"""

import sys
import urllib.request
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config

def download_face_landmarker():
    """Downloads the official MediaPipe FaceLandmarker float16 bundle."""
    target_path = config.FACE_LANDMARKER_MODEL
    if target_path.exists():
        print(f"[OK] FaceLandmarker model already present at: {target_path} ({target_path.stat().st_size} bytes)")
        return True

    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    print(f"Downloading Face Landmarker model from:\n{url}")
    try:
        urllib.request.urlretrieve(url, str(target_path))
        print(f"[OK] Downloaded Face Landmarker to: {target_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download Face Landmarker: {e}")
        return False

def download_yolo():
    """Downloads YOLOv8n pre-trained COCO weights."""
    target_path = config.YOLO_MODEL
    if target_path.exists():
        print(f"[OK] YOLOv8n model already present at: {target_path} ({target_path.stat().st_size} bytes)")
        return True

    print("Downloading YOLOv8n model weights via Ultralytics...")
    try:
        from ultralytics import YOLO
        import os
        original_cwd = os.getcwd()
        os.chdir(str(config.MODELS_DIR))
        model = YOLO("yolov8n.pt")
        os.chdir(original_cwd)
        print(f"[OK] Downloaded YOLOv8n to: {target_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download YOLOv8n: {e}")
        return False

def verify_all_models():
    """Downloads and verifies all models required by the system."""
    print("=" * 60)
    print("AI Road Safety System - Model Verification")
    print("=" * 60)
    face_ok = download_face_landmarker()
    yolo_ok = download_yolo()

    if face_ok and yolo_ok:
        print("\nAll model assets are ready!")
        return True
    else:
        print("\nSome models failed to download. Check network connection.")
        return False

if __name__ == "__main__":
    verify_all_models()
