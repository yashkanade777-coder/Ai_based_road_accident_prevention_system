"""
Mobile Phone & Object Detection Module
Uses YOLOv8n to detect handheld mobile phone usage while driving.
Optimized with asynchronous background inference to preserve 30 FPS camera performance.
"""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

import config

@dataclass
class PhoneDetection:
    """Bounding box and score for a detected phone."""
    box: tuple = (0, 0, 0, 0)          # (x1, y1, x2, y2)
    confidence: float = 0.0
    label: str = "cell phone"

@dataclass
class ObjectDetectorResult:
    """Summary of object detection for current frame."""
    phone_detected: bool = False
    detections: list = field(default_factory=list)
    confidence: float = 0.0
    inference_time_ms: float = 0.0


class PhoneDetector:
    """
    Lightweight, thread-safe mobile phone detector using YOLOv8n.
    Runs asynchronously in a background worker so driver monitoring stays at full 30 FPS.
    """
    def __init__(self, model_path=None, conf_threshold=None):
        self.model_path = str(model_path or config.YOLO_MODEL)
        self.conf_threshold = conf_threshold if conf_threshold is not None else config.PHONE_CONFIDENCE_THRESHOLD
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"YOLO model not found at {self.model_path}. "
                f"Run 'python models/download_models.py' first."
            )

        print("[PhoneDetector] Loading YOLOv8n model...")
        self.model = YOLO(self.model_path)
        
        # Warm up on dummy input
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy, verbose=False, imgsz=320, classes=[config.PHONE_CLASS_ID])
        print("[PhoneDetector] YOLOv8n initialized and warmed up.")

        # Threading state for async execution
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_result = ObjectDetectorResult()
        self._running = True
        self._consecutive_hits = 0

        # Start background worker thread
        self._worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker_thread.start()

    def update_frame(self, frame: np.ndarray):
        """Passes the latest camera frame to the background detector."""
        with self._lock:
            self._latest_frame = frame.copy()

    def get_result(self) -> ObjectDetectorResult:
        """Returns the most recent phone detection result without blocking."""
        with self._lock:
            return self._latest_result

    def _inference_loop(self):
        """Background thread continuously running YOLO on the newest available frame."""
        while self._running:
            frame_to_process = None
            with self._lock:
                if self._latest_frame is not None:
                    frame_to_process = self._latest_frame
                    self._latest_frame = None  # Consume frame

            if frame_to_process is None:
                time.sleep(0.02)
                continue

            t0 = time.time()
            h, w = frame_to_process.shape[:2]

            try:
                # Run inference targeting only class 67 (cell phone)
                results = self.model(
                    frame_to_process,
                    conf=self.conf_threshold,
                    classes=[config.PHONE_CLASS_ID],
                    imgsz=320,
                    verbose=False
                )

                detections = []
                max_conf = 0.0

                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        if conf >= self.conf_threshold:
                            detections.append(PhoneDetection(
                                box=(int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])),
                                confidence=conf,
                                label="Cell Phone"
                            ))
                            if conf > max_conf:
                                max_conf = conf

                dt = (time.time() - t0) * 1000.0

                # Require 2 consecutive hits to avoid single-frame flickering
                if len(detections) > 0:
                    self._consecutive_hits += 1
                else:
                    self._consecutive_hits = max(0, self._consecutive_hits - 1)

                phone_active = self._consecutive_hits >= 2

                res = ObjectDetectorResult(
                    phone_detected=phone_active,
                    detections=detections,
                    confidence=max_conf,
                    inference_time_ms=dt
                )

                with self._lock:
                    self._latest_result = res

            except Exception as e:
                print(f"[PhoneDetector] Inference error: {e}")

            # Small yield to prevent 100% CPU utilization
            time.sleep(0.01)

    def draw_detections(self, frame: np.ndarray, result: ObjectDetectorResult) -> np.ndarray:
        """Draws glowing bounding boxes and warnings around detected mobile phones."""
        if not result.phone_detected and len(result.detections) == 0:
            return frame

        for det in result.detections:
            x1, y1, x2, y2 = det.box
            
            # Draw primary red bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Corner accents for a modern look
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            color = (0, 0, 255)
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)

            # Label badge
            label_text = f"PHONE: {det.confidence * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + tw + 10, y1), (0, 0, 255), -1)
            cv2.putText(frame, label_text, (x1 + 5, y1 - 7),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def close(self):
        """Stops the background worker thread cleanly."""
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        print("[PhoneDetector] Closed.")
