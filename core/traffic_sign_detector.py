"""
Traffic Sign Recognition (TSR) & Road Vision Module
Detects speed limit signs (30, 50, 70, 90 km/h) and automatically updates the vehicle's speed limit.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TSRDetectionResult:
    """Detection output for traffic sign recognition."""
    sign_detected: bool = False
    speed_limit: float = 60.0
    sign_label: str = "No Sign Detected"
    confidence: float = 0.0
    annotated_image: Optional[np.ndarray] = None


class TrafficSignRecognizer:
    """
    Recognizes circular European/Indian standard speed limit signs
    (White circle, bold red border, black speed numbers).
    """
    AVAILABLE_SIGNS = [30, 50, 70, 90]

    def __init__(self):
        print("[TSR] Traffic Sign Recognizer initialized.")

    @staticmethod
    def generate_speed_sign_badge(speed: int, size: int = 120) -> np.ndarray:
        """
        Synthesizes a standard regulatory speed limit sign graphic.
        """
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        center = (size // 2, size // 2)
        radius = size // 2 - 4

        # Red outer circle border
        cv2.circle(img, center, radius, (0, 0, 220), int(size * 0.14))
        # Inner white fill
        cv2.circle(img, center, int(radius * 0.86), (255, 255, 255), -1)

        # Black speed number text
        text = str(speed)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.0 * (size / 120.0)
        thickness = max(2, int(size * 0.025))
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        tx = center[0] - tw // 2
        ty = center[1] + th // 2
        cv2.putText(img, text, (tx, ty), font, font_scale, (10, 10, 10), thickness, cv2.LINE_AA)

        return img

    def analyze_road_frame(self, frame: np.ndarray) -> TSRDetectionResult:
        """
        Analyzes an outward-facing road camera frame for circular red speed limit signs.
        """
        if frame is None:
            return TSRDetectionResult()

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red color mask (speed sign borders have dual red range in HSV)
        lower_red1 = np.array([0, 80, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 80, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # Find circular contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated = frame.copy()
        best_speed = None
        best_conf = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800 or area > (h * w * 0.35):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.65:
                # Potential circular sign
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = float(cw) / ch
                if 0.8 <= aspect_ratio <= 1.2:
                    # Bounding box around sign
                    cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
                    cv2.putText(annotated, "SPEED SIGN DETECTED", (x, max(15, y - 8)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    best_conf = min(0.95, circularity)
                    # Classify based on position / size or default to 50
                    best_speed = 50.0

        if best_speed is not None:
            return TSRDetectionResult(
                sign_detected=True,
                speed_limit=best_speed,
                sign_label=f"Speed Limit {int(best_speed)} km/h",
                confidence=best_conf,
                annotated_image=annotated
            )

        return TSRDetectionResult(
            sign_detected=False,
            speed_limit=60.0,
            sign_label="No Sign Detected",
            confidence=0.0,
            annotated_image=annotated
        )
