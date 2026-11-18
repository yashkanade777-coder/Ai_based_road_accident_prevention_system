"""
Driver Monitoring System (DMS) Core Detector
Uses MediaPipe Face Landmarker with 478 3D landmarks & facial blendshapes to detect:
1. Drowsiness & Prolonged Eye Closure (EAR + Blink Blendshapes)
2. Frequent Yawning (MAR + JawOpen Blendshapes)
3. Distraction / Looking Away (3D Metric Head Pose Matrix: Yaw, Pitch, Roll)
4. Experimental Behavioral Stress Signals (Brow Tension & Blink Cadence)
"""

import time
import math
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config

@dataclass
class DMSResult:
    """Structured telemetry output from driver monitoring."""
    face_detected: bool = False
    ear: float = 0.0                     # Eye Aspect Ratio (0.0 - 0.45)
    blink_score: float = 0.0             # Blendshape blink probability (0.0 - 1.0)
    mar: float = 0.0                     # Mouth Aspect Ratio (0.0 - 1.0)
    jaw_open_score: float = 0.0          # Blendshape jaw open probability (0.0 - 1.0)
    pitch: float = 0.0                   # Head pitch: +up, -down (degrees)
    yaw: float = 0.0                     # Head yaw: +right, -left (degrees)
    roll: float = 0.0                    # Head tilt (degrees)
    
    # State flags
    is_drowsy: bool = False
    drowsy_frames: int = 0
    is_yawning: bool = False
    yawn_frames: int = 0
    is_distracted: bool = False
    distraction_frames: int = 0
    distraction_reason: str = "NONE"     # "LOOKING_LEFT", "LOOKING_RIGHT", "LOOKING_DOWN", "NONE"
    
    # Experimental Stress / Workload Cue
    stress_indicator: str = "NORMAL"     # "NORMAL", "MILD_TENSION", "ELEVATED_LOAD"
    brow_tension: float = 0.0
    
    # Overall summary
    status_level: str = "NORMAL"         # "NORMAL", "CAUTION", "CRITICAL"
    primary_alert: str = "DRIVER ATTENTIVE"
    active_alerts: list = field(default_factory=list)


class DMSDetector:
    """
    High-performance real-time Driver Monitoring System using MediaPipe FaceLandmarker.
    """
    # Key Landmark Indices in the 478 MediaPipe Mesh
    LEFT_EYE_CORNER_L = 33
    LEFT_EYE_CORNER_R = 133
    LEFT_EYE_TOP1 = 160
    LEFT_EYE_TOP2 = 158
    LEFT_EYE_BOT1 = 144
    LEFT_EYE_BOT2 = 153
    LEFT_IRIS = 468

    RIGHT_EYE_CORNER_L = 362
    RIGHT_EYE_CORNER_R = 263
    RIGHT_EYE_TOP1 = 385
    RIGHT_EYE_TOP2 = 387
    RIGHT_EYE_BOT1 = 380
    RIGHT_EYE_BOT2 = 373
    RIGHT_IRIS = 473

    MOUTH_CORNER_L = 61
    MOUTH_CORNER_R = 291
    MOUTH_TOP = 13
    MOUTH_BOT = 14

    NOSE_TIP = 1

    def __init__(self, model_path=None):
        self.model_path = str(model_path or config.FACE_LANDMARKER_MODEL)
        
        # Verify model file exists
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found at: {self.model_path}. "
                f"Run 'python models/download_models.py' first."
            )

        print("[DMSDetector] Initializing MediaPipe FaceLandmarker engine...")
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Internal state counters for sustained alerts
        self._drowsy_counter = 0
        self._yawn_counter = 0
        self._distraction_counter = 0

        # Blink rate tracking (for experimental stress cue)
        self._blink_timestamps = []
        self._last_eye_closed = False

        print("[DMSDetector] Engine ready.")

    def process(self, frame: np.ndarray) -> DMSResult:
        """
        Analyzes a single BGR camera frame and returns a DMSResult telemetry object.
        """
        result = DMSResult()
        h, w = frame.shape[:2]

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run inference
        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            result.face_detected = False
            result.status_level = "CAUTION"
            result.primary_alert = "NO FACE DETECTED"
            result.active_alerts.append("NO_FACE")
            self._drowsy_counter = 0
            self._distraction_counter = 0
            self._yawn_counter = 0
            return result

        result.face_detected = True
        landmarks = detection_result.face_landmarks[0]

        # Helper to convert normalized landmark to pixel numpy array
        def pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float32)

        # -------------------------------------------------------------
        # 1. Calculate Eye Aspect Ratio (EAR)
        # -------------------------------------------------------------
        # Left Eye
        p1, p4 = pt(self.LEFT_EYE_CORNER_L), pt(self.LEFT_EYE_CORNER_R)
        p2, p6 = pt(self.LEFT_EYE_TOP1), pt(self.LEFT_EYE_BOT1)
        p3, p5 = pt(self.LEFT_EYE_TOP2), pt(self.LEFT_EYE_BOT2)
        ear_left = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * (np.linalg.norm(p1 - p4) + 1e-6))

        # Right Eye
        rp1, rp4 = pt(self.RIGHT_EYE_CORNER_L), pt(self.RIGHT_EYE_CORNER_R)
        rp2, rp6 = pt(self.RIGHT_EYE_TOP1), pt(self.RIGHT_EYE_BOT1)
        rp3, rp5 = pt(self.RIGHT_EYE_TOP2), pt(self.RIGHT_EYE_BOT2)
        ear_right = (np.linalg.norm(rp2 - rp6) + np.linalg.norm(rp3 - rp5)) / (2.0 * (np.linalg.norm(rp1 - rp4) + 1e-6))

        result.ear = float((ear_left + ear_right) / 2.0)

        # -------------------------------------------------------------
        # 2. Extract Blendshapes (Blink, JawOpen, Brow Tension)
        # -------------------------------------------------------------
        bs_dict = {}
        if detection_result.face_blendshapes:
            bs_dict = {b.category_name: b.score for b in detection_result.face_blendshapes[0]}

        blink_l = bs_dict.get('eyeBlinkLeft', 0.0)
        blink_r = bs_dict.get('eyeBlinkRight', 0.0)
        result.blink_score = float((blink_l + blink_r) / 2.0)
        result.jaw_open_score = float(bs_dict.get('jawOpen', 0.0))

        # Eyebrow tension (experimental stress signal)
        brow_down = (bs_dict.get('browDownLeft', 0.0) + bs_dict.get('browDownRight', 0.0)) / 2.0
        result.brow_tension = float(brow_down)

        # -------------------------------------------------------------
        # 3. Calculate Mouth Aspect Ratio (MAR)
        # -------------------------------------------------------------
        m_left, m_right = pt(self.MOUTH_CORNER_L), pt(self.MOUTH_CORNER_R)
        m_top, m_bot = pt(self.MOUTH_TOP), pt(self.MOUTH_BOT)
        result.mar = float(np.linalg.norm(m_top - m_bot) / (np.linalg.norm(m_left - m_right) + 1e-6))

        # -------------------------------------------------------------
        # 4. Metric Head Pose (Yaw, Pitch, Roll)
        # -------------------------------------------------------------
        if detection_result.facial_transformation_matrixes:
            mat = np.array(detection_result.facial_transformation_matrixes[0])
            R = mat[:3, :3]
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            if sy > 1e-6:
                pitch_rad = math.atan2(R[2, 1], R[2, 2])
                yaw_rad = math.atan2(-R[2, 0], sy)
                roll_rad = math.atan2(R[1, 0], R[0, 0])
            else:
                pitch_rad = math.atan2(-R[1, 2], R[1, 1])
                yaw_rad = math.atan2(-R[2, 0], sy)
                roll_rad = 0.0

            result.pitch = float(math.degrees(pitch_rad))
            result.yaw = float(math.degrees(yaw_rad))
            result.roll = float(math.degrees(roll_rad))

        # -------------------------------------------------------------
        # 5. Drowsiness & Prolonged Eye Closure Detection
        # -------------------------------------------------------------
        # Eye closure is triggered if geometric EAR is low OR blendshape blink is high
        eye_closed = (result.ear < config.EAR_THRESHOLD) or (result.blink_score > 0.65)
        
        # Track blink timestamp for blink frequency
        current_time = time.time()
        if eye_closed and not self._last_eye_closed:
            self._blink_timestamps.append(current_time)
        self._last_eye_closed = eye_closed

        # Prune blinks older than 60 seconds
        self._blink_timestamps = [t for t in self._blink_timestamps if current_time - t <= 60.0]

        if eye_closed:
            self._drowsy_counter += 1
        else:
            self._drowsy_counter = max(0, self._drowsy_counter - 2)

        result.drowsy_frames = self._drowsy_counter
        if self._drowsy_counter >= config.DROWSINESS_CONSECUTIVE_FRAMES:
            result.is_drowsy = True
            result.active_alerts.append("DROWSINESS")

        # -------------------------------------------------------------
        # 6. Yawning Detection
        # -------------------------------------------------------------
        yawning = (result.mar > config.MAR_THRESHOLD) or (result.jaw_open_score > 0.60)
        if yawning:
            self._yawn_counter += 1
        else:
            self._yawn_counter = max(0, self._yawn_counter - 1)

        result.yawn_frames = self._yawn_counter
        if self._yawn_counter >= config.YAWN_CONSECUTIVE_FRAMES:
            result.is_yawning = True
            result.active_alerts.append("YAWNING")

        # -------------------------------------------------------------
        # 7. Distraction Detection (Head Pose)
        # -------------------------------------------------------------
        looking_left = result.yaw < -config.HEAD_YAW_THRESHOLD
        looking_right = result.yaw > config.HEAD_YAW_THRESHOLD
        looking_down = result.pitch < -config.HEAD_PITCH_DOWN_THRESHOLD

        distracted = looking_left or looking_right or looking_down
        if distracted:
            self._distraction_counter += 1
            if looking_left:
                result.distraction_reason = "LOOKING_LEFT"
            elif looking_right:
                result.distraction_reason = "LOOKING_RIGHT"
            elif looking_down:
                result.distraction_reason = "LOOKING_DOWN"
        else:
            self._distraction_counter = max(0, self._distraction_counter - 1)
            result.distraction_reason = "NONE"

        result.distraction_frames = self._distraction_counter
        if self._distraction_counter >= config.DISTRACTION_CONSECUTIVE_FRAMES:
            result.is_distracted = True
            result.active_alerts.append("DISTRACTION")

        # -------------------------------------------------------------
        # 8. Experimental Stress / Cognitive Workload Signal
        # -------------------------------------------------------------
        # Note: Non-diagnostic indicator based on blink cadence & brow furrow
        blink_rate = len(self._blink_timestamps)  # blinks in last 60s
        if result.brow_tension > 0.55 or blink_rate > 32:
            result.stress_indicator = "ELEVATED_LOAD"
        elif result.brow_tension > 0.35 or blink_rate > 24:
            result.stress_indicator = "MILD_TENSION"
        else:
            result.stress_indicator = "NORMAL"

        # -------------------------------------------------------------
        # 9. Priority Level & Primary Alert Resolution
        # -------------------------------------------------------------
        if result.is_drowsy:
            result.status_level = "CRITICAL"
            result.primary_alert = "DROWSINESS DETECTED! WAKE UP!"
        elif result.is_distracted:
            result.status_level = "CRITICAL"
            reason = result.distraction_reason.replace("_", " ")
            result.primary_alert = f"DISTRACTION ({reason})! KEEP EYES ON ROAD"
        elif result.is_yawning:
            result.status_level = "CAUTION"
            result.primary_alert = "FREQUENT YAWNING: FATIGUE WARNING"
        elif result.stress_indicator == "ELEVATED_LOAD":
            result.status_level = "CAUTION"
            result.primary_alert = "HIGH FACIAL TENSION DETECTED"
        else:
            result.status_level = "NORMAL"
            result.primary_alert = "DRIVER ATTENTIVE & FOCUSED"

        return result

    def draw_dms_hud(self, frame: np.ndarray, result: DMSResult, landmarks=None) -> np.ndarray:
        """
        Draws driver monitoring visualization overlays, facial landmarks, gauges,
        and warning alerts on the frame.
        """
        h, w = frame.shape[:2]

        # 1. Colors based on status level
        if result.status_level == "CRITICAL":
            theme_color = (0, 0, 255)     # Red
        elif result.status_level == "CAUTION":
            theme_color = (0, 215, 255)   # Amber / Yellow
        else:
            theme_color = (0, 255, 0)     # Green

        # 2. Main Alert Banner (Top of screen)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 42), (w, 95), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        # Alert text with glow / badge
        cv2.putText(frame, result.primary_alert, (30, 78),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, theme_color, 2, cv2.LINE_AA)

        # 3. Telemetry Sidebar / Dashboard HUD (Left side)
        bar_x = 20
        start_y = 120
        box_w = 210
        box_h = 160

        # Semi-transparent HUD card
        sub_overlay = frame.copy()
        cv2.rectangle(sub_overlay, (bar_x - 10, start_y - 20), (bar_x + box_w, start_y + box_h), (20, 20, 20), -1)
        cv2.rectangle(sub_overlay, (bar_x - 10, start_y - 20), (bar_x + box_w, start_y + box_h), theme_color, 1)
        cv2.addWeighted(sub_overlay, 0.75, frame, 0.25, 0, frame)

        # EAR Meter
        cv2.putText(frame, f"EAR: {result.ear:.2f}", (bar_x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        ear_bar_w = int(np.clip(result.ear / 0.40, 0, 1) * 100)
        cv2.rectangle(frame, (bar_x + 85, start_y - 10), (bar_x + 185, start_y - 2), (50, 50, 50), -1)
        ear_color = (0, 255, 0) if result.ear >= config.EAR_THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (bar_x + 85, start_y - 10), (bar_x + 85 + ear_bar_w, start_y - 2), ear_color, -1)

        # MAR Meter
        start_y += 28
        cv2.putText(frame, f"MAR: {result.mar:.2f}", (bar_x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        mar_bar_w = int(np.clip(result.mar / 1.0, 0, 1) * 100)
        cv2.rectangle(frame, (bar_x + 85, start_y - 10), (bar_x + 185, start_y - 2), (50, 50, 50), -1)
        mar_color = (0, 0, 255) if result.mar > config.MAR_THRESHOLD else (0, 255, 0)
        cv2.rectangle(frame, (bar_x + 85, start_y - 10), (bar_x + 85 + mar_bar_w, start_y - 2), mar_color, -1)

        # Head Pose (Yaw / Pitch)
        start_y += 28
        cv2.putText(frame, f"Yaw: {result.yaw:+.1f} deg", (bar_x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        start_y += 24
        cv2.putText(frame, f"Pitch: {result.pitch:+.1f} deg", (bar_x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # Stress / Workload Cue (Experimental)
        start_y += 26
        stress_color = (0, 255, 0) if result.stress_indicator == "NORMAL" else ((0, 215, 255) if result.stress_indicator == "MILD_TENSION" else (0, 0, 255))
        cv2.putText(frame, f"Load*: {result.stress_indicator}", (bar_x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, stress_color, 1, cv2.LINE_AA)

        # 4. 3D Head Direction Vector Pointer (Center)
        if result.face_detected:
            center_x, center_y = w // 2, h // 2
            # Project vector based on Yaw and Pitch
            rad_yaw = math.radians(result.yaw)
            rad_pitch = math.radians(result.pitch)
            vec_length = 60
            end_x = int(center_x + vec_length * math.sin(rad_yaw))
            end_y = int(center_y - vec_length * math.sin(rad_pitch))
            
            # Compass crosshair
            cv2.circle(frame, (center_x, center_y), 4, (120, 120, 120), -1)
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), theme_color, 2, tipLength=0.25)

        # Disclaimer at very bottom right
        cv2.putText(frame, "*Experimental Driver Behavior Model", (w - 270, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

        return frame
