"""
Camera Stream Module
Handles real-time webcam capture on Windows with DSHOW backend, low latency buffer,
FPS calculation, and graceful synthetic fallback if camera is unavailable.
"""

import time
import cv2
import numpy as np
import config

class CameraStream:
    """
    Robust video capture manager for local webcams and simulated streams.
    """
    def __init__(self, camera_index=None, width=None, height=None, target_fps=None):
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self.width = width if width is not None else config.CAMERA_WIDTH
        self.height = height if height is not None else config.CAMERA_HEIGHT
        self.target_fps = target_fps if target_fps is not None else config.CAMERA_FPS

        self.cap = None
        self.is_synthetic = False
        self.frame_count = 0
        self.fps = 0.0
        self._prev_time = time.time()
        self._fps_history = []

        self._initialize_camera()

    def _initialize_camera(self):
        """
        Attempts to open the camera using Windows DirectShow (CAP_DSHOW) first,
        falling back to the standard backend, and finally to synthetic mode.
        """
        print(f"[CameraStream] Attempting to open camera index {self.camera_index} with CAP_DSHOW...")
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print("[CameraStream] DirectShow failed. Trying default capture backend...")
                self.cap = cv2.VideoCapture(self.camera_index)
        except Exception as e:
            print(f"[CameraStream] Camera open exception: {e}")
            self.cap = None

        if self.cap is not None and self.cap.isOpened():
            # Optimize camera properties for low-latency driver monitoring
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            # Minimize buffer size to 1 to prevent reading stale queued frames
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Test-read first frame
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[CameraStream] Camera ready. Resolution: {actual_w}x{actual_h}")
                self.is_synthetic = False
                return
            else:
                print("[CameraStream] Camera opened but failed to capture frame.")
                self.cap.release()

        # If we reach here, camera is unavailable -> activate synthetic fallback
        print("[CameraStream] [WARNING] Physical camera unavailable. Activating synthetic driver feed.")
        self.is_synthetic = True
        self.cap = None

    def read(self):
        """
        Reads the next frame.
        Returns: (success: bool, frame: np.ndarray)
        """
        current_time = time.time()
        dt = current_time - self._prev_time
        self._prev_time = current_time

        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps_history.append(instant_fps)
            if len(self._fps_history) > 15:
                self._fps_history.pop(0)
            self.fps = sum(self._fps_history) / len(self._fps_history)

        self.frame_count += 1

        if not self.is_synthetic and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Mirror frame horizontally so driver sees natural reflection
                frame = cv2.flip(frame, 1)
                return True, frame

        # Synthetic fallback frame generator
        return True, self._generate_synthetic_frame()

    def _generate_synthetic_frame(self):
        """
        Generates a synthetic diagnostic frame when physical camera is unavailable.
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Gradient dark blue / gray background
        for y in range(self.height):
            val = int(20 + 40 * (y / self.height))
            frame[y, :] = [val + 20, val + 10, val]

        # Draw a simulation vehicle dashboard boundary
        cv2.rectangle(frame, (30, 30), (self.width - 30, self.height - 30), (80, 80, 80), 2)

        # Draw driver avatar silhouette
        center_x = self.width // 2
        center_y = self.height // 2 + 30
        cv2.circle(frame, (center_x, center_y - 60), 50, (140, 140, 140), -1) # Head
        cv2.ellipse(frame, (center_x, center_y + 110), (110, 80), 0, 0, 180, (120, 120, 120), -1) # Shoulders

        # Add simulated blinking indicator
        blink = (self.frame_count // 30) % 2 == 0
        eye_color = (40, 40, 40) if blink else (240, 240, 240)
        cv2.circle(frame, (center_x - 18, center_y - 65), 6, eye_color, -1)
        cv2.circle(frame, (center_x + 18, center_y - 65), 6, eye_color, -1)

        # Text banner
        cv2.putText(frame, "SYNTHETIC DRIVER SIMULATOR", (center_x - 170, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "No physical webcam detected or camera in use.", (center_x - 190, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        return frame

    def get_fps(self):
        """Returns the smoothed real-time frames per second."""
        return round(self.fps, 1)

    def is_opened(self):
        """True if camera or synthetic simulator is operational."""
        return self.is_synthetic or (self.cap is not None and self.cap.isOpened())

    def release(self):
        """Releases camera hardware resources cleanly."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("[CameraStream] Camera released.")
        self.cap = None

    @staticmethod
    def overlay_telemetry(frame, fps, status_text="MONITORING ACTIVE", status_color=(0, 255, 0)):
        """
        Draws professional on-screen telemetry bar on the camera frame.
        """
        h, w = frame.shape[:2]

        # Top dark HUD bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Status indicator circle
        cv2.circle(frame, (20, 21), 7, status_color, -1)
        cv2.putText(frame, status_text, (35, 27),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS counter & time
        fps_text = f"FPS: {fps:.1f}"
        timestamp_text = time.strftime("%H:%M:%S")
        hud_right = f"{fps_text}  |  {timestamp_text}"
        text_size = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
        cv2.putText(frame, hud_right, (w - text_size[0] - 15, 27),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return frame
