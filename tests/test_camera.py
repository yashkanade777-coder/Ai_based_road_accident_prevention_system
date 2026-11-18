"""
Module 1 Test Script: Camera Stream & FPS Verification
Runs standalone webcam feed with on-screen HUD telemetry.
"""

import sys
from pathlib import Path
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.camera_stream import CameraStream
import config

def run_camera_test():
    print("=" * 65)
    print("AI ROAD ACCIDENT PREVENTION SYSTEM - MODULE 1: CAMERA TEST")
    print("=" * 65)
    print("Controls:")
    print("  [Q] or [ESC] : Quit application")
    print("  [S]          : Toggle Synthetic Driver Simulator mode")
    print("  [C]          : Capture snapshot to data/snapshot.jpg")
    print("=" * 65)

    stream = CameraStream()

    window_name = "Module 1 Test: Live Camera Feed & Telemetry (Press 'Q' to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    snapshots_taken = 0

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                print("[ERROR] Failed to read frame.")
                break

            fps = stream.get_fps()
            status = "SIMULATION ACTIVE" if stream.is_synthetic else "LIVE WEBCAM ACTIVE"
            color = (0, 215, 255) if stream.is_synthetic else (0, 255, 0)

            # Draw HUD bar
            frame = CameraStream.overlay_telemetry(frame, fps, status_text=status, status_color=color)

            # Bottom instructions banner
            h, w = frame.shape[:2]
            cv2.putText(frame, "Press 'Q' to Exit | 'S' to Toggle Simulator | 'C' to Snapshot",
                        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]: # 27 is ESC
                print("[INFO] User requested exit.")
                break
            elif key in [ord('s'), ord('S')]:
                stream.is_synthetic = not stream.is_synthetic
                mode_str = "SYNTHETIC" if stream.is_synthetic else "LIVE HARDWARE"
                print(f"[INFO] Toggled stream mode to: {mode_str}")
            elif key in [ord('c'), ord('C')]:
                snapshots_taken += 1
                out_path = config.DATA_DIR / f"snapshot_{snapshots_taken}.jpg"
                cv2.imwrite(str(out_path), frame)
                print(f"[INFO] Saved snapshot to: {out_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (KeyboardInterrupt).")
    finally:
        stream.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera stream closed successfully.")

if __name__ == "__main__":
    run_camera_test()
