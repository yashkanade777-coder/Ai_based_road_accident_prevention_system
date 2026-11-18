"""
Module 2 Test Script: Driver Monitoring System (DMS)
Runs real-time Drowsiness, Yawning, and Distraction Detection with on-screen HUD.
"""

import sys
from pathlib import Path
import time
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.camera_stream import CameraStream
from core.dms_detector import DMSDetector
import config

def run_dms_test():
    print("=" * 70)
    print("AI ROAD ACCIDENT PREVENTION SYSTEM - MODULE 2: DMS TEST")
    print("=" * 70)
    print("Test Instructions:")
    print("  1. Drowsiness Test  : Close your eyes for ~1.5 seconds.")
    print("  2. Yawn Test        : Open your mouth wide for ~1 second.")
    print("  3. Distraction Test : Turn your head left/right or look down.")
    print("  4. Normal State     : Look straight ahead at the camera.")
    print("----------------------------------------------------------------------")
    print("Controls:")
    print("  [Q] or [ESC] : Quit application")
    print("  [C]          : Capture snapshot to data/snapshot_dms.jpg")
    print("  [T]          : Print current telemetry readings to console")
    print("=" * 70)

    # Initialize Camera and DMS Detector
    stream = CameraStream()
    detector = DMSDetector()

    window_name = "Module 2 Test: Real-Time Driver Monitoring (Press 'Q' to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    last_state = ""
    snapshot_idx = 0

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                print("[ERROR] Failed to grab frame.")
                break

            # 1. Run DMS Core Analysis
            result = detector.process(frame)

            # 2. Log state transitions to console
            current_state = result.primary_alert
            if current_state != last_state:
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] State Change -> [{result.status_level}] {current_state} (EAR={result.ear:.2f}, MAR={result.mar:.2f}, Yaw={result.yaw:+.1f}°, Pitch={result.pitch:+.1f}°)")
                last_state = current_state

            # 3. Draw Telemetry Overlays
            fps = stream.get_fps()
            status_text = "SIMULATION FEED" if stream.is_synthetic else "DMS ONLINE"
            status_color = (0, 215, 255) if stream.is_synthetic else (0, 255, 0)
            
            # Base HUD Bar
            frame = CameraStream.overlay_telemetry(frame, fps, status_text=status_text, status_color=status_color)
            
            # DMS Specific HUD (EAR/MAR Bars, 3D Pose Pointer, Alert Banner)
            frame = detector.draw_dms_hud(frame, result)

            # Bottom instruction strip
            h, w = frame.shape[:2]
            cv2.putText(frame, "Press 'Q' to Exit | 'C' to Snapshot | 'T' to Print Telemetry",
                        (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                print("[INFO] User requested exit.")
                break
            elif key in [ord('c'), ord('C')]:
                snapshot_idx += 1
                out_path = config.DATA_DIR / f"snapshot_dms_{snapshot_idx}.jpg"
                cv2.imwrite(str(out_path), frame)
                print(f"[INFO] Saved DMS snapshot to: {out_path}")
            elif key in [ord('t'), ord('T')]:
                print(f"\n--- [Telemetry Snapshot] ---")
                print(f"Face Detected  : {result.face_detected}")
                print(f"EAR (Eye Ratio): {result.ear:.3f} (Threshold: {config.EAR_THRESHOLD})")
                print(f"Blink Score    : {result.blink_score:.2f}")
                print(f"MAR (Mouth)    : {result.mar:.3f} (Threshold: {config.MAR_THRESHOLD})")
                print(f"Jaw Open Score : {result.jaw_open_score:.2f}")
                print(f"Head Yaw       : {result.yaw:+.1f} deg (Threshold: +/-{config.HEAD_YAW_THRESHOLD} deg)")
                print(f"Head Pitch     : {result.pitch:+.1f} deg (Threshold: -{config.HEAD_PITCH_DOWN_THRESHOLD} deg)")
                print(f"Stress Indicator: {result.stress_indicator} (Experimental)")
                print(f"Active Alerts  : {result.active_alerts}")
                print(f"----------------------------\n")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (KeyboardInterrupt).")
    finally:
        stream.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera and DMS resources safely closed.")

if __name__ == "__main__":
    run_dms_test()
