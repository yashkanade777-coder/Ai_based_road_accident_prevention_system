"""
Modules 3 & 4 Test Script: Mobile Phone Detection & Real-Time Audio Alerts
Integrates Camera + DMS + YOLOv8n Phone Detector + Thread-Safe Voice Alerts with Cooldowns.
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
from core.object_detector import PhoneDetector
from core.audio_alert import AudioAlertManager
import config

def run_phone_audio_test():
    print("=" * 75)
    print("AI ROAD ACCIDENT PREVENTION - MODULES 3 & 4: PHONE DETECTION & AUDIO ALERTS")
    print("=" * 75)
    print("Test Instructions:")
    print("  1. Phone Test       : Hold up your smartphone in front of the camera.")
    print("  2. Drowsiness Test  : Close your eyes -> Listen for voice warning.")
    print("  3. Distraction Test : Turn your head left/right -> Listen for voice warning.")
    print("  4. Cooldown Test    : Keep eyes closed; notice voice does NOT repeat every frame!")
    print("---------------------------------------------------------------------------")
    print("Keyboard Controls:")
    print("  [Q] or [ESC] : Quit application")
    print("  [M]          : Toggle Mute / Unmute voice alerts")
    print("  [C]          : Capture snapshot to data/snapshot_phone_audio.jpg")
    print("  [1]          : Manually test Drowsiness voice alert")
    print("  [2]          : Manually test Distraction voice alert")
    print("  [3]          : Manually test Mobile Phone voice alert")
    print("  [4]          : Manually test Yawning voice alert")
    print("=" * 75)

    # Initialize all subsystems
    stream = CameraStream()
    dms = DMSDetector()
    phone_detector = PhoneDetector()
    audio = AudioAlertManager()

    # Announce system start
    audio.trigger_alert("SYSTEM_START")

    window_name = "Modules 3 & 4 Test: Phone Detection & Voice Alerts (Press 'Q' to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    last_spoken_time = 0
    last_spoken_text = "SYSTEM ACTIVE"
    snapshot_idx = 0

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                print("[ERROR] Failed to read frame.")
                break

            # 1. Update background phone detector with fresh frame
            phone_detector.update_frame(frame)

            # 2. Process DMS facial telemetry
            dms_result = dms.process(frame)

            # 3. Retrieve latest phone detection results (non-blocking)
            phone_result = phone_detector.get_result()

            # 4. Trigger Audio Alerts with Automatic Cooldowns
            # Drowsiness has highest priority
            if dms_result.is_drowsy:
                if audio.trigger_alert("DROWSINESS"):
                    last_spoken_text = "DROWSINESS DETECTED"
            # Distraction (looking away)
            elif dms_result.is_distracted:
                if audio.trigger_alert("DISTRACTION"):
                    last_spoken_text = "DISTRACTION: EYES ON ROAD"
            # Mobile Phone Usage
            elif phone_result.phone_detected:
                if audio.trigger_alert("PHONE"):
                    last_spoken_text = "PHONE USAGE DETECTED"
            # Yawning
            elif dms_result.is_yawning:
                if audio.trigger_alert("YAWNING"):
                    last_spoken_text = "YAWNING DETECTED"

            # 5. Draw Overlays
            fps = stream.get_fps()
            status_text = "DMS & PHONE MONITORING ACTIVE"
            frame = CameraStream.overlay_telemetry(frame, fps, status_text=status_text, status_color=(0, 255, 0))

            # Draw DMS HUD
            frame = dms.draw_dms_hud(frame, dms_result)

            # Draw Phone Detection Bounding Boxes
            frame = phone_detector.draw_detections(frame, phone_result)

            # If phone is detected, draw prominent phone warning banner
            h, w = frame.shape[:2]
            if phone_result.phone_detected:
                cv2.rectangle(frame, (w - 380, 100), (w - 20, 145), (0, 0, 200), -1)
                cv2.rectangle(frame, (w - 380, 100), (w - 20, 145), (0, 0, 255), 2)
                cv2.putText(frame, "WARNING: MOBILE PHONE IN USE!", (w - 370, 130),
                            cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            # Bottom Sound Status Indicator
            sound_icon = "MUTED [M]" if audio.is_muted else "AUDIO ON [M]"
            sound_color = (0, 0, 255) if audio.is_muted else (0, 255, 0)
            cv2.putText(frame, f"Sound: {sound_icon}", (20, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, sound_color, 1, cv2.LINE_AA)
            cv2.putText(frame, "Press 'Q': Exit | 'M': Mute | 'C': Snapshot | '1'-'4': Test Sounds",
                        (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                print("[INFO] User requested exit.")
                break
            elif key in [ord('m'), ord('M')]:
                audio.set_muted(not audio.is_muted)
            elif key in [ord('c'), ord('C')]:
                snapshot_idx += 1
                out_path = config.DATA_DIR / f"snapshot_phone_{snapshot_idx}.jpg"
                cv2.imwrite(str(out_path), frame)
                print(f"[INFO] Saved snapshot to: {out_path}")
            elif key == ord('1'):
                audio.reset_cooldown("DROWSINESS")
                audio.trigger_alert("DROWSINESS")
            elif key == ord('2'):
                audio.reset_cooldown("DISTRACTION")
                audio.trigger_alert("DISTRACTION")
            elif key == ord('3'):
                audio.reset_cooldown("PHONE")
                audio.trigger_alert("PHONE")
            elif key == ord('4'):
                audio.reset_cooldown("YAWNING")
                audio.trigger_alert("YAWNING")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        audio.trigger_alert("SYSTEM_STOP")
        time.sleep(1.0)
        phone_detector.close()
        audio.close()
        stream.release()
        cv2.destroyAllWindows()
        print("[INFO] Subsystems closed cleanly.")

if __name__ == "__main__":
    run_phone_audio_test()
