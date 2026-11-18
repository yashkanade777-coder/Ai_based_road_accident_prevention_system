"""
Modules 5 & 6 Test Script: GPS Speed, Overspeeding & Composite Risk Assessment
Tests simulated drive cycles, sustained overspeed alerts, and composite risk scoring with audio.
"""

import sys
from pathlib import Path
import time
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.camera_stream import CameraStream
from core.dms_detector import DMSDetector
from core.object_detector import PhoneDetector
from core.audio_alert import AudioAlertManager
from core.speed_monitor import SpeedMonitor
from core.risk_evaluator import AccidentRiskEvaluator
from core.database import AlertDatabase
import config

def run_speed_risk_test():
    print("=" * 75)
    print("AI ROAD ACCIDENT PREVENTION - MODULES 5 & 6: SPEED & RISK MONITORING")
    print("=" * 75)
    print("Test Instructions:")
    print("  1. Press [+] / [-] : Accelerate or Brake manually.")
    print("  2. Press [O]       : Trigger OVERSPEED test (accelerates past limit).")
    print("  3. Press [H]       : Highway mode (75-90 km/h).")
    print("  4. Press [T]       : Cycle traffic density (LIGHT -> MODERATE -> HEAVY).")
    print("  5. Notice the 'Composite Risk Score' increase when speeding while distracted!")
    print("---------------------------------------------------------------------------")
    print("Keyboard Controls:")
    print("  [Q] or [ESC] : Quit")
    print("  [+] / [-]    : Accelerate / Brake")
    print("  [O]          : Overspeed test mode")
    print("  [C]          : Normal City mode")
    print("  [H]          : Highway mode")
    print("  [T]          : Toggle Traffic Density")
    print("  [M]          : Mute / Unmute audio")
    print("=" * 75)

    stream = CameraStream()
    dms = DMSDetector()
    phone_detector = PhoneDetector()
    audio = AudioAlertManager()
    speed_monitor = SpeedMonitor(speed_limit=60.0)
    risk_evaluator = AccidentRiskEvaluator()
    db = AlertDatabase()

    window_name = "Modules 5 & 6 Test: Speed & Risk Evaluation (Press 'Q' to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 750)

    last_logged_time = 0

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                break

            phone_detector.update_frame(frame)
            dms_res = dms.process(frame)
            phone_res = phone_detector.get_result()
            speed_telemetry = speed_monitor.update()
            risk_res = risk_evaluator.evaluate(dms_res, phone_res, speed_telemetry)

            # Audio warnings for overspeeding
            if speed_telemetry.is_overspeeding:
                if audio.trigger_alert("OVERSPEED"):
                    db.log_event("OVERSPEED", "CRITICAL", speed_telemetry.speed_kmh,
                                 speed_telemetry.speed_limit, risk_res.risk_level,
                                 f"Sustained overspeed: {speed_telemetry.speed_kmh} km/h")
            elif dms_res.is_drowsy:
                if audio.trigger_alert("DROWSINESS"):
                    db.log_event("DROWSINESS", "CRITICAL", speed_telemetry.speed_kmh,
                                 speed_telemetry.speed_limit, risk_res.risk_level)
            elif phone_res.phone_detected:
                if audio.trigger_alert("PHONE"):
                    db.log_event("PHONE", "CRITICAL", speed_telemetry.speed_kmh,
                                 speed_telemetry.speed_limit, risk_res.risk_level)
            elif dms_res.is_distracted:
                if audio.trigger_alert("DISTRACTION"):
                    db.log_event("DISTRACTION", "CAUTION", speed_telemetry.speed_kmh,
                                 speed_telemetry.speed_limit, risk_res.risk_level)

            # Draw base HUD & DMS
            fps = stream.get_fps()
            frame = CameraStream.overlay_telemetry(frame, fps, status_text="INTEGRATED SYSTEM ONLINE")
            frame = dms.draw_dms_hud(frame, dms_res)
            frame = phone_detector.draw_detections(frame, phone_res)

            h, w = frame.shape[:2]

            # -----------------------------------------------------------------
            # Draw Speedometer & GPS Telemetry HUD (Bottom Right)
            # -----------------------------------------------------------------
            hud_w, hud_h = 320, 160
            rx1 = w - hud_w - 20
            ry1 = h - hud_h - 45

            panel = frame.copy()
            cv2.rectangle(panel, (rx1, ry1), (rx1 + hud_w, ry1 + hud_h), (18, 18, 18), -1)
            # Border changes to red if overspeeding
            border_col = (0, 0, 255) if speed_telemetry.is_overspeeding else (70, 70, 70)
            cv2.rectangle(panel, (rx1, ry1), (rx1 + hud_w, ry1 + hud_h), border_col, 2)
            cv2.addWeighted(panel, 0.80, frame, 0.20, 0, frame)

            # Speedometer Text
            cv2.putText(frame, "VEHICLE SPEED", (rx1 + 15, ry1 + 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

            speed_val_col = (0, 0, 255) if speed_telemetry.is_overspeeding else (0, 255, 0)
            speed_str = f"{speed_telemetry.speed_kmh:.0f}"
            cv2.putText(frame, speed_str, (rx1 + 15, ry1 + 75),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, speed_val_col, 2, cv2.LINE_AA)
            cv2.putText(frame, "km/h", (rx1 + 15 + len(speed_str) * 32, ry1 + 75),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            # Speed Limit Badge (Circular speed sign)
            limit_cx = rx1 + hud_w - 55
            limit_cy = ry1 + 50
            cv2.circle(frame, (limit_cx, limit_cy), 28, (255, 255, 255), -1)
            cv2.circle(frame, (limit_cx, limit_cy), 28, (0, 0, 255), 3)
            cv2.putText(frame, f"{speed_telemetry.speed_limit:.0f}", (limit_cx - 18, limit_cy + 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

            # Overspeed Status Banner
            if speed_telemetry.is_overspeeding:
                cv2.putText(frame, f"OVERSPEED! ({speed_telemetry.overspeed_seconds:.1f}s)",
                            (rx1 + 15, ry1 + 105), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Limit: {speed_telemetry.speed_limit:.0f} km/h (Normal)",
                            (rx1 + 15, ry1 + 105), cv2.FONT_HERSHEY_DUPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

            # Source watermark
            cv2.putText(frame, speed_telemetry.source_label, (rx1 + 15, ry1 + 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1, cv2.LINE_AA)

            # -----------------------------------------------------------------
            # Draw Composite Accident Risk Indicator (Top Right)
            # -----------------------------------------------------------------
            risk_panel_w, risk_panel_h = 320, 85
            rk_x1 = w - risk_panel_w - 20
            rk_y1 = 45

            risk_card = frame.copy()
            cv2.rectangle(risk_card, (rk_x1, rk_y1), (rk_x1 + risk_panel_w, rk_y1 + risk_panel_h), (18, 18, 18), -1)
            cv2.rectangle(risk_card, (rk_x1, rk_y1), (rk_x1 + risk_panel_w, rk_y1 + risk_panel_h), risk_res.risk_color_bgr, 2)
            cv2.addWeighted(risk_card, 0.80, frame, 0.20, 0, frame)

            cv2.putText(frame, f"ACCIDENT RISK: {risk_res.risk_level}", (rk_x1 + 15, rk_y1 + 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.62, risk_res.risk_color_bgr, 2, cv2.LINE_AA)
            
            # Risk progress bar
            cv2.rectangle(frame, (rk_x1 + 15, rk_y1 + 40), (rk_x1 + 295, rk_y1 + 52), (50, 50, 50), -1)
            bar_fill = int((risk_res.risk_score / 100.0) * 280)
            cv2.rectangle(frame, (rk_x1 + 15, rk_y1 + 40), (rk_x1 + 15 + bar_fill, rk_y1 + 52), risk_res.risk_color_bgr, -1)

            cv2.putText(frame, f"Score: {risk_res.risk_score}/100 | Traffic: {risk_res.traffic_density}",
                        (rk_x1 + 15, rk_y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

            # Bottom Key Controls
            cv2.putText(frame, "[+] Accel | [-] Brake | [O] Overspeed Test | [C] City | [H] Highway | [T] Traffic | [Q] Exit",
                        (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('+'), ord('=')]:
                speed_monitor.accelerate(5.0)
            elif key in [ord('-'), ord('_')]:
                speed_monitor.brake(5.0)
            elif key in [ord('o'), ord('O')]:
                speed_monitor.set_mode("OVERSPEED_TEST")
                print("[INFO] Overspeed test mode activated! Vehicle accelerating past limit...")
            elif key in [ord('c'), ord('C')]:
                speed_monitor.set_mode("CITY")
            elif key in [ord('h'), ord('H')]:
                speed_monitor.set_mode("HIGHWAY")
            elif key in [ord('t'), ord('T')]:
                densities = ["LIGHT", "MODERATE", "HEAVY"]
                curr_idx = densities.index(risk_evaluator.traffic_density)
                new_density = densities[(curr_idx + 1) % len(densities)]
                risk_evaluator.set_traffic_density(new_density)
                print(f"[INFO] Traffic density set to: {new_density}")
            elif key in [ord('m'), ord('M')]:
                audio.set_muted(not audio.is_muted)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        phone_detector.close()
        audio.close()
        stream.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources safely closed.")

if __name__ == "__main__":
    run_speed_risk_test()
