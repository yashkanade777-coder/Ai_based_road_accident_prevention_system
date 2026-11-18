"""
Streamlit Web Dashboard
AI-Based Road Accident Prevention and Traffic Management System
Features:
- Live Driver Monitoring (DMS with EAR, MAR, Head Pose)
- Handheld Mobile Phone Detection (YOLOv8n)
- Automated Speed Limits (Dynamic Geofence & TSR Presets)
- Multi-Factor Accident Risk Assessment
- Real-Time Voice Alerts & SQLite Incident History
"""

import sys
from pathlib import Path
import time
import pandas as pd
import streamlit as st
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.camera_stream import CameraStream
from core.dms_detector import DMSDetector
from core.object_detector import PhoneDetector
from core.audio_alert import AudioAlertManager
from core.speed_monitor import SpeedMonitor, ROAD_ZONES
from core.risk_evaluator import AccidentRiskEvaluator
from core.traffic_sign_detector import TrafficSignRecognizer
from core.database import AlertDatabase
import config

st.set_page_config(
    page_title="AI Road Safety & Driver Monitoring System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
    }
    .hud-title {
        color: #58a6ff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .hud-subtitle {
        color: #8b949e;
        font-size: 14px;
        margin-bottom: 16px;
    }
    .zone-badge {
        background-color: #1f6feb;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system_resources():
    print("[Dashboard] Initializing cached system resources...")
    dms = DMSDetector()
    phone_detector = PhoneDetector()
    audio = AudioAlertManager()
    speed_monitor = SpeedMonitor(speed_limit=config.DEFAULT_SPEED_LIMIT)
    risk_evaluator = AccidentRiskEvaluator()
    tsr = TrafficSignRecognizer()
    db = AlertDatabase()
    return dms, phone_detector, audio, speed_monitor, risk_evaluator, tsr, db

try:
    dms, phone_detector, audio, speed_monitor, risk_evaluator, tsr, db = load_system_resources()
except Exception as e:
    st.error(f"Failed to initialize models: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & SETTINGS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ System Controls")
    
    if "monitoring_active" not in st.session_state:
        st.session_state.monitoring_active = False

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ START", use_container_width=True, type="primary"):
            st.session_state.monitoring_active = True
            audio.trigger_alert("SYSTEM_START")
    with col_btn2:
        if st.button("⏹️ STOP", use_container_width=True):
            st.session_state.monitoring_active = False
            audio.trigger_alert("SYSTEM_STOP")

    if st.session_state.monitoring_active:
        st.success("🟢 Monitoring Active")
    else:
        st.warning("⏸️ Monitoring Paused")

    st.markdown("---")
    st.markdown("### 🏎️ Automated Speed Limit System")
    
    speed_mode = st.radio(
        "Speed Limit Source",
        options=["🤖 Auto Dynamic Geofence", "📷 Traffic Sign (TSR Presets)", "Manual Slider"],
        index=0
    )

    if speed_mode == "🤖 Auto Dynamic Geofence":
        speed_monitor.set_auto_mode(True)
        st.info("💡 Speed limit adapts **automatically** as the car crosses geofenced road zones!")
    elif speed_mode == "📷 Traffic Sign (TSR Presets)":
        speed_monitor.set_auto_mode(False)
        st.markdown("**Click a detected road sign to auto-set limit:**")
        tsr_cols = st.columns(4)
        for i, z in enumerate(ROAD_ZONES):
            with tsr_cols[i]:
                if st.button(f"{int(z.speed_limit)}", key=f"tsr_{i}", help=z.name):
                    speed_monitor.apply_detected_speed_sign(z.speed_limit, z.name)
                    audio.reset_cooldown("OVERSPEED")
                    audio.trigger_alert("OVERSPEED", f"Traffic sign detected. Speed limit updated to {int(z.speed_limit)} kilometers per hour.")
                    st.toast(f"TSR Auto-Applied: {z.name} ({int(z.speed_limit)} km/h)")
    else:
        speed_monitor.set_auto_mode(False)
        manual_lim = st.slider("Manual Speed Limit (km/h)", 30, 120, int(speed_monitor.speed_limit), 5)
        speed_monitor.set_speed_limit(manual_lim)

    st.markdown("---")
    st.markdown("### 🚗 Drive Cycle & Traffic")
    drive_mode = st.selectbox(
        "Drive Simulation Mode",
        options=["City Driving (35-55 km/h)", "Highway Cruising (70-90 km/h)", "Overspeed Test Mode"]
    )
    if "City" in drive_mode:
        speed_monitor.set_mode("CITY")
    elif "Highway" in drive_mode:
        speed_monitor.set_mode("HIGHWAY")
    elif "Overspeed" in drive_mode:
        speed_monitor.set_mode("OVERSPEED_TEST")

    traffic_density = st.select_slider(
        "Road Traffic Congestion",
        options=["LIGHT", "MODERATE", "HEAVY"],
        value=risk_evaluator.traffic_density
    )
    risk_evaluator.set_traffic_density(traffic_density)

    st.markdown("---")
    st.markdown("### 🔊 Audio Settings")
    sound_muted = st.checkbox("Mute Spoken Warnings", value=audio.is_muted)
    audio.set_muted(sound_muted)

    st.caption("AI-Based Road Accident Prevention System v2.0")

# -----------------------------------------------------------------------------
# MAIN DASHBOARD HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="hud-title">AI Road Accident Prevention & Traffic Management</div>', unsafe_allow_html=True)
st.markdown('<div class="hud-subtitle">Real-Time Driver Monitoring, Automated Speed Limits, Mobile Phone Detection & Multi-Factor Risk Assessment</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# REAL-TIME METRIC CARDS ROW
# -----------------------------------------------------------------------------
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_speed = metric_col1.empty()
metric_driver = metric_col2.empty()
metric_phone = metric_col3.empty()
metric_risk = metric_col4.empty()

# -----------------------------------------------------------------------------
# CENTER SECTION: VIDEO FEED & TELEMETRY GAUGES
# -----------------------------------------------------------------------------
col_video, col_gauges = st.columns([1.5, 1.0])

with col_video:
    st.markdown("##### 📹 Live Camera & Detection Stream")
    camera_placeholder = st.empty()

with col_gauges:
    st.markdown("##### 📊 Real-Time Driver Telemetry & Zones")
    gauge_zone = st.empty()
    gauge_ear = st.empty()
    gauge_mar = st.empty()
    gauge_head = st.empty()
    gauge_risk_bar = st.empty()
    gauge_factors = st.empty()

st.markdown("---")

# -----------------------------------------------------------------------------
# BOTTOM SECTION: ALERTS HISTORY LOG
# -----------------------------------------------------------------------------
st.markdown("### 📋 Recent Safety Alert History")
history_placeholder = st.empty()

col_hist_btn1, col_hist_btn2 = st.columns([1, 4])
with col_hist_btn1:
    if st.button("🗑️ Clear Alert Log"):
        db.clear_history()
        st.toast("Alert history cleared!")

st.info("⚠️ **Academic Notice & Safety Disclaimer**: This software is an engineering student prototype designed for controlled stationary setups. It is not an automotive-certified ADAS safety system. Stress indicators and risk levels are experimental estimates.")

# -----------------------------------------------------------------------------
# MAIN EXECUTION LOOP
# -----------------------------------------------------------------------------
if st.session_state.monitoring_active:
    stream = CameraStream()
    try:
        while st.session_state.monitoring_active:
            ret, frame = stream.read()
            if not ret or frame is None:
                st.error("Failed to read from camera.")
                break

            # 1. Pipeline processing
            phone_detector.update_frame(frame)
            dms_res = dms.process(frame)
            phone_res = phone_detector.get_result()
            speed_telemetry = speed_monitor.update()
            risk_res = risk_evaluator.evaluate(dms_res, phone_res, speed_telemetry)

            # 2. Automated Speed Limit Announcement (if limit auto-changed)
            if speed_telemetry.limit_recently_changed:
                announcement = f"Attention: Speed limit updated to {int(speed_telemetry.speed_limit)} kilometers per hour."
                audio.trigger_alert("OVERSPEED", announcement)
                db.log_event("LIMIT_CHANGE", "INFO", speed_telemetry.speed_kmh,
                             speed_telemetry.speed_limit, risk_res.risk_level,
                             f"Auto-applied: {speed_telemetry.current_zone_name}")

            # 3. Audio alerts & Database Logging
            if speed_telemetry.is_overspeeding:
                if audio.trigger_alert("OVERSPEED"):
                    db.log_event("OVERSPEED", "CRITICAL", speed_telemetry.speed_kmh,
                                 speed_telemetry.speed_limit, risk_res.risk_level,
                                 f"Overspeeding at {speed_telemetry.speed_kmh:.0f} km/h")
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
            elif dms_res.is_yawning:
                if audio.trigger_alert("YAWNING"):
                    db.log_event("YAWNING", "CAUTION", speed_telemetry.speed_kmh,
                                 speed_telemetry.speed_limit, risk_res.risk_level)

            # 4. Draw annotations on video
            fps = stream.get_fps()
            frame = CameraStream.overlay_telemetry(frame, fps, status_text="LIVE MONITORING ACTIVE")
            frame = dms.draw_dms_hud(frame, dms_res)
            frame = phone_detector.draw_detections(frame, phone_res)

            # Video HUD for Auto Speed Limit
            h, w = frame.shape[:2]
            mode_tag = "AUTO GEOFENCE" if speed_telemetry.auto_mode_enabled else "MANUAL"
            cv2.rectangle(frame, (w - 240, h - 75), (w - 15, h - 25), (25, 25, 25), -1)
            cv2.rectangle(frame, (w - 240, h - 75), (w - 15, h - 25), (0, 215, 255), 1)
            cv2.putText(frame, f"LIMIT: {speed_telemetry.speed_limit:.0f} km/h [{mode_tag}]",
                        (w - 230, h - 50), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, speed_telemetry.current_zone_name[:24],
                        (w - 230, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # 5. Update Top Metric Cards
            delta_speed = speed_telemetry.speed_kmh - speed_telemetry.speed_limit
            delta_str = f"+{delta_speed:.1f} km/h (OVERSPEED!)" if delta_speed > 0 else f"{delta_speed:.1f} km/h (Under limit)"
            metric_speed.metric(
                label=f"🏎️ Speed / Limit [{mode_tag}]",
                value=f"{speed_telemetry.speed_kmh:.0f} / {speed_telemetry.speed_limit:.0f} km/h",
                delta=delta_str,
                delta_color="inverse"
            )

            d_status = "NORMAL"
            if dms_res.is_drowsy:
                d_status = "DROWSY! 🚨"
            elif dms_res.is_distracted:
                d_status = f"DISTRACTED ({dms_res.distraction_reason}) ⚠️"
            elif dms_res.is_yawning:
                d_status = "YAWNING 🥱"
            metric_driver.metric(
                label="👁️ Driver Behavioral State",
                value=d_status,
                delta=f"EAR: {dms_res.ear:.2f} | MAR: {dms_res.mar:.2f}"
            )

            p_val = "PHONE IN HAND! 📱" if phone_res.phone_detected else "HANDS CLEAR ✅"
            metric_phone.metric(
                label="📱 Mobile Phone Detection",
                value=p_val,
                delta=f"Conf: {phone_res.confidence*100:.0f}%" if phone_res.phone_detected else "Safe"
            )

            r_val = f"{risk_res.risk_level} ({risk_res.risk_score}/100)"
            metric_risk.metric(
                label="⚠️ Accident Risk Level",
                value=r_val,
                delta=f"Traffic: {risk_res.traffic_density}",
                delta_color="off" if risk_res.risk_level == "LOW" else "inverse"
            )

            # 6. Update Gauges Column
            gauge_zone.markdown(f"**Active Road Zone:** `{speed_telemetry.current_zone_name}` — *{speed_telemetry.zone_description}*")
            gauge_ear.progress(
                float(np.clip(dms_res.ear / 0.40, 0.0, 1.0)),
                text=f"Eye Aspect Ratio (EAR): {dms_res.ear:.2f} (Threshold: {config.EAR_THRESHOLD:.2f})"
            )
            gauge_mar.progress(
                float(np.clip(dms_res.mar / 1.0, 0.0, 1.0)),
                text=f"Mouth Aspect Ratio (MAR): {dms_res.mar:.2f} (Threshold: {config.MAR_THRESHOLD:.2f})"
            )
            gauge_head.markdown(f"**Head Orientation:** Yaw `{dms_res.yaw:+.1f}°` | Pitch `{dms_res.pitch:+.1f}°` | Roll `{dms_res.roll:+.1f}°`")
            
            risk_pct = float(np.clip(risk_res.risk_score / 100.0, 0.0, 1.0))
            gauge_risk_bar.progress(risk_pct, text=f"Composite Accident Hazard Score: {risk_res.risk_score}%")

            if len(risk_res.risk_factors) > 0:
                factor_str = " • ".join(risk_res.risk_factors)
                gauge_factors.markdown(f"**Contributing Risk Factors:** *{factor_str}*")
            else:
                gauge_factors.markdown("**Contributing Risk Factors:** *None (Optimal Driving Conditions)*")

            # 7. Update SQLite History Table
            recent_alerts = db.get_recent_alerts(limit=10)
            if recent_alerts:
                df = pd.DataFrame(recent_alerts)
                df = df[['timestamp', 'event_type', 'severity', 'speed', 'speed_limit', 'risk_level']]
                df.columns = ['Time', 'Event', 'Severity', 'Speed (km/h)', 'Limit (km/h)', 'Risk Level']
                history_placeholder.dataframe(df, use_container_width=True, hide_index=True)
            else:
                history_placeholder.caption("No alerts logged yet. System running smoothly.")

            time.sleep(0.01)

    except Exception as e:
        st.error(f"Error during video loop: {e}")
    finally:
        stream.release()

else:
    camera_placeholder.info("Click **'▶️ START'** in the sidebar to activate the live camera feed and automated driver monitoring.")
    recent_alerts = db.get_recent_alerts(limit=15)
    if recent_alerts:
        df = pd.DataFrame(recent_alerts)
        df = df[['timestamp', 'event_type', 'severity', 'speed', 'speed_limit', 'risk_level']]
        df.columns = ['Time', 'Event', 'Severity', 'Speed (km/h)', 'Limit (km/h)', 'Risk Level']
        history_placeholder.dataframe(df, use_container_width=True, hide_index=True)
    else:
        history_placeholder.caption("No past alerts in database.")
