# AI-Based Road Accident Prevention and Traffic Management System

A working engineering prototype of an intelligent vehicular safety assistant developed in Python. The system detects risky driver behaviors in real time, monitors vehicle speed, assesses accident hazards, and issues audio-visual alerts to prevent collisions.

---

## 1. Project Overview & Architecture

### System Architecture
The application runs locally on standard Windows hardware without demanding external GPUs.

```
+-----------------------------------------------------------------------------------+
|                                 SENSING LAYER                                     |
|  Laptop Webcam (640x480 @ 30 FPS)  |  GPS Telemetry (Simulated / NMEA Hardware)   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                PROCESSING LAYER                                   |
|  • MediaPipe 478 Landmark Mesh    -> Eye Aspect Ratio (EAR) & Yawn MAR            |
|  • Metric 3D Head Transformation  -> Yaw, Pitch, Roll Distraction Angles          |
|  • Asynchronous YOLOv8n Worker    -> Handheld Mobile Phone Detection              |
|  • Velocity Comparator            -> Sustained Overspeed Duration Tracker         |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         RISK EVALUATION & ALERT LAYER                             |
|  • Multi-Factor Accident Risk Engine -> Low / Medium / High Composite Hazard Score|
|  • Native Windows SAPI + Chime      -> Priority Cooldown Spoken Warnings          |
|  • SQLite Audit Database            -> Timestamped Incident Logging               |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                PRESENTATION LAYER                                 |
|  • Streamlit Web Dashboard          • OpenCV Real-Time HUD Cockpit Overlay        |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Capabilities & Mathematical Principles

### A. Drowsiness Detection via Eye Aspect Ratio (EAR)
The system calculates geometric EAR from eyelid landmark coordinates:
$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
- **Threshold**: $\text{EAR} < 0.22$ or neural blendshape blink $> 0.65$.
- **Temporal Filter**: Alert fires when eye closure persists for $> 24$ consecutive frames ($\approx 1.0\text{--}1.2\text{ s}$).

### B. Fatigue & Yawning Detection via Mouth Aspect Ratio (MAR)
$$\text{MAR} = \frac{\|p_{\text{upper}} - p_{\text{lower}}\|}{\|p_{\text{left\_corner}} - p_{\text{right\_corner}}\|}$$
- **Threshold**: $\text{MAR} > 0.65$ or `jawOpen` $> 0.60$ for $> 20$ frames ($\approx 0.8\text{ s}$).

### C. 3D Head Pose Distraction Tracking
Extracts the $3\times3$ metric head rotation matrix $R$ to compute Euler angles:
- **Yaw (Horizontal)**: $|\text{Yaw}| > 25^\circ$ indicates looking out side windows.
- **Pitch (Vertical)**: $\text{Pitch} < -20^\circ$ indicates looking down at lap or smartphone.

### D. Mobile Phone Detection
Runs a quantized YOLOv8n detector in a background thread targeting class `67` ("cell phone"). Bounding boxes are displayed with confidence percentages.

### E. GPS Speed & Sustained Overspeed Detection
- Compares vehicle velocity to the configurable speed limit.
- Triggers alerts only when overspeed is sustained for $> 3.0\text{ s}$ to prevent nuisance alarms during momentary overtaking maneuvers.

### F. Multi-Factor Composite Risk Index
Computes a composite score ($0\text{--}100$) combining driver impairment, speed ratio, and road congestion:
- **$0\text{--}39$**: 🟢 **LOW RISK** (Attentive driving)
- **$40\text{--}69$**: 🟡 **MEDIUM RISK** (Caution: elevated speed or mild fatigue)
- **$70\text{--}100$**: 🔴 **HIGH ACCIDENT RISK** (Immediate hazard: drowsiness + speed)

---

## 3. Project File Structure

```
ai_road_safety_system/
│
├── main.py                     # Main terminal menu launcher
├── requirements.txt             # Pinned library dependencies
├── config.py                    # Central thresholds, camera parameters & cooldowns
├── README.md                    # Project documentation & presentation guide
│
├── core/
│   ├── camera_stream.py         # DirectShow webcam capture & synthetic fallback
│   ├── dms_detector.py          # MediaPipe EAR, MAR, head pose & experimental cues
│   ├── object_detector.py       # Threaded YOLOv8 mobile phone detector
│   ├── audio_alert.py           # Native Windows SAPI voice engine + chimes + cooldowns
│   ├── speed_monitor.py         # Simulated drive cycles + NMEA GPS parser
│   ├── risk_evaluator.py        # Composite accident risk scoring engine
│   └── database.py              # SQLite alert history logger
│
├── models/
│   ├── download_models.py       # One-click model downloader
│   ├── face_landmarker.task     # MediaPipe 478-point landmark task asset
│   └── yolov8n.pt               # YOLOv8 nano pre-trained weights
│
├── ui/
│   └── dashboard.py             # Streamlit interactive dark cockpit web app
│
├── data/
│   └── accident_prevention.db   # SQLite database storing incident logs
│
└── tests/
    ├── test_camera.py           # Module 1: Camera verification
    ├── test_dms.py              # Module 2: Drowsiness & distraction test
    ├── test_phone_audio.py      # Modules 3 & 4: Phone & voice alerts
    ├── test_audio_direct.py     # Sound diagnostic test
    └── test_speed_risk.py       # Modules 5 & 6: Speed & risk evaluation
```

---

## 4. How to Run the Project

### Option A: Interactive Web Dashboard (Recommended for Demonstrations)
```powershell
cd C:\Users\User\.gemini\antigravity\scratch\ai_road_safety_system
streamlit run ui\dashboard.py
```
Open **`http://localhost:8501`** in your browser.
1. Click **`▶️ START`** in the sidebar.
2. View live camera feeds, speed gauges, phone alerts, and dynamic alert tables.
3. Use the sidebar sliders to change speed limits, test overspeeding, or adjust sensitivities.

---

### Option B: Standalone Desktop Cockpit Window (OpenCV HUD)
```powershell
python tests\test_speed_risk.py
```
- Controls:
  - `+` / `-` : Accelerate / Brake vehicle
  - `O` : Trigger overspeed test mode
  - `T` : Cycle road traffic density (Light / Moderate / Heavy)
  - `M` : Mute / Unmute voice warnings
  - `Q` : Exit

---

### Option C: Central Launcher
```powershell
python main.py
```
Presents a simple numbered terminal menu to launch any module or diagnostic with one keystroke.

---

## 5. College Viva / Demo Questions & Answers

**Q1: Why did you choose MediaPipe Face Landmarker instead of Dlib 68?**
> *Answer*: MediaPipe provides 478 3D landmarks (including iris tracking and metric head rotation) and runs at 30+ FPS directly on CPU without needing heavy C++ compilation or CUDA GPUs. Dlib is older and suffers from perspective ambiguity.

**Q2: How do you prevent the video stream from lagging when running YOLO object detection?**
> *Answer*: YOLOv8n is decoupled into an asynchronous background worker thread (`core/object_detector.py`). The camera capture and MediaPipe facial analysis continue running at a smooth 30 FPS, while YOLO updates phone bounding boxes independently.

**Q3: How do you prevent voice warnings from repeating every second?**
> *Answer*: The `AudioAlertManager` enforces a category cooldown timer (e.g. 5 seconds for drowsiness, 4 seconds for phone usage). Even if eyes stay closed for 100 consecutive frames, the voice speaks once, stays silent during the cooldown window, and only re-triggers if the hazard persists.

**Q4: Is facial stress analysis reliable for clinical diagnosis?**
> *Answer*: No. Facial cues like brow tension and blink cadence are experimental behavioral indicators. The project explicitly labels them as non-diagnostic experimental signals.

---

## 6. Safety Notice & Limitations
- **Stationary Testing Only**: This prototype is developed for educational evaluation and stationary desktop simulation. It must **not** be tested while operating an actual vehicle on public roads.
- **Hardware Variation**: Webcam field of view, lighting changes, and camera angles affect landmark visibility. In production, automotive systems use near-infrared (NIR) sensors for night driving.
- **Sensor Integrity**: Speed readings in the default prototype are simulated drive cycles. Live vehicle speed requires an OBD-II adapter or external GNSS antenna.
