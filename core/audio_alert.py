"""
Audio Alert Manager Module (Native Windows SAPI + pyttsx3 fallback + Acoustic Chime)
Uses Windows native SAPI SpVoice with COM thread initialization and audible warning chimes.
Strict cooldowns prevent audio spam.
"""

import threading
import queue
import time
import winsound
from typing import Dict, Optional

# Try native Windows SAPI COM first (most reliable on Windows)
try:
    import pythoncom
    import win32com.client
    HAS_WIN32COM = True
except ImportError:
    HAS_WIN32COM = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

import config

class AudioAlertManager:
    """
    Thread-safe, non-blocking voice alert synthesizer for Windows.
    Combines an automobile warning chime (winsound) with native Windows speech.
    Enforces per-category cooldowns so repetitive alerts are never spammed.
    """
    PHRASES = {
        "DROWSINESS": "Drowsiness detected. Please take a break.",
        "DISTRACTION": "Warning! Please keep your eyes on the road.",
        "PHONE": "Warning! Mobile phone usage detected.",
        "YAWNING": "Frequent yawning detected. Consider resting.",
        "OVERSPEED": "Warning! Speed limit exceeded. Reduce speed.",
        "HIGH_RISK": "Caution! High accident risk detected.",
        "SYSTEM_START": "Driver safety monitoring activated.",
        "SYSTEM_STOP": "Driver safety monitoring deactivated."
    }

    # Acoustic warning frequencies (Hz, duration_ms)
    CHIME_FREQUENCIES = {
        "DROWSINESS": [(1500, 120), (1800, 180)],   # High two-tone alarm
        "DISTRACTION": [(1200, 150)],                # Single sharp alert
        "PHONE": [(1100, 100), (1400, 120)],         # Tech alert chime
        "YAWNING": [(800, 200)],                     # Gentle reminder
        "OVERSPEED": [(1600, 100), (1600, 100)],     # Double beep
        "HIGH_RISK": [(2000, 150), (2000, 150)]      # Urgent double high beep
    }

    def __init__(self, cooldowns: Optional[Dict[str, float]] = None, volume: int = 100):
        self.cooldowns = cooldowns or config.ALERT_COOLDOWNS
        self.volume = max(0, min(100, volume))
        self.is_muted = False

        self._last_played: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Thread queue for background speech
        self._queue = queue.Queue(maxsize=3)
        self._running = True

        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

        print("[AudioAlertManager] Native Windows Audio Alert Engine initialized.")

    def trigger_alert(self, alert_type: str, custom_message: Optional[str] = None) -> bool:
        """
        Attempts to trigger an alert. Returns True if accepted, False if suppressed by cooldown.
        """
        if self.is_muted:
            return False

        alert_type = alert_type.upper()
        current_time = time.time()
        cooldown_period = self.cooldowns.get(alert_type, 4.0)

        with self._lock:
            last_time = self._last_played.get(alert_type, 0.0)
            if current_time - last_time < cooldown_period:
                return False

            self._last_played[alert_type] = current_time

        spoken_text = custom_message or self.PHRASES.get(alert_type, f"Warning! {alert_type} detected.")

        try:
            self._queue.put_nowait((alert_type, spoken_text))
            return True
        except queue.Full:
            print(f"[AudioAlertManager] Audio queue full, skipping: {alert_type}")
            return False

    def _play_chime(self, alert_type: str):
        """Plays a vehicle warning beep chime through Windows sound hardware."""
        chimes = self.CHIME_FREQUENCIES.get(alert_type, [(1000, 120)])
        try:
            for freq, dur in chimes:
                winsound.Beep(freq, dur)
        except Exception:
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception:
                pass

    def _speech_worker(self):
        """Dedicated background thread handling COM initialization and speech."""
        speaker = None
        engine = None

        if HAS_WIN32COM:
            try:
                pythoncom.CoInitialize()
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                speaker.Volume = self.volume
                speaker.Rate = 1  # 0 is normal, 1 is slightly prompt
                print("[AudioAlertManager] Using native Windows SAPI.SpVoice engine.")
            except Exception as e:
                print(f"[AudioAlertManager] SAPI initialization failed: {e}. Falling back to pyttsx3.")
                speaker = None

        if speaker is None and HAS_PYTTSX3:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 165)
                engine.setProperty('volume', 1.0)
                print("[AudioAlertManager] Using pyttsx3 fallback.")
            except Exception as e:
                print(f"[AudioAlertManager] pyttsx3 fallback failed: {e}")
                engine = None

        while self._running:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            alert_type, spoken_text = item

            # 1. First, play audible vehicle warning chime
            self._play_chime(alert_type)

            # 2. Speak the warning phrase
            try:
                print(f"[AudioAlertManager] [VOICE] Speaking: \"{spoken_text}\"")
                if speaker is not None:
                    # Windows SAPI Dispatch
                    speaker.Speak(spoken_text)
                elif engine is not None:
                    # pyttsx3 fallback
                    engine.say(spoken_text)
                    engine.runAndWait()
            except Exception as e:
                print(f"[AudioAlertManager] Speech error: {e}")
            finally:
                self._queue.task_done()

        if HAS_WIN32COM and speaker is not None:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def set_muted(self, muted: bool):
        """Toggles sound alerts on or off."""
        self.is_muted = muted
        print(f"[AudioAlertManager] Sound muted: {self.is_muted}")

    def reset_cooldown(self, alert_type: str):
        """Manually resets the cooldown timer for an alert category."""
        with self._lock:
            if alert_type in self._last_played:
                del self._last_played[alert_type]

    def close(self):
        """Shuts down the background voice worker thread."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        print("[AudioAlertManager] Closed.")
