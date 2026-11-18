"""
Audio Diagnostic Test Script
Verifies Windows audio speaker output, chimes, and speech synthesis.
"""

import sys
from pathlib import Path
import time
import winsound

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.audio_alert import AudioAlertManager

def run_audio_diagnostic():
    print("=" * 65)
    print("AUDIO DIAGNOSTIC TEST")
    print("=" * 65)
    print("1. Playing warning chime beep (1200 Hz)...")
    try:
        winsound.Beep(1200, 200)
        time.sleep(0.1)
        winsound.Beep(1600, 250)
        print("   [OK] Chimes played successfully.")
    except Exception as e:
        print(f"   [WARNING] Beep failed: {e}")

    print("\n2. Initializing AudioAlertManager...")
    manager = AudioAlertManager(volume=100)
    time.sleep(0.5)

    print("\n3. Testing speech synthesis...")
    print("   -> Speaking: 'Audio test successful. Your vehicle alert system is ready.'")
    manager.trigger_alert("SYSTEM_START", "Audio test successful. Your vehicle alert system is ready.")
    time.sleep(3.5)

    print("\n4. Testing Drowsiness alert with chime...")
    manager.reset_cooldown("DROWSINESS")
    manager.trigger_alert("DROWSINESS")
    time.sleep(3.5)

    manager.close()
    print("\n[OK] Diagnostic complete! If you heard the beeps and voice, your audio is fully working.")
    print("=" * 65)

if __name__ == "__main__":
    run_audio_diagnostic()
