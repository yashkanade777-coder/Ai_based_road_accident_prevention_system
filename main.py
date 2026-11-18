"""
AI-Based Road Accident Prevention and Traffic Management System
Main Entry Point Launcher
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    while True:
        clear_screen()
        print("=" * 70)
        print("   AI-BASED ROAD ACCIDENT PREVENTION & TRAFFIC MANAGEMENT SYSTEM   ")
        print("=" * 70)
        print("Choose an option to launch:")
        print("  [1] Launch Modern Interactive Web Dashboard (Streamlit)")
        print("  [2] Launch Full Desktop HUD Mode (OpenCV Camera + Telemetry)")
        print("  [3] Run Driver Monitoring System (DMS) Test (Drowsiness/Yawn)")
        print("  [4] Run Phone Detection & Audio Alert Test")
        print("  [5] Run Audio Diagnostic (Speaker Test)")
        print("  [6] Verify / Re-download AI Model Weights")
        print("  [0] Exit")
        print("=" * 70)

        choice = input("Enter option [0-6]: ").strip()

        if choice == "1":
            print("\n[INFO] Starting Streamlit Web Dashboard on http://localhost:8501 ...")
            dashboard_path = PROJECT_ROOT / "ui" / "dashboard.py"
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
        elif choice == "2":
            print("\n[INFO] Starting Full Desktop HUD Window...")
            test_path = PROJECT_ROOT / "tests" / "test_speed_risk.py"
            subprocess.run([sys.executable, str(test_path)])
        elif choice == "3":
            print("\n[INFO] Running DMS Test...")
            test_path = PROJECT_ROOT / "tests" / "test_dms.py"
            subprocess.run([sys.executable, str(test_path)])
        elif choice == "4":
            print("\n[INFO] Running Phone & Voice Alert Test...")
            test_path = PROJECT_ROOT / "tests" / "test_phone_audio.py"
            subprocess.run([sys.executable, str(test_path)])
        elif choice == "5":
            print("\n[INFO] Running Audio Diagnostic...")
            test_path = PROJECT_ROOT / "tests" / "test_audio_direct.py"
            subprocess.run([sys.executable, str(test_path)])
        elif choice == "6":
            print("\n[INFO] Checking AI Models...")
            model_script = PROJECT_ROOT / "models" / "download_models.py"
            subprocess.run([sys.executable, str(model_script)])
            input("\nPress Enter to return to menu...")
        elif choice == "0":
            print("\nExiting. Stay safe on the road!")
            break
        else:
            print("\nInvalid choice. Please choose 0-6.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main_menu()
