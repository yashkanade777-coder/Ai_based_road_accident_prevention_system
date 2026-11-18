"""
GPS Speed Monitor & Overspeeding Detection Module
Supports:
1. Real GPS feeds (NMEA sentences via serial/hotspot)
2. Realistic simulated drive cycles (City, Highway, Overspeed Test)
3. Automated Road Zone Geofencing (Auto-detects School Zone, City, Ring Road, Highway)
4. Automated Traffic Sign Recognition (TSR) Speed Limit Overrides
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import config

@dataclass
class RoadZone:
    """Pre-defined road geofence zone."""
    name: str
    speed_limit: float
    recommended_traffic: str
    description: str

ROAD_ZONES = [
    RoadZone("School & Hospital Zone", 30.0, "HEAVY", "School area, pedestrian crossing (30 km/h)"),
    RoadZone("Urban City Center", 50.0, "MODERATE", "City street, commercial district (50 km/h)"),
    RoadZone("Arterial Ring Road", 70.0, "MODERATE", "Multi-lane outer ring road (70 km/h)"),
    RoadZone("National Highway", 90.0, "LIGHT", "High-speed expressway corridor (90 km/h)")
]

@dataclass
class SpeedTelemetry:
    """Current speed telemetry status."""
    speed_kmh: float = 0.0
    speed_limit: float = 60.0
    is_overspeeding: bool = False
    overspeed_seconds: float = 0.0
    source_type: str = "SIMULATED"       # "SIMULATED" or "HARDWARE_GPS"
    source_label: str = "Simulated Drive Cycle"
    latitude: float = 12.9716
    longitude: float = 77.5946
    
    # Automated Speed Limit Info
    auto_mode_enabled: bool = False
    current_zone_name: str = "Manual Speed Limit"
    zone_description: str = ""
    limit_recently_changed: bool = False


class SpeedMonitor:
    """
    Manages vehicle speed readings, automated speed limits, and sustained overspeed alarms.
    """
    def __init__(self, speed_limit: float = None, sustained_threshold: float = None):
        self.speed_limit = float(speed_limit or config.DEFAULT_SPEED_LIMIT)
        self.sustained_threshold = float(sustained_threshold or config.OVERSPEED_SUSTAINED_SECONDS)

        # Telemetry state
        self.current_speed = 42.0
        self.target_speed = 45.0
        self.source_type = "SIMULATED"
        self.mode = "CITY"  # "CITY", "HIGHWAY", "OVERSPEED_TEST", "MANUAL"

        # Coordinates
        self.latitude = 12.9716
        self.longitude = 77.5946

        # Automated Speed Limit Geofencing State
        self.auto_speed_limit = False
        self._current_zone_idx = 1  # Urban City Center by default
        self._zone_entry_time = time.time()
        self._zone_dwell_seconds = 90.0  # Transition road zones every 90 seconds (1.5 min)
        self._limit_just_changed = False
        self._last_announced_limit = self.speed_limit

        # Timing
        self._overspeed_start_time: Optional[float] = None
        self._last_update_time = time.time()

    def update(self) -> SpeedTelemetry:
        """
        Updates speed readings and handles automated zone transitions.
        """
        now = time.time()
        dt = max(0.01, min(1.0, now - self._last_update_time))
        self._last_update_time = now

        # 1. Automated Road Zone Transitions (if enabled)
        if self.auto_speed_limit:
            if now - self._zone_entry_time >= self._zone_dwell_seconds:
                # Cycle to next road zone
                self._current_zone_idx = (self._current_zone_idx + 1) % len(ROAD_ZONES)
                new_zone = ROAD_ZONES[self._current_zone_idx]
                self.speed_limit = new_zone.speed_limit
                self._zone_entry_time = now
                self._limit_just_changed = True
                print(f"[SpeedMonitor] [AUTO GEOFENCE] Entered: {new_zone.name} -> Limit: {new_zone.speed_limit} km/h")

        # 2. Update speed physics
        if self.source_type == "SIMULATED":
            self._update_simulation(dt)

        # 3. Check sustained overspeeding
        is_above_limit = self.current_speed > self.speed_limit
        overspeed_duration = 0.0

        if is_above_limit:
            if self._overspeed_start_time is None:
                self._overspeed_start_time = now
            overspeed_duration = now - self._overspeed_start_time
        else:
            self._overspeed_start_time = None

        is_sustained_overspeed = overspeed_duration >= self.sustained_threshold

        # Source label
        zone_info = ROAD_ZONES[self._current_zone_idx]
        if self.auto_speed_limit:
            source_label = f"Auto Geofence: {zone_info.name} ({zone_info.speed_limit:.0f} km/h)"
            zone_desc = zone_info.description
        else:
            source_label = f"Manual Configuration ({self.mode} Mode)"
            zone_desc = f"Fixed Speed Limit: {self.speed_limit:.0f} km/h"

        telemetry = SpeedTelemetry(
            speed_kmh=round(self.current_speed, 1),
            speed_limit=round(self.speed_limit, 1),
            is_overspeeding=is_sustained_overspeed,
            overspeed_seconds=round(overspeed_duration, 1),
            source_type=self.source_type,
            source_label=source_label,
            latitude=self.latitude,
            longitude=self.longitude,
            auto_mode_enabled=self.auto_speed_limit,
            current_zone_name=zone_info.name if self.auto_speed_limit else "Manual Set",
            zone_description=zone_desc,
            limit_recently_changed=self._limit_just_changed
        )

        self._limit_just_changed = False
        return telemetry

    def _update_simulation(self, dt: float):
        """Generates realistic vehicle speed fluctuations that respect the active speed limit."""
        t = time.time()
        if self.mode == "CITY" or self.auto_speed_limit:
            # Cruising speed tracks roughly 3-5 km/h below the active limit
            cruising_base = max(20.0, self.speed_limit - 4.0)
            target = cruising_base + 3.0 * math.sin(t * 0.20) + random.uniform(-1.0, 1.0)
            self.target_speed = max(15.0, target)
        elif self.mode == "HIGHWAY":
            cruising_base = max(60.0, min(self.speed_limit, 85.0))
            target = cruising_base + 4.0 * math.sin(t * 0.15) + random.uniform(-1.0, 1.0)
            self.target_speed = target
        elif self.mode == "OVERSPEED_TEST":
            # Explicitly speeds 15-20 km/h above the active speed limit
            self.target_speed = self.speed_limit + 18.0

        # Smooth acceleration / braking inertia
        diff = self.target_speed - self.current_speed
        accel_rate = 14.0 * dt
        if abs(diff) <= accel_rate:
            self.current_speed = self.target_speed
        else:
            self.current_speed += math.copysign(accel_rate, diff)

        speed_ms = self.current_speed / 3.6
        dist_m = speed_ms * dt
        self.latitude += (dist_m / 111000.0)

    def set_zone_dwell_seconds(self, seconds: float):
        """Configures the transition interval between automated geofenced road zones."""
        self._zone_dwell_seconds = max(15.0, float(seconds))
        print(f"[SpeedMonitor] Road zone dwell duration set to {self._zone_dwell_seconds:.0f} seconds.")

    def set_auto_mode(self, enabled: bool):
        """Enables or disables automatic speed limit geofencing."""
        self.auto_speed_limit = enabled
        if enabled:
            zone = ROAD_ZONES[self._current_zone_idx]
            self.speed_limit = zone.speed_limit
            self._zone_entry_time = time.time()
            self._limit_just_changed = True
            print(f"[SpeedMonitor] Automated Speed Limits ENABLED. Active Zone: {zone.name}")
        else:
            print("[SpeedMonitor] Automated Speed Limits DISABLED. Reverted to manual control.")

    def apply_detected_speed_sign(self, sign_speed: float, sign_label: str = "Traffic Sign"):
        """
        Called when Traffic Sign Recognition (TSR) detects a speed sign.
        Instantly updates the speed limit automatically!
        """
        self.speed_limit = float(sign_speed)
        self._limit_just_changed = True
        print(f"[SpeedMonitor] [TSR DETECTED] Speed Limit updated to {self.speed_limit:.0f} km/h via {sign_label}!")

    def set_road_zone_by_index(self, idx: int):
        """Manually force selects a specific road zone."""
        idx = max(0, min(len(ROAD_ZONES) - 1, idx))
        self._current_zone_idx = idx
        zone = ROAD_ZONES[idx]
        self.speed_limit = zone.speed_limit
        self._zone_entry_time = time.time()
        self._limit_just_changed = True
        print(f"[SpeedMonitor] Manually applied zone: {zone.name} ({zone.speed_limit:.0f} km/h)")

    def accelerate(self, delta: float = 5.0):
        self.mode = "MANUAL"
        self.target_speed = min(140.0, self.current_speed + delta)

    def brake(self, delta: float = 7.0):
        self.mode = "MANUAL"
        self.target_speed = max(0.0, self.current_speed - delta)

    def set_speed_limit(self, limit: float):
        self.speed_limit = max(20.0, min(140.0, float(limit)))

    def set_mode(self, mode: str):
        mode = mode.upper()
        if mode in ["CITY", "HIGHWAY", "OVERSPEED_TEST", "MANUAL"]:
            self.mode = mode
            self.source_type = "SIMULATED"

    def parse_nmea_sentence(self, sentence: str) -> bool:
        if not sentence.startswith("$"):
            return False
        parts = sentence.strip().split(",")
        if parts[0] in ["$GPRMC", "$GNRMC"] and len(parts) > 7:
            try:
                knots = float(parts[7])
                kmh = knots * 1.852
                self.current_speed = kmh
                self.source_type = "HARDWARE_GPS"
                return True
            except ValueError:
                pass
        return False
