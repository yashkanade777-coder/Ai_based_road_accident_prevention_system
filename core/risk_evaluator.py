"""
Traffic & Accident Risk Evaluation Module
Blends real-time driver behaviors, vehicle velocity, and road congestion
into an experimental composite accident risk score (Low, Medium, High).
"""

from dataclasses import dataclass, field
from typing import List

import config

@dataclass
class RiskAssessment:
    """Multi-factor safety risk evaluation result."""
    risk_score: int = 15                 # 0 to 100
    risk_level: str = "LOW"              # "LOW", "MEDIUM", "HIGH"
    risk_color_bgr: tuple = (0, 255, 0)  # Green
    risk_factors: List[str] = field(default_factory=list)
    traffic_density: str = "MODERATE"    # "LIGHT", "MODERATE", "HEAVY"
    experimental_disclaimer: str = "Experimental Risk Estimate — Academic Prototype Only"


class AccidentRiskEvaluator:
    """
    Computes a composite safety risk indicator by evaluating:
    1. Driver impairment (Drowsiness, Distraction, Phone usage, Yawning)
    2. Vehicle speed relative to speed limit
    3. Road traffic density
    """
    def __init__(self):
        self.traffic_density = "MODERATE"  # Default simulated road traffic condition

    def set_traffic_density(self, density: str):
        """Sets traffic congestion level ('LIGHT', 'MODERATE', 'HEAVY')."""
        density = density.upper()
        if density in ["LIGHT", "MODERATE", "HEAVY"]:
            self.traffic_density = density

    def evaluate(self, dms_result, phone_result, speed_telemetry) -> RiskAssessment:
        """
        Calculates composite risk score (0-100) and assigns Low/Medium/High classification.
        """
        score = 10  # Baseline road presence risk
        factors = []

        # ---------------------------------------------------------
        # 1. Driver Behavior Penalty Weights
        # ---------------------------------------------------------
        if dms_result.is_drowsy:
            score += 45
            factors.append("Severe Driver Drowsiness")
        elif dms_result.is_distracted:
            score += 35
            factors.append(f"Driver Distracted ({dms_result.distraction_reason})")
        
        if phone_result.phone_detected:
            score += 35
            factors.append("Mobile Phone Usage While Driving")

        if dms_result.is_yawning:
            score += 15
            factors.append("Driver Fatigue / Frequent Yawning")

        # ---------------------------------------------------------
        # 2. Speed Ratio Penalty Weights
        # ---------------------------------------------------------
        if speed_telemetry.speed_limit > 0:
            speed_ratio = speed_telemetry.speed_kmh / speed_telemetry.speed_limit

            if speed_telemetry.is_overspeeding:
                score += 30
                factors.append(f"Sustained Overspeeding ({speed_telemetry.speed_kmh:.0f}/{speed_telemetry.speed_limit:.0f} km/h)")
            elif speed_ratio > 1.15:
                score += 20
                factors.append("Speed Exceeding Limit")
            elif speed_ratio > 1.0:
                score += 10
                factors.append("Approaching Overspeed Threshold")

        # ---------------------------------------------------------
        # 3. Traffic Density Factor
        # ---------------------------------------------------------
        if self.traffic_density == "HEAVY":
            score += 20
            factors.append("Heavy Congestion Ahead")
        elif self.traffic_density == "MODERATE":
            score += 5

        # Compound hazard multiplier (e.g. Drowsy / Phone + Overspeed is especially hazardous)
        if (dms_result.is_drowsy or phone_result.phone_detected) and speed_telemetry.speed_kmh > speed_telemetry.speed_limit:
            score += 15
            factors.append("Hazard Multiplier: Impairment + High Speed")

        # Clamp score to [0, 100]
        final_score = max(5, min(100, score))

        # Assign Risk Category
        if final_score >= 70:
            level = "HIGH"
            color = (0, 0, 255)      # Red
        elif final_score >= 40:
            level = "MEDIUM"
            color = (0, 215, 255)    # Amber
        else:
            level = "LOW"
            color = (0, 255, 0)      # Green

        return RiskAssessment(
            risk_score=final_score,
            risk_level=level,
            risk_color_bgr=color,
            risk_factors=factors,
            traffic_density=self.traffic_density
        )
