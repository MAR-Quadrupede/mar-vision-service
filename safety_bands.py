#!/usr/bin/env python3
"""
M.A.R. — Unified distance → risk / command bands.

Test matrix (hold a sheet / hand / box in FRONT of the stereo camera):

| Level      | Distance (front) | Expected command | Expected UI risk |
|------------|------------------|------------------|------------------|
| CRITICAL   |  < 40 cm         | STOP             | CRITICAL         |
| HIGH       |  40 – 70 cm      | FRENA            | HIGH             |
| MODERATE   |  70 – 120 cm     | AVANTI (lento)   | MODERATE         |
| LOW        |  > 120 cm        | AVANTI           | LOW              |

LiDAR/FSM use the same centimetre cuts so dashboard, vision and motors agree.
"""

from __future__ import annotations
from typing import Tuple

# Centimetri — ritara qui in un solo punto
DIST_CRITICAL_CM = 40.0
DIST_HIGH_CM = 70.0
DIST_MODERATE_CM = 120.0

# Metri (dashboard / MQTT consumers)
DIST_CRITICAL_M = DIST_CRITICAL_CM / 100.0
DIST_HIGH_M = DIST_HIGH_CM / 100.0
DIST_MODERATE_M = DIST_MODERATE_CM / 100.0


def classify_distance_cm(dist_cm: float) -> str:
    """Return CRITICAL | HIGH | MODERATE | LOW."""
    if dist_cm <= 0:
        return "CRITICAL"
    if dist_cm < DIST_CRITICAL_CM:
        return "CRITICAL"
    if dist_cm < DIST_HIGH_CM:
        return "HIGH"
    if dist_cm < DIST_MODERATE_CM:
        return "MODERATE"
    return "LOW"


def classify_distance_m(dist_m: float) -> str:
    return classify_distance_cm(float(dist_m) * 100.0)


def command_for_distance_cm(dist_cm: float) -> Tuple[str, str, str]:
    """
    Returns (command, reason, risk_level).
    Commands stay compatible with serial-bridge / FSM vocabulary.
    """
    level = classify_distance_cm(dist_cm)
    if level == "CRITICAL":
        return "STOP", f"CRITICAL: ostacolo a {dist_cm:.0f} cm (<{DIST_CRITICAL_CM:.0f})", level
    if level == "HIGH":
        return "FRENA", f"HIGH: ostacolo a {dist_cm:.0f} cm (<{DIST_HIGH_CM:.0f})", level
    if level == "MODERATE":
        return "AVANTI", f"MODERATE: cautela a {dist_cm:.0f} cm", level
    return "AVANTI", f"LOW: strada libera ({dist_cm:.0f} cm)", level
