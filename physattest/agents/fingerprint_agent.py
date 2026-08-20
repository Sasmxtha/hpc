"""
Agent 3: Fingerprint — Hardware identity checker.

Monitors each sensor's ADC noise pattern and timing jitter.
Every physical ADC has unique manufacturing imperfections that
create a distinctive noise fingerprint. Firmware compromise
changes timing behaviour, changing the fingerprint.

Operates INDEPENDENTLY from Sentinel — different code, different
data path. This is critical for Theorem 6 (agent resilience):
compromising Sentinel doesn't compromise Fingerprint.

Detection bound (Theorem 5):
    P(evasion) ≤ exp(-n × D_KL(p_true || p_fake))

Delegates to security.fingerprint.FingerprintDatabase for the math.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "security"))
from fingerprint import FingerprintDatabase, FingerprintVerdict, NoiseExtractor, FeatureExtractor

from .state import AgentState, DefenseLevel


# Module-level state — persists across cycles
_fp_db = FingerprintDatabase()
_reading_buffers: dict[int, list[float]] = {}

ENROLLMENT_WINDOW = 200  # samples needed for enrollment
VERIFY_WINDOW = 100      # samples per verification


def fingerprint_node(state: AgentState) -> dict:
    """
    Fingerprint agent: hardware identity verification.

    Only runs active verification at defense_level >= L3.
    Always collects noise samples in the background.

    Independent verification path from Sentinel — if Sentinel
    is compromised, Fingerprint still detects firmware tampering.
    """
    defense_level = state.get("defense_level", DefenseLevel.L1_MULTI_DOMAIN)
    msgs = state.get("messages", [])
    raw = state["raw_readings"]
    n_sensors = len(raw)

    # Always collect readings (even when not actively verifying)
    for i in range(n_sensors):
        if i not in _reading_buffers:
            _reading_buffers[i] = []
        _reading_buffers[i].append(raw[i])
        # Keep a bounded buffer
        if len(_reading_buffers[i]) > ENROLLMENT_WINDOW * 3:
            _reading_buffers[i] = _reading_buffers[i][-ENROLLMENT_WINDOW * 3:]

    # Enroll sensors if enough data and not yet enrolled
    for i in range(n_sensors):
        if i not in _fp_db.enrolled and len(_reading_buffers[i]) >= ENROLLMENT_WINDOW:
            readings = np.array(_reading_buffers[i][:ENROLLMENT_WINDOW])
            _fp_db.enroll(i, readings)

    if defense_level < DefenseLevel.L3_FINGERPRINTING:
        return {
            "fingerprint_status": {},
            "fingerprint_scores": [0.5] * n_sensors,
            "messages": msgs,
        }

    # Active verification
    fp_status = {}
    fp_scores = []

    for i in range(n_sensors):
        if len(_reading_buffers.get(i, [])) < VERIFY_WINDOW:
            fp_status[i] = "insufficient_data"
            fp_scores.append(0.5)
            continue

        readings = np.array(_reading_buffers[i][-VERIFY_WINDOW:])
        result = _fp_db.verify(i, readings)
        fp_status[i] = result.verdict.value
        fp_scores.append(result.confidence)

    compromised = [
        s for s, st in fp_status.items()
        if st == FingerprintVerdict.COMPROMISED.value
    ]
    suspicious = [
        s for s, st in fp_status.items()
        if st == FingerprintVerdict.SUSPICIOUS.value
    ]

    if compromised:
        # Get evasion bounds
        bounds = {s: _fp_db.get_evasion_bound(s) for s in compromised}
        msgs = msgs + [
            f"[Fingerprint] COMPROMISED sensors {compromised}. "
            f"ADC noise mismatch — firmware likely modified. "
            f"Evasion bounds: {', '.join(f's{s}≤{b:.2e}' for s, b in bounds.items())}"
        ]
    if suspicious:
        msgs = msgs + [
            f"[Fingerprint] Suspicious sensors {suspicious} — monitoring."
        ]

    return {
        "fingerprint_status": fp_status,
        "fingerprint_scores": fp_scores,
        "messages": msgs,
    }
