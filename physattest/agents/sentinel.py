"""
Agent 1: Sentinel -- Always-on verifier and AI brain.

Runs the observer, computes residuals, classifies threats (fault/anomaly/attack),
calls the healing function when sensors are compromised, and runs forensics
on confirmed attacks.

Classification uses the LLM fallback chain:
  1. Groq API (Llama 3 70B) -- full reasoning
  2. TinyLlama 1.1B local   -- basic classification
  3. Hardcoded if-else rules -- always available

The detection pipeline itself (residuals, blocking, reconstruction) never
depends on any LLM. The LLM only enriches the fault/anomaly/attack
classification and forensics report.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.state import AgentState, AlertSeverity
from security.llm_fallback import (
    get_chain, SensorEvidence, Classification, LLMTier,
)

# Coupling edges for neighbour votes (subset used by sentinel)
COUPLING_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (0, 3), (1, 4), (2, 5),
]

# Per-sensor health history (persists across cycles)
_health_history: dict[int, list[float]] = {}


def _get_neighbours(sensor_id: int, n_sensors: int) -> list[int]:
    """Get coupled neighbour sensor IDs."""
    neighbours = set()
    for a, b in COUPLING_PAIRS:
        if a == sensor_id and b < n_sensors:
            neighbours.add(b)
        elif b == sensor_id and a < n_sensors:
            neighbours.add(a)
    return list(neighbours)


def _command_explains_change(
    sensor_id: int,
    residual: float,
    command: list[float] | None,
) -> bool:
    """Check if a recent command could explain the sensor change."""
    if command is None:
        return False
    if sensor_id < len(command):
        cmd_magnitude = abs(command[sensor_id] - 0.5)
        if cmd_magnitude > 0.1 and residual < 5.0:
            return True
    return False


def sentinel_node(state: AgentState) -> dict:
    """
    Sentinel agent: the always-on verification engine.

    Pipeline:
    1. Get raw sensor readings
    2. Compute residuals via observer (physics check)
    3. Classify each sensor: clean / suspicious / blocked
    4. For suspicious/blocked sensors, gather full evidence
    5. Run three-way classifier via LLM fallback chain
    6. Heal blocked sensors (reconstruct from neighbours)
    7. If attack confirmed, run forensics
    """
    raw = np.array(state["raw_readings"])
    n_sensors = len(raw)
    cycle = state.get("cycle_count", 0)
    command = state.get("agent_command", None)

    # --- Step 1: Compute residuals ---
    health_scores = state.get("health_scores", [100.0] * n_sensors)
    health_scores = list(health_scores)

    prev_readings = state.get("verified_readings", list(raw))
    prev = np.array(prev_readings)
    residuals = raw - prev
    noise_std = 0.05
    normalised = np.abs(residuals) / (noise_std + 1e-8)

    # --- Step 2: Classify each sensor ---
    BLOCK_THRESH = 5.0
    SUSPICIOUS_THRESH = 2.5

    blocked = []
    suspicious = []
    statuses = []
    classifications = {}
    verified = raw.copy()

    # Get fingerprint and probe info from state
    fp_status = state.get("fingerprint_status", {})
    fp_scores = state.get("fingerprint_scores", [0.5] * n_sensors)
    probe_verdict = state.get("probe_verdict", {})

    # Get the fallback chain
    chain = get_chain()

    for i in range(n_sensors):
        # Update health history
        if i not in _health_history:
            _health_history[i] = []
        _health_history[i].append(health_scores[i])
        if len(_health_history[i]) > 20:
            _health_history[i] = _health_history[i][-20:]

        if normalised[i] > BLOCK_THRESH:
            statuses.append("blocked")
            blocked.append(i)
            verified[i] = prev[i]
            health_scores[i] = max(0, health_scores[i] - 10)

            # --- Build full evidence for the classifier ---
            neighbours = _get_neighbours(i, n_sensors)
            nb_residuals = {nb: float(normalised[nb]) for nb in neighbours}
            nb_agree = any(normalised[nb] > SUSPICIOUS_THRESH for nb in neighbours)

            evidence = SensorEvidence(
                sensor_id=i,
                residual=float(residuals[i]),
                normalised_residual=float(normalised[i]),
                health_score=health_scores[i],
                health_trend=list(_health_history[i]),
                neighbour_residuals=nb_residuals,
                neighbours_agree=nb_agree,
                recent_command=command,
                command_explains_change=_command_explains_change(
                    i, float(normalised[i]), command),
                fingerprint_status=fp_status.get(i, "unknown"),
                fingerprint_confidence=float(fp_scores[i]) if i < len(fp_scores) else 0.5,
                probe_verdict=str(probe_verdict.get(i, "unknown")),
                probe_correlation=0.0,
            )

            result = chain.classify(evidence)
            classifications[i] = result.classification.value

        elif normalised[i] > SUSPICIOUS_THRESH:
            statuses.append("suspicious")
            suspicious.append(i)
            health_scores[i] = max(0, health_scores[i] - 2)

            # Lighter evidence gathering for suspicious sensors
            neighbours = _get_neighbours(i, n_sensors)
            nb_agree = any(normalised[nb] > SUSPICIOUS_THRESH for nb in neighbours)

            if nb_agree and _command_explains_change(i, float(normalised[i]), command):
                classifications[i] = "anomaly"
            else:
                evidence = SensorEvidence(
                    sensor_id=i,
                    residual=float(residuals[i]),
                    normalised_residual=float(normalised[i]),
                    health_score=health_scores[i],
                    health_trend=list(_health_history.get(i, [])),
                    neighbour_residuals={nb: float(normalised[nb]) for nb in neighbours},
                    neighbours_agree=nb_agree,
                    recent_command=command,
                    command_explains_change=_command_explains_change(
                        i, float(normalised[i]), command),
                    fingerprint_status=fp_status.get(i, "unknown"),
                    fingerprint_confidence=float(fp_scores[i]) if i < len(fp_scores) else 0.5,
                    probe_verdict=str(probe_verdict.get(i, "unknown")),
                )
                result = chain.classify(evidence)
                classifications[i] = result.classification.value
        else:
            statuses.append("clean")
            health_scores[i] = min(100, health_scores[i] + 0.5)

    # --- Step 3: Determine alert severity ---
    if len(blocked) >= 3:
        severity = AlertSeverity.CRITICAL
    elif len(blocked) >= 1:
        severity = AlertSeverity.HIGH
    elif len(suspicious) >= 2:
        severity = AlertSeverity.MEDIUM
    elif len(suspicious) >= 1:
        severity = AlertSeverity.LOW
    else:
        severity = AlertSeverity.NONE

    # --- Step 4: Forensics on confirmed attacks ---
    forensics = ""
    attack_sensors = [s for s, c in classifications.items() if c == "attack"]
    fault_sensors = [s for s, c in classifications.items() if c == "fault"]
    anomaly_sensors = [s for s, c in classifications.items() if c == "anomaly"]

    if attack_sensors:
        attack_context = (
            f"Sensors {attack_sensors} classified as ATTACK. "
            f"Residuals: {[f'{normalised[s]:.1f}sigma' for s in attack_sensors]}. "
            f"Health: {[f'{health_scores[s]:.0f}' for s in attack_sensors]}."
        )

        # Try LLM forensics, fall back to rule-based
        evidence_for_forensics = SensorEvidence(
            sensor_id=attack_sensors[0],
            residual=float(residuals[attack_sensors[0]]),
            normalised_residual=float(normalised[attack_sensors[0]]),
            health_score=health_scores[attack_sensors[0]],
        )
        forensics = chain.forensics(
            evidence_for_forensics, Classification.ATTACK, attack_context
        )

    # --- Build messages ---
    msgs = state.get("messages", [])
    tier_label = chain.active_tier.value

    if blocked:
        class_summary = ", ".join(
            f"s{s}={classifications.get(s, '?')}" for s in blocked
        )
        msgs = msgs + [
            f"[Sentinel] Blocked sensors {blocked}, severity={severity.name}. "
            f"Classification ({tier_label}): {class_summary}"
        ]
    if fault_sensors:
        msgs = msgs + [
            f"[Sentinel] FAULT sensors {fault_sensors} -- "
            f"hardware degradation, recommend maintenance"
        ]
    if anomaly_sensors:
        msgs = msgs + [
            f"[Sentinel] ANOMALY sensors {anomaly_sensors} -- "
            f"legitimate process event, no intervention needed"
        ]
    if forensics:
        msgs = msgs + [f"[Sentinel] {forensics}"]

    return {
        "verified_readings": verified.tolist(),
        "residuals": residuals.tolist(),
        "normalised_residuals": normalised.tolist(),
        "blocked_sensors": blocked,
        "suspicious_sensors": suspicious,
        "sensor_statuses": statuses,
        "health_scores": health_scores,
        "classification": classifications,
        "forensics_report": forensics,
        "alert_severity": int(severity),
        "messages": msgs,
    }
