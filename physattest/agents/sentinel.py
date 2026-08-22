"""
Agent 1: Sentinel -- Always-on verifier and AI brain.

Runs the observer's residual check, classifies threats (fault/anomaly/attack), and runs
forensics on confirmed attacks. Detection gating (blocked/suspicious/clean, via normalised-
residual z-score thresholds) is Member 3's original, proven design and is left untouched.
What's replaced is the CLASSIFICATION step: instead of security.llm_fallback's evidence
schema and rule set, suspicious/blocked sensors are now classified by
physattest.ai.classifier_3way -- Member 2's 3-way classifier (health trend, neighbour
corroboration, command-log, fingerprint evidence, LLM-driven with a rule-based backstop),
live-tested against Groq and hardened against several real false-positive bugs found during
its own integration testing. Confirmed attacks are handed to
physattest.ai.forensics_llm.run_forensics for causal-chain narrative and severity/response
assessment, instead of security.llm_fallback's simpler canned-text forensics.

physattest.ml.anomaly_transformer (Component 6) is deliberately NOT wired in here yet. It
needs a checkpoint trained on real residual data to be a net positive detection signal --
wiring in a freshly-initialized model would inject near-random noise into classification,
which is exactly the failure mode a from-scratch-vs-transfer-learning comparison during this
project surfaced elsewhere (an undertrained model produced worse-than-baseline results).
Once a trained checkpoint exists, its window-level score is a natural extra term to fold into
_build_ml_evidence's residual_magnitude/onset_abruptness alongside the existing z-score.

Import note: this file previously used `sys.path.insert(...)` plus bare imports
(`from agents.state import AgentState`, `from security.llm_fallback import ...`). That is
incompatible with agents/graph.py, which imports this module via a RELATIVE import
(`from .sentinel import sentinel_node`) -- relative imports only work when `physattest` is
loaded as a real package (project root on sys.path), which means the bare-import path was
loading `agents.state` as a SEPARATE module object from `physattest.agents.state`, silently
duplicating AgentState/DefenseLevel/AlertSeverity. It happened not to break anything because
IntEnum compares by underlying value, but it's fragile. Fixed here to use absolute
`physattest.`-prefixed imports throughout, matching graph.py's convention.
"""

from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from physattest.agents.state import AgentState, AlertSeverity
from physattest.ai.classifier_3way import SensorEvidence as MLSensorEvidence
from physattest.ai.classifier_3way import classify as ml_classify
from physattest.ai.forensics_llm import ForensicsEvidence, add_causal_link, run_forensics
from physattest.ai.llm_fallback import FallbackChain, GroqBackend, TinyLlamaBackend

BLOCK_THRESH = 5.0
SUSPICIOUS_THRESH = 2.5

# Coupling edges for neighbour votes (subset used by sentinel) -- a fixed 6-sensor
# abstraction, as used throughout agents/demo.py. physattest/graph/coupling_graph.py's real
# ~34-tag SWaT graph is a separate, richer representation used by pipeline.py's non-agent
# detection path; reconciling the two is future integration work, not done here.
COUPLING_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (0, 3), (1, 4), (2, 5),
]

# Per-sensor health history (persists across cycles)
_health_history: Dict[int, List[float]] = {}
# Per-sensor rolling normalised-residual history, used to derive onset_abruptness -- how
# abruptly THIS cycle's deviation appeared relative to the sensor's own recent readings, not
# just whether it's currently large.
_residual_history: Dict[int, List[float]] = {}

_llm_chain: Optional[FallbackChain] = None
_causal_graph: Optional[nx.DiGraph] = None


def _get_llm_chain() -> FallbackChain:
    """Lazily-built module-level singleton, mirroring the _prober/_fp_db pattern used
    elsewhere in this codebase (guardian.py, fingerprint_agent.py).
    """
    global _llm_chain
    if _llm_chain is None:
        _llm_chain = FallbackChain(
            [GroqBackend(), TinyLlamaBackend(model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")]
        )
    return _llm_chain


def _get_causal_graph() -> nx.DiGraph:
    """STUB causal graph for forensics, one generic sensor -> decision -> consequence chain
    per sensor index. A real SWaT causal graph (which control decision each sensor actually
    feeds, which actuator that drives, what the real downstream physical consequence is)
    needs plant-specific control-logic knowledge this integration pass doesn't have --
    tracked as follow-up work, not invented here. This keeps forensics reports structured
    and non-empty rather than blank.
    """
    global _causal_graph
    if _causal_graph is None:
        g = nx.DiGraph()
        for i in range(8):  # generous upper bound; add_causal_link creates nodes lazily anyway
            add_causal_link(
                g,
                f"sensor_{i}",
                f"decision_{i}",
                relation="triggers",
                cause_kind="sensor",
                cause_label=f"Sensor {i} reading",
                effect_kind="decision",
                effect_label=f"Agent reacts to sensor {i}",
            )
            add_causal_link(
                g,
                f"decision_{i}",
                f"consequence_{i}",
                relation="causes",
                effect_kind="consequence",
                effect_label=f"Downstream effect of manipulating sensor {i}",
                effect_severity="high",
            )
        _causal_graph = g
    return _causal_graph


def _get_neighbours(sensor_id: int, n_sensors: int) -> List[int]:
    """Get coupled neighbour sensor IDs."""
    neighbours = set()
    for a, b in COUPLING_PAIRS:
        if a == sensor_id and b < n_sensors:
            neighbours.add(b)
        elif b == sensor_id and a < n_sensors:
            neighbours.add(a)
    return list(neighbours)


def _command_explains_change(sensor_id: int, residual: float, command: Optional[List[float]]) -> bool:
    """Check if a recent command could explain the sensor change."""
    if command is None:
        return False
    if sensor_id < len(command):
        cmd_magnitude = abs(command[sensor_id] - 0.5)
        if cmd_magnitude > 0.1 and residual < 5.0:
            return True
    return False


def _build_ml_evidence(
    sensor_id: int,
    n_sensors: int,
    normalised: np.ndarray,
    health_scores: List[float],
    command: Optional[List[float]],
    fp_status: dict,
) -> MLSensorEvidence:
    """Assembles physattest.ai.classifier_3way.SensorEvidence from what this cycle's
    threshold gating already computed, plus rolling history. residual_magnitude maps
    directly onto Member 3's existing normalised (z-score) residual -- same semantic
    meaning (noise-relative deviation), no separate computation needed.
    """
    neighbours = _get_neighbours(sensor_id, n_sensors)
    neighbour_corroboration = (
        float(np.mean([normalised[nb] > SUSPICIOUS_THRESH for nb in neighbours])) if neighbours else 0.0
    )

    history = _residual_history.get(sensor_id, [])
    onset_abruptness = 0.0
    if len(history) >= 3:
        edge = max(1, len(history) // 2)
        baseline_std = float(np.std(history[:edge])) + 1e-6
        max_step = float(np.max(np.abs(np.diff(history))))
        onset_abruptness = float(np.clip(max_step / (baseline_std * 3.0), 0.0, 1.0))

    fp = fp_status.get(sensor_id, "unknown")

    return MLSensorEvidence(
        sensor_id=f"sensor_{sensor_id}",
        health_score=health_scores[sensor_id],
        health_score_history=list(_health_history.get(sensor_id, [])),
        residual_magnitude=float(normalised[sensor_id]),
        noise_level_change=0.0,  # STUB: Member 1's observer owns baseline noise tracking
        neighbour_corroboration=neighbour_corroboration,
        command_explains_change=_command_explains_change(sensor_id, float(normalised[sensor_id]), command),
        fingerprint_changed=fp in ("compromised", "suspicious"),
        onset_abruptness=onset_abruptness,
    )


def sentinel_node(state: AgentState) -> dict:
    """
    Sentinel agent: the always-on verification engine.

    Pipeline:
    1. Get raw sensor readings
    2. Compute residuals (physics check via simple z-score against last verified readings)
    3. Gate: blocked / suspicious / clean, per Member 3's original thresholds
    4. For suspicious/blocked sensors, classify via the 3-way classifier (Member 2)
    5. If attack confirmed, run causal-chain forensics (Member 2)
    """
    raw = np.array(state["raw_readings"])
    n_sensors = len(raw)
    command = state.get("agent_command", None)

    health_scores = list(state.get("health_scores", [100.0] * n_sensors))
    prev = np.array(state.get("verified_readings", list(raw)))
    residuals = raw - prev
    noise_std = 0.05
    normalised = np.abs(residuals) / (noise_std + 1e-8)

    blocked: List[int] = []
    suspicious: List[int] = []
    statuses: List[str] = []
    classifications: Dict[int, str] = {}
    verified = raw.copy()

    fp_status = state.get("fingerprint_status", {})

    chain = _get_llm_chain()
    causal_graph = _get_causal_graph()
    last_backend_used = "rule_based"

    for i in range(n_sensors):
        _health_history.setdefault(i, []).append(health_scores[i])
        if len(_health_history[i]) > 20:
            _health_history[i] = _health_history[i][-20:]
        _residual_history.setdefault(i, []).append(float(normalised[i]))
        if len(_residual_history[i]) > 20:
            _residual_history[i] = _residual_history[i][-20:]

        if normalised[i] > BLOCK_THRESH:
            statuses.append("blocked")
            blocked.append(i)
            verified[i] = prev[i]
            health_scores[i] = max(0, health_scores[i] - 10)

            evidence = _build_ml_evidence(i, n_sensors, normalised, health_scores, command, fp_status)
            result = ml_classify(evidence, chain=chain)
            classifications[i] = max(result.probabilities, key=result.probabilities.get)
            last_backend_used = result.backend_used

        elif normalised[i] > SUSPICIOUS_THRESH:
            statuses.append("suspicious")
            suspicious.append(i)
            health_scores[i] = max(0, health_scores[i] - 2)

            neighbours = _get_neighbours(i, n_sensors)
            nb_agree = any(normalised[nb] > SUSPICIOUS_THRESH for nb in neighbours)

            if nb_agree and _command_explains_change(i, float(normalised[i]), command):
                classifications[i] = "anomaly"
            else:
                evidence = _build_ml_evidence(i, n_sensors, normalised, health_scores, command, fp_status)
                result = ml_classify(evidence, chain=chain)
                classifications[i] = max(result.probabilities, key=result.probabilities.get)
                last_backend_used = result.backend_used
        else:
            statuses.append("clean")
            health_scores[i] = min(100, health_scores[i] + 0.5)

    # --- Alert severity (unchanged from Member 3's original) ---
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

    # --- Forensics on confirmed attacks ---
    forensics = ""
    attack_sensors = [s for s, c in classifications.items() if c == "attack"]
    fault_sensors = [s for s, c in classifications.items() if c == "fault"]
    anomaly_sensors = [s for s, c in classifications.items() if c == "anomaly"]

    if attack_sensors:
        primary = attack_sensors[0]
        series = _residual_history.get(primary, [float(normalised[primary])])
        fe = ForensicsEvidence(
            sensor_id=f"sensor_{primary}",
            residual_series=np.array(series, dtype=float),
            event_context={
                f"sensor_{primary}": (
                    f"reports {raw[primary]:.2f} ({normalised[primary]:.1f} sigma from expected); "
                    f"last verified/reconstructed value was {verified[primary]:.2f}"
                )
            },
            concurrently_attacked_sensors=[f"sensor_{s}" for s in attack_sensors],
            classifier_confidence=1.0,  # already gated on classification == "attack" above
        )
        report = run_forensics(fe, causal_graph, chain=chain)
        forensics = (
            f"{report.attack_pattern} pattern, {report.coordination}, severity={report.severity}. "
            f"{report.inferred_intent} Recommended response: {report.recommended_response}."
        )

    # --- Build messages ---
    msgs = state.get("messages", [])

    if blocked:
        class_summary = ", ".join(f"s{s}={classifications.get(s, '?')}" for s in blocked)
        msgs = msgs + [
            f"[Sentinel] Blocked sensors {blocked}, severity={severity.name}. "
            f"Classification ({last_backend_used}): {class_summary}"
        ]
    if fault_sensors:
        msgs = msgs + [f"[Sentinel] FAULT sensors {fault_sensors} -- hardware degradation, recommend maintenance"]
    if anomaly_sensors:
        msgs = msgs + [f"[Sentinel] ANOMALY sensors {anomaly_sensors} -- legitimate process event, no intervention needed"]
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
