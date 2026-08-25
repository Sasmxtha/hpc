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

physattest.ml.anomaly_transformer (Component 6) IS now wired in, using a checkpoint trained
on residuals from the real, SINDy+PINN-integrated MultiDomainObserver (see
physattest/ml/train_transformer_checkpoint.py), not the freshly-initialized model an earlier
version of this module deliberately avoided wiring in (an undertrained model injects
near-random noise into detection, which is exactly the failure mode a from-scratch-vs-
transfer-learning comparison elsewhere in this project ran into before it was fixed -- the
lesson generalizes here too, so the checkpoint gates whether this signal is used at all: see
_get_transformer()). Its role is specifically the one the spec gives it: catching sustained
drift too subtle for the per-step z-score threshold to ever fire on -- a sensor with a low
z-score every single cycle can still be flagged "suspicious" if its recent residual WINDOW
has a shape the transformer recognises as anomalous. It does not override BLOCK_THRESH's
physics-based blocking, only adds a softer trigger for what would otherwise be "clean".

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

import os
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch

from physattest.agents.state import AgentState, AlertSeverity
from physattest.ai.classifier_3way import SensorEvidence as MLSensorEvidence
from physattest.ai.classifier_3way import classify as ml_classify
from physattest.ai.forensics_llm import ForensicsEvidence, add_causal_link, run_forensics
from physattest.ai.llm_fallback import FallbackChain, GroqBackend, TinyLlamaBackend
from physattest.ml.anomaly_transformer import ResidualAnomalyTransformer

BLOCK_THRESH = 5.0
SUSPICIOUS_THRESH = 2.5
TRANSFORMER_SCORE_THRESH = 0.5
TRANSFORMER_SEQ_LEN = 20
TRANSFORMER_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "ml", "checkpoints", "anomaly_transformer_h1.pt"
)

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
# just whether it's currently large. Values are abs(residual)/noise_std -- magnitude only.
_residual_history: Dict[int, List[float]] = {}
# Per-sensor rolling SIGNED residual history (not abs-valued), for Component 6's transformer
# -- see _transformer_flags_drift's docstring for why this needs to be a separate buffer
# from _residual_history rather than reusing it.
_raw_residual_history: Dict[int, List[float]] = {}

_llm_chain: Optional[FallbackChain] = None
_causal_graph: Optional[nx.DiGraph] = None
_transformer: Optional[ResidualAnomalyTransformer] = None
_transformer_load_attempted = False


def _normalize_window(window: np.ndarray) -> np.ndarray:
    """Z-score a window by its own early-portion baseline. Must match
    physattest/ml/train_transformer_checkpoint.py's normalize_window exactly -- the
    checkpoint was trained on windows normalized this way, so scoring with any other
    convention would silently feed it out-of-distribution input.
    """
    edge = max(1, len(window) // 5)
    baseline_mean = float(np.mean(window[:edge]))
    baseline_std = float(np.std(window[:edge])) + 1e-6
    return (window - baseline_mean) / baseline_std


def _get_transformer() -> Optional[ResidualAnomalyTransformer]:
    """Lazily loads the Component 6 checkpoint. Returns None (and only ever tries once) if
    it isn't there -- a missing checkpoint means "not trained yet," and detection should
    silently fall back to z-score-only gating, not crash or, worse, run a randomly-
    initialized model that would inject noise into classification.
    """
    global _transformer, _transformer_load_attempted
    if _transformer is not None or _transformer_load_attempted:
        return _transformer
    _transformer_load_attempted = True
    if not os.path.exists(TRANSFORMER_CHECKPOINT_PATH):
        return None
    checkpoint = torch.load(TRANSFORMER_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = ResidualAnomalyTransformer(n_sensors=1, **checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _transformer = model
    return _transformer


def _transformer_flags_drift(sensor_id: int) -> bool:
    """True if Component 6 recognises the sensor's recent residual window as anomalous,
    even though the per-step z-score gate did not fire on it -- the "subtle slow-drift
    attacks that threshold detection misses" case from the spec. Silently returns False
    (never flags) if the checkpoint isn't available or there isn't yet a full window of
    history for this sensor -- this is an additive signal, its absence should never block
    normal z-score-based detection from working.

    Uses _raw_residual_history (signed), not _residual_history (abs-valued, used for
    onset_abruptness): the checkpoint was trained on the real observer's signed residual, so
    a ramp going negative or an oscillation crossing zero has a shape the model was actually
    shown. Rectifying the sign away first, as _residual_history does, would silently feed
    the model a distribution it never saw during training.
    """
    model = _get_transformer()
    if model is None:
        return False
    history = _raw_residual_history.get(sensor_id, [])
    if len(history) < TRANSFORMER_SEQ_LEN:
        return False
    window = _normalize_window(np.array(history[-TRANSFORMER_SEQ_LEN:], dtype=np.float32))
    window_t = torch.from_numpy(window).reshape(1, TRANSFORMER_SEQ_LEN, 1)
    with torch.no_grad():
        _, window_logit = model(window_t)
    score = float(torch.sigmoid(window_logit)[0])
    return score > TRANSFORMER_SCORE_THRESH


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
        _raw_residual_history.setdefault(i, []).append(float(residuals[i]))
        if len(_raw_residual_history[i]) > TRANSFORMER_SEQ_LEN:
            _raw_residual_history[i] = _raw_residual_history[i][-TRANSFORMER_SEQ_LEN:]

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
        elif _transformer_flags_drift(i):
            # Below the z-score threshold every single cycle, but Component 6 recognises
            # the recent window's SHAPE as anomalous -- the slow-drift case the spec
            # specifically calls out threshold detection as missing. Deliberately routed to
            # "suspicious", not "blocked": this is a softer, pattern-based signal, not the
            # physics-based BLOCK_THRESH violation that justifies freezing the sensor's
            # verified value outright.
            statuses.append("suspicious")
            suspicious.append(i)
            health_scores[i] = max(0, health_scores[i] - 1)

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
