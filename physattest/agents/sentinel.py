<<<<<<< HEAD
"""Sentinel agent: "Runs observer, transformer, health scores, three-way classifier, LLM
forensics. Calls the healing function when needed. Handles Steps 1-7 of the pipeline."

This graph covers the pieces Sentinel needs that Member 2 owns: transformer scoring
(Component 6), health-score tracking, coupling-graph-based neighbour corroboration (using
Component 2's GNN-discovered or manual adjacency), and the 3-way classifier. It does NOT
call the physics observer itself (Component 1, Member 1) or the healing function (Component
5, Member 1) -- those are explicitly marked STUB below. Forensics runs one level up, in
overseer.py, because it needs to know about OTHER sensors' classifications this same cycle
(coordination), which a single sensor's Sentinel graph invocation doesn't have visibility
into.

One graph invocation = one sensor, one cycle. The Overseer calls it once per sensor per
cycle and aggregates -- see overseer.py.
"""

from typing import Dict, List

import numpy as np
import torch
from langgraph.graph import END, StateGraph

from physattest.agents.state import SentinelDependencies, SentinelState
from physattest.ai.classifier_3way import RESPONSE_ACTIONS, SensorEvidence, classify

# --- STUB integration points -------------------------------------------------------------
# These represent evidence that real components (owned by other team members) would supply.
# Wired to safe defaults so the graph runs end to end today; replace the call sites in
# gather_evidence_node below once the real components exist.
#
#   command_explains_change <- Member 3's bidirectional shield / command log (Component 4)
#   fingerprint_changed     <- Member 3's noise fingerprinting agent (Component 8)
#   noise_level_change      <- Member 1's physics observer baseline noise tracking (Component 1)
# -------------------------------------------------------------------------------------------


def _score_window(transformer, window: List[float]) -> tuple:
    """Runs Component 6 on a single sensor's residual window (n_sensors=1). Returns
    (window_score, per_timestep_scores).
    """
    arr = np.asarray(window, dtype=np.float32).reshape(1, -1, 1)  # (batch=1, seq_len, n_sensors=1)
    with torch.no_grad():
        timestep_logits, window_logit = transformer(torch.from_numpy(arr))
    score = float(torch.sigmoid(window_logit)[0])
    ts_scores = torch.sigmoid(timestep_logits)[0].tolist()
    return score, ts_scores


def make_score_node(deps: SentinelDependencies):
    def score_residual(state: SentinelState) -> dict:
        score, ts_scores = _score_window(deps.transformer, state["residual_window"])
        return {"transformer_score": score, "timestep_scores": ts_scores}

    return score_residual


def make_health_node(deps: SentinelDependencies):
    def update_health(state: SentinelState) -> dict:
        current = state.get("health_score", 100.0)
        score = state["transformer_score"]
        if score > 0.5:
            delta = -deps.health_decay_rate * (score - 0.5) * 2  # scale (0.5, 1.0] -> (0, decay_rate]
        else:
            delta = deps.health_recovery_rate
        new_health = float(np.clip(current + delta, 0.0, 100.0))
        return {"new_health_score": new_health}

    return update_health


def make_evidence_node(deps: SentinelDependencies):
    def gather_evidence(state: SentinelState) -> dict:
        window = np.asarray(state["residual_window"], dtype=np.float32)
        # Compare the window's tail to its own early portion (medians, not means/extremes)
        # rather than "max deviation from the first sample" -- taking a max over many noisy
        # points is biased to look large purely by chance (the same look-elsewhere effect
        # classify_attack_pattern in forensics_llm.py is written to avoid), which was
        # producing inflated residual_magnitude -- and false ATTACK classifications -- on
        # plain normal noise during agent-wiring testing. This mirrors that function's
        # baseline/tail approach so the two stay consistent.
        edge = max(1, len(window) // 5)
        baseline = float(np.median(window[:edge]))
        tail = float(np.median(window[-edge:]))
        baseline_std = float(np.std(window[:edge])) + 1e-6
        residual_magnitude = abs(tail - baseline) / baseline_std

        neighbour_windows: Dict[str, List[float]] = state.get("neighbour_windows", {})
        neighbour_scores = [
            _score_window(deps.transformer, nb_window)[0] for nb_window in neighbour_windows.values()
        ]
        neighbour_corroboration = (
            float(np.mean([s > deps.anomaly_threshold for s in neighbour_scores])) if neighbour_scores else 0.0
        )

        # Deliberately computed from the raw residual window, not from the transformer's
        # per-timestep SCORES: a classifier trained to near-binary output naturally produces
        # a sharp score jump at the true onset when well-converged, but a lightly-trained or
        # imperfectly-calibrated model can just as easily ramp its score up gradually over
        # several timesteps even for a genuinely abrupt step in the underlying data -- which
        # showed up directly during agent-wiring testing as onset_abruptness swinging between
        # ~0.9 and ~0.3 run to run for the identical injected step pattern, purely as a
        # function of how the transformer's training happened to converge. Onset abruptness
        # is a property of the physical signal ("did it jump or drift"), not of how confident
        # a possibly-imperfect classifier is about it, so it should be measured directly.
        diffs = np.diff(window)
        max_step = float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0
        onset_abruptness = float(np.clip(max_step / (baseline_std * 3.0), 0.0, 1.0))

        evidence = SensorEvidence(
            sensor_id=state["sensor_id"],
            health_score=state["new_health_score"],
            health_score_history=(state.get("health_history", []) + [state["new_health_score"]])[-10:],
            residual_magnitude=residual_magnitude,
            noise_level_change=0.0,  # STUB: Member 1's observer owns baseline noise tracking
            neighbour_corroboration=neighbour_corroboration,
            command_explains_change=state.get("command_explains_change", False),  # STUB, see module docstring
            fingerprint_changed=state.get("fingerprint_changed", False),  # STUB, see module docstring
            onset_abruptness=onset_abruptness,
        )
        return {"evidence": evidence}

    return gather_evidence


def make_classify_node(deps: SentinelDependencies):
    def classify_node(state: SentinelState) -> dict:
        result = classify(state["evidence"], chain=deps.llm_chain)
        return {"classification": result, "response_action": result.response_action}

    return classify_node


def make_score_router(deps: SentinelDependencies):
    """Gate on the cheap, always-on transformer score before running the heavier evidence-
    gathering + classification (which may call an LLM) -- matches the spec directly:
    "When a suspicious reading appears, Sentinel gathers [evidence for the 3-way
    classifier]." A quiet sensor shouldn't reach the classifier at all, not just be expected
    to score low there: running classify_rule_based unconditionally on plain normal noise
    every cycle for every sensor was exactly what produced a constant false-positive "attack"
    verdict during agent-wiring testing (the rule-based backstop is calibrated for
    genuinely suspicious evidence, not for "nothing happened").
    """

    def route_by_transformer_score(state: SentinelState) -> str:
        return "investigate" if state["transformer_score"] > deps.anomaly_threshold else "nominal"

    return route_by_transformer_score


def respond_nominal(state: SentinelState) -> dict:
    return {"response_action": "nominal", "log_messages": state.get("log_messages", [])}


def route_by_response(state: SentinelState) -> str:
    return state["response_action"]


def respond_fault(state: SentinelState) -> dict:
    msg = (
        f"[FAULT] {state['sensor_id']}: calibration correction + maintenance tag queued "
        "(STUB -- Member 1's healing function, Component 5, owns the actual correction)."
    )
    return {"log_messages": state.get("log_messages", []) + [msg]}


def respond_anomaly(state: SentinelState) -> dict:
    msg = f"[ANOMALY] {state['sensor_id']}: recorded for learning, no corrective action taken."
    return {"log_messages": state.get("log_messages", []) + [msg]}


def respond_attack(state: SentinelState) -> dict:
    msg = (
        f"[ATTACK] {state['sensor_id']}: flagged for forensics (handled by Overseer) and Guardian "
        "defense chain (STUB -- Member 3's CBF/Guardian, Component 3, owns actual command blocking)."
    )
    return {"log_messages": state.get("log_messages", []) + [msg]}


def build_sentinel_graph(deps: SentinelDependencies):
    graph = StateGraph(SentinelState)
    graph.add_node("score_residual", make_score_node(deps))
    graph.add_node("update_health", make_health_node(deps))
    graph.add_node("gather_evidence", make_evidence_node(deps))
    graph.add_node("classify", make_classify_node(deps))
    graph.add_node("respond_nominal", respond_nominal)
    graph.add_node("respond_fault", respond_fault)
    graph.add_node("respond_anomaly", respond_anomaly)
    graph.add_node("respond_attack", respond_attack)

    graph.set_entry_point("score_residual")
    graph.add_edge("score_residual", "update_health")
    graph.add_conditional_edges(
        "update_health",
        make_score_router(deps),
        {"investigate": "gather_evidence", "nominal": "respond_nominal"},
    )
    graph.add_edge("gather_evidence", "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_response,
        {
            RESPONSE_ACTIONS["fault"]: "respond_fault",
            RESPONSE_ACTIONS["anomaly"]: "respond_anomaly",
            RESPONSE_ACTIONS["attack"]: "respond_attack",
        },
    )
    graph.add_edge("respond_nominal", END)
    graph.add_edge("respond_fault", END)
    graph.add_edge("respond_anomaly", END)
    graph.add_edge("respond_attack", END)
    return graph.compile()
=======
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
>>>>>>> 67782d2bf638fc6e1aa240b226b523957b981a18
