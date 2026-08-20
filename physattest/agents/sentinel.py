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
