"""LLM forensics engine: runs once the 3-way classifier has confirmed an ATTACK.

Three jobs, per the spec: classify the attack type, infer attacker intent by tracing
consequences through the plant's causal graph (the water-treatment example: "sensor 7 pushed
low -> agent increases dosing -> actual chlorine already normal -> over-chlorination -> water
poisoned"), and plan a strategic response -- simple attacks get simple fixes, coordinated
attacks get aggressive preemptive defense.

This module owns the LLM-facing synthesis; it does not own the plant's actual causal
knowledge (which sensor feeds which control decision, which actuator affects which
downstream variable) -- that is plant-specific integration data Member 1/3 supply by
building a networkx.DiGraph following the small node/edge convention documented on
CausalNode/CausalEdge below. build_demo_water_plant_graph() exists only to make this module
runnable and testable today with the spec's own chlorine example, standing in for that real
graph until it exists.

Uses the same physattest.ai.llm_fallback.FallbackChain as classifier_3way.py (Groq ->
TinyLlama -> rule-based backstop here). The causal-chain narrative and a baseline severity /
response tier are always produced deterministically from the graph, independent of any LLM
-- the LLM's job is to turn that structured evidence into a fluent intent inference and to
refine severity/response, not to invent the causal chain itself. If the LLM is unavailable,
the deterministic version is a complete, if less articulate, forensics report on its own.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from physattest.ai.llm_fallback import FallbackChain

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

RESPONSE_TIERS = {
    "targeted": "targeted_response",
    "aggressive": "aggressive_preemptive_defense",
}


# ---------------------------------------------------------------------------
# Causal graph
# ---------------------------------------------------------------------------
#
# Convention (any nx.DiGraph following this shape works, not just the demo below):
#   node attrs:  kind in {"sensor", "decision", "actuator", "consequence"},
#                label: str, static plant-knowledge description of this node,
#                severity: str in SEVERITY_ORDER, only meaningful on "consequence" nodes.
#   edge attrs:  relation: str, short description of the causal link (currently unused by
#                rendering beyond documentation -- narrative text comes from node labels plus
#                per-event detail, since that reads more naturally than edge-verb chaining).


def add_causal_link(
    graph: nx.DiGraph,
    cause_id: str,
    effect_id: str,
    relation: str,
    cause_kind: Optional[str] = None,
    cause_label: Optional[str] = None,
    effect_kind: Optional[str] = None,
    effect_label: Optional[str] = None,
    effect_severity: Optional[str] = None,
) -> None:
    """Adds one causal edge, creating either endpoint node if it doesn't already exist.
    Convenience for building a plant's causal graph incrementally without pre-declaring nodes.
    """
    if cause_id not in graph:
        graph.add_node(cause_id, kind=cause_kind or "event", label=cause_label or cause_id)
    if effect_id not in graph:
        graph.add_node(
            effect_id, kind=effect_kind or "event", label=effect_label or effect_id, severity=effect_severity or "medium"
        )
    graph.add_edge(cause_id, effect_id, relation=relation)


def build_demo_water_plant_graph() -> nx.DiGraph:
    """The spec's own chlorine-dosing example, wired up as an actual causal graph -- exists
    to make this module testable before the real plant graph exists (see module docstring).
    """
    g = nx.DiGraph()
    add_causal_link(
        g,
        "sensor_7_chlorine",
        "decision_increase_dosing",
        relation="triggers",
        cause_kind="sensor",
        cause_label="Sensor 7 (chlorine) reading",
        effect_kind="decision",
        effect_label="Agent increases chlorine dosing",
    )
    add_causal_link(
        g,
        "decision_increase_dosing",
        "actuator_dosing_pump",
        relation="commands",
        effect_kind="actuator",
        effect_label="Dosing pump output raised",
    )
    add_causal_link(
        g,
        "actuator_dosing_pump",
        "consequence_over_chlorination",
        relation="causes",
        effect_kind="consequence",
        effect_label="Over-chlorination of treated water",
        effect_severity="high",
    )
    add_causal_link(
        g,
        "consequence_over_chlorination",
        "consequence_water_poisoned",
        relation="causes",
        effect_kind="consequence",
        effect_label="Water supply rendered unsafe to consume",
        effect_severity="critical",
    )
    return g


def trace_causal_paths(graph: nx.DiGraph, start_node: str, max_depth: int = 6) -> List[List[str]]:
    """All simple paths from start_node to every downstream sink (out-degree 0) node,
    capped at max_depth hops. Real plant causal graphs are small, near-linear DAGs, so this
    stays cheap; the cap is defensive insurance against an unexpectedly dense or cyclic graph.
    """
    if start_node not in graph:
        return []
    descendants = nx.descendants(graph, start_node)
    sinks = [n for n in descendants if graph.out_degree(n) == 0]
    if not sinks:
        return [[start_node]]
    paths = []
    for sink in sinks:
        paths.extend(nx.all_simple_paths(graph, start_node, sink, cutoff=max_depth))
    return paths


def render_narrative(graph: nx.DiGraph, path: List[str], event_context: Optional[Dict[str, str]] = None) -> str:
    """Turns a causal path into the spec's arrow-chain narrative style, e.g.
    "Sensor 7 (chlorine) reading (reported 0.8 mg/L...) -> Agent increases chlorine dosing -> ...".

    event_context maps node_id -> a specific-to-this-event detail string (e.g. the actual
    reported vs. physics-reconstructed value), appended to that node's static label. This is
    where "actual chlorine already normal" from the spec's example enters: it is a fact about
    THIS event (from Component 5's reconstruction), not a property of the sensor node itself.
    """
    event_context = event_context or {}
    parts = []
    for node_id in path:
        label = graph.nodes[node_id].get("label", node_id)
        detail = event_context.get(node_id)
        parts.append(f"{label} ({detail})" if detail else label)
    return " -> ".join(parts)


def _path_max_severity(graph: nx.DiGraph, path: List[str]) -> str:
    best = "low"
    for node_id in path:
        if graph.nodes[node_id].get("kind") == "consequence":
            sev = graph.nodes[node_id].get("severity", "medium")
            if SEVERITY_ORDER.get(sev, 1) > SEVERITY_ORDER.get(best, 0):
                best = sev
    return best


# ---------------------------------------------------------------------------
# Attack-pattern classification (deterministic, from the residual/score time series)
# ---------------------------------------------------------------------------


def classify_attack_pattern(residual_series: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """Classifies the SHAPE of a compromised sensor's residual (or Component 6's per-
    timestep anomaly score) trajectory into step / ramp / oscillation / impulse. Deliberately
    matches the synthetic attack kinds in physattest/ml/data_synth.py (step, ramp,
    oscillation) plus impulse (a spike that returns to baseline), so forensics output during
    testing is directly comparable to ground truth, and so the pattern name means the same
    thing here as it does everywhere else the attack taxonomy is used in this project.

    Deterministic feature thresholds, not learned -- this is a cheap, always-available signal
    the LLM step below gets to use as evidence, not something that itself needs an LLM.
    """
    x = np.asarray(residual_series, dtype=float)
    n = len(x)
    if n < 4:
        return "unknown", {}

    edge = max(1, n // 5)
    baseline = float(np.median(x[:edge]))
    tail = float(np.median(x[-edge:]))
    peak_idx = int(np.argmax(np.abs(x - baseline)))
    peak_dev = float(x[peak_idx] - baseline)
    span = float(x.max() - x.min()) + 1e-9

    returns_to_baseline = abs(tail - baseline) < 0.3 * abs(peak_dev) + 1e-9
    is_mid_window = 0.15 * n < peak_idx < 0.85 * n

    # Oscillation score: strength of the strongest LOCAL MAXIMUM in the autocorrelation at
    # a lag > 1. A step or ramp's autocorrelation decays smoothly/monotonically with lag (no
    # local peak -- each point stays highly correlated with its neighbours, decreasingly so);
    # a genuinely periodic signal has a dip-then-rise, producing a real local peak at the
    # period's lag. Just thresholding the raw autocorrelation values (without requiring a
    # local peak) mistakes a step's slowly-decaying-but-still-high autocorrelation for
    # periodicity, which is what an earlier version of this function got wrong.
    centered = x - x.mean()
    autocorr_full = np.correlate(centered, centered, mode="full")
    autocorr = autocorr_full[n - 1 :]
    autocorr = autocorr / (autocorr[0] + 1e-9)
    oscillation_score = 0.0
    search_end = max(3, min(n // 2, len(autocorr) - 1))
    for lag in range(2, search_end):
        if autocorr[lag] > autocorr[lag - 1] and autocorr[lag] > autocorr[lag + 1]:
            oscillation_score = max(oscillation_score, float(autocorr[lag]))

    diffs = np.diff(x)
    net_change = tail - baseline
    monotonic_fraction = (
        float(np.mean(np.sign(diffs) == np.sign(net_change))) if abs(net_change) > 1e-9 else 0.0
    )
    max_step = float(np.abs(diffs).max()) if n > 1 else 0.0

    features = {
        "peak_deviation": peak_dev,
        "oscillation_score": oscillation_score,
        "monotonic_fraction": monotonic_fraction,
        "max_single_step": max_step,
        "span": span,
    }

    if oscillation_score > 0.5:
        kind = "oscillation"
    elif returns_to_baseline and is_mid_window:
        kind = "impulse"
    elif monotonic_fraction > 0.7 and max_step < 0.5 * span:
        kind = "ramp"
    else:
        kind = "step"

    return kind, features


def assess_coordination(concurrently_attacked_sensors: List[str], primary_sensor: str) -> str:
    """"coordinated" if any OTHER sensor is also currently classified as under attack,
    "isolated" otherwise. concurrently_attacked_sensors comes from running classifier_3way
    across all sensors in the current cycle, not from this module -- forensics doesn't decide
    what's attacked, it explains an attack Sentinel already confirmed.
    """
    others = [s for s in concurrently_attacked_sensors if s != primary_sensor]
    return "coordinated" if others else "isolated"


def _recommended_response(coordination: str, severity: str) -> str:
    """Coordinated attacks always escalate, per the spec. A single-sensor attack whose
    causal chain nonetheless reaches a "high"/"critical" consequence node also escalates --
    an isolated attack on the chlorine sensor that leads to poisoned water deserves the
    aggressive response just as much as a multi-sensor one does, so severity is treated as an
    independent escalation trigger rather than deferring entirely to coordination.
    """
    if coordination == "coordinated" or severity in ("high", "critical"):
        return RESPONSE_TIERS["aggressive"]
    return RESPONSE_TIERS["targeted"]


# ---------------------------------------------------------------------------
# Evidence bundle + report
# ---------------------------------------------------------------------------


@dataclass
class ForensicsEvidence:
    sensor_id: str
    residual_series: np.ndarray
    event_context: Dict[str, str] = field(default_factory=dict)
    concurrently_attacked_sensors: List[str] = field(default_factory=list)
    classifier_confidence: float = 1.0  # the 3-way classifier's attack-class probability


@dataclass
class ForensicsReport:
    attack_pattern: str
    pattern_features: Dict[str, float]
    causal_narratives: List[str]
    coordination: str
    severity: str
    recommended_response: str
    inferred_intent: str
    rationale: str
    backend_used: str


FORENSICS_SYSTEM_PROMPT = """You are the forensics component of PhysAttest, an industrial
control system security layer. An attack has already been CONFIRMED by an upstream
classifier -- your job is not to decide whether it's an attack, but to explain it: infer
what the attacker was trying to achieve by reasoning through the given causal chain(s) from
the compromised sensor to their physical consequences, and refine the severity and
recommended response.

Respond with ONLY a JSON object of the exact form:
{"inferred_intent": "<1-2 sentences tracing the likely goal through the causal chain>",
 "severity": "low"|"medium"|"high"|"critical",
 "recommended_response": "targeted_response"|"aggressive_preemptive_defense",
 "rationale": "<one sentence justifying severity and response>"}
No other text."""


def build_forensics_prompt(
    evidence: ForensicsEvidence, attack_pattern: str, narratives: List[str], coordination: str, baseline_severity: str
) -> str:
    narrative_block = "\n".join(f"- {n}" for n in narratives) if narratives else "(no causal chain available)"
    return f"""Compromised sensor: {evidence.sensor_id}
Attack pattern (from residual shape analysis): {attack_pattern}
Classifier confidence this is an attack: {evidence.classifier_confidence:.2f}
Coordination: {coordination} ({"other sensors also currently flagged" if coordination == "coordinated" else "this sensor only"})
Deterministic baseline severity (from the causal graph alone): {baseline_severity}

Causal chain(s) from sensor to consequence:
{narrative_block}

Infer the attacker's intent and finalize severity/response."""


def _forensics_rule_based(
    attack_pattern: str, narratives: List[str], coordination: str, baseline_severity: str
) -> Tuple[str, str, str, str]:
    """Zero-dependency final fallback. Severity and response come straight from the causal
    graph + coordination (see _recommended_response) with no LLM needed; only the intent
    sentence is templated rather than freely composed.
    """
    if narratives:
        intent = f"Likely goal, inferred from the causal chain: {narratives[0]}"
    else:
        intent = f"A {attack_pattern} pattern was injected into {attack_pattern and 'the sensor'}; no causal chain was available to infer downstream intent."
    response = _recommended_response(coordination, baseline_severity)
    rationale = (
        f"rule-based: severity set from the causal graph's worst reachable consequence "
        f"({baseline_severity}), response escalated because coordination={coordination}"
    )
    return intent, baseline_severity, response, rationale


def run_forensics(
    evidence: ForensicsEvidence, graph: nx.DiGraph, chain: Optional[FallbackChain] = None
) -> ForensicsReport:
    """Top-level entry point, called once the 3-way classifier confirms an attack.

    The causal narrative, pattern classification, coordination, and baseline severity are
    always computed deterministically first (cheap, no LLM). The LLM chain, if provided, is
    then asked to turn that into an intent inference and to have the final say on
    severity/response; if it's unavailable or returns something invalid, the deterministic
    baseline computed a moment ago IS the final answer, not a degraded stand-in for one.
    """
    attack_pattern, features = classify_attack_pattern(evidence.residual_series)

    paths = trace_causal_paths(graph, evidence.sensor_id)
    narratives = [render_narrative(graph, path, evidence.event_context) for path in paths]
    baseline_severity = "low"
    for path in paths:
        sev = _path_max_severity(graph, path)
        if SEVERITY_ORDER[sev] > SEVERITY_ORDER[baseline_severity]:
            baseline_severity = sev

    coordination = assess_coordination(evidence.concurrently_attacked_sensors, evidence.sensor_id)

    intent, severity, response, rationale, backend_used = None, None, None, None, "rule_based"

    if chain is not None:
        try:
            prompt = build_forensics_prompt(evidence, attack_pattern, narratives, coordination, baseline_severity)
            result = chain.run(prompt, system=FORENSICS_SYSTEM_PROMPT)
            data = result.data
            severity = str(data["severity"])
            response = str(data["recommended_response"])
            intent = str(data["inferred_intent"])
            rationale = str(data.get("rationale", ""))
            if severity not in SEVERITY_ORDER or response not in RESPONSE_TIERS.values():
                raise ValueError(f"invalid severity/response in LLM output: {severity}, {response}")
            backend_used = result.backend_used
        except (RuntimeError, ValueError, KeyError, TypeError):
            intent, severity, response, rationale, backend_used = None, None, None, None, "rule_based"

    if intent is None:
        intent, severity, response, rationale = _forensics_rule_based(
            attack_pattern, narratives, coordination, baseline_severity
        )
        backend_used = "rule_based"

    return ForensicsReport(
        attack_pattern=attack_pattern,
        pattern_features=features,
        causal_narratives=narratives,
        coordination=coordination,
        severity=severity,
        recommended_response=response,
        inferred_intent=intent,
        rationale=rationale,
        backend_used=backend_used,
    )
