"""Three-way fault / anomaly / attack classifier (Section 5 of the spec).

Sentinel calls this whenever a sensor's reading looks suspicious. It gathers four kinds of
evidence -- health score history, neighbour corroboration through the coupling graph,
whether a command log entry explains the change, and hardware fingerprint status -- and asks
"is this a degrading sensor, a real-but-unusual physical event, or a deliberate injection?"
The answer is a probability over all three, not a hard label, because the response should be
proportional: a 90%-confident fault gets a calibration correction, a 90%-confident attack
gets the full defense chain, and a genuinely ambiguous 40/35/25 split should not trigger
either extreme.

Classification itself goes through llm_fallback.FallbackChain (Groq -> TinyLlama); if both
are unavailable or return something unparseable, classify_rule_based() below is the final,
zero-dependency backstop -- a fixed, interpretable scoring function encoding the qualitative
rules from the spec directly (e.g. "healthy sensor + sudden onset + fingerprint changed +
no explaining command -> attack"), not a trained model.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from physattest.ai.llm_fallback import FallbackChain

CLASSES = ("fault", "anomaly", "attack")

RESPONSE_ACTIONS = {
    "fault": "calibration_correction_and_maintenance_tag",
    "anomaly": "record_and_learn",
    "attack": "full_defense_chain",
}

SYSTEM_PROMPT = """You are the classification component of PhysAttest, an industrial control
system security layer. Given evidence about one sensor's recent behaviour, decide the
probability that the underlying cause is a FAULT (degrading hardware), an ANOMALY (a real
but unusual physical event), or an ATTACK (deliberate data injection).

Important: the "hardware fingerprint" is a signature of ADC electrical noise and clock
jitter unique to a physical chip, not the sensor's readings. Ordinary wear or degradation
makes a sensor's READINGS noisier while its fingerprint stays the same (its silicon hasn't
been replaced). A CHANGED fingerprint means the physical hardware or firmware answering
requests is no longer the same one that was there before -- this is specifically evidence of
tampering or hardware substitution, i.e. an ATTACK indicator, not a wear/fault indicator.

Respond with ONLY a JSON object of the exact form {"fault": <0-1>, "anomaly": <0-1>,
"attack": <0-1>, "rationale": "<one sentence>"} where the three probabilities sum to 1.0.
No other text."""


@dataclass
class SensorEvidence:
    """What Sentinel gathers before asking for a classification.

    residual_magnitude: current physics-observer residual (Component 1), normalized to a
        noise-relative scale (e.g. |residual| / baseline noise std) so ~0 is unremarkable and
        several units is a clear physical inconsistency -- the caller owns that scaling.
    noise_level_change: (current_noise_std - baseline_noise_std) / baseline_noise_std.
    neighbour_corroboration: fraction (0-1) of this sensor's coupled neighbours (via
        Component 2's graph) that are ALSO currently flagged anomalous. Low = an isolated,
        single-sensor issue (points toward fault or attack); high = a correlated multi-sensor
        shift consistent with a real physical event propagating through the coupling graph
        (points toward anomaly). This is a distinct signal from residual_magnitude: residual
        asks "is this sensor's own reading physically consistent", corroboration asks "are
        other sensors seeing something unusual too, right now".
    command_explains_change: whether a recent logged agent command or known physical event
        (Component 4's bidirectional-shield command log) accounts for the change.
    fingerprint_changed: whether Component 8's ADC-noise/clock-jitter hardware signature has
        shifted, i.e. possible firmware tampering.
    onset_abruptness: 0 = gradual drift across many readings, 1 = an instantaneous step.
    """

    sensor_id: str
    health_score: float
    health_score_history: List[float]
    residual_magnitude: float
    noise_level_change: float
    neighbour_corroboration: float
    command_explains_change: bool
    fingerprint_changed: bool
    onset_abruptness: float


@dataclass
class ClassificationResult:
    probabilities: Dict[str, float]
    rationale: str
    backend_used: str
    response_action: str


def _health_trend(history: List[float]) -> float:
    """Slope of health score over recent history; negative = declining."""
    if len(history) < 2:
        return 0.0
    x = np.arange(len(history))
    return float(np.polyfit(x, history, 1)[0])


def build_prompt(evidence: SensorEvidence) -> str:
    trend = _health_trend(evidence.health_score_history)
    return f"""Sensor: {evidence.sensor_id}
Current health score (0-100, higher is healthier): {evidence.health_score:.1f}
Health score trend over recent history (points/reading, negative = declining): {trend:.3f}
Physics-observer residual magnitude (noise-relative, ~0 is normal): {evidence.residual_magnitude:.3f}
Noise level change vs. baseline: {evidence.noise_level_change:+.2%}
Neighbour corroboration via coupling graph (0=isolated to this sensor, 1=neighbours also flagged): {evidence.neighbour_corroboration:.2f}
Does a recent agent command or logged physical event explain this change?: {evidence.command_explains_change}
Hardware fingerprint changed (chip/firmware substituted, distinct from ordinary reading noise)?: {evidence.fingerprint_changed}
Onset abruptness (0=gradual drift, 1=instantaneous step change): {evidence.onset_abruptness:.2f}

Classify as fault / anomaly / attack."""


def _softmax(scores: np.ndarray) -> np.ndarray:
    z = scores - scores.max()
    e = np.exp(z)
    return e / e.sum()


def classify_rule_based(evidence: SensorEvidence) -> Tuple[Dict[str, float], str]:
    """Zero-dependency final fallback. Fixed, hand-picked weights encoding the spec's
    qualitative rules as a soft scoring function rather than a brittle if/else chain, so it
    still produces a genuine probability distribution (needed by decide_response's
    confidence threshold) rather than a single hard branch.
    """
    trend = _health_trend(evidence.health_score_history)

    # A declining health-score trend is only fault evidence when TODAY's event is itself
    # gradual. Health score is itself derived from the same anomaly signal driving this
    # classification, so under a SUSTAINED attack, health keeps dropping cycle over cycle,
    # which would otherwise make the trend term grow without bound and eventually swamp
    # onset_abruptness -- flipping a multi-cycle attack's classification from attack to
    # fault purely because it lasted several cycles, exactly backwards from the spec's
    # intent. Gating by (1 - onset_abruptness) and capping the raw magnitude both guard
    # against that: this term should fire for "quietly degrading over time", not for
    # "repeatedly hit by abrupt events that happen to also drag the trend down."
    declining_trend = min(max(0.0, -trend), 5.0)
    fault_score = (
        declining_trend * 2.0 * (1 - evidence.onset_abruptness)
        + max(0.0, evidence.noise_level_change) * 1.5
        + (1 - evidence.onset_abruptness) * 1.0
        + (0.0 if evidence.fingerprint_changed else 1.0)
        + (1 - evidence.neighbour_corroboration) * 0.5
    )

    anomaly_score = (
        evidence.neighbour_corroboration * 2.5
        + (2.0 if evidence.command_explains_change else 0.0)
        + (0.5 if not evidence.fingerprint_changed else 0.0)
    )

    # "Was healthy before", "isolated from neighbours", and "no explaining command" are only
    # meaningful AS ATTACK EVIDENCE once something has actually happened -- a quiet sensor
    # with nothing going on is trivially "isolated" and "unexplained by any command" too, and
    # treating those as standalone attack signals produced a constant false-positive "attack"
    # verdict on plain normal noise during agent-wiring testing (every sensor, every cycle,
    # regardless of whether anything was actually wrong). Gating them by onset_abruptness
    # means they amplify an already-present abrupt change rather than manufacturing a signal
    # out of a boring cycle. fingerprint_changed and residual_magnitude are left ungated: a
    # changed hardware fingerprint or a genuinely large tail-vs-baseline shift are each
    # meaningful on their own, independent of how abrupt the onset looked.
    attack_score = (
        (evidence.health_score / 100.0) * evidence.onset_abruptness * 1.0
        + evidence.onset_abruptness * 2.0
        + (1 - evidence.neighbour_corroboration) * evidence.onset_abruptness * 1.5
        + (0.0 if evidence.command_explains_change else evidence.onset_abruptness * 1.5)
        + (2.0 if evidence.fingerprint_changed else 0.0)
        + min(evidence.residual_magnitude, 5.0) * 0.6
    )

    scores = np.array([fault_score, anomaly_score, attack_score])
    probs = _softmax(scores)
    probabilities = {CLASSES[i]: float(probs[i]) for i in range(3)}

    top_class = max(probabilities, key=probabilities.get)
    reasons = {
        "fault": "declining health trend and increased noise with neighbours unaffected",
        "anomaly": "neighbours also show correlated changes, consistent with a real shared event",
        "attack": "isolated, abrupt change unexplained by any command"
        + (" with a shifted hardware fingerprint" if evidence.fingerprint_changed else ""),
    }
    rationale = f"rule-based: {reasons[top_class]} (scores: fault={fault_score:.2f}, anomaly={anomaly_score:.2f}, attack={attack_score:.2f})"

    return probabilities, rationale


def _validate_and_normalize_probs(data: dict) -> Dict[str, float]:
    probs = {c: float(data[c]) for c in CLASSES}  # KeyError/TypeError/ValueError -> caller falls through
    if any(p < 0 for p in probs.values()):
        raise ValueError("negative probability in LLM response")
    total = sum(probs.values())
    if not (0.5 < total < 1.5):
        raise ValueError(f"probabilities do not sum near 1.0: {total}")
    return {c: p / total for c, p in probs.items()}


def decide_response(probabilities: Dict[str, float], confidence_threshold: float = 0.4) -> str:
    """Argmax class -> proportional response, per Section 5:
        fault -> calibration correction + maintenance tag
        anomaly -> record and learn
        attack -> full defense chain

    If the winning class's probability is below confidence_threshold (no class clearly
    stands out), default to the anomaly response. That's the response that neither relaxes
    monitoring (fault's calibration correction, which stops treating the sensor as suspect)
    nor commits to the heaviest response (attack's full defense chain) when the evidence
    genuinely doesn't support either strongly -- "record and learn" keeps watching without
    overreacting or underreacting.
    """
    top_class = max(probabilities, key=probabilities.get)
    if probabilities[top_class] < confidence_threshold:
        return RESPONSE_ACTIONS["anomaly"]
    return RESPONSE_ACTIONS[top_class]


def classify(evidence: SensorEvidence, chain: Optional[FallbackChain] = None) -> ClassificationResult:
    """Top-level entry point Sentinel calls. Tries the LLM chain (if provided) first, falls
    through to classify_rule_based on any failure -- missing credentials, no network, no
    local model, or a response that fails validation.
    """
    probabilities = None
    rationale = ""
    backend_used = "rule_based"

    if chain is not None:
        try:
            prompt = build_prompt(evidence)
            result = chain.run(prompt, system=SYSTEM_PROMPT)
            probabilities = _validate_and_normalize_probs(result.data)
            rationale = str(result.data.get("rationale", ""))
            backend_used = result.backend_used
        except (RuntimeError, ValueError, KeyError, TypeError):
            probabilities = None

    if probabilities is None:
        probabilities, rationale = classify_rule_based(evidence)
        backend_used = "rule_based"

    response_action = decide_response(probabilities)
    return ClassificationResult(
        probabilities=probabilities,
        rationale=rationale,
        backend_used=backend_used,
        response_action=response_action,
    )


if __name__ == "__main__":
    # Three hand-built scenarios, one per class, to sanity-check the rule-based fallback
    # actually separates them the way the spec describes -- run with no chain, so this
    # exercises exactly the zero-dependency backstop path.
    scenarios = {
        "fault": SensorEvidence(
            sensor_id="temp_12",
            health_score=62.0,
            health_score_history=[88, 84, 80, 76, 71, 66, 62],  # gradual decline
            residual_magnitude=0.8,
            noise_level_change=0.35,  # noisier than baseline
            neighbour_corroboration=0.05,  # neighbours unaffected
            command_explains_change=False,
            fingerprint_changed=False,  # same hardware, just noisier
            onset_abruptness=0.1,  # gradual
        ),
        "anomaly": SensorEvidence(
            sensor_id="pressure_7",
            health_score=95.0,
            health_score_history=[96, 95, 96, 95, 95, 96, 95],  # stable
            residual_magnitude=1.2,
            noise_level_change=0.05,
            neighbour_corroboration=0.85,  # neighbours also shifted
            command_explains_change=True,  # e.g. a valve command explains it
            fingerprint_changed=False,
            onset_abruptness=0.6,
        ),
        "attack": SensorEvidence(
            sensor_id="chlorine_3",
            health_score=98.0,
            health_score_history=[97, 98, 98, 97, 98, 98, 98],  # was perfect
            residual_magnitude=4.5,  # large physical inconsistency
            noise_level_change=0.02,
            neighbour_corroboration=0.05,  # isolated, neighbours disagree
            command_explains_change=False,  # no explanation
            fingerprint_changed=True,  # hardware signature shifted
            onset_abruptness=0.95,  # sudden step
        ),
    }

    all_correct = True
    for expected_class, evidence in scenarios.items():
        result = classify(evidence, chain=None)
        predicted = max(result.probabilities, key=result.probabilities.get)
        ok = predicted == expected_class
        all_correct &= ok
        print(
            f"[{'OK' if ok else 'MISS'}] expected={expected_class:8s} predicted={predicted:8s} "
            f"probs={ {k: round(v, 3) for k, v in result.probabilities.items()} } "
            f"response={result.response_action}"
        )
        print(f"         rationale: {result.rationale}")

    assert all_correct, "rule-based classifier failed to separate the three scenarios"

    # Regression case: genuinely boring evidence -- nothing declining, nothing abrupt,
    # nothing unexplained in any interesting sense, just a quiet sensor. This is what
    # agent-wiring testing against real synthetic noise exposed as broken: the classifier
    # was confidently calling this ATTACK on every cycle, because several attack-score terms
    # were unconditional flat bonuses ("no explaining command", "was healthy") that fire on
    # any boring reading, not just a suspicious one. Must not confidently call this attack.
    quiet = SensorEvidence(
        sensor_id="flow_9",
        health_score=100.0,
        health_score_history=[100, 100, 99, 100, 100, 100, 100],
        residual_magnitude=0.3,
        noise_level_change=0.0,
        neighbour_corroboration=0.1,
        command_explains_change=False,
        fingerprint_changed=False,
        onset_abruptness=0.05,
    )
    quiet_result = classify(quiet, chain=None)
    quiet_predicted = max(quiet_result.probabilities, key=quiet_result.probabilities.get)
    print(
        f"\n[{'OK' if quiet_predicted != 'attack' else 'MISS'}] quiet baseline -> predicted={quiet_predicted:8s} "
        f"probs={ {k: round(v, 3) for k, v in quiet_result.probabilities.items()} }"
    )
    assert quiet_predicted != "attack", "quiet/normal evidence must not be classified as attack"

    print("\nclassifier_3way.py smoke test passed")
