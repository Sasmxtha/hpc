"""
Demo: LLM Fallback Chain for three-way classification.

Shows the classifier handling three scenarios:
  1. Hardware fault   -- gradual degradation, neighbours fine
  2. Process anomaly  -- neighbours agree, command explains it
  3. Sensor attack    -- sudden change, neighbours disagree, fingerprint changed

Currently runs on Tier 3 (hardcoded rules). Install groq or
llama-cpp-python to activate higher tiers automatically.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security.llm_fallback import (
    LLMFallbackChain, SensorEvidence, Classification, LLMTier,
)


def safe_print(text):
    print(text.encode("ascii", errors="replace").decode("ascii"))


def demo():
    chain = LLMFallbackChain()

    print("=" * 65)
    print("  PhysAttest LLM Fallback Chain -- Three-Way Classifier")
    print("=" * 65)
    print(f"\n  Active tier: {chain.active_tier.value}")
    print(f"  Groq available: {chain.groq.available}")
    print(f"  Local LLM available: {chain.local.available}")
    print(f"  Rules always available: True")

    # ── Scenario 1: Hardware Fault ──
    print("\n" + "-" * 65)
    print("  SCENARIO 1: Hardware Fault (gradual degradation)")
    print("-" * 65)

    evidence_fault = SensorEvidence(
        sensor_id=3,
        residual=0.15,
        normalised_residual=6.2,
        health_score=35.0,
        health_trend=[95, 88, 80, 72, 60, 48, 35],
        neighbour_residuals={2: 0.8, 4: 1.1},
        neighbours_agree=False,
        recent_command=None,
        command_explains_change=False,
        fingerprint_status="authentic",
        fingerprint_confidence=0.85,
        probe_verdict="faulty",
        probe_correlation=0.45,
    )

    result = chain.classify(evidence_fault)
    print(f"\n  Classification: {result.classification.value}")
    print(f"  Confidence:     {result.confidence:.3f}")
    print(f"  Probabilities:  fault={result.probabilities['fault']:.3f}  "
          f"anomaly={result.probabilities['anomaly']:.3f}  "
          f"attack={result.probabilities['attack']:.3f}")
    safe_print(f"  Reasoning:      {result.reasoning}")
    print(f"  Source:         {result.source.value}")

    forensics = chain.forensics(evidence_fault, result.classification)
    safe_print(f"  Forensics:      {forensics}")

    # ── Scenario 2: Process Anomaly ──
    print("\n" + "-" * 65)
    print("  SCENARIO 2: Process Anomaly (legitimate event)")
    print("-" * 65)

    evidence_anomaly = SensorEvidence(
        sensor_id=1,
        residual=0.08,
        normalised_residual=3.5,
        health_score=92.0,
        health_trend=[100, 100, 99, 98, 95, 92],
        neighbour_residuals={0: 3.2, 2: 2.8},
        neighbours_agree=True,
        recent_command=[0.7, 0.3, 0.3, 0.5, 0.5, 0.5],
        command_explains_change=True,
        fingerprint_status="authentic",
        fingerprint_confidence=0.91,
        probe_verdict="honest",
        probe_correlation=0.95,
    )

    result = chain.classify(evidence_anomaly)
    print(f"\n  Classification: {result.classification.value}")
    print(f"  Confidence:     {result.confidence:.3f}")
    print(f"  Probabilities:  fault={result.probabilities['fault']:.3f}  "
          f"anomaly={result.probabilities['anomaly']:.3f}  "
          f"attack={result.probabilities['attack']:.3f}")
    safe_print(f"  Reasoning:      {result.reasoning}")
    print(f"  Source:         {result.source.value}")

    # ── Scenario 3: Sensor Attack ──
    print("\n" + "-" * 65)
    print("  SCENARIO 3: Sensor Attack (deliberate injection)")
    print("-" * 65)

    evidence_attack = SensorEvidence(
        sensor_id=5,
        residual=2.5,
        normalised_residual=45.0,
        health_score=95.0,
        health_trend=[100, 100, 100, 100, 95],
        neighbour_residuals={4: 0.5},
        neighbours_agree=False,
        recent_command=None,
        command_explains_change=False,
        fingerprint_status="compromised",
        fingerprint_confidence=0.12,
        probe_verdict="attacked",
        probe_correlation=0.02,
    )

    result = chain.classify(evidence_attack)
    print(f"\n  Classification: {result.classification.value}")
    print(f"  Confidence:     {result.confidence:.3f}")
    print(f"  Probabilities:  fault={result.probabilities['fault']:.3f}  "
          f"anomaly={result.probabilities['anomaly']:.3f}  "
          f"attack={result.probabilities['attack']:.3f}")
    safe_print(f"  Reasoning:      {result.reasoning}")
    print(f"  Source:         {result.source.value}")

    forensics = chain.forensics(evidence_attack, result.classification)
    safe_print(f"  Forensics:      {forensics}")

    # ── Scenario 4: Ambiguous case ──
    print("\n" + "-" * 65)
    print("  SCENARIO 4: Ambiguous (could be fault or attack)")
    print("-" * 65)

    evidence_ambiguous = SensorEvidence(
        sensor_id=0,
        residual=0.12,
        normalised_residual=5.5,
        health_score=68.0,
        health_trend=[90, 85, 82, 78, 72, 68],
        neighbour_residuals={1: 1.5, 3: 0.9},
        neighbours_agree=False,
        recent_command=None,
        command_explains_change=False,
        fingerprint_status="suspicious",
        fingerprint_confidence=0.45,
        probe_verdict="unknown",
        probe_correlation=0.0,
    )

    result = chain.classify(evidence_ambiguous)
    print(f"\n  Classification: {result.classification.value}")
    print(f"  Confidence:     {result.confidence:.3f}")
    print(f"  Probabilities:  fault={result.probabilities['fault']:.3f}  "
          f"anomaly={result.probabilities['anomaly']:.3f}  "
          f"attack={result.probabilities['attack']:.3f}")
    safe_print(f"  Reasoning:      {result.reasoning}")
    print(f"  Source:         {result.source.value}")

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    stats = chain.get_stats()
    print(f"  Active tier: {stats['active_tier']}")
    for tier, s in stats["tier_stats"].items():
        if s["calls"] > 0:
            print(f"  {tier}: {s['calls']} calls, {s['successes']} successes")

    print(f"\n  To upgrade tiers:")
    print(f"    Tier 1: pip install groq && set GROQ_API_KEY=your_key")
    print(f"    Tier 2: pip install llama-cpp-python")
    print(f"            + download TinyLlama GGUF to models/")
    print(f"\n  Detection pipeline (CBF, observer, probing) never depends")
    print(f"  on any LLM. The chain only enriches classification.")
    print("=" * 65)


if __name__ == "__main__":
    demo()
