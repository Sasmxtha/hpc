"""
LLM Fallback Chain for PhysAttest.

Three-tier chain with graceful degradation:

  1. Groq API (primary)   -- Llama 3 70B, full forensics + reasoning
  2. TinyLlama local (secondary) -- llama-cpp-python, CPU, basic classification
  3. Hardcoded rules (final)     -- deterministic, zero dependencies

The detection pipeline (observer, CBF, probing) NEVER depends on any LLM.
The LLM only enriches classification and forensics -- if all three tiers
fail, the system still detects and blocks attacks via pure math.

Usage:
    chain = LLMFallbackChain()
    result = chain.classify(evidence)
    # result.source tells you which tier answered
"""

import os
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


class LLMTier(Enum):
    GROQ = "groq"
    LOCAL = "local_tinyllama"
    RULES = "hardcoded_rules"
    UNAVAILABLE = "unavailable"


class Classification(Enum):
    FAULT = "fault"
    ANOMALY = "anomaly"
    ATTACK = "attack"


@dataclass
class SensorEvidence:
    """All evidence about a suspicious sensor, gathered by Sentinel."""
    sensor_id: int
    residual: float
    normalised_residual: float
    health_score: float
    health_trend: list[float] = field(default_factory=list)
    neighbour_residuals: dict[int, float] = field(default_factory=dict)
    neighbours_agree: bool = False
    recent_command: Optional[list[float]] = None
    command_explains_change: bool = False
    fingerprint_status: str = "unknown"
    fingerprint_confidence: float = 0.5
    probe_verdict: str = "unknown"
    probe_correlation: float = 0.0


@dataclass
class ClassificationResult:
    """Output of the three-way classifier."""
    classification: Classification
    confidence: float
    probabilities: dict[str, float]
    reasoning: str
    source: LLMTier
    forensics: str = ""


# System prompt shared by Groq and local tiers
SYSTEM_PROMPT = """You are a sensor forensics classifier for an industrial control system (water treatment plant).

Given evidence about a suspicious sensor, classify it as exactly one of:
- FAULT: sensor hardware degrading. Gradual health decline, noise increasing, neighbours unaffected, same hardware fingerprint.
- ANOMALY: real physical event. Neighbours also changed, a command or process event explains the change.
- ATTACK: deliberate injection. Sudden change, health was fine, neighbours disagree, no command explains it, fingerprint may have changed.

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{
  "classification": "fault" | "anomaly" | "attack",
  "confidence": 0.0-1.0,
  "probabilities": {"fault": 0.0-1.0, "anomaly": 0.0-1.0, "attack": 0.0-1.0},
  "reasoning": "one sentence explaining why"
}"""

FORENSICS_PROMPT = """You are an attack forensics analyst for a water treatment plant.

Given the attack evidence, trace the attacker's intent through the plant's causal graph. Explain:
1. What sensor was targeted and how
2. What the attacker wanted the control agent to do as a result
3. What physical consequence the attacker intended
4. Whether the attack was simple (single sensor) or coordinated

Respond with a concise 2-3 sentence forensics report."""


def _build_evidence_prompt(evidence: SensorEvidence) -> str:
    """Format sensor evidence into a prompt for the LLM."""
    lines = [
        f"Sensor {evidence.sensor_id} flagged as suspicious.",
        f"Residual: {evidence.normalised_residual:.2f} sigma (threshold = 2.5-5.0).",
        f"Health score: {evidence.health_score:.1f}/100.",
    ]
    if evidence.health_trend:
        trend_str = " -> ".join(f"{h:.0f}" for h in evidence.health_trend[-5:])
        lines.append(f"Health trend (recent): {trend_str}.")

    if evidence.neighbour_residuals:
        nb = ", ".join(f"s{k}={v:.2f}sigma" for k, v in evidence.neighbour_residuals.items())
        lines.append(f"Neighbour residuals: {nb}.")
        lines.append(f"Neighbours agree (also anomalous): {evidence.neighbours_agree}.")
    else:
        lines.append("No coupled neighbours available.")

    if evidence.recent_command is not None:
        lines.append(f"Recent command: {[round(c, 3) for c in evidence.recent_command]}.")
        lines.append(f"Command explains the sensor change: {evidence.command_explains_change}.")

    lines.append(f"Hardware fingerprint: {evidence.fingerprint_status} "
                 f"(confidence {evidence.fingerprint_confidence:.2f}).")

    if evidence.probe_verdict != "unknown":
        lines.append(f"Active probe verdict: {evidence.probe_verdict} "
                     f"(correlation {evidence.probe_correlation:.3f}).")

    return "\n".join(lines)


# =================================================================
#  TIER 1: Groq API
# =================================================================

class GroqClassifier:
    """Tier 1: Groq API with Llama 3 70B."""

    def __init__(self):
        self._client = None
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import groq
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                self._available = False
                return False
            self._client = groq.Groq(api_key=api_key)
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def classify(self, evidence: SensorEvidence) -> Optional[ClassificationResult]:
        if not self.available:
            return None
        try:
            user_msg = _build_evidence_prompt(evidence)
            response = self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()
            data = json.loads(text)

            return ClassificationResult(
                classification=Classification(data["classification"]),
                confidence=float(data["confidence"]),
                probabilities=data["probabilities"],
                reasoning=data["reasoning"],
                source=LLMTier.GROQ,
            )
        except Exception as e:
            log.warning("Groq classify failed: %s", e)
            return None

    def forensics(self, evidence: SensorEvidence, attack_context: str) -> Optional[str]:
        if not self.available:
            return None
        try:
            user_msg = (
                f"{_build_evidence_prompt(evidence)}\n\n"
                f"Attack context: {attack_context}"
            )
            response = self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": FORENSICS_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            log.warning("Groq forensics failed: %s", e)
            return None


# =================================================================
#  TIER 2: Local TinyLlama via llama-cpp-python
# =================================================================

class LocalLlamaClassifier:
    """Tier 2: TinyLlama 1.1B Q4 running locally on CPU."""

    MODEL_PATHS = [
        "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        os.path.expanduser("~/.cache/physattest/tinyllama.gguf"),
    ]

    def __init__(self):
        self._llm = None
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from llama_cpp import Llama
            for path in self.MODEL_PATHS:
                if os.path.exists(path):
                    self._llm = Llama(
                        model_path=path,
                        n_ctx=1024,
                        n_threads=4,
                        verbose=False,
                    )
                    self._available = True
                    return True
            self._available = False
        except ImportError:
            self._available = False
        return self._available

    # Simpler prompt that small models can follow
    _CLASSIFY_PROMPT = (
        "Classify this sensor event as FAULT, ANOMALY, or ATTACK. "
        "Reply with ONE word only: FAULT, ANOMALY, or ATTACK.\n\n"
    )

    def classify(self, evidence: SensorEvidence) -> Optional[ClassificationResult]:
        if not self.available:
            return None
        try:
            user_msg = _build_evidence_prompt(evidence)
            prompt = (
                f"<|system|>\nYou classify sensor events. "
                f"Reply with exactly one word: FAULT, ANOMALY, or ATTACK.</s>\n"
                f"<|user|>\n{user_msg}\n\n"
                f"Classification (one word):</s>\n"
                f"<|assistant|>\n"
            )
            output = self._llm(prompt, max_tokens=20, temperature=0.1, stop=["</s>", "\n", "."])
            text = output["choices"][0]["text"].strip().upper()

            # Try JSON first (in case model cooperates)
            if "{" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if end > start:
                    data = json.loads(text[start:end])
                    return ClassificationResult(
                        classification=Classification(data.get("classification", "attack").lower()),
                        confidence=float(data.get("confidence", 0.6)),
                        probabilities=data.get("probabilities", {}),
                        reasoning=data.get("reasoning", "local model"),
                        source=LLMTier.LOCAL,
                    )

            # Keyword extraction from free-text response
            for label in ["ATTACK", "FAULT", "ANOMALY"]:
                if label in text:
                    return ClassificationResult(
                        classification=Classification(label.lower()),
                        confidence=0.6,
                        probabilities={label.lower(): 0.6},
                        reasoning=f"local model keyword: {label}",
                        source=LLMTier.LOCAL,
                    )
        except Exception as e:
            log.warning("Local LLM classify failed: %s", e)
        return None


# =================================================================
#  TIER 3: Hardcoded Rules
# =================================================================

class RuleBasedClassifier:
    """
    Tier 3: Deterministic if-else rules. Always available.

    Rules encode the domain knowledge from the project spec:
      - residual high + fingerprint changed + no command = ATTACK
      - health dropping gradually + neighbours fine = FAULT
      - neighbours also changed + command explains it = ANOMALY
    """

    def classify(self, evidence: SensorEvidence) -> ClassificationResult:
        e = evidence

        p_fault = 0.0
        p_anomaly = 0.0
        p_attack = 0.0

        # --- Rule 1: Fingerprint status ---
        if e.fingerprint_status == "compromised":
            p_attack += 0.4
        elif e.fingerprint_status == "suspicious":
            p_attack += 0.15
            p_fault += 0.1

        # --- Rule 2: Health trend ---
        if len(e.health_trend) >= 3:
            deltas = [e.health_trend[i+1] - e.health_trend[i]
                      for i in range(len(e.health_trend) - 1)]
            avg_delta = sum(deltas) / len(deltas)
            if avg_delta < -3.0:
                # Gradual decline = fault
                p_fault += 0.3
            elif avg_delta < -8.0:
                # Sharp drop = attack
                p_attack += 0.2
        if e.health_score > 90:
            # Health was fine -> sudden change more likely attack
            p_attack += 0.15
        elif e.health_score < 50:
            p_fault += 0.2

        # --- Rule 3: Neighbour agreement ---
        if e.neighbours_agree:
            # Neighbours also anomalous -> real event or coordinated attack
            p_anomaly += 0.3
            if e.normalised_residual > 10.0:
                # But if residual is extremely high, coordinated attack
                p_attack += 0.15
        else:
            # Neighbours fine but this sensor is off -> attack or fault
            p_attack += 0.15

        # --- Rule 4: Command explains change ---
        if e.command_explains_change:
            p_anomaly += 0.35
            p_attack -= 0.1

        # --- Rule 5: Active probe verdict ---
        if e.probe_verdict == "attacked":
            p_attack += 0.35
        elif e.probe_verdict == "faulty":
            p_fault += 0.25
        elif e.probe_verdict == "honest":
            p_anomaly += 0.2
            p_attack -= 0.15

        # --- Rule 6: Residual magnitude ---
        if e.normalised_residual > 10.0:
            p_attack += 0.15
        elif e.normalised_residual > 5.0:
            p_attack += 0.05

        # --- Normalise ---
        p_fault = max(0, p_fault)
        p_anomaly = max(0, p_anomaly)
        p_attack = max(0, p_attack)
        total = p_fault + p_anomaly + p_attack

        if total < 0.01:
            p_anomaly = 1.0
            total = 1.0

        p_fault /= total
        p_anomaly /= total
        p_attack /= total

        probs = {"fault": p_fault, "anomaly": p_anomaly, "attack": p_attack}
        best = max(probs, key=probs.get)
        classification = Classification(best)

        # Build reasoning
        reasons = []
        if e.fingerprint_status == "compromised":
            reasons.append("fingerprint changed")
        if e.probe_verdict == "attacked":
            reasons.append("unresponsive to probes")
        if not e.neighbours_agree and not e.command_explains_change:
            reasons.append("neighbours disagree, no command explains it")
        if e.health_score > 90 and classification == Classification.ATTACK:
            reasons.append("health was fine before sudden change")
        if e.command_explains_change:
            reasons.append("command explains the sensor change")
        if len(e.health_trend) >= 3 and classification == Classification.FAULT:
            reasons.append("gradual health decline")

        reasoning = "; ".join(reasons) if reasons else f"rule-based: highest probability is {best}"

        return ClassificationResult(
            classification=classification,
            confidence=probs[best],
            probabilities=probs,
            reasoning=reasoning,
            source=LLMTier.RULES,
        )

    def forensics(self, evidence: SensorEvidence, classification: Classification) -> str:
        """Rule-based forensics report."""
        sid = evidence.sensor_id
        r = evidence.normalised_residual

        if classification == Classification.ATTACK:
            if evidence.fingerprint_status == "compromised":
                return (f"Sensor {sid}: firmware-level compromise detected. "
                        f"ADC noise signature changed, residual {r:.1f}sigma. "
                        f"Attacker likely replaced sensor firmware to inject false readings.")
            else:
                return (f"Sensor {sid}: data injection attack. "
                        f"Residual {r:.1f}sigma, neighbours unaffected. "
                        f"Attacker spoofing sensor value to manipulate control decisions.")
        elif classification == Classification.FAULT:
            return (f"Sensor {sid}: hardware degradation. "
                    f"Health declining over time, residual {r:.1f}sigma. "
                    f"Recommend calibration check and preventive maintenance.")
        else:
            return (f"Sensor {sid}: legitimate anomaly. "
                    f"Residual {r:.1f}sigma, consistent with process event. "
                    f"Neighbours confirm, no intervention needed.")


# =================================================================
#  FALLBACK CHAIN
# =================================================================

class LLMFallbackChain:
    """
    Three-tier LLM fallback chain.

    Tries Groq API first, then local TinyLlama, then hardcoded rules.
    Always returns a result -- the rules tier never fails.

    The detection pipeline (observer, CBF, probing, fingerprinting)
    never depends on any LLM. This chain only enriches classification
    and forensics. If the entire chain is unavailable, the system
    still detects and blocks attacks via pure math.
    """

    def __init__(self):
        self.groq = GroqClassifier()
        self.local = LocalLlamaClassifier()
        self.rules = RuleBasedClassifier()

        self._tier_stats = {tier: {"calls": 0, "successes": 0} for tier in LLMTier}

    @property
    def active_tier(self) -> LLMTier:
        """Which tier is currently available as primary."""
        if self.groq.available:
            return LLMTier.GROQ
        if self.local.available:
            return LLMTier.LOCAL
        return LLMTier.RULES

    def classify(self, evidence: SensorEvidence) -> ClassificationResult:
        """
        Classify a suspicious sensor through the fallback chain.

        Groq -> TinyLlama -> hardcoded rules.
        """
        # Tier 1: Groq
        if self.groq.available:
            self._tier_stats[LLMTier.GROQ]["calls"] += 1
            result = self.groq.classify(evidence)
            if result is not None:
                self._tier_stats[LLMTier.GROQ]["successes"] += 1
                return result
            log.info("Groq failed, falling back to local LLM")

        # Tier 2: Local TinyLlama
        if self.local.available:
            self._tier_stats[LLMTier.LOCAL]["calls"] += 1
            result = self.local.classify(evidence)
            if result is not None:
                self._tier_stats[LLMTier.LOCAL]["successes"] += 1
                return result
            log.info("Local LLM failed, falling back to rules")

        # Tier 3: Hardcoded rules (always succeeds)
        self._tier_stats[LLMTier.RULES]["calls"] += 1
        self._tier_stats[LLMTier.RULES]["successes"] += 1
        return self.rules.classify(evidence)

    def forensics(self, evidence: SensorEvidence,
                  classification: Classification,
                  attack_context: str = "") -> str:
        """
        Generate forensics report through the fallback chain.
        """
        if self.groq.available:
            report = self.groq.forensics(evidence, attack_context)
            if report is not None:
                return report

        return self.rules.forensics(evidence, classification)

    def get_stats(self) -> dict:
        return {
            "active_tier": self.active_tier.value,
            "tier_stats": {
                k.value: v for k, v in self._tier_stats.items()
            },
        }


# Module-level singleton
_chain = None


def get_chain() -> LLMFallbackChain:
    """Get or create the module-level fallback chain singleton."""
    global _chain
    if _chain is None:
        _chain = LLMFallbackChain()
    return _chain
