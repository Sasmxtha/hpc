"""Tests for the LLM Fallback Chain."""

import pytest
from physattest.security.llm_fallback import (
    LLMFallbackChain, RuleBasedClassifier, SensorEvidence,
    Classification, ClassificationResult, LLMTier,
    _build_evidence_prompt,
)


@pytest.fixture
def chain():
    return LLMFallbackChain()


@pytest.fixture
def rules():
    return RuleBasedClassifier()


def _attack_evidence():
    return SensorEvidence(
        sensor_id=5,
        residual=2.5,
        normalised_residual=45.0,
        health_score=95.0,
        health_trend=[100, 100, 100, 100, 95],
        neighbour_residuals={4: 0.5},
        neighbours_agree=False,
        fingerprint_status="compromised",
        fingerprint_confidence=0.12,
        probe_verdict="attacked",
        probe_correlation=0.02,
    )


def _fault_evidence():
    return SensorEvidence(
        sensor_id=3,
        residual=0.15,
        normalised_residual=6.2,
        health_score=35.0,
        health_trend=[95, 88, 80, 72, 60, 48, 35],
        neighbour_residuals={2: 0.8, 4: 1.1},
        neighbours_agree=False,
        fingerprint_status="authentic",
        fingerprint_confidence=0.85,
        probe_verdict="faulty",
        probe_correlation=0.45,
    )


def _anomaly_evidence():
    return SensorEvidence(
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


class TestRuleBasedClassifier:
    def test_classifies_attack(self, rules):
        result = rules.classify(_attack_evidence())
        assert result.classification == Classification.ATTACK
        assert result.confidence > 0.5
        assert result.source == LLMTier.RULES

    def test_classifies_fault(self, rules):
        result = rules.classify(_fault_evidence())
        assert result.classification == Classification.FAULT

    def test_classifies_anomaly(self, rules):
        result = rules.classify(_anomaly_evidence())
        assert result.classification == Classification.ANOMALY
        assert result.confidence > 0.5

    def test_probabilities_sum_to_one(self, rules):
        for evidence_fn in [_attack_evidence, _fault_evidence, _anomaly_evidence]:
            result = rules.classify(evidence_fn())
            total = sum(result.probabilities.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_forensics_attack(self, rules):
        report = rules.forensics(_attack_evidence(), Classification.ATTACK)
        assert "compromise" in report.lower() or "attack" in report.lower()

    def test_forensics_fault(self, rules):
        report = rules.forensics(_fault_evidence(), Classification.FAULT)
        assert "degradation" in report.lower() or "maintenance" in report.lower()

    def test_forensics_anomaly(self, rules):
        report = rules.forensics(_anomaly_evidence(), Classification.ANOMALY)
        assert "legitimate" in report.lower() or "anomaly" in report.lower()

    def test_minimal_evidence(self, rules):
        evidence = SensorEvidence(
            sensor_id=0, residual=0.1,
            normalised_residual=2.0, health_score=80.0,
        )
        result = rules.classify(evidence)
        assert result.classification in (Classification.FAULT, Classification.ANOMALY, Classification.ATTACK)


class TestFallbackChain:
    def test_active_tier_is_valid(self, chain):
        assert chain.active_tier in (LLMTier.GROQ, LLMTier.LOCAL, LLMTier.RULES)

    def test_classify_returns_result(self, chain):
        result = chain.classify(_attack_evidence())
        assert isinstance(result, ClassificationResult)
        assert result.source in (LLMTier.GROQ, LLMTier.LOCAL, LLMTier.RULES)

    def test_forensics_returns_string(self, chain):
        report = chain.forensics(_attack_evidence(), Classification.ATTACK)
        assert isinstance(report, str)
        assert len(report) > 10

    def test_stats_tracking(self, chain):
        chain.classify(_attack_evidence())
        chain.classify(_fault_evidence())
        stats = chain.get_stats()
        total_calls = sum(v["calls"] for v in stats["tier_stats"].values())
        total_successes = sum(v["successes"] for v in stats["tier_stats"].values())
        assert total_calls >= 2
        assert total_successes >= 2


class TestEvidencePrompt:
    def test_includes_sensor_id(self):
        prompt = _build_evidence_prompt(_attack_evidence())
        assert "Sensor 5" in prompt

    def test_includes_residual(self):
        prompt = _build_evidence_prompt(_attack_evidence())
        assert "45.00 sigma" in prompt

    def test_includes_fingerprint(self):
        prompt = _build_evidence_prompt(_attack_evidence())
        assert "compromised" in prompt

    def test_includes_probe_verdict(self):
        prompt = _build_evidence_prompt(_attack_evidence())
        assert "attacked" in prompt

    def test_no_command_handled(self):
        evidence = SensorEvidence(
            sensor_id=0, residual=0.1,
            normalised_residual=2.0, health_score=80.0,
        )
        prompt = _build_evidence_prompt(evidence)
        assert "Sensor 0" in prompt


class TestSensorEvidence:
    def test_default_fields(self):
        e = SensorEvidence(sensor_id=0, residual=0.1,
                           normalised_residual=2.0, health_score=80.0)
        assert e.neighbours_agree is False
        assert e.fingerprint_status == "unknown"
        assert e.probe_verdict == "unknown"
        assert e.health_trend == []
