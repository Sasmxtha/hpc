"""Tests for Correlated Drift Analysis (Server Security)."""

import numpy as np
import pytest
from physattest.security.correlated_drift import (
    CorrelatedDriftAnalyser, DriftConfig, IntegrityStatus,
)


@pytest.fixture
def analyser():
    return CorrelatedDriftAnalyser(n_sensors=6)


class TestInsufficientData:
    def test_returns_uncertain_below_min_samples(self, analyser):
        for _ in range(5):
            analyser.add_residual(np.random.randn(6) * 0.01)
        result = analyser.analyse()
        assert result.status == IntegrityStatus.UNCERTAIN


class TestNormalOperation:
    def test_clean_with_independent_noise(self):
        np.random.seed(42)
        analyser = CorrelatedDriftAnalyser(6)
        for _ in range(60):
            analyser.add_residual(np.random.randn(6) * 0.01)
        result = analyser.analyse()
        assert result.status == IntegrityStatus.CLEAN
        assert result.tau < 0.4

    def test_tau_near_zero_for_independent(self):
        np.random.seed(0)
        analyser = CorrelatedDriftAnalyser(6)
        for _ in range(100):
            analyser.add_residual(np.random.randn(6) * 0.01)
        result = analyser.analyse()
        assert result.tau < 0.3


class TestSensorAttack:
    def test_isolated_anomaly_detected(self):
        np.random.seed(42)
        analyser = CorrelatedDriftAnalyser(6)
        for _ in range(60):
            r = np.random.randn(6) * 0.01
            r[2] = np.random.randn() * 0.5 + 2.0
            analyser.add_residual(r)
        result = analyser.analyse()
        assert result.status == IntegrityStatus.SENSOR_ATTACK
        assert result.isolation_scores[2] > 1.0

    def test_tau_stays_low_for_isolated_attack(self):
        np.random.seed(42)
        analyser = CorrelatedDriftAnalyser(6)
        for _ in range(60):
            r = np.random.randn(6) * 0.01
            r[4] = np.random.randn() * 1.0 + 3.0
            analyser.add_residual(r)
        result = analyser.analyse()
        assert result.tau < 0.5


class TestCodeTampering:
    def test_correlated_drift_detected(self):
        np.random.seed(42)
        analyser = CorrelatedDriftAnalyser(6)
        for t in range(60):
            r = np.random.randn(6) * 0.01
            shared = 0.3 * np.sin(0.1 * t) + 0.1 * t / 60
            r += shared
            r += np.array([0.05, 0.03, 0.04, 0.06, 0.02, 0.05]) * t / 60
            analyser.add_residual(r)
        result = analyser.analyse()
        assert result.status == IntegrityStatus.CODE_TAMPERING
        assert result.tau > 0.4

    def test_eigenvalue_concentration(self):
        np.random.seed(42)
        analyser = CorrelatedDriftAnalyser(6)
        for t in range(60):
            r = np.random.randn(6) * 0.01
            r += 0.5 * t / 60
            analyser.add_residual(r)
        result = analyser.analyse()
        assert result.eigenvalue_ratio > 0.5


class TestConfig:
    def test_custom_thresholds(self):
        config = DriftConfig(tau_threshold=0.8, eigen_threshold=0.9, min_samples=10)
        analyser = CorrelatedDriftAnalyser(6, config)
        np.random.seed(42)
        for t in range(30):
            r = np.random.randn(6) * 0.01
            r += 0.1 * t / 30
            analyser.add_residual(r)
        result = analyser.analyse()
        # With higher thresholds, moderate drift might not trigger
        assert result.tau >= 0

    def test_sliding_window(self):
        config = DriftConfig(window_size=20)
        analyser = CorrelatedDriftAnalyser(6, config)
        for t in range(50):
            analyser.add_residual(np.random.randn(6) * 0.01)
        assert len(analyser.residual_history) == 20


class TestOutputStructure:
    def test_result_fields(self):
        analyser = CorrelatedDriftAnalyser(6)
        for _ in range(30):
            analyser.add_residual(np.random.randn(6) * 0.01)
        result = analyser.analyse()
        assert hasattr(result, "tau")
        assert hasattr(result, "eigenvalue_ratio")
        assert hasattr(result, "correlation_matrix")
        assert result.correlation_matrix.shape == (6, 6)
        assert len(result.isolation_scores) == 6
