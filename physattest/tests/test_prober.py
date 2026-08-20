"""Tests for Active Probing (Component 7)."""

import numpy as np
import pytest
from physattest.security.prober import (
    ActiveProber, ProbeConfig, ProbeVerdict,
    PerturbationDesigner, ResponseAnalyser,
)


@pytest.fixture
def prober():
    return ActiveProber(n_sensors=6, n_actuators=6)


@pytest.fixture
def system_matrices():
    B = 0.02 * np.eye(6)
    C = np.eye(6)
    return B, C


class TestPerturbationDesigner:
    def test_perturbation_bounded_by_config(self):
        config = ProbeConfig(magnitude=0.03, max_perturbation=0.05)
        designer = PerturbationDesigner(6, config)
        B = 0.02 * np.eye(6)
        for _ in range(20):
            du = designer.design([0, 1, 2], B)
            assert np.max(np.abs(du)) <= config.max_perturbation + 1e-10

    def test_perturbation_nonzero(self):
        designer = PerturbationDesigner(6, ProbeConfig())
        B = 0.02 * np.eye(6)
        du = designer.design([0], B)
        assert np.linalg.norm(du) > 1e-8

    def test_different_perturbations_each_call(self):
        designer = PerturbationDesigner(6, ProbeConfig())
        # Non-diagonal B so SVD doesn't collapse the random component
        B = 0.02 * np.array([
            [2, 1, 0, 0, 0, 0],
            [1, 2, 1, 0, 0, 0],
            [0, 1, 2, 1, 0, 0],
            [0, 0, 1, 2, 1, 0],
            [0, 0, 0, 1, 2, 1],
            [0, 0, 0, 0, 1, 2],
        ], dtype=float)
        perturbations = [designer.design([0, 1, 2], B) for _ in range(10)]
        unique_count = sum(
            1 for i in range(len(perturbations) - 1)
            if not np.allclose(perturbations[i], perturbations[i + 1])
        )
        assert unique_count >= 3, "most perturbations should differ"


class TestResponseAnalyser:
    def test_honest_sensor_high_correlation(self, system_matrices):
        B, C = system_matrices
        config = ProbeConfig(min_probes=1)
        analyser = ResponseAnalyser(6, config)

        du = np.array([0.01, 0, 0, 0, 0, 0])
        y_before = np.array([0.5] * 6)
        y_after = y_before.copy()
        y_after[0] += float(C[0] @ (B @ du))

        result = analyser.analyse(0, du, y_before, y_after, B, C)
        assert result.correlation > 0.5

    def test_attacked_sensor_low_correlation(self, system_matrices):
        B, C = system_matrices
        config = ProbeConfig(min_probes=1)
        analyser = ResponseAnalyser(6, config)

        np.random.seed(42)
        du = np.array([0.01, 0, 0, 0, 0, 0])
        y_before = np.array([0.5] * 6)
        y_after = y_before.copy()
        y_after[0] += 0.1  # random, unrelated to perturbation

        result = analyser.analyse(0, du, y_before, y_after, B, C)
        assert result.correlation < 0.5


class TestActiveProber:
    def test_probe_design_returns_correct_shape(self, prober, system_matrices):
        B, _ = system_matrices
        du = prober.design_probe([0, 1], B)
        assert du.shape == (6,)

    def test_analyse_returns_results_for_targets(self, prober, system_matrices):
        B, C = system_matrices
        du = prober.design_probe([0, 1], B)
        y_before = np.array([0.5] * 6)
        y_after = y_before + C @ (B @ du)
        results = prober.analyse_response(y_before, y_after, du, B, C, [0, 1])
        assert len(results) == 2
        assert results[0].sensor_id == 0
        assert results[1].sensor_id == 1

    def test_cumulative_bounds_start_at_one(self, prober):
        bounds = prober.get_cumulative_bounds()
        for sid in range(6):
            assert bounds[sid] == 1.0

    def test_evasion_decreases_with_probes(self, system_matrices):
        B, C = system_matrices
        np.random.seed(42)
        prober = ActiveProber(6, 6, ProbeConfig(min_probes=1))

        for _ in range(10):
            du = prober.design_probe([4], B)
            y_before = np.array([0.5] * 6)
            y_after = y_before + C @ (B @ du)
            # Sensor 4 attacked: random response
            y_after[4] = y_before[4] + np.random.randn() * 0.01
            prober.analyse_response(y_before, y_after, du, B, C, [4])

        bounds = prober.get_cumulative_bounds()
        assert bounds[4] < 1.0, "evasion bound should decrease for attacked sensor"

    def test_probe_log_grows(self, prober, system_matrices):
        B, C = system_matrices
        du = prober.design_probe([0], B)
        y_before = np.array([0.5] * 6)
        y_after = y_before + C @ (B @ du)
        prober.analyse_response(y_before, y_after, du, B, C, [0])
        assert len(prober.probe_log) == 1
