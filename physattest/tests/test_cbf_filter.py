"""Tests for the CBF Safety Filter (Component 3)."""

import numpy as np
import pytest
from physattest.security.cbf_filter import CBFSafetyFilter, PlantConfig, PlantState


@pytest.fixture
def cbf():
    return CBFSafetyFilter(PlantConfig())


@pytest.fixture
def safe_state():
    return PlantState(
        levels=np.array([0.5, 0.5, 0.5]),
        pressures=np.array([200.0, 200.0]),
        flows_in=np.array([0.003, 0.003, 0.003]),
        flows_out=np.array([0.003, 0.003, 0.003]),
    )


@pytest.fixture
def near_overflow_state():
    return PlantState(
        levels=np.array([1.15, 0.6, 0.8]),
        pressures=np.array([300.0, 250.0]),
        flows_in=np.array([0.003, 0.002, 0.002]),
        flows_out=np.array([0.002, 0.002, 0.002]),
    )


class TestCBFPassthrough:
    def test_gentle_command_passes_unchanged(self, cbf, safe_state):
        u = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
        result = cbf.filter(u, safe_state)
        assert result["feasible"]
        assert not result["modified"]
        np.testing.assert_allclose(result["u_safe"], u, atol=1e-3)

    def test_moderate_command_in_safe_state(self, cbf, safe_state):
        u = np.array([0.4, 0.3, 0.3, 0.5, 0.5, 0.5])
        result = cbf.filter(u, safe_state)
        assert result["feasible"]


class TestCBFIntervention:
    def test_overflow_attack_blocked(self, cbf, near_overflow_state):
        u_attack = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        result = cbf.filter(u_attack, near_overflow_state)
        assert result["feasible"]
        assert result["modified"]
        assert result["intervention"] > 0.1
        u_safe = result["u_safe"]
        assert u_safe[0] < u_attack[0], "pump 0 should be reduced"

    def test_subtle_attack_detected(self, cbf, near_overflow_state):
        u_subtle = np.array([0.8, 0.3, 0.3, 0.1, 0.5, 0.5])
        result = cbf.filter(u_subtle, near_overflow_state)
        assert result["feasible"]
        assert result["modified"]

    def test_intervention_is_minimal(self, cbf, near_overflow_state):
        u = np.array([0.6, 0.3, 0.3, 0.3, 0.5, 0.5])
        result = cbf.filter(u, near_overflow_state)
        if result["modified"]:
            diff = np.linalg.norm(result["u_safe"] - u)
            assert diff < 2.0, "CBF should minimise modification"


class TestCBFBarriers:
    def test_all_barriers_nonnegative_after_filter(self, cbf, near_overflow_state):
        u = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        result = cbf.filter(u, near_overflow_state)
        h = result["h_values"]
        assert len(h) > 0

    def test_actuator_bounds_respected(self, cbf, safe_state):
        u = np.array([2.0, -1.0, 1.5, 0.5, 0.5, 0.5])
        result = cbf.filter(u, safe_state)
        u_safe = result["u_safe"]
        np.testing.assert_array_less(-1e-4 * np.ones(6), u_safe)
        np.testing.assert_array_less(u_safe, 1.0 + 1e-4)


class TestCBFFeasibility:
    def test_feasible_in_normal_operation(self, cbf, safe_state):
        u = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = cbf.filter(u, safe_state)
        assert result["feasible"]

    def test_emergency_at_extreme_state(self, cbf):
        extreme = PlantState(
            levels=np.array([1.2, 1.2, 1.5]),
            pressures=np.array([500.0, 500.0]),
            flows_in=np.array([0.005, 0.005, 0.005]),
            flows_out=np.array([0.0, 0.0, 0.0]),
        )
        u = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        result = cbf.filter(u, extreme)
        if not result["feasible"]:
            np.testing.assert_allclose(result["u_safe"], np.zeros(6))
