"""Tests for the Bidirectional Shield (Component 4)."""

import numpy as np
import pytest
from physattest.security.bidirectional_shield import (
    BidirectionalShield, ObserverInterface, SensorStatus, CommandStatus,
)
from physattest.security.cbf_filter import CBFSafetyFilter, PlantConfig, PlantState


@pytest.fixture
def shield():
    config = PlantConfig()
    cbf = CBFSafetyFilter(config)
    s = BidirectionalShield(
        n_sensors=6, n_actuators=6,
        cbf_filter=cbf,
        residual_threshold=3.0,
        suspicious_threshold=2.0,
    )
    np.random.seed(42)
    plant = PlantState(
        levels=np.array([0.7, 0.6, 0.8]),
        pressures=np.array([300.0, 250.0]),
        flows_in=np.array([0.003, 0.002, 0.002]),
        flows_out=np.array([0.002, 0.002, 0.002]),
    )
    for _ in range(50):
        s.shield_sensors(np.random.randn(6) * 0.05 + 0.5)
        s.shield_command(np.array([0.3, 0.3, 0.3, 0.5, 0.5, 0.5]), plant)
    return s, plant


class TestObserverInterface:
    def test_predict_update_cycle(self):
        obs = ObserverInterface(n_sensors=4, n_actuators=4)
        u = np.zeros(4)
        obs.predict(u)
        r = obs.update(np.ones(4) * 0.5)
        assert r.shape == (4,)

    def test_reconstructed_value(self):
        obs = ObserverInterface(n_sensors=4, n_actuators=4)
        for _ in range(10):
            obs.predict(np.zeros(4))
            obs.update(np.ones(4) * 0.5)
        val = obs.get_reconstructed(0)
        assert isinstance(val, float)


class TestBottomUpShield:
    def test_normal_readings_pass(self, shield):
        s, _ = shield
        result = s.shield_sensors(np.random.randn(6) * 0.05 + 0.5)
        clean_count = sum(1 for st in result["statuses"] if st == SensorStatus.CLEAN)
        assert clean_count >= 4

    def test_spoofed_sensor_blocked(self, shield):
        s, _ = shield
        spoofed = np.random.randn(6) * 0.05 + 0.5
        spoofed[2] = 5.0
        result = s.shield_sensors(spoofed)
        assert 2 in result["blocked_sensors"]
        assert result["verified_readings"][2] != 5.0

    def test_reconstructed_value_used(self, shield):
        s, _ = shield
        spoofed = np.random.randn(6) * 0.05 + 0.5
        spoofed[0] = 100.0
        result = s.shield_sensors(spoofed)
        assert abs(result["verified_readings"][0] - 100.0) > 1.0

    def test_residuals_returned(self, shield):
        s, _ = shield
        result = s.shield_sensors(np.random.randn(6) * 0.05 + 0.5)
        assert "residuals" in result
        assert len(result["residuals"]) == 6


class TestTopDownShield:
    def test_safe_command_passes(self, shield):
        s, plant = shield
        u = np.array([0.3, 0.3, 0.3, 0.5, 0.5, 0.5])
        result = s.shield_command(u, plant)
        assert result["status"] in (CommandStatus.PASSED, CommandStatus.MODIFIED)

    def test_dangerous_command_modified(self, shield):
        s, _ = shield
        near_overflow = PlantState(
            levels=np.array([1.15, 0.6, 0.8]),
            pressures=np.array([300.0, 250.0]),
            flows_in=np.array([0.003, 0.002, 0.002]),
            flows_out=np.array([0.002, 0.002, 0.002]),
        )
        u = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        result = s.shield_command(u, near_overflow)
        assert result["status"] == CommandStatus.MODIFIED
        assert result["intervention"] > 0.1


class TestCommandTracking:
    def test_command_logged(self, shield):
        s, plant = shield
        initial_len = len(s.command_log)
        s.shield_command(np.array([0.3, 0.3, 0.3, 0.5, 0.5, 0.5]), plant)
        assert len(s.command_log) == initial_len + 1

    def test_observer_prediction_updated(self, shield):
        s, plant = shield
        x_before = s.observer.x_hat.copy()
        s.shield_command(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), plant)
        assert not np.allclose(s.observer.x_hat, x_before)


class TestDefenseSummary:
    def test_summary_fields(self, shield):
        s, _ = shield
        summary = s.get_defense_summary()
        assert "total_events" in summary
        assert "blocked_sensors" in summary
        assert "suspicious_sensors" in summary

    def test_event_log_grows(self, shield):
        s, _ = shield
        spoofed = np.random.randn(6) * 0.05 + 0.5
        spoofed[0] = 10.0
        s.shield_sensors(spoofed)
        assert len(s.event_log) > 0
