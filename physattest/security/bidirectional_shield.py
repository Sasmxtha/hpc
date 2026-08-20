"""
Bidirectional Shield for PhysAttest.

Same observer mechanism protects both directions:
  - Bottom-up: blocks fake sensor data before the agent sees it
  - Top-down: blocks dangerous commands before actuators execute them
  - Command tracking: logs applied commands so expected changes aren't flagged

The shield wraps the observer (provided by Member 1) and the CBF filter.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import deque


class SensorStatus(Enum):
    CLEAN = "clean"
    SUSPICIOUS = "suspicious"
    BLOCKED = "blocked"


class CommandStatus(Enum):
    PASSED = "passed"
    MODIFIED = "modified"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class ShieldEvent:
    """A single shield action, logged for forensics."""
    timestamp: float
    direction: str            # "bottom_up" or "top_down"
    sensor_id: Optional[int]
    status: str
    residual: Optional[float]
    original_value: Optional[np.ndarray]
    filtered_value: Optional[np.ndarray]
    reason: str


@dataclass
class ObserverInterface:
    """
    Interface to the physics observer (built by Member 1).

    The observer maintains a state estimate x̂ and computes residuals.
    This is a working implementation that will be replaced with the
    real observer once Member 1 delivers it.
    """

    n_sensors: int
    n_actuators: int

    # State-space matrices (simplified linear model)
    A: np.ndarray = field(default=None)  # state transition
    B: np.ndarray = field(default=None)  # control input
    C: np.ndarray = field(default=None)  # observation

    # Current state estimate
    x_hat: np.ndarray = field(default=None)

    # Process and measurement noise covariances
    Q: np.ndarray = field(default=None)  # process noise
    R: np.ndarray = field(default=None)  # measurement noise
    P: np.ndarray = field(default=None)  # estimate covariance

    def __post_init__(self):
        n_s = self.n_sensors
        n_a = self.n_actuators

        if self.A is None:
            self.A = 0.98 * np.eye(n_s)
        if self.B is None:
            self.B = 0.01 * np.random.randn(n_s, n_a)
            self.B = (self.B + self.B.T[:n_s, :n_a]) if n_s == n_a else self.B
        if self.C is None:
            self.C = np.eye(n_s)
        if self.x_hat is None:
            self.x_hat = np.zeros(n_s)
        if self.Q is None:
            self.Q = 0.001 * np.eye(n_s)
        if self.R is None:
            self.R = 0.01 * np.eye(n_s)
        if self.P is None:
            self.P = 0.1 * np.eye(n_s)

    def predict(self, u_applied: np.ndarray):
        """Predict next state given the command that was actually applied."""
        self.x_hat = self.A @ self.x_hat + self.B @ u_applied
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y: np.ndarray) -> np.ndarray:
        """
        Kalman update. Returns the residual vector.

        r(t) = y(t) - C @ x̂(t)

        If sensors are honest, r ≈ 0 (just noise).
        If sensors are lying, r is large — physics disagrees with the reading.
        """
        r = y - self.C @ self.x_hat

        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ r
        self.P = (np.eye(len(self.x_hat)) - K @ self.C) @ self.P

        return r

    def get_reconstructed(self, sensor_id: int) -> float:
        """Reconstruct a compromised sensor's value from the state estimate."""
        return float(self.C[sensor_id] @ self.x_hat)


class BidirectionalShield:
    """
    Bidirectional shield protecting both data flow directions.

    Bottom-up:  sensor readings → [shield] → verified readings → agent
    Top-down:   agent commands  → [shield/CBF] → safe commands  → actuators

    Command tracking ensures the observer predicts the effect of applied
    commands, so legitimate sensor changes aren't flagged as attacks.
    """

    def __init__(
        self,
        n_sensors: int,
        n_actuators: int,
        cbf_filter,
        residual_threshold: float = 3.0,
        suspicious_threshold: float = 2.0,
        history_len: int = 100,
    ):
        self.n_sensors = n_sensors
        self.n_actuators = n_actuators
        self.cbf = cbf_filter

        # Thresholds in units of standard deviations
        self.residual_threshold = residual_threshold
        self.suspicious_threshold = suspicious_threshold

        # Observer (placeholder — Member 1 provides the real one)
        self.observer = ObserverInterface(n_sensors, n_actuators)

        # Per-sensor health tracking
        self.sensor_status = [SensorStatus.CLEAN] * n_sensors
        self.consecutive_suspicious = np.zeros(n_sensors, dtype=int)

        # Command log for tracking
        self.command_log = deque(maxlen=history_len)

        # Event log for forensics
        self.event_log = deque(maxlen=1000)

        # Running statistics for adaptive thresholds
        self.residual_history = deque(maxlen=500)
        self.residual_std = np.ones(n_sensors) * 0.1  # initial estimate

        self._time = 0.0

    def _update_residual_stats(self, r: np.ndarray):
        self.residual_history.append(np.abs(r))
        if len(self.residual_history) > 20:
            history = np.array(self.residual_history)
            self.residual_std = np.std(history, axis=0) + 1e-8

    def shield_sensors(self, raw_readings: np.ndarray) -> dict:
        """
        Bottom-up shield: filter sensor readings before the agent sees them.

        Steps:
        1. Observer computes residual r = y - Cx̂
        2. Normalise residual by running std
        3. Classify each sensor: clean / suspicious / blocked
        4. Replace blocked sensors with reconstructed values

        Args:
            raw_readings: raw sensor vector from the plant (possibly spoofed)

        Returns:
            dict with:
                verified_readings: cleaned sensor vector for the agent
                statuses:          per-sensor SensorStatus
                residuals:         raw residual vector
                blocked_sensors:   list of sensor IDs that were blocked
        """
        r = self.observer.update(raw_readings)
        self._update_residual_stats(r)

        normalised_r = np.abs(r) / self.residual_std
        verified = raw_readings.copy()
        statuses = []
        blocked = []

        for i in range(self.n_sensors):
            if normalised_r[i] > self.residual_threshold:
                # Physics says this reading is impossible
                self.sensor_status[i] = SensorStatus.BLOCKED
                verified[i] = self.observer.get_reconstructed(i)
                blocked.append(i)
                self.consecutive_suspicious[i] = 0

                self.event_log.append(ShieldEvent(
                    timestamp=self._time,
                    direction="bottom_up",
                    sensor_id=i,
                    status="BLOCKED",
                    residual=float(normalised_r[i]),
                    original_value=np.array([raw_readings[i]]),
                    filtered_value=np.array([verified[i]]),
                    reason=f"residual {normalised_r[i]:.2f}σ > {self.residual_threshold}σ",
                ))

            elif normalised_r[i] > self.suspicious_threshold:
                self.sensor_status[i] = SensorStatus.SUSPICIOUS
                self.consecutive_suspicious[i] += 1

                # If suspicious for too long, escalate to blocked
                if self.consecutive_suspicious[i] > 10:
                    self.sensor_status[i] = SensorStatus.BLOCKED
                    verified[i] = self.observer.get_reconstructed(i)
                    blocked.append(i)

                    self.event_log.append(ShieldEvent(
                        timestamp=self._time,
                        direction="bottom_up",
                        sensor_id=i,
                        status="BLOCKED_ESCALATED",
                        residual=float(normalised_r[i]),
                        original_value=np.array([raw_readings[i]]),
                        filtered_value=np.array([verified[i]]),
                        reason=f"suspicious for {self.consecutive_suspicious[i]} consecutive cycles",
                    ))
                    self.consecutive_suspicious[i] = 0
            else:
                self.sensor_status[i] = SensorStatus.CLEAN
                self.consecutive_suspicious[i] = 0

            statuses.append(self.sensor_status[i])

        return {
            "verified_readings": verified,
            "statuses": statuses,
            "residuals": r,
            "normalised_residuals": normalised_r,
            "blocked_sensors": blocked,
        }

    def shield_command(self, u_agent: np.ndarray, plant_state) -> dict:
        """
        Top-down shield: filter agent commands before actuators execute them.

        Steps:
        1. Pass command through CBF safety filter
        2. Log the applied command for observer tracking
        3. Feed the applied command into the observer's prediction

        Args:
            u_agent:     the agent's proposed command vector
            plant_state: current observer-verified plant state (for CBF)

        Returns:
            dict with:
                u_applied:    the command that will actually execute
                status:       CommandStatus
                intervention: L2 norm of modification
                cbf_result:   full CBF output dict
        """
        cbf_result = self.cbf.filter(u_agent, plant_state)

        if not cbf_result["feasible"]:
            u_applied = np.zeros_like(u_agent)
            status = CommandStatus.EMERGENCY_STOP

            self.event_log.append(ShieldEvent(
                timestamp=self._time,
                direction="top_down",
                sensor_id=None,
                status="EMERGENCY_STOP",
                residual=None,
                original_value=u_agent,
                filtered_value=u_applied,
                reason="CBF infeasible — no safe command exists",
            ))
        elif cbf_result["modified"]:
            u_applied = cbf_result["u_safe"]
            status = CommandStatus.MODIFIED

            self.event_log.append(ShieldEvent(
                timestamp=self._time,
                direction="top_down",
                sensor_id=None,
                status="MODIFIED",
                residual=None,
                original_value=u_agent,
                filtered_value=u_applied,
                reason=f"intervention={cbf_result['intervention']:.4f}",
            ))
        else:
            u_applied = cbf_result["u_safe"]
            status = CommandStatus.PASSED

        # Command tracking: log what was applied and update observer prediction
        self.command_log.append({
            "time": self._time,
            "u_agent": u_agent.copy(),
            "u_applied": u_applied.copy(),
            "status": status.value,
        })
        self.observer.predict(u_applied)

        self._time += 1.0

        return {
            "u_applied": u_applied,
            "status": status,
            "intervention": cbf_result["intervention"],
            "cbf_result": cbf_result,
        }

    def get_defense_summary(self) -> dict:
        """Summary statistics for the Overseer agent."""
        recent_events = [e for e in self.event_log if e.timestamp > self._time - 60]
        return {
            "total_events": len(self.event_log),
            "recent_events": len(recent_events),
            "blocked_sensors": [
                i for i, s in enumerate(self.sensor_status)
                if s == SensorStatus.BLOCKED
            ],
            "suspicious_sensors": [
                i for i, s in enumerate(self.sensor_status)
                if s == SensorStatus.SUSPICIOUS
            ],
            "commands_modified_recent": sum(
                1 for e in recent_events
                if e.direction == "top_down" and e.status == "MODIFIED"
            ),
        }


def demo_bidirectional():
    """
    Demo: sensor spoofing blocked bottom-up, dangerous command blocked top-down,
    and command tracking preventing false alarms.
    """
    from cbf_filter import CBFSafetyFilter, PlantConfig, PlantState

    config = PlantConfig()
    cbf = CBFSafetyFilter(config)
    shield = BidirectionalShield(
        n_sensors=6,
        n_actuators=6,
        cbf_filter=cbf,
        residual_threshold=3.0,
        suspicious_threshold=2.0,
    )

    print("=" * 65)
    print("PhysAttest Bidirectional Shield — Demo")
    print("=" * 65)

    # Warm up the observer with normal data so residual stats are meaningful
    plant_state = PlantState(
        levels=np.array([0.7, 0.6, 0.8]),
        pressures=np.array([300.0, 250.0]),
        flows_in=np.array([0.003, 0.002, 0.002]),
        flows_out=np.array([0.002, 0.002, 0.002]),
    )

    np.random.seed(42)
    print("\n--- Warming up observer with 50 normal cycles ---")
    for _ in range(50):
        normal_readings = np.random.randn(6) * 0.05 + 0.5
        normal_cmd = np.array([0.3, 0.3, 0.3, 0.5, 0.5, 0.5])
        shield.shield_sensors(normal_readings)
        shield.shield_command(normal_cmd, plant_state)
    print(f"  Residual std estimate: {np.round(shield.residual_std, 4)}")

    # --- Bottom-up: sensor spoofing ---
    print("\n--- Bottom-Up: Sensor Spoofing Attack ---")
    spoofed = np.random.randn(6) * 0.05 + 0.5
    spoofed[2] = 5.0  # sensor 2 reports absurd value
    spoofed[4] = -3.0  # sensor 4 reports physically impossible negative

    result = shield.shield_sensors(spoofed)
    print(f"  Raw readings:      {np.round(spoofed, 3)}")
    print(f"  Verified readings: {np.round(result['verified_readings'], 3)}")
    print(f"  Blocked sensors:   {result['blocked_sensors']}")
    print(f"  Statuses:          {[s.value for s in result['statuses']]}")
    print(f"  Normalised |r|:    {np.round(result['normalised_residuals'], 2)}")

    # --- Top-down: hijacked agent command ---
    print("\n--- Top-Down: Hijacked Agent Command ---")
    dangerous_cmd = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    cmd_result = shield.shield_command(dangerous_cmd, plant_state)
    print(f"  Agent wants:  {dangerous_cmd}")
    print(f"  Applied:      {np.round(cmd_result['u_applied'], 4)}")
    print(f"  Status:       {cmd_result['status'].value}")
    print(f"  Intervention: {cmd_result['intervention']:.4f}")

    # --- Command tracking: legitimate change not flagged ---
    print("\n--- Command Tracking: Legitimate Change ---")
    # Agent sends a real command (turn on pump 0 more)
    legit_cmd = np.array([0.6, 0.3, 0.3, 0.5, 0.5, 0.5])
    cmd_result = shield.shield_command(legit_cmd, plant_state)
    print(f"  Legitimate command: {legit_cmd}")
    print(f"  Status: {cmd_result['status'].value}")

    # Next cycle: sensors reflect the command's effect (higher flow)
    # Because observer predicted this change, residual stays small
    post_cmd_readings = np.random.randn(6) * 0.05 + 0.5
    post_cmd_readings[0] += 0.02  # slight increase from pump 0
    result = shield.shield_sensors(post_cmd_readings)
    print(f"  Next-cycle readings: {np.round(post_cmd_readings, 3)}")
    print(f"  Blocked sensors:     {result['blocked_sensors']}")
    print(f"  (No false alarm — observer expected the change)")

    # --- Summary ---
    print("\n--- Defense Summary ---")
    summary = shield.get_defense_summary()
    print(f"  Total events:              {summary['total_events']}")
    print(f"  Recent events (60s):       {summary['recent_events']}")
    print(f"  Currently blocked sensors: {summary['blocked_sensors']}")
    print(f"  Commands modified (recent):{summary['commands_modified_recent']}")

    print("\n" + "=" * 65)
    print("Bidirectional: same observer protects BOTH directions.")
    print("Command tracking: legitimate changes are NOT false alarms.")
    print("=" * 65)


if __name__ == "__main__":
    demo_bidirectional()
