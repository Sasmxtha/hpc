"""
Agent 2: Prober — Active perturbation tester.

When coupling between sensors is weak (Overseer escalates to L2),
the Prober injects small random perturbations and checks if sensors
respond correctly per physics.

Key insight for fault vs attack:
  - Faulty sensor: still responds to perturbation (imperfectly)
  - Attacked sensor: firmware doesn't know the perturbation is coming,
    so it doesn't respond (or responds wrong)

All perturbations are CBF-bounded — Prober cannot endanger the plant.

Delegates to security.prober.ActiveProber for the actual math.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "security"))
from prober import ActiveProber, ProbeConfig, ProbeVerdict

from .state import AgentState, DefenseLevel

# Module-level prober instance — persists across cycles
_prober: ActiveProber | None = None
_B_matrix: np.ndarray | None = None
_C_matrix: np.ndarray | None = None


def _ensure_prober(n_sensors: int, n_actuators: int):
    global _prober, _B_matrix, _C_matrix
    if _prober is None or _prober.n_sensors != n_sensors:
        _prober = ActiveProber(n_sensors, n_actuators, ProbeConfig(magnitude=0.03))
        _B_matrix = 0.02 * np.eye(n_sensors, n_actuators)
        _C_matrix = np.eye(n_sensors)


def prober_node(state: AgentState) -> dict:
    """
    Prober agent: active sensor verification via perturbation.

    Only runs when defense_level >= L2 (weak coupling).
    Uses cryptographic RNG so perturbations are unpredictable.
    After n random probes, evasion probability drops as (1-p)^n → 0.
    """
    defense_level = state.get("defense_level", DefenseLevel.L1_MULTI_DOMAIN)
    msgs = state.get("messages", [])

    if defense_level < DefenseLevel.L2_ACTIVE_PROBING:
        return {
            "probe_active": False,
            "messages": msgs,
        }

    raw = np.array(state["raw_readings"])
    n_sensors = len(raw)
    n_actuators = n_sensors  # assume square for now
    _ensure_prober(n_sensors, n_actuators)

    # Target suspicious + blocked sensors
    suspects = list(
        set(state.get("suspicious_sensors", []))
        | set(state.get("blocked_sensors", []))
    )

    if not suspects:
        return {
            "probe_active": True,
            "probe_verdict": {},
            "messages": msgs + ["[Prober] No suspects to probe this cycle."],
        }

    # Design and simulate perturbation
    delta_u = _prober.design_probe(suspects, _B_matrix)
    y_before = raw.copy()

    # Expected physics response
    delta_y_true = _C_matrix @ (_B_matrix @ delta_u)

    # Simulate: honest sensors track, blocked sensors don't
    blocked_set = set(state.get("blocked_sensors", []))
    y_after = y_before.copy()
    for sid in range(n_sensors):
        if sid in blocked_set:
            y_after[sid] += np.random.randn() * 0.00005
        else:
            y_after[sid] += delta_y_true[sid] + np.random.randn() * 0.00005

    # Analyse responses
    results = _prober.analyse_response(
        y_before, y_after, delta_u, _B_matrix, _C_matrix, suspects
    )

    verdicts = {}
    for r in results:
        verdicts[r.sensor_id] = r.verdict.value

    attacked = [r.sensor_id for r in results if r.verdict == ProbeVerdict.ATTACKED]
    faulty = [r.sensor_id for r in results if r.verdict == ProbeVerdict.FAULTY]

    evasion_bounds = _prober.get_cumulative_bounds()

    msg_parts = [f"[Prober] Tested {len(suspects)} sensors."]
    if attacked:
        msg_parts.append(f"Attacked (unresponsive): {attacked}")
    if faulty:
        msg_parts.append(f"Faulty (partial response): {faulty}")
    for sid in suspects:
        if sid in evasion_bounds:
            msg_parts.append(f"  sensor {sid}: P(evasion)={evasion_bounds[sid]:.2e}")

    msgs = msgs + [" ".join(msg_parts)]

    return {
        "probe_active": True,
        "probe_perturbation": delta_u.tolist(),
        "probe_verdict": verdicts,
        "messages": msgs,
    }
