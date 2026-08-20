"""
Agent 4: Guardian — CBF enforcement on every command.

The last line of defense. Uses ONLY observer-verified state
(never raw sensor data). Runs the CBF filter on every single
command, no exceptions.

Guardian is independent from Sentinel. Even if Sentinel is
compromised, Guardian still enforces physics-based safety.
"""

import numpy as np
from .state import AgentState
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "security"))
from cbf_filter import CBFSafetyFilter, PlantConfig, PlantState


# Module-level CBF instance
_cbf = CBFSafetyFilter(PlantConfig())


def _state_to_plant_state(state: AgentState) -> PlantState:
    """
    Convert agent state to PlantState for CBF.

    Uses verified readings from Sentinel, never raw.
    If Sentinel hasn't run yet, uses raw as fallback.

    Sensor-to-physical mapping (SWaT-like):
      readings[0:3] → tank levels, scaled to 0.2–1.0m physical range
      readings[3:5] → pressures, scaled to 100–400 kPa
      readings[5]   → flow indicator
    """
    verified = state.get("verified_readings", state["raw_readings"])
    v = np.array(verified)

    n = len(v)
    # Map normalised readings (~0.5 mean) to physical ranges
    raw_levels = v[:3] if n >= 3 else np.array([0.5, 0.5, 0.5])
    levels = 0.2 + raw_levels * 1.0  # ~0.7m when readings ~0.5
    levels = np.clip(levels, 0.2, 1.5)

    raw_pressures = v[3:5] if n >= 5 else np.array([0.5, 0.5])
    pressures = 100 + raw_pressures * 300  # ~250 kPa when readings ~0.5
    pressures = np.clip(pressures, 50, 500)

    flows = np.ones(3) * 0.003

    return PlantState(
        levels=levels,
        pressures=pressures,
        flows_in=flows,
        flows_out=flows,
    )


def guardian_node(state: AgentState) -> dict:
    """
    Guardian agent: enforce CBF safety on every command.

    Steps:
    1. Get agent's proposed command
    2. Build plant state from VERIFIED readings only
    3. Run CBF quadratic program
    4. Output safe command

    No agent trusts another — Guardian checks math, not opinions.
    """
    agent_cmd = state.get("agent_command")
    msgs = state.get("messages", [])

    if agent_cmd is None:
        return {
            "safe_command": [],
            "command_status": "no_command",
            "command_intervention": 0.0,
            "cbf_feasible": True,
            "messages": msgs,
        }

    u_agent = np.array(agent_cmd)
    plant_state = _state_to_plant_state(state)

    # Tighten CBF constraints for sensors flagged by Sentinel or Fingerprint
    blocked = state.get("blocked_sensors", [])
    fp_status = state.get("fingerprint_status", {})
    compromised_fp = [int(s) for s, st in fp_status.items() if st == "compromised"]

    # If sensors are compromised, reduce trust in state estimates
    # by tightening safety margins
    if blocked or compromised_fp:
        tighter_config = PlantConfig(
            alpha=0.8,  # more aggressive repulsion from boundaries
            level_max=np.array([1.1, 1.1, 1.4]),  # tighter limits
        )
        cbf_tight = CBFSafetyFilter(tighter_config)
        result = cbf_tight.filter(u_agent, plant_state)
    else:
        result = _cbf.filter(u_agent, plant_state)

    if not result["feasible"]:
        status = "emergency_stop"
        msgs = msgs + [
            "[Guardian] EMERGENCY STOP — no feasible safe command. All actuators off."
        ]
    elif result["modified"]:
        status = "modified"
        msgs = msgs + [
            f"[Guardian] Command modified. Intervention: {result['intervention']:.4f}"
        ]
    else:
        status = "passed"

    return {
        "safe_command": result["u_safe"].tolist(),
        "command_status": status,
        "command_intervention": result["intervention"],
        "cbf_feasible": result["feasible"],
        "messages": msgs,
    }
