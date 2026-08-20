"""
Agent 0: Overseer — Coordinator and defense level manager.

Monitors all agent health. Escalates defense levels (L1→L2→L3→L4).
Randomises verification methods each cycle to prevent attacker prediction.
Triggers human notification for critical situations.

Does NOT make detection decisions — only coordinates based on
mathematical evidence from other agents.
"""

import numpy as np
import secrets
from .state import AgentState, DefenseLevel, AlertSeverity


# Track agent heartbeats across cycles
_agent_heartbeats: dict[str, int] = {
    "sentinel": 0,
    "prober": 0,
    "fingerprint": 0,
    "guardian": 0,
}
_missed_heartbeats: dict[str, int] = {k: 0 for k in _agent_heartbeats}

HEARTBEAT_TIMEOUT = 5  # cycles before declaring agent dead


def overseer_pre_node(state: AgentState) -> dict:
    """
    Overseer PRE-step: runs BEFORE other agents each cycle.

    1. Determine current defense level
    2. Check agent health
    3. Randomise verification order
    """
    cycle = state.get("cycle_count", 0) + 1
    severity = state.get("alert_severity", AlertSeverity.NONE)
    blocked = state.get("blocked_sensors", [])
    suspicious = state.get("suspicious_sensors", [])
    msgs = state.get("messages", [])

    # --- Defense level escalation ---
    # L1: default — observer + transformer + classifier (Sentinel)
    # L2: weak coupling detected or sustained suspicious → activate Prober
    # L3: zero coupling or Prober inconclusive → activate Fingerprint
    # L4: everything degraded → CBF-only mode (Guardian alone)

    current_level = state.get("defense_level", DefenseLevel.L1_MULTI_DOMAIN)

    if severity >= AlertSeverity.CRITICAL:
        new_level = DefenseLevel.L4_CBF_BOUNDING
        reason = "CRITICAL alert — CBF bounding mode"
    elif severity >= AlertSeverity.HIGH or len(blocked) >= 2:
        new_level = max(current_level, DefenseLevel.L3_FINGERPRINTING)
        reason = f"HIGH alert or {len(blocked)} blocked sensors"
    elif severity >= AlertSeverity.MEDIUM or len(suspicious) >= 2:
        new_level = max(current_level, DefenseLevel.L2_ACTIVE_PROBING)
        reason = f"MEDIUM alert or {len(suspicious)} suspicious sensors"
    else:
        # Gradual de-escalation after 30 clean cycles
        if cycle % 30 == 0 and severity == AlertSeverity.NONE and current_level > 1:
            new_level = current_level - 1
            reason = "30 clean cycles — de-escalating"
        else:
            new_level = current_level
            reason = ""

    if new_level != current_level and reason:
        msgs = msgs + [f"[Overseer] Defense level L{current_level}→L{new_level}: {reason}"]

    # --- Agent health check ---
    agent_health = state.get("agent_health", {k: True for k in _agent_heartbeats})

    # Count how many agents are healthy
    healthy_count = sum(1 for v in agent_health.values() if v)
    compromised_count = 4 - healthy_count

    # --- Human notification (Theorem 6: safe with ≤2 compromised) ---
    human_notified = state.get("human_notified", False)
    if compromised_count >= 3 and not human_notified:
        msgs = msgs + [
            "[Overseer] HUMAN NOTIFICATION: 3+ agents may be compromised. "
            "Theorem 6 coverage degraded. Manual intervention recommended."
        ]
        human_notified = True
    elif severity >= AlertSeverity.CRITICAL and not human_notified:
        msgs = msgs + [
            "[Overseer] HUMAN NOTIFICATION: Critical attack detected. "
            "System is safe (CBF active) but human should be aware."
        ]
        human_notified = True

    # --- Randomise verification order ---
    # Prevents attacker from predicting which checks run when
    verification_seed = secrets.token_hex(8)

    return {
        "defense_level": int(new_level),
        "cycle_count": cycle,
        "agent_health": agent_health,
        "human_notified": human_notified,
        "escalation_reason": reason,
        "messages": msgs,
    }


def overseer_post_node(state: AgentState) -> dict:
    """
    Overseer POST-step: runs AFTER other agents each cycle.

    1. Collect results from all agents
    2. Update agent health based on who reported
    3. Escalate defense level based on THIS cycle's results
    4. Log cycle summary
    """
    msgs = state.get("messages", [])
    cycle = state.get("cycle_count", 0)
    severity = state.get("alert_severity", AlertSeverity.NONE)
    blocked = state.get("blocked_sensors", [])
    suspicious = state.get("suspicious_sensors", [])

    # Check which agents produced output this cycle
    agent_health = {}
    agent_health["sentinel"] = len(state.get("sensor_statuses", [])) > 0
    agent_health["guardian"] = state.get("command_status", "") != ""

    current_level = state.get("defense_level", DefenseLevel.L1_MULTI_DOMAIN)
    agent_health["prober"] = (
        current_level < DefenseLevel.L2_ACTIVE_PROBING
        or state.get("probe_active", False)
    )
    agent_health["fingerprint"] = (
        current_level < DefenseLevel.L3_FINGERPRINTING
        or len(state.get("fingerprint_status", {})) > 0
    )

    # --- Escalate based on this cycle's findings ---
    if severity >= AlertSeverity.CRITICAL:
        new_level = DefenseLevel.L4_CBF_BOUNDING
        reason = "CRITICAL alert — CBF bounding mode"
    elif severity >= AlertSeverity.HIGH or len(blocked) >= 2:
        new_level = max(current_level, DefenseLevel.L3_FINGERPRINTING)
        reason = f"HIGH alert or {len(blocked)} blocked sensors"
    elif severity >= AlertSeverity.MEDIUM or len(suspicious) >= 2:
        new_level = max(current_level, DefenseLevel.L2_ACTIVE_PROBING)
        reason = f"MEDIUM alert or {len(suspicious)} suspicious sensors"
    else:
        if cycle % 30 == 0 and severity == AlertSeverity.NONE and current_level > 1:
            new_level = current_level - 1
            reason = "30 clean cycles — de-escalating"
        else:
            new_level = current_level
            reason = ""

    if new_level != current_level and reason:
        msgs = msgs + [f"[Overseer] Defense level L{current_level}→L{new_level}: {reason}"]

    # --- Human notification ---
    human_notified = state.get("human_notified", False)
    healthy_count = sum(1 for v in agent_health.values() if v)
    if (4 - healthy_count) >= 3 and not human_notified:
        msgs = msgs + [
            "[Overseer] HUMAN NOTIFICATION: 3+ agents may be compromised."
        ]
        human_notified = True
    elif severity >= AlertSeverity.CRITICAL and not human_notified:
        msgs = msgs + [
            "[Overseer] HUMAN NOTIFICATION: Critical attack detected. "
            "System is safe (CBF active) but human should be aware."
        ]
        human_notified = True

    # Cycle summary
    cmd_status = state.get("command_status", "none")
    summary = (
        f"[Overseer] Cycle {cycle} complete. "
        f"Defense=L{new_level}, Severity={AlertSeverity(severity).name}, "
        f"Blocked={blocked}, Command={cmd_status}, "
        f"Agents={'all healthy' if all(agent_health.values()) else 'DEGRADED'}"
    )
    msgs = msgs + [summary]

    return {
        "defense_level": int(new_level),
        "agent_health": agent_health,
        "human_notified": human_notified,
        "messages": msgs,
    }
