<<<<<<< HEAD
"""Overseer agent (thin slice): "Coordinator. Monitors all agent health. Escalates defense
levels... Triggers human notification for critical situations."

Full Overseer scope (defense-level escalation L1->L4, coordinating Prober/Fingerprint/
Guardian) needs agents Member 3 owns and that don't exist in this repo yet. What's built
here is the slice that closes the loop on Member 2's components: run Sentinel across every
sensor for a cycle, work out which attacks are coordinated (needs visibility across all
sensors at once, which is exactly what a single Sentinel invocation doesn't have), run the
forensics engine on confirmed attacks, and apply two of Section 10's three human-notification
triggers:

  1. multiple agents compromised            -- STUB: needs Prober/Fingerprint agent status,
                                                 which don't exist yet.
  2. attack targeting critical infrastructure -- IMPLEMENTED: forensics severity == "critical"
  3. sustained attack > 30 minutes            -- IMPLEMENTED: tracked via PlantMemory across cycles

Each Overseer graph invocation is one cycle (one timestep) across the whole plant. Health
scores and sustained-attack timers persist across invocations via PlantMemory -- see
state.py's docstring for why that lives outside LangGraph state rather than in it.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from langgraph.graph import END, StateGraph

from physattest.agents.state import OverseerDependencies, OverseerState
from physattest.ai.classifier_3way import RESPONSE_ACTIONS
from physattest.ai.forensics_llm import ForensicsEvidence, ForensicsReport, run_forensics


def make_sentinel_pass_node(deps: OverseerDependencies):
    def run_sentinel_pass(state: OverseerState) -> dict:
        sensor_windows = state["sensor_windows"]
        coupling_neighbours = state.get("coupling_neighbours", {})
        memory = deps.memory

        sensor_results = {}
        for sensor_id, window in sensor_windows.items():
            neighbour_ids = coupling_neighbours.get(sensor_id, [])
            neighbour_windows = {nb: sensor_windows[nb] for nb in neighbour_ids if nb in sensor_windows}

            sentinel_input = {
                "sensor_id": sensor_id,
                "residual_window": window,
                "neighbour_windows": neighbour_windows,
                "health_score": memory.health_scores.get(sensor_id, 100.0),
                "health_history": memory.health_history[sensor_id][-10:],
            }
            result = deps.sentinel_graph.invoke(sentinel_input)
            sensor_results[sensor_id] = result

            memory.health_scores[sensor_id] = result["new_health_score"]
            memory.health_history[sensor_id].append(result["new_health_score"])

        attacked = [
            sid for sid, r in sensor_results.items() if r["response_action"] == RESPONSE_ACTIONS["attack"]
        ]
        return {"sensor_results": sensor_results, "attacked_sensors": attacked}

    return run_sentinel_pass


def make_forensics_pass_node(deps: OverseerDependencies):
    def run_forensics_pass(state: OverseerState) -> dict:
        attacked = state["attacked_sensors"]
        sensor_windows = state["sensor_windows"]
        sensor_results = state["sensor_results"]
        cycle_index = state.get("cycle_index", 0)
        memory = deps.memory

        reports: Dict[str, ForensicsReport] = {}
        for sensor_id in attacked:
            evidence = ForensicsEvidence(
                sensor_id=sensor_id,
                residual_series=np.asarray(sensor_windows[sensor_id], dtype=float),
                event_context={},  # STUB: Member 1's healing/reconstruction supplies the "true value" detail
                concurrently_attacked_sensors=attacked,
                classifier_confidence=sensor_results[sensor_id]["classification"].probabilities["attack"],
            )
            reports[sensor_id] = run_forensics(evidence, deps.causal_graph, chain=deps.forensics_chain)
            memory.attack_since_cycle.setdefault(sensor_id, cycle_index)
            memory.last_attacked_cycle[sensor_id] = cycle_index

        # Clear sustained-attack timers only after a grace period of consecutive non-attack
        # cycles, not on the very next miss. The 3-way classifier can flicker near its
        # decision boundary cycle to cycle (e.g. onset_abruptness computed fresh each cycle
        # can land just under a threshold even mid-attack) -- resetting the sustained-attack
        # clock on a single borderline cycle would make Section 10's "sustained attack"
        # notification effectively unreachable for anything but a perfectly uniform attack
        # signal, which real sensor noise never produces.
        for sensor_id in list(memory.attack_since_cycle):
            if sensor_id in attacked:
                continue
            last_seen = memory.last_attacked_cycle.get(sensor_id, -10**9)
            if cycle_index - last_seen > deps.sustained_attack_grace_cycles:
                del memory.attack_since_cycle[sensor_id]
                memory.last_attacked_cycle.pop(sensor_id, None)

        return {"forensics_reports": reports}

    return run_forensics_pass


def make_escalate_node(deps: OverseerDependencies):
    def escalate_and_notify(state: OverseerState) -> dict:
        cycle_index = state.get("cycle_index", 0)
        memory = deps.memory
        notifications: List[str] = []

        for sensor_id, report in state.get("forensics_reports", {}).items():
            if report.severity == "critical":
                notifications.append(
                    f"HUMAN NOTIFICATION: attack on '{sensor_id}' targets critical infrastructure "
                    f"(forensics severity=critical, response={report.recommended_response})."
                )

        for sensor_id, since in memory.attack_since_cycle.items():
            duration = cycle_index - since
            if duration >= deps.sustained_attack_cycle_threshold:
                notifications.append(
                    f"HUMAN NOTIFICATION: sustained attack on '{sensor_id}' for {duration} cycles "
                    f"(>= {deps.sustained_attack_cycle_threshold}-cycle threshold), system remains safe but degraded."
                )

        # Section 10 trigger 1 ("multiple agents compromised") is intentionally not evaluated
        # here -- it needs Prober/Fingerprint agent health status, which don't exist in this
        # repo yet. Wire it in here once Member 3's agents report their own compromise state.

        return {"notifications": notifications}

    return escalate_and_notify


def build_overseer_graph(deps: OverseerDependencies):
    graph = StateGraph(OverseerState)
    graph.add_node("run_sentinel_pass", make_sentinel_pass_node(deps))
    graph.add_node("run_forensics_pass", make_forensics_pass_node(deps))
    graph.add_node("escalate_and_notify", make_escalate_node(deps))

    graph.set_entry_point("run_sentinel_pass")
    graph.add_edge("run_sentinel_pass", "run_forensics_pass")
    graph.add_edge("run_forensics_pass", "escalate_and_notify")
    graph.add_edge("escalate_and_notify", END)
    return graph.compile()
=======
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
>>>>>>> 67782d2bf638fc6e1aa240b226b523957b981a18
