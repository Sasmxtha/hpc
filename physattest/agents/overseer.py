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
