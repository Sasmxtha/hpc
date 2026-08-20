"""
PhysAttest Agent Graph — 5-agent architecture in LangGraph.

Graph structure:

    overseer_pre → sentinel → [prober?] → [fingerprint?] → guardian → overseer_post
                                 ↑              ↑
                            (only at L2+)   (only at L3+)

Conditional edges activate Prober and Fingerprint only when the
defense level requires them. This saves compute in normal operation
(L1 runs only Sentinel + Guardian).

No agent trusts another. Every agent reads shared state but writes
only its own fields. Decisions are grounded in math evidence.
"""

from langgraph.graph import StateGraph, END
from .state import AgentState, DefenseLevel
from .overseer import overseer_pre_node, overseer_post_node
from .sentinel import sentinel_node
from .prober import prober_node
from .fingerprint_agent import fingerprint_node
from .guardian import guardian_node


def _should_probe(state: AgentState) -> str:
    """Route to Prober if defense level >= L2, else skip to Fingerprint check."""
    level = state.get("defense_level", DefenseLevel.L1_MULTI_DOMAIN)
    if level >= DefenseLevel.L2_ACTIVE_PROBING:
        return "prober"
    return "check_fingerprint"


def _should_fingerprint(state: AgentState) -> str:
    """Route to Fingerprint if defense level >= L3, else skip to Guardian."""
    level = state.get("defense_level", DefenseLevel.L1_MULTI_DOMAIN)
    if level >= DefenseLevel.L3_FINGERPRINTING:
        return "fingerprint"
    return "guardian"


def build_physattest_graph() -> StateGraph:
    """
    Build the PhysAttest agent graph.

    Returns a compiled LangGraph that processes one defense cycle.
    Call .invoke(state) with raw sensor readings and agent command
    to get the full defense output.
    """
    graph = StateGraph(AgentState)

    # Add all agent nodes
    graph.add_node("overseer_pre", overseer_pre_node)
    graph.add_node("sentinel", sentinel_node)
    graph.add_node("prober", prober_node)
    graph.add_node("check_fingerprint", lambda s: s)  # routing node
    graph.add_node("fingerprint", fingerprint_node)
    graph.add_node("guardian", guardian_node)
    graph.add_node("overseer_post", overseer_post_node)

    # Set entry point
    graph.set_entry_point("overseer_pre")

    # Wire the graph
    graph.add_edge("overseer_pre", "sentinel")

    # Sentinel → conditionally Prober or skip
    graph.add_conditional_edges(
        "sentinel",
        _should_probe,
        {
            "prober": "prober",
            "check_fingerprint": "check_fingerprint",
        },
    )

    # Prober → check fingerprint
    graph.add_edge("prober", "check_fingerprint")

    # Check fingerprint → conditionally Fingerprint or skip to Guardian
    graph.add_conditional_edges(
        "check_fingerprint",
        _should_fingerprint,
        {
            "fingerprint": "fingerprint",
            "guardian": "guardian",
        },
    )

    # Fingerprint → Guardian
    graph.add_edge("fingerprint", "guardian")

    # Guardian → Overseer post
    graph.add_edge("guardian", "overseer_post")

    # Overseer post → END
    graph.add_edge("overseer_post", END)

    return graph.compile()


def run_cycle(
    graph,
    raw_readings: list[float],
    agent_command: list[float] | None = None,
    prev_state: dict | None = None,
) -> dict:
    """
    Run one complete defense cycle.

    Args:
        graph: compiled PhysAttest graph
        raw_readings: sensor readings from the plant
        agent_command: LLM agent's proposed actuator command
        prev_state: state from previous cycle (for continuity)

    Returns:
        Full state dict after all agents have run
    """
    state: AgentState = {
        "raw_readings": raw_readings,
        "timestamp": 0.0,
        "messages": [],
    }

    # Carry forward persistent state from previous cycle
    if prev_state:
        for key in [
            "defense_level", "cycle_count", "agent_health",
            "human_notified", "health_scores", "verified_readings",
            "fingerprint_status", "fingerprint_scores",
        ]:
            if key in prev_state:
                state[key] = prev_state[key]

    if agent_command is not None:
        state["agent_command"] = agent_command

    return graph.invoke(state)
