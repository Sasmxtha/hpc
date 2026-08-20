<<<<<<< HEAD
"""Shared state schemas and dependency-injection containers for the LangGraph agents.

Scope note: this project has 5 agents in the full design (Overseer, Sentinel, Prober,
Fingerprint, Guardian), owned by Member 3. What's wired up here is Sentinel -- the agent
that actually consumes all four of Member 2's components (anomaly transformer, GNN coupling
discovery, 3-way classifier, LLM forensics engine) -- plus a thin Overseer wrapper for the
cross-sensor aggregation and Section 10 human-notification rules that Sentinel alone can't
do. Guardian (CBF enforcement), Prober (active probing), and Fingerprint (hardware signature
checks) need components that don't exist in this repo yet (CBF filter, bidirectional shield,
noise fingerprinting), so they're represented here only as explicit stub interfaces -- see
the STUB comments in sentinel.py -- ready for Member 3 to fill in without needing to touch
the routing logic around them.

Why dependencies are injected via a dataclass instead of module-level globals: LangGraph
nodes are plain functions of (state) -> partial_state_update, with no built-in way to hand
them extra arguments. The usual pattern is a factory that closes over a dependencies object,
which is what build_sentinel_graph(deps) / build_overseer_graph(deps) do -- keeps the trained
model, coupling graph, and LLM chain out of global state and makes the graphs easy to
rebuild with different models/thresholds for testing.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

import networkx as nx

from physattest.ai.classifier_3way import ClassificationResult
from physattest.ai.forensics_llm import ForensicsReport
from physattest.ai.llm_fallback import FallbackChain
from physattest.ml.anomaly_transformer import ResidualAnomalyTransformer


class SentinelState(TypedDict, total=False):
    # --- inputs, set by the Overseer before invoking the Sentinel graph ---
    sensor_id: str
    residual_window: List[float]  # this sensor's own recent residual history
    neighbour_windows: Dict[str, List[float]]  # coupled neighbours' residual histories
    health_score: float  # health score entering this cycle
    health_history: List[float]
    command_explains_change: bool  # STUB input -- real value from Member 3's command log
    fingerprint_changed: bool  # STUB input -- real value from Member 3's fingerprint agent

    # --- populated by nodes as the graph runs ---
    transformer_score: float
    timestep_scores: List[float]
    new_health_score: float
    evidence: Any  # physattest.ai.classifier_3way.SensorEvidence
    classification: ClassificationResult
    response_action: str
    log_messages: List[str]


class OverseerState(TypedDict, total=False):
    cycle_index: int
    sensor_windows: Dict[str, List[float]]
    coupling_neighbours: Dict[str, List[str]]
    sensor_results: Dict[str, SentinelState]
    attacked_sensors: List[str]
    forensics_reports: Dict[str, ForensicsReport]
    notifications: List[str]


@dataclass
class SentinelDependencies:
    """Everything the Sentinel graph's nodes need, injected once at graph-build time."""

    transformer: ResidualAnomalyTransformer  # Component 6, instantiated with n_sensors=1
    anomaly_threshold: float = 0.5
    health_decay_rate: float = 8.0
    health_recovery_rate: float = 1.0
    llm_chain: Optional[FallbackChain] = None  # None -> classifier falls straight to rules


@dataclass
class PlantMemory:
    """Cross-cycle state (health scores/history, sustained-attack timers) for the Overseer.

    Kept as a plain Python object outside LangGraph's state, deliberately: LangGraph state
    is naturally scoped to one graph invocation. Making it survive across many invocations
    (one per timestep, indefinitely) would mean reaching for LangGraph's checkpointer /
    persistence layer, which is built for durability across process restarts -- real
    overkill for what's actually needed here, which is "remember a few numbers between the
    Overseer's next call in the same process."
    """

    health_scores: Dict[str, float] = field(default_factory=dict)
    health_history: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    attack_since_cycle: Dict[str, int] = field(default_factory=dict)
    last_attacked_cycle: Dict[str, int] = field(default_factory=dict)


@dataclass
class OverseerDependencies:
    sentinel_graph: Any  # compiled StateGraph from build_sentinel_graph
    causal_graph: nx.DiGraph  # Component: forensics causal graph (real plant graph or demo stand-in)
    forensics_chain: Optional[FallbackChain] = None
    sustained_attack_cycle_threshold: int = 30  # Section 10 rule 3: sustained attack > 30 min
    sustained_attack_grace_cycles: int = 2  # tolerate this many consecutive non-attack cycles before resetting the timer
    memory: PlantMemory = field(default_factory=PlantMemory)
=======
"""
Shared state for the PhysAttest agent graph.

This TypedDict flows through every node in the LangGraph.
Each agent reads what it needs and writes its outputs.
No agent modifies another agent's fields.
"""

from typing import TypedDict, Optional
from enum import IntEnum
import numpy as np


class DefenseLevel(IntEnum):
    L1_MULTI_DOMAIN = 1    # observer + transformer + classifier
    L2_ACTIVE_PROBING = 2  # + random perturbations
    L3_FINGERPRINTING = 3  # + hardware noise analysis
    L4_CBF_BOUNDING = 4    # everything failed, verify consequences


class AlertSeverity(IntEnum):
    NONE = 0
    LOW = 1       # anomaly, possibly benign
    MEDIUM = 2    # confirmed suspicious
    HIGH = 3      # confirmed attack
    CRITICAL = 4  # multi-sensor coordinated attack


class AgentState(TypedDict, total=False):
    # --- Plant data (input each cycle) ---
    raw_readings: list[float]
    timestamp: float

    # --- Overseer fields ---
    defense_level: int
    cycle_count: int
    agent_health: dict[str, bool]
    human_notified: bool
    alert_severity: int
    escalation_reason: str

    # --- Sentinel fields ---
    verified_readings: list[float]
    residuals: list[float]
    normalised_residuals: list[float]
    blocked_sensors: list[int]
    suspicious_sensors: list[int]
    sensor_statuses: list[str]
    health_scores: list[float]
    classification: dict          # {sensor_id: "fault"|"anomaly"|"attack"}
    forensics_report: str

    # --- Prober fields ---
    probe_active: bool
    probe_perturbation: list[float]
    probe_responses: list[float]
    probe_verdict: dict           # {sensor_id: "responsive"|"unresponsive"}

    # --- Fingerprint fields ---
    fingerprint_status: dict[int, str]  # {sensor_id: "authentic"|"compromised"|"unknown"}
    fingerprint_scores: list[float]

    # --- Guardian fields ---
    agent_command: list[float]
    safe_command: list[float]
    command_status: str           # "passed"|"modified"|"emergency_stop"
    command_intervention: float
    cbf_feasible: bool

    # --- Accumulated messages for logging ---
    messages: list[str]
>>>>>>> 67782d2bf638fc6e1aa240b226b523957b981a18
