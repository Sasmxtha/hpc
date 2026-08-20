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
