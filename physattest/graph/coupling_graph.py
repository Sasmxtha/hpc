"""
Coupling Graph for PhysAttest (Component 2).

A graph G = (V, E) where:
  - Nodes V are sensors/actuators
  - Edges E are physical couplings with weights from conservation laws

The denser this graph, the more sensors an attacker must compromise
simultaneously (Theorem 1). Each edge is a constraint the attacker
must satisfy.

Edge types:
  1. Hydraulic:   sensors on the same tank/pipe (mass conservation)
  2. Inter-stage: sensors connected through pipes between stages
  3. Physics:     pressure-level (P = ρgh), flow-velocity (Q = Av)
  4. Chemistry:   pH-ORP-conductivity correlations from dosing
  5. Discovered:  hidden couplings found by GNN (Member 2)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class CouplingType(Enum):
    HYDRAULIC = "hydraulic"
    INTER_STAGE = "inter_stage"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    DISCOVERED = "discovered"


class CouplingDomain(Enum):
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    MATH = "math"


@dataclass
class CouplingEdge:
    source: str
    target: str
    weight: float
    coupling_type: CouplingType
    domain: CouplingDomain
    equation: str           # human-readable conservation law backing this edge
    bidirectional: bool = True


# -----------------------------------------------------------------------
# SWaT sensor/actuator registry
# -----------------------------------------------------------------------
SWAT_SENSORS = {
    # Stage 1: Raw water intake
    "LIT101":  {"stage": 1, "type": "level",        "unit": "mm",    "desc": "Raw water tank level"},
    "FIT101":  {"stage": 1, "type": "flow",         "unit": "m³/h",  "desc": "Raw water inflow"},
    "MV101":   {"stage": 1, "type": "valve",        "unit": "0/1",   "desc": "Intake valve"},
    "P101":    {"stage": 1, "type": "pump",         "unit": "0/1",   "desc": "Raw water pump"},
    "P102":    {"stage": 1, "type": "pump",         "unit": "0/1",   "desc": "Raw water pump 2"},
    # Stage 2: Chemical dosing
    "AIT201":  {"stage": 2, "type": "pH",           "unit": "pH",    "desc": "pH analyzer"},
    "AIT202":  {"stage": 2, "type": "ORP",          "unit": "mV",    "desc": "ORP meter"},
    "AIT203":  {"stage": 2, "type": "conductivity", "unit": "µS/cm", "desc": "Conductivity"},
    "FIT201":  {"stage": 2, "type": "flow",         "unit": "m³/h",  "desc": "Dosing flow"},
    "P201":    {"stage": 2, "type": "pump",         "unit": "0/1",   "desc": "NaOCl dosing pump"},
    "P203":    {"stage": 2, "type": "pump",         "unit": "0/1",   "desc": "HCl dosing pump"},
    # Stage 3: Ultrafiltration
    "LIT301":  {"stage": 3, "type": "level",        "unit": "mm",    "desc": "UF feed tank level"},
    "FIT301":  {"stage": 3, "type": "flow",         "unit": "m³/h",  "desc": "UF permeate flow"},
    "DPIT301": {"stage": 3, "type": "pressure",     "unit": "kPa",   "desc": "UF differential pressure"},
    "P301":    {"stage": 3, "type": "pump",         "unit": "0/1",   "desc": "UF feed pump"},
    "P302":    {"stage": 3, "type": "pump",         "unit": "0/1",   "desc": "UF feed pump 2"},
    # Stage 4: Dechlorination
    "LIT401":  {"stage": 4, "type": "level",        "unit": "mm",    "desc": "RO feed tank level"},
    "FIT401":  {"stage": 4, "type": "flow",         "unit": "m³/h",  "desc": "RO feed flow"},
    "AIT401":  {"stage": 4, "type": "hardness",     "unit": "mg/L",  "desc": "Water hardness"},
    "UV401":   {"stage": 4, "type": "UV",           "unit": "0/1",   "desc": "UV dechlorinator"},
    # Stage 5: Reverse osmosis
    "AIT501":  {"stage": 5, "type": "pH",           "unit": "pH",    "desc": "RO pH"},
    "AIT502":  {"stage": 5, "type": "ORP",          "unit": "mV",    "desc": "RO ORP"},
    "AIT503":  {"stage": 5, "type": "conductivity", "unit": "µS/cm", "desc": "RO conductivity"},
    "FIT501":  {"stage": 5, "type": "flow",         "unit": "m³/h",  "desc": "RO inlet flow"},
    "FIT502":  {"stage": 5, "type": "flow",         "unit": "m³/h",  "desc": "RO permeate flow"},
    "FIT503":  {"stage": 5, "type": "flow",         "unit": "m³/h",  "desc": "RO concentrate flow"},
    "FIT504":  {"stage": 5, "type": "flow",         "unit": "m³/h",  "desc": "RO recycle flow"},
    "PIT501":  {"stage": 5, "type": "pressure",     "unit": "kPa",   "desc": "RO inlet pressure"},
    "PIT502":  {"stage": 5, "type": "pressure",     "unit": "kPa",   "desc": "RO permeate pressure"},
    "PIT503":  {"stage": 5, "type": "pressure",     "unit": "kPa",   "desc": "RO concentrate pressure"},
    "P501":    {"stage": 5, "type": "pump",         "unit": "0/1",   "desc": "RO high-pressure pump"},
    "P502":    {"stage": 5, "type": "pump",         "unit": "0/1",   "desc": "RO high-pressure pump 2"},
    # Stage 6: Permeate storage
    "FIT601":  {"stage": 6, "type": "flow",         "unit": "m³/h",  "desc": "Backwash flow"},
    "P601":    {"stage": 6, "type": "pump",         "unit": "0/1",   "desc": "Backwash pump"},
    "P602":    {"stage": 6, "type": "pump",         "unit": "0/1",   "desc": "Backwash pump 2"},
}


def build_swat_coupling_graph() -> nx.Graph:
    """
    Build the coupling graph for SWaT from plant engineering documentation.

    Each edge represents a conservation law that constrains the relationship
    between two sensors. An attacker spoofing one sensor must also spoof
    all its neighbours to remain consistent.

    Returns a weighted NetworkX graph.
    """
    G = nx.Graph()

    # Add all sensors as nodes
    for name, info in SWAT_SENSORS.items():
        G.add_node(name, **info)

    # Define all physical couplings
    edges = _get_swat_couplings()

    for edge in edges:
        G.add_edge(
            edge.source, edge.target,
            weight=edge.weight,
            coupling_type=edge.coupling_type.value,
            domain=edge.domain.value,
            equation=edge.equation,
        )

    return G


def _get_swat_couplings() -> List[CouplingEdge]:
    """
    Every edge is backed by a specific conservation law or algebraic identity.
    Weights reflect coupling strength: stronger coupling = harder to spoof.

    Weight scale (0 to 1):
      1.0 = exact algebraic identity (V = Ah, Q_in = Q_out at junction)
      0.8 = strong physics (mass conservation across a tank)
      0.5 = moderate (inter-stage flow coupling with transit delay)
      0.3 = weak (indirect chemistry correlation)
      0.1 = very weak (GNN-discovered, not yet validated)
    """
    edges = []

    # ===================================================================
    # STAGE 1: Raw water intake
    # ===================================================================

    # LIT101 ↔ FIT101: mass conservation in tank 1
    # dh/dt = (Q_in - Q_out) / A  →  if one changes, the other must explain it
    edges.append(CouplingEdge(
        "LIT101", "FIT101", weight=0.9,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.PHYSICS,
        equation="A₁ × dLIT101/dt = FIT101 - Q_P101",
    ))

    # LIT101 ↔ MV101: valve controls inflow
    edges.append(CouplingEdge(
        "LIT101", "MV101", weight=0.8,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.PHYSICS,
        equation="MV101=1 → FIT101>0 → dLIT101/dt changes",
    ))

    # LIT101 ↔ P101: pump controls outflow
    edges.append(CouplingEdge(
        "LIT101", "P101", weight=0.8,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.PHYSICS,
        equation="P101=1 → Q_out>0 → dLIT101/dt decreases",
    ))

    # FIT101 ↔ MV101: flow depends on valve state
    edges.append(CouplingEdge(
        "FIT101", "MV101", weight=1.0,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.MATH,
        equation="MV101=0 → FIT101=0 (algebraic identity)",
    ))

    # P101 ↔ P102: redundant pumps — only one active at a time
    edges.append(CouplingEdge(
        "P101", "P102", weight=0.7,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.MATH,
        equation="P101 + P102 ≤ 1 (mutex constraint)",
    ))

    # ===================================================================
    # STAGE 1 → STAGE 2: inter-stage flow coupling
    # ===================================================================

    # P101 output = stage 2 input
    edges.append(CouplingEdge(
        "P101", "FIT201", weight=0.7,
        coupling_type=CouplingType.INTER_STAGE,
        domain=CouplingDomain.PHYSICS,
        equation="Q_P101 = FIT201 (mass conservation in connecting pipe)",
    ))

    edges.append(CouplingEdge(
        "FIT101", "FIT201", weight=0.6,
        coupling_type=CouplingType.INTER_STAGE,
        domain=CouplingDomain.PHYSICS,
        equation="FIT101 ≈ FIT201 when MV101=1, P101=1 (flow continuity)",
    ))

    # ===================================================================
    # STAGE 2: Chemical dosing
    # ===================================================================

    # AIT201 ↔ P201: NaOCl dosing affects pH
    edges.append(CouplingEdge(
        "AIT201", "P201", weight=0.7,
        coupling_type=CouplingType.CHEMISTRY,
        domain=CouplingDomain.CHEMISTRY,
        equation="dpH/dt = f(P201_state) / (β × V₂)",
    ))

    # AIT201 ↔ P203: HCl dosing affects pH (opposite direction)
    edges.append(CouplingEdge(
        "AIT201", "P203", weight=0.7,
        coupling_type=CouplingType.CHEMISTRY,
        domain=CouplingDomain.CHEMISTRY,
        equation="dpH/dt = -f(P203_state) / (β × V₂)",
    ))

    # AIT201 ↔ AIT202: pH and ORP are chemically correlated
    # Higher chlorine → higher ORP AND higher pH (for NaOCl)
    edges.append(CouplingEdge(
        "AIT201", "AIT202", weight=0.5,
        coupling_type=CouplingType.CHEMISTRY,
        domain=CouplingDomain.CHEMISTRY,
        equation="NaOCl dosing raises both pH and ORP simultaneously",
    ))

    # AIT202 ↔ P201: chlorine dosing affects ORP
    edges.append(CouplingEdge(
        "AIT202", "P201", weight=0.7,
        coupling_type=CouplingType.CHEMISTRY,
        domain=CouplingDomain.CHEMISTRY,
        equation="dORP/dt = k_Cl × Q_P201 / V₂",
    ))

    # AIT203 (conductivity) ↔ FIT201: dilution affects conductivity
    edges.append(CouplingEdge(
        "AIT203", "FIT201", weight=0.4,
        coupling_type=CouplingType.CHEMISTRY,
        domain=CouplingDomain.CHEMISTRY,
        equation="dσ/dt = FIT201 × (σ_in - σ) / V₂",
    ))

    # ===================================================================
    # STAGE 2 → STAGE 3: inter-stage
    # ===================================================================

    edges.append(CouplingEdge(
        "FIT201", "LIT301", weight=0.6,
        coupling_type=CouplingType.INTER_STAGE,
        domain=CouplingDomain.PHYSICS,
        equation="FIT201 contributes to dLIT301/dt (mass conservation)",
    ))

    # ===================================================================
    # STAGE 3: Ultrafiltration
    # ===================================================================

    # LIT301 ↔ FIT301: mass conservation in UF feed tank
    edges.append(CouplingEdge(
        "LIT301", "FIT301", weight=0.9,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.PHYSICS,
        equation="A₃ × dLIT301/dt = Q_in - FIT301",
    ))

    # LIT301 ↔ P301: pump controls outflow
    edges.append(CouplingEdge(
        "LIT301", "P301", weight=0.8,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.PHYSICS,
        equation="P301=1 → Q_out>0 → dLIT301/dt decreases",
    ))

    # DPIT301 ↔ FIT301: pressure-flow across membrane (Darcy's law)
    edges.append(CouplingEdge(
        "DPIT301", "FIT301", weight=0.8,
        coupling_type=CouplingType.PHYSICS,
        domain=CouplingDomain.PHYSICS,
        equation="FIT301 = DPIT301 × A_membrane / (μ × R_membrane)",
    ))

    # DPIT301 ↔ P301: pump drives pressure
    edges.append(CouplingEdge(
        "DPIT301", "P301", weight=0.7,
        coupling_type=CouplingType.PHYSICS,
        domain=CouplingDomain.PHYSICS,
        equation="P301=1 → DPIT301 increases (pump pressurizes membrane)",
    ))

    # P301 ↔ P302: redundant pumps
    edges.append(CouplingEdge(
        "P301", "P302", weight=0.7,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.MATH,
        equation="P301 + P302 ≤ 1 (mutex)",
    ))

    # ===================================================================
    # STAGE 3 → STAGE 4: inter-stage
    # ===================================================================

    edges.append(CouplingEdge(
        "FIT301", "LIT401", weight=0.6,
        coupling_type=CouplingType.INTER_STAGE,
        domain=CouplingDomain.PHYSICS,
        equation="FIT301 → dLIT401/dt (permeate fills RO feed tank)",
    ))

    # ===================================================================
    # STAGE 4: Dechlorination
    # ===================================================================

    # LIT401 ↔ FIT401: mass conservation
    edges.append(CouplingEdge(
        "LIT401", "FIT401", weight=0.9,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.PHYSICS,
        equation="A₄ × dLIT401/dt = Q_in - FIT401",
    ))

    # UV401 ↔ AIT401: UV treatment affects hardness measurement
    edges.append(CouplingEdge(
        "UV401", "AIT401", weight=0.3,
        coupling_type=CouplingType.CHEMISTRY,
        domain=CouplingDomain.CHEMISTRY,
        equation="UV treatment may affect dissolved mineral readings",
    ))

    # ===================================================================
    # STAGE 4 → STAGE 5: inter-stage
    # ===================================================================

    edges.append(CouplingEdge(
        "FIT401", "FIT501", weight=0.6,
        coupling_type=CouplingType.INTER_STAGE,
        domain=CouplingDomain.PHYSICS,
        equation="FIT401 ≈ FIT501 (flow continuity, stage 4 → 5)",
    ))

    # ===================================================================
    # STAGE 5: Reverse osmosis
    # ===================================================================

    # Flow conservation at RO: inlet = permeate + concentrate + recycle
    edges.append(CouplingEdge(
        "FIT501", "FIT502", weight=0.9,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.MATH,
        equation="FIT501 = FIT502 + FIT503 (+ FIT504 recycle)",
    ))

    edges.append(CouplingEdge(
        "FIT501", "FIT503", weight=0.9,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.MATH,
        equation="FIT501 = FIT502 + FIT503 (Kirchhoff for fluids)",
    ))

    edges.append(CouplingEdge(
        "FIT502", "FIT503", weight=0.8,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.MATH,
        equation="FIT502 + FIT503 = FIT501 (complementary streams)",
    ))

    # Pressure-flow: RO membrane relationship
    edges.append(CouplingEdge(
        "PIT501", "FIT502", weight=0.7,
        coupling_type=CouplingType.PHYSICS,
        domain=CouplingDomain.PHYSICS,
        equation="FIT502 ∝ (PIT501 - PIT502 - Δπ) (RO transport eq.)",
    ))

    edges.append(CouplingEdge(
        "PIT501", "PIT502", weight=0.8,
        coupling_type=CouplingType.PHYSICS,
        domain=CouplingDomain.PHYSICS,
        equation="PIT502 < PIT501 always (pressure drop across membrane)",
    ))

    edges.append(CouplingEdge(
        "PIT501", "PIT503", weight=0.7,
        coupling_type=CouplingType.PHYSICS,
        domain=CouplingDomain.PHYSICS,
        equation="PIT503 ≈ PIT501 - friction_loss (concentrate side)",
    ))

    # RO pump drives inlet pressure
    edges.append(CouplingEdge(
        "P501", "PIT501", weight=0.8,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.PHYSICS,
        equation="P501=1 → PIT501 rises (pump pressurizes feed)",
    ))

    edges.append(CouplingEdge(
        "P501", "P502", weight=0.7,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.MATH,
        equation="P501 + P502 ≤ 1 (mutex)",
    ))

    # Chemistry at RO: conductivity rejection
    edges.append(CouplingEdge(
        "AIT503", "AIT203", weight=0.4,
        coupling_type=CouplingType.CHEMISTRY,
        domain=CouplingDomain.CHEMISTRY,
        equation="AIT503 < AIT203 always (RO rejects dissolved solids)",
    ))

    edges.append(CouplingEdge(
        "AIT501", "AIT201", weight=0.3,
        coupling_type=CouplingType.CHEMISTRY,
        domain=CouplingDomain.CHEMISTRY,
        equation="AIT501 tracks AIT201 with transport delay",
    ))

    # ===================================================================
    # STAGE 5 → STAGE 6: inter-stage
    # ===================================================================

    edges.append(CouplingEdge(
        "FIT502", "FIT601", weight=0.5,
        coupling_type=CouplingType.INTER_STAGE,
        domain=CouplingDomain.PHYSICS,
        equation="RO permeate → backwash supply (flow coupling)",
    ))

    # ===================================================================
    # STAGE 6: Backwash
    # ===================================================================

    edges.append(CouplingEdge(
        "FIT601", "P601", weight=0.8,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.PHYSICS,
        equation="P601=1 → FIT601>0 (pump drives backwash flow)",
    ))

    edges.append(CouplingEdge(
        "P601", "P602", weight=0.7,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.MATH,
        equation="P601 + P602 ≤ 1 (mutex)",
    ))

    # Backwash affects UF membrane pressure
    edges.append(CouplingEdge(
        "FIT601", "DPIT301", weight=0.5,
        coupling_type=CouplingType.INTER_STAGE,
        domain=CouplingDomain.PHYSICS,
        equation="Backwash reduces DPIT301 (cleans membrane)",
    ))

    # ===================================================================
    # Missing couplings for isolated sensors
    # ===================================================================

    # AIT502 (RO ORP) ↔ P201: chlorine dosing in stage 2 propagates to stage 5
    edges.append(CouplingEdge(
        "AIT502", "AIT202", weight=0.3,
        coupling_type=CouplingType.CHEMISTRY,
        domain=CouplingDomain.CHEMISTRY,
        equation="AIT502 tracks AIT202 with transport delay (same chlorine)",
    ))

    # AIT502 ↔ FIT501: ORP depends on flow rate through RO
    edges.append(CouplingEdge(
        "AIT502", "FIT501", weight=0.4,
        coupling_type=CouplingType.CHEMISTRY,
        domain=CouplingDomain.CHEMISTRY,
        equation="Higher flow dilutes chlorine → lowers ORP",
    ))

    # FIT504 (RO recycle) ↔ FIT501: recycle is a fraction of inlet flow
    edges.append(CouplingEdge(
        "FIT504", "FIT501", weight=0.7,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.MATH,
        equation="FIT504 = FIT501 - FIT502 - FIT503 (flow balance)",
    ))

    # FIT504 ↔ FIT502: recycle + permeate + concentrate = inlet
    edges.append(CouplingEdge(
        "FIT504", "FIT502", weight=0.6,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.MATH,
        equation="FIT504 + FIT502 + FIT503 = FIT501",
    ))

    # AIT401 (hardness) ↔ AIT203 (conductivity): hardness correlates with conductivity
    edges.append(CouplingEdge(
        "AIT401", "AIT203", weight=0.3,
        coupling_type=CouplingType.CHEMISTRY,
        domain=CouplingDomain.CHEMISTRY,
        equation="Hardness (Ca²⁺, Mg²⁺) contributes to conductivity",
    ))

    # AIT401 ↔ LIT401: same tank
    edges.append(CouplingEdge(
        "AIT401", "LIT401", weight=0.4,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.PHYSICS,
        equation="AIT401 and LIT401 measure the same tank's water",
    ))

    # AIT501 ↔ FIT501: pH depends on flow rate
    edges.append(CouplingEdge(
        "AIT501", "FIT501", weight=0.4,
        coupling_type=CouplingType.CHEMISTRY,
        domain=CouplingDomain.CHEMISTRY,
        equation="Higher flow dilutes chemical concentrations → pH shifts",
    ))

    # AIT503 ↔ FIT502: conductivity rejection depends on permeate flow
    edges.append(CouplingEdge(
        "AIT503", "FIT502", weight=0.5,
        coupling_type=CouplingType.CHEMISTRY,
        domain=CouplingDomain.CHEMISTRY,
        equation="Higher permeate flow at constant pressure → lower rejection → higher AIT503",
    ))

    # P501 ↔ FIT501: pump drives flow
    edges.append(CouplingEdge(
        "P501", "FIT501", weight=0.8,
        coupling_type=CouplingType.HYDRAULIC,
        domain=CouplingDomain.PHYSICS,
        equation="P501=1 → FIT501>0 (pump drives RO feed flow)",
    ))

    # LIT401 ↔ FIT501: tank feeds RO
    edges.append(CouplingEdge(
        "LIT401", "FIT501", weight=0.6,
        coupling_type=CouplingType.INTER_STAGE,
        domain=CouplingDomain.PHYSICS,
        equation="dLIT401/dt depends on FIT501 outflow",
    ))

    return edges
