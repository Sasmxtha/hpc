"""
SWaT dataset configuration — sensor names, physical parameters,
coupling relationships, and safety bounds.

Maps SWaT's 51 sensors to PhysAttest components.
"""

import numpy as np
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════
# SWaT sensor names (from iTrust documentation)
# ═══════════════════════════════════════════════════════════════

# Process 1: Raw water supply
P1_SENSORS = ["FIT101", "LIT101", "MV101", "P101", "P102"]

# Process 2: Pre-treatment (chemical dosing)
P2_SENSORS = ["AIT201", "AIT202", "AIT203", "FIT201", "MV201", "P201",
              "P202", "P203", "P204", "P205", "P206"]

# Process 3: Ultrafiltration
P3_SENSORS = ["DPIT301", "FIT301", "LIT301", "MV301", "MV302",
              "MV303", "MV304", "P301", "P302"]

# Process 4: De-chlorination (UV treatment)
P4_SENSORS = ["AIT401", "AIT402", "FIT401", "LIT401", "P401",
              "P402", "P403", "P404", "UV401"]

# Process 5: Reverse osmosis
P5_SENSORS = ["AIT501", "AIT502", "AIT503", "AIT504", "FIT501",
              "FIT502", "FIT503", "FIT504", "MV501", "MV502",
              "MV503", "MV504", "P501", "P502"]

# Process 6: RO backwash
P6_SENSORS = ["FIT601", "P601", "P602", "P603"]

ALL_SENSORS = P1_SENSORS + P2_SENSORS + P3_SENSORS + P4_SENSORS + P5_SENSORS + P6_SENSORS

# Sensor type classification
LEVEL_SENSORS = ["LIT101", "LIT301", "LIT401"]
FLOW_SENSORS = ["FIT101", "FIT201", "FIT301", "FIT401", "FIT501",
                "FIT502", "FIT503", "FIT504", "FIT601"]
PRESSURE_SENSORS = ["DPIT301"]
CHEMICAL_SENSORS = ["AIT201", "AIT202", "AIT203", "AIT401", "AIT402",
                    "AIT501", "AIT502", "AIT503", "AIT504"]
ACTUATORS = [s for s in ALL_SENSORS if s.startswith(("MV", "P", "UV"))]
CONTINUOUS_SENSORS = LEVEL_SENSORS + FLOW_SENSORS + PRESSURE_SENSORS + CHEMICAL_SENSORS


@dataclass
class SWaTPhysics:
    """Physical parameters for the SWaT plant."""

    # Tank cross-section areas (m²)
    tank_areas = {
        "LIT101": 1.5,   # P1 raw water tank
        "LIT301": 2.0,   # P3 UF feed tank
        "LIT401": 1.0,   # P4 RO feed tank
    }

    # Safety bounds for level sensors (metres)
    level_bounds = {
        "LIT101": (250.0, 800.0),   # mm (SWaT uses mm)
        "LIT301": (250.0, 800.0),
        "LIT401": (250.0, 800.0),
    }

    # Safety bounds for flow sensors (m³/hr)
    flow_bounds = {
        "FIT101": (0.0, 3.0),
        "FIT201": (0.0, 3.0),
        "FIT301": (0.0, 3.0),
        "FIT401": (0.0, 3.0),
        "FIT501": (0.0, 2.0),
        "FIT502": (0.0, 2.0),
        "FIT503": (0.0, 2.0),
        "FIT504": (0.0, 2.0),
        "FIT601": (0.0, 2.0),
    }

    # Pressure bounds (kPa)
    pressure_bounds = {
        "DPIT301": (0.0, 50.0),
    }

    # Chemical bounds
    chemical_bounds = {
        "AIT201": (100.0, 300.0),   # conductivity (μS/cm)
        "AIT202": (6.0, 9.0),       # pH
        "AIT203": (100.0, 400.0),   # ORP (mV)
        "AIT401": (100.0, 300.0),   # conductivity
        "AIT402": (6.0, 9.0),       # pH
        "AIT501": (100.0, 400.0),   # conductivity
        "AIT502": (6.0, 9.0),       # pH
        "AIT503": (100.0, 400.0),   # ORP
        "AIT504": (100.0, 300.0),   # conductivity
    }


# ═══════════════════════════════════════════════════════════════
# Physics coupling graph
# ═══════════════════════════════════════════════════════════════

# Physical couplings: (sensor_a, sensor_b, coupling_type, strength)
# These encode conservation laws and physical relationships
COUPLING_EDGES = [
    # P1: mass conservation — inflow (FIT101) changes level (LIT101)
    ("FIT101", "LIT101", "mass_conservation", 0.9),
    # P1→P2: outflow from P1 = inflow to P2
    ("LIT101", "FIT201", "flow_continuity", 0.7),
    # P2: chemical dosing affects downstream quality
    ("AIT201", "AIT202", "chemical_equilibrium", 0.5),
    ("AIT202", "AIT203", "chemical_equilibrium", 0.5),
    # P2→P3: flow continuity
    ("FIT201", "FIT301", "flow_continuity", 0.8),
    ("FIT201", "LIT301", "mass_conservation", 0.7),
    # P3: pressure-flow relationship
    ("FIT301", "DPIT301", "bernoulli", 0.8),
    ("LIT301", "FIT301", "hydrostatic", 0.6),
    # P3→P4: flow continuity
    ("FIT301", "FIT401", "flow_continuity", 0.7),
    ("FIT301", "LIT401", "mass_conservation", 0.6),
    # P4: chemical treatment
    ("AIT401", "AIT402", "chemical_equilibrium", 0.5),
    # P4→P5: flow continuity
    ("FIT401", "FIT501", "flow_continuity", 0.7),
    # P5: RO membrane relationships
    ("FIT501", "FIT502", "mass_conservation", 0.8),  # feed = permeate + concentrate
    ("AIT501", "AIT502", "membrane_rejection", 0.6),
    ("AIT503", "AIT504", "membrane_rejection", 0.6),
    # P5→P6: backwash
    ("FIT503", "FIT601", "flow_continuity", 0.5),
]


# ═══════════════════════════════════════════════════════════════
# SWaT attack catalogue (36 attacks from the dataset)
# ═══════════════════════════════════════════════════════════════

ATTACK_CATALOGUE = [
    {"id": 1,  "target": "MV101",  "type": "actuator", "description": "Open MV101 to overflow P1 tank"},
    {"id": 2,  "target": "P102",   "type": "actuator", "description": "Turn off P102 to stop outflow"},
    {"id": 3,  "target": "LIT101", "type": "sensor",   "description": "Spoof LIT101 high to stop inflow"},
    {"id": 4,  "target": "LIT301", "type": "sensor",   "description": "Spoof LIT301 low to overflow P3"},
    {"id": 5,  "target": "AIT202", "type": "sensor",   "description": "Spoof pH to cause over-dosing"},
    {"id": 6,  "target": "FIT101", "type": "sensor",   "description": "Spoof flow to hide overflow"},
    {"id": 7,  "target": "LIT101", "type": "sensor",   "description": "Spoof LIT101 low gradually (stealth)"},
    {"id": 10, "target": "FIT401", "type": "sensor",   "description": "Spoof flow in P4"},
    {"id": 13, "target": "MV304",  "type": "actuator", "description": "Close MV304 to block UF"},
    {"id": 16, "target": "LIT301", "type": "sensor",   "description": "Spoof LIT301 high (stealth)"},
    {"id": 19, "target": "AIT504", "type": "sensor",   "description": "Spoof RO conductivity"},
    {"id": 25, "target": "LIT401", "type": "sensor",   "description": "Spoof LIT401 to overflow P4"},
    {"id": 30, "target": ["LIT101", "P101"], "type": "multi", "description": "Coordinated: spoof level + pump"},
    {"id": 31, "target": ["LIT401", "P402"], "type": "multi", "description": "Coordinated: spoof level + pump"},
    {"id": 36, "target": ["LIT301", "P302"], "type": "multi", "description": "Coordinated: spoof level + pump"},
]
