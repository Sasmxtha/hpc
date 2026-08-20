"""
Complete SWaT dataset column registry.

The SWaT testbed has 6 stages. Each stage has sensors (measure the water)
and actuators (control the water). Every column in the dataset is listed
here with its physical meaning, unit, typical range, and which stage it
belongs to.

Understanding what each sensor measures is critical — your observer's
conservation laws are relationships BETWEEN these columns.
"""

# -----------------------------------------------------------------------
# STAGE 1: Raw Water Intake
#
# What happens: Raw water enters through a motorized valve (MV101) into
# Tank 1. Pump P101 (or backup P102) pushes water to Stage 2.
#
# Conservation law: A₁ × dLIT101/dt = FIT101 (when MV101=2) - Q_P101 (when P101=2)
# -----------------------------------------------------------------------
STAGE_1 = {
    "FIT101":  {"type": "sensor",   "measures": "flow",     "unit": "m³/h",
                "range": (0, 5),    "desc": "Inflow to Tank 1 through MV101"},
    "LIT101":  {"type": "sensor",   "measures": "level",    "unit": "mm",
                "range": (0, 1000), "desc": "Water level in Tank 1"},
    "MV101":   {"type": "actuator", "measures": "valve",    "unit": "state",
                "range": (0, 2),    "desc": "Motorized valve: 0=closed, 1=closing, 2=open"},
    "P101":    {"type": "actuator", "measures": "pump",     "unit": "state",
                "range": (0, 2),    "desc": "Pump 1: 0=off, 1=starting, 2=running"},
    "P102":    {"type": "actuator", "measures": "pump",     "unit": "state",
                "range": (0, 2),    "desc": "Pump 2 (backup): 0=off, 1=starting, 2=running"},
}

# -----------------------------------------------------------------------
# STAGE 2: Chemical Dosing
#
# What happens: Water from Stage 1 flows through chemical dosing.
# NaOCl (sodium hypochlorite) is added by P201 to disinfect.
# HCl is added by P203 to control pH.
# NaOH is added by P204 for pH adjustment.
#
# Conservation laws:
#   dpH/dt = f(P201, P203, P204 dosing rates, flow, buffer capacity)
#   dORP/dt = f(P201 chlorine dosing, flow, decay rate)
#   dσ/dt = Q_in(σ_in - σ) / V  (conductivity is a conservative tracer)
# -----------------------------------------------------------------------
STAGE_2 = {
    "AIT201":  {"type": "sensor",   "measures": "pH",           "unit": "pH",
                "range": (6, 9),    "desc": "pH after chemical dosing"},
    "AIT202":  {"type": "sensor",   "measures": "ORP",          "unit": "mV",
                "range": (100, 500),"desc": "Oxidation-reduction potential (chlorine indicator)"},
    "AIT203":  {"type": "sensor",   "measures": "conductivity", "unit": "µS/cm",
                "range": (200, 600),"desc": "Conductivity (dissolved solids indicator)"},
    "FIT201":  {"type": "sensor",   "measures": "flow",         "unit": "m³/h",
                "range": (0, 5),    "desc": "Flow through chemical dosing section"},
    "MV201":   {"type": "actuator", "measures": "valve",        "unit": "state",
                "range": (0, 2),    "desc": "Dosing section inlet valve"},
    "P201":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "NaOCl dosing pump (chlorine)"},
    "P202":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "NaOCl dosing pump 2 (backup)"},
    "P203":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "HCl dosing pump (acid for pH down)"},
    "P204":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "NaOH dosing pump (base for pH up)"},
    "P205":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "NaOCl dosing pump 3"},
    "P206":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "HCl dosing pump 2"},
}

# -----------------------------------------------------------------------
# STAGE 3: Ultrafiltration (UF)
#
# What happens: Water is pushed through UF membranes by P301/P302.
# The membrane removes particles. Differential pressure (DPIT301)
# indicates membrane fouling. Tank 3 (LIT301) is the UF feed tank.
#
# Conservation laws:
#   A₃ × dLIT301/dt = Q_in - FIT301 (when P301=2)
#   FIT301 ∝ DPIT301 / (μ × R_membrane)   (Darcy's law)
#   DPIT301 increases with fouling, resets with backwash
# -----------------------------------------------------------------------
STAGE_3 = {
    "DPIT301": {"type": "sensor",   "measures": "pressure_diff","unit": "kPa",
                "range": (0, 100),  "desc": "Differential pressure across UF membrane"},
    "FIT301":  {"type": "sensor",   "measures": "flow",         "unit": "m³/h",
                "range": (0, 5),    "desc": "UF permeate flow (clean water out)"},
    "LIT301":  {"type": "sensor",   "measures": "level",        "unit": "mm",
                "range": (0, 1200), "desc": "Water level in UF feed tank (Tank 3)"},
    "MV301":   {"type": "actuator", "measures": "valve",        "unit": "state",
                "range": (0, 2),    "desc": "UF inlet valve"},
    "MV302":   {"type": "actuator", "measures": "valve",        "unit": "state",
                "range": (0, 2),    "desc": "UF outlet valve"},
    "MV303":   {"type": "actuator", "measures": "valve",        "unit": "state",
                "range": (0, 2),    "desc": "UF drain valve"},
    "MV304":   {"type": "actuator", "measures": "valve",        "unit": "state",
                "range": (0, 2),    "desc": "UF backwash valve"},
    "P301":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "UF feed pump"},
    "P302":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "UF feed pump 2 (backup)"},
}

# -----------------------------------------------------------------------
# STAGE 4: Dechlorination / RO Feed
#
# What happens: UV lamp (UV401) removes residual chlorine before
# reverse osmosis. Tank 4 (LIT401) feeds the RO system.
#
# Conservation laws:
#   A₄ × dLIT401/dt = Q_from_UF - FIT401 (when P401/P402=2)
#   UV dose = UV_intensity × exposure_time (affects chlorine removal)
# -----------------------------------------------------------------------
STAGE_4 = {
    "AIT401":  {"type": "sensor",   "measures": "hardness",     "unit": "mg/L CaCO₃",
                "range": (0, 500),  "desc": "Water hardness (calcium + magnesium)"},
    "AIT402":  {"type": "sensor",   "measures": "ORP",          "unit": "mV",
                "range": (100, 500),"desc": "ORP after dechlorination"},
    "FIT401":  {"type": "sensor",   "measures": "flow",         "unit": "m³/h",
                "range": (0, 5),    "desc": "Flow to RO system"},
    "LIT401":  {"type": "sensor",   "measures": "level",        "unit": "mm",
                "range": (0, 1000), "desc": "Water level in RO feed tank (Tank 4)"},
    "P401":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "RO booster pump"},
    "P402":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "RO booster pump 2 (backup)"},
    "P403":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "Dosing pump (anti-scalant for RO)"},
    "P404":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "Dosing pump (anti-scalant backup)"},
    "UV401":   {"type": "actuator", "measures": "UV",           "unit": "state",
                "range": (0, 2),    "desc": "UV dechlorinator lamp"},
}

# -----------------------------------------------------------------------
# STAGE 5: Reverse Osmosis (RO)
#
# What happens: High-pressure pump (P501/P502) forces water through
# RO membrane. Three output streams:
#   - Permeate (FIT502): clean water → storage
#   - Concentrate (FIT503): rejected contaminants → drain
#   - Recycle (FIT504): partial return to inlet
#
# Conservation laws:
#   FIT501 = FIT502 + FIT503 + FIT504   (Kirchhoff at RO junction)
#   FIT502 ∝ (PIT501 - PIT502 - Δπ)     (RO transport equation)
#   PIT503 ≈ PIT501 - friction            (concentrate side pressure)
#   AIT503 < AIT203                       (RO rejects dissolved solids)
# -----------------------------------------------------------------------
STAGE_5 = {
    "AIT501":  {"type": "sensor",   "measures": "pH",           "unit": "pH",
                "range": (6, 9),    "desc": "pH of RO inlet"},
    "AIT502":  {"type": "sensor",   "measures": "ORP",          "unit": "mV",
                "range": (100, 500),"desc": "ORP of RO inlet"},
    "AIT503":  {"type": "sensor",   "measures": "conductivity", "unit": "µS/cm",
                "range": (0, 100),  "desc": "RO permeate conductivity (should be low)"},
    "AIT504":  {"type": "sensor",   "measures": "ORP",          "unit": "mV",
                "range": (100, 500),"desc": "RO permeate ORP"},
    "FIT501":  {"type": "sensor",   "measures": "flow",         "unit": "m³/h",
                "range": (0, 5),    "desc": "RO inlet flow (total into membrane)"},
    "FIT502":  {"type": "sensor",   "measures": "flow",         "unit": "m³/h",
                "range": (0, 3),    "desc": "RO permeate flow (clean water out)"},
    "FIT503":  {"type": "sensor",   "measures": "flow",         "unit": "m³/h",
                "range": (0, 2),    "desc": "RO concentrate flow (rejected water)"},
    "FIT504":  {"type": "sensor",   "measures": "flow",         "unit": "m³/h",
                "range": (0, 2),    "desc": "RO recycle flow (returned to inlet)"},
    "PIT501":  {"type": "sensor",   "measures": "pressure",     "unit": "kPa",
                "range": (0, 300),  "desc": "RO inlet pressure (high, from pump)"},
    "PIT502":  {"type": "sensor",   "measures": "pressure",     "unit": "kPa",
                "range": (0, 50),   "desc": "RO permeate pressure (low, after membrane)"},
    "PIT503":  {"type": "sensor",   "measures": "pressure",     "unit": "kPa",
                "range": (0, 300),  "desc": "RO concentrate pressure"},
    "P501":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "RO high-pressure pump"},
    "P502":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "RO high-pressure pump 2 (backup)"},
}

# -----------------------------------------------------------------------
# STAGE 6: Permeate Storage / Backwash
#
# What happens: Clean permeate is stored. Periodically, stored water
# is used to backwash the UF membranes (Stage 3) to remove fouling.
#
# Conservation law:
#   dV_tank6/dt = FIT502 (in) - FIT601 (out for backwash)
# -----------------------------------------------------------------------
STAGE_6 = {
    "FIT601":  {"type": "sensor",   "measures": "flow",         "unit": "m³/h",
                "range": (0, 5),    "desc": "Backwash flow (from permeate tank to UF)"},
    "P601":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "Backwash pump"},
    "P602":    {"type": "actuator", "measures": "pump",         "unit": "state",
                "range": (0, 2),    "desc": "Backwash pump 2 (backup)"},
}

# -----------------------------------------------------------------------
# Combined column list (in the order they appear in the real dataset)
# -----------------------------------------------------------------------
ALL_STAGES = {}
for stage_dict in [STAGE_1, STAGE_2, STAGE_3, STAGE_4, STAGE_5, STAGE_6]:
    ALL_STAGES.update(stage_dict)

SENSOR_COLUMNS = [name for name, info in ALL_STAGES.items() if info["type"] == "sensor"]
ACTUATOR_COLUMNS = [name for name, info in ALL_STAGES.items() if info["type"] == "actuator"]
ALL_COLUMNS = list(ALL_STAGES.keys())

# The real dataset also has these columns:
EXTRA_COLUMNS = {
    "Timestamp":   "Date-time string, 1-second intervals",
    "Normal/Attack": "Label: 'Normal' or 'Attack' (+ attack name like 'A1', 'A2', ...)",
}


def print_dataset_summary():
    """Print a human-readable summary of all SWaT columns."""
    print(f"SWaT Dataset: {len(ALL_COLUMNS)} process columns + Timestamp + Label")
    print(f"  Sensors:   {len(SENSOR_COLUMNS)}")
    print(f"  Actuators: {len(ACTUATOR_COLUMNS)}")
    print()
    for stage_num, stage_dict in enumerate([STAGE_1, STAGE_2, STAGE_3, STAGE_4, STAGE_5, STAGE_6], 1):
        sensors = [n for n, i in stage_dict.items() if i["type"] == "sensor"]
        actuators = [n for n, i in stage_dict.items() if i["type"] == "actuator"]
        print(f"  Stage {stage_num}: {len(sensors)} sensors, {len(actuators)} actuators")
        for name, info in stage_dict.items():
            marker = "S" if info["type"] == "sensor" else "A"
            lo, hi = info["range"]
            print(f"    [{marker}] {name:10s} {info['unit']:>12s}  [{lo:>6.0f} - {hi:>6.0f}]  {info['desc']}")
        print()


if __name__ == "__main__":
    print_dataset_summary()
