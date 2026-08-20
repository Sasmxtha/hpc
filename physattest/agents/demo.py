"""
Full demo of the PhysAttest 5-agent architecture.

Runs multiple cycles simulating:
1. Normal operation (L1)
2. Sensor spoofing attack → escalation to L2, L3
3. Hijacked agent command → Guardian blocks it
4. Coordinated attack → CRITICAL alert + human notification
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from agents.graph import build_physattest_graph, run_cycle


def print_cycle(result: dict, label: str = ""):
    cycle = result.get("cycle_count", 0)
    level = result.get("defense_level", 1)
    severity = result.get("alert_severity", 0)
    sev_names = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}

    print(f"\n{'─' * 65}")
    if label:
        print(f"  {label}")
    print(f"  Cycle {cycle} | Defense L{level} | Severity {sev_names.get(severity, '?')}")
    print(f"{'─' * 65}")

    for msg in result.get("messages", []):
        print(f"  {msg}")

    blocked = result.get("blocked_sensors", [])
    if blocked:
        print(f"  Blocked sensors: {blocked}")

    cmd_status = result.get("command_status", "")
    if cmd_status and cmd_status != "no_command":
        safe = result.get("safe_command", [])
        interv = result.get("command_intervention", 0)
        print(f"  Command: {cmd_status} (intervention={interv:.4f})")

    health = result.get("agent_health", {})
    if health:
        status = "ALL OK" if all(health.values()) else f"DEGRADED: {health}"
        print(f"  Agent health: {status}")


def main():
    print("=" * 65)
    print("  PhysAttest 5-Agent Architecture — Full Demo")
    print("  Overseer → Sentinel → [Prober] → [Fingerprint] → Guardian")
    print("=" * 65)

    graph = build_physattest_graph()
    np.random.seed(42)
    prev_state = None

    # ── Phase 1: Normal operation (10 cycles) ──
    print("\n\n▶ PHASE 1: Normal operation")
    for i in range(10):
        readings = (np.random.randn(6) * 0.02 + 0.5).tolist()
        cmd = [0.3, 0.3, 0.3, 0.5, 0.5, 0.5]
        result = run_cycle(graph, readings, cmd, prev_state)
        prev_state = result

    print_cycle(result, "After 10 normal cycles")

    # ── Phase 2: Single sensor spoofing ──
    print("\n\n▶ PHASE 2: Single sensor spoofing attack (sensor 2)")
    for i in range(5):
        readings = (np.random.randn(6) * 0.02 + 0.5).tolist()
        readings[2] = 5.0  # sensor 2 spoofed to absurd value
        cmd = [0.3, 0.3, 0.3, 0.5, 0.5, 0.5]
        result = run_cycle(graph, readings, cmd, prev_state)
        prev_state = result
        if i == 0 or i == 4:
            print_cycle(result, f"Spoofing cycle {i+1}")

    # ── Phase 3: Hijacked agent sends dangerous command ──
    print("\n\n▶ PHASE 3: Hijacked agent — all pumps max, valves closed")
    readings = (np.random.randn(6) * 0.02 + 0.5).tolist()
    dangerous_cmd = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    result = run_cycle(graph, readings, dangerous_cmd, prev_state)
    prev_state = result
    print_cycle(result, "Hijacked agent command")
    print(f"  Agent wanted: {dangerous_cmd}")
    print(f"  CBF allowed:  {[round(x, 4) for x in result.get('safe_command', [])]}")

    # ── Phase 4: Coordinated multi-sensor attack ──
    print("\n\n▶ PHASE 4: Coordinated attack — 4 sensors spoofed simultaneously")
    for i in range(5):
        readings = (np.random.randn(6) * 0.02 + 0.5).tolist()
        readings[0] = 10.0
        readings[1] = -5.0
        readings[3] = 8.0
        readings[5] = -2.0
        cmd = [0.3, 0.3, 0.3, 0.5, 0.5, 0.5]
        result = run_cycle(graph, readings, cmd, prev_state)
        prev_state = result
        if i == 0 or i == 4:
            print_cycle(result, f"Coordinated attack cycle {i+1}")

    # ── Final state ──
    print("\n\n" + "=" * 65)
    print("  Final Defense State")
    print("=" * 65)
    print(f"  Defense level: L{result.get('defense_level', '?')}")
    print(f"  Health scores: {[round(h, 1) for h in result.get('health_scores', [])]}")
    print(f"  Agent health:  {result.get('agent_health', {})}")
    print(f"  Human notified: {result.get('human_notified', False)}")

    print("\n" + "=" * 65)
    print("  Key properties demonstrated:")
    print("  • Sentinel detects spoofing via physics residuals")
    print("  • Guardian blocks hijacked commands via CBF (QP)")
    print("  • Overseer escalates L1→L2→L3 based on threat severity")
    print("  • No agent trusts another — all decisions grounded in math")
    print("=" * 65)


if __name__ == "__main__":
    main()
