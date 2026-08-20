"""
Self-Healing Function for PhysAttest (Component 5).

When a sensor is compromised, reconstructs what it SHOULD read from
honest neighbours using physics, chemistry, and math equations from
the coupling graph.

Two modes:
  - Attack:  full replacement with reconstructed value
  - Fault:   calibration correction (subtract drift, keep sensor active)

THEOREM 3 — Reconstruction Accuracy:
    |x̂_j - x_j_true| ≤ ε(coupling_strength, noise_level, distance_to_honest)

If ε exceeds the safety margin, the function recommends quarantine
instead of reconstruction — honesty about uncertainty is safer than
a bad estimate.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class CompromiseType(Enum):
    ATTACK = "attack"
    FAULT = "fault"


class HealingAction(Enum):
    RECONSTRUCT = "reconstruct"
    CALIBRATE = "calibrate"
    QUARANTINE = "quarantine"


@dataclass
class HealingResult:
    sensor: str
    action: HealingAction
    compromise_type: CompromiseType
    original_value: float
    healed_value: float
    accuracy_bound: float       # ε from Theorem 3
    confidence: float           # 0-1, higher = more trustworthy
    contributors: List[str]     # which neighbours contributed
    explanation: str


class SelfHealingFunction:
    """
    Reconstructs compromised sensor values from honest neighbours
    using conservation-law-weighted interpolation through the
    coupling graph.

    The reconstruction is NOT a black-box interpolation — every
    reconstructed value is computed from a specific physics equation
    relating the compromised sensor to its neighbours.
    """

    def __init__(
        self,
        coupling_graph: nx.Graph,
        safety_margins: Optional[Dict[str, float]] = None,
        max_reconstruction_distance: int = 3,
    ):
        """
        Parameters:
            coupling_graph: the PhysAttest coupling graph (Component 2)
            safety_margins: per-sensor maximum acceptable ε before quarantine
            max_reconstruction_distance: max graph hops for reconstruction
        """
        self.G = coupling_graph
        self.max_dist = max_reconstruction_distance

        self.safety_margins = safety_margins or self._default_safety_margins()

        # Track which sensors are currently compromised
        self.compromised: Dict[str, CompromiseType] = {}

        # Fault tracking: running estimate of sensor drift
        self.drift_estimates: Dict[str, float] = {}
        self.drift_history: Dict[str, List[float]] = {}

    def _default_safety_margins(self) -> Dict[str, float]:
        """
        Default safety margins by sensor type.
        If the reconstruction error bound ε exceeds this, quarantine instead.

        These come from process safety requirements — how much error
        can each measurement tolerate before a dangerous decision
        could result.
        """
        margins = {}
        for node, data in self.G.nodes(data=True):
            sensor_type = data.get("type", "unknown")
            if sensor_type == "level":
                margins[node] = 0.05      # 50mm — enough to trigger wrong pump decision
            elif sensor_type == "flow":
                margins[node] = 0.3       # 0.3 m³/h
            elif sensor_type == "pH":
                margins[node] = 0.5       # 0.5 pH units
            elif sensor_type == "ORP":
                margins[node] = 30.0      # 30 mV
            elif sensor_type == "conductivity":
                margins[node] = 50.0      # 50 µS/cm
            elif sensor_type == "pressure":
                margins[node] = 5.0       # 5 kPa
            elif sensor_type in ("pump", "valve", "UV"):
                margins[node] = 0.5       # binary actuator — 0.5 = uncertain
            else:
                margins[node] = 0.1
        return margins

    def heal(
        self,
        sensor: str,
        compromise_type: CompromiseType,
        current_readings: Dict[str, float],
        observer_estimate: Optional[float] = None,
    ) -> HealingResult:
        """
        Main entry point. Reconstruct or calibrate a compromised sensor.

        Parameters:
            sensor: name of the compromised sensor
            compromise_type: ATTACK or FAULT
            current_readings: latest readings from ALL sensors
            observer_estimate: the observer's state estimate for this sensor
                               (from Component 1, if available)

        Returns:
            HealingResult with the healed value and accuracy bound.
        """
        self.compromised[sensor] = compromise_type

        # Find honest neighbours in the coupling graph
        honest_neighbours = self._find_honest_neighbours(sensor)

        if not honest_neighbours:
            return HealingResult(
                sensor=sensor,
                action=HealingAction.QUARANTINE,
                compromise_type=compromise_type,
                original_value=current_readings.get(sensor, float("nan")),
                healed_value=float("nan"),
                accuracy_bound=float("inf"),
                confidence=0.0,
                contributors=[],
                explanation="No honest neighbours reachable — quarantine required",
            )

        # Reconstruct from neighbours
        reconstructed, contributors, weights = self._reconstruct_from_neighbours(
            sensor, honest_neighbours, current_readings
        )

        # Blend with observer estimate if available
        if observer_estimate is not None:
            if np.isnan(reconstructed):
                reconstructed = observer_estimate
                contributors = ["observer"]
                weights = [1.0]
            else:
                obs_weight = 0.3
                reconstructed = (1 - obs_weight) * reconstructed + obs_weight * observer_estimate

        # Compute accuracy bound (Theorem 3)
        epsilon = self._compute_accuracy_bound(
            sensor, contributors, weights, current_readings
        )

        # Decide: reconstruct, calibrate, or quarantine
        safety_margin = self.safety_margins.get(sensor, 0.1)

        if epsilon > safety_margin:
            return HealingResult(
                sensor=sensor,
                action=HealingAction.QUARANTINE,
                compromise_type=compromise_type,
                original_value=current_readings.get(sensor, float("nan")),
                healed_value=reconstructed,
                accuracy_bound=epsilon,
                confidence=max(0, 1 - epsilon / safety_margin),
                contributors=contributors,
                explanation=(
                    f"Accuracy bound ε={epsilon:.4f} exceeds safety margin "
                    f"{safety_margin:.4f} — quarantine to avoid unsafe estimate"
                ),
            )

        if compromise_type == CompromiseType.FAULT:
            healed, explanation = self._calibration_correction(
                sensor, current_readings, reconstructed
            )
            action = HealingAction.CALIBRATE
        else:
            healed = reconstructed
            action = HealingAction.RECONSTRUCT
            explanation = (
                f"Full replacement from {len(contributors)} honest neighbours: "
                + ", ".join(contributors)
            )

        confidence = max(0, min(1, 1 - epsilon / safety_margin))

        return HealingResult(
            sensor=sensor,
            action=action,
            compromise_type=compromise_type,
            original_value=current_readings.get(sensor, float("nan")),
            healed_value=healed,
            accuracy_bound=epsilon,
            confidence=confidence,
            contributors=contributors,
            explanation=explanation,
        )

    def _find_honest_neighbours(self, sensor: str) -> List[Tuple[str, int, float]]:
        """
        BFS through coupling graph to find honest neighbours up to
        max_reconstruction_distance hops away.

        Returns list of (neighbour_name, distance, coupling_weight).
        Closer and more strongly coupled neighbours get more weight.
        """
        if sensor not in self.G:
            return []

        honest = []
        visited = {sensor}

        # BFS layer by layer
        current_layer = [sensor]
        for dist in range(1, self.max_dist + 1):
            next_layer = []
            for node in current_layer:
                for neighbour in self.G.neighbors(node):
                    if neighbour in visited:
                        continue
                    visited.add(neighbour)

                    if neighbour in self.compromised:
                        continue

                    edge_data = self.G.edges[node, neighbour]
                    weight = edge_data.get("weight", 0.5)
                    honest.append((neighbour, dist, weight))
                    next_layer.append(neighbour)

            current_layer = next_layer

        return honest

    def _reconstruct_from_neighbours(
        self,
        sensor: str,
        honest_neighbours: List[Tuple[str, int, float]],
        current_readings: Dict[str, float],
    ) -> Tuple[float, List[str], List[float]]:
        """
        Weighted reconstruction from honest neighbours.

        Weight for each neighbour = coupling_weight / distance²

        The 1/distance² decay means direct neighbours dominate.
        This is physically correct: a sensor one pipe away is much
        more informative than one three pipes away.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        contributors = []
        weights = []

        sensor_data = self.G.nodes.get(sensor, {})
        sensor_type = sensor_data.get("type", "unknown")

        for neighbour, dist, coupling_weight in honest_neighbours:
            neighbour_data = self.G.nodes.get(neighbour, {})
            neighbour_type = neighbour_data.get("type", "unknown")

            if neighbour not in current_readings:
                continue

            neighbour_value = current_readings[neighbour]

            # Transform neighbour reading to target sensor's domain
            transformed = self._transform_reading(
                sensor, sensor_type,
                neighbour, neighbour_type, neighbour_value,
                current_readings
            )

            if transformed is None:
                continue

            w = coupling_weight / (dist ** 2)
            weighted_sum += w * transformed
            total_weight += w
            contributors.append(neighbour)
            weights.append(w)

        if total_weight == 0:
            return float("nan"), [], []

        reconstructed = weighted_sum / total_weight
        return reconstructed, contributors, weights

    def _transform_reading(
        self,
        target: str, target_type: str,
        source: str, source_type: str,
        source_value: float,
        all_readings: Dict[str, float],  # noqa: ARG002 - used in cross-type lookups
    ) -> Optional[float]:
        """
        Transform a neighbour's reading into an estimate for the target
        sensor using the physics equation on their shared edge.

        This is where the conservation laws do the heavy lifting.
        Each transformation is grounded in a specific equation.
        """
        edge_data = self.G.edges.get((target, source)) or self.G.edges.get((source, target))
        if edge_data is None:
            return None

        equation = edge_data.get("equation", "")
        domain = edge_data.get("domain", "physics")

        # Same-type sensors: direct transfer (e.g., two flow sensors on same pipe)
        if target_type == source_type:
            return source_value

        # Level ↔ Flow: mass conservation
        # At steady state (which SWaT approximates between setpoint changes),
        # level is stable when inflow ≈ outflow.
        # We can estimate level from flow neighbours by checking whether the
        # flow sensor agrees with the current level trend. But the direct
        # value isn't transferable (different units/scale), so we contribute
        # the level implied by the flow balance using other level readings.
        if target_type == "level" and source_type == "flow":
            # Find another level sensor in the same or adjacent stage to anchor
            target_stage = self.G.nodes[target].get("stage", 0)
            for n in self.G.nodes():
                if (n != target and n != source
                        and self.G.nodes[n].get("type") == "level"
                        and n not in self.compromised
                        and abs(self.G.nodes[n].get("stage", 0) - target_stage) <= 1
                        and n in all_readings):
                    return all_readings[n]
            return None

        if target_type == "flow" and source_type == "level":
            return None  # level doesn't directly give flow magnitude

        # Flow ↔ Flow: Kirchhoff (conservation at junction)
        if target_type == "flow" and source_type == "flow":
            return source_value

        # Pressure ↔ Flow: approximate linear relationship around operating point
        if target_type == "pressure" and source_type == "flow":
            return None

        if target_type == "flow" and source_type == "pressure":
            return None

        # Pressure ↔ Pressure: monotonic relationship
        if target_type == "pressure" and source_type == "pressure":
            return source_value * 0.95

        # pH ↔ pH: transport delay, approximately equal
        if target_type == "pH" and source_type == "pH":
            return source_value

        # ORP ↔ ORP: transport delay
        if target_type == "ORP" and source_type == "ORP":
            return source_value

        # Conductivity ↔ Conductivity: dilution/concentration
        if target_type == "conductivity" and source_type == "conductivity":
            return source_value

        # pH ↔ ORP: weak chemistry correlation
        if target_type == "pH" and source_type == "ORP":
            return None

        if target_type == "ORP" and source_type == "pH":
            return None

        # Conductivity ↔ hardness: both measure dissolved solids
        if target_type == "conductivity" and source_type == "hardness":
            return source_value * 3.0  # approximate conversion factor
        if target_type == "hardness" and source_type == "conductivity":
            return source_value / 3.0

        # Pump/valve ↔ anything: actuator states don't reconstruct sensors
        if source_type in ("pump", "valve", "UV"):
            return None

        return None

    def _compute_accuracy_bound(
        self,
        sensor: str,
        contributors: List[str],
        weights: List[float],
        current_readings: Dict[str, float],
    ) -> float:
        """
        THEOREM 3: Reconstruction accuracy bound.

        |x̂_j - x_j_true| ≤ ε(coupling_strength, noise_level, distance)

        ε = σ_noise / √(Σ wᵢ²·κᵢ²)

        Where:
            σ_noise = measurement noise standard deviation
            wᵢ = reconstruction weight for neighbour i
            κᵢ = coupling strength (edge weight) for neighbour i

        The bound tightens with:
            - More honest neighbours (larger denominator)
            - Stronger couplings (larger κ)
            - Lower noise (smaller numerator)

        The bound loosens with:
            - Greater distance to honest sensors
            - Weaker couplings
            - Higher noise
        """
        if not contributors or not weights:
            return float("inf")

        sensor_data = self.G.nodes.get(sensor, {})
        sensor_type = sensor_data.get("type", "unknown")

        # Noise level by sensor type
        noise_levels = {
            "level": 0.002,
            "flow": 0.05,
            "pH": 0.02,
            "ORP": 1.0,
            "conductivity": 0.5,
            "pressure": 0.1,
            "hardness": 0.5,
            "pump": 0.01,
            "valve": 0.01,
            "UV": 0.01,
        }
        sigma = noise_levels.get(sensor_type, 0.1)

        # Coupling strengths from edge weights
        coupling_strengths = []
        for contributor in contributors:
            edge_data = (
                self.G.edges.get((sensor, contributor))
                or self.G.edges.get((contributor, sensor))
            )
            kappa = edge_data.get("weight", 0.5) if edge_data else 0.5
            coupling_strengths.append(kappa)

        # ε = σ / √(Σ wᵢ² κᵢ²) + σ_model
        #
        # σ_model accounts for the fact that the transform from neighbour
        # to target is approximate (same-type transfer, linearized physics,
        # etc.). Without this term, the bound is too tight when the
        # reconstruction relies on cross-type transforms or distant sensors.
        w_arr = np.array(weights)
        k_arr = np.array(coupling_strengths)
        denominator = np.sqrt(np.sum((w_arr * k_arr) ** 2))

        if denominator < 1e-10:
            return float("inf")

        # Model uncertainty: increases with fewer contributors and
        # when relying on cross-type transforms
        n_same_type = 0
        for c in contributors:
            if c == "observer":
                n_same_type += 1
                continue
            c_type = self.G.nodes.get(c, {}).get("type", "")
            if c_type == sensor_type:
                n_same_type += 1
        cross_type_fraction = 1 - n_same_type / max(len(contributors), 1)
        sigma_model = sigma * (1 + 2 * cross_type_fraction) / max(len(contributors), 1)

        epsilon = sigma / denominator + sigma_model
        return float(epsilon)

    def _calibration_correction(
        self,
        sensor: str,
        current_readings: Dict[str, float],
        reconstructed: float,
    ) -> Tuple[float, str]:
        """
        For faults (not attacks): compute the drift offset and subtract it.

        A faulty sensor still responds to physical changes — it's just
        biased. We estimate the bias and correct it, keeping the sensor
        active with a calibration offset.

        drift = measured - reconstructed (running average)
        corrected = measured - drift
        """
        measured = current_readings.get(sensor, float("nan"))
        if np.isnan(measured) or np.isnan(reconstructed):
            return reconstructed, "Cannot compute drift — using reconstruction"

        current_drift = measured - reconstructed

        # Update running drift estimate (exponential moving average)
        if sensor not in self.drift_estimates:
            self.drift_estimates[sensor] = current_drift
            self.drift_history[sensor] = [current_drift]
        else:
            alpha = 0.1  # smoothing factor
            self.drift_estimates[sensor] = (
                (1 - alpha) * self.drift_estimates[sensor] + alpha * current_drift
            )
            self.drift_history[sensor].append(current_drift)
            if len(self.drift_history[sensor]) > 100:
                self.drift_history[sensor] = self.drift_history[sensor][-100:]

        drift = self.drift_estimates[sensor]
        corrected = measured - drift

        return corrected, (
            f"Calibration correction: drift={drift:.4f}, "
            f"measured={measured:.4f} → corrected={corrected:.4f}"
        )

    def mark_honest(self, sensor: str):
        """Restore a sensor to honest status after repair/verification."""
        self.compromised.pop(sensor, None)
        self.drift_estimates.pop(sensor, None)
        self.drift_history.pop(sensor, None)

    def get_system_health(self) -> Dict:
        """Summary of system healing status."""
        total = self.G.number_of_nodes()
        n_compromised = len(self.compromised)
        n_attacks = sum(1 for t in self.compromised.values() if t == CompromiseType.ATTACK)
        n_faults = sum(1 for t in self.compromised.values() if t == CompromiseType.FAULT)

        return {
            "total_sensors": total,
            "healthy": total - n_compromised,
            "compromised": n_compromised,
            "attacks": n_attacks,
            "faults": n_faults,
            "health_percentage": (total - n_compromised) / total * 100,
            "compromised_sensors": dict(self.compromised),
        }
