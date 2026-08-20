"""Synthetic physics-filtered residual data, for developing and smoke-testing the
anomaly transformer pipeline before SWaT/WADI access (iTrust approval) comes through.

This is NOT a substitute for real data in the paper's experiments -- it exists only so the
model, attacker, and training loop can be built, run, and debugged end to end today. Swap
in Member 1's actual observer output (physattest/observer/multi_domain_observer.py) as soon
as it's available; the model/trainer code does not need to change, only the data source.

Residuals are synthesized as noise that is correlated across sensors according to a random
coupling graph (mimicking Component 2's coupling graph), which is what lets the attacker's
coupling_penalty be exercised meaningfully even on fake data.
"""

import numpy as np
import torch


def make_coupling_graph(n_sensors: int, avg_degree: int = 3, seed: int = 0):
    """Random sparse adjacency + graph Laplacian, standing in for Component 2's coupling graph.

    Returns (adjacency, laplacian) as float32 numpy arrays, both (n_sensors, n_sensors).
    """
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((n_sensors, n_sensors), dtype=np.float32)
    n_edges = max(1, (n_sensors * avg_degree) // 2)
    for _ in range(n_edges):
        i, j = rng.integers(0, n_sensors, size=2)
        if i == j:
            continue
        weight = rng.uniform(0.3, 1.0)
        adjacency[i, j] = weight
        adjacency[j, i] = weight
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    return adjacency, laplacian.astype(np.float32)


def generate_normal_windows(
    n_windows: int,
    seq_len: int,
    n_sensors: int,
    adjacency: np.ndarray,
    noise_std: float = 0.05,
    ar_coef: float = 0.6,
    seed: int = 0,
) -> np.ndarray:
    """Correlated, mean-reverting (AR(1)-like) noise standing in for real "residual is ~0" behavior.

    At each step, independent Gaussian innovation is mixed a little with neighbours' previous
    values (via the coupling adjacency) before an AR(1) mean-reversion step, so that coupled
    sensors' residual noise is mildly correlated -- the way real physically-coupled sensors'
    tiny observer residuals would be, rather than perfectly independent white noise.
    """
    rng = np.random.default_rng(seed)
    # Row-normalize adjacency so the "neighbour mixing" step doesn't blow up magnitude.
    row_sum = adjacency.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    mix = adjacency / row_sum

    windows = np.zeros((n_windows, seq_len, n_sensors), dtype=np.float32)
    for w in range(n_windows):
        x = np.zeros((seq_len, n_sensors), dtype=np.float32)
        prev = np.zeros(n_sensors, dtype=np.float32)
        for t in range(seq_len):
            innovation = rng.normal(0, noise_std, size=n_sensors).astype(np.float32)
            neighbour_term = mix @ prev
            x[t] = ar_coef * prev + 0.3 * neighbour_term + innovation
            prev = x[t]
        windows[w] = x
    return windows


def inject_attack(
    window: np.ndarray,
    kind: str,
    magnitude: float,
    sensor_idx,
    onset_frac: float = 0.5,
) -> np.ndarray:
    """Overlay a labeled attack pattern onto a normal window, for supervised baselines
    and for held-out evaluation (separate from the *learned* adversarial attacker).

    kind: "step" | "ramp" | "oscillation"
    sensor_idx: int or list of ints -- which sensor channels carry the attack.
    onset_frac: fraction of the way through the window where the attack begins.
    """
    x = window.copy()
    seq_len = x.shape[0]
    onset = int(seq_len * onset_frac)
    idx = [sensor_idx] if isinstance(sensor_idx, (int, np.integer)) else list(sensor_idx)

    t = np.arange(seq_len - onset, dtype=np.float32)
    if kind == "step":
        pattern = np.full_like(t, magnitude)
    elif kind == "ramp":
        pattern = magnitude * (t / max(1, len(t) - 1))
    elif kind == "oscillation":
        pattern = magnitude * np.sin(2 * np.pi * t / max(1, len(t) / 3))
    else:
        raise ValueError(f"unknown attack kind: {kind}")

    for i in idx:
        x[onset:, i] += pattern
    return x


def to_tensor(windows: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(windows.astype(np.float32))


def make_labeled_eval_set(
    n_normal: int,
    n_attack: int,
    seq_len: int,
    n_sensors: int,
    adjacency: np.ndarray,
    seed: int = 1,
):
    """Held-out set with known attack labels, for computing F1/AUC (not for adversarial training)."""
    rng = np.random.default_rng(seed)
    normal = generate_normal_windows(n_normal, seq_len, n_sensors, adjacency, seed=seed)
    base_for_attacks = generate_normal_windows(n_attack, seq_len, n_sensors, adjacency, seed=seed + 1)

    kinds = ["step", "ramp", "oscillation"]
    attacks = np.zeros_like(base_for_attacks)
    for i in range(n_attack):
        kind = kinds[i % len(kinds)]
        n_hit = rng.integers(1, max(2, n_sensors // 5))
        sensors = rng.choice(n_sensors, size=n_hit, replace=False)
        magnitude = rng.uniform(0.15, 0.5)
        onset = rng.uniform(0.3, 0.7)
        attacks[i] = inject_attack(base_for_attacks[i], kind, magnitude, sensors, onset)

    windows = np.concatenate([normal, attacks], axis=0)
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_attack)]).astype(np.float32)
    perm = rng.permutation(len(windows))
    return windows[perm], labels[perm]
