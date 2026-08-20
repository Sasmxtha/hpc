"""Synthetic RAW operational sensor data with a known ground-truth coupling structure, for
developing and validating gnn_discovery.py before real SWaT/WADI operational logs are
available.

This is deliberately different from physattest/ml/data_synth.py: that module synthesizes
near-zero physics-filtered RESIDUALS (for the anomaly transformer, Component 6). Coupling
discovery instead needs to watch actual sensor *dynamics* over time -- a residual has had
those dynamics subtracted out by design, so there would be nothing left to discover
structure from. Here we simulate raw, un-filtered multi-sensor readings driven by a real
coupled process, so the GNN has genuine cross-sensor dependencies to find.
"""

import numpy as np


def make_ground_truth_graph(n_sensors: int, avg_degree: int = 3, seed: int = 0) -> np.ndarray:
    """Random sparse symmetric adjacency, playing the role of the "true" physical coupling
    structure that gnn_discovery.py has to recover purely by watching x(t).
    """
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((n_sensors, n_sensors), dtype=np.float32)
    n_edges = max(1, (n_sensors * avg_degree) // 2)
    for _ in range(n_edges):
        i, j = rng.integers(0, n_sensors, size=2)
        if i == j:
            continue
        weight = rng.uniform(0.4, 1.0)
        adjacency[i, j] = weight
        adjacency[j, i] = weight
    return adjacency


def simulate_coupled_series(
    n_steps: int,
    n_sensors: int,
    adjacency: np.ndarray,
    coupling_strength: float = 0.15,
    noise_std: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """A simple coupled diffusion process: each sensor is pulled toward the weighted average
    of its coupled neighbours' previous values (like heat or pressure diffusing along real
    physical couplings), plus its own inertia and noise.

    x_i(t) = (1 - s) * x_i(t-1) + s * mean_{j in N(i)}(x_j(t-1)) + noise

    This is a stand-in for real plant dynamics -- the actual coupling discovery target once
    SWaT/WADI operational logs are available is whatever real cross-sensor dependencies exist
    there (e.g. pump vibration bleeding into a distant sensor), which this toy process cannot
    capture. It exists only so we can check the GNN recovers a graph close to a *known*
    ground truth before trusting it on data where ground truth is unknown.
    """
    rng = np.random.default_rng(seed)
    row_sum = adjacency.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    diffusion = adjacency / row_sum

    x = np.zeros((n_steps, n_sensors), dtype=np.float32)
    x[0] = rng.normal(0, 1, size=n_sensors)
    for t in range(1, n_steps):
        neighbour_avg = diffusion @ x[t - 1]
        x[t] = (
            (1 - coupling_strength) * x[t - 1]
            + coupling_strength * neighbour_avg
            + rng.normal(0, noise_std, size=n_sensors)
        )
    return x.astype(np.float32)


def simulate_shock_coupled_series(
    n_steps: int,
    n_sensors: int,
    adjacency: np.ndarray,
    coupling_strength: float = 0.7,
    ar_coef: float = 0.3,
    noise_std: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Alternative to simulate_coupled_series above, which turned out to have a real
    identifiability problem: it diffuses accumulated STATE through the graph every step
    (x_i(t) depends on neighbours' x(t-1), which itself already depended on ITS neighbours'
    x(t-2), and so on). Repeated diffusion like that is a classic consensus process (the same
    math behind DeGroot opinion dynamics / graph random walks) -- with strong enough coupling
    it converges the WHOLE network toward a shared global mode over many steps, regardless of
    topology, so a sensor three hops away becomes almost as predictive as a direct neighbour.
    Empirically this showed up as edge-recovery F1 getting WORSE, not better, as
    coupling_strength was pushed toward 1.0 -- exactly what consensus theory predicts, and
    the opposite of what "stronger coupling should be easier to detect" would suggest.

    This version instead shares a fresh, single-timestep SHOCK across neighbours -- closer to
    the spec's own example of "pump vibration bleeding into a distant sensor": a shared event
    that hits coupled sensors at the same time, not a quantity that keeps re-diffusing through
    the network. Each sensor keeps strong private persistence (ar_coef, independent of
    coupling) and receives a neighbour-weighted mix of everyone's CURRENT innovation, not of
    accumulated state -- so there is no repeated-averaging mechanism for correlation to spread
    beyond direct graph neighbours, and the coupling should remain locally identifiable even
    at high coupling_strength.

    x_i(t) = ar_coef * x_i(t-1) + coupling_strength * mean_{j in N(i)}(shock_j(t-1))
             + (1 - coupling_strength) * shock_i(t)

    Using the PREVIOUS step's shock (not the current one) for the shared component is
    deliberate, not cosmetic -- an earlier version of this function shared the CURRENT
    timestep's shock across neighbours, which is a purely same-timestep (contemporaneous)
    coupling. That has no cross-time structure for a look-back forecaster to exploit at all:
    a next-step predictor only ever sees x(t-1) and earlier, so if the only thing that makes
    neighbours correlated is something that happens simultaneously at t, nothing in the past
    window carries that information forward. Empirically this showed up as edge recovery
    getting WORSE again as coupling_strength increased -- once each sensor's own private
    noise term shrank, there was nothing left in its history that predicted its future via
    coupling, only via its own (weak) autocorrelation. Lagging the shared shock by one step
    makes it something a real neighbour's PAST reading actually reflects, which a forecaster
    can use, while still avoiding simulate_coupled_series's multi-hop consensus problem: raw
    shocks are exogenous i.i.d. draws, not accumulated diffused state, so there is no
    repeated-averaging mechanism for correlation to leak beyond direct graph neighbours.
    """
    rng = np.random.default_rng(seed)
    row_sum = adjacency.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    mix = adjacency / row_sum

    x = np.zeros((n_steps, n_sensors), dtype=np.float32)
    x[0] = rng.normal(0, 1, size=n_sensors)
    prev_shock = rng.normal(0, noise_std, size=n_sensors)
    for t in range(1, n_steps):
        shock_t = rng.normal(0, noise_std, size=n_sensors)
        shared_component = mix @ prev_shock
        x[t] = ar_coef * x[t - 1] + coupling_strength * shared_component + (1 - coupling_strength) * shock_t
        prev_shock = shock_t
    return x.astype(np.float32)


def make_windows(series: np.ndarray, seq_len: int):
    """series: (n_steps, n_sensors) -> (history windows (N, n_sensors, seq_len), next values (N, n_sensors))."""
    n_steps, n_sensors = series.shape
    n_windows = n_steps - seq_len
    windows = np.zeros((n_windows, n_sensors, seq_len), dtype=np.float32)
    next_values = np.zeros((n_windows, n_sensors), dtype=np.float32)
    for i in range(n_windows):
        windows[i] = series[i : i + seq_len].T
        next_values[i] = series[i + seq_len]
    return windows, next_values
