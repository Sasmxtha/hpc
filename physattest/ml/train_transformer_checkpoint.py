"""Trains and saves the Component 6 checkpoint actually loaded by agents/sentinel.py.

Earlier in this project, the transformer was validated only on toy AR-noise synthetic data
(physattest/ml/data_synth.py) -- fine for proving the adversarial-training mechanism works,
but not something that should be wired into a live agent: an undertrained/arbitrary-scale
model injects near-random signal into detection rather than helping it (this is exactly what
went wrong the first time transfer learning was attempted in this project, before that was
fixed). This script instead trains on residuals produced by the ACTUAL, now SINDy+PINN-
integrated MultiDomainObserver running over a physically-consistent synthetic operational
trajectory (matching physics_equations.py's real TANK_PARAMS/PUMP_FLOW_RATES), with attacks
injected into the raw sensor reading BEFORE the observer sees it -- so the residual pattern
the transformer learns to recognize is what a real spoofing attack actually looks like after
passing through the real Layer 1/2/3 prediction, not an arbitrary synthetic shape.

Every window (train AND at inference time in sentinel.py) is normalized by its own early-
portion baseline std before being shown to the model. This makes the transformer learn
SHAPE, not absolute scale -- necessary because sentinel.py's own toy 6-sensor demo
abstraction and this script's real observer-residual training data are not on the same
numeric scale, and forcing scale-invariance is the honest way to bridge that gap rather than
pretending they match.
"""

import numpy as np
import torch

from physattest.ml.adversarial_training import AdversarialTrainingConfig, warmstart_detector
from physattest.ml.anomaly_transformer import ResidualAnomalyTransformer
from physattest.ml.evaluate import evaluate_detection
from physattest.observer.multi_domain_observer import (
    INPUT_MV101, INPUT_P101, MultiDomainObserver, N_INPUTS, N_STATES, STATE_H1,
)
from physattest.observer.physics_equations import PUMP_FLOW_RATES, TANK_PARAMS

SEQ_LEN = 20
CHECKPOINT_PATH = "physattest/ml/checkpoints/anomaly_transformer_h1.pt"


def normalize_window(window: np.ndarray) -> np.ndarray:
    """Z-score a window by its own early-portion baseline -- the SAME normalization used at
    both training and inference time (agents/sentinel.py), so the model only ever sees
    scale-invariant shape, never absolute magnitude.
    """
    edge = max(1, len(window) // 5)
    baseline_mean = float(np.mean(window[:edge]))
    baseline_std = float(np.std(window[:edge])) + 1e-6
    return (window - baseline_mean) / baseline_std


def simulate_h1_residual_series(
    n_steps: int, attack_kind: str = None, attack_magnitude: float = 0.0,
    attack_onset: int = None, attack_duration: int = None, seed: int = 0,
) -> np.ndarray:
    """Runs a physically-consistent H1 operational trajectory through the real, integrated
    MultiDomainObserver (Layer 1, with SINDy Layer 2 refinement applied partway through),
    optionally with an attack injected into the RAW reading the observer sees. Returns the
    observer's own `combined` residual for STATE_H1 -- what a real Sentinel would see.
    """
    rng = np.random.default_rng(seed)
    area_1 = TANK_PARAMS["T1"]["area"]
    q_p101 = PUMP_FLOW_RATES["P101"] / 3600

    y_traj = np.zeros((n_steps, N_STATES))
    u_traj = np.zeros((n_steps, N_INPUTS))
    h1 = 0.5
    mv101, p101 = 1.0, 0.0
    for t in range(n_steps):
        if rng.random() < 0.03:
            mv101 = 1.0 - mv101
        if rng.random() < 0.03:
            p101 = 1.0 - p101
        dh1 = q_p101 * (mv101 - p101) / area_1
        h1 = float(np.clip(h1 + dh1, 0.0, 1.0))
        y_traj[t, STATE_H1] = h1
        y_traj[t, 1:] = 0.5  # other states held constant -- not this script's concern
        u_traj[t, INPUT_MV101] = mv101
        u_traj[t, INPUT_P101] = p101

    if attack_kind is not None:
        onset = attack_onset if attack_onset is not None else n_steps // 2
        duration = attack_duration if attack_duration is not None else n_steps - onset
        end = min(n_steps, onset + duration)
        span = np.arange(end - onset)
        if attack_kind == "step":
            pattern = np.full_like(span, attack_magnitude, dtype=np.float64)
        elif attack_kind == "ramp":
            pattern = attack_magnitude * (span / max(1, len(span) - 1))
        elif attack_kind == "slow_drift":
            # Deliberately gentler than "ramp": the marquee Component 6 claim is catching
            # drift too subtle for a per-step z-score threshold to fire on at all.
            pattern = attack_magnitude * (span / max(1, len(span) - 1)) * 0.3
        elif attack_kind == "oscillation":
            pattern = attack_magnitude * np.sin(2 * np.pi * span / max(1, len(span) / 3))
        else:
            raise ValueError(attack_kind)
        y_traj[onset:end, STATE_H1] += pattern

    observer = MultiDomainObserver(dt=1.0)
    observer.initialize(y_traj[0])
    residual_series = np.zeros(n_steps)
    for t in range(1, n_steps):
        # Refine with SINDy once mid-run, matching how it would actually be used online
        # (periodic refinement, not every step) -- see refine_with_sindy's own docstring.
        if t == n_steps // 2:
            observer.refine_with_sindy(min_samples=200, threshold=1e-5, blend_factor=0.3)
        result = observer.step(y_traj[t], u_traj[t])
        residual_series[t] = result["combined"][STATE_H1]
    return residual_series


def make_windows(series: np.ndarray, seq_len: int) -> np.ndarray:
    n = len(series) - seq_len
    return np.stack([series[i : i + seq_len] for i in range(n)])


def build_dataset(n_normal_series: int, n_attack_series: int, seed_offset: int = 0):
    normal_windows = []
    for i in range(n_normal_series):
        series = simulate_h1_residual_series(400, seed=seed_offset + i)
        windows = make_windows(series, SEQ_LEN)
        normal_windows.append(windows[::5])  # subsample -- adjacent windows are near-duplicates
    normal_windows = np.concatenate(normal_windows)

    attack_windows, attack_labels = [], []
    kinds = ["step", "ramp", "slow_drift", "oscillation"]
    rng = np.random.default_rng(seed_offset + 1000)
    for i in range(n_attack_series):
        kind = kinds[i % len(kinds)]
        magnitude = rng.uniform(0.15, 0.5)
        series = simulate_h1_residual_series(
            400, attack_kind=kind, attack_magnitude=magnitude,
            attack_onset=150, attack_duration=200, seed=seed_offset + 2000 + i,
        )
        windows = make_windows(series, SEQ_LEN)
        # keep only windows that overlap the attack span [150, 350)
        attacked_idx = [j for j in range(len(windows)) if 150 < j + SEQ_LEN and j < 350]
        chosen = attacked_idx[::4]
        attack_windows.append(windows[chosen])
        attack_labels.extend([kind] * len(chosen))

    attack_windows = np.concatenate(attack_windows)

    normal_norm = np.stack([normalize_window(w) for w in normal_windows])
    attack_norm = np.stack([normalize_window(w) for w in attack_windows])

    return normal_norm, attack_norm, attack_labels


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    print("generating training data from the real, integrated observer...")
    train_normal, train_attack, train_labels = build_dataset(n_normal_series=12, n_attack_series=16, seed_offset=0)
    test_normal, test_attack, test_labels = build_dataset(n_normal_series=6, n_attack_series=12, seed_offset=500)

    print(f"train: {len(train_normal)} normal windows, {len(train_attack)} attack windows ({set(train_labels)})")
    print(f"test:  {len(test_normal)} normal windows, {len(test_attack)} attack windows")

    model = ResidualAnomalyTransformer(n_sensors=1, d_model=16, nhead=2, num_layers=1)

    train_normal_t = torch.from_numpy(train_normal.astype(np.float32)).unsqueeze(-1)
    train_attack_t = torch.from_numpy(train_attack.astype(np.float32)).unsqueeze(-1)
    labeled_windows = torch.cat([train_normal_t, train_attack_t], dim=0)
    labeled_labels = torch.cat([torch.zeros(len(train_normal_t)), torch.ones(len(train_attack_t))])

    cfg = AdversarialTrainingConfig(epochs=1, batch_size=16, warmstart_epochs=60, lr_detector=2e-3)
    print("training...")
    losses = warmstart_detector(model, train_normal_t, labeled_windows, labeled_labels, cfg)
    print("final warmstart loss:", losses[-1])

    test_windows = np.concatenate([test_normal, test_attack])
    test_labels_arr = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_attack))])
    test_windows_t = torch.from_numpy(test_windows.astype(np.float32)).unsqueeze(-1)

    metrics = evaluate_detection(model, test_windows_t, test_labels_arr)
    print(f"\nheld-out test: AUC={metrics.auc:.3f} F1={metrics.f1:.3f}")

    assert metrics.auc > 0.7, f"transformer should discriminate meaningfully better than chance, got AUC={metrics.auc:.3f}"

    torch.save({"state_dict": model.state_dict(), "config": model.config}, CHECKPOINT_PATH)
    print(f"\nsaved checkpoint to {CHECKPOINT_PATH}")
    print("train_transformer_checkpoint.py passed")
