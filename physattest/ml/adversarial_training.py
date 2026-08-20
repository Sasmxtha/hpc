"""Adversarial training loop for the anomaly transformer (Component 6).

This is a minimax game between two networks:
  - Detector D (ResidualAnomalyTransformer): scores a residual window as normal/attack.
  - Attacker G (AttackerGenerator): adds a small, physics-plausible perturbation to a
    normal window, trying to make D call it "normal".

D is trained to correctly flag both real known attacks and G's perturbed windows.
G is trained to fool D while staying smooth/graph-coupled (physics-plausible).

They alternate: D gets slightly better at catching the current G, G gets slightly better
at evading the current D, repeat. At convergence D should be hardened against the
*strongest* stealthy perturbation G could find within the physics-plausibility budget,
which is a much harder bar than fixed hand-designed attack patterns.

Two stability tricks, both standard in GAN training and important here because the
datasets are small and everything runs on CPU:
  1. Detector warm-start: train D for a few epochs on normal vs. simple hand-crafted
     attacks (data_synth.inject_attack) before G ever gets involved, so D isn't starting
     from a random classifier against an already-optimizing adversary.
  2. Epsilon curriculum: G's perturbation budget starts small and grows across training
     rounds, so D is never faced with a large, easy-to-catch-but-also-easy-to-overfit-to
     perturbation before it has learned the small, subtle ones.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from physattest.ml.anomaly_transformer import ResidualAnomalyTransformer
from physattest.ml.attacker import AttackerGenerator, physics_plausibility_penalty


@dataclass
class AdversarialTrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    lr_detector: float = 1e-3
    lr_attacker: float = 1e-3
    warmstart_epochs: int = 5
    epsilon_start: float = 0.05
    epsilon_end: float = 0.4
    lambda_smooth: float = 1.0
    lambda_coupling: float = 1.0
    lambda_magnitude: float = 0.1
    device: str = "cpu"


def _epsilon_for_epoch(epoch: int, total_adv_epochs: int, cfg: AdversarialTrainingConfig) -> float:
    if total_adv_epochs <= 1:
        return cfg.epsilon_end
    frac = epoch / (total_adv_epochs - 1)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def warmstart_detector(
    detector: ResidualAnomalyTransformer,
    normal_windows: torch.Tensor,
    labeled_windows: torch.Tensor,
    labeled_labels: torch.Tensor,
    cfg: AdversarialTrainingConfig,
) -> List[float]:
    """Supervised warm-start on normal vs. hand-crafted attacks, before adversarial rounds."""
    detector.train()
    optim = torch.optim.Adam(detector.parameters(), lr=cfg.lr_detector)
    bce = nn.BCEWithLogitsLoss()

    x = torch.cat([normal_windows, labeled_windows], dim=0)
    y = torch.cat(
        [torch.zeros(len(normal_windows)), labeled_labels.float()], dim=0
    )
    loader = DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=True)

    losses = []
    for _ in range(cfg.warmstart_epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optim.zero_grad()
            _, window_logit = detector(xb)
            loss = bce(window_logit, yb)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(xb)
        losses.append(epoch_loss / len(x))
    return losses


@dataclass
class AdversarialTrainingHistory:
    detector_loss: List[float] = field(default_factory=list)
    attacker_loss: List[float] = field(default_factory=list)
    epsilon: List[float] = field(default_factory=list)


def adversarial_train(
    detector: ResidualAnomalyTransformer,
    attacker: AttackerGenerator,
    normal_windows: torch.Tensor,
    cfg: AdversarialTrainingConfig,
    laplacian: Optional[torch.Tensor] = None,
) -> AdversarialTrainingHistory:
    """Alternating detector/attacker updates over `normal_windows`.

    normal_windows: (N, seq_len, n_sensors) windows known to be attack-free -- these are
    the base signal the attacker perturbs. Real labeled attacks (if any) should go through
    warmstart_detector beforehand; this loop is the pure adversarial hardening phase.
    """
    detector.to(cfg.device)
    attacker.to(cfg.device)
    if laplacian is not None:
        laplacian = laplacian.to(cfg.device)

    optim_d = torch.optim.Adam(detector.parameters(), lr=cfg.lr_detector)
    optim_g = torch.optim.Adam(attacker.parameters(), lr=cfg.lr_attacker)
    bce = nn.BCEWithLogitsLoss()

    loader = DataLoader(TensorDataset(normal_windows), batch_size=cfg.batch_size, shuffle=True)
    history = AdversarialTrainingHistory()

    for epoch in range(cfg.epochs):
        epsilon = _epsilon_for_epoch(epoch, cfg.epochs, cfg)
        d_loss_total, g_loss_total, n_batches = 0.0, 0.0, 0

        for (xb,) in loader:
            xb = xb.to(cfg.device)
            batch = xb.size(0)
            zeros = torch.zeros(batch, device=cfg.device)
            ones = torch.ones(batch, device=cfg.device)

            # --- Detector step ---
            detector.train()
            attacker.eval()
            optim_d.zero_grad()

            _, logit_normal = detector(xb)
            loss_normal = bce(logit_normal, zeros)

            with torch.no_grad():
                delta = attacker(xb, epsilon)
            x_adv = xb + delta
            _, logit_adv = detector(x_adv)
            loss_adv = bce(logit_adv, ones)

            loss_d = loss_normal + loss_adv
            loss_d.backward()
            optim_d.step()

            # --- Attacker step ---
            detector.eval()
            attacker.train()
            optim_g.zero_grad()

            delta = attacker(xb, epsilon)
            x_adv = xb + delta
            _, logit_adv = detector(x_adv)
            loss_fool = bce(logit_adv, zeros)  # wants detector to say "normal"
            loss_phys = physics_plausibility_penalty(
                delta,
                laplacian=laplacian,
                lambda_smooth=cfg.lambda_smooth,
                lambda_coupling=cfg.lambda_coupling,
                lambda_magnitude=cfg.lambda_magnitude,
            )
            loss_g = loss_fool + loss_phys
            loss_g.backward()
            optim_g.step()

            d_loss_total += loss_d.item()
            g_loss_total += loss_g.item()
            n_batches += 1

        history.detector_loss.append(d_loss_total / n_batches)
        history.attacker_loss.append(g_loss_total / n_batches)
        history.epsilon.append(epsilon)

    return history


if __name__ == "__main__":
    # Minimal end-to-end smoke test on synthetic data. Not a real experiment -- just
    # verifies the model, attacker, and training loop run without shape/dtype errors.
    from physattest.ml.data_synth import (
        make_coupling_graph,
        generate_normal_windows,
        make_labeled_eval_set,
        to_tensor,
    )

    n_sensors, seq_len = 12, 40
    adjacency, laplacian_np = make_coupling_graph(n_sensors, avg_degree=3, seed=0)
    laplacian = torch.from_numpy(laplacian_np)

    normal_np = generate_normal_windows(200, seq_len, n_sensors, adjacency, seed=0)
    normal = to_tensor(normal_np)

    eval_windows_np, eval_labels_np = make_labeled_eval_set(
        n_normal=40, n_attack=40, seq_len=seq_len, n_sensors=n_sensors, adjacency=adjacency
    )

    detector = ResidualAnomalyTransformer(n_sensors=n_sensors, d_model=16, nhead=2, num_layers=1)
    attacker = AttackerGenerator(n_sensors=n_sensors, latent_dim=4, hidden_dim=16)

    print(f"detector params: {detector.num_parameters()}")

    cfg = AdversarialTrainingConfig(epochs=2, batch_size=16, warmstart_epochs=1)

    warm_labels = torch.from_numpy(eval_labels_np[:20]).float()
    warm_windows = to_tensor(eval_windows_np[:20])
    warmstart_detector(detector, normal[:50], warm_windows, warm_labels, cfg)

    history = adversarial_train(detector, attacker, normal, cfg, laplacian=laplacian)
    print("detector loss per epoch:", history.detector_loss)
    print("attacker loss per epoch:", history.attacker_loss)
    print("smoke test passed")
