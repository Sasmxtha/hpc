"""Evaluation for the anomaly transformer: standard detection metrics plus, critically,
robustness against the learned physics-aware attacker -- not just fixed attack patterns.

Two different attack sources are evaluated on purpose:
  1. Held-out hand-crafted attacks (data_synth.inject_attack / real SWaT-WADI attacks
     once available) -- the "known attack signature" case. Standard F1/AUC/detection
     delay apply here.
  2. The trained AttackerGenerator itself, swept across perturbation budgets -- the
     "adaptive attacker" case. This measures whether adversarial training actually bought
     robustness, not just accuracy on attacks the model already knows about. A model that
     scores well on (1) but has a high evasion rate at small epsilon in (2) would be
     exactly the kind of defense that "collapses under adaptive attackers" that Section 1
     of the paper is arguing against -- so this comparison is the whole point.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from physattest.ml.anomaly_transformer import ResidualAnomalyTransformer
from physattest.ml.attacker import AttackerGenerator


@dataclass
class DetectionMetrics:
    auc: float
    f1: float
    threshold: float
    mean_detection_delay: float  # timesteps from attack onset to first flagged timestep


@torch.no_grad()
def evaluate_detection(
    detector: ResidualAnomalyTransformer,
    windows: torch.Tensor,
    labels: np.ndarray,
    onset_fracs: np.ndarray = None,
    threshold: float = 0.5,
    device: str = "cpu",
) -> DetectionMetrics:
    """windows: (N, seq_len, n_sensors), labels: (N,) in {0,1}.

    onset_fracs (optional): fraction into the window where each attack begins, used to
    compute detection delay for the positive (label==1) windows only. If not provided,
    detection delay is skipped (reported as nan).
    """
    detector.eval()
    detector.to(device)
    windows = windows.to(device)

    timestep_logits, window_logit = detector(windows)
    scores = torch.sigmoid(window_logit).cpu().numpy()
    preds = (scores >= threshold).astype(np.float32)

    auc = roc_auc_score(labels, scores) if len(set(labels.tolist())) > 1 else float("nan")
    f1 = f1_score(labels, preds)

    delay = float("nan")
    if onset_fracs is not None:
        seq_len = windows.shape[1]
        timestep_scores = torch.sigmoid(timestep_logits).cpu().numpy()
        delays = []
        for i in range(len(labels)):
            if labels[i] != 1:
                continue
            onset = int(seq_len * onset_fracs[i])
            flagged = np.where(timestep_scores[i, onset:] >= threshold)[0]
            if len(flagged) > 0:
                delays.append(flagged[0])
            else:
                delays.append(seq_len - onset)  # never flagged within window
        if delays:
            delay = float(np.mean(delays))

    return DetectionMetrics(auc=auc, f1=f1, threshold=threshold, mean_detection_delay=delay)


@torch.no_grad()
def evasion_rate_vs_budget(
    detector: ResidualAnomalyTransformer,
    attacker: AttackerGenerator,
    normal_windows: torch.Tensor,
    epsilons: List[float],
    threshold: float = 0.5,
    n_samples: int = 5,
    device: str = "cpu",
) -> Dict[float, float]:
    """For each epsilon budget, generate perturbed windows and measure the fraction the
    detector still calls "normal" (score < threshold) -- the evasion rate.

    n_samples: how many independent perturbations to draw per window per epsilon (the
    attacker is stochastic via its latent noise), to reduce variance in the estimate.

    A robust, adversarially-trained detector should show evasion rate staying low even
    as epsilon grows, up to the budget it was trained against. A steep rise at small
    epsilon is the signature of the kind of collapse-under-adaptive-attack failure mode
    this whole component exists to avoid.
    """
    detector.eval()
    attacker.eval()
    detector.to(device)
    attacker.to(device)
    normal_windows = normal_windows.to(device)

    results = {}
    for eps in epsilons:
        evasions, total = 0, 0
        for _ in range(n_samples):
            delta = attacker(normal_windows, eps)
            x_adv = normal_windows + delta
            _, window_logit = detector(x_adv)
            scores = torch.sigmoid(window_logit)
            evasions += (scores < threshold).sum().item()
            total += scores.numel()
        results[eps] = evasions / total
    return results


def compare_adversarial_vs_baseline(
    metrics_baseline: DetectionMetrics,
    metrics_adversarial: DetectionMetrics,
    evasion_baseline: Dict[float, float],
    evasion_adversarial: Dict[float, float],
) -> str:
    """Human-readable ablation summary: adversarially-trained detector vs. a detector
    trained the same way but WITHOUT the attacker (plain supervised on normal + hand-
    crafted attacks only). This is the table that demonstrates adversarial training was
    worth doing, for the paper's Section 5/7.
    """
    lines = [
        "                         baseline      adversarial",
        f"AUC (known attacks)     {metrics_baseline.auc:.3f}         {metrics_adversarial.auc:.3f}",
        f"F1  (known attacks)     {metrics_baseline.f1:.3f}         {metrics_adversarial.f1:.3f}",
        f"detection delay (steps) {metrics_baseline.mean_detection_delay:.2f}          {metrics_adversarial.mean_detection_delay:.2f}",
        "",
        "evasion rate vs. perturbation budget (lower = more robust):",
        "epsilon     baseline      adversarial",
    ]
    for eps in sorted(evasion_baseline):
        lines.append(
            f"{eps:.3f}       {evasion_baseline[eps]:.3f}         {evasion_adversarial.get(eps, float('nan')):.3f}"
        )
    return "\n".join(lines)
