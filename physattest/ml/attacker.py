"""Attacker (generator) network and differentiable physics-plausibility penalties.

The attacker's job is NOT to model an arbitrary adversary. Component 1 (the multi-domain
observer) already rejects anything that grossly violates conservation laws, and the CBF
(Component 3) bounds the damage of anything that gets through. The residual-level attacker
modeled here represents the *residual* an adaptive attacker leaves behind after doing its
best to stay physics-consistent -- i.e. the near-miss, slow-drift style spoofing that is
hard to catch precisely because it is small and smooth, not because it is undetectable in
principle.

That is why the generator is penalized, not just budget-capped: an unconstrained generator
would learn to produce whatever incoherent pattern best fools the current detector, which
is not a realistic attacker (Component 1 would already flag incoherent, high-frequency,
per-sensor-independent deviations). The penalties bias it toward patterns a real physics-
aware attacker could plausibly produce, which is exactly the harder, more realistic case we
want the transformer to learn to catch.
"""

import torch
import torch.nn as nn


class AttackerGenerator(nn.Module):
    """Generates a bounded perturbation delta to add to a normal residual window.

    Conditioned on the normal window itself (so the perturbation can be shaped by local
    noise structure) plus a small latent noise vector (so it isn't deterministic -- the
    same normal window can be attacked in different ways across training steps). A small
    1D-conv stack is enough; there's no need for anything transformer-sized here since the
    generator's job is local, smooth signal shaping, not long-range reasoning.
    """

    def __init__(
        self,
        n_sensors: int,
        latent_dim: int = 8,
        hidden_dim: int = 32,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.latent_dim = latent_dim
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(n_sensors + latent_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, n_sensors, kernel_size, padding=padding),
        )

    def forward(self, x_normal: torch.Tensor, epsilon: float) -> torch.Tensor:
        """x_normal: (batch, seq_len, n_sensors). Returns delta of the same shape.

        epsilon is the perturbation budget (max abs magnitude per entry, in residual
        units). It is passed in per-call rather than fixed at construction time so the
        training loop can run an epsilon curriculum -- start tiny, grow it -- without
        rebuilding the model.
        """
        batch, seq_len, _ = x_normal.shape
        z = torch.randn(batch, seq_len, self.latent_dim, device=x_normal.device)
        inp = torch.cat([x_normal, z], dim=-1)  # (batch, seq_len, n_sensors+latent_dim)
        inp = inp.transpose(1, 2)  # (batch, C, seq_len) for Conv1d
        raw = self.net(inp).transpose(1, 2)  # (batch, seq_len, n_sensors)
        delta = torch.tanh(raw) * epsilon
        return delta


def smoothness_penalty(delta: torch.Tensor) -> torch.Tensor:
    """Total-variation penalty along the time axis.

    Real conservation-law-consistent spoofing has to change state gradually -- mass,
    energy and pressure have inertia, they cannot jump. A perturbation with large
    timestep-to-timestep jumps is exactly the kind Component 1's observer would already
    flag on its own, so we penalize it here as a cheap proxy for "physically abrupt".
    """
    diff = delta[:, 1:, :] - delta[:, :-1, :]
    return diff.pow(2).mean()


def coupling_penalty(delta: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    """Graph-Laplacian smoothness penalty across sensors, using Component 2's coupling graph.

    delta: (batch, seq_len, n_sensors)
    laplacian: (n_sensors, n_sensors) graph Laplacian L = D - A of the coupling graph.

    For a vector v, v^T L v = sum over edges (i,j) of A_ij * (v_i - v_j)^2 -- it is small
    when v agrees across strongly-coupled sensor pairs and large when coupled sensors move
    independently. A real physical disturbance propagates through couplings (heat diffuses
    to the adjacent room, pressure changes propagate along the pipe), so a genuinely
    physics-consistent perturbation should be graph-smooth. Penalizing this discourages the
    generator from cheating with per-sensor-independent noise that no real coupled system
    could produce -- again, the kind of thing Component 1 would already catch, so training
    against it would only teach the transformer to catch an unrealistic attacker.
    """
    # v^T L v per timestep, then average over timesteps and batch.
    lap_term = torch.einsum("btn,nm,btm->bt", delta, laplacian, delta)
    return lap_term.mean()


def magnitude_penalty(delta: torch.Tensor) -> torch.Tensor:
    """Mean squared magnitude of the perturbation.

    The tanh-and-epsilon squashing in AttackerGenerator already hard-bounds delta, so this
    is a soft nudge (not a strict constraint) toward smaller, stealthier perturbations
    rather than maxing out the budget on every entry -- true slow-drift attacks tend to use
    only a fraction of what they could get away with.
    """
    return delta.pow(2).mean()


def physics_plausibility_penalty(
    delta: torch.Tensor,
    laplacian: torch.Tensor = None,
    lambda_smooth: float = 1.0,
    lambda_coupling: float = 1.0,
    lambda_magnitude: float = 0.1,
) -> torch.Tensor:
    """Weighted sum of the differentiable physics-plausibility penalties above.

    laplacian is optional: pass it once Component 2's coupling graph is available for the
    plant being trained on. Without it, the generator is still constrained by smoothness
    and magnitude, just not by cross-sensor coupling.
    """
    penalty = lambda_smooth * smoothness_penalty(delta) + lambda_magnitude * magnitude_penalty(delta)
    if laplacian is not None:
        penalty = penalty + lambda_coupling * coupling_penalty(delta, laplacian)
    return penalty
