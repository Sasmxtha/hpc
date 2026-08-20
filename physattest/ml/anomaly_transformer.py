"""Lightweight transformer that scores physics-filtered residual windows for anomalies.

Input is NOT raw sensor data. It is the residual r(t) = y(t) - C*x_hat(t) produced by
Member 1's multi-domain observer (Component 1). Under normal operation this residual is
near-zero noise; under attack it carries a faint, temporally structured pattern. Because
the physics layer already removed the bulk of the signal's variation, this model can stay
small and CPU-trainable.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal position encoding, added to the input embedding.

    Residual windows are short time series where order matters (a slow drift looks very
    different from a step), so the model needs some notion of timestep position. A learned
    embedding would also work, but sinusoidal costs zero parameters, which matters when the
    model is meant to stay tiny and CPU-trainable.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class ResidualAnomalyTransformer(nn.Module):
    """Encoder-only transformer over a window of multi-sensor residuals.

    Produces two things:
      - per-timestep anomaly logits: lets you localize *when* inside the window the
        pattern turned suspicious (useful for the forensics engine and for measuring
        detection delay).
      - a window-level logit: a single pooled score for "is this window an attack",
        which is what the three-way classifier consumes.

    Kept deliberately small (default ~50-150k params) since the input is already an
    almost-empty residual, not raw sensor data, and the target hardware has no GPU.
    """

    def __init__(
        self,
        n_sensors: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        # Stored so transfer_learning.py can reconstruct an architecturally-identical model
        # for a new n_sensors without needing to introspect nhead/num_layers back out of the
        # encoder's internal submodules.
        self.config = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "max_len": max_len,
        }
        self.input_proj = nn.Linear(n_sensors, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.timestep_head = nn.Linear(d_model, 1)

        # Learned attention pooling instead of a plain mean: lets the model weight the
        # few suspicious timesteps in a window more heavily than the many normal ones,
        # which matters for slow-drift attacks that only become clear near the end of
        # the window.
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.window_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        """x: (batch, seq_len, n_sensors) residual window.

        Returns (timestep_logits, window_logit):
          timestep_logits: (batch, seq_len) raw logits, apply sigmoid for [0,1] scores.
          window_logit: (batch,) raw logit for the whole window.
        """
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)  # (batch, seq_len, d_model)

        timestep_logits = self.timestep_head(h).squeeze(-1)  # (batch, seq_len)

        query = self.pool_query.expand(h.size(0), -1, -1)  # (batch, 1, d_model)
        pooled, _ = self.pool_attn(query, h, h)  # (batch, 1, d_model)
        window_logit = self.window_head(pooled.squeeze(1)).squeeze(-1)  # (batch,)

        return timestep_logits, window_logit

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
