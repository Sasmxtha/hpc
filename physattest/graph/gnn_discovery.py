"""GNN-based coupling discovery (the GNN piece inside Component 2).

Goal: automatically find physical couplings between sensors that no engineer specified --
e.g. pump vibration bleeding into a distant sensor through the building structure -- purely
from operational data. Every edge this recovers, once confirmed, tightens Theorem 1
(k-compromise resilience) and Theorem 2 (impossibility bound): both get stronger as the
coupling graph gets denser, so a discovered edge earns Member 1 a strictly better bound
without touching the proofs themselves.

This is NOT a GNN that runs message passing over a graph someone hands it (the usual PyG
use case). There is no known graph yet -- discovering it *is* the task. So the model learns
a continuous, differentiable adjacency matrix jointly with a forecasting objective: sensors
that help predict each other's next reading are coupled; sensors that don't, aren't. This
kind of self-supervised "graph structure learning" is the same idea behind MTGNN / Graph
WaveNet's learned adjacency, adapted here to physical sensor coupling instead of traffic
forecasting.

Design note on PyTorch Geometric: the project tech stack lists PyTorch Geometric for GNN
work. This module deliberately does NOT depend on it. The adjacency here is dense, small
(at most WADI's 127 sensors -> 127x127 -- trivial to keep dense), and changes every training
step as the learned embeddings move. Converting that to PyG's sparse edge_index/edge_weight
representation every forward pass just to call GCNConv adds overhead and pulls in PyG's
compiled extensions (torch-scatter/torch-sparse), which are a common source of Windows/CPU
wheel-mismatch pain, for zero benefit at this scale. The DenseGraphConv layer below performs
the exact same operation GCNConv does (symmetric-normalized-adjacency times features times a
weight matrix) in plain PyTorch. If a future plant has thousands of sensors and a dense
matrix stops being practical, swap DenseGraphConv for a real PyG GCNConv -- nothing else in
this file needs to change.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TemporalEncoder(nn.Module):
    """One sensor's history window -> one feature vector. Weight-tied across all sensors.

    Applied per-sensor by folding the sensor dimension into the batch dimension, so the
    encoder never sees "which sensor is this" -- only a univariate time series. That's what
    makes it directly transferable between plants with different sensor counts (SWaT's 51 vs
    WADI's 127): a different sensor count is just a different batch size to this module, not
    a different architecture.
    """

    def __init__(self, hidden_dim: int = 16, out_dim: int = 16):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_sensors, seq_len) raw operational readings. Returns (batch, n_sensors, out_dim)."""
        b, n, t = x.shape
        x = x.reshape(b * n, t, 1)
        _, h_n = self.gru(x)  # h_n: (1, b*n, hidden_dim)
        h = self.proj(h_n.squeeze(0))  # (b*n, out_dim)
        return h.reshape(b, n, -1)


class GraphLearningLayer(nn.Module):
    """Learns a persistent, sparse, non-negative, symmetric coupling-strength matrix.

    theta is a plain (n_sensors, embed_dim) parameter table of per-sensor embeddings.
    Coupling strength between i and j is their embedding similarity -- a dot product, which
    is automatically symmetric, matching physical coupling (heat/pressure/vibration transfer
    both ways). Non-negativity (via softplus) keeps the result a valid graph-Laplacian input
    for Member 1's Theorem 1/2 machinery and for Component 6's coupling_penalty.

    Unlike the temporal encoder, theta IS tied to a specific plant's specific sensor set --
    it has to be, since "which sensors are coupled" is exactly the plant-specific fact being
    learned. See transfer_to_new_plant() for how this is handled across plants.

    Top-k sparsification is recomputed fresh every forward call from the *current* theta, not
    fixed after an initial pass -- so as embeddings move during training, which edges survive
    can change, letting gradient descent actually search over structure instead of committing
    early to whatever a random initialization first favored. This follows the same hard
    top-k approach used in MTGNN's graph learning layer; a softer differentiable relaxation
    (Gumbel-softmax over edges) is a reasonable upgrade later if training proves unstable, but
    adds real complexity for a problem this small.
    """

    def __init__(self, n_sensors: int, embed_dim: int = 16, top_k: int = 5):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(n_sensors, embed_dim) * 0.1)
        self.embed_dim = embed_dim
        self.top_k = top_k

    def forward(self) -> torch.Tensor:
        scores = (self.theta @ self.theta.t()) / (self.embed_dim ** 0.5)  # (n, n), symmetric
        weights = F.softplus(scores)  # non-negative
        weights = weights - torch.diag(torch.diagonal(weights))  # drop self-loops

        n = weights.size(0)
        k = min(self.top_k, n - 1)
        _, topk_idx = torch.topk(weights, k=k, dim=1)
        mask = torch.zeros_like(weights)
        mask.scatter_(1, topk_idx, 1.0)
        mask = torch.maximum(mask, mask.t())  # keep an edge if either endpoint ranks it top-k

        return weights * mask  # (n_sensors, n_sensors)


class DenseGraphConv(nn.Module):
    """One graph-convolution step over a dense weighted adjacency. See module docstring for
    why this is plain PyTorch instead of PyG's GCNConv -- mathematically it's the same op.

    use_self_loop controls the standard GCN trick of adding identity to the adjacency before
    normalizing, which lets a node keep some of its own feature alongside its neighbours'.
    Keep it True for normal use. Set it False specifically when the training target has
    already had the self-history-predictable part removed (see SelfBaseline /
    train_coupling_gnn_two_stage below) -- with a self-loop, a node could partially
    reconstruct a residual target using its own mixed-in features regardless of which
    neighbours the graph selected, which quietly defeats the purpose of learning against a
    residual in the first place.
    """

    def __init__(self, in_dim: int, out_dim: int, use_self_loop: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.use_self_loop = use_self_loop

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_sensors, in_dim), adjacency: (n_sensors, n_sensors)."""
        n = adjacency.size(0)
        a_hat = adjacency + torch.eye(n, device=adjacency.device) if self.use_self_loop else adjacency
        degree = a_hat.sum(dim=1).clamp(min=1e-6)  # isolated nodes are possible without a self-loop
        d_inv_sqrt = torch.diag(degree.pow(-0.5))
        a_norm = d_inv_sqrt @ a_hat @ d_inv_sqrt

        propagated = torch.einsum("nm,bmd->bnd", a_norm, x)
        return F.relu(self.linear(propagated))


class CouplingGNN(nn.Module):
    """Full model: temporal encoder -> learned graph -> graph convolutions -> next-step forecast.

    The forecasting task is a means, not the end: the model is optimized to predict each
    sensor's next reading, and the *only* way the learned adjacency can help with that is by
    mixing in genuinely predictive information from other sensors. An edge that doesn't
    improve forecasts gets no gradient signal to keep it, so what survives after training is,
    by construction, edges that measurably help explain one sensor from another -- which is
    the operational signature of a real physical coupling.
    """

    def __init__(
        self,
        n_sensors: int,
        hidden_dim: int = 16,
        embed_dim: int = 16,
        top_k: int = 5,
        gcn_layers: int = 2,
        use_self_loop: bool = True,
    ):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.graph_layer = GraphLearningLayer(n_sensors, embed_dim=embed_dim, top_k=top_k)
        self.gcn_layers = nn.ModuleList(
            DenseGraphConv(hidden_dim, hidden_dim, use_self_loop=use_self_loop) for _ in range(gcn_layers)
        )
        self.predict_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_hist: (batch, n_sensors, seq_len). Returns (next_step_pred, adjacency).

        next_step_pred: (batch, n_sensors). adjacency: (n_sensors, n_sensors), shared across
        the whole batch since coupling structure is a property of the plant, not one window.
        """
        h = self.temporal_encoder(x_hist)
        adjacency = self.graph_layer()
        for layer in self.gcn_layers:
            h = layer(h, adjacency)
        pred = self.predict_head(h).squeeze(-1)
        return pred, adjacency


def train_coupling_gnn(
    model: CouplingGNN,
    windows: np.ndarray,
    next_values: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_sparsity: float = 1e-3,
    device: str = "cpu",
) -> List[float]:
    """Self-supervised training: predict next-step readings, with an L1-style sparsity
    penalty on the learned adjacency so the model is pushed toward a crisp, small set of
    high-confidence couplings rather than a dense blob of weak ones -- interpretability
    matters here, since a human (Member 1) has to be able to look at the surviving edges and
    judge whether they're physically plausible before folding them into the coupling graph.
    """
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    windows_t = torch.from_numpy(windows) if isinstance(windows, np.ndarray) else windows
    next_t = torch.from_numpy(next_values) if isinstance(next_values, np.ndarray) else next_values
    loader = DataLoader(TensorDataset(windows_t, next_t), batch_size=batch_size, shuffle=True)

    losses = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred, adjacency = model(xb)
            loss = F.mse_loss(pred, yb) + lambda_sparsity * adjacency.sum()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(xb)
        losses.append(epoch_loss / len(windows_t))
    return losses


class SelfBaseline(nn.Module):
    """Per-sensor-only forecaster: predicts x_i(t+1) from x_i's own recent history alone,
    with zero information from any other sensor. Weight-tied across sensors, like
    TemporalEncoder.

    Why this exists: fitting CouplingGNN directly against raw next-step values is a weak
    test of topology. A residual sensor's next reading is largely explainable from its own
    recent trend (autocorrelation) regardless of which neighbours the graph module picks, so
    a forecaster can reach low error while the learned adjacency stays close to uncorrelated
    with the true coupling structure -- confirmed empirically: joint training converges to a
    good forecast loss but recovers true edges barely above chance (~0.2 F1 across a sweep of
    top_k/sparsity/epochs). Predicting the LEFTOVER after subtracting what self-history
    already explains forces the graph to earn its keep -- the only way to reduce residual
    error further is genuine cross-sensor information, which is the real identifiability
    signal. This mirrors the project's own "physics first, ML on what's left" philosophy one
    level down: here it's "self-history first, graph on what's left".
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.encoder = TemporalEncoder(hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        """x_hist: (batch, n_sensors, seq_len). Returns (batch, n_sensors)."""
        h = self.encoder(x_hist)
        return self.head(h).squeeze(-1)


def train_self_baseline(
    model: SelfBaseline,
    windows: np.ndarray,
    next_values: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> List[float]:
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    windows_t = torch.from_numpy(windows) if isinstance(windows, np.ndarray) else windows
    next_t = torch.from_numpy(next_values) if isinstance(next_values, np.ndarray) else next_values
    loader = DataLoader(TensorDataset(windows_t, next_t), batch_size=batch_size, shuffle=True)

    losses = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * len(xb)
        losses.append(epoch_loss / len(windows_t))
    return losses


@torch.no_grad()
def compute_residual_targets(
    self_baseline: SelfBaseline, windows: np.ndarray, next_values: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """next_values minus what the (frozen) self-baseline already predicts from own history --
    the part of each sensor's next reading that self-history cannot explain, and therefore
    the part a genuine cross-sensor coupling would have to explain.
    """
    self_baseline.eval()
    windows_t = torch.from_numpy(windows) if isinstance(windows, np.ndarray) else windows
    next_t = torch.from_numpy(next_values) if isinstance(next_values, np.ndarray) else next_values
    preds = self_baseline(windows_t.to(device)).cpu().numpy()
    return next_t.numpy() - preds


def train_coupling_gnn_two_stage(
    n_sensors: int,
    windows: np.ndarray,
    next_values: np.ndarray,
    hidden_dim: int = 16,
    embed_dim: int = 16,
    top_k: int = 5,
    gcn_layers: int = 2,
    baseline_epochs: int = 30,
    graph_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_sparsity: float = 1e-3,
    device: str = "cpu",
) -> Tuple[SelfBaseline, CouplingGNN, List[float], List[float]]:
    """Recommended entry point: fit the self-baseline, freeze it, then fit CouplingGNN
    (with use_self_loop=False) against the residual it leaves behind. Returns
    (self_baseline, coupling_gnn, baseline_losses, graph_losses).

    train_coupling_gnn() above still exists for the plain single-stage version -- useful as
    a quick check or an ablation to show *why* the two-stage split matters, but this function
    is what should be used to actually produce a coupling graph for Member 1 to consume.
    """
    self_baseline = SelfBaseline(hidden_dim=hidden_dim)
    baseline_losses = train_self_baseline(
        self_baseline, windows, next_values, epochs=baseline_epochs, batch_size=batch_size, lr=lr, device=device
    )

    residual_targets = compute_residual_targets(self_baseline, windows, next_values, device=device)

    coupling_gnn = CouplingGNN(
        n_sensors=n_sensors,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        top_k=top_k,
        gcn_layers=gcn_layers,
        use_self_loop=False,
    )
    graph_losses = train_coupling_gnn(
        coupling_gnn,
        windows,
        residual_targets,
        epochs=graph_epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_sparsity=lambda_sparsity,
        device=device,
    )

    return self_baseline, coupling_gnn, baseline_losses, graph_losses


def extract_discovered_edges(
    adjacency: torch.Tensor, sensor_names: Optional[List[str]] = None, threshold: float = 0.1
) -> List[Tuple]:
    """Discovered edges above `threshold`, sorted strongest first. Returns (name_i, name_j, weight)."""
    adj = adjacency.detach().cpu().numpy()
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = adj[i, j]
            if w > threshold:
                name_i = sensor_names[i] if sensor_names else i
                name_j = sensor_names[j] if sensor_names else j
                edges.append((name_i, name_j, float(w)))
    edges.sort(key=lambda e: -e[2])
    return edges


def diff_against_manual_graph(
    discovered_edges: List[Tuple], manual_adjacency: np.ndarray, name_to_index: dict
) -> List[Tuple]:
    """Discovered edges NOT already present in the manually-specified coupling graph.

    Per the spec, "every discovered edge strengthens Theorem 1 and Theorem 2" -- but an edge
    the engineering docs already captured adds nothing new to the bound. This is the list
    worth reporting: genuinely new couplings for Member 1 to sanity-check and fold in.
    """
    new_edges = []
    for name_i, name_j, w in discovered_edges:
        i, j = name_to_index[name_i], name_to_index[name_j]
        if manual_adjacency[i, j] == 0:
            new_edges.append((name_i, name_j, w))
    return new_edges


def merge_graphs(
    manual_adjacency: np.ndarray, discovered_adjacency: np.ndarray, threshold: float = 0.1
) -> np.ndarray:
    """Union of manual edges and high-confidence discovered edges, for Theorem 1/2's k and
    impossibility bound -- a denser graph makes both strictly stronger.
    """
    combined = manual_adjacency.copy()
    mask = discovered_adjacency > threshold
    combined[mask] = np.maximum(combined[mask], discovered_adjacency[mask])
    return combined


def evaluate_edge_recovery(
    discovered_adjacency: torch.Tensor, ground_truth_adjacency: np.ndarray, threshold: float = 0.1
) -> dict:
    """Precision/recall of discovered edges against a *known* ground-truth coupling graph.

    Only meaningful on synthetic data (data_synth.py), where the true coupling structure is
    known by construction. This is the sanity check that structure learning actually works
    at all before trusting its output on real plant data, where no ground truth exists to
    check against -- exactly the same reasoning as validating the anomaly transformer against
    a learned attacker before trusting its accuracy numbers on known attacks alone.
    """
    pred = (discovered_adjacency.detach().cpu().numpy() > threshold).astype(int)
    true = (ground_truth_adjacency > 0).astype(int)
    n = pred.shape[0]
    iu = np.triu_indices(n, k=1)
    pred_edges, true_edges = pred[iu], true[iu]

    tp = int(((pred_edges == 1) & (true_edges == 1)).sum())
    fp = int(((pred_edges == 1) & (true_edges == 0)).sum())
    fn = int(((pred_edges == 0) & (true_edges == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def transfer_to_new_plant(
    pretrained_model: CouplingGNN, new_n_sensors: int, top_k: int = 5, freeze_transferred: bool = True
) -> CouplingGNN:
    """Build a CouplingGNN for a new plant's sensor count, reusing the pretrained model's
    temporal encoder + graph convolutions + prediction head (none of which depend on sensor
    identity or count), and leaving only the new plant's (n_sensors, embed_dim) node
    embedding table to learn from scratch.

    This is what makes "train on SWaT, fine-tune with minimal WADI data" cheap: instead of
    re-learning the whole forecasting pipeline, WADI only has to learn a small embedding
    table -- an easy problem even with little data, since the hard part (how to use a
    sensor's own recent history, how to propagate information across a coupling graph) was
    already learned from SWaT's much larger sensor set.

    freeze_transferred=True locks the reused layers so the small amount of WADI fine-tuning
    data can only move the embeddings, not overwrite what SWaT taught the shared layers.
    Set it False once there is enough WADI data to fine-tune the whole pipeline end to end.
    """
    new_model = CouplingGNN(
        n_sensors=new_n_sensors,
        hidden_dim=pretrained_model.temporal_encoder.proj.out_features,
        embed_dim=pretrained_model.graph_layer.embed_dim,
        top_k=top_k,
        gcn_layers=len(pretrained_model.gcn_layers),
        use_self_loop=pretrained_model.gcn_layers[0].use_self_loop,
    )
    new_model.temporal_encoder.load_state_dict(pretrained_model.temporal_encoder.state_dict())
    new_model.gcn_layers.load_state_dict(pretrained_model.gcn_layers.state_dict())
    new_model.predict_head.load_state_dict(pretrained_model.predict_head.state_dict())

    if freeze_transferred:
        for module in (new_model.temporal_encoder, new_model.gcn_layers, new_model.predict_head):
            for p in module.parameters():
                p.requires_grad_(False)

    return new_model


if __name__ == "__main__":
    # Smoke test + sanity check on synthetic data with a KNOWN ground-truth coupling graph:
    # train the GNN to forecast, then measure how well the discovered edges match the truth.
    from physattest.graph.data_synth import (
        make_ground_truth_graph,
        make_windows,
        simulate_coupled_series,
    )

    n_sensors, seq_len = 15, 20
    ground_truth = make_ground_truth_graph(n_sensors, avg_degree=3, seed=0)
    series = simulate_coupled_series(
        n_steps=4000, n_sensors=n_sensors, adjacency=ground_truth, coupling_strength=0.3, noise_std=0.1, seed=0
    )
    windows, next_values = make_windows(series, seq_len)

    # Single-stage (predicts raw next-step values directly) -- kept as a labeled ablation to
    # show why the two-stage split below matters, not as the recommended approach.
    single_stage_model = CouplingGNN(n_sensors=n_sensors, hidden_dim=16, embed_dim=16, top_k=3, gcn_layers=2)
    single_losses = train_coupling_gnn(single_stage_model, windows, next_values, epochs=60, batch_size=64, lr=2e-3)
    with torch.no_grad():
        single_adjacency = single_stage_model.graph_layer()
    single_metrics = evaluate_edge_recovery(single_adjacency, ground_truth, threshold=0.05)
    print("single-stage (raw next-value target) edge recovery:", single_metrics)

    # Two-stage (recommended): self-baseline first, graph fit against the residual it can't explain.
    self_baseline, model, baseline_losses, graph_losses = train_coupling_gnn_two_stage(
        n_sensors=n_sensors,
        windows=windows,
        next_values=next_values,
        hidden_dim=16,
        embed_dim=16,
        top_k=3,
        gcn_layers=2,
        baseline_epochs=30,
        graph_epochs=60,
        batch_size=64,
        lr=2e-3,
    )
    print("self-baseline loss per epoch:", [round(l, 4) for l in baseline_losses])
    print("graph-on-residual loss per epoch:", [round(l, 4) for l in graph_losses])

    with torch.no_grad():
        adjacency = model.graph_layer()

    edges = extract_discovered_edges(adjacency, threshold=0.05)
    print(f"discovered {len(edges)} edges, top 5: {edges[:5]}")

    metrics = evaluate_edge_recovery(adjacency, ground_truth, threshold=0.05)
    print("two-stage edge recovery vs. ground truth:", metrics)
    print(f"improvement over single-stage: F1 {single_metrics['f1']:.3f} -> {metrics['f1']:.3f}")

    # transfer test: a differently-sized "plant" reusing the pretrained shared layers
    transferred = transfer_to_new_plant(model, new_n_sensors=25, top_k=4)
    dummy = torch.randn(4, 25, seq_len)
    pred, adj = transferred(dummy)
    assert pred.shape == (4, 25)
    assert adj.shape == (25, 25)
    print("transfer_to_new_plant smoke test passed")

    print("gnn_discovery.py smoke test passed")
