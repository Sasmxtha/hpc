"""Transfer learning for the anomaly transformer (Component 6): reuse what training learned
on one plant's sensor set when moving to a differently-sized plant, per the spec's "train on
SWaT, fine-tune with minimal WADI data."

Same split used for the GNN's transfer_to_new_plant (physattest/graph/gnn_discovery.py):
separate what depends on WHICH/HOW MANY sensors exist from what doesn't, transfer the
sensor-count-agnostic part as-is, and let only the sensor-count-dependent part learn fresh.

For ResidualAnomalyTransformer that split is:
  sensor-count-DEPENDENT: input_proj (Linear(n_sensors, d_model)) -- its shape is literally
    tied to n_sensors, so it cannot be reused across a different sensor count.
  sensor-count-AGNOSTIC: positional encoding, transformer encoder layers, timestep_head,
    pool_query/pool_attn, window_head -- everything downstream of the initial per-timestep
    embedding operates purely in d_model space, with no dependence on how many sensors fed
    into it.

AttackerGenerator has the same shape: its first and last Conv1d layers are sized by
n_sensors (input concatenates n_sensors residual channels with latent noise; output must
emit n_sensors channels), the middle Conv1d is not.
"""

import torch

from physattest.ml.anomaly_transformer import ResidualAnomalyTransformer
from physattest.ml.attacker import AttackerGenerator


def transfer_transformer_to_new_plant(
    pretrained: ResidualAnomalyTransformer, new_n_sensors: int, freeze_transferred: bool = True
) -> ResidualAnomalyTransformer:
    """Builds a fresh ResidualAnomalyTransformer for new_n_sensors, reusing every layer of
    `pretrained` except input_proj. freeze_transferred locks the reused layers so a small
    amount of target-plant fine-tuning data can only adapt the new input_proj, not overwrite
    what the source plant taught the shared layers -- this is what makes fine-tuning on
    limited target-plant data viable at all: the part that has to learn from near-zero data
    is a single Linear(new_n_sensors, d_model) layer, not the whole detector.
    """
    new_model = ResidualAnomalyTransformer(n_sensors=new_n_sensors, **pretrained.config)

    new_model.pos_enc.load_state_dict(pretrained.pos_enc.state_dict())
    new_model.encoder.load_state_dict(pretrained.encoder.state_dict())
    new_model.timestep_head.load_state_dict(pretrained.timestep_head.state_dict())
    new_model.pool_attn.load_state_dict(pretrained.pool_attn.state_dict())
    new_model.window_head.load_state_dict(pretrained.window_head.state_dict())
    with torch.no_grad():
        new_model.pool_query.copy_(pretrained.pool_query)

    if freeze_transferred:
        transferred_modules = (
            new_model.pos_enc,
            new_model.encoder,
            new_model.timestep_head,
            new_model.pool_attn,
            new_model.window_head,
        )
        for module in transferred_modules:
            for p in module.parameters():
                p.requires_grad_(False)
        new_model.pool_query.requires_grad_(False)

    return new_model


def transfer_attacker_to_new_plant(
    pretrained: AttackerGenerator, new_n_sensors: int, freeze_transferred: bool = True
) -> AttackerGenerator:
    """Same split for the adversarial-training attacker (physattest/ml/attacker.py): only
    net[0] (first Conv1d, sized by n_sensors+latent_dim) and net[4] (last Conv1d, sized by
    n_sensors) depend on sensor count; net[2], the middle Conv1d, does not.
    """
    new_model = AttackerGenerator(
        n_sensors=new_n_sensors,
        latent_dim=pretrained.latent_dim,
        hidden_dim=pretrained.net[0].out_channels,
        kernel_size=pretrained.net[0].kernel_size[0],
    )
    new_model.net[2].load_state_dict(pretrained.net[2].state_dict())
    if freeze_transferred:
        for p in new_model.net[2].parameters():
            p.requires_grad_(False)
    return new_model


if __name__ == "__main__":
    import numpy as np

    from physattest.ml.adversarial_training import AdversarialTrainingConfig, warmstart_detector
    from physattest.ml.data_synth import generate_normal_windows, make_coupling_graph, make_labeled_eval_set, to_tensor
    from physattest.ml.evaluate import evaluate_detection

    torch.manual_seed(0)
    np.random.seed(0)

    seq_len = 20
    source_n_sensors = 20  # "SWaT-like": more sensors, plenty of data
    target_n_sensors = 8  # "WADI-like": deliberately a DIFFERENT count, forcing the transfer split to matter

    # --- Pretrain a source model with plenty of data ---
    # d_model=32/2 layers/80 warm-start epochs, not the smaller size used elsewhere in this
    # codebase's smoke tests: an earlier attempt at this validation with a much smaller,
    # more lightly-trained source model (d_model=16, 1 layer, 40 epochs) found transfer
    # LOSING to training from scratch on every configuration tried, sometimes badly (AUC
    # near 0.5 -- essentially random). Inspecting the frozen model's pre-fine-tuning output
    # showed why: with so little learned structure in the source model to begin with, the
    # frozen deeper layers plus a freshly-initialized input_proj produced near-constant,
    # uninformative scores that fine-tuning on a handful of examples couldn't recover from.
    # A source model that has actually learned something worth transferring is a
    # precondition for transfer to help at all -- not a given just because the code runs.
    source_adj, _ = make_coupling_graph(source_n_sensors, avg_degree=3, seed=0)
    source_normal = to_tensor(generate_normal_windows(400, seq_len, source_n_sensors, source_adj, seed=0))
    source_labeled_windows, source_labeled_labels = make_labeled_eval_set(
        n_normal=150, n_attack=150, seq_len=seq_len, n_sensors=source_n_sensors, adjacency=source_adj, seed=1
    )
    source_model = ResidualAnomalyTransformer(n_sensors=source_n_sensors, d_model=32, nhead=4, num_layers=2)
    cfg_pretrain = AdversarialTrainingConfig(epochs=1, batch_size=32, warmstart_epochs=80, lr_detector=2e-3)
    warmstart_detector(
        source_model,
        source_normal[:250],
        to_tensor(source_labeled_windows),
        torch.from_numpy(source_labeled_labels).float(),
        cfg_pretrain,
    )

    # Sanity check the source model actually learned something on its OWN domain before
    # trusting its features are worth transferring anywhere.
    source_test_windows, source_test_labels = make_labeled_eval_set(
        n_normal=40, n_attack=40, seq_len=seq_len, n_sensors=source_n_sensors, adjacency=source_adj, seed=2
    )
    source_self_metrics = evaluate_detection(source_model, to_tensor(source_test_windows), source_test_labels)
    print(f"source model pretrained on {source_n_sensors} sensors, self-eval AUC={source_self_metrics.auc:.3f}")

    # --- Target plant: different sensor count, deliberately LIMITED fine-tuning data ---
    target_adj, _ = make_coupling_graph(target_n_sensors, avg_degree=2, seed=5)
    target_normal_all = to_tensor(generate_normal_windows(80, seq_len, target_n_sensors, target_adj, seed=10))
    target_labeled_windows, target_labeled_labels = make_labeled_eval_set(
        n_normal=8, n_attack=8, seq_len=seq_len, n_sensors=target_n_sensors, adjacency=target_adj, seed=11
    )
    # A larger, separately-seeded held-out set purely for evaluation, never used in training.
    test_windows, test_labels = make_labeled_eval_set(
        n_normal=40, n_attack=40, seq_len=seq_len, n_sensors=target_n_sensors, adjacency=target_adj, seed=12
    )

    cfg_finetune = AdversarialTrainingConfig(epochs=1, batch_size=8, warmstart_epochs=80, lr_detector=3e-3)

    # (a) transfer: reuse the source model's shared layers, fine-tune only the new input_proj.
    # freeze_transferred=True matters here, not just as a default: an earlier test with
    # freeze_transferred=False (letting the whole model move) consistently did WORSE than
    # freezing, because gradients from only 16 labeled examples are noisy enough to overwrite
    # the pretrained structure faster than they can improve on it. Freezing protects the
    # transferred layers specifically because the fine-tuning data is this scarce.
    transfer_model = transfer_transformer_to_new_plant(source_model, target_n_sensors, freeze_transferred=True)
    warmstart_detector(
        transfer_model,
        target_normal_all[:16],
        to_tensor(target_labeled_windows),
        torch.from_numpy(target_labeled_labels).float(),
        cfg_finetune,
    )

    # (b) from scratch: identical architecture, identical limited data, identical epoch budget
    scratch_model = ResidualAnomalyTransformer(n_sensors=target_n_sensors, d_model=32, nhead=4, num_layers=2)
    warmstart_detector(
        scratch_model,
        target_normal_all[:16],
        to_tensor(target_labeled_windows),
        torch.from_numpy(target_labeled_labels).float(),
        cfg_finetune,
    )

    transfer_metrics = evaluate_detection(transfer_model, to_tensor(test_windows), test_labels)
    scratch_metrics = evaluate_detection(scratch_model, to_tensor(test_windows), test_labels)

    print(f"\ntarget plant: {target_n_sensors} sensors, {len(target_labeled_windows)} labeled fine-tuning windows")
    print(f"transfer:    AUC={transfer_metrics.auc:.3f} F1={transfer_metrics.f1:.3f}")
    print(f"from-scratch: AUC={scratch_metrics.auc:.3f} F1={scratch_metrics.f1:.3f}")

    assert transfer_metrics.f1 >= scratch_metrics.f1, (
        "transfer learning should be at least as good as training from scratch on limited target-plant data"
    )

    # Lightweight shape/plumbing check for the attacker's transfer path too (not accuracy-
    # validated the way the detector is above -- the attacker has no "correctness" metric of
    # its own outside the full adversarial loop, which is already validated in
    # adversarial_training.py).
    source_attacker = AttackerGenerator(n_sensors=source_n_sensors, latent_dim=4, hidden_dim=16)
    transferred_attacker = transfer_attacker_to_new_plant(source_attacker, target_n_sensors)
    dummy = torch.randn(2, seq_len, target_n_sensors)
    delta = transferred_attacker(dummy, epsilon=0.3)
    assert delta.shape == dummy.shape
    print("transfer_attacker_to_new_plant shape check passed")

    print("\ntransfer_learning.py smoke test passed")
