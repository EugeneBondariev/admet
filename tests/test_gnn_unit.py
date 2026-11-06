"""Unit tests for GNN components that don't need real data.

Skipped automatically on the system Python where torch is not installed.
Run with the venv Python: .venv/Scripts/python.exe -m pytest tests/test_gnn_unit.py
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from src.models.gnn import _TorchRadiusGraph  # noqa: E402


def _make_batch(*molecule_sizes: int) -> torch.Tensor:
    """Build a batch index tensor for molecules of given sizes."""
    return torch.cat([torch.full((n,), i) for i, n in enumerate(molecule_sizes)])


def test_radius_graph_finds_pair_within_cutoff() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    batch = _make_batch(2)

    edge_index, edge_weight = _TorchRadiusGraph(cutoff=2.0)(pos, batch)

    assert edge_index.shape[1] == 2  # one edge each direction
    assert (edge_weight < 2.0).all()


def test_radius_graph_excludes_pair_beyond_cutoff() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    batch = _make_batch(2)

    edge_index, _ = _TorchRadiusGraph(cutoff=2.0)(pos, batch)

    assert edge_index.shape[1] == 0


def test_radius_graph_no_self_loops() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    batch = _make_batch(3)

    edge_index, _ = _TorchRadiusGraph(cutoff=5.0)(pos, batch)

    assert (edge_index[0] != edge_index[1]).all()


def test_radius_graph_no_cross_molecule_edges() -> None:
    # Two molecules of 2 atoms each; atoms 0,1 in mol 0, atoms 2,3 in mol 1.
    # All atoms are close to each other, but cross-molecule edges must not appear.
    pos = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # molecule 0
        [0.5, 0.0, 0.0], [1.5, 0.0, 0.0],  # molecule 1 (overlaps in space)
    ])
    batch = _make_batch(2, 2)

    edge_index, _ = _TorchRadiusGraph(cutoff=5.0)(pos, batch)

    src_batch = batch[edge_index[0]]
    dst_batch = batch[edge_index[1]]
    assert (src_batch == dst_batch).all()


def test_radius_graph_edge_weights_are_euclidean_distances() -> None:
    pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])  # distance = 5.0
    batch = _make_batch(2)

    _, edge_weight = _TorchRadiusGraph(cutoff=6.0)(pos, batch)

    assert edge_weight.shape[0] == 2  # bidirectional
    assert edge_weight[0].item() == pytest.approx(5.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Training checkpoint — save / load / restore roundtrip
# ---------------------------------------------------------------------------

from src.models.gnn import MolSchNet  # noqa: E402


def test_training_checkpoint_roundtrip(tmp_path) -> None:
    """Checkpoint dict can be saved, loaded, and used to restore exact model state."""
    model = MolSchNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    ckpt_path = tmp_path / "train_ckpt.pt"
    torch.save({
        "epoch": 5,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val": 0.42,
        "best_epoch": 3,
        "best_state": {k: v.clone() for k, v in model.state_dict().items()},
        "no_improve": 2,
    }, ckpt_path)

    loaded = torch.load(ckpt_path, weights_only=False)

    assert loaded["epoch"] == 5
    assert loaded["best_val"] == pytest.approx(0.42)
    assert loaded["no_improve"] == 2

    model2 = MolSchNet()
    model2.load_state_dict(loaded["model_state"])
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_training_checkpoint_contains_required_keys(tmp_path) -> None:
    model = MolSchNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    ckpt = {
        "epoch": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val": float("inf"),
        "best_epoch": 0,
        "best_state": {},
        "no_improve": 0,
    }
    ckpt_path = tmp_path / "train_ckpt.pt"
    torch.save(ckpt, ckpt_path)

    loaded = torch.load(ckpt_path, weights_only=False)
    required = {"epoch", "model_state", "optimizer_state", "scheduler_state",
                "best_val", "best_epoch", "best_state", "no_improve"}
    assert required <= loaded.keys()
