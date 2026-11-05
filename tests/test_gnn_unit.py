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
