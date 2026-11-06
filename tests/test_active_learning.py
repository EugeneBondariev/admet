"""Unit tests for the active learning module.

GNN tests are skipped automatically when torch_geometric is not installed.
All other tests run on system Python.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from src.active_learning.acquisition import explore, random_scores, ucb
from src.active_learning.loop import run_active_learning_loop
from src.active_learning.uncertainty import ConformalLGBM
from src.constants import Endpoint

_HAS_TORCH_GEO = importlib.util.find_spec("torch_geometric") is not None
_skip_gnn = pytest.mark.skipif(not _HAS_TORCH_GEO, reason="torch_geometric not installed")


# ---------------------------------------------------------------------------
# acquisition.py
# ---------------------------------------------------------------------------

def test_ucb_higher_sigma_wins() -> None:
    mu = np.array([1.0, 1.0])
    sigma = np.array([0.1, 0.5])
    scores = ucb(mu=mu, sigma=sigma, kappa=1.0)
    assert scores[1] > scores[0]


def test_ucb_kappa_zero_equals_mu() -> None:
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([10.0, 10.0, 10.0])
    np.testing.assert_array_almost_equal(ucb(mu=mu, sigma=sigma, kappa=0.0), mu)


def test_explore_ignores_mu() -> None:
    sigma = np.array([0.1, 0.5, 0.3])
    result = explore(sigma=sigma)
    np.testing.assert_array_equal(result, sigma)
    assert result is not sigma  # returns a copy


def test_random_scores_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    scores = random_scores(n=100, rng=rng)
    assert scores.shape == (100,)
    assert scores.min() >= 0.0
    assert scores.max() < 1.0


def test_random_scores_differ_across_rng_seeds() -> None:
    s1 = random_scores(n=50, rng=np.random.default_rng(0))
    s2 = random_scores(n=50, rng=np.random.default_rng(1))
    assert not np.allclose(s1, s2)


# ---------------------------------------------------------------------------
# ConformalLGBM — regression
# ---------------------------------------------------------------------------

def _regression_data(n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (X[:, 0] * 2.0 + rng.standard_normal(n) * 0.5).astype(np.float32)
    return X, y


def _classification_data(n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 20)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    return X, y


def test_conformal_lgbm_regression_shapes() -> None:
    X, y = _regression_data()
    model = ConformalLGBM(endpoint=Endpoint.SOLUBILITY)
    model.fit(X=X[:100], y=y[:100])
    mu, sigma = model.predict_with_uncertainty(X=X[100:])
    assert mu.shape == (20,)
    assert sigma.shape == (20,)


def test_conformal_lgbm_sigma_nonnegative() -> None:
    X, y = _regression_data()
    model = ConformalLGBM(endpoint=Endpoint.SOLUBILITY)
    model.fit(X=X[:100], y=y[:100])
    _, sigma = model.predict_with_uncertainty(X=X[100:])
    assert (sigma >= 0).all()


def test_conformal_lgbm_sigma_finite() -> None:
    X, y = _regression_data()
    model = ConformalLGBM(endpoint=Endpoint.SOLUBILITY)
    model.fit(X=X[:100], y=y[:100])
    mu, sigma = model.predict_with_uncertainty(X=X[100:])
    assert np.isfinite(mu).all()
    assert np.isfinite(sigma).all()


def test_conformal_lgbm_classification_shapes() -> None:
    X, y = _classification_data()
    model = ConformalLGBM(endpoint=Endpoint.HERG)
    model.fit(X=X[:100], y=y[:100])
    mu, sigma = model.predict_with_uncertainty(X=X[100:])
    assert mu.shape == (20,)
    assert sigma.shape == (20,)


def test_conformal_lgbm_classification_mu_in_zero_one() -> None:
    X, y = _classification_data()
    model = ConformalLGBM(endpoint=Endpoint.HERG)
    model.fit(X=X[:100], y=y[:100])
    mu, _ = model.predict_with_uncertainty(X=X[100:])
    assert (mu >= 0).all() and (mu <= 1).all()


def test_conformal_lgbm_classification_sigma_finite() -> None:
    X, y = _classification_data()
    model = ConformalLGBM(endpoint=Endpoint.HERG)
    model.fit(X=X[:100], y=y[:100])
    _, sigma = model.predict_with_uncertainty(X=X[100:])
    assert np.isfinite(sigma).all()
    assert (sigma >= 0).all()


def test_conformal_lgbm_tiny_calibration_set() -> None:
    """q_hat must be finite even when only 5 calibration samples are available."""
    X, y = _regression_data(n=30)
    model = ConformalLGBM(endpoint=Endpoint.SOLUBILITY)
    model.fit(X=X, y=y)  # n_cal = max(5, 6) = 6 from 30 samples
    assert np.isfinite(model._q_hat)
    assert model._q_hat >= 0


def test_conformal_lgbm_knn_guard_tiny_X() -> None:
    """KNN n_neighbors clamped to len(X_train)-1 for very small datasets."""
    X, y = _regression_data(n=10)
    model = ConformalLGBM(endpoint=Endpoint.SOLUBILITY)
    model.fit(X=X, y=y)  # X_train has ~8 samples; n_neighbors clamped to 5
    mu, sigma = model.predict_with_uncertainty(X=X[:3])
    assert mu.shape == (3,)
    assert (sigma >= 0).all()


# ---------------------------------------------------------------------------
# loop.py — with stub closures
# ---------------------------------------------------------------------------

def _make_stubs(
    n_total: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_total, 10)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)

    all_idx = np.arange(n_total)
    test_idx = all_idx[:40]
    seed_idx = all_idx[40:60]
    pool_idx = all_idx[60:]

    fit_calls: list[int] = []
    eval_calls: list[int] = []

    model = ConformalLGBM(endpoint=Endpoint.HERG)

    def fit_fn(labeled_idx: np.ndarray) -> None:
        fit_calls.append(len(labeled_idx))
        model.fit(X=X[labeled_idx], y=y[labeled_idx])

    def predict_fn(pool_idx_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return model.predict_with_uncertainty(X=X[pool_idx_])

    def evaluate_fn(test_idx_: np.ndarray) -> float:
        mu, _ = model.predict_with_uncertainty(X=X[test_idx_])
        return float(np.mean(np.abs(y[test_idx_] - mu)))

    return fit_fn, predict_fn, evaluate_fn, fit_calls, eval_calls, seed_idx, pool_idx, test_idx  # type: ignore[return-value]


def test_loop_returns_n_iter_plus_one_entries() -> None:
    fit_fn, predict_fn, evaluate_fn, _, _, seed_idx, pool_idx, test_idx = _make_stubs()
    curve = run_active_learning_loop(
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        evaluate_fn=evaluate_fn,
        seed_indices=seed_idx,
        pool_indices=pool_idx,
        test_indices=test_idx,
        batch_size=10,
        n_iter=5,
        kappa=1.0,
        strategy="ucb",
        rng=np.random.default_rng(0),
    )
    assert len(curve) == 6  # seed + 5 iterations


def test_loop_n_labeled_increases_by_batch_size() -> None:
    fit_fn, predict_fn, evaluate_fn, _, _, seed_idx, pool_idx, test_idx = _make_stubs()
    curve = run_active_learning_loop(
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        evaluate_fn=evaluate_fn,
        seed_indices=seed_idx,
        pool_indices=pool_idx,
        test_indices=test_idx,
        batch_size=10,
        n_iter=4,
        kappa=1.0,
        strategy="ucb",
        rng=np.random.default_rng(0),
    )
    for i in range(1, len(curve)):
        assert curve[i]["n_labeled"] == curve[i - 1]["n_labeled"] + 10


def test_loop_random_strategy() -> None:
    fit_fn, predict_fn, evaluate_fn, _, _, seed_idx, pool_idx, test_idx = _make_stubs()
    curve = run_active_learning_loop(
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        evaluate_fn=evaluate_fn,
        seed_indices=seed_idx,
        pool_indices=pool_idx,
        test_indices=test_idx,
        batch_size=10,
        n_iter=3,
        kappa=1.0,
        strategy="random",
        rng=np.random.default_rng(0),
    )
    assert len(curve) == 4


def test_loop_explore_strategy() -> None:
    fit_fn, predict_fn, evaluate_fn, _, _, seed_idx, pool_idx, test_idx = _make_stubs()
    curve = run_active_learning_loop(
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        evaluate_fn=evaluate_fn,
        seed_indices=seed_idx,
        pool_indices=pool_idx,
        test_indices=test_idx,
        batch_size=10,
        n_iter=3,
        kappa=1.0,
        strategy="explore",
        rng=np.random.default_rng(0),
    )
    assert len(curve) == 4


def test_loop_stops_when_pool_exhausted() -> None:
    fit_fn, predict_fn, evaluate_fn, _, _, seed_idx, pool_idx, test_idx = _make_stubs()
    # pool has 140 items; batch=50 → 2 full iters + 1 partial = pool exhausted
    curve = run_active_learning_loop(
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        evaluate_fn=evaluate_fn,
        seed_indices=seed_idx,
        pool_indices=pool_idx,
        test_indices=test_idx,
        batch_size=50,
        n_iter=1000,
        kappa=1.0,
        strategy="random",
        rng=np.random.default_rng(0),
    )
    # Should stop before 1000 iters
    assert len(curve) < 10
    assert curve[-1]["n_labeled"] == float(len(seed_idx) + len(pool_idx))


def test_loop_unknown_strategy_raises() -> None:
    fit_fn, predict_fn, evaluate_fn, _, _, seed_idx, pool_idx, test_idx = _make_stubs()
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_active_learning_loop(
            fit_fn=fit_fn,
            predict_fn=predict_fn,
            evaluate_fn=evaluate_fn,
            seed_indices=seed_idx,
            pool_indices=pool_idx,
            test_indices=test_idx,
            batch_size=10,
            n_iter=1,
            kappa=1.0,
            strategy="invalid",
            rng=np.random.default_rng(0),
        )


# ---------------------------------------------------------------------------
# MCDropoutGNN — skipped when torch_geometric not installed
# ---------------------------------------------------------------------------

_GNN_SMILES = ["CC", "CCC", "CCO", "c1ccccc1", "CCCC"]


def _make_gnn_fixtures():
    from src.features.mol_graph import smiles_to_graph

    graphs = [
        g for s, i in zip(_GNN_SMILES, range(len(_GNN_SMILES)))
        if (g := smiles_to_graph(s, float(i))) is not None
    ]
    gnn_y = np.array([float(i) for i in range(len(graphs))], dtype=np.float32)
    return graphs, gnn_y


@_skip_gnn
def test_mcdropout_gnn_regression_shapes() -> None:
    from src.active_learning.uncertainty import MCDropoutGNN

    graphs, gnn_y = _make_gnn_fixtures()
    model = MCDropoutGNN(endpoint=Endpoint.SOLUBILITY)
    model.fit(graphs=graphs, y=gnn_y, epochs=2)
    mu, sigma = model.predict_with_uncertainty(graphs=graphs)
    assert mu.shape == (len(graphs),)
    assert sigma.shape == (len(graphs),)


@_skip_gnn
def test_mcdropout_gnn_sigma_positive() -> None:
    """With dropout active, sigma should be non-zero across 20 passes."""
    from src.active_learning.uncertainty import MCDropoutGNN

    graphs, gnn_y = _make_gnn_fixtures()
    model = MCDropoutGNN(endpoint=Endpoint.SOLUBILITY, n_passes=20, dropout=0.1)
    model.fit(graphs=graphs, y=gnn_y, epochs=2)
    _, sigma = model.predict_with_uncertainty(graphs=graphs)
    assert (sigma >= 0).all()
    assert sigma.sum() > 0


@_skip_gnn
def test_mcdropout_gnn_classification_mu_in_zero_one() -> None:
    from src.active_learning.uncertainty import MCDropoutGNN

    graphs, gnn_y = _make_gnn_fixtures()
    y_cls = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)[: len(graphs)]
    model = MCDropoutGNN(endpoint=Endpoint.HERG)
    model.fit(graphs=graphs, y=y_cls, epochs=2)
    mu, _ = model.predict_with_uncertainty(graphs=graphs)
    assert (mu >= 0).all() and (mu <= 1).all()


@_skip_gnn
def test_mcdropout_gnn_warm_start_continues_training() -> None:
    """warm_start=True must not reset weights — weights should differ from cold start."""
    import copy

    import torch

    from src.active_learning.uncertainty import MCDropoutGNN

    graphs, gnn_y = _make_gnn_fixtures()
    model = MCDropoutGNN(endpoint=Endpoint.SOLUBILITY, dropout=0.1)
    model.fit(graphs=graphs, y=gnn_y, epochs=5, warm_start=False)

    state_before = copy.deepcopy(model._model.state_dict())  # type: ignore[union-attr]

    model.fit(graphs=graphs, y=gnn_y, epochs=5, warm_start=True)
    state_after = model._model.state_dict()  # type: ignore[union-attr]

    any_changed = any(
        not torch.equal(state_before[k], state_after[k])
        for k in state_before
    )
    assert any_changed, "warm_start=True should continue training, not reset weights"
