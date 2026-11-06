"""Model-agnostic active learning loop.

The loop receives three callables that capture the data and model in closures,
so it only ever manipulates index arrays. This keeps the loop independent of
whether the underlying model is LightGBM (numpy arrays) or a GNN (graph lists).

Iteration 0 fits on the seed set and evaluates before any acquisition.
Each subsequent iteration acquires batch_size new compounds from the pool,
retrains, and re-evaluates. The loop stops when n_iter is reached or the pool
is exhausted.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from src.active_learning.acquisition import explore, random_scores, ucb

logger = logging.getLogger(__name__)

FitFn = Callable[[np.ndarray], None]
PredictFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
EvalFn = Callable[[np.ndarray], float]


def run_active_learning_loop(
    *,
    fit_fn: FitFn,
    predict_fn: PredictFn,
    evaluate_fn: EvalFn,
    seed_indices: np.ndarray,
    pool_indices: np.ndarray,
    test_indices: np.ndarray,
    batch_size: int = 10,
    n_iter: int = 20,
    kappa: float = 1.0,
    strategy: str = "ucb",
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """Run the AL loop and return a list of {n_labeled, metric} dicts (one per iteration)."""
    labeled = seed_indices.copy()
    pool = pool_indices.copy()
    curve: list[dict[str, float]] = []

    fit_fn(labeled)
    metric = evaluate_fn(test_indices)
    curve.append({"n_labeled": float(len(labeled)), "metric": metric})
    logger.info("iter 0 (seed): n_labeled=%d  metric=%.4f", len(labeled), metric)

    for it in range(1, n_iter + 1):
        if len(pool) == 0:
            logger.info("Pool exhausted after %d iterations", it - 1)
            break

        effective_batch = min(batch_size, len(pool))

        if strategy == "random":
            scores = random_scores(n=len(pool), rng=rng)
        else:
            mu, sigma = predict_fn(pool)
            if strategy == "ucb":
                scores = ucb(mu=mu, sigma=sigma, kappa=kappa)
            elif strategy == "explore":
                scores = explore(sigma=sigma)
            else:
                raise ValueError(f"Unknown strategy: {strategy!r}")

        top_local = np.argsort(scores)[-effective_batch:]
        new_global = pool[top_local]
        labeled = np.concatenate([labeled, new_global])
        pool = np.delete(pool, top_local)

        fit_fn(labeled)
        metric = evaluate_fn(test_indices)
        curve.append({"n_labeled": float(len(labeled)), "metric": metric})
        logger.info(
            "iter %d [%s]: n_labeled=%d  pool_remaining=%d  metric=%.4f",
            it, strategy, len(labeled), len(pool), metric,
        )

    return curve
