"""Train and evaluate LightGBM models for all 5 ADMET endpoints.

Scaffold split (not random). Metrics match TDC leaderboard convention:
  Regression:     RMSE, R²
  Classification: ROC-AUC, AUPRC
Run: python -m src.models.predict
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from math import nan

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    auc,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)

from src.constants import ENDPOINT_TASK, Endpoint, Task
from src.features.features import featurize, scaffold_split
from src.paths import PROCESSED, RAW

logger = logging.getLogger(__name__)

LOG_TRANSFORM: set[Endpoint] = {Endpoint.CLEARANCE}

# Quantile bins used to stratify regression splits. Quantile bins are invariant
# to monotonic transforms, so binning raw y matches binning the log-transformed y.
STRATIFY_BINS = 5

# Split seeds to average over. Which equal-sized scaffold groups reach test is
# seed-dependent, so one split is a sample, not an estimate.
SPLIT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

_LGBM_BASE: dict = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def _load_or_featurize(
    endpoint: Endpoint, df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cache = PROCESSED / f"{endpoint}_features.npz"
    if cache.exists():
        raw = RAW / f"{endpoint}.parquet"
        if raw.stat().st_mtime > cache.stat().st_mtime:
            logger.warning(
                "Raw data for %s is newer than feature cache — delete %s to rebuild",
                endpoint,
                cache,
            )
        logger.info("Loading cached features for %s", endpoint)
        data = np.load(cache)
        return data["X"], data["y"], list(data["smiles"])
    X, y, smiles = featurize(df)
    np.savez_compressed(str(cache), X=X, y=y, smiles=np.array(smiles))
    logger.info("Cached features → %s", cache.name)
    return X, y, smiles


def _auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def _run_one_split(
    endpoint: Endpoint,
    X: np.ndarray,
    y: np.ndarray,
    smiles: list[str],
    seed: int,
) -> dict[str, float]:
    task = ENDPOINT_TASK[endpoint]
    train_idx, test_idx = scaffold_split(
        smiles,
        y=y,
        random_state=seed,
        stratify_bins=None if task == Task.CLASSIFICATION else STRATIFY_BINS,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if task == Task.REGRESSION:
        if endpoint in LOG_TRANSFORM:
            y_train = np.log1p(y_train)
        regressor = LGBMRegressor(**_LGBM_BASE)
        regressor.fit(X_train, y_train)
        y_pred = np.asarray(regressor.predict(X_test))
        if endpoint in LOG_TRANSFORM:
            y_pred = np.expm1(y_pred)
        rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
        r2 = float(r2_score(y_test, y_pred))
        logger.info("seed %d — RMSE: %.4f   R²: %.4f", seed, rmse, r2)
        return {"rmse": rmse, "r2": r2}

    classifier = LGBMClassifier(**_LGBM_BASE)
    classifier.fit(X_train, y_train)
    y_prob = np.asarray(classifier.predict_proba(X_test))[:, 1]
    roc = float(roc_auc_score(y_test, y_prob))
    prc = _auprc(y_test, y_prob)
    logger.info("seed %d — ROC-AUC: %.4f   AUPRC: %.4f", seed, roc, prc)
    return {"roc_auc": roc, "auprc": prc}


def run_endpoint(
    endpoint: Endpoint, seeds: Sequence[int] = SPLIT_SEEDS
) -> dict[str, tuple[float, float]]:
    """Evaluate one endpoint over several split seeds.

    Returns {metric: (mean, sd)}. Reseeding decides which equal-sized scaffold
    groups land in test, which moves the metrics by more than most changes being
    measured — so a single split is not a usable estimate. sd is nan for one seed.
    """
    logger.info("--- %s ---", endpoint)
    df = pd.read_parquet(RAW / f"{endpoint}.parquet")
    X, y, smiles = _load_or_featurize(endpoint, df)

    per_seed = [_run_one_split(endpoint, X, y, smiles, seed) for seed in seeds]
    return {
        metric: (
            float(np.mean([r[metric] for r in per_seed])),
            float(np.std([r[metric] for r in per_seed], ddof=1)) if len(per_seed) > 1 else nan,
        )
        for metric in per_seed[0]
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results: dict[str, dict[str, tuple[float, float]]] = {}
    for endpoint in Endpoint:
        results[str(endpoint)] = run_endpoint(endpoint)

    logger.info("\n=== SUMMARY (mean ± sd over %d split seeds) ===", len(SPLIT_SEEDS))
    for name, metrics in results.items():
        metric_str = "   ".join(
            f"{k}: {mean:.4f} ± {sd:.4f}" for k, (mean, sd) in metrics.items()
        )
        logger.info("%-40s %s", name, metric_str)


if __name__ == "__main__":
    main()
