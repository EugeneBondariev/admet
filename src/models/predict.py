"""Train and evaluate LightGBM models for all 6 ADMET endpoints.

Scaffold split (not random). Metrics match TDC leaderboard convention:
  Regression:     RMSE, R²
  Classification: ROC-AUC, AUPRC
Each endpoint is evaluated over several split seeds and reported as mean ± sd.
Run: python -m src.models.predict
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from math import nan

import mlflow
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

from src.constants import ENDPOINT_TASK, LOG_TRANSFORM, Endpoint, Task
from src.features.features import load_or_featurize, scaffold_split
from src.paths import CONFIG, INTERIM, MODELS_DIR, REPORTS_PREDICT

logger = logging.getLogger(__name__)

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

# Quantile bins used to stratify regression splits. Quantile bins are invariant
# to monotonic transforms, so binning raw y matches binning the log-transformed y.
STRATIFY_BINS = 5

# Split seeds to average over. Which equal-sized scaffold groups reach test is
# seed-dependent, so one split is a sample, not an estimate.
SPLIT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def _model_params(endpoint: Endpoint) -> dict:
    tuned = CONFIG / f"{endpoint}_best_params.json"
    if tuned.exists():
        params = json.loads(tuned.read_text())
        logger.info("Using tuned params for %s", endpoint)
        return {**params, "random_state": 42, "n_jobs": -1, "verbose": -1}
    return _LGBM_BASE


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

    params = _model_params(endpoint)
    if task == Task.REGRESSION:
        if endpoint in LOG_TRANSFORM:
            y_train = np.log1p(y_train)
        regressor = LGBMRegressor(**params)
        regressor.fit(X_train, y_train)
        y_pred = np.asarray(regressor.predict(X_test))
        if endpoint in LOG_TRANSFORM:
            y_pred = np.expm1(y_pred)
        rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
        r2 = float(r2_score(y_test, y_pred))
        logger.info("seed %d — RMSE: %.4f   R²: %.4f", seed, rmse, r2)
        return {"rmse": rmse, "r2": r2}

    classifier = LGBMClassifier(**params)
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
    df = pd.read_parquet(INTERIM / f"{endpoint}.parquet")
    X, y, smiles = load_or_featurize(endpoint=endpoint, df=df)

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
    mlflow.set_tracking_uri(f"sqlite:///{MODELS_DIR / 'mlflow.db'}")
    mlflow.set_experiment("admet")

    run_ts = datetime.now()
    results: dict[str, dict[str, tuple[float, float]]] = {}

    with mlflow.start_run(run_name=run_ts.strftime("%Y%m%d_%H%M%S")):
        mlflow.log_params({**_LGBM_BASE, "split_seeds": list(SPLIT_SEEDS)})
        for endpoint in Endpoint:
            metrics = run_endpoint(endpoint)
            results[str(endpoint)] = metrics
            for name, (mean, sd) in metrics.items():
                # sd is logged alongside the mean: a mean without its spread
                # invites reading noise as signal when comparing runs.
                mlflow.log_metric(key=f"{endpoint}.{name}", value=mean)
                mlflow.log_metric(key=f"{endpoint}.{name}_sd", value=sd)

    report_path = REPORTS_PREDICT / f"predict_{run_ts.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(
        json.dumps(
            {
                "timestamp": run_ts.isoformat(),
                "params": _LGBM_BASE,
                "split_seeds": list(SPLIT_SEEDS),
                "results": {
                    endpoint: {
                        name: {"mean": mean, "sd": sd} for name, (mean, sd) in metrics.items()
                    }
                    for endpoint, metrics in results.items()
                },
            },
            indent=2,
        )
    )
    logger.info("Report → %s", report_path)

    logger.info("\n=== SUMMARY (mean ± sd over %d split seeds) ===", len(SPLIT_SEEDS))
    for name, metrics in results.items():
        metric_str = "   ".join(
            f"{k}: {mean:.4f} ± {sd:.4f}" for k, (mean, sd) in metrics.items()
        )
        logger.info("%-40s %s", name, metric_str)


if __name__ == "__main__":
    main()
