"""Train and evaluate LightGBM models for all 6 ADMET endpoints.

Scaffold split (not random). Metrics match TDC leaderboard convention:
  Regression:     RMSE, R²
  Classification: ROC-AUC, AUPRC
Each endpoint is evaluated over several split seeds and reported as mean ± sd.
Run: python -m src.models.predict
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from math import nan
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import (
    auc,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.constants import ENDPOINT_TASK, LOG_TRANSFORM, Endpoint, Task
from src.features.features import (
    FEATURE_HASH,
    get_scaffold_groups,
    load_or_featurize,
    quantile_bins,
)
from src.logging_config import setup_logging
from src.paths import CONFIG, INTERIM, MODELS_DIR, MODELS_LGBM, REPORTS_PREDICT

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

# Quantile bins used to stratify regression folds. Quantile bins are invariant
# to monotonic transforms, so binning raw y matches binning the log-transformed y.
STRATIFY_BINS = 5
N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50


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


def _run_one_fold(
    endpoint: Endpoint,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
) -> dict[str, float]:
    task = ENDPOINT_TASK[endpoint]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Hold out 15% of train as internal val for early stopping. Random split is
    # fine here — scaffold separation is already enforced at the outer fold level.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    callbacks: list[Any] = [early_stopping(EARLY_STOPPING_ROUNDS), log_evaluation(0)]

    params = _model_params(endpoint)
    if task == Task.REGRESSION:
        y_tr_fit = np.log1p(y_tr) if endpoint in LOG_TRANSFORM else y_tr
        y_val_fit = np.log1p(y_val) if endpoint in LOG_TRANSFORM else y_val
        regressor = LGBMRegressor(**params)
        regressor.fit(X_tr, y_tr_fit, eval_X=X_val, eval_y=y_val_fit, callbacks=callbacks)
        y_pred = np.asarray(regressor.predict(X_test))
        if endpoint in LOG_TRANSFORM:
            y_pred = np.expm1(y_pred)
        rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
        r2 = float(r2_score(y_test, y_pred))
        logger.info("fold %d — RMSE: %.4f   R²: %.4f  (trees: %d)", fold, rmse, r2, regressor.best_iteration_)
        return {"rmse": rmse, "r2": r2}

    classifier = LGBMClassifier(**params)
    classifier.fit(X_tr, y_tr, eval_X=X_val, eval_y=y_val, callbacks=callbacks)
    y_prob = np.asarray(classifier.predict_proba(X_test))[:, 1]
    roc = float(roc_auc_score(y_test, y_prob))
    prc = _auprc(y_test, y_prob)
    logger.info("fold %d — ROC-AUC: %.4f   AUPRC: %.4f  (trees: %d)", fold, roc, prc, classifier.best_iteration_)
    return {"roc_auc": roc, "auprc": prc}


def run_endpoint(
    endpoint: Endpoint, n_folds: int = N_FOLDS
) -> dict[str, tuple[float, float]]:
    """Evaluate one endpoint with stratified scaffold GroupKFold cross-validation.

    Returns {metric: (mean, sd)}. No scaffold appears in both train and test
    within any fold — each fold tests on a chemically distinct held-out set.
    CV results are cached; reruns with unchanged features and params are instant.
    """
    logger.info("--- %s ---", endpoint)
    df = pd.read_parquet(INTERIM / f"{endpoint}.parquet")
    X, y, smiles = load_or_featurize(endpoint=endpoint, df=df)

    params = _model_params(endpoint)
    params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    cv_cache = MODELS_LGBM / f"{endpoint}_cv_cache.json"

    if cv_cache.exists():
        cached = json.loads(cv_cache.read_text())
        if (cached.get("feature_hash") == FEATURE_HASH
                and cached.get("params_hash") == params_hash
                and cached.get("n_folds") == n_folds):
            logger.info("Loading cached CV metrics for %s", endpoint)
            return {metric: tuple(v) for metric, v in cached["metrics"].items()}  # type: ignore[return-value]

    groups = get_scaffold_groups(smiles)
    task = ENDPOINT_TASK[endpoint]
    strat_labels = y.astype(int) if task == Task.CLASSIFICATION else quantile_bins(y, STRATIFY_BINS)

    gkf = StratifiedGroupKFold(n_splits=n_folds)
    per_fold = [
        _run_one_fold(endpoint, X, y, train_idx, test_idx, fold)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, strat_labels, groups=groups))
    ]
    result: dict[str, tuple[float, float]] = {
        metric: (
            float(np.mean([r[metric] for r in per_fold])),
            float(np.std([r[metric] for r in per_fold], ddof=1)) if len(per_fold) > 1 else nan,
        )
        for metric in per_fold[0]
    }
    cv_cache.write_text(json.dumps({
        "feature_hash": FEATURE_HASH,
        "params_hash": params_hash,
        "n_folds": n_folds,
        "metrics": {metric: list(v) for metric, v in result.items()},
    }, indent=2))
    return result


def train_and_save(endpoint: Endpoint) -> None:
    """Train on full dataset and save to models/{endpoint}_lgbm.pkl for screening."""
    out = MODELS_LGBM / f"{endpoint}_lgbm.pkl"
    meta = MODELS_LGBM / f"{endpoint}_lgbm_meta.json"
    df = pd.read_parquet(INTERIM / f"{endpoint}.parquet")
    X, y, _ = load_or_featurize(endpoint=endpoint, df=df)
    params = _model_params(endpoint)
    params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]

    if out.exists() and meta.exists():
        m = json.loads(meta.read_text())
        if m.get("feature_hash") == FEATURE_HASH and m.get("params_hash") == params_hash:
            logger.info("Model up to date, skipping → %s", out.name)
            return

    task = ENDPOINT_TASK[endpoint]
    if task == Task.REGRESSION:
        y_fit = np.log1p(y) if endpoint in LOG_TRANSFORM else y
        model: LGBMRegressor | LGBMClassifier = LGBMRegressor(**params)
    else:
        y_fit = y
        model = LGBMClassifier(**params)
    model.fit(X, y_fit)
    joblib.dump(model, out)
    meta.write_text(json.dumps({"feature_hash": FEATURE_HASH, "params_hash": params_hash}))
    logger.info("Saved model → %s", out.name)


def main() -> None:
    setup_logging()
    mlflow.set_tracking_uri(f"sqlite:///{MODELS_DIR / 'mlflow.db'}")
    mlflow.set_experiment("admet")

    run_ts = datetime.now()
    results: dict[str, dict[str, tuple[float, float]]] = {}

    with mlflow.start_run(run_name=run_ts.strftime("%Y%m%d_%H%M%S")):
        mlflow.log_params({**_LGBM_BASE, "n_folds": N_FOLDS})
        for endpoint in Endpoint:
            metrics = run_endpoint(endpoint)
            results[endpoint] = metrics
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
                "n_folds": N_FOLDS,
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

    logger.info("\n=== SUMMARY (mean ± sd over %d folds) ===", N_FOLDS)
    for name, metrics in results.items():
        metric_str = "   ".join(
            f"{k}: {mean:.4f} ± {sd:.4f}" for k, (mean, sd) in metrics.items()
        )
        logger.info("%-40s %s", name, metric_str)

    logger.info("\nTraining final models on full data for screening ...")
    for endpoint in Endpoint:
        train_and_save(endpoint)


if __name__ == "__main__":
    main()
