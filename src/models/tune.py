"""Hyperparameter search for LightGBM models using Optuna.

Performs scaffold-aware train/val split within the training fold so the
held-out test set is never touched. Best params saved to config/.
Run: python -m src.models.tune [--endpoint ENDPOINT] [--n-trials N]
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

from src.constants import ENDPOINT_TASK, LOG_TRANSFORM, Endpoint, Task
from src.features.features import load_or_featurize, scaffold_split
from src.paths import CONFIG, MODELS_DIR, RAW

logger = logging.getLogger(__name__)

N_TRIALS_DEFAULT = 50

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: Task,
    log_transform: bool,
) -> float:
    params: dict[str, Any] = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        subsample=trial.suggest_float("subsample", 0.4, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if task == Task.REGRESSION:
        y_tr = np.log1p(y_train) if log_transform else y_train
        model = LGBMRegressor(**params)
        model.fit(X_train, y_tr)
        y_pred = np.asarray(model.predict(X_val))
        if log_transform:
            y_pred = np.expm1(y_pred)
        return float(mean_squared_error(y_val, y_pred) ** 0.5)
    else:
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        y_prob = np.asarray(model.predict_proba(X_val))[:, 1]
        return -float(roc_auc_score(y_val, y_prob))


def tune_endpoint(endpoint: Endpoint, n_trials: int = N_TRIALS_DEFAULT) -> dict:
    logger.info("--- Tuning %s (%d trials) ---", endpoint, n_trials)
    df = pd.read_parquet(RAW / f"{endpoint}.parquet")
    X, y, smiles = load_or_featurize(endpoint=endpoint, df=df)

    task = ENDPOINT_TASK[endpoint]
    log_transform = endpoint in LOG_TRANSFORM

    # Outer split mirrors predict.py — test set never touched
    train_idx, _ = scaffold_split(
        smiles=smiles, test_size=0.2, y=y if task == Task.CLASSIFICATION else None
    )
    # Inner split: validation carved from train only
    train_smiles = [smiles[i] for i in train_idx]
    train_y = y[train_idx]
    inner_train_idx, val_idx = scaffold_split(
        smiles=train_smiles,
        test_size=0.2,
        y=train_y if task == Task.CLASSIFICATION else None,
    )

    X_tr = X[train_idx][inner_train_idx]
    y_tr = y[train_idx][inner_train_idx]
    X_val = X[train_idx][val_idx]
    y_val = y[train_idx][val_idx]
    logger.info("Train %d / val %d", len(X_tr), len(X_val))

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _objective(
            trial=trial,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            task=task,
            log_transform=log_transform,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    raw_val = study.best_value
    metric_name = "RMSE" if task == Task.REGRESSION else "ROC-AUC"
    display_val = raw_val if task == Task.REGRESSION else -raw_val
    logger.info("Best %s (val): %.4f", metric_name, display_val)

    out = CONFIG / f"{endpoint}_best_params.json"
    out.write_text(json.dumps(best, indent=2))
    logger.info("Saved → %s", out.name)
    return best


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Tune LightGBM hyperparameters with Optuna")
    parser.add_argument("--endpoint", default=None, help="Single endpoint to tune (default: all)")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    args = parser.parse_args()

    endpoints = [Endpoint(args.endpoint)] if args.endpoint else list(Endpoint)

    mlflow.set_tracking_uri(f"sqlite:///{MODELS_DIR / 'mlflow.db'}")
    mlflow.set_experiment("admet-tuning")

    with mlflow.start_run():
        for endpoint in endpoints:
            best = tune_endpoint(endpoint=endpoint, n_trials=args.n_trials)
            for k, v in best.items():
                mlflow.log_param(key=f"{endpoint}.{k}", value=v)


if __name__ == "__main__":
    main()
