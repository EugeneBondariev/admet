"""Uncertainty quantification wrappers for LightGBM and SchNet GNN.

Both classes expose the same interface:
    fit(data, y)                          -> None
    predict_with_uncertainty(data)        -> (mu, sigma)

ConformalLGBM uses normalized split conformal prediction: nonconformity scores
are divided by a KNN-distance difficulty estimate so that pool compounds far from
the training set get wider intervals, giving compound-specific sigma values.

MCDropoutGNN runs N stochastic forward passes with dropout active at inference,
returning mean and std across passes. Requires MolSchNet(dropout>0).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neighbors import NearestNeighbors

from src.constants import ENDPOINT_TASK, LOG_TRANSFORM, Endpoint, Task

if TYPE_CHECKING:
    from src.models.gnn import MolSchNet

logger = logging.getLogger(__name__)

_LGBM_AL_PARAMS: dict = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=63,
    # small datasets at start of loop — relax min samples per leaf
    min_child_samples=5,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


class ConformalLGBM:
    """LightGBM with normalized split conformal prediction for compound-specific uncertainty.

    Calibration is a random 20% internal hold-out from the labeled set; the
    remaining 80% trains the LightGBM model and a KNN used as a difficulty
    estimator. Sigma for a pool compound equals q_hat * knn_distance, so
    compounds far from the training distribution receive larger intervals.
    """

    def __init__(self, endpoint: Endpoint, alpha: float = 0.1) -> None:
        self._task = ENDPOINT_TASK[endpoint]
        self._endpoint = endpoint
        self._alpha = alpha
        self._model: LGBMRegressor | LGBMClassifier | None = None
        self._knn: NearestNeighbors | None = None
        self._q_hat: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n = len(X)
        n_cal = max(5, int(n * 0.2))
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        cal_idx, tr_idx = perm[:n_cal], perm[n_cal:]

        X_tr, X_cal = X[tr_idx], X[cal_idx]
        y_tr, y_cal = y[tr_idx], y[cal_idx]

        k = min(5, len(X_tr) - 1)
        self._knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        self._knn.fit(X_tr)

        if self._task == Task.REGRESSION:
            y_tr_fit = np.log1p(y_tr) if self._endpoint in LOG_TRANSFORM else y_tr
            self._model = LGBMRegressor(**_LGBM_AL_PARAMS)
            self._model.fit(X_tr, y_tr_fit)
            cal_preds = np.asarray(self._model.predict(X_cal))
            if self._endpoint in LOG_TRANSFORM:
                cal_preds = np.expm1(cal_preds)
            distances, _ = self._knn.kneighbors(X_cal)
            difficulty = distances.mean(axis=1) + 1e-6
            scores = np.abs(y_cal - cal_preds) / difficulty
        else:
            self._model = LGBMClassifier(**_LGBM_AL_PARAMS)
            self._model.fit(X_tr, y_tr)
            cal_proba = np.asarray(self._model.predict_proba(X_cal))
            # nonconformity = 1 - P(true class)
            scores = 1.0 - cal_proba[np.arange(n_cal), y_cal.astype(int)]

        level = min(1.0, np.ceil((n_cal + 1) * (1 - self._alpha)) / n_cal)
        self._q_hat = float(np.quantile(scores, level))
        if self._q_hat == np.max(scores):
            logger.warning(
                "Conformal q_hat hit the max nonconformity score "
                "(n_cal=%d, alpha=%.2f) — sigma is uninformative this iteration",
                n_cal, self._alpha,
            )

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._model is not None and self._knn is not None, "call fit() first"

        if self._task == Task.REGRESSION:
            mu = np.asarray(self._model.predict(X))
            if self._endpoint in LOG_TRANSFORM:
                mu = np.expm1(mu)
            distances, _ = self._knn.kneighbors(X)
            difficulty = distances.mean(axis=1) + 1e-6
            sigma = self._q_hat * difficulty
        else:
            proba = np.asarray(self._model.predict_proba(X))
            mu = proba[:, 1]
            sigma = self._q_hat * np.sqrt(mu * (1.0 - mu))

        return mu, sigma


class MCDropoutGNN:
    """SchNet GNN with MC dropout for per-compound uncertainty.

    Runs n_passes stochastic forward passes with dropout active at inference.
    Mu is the mean prediction; sigma is the std across passes.
    For classification, raw logit mean/std are computed first, then sigmoid is
    applied to mu so it lies in (0, 1); sigma stays in logit space.

    Requires MolSchNet(dropout>0); the default MolSchNet(dropout=0) uses
    nn.Identity and will return sigma=0 for all compounds.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        n_passes: int = 20,
        dropout: float = 0.1,
    ) -> None:
        self._task = ENDPOINT_TASK[endpoint]
        self._endpoint = endpoint
        self._n_passes = n_passes
        self._dropout = dropout
        self._model: MolSchNet | None = None
        self._optimizer: object | None = None
        self._device: object | None = None

    def _ensure_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("MCDropoutGNN requires torch — run from the .venv") from exc

    def fit(
        self,
        graphs: list,
        y: np.ndarray,
        epochs: int = 20,
        warm_start: bool = False,
    ) -> None:
        self._ensure_torch()
        import torch
        import torch.nn as nn
        from torch_geometric.loader import DataLoader

        from src.models.gnn import MolSchNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        if not warm_start or self._model is None:
            self._model = MolSchNet(dropout=self._dropout).to(device)
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)  # type: ignore[union-attr]

        model: MolSchNet = self._model  # type: ignore[assignment]
        optimizer = self._optimizer

        batch_size = min(32, len(graphs))
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        criterion = (
            nn.MSELoss() if self._task == Task.REGRESSION else nn.BCEWithLogitsLoss()
        )

        log_transform = self._endpoint in LOG_TRANSFORM
        model.train()
        for epoch in range(epochs):
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()  # type: ignore[union-attr]
                out = model(z=batch.z, pos=batch.pos, batch=batch.batch).squeeze(-1)
                if self._task == Task.REGRESSION and log_transform:
                    target = torch.log1p(batch.y.squeeze())
                else:
                    target = batch.y.squeeze()
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()  # type: ignore[union-attr]
        logger.info(
            "MCDropoutGNN fit: %d graphs, %d epochs, warm_start=%s",
            len(graphs), epochs, warm_start,
        )

    def predict_with_uncertainty(
        self, graphs: list
    ) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_torch()
        import torch
        from torch_geometric.loader import DataLoader

        assert self._model is not None, "call fit() first"
        model = self._model

        log_transform = self._endpoint in LOG_TRANSFORM
        device = self._device
        loader = DataLoader(graphs, batch_size=32)

        all_passes: list[np.ndarray] = []
        model.train()  # keep dropout active
        with torch.no_grad():
            for _ in range(self._n_passes):
                preds: list[np.ndarray] = []
                for batch in loader:
                    batch = batch.to(device)
                    out = model(z=batch.z, pos=batch.pos, batch=batch.batch).squeeze(-1)
                    preds.append(out.cpu().numpy())
                all_passes.append(np.concatenate(preds))

        stacked = np.stack(all_passes)  # (n_passes, n_molecules)
        mu_raw = stacked.mean(axis=0)
        sigma = stacked.std(axis=0)

        if self._task == Task.REGRESSION:
            mu = np.expm1(mu_raw) if log_transform else mu_raw
        else:
            # sigmoid of mean logit; sigma stays in logit space for acquisition
            mu = 1.0 / (1.0 + np.exp(-mu_raw))

        return mu, sigma
