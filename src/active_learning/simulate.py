"""Active learning simulation for all 12 ADMET endpoints.

Simulates the closed-loop drug discovery workflow: start with a small seed set,
iteratively acquire the most informative compounds (by UCB or explore strategy),
retrain, and measure held-out performance. Compares against random acquisition.

Both LightGBM (ConformalLGBM) and GNN (MCDropoutGNN) uncertainty models are
supported. GNN runs only when a trained model file exists in models/gnn/.

Run: python -m src.active_learning.simulate [--endpoint ENDPOINT] [--model lgbm|gnn|both]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

from src.active_learning.loop import run_active_learning_loop
from src.active_learning.uncertainty import ConformalLGBM, MCDropoutGNN
from src.cli import resolve_endpoints
from src.constants import ENDPOINT_TASK, Endpoint, Task
from src.features.features import load_or_featurize, scaffold_split
from src.logging_config import setup_logging
from src.paths import INTERIM, MODELS_GNN, REPORTS_AL

logger = logging.getLogger(__name__)

_SEED_FRACTION = 0.10
_MIN_SEED = 50
_DEFAULT_N_ITER = 10
_DEFAULT_BATCH = 10
_DEFAULT_KAPPA = 1.0
_AL_EPOCHS = 20  # GNN epochs per AL iteration


def _metric_name(task: Task) -> str:
    return "rmse" if task == Task.REGRESSION else "roc_auc"


def _evaluate(
    mu: np.ndarray, y_true: np.ndarray, task: Task
) -> float:
    if task == Task.REGRESSION:
        return float(mean_squared_error(y_true, mu) ** 0.5)
    return float(roc_auc_score(y_true, mu))


def _run_lgbm_simulation(
    endpoint: Endpoint,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    strategies: list[str],
    n_iter: int,
    batch_size: int,
    kappa: float,
) -> dict[str, list[dict[str, float]]]:
    task = ENDPOINT_TASK[endpoint]
    rng = np.random.default_rng(42)

    n_seed = max(_MIN_SEED, int(len(train_idx) * _SEED_FRACTION))
    perm = rng.permutation(len(train_idx))
    seed_local, pool_local = perm[:n_seed], perm[n_seed:]
    seed_idx = train_idx[seed_local]
    pool_idx = train_idx[pool_local]

    logger.info(
        "%s LGBM: seed=%d  pool=%d  test=%d",
        endpoint, len(seed_idx), len(pool_idx), len(test_idx),
    )

    results: dict[str, list[dict[str, float]]] = {}
    for strategy in strategies:
        model = ConformalLGBM(endpoint=endpoint)

        def fit_fn(labeled_idx: np.ndarray, _m: ConformalLGBM = model) -> None:
            _m.fit(X=X[labeled_idx], y=y[labeled_idx])

        def predict_fn(
            pool_idx_: np.ndarray, _m: ConformalLGBM = model
        ) -> tuple[np.ndarray, np.ndarray]:
            return _m.predict_with_uncertainty(X=X[pool_idx_])

        def evaluate_fn(
            test_idx_: np.ndarray, _m: ConformalLGBM = model
        ) -> float:
            mu, _ = _m.predict_with_uncertainty(X=X[test_idx_])
            return _evaluate(mu=mu, y_true=y[test_idx_], task=task)

        logger.info("--- strategy: %s ---", strategy)
        curve = run_active_learning_loop(
            fit_fn=fit_fn,
            predict_fn=predict_fn,
            evaluate_fn=evaluate_fn,
            seed_indices=seed_idx,
            pool_indices=pool_idx,
            test_indices=test_idx,
            batch_size=batch_size,
            n_iter=n_iter,
            kappa=kappa,
            strategy=strategy,
            rng=np.random.default_rng(42),
        )
        results[strategy] = curve

    return results


def _run_gnn_simulation(
    endpoint: Endpoint,
    smiles_full: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    strategies: list[str],
    n_iter: int,
    batch_size: int,
    kappa: float,
    al_epochs: int,
) -> dict[str, list[dict[str, float]]]:
    try:
        from src.features.mol_graph import load_or_build_dataset
    except ImportError:
        logger.warning("torch_geometric not available — skipping GNN simulation")
        return {}

    model_path = MODELS_GNN / f"{endpoint}_gnn.pt"
    if not model_path.exists():
        logger.warning(
            "No trained GNN model found for %s (%s) — skipping GNN simulation. "
            "Run 'make gnn --endpoint %s' first.",
            endpoint, model_path.name, endpoint,
        )
        return {}

    df = pd.read_parquet(INTERIM / f"{endpoint}.parquet")
    graphs, graph_smiles = load_or_build_dataset(endpoint=endpoint, df=df)
    gnn_y = np.array([float(g.y.item()) for g in graphs])

    # Remap LGBM indices (into smiles_full) to GNN indices (into graph_smiles).
    # Molecules that failed conformer generation are absent from graph_smiles.
    smiles_to_gnn: dict[str, int] = {s: i for i, s in enumerate(graph_smiles)}
    task = ENDPOINT_TASK[endpoint]

    def _remap(idx: np.ndarray) -> np.ndarray:
        mapped = [smiles_to_gnn[smiles_full[i]] for i in idx if smiles_full[i] in smiles_to_gnn]
        return np.array(mapped, dtype=int)

    gnn_train = _remap(train_idx)
    gnn_test = _remap(test_idx)

    rng = np.random.default_rng(42)
    n_seed = max(_MIN_SEED, int(len(gnn_train) * _SEED_FRACTION))
    perm = rng.permutation(len(gnn_train))
    seed_local, pool_local = perm[:n_seed], perm[n_seed:]
    seed_idx = gnn_train[seed_local]
    pool_idx = gnn_train[pool_local]

    logger.info(
        "%s GNN: seed=%d  pool=%d  test=%d",
        endpoint, len(seed_idx), len(pool_idx), len(gnn_test),
    )

    results: dict[str, list[dict[str, float]]] = {}
    for strategy in strategies:
        mc_model = MCDropoutGNN(endpoint=endpoint)
        first_fit = True

        def fit_fn(
            labeled_idx: np.ndarray,
            _m: MCDropoutGNN = mc_model,
        ) -> None:
            nonlocal first_fit
            _m.fit(
                graphs=[graphs[i] for i in labeled_idx],
                y=gnn_y[labeled_idx],
                epochs=al_epochs,
                warm_start=not first_fit,
            )
            first_fit = False

        def predict_fn(
            pool_idx_: np.ndarray,
            _m: MCDropoutGNN = mc_model,
        ) -> tuple[np.ndarray, np.ndarray]:
            return _m.predict_with_uncertainty(graphs=[graphs[i] for i in pool_idx_])

        def evaluate_fn(
            test_idx_: np.ndarray,
            _m: MCDropoutGNN = mc_model,
        ) -> float:
            mu, _ = _m.predict_with_uncertainty(graphs=[graphs[i] for i in test_idx_])
            return _evaluate(mu=mu, y_true=gnn_y[test_idx_], task=task)

        logger.info("--- strategy: %s ---", strategy)
        curve = run_active_learning_loop(
            fit_fn=fit_fn,
            predict_fn=predict_fn,
            evaluate_fn=evaluate_fn,
            seed_indices=seed_idx,
            pool_indices=pool_idx,
            test_indices=gnn_test,
            batch_size=batch_size,
            n_iter=n_iter,
            kappa=kappa,
            strategy=strategy,
            rng=np.random.default_rng(42),
        )
        results[strategy] = curve

    return results


def _save_results(
    results: dict[str, list[dict[str, float]]],
    *,
    model: str,
    endpoint: Endpoint,
    task: Task,
    n_iter: int,
    batch_size: int,
    kappa: float,
    n_test: int,
    ts: str,
    extra: dict,
) -> None:
    for strategy, curve in results.items():
        report = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": str(endpoint),
            "task": str(task),
            "model": model,
            "strategy": strategy,
            "n_iter": n_iter,
            "batch_size": batch_size,
            "kappa": kappa,
            "n_test": n_test,
            "metric_name": _metric_name(task),
            **extra,
            "curve": curve,
        }
        path = REPORTS_AL / f"{endpoint}_{model}_{strategy}_{ts}.json"
        path.write_text(json.dumps(report, indent=2))
        logger.info("Report -> %s", path.name)


def run_simulation(
    endpoint: Endpoint,
    models: list[str],
    strategies: list[str],
    n_iter: int = _DEFAULT_N_ITER,
    batch_size: int = _DEFAULT_BATCH,
    kappa: float = _DEFAULT_KAPPA,
    al_epochs: int = _AL_EPOCHS,
) -> None:
    logger.info("=== %s ===", endpoint)
    task = ENDPOINT_TASK[endpoint]
    X, y, smiles = load_or_featurize(endpoint=endpoint)

    train_idx, test_idx = scaffold_split(
        smiles=smiles,
        test_size=0.2,
        y=y if task == Task.CLASSIFICATION else None,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    common = dict(
        endpoint=endpoint,
        task=task,
        n_iter=n_iter,
        batch_size=batch_size,
        kappa=kappa,
        n_test=int(len(test_idx)),
        ts=ts,
    )

    if "lgbm" in models:
        _save_results(
            _run_lgbm_simulation(
                endpoint=endpoint,
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                strategies=strategies,
                n_iter=n_iter,
                batch_size=batch_size,
                kappa=kappa,
            ),
            model="lgbm",
            extra={"n_train": int(len(train_idx))},
            **common,  # type: ignore[arg-type]
        )

    if "gnn" in models:
        _save_results(
            _run_gnn_simulation(
                endpoint=endpoint,
                smiles_full=smiles,
                train_idx=train_idx,
                test_idx=test_idx,
                strategies=strategies,
                n_iter=n_iter,
                batch_size=batch_size,
                kappa=kappa,
                al_epochs=al_epochs,
            ),
            model="gnn",
            extra={"al_epochs": al_epochs},
            **common,  # type: ignore[arg-type]
        )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Active learning simulation for ADMET endpoints"
    )
    parser.add_argument("--endpoint", default=None, help="Endpoint short key or prefix (default: all)")
    parser.add_argument(
        "--model",
        default="lgbm",
        choices=["lgbm", "gnn", "both"],
        help="Uncertainty model (default: lgbm)",
    )
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["ucb", "explore", "random", "all"],
        help="Acquisition strategy; all runs ucb+explore+random (default: all)",
    )
    parser.add_argument("--n-iter", type=int, default=_DEFAULT_N_ITER)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--kappa", type=float, default=_DEFAULT_KAPPA)
    parser.add_argument(
        "--al-epochs",
        type=int,
        default=_AL_EPOCHS,
        help="GNN training epochs per AL iteration (default: 20)",
    )
    args = parser.parse_args()

    endpoints = resolve_endpoints(parser=parser, name=args.endpoint)
    models = ["lgbm", "gnn"] if args.model == "both" else [args.model]
    strategies = ["ucb", "explore", "random"] if args.strategy == "all" else [args.strategy]

    for endpoint in endpoints:
        run_simulation(
            endpoint=endpoint,
            models=models,
            strategies=strategies,
            n_iter=args.n_iter,
            batch_size=args.batch_size,
            kappa=args.kappa,
            al_epochs=args.al_epochs,
        )

    logger.info("Active learning simulation complete. Reports -> %s", REPORTS_AL)


if __name__ == "__main__":
    main()
