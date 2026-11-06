"""SchNet graph neural network for ADMET prediction using 3D molecular geometry.

Uses RDKit ETKDGv3 conformer generation + PyTorch Geometric SchNet.
Runs all 5 endpoints and logs results to MLflow for comparison with LightGBM.
Run: python -m src.models.gnn [--endpoint ENDPOINT] [--epochs N]

GPU is strongly recommended. CPU training for CYP3A4 (12k molecules) can
take several hours. Use --epochs 30 for a quick smoke test on CPU.
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime

# Must precede torch_geometric imports to suppress torch.jit.script FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.jit.script.*")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve, r2_score, roc_auc_score  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.nn import SchNet  # noqa: E402

from src.constants import ENDPOINT_TASK, LOG_TRANSFORM, Endpoint, Task  # noqa: E402
from src.features.features import scaffold_split  # noqa: E402
from src.features.mol_graph import load_or_build_dataset  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.paths import INTERIM, MODELS_DIR, MODELS_GNN, REPORTS_GNN  # noqa: E402

logger = logging.getLogger(__name__)

BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 20
LR = 1e-3
HIDDEN = 128
NUM_INTERACTIONS = 6


CUTOFF = 10.0
TRAIN_CHECKPOINT_EVERY = 5


class _TorchRadiusGraph(nn.Module):
    """Pure-PyTorch radius graph — no pyg-lib required. O(N²) per batch, fine for small molecules."""

    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(
        self, pos: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.cdist(pos, pos)  # [N, N]
        same_mol = batch.unsqueeze(0) == batch.unsqueeze(1)  # [N, N]
        no_self = ~torch.eye(pos.size(0), dtype=torch.bool, device=pos.device)
        mask = (dist < self.cutoff) & same_mol & no_self
        row, col = mask.nonzero(as_tuple=True)
        edge_index = torch.stack([row, col])
        edge_weight = dist[row, col]
        return edge_index, edge_weight


class MolSchNet(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = SchNet(
            hidden_channels=HIDDEN,
            num_filters=HIDDEN,
            num_interactions=NUM_INTERACTIONS,
            num_gaussians=50,
            cutoff=CUTOFF,
        )
        # SchNet's default interaction_graph requires pyg-lib; replace with pure-PyTorch fallback
        self.net.interaction_graph = _TorchRadiusGraph(cutoff=CUTOFF)
        # dropout=0 → Identity; dropout>0 enables MC dropout uncertainty at inference
        self.dropout: nn.Module = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        return self.dropout(self.net(z=z, pos=pos, batch=batch))


def _auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def _eval(
    model: MolSchNet,
    loader: DataLoader,
    device: torch.device,
    task: Task,
    log_transform: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(z=batch.z, pos=batch.pos, batch=batch.batch).squeeze(-1)
            preds.append(out.cpu().numpy())
            targets.append(batch.y.cpu().numpy())
    preds_arr = np.concatenate(preds).ravel()
    targets_arr = np.concatenate(targets).ravel()
    if task == Task.CLASSIFICATION:
        preds_arr = torch.sigmoid(torch.tensor(preds_arr)).numpy()
    elif log_transform:
        preds_arr = np.expm1(preds_arr)
    return preds_arr, targets_arr


def run_gnn_endpoint(
    endpoint: Endpoint,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> dict[str, float]:
    logger.info("--- GNN %s ---", endpoint)
    df = pd.read_parquet(INTERIM / f"{endpoint}.parquet")
    task = ENDPOINT_TASK[endpoint]
    log_transform = endpoint in LOG_TRANSFORM

    graphs, smiles = load_or_build_dataset(endpoint=endpoint, df=df)
    y = np.array([float(g.y.item()) for g in graphs])

    train_idx, test_idx = scaffold_split(
        smiles=smiles, test_size=0.2, y=y if task == Task.CLASSIFICATION else None
    )
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]

    # Validation split from train only
    train_smiles = [smiles[i] for i in train_idx]
    train_y = y[train_idx]
    inner_train_idx, val_idx = scaffold_split(
        smiles=train_smiles,
        test_size=0.2,
        y=train_y if task == Task.CLASSIFICATION else None,
    )
    actual_train = [train_graphs[i] for i in inner_train_idx]
    val_graphs = [train_graphs[i] for i in val_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_loader = DataLoader(actual_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    model = MolSchNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )
    criterion = nn.MSELoss() if task == Task.REGRESSION else nn.BCEWithLogitsLoss()

    best_val = float("inf") if task == Task.REGRESSION else -float("inf")
    best_epoch = 0
    best_state: dict = {}
    no_improve = 0
    start_epoch = 1

    ckpt_path = MODELS_GNN / f"{endpoint}_train_ckpt.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_val = ckpt["best_val"]
        best_epoch = ckpt["best_epoch"]
        best_state = ckpt["best_state"]
        no_improve = ckpt["no_improve"]
        start_epoch = ckpt["epoch"] + 1
        logger.info(
            "Resuming GNN training from epoch %d (best val: %.4f @ epoch %d)",
            start_epoch, best_val, best_epoch,
        )

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(z=batch.z, pos=batch.pos, batch=batch.batch).squeeze(-1)
            if task == Task.REGRESSION and log_transform:
                target = torch.log1p(batch.y.squeeze())
            else:
                target = batch.y.squeeze()
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_preds, val_true = _eval(
            model=model, loader=val_loader, device=device,
            task=task, log_transform=log_transform,
        )
        if task == Task.REGRESSION:
            val_metric = float(mean_squared_error(val_true, val_preds) ** 0.5)
            improved = val_metric < best_val
        else:
            val_metric = float(roc_auc_score(val_true, val_preds))
            improved = val_metric > best_val

        if improved:
            best_val = val_metric
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve == 0:
            metric_name = "RMSE" if task == Task.REGRESSION else "ROC-AUC"
            logger.info(
                "Epoch %3d  val %s: %.4f  (best %.4f @ %d)",
                epoch, metric_name, val_metric, best_val, best_epoch,
            )

        if epoch % TRAIN_CHECKPOINT_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val": best_val,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "no_improve": no_improve,
            }, ckpt_path)

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    if ckpt_path.exists():
        ckpt_path.unlink()

    model.load_state_dict(best_state)
    model_path = MODELS_GNN / f"{endpoint}_gnn.pt"
    torch.save(best_state, model_path)
    logger.info("Saved best model -> %s", model_path.name)

    test_preds, test_true = _eval(
        model=model, loader=test_loader, device=device,
        task=task, log_transform=log_transform,
    )

    if task == Task.REGRESSION:
        rmse = float(mean_squared_error(test_true, test_preds) ** 0.5)
        r2 = float(r2_score(test_true, test_preds))
        logger.info("RMSE: %.4f   R²: %.4f", rmse, r2)
        return {"rmse": rmse, "r2": r2}
    else:
        roc = float(roc_auc_score(test_true, test_preds))
        prc = _auprc(test_true, test_preds)
        logger.info("ROC-AUC: %.4f   AUPRC: %.4f", roc, prc)
        return {"roc_auc": roc, "auprc": prc}


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Train SchNet GNN for ADMET prediction")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument(
        "--epochs", type=int, default=MAX_EPOCHS,
        help=f"Max training epochs per endpoint (default {MAX_EPOCHS}; use 30 for a CPU smoke test)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.warning(
            "No GPU detected — CPU training is very slow for large endpoints. "
            "Consider --epochs 30 for a quick test."
        )

    endpoints = [Endpoint(args.endpoint)] if args.endpoint else list(Endpoint)

    mlflow.set_tracking_uri(f"sqlite:///{MODELS_DIR / 'mlflow.db'}")
    mlflow.set_experiment("admet-gnn")

    run_ts = datetime.now()
    results: dict[str, dict[str, float]] = {}

    with mlflow.start_run(run_name=run_ts.strftime("%Y%m%d_%H%M%S")):
        mlflow.log_params(
            {"hidden": HIDDEN, "batch_size": BATCH_SIZE, "lr": LR, "max_epochs": args.epochs}
        )
        for endpoint in endpoints:
            metrics = run_gnn_endpoint(endpoint=endpoint, max_epochs=args.epochs)
            results[endpoint] = metrics
            for k, v in metrics.items():
                mlflow.log_metric(key=f"{endpoint}.gnn.{k}", value=v)

    report_path = REPORTS_GNN / f"gnn_{run_ts.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(
        json.dumps(
            {"timestamp": run_ts.isoformat(), "epochs": args.epochs, "results": results},
            indent=2,
        )
    )
    logger.info("Report → %s", report_path)

    logger.info("\n=== GNN SUMMARY ===")
    for name, metrics in results.items():
        metric_str = "   ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info("%-40s %s", name, metric_str)


if __name__ == "__main__":
    main()
