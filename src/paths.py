"""Central path registry. Creates data/ subdirectories on import."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
PROC_FEATURES = PROCESSED / "features"
PROC_GRAPHS = PROCESSED / "graphs"

MODELS_DIR = ROOT / "models"
MODELS_LGBM = MODELS_DIR / "lgbm"
MODELS_GNN = MODELS_DIR / "gnn"

REPORTS = ROOT / "reports"
REPORTS_PREDICT = REPORTS / "predict"
REPORTS_GNN = REPORTS / "gnn"
REPORTS_SCREENING = REPORTS / "screening"
REPORTS_AL = REPORTS / "active_learning"
LOGS = ROOT / "logs"
CONFIG = ROOT / "config"

for _dir in (
    RAW, EXTERNAL, INTERIM,
    PROC_FEATURES, PROC_GRAPHS,
    MODELS_LGBM, MODELS_GNN,
    REPORTS_PREDICT, REPORTS_GNN, REPORTS_SCREENING, REPORTS_AL,
    LOGS, CONFIG,
):
    _dir.mkdir(parents=True, exist_ok=True)
