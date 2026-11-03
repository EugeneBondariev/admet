"""Central path registry. Creates data/ subdirectories on import."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"

for _dir in (RAW, EXTERNAL, INTERIM, PROCESSED, MODELS_DIR, REPORTS, LOGS):
    _dir.mkdir(parents=True, exist_ok=True)
