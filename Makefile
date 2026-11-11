# The venv interpreter directly, so targets work whether or not the venv is
# activated. Bare `python` picked up the system Python 3.14 (no rdkit) when make
# ran from an unactivated shell. Override with: make PYTHON=python <target>
PYTHON ?= .venv/Scripts/python.exe
export PYTHONUNBUFFERED := 1

.PHONY: help install install-dev freeze commit push typecheck lint test test-e2e coverage fetch validate fetch-chembl predict predict-merged mlflow tune gnn screen active-learning

help:
	@echo ""
	@echo "Setup:"
	@echo "  install          Install runtime deps + package (editable)"
	@echo "  install-dev      Also install dev tooling (mypy, stubs, ruff)"
	@echo "  freeze           Snapshot environment to requirements-lock.txt"
	@echo ""
	@echo "Quality:"
	@echo "  push             Run commit checks, then git push"
	@echo "  commit           Run lint + typecheck + test; echoes 'All checks passed'"
	@echo "  typecheck        Run mypy over src/"
	@echo "  lint             Run ruff linter"
	@echo "  test             Run the fast test suite (excludes e2e)"
	@echo "  coverage         Run tests with line coverage report (src/ only)"
	@echo "  test-e2e         Real-browser smoke tests (needs playwright)"
	@echo ""
	@echo "Data:"
	@echo "  fetch            Download all ADMET datasets from TDC"
	@echo "  validate         Re-validate all data files on disk"
	@echo "  fetch-chembl     Fetch ChEMBL bioactivity data; saves *_merged.parquet alongside TDC"
	@echo ""
	@echo "Model:"
	@echo "  predict          Train + evaluate all endpoints (TDC data); saves models for screening"
	@echo "  predict-merged   Same but CV uses TDC + ChEMBL merged data (requires fetch-chembl)"
	@echo "  tune             Hyperparameter search with Optuna (saves to config/)"
	@echo "  gnn              Train SchNet GNN with 3D geometry (GPU recommended)"
	@echo "  screen           Virtual screening against ZINC20 lead-like (requires make predict first)"
	@echo "  active-learning  AL simulation: ucb/explore/random strategies (ENDPOINT= MODEL= N_ITER= BATCH= KAPPA= STRATEGY=)"
	@echo "  mlflow           Open MLflow experiment tracking UI"
	@echo ""

install:
	$(PYTHON) -m pip install --no-deps PyTDC
	$(PYTHON) -m pip install -r requirements.txt -e .

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

freeze:
	$(PYTHON) -m pip freeze > requirements-lock.txt

commit: lint typecheck test

push: commit
	git push

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src/

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

test-e2e:
	$(PYTHON) -m pytest -m e2e

fetch:
	$(PYTHON) -m src.data.fetch_data

validate:
	$(PYTHON) -m src.data.validate_data

fetch-chembl:
	$(PYTHON) -m src.data.fetch_chembl_bioactivity

predict:
	$(PYTHON) -m src.models.predict

predict-merged:
	$(PYTHON) -m src.models.predict --source merged

tune:
	$(PYTHON) -m src.models.tune

gnn:
	$(PYTHON) -m src.models.gnn

screen:
	$(PYTHON) -m src.screening.screen $(if $(N),--n-compounds $(N))

active-learning:
	$(PYTHON) -m src.active_learning.simulate \
		$(if $(ENDPOINT),--endpoint $(ENDPOINT)) \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(N_ITER),--n-iter $(N_ITER)) \
		$(if $(BATCH),--batch-size $(BATCH)) \
		$(if $(KAPPA),--kappa $(KAPPA)) \
		$(if $(STRATEGY),--strategy $(STRATEGY))

mlflow:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///models/mlflow.db
