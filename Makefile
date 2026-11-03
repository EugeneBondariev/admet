PYTHON = python
export PYTHONUNBUFFERED := 1

.PHONY: help install install-dev freeze push typecheck lint test test-e2e fetch validate predict

help:
	@echo ""
	@echo "Setup:"
	@echo "  install          Install runtime deps + package (editable)"
	@echo "  install-dev      Also install dev tooling (mypy, stubs, ruff)"
	@echo "  freeze           Snapshot environment to requirements-lock.txt"
	@echo ""
	@echo "Quality:"
	@echo "  push             Run typecheck + test, then git push"
	@echo "  typecheck        Run mypy over src/"
	@echo "  lint             Run ruff linter"
	@echo "  test             Run the fast test suite (excludes e2e)"
	@echo "  test-e2e         Real-browser smoke tests (needs playwright)"
	@echo ""
	@echo "Data:"
	@echo "  fetch            Download all ADMET datasets from TDC"
	@echo "  validate         Re-validate all data files on disk"
	@echo ""
	@echo "Model:"
	@echo "  predict          Train + evaluate all endpoints"
	@echo ""

install:
	$(PYTHON) -m pip install --no-deps PyTDC
	$(PYTHON) -m pip install -r requirements.txt -e .

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

freeze:
	$(PYTHON) -m pip freeze > requirements-lock.txt

push: typecheck test
	git push

typecheck:
	$(PYTHON) -m mypy src/

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

test-e2e:
	$(PYTHON) -m pytest -m e2e

fetch:
	$(PYTHON) -m src.data.fetch_data

validate:
	$(PYTHON) -m src.data.validate_data

predict:
	$(PYTHON) -m src.models.predict
