# ============================================================
# Drowning Detection System — Makefile
# ============================================================

.PHONY: help install install-dev lint format test test-cov clean docker-up docker-down api data-pipeline

PYTHON := python
PIP := pip

# ── Help ───────────────────────────────────────────────────
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Installation ──────────────────────────────────────────
install: ## Install production dependencies
	$(PIP) install -r requirements/base.txt
	$(PIP) install -e .

install-dev: ## Install all dependencies (dev + prod)
	$(PIP) install -r requirements/dev.txt
	$(PIP) install -e ".[dev]"
	pre-commit install

# ── Code Quality ──────────────────────────────────────────
lint: ## Run all linters
	black --check --line-length 100 drowning_detector/
	isort --check-only --profile black --line-length 100 drowning_detector/
	ruff check drowning_detector/
	mypy drowning_detector/ --ignore-missing-imports

format: ## Auto-format code
	black --line-length 100 drowning_detector/
	isort --profile black --line-length 100 drowning_detector/
	ruff check --fix drowning_detector/

# ── Testing ───────────────────────────────────────────────
test: ## Run unit tests
	pytest drowning_detector/tests/ -v -m "not slow and not integration and not gpu"

test-cov: ## Run tests with coverage report
	pytest drowning_detector/tests/ -v --cov=drowning_detector --cov-report=html --cov-report=term-missing

test-all: ## Run all tests including integration
	pytest drowning_detector/tests/ -v

# ── Data Pipeline ─────────────────────────────────────────
data-pipeline: ## Run full data pipeline (clip → pose → verify)
	$(PYTHON) drowning_detector/scripts/extract_poses.py
	$(PYTHON) drowning_detector/scripts/build_annotations.py
	$(PYTHON) drowning_detector/scripts/verify_dataset.py

# ── Training ──────────────────────────────────────────────
train: ## Train the LSTM classifier
	$(PYTHON) drowning_detector/models/classifier/train.py --epochs 50 --batch 32

tensorboard: ## Launch TensorBoard
	tensorboard --logdir runs/ --port 6006

# ── API ───────────────────────────────────────────────────
api: ## Run FastAPI dev server
	uvicorn drowning_detector.api.main:app --reload --port 8000

# ── Docker ────────────────────────────────────────────────
docker-up: ## Start all services with Docker Compose
	docker-compose -f drowning_detector/docker/docker-compose.yml up --build -d

docker-down: ## Stop all Docker services
	docker-compose -f drowning_detector/docker/docker-compose.yml down

docker-logs: ## Tail Docker logs
	docker-compose -f drowning_detector/docker/docker-compose.yml logs -f

# ── Cleanup ───────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info coverage.xml
