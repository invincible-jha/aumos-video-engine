.PHONY: install dev lint type-check test test-cov run run-gpu docker-build docker-run migrate clean help

PYTHON := python3.11
PIP := pip
SERVICE_NAME := aumos-video-engine
DOCKER_IMAGE := aumos/video-engine:latest
PORT := 8000

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	$(PIP) install -e .

dev:  ## Install development dependencies
	$(PIP) install -e ".[dev]"

lint:  ## Run ruff linter and formatter check
	ruff check src/ tests/
	ruff format --check src/ tests/

lint-fix:  ## Auto-fix linting issues
	ruff check --fix src/ tests/
	ruff format src/ tests/

type-check:  ## Run mypy strict type checking
	mypy src/

test:  ## Run tests (no coverage)
	pytest tests/ -v

test-cov:  ## Run tests with 80% coverage requirement
	pytest tests/ -v --cov=aumos_video_engine --cov-report=term-missing --cov-fail-under=80

test-unit:  ## Run unit tests only (no containers)
	pytest tests/ -v -m "not integration"

test-integration:  ## Run integration tests (requires Docker)
	pytest tests/ -v -m "integration"

run:  ## Run development server
	uvicorn aumos_video_engine.main:app --host 0.0.0.0 --port $(PORT) --reload

run-gpu:  ## Run with GPU enabled
	GPU_ENABLED=true uvicorn aumos_video_engine.main:app --host 0.0.0.0 --port $(PORT) --workers 1

docker-build:  ## Build Docker image (CPU)
	docker build --target runtime -t $(DOCKER_IMAGE) .

docker-build-gpu:  ## Build Docker image (GPU)
	docker build --target gpu-runtime -t $(DOCKER_IMAGE)-gpu .

docker-run:  ## Run Docker container
	docker run -p $(PORT):$(PORT) --env-file .env $(DOCKER_IMAGE)

docker-compose-up:  ## Start dev stack with docker-compose
	docker-compose -f docker-compose.dev.yml up -d

docker-compose-down:  ## Stop dev stack
	docker-compose -f docker-compose.dev.yml down

migrate:  ## Run Alembic migrations
	alembic upgrade head

migrate-new:  ## Create new migration (usage: make migrate-new MSG="add video jobs")
	alembic revision --autogenerate -m "$(MSG)"

clean:  ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/

all: lint type-check test-cov  ## Run full CI pipeline locally
