.PHONY: help install dev test test-q lint fmt typecheck ci api up down logs bronze silver gold ollama-pull clean

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Development setup
install: ## Create venv and install dependencies
	uv venv --python 3.13 .venv
	. .venv/bin/activate && uv pip install -e .

dev: ## Create venv and install with dev dependencies
	uv venv --python 3.13 .venv
	. .venv/bin/activate && uv pip install -e ".[dev]"

test: ## Run tests (verbose)
	. .venv/bin/activate && python -m pytest tests/ -v

test-q: ## Run tests (quiet)
	. .venv/bin/activate && python -m pytest tests/ -q

lint: ## Run ruff linter
	ruff check src/ tests/

fmt: ## Run ruff formatter (writes changes)
	ruff format src/ tests/

typecheck: ## Run pyright type checker
	pyright --pythonpath .venv/bin/python src/

ci: lint typecheck test ## Run full CI check locally (lint + types + tests)

api: ## Run API locally with hot reload
	. .venv/bin/activate && uvicorn rag_intelligence.api.main:app --reload --port 8000

up: ## Start all services (docker compose)
	docker compose up -d

down: ## Stop all services
	docker compose down

logs: ## Follow service logs
	docker compose logs -f

bronze: ## Run bronze importer job
	docker compose run --rm bronze-importer

silver: ## Run silver transformer job
	docker compose run --rm silver-transformer

gold: ## Run gold transformer job
	docker compose run --rm gold-transformer

ollama-pull: ## Pull Ollama models (inference + embedding)
	docker exec -it $$(docker compose ps -q ollama) ollama pull qwen2.5:7b-instruct
	docker exec -it $$(docker compose ps -q ollama) ollama pull nomic-embed-text

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
