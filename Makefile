.PHONY: help install dev test test-q lint fmt typecheck ci api up down logs bronze silver gold documents documents-smoke embeddings embeddings-smoke search ollama-pull otel-up otel-down otel-logs clean

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

documents: ## Run document builder job
	docker compose run --rm document-builder

documents-smoke: ## Run document builder with a small row limit
	docker compose run --rm -e DOCUMENT_MAX_ROWS=$${DOCUMENT_MAX_ROWS:-25} -e DOCUMENT_RUN_ID=$${DOCUMENT_RUN_ID:-documents-smoke} document-builder

embeddings: ## Run embedding ingestor job
	docker compose run --rm embedding-ingestor

embeddings-smoke: ## Run embedding ingestor with a small document limit
	docker compose run --rm -e DOCUMENT_SOURCE_RUN_ID=$${DOCUMENT_SOURCE_RUN_ID:-documents-smoke} -e EMBEDDING_MAX_DOCUMENTS=$${EMBEDDING_MAX_DOCUMENTS:-25} -e EMBEDDING_RUN_ID=$${EMBEDDING_RUN_ID:-embeddings-smoke} embedding-ingestor

search: ## Run semantic search against pgvector
	docker compose run --rm semantic-searcher --query "$(QUERY)" --embedding-run-id "$(EMBEDDING_RUN_ID)" $(SEARCH_ARGS)

otel-up: ## Start observability stack (OTEL Collector + Jaeger + Prometheus + Grafana)
	docker compose --profile observability up -d
	@echo ""
	@echo "Observability UIs:"
	@echo "  Jaeger:     http://localhost:16686"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000"
	@echo ""
	@echo "Set OTEL_ENABLED=true in .env and restart rag-api to send traces."

otel-down: ## Stop observability stack
	docker compose --profile observability down

otel-logs: ## Follow observability logs
	docker compose --profile observability logs -f otel-collector jaeger prometheus grafana

ollama-pull: ## Pull Ollama models (inference + embedding)
	docker exec -it $$(docker compose ps -q ollama) ollama pull qwen2.5:7b-instruct
	docker exec -it $$(docker compose ps -q ollama) ollama pull nomic-embed-text

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
