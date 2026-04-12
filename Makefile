.PHONY: help setup install dev test test-q lint fmt typecheck ci ci-frontend ci-all api frontend frontend-build start up up-host-ollama up-local-ollama chat-local-ready pipeline-local-ollama demo-local-ollama down down-local-ollama down-all logs bronze silver gold documents documents-smoke embeddings embeddings-smoke search mlflow-up train-logreg train-histgbt train-baseline embed-docs ollama-pull otel-up otel-down otel-logs clean

LOCAL_RUN_DIR := /tmp/rag-intelligence
RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Development setup
setup: ## Full first-time setup (backend + frontend + env + models)
	@echo "=== Checking prerequisites ==="
	@command -v uv >/dev/null || { echo "ERROR: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	@command -v node >/dev/null || { echo "ERROR: node not found. Install Node.js 20+"; exit 1; }
	@command -v docker >/dev/null || { echo "ERROR: docker not found."; exit 1; }
	@command -v ollama >/dev/null || { echo "ERROR: ollama not found. Install: https://ollama.com"; exit 1; }
	@echo "=== Installing backend (dev + ML) ==="
	uv venv --python 3.13 .venv
	. .venv/bin/activate && uv pip install -e ".[dev,ml]"
	@echo "=== Installing frontend ==="
	cd frontend && npm install
	@if [ ! -f .env ]; then cp .env.example .env; echo "=== Created .env from .env.example — review it ==="; else echo "=== .env already exists ==="; fi
	@echo "=== Pulling Ollama models ==="
	ollama pull qwen2.5:7b-instruct-q4_K_M
	ollama pull nomic-embed-text
	@echo ""
	@echo "Setup complete. Run 'make start' to launch."

install: ## Create venv and install dependencies
	uv venv --python 3.13 .venv
	. .venv/bin/activate && uv pip install -e .

dev: ## Create venv and install with dev + ML dependencies
	uv venv --python 3.13 .venv
	. .venv/bin/activate && uv pip install -e ".[dev,ml]"

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

ci: lint typecheck test ## Run backend CI locally (lint + types + tests)

ci-frontend: ## Run frontend CI locally (lint + build)
	cd frontend && npm run lint && npm run build

ci-all: ci ci-frontend ## Run full CI locally (backend + frontend)

api: ## Run API locally with hot reload
	set -a && . ./.env && set +a && . .venv/bin/activate && uvicorn rag_intelligence.api.main:app --reload --port 8000

frontend: ## Run frontend dev server
	cd frontend && npm run dev

frontend-build: ## Build frontend for production
	cd frontend && npm run build

start: up-local-ollama ## Start local dev stack (infra in Docker, API/frontend locally)

up: ## Start all services (docker compose)
	docker compose up -d

up-host-ollama: ## Start app stack using host Ollama on 127.0.0.1:11434
	COMPOSE_OLLAMA_BASE_URL=http://host.docker.internal:11434 docker compose --profile mlflow up -d --build minio timescaledb mlflow rag-api frontend

up-local-ollama: ## Start infra in Docker and run API/frontend locally against host Ollama
	docker compose --profile mlflow up -d minio timescaledb mlflow
	-docker compose stop rag-api frontend
	mkdir -p $(LOCAL_RUN_DIR)
	zsh -lc 'if [ -f "$(LOCAL_RUN_DIR)/api.pid" ] && kill -0 "$$(cat "$(LOCAL_RUN_DIR)/api.pid")" 2>/dev/null; then echo "API already running"; else set -a; [ -f ./.env ] && . ./.env || true; set +a; nohup env OLLAMA_BASE_URL=http://127.0.0.1:11434 ./.venv/bin/uvicorn rag_intelligence.api.main:app --host 0.0.0.0 --port 8000 >"$(LOCAL_RUN_DIR)/api.log" 2>&1 & echo $$! >"$(LOCAL_RUN_DIR)/api.pid"; fi'
	zsh -lc 'if [ -f "$(LOCAL_RUN_DIR)/frontend.pid" ] && kill -0 "$$(cat "$(LOCAL_RUN_DIR)/frontend.pid")" 2>/dev/null; then echo "Frontend already running"; else set -a; [ -f ./.env ] && . ./.env || true; set +a; cd frontend; nohup env PORT=3002 RAG_API_URL=http://localhost:8000 OLLAMA_BASE_URL=http://127.0.0.1:11434 npm run dev >"$(LOCAL_RUN_DIR)/frontend.log" 2>&1 & echo $$! >"$(LOCAL_RUN_DIR)/frontend.pid"; fi'
	@echo "Frontend: http://localhost:3002"
	@echo "API docs: http://localhost:8000/docs"
	@echo "MLflow: http://localhost:5000"
	@echo "MinIO: http://localhost:9001"

chat-local-ready: ## Start local chat stack, embed docs, ensure lexical metadata, run smoke checks
	./scripts/chat-local-ready.sh

pipeline-local-ollama: ## Run full pipeline using Docker jobs plus local embed-docs against host Ollama
	BRONZE_RUN_ID=$(RUN_ID) docker compose run --rm bronze-importer
	BRONZE_SOURCE_RUN_ID=$(RUN_ID) SILVER_RUN_ID=$(RUN_ID) docker compose run --rm silver-transformer
	SILVER_SOURCE_RUN_ID=$(RUN_ID) GOLD_RUN_ID=$(RUN_ID) GOLD_SOURCE_RUN_ID=$(RUN_ID) docker compose run --rm gold-transformer
	GOLD_SOURCE_RUN_ID=$(RUN_ID) TRAIN_RUN_ID=$(RUN_ID) docker compose --profile mlflow --profile jobs run --rm train-logreg
	GOLD_SOURCE_RUN_ID=$(RUN_ID) TRAIN_RUN_ID=$(RUN_ID) docker compose --profile mlflow --profile jobs run --rm train-histgbt
	GOLD_SOURCE_RUN_ID=$(RUN_ID) TRAIN_RUN_ID=$(RUN_ID) docker compose --profile mlflow --profile jobs run --rm train-baseline
	zsh -lc 'set -a; [ -f ./.env ] && . ./.env || true; set +a; env OLLAMA_BASE_URL=http://127.0.0.1:11434 PG_HOST=localhost PG_PORT=54330 ./.venv/bin/embed-docs'
	@echo "Pipeline completed with RUN_ID=$(RUN_ID)"

demo-local-ollama: ## Start local stack and run the full pipeline against host Ollama
	$(MAKE) up-local-ollama
	$(MAKE) pipeline-local-ollama RUN_ID=$(RUN_ID)

down: ## Stop all Docker services (all profiles)
	docker compose --profile mlflow --profile observability --profile jobs down

down-local-ollama: ## Stop local API/frontend processes started by up-local-ollama
	zsh -lc 'if [ -f "$(LOCAL_RUN_DIR)/api.pid" ]; then kill "$$(cat "$(LOCAL_RUN_DIR)/api.pid")" 2>/dev/null || true; rm -f "$(LOCAL_RUN_DIR)/api.pid"; fi; pids="$$(lsof -tiTCP:8000 -sTCP:LISTEN 2>/dev/null || true)"; [ -n "$$pids" ] && kill $$pids 2>/dev/null || true'
	zsh -lc 'if [ -f "$(LOCAL_RUN_DIR)/frontend.pid" ]; then kill "$$(cat "$(LOCAL_RUN_DIR)/frontend.pid")" 2>/dev/null || true; rm -f "$(LOCAL_RUN_DIR)/frontend.pid"; fi; pids="$$(lsof -tiTCP:3002 -sTCP:LISTEN 2>/dev/null || true)"; [ -n "$$pids" ] && kill $$pids 2>/dev/null || true'

down-all: down-local-ollama down ## Stop local API/frontend processes and Docker services

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

mlflow-up: ## Start MLflow service
	docker compose --profile mlflow up -d mlflow

train-logreg: mlflow-up ## Train logistic regression model
	docker compose --profile mlflow --profile jobs run --rm train-logreg

train-histgbt: mlflow-up ## Train histogram gradient boosting model
	docker compose --profile mlflow --profile jobs run --rm train-histgbt

train-baseline: mlflow-up ## Train baseline model
	docker compose --profile mlflow --profile jobs run --rm train-baseline

embed-docs: ## Embed pipeline documentation into pgvector
	docker compose --profile jobs run --rm doc-embedder

ollama-pull: ## Pull Ollama models (inference + embedding)
	docker exec -it $$(docker compose ps -q ollama) ollama pull qwen2.5:7b-instruct
	docker exec -it $$(docker compose ps -q ollama) ollama pull nomic-embed-text

db-reset: ## Reset frontend chat database
	rm -f frontend/data/chat.db frontend/data/chat.db-wal frontend/data/chat.db-shm
	@echo "Chat database reset."

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
