#!/usr/bin/env bash
# Resume pipeline from embed-docs (silver/gold/train already done)
set -euo pipefail

echo "=== Embed docs ==="
set -a; [ -f ./.env ] && . ./.env || true; set +a
env OLLAMA_BASE_URL=http://127.0.0.1:11434 PG_HOST=localhost PG_PORT=54330 \
  ./.venv/bin/embed-docs

echo "=== Done! ==="
echo "Frontend: http://localhost:3002"
echo "API docs: http://localhost:8000/docs"
echo "MLflow:   http://localhost:5000"
