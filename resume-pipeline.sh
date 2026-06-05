#!/usr/bin/env bash
# Resume pipeline from embed-docs (silver/gold/train already done)
set -euo pipefail

echo "=== Embed docs (Docker) ==="
docker compose --profile jobs run --rm doc-embedder

echo "=== Done! ==="
echo "Frontend: http://localhost:3002"
echo "API docs: http://localhost:8000/docs"
echo "MLflow:   http://localhost:5000"
