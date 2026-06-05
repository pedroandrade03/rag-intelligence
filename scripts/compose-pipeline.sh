#!/usr/bin/env bash
# End-to-end data + RAG docs pipeline — runs entirely inside Docker Compose.
set -euo pipefail

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

step() {
  echo ""
  echo "=== $1 ==="
}

step "Pipeline RUN_ID=${RUN_ID}"

export BRONZE_RUN_ID="${BRONZE_RUN_ID:-$RUN_ID}"
step "Bronze import"
bronze-import

export BRONZE_SOURCE_RUN_ID="${BRONZE_SOURCE_RUN_ID:-$RUN_ID}"
export SILVER_RUN_ID="${SILVER_RUN_ID:-$RUN_ID}"
step "Silver transform"
silver-transform

export SILVER_SOURCE_RUN_ID="${SILVER_SOURCE_RUN_ID:-$RUN_ID}"
export GOLD_RUN_ID="${GOLD_RUN_ID:-$RUN_ID}"
export GOLD_SOURCE_RUN_ID="${GOLD_SOURCE_RUN_ID:-$RUN_ID}"
step "Gold transform"
gold-transform

export GOLD_SOURCE_RUN_ID="${GOLD_SOURCE_RUN_ID:-$RUN_ID}"
export TRAIN_RUN_ID="${TRAIN_RUN_ID:-$RUN_ID}"
step "Train logistic regression"
train-logreg

step "Train HistGBT"
train-histgbt

step "Train baseline"
train-baseline

step "Embed pipeline documentation"
embed-docs

echo ""
echo "=== Pipeline complete: RUN_ID=${RUN_ID} ==="
echo "Frontend:  http://localhost:3002"
echo "API docs:  http://localhost:8000/docs"
echo "MLflow:    http://localhost:5000"
echo "MinIO:     http://localhost:9001"
