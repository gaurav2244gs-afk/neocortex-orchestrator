#!/usr/bin/env bash
# Spin up Redis + the API for local development.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! docker ps --format '{{.Names}}' | grep -q neocortex-redis; then
  docker run -d --rm --name neocortex-redis -p 6379:6379 redis:7-alpine
  echo "Started Redis (container: neocortex-redis)"
fi

export PYTHONPATH="$(pwd)/src"
uvicorn neocortex.api.main:app --reload --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
