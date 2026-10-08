#!/usr/bin/env bash
set -Eeuo pipefail
export MODEL_SUBDIR="Qwen3-1.7B"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-CataOPD-qwen3-1.7b}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_model.sh" "$@"
