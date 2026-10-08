#!/usr/bin/env bash
set -Eeuo pipefail
export MODEL_SUBDIR="Qwen2.5-3B-Instruct"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-CataOPD-qwen2.5-3b}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_model.sh" "$@"
