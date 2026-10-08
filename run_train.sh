#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -n "${INIT_SH:-}" ]]; then
    [[ -f "$INIT_SH" ]] || { echo "INIT_SH does not exist" >&2; exit 1; }
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    source "$INIT_SH"
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
    command -v conda >/dev/null 2>&1 || { echo "conda is unavailable" >&2; exit 1; }
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/cataopd_training_${RUN_ID}}"
mkdir -p "$LOG_DIR"

if [[ "$#" -gt 0 ]]; then
    SCRIPTS=("$@")
else
    SCRIPTS=(
        "train_qwen3-1.7b.sh"
        "train_qwen2.5-3b.sh"
        "train_qwen2.5-7b.sh"
    )
fi

for script in "${SCRIPTS[@]}"; do
    [[ -f "$script" ]] || { echo "Missing script: $script" >&2; exit 1; }
    name="${script%.sh}"
    log_file="$LOG_DIR/${name}.log"
    set +e
    bash "$script" 2>&1 | tee "$log_file"
    status="${PIPESTATUS[0]}"
    set -e
    [[ "$status" -eq 0 ]] || exit "$status"
done
