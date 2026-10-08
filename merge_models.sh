#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

: "${MODELS_DIR:?Set MODELS_DIR}"

STEP="${STEP:-80}"
FORCE="${FORCE:-0}"
MODELS=(CataOPD-qwen3-1.7b CataOPD-qwen2.5-3b CataOPD-qwen2.5-7b)

if [[ "$#" -eq 0 ]]; then
    EXPS=("${MODELS[@]}")
else
    EXPS=("$@")
fi

base_for() {
    case "$1" in
        CataOPD-qwen3-1.7b) echo "$MODELS_DIR/Qwen3-1.7B" ;;
        CataOPD-qwen2.5-3b) echo "$MODELS_DIR/Qwen2.5-3B-Instruct" ;;
        CataOPD-qwen2.5-7b) echo "$MODELS_DIR/Qwen2.5-7B-Instruct" ;;
        *) return 1 ;;
    esac
}

FAIL=0
for EXP in "${EXPS[@]}"; do
    CKPT="$PROJECT_DIR/checkpoints/CataOPD/$EXP/global_step_${STEP}/actor"
    TARGET="$PROJECT_DIR/merge_model/${EXP}_step${STEP}"
    BASE="$(base_for "$EXP")" || { echo "Unsupported model: $EXP" >&2; FAIL=1; continue; }
    if [[ "$FORCE" != 1 && -f "$TARGET/config.json" ]] \
       && find "$TARGET" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q .; then
        continue
    fi
    [[ -d "$CKPT" ]] || { echo "Missing checkpoint: $CKPT" >&2; FAIL=1; continue; }
    python3 verl/scripts/model_merger.py \
        --backend fsdp \
        --hf_model_path "$BASE" \
        --local_dir "$CKPT" \
        --target_dir "$TARGET"
done

[[ "$FAIL" -eq 0 ]]
