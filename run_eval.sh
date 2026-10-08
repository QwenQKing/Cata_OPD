#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

INIT_SH="${INIT_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"
MERGE_ROOT="${MERGE_ROOT:-merge_model}"
INPUT_DIR="dataset/eval"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval-results}"

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
TP_SIZE="${TP_SIZE:-${#GPU_IDS[@]}}"
VLLM_HOST="${VLLM_HOST:-localhost}"
VLLM_PORT="${VLLM_PORT:-}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-agent}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_SEED="${VLLM_SEED:-42}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-120}"

export EVAL_TEMP="${EVAL_TEMP:-0.7}"
export EVAL_BATCH="${EVAL_BATCH:-64}"
export EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-64}"
export EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-8192}"
export PASS_K="${PASS_K:-}"
export EVAL_LIMIT="0"
REQUIRE_TRAIN_PROMPT="${REQUIRE_TRAIN_PROMPT:-1}"

EXPECTED_DATASETS=12
EXPECTED_ROWS_PER_DATASET=128
EXPECTED_TOTAL=$((EXPECTED_DATASETS * EXPECTED_ROWS_PER_DATASET))

MODEL_NAMES=(
    "CataOPD-qwen3-1.7b_step80"
    "CataOPD-qwen2.5-3b_step80"
    "CataOPD-qwen2.5-7b_step80"
)

MODEL_PATHS=(
    "$MERGE_ROOT/CataOPD-qwen3-1.7b_step80"
    "$MERGE_ROOT/CataOPD-qwen2.5-3b_step80"
    "$MERGE_ROOT/CataOPD-qwen2.5-7b_step80"
)

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/run_all.log") 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
    log "ERROR: $*"
    exit 1
}

if [[ -n "$INIT_SH" ]]; then
    [[ -f "$INIT_SH" ]] || die "INIT_SH does not exist"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    source "$INIT_SH"
fi
if [[ -n "$CONDA_ENV" ]]; then
    command -v conda >/dev/null 2>&1 || die "conda is unavailable"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

for command_name in python3 vllm curl nvidia-smi; do
    command -v "$command_name" >/dev/null 2>&1 || die "missing command: $command_name"
done

if [[ -z "$VLLM_PORT" ]]; then
    VLLM_PORT="$(python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"
fi
export VLLM_API_BASE="http://${VLLM_HOST}:${VLLM_PORT}/v1"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_MODEL_NAME

validate_input() {
    python3 - "$INPUT_DIR" "$EXPECTED_DATASETS" "$EXPECTED_ROWS_PER_DATASET" "$REQUIRE_TRAIN_PROMPT" <<'PY'
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
expected_datasets = int(sys.argv[2])
expected_rows = int(sys.argv[3])
require_train_prompt = sys.argv[4] == "1"
required = {"id", "data_source", "prompt", "reward_model", "extra_info"}
if require_train_prompt:
    from casd.prompts.student import STUDENT_INSTRUCTION
files = sorted(root.glob("*.parquet"))
if len(files) != expected_datasets:
    raise SystemExit(f"expected {expected_datasets} parquet files in {root}, got {len(files)}")
for path in files:
    df = pd.read_parquet(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    if len(df) != expected_rows:
        raise SystemExit(f"{path}: expected {expected_rows} rows, got {len(df)}")
    if df["id"].isnull().any() or not df["id"].is_unique:
        raise SystemExit(f"{path}: id must be non-null and unique")
    if require_train_prompt:
        for index, row in df.iterrows():
            extra_info = row["extra_info"]
            question = str(extra_info.get("question", "")).strip() if hasattr(extra_info, "get") else ""
            prompt = row["prompt"]
            expected_prompt = STUDENT_INSTRUCTION + "\n\nQuestion: " + question
            if (not question or len(prompt) != 1 or prompt[0].get("role") != "user"
                    or prompt[0].get("content") != expected_prompt):
                raise SystemExit(f"{path}: row {index} prompt does not exactly match the training prompt")
mode = "training" if require_train_prompt else "unchecked"
print(f"INPUT_OK datasets={len(files)} rows={sum(pd.read_parquet(p, columns=['id']).shape[0] for p in files)} prompt={mode}")
PY
}

validate_model_paths() {
    local index model_path
    for index in "${!MODEL_PATHS[@]}"; do
        model_requested "${MODEL_NAMES[$index]}" || continue
        model_path="${MODEL_PATHS[$index]}"
        [[ -f "$model_path/config.json" ]] || die "missing model config: $model_path/config.json"
        [[ -n "$(find "$model_path" -maxdepth 1 -type f -name '*.safetensors' -print -quit)" ]] \
            || die "missing safetensors weights: $model_path"
    done
}

port_is_busy() {
    python3 - "$VLLM_HOST" "$VLLM_PORT" <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(0.5)
busy = sock.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0
sock.close()
raise SystemExit(0 if busy else 1)
PY
}

selected_gpu_pids() {
    local gpu output
    IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
    for gpu in "${gpu_ids[@]}"; do
        output="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
        awk -v gpu="$gpu" '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, "", $0); print gpu ":" $0}' <<< "$output"
    done
}

ensure_resources_free() {
    if port_is_busy; then
        die "port $VLLM_PORT is already in use; refusing to stop an unknown service"
    fi
    if [[ "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
        local busy_pids
        busy_pids="$(selected_gpu_pids)"
        if [[ -n "$busy_pids" ]]; then
            die "selected GPUs already have compute processes ($busy_pids); set ALLOW_BUSY_GPUS=1 only if intentional"
        fi
    fi
}

result_is_complete() {
    local result_dir="$1"
    python3 - "$INPUT_DIR" "$result_dir" "$EXPECTED_DATASETS" "$EXPECTED_ROWS_PER_DATASET" \
        "$EXPECTED_TOTAL" "$EVAL_TEMP" "$EVAL_MAX_TOKENS" "$VLLM_SEED" "$PASS_K" <<'PY'
import json
import sys
from pathlib import Path

input_dir = Path(sys.argv[1])
result_dir = Path(sys.argv[2])
expected_datasets = int(sys.argv[3])
expected_rows = int(sys.argv[4])
expected_total = int(sys.argv[5])
expected_temperature = float(sys.argv[6])
expected_max_tokens = int(sys.argv[7])
expected_seed = int(sys.argv[8])
summary_path = result_dir / "summary.json"
if not summary_path.is_file():
    raise SystemExit(1)
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if summary.get("n_datasets") != expected_datasets:
    raise SystemExit(1)
if summary.get("average", {}).get("total_samples") != expected_total:
    raise SystemExit(1)
try:
    result_temperature = float(summary.get("temperature"))
except (TypeError, ValueError):
    raise SystemExit(1)
if abs(result_temperature - expected_temperature) > 1e-9:
    raise SystemExit(1)
try:
    result_max_tokens = int(summary.get("max_tokens"))
except (TypeError, ValueError):
    raise SystemExit(1)
if result_max_tokens != expected_max_tokens:
    raise SystemExit(1)
try:
    result_seed = int(summary.get("vllm_seed"))
except (TypeError, ValueError):
    raise SystemExit(1)
if result_seed != expected_seed:
    raise SystemExit(1)

expected_pass_k = sys.argv[9] if len(sys.argv) > 9 else ""
ks = [int(x) for x in expected_pass_k.replace(" ", "").split(",") if x]
avg = summary.get("average", {})
if ks:
    if any(f"pass@{k}" not in avg for k in ks):
        raise SystemExit(1)
else:
    if "em" not in avg:
        raise SystemExit(1)

files = sorted(input_dir.glob("*.parquet"))
if len(files) != expected_datasets:
    raise SystemExit(1)
for parquet_path in files:
    result_path = result_dir / parquet_path.stem / "res.json"
    log_path = result_dir / parquet_path.stem / "log.txt"
    if not result_path.is_file() or not log_path.is_file():
        raise SystemExit(1)
    try:
        rows = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        raise SystemExit(1)
    if len(rows) != expected_rows:
        raise SystemExit(1)

    if ks:
        k_max = max(ks)
        for row in rows:
            samples = row.get("samples")
            if not isinstance(samples, list) or len(samples) != k_max:
                raise SystemExit(1)
            if not any(str(s).strip() for s in samples):
                raise SystemExit(1)
            if any(f"pass@{k}" not in row for k in ks):
                raise SystemExit(1)
    else:
        if any(not str(row.get("response", "")).strip() for row in rows):
            raise SystemExit(1)
        if any("em" not in row or "score" not in row for row in rows):
            raise SystemExit(1)
raise SystemExit(0)
PY
}

SERVER_PID=""
SERVER_USES_PROCESS_GROUP=0

stop_server() {
    if [[ -z "$SERVER_PID" ]]; then
        return
    fi

    if kill -0 "$SERVER_PID" 2>/dev/null; then
        log "stopping vLLM pid=$SERVER_PID"
        if [[ "$SERVER_USES_PROCESS_GROUP" == "1" ]]; then
            kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        else
            kill -TERM "$SERVER_PID" 2>/dev/null || true
        fi

        local waited=0
        while kill -0 "$SERVER_PID" 2>/dev/null && (( waited < SHUTDOWN_TIMEOUT )); do
            sleep 2
            waited=$((waited + 2))
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log "vLLM did not stop in ${SHUTDOWN_TIMEOUT}s; sending KILL to its own process group"
            if [[ "$SERVER_USES_PROCESS_GROUP" == "1" ]]; then
                kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
            else
                kill -KILL "$SERVER_PID" 2>/dev/null || true
            fi
        fi
    fi
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    SERVER_USES_PROCESS_GROUP=0

    local waited=0
    while port_is_busy && (( waited < 60 )); do
        sleep 2
        waited=$((waited + 2))
    done
    port_is_busy && die "port $VLLM_PORT is still busy after stopping vLLM"
    sleep 5
}

cleanup() {
    stop_server || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

start_server() {
    local model_path="$1"
    local server_log="$2"
    local vllm_bin
    vllm_bin="$(command -v vllm)"

    log "starting vLLM: $(basename "$model_path")"
    if command -v setsid >/dev/null 2>&1; then
        setsid "$vllm_bin" serve "$model_path" \
            --served-model-name "$VLLM_MODEL_NAME" \
            --host "$VLLM_HOST" \
            --port "$VLLM_PORT" \
            --tensor-parallel-size "$TP_SIZE" \
            --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
            --seed "$VLLM_SEED" \
            >"$server_log" 2>&1 &
        SERVER_USES_PROCESS_GROUP=1
    else
        "$vllm_bin" serve "$model_path" \
            --served-model-name "$VLLM_MODEL_NAME" \
            --host "$VLLM_HOST" \
            --port "$VLLM_PORT" \
            --tensor-parallel-size "$TP_SIZE" \
            --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
            --seed "$VLLM_SEED" \
            >"$server_log" 2>&1 &
        SERVER_USES_PROCESS_GROUP=0
    fi
    SERVER_PID=$!
    log "vLLM pid=$SERVER_PID log=$server_log"
}

wait_for_server() {
    local server_log="$1"
    local waited=0
    while (( waited < STARTUP_TIMEOUT )); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            tail -n 100 "$server_log" || true
            die "vLLM exited before becoming ready"
        fi
        if curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" \
            "$VLLM_API_BASE/models" 2>/dev/null \
            | python3 -c 'import json,os,sys; data=json.load(sys.stdin); expected=os.environ["VLLM_MODEL_NAME"]; raise SystemExit(0 if any(x.get("id")==expected for x in data.get("data", [])) else 1)' \
            >/dev/null 2>&1; then
            log "vLLM is ready after ${waited}s"
            return
        fi
        sleep 5
        waited=$((waited + 5))
        if (( waited % 30 == 0 )); then
            log "waiting for vLLM: ${waited}/${STARTUP_TIMEOUT}s"
        fi
    done
    tail -n 100 "$server_log" || true
    die "vLLM startup timed out after ${STARTUP_TIMEOUT}s"
}

model_requested() {
    local model_name="$1"
    if (( ${#REQUESTED_MODELS[@]} == 0 )); then
        return 0
    fi
    local requested
    for requested in "${REQUESTED_MODELS[@]}"; do
        [[ "$requested" == "$model_name" ]] && return 0
    done
    return 1
}

rewrite_summary_output_dir() {
    local result_dir="$1"
    python3 - "$result_dir/summary.json" "$result_dir" "$VLLM_SEED" \
        "$EVAL_BATCH" "$EVAL_CONCURRENCY" "$EVAL_MAX_TOKENS" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
result_dir = Path(sys.argv[2])
vllm_seed = int(sys.argv[3])
eval_batch = int(sys.argv[4])
eval_concurrency = int(sys.argv[5])
eval_max_tokens = int(sys.argv[6])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["output_dir"] = result_dir.name
summary["vllm_seed"] = vllm_seed
summary["top_p"] = 0.8
summary["max_tokens"] = eval_max_tokens
summary["batch_size"] = eval_batch
summary["concurrency"] = eval_concurrency
summary["prompt_profile"] = "student_training_exact"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

aggregate_results() {
    python3 - "$OUTPUT_ROOT" "${MODEL_NAMES[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
model_names = sys.argv[2:]
combined = {}
rows = []
columns = [
    "model", "scope", "dataset", "n", "em", "f1", "format", "score",
    "avg_in_tokens", "avg_out_tokens", "avg_total_tokens", "avg_latency_s",
    "throughput_tok_s", "acc_per_1k_tok", "tok_per_correct",
]
for model_name in model_names:
    summary_path = root / model_name / "summary.json"
    if not summary_path.is_file():
        continue
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    combined[model_name] = summary
    average = dict(summary.get("average", {}))
    average_row = {"model": model_name, "scope": "average", "dataset": "ALL", "n": average.pop("total_samples", None)}
    average_row.update(average)
    rows.append(average_row)
    for dataset, metrics in sorted(summary.get("per_dataset", {}).items()):
        row = {"model": model_name, "scope": "dataset", "dataset": dataset}
        row.update(metrics)
        rows.append(row)

(root / "all_models_summary.json").write_text(
    json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
)
with (root / "all_models_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"AGGREGATE_OK models={len(combined)} rows={len(rows)}")
PY
}

REQUESTED_MODELS=("$@")
if (( ${#REQUESTED_MODELS[@]} > 0 )); then
    for requested in "${REQUESTED_MODELS[@]}"; do
        found=0
        for model_name in "${MODEL_NAMES[@]}"; do
            [[ "$requested" == "$model_name" ]] && found=1
        done
        (( found == 1 )) || die "unknown model argument: $requested"
    done
fi

log "models=${#MODEL_NAMES[@]} datasets=$EXPECTED_DATASETS rows=$EXPECTED_TOTAL"
log "eval: temp=$EVAL_TEMP batch=$EVAL_BATCH concurrency=$EVAL_CONCURRENCY max_tokens=$EVAL_MAX_TOKENS seed=$VLLM_SEED"

validate_input
validate_model_paths

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
    log "VALIDATION_ONLY_OK datasets=$EXPECTED_DATASETS rows=$EXPECTED_TOTAL prompt_required=$REQUIRE_TRAIN_PROMPT"
    exit 0
fi

for index in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$index]}"
    model_path="${MODEL_PATHS[$index]}"
    model_requested "$model_name" || continue

    final_dir="$OUTPUT_ROOT/$model_name"
    if [[ "${FORCE:-0}" != "1" ]] && result_is_complete "$final_dir"; then
        log "SKIP complete: $model_name"
        continue
    fi

    timestamp="$(date '+%Y%m%d-%H%M%S')"
    if [[ -e "$final_dir" ]]; then
        archived_dir="$OUTPUT_ROOT/${model_name}.incomplete_${timestamp}"
        log "archiving incomplete result: $final_dir -> $archived_dir"
        mv "$final_dir" "$archived_dir"
    fi
    legacy_work_dir="$OUTPUT_ROOT/.${model_name}.running"
    if [[ -e "$legacy_work_dir" ]]; then
        archived_dir="$OUTPUT_ROOT/${model_name}.legacy_stale_${timestamp}"
        log "archiving legacy hidden work directory: $legacy_work_dir -> $archived_dir"
        mv "$legacy_work_dir" "$archived_dir"
    fi
    work_dir="$OUTPUT_ROOT/${model_name}.running"
    if [[ -e "$work_dir" ]]; then
        archived_dir="$OUTPUT_ROOT/${model_name}.stale_${timestamp}"
        log "archiving stale work directory: $work_dir -> $archived_dir"
        mv "$work_dir" "$archived_dir"
    fi
    mkdir -p "$work_dir"

    ensure_resources_free
    start_server "$model_path" "$work_dir/vllm.log"
    wait_for_server "$work_dir/vllm.log"

    log "evaluating $model_name on $EXPECTED_TOTAL samples"
    set +e
    EVAL_INPUT_DIR="$INPUT_DIR" \
    EVAL_OUTPUT_DIR="$work_dir" \
    EVAL_TAG="$model_name" \
    EVAL_MAX_TOKENS="$EVAL_MAX_TOKENS" \
    python3 inference_and_evaluation.py 2>&1 | tee "$work_dir/eval.log"
    eval_status="${PIPESTATUS[0]}"
    set -e

    stop_server

    if [[ "$eval_status" -ne 0 ]]; then
        failed_dir="$OUTPUT_ROOT/${model_name}.failed_${timestamp}"
        mv "$work_dir" "$failed_dir"
        die "$model_name evaluation exited with status $eval_status; kept at $failed_dir"
    fi
    if grep -qE '\[ERROR\] (generate failed|处理 .*失败)' "$work_dir/eval.log"; then
        failed_dir="$OUTPUT_ROOT/${model_name}.failed_${timestamp}"
        mv "$work_dir" "$failed_dir"
        die "$model_name evaluation log contains generation/dataset errors; kept at $failed_dir"
    fi
    if [[ -f "$work_dir/summary.json" ]]; then
        rewrite_summary_output_dir "$work_dir"
    fi
    if ! result_is_complete "$work_dir"; then
        failed_dir="$OUTPUT_ROOT/${model_name}.failed_${timestamp}"
        mv "$work_dir" "$failed_dir"
        die "$model_name result validation failed; kept at $failed_dir"
    fi

    mv "$work_dir" "$final_dir"
    rewrite_summary_output_dir "$final_dir"
    log "DONE $model_name -> $final_dir"
    aggregate_results
done

aggregate_results
log "ALL REQUESTED MODELS FINISHED"
