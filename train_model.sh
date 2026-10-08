#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

: "${MODEL_SUBDIR:?Set MODEL_SUBDIR}"
: "${EXPERIMENT_NAME:?Set EXPERIMENT_NAME}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES}"
: "${TEACHER_BASE_URL:?Set TEACHER_BASE_URL}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL}"
: "${TEACHER_API_KEY:?Set TEACHER_API_KEY}"

if [[ -z "${BASE_MODEL:-}" ]]; then
    : "${MODELS_DIR:?Set MODELS_DIR or BASE_MODEL}"
    BASE_MODEL="$MODELS_DIR/$MODEL_SUBDIR"
fi

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
export NUM_GPUS="${NUM_GPUS:-${#GPU_IDS[@]}}"
export BASE_MODEL
export PROJECT_NAME="${PROJECT_NAME:-CataOPD}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$PROJECT_DIR/dataset/train/train.parquet}"
export VAL_DATA_PATH="${VAL_DATA_PATH:-$PROJECT_DIR/dataset/val/val.parquet}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${TMPDIR:-/tmp}/cataopd_ray_$$}"
export SPILL_DIR="$RAY_TEMP_DIR/spill"
export RAY_USAGE_STATS_ENABLED=0
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export NUM_CPUS="${NUM_CPUS:-$(getconf _NPROCESSORS_ONLN)}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR}"
export TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.3}"
export TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-1024}"
export TEACHER_MAX_CONCURRENCY="${TEACHER_MAX_CONCURRENCY:-128}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTHONPATH="$PROJECT_DIR/verl:$PROJECT_DIR:${PYTHONPATH:-}"

pick_port() {
    python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
}

export RAY_PORT="${RAY_PORT:-$(pick_port)}"
mkdir -p "$SPILL_DIR" "$WANDB_DIR"

cleanup() {
    pkill -TERM -f "$RAY_TEMP_DIR" 2>/dev/null || true
    sleep 1
    pkill -KILL -f "$RAY_TEMP_DIR" 2>/dev/null || true
    rm -rf "$RAY_TEMP_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

unset RAY_ADDRESS

ray start --head \
    --num-cpus="$NUM_CPUS" \
    --num-gpus="$NUM_GPUS" \
    --port="$RAY_PORT" \
    --include-dashboard=false \
    --temp-dir="$RAY_TEMP_DIR" \
    --disable-usage-stats

export RAY_ADDRESS="localhost:$RAY_PORT"
ray status --address="$RAY_ADDRESS"

python3 -m casd.src.main_agent \
    algorithm.adv_estimator=grpo \
    "data.train_files=['${TRAIN_DATA_PATH}']" \
    "data.val_files=['${VAL_DATA_PATH}']" \
    data.train_batch_size=128 \
    data.max_prompt_length=12800 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=8192 \
    data.use_default_tool_template=False \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$NUM_GPUS" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.stop_token_ids=[151645] \
    actor_rollout_ref.rollout.stop=[] \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.01 \
    'trainer.logger=[console]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    tool.max_turns=1 \
    'tool.tools=[]' \
    tool.max_tool_response_length=8192 \
    +algorithm.use_catalyst=True \
    +algorithm.catalyst_max_rounds=5 \
    +algorithm.catalyst_self_m=5 \
    +actor_rollout_ref.actor.catalyst_coef=0.5 \
    +actor_rollout_ref.actor.catalyst_mode=barrier "$@"
