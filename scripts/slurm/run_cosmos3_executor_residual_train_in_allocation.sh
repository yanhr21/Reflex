#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run formal executor residual training only inside a compute-node srun step from a tmux-held allocation.
example=srun --overlap --jobid=<held_job_id> --ntasks=1 --gres=gpu:2 --cpus-per-task=16 --mem=120G bash scripts/slurm/run_cosmos3_executor_residual_train_in_allocation.sh
EOF
  exit 30
fi

EXECUTOR_JSONL="${EXECUTOR_JSONL:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/executor_dataset_cosmos_predicted_task_path_iter1500_20260615_pred_task_path_train64_diverse/train/executor_dataset_file.jsonl}"
DP_PRIOR_JSONL="${DP_PRIOR_JSONL:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_smoke_20260615_pred_task_path_dp_prior_train64_diverse/dp_prior_dataset_file.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/executor_residual_train_${STAMP}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-50315}"
MIN_WALL_SECONDS="${MIN_WALL_SECONDS:-10800}"
MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-14400}"
MIN_STEPS="${MIN_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HIDDEN_DIM="${HIDDEN_DIM:-1024}"
NUM_LAYERS="${NUM_LAYERS:-4}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"

mkdir -p "${OUTPUT_ROOT}"
cat >"${OUTPUT_ROOT}/run_manifest.txt" <<EOF
timestamp=$(date -Is)
job_id=${SLURM_JOB_ID}
step_id=${SLURM_STEP_ID}
host=$(hostname)
executor_jsonl=${EXECUTOR_JSONL}
dp_prior_jsonl=${DP_PRIOR_JSONL}
output_root=${OUTPUT_ROOT}
nproc_per_node=${NPROC_PER_NODE}
min_wall_seconds=${MIN_WALL_SECONDS}
max_wall_seconds=${MAX_WALL_SECONDS}
min_steps=${MIN_STEPS}
max_steps=${MAX_STEPS}
batch_size=${BATCH_SIZE}
hidden_dim=${HIDDEN_DIM}
num_layers=${NUM_LAYERS}
resource_boundary=tmux-held interactive allocation; compute-node srun step only; no sbatch.
training_floor=formal executor training floor is 2 GPUs for at least 3 hours unless this is explicitly run with a shorter diagnostic override.
method_boundary=causal Cosmos-predicted task path plus frozen DP prior; no GT future task path conditioning.
EOF

cd "${ROOT}"
"${ROOT}/.venv/bin/torchrun" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  "${ROOT}/scripts/world_model/train_cosmos3_executor_residual.py" \
  --executor-jsonl "${EXECUTOR_JSONL}" \
  --dp-prior-jsonl "${DP_PRIOR_JSONL}" \
  --output-root "${OUTPUT_ROOT}" \
  --min-wall-seconds "${MIN_WALL_SECONDS}" \
  --max-wall-seconds "${MAX_WALL_SECONDS}" \
  --min-steps "${MIN_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --val-fraction "${VAL_FRACTION}" \
  2>&1 | tee "${OUTPUT_ROOT}/train.log"
