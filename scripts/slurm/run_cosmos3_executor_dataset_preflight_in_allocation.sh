#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run executor dataset preflight only inside a compute-node srun step from a tmux-held allocation.
example=srun --overlap --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=48G bash scripts/slurm/run_cosmos3_executor_dataset_preflight_in_allocation.sh
EOF
  exit 30
fi

CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/executor_dataset_clean_dense_late299_chunk24_debug_${STAMP}}"
CHUNK_SIZE="${CHUNK_SIZE:-24}"
MAX_ROWS_PER_SPLIT="${MAX_ROWS_PER_SPLIT:-64}"
ALLOW_GT_TASK_PATH_DEBUG="${ALLOW_GT_TASK_PATH_DEBUG:-true}"

mkdir -p "${OUTPUT_ROOT}"
cat >"${OUTPUT_ROOT}/run_manifest.txt" <<EOF
timestamp=$(date -Is)
job_id=${SLURM_JOB_ID}
step_id=${SLURM_STEP_ID}
host=$(hostname)
condition_root=${CONDITION_ROOT}
output_root=${OUTPUT_ROOT}
chunk_size=${CHUNK_SIZE}
max_rows_per_split=${MAX_ROWS_PER_SPLIT}
allow_gt_task_path_debug=${ALLOW_GT_TASK_PATH_DEBUG}
resource_boundary=tmux-held interactive allocation; compute-node srun step only; no sbatch.
method_boundary=executor interface/preflight only. Formal executor training still needs causal Cosmos task-path predictions and frozen-DP prior action chunks.
training_floor=full training is at least 2 GPUs for at least 3 hours after gates pass; this preflight is not formal training.
EOF

args=(
  --condition-root "${CONDITION_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --chunk-size "${CHUNK_SIZE}"
  --max-rows-per-split "${MAX_ROWS_PER_SPLIT}"
  --require-prefix-grasped
)
if [[ "${ALLOW_GT_TASK_PATH_DEBUG}" == "true" ]]; then
  args+=(--allow-gt-task-path-debug)
fi

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/build_cosmos3_executor_training_dataset.py" "${args[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/executor_dataset_preflight.log"
