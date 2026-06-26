#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run predicted task-path dataset build only inside a compute-node srun step from a tmux-held allocation.
example=srun --overlap --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G bash scripts/slurm/run_cosmos3_executor_predicted_task_path_dataset_in_allocation.sh
EOF
  exit 30
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_iter_000001500_extra120_abs4gpu_20260615_0220}"
EXECUTOR_JSONL="${EXECUTOR_JSONL:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/executor_dataset_clean_dense_late299_chunk24_debug_20260615_executor_preflight_full_debug/val/executor_dataset_file.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/executor_dataset_cosmos_predicted_task_path_iter1500_${STAMP}}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

mkdir -p "${OUTPUT_ROOT}"
cat >"${OUTPUT_ROOT}/run_manifest.txt" <<EOF
timestamp=$(date -Is)
job_id=${SLURM_JOB_ID}
step_id=${SLURM_STEP_ID}
host=$(hostname)
executor_jsonl=${EXECUTOR_JSONL}
eval_root=${EVAL_ROOT}
output_root=${OUTPUT_ROOT}
max_samples=${MAX_SAMPLES}
resource_boundary=tmux-held interactive allocation; compute-node srun step only; no sbatch.
method_boundary=replace GT debug task paths with causal Cosmos-predicted WAM sidecar paths; not full executor training.
EOF

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/build_cosmos3_executor_predicted_task_path_dataset.py" \
  --executor-jsonl "${EXECUTOR_JSONL}" \
  --eval-root "${EVAL_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-samples "${MAX_SAMPLES}" \
  2>&1 | tee "${OUTPUT_ROOT}/predicted_task_path_dataset.log"
