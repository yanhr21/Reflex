#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run executor WAM eval input build only inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050}"
EXECUTOR_JSONL="${EXECUTOR_JSONL:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/executor_dataset_clean_dense_late299_chunk24_debug_20260615_executor_preflight_full_debug/train/executor_dataset_file.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/executor_wam_eval_inputs_${STAMP}}"
MAX_SAMPLES="${MAX_SAMPLES:-32}"
SPLIT="${SPLIT:-}"
SELECTION_POLICY="${SELECTION_POLICY:-role_scenario_round_robin}"

mkdir -p "${OUTPUT_ROOT}"
cat >"${OUTPUT_ROOT}/run_manifest.txt" <<EOF
timestamp=$(date -Is)
job_id=${SLURM_JOB_ID}
step_id=${SLURM_STEP_ID}
host=$(hostname)
condition_root=${CONDITION_ROOT}
executor_jsonl=${EXECUTOR_JSONL}
output_root=${OUTPUT_ROOT}
max_samples=${MAX_SAMPLES}
selection_policy=${SELECTION_POLICY}
resource_boundary=tmux-held interactive allocation; compute-node srun step only; no sbatch.
method_boundary=build executor-targeted Cosmos inference inputs; no training and no method evidence yet.
EOF

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/build_cosmos3_executor_wam_eval_inputs.py" \
  --condition-root "${CONDITION_ROOT}" \
  --executor-jsonl "${EXECUTOR_JSONL}" \
  --output-root "${OUTPUT_ROOT}" \
  --split "${SPLIT}" \
  --max-samples "${MAX_SAMPLES}" \
  --selection-policy "${SELECTION_POLICY}" \
  2>&1 | tee "${OUTPUT_ROOT}/build_executor_wam_eval_inputs.log"
