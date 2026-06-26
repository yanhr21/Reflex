#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run DP-prior export only inside a compute-node srun step from a tmux-held allocation.
example=srun --overlap --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G bash scripts/slurm/run_cosmos3_executor_dp_prior_smoke_in_allocation.sh
EOF
  exit 30
fi

EXECUTOR_JSONL="${EXECUTOR_JSONL:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/executor_dataset_clean_dense_late299_chunk24_debug_20260615_executor_preflight_full_debug/train/executor_dataset_file.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_smoke_${STAMP}}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"

mkdir -p "${OUTPUT_ROOT}"
cat >"${OUTPUT_ROOT}/run_manifest.txt" <<EOF
timestamp=$(date -Is)
job_id=${SLURM_JOB_ID}
step_id=${SLURM_STEP_ID}
host=$(hostname)
executor_jsonl=${EXECUTOR_JSONL}
output_root=${OUTPUT_ROOT}
max_samples=${MAX_SAMPLES}
resource_boundary=tmux-held interactive allocation; compute-node srun step only; no sbatch.
method_boundary=frozen static DP proposal chunks only; not dynamic-task success evidence.
EOF

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/export_cosmos3_executor_dp_prior_chunks.py" \
  --executor-jsonl "${EXECUTOR_JSONL}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-samples "${MAX_SAMPLES}" \
  2>&1 | tee "${OUTPUT_ROOT}/dp_prior_smoke.log"
