#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run executor residual calibration watcher only inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

TRAIN_ROOT="${TRAIN_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/executor_residual_train_20260615_executor_residual_train512_formal_2gpu_server54}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-120}"
WATCH_TIMEOUT_SECONDS="${WATCH_TIMEOUT_SECONDS:-18000}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_ROOT}/residual_scale_calibration_final}"

cd "${ROOT}"
mkdir -p "${OUTPUT_ROOT}"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "host=$(hostname)"
  echo "train_root=${TRAIN_ROOT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "check_interval_seconds=${CHECK_INTERVAL_SECONDS}"
  echo "watch_timeout_seconds=${WATCH_TIMEOUT_SECONDS}"
  echo "boundary=Wait for formal residual-executor training summary, then run validation-only residual-scale calibration. Does not launch closed-loop eval."
} | tee "${OUTPUT_ROOT}/watch_calibration_manifest.txt"

summary_json="${TRAIN_ROOT}/training_summary.json"
checkpoint_final="${TRAIN_ROOT}/checkpoint_final.pt"
manifest_json="${TRAIN_ROOT}/training_manifest.json"
start="$(date +%s)"
while [[ ! -s "${summary_json}" || ! -s "${checkpoint_final}" || ! -s "${manifest_json}" ]]; do
  now="$(date +%s)"
  elapsed=$((now - start))
  if (( elapsed > WATCH_TIMEOUT_SECONDS )); then
    echo "executor_residual_calibration_watch_timeout elapsed=${elapsed}" >&2
    exit 32
  fi
  echo "executor_residual_training_not_ready elapsed_seconds=${elapsed}"
  sleep "${CHECK_INTERVAL_SECONDS}"
done

executor_jsonl="$("${PYTHON_BIN}" - "${manifest_json}" <<'PY'
import json, sys
data=json.load(open(sys.argv[1]))
print(data["executor_jsonl"])
PY
)"
dp_prior_jsonl="$("${PYTHON_BIN}" - "${manifest_json}" <<'PY'
import json, sys
data=json.load(open(sys.argv[1]))
print(data["dp_prior_jsonl"])
PY
)"

CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" "${ROOT}/scripts/world_model/calibrate_cosmos3_executor_residual_scale.py" \
  --checkpoint "${checkpoint_final}" \
  --executor-jsonl "${executor_jsonl}" \
  --dp-prior-jsonl "${dp_prior_jsonl}" \
  --training-summary-json "${summary_json}" \
  --output-root "${OUTPUT_ROOT}" \
  2>&1 | tee "${OUTPUT_ROOT}/calibration.log"
