#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

TRAINING_ROOT="${TRAINING_ROOT:-experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k}"
JOB_ID="${JOB_ID:-128023}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_PATH="${TRAINING_ROOT}/formal_watch.log"

mkdir -p "${TRAINING_ROOT}"
{
  echo "schema=cosmos3_contact_executor_formal_watch_v1"
  echo "watcher_start=$(date --iso-8601=seconds)"
  echo "training_root=${TRAINING_ROOT}"
  echo "job_id=${JOB_ID}"
  echo "poll_seconds=${POLL_SECONDS}"
} > "${LOG_PATH}"

run_decision_summary() {
  set +e
  .venv/bin/python scripts/world_model/summarize_cosmos3_contact_executor_decision.py \
    --training-root "${TRAINING_ROOT}" \
    >> "${LOG_PATH}" 2>&1
  decision_summary_rc=$?
  set -e
  echo "formal_decision_summary_done=$(date --iso-8601=seconds) rc=${decision_summary_rc}" >> "${LOG_PATH}"
  return "${decision_summary_rc}"
}

while [[ ! -f "${TRAINING_ROOT}/training_summary.json" || ! -f "${TRAINING_ROOT}/checkpoint_final.pt" ]]; do
  echo "waiting_summary $(date --iso-8601=seconds)" >> "${LOG_PATH}"
  if [[ -f "${TRAINING_ROOT}/checkpoint_final.pt" && ! -f "${TRAINING_ROOT}/training_summary.json" ]]; then
    if ! run_decision_summary; then
      echo "formal_watch_stop_failed_final_without_summary=$(date --iso-8601=seconds)" >> "${LOG_PATH}"
      exit 0
    fi
  fi
  sleep "${POLL_SECONDS}"
done

echo "summary_ready=$(date --iso-8601=seconds)" >> "${LOG_PATH}"
.venv/bin/python scripts/world_model/summarize_cosmos3_contact_executor_history.py \
  --training-root "${TRAINING_ROOT}" \
  >> "${LOG_PATH}" 2>&1
echo "history_summary_done=$(date --iso-8601=seconds)" >> "${LOG_PATH}"

srun --overlap --jobid="${JOB_ID}" --ntasks=1 --cpus-per-task=4 --mem=24G \
  bash -lc "cd '${ROOT}' && .venv/bin/python scripts/world_model/inspect_cosmos3_contact_executor_training.py --training-root '${TRAINING_ROOT}' --require-compute-step" \
  >> "${LOG_PATH}" 2>&1
echo "inspection_done=$(date --iso-8601=seconds)" >> "${LOG_PATH}"

.venv/bin/python scripts/world_model/check_cosmos3_contact_executor_formal_gate.py \
  --training-root "${TRAINING_ROOT}" \
  >> "${LOG_PATH}" 2>&1
echo "formal_gate_done=$(date --iso-8601=seconds)" >> "${LOG_PATH}"

run_decision_summary || true
