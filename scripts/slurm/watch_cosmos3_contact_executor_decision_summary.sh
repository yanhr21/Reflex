#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

TRAINING_ROOT="${TRAINING_ROOT:-experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_PATH="${LOG_PATH:-${TRAINING_ROOT}/formal_decision_watch.log}"

mkdir -p "${TRAINING_ROOT}"
{
  echo "schema=cosmos3_contact_executor_decision_watch_v1"
  echo "watcher_start=$(date --iso-8601=seconds)"
  echo "training_root=${TRAINING_ROOT}"
  echo "poll_seconds=${POLL_SECONDS}"
} > "${LOG_PATH}"

while [[ ! -f "${TRAINING_ROOT}/training_summary.json" || \
          ! -f "${TRAINING_ROOT}/checkpoint_final.pt" || \
          ! -f "${TRAINING_ROOT}/formal_live_eval_gate.json" ]]; do
  echo "waiting_final_decision_inputs $(date --iso-8601=seconds)" >> "${LOG_PATH}"
  if [[ -f "${TRAINING_ROOT}/training_history.json" ]]; then
    set +e
    .venv/bin/python scripts/world_model/summarize_cosmos3_contact_executor_history.py \
      --training-root "${TRAINING_ROOT}" \
      >> "${LOG_PATH}" 2>&1
    history_summary_rc=$?
    .venv/bin/python scripts/world_model/summarize_cosmos3_contact_executor_decision.py \
      --training-root "${TRAINING_ROOT}" \
      >> "${LOG_PATH}" 2>&1
    waiting_decision_rc=$?
    set -e
    echo "waiting_decision_summary_refreshed=$(date --iso-8601=seconds) history_rc=${history_summary_rc} decision_rc=${waiting_decision_rc}" >> "${LOG_PATH}"
    if [[ "${waiting_decision_rc}" -ne 0 ]]; then
      echo "decision_watch_stop_failed_state=$(date --iso-8601=seconds) rc=${waiting_decision_rc}" >> "${LOG_PATH}"
      exit 0
    fi
  fi
  sleep "${POLL_SECONDS}"
done

echo "final_decision_inputs_ready=$(date --iso-8601=seconds)" >> "${LOG_PATH}"
set +e
.venv/bin/python scripts/world_model/summarize_cosmos3_contact_executor_decision.py \
  --training-root "${TRAINING_ROOT}" \
  >> "${LOG_PATH}" 2>&1
decision_summary_rc=$?
set -e
echo "formal_decision_summary_done=$(date --iso-8601=seconds) rc=${decision_summary_rc}" >> "${LOG_PATH}"
