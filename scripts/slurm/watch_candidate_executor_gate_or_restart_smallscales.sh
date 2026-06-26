#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

ALLOC_JOB_ID="${ALLOC_JOB_ID:-128888}"
CURRENT_ROOT="${CURRENT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_meanpen5_4096_finalgate}"
RESTART_ROOT="${RESTART_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_smallscales_4096_finalgate}"
WATCH_LOG="${WATCH_LOG:-${RESTART_ROOT}/smallscale_restart_watch.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RESTART_CANDIDATE_SCALES="${RESTART_CANDIDATE_SCALES:-0.05,0.1,0.2}"

mkdir -p "$(dirname "${WATCH_LOG}")" "${RESTART_ROOT}"

log() {
  echo "$(date -Is) $*" | tee -a "${WATCH_LOG}"
}

gate_status() {
  local summary="$1"
  "${ROOT}/.venv/bin/python" - "${summary}" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
payload = json.loads(summary.read_text())
final = payload.get("final_metrics") if isinstance(payload.get("final_metrics"), dict) else {}
eval_metrics = final.get("eval") if isinstance(final, dict) else {}
selected = eval_metrics.get("selected_action_mse") if isinstance(eval_metrics, dict) else None
dp = eval_metrics.get("dp_prior_action_mse") if isinstance(eval_metrics, dict) else None
ready = payload.get("ready_for_formal_live_eval") is True
formal = payload.get("formal_training_floor_met") is True
print(
    "ready={ready} formal={formal} selected={selected} dp={dp} stop_reason={stop}".format(
        ready=str(ready).lower(),
        formal=str(formal).lower(),
        selected=selected,
        dp=dp,
        stop=payload.get("stop_reason"),
    )
)
PY
}

has_nonextern_step() {
  squeue -s -j "${ALLOC_JOB_ID}" -h -o "%i" | grep -v "^${ALLOC_JOB_ID}\.extern$" | grep -q .
}

log "smallscale_restart_watch_start current_root=${CURRENT_ROOT} restart_root=${RESTART_ROOT} allocation=${ALLOC_JOB_ID}"

while [[ ! -s "${CURRENT_ROOT}/training_summary.json" || ! -s "${CURRENT_ROOT}/checkpoint_final.pt" ]]; do
  log "waiting_current_summary"
  sleep "${POLL_SECONDS}"
done

current_status="$(gate_status "${CURRENT_ROOT}/training_summary.json")"
log "current_gate_status ${current_status}"

if [[ "${current_status}" == ready=true* ]]; then
  log "current_gate_passed_existing_after_gate_watcher_handles_live"
  exit 0
fi

if [[ "${current_status}" != *"formal=true"* ]]; then
  log "current_formal_floor_not_met_not_restarting_automatically"
  exit 10
fi

while has_nonextern_step; do
  log "waiting_allocation_step_exit_before_smallscale_restart"
  sleep "${POLL_SECONDS}"
done

if [[ -s "${RESTART_ROOT}/training_summary.json" && -s "${RESTART_ROOT}/checkpoint_final.pt" ]]; then
  restart_status="$(gate_status "${RESTART_ROOT}/training_summary.json")"
  log "restart_already_has_summary ${restart_status}"
else
  log "launching_smallscale_formal_restart candidate_scales=${RESTART_CANDIDATE_SCALES}"
  set +e
  srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --gres=gpu:2 --cpus-per-task=8 --mem=64G \
    env \
    ROOT="${ROOT}" \
    OUTPUT_ROOT="${RESTART_ROOT}" \
    MAX_SAMPLES=0 \
    NPROC_PER_NODE=2 \
    MIN_STEPS=1000 \
    MAX_STEPS=200000 \
    MIN_WALL_SECONDS=10800 \
    MAX_WALL_SECONDS=12600 \
    BATCH_SIZE=256 \
    HIDDEN_DIM=4096 \
    NUM_LAYERS=6 \
    DROPOUT=0.2 \
    LR=0.00005 \
    WEIGHT_DECAY=0.0002 \
    EVAL_EVERY_STEPS=500 \
    SAVE_EVERY_STEPS=20000 \
    CANDIDATE_SAMPLES=48 \
    CANDIDATE_SCALES="${RESTART_CANDIDATE_SCALES}" \
    SELECTOR_RESIDUAL_L2_CAP_QUANTILE=0.9 \
    SELECTOR_RESIDUAL_L2_CAP_MAX=0.02 \
    DP_FALLBACK_PHASES=all \
    DP_FALLBACK_SCORE_MARGIN=0.0 \
    SCORE_MEAN_SOURCE_PENALTY=5.0 \
    SCORE_LARGE_SCALE_SOURCE_PENALTY=0.5 \
    SCORE_STOCHASTIC_SOURCE_PENALTY=1.0 \
    bash scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh \
    > "${RESTART_ROOT}.console.log" 2>&1
  restart_rc=$?
  set -e
  log "smallscale_formal_restart_rc=${restart_rc}"
  if [[ "${restart_rc}" != "0" ]]; then
    exit "${restart_rc}"
  fi
fi

while has_nonextern_step; do
  log "waiting_smallscale_training_step_exit_before_live"
  sleep "${POLL_SECONDS}"
done

restart_status="$(gate_status "${RESTART_ROOT}/training_summary.json")"
log "smallscale_gate_status ${restart_status}"

if [[ "${restart_status}" != ready=true* ]]; then
  log "smallscale_gate_failed_live_not_launched"
  exit 12
fi

log "launching_smallscale_gated_live_panel"
set +e
srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=32G \
  env ROOT="${ROOT}" FORMAL_ROOT="${RESTART_ROOT}" \
  bash scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh \
  >> "${RESTART_ROOT}/candidate_after_gate_watch.log" 2>&1
live_rc=$?
set -e
log "smallscale_gated_live_panel_rc=${live_rc}"
exit "${live_rc}"
