#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

ALLOC_JOB_ID="${ALLOC_JOB_ID:-128888}"
FORMAL_ROOT="${FORMAL_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_smallscales_nosample_direct_4096_finalgate}"
POST_GATE_STATUS="${POST_GATE_STATUS:-${FORMAL_ROOT}/post_gate_status.json}"
POLL_SECONDS="${POLL_SECONDS:-60}"
POST_FAIL_SETTLE_SECONDS="${POST_FAIL_SETTLE_SECONDS:-75}"
DIFFUSION_OUTPUT_ROOT="${DIFFUSION_OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/candidate_executor_diffusion_smoke_after_nosample_gate_rankcal_20260615}"
FORMAL_DIFFUSION_ROOT="${FORMAL_DIFFUSION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_diffusion_rankcal_after_nosample_gate}"
WATCH_LOG="${WATCH_LOG:-${FORMAL_ROOT}/diffusion_smoke_after_gate_watch.log}"

mkdir -p "$(dirname "${WATCH_LOG}")"

log() {
  echo "$(date -Is) $*" | tee -a "${WATCH_LOG}"
}

status_field() {
  local field="$1"
  "${ROOT}/.venv/bin/python" - "${POST_GATE_STATUS}" "${field}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
field = sys.argv[2]
if not path.is_file():
    print("")
    raise SystemExit(0)
payload = json.loads(path.read_text())
value = payload.get(field)
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
PY
}

summary_ready_field() {
  local path="$1"
  local formal="$2"
  "${ROOT}/.venv/bin/python" - "${path}" "${formal}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
require_formal = sys.argv[2] == "formal"
if not path.is_file():
    print("missing")
    raise SystemExit(0)
payload = json.loads(path.read_text())
if payload.get("generator_type") != "diffusion":
    print("false")
    raise SystemExit(0)
try:
    candidate_samples = int(payload.get("candidate_samples"))
    rank_diffusion_count = int(payload.get("candidate_rank_diffusion_count"))
except (TypeError, ValueError):
    print("false")
    raise SystemExit(0)
if candidate_samples <= 0 or rank_diffusion_count <= 0:
    print("false")
    raise SystemExit(0)
if require_formal and payload.get("formal_training_floor_met") is not True:
    print("false")
    raise SystemExit(0)
if require_formal and payload.get("ready_for_formal_live_eval") is not True:
    print("false")
    raise SystemExit(0)
if not require_formal and payload.get("ready_for_offline_gate") is not True:
    print("false")
    raise SystemExit(0)
final = payload.get("final_metrics") if isinstance(payload.get("final_metrics"), dict) else {}
eval_metrics = final.get("eval") if isinstance(final, dict) else {}
try:
    selected = float(eval_metrics.get("selected_action_mse"))
    dp = float(eval_metrics.get("dp_prior_action_mse"))
except (TypeError, ValueError):
    print("false")
    raise SystemExit(0)
source_counts = eval_metrics.get("candidate_source_counts")
source_counts = source_counts if isinstance(source_counts, dict) else {}
non_dp_selected = 0
for name, count in source_counts.items():
    if str(name) == "dp_prior":
        continue
    try:
        non_dp_selected += int(count)
    except (TypeError, ValueError):
        pass
print("true" if selected <= dp and non_dp_selected > 0 else "false")
PY
}

has_nonextern_step() {
  squeue -s -j "${ALLOC_JOB_ID}" -h -o "%i" \
    | awk -v extern="${ALLOC_JOB_ID}.extern" '$1 != extern {found=1} END {exit found ? 0 : 1}'
}

update_chain_status() {
  "${ROOT}/.venv/bin/python" scripts/world_model/summarize_candidate_executor_diffusion_chain.py \
    --formal-root "${FORMAL_ROOT}" \
    --diffusion-smoke-root "${DIFFUSION_OUTPUT_ROOT}" \
    --formal-diffusion-root "${FORMAL_DIFFUSION_ROOT}" \
    --watch-log "${WATCH_LOG}" \
    >/dev/null 2>&1 || true
}

log "diffusion_after_gate_watch_start allocation=${ALLOC_JOB_ID} formal_root=${FORMAL_ROOT}"
log "diffusion_smoke_root=${DIFFUSION_OUTPUT_ROOT}"
log "formal_diffusion_root=${FORMAL_DIFFUSION_ROOT}"
log "post_fail_settle_seconds=${POST_FAIL_SETTLE_SECONDS}"
log "boundary=lightweight watcher only; diffusion smoke launches after the current post-floor formal decision; non-diffusion live eval is not auto-launched"
update_chain_status

handoff_reason=""
while true; do
  status="$(status_field status)"
  formal_floor="$(status_field formal_training_floor_met)"
  ready="$(status_field ready_for_formal_live_eval)"
  selected="$(status_field final_selected_action_mse)"
  dp="$(status_field final_dp_prior_action_mse)"
  log "post_gate status=${status:-missing} formal_floor=${formal_floor:-missing} ready=${ready:-missing} selected=${selected:-missing} dp=${dp:-missing}"
  update_chain_status

  if [[ "${ready}" == "true" ]]; then
    handoff_reason="formal_gate_passed_non_diffusion_baseline"
    log "formal_gate_passed_non_diffusion_baseline_continue_to_diffusion_smoke"
    update_chain_status
    break
  fi

  if [[ "${status}" == "formal_gate_failed_live_not_allowed" ]]; then
    if [[ "${formal_floor}" != "true" ]]; then
      log "formal_gate_failed_before_floor_not_launching_diffusion_smoke"
      update_chain_status
      exit 10
    fi
    handoff_reason="formal_gate_failed"
    break
  fi

  if [[ "${status}" == "live_panel_summary_available_needs_video_review" || "${status}" == "live_finished_without_panel_summary" ]]; then
    log "live_status_seen_no_diffusion_smoke"
    update_chain_status
    exit 0
  fi

  sleep "${POLL_SECONDS}"
done

if [[ "${handoff_reason}" == "formal_gate_failed" && "${POST_FAIL_SETTLE_SECONDS}" -gt 0 ]]; then
  log "formal_gate_failed_settle_before_diffusion_smoke seconds=${POST_FAIL_SETTLE_SECONDS}"
  sleep "${POST_FAIL_SETTLE_SECONDS}"
  update_chain_status
elif [[ "${handoff_reason}" == "formal_gate_passed_non_diffusion_baseline" ]]; then
  log "formal_gate_passed_non_diffusion_baseline_skipping_live_waiting_for_step_exit"
fi

while has_nonextern_step; do
  log "waiting_allocation_step_exit_before_diffusion_smoke"
  sleep "${POLL_SECONDS}"
done

mkdir -p "${DIFFUSION_OUTPUT_ROOT}"
log "launching_diffusion_smoke output_root=${DIFFUSION_OUTPUT_ROOT}"

if [[ -s "${DIFFUSION_OUTPUT_ROOT}/training_summary.json" && -s "${DIFFUSION_OUTPUT_ROOT}/checkpoint_final.pt" ]]; then
  log "diffusion_smoke_already_has_summary output_root=${DIFFUSION_OUTPUT_ROOT}"
  smoke_rc=0
else
  set +e
  srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=02:00:00 \
    env \
    ROOT="${ROOT}" \
    OUTPUT_ROOT="${DIFFUSION_OUTPUT_ROOT}" \
    GENERATOR_TYPE=diffusion \
    DIFFUSION_STEPS="${DIFFUSION_STEPS:-16}" \
    CANDIDATE_SAMPLES="${CANDIDATE_SAMPLES:-8}" \
    CANDIDATE_SCALES="${CANDIDATE_SCALES:-0.05,0.1,0.2}" \
    CANDIDATE_RANK_LOSS_WEIGHT="${CANDIDATE_RANK_LOSS_WEIGHT:-0.35}" \
    CANDIDATE_RANK_RANDOM_COUNT="${CANDIDATE_RANK_RANDOM_COUNT:-4}" \
    CANDIDATE_RANK_DIFFUSION_COUNT="${CANDIDATE_RANK_DIFFUSION_COUNT:-1}" \
    CANDIDATE_RANK_TEMPERATURE="${CANDIDATE_RANK_TEMPERATURE:-1.0}" \
    MAX_SAMPLES="${MAX_SAMPLES:-512}" \
    NPROC_PER_NODE="${NPROC_PER_NODE:-1}" \
    MIN_STEPS="${MIN_STEPS:-50}" \
    MAX_STEPS="${MAX_STEPS:-100}" \
    MIN_WALL_SECONDS=0 \
    MAX_WALL_SECONDS=0 \
    bash scripts/slurm/run_cosmos3_candidate_executor_diffusion_smoke_in_allocation.sh \
    > "${DIFFUSION_OUTPUT_ROOT}.console.log" 2>&1
  smoke_rc=$?
  set -e
fi

log "diffusion_smoke_rc=${smoke_rc} console=${DIFFUSION_OUTPUT_ROOT}.console.log"
update_chain_status
if [[ "${smoke_rc}" != "0" ]]; then
  exit "${smoke_rc}"
fi

smoke_ready="$(summary_ready_field "${DIFFUSION_OUTPUT_ROOT}/training_summary.json" smoke)"
log "diffusion_smoke_ready_for_formal=${smoke_ready}"
update_chain_status
if [[ "${smoke_ready}" != "true" ]]; then
  log "diffusion_smoke_gate_failed_not_launching_formal"
  update_chain_status
  exit 20
fi

while has_nonextern_step; do
  log "waiting_allocation_step_exit_before_formal_diffusion"
  sleep "${POLL_SECONDS}"
done

mkdir -p "${FORMAL_DIFFUSION_ROOT}"
if [[ -s "${FORMAL_DIFFUSION_ROOT}/training_summary.json" && -s "${FORMAL_DIFFUSION_ROOT}/checkpoint_final.pt" ]]; then
  log "formal_diffusion_already_has_summary output_root=${FORMAL_DIFFUSION_ROOT}"
  formal_rc=0
else
  log "launching_formal_diffusion output_root=${FORMAL_DIFFUSION_ROOT}"
  set +e
  srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --gres=gpu:2 --cpus-per-task=8 --mem=96G --time=04:00:00 \
    env \
    ROOT="${ROOT}" \
    OUTPUT_ROOT="${FORMAL_DIFFUSION_ROOT}" \
    GENERATOR_TYPE=diffusion \
    DIFFUSION_STEPS="${FORMAL_DIFFUSION_STEPS:-16}" \
    CANDIDATE_SAMPLES="${FORMAL_CANDIDATE_SAMPLES:-8}" \
    CANDIDATE_SCALES="${FORMAL_CANDIDATE_SCALES:-0.05,0.1,0.2}" \
    CANDIDATE_RANK_LOSS_WEIGHT="${FORMAL_CANDIDATE_RANK_LOSS_WEIGHT:-0.35}" \
    CANDIDATE_RANK_RANDOM_COUNT="${FORMAL_CANDIDATE_RANK_RANDOM_COUNT:-4}" \
    CANDIDATE_RANK_DIFFUSION_COUNT="${FORMAL_CANDIDATE_RANK_DIFFUSION_COUNT:-1}" \
    CANDIDATE_RANK_TEMPERATURE="${FORMAL_CANDIDATE_RANK_TEMPERATURE:-1.0}" \
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
    DP_FALLBACK_PHASES=all \
    DP_FALLBACK_SCORE_MARGIN=0.0 \
    SCORE_MEAN_SOURCE_PENALTY=5.0 \
    SCORE_LARGE_SCALE_SOURCE_PENALTY=0.5 \
    SCORE_STOCHASTIC_SOURCE_PENALTY=1.0 \
    SELECTOR_RESIDUAL_L2_CAP_QUANTILE=0.9 \
    SELECTOR_RESIDUAL_L2_CAP_MAX=0.02 \
    bash scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh \
    > "${FORMAL_DIFFUSION_ROOT}.console.log" 2>&1
  formal_rc=$?
  set -e
fi

log "formal_diffusion_rc=${formal_rc} console=${FORMAL_DIFFUSION_ROOT}.console.log"
update_chain_status
if [[ "${formal_rc}" != "0" ]]; then
  exit "${formal_rc}"
fi

formal_ready="$(summary_ready_field "${FORMAL_DIFFUSION_ROOT}/training_summary.json" formal)"
log "formal_diffusion_ready_for_live=${formal_ready}"
update_chain_status
if [[ "${formal_ready}" != "true" ]]; then
  log "formal_diffusion_gate_failed_live_not_launched"
  update_chain_status
  exit 21
fi

while has_nonextern_step; do
  log "waiting_allocation_step_exit_before_formal_diffusion_live"
  sleep "${POLL_SECONDS}"
done

log "launching_formal_diffusion_gated_live_panel"
set +e
srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00 \
  env ROOT="${ROOT}" FORMAL_ROOT="${FORMAL_DIFFUSION_ROOT}" REQUIRE_DIFFUSION_GENERATOR=true \
  bash scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh \
  >> "${FORMAL_DIFFUSION_ROOT}/candidate_after_gate_live_watch.log" 2>&1
live_rc=$?
set -e
log "formal_diffusion_gated_live_rc=${live_rc}"
update_chain_status
exit "${live_rc}"
