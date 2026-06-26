#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${ALLOC_JOB_ID:-}" ]]; then
  echo "missing_alloc_job_id=true" >&2
  echo "reason=Set ALLOC_JOB_ID to a tmux-held interactive Slurm allocation." >&2
  exit 2
fi

POLL_SECONDS="${POLL_SECONDS:-1800}"
FORMAL_GPUS="${FORMAL_GPUS:-2}"
FORMAL_NPROC_PER_NODE="${FORMAL_NPROC_PER_NODE:-${FORMAL_GPUS}}"
FORMAL_MIN_GPUS="${FORMAL_MIN_GPUS:-${FORMAL_NPROC_PER_NODE}}"
FORMAL_MIN_WALL_SECONDS="${FORMAL_MIN_WALL_SECONDS:-10800}"
FORMAL_MAX_WALL_SECONDS="${FORMAL_MAX_WALL_SECONDS:-12600}"
SMOKE_GPUS="${SMOKE_GPUS:-1}"
SMOKE_NPROC_PER_NODE="${SMOKE_NPROC_PER_NODE:-1}"
CANARY_GPUS="${CANARY_GPUS:-${FORMAL_GPUS}}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_ROOT="${SMOKE_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/candidate_executor_diffusion_smoke_20260616_rankcal_${STAMP}}"
FORMAL_DIFFUSION_ROOT="${FORMAL_DIFFUSION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_${FORMAL_GPUS}gpu_diffusion_rankcal_finalgate_${STAMP}}"
WATCH_LOG="${WATCH_LOG:-${FORMAL_DIFFUSION_ROOT}/diffusion_chain_watch.log}"
CANARY_LOG="${CANARY_LOG:-${FORMAL_DIFFUSION_ROOT}/cuda_canary.log}"
CANARY_OK="${CANARY_OK:-${FORMAL_DIFFUSION_ROOT}/cuda_canary.ok}"
PRELAUNCH_AUDIT_JSON="${PRELAUNCH_AUDIT_JSON:-${FORMAL_DIFFUSION_ROOT}/prelaunch_audit.json}"
PRELAUNCH_AUDIT_MD="${PRELAUNCH_AUDIT_MD:-${FORMAL_DIFFUSION_ROOT}/prelaunch_audit.md}"
LIVE_WATCH_LOG="${LIVE_WATCH_LOG:-${FORMAL_DIFFUSION_ROOT}/candidate_after_gate_live_watch.log}"
POST_GATE_STATUS_LOG="${POST_GATE_STATUS_LOG:-${FORMAL_DIFFUSION_ROOT}/post_gate_status_watch.log}"
CHAIN_LOCK_DIR="${CHAIN_LOCK_DIR:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/candidate_executor_diffusion_chain_20260616.launch.lock}"
CHAIN_DONE_MARKER="${CHAIN_DONE_MARKER:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/candidate_executor_diffusion_chain_20260616.done}"
CHAIN_TERMINAL_MARKER="${CHAIN_TERMINAL_MARKER:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/candidate_executor_diffusion_chain_20260616.terminal}"
CHAIN_FORMAL_READY_MARKER="${CHAIN_FORMAL_READY_MARKER:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/candidate_executor_diffusion_chain_20260616.formal_ready}"
LIVE_FORBIDDEN_NODES="${LIVE_FORBIDDEN_NODES:-server35}"
ALLOC_TMUX_SESSION="${ALLOC_TMUX_SESSION:-}"
LOCK_ACQUIRED=0

for name in FORMAL_GPUS FORMAL_NPROC_PER_NODE FORMAL_MIN_GPUS FORMAL_MIN_WALL_SECONDS FORMAL_MAX_WALL_SECONDS SMOKE_GPUS SMOKE_NPROC_PER_NODE CANARY_GPUS; do
  value="${!name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -lt 1 ]]; then
    echo "invalid_positive_integer ${name}=${value}" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "${WATCH_LOG}")" "${SMOKE_ROOT}" "${FORMAL_DIFFUSION_ROOT}"

log() {
  echo "$(date -Is) $*" | tee -a "${WATCH_LOG}"
}

release_launch_lock() {
  if [[ "${LOCK_ACQUIRED}" == "1" ]]; then
    rm -rf "${CHAIN_LOCK_DIR}"
  fi
}

release_own_allocation_tmux() {
  local why="$1"
  if [[ -n "${ALLOC_TMUX_SESSION}" ]]; then
    log "releasing_own_allocation_tmux reason=${why} session=${ALLOC_TMUX_SESSION}"
    tmux send-keys -t "${ALLOC_TMUX_SESSION}" C-c 2>/dev/null || true
  else
    log "own_allocation_tmux_unknown_not_released reason=${why}"
  fi
}

write_terminal_marker() {
  local status="$1"
  local rc="${2:-}"
  {
    echo "date=$(date -Is)"
    echo "status=${status}"
    echo "allocation=${ALLOC_JOB_ID}"
    echo "formal_root=${FORMAL_DIFFUSION_ROOT}"
    echo "formal_gpus=${FORMAL_GPUS}"
    if [[ -n "${rc}" ]]; then
      echo "rc=${rc}"
    fi
  } > "${CHAIN_TERMINAL_MARKER}"
  log "chain_terminal_marker_written status=${status} marker=${CHAIN_TERMINAL_MARKER}"
}

write_formal_ready_marker() {
  local status="$1"
  {
    echo "date=$(date -Is)"
    echo "status=${status}"
    echo "allocation=${ALLOC_JOB_ID}"
    echo "formal_root=${FORMAL_DIFFUSION_ROOT}"
    echo "formal_gpus=${FORMAL_GPUS}"
    echo "live_forbidden_nodes=${LIVE_FORBIDDEN_NODES}"
    echo "allocation_nodes=$(allocation_nodelist || true)"
  } > "${CHAIN_FORMAL_READY_MARKER}"
  log "chain_formal_ready_marker_written status=${status} marker=${CHAIN_FORMAL_READY_MARKER}"
}

marker_value() {
  local path="$1"
  local key="$2"
  awk -F= -v k="${key}" '$1 == k {sub(/^[^=]*=/, ""); print; exit}' "${path}"
}

allocation_state() {
  squeue -j "${ALLOC_JOB_ID}" -h -o "%T" | head -n 1
}

allocation_reason() {
  squeue -j "${ALLOC_JOB_ID}" -h -o "%R" | head -n 1
}

allocation_nodelist() {
  squeue -j "${ALLOC_JOB_ID}" -h -o "%N" | head -n 1
}

allocation_hostnames() {
  local nodes="$1"
  if [[ -z "${nodes}" || "${nodes}" == "(null)" || "${nodes}" == "NODELIST" ]]; then
    return 0
  fi
  scontrol show hostnames "${nodes}" 2>/dev/null || echo "${nodes}"
}

live_forbidden_on_current_node() {
  local nodes hosts forbidden
  nodes="$(allocation_nodelist || true)"
  hosts="$(allocation_hostnames "${nodes}" || true)"
  for forbidden in ${LIVE_FORBIDDEN_NODES//,/ }; do
    if [[ -n "${forbidden}" ]] && printf '%s\n' "${hosts}" | grep -qx "${forbidden}"; then
      return 0
    fi
  done
  return 1
}

run_gated_live_panel() {
  LIVE_WATCH_LOG="${LIVE_WATCH_LOG:-${FORMAL_DIFFUSION_ROOT}/candidate_after_gate_live_watch.log}"
  POST_GATE_STATUS_LOG="${POST_GATE_STATUS_LOG:-${FORMAL_DIFFUSION_ROOT}/post_gate_status_watch.log}"
  log "launching_formal_diffusion_gated_live_panel formal_root=${FORMAL_DIFFUSION_ROOT}"
  set +e
  srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00 \
    env ROOT="${ROOT}" FORMAL_ROOT="${FORMAL_DIFFUSION_ROOT}" REQUIRE_DIFFUSION_GENERATOR=true \
    bash scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh \
    >> "${LIVE_WATCH_LOG}" 2>&1
  live_rc=$?
  set -e
  log "formal_diffusion_gated_live_rc=${live_rc}"
  echo "candidate_after_gate_live_rc=${live_rc}" >> "${LIVE_WATCH_LOG}"
  set +e
  srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:20:00 \
    env \
    ROOT="${ROOT}" \
    FORMAL_DIFFUSION_ROOT="${FORMAL_DIFFUSION_ROOT}" \
    LIVE_WATCH_LOG="${LIVE_WATCH_LOG}" \
    bash -lc '
      cd "${ROOT}" &&
      .venv/bin/python scripts/world_model/watch_candidate_executor_post_gate_status.py \
        --formal-root "${FORMAL_DIFFUSION_ROOT}" \
        --watch-log "${LIVE_WATCH_LOG}" \
        --poll-seconds 5
    ' >> "${POST_GATE_STATUS_LOG}" 2>&1
  post_gate_rc=$?
  set -e
  log "post_gate_status_rc=${post_gate_rc} log=${POST_GATE_STATUS_LOG}"
  write_terminal_marker "live_finished" "${live_rc}"
  if [[ "${live_rc}" == "0" ]]; then
    {
      echo "date=$(date -Is)"
      echo "allocation=${ALLOC_JOB_ID}"
      echo "formal_root=${FORMAL_DIFFUSION_ROOT}"
      echo "live_rc=${live_rc}"
    } > "${CHAIN_DONE_MARKER}"
    log "chain_done_marker_written marker=${CHAIN_DONE_MARKER}"
  fi
  update_chain_status
  return "${live_rc}"
}

summary_ready_field() {
  local path="$1"
  local formal="$2"
  local state output rc
  state="$(allocation_state || true)"
  if [[ "${state}" != "RUNNING" ]]; then
    echo "missing"
    return 0
  fi
  set +e
  output="$(srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:05:00 \
    env \
    ROOT="${ROOT}" \
    SUMMARY_READY_PATH="${path}" \
    SUMMARY_READY_MODE="${formal}" \
    bash -lc '
      cd "${ROOT}" &&
      .venv/bin/python - "${SUMMARY_READY_PATH}" "${SUMMARY_READY_MODE}" <<'"'"'PY'"'"'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
require_formal = sys.argv[2] == "formal"
if not path.is_file():
    print("missing")
    raise SystemExit(0)
payload = json.loads(path.read_text())
expected_root = path.parent.resolve()
summary_root_raw = payload.get("output_root")
if not summary_root_raw:
    print("false")
    raise SystemExit(0)
summary_root = Path(str(summary_root_raw)).resolve()
if summary_root != expected_root:
    print("false")
    raise SystemExit(0)
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
metrics_source = "final"
if require_formal:
    metrics_source = str(payload.get("formal_live_eval_metrics_source") or "final")
if metrics_source == "best_gate":
    metric_container = payload.get("best_gate_metrics") if isinstance(payload.get("best_gate_metrics"), dict) else {}
else:
    metric_container = payload.get("final_metrics") if isinstance(payload.get("final_metrics"), dict) else {}
eval_metrics = metric_container.get("eval") if isinstance(metric_container, dict) else {}
try:
    selected = float(eval_metrics.get("selected_action_mse"))
    dp = float(eval_metrics.get("dp_prior_action_mse"))
except (TypeError, ValueError):
    print("false")
    raise SystemExit(0)
if not (math.isfinite(selected) and math.isfinite(dp)):
    print("false")
    raise SystemExit(0)
if not require_formal:
    try:
        steps = int(payload.get("steps"))
        num_sources = int(eval_metrics.get("num_candidate_sources"))
    except (TypeError, ValueError):
        print("false")
        raise SystemExit(0)
    print("true" if steps > 0 and num_sources > 2 else "false")
    raise SystemExit(0)
if payload.get("formal_training_floor_met") is not True:
    print("false")
    raise SystemExit(0)
if payload.get("ready_for_formal_live_eval") is not True:
    print("false")
    raise SystemExit(0)
live_checkpoint = payload.get("formal_live_eval_checkpoint")
if not live_checkpoint or not Path(str(live_checkpoint)).is_file():
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
    ' 2>> "${WATCH_LOG}")"
  rc=$?
  set -e
  if [[ "${rc}" != "0" ]]; then
    echo "false"
    return 0
  fi
  printf '%s\n' "${output}" | tail -n 1
}

update_chain_status() {
  local state
  state="$(allocation_state || true)"
  if [[ "${state}" != "RUNNING" ]]; then
    return 0
  fi
  srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:05:00 \
    env \
    ROOT="${ROOT}" \
    FORMAL_DIFFUSION_ROOT="${FORMAL_DIFFUSION_ROOT}" \
    SMOKE_ROOT="${SMOKE_ROOT}" \
    WATCH_LOG="${WATCH_LOG}" \
    bash -lc '
      cd "${ROOT}" &&
      .venv/bin/python scripts/world_model/summarize_candidate_executor_diffusion_chain.py \
        --formal-root "${FORMAL_DIFFUSION_ROOT}" \
        --diffusion-smoke-root "${SMOKE_ROOT}" \
        --formal-diffusion-root "${FORMAL_DIFFUSION_ROOT}" \
        --watch-log "${WATCH_LOG}" \
        >/dev/null 2>&1
    ' >/dev/null 2>&1 || true
}

run_prelaunch_audit_in_allocation() {
  if [[ -s "${CHAIN_FORMAL_READY_MARKER}" ]]; then
    log "prelaunch_audit_skipped_formal_ready_marker_present marker=${CHAIN_FORMAL_READY_MARKER}"
    return 0
  fi
  log "launching_prelaunch_audit_in_allocation json=${PRELAUNCH_AUDIT_JSON}"
  set +e
  srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --gres=gpu:"${SMOKE_GPUS}" --cpus-per-task=4 --mem=32G --time=00:30:00 \
    env \
    ROOT="${ROOT}" \
    FORMAL_DIFFUSION_ROOT="${FORMAL_DIFFUSION_ROOT}" \
    PRELAUNCH_AUDIT_JSON="${PRELAUNCH_AUDIT_JSON}" \
    PRELAUNCH_AUDIT_MD="${PRELAUNCH_AUDIT_MD}" \
    FORMAL_CANDIDATE_SAMPLES="${FORMAL_CANDIDATE_SAMPLES:-8}" \
    FORMAL_CANDIDATE_RANK_DIFFUSION_COUNT="${FORMAL_CANDIDATE_RANK_DIFFUSION_COUNT:-1}" \
    FORMAL_NPROC_PER_NODE="${FORMAL_NPROC_PER_NODE}" \
    FORMAL_MIN_WALL_SECONDS="${FORMAL_MIN_WALL_SECONDS}" \
    bash -lc '
      cd "${ROOT}" &&
      .venv/bin/python scripts/world_model/audit_candidate_executor_diffusion_prelaunch.py \
        --output-root "${FORMAL_DIFFUSION_ROOT}" \
        --output-json "${PRELAUNCH_AUDIT_JSON}" \
        --output-md "${PRELAUNCH_AUDIT_MD}" \
        --planned-candidate-samples "${FORMAL_CANDIDATE_SAMPLES}" \
        --planned-rank-diffusion-count "${FORMAL_CANDIDATE_RANK_DIFFUSION_COUNT}" \
        --planned-nproc-per-node "${FORMAL_NPROC_PER_NODE}" \
        --planned-min-wall-seconds "${FORMAL_MIN_WALL_SECONDS}" \
        --ready-blocker-class ready_for_chain_launch
    ' >> "${WATCH_LOG}" 2>&1
  prelaunch_rc=$?
  set -e
  log "prelaunch_audit_rc=${prelaunch_rc} md=${PRELAUNCH_AUDIT_MD}"
  update_chain_status
  if [[ "${prelaunch_rc}" != "0" ]]; then
    log "prelaunch_audit_failed json=${PRELAUNCH_AUDIT_JSON}"
    exit 13
  fi
}

log "diffusion_chain_watch_start allocation=${ALLOC_JOB_ID}"
log "smoke_root=${SMOKE_ROOT}"
log "formal_diffusion_root=${FORMAL_DIFFUSION_ROOT}"
log "resource_contract formal_gpus=${FORMAL_GPUS} formal_nproc=${FORMAL_NPROC_PER_NODE} formal_min_gpus=${FORMAL_MIN_GPUS} formal_min_wall_seconds=${FORMAL_MIN_WALL_SECONDS} smoke_gpus=${SMOKE_GPUS} canary_gpus=${CANARY_GPUS}"
log "launch_mutex lock_dir=${CHAIN_LOCK_DIR} done_marker=${CHAIN_DONE_MARKER} terminal_marker=${CHAIN_TERMINAL_MARKER} formal_ready_marker=${CHAIN_FORMAL_READY_MARKER}"
log "allocation_tmux_session=${ALLOC_TMUX_SESSION:-unknown}"
log "live_forbidden_nodes=${LIVE_FORBIDDEN_NODES}"
log "boundary=lightweight watcher; login node only polls Slurm/tmux; project-code audits and compute run through srun inside the tmux-held allocation"
log "prelaunch_audit_deferred_until_allocation_running json=${PRELAUNCH_AUDIT_JSON}"
update_chain_status

while true; do
  state="$(allocation_state || true)"
  reason="$(allocation_reason || true)"
  if [[ "${state}" == "RUNNING" ]]; then
    log "allocation_running reason=${reason:-unknown}"
    break
  fi
  if [[ -z "${state}" ]]; then
    log "allocation_missing_or_finished"
    update_chain_status
    exit 12
  fi
  log "allocation_wait state=${state} reason=${reason:-unknown}"
  update_chain_status
  sleep "${POLL_SECONDS}"
done

if [[ -s "${CHAIN_DONE_MARKER}" ]]; then
  log "chain_done_marker_present_not_launching_compute marker=${CHAIN_DONE_MARKER}"
  release_own_allocation_tmux "chain_already_done"
  update_chain_status
  exit 0
fi
if [[ -s "${CHAIN_TERMINAL_MARKER}" ]]; then
  log "chain_terminal_marker_present_not_launching_compute marker=${CHAIN_TERMINAL_MARKER}"
  release_own_allocation_tmux "chain_already_terminal"
  update_chain_status
  exit 0
fi
if mkdir "${CHAIN_LOCK_DIR}" 2>/dev/null; then
  LOCK_ACQUIRED=1
  trap release_launch_lock EXIT INT TERM
  {
    echo "date=$(date -Is)"
    echo "allocation=${ALLOC_JOB_ID}"
    echo "formal_root=${FORMAL_DIFFUSION_ROOT}"
    echo "host=$(hostname)"
    echo "formal_gpus=${FORMAL_GPUS}"
  } > "${CHAIN_LOCK_DIR}/owner.txt"
  log "launch_mutex_acquired owner=${CHAIN_LOCK_DIR}/owner.txt"
else
  log "launch_mutex_held_by_another_watcher owner=${CHAIN_LOCK_DIR}/owner.txt"
  release_own_allocation_tmux "another_watcher_holds_launch_mutex"
  update_chain_status
  exit 0
fi

run_prelaunch_audit_in_allocation

if [[ ! -s "${CANARY_OK}" ]]; then
  log "launching_cuda_canary"
  set +e
  srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --gres=gpu:"${CANARY_GPUS}" --cpus-per-task=4 --mem=16G --time=00:10:00 \
    env ROOT="${ROOT}" CANARY_GPUS="${CANARY_GPUS}" bash -lc \
    'cd "${ROOT}" && .venv/bin/python -c "import json, os, torch; required=int(os.environ.get(\"CANARY_GPUS\", \"1\")); info={\"required_device_count\": required, \"torch\": torch.__version__, \"cuda_available\": torch.cuda.is_available(), \"device_count\": torch.cuda.device_count(), \"devices\": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}; print(json.dumps(info, sort_keys=True)); raise SystemExit(0 if info[\"cuda_available\"] and info[\"device_count\"] >= required else 3)"' \
    > "${CANARY_LOG}" 2>&1
  canary_rc=$?
  set -e
  log "cuda_canary_rc=${canary_rc} log=${CANARY_LOG}"
  if [[ "${canary_rc}" != "0" ]]; then
    update_chain_status
    exit "${canary_rc}"
  fi
  date -Is > "${CANARY_OK}"
else
  log "cuda_canary_already_passed marker=${CANARY_OK}"
fi

if [[ -s "${CHAIN_FORMAL_READY_MARKER}" ]]; then
  ready_root="$(marker_value "${CHAIN_FORMAL_READY_MARKER}" formal_root || true)"
  if [[ -z "${ready_root}" || ! -d "${ready_root}" ]]; then
    log "formal_ready_marker_invalid marker=${CHAIN_FORMAL_READY_MARKER} formal_root=${ready_root:-missing}"
    update_chain_status
    exit 24
  fi
  FORMAL_DIFFUSION_ROOT="${ready_root}"
  LIVE_WATCH_LOG="${FORMAL_DIFFUSION_ROOT}/candidate_after_gate_live_watch_${ALLOC_JOB_ID}.log"
  POST_GATE_STATUS_LOG="${FORMAL_DIFFUSION_ROOT}/post_gate_status_watch_${ALLOC_JOB_ID}.log"
  log "formal_ready_marker_present_run_live_only formal_root=${FORMAL_DIFFUSION_ROOT}"
  if live_forbidden_on_current_node; then
    log "live_forbidden_on_current_node_defer_again nodes=$(allocation_nodelist || true) forbidden=${LIVE_FORBIDDEN_NODES}"
    release_own_allocation_tmux "formal_ready_but_live_forbidden_node"
    update_chain_status
    exit 0
  fi
  run_gated_live_panel
  exit "$?"
fi

if [[ -s "${SMOKE_ROOT}/training_summary.json" && -s "${SMOKE_ROOT}/checkpoint_final.pt" ]]; then
  log "diffusion_smoke_already_complete"
  smoke_rc=0
else
  log "launching_diffusion_smoke"
  set +e
  srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --gres=gpu:"${SMOKE_GPUS}" --cpus-per-task=8 --mem=80G --time=02:00:00 \
    env \
    ROOT="${ROOT}" \
    OUTPUT_ROOT="${SMOKE_ROOT}" \
    GENERATOR_TYPE=diffusion \
    DIFFUSION_STEPS="${DIFFUSION_STEPS:-16}" \
    CANDIDATE_SAMPLES="${CANDIDATE_SAMPLES:-8}" \
    CANDIDATE_SCALES="${CANDIDATE_SCALES:-0.05,0.1,0.2}" \
    CANDIDATE_RANK_LOSS_WEIGHT="${CANDIDATE_RANK_LOSS_WEIGHT:-0.35}" \
    CANDIDATE_RANK_RANDOM_COUNT="${CANDIDATE_RANK_RANDOM_COUNT:-4}" \
    CANDIDATE_RANK_DIFFUSION_COUNT="${CANDIDATE_RANK_DIFFUSION_COUNT:-1}" \
    CANDIDATE_RANK_TEMPERATURE="${CANDIDATE_RANK_TEMPERATURE:-1.0}" \
    MAX_SAMPLES="${MAX_SAMPLES:-512}" \
    NPROC_PER_NODE="${SMOKE_NPROC_PER_NODE}" \
    MIN_STEPS=50 \
    MAX_STEPS=100 \
    MIN_WALL_SECONDS=0 \
    MAX_WALL_SECONDS=0 \
    bash scripts/slurm/run_cosmos3_candidate_executor_diffusion_smoke_in_allocation.sh \
    > "${SMOKE_ROOT}.console.log" 2>&1
  smoke_rc=$?
  set -e
fi

log "diffusion_smoke_rc=${smoke_rc} console=${SMOKE_ROOT}.console.log"
update_chain_status
if [[ "${smoke_rc}" != "0" ]]; then
  exit "${smoke_rc}"
fi

smoke_ready="$(summary_ready_field "${SMOKE_ROOT}/training_summary.json" smoke)"
log "diffusion_smoke_interface_ready_for_formal=${smoke_ready}"
update_chain_status
if [[ "${smoke_ready}" != "true" ]]; then
  log "diffusion_smoke_interface_gate_failed_not_launching_formal"
  exit 20
fi

if [[ -s "${FORMAL_DIFFUSION_ROOT}/training_summary.json" && -s "${FORMAL_DIFFUSION_ROOT}/checkpoint_final.pt" ]]; then
  log "formal_diffusion_already_complete"
  formal_rc=0
else
  FORMAL_BATCH_SIZE_VALUE="${FORMAL_BATCH_SIZE:-$((128 / FORMAL_NPROC_PER_NODE))}"
  if [[ "${FORMAL_BATCH_SIZE_VALUE}" -lt 1 ]]; then
    FORMAL_BATCH_SIZE_VALUE=1
  fi
  log "launching_formal_diffusion"
  set +e
  srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --gres=gpu:"${FORMAL_GPUS}" --cpus-per-task=8 --mem=96G --time=04:00:00 \
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
    MAX_SAMPLES="${FORMAL_MAX_SAMPLES:-512}" \
    NPROC_PER_NODE="${FORMAL_NPROC_PER_NODE}" \
    FORMAL_MIN_GPUS="${FORMAL_MIN_GPUS}" \
    MIN_STEPS=1000 \
    MAX_STEPS=200000 \
    MIN_WALL_SECONDS="${FORMAL_MIN_WALL_SECONDS}" \
    MAX_WALL_SECONDS="${FORMAL_MAX_WALL_SECONDS}" \
    BATCH_SIZE="${FORMAL_BATCH_SIZE_VALUE}" \
    HIDDEN_DIM="${FORMAL_HIDDEN_DIM:-1024}" \
    NUM_LAYERS="${FORMAL_NUM_LAYERS:-4}" \
    DROPOUT="${FORMAL_DROPOUT:-0.05}" \
    LR="${FORMAL_LR:-0.0002}" \
    WEIGHT_DECAY="${FORMAL_WEIGHT_DECAY:-0.00005}" \
    EVAL_EVERY_STEPS="${FORMAL_EVAL_EVERY_STEPS:-25}" \
    SAVE_EVERY_STEPS="${FORMAL_SAVE_EVERY_STEPS:-1000}" \
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
  write_terminal_marker "formal_diffusion_gate_failed" 21
  exit 21
fi

if live_forbidden_on_current_node; then
  log "formal_diffusion_ready_but_live_forbidden_on_current_node nodes=$(allocation_nodelist || true) forbidden=${LIVE_FORBIDDEN_NODES}"
  write_formal_ready_marker "formal_ready_live_deferred_forbidden_node"
  release_own_allocation_tmux "formal_ready_live_deferred_forbidden_node"
  update_chain_status
  exit 0
fi

run_gated_live_panel
exit "$?"
