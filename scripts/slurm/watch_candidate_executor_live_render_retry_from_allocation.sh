#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
ALLOC_JOB_ID="${ALLOC_JOB_ID:-${SLURM_JOB_ID:-}}"
if [[ -z "${ALLOC_JOB_ID}" ]]; then
  echo "missing_allocation_job_id=true" >&2
  echo "reason=Run this watcher from a tmux-held salloc shell or set ALLOC_JOB_ID." >&2
  exit 30
fi

FORMAL_ROOT="${FORMAL_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_4gpu_alloc131662_diffusion_bestgate_stable_smokecfg}"
SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-1500}"
SAMPLE_INDICES="${SAMPLE_INDICES:-0,1,3,4}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
CANDIDATE_OUTCOME_SCORER_CHECKPOINT="${CANDIDATE_OUTCOME_SCORER_CHECKPOINT:-}"
CANDIDATE_OUTCOME_SCORER_SUMMARY="${CANDIDATE_OUTCOME_SCORER_SUMMARY:-}"
CANDIDATE_OUTCOME_SCORER_DP_MARGIN="${CANDIDATE_OUTCOME_SCORER_DP_MARGIN:-0.0}"
CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA="${CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA:--1000000000.0}"
CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB="${CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB:-0.0}"
CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB="${CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB:-0.0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
NODE_TAG="${SLURM_JOB_NODELIST:-unknown_node}"
NODE_TAG="${NODE_TAG//[^A-Za-z0-9_.-]/_}"

LOG="${LOG:-${FORMAL_ROOT}/candidate_live_render_retry_${STAMP}_alloc${ALLOC_JOB_ID}_${NODE_TAG}.log}"
CANARY_DIR="${CANARY_DIR:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/render_canary_candidate_live_retry_${STAMP}_alloc${ALLOC_JOB_ID}_${NODE_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SFT_ROOT}/live_receding_candidate_executor_iter1500_renderretry_${STAMP}_alloc${ALLOC_JOB_ID}_${NODE_TAG}_samples${SAMPLE_INDICES//,/_}}"

if [[ "${SET_NVIDIA_VK_ICD:-false}" == "true" ]]; then
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
else
  unset VK_ICD_FILENAMES
fi
if [[ -n "${DISPLAY:-}" ]]; then
  export DISPLAY
else
  unset DISPLAY
fi

mkdir -p "${FORMAL_ROOT}"

log() {
  echo "$(date -Is) $*" | tee -a "${LOG}"
}

log "candidate_live_render_retry_start allocation=${ALLOC_JOB_ID} node=${SLURM_JOB_NODELIST:-unknown}"
log "boundary=compute work runs only through srun inside a tmux-held allocation; render canary must pass before gated live panel"
log "canary_dir=${CANARY_DIR}"
log "output_root=${OUTPUT_ROOT}"
log "sample_indices=${SAMPLE_INDICES}"
log "action_exec_horizon=${ACTION_EXEC_HORIZON:-8}"
log "candidate_outcome_scorer_checkpoint=${CANDIDATE_OUTCOME_SCORER_CHECKPOINT:-none}"
log "candidate_outcome_scorer_summary=${CANDIDATE_OUTCOME_SCORER_SUMMARY:-none}"
log "candidate_outcome_scorer_dp_margin=${CANDIDATE_OUTCOME_SCORER_DP_MARGIN}"
log "candidate_outcome_scorer_min_progress_delta=${CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA}"
log "candidate_outcome_scorer_min_continuable_prob=${CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB}"
log "candidate_outcome_scorer_min_inserted_prob=${CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB}"

canary_env=(
  "ROOT=${ROOT}"
  "CANARY_DIR=${CANARY_DIR}"
)
if [[ -n "${DISPLAY:-}" ]]; then
  canary_env+=("DISPLAY=${DISPLAY}")
fi
if [[ -n "${VK_ICD_FILENAMES:-}" ]]; then
  canary_env+=("VK_ICD_FILENAMES=${VK_ICD_FILENAMES}")
fi

set +e
srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:15:00 \
  env "${canary_env[@]}" \
  bash -lc '
    set -euo pipefail
    cd "${ROOT}"
    timeout 600 .venv/bin/python scripts/world_model/render_min_canary.py \
      --output-dir "${CANARY_DIR}" \
      --shader-pack default \
      --width 256 \
      --height 256
  ' >> "${LOG}" 2>&1
canary_rc=$?
set -e
log "render_canary_rc=${canary_rc}"
if [[ "${canary_rc}" -ne 0 ]]; then
  {
    echo "date=$(date -Is)"
    echo "status=render_canary_failed"
    echo "allocation=${ALLOC_JOB_ID}"
    echo "node=${SLURM_JOB_NODELIST:-unknown}"
    echo "canary_rc=${canary_rc}"
    echo "canary_dir=${CANARY_DIR}"
    echo "log=${LOG}"
  } > "${FORMAL_ROOT}/candidate_live_render_retry_${STAMP}_alloc${ALLOC_JOB_ID}_${NODE_TAG}.terminal"
  exit "${canary_rc}"
fi

log "launching_gated_candidate_executor_live_panel"
live_env=(
  "ROOT=${ROOT}"
  "FORMAL_ROOT=${FORMAL_ROOT}"
  "SFT_ROOT=${SFT_ROOT}"
  "CHECKPOINT_ITER=${CHECKPOINT_ITER}"
  "SAMPLE_INDICES=${SAMPLE_INDICES}"
  "MAX_SAMPLES=${MAX_SAMPLES}"
  "OUTPUT_ROOT=${OUTPUT_ROOT}"
  "ACTION_EXEC_HORIZON=${ACTION_EXEC_HORIZON:-8}"
  "LIVE_PROGRESS_INTERVAL=${LIVE_PROGRESS_INTERVAL:-25}"
  "REQUIRE_DIFFUSION_GENERATOR=${REQUIRE_DIFFUSION_GENERATOR:-true}"
  "CANDIDATE_OUTCOME_SCORER_CHECKPOINT=${CANDIDATE_OUTCOME_SCORER_CHECKPOINT}"
  "CANDIDATE_OUTCOME_SCORER_SUMMARY=${CANDIDATE_OUTCOME_SCORER_SUMMARY}"
  "CANDIDATE_OUTCOME_SCORER_DP_MARGIN=${CANDIDATE_OUTCOME_SCORER_DP_MARGIN}"
  "CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA=${CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA}"
  "CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB=${CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB}"
  "CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB=${CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB}"
  "SET_NVIDIA_VK_ICD=${SET_NVIDIA_VK_ICD:-false}"
)
if [[ -n "${DISPLAY:-}" ]]; then
  live_env+=("DISPLAY=${DISPLAY}")
fi
if [[ -n "${VK_ICD_FILENAMES:-}" ]]; then
  live_env+=("VK_ICD_FILENAMES=${VK_ICD_FILENAMES}")
fi
set +e
srun --overlap --jobid="${ALLOC_JOB_ID}" --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 \
  env "${live_env[@]}" \
    bash scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh \
  >> "${LOG}" 2>&1
live_rc=$?
set -e
log "gated_live_rc=${live_rc}"

if [[ -f "${OUTPUT_ROOT}/live_receding_panel_summary.json" ]]; then
  log "live_summary=${OUTPUT_ROOT}/live_receding_panel_summary.json"
fi

if [[ "${live_rc}" -eq 0 ]]; then
  {
    echo "date=$(date -Is)"
    echo "status=live_finished"
    echo "allocation=${ALLOC_JOB_ID}"
    echo "node=${SLURM_JOB_NODELIST:-unknown}"
    echo "live_rc=${live_rc}"
    echo "output_root=${OUTPUT_ROOT}"
    echo "log=${LOG}"
  } > "${FORMAL_ROOT}/candidate_live_render_retry_${STAMP}_alloc${ALLOC_JOB_ID}_${NODE_TAG}.done"
else
  {
    echo "date=$(date -Is)"
    echo "status=live_failed"
    echo "allocation=${ALLOC_JOB_ID}"
    echo "node=${SLURM_JOB_NODELIST:-unknown}"
    echo "live_rc=${live_rc}"
    echo "output_root=${OUTPUT_ROOT}"
    echo "log=${LOG}"
  } > "${FORMAL_ROOT}/candidate_live_render_retry_${STAMP}_alloc${ALLOC_JOB_ID}_${NODE_TAG}.terminal"
fi
exit "${live_rc}"
