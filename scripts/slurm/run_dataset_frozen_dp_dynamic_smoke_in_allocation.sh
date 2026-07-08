#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-frozen_dp_dynamic}"
RUN_NAME="${RUN_NAME:-smoke01}"
DATASET_SMOKE_ONLY="${DATASET_SMOKE_ONLY:-true}"
HUMAN_REVIEW_REQUIRED="${HUMAN_REVIEW_REQUIRED:-true}"
LARGE_SCALE_PRODUCTION_ALLOWED="${LARGE_SCALE_PRODUCTION_ALLOWED:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${ROOT}/logs/01_dataset/${RUN_GROUP}/${RUN_NAME}.log}"

source "${ROOT}/scripts/world_model/require_dataset_runtime_context.sh"

RUN_RENDER_CANARY="${RUN_RENDER_CANARY:-true}"
RENDER_CANARY_TIMEOUT="${RENDER_CANARY_TIMEOUT:-3m}"
RENDER_SHADER_PACK="${RENDER_SHADER_PACK:-minimal}"
RENDER_CANARY_API="${RENDER_CANARY_API:-gym}"
COUNT="${COUNT:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
SCENARIO="${SCENARIO:-constant_lr}"
MOTION_START_STEP="${MOTION_START_STEP:-120}"
MOTION_DURATION_STEPS="${MOTION_DURATION_STEPS:-150}"
MOTION_TRIGGER_MODE="${MOTION_TRIGGER_MODE:-peg_head_l2}"
MOTION_TRIGGER_THRESHOLD_M="${MOTION_TRIGGER_THRESHOLD_M:-0.12}"
MOTION_TRIGGER_MIN_STEP="${MOTION_TRIGGER_MIN_STEP:-0}"
DELTA_X="${DELTA_X:-0.0}"
DELTA_Y="${DELTA_Y:-0.20}"
DELTA_Z="${DELTA_Z:-0.0}"
MAX_STEP_DELTA_M="${MAX_STEP_DELTA_M:-0.004}"
FPS="${FPS:-30}"
CKPT_PATH="${CKPT_PATH:-${ROOT}/experiments/maniskill/dp_checkpoint/run_90201/checkpoints/best_eval_success_at_end.pt}"

MS_DIR="${ROOT}/deps/ManiSkill_clean"
VENV="${ROOT}/.venv"
ADAPTER="${ROOT}/scripts/world_model/active_dynamic_peg_adapter.py"
COLLECTOR="${ROOT}/scripts/world_model/collect_frozen_dp_dynamic_failure_smoke.py"
CANARY="${ROOT}/scripts/world_model/render_min_canary.py"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

cd "${ROOT}"
export PYTHONPATH="${MS_DIR}/examples/baselines/diffusion_policy:${MS_DIR}:${ROOT}/scripts/training:${ROOT}:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY="${DISPLAY:-}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

MANIFEST="${OUTPUT_DIR}/manifest.txt"

{
  echo "timestamp=$(date -Is)"
  echo "phase=01_dataset"
  echo "dataset_class=C_frozen_dp_dynamic_failure"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "source_paths=${ADAPTER},${COLLECTOR},${CKPT_PATH}"
  echo "controller=frozen_official_diffusion_policy"
  echo "action_contract=pd_ee_delta_pose"
  echo "rgb_required=true"
  echo "human_review_required=${HUMAN_REVIEW_REQUIRED}"
  echo "large_scale_production_allowed=${LARGE_SCALE_PRODUCTION_ALLOWED}"
  echo "method_evidence_allowed=false"
  echo "teacher_evidence_allowed=false"
  echo "allowed_losses=negative_classification,discrepancy,infeasible_no_progress,contrastive"
  echo "disallowed_losses=positive_dp_bc_from_failed_actions,target_assisted_success"
  echo "forbidden_state_intervention_expected=false"
  echo "dataset_smoke_only=${DATASET_SMOKE_ONLY}"
  echo "scenario=${SCENARIO}"
  echo "num_rollouts=${COUNT}"
  echo "max_episode_steps=${MAX_EPISODE_STEPS}"
  echo "ckpt_path=${CKPT_PATH}"
  echo "motion_trigger_mode=${MOTION_TRIGGER_MODE}"
  echo "motion_trigger_threshold_m=${MOTION_TRIGGER_THRESHOLD_M}"
  echo "motion_trigger_min_step=${MOTION_TRIGGER_MIN_STEP}"
  echo "motion_delta_xyz=${DELTA_X},${DELTA_Y},${DELTA_Z}"
  echo "run_render_canary=${RUN_RENDER_CANARY}"
  echo "render_canary_timeout=${RENDER_CANARY_TIMEOUT}"
  echo "render_shader_pack=${RENDER_SHADER_PACK}"
  echo "render_canary_api=${RENDER_CANARY_API}"
  echo "fps=${FPS}"
  echo "dynamic_adapter=${ADAPTER}"
  echo "collector=${COLLECTOR}"
  echo "notes=C frozen DP dynamic smoke only; failed chunks are negative/diagnostic, not positive BC; target motion must not begin during grasp/initial approach."
} | tee "${MANIFEST}"

"${ROOT}/scripts/world_model/dataset_dynamic_adapter_status.sh" | tee -a "${MANIFEST}"

if [[ "${RUN_RENDER_CANARY}" == "true" ]]; then
  echo "dataset_smoke_status=render_canary_in_progress" | tee -a "${MANIFEST}"
  set +e
  timeout -k 15s "${RENDER_CANARY_TIMEOUT}" "${VENV}/bin/python" -u \
    "${CANARY}" \
      --output-dir "${OUTPUT_DIR}/render_canary" \
      --shader-pack "${RENDER_SHADER_PACK}" \
      --render-api "${RENDER_CANARY_API}"
  canary_status="${PIPESTATUS[0]}"
  set -e
  echo "render_canary_exit_code=${canary_status}" | tee -a "${MANIFEST}"
  if [[ "${canary_status}" -ne 0 ]]; then
    echo "dataset_smoke_status=blocked_render_canary_failed_no_collection" | tee -a "${MANIFEST}"
    exit "${canary_status}"
  fi
fi

echo "dataset_smoke_status=frozen_dp_dynamic_collection_in_progress" | tee -a "${MANIFEST}"
"${VENV}/bin/python" -u "${COLLECTOR}" \
  --ckpt-path "${CKPT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --scenario "${SCENARIO}" \
  --num-rollouts "${COUNT}" \
  --max-episode-steps "${MAX_EPISODE_STEPS}" \
  --motion-start-step "${MOTION_START_STEP}" \
  --motion-trigger-mode "${MOTION_TRIGGER_MODE}" \
  --motion-trigger-threshold-m "${MOTION_TRIGGER_THRESHOLD_M}" \
  --motion-trigger-min-step "${MOTION_TRIGGER_MIN_STEP}" \
  --motion-duration-steps "${MOTION_DURATION_STEPS}" \
  --delta-x "${DELTA_X}" \
  --delta-y "${DELTA_Y}" \
  --delta-z "${DELTA_Z}" \
  --max-step-delta-m "${MAX_STEP_DELTA_M}" \
  --fps "${FPS}" \
  --dataset-smoke-only "${DATASET_SMOKE_ONLY}" \
  --human-review-required "${HUMAN_REVIEW_REQUIRED}" \
  --large-scale-production-allowed "${LARGE_SCALE_PRODUCTION_ALLOWED}"

{
  echo "summary=${OUTPUT_DIR}/summary.json"
  echo "videos=${OUTPUT_DIR}/videos"
  echo "trace=${OUTPUT_DIR}/trace/frozen_dp_trace.json"
  echo "status=complete"
} | tee -a "${MANIFEST}"
