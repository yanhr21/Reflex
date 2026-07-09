#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-future_teacher}"
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
COUNT="${COUNT:-4}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
EPISODE_START="${EPISODE_START:-0}"
MAX_SOURCE_EPISODE_ATTEMPTS="${MAX_SOURCE_EPISODE_ATTEMPTS:-0}"
SCENARIO="${SCENARIO:-constant_lr}"
MOTION_START_STEP="${MOTION_START_STEP:-120}"
MOTION_DURATION_STEPS="${MOTION_DURATION_STEPS:-150}"
MOTION_TRIGGER_MODE="${MOTION_TRIGGER_MODE:-pre_insert_l2}"
MOTION_TRIGGER_THRESHOLD_M="${MOTION_TRIGGER_THRESHOLD_M:-0.20}"
MOTION_TRIGGER_MIN_STEP="${MOTION_TRIGGER_MIN_STEP:-0}"
MIN_TRIGGER_TO_INSERT_STEPS="${MIN_TRIGGER_TO_INSERT_STEPS:-8}"
DELTA_X="${DELTA_X:-0.0}"
DELTA_Y="${DELTA_Y:-0.20}"
DELTA_Z="${DELTA_Z:-0.0}"
MAX_STEP_DELTA_M="${MAX_STEP_DELTA_M:-0.004}"
FUTURE_TAU_STEPS="${FUTURE_TAU_STEPS:-12}"
FUTURE_RESIDUAL_GAIN="${FUTURE_RESIDUAL_GAIN:-0.5}"
MAX_FUTURE_RESIDUAL_M="${MAX_FUTURE_RESIDUAL_M:-0.03}"
PEG_DISTURB_FORCE_X="${PEG_DISTURB_FORCE_X:-0.0}"
PEG_DISTURB_FORCE_Y="${PEG_DISTURB_FORCE_Y:--25.0}"
PEG_DISTURB_FORCE_Z="${PEG_DISTURB_FORCE_Z:-0.0}"
PEG_DISTURB_DURATION_STEPS="${PEG_DISTURB_DURATION_STEPS:-18}"
FPS="${FPS:-30}"

MS_DIR="${ROOT}/deps/ManiSkill_clean"
VENV="${ROOT}/.venv"
ADAPTER="${ROOT}/scripts/world_model/active_dynamic_peg_adapter.py"
DEMO_COLLECTOR="${ROOT}/scripts/world_model/collect_dynamic_demo_action_smoke.py"
CANARY="${ROOT}/scripts/world_model/render_min_canary.py"
SOURCE_H5="${SOURCE_H5:-${ROOT}/data/official_replay/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5}"
SOURCE_JSON="${SOURCE_JSON:-${ROOT}/data/official_replay/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.json}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

cd "${ROOT}"
export PYTHONPATH="${MS_DIR}:${ROOT}:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY="${DISPLAY:-}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

MANIFEST="${OUTPUT_DIR}/manifest.txt"

{
  echo "timestamp=$(date -Is)"
  echo "phase=01_dataset"
  echo "dataset_class=D_future_frame_cooperation_teacher"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "source_paths=${ADAPTER},${DEMO_COLLECTOR},${SOURCE_H5},${SOURCE_JSON}"
  echo "controller=official_demo_action_gt_future_residual"
  echo "action_contract=pd_ee_delta_pose"
  echo "rgb_required=true"
  echo "human_review_required=${HUMAN_REVIEW_REQUIRED}"
  echo "large_scale_production_allowed=${LARGE_SCALE_PRODUCTION_ALLOWED}"
  echo "method_evidence_allowed=false"
  echo "teacher_evidence_allowed=true"
  echo "allowed_losses=adapter_residual,moving_frame_conditioning,phase_timing,relative_velocity_at_contact"
  echo "disallowed_losses=deployed_method_success_claim,hidden_future_controller"
  echo "forbidden_state_intervention_expected=false"
  echo "dataset_smoke_only=${DATASET_SMOKE_ONLY}"
  echo "scenario=${SCENARIO}"
  echo "num_rollouts=${COUNT}"
  echo "episode_start=${EPISODE_START}"
  echo "max_source_episode_attempts=${MAX_SOURCE_EPISODE_ATTEMPTS}"
  echo "max_episode_steps=${MAX_EPISODE_STEPS}"
  echo "source_h5=${SOURCE_H5}"
  echo "source_json=${SOURCE_JSON}"
  echo "motion_trigger_mode=${MOTION_TRIGGER_MODE}"
  echo "motion_trigger_threshold_m=${MOTION_TRIGGER_THRESHOLD_M}"
  echo "motion_trigger_min_step=${MOTION_TRIGGER_MIN_STEP}"
  echo "min_trigger_to_insert_steps=${MIN_TRIGGER_TO_INSERT_STEPS}"
  echo "motion_delta_xyz=${DELTA_X},${DELTA_Y},${DELTA_Z}"
  echo "teacher_future_target_source=ground_truth_future_motion_plan"
  echo "teacher_action_adapter=official_demo_actions_plus_gt_future_residual"
  echo "future_tau_steps=${FUTURE_TAU_STEPS}"
  echo "future_residual_gain=${FUTURE_RESIDUAL_GAIN}"
  echo "max_future_residual_m=${MAX_FUTURE_RESIDUAL_M}"
  echo "peg_disturb_force_xyz=${PEG_DISTURB_FORCE_X},${PEG_DISTURB_FORCE_Y},${PEG_DISTURB_FORCE_Z}"
  echo "peg_disturb_duration_steps=${PEG_DISTURB_DURATION_STEPS}"
  echo "run_render_canary=${RUN_RENDER_CANARY}"
  echo "render_canary_timeout=${RENDER_CANARY_TIMEOUT}"
  echo "render_shader_pack=${RENDER_SHADER_PACK}"
  echo "render_canary_api=${RENDER_CANARY_API}"
  echo "fps=${FPS}"
  echo "dynamic_adapter=${ADAPTER}"
  echo "collector=${DEMO_COLLECTOR}"
  echo "notes=D future-frame teacher smoke uses official demo actions plus a bounded ground-truth future target residual; teacher-only data generation."
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

echo "dataset_smoke_status=future_teacher_collection_in_progress" | tee -a "${MANIFEST}"
"${VENV}/bin/python" -u "${DEMO_COLLECTOR}" \
  --dataset-class D_future_frame_cooperation_teacher \
  --output-dir "${OUTPUT_DIR}" \
  --source-h5 "${SOURCE_H5}" \
  --source-json "${SOURCE_JSON}" \
  --scenario "${SCENARIO}" \
  --episode-start "${EPISODE_START}" \
  --num-episodes "${COUNT}" \
  --max-source-episode-attempts "${MAX_SOURCE_EPISODE_ATTEMPTS}" \
  --max-episode-steps "${MAX_EPISODE_STEPS}" \
  --motion-start-step "${MOTION_START_STEP}" \
  --motion-trigger-mode "${MOTION_TRIGGER_MODE}" \
  --motion-trigger-threshold-m "${MOTION_TRIGGER_THRESHOLD_M}" \
  --motion-trigger-min-step "${MOTION_TRIGGER_MIN_STEP}" \
  --min-trigger-to-insert-steps "${MIN_TRIGGER_TO_INSERT_STEPS}" \
  --motion-duration-steps "${MOTION_DURATION_STEPS}" \
  --delta-x "${DELTA_X}" \
  --delta-y "${DELTA_Y}" \
  --delta-z "${DELTA_Z}" \
  --max-step-delta-m "${MAX_STEP_DELTA_M}" \
  --future-tau-steps "${FUTURE_TAU_STEPS}" \
  --future-residual-gain "${FUTURE_RESIDUAL_GAIN}" \
  --max-future-residual-m "${MAX_FUTURE_RESIDUAL_M}" \
  --peg-disturb-force-x "${PEG_DISTURB_FORCE_X}" \
  --peg-disturb-force-y "${PEG_DISTURB_FORCE_Y}" \
  --peg-disturb-force-z "${PEG_DISTURB_FORCE_Z}" \
  --peg-disturb-duration-steps "${PEG_DISTURB_DURATION_STEPS}" \
  --fps "${FPS}" \
  --dataset-smoke-only "${DATASET_SMOKE_ONLY}" \
  --human-review-required "${HUMAN_REVIEW_REQUIRED}" \
  --large-scale-production-allowed "${LARGE_SCALE_PRODUCTION_ALLOWED}"

{
  echo "summary=${OUTPUT_DIR}/summary.json"
  echo "videos=${OUTPUT_DIR}/videos"
  echo "trace=${OUTPUT_DIR}/trace/demo_action_trace.json"
  echo "status=complete"
} | tee -a "${MANIFEST}"
