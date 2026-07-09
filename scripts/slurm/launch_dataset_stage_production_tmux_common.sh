#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
DATASET_STAGE="${DATASET_STAGE:?DATASET_STAGE is required}"

case "${DATASET_STAGE}" in
  b_dynamic_production)
    SMOKE_STAGE="b_dynamic_smoke"
    DEFAULT_RUN_GROUP="dynamic_rgb"
    DEFAULT_RUN_NAME="prod01"
    DEFAULT_SESSION="dset_dynamic_rgb_prod01"
    DEFAULT_JOB_NAME="dset_dyn_prod"
    DEFAULT_COUNT="1000"
    DEFAULT_STEPS_PER_EPISODE="300"
    DEFAULT_MAX_EPISODE_STEPS="300"
    DEFAULT_TIME_LIMIT="08:00:00"
    RUNNER="${ROOT}/scripts/slurm/run_dataset_dynamic_rgb_smoke_in_allocation.sh"
    ;;
  c_frozen_dp_production)
    SMOKE_STAGE="c_frozen_dp_smoke"
    DEFAULT_RUN_GROUP="frozen_dp_dynamic"
    DEFAULT_RUN_NAME="prod01"
    DEFAULT_SESSION="dset_frozen_dp_prod01"
    DEFAULT_JOB_NAME="dset_c_prod"
    DEFAULT_COUNT="500"
    DEFAULT_STEPS_PER_EPISODE="300"
    DEFAULT_MAX_EPISODE_STEPS="300"
    DEFAULT_TIME_LIMIT="08:00:00"
    RUNNER="${ROOT}/scripts/slurm/run_dataset_frozen_dp_dynamic_smoke_in_allocation.sh"
    ;;
  d_future_teacher_production)
    SMOKE_STAGE="d_future_teacher_smoke"
    DEFAULT_RUN_GROUP="future_teacher"
    DEFAULT_RUN_NAME="prod01"
    DEFAULT_SESSION="dset_future_teacher_prod01"
    DEFAULT_JOB_NAME="dset_d_prod"
    DEFAULT_COUNT="500"
    DEFAULT_STEPS_PER_EPISODE="300"
    DEFAULT_MAX_EPISODE_STEPS="300"
    DEFAULT_TIME_LIMIT="08:00:00"
    RUNNER="${ROOT}/scripts/slurm/run_dataset_future_frame_teacher_smoke_in_allocation.sh"
    ;;
  e_cosmos_predicted_production)
    SMOKE_STAGE="e_cosmos_predicted_smoke"
    DEFAULT_RUN_GROUP="cosmos_predicted"
    DEFAULT_RUN_NAME="prod01"
    DEFAULT_SESSION="dset_cosmos_pred_prod01"
    DEFAULT_JOB_NAME="dset_e_prod"
    DEFAULT_COUNT="100"
    DEFAULT_STEPS_PER_EPISODE="300"
    DEFAULT_MAX_EPISODE_STEPS="300"
    DEFAULT_TIME_LIMIT="04:00:00"
    RUNNER="${ROOT}/scripts/slurm/run_dataset_cosmos_predicted_coop_smoke_in_allocation.sh"
    ;;
  *)
    echo "refusing_unknown_dataset_production_stage=true" >&2
    echo "dataset_stage=${DATASET_STAGE}" >&2
    exit 40
    ;;
esac

if ! RUN_GROUP=static_rgb RUN_NAME=full01 "${ROOT}/scripts/world_model/require_dataset_static_full_ready.sh"; then
  echo "dataset_production_ready=false" >&2
  echo "reason=a_static_full_not_ready" >&2
  exit 61
fi

if ! RUN_GROUP= RUN_NAME= "${ROOT}/scripts/world_model/require_dataset_class_smoke_approved.sh" "${SMOKE_STAGE}"; then
  echo "dataset_production_ready=false" >&2
  echo "reason=class_smoke_not_approved" >&2
  echo "smoke_stage=${SMOKE_STAGE}" >&2
  exit 62
fi

if ! "${ROOT}/scripts/world_model/dataset_dynamic_adapter_status.sh"; then
  echo "dataset_production_ready=false" >&2
  echo "reason=dynamic_adapter_not_ready" >&2
  exit 63
fi

if ! "${ROOT}/scripts/world_model/audit_dataset_runner_source.sh" "${RUNNER}"; then
  echo "dataset_production_ready=false" >&2
  echo "reason=runner_source_audit_failed" >&2
  echo "runner=${RUNNER}" >&2
  exit 64
fi

RUN_GROUP="${RUN_GROUP:-${DEFAULT_RUN_GROUP}}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
SESSION="${SESSION:-${DEFAULT_SESSION}}"
JOB_NAME="${JOB_NAME:-${DEFAULT_JOB_NAME}}"
PARTITION="${PARTITION:-cpu}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${MEMORY:-8G}"
TIME_LIMIT="${TIME_LIMIT:-${DEFAULT_TIME_LIMIT}}"
COUNT="${COUNT:-${DEFAULT_COUNT}}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-${DEFAULT_STEPS_PER_EPISODE}}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-${DEFAULT_MAX_EPISODE_STEPS}}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_SOURCE_EPISODE_ATTEMPTS="${MAX_SOURCE_EPISODE_ATTEMPTS:-0}"
MAX_ROLLOUT_ATTEMPTS="${MAX_ROLLOUT_ATTEMPTS:-0}"
SCENARIO="${SCENARIO:-constant_lr}"
MOTION_START_STEP="${MOTION_START_STEP:-20}"
MOTION_DURATION_STEPS="${MOTION_DURATION_STEPS:-40}"
MOTION_TRIGGER_MODE="${MOTION_TRIGGER_MODE:-pre_insert_l2}"
MOTION_TRIGGER_THRESHOLD_M="${MOTION_TRIGGER_THRESHOLD_M:-0.20}"
MOTION_TRIGGER_MIN_STEP="${MOTION_TRIGGER_MIN_STEP:-0}"
MIN_TRIGGER_TO_INSERT_STEPS="${MIN_TRIGGER_TO_INSERT_STEPS:-8}"
DELTA_X="${DELTA_X:-0.0}"
DELTA_Y="${DELTA_Y:-0.08}"
DELTA_Z="${DELTA_Z:-0.0}"
MAX_STEP_DELTA_M="${MAX_STEP_DELTA_M:-0.004}"
FUTURE_TAU_STEPS="${FUTURE_TAU_STEPS:-12}"
MAX_ACTION_TRANSLATION="${MAX_ACTION_TRANSLATION:-0.01}"
APPROACH_OFFSET_X="${APPROACH_OFFSET_X:--0.03}"
APPROACH_OFFSET_Y="${APPROACH_OFFSET_Y:-0.0}"
APPROACH_OFFSET_Z="${APPROACH_OFFSET_Z:-0.0}"
RUN_RENDER_CANARY="${RUN_RENDER_CANARY:-true}"
RENDER_CANARY_TIMEOUT="${RENDER_CANARY_TIMEOUT:-3m}"
RENDER_SHADER_PACK="${RENDER_SHADER_PACK:-minimal}"
RENDER_CANARY_API="${RENDER_CANARY_API:-gym}"
REPLAY_SHADER="${REPLAY_SHADER:-minimal}"
FPS="${FPS:-30}"
DATASET_SMOKE_ONLY="${DATASET_SMOKE_ONLY:-false}"
HUMAN_REVIEW_REQUIRED="${HUMAN_REVIEW_REQUIRED:-false}"
LARGE_SCALE_PRODUCTION_ALLOWED="${LARGE_SCALE_PRODUCTION_ALLOWED:-true}"
EXCLUDE_NODES="${EXCLUDE_NODES:-server02,server07,server10,server28,server30,server34,server35,server36,server39,server43,server44,server46,server53,server56,server57,server58,server59,server60,server63}"
NODELIST="${NODELIST:-}"

if [[ "${RUN_GROUP}" =~ p03_|full_pipeline|[0-9]{8}|server[0-9]+|job[0-9]+ ]]; then
  echo "refusing_unreadable_run_group=true" >&2
  echo "run_group=${RUN_GROUP}" >&2
  exit 41
fi
if [[ "${RUN_NAME}" =~ p03_|full_pipeline|[0-9]{8}|server[0-9]+|job[0-9]+ ]]; then
  echo "refusing_unreadable_run_name=true" >&2
  echo "run_name=${RUN_NAME}" >&2
  exit 42
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "refusing_existing_tmux_session=true" >&2
  echo "session=${SESSION}" >&2
  exit 43
fi

OUTPUT_DIR="${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}/${RUN_NAME}"
LOG_DIR="${ROOT}/logs/01_dataset/${RUN_GROUP}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing_existing_output_dir=true" >&2
  echo "output_dir=${OUTPUT_DIR}" >&2
  exit 44
fi

mkdir -p "${LOG_DIR}"

srun_cmd=(
  srun
  --partition="${PARTITION}"
  --job-name="${JOB_NAME}"
  --gres="gpu:${GPUS}"
  --ntasks=1
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEMORY}"
  --time="${TIME_LIMIT}"
)
if [[ -n "${EXCLUDE_NODES}" ]]; then
  srun_cmd+=(--exclude="${EXCLUDE_NODES}")
fi
if [[ -n "${NODELIST}" ]]; then
  srun_cmd+=(--nodelist="${NODELIST}")
fi
srun_cmd+=(bash "${RUNNER}")

inner=$(cat <<EOF
set -euo pipefail
cd "${ROOT}"
export ROOT="${ROOT}"
export DATASET_STAGE="${DATASET_STAGE}"
export RUN_GROUP="${RUN_GROUP}"
export RUN_NAME="${RUN_NAME}"
export COUNT="${COUNT}"
export STEPS_PER_EPISODE="${STEPS_PER_EPISODE}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}"
export NUM_ENVS="${NUM_ENVS}"
export MAX_SOURCE_EPISODE_ATTEMPTS="${MAX_SOURCE_EPISODE_ATTEMPTS}"
export MAX_ROLLOUT_ATTEMPTS="${MAX_ROLLOUT_ATTEMPTS}"
export SCENARIO="${SCENARIO}"
export MOTION_START_STEP="${MOTION_START_STEP}"
export MOTION_DURATION_STEPS="${MOTION_DURATION_STEPS}"
export MOTION_TRIGGER_MODE="${MOTION_TRIGGER_MODE}"
export MOTION_TRIGGER_THRESHOLD_M="${MOTION_TRIGGER_THRESHOLD_M}"
export MOTION_TRIGGER_MIN_STEP="${MOTION_TRIGGER_MIN_STEP}"
export MIN_TRIGGER_TO_INSERT_STEPS="${MIN_TRIGGER_TO_INSERT_STEPS}"
export DELTA_X="${DELTA_X}"
export DELTA_Y="${DELTA_Y}"
export DELTA_Z="${DELTA_Z}"
export MAX_STEP_DELTA_M="${MAX_STEP_DELTA_M}"
export FUTURE_TAU_STEPS="${FUTURE_TAU_STEPS}"
export MAX_ACTION_TRANSLATION="${MAX_ACTION_TRANSLATION}"
export APPROACH_OFFSET_X="${APPROACH_OFFSET_X}"
export APPROACH_OFFSET_Y="${APPROACH_OFFSET_Y}"
export APPROACH_OFFSET_Z="${APPROACH_OFFSET_Z}"
export RUN_RENDER_CANARY="${RUN_RENDER_CANARY}"
export RENDER_CANARY_TIMEOUT="${RENDER_CANARY_TIMEOUT}"
export RENDER_SHADER_PACK="${RENDER_SHADER_PACK}"
export RENDER_CANARY_API="${RENDER_CANARY_API}"
export REPLAY_SHADER="${REPLAY_SHADER}"
export FPS="${FPS}"
export DATASET_SMOKE_ONLY="${DATASET_SMOKE_ONLY}"
export HUMAN_REVIEW_REQUIRED="${HUMAN_REVIEW_REQUIRED}"
export LARGE_SCALE_PRODUCTION_ALLOWED="${LARGE_SCALE_PRODUCTION_ALLOWED}"
export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
export DISPLAY=
export HDF5_USE_FILE_LOCKING=FALSE
echo "launch_timestamp=\$(date -Is)" | tee -a "${LOG_FILE}"
echo "session=${SESSION}" | tee -a "${LOG_FILE}"
echo "dataset_stage=${DATASET_STAGE}" | tee -a "${LOG_FILE}"
echo "job_name=${JOB_NAME}" | tee -a "${LOG_FILE}"
echo "runner=${RUNNER}" | tee -a "${LOG_FILE}"
echo "output_dir=${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "launcher=dataset_stage_production_tmux_common" | tee -a "${LOG_FILE}"
echo "partition=${PARTITION}" | tee -a "${LOG_FILE}"
echo "gpus=${GPUS}" | tee -a "${LOG_FILE}"
echo "cpus_per_task=${CPUS_PER_TASK}" | tee -a "${LOG_FILE}"
echo "memory=${MEMORY}" | tee -a "${LOG_FILE}"
echo "time_limit=${TIME_LIMIT}" | tee -a "${LOG_FILE}"
echo "count=${COUNT}" | tee -a "${LOG_FILE}"
echo "steps_per_episode=${STEPS_PER_EPISODE}" | tee -a "${LOG_FILE}"
echo "max_episode_steps=${MAX_EPISODE_STEPS}" | tee -a "${LOG_FILE}"
echo "max_source_episode_attempts=${MAX_SOURCE_EPISODE_ATTEMPTS}" | tee -a "${LOG_FILE}"
echo "max_rollout_attempts=${MAX_ROLLOUT_ATTEMPTS}" | tee -a "${LOG_FILE}"
echo "scenario=${SCENARIO}" | tee -a "${LOG_FILE}"
echo "motion_start_step=${MOTION_START_STEP}" | tee -a "${LOG_FILE}"
echo "motion_duration_steps=${MOTION_DURATION_STEPS}" | tee -a "${LOG_FILE}"
echo "motion_trigger_mode=${MOTION_TRIGGER_MODE}" | tee -a "${LOG_FILE}"
echo "motion_trigger_threshold_m=${MOTION_TRIGGER_THRESHOLD_M}" | tee -a "${LOG_FILE}"
echo "motion_trigger_min_step=${MOTION_TRIGGER_MIN_STEP}" | tee -a "${LOG_FILE}"
echo "min_trigger_to_insert_steps=${MIN_TRIGGER_TO_INSERT_STEPS}" | tee -a "${LOG_FILE}"
echo "delta_x=${DELTA_X}" | tee -a "${LOG_FILE}"
echo "delta_y=${DELTA_Y}" | tee -a "${LOG_FILE}"
echo "delta_z=${DELTA_Z}" | tee -a "${LOG_FILE}"
echo "max_step_delta_m=${MAX_STEP_DELTA_M}" | tee -a "${LOG_FILE}"
echo "future_tau_steps=${FUTURE_TAU_STEPS}" | tee -a "${LOG_FILE}"
echo "max_action_translation=${MAX_ACTION_TRANSLATION}" | tee -a "${LOG_FILE}"
echo "approach_offset_x=${APPROACH_OFFSET_X}" | tee -a "${LOG_FILE}"
echo "approach_offset_y=${APPROACH_OFFSET_Y}" | tee -a "${LOG_FILE}"
echo "approach_offset_z=${APPROACH_OFFSET_Z}" | tee -a "${LOG_FILE}"
echo "human_review_required=${HUMAN_REVIEW_REQUIRED}" | tee -a "${LOG_FILE}"
echo "dataset_smoke_only=${DATASET_SMOKE_ONLY}" | tee -a "${LOG_FILE}"
echo "large_scale_production_allowed=${LARGE_SCALE_PRODUCTION_ALLOWED}" | tee -a "${LOG_FILE}"
echo "run_render_canary=${RUN_RENDER_CANARY}" | tee -a "${LOG_FILE}"
echo "fps=${FPS}" | tee -a "${LOG_FILE}"
echo "exclude_nodes=${EXCLUDE_NODES}" | tee -a "${LOG_FILE}"
echo "nodelist=${NODELIST}" | tee -a "${LOG_FILE}"
echo "srun_output_start=\$(date -Is)" | tee -a "${LOG_FILE}"
set +e
$(printf '%q ' "${srun_cmd[@]}") 2>&1 | tee -a "${LOG_FILE}"
srun_status=\${PIPESTATUS[0]}
set -e
echo "srun_output_end=\$(date -Is)" | tee -a "${LOG_FILE}"
echo "srun_exit_status=\${srun_status}" | tee -a "${LOG_FILE}"
if [[ "\${srun_status}" -ne 0 ]]; then
  echo "srun_failed=\$(date -Is)" | tee -a "${LOG_FILE}"
  exit "\${srun_status}"
fi
echo "tmux_command_complete=\$(date -Is)" | tee -a "${LOG_FILE}"
EOF
)

tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${inner}")"

echo "launched_tmux_session=${SESSION}"
echo "log_file=${LOG_FILE}"
echo "output_dir=${OUTPUT_DIR}"
echo "launcher=dataset_stage_production_tmux_common"
echo "dataset_stage=${DATASET_STAGE}"
echo "runner=${RUNNER}"
echo "dataset_smoke_only=${DATASET_SMOKE_ONLY}"
echo "large_scale_production_allowed=${LARGE_SCALE_PRODUCTION_ALLOWED}"
