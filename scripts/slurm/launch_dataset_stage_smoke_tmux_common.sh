#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
DATASET_STAGE="${DATASET_STAGE:?DATASET_STAGE is required}"

case "${DATASET_STAGE}" in
  b_dynamic_smoke)
    DEFAULT_RUN_GROUP="dynamic_rgb"
    DEFAULT_RUN_NAME="smoke01"
    DEFAULT_SESSION="dset_dynamic_rgb_smoke01"
    DEFAULT_JOB_NAME="dset_dyn_rgb"
    DEFAULT_COUNT="1"
    DEFAULT_STEPS_PER_EPISODE="300"
    DEFAULT_MAX_EPISODE_STEPS="300"
    RUNNER="${ROOT}/scripts/slurm/run_dataset_dynamic_rgb_smoke_in_allocation.sh"
    ;;
  c_frozen_dp_smoke)
    DEFAULT_RUN_GROUP="frozen_dp_dynamic"
    DEFAULT_RUN_NAME="smoke01"
    DEFAULT_SESSION="dset_frozen_dp_smoke01"
    DEFAULT_JOB_NAME="dset_frz_dp"
    DEFAULT_COUNT="1"
    DEFAULT_STEPS_PER_EPISODE="300"
    DEFAULT_MAX_EPISODE_STEPS="300"
    RUNNER="${ROOT}/scripts/slurm/run_dataset_frozen_dp_dynamic_smoke_in_allocation.sh"
    ;;
  d_future_teacher_smoke)
    DEFAULT_RUN_GROUP="future_teacher"
    DEFAULT_RUN_NAME="smoke01"
    DEFAULT_SESSION="dset_future_teacher_smoke01"
    DEFAULT_JOB_NAME="dset_fteach"
    DEFAULT_COUNT="4"
    DEFAULT_STEPS_PER_EPISODE="300"
    DEFAULT_MAX_EPISODE_STEPS="300"
    RUNNER="${ROOT}/scripts/slurm/run_dataset_future_frame_teacher_smoke_in_allocation.sh"
    ;;
  e_cosmos_predicted_smoke)
    DEFAULT_RUN_GROUP="cosmos_predicted"
    DEFAULT_RUN_NAME="smoke01"
    DEFAULT_SESSION="dset_cosmos_pred_smoke01"
    DEFAULT_JOB_NAME="dset_cosprd"
    DEFAULT_COUNT="4"
    DEFAULT_STEPS_PER_EPISODE="300"
    DEFAULT_MAX_EPISODE_STEPS="300"
    RUNNER="${ROOT}/scripts/slurm/run_dataset_cosmos_predicted_coop_smoke_in_allocation.sh"
    ;;
  *)
    echo "refusing_unknown_dataset_stage=true" >&2
    echo "dataset_stage=${DATASET_STAGE}" >&2
    exit 40
    ;;
esac

"${ROOT}/scripts/world_model/require_dataset_stage_ready.sh" "${DATASET_STAGE}"

RUN_GROUP="${RUN_GROUP:-${DEFAULT_RUN_GROUP}}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
SESSION="${SESSION:-${DEFAULT_SESSION}}"
JOB_NAME="${JOB_NAME:-${DEFAULT_JOB_NAME}}"
PARTITION="${PARTITION:-cpu}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${MEMORY:-8G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
COUNT="${COUNT:-${DEFAULT_COUNT}}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-${DEFAULT_STEPS_PER_EPISODE}}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-${DEFAULT_MAX_EPISODE_STEPS}}"
NUM_ENVS="${NUM_ENVS:-1}"
RUN_RENDER_CANARY="${RUN_RENDER_CANARY:-true}"
RENDER_CANARY_TIMEOUT="${RENDER_CANARY_TIMEOUT:-3m}"
RENDER_SHADER_PACK="${RENDER_SHADER_PACK:-minimal}"
RENDER_CANARY_API="${RENDER_CANARY_API:-gym}"
REPLAY_SHADER="${REPLAY_SHADER:-minimal}"
FPS="${FPS:-30}"
DATASET_SMOKE_ONLY="${DATASET_SMOKE_ONLY:-true}"
HUMAN_REVIEW_REQUIRED="${HUMAN_REVIEW_REQUIRED:-true}"
LARGE_SCALE_PRODUCTION_ALLOWED="${LARGE_SCALE_PRODUCTION_ALLOWED:-false}"
EXCLUDE_NODES="${EXCLUDE_NODES:-server28,server34,server36,server39,server43,server44,server46,server53,server58,server60,server63}"
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
export SCENARIO="${SCENARIO:-}"
export MOTION_START_STEP="${MOTION_START_STEP:-}"
export MOTION_DURATION_STEPS="${MOTION_DURATION_STEPS:-}"
export MOTION_TRIGGER_MODE="${MOTION_TRIGGER_MODE:-}"
export MOTION_TRIGGER_THRESHOLD_M="${MOTION_TRIGGER_THRESHOLD_M:-}"
export MOTION_TRIGGER_MIN_STEP="${MOTION_TRIGGER_MIN_STEP:-}"
export DELTA_X="${DELTA_X:-}"
export DELTA_Y="${DELTA_Y:-}"
export DELTA_Z="${DELTA_Z:-}"
export MAX_STEP_DELTA_M="${MAX_STEP_DELTA_M:-}"
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
echo "launcher=dataset_stage_smoke_tmux_common" | tee -a "${LOG_FILE}"
echo "partition=${PARTITION}" | tee -a "${LOG_FILE}"
echo "gpus=${GPUS}" | tee -a "${LOG_FILE}"
echo "cpus_per_task=${CPUS_PER_TASK}" | tee -a "${LOG_FILE}"
echo "memory=${MEMORY}" | tee -a "${LOG_FILE}"
echo "time_limit=${TIME_LIMIT}" | tee -a "${LOG_FILE}"
echo "human_review_required=${HUMAN_REVIEW_REQUIRED}" | tee -a "${LOG_FILE}"
echo "count=${COUNT}" | tee -a "${LOG_FILE}"
echo "steps_per_episode=${STEPS_PER_EPISODE}" | tee -a "${LOG_FILE}"
echo "max_episode_steps=${MAX_EPISODE_STEPS}" | tee -a "${LOG_FILE}"
echo "scenario=${SCENARIO:-default}" | tee -a "${LOG_FILE}"
echo "dataset_smoke_only=${DATASET_SMOKE_ONLY}" | tee -a "${LOG_FILE}"
echo "large_scale_production_allowed=${LARGE_SCALE_PRODUCTION_ALLOWED}" | tee -a "${LOG_FILE}"
echo "run_render_canary=${RUN_RENDER_CANARY}" | tee -a "${LOG_FILE}"
echo "render_canary_timeout=${RENDER_CANARY_TIMEOUT}" | tee -a "${LOG_FILE}"
echo "render_shader_pack=${RENDER_SHADER_PACK}" | tee -a "${LOG_FILE}"
echo "render_canary_api=${RENDER_CANARY_API}" | tee -a "${LOG_FILE}"
echo "replay_shader=${REPLAY_SHADER}" | tee -a "${LOG_FILE}"
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
echo "launcher=dataset_stage_smoke_tmux_common"
echo "dataset_stage=${DATASET_STAGE}"
echo "runner=${RUNNER}"
echo "human_review_required=${HUMAN_REVIEW_REQUIRED}"
echo "large_scale_production_allowed=${LARGE_SCALE_PRODUCTION_ALLOWED}"
