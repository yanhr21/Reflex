#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
SESSION="${SESSION:-dset_static_rgb_smoke01}"
JOB_NAME="${JOB_NAME:-dset_rgb_smoke}"
PARTITION="${PARTITION:-gpu}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEMORY="${MEMORY:-8G}"
TIME_LIMIT="${TIME_LIMIT:-00:15:00}"
RUN_GROUP="${RUN_GROUP:-static_rgb}"
RUN_NAME="${RUN_NAME:-smoke01}"
COUNT="${COUNT:-4}"
EPISODE_START="${EPISODE_START:-0}"
NUM_ENVS="${NUM_ENVS:-1}"
RUN_RENDER_CANARY="${RUN_RENDER_CANARY:-true}"
RENDER_CANARY_TIMEOUT="${RENDER_CANARY_TIMEOUT:-3m}"
RENDER_SHADER_PACK="${RENDER_SHADER_PACK:-minimal}"
RENDER_CANARY_API="${RENDER_CANARY_API:-gym}"
REPLAY_SHADER="${REPLAY_SHADER:-minimal}"
VIDEO_FPS="${VIDEO_FPS:-30}"
DATASET_SMOKE_ONLY="${DATASET_SMOKE_ONLY:-true}"
HUMAN_REVIEW_REQUIRED="${HUMAN_REVIEW_REQUIRED:-true}"
LARGE_SCALE_PRODUCTION_ALLOWED="${LARGE_SCALE_PRODUCTION_ALLOWED:-false}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
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
: > "${LOG_FILE}"

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
srun_cmd+=(bash "${ROOT}/scripts/slurm/run_dataset_static_rgb_smoke_in_allocation.sh")

inner=$(cat <<EOF
set -euo pipefail
cd "${ROOT}"
export ROOT="${ROOT}"
export RUN_GROUP="${RUN_GROUP}"
export RUN_NAME="${RUN_NAME}"
export COUNT="${COUNT}"
export EPISODE_START="${EPISODE_START}"
export NUM_ENVS="${NUM_ENVS}"
export RUN_RENDER_CANARY="${RUN_RENDER_CANARY}"
export RENDER_CANARY_TIMEOUT="${RENDER_CANARY_TIMEOUT}"
export RENDER_SHADER_PACK="${RENDER_SHADER_PACK}"
export RENDER_CANARY_API="${RENDER_CANARY_API}"
export REPLAY_SHADER="${REPLAY_SHADER}"
export VIDEO_FPS="${VIDEO_FPS}"
export DATASET_SMOKE_ONLY="${DATASET_SMOKE_ONLY}"
export HUMAN_REVIEW_REQUIRED="${HUMAN_REVIEW_REQUIRED}"
export LARGE_SCALE_PRODUCTION_ALLOWED="${LARGE_SCALE_PRODUCTION_ALLOWED}"
export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
export DISPLAY=
export HDF5_USE_FILE_LOCKING=FALSE
echo "launch_timestamp=\$(date -Is)" | tee -a "${LOG_FILE}"
echo "session=${SESSION}" | tee -a "${LOG_FILE}"
echo "job_name=${JOB_NAME}" | tee -a "${LOG_FILE}"
echo "output_dir=${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "launcher=direct_srun_tmux" | tee -a "${LOG_FILE}"
echo "human_review_required=${HUMAN_REVIEW_REQUIRED}" | tee -a "${LOG_FILE}"
echo "dataset_smoke_only=${DATASET_SMOKE_ONLY}" | tee -a "${LOG_FILE}"
echo "large_scale_production_allowed=${LARGE_SCALE_PRODUCTION_ALLOWED}" | tee -a "${LOG_FILE}"
echo "episode_start=${EPISODE_START}" | tee -a "${LOG_FILE}"
echo "run_render_canary=${RUN_RENDER_CANARY}" | tee -a "${LOG_FILE}"
echo "render_canary_timeout=${RENDER_CANARY_TIMEOUT}" | tee -a "${LOG_FILE}"
echo "render_shader_pack=${RENDER_SHADER_PACK}" | tee -a "${LOG_FILE}"
echo "render_canary_api=${RENDER_CANARY_API}" | tee -a "${LOG_FILE}"
echo "replay_shader=${REPLAY_SHADER}" | tee -a "${LOG_FILE}"
echo "video_fps=${VIDEO_FPS}" | tee -a "${LOG_FILE}"
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
echo "launcher=direct_srun_tmux"
echo "human_review_required=${HUMAN_REVIEW_REQUIRED}"
echo "large_scale_production_allowed=${LARGE_SCALE_PRODUCTION_ALLOWED}"
