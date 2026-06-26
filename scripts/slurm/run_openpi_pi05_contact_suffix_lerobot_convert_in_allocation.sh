#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
OPENPI_ROOT="${OPENPI_ROOT:-/public/home/yanhongru/ICLR2027/openpi}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run OpenPI contact-suffix LeRobot conversion only inside a compute-node srun step.
example=srun --jobid=<held_job_id> --ntasks=1 --cpus-per-task=8 --mem=60G bash scripts/slurm/run_openpi_pi05_contact_suffix_lerobot_convert_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REPO_ID="${REPO_ID:-yanhongru/maniskill_peg733_openpi_contact_suffix16}"
LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/lerobot_home}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix_lerobot_${STAMP}_alloc${SLURM_JOB_ID}}"
RENDER_MANIFEST="${RENDER_MANIFEST:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612/manifest.json}"
MAX_SOURCE_EPISODES="${MAX_SOURCE_EPISODES:-0}"
OFFSETS_BEFORE_INSERT="${OFFSETS_BEFORE_INSERT:-64,48,32,24,16,12,8,4}"
SUFFIX_LENGTH="${SUFFIX_LENGTH:-16}"
STATE_MODE="${STATE_MODE:-qpos8}"
CAMERA_STORAGE="${CAMERA_STORAGE:-image}"
VIDEO_CODEC="${VIDEO_CODEC:-libsvtav1}"
OVERWRITE="${OVERWRITE:-false}"
ALLOW_DESTRUCTIVE_CANONICAL_OVERWRITE="${ALLOW_DESTRUCTIVE_CANONICAL_OVERWRITE:-false}"

if [[ "${OVERWRITE}" == "true" \
  && "${REPO_ID}" == "yanhongru/maniskill_peg733_openpi_contact_suffix16" \
  && "${ALLOW_DESTRUCTIVE_CANONICAL_OVERWRITE}" != "true" ]]; then
  cat >&2 <<'EOF'
refusing_destructive_canonical_overwrite=true
reason=Do not overwrite the canonical contact-suffix LeRobot repo in place. Convert to a fresh repo id, audit it, then switch config or explicitly acknowledge ALLOW_DESTRUCTIVE_CANONICAL_OVERWRITE=true.
EOF
  exit 31
fi

mkdir -p "${OUTPUT_ROOT}" "${LEROBOT_HOME_DIR}"

export HF_LEROBOT_HOME="${LEROBOT_HOME_DIR}"
unset LEROBOT_HOME
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/uv_cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/openpi_uv_env_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export GIT_HTTP_VERSION="${GIT_HTTP_VERSION:-HTTP/1.1}"
export GIT_TERMINAL_PROMPT=0
OPENPI_PYTHON="${OPENPI_PYTHON:-}"

cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=openpi_pi05_peg733_contact_suffix_lerobot_convert_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
openpi_root=${OPENPI_ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
repo_id=${REPO_ID}
hf_lerobot_home=${HF_LEROBOT_HOME}
render_manifest=${RENDER_MANIFEST}
output_root=${OUTPUT_ROOT}
max_source_episodes=${MAX_SOURCE_EPISODES}
offsets_before_insert=${OFFSETS_BEFORE_INSERT}
suffix_length=${SUFFIX_LENGTH}
state_mode=${STATE_MODE}
camera_storage=${CAMERA_STORAGE}
video_codec=${VIDEO_CODEC}
overwrite=${OVERWRITE}
allow_destructive_canonical_overwrite=${ALLOW_DESTRUCTIVE_CANONICAL_OVERWRITE}
proxy_policy=unset_proxy_environment_before_uv
git_http_version=${GIT_HTTP_VERSION}
uv_cache_dir=${UV_CACHE_DIR}
uv_project_environment=${UV_PROJECT_ENVIRONMENT}
uv_link_mode=${UV_LINK_MODE}
openpi_python=${OPENPI_PYTHON:-uv_locked_default}
hdf5_use_file_locking=${HDF5_USE_FILE_LOCKING}
resource_boundary=tmux-held interactive Slurm allocation; no login-node conversion.
method_boundary=Official OpenPI LeRobot contact-suffix data preparation; no custom intermediate model.
EOF

cd "${OPENPI_ROOT}"

if [[ -n "${OPENPI_PYTHON}" ]]; then
  if [[ "${OVERWRITE}" == "true" ]]; then
    "${OPENPI_PYTHON}" "${ROOT}/scripts/openpi/convert_maniskill_peg733_contact_suffix_to_lerobot.py" \
      --args.render-manifest "${RENDER_MANIFEST}" \
      --args.repo-id "${REPO_ID}" \
      --args.output-manifest "${OUTPUT_ROOT}/conversion_manifest.json" \
      --args.max-source-episodes "${MAX_SOURCE_EPISODES}" \
      --args.offsets-before-insert "${OFFSETS_BEFORE_INSERT}" \
      --args.suffix-length "${SUFFIX_LENGTH}" \
      --args.state-mode "${STATE_MODE}" \
      --args.camera-storage "${CAMERA_STORAGE}" \
      --args.video-codec "${VIDEO_CODEC}" \
      --args.overwrite 2>&1 | tee "${OUTPUT_ROOT}/convert.log"
  else
    "${OPENPI_PYTHON}" "${ROOT}/scripts/openpi/convert_maniskill_peg733_contact_suffix_to_lerobot.py" \
      --args.render-manifest "${RENDER_MANIFEST}" \
      --args.repo-id "${REPO_ID}" \
      --args.output-manifest "${OUTPUT_ROOT}/conversion_manifest.json" \
      --args.max-source-episodes "${MAX_SOURCE_EPISODES}" \
      --args.offsets-before-insert "${OFFSETS_BEFORE_INSERT}" \
      --args.suffix-length "${SUFFIX_LENGTH}" \
      --args.state-mode "${STATE_MODE}" \
      --args.camera-storage "${CAMERA_STORAGE}" \
      --args.video-codec "${VIDEO_CODEC}" 2>&1 | tee "${OUTPUT_ROOT}/convert.log"
  fi
else
  if [[ "${OVERWRITE}" == "true" ]]; then
    uv run --frozen python "${ROOT}/scripts/openpi/convert_maniskill_peg733_contact_suffix_to_lerobot.py" \
      --args.render-manifest "${RENDER_MANIFEST}" \
      --args.repo-id "${REPO_ID}" \
      --args.output-manifest "${OUTPUT_ROOT}/conversion_manifest.json" \
      --args.max-source-episodes "${MAX_SOURCE_EPISODES}" \
      --args.offsets-before-insert "${OFFSETS_BEFORE_INSERT}" \
      --args.suffix-length "${SUFFIX_LENGTH}" \
      --args.state-mode "${STATE_MODE}" \
      --args.camera-storage "${CAMERA_STORAGE}" \
      --args.video-codec "${VIDEO_CODEC}" \
      --args.overwrite 2>&1 | tee "${OUTPUT_ROOT}/convert.log"
  else
    uv run --frozen python "${ROOT}/scripts/openpi/convert_maniskill_peg733_contact_suffix_to_lerobot.py" \
      --args.render-manifest "${RENDER_MANIFEST}" \
      --args.repo-id "${REPO_ID}" \
      --args.output-manifest "${OUTPUT_ROOT}/conversion_manifest.json" \
      --args.max-source-episodes "${MAX_SOURCE_EPISODES}" \
      --args.offsets-before-insert "${OFFSETS_BEFORE_INSERT}" \
      --args.suffix-length "${SUFFIX_LENGTH}" \
      --args.state-mode "${STATE_MODE}" \
      --args.camera-storage "${CAMERA_STORAGE}" \
      --args.video-codec "${VIDEO_CODEC}" 2>&1 | tee "${OUTPUT_ROOT}/convert.log"
  fi
fi
