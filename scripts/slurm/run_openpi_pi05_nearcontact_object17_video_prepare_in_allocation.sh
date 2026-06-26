#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
OPENPI_ROOT="${OPENPI_ROOT:-/public/home/yanhongru/ICLR2027/openpi}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run OpenPI near-contact data preparation only inside a compute-node srun step.
example=srun --jobid=<held_job_id> --ntasks=1 --cpus-per-task=16 --mem=120G bash scripts/slurm/run_openpi_pi05_nearcontact_object17_video_prepare_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CONFIG_NAME="${CONFIG_NAME:-pi05_maniskill_peg733_contact_suffix16_object17_video_nearcontact_20260626}"
REPO_ID="${REPO_ID:-yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_nearcontact_20260626}"
LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/lerobot_home}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/openpi/object17_video_nearcontact_prepare_${STAMP}_alloc${SLURM_JOB_ID}}"
OFFSETS_BEFORE_INSERT="${OFFSETS_BEFORE_INSERT:-16,12,8,4,2,1}"
SUFFIX_LENGTH="${SUFFIX_LENGTH:-16}"
EXPECTED_STATE_DIM="${EXPECTED_STATE_DIM:-17}"
EXPECTED_ACTION_DIM="${EXPECTED_ACTION_DIM:-7}"
EXPECTED_SUFFIX_EPISODES="${EXPECTED_SUFFIX_EPISODES:-4375}"
EXPECTED_TOTAL_FRAMES="${EXPECTED_TOTAL_FRAMES:-70000}"
CAMERA_STORAGE="${CAMERA_STORAGE:-video}"
VIDEO_CODEC="${VIDEO_CODEC:-h264}"
OPENPI_PYTHON="${OPENPI_PYTHON:-}"
OVERWRITE="${OVERWRITE:-true}"
SKIP_CONVERT="${SKIP_CONVERT:-false}"

mkdir -p "${OUTPUT_ROOT}" "${LEROBOT_HOME_DIR}"

cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=openpi_pi05_nearcontact_object17_video_prepare_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
openpi_root=${OPENPI_ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
config_name=${CONFIG_NAME}
repo_id=${REPO_ID}
output_root=${OUTPUT_ROOT}
lerobot_home_dir=${LEROBOT_HOME_DIR}
offsets_before_insert=${OFFSETS_BEFORE_INSERT}
suffix_length=${SUFFIX_LENGTH}
expected_state_dim=${EXPECTED_STATE_DIM}
expected_action_dim=${EXPECTED_ACTION_DIM}
expected_suffix_episodes=${EXPECTED_SUFFIX_EPISODES}
expected_total_frames=${EXPECTED_TOTAL_FRAMES}
camera_storage=${CAMERA_STORAGE}
video_codec=${VIDEO_CODEC}
overwrite=${OVERWRITE}
skip_convert=${SKIP_CONVERT}
openpi_python=${OPENPI_PYTHON:-uv_locked_default}
resource_boundary=tmux-held interactive Slurm allocation; no login-node conversion/audit/norm stats.
method_boundary=OpenPI/pi0.5 data preparation only. Same official LeRobot/OpenPI fields; no scorer, DP controller, VAE, MLP, diffusion executor, or custom policy model.
EOF

convert_out="${OUTPUT_ROOT}/convert"
audit_out="${OUTPUT_ROOT}/audit"
norm_out="${OUTPUT_ROOT}/norm_stats"

if [[ "${SKIP_CONVERT}" != "true" ]]; then
  REPO_ID="${REPO_ID}" \
  STATE_MODE=object_state17 \
  CAMERA_STORAGE="${CAMERA_STORAGE}" \
  VIDEO_CODEC="${VIDEO_CODEC}" \
  OFFSETS_BEFORE_INSERT="${OFFSETS_BEFORE_INSERT}" \
  SUFFIX_LENGTH="${SUFFIX_LENGTH}" \
  OVERWRITE="${OVERWRITE}" \
  LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR}" \
  OUTPUT_ROOT="${convert_out}" \
  OPENPI_PYTHON="${OPENPI_PYTHON}" \
    bash "${ROOT}/scripts/slurm/run_openpi_pi05_contact_suffix_lerobot_convert_in_allocation.sh"
elif [[ ! -f "${convert_out}/conversion_manifest.json" ]]; then
  echo "skip_convert_requested_but_missing_manifest=${convert_out}/conversion_manifest.json" >&2
  exit 31
fi

REPO_ID="${REPO_ID}" \
DATASET_ROOT="${LEROBOT_HOME_DIR}/${REPO_ID}" \
CONVERSION_MANIFEST="${convert_out}/conversion_manifest.json" \
OUTPUT_ROOT="${audit_out}" \
OPENPI_PYTHON="${OPENPI_PYTHON}" \
EXPECTED_STATE_DIM="${EXPECTED_STATE_DIM}" \
EXPECTED_ACTION_DIM="${EXPECTED_ACTION_DIM}" \
EXPECTED_SUFFIX_EPISODES="${EXPECTED_SUFFIX_EPISODES}" \
EXPECTED_TOTAL_FRAMES="${EXPECTED_TOTAL_FRAMES}" \
EXPECTED_EPISODE_LENGTH="${SUFFIX_LENGTH}" \
  bash "${ROOT}/scripts/slurm/run_openpi_pi05_contact_suffix_lerobot_audit_in_allocation.sh"

run_python() {
  if [[ -n "${OPENPI_PYTHON}" ]]; then
    "${OPENPI_PYTHON}" "$@"
  else
    cd "${OPENPI_ROOT}"
    uv run --frozen python "$@"
  fi
}

mkdir -p "${norm_out}"
run_python "${ROOT}/scripts/openpi/compute_lerobot_state_action_norm_stats.py" \
  --dataset-root "${LEROBOT_HOME_DIR}/${REPO_ID}" \
  --config-name "${CONFIG_NAME}" \
  --repo-id "${REPO_ID}" \
  --expected-state-dim "${EXPECTED_STATE_DIM}" \
  --expected-action-dim "${EXPECTED_ACTION_DIM}" \
  --expected-total-frames "${EXPECTED_TOTAL_FRAMES}" \
  --output-summary "${norm_out}/norm_stats_fallback_summary.json" \
  2>&1 | tee "${norm_out}/norm_stats.log"

cat > "${OUTPUT_ROOT}/prepare_summary.json" <<EOF
{
  "schema": "openpi_pi05_nearcontact_object17_video_prepare_summary_v1",
  "config_name": "${CONFIG_NAME}",
  "repo_id": "${REPO_ID}",
  "conversion_manifest": "${convert_out}/conversion_manifest.json",
  "audit_summary": "${audit_out}/audit_summary.json",
  "norm_stats_summary": "${norm_out}/norm_stats_fallback_summary.json",
  "norm_stats_json": "${OPENPI_ROOT}/assets/${CONFIG_NAME}/${REPO_ID}/norm_stats.json",
  "offsets_before_insert": "${OFFSETS_BEFORE_INSERT}",
  "expected_suffix_episodes": ${EXPECTED_SUFFIX_EPISODES},
  "expected_total_frames": ${EXPECTED_TOTAL_FRAMES},
  "slurm_job_id": "${SLURM_JOB_ID}",
  "slurm_step_id": "${SLURM_STEP_ID}",
  "host": "$(hostname)",
  "passed": true
}
EOF
