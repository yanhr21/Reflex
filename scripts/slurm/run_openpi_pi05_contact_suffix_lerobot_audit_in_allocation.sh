#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
OPENPI_ROOT="${OPENPI_ROOT:-/public/home/yanhongru/ICLR2027/openpi}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run OpenPI contact-suffix LeRobot audit only inside a compute-node srun step.
example=srun --jobid=<held_job_id> --ntasks=1 --cpus-per-task=8 --mem=40G bash scripts/slurm/run_openpi_pi05_contact_suffix_lerobot_audit_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REPO_ID="${REPO_ID:-yanhongru/maniskill_peg733_openpi_contact_suffix16}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/experiments/world_model_task_rebinding/openpi/lerobot_home/${REPO_ID}}"
CONVERSION_MANIFEST="${CONVERSION_MANIFEST:?set CONVERSION_MANIFEST to the matching clean conversion_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/openpi/contact_suffix_lerobot_audit_${STAMP}_alloc${SLURM_JOB_ID}}"
OPENPI_PYTHON="${OPENPI_PYTHON:-}"
EXPECTED_STATE_DIM="${EXPECTED_STATE_DIM:-8}"
EXPECTED_SOURCE_EPISODES="${EXPECTED_SOURCE_EPISODES:-733}"
EXPECTED_SUFFIX_EPISODES="${EXPECTED_SUFFIX_EPISODES:-5853}"
EXPECTED_EPISODE_LENGTH="${EXPECTED_EPISODE_LENGTH:-16}"
EXPECTED_TOTAL_FRAMES="${EXPECTED_TOTAL_FRAMES:-93648}"
EXPECTED_ACTION_DIM="${EXPECTED_ACTION_DIM:-7}"

mkdir -p "${OUTPUT_ROOT}"

export HF_LEROBOT_HOME="${ROOT}/experiments/world_model_task_rebinding/openpi/lerobot_home"
unset LEROBOT_HOME
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/uv_cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/openpi_uv_env_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export GIT_HTTP_VERSION="${GIT_HTTP_VERSION:-HTTP/1.1}"
export GIT_TERMINAL_PROMPT=0

cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=openpi_pi05_contact_suffix_lerobot_audit_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
openpi_root=${OPENPI_ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
repo_id=${REPO_ID}
dataset_root=${DATASET_ROOT}
conversion_manifest=${CONVERSION_MANIFEST}
output_root=${OUTPUT_ROOT}
openpi_python=${OPENPI_PYTHON:-uv_locked_default}
expected_state_dim=${EXPECTED_STATE_DIM}
expected_source_episodes=${EXPECTED_SOURCE_EPISODES}
expected_suffix_episodes=${EXPECTED_SUFFIX_EPISODES}
expected_episode_length=${EXPECTED_EPISODE_LENGTH}
expected_total_frames=${EXPECTED_TOTAL_FRAMES}
expected_action_dim=${EXPECTED_ACTION_DIM}
uv_cache_dir=${UV_CACHE_DIR}
uv_project_environment=${UV_PROJECT_ENVIRONMENT}
uv_link_mode=${UV_LINK_MODE}
resource_boundary=tmux-held interactive Slurm allocation; no login-node audit.
method_boundary=Contact-suffix LeRobot structural audit only. No model, training, replay, or checkpoint mutation.
EOF

ARGS=(
  "${ROOT}/scripts/openpi/audit_maniskill_peg733_contact_suffix_lerobot.py"
  --args.dataset-root "${DATASET_ROOT}"
  --args.conversion-manifest "${CONVERSION_MANIFEST}"
  --args.output-json "${OUTPUT_ROOT}/audit_summary.json"
  --args.expected-state-dim "${EXPECTED_STATE_DIM}"
  --args.expected-source-episodes "${EXPECTED_SOURCE_EPISODES}"
  --args.expected-suffix-episodes "${EXPECTED_SUFFIX_EPISODES}"
  --args.expected-episode-length "${EXPECTED_EPISODE_LENGTH}"
  --args.expected-total-frames "${EXPECTED_TOTAL_FRAMES}"
  --args.expected-action-dim "${EXPECTED_ACTION_DIM}"
)

cd "${OPENPI_ROOT}"
if [[ -n "${OPENPI_PYTHON}" ]]; then
  "${OPENPI_PYTHON}" "${ARGS[@]}" 2>&1 | tee "${OUTPUT_ROOT}/audit.log"
else
  uv run --frozen python "${ARGS[@]}" 2>&1 | tee "${OUTPUT_ROOT}/audit.log"
fi
