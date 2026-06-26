#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
OPENPI_ROOT="${OPENPI_ROOT:-/public/home/yanhongru/ICLR2027/openpi}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run OpenPI object17 post-rebuild debug only inside a compute-node srun step.
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CONFIG_NAME="${CONFIG_NAME:-pi05_maniskill_peg733_contact_suffix16_object17_clean_20260626}"
REPO_ID="${REPO_ID:-yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_clean_20260626}"
LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/lerobot_home}"
DATASET_ROOT="${DATASET_ROOT:-${LEROBOT_HOME_DIR}/${REPO_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/openpi/object17_clean_postrebuild_debug_${STAMP}_alloc${SLURM_JOB_ID}}"
OPENPI_PYTHON="${OPENPI_PYTHON:-}"
FIRST_BATCH_TIMEOUT_SECONDS="${FIRST_BATCH_TIMEOUT_SECONDS:-900}"

mkdir -p "${OUTPUT_ROOT}"

export HF_LEROBOT_HOME="${LEROBOT_HOME_DIR}"
unset LEROBOT_HOME
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${ROOT}/experiments/world_model_task_rebinding/openpi/openpi_data_home}"
export HF_HOME="${HF_HOME:-/tmp/hf_home_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets_cache_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/uv_cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/openpi_uv_env_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export WANDB_MODE="${WANDB_MODE:-offline}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export GIT_HTTP_VERSION="${GIT_HTTP_VERSION:-HTTP/1.1}"
export GIT_TERMINAL_PROMPT=0

cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=openpi_pi05_object17_clean_postrebuild_debug_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
openpi_root=${OPENPI_ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
config_name=${CONFIG_NAME}
repo_id=${REPO_ID}
dataset_root=${DATASET_ROOT}
output_root=${OUTPUT_ROOT}
openpi_python=${OPENPI_PYTHON:-uv_frozen_default}
hf_home=${HF_HOME}
hf_datasets_cache=${HF_DATASETS_CACHE}
xdg_cache_home=${XDG_CACHE_HOME}
first_batch_timeout_seconds=${FIRST_BATCH_TIMEOUT_SECONDS}
resource_boundary=tmux-held interactive Slurm allocation; no login-node norm stats or dataloader debug.
method_boundary=Official OpenPI data/norm/dataloader debug only. No training, replay, scorer, VAE, MLP, or custom policy.
EOF

run_python() {
  cd "${OPENPI_ROOT}"
  if [[ -n "${OPENPI_PYTHON}" ]]; then
    "${OPENPI_PYTHON}" "$@"
  else
    uv run --frozen python "$@"
  fi
}

run_python "${ROOT}/scripts/openpi/compute_lerobot_state_action_norm_stats.py" \
  --dataset-root "${DATASET_ROOT}" \
  --config-name "${CONFIG_NAME}" \
  --repo-id "${REPO_ID}" \
  --output-summary "${OUTPUT_ROOT}/norm_stats_fallback_summary.json" \
  --expected-state-dim 17 \
  --expected-action-dim 7 \
  --expected-total-frames 93648 \
  2>&1 | tee "${OUTPUT_ROOT}/norm_stats.log"

first_batch_rc=0
for mode in raw_item transformed_item first_batch; do
  out_json="${OUTPUT_ROOT}/${mode}_debug.json"
  log="${OUTPUT_ROOT}/${mode}_debug.log"
  set +e
  if [[ -n "${OPENPI_PYTHON}" ]]; then
    timeout "${FIRST_BATCH_TIMEOUT_SECONDS}" "${OPENPI_PYTHON}" \
      "${ROOT}/scripts/openpi/debug_openpi_lerobot_first_batch.py" \
      --config-name "${CONFIG_NAME}" \
      --output-json "${out_json}" \
      --mode "${mode}" \
      --framework pytorch \
      --num-workers 0 \
      --batch-size 16 \
      2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
  else
    cd "${OPENPI_ROOT}"
    timeout "${FIRST_BATCH_TIMEOUT_SECONDS}" uv run --frozen python \
      "${ROOT}/scripts/openpi/debug_openpi_lerobot_first_batch.py" \
      --config-name "${CONFIG_NAME}" \
      --output-json "${out_json}" \
      --mode "${mode}" \
      --framework pytorch \
      --num-workers 0 \
      --batch-size 16 \
      2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
  fi
  set -e
  echo "${mode}_rc=${rc}" | tee -a "${OUTPUT_ROOT}/first_batch_rcs.txt"
  if [[ "${rc}" != "0" ]]; then
    first_batch_rc="${rc}"
    break
  fi
done

cat > "${OUTPUT_ROOT}/postrebuild_debug_summary.json" <<EOF
{
  "schema": "openpi_pi05_object17_clean_postrebuild_debug_summary_v1",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "slurm_step_id": "${SLURM_STEP_ID}",
  "host": "$(hostname)",
  "config_name": "${CONFIG_NAME}",
  "repo_id": "${REPO_ID}",
  "dataset_root": "${DATASET_ROOT}",
  "norm_stats_summary": "${OUTPUT_ROOT}/norm_stats_fallback_summary.json",
  "first_batch_rc": ${first_batch_rc},
  "passed": $(if [[ "${first_batch_rc}" == "0" ]]; then echo true; else echo false; fi)
}
EOF

exit "${first_batch_rc}"
