#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
OPENPI_ROOT="${OPENPI_ROOT:-/public/home/yanhongru/ICLR2027/openpi}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run OpenPI pi0.5 training only inside a compute-node srun step.
example=srun --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=120G bash scripts/slurm/run_openpi_pi05_peg733_train_in_allocation_skipnorm.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CONFIG_NAME="${CONFIG_NAME:-pi05_maniskill_peg733}"
EXP_NAME="${EXP_NAME:-pi05_peg733_1gpu1h_${STAMP}_alloc${SLURM_JOB_ID}}"
LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/lerobot_home}"
OPENPI_DATA_HOME_DIR="${OPENPI_DATA_HOME_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/openpi_data_home}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/openpi/${EXP_NAME}}"
UV_SHARED_CACHE_DIR="${UV_SHARED_CACHE_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/uv_cache}"
MIN_WALL_SECONDS="${MIN_WALL_SECONDS:-3660}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
UV_PYTHON_PLATFORM="${UV_PYTHON_PLATFORM:-}"
UV_RUN_PYTHON="${UV_RUN_PYTHON:-}"

mkdir -p "${OUTPUT_ROOT}" "${LEROBOT_HOME_DIR}" "${OPENPI_DATA_HOME_DIR}" "${CHECKPOINT_BASE_DIR}" "${UV_SHARED_CACHE_DIR}"

export HF_LEROBOT_HOME="${LEROBOT_HOME_DIR}"
unset LEROBOT_HOME
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME_DIR}"
export HF_HOME="${HF_HOME:-/tmp/hf_home_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets_cache_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${UV_SHARED_CACHE_DIR}}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/openpi_uv_env_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export WANDB_MODE="${WANDB_MODE:-offline}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export GIT_HTTP_VERSION="${GIT_HTTP_VERSION:-HTTP/1.1}"
export GIT_TERMINAL_PROMPT=0
OPENPI_PYTHON="${OPENPI_PYTHON:-}"
if [[ -z "${DATASET_ASSET_ID:-}" ]]; then
  case "${CONFIG_NAME}" in
    pi05_maniskill_peg733)
      DATASET_ASSET_ID="yanhongru/maniskill_peg733_openpi_libero"
      ;;
    pi05_maniskill_peg733_contact_suffix16)
      DATASET_ASSET_ID="yanhongru/maniskill_peg733_openpi_contact_suffix16"
      ;;
    pi05_maniskill_peg733_contact_suffix16_qpos8_clean_20260626)
      DATASET_ASSET_ID="yanhongru/maniskill_peg733_openpi_contact_suffix16_qpos8_clean_20260626"
      ;;
    pi05_maniskill_peg733_contact_suffix16_object17)
      DATASET_ASSET_ID="yanhongru/maniskill_peg733_openpi_contact_suffix16_object17"
      ;;
    pi05_maniskill_peg733_contact_suffix16_object17_clean_20260626)
      DATASET_ASSET_ID="yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_clean_20260626"
      ;;
    pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626)
      DATASET_ASSET_ID="yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_clean_20260626"
      ;;
    *)
      DATASET_ASSET_ID=""
      ;;
  esac
fi
NORM_STATS_PATH="${NORM_STATS_PATH:-${OPENPI_ROOT}/assets/${CONFIG_NAME}/${DATASET_ASSET_ID}/norm_stats.json}"

cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=openpi_pi05_peg733_train_skipnorm_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
openpi_root=${OPENPI_ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
config_name=${CONFIG_NAME}
exp_name=${EXP_NAME}
hf_lerobot_home=${HF_LEROBOT_HOME}
openpi_data_home=${OPENPI_DATA_HOME}
checkpoint_base_dir=${CHECKPOINT_BASE_DIR}
output_root=${OUTPUT_ROOT}
min_wall_seconds=${MIN_WALL_SECONDS}
norm_stats_path=${NORM_STATS_PATH}
norm_stats_must_exist=true
dataset_asset_id=${DATASET_ASSET_ID}
proxy_policy=unset_proxy_environment_before_uv
git_http_version=${GIT_HTTP_VERSION}
uv_cache_dir=${UV_CACHE_DIR}
uv_index_url=${UV_INDEX_URL:-}
uv_default_index=${UV_DEFAULT_INDEX:-}
uv_project_environment=${UV_PROJECT_ENVIRONMENT}
uv_link_mode=${UV_LINK_MODE}
uv_python_platform=${UV_PYTHON_PLATFORM:-native}
uv_run_python=${UV_RUN_PYTHON:-uv_default}
hf_hub_offline=${HF_HUB_OFFLINE:-}
hf_datasets_offline=${HF_DATASETS_OFFLINE:-}
hf_datasets_cache=${HF_DATASETS_CACHE:-}
hf_home=${HF_HOME:-}
xdg_cache_home=${XDG_CACHE_HOME:-}
transformers_offline=${TRANSFORMERS_OFFLINE:-}
wandb_mode=${WANDB_MODE:-}
openpi_python=${OPENPI_PYTHON:-uv_frozen_default}
resource_boundary=tmux-held interactive Slurm allocation; no login-node norm stats/training.
method_boundary=Official OpenPI pi0.5 training only. No custom VAE/MLP/diffusion intermediate model.
weight_boundary=Config must use official OpenPI CheckpointWeightLoader and official pi0.5 checkpoint path.
EOF

if [[ ! -f "${NORM_STATS_PATH}" ]]; then
  echo "missing_required_norm_stats=${NORM_STATS_PATH}" | tee "${OUTPUT_ROOT}/compute_norm_stats.log" >&2
  exit 42
fi
{
  echo "skip_existing_norm_stats=true"
  echo "norm_stats_path=${NORM_STATS_PATH}"
  echo "reason=Existing OpenPI norm_stats.json found for ${CONFIG_NAME}; skipping duplicate stats pass."
} | tee "${OUTPUT_ROOT}/compute_norm_stats.log"

cd "${OPENPI_ROOT}"

start_ts="$(date +%s)"
set +e
if [[ -n "${OPENPI_PYTHON}" ]]; then
  "${OPENPI_PYTHON}" scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}" \
    --overwrite \
    ${TRAIN_ARGS} \
    2>&1 | tee "${OUTPUT_ROOT}/train.log"
  train_rc=${PIPESTATUS[0]}
else
  UV_RUN_ARGS=(uv run --frozen)
  if [[ -n "${UV_RUN_PYTHON}" ]]; then
    UV_RUN_ARGS+=(--python "${UV_RUN_PYTHON}")
  fi
  if [[ -n "${UV_PYTHON_PLATFORM}" ]]; then
    UV_RUN_ARGS+=(--python-platform "${UV_PYTHON_PLATFORM}")
  fi
  "${UV_RUN_ARGS[@]}" scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}" \
    --overwrite \
    ${TRAIN_ARGS} \
    2>&1 | tee "${OUTPUT_ROOT}/train.log"
  train_rc=${PIPESTATUS[0]}
fi
set -e
end_ts="$(date +%s)"
elapsed=$((end_ts - start_ts))

cat > "${OUTPUT_ROOT}/training_walltime_summary.json" <<EOF
{
  "config_name": "${CONFIG_NAME}",
  "exp_name": "${EXP_NAME}",
  "elapsed_seconds": ${elapsed},
  "min_wall_seconds": ${MIN_WALL_SECONDS},
  "formal_one_gpu_hour_floor_met": $(if [[ "${elapsed}" -ge "${MIN_WALL_SECONDS}" ]]; then echo true; else echo false; fi),
  "train_return_code": ${train_rc},
  "slurm_job_id": "${SLURM_JOB_ID}",
  "slurm_step_id": "${SLURM_STEP_ID}",
  "checkpoint_base_dir": "${CHECKPOINT_BASE_DIR}",
  "note": "Elapsed starts at official train.py launch. Weight download time may be included in wall time, so optimization-step logs are still required before interpreting training quality."
}
EOF

exit "${train_rc}"
