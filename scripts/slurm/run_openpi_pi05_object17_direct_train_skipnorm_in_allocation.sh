#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
OPENPI_ROOT="${OPENPI_ROOT:-/public/home/yanhongru/ICLR2027/openpi}"
CONFIG_NAME="${CONFIG_NAME:-pi05_maniskill_peg733_contact_suffix16_object17}"
REPO_ID="${REPO_ID:-yanhongru/maniskill_peg733_openpi_contact_suffix16_object17}"
EXP_NAME="${EXP_NAME:-pi05_peg733_contact_suffix16_object17_direct1700_skipnorm_$(date +%Y%m%d_%H%M%S)_alloc${SLURM_JOB_ID:-unknown}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/openpi/${EXP_NAME}}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/tmp/openpi_pi05_checkpoints_${USER}_${SLURM_JOB_ID:-unknown}_object17_direct}"
PRESERVE_ROOT="${PRESERVE_ROOT:-${ROOT}/experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/${CONFIG_NAME}/${EXP_NAME}}"
OPENPI_PYTHON="${OPENPI_PYTHON:-/tmp/openpi_uv_env_${USER}_${SLURM_JOB_ID:-unknown}_2/bin/python}"
LEROBOT_HOME_DIR="${LEROBOT_HOME_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/lerobot_home}"
OPENPI_DATA_HOME_DIR="${OPENPI_DATA_HOME_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/openpi_data_home}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1700}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
MIN_WALL_SECONDS="${MIN_WALL_SECONDS:-3660}"
FINAL_STEP="${FINAL_STEP:-1699}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run direct OpenPI object17 training only inside a compute-node srun step.
EOF
  exit 30
fi

mkdir -p "${OUTPUT_ROOT}" "${CHECKPOINT_BASE_DIR}" "${PRESERVE_ROOT}" "${LEROBOT_HOME_DIR}" "${OPENPI_DATA_HOME_DIR}"

export HF_LEROBOT_HOME="${LEROBOT_HOME_DIR}"
unset LEROBOT_HOME
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME_DIR}"
export HF_HOME="${HF_HOME:-/tmp/hf_home_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets_cache_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export GIT_TERMINAL_PROMPT=0
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

NORM_STATS_PATH="${OPENPI_ROOT}/assets/${CONFIG_NAME}/${REPO_ID}/norm_stats.json"
if [[ ! -f "${NORM_STATS_PATH}" ]]; then
  echo "missing_norm_stats=${NORM_STATS_PATH}" >&2
  exit 31
fi
if [[ ! -x "${OPENPI_PYTHON}" ]]; then
  echo "missing_openpi_python=${OPENPI_PYTHON}" >&2
  exit 32
fi

cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=openpi_pi05_object17_direct_train_skipnorm_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
openpi_root=${OPENPI_ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
config_name=${CONFIG_NAME}
repo_id=${REPO_ID}
exp_name=${EXP_NAME}
output_root=${OUTPUT_ROOT}
checkpoint_base_dir=${CHECKPOINT_BASE_DIR}
preserve_root=${PRESERVE_ROOT}
norm_stats_path=${NORM_STATS_PATH}
openpi_python=${OPENPI_PYTHON}
hf_home=${HF_HOME}
hf_datasets_cache=${HF_DATASETS_CACHE}
xdg_cache_home=${XDG_CACHE_HOME}
lerobot_video_backend=${LEROBOT_VIDEO_BACKEND}
num_train_steps=${NUM_TRAIN_STEPS}
save_interval=${SAVE_INTERVAL}
min_wall_seconds=${MIN_WALL_SECONDS}
resource_boundary=tmux-held interactive Slurm allocation; no login-node training.
method_boundary=Official OpenPI scripts/train.py and Pi0Config(pi05=True); norm stats precomputed from LeRobot state/actions with OpenPI normalize.save fallback.
EOF

cd "${OPENPI_ROOT}"
start_ts="$(date +%s)"
set +e
"${OPENPI_PYTHON}" scripts/train.py "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}" \
  --overwrite \
  --num-train-steps="${NUM_TRAIN_STEPS}" \
  --save-interval="${SAVE_INTERVAL}" \
  2>&1 | tee "${OUTPUT_ROOT}/train.log"
train_rc=${PIPESTATUS[0]}
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
  "norm_stats_path": "${NORM_STATS_PATH}"
}
EOF

src_ckpt="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}/${FINAL_STEP}"
if [[ "${train_rc}" == "0" && -d "${src_ckpt}" ]]; then
  tmp_dest="${PRESERVE_ROOT}/${FINAL_STEP}.tmp.$$"
  final_dest="${PRESERVE_ROOT}/${FINAL_STEP}"
  rm -rf "${tmp_dest}"
  mkdir -p "$(dirname "${tmp_dest}")"
  cp -a "${src_ckpt}" "${tmp_dest}"
  mv "${tmp_dest}" "${final_dest}"
  {
    echo "source=${src_ckpt}"
    echo "destination=${final_dest}"
    du -sh "${src_ckpt}" "${final_dest}"
    find "${final_dest}" -type f | wc -l
  } > "${OUTPUT_ROOT}/checkpoint_copy_summary.txt"
fi

exit "${train_rc}"
