#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
OPENPI_ROOT="${OPENPI_ROOT:-/public/home/yanhongru/ICLR2027/openpi}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run OpenPI pi0.5 receding snapshot rollout only inside a compute-node srun step.
example=srun --jobid=<held_job_id> --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=120G bash scripts/slurm/run_openpi_pi05_receding_snapshot_rollout_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CONFIG_NAME="${CONFIG_NAME:-pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626/pi05_object17_video_clean_direct1700_1gpu1h_20260626_alloc153455/1699}"
PANEL_ROOT="${PANEL_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_safegate1_source_suffix_offsets64_48_32_24_panel0134_20260623_alloc146658}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/openpi/openpi_pi05_receding_snapshot_rollout_${STAMP}_alloc${SLURM_JOB_ID}}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
MAX_ITER_DIRS="${MAX_ITER_DIRS:-1}"
ITERATION_INDICES="${ITERATION_INDICES:-}"
MAX_RECEDING_QUERIES="${MAX_RECEDING_QUERIES:-3}"
EXECUTE_STEPS_PER_QUERY="${EXECUTE_STEPS_PER_QUERY:-4}"
IMAGE_SOURCE="${IMAGE_SOURCE:-observed_prefix_static}"
UV_SHARED_CACHE_DIR="${UV_SHARED_CACHE_DIR:-${ROOT}/experiments/world_model_task_rebinding/openpi/uv_cache}"
UV_RUN_PYTHON="${UV_RUN_PYTHON:-}"
UV_PYTHON_PLATFORM="${UV_PYTHON_PLATFORM:-}"
OPENPI_PYTHON="${OPENPI_PYTHON:-}"

mkdir -p "${OUTPUT_ROOT}" "${UV_SHARED_CACHE_DIR}"

export OPENPI_ROOT
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${ROOT}/experiments/world_model_task_rebinding/openpi/lerobot_home}"
unset LEROBOT_HOME
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${ROOT}/experiments/world_model_task_rebinding/openpi/openpi_data_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${UV_SHARED_CACHE_DIR}}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/openpi_uv_env_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export HF_HOME="${HF_HOME:-/tmp/hf_home_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets_cache_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${USER}_${SLURM_JOB_ID}_${SLURM_STEP_ID}}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export GIT_HTTP_VERSION="${GIT_HTTP_VERSION:-HTTP/1.1}"
export GIT_TERMINAL_PROMPT=0

cat > "${OUTPUT_ROOT}/run_manifest.txt" <<EOF
schema=openpi_pi05_receding_snapshot_rollout_wrapper_v1
date=$(date --iso-8601=seconds)
root=${ROOT}
openpi_root=${OPENPI_ROOT}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
config_name=${CONFIG_NAME}
checkpoint_dir=${CHECKPOINT_DIR}
panel_root=${PANEL_ROOT}
output_root=${OUTPUT_ROOT}
max_samples=${MAX_SAMPLES}
max_iter_dirs=${MAX_ITER_DIRS}
iteration_indices=${ITERATION_INDICES}
max_receding_queries=${MAX_RECEDING_QUERIES}
execute_steps_per_query=${EXECUTE_STEPS_PER_QUERY}
image_source=${IMAGE_SOURCE}
project_python=${ROOT}/.venv/bin/python
openpi_python=${OPENPI_PYTHON:-uv_frozen_default}
uv_run_python=${UV_RUN_PYTHON:-uv_default}
uv_cache_dir=${UV_CACHE_DIR}
uv_project_environment=${UV_PROJECT_ENVIRONMENT}
hf_home=${HF_HOME}
hf_datasets_cache=${HF_DATASETS_CACHE}
xdg_cache_home=${XDG_CACHE_HOME}
resource_boundary=tmux-held interactive Slurm allocation; no login-node inference/replay/render.
method_boundary=Official OpenPI pi0.5 inference in subprocess plus ManiSkill live execution; no custom VAE/MLP/diffusion/intermediate model and no scorer-only selector.
diagnostic_boundary=Uses refreshed simulator object17 state and static observed-prefix image by default; privileged upper-bound diagnostic, not final RGB-derived method evidence.
EOF

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/openpi/run_openpi_pi05_receding_snapshot_rollout.py" \
  --panel-root "${PANEL_ROOT}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --config-name "${CONFIG_NAME}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-iter-dirs "${MAX_ITER_DIRS}" \
  --iteration-indices "${ITERATION_INDICES}" \
  --max-receding-queries "${MAX_RECEDING_QUERIES}" \
  --execute-steps-per-query "${EXECUTE_STEPS_PER_QUERY}" \
  --image-source "${IMAGE_SOURCE}" \
  --openpi-python "${OPENPI_PYTHON}" \
  --uv-run-python "${UV_RUN_PYTHON}" \
  --uv-python-platform "${UV_PYTHON_PLATFORM}" \
  2>&1 | tee "${OUTPUT_ROOT}/openpi_pi05_receding_snapshot_rollout.log"
