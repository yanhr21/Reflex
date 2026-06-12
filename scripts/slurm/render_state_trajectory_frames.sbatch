#!/usr/bin/env bash
#SBATCH --job-name=render_state_frames
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --export=ALL,VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json,DISPLAY=
#SBATCH --output=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.out
#SBATCH --error=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/public/home/yanhongru/ICLR2027/Reflex
MS_DIR="${ROOT}/deps/ManiSkill_clean"
VENV="${ROOT}/.venv"
TRAJ_PATH="${TRAJ_PATH:?Set TRAJ_PATH to a recorded ManiSkill .h5}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for rendered frames}"

cd "${ROOT}"
source "${VENV}/bin/activate"
mkdir -p "${OUTPUT_DIR}" logs/slurm

export PYTHONPATH="${MS_DIR}:${MS_DIR}/examples/baselines/diffusion_policy:${ROOT}/scripts/training:${ROOT}/scripts/reflex:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY="${DISPLAY:-}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

echo "timestamp=$(date -Is)"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "node_list=${SLURM_JOB_NODELIST:-none}"
echo "traj_path=${TRAJ_PATH}"
echo "output_dir=${OUTPUT_DIR}"
echo "hdf5_use_file_locking=${HDF5_USE_FILE_LOCKING:-unset}"
nvidia-smi || true

python "${ROOT}/scripts/tools/render_state_trajectory_frames.py" \
  --traj-path "${TRAJ_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --episode-index "${EPISODE_INDEX:-0}" \
  --frame-indices "${FRAME_INDICES:-}" \
  --stride "${FRAME_STRIDE:-10}" \
  --max-frames "${MAX_FRAMES:-80}" \
  --sim-backend "${SIM_BACKEND:-physx_cpu}" \
  --render-mode "${RENDER_MODE:-rgb_array}" \
  --make-video "${MAKE_VIDEO:-True}" \
  --video-fps "${VIDEO_FPS:-30}" \
  --thumb-width "${THUMB_WIDTH:-240}"
