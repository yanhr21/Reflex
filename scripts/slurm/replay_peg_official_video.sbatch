#!/usr/bin/env bash
#SBATCH --job-name=reflex_replay_video
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --export=ALL,VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json,DISPLAY=
#SBATCH --output=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.out
#SBATCH --error=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/public/home/yanhongru/ICLR2027/Reflex
MS_DIR="${ROOT}/deps/ManiSkill_clean"
VENV="${ROOT}/.venv"
SRC_DIR=/public/home/yanhongru/.maniskill/demos/PegInsertionSide-v1/motionplanning
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiments/replay_video/official_motionplanning_preview/run_${SLURM_JOB_ID:-manual}}"
COUNT="${COUNT:-10}"
NUM_ENVS="${NUM_ENVS:-1}"

cd "${ROOT}"
source "${VENV}/bin/activate"
mkdir -p "${OUTPUT_DIR}" logs/slurm

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY="${DISPLAY:-}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export PYTHONPATH="${MS_DIR}/examples/baselines/diffusion_policy:${ROOT}/scripts/training:${PYTHONPATH:-}"

SRC_H5="${SRC_DIR}/trajectory.h5"
SRC_JSON="${SRC_DIR}/trajectory.json"
WORK_H5="${OUTPUT_DIR}/trajectory.h5"
WORK_JSON="${OUTPUT_DIR}/trajectory.json"
MANIFEST="${OUTPUT_DIR}/manifest.txt"

cp -f "${SRC_H5}" "${WORK_H5}"
cp -f "${SRC_JSON}" "${WORK_JSON}"

{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "node_list=${SLURM_JOB_NODELIST:-none}"
  echo "vk_icd_filenames=${VK_ICD_FILENAMES:-unset}"
  echo "hdf5_use_file_locking=${HDF5_USE_FILE_LOCKING:-unset}"
  echo "src_h5=${SRC_H5}"
  echo "work_h5=${WORK_H5}"
  echo "count=${COUNT}"
  echo "num_envs=${NUM_ENVS}"
  echo "mode=official_motionplanning_replay_video"
  echo "maniskill_commit=$(git -C "${MS_DIR}" rev-parse HEAD)"
} | tee "${MANIFEST}"

nvidia-smi || true

cd "${MS_DIR}"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "${WORK_H5}" \
  --use-first-env-state \
  -c pd_ee_delta_pose \
  -o state \
  --save-video \
  --allow-failure \
  --count "${COUNT}" \
  --num-envs "${NUM_ENVS}" \
  -b physx_cpu

cd "${ROOT}"
python "${ROOT}/scripts/tools/extract_video_frames.py" \
  --video-dir "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}/frames" \
  --max-videos "${MAX_VIDEOS:-16}"

{
  echo "video_files:"
  find "${OUTPUT_DIR}" -type f -name '*.mp4' -printf '%p %s\n' | sort
  echo "frame_files:"
  find "${OUTPUT_DIR}/frames" -type f -name '*.png' -printf '%p %s\n' | sort | head -n 80
  echo "status=complete"
} | tee -a "${MANIFEST}"
