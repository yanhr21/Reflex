#!/usr/bin/env bash
#SBATCH --job-name=reflex_eval_dp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --export=ALL,VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json,DISPLAY=
#SBATCH --output=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.out
#SBATCH --error=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/public/home/yanhongru/ICLR2027/Reflex
MS_DIR="${ROOT}/deps/ManiSkill_clean"
VENV="${ROOT}/.venv"
CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH to a trained checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiments/eval/run_${SLURM_JOB_ID:-manual}}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-0}"

cd "${ROOT}"
source "${VENV}/bin/activate"
mkdir -p "${OUTPUT_DIR}" logs/slurm

export PYTHONPATH="${MS_DIR}/examples/baselines/diffusion_policy:${ROOT}/scripts/training:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY="${DISPLAY:-}"

echo "timestamp=$(date -Is)"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "node_list=${SLURM_JOB_NODELIST:-none}"
echo "vk_icd_filenames=${VK_ICD_FILENAMES:-unset}"
echo "ckpt_path=${CKPT_PATH}"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi || true

VIDEO_ARGS=()
if [[ "${CAPTURE_VIDEO}" == "1" ]]; then
  VIDEO_ARGS=(--capture-video)
fi

python "${ROOT}/scripts/training/eval_dp_state.py" \
  --ckpt-path "${CKPT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-eval-episodes "${NUM_EVAL_EPISODES:-100}" \
  --num-eval-envs "${NUM_EVAL_ENVS:-10}" \
  "${VIDEO_ARGS[@]}"
