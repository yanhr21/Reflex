#!/usr/bin/env bash
#SBATCH --job-name=reflex_dp_peg1000
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --export=ALL,VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json,DISPLAY=
#SBATCH --output=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.out
#SBATCH --error=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/public/home/yanhongru/ICLR2027/Reflex
MS_DIR="${ROOT}/deps/ManiSkill_clean"
VENV="${ROOT}/.venv"
DATA_H5="${ROOT}/data/official_replay/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
RUN_ROOT="${ROOT}/experiments/dp_peg1000"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8192}"
MASTER_PORT="${MASTER_PORT:-29541}"

cd "${ROOT}"
source "${VENV}/bin/activate"
mkdir -p "${RUN_ROOT}" logs/slurm

export PYTHONPATH="${MS_DIR}/examples/baselines/diffusion_policy:${ROOT}/scripts/training:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY="${DISPLAY:-}"
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT

echo "timestamp=$(date -Is)"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "node_list=${SLURM_JOB_NODELIST:-none}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "gpus_per_node=${GPUS_PER_NODE}"
echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "vk_icd_filenames=${VK_ICD_FILENAMES:-unset}"
echo "data_h5=${DATA_H5}"
git -C "${MS_DIR}" rev-parse HEAD
nvidia-smi || true

test -f "${DATA_H5}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${GPUS_PER_NODE}" \
  "${ROOT}/scripts/training/train_dp_state_ddp.py" \
  --env-id PegInsertionSide-v1 \
  --demo-path "${DATA_H5}" \
  --control-mode pd_ee_delta_pose \
  --sim-backend physx_cpu \
  --num-demos 1000 \
  --max-episode-steps 300 \
  --total-iters "${TOTAL_ITERS:-100000}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --eval-freq "${EVAL_FREQ:-2500}" \
  --save-freq "${SAVE_FREQ:-2500}" \
  --log-freq "${LOG_FREQ:-100}" \
  --num-eval-episodes "${NUM_EVAL_EPISODES:-100}" \
  --num-eval-envs "${NUM_EVAL_ENVS:-10}" \
  --exp-name "dp_peg1000_${SLURM_JOB_ID:-manual}" \
  --run-dir "${RUN_ROOT}/run_${SLURM_JOB_ID:-manual}" \
  --demo-type motionplanning \
  --no-capture-video
