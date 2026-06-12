#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -A mayi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=cosmos3_render_shard
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
PATHS_FILE="${PATHS_FILE:?set PATHS_FILE}"
DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"
START_INDEX="${START_INDEX:?set START_INDEX}"
END_INDEX="${END_INDEX:?set END_INDEX}"
SHARD_ID="${SHARD_ID:-shard_${START_INDEX}_${END_INDEX}_${SLURM_JOB_ID:-local}}"
LOG_ROOT="${LOG_ROOT:-${DATASET_ROOT}/shard_logs}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${SHARD_ID}.log}"
MAX_FRAMES="${MAX_FRAMES:-0}"
RENDER_TIMEOUT_SECONDS="${RENDER_TIMEOUT_SECONDS:-0}"

cd "${ROOT}"
mkdir -p "${LOG_ROOT}" logs/slurm

export PYTHONPATH="${ROOT}/deps/ManiSkill_clean:${ROOT}:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY=
export HDF5_USE_FILE_LOCKING=FALSE

{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
  echo "paths_file=${PATHS_FILE}"
  echo "dataset_root=${DATASET_ROOT}"
  echo "start_index=${START_INDEX}"
  echo "end_index=${END_INDEX}"
  echo "shard_id=${SHARD_ID}"
  echo "max_frames=${MAX_FRAMES}"
  echo "render_timeout_seconds=${RENDER_TIMEOUT_SECONDS}"
  echo "boundary=disjoint_fresh_render_shard_no_canonical_metadata_no_old_full1000_artifacts"
} | tee "${LOG_FILE}"

set +e
nvidia-smi 2>&1 | tee -a "${LOG_FILE}"
nvidia_status=${PIPESTATUS[0]}
set -e
echo "nvidia_smi_status=${nvidia_status}" | tee -a "${LOG_FILE}"
if [[ "${nvidia_status}" -ne 0 ]]; then
  echo "render_shard_failed_preflight=$(date -Is)" | tee -a "${LOG_FILE}"
  exit "${nvidia_status}"
fi

render_cmd=(
  "${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/render_cosmos3_maniskill_sft_dataset.py"
  --paths-file "${PATHS_FILE}" \
  --output-root "${DATASET_ROOT}" \
  --width 1024 \
  --height 1024 \
  --fps 30 \
  --frame-stride 1 \
  --max-frames "${MAX_FRAMES}" \
  --val-fraction 0.1 \
  --sheet-limit 0 \
  --resume \
  --start-index "${START_INDEX}" \
  --end-index "${END_INDEX}" \
  --shard-id "${SHARD_ID}" \
  --no-write-canonical-metadata
)

set +e
if [[ "${RENDER_TIMEOUT_SECONDS}" -gt 0 ]]; then
  timeout --preserve-status "${RENDER_TIMEOUT_SECONDS}" "${render_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  "${render_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi
render_status=${PIPESTATUS[0]}
set -e

echo "render_status=${render_status}" | tee -a "${LOG_FILE}"
if [[ "${render_status}" -eq 0 ]]; then
  echo "render_shard_complete=$(date -Is)" | tee -a "${LOG_FILE}"
else
  echo "render_shard_failed=$(date -Is)" | tee -a "${LOG_FILE}"
fi
exit "${render_status}"
