#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -A mayi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=cosmos3_render_ranges
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
PATHS_FILE="${PATHS_FILE:?set PATHS_FILE}"
DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"
RANGES="${RANGES:?set RANGES as comma-separated start:end pairs}"
SHARD_ID_PREFIX="${SHARD_ID_PREFIX:-ranges_${SLURM_JOB_ID:-local}}"
LOG_ROOT="${LOG_ROOT:-${DATASET_ROOT}/shard_logs}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${SHARD_ID_PREFIX}.log}"
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
  echo "ranges=${RANGES}"
  echo "shard_id_prefix=${SHARD_ID_PREFIX}"
  echo "max_frames=${MAX_FRAMES}"
  echo "render_timeout_seconds=${RENDER_TIMEOUT_SECONDS}"
  echo "boundary=sequential_disjoint_fresh_render_ranges_no_canonical_metadata_no_old_full1000_artifacts"
} | tee "${LOG_FILE}"

set +e
nvidia-smi 2>&1 | tee -a "${LOG_FILE}"
nvidia_status=${PIPESTATUS[0]}
set -e
echo "nvidia_smi_status=${nvidia_status}" | tee -a "${LOG_FILE}"
if [[ "${nvidia_status}" -ne 0 ]]; then
  echo "render_ranges_failed_preflight=$(date -Is)" | tee -a "${LOG_FILE}"
  exit "${nvidia_status}"
fi

normalized_ranges="${RANGES//;/,}"
IFS=',' read -r -a range_items <<< "${normalized_ranges}"
overall_status=0
for range_item in "${range_items[@]}"; do
  start_index="${range_item%%:*}"
  end_index="${range_item##*:}"
  if [[ -z "${start_index}" || -z "${end_index}" || "${start_index}" == "${range_item}" ]]; then
    echo "invalid_range=${range_item}" | tee -a "${LOG_FILE}" >&2
    exit 65
  fi
  range_shard_id="${SHARD_ID_PREFIX}_${start_index}_${end_index}"
  echo "range_start=$(date -Is) start_index=${start_index} end_index=${end_index} shard_id=${range_shard_id}" | tee -a "${LOG_FILE}"
  render_cmd=(
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/render_cosmos3_maniskill_sft_dataset.py"
    --paths-file "${PATHS_FILE}"
    --output-root "${DATASET_ROOT}"
    --width 1024
    --height 1024
    --fps 30
    --frame-stride 1
    --max-frames "${MAX_FRAMES}"
    --val-fraction 0.1
    --sheet-limit 0
    --resume
    --start-index "${start_index}"
    --end-index "${end_index}"
    --shard-id "${range_shard_id}"
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
  echo "range_status=${render_status} start_index=${start_index} end_index=${end_index}" | tee -a "${LOG_FILE}"
  if [[ "${render_status}" -ne 0 ]]; then
    overall_status="${render_status}"
    break
  fi
done

echo "render_ranges_status=${overall_status}" | tee -a "${LOG_FILE}"
if [[ "${overall_status}" -eq 0 ]]; then
  echo "render_ranges_complete=$(date -Is)" | tee -a "${LOG_FILE}"
else
  echo "render_ranges_failed=$(date -Is)" | tee -a "${LOG_FILE}"
fi
exit "${overall_status}"
