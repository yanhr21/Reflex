#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

PATHS_FILE="${PATHS_FILE:?set PATHS_FILE to the canonical fix3 H5 path list}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the RGB dataset root}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/render_logs}"
SHARD_SPECS="${SHARD_SPECS:?set SHARD_SPECS as comma-separated gpu:start:end:shard_id entries}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
FPS="${FPS:-30}"
SHEET_LIMIT="${SHEET_LIMIT:-36}"
SHEET_FRAMES="${SHEET_FRAMES:-24}"
SHEET_THUMB_WIDTH="${SHEET_THUMB_WIDTH:-384}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"

cd "${ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

export PYTHONPATH="${ROOT}/deps/ManiSkill_clean:${ROOT}:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY=

{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
  echo "paths_file=${PATHS_FILE}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "log_dir=${LOG_DIR}"
  echo "shard_specs=${SHARD_SPECS}"
  echo "width=${WIDTH}"
  echo "height=${HEIGHT}"
  echo "fps=${FPS}"
  echo "boundary=render frozen user-override fix3 v7 H5 source to RGB SFT dataset; no trajectory generation"
} | tee "${LOG_DIR}/allocation_${SLURM_JOB_ID}_${SLURM_STEP_ID}_manifest.txt"

"${ROOT}/.venv/bin/python" -m py_compile \
  scripts/world_model/render_cosmos3_maniskill_sft_dataset.py

IFS=',' read -ra specs <<< "${SHARD_SPECS}"
pids=()
for spec in "${specs[@]}"; do
  IFS=':' read -r gpu_id start_index end_index shard_id <<< "${spec}"
  if [[ -z "${gpu_id:-}" || -z "${start_index:-}" || -z "${end_index:-}" || -z "${shard_id:-}" ]]; then
    echo "invalid shard spec: ${spec}" >&2
    exit 2
  fi
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    echo "shard_start=$(date -Is) gpu=${gpu_id} start=${start_index} end=${end_index} shard=${shard_id}"
    "${ROOT}/.venv/bin/python" -u scripts/world_model/render_cosmos3_maniskill_sft_dataset.py \
      --paths-file "${PATHS_FILE}" \
      --output-root "${OUTPUT_ROOT}" \
      --width "${WIDTH}" \
      --height "${HEIGHT}" \
      --fps "${FPS}" \
      --frame-stride 1 \
      --max-frames 0 \
      --val-fraction "${VAL_FRACTION}" \
      --sheet-limit "${SHEET_LIMIT}" \
      --sheet-frames "${SHEET_FRAMES}" \
      --sheet-thumb-width "${SHEET_THUMB_WIDTH}" \
      --start-index "${start_index}" \
      --end-index "${end_index}" \
      --shard-id "${shard_id}" \
      --no-write-canonical-metadata \
      --overwrite
    echo "shard_done=$(date -Is) gpu=${gpu_id} shard=${shard_id}"
  ) 2>&1 | tee "${LOG_DIR}/${shard_id}.log" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "render_shards_failed=$(date -Is)" | tee -a "${LOG_DIR}/allocation_${SLURM_JOB_ID}_${SLURM_STEP_ID}_manifest.txt"
  exit "${status}"
fi

echo "render_shards_complete=$(date -Is)" | tee -a "${LOG_DIR}/allocation_${SLURM_JOB_ID}_${SLURM_STEP_ID}_manifest.txt"
