#!/usr/bin/env bash
#SBATCH --job-name=wm_slot_export_triage
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.out
#SBATCH --error=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/public/home/yanhongru/ICLR2027/Reflex
VENV="${ROOT}/.venv"
TARGET_JOB_ID="${TARGET_JOB_ID:?Set TARGET_JOB_ID to the RGB-D predicted-slot export Slurm job id.}"
EXPORT_DIR="${EXPORT_DIR:?Set EXPORT_DIR to the RGB-D predicted-slot export directory.}"
LOG_BASENAME="${LOG_BASENAME:-wm_rgbd_slot_export}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiments/world_model_task_rebinding/rgbd_predicted_slots/diagnostics/job${TARGET_JOB_ID}_triage/job${SLURM_JOB_ID:-manual}}"

cd "${ROOT}"
source "${VENV}/bin/activate"
mkdir -p "${OUTPUT_DIR}" logs/slurm

STDOUT_PATH="${ROOT}/logs/slurm/${LOG_BASENAME}-${TARGET_JOB_ID}.out"
STDERR_PATH="${ROOT}/logs/slurm/${LOG_BASENAME}-${TARGET_JOB_ID}.err"

MANIFEST="${OUTPUT_DIR}/slurm_manifest_${SLURM_JOB_ID:-manual}.txt"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "target_job_id=${TARGET_JOB_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-none}"
  echo "export_dir=${EXPORT_DIR}"
  echo "stdout_path=${STDOUT_PATH}"
  echo "stderr_path=${STDERR_PATH}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "boundary=diagnostic_only_does_not_change_exact96_predicted_slot_quality_or_method_gates"
  echo "reason=classify_predicted_slot_export_failure_modes_after_target_job_finishes_without_interpreting_method_success"
} | tee "${MANIFEST}"

python "${ROOT}/scripts/world_model/triage_predicted_slot_export_job.py" \
  --job-id "${TARGET_JOB_ID}" \
  --export-dir "${EXPORT_DIR}" \
  --stdout "${STDOUT_PATH}" \
  --stderr "${STDERR_PATH}" \
  --output-json "${OUTPUT_DIR}/triage.json" \
  --output-md "${OUTPUT_DIR}/triage.md"
