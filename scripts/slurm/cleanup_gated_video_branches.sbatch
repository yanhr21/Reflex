#!/usr/bin/env bash
#SBATCH --job-name=wm_rebind_vidclean
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.out
#SBATCH --error=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/public/home/yanhongru/ICLR2027/Reflex
VENV="${ROOT}/.venv"
BRANCHES_FILE="${BRANCHES_FILE:?Set BRANCHES_FILE}"
OUTPUT_JSON="${OUTPUT_JSON:?Set OUTPUT_JSON}"
OUTPUT_MD="${OUTPUT_MD:?Set OUTPUT_MD}"
DRY_RUN="${DRY_RUN:-false}"

cd "${ROOT}"
source "${VENV}/bin/activate"
mkdir -p logs/slurm "$(dirname "${OUTPUT_JSON}")" "$(dirname "${OUTPUT_MD}")"

MANIFEST="$(dirname "${OUTPUT_JSON}")/cleanup_gated_video_manifest_${SLURM_JOB_ID:-manual}.txt"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "node_list=${SLURM_JOB_NODELIST:-none}"
  echo "root=${ROOT}"
  echo "branches_file=${BRANCHES_FILE}"
  echo "output_json=${OUTPUT_JSON}"
  echo "output_md=${OUTPUT_MD}"
  echo "dry_run=${DRY_RUN}"
  echo "purpose=cancel_dead_gated_video_branches_after_failed_no_video_gates"
  echo "note=This is Slurm evidence-chain hygiene only; it does not change metrics, gates, or video review requirements."
} | tee "${MANIFEST}"

PY_ARGS=(
  --branches-file "${BRANCHES_FILE}"
  --output-json "${OUTPUT_JSON}"
  --output-md "${OUTPUT_MD}"
)
if [[ "${DRY_RUN}" == "true" ]]; then
  PY_ARGS+=(--dry-run)
fi

python "${ROOT}/scripts/world_model/cleanup_gated_video_branches.py" "${PY_ARGS[@]}"
