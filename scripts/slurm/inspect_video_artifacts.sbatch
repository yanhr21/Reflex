#!/usr/bin/env bash
#SBATCH --job-name=wm_video_artifacts
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.out
#SBATCH --error=/public/home/yanhongru/ICLR2027/Reflex/logs/slurm/%x-%j.err

set -euo pipefail

ROOT=/public/home/yanhongru/ICLR2027/Reflex
VENV="${ROOT}/.venv"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a controller run directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/video_review}"
REQUIRE_VIDEO="${REQUIRE_VIDEO:-true}"
REQUIRE_NONBLANK="${REQUIRE_NONBLANK:-true}"

cd "${ROOT}"
source "${VENV}/bin/activate"
source "${ROOT}/scripts/slurm/python_retry.sh"
mkdir -p logs/slurm "${OUTPUT_DIR}"

export PYTHONPATH="${ROOT}/scripts/world_model:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

MANIFEST="${OUTPUT_DIR}/slurm_manifest_${SLURM_JOB_ID:-manual}.txt"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "node_list=${SLURM_JOB_NODELIST:-none}"
  echo "root=${ROOT}"
  echo "run_dir=${RUN_DIR}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "require_video=${REQUIRE_VIDEO}"
  echo "require_nonblank=${REQUIRE_NONBLANK}"
  echo "purpose=video_artifact_review_sheet_generation"
  echo "note=This prepares readable video sheets; it is not semantic task-success inspection."
} | tee "${MANIFEST}"

PY_ARGS=(
  --run-dir "${RUN_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --sample-count "${SAMPLE_COUNT:-10}"
  --max-frames-to-scan "${MAX_FRAMES_TO_SCAN:-2000}"
  --thumb-width "${THUMB_WIDTH:-480}"
)
if [[ "${REQUIRE_VIDEO}" == "true" ]]; then
  PY_ARGS+=(--require-video)
else
  PY_ARGS+=(--no-require-video)
fi
if [[ "${REQUIRE_NONBLANK}" == "true" ]]; then
  PY_ARGS+=(--require-nonblank)
else
  PY_ARGS+=(--no-require-nonblank)
fi

run_python_with_retry "${ROOT}/scripts/world_model/inspect_video_artifacts.py" "${PY_ARGS[@]}"
