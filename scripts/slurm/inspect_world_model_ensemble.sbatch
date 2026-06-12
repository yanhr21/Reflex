#!/usr/bin/env bash
#SBATCH --job-name=wm_object_inspect
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
ENSEMBLE_DIR="${ENSEMBLE_DIR:?Set ENSEMBLE_DIR to the completed object-state world-model ensemble directory}"
OUTPUT_JSON="${OUTPUT_JSON:-${ENSEMBLE_DIR}/inspection.json}"
OUTPUT_MD="${OUTPUT_MD:-${ENSEMBLE_DIR}/inspection.md}"
REQUIRE_COMPLIANT="${REQUIRE_COMPLIANT:-true}"
REQUIRE_RGBD_DERIVED="${REQUIRE_RGBD_DERIVED:-false}"

cd "${ROOT}"
source "${VENV}/bin/activate"
source "${ROOT}/scripts/slurm/python_retry.sh"
mkdir -p logs/slurm

export PYTHONPATH="${ROOT}/scripts/world_model:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

MANIFEST="${ENSEMBLE_DIR}/inspection_slurm_manifest_${SLURM_JOB_ID:-manual}.txt"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "node_list=${SLURM_JOB_NODELIST:-none}"
  echo "root=${ROOT}"
  echo "ensemble_dir=${ENSEMBLE_DIR}"
  echo "output_json=${OUTPUT_JSON}"
  echo "output_md=${OUTPUT_MD}"
  echo "require_compliant=${REQUIRE_COMPLIANT}"
  echo "require_rgbd_derived=${REQUIRE_RGBD_DERIVED}"
  echo "purpose=post_training_compliance_inspection"
  echo "note=This verifies completion and 3h compliance; it is not controller success evidence."
} | tee "${MANIFEST}"

run_python_with_retry "${ROOT}/scripts/world_model/inspect_world_model_ensemble.py" \
  --ensemble-dir "${ENSEMBLE_DIR}" \
  --output-json "${OUTPUT_JSON}" \
  --output-md "${OUTPUT_MD}"

if [[ "${REQUIRE_COMPLIANT}" == "true" ]]; then
  run_python_stdin_with_retry "${OUTPUT_JSON}" "${REQUIRE_RGBD_DERIVED}" <<'PY'
import json
import sys

path = sys.argv[1]
require_rgbd_derived = sys.argv[2] == "true"
report = json.loads(open(path).read())
if not report.get("compliant_training_evidence", False):
    print(
        "compliant_training_evidence=false; refusing to pass post-training gate",
        file=sys.stderr,
    )
    sys.exit(65)
if require_rgbd_derived and not report.get("rgbd_derived_training_evidence", False):
    print(
        "rgbd_derived_training_evidence=false; refusing to pass RGB-D world-model gate",
        file=sys.stderr,
    )
    sys.exit(65)
PY
fi
