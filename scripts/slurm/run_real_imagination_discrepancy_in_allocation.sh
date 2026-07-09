#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run inside a compute-node srun step from a tmux-held Slurm allocation.
EOF
  exit 30
fi

RUN_GROUP="${RUN_GROUP:-real_imagination_discrepancy}"
RUN_NAME="${RUN_NAME:-eval02}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/03_imagination_policy/${RUN_GROUP}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/03_imagination_policy/${RUN_GROUP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
PROJECT_PYTHON="${PROJECT_PYTHON:-${ROOT}/.venv/bin/python}"
REFERENCE_VIDEO="${REFERENCE_VIDEO:-${ROOT}/experiments/maniskill/runs/02_joint_training/joint_cosmos_condition/overfit10/condition_root/samples/B_cont_episode_000002/window_rgb.mp4}"
IMAGINED_VIDEO="${IMAGINED_VIDEO:-${ROOT}/experiments/maniskill/runs/02_joint_training/cosmos_forward_eval/eval02/inference_output/forward_B_cont_episode_000002/vision.mp4}"
EXPECTED_FRAMES="${EXPECTED_FRAMES:-93}"
TRUST_MAE_THRESHOLD="${TRUST_MAE_THRESHOLD:-0.08}"
TRUST_PSNR_THRESHOLD="${TRUST_PSNR_THRESHOLD:-18.0}"
SCRIPT="${ROOT}/scripts/world_model/compute_real_imagination_discrepancy.py"

case "${RUN_DIR}" in
  "${ROOT}/experiments/maniskill/runs/03_imagination_policy/"*) ;;
  *)
    echo "refusing_output_dir_outside_03_imagination_policy=true" >&2
    echo "run_dir=${RUN_DIR}" >&2
    exit 41
    ;;
esac
case "${LOG_FILE}" in
  "${ROOT}/logs/03_imagination_policy/"*) ;;
  *)
    echo "refusing_log_file_outside_03_imagination_policy=true" >&2
    echo "log_file=${LOG_FILE}" >&2
    exit 42
    ;;
esac
if [[ -e "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing_existing_nonempty_run_dir=true" >&2
  echo "run_dir=${RUN_DIR}" >&2
  exit 43
fi
for required in "${PROJECT_PYTHON}" "${SCRIPT}" "${REFERENCE_VIDEO}" "${IMAGINED_VIDEO}"; do
  if [[ ! -e "${required}" ]]; then
    echo "refusing_missing_required_input=true" >&2
    echo "missing=${required}" >&2
    exit 44
  fi
done

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
cd "${ROOT}"

export ROOT
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export AWS_EC2_METADATA_DISABLED=true

{
  echo "timestamp=$(date -Is)"
  echo "phase=03_imagination_policy"
  echo "stage=real_imagination_discrepancy"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID}"
  echo "node=$(hostname)"
  echo "project_python=${PROJECT_PYTHON}"
  echo "reference_video=${REFERENCE_VIDEO}"
  echo "imagined_video=${IMAGINED_VIDEO}"
  echo "expected_frames=${EXPECTED_FRAMES}"
  echo "trust_mae_threshold=${TRUST_MAE_THRESHOLD}"
  echo "trust_psnr_threshold=${TRUST_PSNR_THRESHOLD}"
  echo "method_evidence_allowed=false"
  echo "closed_loop_evidence=false"
  echo "uses_toy_model=false"
  echo "boundary=Discrepancy/trust signal only. This is not reset-to-end closed-loop control evidence."
} | tee "${RUN_DIR}/manifest.txt" | tee -a "${LOG_FILE}"

echo "compile_start=$(date -Is)" | tee -a "${LOG_FILE}"
"${PROJECT_PYTHON}" -m py_compile "${SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"
echo "compile_status=0" | tee -a "${LOG_FILE}"

echo "discrepancy_start=$(date -Is)" | tee -a "${LOG_FILE}"
"${PROJECT_PYTHON}" -u "${SCRIPT}" \
  --reference-video "${REFERENCE_VIDEO}" \
  --imagined-video "${IMAGINED_VIDEO}" \
  --expected-frames "${EXPECTED_FRAMES}" \
  --trust-mae-threshold "${TRUST_MAE_THRESHOLD}" \
  --trust-psnr-threshold "${TRUST_PSNR_THRESHOLD}" \
  --output-json "${RUN_DIR}/discrepancy.json" \
  --output-md "${RUN_DIR}/discrepancy.md" \
  2>&1 | tee -a "${LOG_FILE}"
echo "discrepancy_status=0" | tee -a "${LOG_FILE}"

{
  echo "real_imagination_discrepancy_status=complete"
  echo "discrepancy_json=${RUN_DIR}/discrepancy.json"
  echo "completed_at=$(date -Is)"
} | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
