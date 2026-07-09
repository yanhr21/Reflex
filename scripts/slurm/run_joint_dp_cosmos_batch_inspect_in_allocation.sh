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

RUN_GROUP="${RUN_GROUP:-joint_overfit_dataset}"
RUN_NAME="${RUN_NAME:-overfit01}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/${RUN_GROUP}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/02_joint_training/${RUN_GROUP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
DATASET_STAGE="${DATASET_STAGE:-joint_overfit_dataset}"
PROJECT_PYTHON="${PROJECT_PYTHON:-${ROOT}/.venv/bin/python}"
INTERFACE_RUN_DIR="${INTERFACE_RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/interface_inspect/inspect02}"
BUILDER="${ROOT}/scripts/world_model/build_joint_dp_cosmos_dataset.py"
INSPECTOR="${ROOT}/scripts/world_model/inspect_joint_dp_cosmos_batch.py"
A_COUNT="${A_COUNT:-4}"
B_COUNT="${B_COUNT:-4}"
C_COUNT="${C_COUNT:-4}"
D_COUNT="${D_COUNT:-4}"
OBS_HORIZON="${OBS_HORIZON:-2}"
PRED_HORIZON="${PRED_HORIZON:-16}"

case "${RUN_DIR}" in
  "${ROOT}/experiments/maniskill/runs/02_joint_training/"*) ;;
  *)
    echo "refusing_output_dir_outside_02_joint_training=true" >&2
    echo "run_dir=${RUN_DIR}" >&2
    exit 41
    ;;
esac
case "${LOG_FILE}" in
  "${ROOT}/logs/02_joint_training/"*) ;;
  *)
    echo "refusing_log_file_outside_02_joint_training=true" >&2
    echo "log_file=${LOG_FILE}" >&2
    exit 42
    ;;
esac
if [[ -e "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing_existing_nonempty_run_dir=true" >&2
  echo "run_dir=${RUN_DIR}" >&2
  exit 43
fi

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

export ROOT
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY=
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export AWS_EC2_METADATA_DISABLED=true

{
  echo "timestamp=$(date -Is)"
  echo "phase=02_joint_training"
  echo "stage=${DATASET_STAGE}_batch_inspect"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID}"
  echo "node=$(hostname)"
  echo "project_python=${PROJECT_PYTHON}"
  echo "interface_run_dir=${INTERFACE_RUN_DIR}"
  echo "builder=${BUILDER}"
  echo "inspector=${INSPECTOR}"
  echo "a_count=${A_COUNT}"
  echo "b_count=${B_COUNT}"
  echo "c_count=${C_COUNT}"
  echo "d_count=${D_COUNT}"
  echo "obs_horizon=${OBS_HORIZON}"
  echo "pred_horizon=${PRED_HORIZON}"
  echo "controller_action_contract=pd_ee_delta_pose"
  echo "evidence_type=diagnostic_batch_check"
  echo "method_evidence_allowed=false"
  echo "training_started=false"
  echo "data_generation_started=false"
  echo "uses_toy_model=false"
  echo "forbidden_state_intervention_used=false"
} | tee -a "${LOG_FILE}"

nvidia-smi 2>&1 | tee -a "${LOG_FILE}" || true

echo "compile_start=$(date -Is)" | tee -a "${LOG_FILE}"
"${PROJECT_PYTHON}" -m py_compile "${BUILDER}" "${INSPECTOR}" 2>&1 | tee -a "${LOG_FILE}"
echo "compile_status=0" | tee -a "${LOG_FILE}"

echo "build_joint_dataset_start=$(date -Is)" | tee -a "${LOG_FILE}"
"${PROJECT_PYTHON}" -u "${BUILDER}" \
  --root "${ROOT}" \
  --output-dir "${RUN_DIR}" \
  --interface-run-dir "${INTERFACE_RUN_DIR}" \
  --a-count "${A_COUNT}" \
  --b-count "${B_COUNT}" \
  --c-count "${C_COUNT}" \
  --d-count "${D_COUNT}" \
  --obs-horizon "${OBS_HORIZON}" \
  --pred-horizon "${PRED_HORIZON}" \
  --stage-name "${DATASET_STAGE}" \
  2>&1 | tee -a "${LOG_FILE}"
echo "build_joint_dataset_status=0" | tee -a "${LOG_FILE}"

{
  echo "timestamp=$(date -Is)"
  echo "phase=02_joint_training"
  echo "stage=${DATASET_STAGE}_batch_inspect"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID}"
  echo "node=$(hostname)"
  echo "project_python=${PROJECT_PYTHON}"
  echo "interface_run_dir=${INTERFACE_RUN_DIR}"
  echo "method_evidence_allowed=false"
  echo "training_started=false"
  echo "data_generation_started=false"
  echo "uses_toy_model=false"
} > "${RUN_DIR}/allocation_manifest.txt"

echo "batch_inspect_start=$(date -Is)" | tee -a "${LOG_FILE}"
"${PROJECT_PYTHON}" -u "${INSPECTOR}" \
  --root "${ROOT}" \
  --dataset-dir "${RUN_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"
echo "batch_inspect_status=0" | tee -a "${LOG_FILE}"

{
  echo "joint_overfit_batch_status=complete"
  echo "dataset_summary=${RUN_DIR}/summary.json"
  echo "samples=${RUN_DIR}/joint_overfit_samples.jsonl"
  echo "batch_inspection=${RUN_DIR}/batch_inspection.json"
  echo "completed_at=$(date -Is)"
} | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
