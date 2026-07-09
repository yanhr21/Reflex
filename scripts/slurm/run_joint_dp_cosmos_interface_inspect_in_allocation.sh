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

RUN_GROUP="${RUN_GROUP:-interface_inspect}"
RUN_NAME="${RUN_NAME:-inspect01}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/${RUN_GROUP}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/02_joint_training/${RUN_GROUP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
PROJECT_PYTHON="${PROJECT_PYTHON:-${ROOT}/.venv/bin/python}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${ROOT}/.venv_cosmos313/bin/python}"
COSMOS_ROOT="${COSMOS_ROOT:-${ROOT}/experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam}"
COSMOS_LOCAL_TOKENIZER_DIR="${COSMOS_LOCAL_TOKENIZER_DIR:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${ROOT}/checkpoints/cosmos3/wan22_vae/Wan2.2_VAE.pth}"
DP_CHECKPOINT="${DP_CHECKPOINT:-${ROOT}/experiments/maniskill/dp_checkpoint/run_90201/checkpoints/best_eval_success_at_end.pt}"
REGISTRY="${REGISTRY:-${ROOT}/experiments/maniskill/data/active}"
INSPECT_SCRIPT="${ROOT}/scripts/world_model/inspect_joint_dp_cosmos_interfaces.py"
LIBFFI_COMPAT_DIR="${LIBFFI_COMPAT_DIR:-${ROOT}/.venv_cosmos313/lib_compat}"

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

cosmos_nvidia_lib_dirs() {
  local venv
  venv="$(dirname "$(dirname "${COSMOS_PYTHON}")")"
  local dirs=()
  local site d
  shopt -s nullglob
  for site in "${venv}"/lib/python*/site-packages; do
    for d in "${site}"/nvidia/*/lib "${site}"/nvidia/*/lib64; do
      [[ -d "${d}" ]] && dirs+=("${d}")
    done
  done
  shopt -u nullglob
  local IFS=:
  printf '%s' "${dirs[*]}"
}

refresh_cosmos_ld_library_path() {
  local nvidia_libs
  nvidia_libs="$(cosmos_nvidia_lib_dirs)"
  if [[ -d "${LIBFFI_COMPAT_DIR}" && -n "${nvidia_libs}" ]]; then
    export LD_LIBRARY_PATH="${LIBFFI_COMPAT_DIR}:${nvidia_libs}:${LD_LIBRARY_PATH:-}"
  elif [[ -d "${LIBFFI_COMPAT_DIR}" ]]; then
    export LD_LIBRARY_PATH="${LIBFFI_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
  elif [[ -n "${nvidia_libs}" ]]; then
    export LD_LIBRARY_PATH="${nvidia_libs}:${LD_LIBRARY_PATH:-}"
  fi
}

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
cd "${ROOT}"

export ROOT
export REGISTRY
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY=
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export AWS_EC2_METADATA_DISABLED=true

{
  echo "timestamp=$(date -Is)"
  echo "phase=02_joint_training"
  echo "stage=interface_inspect"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID}"
  echo "node=$(hostname)"
  echo "project_python=${PROJECT_PYTHON}"
  echo "cosmos_python=${COSMOS_PYTHON}"
  echo "registry=${REGISTRY}"
  echo "dp_checkpoint=${DP_CHECKPOINT}"
  echo "cosmos_root=${COSMOS_ROOT}"
  echo "cosmos_local_tokenizer_dir=${COSMOS_LOCAL_TOKENIZER_DIR}"
  echo "wan_vae_path=${WAN_VAE_PATH}"
  echo "controller_action_contract=pd_ee_delta_pose"
  echo "evidence_type=diagnostic_interface_check"
  echo "method_evidence_allowed=false"
  echo "training_started=false"
  echo "data_generation_started=false"
  echo "uses_toy_model=false"
  echo "forbidden_state_intervention_used=false"
} | tee "${RUN_DIR}/manifest.txt" | tee -a "${LOG_FILE}"

nvidia-smi 2>&1 | tee -a "${LOG_FILE}" || true

echo "joint_overfit_abcd_gate_start=$(date -Is)" | tee -a "${LOG_FILE}"
"${ROOT}/scripts/world_model/require_dataset_training_inputs_ready.sh" joint_overfit_abcd \
  > "${RUN_DIR}/joint_overfit_abcd_gate.txt" 2>&1
sed 's/^/joint_overfit_abcd_gate_/' "${RUN_DIR}/joint_overfit_abcd_gate.txt" | tee -a "${LOG_FILE}"
echo "joint_overfit_abcd_gate_status=ok" | tee -a "${LOG_FILE}"

echo "project_interface_check_start=$(date -Is)" | tee -a "${LOG_FILE}"
set +e
"${PROJECT_PYTHON}" -m py_compile "${INSPECT_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"
project_compile_status="${PIPESTATUS[0]}"
set -e
echo "project_interface_compile_status=${project_compile_status}" | tee -a "${LOG_FILE}"
if [[ "${project_compile_status}" -ne 0 ]]; then
  echo "interface_inspect_status=failed_project_compile" | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
  exit "${project_compile_status}"
fi
set +e
"${PROJECT_PYTHON}" -u "${INSPECT_SCRIPT}" \
  --mode project \
  --root "${ROOT}" \
  --registry "${REGISTRY}" \
  --output-dir "${RUN_DIR}" \
  --dp-checkpoint "${DP_CHECKPOINT}" \
  2>&1 | tee -a "${LOG_FILE}"
project_status="${PIPESTATUS[0]}"
set -e
echo "project_interface_check_status=${project_status}" | tee -a "${LOG_FILE}"
if [[ "${project_status}" -ne 0 ]]; then
  echo "interface_inspect_status=failed_project_interface" | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
  exit "${project_status}"
fi

refresh_cosmos_ld_library_path
export PYTHONPATH="${ROOT}/external/cosmos-framework:${PYTHONPATH:-}"
export COSMOS3_LOCAL_TOKENIZER_DIR="${COSMOS_LOCAL_TOKENIZER_DIR}"
export WAN_VAE_PATH="${WAN_VAE_PATH}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

echo "cosmos_interface_check_start=$(date -Is)" | tee -a "${LOG_FILE}"
set +e
"${COSMOS_PYTHON}" -m py_compile "${INSPECT_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"
cosmos_compile_status="${PIPESTATUS[0]}"
set -e
echo "cosmos_interface_compile_status=${cosmos_compile_status}" | tee -a "${LOG_FILE}"
if [[ "${cosmos_compile_status}" -ne 0 ]]; then
  echo "interface_inspect_status=failed_cosmos_compile" | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
  exit "${cosmos_compile_status}"
fi
set +e
"${COSMOS_PYTHON}" -u "${INSPECT_SCRIPT}" \
  --mode cosmos \
  --root "${ROOT}" \
  --output-dir "${RUN_DIR}" \
  --cosmos-root "${COSMOS_ROOT}" \
  --cosmos-local-tokenizer-dir "${COSMOS_LOCAL_TOKENIZER_DIR}" \
  --wan-vae-path "${WAN_VAE_PATH}" \
  2>&1 | tee -a "${LOG_FILE}"
cosmos_status="${PIPESTATUS[0]}"
set -e
echo "cosmos_interface_check_status=${cosmos_status}" | tee -a "${LOG_FILE}"
if [[ "${cosmos_status}" -ne 0 ]]; then
  echo "interface_inspect_status=failed_cosmos_interface" | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
  exit "${cosmos_status}"
fi

{
  echo "interface_inspect_status=complete"
  echo "project_summary=${RUN_DIR}/project_interface_summary.json"
  echo "cosmos_summary=${RUN_DIR}/cosmos_interface_summary.json"
  echo "completed_at=$(date -Is)"
} | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
