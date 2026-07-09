#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

PHASE="03_oracle"
RUN_GROUP="${RUN_GROUP:-render_probe}"
RUN_NAME="${RUN_NAME:-try01}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/${PHASE}/${RUN_GROUP}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${PHASE}/${RUN_GROUP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

validate_short_component() {
  local kind="$1"
  local component="$2"
  if [[ -z "${component}" || "${component}" == "." || "${component}" == ".." ]]; then
    echo "invalid_short_name=true kind=${kind} value=${component}" >&2
    exit 42
  fi
  if [[ ! "${component}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "invalid_short_name=true kind=${kind} value=${component} reason=bad_characters" >&2
    exit 42
  fi
  if [[ "${#component}" -gt 32 ]]; then
    echo "invalid_short_name=true kind=${kind} value=${component} reason=too_long_max32" >&2
    exit 42
  fi
  if [[ "${component}" =~ p[0-9][0-9]_ || "${component}" =~ [0-9]{8} || "${component}" =~ full_pipeline_ || "${component}" =~ server[0-9]+ || "${component}" =~ mgmtserver ]]; then
    echo "invalid_short_name=true kind=${kind} value=${component} reason=metadata_belongs_in_manifest" >&2
    exit 42
  fi
}

validate_short_path_components() {
  local kind="$1"
  local value="$2"
  local IFS='/'
  read -r -a parts <<< "${value}"
  for part in "${parts[@]}"; do
    validate_short_component "${kind}" "${part}"
  done
}

PROJECT_PYTHON="${PROJECT_PYTHON:-${ROOT}/.venv/bin/python}"
RENDER_SHADER_PACK="${RENDER_SHADER_PACK:-minimal}"
RENDER_CANARY_API="${RENDER_CANARY_API:-gym}"
RENDER_PROBE_TIMEOUT="${RENDER_PROBE_TIMEOUT:-75s}"
FULL_RUN_NAME="${FULL_RUN_NAME:-}"
FULL_RUN_GROUP="${FULL_RUN_GROUP:-peg_disturb}"
FULL_RUN_SCRIPT="${FULL_RUN_SCRIPT:-scripts/slurm/phase03_peg_disturb.sh}"

validate_short_path_components "run_group" "${RUN_GROUP}"
validate_short_component "run_name" "${RUN_NAME}"
if [[ -n "${FULL_RUN_NAME}" ]]; then
  validate_short_component "full_run_name" "${FULL_RUN_NAME}"
  validate_short_path_components "full_run_group" "${FULL_RUN_GROUP}"
fi
run_root_prefix="${ROOT}/experiments/maniskill/runs/${PHASE}/"
if [[ "${RUN_DIR}" == "${run_root_prefix}"* ]]; then
  validate_short_path_components "run_dir" "${RUN_DIR#${run_root_prefix}}"
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
export PYTHONPATH="${ROOT}/deps/ManiSkill_clean/examples/baselines/diffusion_policy:${ROOT}/deps/ManiSkill_clean:${ROOT}:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export DISPLAY=

{
  echo "timestamp=$(date -Is)"
  echo "phase=${PHASE}"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID}"
  echo "node=$(hostname -s)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "slurm_step_gpus=${SLURM_STEP_GPUS:-}"
  echo "render_shader_pack=${RENDER_SHADER_PACK}"
  echo "render_canary_api=${RENDER_CANARY_API}"
  echo "render_probe_timeout=${RENDER_PROBE_TIMEOUT}"
  echo "full_run_name=${FULL_RUN_NAME}"
  echo "full_run_group=${FULL_RUN_GROUP}"
  echo "full_run_script=${FULL_RUN_SCRIPT}"
  echo "method_evidence_allowed=false"
  echo "probe_only_until_full_run_starts=true"
} | tee "${RUN_DIR}/manifest.txt" | tee "${LOG_FILE}"

nvidia-smi -L 2>&1 | tee -a "${LOG_FILE}" || true
nvidia-smi 2>&1 | tee -a "${LOG_FILE}" || true

IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES:-0}"
first_good=""

for dev in "${visible_devices[@]}"; do
  dev="$(echo "${dev}" | xargs)"
  [[ -z "${dev}" ]] && continue
  probe_dir="${RUN_DIR}/gpu_${dev}"
  mkdir -p "${probe_dir}"
  {
    echo "render_probe_device_start=${dev}"
    echo "render_probe_device_dir=${probe_dir}"
  } | tee -a "${LOG_FILE}"
  set +e
  CUDA_VISIBLE_DEVICES="${dev}" timeout -k 10s "${RENDER_PROBE_TIMEOUT}" \
    "${PROJECT_PYTHON}" -u scripts/world_model/render_min_canary.py \
      --output-dir "${probe_dir}" \
      --shader-pack "${RENDER_SHADER_PACK}" \
      --render-api "${RENDER_CANARY_API}" \
      2>&1 | tee -a "${LOG_FILE}"
  status="${PIPESTATUS[0]}"
  set -e
  echo "render_probe_device_exit_code_${dev}=${status}" | tee -a "${LOG_FILE}"
  if [[ "${status}" -eq 0 && -z "${first_good}" ]]; then
    first_good="${dev}"
    echo "render_probe_first_good_device=${first_good}" | tee -a "${LOG_FILE}"
  fi
done

if [[ -z "${first_good}" ]]; then
  {
    echo "phase03_status=blocked_no_render_capable_visible_gpu"
    echo "method_evidence_allowed=false"
    echo "physical_insertion_success=false"
    echo "reason=No visible GPU rendered the first RGB canary frame; no Oracle rollout started."
  } | tee -a "${LOG_FILE}" | tee "${RUN_DIR}/classification.txt"
  exit 124
fi

{
  echo "phase03_status=render_probe_found_good_gpu"
  echo "render_probe_first_good_device=${first_good}"
  echo "method_evidence_allowed=false"
  echo "physical_insertion_success=false"
} | tee -a "${LOG_FILE}" | tee "${RUN_DIR}/classification.txt"

if [[ -n "${FULL_RUN_NAME}" ]]; then
  if [[ ! -f "${FULL_RUN_SCRIPT}" ]]; then
    echo "phase03_status=blocked_full_run_script_missing script=${FULL_RUN_SCRIPT}" | tee -a "${LOG_FILE}" | tee "${RUN_DIR}/classification.txt"
    exit 46
  fi
  echo "phase03_status=starting_full_run_on_good_gpu run_group=${FULL_RUN_GROUP} run_name=${FULL_RUN_NAME} cuda_visible_devices=${first_good}" | tee -a "${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="${first_good}" RUN_GROUP="${FULL_RUN_GROUP}" RUN_NAME="${FULL_RUN_NAME}" bash "${FULL_RUN_SCRIPT}"
fi
