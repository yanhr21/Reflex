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

RUN_ID="${RUN_ID:-p02_cosmos_v2v_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}_$(hostname)}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_cosmos_imagination/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/02_cosmos_imagination}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/experiments/maniskill/data/fix3_733}"
ACTIVE_COSMOS_ROOT="${ACTIVE_COSMOS_ROOT:-${ROOT}/experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam}"
LATEST_CKPT="$(tr -d '[:space:]' < "${ACTIVE_COSMOS_ROOT}/checkpoints/latest_checkpoint.txt")"
COSMOS_CHECKPOINT_PATH="${COSMOS_CHECKPOINT_PATH:-${ACTIVE_COSMOS_ROOT}/checkpoints/${LATEST_CKPT}}"
COSMOS_CONFIG_FILE="${COSMOS_CONFIG_FILE:-${ACTIVE_COSMOS_ROOT}/config.yaml}"
COSMOS_LOCAL_TOKENIZER_DIR="${COSMOS_LOCAL_TOKENIZER_DIR:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${ROOT}/.venv_cosmos313/bin/python}"
PROJECT_PYTHON="${PROJECT_PYTHON:-${ROOT}/.venv/bin/python}"
LIBFFI_COMPAT_DIR="${LIBFFI_COMPAT_DIR:-${ROOT}/.venv_cosmos313/lib_compat}"
SCENARIOS="${SCENARIOS:-none,hole_late_fast_shift,hole_late_reverse,hole_late_continuous_insert,peg_drop,peg_disturb}"
SAMPLES_PER_SCENARIO="${SAMPLES_PER_SCENARIO:-1}"
MAX_FRAMES="${MAX_FRAMES:-96}"
PREFIX_FRAMES="${PREFIX_FRAMES:-16}"
FPS="${FPS:-24}"
COSMOS_NUM_FRAMES="${COSMOS_NUM_FRAMES:-121}"
COSMOS_RESOLUTION="${COSMOS_RESOLUTION:-256}"
COSMOS_ASPECT_RATIO="${COSMOS_ASPECT_RATIO:-1,1}"
COSMOS_EXPERIMENT="${COSMOS_EXPERIMENT:-vision_sft_nano}"
RUN_COSMOS="${RUN_COSMOS:-1}"

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

main() {
  cd "${ROOT}"
  mkdir -p "${RUN_DIR}" "${LOG_DIR}"
  DATASET_PATH="${DATASET_PATH:-${RUN_DIR}/cosmos_config_dataset_stub}"
  mkdir -p "${DATASET_PATH}/train" "${DATASET_PATH}/val"
  : > "${DATASET_PATH}/train/video_dataset_file.jsonl"
  : > "${DATASET_PATH}/val/video_dataset_file.jsonl"
  export DATASET_PATH
  refresh_cosmos_ld_library_path
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
  export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
  export DISPLAY=

  {
    echo "timestamp=$(date -Is)"
    echo "phase=02_cosmos_imagination"
    echo "run_id=${RUN_ID}"
    echo "run_dir=${RUN_DIR}"
    echo "log_file=${LOG_FILE}"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "slurm_step_id=${SLURM_STEP_ID}"
    echo "node=$(hostname)"
    echo "data_root=${DATA_ROOT}"
    echo "cosmos_config_dataset_path=${DATASET_PATH}"
    echo "active_cosmos_root=${ACTIVE_COSMOS_ROOT}"
    echo "cosmos_checkpoint_path=${COSMOS_CHECKPOINT_PATH}"
    echo "cosmos_config_file=${COSMOS_CONFIG_FILE}"
    echo "cosmos_experiment=${COSMOS_EXPERIMENT}"
    echo "cosmos_local_tokenizer_dir=${COSMOS_LOCAL_TOKENIZER_DIR}"
    echo "project_python=${PROJECT_PYTHON}"
    echo "cosmos_python=${COSMOS_PYTHON}"
    echo "vk_icd_filenames=${VK_ICD_FILENAMES}"
    echo "hdf5_use_file_locking=${HDF5_USE_FILE_LOCKING}"
    echo "scenarios=${SCENARIOS}"
    echo "samples_per_scenario=${SAMPLES_PER_SCENARIO}"
    echo "method_evidence_allowed=false"
    echo "boundary=phase02_rgb_cosmos_imagination_only_no_controller_success_claim"
    echo "forbidden_state_intervention_used=false"
  } | tee "${RUN_DIR}/manifest.txt" | tee "${LOG_FILE}"

  nvidia-smi 2>&1 | tee -a "${LOG_FILE}"

  "${PROJECT_PYTHON}" -m py_compile \
    scripts/world_model/phase02_extract_rgb_v2v_inputs.py \
    2>&1 | tee -a "${LOG_FILE}"

  set +e
  "${PROJECT_PYTHON}" -u scripts/world_model/phase02_extract_rgb_v2v_inputs.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${RUN_DIR}" \
    --scenarios "${SCENARIOS}" \
    --samples-per-scenario "${SAMPLES_PER_SCENARIO}" \
    --max-frames "${MAX_FRAMES}" \
    --prefix-frames "${PREFIX_FRAMES}" \
    --fps "${FPS}" \
    --cosmos-num-frames "${COSMOS_NUM_FRAMES}" \
    --cosmos-resolution "${COSMOS_RESOLUTION}" \
    --cosmos-aspect-ratio "${COSMOS_ASPECT_RATIO}" \
    2>&1 | tee -a "${LOG_FILE}"
  extract_status=${PIPESTATUS[0]}
  set -e
  echo "extract_status=${extract_status}" | tee -a "${LOG_FILE}"
  if [[ "${extract_status}" -eq 42 ]]; then
    echo "direct_h5_rgb_status=blocked_no_rgb_dataset_found_running_state_audit_renderer" | tee -a "${LOG_FILE}"
    "${PROJECT_PYTHON}" -m py_compile \
      scripts/world_model/phase02_render_state_audit_rgb_v2.py \
      2>&1 | tee -a "${LOG_FILE}"
    set +e
    "${PROJECT_PYTHON}" -u scripts/world_model/phase02_render_state_audit_rgb_v2.py \
      --data-root "${DATA_ROOT}" \
      --output-dir "${RUN_DIR}" \
      --scenarios "${SCENARIOS}" \
      --samples-per-scenario "${SAMPLES_PER_SCENARIO}" \
      --max-frames "${MAX_FRAMES}" \
      --prefix-frames "${PREFIX_FRAMES}" \
      --fps "${FPS}" \
      --cosmos-num-frames "${COSMOS_NUM_FRAMES}" \
      --cosmos-resolution "${COSMOS_RESOLUTION}" \
      --cosmos-aspect-ratio "${COSMOS_ASPECT_RATIO}" \
      2>&1 | tee -a "${LOG_FILE}"
    render_status=${PIPESTATUS[0]}
    set -e
    echo "state_audit_render_status=${render_status}" | tee -a "${LOG_FILE}"
    if [[ "${render_status}" -ne 0 ]]; then
      echo "phase02_status=blocked_or_failed_before_cosmos_state_audit_render_failed" | tee -a "${LOG_FILE}" "${RUN_DIR}/classification.txt"
      exit "${render_status}"
    fi
  elif [[ "${extract_status}" -ne 0 ]]; then
    echo "phase02_status=blocked_or_failed_before_cosmos" | tee -a "${LOG_FILE}" "${RUN_DIR}/classification.txt"
    exit "${extract_status}"
  fi

  if [[ "${RUN_COSMOS}" != "1" ]]; then
    echo "phase02_status=rgb_inputs_ready_cosmos_skipped_by_RUN_COSMOS" | tee -a "${LOG_FILE}" "${RUN_DIR}/classification.txt"
    exit 0
  fi

  if ! compgen -G "${RUN_DIR}/cosmos_inputs/*.json" > /dev/null; then
    echo "phase02_status=blocked_no_cosmos_input_json" | tee -a "${LOG_FILE}" "${RUN_DIR}/classification.txt"
    exit 43
  fi

  export PYTHONPATH="${ROOT}/external/cosmos-framework:${PYTHONPATH:-}"
  export WAN_VAE_PATH="${ROOT}/checkpoints/cosmos3/wan22_vae/Wan2.2_VAE.pth"
  export COSMOS3_LOCAL_TOKENIZER_DIR="${COSMOS_LOCAL_TOKENIZER_DIR}"
  export AWS_EC2_METADATA_DISABLED=true
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

  set +e
  "${COSMOS_PYTHON}" -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    --no-guardrails \
    --config-file="${COSMOS_CONFIG_FILE}" \
    --experiment="${COSMOS_EXPERIMENT}" \
    -i "${RUN_DIR}/cosmos_inputs/*.json" \
    -o "${RUN_DIR}/cosmos_outputs" \
    --checkpoint-path "${COSMOS_CHECKPOINT_PATH}" \
    --seed=0 \
    2>&1 | tee -a "${LOG_FILE}"
  cosmos_status=${PIPESTATUS[0]}
  set -e
  echo "cosmos_status=${cosmos_status}" | tee -a "${LOG_FILE}"

  if [[ "${cosmos_status}" -eq 0 ]]; then
    echo "phase02_status=cosmos_v2v_complete_needs_visual_review" | tee -a "${LOG_FILE}" "${RUN_DIR}/classification.txt"
  else
    echo "phase02_status=cosmos_v2v_failed" | tee -a "${LOG_FILE}" "${RUN_DIR}/classification.txt"
  fi
  exit "${cosmos_status}"
}

main "$@"
