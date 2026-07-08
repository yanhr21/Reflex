#!/usr/bin/env bash

DATASET_RUNTIME_CONTEXT_SOURCED=true
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  DATASET_RUNTIME_CONTEXT_SOURCED=false
  set -euo pipefail
fi

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_GROUP="${RUN_GROUP:-}"
RUN_NAME="${RUN_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
LOG_FILE="${LOG_FILE:-}"
DATASET_SMOKE_ONLY="${DATASET_SMOKE_ONLY:-true}"
HUMAN_REVIEW_REQUIRED="${HUMAN_REVIEW_REQUIRED:-true}"
LARGE_SCALE_PRODUCTION_ALLOWED="${LARGE_SCALE_PRODUCTION_ALLOWED:-false}"

runtime_fail() {
  echo "dataset_runtime_context_ready=false" >&2
  echo "reason=$1" >&2
  shift || true
  for item in "$@"; do
    echo "${item}" >&2
  done
  if [[ "${DATASET_RUNTIME_CONTEXT_SOURCED}" == "true" ]]; then
    return 30
  fi
  exit 30
}

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  runtime_fail \
    "not_inside_compute_srun_step" \
    "refusing_login_node_execution=true" \
    "slurm_job_id=${SLURM_JOB_ID:-}" \
    "slurm_step_id=${SLURM_STEP_ID:-}" || return $?
fi

if [[ -z "${RUN_GROUP}" || -z "${RUN_NAME}" ]]; then
  runtime_fail "run_group_or_run_name_missing" \
    "run_group=${RUN_GROUP}" \
    "run_name=${RUN_NAME}" || return $?
fi

if [[ "${RUN_GROUP}" =~ p03_|full_pipeline|[0-9]{8}|server[0-9]+|job[0-9]+ ]]; then
  runtime_fail "unreadable_run_group" "run_group=${RUN_GROUP}" || return $?
fi
if [[ "${RUN_NAME}" =~ p03_|full_pipeline|[0-9]{8}|server[0-9]+|job[0-9]+ ]]; then
  runtime_fail "unreadable_run_name" "run_name=${RUN_NAME}" || return $?
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${ROOT}/experiments/maniskill/runs/01_dataset/${RUN_GROUP}/${RUN_NAME}"
fi
if [[ -z "${LOG_FILE}" ]]; then
  LOG_FILE="${ROOT}/logs/01_dataset/${RUN_GROUP}/${RUN_NAME}.log"
fi

case "${OUTPUT_DIR}" in
  "${ROOT}/experiments/maniskill/runs/01_dataset/"*) ;;
  *)
    runtime_fail "output_dir_outside_active_dataset_runs" \
      "output_dir=${OUTPUT_DIR}" || return $?
    ;;
esac

case "${LOG_FILE}" in
  "${ROOT}/logs/01_dataset/"*) ;;
  *)
    runtime_fail "log_file_outside_active_dataset_logs" \
      "log_file=${LOG_FILE}" || return $?
    ;;
esac

case "${DATASET_SMOKE_ONLY}" in true|false) ;; *)
  runtime_fail "dataset_smoke_only_must_be_boolean" \
    "dataset_smoke_only=${DATASET_SMOKE_ONLY}" || return $?
  ;;
esac
case "${HUMAN_REVIEW_REQUIRED}" in true|false) ;; *)
  runtime_fail "human_review_required_must_be_boolean" \
    "human_review_required=${HUMAN_REVIEW_REQUIRED}" || return $?
  ;;
esac
case "${LARGE_SCALE_PRODUCTION_ALLOWED}" in true|false) ;; *)
  runtime_fail "large_scale_production_allowed_must_be_boolean" \
    "large_scale_production_allowed=${LARGE_SCALE_PRODUCTION_ALLOWED}" || return $?
  ;;
esac

if [[ "${DATASET_SMOKE_ONLY}" == "true" && "${HUMAN_REVIEW_REQUIRED}" != "true" ]]; then
  runtime_fail "smoke_must_require_human_review" || return $?
fi
if [[ "${DATASET_SMOKE_ONLY}" == "true" && "${LARGE_SCALE_PRODUCTION_ALLOWED}" != "false" ]]; then
  runtime_fail "smoke_must_not_allow_large_scale_production" || return $?
fi

export ROOT
export RUN_GROUP
export RUN_NAME
export OUTPUT_DIR
export LOG_FILE
export DATASET_SMOKE_ONLY
export HUMAN_REVIEW_REQUIRED
export LARGE_SCALE_PRODUCTION_ALLOWED
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY="${DISPLAY:-}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

echo "dataset_runtime_context_ready=true"
echo "read_only=true"
echo "submits_slurm=false"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_step_id=${SLURM_STEP_ID}"
echo "run_group=${RUN_GROUP}"
echo "run_name=${RUN_NAME}"
echo "output_dir=${OUTPUT_DIR}"
echo "log_file=${LOG_FILE}"
echo "dataset_smoke_only=${DATASET_SMOKE_ONLY}"
echo "human_review_required=${HUMAN_REVIEW_REQUIRED}"
echo "large_scale_production_allowed=${LARGE_SCALE_PRODUCTION_ALLOWED}"
echo "vk_icd_filenames=${VK_ICD_FILENAMES}"
echo "display=${DISPLAY}"
echo "hdf5_use_file_locking=${HDF5_USE_FILE_LOCKING}"
