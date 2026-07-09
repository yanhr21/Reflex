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

RUN_GROUP="${RUN_GROUP:-joint_cosmos_condition}"
RUN_NAME="${RUN_NAME:-overfit01}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/${RUN_GROUP}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/02_joint_training/${RUN_GROUP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
PROJECT_PYTHON="${PROJECT_PYTHON:-${ROOT}/.venv/bin/python}"
JOINT_DATASET_DIR="${JOINT_DATASET_DIR:-${ROOT}/experiments/maniskill/runs/02_joint_training/joint_overfit_dataset/overfit01}"
CONDITION_ROOT="${CONDITION_ROOT:-${RUN_DIR}/condition_root}"
BUILDER="${ROOT}/scripts/world_model/build_joint_dp_cosmos_sft_condition_root.py"
PREFLIGHT="${ROOT}/scripts/world_model/preflight_cosmos3_full_episode_wam_contract.py"
SOURCE_VIDEO_FRAMES="${SOURCE_VIDEO_FRAMES:-300}"
PREFLIGHT_VIDEO_FRAMES="${PREFLIGHT_VIDEO_FRAMES:-${EXPECTED_VIDEO_FRAMES:-297}}"
EXPECTED_VIDEO_FRAMES="${EXPECTED_VIDEO_FRAMES:-297}"
EXPECTED_STEPS="${EXPECTED_STEPS:-296}"
EXPECTED_SOURCE_EPISODES="${EXPECTED_SOURCE_EPISODES:-8}"

case "${RUN_DIR}" in
  "${ROOT}/experiments/maniskill/runs/02_joint_training/"*) ;;
  *)
    echo "refusing_output_dir_outside_02_joint_training=true" >&2
    echo "run_dir=${RUN_DIR}" >&2
    exit 41
    ;;
esac
if [[ -e "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing_existing_nonempty_run_dir=true" >&2
  echo "run_dir=${RUN_DIR}" >&2
  exit 42
fi
if [[ ! -s "${JOINT_DATASET_DIR}/batch_inspection.json" ]]; then
  echo "refusing_missing_joint_batch_inspection=true" >&2
  echo "joint_dataset_dir=${JOINT_DATASET_DIR}" >&2
  exit 43
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
cd "${ROOT}"

export ROOT
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY=
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export AWS_EC2_METADATA_DISABLED=true

{
  echo "timestamp=$(date -Is)"
  echo "phase=02_joint_training"
  echo "stage=joint_cosmos_condition"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "condition_root=${CONDITION_ROOT}"
  echo "log_file=${LOG_FILE}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID}"
  echo "node=$(hostname)"
  echo "joint_dataset_dir=${JOINT_DATASET_DIR}"
  echo "project_python=${PROJECT_PYTHON}"
  echo "source_video_frames=${SOURCE_VIDEO_FRAMES}"
  echo "preflight_video_frames=${PREFLIGHT_VIDEO_FRAMES}"
  echo "expected_video_frames=${EXPECTED_VIDEO_FRAMES}"
  echo "expected_steps=${EXPECTED_STEPS}"
  echo "expected_source_episodes=${EXPECTED_SOURCE_EPISODES}"
  echo "method_evidence_allowed=false"
  echo "training_started=false"
  echo "data_generation_started=false"
  echo "uses_toy_model=false"
} | tee "${RUN_DIR}/allocation_manifest.txt" | tee -a "${LOG_FILE}"

nvidia-smi 2>&1 | tee -a "${LOG_FILE}" || true

echo "compile_start=$(date -Is)" | tee -a "${LOG_FILE}"
"${PROJECT_PYTHON}" -m py_compile "${BUILDER}" "${PREFLIGHT}" 2>&1 | tee -a "${LOG_FILE}"
echo "compile_status=0" | tee -a "${LOG_FILE}"

echo "condition_root_build_start=$(date -Is)" | tee -a "${LOG_FILE}"
"${PROJECT_PYTHON}" -u "${BUILDER}" \
  --joint-dataset-dir "${JOINT_DATASET_DIR}" \
  --output-root "${CONDITION_ROOT}" \
  --source-video-frames "${SOURCE_VIDEO_FRAMES}" \
  --expected-video-frames "${EXPECTED_VIDEO_FRAMES}" \
  --expected-steps "${EXPECTED_STEPS}" \
  2>&1 | tee -a "${LOG_FILE}"
echo "condition_root_build_status=0" | tee -a "${LOG_FILE}"

echo "condition_root_preflight_start=$(date -Is)" | tee -a "${LOG_FILE}"
"${PROJECT_PYTHON}" -u "${PREFLIGHT}" \
  --condition-root "${CONDITION_ROOT}" \
  --expected-source-episodes "${EXPECTED_SOURCE_EPISODES}" \
  --expected-source-video-frames "${SOURCE_VIDEO_FRAMES}" \
  --expected-exported-video-frames "${PREFLIGHT_VIDEO_FRAMES}" \
  --expected-video-frames "${EXPECTED_VIDEO_FRAMES}" \
  --expected-action-steps "${EXPECTED_STEPS}" \
  --expected-action-dim 32 \
  --output-json "${RUN_DIR}/condition_preflight.json" \
  --output-md "${RUN_DIR}/condition_preflight.md" \
  2>&1 | tee -a "${LOG_FILE}"
echo "condition_root_preflight_status=0" | tee -a "${LOG_FILE}"

{
  echo "joint_cosmos_condition_status=complete"
  echo "condition_root=${CONDITION_ROOT}"
  echo "condition_preflight=${RUN_DIR}/condition_preflight.json"
  echo "completed_at=$(date -Is)"
} | tee "${RUN_DIR}/classification.txt" | tee -a "${LOG_FILE}"
