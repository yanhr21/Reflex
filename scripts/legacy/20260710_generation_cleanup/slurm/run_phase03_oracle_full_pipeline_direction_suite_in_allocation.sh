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
SUITE_GROUP="${SUITE_GROUP:-direction_suite}"

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

if [[ -z "${SUITE_NAME:-}" ]]; then
  if [[ -n "${SUITE_DIR:-}" ]]; then
    SUITE_NAME="$(basename "${SUITE_DIR}")"
  else
    SUITE_ROOT="${ROOT}/experiments/maniskill/runs/${PHASE}/${SUITE_GROUP}"
    for try_idx in $(seq 1 99); do
      candidate="$(printf 'try%02d' "${try_idx}")"
      if [[ ! -e "${SUITE_ROOT}/${candidate}" ]]; then
        SUITE_NAME="${candidate}"
        break
      fi
    done
    if [[ -z "${SUITE_NAME:-}" ]]; then
      echo "no_available_short_suite_name=true group=${SUITE_GROUP}" >&2
      exit 41
    fi
  fi
fi

SUITE_ID="${SUITE_ID:-${SUITE_GROUP}/${SUITE_NAME}}"
SUITE_DIR="${SUITE_DIR:-${ROOT}/experiments/maniskill/runs/${PHASE}/${SUITE_GROUP}/${SUITE_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${PHASE}/${SUITE_GROUP}}"
SUITE_LOG="${SUITE_LOG:-${LOG_DIR}/${SUITE_NAME}.log}"

validate_short_path_components "suite_group" "${SUITE_GROUP}"
validate_short_component "suite_name" "${SUITE_NAME}"
suite_root_prefix="${ROOT}/experiments/maniskill/runs/${PHASE}/"
if [[ "${SUITE_DIR}" == "${suite_root_prefix}"* ]]; then
  validate_short_path_components "suite_dir" "${SUITE_DIR#${suite_root_prefix}}"
fi

TARGET_MOTION_MAG="${TARGET_MOTION_MAG:-0.0125}"
TARGET_MOTION_PER_STEP="${TARGET_MOTION_PER_STEP:-0.00125}"
NEAR_TARGET_L2="${NEAR_TARGET_L2:-0.16}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-420}"
MAX_FINISHER_STEPS="${MAX_FINISHER_STEPS:-180}"
FINISHER_CONTROLLER="${FINISHER_CONTROLLER:-diffusion_policy}"
MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER="${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER:-1}"
COSMOS_ACTION_HORIZON="${COSMOS_ACTION_HORIZON:-8}"
MAX_COSMOS_ROUNDS="${MAX_COSMOS_ROUNDS:-4}"

mkdir -p "${SUITE_DIR}" "${LOG_DIR}"

cat > "${SUITE_DIR}/manifest.txt" <<EOF
timestamp=$(date -Is)
phase=${PHASE}
suite_id=${SUITE_ID}
suite_group=${SUITE_GROUP}
suite_name=${SUITE_NAME}
suite_dir=${SUITE_DIR}
log_file=${SUITE_LOG}
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
node=$(hostname -s)
method_evidence_allowed=false
boundary=direction_suite_only_after_single_case_is_not_overall_completion
note=Runs directional target-motion coverage. Peg/stick disturbance and multi-key coverage still require separate evidence.
target_motion_mag=${TARGET_MOTION_MAG}
target_motion_per_step=${TARGET_MOTION_PER_STEP}
near_target_l2=${NEAR_TARGET_L2}
max_episode_steps=${MAX_EPISODE_STEPS}
max_finisher_steps=${MAX_FINISHER_STEPS}
finisher_controller=${FINISHER_CONTROLLER}
min_cosmos_dynamic_actions_before_finisher=${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER}
EOF
cat "${SUITE_DIR}/manifest.txt" | tee "${SUITE_LOG}"

SUITE_FAILURES=0

run_case() {
  local name="$1"
  local dx="$2"
  local dy="$3"
  local dz="$4"
  local seed="$5"
  local run_dir="${SUITE_DIR}/${name}"
  local run_log_dir="${LOG_DIR}/${SUITE_NAME}"
  local run_log="${run_log_dir}/${name}.log"
  {
    echo "suite_case_start=$(date -Is)"
    echo "scenario_name=${name}"
    echo "target_motion_x=${dx}"
    echo "target_motion_y=${dy}"
    echo "target_motion_z=${dz}"
    echo "seed=${seed}"
  } | tee -a "${SUITE_LOG}"

  set +e
  SCENARIO_NAME="${name}" \
  TARGET_MOTION_X="${dx}" \
  TARGET_MOTION_Y="${dy}" \
  TARGET_MOTION_Z="${dz}" \
  TARGET_MOTION_PER_STEP="${TARGET_MOTION_PER_STEP}" \
  SEED="${seed}" \
  NEAR_TARGET_L2="${NEAR_TARGET_L2}" \
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
  MAX_FINISHER_STEPS="${MAX_FINISHER_STEPS}" \
  FINISHER_CONTROLLER="${FINISHER_CONTROLLER}" \
  MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER="${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER}" \
  COSMOS_ACTION_HORIZON="${COSMOS_ACTION_HORIZON}" \
  MAX_COSMOS_ROUNDS="${MAX_COSMOS_ROUNDS}" \
  RUN_GROUP="${SUITE_GROUP}/${SUITE_NAME}" \
  RUN_NAME="${name}" \
  RUN_ID="${SUITE_ID}/${name}" \
  RUN_DIR="${run_dir}" \
  LOG_DIR="${run_log_dir}" \
  LOG_FILE="${run_log}" \
  bash scripts/slurm/run_phase03_oracle_full_pipeline_in_allocation.sh \
    2>&1 | tee -a "${SUITE_LOG}"
  local case_status="${PIPESTATUS[0]}"
  set -e
  echo "scenario_name=${name} exit_code=${case_status}" | tee -a "${SUITE_LOG}" | tee -a "${SUITE_DIR}/case_status.tsv"
  if [[ "${case_status}" -ne 0 ]]; then
    SUITE_FAILURES=$((SUITE_FAILURES + 1))
  fi
}

run_case "target_y_positive" "0.0" "${TARGET_MOTION_MAG}" "0.0" "${SEED_Y_POS:-2}"
run_case "target_y_negative" "0.0" "-${TARGET_MOTION_MAG}" "0.0" "${SEED_Y_NEG:-3}"
run_case "target_x_positive" "${TARGET_MOTION_MAG}" "0.0" "0.0" "${SEED_X_POS:-4}"
run_case "target_x_negative" "-${TARGET_MOTION_MAG}" "0.0" "0.0" "${SEED_X_NEG:-5}"

cat > "${SUITE_DIR}/classification.txt" <<'EOF'
phase03_suite_status=direction_suite_finished_needs_artifact_and_video_review
method_evidence_allowed=false
physical_insertion_success=false
overall_oracle_complete=false
reason=Directional suite is not complete without visual success review, peg/stick disturbance, and multiple approved 733 keys.
EOF
{
  echo "suite_failed_cases=${SUITE_FAILURES}"
  echo "suite_exit_code=$([[ "${SUITE_FAILURES}" -eq 0 ]] && echo 0 || echo 3)"
} | tee -a "${SUITE_LOG}" | tee -a "${SUITE_DIR}/classification.txt"
cat "${SUITE_DIR}/classification.txt" | tee -a "${SUITE_LOG}"
if [[ "${SUITE_FAILURES}" -ne 0 ]]; then
  exit 3
fi
