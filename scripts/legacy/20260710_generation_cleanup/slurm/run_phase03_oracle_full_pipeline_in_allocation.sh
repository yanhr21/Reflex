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
RUN_GROUP="${RUN_GROUP:-}"

if [[ -z "${RUN_GROUP}" && -n "${RUN_DIR:-}" ]]; then
  RUN_GROUP="$(basename "$(dirname "${RUN_DIR}")")"
fi

if [[ -z "${RUN_GROUP}" ]]; then
  cat >&2 <<'EOF'
missing_required_run_group=true
reason=Set RUN_GROUP to a short case group such as h5_reverse, h5_continuous_insert, or peg_disturb. Do not use an unclassified full_pipeline default.
EOF
  exit 41
fi

if [[ "${RUN_GROUP}" == "full_pipeline" ]]; then
  cat >&2 <<'EOF'
invalid_run_group=true
value=full_pipeline
reason=Use a short case group instead of the legacy generic full_pipeline group.
EOF
  exit 42
fi

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

if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ -n "${RUN_DIR:-}" ]]; then
    RUN_NAME="$(basename "${RUN_DIR}")"
  else
    RUN_ROOT="${ROOT}/experiments/maniskill/runs/${PHASE}/${RUN_GROUP}"
    for try_idx in $(seq 1 99); do
      candidate="$(printf 'try%02d' "${try_idx}")"
      if [[ ! -e "${RUN_ROOT}/${candidate}" ]]; then
        RUN_NAME="${candidate}"
        break
      fi
    done
    if [[ -z "${RUN_NAME:-}" ]]; then
      echo "no_available_short_run_name=true group=${RUN_GROUP}" >&2
      exit 41
    fi
  fi
fi

RUN_ID="${RUN_ID:-${RUN_GROUP}/${RUN_NAME}}"
RUN_DIR="${RUN_DIR:-${ROOT}/experiments/maniskill/runs/${PHASE}/${RUN_GROUP}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/${PHASE}/${RUN_GROUP}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

validate_short_path_components "run_group" "${RUN_GROUP}"
validate_short_component "run_name" "${RUN_NAME}"
validate_short_path_components "run_id" "${RUN_ID}"
log_basename="$(basename "${LOG_FILE}")"
validate_short_component "log_file_stem" "${log_basename%.log}"
run_root_prefix="${ROOT}/experiments/maniskill/runs/${PHASE}/"
if [[ "${RUN_DIR}" == "${run_root_prefix}"* ]]; then
  validate_short_path_components "run_dir" "${RUN_DIR#${run_root_prefix}}"
fi
log_root_prefix="${ROOT}/logs/${PHASE}/"
if [[ "${LOG_FILE}" == "${log_root_prefix}"* ]]; then
  validate_short_path_components "log_file" "${LOG_FILE#${log_root_prefix}}"
fi

PROJECT_PYTHON="${PROJECT_PYTHON:-${ROOT}/.venv/bin/python}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${ROOT}/.venv_cosmos313/bin/python}"
COSMOS_FRAMEWORK_PATH="${COSMOS_FRAMEWORK_PATH:-${ROOT}/external/cosmos-framework}"
CKPT_PATH="${CKPT_PATH:-${ROOT}/experiments/maniskill/dp_checkpoint/run_90201/checkpoints/best_eval_success_at_end.pt}"
ACTIVE_COSMOS_ROOT="${ACTIVE_COSMOS_ROOT:-${ROOT}/experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam}"
LATEST_CKPT="$(tr -d '[:space:]' < "${ACTIVE_COSMOS_ROOT}/checkpoints/latest_checkpoint.txt")"
COSMOS_CHECKPOINT_PATH="${COSMOS_CHECKPOINT_PATH:-${ACTIVE_COSMOS_ROOT}/checkpoints/${LATEST_CKPT}}"
COSMOS_CONFIG_FILE="${COSMOS_CONFIG_FILE:-${ACTIVE_COSMOS_ROOT}/config.yaml}"
COSMOS_LOCAL_TOKENIZER_DIR="${COSMOS_LOCAL_TOKENIZER_DIR:-${ROOT}/checkpoints/cosmos3/Cosmos3-Nano}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${ROOT}/checkpoints/cosmos3/wan22_vae/Wan2.2_VAE.pth}"
COSMOS_NORMALIZATION_STATS="${COSMOS_NORMALIZATION_STATS:-${ACTIVE_COSMOS_ROOT}/normalization_stats.json}"

SEED="${SEED:-2}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
SCENARIO_NAME="${SCENARIO_NAME:-target_y_positive}"
SOURCE_H5_PATH="${SOURCE_H5_PATH:-}"
SOURCE_KEY="${SOURCE_KEY:-}"
SOURCE_H5_REQUIRE_LIVE_MOTION_GATE="${SOURCE_H5_REQUIRE_LIVE_MOTION_GATE:-true}"
SOURCE_H5_GATE_X_MARGIN="${SOURCE_H5_GATE_X_MARGIN:-0.03}"
SOURCE_H5_GATE_YZ_MARGIN="${SOURCE_H5_GATE_YZ_MARGIN:-0.015}"
SOURCE_H5_PEG_PERTURB_MODE="${SOURCE_H5_PEG_PERTURB_MODE:-block}"
SOURCE_H5_PEG_FORCE_SCALE="${SOURCE_H5_PEG_FORCE_SCALE:-25.0}"
SOURCE_H5_PEG_FORCE_STEPS="${SOURCE_H5_PEG_FORCE_STEPS:-8}"
REQUIRE_SOURCE_H5_PROTOCOL="${REQUIRE_SOURCE_H5_PROTOCOL:-false}"
TARGET_MOTION_STEP="${TARGET_MOTION_STEP:-84}"
TARGET_MOTION_X="${TARGET_MOTION_X:-0.0}"
TARGET_MOTION_Y="${TARGET_MOTION_Y:-0.025}"
TARGET_MOTION_Z="${TARGET_MOTION_Z:-0.0}"
TARGET_MOTION_PER_STEP="${TARGET_MOTION_PER_STEP:-0.00125}"
TARGET_MOTION_DURING_FINISHER="${TARGET_MOTION_DURING_FINISHER:-true}"
REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER="${REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER:-false}"
PREMOTION_COSMOS_STEP="${PREMOTION_COSMOS_STEP:-20}"
PREMOTION_COSMOS_INTERVAL="${PREMOTION_COSMOS_INTERVAL:-16}"
MAX_PREMOTION_COSMOS_PREDICTIONS="${MAX_PREMOTION_COSMOS_PREDICTIONS:-0}"
COSMOS_ACTION_HORIZON="${COSMOS_ACTION_HORIZON:-8}"
MAX_COSMOS_ROUNDS="${MAX_COSMOS_ROUNDS:-4}"
COSMOS_BUILD_TIMEOUT_S="${COSMOS_BUILD_TIMEOUT_S:-180}"
COSMOS_INFERENCE_TIMEOUT_S="${COSMOS_INFERENCE_TIMEOUT_S:-900}"
COSMOS_EXTRACT_TIMEOUT_S="${COSMOS_EXTRACT_TIMEOUT_S:-180}"
COSMOS_ACTION_ROW_OFFSET="${COSMOS_ACTION_ROW_OFFSET:-0}"
COSMOS_ACTION_ROW_OFFSET_SOURCE="${COSMOS_ACTION_ROW_OFFSET_SOURCE:-}"
COSMOS_ACTION_SCALE_X="${COSMOS_ACTION_SCALE_X:-1.0}"
COSMOS_ACTION_SCALE_Y="${COSMOS_ACTION_SCALE_Y:-1.0}"
COSMOS_ACTION_SCALE_Z="${COSMOS_ACTION_SCALE_Z:-1.0}"
COSMOS_ACTION_SCALE_ROT="${COSMOS_ACTION_SCALE_ROT:-1.0}"
COSMOS_ACTION_SCALE_GRIPPER="${COSMOS_ACTION_SCALE_GRIPPER:-1.0}"
COSMOS_ACTION_DIRECTION_GUARD="${COSMOS_ACTION_DIRECTION_GUARD:-none}"
COSMOS_ACTION_DIRECTION_GUARD_MODE="${COSMOS_ACTION_DIRECTION_GUARD_MODE:-clip_opposite}"
ALLOW_UNTRUSTED_MOVE_STOP_SOURCE_MOTION_GUARD="${ALLOW_UNTRUSTED_MOVE_STOP_SOURCE_MOTION_GUARD:-false}"
DYNAMIC_CONTROLLER="${DYNAMIC_CONTROLLER:-cosmos3_policy}"
MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER="${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER:-4}"
NEAR_TARGET_L2="${NEAR_TARGET_L2:-0.08}"
MAX_FINISHER_STEPS="${MAX_FINISHER_STEPS:-80}"
FINISHER_CONTROLLER="${FINISHER_CONTROLLER:-manual_oracle_servo}"
MANUAL_FORWARD_BIAS="${MANUAL_FORWARD_BIAS:-0.0}"
MANUAL_FORWARD_GAIN="${MANUAL_FORWARD_GAIN:-1.6}"
MANUAL_FORWARD_LIMIT="${MANUAL_FORWARD_LIMIT:-0.28}"
MANUAL_LATERAL_GAIN="${MANUAL_LATERAL_GAIN:-1.2}"
MANUAL_LATERAL_LIMIT="${MANUAL_LATERAL_LIMIT:-0.08}"
MANUAL_VERTICAL_GAIN="${MANUAL_VERTICAL_GAIN:-1.2}"
MANUAL_VERTICAL_LIMIT="${MANUAL_VERTICAL_LIMIT:-0.08}"
MANUAL_YAW_ACTION="${MANUAL_YAW_ACTION:-0.22}"
MANUAL_YAW_STOP_L2="${MANUAL_YAW_STOP_L2:--1.0}"
MANUAL_GRIPPER_ACTION="${MANUAL_GRIPPER_ACTION:--1.0}"
MANUAL_ALIGN_THRESHOLD="${MANUAL_ALIGN_THRESHOLD:-0.012}"
MANUAL_YZ_ABORT_THRESHOLD="${MANUAL_YZ_ABORT_THRESHOLD:-0.045}"
MANUAL_SOFT_INSERT_THRESHOLD="${MANUAL_SOFT_INSERT_THRESHOLD:--1.0}"
MANUAL_SOFT_INSERT_SCALE="${MANUAL_SOFT_INSERT_SCALE:-0.35}"
MANUAL_INSERT_SPEED="${MANUAL_INSERT_SPEED:-0.035}"
MANUAL_RETREAT_SPEED="${MANUAL_RETREAT_SPEED:-0.015}"
MANUAL_INSERT_ROLL_ACTION="${MANUAL_INSERT_ROLL_ACTION:-0.0}"
MANUAL_INSERT_PITCH_ACTION="${MANUAL_INSERT_PITCH_ACTION:-0.0}"
MANUAL_INSERT_YAW_ACTION="${MANUAL_INSERT_YAW_ACTION:-0.0}"
MANUAL_POSE_ROT_GAIN="${MANUAL_POSE_ROT_GAIN:-1.0}"
MANUAL_POSE_ROT_LIMIT="${MANUAL_POSE_ROT_LIMIT:-0.25}"
MANUAL_POSE_ROT_YZ_THRESHOLD="${MANUAL_POSE_ROT_YZ_THRESHOLD:-0.025}"
MANUAL_DP_TO_MANUAL_L2="${MANUAL_DP_TO_MANUAL_L2:-0.04}"
SOURCE_H5_TEACHER_DYNAMIC_ACTION_START_OFFSET="${SOURCE_H5_TEACHER_DYNAMIC_ACTION_START_OFFSET:-0}"
PIPELINE_TIMEOUT="${PIPELINE_TIMEOUT:-45m}"
RUN_PREFLIGHT_PY_COMPILE="${RUN_PREFLIGHT_PY_COMPILE:-true}"
RUN_RENDER_CANARY="${RUN_RENDER_CANARY:-true}"
RENDER_CANARY_TIMEOUT="${RENDER_CANARY_TIMEOUT:-3m}"
RENDER_SHADER_PACK="${RENDER_SHADER_PACK:-minimal}"
RENDER_CANARY_API="${RENDER_CANARY_API:-gym}"

if [[ "${COSMOS_ACTION_DIRECTION_GUARD}" == "source_motion_sign" ]]; then
  if [[ "${RUN_GROUP}" == "h5_move_stop" || "${SOURCE_KEY}" == hole_late_move_stop_* ]]; then
    if [[ "${ALLOW_UNTRUSTED_MOVE_STOP_SOURCE_MOTION_GUARD}" != "true" ]]; then
      cat >&2 <<'EOF'
refusing_untrusted_move_stop_source_motion_guard=true
reason=Existing action_diag/try12 evidence shows source_motion_sign worsens the approved move-stop key: executed xyz RMSE is about 0.1264, clip_opposite about 0.1389, and rectify_opposite about 0.1565. Run read-only action diagnostics first or set ALLOW_UNTRUSTED_MOVE_STOP_SOURCE_MOTION_GUARD=true for an explicitly diagnostic-only run.
method_evidence_allowed=false
EOF
      exit 46
    fi
  fi
fi

if [[ ("${RUN_GROUP}" == "h5_continuous_insert" || "${SOURCE_KEY}" == hole_late_continuous_insert_*) && "${ALLOW_CONTINUOUS_INSERT_WHILE_NEXT_COVERAGE_MISSING:-false}" != "true" ]]; then
  gate_out="$(mktemp)"
  trap 'rm -f "${gate_out}"' EXIT
  set +e
  scripts/world_model/check_phase03_oracle_completion.sh >"${gate_out}"
  gate_status="$?"
  set -e
  next_group="$(awk -F= '$1 == "next_required_coverage_group" { print $2 }' "${gate_out}" | tail -1)"
  overall_complete="$(awk -F= '$1 == "phase03_oracle_overall_complete" { print $2 }' "${gate_out}" | tail -1)"
  if [[ "${overall_complete:-unknown}" != "true" && "${next_group:-}" != "continuous_insert" && "${next_group:-}" != "h5_continuous_insert" ]]; then
    cat >&2 <<EOF
refusing_continuous_insert_while_other_coverage_missing=true
run_group=${RUN_GROUP}
source_key=${SOURCE_KEY:-}
completion_gate_exit_status=${gate_status}
phase03_oracle_overall_complete=${overall_complete:-unknown}
next_required_coverage_group=${next_group:-unknown}
reason=Continuous-insert samples are accepted only as single-case references. The next useful Phase 03 Oracle coverage must follow the completion gate, currently forward/backward target motion.
override=ALLOW_CONTINUOUS_INSERT_WHILE_NEXT_COVERAGE_MISSING=true
EOF
    exit 48
  fi
fi

if [[ "${ALLOW_NON_NEXT_PHASE03_COVERAGE:-false}" != "true" ]]; then
  gate_out="$(mktemp)"
  trap 'rm -f "${gate_out}"' EXIT
  set +e
  scripts/world_model/check_phase03_oracle_completion.sh >"${gate_out}"
  gate_status="$?"
  set -e
  next_group="$(awk -F= '$1 == "next_required_coverage_group" { print $2 }' "${gate_out}" | tail -1)"
  overall_complete="$(awk -F= '$1 == "phase03_oracle_overall_complete" { print $2 }' "${gate_out}" | tail -1)"
  allowed=false
  case "${next_group:-}" in
    forward_backward_target_motion)
      [[ "${RUN_GROUP}" == "h5_reverse" || "${RUN_GROUP}" == "h5_move_stop" ]] && allowed=true
      ;;
    left_right_target_motion)
      [[ "${RUN_GROUP}" == "h5_fastshift" ]] && allowed=true
      ;;
    peg_or_wooden_stick_disturbance)
      [[ "${RUN_GROUP}" == "peg_disturb" || "${RUN_GROUP}" == "peg_drop" ]] && allowed=true
      ;;
    multiple_approved_fix3_733_keys|none|"")
      allowed=true
      ;;
    *)
      allowed=true
      ;;
  esac
  if [[ "${overall_complete:-unknown}" != "true" && "${allowed}" != "true" ]]; then
    cat >&2 <<EOF
refusing_non_next_phase03_coverage=true
run_group=${RUN_GROUP}
source_key=${SOURCE_KEY:-}
completion_gate_exit_status=${gate_status}
phase03_oracle_overall_complete=${overall_complete:-unknown}
next_required_coverage_group=${next_group:-unknown}
reason=Phase 03 Oracle must follow the completion gate's next required coverage group instead of launching another case family.
override=ALLOW_NON_NEXT_PHASE03_COVERAGE=true
EOF
    exit 49
  fi
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
export PYTHONPATH="${ROOT}/deps/ManiSkill_clean/examples/baselines/diffusion_policy:${ROOT}/deps/ManiSkill_clean:${ROOT}:${PYTHONPATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export DISPLAY=

{
  echo "timestamp=$(date -Is)"
  echo "phase=${PHASE}"
  echo "run_id=${RUN_ID}"
  echo "run_group=${RUN_GROUP}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_step_id=${SLURM_STEP_ID}"
  echo "node=$(hostname -s)"
  echo "project_python=${PROJECT_PYTHON}"
  echo "cosmos_python=${COSMOS_PYTHON}"
  echo "ckpt_path=${CKPT_PATH}"
  echo "active_cosmos_root=${ACTIVE_COSMOS_ROOT}"
  echo "cosmos_checkpoint_path=${COSMOS_CHECKPOINT_PATH}"
  echo "cosmos_config_file=${COSMOS_CONFIG_FILE}"
  echo "cosmos_normalization_stats=${COSMOS_NORMALIZATION_STATS}"
  if [[ -z "${COSMOS_NORMALIZATION_STATS}" ]]; then
    echo "missing_required_cosmos_normalization_stats=true"
  fi
  echo "seed=${SEED}"
  echo "max_episode_steps=${MAX_EPISODE_STEPS}"
  echo "scenario_name=${SCENARIO_NAME}"
  echo "source_h5_path=${SOURCE_H5_PATH}"
  echo "source_key=${SOURCE_KEY}"
  echo "source_h5_require_live_motion_gate=${SOURCE_H5_REQUIRE_LIVE_MOTION_GATE}"
  echo "source_h5_gate_x_margin=${SOURCE_H5_GATE_X_MARGIN}"
  echo "source_h5_gate_yz_margin=${SOURCE_H5_GATE_YZ_MARGIN}"
  echo "source_h5_peg_perturb_mode=${SOURCE_H5_PEG_PERTURB_MODE}"
  echo "source_h5_peg_force_scale=${SOURCE_H5_PEG_FORCE_SCALE}"
  echo "source_h5_peg_force_steps=${SOURCE_H5_PEG_FORCE_STEPS}"
  echo "require_source_h5_protocol=${REQUIRE_SOURCE_H5_PROTOCOL}"
  echo "target_motion_step=${TARGET_MOTION_STEP}"
  echo "target_motion_x=${TARGET_MOTION_X}"
  echo "target_motion_y=${TARGET_MOTION_Y}"
  echo "target_motion_z=${TARGET_MOTION_Z}"
  echo "target_motion_per_step=${TARGET_MOTION_PER_STEP}"
  echo "target_motion_during_finisher=${TARGET_MOTION_DURING_FINISHER}"
  echo "require_target_motion_complete_before_finisher=${REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER}"
  echo "premotion_cosmos_step=${PREMOTION_COSMOS_STEP}"
  echo "premotion_cosmos_interval=${PREMOTION_COSMOS_INTERVAL}"
  echo "max_premotion_cosmos_predictions=${MAX_PREMOTION_COSMOS_PREDICTIONS}"
  echo "cosmos_action_horizon=${COSMOS_ACTION_HORIZON}"
  echo "max_cosmos_rounds=${MAX_COSMOS_ROUNDS}"
  echo "cosmos_build_timeout_s=${COSMOS_BUILD_TIMEOUT_S}"
  echo "cosmos_inference_timeout_s=${COSMOS_INFERENCE_TIMEOUT_S}"
  echo "cosmos_extract_timeout_s=${COSMOS_EXTRACT_TIMEOUT_S}"
  echo "cosmos_action_row_offset=${COSMOS_ACTION_ROW_OFFSET}"
  echo "cosmos_action_row_offset_source=${COSMOS_ACTION_ROW_OFFSET_SOURCE}"
  if [[ "${COSMOS_ACTION_ROW_OFFSET}" != "0" ]]; then
    echo "audit_allow_diagnostic_action_row_offset=true"
  else
    echo "audit_allow_diagnostic_action_row_offset=false"
  fi
  echo "cosmos_action_scale_x=${COSMOS_ACTION_SCALE_X}"
  echo "cosmos_action_scale_y=${COSMOS_ACTION_SCALE_Y}"
  echo "cosmos_action_scale_z=${COSMOS_ACTION_SCALE_Z}"
  echo "cosmos_action_scale_rot=${COSMOS_ACTION_SCALE_ROT}"
  echo "cosmos_action_scale_gripper=${COSMOS_ACTION_SCALE_GRIPPER}"
  echo "cosmos_action_direction_guard=${COSMOS_ACTION_DIRECTION_GUARD}"
  echo "cosmos_action_direction_guard_mode=${COSMOS_ACTION_DIRECTION_GUARD_MODE}"
  echo "dynamic_controller=${DYNAMIC_CONTROLLER}"
  echo "min_cosmos_dynamic_actions_before_finisher=${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER}"
  echo "near_target_l2=${NEAR_TARGET_L2}"
  echo "max_finisher_steps=${MAX_FINISHER_STEPS}"
  echo "finisher_controller=${FINISHER_CONTROLLER}"
  echo "manual_forward_bias=${MANUAL_FORWARD_BIAS}"
  echo "manual_forward_gain=${MANUAL_FORWARD_GAIN}"
  echo "manual_forward_limit=${MANUAL_FORWARD_LIMIT}"
  echo "manual_lateral_gain=${MANUAL_LATERAL_GAIN}"
  echo "manual_lateral_limit=${MANUAL_LATERAL_LIMIT}"
  echo "manual_vertical_gain=${MANUAL_VERTICAL_GAIN}"
  echo "manual_vertical_limit=${MANUAL_VERTICAL_LIMIT}"
  echo "manual_yaw_action=${MANUAL_YAW_ACTION}"
  echo "manual_yaw_stop_l2=${MANUAL_YAW_STOP_L2}"
  echo "manual_gripper_action=${MANUAL_GRIPPER_ACTION}"
  echo "manual_align_threshold=${MANUAL_ALIGN_THRESHOLD}"
  echo "manual_yz_abort_threshold=${MANUAL_YZ_ABORT_THRESHOLD}"
  echo "manual_soft_insert_threshold=${MANUAL_SOFT_INSERT_THRESHOLD}"
  echo "manual_soft_insert_scale=${MANUAL_SOFT_INSERT_SCALE}"
  echo "manual_insert_speed=${MANUAL_INSERT_SPEED}"
  echo "manual_retreat_speed=${MANUAL_RETREAT_SPEED}"
  echo "manual_insert_roll_action=${MANUAL_INSERT_ROLL_ACTION}"
  echo "manual_insert_pitch_action=${MANUAL_INSERT_PITCH_ACTION}"
  echo "manual_insert_yaw_action=${MANUAL_INSERT_YAW_ACTION}"
  echo "manual_pose_rot_gain=${MANUAL_POSE_ROT_GAIN}"
  echo "manual_pose_rot_limit=${MANUAL_POSE_ROT_LIMIT}"
  echo "manual_pose_rot_yz_threshold=${MANUAL_POSE_ROT_YZ_THRESHOLD}"
  echo "manual_dp_to_manual_l2=${MANUAL_DP_TO_MANUAL_L2}"
  echo "source_h5_teacher_dynamic_action_start_offset=${SOURCE_H5_TEACHER_DYNAMIC_ACTION_START_OFFSET}"
  echo "pipeline_timeout=${PIPELINE_TIMEOUT}"
  echo "run_preflight_py_compile=${RUN_PREFLIGHT_PY_COMPILE}"
  echo "run_render_canary=${RUN_RENDER_CANARY}"
  echo "render_canary_timeout=${RENDER_CANARY_TIMEOUT}"
  echo "render_shader_pack=${RENDER_SHADER_PACK}"
  echo "render_canary_api=${RENDER_CANARY_API}"
  echo "method_evidence_allowed=false"
  echo "boundary=phase03_full_oracle_pipeline_not_complete_until_inserted_and_multi_case_covered"
  echo "forbidden_peg_state_intervention_allowed=false"
} | tee "${RUN_DIR}/manifest.txt" | tee "${LOG_FILE}"

if [[ -z "${COSMOS_NORMALIZATION_STATS}" ]]; then
  {
    echo "phase03_status=blocked_missing_COSMOS_NORMALIZATION_STATS_no_action_execution"
    echo "method_evidence_allowed=false"
    echo "physical_insertion_success=false"
    echo "reason=Set COSMOS_NORMALIZATION_STATS to the active WAM normalization_stats.json before running official Cosmos policy action inference."
  } | tee -a "${LOG_FILE}" | tee "${RUN_DIR}/classification.txt"
  exit 44
fi

nvidia-smi 2>&1 | tee -a "${LOG_FILE}" || true

if [[ "${RUN_PREFLIGHT_PY_COMPILE}" == "true" ]]; then
  set +e
  "${PROJECT_PYTHON}" -m py_compile \
    scripts/training/eval_dp_oracle_full_pipeline.py \
    scripts/world_model/build_cosmos3_live_prefix_wam_input.py \
    scripts/world_model/extract_cosmos3_policy_action_chunk.py \
    scripts/world_model/audit_phase03_oracle_full_pipeline_outputs.py \
    scripts/world_model/render_min_canary.py \
    2>&1 | tee -a "${LOG_FILE}"
  PREFLIGHT_STATUS="${PIPESTATUS[0]}"
  set -e
  if [[ "${PREFLIGHT_STATUS}" -ne 0 ]]; then
    {
      echo "phase03_status=blocked_preflight_py_compile_failed_no_rollout"
      echo "preflight_py_compile_exit_code=${PREFLIGHT_STATUS}"
      echo "method_evidence_allowed=false"
      echo "physical_insertion_success=false"
      echo "reason=Python syntax/preflight failed inside compute allocation before rollout."
    } | tee -a "${LOG_FILE}" | tee "${RUN_DIR}/classification.txt"
    exit "${PREFLIGHT_STATUS}"
  fi
fi

if [[ "${RUN_RENDER_CANARY}" == "true" ]]; then
  {
    echo "phase03_status=render_canary_in_progress_no_rollout"
    echo "method_evidence_allowed=false"
    echo "physical_insertion_success=false"
    echo "reason=Render canary started before Oracle rollout. If this file remains, the canary or wrapper crashed before rollout."
  } | tee -a "${LOG_FILE}" | tee "${RUN_DIR}/classification.txt"
  set +e
  timeout -k 15s "${RENDER_CANARY_TIMEOUT}" "${PROJECT_PYTHON}" -u scripts/world_model/render_min_canary.py \
    --output-dir "${RUN_DIR}/render_canary" \
    --shader-pack "${RENDER_SHADER_PACK}" \
    --render-api "${RENDER_CANARY_API}" \
    2>&1 | tee -a "${LOG_FILE}"
  RENDER_CANARY_STATUS="${PIPESTATUS[0]}"
  set -e
  if [[ "${RENDER_CANARY_STATUS}" -ne 0 ]]; then
    {
      echo "phase03_status=blocked_render_canary_failed_no_rollout"
      echo "render_canary_exit_code=${RENDER_CANARY_STATUS}"
      echo "render_canary_dir=${RUN_DIR}/render_canary"
      echo "method_evidence_allowed=false"
      echo "physical_insertion_success=false"
      echo "reason=Render canary failed inside compute allocation before Oracle rollout; do not treat this as Oracle evidence."
    } | tee -a "${LOG_FILE}" | tee "${RUN_DIR}/classification.txt"
    exit "${RENDER_CANARY_STATUS}"
  fi
  rm -f "${RUN_DIR}/classification.txt"
fi

SOURCE_GATE_ARGS=()
if [[ "${SOURCE_H5_REQUIRE_LIVE_MOTION_GATE}" == "true" ]]; then
  SOURCE_GATE_ARGS+=(--source-h5-require-live-motion-gate)
else
  SOURCE_GATE_ARGS+=(--no-source-h5-require-live-motion-gate)
fi

FINISHER_TARGET_MOTION_ARGS=()
if [[ "${TARGET_MOTION_DURING_FINISHER}" == "true" ]]; then
  FINISHER_TARGET_MOTION_ARGS+=(--target-motion-during-finisher)
else
  FINISHER_TARGET_MOTION_ARGS+=(--no-target-motion-during-finisher)
fi

REQUIRE_TARGET_MOTION_COMPLETE_ARGS=()
if [[ "${REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER}" == "true" ]]; then
  REQUIRE_TARGET_MOTION_COMPLETE_ARGS+=(--require-target-motion-complete-before-finisher)
else
  REQUIRE_TARGET_MOTION_COMPLETE_ARGS+=(--no-require-target-motion-complete-before-finisher)
fi

set +e
timeout "${PIPELINE_TIMEOUT}" "${PROJECT_PYTHON}" -u scripts/training/eval_dp_oracle_full_pipeline.py \
  --ckpt-path "${CKPT_PATH}" \
  --output-dir "${RUN_DIR}" \
  --cosmos-config-file "${COSMOS_CONFIG_FILE}" \
  --cosmos-checkpoint-path "${COSMOS_CHECKPOINT_PATH}" \
  --cosmos-normalization-stats "${COSMOS_NORMALIZATION_STATS}" \
  --cosmos-python "${COSMOS_PYTHON}" \
  --project-python "${PROJECT_PYTHON}" \
  --cosmos-framework-path "${COSMOS_FRAMEWORK_PATH}" \
  --cosmos-tokenizer-dir "${COSMOS_LOCAL_TOKENIZER_DIR}" \
  --wan-vae-path "${WAN_VAE_PATH}" \
  --seed "${SEED}" \
  --max-episode-steps "${MAX_EPISODE_STEPS}" \
  --scenario-name "${SCENARIO_NAME}" \
  --source-h5-path "${SOURCE_H5_PATH}" \
  --source-key "${SOURCE_KEY}" \
  "${SOURCE_GATE_ARGS[@]}" \
  --source-h5-gate-x-margin "${SOURCE_H5_GATE_X_MARGIN}" \
  --source-h5-gate-yz-margin "${SOURCE_H5_GATE_YZ_MARGIN}" \
  --source-h5-peg-perturb-mode "${SOURCE_H5_PEG_PERTURB_MODE}" \
  --source-h5-peg-force-scale "${SOURCE_H5_PEG_FORCE_SCALE}" \
  --source-h5-peg-force-steps "${SOURCE_H5_PEG_FORCE_STEPS}" \
  --target-motion-step "${TARGET_MOTION_STEP}" \
  --target-motion-x "${TARGET_MOTION_X}" \
  --target-motion-y "${TARGET_MOTION_Y}" \
  --target-motion-z "${TARGET_MOTION_Z}" \
  --target-motion-per-step "${TARGET_MOTION_PER_STEP}" \
  "${FINISHER_TARGET_MOTION_ARGS[@]}" \
  "${REQUIRE_TARGET_MOTION_COMPLETE_ARGS[@]}" \
  --premotion-cosmos-step "${PREMOTION_COSMOS_STEP}" \
  --premotion-cosmos-interval "${PREMOTION_COSMOS_INTERVAL}" \
  --max-premotion-cosmos-predictions "${MAX_PREMOTION_COSMOS_PREDICTIONS}" \
  --cosmos-action-horizon "${COSMOS_ACTION_HORIZON}" \
  --max-cosmos-rounds "${MAX_COSMOS_ROUNDS}" \
  --cosmos-build-timeout-s "${COSMOS_BUILD_TIMEOUT_S}" \
  --cosmos-inference-timeout-s "${COSMOS_INFERENCE_TIMEOUT_S}" \
  --cosmos-extract-timeout-s "${COSMOS_EXTRACT_TIMEOUT_S}" \
  --cosmos-action-row-offset "${COSMOS_ACTION_ROW_OFFSET}" \
  --cosmos-action-row-offset-source "${COSMOS_ACTION_ROW_OFFSET_SOURCE}" \
  --cosmos-action-scale-x "${COSMOS_ACTION_SCALE_X}" \
  --cosmos-action-scale-y "${COSMOS_ACTION_SCALE_Y}" \
  --cosmos-action-scale-z "${COSMOS_ACTION_SCALE_Z}" \
  --cosmos-action-scale-rot "${COSMOS_ACTION_SCALE_ROT}" \
  --cosmos-action-scale-gripper "${COSMOS_ACTION_SCALE_GRIPPER}" \
  --cosmos-action-direction-guard "${COSMOS_ACTION_DIRECTION_GUARD}" \
  --cosmos-action-direction-guard-mode "${COSMOS_ACTION_DIRECTION_GUARD_MODE}" \
  --dynamic-controller "${DYNAMIC_CONTROLLER}" \
  --min-cosmos-dynamic-actions-before-finisher "${MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER}" \
  --near-target-l2 "${NEAR_TARGET_L2}" \
  --max-finisher-steps "${MAX_FINISHER_STEPS}" \
  --finisher-controller "${FINISHER_CONTROLLER}" \
  --manual-forward-bias "${MANUAL_FORWARD_BIAS}" \
  --manual-forward-gain "${MANUAL_FORWARD_GAIN}" \
  --manual-forward-limit "${MANUAL_FORWARD_LIMIT}" \
  --manual-lateral-gain "${MANUAL_LATERAL_GAIN}" \
  --manual-lateral-limit "${MANUAL_LATERAL_LIMIT}" \
  --manual-vertical-gain "${MANUAL_VERTICAL_GAIN}" \
  --manual-vertical-limit "${MANUAL_VERTICAL_LIMIT}" \
  --manual-yaw-action "${MANUAL_YAW_ACTION}" \
  --manual-yaw-stop-l2 "${MANUAL_YAW_STOP_L2}" \
  --manual-gripper-action "${MANUAL_GRIPPER_ACTION}" \
  --manual-align-threshold "${MANUAL_ALIGN_THRESHOLD}" \
  --manual-yz-abort-threshold "${MANUAL_YZ_ABORT_THRESHOLD}" \
  --manual-soft-insert-threshold "${MANUAL_SOFT_INSERT_THRESHOLD}" \
  --manual-soft-insert-scale "${MANUAL_SOFT_INSERT_SCALE}" \
  --manual-insert-speed "${MANUAL_INSERT_SPEED}" \
  --manual-retreat-speed "${MANUAL_RETREAT_SPEED}" \
  --manual-insert-roll-action "${MANUAL_INSERT_ROLL_ACTION}" \
  --manual-insert-pitch-action "${MANUAL_INSERT_PITCH_ACTION}" \
  --manual-insert-yaw-action "${MANUAL_INSERT_YAW_ACTION}" \
  --manual-pose-rot-gain "${MANUAL_POSE_ROT_GAIN}" \
  --manual-pose-rot-limit "${MANUAL_POSE_ROT_LIMIT}" \
  --manual-pose-rot-yz-threshold "${MANUAL_POSE_ROT_YZ_THRESHOLD}" \
  --manual-dp-to-manual-l2 "${MANUAL_DP_TO_MANUAL_L2}" \
  --source-h5-teacher-dynamic-action-start-offset "${SOURCE_H5_TEACHER_DYNAMIC_ACTION_START_OFFSET}" \
  --render-shader-pack "${RENDER_SHADER_PACK}" \
  2>&1 | tee -a "${LOG_FILE}"
PIPE_STATUS=("${PIPESTATUS[@]}")
PIPELINE_STATUS="${PIPE_STATUS[0]}"

AUDIT_SOURCE_ARGS=()
if [[ "${REQUIRE_SOURCE_H5_PROTOCOL}" == "true" ]]; then
  AUDIT_SOURCE_ARGS+=(--require-source-h5)
fi
if [[ "${COSMOS_ACTION_ROW_OFFSET}" != "0" ]]; then
  AUDIT_SOURCE_ARGS+=(--allow-diagnostic-action-row-offset)
fi

"${PROJECT_PYTHON}" -u scripts/world_model/audit_phase03_oracle_full_pipeline_outputs.py \
  --run-dir "${RUN_DIR}" \
  --output-json "${RUN_DIR}/artifact_audit.json" \
  "${AUDIT_SOURCE_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
AUDIT_STATUS="${PIPESTATUS[0]}"
set -e

nvidia-smi 2>&1 | tee -a "${LOG_FILE}" || true

{
  echo "pipeline_exit_code=${PIPELINE_STATUS}"
  echo "artifact_audit_exit_code=${AUDIT_STATUS}"
} | tee -a "${LOG_FILE}" | tee -a "${RUN_DIR}/classification.txt"

if [[ "${PIPELINE_STATUS}" -ne 0 ]]; then
  exit "${PIPELINE_STATUS}"
fi
if [[ "${AUDIT_STATUS}" -ne 0 ]]; then
  exit "${AUDIT_STATUS}"
fi
