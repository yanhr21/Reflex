#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
if [[ "${SET_NVIDIA_VK_ICD:-false}" == "true" ]]; then
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
fi
if [[ -n "${DISPLAY:-}" ]]; then
  export DISPLAY
else
  unset DISPLAY
fi

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this live-receding panel only inside a compute-node srun step.
EOF
  exit 30
fi

if [[ "${SLURM_NTASKS:-1}" != "1" || "${SLURM_PROCID:-0}" != "0" ]]; then
  cat >&2 <<'EOF'
refusing_multi_task_execution=true
reason=Run this wrapper as exactly one Slurm task so live receding rollouts are serialized.
EOF
  exit 31
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-2700}"
printf -v CHECKPOINT_NAME 'iter_%09d' "${CHECKPOINT_ITER}"

JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
RUN_DIR="${RUN_DIR:-${SFT_ROOT}/outputs/cosmos3/sft/${JOB_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RUN_DIR}/checkpoints/${CHECKPOINT_NAME}}"
CONFIG_FILE="${CONFIG_FILE:-${RUN_DIR}/config.yaml}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_${CHECKPOINT_NAME}}"
SOURCE_H5_ROOT="${SOURCE_H5_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_rgb_300step_20260612_0245}"
DP_MANIFEST="${DP_MANIFEST:-${ROOT}/experiments/dp_peg1000/run_90201/manifest.json}"
DP_CHECKPOINT="${DP_CHECKPOINT:-${ROOT}/experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt}"
CONTINUABILITY_STATS_JSON="${CONTINUABILITY_STATS_JSON:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/dp_static_continuability_stats_20260613/dp_static_continuability_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SFT_ROOT}/live_receding_full300_panel_${CHECKPOINT_NAME}_samples${SAMPLE_INDICES:-0_1_3_8}_${STAMP}}"

SAMPLE_INDICES="${SAMPLE_INDICES:-0,1,3,4}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
PREFIX_ROLE_MODE="${PREFIX_ROLE_MODE:-auto}"
PREFIX_START_MODE="${PREFIX_START_MODE:-target_motion_onset}"
PRETRIGGER_CONTROL_MODE="${PRETRIGGER_CONTROL_MODE:-frozen_dp_until_target_motion}"
MIN_DYNAMIC_PREFIX_FRAME="${MIN_DYNAMIC_PREFIX_FRAME:-8}"
TARGET_MOTION_CONSECUTIVE_FRAMES="${TARGET_MOTION_CONSECUTIVE_FRAMES:-2}"
MAX_RECEDING_ITERATIONS="${MAX_RECEDING_ITERATIONS:-40}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-8}"
DP_HANDOFF_HORIZON="${DP_HANDOFF_HORIZON:-32}"
DP_HANDOFF_CHUNK_HORIZON="${DP_HANDOFF_CHUNK_HORIZON:-0}"
DP_ACTION_SEED_BASE="${DP_ACTION_SEED_BASE:--1}"
SCRIPTED_INSERTION_MODE="${SCRIPTED_INSERTION_MODE:-disabled}"
SCRIPTED_INSERTION_ACTION="${SCRIPTED_INSERTION_ACTION:-0.004,0,0,0,0,0,-1}"
SCRIPTED_INSERTION_ACTION_FRAME="${SCRIPTED_INSERTION_ACTION_FRAME:-fixed}"
SCRIPTED_INSERTION_STEP_REL_X="${SCRIPTED_INSERTION_STEP_REL_X:-0.32}"
SCRIPTED_INSERTION_LATERAL_GAIN="${SCRIPTED_INSERTION_LATERAL_GAIN:-10.0}"
SCRIPTED_INSERTION_MAX_LATERAL_STEP="${SCRIPTED_INSERTION_MAX_LATERAL_STEP:-0.12}"
SCRIPTED_INSERTION_ACTION_TARGET_SOURCE_FRAME="${SCRIPTED_INSERTION_ACTION_TARGET_SOURCE_FRAME:--1}"
SCRIPTED_INSERTION_TEACHER_TABLE_JSONL="${SCRIPTED_INSERTION_TEACHER_TABLE_JSONL:-}"
SCRIPTED_INSERTION_TEACHER_TABLE_K="${SCRIPTED_INSERTION_TEACHER_TABLE_K:-1}"
SCRIPTED_INSERTION_TEACHER_TABLE_MIN_OFFSET="${SCRIPTED_INSERTION_TEACHER_TABLE_MIN_OFFSET:-1}"
SCRIPTED_INSERTION_TEACHER_TABLE_MAX_OFFSET="${SCRIPTED_INSERTION_TEACHER_TABLE_MAX_OFFSET:-48}"
SCRIPTED_INSERTION_TEACHER_TABLE_QUERY_WEIGHTS="${SCRIPTED_INSERTION_TEACHER_TABLE_QUERY_WEIGHTS:-1,2,4}"
SCRIPTED_INSERTION_TEACHER_TABLE_MAX_DISTANCE="${SCRIPTED_INSERTION_TEACHER_TABLE_MAX_DISTANCE:--1.0}"
SCRIPTED_INSERTION_TEACHER_TABLE_MIN_ACTION_X="${SCRIPTED_INSERTION_TEACHER_TABLE_MIN_ACTION_X:--1000000000.0}"
SCRIPTED_INSERTION_TEACHER_TABLE_MAX_ACTION_X="${SCRIPTED_INSERTION_TEACHER_TABLE_MAX_ACTION_X:-1000000000.0}"
SCRIPTED_INSERTION_TEACHER_TABLE_ACTION_X_FLOOR="${SCRIPTED_INSERTION_TEACHER_TABLE_ACTION_X_FLOOR:-nan}"
SCRIPTED_INSERTION_TEACHER_TABLE_SCENARIO_MATCH="${SCRIPTED_INSERTION_TEACHER_TABLE_SCENARIO_MATCH:-false}"
SCRIPTED_INSERTION_ACTION_TARGET_GAIN="${SCRIPTED_INSERTION_ACTION_TARGET_GAIN:-10.0}"
SCRIPTED_INSERTION_ACTION_TARGET_ROT_GAIN="${SCRIPTED_INSERTION_ACTION_TARGET_ROT_GAIN:-3.0}"
SCRIPTED_INSERTION_ACTION_TARGET_ROT_CAP="${SCRIPTED_INSERTION_ACTION_TARGET_ROT_CAP:-0.35}"
SCRIPTED_INSERTION_ACTION_TARGET_ROT_SOURCE="${SCRIPTED_INSERTION_ACTION_TARGET_ROT_SOURCE:-peg}"
SCRIPTED_INSERTION_CONTACT_SEAT_TARGET_L2="${SCRIPTED_INSERTION_CONTACT_SEAT_TARGET_L2:--1.0}"
SCRIPTED_INSERTION_CONTACT_SEAT_LATCH="${SCRIPTED_INSERTION_CONTACT_SEAT_LATCH:-false}"
SCRIPTED_INSERTION_CONTACT_SEAT_STEP_REL_X="${SCRIPTED_INSERTION_CONTACT_SEAT_STEP_REL_X:-0.06}"
SCRIPTED_INSERTION_CONTACT_SEAT_X_SERVO="${SCRIPTED_INSERTION_CONTACT_SEAT_X_SERVO:-false}"
SCRIPTED_INSERTION_CONTACT_SEAT_TARGET_REL_X="${SCRIPTED_INSERTION_CONTACT_SEAT_TARGET_REL_X:-0.0}"
SCRIPTED_INSERTION_CONTACT_SEAT_X_SERVO_GAIN="${SCRIPTED_INSERTION_CONTACT_SEAT_X_SERVO_GAIN:-8.0}"
SCRIPTED_INSERTION_CONTACT_SEAT_X_SERVO_DIRECTION="${SCRIPTED_INSERTION_CONTACT_SEAT_X_SERVO_DIRECTION:-1.0}"
SCRIPTED_INSERTION_CONTACT_SEAT_MIN_STEP_REL_X="${SCRIPTED_INSERTION_CONTACT_SEAT_MIN_STEP_REL_X:-0.0}"
SCRIPTED_INSERTION_CONTACT_SEAT_MAX_STEP_REL_X="${SCRIPTED_INSERTION_CONTACT_SEAT_MAX_STEP_REL_X:--1.0}"
SCRIPTED_INSERTION_CONTACT_SEAT_STOP_ON_X_REGRESSION="${SCRIPTED_INSERTION_CONTACT_SEAT_STOP_ON_X_REGRESSION:-false}"
SCRIPTED_INSERTION_CONTACT_SEAT_X_REGRESSION_TOLERANCE="${SCRIPTED_INSERTION_CONTACT_SEAT_X_REGRESSION_TOLERANCE:-0.0005}"
SCRIPTED_INSERTION_CONTACT_SEAT_X_REGRESSION_POLICY="${SCRIPTED_INSERTION_CONTACT_SEAT_X_REGRESSION_POLICY:-stop}"
SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_POLICY="${SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_POLICY:-disabled}"
SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_MIN_X_IMPROVEMENT="${SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_MIN_X_IMPROVEMENT:-0.0001}"
SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_MAX_LATERAL_REGRESSION="${SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_MAX_LATERAL_REGRESSION:-0.0002}"
SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_REQUIRE_GRASP="${SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_REQUIRE_GRASP:-true}"
SCRIPTED_INSERTION_CONTACT_SEAT_LATERAL_GAIN="${SCRIPTED_INSERTION_CONTACT_SEAT_LATERAL_GAIN:-0.0}"
SCRIPTED_INSERTION_CONTACT_SEAT_MAX_LATERAL_STEP="${SCRIPTED_INSERTION_CONTACT_SEAT_MAX_LATERAL_STEP:-0.02}"
SCRIPTED_INSERTION_CONTACT_SEAT_MIN_SERVO_STEPS="${SCRIPTED_INSERTION_CONTACT_SEAT_MIN_SERVO_STEPS:--1}"
SCRIPTED_INSERTION_CONTACT_SEAT_MAX_SERVO_L2="${SCRIPTED_INSERTION_CONTACT_SEAT_MAX_SERVO_L2:--1.0}"
SCRIPTED_INSERTION_CONTACT_SEAT_MAX_SERVO_STEPS="${SCRIPTED_INSERTION_CONTACT_SEAT_MAX_SERVO_STEPS:--1}"
SCRIPTED_INSERTION_CONTACT_SEAT_PLATEAU_WINDOW="${SCRIPTED_INSERTION_CONTACT_SEAT_PLATEAU_WINDOW:-0}"
SCRIPTED_INSERTION_CONTACT_SEAT_PLATEAU_MAX_L2="${SCRIPTED_INSERTION_CONTACT_SEAT_PLATEAU_MAX_L2:--1.0}"
SCRIPTED_INSERTION_CONTACT_SEAT_IMPROVEMENT_EPSILON="${SCRIPTED_INSERTION_CONTACT_SEAT_IMPROVEMENT_EPSILON:-0.0005}"
SCRIPTED_INSERTION_ANCHOR_STOP_ON_TARGET_REGRESSION="${SCRIPTED_INSERTION_ANCHOR_STOP_ON_TARGET_REGRESSION:-false}"
SCRIPTED_INSERTION_ANCHOR_REGRESSION_TOLERANCE="${SCRIPTED_INSERTION_ANCHOR_REGRESSION_TOLERANCE:-0.0005}"
SCRIPTED_INSERTION_ADAPTIVE_FLIP_AXES="${SCRIPTED_INSERTION_ADAPTIVE_FLIP_AXES:-}"
SCRIPTED_INSERTION_LATERAL_SIGN="${SCRIPTED_INSERTION_LATERAL_SIGN:--1.0}"
SCRIPTED_INSERTION_Z_SIGN="${SCRIPTED_INSERTION_Z_SIGN:--1.0}"
SCRIPTED_INSERTION_AXIS_ONLY_AFTER_STEP="${SCRIPTED_INSERTION_AXIS_ONLY_AFTER_STEP:--1}"
SCRIPTED_INSERTION_IGNORE_GRASP_AFTER_TRIGGER="${SCRIPTED_INSERTION_IGNORE_GRASP_AFTER_TRIGGER:-false}"
SCRIPTED_INSERTION_MAX_STEPS="${SCRIPTED_INSERTION_MAX_STEPS:-8}"
SCRIPTED_INSERTION_MIN_REL_X="${SCRIPTED_INSERTION_MIN_REL_X:--0.04}"
SCRIPTED_INSERTION_MAX_REL_X="${SCRIPTED_INSERTION_MAX_REL_X:-0.03}"
SCRIPTED_INSERTION_MAX_ABS_Y="${SCRIPTED_INSERTION_MAX_ABS_Y:-0.018}"
SCRIPTED_INSERTION_MAX_ABS_Z="${SCRIPTED_INSERTION_MAX_ABS_Z:-0.018}"
SCRIPTED_INSERTION_MAX_HOLE_SPEED="${SCRIPTED_INSERTION_MAX_HOLE_SPEED:-0.01}"
SCRIPTED_INSERTION_TERMINAL_POSE_GATE="${SCRIPTED_INSERTION_TERMINAL_POSE_GATE:-disabled}"
SCRIPTED_INSERTION_TERMINAL_ANCHOR_SOURCE_FRAME="${SCRIPTED_INSERTION_TERMINAL_ANCHOR_SOURCE_FRAME:-165}"
SCRIPTED_INSERTION_TERMINAL_MAX_REL_L2="${SCRIPTED_INSERTION_TERMINAL_MAX_REL_L2:-0.01}"
SCRIPTED_INSERTION_TERMINAL_MAX_PEG_HOLE_POS_L2="${SCRIPTED_INSERTION_TERMINAL_MAX_PEG_HOLE_POS_L2:-0.02}"
SCRIPTED_INSERTION_TERMINAL_MAX_PEG_HOLE_ROT_RAD="${SCRIPTED_INSERTION_TERMINAL_MAX_PEG_HOLE_ROT_RAD:-0.20}"
SCRIPTED_INSERTION_TERMINAL_MAX_TCP_HOLE_POS_L2="${SCRIPTED_INSERTION_TERMINAL_MAX_TCP_HOLE_POS_L2:-0.08}"
SCRIPTED_INSERTION_TERMINAL_MAX_HOLE_WORLD_POS_L2="${SCRIPTED_INSERTION_TERMINAL_MAX_HOLE_WORLD_POS_L2:-0.05}"
SCRIPTED_INSERTION_TERMINAL_MAX_HOLE_WORLD_ROT_RAD="${SCRIPTED_INSERTION_TERMINAL_MAX_HOLE_WORLD_ROT_RAD:-0.05}"
SCRIPTED_INSERTION_REQUIRE_GRASP="${SCRIPTED_INSERTION_REQUIRE_GRASP:-true}"
SCRIPTED_INSERTION_STOP_ON_GATE_FAIL="${SCRIPTED_INSERTION_STOP_ON_GATE_FAIL:-true}"
EXTERNAL_TARGET_MODE="${EXTERNAL_TARGET_MODE:-source_env_state}"
RUN_COSMOS_INFERENCE="${RUN_COSMOS_INFERENCE:-true}"
RENDER_LIVE_VIDEO="${RENDER_LIVE_VIDEO:-true}"
CONTROLLER_ACTION_SOURCE="${CONTROLLER_ACTION_SOURCE:-cosmos_robot_action}"
EXECUTOR_CHECKPOINT="${EXECUTOR_CHECKPOINT:-}"
CANDIDATE_OUTCOME_SCORER_CHECKPOINT="${CANDIDATE_OUTCOME_SCORER_CHECKPOINT:-}"
CANDIDATE_OUTCOME_SCORER_SUMMARY="${CANDIDATE_OUTCOME_SCORER_SUMMARY:-}"
CANDIDATE_OUTCOME_SCORER_DP_MARGIN="${CANDIDATE_OUTCOME_SCORER_DP_MARGIN:-0.0}"
CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA="${CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA:--1000000000.0}"
CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB="${CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB:-0.0}"
CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB="${CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB:-0.0}"
CANDIDATE_OUTCOME_SCORER_MIN_PRED_STATE_X="${CANDIDATE_OUTCOME_SCORER_MIN_PRED_STATE_X:--1000000000.0}"
CANDIDATE_OUTCOME_SCORER_MAX_PRED_STATE_X="${CANDIDATE_OUTCOME_SCORER_MAX_PRED_STATE_X:-1000000000.0}"
CANDIDATE_OUTCOME_SCORER_MAX_PRED_STATE_ABS_Y="${CANDIDATE_OUTCOME_SCORER_MAX_PRED_STATE_ABS_Y:--1.0}"
CANDIDATE_OUTCOME_SCORER_MAX_PRED_STATE_ABS_Z="${CANDIDATE_OUTCOME_SCORER_MAX_PRED_STATE_ABS_Z:--1.0}"
CANDIDATE_OUTCOME_SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS="${CANDIDATE_OUTCOME_SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS:-}"
CANDIDATE_OUTCOME_SCORER_SCORE_STATE_TARGET="${CANDIDATE_OUTCOME_SCORER_SCORE_STATE_TARGET:-}"
CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS="${CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS:-}"
CANDIDATE_EXECUTOR_ALLOW_DP_PRIOR_BEFORE_GATE="${CANDIDATE_EXECUTOR_ALLOW_DP_PRIOR_BEFORE_GATE:-false}"
SOURCE_INSERTION_SUFFIX_BANK="${SOURCE_INSERTION_SUFFIX_BANK:-}"
SOURCE_SUFFIX_K="${SOURCE_SUFFIX_K:-0}"
SOURCE_SUFFIX_BLENDS="${SOURCE_SUFFIX_BLENDS:-1.0}"
SOURCE_SUFFIX_EXECUTE_STEPS="${SOURCE_SUFFIX_EXECUTE_STEPS:-32}"
SOURCE_SUFFIX_OFFSETS="${SOURCE_SUFFIX_OFFSETS:-}"
SOURCE_SUFFIX_SCENARIO_MATCH="${SOURCE_SUFFIX_SCENARIO_MATCH:-false}"
SOURCE_SUFFIX_QUERY_X_WEIGHT="${SOURCE_SUFFIX_QUERY_X_WEIGHT:-1.0}"
SOURCE_SUFFIX_QUERY_Y_WEIGHT="${SOURCE_SUFFIX_QUERY_Y_WEIGHT:-2.0}"
SOURCE_SUFFIX_QUERY_Z_WEIGHT="${SOURCE_SUFFIX_QUERY_Z_WEIGHT:-4.0}"
SOURCE_SUFFIX_MAX_DISTANCE="${SOURCE_SUFFIX_MAX_DISTANCE:--1.0}"
SOURCE_SUFFIX_IGNORE_RESIDUAL_CAP="${SOURCE_SUFFIX_IGNORE_RESIDUAL_CAP:-false}"
EXECUTOR_RESIDUAL_SCALE="${EXECUTOR_RESIDUAL_SCALE:-1.0}"
CLIP_LIVE_ACTIONS="${CLIP_LIVE_ACTIONS:-true}"
SAVE_LIVE_STATE_SNAPSHOTS="${SAVE_LIVE_STATE_SNAPSHOTS:-false}"
SAVE_CANDIDATE_ACTION_BANK="${SAVE_CANDIDATE_ACTION_BANK:-false}"
LIVE_PROGRESS_INTERVAL="${LIVE_PROGRESS_INTERVAL:-0}"
PRETRIGGER_DEBUG_STEPS="${PRETRIGGER_DEBUG_STEPS:-3}"
VIDEO_FPS="${VIDEO_FPS:-30}"
ORACLE_FINAL_SEAT_MODE="${ORACLE_FINAL_SEAT_MODE:-disabled}"
ORACLE_FINAL_SEAT_SOURCE_FRAME="${ORACLE_FINAL_SEAT_SOURCE_FRAME:-300}"
LIVE_GEOMETRIC_FINAL_SEAT_MODE="${LIVE_GEOMETRIC_FINAL_SEAT_MODE:-disabled}"
LIVE_GEOMETRIC_FINAL_SEAT_TARGET_REL="${LIVE_GEOMETRIC_FINAL_SEAT_TARGET_REL:--0.006,0,0}"
ALLOW_LIVE_RECEDING_DIAGNOSTIC="${ALLOW_LIVE_RECEDING_DIAGNOSTIC:-false}"

diagnostic_reasons=()
if [[ "${PREFIX_ROLE_MODE}" != "auto" ]]; then
  diagnostic_reasons+=("prefix_role_mode_not_auto:${PREFIX_ROLE_MODE}")
fi
if [[ "${PREFIX_START_MODE}" != "target_motion_onset" ]]; then
  diagnostic_reasons+=("prefix_start_mode_not_causal_target_motion_onset:${PREFIX_START_MODE}")
fi
if [[ "${PRETRIGGER_CONTROL_MODE}" != "frozen_dp_until_target_motion" ]]; then
  diagnostic_reasons+=("pretrigger_control_mode_not_live_frozen_dp:${PRETRIGGER_CONTROL_MODE}")
fi
if [[ "${RUN_COSMOS_INFERENCE}" != "true" ]]; then
  diagnostic_reasons+=("cosmos_inference_disabled")
fi
case "${CONTROLLER_ACTION_SOURCE}" in
  residual_executor|contact_executor|candidate_executor)
    if [[ ! -s "${EXECUTOR_CHECKPOINT}" ]]; then
      diagnostic_reasons+=("missing_${CONTROLLER_ACTION_SOURCE}_checkpoint")
    fi
    ;;
esac
if [[ -n "${CANDIDATE_OUTCOME_SCORER_CHECKPOINT}" && ! -s "${CANDIDATE_OUTCOME_SCORER_CHECKPOINT}" ]]; then
  diagnostic_reasons+=("missing_candidate_outcome_scorer_checkpoint")
fi
if [[ -n "${CANDIDATE_OUTCOME_SCORER_CHECKPOINT}" && "${CONTROLLER_ACTION_SOURCE}" != "candidate_executor" ]]; then
  diagnostic_reasons+=("candidate_outcome_scorer_requires_candidate_executor")
fi
if [[ -n "${CANDIDATE_OUTCOME_SCORER_CHECKPOINT}" && -z "${CANDIDATE_OUTCOME_SCORER_SUMMARY}" ]]; then
  CANDIDATE_OUTCOME_SCORER_SUMMARY="$(dirname "${CANDIDATE_OUTCOME_SCORER_CHECKPOINT}")/training_summary.json"
fi
if [[ -n "${CANDIDATE_OUTCOME_SCORER_CHECKPOINT}" ]]; then
  if [[ ! -s "${CANDIDATE_OUTCOME_SCORER_SUMMARY}" ]]; then
    diagnostic_reasons+=("missing_candidate_outcome_scorer_summary")
  elif ! "${ROOT}/.venv/bin/python" - "${CANDIDATE_OUTCOME_SCORER_SUMMARY}" "${CANDIDATE_OUTCOME_SCORER_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
checkpoint = Path(sys.argv[2]).resolve()
expected_raw = summary.get("formal_live_eval_checkpoint")
expected = Path(str(expected_raw)).resolve() if expected_raw else None
if summary.get("ready_for_formal_live_eval") is not True:
    raise SystemExit(1)
if expected is None or expected != checkpoint:
    raise SystemExit(1)
PY
  then
    diagnostic_reasons+=("candidate_outcome_scorer_not_formal_live_ready")
  fi
fi
if (( ${#diagnostic_reasons[@]} > 0 )) && [[ "${ALLOW_LIVE_RECEDING_DIAGNOSTIC}" != "true" ]]; then
  {
    echo "refusing_non_method_live_receding_panel=true"
    echo "reason=${diagnostic_reasons[*]}"
    echo "override_for_non_method_diagnostic=ALLOW_LIVE_RECEDING_DIAGNOSTIC=true"
    echo "required_method_modes=PREFIX_ROLE_MODE=auto PREFIX_START_MODE=target_motion_onset PRETRIGGER_CONTROL_MODE=frozen_dp_until_target_motion RUN_COSMOS_INFERENCE=true"
  } >&2
  exit 42
fi

cd "${ROOT}"
mkdir -p "${OUTPUT_ROOT}"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-}"
  echo "sft_root=${SFT_ROOT}"
  echo "checkpoint_iter=${CHECKPOINT_ITER}"
  echo "checkpoint_path=${CHECKPOINT_PATH}"
  echo "config_file=${CONFIG_FILE}"
  echo "eval_root=${EVAL_ROOT}"
  echo "source_h5_root=${SOURCE_H5_ROOT}"
  echo "condition_root=${CONDITION_ROOT}"
  echo "dp_manifest=${DP_MANIFEST}"
  echo "dp_checkpoint=${DP_CHECKPOINT}"
  echo "continuability_stats_json=${CONTINUABILITY_STATS_JSON}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "sample_indices=${SAMPLE_INDICES}"
  echo "prefix_role_mode=${PREFIX_ROLE_MODE}"
  echo "prefix_start_mode=${PREFIX_START_MODE}"
  echo "pretrigger_control_mode=${PRETRIGGER_CONTROL_MODE}"
  echo "min_dynamic_prefix_frame=${MIN_DYNAMIC_PREFIX_FRAME}"
  echo "target_motion_consecutive_frames=${TARGET_MOTION_CONSECUTIVE_FRAMES}"
  echo "max_receding_iterations=${MAX_RECEDING_ITERATIONS}"
  echo "action_exec_horizon=${ACTION_EXEC_HORIZON}"
  echo "dp_handoff_horizon=${DP_HANDOFF_HORIZON}"
  echo "dp_handoff_chunk_horizon=${DP_HANDOFF_CHUNK_HORIZON}"
  echo "dp_action_seed_base=${DP_ACTION_SEED_BASE}"
  echo "scripted_insertion_mode=${SCRIPTED_INSERTION_MODE}"
  echo "oracle_final_seat_mode=${ORACLE_FINAL_SEAT_MODE}"
  echo "oracle_final_seat_source_frame=${ORACLE_FINAL_SEAT_SOURCE_FRAME}"
  echo "live_geometric_final_seat_mode=${LIVE_GEOMETRIC_FINAL_SEAT_MODE}"
  echo "live_geometric_final_seat_target_rel=${LIVE_GEOMETRIC_FINAL_SEAT_TARGET_REL}"
  echo "scripted_insertion_action=${SCRIPTED_INSERTION_ACTION}"
  echo "scripted_insertion_action_frame=${SCRIPTED_INSERTION_ACTION_FRAME}"
  echo "scripted_insertion_step_rel_x=${SCRIPTED_INSERTION_STEP_REL_X}"
  echo "scripted_insertion_lateral_gain=${SCRIPTED_INSERTION_LATERAL_GAIN}"
  echo "scripted_insertion_max_lateral_step=${SCRIPTED_INSERTION_MAX_LATERAL_STEP}"
  echo "scripted_insertion_action_target_source_frame=${SCRIPTED_INSERTION_ACTION_TARGET_SOURCE_FRAME}"
  echo "scripted_insertion_teacher_table_jsonl=${SCRIPTED_INSERTION_TEACHER_TABLE_JSONL:-none}"
  echo "scripted_insertion_teacher_table_k=${SCRIPTED_INSERTION_TEACHER_TABLE_K}"
  echo "scripted_insertion_teacher_table_min_offset=${SCRIPTED_INSERTION_TEACHER_TABLE_MIN_OFFSET}"
  echo "scripted_insertion_teacher_table_max_offset=${SCRIPTED_INSERTION_TEACHER_TABLE_MAX_OFFSET}"
  echo "scripted_insertion_teacher_table_query_weights=${SCRIPTED_INSERTION_TEACHER_TABLE_QUERY_WEIGHTS}"
  echo "scripted_insertion_teacher_table_max_distance=${SCRIPTED_INSERTION_TEACHER_TABLE_MAX_DISTANCE}"
  echo "scripted_insertion_teacher_table_min_action_x=${SCRIPTED_INSERTION_TEACHER_TABLE_MIN_ACTION_X}"
  echo "scripted_insertion_teacher_table_max_action_x=${SCRIPTED_INSERTION_TEACHER_TABLE_MAX_ACTION_X}"
  echo "scripted_insertion_teacher_table_action_x_floor=${SCRIPTED_INSERTION_TEACHER_TABLE_ACTION_X_FLOOR}"
  echo "scripted_insertion_teacher_table_scenario_match=${SCRIPTED_INSERTION_TEACHER_TABLE_SCENARIO_MATCH}"
  echo "scripted_insertion_action_target_gain=${SCRIPTED_INSERTION_ACTION_TARGET_GAIN}"
  echo "scripted_insertion_action_target_rot_gain=${SCRIPTED_INSERTION_ACTION_TARGET_ROT_GAIN}"
  echo "scripted_insertion_action_target_rot_cap=${SCRIPTED_INSERTION_ACTION_TARGET_ROT_CAP}"
  echo "scripted_insertion_action_target_rot_source=${SCRIPTED_INSERTION_ACTION_TARGET_ROT_SOURCE}"
  echo "scripted_insertion_contact_seat_target_l2=${SCRIPTED_INSERTION_CONTACT_SEAT_TARGET_L2}"
  echo "scripted_insertion_contact_seat_latch=${SCRIPTED_INSERTION_CONTACT_SEAT_LATCH}"
  echo "scripted_insertion_contact_seat_step_rel_x=${SCRIPTED_INSERTION_CONTACT_SEAT_STEP_REL_X}"
  echo "scripted_insertion_contact_seat_x_servo=${SCRIPTED_INSERTION_CONTACT_SEAT_X_SERVO}"
  echo "scripted_insertion_contact_seat_target_rel_x=${SCRIPTED_INSERTION_CONTACT_SEAT_TARGET_REL_X}"
  echo "scripted_insertion_contact_seat_x_servo_gain=${SCRIPTED_INSERTION_CONTACT_SEAT_X_SERVO_GAIN}"
  echo "scripted_insertion_contact_seat_x_servo_direction=${SCRIPTED_INSERTION_CONTACT_SEAT_X_SERVO_DIRECTION}"
  echo "scripted_insertion_contact_seat_min_step_rel_x=${SCRIPTED_INSERTION_CONTACT_SEAT_MIN_STEP_REL_X}"
  echo "scripted_insertion_contact_seat_max_step_rel_x=${SCRIPTED_INSERTION_CONTACT_SEAT_MAX_STEP_REL_X}"
  echo "scripted_insertion_contact_seat_stop_on_x_regression=${SCRIPTED_INSERTION_CONTACT_SEAT_STOP_ON_X_REGRESSION}"
  echo "scripted_insertion_contact_seat_x_regression_tolerance=${SCRIPTED_INSERTION_CONTACT_SEAT_X_REGRESSION_TOLERANCE}"
  echo "scripted_insertion_contact_seat_x_regression_policy=${SCRIPTED_INSERTION_CONTACT_SEAT_X_REGRESSION_POLICY}"
  echo "scripted_insertion_contact_seat_probe_policy=${SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_POLICY}"
  echo "scripted_insertion_contact_seat_probe_min_x_improvement=${SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_MIN_X_IMPROVEMENT}"
  echo "scripted_insertion_contact_seat_probe_max_lateral_regression=${SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_MAX_LATERAL_REGRESSION}"
  echo "scripted_insertion_contact_seat_probe_require_grasp=${SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_REQUIRE_GRASP}"
  echo "scripted_insertion_contact_seat_lateral_gain=${SCRIPTED_INSERTION_CONTACT_SEAT_LATERAL_GAIN}"
  echo "scripted_insertion_contact_seat_max_lateral_step=${SCRIPTED_INSERTION_CONTACT_SEAT_MAX_LATERAL_STEP}"
  echo "scripted_insertion_contact_seat_min_servo_steps=${SCRIPTED_INSERTION_CONTACT_SEAT_MIN_SERVO_STEPS}"
  echo "scripted_insertion_contact_seat_max_servo_l2=${SCRIPTED_INSERTION_CONTACT_SEAT_MAX_SERVO_L2}"
  echo "scripted_insertion_contact_seat_max_servo_steps=${SCRIPTED_INSERTION_CONTACT_SEAT_MAX_SERVO_STEPS}"
  echo "scripted_insertion_contact_seat_plateau_window=${SCRIPTED_INSERTION_CONTACT_SEAT_PLATEAU_WINDOW}"
  echo "scripted_insertion_contact_seat_plateau_max_l2=${SCRIPTED_INSERTION_CONTACT_SEAT_PLATEAU_MAX_L2}"
  echo "scripted_insertion_contact_seat_improvement_epsilon=${SCRIPTED_INSERTION_CONTACT_SEAT_IMPROVEMENT_EPSILON}"
  echo "scripted_insertion_anchor_stop_on_target_regression=${SCRIPTED_INSERTION_ANCHOR_STOP_ON_TARGET_REGRESSION}"
  echo "scripted_insertion_anchor_regression_tolerance=${SCRIPTED_INSERTION_ANCHOR_REGRESSION_TOLERANCE}"
  echo "scripted_insertion_lateral_sign=${SCRIPTED_INSERTION_LATERAL_SIGN}"
  echo "scripted_insertion_z_sign=${SCRIPTED_INSERTION_Z_SIGN}"
  echo "scripted_insertion_axis_only_after_step=${SCRIPTED_INSERTION_AXIS_ONLY_AFTER_STEP}"
  echo "scripted_insertion_ignore_grasp_after_trigger=${SCRIPTED_INSERTION_IGNORE_GRASP_AFTER_TRIGGER}"
  echo "scripted_insertion_max_steps=${SCRIPTED_INSERTION_MAX_STEPS}"
  echo "scripted_insertion_min_rel_x=${SCRIPTED_INSERTION_MIN_REL_X}"
  echo "scripted_insertion_max_rel_x=${SCRIPTED_INSERTION_MAX_REL_X}"
  echo "scripted_insertion_max_abs_y=${SCRIPTED_INSERTION_MAX_ABS_Y}"
  echo "scripted_insertion_max_abs_z=${SCRIPTED_INSERTION_MAX_ABS_Z}"
  echo "scripted_insertion_max_hole_speed=${SCRIPTED_INSERTION_MAX_HOLE_SPEED}"
  echo "scripted_insertion_terminal_pose_gate=${SCRIPTED_INSERTION_TERMINAL_POSE_GATE}"
  echo "scripted_insertion_terminal_anchor_source_frame=${SCRIPTED_INSERTION_TERMINAL_ANCHOR_SOURCE_FRAME}"
  echo "scripted_insertion_terminal_max_rel_l2=${SCRIPTED_INSERTION_TERMINAL_MAX_REL_L2}"
  echo "scripted_insertion_terminal_max_peg_hole_pos_l2=${SCRIPTED_INSERTION_TERMINAL_MAX_PEG_HOLE_POS_L2}"
  echo "scripted_insertion_terminal_max_peg_hole_rot_rad=${SCRIPTED_INSERTION_TERMINAL_MAX_PEG_HOLE_ROT_RAD}"
  echo "scripted_insertion_terminal_max_tcp_hole_pos_l2=${SCRIPTED_INSERTION_TERMINAL_MAX_TCP_HOLE_POS_L2}"
  echo "scripted_insertion_terminal_max_hole_world_pos_l2=${SCRIPTED_INSERTION_TERMINAL_MAX_HOLE_WORLD_POS_L2}"
  echo "scripted_insertion_terminal_max_hole_world_rot_rad=${SCRIPTED_INSERTION_TERMINAL_MAX_HOLE_WORLD_ROT_RAD}"
  echo "scripted_insertion_require_grasp=${SCRIPTED_INSERTION_REQUIRE_GRASP}"
  echo "scripted_insertion_stop_on_gate_fail=${SCRIPTED_INSERTION_STOP_ON_GATE_FAIL}"
  echo "external_target_mode=${EXTERNAL_TARGET_MODE}"
  echo "run_cosmos_inference=${RUN_COSMOS_INFERENCE}"
  echo "render_live_video=${RENDER_LIVE_VIDEO}"
  echo "controller_action_source=${CONTROLLER_ACTION_SOURCE}"
  echo "executor_checkpoint=${EXECUTOR_CHECKPOINT:-none}"
  echo "candidate_outcome_scorer_checkpoint=${CANDIDATE_OUTCOME_SCORER_CHECKPOINT:-none}"
  echo "candidate_outcome_scorer_summary=${CANDIDATE_OUTCOME_SCORER_SUMMARY:-none}"
  echo "candidate_outcome_scorer_dp_margin=${CANDIDATE_OUTCOME_SCORER_DP_MARGIN}"
  echo "candidate_outcome_scorer_min_progress_delta=${CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA}"
  echo "candidate_outcome_scorer_min_continuable_prob=${CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB}"
  echo "candidate_outcome_scorer_min_inserted_prob=${CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB}"
  echo "candidate_outcome_scorer_min_pred_state_x=${CANDIDATE_OUTCOME_SCORER_MIN_PRED_STATE_X}"
  echo "candidate_outcome_scorer_max_pred_state_x=${CANDIDATE_OUTCOME_SCORER_MAX_PRED_STATE_X}"
  echo "candidate_outcome_scorer_max_pred_state_abs_y=${CANDIDATE_OUTCOME_SCORER_MAX_PRED_STATE_ABS_Y}"
  echo "candidate_outcome_scorer_max_pred_state_abs_z=${CANDIDATE_OUTCOME_SCORER_MAX_PRED_STATE_ABS_Z}"
  echo "candidate_outcome_scorer_score_state_abs_axis_weights=${CANDIDATE_OUTCOME_SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS:-checkpoint_default}"
  echo "candidate_outcome_scorer_score_state_target=${CANDIDATE_OUTCOME_SCORER_SCORE_STATE_TARGET:-checkpoint_default}"
  echo "candidate_executor_short_prefix_steps=${CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS:-none}"
  echo "candidate_executor_allow_dp_prior_before_gate=${CANDIDATE_EXECUTOR_ALLOW_DP_PRIOR_BEFORE_GATE}"
  echo "source_insertion_suffix_bank=${SOURCE_INSERTION_SUFFIX_BANK:-none}"
  echo "source_suffix_k=${SOURCE_SUFFIX_K}"
  echo "source_suffix_blends=${SOURCE_SUFFIX_BLENDS}"
  echo "source_suffix_execute_steps=${SOURCE_SUFFIX_EXECUTE_STEPS}"
  echo "source_suffix_offsets=${SOURCE_SUFFIX_OFFSETS:-none}"
  echo "source_suffix_scenario_match=${SOURCE_SUFFIX_SCENARIO_MATCH}"
  echo "source_suffix_max_distance=${SOURCE_SUFFIX_MAX_DISTANCE}"
  echo "source_suffix_ignore_residual_cap=${SOURCE_SUFFIX_IGNORE_RESIDUAL_CAP}"
  echo "executor_residual_scale=${EXECUTOR_RESIDUAL_SCALE}"
  echo "save_live_state_snapshots=${SAVE_LIVE_STATE_SNAPSHOTS}"
  echo "save_candidate_action_bank=${SAVE_CANDIDATE_ACTION_BANK}"
  echo "live_progress_interval=${LIVE_PROGRESS_INTERVAL}"
  echo "pretrigger_debug_steps=${PRETRIGGER_DEBUG_STEPS}"
  echo "allow_live_receding_diagnostic=${ALLOW_LIVE_RECEDING_DIAGNOSTIC}"
  echo "diagnostic_reasons=${diagnostic_reasons[*]:-none}"
  echo "boundary=corrected full-300 live-receding diagnostic panel; unified target-motion detector controls DP vs Cosmos. If the detector never fires, DP continues by the same rule; this is not a separate static-sample branch. Not one-shot DP takeover evidence"
} | tee "${OUTPUT_ROOT}/live_receding_panel_wrapper_manifest.txt"

bool_args=()
if [[ "${RUN_COSMOS_INFERENCE}" == "true" ]]; then
  bool_args+=(--run-cosmos-inference)
else
  bool_args+=(--no-run-cosmos-inference)
fi
if [[ "${CLIP_LIVE_ACTIONS}" == "true" ]]; then
  bool_args+=(--clip-live-actions)
else
  bool_args+=(--no-clip-live-actions)
fi
if [[ "${SAVE_LIVE_STATE_SNAPSHOTS}" == "true" ]]; then
  bool_args+=(--save-live-state-snapshots)
else
  bool_args+=(--no-save-live-state-snapshots)
fi
if [[ "${SAVE_CANDIDATE_ACTION_BANK}" == "true" ]]; then
  bool_args+=(--save-candidate-action-bank)
else
  bool_args+=(--no-save-candidate-action-bank)
fi
if [[ "${RENDER_LIVE_VIDEO}" == "true" ]]; then
  bool_args+=(--render-live-video)
else
  bool_args+=(--no-render-live-video)
fi
scripted_args=(
  --scripted-insertion-mode "${SCRIPTED_INSERTION_MODE}"
  "--scripted-insertion-action=${SCRIPTED_INSERTION_ACTION}"
  --scripted-insertion-action-frame "${SCRIPTED_INSERTION_ACTION_FRAME}"
  --scripted-insertion-step-rel-x "${SCRIPTED_INSERTION_STEP_REL_X}"
  --scripted-insertion-lateral-gain "${SCRIPTED_INSERTION_LATERAL_GAIN}"
  --scripted-insertion-max-lateral-step "${SCRIPTED_INSERTION_MAX_LATERAL_STEP}"
  --scripted-insertion-action-target-source-frame "${SCRIPTED_INSERTION_ACTION_TARGET_SOURCE_FRAME}"
  --scripted-insertion-teacher-table-jsonl "${SCRIPTED_INSERTION_TEACHER_TABLE_JSONL}"
  --scripted-insertion-teacher-table-k "${SCRIPTED_INSERTION_TEACHER_TABLE_K}"
  --scripted-insertion-teacher-table-min-offset "${SCRIPTED_INSERTION_TEACHER_TABLE_MIN_OFFSET}"
  --scripted-insertion-teacher-table-max-offset "${SCRIPTED_INSERTION_TEACHER_TABLE_MAX_OFFSET}"
  --scripted-insertion-teacher-table-query-weights "${SCRIPTED_INSERTION_TEACHER_TABLE_QUERY_WEIGHTS}"
  --scripted-insertion-teacher-table-max-distance "${SCRIPTED_INSERTION_TEACHER_TABLE_MAX_DISTANCE}"
  --scripted-insertion-teacher-table-min-action-x "${SCRIPTED_INSERTION_TEACHER_TABLE_MIN_ACTION_X}"
  --scripted-insertion-teacher-table-max-action-x "${SCRIPTED_INSERTION_TEACHER_TABLE_MAX_ACTION_X}"
  --scripted-insertion-teacher-table-action-x-floor "${SCRIPTED_INSERTION_TEACHER_TABLE_ACTION_X_FLOOR}"
  --scripted-insertion-action-target-gain "${SCRIPTED_INSERTION_ACTION_TARGET_GAIN}"
  --scripted-insertion-action-target-rot-gain "${SCRIPTED_INSERTION_ACTION_TARGET_ROT_GAIN}"
  --scripted-insertion-action-target-rot-cap "${SCRIPTED_INSERTION_ACTION_TARGET_ROT_CAP}"
  --scripted-insertion-action-target-rot-source "${SCRIPTED_INSERTION_ACTION_TARGET_ROT_SOURCE}"
  --scripted-insertion-contact-seat-target-l2 "${SCRIPTED_INSERTION_CONTACT_SEAT_TARGET_L2}"
  --scripted-insertion-contact-seat-step-rel-x "${SCRIPTED_INSERTION_CONTACT_SEAT_STEP_REL_X}"
  --scripted-insertion-contact-seat-target-rel-x "${SCRIPTED_INSERTION_CONTACT_SEAT_TARGET_REL_X}"
  --scripted-insertion-contact-seat-x-servo-gain "${SCRIPTED_INSERTION_CONTACT_SEAT_X_SERVO_GAIN}"
  --scripted-insertion-contact-seat-x-servo-direction "${SCRIPTED_INSERTION_CONTACT_SEAT_X_SERVO_DIRECTION}"
  --scripted-insertion-contact-seat-min-step-rel-x "${SCRIPTED_INSERTION_CONTACT_SEAT_MIN_STEP_REL_X}"
  --scripted-insertion-contact-seat-max-step-rel-x "${SCRIPTED_INSERTION_CONTACT_SEAT_MAX_STEP_REL_X}"
  --scripted-insertion-contact-seat-x-regression-tolerance "${SCRIPTED_INSERTION_CONTACT_SEAT_X_REGRESSION_TOLERANCE}"
  --scripted-insertion-contact-seat-x-regression-policy "${SCRIPTED_INSERTION_CONTACT_SEAT_X_REGRESSION_POLICY}"
  --scripted-insertion-contact-seat-probe-policy "${SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_POLICY}"
  --scripted-insertion-contact-seat-probe-min-x-improvement "${SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_MIN_X_IMPROVEMENT}"
  --scripted-insertion-contact-seat-probe-max-lateral-regression "${SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_MAX_LATERAL_REGRESSION}"
  --scripted-insertion-contact-seat-lateral-gain "${SCRIPTED_INSERTION_CONTACT_SEAT_LATERAL_GAIN}"
  --scripted-insertion-contact-seat-max-lateral-step "${SCRIPTED_INSERTION_CONTACT_SEAT_MAX_LATERAL_STEP}"
  --scripted-insertion-contact-seat-min-servo-steps "${SCRIPTED_INSERTION_CONTACT_SEAT_MIN_SERVO_STEPS}"
  --scripted-insertion-contact-seat-max-servo-l2 "${SCRIPTED_INSERTION_CONTACT_SEAT_MAX_SERVO_L2}"
  --scripted-insertion-contact-seat-max-servo-steps "${SCRIPTED_INSERTION_CONTACT_SEAT_MAX_SERVO_STEPS}"
  --scripted-insertion-contact-seat-plateau-window "${SCRIPTED_INSERTION_CONTACT_SEAT_PLATEAU_WINDOW}"
  --scripted-insertion-contact-seat-plateau-max-l2 "${SCRIPTED_INSERTION_CONTACT_SEAT_PLATEAU_MAX_L2}"
  --scripted-insertion-contact-seat-improvement-epsilon "${SCRIPTED_INSERTION_CONTACT_SEAT_IMPROVEMENT_EPSILON}"
  --scripted-insertion-anchor-regression-tolerance "${SCRIPTED_INSERTION_ANCHOR_REGRESSION_TOLERANCE}"
  --scripted-insertion-adaptive-flip-axes "${SCRIPTED_INSERTION_ADAPTIVE_FLIP_AXES}"
  --scripted-insertion-lateral-sign "${SCRIPTED_INSERTION_LATERAL_SIGN}"
  --scripted-insertion-z-sign "${SCRIPTED_INSERTION_Z_SIGN}"
  --scripted-insertion-axis-only-after-step "${SCRIPTED_INSERTION_AXIS_ONLY_AFTER_STEP}"
  --scripted-insertion-max-steps "${SCRIPTED_INSERTION_MAX_STEPS}"
  --scripted-insertion-min-rel-x "${SCRIPTED_INSERTION_MIN_REL_X}"
  --scripted-insertion-max-rel-x "${SCRIPTED_INSERTION_MAX_REL_X}"
  --scripted-insertion-max-abs-y "${SCRIPTED_INSERTION_MAX_ABS_Y}"
  --scripted-insertion-max-abs-z "${SCRIPTED_INSERTION_MAX_ABS_Z}"
  --scripted-insertion-max-hole-speed "${SCRIPTED_INSERTION_MAX_HOLE_SPEED}"
  --scripted-insertion-terminal-pose-gate "${SCRIPTED_INSERTION_TERMINAL_POSE_GATE}"
  --scripted-insertion-terminal-anchor-source-frame "${SCRIPTED_INSERTION_TERMINAL_ANCHOR_SOURCE_FRAME}"
  --scripted-insertion-terminal-max-rel-l2 "${SCRIPTED_INSERTION_TERMINAL_MAX_REL_L2}"
  --scripted-insertion-terminal-max-peg-hole-pos-l2 "${SCRIPTED_INSERTION_TERMINAL_MAX_PEG_HOLE_POS_L2}"
  --scripted-insertion-terminal-max-peg-hole-rot-rad "${SCRIPTED_INSERTION_TERMINAL_MAX_PEG_HOLE_ROT_RAD}"
  --scripted-insertion-terminal-max-tcp-hole-pos-l2 "${SCRIPTED_INSERTION_TERMINAL_MAX_TCP_HOLE_POS_L2}"
  --scripted-insertion-terminal-max-hole-world-pos-l2 "${SCRIPTED_INSERTION_TERMINAL_MAX_HOLE_WORLD_POS_L2}"
  --scripted-insertion-terminal-max-hole-world-rot-rad "${SCRIPTED_INSERTION_TERMINAL_MAX_HOLE_WORLD_ROT_RAD}"
)
if [[ "${SCRIPTED_INSERTION_CONTACT_SEAT_LATCH}" == "true" ]]; then
  scripted_args+=(--scripted-insertion-contact-seat-latch)
else
  scripted_args+=(--no-scripted-insertion-contact-seat-latch)
fi
if [[ "${SCRIPTED_INSERTION_CONTACT_SEAT_X_SERVO}" == "true" ]]; then
  scripted_args+=(--scripted-insertion-contact-seat-x-servo)
else
  scripted_args+=(--no-scripted-insertion-contact-seat-x-servo)
fi
if [[ "${SCRIPTED_INSERTION_CONTACT_SEAT_STOP_ON_X_REGRESSION}" == "true" ]]; then
  scripted_args+=(--scripted-insertion-contact-seat-stop-on-x-regression)
else
  scripted_args+=(--no-scripted-insertion-contact-seat-stop-on-x-regression)
fi
if [[ "${SCRIPTED_INSERTION_CONTACT_SEAT_PROBE_REQUIRE_GRASP}" == "true" ]]; then
  scripted_args+=(--scripted-insertion-contact-seat-probe-require-grasp)
else
  scripted_args+=(--no-scripted-insertion-contact-seat-probe-require-grasp)
fi
if [[ "${SCRIPTED_INSERTION_REQUIRE_GRASP}" == "true" ]]; then
  scripted_args+=(--scripted-insertion-require-grasp)
else
  scripted_args+=(--no-scripted-insertion-require-grasp)
fi
if [[ "${SCRIPTED_INSERTION_ANCHOR_STOP_ON_TARGET_REGRESSION}" == "true" ]]; then
  scripted_args+=(--scripted-insertion-anchor-stop-on-target-regression)
else
  scripted_args+=(--no-scripted-insertion-anchor-stop-on-target-regression)
fi
if [[ "${SCRIPTED_INSERTION_STOP_ON_GATE_FAIL}" == "true" ]]; then
  scripted_args+=(--scripted-insertion-stop-on-gate-fail)
else
  scripted_args+=(--no-scripted-insertion-stop-on-gate-fail)
fi
if [[ "${SCRIPTED_INSERTION_IGNORE_GRASP_AFTER_TRIGGER}" == "true" ]]; then
  scripted_args+=(--scripted-insertion-ignore-grasp-after-trigger)
else
  scripted_args+=(--no-scripted-insertion-ignore-grasp-after-trigger)
fi
if [[ "${SCRIPTED_INSERTION_TEACHER_TABLE_SCENARIO_MATCH}" == "true" ]]; then
  scripted_args+=(--scripted-insertion-teacher-table-scenario-match)
else
  scripted_args+=(--no-scripted-insertion-teacher-table-scenario-match)
fi
executor_args=()
if [[ -n "${EXECUTOR_CHECKPOINT}" ]]; then
  executor_args+=(--executor-checkpoint "${EXECUTOR_CHECKPOINT}")
fi
if [[ -n "${CANDIDATE_OUTCOME_SCORER_CHECKPOINT}" ]]; then
  executor_args+=(
    --candidate-outcome-scorer-checkpoint "${CANDIDATE_OUTCOME_SCORER_CHECKPOINT}"
    --candidate-outcome-scorer-dp-margin "${CANDIDATE_OUTCOME_SCORER_DP_MARGIN}"
    --candidate-outcome-scorer-min-progress-delta "${CANDIDATE_OUTCOME_SCORER_MIN_PROGRESS_DELTA}"
    --candidate-outcome-scorer-min-continuable-prob "${CANDIDATE_OUTCOME_SCORER_MIN_CONTINUABLE_PROB}"
    --candidate-outcome-scorer-min-inserted-prob "${CANDIDATE_OUTCOME_SCORER_MIN_INSERTED_PROB}"
    --candidate-outcome-scorer-min-pred-state-x "${CANDIDATE_OUTCOME_SCORER_MIN_PRED_STATE_X}"
    --candidate-outcome-scorer-max-pred-state-x "${CANDIDATE_OUTCOME_SCORER_MAX_PRED_STATE_X}"
    --candidate-outcome-scorer-max-pred-state-abs-y "${CANDIDATE_OUTCOME_SCORER_MAX_PRED_STATE_ABS_Y}"
    --candidate-outcome-scorer-max-pred-state-abs-z "${CANDIDATE_OUTCOME_SCORER_MAX_PRED_STATE_ABS_Z}"
  )
  if [[ -n "${CANDIDATE_OUTCOME_SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS}" ]]; then
    executor_args+=(
      --candidate-outcome-scorer-score-state-abs-axis-weights "${CANDIDATE_OUTCOME_SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS}"
    )
  fi
  if [[ -n "${CANDIDATE_OUTCOME_SCORER_SCORE_STATE_TARGET}" ]]; then
    executor_args+=(
      --candidate-outcome-scorer-score-state-target "${CANDIDATE_OUTCOME_SCORER_SCORE_STATE_TARGET}"
    )
  fi
fi
if [[ -n "${CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS}" ]]; then
  executor_args+=(
    --candidate-executor-short-prefix-steps "${CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS}"
  )
fi
if [[ "${CANDIDATE_EXECUTOR_ALLOW_DP_PRIOR_BEFORE_GATE}" == "true" ]]; then
  executor_args+=(--candidate-executor-allow-dp-prior-before-gate)
else
  executor_args+=(--no-candidate-executor-allow-dp-prior-before-gate)
fi
if [[ -n "${SOURCE_INSERTION_SUFFIX_BANK}" && "${SOURCE_SUFFIX_K}" != "0" ]]; then
  executor_args+=(
    --source-insertion-suffix-bank "${SOURCE_INSERTION_SUFFIX_BANK}"
    --source-suffix-k "${SOURCE_SUFFIX_K}"
    --source-suffix-blends "${SOURCE_SUFFIX_BLENDS}"
    --source-suffix-execute-steps "${SOURCE_SUFFIX_EXECUTE_STEPS}"
    --source-suffix-query-x-weight "${SOURCE_SUFFIX_QUERY_X_WEIGHT}"
    --source-suffix-query-y-weight "${SOURCE_SUFFIX_QUERY_Y_WEIGHT}"
    --source-suffix-query-z-weight "${SOURCE_SUFFIX_QUERY_Z_WEIGHT}"
    --source-suffix-max-distance "${SOURCE_SUFFIX_MAX_DISTANCE}"
  )
  if [[ -n "${SOURCE_SUFFIX_OFFSETS}" ]]; then
    executor_args+=(--source-suffix-offsets "${SOURCE_SUFFIX_OFFSETS}")
  fi
  if [[ "${SOURCE_SUFFIX_SCENARIO_MATCH}" == "true" ]]; then
    executor_args+=(--source-suffix-scenario-match)
  else
    executor_args+=(--no-source-suffix-scenario-match)
  fi
  if [[ "${SOURCE_SUFFIX_IGNORE_RESIDUAL_CAP}" == "true" ]]; then
    executor_args+=(--source-suffix-ignore-residual-cap)
  else
    executor_args+=(--no-source-suffix-ignore-residual-cap)
  fi
fi

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/run_cosmos3_live_receding_panel.py" \
  --eval-root "${EVAL_ROOT}" \
  --source-h5-root "${SOURCE_H5_ROOT}" \
  --condition-root "${CONDITION_ROOT}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-file "${CONFIG_FILE}" \
  --dp-manifest "${DP_MANIFEST}" \
  --dp-checkpoint "${DP_CHECKPOINT}" \
  --output-root "${OUTPUT_ROOT}" \
  --sample-indices "${SAMPLE_INDICES}" \
  --max-samples "${MAX_SAMPLES}" \
  --prefix-role-mode "${PREFIX_ROLE_MODE}" \
  --prefix-start-mode "${PREFIX_START_MODE}" \
  --pretrigger-control-mode "${PRETRIGGER_CONTROL_MODE}" \
  --min-dynamic-prefix-frame "${MIN_DYNAMIC_PREFIX_FRAME}" \
  --target-motion-consecutive-frames "${TARGET_MOTION_CONSECUTIVE_FRAMES}" \
  --max-receding-iterations "${MAX_RECEDING_ITERATIONS}" \
  --action-exec-horizon "${ACTION_EXEC_HORIZON}" \
  --dp-handoff-horizon "${DP_HANDOFF_HORIZON}" \
  --dp-handoff-chunk-horizon "${DP_HANDOFF_CHUNK_HORIZON}" \
  --dp-action-seed-base "${DP_ACTION_SEED_BASE}" \
  --continuability-stats-json "${CONTINUABILITY_STATS_JSON}" \
  --external-target-mode "${EXTERNAL_TARGET_MODE}" \
  --controller-action-source "${CONTROLLER_ACTION_SOURCE}" \
  --executor-residual-scale "${EXECUTOR_RESIDUAL_SCALE}" \
  --video-fps "${VIDEO_FPS}" \
  --live-progress-interval "${LIVE_PROGRESS_INTERVAL}" \
  --pretrigger-debug-steps "${PRETRIGGER_DEBUG_STEPS}" \
  --oracle-final-seat-mode "${ORACLE_FINAL_SEAT_MODE}" \
  --oracle-final-seat-source-frame "${ORACLE_FINAL_SEAT_SOURCE_FRAME}" \
  --live-geometric-final-seat-mode "${LIVE_GEOMETRIC_FINAL_SEAT_MODE}" \
  --live-geometric-final-seat-target-rel="${LIVE_GEOMETRIC_FINAL_SEAT_TARGET_REL}" \
  "${scripted_args[@]}" \
  "${executor_args[@]}" \
  "${bool_args[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/live_receding_panel.log"
