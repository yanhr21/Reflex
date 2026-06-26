#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run Cosmos3 live receding loop only inside a compute-node srun step.
EOF
  exit 30
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_rgb_300step_20260612_0245}"
JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
RUN_DIR="${RUN_DIR:-${SFT_ROOT}/outputs/cosmos3/sft/${JOB_NAME}}"
CONFIG_FILE="${CONFIG_FILE:-${RUN_DIR}/config.yaml}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-2700}"
printf -v CHECKPOINT_NAME 'iter_%09d' "${CHECKPOINT_ITER}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RUN_DIR}/checkpoints/${CHECKPOINT_NAME}}"
SOURCE_H5="${SOURCE_H5:?set SOURCE_H5 to the source H5 trajectory}"
DP_MANIFEST="${DP_MANIFEST:-${ROOT}/experiments/dp_peg1000/run_90201/manifest.json}"
DP_CHECKPOINT="${DP_CHECKPOINT:-${ROOT}/experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT for live receding outputs}"
PREFIX_START_MODE="${PREFIX_START_MODE:-target_motion_onset}"
PRETRIGGER_CONTROL_MODE="${PRETRIGGER_CONTROL_MODE:-frozen_dp_until_target_motion}"
PREFIX_FRAME_INDEX="${PREFIX_FRAME_INDEX:--1}"
MIN_DYNAMIC_PREFIX_FRAME="${MIN_DYNAMIC_PREFIX_FRAME:-8}"
TARGET_MOTION_CONSECUTIVE_FRAMES="${TARGET_MOTION_CONSECUTIVE_FRAMES:-2}"
SCENARIO="${SCENARIO:-hole_late_move_stop}"
PREFIX_ROLE="${PREFIX_ROLE:-auto}"
SAMPLE_NAME="${SAMPLE_NAME:-live_receding}"
MAX_RECEDING_ITERATIONS="${MAX_RECEDING_ITERATIONS:-40}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-8}"
DP_HANDOFF_HORIZON="${DP_HANDOFF_HORIZON:-64}"
CONTINUABILITY_STATS_JSON="${CONTINUABILITY_STATS_JSON:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/dp_static_continuability_stats_20260613/dp_static_continuability_stats.json}"
CONTINUABILITY_STATS_HORIZON="${CONTINUABILITY_STATS_HORIZON:-32}"
RUN_COSMOS_INFERENCE="${RUN_COSMOS_INFERENCE:-true}"
CONTROLLER_ACTION_SOURCE="${CONTROLLER_ACTION_SOURCE:-cosmos_robot_action}"
EXECUTOR_CHECKPOINT="${EXECUTOR_CHECKPOINT:-}"
EXECUTOR_RESIDUAL_SCALE="${EXECUTOR_RESIDUAL_SCALE:-1.0}"
CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS="${CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS:-}"
ALLOW_LIVE_RECEDING_DIAGNOSTIC="${ALLOW_LIVE_RECEDING_DIAGNOSTIC:-false}"
SAVE_LIVE_STATE_SNAPSHOTS="${SAVE_LIVE_STATE_SNAPSHOTS:-false}"
SAVE_CANDIDATE_ACTION_BANK="${SAVE_CANDIDATE_ACTION_BANK:-false}"
LIVE_PROGRESS_INTERVAL="${LIVE_PROGRESS_INTERVAL:-0}"

main() {
  cd "${ROOT}"
  local diagnostic_reasons=()
  if [[ "${PREFIX_ROLE}" != "auto" ]]; then
    diagnostic_reasons+=("prefix_role_not_auto:${PREFIX_ROLE}")
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
  if [[ "${CONTROLLER_ACTION_SOURCE}" == "residual_executor" || "${CONTROLLER_ACTION_SOURCE}" == "contact_executor" || "${CONTROLLER_ACTION_SOURCE}" == "candidate_executor" ]]; then
    if [[ ! -s "${EXECUTOR_CHECKPOINT}" ]]; then
      diagnostic_reasons+=("missing_${CONTROLLER_ACTION_SOURCE}_checkpoint")
    fi
  fi
  if (( ${#diagnostic_reasons[@]} > 0 )) && [[ "${ALLOW_LIVE_RECEDING_DIAGNOSTIC}" != "true" ]]; then
    {
      echo "refusing_non_method_live_receding_loop=true"
      echo "reason=${diagnostic_reasons[*]}"
      echo "override_for_non_method_diagnostic=ALLOW_LIVE_RECEDING_DIAGNOSTIC=true"
      echo "required_method_modes=PREFIX_ROLE=auto PREFIX_START_MODE=target_motion_onset PRETRIGGER_CONTROL_MODE=frozen_dp_until_target_motion RUN_COSMOS_INFERENCE=true"
    } >&2
    exit 42
  fi

  [[ -x "${ROOT}/.venv/bin/python" ]] || { echo "missing ${ROOT}/.venv/bin/python" >&2; exit 2; }
  [[ -s "${SOURCE_H5}" ]] || { echo "missing SOURCE_H5=${SOURCE_H5}" >&2; exit 3; }
  [[ -d "${CONDITION_ROOT}" ]] || { echo "missing CONDITION_ROOT=${CONDITION_ROOT}" >&2; exit 4; }
  [[ -d "${CHECKPOINT_PATH}" ]] || { echo "missing CHECKPOINT_PATH=${CHECKPOINT_PATH}" >&2; exit 5; }
  [[ -s "${CONFIG_FILE}" ]] || { echo "missing CONFIG_FILE=${CONFIG_FILE}" >&2; exit 6; }
  [[ -s "${DP_MANIFEST}" ]] || { echo "missing DP_MANIFEST=${DP_MANIFEST}" >&2; exit 7; }
  [[ -s "${DP_CHECKPOINT}" ]] || { echo "missing DP_CHECKPOINT=${DP_CHECKPOINT}" >&2; exit 8; }
  [[ -s "${CONTINUABILITY_STATS_JSON}" ]] || { echo "missing CONTINUABILITY_STATS_JSON=${CONTINUABILITY_STATS_JSON}" >&2; exit 9; }

  mkdir -p "${OUTPUT_ROOT}"
  {
    echo "timestamp=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "step_id=${SLURM_STEP_ID}"
    echo "source_h5=${SOURCE_H5}"
    echo "condition_root=${CONDITION_ROOT}"
    echo "checkpoint_path=${CHECKPOINT_PATH}"
    echo "config_file=${CONFIG_FILE}"
    echo "dp_manifest=${DP_MANIFEST}"
    echo "dp_checkpoint=${DP_CHECKPOINT}"
    echo "output_root=${OUTPUT_ROOT}"
    echo "prefix_start_mode=${PREFIX_START_MODE}"
    echo "pretrigger_control_mode=${PRETRIGGER_CONTROL_MODE}"
    echo "prefix_frame_index=${PREFIX_FRAME_INDEX}"
    echo "min_dynamic_prefix_frame=${MIN_DYNAMIC_PREFIX_FRAME}"
    echo "target_motion_consecutive_frames=${TARGET_MOTION_CONSECUTIVE_FRAMES}"
    echo "scenario=${SCENARIO}"
    echo "prefix_role=${PREFIX_ROLE}"
    echo "max_receding_iterations=${MAX_RECEDING_ITERATIONS}"
    echo "action_exec_horizon=${ACTION_EXEC_HORIZON}"
    echo "dp_handoff_horizon=${DP_HANDOFF_HORIZON}"
    echo "continuability_stats_json=${CONTINUABILITY_STATS_JSON}"
    echo "continuability_stats_horizon=${CONTINUABILITY_STATS_HORIZON}"
    echo "run_cosmos_inference=${RUN_COSMOS_INFERENCE}"
    echo "controller_action_source=${CONTROLLER_ACTION_SOURCE}"
    echo "executor_checkpoint=${EXECUTOR_CHECKPOINT:-none}"
    echo "executor_residual_scale=${EXECUTOR_RESIDUAL_SCALE}"
    echo "candidate_executor_short_prefix_steps=${CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS:-none}"
    echo "save_live_state_snapshots=${SAVE_LIVE_STATE_SNAPSHOTS}"
    echo "save_candidate_action_bank=${SAVE_CANDIDATE_ACTION_BANK}"
    echo "live_progress_interval=${LIVE_PROGRESS_INTERVAL}"
    echo "allow_live_receding_diagnostic=${ALLOW_LIVE_RECEDING_DIAGNOSTIC}"
    echo "diagnostic_reasons=${diagnostic_reasons[*]:-none}"
    echo "boundary=Full-300 live receding Cosmos eval wrapper; one causal target-motion detector controls DP vs Cosmos. If the detector never fires, DP continues by the same rule; this is not a separate static-sample branch. No SFT checkpoint writes."
  } | tee "${OUTPUT_ROOT}/live_receding_wrapper_manifest.txt"

  local args=(
    --source-h5 "${SOURCE_H5}"
    --prefix-frame-source render_env_states
    --condition-root "${CONDITION_ROOT}"
    --checkpoint-path "${CHECKPOINT_PATH}"
    --config-file "${CONFIG_FILE}"
    --dp-manifest "${DP_MANIFEST}"
    --dp-checkpoint "${DP_CHECKPOINT}"
    --output-root "${OUTPUT_ROOT}"
    --prefix-start-mode "${PREFIX_START_MODE}"
    --pretrigger-control-mode "${PRETRIGGER_CONTROL_MODE}"
    --prefix-frame-index "${PREFIX_FRAME_INDEX}"
    --min-dynamic-prefix-frame "${MIN_DYNAMIC_PREFIX_FRAME}"
    --target-motion-consecutive-frames "${TARGET_MOTION_CONSECUTIVE_FRAMES}"
    --scenario "${SCENARIO}"
    --prefix-role "${PREFIX_ROLE}"
    --sample-name "${SAMPLE_NAME}"
    --max-receding-iterations "${MAX_RECEDING_ITERATIONS}"
    --action-exec-horizon "${ACTION_EXEC_HORIZON}"
    --dp-handoff-horizon "${DP_HANDOFF_HORIZON}"
    --continuability-stats-json "${CONTINUABILITY_STATS_JSON}"
    --continuability-stats-horizon "${CONTINUABILITY_STATS_HORIZON}"
    --live-progress-interval "${LIVE_PROGRESS_INTERVAL}"
    --controller-action-source "${CONTROLLER_ACTION_SOURCE}"
    --executor-residual-scale "${EXECUTOR_RESIDUAL_SCALE}"
  )
  if [[ -n "${EXECUTOR_CHECKPOINT}" ]]; then
    args+=(--executor-checkpoint "${EXECUTOR_CHECKPOINT}")
  fi
  if [[ -n "${CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS}" ]]; then
    args+=(--candidate-executor-short-prefix-steps "${CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS}")
  fi
  if [[ "${RUN_COSMOS_INFERENCE}" == "true" ]]; then
    args+=(--run-cosmos-inference)
  else
    args+=(--no-run-cosmos-inference)
  fi
  if [[ "${SAVE_LIVE_STATE_SNAPSHOTS}" == "true" ]]; then
    args+=(--save-live-state-snapshots)
  else
    args+=(--no-save-live-state-snapshots)
  fi
  if [[ "${SAVE_CANDIDATE_ACTION_BANK}" == "true" ]]; then
    args+=(--save-candidate-action-bank)
  else
    args+=(--no-save-candidate-action-bank)
  fi

  "${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/run_cosmos3_live_receding_loop.py" "${args[@]}"
}

main "$@"
