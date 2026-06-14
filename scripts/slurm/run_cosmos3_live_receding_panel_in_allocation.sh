#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

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
EXTERNAL_TARGET_MODE="${EXTERNAL_TARGET_MODE:-source_env_state}"
RUN_COSMOS_INFERENCE="${RUN_COSMOS_INFERENCE:-true}"
CLIP_LIVE_ACTIONS="${CLIP_LIVE_ACTIONS:-true}"
VIDEO_FPS="${VIDEO_FPS:-30}"
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
  echo "external_target_mode=${EXTERNAL_TARGET_MODE}"
  echo "run_cosmos_inference=${RUN_COSMOS_INFERENCE}"
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
  --continuability-stats-json "${CONTINUABILITY_STATS_JSON}" \
  --external-target-mode "${EXTERNAL_TARGET_MODE}" \
  --video-fps "${VIDEO_FPS}" \
  "${bool_args[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/live_receding_panel.log"
