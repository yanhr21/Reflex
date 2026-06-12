#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this wrapper only inside a compute-node srun step. It preflights the Cosmos3 closed-loop gate before any live controller work.
EOF
  exit 30
fi

if [[ "${SLURM_NTASKS:-1}" != "1" || "${SLURM_PROCID:-0}" != "0" ]]; then
  cat >&2 <<'EOF'
refusing_multi_task_execution=true
reason=Run this wrapper as a single Slurm task, e.g. srun --ntasks=1 --jobid=$SLURM_JOB_ID bash scripts/slurm/run_cosmos3_receding_closed_loop_in_allocation.sh
EOF
  exit 31
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-300}"
printf -v CHECKPOINT_NAME 'iter_%09d' "${CHECKPOINT_ITER}"

JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SFT_ROOT}/outputs/cosmos3/sft/${JOB_NAME}/checkpoints/${CHECKPOINT_NAME}}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_${CHECKPOINT_NAME}}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_rgb_300step_20260612_0245}"
DP_CHECKPOINT="${DP_CHECKPOINT:-${ROOT}/experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt}"
DP_MANIFEST="${DP_MANIFEST:-${ROOT}/experiments/dp_peg1000/run_90201/manifest.json}"
DP_STATE_KEY="${DP_STATE_KEY:-ema_agent}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SFT_ROOT}/closed_loop_preflight_${CHECKPOINT_NAME}}"
READOUT_SUBDIR="${READOUT_SUBDIR:-task_state_readout_v7_733}"
VISUAL_REVIEW_STATUS="${VISUAL_REVIEW_STATUS:-missing}"
VISUAL_REVIEW_NOTE="${VISUAL_REVIEW_NOTE:-}"
MODE="${MODE:-preflight}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-8}"
DP_RESUME_HORIZON="${DP_RESUME_HORIZON:-8}"
ACTION_PREVIEW_SAMPLE_INDEX="${ACTION_PREVIEW_SAMPLE_INDEX:-0}"
EXPECTED_VIDEO_FRAMES="${EXPECTED_VIDEO_FRAMES:-301}"
EXPECTED_ACTION_STEPS="${EXPECTED_ACTION_STEPS:-300}"
EXPECTED_ACTION_DIM="${EXPECTED_ACTION_DIM:-32}"
ROBOT_ACTION_DIM="${ROBOT_ACTION_DIM:-7}"
CAPTURE_LIVE_VIDEO="${CAPTURE_LIVE_VIDEO:-true}"
LIVE_VIDEO_FPS="${LIVE_VIDEO_FPS:-10}"
CLIP_LIVE_ACTIONS="${CLIP_LIVE_ACTIONS:-true}"

main() {
  cd "${ROOT}"
  mkdir -p "${OUTPUT_ROOT}"
  {
    echo "timestamp=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "step_id=${SLURM_STEP_ID}"
    echo "ntasks=${SLURM_NTASKS:-1}"
    echo "sft_root=${SFT_ROOT}"
    echo "checkpoint_path=${CHECKPOINT_PATH}"
    echo "eval_root=${EVAL_ROOT}"
    echo "condition_root=${CONDITION_ROOT}"
    echo "dp_checkpoint=${DP_CHECKPOINT}"
    echo "dp_manifest=${DP_MANIFEST}"
    echo "dp_state_key=${DP_STATE_KEY}"
    echo "output_root=${OUTPUT_ROOT}"
    echo "visual_review_status=${VISUAL_REVIEW_STATUS}"
    echo "mode=${MODE}"
    echo "action_preview_sample_index=${ACTION_PREVIEW_SAMPLE_INDEX}"
    echo "action_exec_horizon=${ACTION_EXEC_HORIZON}"
    echo "dp_resume_horizon=${DP_RESUME_HORIZON}"
    echo "capture_live_video=${CAPTURE_LIVE_VIDEO}"
    echo "live_video_fps=${LIVE_VIDEO_FPS}"
    echo "clip_live_actions=${CLIP_LIVE_ACTIONS}"
    echo "evidence_boundary=Preflight/gate only; not controller success evidence."
  } | tee "${OUTPUT_ROOT}/closed_loop_wrapper_manifest.txt"

  local bool_args=()
  if [[ "${CAPTURE_LIVE_VIDEO}" == "true" ]]; then
    bool_args+=(--capture-live-video)
  else
    bool_args+=(--no-capture-live-video)
  fi
  if [[ "${CLIP_LIVE_ACTIONS}" == "true" ]]; then
    bool_args+=(--clip-live-actions)
  else
    bool_args+=(--no-clip-live-actions)
  fi

  "${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/run_cosmos3_receding_closed_loop.py" \
    --eval-root "${EVAL_ROOT}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --condition-root "${CONDITION_ROOT}" \
    --dp-checkpoint "${DP_CHECKPOINT}" \
    --dp-manifest "${DP_MANIFEST}" \
    --dp-state-key "${DP_STATE_KEY}" \
    --output-root "${OUTPUT_ROOT}" \
    --readout-subdir "${READOUT_SUBDIR}" \
    --visual-review-status "${VISUAL_REVIEW_STATUS}" \
    --visual-review-note "${VISUAL_REVIEW_NOTE}" \
    --expected-video-frames "${EXPECTED_VIDEO_FRAMES}" \
    --expected-action-steps "${EXPECTED_ACTION_STEPS}" \
    --expected-action-dim "${EXPECTED_ACTION_DIM}" \
    --robot-action-dim "${ROBOT_ACTION_DIM}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --action-exec-horizon "${ACTION_EXEC_HORIZON}" \
    --dp-resume-horizon "${DP_RESUME_HORIZON}" \
    --live-video-fps "${LIVE_VIDEO_FPS}" \
    --action-preview-sample-index "${ACTION_PREVIEW_SAMPLE_INDEX}" \
    --mode "${MODE}" \
    "${bool_args[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/closed_loop_preflight.log"
}

main "$@"
