#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this panel wrapper only inside a compute-node srun step. It executes live ManiSkill smoke rollouts.
EOF
  exit 30
fi

if [[ "${SLURM_NTASKS:-1}" != "1" || "${SLURM_PROCID:-0}" != "0" ]]; then
  cat >&2 <<'EOF'
refusing_multi_task_execution=true
reason=Run this panel wrapper as exactly one Slurm task so live rollouts are serialized and auditable.
EOF
  exit 31
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-2400}"
printf -v CHECKPOINT_NAME 'iter_%09d' "${CHECKPOINT_ITER}"

JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SFT_ROOT}/outputs/cosmos3/sft/${JOB_NAME}/checkpoints/${CHECKPOINT_NAME}}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_${CHECKPOINT_NAME}}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_rgb_300step_20260612_0245}"
DP_CHECKPOINT="${DP_CHECKPOINT:-${ROOT}/experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt}"
DP_MANIFEST="${DP_MANIFEST:-${ROOT}/experiments/dp_peg1000/run_90201/manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SFT_ROOT}/closed_loop_smoke_${CHECKPOINT_NAME}_panel${N_SAMPLES:-10}_dp${DP_RESUME_HORIZON:-8}_recompute_${STAMP}}"
READOUT_SUBDIR="${READOUT_SUBDIR:-task_state_readout_v7_733}"
VISUAL_REVIEW_STATUS="${VISUAL_REVIEW_STATUS:-missing}"
VISUAL_REVIEW_NOTE="${VISUAL_REVIEW_NOTE:-panel_run_requires_explicit_prior_manual_visual_gate}"
N_SAMPLES="${N_SAMPLES:-10}"
SAMPLE_INDICES="${SAMPLE_INDICES:-}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-8}"
DP_RESUME_HORIZON="${DP_RESUME_HORIZON:-8}"
CAPTURE_LIVE_VIDEO="${CAPTURE_LIVE_VIDEO:-true}"
LIVE_VIDEO_FPS="${LIVE_VIDEO_FPS:-10}"
CLIP_LIVE_ACTIONS="${CLIP_LIVE_ACTIONS:-true}"
ALLOW_LONG_DP_DIAGNOSTIC="${ALLOW_LONG_DP_DIAGNOSTIC:-false}"
ALLOW_OLD_ONESHOT_CLOSED_LOOP_DIAGNOSTIC="${ALLOW_OLD_ONESHOT_CLOSED_LOOP_DIAGNOSTIC:-false}"
EXPECTED_VIDEO_FRAMES="${EXPECTED_VIDEO_FRAMES:-301}"
EXPECTED_ACTION_STEPS="${EXPECTED_ACTION_STEPS:-300}"
EXPECTED_ACTION_DIM="${EXPECTED_ACTION_DIM:-32}"
ROBOT_ACTION_DIM="${ROBOT_ACTION_DIM:-7}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"

if [[ "${ALLOW_OLD_ONESHOT_CLOSED_LOOP_DIAGNOSTIC}" != "true" ]]; then
  cat >&2 <<'EOF'
refusing_old_oneshot_closed_loop_panel=true
reason=This wrapper runs the old one-shot Cosmos chunk plus optional DP resume diagnostic, not the corrected full-300 live-receding method.
override_for_non_method_diagnostic=ALLOW_OLD_ONESHOT_CLOSED_LOOP_DIAGNOSTIC=true
required_method_wrapper=scripts/slurm/run_cosmos3_live_receding_panel_in_allocation.sh
EOF
  exit 43
fi

sample_indices() {
  if [[ -n "${SAMPLE_INDICES}" ]]; then
    printf '%s\n' "${SAMPLE_INDICES}" | tr ',' '\n' | awk 'NF {print $1}'
  else
    seq 0 "$((N_SAMPLES - 1))"
  fi
}

mkdir -p "${OUTPUT_ROOT}"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-}"
  echo "sft_root=${SFT_ROOT}"
  echo "checkpoint_iter=${CHECKPOINT_ITER}"
  echo "checkpoint_path=${CHECKPOINT_PATH}"
  echo "eval_root=${EVAL_ROOT}"
  echo "condition_root=${CONDITION_ROOT}"
  echo "dp_checkpoint=${DP_CHECKPOINT}"
  echo "dp_manifest=${DP_MANIFEST}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "readout_subdir=${READOUT_SUBDIR}"
  echo "visual_review_status=${VISUAL_REVIEW_STATUS}"
  echo "visual_review_note=${VISUAL_REVIEW_NOTE}"
  echo "n_samples=${N_SAMPLES}"
  echo "sample_indices=${SAMPLE_INDICES:-0..$((N_SAMPLES - 1))}"
  echo "action_exec_horizon=${ACTION_EXEC_HORIZON}"
  echo "dp_resume_horizon=${DP_RESUME_HORIZON}"
  echo "allow_long_dp_diagnostic=${ALLOW_LONG_DP_DIAGNOSTIC}"
  echo "allow_old_oneshot_closed_loop_diagnostic=${ALLOW_OLD_ONESHOT_CLOSED_LOOP_DIAGNOSTIC}"
  echo "capture_live_video=${CAPTURE_LIVE_VIDEO}"
  echo "live_video_fps=${LIVE_VIDEO_FPS}"
  echo "clip_live_actions=${CLIP_LIVE_ACTIONS}"
  echo "boundary=gated diagnostic one-shot live smoke panel; not full receding-Cosmos controller success evidence"
} | tee "${OUTPUT_ROOT}/panel_manifest.txt"

if [[ "${VISUAL_REVIEW_STATUS}" != "pass" ]]; then
  cat >&2 <<EOF
closed_loop_panel_refused=true
reason=VISUAL_REVIEW_STATUS must be explicitly set to pass after manual review.
visual_review_status=${VISUAL_REVIEW_STATUS}
EOF
  exit 41
fi

if (( DP_RESUME_HORIZON > ACTION_EXEC_HORIZON )) && [[ "${ALLOW_LONG_DP_DIAGNOSTIC}" != "true" ]]; then
  cat >&2 <<EOF
closed_loop_panel_refused=true
reason=Long frozen-DP takeover requires ALLOW_LONG_DP_DIAGNOSTIC=true and remains non-method diagnostic evidence.
action_exec_horizon=${ACTION_EXEC_HORIZON}
dp_resume_horizon=${DP_RESUME_HORIZON}
EOF
  exit 42
fi

cd "${ROOT}"
failed=0
for sample_index in $(sample_indices); do
  sample_dir="${OUTPUT_ROOT}/sample_$(printf '%02d' "${sample_index}")"
  mkdir -p "${sample_dir}"
  {
    echo "sample_index=${sample_index}"
    echo "start_time=$(date -Is)"
  } > "${sample_dir}/launch_status.txt"

  set +e
  SFT_ROOT="${SFT_ROOT}" \
    CHECKPOINT_ITER="${CHECKPOINT_ITER}" \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    EVAL_ROOT="${EVAL_ROOT}" \
    CONDITION_ROOT="${CONDITION_ROOT}" \
    DP_CHECKPOINT="${DP_CHECKPOINT}" \
    DP_MANIFEST="${DP_MANIFEST}" \
    OUTPUT_ROOT="${sample_dir}" \
    READOUT_SUBDIR="${READOUT_SUBDIR}" \
    VISUAL_REVIEW_STATUS="${VISUAL_REVIEW_STATUS}" \
    VISUAL_REVIEW_NOTE="${VISUAL_REVIEW_NOTE}" \
    MODE=smoke \
    ACTION_PREVIEW_SAMPLE_INDEX="${sample_index}" \
    ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON}" \
    DP_RESUME_HORIZON="${DP_RESUME_HORIZON}" \
    ALLOW_LONG_DP_DIAGNOSTIC="${ALLOW_LONG_DP_DIAGNOSTIC}" \
    ALLOW_OLD_ONESHOT_CLOSED_LOOP_DIAGNOSTIC="${ALLOW_OLD_ONESHOT_CLOSED_LOOP_DIAGNOSTIC}" \
    CAPTURE_LIVE_VIDEO="${CAPTURE_LIVE_VIDEO}" \
    LIVE_VIDEO_FPS="${LIVE_VIDEO_FPS}" \
    CLIP_LIVE_ACTIONS="${CLIP_LIVE_ACTIONS}" \
    EXPECTED_VIDEO_FRAMES="${EXPECTED_VIDEO_FRAMES}" \
    EXPECTED_ACTION_STEPS="${EXPECTED_ACTION_STEPS}" \
    EXPECTED_ACTION_DIM="${EXPECTED_ACTION_DIM}" \
    ROBOT_ACTION_DIM="${ROBOT_ACTION_DIM}" \
    MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
    bash scripts/slurm/run_cosmos3_receding_closed_loop_in_allocation.sh \
    > "${sample_dir}/wrapper_stdout.log" 2>&1
  code=$?
  set -e

  {
    echo "end_time=$(date -Is)"
    echo "exit_code=${code}"
  } >> "${sample_dir}/launch_status.txt"
  if [[ "${code}" != "0" ]]; then
    failed=1
  fi
done

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/summarize_cosmos3_closed_loop_panel.py" \
  --panel-root "${OUTPUT_ROOT}" \
  --output-summary "${OUTPUT_ROOT}/panel_summary.json" \
  --output-visual-review "${OUTPUT_ROOT}/panel_visual_review.json" \
  --contact-sheet "${OUTPUT_ROOT}/panel_contact_sheet_start_mid_final.png" \
  --dp-resume-horizon "${DP_RESUME_HORIZON}" \
  2>&1 | tee "${OUTPUT_ROOT}/summarize_panel.log"

if [[ "${failed}" != "0" ]]; then
  echo "panel_failed_samples_present=true" | tee -a "${OUTPUT_ROOT}/panel_manifest.txt"
  exit 50
fi

echo "panel_complete=$(date -Is)" | tee -a "${OUTPUT_ROOT}/panel_manifest.txt"
