#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this only inside a compute-node srun step from a tmux-held allocation.
example=srun --overlap --jobid=$SLURM_JOB_ID --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=120G bash scripts/slurm/run_cosmos3_targeted_recovery_supplement_after_approval_in_allocation.sh
EOF
  exit 30
fi

if [[ "${ALLOW_TARGETED_RECOVERY_SUPPLEMENT:-false}" != "true" ]]; then
  cat >&2 <<'EOF'
refusing_targeted_recovery_supplement=true
reason=This changes the data boundary beyond the frozen 733-only export repair. Set ALLOW_TARGETED_RECOVERY_SUPPLEMENT=true only after explicit user approval.
boundary=This wrapper generates targeted hard-teacher H5 rows and RGB review sheets only. It does not merge into the 733 source, export WAM conditions, or start SFT.
EOF
  exit 45
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/targeted_recovery_supplement_after_approval_${STAMP}}"
H5_ROOT="${H5_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/fix3_targeted_recovery_hard_teacher_${STAMP}}"
RGB_ROOT="${RGB_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_targeted_recovery_hard_teacher_rgb_512_${STAMP}}"
GAP_MANIFEST="${GAP_MANIFEST:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/targeted_recovery_gap_manifest_20260614_from_late299.json}"

SCENARIO_SEQUENCE="${SCENARIO_SEQUENCE:-hole_late_sine,hole_late_constant,hole_late_continuous_insert,hole_late_fast_shift}"
SCENARIO_QUOTAS="${SCENARIO_QUOTAS:-hole_late_sine=40,hole_late_constant=24,hole_late_continuous_insert=24,hole_late_fast_shift=24}"
NUM_DEMOS="${NUM_DEMOS:-112}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3000}"
SEED="${SEED:-56000000}"
SCENARIO_SEED_BASES="${SCENARIO_SEED_BASES:-hole_late_sine=56510000,hole_late_constant=56250000,hole_late_continuous_insert=56241000,hole_late_fast_shift=56300000}"
POST_MOTION_RELEASE_REGRASP_SCENARIOS="${POST_MOTION_RELEASE_REGRASP_SCENARIOS:-hole_late_fast_shift}"
POST_MOTION_RELEASE_REGRASP_HOLD_STEPS="${POST_MOTION_RELEASE_REGRASP_HOLD_STEPS:-4}"
MOTION_MIN_M="${MOTION_MIN_M:-0.10}"
MOTION_MAX_M="${MOTION_MAX_M:-0.22}"
MIN_ABS_Y_MOTION_M="${MIN_ABS_Y_MOTION_M:-0.025}"
ANTI_SELF_INSERT_FINAL_YZ_MIN_M="${ANTI_SELF_INSERT_FINAL_YZ_MIN_M:-0.035}"
ANTI_SELF_INSERT_RADIUS_MULTIPLIER="${ANTI_SELF_INSERT_RADIUS_MULTIPLIER:-1.4}"
ANTI_SELF_INSERT_EXEMPT_SCENARIOS="${ANTI_SELF_INSERT_EXEMPT_SCENARIOS:-hole_late_fast_shift}"
INITIAL_PREINSERT_RETREAT_EXTRA_M="${INITIAL_PREINSERT_RETREAT_EXTRA_M:-0.06}"
INITIAL_PREINSERT_LINE_YZ_MAX_M="${INITIAL_PREINSERT_LINE_YZ_MAX_M:-0.008}"

WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
FPS="${FPS:-30}"
SHEET_LIMIT="${SHEET_LIMIT:-10}"
SHEET_FRAMES="${SHEET_FRAMES:-24}"
SHEET_THUMB_WIDTH="${SHEET_THUMB_WIDTH:-384}"

cd "${ROOT}"
mkdir -p "${RUN_ROOT}" "${H5_ROOT}" "${RGB_ROOT}"

export PYTHONPATH="${ROOT}/deps/ManiSkill_clean:${ROOT}:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export DISPLAY=

{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "host=$(hostname)"
  echo "run_root=${RUN_ROOT}"
  echo "h5_root=${H5_ROOT}"
  echo "rgb_root=${RGB_ROOT}"
  echo "gap_manifest=${GAP_MANIFEST}"
  echo "scenario_sequence=${SCENARIO_SEQUENCE}"
  echo "scenario_quotas=${SCENARIO_QUOTAS}"
  echo "num_demos=${NUM_DEMOS}"
  echo "max_attempts=${MAX_ATTEMPTS}"
  echo "seed=${SEED}"
  echo "scenario_seed_bases=${SCENARIO_SEED_BASES}"
  echo "post_motion_release_regrasp_scenarios=${POST_MOTION_RELEASE_REGRASP_SCENARIOS}"
  echo "post_motion_release_regrasp_hold_steps=${POST_MOTION_RELEASE_REGRASP_HOLD_STEPS}"
  echo "motion_min_m=${MOTION_MIN_M}"
  echo "motion_max_m=${MOTION_MAX_M}"
  echo "min_abs_y_motion_m=${MIN_ABS_Y_MOTION_M}"
  echo "anti_self_insert_final_yz_min_m=${ANTI_SELF_INSERT_FINAL_YZ_MIN_M}"
  echo "anti_self_insert_radius_multiplier=${ANTI_SELF_INSERT_RADIUS_MULTIPLIER}"
  echo "anti_self_insert_exempt_scenarios=${ANTI_SELF_INSERT_EXEMPT_SCENARIOS}"
  echo "initial_preinsert_retreat_extra_m=${INITIAL_PREINSERT_RETREAT_EXTRA_M}"
  echo "initial_preinsert_line_yz_max_m=${INITIAL_PREINSERT_LINE_YZ_MAX_M}"
  echo "render_width=${WIDTH}"
  echo "render_height=${HEIGHT}"
  echo "render_fps=${FPS}"
  echo "sheet_limit=${SHEET_LIMIT}"
  echo "boundary=targeted supplement generation and visual review only; no merge, no WAM export, no SFT"
  echo "physical_reason=cover live closed-loop states still missing after 733-only dense export: late target_post_motion y/z offsets and fast-shift peg_recovery."
} | tee "${RUN_ROOT}/manifest.txt"

"${ROOT}/.venv/bin/python" -m py_compile \
  scripts/world_model/generate_cosmos3_fix3_hard_dynamic_teacher.py \
  scripts/world_model/render_cosmos3_maniskill_sft_dataset.py \
  scripts/world_model/inspect_cosmos3_targeted_recovery_supplement.py

echo "stage=generate_targeted_h5 start=$(date -Is)" | tee -a "${RUN_ROOT}/timeline.log"
"${ROOT}/.venv/bin/python" -u scripts/world_model/generate_cosmos3_fix3_hard_dynamic_teacher.py \
  --output-root "${H5_ROOT}" \
  --num-demos "${NUM_DEMOS}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --seed "${SEED}" \
  --scenario-sequence "${SCENARIO_SEQUENCE}" \
  --scenario-quotas "${SCENARIO_QUOTAS}" \
  --scenario-seed-bases "${SCENARIO_SEED_BASES}" \
  --post-motion-release-regrasp-scenarios "${POST_MOTION_RELEASE_REGRASP_SCENARIOS}" \
  --post-motion-release-regrasp-hold-steps "${POST_MOTION_RELEASE_REGRASP_HOLD_STEPS}" \
  --motion-min-m "${MOTION_MIN_M}" \
  --motion-max-m "${MOTION_MAX_M}" \
  --min-abs-y-motion-m "${MIN_ABS_Y_MOTION_M}" \
  --anti-self-insert-final-yz-min-m "${ANTI_SELF_INSERT_FINAL_YZ_MIN_M}" \
  --anti-self-insert-radius-multiplier "${ANTI_SELF_INSERT_RADIUS_MULTIPLIER}" \
  --anti-self-insert-exempt-scenarios "${ANTI_SELF_INSERT_EXEMPT_SCENARIOS}" \
  --initial-preinsert-retreat-extra-m "${INITIAL_PREINSERT_RETREAT_EXTRA_M}" \
  --initial-preinsert-line-yz-max-m "${INITIAL_PREINSERT_LINE_YZ_MAX_M}" \
  --source-kind "hard_dynamic_teacher_targeted_recovery_gap_20260614" \
  --save-reject-log-limit 2000 \
  --reject-log-every 25 \
  --val-fraction 0.1 \
  --overwrite \
  2>&1 | tee "${RUN_ROOT}/generate_targeted_h5.log"

echo "stage=render_targeted_rgb_review start=$(date -Is)" | tee -a "${RUN_ROOT}/timeline.log"
"${ROOT}/.venv/bin/python" -u scripts/world_model/render_cosmos3_maniskill_sft_dataset.py \
  --paths-file "${H5_ROOT}/fix3_h5_paths.txt" \
  --output-root "${RGB_ROOT}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  --frame-stride 1 \
  --max-frames 0 \
  --val-fraction 0.1 \
  --sheet-limit "${SHEET_LIMIT}" \
  --sheet-frames "${SHEET_FRAMES}" \
  --sheet-thumb-width "${SHEET_THUMB_WIDTH}" \
  --overwrite \
  2>&1 | tee "${RUN_ROOT}/render_targeted_rgb_review.log"

echo "stage=inspect_targeted_supplement start=$(date -Is)" | tee -a "${RUN_ROOT}/timeline.log"
"${ROOT}/.venv/bin/python" -u scripts/world_model/inspect_cosmos3_targeted_recovery_supplement.py \
  --h5-root "${H5_ROOT}" \
  --rgb-root "${RGB_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --gap-manifest "${GAP_MANIFEST}" \
  --expected-count "${NUM_DEMOS}" \
  --expected-scenario-quotas "${SCENARIO_QUOTAS}" \
  --expected-source-kind "hard_dynamic_teacher_targeted_recovery_gap_20260614" \
  --expected-width "${WIDTH}" \
  --expected-height "${HEIGHT}" \
  --expected-fps "${FPS}" \
  --expected-frames 301 \
  --expected-actions 300 \
  --min-review-sheets "${SHEET_LIMIT}" \
  --require-regrasp-scenarios "${POST_MOTION_RELEASE_REGRASP_SCENARIOS}" \
  --output-json "${RUN_ROOT}/targeted_recovery_supplement_inspection.json" \
  --output-md "${RUN_ROOT}/targeted_recovery_supplement_inspection.md" \
  2>&1 | tee "${RUN_ROOT}/inspect_targeted_supplement.log"

{
  echo "timestamp=$(date -Is)"
  echo "h5_paths=${H5_ROOT}/fix3_h5_paths.txt"
  echo "h5_manifest=${H5_ROOT}/manifest.json"
  echo "h5_audit=${H5_ROOT}/source_audit.json"
  echo "rgb_manifest=${RGB_ROOT}/manifest.json"
  echo "inspection_json=${RUN_ROOT}/targeted_recovery_supplement_inspection.json"
  echo "inspection_md=${RUN_ROOT}/targeted_recovery_supplement_inspection.md"
  echo "review_sheets=${RGB_ROOT}/review_sheets"
  echo "next_gate=agent must open rendered review sheets and verify real final insertion before any merge/export/SFT."
} | tee "${RUN_ROOT}/completion_manifest.txt"

echo "stage=complete end=$(date -Is)" | tee -a "${RUN_ROOT}/timeline.log"
