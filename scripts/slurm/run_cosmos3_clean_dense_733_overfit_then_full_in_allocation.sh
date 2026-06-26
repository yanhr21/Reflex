#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this only inside a compute-node srun step from the tmux-held salloc shell.
example=srun --overlap --ntasks=1 --gres=gpu:4 --cpus-per-task=32 --mem=220G bash scripts/slurm/run_cosmos3_clean_dense_733_overfit_then_full_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/clean_dense_733_overfit_then_full_${STAMP}}"
FULL_CONDITION_ROOT="${FULL_CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_rgb_300step_${STAMP}}"
FULL_PREFLIGHT_ROOT="${FULL_PREFLIGHT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_preflight_${STAMP}}"
OVERFIT_CONDITION_ROOT="${OVERFIT_CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_overfit2_rgb_300step_${STAMP}}"
OVERFIT_PREFLIGHT_ROOT="${OVERFIT_PREFLIGHT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_preflight_${STAMP}}"
OVERFIT_SFT_ROOT="${OVERFIT_SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_rgb_300step_fix1recipe_2gpu_${STAMP}}"
FULL_SFT_ROOT="${FULL_SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_rgb_300step_fix1recipe_4gpu_${STAMP}}"

OVERFIT_CHECKPOINT_ITER="${OVERFIT_CHECKPOINT_ITER:-100}"
OVERFIT_EVAL_ROOT="${OVERFIT_EVAL_ROOT:-${OVERFIT_SFT_ROOT}/eval_full_episode_wam_iter_000000100}"

mkdir -p "${RUN_ROOT}"

write_run_manifest() {
  cat >"${RUN_ROOT}/run_manifest.txt" <<EOF
timestamp=$(date -Is)
job_id=${SLURM_JOB_ID}
step_id=${SLURM_STEP_ID}
host=$(hostname)
stamp=${STAMP}
full_condition_root=${FULL_CONDITION_ROOT}
full_preflight_root=${FULL_PREFLIGHT_ROOT}
overfit_condition_root=${OVERFIT_CONDITION_ROOT}
overfit_preflight_root=${OVERFIT_PREFLIGHT_ROOT}
overfit_sft_root=${OVERFIT_SFT_ROOT}
overfit_checkpoint_iter=${OVERFIT_CHECKPOINT_ITER}
overfit_eval_root=${OVERFIT_EVAL_ROOT}
full_sft_root=${FULL_SFT_ROOT}
resource_boundary=tmux-held interactive allocation; no sbatch; compute-node srun step only
visual_boundary=733 source review sheets 0000 and 0001 were inspected before launch and showed normal camera/framing plus visible final insertion.
training_boundary=short overfit uses 2 GPUs and 100 steps; full training uses 4 GPUs and is the only formal training-evidence run.
EOF
}

check_ready_summary() {
  local summary="$1"
  local condition="$2"
  local out="$3"
  "${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/check_cosmos3_clean_dense_preflight_summary_ready.py" \
    --summary-json "${summary}" \
    --condition-root "${condition}" \
    --output-json "${out}"
}

check_overfit_passed() {
  local val_summary="${OVERFIT_SFT_ROOT}/val_loss_summary.json"
  local eval_summary="${OVERFIT_EVAL_ROOT}/eval_artifact_inspection.json"
  "${ROOT}/.venv/bin/python" - "${val_summary}" "${eval_summary}" "${RUN_ROOT}/overfit_pass_gate.json" <<'PY'
import json
import math
import sys
from pathlib import Path

val_path = Path(sys.argv[1])
eval_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
reasons = []
if not val_path.is_file():
    reasons.append("missing_val_loss_summary")
    val = {}
else:
    val = json.loads(val_path.read_text())
latest = val.get("latest_val_loss")
if latest is None or not math.isfinite(float(latest)):
    reasons.append("latest_val_loss_not_finite")
if not eval_path.is_file():
    reasons.append("missing_eval_artifact_inspection")
    ev = {}
else:
    ev = json.loads(eval_path.read_text())
if ev.get("strict_eval_artifacts_ok") is not True:
    reasons.append("strict_eval_artifacts_not_ok")
if int(ev.get("num_samples") or 0) < 2:
    reasons.append("overfit_eval_has_fewer_than_two_samples")
report = {
    "ok": not reasons,
    "reasons": reasons,
    "val_loss_summary": str(val_path),
    "latest_val_loss": latest,
    "eval_artifact_inspection": str(eval_path),
    "strict_eval_artifacts_ok": ev.get("strict_eval_artifacts_ok"),
    "num_samples": ev.get("num_samples"),
    "review_sheets": [
        item.get("review_sheet")
        for item in ev.get("samples", [])
        if isinstance(item, dict) and item.get("review_sheet")
    ],
    "boundary": "This gate allows full training startup only after short overfit completes and strict generated 301/300 artifacts pass. Direct visual sheet review is still required for final interpretation.",
}
out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
if reasons:
    raise SystemExit(72)
PY
}

main() {
  cd "${ROOT}"
  write_run_manifest

  echo "stage=full_clean_dense_preflight start=$(date -Is)" | tee -a "${RUN_ROOT}/timeline.log"
  ALLOW_CLEAN_DENSE_PREFLIGHT=true \
  CONDITION_ROOT="${FULL_CONDITION_ROOT}" \
  OUTPUT_ROOT="${FULL_PREFLIGHT_ROOT}" \
    bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_preflight_in_allocation.sh" \
    2>&1 | tee "${RUN_ROOT}/full_clean_dense_preflight.log"
  check_ready_summary \
    "${FULL_PREFLIGHT_ROOT}/clean_dense_preflight_summary.json" \
    "${FULL_CONDITION_ROOT}" \
    "${RUN_ROOT}/full_clean_dense_ready_gate.json"

  echo "stage=overfit2_preflight start=$(date -Is)" | tee -a "${RUN_ROOT}/timeline.log"
  ALLOW_CLEAN_DENSE_PREFLIGHT=true \
  CONDITION_ROOT="${OVERFIT_CONDITION_ROOT}" \
  OUTPUT_ROOT="${OVERFIT_PREFLIGHT_ROOT}" \
    bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_overfit2_preflight_in_allocation.sh" \
    2>&1 | tee "${RUN_ROOT}/overfit2_preflight.log"
  check_ready_summary \
    "${OVERFIT_PREFLIGHT_ROOT}/clean_dense_preflight_summary.json" \
    "${OVERFIT_CONDITION_ROOT}" \
    "${RUN_ROOT}/overfit2_ready_gate.json"

  echo "stage=overfit_sft_100_steps start=$(date -Is)" | tee -a "${RUN_ROOT}/timeline.log"
  ALLOW_CLEAN_DENSE_OVERFIT_SFT=true \
  CONDITION_ROOT="${OVERFIT_CONDITION_ROOT}" \
  CLEAN_DENSE_PREFLIGHT_SUMMARY="${OVERFIT_PREFLIGHT_ROOT}/clean_dense_preflight_summary.json" \
  OUTPUT_ROOT="${OVERFIT_SFT_ROOT}" \
  NPROC_PER_NODE=2 \
  DATA_PARALLEL_SHARD_DEGREE=2 \
  MAX_ITER=100 \
  SAVE_ITER=50 \
  VALIDATION_ITER=50 \
  MAX_VAL_ITER=2 \
    bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_overfit2_fix1recipe_in_allocation.sh" \
    2>&1 | tee "${RUN_ROOT}/overfit_sft_100_steps.log"

  echo "stage=overfit_eval_iter100 start=$(date -Is)" | tee -a "${RUN_ROOT}/timeline.log"
  SFT_ROOT="${OVERFIT_SFT_ROOT}" \
  CONDITION_ROOT="${OVERFIT_CONDITION_ROOT}" \
  CHECKPOINT_ITER="${OVERFIT_CHECKPOINT_ITER}" \
  EVAL_ROOT="${OVERFIT_EVAL_ROOT}" \
  N_EVAL_SAMPLES=2 \
  NPROC_PER_NODE=1 \
  INFERENCE_NUM_STEPS=30 \
    bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_eval_in_allocation.sh" \
    2>&1 | tee "${RUN_ROOT}/overfit_eval_iter100.log"
  check_overfit_passed

  echo "stage=full_clean_dense_sft start=$(date -Is)" | tee -a "${RUN_ROOT}/timeline.log"
  ALLOW_CLEAN_DENSE_FULL_SFT=true \
  CONDITION_ROOT="${FULL_CONDITION_ROOT}" \
  CLEAN_DENSE_PREFLIGHT_SUMMARY="${FULL_PREFLIGHT_ROOT}/clean_dense_preflight_summary.json" \
  OUTPUT_ROOT="${FULL_SFT_ROOT}" \
  NPROC_PER_NODE=4 \
  DATA_PARALLEL_SHARD_DEGREE=4 \
    bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_full_fix1recipe_in_allocation.sh" \
    2>&1 | tee "${RUN_ROOT}/full_clean_dense_sft.log"
}

main "$@"
