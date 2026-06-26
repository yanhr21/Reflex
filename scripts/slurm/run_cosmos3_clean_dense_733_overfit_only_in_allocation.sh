#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this only inside a compute-node srun step from the tmux-held Slurm allocation.
example=srun --overlap --ntasks=1 --gres=gpu:2 --cpus-per-task=24 --mem=160G bash scripts/slurm/run_cosmos3_clean_dense_733_overfit_only_in_allocation.sh
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/clean_dense_733_overfit_only_${STAMP}}"
SOURCE_CONDITION_ROOT="${SOURCE_CONDITION_ROOT:-}"
OVERFIT_SOURCE_SPLIT="${OVERFIT_SOURCE_SPLIT:-val}"
OVERFIT_CONDITION_ROOT="${OVERFIT_CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_overfit2_rgb_300step_${STAMP}}"
OVERFIT_PREFLIGHT_ROOT="${OVERFIT_PREFLIGHT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_preflight_${STAMP}}"
OVERFIT_SFT_ROOT="${OVERFIT_SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_rgb_300step_fix1recipe_2gpu_${STAMP}}"
OVERFIT_CHECKPOINT_ITER="${OVERFIT_CHECKPOINT_ITER:-100}"
OVERFIT_EVAL_ROOT="${OVERFIT_EVAL_ROOT:-${OVERFIT_SFT_ROOT}/eval_full_episode_wam_iter_$(printf '%09d' "${OVERFIT_CHECKPOINT_ITER}")}"

mkdir -p "${RUN_ROOT}"

write_run_manifest() {
  cat >"${RUN_ROOT}/run_manifest.txt" <<EOF
timestamp=$(date -Is)
job_id=${SLURM_JOB_ID}
step_id=${SLURM_STEP_ID}
host=$(hostname)
stamp=${STAMP}
overfit_condition_root=${OVERFIT_CONDITION_ROOT}
source_condition_root=${SOURCE_CONDITION_ROOT}
overfit_source_split=${OVERFIT_SOURCE_SPLIT}
overfit_preflight_root=${OVERFIT_PREFLIGHT_ROOT}
overfit_sft_root=${OVERFIT_SFT_ROOT}
overfit_checkpoint_iter=${OVERFIT_CHECKPOINT_ITER}
overfit_eval_root=${OVERFIT_EVAL_ROOT}
resource_boundary=tmux-held interactive allocation; no sbatch; compute-node srun step only
training_boundary=short overfit uses 2 GPUs and 100 steps by default; it is pipeline sanity only, not method evidence.
full_training_boundary=full 4-GPU SFT remains blocked until the full 733 live-query coverage gate passes or the missing failure/recovery data are added and re-audited.
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
    "boundary": "This is only a two-source overfit sanity gate. Passing it does not unblock full SFT while full live-query coverage is still undercovered.",
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

  if [[ -z "${SOURCE_CONDITION_ROOT}" ]]; then
    cat >&2 <<'EOF'
refusing_overfit_without_source_condition_root=true
reason=SOURCE_CONDITION_ROOT must point at a full clean/dense condition root. The overfit sanity root is sampled from that root so train and val can contain the same two valid rows.
EOF
    exit 47
  fi
  if [[ ! -s "${SOURCE_CONDITION_ROOT}/manifest.json" ]]; then
    cat >&2 <<EOF
refusing_overfit_missing_source_condition_root=true
source_condition_root=${SOURCE_CONDITION_ROOT}
reason=SOURCE_CONDITION_ROOT/manifest.json is missing.
EOF
    exit 48
  fi

  echo "stage=build_two_sample_overfit_root start=$(date -Is)" | tee -a "${RUN_ROOT}/timeline.log"
  "${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/build_cosmos3_two_sample_overfit_condition_root.py" \
    --source-condition-root "${SOURCE_CONDITION_ROOT}" \
    --output-root "${OVERFIT_CONDITION_ROOT}" \
    --split "${OVERFIT_SOURCE_SPLIT}" \
    2>&1 | tee "${RUN_ROOT}/build_two_sample_overfit_root.log"

  echo "stage=overfit2_preflight start=$(date -Is)" | tee -a "${RUN_ROOT}/timeline.log"
  SOURCE_DATASET_ROOT="${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612" \
  CONDITION_ROOT="${OVERFIT_CONDITION_ROOT}" \
  OUTPUT_ROOT="${OVERFIT_PREFLIGHT_ROOT}" \
  EXPECTED_SOURCE_EPISODES=2 \
  MAX_RECORDS=0 \
  FORCE_EXPORT=false \
  RUN_SFT=false \
  PREFIX_ROLE_SOURCE=physical_mode \
  DENSE_RECEDING_PREFIX_STRIDE=8 \
  RUN_LIVE_QUERY_COVERAGE_AUDIT=false \
    bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh" \
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
  NPROC_PER_NODE="${NPROC_PER_NODE:-2}" \
  DATA_PARALLEL_SHARD_DEGREE="${DATA_PARALLEL_SHARD_DEGREE:-2}" \
  MAX_ITER="${MAX_ITER:-100}" \
  SAVE_ITER="${SAVE_ITER:-50}" \
  VALIDATION_ITER="${VALIDATION_ITER:-50}" \
  MAX_VAL_ITER="${MAX_VAL_ITER:-2}" \
    bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_overfit2_fix1recipe_in_allocation.sh" \
    2>&1 | tee "${RUN_ROOT}/overfit_sft_100_steps.log"

  echo "stage=overfit_eval_iter${OVERFIT_CHECKPOINT_ITER} start=$(date -Is)" | tee -a "${RUN_ROOT}/timeline.log"
  SFT_ROOT="${OVERFIT_SFT_ROOT}" \
  CONDITION_ROOT="${OVERFIT_CONDITION_ROOT}" \
  CHECKPOINT_ITER="${OVERFIT_CHECKPOINT_ITER}" \
  EVAL_ROOT="${OVERFIT_EVAL_ROOT}" \
  N_EVAL_SAMPLES=2 \
  NPROC_PER_NODE=1 \
  INFERENCE_NUM_STEPS=30 \
    bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_eval_in_allocation.sh" \
    2>&1 | tee "${RUN_ROOT}/overfit_eval_iter${OVERFIT_CHECKPOINT_ITER}.log"
  check_overfit_passed
}

main "$@"
