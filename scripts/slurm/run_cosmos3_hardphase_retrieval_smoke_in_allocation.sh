#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run hard-phase retrieval smoke only inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
HARD_MAX_ROWS="${HARD_MAX_ROWS:-64}"
HARD_SKIP_ROWS="${HARD_SKIP_ROWS:-0}"
RETRIEVAL_K="${RETRIEVAL_K:-8}"
RETRIEVAL_RESIDUAL_SCALES="${RETRIEVAL_RESIDUAL_SCALES:-0.5,1.0,1.5}"
MODEL_CANDIDATE_SAMPLES="${MODEL_CANDIDATE_SAMPLES:-48}"
CANDIDATE_SHORT_PREFIX_STEPS="${CANDIDATE_SHORT_PREFIX_STEPS:-}"
DP_ROLLOUT_CONTINUABILITY_HORIZON="${DP_ROLLOUT_CONTINUABILITY_HORIZON:-8}"
DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS="${DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS:-4}"
CONTACT_STABLE_MIN_REL_X="${CONTACT_STABLE_MIN_REL_X:--0.06}"
CONTACT_STABLE_MAX_REL_X="${CONTACT_STABLE_MAX_REL_X:-0.03}"
CONTACT_STABLE_MAX_ABS_Y="${CONTACT_STABLE_MAX_ABS_Y:-0.018}"
CONTACT_STABLE_MAX_ABS_Z="${CONTACT_STABLE_MAX_ABS_Z:-0.012}"
SHARD_CLAIM_ROOT="${SHARD_CLAIM_ROOT:-experiments/world_model_task_rebinding/cosmos3/hardphase_retrieval_shard_claims_20260618}"

OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval${HARD_MAX_ROWS}_skip${HARD_SKIP_ROWS}_k${RETRIEVAL_K}_${STAMP}_alloc${SLURM_JOB_ID}}"
HEADROOM_ROOT="${HEADROOM_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_retrieval${HARD_MAX_ROWS}_skip${HARD_SKIP_ROWS}_k${RETRIEVAL_K}_${STAMP}_alloc${SLURM_JOB_ID}}"
FILTER_ROOT="${FILTER_ROOT:-experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_hardphase_retrieval${HARD_MAX_ROWS}_skip${HARD_SKIP_ROWS}_${STAMP}_alloc${SLURM_JOB_ID}}"
RUN_SCORER_AFTER_HEADROOM_GATE="${RUN_SCORER_AFTER_HEADROOM_GATE:-true}"
SCORER_MAX_STEPS="${SCORER_MAX_STEPS:-500}"
SCORER_ROOT="${SCORER_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval${HARD_MAX_ROWS}_skip${HARD_SKIP_ROWS}_k${RETRIEVAL_K}_rank1_canddesc_smoke${SCORER_MAX_STEPS}_${STAMP}_alloc${SLURM_JOB_ID}}"
MARGIN_ROOT="${MARGIN_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval${HARD_MAX_ROWS}_skip${HARD_SKIP_ROWS}_k${RETRIEVAL_K}_margin_eval_${STAMP}_alloc${SLURM_JOB_ID}}"
MIN_ORACLE_SUCCESS_GAIN="${MIN_ORACLE_SUCCESS_GAIN:-2}"
MIN_ORACLE_SUCCESS_COUNT="${MIN_ORACLE_SUCCESS_COUNT:-4}"
MAX_MEAN_ORACLE_MINUS_DP_ERROR="${MAX_MEAN_ORACLE_MINUS_DP_ERROR:--0.025}"
MIN_FAR_ORACLE_SUCCESS="${MIN_FAR_ORACLE_SUCCESS:-1}"
MIN_RETRIEVAL_ORACLE_COUNT="${MIN_RETRIEVAL_ORACLE_COUNT:-1}"
MIN_RETRIEVAL_SUCCESS_GROUPS="${MIN_RETRIEVAL_SUCCESS_GROUPS:-1}"
ALLOW_PROGRESS_HEADROOM_SCORER_GATE="${ALLOW_PROGRESS_HEADROOM_SCORER_GATE:-true}"
MIN_MEANINGFUL_IMPROVEMENT_FRACTION="${MIN_MEANINGFUL_IMPROVEMENT_FRACTION:-0.25}"
MIN_LARGE_IMPROVEMENT_FRACTION="${MIN_LARGE_IMPROVEMENT_FRACTION:-0.05}"
MAX_FAR_MEAN_ORACLE_MINUS_DP_ERROR="${MAX_FAR_MEAN_ORACLE_MINUS_DP_ERROR:-${MAX_MEAN_ORACLE_MINUS_DP_ERROR}}"
SCORER_MIN_EVAL_GROUPS="${SCORER_MIN_EVAL_GROUPS:-8}"
SCORER_MIN_STEPS="${SCORER_MIN_STEPS:-0}"
SCORER_MIN_WALL_SECONDS="${SCORER_MIN_WALL_SECONDS:-0}"
SCORER_MAX_WALL_SECONDS="${SCORER_MAX_WALL_SECONDS:-0}"
SCORER_FORMAL_MIN_GPUS="${SCORER_FORMAL_MIN_GPUS:-1}"
SCORER_RANK_LOSS_WEIGHT="${SCORER_RANK_LOSS_WEIGHT:-0.0}"
SCORER_PROGRESS_LOSS_WEIGHT="${SCORER_PROGRESS_LOSS_WEIGHT:-0.25}"
SCORER_SCORE_SUCCESS_WEIGHT="${SCORER_SCORE_SUCCESS_WEIGHT:-0.5}"
SCORER_SCORE_INSERTED_WEIGHT="${SCORER_SCORE_INSERTED_WEIGHT:-0.25}"
SCORER_SCORE_GRASPED_WEIGHT="${SCORER_SCORE_GRASPED_WEIGHT:-0.1}"
SCORER_SCORE_PROGRESS_WEIGHT="${SCORER_SCORE_PROGRESS_WEIGHT:-0.25}"
SCORER_SCORE_PROGRESS_DELTA_WEIGHT="${SCORER_SCORE_PROGRESS_DELTA_WEIGHT:-0.15}"
SCORER_SCORE_CONTINUABLE_WEIGHT="${SCORER_SCORE_CONTINUABLE_WEIGHT:-0.25}"
SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS="${SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS:-0,0,0}"
SCORER_SCORE_STATE_TARGET="${SCORER_SCORE_STATE_TARGET:-0,0,0}"
SCORER_MIN_SELECTED_PROGRESS_DELTA_IMPROVEMENT="${SCORER_MIN_SELECTED_PROGRESS_DELTA_IMPROVEMENT:-0.0}"
VAL_FRACTION="${VAL_FRACTION:-0.25}"
SEED="${SEED:-20260618}"

".venv/bin/python" -m py_compile scripts/world_model/export_cosmos3_candidate_outcome_labels.py
".venv/bin/python" -m py_compile scripts/world_model/summarize_cosmos3_candidate_outcome_headroom.py
if [[ "${RUN_SCORER_AFTER_HEADROOM_GATE}" == "true" ]]; then
  ".venv/bin/python" -m py_compile scripts/world_model/train_cosmos3_candidate_outcome_scorer.py
  ".venv/bin/python" -m py_compile scripts/world_model/eval_cosmos3_candidate_outcome_scorer_margins.py
fi
bash -n scripts/slurm/run_cosmos3_hardphase_exploratory_candidate_replay_in_allocation.sh

INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES=true \
RETRIEVAL_K="${RETRIEVAL_K}" \
RETRIEVAL_RESIDUAL_SCALES="${RETRIEVAL_RESIDUAL_SCALES}" \
MODEL_CANDIDATE_SAMPLES="${MODEL_CANDIDATE_SAMPLES}" \
CANDIDATE_SHORT_PREFIX_STEPS="${CANDIDATE_SHORT_PREFIX_STEPS}" \
DP_ROLLOUT_CONTINUABILITY_HORIZON="${DP_ROLLOUT_CONTINUABILITY_HORIZON}" \
DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS="${DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS}" \
CONTACT_STABLE_MIN_REL_X="${CONTACT_STABLE_MIN_REL_X}" \
CONTACT_STABLE_MAX_REL_X="${CONTACT_STABLE_MAX_REL_X}" \
CONTACT_STABLE_MAX_ABS_Y="${CONTACT_STABLE_MAX_ABS_Y}" \
CONTACT_STABLE_MAX_ABS_Z="${CONTACT_STABLE_MAX_ABS_Z}" \
SHARD_CLAIM_ROOT="${SHARD_CLAIM_ROOT}" \
HARD_MAX_ROWS="${HARD_MAX_ROWS}" \
HARD_SKIP_ROWS="${HARD_SKIP_ROWS}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
HEADROOM_ROOT="${HEADROOM_ROOT}" \
FILTER_ROOT="${FILTER_ROOT}" \
bash scripts/slurm/run_cosmos3_hardphase_exploratory_candidate_replay_in_allocation.sh

if [[ -s "${HEADROOM_ROOT}/shard_claim_skipped.txt" ]]; then
  cat > "${HEADROOM_ROOT}/retrieval_runner_skipped_after_claim.txt" <<EOF
skipped=true
reason=shard_claim_skipped
date=$(date --iso-8601=seconds)
headroom_root=${HEADROOM_ROOT}
claim_note=${HEADROOM_ROOT}/shard_claim_skipped.txt
EOF
  exit 0
fi

GATE_SUMMARY="${HEADROOM_ROOT}/retrieval_scorer_gate_summary.json"
".venv/bin/python" - "${HEADROOM_ROOT}/candidate_outcome_headroom_summary.json" "${GATE_SUMMARY}" \
  "${MIN_ORACLE_SUCCESS_GAIN}" "${MIN_ORACLE_SUCCESS_COUNT}" \
  "${MAX_MEAN_ORACLE_MINUS_DP_ERROR}" "${MIN_FAR_ORACLE_SUCCESS}" \
  "${MIN_RETRIEVAL_ORACLE_COUNT}" "${MIN_RETRIEVAL_SUCCESS_GROUPS}" \
  "${ALLOW_PROGRESS_HEADROOM_SCORER_GATE}" "${MIN_MEANINGFUL_IMPROVEMENT_FRACTION}" \
  "${MIN_LARGE_IMPROVEMENT_FRACTION}" "${MAX_FAR_MEAN_ORACLE_MINUS_DP_ERROR}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
gate_path = Path(sys.argv[2])
min_gain = int(sys.argv[3])
min_success_count = int(sys.argv[4])
max_mean_delta = float(sys.argv[5])
min_far_success = int(sys.argv[6])
min_retrieval_oracle = int(sys.argv[7])
min_retrieval_success_groups = int(sys.argv[8])
allow_progress_gate = sys.argv[9].strip().lower() in {"1", "true", "yes", "y"}
min_meaningful_fraction = float(sys.argv[10])
min_large_fraction = float(sys.argv[11])
max_far_mean_delta = float(sys.argv[12])

summary = json.loads(summary_path.read_text())
overall = summary.get("overall") or {}
by_phase = summary.get("by_current_phase") or {}
far = by_phase.get("far") or {}
oracle_families = overall.get("oracle_candidate_family_counts") or {}
success_family_groups = overall.get("success_candidate_family_group_counts") or {}

dp_success = int(overall.get("dp_success_count") or 0)
oracle_success = int(overall.get("oracle_success_count") or 0)
success_gain = oracle_success - dp_success
mean_delta = float(overall.get("mean_oracle_minus_dp_error") or 0.0)
num_groups = int(summary.get("num_uuid_groups") or overall.get("num_groups") or 0)
meaningful_improvement_count = int(overall.get("meaningful_improvement_count") or 0)
large_improvement_count = int(overall.get("large_improvement_count") or 0)
meaningful_improvement_fraction = meaningful_improvement_count / max(num_groups, 1)
large_improvement_fraction = large_improvement_count / max(num_groups, 1)
far_groups = int(far.get("num_groups") or 0)
far_oracle_success = int(far.get("oracle_success_count") or 0)
far_mean_delta = float(far.get("mean_oracle_minus_dp_error") or 0.0)
far_gate_ok = far_groups == 0 or far_oracle_success >= min_far_success
far_progress_gate_ok = far_groups == 0 or far_mean_delta <= max_far_mean_delta
retrieval_oracle_count = int(oracle_families.get("retrieval_success_residual") or 0)
retrieval_success_groups = int(success_family_groups.get("retrieval_success_residual") or 0)
retrieval_contribution_ok = bool(
    retrieval_oracle_count >= min_retrieval_oracle
    or retrieval_success_groups >= min_retrieval_success_groups
)

terminal_success_gate = bool(
    success_gain >= min_gain
    and oracle_success >= min_success_count
    and mean_delta <= max_mean_delta
    and far_gate_ok
    and retrieval_contribution_ok
)
progress_headroom_gate = bool(
    allow_progress_gate
    and mean_delta <= max_mean_delta
    and meaningful_improvement_fraction >= min_meaningful_fraction
    and large_improvement_fraction >= min_large_fraction
    and far_progress_gate_ok
    and retrieval_contribution_ok
)
passed = bool(terminal_success_gate or progress_headroom_gate)
if terminal_success_gate:
    plain_reason = "Retrieval produced enough successful hard-phase action candidates to justify a short scorer smoke."
elif progress_headroom_gate:
    plain_reason = (
        "Retrieval produced enough receding-chunk progress/contact headroom to train the progress/value scorer, "
        "even though one chunk is not yet a terminal success."
    )
else:
    plain_reason = (
        "Retrieval did not yet create enough real hard-phase candidate headroom; "
        "skip scorer and improve the candidate source."
    )
gate = {
    "schema": "cosmos3_hardphase_retrieval_scorer_gate_v1",
    "headroom_summary": str(summary_path),
    "passed": passed,
    "terminal_success_gate_passed": terminal_success_gate,
    "progress_headroom_gate_passed": progress_headroom_gate,
    "thresholds": {
        "min_oracle_success_gain": min_gain,
        "min_oracle_success_count": min_success_count,
        "max_mean_oracle_minus_dp_error": max_mean_delta,
        "min_far_oracle_success": min_far_success,
        "min_retrieval_oracle_count": min_retrieval_oracle,
        "min_retrieval_success_groups": min_retrieval_success_groups,
        "allow_progress_headroom_scorer_gate": allow_progress_gate,
        "min_meaningful_improvement_fraction": min_meaningful_fraction,
        "min_large_improvement_fraction": min_large_fraction,
        "max_far_mean_oracle_minus_dp_error": max_far_mean_delta,
    },
    "observed": {
        "num_uuid_groups": num_groups,
        "dp_success_count": dp_success,
        "oracle_success_count": oracle_success,
        "oracle_success_gain": success_gain,
        "mean_oracle_minus_dp_error": mean_delta,
        "meaningful_improvement_count": meaningful_improvement_count,
        "meaningful_improvement_fraction": meaningful_improvement_fraction,
        "large_improvement_count": large_improvement_count,
        "large_improvement_fraction": large_improvement_fraction,
        "far_num_groups": far_groups,
        "far_oracle_success_count": far_oracle_success,
        "far_mean_oracle_minus_dp_error": far_mean_delta,
        "retrieval_oracle_count": retrieval_oracle_count,
        "retrieval_success_groups": retrieval_success_groups,
    },
    "plain_reason": plain_reason,
    "boundary": (
        "This gate decides whether a short outcome-scorer smoke is worth running. "
        "It is not formal training evidence and not live controller evidence. "
        "The progress gate exists because receding control can need several short chunks; "
        "terminal success in a single hard-phase chunk is useful but not the only aligned scorer signal."
    ),
}
gate_path.parent.mkdir(parents=True, exist_ok=True)
gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
print(json.dumps(gate, sort_keys=True))
PY

if [[ "${RUN_SCORER_AFTER_HEADROOM_GATE}" != "true" ]]; then
  cat > "${HEADROOM_ROOT}/retrieval_scorer_skipped.txt" <<EOF
skipped=true
reason=RUN_SCORER_AFTER_HEADROOM_GATE is not true
gate_summary=${GATE_SUMMARY}
EOF
  exit 0
fi

if ! ".venv/bin/python" - "${GATE_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path
raise SystemExit(0 if json.loads(Path(sys.argv[1]).read_text()).get("passed") else 1)
PY
then
  cat > "${HEADROOM_ROOT}/retrieval_scorer_skipped.txt" <<EOF
skipped=true
reason=headroom_gate_failed
gate_summary=${GATE_SUMMARY}
EOF
  exit 0
fi

".venv/bin/python" scripts/world_model/train_cosmos3_candidate_outcome_scorer.py \
  --contact-executor-jsonl "${FILTER_ROOT}/contact_executor_dataset_file.jsonl" \
  --outcome-jsonl "${OUTPUT_ROOT}/candidate_outcome_labels.jsonl" \
  --output-root "${SCORER_ROOT}" \
  --val-fraction "${VAL_FRACTION}" \
  --max-steps "${SCORER_MAX_STEPS}" \
  --min-steps "${SCORER_MIN_STEPS:-0}" \
  --min-wall-seconds "${SCORER_MIN_WALL_SECONDS:-0}" \
  --max-wall-seconds "${SCORER_MAX_WALL_SECONDS:-0}" \
  --formal-min-gpus "${SCORER_FORMAL_MIN_GPUS:-1}" \
  --eval-every-steps 25 \
  --save-every-steps 100 \
  --batch-size 256 \
  --hidden-dim 512 \
  --num-layers 3 \
  --progress-loss-weight "${SCORER_PROGRESS_LOSS_WEIGHT}" \
  --rank-loss-weight "${SCORER_RANK_LOSS_WEIGHT}" \
  --rank-loss-temperature 0.05 \
  --score-success-weight "${SCORER_SCORE_SUCCESS_WEIGHT}" \
  --score-inserted-weight "${SCORER_SCORE_INSERTED_WEIGHT}" \
  --score-grasped-weight "${SCORER_SCORE_GRASPED_WEIGHT}" \
  --score-progress-weight "${SCORER_SCORE_PROGRESS_WEIGHT}" \
  --score-progress-delta-weight "${SCORER_SCORE_PROGRESS_DELTA_WEIGHT}" \
  --score-continuable-weight "${SCORER_SCORE_CONTINUABLE_WEIGHT}" \
  --score-state-abs-axis-weights "${SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS}" \
  --score-state-target "${SCORER_SCORE_STATE_TARGET}" \
  --min-selected-error-improvement 0.005 \
  --min-selected-progress-delta-improvement "${SCORER_MIN_SELECTED_PROGRESS_DELTA_IMPROVEMENT}" \
  --min-non-dp-selected-fraction 0.1 \
  --min-eval-groups-for-gate "${SCORER_MIN_EVAL_GROUPS}" \
  --require-cuda \
  --seed "${SEED}"

SCORER_MARGIN_CHECKPOINT="${SCORER_ROOT}/checkpoint_best_offline.pt"
if [[ -s "${SCORER_ROOT}/checkpoint_best_gate.pt" ]]; then
  SCORER_MARGIN_CHECKPOINT="${SCORER_ROOT}/checkpoint_best_gate.pt"
fi

".venv/bin/python" scripts/world_model/eval_cosmos3_candidate_outcome_scorer_margins.py \
  --contact-executor-jsonl "${FILTER_ROOT}/contact_executor_dataset_file.jsonl" \
  --outcome-jsonl "${OUTPUT_ROOT}/candidate_outcome_labels.jsonl" \
  --checkpoint "${SCORER_MARGIN_CHECKPOINT}" \
  --output-root "${MARGIN_ROOT}" \
  --val-fraction "${VAL_FRACTION}" \
  --min-selected-error-improvement 0.005 \
  --min-selected-progress-delta-improvement "${SCORER_MIN_SELECTED_PROGRESS_DELTA_IMPROVEMENT}" \
  --min-non-dp-selected-fraction 0.1 \
  --require-cuda \
  --seed "${SEED}"

cat > "${SCORER_ROOT}/retrieval_gate_runner_done.txt" <<EOF
done=true
date=$(date --iso-8601=seconds)
headroom_root=${HEADROOM_ROOT}
gate_summary=${GATE_SUMMARY}
scorer_root=${SCORER_ROOT}
margin_root=${MARGIN_ROOT}
margin_checkpoint=${SCORER_MARGIN_CHECKPOINT}
boundary=Short scorer smoke after retrieval candidate-headroom gate. Not formal training evidence and not live controller evidence.
EOF
