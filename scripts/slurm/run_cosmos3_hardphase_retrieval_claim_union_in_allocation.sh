#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run retrieval claim-union summary only inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SHARD_CLAIM_ROOT="${SHARD_CLAIM_ROOT:-experiments/world_model_task_rebinding/cosmos3/hardphase_retrieval_shard_claims_20260618}"
CONTACT_EXECUTOR_JSONL="${CONTACT_EXECUTOR_JSONL:-experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_rollout24_predpath_train512_20260617_server23_alloc135069/contact_executor_dataset_file.jsonl}"
SUMMARY_ROOT="${SUMMARY_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_retrieval_claim_union_${STAMP}_alloc${SLURM_JOB_ID}}"
RUN_SCORER_AFTER_UNION_GATE="${RUN_SCORER_AFTER_UNION_GATE:-true}"
MIN_UNION_DONE_CLAIMS_FOR_SCORER="${MIN_UNION_DONE_CLAIMS_FOR_SCORER:-2}"
SCORER_MAX_STEPS="${SCORER_MAX_STEPS:-500}"
SCORER_ROOT="${SCORER_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval_claim_union_rank1_canddesc_smoke${SCORER_MAX_STEPS}_${STAMP}_alloc${SLURM_JOB_ID}}"
MARGIN_ROOT="${MARGIN_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval_claim_union_margin_eval_${STAMP}_alloc${SLURM_JOB_ID}}"
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
SCORER_BATCH_SIZE="${SCORER_BATCH_SIZE:-256}"
SCORER_HIDDEN_DIM="${SCORER_HIDDEN_DIM:-512}"
SCORER_NUM_LAYERS="${SCORER_NUM_LAYERS:-3}"
SCORER_DROPOUT="${SCORER_DROPOUT:-0.05}"
SCORER_LR="${SCORER_LR:-3e-4}"
SCORER_WEIGHT_DECAY="${SCORER_WEIGHT_DECAY:-1e-5}"
SCORER_EVAL_EVERY_STEPS="${SCORER_EVAL_EVERY_STEPS:-25}"
SCORER_SAVE_EVERY_STEPS="${SCORER_SAVE_EVERY_STEPS:-100}"
SCORER_BINARY_LOSS_WEIGHT="${SCORER_BINARY_LOSS_WEIGHT:-0.2}"
SCORER_RANK_LOSS_WEIGHT="${SCORER_RANK_LOSS_WEIGHT:-0.0}"
SCORER_PROGRESS_LOSS_WEIGHT="${SCORER_PROGRESS_LOSS_WEIGHT:-0.25}"
SCORER_SCORE_SUCCESS_WEIGHT="${SCORER_SCORE_SUCCESS_WEIGHT:-0.5}"
SCORER_SCORE_HANDOFF_SUCCESS_WEIGHT="${SCORER_SCORE_HANDOFF_SUCCESS_WEIGHT:-0.0}"
SCORER_SCORE_INSERTED_WEIGHT="${SCORER_SCORE_INSERTED_WEIGHT:-0.25}"
SCORER_SCORE_GRASPED_WEIGHT="${SCORER_SCORE_GRASPED_WEIGHT:-0.1}"
SCORER_SCORE_PROGRESS_WEIGHT="${SCORER_SCORE_PROGRESS_WEIGHT:-0.25}"
SCORER_SCORE_PROGRESS_DELTA_WEIGHT="${SCORER_SCORE_PROGRESS_DELTA_WEIGHT:-0.15}"
SCORER_SCORE_CONTINUABLE_WEIGHT="${SCORER_SCORE_CONTINUABLE_WEIGHT:-0.25}"
SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS="${SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS:-0,0,0}"
SCORER_SCORE_STATE_TARGET="${SCORER_SCORE_STATE_TARGET:-0,0,0}"
SCORER_ALLOWED_CANDIDATE_FAMILIES="${SCORER_ALLOWED_CANDIDATE_FAMILIES:-}"
SCORER_MIN_SELECTED_PROGRESS_DELTA_IMPROVEMENT="${SCORER_MIN_SELECTED_PROGRESS_DELTA_IMPROVEMENT:-0.0}"
SCORER_MIN_SELECTED_HANDOFF_SUCCESS_IMPROVEMENT="${SCORER_MIN_SELECTED_HANDOFF_SUCCESS_IMPROVEMENT:-0.0}"
SCORER_MAX_SELECTED_ERROR_DEGRADATION_FOR_HANDOFF_GATE="${SCORER_MAX_SELECTED_ERROR_DEGRADATION_FOR_HANDOFF_GATE:-0.0}"
SCORER_ALLOW_HANDOFF_ONLY_GATE="${SCORER_ALLOW_HANDOFF_ONLY_GATE:-false}"
VAL_FRACTION="${VAL_FRACTION:-0.25}"
SEED="${SEED:-20260618}"

SCORER_HANDOFF_GATE_FLAG="--no-allow-handoff-only-gate"
case "${SCORER_ALLOW_HANDOFF_ONLY_GATE}" in
  1|true|TRUE|yes|YES|y|Y) SCORER_HANDOFF_GATE_FLAG="--allow-handoff-only-gate" ;;
esac

mkdir -p "${SUMMARY_ROOT}"

OUTCOME_ARGS=()
DONE_CLAIMS=()
if [[ -d "${SHARD_CLAIM_ROOT}" ]]; then
  while IFS= read -r -d '' done_file; do
    output_root=""
    while IFS= read -r line; do
      case "${line}" in
        output_root=*) output_root="${line#output_root=}" ;;
      esac
    done < "${done_file}"
    if [[ -n "${output_root}" && -s "${output_root}/candidate_outcome_labels.jsonl" ]]; then
      OUTCOME_ARGS+=(--outcome-jsonl "${output_root}/candidate_outcome_labels.jsonl")
      DONE_CLAIMS+=("${done_file}")
    fi
  done < <(find "${SHARD_CLAIM_ROOT}" -mindepth 2 -maxdepth 2 -name done.txt -print0 | sort -z)
fi

ACTIVE_INPROGRESS_CLAIMS=()
STALE_INPROGRESS_CLAIMS=()
if [[ -d "${SHARD_CLAIM_ROOT}" ]]; then
  while IFS= read -r -d '' claim_file; do
    claim_dir="$(dirname "${claim_file}")"
    if [[ -s "${claim_dir}/done.txt" || -s "${claim_dir}/failed.txt" ]]; then
      continue
    fi
    owner_job="$(awk -F= '$1 == "slurm_job_id" {print substr($0, length($1) + 2); exit}' "${claim_file}" 2>/dev/null || true)"
    owner_state="$(squeue -h -j "${owner_job}" -o %T 2>/dev/null | head -n 1 || true)"
    case "${owner_state}" in
      PENDING|CONFIGURING|RUNNING|COMPLETING|SUSPENDED)
        ACTIVE_INPROGRESS_CLAIMS+=("${claim_dir}")
        ;;
      *)
        STALE_INPROGRESS_CLAIMS+=("${claim_dir}")
        ;;
    esac
  done < <(find "${SHARD_CLAIM_ROOT}" -mindepth 2 -maxdepth 2 -name claim.txt -print0 | sort -z)
fi

cat > "${SUMMARY_ROOT}/claim_union_manifest.txt" <<EOF
schema=cosmos3_hardphase_retrieval_claim_union_wrapper_v1
date=$(date --iso-8601=seconds)
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
shard_claim_root=${SHARD_CLAIM_ROOT}
contact_executor_jsonl=${CONTACT_EXECUTOR_JSONL}
summary_root=${SUMMARY_ROOT}
num_done_claims=${#DONE_CLAIMS[@]}
num_active_inprogress_claims=${#ACTIVE_INPROGRESS_CLAIMS[@]}
num_stale_inprogress_claims=${#STALE_INPROGRESS_CLAIMS[@]}
run_scorer_after_union_gate=${RUN_SCORER_AFTER_UNION_GATE}
min_union_done_claims_for_scorer=${MIN_UNION_DONE_CLAIMS_FOR_SCORER}
allow_progress_headroom_scorer_gate=${ALLOW_PROGRESS_HEADROOM_SCORER_GATE}
min_meaningful_improvement_fraction=${MIN_MEANINGFUL_IMPROVEMENT_FRACTION}
min_large_improvement_fraction=${MIN_LARGE_IMPROVEMENT_FRACTION}
max_far_mean_oracle_minus_dp_error=${MAX_FAR_MEAN_ORACLE_MINUS_DP_ERROR}
scorer_rank_loss_weight=${SCORER_RANK_LOSS_WEIGHT}
scorer_min_steps=${SCORER_MIN_STEPS}
scorer_min_wall_seconds=${SCORER_MIN_WALL_SECONDS}
scorer_max_wall_seconds=${SCORER_MAX_WALL_SECONDS}
scorer_formal_min_gpus=${SCORER_FORMAL_MIN_GPUS}
scorer_batch_size=${SCORER_BATCH_SIZE}
scorer_hidden_dim=${SCORER_HIDDEN_DIM}
scorer_num_layers=${SCORER_NUM_LAYERS}
scorer_dropout=${SCORER_DROPOUT}
scorer_lr=${SCORER_LR}
scorer_weight_decay=${SCORER_WEIGHT_DECAY}
scorer_eval_every_steps=${SCORER_EVAL_EVERY_STEPS}
scorer_save_every_steps=${SCORER_SAVE_EVERY_STEPS}
scorer_binary_loss_weight=${SCORER_BINARY_LOSS_WEIGHT}
scorer_progress_loss_weight=${SCORER_PROGRESS_LOSS_WEIGHT}
scorer_score_success_weight=${SCORER_SCORE_SUCCESS_WEIGHT}
scorer_score_handoff_success_weight=${SCORER_SCORE_HANDOFF_SUCCESS_WEIGHT}
scorer_score_inserted_weight=${SCORER_SCORE_INSERTED_WEIGHT}
scorer_score_grasped_weight=${SCORER_SCORE_GRASPED_WEIGHT}
scorer_score_progress_weight=${SCORER_SCORE_PROGRESS_WEIGHT}
scorer_score_progress_delta_weight=${SCORER_SCORE_PROGRESS_DELTA_WEIGHT}
scorer_score_continuable_weight=${SCORER_SCORE_CONTINUABLE_WEIGHT}
scorer_score_state_abs_axis_weights=${SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS}
scorer_score_state_target=${SCORER_SCORE_STATE_TARGET}
scorer_allowed_candidate_families=${SCORER_ALLOWED_CANDIDATE_FAMILIES}
scorer_min_selected_progress_delta_improvement=${SCORER_MIN_SELECTED_PROGRESS_DELTA_IMPROVEMENT}
scorer_min_selected_handoff_success_improvement=${SCORER_MIN_SELECTED_HANDOFF_SUCCESS_IMPROVEMENT}
scorer_max_selected_error_degradation_for_handoff_gate=${SCORER_MAX_SELECTED_ERROR_DEGRADATION_FOR_HANDOFF_GATE}
scorer_allow_handoff_only_gate=${SCORER_ALLOW_HANDOFF_ONLY_GATE}
boundary=Compute-node union of completed retrieval shards. It is a replay/headroom diagnostic, not live controller evidence.
EOF

if [[ "${#OUTCOME_ARGS[@]}" -eq 0 ]]; then
  printf '%s\n' "${ACTIVE_INPROGRESS_CLAIMS[@]}" > "${SUMMARY_ROOT}/active_inprogress_claims.txt"
  printf '%s\n' "${STALE_INPROGRESS_CLAIMS[@]}" > "${SUMMARY_ROOT}/stale_inprogress_claims.txt"
  cat > "${SUMMARY_ROOT}/no_completed_shards.txt" <<EOF
done=false
reason=$([[ "${#ACTIVE_INPROGRESS_CLAIMS[@]}" -gt 0 ]] && printf 'no_completed_retrieval_shards_but_active_claims_exist' || printf 'no_completed_retrieval_shards')
date=$(date --iso-8601=seconds)
shard_claim_root=${SHARD_CLAIM_ROOT}
num_active_inprogress_claims=${#ACTIVE_INPROGRESS_CLAIMS[@]}
num_stale_inprogress_claims=${#STALE_INPROGRESS_CLAIMS[@]}
EOF
  if [[ "${#ACTIVE_INPROGRESS_CLAIMS[@]}" -gt 0 ]]; then
    exit 0
  fi
  exit 20
fi

printf '%s\n' "${DONE_CLAIMS[@]}" > "${SUMMARY_ROOT}/done_claims.txt"

".venv/bin/python" -m py_compile \
  scripts/world_model/summarize_cosmos3_candidate_outcome_headroom.py \
  scripts/world_model/train_cosmos3_candidate_outcome_scorer.py \
  scripts/world_model/eval_cosmos3_candidate_outcome_scorer_margins.py

".venv/bin/python" scripts/world_model/summarize_cosmos3_candidate_outcome_headroom.py \
  "${OUTCOME_ARGS[@]}" \
  --output-root "${SUMMARY_ROOT}" \
  --val-fraction "${VAL_FRACTION}" \
  --seed "${SEED}"

GATE_SUMMARY="${SUMMARY_ROOT}/retrieval_union_scorer_gate_summary.json"
".venv/bin/python" - "${SUMMARY_ROOT}/candidate_outcome_headroom_summary.json" "${GATE_SUMMARY}" \
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
    and (far_groups == 0 or far_oracle_success >= min_far_success)
    and retrieval_contribution_ok
)
progress_headroom_gate = bool(
    allow_progress_gate
    and mean_delta <= max_mean_delta
    and meaningful_improvement_fraction >= min_meaningful_fraction
    and large_improvement_fraction >= min_large_fraction
    and (far_groups == 0 or far_mean_delta <= max_far_mean_delta)
    and retrieval_contribution_ok
)
passed = bool(terminal_success_gate or progress_headroom_gate)
if terminal_success_gate:
    plain_reason = "Union retrieval shards produced enough successful hard-phase candidates to justify a short scorer smoke."
elif progress_headroom_gate:
    plain_reason = (
        "Union retrieval shards produced enough receding-chunk progress/contact headroom to train the progress/value scorer, "
        "even though one chunk is not yet a terminal success."
    )
else:
    plain_reason = (
        "Union retrieval shards did not create enough hard-phase candidate headroom; "
        "improve the candidate source instead of training another scorer."
    )
gate = {
    "schema": "cosmos3_hardphase_retrieval_union_scorer_gate_v1",
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
        "Offline replay-headroom gate only. It is not formal training evidence or live controller evidence. "
        "The progress gate preserves the receding-control objective: far/hard chunks may be useful by moving toward contact-continuable states before a later chunk inserts."
    ),
}
gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
print(json.dumps(gate, sort_keys=True))
PY

if [[ "${RUN_SCORER_AFTER_UNION_GATE}" != "true" ]]; then
  cat > "${SUMMARY_ROOT}/union_scorer_skipped.txt" <<EOF
skipped=true
reason=RUN_SCORER_AFTER_UNION_GATE is not true
gate_summary=${GATE_SUMMARY}
EOF
  exit 0
fi

if [[ "${#DONE_CLAIMS[@]}" -lt "${MIN_UNION_DONE_CLAIMS_FOR_SCORER}" ]]; then
  cat > "${SUMMARY_ROOT}/union_scorer_skipped.txt" <<EOF
skipped=true
reason=union_done_claim_count_below_floor
done_claims=${#DONE_CLAIMS[@]}
min_union_done_claims_for_scorer=${MIN_UNION_DONE_CLAIMS_FOR_SCORER}
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
  cat > "${SUMMARY_ROOT}/union_scorer_skipped.txt" <<EOF
skipped=true
reason=union_headroom_gate_failed
gate_summary=${GATE_SUMMARY}
EOF
  exit 0
fi

".venv/bin/python" scripts/world_model/train_cosmos3_candidate_outcome_scorer.py \
  --contact-executor-jsonl "${CONTACT_EXECUTOR_JSONL}" \
  "${OUTCOME_ARGS[@]}" \
  --output-root "${SCORER_ROOT}" \
  --val-fraction "${VAL_FRACTION}" \
  --max-steps "${SCORER_MAX_STEPS}" \
  --min-steps "${SCORER_MIN_STEPS:-0}" \
  --min-wall-seconds "${SCORER_MIN_WALL_SECONDS:-0}" \
  --max-wall-seconds "${SCORER_MAX_WALL_SECONDS:-0}" \
  --formal-min-gpus "${SCORER_FORMAL_MIN_GPUS:-1}" \
  --eval-every-steps "${SCORER_EVAL_EVERY_STEPS}" \
  --save-every-steps "${SCORER_SAVE_EVERY_STEPS}" \
  --batch-size "${SCORER_BATCH_SIZE}" \
  --hidden-dim "${SCORER_HIDDEN_DIM}" \
  --num-layers "${SCORER_NUM_LAYERS}" \
  --dropout "${SCORER_DROPOUT}" \
  --lr "${SCORER_LR}" \
  --weight-decay "${SCORER_WEIGHT_DECAY}" \
  --binary-loss-weight "${SCORER_BINARY_LOSS_WEIGHT}" \
  --progress-loss-weight "${SCORER_PROGRESS_LOSS_WEIGHT}" \
  --rank-loss-weight "${SCORER_RANK_LOSS_WEIGHT}" \
  --rank-loss-temperature 0.05 \
  --score-success-weight "${SCORER_SCORE_SUCCESS_WEIGHT}" \
  --score-handoff-success-weight "${SCORER_SCORE_HANDOFF_SUCCESS_WEIGHT}" \
  --score-inserted-weight "${SCORER_SCORE_INSERTED_WEIGHT}" \
  --score-grasped-weight "${SCORER_SCORE_GRASPED_WEIGHT}" \
  --score-progress-weight "${SCORER_SCORE_PROGRESS_WEIGHT}" \
  --score-progress-delta-weight "${SCORER_SCORE_PROGRESS_DELTA_WEIGHT}" \
  --score-continuable-weight "${SCORER_SCORE_CONTINUABLE_WEIGHT}" \
  --score-state-abs-axis-weights "${SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS}" \
  --score-state-target "${SCORER_SCORE_STATE_TARGET}" \
  --allowed-candidate-families "${SCORER_ALLOWED_CANDIDATE_FAMILIES}" \
  --min-selected-error-improvement 0.005 \
  --min-selected-progress-delta-improvement "${SCORER_MIN_SELECTED_PROGRESS_DELTA_IMPROVEMENT}" \
  --min-selected-handoff-success-improvement "${SCORER_MIN_SELECTED_HANDOFF_SUCCESS_IMPROVEMENT}" \
  --max-selected-error-degradation-for-handoff-gate "${SCORER_MAX_SELECTED_ERROR_DEGRADATION_FOR_HANDOFF_GATE}" \
  --min-non-dp-selected-fraction 0.1 \
  --min-eval-groups-for-gate "${SCORER_MIN_EVAL_GROUPS}" \
  "${SCORER_HANDOFF_GATE_FLAG}" \
  --require-cuda \
  --seed "${SEED}"

SCORER_MARGIN_CHECKPOINT="${SCORER_ROOT}/checkpoint_best_offline.pt"
if [[ -s "${SCORER_ROOT}/checkpoint_best_gate.pt" ]]; then
  SCORER_MARGIN_CHECKPOINT="${SCORER_ROOT}/checkpoint_best_gate.pt"
fi

".venv/bin/python" scripts/world_model/eval_cosmos3_candidate_outcome_scorer_margins.py \
  --contact-executor-jsonl "${CONTACT_EXECUTOR_JSONL}" \
  "${OUTCOME_ARGS[@]}" \
  --checkpoint "${SCORER_MARGIN_CHECKPOINT}" \
  --output-root "${MARGIN_ROOT}" \
  --val-fraction "${VAL_FRACTION}" \
  --allowed-candidate-families "${SCORER_ALLOWED_CANDIDATE_FAMILIES}" \
  --min-selected-error-improvement 0.005 \
  --min-selected-progress-delta-improvement "${SCORER_MIN_SELECTED_PROGRESS_DELTA_IMPROVEMENT}" \
  --min-selected-handoff-success-improvement "${SCORER_MIN_SELECTED_HANDOFF_SUCCESS_IMPROVEMENT}" \
  --max-selected-error-degradation-for-handoff-gate "${SCORER_MAX_SELECTED_ERROR_DEGRADATION_FOR_HANDOFF_GATE}" \
  --min-non-dp-selected-fraction 0.1 \
  "${SCORER_HANDOFF_GATE_FLAG}" \
  --require-cuda \
  --seed "${SEED}"

cat > "${SUMMARY_ROOT}/claim_union_done.txt" <<EOF
done=true
date=$(date --iso-8601=seconds)
summary_root=${SUMMARY_ROOT}
gate_summary=${GATE_SUMMARY}
scorer_root=${SCORER_ROOT}
margin_root=${MARGIN_ROOT}
margin_checkpoint=${SCORER_MARGIN_CHECKPOINT}
boundary=Union retrieval headroom and short scorer smoke completed inside compute-node allocation. Not formal or live evidence.
EOF
