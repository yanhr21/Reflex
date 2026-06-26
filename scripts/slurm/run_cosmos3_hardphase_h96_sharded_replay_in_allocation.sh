#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run h96 hard-phase replay shards only inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CONTACT_EXECUTOR_JSONL="${CONTACT_EXECUTOR_JSONL:-experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_rollout24_predpath_train512_20260617_server23_alloc135069/contact_executor_dataset_file.jsonl}"
CHECKPOINT="${CHECKPOINT:-experiments/world_model_task_rebinding/cosmos3/outcome_oracle_candidate_executor_shortprefix128_union_smoke200_20260620_2229_retry1_alloc143735/checkpoint_best_offline.pt}"
SHARD_CLAIM_ROOT="${SHARD_CLAIM_ROOT:-experiments/world_model_task_rebinding/cosmos3/hardphase_h96_retrieval_shard_claims_${STAMP}_alloc${SLURM_JOB_ID}}"
LOG_ROOT="${LOG_ROOT:-experiments/world_model_task_rebinding/cosmos3/hardphase_h96_retrieval_sharded_${STAMP}_alloc${SLURM_JOB_ID}}"
SUMMARY_ROOT="${SUMMARY_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_h96_retrieval_claim_union_${STAMP}_alloc${SLURM_JOB_ID}}"

HARD_PHASES="${HARD_PHASES:-far,lateral_align,preinsert_aligned}"
SHARD_SIZE="${SHARD_SIZE:-8}"
SKIP_LIST="${SKIP_LIST:-0,8,16,24,32,40,48,56}"
DP_ROLLOUT_CONTINUABILITY_HORIZON="${DP_ROLLOUT_CONTINUABILITY_HORIZON:-96}"
DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS="${DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS:-16}"
MODEL_CANDIDATE_SAMPLES="${MODEL_CANDIDATE_SAMPLES:-8}"
MODEL_CANDIDATE_TEMPS="${MODEL_CANDIDATE_TEMPS:-0.5,1.0,1.5}"
MODEL_CANDIDATE_SCALES="${MODEL_CANDIDATE_SCALES:-0.2,0.5,1.0}"
LEGACY_CANDIDATE_SCALES="${LEGACY_CANDIDATE_SCALES:-0.05,0.1,0.2,0.5,1.0}"
CANDIDATE_SHORT_PREFIX_STEPS="${CANDIDATE_SHORT_PREFIX_STEPS:-8,12,16}"
INCLUDE_LEGACY_TEACHER_SCALE_CANDIDATES="${INCLUDE_LEGACY_TEACHER_SCALE_CANDIDATES:-true}"
INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES="${INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES:-true}"
RETRIEVAL_K="${RETRIEVAL_K:-4}"
RETRIEVAL_SOURCE_JSONL="${RETRIEVAL_SOURCE_JSONL:-${CONTACT_EXECUTOR_JSONL}}"
RETRIEVAL_POSITIVE_FIELDS="${RETRIEVAL_POSITIVE_FIELDS:-future_inserted_within_chunk,future_dp_continuable_within_chunk}"
RETRIEVAL_RESIDUAL_SCALES="${RETRIEVAL_RESIDUAL_SCALES:-0.5,1.0,1.5}"
RUN_UNION_AFTER_SHARDS="${RUN_UNION_AFTER_SHARDS:-true}"

mkdir -p "${LOG_ROOT}" "${SHARD_CLAIM_ROOT}" "${SUMMARY_ROOT}"
cat > "${LOG_ROOT}/driver_manifest.txt" <<EOF
schema=cosmos3_hardphase_h96_sharded_replay_driver_v1
date=$(date --iso-8601=seconds)
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
contact_executor_jsonl=${CONTACT_EXECUTOR_JSONL}
checkpoint=${CHECKPOINT}
hard_phases=${HARD_PHASES}
shard_size=${SHARD_SIZE}
skip_list=${SKIP_LIST}
dp_rollout_continuability_horizon=${DP_ROLLOUT_CONTINUABILITY_HORIZON}
dp_rollout_continuability_min_stable_steps=${DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS}
model_candidate_samples=${MODEL_CANDIDATE_SAMPLES}
model_candidate_temps=${MODEL_CANDIDATE_TEMPS}
model_candidate_scales=${MODEL_CANDIDATE_SCALES}
include_legacy_teacher_scale_candidates=${INCLUDE_LEGACY_TEACHER_SCALE_CANDIDATES}
include_retrieval_residual_candidates=${INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES}
legacy_candidate_scales=${LEGACY_CANDIDATE_SCALES}
candidate_short_prefix_steps=${CANDIDATE_SHORT_PREFIX_STEPS}
shard_claim_root=${SHARD_CLAIM_ROOT}
summary_root=${SUMMARY_ROOT}
boundary=Generates h96 handoff-aware hard-phase outcome labels in small completed shards. This is offline label/headroom data, not live method evidence.
run_union_after_shards=${RUN_UNION_AFTER_SHARDS}
EOF

IFS=',' read -r -a SKIPS <<< "${SKIP_LIST}"
for skip in "${SKIPS[@]}"; do
  skip="$(printf '%s' "${skip}" | xargs)"
  [[ -n "${skip}" ]] || continue
  tag="skip${skip}"
  shard_stamp="${STAMP}_${tag}"
  output_root="experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_h96_retrieval_${tag}_${STAMP}_alloc${SLURM_JOB_ID}"
  headroom_root="experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_h96_retrieval_${tag}_${STAMP}_alloc${SLURM_JOB_ID}"
  filter_root="experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_hardphase_h96_retrieval_${tag}_${STAMP}_alloc${SLURM_JOB_ID}"
  (
    STAMP="${shard_stamp}" \
    CONTACT_EXECUTOR_JSONL="${CONTACT_EXECUTOR_JSONL}" \
    CHECKPOINT="${CHECKPOINT}" \
    HARD_PHASES="${HARD_PHASES}" \
    HARD_MAX_ROWS="${SHARD_SIZE}" \
    HARD_SKIP_ROWS="${skip}" \
    OUTPUT_ROOT="${output_root}" \
    HEADROOM_ROOT="${headroom_root}" \
    FILTER_ROOT="${filter_root}" \
    DP_ROLLOUT_CONTINUABILITY_HORIZON="${DP_ROLLOUT_CONTINUABILITY_HORIZON}" \
    DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS="${DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS}" \
    MODEL_CANDIDATE_SAMPLES="${MODEL_CANDIDATE_SAMPLES}" \
    MODEL_CANDIDATE_TEMPS="${MODEL_CANDIDATE_TEMPS}" \
    MODEL_CANDIDATE_SCALES="${MODEL_CANDIDATE_SCALES}" \
    LEGACY_CANDIDATE_SCALES="${LEGACY_CANDIDATE_SCALES}" \
    CANDIDATE_SHORT_PREFIX_STEPS="${CANDIDATE_SHORT_PREFIX_STEPS}" \
    INCLUDE_LEGACY_TEACHER_SCALE_CANDIDATES="${INCLUDE_LEGACY_TEACHER_SCALE_CANDIDATES}" \
    INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES="${INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES}" \
    RETRIEVAL_SOURCE_JSONL="${RETRIEVAL_SOURCE_JSONL}" \
    RETRIEVAL_K="${RETRIEVAL_K}" \
    RETRIEVAL_POSITIVE_FIELDS="${RETRIEVAL_POSITIVE_FIELDS}" \
    RETRIEVAL_RESIDUAL_SCALES="${RETRIEVAL_RESIDUAL_SCALES}" \
    SHARD_CLAIM_ROOT="${SHARD_CLAIM_ROOT}" \
    bash scripts/slurm/run_cosmos3_hardphase_exploratory_candidate_replay_in_allocation.sh
  ) 2>&1 | tee "${LOG_ROOT}/shard_${tag}.log"
done

case "${RUN_UNION_AFTER_SHARDS}" in
  0|false|FALSE|no|NO|n|N)
    cat > "${LOG_ROOT}/driver_done.txt" <<EOF
done=true
date=$(date --iso-8601=seconds)
shard_claim_root=${SHARD_CLAIM_ROOT}
summary_root=${SUMMARY_ROOT}
union_skipped=true
reason=RUN_UNION_AFTER_SHARDS_false
EOF
    exit 0
    ;;
esac

RUN_SCORER_AFTER_UNION_GATE=false \
STAMP="${STAMP}_union" \
SHARD_CLAIM_ROOT="${SHARD_CLAIM_ROOT}" \
CONTACT_EXECUTOR_JSONL="${CONTACT_EXECUTOR_JSONL}" \
SUMMARY_ROOT="${SUMMARY_ROOT}" \
bash scripts/slurm/run_cosmos3_hardphase_retrieval_claim_union_in_allocation.sh \
  2>&1 | tee "${LOG_ROOT}/union.log"

cat > "${LOG_ROOT}/driver_done.txt" <<EOF
done=true
date=$(date --iso-8601=seconds)
shard_claim_root=${SHARD_CLAIM_ROOT}
summary_root=${SUMMARY_ROOT}
EOF
