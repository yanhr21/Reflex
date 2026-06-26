#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run multiworker live-family h96 label expansion only inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
WORKER_GPU_IDS="${WORKER_GPU_IDS:-0,1,2,3}"

export SHARD_CLAIM_ROOT="${SHARD_CLAIM_ROOT:-experiments/world_model_task_rebinding/cosmos3/hardphase_h96_live_family_shard_claims_${STAMP}_alloc${SLURM_JOB_ID}}"
export LOG_ROOT="${LOG_ROOT:-experiments/world_model_task_rebinding/cosmos3/hardphase_h96_live_family_sharded_multiworker_${STAMP}_alloc${SLURM_JOB_ID}}"
export SUMMARY_ROOT="${SUMMARY_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_h96_live_family_claim_union_${STAMP}_alloc${SLURM_JOB_ID}}"

export SKIP_LIST="${SKIP_LIST:-64,72,80,88,96,104,112,120}"
export SHARD_SIZE="${SHARD_SIZE:-8}"
export HARD_PHASES="${HARD_PHASES:-far,lateral_align,preinsert_aligned}"

export DP_ROLLOUT_CONTINUABILITY_HORIZON="${DP_ROLLOUT_CONTINUABILITY_HORIZON:-96}"
export DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS="${DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS:-16}"
export MODEL_CANDIDATE_SAMPLES="${MODEL_CANDIDATE_SAMPLES:-8}"
export MODEL_CANDIDATE_TEMPS="${MODEL_CANDIDATE_TEMPS:-0.5,1.0,1.5}"
export MODEL_CANDIDATE_SCALES="${MODEL_CANDIDATE_SCALES:-0.2,0.5,1.0}"
export CANDIDATE_SHORT_PREFIX_STEPS="${CANDIDATE_SHORT_PREFIX_STEPS:-8,12,16}"

export INCLUDE_LEGACY_TEACHER_SCALE_CANDIDATES=false
export INCLUDE_RETRIEVAL_RESIDUAL_CANDIDATES=false

mkdir -p "${LOG_ROOT}" "${SUMMARY_ROOT}" "${SHARD_CLAIM_ROOT}"

".venv/bin/python" -m py_compile \
  scripts/world_model/export_cosmos3_candidate_outcome_labels.py \
  scripts/world_model/summarize_cosmos3_candidate_outcome_headroom.py

cat > "${LOG_ROOT}/driver_manifest.txt" <<EOF
schema=cosmos3_h96_live_family_sharded_replay_multiworker_v1
date=$(date --iso-8601=seconds)
slurm_job_id=${SLURM_JOB_ID}
slurm_step_id=${SLURM_STEP_ID}
host=$(hostname)
worker_gpu_ids=${WORKER_GPU_IDS}
shard_claim_root=${SHARD_CLAIM_ROOT}
summary_root=${SUMMARY_ROOT}
skip_list=${SKIP_LIST}
boundary=Parallel live-family h96 label expansion from the same causal candidate interface. This is offline label/headroom data, not live method evidence.
EOF

IFS=',' read -r -a GPUS <<< "${WORKER_GPU_IDS}"
pids=()
for raw_gpu in "${GPUS[@]}"; do
  gpu="$(printf '%s' "${raw_gpu}" | xargs)"
  [[ -n "${gpu}" ]] || continue
  worker_stamp="${STAMP}_gpu${gpu}"
  worker_log="${LOG_ROOT}/worker_gpu${gpu}.log"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export STAMP="${worker_stamp}"
    export LOG_ROOT="${LOG_ROOT}/worker_gpu${gpu}_shards"
    export RUN_UNION_AFTER_SHARDS=false
    bash scripts/slurm/run_cosmos3_h96_live_family_sharded_replay_in_allocation.sh
  ) > "${worker_log}" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done

if [[ "${rc}" -ne 0 ]]; then
  cat > "${LOG_ROOT}/driver_failed.txt" <<EOF
done=false
date=$(date --iso-8601=seconds)
reason=one_or_more_workers_failed
shard_claim_root=${SHARD_CLAIM_ROOT}
summary_root=${SUMMARY_ROOT}
EOF
  exit "${rc}"
fi

RUN_SCORER_AFTER_UNION_GATE=false \
STAMP="${STAMP}_union" \
SHARD_CLAIM_ROOT="${SHARD_CLAIM_ROOT}" \
SUMMARY_ROOT="${SUMMARY_ROOT}" \
bash scripts/slurm/run_cosmos3_hardphase_retrieval_claim_union_in_allocation.sh \
  2>&1 | tee "${LOG_ROOT}/union.log"

cat > "${LOG_ROOT}/driver_done.txt" <<EOF
done=true
date=$(date --iso-8601=seconds)
shard_claim_root=${SHARD_CLAIM_ROOT}
summary_root=${SUMMARY_ROOT}
EOF
