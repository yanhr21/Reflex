#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run live-family h96 label expansion only inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

export SHARD_CLAIM_ROOT="${SHARD_CLAIM_ROOT:-experiments/world_model_task_rebinding/cosmos3/hardphase_h96_live_family_shard_claims_${STAMP}_alloc${SLURM_JOB_ID}}"
export LOG_ROOT="${LOG_ROOT:-experiments/world_model_task_rebinding/cosmos3/hardphase_h96_live_family_sharded_${STAMP}_alloc${SLURM_JOB_ID}}"
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

".venv/bin/python" -m py_compile \
  scripts/world_model/export_cosmos3_candidate_outcome_labels.py \
  scripts/world_model/summarize_cosmos3_candidate_outcome_headroom.py

exec bash scripts/slurm/run_cosmos3_hardphase_h96_sharded_replay_in_allocation.sh
