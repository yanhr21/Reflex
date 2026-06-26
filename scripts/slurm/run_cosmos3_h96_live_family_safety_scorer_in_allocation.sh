#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID:-}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this live-family safety scorer only inside a compute-node srun step from a tmux-held allocation.
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TAG="${TAG:-hardphase_h96_live_family_safety_scorer_${STAMP}_alloc${SLURM_JOB_ID}}"

export SHARD_CLAIM_ROOT="${SHARD_CLAIM_ROOT:-experiments/world_model_task_rebinding/cosmos3/hardphase_h96_retrieval_shard_claims_20260621_1242_h96shard64_alloc145276}"
export CONTACT_EXECUTOR_JSONL="${CONTACT_EXECUTOR_JSONL:-experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_rollout24_predpath_train512_20260617_server23_alloc135069/contact_executor_dataset_file.jsonl}"
export SUMMARY_ROOT="${SUMMARY_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_${TAG}}"
export SCORER_ROOT="${SCORER_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_${TAG}_formal3h}"
export MARGIN_ROOT="${MARGIN_ROOT:-experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_margin_${TAG}}"

export RUN_SCORER_AFTER_UNION_GATE=true
export MIN_UNION_DONE_CLAIMS_FOR_SCORER="${MIN_UNION_DONE_CLAIMS_FOR_SCORER:-8}"
export MIN_RETRIEVAL_ORACLE_COUNT="${MIN_RETRIEVAL_ORACLE_COUNT:-0}"
export MIN_RETRIEVAL_SUCCESS_GROUPS="${MIN_RETRIEVAL_SUCCESS_GROUPS:-0}"
export SCORER_MIN_EVAL_GROUPS="${SCORER_MIN_EVAL_GROUPS:-16}"
export SCORER_MIN_STEPS="${SCORER_MIN_STEPS:-0}"
export SCORER_MIN_WALL_SECONDS="${SCORER_MIN_WALL_SECONDS:-10800}"
export SCORER_MAX_WALL_SECONDS="${SCORER_MAX_WALL_SECONDS:-10800}"
export SCORER_FORMAL_MIN_GPUS="${SCORER_FORMAL_MIN_GPUS:-1}"

export SCORER_ALLOWED_CANDIDATE_FAMILIES="${SCORER_ALLOWED_CANDIDATE_FAMILIES:-dp_prior,checkpoint_model,model_generated,model_mean,model_scale,model_sample,model_diffusion}"
export SCORER_BINARY_LOSS_WEIGHT="${SCORER_BINARY_LOSS_WEIGHT:-1.0}"
export SCORER_RANK_LOSS_WEIGHT="${SCORER_RANK_LOSS_WEIGHT:-0.0}"
export SCORER_PROGRESS_LOSS_WEIGHT="${SCORER_PROGRESS_LOSS_WEIGHT:-0.5}"
export SCORER_SCORE_SUCCESS_WEIGHT="${SCORER_SCORE_SUCCESS_WEIGHT:-0.5}"
export SCORER_SCORE_HANDOFF_SUCCESS_WEIGHT="${SCORER_SCORE_HANDOFF_SUCCESS_WEIGHT:-4.0}"
export SCORER_SCORE_INSERTED_WEIGHT="${SCORER_SCORE_INSERTED_WEIGHT:-0.25}"
export SCORER_SCORE_GRASPED_WEIGHT="${SCORER_SCORE_GRASPED_WEIGHT:-0.1}"
export SCORER_SCORE_PROGRESS_WEIGHT="${SCORER_SCORE_PROGRESS_WEIGHT:-0.5}"
export SCORER_SCORE_PROGRESS_DELTA_WEIGHT="${SCORER_SCORE_PROGRESS_DELTA_WEIGHT:-0.5}"
export SCORER_SCORE_CONTINUABLE_WEIGHT="${SCORER_SCORE_CONTINUABLE_WEIGHT:-1.0}"
export SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS="${SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS:-0,0,0}"
export SCORER_SCORE_STATE_TARGET="${SCORER_SCORE_STATE_TARGET:-0,0,0}"

export SCORER_MIN_SELECTED_PROGRESS_DELTA_IMPROVEMENT="${SCORER_MIN_SELECTED_PROGRESS_DELTA_IMPROVEMENT:-0.0}"
export SCORER_MIN_SELECTED_HANDOFF_SUCCESS_IMPROVEMENT="${SCORER_MIN_SELECTED_HANDOFF_SUCCESS_IMPROVEMENT:-0.0}"
export SCORER_MAX_SELECTED_ERROR_DEGRADATION_FOR_HANDOFF_GATE="${SCORER_MAX_SELECTED_ERROR_DEGRADATION_FOR_HANDOFF_GATE:-0.0}"
export SCORER_ALLOW_HANDOFF_ONLY_GATE=false
export VAL_FRACTION="${VAL_FRACTION:-0.25}"
export SEED="${SEED:-20260618}"

".venv/bin/python" -m py_compile \
  scripts/world_model/train_cosmos3_candidate_outcome_scorer.py \
  scripts/world_model/eval_cosmos3_candidate_outcome_scorer_margins.py \
  scripts/world_model/run_cosmos3_live_receding_loop.py

exec bash scripts/slurm/run_cosmos3_hardphase_retrieval_claim_union_in_allocation.sh
