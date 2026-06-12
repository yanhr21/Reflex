#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ "${ALLOW_LEGACY_V7_NORMACTIVE_CLIP1_BAD_RECIPE:-false}" != "true" ]]; then
  cat >&2 <<'EOF'
refusing_legacy_v7_normactive_clip1_bad_recipe=true
reason=This wrapper produced the 2026-06-12 v7_733 run that selected action adapter tensors but forgot the overfit-validated fix1 action recipe: lr=1e-4, warmup=10, f_min=0.5, action_loss_weight=2.0, independent_action_schedule=true, shift_action=1.
use=scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_fix1recipe_in_allocation.sh
EOF
  exit 64
fi

export SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612}"
export CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_rgb_300step_20260612_0245}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_normactive_clip1_4gpu_${STAMP}}"

export EXPECTED_SOURCE_EPISODES="${EXPECTED_SOURCE_EPISODES:-733}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export DATA_PARALLEL_SHARD_DEGREE="${DATA_PARALLEL_SHARD_DEGREE:-4}"
export DATA_PARALLEL_REPLICATE_DEGREE="${DATA_PARALLEL_REPLICATE_DEGREE:-1}"
export CONTEXT_PARALLEL_SHARD_DEGREE="${CONTEXT_PARALLEL_SHARD_DEGREE:-1}"

export NORMALIZE_LOSS_BY_ACTIVE="${NORMALIZE_LOSS_BY_ACTIVE:-true}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export MAX_ITER="${MAX_ITER:-1500}"
export SAVE_ITER="${SAVE_ITER:-300}"
export VALIDATION_ITER="${VALIDATION_ITER:-300}"
export MAX_VAL_ITER="${MAX_VAL_ITER:-40}"
export MASTER_PORT="${MASTER_PORT:-50341}"
export FORCE_EXPORT="${FORCE_EXPORT:-false}"
export RUN_SFT="${RUN_SFT:-true}"

# This is a diagnostic retrain on the same v7_733 bootstrap source, not
# hard-dynamic method evidence. The changed settings address future-token loss
# dilution and overly aggressive clipping observed in the first v7_733 SFT.
exec bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh"
