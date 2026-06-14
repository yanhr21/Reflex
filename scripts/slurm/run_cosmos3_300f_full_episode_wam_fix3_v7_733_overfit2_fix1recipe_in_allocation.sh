#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

export SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612}"
export CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_overfit2_rgb_300step_20260612_1830}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_overfit2_rgb_300step_fix1recipe_4gpu_${STAMP}}"

export EXPECTED_SOURCE_EPISODES="${EXPECTED_SOURCE_EPISODES:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export DATA_PARALLEL_SHARD_DEGREE="${DATA_PARALLEL_SHARD_DEGREE:-4}"
export DATA_PARALLEL_REPLICATE_DEGREE="${DATA_PARALLEL_REPLICATE_DEGREE:-1}"
export CONTEXT_PARALLEL_SHARD_DEGREE="${CONTEXT_PARALLEL_SHARD_DEGREE:-1}"

export MAX_ITER="${MAX_ITER:-300}"
export SAVE_ITER="${SAVE_ITER:-100}"
export VALIDATION_ITER="${VALIDATION_ITER:-50}"
export MAX_VAL_ITER="${MAX_VAL_ITER:-2}"
export MASTER_PORT="${MASTER_PORT:-50353}"
export FORCE_EXPORT="${FORCE_EXPORT:-false}"
export RUN_SFT="${RUN_SFT:-true}"
export ALLOW_LEGACY_V7_733_OVERFIT2_SFT_DIAGNOSTIC="${ALLOW_LEGACY_V7_733_OVERFIT2_SFT_DIAGNOSTIC:-false}"

if [[ "${DRY_RUN_CONFIG_ONLY:-false}" == "true" ]]; then
  cat <<EOF
dry_run_config_only=true
source_dataset_root=${SOURCE_DATASET_ROOT}
condition_root=${CONDITION_ROOT}
output_root=${OUTPUT_ROOT}
expected_source_episodes=${EXPECTED_SOURCE_EPISODES}
run_sft=${RUN_SFT}
allow_legacy_v7_733_overfit2_sft_diagnostic=${ALLOW_LEGACY_V7_733_OVERFIT2_SFT_DIAGNOSTIC}
would_refuse_legacy_training=$([[ "${RUN_SFT}" == "true" && "${ALLOW_LEGACY_V7_733_OVERFIT2_SFT_DIAGNOSTIC}" != "true" ]] && echo true || echo false)
replacement=scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_overfit2_preflight_in_allocation.sh
boundary=configuration-only dry run; no export, training, rendering, or eval is launched.
EOF
  exit 0
fi

if [[ "${RUN_SFT}" == "true" && "${ALLOW_LEGACY_V7_733_OVERFIT2_SFT_DIAGNOSTIC}" != "true" ]]; then
  cat >&2 <<'EOF'
refusing_legacy_v7_733_overfit2_sft=true
reason=The old sampled-role overfit2 chain is historical. Use the clean/dense overfit2 preflight and guarded overfit SFT entry for active repair.
use_preflight=scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_overfit2_preflight_in_allocation.sh
use_sft=scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_overfit2_fix1recipe_in_allocation.sh
override_for_non_method_diagnostic=ALLOW_LEGACY_V7_733_OVERFIT2_SFT_DIAGNOSTIC=true
EOF
  exit 67
fi

# These are also the base defaults. They stay explicit here because overfit is
# the regression gate that catches accidental action-training recipe drift.
export OPTIMIZER_KEYS_TO_SELECT="${OPTIMIZER_KEYS_TO_SELECT:-moe_gen,time_embedder,vae2llm,llm2vae,action2llm,llm2action,action_modality_embed}"
export OPTIMIZER_LR="${OPTIMIZER_LR:-1.0e-4}"
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-10}"
export SCHEDULER_F_MIN="${SCHEDULER_F_MIN:-0.5}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"
export NORMALIZE_LOSS_BY_ACTIVE="${NORMALIZE_LOSS_BY_ACTIVE:-true}"
export INDEPENDENT_ACTION_SCHEDULE="${INDEPENDENT_ACTION_SCHEDULE:-true}"
export SHIFT_ACTION="${SHIFT_ACTION:-1}"
export ENFORCE_FIX1_ACTION_RECIPE="${ENFORCE_FIX1_ACTION_RECIPE:-true}"

exec bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh"
