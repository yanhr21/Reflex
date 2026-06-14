#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

# Train the two-source clean/dense overfit gate only after explicit user
# approval and only from a condition root that passed clean_dense_preflight.
if [[ "${DRY_RUN_CONFIG_ONLY:-false}" == "true" ]]; then
  cat <<EOF
dry_run_config_only=true
condition_root=${CONDITION_ROOT:-<required>}
clean_dense_preflight_summary=${CLEAN_DENSE_PREFLIGHT_SUMMARY:-<required>}
next_action_gate_output=${NEXT_ACTION_GATE_OUTPUT:-<summary_dir>/next_action_gate_clean_dense_overfit_sft.json}
output_root=${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_rgb_300step_fix1recipe_4gpu_${STAMP}}
expected_source_episodes=${EXPECTED_SOURCE_EPISODES:-2}
run_sft=${RUN_SFT:-true}
allow_clean_dense_overfit_sft=${ALLOW_CLEAN_DENSE_OVERFIT_SFT:-false}
prefix_role_source=${PREFIX_ROLE_SOURCE:-physical_mode}
dense_receding_prefix_stride=${DENSE_RECEDING_PREFIX_STRIDE:-8}
late_rebind_weight=${LATE_REBIND_WEIGHT:-3}
max_iter=${MAX_ITER:-300}
save_iter=${SAVE_ITER:-100}
validation_iter=${VALIDATION_ITER:-50}
nproc_per_node=${NPROC_PER_NODE:-4}
would_refuse_without_allow=$([[ "${ALLOW_CLEAN_DENSE_OVERFIT_SFT:-false}" == "true" ]] && echo false || echo true)
would_refuse_without_condition_root=$([[ -n "${CONDITION_ROOT:-}" ]] && echo false || echo true)
would_refuse_without_preflight_summary=$([[ -n "${CLEAN_DENSE_PREFLIGHT_SUMMARY:-}" ]] && echo false || echo true)
boundary=configuration-only dry run; no preflight validation, export, training, rendering, or eval is launched.
EOF
  exit 0
fi

if [[ "${ALLOW_CLEAN_DENSE_OVERFIT_SFT:-false}" != "true" ]]; then
  cat >&2 <<'EOF'
refusing_clean_dense_overfit_sft=true
reason=Set ALLOW_CLEAN_DENSE_OVERFIT_SFT=true only after the user approves clean/dense overfit training.
required_preflight=clean_dense_preflight_summary.json with ready_for_overfit=true
EOF
  exit 41
fi

if [[ -z "${CONDITION_ROOT:-}" ]]; then
  cat >&2 <<'EOF'
refusing_clean_dense_overfit_sft=true
reason=CONDITION_ROOT must point at the approved clean/dense overfit2 condition root.
EOF
  exit 42
fi

if [[ -z "${CLEAN_DENSE_PREFLIGHT_SUMMARY:-}" ]]; then
  cat >&2 <<'EOF'
refusing_clean_dense_overfit_sft=true
reason=CLEAN_DENSE_PREFLIGHT_SUMMARY must explicitly point at the approved clean_dense_preflight_summary.json for CONDITION_ROOT.
EOF
  exit 43
fi

if [[ ! -s "${CLEAN_DENSE_PREFLIGHT_SUMMARY}" ]]; then
  cat >&2 <<EOF
refusing_clean_dense_overfit_sft=true
reason=Missing clean dense preflight summary.
clean_dense_preflight_summary=${CLEAN_DENSE_PREFLIGHT_SUMMARY}
EOF
  exit 44
fi

NEXT_ACTION_GATE_OUTPUT="${NEXT_ACTION_GATE_OUTPUT:-$(dirname "${CLEAN_DENSE_PREFLIGHT_SUMMARY}")/next_action_gate_clean_dense_overfit_sft.json}"
USER_APPROVED=true \
CLEAN_DENSE_PREFLIGHT_SUMMARY="${CLEAN_DENSE_PREFLIGHT_SUMMARY}" \
OUTPUT_JSON="${NEXT_ACTION_GATE_OUTPUT}" \
  bash "${ROOT}/scripts/world_model/check_current_cosmos3_next_action_gate.sh" \
  clean_dense_overfit_sft_after_user_approval

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/world_model/check_cosmos3_clean_dense_preflight_summary_ready.py" \
  --summary-json "${CLEAN_DENSE_PREFLIGHT_SUMMARY}" \
  --condition-root "${CONDITION_ROOT}" \
  --output-json "$(dirname "${CLEAN_DENSE_PREFLIGHT_SUMMARY}")/clean_dense_preflight_summary_sft_entry_gate.json"

export SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_rgb_300step_fix1recipe_4gpu_${STAMP}}"

export EXPECTED_SOURCE_EPISODES="${EXPECTED_SOURCE_EPISODES:-2}"
export MAX_RECORDS="${MAX_RECORDS:-0}"
export FORCE_EXPORT="${FORCE_EXPORT:-false}"
export RUN_SFT="${RUN_SFT:-true}"

export PREFIX_ROLE_SOURCE="${PREFIX_ROLE_SOURCE:-physical_mode}"
export DENSE_RECEDING_PREFIX_STRIDE="${DENSE_RECEDING_PREFIX_STRIDE:-8}"
export ROLE_WEIGHT_CONFIG="${ROLE_WEIGHT_CONFIG:-}"
export LATE_REBIND_WEIGHT="${LATE_REBIND_WEIGHT:-3}"
export LATE_REBIND_ROLES="${LATE_REBIND_ROLES:-target_motion_observed,target_post_motion,insert_resume}"
export LATE_REBIND_MIN_ABS_X="${LATE_REBIND_MIN_ABS_X:-0.05}"
export LATE_REBIND_MIN_ABS_Y="${LATE_REBIND_MIN_ABS_Y:-0.01}"
export LATE_REBIND_MIN_ABS_Z="${LATE_REBIND_MIN_ABS_Z:-0.004}"
export MIN_LATE_REBIND_CANDIDATES="${MIN_LATE_REBIND_CANDIDATES:-1}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export DATA_PARALLEL_SHARD_DEGREE="${DATA_PARALLEL_SHARD_DEGREE:-4}"
export DATA_PARALLEL_REPLICATE_DEGREE="${DATA_PARALLEL_REPLICATE_DEGREE:-1}"
export CONTEXT_PARALLEL_SHARD_DEGREE="${CONTEXT_PARALLEL_SHARD_DEGREE:-1}"

export MAX_ITER="${MAX_ITER:-300}"
export SAVE_ITER="${SAVE_ITER:-100}"
export VALIDATION_ITER="${VALIDATION_ITER:-50}"
export MAX_VAL_ITER="${MAX_VAL_ITER:-2}"
export MASTER_PORT="${MASTER_PORT:-50363}"

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
