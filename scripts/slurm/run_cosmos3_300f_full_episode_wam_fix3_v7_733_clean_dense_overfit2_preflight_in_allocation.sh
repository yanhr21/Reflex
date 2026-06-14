#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

# Prepare a two-source clean/dense condition root for the next overfit gate.
# This wrapper does not train by default; it only exports/audits the condition
# root after the user approves the repair direction.
export SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612}"
export CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_overfit2_rgb_300step_${STAMP}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_preflight_${STAMP}}"

export EXPECTED_SOURCE_EPISODES="${EXPECTED_SOURCE_EPISODES:-2}"
export MAX_RECORDS="${MAX_RECORDS:-2}"
export FORCE_EXPORT="${FORCE_EXPORT:-true}"
export RUN_SFT="${RUN_SFT:-false}"

export PREFIX_ROLE_SOURCE="${PREFIX_ROLE_SOURCE:-physical_mode}"
export DENSE_RECEDING_PREFIX_STRIDE="${DENSE_RECEDING_PREFIX_STRIDE:-8}"
export ROLE_WEIGHT_CONFIG="${ROLE_WEIGHT_CONFIG:-}"
export LATE_REBIND_WEIGHT="${LATE_REBIND_WEIGHT:-3}"
export LATE_REBIND_ROLES="${LATE_REBIND_ROLES:-target_motion_observed,target_post_motion,insert_resume}"
export LATE_REBIND_MIN_ABS_X="${LATE_REBIND_MIN_ABS_X:-0.05}"
export LATE_REBIND_MIN_ABS_Y="${LATE_REBIND_MIN_ABS_Y:-0.01}"
export LATE_REBIND_MIN_ABS_Z="${LATE_REBIND_MIN_ABS_Z:-0.004}"
export MIN_LATE_REBIND_CANDIDATES="${MIN_LATE_REBIND_CANDIDATES:-1}"
export RUN_LIVE_QUERY_COVERAGE_AUDIT="${RUN_LIVE_QUERY_COVERAGE_AUDIT:-false}"
export LIVE_QUERY_COVERAGE_SUMMARIES="${LIVE_QUERY_COVERAGE_SUMMARIES:-}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export DATA_PARALLEL_SHARD_DEGREE="${DATA_PARALLEL_SHARD_DEGREE:-1}"
export DATA_PARALLEL_REPLICATE_DEGREE="${DATA_PARALLEL_REPLICATE_DEGREE:-1}"
export CONTEXT_PARALLEL_SHARD_DEGREE="${CONTEXT_PARALLEL_SHARD_DEGREE:-1}"

if [[ "${DRY_RUN_CONFIG_ONLY:-false}" == "true" ]]; then
  cat <<EOF
dry_run_config_only=true
allow_clean_dense_preflight=${ALLOW_CLEAN_DENSE_PREFLIGHT:-false}
source_dataset_root=${SOURCE_DATASET_ROOT}
condition_root=${CONDITION_ROOT}
output_root=${OUTPUT_ROOT}
expected_source_episodes=${EXPECTED_SOURCE_EPISODES}
max_records=${MAX_RECORDS}
force_export=${FORCE_EXPORT}
run_sft=${RUN_SFT}
prefix_role_source=${PREFIX_ROLE_SOURCE}
dense_receding_prefix_stride=${DENSE_RECEDING_PREFIX_STRIDE}
late_rebind_weight=${LATE_REBIND_WEIGHT}
min_late_rebind_candidates=${MIN_LATE_REBIND_CANDIDATES}
run_live_query_coverage_audit=${RUN_LIVE_QUERY_COVERAGE_AUDIT}
live_query_coverage_summaries=${LIVE_QUERY_COVERAGE_SUMMARIES}
base_wrapper=${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh
boundary=configuration-only dry run; no export, training, rendering, or eval is launched.
EOF
  exit 0
fi

if [[ "${ALLOW_CLEAN_DENSE_PREFLIGHT:-false}" != "true" ]]; then
  cat >&2 <<'EOF'
refusing_clean_dense_overfit2_preflight=true
reason=Set ALLOW_CLEAN_DENSE_PREFLIGHT=true only after the user approves clean-role/dense-receding overfit2 condition preflight.
boundary=This wrapper only prepares/audits a two-source condition root with RUN_SFT=false, but it still writes experiment artifacts and must not run without approval.
EOF
  exit 45
fi

USER_APPROVED=true \
OUTPUT_JSON="${OUTPUT_ROOT}/next_action_gate_clean_dense_preflight.json" \
  bash "${ROOT}/scripts/world_model/check_current_cosmos3_next_action_gate.sh" \
  clean_dense_preflight_after_user_approval

exec bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh"
