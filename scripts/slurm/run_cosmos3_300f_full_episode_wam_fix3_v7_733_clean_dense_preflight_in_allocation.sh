#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

# This wrapper prepares the post-closed-loop-failure repair condition root only.
# It does not train by default. Run it inside a held Slurm allocation after the
# user approves the clean-role/dense-receding repair direction.
export SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612}"
export CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_rgb_300step_${STAMP}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_preflight_${STAMP}}"

export EXPECTED_SOURCE_EPISODES="${EXPECTED_SOURCE_EPISODES:-733}"
export FORCE_EXPORT="${FORCE_EXPORT:-true}"
export RUN_SFT="${RUN_SFT:-false}"

export PREFIX_ROLE_SOURCE="${PREFIX_ROLE_SOURCE:-physical_mode}"
export DENSE_RECEDING_PREFIX_STRIDE="${DENSE_RECEDING_PREFIX_STRIDE:-8}"
export MIN_PREFIX_FRAMES="${MIN_PREFIX_FRAMES:-12}"
export MAX_PREFIX_FRAMES="${MAX_PREFIX_FRAMES:-260}"
export ROLE_WEIGHT_CONFIG="${ROLE_WEIGHT_CONFIG:-}"
export LATE_REBIND_WEIGHT="${LATE_REBIND_WEIGHT:-3}"
export LATE_REBIND_ROLES="${LATE_REBIND_ROLES:-target_motion_observed,target_post_motion,insert_resume}"
export LATE_REBIND_MIN_ABS_X="${LATE_REBIND_MIN_ABS_X:-0.05}"
export LATE_REBIND_MIN_ABS_Y="${LATE_REBIND_MIN_ABS_Y:-0.01}"
export LATE_REBIND_MIN_ABS_Z="${LATE_REBIND_MIN_ABS_Z:-0.004}"
export MIN_LATE_REBIND_CANDIDATES="${MIN_LATE_REBIND_CANDIDATES:-1}"
DEFAULT_LIVE_QUERY_COVERAGE_SUMMARIES="${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_panel3_dynamic_20260613_alloc127559/live_receding_panel_summary.json,${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen2_20260614/cosmos_iter2700_puredp_fail_4_9_10_11_12_13/live_receding_panel_summary.json"
export RUN_LIVE_QUERY_COVERAGE_AUDIT="${RUN_LIVE_QUERY_COVERAGE_AUDIT:-true}"
export LIVE_QUERY_COVERAGE_SUMMARIES="${LIVE_QUERY_COVERAGE_SUMMARIES:-${DEFAULT_LIVE_QUERY_COVERAGE_SUMMARIES}}"
export LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_COUNT="${LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_COUNT:-0}"
export LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_FRACTION="${LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_FRACTION:-0.0}"
export ALLOW_SKIP_LIVE_QUERY_COVERAGE_DIAGNOSTIC="${ALLOW_SKIP_LIVE_QUERY_COVERAGE_DIAGNOSTIC:-false}"

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
force_export=${FORCE_EXPORT}
run_sft=${RUN_SFT}
prefix_role_source=${PREFIX_ROLE_SOURCE}
dense_receding_prefix_stride=${DENSE_RECEDING_PREFIX_STRIDE}
min_prefix_frames=${MIN_PREFIX_FRAMES}
max_prefix_frames=${MAX_PREFIX_FRAMES}
late_rebind_weight=${LATE_REBIND_WEIGHT}
late_rebind_roles=${LATE_REBIND_ROLES}
late_rebind_min_abs_x=${LATE_REBIND_MIN_ABS_X}
late_rebind_min_abs_y=${LATE_REBIND_MIN_ABS_Y}
late_rebind_min_abs_z=${LATE_REBIND_MIN_ABS_Z}
min_late_rebind_candidates=${MIN_LATE_REBIND_CANDIDATES}
run_live_query_coverage_audit=${RUN_LIVE_QUERY_COVERAGE_AUDIT}
live_query_coverage_summaries=${LIVE_QUERY_COVERAGE_SUMMARIES}
live_query_coverage_max_undercovered_count=${LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_COUNT}
live_query_coverage_max_undercovered_fraction=${LIVE_QUERY_COVERAGE_MAX_UNDERCOVERED_FRACTION}
allow_skip_live_query_coverage_diagnostic=${ALLOW_SKIP_LIVE_QUERY_COVERAGE_DIAGNOSTIC}
base_wrapper=${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh
boundary=configuration-only dry run; no export, training, rendering, or eval is launched.
EOF
  exit 0
fi

if [[ "${ALLOW_CLEAN_DENSE_PREFLIGHT:-false}" != "true" ]]; then
  cat >&2 <<'EOF'
refusing_clean_dense_preflight=true
reason=Set ALLOW_CLEAN_DENSE_PREFLIGHT=true only after the user approves clean-role/dense-receding condition preflight.
boundary=This wrapper only prepares/audits a condition root with RUN_SFT=false, but it still writes experiment artifacts and must not run without approval.
EOF
  exit 45
fi

if [[ "${RUN_LIVE_QUERY_COVERAGE_AUDIT}" != "true" && "${ALLOW_SKIP_LIVE_QUERY_COVERAGE_DIAGNOSTIC}" != "true" ]]; then
  cat >&2 <<'EOF'
refusing_clean_dense_preflight_without_live_query_coverage=true
reason=Full v7_733 clean/dense preflight must include live-query coverage from the current val/hard closed-loop failures.
boundary=Set ALLOW_SKIP_LIVE_QUERY_COVERAGE_DIAGNOSTIC=true only for a non-method diagnostic preflight that must not unlock overfit/full SFT evidence.
EOF
  exit 46
fi

if [[ "${RUN_LIVE_QUERY_COVERAGE_AUDIT}" != "true" && "${ALLOW_SKIP_LIVE_QUERY_COVERAGE_DIAGNOSTIC}" == "true" ]]; then
  export CLEAN_DENSE_PREFLIGHT_DIAGNOSTIC_NOT_READY_REASON="${CLEAN_DENSE_PREFLIGHT_DIAGNOSTIC_NOT_READY_REASON:-live_query_coverage_audit_skipped_by_diagnostic_override}"
fi

if [[ "${RUN_LIVE_QUERY_COVERAGE_AUDIT}" == "true" ]]; then
  IFS=',' read -r -a live_query_coverage_summaries <<< "${LIVE_QUERY_COVERAGE_SUMMARIES}"
  if [[ "${#live_query_coverage_summaries[@]}" -eq 0 ]]; then
    echo "missing_live_query_coverage_summaries=true" >&2
    exit 47
  fi
  for live_summary in "${live_query_coverage_summaries[@]}"; do
    if [[ -z "${live_summary}" || ! -s "${live_summary}" ]]; then
      echo "missing_live_query_coverage_summary=${live_summary}" >&2
      exit 47
    fi
  done
fi

USER_APPROVED=true \
OUTPUT_JSON="${OUTPUT_ROOT}/next_action_gate_clean_dense_preflight.json" \
  bash "${ROOT}/scripts/world_model/check_current_cosmos3_next_action_gate.sh" \
  clean_dense_preflight_after_user_approval

exec bash "${ROOT}/scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh"
