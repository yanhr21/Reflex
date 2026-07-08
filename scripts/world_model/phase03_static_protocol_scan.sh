#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

files=(
  scripts/training/eval_dp_oracle_full_pipeline.py
  scripts/world_model/audit_phase03_oracle_full_pipeline_outputs.py
  scripts/slurm/run_phase03_oracle_full_pipeline_in_allocation.sh
  scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh
  scripts/slurm/phase03_h5_source.sh
  scripts/slurm/phase03_forward_backward_next.sh
  scripts/slurm/phase03_forward_backward_probe.sh
  scripts/world_model/phase03_forward_backward_readiness.sh
  scripts/world_model/require_phase03_forward_backward_launch_allowed.sh
  scripts/world_model/phase03_next_coverage_status.sh
)

failures=()

for file in "${files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    failures+=("missing_file:${file}")
    continue
  fi
  if perl -0777 -ne 'exit((index($_, "\0") >= 0) ? 0 : 1)' "${file}"; then
    failures+=("nul_byte_in_file:${file}")
  fi
done

if rg -n 'peg\.set_(pose|state|state_dict)\s*\(' "${files[@]}" >/tmp/phase03_static_scan_forbidden_peg_state.$$ 2>/dev/null; then
  failures+=("forbidden_direct_peg_state_edit")
fi

if rg -n 'world_model_task_rebinding|experiments/dp_peg1000|source_env_state|restore_state|saved_state_replay|final_seat|geometric_final|hand_selected_suffix' "${files[@]}" >/tmp/phase03_static_scan_forbidden_legacy.$$ 2>/dev/null; then
  failures+=("forbidden_legacy_state_intervention_or_suffix_text")
fi

if ! rg -n 'phase03_forward_backward_readiness\.sh' scripts/slurm/phase03_forward_backward_probe.sh >/dev/null; then
  failures+=("forward_backward_probe_missing_readiness_gate")
fi

if ! rg -n 'require_phase03_next_coverage\.sh forward_backward_target_motion' scripts/world_model/phase03_forward_backward_readiness.sh >/dev/null; then
  failures+=("readiness_missing_next_coverage_guard")
fi

if ! rg -n 'require_phase03_forward_backward_launch_allowed\.sh' scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh >/dev/null; then
  failures+=("forward_backward_launcher_missing_launch_allowed_gate")
fi

if rg -n 'phase03_forward_backward_readiness\.sh' scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh >/tmp/phase03_static_scan_login_readiness.$$ 2>/dev/null; then
  failures+=("forward_backward_launcher_runs_readiness_on_login_node")
fi

if ! rg -n 'launch_block_reasons\}" == "scheduler_delay_exceeds_threshold"' scripts/world_model/require_phase03_forward_backward_launch_allowed.sh >/dev/null; then
  failures+=("launch_allowed_helper_missing_exact_scheduler_delay_override")
fi

if ! rg -n 'scheduler_delay_exceeds_immediate_window' scripts/world_model/phase03_next_coverage_status.sh >/dev/null; then
  failures+=("next_coverage_status_missing_immediate_window_guard")
fi

if ! rg -n 'invalid_immediate_seconds' scripts/world_model/phase03_next_coverage_status.sh >/dev/null; then
  failures+=("next_coverage_status_missing_immediate_seconds_validation")
fi

if ! rg -n 'invalid_scheduler_test_cache_seconds' scripts/world_model/phase03_next_coverage_status.sh >/dev/null; then
  failures+=("next_coverage_status_missing_scheduler_cache_validation")
fi

if ! rg -n 'scheduler_test_cache_hit' scripts/world_model/phase03_next_coverage_status.sh >/dev/null; then
  failures+=("next_coverage_status_missing_scheduler_cache_output")
fi

if ! rg -n 'scheduler_test_skipped=true' scripts/world_model/phase03_next_coverage_status.sh >/dev/null; then
  failures+=("next_coverage_status_missing_preblock_scheduler_skip")
fi

if ! rg -n 'same_name_slurm_job_query_failed' scripts/world_model/phase03_next_coverage_status.sh >/dev/null; then
  failures+=("next_coverage_status_missing_same_name_job_query_failure_guard")
fi

if ! rg -n 'partition_idle_nodes_zero' scripts/world_model/phase03_next_coverage_status.sh >/dev/null; then
  failures+=("next_coverage_status_missing_partition_idle_preblock")
fi

if ! rg -n 'completion_gate_failed' scripts/world_model/phase03_next_coverage_status.sh >/dev/null; then
  failures+=("next_coverage_status_missing_completion_gate_failure_guard")
fi

if ! rg -n 'completion_status.*-eq 3' scripts/world_model/phase03_next_coverage_status.sh >/dev/null; then
  failures+=("next_coverage_status_must_accept_incomplete_completion_gate_exit")
fi

if ! rg -n 'pre_scheduler_block_reasons_text.*== "none"' scripts/world_model/phase03_next_coverage_status.sh >/dev/null; then
  failures+=("next_coverage_status_may_override_preblock_with_unknown")
fi

if ! rg -n 'IMMEDIATE_SECONDS="\$\{IMMEDIATE_SECONDS:-60\}"' scripts/world_model/require_phase03_forward_backward_launch_allowed.sh >/dev/null; then
  failures+=("launch_allowed_helper_missing_immediate_seconds_pass_through")
fi

if ! rg -n 'SCHEDULER_TEST_CACHE_SECONDS="\$\{SCHEDULER_TEST_CACHE_SECONDS:-0\}"' scripts/world_model/require_phase03_forward_backward_launch_allowed.sh >/dev/null; then
  failures+=("launch_allowed_helper_must_default_to_no_scheduler_cache")
fi

if ! rg -n 'refusing_active_launch_scheduler_cache=true' scripts/world_model/require_phase03_forward_backward_launch_allowed.sh >/dev/null; then
  failures+=("launch_allowed_helper_must_reject_scheduler_cache_override")
fi

if ! rg -n 'refusing_missing_mandatory_exclude_node=true' scripts/world_model/require_phase03_forward_backward_launch_allowed.sh >/dev/null; then
  failures+=("launch_allowed_helper_must_reject_missing_mandatory_exclude_node")
fi

if ! rg -n 'mandatory_exclude_nodes_missing' scripts/world_model/phase03_next_coverage_status.sh >/dev/null; then
  failures+=("next_coverage_status_missing_mandatory_exclude_guard")
fi

if ! rg -n 'active_launcher_would_fail_salloc_immediate_window' scripts/world_model/require_phase03_forward_backward_launch_allowed.sh >/dev/null; then
  failures+=("launch_allowed_helper_allows_far_scheduler_immediate_bypass")
fi

if rg -n 'STATUS_LAUNCH_GATE|SCHEDULER_TEST_ONLY_GUARD' scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh >/tmp/phase03_static_scan_launch_bypass.$$ 2>/dev/null; then
  failures+=("forward_backward_launcher_has_launch_gate_bypass")
fi

if rg -n 'SKIP_PHASE03_NEXT_COVERAGE_GUARD.*!= "true"' \
  scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh \
  scripts/slurm/phase03_forward_backward_probe.sh \
  scripts/slurm/phase03_forward_backward_next.sh \
  >/tmp/phase03_static_scan_skip_bypass.$$ 2>/dev/null; then
  failures+=("forward_backward_launcher_allows_skip_next_coverage_guard")
fi

for skip_guard_file in \
  scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh \
  scripts/slurm/phase03_forward_backward_probe.sh \
  scripts/slurm/phase03_forward_backward_next.sh
do
  if ! rg -n 'refusing_skip_phase03_next_coverage_guard=true' "${skip_guard_file}" >/dev/null; then
    failures+=("missing_skip_next_coverage_refusal:${skip_guard_file}")
  fi
done

if ! rg -n 'bash scripts/slurm/phase03_h5_source\.sh' scripts/slurm/phase03_forward_backward_next.sh >/dev/null; then
  failures+=("forward_backward_next_missing_h5_source_launcher")
fi

for required_guard in \
  refusing_row_offset_diagnostic_for_coverage \
  refusing_future_label_dynamic_controller_for_coverage \
  refusing_future_label_teacher_for_coverage \
  refusing_direction_guard_diagnostic_for_coverage
do
  if ! rg -n "${required_guard}" scripts/slurm/phase03_forward_backward_next.sh >/dev/null; then
    failures+=("forward_backward_next_missing_${required_guard}")
  fi
done

if ! rg -n 'refusing_login_node_execution=true' scripts/slurm/run_phase03_oracle_full_pipeline_in_allocation.sh scripts/slurm/phase03_render_gpu_probe.sh >/dev/null; then
  failures+=("missing_login_node_execution_refusal")
fi

for slurm_only_entrypoint in \
  scripts/slurm/phase03_forward_backward_probe.sh \
  scripts/slurm/phase03_forward_backward_next.sh
do
  if ! rg -n 'refusing_login_node_execution=true' "${slurm_only_entrypoint}" >/dev/null; then
    failures+=("missing_forward_backward_login_node_execution_refusal:${slurm_only_entrypoint}")
  fi
done

if ((${#failures[@]})); then
  echo "phase03_static_protocol_scan_ok=false"
  printf 'failure=%s\n' "${failures[@]}"
  if [[ -f /tmp/phase03_static_scan_forbidden_peg_state.$$ ]]; then
    cat /tmp/phase03_static_scan_forbidden_peg_state.$$
  fi
  if [[ -f /tmp/phase03_static_scan_forbidden_legacy.$$ ]]; then
    cat /tmp/phase03_static_scan_forbidden_legacy.$$
  fi
  if [[ -f /tmp/phase03_static_scan_launch_bypass.$$ ]]; then
    cat /tmp/phase03_static_scan_launch_bypass.$$
  fi
  if [[ -f /tmp/phase03_static_scan_skip_bypass.$$ ]]; then
    cat /tmp/phase03_static_scan_skip_bypass.$$
  fi
  if [[ -f /tmp/phase03_static_scan_login_readiness.$$ ]]; then
    cat /tmp/phase03_static_scan_login_readiness.$$
  fi
  rm -f \
    /tmp/phase03_static_scan_forbidden_peg_state.$$ \
    /tmp/phase03_static_scan_forbidden_legacy.$$ \
    /tmp/phase03_static_scan_launch_bypass.$$ \
    /tmp/phase03_static_scan_skip_bypass.$$ \
    /tmp/phase03_static_scan_login_readiness.$$
  exit 2
fi

rm -f \
  /tmp/phase03_static_scan_forbidden_peg_state.$$ \
  /tmp/phase03_static_scan_forbidden_legacy.$$ \
  /tmp/phase03_static_scan_launch_bypass.$$ \
  /tmp/phase03_static_scan_skip_bypass.$$ \
  /tmp/phase03_static_scan_login_readiness.$$
echo "phase03_static_protocol_scan_ok=true"
echo "checked_files=${#files[@]}"
