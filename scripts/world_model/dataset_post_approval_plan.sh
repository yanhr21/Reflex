#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
APPROVAL_RUN_GROUP="${APPROVAL_RUN_GROUP:-static_rgb}"
APPROVAL_RUN_NAME="${APPROVAL_RUN_NAME:-smoke05}"

echo "dataset_post_approval_plan_ok=true"
echo "approval_smoke=${APPROVAL_RUN_GROUP}/${APPROVAL_RUN_NAME}"
echo "read_only=true"
echo "submits_slurm=false"

stage_ready_summary() {
  local stage="$1"
  local status_file
  status_file="$(mktemp)"
  if APPROVAL_RUN_GROUP="${APPROVAL_RUN_GROUP}" APPROVAL_RUN_NAME="${APPROVAL_RUN_NAME}" \
    "${ROOT}/scripts/world_model/require_dataset_stage_ready.sh" "${stage}" >"${status_file}" 2>&1; then
    echo "  ready_now=true"
    sed 's/^/  readiness_/' "${status_file}"
  else
    echo "  ready_now=false"
    sed 's/^/  readiness_/' "${status_file}"
  fi
  rm -f "${status_file}"
}

echo "[current_gate]"
gate_file="$(mktemp)"
if RUN_GROUP="${APPROVAL_RUN_GROUP}" RUN_NAME="${APPROVAL_RUN_NAME}" \
  "${ROOT}/scripts/world_model/require_dataset_smoke_approved.sh" >"${gate_file}" 2>&1; then
  sed 's/^/  /' "${gate_file}"
  gate_passed=true
else
  sed 's/^/  /' "${gate_file}"
  gate_passed=false
fi
rm -f "${gate_file}"

echo "[dynamic_adapter_gate]"
adapter_file="$(mktemp)"
if "${ROOT}/scripts/world_model/dataset_dynamic_adapter_status.sh" >"${adapter_file}" 2>&1; then
  sed 's/^/  /' "${adapter_file}"
else
  sed 's/^/  /' "${adapter_file}"
fi
rm -f "${adapter_file}"

echo "[stage_a_full_static_rgb]"
echo "  purpose=render_official_1000_static_demos_to_rgb"
echo "  allowed_after_smoke_approval=true"
echo "  launcher=${ROOT}/scripts/slurm/launch_dataset_static_rgb_full_tmux.sh"
echo "  default_resources=partition_cpu_gpu_1_cpu_4_mem_32G_time_04:00:00"
echo "  output=experiments/maniskill/runs/01_dataset/static_rgb/full01"
echo "  log=logs/01_dataset/static_rgb/full01.log"
echo "  command_begin"
echo "    cd ${ROOT}"
echo "    scripts/slurm/launch_dataset_static_rgb_full_tmux.sh"
echo "  command_end"
if [[ "${gate_passed}" == "true" ]]; then
  echo "  ready_now=true"
else
  echo "  ready_now=false"
  echo "  reason=stage1_smoke_review_pending"
fi

echo "[stage_b_dynamic_rgb_smoke]"
echo "  target=small_smoke_before_training_scale_B"
echo "  training_scale_target=at_least_1000_episodes_after_smoke_review"
echo "  default_count=1_episode"
echo "  default_steps_per_episode=80"
echo "  default_motion_window=starts_at_step20_duration40"
echo "  launcher=${ROOT}/scripts/slurm/launch_dataset_dynamic_rgb_smoke_tmux.sh"
echo "  common_launcher=${ROOT}/scripts/slurm/launch_dataset_stage_smoke_tmux_common.sh"
echo "  runner=${ROOT}/scripts/slurm/run_dataset_dynamic_rgb_smoke_in_allocation.sh"
echo "  output=experiments/maniskill/runs/01_dataset/dynamic_rgb/smoke01"
echo "  log=logs/01_dataset/dynamic_rgb/smoke01.log"
echo "  command_begin"
echo "    cd ${ROOT}"
echo "    scripts/slurm/launch_dataset_dynamic_rgb_smoke_tmux.sh"
echo "  command_end"
stage_ready_summary "b_dynamic_smoke"

echo "[stage_c_frozen_dp_dynamic_failure]"
echo "  target=at_least_500_rollouts_after_smoke_review"
echo "  default_count=1_rollout"
echo "  default_max_episode_steps=80"
echo "  default_motion_window=starts_at_step20_duration40"
echo "  launcher=${ROOT}/scripts/slurm/launch_dataset_frozen_dp_dynamic_smoke_tmux.sh"
echo "  common_launcher=${ROOT}/scripts/slurm/launch_dataset_stage_smoke_tmux_common.sh"
echo "  runner=${ROOT}/scripts/slurm/run_dataset_frozen_dp_dynamic_smoke_in_allocation.sh"
echo "  output=experiments/maniskill/runs/01_dataset/frozen_dp_dynamic/smoke01"
echo "  log=logs/01_dataset/frozen_dp_dynamic/smoke01.log"
echo "  command_begin"
echo "    cd ${ROOT}"
echo "    scripts/slurm/launch_dataset_frozen_dp_dynamic_smoke_tmux.sh"
echo "  command_end"
stage_ready_summary "c_frozen_dp_smoke"

echo "[stage_d_future_frame_teacher]"
echo "  target=100_smoke_overfit_quality_then_500_to_1000_if_interface_works"
echo "  launcher=${ROOT}/scripts/slurm/launch_dataset_future_frame_teacher_smoke_tmux.sh"
echo "  common_launcher=${ROOT}/scripts/slurm/launch_dataset_stage_smoke_tmux_common.sh"
echo "  runner=${ROOT}/scripts/slurm/run_dataset_future_frame_teacher_smoke_in_allocation.sh"
echo "  output=experiments/maniskill/runs/01_dataset/future_teacher/smoke01"
echo "  log=logs/01_dataset/future_teacher/smoke01.log"
echo "  command_begin"
echo "    cd ${ROOT}"
echo "    scripts/slurm/launch_dataset_future_frame_teacher_smoke_tmux.sh"
echo "  command_end"
stage_ready_summary "d_future_teacher_smoke"

echo "[stage_e_cosmos_predicted_cooperation]"
echo "  target=100_to_300_after_B_D_and_cosmos_readout_validation"
echo "  launcher=${ROOT}/scripts/slurm/launch_dataset_cosmos_predicted_coop_smoke_tmux.sh"
echo "  common_launcher=${ROOT}/scripts/slurm/launch_dataset_stage_smoke_tmux_common.sh"
echo "  runner=${ROOT}/scripts/slurm/run_dataset_cosmos_predicted_coop_smoke_in_allocation.sh"
echo "  output=experiments/maniskill/runs/01_dataset/cosmos_predicted/smoke01"
echo "  log=logs/01_dataset/cosmos_predicted/smoke01.log"
echo "  command_begin"
echo "    cd ${ROOT}"
echo "    scripts/slurm/launch_dataset_cosmos_predicted_coop_smoke_tmux.sh"
echo "  command_end"
stage_ready_summary "e_cosmos_predicted_smoke"
echo "  prereq_guard=${ROOT}/scripts/world_model/require_dataset_cosmos_predicted_prereqs_ready.sh"
e_prereq_file="$(mktemp)"
if "${ROOT}/scripts/world_model/require_dataset_cosmos_predicted_prereqs_ready.sh" >"${e_prereq_file}" 2>&1; then
  sed 's/^/  prereq_/' "${e_prereq_file}"
else
  sed 's/^/  prereq_/' "${e_prereq_file}"
fi
rm -f "${e_prereq_file}"

echo "[resource_policy]"
echo "  default_gpus=1"
echo "  if_pending_reduce=cpu_memory_walltime_first_when_scientifically_acceptable"
echo "  wait_on_valid_queued_tmux_allocation=true"
echo "  use_tmux_held_slurm_allocation=true"
echo "  no_login_node_project_compute=true"
echo "  previously_bad_nodes=smoke_or_canary_diagnostic_only_record_node_evidence"
echo "  render_risk_status=${ROOT}/scripts/world_model/dataset_render_risk_status.sh"

echo "[production_after_class_smoke_approval]"
echo "  b_launcher=${ROOT}/scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh"
echo "  b_target_count=1000"
echo "  b_output=experiments/maniskill/runs/01_dataset/dynamic_rgb/prod01"
echo "  c_launcher=${ROOT}/scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh"
echo "  c_target_count=500"
echo "  c_output=experiments/maniskill/runs/01_dataset/frozen_dp_dynamic/prod01"
echo "  d_launcher=${ROOT}/scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh"
echo "  d_target_count=500"
echo "  d_output=experiments/maniskill/runs/01_dataset/future_teacher/prod01"
echo "  e_launcher=${ROOT}/scripts/slurm/launch_dataset_cosmos_predicted_production_tmux.sh"
echo "  e_target_count=100"
echo "  e_output=experiments/maniskill/runs/01_dataset/cosmos_predicted/prod01"
echo "  e_prereq_guard=${ROOT}/scripts/world_model/require_dataset_cosmos_predicted_prereqs_ready.sh"
echo "  status_helper=${ROOT}/scripts/world_model/dataset_production_status.sh"
echo "  index_builder=${ROOT}/scripts/world_model/build_dataset_production_index.sh"
echo "  index_status_helper=${ROOT}/scripts/world_model/dataset_production_index_status.sh"
echo "  index_commands_begin"
echo "    scripts/world_model/build_dataset_production_index.sh b_dynamic_production"
echo "    scripts/world_model/build_dataset_production_index.sh c_frozen_dp_production"
echo "    scripts/world_model/build_dataset_production_index.sh d_future_teacher_production"
echo "    scripts/world_model/build_dataset_production_index.sh e_cosmos_predicted_production"
echo "  index_commands_end"
