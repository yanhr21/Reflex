#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

echo "dataset_production_status_ok=true"
echo "read_only=true"
echo "submits_slurm=false"
echo "bcd_review_block_status=${ROOT}/scripts/world_model/dataset_bcd_review_block_status.sh"
echo "bcd_shard_plan=${ROOT}/scripts/world_model/dataset_bcd_production_shard_plan.sh"
echo "bcd_shard_launcher=${ROOT}/scripts/slurm/launch_dataset_bcd_production_shards_tmux.sh"
echo "bcd_next_shard_launcher=${ROOT}/scripts/slurm/launch_dataset_bcd_next_production_shard_tmux.sh"
echo "bcd_shard_launcher_default=dry_run_no_slurm"
echo "bcd_shard_index_builder=${ROOT}/scripts/world_model/build_dataset_production_shard_index.sh"

production_stage() {
  case "$1" in
    b_dynamic_production)
      echo "b_dynamic_smoke dynamic_rgb prod01 ${ROOT}/scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh ${ROOT}/scripts/slurm/run_dataset_dynamic_rgb_smoke_in_allocation.sh 1000 300 episodes"
      ;;
    c_frozen_dp_production)
      echo "c_frozen_dp_smoke frozen_dp_dynamic prod01 ${ROOT}/scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh ${ROOT}/scripts/slurm/run_dataset_frozen_dp_dynamic_smoke_in_allocation.sh 500 300 rollouts"
      ;;
    d_future_teacher_production)
      echo "d_future_teacher_smoke future_teacher prod01 ${ROOT}/scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh ${ROOT}/scripts/slurm/run_dataset_future_frame_teacher_smoke_in_allocation.sh 500 300 teacher_rollouts"
      ;;
    e_cosmos_predicted_production)
      echo "e_cosmos_predicted_smoke cosmos_predicted prod01 ${ROOT}/scripts/slurm/launch_dataset_cosmos_predicted_production_tmux.sh ${ROOT}/scripts/slurm/run_dataset_cosmos_predicted_coop_smoke_in_allocation.sh 100 300 predicted_rollouts"
      ;;
  esac
}

for stage in \
  b_dynamic_production \
  c_frozen_dp_production \
  d_future_teacher_production \
  e_cosmos_predicted_production; do
  echo "[${stage}]"
  read -r smoke_stage run_group run_name launcher runner target_count steps_per_item count_unit < <(production_stage "${stage}")
  out_dir="${ROOT}/experiments/maniskill/runs/01_dataset/${run_group}/${run_name}"
  echo "  smoke_stage=${smoke_stage}"
  echo "  target_count=${target_count}"
  echo "  count_unit=${count_unit}"
  echo "  steps_per_item=${steps_per_item}"
  echo "  launcher=${launcher}"
  echo "  launcher_exists=$([[ -x "${launcher}" ]] && echo true || echo false)"
  echo "  runner=${runner}"
  echo "  runner_exists=$([[ -x "${runner}" ]] && echo true || echo false)"
  collector_file="$(mktemp)"
  if "${ROOT}/scripts/world_model/audit_dataset_collector_source.sh" "${smoke_stage}" >"${collector_file}" 2>&1; then
    sed 's/^/  collector_audit_/' "${collector_file}"
  else
    sed 's/^/  collector_audit_/' "${collector_file}"
  fi
  rm -f "${collector_file}"
  echo "  output_dir=${out_dir}"
  echo "  output_dir_exists=$([[ -d "${out_dir}" ]] && echo true || echo false)"
  if [[ "${stage}" == "b_dynamic_production" || "${stage}" == "c_frozen_dp_production" || "${stage}" == "d_future_teacher_production" ]]; then
    echo "  shard_root=${out_dir}"
    for family in lr fb reverse stop sine cont; do
      echo "  shard_${family}_exists=$([[ -d "${out_dir}/${family}" ]] && echo true || echo false)"
    done
  fi

  static_file="$(mktemp)"
  if "${ROOT}/scripts/world_model/require_dataset_static_full_ready.sh" >"${static_file}" 2>&1; then
    echo "  a_static_full_ready=true"
  else
    echo "  a_static_full_ready=false"
    sed 's/^/  a_static_full_/' "${static_file}"
  fi
  rm -f "${static_file}"

  smoke_file="$(mktemp)"
  if "${ROOT}/scripts/world_model/require_dataset_class_smoke_approved.sh" "${smoke_stage}" >"${smoke_file}" 2>&1; then
    echo "  class_smoke_approved=true"
  else
    echo "  class_smoke_approved=false"
    sed 's/^/  class_smoke_/' "${smoke_file}"
  fi
  rm -f "${smoke_file}"

  validation_file="$(mktemp)"
  if "${ROOT}/scripts/world_model/validate_dataset_production_run.sh" "${stage}" >"${validation_file}" 2>&1; then
    sed 's/^/  production_validation_/' "${validation_file}"
  else
    sed 's/^/  production_validation_/' "${validation_file}"
  fi
  rm -f "${validation_file}"

  if [[ "${stage}" == "e_cosmos_predicted_production" ]]; then
    prereq_file="$(mktemp)"
    if "${ROOT}/scripts/world_model/require_dataset_cosmos_predicted_prereqs_ready.sh" >"${prereq_file}" 2>&1; then
      sed 's/^/  prereq_/' "${prereq_file}"
    else
      sed 's/^/  prereq_/' "${prereq_file}"
    fi
    rm -f "${prereq_file}"
  fi
done
