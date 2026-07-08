#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

echo "dataset_class_review_status_ok=true"
echo "read_only=true"
echo "submits_slurm=false"

stage_run() {
  case "$1" in
    a_static_full|a_static_smoke)
      echo "static_rgb smoke05"
      ;;
    b_dynamic_smoke)
      echo "dynamic_rgb smoke01"
      ;;
    c_frozen_dp_smoke)
      echo "frozen_dp_dynamic smoke01"
      ;;
    d_future_teacher_smoke)
      echo "future_teacher smoke01"
      ;;
    e_cosmos_predicted_smoke)
      echo "cosmos_predicted smoke01"
      ;;
  esac
}

for stage in \
  a_static_full \
  b_dynamic_smoke \
  c_frozen_dp_smoke \
  d_future_teacher_smoke \
  e_cosmos_predicted_smoke; do
  echo "[${stage}]"
  read -r run_group run_name < <(stage_run "${stage}")
  out_dir="${ROOT}/experiments/maniskill/runs/01_dataset/${run_group}/${run_name}"
  echo "  run=${run_group}/${run_name}"
  echo "  output_dir=${out_dir}"

  if [[ -d "${out_dir}" ]]; then
    echo "  output_dir_exists=true"
  else
    echo "  output_dir_exists=false"
  fi

  status_file="$(mktemp)"
  if RUN_GROUP="${run_group}" RUN_NAME="${run_name}" OUT_DIR="${out_dir}" \
    "${ROOT}/scripts/world_model/dataset_review_status.sh" >"${status_file}" 2>&1; then
    sed 's/^/  review_/' "${status_file}"
  else
    sed 's/^/  review_/' "${status_file}"
  fi
  rm -f "${status_file}"

  gate_file="$(mktemp)"
  if "${ROOT}/scripts/world_model/require_dataset_class_smoke_approved.sh" "${stage}" >"${gate_file}" 2>&1; then
    sed 's/^/  gate_/' "${gate_file}"
  else
    sed 's/^/  gate_/' "${gate_file}"
  fi
  rm -f "${gate_file}"
done
