#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAGE="${1:-${DATASET_STAGE:-a_static_full}}"

case "${STAGE}" in
  a_static_full|a_static_smoke)
    run_group="${RUN_GROUP:-static_rgb}"
    run_name="${RUN_NAME:-smoke05}"
    ;;
  b_dynamic_smoke)
    run_group="${RUN_GROUP:-dynamic_rgb}"
    run_name="${RUN_NAME:-smoke01}"
    ;;
  c_frozen_dp_smoke)
    run_group="${RUN_GROUP:-frozen_dp_dynamic}"
    run_name="${RUN_NAME:-smoke01}"
    ;;
  d_future_teacher_smoke)
    run_group="${RUN_GROUP:-future_teacher}"
    run_name="${RUN_NAME:-smoke01}"
    ;;
  e_cosmos_predicted_smoke)
    run_group="${RUN_GROUP:-cosmos_predicted}"
    run_name="${RUN_NAME:-smoke01}"
    ;;
  *)
    echo "dataset_class_smoke_approved=false"
    echo "stage=${STAGE}"
    echo "reason=unknown_stage"
    exit 60
    ;;
esac

matrix_review_md="${ROOT}/experiments/maniskill/runs/01_dataset/review/bcd_smoke_matrix_review_20260709_preinsert_lead8.md"
matrix_families=(lr_pos lr_neg fb_pos fb_neg reverse sine peg_disturb)

out_dir="${OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${run_group}/${run_name}}"

echo "dataset_class_smoke_approval_check=true"
echo "stage=${STAGE}"
echo "run=${run_group}/${run_name}"
echo "output_dir=${out_dir}"

case "${STAGE}" in
  b_dynamic_smoke|c_frozen_dp_smoke|d_future_teacher_smoke)
    echo "approval_scope=multi_motion_matrix"
    echo "review_request=${matrix_review_md}"
    if [[ ! -f "${matrix_review_md}" ]]; then
      echo "dataset_class_smoke_approved=false"
      echo "reason=matrix_review_missing"
      exit 62
    fi
    for family in "${matrix_families[@]}"; do
      family_out_dir="${ROOT}/experiments/maniskill/runs/01_dataset/${run_group}/smoke_${family}"
      echo "[smoke_${family}]"
      if ! RUN_GROUP="${run_group}" RUN_NAME="smoke_${family}" OUT_DIR="${family_out_dir}" REVIEW_MD="${matrix_review_md}" \
        "${ROOT}/scripts/world_model/require_dataset_smoke_approved.sh"; then
        echo "dataset_class_smoke_approved=false"
        echo "failed_family=${family}"
        exit 61
      fi
    done
    echo "dataset_class_smoke_approved=true"
    exit 0
    ;;
esac

if ! RUN_GROUP="${run_group}" RUN_NAME="${run_name}" OUT_DIR="${out_dir}" \
  "${ROOT}/scripts/world_model/require_dataset_smoke_approved.sh"; then
  echo "dataset_class_smoke_approved=false"
  exit 61
fi

echo "dataset_class_smoke_approved=true"
