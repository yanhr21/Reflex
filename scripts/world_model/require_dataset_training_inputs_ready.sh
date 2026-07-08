#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
MODE="${1:-${DATASET_TRAINING_MODE:-full_joint}}"
REGISTRY="${REGISTRY:-${ROOT}/experiments/maniskill/data/active}"
SMOKE_OUT_DIR="${SMOKE_OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/static_rgb/smoke05}"
FULL_OUT_DIR="${FULL_OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/static_rgb/full01}"

case "${MODE}" in
  diagnostic_b_bootstrap|full_joint)
    ;;
  *)
    echo "dataset_training_inputs_ready=false"
    echo "mode=${MODE}"
    echo "reason=unknown_mode"
    exit 50
    ;;
esac

echo "dataset_training_inputs_ready_check=true"
echo "mode=${MODE}"
echo "read_only=true"
echo "submits_slurm=false"

a_h5="${REGISTRY}/a_static/official_state_pd_ee_delta_pose.h5"
a_json="${REGISTRY}/a_static/official_state_pd_ee_delta_pose.json"
legacy_paths="${REGISTRY}/legacy_733/fix3_h5_paths_canonical.txt"
b_dir="${REGISTRY}/b_dynamic_legacy_bootstrap"
train_jsonl="${b_dir}/train_samples.jsonl"
val_jsonl="${b_dir}/val_samples.jsonl"

failures=0
require_file() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    echo "${label}_exists=true"
    echo "${label}=${path}"
  else
    echo "${label}_exists=false"
    echo "${label}=${path}"
    failures=$((failures + 1))
  fi
}

require_line_count() {
  local label="$1"
  local path="$2"
  local expected="$3"
  local actual=0
  if [[ -f "${path}" ]]; then
    actual="$(wc -l < "${path}" | tr -d ' ')"
  fi
  echo "${label}_lines=${actual}"
  echo "${label}_expected_lines=${expected}"
  if [[ "${actual}" != "${expected}" ]]; then
    failures=$((failures + 1))
  fi
}

require_no_missing_field() {
  local label="$1"
  local path="$2"
  local needle="$3"
  local missing=0
  if [[ -f "${path}" ]]; then
    missing="$(awk -v needle="${needle}" 'index($0, needle) == 0 {c++} END {print c+0}' "${path}")"
  else
    missing=1
  fi
  echo "${label}_missing=${missing}"
  if [[ "${missing}" -ne 0 ]]; then
    failures=$((failures + 1))
  fi
}

require_production_index() {
  local stage="$1"
  local index_name="$2"
  local dataset_class="$3"
  local target_count="$4"
  local teacher_allowed="$5"
  local index_dir="${REGISTRY}/${index_name}"
  local samples_jsonl="${index_dir}/samples.jsonl"
  local train_jsonl="${index_dir}/train_samples.jsonl"
  local val_jsonl="${index_dir}/val_samples.jsonl"
  local index_manifest="${index_dir}/index_manifest.txt"
  echo "${stage}_index_dir=${index_dir}"
  require_file "${stage}_index_manifest" "${index_manifest}"
  require_file "${stage}_index_samples_jsonl" "${samples_jsonl}"
  require_file "${stage}_index_train_jsonl" "${train_jsonl}"
  require_file "${stage}_index_val_jsonl" "${val_jsonl}"
  local sample_count=0
  local train_count=0
  local val_count=0
  if [[ -f "${samples_jsonl}" ]]; then
    sample_count="$(wc -l < "${samples_jsonl}" | tr -d ' ')"
  fi
  if [[ -f "${train_jsonl}" ]]; then
    train_count="$(wc -l < "${train_jsonl}" | tr -d ' ')"
  fi
  if [[ -f "${val_jsonl}" ]]; then
    val_count="$(wc -l < "${val_jsonl}" | tr -d ' ')"
  fi
  echo "${stage}_index_sample_lines=${sample_count}"
  echo "${stage}_index_train_lines=${train_count}"
  echo "${stage}_index_val_lines=${val_count}"
  echo "${stage}_index_target_count=${target_count}"
  if [[ "${sample_count}" -lt "${target_count}" ]]; then
    failures=$((failures + 1))
  fi
  require_no_missing_field "${stage}_index_dataset_class" "${samples_jsonl}" "\"dataset_class\":\"${dataset_class}\""
  require_no_missing_field "${stage}_index_method_evidence" "${samples_jsonl}" '"method_evidence_allowed":"false"'
  require_no_missing_field "${stage}_index_teacher_evidence" "${samples_jsonl}" "\"teacher_evidence_allowed\":\"${teacher_allowed}\""
  require_no_missing_field "${stage}_index_positive_dp_bc" "${samples_jsonl}" '"positive_dp_bc_allowed":"false"'
}

require_file "a_static_h5" "${a_h5}"
require_file "a_static_json" "${a_json}"
require_file "legacy_733_paths" "${legacy_paths}"
require_line_count "legacy_733_paths" "${legacy_paths}" "733"
require_file "b_train_jsonl" "${train_jsonl}"
require_line_count "b_train_jsonl" "${train_jsonl}" "900"
require_file "b_val_jsonl" "${val_jsonl}"
require_line_count "b_val_jsonl" "${val_jsonl}" "100"

for split in train val; do
  if [[ "${split}" == "train" ]]; then
    jsonl="${train_jsonl}"
  else
    jsonl="${val_jsonl}"
  fi
  require_no_missing_field "b_${split}_dataset_class" "${jsonl}" '"dataset_class":"B_dynamic_rgb_observation"'
  require_no_missing_field "b_${split}_dataset_role" "${jsonl}" '"dataset_role":"legacy_bootstrap"'
  require_no_missing_field "b_${split}_split" "${jsonl}" "\"split\":\"${split}\""
  require_no_missing_field "b_${split}_allowed_losses" "${jsonl}" '"allowed_losses":"cosmos_dynamic_future,target_frame_readout,trajectory_consistency,uncertainty,diagnostic_ablation"'
  require_no_missing_field "b_${split}_disallowed_losses" "${jsonl}" '"disallowed_losses":"positive_dp_bc,final_method_success,active_new_production_success"'
  require_no_missing_field "b_${split}_method_evidence" "${jsonl}" '"method_evidence_allowed":"false"'
  require_no_missing_field "b_${split}_positive_dp_bc" "${jsonl}" '"positive_dp_bc_allowed":"false"'
  require_no_missing_field "b_${split}_replaces_new_production" "${jsonl}" '"replaces_new_production":"false"'
done

if [[ "${MODE}" == "diagnostic_b_bootstrap" ]]; then
  if [[ "${failures}" -eq 0 ]]; then
    echo "dataset_training_inputs_ready=true"
    echo "allowed_scope=diagnostic_b_bootstrap_only"
    echo "positive_dp_bc_allowed=false"
    echo "final_method_evidence_allowed=false"
    echo "full_joint_training_allowed=false"
    echo "failure_count=0"
    exit 0
  fi
  echo "dataset_training_inputs_ready=false"
  echo "reason=diagnostic_bootstrap_inputs_invalid"
  echo "failure_count=${failures}"
  exit 61
fi

require_file "stage1_smoke_summary" "${SMOKE_OUT_DIR}/summary.json"
require_file "stage1_smoke_video" "${SMOKE_OUT_DIR}/0.mp4"
if [[ -f "${SMOKE_OUT_DIR}/human_review_approved.txt" ]] && \
  grep -qx 'approved=true' "${SMOKE_OUT_DIR}/human_review_approved.txt"; then
  echo "stage1_smoke_approved=true"
else
  echo "stage1_smoke_approved=false"
  failures=$((failures + 1))
fi
static_full_file="$(mktemp)"
if FULL_OUT_DIR="${FULL_OUT_DIR}" "${ROOT}/scripts/world_model/require_dataset_static_full_ready.sh" >"${static_full_file}" 2>&1; then
  echo "a_static_full_ready=true"
  sed 's/^/a_static_full_/' "${static_full_file}"
else
  echo "a_static_full_ready=false"
  sed 's/^/a_static_full_/' "${static_full_file}"
  failures=$((failures + 1))
fi
rm -f "${static_full_file}"

for production_stage in \
  b_dynamic_production \
  c_frozen_dp_production \
  d_future_teacher_production \
  e_cosmos_predicted_production; do
  validation_file="$(mktemp)"
  if "${ROOT}/scripts/world_model/validate_dataset_production_run.sh" "${production_stage}" >"${validation_file}" 2>&1; then
    sed "s/^/${production_stage}_/" "${validation_file}"
  else
    sed "s/^/${production_stage}_/" "${validation_file}"
    failures=$((failures + 1))
  fi
  rm -f "${validation_file}"
done

require_production_index b_dynamic_production b_dynamic_production B_dynamic_rgb_observation 1000 false
require_production_index c_frozen_dp_production c_frozen_dp_production C_frozen_dp_dynamic_failure 500 false
require_production_index d_future_teacher_production d_future_teacher_production D_future_frame_cooperation_teacher 500 true
require_production_index e_cosmos_predicted_production e_cosmos_predicted_production E_cosmos_predicted_cooperation 100 false

if [[ "${failures}" -eq 0 ]]; then
  echo "dataset_training_inputs_ready=true"
  echo "allowed_scope=full_joint_training"
  echo "failure_count=0"
  exit 0
fi

echo "dataset_training_inputs_ready=false"
echo "reason=full_joint_inputs_incomplete"
echo "failure_count=${failures}"
exit 62
