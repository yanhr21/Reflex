#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
REGISTRY="${REGISTRY:-${ROOT}/experiments/maniskill/data/active}"
SMOKE_RUN_GROUP="${SMOKE_RUN_GROUP:-static_rgb}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-smoke05}"
SMOKE_OUT_DIR="${SMOKE_OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${SMOKE_RUN_GROUP}/${SMOKE_RUN_NAME}}"
FULL_OUT_DIR="${FULL_OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/static_rgb/full01}"

echo "dataset_training_inputs_status_ok=true"
echo "registry=${REGISTRY}"
echo "read_only=true"
echo "submits_slurm=false"
echo "bcd_shard_plan=${ROOT}/scripts/world_model/dataset_bcd_production_shard_plan.sh"
echo "bcd_shard_launcher=${ROOT}/scripts/slurm/launch_dataset_bcd_production_shards_tmux.sh"
echo "bcd_shard_index_builder=${ROOT}/scripts/world_model/build_dataset_production_shard_index.sh"

exists_report() {
  local label="$1"
  local path="$2"
  echo "${label}=${path}"
  if [[ -e "${path}" ]]; then
    echo "${label}_exists=true"
    echo "${label}_target=$(readlink -f "${path}")"
  else
    echo "${label}_exists=false"
  fi
}

line_count_report() {
  local label="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    echo "${label}_lines=$(wc -l < "${path}" | tr -d ' ')"
  else
    echo "${label}_lines=0"
  fi
}

missing_substring_count() {
  local path="$1"
  local needle="$2"
  if [[ -f "${path}" ]]; then
    awk -v needle="${needle}" 'index($0, needle) == 0 {c++} END {print c+0}' "${path}"
  else
    echo 0
  fi
}

echo "[a_static_official]"
a_h5="${REGISTRY}/a_static/official_state_pd_ee_delta_pose.h5"
a_json="${REGISTRY}/a_static/official_state_pd_ee_delta_pose.json"
exists_report "  official_h5" "${a_h5}"
exists_report "  official_json" "${a_json}"
echo "  dataset_class=A_static_expert"
echo "  state_action_source_available=$([[ -e "${a_h5}" && -e "${a_json}" ]] && echo true || echo false)"
echo "  allowed_losses_now=dp_bc_from_state,static_phase_extraction"
echo "  cosmos_rgb_losses_allowed_after_full_rgb=true"

echo "[a_static_rgb]"
exists_report "  smoke_summary" "${SMOKE_OUT_DIR}/summary.json"
exists_report "  smoke_video" "${SMOKE_OUT_DIR}/0.mp4"
if [[ -f "${SMOKE_OUT_DIR}/human_review_approved.txt" ]] && \
  grep -qx 'approved=true' "${SMOKE_OUT_DIR}/human_review_approved.txt"; then
  echo "  smoke_approved=true"
else
  echo "  smoke_approved=false"
fi
exists_report "  full_summary" "${FULL_OUT_DIR}/summary.json"
static_full_status_file="$(mktemp)"
if FULL_OUT_DIR="${FULL_OUT_DIR}" "${ROOT}/scripts/world_model/require_dataset_static_full_ready.sh" >"${static_full_status_file}" 2>&1; then
  echo "  full_rgb_ready=true"
else
  echo "  full_rgb_ready=false"
fi
sed 's/^/  full_rgb_gate_/' "${static_full_status_file}"
rm -f "${static_full_status_file}"

echo "[legacy_733_reference]"
legacy_paths="${REGISTRY}/legacy_733/fix3_h5_paths_canonical.txt"
legacy_manifest="${REGISTRY}/legacy_733/manifest.json"
exists_report "  fix3_h5_paths" "${legacy_paths}"
line_count_report "  fix3_h5_paths" "${legacy_paths}"
exists_report "  manifest_json" "${legacy_manifest}"
echo "  allowed_use=context_ablation_early_readout_reference"
echo "  positive_dp_bc_allowed=false_without_revalidation"
echo "  final_method_evidence_allowed=false_without_revalidation"

echo "[b_dynamic_legacy_bootstrap]"
b_dir="${REGISTRY}/b_dynamic_legacy_bootstrap"
train_jsonl="${b_dir}/train_samples.jsonl"
val_jsonl="${b_dir}/val_samples.jsonl"
all_jsonl="${b_dir}/samples.jsonl"
manifest="${b_dir}/manifest.txt"
split_counts="${b_dir}/split_scenario_counts.txt"
exists_report "  manifest" "${manifest}"
exists_report "  train_jsonl" "${train_jsonl}"
line_count_report "  train_jsonl" "${train_jsonl}"
exists_report "  val_jsonl" "${val_jsonl}"
line_count_report "  val_jsonl" "${val_jsonl}"
exists_report "  samples_jsonl" "${all_jsonl}"
line_count_report "  samples_jsonl" "${all_jsonl}"
exists_report "  split_scenario_counts" "${split_counts}"
if [[ -f "${split_counts}" ]]; then
  echo "  split_scenario_counts_begin"
  sed 's/^/    /' "${split_counts}"
  echo "  split_scenario_counts_end"
fi

failures=0
check_jsonl_field() {
  local label="$1"
  local path="$2"
  local needle="$3"
  local missing
  missing="$(missing_substring_count "${path}" "${needle}")"
  echo "  ${label}_missing=${missing}"
  if [[ "${missing}" -ne 0 ]]; then
    failures=$((failures + 1))
  fi
}

check_jsonl_field "train_dataset_class" "${train_jsonl}" '"dataset_class":"B_dynamic_rgb_observation"'
check_jsonl_field "train_dataset_role" "${train_jsonl}" '"dataset_role":"legacy_bootstrap"'
check_jsonl_field "train_split" "${train_jsonl}" '"split":"train"'
check_jsonl_field "train_allowed_losses" "${train_jsonl}" '"allowed_losses":"cosmos_dynamic_future,target_frame_readout,trajectory_consistency,uncertainty,diagnostic_ablation"'
check_jsonl_field "train_disallowed_losses" "${train_jsonl}" '"disallowed_losses":"positive_dp_bc,final_method_success,active_new_production_success"'
check_jsonl_field "train_method_evidence" "${train_jsonl}" '"method_evidence_allowed":"false"'
check_jsonl_field "train_positive_dp_bc" "${train_jsonl}" '"positive_dp_bc_allowed":"false"'
check_jsonl_field "train_replaces_new_production" "${train_jsonl}" '"replaces_new_production":"false"'

check_jsonl_field "val_dataset_class" "${val_jsonl}" '"dataset_class":"B_dynamic_rgb_observation"'
check_jsonl_field "val_dataset_role" "${val_jsonl}" '"dataset_role":"legacy_bootstrap"'
check_jsonl_field "val_split" "${val_jsonl}" '"split":"val"'
check_jsonl_field "val_allowed_losses" "${val_jsonl}" '"allowed_losses":"cosmos_dynamic_future,target_frame_readout,trajectory_consistency,uncertainty,diagnostic_ablation"'
check_jsonl_field "val_disallowed_losses" "${val_jsonl}" '"disallowed_losses":"positive_dp_bc,final_method_success,active_new_production_success"'
check_jsonl_field "val_method_evidence" "${val_jsonl}" '"method_evidence_allowed":"false"'
check_jsonl_field "val_positive_dp_bc" "${val_jsonl}" '"positive_dp_bc_allowed":"false"'
check_jsonl_field "val_replaces_new_production" "${val_jsonl}" '"replaces_new_production":"false"'

train_lines="$(wc -l < "${train_jsonl}" 2>/dev/null | tr -d ' ' || echo 0)"
val_lines="$(wc -l < "${val_jsonl}" 2>/dev/null | tr -d ' ' || echo 0)"
if [[ "${train_lines}" == "900" && "${val_lines}" == "100" && "${failures}" -eq 0 ]]; then
  echo "  bootstrap_training_index_valid=true"
else
  echo "  bootstrap_training_index_valid=false"
fi
echo "  validation_failure_count=${failures}"
echo "  allowed_use=bootstrap_cosmos_dynamic_future,target_frame_readout,trajectory_consistency,uncertainty,diagnostic_ablation"
echo "  positive_dp_bc_allowed=false"
echo "  final_method_evidence_allowed=false"
echo "  replaces_new_b_c_d_e_production=false"

echo "[production_training_indexes]"
"${ROOT}/scripts/world_model/dataset_production_index_status.sh" | sed 's/^/  /'

echo "[training_readiness]"
static_full_ready_file="$(mktemp)"
if FULL_OUT_DIR="${FULL_OUT_DIR}" "${ROOT}/scripts/world_model/require_dataset_static_full_ready.sh" >"${static_full_ready_file}" 2>&1; then
  echo "  a_full_rgb_ready=true"
else
  echo "  a_full_rgb_ready=false"
fi
rm -f "${static_full_ready_file}"
if [[ "${train_lines}" == "900" && "${val_lines}" == "100" && "${failures}" -eq 0 ]]; then
  echo "  b_bootstrap_ready_for_diagnostic=true"
else
  echo "  b_bootstrap_ready_for_diagnostic=false"
fi
full_joint_file="$(mktemp)"
if "${ROOT}/scripts/world_model/require_dataset_training_inputs_ready.sh" full_joint >"${full_joint_file}" 2>&1; then
  echo "  full_joint_training_ready=true"
else
  echo "  full_joint_training_ready=false"
fi
sed 's/^/  full_joint_gate_/' "${full_joint_file}"
rm -f "${full_joint_file}"

echo "[training_guards]"
for mode in diagnostic_b_bootstrap full_joint; do
  echo "  [${mode}]"
  status_file="$(mktemp)"
  if "${ROOT}/scripts/world_model/require_dataset_training_inputs_ready.sh" "${mode}" >"${status_file}" 2>&1; then
    sed 's/^/    /' "${status_file}"
  else
    sed 's/^/    /' "${status_file}"
  fi
  rm -f "${status_file}"
done
