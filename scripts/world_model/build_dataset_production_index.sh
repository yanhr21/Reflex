#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
STAGE="${1:-${DATASET_PRODUCTION_STAGE:-b_dynamic_production}}"
REGISTRY="${REGISTRY:-${ROOT}/experiments/maniskill/data/active}"
VAL_MOD="${VAL_MOD:-10}"

case "${STAGE}" in
  b_dynamic_production)
    run_group="dynamic_rgb"
    run_name="prod01"
    dataset_class="B_dynamic_rgb_observation"
    dataset_role="new_production"
    trace_rel="trace/demo_action_trace.json"
    allowed_losses="cosmos_dynamic_future,target_frame_readout,trajectory_consistency,uncertainty"
    disallowed_losses="positive_dp_bc_from_failed_actions,final_method_evidence"
    target_count=1000
    teacher_allowed=false
    registry_name="b_dynamic_production"
    ;;
  c_frozen_dp_production)
    run_group="frozen_dp_dynamic"
    run_name="prod01"
    dataset_class="C_frozen_dp_dynamic_failure"
    dataset_role="new_production_negative"
    trace_rel="trace/frozen_dp_trace.json"
    allowed_losses="negative_classification,discrepancy,infeasible_no_progress,contrastive"
    disallowed_losses="positive_dp_bc_from_failed_actions,target_assisted_success"
    target_count=500
    teacher_allowed=false
    registry_name="c_frozen_dp_production"
    ;;
  d_future_teacher_production)
    run_group="future_teacher"
    run_name="prod01"
    dataset_class="D_future_frame_cooperation_teacher"
    dataset_role="new_production_teacher"
    trace_rel="trace/demo_action_trace.json"
    allowed_losses="adapter_residual,moving_frame_conditioning,phase_timing,relative_velocity_at_contact"
    disallowed_losses="deployed_method_success_claim,hidden_future_controller"
    target_count=500
    teacher_allowed=true
    registry_name="d_future_teacher_production"
    ;;
  e_cosmos_predicted_production)
    run_group="cosmos_predicted"
    run_name="prod01"
    dataset_class="E_cosmos_predicted_cooperation"
    dataset_role="new_production_cosmos_predicted"
    trace_rel="trace/cosmos_predicted_trace.json"
    allowed_losses="adapter_robustness,uncertainty_conditioned_control,live_method_evaluation"
    disallowed_losses="hidden_ground_truth_future,target_assisted_success"
    target_count=100
    teacher_allowed=false
    registry_name="e_cosmos_predicted_production"
    ;;
  *)
    echo "dataset_production_index_built=false"
    echo "stage=${STAGE}"
    echo "reason=unknown_stage"
    exit 60
    ;;
esac

out_dir="${OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${run_group}/${run_name}}"
summary="${out_dir}/summary.json"
manifest="${out_dir}/manifest.txt"
trace="${out_dir}/${trace_rel}"
videos_dir="${out_dir}/videos"
index_dir="${REGISTRY}/${registry_name}"

echo "dataset_production_index_build=true"
echo "stage=${STAGE}"
echo "run=${run_group}/${run_name}"
echo "output_dir=${out_dir}"
echo "index_dir=${index_dir}"
echo "target_count=${target_count}"
echo "val_mod=${VAL_MOD}"

validation_file="$(mktemp)"
if ! "${ROOT}/scripts/world_model/validate_dataset_production_run.sh" "${STAGE}" >"${validation_file}" 2>&1; then
  echo "dataset_production_index_built=false"
  echo "reason=production_validation_failed"
  sed 's/^/validation_/' "${validation_file}"
  rm -f "${validation_file}"
  exit 70
fi
rm -f "${validation_file}"

failures=0
require_summary_pattern() {
  local label="$1"
  local pattern="$2"
  if grep -qE "${pattern}" "${summary}"; then
    echo "${label}=true"
  else
    echo "${label}=false"
    failures=$((failures + 1))
  fi
}

require_manifest_line() {
  local label="$1"
  local expected="$2"
  if grep -qxF "${expected}" "${manifest}"; then
    echo "${label}=true"
  else
    echo "${label}=false"
    failures=$((failures + 1))
  fi
}

require_summary_pattern source_summary_dataset_class "\"dataset_class\"[[:space:]]*:[[:space:]]*\"${dataset_class}\""
require_summary_pattern source_summary_status_production_complete '"status"[[:space:]]*:[[:space:]]*"production_complete"'
require_summary_pattern source_summary_dataset_smoke_only_false '"dataset_smoke_only"[[:space:]]*:[[:space:]]*false'
require_summary_pattern source_summary_method_evidence_false '"method_evidence_allowed"[[:space:]]*:[[:space:]]*false'
require_summary_pattern source_summary_teacher_evidence_expected "\"teacher_evidence_allowed\"[[:space:]]*:[[:space:]]*${teacher_allowed}"
require_summary_pattern source_summary_positive_policy_data_false '"positive_policy_data_allowed"[[:space:]]*:[[:space:]]*false'
require_summary_pattern source_summary_output_dir "\"output_dir\"[[:space:]]*:[[:space:]]*\"${out_dir}\""

require_manifest_line source_manifest_phase "phase=01_dataset"
require_manifest_line source_manifest_dataset_class "dataset_class=${dataset_class}"
require_manifest_line source_manifest_run_group "run_group=${run_group}"
require_manifest_line source_manifest_run_name "run_name=${run_name}"
require_manifest_line source_manifest_output_dir "output_dir=${out_dir}"
require_manifest_line source_manifest_method_evidence "method_evidence_allowed=false"
require_manifest_line source_manifest_teacher_evidence "teacher_evidence_allowed=${teacher_allowed}"
require_manifest_line source_manifest_forbidden_state_intervention "forbidden_state_intervention_expected=false"

if [[ "${failures}" -ne 0 ]]; then
  echo "dataset_production_index_built=false"
  echo "reason=source_summary_or_manifest_mismatch"
  echo "failure_count=${failures}"
  exit 72
fi

mkdir -p "${index_dir}"

samples_tsv="${index_dir}/samples.tsv"
train_tsv="${index_dir}/train_samples.tsv"
val_tsv="${index_dir}/val_samples.tsv"
samples_jsonl="${index_dir}/samples.jsonl"
train_jsonl="${index_dir}/train_samples.jsonl"
val_jsonl="${index_dir}/val_samples.jsonl"
index_manifest="${index_dir}/index_manifest.txt"

tmp_samples="$(mktemp)"
trap 'rm -f "${tmp_samples}"' EXIT

{
  printf 'sample_id\tsplit\tdataset_class\tdataset_role\tstage\trun_group\trun_name\tvideo\ttrace\tsummary\tmanifest\tallowed_losses\tdisallowed_losses\tmethod_evidence_allowed\tteacher_evidence_allowed\tpositive_dp_bc_allowed\treplaces_legacy_bootstrap\n'
  find "${videos_dir}" -type f -name '*.mp4' | sort | awk \
    -v val_mod="${VAL_MOD}" \
    -v dataset_class="${dataset_class}" \
    -v dataset_role="${dataset_role}" \
    -v stage="${STAGE}" \
    -v run_group="${run_group}" \
    -v run_name="${run_name}" \
    -v trace="${trace}" \
    -v summary="${summary}" \
    -v manifest="${manifest}" \
    -v allowed_losses="${allowed_losses}" \
    -v disallowed_losses="${disallowed_losses}" \
    -v teacher_allowed="${teacher_allowed}" '
      {
        video=$0
        n=split(video, parts, "/")
        sample_id=parts[n]
        sub(/\.mp4$/, "", sample_id)
        split_name=(i % val_mod == 0) ? "val" : "train"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfalse\t%s\tfalse\ttrue\n", sample_id, split_name, dataset_class, dataset_role, stage, run_group, run_name, video, trace, summary, manifest, allowed_losses, disallowed_losses, teacher_allowed
        i++
      }
    '
} > "${tmp_samples}"

mv "${tmp_samples}" "${samples_tsv}"
awk -F'\t' 'NR==1 || $2=="train"' "${samples_tsv}" > "${train_tsv}"
awk -F'\t' 'NR==1 || $2=="val"' "${samples_tsv}" > "${val_tsv}"

tsv_to_jsonl() {
  local input="$1"
  local output="$2"
  awk -F'\t' '
    NR==1 {
      for (i=1; i<=NF; i++) header[i]=$i
      next
    }
    {
      printf "{"
      for (i=1; i<=NF; i++) {
        value=$i
        gsub(/\\/,"\\\\",value)
        gsub(/"/,"\\\"",value)
        printf "%s\"%s\":\"%s\"", (i==1 ? "" : ","), header[i], value
      }
      printf "}\n"
    }
  ' "${input}" > "${output}"
}

tsv_to_jsonl "${samples_tsv}" "${samples_jsonl}"
tsv_to_jsonl "${train_tsv}" "${train_jsonl}"
tsv_to_jsonl "${val_tsv}" "${val_jsonl}"

sample_count="$(($(wc -l < "${samples_tsv}") - 1))"
train_count="$(($(wc -l < "${train_tsv}") - 1))"
val_count="$(($(wc -l < "${val_tsv}") - 1))"

{
  echo "timestamp=$(date -Is)"
  echo "stage=${STAGE}"
  echo "dataset_class=${dataset_class}"
  echo "dataset_role=${dataset_role}"
  echo "run=${run_group}/${run_name}"
  echo "output_dir=${out_dir}"
  echo "summary=${summary}"
  echo "manifest=${manifest}"
  echo "trace=${trace}"
  echo "videos_dir=${videos_dir}"
  echo "samples_tsv=${samples_tsv}"
  echo "train_tsv=${train_tsv}"
  echo "val_tsv=${val_tsv}"
  echo "samples_jsonl=${samples_jsonl}"
  echo "train_jsonl=${train_jsonl}"
  echo "val_jsonl=${val_jsonl}"
  echo "sample_count=${sample_count}"
  echo "train_count=${train_count}"
  echo "val_count=${val_count}"
  echo "target_count=${target_count}"
  echo "val_mod=${VAL_MOD}"
  echo "method_evidence_allowed=false"
  echo "teacher_evidence_allowed=${teacher_allowed}"
  echo "positive_dp_bc_allowed=false"
  echo "replaces_legacy_bootstrap=true"
} > "${index_manifest}"

cat "${index_manifest}"

if [[ "${sample_count}" -lt "${target_count}" ]]; then
  echo "dataset_production_index_built=false"
  echo "reason=sample_count_below_target"
  exit 71
fi

echo "dataset_production_index_built=true"
