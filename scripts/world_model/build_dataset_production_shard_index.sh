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
    trace_rel="trace/motion_trace.json"
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
    trace_rel="trace/future_teacher_trace.json"
    allowed_losses="adapter_residual,moving_frame_conditioning,phase_timing,relative_velocity_at_contact"
    disallowed_losses="deployed_method_success_claim,hidden_future_controller"
    target_count=500
    teacher_allowed=true
    registry_name="d_future_teacher_production"
    ;;
  *)
    echo "dataset_production_shard_index_built=false"
    echo "stage=${STAGE}"
    echo "reason=unsupported_stage"
    exit 60
    ;;
esac

out_dir="${OUT_DIR:-${ROOT}/experiments/maniskill/runs/01_dataset/${run_group}/${run_name}}"
index_dir="${REGISTRY}/${registry_name}"

echo "dataset_production_shard_index_build=true"
echo "stage=${STAGE}"
echo "run=${run_group}/${run_name}"
echo "output_dir=${out_dir}"
echo "index_dir=${index_dir}"
echo "target_count=${target_count}"
echo "val_mod=${VAL_MOD}"

validation_file="$(mktemp)"
if ! "${ROOT}/scripts/world_model/validate_dataset_production_run.sh" "${STAGE}" >"${validation_file}" 2>&1; then
  echo "dataset_production_shard_index_built=false"
  echo "reason=production_validation_failed"
  sed 's/^/validation_/' "${validation_file}"
  rm -f "${validation_file}"
  exit 70
fi
rm -f "${validation_file}"

mapfile -t videos < <(find "${out_dir}" -mindepth 3 -maxdepth 3 -type f -path '*/videos/*.mp4' | sort)
if [[ "${#videos[@]}" -lt "${target_count}" ]]; then
  echo "dataset_production_shard_index_built=false"
  echo "reason=video_count_below_target"
  echo "video_count=${#videos[@]}"
  exit 71
fi

mkdir -p "${index_dir}"

samples_tsv="${index_dir}/samples.tsv"
train_tsv="${index_dir}/train_samples.tsv"
val_tsv="${index_dir}/val_samples.tsv"
samples_jsonl="${index_dir}/samples.jsonl"
train_jsonl="${index_dir}/train_samples.jsonl"
val_jsonl="${index_dir}/val_samples.jsonl"
index_manifest="${index_dir}/index_manifest.txt"

{
  printf 'sample_id\tsplit\tdataset_class\tdataset_role\tstage\trun_group\trun_name\tvideo\ttrace\tsummary\tmanifest\tallowed_losses\tdisallowed_losses\tmethod_evidence_allowed\tteacher_evidence_allowed\tpositive_dp_bc_allowed\treplaces_legacy_bootstrap\n'
  idx=0
  for video in "${videos[@]}"; do
    shard_dir="$(dirname "$(dirname "${video}")")"
    shard_name="$(basename "${shard_dir}")"
    video_name="$(basename "${video}" .mp4)"
    sample_id="${shard_name}_${video_name}"
    split="train"
    if (( idx % VAL_MOD == 0 )); then
      split="val"
    fi
    trace="${shard_dir}/${trace_rel}"
    summary="${shard_dir}/summary.json"
    manifest="${shard_dir}/manifest.txt"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfalse\t%s\tfalse\ttrue\n' \
      "${sample_id}" "${split}" "${dataset_class}" "${dataset_role}" "${STAGE}" \
      "${run_group}/${run_name}" "${shard_name}" "${video}" "${trace}" "${summary}" \
      "${manifest}" "${allowed_losses}" "${disallowed_losses}" "${teacher_allowed}"
    idx=$((idx + 1))
  done
} > "${samples_tsv}"

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
  echo "shard_index=true"
} > "${index_manifest}"

cat "${index_manifest}"
echo "dataset_production_shard_index_built=true"
