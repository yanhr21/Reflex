#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
REGISTRY="${REGISTRY:-${ROOT}/experiments/maniskill/data/active}"

echo "dataset_production_index_status_ok=true"
echo "registry=${REGISTRY}"
echo "read_only=true"
echo "submits_slurm=false"

stage_meta() {
  case "$1" in
    b_dynamic_production)
      echo "b_dynamic_production B_dynamic_rgb_observation 1000 false"
      ;;
    c_frozen_dp_production)
      echo "c_frozen_dp_production C_frozen_dp_dynamic_failure 500 false"
      ;;
    d_future_teacher_production)
      echo "d_future_teacher_production D_future_frame_cooperation_teacher 500 true"
      ;;
    e_cosmos_predicted_production)
      echo "e_cosmos_predicted_production E_cosmos_predicted_cooperation 100 false"
      ;;
  esac
}

for stage in \
  b_dynamic_production \
  c_frozen_dp_production \
  d_future_teacher_production \
  e_cosmos_predicted_production; do
  read -r index_name dataset_class target_count teacher_allowed < <(stage_meta "${stage}")
  index_dir="${REGISTRY}/${index_name}"
  samples_jsonl="${index_dir}/samples.jsonl"
  train_jsonl="${index_dir}/train_samples.jsonl"
  val_jsonl="${index_dir}/val_samples.jsonl"
  manifest="${index_dir}/index_manifest.txt"
  echo "[${stage}]"
  echo "  index_dir=${index_dir}"
  echo "  dataset_class=${dataset_class}"
  echo "  target_count=${target_count}"
  echo "  teacher_evidence_allowed_expected=${teacher_allowed}"
  echo "  index_dir_exists=$([[ -d "${index_dir}" ]] && echo true || echo false)"
  for path in "${manifest}" "${samples_jsonl}" "${train_jsonl}" "${val_jsonl}"; do
    label="$(basename "${path}")"
    echo "  ${label}=${path}"
    if [[ -f "${path}" ]]; then
      echo "  ${label}_exists=true"
      echo "  ${label}_lines=$(wc -l < "${path}" | tr -d ' ')"
    else
      echo "  ${label}_exists=false"
      echo "  ${label}_lines=0"
    fi
  done

  failures=0
  if [[ ! -f "${samples_jsonl}" ]]; then
    failures=$((failures + 1))
  else
    sample_lines="$(wc -l < "${samples_jsonl}" | tr -d ' ')"
    if [[ "${sample_lines}" -ge "${target_count}" ]]; then
      echo "  target_count_met=true"
    else
      echo "  target_count_met=false"
      failures=$((failures + 1))
    fi
    for needle in \
      "\"dataset_class\":\"${dataset_class}\"" \
      '"method_evidence_allowed":"false"' \
      "\"teacher_evidence_allowed\":\"${teacher_allowed}\"" \
      '"positive_dp_bc_allowed":"false"'; do
      missing="$(awk -v needle="${needle}" 'index($0, needle) == 0 {c++} END {print c+0}' "${samples_jsonl}")"
      label="$(printf '%s' "${needle}" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//; s/_$//')"
      echo "  ${label}_missing=${missing}"
      if [[ "${missing}" -ne 0 ]]; then
        failures=$((failures + 1))
      fi
    done
  fi
  if [[ -f "${manifest}" ]]; then
    if grep -qxF "shard_index=true" "${manifest}"; then
      echo "  shard_index=true"
    else
      echo "  shard_index=false"
    fi
    if grep -q '^run=' "${manifest}"; then
      sed -nE 's/^(run|output_dir|sample_count|train_count|val_count)=/  manifest_\1=/p' "${manifest}"
    fi
    if grep -qxF "teacher_evidence_allowed=${teacher_allowed}" "${manifest}"; then
      echo "  manifest_teacher_evidence_allowed_expected=true"
    else
      echo "  manifest_teacher_evidence_allowed_expected=false"
      failures=$((failures + 1))
    fi
    if grep -qxF "method_evidence_allowed=false" "${manifest}"; then
      echo "  manifest_method_evidence_allowed_false=true"
    else
      echo "  manifest_method_evidence_allowed_false=false"
      failures=$((failures + 1))
    fi
    if grep -qxF "positive_dp_bc_allowed=false" "${manifest}"; then
      echo "  manifest_positive_dp_bc_allowed_false=true"
    else
      echo "  manifest_positive_dp_bc_allowed_false=false"
      failures=$((failures + 1))
    fi
  fi
  if [[ "${failures}" -eq 0 ]]; then
    echo "  production_index_ready=true"
  else
    echo "  production_index_ready=false"
  fi
  echo "  failure_count=${failures}"
done
