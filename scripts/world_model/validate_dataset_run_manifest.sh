#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-}"
ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "dataset_run_manifest_valid=false"
  echo "reason=run_dir_required"
  echo "usage=validate_dataset_run_manifest.sh <run_dir>"
  exit 20
fi

case "${RUN_DIR}" in
  /*) ;;
  *) RUN_DIR="${ROOT}/${RUN_DIR}" ;;
esac

MANIFEST="${RUN_DIR}/manifest.txt"
SUMMARY="${RUN_DIR}/summary.json"
CORRECTIONS="${RUN_DIR}/manifest_corrections.txt"

echo "dataset_run_manifest_validation_ok=true"
echo "run_dir=${RUN_DIR}"
echo "manifest=${MANIFEST}"
echo "summary=${SUMMARY}"
echo "manifest_corrections=${CORRECTIONS}"
echo "read_only=true"
echo "submits_slurm=false"

failures=0

require_file() {
  local label="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    echo "${label}_exists=true"
  else
    echo "${label}_exists=false"
    failures=$((failures + 1))
  fi
}

require_key() {
  local key="$1"
  if grep -qE "^${key}=" "${MANIFEST}" || \
    { [[ -f "${CORRECTIONS}" ]] && grep -qE "^${key}=" "${CORRECTIONS}"; } || \
    grep -qE "\"${key}\"[[:space:]]*:" "${SUMMARY}" 2>/dev/null; then
    echo "required_${key}=true"
  else
    echo "required_${key}=false"
    failures=$((failures + 1))
  fi
}

require_manifest_value() {
  local key="$1"
  local expected="$2"
  if grep -qx "${key}=${expected}" "${MANIFEST}" || \
    { [[ -f "${CORRECTIONS}" ]] && grep -qx "${key}=${expected}" "${CORRECTIONS}"; }; then
    echo "${key}_is_${expected}=true"
  else
    echo "${key}_is_${expected}=false"
    failures=$((failures + 1))
  fi
}

require_summary_bool() {
  local key="$1"
  local expected="$2"
  if grep -qE "\"${key}\"[[:space:]]*:[[:space:]]*${expected}" "${SUMMARY}"; then
    echo "summary_${key}_is_${expected}=true"
  else
    echo "summary_${key}_is_${expected}=false"
    failures=$((failures + 1))
  fi
}

require_file "manifest" "${MANIFEST}"
require_file "summary" "${SUMMARY}"

if [[ ! -f "${MANIFEST}" || ! -f "${SUMMARY}" ]]; then
  echo "dataset_run_manifest_valid=false"
  echo "failure_count=${failures}"
  exit 60
fi

for key in \
  phase \
  dataset_class \
  run_group \
  run_name \
  job_id \
  step_id \
  node_list \
  output_dir \
  log_file \
  source_paths \
  controller \
  action_contract \
  rgb_required \
  human_review_required \
  large_scale_production_allowed \
  method_evidence_allowed \
  teacher_evidence_allowed \
  allowed_losses \
  disallowed_losses \
  forbidden_state_intervention_expected; do
  require_key "${key}"
done

dataset_class="$(awk -F= '$1=="dataset_class"{print $2; exit}' "${MANIFEST}")"
if [[ -z "${dataset_class}" && -f "${CORRECTIONS}" ]]; then
  dataset_class="$(awk -F= '$1=="dataset_class"{print $2; exit}' "${CORRECTIONS}")"
fi
case "${dataset_class}" in
  A_static_expert|B_dynamic_rgb_observation|C_frozen_dp_dynamic_failure|D_future_frame_cooperation_teacher|E_cosmos_predicted_cooperation)
    echo "dataset_class_allowed=true"
    ;;
  *)
    echo "dataset_class_allowed=false"
    echo "dataset_class=${dataset_class}"
    failures=$((failures + 1))
    ;;
esac

output_dir="$(awk -F= '$1=="output_dir"{print $2; exit}' "${MANIFEST}")"
if [[ -z "${output_dir}" && -f "${CORRECTIONS}" ]]; then
  output_dir="$(awk -F= '$1=="output_dir"{print $2; exit}' "${CORRECTIONS}")"
fi
case "${output_dir}" in
  "${ROOT}/experiments/maniskill/runs/01_dataset/"*)
    echo "output_dir_active_layout=true"
    ;;
  *)
    echo "output_dir_active_layout=false"
    failures=$((failures + 1))
    ;;
esac

run_group="$(awk -F= '$1=="run_group"{print $2; exit}' "${MANIFEST}")"
run_name="$(awk -F= '$1=="run_name"{print $2; exit}' "${MANIFEST}")"
if [[ -z "${run_group}" && -f "${CORRECTIONS}" ]]; then
  run_group="$(awk -F= '$1=="run_group"{print $2; exit}' "${CORRECTIONS}")"
fi
if [[ -z "${run_name}" && -f "${CORRECTIONS}" ]]; then
  run_name="$(awk -F= '$1=="run_name"{print $2; exit}' "${CORRECTIONS}")"
fi
if [[ "${run_group}" =~ p03_|full_pipeline|[0-9]{8}|server[0-9]+|job[0-9]+ ]]; then
  echo "run_group_human_readable=false"
  failures=$((failures + 1))
else
  echo "run_group_human_readable=true"
fi
if [[ "${run_name}" =~ p03_|full_pipeline|[0-9]{8}|server[0-9]+|job[0-9]+ ]]; then
  echo "run_name_human_readable=false"
  failures=$((failures + 1))
else
  echo "run_name_human_readable=true"
fi

require_manifest_value "rgb_required" "true"
require_manifest_value "forbidden_state_intervention_expected" "false"

if grep -qx 'dataset_smoke_only=true' "${MANIFEST}" || grep -qE '"dataset_smoke_only"[[:space:]]*:[[:space:]]*true' "${SUMMARY}"; then
  echo "dataset_smoke_only=true"
  require_summary_bool "human_review_required" "true"
  require_summary_bool "large_scale_production_allowed" "false"
else
  echo "dataset_smoke_only=false_or_missing"
fi

video_count=0
frame_count=0
for artifact_count_attempt in 1 2 3 4 5; do
  if [[ -d "${RUN_DIR}/videos" ]]; then
    video_count="$({ find "${RUN_DIR}/videos" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null || true; } | wc -l | tr -d ' ')"
  else
    video_count="$({ find "${RUN_DIR}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null || true; } | wc -l | tr -d ' ')"
  fi
  if [[ -d "${RUN_DIR}/review/frames" ]]; then
    frame_count="$({ find "${RUN_DIR}/review/frames" -maxdepth 1 -type f -name '*.png' 2>/dev/null || true; } | wc -l | tr -d ' ')"
  else
    frame_count=0
  fi
  if [[ "${video_count}" -ne 0 && "${frame_count}" -ne 0 ]]; then
    break
  fi
  # NFS metadata can briefly report an empty directory during concurrent
  # artifact inspection. Recount a few times before declaring evidence absent.
  sleep 1
done
echo "video_count=${video_count}"
echo "review_frame_count=${frame_count}"
if [[ "${video_count}" -eq 0 || "${frame_count}" -eq 0 ]]; then
  echo "visual_review_artifacts_present=false"
  failures=$((failures + 1))
else
  echo "visual_review_artifacts_present=true"
fi

if [[ "${failures}" -eq 0 ]]; then
  echo "dataset_run_manifest_valid=true"
  echo "failure_count=0"
  exit 0
fi

echo "dataset_run_manifest_valid=false"
echo "failure_count=${failures}"
exit 61
