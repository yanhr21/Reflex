#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
RUN_DIR="${1:-}"

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(find "${ROOT}/experiments/maniskill/runs/03_oracle" -mindepth 3 -maxdepth 5 -type f -name summary.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}')"
fi

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "phase03_full_pipeline_run_found=false"
  echo "run_dir="
  exit 2
fi

echo "phase03_full_pipeline_run_found=true"
echo "run_dir=${RUN_DIR}"
for rel in \
  manifest.txt \
  manifest.json \
  classification.txt \
  summary.json \
  artifact_audit.json \
  action_trace.json \
  videos/raw.mp4 \
  videos/annotated.mp4
do
  if [[ -f "${RUN_DIR}/${rel}" ]]; then
    echo "exists ${rel}=true"
  else
    echo "exists ${rel}=false"
  fi
done

if [[ -f "${RUN_DIR}/classification.txt" ]]; then
  echo "--- classification.txt ---"
  sed -n '1,80p' "${RUN_DIR}/classification.txt"
fi

if [[ -f "${RUN_DIR}/artifact_audit.json" ]]; then
  echo "--- artifact_audit key lines ---"
  rg -n '"ok"|"failures"|"classification"|"cosmos_dynamic_rows"|"simulator_success_metric"|"physical_insertion_success_claimed"|"visual_full_insertion_confirmed"|"action_row_offset"|"action_row_offset_diagnostic"|"action_row_offset_source"|"action_row_offset_trace_values"|"action_row_offset_trace_index_mismatch_steps"' "${RUN_DIR}/artifact_audit.json" || true
fi

if [[ -f "${RUN_DIR}/summary.json" ]]; then
  echo "--- summary key lines ---"
  rg -n '"classification"|"target_motion_trigger_frame"|"cosmos_dynamic_actions_executed"|"dp_actions_used_during_dynamic_stage"|"near_target_before_finisher"|"simulator_success_metric"|"physical_insertion_success_claimed"|"visual_full_insertion_confirmed"|"video"|"annotated_video"|"snap_detected"|"action_row_offset"|"action_row_offset_diagnostic"|"action_row_offset_source"|"validation_key_success_allowed"' "${RUN_DIR}/summary.json" || true
fi
