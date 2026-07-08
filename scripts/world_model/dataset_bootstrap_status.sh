#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
BOOT="${BOOT:-${ROOT}/experiments/maniskill/data/active/b_dynamic_legacy_bootstrap}"
RGBD_ROOT_LINK="${BOOT}/rgbd_root"
RGBD_LIST="${BOOT}/rgbd_h5_paths.txt"
MP4_LIST="${BOOT}/mp4_paths.txt"
IMAGE_LIST="${BOOT}/image_paths.txt"
SCENARIO_COUNTS="${BOOT}/scenario_counts.txt"
SAMPLES_TSV="${BOOT}/samples.tsv"
TRAIN_TSV="${BOOT}/train_samples.tsv"
VAL_TSV="${BOOT}/val_samples.tsv"
SAMPLES_JSONL="${BOOT}/samples.jsonl"
TRAIN_JSONL="${BOOT}/train_samples.jsonl"
VAL_JSONL="${BOOT}/val_samples.jsonl"
INDEX_MANIFEST="${BOOT}/index_manifest.txt"
SPLIT_SCENARIO_COUNTS="${BOOT}/split_scenario_counts.txt"

echo "dataset_bootstrap_status_ok=true"
echo "bootstrap_dir=${BOOT}"

if [[ -L "${RGBD_ROOT_LINK}" || -d "${RGBD_ROOT_LINK}" ]]; then
  rgbd_root="$(readlink -f "${RGBD_ROOT_LINK}")"
  echo "rgbd_root_exists=true"
  echo "rgbd_root=${rgbd_root}"
else
  echo "rgbd_root_exists=false"
  echo "rgbd_root=${RGBD_ROOT_LINK}"
  exit 20
fi

for path in "${RGBD_LIST}" "${MP4_LIST}" "${IMAGE_LIST}" "${SCENARIO_COUNTS}"; do
  label="$(basename "${path}")"
  if [[ -f "${path}" ]]; then
    echo "${label}_exists=true"
    echo "${label}_lines=$(wc -l < "${path}" | tr -d ' ')"
  else
    echo "${label}_exists=false"
    echo "${label}_lines=0"
  fi
done

for path in \
  "${SAMPLES_TSV}" \
  "${TRAIN_TSV}" \
  "${VAL_TSV}" \
  "${SAMPLES_JSONL}" \
  "${TRAIN_JSONL}" \
  "${VAL_JSONL}" \
  "${INDEX_MANIFEST}" \
  "${SPLIT_SCENARIO_COUNTS}"; do
  label="$(basename "${path}")"
  if [[ -f "${path}" ]]; then
    echo "${label}_exists=true"
    echo "${label}_lines=$(wc -l < "${path}" | tr -d ' ')"
  else
    echo "${label}_exists=false"
    echo "${label}_lines=0"
  fi
done

if [[ -f "${SCENARIO_COUNTS}" ]]; then
  echo "scenario_counts_begin"
  sed 's/^/  /' "${SCENARIO_COUNTS}"
  echo "scenario_counts_end"
fi

if [[ -f "${SPLIT_SCENARIO_COUNTS}" ]]; then
  echo "split_scenario_counts_begin"
  sed 's/^/  /' "${SPLIT_SCENARIO_COUNTS}"
  echo "split_scenario_counts_end"
fi

echo "allowed_use=bootstrap_cosmos_dynamic_future_target_readout_schema_inspection_ablation"
echo "positive_dp_bc_allowed=false"
echo "final_method_evidence_allowed=false"
echo "replaces_new_b_c_d_e_production=false"
