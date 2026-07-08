#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
BOOT="${BOOT:-${ROOT}/experiments/maniskill/data/active/b_dynamic_legacy_bootstrap}"
VAL_MOD="${VAL_MOD:-10}"

RGBD_LIST="${BOOT}/rgbd_h5_paths.txt"
MP4_LIST="${BOOT}/mp4_paths.txt"
SAMPLES_TSV="${BOOT}/samples.tsv"
TRAIN_TSV="${BOOT}/train_samples.tsv"
VAL_TSV="${BOOT}/val_samples.tsv"
JSONL="${BOOT}/samples.jsonl"
TRAIN_JSONL="${BOOT}/train_samples.jsonl"
VAL_JSONL="${BOOT}/val_samples.jsonl"
MANIFEST="${BOOT}/index_manifest.txt"
SPLIT_SCENARIO_COUNTS="${BOOT}/split_scenario_counts.txt"

if [[ ! -f "${RGBD_LIST}" ]]; then
  echo "missing_rgbd_list=${RGBD_LIST}" >&2
  exit 20
fi
if [[ ! -f "${MP4_LIST}" ]]; then
  echo "missing_mp4_list=${MP4_LIST}" >&2
  exit 21
fi

tmp_mp4="$(mktemp)"
tmp_samples="$(mktemp)"
trap 'rm -f "${tmp_mp4}" "${tmp_samples}"' EXIT

awk -F/ '
  {
    key=$(NF-1)
    mp4[key]=$0
  }
  END {
    for (k in mp4) print k "\t" mp4[k]
  }
' "${MP4_LIST}" | sort > "${tmp_mp4}"

{
  printf 'sample_id\tsplit\tdataset_class\tdataset_role\tscenario\tseed\ttraj_id\trgbd_h5\tmp4\tallowed_losses\tdisallowed_losses\tmethod_evidence_allowed\tpositive_dp_bc_allowed\treplaces_new_production\n'
  awk -v val_mod="${VAL_MOD}" '
    BEGIN {
      FS="\t"
      while ((getline line < ARGV[2]) > 0) {
        split(line, parts, "\t")
        mp4[parts[1]]=parts[2]
      }
      close(ARGV[2])
      ARGV[2]=""
    }
    {
      h5=$0
      n=split(h5, path_parts, "/")
      sample_dir=path_parts[n-1]
      file=path_parts[n]
      sample_id=sample_dir
      scenario=file
      sub(/_seed.*/, "", scenario)
      seed=file
      sub(/.*_seed/, "", seed)
      sub(/_n.*/, "", seed)
      traj_id=sample_dir
      sub(/.*_traj_/, "", traj_id)
      sub(/\.rgbd$/, "", traj_id)
      split_name=(i % val_mod == 0) ? "val" : "train"
      video=(sample_dir in mp4) ? mp4[sample_dir] : ""
      printf "%s\t%s\tB_dynamic_rgb_observation\tlegacy_bootstrap\t%s\t%s\t%s\t%s\t%s\tcosmos_dynamic_future,target_frame_readout,trajectory_consistency,uncertainty,diagnostic_ablation\tpositive_dp_bc,final_method_success,active_new_production_success\tfalse\tfalse\tfalse\n", sample_id, split_name, scenario, seed, traj_id, h5, video
      i++
    }
  ' "${RGBD_LIST}" "${tmp_mp4}"
} > "${tmp_samples}"

mv "${tmp_samples}" "${SAMPLES_TSV}"
awk -F'\t' 'NR==1 || $2=="train"' "${SAMPLES_TSV}" > "${TRAIN_TSV}"
awk -F'\t' 'NR==1 || $2=="val"' "${SAMPLES_TSV}" > "${VAL_TSV}"

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

tsv_to_jsonl "${SAMPLES_TSV}" "${JSONL}"
tsv_to_jsonl "${TRAIN_TSV}" "${TRAIN_JSONL}"
tsv_to_jsonl "${VAL_TSV}" "${VAL_JSONL}"

awk -F'\t' 'NR>1 {count[$2 "\t" $5]++} END {for (k in count) print count[k] "\t" k}' \
  "${SAMPLES_TSV}" | sort -k2,2 -k3,3 > "${SPLIT_SCENARIO_COUNTS}"

{
  echo "timestamp=$(date -Is)"
  echo "dataset_class=B_dynamic_rgb_observation"
  echo "dataset_role=legacy_bootstrap"
  echo "val_mod=${VAL_MOD}"
  echo "samples_tsv=${SAMPLES_TSV}"
  echo "train_tsv=${TRAIN_TSV}"
  echo "val_tsv=${VAL_TSV}"
  echo "samples_jsonl=${JSONL}"
  echo "train_jsonl=${TRAIN_JSONL}"
  echo "val_jsonl=${VAL_JSONL}"
  echo "split_scenario_counts=${SPLIT_SCENARIO_COUNTS}"
  echo "sample_count=$(($(wc -l < "${SAMPLES_TSV}") - 1))"
  echo "train_count=$(($(wc -l < "${TRAIN_TSV}") - 1))"
  echo "val_count=$(($(wc -l < "${VAL_TSV}") - 1))"
  echo "method_evidence_allowed=false"
  echo "positive_dp_bc_allowed=false"
  echo "replaces_new_b_c_d_e_production=false"
} > "${MANIFEST}"

cat "${MANIFEST}"
