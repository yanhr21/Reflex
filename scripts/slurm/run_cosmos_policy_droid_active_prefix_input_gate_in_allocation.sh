#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run Cosmos Policy-DROID live-prefix input gates only inside a compute-node srun step.
EOF
  exit 30
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299}"
CONDITION_ROOT="${CONDITION_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050}"
DATASET_JSONL="${DATASET_JSONL:-${CONDITION_ROOT}/val/video_action_dataset_file.jsonl}"
JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
RUN_DIR="${RUN_DIR:-${SFT_ROOT}/outputs/cosmos3/sft/${JOB_NAME}}"
CHECKPOINT_ITER="${CHECKPOINT_ITER:-1500}"
printf -v CHECKPOINT_NAME 'iter_%09d' "${CHECKPOINT_ITER}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RUN_DIR}/checkpoints/${CHECKPOINT_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SFT_ROOT}/cosmos_policy_droid_active_prefix_input_gate_${CHECKPOINT_NAME}_${STAMP}_alloc${SLURM_JOB_ID}}"

ROW_INDEX="${ROW_INDEX:-0}"
ROW_UUID="${ROW_UUID:-}"
ROW_PREFIX_ROLE="${ROW_PREFIX_ROLE:-target_motion_observed}"
FPS="${FPS:-30}"
RUN_INFERENCE="${RUN_INFERENCE:-false}"

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

main() {
  cd "${ROOT}"
  [[ -x "${PYTHON}" ]] || { echo "missing Python: ${PYTHON}" >&2; exit 2; }
  [[ -s "${DATASET_JSONL}" ]] || { echo "missing dataset jsonl: ${DATASET_JSONL}" >&2; exit 3; }
  [[ -d "${CHECKPOINT_PATH}" ]] || { echo "missing checkpoint path: ${CHECKPOINT_PATH}" >&2; exit 4; }
  [[ -d "${CONDITION_ROOT}" ]] || { echo "missing condition root: ${CONDITION_ROOT}" >&2; exit 5; }

  mkdir -p "${OUTPUT_ROOT}"
  local row_json="${OUTPUT_ROOT}/selected_condition_row.json"
  local prefix_video="${OUTPUT_ROOT}/observed_prefix.mp4"
  local prefix_manifest="${OUTPUT_ROOT}/observed_prefix_manifest.json"

  "${PYTHON}" - "${DATASET_JSONL}" "${row_json}" "${ROW_INDEX}" "${ROW_UUID}" "${ROW_PREFIX_ROLE}" <<'PY'
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
out = Path(sys.argv[2])
row_index = int(sys.argv[3])
row_uuid = sys.argv[4].strip()
row_prefix_role = sys.argv[5].strip()

selected = None
matched = 0
with dataset.open() as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        row = json.loads(line)
        uuid = str(row.get("uuid", ""))
        role = str(row.get("prefix_role", ""))
        if row_uuid:
            ok = uuid == row_uuid
        else:
            ok = role == row_prefix_role
        if not ok:
            continue
        if matched == row_index:
            selected = row
            selected["_dataset_line_index"] = i
            break
        matched += 1

if selected is None:
    raise SystemExit(
        f"no row selected from {dataset}; ROW_UUID={row_uuid!r} "
        f"ROW_PREFIX_ROLE={row_prefix_role!r} ROW_INDEX={row_index}"
    )

required = ["vision_path", "prefix_frame_index", "uuid"]
missing = [k for k in required if k not in selected]
if missing:
    raise SystemExit(f"selected row missing required fields: {missing}")
meta = selected.get("metadata") or {}
if not meta.get("source_h5"):
    raise SystemExit("selected row has no metadata.source_h5; cannot build causal source-history conditions")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "selected_uuid": selected.get("uuid"),
    "dataset_line_index": selected.get("_dataset_line_index"),
    "prefix_frame_index": selected.get("prefix_frame_index"),
    "prefix_role": selected.get("prefix_role"),
    "vision_path": selected.get("vision_path"),
    "source_h5": meta.get("source_h5"),
}, sort_keys=True))
PY

  local prefix_frame_index vision_path source_h5 sample_name scenario prefix_role prompt
  prefix_frame_index="$("${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(int(r["prefix_frame_index"]))' "${row_json}")"
  vision_path="$("${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["vision_path"])' "${row_json}")"
  source_h5="$("${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); print((r.get("metadata") or {})["source_h5"])' "${row_json}")"
  sample_name="$("${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["uuid"])' "${row_json}")"
  scenario="$("${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); print((r.get("metadata") or {}).get("scenario", r.get("scenario", "live_dynamic")))' "${row_json}")"
  prefix_role="$("${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r.get("prefix_role", "live_receding"))' "${row_json}")"
  prompt="$("${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); w=r.get("t2w_windows") or []; print((w[0].get("caption") if w else r.get("prompt", "")) or "")' "${row_json}")"

  "${PYTHON}" "${ROOT}/scripts/world_model/write_video_prefix.py" \
    --input-video "${vision_path}" \
    --output-video "${prefix_video}" \
    --prefix-frame-index "${prefix_frame_index}" \
    --fps "${FPS}" \
    --output-json "${prefix_manifest}" \
    2>&1 | tee "${OUTPUT_ROOT}/write_prefix_video.log"

  {
    echo "timestamp=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "step_id=${SLURM_STEP_ID}"
    echo "sft_root=${SFT_ROOT}"
    echo "condition_root=${CONDITION_ROOT}"
    echo "dataset_jsonl=${DATASET_JSONL}"
    echo "checkpoint_path=${CHECKPOINT_PATH}"
    echo "selected_row_json=${row_json}"
    echo "prefix_video=${prefix_video}"
    echo "prefix_frame_index=${prefix_frame_index}"
    echo "source_h5=${source_h5}"
    echo "sample_name=${sample_name}"
    echo "scenario=${scenario}"
    echo "prefix_role=${prefix_role}"
    echo "prompt_source=selected_condition_row_t2w_caption"
    echo "run_inference=${RUN_INFERENCE}"
    echo "boundary=Current 733 clean-dense Policy-DROID live-prefix input gate. It is not controller or method evidence."
  } | tee "${OUTPUT_ROOT}/active_prefix_gate_manifest.txt"

  SFT_ROOT="${SFT_ROOT}" \
  CONDITION_ROOT="${CONDITION_ROOT}" \
  CHECKPOINT_ITER="${CHECKPOINT_ITER}" \
  CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}/cosmos_live_prefix" \
  PREFIX_VIDEO="${prefix_video}" \
  PREFIX_FRAME_INDEX="${prefix_frame_index}" \
  SOURCE_H5="${source_h5}" \
  SAMPLE_NAME="${sample_name}" \
  SCENARIO="${scenario}" \
  PREFIX_ROLE="${prefix_role}" \
  PROMPT="${prompt}" \
  RUN_INFERENCE="${RUN_INFERENCE}" \
  "${ROOT}/scripts/slurm/run_cosmos3_live_prefix_policy_inference_in_allocation.sh" \
    2>&1 | tee "${OUTPUT_ROOT}/run_active_prefix_builder.log"
}

main "$@"
