#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this formal pipeline only inside a compute-node srun step. It waits for a current clean-dense checkpoint saved after the formal training-time boundary, then runs generated eval and live-receding closed-loop eval without blocking on readout.
EOF
  exit 30
fi

SFT_ROOT="${SFT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299}"
JOB_NAME="${JOB_NAME:-vision_sft_droid_policy_full1000_rgb_300step_wam}"
RUN_DIR="${RUN_DIR:-${SFT_ROOT}/outputs/cosmos3/sft/${JOB_NAME}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_DIR}/checkpoints}"
FORMAL_AFTER_TIME="${FORMAL_AFTER_TIME:-2026-06-14T20:57:46+08:00}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-120}"
WATCH_TIMEOUT_SECONDS="${WATCH_TIMEOUT_SECONDS:-43200}"
CHECKPOINT_STABLE_SECONDS="${CHECKPOINT_STABLE_SECONDS:-120}"
PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT:-${SFT_ROOT}/formal_eval_live_pipeline_after_3h_skip_readout}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"

cd "${ROOT}"
mkdir -p "${PIPELINE_LOG_ROOT}"
{
  echo "timestamp=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-}"
  echo "sft_root=${SFT_ROOT}"
  echo "checkpoint_root=${CHECKPOINT_ROOT}"
  echo "formal_after_time=${FORMAL_AFTER_TIME}"
  echo "boundary=formal current-run pipeline; skips readout because readout is a non-authoritative diagnostic and previously blocked on IO/model loading. Generated eval plus live final-state/video evidence are still required."
} | tee "${PIPELINE_LOG_ROOT}/formal_pipeline_manifest.txt"

start="$(date +%s)"
selected=""
while [[ -z "${selected}" ]]; do
  selected="$("${PYTHON_BIN}" - "${CHECKPOINT_ROOT}" "${FORMAL_AFTER_TIME}" <<'PY'
from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
after = dt.datetime.fromisoformat(sys.argv[2]).timestamp()

if not root.exists():
    raise SystemExit(0)

candidates = []
for path in root.iterdir():
    if not path.is_dir():
        continue
    m = re.fullmatch(r"iter_(\d{9})", path.name)
    if not m:
        continue
    model = path / "model"
    metadata = model / ".metadata"
    if not model.is_dir() or not metadata.is_file():
        continue
    stamp = metadata.stat().st_mtime
    if stamp <= after:
        continue
    candidates.append((int(m.group(1)), stamp, path))

if candidates:
    iteration, stamp, path = sorted(candidates)[0]
    print(f"{iteration}\t{stamp:.6f}\t{path}")
PY
)"
  if [[ -n "${selected}" ]]; then
    break
  fi
  now="$(date +%s)"
  elapsed=$((now - start))
  if (( elapsed > WATCH_TIMEOUT_SECONDS )); then
    echo "formal_checkpoint_watch_timeout_seconds=${WATCH_TIMEOUT_SECONDS}" >&2
    exit 31
  fi
  echo "formal_checkpoint_not_ready elapsed_seconds=${elapsed}"
  sleep "${CHECK_INTERVAL_SECONDS}"
done

IFS=$'\t' read -r selected_iter selected_stamp selected_checkpoint <<< "${selected}"
printf -v CHECKPOINT_ITER '%d' "${selected_iter}"
printf -v CHECKPOINT_NAME 'iter_%09d' "${CHECKPOINT_ITER}"
EVAL_ROOT="${EVAL_ROOT:-${SFT_ROOT}/eval_full_episode_wam_${CHECKPOINT_NAME}_formal_after_3h}"
{
  echo "selected_checkpoint_iter=${CHECKPOINT_ITER}"
  echo "selected_checkpoint_name=${CHECKPOINT_NAME}"
  echo "selected_checkpoint_path=${selected_checkpoint}"
  echo "selected_checkpoint_mtime_epoch=${selected_stamp}"
  echo "selected_checkpoint_mtime_iso=$(date -d "@${selected_stamp}" -Is)"
  echo "eval_root=${EVAL_ROOT}"
} | tee -a "${PIPELINE_LOG_ROOT}/formal_pipeline_manifest.txt"

stable_start="$(date +%s)"
previous_signature=""
while true; do
  current_signature="$(find "${selected_checkpoint}" -type f -printf '%P %s\n' | sort)"
  now="$(date +%s)"
  if [[ -n "${current_signature}" && "${current_signature}" == "${previous_signature}" ]]; then
    if (( now - stable_start >= CHECKPOINT_STABLE_SECONDS )); then
      break
    fi
  else
    stable_start="${now}"
    previous_signature="${current_signature}"
  fi
  echo "formal_checkpoint_waiting_for_stable_files stable_seconds=$((now - stable_start))"
  sleep "${CHECK_INTERVAL_SECONDS}"
done

SFT_ROOT="${SFT_ROOT}" \
CHECKPOINT_ITER="${CHECKPOINT_ITER}" \
CHECKPOINT_PATH="${selected_checkpoint}" \
EVAL_ROOT="${EVAL_ROOT}" \
bash "${ROOT}/scripts/slurm/watch_cosmos3_clean_dense_late299_iter300_eval_in_allocation.sh" \
  2>&1 | tee "${PIPELINE_LOG_ROOT}/01_formal_generated_eval.log"

SFT_ROOT="${SFT_ROOT}" \
CHECKPOINT_ITER="${CHECKPOINT_ITER}" \
CHECKPOINT_PATH="${selected_checkpoint}" \
EVAL_ROOT="${EVAL_ROOT}" \
bash "${ROOT}/scripts/slurm/watch_cosmos3_clean_dense_late299_iter300_live_receding_panel_in_allocation.sh" \
  2>&1 | tee "${PIPELINE_LOG_ROOT}/02_formal_live_receding_panel.log"

echo "formal_pipeline_complete=$(date -Is)" | tee -a "${PIPELINE_LOG_ROOT}/formal_pipeline_manifest.txt"
