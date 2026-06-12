#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "extern" ]]; then
  cat >&2 <<'EOF'
refusing_login_node_execution=true
reason=Run this watcher only inside a compute-node srun step. It waits for generated-RGB readout artifacts and then writes a diagnostic failure profile.
EOF
  exit 30
fi

READOUT_EVAL_ROOT="${READOUT_EVAL_ROOT:?set READOUT_EVAL_ROOT to the generated-RGB readout output root}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
WATCH_TIMEOUT_SECONDS="${WATCH_TIMEOUT_SECONDS:-14400}"
OUTPUT_JSON="${OUTPUT_JSON:-${READOUT_EVAL_ROOT}/readout_failure_profile.json}"
OUTPUT_MD="${OUTPUT_MD:-${READOUT_EVAL_ROOT}/readout_failure_profile.md}"
THRESHOLDS_M="${THRESHOLDS_M:-0.002,0.005,0.01,0.02,0.05}"

main() {
  cd "${ROOT}"
  [[ -x "${PYTHON_BIN}" ]] || { echo "missing python: ${PYTHON_BIN}" >&2; exit 2; }

  local summary_json="${READOUT_EVAL_ROOT}/readout_eval_summary.json"
  local samples_dir="${READOUT_EVAL_ROOT}/samples"
  local start now elapsed
  start="$(date +%s)"
  echo "watch_readout_eval_root=${READOUT_EVAL_ROOT}"
  echo "watch_readout_summary=${summary_json}"
  echo "profile_output_json=${OUTPUT_JSON}"
  echo "profile_output_md=${OUTPUT_MD}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "step_id=${SLURM_STEP_ID}"

  while [[ ! -s "${summary_json}" || ! -d "${samples_dir}" ]]; do
    now="$(date +%s)"
    elapsed=$((now - start))
    if (( elapsed > WATCH_TIMEOUT_SECONDS )); then
      echo "readout_profile_watch_timeout_seconds=${WATCH_TIMEOUT_SECONDS}" >&2
      exit 31
    fi
    echo "readout_not_ready elapsed_seconds=${elapsed}"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done

  echo "readout_ready=${summary_json}"
  "${PYTHON_BIN}" "${ROOT}/scripts/world_model/profile_cosmos3_readout_failure_modes.py" \
    --readout-root "${READOUT_EVAL_ROOT}" \
    --output-json "${OUTPUT_JSON}" \
    --output-md "${OUTPUT_MD}" \
    --thresholds-m "${THRESHOLDS_M}" \
    2>&1 | tee "${READOUT_EVAL_ROOT}/run_readout_failure_profile.log"
}

main "$@"
