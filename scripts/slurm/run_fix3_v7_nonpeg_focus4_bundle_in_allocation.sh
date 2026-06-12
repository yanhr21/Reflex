#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}
JOB_ID=${JOB_ID:-${SLURM_JOB_ID:-}}
if [[ -z "${JOB_ID}" ]]; then
  echo "JOB_ID or SLURM_JOB_ID must be set inside a held Slurm allocation" >&2
  exit 2
fi

cd "${ROOT}"

BASE_ROOT="${ROOT}/experiments/world_model_task_rebinding/cosmos3"
WRAPPER="${ROOT}/scripts/slurm/run_fix3_v7_approved_full1000_generation_in_allocation.sh"

start_focus_root() {
  local name="$1"
  local num_demos="$2"
  local scenario_sequence="$3"
  local scenario_quotas="$4"
  local scenario_seed_bases="$5"
  local accepted_index_offset="$6"
  local output_root="${BASE_ROOT}/fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_${name}"

  (
    set -euo pipefail
    export JOB_ID="${JOB_ID}"
    export ROOT="${ROOT}"
    export OUTPUT_ROOT="${output_root}"
    export LOG_PATH="${output_root}.generate.log"
    export NUM_DEMOS="${num_demos}"
    export MAX_ATTEMPTS="${MAX_ATTEMPTS:-60000}"
    export SEED="${SEED:-17000000}"
    export VAL_FRACTION="${VAL_FRACTION:-0.1}"
    export SAVE_REJECT_LOG_LIMIT="${SAVE_REJECT_LOG_LIMIT:-3000}"
    export REJECT_LOG_EVERY="${REJECT_LOG_EVERY:-100}"
    export SCENARIO_SEQUENCE="${scenario_sequence}"
    export SCENARIO_QUOTAS="${scenario_quotas}"
    export SCENARIO_SEED_BASES="${scenario_seed_bases}"
    export USE_PRIORITY_SEEDS=false
    export RESET_CANDIDATE_RNG=false
    export POLICY_RNG_SEED_BASE=-1
    export NON_PRIORITY_SEED_OFFSET=0
    export ACCEPTED_INDEX_OFFSET="${accepted_index_offset}"
    export SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:---overlap}"
    bash "${WRAPPER}"
  ) &
}

echo "timestamp=$(date -Is)"
echo "job_id=${JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "bundle=fix3_v7_nonpeg_focus4"
echo "reason=focus scarce non-peg moving-hole quotas before peg_drop/peg_disturb"

start_focus_root \
  "nonpeg_focus_continuous_aux120_seedbase16241000" \
  120 \
  "hole_late_continuous_insert" \
  "hole_late_continuous_insert=120" \
  "hole_late_continuous_insert=16241000" \
  7000

start_focus_root \
  "nonpeg_focus_constant_aux90_seedbase16250000" \
  90 \
  "hole_late_constant" \
  "hole_late_constant=90" \
  "hole_late_constant=16250000" \
  7120

start_focus_root \
  "nonpeg_focus_move_stop_aux70_seedbase16280000" \
  70 \
  "hole_late_move_stop" \
  "hole_late_move_stop=70" \
  "hole_late_move_stop=16280000" \
  7220

start_focus_root \
  "nonpeg_focus_sine_reverse_aux190_seedbase1624x" \
  190 \
  "hole_late_sine,hole_late_reverse" \
  "hole_late_sine=90,hole_late_reverse=100" \
  "hole_late_sine=16251000,hole_late_reverse=16240000" \
  7320

set +e
wait_status=0
for pid in $(jobs -p); do
  wait "${pid}" || wait_status=$?
done
set -e

echo "focus4_steps_finished=$(date -Is)"
echo "wait_status=${wait_status}"
echo "holding allocation shell for inspection/reuse"
exec bash -l
