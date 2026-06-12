#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}
JOB_ID=${JOB_ID:?set JOB_ID to the held Slurm allocation id}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611}
LOG_PATH=${LOG_PATH:-${OUTPUT_ROOT}.generate.log}
NUM_DEMOS=${NUM_DEMOS:-1000}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-80000}
SEED=${SEED:-1200000}
VAL_FRACTION=${VAL_FRACTION:-0.1}
SAVE_REJECT_LOG_LIMIT=${SAVE_REJECT_LOG_LIMIT:-3000}
REJECT_LOG_EVERY=${REJECT_LOG_EVERY:-100}
SCENARIO_SEQUENCE=${SCENARIO_SEQUENCE:-}
SCENARIO_QUOTAS=${SCENARIO_QUOTAS:-}
SCENARIO_SEED_BASES=${SCENARIO_SEED_BASES:-hole_late_move_stop=1080000,hole_late_constant=1050000,hole_late_reverse=1040000,hole_late_sine=1050000,hole_late_continuous_insert=1040000,hole_late_fast_shift=1100000,none=700000,peg_drop=705000,peg_disturb=1051000}
SCENARIO_TRIGGER_YZ_MIN=${SCENARIO_TRIGGER_YZ_MIN:-}
SCENARIO_MIN_TRIGGER_STEP=${SCENARIO_MIN_TRIGGER_STEP:-}
SCENARIO_FALLBACK_TRIGGER_STEP=${SCENARIO_FALLBACK_TRIGGER_STEP:-}
SCENARIO_FORCE_FALLBACK_STEP=${SCENARIO_FORCE_FALLBACK_STEP:-}
SCENARIO_MIN_INSERT_AFTER_TARGET_MOTION_END_STEPS=${SCENARIO_MIN_INSERT_AFTER_TARGET_MOTION_END_STEPS:-}
USE_PRIORITY_SEEDS=${USE_PRIORITY_SEEDS:-true}
RESET_CANDIDATE_RNG=${RESET_CANDIDATE_RNG:-false}
POLICY_RNG_SEED_BASE=${POLICY_RNG_SEED_BASE:--1}
SCENARIO_POLICY_RNG_SEED_BASES=${SCENARIO_POLICY_RNG_SEED_BASES:-}
NON_PRIORITY_SEED_OFFSET=${NON_PRIORITY_SEED_OFFSET:-0}
ACCEPTED_INDEX_OFFSET=${ACCEPTED_INDEX_OFFSET:-0}
SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS:-}

srun_extra_args=()
if [[ -n "${SRUN_EXTRA_ARGS}" ]]; then
  # Intended for allocation-internal resource management, e.g. --overlap.
  read -r -a srun_extra_args <<< "${SRUN_EXTRA_ARGS}"
fi

extra_args=()
if [[ -n "${SCENARIO_SEQUENCE}" ]]; then
  extra_args+=(--scenario-sequence "${SCENARIO_SEQUENCE}")
fi
if [[ -n "${SCENARIO_QUOTAS}" ]]; then
  extra_args+=(--scenario-quotas "${SCENARIO_QUOTAS}")
fi
if [[ -n "${SCENARIO_TRIGGER_YZ_MIN}" ]]; then
  extra_args+=(--scenario-trigger-yz-min "${SCENARIO_TRIGGER_YZ_MIN}")
fi
if [[ -n "${SCENARIO_MIN_TRIGGER_STEP}" ]]; then
  extra_args+=(--scenario-min-trigger-step "${SCENARIO_MIN_TRIGGER_STEP}")
fi
if [[ -n "${SCENARIO_FALLBACK_TRIGGER_STEP}" ]]; then
  extra_args+=(--scenario-fallback-trigger-step "${SCENARIO_FALLBACK_TRIGGER_STEP}")
fi
if [[ -n "${SCENARIO_FORCE_FALLBACK_STEP}" ]]; then
  extra_args+=(--scenario-force-fallback-step "${SCENARIO_FORCE_FALLBACK_STEP}")
fi
if [[ -n "${SCENARIO_MIN_INSERT_AFTER_TARGET_MOTION_END_STEPS}" ]]; then
  extra_args+=(--scenario-min-insert-after-target-motion-end-steps "${SCENARIO_MIN_INSERT_AFTER_TARGET_MOTION_END_STEPS}")
fi
if [[ -n "${SCENARIO_POLICY_RNG_SEED_BASES}" ]]; then
  extra_args+=(--scenario-policy-rng-seed-bases "${SCENARIO_POLICY_RNG_SEED_BASES}")
fi
if [[ "${USE_PRIORITY_SEEDS}" == "false" || "${USE_PRIORITY_SEEDS}" == "0" ]]; then
  extra_args+=(--no-use-priority-seeds)
fi
if [[ "${RESET_CANDIDATE_RNG}" == "true" || "${RESET_CANDIDATE_RNG}" == "1" ]]; then
  extra_args+=(--reset-candidate-rng)
fi

mkdir -p "$(dirname "${OUTPUT_ROOT}")"

generator_cmd=(
  .venv/bin/python scripts/world_model/generate_cosmos3_fix3_original_protocol_large_motion_dp.py
  --output-root "${OUTPUT_ROOT}"
  --num-demos "${NUM_DEMOS}"
  --max-attempts "${MAX_ATTEMPTS}"
  --seed "${SEED}"
  --val-fraction "${VAL_FRACTION}"
  --scenario-seed-bases "${SCENARIO_SEED_BASES}"
  --accepted-index-offset "${ACCEPTED_INDEX_OFFSET}"
  --non-priority-seed-offset "${NON_PRIORITY_SEED_OFFSET}"
  --policy-rng-seed-base "${POLICY_RNG_SEED_BASE}"
  --save-reject-log-limit "${SAVE_REJECT_LOG_LIMIT}"
  --reject-log-every "${REJECT_LOG_EVERY}"
)
generator_cmd+=("${extra_args[@]}")
printf -v generator_cmd_q "%q " "${generator_cmd[@]}"

srun "${srun_extra_args[@]}" --jobid="${JOB_ID}" --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -lc "
set -euo pipefail
cd '${ROOT}'
export PYTHONPATH='${ROOT}/deps/ManiSkill_clean:${ROOT}':\${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=FALSE
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export DISPLAY=
echo timestamp=\$(date -Is)
echo job_id=\${SLURM_JOB_ID:-unknown}
echo node_list=\${SLURM_JOB_NODELIST:-unknown}
echo output_root='${OUTPUT_ROOT}'
echo physical_reason=v7_complete9_combined_user_approved_full1000_generation
${generator_cmd_q}
" 2>&1 | tee "${LOG_PATH}"
