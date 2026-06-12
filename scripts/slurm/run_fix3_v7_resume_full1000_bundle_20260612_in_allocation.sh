#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}
JOB_ID=${JOB_ID:-${SLURM_JOB_ID:-}}
BUNDLE_KIND=${BUNDLE_KIND:?set BUNDLE_KIND to nonpeg_resume_a or mixed_peg_resume_a}
if [[ -z "${JOB_ID}" ]]; then
  echo "JOB_ID or SLURM_JOB_ID must be set for an existing held allocation" >&2
  exit 2
fi

srun --overlap --jobid="${JOB_ID}" --gres=gpu:4 --ntasks=1 --cpus-per-task=32 bash -lc "
set -euo pipefail
cd '${ROOT}'
export PYTHONPATH='${ROOT}/deps/ManiSkill_clean:${ROOT}':\${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=FALSE
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export DISPLAY=

BASE_ROOT='${ROOT}/experiments/world_model_task_rebinding/cosmos3'
GEN='.venv/bin/python scripts/world_model/generate_cosmos3_fix3_original_protocol_large_motion_dp.py'
BUNDLE_KIND='${BUNDLE_KIND}'

start_one() {
  local gpu_id=\"\$1\"
  local name=\"\$2\"
  local scenario_sequence=\"\$3\"
  local scenario_quotas=\"\$4\"
  local scenario_seed_bases=\"\$5\"
  local num_demos=\"\$6\"
  local accepted_offset=\"\$7\"
  local max_attempts=\"\$8\"
  local policy_seed_bases=\"\${9:-}\"
  local force_fallback=\"\${10:-}\"
  local output_root=\"\${BASE_ROOT}/fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260612_resume_\${name}\"
  local log_path=\"\${output_root}.generate.log\"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=\"\${gpu_id}\"
    cmd=(
      \${GEN}
      --output-root \"\${output_root}\"
      --num-demos \"\${num_demos}\"
      --max-attempts \"\${max_attempts}\"
      --seed 36000000
      --val-fraction 0.1
      --scenario-seed-bases \"\${scenario_seed_bases}\"
      --accepted-index-offset \"\${accepted_offset}\"
      --non-priority-seed-offset 0
      --policy-rng-seed-base -1
      --save-reject-log-limit 3000
      --reject-log-every 100
      --scenario-sequence \"\${scenario_sequence}\"
      --scenario-quotas \"\${scenario_quotas}\"
      --no-use-priority-seeds
    )
    if [[ -n \"\${policy_seed_bases}\" ]]; then
      cmd+=(--scenario-policy-rng-seed-bases \"\${policy_seed_bases}\")
    fi
    if [[ -n \"\${force_fallback}\" ]]; then
      cmd+=(--scenario-force-fallback-step \"\${force_fallback}\")
    fi
    echo timestamp=\$(date -Is)
    echo job_id=\${SLURM_JOB_ID:-unknown}
    echo node_list=\${SLURM_JOB_NODELIST:-unknown}
    echo cuda_visible_devices=\${CUDA_VISIBLE_DEVICES}
    echo output_root=\"\${output_root}\"
    echo physical_reason=v7_dp_full1000_resume_after_hard_teacher_deferred
    printf 'cmd='
    printf '%q ' \"\${cmd[@]}\"
    printf '\\n'
    \"\${cmd[@]}\"
  ) 2>&1 | tee \"\${log_path}\" &
}

echo timestamp=\$(date -Is)
echo job_id=\${SLURM_JOB_ID:-unknown}
echo node_list=\${SLURM_JOB_NODELIST:-unknown}
echo outer_cuda_visible_devices=\${CUDA_VISIBLE_DEVICES:-unset}
echo bundle=\"\${BUNDLE_KIND}\"

case \"\${BUNDLE_KIND}\" in
  nonpeg_resume_a)
    start_one 0 nonpeg_a_constant_seedbase36250000 hole_late_constant hole_late_constant=60 hole_late_constant=36250000 60 12000 70000
    start_one 1 nonpeg_a_continuous_seedbase36241000 hole_late_continuous_insert hole_late_continuous_insert=50 hole_late_continuous_insert=36241000 50 12080 70000
    start_one 2 nonpeg_a_sine_seedbase36251000 hole_late_sine hole_late_sine=55 hole_late_sine=36251000 55 12150 70000
    start_one 3 nonpeg_a_move_stop_seedbase36280000 hole_late_move_stop hole_late_move_stop=45 hole_late_move_stop=36280000 45 12220 70000
    ;;
  mixed_peg_resume_a)
    start_one 0 mixed_a_reverse_seedbase36240000 hole_late_reverse hole_late_reverse=45 hole_late_reverse=36240000 45 12300 70000
    start_one 1 mixed_a_fast_shift_seedbase36300000 hole_late_fast_shift hole_late_fast_shift=40 hole_late_fast_shift=36300000 40 12360 70000
    start_one 2 mixed_a_peg_drop_seedbase36705000 peg_drop peg_drop=100 peg_drop=36705000 100 12420 90000 peg_drop=39705000
    start_one 3 mixed_a_peg_disturb_seedbase36751000 peg_disturb peg_disturb=120 peg_disturb=36751000 120 12540 120000 peg_disturb=39751000 peg_disturb=90
    ;;
  *)
    echo \"unknown BUNDLE_KIND=\${BUNDLE_KIND}\" >&2
    exit 2
    ;;
esac

wait
echo bundle_finished=\$(date -Is)
"
