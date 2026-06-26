#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"

cat <<EOF
# Run only after explicit user approval, from inside a compute-node srun step
# in the held Slurm allocation. These commands prepare/audit condition roots
# only; both wrappers default RUN_SFT=false.

cd ${ROOT}

ALLOW_CLEAN_DENSE_PREFLIGHT=true \\
  bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_preflight_in_allocation.sh

# Two-source overfit condition preflight, for the later overfit gate:
ALLOW_CLEAN_DENSE_PREFLIGHT=true \\
  bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_overfit2_preflight_in_allocation.sh

# Short overfit SFT after the overfit2 preflight summary is ready:
ALLOW_CLEAN_DENSE_OVERFIT_SFT=true \\
CONDITION_ROOT=<overfit2_condition_root> \\
CLEAN_DENSE_PREFLIGHT_SUMMARY=<overfit2_output_root>/clean_dense_preflight_summary.json \\
  bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_overfit2_fix1recipe_in_allocation.sh

# Full clean/dense SFT after full preflight and short overfit pass:
ALLOW_CLEAN_DENSE_FULL_SFT=true \\
CONDITION_ROOT=<full_clean_dense_condition_root> \\
CLEAN_DENSE_PREFLIGHT_SUMMARY=<full_preflight_output_root>/clean_dense_preflight_summary.json \\
  bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_full_fix1recipe_in_allocation.sh
EOF
