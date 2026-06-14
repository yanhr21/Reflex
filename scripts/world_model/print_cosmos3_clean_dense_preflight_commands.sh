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
EOF
