#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
export DATASET_STAGE="e_cosmos_predicted_smoke"
exec "${ROOT}/scripts/slurm/launch_dataset_stage_smoke_tmux_common.sh"
