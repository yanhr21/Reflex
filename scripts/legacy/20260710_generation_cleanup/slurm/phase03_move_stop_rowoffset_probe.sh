#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/public/home/yanhongru/ICLR2027/Reflex}"
cd "${ROOT}"

export RUN_GROUP="${RUN_GROUP:-render_probe}"
export RUN_NAME="${RUN_NAME:-move17}"
export FULL_RUN_GROUP="${FULL_RUN_GROUP:-h5_move_stop}"
export FULL_RUN_NAME="${FULL_RUN_NAME:-try17}"
export FULL_RUN_SCRIPT="${FULL_RUN_SCRIPT:-scripts/slurm/phase03_move_stop_rowoffset.sh}"
export RENDER_SHADER_PACK="${RENDER_SHADER_PACK:-minimal}"
export RENDER_CANARY_API="${RENDER_CANARY_API:-gym}"

bash scripts/slurm/phase03_render_gpu_probe.sh
