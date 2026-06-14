# Plan

Active plan entry point after the 2026-06-09 Cosmos3 reset.

## Active Directory

- `PLAN/cosmos3_300f_world_model/`

This is the only active plan for Cosmos3 training and testing. The previous
`PLAN/world_model_task_rebinding/` directory was moved to:

- `PLAN/_backup_world_model_task_rebinding_20260609_before_cosmos3_300f_reset/`

## Proposed Method Branch

- `PLAN/cosmos3_lowfreq_wm_executor/`

This is a new, separate proposal after the 2026-06-14 finding that the
full-length iter2700 closed loop satisfies the implementation contract but is
not method-effective. It separates low-frequency Cosmos task-world prediction
from high-frequency robot execution and must not be mixed with the old
direct raw-Cosmos-action checkpoint claims.

## Current Boundary

- Keep the approved full1000 ManiSkill default-view RGB dataset.
- Treat the episode as a 300-step rollout, with 301 RGB/state frames when the
  reset frame is included.
- Do not construct or use 129-frame videos, 128-action chunks, old 93-frame
  exports, cropped metrics, or any `min(pred, ref)` evaluation.
- Do not start training or evaluation until the new plan/TODO is reviewed.

## Active Assets

- Cosmos3 checkpoints: `checkpoints/cosmos3/`
- Base static DP checkpoints: `experiments/dp_peg1000/run_90201/checkpoints/`
- Approved full1000 RGB videos:
  `experiments/world_model_task_rebinding/cosmos3/sft_dataset_full1000_maniskill_default_regen_20260606_0055`
- Full1000 source H5/specs:
  `data/cosmos3/full1000_rgbd_env_states_20260603_1938`

Old experiment results and old evidence notes were moved outside the current
ICLR2027 tree under:

- `/public/home/yanhongru/ICLR2027_archive/reflex_20260609_cosmos3_300f_reset/`
