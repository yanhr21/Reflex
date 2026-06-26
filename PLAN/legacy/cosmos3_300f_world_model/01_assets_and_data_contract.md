# Assets And Data Contract

## Kept Active Assets

- Cosmos3 checkpoints:
  - `checkpoints/cosmos3/Cosmos3-Nano`
  - `checkpoints/cosmos3/Cosmos3-Nano-DCP`
  - `checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID`
  - `checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID-DCP`
  - `checkpoints/cosmos3/wan22_vae`
- Base static DP checkpoints:
  - `experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt`
  - `experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_once.pt`
  - `experiments/dp_peg1000/run_90201/checkpoints/final.pt`
- Approved full1000 RGB dataset:
  - `experiments/world_model_task_rebinding/cosmos3/sft_dataset_full1000_maniskill_default_regen_20260606_0055`
- Full1000 source H5/specs:
  - `data/cosmos3/full1000_rgbd_env_states_20260603_1938`

## Superseded Source Dataset Facts

The following facts describe the old 6/6 full1000 dataset structure, but the
dataset is no longer valid as active fix3 SFT source after the 2026-06-10
success/motion audit. It remains historical diagnostic data only.

- records: `1000`
- final MP4s: `1000`
- FPS: `30`
- RGB size: `1024x1024`
- viewpoint: ManiSkill default human-render camera,
  `look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])`, `fov=1`
- frame stride: `1`
- source frame span: `0..300`
- video/state frames: `301` when frame 0 is included
- action steps: `300`
- scenarios: `none`, `hole_move_stop`, `hole_constant`, `hole_reverse`,
  `peg_disturb`, `peg_drop`

## Active Contract

Every active SFT/eval sample must be based on the complete 300-step episode
contract:

```text
RGB/state frames: 301 if frame 0 is included
action steps:     300
sampling:         stride 1, 30 fps
visual input:     approved RGB only
```

If a Cosmos adapter uses prefix conditioning, the physical sample still remains
the full episode. Prefix masks may select observed frames/actions, and losses
may apply only to the causal future after the prefix, but the saved record must
carry the full episode reference and exact length accounting.

Rejected constructions:

- `129` video frames paired with `128` actions
- old `93` predicted frames compared to `301` references
- cropped PSNR or cropped state/action metrics
- `min(pred, ref)` length matching
- future ground-truth object poses as model conditions
- old invalid `action_state_conditions_full1000_maniskill_default_regen_20260606_0055`
  condition root

## Path Rule

The old source H5 paths under `experiments/_archive/...` were replaced by the
active path under `data/cosmos3/full1000_rgbd_env_states_20260603_1938`. Before
any training, run a strict path preflight that proves:

- the source list contains `1000` H5 paths,
- every listed H5 exists,
- every selected RGB video exists,
- each pair has the same source index, scenario, trajectory id, and frame span,
- the action tensor has `300` steps,
- the state/readout target tensor has `301` frames or a documented `300`-frame
  convention that does not drop an action transition.
