# Reflex Clean Start

Last updated: 2026-06-01.

This directory is now reset for the world-model task-rebinding direction:

> Apply a policy trained on static tasks to dynamic test scenes by streaming
> perception, object-centric future prediction, event-level interruption,
> task-frame rebinding, short-horizon physical bridging, and policy
> continuability checking.

## Kept Assets

- Official PegInsertionSide demos:
  - `data/official_replay/PegInsertionSide-v1/motionplanning/trajectory.h5`
  - `data/official_replay/PegInsertionSide-v1/motionplanning/trajectory.json`
  - `data/official_replay/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5`
  - `data/official_replay/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.json`
- Base DP run:
  - `experiments/dp_peg1000/run_90201/manifest.json`
  - `experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt`
  - `experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_once.pt`
  - `experiments/dp_peg1000/run_90201/checkpoints/final.pt`
- Base DP eval summaries:
  - `experiments/eval/final_best_at_end_after_90201/metrics.json`
  - `experiments/eval/final_best_once_after_90201/metrics.json`
- Environment and upstream code:
  - `.venv/`
  - `deps/ManiSkill_clean/`
  - `scripts/training/`
  - `scripts/tools/`
  - selected non-reflex `scripts/slurm/` entries
- Rendering and cluster notes:
  - `docs/RENDERING_CLUSTER_VULKAN.md`

## Archived Out Of Tree

Old C/D1/restoration experiments, smoke jobs, videos, old PLAN/TODO,
old instruction docs, old reflex scripts, and old slurm logs were moved to:

`/public/home/yanhongru/ICLR2027/Reflex_legacy_backup_20260601_world_model_rebind`

Do not use archived C/D1 runs as the method direction. They are retained only
as evidence and for code archaeology.

## Useful Old Conclusion

The only conclusion carried forward is negative and narrow:

- Static state DP is a useful frozen skill prior.
- Previous restoration-style D1 work sometimes aligned geometry, but it did not
  solve dynamic task completion because policy continuation remained unstable.
- More D1 tuning is not the next step.
- The new problem is dynamic task-frame rebinding and continuability, not
  restoring a scene to the old static training pattern.

## Current Data Reality

The current base DP is state-only:

- environment: `PegInsertionSide-v1`
- observation mode: `state`
- control mode: `pd_ee_delta_pose`
- processed state observation shape: 43
- action shape: 7
- obs horizon: 2
- action horizon: 8
- prediction horizon: 16

The current processed H5 does not contain RGB frames. RGB-D must be replayed or
rendered into a new dataset before training a visual slot extractor or a
DDP-style visual world model.
