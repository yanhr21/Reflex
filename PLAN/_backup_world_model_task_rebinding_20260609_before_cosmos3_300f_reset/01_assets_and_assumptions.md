# Assets And Assumptions

## Current Assets

- State demos:
  - `data/official_replay/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5`
  - `data/official_replay/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.json`
- Raw official demos:
  - `data/official_replay/PegInsertionSide-v1/motionplanning/trajectory.h5`
  - `data/official_replay/PegInsertionSide-v1/motionplanning/trajectory.json`
- Frozen state DP:
  - `experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt`
  - `experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_once.pt`
  - `experiments/dp_peg1000/run_90201/checkpoints/final.pt`
- Training/eval scripts:
  - `scripts/training/train_dp_state_ddp.py`
  - `scripts/training/eval_dp_state.py`
  - `scripts/slurm/train_dp_peg_1000.sbatch`
  - `scripts/slurm/eval_dp_official.sbatch`

Workability check: these are sufficient to reproduce a static baseline and to
generate new dynamic state rollouts. They are not sufficient for RGB-D training
until RGB-D replay is added.

## Known Data Reality

The current base DP is state-only:

```text
env: PegInsertionSide-v1
obs_mode: state
control_mode: pd_ee_delta_pose
obs_dim: 43
action_dim: 7
obs_horizon: 2
act_horizon: 8
pred_horizon: 16
```

Workability check: a state-only first method is aligned with the checkpoint and
data. Adding RGB-D first would create a second policy problem before we have
tested the rebinding idea.

## Cluster Constraints

Heavy replay, rollout, rendering, training, and labeling must run on Slurm
compute nodes. Login-node work should be limited to file inspection,
lightweight metadata checks, and downloads.

Use the rendering notes in `docs/RENDERING_CLUSTER_VULKAN.md`, but do not use
any standing node exclusion list. Check current Slurm node state and use targeted
canaries for node-specific render concerns.

Workability check: this keeps the implementation path compatible with the
cluster. Every new generator or trainer should have a Slurm entry before large
runs.
