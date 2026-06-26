# RGB-D And Baselines

## Role Of RGB-D

RGB-D is required method evidence for the current work, not optional visual
polish. State/oracle slots remain useful for debugging, labels, and upper
bounds, but they cannot be the final controller evidence.

Workability check: this still isolates two hard problems without lowering the
goal. State/oracle runs can localize control or geometry failures; RGB-D runs
must prove that the perception-to-world-model-to-controller interface works.

## RGB-D Data Path

Add a Slurm replay/render path that saves synchronized:

```text
rgb
depth
state
actions
env_states
camera parameters
object pose labels
visibility/confidence labels if available
```

Do not confuse preview videos with an RGB-D training dataset.

Workability check: ManiSkill already has RGB-D wrappers and PegInsertionSide
has a default `base_camera`. The missing piece is a project-local dataset
export script with validation.

Current implementation:

- `scripts/world_model/render_dynamic_rgbd_dataset.py` renders RGB-D from saved
  dynamic env-state traces and writes RGB, depth, camera parameters, actions,
  pose slots, and frame-aligned env states.
- `scripts/slurm/render_dynamic_rgbd_dataset.sbatch` is the single-GPU smoke or
  small-shard wrapper.
- `scripts/slurm/render_dynamic_rgbd_dataset_dense.sbatch` and
  `scripts/slurm/render_dynamic_rgbd_dataset_from_rollout_dir_dense.sbatch`
  are large renderers. They refuse one-GPU-per-node waste, allow non-wasteful
  partial-node shard layouts such as 1x2, 1x4, 4x4, or 8x2, and keep the cap
  at 8 nodes / 64 GPUs.
- `scripts/slurm/replay_peg_official_rgbd.sbatch` generates RGB-D from the
  official static demos for visual baselines.

Workability check: rendering from saved env states makes the visual dataset a
view of the same dynamic control problem, not a separately replayed task with
different physics.

## Slot Extractor

Train:

```text
RGB-D history -> object slots
  hole pose
  peg pose
  TCP/robot proprio state
  grasp/contact predicates
  confidence and visibility
```

Use simulator labels first. Later replace or compare with FoundationPose,
SAM-like segmentation, or other vision foundation model components.

Workability check: the controller only needs slots and uncertainty. It does
not need a visual model to generate future pixels.

## DDP-Style Baseline

Implement a DDP-style baseline after the state method is measurable:

- train or attach a latent predictor
- compute real-imagination discrepancy
- use pose offset or tracking compensation
- compare against task-frame rebinding on the same dynamic tests

Workability check: DDP is a serious baseline, but its published mechanism is
closer to static displacement tracking and imagination bypass. It should not
define our method.

## VLA / Large Model Baselines

Use GR00T/pi-style/VLA systems as comparison points or high-level advisors only
after the state and RGB-D object-slot pipeline exists. Use Cosmos-like world
models for synthetic data or high-level visual prediction, not the first
inner-loop peg insertion controller.

Workability check: large models are useful if they improve object prediction,
skill selection, or data generation. They are not a substitute for measurable
closed-loop task-frame control.
