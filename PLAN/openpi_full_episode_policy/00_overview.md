# OpenPI Full-Episode Policy Plan

## 1. Goal

Train and evaluate an official OpenPI/pi0.5 policy that completes
PegInsertionSide from the beginning of the episode. The policy must place the
real simulated peg into the real simulated target hole through live simulator
execution.

The method is not DP takeover. It is not "DP first, OpenPI later." It is not a
saved dynamic snapshot replay result. OpenPI is the action policy from step `0`
in both static and dynamic scenes.

## 2. Expected Result

The expected evidence is:

- Static scene: OpenPI-only full-episode rollouts with success/failure metrics,
  final simulator state, and rendered videos/contact sheets.
- Dynamic scene: OpenPI starts the episode, observes the dynamic change through
  causal history, receives world-model-imagined future scene/X-chat/task
  conditions after the event, and continues executing OpenPI actions to finish
  insertion.
- Every reported result must state whether it is full-episode live execution,
  offline replay, diagnostic inference, or legacy snapshot/takeover evidence.

## 3. Data Contract

Initial adaptation data source:

- `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733`

Required contract:

- 733 accepted ManiSkill PegInsertionSide trajectories as the initial real-data
  source.
- 301 RGB/state frames when reset frame is included.
- 300 action steps.
- Action space must be compatible with the official OpenPI transform path. The
  source actions are `pd_ee_delta_pose`, so start from `pi05_base` plus a fresh
  normalization pass unless a concrete adapter proves another checkpoint's
  action convention is valid.
- RGB/action/state exports may be used only if the exact frame/action contract
  is preserved.

## 4. Static Scene Plan

Static scene means the target does not move during evaluation.

Execution protocol:

1. Convert/audit the 733 trajectories into an OpenPI-compatible LeRobot/RLDS
   style dataset.
2. Compute normalization stats from the real 733-derived data.
3. Adapt official `pi05_base` using official OpenPI training code in a
   tmux-held interactive Slurm allocation.
4. Run OpenPI from step `0` for the whole 300-step episode.
5. Render videos/contact sheets and record success, insertion depth, contact,
   final peg/hole relation, and failure mode.

Static success requires full-episode OpenPI-only live rollout evidence. Offline
inference or snapshot replay is diagnostic only.

## 5. Dynamic Scene Plan

Dynamic scene means the target/hole changes during the episode.

Execution protocol:

1. Start the episode with OpenPI from step `0`.
2. Use only observed history/current perception to detect scene change.
3. After the event, invoke the world model causally from observed RGB/state
   history and current task context.
4. The world model imagines future scene/task evolution and produces the
   future X-chat/task condition needed by OpenPI.
5. OpenPI consumes current observation plus world-model condition and continues
   producing robot actions.
6. Execute only short horizons before re-observing; refresh the world-model
   context with real observations.

Dynamic success requires live simulator evidence that OpenPI, not DP, executed
the episode and completed insertion in the changed world.

## 6. World Model Role

The world model is not the robot action policy. Its role is to provide
task-relevant future scene imagination and future X-chat/task conditioning for
OpenPI after dynamic changes.

Required world-model evidence before dynamic claims:

- Input boundary: current/history observations only, no future ground-truth
  privileged state.
- Output boundary: future scene/task/X-chat condition usable by OpenPI.
- Receding interface: short prediction, OpenPI action, real re-observation,
  updated prediction.
- Visual/task sanity checks showing the imagined target/hole state is coherent
  enough to condition OpenPI.

## 7. Forbidden Evidence

The following must remain legacy or diagnostic only:

- DP prefix followed by OpenPI action.
- OpenPI saved-snapshot takeover from a DP-generated state as the main result.
- Contact-suffix or near-contact datasets presented as full-episode OpenPI
  success.
- Scorer-only selection, DP action-bank replay, or hand-coded bridge as the
  active method.
- Toy in-repo action models presented as OpenPI progress.

## 8. Execution Discipline

- Training/evaluation/rendering/replay must run in tmux-held interactive Slurm
  allocations, not on the login node.
- New OpenPI training must run for at least one GPU-hour on real 733-derived
  data before it is interpreted as a training result.
- Short smoke checks may be labeled only as diagnostics.
- Every run must produce a manifest with command, allocation, data source,
  checkpoint, frame/action contract, and evidence type.
