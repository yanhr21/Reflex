# Dynamic State Data

## Goal

Create a new dataset that exposes the dynamic task-rebinding problem directly:
object slots, actions, perturbations, bridge actions, frozen-policy rollouts,
and final task outcomes.

Workability check: the method cannot be learned or evaluated from only static
expert demos. The dynamic dataset is the first real deliverable.

## Rollout Families

Generate Slurm rollouts for:

- static frozen-DP baseline
- hole moves then stops
- hole keeps moving with bounded velocity
- hole reverses near the pre-insertion or insertion phase
- peg is knocked in gripper but remains grasped
- peg is dropped and must be regrasped
- oracle or geometric servo bridges toward predicted task frames
- failed bridge attempts and infeasible states

Workability check: these are data-generation regimes, not hand-coded policy
branches. The online method will see object slots and uncertainty, not a case
label like "moving" or "reverse".

## Episode Schema

Each episode should save:

```text
obs_state[t]
action[t]
env_state[t]
hole_pose[t], hole_velocity[t]
peg_pose[t], tcp_pose[t], qpos[t], qvel[t]
peg_head_in_hole_frame[t]
grasped[t]
inserted[t]
perturbation metadata
bridge metadata
policy query metadata
success_once, success_at_end
failure or infeasible reason
```

Workability check: every field is available from simulator state or existing
task helpers. If a field is not reliably available, it should be explicitly
logged as unknown instead of inferred silently.

## First Implementation

Start with oracle object slots from simulator state. Do not block on RGB-D.
Implement dynamic perturbation wrappers and a lightweight state trace writer.
Use simple deterministic perturbation schedules first, then randomize position,
velocity, reversal time, and drop events.

Workability check: oracle-state data is not the final method, but it tests the
control and modeling problem cleanly. If oracle-state rebinding cannot work,
visual tracking will not solve the underlying issue.

## RGB-D Companion Data

Dynamic state rollouts should be treated as the control/debug spine, not the
only dataset. For vision and DDP/VLA comparison, add a companion RGB-D export
path with synchronized:

```text
rgb[t], depth[t]
state[t], actions[t], env_state[t]
camera parameters
hole/peg/TCP pose labels
visibility/confidence labels when available
```

For large RGB-D generation, use dense Slurm GPU allocation. Up to 8 nodes / 64
GPUs is acceptable when justified, but avoid sparse allocations that occupy many
servers with only one GPU per server.

Workability check: RGB-D should not block the first state-space proof, but it
must be generated from the same task families so the visual version is not a
different problem.

## Validation

Before training any model, verify:

- static replay and frozen-DP rollout still match expected behavior
- dynamic perturbations actually occur before insertion completion
- saved object slots reconstruct the visual scene when rendered
- grasp/drop labels match simulator state
- success metrics are computed from real final state, not from intended target

Workability check: bad labels would make the world model look plausible while
training on the wrong problem. Dataset validation is a gate, not a nice-to-have.
