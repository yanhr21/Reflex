# World-Model Task Rebinding

## One Sentence

Train the manipulation skill in static scenes, but deploy it in dynamic scenes
by using a streaming object-centric world model to predict future task frames,
then use geometry, IK, servo, and continuability checking to reconnect the
frozen policy only when the real state is locally in-distribution again.

## Core Claim

The base DP/VLA should not be asked to absorb absolute dynamic OOD directly.
If the target hole moves outside the static training distribution but remains
reachable, the dynamic part should be handled outside the policy in task-frame
space: track objects, predict the future hole frame, bridge the peg to a
continuable relative state, and then hand control back to the frozen skill.

Workability check: this is reasonable in the current repo because the state
observation exposes peg, TCP, qpos, and hole pose for debugging and upper
bounds. But the actual method claim cannot stop at state: it must replace
oracle state with RGB-D-derived slots or latents before controller success is
treated as evidence.

## What This Is Not

This is not scene restoration as the main objective. It is also not only
failure detection. The objective is task completion under external changes:
the task state can change, the future task frame can move, and the system must
rebind execution to the new physical situation.

Workability check: this avoids reusing weak previous results as the method.
The old work is useful only as negative evidence that geometry alignment alone
does not guarantee policy continuation.

## System Sketch

```text
streaming state or vision tracker
  -> object slots: hole, peg, TCP, qpos, grasp/contact predicates
  -> object-centric world model predicts future task frames
  -> planner searches future tau, bridge action, and policy phase
  -> feasibility and continuability scoring
  -> servo/IK/MPC bridge in real state
  -> frozen DP/VLA only when continuable
  -> rebind whenever prediction and observation diverge
```

Workability check: each box can be tested separately with simulator state for
failure localization, but state-only success is scaffold evidence. The final
method path is RGB-D -> object slots/latents -> world model -> rebinding
controller -> final-state success plus inspected video/replay evidence.

## Why A State World Model First

The current base DP uses `obs_mode=state`; the processed demos contain
`obs: (T, 43)`, actions, and env states, but no RGB frames. A scratch video
world model from 1000 demos would be underpowered and would hide the real
question. The first world model should predict object slots and task frames.

Workability check: low-dimensional object dynamics is learnable with moderate
sim rollout data, and its errors are measurable in meters, radians, success
probability, and calibrated uncertainty.

## Why RGB-D Still Matters

RGB-D is needed for realism and for comparison to DDP/VLA-style systems, and
it is required for the current method evidence. The visual path should
estimate object slots and uncertainty; the controller should still reason in
task-frame space.

Workability check: state-oracle experiments localize geometry, prediction, and
control failures, but they do not prove the method. If RGB-D slot extraction is
weak, the aligned fix is to improve the RGB-D representation or uncertainty
handling, not to report oracle-state controller success as the method.

## General Cases

The same receding-horizon loop should handle:

- hole moves and stops
- hole keeps moving
- hole reverses right before insertion
- peg is disturbed, dropped, or no longer grasped
- target is reachable but outside the base policy's absolute training range

Workability check: these should not become separate hard-coded regimes. They
are different observations under the same loop: update slots, update future
task-frame predictions, update feasibility, and rebind.

## Boundary Conditions

The method should report infeasible instead of pretending to be general when:

- the hole is outside reachable workspace
- target motion exceeds tracking or insertion bandwidth
- the peg is not visible or cannot be regrasped
- the frozen policy has no continuable region from reachable states
- uncertainty is too high to choose a safe bridge

Workability check: explicit infeasibility is part of the method. A real closed
loop must know when the physical problem is not solvable by the available
robot, sensors, and skill library.
