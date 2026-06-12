# Rebinding Controller

## Online Objective

At each control cycle, choose a future task binding and bridge:

```text
argmax over tau, bridge, phase:
  predicted_success
  - uncertainty_cost
  - time_cost
  - collision_cost
  - policy_ood_cost

subject to:
  hole(tau) reachable
  peg state recoverable
  bridge feasible
  C_pi(state_tau, task_frame_tau, phase) high enough
```

Workability check: this is an optimizer over physical options, not a label
switch. It can be approximated first by sampling candidate `tau` and bridge
targets.

## First Controller

Use a simple receding-horizon implementation:

1. Predict hole task frames for a short horizon.
2. Sample candidate intercept times.
3. Build pre-insert and insert target frames for each candidate.
4. Run reachability and collision checks.
5. Servo or IK toward the best candidate for a short chunk.
6. Re-evaluate after every chunk.
7. Hand off to DP only if `C_pi` is high.

Workability check: this is implementable without a full MPC solver. If simple
sampling works, later MPC can improve smoothness and speed.

## Corrected Controller Role

The controller is not allowed to become a separate hand-coded manipulation
policy that destroys the frozen DP skill. Its first role is to preserve the
static DP's grasp-hold and insertion competence. Its second role is to extend
that competence into dynamic scenes by conditioning on world-model predicted
task trajectories.

The corrected policy path is DP-preserving distillation:

1. Keep official/static DP demonstrations as base-skill preservation data.
2. Add takeover teacher samples only when video and metric evidence show the
   peg is actually held and the final task state is successful.
3. Condition the learned policy on current RGB-D-derived slots plus a short
   world-model predicted trajectory, not only on a one-step geometric target.
4. Train with mixed losses: base DP behavior cloning, takeover behavior
   cloning, grasp-hold consistency, task-frame progress, and conservative
   handoff/retreat labels.
5. Evaluate the learned policy visually and metrically before calling it a
   controller result.

Workability check: if a takeover video visibly loses the peg or the peg drifts
away from the gripper, it is a negative diagnostic regardless of transient
numeric slot predicates. Such rollouts must not be distilled as positive
controller-teacher data.

This is the part of the plan that should borrow from Dream Diffusion Policy:
use a world model to imagine/predict where the robot should move, but train
the policy to execute that imagined path while preserving the original
imitation skill. The world model supplies the predicted path; the policy learns
the actions that keep the manipulator physically competent along that path.

2026-06-07 controller boundary: fixed action-prior replay has been tested and
is not enough. DP-only and converted official-planner insertion chunks are
allowed as action-prior data sources, but the controller must not keep tuning
fixed replay gates after those probes failed. The next controller should
generate or optimize a short action chunk from the current live state,
conditioned on Cosmos/RGB-D task state and robot/object state, execute only a
short prefix, then re-observe and regenerate. This is the DDP-style path that
preserves task-frame rebinding; it is not a hand-coded case split or a
standalone bridge replacement for the frozen DP skill.

## General Case Handling

The controller should not branch on case names. The same loop handles:

- stopped target: predicted velocity decays, best `tau` becomes near-current
- continuously moving target: best `tau` is an intercept time
- sudden reverse: discrepancy spikes, old plan loses value, rebind happens
- peg disturbed in gripper: peg/TCP relation changes, bridge target updates
- peg dropped: grasp predicate false, insertion binding invalid, regrasp task
  becomes the highest-value prerequisite

Workability check: each scenario changes object slots and feasibility. If the
implementation needs hard-coded case names, the abstraction is too weak.

## Skill Library

Minimum skills:

- hold current pose
- servo TCP to task-frame pre-insert pose
- align peg head in hole frame
- short insertion motion in moving task frame
- retreat to safe pose
- regrasp peg
- report infeasible

Workability check: these are physical primitives with measurable endpoints.
They do not need to be learned in the first version.

## Handoff Rule

The frozen DP can act only when:

```text
C_pi >= threshold
relative peg/hole velocity within learned support
grasped is true
world-model uncertainty below limit
remaining budget is sufficient
```

Workability check: the handoff rule is intentionally conservative. The system
should bridge longer rather than push impossible states into the base policy.

## Infeasibility

Return infeasible when the target is unreachable, too fast, too uncertain, or
the peg cannot be recovered. Count this separately from policy failure.

Workability check: correct infeasibility is necessary for a real closed loop.
It also prevents inflated claims on physically impossible test cases.
