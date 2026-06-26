# Continuability

## Goal

Learn when the frozen base DP can finish from a candidate real state and task
frame. This is the gate that prevents sending absolute OOD states directly into
the static policy.

Workability check: without `C_pi`, the system falls back to hand thresholds.
That would not be general enough for moving, reversing, and dropped-peg cases.

## Definition

```text
C_pi(s, task_frame, phase) =
  probability that frozen policy pi completes the task
  when resumed from state s under the current task binding
```

The label comes from actual frozen-DP rollouts, not from geometric closeness.

Workability check: this makes continuability empirical. A state is continuable
only if the policy can actually use it.

## Candidate State Generation

Create candidate states from:

- static demo states
- DP rollout states
- dynamic perturbation states
- bridge endpoint states
- nearby perturbed peg/hole/TCP states
- regrasp endpoint states
- intentionally bad states for negative examples

Workability check: if positives only come from demos, `C_pi` will not learn the
boundary needed for dynamic deployment. Hard negatives are required.

## Labels

For each candidate:

1. Reset or bridge simulator to the candidate state.
2. Query the frozen DP with the real state history.
3. Roll out for a fixed remaining budget.
4. Save success, final relative geometry, timeout, collision, drop, and
   infeasible reason.

Workability check: labels are expensive but parallelizable on Slurm. Start with
small balanced shards and calibrate before scaling.

## Model Inputs

Use object-centric features:

```text
peg pose in hole frame
peg head in hole frame
TCP pose in hole frame
qpos and qvel
grasp/contact predicates
hole velocity in world and task frame
phase estimate
remaining budget
world-model uncertainty
```

Workability check: these features describe what matters for continuation, not
the absolute scene location that caused OOD in the first place.

## Metrics

Report:

- AUROC/AUPRC for success prediction
- calibration curve
- false-positive rate at high continuability threshold
- success rate when DP is gated by `C_pi`
- regret versus oracle best handoff time

Workability check: false positives are the most damaging error. A conservative
gate that delays handoff is acceptable; a gate that hands off too early breaks
the task.
