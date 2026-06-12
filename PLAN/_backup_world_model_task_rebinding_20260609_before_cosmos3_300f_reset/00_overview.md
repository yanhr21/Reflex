# Overview

This plan is separate from `PLAN/README.md`. It is the active plan for
static-policy-to-dynamic-scene transfer through world-model task rebinding.

## Problem

The base policy is trained in static PegInsertionSide scenes. At test time the
hole may move, stop, reverse, or continue moving. The peg may also be disturbed
or dropped. The goal is not to restore the old scene; the goal is to finish the
original task in the changed world.

Workability check: the simulator gives enough state observability to test this
without solving vision first. If the state version fails, the RGB-D version
will not rescue the method.

## Method In One Loop

```text
for each control cycle:
  estimate object slots
  predict future task frames
  search tau, bridge, and policy phase
  score feasibility and C_pi continuability
  execute short bridge action
  hand over to frozen DP only if continuable
  interrupt and rebind if prediction disagrees with observation
```

Workability check: the loop can be implemented incrementally. Oracle state
slots, deterministic hole prediction, and simple servo are allowed as scaffold
pieces for localization and upper bounds. A method claim requires those oracle
pieces to be replaced by RGB-D-derived slots or learned representations at the
world-model/controller interface.

## Key Decision

The base DP is a static terminal skill prior, not the dynamic task solver.
Absolute OOD is absorbed by object-centric prediction and geometry control.
The frozen policy only receives states that are likely to be locally
continuable.

Workability check: this is only credible if `C_pi` predicts policy success
better than hand thresholds. That is a required experiment, not an assumption.

## File Map

- `01_assets_and_assumptions.md`: current assets and constraints.
- `02_dynamic_state_data.md`: dynamic rollout data generation.
- `03_object_state_world_model.md`: world model design and losses.
- `04_continuability.md`: frozen-policy continuation labels and model.
- `05_rebinding_controller.md`: online planning/control loop.
- `06_rgbd_and_baselines.md`: RGB-D path and DDP/VLA comparisons.
- `07_experiment_matrix.md`: evaluation protocol and pass/fail criteria.
