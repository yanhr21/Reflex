# Joint Overfit Contract

Date: 2026-07-09

This file defines the immediate A/B/C/D overfit contract before full joint
training or closed-loop evaluation. It is not method evidence by itself.

## Gate

The immediate guard is:

```bash
scripts/world_model/require_dataset_training_inputs_ready.sh joint_overfit_abcd
```

This guard requires A full RGB and B/C/D production indexes. It intentionally
does not require E, because E depends on Cosmos/readout predicted future target
frames and comes after the first joint overfit.

The later all-class guard remains:

```bash
scripts/world_model/require_dataset_training_inputs_ready.sh full_joint
```

`full_joint` still requires E.

## Class Roles

A static expert:

- allowed for protected DP behavior cloning / distillation;
- allowed for Cosmos static future prediction;
- allowed for action-target sanity checks against official
  `pd_ee_delta_pose`;
- not dynamic recovery evidence.

B dynamic RGB observation:

- allowed for Cosmos future RGB / latent prediction;
- allowed for target-frame readout, trajectory consistency, and uncertainty;
- not allowed for positive DP behavior cloning from failed or incidental
  actions.

C frozen-DP dynamic outcome:

- allowed for outcome labels, discrepancy, infeasible/no-progress, and
  contrastive losses;
- success and failure are both labels;
- physically valid frozen-DP success must not be rejected or rewritten as
  failure;
- not allowed for positive DP behavior cloning;
- invalid only for forbidden mechanisms such as state intervention, snap /
  teleport, or target-assisted self-insertion.

D future-frame cooperation teacher:

- allowed for moving-frame adapter / residual supervision;
- allowed for phase, timing, and relative velocity supervision;
- allowed as teacher-only action evidence because the controller uses
  ground-truth future target trajectory during data generation;
- not allowed as deployed-method success.

## Tiny Overfit Slice

The first overfit slice should contain a small, inspectable set:

- A: static expert examples for the protected DP action contract;
- B: dynamic RGB examples for future visual / target-frame prediction;
- C: frozen-DP dynamic examples for discrepancy / outcome classification;
- D: teacher examples for moving-frame adapter actions.

Every selected row must keep:

- sample id, split, dataset class, allowed losses;
- video path and decoded frame count;
- trace path and action / motion / task row counts;
- summary and manifest path;
- source checkpoint or source demo reference;
- whether the row is method evidence, teacher evidence, or diagnostic-only.

## Required Overfit Evidence

Before full training, the overfit run must write:

- dataset manifest with selected A/B/C/D rows;
- model/config manifest using the real DP checkpoint and real Cosmos-3
  checkpoint path;
- action-target sanity report for DP-compatible `pd_ee_delta_pose` chunks;
- future-state imagination or latent prediction report for the same temporal
  windows;
- alignment review proving the action chunk and future-state target refer to
  the same physical episode;
- explicit loss breakdown, at least DP/action loss, Cosmos future loss, and
  any adapter / discrepancy losses.

## Prohibited Shortcuts

- Do not replace DP, Cosmos-3, or Dream Diffusion Policy with a toy model.
- Do not train C frozen-DP failed actions as positive expert actions.
- Do not treat C success as invalid.
- Do not use hidden future target labels as deployed controller inputs.
- Do not call teacher-only D success a deployed-method success.
- Do not use state edits, saved-state replay, snap / teleport, geometric final
  placement, or target-assisted self-insertion as success.
