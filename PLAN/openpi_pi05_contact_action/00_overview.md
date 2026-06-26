# OpenPI-First Full-Episode Plan

Date: 2026-06-26

## Boundary

This replaces the previous OpenPI takeover/saved-snapshot plan. The old plan is
archived under:

`legacy/plan_todo_openpi_takeover_protocol_error_20260626/PLAN_openpi_pi05_contact_action_takeover_legacy`

The previous takeover protocol is not valid main method evidence because it
starts from DP/Cosmos-derived dynamic snapshots and often evaluates DP96
continuation. It may be used only as diagnostic history.

## Correct Method

OpenPI/pi0.5 must replace Diffusion Policy as the policy from the beginning of
the episode.

Static scene:

- initialize a static PegInsertionSide episode;
- run OpenPI from step `0` through the full horizon;
- no DP execution and no saved-snapshot takeover;
- evaluate final success, inserted/contact-stable predicates, grasp, action
  traces, and video/contact sheets.

Dynamic scene:

- run OpenPI from step `0`;
- when the scene changes, use a causal world model to predict future scene/task
  state or future `x_t`;
- condition OpenPI on current observation/history plus allowed WM output;
- OpenPI executes the action chunks and finishes the task;
- no DP handoff in the main method rollout.

## Reusable Assets

Keep these as active/reusable:

- accepted 733 ManiSkill H5 trajectories;
- approved RGB/action/state exports with the `301/300` contract;
- official OpenPI/pi0.5 configs and checkpoint-loading code;
- LeRobot conversion, norm-stat, training, and inference tooling;
- preserved OpenPI checkpoints for diagnostics or initialization;
- contact-state sheet tooling.

## Legacy Diagnostics

The following are legacy diagnostics, not active method evidence:

- saved dynamic takeover snapshot replay;
- DP96 continuability or DP96 handoff success;
- object17 privileged-state takeover replay;
- near-contact takeover replay;
- scorer-only or candidate-selection branches.

These results can explain failures, but they must not define the success
protocol.

## Required Experiment Order

1. OpenPI-only static full-episode baseline.
2. OpenPI-only dynamic full-episode baseline, with no world model.
3. World-model inference audit: causal future scene/task-state or future `x_t`
   prediction from OpenPI-started dynamic episodes.
4. OpenPI + world-model dynamic full-episode rollout.

Only step 4 is the intended method result. Steps 1-3 are required gates.

## Evidence Requirements

For any main result, record:

- full horizon length;
- OpenPI checkpoint/config/norm-stat source;
- action space and action dimension;
- whether WM conditioning was used;
- final simulator success;
- inserted/contact-stable/grasp metrics;
- video/contact-sheet review;
- clear statement that DP did not execute in the main rollout.
