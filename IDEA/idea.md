# Current Idea: OpenPI-First World-Model Task Rebinding

Date: 2026-06-26

## Latest Correction

The active method is not an OpenPI takeover from a DP/Cosmos failure snapshot.
That protocol is legacy diagnostic only.

The correct method is:

1. Static scene: OpenPI/pi0.5 replaces Diffusion Policy from the first action
   through the end of the episode.
2. Dynamic scene: OpenPI/pi0.5 also starts from the first action. After the
   target/object scene changes, a world model predicts future scene/task state
   or future `x_t`, and OpenPI uses that causal prediction/conditioning to
   finish the task.
3. Diffusion Policy is only a historical baseline, teacher/data source, or
   comparison. It must not execute the first part of the method rollout, and a
   DP handoff result is not OpenPI method success.

Therefore, saved dynamic takeover snapshot experiments are not valid main
method evidence. They may remain archived as diagnostics showing that previous
OpenPI chunks preserved grasp but did not directly insert.

## One Sentence

Train and evaluate an official OpenPI/pi0.5 policy as the full manipulation
policy, then add a causal world-model conditioning path for dynamic scenes so
OpenPI can complete peg insertion after the target moves.

## Research Objective

The objective is task completion in the changed world. The system should not
restore an old scene layout, switch to hand-coded recovery, or hand execution
back to Diffusion Policy as the main method.

The desired final pipeline is:

```text
current RGB / state / object-task slots
  -> OpenPI executes from episode start
  -> target/object motion is observed causally
  -> world model predicts future scene/task state or future x_t
  -> OpenPI conditions on current observation plus WM output
  -> OpenPI completes insertion
  -> final simulator state and video/contact evidence verify success
```

## Static Baseline

The first required baseline is OpenPI-only static full-episode insertion:

- initialize the original static PegInsertionSide scene;
- run OpenPI from step `0` through the full horizon;
- use no DP execution and no takeover snapshot;
- record final success, inserted/contact-stable predicates, grasp, action
  traces, and video/contact sheets.

If this baseline fails, the blocker is OpenPI/ManiSkill action-space,
normalization, observation, or imitation alignment. Dynamic world-model work
should not be interpreted until this baseline is understood.

## Dynamic Method

The dynamic method must also start with OpenPI at step `0`.

When the object/target changes:

- perception or simulator-derived diagnostics provide only causal current and
  history information;
- the world model predicts future scene/task state or future `x_t`;
- OpenPI receives the current observation plus the allowed WM conditioning;
- OpenPI executes the action chunks;
- the loop reobserves and replans without using future ground-truth states.

DP may be reported as a baseline, but not as the first-stage executor or
handoff finisher for the main result.

## What The Old Ideas Contribute

The original world-model task-rebinding idea remains useful at a high level:
dynamic task completion should be solved by observing objects, predicting
future task frames, and rebinding action to the changed scene.

The contact-action reset contributed an important negative diagnosis:
scorer-only selection and weak local action generators cannot create the
missing insertion/contact behavior. The action model itself must be strong and
must produce real insertion actions.

The OpenPI/pi0.5 pivot contributed reusable infrastructure:

- official OpenPI/pi0.5 configs and checkpoint loading;
- accepted 733 ManiSkill trajectories converted to LeRobot-style data;
- OpenPI norm-stat, training, checkpoint-preservation, and inference tooling;
- object/task-frame conditioning diagnostics such as object17;
- evidence that previous takeover-style OpenPI chunks preserved grasp but did
  not directly insert.

Those contributions are infrastructure and diagnostics. They do not prove the
correct full-episode OpenPI method.

## Current Evidence Boundary

As of this cleanup, there is no accepted evidence that OpenPI completes either:

- static full-episode insertion from step `0`; or
- dynamic full-episode insertion with world-model conditioning.

The existing OpenPI saved-snapshot replay results are protocol-misaligned for
the main method because they start from DP/Cosmos-derived takeover states and
often evaluate DP96 continuation. They must be treated as legacy diagnostics.

## Data Boundary

The accepted 733 trajectories are still valuable. They contain successful
insertion behavior and can be used for OpenPI adaptation, normalization,
diagnostics, or teacher data.

If new data are generated, it should be because the corrected OpenPI-from-start
protocol exposes a concrete coverage gap, not because the old takeover replay
failed. New ManiSkill generation should target the missing behavior explicitly:
OpenPI-compatible full-episode static demonstrations, dynamic episodes from
step `0`, and contact-positive segments that remain consistent with the
official OpenPI action space.

## Required Success Evidence

A result can be treated as main method evidence only if it has:

- OpenPI executing from episode start;
- no DP handoff in the main rollout;
- causal observations and WM conditioning only;
- full horizon accounting;
- final simulator success state;
- inserted/contact-stable/grasp metrics;
- inspected video/contact artifacts.

Training loss, saved-snapshot takeover replay, DP96 continuability, or
privileged future-state conditioning are diagnostics only.
