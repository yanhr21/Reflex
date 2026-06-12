# Cosmos3 300-Step World Model Overview

## Objective

Train and test a Cosmos3-based world/action model for dynamic
`PegInsertionSide-v1` completion without degrading the task into short
truncated clips.

The model should support the user's desired execution loop:

1. Before the target object moves, the frozen/static DP may continue executing.
   The world model monitors the target object and predicts whether/where it
   will move.
2. When target motion is observed or strongly predicted, the world model
   produces a causal rollout of the changed task state, future RGB
   reconstruction, and executable robot/peg actions so the policy can continue
   or resume under live re-observation.
3. Peg disturbance/drop cases must preserve or recover the grasp before
   continuing insertion. Failed old controller branches are not positive
   training evidence.

## Non-Negotiable Reset

- The full1000 data does not need to be regenerated.
- "300 frames" means the total episode horizon, not an extra 300 future frames.
- The accepted source data has 300 action steps and 301 RGB/state frames when
  frame 0 is included.
- Do not use the rejected 129-frame / 128-action chunk construction.
- Do not use old 93-frame exports, cropped comparisons, stale SFT checkpoints,
  state-only/object-slot world models, or hand-coded controller results as
  active method evidence.
- If Cosmos tooling cannot support the full episode/equal-length contract, stop
  and report the concrete limitation instead of falling back to truncation.

## Active Data And Model Line

- Visual input: approved RGB videos only. Depth is not an active visual input
  for the ManiSkill Cosmos/DROID WAM SFT.
- State/action/proprio from simulator H5 may be used only as causal metadata,
  supervision labels, diagnostics, and readout targets. Future object states
  must not be provided as privileged conditions during controller-facing
  inference.
- Primary backbone: local Cosmos3 DROID/Policy checkpoints under
  `checkpoints/cosmos3/`. If a stronger applicable Cosmos3 variant is to be
  considered, audit it explicitly before training.

## Review Gate

This plan defines the next training/testing direction only. No new training,
generation, controller integration, or evaluation job should be started until
the user reviews this reset.
