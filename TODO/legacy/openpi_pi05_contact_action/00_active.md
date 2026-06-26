# OpenPI-First Full-Episode TODO

Date: 2026-06-26

## Protocol Correction

- [x] Mark previous OpenPI saved-snapshot takeover experiments as legacy
      diagnostics, not main method evidence.
- [x] Archive previous takeover-oriented OpenPI plan/TODO under
      `legacy/plan_todo_openpi_takeover_protocol_error_20260626/`.
- [ ] Audit remaining docs/scripts for wording that presents DP handoff or
      saved-snapshot takeover as active OpenPI method success.

## Static OpenPI Baseline

- [ ] Build or reuse a Slurm-only OpenPI full-episode rollout wrapper that
      starts from a static PegInsertionSide reset state and lets OpenPI execute
      every action from step `0`.
- [ ] Run a small static full-episode panel inside a tmux-held Slurm
      allocation.
- [ ] Save final metrics, video/contact sheets, action traces, and exact
      checkpoint/config/norm-stat provenance.
- [ ] Interpret failure before moving to dynamic WM integration. If static
      full-episode OpenPI fails, diagnose action-space, observation, prompt,
      normalization, checkpoint, and dataset alignment first.

## Dynamic OpenPI Baseline

- [ ] Build or reuse a Slurm-only OpenPI dynamic full-episode rollout wrapper.
      OpenPI must execute from step `0`; DP must not run in the main rollout.
- [ ] Run a small dynamic panel without world-model conditioning.
- [ ] Record whether OpenPI handles target motion without WM help.

## World-Model Conditioning

- [ ] Define the causal WM output that OpenPI should consume: future scene
      state, future task-frame trajectory, future `x_t`, or another
      OpenPI-native conditioning representation.
- [ ] Audit current Cosmos/WAM checkpoints and exports only for compatibility
      with the corrected protocol. Do not reuse old takeover-controller
      claims.
- [ ] Build an OpenPI + WM full-episode dynamic rollout path where OpenPI
      starts at step `0`, WM predicts after observed target motion, and OpenPI
      completes the task.

## Legacy Boundary

- [ ] Keep old saved-snapshot takeover replay outputs in legacy. They can be
      referenced as negative diagnostics only.
- [ ] Keep useful data/checkpoints/tooling active; do not move accepted 733
      source data, approved RGB/action/state exports, preserved OpenPI
      checkpoints, or LeRobot conversion assets unless explicitly requested.
- [ ] Do not claim DP96 handoff success as OpenPI success.
