# ManiSkill TODO Index

Date: 2026-07-02

This file is only an index. Concrete tasks live in the phase folders.

## Required Order

1. `TODO/01/dp_static.md`
2. `TODO/02/cosmos_imagination.md`
3. `TODO/03/integration.md`
4. `TODO/04/oracle.md`
5. `TODO/05/live.md`

## Current Rule

Do Phase 01 first. Do not start Oracle as the next experiment until DP static
and Cosmos imagination have been checked. Phase 03 must convert Cosmos output
into controller-facing task-state, candidate-score, and trust-gate signals
before Phase 04 Oracle or Phase 05 live claims.

## Error Record

- [x] Document the old snap / suction error in
      `docs/oracle_snap_error_review.md`.
- [x] Mark old geometric final-seat and source-state Oracle videos invalid as
      physical success.
- [x] Add current literature review for controller-facing world-model bridge:
      `docs/controller_facing_world_model_research_20260702.md`.

## Workspace Rule

- [ ] Put new experiment outputs under `experiments/maniskill/runs/<phase>/`.
- [ ] Put new logs under `logs/<phase>/`.
- [ ] Move any experiment proven useless or invalid under
      `/public/home/yanhongru/ICLR2027/archive/Reflex/`, preserving the original
      repository-relative structure.
- [ ] Before reporting success, inspect wrapper, manifest, summary, relevant
      Python path, action trace, and video for `set_pose`, `set_state`,
      `set_state_dict`, source-state restore, saved-state replay, future
      labels, geometric final placement, hand-selected suffixes, and
      discontinuous pose jumps.
