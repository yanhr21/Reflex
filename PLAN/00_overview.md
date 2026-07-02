# ManiSkill Plan Index

Date: 2026-07-02

This file is only an index. Concrete steps live in the phase folders.

## Active Assets

- 733 data: `experiments/maniskill/data/fix3_733/`
- DP checkpoint: `experiments/maniskill/dp_checkpoint/run_90201/checkpoints/`
- RGB Cosmos-3 checkpoint:
  `experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam/`

## Required Order

1. `PLAN/01/dp_static.md`
2. `PLAN/02/cosmos_imagination.md`
3. `PLAN/03/integration.md`
4. `PLAN/04/oracle.md`
5. `PLAN/05/live.md`

Do not skip ahead to Oracle or live control before the DP static and Cosmos
imagination checks are understood.

## Error Record

The previous Oracle/geometric final-seat snap problem is documented in:

- `docs/oracle_snap_error_review.md`

Summary: the snap was caused by explicit simulator-state intervention such as
`set_pose`, then reported too close to success. It was not database failure,
Git failure, or spontaneous physics behavior.

## Research Guidance

The current controller-facing world-model review is documented in:

- `docs/controller_facing_world_model_research_20260702.md`
- `docs/dream_diffusion_policy_review.md`

The active lesson is that Cosmos-3 future RGB must be converted into
controller-facing variables before it can help insertion:

- task-frame and insertion-axis estimates;
- peg/hole displacement and near-hole/preinsert gates;
- candidate action scoring;
- prediction-observation trust;
- executability checks against DP-compatible `pd_ee_delta_pose` actions.

Do not treat visually plausible future videos as sufficient method progress.
The bridge must prove that the future prediction changes action selection or
execution through a logged controller-facing signal.

## Workspace Rule

Active experiment outputs must go under short phase directories in
`experiments/maniskill/runs/`.

Logs must go under `logs/`.

Useless or invalid experiments must be moved under
`/public/home/yanhongru/ICLR2027/archive/Reflex/` after classification,
preserving the original repository-relative structure.

Before any run is called success, inspect the wrapper, manifest, summary,
relevant Python path, action trace, and video for `set_pose`, `set_state`,
`set_state_dict`, source-state restore, saved-state replay, future labels,
geometric final placement, hand-selected suffixes, or discontinuous pose jumps.
