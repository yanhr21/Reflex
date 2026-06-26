# sample00 Source-Suffix Offset Coverage

Date: 2026-06-22

## Question

Why did corrected sample00 iter0 still have no usable action while sample05
could succeed with source-suffix chunks?

## Evidence

Corrected sample00 root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_panel0245_cap_scenario_k2_dist002_exec8_nodpunsafe_20260622_alloc145920`

The iter0 live action chunk had `213` candidates but
`source_suffix_candidate_count=0`.

The source insertion suffix bank contains `4711` suffixes, including `281`
from scenario `hole_late_move_stop`.

sample00 iter0 live state before controller:

`[-0.132808, 0.029728, -0.011192]`

Distance audit:

- same scenario, offsets `32,24`: `84` suffix starts, nearest distance
  `0.0207339`, `0` within cap `0.02`, `7` within `0.04`;
- same scenario, all offsets: `281` suffix starts, nearest distance
  `0.0065169`, `8` within cap `0.02`, `62` within `0.04`;
- nearest same-scenario in-cap offsets are `48` and `64`.

The attempted diagnostic rerun with offsets `64,48,32,24` did not execute as
method evidence. Slurm revoked allocation `145920`; `sacct` shows step
`145920.316` was cancelled after `00:00:04`, and tmux output reported
`salloc: Job allocation 145920 has been revoked`.

## Conclusion

sample00 currently has an action-candidate coverage problem. The failed live
bank used offsets `32,24`, which missed same-scenario source suffixes under
the active distance cap. This is not evidence that the distance threshold
should be loosened.

## Next Action

Acquire a fresh tmux-held GPU allocation and rerun only the one-iteration
sample00 diagnostic with offsets `64,48,32,24`, same-scenario suffixes,
distance cap `0.02`, 8-step suffix execution, and no pure DP-prior execution
while live `C_pi=false`.

The first pass criterion is simple: `source_suffix_candidate_count` should
become greater than `0`. If it does, evaluate whether those candidates improve
the real after-state and create DP96 headroom. If it does not, inspect the
live candidate-construction path rather than changing the method objective.
