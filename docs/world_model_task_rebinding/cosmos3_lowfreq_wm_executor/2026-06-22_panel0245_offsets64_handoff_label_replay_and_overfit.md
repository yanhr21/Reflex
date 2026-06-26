# 2026-06-22 Panel0245 Handoff Label Replay and Scorer Sanity

## Purpose

The `2/4` panel showed that instantaneous `C_pi` is not a reliable handoff
label. This replay checks the real target: after a short live candidate chunk,
can frozen DP actually finish or remain continuable for the real simulator
state?

This is label generation and scorer debugging. It is not closed-loop method
success evidence.

## Targeted Replay

Panel root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658`

Selected/source-suffix replay root:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_panel0245_offsets64_selected_sourcesuffix_dp96_20260622_alloc146658`

Main summary:

- `valid_records=97`
- `selected_records=42`
- `dp_rollout_success_count=54`
- `dp_rollout_continuable_count=55`
- `dp_rollout_final_contact_stable_count=60`
- source-suffix-name rows: `76`
- source-suffix-name DP96 successes: `53`
- selected-candidate DP96 successes: `15/42`
- `C_pi=true` but DP96 failed: `2`
- `C_pi=false` but DP96 succeeded: `45`

Per scenario:

| scenario | rows | selected DP96 success | source-suffix DP96 success | DP96 success | key reading |
|---|---:|---:|---:|---:|---|
| `hole_late_continuous_insert` | 40 | 9/10 | 33/40 | 33/40 | source suffix broadly works |
| `hole_late_move_stop` | 28 | 2/13 | 11/22 | 11/28 | good chunks exist, selected path unstable |
| `hole_late_reverse` | 17 | 0/13 | 1/4 | 1/17 | selector missed the one DP96-positive source suffix |
| `hole_late_sine` | 12 | 4/6 | 8/10 | 9/12 | source suffix and DP handoff work |

DP-prior replay root:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_panel0245_offsets64_dpprior_only_dp96_20260622_alloc146658`

DP-prior summary:

- `valid_records=42`
- `dp_rollout_success_count=16`
- `dp_rollout_continuable_count=18`
- `dp_rollout_final_contact_stable_count=18`

## Converted Scorer Dataset

Combined conversion root:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_panel0245_offsets64_selected_sourcesuffix_dpprior_dp96_20260622_alloc146658`

Summary:

- `valid_rows=139`
- `base_groups=42`
- `groups_with_dp_prior=42`
- `groups_with_source_suffix_dp96_success=19`
- `dp_rollout_success_count=70`
- family counts: `dp=42`, `source_suffix=76`, `scale=8`,
  `short8=12`, `short12=1`

## Short Overfit Sanity

Overfit root:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_panel0245_offsets64_handoff_rank_overfit100_20260622_alloc146658`

It ran `100` steps with `val_fraction=0`. Result:

- DP-prior handoff success: `16/42`
- selected handoff success: `20/42`
- handoff oracle success: `20/42`
- selected-minus-DP handoff success: `+0.0952`
- selected non-DP fraction: `0.714`
- formal floor: false

Plain reading: the new labels are learnable in an overfit sanity. This does
not prove generalization and must not drive method evidence.

## Single-Panel Formal Attempts

Two single-panel formal attempts were started and then interrupted because
they were too small for the held H200 allocation and showed low GPU
utilization. They are not method evidence.

Small model root:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_panel0245_offsets64_handoff_rank_formal3h_20260622_alloc146658`

Large model root:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_panel0245_offsets64_handoff_rank_formal3h_h4096_20260622_alloc146658`

Boundary:

- 1 GPU in allocation `146658`
- `min_wall_seconds=10800`
- `max_wall_seconds=10800`
- train groups: `32`
- validation groups: `10`
- rank loss weight: `1.0`
- score includes DP-rollout handoff success weight `1.0`

Early step `1000` is not a final result. It shows train improving but
validation worse than DP on handoff success:

- validation DP-prior handoff success: `7/10`
- validation selected handoff success: `5/10`
- train DP-prior handoff success: `9/32`
- train selected handoff success: `13/32`

Later early evaluations stayed below DP on validation handoff success. Because
the dataset had only `139` rows, the run also underused the GPU. The aligned
replacement is the union+panel formal run below.

## Union + Panel Formal Run Started

The old source-suffix union dataset and the new panel0245 dataset had `5`
overlapping UUID strings, so they were not concatenated directly. A namespaced
combined dataset was created:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_source_suffix_union_plus_panel0245_offsets64_dp96_namespaced_20260622_alloc146658`

Summary:

- old union rows: `8877`
- new panel rows: `139`
- combined base rows: `83`
- combined outcome rows: `9016`
- namespace policy: prefix old UUIDs with `oldunion__` and new panel UUIDs
  with `panel0245__`

Current formal root:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_source_suffix_union_plus_panel0245_handoff_rank_formal3h_20260622_alloc146658`

Manifest at launch:

- `num_joined_candidate_rows=9016`
- `num_train_groups=62`
- `num_val_groups=21`
- `visible_cuda_device_count=1`
- `min_wall_seconds=10800`
- `max_wall_seconds=10800`
- `rank_loss_weight=1.0`
- score weights: handoff success `8.0`, continuability `4.0`, insertion
  `0.5`, progress/progress-delta `0.5`, state penalty weights
  `[0.05, 0.1, 0.2]`

Step `1` is only initialization evidence:

- validation DP-prior handoff success: `4/21`
- validation selected handoff success: `4/21`
- train DP-prior handoff success: `12/62`
- train selected handoff success: `11/62`

The formal conclusion must wait for the 3-hour summary from this union+panel
run.
