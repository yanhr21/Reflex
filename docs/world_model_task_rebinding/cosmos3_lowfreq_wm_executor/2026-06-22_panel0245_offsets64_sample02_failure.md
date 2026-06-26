# 2026-06-22 Panel0245 Offsets64 Sample02 Failure

This note records the second completed sample in the strict offsets
`64,48,32,24` panel on allocation `146658`.

## Run

Sample directory:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658/sample_02_hole_late_reverse`

Panel settings kept the corrected strict boundary:

- same-scenario source suffixes
- source-suffix offsets `64,48,32,24`
- source-suffix max distance `0.02`
- source-suffix execution `8` steps
- pure `dp_prior` blocked while live `C_pi=false`
- full `301` RGB/state frames and `300` action steps

## Result

`sample_panel_result.json` reports:

- `final_observed_frames=301`
- `full_episode_length_ok=true`
- `video_file_contract_ok=true`
- `final_success=false`
- `final_peg_head_pos_at_hole=[-0.1129306257, 0.0470733792, -0.0610562712]`
- controller frames: `DP_SCAN_TARGET=104`, `EXECUTOR_ACTIVE=100`,
  `DP_HANDOFF=96`, `INIT_OBS=1`

Visual review sheet:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658/sample_02_hole_late_reverse/live_observed_rollout_annotated_review_sheet.jpg`

The sheet was opened. It matches the metric failure: the peg remains outside
the hole region at the final frame.

## Causal Trace

Early live choices:

```text
iter frame selected             candidates source_suffix post_C_pi
0    104   scale_0.2            213        0             false
1    112   scale_1.5            213        0             false
2    120   scale_1.5            213        0             false
3    128   scale_1.5            213        0             false
4    136   scale_1.5            215        2             false
5    144   scale_1.5            215        2             true
```

At frame `144`, the executor made live `C_pi=true`, but the selected action
was still a checkpoint-model scale candidate even though source-suffix
candidates existed in the bank.

At frame `148`, the controller executed frozen DP for `96` steps. The handoff
failed in the real live rollout:

```text
after DP96 peg_head_pos_at_hole =
[-0.1130290329, 0.0316890031, -0.0545559488]
success = false
```

After that, frames `244,252,260,268,276,284,292` selected checkpoint-model
short chunks only. The live state stayed far from insertion, ending at:

```text
[-0.1129306257, 0.0470733792, -0.0610562712]
```

## Interpretation

This is the same kind of blocker as the sample00 panel replay, with an extra
selector symptom:

- `C_pi=true` is not a reliable handoff label. It can allow DP96 when the
  physical contact/insertion state is not actually DP-continuable.
- Source-suffix coverage exists briefly in this sample, but the scorer chooses
  checkpoint-model scale candidates instead.
- After the failed DP handoff, late states have no useful source-suffix
  candidate coverage, and checkpoint-model chunks do not recover insertion.

This result is not method success. It strengthens the current diagnosis:
the next repair target is real DP-rollout continuability/contact scoring plus
better late live candidate coverage, not relaxing geometric thresholds or
hand-coding recovery cases.
