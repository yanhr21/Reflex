# 2026-06-22 Panel0245 Offsets64 Sample04 Success

This note records the third completed sample in the strict offsets
`64,48,32,24` panel on allocation `146658`.

## Run

Sample directory:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658/sample_04_hole_late_sine`

The run preserved the strict panel boundary: same-scenario source suffixes,
distance cap `0.02`, offsets `64,48,32,24`, source-suffix execution `8`, no
pure DP prior while live `C_pi=false`, and the full `301/300` contract.

## Result

`sample_panel_result.json` reports:

- `final_observed_frames=301`
- `full_episode_length_ok=true`
- `video_file_contract_ok=true`
- `final_success=true`
- `final_peg_head_pos_at_hole=[-0.0045138597, -0.0029414743, -0.0029432923]`
- controller frames: `DP_SCAN_TARGET=116`, `EXECUTOR_ACTIVE=46`,
  `DP_HANDOFF=138`, `INIT_OBS=1`

Visual review sheet:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658/sample_04_hole_late_sine/live_observed_rollout_annotated_review_sheet.jpg`

The sheet was opened. It shows the robot at the box/hole region through the
handoff and does not contradict the metric success.

## Causal Trace

```text
iter frame selected                            after peg_head_pos_at_hole
0    116   short12_scale_1.5                   [-0.16193,  0.04162, 0.00234]
1    128   retrieval_resid_srcsuffix_r0_s1_o48 [-0.22833,  0.02015, 0.00633]
2    136   retrieval_resid_srcsuffix_r0_s1_o24 [-0.22616,  0.01865, 0.00775]
3    144   retrieval_resid_srcsuffix_r0_s0p5_o24 [-0.18702, 0.00602, 0.00766]
4    152   scale_1.5                           [-0.14009, -0.00691, 0.00469]
5    160   retrieval_resid_srcsuffix_r0_s1_o24 [-0.13226, -0.00533, 0.00349]
6    162   DP handoff 96 steps                 [-0.00445, -0.00297, -0.00297]
7    258   DP handoff 42 steps                 [-0.00451, -0.00294, -0.00294]
```

The key handoff happened after frame `160`, when the selected source-suffix
candidate brought the live state across `C_pi`. Frozen DP then completed
insertion in the real live rollout.

## Interpretation

This is a real positive panel sample, not just replay. It shows that the
corrected offsets/source-suffix interface can produce a DP-continuable live
state for sample04.

It does not erase the current blocker:

- sample00 and sample02 in the same panel failed with full visual/metric
  evidence;
- sample00 and sample02 show that `C_pi=true` is not sufficient by itself;
- sample04 shows that `C_pi` can be useful when the physical contact state is
  actually continuable.

The practical next target is still a learned real-DP-rollout continuability
and contact-quality scorer, plus better late candidate coverage. The current
panel is mixed evidence, not broad method success.
