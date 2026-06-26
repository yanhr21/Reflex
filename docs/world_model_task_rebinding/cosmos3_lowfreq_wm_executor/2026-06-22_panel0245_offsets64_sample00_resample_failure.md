# panel0245 Offsets64 sample00 Resample Failure

Date: 2026-06-22

## Question

Does the sample00 offsets `64,48,32,24` repair stay reliable when run as part
of the same small panel `0,2,4,5`?

## Run

Panel root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658`

Resource:

- allocation `146658`;
- node `server56`;
- system Vulkan ICD set to `/etc/vulkan/icd.d/nvidia_icd.json`;
- protocol preserved same-scenario source suffixes, distance cap `0.02`,
  source-suffix offsets `64,48,32,24`, 8-step execution, DP96 handoff, and the
  full `301/300` episode contract.

## Result

sample00 finished the full episode but failed:

- `completed_iterations=14`;
- `final_prefix_frame_index=300`;
- `final_observed_frames=301`;
- `full_episode_length_ok=true`;
- `video_file_contract_ok=true`;
- final simulator `success=false`;
- final peg-head-at-hole:
  `[-0.095756, 0.016172, -0.065285]`;
- controller frames:
  `DP_SCAN_TARGET=106`, `EXECUTOR_ACTIVE=98`, `DP_HANDOFF=96`, `INIT_OBS=1`.

Visual evidence:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658/sample_00_hole_late_move_stop/live_observed_rollout_annotated_review_sheet.jpg`

The sheet was opened. It agrees with the metric failure: the final frame shows
the peg outside the moved hole/box insertion pose.

## Causal Trace

The early executor did use source-suffix chunks:

| frame | selected candidate | source-suffix count | source |
| --- | --- | ---: | --- |
| 106 | `retrieval_resid_srcsuffix_r0_s1_o48` | 2 | source suffix |
| 114 | `retrieval_resid_srcsuffix_r1_s1_o48` | 4 | source suffix |
| 122 | `retrieval_resid_srcsuffix_r1_s1_o32` | 4 | source suffix |
| 130 | `retrieval_resid_srcsuffix_r0_s1_o32` | 2 | source suffix |
| 138 | `scale_1.5` | 0 | checkpoint model |
| 146 | `retrieval_resid_srcsuffix_r1_s1_o64` | 4 | source suffix |
| 154 | `retrieval_resid_srcsuffix_r0_s1_o48` | 4 | source suffix |
| 162 | `retrieval_resid_srcsuffix_r0_s1_o48` | 2 | source suffix |
| 264 | `short8_sample_t2_1` | 0 | checkpoint model |
| 272 | `short8_sample_t2_6` | 0 | checkpoint model |
| 280 | `short8_sample_t2_7` | 0 | checkpoint model |
| 288 | `short8_sample_t2_2` | 0 | checkpoint model |
| 296 | `short8_scale_1.5` | 0 | checkpoint model |

Important gate events:

- At frame `162`, the post-executor state crossed C_pi:
  `[-0.092741, 0.002353, -0.003365]`.
- The next iteration at frame `168` ran DP96 because pre-gate was true.
- After DP96, the state was no longer continuable:
  `[-0.097337, 0.006050, -0.049492]`.
- From frame `264` onward, no source-suffix candidates were available, and
  checkpoint-model short chunks did not recover insertion.

## Interpretation

This does not erase the earlier single-run sample00 success. It changes the
claim: offsets `64,48,32,24` are necessary to make useful source-suffix
candidates available, but they are not sufficient for stable closed-loop task
completion.

The current sample00 blocker is not a loose threshold problem. The failure is
live handoff/contact continuability:

1. C_pi can become true once, but DP96 can still drive the real rollout into a
   bad contact/pose state.
2. Once DP handoff fails late, the remaining source-suffix candidate coverage
   can disappear.
3. A single successful run is not broad method evidence; the same protocol
   inside the panel produced a full-contract metric-and-visual failure.

Next useful work is to keep the panel running for samples `2,4,5`, then train
or select against real DP-rollout continuability/contact outcomes. Do not turn
this into a hand-written recovery case or a looser final-state rule.
