# 2026-06-22 Source-Suffix Best-Gate Live Sample 5

## Result

This is the first source-suffix-aware live closed-loop success.

Artifact:
`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_sample5_20260622_alloc145920`

Sample:
`sample_05_hole_late_continuous_insert`

Key metrics from `live_receding_panel_summary.json` and
`live_receding_loop_summary.json`:

- `completed_samples=1`
- `final_success_count=1`
- `panel_full_episode_contract_ok=true`
- `failed_process_count=0`
- `final_success=true`
- `full_episode_length_ok=true`
- `video_file_contract_ok=true`
- `final_observed_frames=301`
- `final_prefix_frame_index=300`
- decoded video frame count `301` at `30 fps`
- final peg-head-in-hole state:
  `[0.0365917, -0.0003869, -0.0028896]`

Controller frame counts:

- `INIT_OBS=1`
- `DP_SCAN_TARGET=78`
- `EXECUTOR_ACTIVE=5`
- `DP_HANDOFF=217`

## Causal Interpretation

The old source-suffix smoke had the source-suffix candidates wired in, but the
old scorer did not select them and the rollout failed.

This run used the source-suffix-aware outcome scorer:
`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_live_snapshot_source_suffix_weighted_full_20260622_alloc145920_formal3h/checkpoint_best_gate.pt`

That scorer met the formal training floor and was marked as the live-eval
checkpoint in its own summary.

At the first live decision, the selector chose:
`retrieval_resid_srcsuffix_r1_s1_o32`

This was a source insertion suffix candidate. It was executed for a short live
chunk, then the real simulator state was re-observed and DP handoff continued.
By iteration 2, the DP handoff reached simulator success; iteration 3 preserved
success through the final frame.

Plain meaning: the useful unit is not "the candidate chunk inserts by itself."
The useful unit is "a short contact/insertion chunk moves the real live state
onto a DP-finishable manifold, then DP finishes after re-observation." This
matches the DDP/HDP lesson being tested.

## Visual Review

Reviewed contact sheet:
`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_sample5_20260622_alloc145920/live_receding_panel_contact_sheet.png`

The sheet shows the peg outside the moved hole initially, then aligned and
inserted by the final frame. This agrees with the final simulator success
metric.

The generated panel summary still has
`visual_review_status=needs_direct_agent_or_user_review`; this note is the
direct agent visual review record.

## What This Proves

It proves that the source-suffix-aware scorer can convert at least one live
dynamic sample under the full `301/300` contract.

It also proves that the previous failure was not only "source suffix candidates
are not wired." They were wired before. The missing piece was selector
calibration against real `candidate + DP96` outcomes.

## What It Does Not Prove

It is not broad method success. It is one sample.

The next active run is a small generality panel on samples `0,2,4,5` with the
same source-suffix-aware best-gate scorer, source suffix bank, DP96 handoff,
and full-episode contract:
`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_panel0245_20260622_alloc145920`
