# Source-Suffix Live Smoke

Date: 2026-06-22

## Artifact

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_smoke_sample5_20260622_alloc145920`

## Result

The run completed three receding iterations and wrote candidate action banks
and videos. It was a diagnostic smoke, not method evidence:
`full_episode_length_ok=false` because the run intentionally stopped after
`151` observed frames.

Final real simulator state:

- final success: `false`
- final peg-head-in-hole: `[-0.117506, 0.019883, -0.015949]`
- visual review: contact sheet shows the peg still outside the hole

## Scorer Behavior

Source-suffix candidates were present:

- iteration 0: `16` source-suffix candidates, `229` total candidates
- iteration 1: `16` source-suffix candidates, `229` total candidates
- iteration 2: `16` source-suffix candidates, `229` total candidates

The old scorer selected:

- iteration 0: `scale_0.2`
- iteration 1: `dp_prior`
- iteration 2: `dp_prior`

It never selected a source-suffix candidate.

## Interpretation

The source-suffix candidate path is wired into live closed loop, but the old
scorer is not calibrated to select it. This matches the earlier replay result:
source suffixes can create DP96-continuable outcomes in label replay, but the
old scorer was trained on the older candidate family and still ranks DP/model
candidates too high.

Next action: convert live snapshot labels into outcome-scorer training rows
and train a source-suffix-aware scorer using real `dp_rollout_success` and
`dp_rollout_continuable` targets.
