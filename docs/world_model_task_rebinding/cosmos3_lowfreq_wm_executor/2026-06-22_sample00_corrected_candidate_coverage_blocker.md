# 2026-06-22 Sample 00 Corrected Candidate Coverage Blocker

## Question

After the sample 5 source-suffix success, the main question was whether the
same mechanism generalizes or whether it only found one lucky suffix.

## Safety Corrections

The first broad panel showed unsafe source-suffix choices. The corrected live
boundary is now:

- source suffixes must match the scenario;
- source-suffix start distance must be at most `0.02`;
- selected candidate chunks execute only `8` steps before re-observation;
- pure `dp_prior` is not executable when live `C_pi` is false.

These changes preserve the real objective: short chunk, real re-observation,
and conservative DP handoff.

## Evidence

Corrected live root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_panel0245_cap_scenario_k2_dist002_exec8_nodpunsafe_20260622_alloc145920`

All-candidate replay root:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_sample00_iter0_all_candidates_dist002_exec8_nodpunsafe_20260622_alloc145920`

Replay result:

- `213` valid candidates;
- `0` direct successes;
- `0` after-gate states;
- `0` y/z-improving chunks;
- `213/213` worsened y/z.

## Plain Meaning

For sample 00 iter0, after the corrected safety boundaries, there was no good
candidate action in the saved bank. This is no longer just a scorer-selection
mistake.

The current blocker is action-candidate/executor coverage at early live
decision states. The next useful repair is to train or generate stronger short
contact/insertion chunks from dense/source/live-state data and then score them
with real DP-continuability/contact labels.

This matches the Dream Diffusion Policy lesson: the world model can guide
short-horizon action choice, but the policy/action generator must already have
usable local actions in the current task frame. A selector cannot recover an
action that is absent from the candidate set.
