# 2026-06-22 Sample 5 Strict Source-Suffix Success

This note records the strict corrected rerun of sample 05. It is a positive
control for the source-suffix + real-state DP handoff interface, not broad
method success.

## Artifact

Run root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_sample5_dist002_exec8_nodpunsafe_20260622_alloc145920`

Sample:

`sample_05_hole_late_continuous_insert`

Review artifacts:

- `sample_05_hole_late_continuous_insert/live_receding_loop_summary.json`
- `sample_05_hole_late_continuous_insert/live_observed_rollout_annotated.mp4`
- `sample_05_hole_late_continuous_insert/live_observed_rollout_annotated_review_sheet.png`
- `sample_05_hole_late_continuous_insert/review_frame_300.png`

The review sheet was generated inside Slurm allocation `145920` from the
annotated video. It contains `301` frames at `30fps`.

## Boundary

The rerun used the corrected safety boundary:

- same-scenario source suffixes only;
- source-suffix start distance at most `0.02`;
- execute only `8` actions before re-observation;
- do not execute pure `dp_prior` while the real-state `C_pi` gate is false;
- preserve the full `301` RGB frame / `300` action contract.

## Result

The target-motion detector triggered at frame `78`. Iterations `0-5` all
selected `retrieval_resid_srcsuffix_r0_s0p5_o32` with 8-step execution.

State progression in peg-head-at-hole coordinates:

- iter 0: `[-0.126668, 0.007642, -0.009377]` to
  `[-0.112521, 0.017865, -0.005095]`, `C_pi=false`;
- iter 1: `[-0.112521, 0.017865, -0.005095]` to
  `[-0.113402, 0.016121, -0.006915]`, `C_pi=false`;
- iter 2: `[-0.113402, 0.016121, -0.006915]` to
  `[-0.113499, 0.012066, -0.003661]`, `C_pi=false`;
- iter 3: `[-0.113499, 0.012066, -0.003661]` to
  `[-0.112900, 0.012354, -0.003005]`, `C_pi=false`;
- iter 4: `[-0.112900, 0.012354, -0.003005]` to
  `[-0.112987, 0.011524, -0.004949]`, `C_pi=false`;
- iter 5: `[-0.112987, 0.011524, -0.004949]` to
  `[-0.112408, 0.009808, -0.003034]`, `C_pi=true`.

Frozen DP handoff then ran from frame `123`. At frame `219`, the state was
`[0.032834, 0.001223, -0.002880]`. Final real simulator state was
`[0.030646, -0.003007, -0.002958]` with `success=true`.

Visual review of the contact sheet and final frame agrees with the metric:
the peg remains held and reaches the box/hole region during DP handoff. There
is no visual peg drop or obvious retreat failure in the inspected frames.

## Interpretation

This proves the corrected source-suffix + real-state DP handoff interface can
complete at least one dynamic sample. It also shows why dense receding matters
in the current implementation: the system needed six short real-observation
cycles before DP became safe to execute.

This does not prove broad method success. Under the same corrected boundary,
sample 00 iter0 replay had `213/213` valid candidates with `0` success, `0`
after-gate states, `0` y/z-improving chunks, and `213` y/z-worsening chunks.
That remains the main blocker: the current action bank does not cover every
early live state.

Next diagnostic: replay all saved candidate banks from this strict sample 05
run. If many unselected candidates cross `C_pi`, the next priority is selector
training. If only the selected chain works, the next priority is stronger
action generation/executor coverage.

## All-Candidate Replay

Replay root:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_sample05_strict_all_candidates_dist002_exec8_nodpunsafe_20260622_alloc145920`

Result:

- `1300` valid candidate replays;
- `0` direct 8-step success;
- `214` after-chunk `C_pi` states;
- `651` y/z-improving candidates;
- `649` y/z-worsening candidates;
- iteration `2` had `213/215` after-gate candidates;
- the six live-selected source-suffix chunks had `0/6` one-step after-gate
  outcomes in this replay.

This changes the diagnosis for sample 05: its saved candidate banks do contain
many chunks that can reach the handoff gate. The inefficiency is selector and
handoff validation, not complete action absence. This is different from
sample 00, where all `213` candidates from the corrected first live state
worsened y/z and none reached the gate.

The current follow-up is DP96 replay on the `214` sample 05 after-gate
candidates:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_sample05_strict_aftergate_dp96_20260622_alloc145920`

## After-Gate DP96 Replay

Filter:

`experiments/world_model_task_rebinding/cosmos3/sample05_strict_after_gate_candidate_filter_20260622.tsv`

Result:

- `214/214` valid filtered candidates;
- `214/214` after-gate states;
- `214/214` y/z-improving chunks;
- `118/214` final DP96 success;
- `208/214` DP-continuable;
- `209/214` final contact-stable.

The successful DP96 labels came from iteration `2`; the single iteration `4`
after-gate candidate did not survive DP96.

Interpretation: sample 05 has real selectable action headroom. The current
selector missed many candidates that could have handed off to DP earlier. This
is a selector/continuability-scoring problem on sample 05, while sample 00
remains an action-generation coverage problem.
