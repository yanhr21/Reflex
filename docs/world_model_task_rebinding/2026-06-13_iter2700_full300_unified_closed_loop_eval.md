# Iter2700 Full-300 Unified-Detector Closed-Loop Eval

## Boundary

This run was launched after the user correction that no sample may be
predeclared as `DP-only` from a static/no-motion label. The controller boundary
is one causal target-motion detector over observed hole poses:

- before the detector fires, frozen DP scans/continues the task;
- after the detector fires, Cosmos3 generates short receding action chunks;
- frozen DP can resume only through the same real-state continuability gate;
- if the detector never fires, DP running the full episode is a consequence of
  the same rule, not a separate static-sample protocol.

The `none`/no-motion label is used only to describe the inspected data row in
this note. It is not a controller-mode input.

The first evaluated sample is dynamic (`hole_late_move_stop`), so the detector
fired and Cosmos3 was active. A later scenario-diverse panel completed three
dynamic samples (`hole_late_constant`, `hole_late_reverse`, and
`hole_late_fast_shift`) under the same boundary. The panel itself is not
positive method evidence: Cosmos closed-loop succeeded on `1/3`, while the
same-source full-episode pure-DP baseline succeeded on `3/3`.

## Run

### Dynamic Sample

- Slurm allocation: `127559` on `server10`.
- Checkpoint:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/checkpoints/iter_000002700`
- Source H5:
  `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5/hole_late_move_stop_seed3280649_idx2518.fix3/hole_late_move_stop_seed3280649_idx2518.h5`
- Output root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_sample00_20260613_alloc127559`

### No-Motion Sample

- Slurm allocation: `127559` on `server10`, later step.
- Checkpoint: same `iter_000002700`.
- Source H5:
  `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5/none_seed700082_idx1657.fix3/none_seed700082_idx1657.h5`
- Output root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_static_none_fix1_20260613_alloc127559`

### Panel Dynamic Sample 01

- Slurm allocation: `127559` on `server10`, same held allocation.
- Checkpoint: same `iter_000002700`.
- Scenario: `hole_late_constant`.
- Source H5:
  `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5/hole_late_constant_seed1050118_idx0000.fix3/hole_late_constant_seed1050118_idx0000.h5`
- Output root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_panel3_dynamic_20260613_alloc127559/sample_01_hole_late_constant`

### Panel Dynamic Sample 02

- Slurm allocation: `127559` on `server10`, same held allocation.
- Checkpoint: same `iter_000002700`.
- Scenario: `hole_late_reverse`.
- Source H5:
  `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5/hole_late_reverse_seed17240243_idx8322.fix3/hole_late_reverse_seed17240243_idx8322.h5`
- Output root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_panel3_dynamic_20260613_alloc127559/sample_02_hole_late_reverse`

### Panel Dynamic Sample 03

- Slurm allocation: `127559` on `server10`, same held allocation.
- Checkpoint: same `iter_000002700`.
- Scenario: `hole_late_fast_shift`.
- Source H5:
  `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5/hole_late_fast_shift_seed5300420_idx3011.fix3/hole_late_fast_shift_seed5300420_idx3011.h5`
- Output root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_panel3_dynamic_20260613_alloc127559/sample_03_hole_late_fast_shift`

### Pure-DP Baseline Comparison

- Output root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/dp_full_episode_baseline_iter2700_panel3_dynamic_20260614_alloc127559`
- Summary:
  `.../pure_dp_panel_summary.json`

This baseline replays the same source target actor motion for the same three
source H5 files, but the controller is frozen static DP for all `300` actions.
The target-motion detector is recorded only for comparison and never switches
controller mode.

## Evidence

### Dynamic Sample

- Summary:
  `.../live_receding_loop_summary.json`
- Raw rollout:
  `.../live_observed_rollout.mp4`
- Annotated rollout:
  `.../live_observed_rollout_annotated.mp4`
- Raw review sheet:
  `.../video_review/live_observed_rollout_review_sheet.png`
- Annotated review sheet:
  `.../video_review/live_observed_rollout_annotated_review_sheet.png`
- Annotated final frame:
  `.../video_review/live_observed_rollout_annotated_frame300.png`
- Video inspection report:
  `.../video_review/video_artifact_inspection.json`

Both raw and annotated videos are readable, nonblank, and have `301` scanned
frames.

### No-Motion Sample

- Summary:
  `.../live_receding_loop_summary.json`
- Raw rollout:
  `.../live_observed_rollout.mp4`
- Annotated rollout:
  `.../live_observed_rollout_annotated.mp4`
- Annotated review sheet:
  `.../video_review/live_observed_rollout_annotated_review_sheet.png`
- Annotated final frame:
  `.../video_review/live_observed_rollout_annotated_frame300.png`
- Video inspection report:
  `.../video_review/video_artifact_inspection.json`

The no-motion run has no `cosmos_live_prefix` or inference directory. Both raw
and annotated videos are readable, nonblank, and have `301` scanned frames.

### Panel Dynamic Sample 01

- Summary:
  `.../sample_01_hole_late_constant/live_receding_loop_summary.json`
- Raw rollout:
  `.../sample_01_hole_late_constant/live_observed_rollout.mp4`
- Annotated rollout:
  `.../sample_01_hole_late_constant/live_observed_rollout_annotated.mp4`
- Raw review sheet:
  `.../sample_01_hole_late_constant/video_review/live_observed_rollout_review_sheet.png`
- Annotated review sheet:
  `.../sample_01_hole_late_constant/video_review/live_observed_rollout_annotated_review_sheet.png`
- Annotated final frame:
  `.../sample_01_hole_late_constant/video_review/live_observed_rollout_annotated_frame300.png`
- Video inspection report:
  `.../sample_01_hole_late_constant/video_review/video_artifact_inspection.json`

Both raw and annotated videos are readable, nonblank, and have `301` scanned
frames.

### Panel Dynamic Sample 02

- Summary:
  `.../sample_02_hole_late_reverse/live_receding_loop_summary.json`
- Raw rollout:
  `.../sample_02_hole_late_reverse/live_observed_rollout.mp4`
- Annotated rollout:
  `.../sample_02_hole_late_reverse/live_observed_rollout_annotated.mp4`
- Raw review sheet:
  `.../sample_02_hole_late_reverse/video_review/live_observed_rollout_review_sheet.png`
- Annotated review sheet:
  `.../sample_02_hole_late_reverse/video_review/live_observed_rollout_annotated_review_sheet.png`
- Annotated final frame:
  `.../sample_02_hole_late_reverse/video_review/live_observed_rollout_annotated_frame300.png`
- Video inspection report:
  `.../sample_02_hole_late_reverse/video_review/video_artifact_inspection.json`

Both raw and annotated videos are readable, nonblank, and have `301` scanned
frames.

### Panel Dynamic Sample 03

- Summary:
  `.../sample_03_hole_late_fast_shift/live_receding_loop_summary.json`
- Raw rollout:
  `.../sample_03_hole_late_fast_shift/live_observed_rollout.mp4`
- Annotated rollout:
  `.../sample_03_hole_late_fast_shift/live_observed_rollout_annotated.mp4`
- Raw review sheet:
  `.../sample_03_hole_late_fast_shift/video_review/live_observed_rollout_review_sheet.png`
- Annotated review sheet:
  `.../sample_03_hole_late_fast_shift/video_review/live_observed_rollout_annotated_review_sheet.png`
- Annotated final frame:
  `.../sample_03_hole_late_fast_shift/video_review/live_observed_rollout_annotated_frame300.png`
- Video inspection report:
  `.../sample_03_hole_late_fast_shift/video_review/video_artifact_inspection.json`

Both raw and annotated videos are readable, nonblank, and have `301` scanned
frames.

### Pure-DP Baseline Videos

For each of `sample_01_hole_late_constant`,
`sample_02_hole_late_reverse`, and `sample_03_hole_late_fast_shift`, the pure
DP baseline output contains:

- `pure_dp_full_episode_summary.json`
- `pure_dp_observed_rollout.mp4`
- `pure_dp_observed_rollout_annotated.mp4`
- `video_review/pure_dp_observed_rollout_annotated_review_sheet.png`
- `video_review/pure_dp_observed_rollout_annotated_frame300.png`
- `video_review/video_artifact_inspection.json`

All three pure-DP videos are readable, nonblank, and have `301` frames.

## Key Metrics

### Dynamic Sample

- `final_prefix_frame_index`: `300`
- `final_observed_frames`: `301`
- `full_episode_length_ok`: `true`
- `completed_iterations`: `25`
- `prefix_selection.detected_frame_index`: `106`
- `prefix_selection.triggered`: `true`
- `prefix_selection.wm_triggered`: `true`
- `controller_frame_counts`:
  - `INIT_OBS`: `1`
  - `DP_SCAN_TARGET`: `106`
  - `WM_ACTIVE`: `186`
  - `DP_HANDOFF`: `8`
- Final `peg_head_pos_at_hole`:
  `[-0.10566872358322144, -0.014312520623207092, -0.05501629412174225]`
- Final `success`: `false`

The last iteration at prefix `f298` executed only `2` action steps, so the
rollout ended exactly at the 300-action/301-frame contract rather than
overshooting or truncating.

### No-Motion Sample

- `final_prefix_frame_index`: `300`
- `final_observed_frames`: `301`
- `full_episode_length_ok`: `true`
- `completed_iterations`: `0`
- `prefix_selection.mode`:
  `target_motion_detector_never_triggered_after_terminal_completion`
- `prefix_selection.triggered`: `false`
- `prefix_selection.wm_triggered`: `false`
- `future_target_motion_scan_after_terminal.would_trigger`: `false`
- `future_target_motion_scan_after_terminal.max_delta`: `0.0`
- `future_target_motion_scan_after_terminal.max_speed`: `0.0`
- `controller_frame_counts`:
  - `INIT_OBS`: `1`
  - `DP_SCAN_TARGET`: `300`
- `wm_active_frame_count`: `0`
- `dp_active_frame_count`: `300`
- Final `success`: `false`

The no-motion sample is therefore not a separate hard-coded static branch or a
label-driven bypass. The same target-motion detector never fired, so the
unified controller never entered WM-active mode. The no-motion DP rollout's
task failure is a frozen-DP performance observation, not evidence that Cosmos
was incorrectly bypassed.

### Panel Dynamic Sample 01

- `final_prefix_frame_index`: `300`
- `final_observed_frames`: `301`
- `full_episode_length_ok`: `true`
- `completed_iterations`: `26`
- `prefix_selection.detected_frame_index`: `94`
- `prefix_selection.triggered`: `true`
- `prefix_selection.wm_triggered`: `true`
- `controller_frame_counts`:
  - `INIT_OBS`: `1`
  - `DP_SCAN_TARGET`: `94`
  - `WM_ACTIVE`: `206`
- Final `peg_head_pos_at_hole`:
  `[-0.1030322015285492, -0.04500763118267059, -0.0782131552696228]`
- Final `success`: `false`

This sample satisfies the full 300-action/301-frame length contract and shows
Cosmos active for most of the dynamic rollout after the observed target-motion
trigger. It is still negative controller evidence because the final peg is not
inserted.

### Panel Dynamic Sample 02

- `final_prefix_frame_index`: `300`
- `final_observed_frames`: `301`
- `full_episode_length_ok`: `true`
- `completed_iterations`: `25`
- `prefix_selection.detected_frame_index`: `104`
- `prefix_selection.triggered`: `true`
- `prefix_selection.wm_triggered`: `true`
- `controller_frame_counts`:
  - `INIT_OBS`: `1`
  - `DP_SCAN_TARGET`: `104`
  - `WM_ACTIVE`: `40`
  - `DP_HANDOFF`: `156`
- Final `peg_head_pos_at_hole`:
  `[0.006168931722640991, 0.001419857144355774, -0.0005607306957244873]`
- Final `success`: `true`

This sample satisfies the full 300-action/301-frame length contract. The first
five post-trigger chunks were Cosmos rebind chunks; the post-Cosmos
continuability gate became true at prefix `f144`, after which the loop switched
to short reobserved frozen-DP handoff chunks. This is positive evidence for
the intended Cosmos-rebind then DP-continuable handoff behavior on this sample.

### Panel Dynamic Sample 03

- `final_prefix_frame_index`: `300`
- `final_observed_frames`: `301`
- `full_episode_length_ok`: `true`
- `completed_iterations`: `21`
- `prefix_selection.detected_frame_index`: `132`
- `prefix_selection.triggered`: `true`
- `prefix_selection.wm_triggered`: `true`
- `controller_frame_counts`:
  - `INIT_OBS`: `1`
  - `DP_SCAN_TARGET`: `132`
  - `WM_ACTIVE`: `32`
  - `DP_HANDOFF`: `136`
- Final `peg_head_pos_at_hole`:
  `[-0.06497782468795776, 0.00016270577907562256, -0.0006470456719398499]`
- Final `success`: `false`

This sample satisfies the full 300-action/301-frame length contract and shows
the intended controller annotations: DP scans until target-motion detection,
Cosmos is active for the first post-trigger chunks, and DP handoff is used
after the real-state gate passes. It is still negative controller evidence
because the final peg is visibly outside the moved hole and simulator success
is `false`.

### Pure-DP Baseline Comparison

- Pure-DP sample 01 (`hole_late_constant`): detector trigger `f94`, final
  success `true`, final peg-head-at-hole
  `[-0.0015992820262908936, -0.0014668405055999756, -0.0004625469446182251]`.
- Pure-DP sample 02 (`hole_late_reverse`): detector trigger `f104`, final
  success `true`, final peg-head-at-hole
  `[0.007763117551803589, -0.003012523055076599, -0.0029997676610946655]`.
- Pure-DP sample 03 (`hole_late_fast_shift`): detector trigger `f132`, final
  success `true`, final peg-head-at-hole
  `[0.006202518939971924, -0.0018486827611923218, 0.002668313682079315]`.

Thus the same-source comparison is:

- Cosmos closed-loop: `1/3` final success.
- Pure frozen DP full episode: `3/3` final success.

This is direct negative comparison evidence for the current `iter2700`
closed-loop interface. It does not show that Cosmos3 improves the controller
on this val panel; it shows that the current Cosmos action/rebind handoff can
degrade samples that the frozen DP already completes under the same target
replay.

### Hard-Case Pure-DP Screen

A harder source-H5 screen was selected by target-motion magnitude/speed from
the v7 frozen source set:

- Manifest:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen_20260614/hard_case_eval_manifest.json`
- Motion stats:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_candidate_motion_stats_20260614.json`

Pure-DP screen results:

- `hole_late_constant` index `0`: success `true`.
- `hole_late_continuous_insert` index `1`: success `false`,
  final peg-head-at-hole
  `[-0.09235715866088867, 0.009750805795192719, -0.040482018142938614]`.
- `hole_late_fast_shift` index `2`: success `true`.
- `hole_late_move_stop` index `3`: success `false`,
  final peg-head-at-hole
  `[-0.07751390337944031, 0.000911906361579895, 0.004029124975204468]`.
- `hole_late_reverse` index `4`: success `true`.
- `hole_late_sine` index `5`: success `true`.

The hard closed-loop Cosmos comparison was therefore run only for the two
pure-DP failures, indices `1` and `3`, under:
`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen_20260614/cosmos_iter2700_hard_dp_fail_1_3`.

Hard Cosmos results:

- `hole_late_continuous_insert` index `1`: full `300/301`, detector trigger
  `f134`, `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:134,
  WM_ACTIVE:166}`, no DP handoff, final success `false`, final
  peg-head-at-hole
  `[-0.09744921326637268, -0.03100036084651947, -0.028185606002807617]`.
  This remains a failure; both pure DP and Cosmos fail this hard sample.
- `hole_late_move_stop` index `3`: full `300/301`, detector trigger `f84`,
  `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:84, WM_ACTIVE:48,
  DP_HANDOFF:168}`, final success `true`, final peg-head-at-hole
  `[-0.009674787521362305, -0.0019456297159194946, -0.002975963056087494]`.
  This is the current positive comparison sample: same-source pure DP failed,
  while Cosmos rebind chunks plus real-state DP handoff succeeded.

The hard panel summary reports `completed_samples=2`,
`final_success_count=1`, and `failed_process_count=0`. The contact sheet was
opened directly:
`.../hard_case_screen_20260614/cosmos_iter2700_hard_dp_fail_1_3/live_receding_panel_contact_sheet.png`.

## Visual Review

### Dynamic Sample

The annotated sheet shows the causal control switch:

- frames before `f106`: `DP_SCAN_TARGET`;
- after detector trigger: `WM_ACTIVE`;
- one short real-state `DP_HANDOFF` segment around `f166`;
- then return to `WM_ACTIVE`.

The raw and annotated sheets agree with the metrics. The annotated final frame
was opened directly and shows `frame 300/300 controller=WM_ACTIVE`; the peg is
beside/outside the block rather than inserted in the hole. This is a corrected
full-length closed-loop failure for the current `iter2700`
checkpoint/interface, not a short-video artifact.

### No-Motion Sample

The annotated sheet and final frame were opened directly. All sampled frames
show `controller=DP_SCAN_TARGET`, `target_motion_detected=False`,
`trigger=none`, and `wm_active=False`. The final frame is `frame 300/300`,
confirming a full-length no-motion video without invoking Cosmos.

### Panel Dynamic Sample 01

The annotated review sheet and final frame were opened directly. Frames before
the detected target motion show `controller=DP_SCAN_TARGET`. From the detected
motion at `f94` through the final frame, the overlay shows
`controller=WM_ACTIVE`, `target_motion_detected=True`, and `trigger=94`.
The final frame is `frame 300/300 controller=WM_ACTIVE`; the peg remains
outside/beside the moved hole. This is another corrected full-length
closed-loop failure for the current `iter2700` checkpoint/interface, not a
short-video artifact or a Cosmos-inactive dynamic rollout.

### Panel Dynamic Sample 02

The annotated review sheet and final frame were opened directly. Frames before
the detected target motion show `controller=DP_SCAN_TARGET`. After the
detected motion at `f104`, the overlay first shows `controller=WM_ACTIVE`.
From roughly frame `145` onward it shows `controller=DP_HANDOFF`, with
`target_motion_detected=True` and `trigger=104`. The final frame is
`frame 300/300 controller=DP_HANDOFF`; the peg is visibly inserted in the moved
hole, matching the final simulator `success=true` metric.

### Panel Dynamic Sample 03

The annotated review sheet and final frame were opened directly. Frames before
`f132` show `controller=DP_SCAN_TARGET`; after `f132`, the video shows a short
`WM_ACTIVE` segment followed by `DP_HANDOFF` through `frame 300/300`. The peg
is still outside the hole in the final frame, matching the final simulator
`success=false` metric.

### Pure-DP Baseline

The annotated pure-DP sheets for all three val-panel samples were opened
directly. They show `controller=PURE_DP`, `wm_active=False`, and
`dp_active=True` through all post-initial frames. The final frames visually
match the success metrics: the peg is inserted for all three samples.

### Hard-Case Comparison

The hard continuous-insert Cosmos annotated sheet and final frame were opened
directly. The video is full length and readable. It shows
`controller=WM_ACTIVE` through the final frame after trigger `f134`; the peg
approaches the moved block but remains outside the hole at `frame 300/300`.

The hard move-stop Cosmos annotated sheet and final frame were opened directly.
The video is full length and readable. It shows `DP_SCAN_TARGET` until trigger
`f84`, a short `WM_ACTIVE` phase, then `DP_HANDOFF` through `frame 300/300`;
the peg is visibly inserted, matching simulator `success=true`.

The same hard move-stop pure-DP baseline video was also opened directly. It
shows `controller=PURE_DP` throughout, with the final peg still outside the
hole and simulator `success=false`. This is the visual basis for calling
move-stop a positive Cosmos-vs-pure-DP comparison sample.

### Hard-Case Screen 2

A second hard screen selected 15 additional high-motion source H5s, again
running full-episode pure DP first:

- Manifest:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen2_20260614/hard_case_eval_manifest.json`
- Pure-DP output:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen2_20260614/pure_dp_hard15`
- Cosmos output for pure-DP failures:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen2_20260614/cosmos_iter2700_puredp_fail_4_9_10_11_12_13`

Pure DP failed on six of the 15 screened samples: indices `4`, `9`, `10`,
`11`, `12`, and `13`. Cosmos closed-loop was run only on those failures.
The completed Cosmos panel reports `completed_samples=6`,
`final_success_count=1`, and `failed_process_count=0`:

- index `4`, `hole_late_fast_shift`: pure DP failed; Cosmos succeeded.
  Cosmos ran full `300/301`, detector trigger `f86`,
  `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:86, WM_ACTIVE:56,
  DP_HANDOFF:158}`, final peg-head-at-hole
  `[-0.0071999430656433105, 0.0015901923179626465, 0.0021036043763160706]`,
  final success `true`.
- index `9`, `hole_late_sine`: pure DP failed; Cosmos failed.
  `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:80, WM_ACTIVE:212,
  DP_HANDOFF:8}`, final success `false`.
- index `10`, `hole_late_sine`: pure DP failed; Cosmos failed.
  `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:96, WM_ACTIVE:204}`,
  final success `false`.
- index `11`, `hole_late_continuous_insert`: pure DP failed; Cosmos failed.
  `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:98, WM_ACTIVE:202}`,
  final success `false`.
- index `12`, `hole_late_move_stop`: pure DP failed; Cosmos failed.
  `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:87, WM_ACTIVE:189,
  DP_HANDOFF:24}`, final success `false`.
- index `13`, `hole_late_fast_shift`: pure DP failed; Cosmos failed.
  `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:72, WM_ACTIVE:220,
  DP_HANDOFF:8}`, final success `false`.

The hard-screen-2 panel contact sheet was opened directly. It visually matches
the metrics: index `4` is inserted at the end, while the other five remain
outside/misaligned. The index `4` Cosmos and pure-DP detailed videos were also
inspected directly. Cosmos shows the required causal sequence
`DP_SCAN_TARGET -> WM_ACTIVE -> DP_HANDOFF` and final insertion; the same-source
pure-DP video stays `controller=PURE_DP` and ends outside the hole. This adds
a second positive comparison sample, but the broader hard-screen-2 success
rate remains only `1/6`, so the current iter2700 closed-loop interface is
partially useful rather than broadly solved.

### Hard-Case Screen 2 Action/Rebind Diagnostic

Read-only action diagnostics were added at
`scripts/world_model/analyze_cosmos3_hard_case_action_rebind.py` and run on the
six hard-screen-2 samples where pure DP failed. Output:

`docs/world_model_task_rebinding/2026-06-14_iter2700_hard_screen2_action_rebind_analysis.json`

The diagnostic compares each executed Cosmos robot-action chunk against the
matching source-teacher action rows from the same H5. This is only failure
localization: teacher actions were not executed, and final authority remains
the real live simulator state plus inspected video.

The matched set has `sample_count=6`, `pure_dp_final_success_count_on_matched=0`,
and `cosmos_final_success_count=1`. The per-sample action/rebind summary is:

- index `4`, `hole_late_fast_shift`, Cosmos success: mean action RMSE
  `0.0718`, median `0.0302`; final peg-head-at-hole
  `[-0.0071999, 0.0015902, 0.0021036]`.
- index `9`, `hole_late_sine`, Cosmos failure: mean action RMSE `0.0957`,
  median `0.0776`; x action sign agreement is negative on average
  (`-0.41`), and C_pi is mostly blocked by `rel_y_abs`/`rel_z_abs`.
- index `10`, `hole_late_sine`, Cosmos failure: mean action RMSE `0.1157`,
  median `0.0962`; y/z sign agreement is near or below zero, and the final
  state remains far outside the hole in y/z.
- index `11`, `hole_late_continuous_insert`, Cosmos failure: mean action RMSE
  `0.0616`, median `0.0556`; y/z C_pi failures dominate despite moderate
  teacher-action RMSE, indicating the live rebind trajectory is not reaching
  the insertion manifold.
- index `12`, `hole_late_move_stop`, Cosmos failure: mean action RMSE
  `0.1004`, median `0.0897`; x/y remain outside the DP-continuable gate.
- index `13`, `hole_late_fast_shift`, Cosmos failure: mean action RMSE
  `0.0756`, median `0.0525`; it also records many `grasped` and `rel_z_abs`
  gate failures, consistent with the video-level failure.

This strengthens the diagnosis: the current checkpoint is not failing because
Cosmos is unused, because the rollout is short, or because the detector is
still hand-picked. It is failing because the generated raw robot-action chunks
do not reliably put the grasped peg into the moved-hole DP-continuable
manifold. The aligned repair remains the clean-role/dense-receding condition
export and approved overfit/full SFT path, or a learned short-chunk executor
conditioned on Cosmos-predicted task state if direct raw action generation
continues to fail. It should not be addressed by relaxing C_pi or switching to
a threshold-only bridge.

## Implementation Notes

- Added direct video-length audit tool:
  `scripts/world_model/audit_video_length_contract.py`.
  Current old-vs-current audit:
  `docs/world_model_task_rebinding/2026-06-14_iter2100_vs_iter2700_video_length_audit.json`
  and `.md`.
  It directly scans the user-flagged old
  `live_receding_panel10_corrected_iter2100_20260613_161006` path and the
  current iter2700 evidence roots. The old iter2100 final rollout videos are
  `131` frames / `4.366666666666666` seconds and `119` frames /
  `3.966666666666667` seconds, so that path is invalid for the current
  full-episode objective. The current iter2700 val, hard-screen-2, pure-DP,
  and static/no-motion roots have `26/26` final raw/annotated videos at
  `301` frames and `10.033333333333333` seconds.
- Future closed-loop and pure-DP summaries now include video-file decode
  evidence at artifact creation time. `run_cosmos3_live_receding_loop.py` and
  `run_dp_full_episode_baseline.py` decode their just-written final raw and
  annotated mp4 files and write `final_observed_video_inspection`,
  `final_observed_annotated_video_inspection`, and `video_file_contract_ok`.
  Their panel wrappers propagate the fields. This makes incomplete future
  videos an explicit implementation failure rather than a silent artifact.
- Added objective-level gate:
  `scripts/world_model/check_cosmos3_closed_loop_objective_gate.py`.
  Current output:
  `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_objective_gate.json`
  and `.md`.
  The gate verifies the current artifacts against the user's implementation
  requirements: full `300` actions / `301` frames, causal target-motion
  detector rather than manual trigger disclosure, explicit controller
  annotation in videos, no-motion DP-only behavior through the same detector,
  moving-target WM activity, same-source pure-DP comparison, and hard-case
  action/rebind evidence. It reports `implementation_contract_ok=true` and
  `method_effectiveness_ok=false`. Contract failures are empty. The gate
  directly scanned the actual mp4 files: `26/26` raw/annotated videos have
  `301` frames, with duration `10.033333333333333` seconds at `30 fps`.
  Remaining method failures are: val Cosmos `1/3` underperforms same-source
  pure DP `3/3`, and hard pure-DP-failure samples are rescued only `1/6`.
- `run_cosmos3_live_receding_loop.py` now records
  `full_episode_length_ok`, controller timeline/counts, and an annotated
  rollout video.
- `infer_prefix_role` no longer maps `scenario == "none"` to
  `static_monitor` or `no_target_motion_static_scenario`; no observed target
  motion is represented as the same pre-trigger condition used in dynamic
  samples before target motion.
- `run_live_dp_until_trigger` no longer treats "DP reached terminal before any
  target-motion trigger" as an unconditional error. It first scans the
  remaining source target poses. If that scan says the detector would still
  never fire, the final observed state can be held to the 300-action/301-frame
  evidence contract and is recorded as
  `target_motion_detector_never_triggered_after_terminal_completion`. If future
  target motion would trigger, the run still fails instead of hiding a dynamic
  data/eval problem.
- `AGENTS.md` and
  `TODO/cosmos3_300f_world_model/08_receding_closed_loop.md` record the
  unified-detector correction.
- The wrapper printed an `unexpected EOF` after the Python summary was already
  written. The on-disk wrapper passes `bash -n`; the likely cause is that the
  shell script was edited while the running shell was still reading it. Treat
  the Python summary/video artifacts as the authoritative run result.
- Added `run_dp_full_episode_baseline.py` and
  `run_dp_full_episode_baseline_panel.py` for same-source full-episode pure-DP
  comparison. The baseline keeps the detector reporting-only and never switches
  controller mode. Both the baseline and live loop set
  `HDF5_USE_FILE_LOCKING=FALSE` at process import time to avoid HDF5 lock
  failures on the cluster filesystem.
