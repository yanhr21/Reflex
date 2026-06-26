# Dense Full-SFT Closed-Loop Failure Evidence Index

Date: 2026-06-15

## Plain Result

The dense full Cosmos3 SFT did run correctly enough to test the hypothesis.
The current failure is not the training launch, not hidden truncation, and not
the `301` RGB frames / `300` action steps contract.

The failure is the live closed-loop controller:

- `iter900`: `0/4` final real-state successes.
- `iter1200`: `0/4` final real-state successes.
- `iter1500`: `0/4` final real-state successes.

This is the stop point for the direct raw-Cosmos-action controller. It should
not be fixed by threshold tuning or enumerated recovery cases. The next aligned
method direction is low-frequency world/task prediction plus a learned
high-frequency executor or DP-prior residual controller.

## What Worked

- Dense condition root passed the strict full-episode contract:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050`
- Formal dense SFT root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299`
- `iter_000001500` checkpoint exists and came after the formal training floor.
- Generated eval strict gates passed for post-floor checkpoints, including
  `iter1500`.
- Extra generated inspection at `iter1500` passed `72` samples with no strict
  artifact failures:
  `eval_full_episode_wam_iter_000001500_extra120_abs4gpu_20260615_0220/eval_artifact_inspection.json`

## What Failed

The robot often moves near the new hole/box after target motion, but the peg
does not enter the insertion manifold. DP handoff sometimes runs, but it starts
from a bad physical state and does not complete insertion.

This matters because the project objective is task completion in the changed
world, not visual similarity or a near-hole pose.

## Iter1200 Live Evidence

Root:
`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_full300_panel_iter_000001200_clean_dense_20260615_002523`

- `sample_00_hole_late_move_stop`: `final_success=false`,
  final peg-head-in-hole `[-0.1085, -0.0441, -0.0400]`, no DP handoff.
  Review sheet:
  `sample_00_hole_late_move_stop/video_review/live_observed_rollout_annotated_review_sheet.png`
- `sample_01_hole_late_constant`: `final_success=false`,
  final peg-head-in-hole `[-0.0331, -0.0004, 0.0026]`,
  `136` DP handoff steps.
  Review sheet:
  `sample_01_hole_late_constant/video_review/live_observed_rollout_annotated_review_sheet.png`
- `sample_03_hole_late_fast_shift`: `final_success=false`,
  final peg-head-in-hole `[-0.1294, -0.0426, -0.0384]`,
  `8` DP handoff steps.
  Review sheet:
  `sample_03_hole_late_fast_shift/video_review/live_observed_rollout_annotated_review_sheet.png`
- `sample_04_hole_late_sine`: `final_success=false`,
  final peg-head-in-hole `[-0.0724, 0.0008, -0.0021]`,
  `68` DP handoff steps.
  Review sheet:
  `sample_04_hole_late_sine/video_review/live_observed_rollout_annotated_review_sheet.png`

Visual review: all four videos agree with the metrics. The robot reaches near
the moved hole/box, but the peg remains outside or against the box, not
inserted.

## Iter1500 Live Evidence

The original `server35` live render path is not reliable evidence because it
hit Vulkan `DeviceLost`/stall failures. Final `iter1500` live evidence was
collected on render-capable `server24` and `server62` allocations with the
same checkpoint, same generated strict gate, and same live-receding protocol.

- `sample_00_hole_late_move_stop`
  - Root:
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_full300_panel_iter_000001500_clean_dense_server62_20260615_0224/sample_00_hole_late_move_stop`
  - Result: `final_success=false`, final peg-head-in-hole
    `[-0.0829, -0.0107, 0.0027]`, `30` DP handoff steps.
  - Review sheet:
    `video_review/live_observed_rollout_annotated_review_sheet.png`
- `sample_01_hole_late_constant`
  - Root:
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_full300_panel_iter_000001500_clean_dense_server24_sample01_20260615_025555/sample_01_hole_late_constant`
  - Result: `final_success=false`, final peg-head-in-hole
    `[-0.0822, -0.0015, 0.0051]`, `8` DP handoff steps.
  - Review sheet:
    `video_review/live_observed_rollout_annotated_review_sheet.png`
- `sample_03_hole_late_fast_shift`
  - Root:
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_full300_panel_iter_000001500_clean_dense_server62_sample03_20260615_033427/sample_03_hole_late_fast_shift`
  - Result: `final_success=false`, final peg-head-in-hole
    `[-0.1246, -0.0129, -0.0276]`, no DP handoff.
  - Review sheet:
    `video_review/live_observed_rollout_annotated_review_sheet.png`
- `sample_04_hole_late_sine`
  - Root:
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_full300_panel_iter_000001500_clean_dense_server24_sample04_20260615_041209/sample_04_hole_late_sine`
  - Result: `final_success=false`, final peg-head-in-hole
    `[-0.0424, 0.0026, -0.0030]`, `72` DP handoff steps.
  - Review sheet:
    `video_review/live_observed_rollout_annotated_review_sheet.png`

Visual review: the iter1500 videos repeat the iter1200 pattern. Some samples
reach near the moved hole and hand off to DP, but the peg is still outside the
hole. This is not a final-state success.

## Operational Notes

- `server35` should not be used for live ManiSkill render evidence on this
  chain. It produced Vulkan/device-loss or live-stall behavior.
- The old duplicate `server62` panel step was interrupted inside the held
  allocation after independent sample evidence was collected. The allocation
  itself was preserved.
- Held allocations after this evidence pass were `127825` on `server24` and
  `128006` on `server62`; no active foreground experiment was intentionally
  left running.

## Decision Needed

Do not continue direct raw Cosmos action chunks plus threshold handoff as the
main controller.

The aligned next step is to implement the planned low-frequency WM plus
executor path:

1. Cosmos predicts low-frequency task-frame / peg-hole / contact-progress
   targets from live history.
2. A learned executor or DP-prior residual controller outputs high-frequency
   robot actions.
3. Real observations remain the authority; video and final real-state success
   remain required evidence.

