# 2026-06-13 Cosmos3 Demo And Archive Index

This note records the current user-facing demo locations and the cleanup
boundary after archiving unrelated process experiments. No new training,
rendering, rollout, or evaluation was run for this cleanup.

## Kept Active Cosmos3 SFT Root

`/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`

Kept active contents:

- `outputs/`
- `action_target_audit/`
- `full_episode_wam_preflight.json`
- `full_episode_wam_preflight.md`
- `condition_audit.log`
- `sft_manifest.txt`
- `sft_train.log`
- `val_loss_summary.json`
- `eval_full_episode_wam_iter_000002100/`
- `eval_full_episode_wam_iter_000002700/`
- `live_receding_panel10_corrected_iter2100_20260613_161006/`
- `live_receding_promptfix_sample0_longhorizon_iter2100_20260613/`

## SFT Demo To Inspect

Latest full-episode WAM SFT demo, checkpoint `iter_000002700`:

`/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_full_episode_wam_iter_000002700`

Human-inspection files:

- Generated videos: `inference/*/vision.mp4` (10 files)
- Ref/pred sheets: `review_sheets/*_ref_pred_sheet.png` (10 files)
- Summaries: `closed_loop_gate_visual_review.json`, `manual_visual_review.json`, `task_state_readout_v7_733/readout_eval_summary.json`

## Closed-Loop Eval Demos To Inspect

Corrected panel-10 closed-loop failure demo:

`/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_panel10_corrected_iter2100_20260613_161006`

Key files:

- `closed_loop_failure_sample00_01_live_rollout_sheet.png`
- `closed_loop_failure_live_vs_cosmos_predictions_sheet.png`
- `sample_00_hole_late_move_stop/live_observed_rollout.mp4`
- `sample_01_hole_late_constant/live_observed_rollout.mp4`
- `sample_*/live_receding_loop_summary.json`

Long-horizon sample-0 closed-loop failure demo:

`/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_promptfix_sample0_longhorizon_iter2100_20260613`

Key files:

- `live_receding_panel_contact_sheet.png`
- `sample_00_hole_late_move_stop/live_observed_rollout.mp4`
- `sample_00_hole_late_move_stop/live_observed_rollout_dense_sheet.png`
- `sample_00_hole_late_move_stop/live_receding_loop_summary.json`

## Archive Location

Unrelated/superseded process experiments were moved out of the active repo
tree into:

`/public/home/yanhongru/ICLR2027_archive/reflex_cosmos3_process_archive_20260613_202744`

The archive manifest is:

`/public/home/yanhongru/ICLR2027_archive/reflex_cosmos3_process_archive_20260613_202744/MOVED_ITEMS.txt`

Archived categories:

- rejected/superseded top-level SFT attempt roots from the same v7 733 line;
- old intermediate eval directories from the active SFT root, except the kept
  `iter_000002100` and `iter_000002700` evidence directories;
- old closed-loop smoke/preflight/live-receding process directories;
- process logs and temporary SFT manifest snapshots from the active SFT root.

The v7 source data, approved RGB dataset, condition export, overfit evidence,
current checkpoints, current SFT evidence, and current closed-loop evidence
remain in the active Cosmos3 tree.
