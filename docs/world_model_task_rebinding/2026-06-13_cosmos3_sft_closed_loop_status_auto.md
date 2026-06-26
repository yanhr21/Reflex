# Cosmos3 SFT Closed-Loop Status

- generated_at: `2026-06-14T12:47:38.824923+08:00`
- sft_root: `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`
- active_matching_process_count: `0`
- slurm_extern_only: `False`
- latest_checkpoint: `iter_000002700`
- latest_visible_train_iteration: `2735`
- sft_completed_stale_vs_log: `True`
- latest_val_loss: `0.123388`
- latest_eval_root: `eval_full_episode_wam_iter_000002700`
- latest_eval_closed_loop_allowed: `False`
- latest_eval_visual_review_status: `fail`
- recent_live_any_success: `True`

## Boundary

Corrected live closed-loop evidence is negative for the current checkpoint/condition root. Do not treat validation loss or generated readout metrics as controller success, and do not continue SFT from the current condition root as if more iterations alone are the fix.

## Next Allowed Work

After explicit user approval, run the clean-role/dense-receding condition preflight inside a compute allocation with RUN_SFT=false. Training requires a matching clean_dense_preflight_summary.json with ready_for_overfit=true and another explicit approval.
