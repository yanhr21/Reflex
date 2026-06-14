# Video Length Contract Audit

- expected_frames: `301`
- expected_fps: `30.0`
- include_globs: `['sample_*/live_observed_rollout*.mp4', 'sample_*/pure_dp_observed_rollout*.mp4', 'live_observed_rollout*.mp4']`

## Roots

### old_iter2100

- root: `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_panel10_corrected_iter2100_20260613_161006`
- video_count: `2`
- all_videos_match_expected_frames: `False`
- frame_count_range: `119..131`
- duration_seconds_range: `3.966666666666667..4.366666666666666`
- frame_failure_count: `2`
- scan_error_count: `0`

Frame failures:
- `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_panel10_corrected_iter2100_20260613_161006/sample_00_hole_late_move_stop/live_observed_rollout.mp4`: `131` frames, `4.366666666666666` seconds
- `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_panel10_corrected_iter2100_20260613_161006/sample_01_hole_late_constant/live_observed_rollout.mp4`: `119` frames, `3.966666666666667` seconds

### iter2700_val

- root: `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_panel3_dynamic_20260613_alloc127559`
- video_count: `6`
- all_videos_match_expected_frames: `True`
- frame_count_range: `301..301`
- duration_seconds_range: `10.033333333333333..10.033333333333333`
- frame_failure_count: `0`
- scan_error_count: `0`

### iter2700_puredp

- root: `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/dp_full_episode_baseline_iter2700_panel3_dynamic_20260614_alloc127559`
- video_count: `6`
- all_videos_match_expected_frames: `True`
- frame_count_range: `301..301`
- duration_seconds_range: `10.033333333333333..10.033333333333333`
- frame_failure_count: `0`
- scan_error_count: `0`

### iter2700_hard2

- root: `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen2_20260614/cosmos_iter2700_puredp_fail_4_9_10_11_12_13`
- video_count: `12`
- all_videos_match_expected_frames: `True`
- frame_count_range: `301..301`
- duration_seconds_range: `10.033333333333333..10.033333333333333`
- frame_failure_count: `0`
- scan_error_count: `0`

### iter2700_static

- root: `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_static_none_fix1_20260613_alloc127559`
- video_count: `2`
- all_videos_match_expected_frames: `True`
- frame_count_range: `301..301`
- duration_seconds_range: `10.033333333333333..10.033333333333333`
- frame_failure_count: `0`
- scan_error_count: `0`
