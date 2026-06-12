# 2026-06-13 Cosmos3 v7_733 Fix1-Recipe Resume And Iter900 Gate

## Boundary

This note records the active Cosmos3 full-episode WAM SFT continuation and
the latest generated-eval gate. It is not controller success evidence.
Live DP/controller integration remains blocked until a checkpoint passes:

- strict same-length generated artifacts,
- generated-RGB readout and failure-profile checks,
- direct visual review of all validation sheets/videos.

The 301 RGB/state frame and 300 action-step contract is preserved.

## Training State

Active SFT root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`

The run uses the frozen user-override v7 DP source with 733 H5 episodes and
the enforced fix1 action recipe:

- `lr=1.0e-4`
- `action_loss_weight=2.0`
- `independent_action_schedule=true`
- `shift_action=1`
- trainable action modules include `action2llm`, `llm2action`, and
  `action_modality_embed`

After the original 4-H200 training allocation hit wall time at iteration 743,
the run resumed on 2026-06-13 CST. A 2-H200 continuation reached and saved
`iter_000000900`. When a held 4-H200 allocation became available, the 2-H200
foreground training command was stopped inside the allocation without
`scancel`; the durable resume point remained `iter_000000900`.

Current continuation:

- Slurm job `127281`
- node `server31`
- `4xH200`
- tmux `cosmos3_v7_733_resume4_600_realloc_0613`
- resumed from `iter_000000900`
- logs confirmed `Resuming ckpt ... iter_000000900`
- four GPUs were observed at 100 percent utilization with about 59-60 GB used

The 4-GPU continuation still runs at roughly 17.3 seconds per iteration. This
appears to be FSDP sharding behavior rather than throughput-linear data
parallel scaling.

## Iter900 Generated-Eval Gate

Eval root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_full_episode_wam_iter_000000900`

The strict artifact check passed:

- `strict_eval_artifacts_ok=true`
- `strict_failures=[]`
- 10 generated samples
- generated/reference video length stayed `301/301`
- action targets stayed `300x32`

Readout/profile also passed structurally, but controller-facing quality did
not pass visual review:

- mean future PSNR: `19.7863478598` dB
- mean action RMSE: `0.5018949204`
- mean robot-action future RMSE: `0.7192880640`
- mean state-sidecar future RMSE: `0.5826733010`
- mean final hole position error: `0.1455857199` m
- mean future hole RMSE: `0.0790919085` m
- mean future peg RMSE: `0.0739520742` m
- mean future TCP RMSE: `0.0702958154` m
- mean future peg-head-hole RMSE: `0.0347010374` m

The agent opened all 10 `review_sheets/*_ref_pred_sheet.png`. The generated
videos are not blank and do not show the old total geometry-collapse failure,
but they are not executable-handoff quality. The visible failure is relative
pose drift among the target hole, peg, and gripper/hand, especially in
post-motion, insert-resume, and peg-recovery cases.

The closed-loop gate file was rewritten after visual review:

`eval_full_episode_wam_iter_000000900/closed_loop_gate_pre_visual.json`

It records:

- `visual_review_status=fail`
- `closed_loop_allowed=false`
- reason `explicit_visual_review_not_passed`

No live DP/controller rollout should start from `iter_000000900`.

## Iter1200 Generated-Eval Gate

Eval root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_full_episode_wam_iter_000001200`

The strict artifact check passed:

- `strict_eval_artifacts_ok=true`
- `strict_failures=[]`
- 10 generated samples
- generated/reference video length stayed `301/301`
- action targets stayed `300x32`

Readout/profile also passed structurally, but direct visual review failed
controller-facing handoff:

- mean future PSNR: `21.4226963555` dB
- mean action RMSE: `0.4558975280`
- mean robot-action future RMSE: `0.7271762588`
- mean state-sidecar future RMSE: `0.4703645349`
- mean final hole position error: `0.0992804290` m
- mean future hole RMSE: `0.0567801252` m
- mean future peg RMSE: `0.0555383943` m
- mean future TCP RMSE: `0.0515553031` m
- mean future peg-head-hole RMSE: `0.0347354275` m

The agent opened all 10 `review_sheets/*_ref_pred_sheet.png`. Several
moving-hole samples are visually stable but not reliable enough for
controller handoff. The clear failures are:

- `hole_late_fast_shift / insert_resume`: final block/hand/peg relative
  geometry is wrong; the predicted peg is not in a DP-resumable insertion
  relation.
- `hole_late_sine / target_pre_motion`: final target and hand/peg relative
  geometry are visibly wrong.
- `none / static_monitor` and `none / static_late_monitor`: the target and
  hand/object relation drift despite the target being static, matching the
  failure-profile false target-motion onsets.

The closed-loop gate file is:

`eval_full_episode_wam_iter_000001200/closed_loop_gate_visual_review.json`

It records:

- `visual_review_status=fail`
- `closed_loop_allowed=false`
- reason `explicit_visual_review_not_passed`

No live DP/controller rollout should start from `iter_000001200`.

## Current Watchers And Spare-GPU Use

The next strict eval/readout/profile/gate watcher is waiting for
`iter_000001500`:

- Slurm job `127288`
- node `server03`
- tmux `cosmos3_v7_733_iter1500_watch_0613`
- target checkpoint `iter_000001500`
- output root
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_full_episode_wam_iter_000001500`
- log
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_iter1500_watch_chain_20260613_0302.log`

A 2-H200 allocation is also held on Slurm job `127286` (`server40`). It must
not run a concurrent SFT writer into the same root while the 4-H200 job
`127281` is alive. It has two safe uses:

- fallback resume after job `127281` disappears before `iter_000001500`;
- read-only validation/eval that fixes `CHECKPOINT_PATH` and writes to an
  independent eval root.

The current read-only spare-GPU task is:

- tmux `cosmos3_v7_733_iter1200_extra30_2gpu_0613`
- Slurm step `127286.4`
- fixed checkpoint `iter_000001200`
- output root
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_full_episode_wam_iter_000001200_extra30_2gpu_20260613_0430`
- `30` validation samples

This task is diagnostic only and does not override the failed 10-sample
iter1200 visual gate.

## Closed-Loop Preflight Code

The guarded closed-loop preflight has a new source-H5 structure check. A local
read-only function smoke on the active `iter_000000900` eval manifest recovered
the first eval sample's source H5:

`fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5/hole_late_move_stop_seed3280649_idx2518.fix3/hole_late_move_stop_seed3280649_idx2518.h5`

The check verified:

- source actions shape `[300,7]`
- peg env-state frames `[301,1,13]`
- box-with-hole env-state frames `[301,1,13]`
- panda wrist camera articulation frames `[301,1,31]`
- TCP, peg, and hole slots present with `301` frames

This is only a structural guard before live smoke. It does not restore
simulator state, execute DP, or prove controller success.
