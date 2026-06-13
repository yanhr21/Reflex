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

The completed read-only spare-GPU task was:

- tmux `cosmos3_v7_733_iter1200_extra30_2gpu_0613`
- Slurm step `127286.4`
- fixed checkpoint `iter_000001200`
- output root
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_full_episode_wam_iter_000001200_extra30_2gpu_20260613_0430`
- `30` validation samples

This task is diagnostic only and does not override the failed 10-sample
iter1200 visual gate. It completed with strict structure intact:

- `strict_eval_artifacts_ok=true`
- `strict_failures=[]`
- `30/30` generated-RGB readout samples strict-ok
- mean future PSNR `21.6608184725` dB
- mean action RMSE `0.4395335228`
- mean robot-action future RMSE `0.7056837776`
- mean state-sidecar future RMSE `0.4472291969`
- generated-RGB mean final hole position error `0.0884815329` m
- mean future hole/peg/TCP RMSE `0.0504114347` / `0.0535810572` /
  `0.0511667308` m
- mean future peg-head-hole RMSE `0.0318076271` m

The agent opened representative review sheets from the 30-sample panel. The
panel is not a controller pass. `hole_late_fast_shift` and `hole_late_sine`
still show wrong future target/peg/hand relative geometry; `none` static
samples still show drift in the target/hand relation; some `peg_drop` and
`hole_late_constant` insert-resume sheets are visually cleaner, but they do
not remove the failed handoff evidence.

The same 2-H200 allocation is now waiting for an `iter_000001500` extra-30
read-only chain:

- tmux `cosmos3_v7_733_iter1500_extra30_2gpu_0613`
- Slurm step `127286.22`
- target checkpoint `iter_000001500`
- output root
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_full_episode_wam_iter_000001500_extra30_2gpu_20260613`
- log
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_iter1500_extra30_2gpu_watch_chain_20260613.log`

This watcher waits for the checkpoint, runs 30-sample strict full-episode eval,
and then runs generated-RGB readout/profile. It is a read-only eval path and
must not be confused with a second SFT writer.

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

## Iteration 1500 Gate And Continuation

The 4-H200 run on Slurm job `127281` saved
`iter_000001500` at `2026-06-13 05:48 CST` and completed validation. The
validation loss was `0.121415`, and the logged final average loss was
`0.1146170124411583`. After the checkpoint completed and the original training
step exited, the in-allocation watcher
`cosmos3_v7_733_auto_resume4_after1500_to2100_0613` resumed from
`iter_000001500` toward `iter_000002100` in the same held 4-H200 allocation.

This avoids a second concurrent checkpoint writer. The spare 2-H200 jobs are
used only for read-only eval/watchers with independent eval roots while the
4-H200 writer is alive.

The main generated eval root is:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_full_episode_wam_iter_000001500`

Strict artifact/readout/profile checks passed structurally:

- `strict_eval_artifacts_ok=true`
- `strict_failures=[]`
- `10/10` generated-RGB readout samples strict-ok
- mean future PSNR `21.0053897452` dB
- mean action RMSE `0.4544845864`
- mean robot-action future RMSE `0.7124409493`
- mean state-sidecar future RMSE `0.4863403918`
- generated-RGB mean final hole position error `0.0950135280` m
- mean future hole/peg/TCP RMSE `0.0543056440` / `0.0559011667` /
  `0.0520678853` m
- mean future peg-head-hole RMSE `0.0358502661` m

Manual visual review still failed the controller handoff gate. The agent opened
all 10 review sheets. Sheets `03`, `04`, `08`, and `09` are not DP-resumable:

- `03_insert_resume_hole_late_fast_shift`: late prediction drifts the
  robot/peg to the side of the block instead of preserving the hole-face
  insertion relation.
- `04_target_pre_motion_hole_late_sine`: late robot/peg/target relative
  geometry is visibly wrong.
- `08_static_monitor_none` and `09_static_late_monitor_none`: static target
  cases drift in late/final frames despite no target motion.

The review artifact is:

`eval_full_episode_wam_iter_000001500/manual_visual_review.json`

The closed-loop gate file is:

`eval_full_episode_wam_iter_000001500/closed_loop_gate_visual_review.json`

It records `visual_review_status=fail`,
`closed_loop_allowed=false`, and reason
`explicit_visual_review_not_passed`. Therefore `iter_000001500` is not a
controller/DP integration checkpoint.

## Iteration 1500 Extra-30 Two-GPU Panel

The spare 2-H200 allocation on Slurm job `127286` completed a read-only
extra-30 panel at:

`eval_full_episode_wam_iter_000001500_extra30_2gpu_20260613`

It did not write checkpoints, touch `latest_checkpoint.txt`, or replace the
4-H200 SFT writer. The panel passed structural checks:

- `strict_eval_artifacts_ok=true`
- `strict_failures=[]`
- `30/30` generated-RGB readout samples strict-ok
- mean future PSNR `21.7650756531` dB
- mean action RMSE `0.4336619230`
- mean robot-action future RMSE `0.6891195786`
- mean state-sidecar future RMSE `0.4483521834`
- generated-RGB mean final hole position error `0.0771530019` m
- mean future hole/peg/TCP RMSE `0.0461612608` / `0.0506419007` /
  `0.0485433161` m
- mean future peg-head-hole RMSE `0.0313440083` m

Representative visual review still confirms the same failure family. The
`hole_late_fast_shift`, `hole_late_sine`, and `none` static sheets preserve
object appearance but do not preserve the final robot/peg/hole relative
geometry needed for DP resume. This extra panel is diagnostic only and does
not override the failed main visual gate.

## Iteration 1800 Watchers And Two-GPU Fallback

Two read-only watchers were started for the next checkpoint:

- Main gate: tmux `cosmos3_v7_733_iter1800_watch_0613`, Slurm job `127288`,
  target checkpoint `iter_000001800`, output root
  `eval_full_episode_wam_iter_000001800`, `10` samples.
- Extra panel: tmux `cosmos3_v7_733_iter1800_extra30_2gpu_0613`, Slurm job
  `127286`, target checkpoint `iter_000001800`, output root
  `eval_full_episode_wam_iter_000001800_extra30_2gpu_20260613`, `30` samples.
  This optional panel was later stopped by interrupting the foreground
  tmux/srun command, not by `scancel`, to preserve the allocation for training
  continuation insurance.

The main watcher runs inside a compute-node Slurm step, waits for a stable
checkpoint, runs strict full-episode eval, generated-RGB readout, failure
profiling, and a pre-visual closed-loop gate. It is a read-only path and must
not be confused with a second SFT training writer.

At `2026-06-13T06:31:55+08:00`, job `127286` on `server40` had only its
`extern` step left and was reassigned to tmux
`cosmos3_v7_733_resume2_fallback_to2100_0613`. The fallback target is
`iter_000002100`. It keeps the no-concurrent-writer invariant: it polls the
4-H200 writer job `127281`, and only launches two-GPU SFT into the active root
if that primary writer disappears while `latest_checkpoint.txt` is still below
the target iteration.

After checking the idle-GPU state, the two-GPU fallback was upgraded to an
active shadow continuation instead of keeping the cards idle. The foreground
fallback tmux command was interrupted, preserving allocation `127286`. A new
root was created:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_2gpu_shadow_from1500_to2100_20260613_0637`

Its `iter_000001500` checkpoint is a hardlink copy of the main checkpoint,
with an independent `latest_checkpoint.txt`, so the branch can resume from the
same weights without duplicating the 55G checkpoint or writing into the main
4-H200 root. Tmux `cosmos3_v7_733_shadow2gpu_from1500_to2100_0613` launched
on Slurm job `127286`, step `127286.27`, with `NPROC_PER_NODE=2`,
`DATA_PARALLEL_SHARD_DEGREE=2`, `MAX_ITER=2100`, and the same enforced fix1
action recipe. The log confirms `strict_alignment_ok=true`,
`strict_action_target_ok=true`, `fix1_action_recipe_check=passed`, and
`Loaded checkpoint ... in iteration 1500`. This shadow run is a backup/extra
continuation branch; the 4-H200 root remains the primary gate source unless it
fails or the shadow produces stronger inspected evidence later.

The first read-only `iter1800` watcher allocation `127288` did not survive long
enough to see the checkpoint. Its log ends with Slurm terminating step
`127288.23` at `2026-06-13T06:46:34+08:00` while
`latest_checkpoint.txt` was still `iter_000001500`. No eval artifacts were
produced by that terminated watcher.

To avoid another idle GPU allocation, a replacement login-side request watcher
was started in tmux `cosmos3_v7_733_iter1800_eval_request_on_ckpt_0613` using
`scripts/slurm/watch_cosmos3_checkpoint_then_salloc_eval.sh`. The script only
polls `latest_checkpoint.txt` and the target checkpoint directory on the login
node. After `iter_000001800` exists and remains stable, it requests a fresh
1-H200 `salloc` allocation and then runs the standard strict eval, generated-RGB
readout, readout profile, and pre-visual closed-loop gate inside the compute
allocation. The first poll at `2026-06-13T06:58:54+08:00` reported
`latest=iter_000001500` and `has_target_dir=no`.

Main SFT saved `iter_000001800` at `2026-06-13T07:28:00+08:00`. The target
checkpoint directory and `model/.metadata` existed, but `latest_checkpoint.txt`
still reported `iter_000001500`. Because the eval wrapper takes an explicit
`CHECKPOINT_PATH`, the request watcher was repaired to trigger from a stable
target checkpoint directory and record a latest mismatch instead of blocking on
the stale latest file. Tmux
`cosmos3_v7_733_iter1800_eval_request_on_ckpt_v2_0613` immediately detected
the stable target checkpoint, logged
`latest_checkpoint_mismatch_allowed=iter_000001500`, and requested Slurm job
`127350` for a 1-H200 iter1800 strict eval/readout/gate allocation.

## Closed-Loop Smoke Preparation

The guarded closed-loop entry point was extended while the SFT continued
training. The gate behavior is unchanged: failed generated artifact/readout/
visual gates still stop before any live simulator work.

The new gate-passed `MODE=smoke` branch in
`scripts/world_model/run_cosmos3_receding_closed_loop.py` now has a real short
live-smoke path:

- recover source H5/source UUID from the eval sample;
- restore the real ManiSkill state env to `env_states[chunk_start]`;
- execute only the de-normalized robot-action columns `0..6` from the Cosmos
  predicted action chunk, with action-space violation/clipping recorded;
- optionally execute one short frozen-DP resume horizon from
  `best_eval_success_at_end.pt`;
- record live `base_env.evaluate()` success and `peg_head_pos_at_hole` before
  the chunk, after the Cosmos chunk, and after DP resume;
- write `live_smoke_result.json` and a short RGB video when requested.

This is still a guarded smoke path, not a task-success claim. It runs only
after a checkpoint passes strict artifacts, generated-RGB readout/profile, and
explicit visual review.

A failed-gate compute codecheck was run on held job `127288`, step `127288.24`,
using `iter_000001500` with `VISUAL_REVIEW_STATUS=fail`:

`closed_loop_preflight_iter_000001500_gate_fail_smoke_codecheck_20260613`

The wrapper exited with expected code `40`, recovered source context/source H5,
and reported only:

- `closed_loop_preflight_ok=false`
- failure `closed_loop_gate_blocked`

No `live_smoke_result.json` was written. This confirms the new smoke branch is
not reachable through a failed visual gate.

## Iteration 1800 Gate And Live Smoke

The strict `iter_000001800` eval/readout/profile chain completed in Slurm job
`127350` on `server10` after the watcher repair described above. The eval root
is:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_full_episode_wam_iter_000001800`

Strict artifacts and generated-RGB readout/profile passed structurally:

- `strict_eval_artifacts_ok=true`
- `strict_failures=[]`
- `10/10` generated samples
- generated/reference video length stayed `301/301`
- action targets stayed `300x32`
- mean future PSNR `21.3364889524` dB
- mean action RMSE `0.4717610396`
- mean robot-action future RMSE `0.7097596116`
- mean state-sidecar future RMSE `0.4995639550`
- generated-RGB mean final hole position error `0.1150855990` m
- mean future hole/peg/TCP RMSE `0.0634788298` / `0.0602868412` /
  `0.0568269437` m
- mean future peg-head-hole RMSE `0.0340992235` m

The agent opened all `10` review sheets. Manual review is recorded in:

`eval_full_episode_wam_iter_000001800/manual_visual_review.json`

The visual verdict was `pass_with_caution`: `8` pass, `2`
pass-with-caution, and `0` fail. The closed-loop gate file:

`eval_full_episode_wam_iter_000001800/closed_loop_gate_visual_review.json`

records `closed_loop_allowed=true`, but the boundary is only permission to run
short live smoke. It is not controller success evidence.

Two live smoke diagnostics were run from `iter1800`:

1. Sample `0`:
   `closed_loop_smoke_iter_000001800_20260613_0750`.
   The smoke executed `8` Cosmos robot-action steps plus `8` frozen-DP resume
   steps. It moved the peg-head closer to the hole but ended with
   `final_eval.success=false`.
2. Sample `3`:
   `closed_loop_smoke_iter_000001800_sample3_dp32_recompute_20260613_0805`.
   Before this run, a DP-resume implementation bug was fixed: longer resume
   requests now repeatedly recompute the DP action chunk from the latest live
   observation instead of silently executing only one `act_horizon=8` block.
   The corrected smoke executed `8` Cosmos steps plus `32` recomputed
   frozen-DP resume steps and still ended with `final_eval.success=false`.

For sample `3`, the live metrics were:

- before chunk: `peg_head_pos_at_hole=[-0.21998, 0.06211, -0.02498]`,
  `success=false`
- after Cosmos chunk: `[-0.16315, 0.03227, -0.01348]`, `success=false`
- after 32 DP resume steps: `[-0.13554, 0.00051, 0.00050]`,
  `success=false`

The video contact sheet was opened:

`closed_loop_smoke_iter_000001800_sample3_dp32_recompute_20260613_0805/live_smoke_short_chunk_contact_sheet.png`

It visually matches the metrics: the robot/peg approach the target, but there
is no completed insertion. Therefore `iter1800` proves the gated live smoke
code path can run and that the current checkpoint produces partially useful
approach motion; it does not prove dynamic task completion or DP handoff
success.

### Iter1800 10-Sample Live-Smoke Panels

After the single-sample smoke, the held 1-H200 eval allocation `127350` was
used for controller diagnostics instead of sitting idle. These diagnostics do
not write checkpoints and do not affect the ongoing main or shadow SFT jobs.

The first panel used one `8`-step Cosmos robot-action chunk followed by `32`
frozen-DP resume steps, where the DP action chunk is recomputed from the latest
live observation after each `act_horizon=8` segment:

`closed_loop_smoke_iter_000001800_panel10_dp32_recompute_20260613_0810`

Artifacts:

- `panel_summary.json`
- `panel_visual_review.json`
- `panel_contact_sheet_start_cosmos_final.png`

Result: `1/10` final simulator success. The single success was sample `6`, a
`target_post_motion` continuous-insert case whose restored start state was
already nearly at the hole (`before_norm=0.0034`). The inspected contact sheet
shows the remaining samples approach or touch the block but do not complete
insertion. This is negative diagnostic evidence for short DP handoff.

The second panel used the same `8`-step Cosmos chunk but extended the
recomputed frozen-DP resume horizon to `96` steps:

`closed_loop_smoke_iter_000001800_representative_dp96_recompute_20260613_0816`

Artifacts:

- `panel_summary_full10_dp96.json`
- `panel_contact_sheet_full10_dp96_start_cosmos_mid_final.png`

Result: `6/10` final simulator success. Successes were samples
`0, 1, 3, 4, 6, 8`; failures were `2, 5, 7, 9`.

Per-sample final result:

- `0` `hole_late_move_stop / target_pre_motion`: success, final norm `0.0076`
- `1` `hole_late_constant / target_motion_observed`: success, final norm
  `0.0252`
- `2` `hole_late_reverse / target_post_motion`: failure, final norm `0.1201`
- `3` `hole_late_fast_shift / insert_resume`: success, final norm `0.0104`
- `4` `hole_late_sine / target_pre_motion`: success, final norm `0.0075`
- `5` `hole_late_continuous_insert / target_motion_observed`: failure, final
  norm `0.1185`
- `6` `hole_late_continuous_insert / target_post_motion`: success, final norm
  `0.0064`, but the start state was already very close to the hole
- `7` `peg_drop / peg_recovery`: failure, final norm `0.1153`
- `8` `none / static_monitor`: success, final norm `0.0051`
- `9` `none / static_late_monitor`: failure, final norm `0.1321`

The inspected DP96 contact sheet is visually consistent with the live metrics:
the success rows show plausible final insertion positions, while the failed
rows remain visibly non-inserted or misaligned. The interpretation is narrow:
the current one-shot Cosmos chunk can put several samples into a state where a
long DP resume eventually succeeds, but it is not the intended receding Cosmos
world-model controller. It does not re-run Cosmos after each live observation,
does not solve peg-drop recovery, and is not sufficient method evidence.

## Current Resource State After Two-GPU Correction

At the latest check on 2026-06-13 CST:

- Primary training job `127281` on `server31` is still the only writer to the
  main SFT root. Slurm step `127281.38` requests `gres/gpu=4`. The log reached
  rank-0 iteration `1893`, and `nvidia-smi` inside the allocation showed all
  four GPUs at `100%` utilization.
- Independent shadow training job `127286` on `server40` writes only the
  separate 2-GPU shadow root. Slurm step `127286.27` requests `gres/gpu=2`.
  The log reached rank-0 iteration `1774`, and `nvidia-smi` showed both GPUs
  at `100%` utilization.
- Eval/smoke job `127350` on `server10` was used for the `iter1800` live
  smoke diagnostics. It is not an SFT writer.

The next aligned checkpoint gate is `iter_000002100` from the primary main
root or, if needed, from the independent 2-GPU shadow root. Any controller
claim still requires strict artifacts, generated-RGB readout/profile, direct
visual review, live simulator metrics, and inspected video evidence.

## Shadow Iter1800 Branch Diagnostic

While the primary 4-H200 SFT continued toward `iter_000002100`, the held
1-H200 eval allocation `127350` was used to evaluate the independent 2-GPU
shadow branch at `iter_000001800`. This branch writes only:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_2gpu_shadow_from1500_to2100_20260613_0637`

The eval root is:

`sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_2gpu_shadow_from1500_to2100_20260613_0637/eval_full_episode_wam_iter_000001800`

Strict artifacts and generated-RGB readout/profile passed structurally:

- `strict_eval_artifacts_ok=true`
- `strict_failures=[]`
- `strict_readout_eval_ok=true`
- mean future PSNR `20.8650219770` dB
- mean action RMSE `0.4920883197`
- mean robot-action future RMSE `0.7290105271`
- mean state-sidecar future RMSE `0.5402163232`
- generated-RGB mean final hole position error `0.1166951584` m
- mean future hole/peg/TCP RMSE `0.0651412586` / `0.0658651109` /
  `0.0603592611` m
- mean future peg-head-hole RMSE `0.0339960731` m

The agent inspected the review-sheet overview and opened the key failure
sheets directly. Manual visual review is recorded at:

`eval_full_episode_wam_iter_000001800/manual_visual_review.json`

The branch visual gate is failed. `hole_late_fast_shift` insert-resume,
`peg_drop` recovery, and static-monitor samples show non-resumable
peg/hand/target relative geometry. The gate file:

`eval_full_episode_wam_iter_000001800/closed_loop_gate_visual_review.json`

records `visual_review_status=fail` and `closed_loop_allowed=false`. The
shadow branch therefore does not replace the primary main-root gate.

## Iter2100 Eval Watcher Adjustment

The original `iter2100` request watcher would have requested a fresh 1-H200
`salloc` after the checkpoint appeared. After `127350` became idle again, that
tmux watcher was stopped by sending `Ctrl-C` to its pane; no Slurm allocation
was cancelled.

A new reusable script was added:

`scripts/slurm/watch_cosmos3_checkpoint_then_existing_alloc_eval.sh`

It only polls checkpoint files from the login node. After the target checkpoint
directory and `model/.metadata` are stable, it runs the standard strict
eval/readout/profile/gate chain through `srun --jobid=$EXISTING_JOB_ID` inside
an already held allocation. The active tmux watcher is:

`cosmos3_v7_733_iter2100_eval_existing127350_0613`

It targets the primary main-root `iter_000002100` checkpoint and existing job
`127350`.

## Iter2100 Gate, Live Smoke, And Continuation

The primary 4-H200 root reached `iter_000002100` at
`2026-06-13T09:00:05+08:00` and completed validation with validation loss
`0.112557`. The checkpoint path is:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/checkpoints/iter_000002100`

The held 1-H200 eval allocation `127350` ran strict eval/readout/profile at:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_full_episode_wam_iter_000002100`

Strict structure passed:

- `strict_eval_artifacts_ok=true`
- `strict_failures=[]`
- generated/reference videos stayed `301/301`
- action targets stayed `300x32`
- mean future PSNR `22.7164398514` dB
- mean action RMSE `0.4048669528`
- mean robot-action future RMSE `0.6779193138`
- mean state-sidecar future RMSE `0.4098501999`
- generated-RGB mean final hole position error `0.0687830707` m
- mean future hole/peg/TCP RMSE `0.0420433315` / `0.0433298166` /
  `0.0416048895` m
- mean future peg-head-hole RMSE `0.0321321102` m

The agent opened all ten review sheets directly. Manual review is recorded at:

`eval_full_episode_wam_iter_000002100/manual_visual_review.json`

Verdict: `pass_with_caution`, with `8` pass, `2` pass-with-caution, and `0`
fail. The caution cases are `05_target_motion_observed_hole_late_continuous_insert`
and `08_static_monitor_none`, both for mild live-handoff geometry offsets
rather than visible wall insertion, peg loss, target false-motion collapse, or
global video failure. The formal gate file:

`eval_full_episode_wam_iter_000002100/closed_loop_gate_visual_review.json`

records `closed_loop_allowed=true`. This only permits diagnostic live smoke; it
is not controller success evidence.

The comparable DP96 diagnostic live-smoke panel ran in allocation `127350`:

`closed_loop_smoke_iter_000002100_representative_dp96_recompute_20260613_0928`

It executed one `8`-step Cosmos robot-action chunk followed by `96` frozen-DP
resume steps, recomputing the DP action chunk every `8` live steps. The panel
completed all ten samples:

- successes: samples `0`, `1`, `3`, `6`, `8`
- failures: samples `2`, `4`, `5`, `7`, `9`
- final simulator success: `5/10`

The opened contact sheet:

`closed_loop_smoke_iter_000002100_representative_dp96_recompute_20260613_0928/panel_contact_sheet_full10_dp96_start_cosmos_mid_final.png`

matches the live metrics. Success rows show plausible insertion; failed rows
remain visibly uninserted or misaligned. Therefore `iter2100` has better
generated-video/readout metrics than earlier checkpoints, but the current
one-shot-Cosmos-plus-long-DP diagnostic still does not prove the required
receding world-model controller.

After the `iter2100` checkpoint finished, the primary 4-H200 allocation
`127281` was kept and the same root was resumed from `iter_000002100` toward
`iter_000002700`. The first retry requested `64` CPUs and Slurm rejected it
because job `127281` has `32` CPUs; it was immediately retried with
`--cpus-per-task=32`, and step `127281.40` resumed training. The prior log was
snapshotted before the wrapper overwrote `sft_train.log`.

The two-H200 shadow allocation was also kept active. Tmux
`cosmos3_v7_733_shadow2gpu_auto_from2100_to2700_0613` now waits for current
shadow step `127286.27` to finish and for shadow `iter_000002100` to be
complete, then resumes that independent shadow root from `2100 -> 2700` inside
the same held allocation. This preserves the no-concurrent-writer rule while
avoiding idle two-GPU resources.

After the correction that remaining two GPUs must also be used for continued
SFT, both branches were confirmed active:

- primary 4-H200 main-root step `127281.40` on `server31`, observed training
  around rank-0 iteration `2199`
- independent 2-H200 shadow-root step `127286.33` on `server40`, observed
  training around rank-0 iteration `2108`

Two additional no-concurrent-writer continuation watchers were armed:

- main tmux `cosmos3_v7_733_main4gpu_auto_from2700_to3300_0613`, log
  `sft_train_4gpu_auto_resume_after2700_to3300_20260613_0951.watch.log`
- shadow tmux `cosmos3_v7_733_shadow2gpu_auto_from2700_to3300_0613`, log
  `sft_train_2gpu_auto_resume_after2700_to3300_20260613_0953.watch.log`

Each watcher only launches after `iter_000002700` exists, is the latest
checkpoint, and the corresponding allocation has no active non-extern Slurm
step. This is a resource-protection continuation and does not change the
training recipe, checkpoint roots, frame/action contract, or evidence gate.

The next primary checkpoint gate is `iter_000002400`. Tmux
`cosmos3_v7_733_iter2400_eval_existing127350_0613` polls checkpoint files on
the login node and will use held allocation `127350` for strict
eval/readout/profile once `iter_000002400` is stable.
