# 2026-06-09 Cosmos3 300-Step Workspace Reset

## User Correction

The user clarified that the full1000 data does not need regeneration, but the
128-action / 129-frame chunked construction is rejected. The intended "300
frames" boundary means the total episode horizon, not an additional 300-frame
future prediction target. The active data should be treated as a 300-step
episode, with 301 RGB/state frames when frame 0 is included.

## Actions Taken

- Stopped Slurm job `122177`, the rejected chunked SFT run.
- Killed the waiting chunked action-eval and readout tmux watcher sessions.
- Moved old experiment results, old evidence notes, and old logs outside the
  active `/public/home/yanhongru/ICLR2027` tree.
- Moved scripts whose names/default paths explicitly targeted the rejected
  chunked/128-action/129-frame chain into the same external archive.
- Moved old Cosmos SFT/eval/watch/controller wrappers and legacy
  object-state/RGB-D-slot/controller method scripts that could restart rejected
  branches. Basic Cosmos data/render/manifest/preflight/readout utilities were
  left for later review and full-episode adaptation.
- Moved remaining legacy `rgbd`/`full96` script entry points out of the active
  script tree, since the reviewed next path starts from the approved RGB
  full1000 dataset.
- Preserved local Cosmos3 checkpoints, base DP checkpoints, official replay
  data, approved full1000 RGB videos, and full1000 source H5/specs.
- Moved full1000 source H5/specs from the old `experiments/_archive/...` path
  into `data/cosmos3/full1000_rgbd_env_states_20260603_1938`.
- Updated the approved full1000 dataset manifests/source list to point at the
  active `data/cosmos3/...` source path.
- Moved old PLAN/TODO active directories into `_backup_*` directories.
- Created new active plan/TODO directories:
  `PLAN/cosmos3_300f_world_model/` and
  `TODO/cosmos3_300f_world_model/`.

## Archive Root

Archived files were moved to:

```text
/public/home/yanhongru/ICLR2027_archive/reflex_20260609_cosmos3_300f_reset/
```

The archive is historical only. It should not be used as an active training,
evaluation, or controller input root.

## Stop Point

No new training, generation, readout, controller, or DP integration was started
after this reset. The next step is user review of the new plan/TODO.

## Execution Approval

After reviewing the plan, the user approved execution. The next active work is
the full-episode action/proprio/state condition export and strict preflight.
Training must run on GPU/Slurm, count as evidence only at 1 GPU / 1 hour or
more, continue until validation no longer clearly improves, and produce
validation videos/contact sheets for agent inspection plus user review backup.
The user further clarified that probe/export/preflight/debug tasks must not run
on the login node either. The login node is only for file edits, downloads,
status reads, and starting tmux/salloc sessions. GPU resources should be held
through tmux-managed `salloc` first, with long enough allocations preferred
when available, so export/debug/train/validation can run inside the allocation
without one-shot `sbatch` queue churn.

## Execution Scaffold

After approval, the active scaffold was written but not executed on the login
node:

- Full-episode condition exporter:
  `scripts/world_model/export_cosmos3_maniskill_full_episode_wam_conditions.py`
- Strict full-episode WAM preflight:
  `scripts/world_model/preflight_cosmos3_full_episode_wam_contract.py`
- Allocation-only export/preflight/SFT wrapper:
  `scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh`

The wrapper refuses to run unless `SLURM_JOB_ID` is present, so export,
OpenCV/video checks, action-target audits, and SFT start only inside a held
Slurm allocation. The exported row contract is still the complete 301-frame
RGB/state and 300-step action episode. Multiple rows may use different causal
prefix masks, but they reference the same full episode and do not create a
128/129-frame sliced dataset.

## Allocation Execution Notes

- A tmux-held 1-H200 allocation was acquired as Slurm job `122736` on
  `server44`.
- The first export attempt entered the compute step but stalled after the
  first record because the exporter reopened each H5 repeatedly for every
  prefix. That partial root was archived under
  `/public/home/yanhongru/ICLR2027_archive/reflex_20260609_cosmos3_300f_reset/failed_partial_active_runs/`.
- The exporter was repaired to load each episode H5 once and derive all prefix
  rows from memory arrays. The second attempt planned all 1000 source episodes
  and 4192 full-episode prefix rows, but JSON action sidecar serialization was
  too slow for the allocation workflow. That partial root was also archived.
- The active repair keeps the same 300-step action target but stores each
  action sidecar as a rank-2 float32 `.npy` file. The local Cosmos SFT
  dataloader and preflight/action-target audit tools now support `.npy` action
  sidecars while preserving JSON compatibility.
- To avoid repeatedly writing identical supervision, state targets and
  full-episode task summaries are shared once per source episode; each prefix
  row keeps its own action sidecar and prefix label that points to the shared
  summary. This is a storage/layout optimization only. The row contract remains
  full 301 RGB/state frames and 300 action steps.
- After sanitizing causal captions to avoid the phrase "final target pose", the
  strict preflight passed for
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_full1000_rgb_300step_20260609_203902`.
  It contains `4192` full-episode prefix rows from `1000` source episodes:
  train `3808` rows / `912` source episodes and val `384` rows / `88` source
  episodes. Prefix-role coverage includes static monitor, target pre-motion,
  target-motion observed, target post-motion, peg recovery, and insert resume.
- The action-target audit under
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718/action_target_audit/`
  passed with every action sidecar shaped `300x32`, nonzero robot action
  values, robot-action temporal variation, and time-fraction variation.
- The active SFT run is
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718`.
  It started on Slurm job `122736` using 1 H200. Initial validation completed
  with validation loss `15.253426` at iteration `0`, and training reached at
  least iteration `12` with per-step losses being logged. This is SFT startup
  and early training evidence only. It is not yet post-SFT world-model evidence
  and not controller evidence; generated validation videos/readouts remain
  required before any DP/controller integration.

## 2026-06-09 Continued Training And Eval Prep

- The active SFT run remained on Slurm job `122736` in the tmux-held 1-H200
  allocation. Checkpoint loading was sanity-checked from the training log:
  `Cosmos3-Nano-Policy-DROID-DCP/model` loaded successfully at iteration `0`.
  The skipped `net_ema.*` keys are expected because this run has
  `model.config.ema.enabled=false`.
- The training log confirms the active condition root, not an old truncated
  root, was used by the dataloader: train `3808` records/windows/videos and
  val `384` records/windows/videos. The saved config records
  `num_video_frames=301`, `state_t=300`, `vision_gen=true`,
  `action_gen=true`, and `action_loss_weight=10.0`.
- By `2026-06-09 21:45:55`, the run had reached about iteration `177`. Loss
  had moved from early `10-14` values into mostly `4.7-6.5`, with occasional
  finite random-timestep spikes. This explains why an early loss around
  `9-10` is not by itself a load or data failure. No NaN/crash pattern was
  observed in the inspected log tail.
- The run had not yet reached the `iter 300` validation/save point, and actual
  SFT training time had not yet passed the required 1 GPU-hour floor. Therefore
  the current state is still in-progress training evidence only.
- Post-SFT validation generation was prepared but not run before a checkpoint:
  `scripts/world_model/build_cosmos3_full_episode_wam_eval_inputs.py`,
  `scripts/world_model/inspect_cosmos3_full_episode_wam_eval_artifacts.py`,
  and
  `scripts/slurm/run_cosmos3_300f_full_episode_wam_eval_in_allocation.sh`.
  These scripts were syntax-checked inside Slurm job `122736` via `srun`, not
  on the login node.
- The prepared eval input manifest is under
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718/eval_full_episode_wam_latest/`.
  It selected 10 strict validation inputs with all rows under the 301 RGB
  frame / 300 action step contract. The selected validation set covers target
  pre-motion, target motion observed, target post-motion, insert resume, peg
  recovery, static monitor, and static-late monitor roles across
  `hole_move_stop`, `hole_reverse`, `hole_constant`, `peg_drop`,
  `peg_disturb`, and `none` scenarios.
- This eval manifest is not generated model evidence. It becomes evidence only
  after a saved SFT checkpoint is used to generate full-length validation
  videos/actions, those artifacts pass strict 301/300 inspection, readout
  metrics are produced, and the generated contact sheets/videos are visually
  reviewed.

## 2026-06-09 Loss/Load Sanity Update

- In response to the observed early training loss around `9-10`, the active
  log was rechecked inside Slurm job `122736`. The run is not stuck at that
  scale: by `2026-06-09 21:56:34` it had reached iteration `215`, with recent
  losses mostly in the `3.7-5.7` range. Grad norms were finite. No NaN, OOM,
  crash, action shape mismatch, checkpoint key shape mismatch, or missing
  model-load failure was found in the inspected log.
- The load path remains the intended
  `checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID-DCP/model`; the log reports
  successful checkpoint load at iteration `0`. The only skipped key pattern is
  `net_ema.*`, expected because EMA is disabled for this SFT.
- The saved SFT config still points train/val to the active
  `full_episode_wam_conditions_full1000_rgb_300step_20260609_203902`
  JSONLs with `num_video_frames=301`, `state_t=300`, `vision_gen=true`,
  `action_gen=true`, `max_action_dim=64`, and `action_loss_weight=10.0`.
  The `.npy` action sidecar loader in the Cosmos SFT dataset code was checked:
  it loads rank-2 float32 sidecars, requires action length equal to
  `video_frames - 1`, pads to `max_action_dim`, and passes
  `condition_frame_indexes_action` to the sequence plan.
- Causal conditioning was spot-checked on actual JSONL rows. Each row keeps a
  full `300x32` action/state sidecar as the prediction target, but
  `condition_frame_indexes_action` exposes only history actions before the
  prefix frame. The future action rows are therefore supervised targets, not
  privileged future conditions. The visual prefix is exposed through
  `condition_frame_indexes_vision` over the compressed latent prefix.
- Role grounding was also spot-checked. The top-level JSONL row does not use a
  top-level `caption`, but the dataloader reads the `caption` field inside the
  `t2w_windows` entry. A sampled validation window caption explicitly names
  `TARGET_OBJECT=hole`, `TOOL_OBJECT=peg`, and `ACTOR=robot_gripper_tcp`, and
  includes prefix role, current hole/peg/TCP positions, peg-head relative
  geometry, target velocity, `target_motion_observed`, grasp, and insertion
  state. This is paired with the fixed structured action/state vector layout.
- Generated-video task-state readout tooling was prepared but not launched:
  `scripts/world_model/run_cosmos3_full_episode_readout_eval.py` and
  `scripts/slurm/run_cosmos3_300f_task_state_readout_in_allocation.sh`.
  `inspect_cosmos3_task_state_prediction.py` now accepts the existing
  full-episode `state_targets/*.json` as reference labels for readout metrics.
  This preserves the method boundary: readout metrics are diagnostics on top
  of Cosmos3-generated RGB, not a replacement world model and not controller
  success evidence.

## 2026-06-09 Iteration-300 Full-Episode Diagnostic

The first full-episode checkpoint was evaluated because the loss/load sanity
question alone cannot establish whether the WAM has learned the target-motion
and robot/peg rollout behavior required by the method.

- Checkpoint `iter_000000300` was saved under
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718/outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/checkpoints/iter_000000300`.
- Validation loss at iteration `300` was `6.964001`, down from initial
  validation loss `15.253426`. Around the checkpoint, train losses were mostly
  in the `3-5` range with occasional finite spikes. This supports
  checkpoint-load, data-contract, and numeric sanity. It is not success
  evidence by itself.
- Full-episode validation generation was run in Slurm allocation `122782`,
  not on the login node. The strict artifact inspection under
  `eval_full_episode_wam_latest/eval_artifact_inspection.json` reports
  `strict_eval_artifacts_ok=true` with no strict failures across 10 validation
  samples.
- The structural contract was preserved for every generated sample:
  predicted/reference videos are `301/301` frames, and predicted/reference
  action arrays are `300x32/300x32`. Aggregate action RMSE is
  `0.6931767245`; aggregate future-video PSNR is `19.866555` dB.
- Representative per-sample action diagnostics include future-action RMSE
  `0.6447` for target pre-motion `hole_move_stop`, `0.8441` for target-motion
  observed `hole_move_stop`, `1.0028` for target post-motion `hole_reverse`,
  `0.8565` for `peg_drop`, and `0.8073` for `peg_disturb`.
- Visual review of the saved contact sheets is a negative method diagnostic.
  Early-prefix target-motion samples and peg-recovery samples develop severe
  white/transparent mid-to-late rollout artifacts and lose reliable
  robot/peg/target geometry. Static-late and some later-prefix sheets are more
  stable, which suggests the immediate failure is long future prediction under
  dynamic object/contact change rather than checkpoint-load failure,
  128-frame truncation, or shape mismatch.
- Additional sheet review sharpened the failure mode: `peg_disturb` begins
  showing large white/blue fog-like corruption by around frame `136`;
  `static_monitor` also becomes semi-transparent after roughly frame `109`;
  `hole_constant` target pre-motion nearly fully fogs out by around frame
  `82`; and the late `insert_resume none` sheet is better but still shows
  white streaking around the hand/peg after roughly frame `109`. The generated
  videos are therefore not physically reliable task rollouts.

Generated-video task-state readout was also run as a diagnostic on top of the
Cosmos-generated RGB:

- The reference-video readout run in Slurm allocation `122767` produced a
  step-250 snapshot at
  `task_state_readout_reference_rgb_301f/best_model_step250_snapshot.pt`.
  Its reference-video metrics were weak but usable for an early diagnostic:
  all-frame hole-position RMSE `0.079144` m, peg RMSE `0.088382` m, TCP RMSE
  `0.073720` m, grasp accuracy `0.807392`, and insertion accuracy `0.768272`.
- Generated-video readout evaluation under
  `eval_full_episode_wam_latest/task_state_readout_step250/readout_eval_summary.json`
  passed strict shape/finite/reference checks:
  `strict_readout_eval_ok=true`, `num_strict_ok=10`, and mean final
  hole-position error `0.098346` m.
- The readout is not target-motion success evidence. It predicts target-motion
  onset at frames `3-8` for all 10 generated samples, including static `none`
  scenarios that should not have a target-motion onset. For moving-target
  samples, the target onset is around frames `84-94`, so the generated
  video/readout chain is detecting motion far too early.

Conclusion: the early loss around `9-10` was not abnormal by itself, and the
code/checkpoint/data length sanity checks are currently clean. The
`iter_000000300` model nevertheless fails the active world-model objective.
It is only a negative diagnostic: do not use it for controller/DP integration,
positive takeover distillation, or method claims. Continue SFT to later
validation points and re-run the same strict full-episode artifact,
readout-metric, and visual-review gate before considering any controller
handoff.

## 2026-06-09 Source Sanity Repair

While doing the code-level sanity check, `rg` reported that
`scripts/world_model/export_cosmos3_maniskill_full_episode_wam_conditions.py`
contained binary data. The file had 18 trailing NUL bytes after the final
`main()` call. This did not explain the active training loss because the
condition root had already been exported, structurally preflighted, and loaded
by the current SFT dataloader, but it would have broken a future `py_compile`
or exporter rerun.

The trailing NUL bytes were removed without changing the source logic.
Allocation-side syntax checks now pass for the exporter, strict preflight,
Cosmos SFT dataloader, task-state inspector, and generated-video readout eval
script. The exporter file now reports `nul_count=0`.

## 2026-06-09 Iteration-600 Eval Watcher

To avoid overwriting the inspected iteration-300 negative diagnostic, a small
allocation-only watcher was added:
`scripts/slurm/watch_cosmos3_300f_checkpoint_eval_in_allocation.sh`. It refuses
login-node execution, waits for a named checkpoint inside a compute-node
`srun` step, and then calls the existing strict full-episode eval wrapper with
a checkpoint-specific `EVAL_ROOT`.

The watcher for `iter_000000600` is running in Slurm job `122782`, step `18`,
with output log
`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718/eval_iter_000000600_watch.log`
and eval root
`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718/eval_full_episode_wam_iter_000000600`.
It is currently waiting for the checkpoint and has not produced new method
evidence yet.

## 2026-06-09 Inference Interface Diagnosis

The iteration-300 failure is not explained by 128-frame truncation, missing
checkpoint load, or action-shape mismatch. The current action/video contract is:

- `model_mode=policy`.
- `vision_path` points to the full 301-frame reference video, but the sequence
  plan keeps only the prefix latent frames in `condition_frame_indexes_vision`
  as clean conditioning. Future visual tokens are initialized from noise and
  must be generated.
- `action_path` points to a full `300x32` action/state array, but the sequence
  plan keeps only history action indexes before the prefix in
  `condition_frame_indexes_action` as clean conditioning. Future action tokens
  are initialized from noise and must be generated.
- The text prompt names `TARGET_OBJECT=hole`, `TOOL_OBJECT=peg`, and
  `ACTOR=robot_gripper_tcp`, and includes prefix hole/peg/TCP positions,
  target velocity, target-motion-observed, peg-recovery, grasp, and insertion
  status.

This matches the full-episode WAM SFT contract, but it is still a very hard
open-loop diagnostic: early-prefix samples ask the model to generate more than
200 future RGB frames plus future actions and contact geometry from one prefix.
It is stricter than the intended controller-facing receding interface and does
not yet implement DP handoff, live re-observation, or discrete mode switching.

One real inference/SFT mismatch was found: training captions are plain SFT text
with appended duration/FPS/resolution, while the default Cosmos action
inference path wrapped the prompt as a JSON action prompt. The eval input also
used `view_point=maniskill_default_human_render`, which the Cosmos action
formatter does not recognize. This mismatch is a plausible contributor to poor
post-SFT generation quality, although it does not invalidate the length/action
artifact checks.

The next checkpoint eval was repaired before `iter_000000600` starts:

- `external/cosmos-framework/cosmos_framework/inference/action.py` now supports
  `COSMOS3_ACTION_PROMPT_STYLE=plain_sft`, which formats inference text like
  the SFT dataloader: plain caption plus duration/FPS/resolution.
- `scripts/slurm/run_cosmos3_300f_full_episode_wam_eval_in_allocation.sh`
  exports `COSMOS3_ACTION_PROMPT_STYLE=plain_sft`.
- `scripts/world_model/build_cosmos3_full_episode_wam_eval_inputs.py` now uses
  the Cosmos-known `third_person_view` label for eval metadata while preserving
  the actual ManiSkill default human-render video files.

Syntax checks and a small prompt-format check passed inside Slurm allocation
`122782`. The `iter_000000600` watcher was still waiting for the checkpoint
when this fix was applied, so the next eval should use the repaired prompt
interface.

## 2026-06-09 Four-GPU Acceleration Attempt

The user clarified that poor `iter_000000300` quality may also be explained by
insufficient training, and requested trying 4-GPU or 8-GPU acceleration if it
can start immediately, while preserving the rule that any training evidence
must run for at least 1 hour.

- A tmux-held 4-H200 allocation was requested and started immediately as Slurm
  job `123131` on `server09`. No extra 8-GPU pending allocation was kept after
  the 4-GPU allocation was granted, to avoid queue pollution or duplicate
  resource waste.
- The 1-GPU SFT job `122736` was kept alive. It is near the `iter_000000600`
  checkpoint and remains the currently active continuous chain until the 4-GPU
  run has actual training/eval evidence.
- `scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh` was
  parameterized for multi-GPU launch by exposing
  `NPROC_PER_NODE`, `DATA_PARALLEL_SHARD_DEGREE`,
  `DATA_PARALLEL_REPLICATE_DEGREE`, and
  `CONTEXT_PARALLEL_SHARD_DEGREE` in the manifest and train command.
- A 4-GPU warm-start SFT was launched from the clean `iter_000000300` model
  checkpoint into
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_4gpu_warm_iter300_20260609_234018`.
  It reuses the strict full-episode condition root
  `full_episode_wam_conditions_full1000_rgb_300step_20260609_203902`, uses
  `NPROC_PER_NODE=4`, `DATA_PARALLEL_SHARD_DEGREE=-1`,
  `DATA_PARALLEL_REPLICATE_DEGREE=1`, and
  `CONTEXT_PARALLEL_SHARD_DEGREE=1`.
- This 4-GPU run is not evidence yet. It must pass startup/load sanity, run at
  least 1 wall-clock allocation hour, save/validate, and produce strict
  generated-video/action/readout/visual review artifacts before it can replace
  or supersede the 1-GPU chain.

## 2026-06-09 Four-GPU Handoff And One-GPU SFT Stop

The 4-GPU warm-start crossed the minimum "actually running" bar: job `123131`
loaded the `iter_000000300` model, completed startup validation with validation
loss `7.292821`, and entered training with all 4 ranks logging losses from
iteration `1`. After the startup validation, iterations `2-8` ran at roughly
`17.1` seconds per optimizer step, with finite grad norms and finite losses
apart from expected diffusion-timestep spikes.

The duplicate 1-GPU SFT chain had already saved checkpoint `iter_000000600`
under
`sft_full_episode_wam_full1000_rgb_300step_20260609_204718/outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/checkpoints/iter_000000600`.
Following the user's instruction, step `122736.6` was then stopped with
`Ctrl-C` through the tmux session instead of cancelling job `122736`. The
parent allocation `122736.extern` remains alive on `server44` for later sanity
checks, strict eval support, or video/render tasks.

The `iter_000000600` checkpoint watcher in job `122782` detected the
checkpoint and started checkpoint-specific eval under
`eval_full_episode_wam_iter_000000600`. Generated files have begun appearing
for validation samples, but this is not method evidence yet: strict
301-frame/300-action inspection, generated-RGB task readout, contact-sheet
rendering, and visual review are still required.

Current boundary: the active training path is now the 4-GPU warm-start job
`123131`, while `iter_000000600` eval remains a checkpoint diagnostic from the
stopped 1-GPU chain. Neither path is controller evidence yet.

## 2026-06-09 Iteration-600 Full-Episode Diagnostic

The stopped 1-GPU chain produced checkpoint `iter_000000600`, so the waiting
eval allocation ran the same strict 10-sample full-episode diagnostic with the
repaired plain-SFT inference prompt.

Structural results:

- Eval root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_20260609_204718/eval_full_episode_wam_iter_000000600`.
- `strict_eval_artifacts_ok=true`, `strict_failures=[]`, and 10/10 samples
  passed strict artifact inspection.
- Every generated/reference video pair has `301/301` frames, and every
  generated/reference action pair has shape `300x32/300x32`.
- Mean action RMSE improved from the `iter_000000300` diagnostic
  `0.6931767245` to `0.6496965470`.
- Mean future-video PSNR is effectively unchanged: `19.860613` dB versus the
  `iter_000000300` value `19.866555` dB.

Visual review:

- The generated videos are visibly cleaner than `iter_000000300`; the
  inspected target-motion and static sheets no longer show the same
  whole-frame white/fog collapse.
- The target pre-motion and target-motion-observed sheets still fail the
  controller-facing requirement: peg/hand geometry and the peg-head-to-hole
  relationship drift in the middle and late rollout.
- The peg-drop recovery sheet is a clear negative example. The peg remains on
  the table, the generated robot does not regrasp it, and semi-transparent
  hand/object artifacts appear after about frame `109`.
- Static and late insert-resume sheets are more stable, but they still do not
  preserve precise final insertion geometry or reliable peg/hand contact.

Generated-RGB task-state readout was then run inside the freed 1-H200
allocation `122736` using the current reference-video readout
`task_state_readout_reference_rgb_301f/best_model.pt`:

- Readout eval root:
  `eval_full_episode_wam_iter_000000600/task_state_readout_best_current`.
- `strict_readout_eval_ok=true`, `num_strict_ok=10`, so the readout files are
  structurally valid diagnostics.
- Mean final hole-position error is `0.1153032929` m, worse than the
  `iter_000000300` readout diagnostic value `0.098346` m.
- For moving-target samples, predicted target-motion onset is still far too
  early: examples include predicted frame `11` versus GT `88`, frame `6`
  versus GT `84`, frame `10` versus GT `94`, and frame `13` versus GT `91`.
- Static or peg-only scenarios also show false target-motion onsets at frames
  `6-19`.

Conclusion: `iter_000000600` is not a controller/DP handoff checkpoint. It is
useful evidence that continued SFT and the repaired prompt improved visible
stability and action RMSE, but it still fails the physical objective:
target-motion timing, final target pose, peg recovery, and contact-preserving
rollout are unreliable. Continue the 4-GPU SFT path and evaluate its next
checkpoint under the same strict artifact/readout/visual gate.

On `2026-06-10`, the source full1000 review sheet
`sft_dataset_full1000_maniskill_default_regen_20260606_0055/review_sheets/0000_hole_constant_seed702000_n167_traj_0_traj_0_review_sheet.png`
was opened as a data-visual sanity check. The approved default human-render
view is readable across the full episode and clearly shows the robot, peg,
target block, and hole. The `iter_000000600` generated sheets were also
re-opened for target-motion, static, and peg-drop recovery. The re-check
matches the earlier negative interpretation: static is comparatively stable
but still drifts late, target-motion loses reliable peg/hand/contact geometry,
and peg-drop recovery fails to regrasp while producing semi-transparent
artifacts in the latter half. This is not controller evidence.

## 2026-06-09 Four-GPU Checkpoint Watcher

After `iter_000000600` eval finished, the eval allocation was reused for a
4-GPU checkpoint watcher. Slurm step `122782.26` now waits for the 4-GPU
warm-start checkpoint
`sft_full_episode_wam_full1000_rgb_300step_4gpu_warm_iter300_20260609_234018/outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/checkpoints/iter_000000300`
and will write strict eval artifacts under
`sft_full_episode_wam_full1000_rgb_300step_4gpu_warm_iter300_20260609_234018/eval_full_episode_wam_iter_000000300`.

This watcher is only for post-checkpoint evidence generation. It does not
start controller integration and does not relax the 1-hour training floor or
the visual/readout gate.

Before the 4-GPU checkpoint arrived, the watcher implementation was repaired.
The old wait condition could have launched eval as soon as the `model/`
directory existed, even if the DCP `model/.metadata` file had not been written
yet. The watcher now requires `model/.metadata` and a stable checkpoint file
signature before launching eval. Syntax checking was run inside Slurm
allocation `122782`, the old watcher step `122782.26` was canceled, and the
updated watcher is running as step `122782.28`.

## 2026-06-10 Four-GPU Readout Watcher

To avoid a partial 4-GPU checkpoint review that only inspects video/action
artifacts, an allocation-only generated-RGB readout watcher was added:
`scripts/slurm/watch_cosmos3_eval_readout_in_allocation.sh`. It refuses
execution outside a compute-node Slurm step, waits for
`eval_artifact_inspection.json` and `eval_input_manifest.json`, then runs
`scripts/world_model/run_cosmos3_full_episode_readout_eval.py` using the
reference-video readout checkpoint.

The watcher was launched in the freed 1-H200 allocation `122736` as Slurm step
`122736.92`. It waits for the 4-GPU eval artifact under
`sft_full_episode_wam_full1000_rgb_300step_4gpu_warm_iter300_20260609_234018/eval_full_episode_wam_iter_000000300`
and will write generated-RGB readout diagnostics under
`eval_full_episode_wam_iter_000000300/task_state_readout_best_current`.

The reference-video readout training completed with `metrics_best.json` at
step `2000`: future hole RMSE `0.0499867350` m, future peg RMSE
`0.0441538133` m, future TCP RMSE `0.0378160141` m, future peg-head-hole
RMSE `0.0770236030` m, future grasp accuracy `0.9067095518`, and future
insertion accuracy `0.8026654124`. These numbers remain diagnostics on top of
generated RGB only; they are not controller success evidence.

After the readout training step completed and returned to an idle shell, the
separate reference-readout allocation `122767` was released. The old 1-H200
SFT allocation `122736` remains alive as the reserved single-GPU resource for
pending generated-RGB readout watching, later sanity checks, or video/render
work. The stopped 1-GPU SFT step `122736.6` remains gone.

## 2026-06-10 Four-GPU Training-Floor Check

At `2026-06-10 00:42:30 CST`, the 4-GPU warm-start SFT job `123131` had passed
the user's minimum training-duration floor. The allocation had been running
for `1:04:47`, and the SFT step `123131.2` had been running for `1:02:12`.
The log showed the run continuing at iteration `179`, about `17.1` seconds per
iteration, with finite losses and gradient norms.

At `2026-06-10 00:49:16 CST`, the same job was still running at iteration
`203`, about `17.1` seconds per iteration. Recent losses were in the
`2.8-4.8` range, gradient norms were finite, and the device monitor showed
normal multi-GPU memory use. The 4-GPU `iter_000000300` checkpoint had not
been saved yet, so watcher step `122782.26` was still correctly waiting for
the checkpoint path before running strict eval.

At `2026-06-10 00:56:24 CST`, the 4-GPU warm-start was still running at
iteration `228`, still about `17.1` seconds per iteration, with finite
gradient norms and recent losses mostly around `3.0-3.8`. The checkpoint
directory had not appeared yet, so the eval watcher remained in
`checkpoint_not_ready`.

This satisfies the resource/time floor for the 4-GPU training attempt, but it
does not make the checkpoint a world-model result. The run still needs a saved
checkpoint, validation loss inspection, strict 301-frame / 300-action
generation, generated-RGB readout metrics, and visual contact-sheet review
before any controller or DP integration can be considered.

## 2026-06-10 Four-GPU Iteration-300 Diagnostic

The repaired watcher step `122782.28` waited for
`model/.metadata` and a stable DCP checkpoint signature, then evaluated the
4-GPU warm-start checkpoint `iter_000000300` under:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_4gpu_warm_iter300_20260609_234018/eval_full_episode_wam_iter_000000300`.

Strict artifact inspection passed:

- `strict_eval_artifacts_ok=true`, `strict_failures=[]`, 10/10 samples.
- Generated/reference videos are `301/301` frames.
- Generated/reference action tensors are `300x32/300x32`.
- Mean action RMSE is `0.6329251997`, better than the stopped 1-GPU
  `iter_000000600` value `0.6496965470`.
- Mean future-video PSNR is `19.8388587420` dB.

The generated-RGB task-state readout ran in the kept 1-H200 allocation
`122736` as step `122736.92` and wrote:

`eval_full_episode_wam_iter_000000300/task_state_readout_best_current`.

Readout structure passed (`strict_readout_eval_ok=true`, 10/10 strict), but
the target-motion evidence failed:

- Mean final hole-position error is `0.1043972932` m.
- Moving-target examples predict target-motion onset at frames `4-7`, while
  GT onset is frames `84-94`.
- Static or peg-only examples also produce false target-motion onsets at
  frames `8-10`.
- Peg-drop and peg-disturb future grasp accuracies are `0.417910` and
  `0.000000`.

Visual review matches the readout failure. The target-motion and insert-resume
sheets are more stable than the worst early 1-GPU `iter_000000300` artifacts,
but peg/hand/contact geometry still drifts. The peg-drop sample does not
recover or regrasp the peg. The inspected `hole_constant` target-pre-motion
sample develops severe transparent/white artifacts from the middle of the
rollout onward.

Conclusion: 4-GPU `iter_000000300` is a useful training-progress diagnostic,
not a controller/DP handoff checkpoint. It improves the action RMSE but still
fails target-motion timing, final target pose, peg recovery, and visually
consistent contact-preserving rollout. No controller or DP integration should
start from this checkpoint.

## 2026-06-10 Four-GPU Iteration-600 Watchers

Following the user's instruction, the duplicate 1-GPU SFT remains stopped.
The old 1-H200 allocation `122736` is retained only for readout, sanity checks,
and video/render work.

The next 4-GPU checkpoint gate is now active:

- Strict eval watcher `122782.35` is running inside the eval allocation
  `122782`. It waits for 4-GPU `iter_000000600`, again requiring
  `model/.metadata` and stable checkpoint files before eval, and will write
  `eval_full_episode_wam_iter_000000600`.
- Generated-RGB readout watcher `122736.93` is running inside the kept 1-H200
  allocation. It waits for strict eval artifacts from the 4-GPU `iter_000000600`
  eval root and then writes
  `eval_full_episode_wam_iter_000000600/task_state_readout_best_current`.
- Readout-failure-profile watcher `122736.104` is also running inside the kept
  1-H200 allocation. It waits for the 4-GPU `iter_000000600` generated-RGB
  readout summary and then writes
  `eval_full_episode_wam_iter_000000600/task_state_readout_best_current/readout_failure_profile.{json,md}`.

The active 4-GPU SFT job `123131` continues running; controller/DP integration
remains paused until a checkpoint passes strict 301-frame / 300-action
artifacts, generated-RGB task-state readout, and visual review.

At `2026-06-10 01:45:30 CST`, job `123131` was still training at iteration
`385`, about `17.1` seconds per iteration, with finite recent losses mostly
around `2.5-3.8` and finite gradient norms. The old 1-GPU SFT step remains
stopped; allocation `122736` is being used only for readout, profiling,
sanity-check, and video/render support work as requested.

## 2026-06-10 Inference Contract Probe

The 4-GPU `iter_000000300` failure could have been caused either by model
quality or by a framework-facing conditioning mismatch. To localize this
without running another generation job, a lightweight probe was added:

`scripts/world_model/probe_cosmos3_full_episode_inference_contract.py`.

It imports the same Cosmos action inference helper used by eval, builds the
batch from saved `sample_args.json`, and checks the resolved video/action
shapes plus the `SequencePlan` masks. This was run through Slurm allocation
`122736`, not on the login node.

Probe result:

- `strict_full_episode_inference_contract_ok=true`.
- For the representative target-motion sample with prefix frame `92`, Cosmos
  sees video shape `[3, 301, 256, 256]`, action shape `[300, 64]`, raw action
  dim `32`, vision latent condition indexes `0-23`, and action condition
  indexes `0-91`.
- For the representative static sample with prefix frame `80`, Cosmos sees
  video shape `[3, 301, 256, 256]`, action shape `[300, 64]`, raw action dim
  `32`, vision latent condition indexes `0-20`, and action condition indexes
  `0-79`.

The vision indexes are latent-frame indexes, not RGB-frame truncation; with
temporal compression factor `4`, they cover the full RGB prefix. This probe
therefore rules out several implementation-failure hypotheses for the current
negative result: no 128/129/93-frame truncation, no fixed 8-latent default
overriding the per-sample prefix, and no missing framework-facing action
history mask. The remaining failure is a real generated-rollout/readout
quality problem: the model is still not preserving target-motion timing,
final target pose, peg recovery, and contact geometry well enough for
controller/DP handoff.

## 2026-06-10 Readout Failure Profile

To separate a too-sensitive onset threshold from real generated trajectory
drift, a reusable diagnostic was added:

`scripts/world_model/profile_cosmos3_readout_failure_modes.py`.

It reads generated-RGB readout trajectories, compares them against the
full-episode state targets, and reports onset behavior under multiple
displacement thresholds. It does not change any evaluation gate and does not
replace visual review or controller evidence.

The profiler was run inside Slurm allocation `122736` for the 4-GPU
`iter_000000300` readout root and wrote:

`eval_full_episode_wam_iter_000000300/task_state_readout_best_current/readout_failure_profile.json`

and

`eval_full_episode_wam_iter_000000300/task_state_readout_best_current/readout_failure_profile.md`.

Aggregate diagnostic results:

- Mean final hole-position error: `0.1043972907` m.
- Mean future hole RMSE: `0.0519564251` m.
- Mean future peg RMSE: `0.0741150761` m.
- Mean future TCP RMSE: `0.0719546468` m.
- Mean future peg-head-hole RMSE: `0.0539310857` m.

Threshold profile:

- The dynamic target samples still predict target motion far too early at
  larger thresholds such as `0.010` m and often `0.020` m, not only at the
  advisory `0.002` m threshold.
- Static or peg-only samples also show false target drift at `0.010-0.020` m
  in multiple cases, consistent with visual drift/artifacts rather than only
  a thresholding issue.

Conclusion: the current negative result should be interpreted as generated
rollout/readout quality failure. The input contract is intact, but the model
has not yet learned stable target timing, final target pose, peg/TCP
continuity, and peg recovery well enough for controller/DP handoff. The same
profile should be run on the 4-GPU `iter_000000600` eval after its readout
watcher completes.

## 2026-06-10 Four-GPU Iteration-600 Diagnostic

The 4-GPU SFT checkpoint `iter_000000600` was saved at
`2026-06-10 02:46:59 CST`. Watcher `122782.35` waited for stable DCP files and
then completed the same full-episode validation generation under:

`eval_full_episode_wam_iter_000000600`.

Strict artifact inspection passed:

- `strict_eval_artifacts_ok=true`, `strict_failures=[]`, 10/10 samples.
- Generated/reference videos are `301/301` frames.
- Generated/reference action tensors are `300x32/300x32`.
- Mean action RMSE is `0.6199095227`, improved from 4-GPU `iter_000000300`
  at `0.6329251997`.
- Mean future-video PSNR is `20.1762433861` dB, improved from 4-GPU
  `iter_000000300` at `19.8388587420` dB.

Generated-RGB readout and the multi-threshold failure profile also completed
inside the kept 1-H200 allocation `122736`:

- Strict readout/profile structure passed.
- Mean final hole-position error is `0.1023905702` m.
- Mean future hole RMSE is `0.0475034488` m.
- Mean future peg RMSE is `0.0674232871` m.
- Mean future TCP RMSE is `0.0660298159` m.
- Mean future peg-head-hole RMSE is `0.0517559459` m.

The profile remains negative for the active handoff objective. Dynamic target
motion is still predicted tens of frames too early at several thresholds, and
static/peg-only scenes still show false target drift.

Visual review confirms the negative interpretation. The 4-GPU `iter_000000600`
sheets are generally cleaner than `iter_000000300`, but they are not physically
reliable rollouts:

- Target-motion samples preserve more scene structure, but peg/hand/target
  contact geometry drifts late in the rollout.
- The target-post-motion `hole_reverse` sample shows transparent robot/hand
  artifacts around the middle of the rollout.
- Peg-drop recovery does not regrasp the peg; the peg stays on the table while
  the arm moves near the target.
- Peg-disturb recovery develops a large cloud-like artifact over the target.
- Insert-resume samples lose or mis-handle the peg, so the predicted rollout
  would not let the frozen DP resume insertion.
- Static samples are comparatively stable, but still show target/hand/peg
  offsets and false target drift in the readout profile.

Conclusion: 4-GPU `iter_000000600` is training progress, not controller/DP
handoff evidence. It improves action/video metrics but still fails target
motion timing, final target pose readout, peg recovery, and executable
peg/hand continuity. No controller or DP integration should start from this
checkpoint.

## 2026-06-10 Four-GPU Iteration-900 Watchers

After the `iter_000000600` negative review, the next checkpoint gate was
launched while the 4-GPU SFT continued training:

- Strict eval watcher `122782.36` waits for `iter_000000900` and will write
  `eval_full_episode_wam_iter_000000900`.
- Generated-RGB readout watcher `122736.105` waits for that eval root and will
  write `eval_full_episode_wam_iter_000000900/task_state_readout_best_current`.
- Readout-failure-profile watcher `122736.106` waits for the `iter_000000900`
  readout summary and will write `readout_failure_profile.{json,md}` under the
  same readout root.

At the time these watchers were launched, the 4-GPU training job `123131` was
still running at about iteration `638` with finite losses and gradient norms.
The old 1-GPU SFT remains stopped; allocation `122736` is still used only for
readout, profiling, sanity-check, and video/render support work.

## 2026-06-10 One-GPU Support Allocation Correction

Later inspection showed that old allocation `122736` had been revoked:
`squeue -j 122736` returned an invalid job id, and the tmux shell reported
`salloc: Job allocation 122736 has been revoked`. Its stale tmux session was
removed so it cannot be mistaken for an active 1-GPU SFT run.

This is a scheduling/allocation-expiration event, not model evidence. The
cancelled support steps `122736.105` and `122736.106` did not complete the
4-GPU `iter_000000900` readout/profile gate. The active `iter_000000900`
strict eval watcher remains `122782.36` inside eval allocation `122782`.

Per the user's resource instruction, no 1-GPU SFT should be restarted while
the 4-GPU experiment is healthy. A replacement tmux-held auxiliary 1-H200
allocation was opened as Slurm job `123366` (`cosmos3_aux_1h200_0610`) with a
one-day time limit. It started on `server21`. A compute-node canary through
`srun --jobid=123366` confirmed one visible H200 and PyTorch CUDA availability.

The cancelled iter900 support steps were then relaunched inside `123366`:

- `123366.1`: generated-RGB readout watcher, waiting for
  `eval_full_episode_wam_iter_000000900/eval_artifact_inspection.json`.
- `123366.2`: readout-failure-profile watcher, waiting for the generated-RGB
  readout summary.

These steps are support work only. They do not restart 1-GPU SFT and do not
change the controller/DP gate.

## 2026-06-10 Four-GPU Iteration-900 Diagnostic

The 4-GPU checkpoint `iter_000000900` was saved at
`2026-06-10 04:16:54 CST`. Watcher `122782.36` waited for stable files and ran
the same full-episode validation generation under:

`eval_full_episode_wam_iter_000000900`.

Strict artifact inspection passed:

- `strict_eval_artifacts_ok=true`, `strict_failures=[]`, 10/10 samples.
- Generated/reference videos are `301/301` frames.
- Generated/reference action tensors are `300x32/300x32`.
- Mean action RMSE is `0.6172093662`, slightly improved from
  `iter_000000600` at `0.6199095227`.
- Mean future-video PSNR is `20.2230188270` dB, slightly improved from
  `iter_000000600` at `20.1762433861` dB.

The generated-RGB readout and failure profile completed inside the auxiliary
1-H200 allocation `123366`:

- Strict readout/profile structure passed.
- Mean final hole-position error is `0.1021599918` m.
- Mean future hole RMSE is `0.0477525963` m.
- Mean future peg RMSE is `0.0668954306` m.
- Mean future TCP RMSE is `0.0657577841` m.
- Mean future peg-head-hole RMSE is `0.0519184658` m.

The profile remains negative for the active handoff objective. Dynamic target
onsets are still predicted tens of frames too early at several thresholds, and
static/peg-only scenes still show false target drift.

Visual review also remains negative. Target-motion and target-post-motion
sheets are cleaner than early checkpoints but still drift in robot, peg, and
target contact geometry. Insert-resume does not show reliable peg holding.
Peg-drop recovery does not regrasp the peg. Peg-disturb recovery still has a
large cloud-like artifact over the target. Static-late is comparatively stable,
but that does not prove the dynamic handoff objective.

Conclusion: `iter_000000900` is training progress, not controller/DP handoff
evidence. It improves action/video metrics slightly but still fails target
motion timing, final target pose readout, peg recovery, and executable
peg/hand continuity. No controller or DP integration should start from this
checkpoint.

The next checkpoint gate was launched while 4-GPU training continued:

- Strict eval watcher `122782.37` waits for `iter_000001200`.
- Generated-RGB readout watcher `123366.3` waits for the `iter_000001200`
  eval root.
- Readout-failure-profile watcher `123366.4` waits for the `iter_000001200`
  readout summary.

## 2026-06-10 Four-GPU Iteration-1200 Waiting State

At `2026-06-10 04:34:53 CST`, the active 4-GPU SFT job `123131` was still
running on `server09` at iteration `948`, with finite losses, finite gradient
norms, and about `17.1` seconds per iteration. Existing checkpoints under the
active 4-GPU root are `iter_000000300`, `iter_000000600`, and
`iter_000000900`; `iter_000001200` has not been saved yet.

The next gate is therefore still waiting, not failed:

- Strict eval watcher `122782.37` is waiting in the eval allocation on
  `server52`.
- Generated-RGB readout watcher `123366.3` is waiting in the auxiliary
  1-H200 allocation on `server21`.
- Readout-failure-profile watcher `123366.4` is waiting in the same auxiliary
  allocation.

The old 1-GPU SFT remains stopped. The old allocation `122736` is gone; the
replacement 1-H200 allocation `123366` is reserved for readout, sanity checks,
and video/render support only. This section records resource/provenance state,
not world-model success evidence.

The active training config confirms this is the intended final stretch of the
current SFT run: `trainer.max_iter=1500`, `checkpoint.save_iter=300`,
`trainer.validation_iter=300`, and `trainer.max_val_iter=40`. Validation loss
for the 4-GPU warm-start run was:

- iteration `0`: `7.292821`
- iteration `300`: `6.597002`
- iteration `600`: `6.005106`
- iteration `900`: `6.178823`

The iteration-900 validation loss increase means loss has not shown a clean
monotone improvement trend. This is not a reason to stop early or to launch
controller/DP integration. The gate still requires strict generated artifacts,
generated-RGB readout/profile metrics, and visual review for `iter_000001200`
and the final `iter_000001500`.

Additional training-log sanity at `2026-06-10 04:38 CST` found no Traceback,
OOM, NaN, or runtime-error evidence. The only `nan` text hit is the configured
`skip_nan_step` callback. Training contains occasional single-step large finite
losses, including `iter 885` loss `49.2275` and `iter 951` loss `49.5979`, but
they are immediately followed by normal finite losses and finite gradient
norms. This supports continuing the run rather than intervening before the
strict `iter_000001200` gate.

At `2026-06-10 04:41:52 CST`, the same 4-GPU job was at iteration `972`.
`iter_000001200` was still not saved, so the currently correct state is
waiting, not failure. The active waiting steps are:

- `122782.37`: strict eval watcher in the eval allocation.
- `123366.3`: generated-RGB readout watcher in the auxiliary 1-H200 allocation.
- `123366.4`: readout-failure-profile watcher in the auxiliary 1-H200
  allocation.

After the `iter_000001200` gate completes, the next configured checkpoint is
the final `iter_000001500` because the active training config has
`trainer.max_iter=1500`. The final gate must use the same strict 301-frame /
300-action generated-video/action/readout/profile/visual review. It must not
restart 1-GPU SFT, relax the length contract, or start controller/DP
integration unless the generated artifacts, metrics, and inspected videos
actually pass the handoff gate.

At `2026-06-10 04:43:53 CST`, the training job was still healthy at iteration
`980` with recent finite losses in the `2.4-3.5` range. The `iter_000001200`
checkpoint still had not been saved, and no `eval_full_episode_wam_iter_000001200`
artifacts existed yet. The eval/readout/profile watchers remain in the
expected waiting state. This remains a waiting/provenance note, not model
evidence and not a controller/DP gate opening.

Resource/timing check at `2026-06-10 04:45 CST`:

- Eval allocation `122782` runs until `2026-06-10 10:22:09 CST`.
- Auxiliary 1-H200 allocation `123366` runs until
  `2026-06-11 04:08:09 CST`.
- Active 4-GPU SFT allocation `123131` runs until
  `2026-06-10 23:37:43 CST`.
- The active watcher timeouts are `7200` seconds for checkpoint/eval,
  `10800` seconds for generated-RGB readout, and `14400` seconds for the
  readout-failure profile.

At roughly `iter 984`, those held allocations should cover the `iter_000001200`
gate and the final `iter_000001500` gate. No 1-GPU SFT restart is justified by
resource state.

At `2026-06-10 04:47:48 CST`, the 4-GPU job was still healthy at iteration
`993` with finite recent losses around `2.8-3.4`. `iter_000001200` had still
not been saved, and no `eval_full_episode_wam_iter_000001200` artifacts were
present. The eval/readout/profile watchers remain in the expected waiting
state. This is still waiting/provenance only, not generated world-model
evidence.

At `2026-06-10 04:48:51 CST`, the 4-GPU job was still training at iteration
`997` with finite recent losses around `2.7-3.5`. `iter_000001200` was still
not saved, and no eval/readout/profile artifacts existed. The correct action
remains waiting for the existing compute-node watchers rather than changing
the protocol, restarting 1-GPU SFT, or starting controller/DP work.

At `2026-06-10 04:50:59 CST`, the resource state was checked again against
the user's latest instruction. The only active SFT line is the 4-GPU job
`123131` on `server09`, now around iteration `1004` with finite recent losses.
No 1-GPU SFT step is running, and no stale old 1-GPU SFT tmux session remains.
The single-GPU resources are support resources only: eval watcher `122782.37`
on `server52`, generated-RGB readout watcher `123366.3` on `server21`, and
readout-failure-profile watcher `123366.4` on `server21`. While the 4-GPU SFT
remains healthy, 1-H200 time must be kept for strict eval, readout/profile,
sanity checks, and video or render support rather than restarting 1-GPU SFT.

At `2026-06-10 04:52:48 CST`, training was still healthy at iteration `1011`
with finite losses and gradient norms. The `iter_000001200` checkpoint was
not yet present; existing checkpoint directories remained only
`iter_000000300`, `iter_000000600`, and `iter_000000900`. The strict eval
watcher and both auxiliary readout/profile watchers were correctly waiting.
Because the single-GPU allocations are already reserved by those waiting
steps, the final `iter_000001500` watcher should be launched only after the
`iter_000001200` gate completes or releases resources. This preserves the
same full 301-frame / 300-action contract and avoids turning 1-H200 time back
into a duplicate SFT path.

At `2026-06-10 05:25:48 CST`, the 4-GPU SFT was still running at iteration
`1126`. The `iter_000001200` checkpoint remained absent and watcher
`122782.37` was still correctly waiting. The last 30-minute check observed
a large finite single-step loss at iteration `1121` (`73.6864`), followed
immediately by normal finite losses at iterations `1122-1126`
(`2.9853`, `2.7811`, `2.8972`, `3.0977`, `2.6466`) with finite gradient
norms. A log scan found prior similar finite spikes and no NaN, OOM, or
Traceback evidence. This is a training-monitor note only; method evidence
still requires the generated artifacts, readout/profile, and visual review
at `iter_000001200`.

## 2026-06-10 Four-GPU Iteration-1200 Gate

The 4-GPU `iter_000001200` checkpoint was saved and evaluated under the same
full-episode contract. Strict artifact inspection passed:

- `10/10` validation samples inspected.
- Predicted/reference videos are `301/301` frames.
- Predicted/reference actions are `300x32/300x32`.
- Mean action RMSE is `0.6162060709`.
- Mean future-video PSNR is `20.2295638395` dB.

Generated-RGB task-state readout and failure profiling also completed inside
the auxiliary 1-H200 allocation. The readout/profile structure passed, but the
diagnostic quality remains negative:

- Mean final hole-position error: `0.1028519175` m.
- Mean future hole RMSE: `0.0481315461` m.
- Mean future peg RMSE: `0.0667989680` m.
- Mean future TCP RMSE: `0.0662642823` m.
- Mean future peg-head-hole RMSE: `0.0515516010` m.
- Target-motion onsets are still predicted tens of frames too early, including
  false target drift in static and peg-only scenarios.

All 10 review sheets were opened and inspected. The visual result is still
negative for controller/DP handoff:

- Target pre-motion and target-motion-observed samples keep drifting in
  peg/hand/contact geometry and do not produce a reliable final target/peg
  relation.
- Target-post-motion/reverse develops semi-transparent robot/hand artifacts
  and unstable peg-hole contact geometry.
- Insert-resume samples do not maintain the peg in a continuable insertion
  pose relative to the target.
- Peg-drop does not regrasp the peg; the peg remains on the table while the
  gripper/cube geometry becomes unreliable.
- Peg-disturb produces a large noisy cloud-like artifact over the target.
- Static-late is visually cleaner than dynamic cases, but it does not prove
  target-motion monitoring, final-pose prediction, or recovery competence.

Conclusion: `iter_000001200` is not controller/DP evidence. It preserves the
strict equal-length contract and slightly improves aggregate action/video
metrics, but it still fails target-motion timing, final target pose readout,
peg recovery, and executable hand/peg continuity.

After this negative review, the final configured checkpoint gate was launched:

- Strict eval watcher `122782.38` is running in the eval allocation.
- Generated-RGB readout watcher `123366.6` is running in the auxiliary
  1-H200 allocation.
- Readout-failure-profile watcher `123366.5` is running in the same auxiliary
  allocation.

The `iter_000001500` checkpoint was saved at `2026-06-10 07:16:51 CST`.
Watcher `122782.38` is waiting for stable checkpoint files before running the
same strict generated-video/action eval. The final gate must still include
generated-RGB readout, readout-failure profile, and visual review before any
controller/DP decision.

## 2026-06-10 Four-GPU Iteration-1500 Final Gate

The 4-GPU SFT reached the configured final iteration and stopped normally.
Validation loss at `iter_000001500` was `6.158292`. The validation-loss series
for the 4-GPU warm-start run was:

- `iter 0`: `7.292821`
- `iter 300`: `6.597002`
- `iter 600`: `6.005106`
- `iter 900`: `6.178823`
- `iter 1200`: `6.284197`
- `iter 1500`: `6.158292`

The final checkpoint passed strict artifact accounting under the active
full-episode contract:

- `10/10` validation samples inspected.
- Predicted/reference videos are `301/301` frames.
- Predicted/reference actions are `300x32/300x32`.
- Mean action RMSE is `0.6161383192`.
- Mean future-video PSNR is `20.2640810606` dB.

Generated-RGB task-state readout and failure profiling completed inside the
auxiliary 1-H200 allocation. The readout/profile structure passed, but the
diagnostic quality remains negative:

- Mean final hole-position error: `0.1028195255` m.
- Mean future hole RMSE: `0.0485962523` m.
- Mean future peg RMSE: `0.0669838827` m.
- Mean future TCP RMSE: `0.0666590482` m.
- Mean future peg-head-hole RMSE: `0.0516100994` m.
- Target-motion onsets are still predicted far too early. At the 2 mm
  threshold, moving-target samples predict onset around frames `4-7` while
  ground truth is around frames `84-94`, and static/peg-only samples still
  produce false target drift.

All 10 final review sheets were opened and inspected. The visual result is
negative for controller/DP handoff:

- Target pre-motion, target-motion-observed, and target-post-motion samples
  still drift in robot/peg/target contact geometry.
- Insert-resume samples do not maintain a continuable peg pose relative to the
  target.
- Peg-drop does not regrasp the peg; the peg remains on the table while the
  predicted gripper/target geometry becomes unreliable.
- Peg-disturb produces a large cloud-like artifact over the target.
- Static-late is comparatively stable, but it does not prove target-motion
  monitoring, final-pose prediction, or recovery competence.

Conclusion: `iter_000001500` is not controller/DP evidence. It preserves the
strict equal-length contract and is the best completed final SFT diagnostic,
but it still fails target-motion timing, final target pose readout, peg
recovery, and executable hand/peg continuity.

After the final gate completed, idle Slurm allocations `123131` and `122782`
were released. The old auxiliary allocation `123366` was then revoked after its
watcher work completed, so a replacement tmux-held 1-H200 allocation `123381`
was acquired on `server21`, reserved for later sanity checks, readout/video
inspection, or render work. No 1-GPU SFT is active or planned.

Latest resource boundary: since the 4-GPU full-episode SFT ran through to the
strict eval gate, no 1-GPU SFT fallback should be restarted from this chain. The
held 1-H200 allocation is reserved for short sanity checks, readout/debug steps,
and video/render work needed by the next reviewed repair.

## 2026-06-10 Reference-RGB Readout Calibration

After the final generated-RGB gate failed, a reference-RGB calibration was run
inside held Slurm allocation `123381` on the same 10 validation samples. The
calibration root is:

```text
experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_4gpu_warm_iter300_20260609_234018/eval_full_episode_wam_iter_000001500/reference_rgb_readout_calibration/
```

This root uses the same `eval_input_manifest.json`, but
`inference/<sample>/vision.mp4` is a symlink to each sample's GT reference RGB.
It is readout calibration only, not Cosmos3 generated world-model evidence.

The calibration completed with strict readout/profile structure on all 10
samples:

- Reference-RGB mean final hole error: `0.0456793996` m.
- Reference-RGB mean future hole RMSE: `0.0277646941` m.
- Reference-RGB mean future peg RMSE: `0.0248001007` m.
- Reference-RGB mean future TCP RMSE: `0.0209654087` m.
- Reference-RGB mean future peg-head-hole RMSE: `0.0359646693` m.

Compared with generated RGB at `iter_000001500`, the reference-RGB floor is
substantially better. Generated RGB remains at `0.1028195255` m mean final hole
error and `0.0485962523` / `0.0669838827` / `0.0666590482` m future
hole/peg/TCP RMSE. Therefore, the final checkpoint failure is not only a
readout artifact; Cosmos3 rollout quality still damages target, peg, and TCP
geometry.

Target-onset scoring was also run from readout displacement:

- Generated-RGB frame AUROC: `0.7805157241`.
- Generated-RGB fixed 2 mm F1: `0.5295591182`.
- Generated-RGB best F1: `0.6556728232` at score threshold
  `0.0193880412` m.
- Generated-RGB moving-onset mean absolute error: `82.4` frames.
- Generated-RGB static false-positive samples: `5/5`.
- Reference-RGB frame AUROC: `0.8549271165`.
- Reference-RGB fixed 2 mm F1: `0.5304893350`.
- Reference-RGB best F1: `0.7244258873` at score threshold
  `0.0610896768` m.
- Reference-RGB moving-onset mean absolute error: `82.0` frames.
- Reference-RGB static false-positive samples: `5/5`.

Conclusion: low-threshold displacement-on-readout is not a valid
controller-facing mode-switch signal. The next target-monitoring interface
needs a calibrated target-motion/onset head or score. This does not open
controller/DP integration, because the generated RGB/action/readout/visual gate
still fails.

## 2026-06-10 Calibrated Target-Motion Head

A learned temporal target-motion head was added and evaluated after the
low-threshold displacement score failed. The scripts are:

- `scripts/world_model/build_cosmos3_reference_rgb_eval_root_from_jsonl.py`
- `scripts/world_model/train_target_motion_head_from_readout.py`
- `scripts/slurm/run_cosmos3_target_motion_head_calibration_in_allocation.sh`

The run used Slurm allocation `123381` and wrote artifacts under:

```text
experiments/world_model_task_rebinding/cosmos3/target_motion_readout_calibration_20260610_0751/
```

The calibration set was built from the full-episode condition root without
creating short clips:

- Train reference RGB: `120` unique source episodes, balanced as `20` samples
  each from `none`, `hole_move_stop`, `hole_constant`, `hole_reverse`,
  `peg_disturb`, and `peg_drop`.
- Val reference RGB: `88` unique source episodes covering all available val
  scenarios.
- Generated stress test: the final `iter_000001500` generated-RGB readout
  panel.

The head is trained on features from RGB-derived task-state readout
trajectories, not oracle simulator state at inference. The labels come from
state targets only for supervised diagnostic training/evaluation.

Metrics:

- Train reference RGB: AUROC `0.9994329622`, F1@0.5 `0.9850033282`, best F1
  `0.9863672183`.
- Held-out val reference RGB: AUROC `0.9115353604`, F1@0.5 `0.7669897596`,
  best F1 `0.7788987602`.
- Final iter1500 generated RGB: AUROC `0.7756010330`, F1@0.5
  `0.6111111111`, best F1 `0.6428281187`.

Conclusion: the target-motion switch should not use raw 2 mm displacement, and
a calibrated temporal head is a better target-monitoring diagnostic. However,
the drop from reference RGB to generated RGB shows that the final Cosmos3
rollout still corrupts the readout regime enough that controller/DP integration
must remain closed.
