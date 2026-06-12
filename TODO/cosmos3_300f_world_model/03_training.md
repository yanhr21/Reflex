# Training TODO

## Current Active Boundary

- [x] The 2026-06-12 `normactive_clip1` v7_733 SFT is no longer active
      evidence. User visual review matched the old "action adapter not really
      trained" failure mode, and the config audit confirmed the root cause:
      the wrapper selected the action tensors but failed to carry the
      overfit-validated fix1 action recipe. Its log used `lr=2e-5`,
      `action_loss_weight=10.0`, `independent_action_schedule=false`, and
      `shift_action=None`. That run was interrupted with tmux `Ctrl-C`, not
      `scancel`, preserving the 4-H200 allocation.
- [x] Base full-episode WAM SFT defaults now use and enforce the fix1 action
      recipe: optimizer keys include `action2llm,llm2action,action_modality_embed`,
      `lr=1e-4`, warmup `10`, `f_min=0.5`, `grad_clip_norm=1.0`,
      `action_loss_weight=2.0`, `normalize_loss_by_active=true`,
      `independent_action_schedule=true`, and `shift_action=1`. Non-fix1
      action recipes now require explicit diagnostic override.
- [x] Current active gate is a fresh v7_733 two-sample overfit, not full-data
      SFT. Condition root:
      `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_overfit2_rgb_300step_20260612_1830`.
      Train and val contain the same two full-episode rows:
      `hole_late_move_stop / target_motion_observed` and
      `peg_drop / peg_recovery`. Preflight and action-target audit passed:
      both rows are `301` RGB frames with `300x32` future-aligned action/state
      targets, robot-action variation, and task-state sidecar variation.
- [x] Current overfit SFT root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_overfit2_rgb_300step_fix1recipe_4gpu_20260612_1840`.
      It ran on Slurm step `126210.40` on `server56` with 4 H200 GPUs,
      `MAX_ITER=300`, `SAVE_ITER=100`, and `VALIDATION_ITER=50`. Manifest and
      Hydra logs confirm the fix1 recipe (`action_loss_weight=2.0`,
      `independent_action_schedule=True`, `shift_action=1`, `lr=0.0001`) and
      optimizer selection of `410` tensors / `6,982,401,216` elements. The
      loss curve passed the overfit sanity signal: validation loss `3.393175`
      at iter0, `0.390711` at iter50, and `0.125564` at iter100. The foreground
      training step was stopped with tmux `Ctrl-C` after iter100 checkpoint
      save and eval launch; the held allocation was not cancelled.
- [x] Iter100 strict overfit eval completed in the held 1-H200 auxiliary
      allocation. Eval root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_overfit2_rgb_300step_fix1recipe_4gpu_20260612_1840/eval_full_episode_wam_iter_000000100`.
      Strict inspection passed for both samples: generated/reference videos
      `301/301`, actions `[300,32]`, `strict_failures=[]`, mean action RMSE
      `0.1921146387`, mean state-sidecar future RMSE `0.2256914438`, and mean
      future-video PSNR `28.3125916850`.
- [x] User visual approval was received on 2026-06-12 for these two generated
      overfit videos:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_overfit2_rgb_300step_fix1recipe_4gpu_20260612_1840/eval_full_episode_wam_iter_000000100/inference/00_peg_recovery_peg_drop_peg_drop_seed705095_idx0004.fix3_traj_0__peg_recovery_f131/vision.mp4`
      and
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_overfit2_rgb_300step_fix1recipe_4gpu_20260612_1840/eval_full_episode_wam_iter_000000100/inference/01_target_motion_observed_hole_late_move_stop_hole_late_move_stop_seed1080087_idx1760.fix3_traj_0__target_motion_observed_f117/vision.mp4`.
- [x] Start full-data v7_733 SFT again only with the enforced fix1 recipe and
      the frozen 733-row condition root. Do not reuse the rejected
      `normactive_clip1` recipe. After startup, monitor validation loss and
      launch strict generated-video/action/readout/visual gates at saved
      checkpoints before any controller/DP integration.
- [x] Full-data v7_733 SFT was restarted at
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`.
      It runs in tmux `cosmos3_sft_v7_733_full_fix1recipe_4gpu_126210` on
      Slurm step `126210.41` (`server56`, `4xH200`). The first attempt failed
      before training because it requested more memory/CPU than the held
      allocation had; the successful launch uses `240G/32CPU` inside the same
      allocation.
- [x] Startup checks passed: `fix1_action_recipe_check=passed`, `410` selected
      trainable tensors, optimizer `lr=0.0001`, `action_loss_weight=2.0`,
      `independent_action_schedule=true`, and `shift_action=1`. Training
      reached `Starting training...`; iter0 validation loss is `3.606580`, and
      early rank-0 losses are finite (`3.0716` at iteration 7).
- [x] Iter300 strict generated-eval watcher completed in tmux
      `cosmos3_v7_733_full_eval_aux_request_0612` on fresh auxiliary Slurm job
      `127120` (`server40`, `1xH200`), after old auxiliary job `126985` was
      cancelled by Slurm. No `scancel` was used by the agent.
- [x] Inspect iter300 strict generated videos/action metrics/readout and visual
      sheets. Iter300 checkpoint is structurally valid and much better than
      the rejected bad-recipe run: validation loss `0.155843`, future video
      PSNR `21.6543`, action tensor shape `300x32`, robot-action future RMSE
      `0.6354`, state-sidecar future RMSE `0.3534`, generated-RGB mean final
      hole error `0.0655` m, and direct review of all `10` sheets found no
      old white-fog/geometry-collapse failure. It is still not controller-ready
      because target-onset diagnostics fire early/false on low thresholds and
      several final peg/hole relative poses remain imprecise.
- [ ] Continue the same SFT toward iter600/900 while eval gates run. An
      `iter_000000600` strict eval watcher is active in tmux
      `cosmos3_v7_733_full_iter600_eval_watch_127120` on the held auxiliary
      Slurm allocation `127120`.
- [x] Historical entry superseded: the rejected follow-up run was:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_normactive_clip1_4gpu_20260612_124500`.
      It uses the frozen user-override `733`-row v7 DP source and the existing
      WAM condition root under the strict 301-frame / 300-action contract.
- [x] Historical status before rejection: as of `2026-06-12 18:08 CST`, it was
      live on Slurm step `126210.38` on
      `server56`, has reached rank-0 iteration `1060`, and has no NaN/OOM/error
      marker in the inspected logs. Checkpoints available so far are
      `iter_000000300`, `iter_000000600`, and `iter_000000900`.
- [x] Historical status before rejection: the held auxiliary eval allocation was
      `126985` on `server40`.
      Watcher step `126985.9` is waiting for `iter_000001200`; no iter1200
      strict eval/readout/profile artifact exists yet. Premature
      readout/profile watchers that were competing for the same 1-GPU
      allocation were interrupted with tmux `Ctrl-C`, preserving the
      allocation and leaving only the strict eval watcher active. Readout and
      profile should be relaunched after strict iter1200 eval artifacts exist.
- [x] Historical entries below about the 2026-06-09/10 full1000/fix1 runs,
      old allocations, and previous warm-start checkpoints remain useful
      training/software diagnostics, but they are not the current active SFT
      status.

## Before SFT

- [x] Audit local Cosmos3 checkpoints and decide between
      `Cosmos3-Nano-Policy-DROID`, `Cosmos3-Nano-Policy-DROID-DCP`, and any
      stronger available variant. The active run uses the available
      `Cosmos3-Nano-Policy-DROID-DCP` constrained path: it started as a
      1-H200 baseline, then moved to the current 4-H200 warm-start run after
      the 4-GPU allocation became healthy.
- [x] Verify the selected Cosmos adapter can represent the full 300-step
      episode/equal-length contract.
- [ ] If tooling cannot support it, stop and report the limitation; do not
      fall back to 128-step chunks.
- [x] Run syntax/path/preflight/debug checks only inside a Slurm allocation,
      not on the login node.
- [x] Create a clean full-episode exporter:
      `scripts/world_model/export_cosmos3_maniskill_full_episode_wam_conditions.py`.
- [x] Create a strict full-episode preflight:
      `scripts/world_model/preflight_cosmos3_full_episode_wam_contract.py`.
- [x] Create an allocation-only export/preflight/SFT wrapper:
      `scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh`.
- [x] Test those scripts inside the tmux-held `salloc` allocation before
      treating any exported condition root as valid.

## SFT Job

- [x] Submit training through Slurm on H200 resources. User approval to begin
      execution has been given.
- [x] Acquire resources with a tmux-held `salloc` first. Prefer an allocation
      long enough for iterative debug, export, preflight, training, validation
      generation, and monitoring. Do not default to one-shot `sbatch`.
- [x] Request useful GPU shapes pragmatically: prefer 4 or 8 GPUs when they
      can start, allow 1 GPU to avoid waiting, and never count training under
      1 GPU / 1 hour as method evidence.
- [x] Run export, preflight, script checks, and SFT startup inside the
      allocation, not on the login node.
- [ ] Run post-SFT validation generation/rendering inside the allocation, not
      on the login node.
- [x] Prepare post-SFT validation generation/rendering scripts before the first
      checkpoint:
      `scripts/world_model/build_cosmos3_full_episode_wam_eval_inputs.py`,
      `scripts/world_model/inspect_cosmos3_full_episode_wam_eval_artifacts.py`,
      and
      `scripts/slurm/run_cosmos3_300f_full_episode_wam_eval_in_allocation.sh`.
      Syntax checks and eval-input generation were run through Slurm job
      `122736` with `srun --overlap`, not on the login node.
- [x] Use a new condition root named for full-episode 300-step WAM conditions;
      do not reuse old `chunked`, `128`, `129`, or `93` roots.
- [x] Strict full-episode preflight passed for
      `full_episode_wam_conditions_full1000_rgb_300step_20260609_203902`:
      train `3808` rows / `912` source episodes, val `384` rows / `88`
      source episodes, all rows 301 RGB frames and 300 action steps.
- [x] Action target audit passed: every row has `300x32` structured action
      sidecar and nondegenerate robot action variation.
- [ ] Monitor validation and continue until validation no longer clearly
      improves rather than stopping at the first checkpoint.
- [x] Let the active SFT run for at least 1 GPU-hour before treating it as
      training evidence. The initial 1-H200 root is
      `sft_full_episode_wam_full1000_rgb_300step_20260609_204718`; the current
      active 4-H200 root is
      `sft_full_episode_wam_full1000_rgb_300step_4gpu_warm_iter300_20260609_234018`.
- [ ] Current monitor note: by `2026-06-09 21:45:55` the active run reached
      about iteration `177`. Loss had dropped from early `10-14` values into
      mostly `4.7-6.5`, with occasional finite diffusion/action timestep
      spikes. No checkpoint or iteration-300 validation has been written yet,
      so this remains in-progress SFT evidence only.
- [ ] Updated monitor note: by `2026-06-09 21:56:34` the active run reached
      iteration `215`. Recent loss values are mostly `3.7-5.7`, grad norms are
      finite, and the log still shows no NaN/OOM/crash or checkpoint
      key/shape-mismatch failure. This supports the load/data sanity check but
      is still not generated-video evidence.
- [ ] Iteration-300 monitor note: checkpoint `iter_000000300` was saved and
      validation loss was `6.964001`, down from initial validation loss
      `15.253426`. Recent train losses around this checkpoint are mostly
      `3-5` with finite occasional spikes. This supports code/load/numeric
      sanity, but validation loss alone is not method evidence.
- [x] Save config, strict manifest, checkpoint metadata, and logs under a new
      clean full-episode root.
- [ ] Track validation loss but do not treat it as method evidence by itself.
- [ ] Preserve the full action/state/video length accounting in every saved
      manifest.

## 2026-06-10 Two-Sample Overfit Code Audit

- [x] Run a strict two-sample overfit diagnostic using the accepted full
      301-frame / 300-action contract, not a 128/129-frame chunk. The root
      `sft_full_episode_wam_overfit2_v1_4gpu_20260610_0908` reached
      `iter_000000300` on 4 H200 GPUs, but validation loss only reached
      `4.000712` and the training loss remained around `3.8-4.5`. This is
      negative code/training-logic evidence, not a "needs more full-data
      training" explanation.
- [x] Inspect the current Cosmos3 WAM training code against the existing robot
      world-model reference style in `Genie-Envisioner`. The reference trains
      memory-conditioned future chunks with explicit future masks and an
      independent action noise/loss path. The current Cosmos3 path is still
      full-episode/equal-length, but the old config froze action adapter
      modules and did not expose the action-loss/scheduler knobs needed for
      a meaningful two-sample overfit sanity check.
- [x] Patch the active SFT wrapper so the action interface is trainable:
      `optimizer.keys_to_select` now includes `action2llm`, `llm2action`, and
      `action_modality_embed` in addition to the video/text bridge modules.
- [x] Patch the active SFT wrapper to expose and record overfit-critical
      knobs: optimizer LR, scheduler cycle/warmup/f_min, grad clip norm,
      action loss weight, active-token loss normalization, independent action
      schedule, and `shift_action`.
- [x] Patch training logs to print component losses
      `flow_matching_loss_vision` and `flow_matching_loss_action`, because a
      single total loss hid the fact that the two-sample failure was dominated
      by the action/structured branch.
- [x] Launch fix1 on the existing 4-H200 allocation, not as a smaller
      fallback:
      `sft_full_episode_wam_overfit2_fix1_4gpu_20260610_1045`, Slurm job
      `123385`, step `123385.6`, `NPROC_PER_NODE=4`,
      `DATA_PARALLEL_SHARD_DEGREE=4`, `lr=1e-4`, `warmup=10`,
      `f_min=0.5`, `grad_clip_norm=1.0`,
      `normalize_loss_by_active=true`,
      `independent_action_schedule=true`, `shift_action=1`, and
      `action_loss_weight=2.0`.
- [x] Initial fix1 sanity: optimizer selected `410` tensors /
      `6,982,401,216` elements, confirming action adapter parameters are
      included. Early loss split shows vision loss already small
      (`~0.05-0.14`) while action loss drops from `~1.88` to `~1.42` by
      iteration `6`, so the current overfit bottleneck is explicitly the
      action/structured branch.
- [ ] Continue fix1 for at least the user's minimum training floor and until
      validation no longer clearly improves; do not stop at the first lower
      loss.
- [x] Evaluate fix1 checkpoint `iter_000000100` under the strict same-length
      301-frame / 300-action contract. Strict artifact inspection passed for
      both two-sample overfit rows: predicted/reference videos were `301/301`
      frames, predicted/reference actions were `300x32/300x32`, mean future
      video PSNR was about `30.94` dB, mean robot-action future RMSE was about
      `0.32654`, and mean state-sidecar future RMSE was about `0.21112`.
      The agent opened both ref/pred contact sheets and the user explicitly
      confirmed the overfit videos look correct. Treat the two-sample overfit
      as passed; do not spend more time training overfit.
- [x] Do not spend more time on later two-sample overfit checkpoints unless a
      future regression requires it. Latest user direction is to treat overfit
      as successful and move the same 4-H200 allocation directly to full1000
      training.
- [x] Per latest user direction, stop the two-sample overfit run with tmux
      `Ctrl-C` while preserving the 4-H200 allocation. The old overfit srun
      step `123385.6` was terminated without using `scancel`; Slurm job
      `123385` remains allocated on `server32`.
- [x] Launch the full1000 fix1 SFT on the same 4-H200 allocation as Slurm step
      `123385.7`, tmux session `cosmos3_full_fix1_4gpu_20260610`, output root
      `sft_full_episode_wam_full1000_rgb_300step_fix1_4gpu_20260610_1131`.
      It reuses the accepted full-episode condition root
      `full_episode_wam_conditions_full1000_rgb_300step_20260609_203902`,
      uses `NPROC_PER_NODE=4` and `DATA_PARALLEL_SHARD_DEGREE=4`, and carries
      over the overfit-validated fix1 settings:
      action adapter optimizer keys, `lr=1e-4`, `warmup=10`, `f_min=0.5`,
      `grad_clip_norm=1.0`, `normalize_loss_by_active=true`,
      `independent_action_schedule=true`, `shift_action=1`, and
      `action_loss_weight=2.0`.
- [x] Full1000 fix1 startup sanity: the job loaded
      `Cosmos3-Nano-Policy-DROID-DCP`, entered startup validation, and reported
      iteration-0 validation loss `4.279975`. It then entered real training:
      iterations `1-4` logged finite total/action/vision losses and finite
      grad norms at about `17.4` seconds per step after startup. The inspected
      component split is action dominated while the video loss is already low,
      which matches the two-sample fix1 diagnosis instead of indicating a
      frozen action branch.
- [x] Continue full1000 fix1 past the 1-hour floor and through checkpoint
      validation/eval. Iter300 and iter600 checkpoints have both completed
      strict generated-video/action/readout evaluation and direct visual
      review. No controller/DP integration is allowed yet because the
      generated-RGB readout still predicts target-motion onset too early and
      peg-drop/regrasp handoff is not proven.
- [x] Acquire a separate tmux-held 1-H200 allocation for checkpoint eval/render
      without interrupting the active 4-H200 SFT. Aux job `123499`
      (`cosmos3_aux_eval_0610`) is running on `server32`, GPU visibility was
      confirmed with `nvidia-smi`, and it should be used for generated-video
      eval/render once a full1000 fix1 checkpoint is ready.
- [x] Launch an allocation-only watcher for full1000 fix1
      `iter_000000300` eval. Tmux session
      `cosmos3_full_fix1_iter300_eval_watch_20260610` runs inside aux Slurm
      job `123499` step `1`, watches the new full1000 fix1 checkpoint root,
      and will write
      `sft_full_episode_wam_full1000_rgb_300step_fix1_4gpu_20260610_1131/eval_full_episode_wam_iter_000000300`
      with `N_EVAL_SAMPLES=10` and `INFERENCE_NUM_STEPS=50` after the
      checkpoint files are stable.
- [x] Full1000 fix1 mid-run sanity at `iter_00000100`: validation loss dropped
      from `4.279975` at iteration `0` to `0.947078`, and recent rank-0 train
      losses around iterations `100-113` were roughly `0.18-0.34` with action
      component mostly `0.08-0.16`. This is strong training/load sanity only;
      it is not generated-video method evidence.
- [x] Full1000 fix1 passed the user's minimum 1-hour training floor while
      still running on the same 4-H200 allocation. Around iterations `170-185`,
      rank-0 loss was roughly `0.11-0.17`, action component roughly
      `0.04-0.07`, and grad norms stayed finite. Continue toward iter200
      validation and iter300 checkpoint/eval; do not stop on training loss.
- [x] Iteration-200 validation sanity: validation loss dropped further to
      `0.756682` from `0.947078` at iteration `100` and `4.279975` at
      iteration `0`. Recent rank-0 losses around iterations `201-213` were
      about `0.09-0.19`. The validation curve is still improving, so the run
      should continue; no generated-video evidence exists yet.
- [x] Full1000 fix1 checkpoint `iter_000000300` was saved at `2026-06-10
      13:17:25 CST` with finite training loss (`rank0 loss=0.1193`,
      `vision=0.0232`, `action=0.0481`). The aux watcher detected the
      checkpoint and is waiting for stable files before launching strict
      eval/render. The SFT run should continue beyond this checkpoint because
      validation was still improving at iter200.
- [x] Iteration-300 training validation loss was `0.787787`, slightly worse
      than iter200 `0.756682` after a large improvement from iter0/100. Treat
      this as a reason to continue through later checkpoints, not as a reason
      to stop immediately.
- [x] Iteration-300 generated artifact eval completed inside aux Slurm job
      `123499`. Strict artifact inspection passed for all 10 samples:
      generated/reference videos are `301/301` frames, actions are
      `300x32/300x32`, mean action RMSE is `0.3643254212`, mean robot-action
      future RMSE is `0.9349301391`, mean state-sidecar future RMSE is
      `0.0999715768`, and mean future-video PSNR is `21.2818860253`.
- [x] Iteration-300 visual review: the agent opened all 10 ref/pred review
      sheets. Visual geometry is much better than the previous failed full-data
      chain: no white/fog collapse, no transparent full-frame drift, and the
      robot/peg/target block remain visually coherent through the full
      301-frame rollout. Target-motion, static, and insert-resume sheets are
      visually close. Boundary: peg-drop recovery still does not prove a
      reliable regrasp/handoff behavior, and visual similarity alone is not
      controller evidence.
- [x] Iteration-300 generated-RGB task-state readout/profile completed with
      strict structure passing. Aggregate readout diagnostics: mean final hole
      error `0.0879152346` m, mean future hole RMSE `0.0369255092` m, mean
      future peg RMSE `0.0495181164` m, mean future TCP RMSE `0.0480951126` m,
      and mean future peg-head-hole RMSE `0.0430254817` m. This remains a
      controller-facing diagnostic failure/limitation, not a controller-ready
      world-model handoff.
- [x] Iteration-600 eval/readout watchers completed in aux allocation
      `123499`. Strict generated artifacts, generated-RGB readout/profile, and
      direct visual review are recorded below as progress but not
      controller-ready evidence.
- [x] Iteration-900 eval/readout watchers are now running in the same aux
      allocation `123499`: eval watcher step `123499.5` will write
      `eval_full_episode_wam_iter_000000900` after checkpoint
      `iter_000000900` appears, and readout/profile watcher step `123499.6`
      waits for that eval artifact before decoding generated RGB and
      profiling readout failure modes.

## Iteration-300 Post-SFT Diagnostic

- [x] Generate validation rollouts under the same full-episode/equal-length
      contract for checkpoint `iter_000000300`.
- [x] Prepare strict validation input JSONL under
      `sft_full_episode_wam_full1000_rgb_300step_20260609_204718/eval_full_episode_wam_latest/`.
      This is only an input manifest, not generated model evidence.
- [x] Decode/read out target, peg, TCP, grasp, insertion, and final pose for
      the `iter_000000300` generated videos using the step-250 reference-video
      readout snapshot. This is a diagnostic readout on generated RGB, not a
      replacement world model and not controller evidence.
- [x] Prepare generated-video task-state readout tooling:
      `scripts/world_model/run_cosmos3_full_episode_readout_eval.py` and
      `scripts/slurm/run_cosmos3_300f_task_state_readout_in_allocation.sh`.
      The inspector now accepts the existing full-episode `state_targets`
      JSON as the reference for target/peg/TCP/onset/final-pose metrics,
      preserving the rule that readout metrics are diagnostics on top of
      Cosmos3-generated RGB, not a replacement world model.
- [x] Produce action metrics over the intended action length for
      `iter_000000300`: all 10 eval samples have prediction/reference video
      length `301/301` and action prediction/reference shape `300x32/300x32`.
      Aggregate action RMSE is `0.6931767245`; aggregate future-video PSNR is
      `19.866555` dB.
- [x] Produce visual contact sheets/videos and inspect them before any success
      claim. Iteration-300 visual review is a negative diagnostic: early-prefix
      target-motion and peg-recovery samples develop severe white/transparent
      artifacts and lose reliable robot/peg/target geometry after roughly the
      middle of the rollout. Late/static samples look more stable, but this is
      not sufficient for the active target-motion WAM objective.
- [x] Preserve generated validation videos/contact sheets for user human
      review under
      `sft_full_episode_wam_full1000_rgb_300step_20260609_204718/eval_full_episode_wam_latest/`.
- [x] Do not start controller or DP integration from `iter_000000300`: strict
      length/action structure passed, but visual rollout and target-motion
      readout quality failed. Continue SFT to later validation points and
      re-run the same full-episode eval/readout gate when validation improves.
- [ ] For the next checkpoint eval, override `EVAL_ROOT` to a checkpoint-
      specific directory such as `eval_full_episode_wam_iter000000600`.
      Preserve `eval_full_episode_wam_latest` / iter300 artifacts as a negative
      diagnostic and do not overwrite already inspected videos or sheets.
- [x] Install and launch an allocation-only `iter_000000600` watcher:
      `scripts/slurm/watch_cosmos3_300f_checkpoint_eval_in_allocation.sh` is
      running inside Slurm job `122782` step `18`. It waits inside the compute
      allocation and will write `eval_full_episode_wam_iter_000000600` when the
      checkpoint appears.
- [x] User correction: poor `iter300` quality may also be undertraining, not
      only interface or method failure. A 4-H200 tmux-held allocation was
      acquired immediately as Slurm job `123131` on `server09`; no extra 8-GPU
      pending job was kept after the 4-GPU allocation started.
- [x] Prepare 4-GPU SFT support by parameterizing
      `run_cosmos3_300f_full_episode_wam_in_allocation.sh` with
      `NPROC_PER_NODE`, `DATA_PARALLEL_SHARD_DEGREE`,
      `DATA_PARALLEL_REPLICATE_DEGREE`, and
      `CONTEXT_PARALLEL_SHARD_DEGREE`.
- [x] A 4-GPU warm-start SFT was launched from the clean `iter_000000300`
      model checkpoint into
      `sft_full_episode_wam_full1000_rgb_300step_4gpu_warm_iter300_20260609_234018`.
      It reuses the strict full-episode condition root and must still train for
      at least 1 wall-clock GPU allocation hour before being considered
      training evidence.
- [x] The 4-GPU replacement passed startup/load sanity and entered real
      training: Slurm job `123131` loaded the clean `iter_000000300` model,
      completed startup validation with loss `7.292821`, and produced
      multi-rank training losses from iteration `1` onward. Iterations `2-8`
      were running at about `17.1` seconds/step. This satisfies the user's
      condition for stopping the duplicate 1-GPU SFT, but it is still not
      post-SFT world-model evidence until the 1-hour floor, checkpoint eval,
      strict artifact inspection, readout metrics, and visual review are done.
- [x] The 4-GPU warm-start SFT has now passed the 1-hour training floor:
      by `2026-06-10 00:42:30 CST`, Slurm job `123131` had allocation time
      `1:04:47`, SFT step time `1:02:12`, and was still training at iteration
      `179` with finite losses/grad norms. It still is not method evidence
      until checkpoint save/validation/eval/readout/visual review complete.
- [x] Latest 4-GPU monitor at `2026-06-10 00:49:16 CST`: job `123131` was
      still running at iteration `203`, about `17.1` seconds/step, with losses
      around `2.8-4.8`, finite gradient norms, and normal multi-GPU memory use.
      The 4-GPU `iter_000000300` checkpoint has not been saved yet, so Slurm
      step `122782.26` is still correctly waiting.
- [x] Latest 4-GPU monitor at `2026-06-10 00:56:24 CST`: job `123131` was
      still running at iteration `228`, still about `17.1` seconds/step, with
      finite gradient norms and recent losses mostly around `3.0-3.8`.
      No 4-GPU checkpoint directory exists yet, so the eval watcher remains
      correctly in `checkpoint_not_ready`.
- [x] After the 4-GPU run had real training losses, the 1-GPU SFT step
      `122736.6` was stopped intentionally after saving checkpoint
      `iter_000000600`. The parent allocation was initially kept for support
      work, then later expired/revoked; the active support allocation is now
      `123366`, not `122736`.
- [x] The allocation-only watcher in job `122782` detected
      `iter_000000600` and completed checkpoint-specific eval under
      `eval_full_episode_wam_iter_000000600`. Strict structure passed:
      `strict_eval_artifacts_ok=true`, 10/10 samples, generated/reference
      videos `301/301` frames, generated/reference actions `300x32/300x32`,
      mean action RMSE `0.6496965469564411`, and mean future-video PSNR
      `19.860613478278417`.
- [x] Visual review of `iter_000000600` is still negative for controller use.
      It is visibly cleaner than `iter_000000300` and no longer has the same
      full-frame fog collapse on the inspected static/target-motion examples,
      but target-motion and insert-resume sheets still show peg/hand geometry
      drift, and peg-drop recovery still fails to regrasp the peg while
      producing semi-transparent hand/object artifacts after about frame `109`.
- [x] A fresh visual re-check on `2026-06-10` confirmed the `iter_000000600`
      boundary: the source RGB sheet is readable and full-length, the static
      prediction is comparatively stable but still drifts late, the moving
      target prediction loses reliable peg/hand/contact geometry, and the
      peg-drop recovery prediction still fails to regrasp while producing
      semi-transparent artifacts. This keeps `iter_000000600` negative.
- [x] Generated-RGB readout eval for `iter_000000600` was run inside the freed
      1-H200 allocation `122736`, not on the login node. It passed strict
      readout structure but failed target-motion evidence: mean final
      hole-position error is `0.1153032929` m; moving-target onset is predicted
      at frames `6-13` while GT onset is frames `84-94`, and static/peg
      disturbance samples also get false target-motion onsets at frames
      `6-19`.
- [ ] Do not start controller or DP integration from `iter_000000600`. It is
      a useful training-progress diagnostic only: structure and visual
      stability improved, but target-motion timing, final target pose, peg
      recovery, and executable contact geometry remain unreliable.
- [x] A 4-GPU checkpoint watcher was launched in the eval allocation as Slurm
      step `122782.26`. It waited for the 4-GPU warm-start checkpoint
      `iter_000000300` and will run the same strict full-episode eval under
      `sft_full_episode_wam_full1000_rgb_300step_4gpu_warm_iter300_20260609_234018/eval_full_episode_wam_iter_000000300`.
- [x] The checkpoint watcher was repaired before the 4-GPU checkpoint arrived:
      it now requires `model/.metadata` and a stable file-size signature before
      launching eval, rather than allowing a partially written DCP checkpoint
      when `model/` first appears. The old watcher step `122782.26` was
      canceled, syntax was checked inside Slurm allocation `122782`, and
      updated watcher step `122782.28` completed the 4-GPU `iter_000000300`
      eval.
- [x] Added allocation-only generated-RGB readout watcher
      `scripts/slurm/watch_cosmos3_eval_readout_in_allocation.sh`. It refuses
      login-node execution, waits for a strict eval root, then runs
      `run_cosmos3_full_episode_readout_eval.py` with the current reference
      readout checkpoint.
- [x] A 4-GPU readout watcher ran in the freed 1-H200 allocation as Slurm step
      `122736.92`. It waited for the 4-GPU
      `eval_full_episode_wam_iter_000000300/eval_artifact_inspection.json`,
      then wrote generated-RGB readout diagnostics under that eval root.
- [x] The 4-GPU warm-start `iter_000000300` checkpoint passed strict artifact
      inspection under the full-episode contract: 10/10 samples, generated/
      reference videos `301/301`, generated/reference actions `300x32/300x32`,
      mean action RMSE `0.6329251997`, and mean future-video PSNR
      `19.8388587420`.
- [x] Visual review of 4-GPU `iter_000000300` is negative for controller use.
      Target-motion and insert-resume samples still drift in peg/hand/contact
      geometry, the peg-drop sample does not recover/regrasp the peg, and the
      `hole_constant` target-pre-motion sample develops severe transparent
      white artifacts in the middle/later rollout.
- [x] Generated-RGB readout for 4-GPU `iter_000000300` passed strict structure
      but failed the target-motion evidence gate. Mean final hole-position
      error is `0.1043972932` m. Moving-target onsets are predicted at frames
      `4-7` while GT onset is frames `84-94`, and static/peg-only scenarios
      also get false target-motion onsets at frames `8-10`. Peg-drop and
      peg-disturb future grasp accuracies are `0.417910` and `0.000000`.
- [ ] Do not start controller or DP integration from 4-GPU `iter_000000300`.
      It improves action RMSE over the 1-GPU diagnostics but still fails
      target-motion timing, final target pose, peg recovery, and contact-
      preserving visual rollout.
- [x] Following the user's instruction, the 1-GPU SFT remains stopped. The
      old 1-H200 allocation `122736` later expired/revoked; the replacement
      1-H200 allocation `123366` is kept only for readout, sanity checks, and
      video/render tasks.
- [x] A new 4-GPU `iter_000000600` strict-eval watcher was launched inside the
      eval allocation as Slurm step `122782.35`. It waits for the complete DCP
      checkpoint and writes
      `eval_full_episode_wam_iter_000000600`.
- [x] A paired generated-RGB readout watcher for 4-GPU `iter_000000600` was
      launched inside the kept 1-H200 allocation as Slurm step `122736.93`.
      It waits for strict eval artifacts and then writes
      `eval_full_episode_wam_iter_000000600/task_state_readout_best_current`.
- [x] Per the user's resource instruction, the kept 1-H200 allocation `122736`
      is not running SFT while the 4-GPU experiment is alive. A matching
      readout-failure-profile watcher was syntax-checked inside `122736` and
      launched as Slurm step `122736.104`. It waits for the 4-GPU
      `iter_000000600` readout summary and will write
      `eval_full_episode_wam_iter_000000600/task_state_readout_best_current/readout_failure_profile.{json,md}`.
- [x] Latest 4-GPU monitor at `2026-06-10 01:45:30 CST`: job `123131` was
      still training at iteration `385`, about `17.1` seconds/step, with
      finite recent losses mostly around `2.5-3.8` and finite gradient norms.
      The `iter_000000600` eval watcher `122782.35`, readout watcher
      `122736.93`, and profile watcher `122736.104` are correctly waiting for
      the next checkpoint/eval/readout artifacts.
- [x] The 4-GPU `iter_000000600` checkpoint was saved at
      `2026-06-10 02:46:59 CST`; watcher `122782.35` waited for stable DCP
      files and completed strict full-episode eval under
      `eval_full_episode_wam_iter_000000600`. Strict artifact inspection
      passed: 10/10 samples, generated/reference videos `301/301`, generated/
      reference actions `300x32/300x32`, mean action RMSE `0.6199095227`, and
      mean future-video PSNR `20.1762433861` dB.
- [x] Generated-RGB readout and failure profiling for 4-GPU `iter_000000600`
      completed inside the kept 1-H200 allocation `122736`. Strict readout
      structure passed, but the readout evidence failed: mean final hole error
      is `0.1023905702` m; mean future hole/peg/TCP RMSE are
      `0.0475034488` / `0.0674232871` / `0.0660298159` m; dynamic target
      onsets are still predicted tens of frames too early, and static/peg-only
      scenes still show false target drift at multiple thresholds.
- [x] Visual review of 4-GPU `iter_000000600` is still negative for
      controller/DP handoff. Compared with `iter_000000300`, many sheets are
      cleaner and less foggy, but target-motion and target-post-motion samples
      still lose contact geometry, peg-drop and peg-disturb recovery do not
      produce a valid regrasp/recovery path, one peg-disturb sample develops a
      large cloud-like artifact over the target, and insert-resume samples
      lose or mis-handle the peg. This is progress, not method success.
- [ ] Do not start controller or DP integration from 4-GPU `iter_000000600`.
      It improved action RMSE and PSNR over `iter_000000300`, but it still
      fails target-motion timing, final target pose readout, peg recovery, and
      executable peg/hand continuity.
- [x] The next 4-GPU checkpoint gate was launched after `iter_000000600`
      review: strict eval watcher `122782.36` waits for `iter_000000900`.
      The generated-RGB readout watcher `122736.105` and
      readout-failure-profile watcher `122736.106` were later cancelled when
      the old 1-H200 allocation `122736` was revoked. This is scheduling/
      allocation expiration, not model evidence. Relaunch readout/profile after
      `iter_000000900` eval artifacts exist, using an available compute-node
      allocation only.
- [x] Added and ran a compute-node-only framework contract probe:
      `scripts/world_model/probe_cosmos3_full_episode_inference_contract.py`.
      It was executed through Slurm allocation `122736`, not on the login
      node, for representative 4-GPU `iter_000000300` target-motion and static
      samples. The probe confirms Cosmos inference sees full `301` video
      frames, `300` action steps, raw action dim `32` padded to `64`, and the
      expected per-sample prefix masks: e.g. frame-92 target-motion has vision
      latent indexes `0-23` and action indexes `0-91`, while frame-80 static
      monitor has vision latent indexes `0-20` and action indexes `0-79`.
      Therefore the observed early target-motion/readout failure is not caused
      by a 128/129/93-frame truncation, fixed 8-latent defaulting, or missing
      framework-facing action/history masks.
- [x] Added and ran a generated-RGB readout failure profiler:
      `scripts/world_model/profile_cosmos3_readout_failure_modes.py`. It was
      run inside Slurm allocation `122736` for 4-GPU `iter_000000300` and
      wrote
      `eval_full_episode_wam_iter_000000300/task_state_readout_best_current/readout_failure_profile.{json,md}`.
      The profile is diagnostic only, but it shows the failure is not just a
      `0.002` m onset-threshold artifact: mean future hole RMSE is
      `0.0519564251` m, mean future peg RMSE is `0.0741150761` m, mean future
      TCP RMSE is `0.0719546468` m, and static/peg-only scenes still show
      `0.02` m-scale false target drift in several samples. Continue training
      and compare the same profile at `iter_000000600`.
- [x] The old 1-GPU SFT step `122736.6` was stopped after the 4-GPU experiment
      was healthy. Allocation `122736` later expired and was revoked, so it is
      no longer available for support work. The stale tmux session was removed.
      A replacement tmux-held 1-H200 support allocation `123366` is running on
      `server21` for readout, sanity checks, and video/render tasks only. The
      iter900 generated-RGB readout/profile watchers were relaunched as
      compute steps `123366.1` and `123366.2`.
- [x] The 4-GPU `iter_000000900` checkpoint was saved at
      `2026-06-10 04:16:54 CST`; watcher `122782.36` completed strict
      full-episode eval under `eval_full_episode_wam_iter_000000900`.
      Strict artifacts passed: 10/10 samples, generated/reference videos
      `301/301`, generated/reference actions `300x32/300x32`, mean action
      RMSE `0.6172093662`, and mean future-video PSNR `20.2230188270` dB.
- [x] Generated-RGB readout/profile for 4-GPU `iter_000000900` completed
      inside auxiliary 1-H200 allocation `123366`. Strict readout/profile
      structure passed, but the evidence is still negative: mean final hole
      error `0.1021599918` m; mean future hole/peg/TCP RMSE
      `0.0477525963` / `0.0668954306` / `0.0657577841` m; mean future
      peg-head-hole RMSE `0.0519184658` m. Dynamic target onsets remain tens
      of frames too early, and static/peg-only samples still show false target
      drift.
- [x] Visual review of 4-GPU `iter_000000900` is still negative for
      controller/DP handoff. Target-motion and target-post-motion sheets are
      cleaner than early checkpoints but still drift in robot/peg/target
      contact geometry; insert-resume does not prove stable peg holding;
      peg-drop does not regrasp the peg; peg-disturb still has a large
      cloud-like artifact over the target. Static-late is comparatively
      stable but cannot support the dynamic handoff claim.
- [ ] Do not start controller or DP integration from 4-GPU `iter_000000900`.
      It slightly improves action RMSE/PSNR over `iter_000000600`, but still
      fails target-motion timing, final target pose readout, peg recovery, and
      executable peg/hand continuity.
- [x] The next 4-GPU checkpoint gate was launched after the `iter_000000900`
      negative review: strict eval watcher `122782.37` waits for
      `iter_000001200`, generated-RGB readout watcher `123366.3` waits for
      that eval root, and readout-failure-profile watcher `123366.4` waits for
      the `iter_000001200` readout summary.
- [x] Latest 4-GPU monitor at `2026-06-10 04:34:53 CST`: job `123131` was
      still training at iteration `948`, about `17.1` seconds/step, with
      finite recent losses mostly around `2.3-4.1` and finite gradient norms.
      Checkpoints present are `iter_000000300`, `iter_000000600`, and
      `iter_000000900`; `iter_000001200` is not saved yet, so watcher
      `122782.37` and support watchers `123366.3` / `123366.4` are correctly
      waiting. No 1-GPU SFT step is running.
- [x] Training config check: the active 4-GPU run has `trainer.max_iter=1500`,
      `checkpoint.save_iter=300`, `trainer.validation_iter=300`, and
      `trainer.max_val_iter=40`. Validation loss moved from `7.292821`
      at warm-start iteration `0`, to `6.597002` at `300`, to `6.005106` at
      `600`, then back up to `6.178823` at `900`. This reinforces that loss
      alone is not method evidence and that the active gate should continue
      through generated-video/action/readout/visual checks at `iter_000001200`
      and final `iter_000001500`.
- [x] Loss/error sanity at `2026-06-10 04:38 CST`: the training log has no
      Traceback, OOM, NaN, or runtime-error evidence. The only `nan` text hit
      is the configured `skip_nan_step` callback. Single-step large finite
      losses appear occasionally, including `iter 885` loss `49.2275` and
      `iter 951` loss `49.5979`, but both are followed immediately by normal
      `~2.7-3.5` losses with finite gradient norms. Treat these as finite
      diffusion/action timestep spikes, not a reason to stop the 4-GPU SFT.
- [x] Latest 4-GPU monitor at `2026-06-10 04:41:52 CST`: job `123131` was
      still training at iteration `972`, about `17.1` seconds/step, with
      finite losses and gradient norms. `iter_000001200` is not saved yet;
      eval watcher `122782.37` and support watchers `123366.3` / `123366.4`
      are correctly waiting. No 1-GPU SFT step is running.
- [x] Latest 4-GPU monitor at `2026-06-10 04:43:53 CST`: job `123131` was
      still training at iteration `980`, about `17.1` seconds/step. Recent
      losses are again in the normal finite range (`2.4-3.5`), and no
      `iter_000001200` checkpoint directory exists yet. Watchers `122782.37`,
      `123366.3`, and `123366.4` remain in their expected waiting states.
- [x] Resource/timing check at `2026-06-10 04:45 CST`: eval allocation
      `122782` runs until `2026-06-10 10:22:09 CST`, auxiliary 1-H200
      allocation `123366` runs until `2026-06-11 04:08:09 CST`, and 4-GPU SFT
      allocation `123131` runs until `2026-06-10 23:37:43 CST`. The
      checkpoint watcher timeout is `7200` seconds, readout watcher timeout is
      `10800` seconds, and profile watcher timeout is `14400` seconds. At
      about `iter 984`, the run should still have enough held resource time
      for `iter_000001200` and final `iter_000001500` gates without restarting
      1-GPU SFT.
- [x] Latest 4-GPU monitor at `2026-06-10 04:47:48 CST`: job `123131` was
      still training at iteration `993`, about `17.1` seconds/step, with
      finite recent losses around `2.8-3.4`. No `iter_000001200` checkpoint
      or eval artifacts exist yet. Watchers `122782.37`, `123366.3`, and
      `123366.4` remain in the expected waiting state.
- [x] Latest 4-GPU monitor at `2026-06-10 04:48:51 CST`: job `123131` was
      still training at iteration `997`, about `17.1` seconds/step, with
      finite recent losses around `2.7-3.5`. `iter_000001200` is still not
      saved, and no eval/readout/profile artifacts exist yet. Existing
      watchers remain correct; continue waiting rather than changing training
      or starting controller/DP work.
- [x] Resource boundary confirmed at `2026-06-10 04:50:59 CST`: the healthy
      4-GPU SFT job `123131` is the only active SFT line. There is no running
      1-GPU SFT step or old 1-GPU SFT tmux session. The single-GPU allocations
      now present are support resources only: eval watcher `122782.37` and
      auxiliary readout/profile watchers `123366.3` / `123366.4`. Do not
      restart 1-GPU SFT while the 4-GPU experiment is healthy; keep 1-H200
      time for strict eval, generated-RGB readout, sanity checks, and video or
      render support.
- [x] Latest 4-GPU monitor at `2026-06-10 04:52:48 CST`: Slurm job `123131`
      was still running at iteration `1011`, about `17.1` seconds/step, with
      finite recent losses around `2.6-3.8`. Checkpoints present are still
      `iter_000000300`, `iter_000000600`, and `iter_000000900`; no
      `iter_000001200` checkpoint/eval/readout/profile artifacts exist yet.
      Existing watcher steps `122782.37`, `123366.3`, and `123366.4` are in
      the expected waiting state. Do not launch overlapping `iter_000001500`
      watchers until the `iter_000001200` gate releases the single-GPU support
      resources.
- [x] Latest 4-GPU monitor at `2026-06-10 05:25:48 CST`: Slurm job `123131`
      was still training at iteration `1126`. The `iter_000001200` checkpoint
      still was not present, and watcher `122782.37` remained in
      `checkpoint_not_ready`. A single-step loss spike occurred at iteration
      `1121` (`Loss: 73.6864`), but iterations `1122-1126` immediately
      returned to finite normal losses around `2.6-3.1` with finite gradient
      norms. No NaN/OOM/Traceback evidence was found. Continue to the strict
      `iter_000001200` gate rather than intervening in training.
- [x] The 4-GPU `iter_000001200` gate completed under the same 301-frame /
      300-action contract. Strict artifacts passed: 10/10 samples,
      generated/reference videos `301/301`, generated/reference actions
      `300x32/300x32`, mean action RMSE `0.6162060709`, and mean future-video
      PSNR `20.2295638395` dB.
- [x] Generated-RGB readout/profile for 4-GPU `iter_000001200` completed
      inside auxiliary 1-H200 allocation `123366`. Strict readout/profile
      structure passed, but evidence remains negative: mean final hole error
      `0.1028519175` m; mean future hole/peg/TCP RMSE
      `0.0481315461` / `0.0667989680` / `0.0662642823` m; mean future
      peg-head-hole RMSE `0.0515516010` m. Target motion is still predicted
      tens of frames too early, and static/peg-only cases still show false
      target drift.
- [x] Visual review of all 10 `iter_000001200` sheets is negative for
      controller/DP handoff. Target-motion and insert-resume samples still
      drift in peg/hand/contact geometry; target-post-motion/reverse develops
      semi-transparent robot/hand artifacts; peg-drop does not regrasp and
      leaves the peg on the table; peg-disturb produces a large noisy cloud
      over the target; static-late is cleaner but does not prove dynamic
      target or recovery competence.
- [ ] Do not start controller or DP integration from 4-GPU `iter_000001200`.
      It is marginal metric progress over earlier checkpoints, not a valid
      world-model handoff checkpoint.
- [x] The final `iter_000001500` gate was launched after the `iter_000001200`
      review released the single-GPU support resources. Strict eval watcher
      `122782.38` waits/runs in the eval allocation; generated-RGB readout
      watcher `123366.6` and readout-failure-profile watcher `123366.5`
      wait/run in the auxiliary 1-H200 allocation. The `iter_000001500`
      checkpoint was saved at `2026-06-10 07:16:51 CST`; watcher `122782.38`
      is now waiting for stable checkpoint files before eval. Use the same
      strict 301-frame / 300-action eval, generated-RGB readout, profile, and
      visual review before any controller/DP decision.
- [x] Reference-video readout training completed with `metrics_best.json` at
      step `2000`: future hole RMSE `0.0499867350` m, future peg RMSE
      `0.0441538133` m, future TCP RMSE `0.0378160141` m, future
      peg-head-hole RMSE `0.0770236030` m, future grasp accuracy
      `0.9067095518`, and future insertion accuracy `0.8026654124`. Keep
      treating readout as a generated-RGB diagnostic, not a replacement world
      model or controller result.
- [x] The 4-GPU SFT reached the configured final iteration and stopped
      normally. The final validation loss at `iter_000001500` was `6.158292`;
      the validation series was `7.292821` at warm-start iteration `0`,
      `6.597002` at `300`, `6.005106` at `600`, `6.178823` at `900`,
      `6.284197` at `1200`, and `6.158292` at `1500`. Loss improved strongly
      early but did not translate into a passing generated-video/readout/visual
      gate, so loss remains a training diagnostic only.
- [x] Historical pre-fix1 final `iter_000001500` strict eval completed under the same
      full-episode contract: 10/10 samples, generated/reference videos
      `301/301`, generated/reference actions `300x32/300x32`, mean action
      RMSE `0.6161383192`, and mean future-video PSNR `20.2640810606` dB.
- [x] Historical pre-fix1 generated-RGB readout/profile for 4-GPU `iter_000001500` completed
      inside the auxiliary 1-H200 allocation `123366`. Strict readout/profile
      structure passed, but the evidence is still negative: mean final hole
      error `0.1028195255` m; mean future hole/peg/TCP RMSE
      `0.0485962523` / `0.0669838827` / `0.0666590482` m; mean future
      peg-head-hole RMSE `0.0516100994` m. Dynamic target onsets are still
      predicted far too early, and static/peg-only samples still show false
      target drift.
- [x] Historical pre-fix1 visual review of all 10 `iter_000001500` sheets is negative for
      controller/DP handoff. Target-motion and insert-resume samples still
      drift in peg/hand/contact geometry; peg-drop does not regrasp the peg
      and leaves it on the table; peg-disturb creates a large cloud-like
      artifact over the target; static-late is comparatively stable but does
      not prove dynamic rebinding. This checkpoint cannot produce executable
      action chunks or a reliable future task state for DP resume.
- [x] After the final gate completed, idle Slurm allocations `123131`
      (4-GPU SFT) and `122782` (extra eval allocation) were released. The old
      auxiliary allocation `123366` was later revoked after all watcher steps
      completed, so a replacement tmux-held 1-H200 allocation `123381` is now
      running on `server21` for later sanity checks, generated-video/readout
      inspection, or render tasks. No 1-GPU SFT is running.
- [x] Latest resource boundary: because the 4-GPU full-episode SFT ran through
      to strict eval, do not restart or continue a 1-GPU SFT fallback. Keep the
      held 1-H200 allocation for short sanity checks, readout/debug steps, and
      video/render work needed by the next reviewed repair.
- [ ] Do not start controller or DP integration from 4-GPU `iter_000001500`.
      It passed strict length/action accounting but failed target-motion
      timing, final target pose readout, peg recovery, and executable
      hand/peg continuity.
- [x] Added a reference-RGB readout calibration path:
      `scripts/world_model/build_cosmos3_reference_rgb_eval_root.py` and
      `scripts/slurm/run_cosmos3_reference_rgb_readout_calibration_in_allocation.sh`.
      It was run inside held 1-H200 allocation `123381` on the same final
      10-sample eval panel, replacing only the readout input videos with GT
      reference RGB symlinks. This is calibration evidence only, not Cosmos3
      world-model success.
- [x] Reference-RGB calibration for `iter_000001500` completed with strict
      structure and all 10 samples. It shows the readout floor is nontrivial:
      mean final hole error `0.0456793996` m, mean future hole/peg/TCP RMSE
      `0.0277646941` / `0.0248001007` / `0.0209654087` m, and mean future
      peg-head-hole RMSE `0.0359646693` m. Generated-RGB remains worse
      (`0.1028195255` final hole error and `0.0485962523` /
      `0.0669838827` / `0.0666590482` future hole/peg/TCP RMSE), so the world
      model rollout quality is still a real failure even after accounting for
      readout error.
- [x] Added and ran target-onset readout scoring:
      `scripts/world_model/score_cosmos3_target_onset_readout.py`. The 2 mm
      displacement score produces `5/5` static false-positive samples and
      about `82` frames mean moving-onset error on both generated RGB and
      reference RGB. This means the current readout-displacement onset metric
      is too sensitive for mode switching; future controller-facing target
      detection needs a calibrated onset/motion head or score.
- [x] Trained/evaluated a calibrated target-motion head from RGB-derived
      readout trajectories inside Slurm allocation `123381`. The calibration
      root is
      `experiments/world_model_task_rebinding/cosmos3/target_motion_readout_calibration_20260610_0751`.
      The run used balanced `120` train reference-RGB episodes and all `88`
      val reference-RGB episodes available from the full-episode condition
      root. Metrics were saved under
      `target_motion_head/target_motion_head_metrics.{json,md}` and the small
      diagnostic checkpoint under `target_motion_head/target_motion_head.pt`.
- [x] The calibrated target-motion head improves held-out reference-RGB target
      monitoring but does not rescue the final generated-RGB world-model
      checkpoint in the historical pre-fix1 chain. Held-out reference RGB:
      AUROC `0.9115353604`, F1@0.5 `0.7669897596`, best F1 `0.7788987602`.
      Historical pre-fix1 iter1500 generated RGB:
      AUROC `0.7756010330`, F1@0.5 `0.6111111111`, best F1
      `0.6428281187`. This keeps the diagnosis focused: target switching
      needs a calibrated head, and Cosmos3 rollout quality still has to improve
      before controller/DP integration.

## 2026-06-10 Fresh Full1000 Fix1 Continuation

- [x] The user confirmed the two-sample overfit videos are acceptable, so the
      overfit diagnostic is closed. The overfit srun was stopped by tmux
      `Ctrl-C`, not `scancel`, preserving the 4-H200 allocation.
- [x] Fresh full1000 fix1 SFT is running in the preserved allocation
      `123385` on `server32`, tmux `cosmos3_full_fix1_4gpu_20260610`, Slurm
      step `123385.7`, output root
      `sft_full_episode_wam_full1000_rgb_300step_fix1_4gpu_20260610_1131`.
      This is the active run after the overfit success; older negative
      iter1500 notes above are historical evidence from a previous chain, not
      a reason to stop the current fix1 run.
- [x] First-principles reason for continuing: the overfit pass shows the
      trainable action/video interface can memorize the strict 301-frame /
      300-action contract. The physical question now is whether full1000 SFT
      learns target-motion response, peg/hand continuity, and executable
      action/state rollouts on diverse episodes, so evidence must come from
      full-data validation videos, action metrics, and generated-RGB readout.
- [x] Current validation curve through iter700: iter0 `4.279975`, iter100
      `0.947078`, iter200 `0.756682`, iter300 `0.787787`, iter400
      `0.733358`, iter500 `0.746084`, iter600 `0.718841`, and iter700
      `0.725782`. Iter700 is slightly worse than the iter600 best but still
      better than iter400/iter500, so it is not a stopping signal before the
      iter900 full generated-video/readout/visual gate.
- [x] Iter800 validation loss was `0.641040`, a clear new best over iter600
      `0.718841`. This confirms the full1000 fix1 run is still improving by
      validation loss, so it must continue to the iter900 strict
      generated-video/action/readout/visual gate and should not be stopped
      because iter700 temporarily rebounded.
- [x] Iter900 checkpoint saved at `2026-06-10 16:36:08 CST`, and iter900
      validation loss was `0.634057`, again a new best over iter800
      `0.641040`. The iter900 aux eval watcher detected the checkpoint,
      waited for stable files, passed strict eval input construction for 10
      diverse samples under the `301` RGB frame / `300x32` action contract,
      and has started inference. Continue training to later gates while
      iter900 generated eval/readout/visual evidence is inspected.
- [x] Iter500 validation loss was `0.746084`, a small rebound from iter400
      `0.733358` but still better than iter200/iter300. The training step
      immediately resumed with low loss at iter501. Treat this as a single
      validation fluctuation, not a stopping criterion; continue to iter600
      strict artifact/readout/video gate.
- [x] Checkpoint `iter_000000600` saved at `2026-06-10 14:56:48 CST` with
      rank-0 checkpoint-step loss `0.0940` (`vision=0.0194`,
      `action=0.0373`). Iter600 validation loss was `0.718841`, the best
      validation point so far. The run resumed at iter601 with low finite
      loss, so full1000 fix1 training should continue.
- [x] The iter600 aux eval watcher detected the checkpoint, waited for stable
      files, and passed strict eval input construction for 10 diverse samples
      under the `301` RGB frame / `300x32` action contract. Generated eval is
      now running in aux allocation `123499`; readout/profile remains waiting
      for `eval_artifact_inspection.json`.
- [x] Iter600 strict generated artifact inspection passed for all 10 samples:
      generated/reference videos `301/301`, actions `300x32/300x32`, mean
      action RMSE `0.3673552634`, mean robot-action future RMSE
      `0.9428412691`, mean state-sidecar future RMSE `0.0746684971`, and
      mean future-video PSNR `22.6652227928`. This improves over iter300 video
      PSNR `21.2818860253` and state-sidecar RMSE `0.0999715768`, while
      robot-action RMSE remains high.
- [x] Iter600 visual review completed for all 10 review sheets. Static,
      target-motion, target-post-motion, and most insert-resume panels show
      close ref/pred robot, target block, hole, and peg geometry without the
      old transparent/white-collapse failure. Peg disturbance/recovery is also
      visually cleaner than the old failed chain. Boundary: peg-drop and some
      insert-resume samples still do not prove executable regrasp or DP-ready
      handoff by visual inspection alone; generated-RGB readout/profile must
      decide whether the target/peg/TCP state is controller-facing.
- [x] Iter600 generated-RGB readout/profile completed with strict structure
      passing for all 10 samples. Aggregate diagnostics improved over iter300:
      mean final hole error `0.0613948691` m, mean future hole RMSE
      `0.0328006673` m, mean future peg RMSE `0.0435265400` m, mean future
      TCP RMSE `0.0433667297` m, and mean future peg-head-hole RMSE
      `0.0449233321` m. This is progress but not controller-ready evidence:
      threshold target-motion onset is still predicted far too early on moving
      samples and still fires on static/peg-only samples.
- [x] Because iter600 validation is the best so far and readout/visuals are
      improving but target-onset/controller readiness is still not solved,
      continue training and use the already-launched next strict generated
      eval/readout gate for `iter_000000900` in aux allocation `123499`.
- [x] Iter300 strict eval/readout was completed and inspected. Artifacts passed
      the full-episode contract for all 10 samples (`301/301` RGB frames,
      `300x32/300x32` actions), visual geometry is much healthier than the old
      failed full-data chain, but generated-RGB readout is still not
      controller-ready: mean final hole error `0.0879152346` m and early
      target-motion false onsets remain controller-facing limitations.
- [x] Iter900 strict generated artifact inspection completed for all 10
      samples under the same full-episode contract: generated/reference videos
      `301/301`, actions `300x32/300x32`, mean action RMSE
      `0.2946495338`, mean robot-action future RMSE `0.7637177886`, mean
      state-sidecar future RMSE `0.0684030772`, and mean future-video PSNR
      `22.6316859914`. This is a clear improvement over iter600 on action,
      robot-action, and state-sidecar metrics, while video PSNR is essentially
      flat.
- [x] Iter900 visual review completed for all 10 review sheets. The videos
      remain coherent and do not show the old white/fog/transparent collapse.
      Static and insert-resume panels are stable, target-motion panels are
      visually close to GT, and peg-disturb is cleaner. Peg-drop/regrasp is
      still the main unresolved behavior and does not yet prove executable
      handoff.
- [x] Iter900 generated-RGB readout/profile completed with strict structure:
      mean final hole error `0.0678502409` m, mean future hole RMSE
      `0.0335725300` m, mean future peg RMSE `0.0417441050` m, mean future
      TCP RMSE `0.0375097505` m, and mean future peg-head-hole RMSE
      `0.0416219164` m. Compared with iter600, peg/TCP/action-side diagnostics
      improved, but final/future hole readout slightly regressed and target
      onset still fires early or falsely on no-target-motion samples.
- [ ] Continue full1000 fix1 SFT on the preserved 4-H200 allocation through
      the final configured checkpoint while validation/generated evidence is
      still being collected. Do not resume overfit, do not start controller/DP
      integration from iter900/iter1200, and use the final checkpoint gate for
      strict generated-video/action/readout/visual review.
- [x] Iter1200 gate completed in aux allocation `123499`: tmux
      `cosmos3_full_fix1_iter1200_eval_watch_20260610` ran Slurm step
      `123499.8`, and
      `cosmos3_full_fix1_iter1200_readout_profile_20260610` ran Slurm step
      `123499.7`. Both executed inside the compute-node allocation; neither
      changed the active SFT run.
- [x] Iter1000 validation loss was `0.697886`, worse than the iter900 best
      `0.634057`. Treat this as a validation rebound, not a stop signal,
      because iter1000 is not a saved checkpoint/eval gate and the run resumed
      normally with finite losses at iter1001/1002. Continue to iter1200
      strict generated artifact/readout/visual review.
- [x] Iter1100 validation loss was `0.607350`, a new best for the fresh
      full1000 fix1 run. The run resumed normally at iter1101/1102 with finite
      low losses, so continue training to the iter1200 checkpoint and strict
      generated-video/action/readout/visual gate.
- [x] Checkpoint `iter_000001200` saved at `2026-06-10 18:15:30 CST`.
      Iter1200 validation loss was `0.675294`, a rebound from the iter1100
      best. The training step resumed at iter1201 with finite low loss, so the
      run should continue while iter1200 strict generated eval is inspected.
- [x] Iter1200 strict generated artifact inspection passed for all 10 samples
      under the full-episode contract: generated/reference videos `301/301`,
      actions `300x32/300x32`, mean action RMSE `0.3327156417`, mean
      robot-action future RMSE `0.8618907214`, mean state-sidecar future RMSE
      `0.0662339055`, and mean future-video PSNR `22.7099074529`.
- [x] Iter1200 visual review completed for all 10 review sheets. The videos
      remain coherent and materially healthier than the old failed chain, with
      no global white/fog/transparent collapse. Peg-drop/regrasp remains the
      main unresolved visible behavior and is not executable handoff evidence.
- [x] Iter1200 generated-RGB readout/profile completed with strict structure:
      mean final hole error `0.0620453945` m, mean future hole RMSE
      `0.0323648744` m, mean future peg RMSE `0.0414618213` m, mean future
      TCP RMSE `0.0359076100` m, and mean future peg-head-hole RMSE
      `0.0419642333` m. Target-motion onset still fires too early or falsely
      on no-target-motion samples, so iter1200 is not controller/DP evidence.
- [x] Iter1300 validation loss was `0.694085`, worse than the iter1100 best
      `0.607350` and slightly worse than iter1200 `0.675294`. The run resumed
      normally at iter1301/1302 with finite low loss, so this is recorded as a
      validation rebound rather than a crash or stop condition.
- [x] Iter1400 validation loss was `0.625498`, better than iter1200/iter1300
      but still worse than the iter1100 best `0.607350`. The run resumed
      normally at iter1401 with finite low loss, so continue to the final
      checkpoint/eval gate rather than making a validation-only decision.
- [x] Iter1500 checkpoint saved at `2026-06-10 19:54:51 CST`; final validation
      loss was `0.662956`, worse than the iter1100 best `0.607350`, and the
      trainer reported done. The final checkpoint is therefore an evaluation
      gate, not a validation-only success claim.
- [x] Iter1500 final gate completed in aux allocation `123499`: strict
      artifact inspection passed all 10 samples with `301/301` videos and
      `300x32/300x32` actions; mean action RMSE `0.3157882582`, mean
      robot-action future RMSE `0.8203911022`, mean state-sidecar future RMSE
      `0.0626562464`, and mean future-video PSNR `23.1641813684`.
- [x] Iter1500 readout/profile and visual review completed. Mean final hole
      error was `0.0538804681` m and future hole/peg/TCP RMSE were
      `0.0301422454` / `0.0386116191` / `0.0347898968` m, but target-motion
      onset remains early/false-positive and peg/contact continuity is not
      visually reliable enough for controller/DP integration.
