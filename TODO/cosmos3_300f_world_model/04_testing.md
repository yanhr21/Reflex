# Testing TODO

## Current Active Boundary

- [ ] Current source line is the fix3 v7 user-override `733`-row SFT source.
      The first SFT root
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_4gpu_20260612_0245`
      completed as negative diagnostic evidence. The later
      `normactive_clip1` root is also rejected because it selected the action
      tensors but used the wrong action-training recipe. The current active
      follow-up root is
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`,
      which uses the overfit-approved fix1 defaults (`lr=1e-4`,
      `action_loss_weight=2.0`, `independent_action_schedule=true`,
      `shift_action=1`) and passed the recipe guard at launch. This data is
      still DP-success-filtered bootstrap data, not hard-dynamic proof. Stored
      accepted H5 replay should succeed by construction, and same-seed DP
      reruns on this accepted subset are expected to be near ceiling if the
      environment is reproducible. The immediate gate is strict same-length
      generated-video/action inspection, generated-RGB task-state
      readout/profile, and direct visual review after each evaluated
      checkpoint. Do not report task-success improvement over DP from this
      accepted subset. Do not start closed-loop DP/controller evaluation until
      this gate produces acceptable target-motion/final-target/peg-contact
      evidence, and later method-gain claims must use an unfiltered/hard
      dynamic baseline where frozen DP success is measured separately.
- [ ] Active iter300 gate for the current fix1-recipe full run: Slurm step
      `126210.41` on `server56` is training, and auxiliary job `127120` on
      `server40` is waiting for checkpoint `iter_000000300` to run strict eval
      on 10 samples. After strict eval passes, run generated-RGB readout with
      the existing v7_733 reference readout checkpoint
      `sft_full_episode_wam_fix3_v7_733_rgb_300step_4gpu_20260612_0245/task_state_readout_reference_rgb_301f_v7_733/best_model.pt`,
      then run the readout failure profile and open review sheets before
      deciding whether to continue to iter600 or debug.
- [x] Current 2026-06-12 19:38 CST monitor: the fix1-recipe full run is live
      and healthy through rank-0 iteration 45. Iter0 validation loss was
      `3.606580`; rank-0 iteration 45 train loss is `0.9703` with
      `vision=0.0544`, `action=0.4580`, and no OOM/NaN/traceback marker in the
      inspected log. The iter300 watcher remains live on auxiliary job
      `127120` and is correctly waiting; no iter300 checkpoint exists yet.
- [x] Post-SFT training/eval code-path audit found no obvious 128/93-frame
      truncation, fixed-eight-prefix misuse, or future-action leakage through
      the condition mask. Rows reference full `301`-frame videos and `300x32`
      targets; per-row variable `condition_frame_indexes_vision/action` are
      used for action-conditioned samples; only prefix action indexes are clean
      conditions. The current negative result is therefore a data/model
      capability issue, not an immediately visible prefix-mask contract bug.
- [x] Extend and rerun the full-episode eval artifact inspector to split prefix
      and future action/state errors. The current `iter_000001500` eval remains
      strict-ok and preserves conditioned history (`robot_action_prefix_rmse`
      `0.0017889231`, `state_sidecar_prefix_rmse` `0.0015759602`), while future
      errors remain large (`robot_action_future_rmse` `1.0473393741`,
      `state_sidecar_future_rmse` `0.7528493327`). This localizes the failure
      to future rollout quality rather than missing/ignored prefix conditions.
- [x] Current v7_733 distribution diagnosis completed. The source has `733`
      episodes expanded to `2899` full-episode prefix rows, but coverage is
      skewed: `peg_disturb` has only `2` source episodes / `10` prefix rows,
      and `peg_recovery` is mostly `peg_drop` rather than moving-hole recovery.
      This is enough to validate the WAM SFT interface, but not enough to claim
      robust disturbance recovery or hard target-rebinding behavior.
- [ ] Monitor the follow-up v7_733 diagnostic SFT:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_normactive_clip1_4gpu_20260612_124500`.
      It reuses the same rendered dataset and WAM condition root, but changes
      `normalize_loss_by_active=true` and `grad_clip_norm=1.0` to address the
      observed failure where prefix conditions are preserved but future
      action/state/video rollout is poor. Startup passed; iter-0 validation
      loss was finite at `17.642225`. Do not compare this loss directly with
      the first run's iter-0 loss because active-token normalization changes
      the scale. Evaluate generated artifacts at saved checkpoints before any
      controller decision.
      Update `2026-06-12 13:11 CST`: SFT step `126210.38` remains live on
      `server56`, recent logs reached about iteration `63`, and there is no
      failure marker or saved checkpoint yet. A tmux-held auxiliary 1-GPU eval
      allocation was requested as pending job `126985`; use it for strict
      checkpoint eval if it starts, otherwise wait until the 4-GPU training
      allocation is free rather than running eval concurrently on the same
      GPUs.
      Update `2026-06-12 13:14 CST`: auxiliary eval allocation `126985`
      started on `server40`; watcher step `126985.0` is waiting for
      `iter_000000300` and will run strict full-episode eval into
      `eval_full_episode_wam_iter_000000300` once checkpoint files are stable.
      Generated-RGB readout and failure-profile watchers are queued behind the
      iter300 eval step in the same held allocation. The readout watcher now
      checks `strict_eval_artifacts_ok=true` before running, so downstream
      diagnostics will not proceed from a failed length/action artifact gate.
      Update `2026-06-12 13:27 CST`: training remains healthy at about
      iteration `125`, throughput about `17.3s/iter`, with recent rank-0 train
      losses mostly in the `6-8` range and action loss around `0.6-0.8`.
      `iter_000000300` is not expected yet and the eval watcher is still
      correctly waiting.
      Update `2026-06-12 14:35 CST`: `iter_000000300` was saved and evaluated
      in the auxiliary 1-GPU allocation. The strict artifact gate passed
      structurally (`10` samples, generated/reference videos `301/301`,
      actions `300x32`, `strict_failures=[]`, mean future PSNR
      `19.6801159248`), and prefix preservation remains good
      (`robot_action_prefix_rmse=0.0017889231`,
      `state_sidecar_prefix_rmse=0.0015759602`). The future rollout remains
      weak (`robot_action_future_rmse=1.0274358407`,
      `state_sidecar_future_rmse=0.7472363522`, mean action RMSE
      `0.6560889600`).
      Generated-RGB readout/profile passed strict structure but is negative:
      mean final hole error `0.1143826719` m, mean future hole/peg/TCP RMSE
      `0.0548931954` / `0.0669841219` / `0.0657259292` m, and mean future
      peg-head-hole RMSE `0.0522709684` m. Target-onset diagnostics still
      fire early on moving samples and false-fire on static samples.
      All `10` review sheets were opened; visual review is negative for
      controller handoff because multiple panels show robot/peg divergence
      after the prefix, dropped/misplaced pegs, large white/fog-like robot
      ghosting, and unreliable peg-hole contact. Do not start closed-loop from
      `iter_000000300`.
      An `iter_000000600` strict eval/readout/profile watcher chain is now
      running/queued in held allocation `126985`; the eval watcher is Slurm
      step `126985.3` on `server40` and is waiting for the checkpoint.
      Update `2026-06-12 14:41 CST`: follow-up SFT remains live on 4 GPUs as
      Slurm step `126210.38` and has reached rank-0 iteration `367`. Recent
      train losses are finite, mostly in the `4-6` range, with action loss
      around `0.4-0.6`. No checkpoint beyond `iter_000000300` exists yet.
      The iter-600 eval watcher is still active on compute node `server40` as
      step `126985.3`; readout/profile watchers remain queued behind it in the
      same held allocation.
      Update `2026-06-12 14:45 CST`: follow-up SFT is still healthy on
      `server56`, now at rank-0 iteration `379`, with finite recent train loss
      in the same `4-6` range. No error or `sft_failed` marker was found, and
      checkpoints still contain only `iter_000000300`. The iter-600 watcher
      remains active on compute node `server40` and is still correctly waiting
      for `iter_000000600`; generated eval/readout/profile for iter600 do not
      exist yet.
      Update `2026-06-12 14:48 CST`: follow-up SFT remains live and healthy at
      rank-0 iteration `390`. Recent train loss is still finite in the same
      range, and there is still no traceback, runtime error, or `sft_failed`
      marker. No checkpoint beyond `iter_000000300` exists yet, so the
      iter-600 strict eval/readout/profile chain is still correctly waiting.
      Update `2026-06-12 14:51 CST`: follow-up SFT reached rank-0 iteration
      `400` with finite loss (`4.1276`, vision `0.0416`, action `0.4086`) and
      no error marker. Checkpoints still contain only `iter_000000300`;
      `iter_000000600` is still pending. At the current `~17.3s/iter` pace,
      the iter-600 checkpoint is roughly one hour away from this timestamp.
      Update `2026-06-12 15:00 CST`: follow-up SFT remains live on
      `server56`, now at rank-0 iteration `435`, with finite train loss still
      dominated by the weighted action component (`Loss=4.6974`, vision
      `0.0398`, action `0.4658`). Checkpoints still contain only
      `iter_000000300`; no `iter_000000600` eval/readout/profile artifacts
      exist yet. The iter-600 eval watcher is active as step `126985.3` on
      `server40` and is correctly waiting for the checkpoint. Readout/profile
      watcher `Requested nodes are busy` messages are expected because the
      eval watcher owns the single held auxiliary GPU step.
      Update `2026-06-12 15:14 CST`: follow-up SFT reached rank-0 iteration
      `480` with finite loss (`4.3913`, vision `0.0347`, action `0.4357`).
      Checkpoints still contain only `iter_000000300`; the iter-600 eval
      watcher remains active and correctly reports `checkpoint_not_ready`.
      No `iter_000000600` eval/readout/profile artifact exists yet, so the
      closed-loop gate remains closed.
      Update `2026-06-12 15:30 CST`: follow-up SFT reached rank-0 iteration
      `535` with finite loss (`4.6347`, vision `0.0418`, action `0.4593`).
      Checkpoints still contain only `iter_000000300`; no iter-600 eval files
      exist. The held aux eval watcher continues to wait for
      `iter_000000600`, so no controller/closed-loop action is allowed yet.
      Update `2026-06-12 16:05 CST`: checkpoint `iter_000000600` was saved
      and evaluated in held aux allocation `126985`. Strict artifact
      inspection passed structurally for `10` samples with generated/reference
      videos `301/301`, action tensors `300x32`, and `strict_failures=[]`.
      Aggregate metrics improved only mildly over iter300: mean future PSNR
      `19.8479565666`, mean action RMSE `0.6165386443`,
      robot-action future RMSE `1.0047414112`, state-sidecar future RMSE
      `0.6844772048`; prefix preservation stayed clean
      (`robot_action_prefix_rmse=0.0017889231`,
      `state_sidecar_prefix_rmse=0.0015759602`).
      Generated-RGB readout/profile also passed strict structure, with mean
      final hole error `0.0861941030` m, mean future hole/peg/TCP RMSE
      `0.0478804191` / `0.0555931353` / `0.0568463799` m, and mean future
      peg-head-hole RMSE `0.0468595485` m. This is numerically better than
      iter300, but target-onset remains controller-negative: static/peg-only
      samples false-fire target motion around frames `5-17`, and moving
      samples still often fire tens of frames before the target onset at low
      thresholds.
      Visual review opened all `10` iter600 review sheets. The gate remains
      negative for controller handoff: several samples still show robot/peg
      divergence after the prefix, white/fog-like geometry collapse around the
      gripper/block, peg drops or contact discontinuities, and moving-hole
      insert/resume predictions that cannot be used as DP-resume future-state
      images. Do not start closed-loop from `iter_000000600`.
      An `iter_000000900` strict eval/readout/profile watcher chain is now
      running/queued in the held auxiliary allocation `126985`. The eval
      watcher is tmux session `cosmos3_v7_733_iter900_eval_watch_126985` and
      Slurm step `126985.6`, waiting for checkpoint `iter_000000900` on
      compute node `server40`; readout/profile watcher sessions
      `cosmos3_v7_733_iter900_readout_watch_126985` and
      `cosmos3_v7_733_iter900_profile_watch_126985` are queued behind the same
      1-GPU allocation.
      Update `2026-06-12 16:14 CST`: follow-up SFT remains healthy on
      `server56` as Slurm step `126210.38`, with rank-0 logs at iteration
      `678` and finite loss (`3.9466`, vision `0.0319`, action `0.3915`).
      Checkpoints still contain only `iter_000000300` and `iter_000000600`.
      The iter900 eval watcher remains active as Slurm step `126985.6` on
      `server40` and is correctly reporting `checkpoint_not_ready`; no iter900
      eval/readout/profile artifacts exist yet.
      Update `2026-06-12 16:26 CST`: follow-up SFT remains live on
      `server56` as Slurm step `126210.38`, with rank-0 logs at iteration
      `718` and finite loss (`3.7110`, vision `0.0336`, action `0.3677`).
      Checkpoints still contain only `iter_000000300` and `iter_000000600`.
      The iter900 eval watcher remains active as Slurm step `126985.6` on
      `server40`, with `checkpoint_not_ready` through `960` seconds. No
      `iter_000000900` strict eval/readout/profile artifact exists yet, so the
      closed-loop gate remains closed.
      Update `2026-06-12 16:32 CST`: follow-up SFT remains live on
      `server56` as Slurm step `126210.38`, with rank-0 logs at iteration
      `739` and finite loss (`4.0361`, vision `0.0451`, action `0.3991`).
      Checkpoints still contain only `iter_000000300` and `iter_000000600`;
      the iter900 watcher remains active as Slurm step `126985.6` on
      `server40`, with `checkpoint_not_ready` through `1320` seconds. A
      light active-script/TODO audit during the wait confirmed that current
      active scripts mainly cover strict SFT eval, generated-RGB readout, and
      readout profiling; old controller wrappers were intentionally moved by
      the archive boundary. Therefore, if a future checkpoint passes the
      generated-video/action/readout/visual gate, the next controller step must
      follow the PLAN receding interface and not restart an old archived
      controller wrapper.
      Update `2026-06-12 16:39 CST`: follow-up SFT remains live on
      `server56` as Slurm step `126210.38`, with rank-0 logs at iteration
      `762` and finite loss (`4.0525`, vision `0.0292`, action `0.4023`).
      Checkpoints still contain only `iter_000000300` and `iter_000000600`;
      no `iter_000000900` eval/readout/profile artifacts exist yet. The
      iter900 eval watcher remains active as Slurm step `126985.6` on
      `server40`, with `checkpoint_not_ready` through `1740` seconds.
      Readout/profile watcher `Requested nodes are busy` messages remain
      expected because the strict eval watcher owns the single held auxiliary
      GPU step while waiting. A focused script audit found no active source
      controller/receding wrapper under `scripts/world_model` or
      `scripts/slurm`; only stale `__pycache__` names mention receding or
      controller distillation. If a future checkpoint passes the gate, the
      next closed-loop step must implement or restore the PLAN receding
      interface before running live DP/controller tests.
      Update `2026-06-12 16:43 CST`: live check still shows only
      `iter_000000300` and `iter_000000600`; SFT is healthy at rank-0
      iteration `777` with finite loss (`4.0125`, vision `0.0488`, action
      `0.3964`), and the iter900 watcher is still waiting. During the wait,
      a closed-loop dependency audit confirmed the future receding wrapper
      must combine three existing contracts: Cosmos eval emits a full
      normalized `300x32` sequence where the first `7` dimensions are robot
      action and the remaining `25` are task-state sidecars; the condition
      root stores `normalization_stats.json` and must be used to
      de-normalize only the first `7` robot-action dimensions before any
      live `env.step`; the frozen DP checkpoint manifest uses
      `PegInsertionSide-v1`, `pd_ee_delta_pose`, `obs_horizon=2`,
      `act_horizon=8`, and `max_episode_steps=300`. Therefore a future
      closed-loop test, if the SFT gate ever passes, must execute only short
      8-step-or-smaller action prefixes followed by real re-observation, not
      a one-shot 300-step open-loop Cosmos rollout, and must treat the
      sidecar dimensions as diagnostic predicted task state rather than
      simulator oracle state.
      Update `2026-06-12 16:56 CST`: follow-up SFT remains live on
      `server56` as Slurm step `126210.38`, with rank-0 logs at iteration
      `820` and finite loss (`4.4826`, vision `0.0420`, action `0.4441`).
      Checkpoints still contain only `iter_000000300` and `iter_000000600`;
      the `iter_000000900` watcher remains active as Slurm step `126985.6`
      on `server40`, with `checkpoint_not_ready` through `2701` seconds.
      Readout/profile watcher busy-node messages remain expected because the
      strict eval watcher owns the single auxiliary GPU step while waiting.
      No iter900 generated artifact exists yet, so visual review and
      closed-loop gating cannot advance.
      Update `2026-06-12 17:38 CST`: checkpoint `iter_000000900` was saved,
      evaluated, read out, and visually reviewed. Strict artifact inspection
      still passed structurally (`10` samples, generated/reference videos
      `301/301`, action tensors `300x32`, `strict_failures=[]`). Aggregate
      metrics improved over iter600 but remain weak for controller use:
      mean future PSNR `20.0838192282`, mean action RMSE `0.5730889361`,
      robot-action future RMSE `0.8931446655`, and state-sidecar future RMSE
      `0.6542504528`; prefix preservation stayed clean
      (`robot_action_prefix_rmse=0.0017889231`,
      `state_sidecar_prefix_rmse=0.0015759602`).
      Generated-RGB readout/profile passed strict structure, with mean final
      hole error `0.0662412508` m, mean future hole/peg/TCP RMSE
      `0.0401320661` / `0.0537541486` / `0.0534759435` m, and mean future
      peg-head-hole RMSE `0.0396896749` m. Target-onset diagnostics remain
      controller-negative: static/peg-only samples false-fire target motion
      near early frames, and moving samples often predict target motion tens
      of frames before the real onset at low thresholds.
      All `10` iter900 review sheets were opened. The gate remains negative
      for controller handoff: several samples still show robot/peg divergence,
      dropped or table-contact peg paths, white/transparent geometry collapse
      around the gripper/block, and non-physical contact continuity. Do not
      start closed-loop from `iter_000000900`.
      An `iter_000001200` strict eval/readout/profile watcher chain is now
      active in held auxiliary allocation `126985`. As of `17:38 CST`,
      SFT step `126210.38` is live on `server56` at rank-0 iteration `955`,
      checkpoints contain only `300/600/900`, and watcher step `126985.9` on
      `server40` is correctly reporting `checkpoint_not_ready`.
- [x] Current v7_733 SFT startup is valid: source/render/condition export and
      action-target audit passed under the full episode contract (`301` RGB
      frames, `300x32` action targets, no 128/129-frame slicing), and training
      reached iteration `1400+` on `4xH200` with finite loss.
- [x] After final checkpoint `iter_000001500` is stable, run
      `scripts/slurm/run_cosmos3_300f_full_episode_wam_eval_in_allocation.sh`
      with `SFT_ROOT` and `CONDITION_ROOT` pointing to the fix3 v7_733 roots.
      The output root should be a fix3-specific
      `eval_full_episode_wam_iter_000001500` directory and must pass strict
      `301/301` video and `300x32/300x32` action inspection.
- [x] Current fix3 v7_733 `iter_000001500` strict artifact inspection passed:
      `strict_eval_artifacts_ok=true`, `strict_failures=[]`, `10` samples,
      generated/reference videos `301/301`, action tensors `300x32`, mean
      future video PSNR `19.6738554813`, mean action RMSE `0.6634088986`,
      mean robot-action future RMSE `1.0473393741`, and mean state-sidecar
      future RMSE `0.7528493327`.
- [x] Current fix3 v7_733 `iter_000001500` visual review opened all `10`
      review sheets. This gate is negative for controller handoff: prediction
      after the prefix often has semi-transparent/ghosted robot geometry, poor
      peg/gripper/contact continuity, and unreliable insert/resume or
      peg-recovery behavior.
- [x] Continue generated-RGB
      task-state readout using
      `scripts/slurm/run_cosmos3_300f_task_state_readout_in_allocation.sh` or
      the readout watcher with the current dataset manifest and eval root.
      Then run the readout failure profile and inspect review sheets/videos.
      Current v7_733 readout completed `2000` steps. Reference-RGB validation
      at step `2000` reached future hole/peg/TCP/peg-head-hole RMSE
      `0.0390807018` / `0.0338327438` / `0.0326168649` /
      `0.0579820797` m. Generated-RGB readout strict structure passed, but
      failure profile is negative: mean final hole error `0.1211597189` m,
      mean future hole/peg/TCP/peg-head-hole RMSE `0.0612715537` /
      `0.0640420842` / `0.0652820180` / `0.0502333957` m. Moving target onset
      is predicted `70-110` frames too early at the 2 mm threshold, and
      static/peg-only samples false-fire target motion around frames `5-8`.
- [ ] Closed-loop/receding DP tests may start only after the v7_733 generated
      video/action/readout/visual gate is reviewed. If the gate is weak, record
      it as SFT diagnostic evidence and do not present controller success.
      This gate is currently weak/negative; closed-loop remains blocked.

## Historical 2026-06-10 Fix1 Diagnostic Records

- [x] Historical, not active: the 2026-06-10 full1000 fix1 SFT was run after
      the user accepted the two-sample overfit result. Its root is
      `sft_full_episode_wam_full1000_rgb_300step_fix1_4gpu_20260610_1131`.
      These entries are preserved as log-backed training/software sanity and
      negative diagnostic evidence only. They must not be interpreted as
      completed gates for the current active v7_733 follow-up SFT.
- [x] Historical fresh fix1 gates completed: iter300, iter600, iter900,
      iter1200, and iter1500 strict eval/readout/visual review. Iter1100
      remained the best validation point at `0.607350`; iter1500 final
      validation was `0.662956`, so the final checkpoint was evaluated as
      generated evidence, not as a validation-best claim.
- [x] Historical fresh fix1 iter1500 completed in aux allocation `123499`, not
      on the login node. Strict eval artifacts passed for all 10 samples,
      readout and failure profile passed strict structure, and all 10 review
      sheets were opened. The conclusion was not controller-ready because
      target onset still fired too early or falsely and peg/drop/contact
      continuity was still visually unreliable.

## Validation Panel

- [x] Select a scenario-diverse validation panel after data preflight:
      `none`, pre-motion target, observed target motion, move-stop, reverse,
      peg disturbance, peg drop. The current fixed 10-sample panel covers
      `hole_move_stop`, `hole_reverse`, `hole_constant`, `peg_drop`,
      `peg_disturb`, and `none`.
- [x] Generate full-episode or remaining-episode rollouts from causal prefixes
      for current completed fresh-fix1 checkpoints `iter_000000300`,
      `iter_000000600`, `iter_000000900`, `iter_000001200`, and
      `iter_000001500` under the active 301-frame / 300-action contract.
- [x] Refuse 129-frame validation clips as method evidence. All completed
      active checkpoint evals have passed strict `301/301` video and
      `300x32/300x32` action artifact inspection.

## Metrics

- [x] Patch the strict eval artifact inspector to report executable robot
      action and structured state sidecar separately. The active action tensor
      is `300x32`: the first `7` dimensions are robot action targets and the
      remaining dimensions are causal/proprio/object-state sidecar/readout
      targets. Future reports must include `robot_action_future_rmse` and
      `state_sidecar_future_rmse`, not only a mixed 32-D action RMSE.
- [x] Patch the future-action metric boundary for the two-sample fix1 gate:
      action step `i` advances video frame `i` to `i+1`, so if the visual
      prefix ends at frame `f`, action `f` is the first future action. The
      inspector now starts future-action RMSE at `prefix_frame_index`, not
      `prefix_frame_index + 1`. This preserves the strict controller-facing
      objective and does not change training.
- [x] Two-sample overfit fix1 strict validation generated and inspected enough
      for its purpose: `iter_000000100` produced strict `301/301` videos and
      `300x32/300x32` actions for both rows, and the user accepted the videos.
      Later overfit checkpoints are intentionally skipped; do not spend more
      time on overfit unless a future regression requires it.
- [x] Visually inspect generated videos/contact sheets before any success
      claim. Current fresh fix1 completed gates through iter1500 have all been
      inspected. The result is SFT progress evidence only, not controller
      success evidence, because target-motion onset and peg-drop/regrasp remain
      unresolved.
- [x] Current fresh fix1 iter1500 strict artifact inspection passed for all 10
      samples: generated/reference videos `301/301`, actions `300x32/300x32`,
      strict failures empty, mean action RMSE `0.3157882582`, mean
      robot-action future RMSE `0.8203911022`, mean state-sidecar future RMSE
      `0.0626562464`, and mean future-video PSNR `23.1641813684`.
- [x] Current fresh fix1 iter1500 generated-RGB readout/profile passed strict
      structure: mean final hole error `0.0538804681` m, mean future
      hole/peg/TCP RMSE `0.0301422454` / `0.0386116191` / `0.0347898968` m,
      and mean future peg-head-hole RMSE `0.0413906728` m. These are the best
      final-hole/future-hole/peg/TCP readout values among the current
      inspected fresh-fix1 checkpoints, but they do not solve mode switching
      or executable peg/contact handoff.
- [x] Current fresh fix1 iter1500 visual review completed for all 10 review
      sheets. The rollouts remain readable and coherent with no old white/fog
      collapse. Static and many target-motion panels are visually close, but
      peg_drop/peg_disturb/insert_resume still show unreliable peg-gripper-hole
      contact continuity or visible pose/contact mismatch, so this is not
      controller/DP handoff evidence.
- [x] Current fresh fix1 iter1500 target-onset profile remains negative for a
      controller-facing switch: moving target samples still fire early by
      tens of frames under low displacement thresholds, and no-target-motion
      static/peg-only samples still false-fire. Example: sample 00 predicts
      onset `7/9/11/70/125` vs target `88/88/89/91/96`; sample 06 static
      predicts onset `9/12/19/69/75` despite no target motion.
- [x] Target onset AUROC/F1 and frame timing diagnostics were produced for the
      final `iter_000001500` validation panel from readout displacement scores.
      Generated-RGB readout has frame AUROC `0.7805157241`, fixed 2 mm F1
      `0.5295591182`, best F1 `0.6556728232` at score threshold
      `0.0193880412` m, moving-onset mean abs error `82.4` frames, and
      `5/5` static false-positive samples. Reference-RGB calibration on the
      same 10 samples has frame AUROC `0.8549271165`, fixed 2 mm F1
      `0.5304893350`, best F1 `0.7244258873` at score threshold
      `0.0610896768` m, moving-onset mean abs error `82.0` frames, and
      `5/5` static false-positive samples. This proves the 2 mm displacement
      onset score is not reliable as a mode-switch signal, even on GT RGB.
- [ ] Replace the low-threshold displacement-on-readout target switch with a
      calibrated target-motion/onset head or score before any controller-facing
      mode switch. The current readout displacement score is diagnostic only.
- [x] Added and evaluated a calibrated temporal target-motion head over
      RGB-derived task-state readout:
      `scripts/world_model/train_target_motion_head_from_readout.py`, with
      reference-RGB roots built by
      `scripts/world_model/build_cosmos3_reference_rgb_eval_root_from_jsonl.py`
      and run through
      `scripts/slurm/run_cosmos3_target_motion_head_calibration_in_allocation.sh`.
      It used `120` train reference videos and `88` held-out val reference
      videos inside Slurm. On held-out reference RGB it reached frame AUROC
      `0.9115353604`, F1@0.5 `0.7669897596`, and best F1 `0.7788987602`,
      improving over the raw displacement score.
- [x] Re-ran the calibrated target-motion head diagnostic on the current fresh
      fix1 `iter_000001500` generated-RGB readout in aux allocation `123499`
      step `12`, with output root
      `target_motion_readout_calibration_fresh_fix1_iter1500_20260610`. The
      current generated panel reaches frame AUROC `0.7808262378`, F1@0.5
      `0.6005305040`, and best F1 `0.6179090483` at threshold
      `0.0358722173`. This confirms the target monitor is calibratable on
      reference-like RGB, but the current fresh generated rollouts/readout are
      still too poor for controller-facing switching.
- [ ] Do not use the calibrated target-motion head as a controller switch until
      a future Cosmos3 checkpoint or interface produces generated-RGB/readout
      quality close enough to the reference-RGB calibration regime.
- [x] Target path RMSE and final target pose error are reported for completed
      generated-RGB readout/profile gates. Historical pre-fix1 4-GPU
      `iter_000001500` mean final hole error was `0.1028195255` m and remains
      negative historical evidence. The current fresh fix1 `iter_000001500`
      mean final hole error is `0.0538804681` m, improved but still not
      controller-ready because target timing and contact continuity fail.
- [x] TCP/peg pose readout error is reported for completed generated-RGB
      readout/profile gates. Historical pre-fix1 4-GPU `iter_000001500` mean
      future peg/TCP RMSE were `0.0669838827` / `0.0666590482` m. The current
      fresh fix1 `iter_000001500` mean future peg/TCP RMSE are
      `0.0386116191` / `0.0347898968` m, improved but not sufficient for
      executable peg/contact handoff by visual inspection.
- [x] Peg-head-in-hole-frame error is reported for completed generated-RGB
      readout/profile gates. Historical pre-fix1 4-GPU `iter_000001500` mean
      future peg-head-hole RMSE was `0.0516100994` m; current fresh fix1
      `iter_000001500` mean future peg-head-hole RMSE is `0.0413906728` m.
- [x] Grasp/hold and inserted predicate diagnostics are produced by the
      generated-RGB readout eval. They remain diagnostics only, and visual
      review overrides favorable scalar values when contact/hold is visibly
      wrong.
- [x] Predicted action sequence metrics over intended action steps are
      reported for completed strict eval gates. Historical pre-fix1 4-GPU
      `iter_000001500` mean action RMSE was `0.6161383192`; current fresh fix1
      `iter_000001500` mean action RMSE is `0.3157882582` over `300x32`
      action tensors.
- [x] RGB reconstruction diagnostics over exact intended frames are reported
      for completed strict eval gates. Historical pre-fix1 4-GPU
      `iter_000001500` mean future-video PSNR was `20.2640810606` dB; current
      fresh fix1 `iter_000001500` mean future-video PSNR is `23.1641813684`
      dB over the exact inspected full-episode contract.

## Visual Evidence

- [x] Inspect every major validation video/contact sheet before recording a
      result as positive. Completed `iter_000000300`, `iter_000000600`,
      `iter_000000900`, `iter_000001200`, and `iter_000001500` sheets were
      inspected and recorded as negative diagnostics, not success evidence.
- [x] Mark visual peg-drop/no-hold failures as negative diagnostics even if a
      scalar metric is favorable. The completed checkpoints still fail
      peg-drop/regrasp, insert-resume peg continuity, target-motion timing,
      and contact geometry.
- [ ] Do not start controller/DP integration from the completed SFT
      checkpoints. Validation generation, readout, action metrics, and visual
      review now exist through `iter_000001500`, but the evidence is negative
      for target-motion detection, final target prediction, peg recovery, and
      executable handoff.
