# Review Gate TODO

## Current Active Boundary

- [x] The current authoritative v7_733 follow-up Cosmos3 SFT is the fix1-recipe
      root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`.
      It used the overfit-approved action recipe and ended at Slurm wall time
      after rank-0 iteration `743`. The frozen source is the user-override
      `733`-row v7 DP source, not the old full1000/fix1 chain and not a
      hard-dynamic baseline.
- [x] Current fix1-recipe v7_733 review gates completed: `iter_000000300` and
      `iter_000000600` strict generated artifacts, generated-RGB
      readout/profile, and direct visual review. Both passed structural
      length/action checks but failed controller/DP readiness. Iter300 is the
      best qualitative sanity checkpoint so far but remains imprecise for
      handoff; iter600 is worse on rollout/readout/visual evidence despite
      lower validation loss.
- [x] No current v7_733 review gates are pending. The latest fix1-recipe root
      saved no `iter_000000900` checkpoint, no `iter_000001200` checkpoint, and
      has no active watcher. Old `normactive_clip1` iter900/iter1200 notes are
      historical negative diagnostics and must not be treated as the active
      review gate.
- [ ] No controller or DP integration may be launched from the current v7_733
      line until a future completed checkpoint passes strict artifact accounting,
      generated-RGB readout/profile, and visual review for target motion,
      final target pose, peg/hand continuity, and executable handoff.
- [x] Entries below that refer to the 2026-06-10 full1000/fix1 runs, old
      allocations `123385`/`123499`, or older 4-GPU warm-start checkpoints are
      historical diagnostics only. They are not the active gate for the
      current v7_733 SFT.

## Review Status

- [x] User review gate passed by user direction after the new PLAN/TODO was
      reviewed.
- [x] Full-episode condition export was launched inside held Slurm allocation
      `122736`, not on the login node.
- [x] Fresh Cosmos3 full-episode SFT was launched inside the same allocation.
- [x] Validation generation input/inspection scripts were prepared before the
      first checkpoint.
- [x] Task-state readout prediction/inspection wrappers were prepared.
- [x] Checkpoint `iter_000000300` was generated and evaluated under the strict
      301-frame RGB / 300-action contract.
- [x] Step-250 reference-video readout snapshot was used to run generated-video
      task-state diagnostics for the 10 validation rollouts.

## Still Not Started

- [ ] No controller or DP integration has been launched.
- [ ] No controller or DP integration may be launched from the latest
      fix1-recipe `iter_000000300`. That checkpoint passed length/action
      artifact checks and is the best current qualitative sanity checkpoint,
      but visual/readout evidence is still too imprecise for executable
      handoff.
- [ ] No controller or DP integration may be launched from the latest
      fix1-recipe `iter_000000600`.
- [x] The latest fix1-recipe `iter_000000600` specifically is not acceptable
      for controller or DP integration. Its strict artifact contract passed,
      but generated-RGB readout and visual inspection still fail the
      target-motion/recovery handoff requirement.

## Current Negative Diagnostic

- [x] Iteration-300 generated artifacts passed strict structural checks:
      `strict_eval_artifacts_ok=true`, 10/10 strict samples, predicted/reference
      videos `301/301` frames, predicted/reference actions `300x32/300x32`,
      aggregate action RMSE `0.6931767245`, aggregate future-video PSNR
      `19.866555` dB.
- [x] Visual review failed the active objective: representative early-prefix
      target-motion and peg-recovery sheets show severe mid/late-rollout visual
      corruption and unreliable robot/peg/target geometry. Static-late samples
      are better, which suggests the failure is long future prediction and
      dynamic contact/object interaction, not a simple checkpoint-load or
      length-accounting issue.
- [x] Generated-video readout on the step-250 reference readout snapshot passed
      strict shape/finite checks but failed as target-motion evidence. It
      predicts motion onset at frames `3-8` for all 10 samples, including
      static `none` scenarios. Mean final hole-position error is
      `0.098346` m. Treat this as a readout/world-rollout diagnostic failure,
      not controller evidence.

## Completed Evidence Gate

- [x] The duplicate 1-GPU SFT was stopped only after checkpoint
      `iter_000000600` had been saved and the 4-GPU warm-start had produced
      real training losses. Step `122736.6` is gone, and the old allocation
      `122736` later expired/revoked.
- [x] The separate reference-readout allocation `122767` finished training,
      returned to an idle shell, and was released. The duplicate 1-GPU SFT
      stayed stopped; single-GPU resources were used only for readout, sanity,
      and video/render support. The current held support allocation is
      `123381`.
- [x] The `iter_000000600` checkpoint-specific eval completed under
      `eval_full_episode_wam_iter_000000600` with
      `strict_eval_artifacts_ok=true`, mean action RMSE `0.6496965469564411`,
      and mean future-video PSNR `19.860613478278417`.
- [x] Visual review of `iter_000000600` failed the active objective despite
      better visual stability than `iter_000000300`: target-motion and
      insert-resume rollouts still drift in peg/hand/contact geometry, and the
      peg-drop recovery rollout does not regrasp the peg.
- [x] Generated-RGB readout for `iter_000000600` failed target-motion evidence:
      mean final hole-position error is `0.1153032929` m; moving-target onsets
      are predicted roughly `77-84` frames too early; static/peg scenarios also
      produce false target-motion onsets.
- [x] The 4-GPU warm-start SFT ran in Slurm job `123131`. It passed startup
      validation, entered training, passed the user's 1-hour floor, and
      completed through `iter_000001500`. Its evaluated checkpoints produced
      strict eval/readout/visual diagnostics, but those diagnostics failed the
      active handoff objective.
- [x] A 4-GPU checkpoint watcher ran in Slurm step `122782.28`, waited for
      `iter_000000300` of the 4-GPU warm-start, and completed the same strict
      full-episode eval.
- [x] The 4-GPU checkpoint watcher was repaired and restarted before the
      checkpoint arrived. The current watcher is step `122782.28`; it waits for
      `model/.metadata` and stable checkpoint files before eval, preventing a
      half-written DCP checkpoint from becoming eval input.
- [x] A paired 4-GPU generated-RGB readout watcher ran in Slurm step
      `122736.92` and completed task-state readout diagnostics under the same
      4-GPU eval root after video/action inspection completed.
- [x] The 4-GPU `iter_000000300` strict artifact contract passed:
      `strict_eval_artifacts_ok=true`, mean action RMSE `0.6329251997`, and
      mean future-video PSNR `19.8388587420`.
- [x] The 4-GPU `iter_000000300` visual/readout gate failed. Visual sheets
      still show peg/hand/contact drift, peg-drop recovery does not regrasp,
      and one target-pre-motion constant-hole sample develops severe
      transparent artifacts. Generated-RGB readout reports mean final
      hole-position error `0.1043972932` m, moving-target onsets predicted
      about `78-87` frames early, false target-motion onsets in static/peg-only
      scenes, and peg-drop/peg-disturb future grasp accuracy `0.417910` /
      `0.000000`.
- [ ] No controller or DP integration may be launched from the 4-GPU
      `iter_000000300` checkpoint.
- [x] The 4-GPU `iter_000000900` gate completed under the same 301-frame /
      300-action contract. Strict artifacts passed with mean action RMSE
      `0.6172093662` and mean future-video PSNR `20.2230188270` dB, improving
      slightly over `iter_000000600`.
- [x] The kept 1-H200 allocation `122736` was reserved for non-SFT support
      work while it existed. The `iter_000000600` readout watcher `122736.93`
      and readout-failure-profile watcher `122736.104` completed as negative
      diagnostics. Allocation `122736` is now gone; a replacement tmux-held
      1-H200 support allocation `123366` ran on `server21` for readout,
      sanity, and video/render support. Iter900, iter1200, and iter1500
      readout/profile watchers completed as negative diagnostics. After
      `123366` was revoked, replacement tmux-held allocation `123381` was
      acquired on `server21` for later sanity and video/render tasks only.
- [x] Framework-facing inference contract probe passed for representative
      4-GPU `iter_000000300` generated samples. It confirms the evaluation
      helper resolves the intended `301` frames, `300` action steps, raw action
      dim `32`, and per-prefix vision/action condition masks. This narrows the
      current failure to model/long-horizon dynamics/readout quality rather
      than accidental truncation or a framework default mask.
- [x] Generated-RGB readout failure profile for 4-GPU `iter_000000300` also
      completed. It confirms the early-target-motion diagnosis persists at
      larger displacement thresholds and co-occurs with future peg/TCP drift:
      mean future hole RMSE `0.0519564251` m, future peg RMSE
      `0.0741150761` m, future TCP RMSE `0.0719546468` m. This remains a
      diagnostic on generated RGB, not a new gate or controller success.
- [x] The 4-GPU `iter_000000600` gate completed under the same contract.
      Strict artifacts passed with mean action RMSE `0.6199095227` and mean
      future-video PSNR `20.1762433861` dB, improving over 4-GPU
      `iter_000000300`.
- [x] The 4-GPU `iter_000000600` readout/visual gate still failed. Generated-
      RGB readout reports mean final hole error `0.1023905702` m and mean
      future hole/peg/TCP RMSE `0.0475034488` / `0.0674232871` /
      `0.0660298159` m; target onsets remain far too early and static/peg-only
      samples still show false target drift. Visual sheets are cleaner but
      still show failed peg recovery, peg loss in insert-resume, contact
      geometry drift, and a severe peg-disturb artifact cloud.
- [ ] No controller or DP integration may be launched from the 4-GPU
      `iter_000000600` checkpoint.
- [x] The 4-GPU `iter_000000900` readout/visual gate still failed. Generated-
      RGB readout reports mean final hole error `0.1021599918` m and mean
      future hole/peg/TCP RMSE `0.0477525963` / `0.0668954306` /
      `0.0657577841` m; target onsets remain far too early and static/peg-only
      scenes still show false target drift. Visual sheets are cleaner than
      early checkpoints in several samples, but still fail peg recovery,
      insert-resume peg continuity, target-motion timing, and peg-disturb
      artifact quality.
- [ ] No controller or DP integration may be launched from the 4-GPU
      `iter_000000900` checkpoint.
- [x] The 4-GPU `iter_000001200` gate completed under the same 301-frame /
      300-action contract. Strict artifacts passed with mean action RMSE
      `0.6162060709` and mean future-video PSNR `20.2295638395` dB.
- [x] Current resource rule is explicit: because the 4-GPU SFT is healthy,
      there is no active or planned 1-GPU SFT fallback. The 1-H200 support
      resources are reserved for strict eval, generated-RGB readout/profile,
      sanity checks, and video/render work after checkpoints become available.
- [x] The 4-GPU `iter_000001200` readout/visual gate failed. Generated-RGB
      readout reports mean final hole error `0.1028519175` m and mean future
      hole/peg/TCP RMSE `0.0481315461` / `0.0667989680` / `0.0662642823` m;
      target onsets remain far too early and static/peg-only scenes still
      show false target drift. Visual sheets remain negative for target
      timing/final pose, insert-resume peg continuity, peg-drop regrasp, and
      peg-disturb recovery.
- [ ] No controller or DP integration may be launched from the 4-GPU
      `iter_000001200` checkpoint.
- [x] Historical pre-fix1 final checkpoint gate `iter_000001500` completed and
      failed handoff. It remains archived diagnostic evidence only: mean action
      RMSE `0.6161383192`, mean future-video PSNR `20.2640810606` dB, mean
      final hole error `0.1028195255` m, mean future hole/peg/TCP RMSE
      `0.0485962523` / `0.0669838827` / `0.0666590482` m, and mean future
      peg-head-hole RMSE `0.0516100994` m.
- [x] Current fresh fix1 final checkpoint gate `iter_000001500` completed
      using the same strict generated-video/action/readout/profile/visual
      protocol. Strict artifacts passed over 10 samples with every
      generated/reference video `301/301` frames and every action target
      `300x32/300x32`. Mean action RMSE is `0.3157882582`, robot-action future
      RMSE `0.8203911022`, state-sidecar future RMSE `0.0626562464`, and
      future-video PSNR `23.1641813684` dB.
- [x] Current fresh fix1 `iter_000001500` generated-RGB readout/profile passed
      strict structure but remains negative for active handoff. Mean final
      hole error is `0.0538804681` m; mean future hole/peg/TCP RMSE are
      `0.0301422454` / `0.0386116191` / `0.0347898968` m; mean future
      peg-head-hole RMSE is `0.0413906728` m. Target onsets remain far too
      early or false-positive on static/peg-only scenes.
- [x] All 10 current fresh fix1 final `iter_000001500` review sheets were
      opened and inspected. The visual gate is still not controller-ready:
      videos are coherent without old global collapse, but peg_drop,
      peg_disturb, and insert_resume do not prove reliable peg-gripper-hole
      contact continuity or executable DP-resume handoff.
- [ ] No controller or DP integration may be launched from the current fresh
      fix1 `iter_000001500` checkpoint. It is SFT progress evidence, not a
      valid world-model handoff checkpoint.
- [x] Post-final reference-RGB calibration sharpened the failure
      interpretation. The same task-state readout on GT reference RGB still
      produces early low-threshold target-onset false positives, so the
      current 2 mm displacement onset score is not a trustworthy switch
      signal. But generated RGB is much worse than reference RGB on final hole
      and peg/TCP geometry, so the visual/world rollout failure remains real.
- [x] A learned temporal target-motion head over readout trajectories confirms
      that target monitoring is partially recoverable when the RGB/readout
      quality is reference-like. It reaches held-out reference-RGB AUROC
      `0.9115353604` and F1@0.5 `0.7669897596`. Re-running the head on the
      current fresh fix1 generated-RGB readout gives AUROC `0.7808262378`,
      F1@0.5 `0.6005305040`, and best F1 `0.6179090483`; this is useful
      diagnostic progress, not permission to start controller or DP
      integration.

## Review Questions To Resolve Later

- [x] Exact prefix policy for the first SFT is the multi-mode full-episode mask:
      static monitor, target pre-motion, target-motion observed, target
      post-motion, insert-resume, and peg recovery.
- [x] The first active SFT uses the available
      `Cosmos3-Nano-Policy-DROID-DCP` constrained path.
- [x] The active convention is 301 RGB/state frames and 300 action steps.
- [x] The first validation panel is 10 samples selected from val to cover
      moving target, non-moving, peg disturbance, and peg drop scenarios.
