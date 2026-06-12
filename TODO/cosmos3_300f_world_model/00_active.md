# Active TODO

## Current Status

- [x] 2026-06-12 training correction: the v7_733 `normactive_clip1` SFT was
      stopped after config audit confirmed it did not carry the fix1 action
      recipe. The failure was not literally missing selected action tensors:
      optimizer selected `410` tensors including the adapters, but the run
      used the old bad recipe (`lr=2e-5`, `action_loss_weight=10.0`,
      `independent_action_schedule=false`, `shift_action=None`). The foreground
      tmux commands were interrupted with `Ctrl-C`; no Slurm allocation was
      cancelled.
- [x] Default Cosmos3 full-episode WAM training now enforces the fix1 action
      recipe unless an explicit diagnostic override is set. This is the
      default regression guard against repeating the action-adapter training
      mistake.
- [x] User visual approval received on 2026-06-12 for the two-sample overfit
      iter100 generated videos. Current overfit condition root is
      `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_overfit2_rgb_300step_20260612_1830`;
      current SFT root is
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_overfit2_rgb_300step_fix1recipe_4gpu_20260612_1840`;
      iter100 validation loss is `0.125564`; strict eval passed for both
      samples with `301/301` videos, `[300,32]` actions, and no strict
      failures. Generated videos are under
      `eval_full_episode_wam_iter_000000100/inference/*/vision.mp4`.
- [x] Code and documentation were committed/pushed, and the full v7_733 SFT was
      started from the frozen 733-row condition root using the enforced fix1
      action recipe. Do not use the rejected `normactive_clip1` recipe or any
      chunked/93-frame/128-frame path.
- [x] Code/docs were committed and pushed to `yanhr21/Reflex` branch `main` as
      commit `1bd4691`. Large local artifacts are ignored by `.gitignore` and
      were not committed.
- [x] Full v7_733 SFT restarted with the enforced fix1 recipe at
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`.
      It runs in tmux `cosmos3_sft_v7_733_full_fix1recipe_4gpu_126210` on
      Slurm step `126210.41` (`server56`, `4xH200`). The launch manifest
      reports `fix1_action_recipe_check=passed`, `lr=1.0e-4`,
      `action_loss_weight=2.0`, `independent_action_schedule=true`, and
      `shift_action=1`.
- [x] Startup sanity: training reached `Starting training...`, iter0 validation
      loss was `3.606580`, and rank-0 iteration 7 loss was `3.0716` with
      finite vision/action losses.
- [x] Iter300 strict eval/readout/profile completed in auxiliary allocation
      `127120` on `server40`. Structural gates passed for `10` samples:
      generated/reference videos are `301/301`, actions are `300x32`, and
      `strict_failures=[]`. Iter300 validation loss is `0.155843`; generated
      eval has mean future PSNR `21.6543`, robot-action future RMSE `0.6354`,
      and state-sidecar future RMSE `0.3534`. Generated-RGB readout/profile
      also passed structure but is not controller-ready: mean final hole error
      `0.0655` m, future hole/peg/TCP RMSE `0.0392/0.0401/0.0399` m, and
      future peg-head-hole RMSE `0.0318` m. Direct review of all `10` sheets
      found no old geometry-collapse/white-fog failure, but several final
      relative poses are still too imprecise for closed-loop handoff.
- [x] Iter600 strict eval/readout/profile completed in held auxiliary
      allocation `127120` on `server40`. Structural gates still pass for
      `10` samples: generated/reference videos are `301/301`, actions are
      `300x32`, and `strict_failures=[]`. However, it is worse than iter300
      on controller-facing metrics despite lower validation loss:
      validation loss `0.131243`, mean future PSNR `20.2910`, robot-action
      future RMSE `0.9831`, state-sidecar future RMSE `0.6805`, generated-RGB
      mean final hole error `0.1058` m, future hole/peg/TCP RMSE
      `0.0603/0.0795/0.0762` m, and future peg-head-hole RMSE `0.0457` m.
      The agent opened all `10` review sheets. There is no old global
      white-fog collapse, but several samples show block/robot relative-pose
      drift, peg/contact discontinuity, or target-position errors that make
      DP resume unsafe. Do not start controller/DP integration from iter600.
- [x] The live 4-H200 SFT allocation ended naturally at wall time. Slurm
      reported step `126210.41` cancelled at `2026-06-12 23:06:27 CST` due to
      time limit, not by agent `scancel`. The last rank-0 log reached
      iteration `743` with finite loss; no traceback/OOM/NaN marker was found.
      No checkpoint beyond `iter_000000600` was saved, so there is no iter900
      evidence. Preserve iter300 as the current best qualitative sanity
      checkpoint, record iter600 as controller-negative, and keep controller
      gated.
- [x] Stop the rejected 128-action / 129-frame chunked SFT job and its waiting
      action-eval/readout watcher sessions.
- [x] Move old method results, old evidence conclusions, and old logs out of
      the active `/public/home/yanhongru/ICLR2027` tree.
- [x] Move scripts whose names/default paths explicitly targeted the rejected
      chunked/128-action/129-frame chain to the external archive.
- [x] Move old Cosmos SFT/eval/watch/controller wrappers and legacy
      object-state/RGB-D-slot/controller method scripts to the external
      archive.
- [x] Move remaining legacy `rgbd`/`full96` script entry points to the
      external archive.
- [x] Preserve the approved full1000 RGB dataset.
- [x] Move full1000 source H5/specs into active `data/cosmos3/`.
- [x] Back up old PLAN/TODO directories under `_backup_*`.
- [x] Create the new active Cosmos3 300-step PLAN/TODO directories.
- [x] User review gate passed by user direction after plan review. Start
      execution, but keep controller/DP integration gated until SFT evidence,
      validation videos, metrics, and visual review exist.
- [x] The previous Qwen2.5 direct-video `60/60` gate is invalidated by user
      visual review and is not approval evidence. The rejected fix3 roots were
      moved outside the active repo under
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate`.
- [x] The constrained-insert v8 smoke is also rejected by user direct video
      review. User-identified failures: visible peg/target penetration, poor
      peg-hole alignment, and an unphysical final phase where the peg appears
      to drill/crawl into the hole instead of being inserted by robot motion.
      The old key-frame-only agent judgment is not evidence. The v8 root was
      moved outside the active repo under the same archive. No fix3 SFT has
      been started.
- [x] The fix2/official static insertion reproduction baseline is also
      rejected after user direct video review. User-identified failure:
      severe peg/target penetration in
      `0001_none_fix2_official_seed1002002_idx0001.fix2_traj_0.mp4`.
      That root was moved outside active experiments under
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate/fix2_repro_user_rejected_penetration_20260611`.
      It is invalid as a baseline, approval package, method evidence, or SFT
      source.
- [x] Current user correction: return to the originally effective 2026-06-06
      full1000 dataset/protocol as the physical baseline:
      `experiments/world_model_task_rebinding/cosmos3/sft_dataset_full1000_maniskill_default_regen_20260606_0055`.
      The expected behavior is target static at the beginning, peg grasp and
      alignment first, target motion after alignment, and true final
      insertion. Its remaining problems are small target-motion range and many
      failed final insertions.
- [x] Fix3 data action after original reset: audit the original full1000 source
      H5/protocol, then rebuild from a copied original-protocol generator until
      accepted smoke demos have true final insertion, late target motion,
      larger target displacement, and continuous-moving-target cases. If the
      physical gate rejects a trajectory, mark it failed and resample; never
      force insertion by state projection or penetration.
- [x] Copy-on-write constraint for the rebuild: do not modify the original
      full1000 dataset root, source H5 tree, original generation environment,
      or original generation scripts in place. The original 2026-06-06 chain is
      a read-only baseline. Any large-motion/fix3 regeneration must copy the
      needed generator/config into a new fix3-named script/output root, record
      the source commit/path, and write only to new experiment directories.
- [x] Run a read-only audit of the original 2026-06-06 full1000 H5 sources
      inside Slurm allocation `125642`; output:
      `experiments/world_model_task_rebinding/cosmos3/original_full1000_readonly_audit_20260611`.
      The audit confirms the original protocol is the correct physical
      baseline but not a direct SFT source for fix3: many rows are not final
      inserted (`hole_constant 33/167`, `hole_move_stop 21/167`,
      `hole_reverse 125/167`, `none 104/166`, `peg_disturb 2/166`,
      `peg_drop 43/167`), and moving-hole target motion is small
      (`~0.09-0.14m`).
- [x] Move rejected fix2 and old untrusted SFT/readout roots out of active
      `experiments/world_model_task_rebinding/cosmos3`. The active cosmos3
      directory now keeps only the original 6/6 full1000 dataset, the read-only
      audit, and render canaries.
- [x] Reconstruct a copied fix3 generator from the original-protocol H5
      provenance:
      `scripts/world_model/generate_cosmos3_fix3_original_protocol_large_motion_dp.py`.
      Do not use the rejected overlay postprocessor or the rejected
      late-trigger motion-planning script as the active baseline.
- [x] Current approval gate: v7 complete-nine smoke videos are rendered at
      `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21`.
      The agent inspected every all-frame framebook page for all nine videos
      (`301` frames each) and did not find target self-insertion, early target
      motion, wall insertion, visible penetration, or final failure. By the
      latest user continuation directive, this permits full1000 generation.
- [x] Latest 2026-06-12 user correction: keep the already accepted v7 H5
      samples and do not regenerate them. A fresh active-root filename scan
      after stopping the old generator steps reached `630/1000` unique H5
      rows (`647` raw, `17` duplicate) across v7 work buffers. These rows
      remain valid full1000 candidates subject to the normal final merge,
      structural audit, rendered review, and approval gates. They must not be
      relabeled as throwaway diagnostics merely because the remainder changes
      strategy.
- [x] Stop the current DP-success-filtered v7 generator steps after the user
      pointed out the semantic issue. The stopped generator used the
      frozen/static DP for all robot actions and accepted only cases where that
      DP still finished the dynamic episode. This is acceptable for already
      produced successful rows, but it is not sufficient as the only
      construction for the remaining hard dynamic data. The stop used
      allocation-internal process interrupts/termination, not `scancel`; the
      accepted H5 files were left in place.
- [x] 2026-06-12 hard-teacher direction recorded but deferred by latest user
      instruction. The active path is not the 1500/hard-teacher supplement.
      Current near-term target is the original v7 DP-generated `full1000`
      source set, then strict merge/audit/render approval, then Cosmos3 SFT.
      The hard-teacher script remains available for later work but must not
      block the current v7 DP full1000 chain.
- [x] Superseded for the current run: continue v7 DP full1000 generation from
      the latest live merge count
      `723/1000`. The merge now includes the already approved v7 complete-nine
      `fix3_h5_paths.txt` explicitly, because that combined review root points
      at H5 files under earlier smoke/search roots rather than storing H5
      locally. Current selected counts are
      `move_stop=43/70`, `constant=48/90`, `reverse=97/100`,
      `sine=60/90`, `continuous_insert=95/120`, `fast_shift=105/120`,
      `none=160/160`, `peg_drop=113/150`, and `peg_disturb=2/100`.
      Missing counts are `27/42/3/30/25/15/0/37/98`.
      Hard-teacher roots are excluded from this merge. Latest 2026-06-12 user
      override stopped data construction at the frozen `733` rows and moved
      directly to SFT; do not resume this full1000 fill unless explicitly
      reopened.
- [x] Add hard-teacher supplement entry point:
      `scripts/world_model/generate_cosmos3_fix3_hard_dynamic_teacher.py`.
      It is copied from the motion-planning teacher framework, extended to all
      nine classes, quota/seed-base driven generation, `robust_held` schema,
      `source_kind=hard_dynamic_teacher`, and explicit peg-drop/peg-disturb
      recovery by regrasping/inserting with the motion-planning teacher.
- [x] Stop the hard-teacher smoke attempts by process interrupt, not `scancel`,
      after the latest user direction to defer hard teacher. Held allocation
      `126223` was preserved and repurposed to v7 DP generation.
- [x] Start v7 DP resume generation on the held allocations:
      `126223` runs `nonpeg_resume_a`; `126210` runs
      `mixed_peg_resume_a`; `126219` runs peg-disturb supplement;
      `126174` runs peg-drop supplement; and `126175` runs a mixed non-peg
      supplement. New wrapper:
      `scripts/slurm/run_fix3_v7_resume_full1000_bundle_20260612_in_allocation.sh`.
      Early logs reached the generator; `hole_late_reverse` and `peg_drop`
      already accepted early rows, while `peg_disturb` is still being monitored
      for real accepts rather than assumed solved.
- [x] Latest 2026-06-12 user override: stop data construction immediately and
      proceed to Cosmos3 SFT. The active frozen source is
      `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733`
      with `733` unique H5 trajectories. Current counts are
      `move_stop=44`, `constant=48`, `reverse=99`, `sine=60`,
      `continuous_insert=96`, `fast_shift=105`, `none=160`,
      `peg_drop=119`, and `peg_disturb=2`. This is a user-approved override
      of the previous "complete 1000 then stop for approval" gate. Do not
      continue data generation unless the user explicitly reopens it.
- [x] Stop the active v7 generator processes by allocation-internal SIGINT,
      not `scancel`. The held allocations `126210`, `126219`, and `126223`
      were preserved for render/export/SFT work. A user-override strict H5
      audit using the frozen 733-row class counts passed with
      `strict_ok=true` and `num_failed_records=0` at
      `fix3_v7_dp_user_override_sft_source_20260612_733/strict_source_h5_audit_user_override_quota733`.
- [x] Render the frozen 733-row H5 source into an RGB SFT dataset root, then
      run full-episode WAM condition export/preflight/action-target audits and
      launch Cosmos3 SFT without waiting for a separate human approval gate.
      Output root:
      `experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612`.
      It has `733` videos (`661` train, `72` val), `512x512`, `30 fps`,
      `301` frames; train/val artifact inspections passed, and SFT is running
      from
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_4gpu_20260612_0245`.
- [x] Full1000 v7 complete-nine source generation ran as grouped,
      quota-capped roots instead of one sequential all-class job. This follows
      the latest user correction: nonuniform quotas are acceptable, but no
      class should be tiny; generate several classes in parallel; do not treat
      attempt-seed search itself as the blocker; when full1000 generation is
      complete, stop and wait for user approval. Current target quotas:
      `70/90/100/90/120/120/160/150/100` over
      `hole_late_move_stop`, `hole_late_constant`, `hole_late_reverse`,
      `hole_late_sine`, `hole_late_continuous_insert`,
      `hole_late_fast_shift`, `none`, `peg_drop`, and `peg_disturb`.
      Active roots:
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_moving590`,
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_static_peg410`,
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_peg_aux250`,
      and
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_moving_hard_aux260`.
      Additional quota buffers started after early low-count monitoring:
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_none_aux160`,
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_move_sine_aux160`,
      and
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_pegdist_aux100`.
      Latest 2026-06-11 user correction: with the merged count at `276/1000`
      unique H5 rows, do not let peg/peg-disturb block the rest of the source
      generation. Stop only the peg-focused run steps, preserve their held
      Slurm allocations, and use those allocations to finish the non-peg
      moving-hole quotas first. Peg-drop/peg-disturb remain required complete-
      nine classes, but should be handled after the non-peg classes are filled.
      Four non-peg buffer roots were started from the preserved allocations:
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_cfs_aux330_seedshift4`,
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_mrs_aux260_seedshift3`,
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_cfs_aux330_seedshift5`,
      and
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_mrs_aux260_seedshift4`.
      A later full active-root filename scan reached `305/1000` unique rows
      when already-generated valid `peg_drop` rows from the stopped pegmix root
      were included. Because the six non-peg moving-hole classes still had a
      combined `512`-row missing count and all held allocations were busy, two
      extra tmux-managed 1-H200 allocations were requested and started:
      `126174` running
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_cfs_aux330_seedshift6`
      and `126175` running
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_mrs_aux260_seedshift5`.
      Both reached the generator log after model load; early rows are being
      filtered by the existing physical gates. The latest active-root filename
      scan reached `340/1000` unique rows. This is higher than the user's
      latest merged count of `276/1000` because accepted rows have landed in
      work-buffer roots before the merged path list is refreshed. The six
      non-peg moving-hole classes still have `477` missing rows, so all held
      allocations remain focused on non-peg generation and peg-drop/peg-disturb
      remain deferred. A later scan reached `346/1000` unique rows with
      `471` non-peg moving-hole rows still missing. Because the active mixed
      CFS roots are accepting `hole_late_fast_shift` more often than
      `hole_late_constant`/`hole_late_continuous_insert`, a focused 4-GPU
      bundle script was added to run four existing-wrapper roots for
      continuous, constant, move-stop, and sine/reverse if resources start:
      `scripts/slurm/run_fix3_v7_nonpeg_focus4_bundle_in_allocation.sh`.
      The script passed `bash -n` inside Slurm allocation `126174`, and tmux
      allocation `126188`
      (`cosmos3_fix3_full1000_nonpeg_focus4_4h200_0611`) started on
      `server42`. Startup exposed a Slurm-internal step serialization issue
      from nested non-overlap `srun`, so the approved generation wrapper now
      has optional `SRUN_EXTRA_ARGS` support and the focus bundle defaults to
      `SRUN_EXTRA_ARGS=--overlap`; default wrapper behavior is unchanged. The
      running focus allocation has `hole_late_continuous_insert` plus
      additional focus2 roots for constant, move-stop, and sine/reverse. The
      latest active-root count including focus roots reached `366/1000`
      unique rows, with `451` non-peg moving-hole rows still missing. A later
      scan reached `368/1000` unique rows and `449` non-peg rows still
      missing. The stale local non-overlap `srun` waiters from the original
      focus4 bundle were terminated by PID only after the replacement focus2
      roots were running; `126188` and all useful generation steps were
      preserved, and no `scancel` was used. A later GPU-process check showed
      that the focus overlap steps were all landing on logical GPU `0`, so a
      pinned allocation-internal launcher was added:
      `scripts/slurm/run_fix3_v7_nonpeg_focus3_pinned_in_allocation.sh`.
      It uses one `gpu:4` overlap step and explicitly launches additional
      focused roots on `CUDA_VISIBLE_DEVICES=1/2/3` for constant, sine, and
      continuous. The generator and gates are unchanged. `nvidia-smi`
      confirmed the new `.venv/bin/python` processes are distributed across
      the other GPUs. The latest active-root count before pinned roots accepted
      rows reached `378/1000`, with `439` non-peg moving-hole rows still
      missing. A later live scan reached `403/1000` active-root unique H5
      rows while the user-facing merged count remained `276/1000`; this gap is
      expected because buffer roots are not yet folded into the final merged
      path list. Current active-root class counts are
      `move_stop=18/70`, `constant=12/90`, `reverse=45/100`,
      `sine=20/90`, `continuous_insert=24/120`, `fast_shift=57/120`,
      `none=161/160`, `peg_drop=66/150`, and `peg_disturb=0/100`.
      The six non-peg moving-hole classes still have `414` rows missing.
      Resources remain focused on non-peg generation first; `peg_drop` and
      `peg_disturb` are intentionally deferred until the other classes are
      filled. Because every useful held allocation was already busy and the
      non-peg gap remained large, a new disjoint-seed focused launcher was
      added:
      `scripts/slurm/run_fix3_v7_nonpeg_focus4_pinned2_in_allocation.sh`.
      It pins four extra v7 generator roots to a new 4-H200 allocation for
      `hole_late_constant`, `hole_late_continuous_insert`, `hole_late_sine`,
      and `hole_late_move_stop`; it does not change the generator or physics
      gates. The script passed `bash -n` inside allocation `126188`, and tmux
      session `cosmos3_fix3_focus4b_pinned_4h200_0611` started Slurm job
      `126210` on `server56`. Follow-up checks confirmed one generator process
      on each of the four GPUs, all four focus4b logs reached the generator,
      and `hole_late_sine` accepted the first focus4b row. A following
      active-root scan reached `421/1000` unique buffer H5 rows with `396`
      non-peg moving-hole rows still missing. A later check found no done
      marker, no active traceback/CUDA/Vulkan/OOM/filesystem/segmentation
      error, and eleven held allocations still running with active steps. The
      latest active-root scan reached `425/1000` unique buffer H5 rows:
      `move_stop=20/70`, `constant=12/90`, `reverse=47/100`,
      `sine=24/90`, `continuous_insert=29/120`, `fast_shift=66/120`,
      `none=161/160`, `peg_drop=66/150`, and `peg_disturb=0/100`.
      The six non-peg moving-hole classes still have `392` rows missing, so
      keep current resources on non-peg generation and do not start peg-only
      work yet. Final-merge tooling was prepared while generation continued:
      `scripts/world_model/merge_fix3_full1000_unique_h5.py` remains the
      file-level quota/dedup merge, and the new
      `scripts/world_model/audit_fix3_merged_source_h5.py` is the read-only
      strict source H5 audit before render expansion/WAM export/SFT. The new
      audit script passed `py_compile` inside allocation `126188` and passed
      the approved v7 complete-nine smoke canary with `strict_ok=true`,
      `num_paths=9`, and `num_failed_records=0` under
      `fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/strict_source_h5_audit_20260611`.
      A later active-root scan reached `449/1000` unique buffer H5 rows:
      `move_stop=24/70`, `constant=19/90`, `reverse=50/100`,
      `sine=28/90`, `continuous_insert=35/120`, `fast_shift=66/120`,
      `none=161/160`, `peg_drop=66/150`, and `peg_disturb=0/100`.
      The six non-peg moving-hole classes still have `368` rows missing; keep
      non-peg source generation running and defer peg-only work. A later
      read-only continuation check reached `455/1000` unique buffer H5 rows:
      `move_stop=25/70`, `constant=19/90`, `reverse=51/100`,
      `sine=29/90`, `continuous_insert=38/120`, `fast_shift=66/120`,
      `none=161/160`, `peg_drop=66/150`, and `peg_disturb=0/100`.
      The six non-peg moving-hole classes still have `362` rows missing. All
      eleven held allocations still have active steps; no error-scan hits or
      done markers were found. Continue non-peg generation.
      Do not start WAM export, render expansion, or Cosmos3 SFT after the
      source count completes until the user approves.
- [ ] Current data-generation priority: finish the non-peg moving-hole quotas
      before returning to peg-drop/peg-disturb. `peg_disturb` remained `0/100`
      at the `276/1000` merged-count check and is now explicitly deferred by
      user direction rather than treated as the active blocker. Previous
      evidence still matters for later peg debugging: accepted/rejected
      outcomes are not determined by the env seed alone because the DP action
      sampler uses the global torch random stream. The old v5b
      `peg_disturb_seed1051032` succeeded when it appeared at global attempt
      32, but the same seed failed when placed at attempt 1/4 in grouped
      full1000 runs with a different diffusion action RNG state. The
      peg-focused local run steps were stopped with interruption/termination of
      the local `srun` processes only; the Slurm allocations were preserved and
      repurposed to non-peg generation. Do not use `scancel` to solve this
      kind of step-level issue.
- [x] During the grouped full1000 startup, a real seed-stream bug was found and
      fixed: priority smoke seeds could be reused later by the normal
      scenario seed stream. Duplicate-seed and excessive-offset partial roots
      were moved out of active experiments under
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_duplicate_seed_partial`
      and
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_offset_seed_partial`.
      The active generator now enumerates original scenario seeds while
      skipping priority values, preserving distribution without duplicate
      scenario/seed rows.

## Active Contract

- full1000 data is accepted and does not need regeneration.
- The accepted full1000 generation code/environment/data are read-only
  baseline assets. Do not edit them in place while rebuilding fix3; copy them
  first and modify only the copied fix3 generator/config.
- The rollout is a 300-step episode; including frame 0 gives 301 RGB/state
  frames.
- Training/testing must use the full episode/equal-length contract.
- No 129-frame video clips, 128-action chunks, 93-frame outputs, cropped
  metrics, or stale checkpoints may be active method evidence.
- SFT training must run on Slurm/GPU, not the login node.
- Do not run probe/export/preflight/syntax-test/debug tasks on the login node.
  The login node may be used only for file edits, starting tmux/salloc
  sessions, downloads, and reading status. Run all data export, preflight,
  script tests, rendering, training, validation generation, and debugging
  inside a Slurm allocation.
- Prefer holding GPU resources through tmux-managed `salloc` for repeated
  debugging and long monitoring. Use long enough allocations when resources are
  available. Avoid one-shot `sbatch` as the default path because queued
  dependency chains waste time when a held interactive allocation can do the
  work.
- Training evidence must use at least 1 GPU for at least 1 hour, and should
  continue until validation no longer clearly improves. Prefer 4 or 8 GPUs
  when schedulable; use 1 GPU to make progress if larger jobs would sit idle.
- Validation must render/generate videos for agent inspection and human review
  backup.
- If a concrete blocker repeats and cannot be resolved safely, stop for user
  direction under the goal-blocked rule instead of guessing.

## Next After Review

- [x] Write the clean full-episode action/proprio/state condition exporter and
      strict preflight script for approved RGB videos plus active source H5s.
- [x] Write an allocation-only wrapper for export, preflight, action-target
      audit, and SFT startup.
- [x] Wait for the tmux-held `salloc` allocation, then run exporter and script
      checks inside the allocation.
- [x] Run strict preflight over all 1000 source episodes.
- [x] Inspect a small validation contact-sheet panel. Source review sheet
      `sft_dataset_full1000_maniskill_default_regen_20260606_0055/review_sheets/0000_hole_constant_seed702000_n167_traj_0_traj_0_review_sheet.png`
      was opened on `2026-06-10`: the approved default-view RGB is readable,
      full-episode ordered, and shows robot, peg, target block, and target hole
      clearly. This is data-visual sanity only, not model success evidence.
- [x] Start SFT inside the held allocation only after the above passes.
- [x] Historical pre-fix1 4-GPU SFT validation monitoring reached the
      configured final checkpoint `iter_000001500` and is preserved as
      negative diagnostic evidence only. That older chain passed strict
      301-frame / 300-action accounting but failed the generated-RGB
      readout/visual handoff gate. It must not be confused with the current
      overfit-validated full1000 fix1 run below.
- [x] Prepare the post-SFT full-episode validation generation path before the
      first checkpoint: allocation-only eval runner plus strict eval-input and
      artifact-inspection scripts. The prepared val input set has 10 samples
      covering `hole_move_stop`, `hole_reverse`, `hole_constant`, `peg_drop`,
      `peg_disturb`, and `none`, with all inputs still under the 301/300
      contract.
- [x] After the 4-GPU run produced real losses and passed startup sanity, the
      duplicate 1-GPU SFT step was stopped. The old 1-GPU allocation `122736`
      later ended and was revoked; its stale tmux session has been removed so
      it cannot be mistaken for an active SFT run.
- [x] Per the latest user instruction, keep one-GPU support capacity for
      readout, sanity checks, and video/render work, not SFT fallback. A new
      tmux-held auxiliary 1-H200 allocation `123366`
      (`cosmos3_aux_1h200_0610`) is running on `server21`. CUDA canary passed
      through `srun --jobid=123366`; iter900, iter1200, and iter1500
      readout/profile gates completed as negative diagnostics. After the final
      4-GPU gate completed, idle allocation `123131` and the extra eval
      allocation `122782` were released. The old auxiliary allocation
      `123366` was later revoked after its watcher work completed, so a new
      tmux-held 1-H200 allocation `123381` is now running on `server21` for
      later sanity checks, readout, video, or render work. No 1-GPU SFT is
      active or planned.
- [x] Two-sample overfit fix1 passed the user's visual sanity check. The
      inspected checkpoint `iter_000000100` produced strict same-length
      artifacts for both overfit rows (`301/301` video frames and
      `300x32/300x32` actions), and the user confirmed the videos are good
      enough to stop spending time on overfit.
- [x] The overfit srun step was stopped with tmux `Ctrl-C`, preserving the
      4-H200 allocation. No `scancel` was used. Full1000 fix1 SFT is now
      running on the same allocation `123385` on `server32` as tmux session
      `cosmos3_full_fix1_4gpu_20260610`, Slurm step `123385.7`, output root
      `sft_full_episode_wam_full1000_rgb_300step_fix1_4gpu_20260610_1131`.
      Preflight passed with `strict_alignment_ok=true`, action target audit
      passed with `strict_action_target_ok=true`, the optimizer selected the
      action adapter modules (`410` tensors / `6,982,401,216` elements), and
      the run entered 4-GPU training/startup validation from the
      `Cosmos3-Nano-Policy-DROID-DCP` base checkpoint.
- [x] Full1000 fix1 startup sanity is now real training, not only process
      launch. Iteration-0 validation loss was `4.279975`; training iterations
      `1-4` logged finite losses, finite grad norms, low vision loss
      (`~0.06-0.13`), and action-dominated loss (`~1.44-4.91` action
      component on the inspected ranks). The run is still under the strict
      `301` RGB frame / `300x32` action contract and has not produced a
      checkpoint yet.
- [x] Current full1000 fix1 status after overfit success: the active 4-H200
      SFT continues in preserved allocation `123385`, step `123385.7`, without
      any overfit job still running. The validation curve is `4.279975`
      (iter0), `0.947078` (iter100), `0.756682` (iter200), `0.787787`
      (iter300), `0.733358` (iter400), `0.746084` (iter500), `0.718841`
      (iter600), `0.725782` (iter700), `0.641040` (iter800), and `0.634057`
      (iter900). Iter900 is the current best, so training must continue to
      later gates rather than stopping early.
- [x] Iter600 checkpoint saved at `2026-06-10 14:56:48 CST`; iter600
      validation loss was `0.718841`, the best point so far. The SFT run
      resumed at iter601 with low finite loss, so training should continue
      beyond iter600 while generated eval/readout evidence is inspected.
- [x] Iter600 strict eval input construction passed in aux allocation
      `123499`: 10 samples, scenario-diverse, `301` expected RGB frames,
      `300` expected action steps, action dim `32`, no rejected rows.
- [x] Iter600 generated artifact inspection also passed strict accounting for
      all 10 samples. Mean future-video PSNR is `22.6652227928`, mean action
      RMSE `0.3673552634`, mean robot-action future RMSE `0.9428412691`, and
      mean state-sidecar future RMSE `0.0746684971`.
- [x] Iter600 visual review is complete for all 10 review sheets. The generated
      videos are visibly healthier than the old failed chain and cleaner than
      iter300 on most static/target-motion cases, with no global white/fog or
      transparent-frame collapse. Peg-drop/regrasp and controller-ready
      handoff remain unproven until readout/profile finishes.
- [x] Iter600 generated-RGB readout/profile completed with strict structure
      passing. Mean final hole error improved to `0.0613948691` m; mean future
      hole/peg/TCP RMSE are `0.0328006673` / `0.0435265400` /
      `0.0433667297` m; mean future peg-head-hole RMSE is `0.0449233321` m.
      This is progress over iter300 but still not controller-ready, because
      target-motion onset is still predicted far too early and static/peg-only
      samples still trigger false target-motion onset under simple threshold
      scoring.
- [x] Iter300 full-episode eval passed strict artifact accounting for all 10
      samples and the agent visually inspected all review sheets. This is
      encouraging world-model training evidence, but not controller/DP
      handoff evidence: generated-RGB readout still has mean final hole error
      `0.0879152346` m and unreliable early target-motion onsets.
- [x] Iter900 full-episode eval/readout completed under the strict contract:
      10/10 generated/reference videos are `301/301` frames, actions are
      `300x32/300x32`, strict artifact/readout failures are empty, mean action
      RMSE improved to `0.2946495338`, mean robot-action future RMSE improved
      to `0.7637177886`, mean state-sidecar future RMSE improved to
      `0.0684030772`, and mean future-video PSNR is `22.6316859914`.
- [x] Iter900 visual review is complete for all 10 review sheets. The result
      remains visibly healthy relative to the old failed chain: no global
      white/fog/transparent collapse, stable static and insert-resume panels,
      coherent target/robot/peg geometry on target-motion panels, and cleaner
      peg-disturb behavior. Boundary: peg-drop/regrasp is still not proven as
      executable controller handoff evidence.
- [x] Iter900 generated-RGB readout/profile passed strict structure with mean
      final hole error `0.0678502409` m, mean future hole/peg/TCP RMSE
      `0.0335725300` / `0.0417441050` / `0.0375097505` m, and mean future
      peg-head-hole RMSE `0.0416219164` m. Action/state and peg/TCP
      diagnostics improved over iter600, but final/future hole readout is
      slightly worse than iter600 and target-motion onset remains too early
      with false positives on static/peg-only samples. Therefore this is not
      controller/DP integration evidence.
- [x] Continue the current overfit-validated full1000 fix1 4-H200 SFT through
      the configured final checkpoint while validation/generation evidence is
      collected. Iter1100 remained the best validation point (`0.607350`) after
      the iter1200/1300/1400/1500 rebounds, and final iter1500 strict
      generated-video/action/readout/visual evidence is recorded below. No
      overfit srun was resumed and no allocation was cancelled.
- [x] Iter1200 strict eval/readout watchers completed in the auxiliary 1-H200
      allocation `123499`, not on the login node. Eval watcher step `123499.8`
      generated strict artifacts, and readout/profile watcher step `123499.7`
      wrote `task_state_readout_best_current` and
      `readout_failure_profile.json`. The active 4-H200 SFT step `123385.7`
      continued independently.
- [x] To avoid future confusion, the preserved 4-H200 allocation display name
      was changed from the stale overfit name to
      `cosmos3_full1000_fix1_4h200_0610`, and the allocation shell tmux session
      was renamed to `cosmos3_4h200_full1000_hold_20260610`. No allocation was
      cancelled and the active SFT step remains `123385.7`.
- [x] Iter1000 validation completed at `0.697886`, rebounding from the iter900
      best `0.634057`. This is not a checkpoint gate and not a reason to stop:
      the run resumed normally at iter1001/1002 with finite low losses. Keep
      training to the iter1200 strict generated-video/action/readout/visual
      gate before making a stop or controller/DP decision.
- [x] Iter1100 validation completed at `0.607350`, a new best over iter900 and
      iter1000. The run resumed normally at iter1101/1102 with finite low
      losses. Continue to iter1200 checkpoint/eval because validation is still
      improving and no strict generated-video/action/readout/visual gate exists
      for iter1100.
- [x] Iter1200 checkpoint saved at `2026-06-10 18:15:30 CST`. Iter1200
      validation loss later reported `0.675294`, worse than the iter1100 best,
      but the run resumed normally at iter1201 and this checkpoint is now the
      next strict generated-video/action/readout/visual evidence gate.
- [x] Iter1200 strict generated artifact inspection passed for all 10 samples:
      generated/reference videos are `301/301` frames, actions are
      `300x32/300x32`, strict failures are empty, mean action RMSE is
      `0.3327156417`, mean robot-action future RMSE is `0.8618907214`, mean
      state-sidecar future RMSE is `0.0662339055`, and mean future-video PSNR
      is `22.7099074529`.
- [x] Iter1200 visual review is complete for all 10 review sheets. The
      generated rollouts remain readable and coherent, with no return of the
      old white/fog/transparent collapse. Target-motion/static/insert-resume
      geometry is usable as SFT progress evidence, but peg-drop/regrasp is
      still not a reliable executable handoff by visual inspection.
- [x] Iter1200 generated-RGB readout/profile completed with strict structure:
      mean final hole error `0.0620453945` m, mean future hole/peg/TCP RMSE
      `0.0323648744` / `0.0414618213` / `0.0359076100` m, and mean future
      peg-head-hole RMSE `0.0419642333` m. This is not controller-ready:
      target-motion onset remains early on moving samples and still false-fires
      on static/peg-only samples under the current displacement profile.
- [x] Iter1300 validation completed at `0.694085`, still worse than the
      iter1100 best and slightly worse than iter1200. This is not a checkpoint
      gate and not a reason to start controller work. The SFT run resumed
      normally at iter1301/1302 with low finite losses, so continue to the
      configured iter1500 strict generated-video/action/readout/visual gate.
- [x] Iter1400 validation completed at `0.625498`, improving over iter1200 and
      iter1300 but still worse than the iter1100 best `0.607350`. The run
      resumed normally at iter1401 with finite low loss. Keep waiting for the
      iter1500 checkpoint because validation alone is not generated-video or
      controller evidence.
- [x] Iter1500 checkpoint saved at `2026-06-10 19:54:51 CST`, final
      validation completed at `0.662956`, and the trainer reported done. This
      final validation is worse than the iter1100 best and is not enough for a
      method claim by itself.
- [x] Iter1500 final gate completed in aux allocation `123499`, not on the
      login node. Strict generated artifact inspection passed for all 10
      samples (`301/301` videos, `300x32/300x32` actions, no strict failures);
      mean action RMSE `0.3157882582`, robot-action future RMSE
      `0.8203911022`, state-sidecar future RMSE `0.0626562464`, and
      future-video PSNR `23.1641813684`.
- [x] Iter1500 generated-RGB readout/profile completed with strict structure:
      mean final hole error `0.0538804681` m, mean future hole/peg/TCP RMSE
      `0.0301422454` / `0.0386116191` / `0.0347898968` m, and mean future
      peg-head-hole RMSE `0.0413906728` m.
- [x] Iter1500 visual review is complete for all 10 review sheets. The
      videos remain coherent and do not show the old global collapse. However,
      peg_drop/peg_disturb/insert_resume still do not prove reliable
      peg-gripper-hole contact continuity, and the target-onset profile still
      fires too early or falsely on static/peg-only samples. Do not start
      controller/DP integration from this checkpoint as method evidence.
- [x] Re-ran the calibrated target-motion head diagnostic on the current fresh
      fix1 iter1500 generated-RGB readout inside aux allocation `123499`, step
      `12`. Output root:
      `target_motion_readout_calibration_fresh_fix1_iter1500_20260610`.
      Held-out reference RGB remains much stronger (AUROC `0.9115353604`,
      F1@0.5 `0.7669897596`, best F1 `0.7788987602`) than current fresh
      generated RGB (AUROC `0.7808262378`, F1@0.5 `0.6005305040`, best F1
      `0.6179090483`). This confirms calibrated switching is feasible only
      when readout quality is reference-like; the current generated rollout is
      still not a controller-facing switch input.
- [x] Failure localization selected a concrete repair that preserves the full
      `301` RGB frame / `300` action-state row contract. The main training
      target bug was that fix1's 25-D state sidecar repeated the prefix
      TCP/peg/hole/contact state for every future step, so Cosmos was not
      directly supervised to output future target/peg/TCP/contact state in the
      action branch. See
      `TODO/cosmos3_300f_world_model/06_post_fix1_repair.md` and
      `docs/world_model_task_rebinding/2026-06-10_cosmos3_fresh_fix1_failure_localization.md`.
- [x] Fix2 code is prepared: full-episode exporter now supports
      `sidecar_target_mode=future_aligned_state`, the Slurm wrapper passes it
      explicitly, action-target audit checks future sidecar variation, and an
      optional role-weighted JSONL helper can repeat full-episode rows without
      clipping or regeneration.
- [x] User inspection exposed a real source-data flaw: many old full1000
      validation/reference videos do not end inserted. A Slurm-side metadata
      audit confirmed poor `inserted_end` rates across several scenarios and
      narrow target motion. Therefore fix1/fix2 runs on that source are
      negative diagnostics only, not active method evidence.
- [x] Stop continuing fix2 SFT on the invalid old full1000 source. The stop was
      due to a concrete data invalidation, not a convenience stop.
- [x] Add fix3 plan/TODO/evidence:
      `PLAN/cosmos3_300f_world_model/06_fix3_successful_large_motion_data.md`,
      `TODO/cosmos3_300f_world_model/07_fix3_successful_large_motion_data.md`,
      and
      `docs/world_model_task_rebinding/2026-06-10_cosmos3_fix3_data_reset.md`.
- [x] Add a hard code boundary against the invalid static-terminal bootstrap:
      the current fix3 dynamic postprocessor refuses direct official-static
      replay input unless explicitly marked as a non-method diagnostic.
- [x] Implement and structurally smoke-test the correct fix3 source generator:
      widened/new final target poses, successful expert/manual insertion at
      those poses, large/varied target motion, and strict all-success 301/300
      source gates before any new SFT. The 6-sample smoke set passed H5/source
      audit with `inserted_end=true`, `301` state frames, `300` actions, and
      `0.208-0.321m` target motion.
- [x] Render default-view 30 fps smoke videos/contact sheets on a
      render-capable Slurm node and inspect them before scaling. `server32`
      and `server13` were concrete SAPIEN/Vulkan/CUDA failures, but allocation
      `125385` on `server35` passed the render canary and rendered the 6-sample
      smoke set. The agent opened all six smoke review sheets and verified
      readable large target motion plus final insertion.
- [x] Rebuild fix3 from the fix2/full-episode physical construction logic.
      User inspection rejected the previous `review60` overlay package:
      target motion starts at the beginning, visual insertion can look like
      inserting into the wall, the target effectively comes to the peg, and
      the motion is too slow. The corrected data must trigger target motion
      only after peg grasp and near-hole prealignment, keep the final target
      pose inside the static DP/expert reachable domain, force a late
      rebinding/prediction challenge, and pass real physical/visual insertion
      review. A first late-trigger physical review60 source root was generated:
      `experiments/world_model_task_rebinding/cosmos3/fix3_late_trigger_review60_20260610_6x10`.
      It is now rejected by user inspection on 2026-06-11. User-identified
      videos including `0001_hole_late_constant...mp4` and
      `0002_hole_late_reverse...mp4` still visibly insert into the block
      side/wall, and the trigger/alignment protocol is not the fix2-style
      "about to insert" protocol. The old numeric H5/source audit is therefore
      insufficient and invalidated as approval evidence. The later
      constrained-insert physical-gate v8 smoke is also rejected because it
      introduced visible penetration and peg self-drilling artifacts. A static
      fix2/official reproduction baseline was then generated/rendered under
      `fix2_official_insert_repro_smoke6_20260611_server56_padded301`. The next
      valid action was reset again to the originally effective 2026-06-06
      full1000 protocol and then implemented as a copy-on-write
      original-protocol generator.
- [x] Render the current v7 complete-nine original-protocol smoke package on
      Slurm allocation `125951` / `server21` after the user rejected v4 for
      excessive target speed and target self-insert semantics. H5 and render
      audits report `failures=[]`; all videos are `512x512`, `30 fps`, and
      `301` frames. This is not user approval.
- [x] The v7 complete-nine source protocol is the current approved scale-up
      basis after the user's 2026-06-11 direct-video review. Scale-up is
      limited to full1000 source H5 generation only.
- [x] The old full1000 source-generation gate is superseded for the current
      run by the 2026-06-12 user override. Data generation was stopped at the
      frozen `733`-row v7 DP-success-filtered source, and the held allocations
      were preserved rather than cancelled. Do not continue full1000
      generation unless the user explicitly reopens it.
- [x] The old "stop after exactly 1000 and wait for approval" gate is
      superseded for the current run. The user explicitly approved moving
      directly from the frozen `733` rows to render/WAM export/Cosmos3 SFT
      after visual review.
- [x] Add a code-level guard for that gate: the normal 300-frame full-episode
      SFT wrapper refuses invalid old 6/6 full1000 source data by default, and
      refuses fix3 `RUN_SFT=true` unless an approval file explicitly contains
      `approved_for_sft=true` or `user_approved=true`.
- [x] Frozen v7 `733` source was rendered into RGB SFT data at
      `experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612`
      using the approved ManiSkill default human camera, `512x512`, `30 fps`,
      `301` frames, and frame stride `1`. Structural inspection passed with
      `valid=true`; train/val video artifact checks passed for `661/72`
      videos, all readable and nonblank.
- [x] Agent visual review opened representative review sheets across
      `hole_late_move_stop`, `hole_late_constant`, `hole_late_sine`,
      `hole_late_continuous_insert`, `hole_late_fast_shift`, `none`,
      `peg_drop`, and both `peg_disturb` rows. User also reported the videos
      looked acceptable. This is data/render sanity evidence, not a final
      hard-dynamic method claim.
- [x] Full-episode WAM condition export/preflight/action-target audit passed
      for the frozen `733` source at
      `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_rgb_300step_20260612_0245`
      and
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_4gpu_20260612_0245`.
      The wrapper preserves the `301` RGB/state frame and `300` action row
      contract; action targets are `300x32`; `strict_action_target_ok=true`.
- [x] Cosmos3 SFT has started in tmux session
      `cosmos3_sft_fix3_v7_733_4gpu_126210` on Slurm allocation `126210`
      (`server56`, `4xH200`). The run loaded
      `checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID-DCP`, initialized the
      train/val dataloaders, entered `Starting training...`, and each GPU is
      allocated roughly `25-26GB`.
- [x] Cleaned the active cosmos3 experiment root after render/SFT start. Kept
      only current source/render/condition/SFT roots, the v7 Complete9 review
      root, the original 6/6 baseline/audit, and approval/provenance files.
      Moved `93` old process directories and `84` scattered logs/lists to
      `/public/home/yanhongru/ICLR2027/archived/reflex_cosmos3_process_artifacts_after_fix3_v7_733_sft_20260612`.
      Nothing was deleted.
- [ ] Current active SFT monitor is the full v7_733 fix1-recipe run
      `sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`.
      The rejected `normactive_clip1` full-data run is historical negative
      evidence. Controller/DP integration remains gated until the new
      full-data checkpoints pass strict generated-video/action/readout and
      visual review.
- [ ] Future closed-loop work must follow
      `TODO/cosmos3_300f_world_model/08_receding_closed_loop.md`: no one-shot
      300-step open-loop Cosmos execution, no sidecar/oracle simulator state,
      de-normalize only robot-action columns before live `env.step`, and
      execute short `<=8`-step prefixes with real re-observation.
