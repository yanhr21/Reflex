# Fix3 Late-Trigger Successful Dynamic Data TODO

## Current Status

- [x] Stop continuing fix2/fix1 as method evidence after the user found many
      validation/reference videos do not end inserted.
- [x] Audit the old full1000 source metadata inside Slurm allocation: the old
      data has many `inserted_end=false` samples and target motion mostly in
      the `0.08-0.14m` range for moving-hole families.
- [x] Mark the naive static-terminal bootstrap as invalid. The postprocessor
      now refuses direct official-static replay input unless explicitly marked
      as a non-method diagnostic.
- [x] Rebuild fix3 from the fix2/full-episode construction logic with
      late-trigger physical target motion. The previous widened-final-pose
      overlay review package is rejected by user inspection and must not be
      used for SFT.
- [x] Generate a small smoke set in a Slurm allocation, not on the login node.
      The source generator produced 2 successful widened-final-pose expert
      trajectories, replayed them to `pd_ee_delta_pose`, and the dynamic
      postprocessor produced one smoke trajectory for each new large-motion
      scenario.
- [x] Rendered the old overlay smoke videos/contact sheets and inspected them.
      Attempts on `server32` hit SAPIEN/Vulkan `ErrorDeviceLost` or hung
      during first render, and attempts on `server13` hit CUDA
      initialization/render-device failures. The render-capable allocation
      `125385` on `server35` passed canary and rendered the 6-sample smoke set
      at `1024x1024`, `30 fps`, `301` frames. The agent opened all six smoke
      review sheets, but the later user inspection rejected the construction
      semantics and visual physical correctness.
- [x] Run a new strict source audit for the late-trigger physical rebuild:
      final inserted under real ManiSkill episode success, `301` state frames,
      `300x7` `pd_ee_delta_pose`-style action labels, target motion starts
      only after peg grasp and near-hole prealignment, target motion completes
      quickly enough to be visually clear, and no failed/insert-wall demos.
      The previous late-trigger review60 audit is invalidated by user visual
      review on 2026-06-11: samples including
      `0001_hole_late_constant...mp4` and `0002_hole_late_reverse...mp4`
      visibly insert into the block side/wall, so `has_peg_inserted` plus the
      old peg-head gate is not sufficient evidence.
- [ ] After user approval, scale to full fix3 training source count and render
      RGB. This has not been claimed complete: the current package is the
      mandatory per-scenario human-review gate, not full training-scale method
      evidence.
- [x] Rendered exactly `10` demo videos per old overlay scenario, but this
      package is now rejected and historical only. The rejected package is
      `experiments/world_model_task_rebinding/cosmos3/fix3_dynamic_overlay_review60_render_server35_20260610_6x10`,
      with approval index
      `fix3_review60_approval_index.md` / `.json`.
- [x] The replacement `6x10` Qwen2.5 VLM gate is now rejected and historical,
      not approval evidence. User visual review found initial penetration,
      misalignment, and side/wall insertion in samples including
      `0000_hole_late_constant...mp4`, `0003_hole_late_move_stop...mp4`, and
      `0004_hole_late_constant...mp4`. The previous root
      `experiments/world_model_task_rebinding/cosmos3/fix3_late_trigger_smoke60_20260611_official_insert_vlm_gate`
      was moved out of the active repo under
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate`.
- [x] Stop at the approval gate after per-class demos are rejected. Do not
      start SFT until replacement rendered demos pass user approval. The later
      constrained-insert physical-gate `6`-smoke package was also rejected by
      user direct video review; no fix3 SFT was launched.
- [x] The later fix2/official static repro package is also rejected by user
      direct video review: sample
      `0001_none_fix2_official_seed1002002_idx0001.fix2_traj_0.mp4` shows
      severe penetration, so fix1/fix2 repro roots are not trusted baselines.
      The active baseline is reset to the originally effective 2026-06-06
      full1000 dataset/protocol
      `sft_dataset_full1000_maniskill_default_regen_20260606_0055`.
- [x] Read-only audit of the original 2026-06-06 full1000 H5 sources completed
      in Slurm allocation `125642`:
      `experiments/world_model_task_rebinding/cosmos3/original_full1000_readonly_audit_20260611`.
      Final inserted counts are `hole_constant 33/167`,
      `hole_move_stop 21/167`, `hole_reverse 125/167`, `none 104/166`,
      `peg_disturb 2/166`, and `peg_drop 43/167`. Moving-hole target
      displacement is only about `0.09-0.14m`. This confirms fix3 must
      regenerate/resample accepted demos rather than train directly on the
      original full1000 rows.
- [x] Move rejected fix2/fix1-era active outputs outside active experiments:
      fix2 roots under
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate/fix2_chain_user_rejected_after_original_reset_20260611`,
      and old SFT/readout roots under
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate/old_sft_and_readout_untrusted_after_original_reset_20260611`.
- [x] Add a training-entry safety guard so fix3 SFT cannot start from the
      normal wrapper unless the approval file contains
      `approved_for_sft=true` or `user_approved=true`. The same wrapper now
      refuses the invalid 6/6 full1000 source by default unless it is explicitly
      marked as a non-method diagnostic.
- [x] Render a replacement v7 complete-nine smoke package after the user
      rejected v4/current6 for excessive target speed and target self-insert
      semantics. V7 keeps the nine required classes and adds stricter
      moving-hole rejection gates: counterfactual future-path alignment at
      trigger time and a post-motion self-insert check requiring meaningful
      TCP rebinding before insertion. The rendered root is
      `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21`.
      H5 and render audits both have `failures=[]`, but this is still smoke
      evidence only and requires user direct-video approval before 60/full data
      generation or SFT.
- [x] Agent all-frame v7 smoke inspection completed on 2026-06-11. Every
      framebook page for all nine `301`-frame videos was opened and inspected,
      not only a sparse sheet. The smoke set passed the current continuation
      gate: no visible early target motion, target self-insertion, wall
      insertion, penetration, or final failure was observed. Borderline
      close-peg moving classes (`hole_late_sine` and
      `hole_late_continuous_insert`) were explicitly inspected and did not
      show stable insertion before target motion ended.
- [ ] Full1000 v7 complete-nine source generation is now split across
      tmux-held Slurm allocations so generation does not stall sequentially on
      one low-pass class. Current target quotas are nonuniform but no class is
      tiny: `hole_late_move_stop=70`, `hole_late_constant=90`,
      `hole_late_reverse=100`, `hole_late_sine=90`,
      `hole_late_continuous_insert=120`, `hole_late_fast_shift=120`,
      `none=160`, `peg_drop=150`, and `peg_disturb=100`.
      Active roots:
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_moving590`,
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_static_peg410`,
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_peg_aux250`,
      and
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_moving_hard_aux260`.
      Later low-count balancing roots:
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_none_aux160`,
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_move_sine_aux160`,
      and
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_pegdist_aux100`.
      Latest 2026-06-11 user correction: at `276/1000` merged unique H5 rows,
      do not keep peg/peg-disturb as the active bottleneck. First finish the
      non-peg classes, then return to peg-drop/peg-disturb. The peg-focused
      local `srun` steps were stopped without cancelling their allocations, and
      the preserved allocations were repurposed to four non-peg moving-hole
      buffer roots:
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_cfs_aux330_seedshift4`,
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_mrs_aux260_seedshift3`,
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_cfs_aux330_seedshift5`,
      and
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_mrs_aux260_seedshift4`.
      Later status: an all-active-root filename scan reached `305/1000`
      unique rows after including already accepted `peg_drop` rows from the
      stopped pegmix root. Since the six non-peg moving-hole scenarios still
      had `512` missing rows and the existing held allocations were all busy,
      two additional tmux-held 1-H200 allocations were started for non-peg
      generation:
      `126174` on
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_cfs_aux330_seedshift6`
      and `126175` on
      `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_mrs_aux260_seedshift5`.
      These extra roots are buffers only; final merge still deduplicates and
      caps each scenario at its quota. A later active-root scan reached
      `340/1000` unique rows while the user's latest merged count was
      `276/1000`; the difference is unmerged work-buffer H5 files. The six
      non-peg moving-hole classes still have `477` missing rows, so current
      resources stay on non-peg generation and peg-drop/peg-disturb remain a
      later step. A focused 4-H200 allocation `126188` later started on
      `server42` to concentrate attempts on scarce non-peg classes:
      `hole_late_continuous_insert`, `hole_late_constant`,
      `hole_late_move_stop`, and `hole_late_sine/hole_late_reverse`.
      The focus bundle uses the same approved generator/wrapper and only
      passes `SRUN_EXTRA_ARGS=--overlap` for step scheduling inside the held
      allocation; physical gates and source protocol are unchanged. The latest
      active-root scan including focus roots reached `366/1000` unique rows
      with `451` non-peg moving-hole rows still missing. A later scan reached
      `368/1000` unique rows with `449` non-peg rows still missing. The stale
      local non-overlap `srun` waiters from the first focus4 launch were
      terminated by PID only after the replacement focus2 roots were running;
      the 4-H200 allocation and useful generation steps were preserved. A
      later GPU-process check showed the overlap steps had all landed on
      logical GPU `0`, so
      `scripts/slurm/run_fix3_v7_nonpeg_focus3_pinned_in_allocation.sh` was
      added and started inside allocation `126188`. It explicitly pins new
      focused constant/sine/continuous roots to logical GPUs `1/2/3`; the
      generator, physical gates, and source protocol are unchanged. The latest
      active-root scan before pinned roots accepted rows reached `378/1000`
      unique rows with `439` non-peg moving-hole rows still missing. A later
      live scan reached `403/1000` active-root unique H5 rows while the
      user-facing merged count remained `276/1000`, because accepted buffer
      roots have not all been merged into the final path list. Current
      active-root class counts are `move_stop=18/70`,
      `constant=12/90`, `reverse=45/100`, `sine=20/90`,
      `continuous_insert=24/120`, `fast_shift=57/120`,
      `none=161/160`, `peg_drop=66/150`, and `peg_disturb=0/100`.
      The six non-peg moving-hole classes still have `414` rows missing.
      Continue using held allocations for non-peg generation first; handle
      `peg_drop` and `peg_disturb` last per user direction. Since all useful
      held allocations were busy and the non-peg gap remained large, an
      additional disjoint-seed focused launcher was added:
      `scripts/slurm/run_fix3_v7_nonpeg_focus4_pinned2_in_allocation.sh`.
      It pins four extra v7 generator roots inside a new 4-H200 allocation for
      `hole_late_constant`, `hole_late_continuous_insert`,
      `hole_late_sine`, and `hole_late_move_stop`; the source generator and
      physics gates are unchanged. The script passed `bash -n` inside
      allocation `126188`, and tmux session
      `cosmos3_fix3_focus4b_pinned_4h200_0611` started Slurm job `126210` on
      `server56`. Follow-up checks confirmed one generator process on each of
      the four GPUs, all four focus4b logs reached the generator, and
      `hole_late_sine` accepted the first focus4b row. The next active-root
      scan reached `421/1000` unique buffer H5 rows with `396` non-peg
      moving-hole rows still missing. A later check found no done marker, no
      active traceback/CUDA/Vulkan/OOM/filesystem/segmentation error, and
      eleven held allocations still running with active steps. The latest
      active-root scan reached `425/1000` unique buffer H5 rows:
      `move_stop=20/70`, `constant=12/90`, `reverse=47/100`,
      `sine=24/90`, `continuous_insert=29/120`, `fast_shift=66/120`,
      `none=161/160`, `peg_drop=66/150`, and `peg_disturb=0/100`.
      The six non-peg moving-hole classes still have `392` rows missing, so
      the current action is to keep source generation focused on non-peg and
      not start peg-only work yet. Final-merge tooling was prepared while
      generation continued: `scripts/world_model/merge_fix3_full1000_unique_h5.py`
      remains the file-level quota/dedup merge, and the new
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
      Latest user-provided merged count remains `276/1000`, but a fresh
      active-root buffer scan reached `493/1000` unique H5 rows:
      `move_stop=28/70`, `constant=24/90`, `reverse=58/100`,
      `sine=34/90`, `continuous_insert=49/120`, `fast_shift=73/120`,
      `none=161/160`, `peg_drop=66/150`, and `peg_disturb=0/100`.
      The six non-peg moving-hole classes still have `324` rows missing.
      Allocation `126065` was lost by Slurm revocation rather than agent
      cancellation; replacement tmux-held allocation `126219` is running on
      `server40` for focused `hole_late_constant` and
      `hole_late_continuous_insert` attempts with disjoint seed bases. Its
      early log shows expected physical gate rejects and an active generator
      process, not a runtime failure. Continue finishing non-peg first, then
      handle `peg_drop`/`peg_disturb` last. Since the non-peg gap remains
      large and the current generators are productive, an additional
      disjoint-seed 4-H200 launcher was added:
      `scripts/slurm/run_fix3_v7_nonpeg_focus4_pinned3_in_allocation.sh`.
      It passed `bash -n` and started tmux-held Slurm allocation `126223` on
      `server38`, running four unchanged-v7-generator roots for
      `hole_late_constant`, `hole_late_continuous_insert`, `hole_late_sine`,
      and `hole_late_move_stop` with seed bases `21250000`, `21241000`,
      `21251000`, and `21280000`. Compute-node process inspection confirmed
      four active `.venv/bin/python` generators across four GPUs; logs reached
      the reject/accept loop and accepted an early `hole_late_sine` row.
      A later active-root scan reached `606/1000` unique H5 rows:
      `move_stop=37/70`, `constant=41/90`, `reverse=75/100`,
      `sine=52/90`, `continuous_insert=81/120`, `fast_shift=93/120`,
      `none=161/160`, `peg_drop=66/150`, and `peg_disturb=0/100`.
      The six non-peg moving-hole classes still have `211` rows missing, so
      the current action remains non-peg generation. Tmux sessions whose names
      still contain `pegdist` or `pegmix` were checked and are running
      repurposed non-peg roots, not peg-only work. Error scan remains clean and
      no done marker exists. Allocation `126074` was revoked by Slurm and is
      no longer in the queue, but eleven allocations remain running and the
      source count increased from `510` to `523`, `529`, `538`, `550`,
      `564`, `570`, `582`, `598`, and then `606` across successive monitoring passes. Allocation `126089` was later
      revoked by Slurm after accepting additional non-peg reverse rows; ten
      allocations remain running and no extra replacement allocation was
      requested at this moment because production is still active.
      Duplicate supplemental roots are allowed only as generation work buffers;
      the final combined full1000 path list must select at most the target
      quota per scenario and record all source roots. When the full1000 source
      count is complete, stop and wait for user approval before WAM export,
      rendering-scale expansion, or Cosmos3 SFT.
- [ ] Current full1000 priority is non-peg completion, not peg debugging.
      As of 2026-06-11 late evening the merged unique count is `276/1000`;
      `none` is already over quota, moving-hole families are slow but nonzero,
      and `peg_disturb` remains `0/100`. The concrete peg finding is preserved
      for later: accepted/rejected outcomes are not determined by the env seed
      alone because the DP action sampler uses the global torch random stream.
      The old v5b `peg_disturb_seed1051032` succeeded when it appeared at
      global attempt 32, but the same seed failed when placed at attempt 1/4 in
      grouped full1000 runs with a different diffusion action RNG state. Per
      user instruction, defer peg-drop/peg-disturb until the other classes are
      filled instead of spending the current allocation window on peg-only
      attempts. When returning to peg, do not blind-churn if repeated attempts
      fail; stop and report the concrete blocker for user direction.
- [x] Fixed a source-generation seed bug during full1000 startup: priority
      smoke seeds such as `1040017` were reused later by the ordinary
      `scenario_seed_base + scenario_attempt` stream, which could create
      duplicate scenario/seed H5 rows. The generator now keeps the priority
      attempt first and then enumerates the original base stream while
      skipping only priority seed values. The duplicate-seed partial roots were
      moved to
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_duplicate_seed_partial`.
      A too-large interim `base+1000` offset partial was also rejected and
      moved to
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_offset_seed_partial`.
- [ ] Export full-episode WAM conditions with
      `sidecar_target_mode=future_aligned_state`.
- [ ] Run full condition preflight/action-target audit before any SFT.
- [ ] Only after the above, start Cosmos3 SFT from the same 300-step full
      episode contract.

## Non-Negotiable Fix3 Gates

- [ ] `inserted_end=true` for every training and validation source demo, with
      a stricter visual/geometric check that rejects peg insertion into the
      block wall. This must be achieved by original-protocol acceptance and
      resampling, not by forcing peg/target states.
- [ ] target stays static while the robot grasps the peg and approaches the
      current hole; target motion starts only when peg is held and near
      pre-insertion alignment.
- [ ] final target pose remains inside the static DP/expert reachable training
      domain so final insertion is physically feasible.
- [ ] target motion is difficult for the original static DP because it happens
      late, after commitment toward the old/current hole, but it must not be
      the block simply moving into the peg.
- [ ] target motion has displacement, path-shape, and timing diversity, but
      the move duration must be short enough to be visually fast and to force
      prediction/rebinding.
- [ ] include continuous/fast late-moving target demos where the robot must
      redirect to the future target pose before insertion.
- [x] no 128/129-frame slicing, no 93-frame eval, no cropped comparison.
- [x] no DP-generated OOD failures as positive training demos.
- [ ] user approval after per-scenario demo video review is mandatory before
      any Cosmos3 SFT job starts.
- [x] v7 smoke all-frame inspection has passed the latest continuation gate for
      full1000 generation.
- [ ] Do not start WAM export or Cosmos3 SFT until the full1000 source passes
      strict source audits, rendered video/frame checks, and full-episode
      condition/action-target preflight.

## Rejected Review60 Overlay Package

- [x] Generated `60` widened-final-pose expert source attempts under
      `experiments/world_model_task_rebinding/cosmos3/fix3_final_pose_source_review60_20260610`.
      The source generator accepted `60/127` attempts with true final
      insertion.
- [x] Replayed the source set to `pd_ee_delta_pose`. Replay saved `57/60`
      successful demos; the three replay failures are excluded from the
      dynamic overlay review package.
- [x] Built the review package under
      `experiments/world_model_task_rebinding/cosmos3/fix3_dynamic_overlay_review60_20260610_6x10`.
      The strict audit has `failures=[]`, six scenarios with exactly `10`
      paths each, `301` state frames, `300x7` actions, `inserted_end=true`,
      target-motion norm range `0.1886-0.3782m`, mean motion `0.2815m`, and
      first insertion between steps `165` and `294`.
- [x] Rendered `60` default-view videos under
      `experiments/world_model_task_rebinding/cosmos3/fix3_dynamic_overlay_review60_render_server35_20260610_6x10`.
      OpenCV verification found all videos openable with `301` frames,
      `30 fps`, `1024x1024`, existing review sheets, and final
      `inserted_end=true`.
- [x] Agent visual inspection opened one representative review sheet from each
      of the six large-motion families:
      `hole_move_stop_large`, `hole_constant_large`, `hole_reverse_large`,
      `hole_sine_large`, `hole_continuous_insert_large`, and
      `hole_late_shift_large`. Each representative video showed readable
      default-view target motion and final insertion.
- [x] User rejected this package on 2026-06-10. Failure reasons: target motion
      starts at the beginning instead of after peg grasp/near-hole alignment;
      visual insertion appears to go into the block wall rather than the hole;
      the construction changes target motion amplitude without preserving the
      desired fix2 late-rebinding task semantics; and target movement is too
      slow. This package is not approval evidence and must not be used for SFT.

## Corrected Rebuild Requirements

- [x] Add a late-trigger physical expert generator that records target motion
      inside the real ManiSkill episode, rather than kinematically overlaying
      target slots after the robot/peg trajectory is complete.
- [x] Generate one smoke sample per corrected late-trigger scenario inside a
      Slurm allocation and render default-view review sheets.
- [x] Generate `6x10` corrected approval videos only after the smoke set passes
      strict source checks and visual inspection.
- [x] Mark the goal blocked again after corrected demos are rendered and
      inspected; do not start SFT.

## Resource State

- [x] Old 4-H200 allocation `123385` expired after fix1/fix2 diagnostics.
- [x] Existing aux 1-H200 allocation `123499` remains available for smoke,
      render, and sanity checks, but `server32` render attempts failed with
      SAPIEN/Vulkan device loss or first-render hangs.
- [x] Requested new tmux-held 4-H200 allocation `125318`
      (`cosmos3_fix3_4h200_hold_20260610`) for fix3 data/SFT work.
- [x] Allocation `125318` is running on `server13`, but current evidence says
      it is not usable for ManiSkill/Cosmos render/debug work: torch CUDA
      initialization is unhealthy and SAPIEN render-device creation fails even
      when CUDA is hidden.
- [x] Requested tmux-held 1-H200 render allocation `125351`
      (`cosmos3_fix3_render_1h200_hold_20260610`) excluding only the two
      currently failed nodes. It later started on `server38`; no SFT has been
      started from it.
- [x] Short render allocation `125385` on `server35` was sufficient to finish
      the fix3 smoke render and review60 approval render package.
- [x] Allocation `125351` on `server38` generated the corrected late-trigger
      H5 data successfully but failed rendering: minimal `256x256` canary and
      `render_rgb_array("render_camera")` timing both hung at first render
      despite SAPIEN seeing the H200. This is node/render scheduling evidence,
      not a data failure.
- [x] New tmux-held render allocation `125516` on `server35` passed
      `render_min_canary.py` and rendered the corrected late-trigger smoke and
      review60 videos.

## Rejected Late-Trigger Review60 Package

- [x] Added `scripts/world_model/generate_cosmos3_fix3_late_trigger_dynamic_experts.py`.
      The generator records real ManiSkill episodes: the robot grasps the peg,
      moves near the current hole, then the target block moves quickly in the
      live episode, and the expert replans to the final reachable target pose.
      It rejects sources unless real episode success and stricter peg-head
      geometry both pass.
- [x] Fixed two implementation issues during smoke: `RecordEpisode` step count
      is cumulative, so trigger/raw steps now use episode-relative counts; and
      `transforms3d.quat2axangle` return order is `axis, angle`, not
      `angle, axis`.
- [x] Smoke set:
      `experiments/world_model_task_rebinding/cosmos3/fix3_late_trigger_smoke2_20260610_6`.
      It generated one sample per corrected scenario, passed
      `audit_failures=[]`, and rendered on `server35` under
      `render_512_server35`. The agent opened all six smoke review sheets.
- [x] Review60 source root:
      `experiments/world_model_task_rebinding/cosmos3/fix3_late_trigger_review60_20260610_6x10`.
      The generator accepted `60/99` physical attempts. Audit:
      `60` paths, six scenarios with `10` each, `failures=[]`,
      target-motion norm min/mean/max `0.0936/0.1297/0.1689m`,
      trigger min/max `62/159`, move duration min/max `7/52`, and
      first insertion min/max `127/240`.
- [x] Review60 render root:
      `experiments/world_model_task_rebinding/cosmos3/fix3_late_trigger_review60_20260610_6x10/render_512_server35`.
      Render manifest reports `60` videos, `60` sheets, six scenarios with
      `10` each, `301` frames, `30 fps`, `10.033s`, approved default camera,
      and `512x512`.
- [x] User rejected this package on 2026-06-11. The previous agent visual
      inspection was unreliable: user-identified samples
      `0001_hole_late_constant...mp4` and
      `0002_hole_late_reverse...mp4` still visibly insert into the block
      side/wall, and the initial alignment/trigger protocol is not the
      successful fix2-style "about to insert" protocol. Additional diagnosis:
      the script allowed final target poses outside the original static
      PegInsertionSide box support (`x=[-0.05,0.05]`, `y=[0.20,0.40]`);
      `0002` ended at `y=0.4257`, and `0001` was outside the x lower bound.
      This package must not be used for SFT or method evidence.

## Rejected Official-Insert VLM Package After 2026-06-11 Rejection

- [x] Patch the generator defaults so final target poses stay inside the
      original static box-pose support: `x=[-0.05,0.05]`,
      `y=[0.20,0.40]`.
- [x] Patch the generator so late target motion triggers only after the
      official pre-insertion refinement and a strict preinsert gate requiring
      both peg-head and peg-center YZ alignment in the goal frame.
- [x] Patch final insertion back to the official/fix2-style protocol: refined
      pre-insert pose followed by the final `0.05m` insertion target, with
      real ManiSkill episode success used as the live success authority.
- [x] Run a Slurm-side 6-video smoke generation/render after the patch.
      Direct-video VLM review with local
      `Qwen2.5-VL-7B-Instruct` passed `6/6` under
      `experiments/world_model_task_rebinding/cosmos3/fix3_late_trigger_smoke6_20260611_official_insert_6/render_512_server35/vlm_video_review_qwen25_7b_v3`.
- [x] Replace sheet-based approval with direct-video VLM approval per user
      instruction. The VLM checks late target motion, pre-motion alignment,
      visible target movement, visible red peg-head entry into the hole,
      no side/wall insertion, final insertion, and peg grasp retention.
- [x] Regenerate exactly `10` replacement videos per scenario and stop for
      user review before any SFT. Source audit:
      `num_paths=60`, scenario counts `10` each, `failures=[]`,
      target-motion norm min/mean/max `0.0901/0.1137/0.1665m`, trigger
      min/max `72/164`, move duration min/max `6/51`, and first insertion
      min/max `138/248`.
- [x] Render replacement `6x10` videos at `512x512`, `30 fps`, `301` frames
      using the approved ManiSkill default camera in allocation `125516` on
      `server35`.
- [x] Direct-video VLM gate passed all replacement videos:
      `num_videos=60`, `num_pass=60`, `all_pass=true`, model
      `/public/home/yanhongru/ICLR2027/predictive-interruption-v2/checkpoints/Qwen2.5-VL-7B-Instruct`.
      Result path:
      `experiments/world_model_task_rebinding/cosmos3/fix3_late_trigger_smoke60_20260611_official_insert_vlm_gate/render_512_server35/vlm_video_review_qwen25_7b/summary.json`.
- [x] User rejected the above gate by direct visual inspection. This package is
      invalid because the Qwen gate missed initial penetration, misalignment,
      and side/wall insertion. It must not be used for SFT or method evidence.

## Rejected Physical-Gate Rebuild After User Rejection

- [x] Archive all rejected/nonconforming fix3 active roots outside Reflex under
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate`.
- [x] Add strict line/clearance/wall gates: sample points along the whole peg
      centerline in the hole frame, require true clearance with no slack for
      standardized evidence, and fail any frame with wall-collision risk.
- [x] Added constrained insert projection for the final insertion segment only.
      User direct video review rejected this as non-physical because it caused
      visible penetration and a peg self-drilling/crawling insertion artifact.
      This path is historical diagnostic code only and is disabled by default
      in the generator.
- [x] Generated and rendered the rejected 6-smoke package:
      `experiments/world_model_task_rebinding/cosmos3/fix3_physics_gate_constrained_insert_smoke6_20260611_v8`.
      It contains exactly one demo per scenario, `301` frames, `300` actions,
      `30 fps` rendered videos, raw action steps `235-275`, target motion norm
      `0.0909-0.1224m`, trigger steps `90-129`, first insertion steps

## Reset To Original Effective Dataset After Fix2 Repro Rejection

- [x] Archive the user-rejected fix2 repro root outside active experiments:
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate/fix2_repro_user_rejected_penetration_20260611`.
- [x] Audit the original 2026-06-06 full1000 dataset and source H5s as the
      only trusted physical-protocol baseline. The audit must report per-scenario
      final success counts, target-motion displacement, target-motion onset
      relative to peg grasp/alignment, and representative rendered evidence.
- [x] Treat the original full1000 generation environment, source H5s, dataset
      root, and generator code as read-only. Before changing target-motion
      ranges or acceptance rules, copy the relevant generator/config into a new
      fix3-specific script/output root and record provenance. Never edit the
      original effective generation chain in place.
- [x] Locate the original 6/3 DP rollout generator if it still exists in the
      workspace/archive; if not, reconstruct a copied fix3 generator from the
      H5 `source_manifest_json` provenance. The rejected overlay postprocessor
      and rejected late-trigger motion-planning script are not valid baselines.
- [x] Implement the large-motion rebuild by modifying only the copied
      accepted physical generation path: target stays static until peg is
      aligned, target motion is larger and may continue while the robot
      predicts/rebinds, and accepted demos require true final insertion.
- [x] If a candidate fails the original physical gate or final insertion, reject
      and resample. No projection, no self-drilling, no wall insertion, and no
      accepting failed samples as positive training data.
- [x] The previous agent key-frame-only inspection was insufficient and is
      invalidated. User specifically identified
      `0000_hole_late_move_stop_seed999000_idx0000.fix3_traj_0.mp4` as
      visibly penetrating/misaligned, and noted that the peg appears to move
      itself into the hole instead of being physically inserted by the robot.
      The package was moved outside active Reflex under
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate/user_rejected_constrained_insert_v8`.

## Current Fix2-Protocol Reproduction Requirement

- [x] Reproduce the fix2/official ManiSkill insertion protocol first as a
      small smoke set, with no constrained projection and no target-motion
      enlargement. It accepted `6/8` official-solver attempts under Slurm job
      `125642` on `server56`.
- [x] Render the smoke videos in a Slurm allocation using the approved default
      human camera, inspect dense frame sequences around the full trajectory,
      and record concrete evidence. Kept baseline:
      `experiments/world_model_task_rebinding/cosmos3/fix2_official_insert_repro_smoke6_20260611_server56_padded301`.
      It has `6` videos, each `301` frames at `30 fps`; the agent opened all
      six dense review sheets. This is static baseline smoke only.
- [x] Archive noncontract intermediate outputs outside Reflex:
      the strict-v8-gate failure and the rawsplit short-video render are under
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate/fix2_repro_failed_or_noncontract_20260611`.
- [x] Record the core diagnosis: if fix2 works and modified fix3 fails with
      penetration/self-drilling, the failure is caused by added modifications
      or invalid evidence gates, not by wording around the protocol.
- [x] After resetting to the original effective 2026-06-06 protocol, increase
      late target displacement only in a copied fix3 generator and new output
      root. No 60-video package and no SFT may start before user approval.

## Original-Protocol Large-Motion Smoke6 V1 Rejected For Timing/Amplitude

- [x] Added copied generator
      `scripts/world_model/generate_cosmos3_fix3_original_protocol_large_motion_dp.py`.
      It reconstructs the original DP-driven late-target protocol from the
      H5 provenance and writes only to new fix3 output roots. It refuses to
      write into the original 6/6 full1000 dataset root.
- [x] Render metadata bug fix only:
      `scripts/world_model/render_cosmos3_maniskill_sft_dataset.py` now marks
      late-trigger samples as `triggered=true` if any step is triggered, not
      only if frame 0 is triggered. This preserves the rendered trajectories
      and fixes task-state text/provenance.
- [x] Generated one codepath canary and then a six-scenario smoke set inside
      Slurm allocation `125642` on `server56`; no rollout/render ran on the
      login node.
- [x] Smoke root:
      `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_smoke6_20260611_v1`.
      The manifest has `6` accepted records, each with `success_at_end=true`,
      `301` state/source frames, `300` action steps, and target motion
      `0.1807-0.1918m`.
- [x] Render root:
      `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_smoke6_20260611_v1/render_512_server56`.
      It contains `6` videos rendered with the approved ManiSkill default human
      camera at `512x512`, `30 fps`, `301` frames.
- [x] User direct video review on 2026-06-11 accepted the insertion quality
      but rejected the timing/amplitude: `0002_hole_late_reverse...mp4` and
      `0005_hole_late_fast_shift...mp4` move the target before peg pickup,
      while `0001_hole_late_constant...mp4`,
      `0003_hole_late_sine...mp4`, and
      `0004_hole_late_continuous_insert...mp4` have too-small and too-slow
      target motion. V1 is not approved for 60-video generation or SFT.

## Original-Protocol Large-Motion Complete9 Smoke Requirements

- [x] Patch the copied fix3 generator so fallback time alone cannot trigger
      target motion. Target motion now requires stable peg hold/lift, a delay
      after first robust hold, and preinsert geometry; fallback still requires
      robust hold plus relaxed preinsert geometry.
- [x] Patch the copied fix3 generator so moving-target scenarios complete
      larger displacement in short windows instead of spreading the motion
      over the remaining episode. Defaults now target `0.22-0.30m` motion,
      with faster `move_stop`, `fast_shift`, `constant`, `reverse`, `sine`,
      and `continuous_insert` schedules.
- [x] User correction: smoke taxonomy must add to the previous six moving-hole
      variants, not replace them with the six original top-level classes.
      The wrong `v2_smoke6` / `v3_complete_smoke6` partial roots were moved to
      `/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate/wrong_taxonomy_smoke6_replaced_by_complete9_20260611`.
- [x] Patch the copied fix3 generator taxonomy to `9` smoke classes:
      `hole_late_move_stop`, `hole_late_constant`, `hole_late_reverse`,
      `hole_late_sine`, `hole_late_continuous_insert`,
      `hole_late_fast_shift`, `peg_disturb`, `peg_drop`, and `none`.
      The first six preserve the previous moving-hole variants; the last
      three are added from the original full1000 taxonomy.
- [x] User rejected the partial v4 `current6` render after direct video
      review. `hole_late_move_stop` and `hole_late_fast_shift` are too fast
      and collide into the target; `hole_late_constant` and `hole_late_sine`
      show the target moving toward the peg / self-inserting, so the original
      policy could effectively stay on the old peg path and still succeed.
      This v4 package is rejected and must not be used for 60-video generation
      or SFT.
- [x] Patch the copied fix3 generator for v5: double the
      `move_stop`/`fast_shift` motion windows to reduce speed by 50%, and add
      a counterfactual anti-self-insert path gate. At the trigger, the script
      temporarily places the target at sampled future path poses and rejects
      the candidate if the current peg head would already align with the
      moved hole. This directly targets the failure where the block moves into
      the peg instead of forcing policy/world-model rebinding.
- [x] Patch v5 sampling hygiene: the full smoke order remains nine classes,
      but static/peg classes use priority seeds from the read-only original
      audit where those low-success classes already ended inserted. This
      reduces wasted Slurm time without weakening final live-success or
      slot-insertion gates. The generator also writes partial paths/manifest
      after every accepted demo so partial renders can be audited if a later
      low-success class stalls.
- [x] Patch v5b continuation hygiene after `continuous_insert` stalled:
      preserve accepted smoke rows, add an explicit `scenario_sequence` and
      `accepted_index_offset` CLI for generating the remaining scenarios, and
      set the anti-self-insert YZ hard floor to `0.055m` while still keeping
      the `2.4x` hole-radius multiplier. This avoids rejecting borderline
      non-self-inserting paths while still rejecting target paths that run
      into the current peg.
- [x] Run py_compile and a state audit for the Complete9 copied generator inside a
      Slurm allocation, not on the login node.
- [x] Generate/render the v5b nine-scenario Complete9 smoke package and stop
      for user direct video approval. V5b is now superseded by v7 and must not
      be used for 60-video generation or SFT.
- [x] Agent all-frame framebook review completed for the v5b Complete9 smoke
      package: all `144` framebook pages were opened, covering `9` videos with
      `301` frames each. The review note is
      `docs/world_model_task_rebinding/2026-06-11_v5b_complete9_smoke_frame_review.md`.
      This remains historical smoke/tooling evidence only after v7 replacement.
- [x] Patch v7 self-insert rejection after further audit showed v5b still
      allowed weak move-stop/fast-shift candidates. The new gate rejects cases
      where the target moves substantially, TCP moves too little, and
      insertion has already happened during/near the target-motion phase.
- [x] Render the v7 Complete9 smoke package on Slurm allocation `125951`
      (`server21`) with approved default ManiSkill camera, `512x512`,
      `30 fps`, and `301` frames per video. H5 audit and video-frame audit
      both report `failures=[]`.
- [x] User direct-video approval for the v7 Complete9 smoke package is now the
      active scale-up basis. The user explicitly stated on 2026-06-11 that
      `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611`
      is acceptable; the remaining task is full1000 source generation with
      strict per-class uniqueness and final user approval before any SFT.

## Full1000 Source Generation Gate

- [x] Latest 2026-06-12 user override supersedes the remaining full1000 gate
      for the immediate Cosmos3 run: stop data construction now and proceed to
      SFT from the frozen v7 DP source
      `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733`.
      The frozen source has `733` unique H5 rows across all nine classes:
      `hole_late_move_stop=44`, `hole_late_constant=48`,
      `hole_late_reverse=99`, `hole_late_sine=60`,
      `hole_late_continuous_insert=96`, `hole_late_fast_shift=105`,
      `none=160`, `peg_drop=119`, and `peg_disturb=2`.
      Do not continue generation or backfill missing quota rows unless the
      user explicitly reopens data generation.
- [x] Stop active v7 generator processes by SIGINT inside held allocations, not
      by `scancel`. The frozen 733-row H5 strict audit passed under the
      user-override class counts:
      `strict_ok=true`, `num_failed_records=0`, output
      `fix3_v7_dp_user_override_sft_source_20260612_733/strict_source_h5_audit_user_override_quota733`.
- [x] Build the RGB SFT dataset from the frozen H5 paths and proceed through
      full-episode WAM condition export, preflight, action-target audit, and
      Cosmos3 SFT startup without stopping at the old full1000 approval gate.
      Render root:
      `experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612`.
      It contains `733` videos split into `661` train and `72` val rows,
      rendered at `512x512`, `30 fps`, and `301` frames with the approved
      default ManiSkill camera. Dataset inspection passed with `valid=true`;
      train/val artifact checks passed for all `661/72` videos as readable and
      nonblank.
- [x] Agent visual review covered representative moving-hole, static, peg-drop,
      and the two peg-disturb review sheets. The user subsequently reported
      that the videos looked acceptable. This supports using the data for SFT
      chain progress, but the source remains DP-success-filtered bootstrap
      data rather than hard-dynamic proof.
- [x] WAM condition export/preflight/action-target audit passed under
      `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_rgb_300step_20260612_0245`
      and
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_4gpu_20260612_0245`.
      The exported contract is full episode only: `301` RGB/state frames,
      `300` action rows, `300x32` action targets, future-aligned state sidecar,
      no 128/129-frame slicing, and `strict_action_target_ok=true`.
- [x] Cosmos3 SFT is running in tmux session
      `cosmos3_sft_fix3_v7_733_4gpu_126210` on Slurm allocation `126210`
      (`server56`, `4xH200`). Startup evidence: checkpoint load completed from
      `checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID-DCP`, train/val
      dataloaders prewarmed, and the trainer entered `Starting training...`.
- [x] Active cosmos3 experiment root was cleaned after render/SFT startup.
      Current active root keeps only the frozen `733` source, rendered SFT
      dataset, WAM condition root, SFT output root, v7 Complete9 review root,
      original 6/6 baseline/audit, and approval/provenance files. Old process
      roots/logs were moved, not deleted, to
      `/public/home/yanhongru/ICLR2027/archived/reflex_cosmos3_process_artifacts_after_fix3_v7_733_sft_20260612`.

- [x] Current user direction: nonuniform class quotas are allowed only if no
      class becomes tiny; generate class groups in parallel rather than
      sequentially blocking on one low-pass class; do not treat
      attempt/seed enumeration as a research blocker.
- [x] The 1500/hard-teacher supplement direction is recorded but currently
      deferred by the later 2026-06-12 user instruction. Do not use it as the
      active gate. The active source target is again v7 DP-generated
      `full1000`, using the original quotas
      `70/90/100/90/120/120/160/150/100`.
- [x] Add hard-teacher supplement script:
      `scripts/world_model/generate_cosmos3_fix3_hard_dynamic_teacher.py`.
      It keeps existing accepted rows, generates only supplemental rows, uses
      motion-planning/scripted teacher phases rather than DP-success filtering,
      writes `source_kind=hard_dynamic_teacher`, and records robust hold plus
      peg perturb/drop events in the existing H5 schema.
- [x] Patch the full1000 Slurm wrapper so auxiliary class-generation roots can
      disable priority seeds and use disjoint seed bases:
      `scripts/slurm/run_fix3_v7_approved_full1000_generation_in_allocation.sh`.
      This preserves the v7 physical protocol while preventing duplicate
      trajectories from being counted twice at merge time.
- [x] The exact `1000`-trajectory v7 DP quota plan is currently superseded by
      the latest 2026-06-12 user override. The active SFT source is the frozen
      `733`-row set, not a completed full1000 set. Do not resume filling the
      missing quota rows unless the user explicitly reopens data generation.
- [x] The frozen `733` source was deduplicated into a canonical H5 tree and
      passed strict source H5 audit under the user-override class counts with
      `strict_ok=true` and `num_failed_records=0`. This replaces the old
      full1000 merge/audit gate for the current SFT run only.
- [x] The old full1000 approval stop is superseded for the current run by user
      instruction and visual acceptance of the rendered videos. Controller/DP
      integration is still not approved by this data step; it remains gated on
      post-SFT generated video/action/readout evidence.
