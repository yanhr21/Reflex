# Receding Closed-Loop TODO

## Current Gate

- [x] 2026-06-14 objective-level closed-loop gate added and run:
      `scripts/world_model/check_cosmos3_closed_loop_objective_gate.py`.
      Output:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_objective_gate.json`
      and `.md`. It verifies the implementation contract requested by the
      user: full `300` actions / `301` frames, causal target-motion detector
      instead of manual trigger disclosure, explicit annotated controller
      timeline, no-motion samples using DP only because the same detector
      never fires, moving-target samples using `WM_ACTIVE`, same-source pure-DP
      comparison, and hard-case action/rebind evidence. Current result:
      `implementation_contract_ok=true`, `method_effectiveness_ok=false`.
      The gate is now stricter than the original first pass: moving-target
      samples must record `prefix_selection.mode=target_motion_onset`, a
      concrete detector frame/streak frame, and at least one full short chunk
      worth of WM use (`min_moving_wm_active_frames=8` by default). The
      static/no-motion sample must record the causal detector-never-triggered
      mode and no detected frame. This prevents future evidence from passing
      with a manifest prefix, hidden target-onset disclosure, or token one-frame
      Cosmos activation.
      The gate also checks the structured annotation timeline used to write
      the overlay video: every sample must have a `controller_timeline` and
      `annotated_video_summary.timeline` of length `301`, controller counts
      must match the summary, moving samples must show WM-active frames in the
      annotated timeline, and static/no-motion must show zero annotated
      WM-active and zero target-motion-detected frames. Missing/invalid
      controller counts, count sums other than `301`, and mismatched
      annotated-summary counts are hard failures. This makes explicit Cosmos
      takeover annotation a machine-checkable artifact contract, not just an
      mp4 existence check.
      The gate now scans the actual mp4 files, not only summaries:
      `26/26` raw/annotated videos have `301` frames and duration
      `10.033333333333333` seconds. The old short-video failure mode is
      therefore guarded against in current artifacts. The scanner uses
      OpenCV as a fallback only when the primary Python decoder fails; if no
      decoder can read the file, the video contract still fails.
      The remaining failures are method performance, not the old incomplete
      video/eval-interface bug: val Cosmos `1/3` is worse than pure DP `3/3`,
      and hard pure-DP failures are only rescued at `1/6`.
- [x] 2026-06-14 hardened the live-receding Slurm wrappers against accidental
      non-method launches. `run_cosmos3_live_receding_panel_in_allocation.sh`
      and `run_cosmos3_live_receding_loop_in_allocation.sh` now refuse unless
      the method-safe modes are active:
      `PREFIX_ROLE_MODE=auto` / `PREFIX_ROLE=auto`,
      `PREFIX_START_MODE=target_motion_onset`,
      `PRETRIGGER_CONTROL_MODE=frozen_dp_until_target_motion`, and
      `RUN_COSMOS_INFERENCE=true`. Manifest prefixes, source-restored
      pretriggers, explicit role overrides, or no-Cosmos dry runs now require
      `ALLOW_LIVE_RECEDING_DIAGNOSTIC=true` and remain non-method diagnostics.
      Login-safe checks passed: `bash -n` for both wrappers, and fake-Slurm
      `PREFIX_START_MODE=manifest` calls exited `42` before any heavy work or
      output directory creation.
- [x] 2026-06-14 added a local objective-gate self-test:
      `scripts/world_model/selftest_cosmos3_closed_loop_objective_gate.py`.
      It builds synthetic moving/static summaries with temporary placeholder
      video paths and directly verifies the hardened gate accepts a valid
      causal full-episode moving sample, rejects token one-frame WM use,
      rejects manual/manifest-style prefixing, rejects missing detector-frame
      provenance, accepts detector-never-triggered no-motion evidence, and
      rejects a static sample that triggers WM. It also covers the structured
      annotation-timeline contract and a subprocess end-to-end check that
      `panel_full_episode_contract_ok=false` rejects val Cosmos, hard Cosmos,
      and pure-DP panels. The test runs on the login node without simulation,
      rendering, training, Slurm, or video decoding.
      `py_compile`, this self-test, the next-action-gate self-test, and
      `bash -n` on the live-receding wrappers passed.
- [x] 2026-06-14 hardened the old one-shot closed-loop wrappers so they cannot
      be launched accidentally as corrected evidence. Both
      `scripts/slurm/run_cosmos3_closed_loop_panel_in_allocation.sh` and
      `scripts/slurm/run_cosmos3_receding_closed_loop_in_allocation.sh` now
      default to refusal unless
      `ALLOW_OLD_ONESHOT_CLOSED_LOOP_DIAGNOSTIC=true` is explicitly set. The
      refusal text points to the corrected full-300 live-receding wrappers.
      Login-safe fake-Slurm checks passed: the panel wrapper exited `43` even
      with `VISUAL_REVIEW_STATUS=pass`, the single wrapper exited `43`, and
      neither created an output directory before refusing.
- [x] 2026-06-14 added a reusable wrapper-guard self-test:
      `scripts/world_model/selftest_cosmos3_closed_loop_wrapper_guards.sh`.
      It runs fake-Slurm, login-safe probes for four launch-time guard paths:
      live-receding panel with manifest prefix, live-receding single loop with
      explicit role, old one-shot panel, and old one-shot single loop. It
      checks the expected refusal exit codes (`42` for non-method live
      diagnostic modes, `43` for old one-shot wrappers), expected refusal text,
      and that no requested output directory is created before refusal. The
      script passed together with the objective-gate and next-action-gate
      self-tests.
- [x] 2026-06-14 direct old-vs-current video length audit completed with
      `scripts/world_model/audit_video_length_contract.py`. Output:
      `docs/world_model_task_rebinding/2026-06-14_iter2100_vs_iter2700_video_length_audit.json`
      and `.md`. The user-flagged old
      `live_receding_panel10_corrected_iter2100_20260613_161006` run is
      confirmed invalid for the current objective: its final rollout videos
      are `131` frames / `4.3667s` and `119` frames / `3.9667s`. Current
      iter2700 evidence roots have `26/26` final raw/annotated videos at
      `301` frames / `10.0333s`. Use the current iter2700 artifacts for
      full-episode implementation evidence; keep the old iter2100 path only as
      historical negative evidence explaining the previous bug.
- [x] 2026-06-14 future closed-loop runs now record decoded video contract
      evidence inside their own summaries. `run_cosmos3_live_receding_loop.py`
      decodes both final raw and annotated videos immediately after writing
      them and writes `video_file_contract_ok`; `run_dp_full_episode_baseline.py`
      does the same for pure-DP comparisons. `run_cosmos3_live_receding_panel.py`
      and `run_dp_full_episode_baseline_panel.py` propagate those fields to
      panel summaries. This prevents a future short-video artifact from being
      silently packaged as a completed 300-action run.
      The shared helper now lives in
      `scripts/world_model/video_contract_utils.py`, uses OpenCV fallback
      after imageio failures, and treats the expected inspection count as part
      of the contract. With annotation enabled, raw and annotated videos must
      both exist and decode to `301` frames for `video_file_contract_ok=true`.
      The objective gate and direct video-length audit now use this same
      helper, preventing future disagreement between summary-time and
      evidence-time video validation.
      The panel aggregators now also compute
      `panel_full_episode_contract_ok` and `sample_contract_failures`. A panel
      run returns failure if any sample lacks a summary, misses `300/301`, has
      bad video files, or has controller counts that do not sum to `301`;
      pure-DP panels additionally require `PURE_DP=300` and `WM_ACTIVE=0`.
      The objective gate consumes this panel flag when present. The shared
      panel contract helper is
      `scripts/world_model/panel_contract_utils.py`, with local coverage in
      `scripts/world_model/selftest_panel_contract_utils.py`.
- [x] 2026-06-14 hard-screen-2 completed. A second 15-sample high-motion
      manifest was selected at
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen2_20260614/hard_case_eval_manifest.json`.
      Full pure-DP baseline succeeded on `9/15` and failed on indices
      `4,9,10,11,12,13`. Cosmos closed-loop was run only on those six pure-DP
      failures under
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen2_20260614/cosmos_iter2700_puredp_fail_4_9_10_11_12_13`.
      Cosmos succeeded on `1/6`, the `hole_late_fast_shift` index `4` sample:
      full `300/301`, detector trigger `f86`,
      `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:86,
      WM_ACTIVE:56, DP_HANDOFF:158}`, final success `true`, and final
      peg-head-at-hole
      `[-0.0071999, 0.0015902, 0.0021036]`. The same-source pure-DP baseline
      for index `4` was full length and failed with peg-head-at-hole
      `[-0.0958784, 0.0228078, -0.0324848]`. The Cosmos panel contact sheet,
      the index `4` Cosmos annotated sheet/final frame, and the index `4`
      pure-DP annotated sheet/final frame were opened directly; visuals match
      the metrics. The other five pure-DP-failure samples also failed under
      Cosmos. This provides a second positive comparison sample but also
      shows the current iter2700 closed-loop interface is not broadly reliable
      on hard cases.
- [x] 2026-06-14 hard-screen-2 action/rebind diagnosis completed with
      `scripts/world_model/analyze_cosmos3_hard_case_action_rebind.py`.
      Output:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_hard_screen2_action_rebind_analysis.json`.
      This is a read-only comparison of executed Cosmos chunks against
      same-H5 source-teacher action rows; it does not execute teacher actions
      or change success criteria. On the six pure-DP failures, matched pure DP
      success is `0/6` and Cosmos success is `1/6`. The failure samples show
      persistent `rel_y_abs`/`rel_z_abs` C_pi blocks, occasional grasp loss,
      and action direction/scale mismatches versus teacher rows. This confirms
      the current boundary: the closed-loop implementation now runs complete
      annotated 300-action episodes with causal target-motion detection, but
      the direct raw Cosmos action chunks are not reliable enough to claim a
      broadly effective world-model controller.
- [x] 2026-06-14 failure-mode aggregation completed with
      `scripts/world_model/summarize_cosmos3_closed_loop_failure_modes.py`.
      Output:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_failure_modes.json`
      and `.md`. The report combines the objective gate and hard action/rebind
      analysis. Current primary failure is
      `direct_raw_cosmos_action_rebind_and_dp_continuability_are_unreliable`.
      On the five hard-screen-2 Cosmos failures, real-state C_pi blocks are
      dominated by `rel_y_abs=203` and `rel_z_abs=195`, with additional
      action-scale/sign flags against same-H5 teacher chunks. This is the
      current technical blocker to method effectiveness; the implementation
      issues already repaired are video length, causal detector provenance,
      explicit Cosmos annotation, and unified no-motion DP behavior.
- [x] 2026-06-14 requirement-level audit completed with
      `scripts/world_model/audit_cosmos3_closed_loop_requirements.py`.
      Output:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_requirement_audit.json`
      and `.md`. The audit reports `current_goal_achieved=false`,
      `passed=6`, `partial=1`, and `failed=1`. Implementation requirements
      pass: the old iter2100 short artifacts are rejected, current iter2700
      videos are full 300/301, target motion is detected causally, Cosmos is
      active on moving cases, takeover annotation is explicit, and no-motion
      uses DP-only through the same detector. The DP handoff item is partial
      because it exists and can succeed, but is not reliable. Method
      effectiveness against pure DP fails, so the user's full objective is
      not complete.
- [x] 2026-06-14 user-requested harder-case comparison is recorded in
      `docs/world_model_task_rebinding/2026-06-14_iter2700_val_hard_puredp_comparison_and_stepgate_probe.md`.
      The completed evidence is: val Cosmos `1/3` versus same-source pure DP
      `3/3`, hard-screen-2 pure DP `9/15`, and Cosmos `1/6` on the six hard
      samples where pure DP failed. A follow-up diagnostic run on hard-screen-2
      index `12` with `--cosmos-step-handoff-gate` was allowed to run until
      allocation `127559` hit its time limit. It has no complete panel/sample
      summary and therefore is not full closed-loop evidence. The partial
      summary reached frame `188` with `success=false`; one step-level C_pi
      pass occurred near frame `132`, but the 8-step frozen-DP handoff drifted
      laterally and the sample fell back out of continuability. Treat this as
      failure localization, not as a changed success rate.
- [x] 2026-06-14 hard closed-loop comparison completed on held allocation
      `127559` for the two hard-screen samples where full-episode pure DP
      failed: `hole_late_continuous_insert` index `1` and
      `hole_late_move_stop` index `3`. Run root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen_20260614/cosmos_iter2700_hard_dp_fail_1_3`.
      The corresponding hard pure-DP screen found failures on continuous-insert
      and move-stop, but successes on extreme constant, fast-shift, reverse,
      and sine samples. Cosmos failed continuous-insert as well:
      full `300/301`, trigger `f134`,
      `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:134,
      WM_ACTIVE:166}`, final success `false`, peg-head-at-hole
      `[-0.0974492, -0.0310004, -0.0281856]`. Cosmos succeeded on move-stop:
      full `300/301`, trigger `f84`,
      `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:84,
      WM_ACTIVE:48, DP_HANDOFF:168}`, final success `true`,
      peg-head-at-hole `[-0.0096748, -0.0019456, -0.0029760]`.
      The Cosmos and pure-DP move-stop videos were both opened directly; pure
      DP remains outside the hole, while Cosmos+handoff is visibly inserted.
      This is the current positive comparison evidence for Cosmos on a hard
      sample where pure DP fails.
- [x] 2026-06-13/14 `iter2700` scenario-diverse dynamic panel completed and
      was compared against the same-source full-episode pure-DP baseline.
      Run root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_panel3_dynamic_20260613_alloc127559`.
      `sample_01_hole_late_constant` is now complete and inspected:
      `final_prefix_frame_index=300`, `final_observed_frames=301`,
      `full_episode_length_ok=true`, detector trigger `f94`,
      `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:94,
      WM_ACTIVE:206}`, final simulator success `false`, and final
      peg-head-at-hole
      `[-0.1030322, -0.0450076, -0.0782132]`. Raw and annotated videos are
      readable/nonblank `301`-frame videos. The opened annotated review sheet
      and final frame show explicit `controller=WM_ACTIVE`,
      `target_motion_detected=True`, and `trigger=94` through frame `300/300`;
      the peg remains outside/beside the moved hole. This proves the full
      length and Cosmos-active annotation requirements for this sample, but it
      is negative controller evidence for insertion success. As of
      `2026-06-13T23:48:58+08:00`, `sample_02_hole_late_reverse` is also
      complete and inspected: `final_prefix_frame_index=300`,
      `final_observed_frames=301`, `full_episode_length_ok=true`, detector
      trigger `f104`, `controller_frame_counts={INIT_OBS:1,
      DP_SCAN_TARGET:104, WM_ACTIVE:40, DP_HANDOFF:156}`, final simulator
      success `true`, and final peg-head-at-hole
      `[0.0061689, 0.0014199, -0.0005607]`. The first five post-trigger
      chunks were Cosmos rebind chunks; the post-Cosmos continuability gate
      became true at prefix `f144`, after which the loop switched to short
      reobserved DP handoff chunks. Raw and annotated videos are
      readable/nonblank `301`-frame videos. The opened annotated review sheet
      and final frame show explicit `WM_ACTIVE` after target-motion detection,
      then `DP_HANDOFF` through frame `300/300`, with the peg visibly inserted
      in the moved hole. This is positive evidence for the intended
      Cosmos-rebind then DP-continuable handoff behavior on this dynamic
      sample. `sample_03_hole_late_fast_shift` also completed:
      `final_prefix_frame_index=300`, `final_observed_frames=301`,
      `full_episode_length_ok=true`, detector trigger `f132`,
      `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:132,
      WM_ACTIVE:32, DP_HANDOFF:136}`, final simulator success `false`, and
      final peg-head-at-hole
      `[-0.0649778, 0.0001627, -0.0006470]`. Its annotated review sheet and
      final frame were opened and show WM then DP handoff, but the peg remains
      outside the moved hole. The completed panel success count is therefore
      Cosmos closed-loop `1/3`.
      Full pure-DP baseline for the same three source H5 files is under
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/dp_full_episode_baseline_iter2700_panel3_dynamic_20260614_alloc127559`.
      It completed all three 300-action/301-frame videos with
      `controller=PURE_DP` and succeeded on `3/3`. This is negative
      comparison evidence for the current iter2700 Cosmos interface: the val
      panel does not prove Cosmos helps and shows two samples where current
      Cosmos closed-loop degrades the frozen DP baseline.
- [x] 2026-06-13 iter2700 full-300 unified-detector closed-loop eval completed
      on held Slurm allocation `127559` without stopping at the old short
      horizon. Run root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_sample00_20260613_alloc127559`.
      Summary reports `final_prefix_frame_index=300`,
      `final_observed_frames=301`, `full_episode_length_ok=true`,
      detector trigger at `f106`, controller counts `DP_SCAN_TARGET=106`,
      `WM_ACTIVE=186`, `DP_HANDOFF=8`, and final simulator success `false`
      with peg-head-at-hole
      `[-0.1056687, -0.0143125, -0.0550163]`. Raw and annotated videos are
      both readable 301-frame videos; review sheets were opened directly and
      show the peg remains outside the moved hole through the end. Evidence
      note:
      `docs/world_model_task_rebinding/2026-06-13_iter2700_full300_unified_closed_loop_eval.md`.
- [x] 2026-06-13 no-motion control-boundary check completed with the
      same unified detector after fixing the pretrigger terminal handling.
      Run root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_static_none_fix1_20260613_alloc127559`.
      Summary reports `final_prefix_frame_index=300`,
      `final_observed_frames=301`, `full_episode_length_ok=true`,
      `completed_iterations=0`, `wm_active_frame_count=0`,
      `dp_active_frame_count=300`, `controller_frame_counts={INIT_OBS:1,
      DP_SCAN_TARGET:300}`, `triggered=false`, and
      `future_target_motion_scan_after_terminal.would_trigger=false`.
      There is no `cosmos_live_prefix`/inference directory. The opened
      annotated sheet and final frame show `controller=DP_SCAN_TARGET`,
      `target_motion_detected=False`, `trigger=none`, and `wm_active=False`
      through frame `300/300`. The frozen DP still failed final insertion on
      this seed, but the controller-selection requirement is satisfied:
      Cosmos is bypassed only because the same causal detector never fires,
      not because static/no-motion labels use a separate branch.
- [x] 2026-06-13 no-motion misunderstanding repaired. No sample may be
      predeclared as `DP-only` from a static/no-motion label, and there is no
      separate static-vs-dynamic protocol. The controller uses one causal
      target-motion detector over observed history/state. If it never fires,
      frozen DP runs the full episode as the natural result of the same rule.
      The live receding loop summary already records this boundary, and
      `infer_prefix_role` no longer maps `scenario == "none"` to
      `static_monitor`/`no_target_motion_static_scenario`; no target motion is
      represented as the same observed no-motion pre-trigger condition used in
      dynamic samples before motion.
- [x] 2026-06-13 post-failure root-cause audit completed after the
      user-directed closed-loop eval. No simple action-indexing bug was found:
      `shift_action=1` is the action diffusion schedule shift, not a temporal
      offset, and action row `f` is still the correct first future action when
      the observed prefix ends at frame `f`. The stronger current diagnosis is
      training/interface distribution mismatch: exported `prefix_role` labels
      are semantically mixed with actual physical `mode`, the active SFT root
      contains only sparse hand-picked prefix masks per trajectory, and live
      receding eval quickly becomes off-policy after the model's own early
      under-reaction. The current v7_733 data teach successful DP-filtered
      trajectories, but not enough late/outside-hole recovery chunks of the
      form "observe current miss, predict the next 8-step correction, reobserve
      again." Later `iter2700` evidence also remains controller-blocked by
      visual review, so the long-horizon closed-loop failure is not best
      explained as stopping at `iter2100` too early. Detailed evidence is in
      `docs/world_model_task_rebinding/2026-06-13_corrected_live_receding_panel_failure.md`.
      Do not launch more smoke training from this state.
- [x] Added a reproducible read-only distribution audit for the current SFT
      condition root:
      `scripts/world_model/audit_cosmos3_receding_training_distribution.py`.
      It writes
      `docs/world_model_task_rebinding/2026-06-13_receding_training_distribution_audit.json`
      and confirms the diagnosis numerically: `1193/2899` role/mode
      mismatches, no action condition-mask errors, and `1287` late-rebind proxy
      rows. The corresponding repair plan is
      `PLAN/cosmos3_300f_world_model/07_post_closed_loop_failure_repair.md`;
      do not use this as permission to launch more training without user
      approval.
- [x] Implemented and probed the first code repair for that plan. Clean export
      mode uses physical `mode` as controller-facing `prefix_role`, stores the
      old/curriculum source as `sampled_prefix_role`, and can add dense
      8-frame receding prefix masks while preserving full `301/300` rows. A
      two-episode `/tmp` probe produced `0` role/mode mismatches; no training,
      rendering, or new live eval was launched.
- [x] The receding distribution audit now has hard-gate options for the
      repair path. When `--prefix-role-source physical_mode` is used, any
      role/mode mismatch exits nonzero; `--require-no-condition-mask-errors`
      and `--min-late-rebind-candidates` cover the other two failure modes.
      The main Slurm SFT wrapper now uses these flags during condition audit,
      and the clean/dense preflight wrapper defaults
      `MIN_LATE_REBIND_CANDIDATES=1`.
- [x] 2026-06-13 user-directed closed-loop eval advanced without further
      smoke training. The 3-chunk prompt-fixed sample-00 result was not enough
      by itself to declare the sample complete, because the source teacher
      first inserts at frame `166`. A longer corrected live-receding run was
      therefore completed:
      `live_receding_promptfix_sample0_longhorizon_iter2100_20260613`.
      It used checkpoint `iter_000002100`, target-motion-onset prefix
      `f106`, live frozen-DP pretrigger, target-only source replay, `12`
      receding Cosmos calls, 8-step chunks, and real-state `C_pi` for the
      optional 32-step DP handoff. Final frame was `f202`; final live success
      was `false`; final peg-head-at-hole was
      `[-0.1423, 0.0004, 0.0101]`; DP handoff executed `0` steps. The source
      teacher for the same H5 is inserted by `f166` and at `f202` is near
      `[-0.0056, 0.0005, -0.0022]`, inserted `true`. The final gate block is
      justified: grasp and y pass, target speed is zero, but x fails the
      static-DP `min_rel_x=-0.1342566` threshold and z fails
      `max_abs_z=0.0038843`. The opened panel sheet and dense 18-frame rollout
      sheet show the same failure: the robot follows the moved target but the
      peg remains outside the hole. Treat this as corrected closed-loop
      negative evidence for the current checkpoint/action path. Do not launch
      more smoke training from this result; the next work is failure analysis
      of action/rebind capability or a user-directed method change.
- [x] 2026-06-13 direct corrected closed-loop eval after the user's instruction
      to stop smoke/training and push eval:
      `live_receding_panel10_corrected_iter2100_20260613_161006`.
      The panel used the corrected live-receding interface, checkpoint
      `iter_000002100`, causal frozen-DP pretrigger, source target-only replay,
      3 receding iterations, 8-step Cosmos chunks, and gated 32-step DP
      handoff. It was stopped after clear corrected-protocol failure on the
      first moving-target samples: sample `00` (`hole_late_move_stop`) failed
      after 3 iterations with final peg-head-at-hole
      `[-0.2652, 0.0979, 0.0020]`, sample `01`
      (`hole_late_constant`) failed after 3 iterations with
      `[-0.2281, 0.0420, 0.0020]`, and sample `02`
      (`hole_late_reverse`) was already failing after one iteration with
      `[-0.3051, 0.1426, -0.0081]`. DP handoff executed `0` steps in all
      inspected completed/partial samples because real-state `C_pi` correctly
      blocked non-continuable states. Inspected live-prefix manifests all used
      `source=provided_live_history_action_rows_only`, so this is not the old
      source-H5 conditioning bug. A contact sheet was opened directly:
      `closed_loop_failure_sample00_01_live_rollout_sheet.png`, showing peg/hand
      failing to rebind to the moved hole. Stop further eval/training variants
      until this failure is analyzed through action-chunk/source-action and
      generated-video/sidecar inspection.
- [x] 2026-06-13 failure localization completed for the corrected panel above.
      Generated-vs-live video sheet
      `closed_loop_failure_live_vs_cosmos_predictions_sheet.png` was opened
      directly. Cosmos predicts the target/hole motion visually, but it does
      not generate a convincing robot/peg rebind trajectory. Numeric artifact
      `closed_loop_failure_action_sidecar_analysis.json` compares predicted
      chunks with source-teacher chunks and predicted sidecars with real
      post-execution states. Sample `00` iter `2` is especially clear:
      predicted mean absolute xyz action is `[0.0393, 0.0494, 0.0200]`, while
      the source teacher rows require `[0.1299, 0.1217, 0.0323]`, and the real
      final peg-head-at-hole remains `[-0.2652, 0.0979, 0.0020]`. A live
      interface schema drift was also found and fixed: when
      `history_action_path` is used, `build_cosmos3_live_prefix_wam_input.py`
      now reconstructs the same current-geometry caption used by SFT/eval from
      the live history row. A builder-only probe confirmed the prompt now
      includes current hole/peg/TCP/peg-head-at-hole and observed target
      velocity for prefix `106`. Do not resume SFT or launch broad panels from
      this evidence alone; the next allowed action is a minimal prompt-fixed
      corrected closed-loop check, then capability/data/action-head diagnosis
      if it still fails.
- [x] 2026-06-13 minimal prompt-fixed corrected closed-loop recheck completed
      on held allocation `127350`, step `48`, with no SFT continuation:
      `live_receding_promptfix_sample0_iter2100_20260613`. The live-prefix
      prompt now contained current live geometry, but sample `00`
      (`hole_late_move_stop`) still failed after the same 3 receding chunks:
      final peg-head-at-hole `[-0.2666, 0.0985, 0.0029]`, final success
      `false`, and DP handoff blocked by real-state `C_pi` at every iteration.
      The opened contact sheet and
      `promptfix_live_vs_cosmos_predictions_sheet.png` show the same qualitative
      failure: target moves, but the robot/peg do not rebind to the moved hole.
      `promptfix_action_sidecar_analysis.json` again shows under-reaction in
      the late segment; iter `2` predicted mean absolute xyz action
      `[0.0380, 0.0538, 0.0211]` versus source-teacher
      `[0.1299, 0.1217, 0.0323]`. This rules out the prompt mismatch as the
      main cause. Treat the current checkpoint/interface as controller-negative
      and move to model/action-objective/data-hardness diagnosis rather than
      more smoke training or broad panels.
- [x] 2026-06-13 user-requested closed-loop re-audit found a real live-history
      conditioning bug. The live loop passed `HISTORY_ACTION_PATH` for the
      current real rollout, but the outer environment still carried
      `SOURCE_H5`; the live-prefix wrapper passed both to
      `build_cosmos3_live_prefix_wam_input.py`, and the builder preferred
      source-H5 rows. That made the structured WAM condition come from the
      teacher/source trajectory instead of the real reobserved rollout after
      the first Cosmos chunk. The in-progress
      `live_receding_shortdp_state_machine_iter2100_sample00_20260613_153241`
      run was interrupted without `scancel` and must be treated as invalid
      eval evidence. Code is repaired so live history takes precedence, the
      wrapper does not pass source H5 when live history is present, and the
      live loop clears inherited `SOURCE_H5` for per-reobservation Cosmos
      calls. A local builder probe confirmed
      `source=provided_live_history_action_rows_only`,
      `condition_frame_indexes_action=0..177`, and `179` prefix video frames
      for prefix frame `178`.
- [x] 2026-06-13 audit after the user stopped further SFT: the current live
      wrapper name is misleading. It is a guarded one-shot diagnostic, not the
      planned receding controller. It restores one source-H5 prefix state,
      executes one precomputed Cosmos chunk, then optionally runs frozen DP for
      a resume horizon. It does not rebuild a causal Cosmos prefix from each
      new live observation, does not rerun Cosmos after the action prefix, does
      not replay the source H5 external target motion after prefix reset, and
      does not check `C_pi`/continuability before long DP takeover. Therefore
      the existing `iter1800` and `iter2100` `8+96` panels are not evidence
      that the planned closed-loop DP handoff works or fails.
- [x] Guard the diagnostic path against future misread: long DP takeover
      (`dp_resume_horizon > action_exec_horizon`) now requires an explicit
      diagnostic override, panel runs no longer default to
      `VISUAL_REVIEW_STATUS=pass`, and manifests/summaries mark the current
      interface as `method_evidence_allowed=false`.
- [x] Add the first real receding-interface building block:
      `scripts/world_model/build_cosmos3_live_prefix_wam_input.py` plus
      compute-node wrapper
      `scripts/slurm/run_cosmos3_live_prefix_policy_inference_in_allocation.sh`.
      The builder writes a one-sample Cosmos policy JSONL from the latest
      observed prefix video and history action/state rows. Only RGB prefix
      latent frames and action/state rows strictly before
      `prefix_frame_index` are clean conditions; future target/peg/TCP rows
      are zero/unconditioned. A local `/tmp` builder probe on the iter2100
      sample-00 source at prefix frame `97` produced a valid `300x32` action
      JSON with `condition_frame_indexes_action=0..96` and vision latent
      conditions `0..24`; no Cosmos inference or simulator rollout was run.
- [x] Add `scripts/world_model/extract_cosmos3_policy_action_chunk.py` and
      connect it to the live-prefix inference wrapper. It parses a Cosmos
      policy `sample_outputs.json`, selects rows
      `[prefix_frame_index, prefix_frame_index + action_exec_horizon)`, and
      de-normalizes only robot action columns `0..6`. A local read-only probe
      on iter2100 sample-00 extracted rows `97:105`, `8` steps, all finite.
      Sidecar columns remain diagnostics/scoring context and are not simulator
      state.
- [x] Tighten the causal RGB boundary for live-prefix inference. The builder
      now rejects prefix videos whose frame count is not exactly
      `prefix_frame_index + 1` unless an explicit non-method diagnostic
      override is passed. Added `scripts/world_model/write_video_prefix.py`;
      a `/tmp` probe cut the iter2100 sample-00 reference to `98` frames for
      prefix frame `97`, and the builder accepted only that prefix-only video.
- [x] Add `scripts/world_model/run_cosmos3_live_receding_loop.py` as the
      compute-node-only scaffold for the intended execution loop. In dry-run
      mode it restores the source prefix, writes the observed prefix-only
      video and raw `300x32` WAM history. With `--run-cosmos-inference`, each
      iteration can call the live-prefix wrapper, execute only the extracted
      short robot-action chunk, append real env observations into history, and
      reobserve before the next Cosmos call. This script has passed syntax
      checks and one compute-node dry-run on Slurm job `127350`:
      `live_receding_dryrun_iter2100_sample00_20260613_131044`. The dry-run
      restored prefix frame `97`, wrote `98` observed prefix frames and a raw
      `300x32` history with rows `0..96` conditioned and row `97` still zero/
      unconditioned. A follow-up builder check consumed those dry-run files
      successfully. Cosmos inference was disabled, so this is interface
      evidence only, not controller evidence.
- [x] Repair the live dynamic-world semantics in
      `run_cosmos3_live_receding_loop.py`. Source fix3 H5s do not contain an
      autonomous ManiSkill target-motion controller; their moving-hole
      trajectories were created by explicitly setting the box pose after each
      action step. The live receding loop now defaults to replaying only
      `env_states/actors/box_with_hole[frame+1]` after each live robot action,
      then reads live task state and appends it to the WAM history. Robot and
      peg state are not restored from source after the prefix. The loop also
      keeps a two-frame post-replay state-observation history for frozen-DP
      handoff, so DP does not see the stale observation returned by
      `env.step()` before external target replay.
- [x] Run the first post-repair single-sample live receding smoke:
      `live_receding_statsprofile32_gate_iter2100_sample00_20260613_1518`.
      It performed three causal Cosmos calls at prefixes `97`, `105`, and
      `113`, replayed source target actor pose from frames `98..121` after
      live robot actions, used the static-DP success-manifold gate for a
      possible 32-step handoff, and never allowed DP because the real state was
      not continuable. Final live state was `success=false` with
      peg-head-at-hole about `[-0.2527, 0.0825, -0.0031]`. The opened contact
      sheet shows the target moving while the robot/peg fail to rebind to the
      new hole. This validates the repaired eval semantics and makes the
      current result more negative, not better.
- [x] Run the first live-prefix Cosmos inference smoke with real simulator
      execution and saved live video:
      `live_receding_oneiter_cosmos_iter2100_sample00_with_live_video_20260613_1329`.
      This used the `iter_000002100` checkpoint, sample `0`, prefix frame
      `97`, and a single `8`-step action chunk. The wrapper built the causal
      prefix, invoked Cosmos once, extracted de-normalized robot action rows
      `97:105`, executed them in the live env, and wrote
      `live_observed_rollout.mp4`. The final live state is still
      `success=false`; peg-head-to-hole moved from roughly
      `[-0.194, 0.057, -0.030]` to `[-0.140, 0.031, -0.016]`. A contact sheet
      containing the input prefix, Cosmos future video, and real live rollout
      was opened directly. This proves the new interface can run one causal
      reobservation/action-execution step, but it is not closed-loop method
      success and it does not justify more SFT.
- [x] Add conservative live `C_pi` handoff support to
      `scripts/world_model/run_cosmos3_live_receding_loop.py`. Frozen DP is
      now disabled by default and can execute only when `--dp-handoff-horizon`
      is positive and the real live state passes the gate: peg held, peg-head
      close to the current hole frame, and recent hole speed below threshold.
      DP handoff writes real executed actions and real post-step states back
      into the live history. Sidecar predictions are still never written into
      simulator state.
- [x] Run a real 5-iteration receding smoke from `iter_000002100`, sample `0`:
      `live_receding_fiveiter_gated_dp_iter2100_sample00_20260613_1353`.
      Each iteration rebuilt the causal prefix from the real live rollout
      (`f097`, `f105`, `f113`, `f121`, `f129`), conditioned only observed
      action/state rows up to the previous step, invoked Cosmos, and executed
      one 8-step chunk. Final live state after `40` Cosmos steps was
      `success=false`, with peg-head-to-hole about
      `[-0.1085, 0.0067, -0.0012]`. The conservative `C_pi` gate correctly
      blocked DP in all five iterations because insertion-axis distance
      stayed outside the allowed range. The contact sheet was opened directly:
      `live_receding_fiveiter_gated_contact_sheet.png`.
- [x] Record a relaxed-gate diagnostic, not method evidence:
      `live_receding_dpstatic32_gate_iter2100_sample00_20260613_1410`.
      This manually loosened `continuability_min_rel_x` to `-0.1343` and used
      explicit `target_pre_motion`, so it is not a valid controller gate. It
      triggered DP after the third Cosmos chunk and ran `32` DP steps. Final
      live state improved to about `[-0.0197, 0.0027, -0.0003]` but still
      had `success=false`. The reviewed contact sheet shows the peg close to
      the hole, not completed insertion. This reinforces that premature or
      relaxed DP handoff cannot be reported as success.
- [x] Replace the hand-loosened gate path with an explicit static-DP
      continuability profile option in `run_cosmos3_live_receding_loop.py`.
      The script now accepts `--continuability-stats-json` and derives
      `min_rel_x`, `max_abs_y`, and `max_abs_z` from
      `within_<horizon>_steps_to_first_success` in
      `dp_static_continuability_stats.json`, then records the profile source,
      horizon, quantiles, sample count, and final thresholds in both the run
      summary and each gate record. A local loader check on the existing
      static-DP stats for horizon `32`, x lower quantile `0.01`, and y/z abs
      quantile `0.95` produced thresholds
      `min_rel_x=-0.1342566`, `max_abs_y=0.0098417`,
      `max_abs_z=0.0038843`, preserving the CLI x safety cap
      `max_rel_x=0.04`. This is a data-calibrated diagnostic `C_pi` boundary,
      not a learned continuability model or success evidence.
- [x] Smoke the data-calibrated profile path on held compute allocation
      `127350` without resuming SFT:
      `live_receding_statsprofile32_gate_iter2100_sample00_20260613_1518`.
      This used `iter_000002100`, sample `0`, `--prefix-role auto`, three
      receding Cosmos calls at prefixes `97`, `105`, and `113`, and the
      static-DP 32-step profile above. The run rebuilt live prefixes and
      history after each real action chunk. The automatic role stayed
      `target_pre_motion` for prefixes `97` and `105`, then switched to
      `target_motion_observed` at prefix `113`. The profile gate blocked DP at
      every iteration: after chunk 0 the state was too far/laterally off, after
      chunk 1 the target was still moving and y/z were off, and after chunk 2
      the target was still moving and peg-head-to-hole worsened to about
      `[-0.2527, 0.0825, -0.0031]`. Final live simulator success was `false`.
      The contact sheet `live_receding_statsprofile32_contact_sheet.png` was
      opened directly; the true live rollout shows no insertion and the
      peg/hand failing to keep up after target motion. This is
      controller-negative but confirms the intended receding/evidence path now
      runs without silent long-DP takeover.
- [x] Add automatic observed-prefix role inference for future live receding
      runs. `--prefix-role auto` derives `static_monitor`,
      `target_pre_motion`, `target_motion_observed`, `target_post_motion`, or
      `peg_recovery` from observed live history. A bug was fixed where one
      noisy current `is_grasping=false` at the restore boundary caused
      `peg_recovery`; the rule now requires a stable lost-grasp condition
      after recent history no longer shows grasp, or an explicit peg
      perturbation scenario. A local function check on the f97 sample history
      now returns `target_pre_motion`.
- [x] Do not start live DP/controller evaluation from the current v7_733
      follow-up SFT until a checkpoint passes all three gates:
      strict same-length generated artifacts, generated-RGB readout/profile,
      and direct visual review of all validation sheets/videos. This gate was
      first satisfied only for a guarded diagnostic live smoke at `iter1800`,
      not for a controller-success claim.
- [x] The latest fix1-recipe root
      `sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`
      first ended at Slurm wall time after rank-0 iteration `743`, then was
      resumed. It reached `iter_000001500`, failed visual handoff, and was
      auto-resumed on 4-H200 Slurm job `127281` (`server31`) toward
      `iter_000002100`. The current 4-H200 job is the only SFT checkpoint
      writer.
- [x] The active fix1-recipe gates through `iter_000001500` completed and are
      controller-negative. Strict artifacts and generated-RGB readout/profile
      passed structurally at `iter_000001500`, but direct review failed sheets
      `03`, `04`, `08`, and `09`: late robot/peg/target relative geometry
      drifts and is not DP-resumable. `closed_loop_allowed=false` is recorded
      with `visual_review_status=fail`.
- [x] The `iter_000001800` gate completed through strict eval, generated-RGB
      readout/profile, manual visual review, and short live smoke. Strict
      artifacts and readout passed structurally; manual review of all `10`
      sheets recorded `pass_with_caution` and allowed diagnostic smoke only.
      Sample `0` smoke executed `8 + 8` live steps and ended
      `success=false`. After repairing DP resume to recompute repeated
      8-step DP chunks, sample `3` smoke executed `8 + 32` live steps and also
      ended `success=false`. Therefore `iter1800` is useful evidence that the
      gate/smoke code path runs, but it is not controller success evidence.
- [x] Run an `iter1800` diagnostic live-smoke panel with `8` Cosmos action
      steps plus `32` recomputed frozen-DP resume steps for all `10` validation
      samples:
      `closed_loop_smoke_iter_000001800_panel10_dp32_recompute_20260613_0810`.
      The live wrapper completed for all samples, but only `1/10` reached
      final simulator success, and that one was a `target_post_motion`
      continuous-insert case that began essentially at the hole. The inspected
      contact sheet records this as negative diagnostic evidence, not a method
      success.
- [x] Run a longer `iter1800` diagnostic live-smoke panel with `8` Cosmos
      action steps plus `96` recomputed frozen-DP resume steps:
      `closed_loop_smoke_iter_000001800_representative_dp96_recompute_20260613_0816`.
      It reached final simulator success on `6/10` samples
      (`hole_late_move_stop`, `hole_late_constant`, `hole_late_fast_shift`,
      `hole_late_sine`, one `hole_late_continuous_insert`, and one static
      `none`) and failed on `4/10` (`hole_late_reverse`, one
      `hole_late_continuous_insert`, `peg_drop`, and static-late `none`).
      The inspected contact sheet shows the successes are plausible insertions
      after long DP resume, but this is still not full receding-Cosmos method
      evidence: it uses one precomputed Cosmos action chunk followed by a long
      DP takeover and does not re-run Cosmos after each live observation.
- [x] `iter_000002100` completed the full generated-artifact/readout/profile
      gate and direct visual review. The ten review sheets were opened
      directly; verdict was `pass_with_caution` (`8` pass, `2`
      pass-with-caution, `0` fail), and
      `closed_loop_gate_visual_review.json` records
      `closed_loop_allowed=true`. This gate only permits live smoke; it is
      not controller success evidence.
- [x] Run the comparable `iter2100` diagnostic live-smoke panel with `8`
      Cosmos action steps plus `96` recomputed frozen-DP resume steps:
      `closed_loop_smoke_iter_000002100_representative_dp96_recompute_20260613_0928`.
      The panel completed for all ten samples and reached final simulator
      success on `5/10` samples. Successes were samples `0`, `1`, `3`, `6`,
      and `8`; failures were `2`, `4`, `5`, `7`, and `9`. The inspected
      contact sheet
      `panel_contact_sheet_full10_dp96_start_cosmos_mid_final.png` agrees
      with the live metrics: success rows show plausible final insertion, and
      failed rows remain visibly outside or misaligned with the hole. This is
      controller-negative relative to the required method because it still
      uses one precomputed Cosmos chunk followed by long DP takeover and does
      not improve over the previous `iter1800` DP96 diagnostic.
- [x] `iter_000002400` completed strict eval/readout/profile and direct visual
      review, but it is blocked for closed-loop. Structural artifacts passed
      under the full `301` RGB/state frame and `300x32` action contract, but
      controller-facing metrics regressed versus `iter2100`
      (`mean_final_hole_pos_error_m=0.1161`, future hole/peg/TCP RMSE
      `0.0647/0.0647/0.0603`, robot-action future RMSE `0.7619`). Direct
      review of all ten sheets failed samples `00`, `04`, and `08` for
      late/final robot-peg-hole relative-geometry drift. The gate file records
      `closed_loop_allowed=false` with reason
      `explicit_visual_review_not_passed`; do not launch smoke from
      `iter2400`.
- [x] `iter_000002700` completed strict eval/readout/profile and direct visual
      review. Structural artifacts and generated-RGB readout/profile passed,
      and numerical metrics recovered versus `iter2400`
      (`mean_final_hole_pos_error_m=0.0669`, future hole/peg/TCP RMSE
      `0.0410/0.0477/0.0466`, robot-action future RMSE `0.6147`, PSNR
      `22.37 dB`). Direct review still failed the controller gate: samples
      `03`, `05`, `07`, and `08` show unsafe late robot-peg-hole geometry,
      including `fast_shift`, `peg_drop`, and a `none/static_monitor` false
      drift with final hole error about `0.266m`. The gate file records
      `closed_loop_allowed=false` with reason
      `explicit_visual_review_not_passed`; do not launch closed-loop smoke from
      `iter2700`.
- [x] The old 2-H200 fallback/watch logic is no longer the active path for
      `iter_000001500`; the 4-H200 auto-resume-after-checkpoint watcher
      launched the current `1500 -> 2100` continuation without concurrent
      checkpoint writers. Any future fallback must still preserve the
      no-concurrent-writers rule.
- [x] The already evaluated latest fix1-recipe `iter_000000300` and
      `iter_000000600` checkpoints are not controller-ready. They preserve
      prefix conditions and pass the 301-frame / 300-action structural gate,
      but future action/state/video quality and visual peg/contact continuity
      remain negative. Iter300 is the best qualitative sanity checkpoint so
      far; iter600 is worse despite lower validation loss.
- [x] The corrected active-v7 panel for `iter_000000300`
      (`eval_full_episode_wam_iter_000000300_v7panel`) also remains
      controller-blocked. It covers every scenario present in val
      (`hole_late_*`, `none`, `peg_drop`; `peg_disturb` is absent from val),
      passes strict artifacts/readout/profile, and then blocks at visual
      review because target-motion onset false-fires and final contact/relative
      geometry is not executable. The closed-loop gate verdict is
      `closed_loop_allowed=false` with reason
      `explicit_visual_review_not_passed`.
- [x] Historical `normactive_clip1` iter900/iter1200 notes are not the active
      closed-loop gate. That run was rejected because it did not use the
      overfit-approved fix1 action recipe.

## Required Execution Contract

- [x] Add a conservative closed-loop gate checker:
      `scripts/world_model/check_cosmos3_closed_loop_gate.py`. It reads a
      Cosmos eval root, strict artifact inspection, generated-RGB
      readout/profile, and an explicit visual-review verdict. It blocks by
      default unless all three gates pass and should be called by any future
      live closed-loop wrapper before it touches DP or the simulator.
- [x] Add a compute-node-only guarded closed-loop preflight wrapper:
      `scripts/slurm/run_cosmos3_receding_closed_loop_in_allocation.sh`
      and Python entry point
      `scripts/world_model/run_cosmos3_receding_closed_loop.py`. They refuse
      login-node execution, record checkpoint/condition/DP/gate manifests, and
      stop before simulator work when the gate fails.
- [x] Implement the gate-passed live smoke branch in
      `scripts/world_model/run_cosmos3_receding_closed_loop.py`. If all gates
      pass and `MODE=smoke`, the entry point now constructs the real ManiSkill
      state-DP env inside a compute allocation, restores the selected source
      H5 state at `chunk_start`, executes only a short `<=8` de-normalized
      Cosmos robot-action chunk, optionally executes one short frozen-DP resume
      horizon, records action-space clipping/validation, live
      `base_env.evaluate()` metrics, and writes a short video when requested.
      This branch has not been run as controller evidence yet because current
      SFT gates still fail visual review.
- [x] Smoke the guarded preflight inside held compute allocation `127120` as a
      single Slurm task (`step 127120.7`) on latest fix1-recipe iter300. The
      DP manifest contract and full-episode condition/normalization contract
      passed; the closed-loop gate returned `closed_loop_allowed=false` because
      explicit visual review is `fail`. The wrapper exited with expected code
      `40` and did not start live environment rollout.
- [x] Extend the guarded preflight to build an offline candidate robot-action
      chunk preview from Cosmos eval output before the gate decision. Compute
      smoke step `127120.8` on latest fix1-recipe iter300 loaded a `300x32`
      predicted action tensor, used `future_start_action_index=131`, selected
      an `8`-step chunk `[131,139)`, de-normalized only columns `0..6` with
      `normalization_stats.json`, and wrote
      `candidate_action_chunk_preview.json`. The preview contract passed
      (`finite=true`, `chunk_steps=8`, shape `[300,32]`), then the wrapper
      still stopped with expected exit code `40` because the visual gate
      remains failed. No live environment rollout was started.
- [x] Extend the guarded preflight to validate the frozen DP checkpoint
      structure without constructing ManiSkill envs. Compute smoke step
      `127120.10` loaded
      `experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt`
      with `weights_only=True`, verified checkpoint keys
      `agent/ema_agent/args/iteration`, confirmed `ema_agent` is nonempty,
      and checked the saved args against the DP manifest
      (`PegInsertionSide-v1`, `pd_ee_delta_pose`, `obs_horizon=2`,
      `act_horizon=8`, `pred_horizon=16`, `max_episode_steps=300`). DP
      checkpoint, DP manifest, condition contract, and action preview all
      passed; the wrapper still exited with expected code `40` solely because
      the Cosmos visual gate remains failed. No live `env.step` rollout was
      started.
- [x] Extend the guarded preflight to recover live-smoke source context from
      eval artifacts. The script now maps an eval sample through
      `eval_input_manifest.json` and the original condition JSONL row to
      recover `source_h5`, `source_uuid`, scenario, prefix role/frame, RGB
      path, action path, and task label paths. A local read-only check on
      `iter_000000600` recovered
      `canonical_h5/peg_drop_seed705165_idx1029.fix3/peg_drop_seed705165_idx1029.h5`
      for the first eval sample. In `mode=smoke`, missing `source_h5` is now a
      hard preflight failure before any live simulator work can start.
- [x] Extend the guarded preflight source-context check to validate the source
      H5 episode structure before any live simulator work. A local read-only
      function smoke on active fix1-recipe `iter_000000900` recovered
      `canonical_h5/hole_late_move_stop_seed3280649_idx2518.fix3/hole_late_move_stop_seed3280649_idx2518.h5`
      and verified `actions=[300,7]`, peg/hole/robot env-state groups with
      `301` frames, and TCP/peg/hole slot trajectories with `301` frames.
      This is a structural preflight only; it does not restore simulator
      state or prove controller success.
- [x] Smoke-check the updated `MODE=smoke` failed-gate behavior on held compute
      job `127288`, step `127288.24`, using `iter_000001500` with
      `VISUAL_REVIEW_STATUS=fail`. The wrapper exited with expected code `40`,
      reported `closed_loop_gate_blocked`, recovered source context and source
      H5, and did not write `live_smoke_result.json`; therefore the new
      gate-passed live branch cannot be reached through a failed visual gate.
- [x] Use the frozen static DP checkpoint only through its real ManiSkill
      state-policy interface:
      `experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt`.
      Its manifest fixes `PegInsertionSide-v1`, `pd_ee_delta_pose`,
      `obs_horizon=2`, `act_horizon=8`, and `max_episode_steps=300`. Current
      code validates the saved checkpoint format and selected `ema_agent`
      state key before the gate, and the gate-passed `MODE=smoke` branch loads
      the same DP agent for the optional short DP-resume horizon. Actual DP
      env interaction remains unrun until a future Cosmos checkpoint passes
      the closed-loop gate.
- [ ] Observe live RGB/state, build a causal Cosmos prefix from the latest
      observed frames/state, predict the remaining or short future horizon,
      execute only a short action prefix, then reobserve and refresh. The
      default action execution prefix should be `<=8` steps to match DP
      `act_horizon`; never execute a one-shot 300-step open-loop Cosmos
      trajectory as method evidence.
- [ ] The authority for success is the live simulator final state plus video
      review. A restored planner state, generated RGB, or readout-only success
      is diagnostic, not controller evidence.

## Cosmos Action Contract

- [x] Cosmos WAM eval output is a normalized `300x32` sequence. Columns
      `0..6` are robot actions; columns `7..31` are predicted task-state
      sidecars.
- [x] Before live execution, load
      `full_episode_wam_conditions_fix3_v7_733_rgb_300step_20260612_0245/normalization_stats.json`
      and de-normalize only columns `0..6` with the matching vector-name
      stats. Clip/validate against the ManiSkill action space before
      `env.step`.
      Current preflight implements the de-normalization preview and verifies
      finite `<=8` robot-action chunks. The gate-passed live branch resolves
      the ManiSkill action space, records pre-clip violation, and clips only
      when `CLIP_LIVE_ACTIONS=true`.
- [ ] Treat columns `7..31` as predicted task-state diagnostics and controller
      scoring/readout context only. They must not be written into simulator
      state, used as oracle object poses, or used to bypass RGB/state
      re-observation.
- [ ] Preserve the 301 RGB/state frame and 300 action-step accounting in all
      manifests. A receding controller may execute short prefixes, but the
      generated/evaluated sample contract must not become 128/129-frame clips.

## Implementation Steps After A Passed Gate

- [x] Add a compute-node-only wrapper, e.g.
      `scripts/slurm/run_cosmos3_receding_closed_loop_in_allocation.sh`, that
      records job id, checkpoint path, condition root, DP checkpoint, action
      normalization stats, validation seeds/scenarios, and evidence boundary.
      Current implementation is guarded by generated artifact/readout/visual
      gates and can run the short live smoke only after those gates pass.
- [x] Add a Python entry point, e.g.
      `scripts/world_model/run_cosmos3_receding_closed_loop.py`, with an
      initial one-env smoke mode:
      reset dynamic ManiSkill episode, maintain real observation history,
      call the Cosmos full-episode policy inference for a causal prefix,
      de-normalize and execute at most `8` robot actions, reobserve, and
      repeat until termination or `300` steps.
      Current implementation performs compute-node/gate/contract preflight,
      refuses weak checkpoints, restores the selected source H5 prefix state,
      executes a short de-normalized Cosmos action chunk, then can recompute
      repeated 8-step frozen-DP resume chunks from live observations. It has
      run as a negative diagnostic on `iter1800`; a full receding Cosmos
      re-prediction loop after every live observation is still pending.
- [ ] Save for every rollout: live RGB video, per-step executed robot action,
      generated action/state sidecars, real simulator metrics, target/peg/TCP
      readout trajectory, final success predicates, and a review sheet.
- [x] Add a reusable compute-node-only live-smoke panel wrapper for future
      gated checkpoints:
      `scripts/slurm/run_cosmos3_closed_loop_panel_in_allocation.sh`, plus
      summarizer/contact-sheet tool
      `scripts/world_model/summarize_cosmos3_closed_loop_panel.py`. This
      preserves the current diagnostic boundary: it serializes the existing
      gated `MODE=smoke` wrapper across validation sample indices, executes
      one `<=8`-step Cosmos action chunk plus a configured recomputed-DP
      resume horizon, records live simulator metrics/videos, and generates a
      start/mid/final contact sheet for agent review. It does not implement
      full online receding Cosmos re-inference and must not be reported as
      method-level controller evidence. The summarizer was tested read-only
      on the existing `iter2100` DP96 panel and reproduced `10/10` completed,
      `5/10` final success, with success indices `[0,1,3,6,8]`.
- [x] Add the corrected live-receding panel entry point:
      `scripts/world_model/run_cosmos3_live_receding_panel.py` plus
      `scripts/slurm/run_cosmos3_live_receding_panel_in_allocation.sh`. This
      wrapper does not call the old one-shot closed-loop path. For each
      selected eval-manifest sample it resolves the frozen v7 source H5 and
      runs the real `run_cosmos3_live_receding_loop.py`, replays only source
      target actor motion with `--external-target-mode source_env_state`,
      defaults `--prefix-role auto`, and allows DP only after the live
      static-DP success-manifold continuability gate. The panel writes
      per-sample summaries and a contact sheet, and sets
      `method_evidence_allowed=false` until a scenario-diverse compute-node
      run plus direct video review exists. Local checks passed:
      `py_compile` for the new Python entry point and `bash -n` for the new
      Slurm wrapper. No training, Cosmos inference, or simulator rollout was
      launched for this code-path check.
- [x] Correct the prefix-start mistake called out by the user on 2026-06-13.
      A fixed initial `--prefix-frame-index` from the eval manifest is not the
      intended closed-loop protocol; it is only a diagnostic override. The
      live loop now supports `--prefix-start-mode target_motion_onset`, which
      causally detects the first usable Cosmos prefix from observed
      `box_with_hole` pose motion using delta/speed thresholds and a
      consecutive-frame rule. The detected prefix is the frame where the
      motion rule becomes observable, not the first moving frame retroactively.
      The live-receding panel wrapper now defaults to
      `PREFIX_START_MODE=target_motion_onset`; `PREFIX_START_MODE=manifest`
      must be set explicitly to reproduce fixed-prefix diagnostics. A local
      H5 check on source sample
      `hole_late_move_stop_seed3280649_idx2518` selected frame `106`
      dynamically (`105` was the first moving frame, `106` was the second
      consecutive observed moving frame), replacing the invalid hand-picked
      manifest frame `97`.
- [x] Stop the invalid fixed-prefix live evals that were still running after
      the prefix-start correction. The `f097` `long8` and `long7` steps on
      held allocation `127350` were interrupted/terminated at the process
      group level, not with `scancel`; `127350.extern` remains held. Their
      partial artifacts are diagnostic trash and must not be used as evidence.
- [x] Repair the single-sample live-loop Slurm wrapper
      `scripts/slurm/run_cosmos3_live_receding_loop_in_allocation.sh`, which
      still defaulted to `PREFIX_FRAME_INDEX=97`. It now defaults
      `PREFIX_START_MODE=target_motion_onset`, `PREFIX_FRAME_INDEX=-1`,
      `MIN_DYNAMIC_PREFIX_FRAME=8`, and
      `TARGET_MOTION_CONSECUTIVE_FRAMES=2`, and forwards those flags into
      `run_cosmos3_live_receding_loop.py`. The stray `long7_v3` fixed-prefix
      step on `127350.35` was stopped at the process-group level; its partial
      `iter_00_prefix_f097` artifacts are invalid diagnostics.
- [x] Run the first corrected dynamic-trigger live smoke on held compute
      allocation `127350`, Slurm step `127350.34`, without training:
      `live_receding_dynamic_trigger_smoke_iter2100_sample00_20260613_151213`.
      It used sample `0` from `iter_000002100`, but did not use the manifest
      prefix frame `97` as the start. `--prefix-start-mode target_motion_onset`
      detected target motion causally: first moving frame `105`, trigger frame
      `106` after two consecutive moving observations. The single receding
      iteration executed one 8-step Cosmos robot-action chunk and replayed
      source target frames `107..114`. Final live success was `false`, with
      peg-head-at-hole about `[-0.1743, 0.0493, -0.0026]`. The
      static-DP success-manifold gate correctly blocked DP because the real
      live state was not continuable (`rel_x_min=false`, `rel_y_abs=false`,
      and target speed above threshold). Direct review of
      `live_observed_rollout_16f_sheet.png` shows the target moving while the
      robot/peg remain outside the hole. This validates the dynamic-trigger
      code path and remains controller-negative for the current checkpoint.
- [x] Tighten the corrected smoke to include a true pre-trigger live DP phase
      instead of restoring the source prefix at the trigger frame. The active
      live loop now defaults `--pretrigger-control-mode
      frozen_dp_until_target_motion`: it restores only source frame `0`, runs
      the frozen static DP in the live simulator, replays only the source
      target actor pose after each live DP step, records real RGB/state/action
      history, and triggers Cosmos when target motion becomes observable.
      A compute-node dry-run on `127350.36` completed at
      `live_pretrigger_dp_dynamic_trigger_dryrun_iter2100_sample00_20260613_152400`:
      frozen DP executed `106` pretrigger steps, target motion was first
      observed at frame `105`, trigger frame was `106`, and the observed
      prefix video had `107` frames. Direct review of
      `observed_prefix_16f_sheet.png` shows the DP pretrigger approach and
      the target moving near the trigger point.
- [x] Run the first full corrected pretrigger-DP dynamic-trigger Cosmos smoke
      on `127350.37`:
      `live_pretrigger_dp_dynamic_trigger_cosmos1_iter2100_sample00_20260613_152608`.
      It used `iter_000002100`, restored only source frame `0`, executed
      frozen DP live for `106` steps, triggered at frame `106`, invoked Cosmos
      once, executed one 8-step Cosmos chunk, and replayed target actor frames
      `107..114`. Final live success was `false`, with peg-head-at-hole about
      `[-0.1740, 0.0492, -0.0015]`. The static-DP gate correctly blocked
      handoff (`rel_x_min=false`, `rel_y_abs=false`, target speed above
      threshold). Direct review of `live_observed_rollout_16f_sheet.png`
      agrees with the metrics: after target motion, the robot/peg are still
      outside the moved hole. This is the first valid minimal smoke for the
      intended DP-pretrigger -> dynamic target trigger -> Cosmos chunk ->
      live gate interface, and it is controller-negative for the current
      checkpoint.
- [x] 2026-06-13 re-audit after the user suspected the closed-loop eval code:
      the remaining DP handoff branch was still too close to a blind takeover.
      Even in the repaired live-receding loop, a passing `C_pi` gate could run
      `dp_handoff_horizon` steps (`32`/`64` by wrapper default) and then stop
      the sample instead of returning to the observe-decide loop. This does
      not match the first-principles plan, where both Cosmos and DP execution
      must be short chunks with real re-observation. The loop now evaluates
      `C_pi` at the start of each controller iteration. If the real state is
      DP-continuable, it executes at most
      `min(action_exec_horizon, dp_handoff_horizon)` frozen-DP steps, records
      `controller_step_type=frozen_dp_short_chunk`, reobserves, and either
      stops only on real success/termination/end or continues to the next
      observe-decide iteration. If `C_pi` fails, it runs the Cosmos rebind
      chunk and records `controller_step_type=cosmos_rebind_short_chunk` plus
      the post-Cosmos gate for the next iteration. Local `py_compile`,
      `bash -n`, and `git diff --check` passed. No training, Cosmos inference,
      or simulator rollout was launched for this code audit.
- [x] Run a tiny compute-node smoke first. It passes only if length accounting,
      action de-normalization, live `env.step`, video recording, and final
      metrics all complete without using sidecar/oracle state.
      Current compute-node smoke has progressed past the gate on `iter1800`
      and `iter2100`: live `env.step`, short Cosmos action execution,
      repeated DP resume, video recording, and final simulator metrics all
      run. The result is still diagnostic and controller-negative because the
      interface is one precomputed Cosmos chunk plus long DP takeover, not a
      full receding world-model controller. The corrected dynamic-trigger
      smoke above is the current valid tiny-smoke evidence: it uses target
      motion onset for the initial prefix, executes one live Cosmos chunk,
      replays target motion causally, records final simulator metrics/video,
      and blocks DP through the live continuability gate.
- [ ] Only after the smoke passes, run the fixed scenario-diverse validation
      panel from the testing plan: static/none, pre-motion target forecast,
      observed target motion, move-stop, reverse, peg disturbance, and
      peg-drop/regrasp.
- [ ] Freeze further SFT as of the 2026-06-13 user correction. Do not treat
      `iter_000002700` or higher as the next target. The immediate target is
      the closed-loop implementation: repeat Cosmos after real re-observation,
      execute only short chunks, add a documented continuability gate before
      any DP handoff, and save live rollout video/contact sheets for every
      controller claim.

## Negative Cases To Preserve

- [ ] If the passed SFT gate never happens, keep this as an implementation
      plan only. Do not force a closed-loop run from a weak checkpoint.
- [ ] If a live closed-loop rollout loses the peg, visibly inserts into the
      wall, relies on target self-insertion, or disagrees with its metric, mark
      it negative and keep it out of positive controller evidence.
- [ ] If the same implementation blocker repeats after concrete log/artifact
      inspection and no safe aligned fix is clear, preserve held resources when
      possible and stop for user direction rather than trying random variants.
