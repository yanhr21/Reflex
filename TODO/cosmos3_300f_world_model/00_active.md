# Active TODO

## Current Status

- [x] 2026-06-14 added an objective-level closed-loop gate:
      `scripts/world_model/check_cosmos3_closed_loop_objective_gate.py`.
      Current output:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_objective_gate.json`
      and `.md`. The gate checks the user's concrete closed-loop requirements
      from current artifacts: full `300/301` videos, causal target-motion
      detector rather than manual prefix selection, explicit annotated
      controller timeline, no-motion DP-only behavior from the same detector,
      moving-target WM activity, pure-DP comparison, and hard-case
      action/rebind evidence. Current verdict:
      `implementation_contract_ok=true`, `method_effectiveness_ok=false`.
      Contract failures are empty. The gate now directly scans video files:
      `26/26` raw/annotated videos have `301` frames with duration
      `10.033333333333333` seconds at `30 fps`, so the old "less than five
      seconds" failure is not present in the current iter2700 artifacts.
      2026-06-14 hardening: moving-target samples now also must prove
      causal detector provenance with
      `prefix_selection.mode=target_motion_onset`, concrete detector/streak
      frames, and at least `8` WM-active frames; no-motion must prove the same
      detector never triggered. Current artifacts still pass this stricter
      implementation contract.
      2026-06-14 annotation hardening: the gate now also requires complete
      `301`-frame `controller_timeline` and `annotated_video_summary.timeline`
      records whose controller counts match the summary. Moving samples must
      show WM-active frames in the annotated timeline; no-motion samples must
      show zero annotated WM-active and zero target-motion-detected frames.
      It also rejects missing/invalid `controller_frame_counts`, count sums
      other than `301`, or mismatched annotated-summary counts. Video scanning
      now has an OpenCV fallback so a single Python decoder import/read issue
      does not convert a valid 301-frame artifact into a false contract
      failure; all decoder failures still fail the video contract.
      Method failures are
      `val_cosmos_underperforms_same_source_pure_dp:1/3<pure_dp:3/3` and
      `hard_case_not_broadly_reliable:1/6`. This means the short-video/hidden
      trigger implementation problem is now repaired in the current artifacts,
      but iter2700 is still not acceptable method evidence.
      2026-06-14 no-motion hardening: the gate now also requires the
      no-motion witness to record
      `pretrigger_control_mode=frozen_dp_until_target_motion`, so a
      source-restored/static-special branch cannot satisfy the DP-only
      behavior requirement. The regenerated gate still has
      `implementation_contract_ok=true` and `method_effectiveness_ok=false`.
      2026-06-14 hard-case usefulness hardening: the gate now records
      `min_hard_case_success_fraction=0.5` for the panel restricted to full
      pure-DP failures, while preserving the stricter
      `hard_case_not_broadly_reliable` all-success failure. Current iter2700
      explicitly fails both usefulness checks:
      `hard_case_success_fraction_below_minimum:1/6<min_fraction:0.5` and
      `hard_case_not_broadly_reliable:1/6`.
- [x] 2026-06-14 added wrapper-level protection for future live closed-loop
      launches. The live-receding panel and single-sample wrappers now refuse
      method runs that use manifest/manual prefixing, source-restored
      pretrigger control, explicit role overrides, or disabled Cosmos
      inference unless `ALLOW_LIVE_RECEDING_DIAGNOSTIC=true` is set. This is a
      launch-time guard in addition to the objective gate, and prevents the old
      hidden-onset/no-Cosmos diagnostic modes from being accidentally presented
      as corrected closed-loop evidence.
- [x] 2026-06-14 added
      `scripts/world_model/selftest_cosmos3_closed_loop_objective_gate.py` so
      the hardened gate can be checked locally without Slurm or video decoding.
      It verifies valid causal moving/static summaries pass, while manual
      prefixes, missing detector provenance, token one-frame WM use, and static
      WM activation fail. It now also checks the structured annotation-timeline
      contract behind the overlay videos. It also runs a subprocess end-to-end
      gate check proving `panel_full_episode_contract_ok=false` rejects val
      Cosmos, hard Cosmos, and pure-DP panel evidence.
- [x] 2026-06-14 blocked accidental use of the old one-shot closed-loop
      wrappers. `run_cosmos3_closed_loop_panel_in_allocation.sh` and
      `run_cosmos3_receding_closed_loop_in_allocation.sh` now require
      `ALLOW_OLD_ONESHOT_CLOSED_LOOP_DIAGNOSTIC=true`; otherwise they exit
      before creating outputs and point to the corrected full-300 live-receding
      wrappers.
- [x] 2026-06-14 added
      `scripts/world_model/selftest_cosmos3_closed_loop_wrapper_guards.sh` as
      a single local check for the wrapper guards. It verifies non-method
      live-receding modes exit `42`, old one-shot wrappers exit `43`, and no
      output directory is created before refusal.
- [x] 2026-06-14 added a direct video-length audit tool:
      `scripts/world_model/audit_video_length_contract.py`. It compares the
      user-flagged old short run
      `live_receding_panel10_corrected_iter2100_20260613_161006` against the
      current iter2700 evidence roots. Output:
      `docs/world_model_task_rebinding/2026-06-14_iter2100_vs_iter2700_video_length_audit.json`
      and `.md`. The old iter2100 path has only `2` final rollout videos and
      both fail the `301`-frame contract: `131` frames / `4.3667s` and
      `119` frames / `3.9667s`. The current iter2700 val, hard2, pure-DP, and
      no-motion roots have `26/26` final raw/annotated videos matching
      `301` frames and `10.0333s`. The old path is historical negative
      evidence only and must not be used as current closed-loop method
      evidence.
- [x] 2026-06-14 hardened future closed-loop artifacts against the same
      incomplete-video failure. `run_cosmos3_live_receding_loop.py` now
      decodes the just-written raw and annotated mp4 files and records
      `final_observed_video_inspection`,
      `final_observed_annotated_video_inspection`, and
      `video_file_contract_ok` in each sample summary. The pure-DP baseline
      writes the same fields, and both panel wrappers propagate them into the
      panel summaries. This is future-run enforcement: current evidence is
      still proven by the objective gate and direct video-length audit above.
      2026-06-14 hardening: the video inspection helper was moved to
      `scripts/world_model/video_contract_utils.py`, now uses OpenCV fallback
      after imageio decode/import failures, and `video_file_contract_ok`
      requires the exact expected number of inspections. If annotated video is
      enabled, both raw and annotated videos must exist, decode without error,
      and contain `301` frames; a missing annotated inspection can no longer
      be skipped by filtering out `None`. The objective gate and video-length
      audit now reuse the same helper, so closed-loop summaries, current
      evidence gates, and old-vs-current video audits all apply the same mp4
      decoding contract.
      2026-06-14 panel hardening: the live-receding panel and pure-DP panel
      now compute `panel_full_episode_contract_ok` and
      `sample_contract_failures`. A future panel returns failure if any sample
      lacks a summary, misses the `300/301` contract, has a bad video contract,
      or has controller-frame counts that do not sum to `301`. Pure-DP panels
      also require `PURE_DP=300` and `WM_ACTIVE=0`. The objective gate treats
      `panel_full_episode_contract_ok=false` as a contract failure when that
      field is present. The shared logic lives in
      `scripts/world_model/panel_contract_utils.py` and is covered by
      `scripts/world_model/selftest_panel_contract_utils.py`.
- [x] 2026-06-14 hard-screen-2 comparison completed to test whether Cosmos
      helps specifically where full-episode pure DP fails. A 15-sample
      high-motion manifest was selected at
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen2_20260614/hard_case_eval_manifest.json`.
      Pure DP completed all `15` full `300/301` rollouts and succeeded on
      `9/15`; the six pure-DP failures were indices `4,9,10,11,12,13`.
      Cosmos closed-loop was run only on those six failures under
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen2_20260614/cosmos_iter2700_puredp_fail_4_9_10_11_12_13`.
      Cosmos succeeded on `1/6`: `hole_late_fast_shift` index `4`, with
      full `300/301`, detector trigger `f86`, frame counts
      `{INIT_OBS:1, DP_SCAN_TARGET:86, WM_ACTIVE:56, DP_HANDOFF:158}`,
      final success `true`, and peg-head-at-hole
      `[-0.0071999, 0.0015902, 0.0021036]`. The same-source pure-DP baseline
      for index `4` failed with peg-head-at-hole
      `[-0.0958784, 0.0228078, -0.0324848]`. The Cosmos panel sheet plus
      the index-4 Cosmos and pure-DP annotated sheets/final frames were opened
      directly. This gives a second positive hard-case comparison, but the
      broader `1/6` result means the current iter2700 closed-loop is not a
      broadly reliable solution yet.
- [x] 2026-06-14 added a read-only hard-screen-2 action/rebind diagnostic:
      `scripts/world_model/analyze_cosmos3_hard_case_action_rebind.py`.
      Output:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_hard_screen2_action_rebind_analysis.json`.
      It compares executed Cosmos action chunks against same-H5 source-teacher
      action rows for failure localization only; teacher actions are not used
      for control. On the six matched pure-DP failures, pure DP success is
      `0/6` and Cosmos success is `1/6`. The five Cosmos failures are dominated
      by `rel_y_abs`, `rel_z_abs`, occasional `grasped`, and action
      direction/scale mismatch, so the current failure mode is raw action
      rebind reliability, not incomplete 300-frame rollout or missing Cosmos
      activation. Do not treat this checkpoint as broadly solved; the aligned
      repair remains clean-role/dense-receding condition export plus approved
      overfit/full SFT, or a learned short-chunk executor if direct raw Cosmos
      actions remain unstable.
- [x] 2026-06-14 added a read-only failure-mode reducer:
      `scripts/world_model/summarize_cosmos3_closed_loop_failure_modes.py`.
      Output:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_failure_modes.json`
      and `.md`. It combines the objective gate and hard action/rebind
      analysis into one method-boundary summary. Current verdict remains
      `implementation_contract_ok=true` and `method_effectiveness_ok=false`.
      The hard failures are dominated by real-state handoff blocks on
      `rel_y_abs=203` and `rel_z_abs=195`, with additional raw action
      direction/scale flags versus teacher chunks. This explicitly rules out
      the old short-video/manual-trigger/missing-Cosmos-annotation issue as
      the primary current blocker and records the active blocker as direct raw
      Cosmos action rebinding plus DP-continuability reliability.
- [x] 2026-06-14 added a requirement-level audit:
      `scripts/world_model/audit_cosmos3_closed_loop_requirements.py`.
      Output:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_requirement_audit.json`
      and `.md`. It maps the user's original closed-loop requirements to
      current evidence instead of treating one gate as the whole objective.
      Current status is `current_goal_achieved=false` with
      `{'passed': 6, 'partial': 1, 'failed': 1}`. Passed items are old-short
      iter2100 rejection, current 300/301 videos, causal target-motion
      detection, nonzero Cosmos takeover, explicit takeover annotation, and
      no-motion DP-only behavior from the same detector. The partial item is
      DP handoff: available and sometimes successful, but not reliable. The
      failed item is method effectiveness against pure DP. This audit is the
      current requirement-level proof that the implementation boundary is
      repaired but the full objective is not complete.
- [x] 2026-06-14 added a live-query/training-coverage audit:
      `scripts/world_model/audit_cosmos3_live_query_training_coverage.py`.
      Output:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_live_query_training_coverage_audit.json`
      and `.md`. It compares actual val/hard2 live Cosmos query states against
      the old SFT condition rows. Current result: `173` live Cosmos queries,
      `1193/2899` training role/mode mismatches, and `74/173` live queries
      without a local role/mode-consistent training neighbor under the chosen
      prefix/geometry tolerances. This supports the current repair direction:
      do not keep broad-evaluating iter2700; fix the clean-role/dense-receding
      condition distribution first. The audit is covered by
      `scripts/world_model/selftest_cosmos3_live_query_training_coverage.py`.
      The full v7_733 clean/dense preflight wrapper now defaults to running
      this coverage audit and passes it into
      `summarize_cosmos3_clean_dense_preflight.py`; `ready_for_overfit` fails
      if the new condition root does not cover the recorded live queries under
      the configured role/mode and geometry thresholds. The overfit2 preflight
      keeps the audit disabled by default because it is only a two-sample
      chain sanity check. The full preflight wrapper now refuses real
      execution if the coverage audit is disabled without the explicit
      diagnostic override
      `ALLOW_SKIP_LIVE_QUERY_COVERAGE_DIAGNOSTIC=true`, and it checks that the
      configured panel summary paths exist before handing off to the base
      preflight wrapper. If that diagnostic override is used, the base summary
      receives `diagnostic-not-ready` metadata and must report
      `ready_for_overfit=false`. The overfit SFT entry also validates the
      summary with
      `scripts/world_model/check_cosmos3_clean_dense_preflight_summary_ready.py`,
      so `failed_checks` or diagnostic metadata cannot be bypassed by a
      hand-edited `ready_for_overfit=true` flag.
- [x] 2026-06-14 old sampled-role v7_733 SFT wrappers are no longer active
      repair entries. The full sampled-role fix1 wrapper exits `66` by
      default when `RUN_SFT=true`, and the old sampled-role overfit2 wrapper
      exits `67`; both require explicit legacy diagnostic overrides and have
      login-safe `DRY_RUN_CONFIG_ONLY=true` output. Active SFT repair must go
      through clean-role/dense-receding preflight and a matching
      ready-for-overfit summary. Guard behavior is covered by
      `scripts/world_model/selftest_cosmos3_legacy_sft_wrapper_guards.sh`.
- [x] 2026-06-14 recorded the user-requested val/hard pure-DP comparison and
      the follow-up step-level handoff probe:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_val_hard_puredp_comparison_and_stepgate_probe.md`.
      The val comparison is negative (`Cosmos=1/3`, same-source pure
      `DP=3/3`). The harder comparison is mixed but not sufficient
      (`Cosmos=1/6` on the six hard-screen-2 samples where full pure DP
      failed). A diagnostic `--cosmos-step-handoff-gate` probe on hard-screen-2
      index `12` was interrupted only by the Slurm allocation time limit, not
      by manual cancellation; no full panel/sample summary exists, so it is
      not complete method evidence. Its partial summary reached frame `188`
      with `success=false`; it did show one real-state C_pi pass followed by an
      8-step DP handoff that drifted laterally and lost continuability again.
      This reinforces the current diagnosis: the remaining blocker is unstable
      action/rebind plus DP-continuability, not missing hard-case comparison.
- [x] 2026-06-14 hard comparison completed on hard-screen samples where full
      pure DP failed: continuous-insert index `1` and move-stop index `3` from
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen_20260614/hard_case_eval_manifest.json`.
      Cosmos run root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen_20260614/cosmos_iter2700_hard_dp_fail_1_3`.
      Continuous-insert stayed failed under Cosmos (`300/301`, trigger `f134`,
      `WM_ACTIVE=166`, final success `false`). Move-stop is the positive
      comparison sample: pure DP failed, while Cosmos ran `WM_ACTIVE=48` after
      trigger `f84`, then real-state-gated `DP_HANDOFF=168`, ending with
      final success `true` and peg-head-at-hole
      `[-0.0096748, -0.0019456, -0.0029760]`. Cosmos and pure-DP videos for
      move-stop were opened directly and visually match the metrics.
- [x] 2026-06-14 same-source full-episode pure-DP baseline was added and run
      for the iter2700 val dynamic panel. Code:
      `scripts/world_model/run_dp_full_episode_baseline.py` and
      `scripts/world_model/run_dp_full_episode_baseline_panel.py`. Output:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/dp_full_episode_baseline_iter2700_panel3_dynamic_20260614_alloc127559`.
      Pure DP succeeded on all three same-source panel samples (`3/3`), while
      Cosmos closed-loop succeeded on only `1/3`. This is negative comparison
      evidence for the current iter2700 closed-loop interface; it does not
      validate Cosmos as useful on this val panel.
- [x] 2026-06-13/14 the scenario-panel `sample_03_hole_late_fast_shift`
      completed and was inspected. It reached `300` actions / `301` frames,
      detector trigger `f132`, and
      `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:132,
      WM_ACTIVE:32, DP_HANDOFF:136}`. Final success is `false` with
      peg-head-at-hole
      `[-0.0649778, 0.0001627, -0.0006470]`. The annotated sheet and final
      frame show WM then DP handoff through `frame 300/300`, but the peg is
      still outside the hole.
- [x] 2026-06-13 iter2700 full-300 unified-detector closed-loop eval completed
      after the user clarified that no sample may be predeclared as
      `DP-only` from a static/no-motion label. The controller boundary is one
      causal target-motion detector; if it never fires, frozen DP continues by
      that same rule. The dynamic sample `hole_late_move_stop` did trigger at
      `f106` and ran to the full `300` action / `301` frame contract with
      `full_episode_length_ok=true`. Final success is `false`; final
      peg-head-at-hole is `[-0.1056687, -0.0143125, -0.0550163]`.
      Annotated/raw videos and opened review sheets are under
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_sample00_20260613_alloc127559`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-13_iter2700_full300_unified_closed_loop_eval.md`.
- [x] 2026-06-13 the same `iter2700` full-300 closed-loop path has now also
      produced an inspected complete scenario-panel dynamic sample:
      `sample_01_hole_late_constant` under
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_panel3_dynamic_20260613_alloc127559`.
      It reached `final_prefix_frame_index=300` and
      `final_observed_frames=301`, with detector trigger `f94` and
      `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:94,
      WM_ACTIVE:206}`. The annotated sheet and final frame were opened
      directly and show explicit `controller=WM_ACTIVE`,
      `target_motion_detected=True`, `trigger=94` through `frame 300/300`.
      Final success is still `false`, with peg-head-at-hole
      `[-0.1030322, -0.0450076, -0.0782132]`, so this is full-length
      controller-negative evidence, not task success.
- [x] 2026-06-13 the scenario-panel `sample_02_hole_late_reverse` produced
      the first inspected positive full-length closed-loop result for the
      current `iter2700` interface. It reached `300` actions / `301` frames,
      detector trigger `f104`, and
      `controller_frame_counts={INIT_OBS:1, DP_SCAN_TARGET:104,
      WM_ACTIVE:40, DP_HANDOFF:156}`. The first five post-trigger chunks were
      Cosmos rebind chunks; the real-state continuability gate became true at
      prefix `f144`, and the loop then used short reobserved frozen-DP handoff
      chunks through the full horizon. Final success is `true` with
      peg-head-at-hole `[0.0061689, 0.0014199, -0.0005607]`. Raw and
      annotated videos are readable/nonblank `301`-frame videos; opened
      review sheets show `WM_ACTIVE` after target-motion detection and
      `DP_HANDOFF` through `frame 300/300`, with the peg visibly inserted in
      the moved hole. This supports the intended Cosmos-rebind plus
      DP-continuable handoff design on this sample, while the panel is still
      running on `sample_03_hole_late_fast_shift`.
- [x] The same iter2700 full-300 closed-loop code path was also checked on a
      no-motion `none` sample:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_full300_unified_iter2700_static_none_fix1_20260613_alloc127559`.
      It ran to `300` actions / `301` frames with
      `triggered=false`, `wm_active_frame_count=0`, no Cosmos inference
      directory, and annotated video showing `DP_SCAN_TARGET` through
      `frame 300/300`. This verifies that no-motion uses the same detector
      rule rather than a separate static branch or label-driven bypass. The
      DP final task success was false on this seed, so static preservation
      remains a separate performance issue.
- [x] 2026-06-13 closed-loop failure analysis is now the active boundary.
      The corrected long-horizon live-receding eval failed on sample `00`, and
      a follow-up code/data audit did not find a simple action-row offset or
      remaining live-history conditioning bug. The current failure is best
      classified as model/action rebind capability under distribution drift:
      role captions are not cleanly aligned with actual physical modes,
      training prefixes are sparse rather than dense receding states, and the
      live loop asks the model to recover from its own off-source misses after
      target motion. Later checkpoint evidence through `iter_000002700` is
      still controller-blocked by visual review, so continuing smoke training
      or running broader panels is not justified from the current evidence.
      See
      `docs/world_model_task_rebinding/2026-06-13_corrected_live_receding_panel_failure.md`.
- [x] Added a read-only training-distribution audit for the failed receding
      interface:
      `scripts/world_model/audit_cosmos3_receding_training_distribution.py`.
      The current v7_733 condition root has `1193/2899` role/mode mismatches,
      no action condition-mask errors, and `1287` late-rebind proxy rows.
      The repair plan is now
      `PLAN/cosmos3_300f_world_model/07_post_closed_loop_failure_repair.md`
      with execution TODO in
      `TODO/cosmos3_300f_world_model/09_post_closed_loop_failure_repair.md`.
- [x] Implemented the clean-role/dense-receding condition-export repair
      entry point without launching SFT. The exporter now accepts
      `--prefix-role-source physical_mode` and
      `--dense-receding-prefix-stride N`, while default sampled-role behavior
      remains unchanged for old-root reproduction. A `/tmp` two-episode probe
      produced `23` full-episode rows with `0` role/mode mismatches.
      Re-exporting the full v7_733 condition root and any training remain
      pending user approval.
- [x] Hardened the clean/dense preflight path without launching SFT. The
      receding-distribution audit now owns hard gates:
      physical-mode exports fail on nonzero role/mode mismatch, condition-mask
      errors can be hard-failed, and minimum late-rebind coverage can be
      required. The Slurm preflight wrapper calls this audit directly and no
      longer relies on inline Python shell checks. `/tmp` validation passed:
      clean physical-mode probe `strict_ok=true`, mismatch `0`, late-rebind
      `18`; old sampled root remains report-only with mismatch `1193`.
- [x] Added a clean/dense preflight summarizer:
      `scripts/world_model/summarize_cosmos3_clean_dense_preflight.py`.
      Approval-time preflight will now write `clean_dense_preflight_summary`
      artifacts and must show `ready_for_overfit=true` before any overfit SFT.
- [x] Added `MAX_RECORDS` passthrough and a clean/dense overfit2 preflight
      wrapper. Dry-run verified it defaults to `EXPECTED_SOURCE_EPISODES=2`,
      `MAX_RECORDS=2`, and `RUN_SFT=false`, so future overfit condition export
      can be audited separately before any training.
- [x] Hardened the clean/dense overfit2 SFT wrapper. It now has a
      login-safe `DRY_RUN_CONFIG_ONLY=true` mode that reports missing
      approval/preflight requirements without launching work, and real
      execution refuses unless `ALLOW_CLEAN_DENSE_OVERFIT_SFT=true`, an
      explicit `CONDITION_ROOT`, and a matching
      `CLEAN_DENSE_PREFLIGHT_SUMMARY` with `ready_for_overfit=true` are
      provided.
- [x] Added a current SFT/closed-loop monitoring note:
      `docs/world_model_task_rebinding/2026-06-13_cosmos3_sft_closed_loop_current_status.md`.
      It records that `127350` is extern-only, no training/eval process is
      active, checkpoints exist through `iter_000002700`, the latest log line
      is around iteration `2735`, `sft_completed` is stale relative to the
      log, `iter2700` generated eval is visually gated off, and corrected
      live closed-loop remains negative.
- [x] Added a reusable read-only status summarizer:
      `scripts/world_model/summarize_cosmos3_sft_closed_loop_status.py`.
      It reads Slurm steps, matching training/eval processes, SFT logs,
      checkpoints, generated-eval gates, and live-receding summaries without
      running Cosmos/ManiSkill. Current output:
      `docs/world_model_task_rebinding/2026-06-13_cosmos3_sft_closed_loop_status_auto.json`
      and `.md`; it reports active process count `0`, extern-only Slurm
      allocation, latest checkpoint `iter_000002700`, latest visible train
      iteration `2735`, stale `sft_completed`, latest eval
      `closed_loop_allowed=false`, visual status `fail`, and no successful
      recent live run.
- [x] Added a short monitor wrapper:
      `scripts/world_model/monitor_current_cosmos3_sft_closed_loop_status.sh`.
      It reruns the read-only status summarizer with the current SFT root and
      Slurm job `127350`, updating the auto JSON/MD status files. The latest
      wrapper run still reports active process count `0`, extern-only
      allocation, latest checkpoint `iter_000002700`,
      `closed_loop_allowed=false`, visual status `fail`, and no successful
      recent live run.
- [x] Added a read-only next-action gate:
      `scripts/world_model/check_cosmos3_next_action_gate.py`. Current status
      JSON rejects `resume_current_condition_sft` and
      `launch_broad_panel_current_checkpoint` with
      `latest_generated_eval_controller_blocked`; it rejects
      `clean_dense_preflight_after_user_approval` unless the user approval
      flag is explicit; and it still rejects
      `clean_dense_overfit_sft_after_user_approval` until a matching
      clean-dense preflight summary records `ready_for_overfit=true`.
      Current approved-preflight-only verdict is recorded at
      `docs/world_model_task_rebinding/2026-06-13_cosmos3_next_action_gate_clean_dense_preflight.json`.
      2026-06-14 hardening: the gate now accepts
      `--requirement-audit-json` and refuses old-checkpoint resume/broad-panel
      actions when the requirement audit has `current_goal_achieved=false`.
      Current outputs:
      `docs/world_model_task_rebinding/2026-06-14_cosmos3_next_action_gate_launch_broad_panel_requirement_audit.json`
      and
      `docs/world_model_task_rebinding/2026-06-14_cosmos3_next_action_gate_resume_current_requirement_audit.json`.
      Both are denied with `closed_loop_requirement_audit_not_achieved`, and
      list `dp_handoff_available_but_not_proven_reliable` plus
      `method_effectiveness_against_pure_dp` as the remaining failed/partial
      requirement IDs.
      2026-06-14 repair-path hardening: the gate now also accepts
      `--clean-dense-preflight-summary-json`. The
      `clean_dense_overfit_sft_after_user_approval` action is allowed only
      when the user approval flag is set, no active process is detected, and
      that summary records `ready_for_overfit=true`. It is not blocked merely
      because the old iter2700 requirement audit is incomplete, since this is
      a repair action rather than a claim that the old checkpoint works.
      Current output without a ready summary is
      `docs/world_model_task_rebinding/2026-06-14_cosmos3_next_action_gate_clean_dense_overfit_requires_ready_summary.json`
      and is correctly denied with
      `requires_clean_dense_preflight_summary_ready_for_overfit`.
      The guarded clean/dense overfit SFT Slurm wrapper now calls this
      next-action gate before its inline summary/condition-root check and
      before delegating to the base training wrapper. Its gate output defaults
      to `next_action_gate_clean_dense_overfit_sft.json` next to the supplied
      preflight summary.
- [x] Added a status-refreshing next-action gate wrapper:
      `scripts/world_model/check_current_cosmos3_next_action_gate.sh`. It
      first reruns the read-only monitor, then checks the requested action so
      decisions do not rely on stale status JSON. While validating this path,
      the process monitor was repaired to use `ps` filtering instead of
      `pgrep -af`, avoiding false active-process counts caused by concurrent
      read-only gate checks. Current gate validation: old SFT resume and
      current-checkpoint broad panel both exit `2`; user-approved clean/dense
      preflight is allowed; clean/dense overfit SFT still exits `2` until
      `ready_for_overfit=true` exists.
      The wrapper now passes
      `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_requirement_audit.json`
      to the gate automatically when that file exists, so status refreshes and
      action checks include the current requirement-level audit.
      If `CLEAN_DENSE_PREFLIGHT_SUMMARY` points at an existing summary file,
      the wrapper also passes it to the gate for overfit-readiness checks.
- [x] Added and ran a lightweight next-action gate self-test:
      `scripts/world_model/selftest_cosmos3_next_action_gate.py`. It verifies
      that old-condition SFT resume and current-checkpoint broad panel are
      rejected, clean/dense preflight requires explicit user approval, approved
      preflight is the only allowed action in the idle blocked state, and
      clean/dense overfit SFT remains rejected until a ready-for-overfit
      preflight summary exists. It now also verifies that even if status flags
      claim old-checkpoint resume/panel are safe, an incomplete requirement
      audit still blocks those actions while leaving approved clean/dense
      preflight possible. It now also verifies clean/dense overfit remains
      denied without a ready preflight summary or without user approval, and
      becomes allowed only when `ready_for_overfit=true` is supplied. Current
      run printed
      `cosmos3_next_action_gate_selftest=passed`.
- [x] Enforced that user-approved clean/dense preflight is explicit at the
      Slurm wrapper level. The full v7_733 and two-source overfit2 preflight
      wrappers now refuse real execution unless
      `ALLOW_CLEAN_DENSE_PREFLIGHT=true` is set. `DRY_RUN_CONFIG_ONLY=true`
      remains login-safe; with the approval variable on a login node, the
      read-only next-action gate can pass, but the base wrapper still refuses
      before export because there is no compute-node `srun` step.
- [x] Added a read-only command printer:
      `scripts/world_model/print_cosmos3_clean_dense_preflight_commands.sh`.
      It prints the exact approved full and overfit2 clean/dense preflight
      commands with `ALLOW_CLEAN_DENSE_PREFLIGHT=true`, while explicitly
      noting they must be run only after user approval from inside a
      compute-node `srun` step and default to `RUN_SFT=false`. The helper
      does not call Slurm, export data, train, render, or run eval.
- [x] 2026-06-13 user-directed closed-loop eval was advanced without further
      smoke training. The corrected prompt-fixed long-horizon live-receding
      run
      `live_receding_promptfix_sample0_longhorizon_iter2100_20260613`
      completed on allocation `127350` with checkpoint `iter_000002100`,
      sample `00` (`hole_late_move_stop`), target-motion-onset prefix `f106`,
      target-only source replay, `12` receding Cosmos chunks, and real-state
      `C_pi` DP handoff gating. It ran to frame `202` and failed:
      final peg-head-at-hole `[-0.1423, 0.0004, 0.0101]`,
      final success `false`, and DP handoff `0` steps. The source teacher for
      the same H5 first inserts at `f166` and is inserted at `f202`, so this
      is a real corrected closed-loop failure rather than an eval horizon
      artifact. Visual review of the panel sheet plus the dense 18-frame
      rollout sheet confirms the peg remains outside the moved hole. Current
      failure classification: executable Cosmos action/rebind capability,
      especially late post-motion insertion correction, not the old
      source-H5 conditioning bug, not a prompt-only mismatch, and not blind
      DP takeover. Do not resume smoke training from this evidence.
- [x] 2026-06-13 closed-loop audit after user stop request: both active SFT
      foreground training commands were interrupted inside their held tmux/
      Slurm allocations, not cancelled with `scancel`. Jobs `127281`,
      `127286`, and `127350` remain held as extern allocations only, and no
      `torchrun`/SFT `srun` process is active. Do not continue training until
      the closed-loop/DP handoff code path is repaired.
- [x] Closed-loop eval code audit found the current
      `run_cosmos3_receding_closed_loop.py` path is not the method described
      in `IDEA.md` and `PLAN/cosmos3_300f_world_model/03_testing_and_metrics.md`.
      It restores one source prefix state, executes one precomputed Cosmos
      robot-action chunk, then optionally lets frozen DP run for a long
      takeover horizon. It does not rerun Cosmos after live re-observation,
      does not replay the source H5 external target motion after prefix reset,
      does not use a continuability guard before long DP resume, and does not
      provide a predicted future task-state/reconstruction interface for DP
      resume. Treat existing `8+96` DP panels as diagnostic only.
- [x] First repair toward the intended closed-loop interface: added a
      live-prefix Cosmos policy input builder and compute-node inference
      wrapper. This creates one causal re-observation sample from observed
      prefix RGB plus history action/state rows so the future loop can rerun
      Cosmos after each short executed chunk. A local builder-only probe passed
      on an iter2100 sample; no training, Cosmos inference, or simulator
      rollout was launched for this probe.
- [x] Added the next closed-loop scaffold:
      `run_cosmos3_live_receding_loop.py`. It is compute-node-only and, in
      dry-run mode, prepares the per-reobservation prefix video plus raw WAM
      history without running Cosmos. The required causal video rule is now
      explicit: live-prefix videos must contain only frames
      `[0, prefix_frame_index]`; full future reference videos are refused by
      default. Syntax checks pass, but no live controller evidence exists yet.
- [x] 2026-06-13 eval-physics repair after rereading the plan: fix3 moving-hole
      source H5s encode target motion by manually setting
      `env_states/actors/box_with_hole` after each action step. The old live
      eval restored `env_states[prefix]` once and then stepped robot actions
      without continuing that external target motion, so pre-motion live
      rollouts were testing the wrong physical world. The live receding loop
      now defaults to `--external-target-mode source_env_state`, replaying only
      the source target actor pose at `frame+1` after each live robot action
      while leaving robot/peg state live. It also maintains a fresh two-frame
      state-observation history after the target replay for any later DP
      handoff, instead of using the stale observation returned before target
      replay.
- [x] First post-repair live receding smoke completed without any new training:
      `live_receding_statsprofile32_gate_iter2100_sample00_20260613_1518`.
      It used `iter_000002100`, source sample
      `hole_late_move_stop_seed3280649_idx2518`, prefixes `97`, `105`, and
      `113`, and `--external-target-mode source_env_state`. The run replayed
      source target actor states through frame `121`, so the target actually
      moved during live eval. Final simulator success remained `false` with
      peg-head-at-hole about `[-0.2527, 0.0825, -0.0031]`; the static-DP
      continuability gate correctly blocked DP on all three iterations. The
      contact sheet `live_observed_rollout_contact_sheet.png` was opened
      directly and shows the target moving away while the robot/peg do not
      rebind to the new hole. This is controller-negative evidence for the
      current checkpoint/action loop, not a reason to train longer.
- [x] Live receding dry-run smoke completed inside held compute allocation
      `127350` without Cosmos inference:
      `live_receding_dryrun_iter2100_sample00_20260613_131044`. It restored
      sample-00 prefix frame `97`, wrote a `98`-frame observed-prefix video
      and raw `300x32` WAM history, then stopped because Cosmos inference was
      disabled. This verifies the reobservation input path, not controller
      success.
- [x] Patched live-prefix inference smoke completed on Slurm allocation
      `127350` using `iter_000002100`, sample `0`, prefix frame `97`, and an
      `8`-step action chunk:
      `live_receding_oneiter_cosmos_iter2100_sample00_with_live_video_20260613_1329`.
      The run rebuilt a causal prefix from rendered env states, invoked
      Cosmos once, executed only rows `97:105` in the live simulator, and
      saved `live_observed_rollout.mp4`. Final simulator success is `false`;
      peg-head-to-hole changed from about `[-0.194, 0.057, -0.030]` before
      the chunk to `[-0.140, 0.031, -0.016]` after the chunk. The contact
      sheet was opened directly. This is useful interface evidence that
      live-prefix inference and short action execution work, but it is not
      closed-loop method success and it does not justify more SFT training.
- [x] Live receding implementation progressed beyond one-step smoke. The
      `iter_000002100` sample-0 five-iteration run
      `live_receding_fiveiter_gated_dp_iter2100_sample00_20260613_1353`
      rebuilt a fresh causal prefix after each real 8-step execution and
      reran Cosmos at prefixes `97`, `105`, `113`, `121`, and `129`. It
      executed `40` real Cosmos robot actions, saved `live_observed_rollout.mp4`,
      and ended `success=false` with peg-head-to-hole about
      `[-0.1085, 0.0067, -0.0012]`. The conservative live `C_pi` gate blocked
      DP every time, correctly preventing long static-DP handoff while the
      insertion-axis distance was still too large.
- [x] A relaxed-gate DP handoff diagnostic was run and must not be used as
      method evidence:
      `live_receding_dpstatic32_gate_iter2100_sample00_20260613_1410`.
      It manually loosened `continuability_min_rel_x` and triggered `32`
      frozen-DP steps after the third Cosmos chunk; final live state improved
      to about `[-0.0197, 0.0027, -0.0003]` but still had
      `success=false`. The reviewed contact sheet shows near-hole alignment,
      not completed insertion. This is evidence against premature/relaxed DP
      handoff, not a success.
- [x] The relaxed gate has now been converted into an explicit static-DP
      success-manifold profile option. `run_cosmos3_live_receding_loop.py`
      accepts `--continuability-stats-json` and derives the handoff bounds
      from `dp_static_continuability_stats.json` instead of unrecorded
      threshold edits. A loader check on the current stats file with horizon
      `32` used `31,218` static-DP success-predecessor records and produced
      `min_rel_x=-0.1342566`, `max_abs_y=0.0098417`, and
      `max_abs_z=0.0038843` while preserving `max_rel_x=0.04` as the safety
      cap. Future closed-loop smoke/panels should use this recorded profile
      path when testing DP handoff from the live receding loop.
- [x] The first stats-profile live receding smoke completed on compute
      allocation `127350` without any SFT continuation:
      `live_receding_statsprofile32_gate_iter2100_sample00_20260613_1518`.
      It ran three true reobserve/rerun Cosmos iterations from prefixes
      `97`, `105`, and `113` with `--prefix-role auto` and the 32-step
      static-DP profile. The profile gate blocked DP at every iteration, and
      final live success was `false`; peg-head-to-hole ended around
      `[-0.2527, 0.0825, -0.0031]`. The opened contact sheet shows the target
      moving while the real peg/hand does not keep up. This is negative
      controller evidence, but it verifies the corrected closed-loop eval path
      is now testing the intended mechanism instead of the old one-shot
      Cosmos plus blind long-DP takeover.
- [x] Added a corrected live-receding panel wrapper so future eval cannot
      silently fall back to the old one-shot DP96 diagnostic path:
      `scripts/world_model/run_cosmos3_live_receding_panel.py` and
      `scripts/slurm/run_cosmos3_live_receding_panel_in_allocation.sh`. The
      wrapper reads `eval_input_manifest.json`, resolves each source H5 from
      the frozen v7 canonical source root, defaults `prefix-role=auto`, keeps
      `external-target-mode=source_env_state`, and runs
      `run_cosmos3_live_receding_loop.py` per sample. It records
      `method_evidence_allowed=false` and emits a contact sheet for direct
      review. Local `py_compile`, `bash -n`, and `git diff --check` passed;
      no login-node rollout/Cosmos inference/training was run.
- [x] 2026-06-13 prefix-start correction: fixed manifest prefix frames are not
      the planned closed-loop interface. The live loop now has
      `--prefix-start-mode target_motion_onset`, and the live-receding panel
      defaults to that mode. It selects the first Cosmos prefix from observed
      target-object motion using the current/past `box_with_hole` pose only;
      the old `manifest`/manual prefix mode remains explicit diagnostic-only
      behavior. A source-H5 check selected frame `106` for
      `hole_late_move_stop_seed3280649_idx2518`, instead of the manifest's
      hand-picked frame `97`. Any fixed-`f097` `long7/long8` live eval
      artifacts from today are invalid diagnostic leftovers and should not be
      interpreted.
- [x] First corrected dynamic-trigger smoke completed on allocation `127350`
      with no training:
      `live_receding_dynamic_trigger_smoke_iter2100_sample00_20260613_151213`.
      The panel wrapper ran with `PREFIX_START_MODE=target_motion_onset`,
      selected dynamic prefix frame `106` instead of manifest frame `97`,
      executed one 8-step live Cosmos action chunk, and replayed only target
      source frames `107..114`. Final live success was `false`; DP handoff was
      blocked by the static-DP continuability profile, with final
      peg-head-at-hole about `[-0.1743, 0.0493, -0.0026]`. Direct visual
      review of the generated 16-frame sheet shows no insertion and no valid
      DP-continuable pose. This is a corrected eval-path smoke, not method
      success and not a reason to resume SFT.
- [x] Also repaired the older single-sample live-loop Slurm wrapper
      `scripts/slurm/run_cosmos3_live_receding_loop_in_allocation.sh`, which
      was still capable of launching fixed `PREFIX_FRAME_INDEX=97` runs by
      default. Its default is now dynamic target-motion onset. The accidental
      `long7_v3` fixed-prefix step on `127350.35` was terminated without
      cancelling the held allocation, and its partial artifacts should be
      ignored.
- [x] The active live loop now also fixes the pretrigger source-restore
      weakness. Default `--pretrigger-control-mode` is
      `frozen_dp_until_target_motion`: restore source frame `0`, run frozen DP
      live while replaying only target actor motion, trigger on observed
      target motion, then call Cosmos. Dry-run
      `live_pretrigger_dp_dynamic_trigger_dryrun_iter2100_sample00_20260613_152400`
      verified `106` real DP pretrigger steps and a causal trigger at frame
      `106`. Full smoke
      `live_pretrigger_dp_dynamic_trigger_cosmos1_iter2100_sample00_20260613_152608`
      then executed one Cosmos chunk after that live prefix. Final live
      success remained `false`; DP handoff was correctly blocked by the
      static-DP continuability profile. The reviewed 16-frame rollout sheet
      shows target motion and no valid insertion/rebind.
- [x] `run_cosmos3_live_receding_loop.py` now supports automatic observed
      prefix role inference from live history. A false `peg_recovery` trigger
      caused by one noisy current grasp predicate at f97 was fixed by
      requiring stable lost grasp after recent grasp history is absent, or an
      explicit peg perturbation scenario. A local function check on the f97
      history returns `target_pre_motion`.
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
- [x] 2026-06-13 continuation status: the current active root
      `sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`
      has resumed beyond the earlier wall-time stop. The 4-H200 Slurm job
      `127281` on `server31` is the only active SFT checkpoint writer. It
      saved `iter_000001500`, finished validation, and was auto-resumed from
      that checkpoint toward `iter_000002100`. Do not start a second two-GPU
      SFT writer into the same root while this job is alive; spare 2-H200 jobs
      may run read-only eval/watchers with independent eval roots, or resume
      only after the 4-H200 writer disappears.
- [x] Iter1200 strict eval/readout/profile completed for the current root.
      Structure passed for `10` samples: generated/reference videos stayed
      `301/301`, action tensors stayed `300x32`, and `strict_failures=[]`.
      Aggregate metrics were mean future PSNR `21.4227`, mean action RMSE
      `0.4559`, robot-action future RMSE `0.7272`, state-sidecar future RMSE
      `0.4704`, and generated-RGB mean final hole error `0.0993` m.
      Direct review of all `10` sheets failed closed-loop handoff: fast-shift
      and sine moving-hole samples have wrong final peg/hole/hand relative
      geometry, and static `none` samples show target/object drift with
      target-motion false firing. The gate file records
      `closed_loop_allowed=false` with reason
      `explicit_visual_review_not_passed`.
- [x] The spare 2-H200 allocation `127286` completed a read-only
      `iter_000001200` extra-30 validation panel at
      `eval_full_episode_wam_iter_000001200_extra30_2gpu_20260613_0430`.
      It fixed `CHECKPOINT_PATH` to `iter_000001200` and wrote to a separate
      eval root; it did not write checkpoints, touch `latest_checkpoint.txt`,
      or replace the 4-H200 SFT writer. Strict artifacts and generated-RGB
      readout passed structure for `30/30` samples, but the panel confirms the
      iter1200 controller-negative diagnosis: mean final hole error is
      `0.0885` m, mean future hole/peg/TCP RMSE is
      `0.0504/0.0536/0.0512` m, and direct sheet review still shows unsafe
      target/peg/hand relative-pose errors in fast-shift/sine/static cases.
- [x] The same 2-H200 allocation `127286` is now reserved for a read-only
      `iter_000001500` extra-30 panel, now complete at
      `eval_full_episode_wam_iter_000001500_extra30_2gpu_20260613`. Strict
      structure/readout passed for `30/30` samples, but representative visual
      review still shows fast-shift/sine/static relative-geometry failures, so
      it does not override the failed main gate.
- [x] Iter1500 main strict eval/readout/profile completed for `10/10` samples,
      but visual review failed sheets `03`, `04`, `08`, and `09`. The file
      `eval_full_episode_wam_iter_000001500/manual_visual_review.json` records
      the per-sheet verdict, and
      `eval_full_episode_wam_iter_000001500/closed_loop_gate_visual_review.json`
      records `closed_loop_allowed=false`.
- [x] Iter1800 main watcher remains active and read-only: tmux
      `cosmos3_v7_733_iter1800_watch_0613` on job `127288` waits for the main
      10-sample gate. The earlier extra30 watcher on job `127286` was stopped
      by interrupting the foreground tmux/srun command, not by `scancel`, so
      the 2-H200 allocation stayed alive. It was first converted to a
      no-concurrent-writer fallback, then upgraded to an active independent
      two-GPU shadow continuation in tmux
      `cosmos3_v7_733_shadow2gpu_from1500_to2100_0613`. The shadow output root
      is
      `sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_2gpu_shadow_from1500_to2100_20260613_0637`.
      It hardlinks the main `iter_000001500` checkpoint into a separate root,
      loaded that checkpoint successfully in iteration `1500`, and trains
      toward `iter_000002100` without writing the 4-H200 main root.
- [x] The old read-only `iter1800` watcher step `127288.23` on `server03` was
      Slurm-terminated at `2026-06-13T06:46:34+08:00` while waiting for the
      checkpoint. To avoid holding a GPU allocation idle, replacement tmux
      `cosmos3_v7_733_iter1800_eval_request_on_ckpt_0613` now only polls the
      checkpoint files from the login node and requests a fresh 1-H200
      `salloc` eval allocation after `iter_000001800` exists and is stable.
      This login watcher performs no rollout/render/training work before the
      compute allocation starts.
- [x] Main SFT saved `iter_000001800` at `2026-06-13T07:28:00+08:00`, but
      `latest_checkpoint.txt` still read `iter_000001500`. The eval request
      watcher was repaired to use the stable target checkpoint directory plus
      `model/.metadata` as the trigger and to record latest mismatch instead
      of blocking. Tmux
      `cosmos3_v7_733_iter1800_eval_request_on_ckpt_v2_0613` requested
      1-H200 Slurm job `127350` for the strict iter1800 eval/readout/gate.
- [x] Iter1800 strict eval/readout/profile completed in Slurm job `127350` on
      `server10`. Structural gates passed for `10/10` samples:
      generated/reference videos stayed `301/301`, action tensors stayed
      `300x32`, and `strict_failures=[]`. Manual review opened all `10`
      review sheets and recorded `pass_with_caution`: `8` pass,
      `2` pass-with-caution, `0` fail. This allowed only gated live smoke,
      not controller success evidence.
- [x] Iter1800 live-smoke diagnostics ran after the visual gate. Sample `0`
      executed an `8`-step Cosmos action chunk plus `8` DP resume steps and
      did not reach final success. A DP-resume implementation bug was then
      fixed: requests longer than the DP `act_horizon=8` now repeatedly
      recompute DP actions from the latest observation instead of silently
      executing only one 8-step block. Corrected sample `3` executed
      `8 + 32` steps and still ended with `success=false`; visual contact
      sheet review agrees that insertion was not completed. These are
      negative live controller diagnostics, not method success.
- [x] Superseded resource note from before the 2026-06-13 closed-loop audit:
      primary and shadow SFT were active then, but the latest user instruction
      stops further training while the closed-loop/DP handoff implementation
      is audited. Current checked Slurm steps for `127281`, `127286`, and
      `127350` are extern allocations only; no active SFT step is running.
- [x] Historical resource use before the 2026-06-13 closed-loop audit:
      primary SFT job
      `127281` on `server31` was the only writer to the main root and was
      observed at rank-0 iteration `1893`; all `4` GPUs were then at `100%`
      utilization. Independent two-GPU shadow continuation job `127286` on
      `server40` wrote only
      `sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_2gpu_shadow_from1500_to2100_20260613_0637`,
      was observed at iteration `1774`, and both GPUs were then at `100%`
      utilization. The one-GPU eval allocation `127350` was used for live
      smoke diagnostics and must not be counted as SFT training.
- [x] A lightweight login-side request watcher was started in tmux
      `cosmos3_v7_733_iter2100_eval_request_on_ckpt_0613` for the next main
      checkpoint. It only polls checkpoint files on the login node and will
      request a fresh 1-H200 eval allocation after `iter_000002100` is stable.
      Its first poll at `2026-06-13T08:03:32+08:00` reported
      `latest=iter_000001500`, `has_target_dir=no`, `has_model_metadata=no`.
- [x] While waiting for the main `iter2100` checkpoint, the held 1-H200 eval
      allocation `127350` was used for useful controller diagnostics instead
      of sitting idle. The `iter1800` full 10-sample live-smoke panel with
      `8 + 32` steps reached only `1/10` final simulator success. The longer
      `8 + 96` recomputed-DP-resume panel reached `6/10` final simulator
      success but still failed reverse, one continuous-insert, peg-drop, and
      static-late cases. This shows longer DP resume can complete several
      samples after a Cosmos chunk, but the current interface is still a
      diagnostic one-shot-chunk plus long DP takeover, not full receding
      Cosmos closed-loop evidence.
- [x] A read-only strict eval/readout/profile chain for the independent
      2-GPU shadow branch `iter_000001800` was launched on job `127350` with
      output root
      `sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_2gpu_shadow_from1500_to2100_20260613_0637/eval_full_episode_wam_iter_000001800`.
      It passed strict structure and generated-RGB readout/profile, but manual
      visual review failed fast-shift, peg-drop, and static-monitor handoff
      geometry. `closed_loop_allowed=false` is recorded with
      `visual_review_status=fail`. This is a branch diagnostic only and does
      not replace the primary main-root `iter2100` gate.
- [x] The original `iter2100` watcher that would request a fresh `salloc` was
      stopped by sending `Ctrl-C` to its tmux pane, not by cancelling any Slurm
      allocation. It was replaced by tmux
      `cosmos3_v7_733_iter2100_eval_existing127350_0613`, which only polls
      checkpoint files on the login node and then uses the already held
      1-H200 allocation `127350` for eval once `iter_000002100` is stable.
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
- [x] 2026-06-13 resource-protection update after the two-card correction:
      primary 4-H200 step `127281.40` is actively training the main root from
      `iter_000002100 -> iter_000002700`, and independent 2-H200 step
      `127286.33` is actively training the shadow root from
      `iter_000002100 -> iter_000002700`. Both roots now also have
      no-concurrent-writer protection watchers for `2700 -> 3300`:
      `cosmos3_v7_733_main4gpu_auto_from2700_to3300_0613` and
      `cosmos3_v7_733_shadow2gpu_auto_from2700_to3300_0613`. These watchers
      only launch after the target `iter_000002700` checkpoint exists, is the
      latest checkpoint, and the allocation has no non-extern Slurm step.
      They are intended to prevent held GPUs from idling after the current
      600-step continuation, not to change the SFT recipe or write into the
      same root concurrently.
- [x] 2026-06-13 two-card utilization recheck: the 2-H200 allocation is not
      idle. Slurm job `127286` requests `gres/gpu=2`, and compute-node
      `nvidia-smi` inside `server40` showed both GPUs at `100%` utilization
      with about `79GB` used per GPU while step `127286.33` trained the
      independent shadow root. Main job `127281` on `server31` simultaneously
      showed all four GPUs at `100%` utilization while step `127281.40`
      trained the main root.
- [x] Primary `iter_000002400` strict eval/readout/profile and visual review
      completed. Structural artifacts passed (`301/301` videos,
      `300x32` actions, `strict_failures=[]`), but metrics regressed versus
      `iter2100` and manual review failed sheets `00`, `04`, and `08` for
      late/final robot-peg-hole relative-geometry drift. The recorded gate is
      `closed_loop_allowed=false`, reason `explicit_visual_review_not_passed`;
      no closed-loop smoke should run from `iter2400`.
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
- [x] 2026-06-13 resource correction: the primary 4-H200 root reached
      `iter_000002100`, passed strict eval/readout and direct visual review
      as a smoke-permitted checkpoint, then was immediately resumed in the
      same held allocation `127281` from `2100 -> 2700` with `MAX_ITER=2700`.
      A failed first `srun` used too many CPUs for this allocation; it was
      retried with the actual `32` CPU allotment and resumed successfully.
      The previous log was snapshotted before the wrapper overwrote
      `sft_train.log`.
- [x] The spare 2-H200 allocation was not left idle: independent shadow SFT
      `127286.27` continues in
      `sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_2gpu_shadow_from1500_to2100_20260613_0637`
      and is watched by tmux
      `cosmos3_v7_733_shadow2gpu_auto_from2100_to2700_0613`. That watcher only
      polls Slurm/checkpoint files until the current shadow writer exits and
      `iter_000002100` is complete, then resumes the same shadow root from
      `2100 -> 2700` inside allocation `127286`.
- [x] Primary `iter_000002100` strict eval/readout/profile completed in held
      eval allocation `127350`. The generated artifacts passed the full
      `301` RGB frame / `300x32` action contract with
      `strict_failures=[]`. The gate metrics were mean future PSNR
      `22.7164` dB, mean action RMSE `0.4049`, robot-action future RMSE
      `0.6779`, state-sidecar future RMSE `0.4099`, generated-RGB mean final
      hole error `0.0688` m, and future hole/peg/TCP RMSE
      `0.0420/0.0433/0.0416` m. Manual review opened all ten sheets and
      recorded `pass_with_caution` (`8` pass, `2` pass-with-caution, `0`
      fail), so the closed-loop gate permits diagnostic live smoke only.
- [x] Primary `iter_000002100` DP96 live-smoke panel completed:
      `closed_loop_smoke_iter_000002100_representative_dp96_recompute_20260613_0928`.
      It executed one `8`-step Cosmos robot-action chunk followed by `96`
      recomputed frozen-DP resume steps for all ten validation samples.
      Simulator success was `5/10` (`sample_00`, `01`, `03`, `06`, `08`);
      failures were `sample_02`, `04`, `05`, `07`, `09`. The opened contact
      sheet matches the metrics: failed rows remain visibly uninserted or
      misaligned. This is not full receding-Cosmos controller evidence and not
      a method success claim.
- [x] The login-side watcher for primary `iter_000002400` eval used existing
      allocation `127350` and completed the strict eval/readout/profile chain.
      Agent visual review then blocked the checkpoint; the next primary gate
      is `iter_000002700`, not `iter2400`.
- [ ] Future closed-loop work must follow
      `TODO/cosmos3_300f_world_model/08_receding_closed_loop.md`: no one-shot
      300-step open-loop Cosmos execution, no sidecar/oracle simulator state,
      de-normalize only robot-action columns before live `env.step`, and
      execute short `<=8`-step prefixes with real re-observation. The latest
      2026-06-13 DP handoff correction also applies here: a passing live
      continuability gate may choose frozen DP, but only for a short
      reobserved chunk before returning to the same observe-decide loop. Do
      not report any eval that runs `32`/`64`/`96` blind DP steps after one
      Cosmos chunk as closed-loop method evidence.
