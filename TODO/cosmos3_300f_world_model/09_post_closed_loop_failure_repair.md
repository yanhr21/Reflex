# Post Closed-Loop Failure Repair TODO

## Current Boundary

- [x] Corrected long-horizon live-receding closed-loop eval failed on
      `iter_000002100`, sample `00`, after `12` Cosmos chunks through frame
      `202`. The source teacher inserts by frame `166`; the live rollout ends
      with peg-head-at-hole `[-0.1423, 0.0004, 0.0101]` and
      `success=false`.
- [x] Action-index audit found no simple temporal offset: action row `f` is
      the correct first future action when the observed prefix ends at frame
      `f`; `shift_action=1` is an action noise-schedule shift.
- [x] Added read-only distribution audit script:
      `scripts/world_model/audit_cosmos3_receding_training_distribution.py`.
      Report:
      `docs/world_model_task_rebinding/2026-06-13_receding_training_distribution_audit.json`.
      It found `1193/2899` role/mode mismatches, no action condition-mask
      errors, and `1287` late-rebind proxy rows.
- [x] Added read-only hard-case action/rebind diagnostic:
      `scripts/world_model/analyze_cosmos3_hard_case_action_rebind.py`.
      Report:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_hard_screen2_action_rebind_analysis.json`.
      On six hard-screen-2 samples where pure DP failed, Cosmos succeeded on
      only one. The failures are dominated by C_pi y/z blocks, occasional
      grasp loss, and predicted robot-action direction/scale mismatch versus
      same-H5 source-teacher rows. This supports the repair plan's premise:
      the current full-length live loop is no longer the main blocker; direct
      raw Cosmos action rebinding is the unreliable component.
- [x] Added objective-level closed-loop gate:
      `scripts/world_model/check_cosmos3_closed_loop_objective_gate.py`.
      Report:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_objective_gate.json`.
      It marks `implementation_contract_ok=true` and
      `method_effectiveness_ok=false`. The gate directly scanned `26/26`
      raw/annotated mp4 files and found `301` frames for every one, with
      duration `10.033333333333333` seconds. This is the current repair
      boundary: the old short-video/manual-trigger issue has current artifact
      evidence showing it is fixed, but the checkpoint still underperforms
      pure DP on val and only rescues `1/6` hard pure-DP failures. Do not
      launch broad method claims from the old checkpoint; proceed to
      clean-role/dense-receding preflight and approved overfit/full SFT when
      allowed.
- [x] Recorded the completed val/hard pure-DP comparison plus an incomplete
      step-level handoff probe:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_val_hard_puredp_comparison_and_stepgate_probe.md`.
      The completed comparisons are negative/mixed: val Cosmos `1/3` versus
      pure DP `3/3`, and hard-screen-2 Cosmos `1/6` on pure-DP failures. The
      `--cosmos-step-handoff-gate` probe on hard-screen-2 index `12` hit the
      allocation time limit before full `300/301` completion, so it is not
      method evidence. Its partial path reached frame `188` with
      `success=false`; it briefly found a real-state C_pi pass, then DP
      handoff drifted out of the gate. This supports the repair premise that
      direct action rebinding and DP-continuability are unstable, not that
      another broad eval from the same checkpoint is likely to solve the issue.
- [x] Added direct video-length contract audit:
      `scripts/world_model/audit_video_length_contract.py`.
      Report:
      `docs/world_model_task_rebinding/2026-06-14_iter2100_vs_iter2700_video_length_audit.json`.
      It confirms the old user-flagged iter2100 panel was genuinely short
      (`131` and `119` final rollout frames, under `4.4s`) and is historical
      negative evidence only. The current iter2700 val/hard/pure-DP/static
      evidence roots scan as `26/26` final videos with `301` frames and
      `10.0333s`. This prevents the old incomplete-video bug from being
      confused with the remaining current failure, which is method
      effectiveness/action rebinding.
- [x] Hardened future video artifact summaries. The live receding loop and
      pure-DP baseline now decode their just-written raw/annotated mp4 files
      and record `final_observed_video_inspection`,
      `final_observed_annotated_video_inspection`, and
      `video_file_contract_ok`. Their panel wrappers propagate those fields.
      Future repair/eval runs should treat `video_file_contract_ok=false` as
      an implementation artifact failure, not as controller evidence.
- [x] 2026-06-14 post-hard-comparison code audit recorded in
      `docs/world_model_task_rebinding/2026-06-14_iter2700_post_hard_compare_code_audit.md`.
      The live loop, live-prefix input builder, action extractor, and prefix
      inference wrapper were inspected for the current failure. The action
      row timing is internally consistent: frame `f` conditions rows `0..f-1`
      and executes predicted row `f` as the next action. This does not support
      a simple off-by-one explanation. The stronger diagnosis remains
      training/query distribution mismatch: the iter2700 condition root audit
      found `1193/2899` role/mode mismatches, while the corrected live loop
      queries physical-mode roles from real history. The next-action gate was
      rerun and still rejects broad panels and SFT resume from this checkpoint.
- [x] 2026-06-14 added live-query training coverage audit:
      `scripts/world_model/audit_cosmos3_live_query_training_coverage.py`,
      with coverage self-test
      `scripts/world_model/selftest_cosmos3_live_query_training_coverage.py`.
      Current output:
      `docs/world_model_task_rebinding/2026-06-14_iter2700_live_query_training_coverage_audit.json`
      and `.md`. On the val panel plus hard-screen-2 pure-DP-failure panel,
      the old iter2700 SFT distribution has `1193/2899` role/mode mismatches
      and `74/173` live Cosmos queries lack a local role/mode-consistent
      training neighbor under the configured prefix and peg-head-at-hole
      tolerances. This turns the distribution-mismatch diagnosis into a
      machine-checkable repair criterion for the clean-role/dense-receding
      condition root.
- [x] 2026-06-14 wired live-query coverage into the clean/dense preflight
      readiness chain. `summarize_cosmos3_clean_dense_preflight.py` now accepts
      `--live-query-coverage-audit-json` and fails `ready_for_overfit` if the
      coverage audit has role/mode mismatch, no live queries, condition-root
      mismatch, or more undercovered live queries than the configured
      threshold. The base full-episode WAM wrapper can generate this audit
      during condition preflight. The full v7_733 clean/dense preflight wrapper
      defaults `RUN_LIVE_QUERY_COVERAGE_AUDIT=true` using the current val and
      hard-screen-2 panel summaries; the two-source overfit2 preflight defaults
      it off because that check is only a chain/overfit sanity path.
      `scripts/world_model/selftest_cosmos3_clean_dense_preflight_summary.py`
      covers pass/fail readiness behavior.
- [x] 2026-06-14 hardened the full v7_733 clean/dense preflight wrapper
      against accidentally skipping live-query coverage. Real execution now
      refuses when `RUN_LIVE_QUERY_COVERAGE_AUDIT=false` unless
      `ALLOW_SKIP_LIVE_QUERY_COVERAGE_DIAGNOSTIC=true` is explicitly set, and
      the diagnostic override must not unlock overfit/full SFT evidence. The
      wrapper also checks that all configured live-query panel summary paths
      exist before calling the base preflight wrapper. Login-safe coverage
      guard behavior is covered by
      `scripts/world_model/selftest_cosmos3_clean_dense_preflight_guards.sh`.
      The base preflight summary also accepts
      `--diagnostic-not-ready-reason`; when the skip override is used, the
      wrapper passes `live_query_coverage_audit_skipped_by_diagnostic_override`
      so `clean_dense_preflight_summary.json` records
      `diagnostic_not_method_ready` and `ready_for_overfit=false`.
- [x] 2026-06-14 hardened the clean/dense overfit SFT entry against
      inconsistent readiness summaries. `check_cosmos3_next_action_gate.py`
      now denies overfit SFT if `failed_checks` is nonempty or
      `diagnostic_not_ready_reason` is present, even when
      `ready_for_overfit=true`. The overfit SFT wrapper now calls
      `scripts/world_model/check_cosmos3_clean_dense_preflight_summary_ready.py`,
      which independently requires `ready_for_overfit=true`, zero failed
      checks, no diagnostic-not-ready reason, and a matching condition root.
      This validator is covered by
      `scripts/world_model/selftest_cosmos3_clean_dense_preflight_summary_ready.py`.

## Do Not Do

- [ ] Do not resume SFT from the current condition root as if more iterations
      are the answer.
- [ ] Do not run more broad smoke/panel eval from the failed checkpoint.
- [ ] Do not relax `C_pi`, use generated sidecar optimism as authority, or
      return to one-shot Cosmos plus long DP takeover.
- [ ] Do not build 128/129-frame or 93-frame diagnostic samples as method data.

## Next Repair Items

- [x] Add a clean-role/dense-receding export mode to
      `export_cosmos3_maniskill_full_episode_wam_conditions.py`.
      It should use the actual physical mode as the controller-facing role,
      store any sampled curriculum label separately, and keep all rows as full
      `301/300` samples. Implemented with
      `--prefix-role-source physical_mode` and
      `--dense-receding-prefix-stride N`; defaults preserve old sampled-role
      behavior.
- [x] Add hard preflight/reporting for role/mode mismatch, prefix-frame
      histograms by physical mode, late-rebind proxy coverage, and future
      8-step teacher-action statistics. `preflight_cosmos3_full_episode_wam_contract.py`
      now reports physical mode and role/mode counts, and
      `audit_cosmos3_receding_training_distribution.py` reports mismatch,
      prefix, late-rebind, and teacher-action statistics.
- [x] Run a small no-training/no-render probe of the clean/dense export path.
      `/tmp/cosmos3_clean_dense_export_probe_2821833` used `max_records=2`,
      `--prefix-role-source physical_mode`, and
      `--dense-receding-prefix-stride 8`; it wrote `23` full-episode rows,
      `sampled_prefix_role_counts={'dense_receding': 15, ...}`, and
      `prefix_role_mode_mismatch_count=0`.
- [x] Add optional late-rebind row repetition to
      `build_cosmos3_role_weighted_sft_jsonl.py`. With
      `--late-rebind-weight 3` on the `/tmp` clean/dense probe, `23` full
      rows became `59` full rows; the `18` late-rebind proxy rows were
      repeated, not sliced or relabeled.
- [x] Add receding-distribution audit to the Slurm condition audit chain. If
      `PREFIX_ROLE_SOURCE=physical_mode`, nonzero role/mode mismatch is now a
      hard audit failure before SFT. The audit script also owns the hard gate
      now; the wrapper no longer uses an inline Python check.
- [x] Add a compute-node-only clean/dense preflight wrapper:
      `scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_preflight_in_allocation.sh`.
      It defaults `RUN_SFT=false`, `FORCE_EXPORT=true`,
      `PREFIX_ROLE_SOURCE=physical_mode`,
      `DENSE_RECEDING_PREFIX_STRIDE=8`, `LATE_REBIND_WEIGHT=3`, and
      `MIN_LATE_REBIND_CANDIDATES=1`.
- [x] Harden the clean/dense preflight wrappers against accidental execution.
      Both the full v7_733 preflight wrapper and the two-source overfit2
      preflight wrapper now require `ALLOW_CLEAN_DENSE_PREFLIGHT=true` for
      real execution. Their `DRY_RUN_CONFIG_ONLY=true` mode remains login-safe
      and does not require approval. With the approval variable set on a login
      node, the wrapper runs only the read-only next-action gate and is then
      stopped by the base compute-node guard before any export/training.
- [x] Add clean/dense preflight summary:
      `scripts/world_model/summarize_cosmos3_clean_dense_preflight.py`.
      It writes `clean_dense_preflight_summary.json/.md` and exposes a single
      `ready_for_overfit` flag after checking condition manifest, strict
      full-episode preflight, receding distribution audit, and weighted train
      JSONL metadata. A `/tmp` simulated summary probe reached
      `ready_for_overfit=true` after expected files were present.
- [x] Add a login-safe config dry-run to the clean/dense preflight wrapper.
      `DRY_RUN_CONFIG_ONLY=true bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_preflight_in_allocation.sh`
      prints the exact source/condition/output roots and confirms
      `RUN_SFT=false`, `PREFIX_ROLE_SOURCE=physical_mode`,
      `DENSE_RECEDING_PREFIX_STRIDE=8`, `LATE_REBIND_WEIGHT=3`, and
      `MIN_LATE_REBIND_CANDIDATES=1` without launching export, training,
      rendering, or eval.
- [x] Re-ran the login-safe dry runs after the iter2700 objective gate. The
      full v7_733 clean/dense preflight still reports
      `expected_source_episodes=733`, `force_export=true`, `run_sft=false`,
      `prefix_role_source=physical_mode`,
      `dense_receding_prefix_stride=8`, `late_rebind_weight=3`, and
      `min_late_rebind_candidates=1`. The overfit2 preflight still reports
      `expected_source_episodes=2`, `max_records=2`, and `run_sft=false`.
      `scripts/world_model/print_cosmos3_clean_dense_preflight_commands.sh`
      prints the approval-time commands and explicitly says they must be run
      from inside a compute-node `srun` step after explicit user approval.
- [x] Add `MAX_RECORDS` passthrough to the base full-episode WAM wrapper and a
      clean/dense two-source overfit-preflight wrapper:
      `scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_overfit2_preflight_in_allocation.sh`.
      It defaults `EXPECTED_SOURCE_EPISODES=2`, `MAX_RECORDS=2`, and
      `RUN_SFT=false`, so the future overfit condition root can be audited
      before any training starts.
- [x] Add the guarded clean/dense two-source overfit SFT wrapper:
      `scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_overfit2_fix1recipe_in_allocation.sh`.
      It refuses to run unless `ALLOW_CLEAN_DENSE_OVERFIT_SFT=true`,
      `CONDITION_ROOT` points at the approved clean/dense overfit2 condition
      root, and `CLEAN_DENSE_PREFLIGHT_SUMMARY` records
      `ready_for_overfit=true` for the same condition root.
- [x] Harden the overfit SFT wrapper's approval boundary. A login-safe
      `DRY_RUN_CONFIG_ONLY=true` now reports the required
      `CONDITION_ROOT`, explicit `CLEAN_DENSE_PREFLIGHT_SUMMARY`, fix1 recipe
      defaults, and refusal conditions without validating preflight or
      launching export/training/render/eval. Real execution still exits before
      any work unless explicit user approval and a matching ready-for-overfit
      summary are present.
- [x] 2026-06-14 re-ran the login-safe clean/dense dry-run checks after the
      hard pure-DP comparison. The full v7_733 clean/dense preflight still
      prints `RUN_SFT=false`, `PREFIX_ROLE_SOURCE=physical_mode`,
      `DENSE_RECEDING_PREFIX_STRIDE=8`, `LATE_REBIND_WEIGHT=3`, and
      `EXPECTED_SOURCE_EPISODES=733`. The overfit2 preflight still prints
      `EXPECTED_SOURCE_EPISODES=2`, `MAX_RECORDS=2`, and `RUN_SFT=false`.
      The guarded overfit SFT dry-run still refuses real execution without
      `ALLOW_CLEAN_DENSE_OVERFIT_SFT=true`, an explicit `CONDITION_ROOT`, and
      a matching ready-for-overfit summary. `selftest_cosmos3_next_action_gate.py`
      passed, and a corrected objective-gate rerun over the current artifacts
      again returned `implementation_contract_ok=true`,
      `method_effectiveness_ok=false`, with next action
      `proceed_to_clean_dense_repair_preflight_when_user_approved`. No export,
      training, rendering, or eval was launched by these checks.
- [x] 2026-06-14 hardened the next-action gate with the requirement-level
      audit. `check_cosmos3_next_action_gate.py` now accepts
      `--requirement-audit-json`, and
      `check_current_cosmos3_next_action_gate.sh` automatically passes
      `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_requirement_audit.json`
      when it exists. Old-checkpoint resume and broad-panel actions are now
      denied not only by controller-blocked status flags, but also by
      `closed_loop_requirement_audit_not_achieved`. Current gate outputs:
      `docs/world_model_task_rebinding/2026-06-14_cosmos3_next_action_gate_resume_current_requirement_audit.json`
      and
      `docs/world_model_task_rebinding/2026-06-14_cosmos3_next_action_gate_launch_broad_panel_requirement_audit.json`.
      The self-test now covers the case where status flags falsely claim the
      old checkpoint is safe while the requirement audit is incomplete.
      The gate also now accepts `--clean-dense-preflight-summary-json` for the
      repair path. `clean_dense_overfit_sft_after_user_approval` is denied
      until user approval is explicit and a supplied clean/dense preflight
      summary records `ready_for_overfit=true`. This repair action is not
      blocked merely because the old iter2700 requirement audit is incomplete.
      Current missing-summary output:
      `docs/world_model_task_rebinding/2026-06-14_cosmos3_next_action_gate_clean_dense_overfit_requires_ready_summary.json`.
      The guarded overfit SFT Slurm wrapper now invokes this gate before the
      inline ready-summary/condition-root check and before the base training
      wrapper. Its default gate output is
      `next_action_gate_clean_dense_overfit_sft.json` beside the supplied
      preflight summary, so training entry has the same audit trail as the
      standalone gate.
- [x] 2026-06-14 hardened the objective gate against weak future evidence.
      Moving-target samples must now have causal detector provenance
      (`prefix_selection.mode=target_motion_onset`, concrete detector and
      streak frame indexes) and at least `8` WM-active frames, so a manifest
      prefix, hidden target-onset disclosure, or token one-frame Cosmos
      activation cannot pass the implementation contract. The no-motion sample
      must explicitly record the causal detector-never-triggered mode with no
      detected frame and
      `pretrigger_control_mode=frozen_dp_until_target_motion`, so it verifies
      the same detector/controller path instead of a static-special branch.
      The gate now also checks the structured `301`-frame
      controller/annotated timelines used for the overlay video, so explicit
      Cosmos takeover annotation is verified from metadata in addition to mp4
      existence/frame count. The official gate output was regenerated and
      still says `implementation_contract_ok=true`,
      `method_effectiveness_ok=false`.
- [x] 2026-06-14 added launch-time guards to the live-receding wrappers. The
      panel wrapper and single-sample loop wrapper now refuse non-method modes
      unless `ALLOW_LIVE_RECEDING_DIAGNOSTIC=true` is explicitly set:
      manifest/manual prefixing, source-restored pretrigger control, explicit
      role overrides, or disabled Cosmos inference. This guard prevents future
      closed-loop repair/eval jobs from silently falling back to the exact
      hidden-onset or no-Cosmos diagnostic paths the user rejected. `bash -n`
      passed for both wrappers, and fake-Slurm diagnostic-mode probes exited
      `42` before heavy work.
- [x] 2026-06-14 added and ran
      `scripts/world_model/selftest_cosmos3_closed_loop_objective_gate.py`.
      The self-test covers the non-heavy gate logic for causal detector
      provenance, minimum WM-active use, and no-motion detector-never-triggered
      behavior, including the structured annotation-timeline contract. It
      passed together with `selftest_cosmos3_next_action_gate.py` and wrapper
      syntax checks.
- [x] 2026-06-14 guarded the legacy one-shot closed-loop wrappers. The old
      panel/single wrappers now require
      `ALLOW_OLD_ONESHOT_CLOSED_LOOP_DIAGNOSTIC=true` and otherwise exit `43`
      before creating outputs. This prevents the old one-shot Cosmos chunk plus
      optional DP resume path from being accidentally relaunched as the
      corrected full-episode live-receding method.
- [x] 2026-06-14 added
      `scripts/world_model/selftest_cosmos3_closed_loop_wrapper_guards.sh`.
      It consolidates the fake-Slurm wrapper refusal checks for non-method
      live-receding modes and old one-shot wrappers into one local command.
      The wrapper-guard self-test passed with the objective-gate and
      next-action-gate self-tests.
- [x] 2026-06-14 guarded the old sampled-role v7_733 SFT launch wrappers.
      `run_cosmos3_300f_full_episode_wam_fix3_v7_733_fix1recipe_in_allocation.sh`
      now refuses `RUN_SFT=true` by default with exit `66`, and the historical
      sampled-role overfit2 wrapper refuses with exit `67`. Both wrappers keep
      a login-safe `DRY_RUN_CONFIG_ONLY=true` mode and require explicit
      legacy diagnostic overrides for non-method reproduction. Active repair
      must use the clean-role/dense-receding preflight and guarded clean/dense
      overfit SFT path instead. This prevents the already diagnosed
      role/mode-mismatched condition root from being accidentally resumed as
      if more SFT iterations were the fix. Guard behavior is covered by
      `scripts/world_model/selftest_cosmos3_legacy_sft_wrapper_guards.sh`.
- [x] 2026-06-14 made hard-case usefulness explicit in the objective gate.
      The gate now records `min_hard_case_success_fraction=0.5` on the panel
      restricted to full pure-DP failures, so a single rescued sample cannot be
      mistaken for "Cosmos works on most large-motion dynamic cases." The old
      stricter all-success failure remains. Current iter2700 fails with
      `hard_case_success_fraction_below_minimum:1/6<min_fraction:0.5` and
      `hard_case_not_broadly_reliable:1/6`.

## Approval-Time Command

- [ ] After user approval only, run inside a held compute allocation:

      `ALLOW_CLEAN_DENSE_PREFLIGHT=true bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_preflight_in_allocation.sh`

      This should produce a new clean/dense condition root and preflight output
      root with `RUN_SFT=false`. Do not start overfit or full SFT from it until
      `clean_dense_preflight_summary.json` says `ready_for_overfit=true` and
      the user approves the next training step.
- [ ] Re-export a condition root from the existing accepted v7_733 RGB/H5
      sources only after the user approves this repair direction. Do not
      regenerate H5 data for this step.
- [ ] If training is approved later, start with the user-approved two-sample
      overfit pattern on the clean/dense condition root before any full SFT.
- [ ] If clean/dense direct Cosmos actions still underreact after overfit and
      full-run evidence, switch the next method discussion to a learned
      short-chunk executor or DP-prior policy conditioned on Cosmos-predicted
      task state. Keep real-state `C_pi` as the handoff authority.

## Evidence To Collect After Approval

- [ ] Condition audit JSON proving full `301/300` contract and clean role/mode
      semantics.
- [ ] Role/mode/prefix distribution report from
      `audit_cosmos3_receding_training_distribution.py`.
- [ ] Overfit eval videos with strict same-length artifacts and direct visual
      review.
- [ ] Full eval videos/readout/action metrics only after overfit passes.
- [ ] Corrected live-receding closed-loop panel only after generated artifacts
      and visual gate pass.
