# 2026-06-14 Iter2700 Post-Hard-Compare Code Audit

## Scope

This is a login-safe code/evidence audit after the val and hard pure-DP
comparisons. It does not launch Slurm work, rendering, training, or simulator
rollout.

Physical question: after target motion is detected, can the current Cosmos3
interface generate a short action chunk that moves the live peg toward a
DP-continuable insertion state, and can the loop hand back to DP only when the
real state is safe?

## Current Evidence

- Objective gate:
  `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_objective_gate.json`
- Requirement audit:
  `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_requirement_audit.json`
- Failure-mode reducer:
  `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_failure_modes.json`
- Hard action/rebind analysis:
  `docs/world_model_task_rebinding/2026-06-14_iter2700_hard_screen2_action_rebind_analysis.json`

The implementation contract is currently repaired:

- `26/26` current raw/annotated videos scan as `301` frames, `30 fps`, and
  `10.0333s`.
- Moving-target samples use causal `target_motion_onset` detector metadata.
- No-motion evidence uses the same detector/controller path, records
  `pretrigger_control_mode=frozen_dp_until_target_motion`, and records no
  Cosmos activation.
- Videos carry explicit controller timeline annotations.

The method effectiveness gate still fails:

- Val comparison: Cosmos `1/3`, same-source pure DP `3/3`.
- Hard-screen-2 comparison on six pure-DP failures: Cosmos `1/6`, matched pure
  DP `0/6`.
- Hard-case usefulness threshold: the gate now records
  `min_hard_case_success_fraction=0.5`; current hard-case success fraction is
  `0.1667`, so the artifact explicitly fails
  `hard_case_success_fraction_below_minimum:1/6<min_fraction:0.5`.

## Code Path Checked

- Live loop:
  `scripts/world_model/run_cosmos3_live_receding_loop.py`
- Live prefix input builder:
  `scripts/world_model/build_cosmos3_live_prefix_wam_input.py`
- Policy action extraction:
  `scripts/world_model/extract_cosmos3_policy_action_chunk.py`
- Live prefix wrapper:
  `scripts/slurm/run_cosmos3_live_prefix_policy_inference_in_allocation.sh`

The action-time indexing is internally consistent:

- At live frame `f`, the builder conditions action/state rows `0..f-1`.
- Future rows are zero/unconditioned for controller-facing inference.
- The extractor reads predicted action row `f` as the next executable action,
  then executes only a short chunk before reobserving.
- This matches the intended row semantics: action row `f` transitions frame
  `f` to frame `f+1`.

This audit therefore does not support a simple off-by-one explanation for the
current hard failures.

## Main Technical Diagnosis

The stronger current explanation is training/query distribution mismatch.
The condition root used for the iter2700 checkpoint has:

- `num_rows=2899`
- `role_mode_mismatch_count=1193`
- `role_mode_mismatch_fraction=0.4115`

from:

`docs/world_model_task_rebinding/2026-06-13_receding_training_distribution_audit.json`

Live receding evaluation now queries the model using the physical current mode
and causal history. The old SFT rows often used sampled prefix roles whose text
role did not match the actual physical mode. That mismatch is aligned with the
observed failure: direct raw Cosmos action chunks can be active and full length,
but their direction/scale and resulting real-state DP continuability are not
reliable.

## Gate Rerun

The current safety gate was rerun after the hard comparison and code audit.

- Broad panel from current checkpoint is rejected:
  `docs/world_model_task_rebinding/2026-06-14_cosmos3_next_action_gate_launch_broad_panel_requirement_audit_rerun.json`
- Resume current condition SFT is rejected:
  `docs/world_model_task_rebinding/2026-06-14_cosmos3_next_action_gate_resume_current_requirement_audit_rerun.json`

Both rejections cite the incomplete requirement audit. The unresolved items are:

- `dp_handoff_available_but_not_proven_reliable`
- `method_effectiveness_against_pure_dp`

## Live Query Coverage Audit

A new read-only audit was added:

`scripts/world_model/audit_cosmos3_live_query_training_coverage.py`

It compares recorded live Cosmos query states against the old SFT condition
rows. It does not run simulator, video, inference, or training. Output:

- `docs/world_model_task_rebinding/2026-06-14_iter2700_live_query_training_coverage_audit.json`
- `docs/world_model_task_rebinding/2026-06-14_iter2700_live_query_training_coverage_audit.md`

On the current val panel plus hard-screen-2 pure-DP-failure panel:

- Training rows: `2899`
- Live Cosmos queries: `173`
- Training role/mode mismatches: `1193` (`0.4115`)
- Strictly undercovered live queries: `74/173` (`0.4277`)
- Undercovered roles: mostly `target_post_motion`, plus all `8` observed
  `peg_recovery` live queries

The strict local coverage rule requires a role/mode-consistent training row
near the live query in both prefix frame and peg-head-at-hole geometry. This is
not a method-success metric, but it is direct evidence that many real live
queries from the failed closed-loop panels were not locally represented by the
old clean physical-role distribution.

The clean/dense preflight path now consumes this audit. When
`RUN_LIVE_QUERY_COVERAGE_AUDIT=true`, the full WAM wrapper writes
`live_query_training_coverage_audit.json/.md`, and
`summarize_cosmos3_clean_dense_preflight.py` includes it in
`ready_for_overfit`. The full v7_733 clean/dense preflight wrapper defaults this
coverage gate on using the current val and hard-screen-2 panel summaries. The
two-source overfit2 preflight defaults it off, because overfit2 is a chain
sanity check and should not be required to cover the full failed live-query
distribution.

The full v7_733 clean/dense wrapper now refuses real execution if the live-query
coverage audit is disabled, unless
`ALLOW_SKIP_LIVE_QUERY_COVERAGE_DIAGNOSTIC=true` is set. That override is
diagnostic-only and must not unlock overfit/full SFT evidence. The wrapper also
checks that configured live-query summary paths exist before handing off to the
base preflight wrapper. If the diagnostic override is used, the wrapper now
passes a `diagnostic_not_ready_reason` through the base preflight summary; the
summary records `diagnostic_not_method_ready` as a failed check and
`ready_for_overfit=false`.

The clean/dense overfit SFT entry now has its own summary validator:
`scripts/world_model/check_cosmos3_clean_dense_preflight_summary_ready.py`.
It accepts only a matching summary with `ready_for_overfit=true`, zero
`failed_checks`, and no `diagnostic_not_ready_reason`. The next-action gate uses
the same stricter semantics, so a hand-edited or inconsistent summary cannot
unlock overfit SFT just by setting `ready_for_overfit=true`.

The old sampled-role SFT launch wrappers are also guarded now:

- `scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_fix1recipe_in_allocation.sh`
- `scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_overfit2_fix1recipe_in_allocation.sh`

Both refuse training by default and point to the clean/dense preflight path.
They can only be run with explicit legacy diagnostic overrides and must not be
used as active method repair evidence.

The refusal behavior is covered by
`scripts/world_model/selftest_cosmos3_legacy_sft_wrapper_guards.sh`: the full
legacy wrapper must exit `66`, the old overfit2 wrapper must exit `67`, and
their dry-run modes must report that legacy training would be refused.

## Next Aligned Step

Do not use iter2700 for more broad method evidence. The aligned repair is the
clean-role/dense-receding path:

1. Run the guarded clean/dense preflight inside a compute allocation after
   explicit approval.
2. Confirm role/mode mismatch is zero, full `301/300` rows are preserved, and
   late rebind coverage exists.
3. Run the approved two-sample overfit SFT from that condition root.
4. Only after overfit video and action chunks pass, scale to the full frozen
   v7 source.

If direct raw Cosmos actions remain unstable after that repair, the next method
repair should be a learned short-chunk executor or DP-prior policy conditioned
on Cosmos-predicted task state, still evaluated with real reobservation and
same-source pure-DP comparison.
