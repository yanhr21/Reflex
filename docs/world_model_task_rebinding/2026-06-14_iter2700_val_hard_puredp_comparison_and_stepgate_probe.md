# 2026-06-14 Iter2700 Val/Hard Pure-DP Comparison

## Objective

The user requested that, after val closed-loop completion, harder closed-loop
cases should be tried and compared against full pure DP to test whether the
Cosmos3 world model is genuinely useful.

This note records the current evidence boundary. The implementation contract is
full-episode: `300` executed actions, `301` observed frames, causal target-motion
detection, and explicit controller labels in the videos. The method claim still
requires success against a same-source pure-DP baseline.

## Artifacts

- SFT/eval root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`
- Checkpoint:
  `outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/checkpoints/iter_000002700`
- Objective gate:
  `docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_objective_gate.json`
- Video length audit:
  `docs/world_model_task_rebinding/2026-06-14_iter2100_vs_iter2700_video_length_audit.json`
- Hard action/rebind audit:
  `docs/world_model_task_rebinding/2026-06-14_iter2700_hard_screen2_action_rebind_analysis.json`

## Val Comparison

Cosmos closed-loop val panel:

`live_receding_full300_unified_iter2700_panel3_dynamic_20260613_alloc127559`

- Completed samples: `3`
- Success: `1/3`
- Per-sample:
  - `hole_late_constant`: fail, `DP_SCAN_TARGET=94`, `WM_ACTIVE=206`
  - `hole_late_reverse`: success, `DP_SCAN_TARGET=104`, `WM_ACTIVE=40`, `DP_HANDOFF=156`
  - `hole_late_fast_shift`: fail, `DP_SCAN_TARGET=132`, `WM_ACTIVE=32`, `DP_HANDOFF=136`

Same-source pure-DP baseline:

`dp_full_episode_baseline_iter2700_panel3_dynamic_20260614_alloc127559`

- Completed samples: `3`
- Success: `3/3`
- Every sample ran as `PURE_DP=300`

Interpretation: the current Cosmos3 closed loop does not improve the val panel.
It degrades two cases that pure DP solves.

## Hard Comparison

Hard-screen-2 pure-DP baseline:

`hard_case_screen2_20260614/pure_dp_hard15`

- Completed samples: `15`
- Pure DP success: `9/15`
- Pure DP failures: indices `4,9,10,11,12,13`

Cosmos closed loop on only those six pure-DP failures:

`hard_case_screen2_20260614/cosmos_iter2700_puredp_fail_4_9_10_11_12_13`

- Completed samples: `6`
- Cosmos success: `1/6`
- Positive sample:
  - index `4`, `hole_late_fast_shift`: Cosmos success with
    `DP_SCAN_TARGET=86`, `WM_ACTIVE=56`, `DP_HANDOFF=158`
- Failures:
  - indices `9,10`: `hole_late_sine`
  - index `11`: `hole_late_continuous_insert`
  - index `12`: `hole_late_move_stop`
  - index `13`: `hole_late_fast_shift`

This gives one hard positive comparison where pure DP fails and Cosmos succeeds,
but it is not broad method evidence. On the hard screen the current checkpoint
rescues only `1/6` pure-DP failures.

## Step-Level Handoff Probe

A diagnostic probe was launched inside held allocation `127559`:

`hard_case_screen2_20260614/cosmos_iter2700_stepgate_probe_idx12_20260614_alloc127559`

The probe used `--cosmos-step-handoff-gate`, which checks the real-state
continuability gate after every executed Cosmos step and can stop a Cosmos chunk
early so the next iteration immediately tries DP handoff. This preserves the
same authority boundary: real simulator state and final task success remain the
evidence, not generated sidecars.

The allocation hit its 12-hour time limit before the probe completed:

- No `live_receding_panel_summary.json`
- No full `live_receding_loop_summary.json`
- Partial summary exists:
  `sample_12_hole_late_move_stop/live_receding_loop_partial_summary.json`
- Partial endpoint: frame `188` / `189` observed frames
- Partial eval: `success=false`,
  `peg_head_pos_at_hole=[-0.0962894559, -0.0088737756, -0.0127753913]`

Diagnostic detail:

- Cosmos reached a step-level C_pi pass at iteration `5`, around prefix `127`,
  with peg-head-at-hole approximately
  `[-0.1251378357, -0.0066177398, -0.0010708719]`.
- The next iteration executed an 8-step frozen-DP handoff from frame `132` to
  `140`; x improved but lateral alignment drifted, ending around
  `[-0.0948580, -0.0203063, -0.0029298]`, so C_pi failed again.
- Later Cosmos chunks did not recover the sample by the time the allocation was
  revoked; z error grew to about `-0.0128` by frame `188`.

This partial probe is not complete closed-loop evidence. It is useful failure
localization: for index `12`, the remaining issue is not only delayed handoff.
The direct Cosmos action/rebind path reaches only a fragile boundary state, and
the frozen DP can drift out of continuability instead of completing insertion.

## Current Conclusion

The full-episode closed-loop implementation and hard/pure-DP comparison are in
place. Current method effectiveness is still false:

- Val: Cosmos `1/3`, pure DP `3/3`
- Hard-screen-2 pure-DP failures: Cosmos `1/6`
- Step-level handoff probe: incomplete, partial failure through frame `188`

The objective gate was hardened after this comparison. Moving-target samples
must now prove causal detector provenance with
`prefix_selection.mode=target_motion_onset`, concrete detector/streak frame
indexes, and at least `8` WM-active frames. Static/no-motion evidence must
record the same detector never triggering and no detected frame. The official
gate output was regenerated under
`docs/world_model_task_rebinding/2026-06-14_iter2700_closed_loop_objective_gate.json`
and still reports `implementation_contract_ok=true` and
`method_effectiveness_ok=false`.
The regenerated gate also checks the structured overlay metadata: every sample
must have `301`-frame controller and annotated timelines with matching
controller counts; moving samples must show WM-active frames in the annotated
timeline, and static/no-motion samples must show zero annotated WM-active and
zero target-motion-detected frames.

The live-receding Slurm wrappers were also hardened at launch time. The
corrected method path now requires auto role inference, causal target-motion
onset detection, live frozen-DP pretrigger control, and enabled Cosmos
inference. Manifest/manual prefixes, source-restored pretriggers, explicit
role overrides, or no-Cosmos runs require
`ALLOW_LIVE_RECEDING_DIAGNOSTIC=true` and remain non-method diagnostics.

A local self-test now covers those objective-gate semantics without Slurm:
`scripts/world_model/selftest_cosmos3_closed_loop_objective_gate.py`. It
constructs synthetic summaries and verifies that valid causal moving/static
evidence passes, while manual prefixes, missing detector provenance, token
one-frame WM use, static WM activation, and malformed overlay-timeline evidence
fail.

The legacy one-shot closed-loop wrappers are now guarded as diagnostics too.
`scripts/slurm/run_cosmos3_closed_loop_panel_in_allocation.sh` and
`scripts/slurm/run_cosmos3_receding_closed_loop_in_allocation.sh` refuse by
default unless `ALLOW_OLD_ONESHOT_CLOSED_LOOP_DIAGNOSTIC=true` is explicitly
set, and point users to the corrected full-300 live-receding wrappers.
The wrapper refusal behavior is covered by
`scripts/world_model/selftest_cosmos3_closed_loop_wrapper_guards.sh`, which
checks both new live-receding diagnostic-mode guards and old one-shot wrapper
guards without launching Slurm work.

The next aligned method work should not be more evidence collection from the
same checkpoint. It should repair the training/interface distribution problem
already identified in the clean-role/dense-receding plan, or move to a learned
short-chunk executor / DP-prior policy conditioned on Cosmos-predicted task
state if direct Cosmos action chunks remain unstable.
