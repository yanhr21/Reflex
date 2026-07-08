# ManiSkill TODO Index

Date: 2026-07-06

This file is only an index. Concrete tasks live in the phase folders.

## Required Order

1. `TODO/01/dp_static.md`
2. `TODO/02/cosmos_imagination.md`
3. `TODO/03/oracle.md`
4. `TODO/04/integration.md`
5. `TODO/05/live.md`

## Current Rule

Do Phase 01 and Phase 02 first. Phase 03 Oracle may now run before the bridge,
but only as an explicitly labeled upper-bound diagnostic after DP static and
RGB Cosmos evidence exist. Phase 04 must then convert Cosmos and Oracle
diagnostic evidence into controller-facing task-state, candidate-score, and
trust-gate signals before Phase 05 live claims.

## Current Phase 03 Status

- Corrected full-pipeline Oracle reruns on `2026-07-04` executed DP prefix,
  premotion RGB Cosmos, postmotion Cosmos action control, and DP/manual
  physical finisher variants without the old visible snap / peg teleport.
- One older synthetic DP-finisher diagnostic reached simulator success after 4
  premotion Cosmos reports, 4 postmotion Cosmos reports, 26 Cosmos dynamic
  actions, and a DP finisher starting at step 111. Discontinuity audit reports
  `snap_detected=false`; final `peg_head_l2=0.01369317842884192`.
- This run used runner-specified synthetic `TARGET_MOTION_Y=0.0125`, not an
  approved `fix3_733` validation/canonical H5 key. It is therefore diagnostic
  only and must not be counted as validation-set success. Its
  `visual_review_replay_v2/visual_review_verdict.json` records
  `single_case_success_confirmed=false` and
  `invalid_as_validation_set_success=true`.
- The old visible-snap Oracle video remains invalid/archive evidence and must
  not be cited as progress.
- Current valid Phase 03 Oracle validation-key single-case success count is
  `2` under the stricter active-insertion visual standard:
  `h5_continuous_insert/try04` and `h5_continuous_insert/try11`.
- Current completion gate still reports
  `phase03_oracle_overall_complete=false`; missing coverage is
  `forward_backward_target_motion,left_right_target_motion,peg_or_wooden_stick_disturbance,multiple_approved_fix3_733_keys`,
  and the next required group is `forward_backward_target_motion`.
- The prepared next run is approved key
  `hole_late_reverse_seed1040038_idx0004` as `h5_reverse/try21`, guarded by
  `scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh` /
  `scripts/slurm/phase03_forward_backward_probe.sh`. As of
  `2026-07-06T17:10:47+08:00`, no `p03_fb21` job is running or pending, no
  `render_probe/fwdback21` or `h5_reverse/try21` artifacts exist, and the
  latest 4 CPU / 32G / 1 GPU / 1.5h Slurm test-only estimate, after excluding
  known render-bad nodes, is 2026-07-08 17:13:05 on `server44`. Other visible
  GPU partitions were not usable for this account/request.
- The latest guarded immediate attempt for that run, Slurm allocation `168276`
  (`p03_fb21`) at 2026-07-06 16:24 CST, stayed priority-pending and was
  canceled before node assignment (`None assigned`, elapsed `00:00:00`). It
  produced no render probe, full run, or log artifact.
- The forward/backward tmux launcher now has a default Slurm `--test-only`
  precheck. On 2026-07-06 16:27 CST, precheck job `168283` estimated
  2026-07-08 15:42:33 on `server27`, so the launcher refused with
  `refusing_far_scheduler_test_only=true` / exit `43` before creating any tmux
  session, allocation, run artifact, or log.
- The same launcher now also excludes known render-bad nodes by default for
  this render-bearing attempt:
  `server02,server21,server27,server28,server30,server39,server53,server57`. A
  2026-07-06 16:30 CST precheck with that exclusion list produced test-only
  job `168292`, estimated 2026-07-08 16:45:05 on `server44`, and refused with
  exit `43` before creating tmux, Slurm allocation, run artifact, or log.
- `scripts/world_model/phase03_next_coverage_status.sh` now uses the same
  exclude list for `INCLUDE_SCHEDULER_TEST=true`, so status output matches the
  guarded launcher. Its 2026-07-06 16:32 CST test-only job `168305` estimated
  2026-07-08 17:13:05 on `server44`.
- The status helper and launcher now also print machine-readable scheduler
  fields: `scheduler_test_job`, `scheduler_estimated_start`, and
  `scheduler_estimated_node`. A 2026-07-06 16:34 CST launcher precheck reported
  `168309`, `2026-07-08T17:13:05`, and `server44`, then refused with exit `43`
  before creating tmux, allocation, artifacts, or logs.
- `server21` was added to the default exclusion list after the status helper
  estimated it and existing docs showed a prior `vk::DeviceLostError` render
  failure there. The latest 2026-07-06 16:38 CST status-helper precheck
  reported `scheduler_test_job=168314`,
  `scheduler_estimated_start=2026-07-08T17:14:05`,
  `scheduler_estimated_node=server44`, and
  `scheduler_within_delay_threshold=false`.
- `scripts/world_model/phase03_next_coverage_status.sh` now accepts
  `PARTITION`, `JOB_NAME`, `TIME_LIMIT`, `CPUS_PER_TASK`, and `MEMORY` so
  scheduler status can match launcher resource settings instead of using
  hard-coded defaults. A 2026-07-06 16:40 CST default-resource check reported
  `scheduler_test_job=168319`, `2026-07-08T17:13:05`, `server44`; a reduced
  2 CPU / 12G / 45min check reported `scheduler_test_job=168320` with the same
  estimated start and node.
- `scripts/world_model/phase03_forward_backward_readiness.sh` now verifies that
  the launcher and status helper share the same default bad-node exclude list.
  It also rejects caller overrides that drift from that default; a negative
  `EXCLUDE_NODES=server44` check exits `51` with
  `reason=requested_exclude_nodes_do_not_match_default`. With scheduler test
  enabled, the readiness helper now also uses the shared exclude list; its
  2026-07-06 16:43 CST test-only job `168321` estimated
  2026-07-08 17:13:05 on `server44`.
- The readiness scheduler-test path now also prints the same parsed scheduler
  fields as the status helper, including `scheduler_delay_seconds` and
  `scheduler_within_delay_threshold`. Its 2026-07-06 16:45 CST test-only job
  `168325` estimated 2026-07-08 17:16:05 on `server44` and reported
  `scheduler_within_delay_threshold=false`.
- `scripts/world_model/phase03_next_coverage_status.sh` now combines the
  prepared-artifact check, same-name Slurm job check, and scheduler threshold
  into `phase03_forward_backward_launch_allowed` plus
  `phase03_forward_backward_launch_block_reasons`. With scheduler test enabled
  on 2026-07-06 16:47 CST, it reported no artifacts and no same-name job, but
  the latest test-only job `168330` estimated 2026-07-08 17:16:05 on
  `server44`, so
  `phase03_forward_backward_launch_allowed=false` and the block reason is
  `scheduler_delay_exceeds_threshold`. Without scheduler test, launch
  permission is `unknown`.
- `scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh` now consumes
  that combined status gate before launching. A
  2026-07-06 16:49 CST launcher invocation ran readiness and the status helper,
  saw test-only job `168332` estimated 2026-07-08 17:12:05 on `server44`, and
  refused with `refusing_status_launch_gate=true` / exit `44` before creating
  tmux, Slurm allocation, run artifact, or log.
- `AGENTS.md` now makes this launch gate a standing rule for Phase 03
  forward/backward launchers. If the status helper reports
  `phase03_forward_backward_launch_allowed=false` or `unknown`, do not create
  tmux / `salloc`; report the block reasons instead. Latest 2026-07-06 16:51
  CST status: `scheduler_test_job=168334`, estimated 2026-07-08 17:10:05 on
  `server44`, launch allowed false, block reason
  `scheduler_delay_exceeds_threshold`.
- The launch gate has been factored into executable helper
  `scripts/world_model/require_phase03_forward_backward_launch_allowed.sh`, and
  the tmux launcher now calls that helper. A 2026-07-06 16:54 CST launcher
  check reported test-only job `168338`, estimated 2026-07-08 17:10:05 on
  `server44`, and exited `44` with
  `phase03_forward_backward_launch_required_ok=false` before creating tmux,
  Slurm allocation, run artifact, or log.
- `phase03_static_protocol_scan.sh` and
  `phase03_forward_backward_readiness.sh` now include the launch-gate helper in
  their checked file lists, and the static scan verifies that the tmux launcher
  calls the helper. Static scan now reports `checked_files=9`. Latest
  2026-07-06 16:56 CST status refresh:
  `scheduler_test_job=168341`, estimated 2026-07-08 18:13:56 on `server44`,
  launch still blocked by `scheduler_delay_exceeds_threshold`.
- The launcher no longer exposes a `STATUS_LAUNCH_GATE=false` bypass or the old
  scheduler-only fallback path; it unconditionally calls
  `require_phase03_forward_backward_launch_allowed.sh`. A 2026-07-06 16:58 CST
  launcher check reported test-only job `168345`, estimated
  2026-07-08 20:17:10 on `server63`, and exited `44` with
  `phase03_forward_backward_launch_required_ok=false` before creating tmux,
  Slurm allocation, run artifact, or log.
- The static protocol scan now also verifies that
  `phase03_forward_backward_next.sh` still calls `phase03_h5_source.sh` and
  still contains the coverage guards that reject row-offset diagnostics,
  future-label dynamic controllers, future-label teacher paths, and
  direction-guard diagnostics. Static scan and readiness pass with
  `checked_files=9`. Latest 2026-07-06 17:01 CST status:
  `scheduler_test_job=168349`, estimated 2026-07-08 20:23:10 on `server63`,
  launch still blocked by `scheduler_delay_exceeds_threshold`.
- The static scan now rejects reintroducing `STATUS_LAUNCH_GATE` or
  `SCHEDULER_TEST_ONLY_GUARD` text in the tmux launcher, preventing the old
  launch-gate bypass path from returning. It passes with `checked_files=9`.
  Latest 2026-07-06 17:03 CST status: `scheduler_test_job=168351`, estimated
  2026-07-08 20:23:10 on `server63`, launch still blocked by
  `scheduler_delay_exceeds_threshold`.
- The active forward/backward tmux, probe, and full-run launchers now reject
  `SKIP_PHASE03_NEXT_COVERAGE_GUARD=true` with exit `52` rather than using it
  to skip readiness / coverage / launch gates. The static scan rejects the old
  `SKIP_PHASE03_NEXT_COVERAGE_GUARD != true` bypass pattern. Negative checks
  for all three entry points refused before any resource allocation or artifact
  creation.
- `AGENTS.md` now records this as a standing rule: active Phase 03
  forward/backward launchers must reject
  `SKIP_PHASE03_NEXT_COVERAGE_GUARD=true` before readiness, coverage, launch,
  tmux, `salloc`, render, or artifact creation. Latest 2026-07-06 17:08 CST
  status: `scheduler_test_job=168357`, estimated 2026-07-08 20:28:10 on
  `server63`, launch still blocked by `scheduler_delay_exceeds_threshold`.
- The launch-gate helper override path was checked without launching resources:
  with `ALLOW_FAR_SCHEDULER_TEST_ONLY=true`, helper-only test job `168360`
  returned `phase03_forward_backward_launch_required_ok=true` only because the
  sole block reason was `scheduler_delay_exceeds_threshold`. This created no
  tmux, `salloc`, render probe, run artifact, or log. The static scan now
  verifies the override remains an exact single-reason match and cannot broaden
  to artifacts, duplicate jobs, stale coverage, or other guards.
- 2026-07-06 17:15 CST login-node guard fix: the forward/backward tmux launcher
  no longer runs `phase03_forward_backward_readiness.sh` before `salloc`.
  Readiness stays inside the Slurm allocation through
  `phase03_forward_backward_probe.sh`; the probe and full-run entry points now
  refuse direct execution when `SLURM_JOB_ID` is missing. The static scan now
  covers the tmux launcher, rejects a reintroduced login-node readiness call,
  and requires the forward/backward login-node refusal guards; its current
  checked-file list is now 10 files. This was a text/script guard update only
  and the static preflight was not executed on the login node. This created no
  tmux session, Slurm allocation, render probe, run artifact, or log. Targeted
  checks found no `h5_reverse/try21` / `render_probe/fwdback21` artifacts and
  no queued `p03_fb21` job.
- Latest status-only launch gate: `phase03_oracle_overall_complete=false`;
  missing coverage remains
  `forward_backward_target_motion,left_right_target_motion,peg_or_wooden_stick_disturbance,multiple_approved_fix3_733_keys`.
  Next required group is still `forward_backward_target_motion`; prepared
  artifact count and same-name Slurm job count are `0`. The launch gate now
  also requires the scheduler estimate to fit the launcher's
  `salloc --immediate` window. Latest status-only gate:
  `scheduler_test_job=168374`, estimated `2026-07-08T20:49:39` on `server44`,
  `scheduler_delay_seconds=185220`,
  `scheduler_within_immediate_window=false`, and
  `phase03_forward_backward_launch_allowed=false` with
  `scheduler_delay_exceeds_immediate_window,scheduler_delay_exceeds_threshold`.
  Scheduler-only variants lowering CPU/memory did not improve the estimate;
  alternate partitions were not valid for this account, and no-exclude GPU
  would target excluded bad-render node `server39`. Do not launch
  `h5_reverse/try21` until the gate allows it.
- Follow-up scheduler-only walltime variants `00:20:00`, `00:30:00`,
  `00:45:00`, `01:00:00`, and `01:30:00` all estimated
  `2026-07-08T22:40:15` on `server63`; shortening walltime does not currently
  make the run immediate. The status helper now validates `IMMEDIATE_SECONDS`
  and `MAX_TEST_ONLY_DELAY_MINUTES` as nonnegative integers; `IMMEDIATE_SECONDS=abc`
  exits with `reason=invalid_immediate_seconds`. Latest status-only gate:
  `scheduler_test_job=168387`, estimated `2026-07-08T22:40:15` on `server63`,
  `scheduler_delay_seconds=191681`, still blocked by
  `scheduler_delay_exceeds_immediate_window,scheduler_delay_exceeds_threshold`.
  `ALLOW_FAR_SCHEDULER_TEST_ONLY=true` now exits `44` and reports
  `scheduler_delay_override_refused_by_immediate_window=true` when the active
  launcher would fail `salloc --immediate`.
- The status helper now caches exact-parameter `srun --test-only` output under
  `/tmp` for `SCHEDULER_TEST_CACHE_SECONDS=120` by default, and validates that
  cache TTL as a nonnegative integer. A two-query status-only check produced
  `scheduler_test_job=168392`, estimated `2026-07-08T22:40:15` on `server63`;
  the first query had `scheduler_test_cache_hit=false`, the immediate second
  query had `scheduler_test_cache_hit=true`, and launch stayed blocked by
  `scheduler_delay_exceeds_immediate_window,scheduler_delay_exceeds_threshold`.
- Safe retry runbook for `h5_reverse/try21`: first check `squeue -u "$USER"`
  for an existing reusable Slurm allocation or same-name `p03_fb21` job. Then
  run a no-cache status-only gate with
  `SCHEDULER_TEST_CACHE_SECONDS=0 INCLUDE_SCHEDULER_TEST=true
  scripts/world_model/phase03_next_coverage_status.sh`. Launch only if it
  reports `phase03_forward_backward_launch_allowed=true`; otherwise record
  `phase03_forward_backward_launch_block_reasons` and do not create tmux,
  `salloc`, render, or artifacts. Latest no-cache check reported
  `scheduler_test_job=168393`, estimated `2026-07-08T22:33:15` on `server63`,
  `scheduler_delay_seconds=191018`, and the same immediate-window /
  delay-threshold block. There are currently no Slurm jobs for this user and no
  `h5_reverse/try21` / `render_probe/fwdback21` artifacts.
- Active launch decisions are now no-cache by default:
  `require_phase03_forward_backward_launch_allowed.sh` defaults
  `SCHEDULER_TEST_CACHE_SECONDS=0`, and the tmux launcher passes
  `SCHEDULER_TEST_CACHE_SECONDS=0` explicitly. The 120-second scheduler-test
  cache is only for human status inspection; it must not allow tmux / `salloc`
  creation from stale scheduler state.
- The active launch-required helper now rejects any nonzero
  `SCHEDULER_TEST_CACHE_SECONDS` before calling the status helper. The checked
  negative path `SCHEDULER_TEST_CACHE_SECONDS=120` exits `45` with
  `refusing_active_launch_scheduler_cache=true`, so callers cannot override the
  no-cache launch rule.
- Mandatory bad-render-node exclusions are now enforced before scheduler test:
  active forward/backward launch requires
  `server02,server21,server27,server28,server30,server39,server53,server57` in
  `EXCLUDE_NODES`. The checked negative path removed `server39` and exited `46`
  with `refusing_missing_mandatory_exclude_node=true`, before any scheduler
  query, tmux, `salloc`, render, log, or run artifact.
- The status helper now skips `srun --test-only` when pre-scheduler blockers
  already prove launch is impossible. Checked negative path: with `server39`
  removed from `EXCLUDE_NODES`, it reported
  `pre_scheduler_block_reasons=mandatory_exclude_nodes_missing`,
  `scheduler_test_skipped=true`, and
  `phase03_forward_backward_launch_block_reasons=mandatory_exclude_nodes_missing`
  without a scheduler test job line.
- Same-name Slurm job status is now fail-closed. If `squeue` is missing or
  fails, the status helper blocks with `same_name_slurm_job_query_failed`
  rather than treating the job count as zero. Current status:
  `same_name_slurm_job_query_ok=true`, `same_name_slurm_job_count=0`, no target
  artifacts, scheduler test job `168416`, still blocked by
  `scheduler_delay_exceeds_immediate_window,scheduler_delay_exceeds_threshold`.
- Pre-scheduler blockers now stay `false` even when scheduler testing is not
  requested. Checked cases: with `INCLUDE_SCHEDULER_TEST=false` and `server39`
  removed from `EXCLUDE_NODES`, launch allowed is `false` with
  `mandatory_exclude_nodes_missing`; with no preblock and no scheduler test,
  launch allowed remains `unknown` with `scheduler_test_not_requested`.
- The same-name job query failure path no longer prints `squeue` error text as
  job rows or counts it as `same_name_slurm_job_count`; failures are expressed
  by `same_name_slurm_job_query_ok=false` plus
  `same_name_slurm_job_query_failed`. Current no-scheduler status:
  `same_name_slurm_job_query_ok=true`, `same_name_slurm_job_count=0`,
  `pre_scheduler_block_reasons=none`, launch `unknown` only because
  `scheduler_test_not_requested`.
- Latest no-cache launch gate refresh: no `p03_fb21` job, no `p03_fwdback21`
  tmux session, and no `h5_reverse/try21` / `render_probe/fwdback21` artifacts.
  The gate still reports `phase03_oracle_overall_complete=false`, next coverage
  `forward_backward_target_motion`, `mandatory_exclude_ok=true`,
  `pre_scheduler_block_reasons=none`, scheduler test job `168417`, estimated
  `2026-07-08T21:22:16` on `server44`, `scheduler_delay_seconds=185999`, and
  `phase03_forward_backward_launch_allowed=false` with
  `scheduler_delay_exceeds_immediate_window,scheduler_delay_exceeds_threshold`.
  No tmux, `salloc`, render, or rollout was launched.
- Fresh no-cache refresh: still no `p03_fb21` job, no `p03_fwdback21` tmux
  session, and no `h5_reverse/try21` / `render_probe/fwdback21` artifacts.
  Completion gate reads successfully with exit `3`, meaning incomplete; next
  coverage remains `forward_backward_target_motion`. Scheduler test job
  `168422` estimated `2026-07-08T21:20:16` on `server44`,
  `scheduler_delay_seconds=185482`, so
  `phase03_forward_backward_launch_allowed=false` with
  `scheduler_delay_exceeds_immediate_window,scheduler_delay_exceeds_threshold`.
  No tmux, `salloc`, render, or rollout was launched.
- The status helper now does a cheap `sinfo` idle-node precheck for immediate
  launches before `srun --test-only`. Current status:
  `partition_idle_query_ok=true`, `partition_idle_node_count=0`,
  `partition_idle_immediate_ok=false`, so
  `pre_scheduler_block_reasons=partition_idle_nodes_zero`,
  `scheduler_test_skipped=true`, and launch stays false without a new scheduler
  test job. No tmux, `salloc`, render, or rollout was launched.
- Latest fresh gate check is unchanged: no `p03_fb21` job, no `p03_fwdback21`
  tmux session, no `h5_reverse/try21` / `render_probe/fwdback21` artifacts,
  completion gate read ok with exit `3`, next coverage still
  `forward_backward_target_motion`, and `partition_idle_node_count=0`.
  Launch remains false with block reason `partition_idle_nodes_zero`. No
  scheduler test job, tmux, `salloc`, render, or rollout was launched.
- The active launch-required helper was checked end to end. It exited `44` with
  `phase03_forward_backward_launch_required_ok=false` and
  `phase03_forward_backward_launch_block_reasons=partition_idle_nodes_zero`,
  before creating `p03_fwdback21`, `p03_fb21`, `h5_reverse/try21`, or
  `render_probe/fwdback21`.
- Completion gate exit semantics are now handled correctly by the status
  helper: `check_phase03_oracle_completion.sh` exits `3` when the check ran
  successfully but Phase 03 Oracle is still incomplete. The status helper now
  accepts exit `0` or `3` only when `phase03_oracle_completion_check_ok=true`,
  reports `completion_gate_read_ok=true`, and uses `completion_gate_failed`
  only for an actual unreadable / failed completion gate. `AGENTS.md` now
  records the same exit-code contract so future launch / completion reports do
  not treat exit `3` as a broken checker.
- User review on 2026-07-06: do not keep searching within
  `h5_continuous_insert`; those samples can be treated as single-case
  references, but the next useful work is other types, starting with
  `forward_backward_target_motion` via `h5_reverse/try21`.
- This is now enforced by launcher guard: continuous-insert source keys /
  groups are refused while the completion gate says forward/backward is next,
  unless explicitly overridden as diagnostic-only.
- The generic H5 launcher now also refuses other non-next case families while
  forward/backward is next; `h5_fastshift` was checked and exits `49` unless
  explicitly overridden as diagnostic-only.
- The forward/backward readiness and full-run launchers now refuse
  `MAX_PREMOTION_COSMOS_PREDICTIONS < 4` and
  `MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER < 4`, closing the path where a
  run could skip repeated premotion RGB Cosmos evidence or the required Cosmos
  dynamic-control stage before the finisher.
- The completion gate now also requires structured full-sequence video review:
  `full_sequence_video_reviewed=true`,
  `video_covers_cosmos_control_and_finisher=true`, and
  `video_covers_final_insertion_or_physical_failure=true`. This prevents a
  boundary-only or target-motion-start video from counting merely because an
  MP4 file exists.
- The completion gate now also requires first simulator success to occur in a
  DP/manual physical finisher row after the finisher starts; success during the
  Cosmos dynamic-control or target-motion stage is not strict single-case
  success.
- The completion gate now also requires action-trace stage order:
  first target-motion increment before first `cosmos_dynamic_control`, first
  Cosmos dynamic-control before first DP/manual physical finisher, and no
  `dp_static_prefix` rows after target motion starts.
- Active Phase 03 artifact hygiene has been cleaned: superseded
  `action_diag/try02` through `try13`, their logs, and diagnostic
  `h5_continuous_insert/try09` plus its log were moved under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/`, leaving active run/log
  trees with only accepted `h5_continuous_insert/try04` and `try11`.
- Fresh approved continuous-insert key `h5_continuous_insert/try14`
  (`hole_late_continuous_insert_seed12241047_idx5594`) is rejected as strict
  success despite `simulator_success_metric=true`. It used real
  `cosmos3_policy` dynamic actions, produced 4 premotion and 5 postmotion
  Cosmos reports plus 33 Cosmos dynamic actions, and had no snap, but the
  metric flipped during target motion before any finisher started. Visual
  review shows target/hole motion continuing onto the held peg; verdict:
  `target_assisted_insertion_detected=true`,
  `strict_success_confirmed=false`. The accepted success count remains `2`.
- Fresh approved move-stop key `h5_move_stop/try16`
  (`hole_late_move_stop_seed17280909_idx8226`) completed the full three-stage
  protocol rather than stopping at the boundary: 4 premotion Cosmos reports,
  3 postmotion Cosmos reports, 23 real `cosmos3_policy` dynamic actions, then
  530 original-DP finisher rows after target motion completed. It is negative
  evidence, not success: `simulator_success_metric=false`,
  `visual_full_insertion_confirmed=false`, best finisher `peg_head_l2` about
  `0.0954m`, final about `0.1321m`, and `snap_detected=false`. The accepted
  success count remains `2`.
- Follow-up upper-bound teacher-dynamic diagnostics for that same move-stop key
  did not reach rollout: `render_probe/movestop17` failed on `server02`
  (`render_gym_start` timeout), and `render_probe/movestop17b` failed on
  `server21` with `vk::DeviceLostError`. No `h5_move_stop/try17*` full-run
  summary/action trace/video exists. These are infrastructure failures only.
- Read-only `action_diag/try12` compared archived `h5_move_stop/try16` Cosmos
  dynamic actions against the matching source-H5 teacher rows. It ran inside
  Slurm job `167963` / `server02` and did not execute teacher actions. Result:
  23 valid dynamic rows, 7D RMSE about `0.1352`, xyz RMSE about `0.1264`,
  dynamic-stage L2 only improved by about `0.0045m`, and mean xyz sign
  agreement was about `[0.13, 0.48, 0.30]`. This points to a Cosmos dynamic
  action-interface / action-selection mismatch on this move-stop key, not a
  pure DP-finisher problem.
- The next move-stop step must not be a blind `source_motion_sign` guard
  rollout. Text-level inspection of `action_diag/try12` found that 16 of 23
  Cosmos dynamic rows have negative logged target-y motion while the approved
  H5 teacher y action is positive; 14 rows have both positive Cosmos y and
  positive teacher y under negative target-y motion. The existing guard would
  clip or rectify many of those rows in the wrong direction.
- A login-node text-only calculation from the existing `try12` JSON quantified
  that risk without running project code: executed xyz RMSE is about `0.1264`,
  source-motion `clip_opposite` would worsen it to about `0.1389`, and
  `rectify_opposite` would worsen it to about `0.1565`. Per-axis
  future-label gain headroom is small: xyz gains about
  `[1.1603, 2.0462, 0.5331]` reduce xyz RMSE only to about `0.1220`. This is
  not enough evidence to launch another full Oracle rollout.
- The full-pipeline Slurm wrapper now refuses the known-bad
  `COSMOS_ACTION_DIRECTION_GUARD=source_motion_sign` setting for
  `h5_move_stop` / `hole_late_move_stop_*` by default, before creating new run
  artifacts. Bypass requires
  `ALLOW_UNTRUSTED_MOVE_STOP_SOURCE_MOTION_GUARD=true` and must stay
  diagnostic-only.
- `scripts/world_model/analyze_phase03_oracle_action_interface.py` now has a
  read-only future-label adapter-candidate diagnostic that can rank raw
  Cosmos, executed trace, source-motion-sign guard simulations, and simple
  per-axis gain headroom against the source-H5 teacher. It also now sweeps
  nearby source-H5 teacher temporal offsets so action mismatch is not confused
  with a teacher-index alignment bug. This is diagnostic only and cannot count
  as method or Oracle success. `action_diag/try13` was prepared for archived
  `h5_move_stop/try16` and completed inside Slurm job `167982` / `server02`.
  It wrote `action_interface_diagnostic.json`, `manifest.txt`, and
  `classification.txt`; it did not execute a rollout or teacher actions.
- `action_diag/try13` found a concrete temporal alignment signal: matching
  executed Cosmos dynamic actions to source-H5 teacher rows at offset `-9`
  improves xyz RMSE from about `0.1264` to about `0.0671`, 7D RMSE from about
  `0.1352` to about `0.0773`, and xyz sign agreement from about
  `[0.13, 0.48, 0.30]` to about `[0.65, 0.83, 0.48]`. The next move-stop work
  should inspect / fix dynamic action index alignment between the live trace,
  Cosmos action chunk extraction, and source-H5 teacher rows before any new
  full rollout.
- `COSMOS_ACTION_ROW_OFFSET` / `--cosmos-action-row-offset` is now wired into
  the Cosmos action extractor and Phase 03 full-pipeline wrapper. Default `0`
  preserves old behavior. The concrete next diagnostic full-rollout candidate
  for the same move-stop key is `COSMOS_ACTION_ROW_OFFSET=9`, because the
  `try13` offset `-9` means extracted row `t` best matches teacher row `t-9`.
  Treat any nonzero row-offset run as diagnostic-only until it generalizes
  beyond this one future-label analysis.
  Next short command inside a tmux-held Slurm `srun` step is
  `scripts/slurm/phase03_move_stop_rowoffset_probe.sh`; it runs render canary
  first and then launches `phase03_move_stop_rowoffset.sh` only on a good GPU.
  The full-run launcher sets the approved source key, `RUN_NAME=try17`,
  `COSMOS_ACTION_ROW_OFFSET=9`, no source-motion guard, strict
  target-motion-complete finisher gate, and records
  `COSMOS_ACTION_ROW_OFFSET_SOURCE=action_diag_try13_best_teacher_temporal_offset_minus9`.
  The launcher refuses non-`hole_late_move_stop_*` keys so this future-label
  offset is not silently reused on other case groups.
  The Phase 03 artifact audit now requires
  `--allow-diagnostic-action-row-offset` for any nonzero row-offset run and
  then checks that `validation_key_success_allowed=false` plus a documented
  `action_row_offset_source` are present. It also now checks the dynamic-stage
  action trace row by row: every `cosmos_dynamic_control` row in a nonzero
  row-offset diagnostic must record the offset, offset source, chunk start/end,
  raw chunk start, and actual predicted action row index, and the predicted
  row index must match `chunk_start + cosmos_action_index`. The full-pipeline
  wrapper now passes this audit override automatically when
  `COSMOS_ACTION_ROW_OFFSET != 0`.
- A render-probe-gated launch attempt for that command used Slurm allocation
  `167985` (`p03_mv17`) but remained pending with reason `Priority` and was
  canceled before compute. No `render_probe/move17`, no `h5_move_stop/try17`,
  no render canary, and no Oracle rollout artifacts exist from that attempt.
- `scripts/slurm/launch_phase03_move_stop_rowoffset_probe_tmux.sh` now wraps
  that probe in a tmux-held `salloc --immediate` launcher and cancels same-name
  pending leftovers. A smoke launch with `IMMEDIATE_SECONDS=5` produced Slurm
  allocation `168000` (`p03_mv17`) but it was canceled before node assignment;
  no Oracle artifacts were created.
- A follow-up immediate launch on 2026-07-06 13:30 CST produced Slurm
  allocation `168016` (`p03_mv17`), stayed pending with reason `Priority`, and
  was canceled before node assignment (`None assigned`, elapsed `00:00:00`).
  No `render_probe/move17`, `h5_move_stop/try17`, or log artifacts were
  created. The latest `srun --test-only` for the same 4 CPU / 32G / 1 GPU /
  1.5h request predicted start at 2026-07-08 04:08:15 on `server39`; shorter
  CPU/memory/walltime probes and node-specific probes were no better, so do
  not leave a pending Oracle job unattended.
- A 2026-07-06 13:36 CST retry of the same immediate launcher produced Slurm
  allocation `168043` (`p03_mv17`), again stayed pending with reason
  `Priority`, and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/move17`, `h5_move_stop/try17`, or log artifacts
  were created. The latest test-only estimate for the same request is now
  2026-07-08 07:13:53 on `server30`; no new pending Oracle job was launched.
- Continuous-insert retry `h5_continuous_insert/try06`, Slurm job `166874` /
  `server57`, used approved key
  `hole_late_continuous_insert_seed10241574_idx5018`. It reached simulator
  metric true with 4 premotion Cosmos reports, 1 postmotion Cosmos report,
  4 Cosmos-derived dynamic actions, original DP finisher, final
  `peg_head_l2` about `0.01298`, `artifact_audit.ok=true`, and no snap.
  Visual review rejected it as strict success because the target/hole moves
  onto the held peg before a visually confirmed active robot insertion, and the
  final metric-true frame is occluded. The validation-key success count remains
  `1`.
- Artifact review guard has been tightened: the Phase 03 audit output and
  review runbook now explicitly require active robot insertion visual review
  and reject target-assisted insertion where target / hole motion creates
  success by moving onto a mostly stationary peg / wooden stick. Simulator
  metric true is not success without that review.
- Fastshift approved-key reruns with corrected source-H5 gating are protocol
  evidence, not success. `h5_fastshift/try03` used source key
  `hole_late_fast_shift_seed10300001_idx5000`, triggered target motion at
  frame 99 only after the live H5 preinsert gate passed, executed 4 premotion
  and 4 postmotion Cosmos reports plus 27 Cosmos dynamic actions, and failed
  physical insertion (`simulator_success_metric=false`, final
  `peg_head_l2=0.12979192961772668`). `h5_fastshift/try04`, Slurm job
  `166830` / `server57`, completed with 4 premotion Cosmos reports, 4
  postmotion Cosmos reports, 29 Cosmos-derived dynamic actions, original DP
  finisher, `snap_detected=false`, and `peg_state_guard.ok=true`, but also
  failed (`simulator_success_metric=false`, final `peg_head_l2` about
  `0.1219`, best about `0.0994`). Failed fastshift attempts have been archived
  out of the active runs tree.
- New fast-shift key attempt `render_probe/fast06` targeted approved key
  `hole_late_fast_shift_seed10300253_idx5010` for a standard three-stage
  `h5_fastshift/try06` full-pipeline run. Slurm job `167534` landed on
  `server53`, but the render canary timed out at `render_rgb_array_start`.
  No Oracle rollout, action trace, Cosmos output, or video was produced, and no
  `h5_fastshift/try06` run directory exists. The failed probe/log were archived
  under `/public/home/yanhongru/ICLR2027/archive/Reflex/`. A server44-specific
  retry allocation, Slurm job `167535`, remained `Priority` pending and was
  canceled before compute. This is infrastructure evidence only; the new
  fast-shift key remains untested.
- Follow-up `render_probe/fast06b` again targeted the same approved key.
  Server63-specific allocation `167538` stayed pending and was canceled; an
  accidental four-node nodelist request `167539` was canceled before compute;
  corrected single-node allocation `167540` landed on `server46`, but its
  render canary also timed out at `render_rgb_array_start`. The probe/log were
  archived under `/public/home/yanhongru/ICLR2027/archive/Reflex/`. No
  `h5_fastshift/try06` run exists and no Oracle rollout started.
- Follow-up `render_probe/fast06c` checked the remaining live resource options.
  A smaller server44 request `167547` stayed pending and was canceled after
  server44 became fully allocated. Server39 allocation `167549` landed on bus
  `2C:00`, but the default `render_rgb_array` canary failed with
  `vk::DeviceLostError` at `camera.take_picture()`. The probe/log were archived
  under `/public/home/yanhongru/ICLR2027/archive/Reflex/`. No
  `h5_fastshift/try06` run exists and no Oracle rollout started.
- Follow-up `render_probe/fast06d` used a short server39 multi-GPU probe only.
  Slurm job `167551` exposed buses `2C:00`, `9A:00`, and `9B:00`; all three
  visible devices timed out at `render_rgb_array_start` under the default
  `render_rgb_array` canary. The probe/log were archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/`. No `h5_fastshift/try06`
  run exists and no Oracle rollout started.
- Current resource check after `fast06d`: the only nodes with free GPUs were
  `server28`, `server30`, `server39`, and `server53`. All four already have
  recent render-canary failure evidence in this protocol window. `server63`,
  the historical positive render node for `default + render_rgb_array`, had all
  eight GPUs allocated. Do not launch `h5_fastshift/try06` until a node/GPU
  first passes the same RGB render canary required by the full pipeline.
- Follow-up `render_probe/fast06e` first queued `server63` as Slurm job
  `167561`, but canceled before compute because the node stayed full. It then
  tested `server30` on Slurm job `167563`, bus `29:00`, with
  `default + render_rgb_array`; the canary timed out at
  `render_rgb_array_start` (`exit_code=124`). The probe/log were archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/`. No `h5_fastshift/try06`
  run exists and no Oracle rollout started.
- Follow-up `render_probe/fast06g` on Slurm job `167574` / `server44` passed
  `default + render_rgb_array` canary on bus `AA:00` and launched
  `h5_fastshift/try06` for approved key
  `hole_late_fast_shift_seed10300253_idx5010`. The full pipeline completed:
  4 premotion Cosmos reports, 6 postmotion Cosmos reports, 48 Cosmos-derived
  dynamic actions, videos, and `artifact_audit.ok=true`. It is negative
  evidence, not success: `simulator_success_metric=false`,
  `near_target_before_finisher=false`, `finisher_start_step=null`, final
  `peg_head_l2` about `0.3285m`, and `visual_full_insertion_confirmed=false`.
  The failed run/probe were archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/`.
  The live trigger L2 was about `0.1503m`, but Cosmos dynamic action control
  worsened it to about `0.1735m` after the first action and about `0.3285m` by
  the end, so the next change must diagnose the Cosmos action interface /
  selection rather than finisher gains.
- `action_diag/try11` on archived `h5_fastshift/try06` confirms that diagnosis:
  48 valid dynamic rows, 7D RMSE about `0.0622`, xyz RMSE about `0.0897`, L2
  delta about `+0.1550m`, and y sign agreement about `-0.542` versus teacher
  rows. This is read-only diagnostic evidence only.
- `h5_fastshift/try07` then ran the same approved key as an explicitly labeled
  `source_h5_teacher_dynamic` future-label upper-bound diagnostic. It completed
  4 premotion Cosmos reports, 5 postmotion Cosmos reports, 39 teacher dynamic
  rows, original DP finisher from step 150, videos, and no snap. It is still
  negative diagnostic evidence, not success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`.
  The run reached the near-target gate, and DP finisher best distance was about
  `0.0162m` at step 187, but the simulator success metric stayed false and the
  final state drifted back to about `0.1167m`. This shows the Cosmos dynamic
  interface is not the only issue; the DP finisher can get very close but does
  not complete or hold insertion on this live state.
- `h5_fastshift/try08` replaced only the finisher with
  `source_h5_teacher_suffix`, while keeping the same future-label teacher
  dynamic diagnostic and the same approved validation key. It also failed:
  `classification=diagnostic_future_label_teacher_suffix_exhausted_without_success`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  `finisher_start_step=151`, best finisher distance about `0.1138m`, final
  about `0.1202m`, and the H5 suffix exhausted at source action index `300`.
  Therefore the validation H5 action suffix is not directly replayable from the
  current live state; do not report it as validation-set success.
- `h5_fastshift/try09` then tested the targeted close-band hypothesis with
  `dp_then_manual_close` after the same future-label teacher dynamic diagnostic.
  It produced 4 premotion Cosmos reports, 5 postmotion Cosmos reports,
  39 teacher dynamic rows, and 16 DP-finisher rows. The simulator metric became
  true at step 165 with final `peg_head_l2` about `0.0087m`, but this is
  explicitly rejected as success:
  `classification=diagnostic_future_label_teacher_dynamic_metric_true_not_success`,
  `visual_full_insertion_confirmed=false`, `physical_insertion_success_claimed=false`,
  and artifact audit set `target_assisted_insertion_must_be_rejected=true`.
  Manual close did not actually take over; the final action source was still
  `diffusion_policy_before_manual_close_gate`. Treat this as another metric-true
  counterexample, not a successful sample. The run/probe/log were archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/`.
- `h5_fastshift/try10` repeated that close-band diagnostic but widened the
  manual handoff to `MANUAL_DP_TO_MANUAL_L2=0.06`,
  `MANUAL_SOFT_INSERT_THRESHOLD=0.05`, froze target motion during the finisher,
  and disabled manual yaw. It again reached simulator metric true at step 165
  with final `peg_head_l2` about `0.0116m`, but classification remained
  `diagnostic_future_label_teacher_dynamic_metric_true_not_success`, artifact
  audit again required rejecting target-assisted insertion, and action trace
  still contained zero `manual_close_after_dp_gate` rows. The 16 finisher rows
  were all `diffusion_policy_before_manual_close_gate`. This rejects the simple
  threshold-widening explanation; forcing an active manual insertion would need
  a different controller path, not another small `dp_then_manual_close`
  threshold tweak. The run/probe/log were archived.
- `h5_fastshift/try11` directly replaced the finisher with
  `manual_staged_hole_servo` to force a true manual physical finisher instead
  of another DP-before-manual gate. It completed the full trace with
  4 premotion Cosmos reports, 5 postmotion Cosmos reports, 38 future-label
  teacher dynamic rows, and 502 manual finisher rows. It failed physical
  insertion: `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, best finisher `peg_head_l2` about
  `0.1326m`, final about `0.1856m`, and `snap_detected=false`. This proves the
  direct manual staged finisher is worse on this live state and fast-shift
  should not continue with another finisher/gain/threshold sweep without a new
  diagnosis.
- `render_probe/fast08suffix` and `render_probe/fast08b` were no-rollout
  render-infrastructure failures on `server02` and `server46` while the outer
  probe launcher still forced `default + render_rgb_array`. The launcher now
  allows `RENDER_SHADER_PACK` and `RENDER_CANARY_API` overrides and defaults to
  `minimal + gym`, matching the guarded full-run wrapper. `fast08c` passed that
  canary on `server44` and produced the `try08` result above.
- Active Phase 03 run/log folders have been cleaned repeatedly after failed or
  invalid attempts. The active tree keeps the accepted strict single-case
  evidence `h5_continuous_insert/try04` and `h5_continuous_insert/try11`, the
  frozen-finisher diagnostic `h5_continuous_insert/try09`, and compact
  `action_diag/*` read-only diagnostics. Failed H5 retries, peg-disturbance
  retries, render/probe diagnostics, stale latest-summary files, empty run/log
  directories, and legacy long-name runs/logs were moved to
  `/public/home/yanhongru/ICLR2027/archive/Reflex/` preserving
  repository-relative structure.
- Validation-key metric-true diagnostic now rejected as strict success:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_constant/try01/`.
  It used approved
  key `hole_late_constant_seed10250253_idx5009`, triggered target motion at
  frame 120 only after the live H5 preinsert gate passed, executed one
  Cosmos-derived dynamic action before the DP finisher, reached simulator
  success at step 135, had `snap_detected=false`, and has visual verdict
  `visual_review_verdict.json`. After review, this is not counted as strict
  success because the moving target/hole appears to create a target-assisted
  insertion by moving onto the peg / wooden stick. It is protocol evidence and
  a metric-true counterexample, not active robot insertion success.
- The overall Oracle goal is still incomplete: forward/backward target motion,
  left/right target motion, peg/wooden-stick disturbance, and successful
  coverage across multiple approved `fix3_733` keys are not yet covered.
- Recent action-interface diagnostics compared reverse failure against the two
  accepted continuous successes. Reverse `action_diag/try03` has 48 rows, 7D
  RMSE about `0.0803`, xyz RMSE about `0.1144`, L2 delta about `+0.1505`, and
  poor x/y sign agreement. Successful continuous diagnostics
  `action_diag/try04` and `action_diag/try05` are lower-error and L2-decreasing
  (`try04` 7D RMSE about `0.0296`; `try05` about `0.0464`). The next reverse
  retry is therefore a Cosmos-action-interface diagnostic with a documented
  direction guard, not a finisher gain sweep.
- Reverse `h5_reverse/try14` attempted that diagnostic on Slurm job `167380` /
  `server39`, but the render canary hung at `render_rgb_array_start`; the srun
  step was canceled before rollout. It has no summary, action trace, or video
  and is archived as infrastructure evidence only. A replacement `try15`
  allocation excluding `server39` ran on Slurm job `167385` / `server44`.
  `try15` completed 4 premotion Cosmos reports, 6 postmotion Cosmos reports,
  and 48 Cosmos-derived dynamic actions, but failed before finisher entry:
  `near_target_before_finisher=false`, final dynamic L2 about `0.2077`, and
  simulator success false. `action_diag/try06` shows the scale/rectify guard
  overcorrected: 7D RMSE about `0.3229`, xyz RMSE about `0.3857`, and executed
  y mean absolute about `0.747` versus teacher about `0.145`. Stop this
  reverse guard family unless the next diagnosis changes the adapter design,
  not just the y gain or sign toggle.
- Reverse `h5_reverse/try16`, Slurm job `167425` / `server44`, ran the
  explicitly labeled future-label `source_h5_teacher_dynamic` diagnostic on
  approved key `hole_late_reverse_seed1040017_idx1250`. It kept DP prefix,
  premotion/postmotion Cosmos RGB/action evidence, target-motion-complete
  gating, and DP finisher, but executed 31 dynamic actions from the matching
  source H5 teacher rows. The simulator metric became true and best finisher
  `peg_head_l2` was about `0.0128`, but this is invalid as success:
  classification is `diagnostic_future_label_teacher_dynamic_metric_true_not_success`,
  `physical_insertion_success_claimed=false`, and artifact audit failed with
  `non_cosmos_action_source_in_dynamic_stage`. The run/log are archived. This
  supports the diagnosis that reverse is blocked by Cosmos/action selection and
  controller interface, not by a simple finisher budget/gain issue.
- Reverse `h5_reverse/try17`, Slurm job `167436` / `server02`, repeated the
  same future-label teacher-dynamic diagnostic on second approved reverse key
  `hole_late_reverse_seed1040025_idx0003`. It completed 4 premotion Cosmos
  reports, 5 postmotion Cosmos reports, 39 teacher dynamic rows, and target
  motion complete before the DP finisher, but still failed:
  `simulator_success_metric=false`, dynamic L2 improved only to about `0.0958`,
  best finisher L2 was about `0.0902`, and final L2 was about `0.1048`.
  `action_diag/try08` has zero executed/teacher action error by construction.
  The run/log are archived. This second-key result means reverse is not a
  single-bottleneck problem; at least this key also has a state-alignment or
  finisher-region issue after target motion, not merely a Cosmos action
  sign/scale mismatch.
- Reverse `h5_reverse/try18`, Slurm job `167457` / `server02`, returned to
  real Cosmos-action dynamic control with the conservative
  `source_motion_sign` / `clip_opposite` guard and identity scales. It produced
  4 premotion Cosmos reports, 6 postmotion Cosmos reports, 48 Cosmos-derived
  dynamic actions, videos, and `artifact_audit.ok=true`, but failed before
  finisher entry: final L2 was about `0.2861` and simulator success stayed
  false. `action_diag/try09` shows L2 delta about `+0.1463`, x sign agreement
  about `-0.0625`, and y magnitude still far below teacher (`0.0304` versus
  `0.1449`). Do not continue reverse with another blind sign/scale tweak; the
  next change needs a different Cosmos action-interface design or a documented
  bridge diagnostic.
- Reverse `h5_reverse/try19` attempted `COSMOS_ACTION_HORIZON=1`, but server28
  failed the render canary before rollout; it is archived infrastructure
  evidence only. Replacement `h5_reverse/try20` ran on server02 with
  receding-horizon-1 real Cosmos dynamic actions, no teacher labels, identity
  scales, 4 premotion reports, 40 postmotion reports, and `artifact_audit.ok=true`.
  It still failed before finisher entry: final L2 about `0.3094`, simulator
  success false. `action_diag/try10` shows 7D RMSE about `0.0952`, xyz RMSE
  about `0.1303`, L2 delta about `+0.1735`, x sign agreement about `0.25`, and
  y sign agreement about `0.10`. Horizon 1 is rejected as a reverse fix; the
  first action is close but live-history receding predictions drift off the
  demonstrated action manifold from the second step onward.
- Reverse-key retry `h5_reverse/try10`, Slurm job `166852` / `server63`, used
  approved key `hole_late_reverse_seed1040017_idx1250`, 6 premotion Cosmos
  reports, 1 postmotion Cosmos report, 4 Cosmos-derived dynamic actions, and
  `dp_then_manual_close` with a `0.024m` close-band handoff. Artifact audit
  passed and `snap_detected=false`, but it failed physical insertion:
  `simulator_success_metric=false`, final `peg_head_l2` about `0.1678`, best
  about `0.1037`. The trace shows 410
  `diffusion_policy_before_manual_close_gate` rows and zero
  `manual_close_after_dp_gate` rows, so the manual close stage did not trigger.
  This is negative reverse-key evidence, not directional success.
- Reverse-key retry `h5_reverse/try11` targeted a second approved reverse key,
  `hole_late_reverse_seed1040025_idx0003`, but stopped before rollout on Slurm
  job `166896` / `server63` when compute-node preflight reported
  `ValueError: source code string cannot contain null bytes`. A byte scan of
  the listed source files found no NUL bytes afterward, so record this as
  preflight / infrastructure failure only. Follow-up `render_probe/try15`
  attempted to launch `h5_reverse/try12`, but server36 timed out at
  `render_rgb_array_start` and no full rollout started. Neither run affects the
  success count.
- Reverse-key retry `h5_reverse/try12`, Slurm job `166912` / `server57`, then
  ran the same second approved reverse key with the prior DP-finisher settings
  after disabling only the py-compile preflight. It completed the full protocol
  with 6 premotion Cosmos reports, 1 postmotion Cosmos report, 4 Cosmos-derived
  dynamic actions, videos, and `artifact_audit.ok=true`, but failed:
  `simulator_success_metric=false`, final `peg_head_l2` about `0.1113`, best
  about `0.0895`, `snap_detected=false`, and `peg_state_guard.ok=true`. This is
  negative evidence on a second reverse key, not directional success.
- Additional source-H5 attempts completed on Slurm job `165423` / `server57`
  with short names:
  `h5_continuous_insert/try01`, `h5_constant/try02`, and `h5_sine/try01`.
  All used approved `fix3_733` keys, live source-H5 motion gating, premotion
  and postmotion Cosmos RGB/action evidence, and no peg state intervention.
  None increases the success count: `h5_continuous_insert/try01` has simulator
  metric true but failed close-up replay visual confirmation; `h5_constant/try02`
  and `h5_sine/try01` are physical failures.
- Reverse-key retry `h5_reverse/try02`, Slurm job `166477` / `server63`, used
  approved key `hole_late_reverse_seed1040017_idx1250` with source-H5 gating,
  six pre-motion Cosmos reports, one post-motion Cosmos report, 4
  Cosmos-derived dynamic actions, and `manual_staged_hole_servo` as the
  near-target physical finisher. It passed artifact audit and had
  `snap_detected=false`, but it is still not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, final `peg_head_l2` about `0.1865`, best
  finisher `peg_head_l2` about `0.1400`. Trace review shows the manual finisher
  worsened the state after the Cosmos segment, so do not count it as reverse
  direction coverage.
- Reverse-key retry `h5_reverse/try03`, Slurm job `166502` / `server63`, kept
  the same approved key and 4 Cosmos-derived dynamic actions, then used the
  original DP checkpoint as the near-target finisher. It passed artifact audit
  and had `snap_detected=false`, but is still not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, final `peg_head_l2` about `0.0220`, best
  about `0.0210`. The DP finisher ran 410 rows and plateaued near 2.1-2.2 cm,
  so the next diagnostic is a close-band action-level handoff, not more
  simulator-state intervention.
- The runner now has `dp_then_manual_close`, an Oracle diagnostic finisher that
  uses the original DP checkpoint until live `peg_head_l2` enters a small
  threshold, then switches to the existing staged physical action controller for
  the final close-band push. This remains controller-action only and must still
  pass artifact audit and visual review; it is not method evidence.
- Reverse-key retry `h5_reverse/try04`, Slurm job `166542` / `server63`, tested
  `dp_then_manual_close` with a `0.03m` close-band threshold. It passed artifact
  audit and had `snap_detected=false`, but failed:
  `simulator_success_metric=false`, final `peg_head_l2` about `0.0327`, best
  about `0.0302`. Trace counts show 410
  `diffusion_policy_before_manual_close_gate` rows and 0
  `manual_close_after_dp_gate` rows, so the threshold was too tight and the
  intended manual close push never executed.
- Reverse-key retry `h5_reverse/try05`, Slurm job `166581` / `server63`, tried
  to loosen the close handoff, but the wrapper did not pass the new
  `manual_dp_to_manual_l2` field into the runner, so Python still used the old
  `0.03m` default. It failed with `simulator_success_metric=false`, final
  `peg_head_l2` about `0.0347`, best about `0.0332`, and 0
  `manual_close_after_dp_gate` rows. The Python default has been raised to
  `0.04m` before the next retry so the close handoff is actually tested.
- Reverse-key retry `h5_reverse/try06`, Slurm job `166613` / `server63`, used
  the same approved source-H5 key with short grouped output names. It completed
  the full protocol with 6 premotion Cosmos reports, 1 postmotion Cosmos
  report, 4 Cosmos dynamic actions, 200 DP close-band rows, and 210
  `manual_close_after_dp_gate` rows, so the close handoff was finally exercised.
  Artifact audit passed and `snap_detected=false`, but it still failed:
  `simulator_success_metric=false`, final `peg_head_l2` about `0.0398`, best
  about `0.0278`. This is not a success and does not increase the validation-key
  success count.
- Reverse-key retry `h5_reverse/try07`, Slurm job `166648` / `server46`, used
  the narrower `0.025m` DP-to-manual close threshold but did not enter rollout:
  the render canary hung at `render_rgb_array_start` and exited 124. It wrote
  only `manifest.txt` and `classification.txt`; this is infrastructure failure
  evidence only, not Oracle evidence.
- Reverse-key retry `h5_reverse/try08` was queued on `server63` with the same
  narrower close-threshold settings because `server46` failed canary, then
  completed rollout on Slurm job `166651` / `server63`, with render canary
  passed, 6 premotion Cosmos reports, 1 postmotion Cosmos report, 4 Cosmos
  dynamic actions, and 510 DP close-band rows. The `0.025m` handoff threshold
  was too low: `manual_close_after_dp_gate` never triggered. It failed with
  `simulator_success_metric=false`, final `peg_head_l2` about `0.0350`, best
  about `0.0334`, and summary `snap_detected=false`. The run wrote
  summary/action trace/videos but no external `artifact_audit.json`, so treat it
  as complete negative rollout evidence with missing wrapper audit, not success.
- Completed / failed reverse attempts `h5_reverse/try01` through `try08` have
  been moved out of the active runs tree to
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_reverse/`,
  with matching logs under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/h5_reverse/`.
- Reverse-key retry `h5_reverse/try09` is queued as Slurm job `166816` on
  `server57` with approved key `hole_late_reverse_seed1040017_idx1250`,
  original DP finisher, `MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER=4`,
  `MAX_FINISHER_STEPS=520`, and `MAX_EPISODE_STEPS=650`. As of
  `2026-07-05T13:52+08:00`, it was pending for `Priority` and had no active
  run files yet.
- Reverse-key retry `h5_reverse/try09` then completed on Slurm job `166816` /
  `server57`. It passed artifact audit with 4 premotion Cosmos reports,
  1 postmotion Cosmos report, 4 Cosmos-derived dynamic actions, and 540
  DP-finisher rows, with `snap_detected=false` and `peg_state_guard.ok=true`,
  but failed physical insertion: `simulator_success_metric=false`, final
  `peg_head_l2` about `0.1054`, best about `0.0981`. The longer pure-DP
  finisher worsened reverse-key performance and does not count as directional
  success.
- Continuous-insert retry `h5_continuous_insert/try04`, Slurm job `166789` /
  `server57`, is the first accepted strict validation-key single-case Oracle
  success. It used approved key
  `hole_late_continuous_insert_seed10241044_idx5004`, `NEAR_TARGET_L2=0.16`,
  original DP finisher, and `MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER=4`.
  It completed with 4 premotion Cosmos reports, 1 postmotion Cosmos report,
  4 Cosmos-derived dynamic actions, DP finisher from step 145,
  `artifact_audit.ok=true`, `simulator_success_metric=true`, final
  `peg_head_l2` about `0.01446`, `snap_detected=false`, and
  `peg_state_guard.ok=true`. Visual review of `videos/annotated.mp4`,
  `videos/raw.mp4`, and extracted review keyframes confirmed active robot
  insertion rather than target-assisted insertion; verdict:
  `experiments/maniskill/runs/03_oracle/h5_continuous_insert/try04/visual_review_verdict.json`.
  This is still `method_evidence_allowed=false` and does not complete the
  overall Oracle task.
- Older continuous-insert attempts `try01`, `try02`, `try03`, and infrastructure
  failure `try05` have been moved out of the active runs tree to
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_continuous_insert/`,
  with matching logs under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/h5_continuous_insert/`.
- Constant-key retry `h5_constant/try03`, Slurm job `166682` / `server63`, used
  a different approved key `hole_late_constant_seed10250338_idx5012` with the
  same `NEAR_TARGET_L2=0.16` DP-finisher protocol. It passed artifact audit and
  had `snap_detected=false`, but failed: `simulator_success_metric=false`,
  final `peg_head_l2` about `0.1303`, best about `0.0749`. This key does not
  add a second success.
- Move-stop retry `h5_move_stop/try02`, Slurm job `166700` / `server63`, used
  approved key `hole_late_move_stop_seed1080064_idx0000` with the short
  source-H5 launcher, 4 premotion Cosmos reports, 1 postmotion Cosmos report,
  1 Cosmos dynamic action, and `manual_align_then_dp` as the physical finisher.
  It passed artifact audit and had `snap_detected=false`, but failed:
  `simulator_success_metric=false`, final `peg_head_l2` about `0.4783`, best
  about `0.1252`. Trace counts show 260
  `manual_staged_align_before_dp_gate` rows and no
  `diffusion_policy_after_manual_align_gate` rows, so this finisher never
  reached its DP handoff and actively worsened the state. Do not count it as
  success or reuse this setting.
- Move-stop retry `h5_move_stop/try03` was first queued as fixed-`server63`
  Slurm job `166727`, but that pending job was canceled because Slurm estimated
  a next-day start. It was replaced by probe-backed Slurm job `166729`, using
  the same approved key, short grouped path, at least 4 required
  Cosmos-derived dynamic actions before finisher, and original DP as the
  finisher. Job `166729` landed on `server36` and failed the render canary at
  `render_rgb_array_start` with exit code 124 before any Oracle rollout. This
  is infrastructure failure only, not Oracle evidence. A follow-up probe-backed
  attempt, Slurm job `166734` on `server46`, also failed the render canary at
  `render_rgb_array_start` with exit code 124 before any Oracle rollout. The
  next probe-backed attempt, Slurm job `166742` on `server02`, also failed the
  same render canary with exit code 124. These are infrastructure failures
  only. A reduced-resource `server63` retry was queued as Slurm job `166746`
  and then replaced by shorter-walltime Slurm job `166748` with `2 CPU`, `40G`
  memory, the same approved key, short path `h5_move_stop/try03`, at least 4
  required Cosmos-derived dynamic actions, and original DP finisher. It is
  still pending with Slurm-estimated start `2026-07-06T07:19:22+08:00` and has
  produced no rollout evidence yet. Rechecked on `2026-07-05T12:53+08:00`:
  job `166748` was still pending for `Priority`, and
  `experiments/maniskill/runs/03_oracle/h5_move_stop/try03/` still had no
  files. Rechecked again on `2026-07-05T12:54:51+08:00`: status remained
  pending for `Priority`, estimated start remained
  `2026-07-06T07:19:22+08:00`, and the run directory still had no files.
  Rechecked again on `2026-07-05T12:57+08:00`: job `166748` was still pending
  for `Priority`, and `h5_move_stop/try03` still had no files. A parallel
  short-walltime `server56` attempt, `h5_move_stop/try04` / Slurm job `166754`,
  started on `server56` but failed before rollout because the default
  `render_rgb_array` canary hung at `render_rgb_array_start` and exited 124.
  `try04` wrote only `manifest.txt` and `classification.txt`, with
  `phase03_status=blocked_render_canary_failed_no_rollout`; it is infrastructure
  failure only, not Oracle evidence. A follow-up immediate-allocation attempt,
  `h5_move_stop/try05` / Slurm job `166759` targeting `server10`, never obtained
  a node and wrote no run files; it is a scheduler attempt only.
  `h5_move_stop/try06` then started on Slurm job `166764` / `server57` and
  completed the full source-H5-gated protocol with 4 premotion Cosmos reports,
  1 postmotion Cosmos report, 4 Cosmos-derived dynamic actions, and the
  original DP finisher. Artifact audit passed, `snap_detected=false`, and
  `peg_state_guard.ok=true`, but it failed physical insertion:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.1005`, best about `0.0815`. The redundant
  fixed-`server63` pending job `166748` was canceled after `try06` produced
  this result. Strict validation-key success count remains `1`.
- Peg-disturbance coverage now has one completed protocol attempt:
  `peg_disturb/try09`. It used approved key `peg_disturb_seed1051032_idx0008`,
  render canary passed on `server46`, four premotion Cosmos predictions and one
  postmotion Cosmos prediction completed, one Cosmos-derived dynamic action
  executed after the peg perturb trigger, and the DP finisher ran after the
  near-target gate. It is not completed peg-disturbance coverage because the
  physical perturb was under-strength / wrong-direction relative to the source
  key's intended `[0, -0.04, 0.02]` peg perturb. It is not a success:
  `simulator_success_metric=false`, `physical_insertion_success=false`, and
  `visual_full_insertion_confirmed=false`.
- Peg-disturbance calibration `peg_disturb/calib01` on Slurm job `165697` /
  `server36` is diagnostic only. It confirmed that a force-only peg perturb can
  match the approved key without `peg.set_pose`: best trial
  `force_scale=25.0`, `force_steps=8`, observed delta about
  `[0.0119, -0.0371, 0.00993]`, fraction about `0.898`, cosine about `0.936`.
- Peg-disturbance retry `peg_disturb/try10` on the same allocation did not
  enter rollout. The wrapper wrote
  `blocked_render_canary_failed_no_rollout` after the default-shader render
  canary hung at `render_rgb_array_start` and timed out on `server36`. This is
  infrastructure failure evidence only, not Oracle evidence.
- Peg-disturbance retry `peg_disturb/try11` on Slurm job `165729` /
  `server28` also did not enter rollout. The default-shader render canary hung
  at `render_rgb_array_start` and timed out with exit code `124`; only
  `manifest.txt` and `classification.txt` were written. This is infrastructure
  failure evidence only, not Oracle evidence.
- Peg-disturbance retry `peg_disturb/try12` on Slurm job `165737` /
  `server63` completed the full pipeline with short names: four premotion
  Cosmos RGB/action calls, one postmotion Cosmos RGB/action call, four
  Cosmos-derived dynamic actions, DP finisher, and `videos/raw.mp4` /
  `videos/annotated.mp4`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  and `artifact_audit.ok=false`. The audit failure is
  `peg_perturb_observed_delta_wrong_direction_for_source_key`; the force window
  was still robot-action-confounded in this runner version.
- After `try12`, `scripts/training/eval_dp_oracle_full_pipeline.py` was patched
  so peg-disturbance force is applied through the full calibrated force window
  using zero robot actions before postmotion Cosmos control begins. The next
  retry is `peg_disturb/try13`.
- Peg-disturbance retry `peg_disturb/try13` on Slurm job `165762` /
  `server63` completed the full pipeline audit with the patched zero-action
  force window. It produced 4 premotion and 4 postmotion Cosmos RGB/action
  calls, executed 32 Cosmos-derived dynamic actions, wrote videos, and passed
  `artifact_audit.ok=true`; observed perturb matched the source key
  (`cosine=0.9417`, fraction `0.9967`). It is still not a success:
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  `near_target_before_finisher=false`, and no DP finisher row ran because final
  `peg_head_l2` stayed about `0.1967`.
- Peg-disturbance retry `peg_disturb/try14` used `NEAR_TARGET_L2=0.20` while
  preserving `MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER=4`. It passed audit,
  executed 4 Cosmos-derived dynamic actions, and then entered the DP finisher
  for 273 rows. It is still not a success: `simulator_success_metric=false`,
  `visual_full_insertion_confirmed=false`, and final `peg_head_l2` worsened to
  about `0.2059`. The next retry should use an existing manual physical
  finisher instead of the DP finisher.
- Peg-disturbance retry `peg_disturb/try15` requested
  `manual_hole_frame_servo`, but `server46` failed the render canary and no
  rollout started. This is infrastructure failure only, not Oracle evidence.
- Peg-disturbance retry `peg_disturb/try16` on Slurm job `165821` /
  `server63` completed the full pipeline with `manual_hole_frame_servo`.
  It passed artifact audit, matched the approved peg perturb direction
  (`cosine=0.9397`, fraction `0.9244`), executed 4 Cosmos-derived dynamic
  actions and 180 manual finisher rows, and had `snap_detected=false`.
  It is still not a success: `simulator_success_metric=false`,
  `visual_full_insertion_confirmed=false`, and final `peg_head_l2` was about
  `0.1843` with large lateral error. The next retry should use a staged manual
  physical finisher rather than the failed DP finisher or the failed
  `manual_hole_frame_servo`.
- Peg-disturbance retry `peg_disturb/try17` requested
  `manual_staged_hole_servo`, but Slurm job `165844` / `server27` failed the
  render canary at `render_rgb_array_start` with exit code `124`; no rollout
  started. This is infrastructure failure only, not Oracle evidence. Exclude
  `server27` from the next render-bearing attempt.
- Peg-disturbance retry `peg_disturb/try18` requested the same staged manual
  finisher on Slurm job `165851` / `server63`, but it also failed the render
  canary at `render_rgb_array_start` with exit code `124`; no rollout started.
  This is infrastructure failure only, not Oracle evidence.
- `peg_disturb/try19` and `peg_disturb/try20` were Slurm allocation attempts
  that were canceled while still pending and before any allocation, manifest,
  rollout, Cosmos call, or video existed. They are scheduling attempts only,
  not Oracle evidence and not render failures.
- `peg_disturb/try21` was canceled while still pending and before any
  allocation, manifest, rollout, Cosmos call, or video existed. It is a
  scheduling attempt only.
- `peg_disturb/try22` requested `server63` and did get an allocation, but it
  failed the render canary at `render_rgb_array_start` with exit code `124`;
  no rollout started. This is infrastructure failure only, not Oracle evidence.
- After `try22`, the full-pipeline wrapper and runner were patched so the
  render canary and full rollout use a shared configurable
  `RENDER_SHADER_PACK`, defaulting to `minimal` instead of hard-coded
  `default`. This does not skip video evidence; it is intended to avoid the
  repeated default-shader hang while still requiring rendered RGB frames before
  rollout.
- `peg_disturb/try23` used the patched shared `RENDER_SHADER_PACK=minimal`
  path on Slurm job `165900` / `server63`, but the render canary still hung at
  `render_rgb_array_start` and timed out with exit code `124`; no rollout
  started. This is infrastructure failure only, not Oracle evidence. The next
  attempt should try a node that has not yet been tested with the minimal
  shader canary.
- `peg_disturb/try24` used `RENDER_SHADER_PACK=minimal` on Slurm job `165910`
  / `server27`, but the render canary still hung at `render_rgb_array_start`
  and timed out with exit code `124`; no rollout started. This is
  infrastructure failure only, not Oracle evidence.
- `peg_disturb/try25` used `RENDER_SHADER_PACK=minimal` on Slurm job `165918`
  / `server28`, but the render canary also hung at `render_rgb_array_start`
  and timed out with exit code `124`; no rollout started. This is
  infrastructure failure only, not Oracle evidence.
- After `try25`, `scripts/world_model/render_min_canary.py` was patched so the
  default canary render API is `env.render()`, matching the full pipeline's
  main video path, instead of directly calling
  `env.unwrapped.render_rgb_array("render_camera")`. The canary still must
  write a real RGB frame before rollout starts.
- `peg_disturb/try26` used the patched `env.render()` canary with
  `RENDER_SHADER_PACK=minimal` on Slurm job `165931` / `server28`, but it still
  hung at `render_gym_start` and timed out with exit code `124`; no rollout
  started. This is infrastructure failure only, not Oracle evidence.
- `peg_disturb/try27` used the patched `env.render()` canary with
  `RENDER_SHADER_PACK=minimal` on Slurm job `165942` / `server46`, but it still
  hung at `render_gym_start` and timed out with exit code `124`; no rollout
  started. This is infrastructure failure only, not Oracle evidence.
- `peg_disturb/try28` used the patched `env.render()` canary with
  `RENDER_SHADER_PACK=minimal` on Slurm job `165952` / `server30`, but it still
  hung at `render_gym_start` and timed out with exit code `124`; no rollout
  started. This is infrastructure failure only, not Oracle evidence. At this
  point the available tested nodes in this retry window cannot produce the
  required first RGB render frame, so no Oracle rollout with valid video
  evidence can start until a render-capable node / render stack is available.
- Render GPU probe `render_probe/try01` on Slurm job `165968` / `server28`
  tested three visible GPUs, including bus ids `29:00`, `5C:00`, and `AA:00`,
  with `RENDER_SHADER_PACK=minimal` and `RENDER_CANARY_API=gym`. All three
  timed out at `render_gym_start`, so no full Oracle run was launched. This is
  infrastructure diagnosis only, not Oracle evidence.
- Render GPU probe `render_probe/try02` on Slurm job `165980` / `server28`
  retested the same three visible GPUs, including bus ids `29:00`, `5C:00`,
  and `AA:00`, with the old successful `try16` canary path:
  `RENDER_SHADER_PACK=default` and `RENDER_CANARY_API=render_rgb_array`. All
  three timed out at `render_rgb_array_start`, so no `peg_disturb/try29` full
  Oracle run was launched. This is infrastructure diagnosis only, not Oracle
  evidence.
- Render GPU probe `render_probe/try03` on Slurm job `165988` / `server63`
  found a render-capable `AA:00` GPU with the old successful canary path
  (`default` + `render_rgb_array`). It then launched a full pipeline run, but
  because `RUN_GROUP` leaked from the probe environment the full run landed at
  `experiments/maniskill/runs/03_oracle/render_probe/try29/` instead of
  `peg_disturb/try29/`. The probe script has been patched so future automatic
  full runs force `RUN_GROUP=peg_disturb`.
- The probe-launched full run at `render_probe/try29` completed the protocol:
  4 premotion Cosmos reports, 1 postmotion Cosmos report, 4 Cosmos-derived
  dynamic actions, matched physical peg perturb, 180 `manual_staged_hole_servo`
  finisher rows, videos, `artifact_audit.ok=true`, and `snap_detected=false`.
  It is not a success: `simulator_success_metric=false`,
  `visual_full_insertion_confirmed=false`, and final `peg_head_l2` was about
  `0.1364`. The staged finisher aligned y/z but did not push far enough along
  the insertion axis.
- Render GPU probe `render_probe/try04` on Slurm job `166014` / `server63`
  used the corrected auto-launch path and wrote the full run to
  `peg_disturb/try29`. It completed the protocol with 4 premotion Cosmos
  reports, 1 postmotion report, 4 Cosmos-derived dynamic actions, matched
  physical peg perturb (`cosine=0.9459`, fraction `0.9514`), 320
  `manual_staged_hole_servo` finisher rows, videos, `artifact_audit.ok=true`,
  and `snap_detected=false`. It is still not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  and final `peg_head_l2` was about `0.2032`. The summary records
  `target_motion_state_intervention_used=true`, so do not treat this run as
  method evidence. Trace review shows it reached a better intermediate state
  around step 188 (`peg_head_l2` about `0.0831`, y/z nearly aligned), then the
  fixed-yaw staged finisher swept the peg away.
- `scripts/training/eval_dp_oracle_full_pipeline.py` now ignores target-motion
  residuals at or below `1e-6 m`; this prevents the approved `peg_disturb` key's
  `7e-9 m` floating-point target delta from being recorded as target state
  intervention.
- `render_probe/try05`, Slurm job `166046`, launched `peg_disturb/try30` after
  real RGB canary frames. It completed the full protocol with 4 premotion Cosmos
  reports, 1 postmotion report, 4 Cosmos-derived dynamic actions, matched
  physical peg perturb, 140 `manual_staged_hole_servo` finisher rows, videos,
  `artifact_audit.ok=true`, and `snap_detected=false`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.0979`, and best finisher `peg_head_l2` about
  `0.0966`. The target residual fix worked:
  `target_motion_state_intervention_used=false`.
- `render_probe/try07`, Slurm job `166078`, launched `peg_disturb/try32` after
  real RGB canary frames. It completed the full protocol with 4 premotion Cosmos
  reports, 1 postmotion report, 4 Cosmos-derived dynamic actions, matched
  physical peg perturb, 220 `manual_staged_hole_servo` finisher rows, videos,
  `artifact_audit.ok=true`, and `snap_detected=false`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.1170`, and best finisher `peg_head_l2` about
  `0.0991`. Small continuous yaw worsened the final state.
- `render_probe/try08`, Slurm job `166117`, launched `peg_disturb/try33` after
  real RGB canary frames. It completed the full protocol with 4 premotion Cosmos
  reports, 1 postmotion report, 4 Cosmos-derived dynamic actions, matched
  physical peg perturb, 220 `manual_staged_hole_servo` finisher rows, videos,
  `artifact_audit.ok=true`, and `snap_detected=false`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.1537`, and best finisher `peg_head_l2` about
  `0.1027`. The yaw stop threshold `0.10` was too late.
- `render_probe/try09`, Slurm job `166157`, found a render-capable GPU on
  `server63` and launched `peg_disturb/try34` with short paths,
  `MANUAL_YAW_ACTION=0.22`, and `MANUAL_YAW_STOP_L2=0.13`. The run completed
  the full protocol with 4 premotion Cosmos reports, 1 postmotion report, 4
  Cosmos-derived dynamic actions, matched physical peg perturb, 220
  `manual_staged_hole_servo` finisher rows, videos, `artifact_audit.ok=true`,
  and `snap_detected=false`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.1563`, and best finisher `peg_head_l2` about
  `0.1025`. Target residual accounting remains fixed with
  `target_motion_state_intervention_used=false`.
- After `try34`, `scripts/training/eval_dp_oracle_full_pipeline.py` was patched
  so `manual_staged_hole_servo_action` actually honors `MANUAL_YAW_STOP_L2`.
  The previous yaw-stop option only applied to `manual_hole_frame_servo_action`.
- `render_probe/try10`, Slurm job `166189`, found a render-capable GPU on
  `server63` and launched `peg_disturb/try35` with short paths,
  `MANUAL_YAW_ACTION=0.22`, and `MANUAL_YAW_STOP_L2=0.145`. The run completed
  the full protocol with 4 premotion Cosmos reports, 1 postmotion report, 4
  Cosmos-derived dynamic actions, matched physical peg perturb, 220
  `manual_staged_hole_servo` finisher rows, videos, `artifact_audit.ok=true`,
  and `snap_detected=false`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.1062`, and best finisher `peg_head_l2` about
  `0.0996`. The yaw stop now works: 23 finisher rows used yaw `0.22`, then 197
  rows used yaw `0`. The remaining failure is insufficient insertion-axis
  progress after alignment.
- `render_probe/try11`, Slurm job `166208`, found a render-capable GPU on
  `server63` and launched `peg_disturb/try36` with short paths, patched yaw
  stop, `MANUAL_INSERT_SPEED=0.12`, and `MANUAL_FORWARD_GAIN=3.0`. The run
  completed the full protocol with 4 premotion Cosmos reports, 1 postmotion
  report, 4 Cosmos-derived dynamic actions, matched physical peg perturb, 220
  `manual_staged_hole_servo` finisher rows, videos, `artifact_audit.ok=true`,
  and `snap_detected=false`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.1373`, and best finisher `peg_head_l2` about
  `0.0989`. Stronger insertion push did not solve the peg-disturb case and
  worsened final alignment, so do not keep blindly increasing force / forward
  gain.
- After `try36`, `scripts/training/eval_dp_oracle_full_pipeline.py` gained
  `manual_staged_twist_insert`, a physical-action finisher that applies
  configurable roll/pitch/yaw only during the insertion stage. This was
  motivated by the metric-true `h5_constant/try01` DP finisher trace, but that
  trace is now rejected as strict success because of target-assisted insertion
  concern.
- `render_probe/try12`, Slurm job `166228`, found a render-capable GPU on
  `server63` and launched `peg_disturb/try37` with short paths,
  `manual_staged_twist_insert`, insert roll `-0.18`, and insert yaw `-0.28`.
  The run completed the full protocol with 4 premotion Cosmos reports, 1
  postmotion report, 4 Cosmos-derived dynamic actions, matched physical peg
  perturb, 220 manual finisher rows, videos, `artifact_audit.ok=true`, and
  `snap_detected=false`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.1117`, and best finisher `peg_head_l2` about
  `0.0950`. The twist ran for 131 finisher rows and changed peg pose but still
  did not insert; do not use long continuous twist as a success path.
- Close-up replay review for `try37` ran under Slurm job `166248`, but its replay
  trace diverges from the original action trace during the DP prefix and physical
  force response. Treat `peg_disturb/try37/review/*/trace.json` as replay-debug
  only, not contact evidence. Original annotated-video frames extracted under
  Slurm job `166268` show the real failure: at finisher start, best frame, and
  final frame the peg / wooden stick remains outside the hole with pose/axis
  mismatch.
- After original-frame review, `scripts/training/eval_dp_oracle_full_pipeline.py`
  gained `manual_staged_dp_rot`, a hybrid physical finisher that keeps staged
  live-error xyz control but takes wrist rotation / gripper from the original DP
  checkpoint at each near-target finisher step. This keeps DP out of the dynamic
  Cosmos-control stage.
- `render_probe/try13`, Slurm job `166277`, found a render-capable GPU on
  `server63` and launched `peg_disturb/try38` with `manual_staged_dp_rot`. The
  run completed the full protocol with 4 premotion Cosmos reports, 1 postmotion
  report, 4 Cosmos-derived dynamic actions, matched physical peg perturb, 220
  hybrid finisher rows, videos, `artifact_audit.ok=true`, and
  `snap_detected=false`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final and best `peg_head_l2` about `0.1503`. The DP rotation components were
  too small to recover pose/axis mismatch.
- After `try38`, trace analysis showed `manual_staged_dp_rot` never entered the
  hard insertion gate because y/z error stayed just above the align threshold.
  The runner and wrapper now expose an explicitly logged soft-insert gate
  (`MANUAL_SOFT_INSERT_THRESHOLD`, `MANUAL_SOFT_INSERT_SCALE`) so the next retry
  can apply reduced physical insertion-axis action while continuing y/z
  correction. This remains a controller-action path only; no state edit is
  allowed.
- `peg_disturb/try39` is launcher failure only. The render canary wrote
  `render_canary/frame.png`, but the wrapper exited before
  `eval_dp_oracle_full_pipeline.py` started; no summary, action trace, Cosmos
  artifacts, or rollout video exist.
- `peg_disturb/try40`, Slurm job `166327`, completed the full protocol with the
  logged soft-insert gate and `manual_staged_dp_rot`. It passed artifact audit,
  matched physical peg perturb (`cosine=0.9385`, fraction `0.9906`), executed 4
  Cosmos-derived dynamic actions after 4 pre-motion and 1 post-motion Cosmos
  reports, wrote videos, and had `snap_detected=false`. It is still not a
  success: `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.0981`, best about `0.0973`. The soft gate
  improved the hybrid but y/z error stayed about `0.019m`, so no full insertion.
- `peg_disturb/try41`, Slurm job `166356`, completed the full protocol with
  soft-insert and pure `manual_staged_hole_servo` using stronger y/z correction.
  It passed artifact audit and had `snap_detected=false`, but is still not a
  success: final `peg_head_l2` about `0.1205`, best about `0.0992`. The stronger
  y/z correction reached best y/z about `0.007m`, but the peg still stalled
  about `0.099m` short along the insertion axis. Next retry should add controlled
  twist only during the aligned insertion stage.
- `peg_disturb/try42`, Slurm job `166377`, completed the full protocol with
  soft-insert, stronger y/z correction, and controlled manual twist during the
  aligned insertion stage (`roll=-0.23`, `yaw=-0.48`). It passed artifact audit
  and had `snap_detected=false`, but is still not a success: final
  `peg_head_l2` about `0.1002`, best about `0.0978`. The current manual finisher
  family is stuck around the same 9.7-10.0 cm insertion-axis plateau on this
  peg-disturb key.
- After `try42`, trace analysis showed the failed best frames still have a
  nontrivial peg/hole orientation gap: quaternion dot products around
  `0.984-0.988`, versus about `0.999` in the legacy successful diagnostic. The
  next retry should test `manual_staged_pose_servo`, a bounded live
  quaternion-error wrist correction added on top of the same physical staged /
  soft-insert translation controller. It remains Oracle diagnostic action
  control only, not method evidence and not state editing.
- `peg_disturb/try43`, Slurm job `166402`, completed the full protocol with
  direct `manual_staged_pose_servo`. It passed artifact audit and had
  `snap_detected=false`, but failed worse than the previous plateau: final
  `peg_head_l2` about `0.1219`, best about `0.1078`, and best-frame peg/hole
  quaternion dot dropped to about `0.881`. The direct quaternion-error command is
  likely wrong-frame / too aggressive. Do not use this sign/gain as a valid
  finisher; the next diagnostic should test a smaller reversed correction if
  continuing this family.
- `manual_align_then_dp` is now available for the next Oracle-finisher
  diagnostic. It uses manual staged/soft-insert actions until live y/z error is
  within the alignment threshold, then switches to the original DP checkpoint
  for the near-target finisher action chunk. This still keeps DP out of the
  post-motion Cosmos dynamic-control stage; action trace rows distinguish
  `manual_staged_align_before_dp_gate` from
  `diffusion_policy_after_manual_align_gate`.
- `peg_disturb/try44`, Slurm job `166424`, tested a smaller reversed
  `manual_staged_pose_servo`. It completed the full protocol and passed artifact
  audit with `snap_detected=false`, but still failed: final `peg_head_l2` about
  `0.1123`, best about `0.0982`. The direct pose-servo family remains a failed
  diagnostic path for this peg-disturb key.
- `peg_disturb/try45`, Slurm job `166449`, tested `manual_align_then_dp`. It
  completed the full protocol with 4 pre-motion Cosmos reports, 1 post-motion
  Cosmos report, 4 Cosmos dynamic actions, matched physical peg perturb, videos,
  `artifact_audit.ok=true`, and `snap_detected=false`. It is still not a
  success: `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.0950`, best about `0.0945`. Trace counts confirm
  78 `manual_staged_align_before_dp_gate` rows and 239
  `diffusion_policy_after_manual_align_gate` rows, so the DP finisher gate
  really executed; it still stalled about 9.5 cm short.
- `peg_disturb/try46`, Slurm job `166919` / `server57`, tested the second
  approved peg-disturbance key
  `peg_disturb_seed40751016_pseed42751016_idx13000` with the current
  `manual_align_then_dp` settings. It completed the full protocol with 4
  premotion Cosmos reports, 4 postmotion Cosmos reports, 32 Cosmos-derived
  dynamic actions, videos, `artifact_audit.ok=true`, no snap, and
  `peg_state_guard.ok=true`, but failed: `simulator_success_metric=false`,
  final `peg_head_l2` about `0.2229`. The near-target gate was not reached, so
  the finisher did not run. This is negative evidence on the second
  peg-disturb key, not disturbance success.
- `peg_disturb/try47`, Slurm job `166935` / `server57`, was a loosened
  near-gate retry on the same second peg-disturbance key. It was canceled by
  user direction before summary, action trace, or full attempt video were
  written. Only render canary and partial Cosmos prefix/action artifacts exist.
  Stop further peg-disturbance sweeps until explicit user direction is given.

## Current Phase 04 Status

- Phase 04 bridge entry `bridge_entry/try01` ran inside tmux-held Slurm job
  `167531` on `server02` and wrote active outputs under
  `experiments/maniskill/runs/04_integration/bridge_entry/try01/` plus log
  `logs/04_integration/bridge_entry/try01.log`.
- The run first passed `py_compile` for
  `scripts/world_model/phase03_bridge_diagnostic_entry.py`, then generated a
  manifest, candidate chunks, candidate table, manual review overlays,
  `trust_gate.json`, `bridge_entry_report.md`, and `classification.txt`.
- It did not execute a controller, did not use Oracle, did not edit simulator
  state, and does not claim physical insertion success. Its evidence label is
  `offline_bridge_diagnostic_entry_no_controller_execution` with
  `method_evidence_allowed=false`.
- The trust gate loaded Phase 03 action diagnostics from two accepted
  continuous references (`action_diag/try04`, `action_diag/try05`) and two
  reverse failures (`action_diag/try09`, `action_diag/try10`). The active
  action ruling is `reverse_cosmos_action_not_trusted`.
- Current bridge ruling: `trust_cosmos=false`, `execute_chunk_len=0`,
  `handoff_mode=hold_reobserve_only`, and selected candidate
  `hold_reobserve`. Do not execute insertion candidates from this bridge until
  the RGB task-state extractor / action interface produces reliable overlays
  and a passing trust signal.

## Naming Rule

- New outputs must use short grouped paths, for example
  `experiments/maniskill/runs/03_oracle/h5_constant/try01/` and
  `logs/03_oracle/h5_constant/try01.log`.
- Common context must be represented by folders, not repeated in every leaf
  name. Use `03_oracle/peg_disturb/try17/`, not
  `p03_oracle_full_pipeline_peg_disturb_manual_staged_20260705_<job>_<host>/`.
- If several outputs share the same prefix, create one folder for that shared
  prefix and keep the files inside short. Human-readable names beat exhaustive
  names; job id, host, timestamp, checkpoint, and controller settings belong in
  metadata.
- Directory names should only carry phase, case, and attempt. Prefer `tryNN`
  leaves. Put long explanations in `manifest.txt`, `summary.json`, or the log.
- Put job id, hostname, timestamp, command, and checkpoint details in
  `manifest.txt` / logs, not in the run directory name.
- Active Phase 03 wrappers now reject new long run names containing `p03_`,
  date strings, host/job metadata, or `full_pipeline_<details>`. Fix
  `RUN_GROUP` / `RUN_NAME` before launch rather than accepting an unreadable
  path.
- Do not repeat common prefixes in every artifact filename. Inside a run
  directory, use role names like `summary.json`, `action_trace.json`,
  `videos/raw.mp4`, and `videos/annotated.mp4`.
- For repeated files, group by directory and keep leaf filenames generic:
  `cosmos_policy/pre/00/prefix_rgb.mp4`,
  `cosmos_policy/pre/00/actions/sample.json`,
  `cosmos_policy/post/00/outputs/sample/vision.mp4`. Put step/frame/job
  details in JSON metadata, not in the filename.
- For replay / visual review, use folders for the view and generic leaves:
  `review/oblique/raw.mp4`, `review/oblique/annotated.mp4`,
  `review/oblique/trace.json`.

## Error Record

- [x] Document the old snap / suction error in
      `docs/oracle_snap_error_review.md`.
- [x] Mark old geometric final-seat and source-state Oracle videos invalid as
      physical success.
- [x] Add current literature review for controller-facing world-model bridge:
      `docs/controller_facing_world_model_research_20260702.md`.

## Workspace Rule

- [ ] Put new experiment outputs under `experiments/maniskill/runs/<phase>/`.
- [ ] Put new logs under `logs/<phase>/`.
- [ ] Move any experiment proven useless or invalid under
      `/public/home/yanhongru/ICLR2027/archive/Reflex/`, preserving the original
      repository-relative structure.
- [ ] Before reporting success, inspect wrapper, manifest, summary, relevant
      Python path, action trace, and video for `set_pose`, `set_state`,
      `set_state_dict`, source-state restore, saved-state replay, future
      labels, geometric final placement, hand-selected suffixes, and
      discontinuous pose jumps.
