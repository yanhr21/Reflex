# Phase 03 TODO: Oracle Full Pipeline

## Current Stop / Visual Index

Status as of `2026-07-06T17:10:47+08:00`: overall Phase 03 Oracle is not
complete. The active run tree contains only two accepted validation-key
single-case successes, `h5_continuous_insert/try04` and
`h5_continuous_insert/try11`; superseded diagnostics / failed candidates have
been moved to `/public/home/yanhongru/ICLR2027/archive/Reflex/`. New attempts
still must be reviewed against the active-insertion standard before they can
count.
User review on 2026-07-06: the continuous-insert cases can stand as single-case
references, but they are not the next useful search direction because the
videos still have a target/hole-moving-toward-peg feel. Do not spend the next
attempt on more continuous-insert cases; prioritize other types, starting with
`forward_backward_target_motion`.
This is now enforced by launcher guard, not just documentation:
`phase03_h5_source.sh` and the shared full-pipeline wrapper refuse
`hole_late_continuous_insert_*` / `h5_continuous_insert` while the gate's next
required coverage is forward/backward, unless explicitly overridden with
`ALLOW_CONTINUOUS_INSERT_WHILE_NEXT_COVERAGE_MISSING=true` for a
diagnostic-only run. The guard was checked and exits `48` before creating run
artifacts.
The generic H5 launcher now also refuses other non-next case families. While
the gate says `forward_backward_target_motion`, `h5_fastshift` is rejected with
`refusing_non_next_phase03_coverage=true` / exit `49`; only `h5_reverse` and
`h5_move_stop` are allowed by default.

Current next required coverage:

- Gap:
  `forward_backward_target_motion`
- Prepared approved source key:
  `hole_late_reverse_seed1040038_idx0004`
- Intended active output:
  `experiments/maniskill/runs/03_oracle/h5_reverse/try21/`
- Probe output:
  `experiments/maniskill/runs/03_oracle/render_probe/fwdback21/`
- Status:
  no `p03_fb21` job is running or pending, no `try21` / `fwdback21` artifact
  exists, and the latest Slurm `--test-only` estimate for 4 CPU / 32G / 1 GPU /
  1.5h, after excluding known render-bad nodes, is 2026-07-08 17:13:05 on
  `server44`. Smaller resource test-only
  requests did not improve that estimate. Other visible GPU partitions were
  also unavailable or invalid for this account/request (`test`, `gaosh`,
  `engram`, `long`, `mgpu`, `debug`, `gpux`), so no unattended pending Oracle
  job should be left in the queue.
- Latest immediate allocation attempt:
  Slurm allocation `168276` (`p03_fb21`) on 2026-07-06 16:24 CST stayed
  priority-pending and was canceled before node assignment (`None assigned`,
  elapsed `00:00:00`). It produced no render probe, no full Oracle run, and no
  log artifact.
- Latest scheduler precheck:
  `scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh` now runs a
  default Slurm `--test-only` guard before creating tmux / `salloc`. On
  2026-07-06 16:27 CST, test-only job `168283` estimated
  2026-07-08 15:42:33 on `server27`, so the launcher refused with
  `refusing_far_scheduler_test_only=true` / exit `43`. This created no tmux
  session, allocation, render probe, full Oracle run, or log artifact.
- Latest render-node guard:
  the same launcher now excludes known render-bad nodes by default for
  render-bearing Oracle attempts:
  `server02,server21,server27,server28,server30,server39,server53,server57`. On
  2026-07-06 16:30 CST, precheck job `168292` with that exclusion list
  estimated 2026-07-08 16:45:05 on `server44`; the launcher refused with
  `refusing_far_scheduler_test_only=true` / exit `43` before creating tmux,
  Slurm allocation, render probe, full Oracle run, or log artifact. `server44`
  is intentionally not excluded because earlier evidence includes successful
  render canary / full-protocol runs there.
- `server21` was added to this default exclusion list on 2026-07-06 after the
  status helper briefly estimated it and existing docs showed a prior
  `vk::DeviceLostError` render failure there.
- Latest status-helper alignment:
  `scripts/world_model/phase03_next_coverage_status.sh` now uses the same
  default exclude list for `INCLUDE_SCHEDULER_TEST=true`, so read-only status
  cannot report an unexcluded bad-node estimate while the launcher would use a
  different node set. Its 2026-07-06 16:32 CST test-only job `168305`
  estimated 2026-07-08 17:13:05 on `server44`.
- Latest machine-readable scheduler fields:
  `scripts/world_model/phase03_next_coverage_status.sh` and
  `scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh` now parse
  Slurm test-only output into `scheduler_test_job`,
  `scheduler_estimated_start`, and `scheduler_estimated_node`. A 2026-07-06
  16:34 CST launcher precheck reported `scheduler_test_job=168309`,
  `scheduler_estimated_start=2026-07-08T17:13:05`, and
  `scheduler_estimated_node=server44`, then refused with exit `43` before
  creating tmux, Slurm allocation, render probe, full Oracle run, or log
  artifact.
- Latest post-`server21` exclusion status-helper precheck:
  `scheduler_test_job=168314`,
  `scheduler_estimated_start=2026-07-08T17:14:05`,
  `scheduler_estimated_node=server44`, and
  `scheduler_within_delay_threshold=false`.
- The status helper now accepts `PARTITION`, `JOB_NAME`, `TIME_LIMIT`,
  `CPUS_PER_TASK`, and `MEMORY` so resource probes can be audited with the
  same field names as the launcher. A 2026-07-06 16:40 CST default-resource
  check reported `scheduler_test_job=168319`,
  `scheduler_estimated_start=2026-07-08T17:13:05`, and
  `scheduler_estimated_node=server44`. A reduced 2 CPU / 12G / 45min check
  reported `scheduler_test_job=168320` with the same estimated start and node,
  so reducing resources did not improve availability.
- The readiness helper now checks exclude-list consistency between the tmux
  launcher and status helper, and rejects caller overrides that drift from the
  default. A negative `EXCLUDE_NODES=server44` check exits `51` with
  `reason=requested_exclude_nodes_do_not_match_default`. With
  `INCLUDE_SCHEDULER_TEST=true`, readiness now uses the same exclude list and
  reported test-only job `168321`, estimated 2026-07-08 17:13:05 on
  `server44`.
- The readiness scheduler-test path now also parses and prints
  `scheduler_test_job`, `scheduler_estimated_start`,
  `scheduler_estimated_node`, `scheduler_delay_seconds`, and
  `scheduler_within_delay_threshold`. Its 2026-07-06 16:45 CST test-only job
  `168325` estimated 2026-07-08 17:16:05 on `server44` and reported
  `scheduler_within_delay_threshold=false`.
- The next-coverage status helper now emits a combined launch decision:
  `phase03_forward_backward_launch_allowed` and
  `phase03_forward_backward_launch_block_reasons`. With scheduler test enabled
  on 2026-07-06 16:47 CST, it reported no prepared artifacts and no same-name
  Slurm job, but the latest test-only job `168330` estimated
  2026-07-08 17:16:05 on `server44`; launch is therefore blocked only by
  `scheduler_delay_exceeds_threshold`. Without scheduler test, launch
  permission is reported as `unknown`, not implicitly allowed.
- The tmux launcher now consumes that combined status gate before launching, so
  it cannot create tmux / `salloc` when the
  machine-readable launch decision is false. A 2026-07-06 16:49 CST launcher
  check saw test-only job `168332`, estimated 2026-07-08 17:12:05 on
  `server44`, and refused with `refusing_status_launch_gate=true` / exit `44`
  before creating tmux, Slurm allocation, render probe, full Oracle run, or
  log artifact.
- `AGENTS.md` now makes this launch gate mandatory for Phase 03
  forward/backward launchers: do not create tmux / `salloc` when
  `phase03_forward_backward_launch_allowed` is `false` or `unknown`; report the
  block reasons instead. Latest 2026-07-06 16:51 CST status:
  `scheduler_test_job=168334`, estimated 2026-07-08 17:10:05 on `server44`,
  launch allowed false, block reason `scheduler_delay_exceeds_threshold`.
- The launch gate is now factored into executable helper
  `scripts/world_model/require_phase03_forward_backward_launch_allowed.sh`.
  The tmux launcher calls that helper rather than embedding a separate copy of
  the status parsing logic. A 2026-07-06 16:54 CST launcher check reported
  test-only job `168338`, estimated 2026-07-08 17:10:05 on `server44`, and
  exited `44` with `phase03_forward_backward_launch_required_ok=false` before
  creating tmux, Slurm allocation, render probe, full Oracle run, or log
  artifact.
- The static protocol scan and forward/backward readiness helper now include
  the launch-gate helper in their checked file lists. The static scan also
  verifies that the tmux launcher calls
  `require_phase03_forward_backward_launch_allowed.sh`, and now reports
  `checked_files=9`. Latest 2026-07-06 16:56 CST status:
  `scheduler_test_job=168341`, estimated 2026-07-08 18:13:56 on `server44`,
  launch still blocked by `scheduler_delay_exceeds_threshold`.
- The launcher no longer exposes a `STATUS_LAUNCH_GATE=false` bypass or the old
  scheduler-only fallback path; it unconditionally calls
  `require_phase03_forward_backward_launch_allowed.sh`. A 2026-07-06 16:58 CST
  launcher check reported test-only job `168345`, estimated
  2026-07-08 20:17:10 on `server63`, and exited `44` with
  `phase03_forward_backward_launch_required_ok=false` before creating tmux,
  Slurm allocation, render probe, full Oracle run, or log artifact.
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

Accepted single-case evidence:

- Run:
  `experiments/maniskill/runs/03_oracle/h5_continuous_insert/try04/`
- Source key:
  `hole_late_continuous_insert_seed10241044_idx5004`
- Visuals:
  `videos/annotated.mp4`, `videos/raw.mp4`,
  `review/sheets/annotated_review_sheet.png`,
  `review/sheets/raw_review_sheet.png`
- Verdict:
  `visual_review_verdict.json` records
  `validation_key_single_case_success_confirmed=true`,
  `active_robot_insertion_confirmed=true`,
  `target_assisted_insertion_rejected=true`,
  `overall_task_complete=false`, and `method_evidence_allowed=false`.
- Run:
  `experiments/maniskill/runs/03_oracle/h5_continuous_insert/try11/`
- Source key:
  `hole_late_continuous_insert_seed1040084_idx0006`
- Visuals:
  `videos/annotated.mp4`, `videos/raw.mp4`,
  `review/annotated_keyframes.jpg`, `review/raw_keyframes.jpg`
- Verdict:
  `review/visual_review_verdict.json` records
  `validation_key_single_case_success_confirmed=true`,
  `validation_key_full_source_trajectory_success_confirmed=true`,
  `active_robot_insertion_confirmed=true`,
  `target_assisted_insertion_rejected=true`,
  `overall_task_complete=false`, and `method_evidence_allowed=false`.

Archived Oracle diagnostic evidence that must not be promoted to full source
trajectory success:

- Run:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_continuous_insert/try09/`
- Source key:
  `hole_late_continuous_insert_seed1040042_idx0001`
- Diagnostic:
  frozen-finisher Oracle upper bound. The run preserves reset, DP prefix,
  causal target-motion trigger, 4 premotion Cosmos reports, 1 postmotion Cosmos
  report, and 4 Cosmos-derived dynamic actions, then freezes target/hole motion
  during the DP finisher to prevent the target-assisted failure mode.
- Verdict:
  `visual_review_verdict.json` records
  `active_robot_insertion_confirmed=true`,
  `target_assisted_insertion_rejected=true`,
  `validation_key_frozen_finisher_diagnostic_success_confirmed=true`,
  `validation_key_full_source_trajectory_success_confirmed=false`,
  `overall_task_complete=false`, and `method_evidence_allowed=false`.

Rejected / negative evidence that must not be counted as success:

- `h5_continuous_insert/try06`: simulator metric true, but visual review
  rejected it because target/hole motion appears to create insertion by moving
  onto the held peg / wooden stick before a clearly confirmed robot-driven
  insertion.
- `h5_continuous_insert/try07`: approved key
  `hole_late_continuous_insert_seed1040042_idx0001`, simulator metric true,
  4 premotion Cosmos reports, 1 postmotion Cosmos report, 4 Cosmos-derived
  dynamic actions, original DP finisher, `artifact_audit.ok=true`, and no
  snap. Visual review rejected it as another target-assisted counterexample:
  the target block / hole continues moving onto the held peg during the metric
  success window, with target cumulative motion around
  `[0.0753, -0.2217, 0.0]` by the success frame. It must not increase the
  accepted success count. It is archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_continuous_insert/try07/`.
- `h5_continuous_insert/try08`: no-rollout wrapper argument failure while
  introducing the frozen-finisher diagnostic switch. It produced no summary,
  action trace, or video and is archived as infrastructure / wrapper evidence
  only.
- `h5_continuous_insert/try10`: approved key
  `hole_late_continuous_insert_seed1040084_idx0006`, strict
  target-motion-complete gate enabled, but 6 postmotion Cosmos rounds produced
  only 48 Cosmos-derived dynamic actions while the source key requires
  53 target-motion steps. It ended with
  `target_motion_complete_before_finisher=false`, final `peg_head_l2` about
  `0.0980m`, and no finisher success. It is archived as a configuration
  shortfall. `h5_continuous_insert/try11` corrected this and is listed under
  accepted single-case evidence.
- `h5_continuous_insert/try12`: approved key
  `hole_late_continuous_insert_seed12241022_idx5591`, strict
  target-motion-complete gate enabled, 50 Cosmos-derived dynamic actions, and
  original DP finisher. The protocol completed without snap and source target
  motion was complete before finisher, but physical insertion failed: best
  finisher `peg_head_l2` was about `0.0866m`, final was about `0.1274m`, and
  simulator success stayed false. It is archived as negative evidence. Do not
  rerun this key with the same DP-finisher family; only a clearly labeled
  more-Oracle physical-action finisher diagnostic is justified.
- `h5_continuous_insert/try13`: same approved key as `try12`, same strict
  complete-motion protocol, but `manual_staged_pose_servo` as a more-Oracle
  physical-action finisher. It also failed: 50 Cosmos-derived dynamic actions,
  no snap, 500 manual-finisher steps, best `peg_head_l2` about `0.0963m`, and
  final about `0.3026m`. This rules out the simple diagnosis that `try12`
  failed only because the original DP finisher was too weak. Stop unattended
  retries on this key unless the next change is a new controller interface or
  state-estimation diagnosis, not another gain/threshold/step sweep.
- `h5_continuous_insert/try14`: approved key
  `hole_late_continuous_insert_seed12241047_idx5594`, real
  `cosmos3_policy` dynamic control, 4 premotion Cosmos reports,
  5 postmotion Cosmos reports, 33 Cosmos-derived dynamic actions, and
  `artifact_audit.ok=true`. It reached `simulator_success_metric=true` at
  step 117 with final `peg_head_l2` about `0.0131m`, but visual review rejects
  it as strict success. The success frame occurs before target motion
  completes, no finisher starts, and reviewed keyframes show the target block /
  hole continuing to move onto the held peg. `visual_review/visual_review_verdict.json`
  records `target_assisted_insertion_detected=true` and
  `strict_success_confirmed=false`. This is a target-assisted metric-true
  counterexample, not additional accepted multi-key coverage.
- `h5_move_stop/try16`: approved key
  `hole_late_move_stop_seed17280909_idx8226`, selected as a fresh negative-Y
  move-stop coverage diagnostic. It completed the corrected full pipeline with
  4 premotion Cosmos reports, 3 postmotion Cosmos reports, 23 real
  `cosmos3_policy` dynamic actions, target motion completed before the finisher,
  and 530 original-DP finisher rows. Artifact audit passed and
  `snap_detected=false`, but it failed physical insertion:
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  best finisher `peg_head_l2` about `0.0954m`, and final about `0.1321m`.
  Treat this as full-protocol negative evidence, not success and not overall
  coverage completion.
- `render_probe/movestop17` and `render_probe/movestop17b`: attempted a
  more-Oracle `source_h5_teacher_dynamic` upper-bound diagnostic on the same
  approved move-stop key after `try16` failed with real Cosmos actions. Both
  stopped before rollout. `movestop17` landed on `server02` and timed out at
  `render_gym_start`; `movestop17b` landed on `server21` and failed with
  `vk::DeviceLostError`. No `h5_move_stop/try17*` summary, action trace,
  Cosmos output, or video exists. These are render-infrastructure failures
  only, not Oracle physical attempts.
- `action_diag/try12`: read-only diagnostic on archived `h5_move_stop/try16`,
  run inside Slurm job `167963` / `server02`. It compared the 23 executed
  Cosmos dynamic rows against matching source-H5 teacher rows without
  executing teacher actions. Result: 7D RMSE about `0.1352`, xyz RMSE about
  `0.1264`, dynamic-stage L2 improved only from about `0.1093m` to
  `0.1048m`, and mean xyz sign agreement was about `[0.13, 0.48, 0.30]`.
  This is diagnostic-only evidence that the move-stop failure includes a
  Cosmos action-interface / action-selection mismatch. Do not respond by
  sweeping DP-finisher gains on this key.
- Do not run the current `source_motion_sign` guard as the next move-stop
  rollout without a stronger diagnosis. Inspection of `action_diag/try12`
  found that 16 of 23 rows have negative target-y motion but positive H5
  teacher y action, and 14 rows have both positive Cosmos y and positive
  teacher y under negative target-y motion. The current guard would therefore
  clip or flip many actions that are actually aligned with the teacher.
- A text-only calculation from the existing `try12` JSON shows the current
  source-motion guard is counterproductive on this key: executed xyz RMSE is
  about `0.1264`, simulated `clip_opposite` is about `0.1389`, and simulated
  `rectify_opposite` is about `0.1565`. Future-label per-axis xyz gain
  `[1.1603, 2.0462, 0.5331]` only reduces xyz RMSE to about `0.1220`, so
  simple scaling is weak headroom rather than a concrete next full-rollout
  fix.
- `scripts/slurm/run_phase03_oracle_full_pipeline_in_allocation.sh` now refuses
  `COSMOS_ACTION_DIRECTION_GUARD=source_motion_sign` for `h5_move_stop` /
  `hole_late_move_stop_*` by default, before creating new run artifacts. This
  prevents the known-bad guard from becoming another blind full rollout.
  Override only with
  `ALLOW_UNTRUSTED_MOVE_STOP_SOURCE_MOTION_GUARD=true`, and then label the run
  diagnostic-only with `method_evidence_allowed=false`.
- `scripts/world_model/analyze_phase03_oracle_action_interface.py` now
  includes a read-only future-label adapter-candidate diagnostic for raw
  Cosmos, executed trace, target-motion-sign guard simulations, and simple
  per-axis gain headroom. It also now sweeps nearby source-H5 teacher temporal
  offsets, so the next diagnostic can separate action-interface failure from
  teacher-index alignment failure. It must be run only inside a tmux-held
  Slurm allocation and cannot count as method evidence or Oracle success.
  Use `scripts/slurm/phase03_action_diag.sh` for the next run rather than the
  legacy try-numbered launcher.
  Exact next command, from inside a tmux-held Slurm allocation and then an
  `srun` step, is:

  ```bash
  RUN_DIR=/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_move_stop/try16 \
  OUT_DIR=/public/home/yanhongru/ICLR2027/Reflex/experiments/maniskill/runs/03_oracle/action_diag/try13 \
  LOG_FILE=/public/home/yanhongru/ICLR2027/Reflex/logs/03_oracle/action_diag/try13.log \
  OFFSET_WINDOW=16 \
  scripts/slurm/phase03_action_diag.sh
  ```

- `action_diag/try13` completed this read-only diagnostic on Slurm job
  `167982` / `server02` using archived `h5_move_stop/try16`. It wrote
  `manifest.txt`, `action_interface_diagnostic.json`, and `classification.txt`.
  It did not run rollout, did not execute teacher actions, and is not Oracle
  success.
- `try13` result: the best source-H5 teacher temporal offset is `-9`. At
  offset `0`, executed-vs-teacher xyz RMSE is about `0.1264` and 7D RMSE is
  about `0.1352`; at offset `-9`, xyz RMSE improves to about `0.0671` and 7D
  RMSE improves to about `0.0773`. Mean xyz sign agreement improves from about
  `[0.13, 0.48, 0.30]` to about `[0.65, 0.83, 0.48]`. Next work should inspect
  and fix action-index alignment in the dynamic stage / Cosmos action chunk
  extraction before any new move-stop full rollout.
- The action extractor and full-pipeline wrapper now expose
  `COSMOS_ACTION_ROW_OFFSET` / `--cosmos-action-row-offset`. Default `0`
  preserves the old behavior. Because `try13` shows extracted row `t` resembles
  teacher row `t-9`, the next move-stop diagnostic full rollout candidate is
  `COSMOS_ACTION_ROW_OFFSET=9` so control executes rows farther ahead in the
  Cosmos-predicted action sequence. This is a diagnostic alignment adapter
  derived from future-label analysis, not method evidence, until validated
  across additional approved keys.
- Exact next diagnostic full-rollout command, only from inside a tmux-held
  Slurm allocation and `srun` step after a render-capable GPU is available:

  ```bash
  scripts/slurm/phase03_move_stop_rowoffset_probe.sh
  ```

  This run must remain `method_evidence_allowed=false`; it is a diagnostic
  check of Cosmos action-row alignment, not validation success by itself. The
  probe wrapper runs a render canary first and only launches
  `phase03_move_stop_rowoffset.sh` after a visible GPU renders successfully.
  The short launcher now refuses non-`hole_late_move_stop_*` keys and records
  `COSMOS_ACTION_ROW_OFFSET_SOURCE=action_diag_try13_best_teacher_temporal_offset_minus9`
  in the full-pipeline manifest / summary so the future-label origin of the
  adapter is visible during audit.
  The Phase 03 artifact audit now requires explicit
  `--allow-diagnostic-action-row-offset` for nonzero row-offset runs. With that
  override, audit expects a nonzero offset, an `action_row_offset_source`, and
  `validation_key_success_allowed=false`; without the override, row-offset
  diagnostic artifacts fail audit. The audit also now requires row-level
  dynamic-stage evidence: every `cosmos_dynamic_control` row must record the
  offset, offset source, chunk start/end, raw chunk start, and actual predicted
  action row index, and that row index must equal
  `chunk_start + cosmos_action_index`.
  The full-pipeline wrapper now automatically passes this audit override when
  `COSMOS_ACTION_ROW_OFFSET != 0`, and records
  `audit_allow_diagnostic_action_row_offset=true` in the wrapper manifest/log.
- Attempted to launch this row-offset diagnostic via render-probe-gated Slurm
  allocation `167985` (`p03_mv17`), but the GPU allocation remained pending
  with reason `Priority` and was canceled before compute. No
  `render_probe/move17` or `h5_move_stop/try17` artifacts were created, no
  render canary ran, and no Oracle rollout started.
- Added `scripts/slurm/launch_phase03_move_stop_rowoffset_probe_tmux.sh`, a
  guarded tmux launcher that uses `salloc --immediate` and cancels any same-name
  pending leftover. A smoke launch with `IMMEDIATE_SECONDS=5` created Slurm
  allocation `168000` (`p03_mv17`) but it was canceled before assignment
  (`None assigned`, elapsed `00:00:00`). No `render_probe/move17` or
  `h5_move_stop/try17` artifacts were created.
- A second immediate launch on 2026-07-06 13:30 CST created allocation
  `168016` (`p03_mv17`) but also stayed pending with reason `Priority` and was
  canceled before assignment (`None assigned`, elapsed `00:00:00`). No
  `render_probe/move17`, `h5_move_stop/try17`, or matching logs were created.
  The current test-only scheduler estimate for the same 4 CPU / 32G / 1 GPU /
  1.5h request is 2026-07-08 04:08:15 on `server39`; shorter
  CPU/memory/walltime probes and node-specific probes were no better. Keep
  using immediate launches or an already-held allocation rather than leaving
  this diagnostic queued unattended.
- A 2026-07-06 13:36 CST retry of the same immediate launcher created
  allocation `168043` (`p03_mv17`), again stayed pending with reason
  `Priority`, and was canceled before assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/move17`, `h5_move_stop/try17`, or matching
  logs were created. The latest test-only estimate for the same request is now
  2026-07-08 07:13:53 on `server30`; no new pending Oracle job was launched.
- `h5_fastshift/try05`: approved key
  `hole_late_fast_shift_seed10300001_idx5000`, strict target-motion-complete
  gate enabled, 29 Cosmos-derived dynamic actions, and
  `manual_staged_pose_servo` physical-action finisher. It still failed:
  target motion completed before finisher, but the 500-row manual finisher only
  reached best `peg_head_l2` about `0.1305m` and ended around `0.3587m`. Stop
  unattended fast-shift retries on this key unless a new diagnosis changes more
  than finisher choice, gains, thresholds, or step budget.
- `render_probe/fast06`: attempted to start a new approved fast-shift key,
  `hole_late_fast_shift_seed10300253_idx5010`, as the next multi-key coverage
  attempt with the standard full three-stage protocol and original DP finisher
  after source-H5 target-motion-complete gating. Slurm job `167534` landed on
  `server53`, but the render canary timed out at `render_rgb_array_start`
  before any Oracle rollout. No `h5_fastshift/try06` summary, action trace,
  Cosmos output, or video was produced. This is infrastructure evidence only,
  archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/render_probe/fast06/`
  and
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/render_probe/fast06.log`.
  A follow-up server44-specific allocation request, Slurm job `167535`, stayed
  pending for priority and was canceled before any compute step. The fast-shift
  key remains untested; do not count this as Oracle evidence or a failed
  physical insertion attempt.
- `render_probe/fast06b`: second attempt to start the same new fast-shift key
  through the same probe-backed launcher. A server63-specific allocation,
  Slurm job `167538`, stayed priority-pending and was canceled. A mistaken
  multi-node nodelist request, Slurm job `167539`, was immediately canceled
  before compute. The corrected single-node request excluding known bad render
  nodes landed on `server46` as Slurm job `167540`, but the render canary again
  timed out at `render_rgb_array_start`. No Oracle rollout started and no
  `h5_fastshift/try06` artifacts exist. The probe/log were archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/render_probe/fast06b/`
  and
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/render_probe/fast06b.log`.
  Treat this as infrastructure evidence only; the key remains untested.
- `render_probe/fast06c`: third probe-backed attempt for the same key. A
  smaller server44 request, Slurm job `167547`, stayed priority-pending and
  was canceled after the node became fully allocated. A server39 request,
  Slurm job `167549`, then landed on bus `2C:00`, but the default
  `render_rgb_array` canary failed with `vk::DeviceLostError` at
  `camera.take_picture()`. No Oracle rollout started and no
  `h5_fastshift/try06` artifacts exist. The probe/log were archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/render_probe/fast06c/`
  and
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/render_probe/fast06c.log`.
  Current live resource state during this attempt had server57/server02/server63
  fully allocated and server44 becoming fully allocated before the pending job
  could start. Treat this as infrastructure evidence only; the new fast-shift
  key remains untested.
- `render_probe/fast06d`: short multi-GPU server39 render probe only, with no
  full-run target configured. Slurm job `167551` exposed CUDA devices `0,1,2`
  with physical buses `2C:00`, `9A:00`, and `9B:00`. The default
  `render_rgb_array` canary timed out at `render_rgb_array_start` on all three
  visible devices (`exit_code=124`). No Oracle rollout started and no
  `h5_fastshift/try06` artifacts exist. The probe/log were archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/render_probe/fast06d/`
  and
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/render_probe/fast06d.log`.
  Treat this as render-infrastructure evidence only. The new fast-shift key
  remains untested until a GPU/node can pass the same `render_rgb_array` canary
  required by the full pipeline.
- Resource check after `fast06d`: nodes with actually free GPUs were
  `server28`, `server30`, `server39`, and `server53`; all have recent
  render-canary failure evidence in this protocol window. `server63` had
  positive historical evidence for `default + render_rgb_array`, but currently
  had all eight GPUs allocated. Do not bypass the render canary or launch
  `h5_fastshift/try06` on a known-failing node just to get another attempt.
- `render_probe/fast06e`: a canary-gated attempt was first queued for `server63`
  as Slurm job `167561`, but that historical positive node had no free GPU and
  the pending allocation was canceled before compute. The follow-up short
  allocation landed on `server30` as Slurm job `167563`, bus `29:00`, using the
  exact full-pipeline canary path `RENDER_SHADER_PACK=default` and
  `RENDER_CANARY_API=render_rgb_array`. The canary timed out at
  `render_rgb_array_start` with `exit_code=124`; therefore the automatic
  `h5_fastshift/try06` full run did not start. The probe/log were archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/render_probe/fast06e/`
  and
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/render_probe/fast06e.log`.
  Treat this as render-infrastructure evidence only. The approved key
  `hole_late_fast_shift_seed10300253_idx5010` remains untested by Oracle.
- `render_probe/fast06g` / `h5_fastshift/try06`: successful canary-gated launch
  on Slurm job `167574` / `server44`, bus `AA:00`. The probe used
  `default + render_rgb_array`, rendered and wrote `frame.png`, then started the
  approved key `hole_late_fast_shift_seed10300253_idx5010` as `h5_fastshift/try06`.
  The full pipeline completed with the source-H5 protocol, 4 premotion Cosmos
  reports, 6 postmotion Cosmos reports, 48 Cosmos-derived dynamic actions,
  `artifact_audit.ok=true`, and no peg state intervention. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  `near_target_before_finisher=false`, `finisher_start_step=null`, and final
  `peg_head_l2` was about `0.3285m`. This is useful negative full-protocol
  evidence on the new fast-shift key, not a successful sample and not completion
  of directional coverage. The run/probe and logs were archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_fastshift/try06/`,
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/render_probe/fast06g/`,
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/h5_fastshift/try06.log`,
  and
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/render_probe/fast06g.log`.
  Immediate diagnosis: the target-motion trigger was valid with live
  `peg_head_l2` about `0.1503m`, but the first Cosmos dynamic action worsened
  it to about `0.1735m` and the final state was about `0.3285m`. The failure is
  in the Cosmos dynamic action interface / selection before near-target handoff,
  not in DP finisher tuning.
- `action_diag/try11`: read-only diagnostic on archived `h5_fastshift/try06`,
  run inside Slurm job `167587` / `server02`. It compared the 48 executed
  Cosmos dynamic actions against the matching source-H5 teacher action rows
  without executing teacher actions. Result: 48 valid rows, 7D RMSE about
  `0.0622`, xyz RMSE about `0.0897`, dynamic-stage L2 increased by about
  `0.1550m` (`0.1735m` to `0.3285m`), and y sign agreement was about `-0.542`.
  This confirms the new fast-shift failure is an action-interface / selection
  problem in the Cosmos dynamic stage. It is diagnostic only and not success
  evidence.
- `h5_fastshift/try07`: same approved key, explicitly labeled
  `source_h5_teacher_dynamic` future-label upper-bound diagnostic. It completed
  4 premotion Cosmos reports, 5 postmotion Cosmos reports, 39 teacher dynamic
  rows, original DP finisher from step 150, videos, no snap, and no peg state
  intervention. It failed: simulator success false, visual full insertion
  false, final `peg_head_l2` about `0.1167m`. The best DP-finisher frame was
  much closer, about `0.0162m`, but still did not satisfy the simulator success
  metric or become accepted visual insertion. Archive path:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_fastshift/try07/`.
- `h5_fastshift/try08`: same approved key and teacher dynamic diagnostic, but
  with `source_h5_teacher_suffix` as the near-target finisher. It failed with
  `classification=diagnostic_future_label_teacher_suffix_exhausted_without_success`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  `finisher_start_step=151`, best `peg_head_l2` about `0.1138m`, and final
  about `0.1202m`; source action index `300` exhausted before success. This
  rules out the simple idea that replaying the validation H5 suffix from the
  current live state is enough. Archive path:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_fastshift/try08/`.
- `h5_fastshift/try09`: same approved key and teacher dynamic diagnostic, with
  `dp_then_manual_close` configured to hand off near `0.03m`. It reached
  simulator metric true at step 165 and final `peg_head_l2` about `0.0087m`,
  but is rejected as strict success:
  `classification=diagnostic_future_label_teacher_dynamic_metric_true_not_success`,
  `visual_full_insertion_confirmed=false`, `physical_insertion_success_claimed=false`,
  and `target_assisted_insertion_must_be_rejected=true`. Action trace counts:
  110 `diffusion_policy` prefix rows, 1 target-motion trigger row,
  39 `source_h5_teacher_dynamic_future_label_diagnostic` rows, and 16 DP
  finisher rows. Manual close did not actually take over. Archive path:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_fastshift/try09/`.
- `h5_fastshift/try10`: same approved key and teacher dynamic diagnostic, with
  a widened `dp_then_manual_close` handoff
  (`MANUAL_DP_TO_MANUAL_L2=0.06`, `MANUAL_SOFT_INSERT_THRESHOLD=0.05`),
  `TARGET_MOTION_DURING_FINISHER=false`, and `MANUAL_YAW_ACTION=0.0`. It again
  reached simulator metric true at step 165 and final `peg_head_l2` about
  `0.0116m`, but is rejected:
  `classification=diagnostic_future_label_teacher_dynamic_metric_true_not_success`,
  `visual_full_insertion_confirmed=false`,
  `target_assisted_insertion_must_be_rejected=true`, and
  `physical_insertion_success_claimed=false`. Action trace counts: 110 DP
  prefix rows, 1 target-motion trigger row, 39 future-label teacher dynamic
  rows, and 16 DP finisher rows. There were zero
  `manual_close_after_dp_gate` rows, so the widened handoff still did not force
  active manual insertion. Archive path:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_fastshift/try10/`.
- `h5_fastshift/try11`: same approved key and teacher dynamic diagnostic, but
  with `manual_staged_hole_servo` as the finisher from the near-target gate.
  This forced manual control to run: action trace counts include
  502 `manual_staged_hole_servo_after_near_target_gate` rows and zero DP
  finisher rows after the gate. It still failed:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  best finisher `peg_head_l2` about `0.1326m`, final about `0.1856m`, and
  `snap_detected=false`. This rejects direct staged manual finisher as a
  fast-shift fix on this live state. Archive path:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_fastshift/try11/`.
- `render_probe/fast08suffix` and `render_probe/fast08b`: no-rollout render
  infrastructure failures on `server02` and `server46` while the outer probe
  launcher forced `default + render_rgb_array`. The launcher now allows
  `RENDER_SHADER_PACK` and `RENDER_CANARY_API` overrides and defaults to
  `minimal + gym`; `render_probe/fast08c` passed that canary on `server44` and
  launched `try08`.
- `h5_move_stop/try07`: approved key
  `hole_late_move_stop_seed1080064_idx0000`, complete protocol with
  4 premotion Cosmos reports, 1 postmotion Cosmos report, 4 Cosmos-derived
  dynamic actions, and `manual_staged_hole_servo` finisher, but no insertion.
  It failed because the near-target gate allowed finisher entry while source-H5
  target motion was still early: target cumulative motion was only about
  `[0.0026, 0.0434, 0.0]` at finisher start versus total
  `[0.0133, 0.2243, 0.0]`. The target kept moving away during the manual
  finisher and `peg_head_l2` worsened from about `0.1463m` to `0.3221m`.
  The runner now has `REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER` to keep
  Cosmos dynamic control active until source-H5 target motion completes before
  any DP/manual finisher can start.
- `h5_move_stop/try08`: same approved key with
  `REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER=true`. The gate worked:
  the run executed 30 Cosmos-derived dynamic actions and 4 postmotion Cosmos
  reports, target motion was complete before finisher, and finisher target
  delta rows were all zero. It still failed with `manual_staged_hole_servo`:
  best finisher `peg_head_l2` was about `0.0948m` and final was about
  `0.1434m`. The next move-stop attempt, if any, must be the last unattended
  retry in this controller family and should test the original DP finisher
  after the corrected target-motion-complete gate, not another manual gain
  sweep.
- `h5_move_stop/try09`: final unattended move-stop retry on the same approved
  key. It kept `REQUIRE_TARGET_MOTION_COMPLETE_BEFORE_FINISHER=true`, executed
  30 Cosmos-derived dynamic actions and 4 postmotion Cosmos reports, then used
  the original DP finisher after target motion completed. It still failed:
  best finisher `peg_head_l2` was about `0.0946m` at finisher start and final
  was about `0.1136m`. Stop unattended move-stop retries for this key /
  controller family unless a new diagnosis beyond finisher choice or simple
  gain/step changes is available.
- `h5_move_stop/try10`: new approved move-stop key
  `hole_late_move_stop_seed1080087_idx1760`, selected from the source-H5 audit
  because its validation trajectory succeeds and target motion lasts only
  17 frames. The run completed 32 Cosmos-derived dynamic actions and the
  target-motion-complete gate, but it never reached an effective finisher:
  `cosmos_dynamic_control` worsened `peg_head_l2` from about `0.1832m` best to
  about `0.3054m` final. This is dynamic-control negative evidence, not a
  near-final insertion failure. Do not force a finisher by merely loosening
  `NEAR_TARGET_L2`; the next change needs a Cosmos/action-interface diagnosis.
- `action_diag/try02`: read-only diagnostic on archived `h5_move_stop/try10`.
  It compared executed Cosmos dynamic actions against the matching approved
  source-H5 teacher rows without executing teacher labels. The signs matched,
  but Cosmos translational magnitude was too small, especially y: x was about
  60% of teacher and y about 39% of teacher. This justified only a labeled
  diagnostic action-scale adapter for Cosmos-derived dynamic actions; it is
  not success evidence.
- `h5_move_stop/try11`: same approved key as `try10`, with the diagnostic
  action adapter x `1.6`, y `2.4`, z `0.75`, gripper `1.05`, then original DP
  finisher. The adapter improved dynamic control from trigger `peg_head_l2`
  about `0.1483m` to about `0.0964m` after 16 Cosmos-derived dynamic actions,
  and target motion completed before the finisher. The DP finisher still
  failed after 575 rows: best `peg_head_l2` about `0.0898m`, final about
  `0.1533m`, simulator success false, no snap. Archived negative evidence;
  do not report as success.
- `h5_move_stop/try12`: same adapter and key as `try11`, but
  `manual_staged_hole_servo` as a more-Oracle physical-action finisher. It
  also failed without snap: 16 Cosmos-derived dynamic actions reached about
  `0.0950m`, 520 manual-finisher rows reached best about `0.0897m`, and final
  was about `0.1000m`. Action inspection shows the staged finisher mostly
  corrected y/z but did not sustain forward insertion; `rel_x` stayed near
  `-0.089m` at best.
- `h5_move_stop/try13`: same direct-forward `manual_oracle_servo` diagnostic
  planned from the `try12` action inspection, but server02 failed the render
  canary before rollout. It produced no Oracle action trace or video and is
  archived as infrastructure evidence only.
- `h5_move_stop/try14`: reran that direct-forward `manual_oracle_servo`
  diagnostic on render-capable server57. It completed 16 Cosmos-derived
  dynamic actions, target motion completed before the finisher, and no snap was
  detected. The finisher did command forward motion, but insertion still
  failed: best `peg_head_l2` about `0.0897m`, final about `0.1496m`, simulator
  success false. Stop this move-stop key after `try14`; further work on this
  key should be an explicitly labeled teacher-action replay /
  controller-contract diagnostic only, not another gain/threshold/step sweep
  and not method evidence.
- `h5_move_stop/try15`: that explicitly labeled future-label teacher-action
  suffix diagnostic. It kept the DP prefix, Cosmos RGB/action dynamic stage,
  target-motion-complete gate, and no state restore, then executed source-H5
  action rows through `env.step` from row 126 to row 299. This cannot count as
  success or method evidence because it uses future teacher labels. It still
  failed: best `peg_head_l2` about `0.0887m`, final about `0.1394m`,
  simulator success false, artifact audit ok, no snap. This shows the
  post-Cosmos live state on this key is not rescued by the matching H5 action
  suffix; stop this move-stop key unless a new state-alignment diagnosis is
  introduced.
- `h5_sine/try02`: approved key
  `hole_late_sine_seed1050232_idx0015`, strict target-motion-complete gate
  enabled, but `MAX_COSMOS_ROUNDS=4` only executed 32 Cosmos-derived dynamic
  actions while the source target motion requires 34 steps. The run ended with
  `target_motion_complete_before_finisher=false`, no finisher success, and
  final `peg_head_l2` about `0.1174m`. This is a configuration-shortfall
  negative diagnostic and is archived.
- `h5_sine/try03`: same approved key with `MAX_COSMOS_ROUNDS=5`. This fixed
  the gate issue: 33 Cosmos-derived dynamic actions completed source target
  motion before finisher, and `finisher_allowed_by_target_motion=true`. The
  original DP finisher still failed after 506 rows: best `peg_head_l2` about
  `0.1076m`, final about `0.1458m`, simulator success false.
- `h5_sine/try04`: same strict complete-motion protocol, but with the more
  Oracle `manual_staged_pose_servo` physical-action finisher. It still failed
  after 500 manual finisher rows: best `peg_head_l2` about `0.1071m`, final
  about `0.2254m`, simulator success false. Stop unattended retries for this
  sine key unless a new diagnosis changes the controller interface or state
  estimation, not just gains, thresholds, or step budget.
- `h5_constant/try01`: approved key and metric true, but rejected as
  target-assisted / passive insertion; it is archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/`.
- `h5_reverse/*`: multiple approved-key attempts completed protocol-compliant
  runs with no snap, but no reverse-direction success has been accepted.
- `h5_reverse/try13`: future-label teacher-action suffix diagnostic on
  approved key `hole_late_reverse_seed1040017_idx1250`. It completed 4
  premotion Cosmos reports, 6 postmotion Cosmos reports, 48 Cosmos-derived
  dynamic actions, and the source-H5 target-motion-complete gate, but the
  dynamic stage worsened `peg_head_l2` from about `0.1348m` to about `0.2853m`
  and never reached the near-target gate, so the teacher suffix finisher did
  not start. Artifact audit passed and no snap was detected. `action_diag/try03`
  shows the 48 Cosmos dynamic actions differ strongly from matching source-H5
  teacher rows: 7D RMSE about `0.0803`, xyz RMSE about `0.1144`, L2 delta about
  `+0.1505m`, Cosmos mean y about `0.0288` versus teacher mean y about
  `0.1433`, and late x actions flip sign relative to teacher. This is reverse
  dynamic-action/interface mismatch evidence, not success and not method
  evidence.
- `action_diag/try04` and `action_diag/try05`: read-only diagnostics on the two
  accepted continuous-insert successes. `try04` found 4 dynamic rows with 7D
  RMSE about `0.0296`, xyz RMSE about `0.0088`, sign agreement `[1, 1, 1]`,
  and L2 delta about `-0.0190`. `try05` found 52 dynamic rows with 7D RMSE
  about `0.0464`, xyz RMSE about `0.0418`, and L2 delta about `-0.0038`.
  These diagnostics show that the accepted continuous cases have materially
  better Cosmos/action agreement than reverse `action_diag/try03`; they justify
  only a diagnostic reverse Cosmos-action adapter / direction-guard test, not
  a success claim.
- `h5_reverse/try14`: attempted that reverse diagnostic with Cosmos action
  scales x/y/z `2.0/4.0/1.15` plus
  `COSMOS_ACTION_DIRECTION_GUARD=source_motion_sign` and
  `COSMOS_ACTION_DIRECTION_GUARD_MODE=rectify_opposite`, but server39 hung in
  the render canary at `render_rgb_array_start`. The srun step was manually
  canceled before rollout. No summary, action trace, or video exists; it is
  archived infrastructure evidence only under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_reverse/try14/`.
- `h5_reverse/try15`: reran the same action-interface diagnostic on server44.
  It completed the full no-state-edit protocol with 4 premotion Cosmos reports,
  6 postmotion Cosmos reports, 48 Cosmos-derived dynamic actions, videos,
  `artifact_audit.ok=true`, and target motion complete before any finisher.
  It still failed before the finisher: `near_target_before_finisher=false`,
  `finisher_start_step=null`, simulator success false, dynamic L2 min about
  `0.1024`, and final dynamic L2 about `0.2077`. `action_diag/try06` shows the
  guard/scale overcorrected badly: 7D RMSE about `0.3229`, xyz RMSE about
  `0.3857`, L2 delta about `+0.0676`, and executed y mean absolute about
  `0.747` versus teacher about `0.145`, with clipped y actions at `1.0`.
  Reject `scale_y=4.0` plus rectify-opposite guard; do not continue this
  reverse family with another blind y-scale/sign tweak.
- `h5_reverse/try16`: explicitly labeled future-label
  `source_h5_teacher_dynamic` diagnostic on approved key
  `hole_late_reverse_seed1040017_idx1250`, Slurm job `167425` / `server44`.
  It kept the DP prefix, 4 premotion Cosmos RGB/action reports,
  4 postmotion Cosmos RGB/action reports, target-motion-complete gating, and a
  DP finisher, but executed the 31 dynamic-stage robot actions from matching
  source-H5 teacher rows. The simulator metric became true and the DP finisher
  reached best `peg_head_l2` about `0.0128m`, but this is not success:
  `classification=diagnostic_future_label_teacher_dynamic_metric_true_not_success`,
  `physical_insertion_success_claimed=false`, and artifact audit failed with
  `non_cosmos_action_source_in_dynamic_stage`. `action_diag/try07` is the
  corresponding read-only check and reports zero teacher/executed action error
  by construction, with dynamic L2 decreasing only about `0.0222m` before the
  finisher. Conclusion: exact future teacher dynamic actions can rescue this
  reverse state, so the remaining reverse bottleneck is Cosmos/action selection
  and interface, not another finisher gain sweep. The run/log are archived
  under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_reverse/try16/`
  and
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/h5_reverse/try16.log`.
- `h5_reverse/try17`: same future-label `source_h5_teacher_dynamic`
  diagnostic on a second approved reverse key,
  `hole_late_reverse_seed1040025_idx0003`, Slurm job `167436` / `server02`.
  It produced 4 premotion Cosmos reports, 5 postmotion Cosmos reports,
  39 dynamic teacher-action rows, and target-motion-complete gating before the
  DP finisher. It still failed physically:
  `simulator_success_metric=false`, dynamic L2 improved from about `0.1534m`
  to about `0.0958m`, and the DP finisher ran until episode end with best
  `peg_head_l2` about `0.0902m` and final about `0.1048m`. `action_diag/try08`
  reports zero executed/teacher action error by construction and L2 delta about
  `-0.0576m`. This is not success and not method evidence. It shows the second
  reverse key is not rescued by exact future teacher dynamic actions plus the
  current DP finisher, so do not treat all reverse failures as only a Cosmos
  action-sign/scale problem. The run/log are archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_reverse/try17/`
  and
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/h5_reverse/try17.log`.
- `h5_reverse/try18`: returned to real Cosmos-action dynamic control on
  approved key `hole_late_reverse_seed1040017_idx1250`, Slurm job `167457` /
  `server02`, using the existing conservative
  `COSMOS_ACTION_DIRECTION_GUARD=source_motion_sign` with
  `COSMOS_ACTION_DIRECTION_GUARD_MODE=clip_opposite` and identity action
  scales. It completed 4 premotion Cosmos reports, 6 postmotion Cosmos reports,
  48 Cosmos-derived dynamic actions, videos, and `artifact_audit.ok=true`, but
  failed before finisher entry:
  `near_target_before_finisher=false`, final `peg_head_l2` about `0.2861m`,
  and `simulator_success_metric=false`. `action_diag/try09` reports 7D RMSE
  about `0.0802`, xyz RMSE about `0.1093`, L2 delta about `+0.1463m`, x sign
  agreement about `-0.0625`, y sign agreement about `0.6042`, and y mean
  absolute about `0.0304` versus teacher about `0.1449`. This rejects the
  simple conservative sign-clipping guard as a reverse fix; do not continue
  with another blind sign/scale tweak. The run/log are archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_reverse/try18/`
  and
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/h5_reverse/try18.log`.
- `h5_reverse/try19`: attempted the receding-horizon-1 diagnostic, but
  server28 hung in the render canary at `render_rgb_array_start`. No rollout,
  action trace, summary, or video was produced. It is archived infrastructure
  evidence only, not Oracle evidence.
- `h5_reverse/try20`: reran the receding-horizon-1 diagnostic on
  render-capable server02, Slurm job `167476`. It used real Cosmos dynamic
  actions only, with `COSMOS_ACTION_HORIZON=1`, identity scales, no direction
  guard, 4 premotion Cosmos reports, 40 postmotion Cosmos reports, 40
  Cosmos-derived dynamic actions, videos, and `artifact_audit.ok=true`. It
  still failed before finisher entry: `near_target_before_finisher=false`,
  final `peg_head_l2` about `0.3094m`, and `simulator_success_metric=false`.
  `action_diag/try10` reports 7D RMSE about `0.0952`, xyz RMSE about `0.1303`,
  L2 delta about `+0.1735m`, x sign agreement about `0.25`, y sign agreement
  about `0.10`, and y mean absolute about `0.0389` versus teacher about
  `0.1704`. The first receding action was close to teacher, but from the
  second step onward the live-history Cosmos action output drifted in x/y sign
  and magnitude. This rejects `COSMOS_ACTION_HORIZON=1` as a reverse fix; the
  next step should be a documented bridge/action-interface design rather than
  another chunk-length, sign, or scalar-gain sweep. The run/log are archived
  under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_reverse/try20/`
  and
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/h5_reverse/try20.log`.
- `peg_disturb/try41` through `try47`: repeated controller-family failures.
  `try47` was canceled by user direction before summary/action-trace/full-video
  artifacts were written. Do not launch more peg-disturbance retries without
  explicit user direction and a concrete new diagnosis.
- `peg_drop/try01` through `peg_drop/try03`: approved peg-drop key
  `peg_drop_seed36705002_pseed39705002_idx12420` completed protocol-compliant
  full-pipeline attempts with 4 premotion Cosmos reports, 1 postmotion Cosmos
  report, 4 Cosmos-derived dynamic actions, physical peg-drop perturbation, no
  snap, videos, and `artifact_audit.ok=true`, but none inserted. These runs are
  archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/peg_drop/`.
  `try01` used `manual_align_then_dp`, `try02` used stronger
  `manual_hole_frame_servo`, and `try03` added
  `manual_regrasp_then_hole_servo`. The key diagnostic from `try03` is that the
  TCP moved back near the dropped peg, but the peg position and
  `peg_head_l2` stayed essentially fixed around `0.1957m`; the dropped peg was
  not physically recovered by this controller-action regrasp attempt. Stop this
  peg-drop key after these three failures unless a new controller diagnosis is
  made from the archived videos/action traces.

Missing coverage before overall completion can be marked:

- successful forward/backward target-motion coverage;
- successful left/right target-motion coverage;
- successful peg/wooden-stick disturbance coverage;
- successful coverage across multiple approved `fix3_733` keys.

Read-only completion gate:

- `scripts/world_model/check_phase03_oracle_completion.sh` scans only existing
  visual-review verdict JSON files and prints accepted single-case rows plus
  missing coverage booleans. It must report
  `phase03_oracle_overall_complete=true` before any future response or doc may
  say the overall Oracle task is complete. Current expected status is nonzero /
  incomplete because accepted evidence exists only for continuous-insert cases,
  while forward/backward, left/right, and peg/wooden-stick disturbance success
  coverage is missing.
- Current gate output on 2026-07-06: `accepted_single_case_count=2`,
  `accepted_unique_source_key_count=2`,
  `modern_strict_single_case_count=2`,
  `modern_strict_unique_source_key_count=2`,
  `accepted_h5_continuous_insert_count=2`,
  `modern_strict_h5_continuous_insert_count=2`,
  `modern_strict_forward_backward_group_count=0`,
  `modern_strict_left_right_group_count=0`,
  `modern_strict_peg_disturb_count=0`, and
  `phase03_oracle_overall_complete=false`. The older
  `h5_continuous_insert/try04` verdict has now been modernized with an explicit
  target-assisted rejection review based on its existing review notes, annotated
  sheet, action trace, final distance, simulator metric, and no-snap audit.
  Coverage decisions still must use the `modern_strict_*` rows: both strict
  rows are continuous-insert only, so directional and peg/wooden-stick
  disturbance coverage remain missing. The gate also reports
  `accepted_rows_missing_protocol_artifact_check_count=0`, meaning both strict
  continuous-insert rows have summary / artifact audit / action trace evidence,
  premotion and postmotion Cosmos reports, at least four Cosmos dynamic rows,
  dynamic action sources all `cosmos3_policy_output`, and no snap. The gate
  also reports `accepted_rows_missing_source_h5_check_count=0` and
  `accepted_rows_failing_diagnostic_exclusion_count=0`, so these strict rows
  use approved `fix3_733_source_h5_protocol` keys and are not synthetic,
  future-label, or row-offset diagnostic successes.
- The gate now prints machine-readable coverage gaps. Current output:
  `missing_coverage_items=forward_backward_target_motion,left_right_target_motion,peg_or_wooden_stick_disturbance,multiple_approved_fix3_733_keys`
  and `next_required_coverage_group=forward_backward_target_motion`. The next
  Oracle execution should therefore target a forward/backward case first,
  using a real approved source-H5 key and the full three-stage protocol.
- Prepared the short guarded forward/backward launch path:
  `scripts/slurm/phase03_forward_backward_probe.sh`, with tmux helper
  `scripts/slurm/launch_phase03_forward_backward_probe_tmux.sh`. It defaults
  to approved key `hole_late_reverse_seed1040038_idx0004` under
  `h5_reverse/try21`, runs the render probe first, and then calls
  `scripts/slurm/phase03_forward_backward_next.sh`. The full-run launcher
  refuses non-`hole_late_reverse_*` / non-`hole_late_move_stop_*` keys, refuses
  nonzero `COSMOS_ACTION_ROW_OFFSET`, refuses future-label teacher dynamic /
  suffix modes, and keeps `DYNAMIC_CONTROLLER=cosmos3_policy` for the dynamic
  stage. This is the next real coverage attempt path, not the row-offset
  diagnostic path.
- The forward/backward readiness and full-run launchers now also reject
  attempts that lower `MAX_PREMOTION_COSMOS_PREDICTIONS` or
  `MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER` below `4`. This directly blocks
  the old failure mode where a run could show only early target motion /
  one-off Cosmos preparation and then skip the required Cosmos dynamic-control
  stage before the finisher. The shell-only rejection paths were checked:
  invalid values exit with status `47` before any run artifact is created.
- The completion gate now also requires structured full-sequence video review
  fields in the visual verdict:
  `full_sequence_video_reviewed=true`,
  `video_covers_cosmos_control_and_finisher=true`, and
  `video_covers_final_insertion_or_physical_failure=true`. This means a file
  named `videos/annotated.mp4` is not enough; a target-motion-start /
  boundary-only video remains severe partial evidence and cannot enter the
  modern strict success table.
- The completion gate also now checks first simulator-success timing from the
  action trace. A strict row must first become `live_eval.success=true` inside
  `oracle_physical_dp_finisher` or `oracle_physical_manual_finisher`, after the
  finisher has started. This blocks target-assisted metric-true rows where the
  target / hole creates success before the physical finisher. Current accepted
  rows have `accepted_rows_success_before_finisher_count=0`.
- The completion gate also now checks direct stage order:
  `target-motion increment -> cosmos_dynamic_control -> DP/manual physical
  finisher`, with zero `dp_static_prefix` rows after target motion starts. This
  blocks malformed traces that contain the right stage counts but in a
  protocol-invalid order. Current accepted rows have
  `accepted_rows_bad_stage_order_count=0`.
- Resource check for this forward/backward launcher: `srun --test-only` for
  `p03_fb21` with 4 CPU / 32G / 1 GPU / 1.5h currently predicts
  2026-07-08 15:08:10 on `server63`. No pending `p03_fb21` job is being left,
  and no `render_probe/fwdback21` or `h5_reverse/try21` artifacts exist yet.
  Smaller CPU/memory/walltime requests did not improve the schedule, and
  other visible GPU partitions were unavailable, account-mismatched, or
  rejected the requested configuration.
- A guarded immediate launch attempt on 2026-07-06 13:57 CST created Slurm
  allocation `168069` (`p03_fb21`) but it remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- A second guarded immediate launch attempt on 2026-07-06 14:05 CST created
  Slurm allocation `168078` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- A third guarded immediate launch attempt on 2026-07-06 14:09 CST created
  Slurm allocation `168082` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- A fourth guarded immediate launch attempt on 2026-07-06 created Slurm
  allocation `168084` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- A fifth guarded immediate launch attempt on 2026-07-06 14:52 CST created
  Slurm allocation `168167` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- A sixth guarded immediate launch attempt on 2026-07-06 created Slurm
  allocation `168176` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- A seventh guarded immediate launch attempt on 2026-07-06 15:32 CST created
  Slurm allocation `168210` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- An eighth guarded immediate launch attempt on 2026-07-06 15:38 CST created
  Slurm allocation `168220` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- A ninth guarded immediate launch attempt on 2026-07-06 15:47 CST created
  Slurm allocation `168233` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- A tenth guarded immediate launch attempt on 2026-07-06 15:54 CST created
  Slurm allocation `168241` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- An eleventh guarded immediate launch attempt on 2026-07-06 16:02 CST created
  Slurm allocation `168246` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- A twelfth guarded immediate launch attempt on 2026-07-06 16:10 CST created
  Slurm allocation `168249` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- A thirteenth guarded immediate launch attempt on 2026-07-06 16:14 CST created
  Slurm allocation `168251` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- A fourteenth guarded immediate launch attempt on 2026-07-06 16:20 CST created
  Slurm allocation `168274` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- A fifteenth guarded immediate launch attempt on 2026-07-06 16:24 CST created
  Slurm allocation `168276` (`p03_fb21`) but again remained pending with reason
  `Priority` and was canceled before node assignment (`None assigned`, elapsed
  `00:00:00`). No `render_probe/fwdback21`, `h5_reverse/try21`, or matching log
  artifacts were created.
- The forward/backward tmux launcher now refuses to create another immediate
  allocation when the scheduler's `--test-only` estimate is far beyond the
  local threshold. A 2026-07-06 16:27 CST precheck produced test-only job
  `168283`, estimated 2026-07-08 15:42:33 on `server27`, and exited `43`
  before creating tmux, Slurm allocation, or artifacts.
- The forward/backward tmux launcher now passes
  `--exclude=server02,server21,server27,server28,server30,server39,server53,server57`
  by default to both the scheduler precheck and `salloc`. A 2026-07-06 16:30
  CST precheck produced test-only job `168292`, estimated
  2026-07-08 16:45:05 on `server44`, and exited `43` before creating tmux,
  Slurm allocation, or artifacts.
- Added `scripts/world_model/phase03_next_coverage_status.sh`, a read-only
  status helper that runs the completion gate, prints
  `missing_coverage_items`, `next_required_coverage_group`, the prepared
  forward/backward launcher paths, and whether `render_probe/fwdback21` /
  `h5_reverse/try21` artifacts or logs exist. Current output confirms
  `phase03_oracle_overall_complete=false`,
  `next_required_coverage_group=forward_backward_target_motion`, no
  `p03_fb21` Slurm job, and no `fwdback21` / `try21` artifacts.
  With `INCLUDE_SCHEDULER_TEST=true`, it also prints a Slurm `--test-only`
  estimate for `p03_fb21`; current estimate is 2026-07-08 15:08:10 on
  `server63`, so no unattended pending job should be left.
- Added `scripts/world_model/require_phase03_next_coverage.sh`, and wired it
  into the forward/backward probe, full-run launcher, and tmux helper. Current
  guard output permits the launcher because the completion gate still reports
  `next_required_coverage_group=forward_backward_target_motion`. If the gate
  changes, the stale forward/backward launcher will refuse to run unless
  explicitly bypassed.
- The forward/backward probe/full-run launchers now also refuse to run if the
  target probe/full-run directory or log already exists, unless
  `ALLOW_EXISTING_PHASE03_RUN_DIR=true` is explicitly set. This prevents
  accidental overwrite/reuse of `render_probe/fwdback21` or `h5_reverse/try21`
  artifacts.
- The forward/backward full-run launcher now also refuses diagnostic or
  misclassified coverage settings before it reaches the shared H5 wrapper:
  mismatched `RUN_GROUP`, disabled source-H5 protocol, disabled live source-H5
  motion gate, frozen target/hole finisher, incomplete source target motion
  before finisher, nonzero Cosmos row offset, non-Cosmos dynamic controller,
  `METHOD_EVIDENCE_ALLOWED=true`, future-label teacher dynamic/suffix flags,
  nonzero teacher temporal offsets, and source-motion direction guards.
- Added `scripts/world_model/phase03_forward_backward_readiness.sh`, a
  shell-only readiness check for the allocated probe/full-run path. It is now
  called by the direct forward/backward probe inside the tmux-held Slurm
  allocation before any render canary or full run, not by the tmux launcher
  before `salloc`. It runs the next-coverage guard, verifies the approved
  source-H5 key and short target paths, checks launcher syntax, refuses
  existing `render_probe/fwdback21` / `h5_reverse/try21` artifacts or logs,
  and refuses an existing same-name `p03_fb21` Slurm job. When called inside
  the intended `p03_fb21` Slurm job, it ignores the current `SLURM_JOB_ID` so
  the direct probe does not reject its own allocation.
  The direct probe now passes the full-run target as `h5_reverse/try21` and the
  probe target as `render_probe/fwdback21` explicitly into readiness, so an
  external `RUN_GROUP=render_probe` environment cannot make the full-run
  source-key check inspect the probe group.
- Added `scripts/world_model/phase03_static_protocol_scan.sh`, a shell-only
  static scan now called by readiness. It checks that active Oracle scripts are
  text files, contain no direct `peg.set_pose` / `peg.set_state` /
  `peg.set_state_dict` calls, keep the forward/backward readiness gate wired,
  and do not revive known legacy state-intervention routes.
- Strengthened `scripts/world_model/audit_phase03_oracle_full_pipeline_outputs.py`
  against the old target-teleport class of mistakes. The artifact audit now
  fails if any logged `target_motion_delta_xyz` exceeds the configured
  `target_motion_per_step`, if cumulative target motion is not the sum of
  logged deltas, if any action-trace row lacks parseable
  `target_motion_delta_xyz` / `target_motion_cumulative_xyz`, or if a run
  claims target motion completed before the finisher while the cumulative
  motion does not match `target_motion_xyz`.
- Strengthened premotion Cosmos evidence requirements. The runner now records
  `max_premotion_cosmos_predictions` and
  `required_premotion_cosmos_predictions`; the artifact audit and completion
  gate require repeated premotion Cosmos reports rather than accepting a
  single premotion clip. The forward/backward launcher requires 4 premotion
  predictions.
- Strengthened Cosmos action provenance audit. Each `cosmos_dynamic_control`
  row must reference a valid postmotion Cosmos report, and its
  `raw_cosmos_action` must match the corresponding row of that report's
  `denormalized_robot_action_chunk`. This prevents a trace from merely labeling
  an action as `cosmos3_policy_output` without matching RGB/action prediction
  evidence.
- Strengthened finisher audit. If `near_target_before_finisher=true`, the
  action trace must contain physical DP/manual finisher rows; if
  `simulator_success_metric=true`, a physical finisher must have been
  attempted. This blocks metric-only, boundary-only, or target-assisted
  apparent successes from being accepted without robot-driven near-target
  insertion actions.
- Strengthened the completion gate itself against stale audit files. It now
  recomputes physical-finisher evidence directly from `action_trace.json` and
  `summary.json`; a modern strict single-case row must have
  `trace_finisher_rows > 0`, `near_target_before_finisher=true`, and a
  non-null `finisher_start_step`.
- Strengthened the completion gate for Cosmos RGB evidence. A modern strict
  row must have non-empty premotion `prefix_rgb.mp4` and
  `outputs/sample/vision.mp4` counts meeting the required premotion report
  count, plus at least one non-empty postmotion `prefix_rgb.mp4` and
  `outputs/sample/vision.mp4`. Current strict rows pass:
  `h5_continuous_insert/try04` has `4/4/1/1`, and
  `h5_continuous_insert/try11` has `4/4/7/7`.
- Strengthened the completion gate for active insertion review. A modern
  strict row now also requires `active_robot_insertion_confirmed=true` in the
  visual-review verdict; current strict rows pass and
  `accepted_rows_missing_active_robot_insertion_count=0`.
- Strengthened the completion gate for full rendered video evidence. A modern
  strict row now also requires non-empty `videos/raw.mp4` and
  `videos/annotated.mp4`; current strict rows pass and
  `accepted_rows_missing_rendered_video_count=0`.
- Strengthened the completion gate for before/after distance evidence. A
  modern strict row now requires parseable `initial_eval.peg_head_l2`,
  `target_motion_live_gate.peg_head_l2`, and `final_eval.peg_head_l2`,
  `final_eval.success=true`, `final_success=true`, and final distance lower
  than the pre-finisher live-gate distance. Current strict rows pass:
  `try04` is `0.6058 -> 0.1441 -> 0.0145`, `try11` is
  `0.4232 -> 0.1133 -> 0.0078`.
- Strengthened the completion gate for visual physical-validity flags. A
  modern strict row now requires `no_snap_or_teleport_observed=true`,
  `no_wall_insertion_or_wall_penetration_observed=true`, and
  `no_disappearing_objects_observed=true` in the visual-review verdict.
  `try11` already had this conclusion in its visual notes; its verdict now
  records the structured fields directly. Current strict rows pass and
  `accepted_rows_missing_physical_validity_visual_flags_count=0`.
- Strengthened the completion gate for visual Cosmos evidence confirmation. A
  modern strict row now requires `cosmos_rgb_prediction_confirmed=true` and
  `cosmos_action_prediction_confirmed=true` in the visual-review verdict.
  `try11` already had matching Cosmos artifacts and trace evidence; its
  verdict now records the structured fields directly. Current strict rows pass
  and `accepted_rows_missing_cosmos_visual_confirmation_count=0`.
- Strengthened the completion gate against single-case verdicts that
  accidentally claim overall completion. A modern strict single-case row now
  requires `overall_task_complete=false`; current accepted rows pass and
  `accepted_rows_with_overall_complete_flag_count=0`.
- Strengthened the completion gate for the target-motion controller switch.
  After the first logged target-motion increment, a modern strict row must have
  zero `dp_static_prefix` rows. Current strict rows pass:
  `accepted_rows_with_dp_static_after_target_motion_count=0`; `try04` first
  target motion is trace index `140`, and `try11` first target motion is trace
  index `100`.
- Added `scripts/world_model/phase03_forward_backward_candidate_status.sh`, a
  read-only helper for the current default key. It confirms
  `hole_late_reverse_seed1040038_idx0004` exists in the approved
  `fix3_733/canonical_h5` tree and has no existing active/archive artifacts.
  Its closest-prior-failure table also supports choosing reverse before
  move-stop for the next forward/backward attempt: prior real reverse attempts
  reached best `peg_head_l2` around `0.021m`, while the closest real move-stop
  attempts are around `0.08-0.09m`. Future-label/row-offset diagnostic rows
  are excluded from this helper.

- [x] Acquire a tmux-held interactive Slurm allocation for the full-pipeline
      retry.
- [x] Create a full-pipeline run directory under
      `experiments/maniskill/runs/03_oracle/`.
- [x] Create a full-pipeline log file under `logs/03_oracle/`.
- [x] Start the full-pipeline retry from reset.
- [x] Run DP initial policy before target motion in the full-pipeline retry.
- [x] Detect target/hole motion causally in a partial boundary run.
- [x] During static-target prefix, keep Cosmos-3 predicting future RGB target
      state for the upcoming switch.
- [x] Run RGB Cosmos-3 imagination inline from the same live RGB history.
- [x] Save Cosmos-rendered RGB prediction from the live history.
- [x] Save Cosmos action prediction / action chunk for the robot.
- [x] After target motion starts, execute Cosmos-derived actions only, not the
      original DP action chunk.
- [x] Continue Cosmos-control until the peg/stick and robot are near the target
      according to logged live observations.
- [x] Start the Oracle finisher only after the near-target condition is met.
- [x] Execute the finisher with original DP or manual controller actions through
      physics only.
- [x] Complete insertion or record a physical failure without teleporting,
      wall penetration, insertion into wall, or disappearing objects.
- [x] Require `peg_state_guard.ok=true`; if the runtime guard cannot intercept
      `peg.set_pose`, `peg.set_state`, and `peg.set_state_dict`, block the run
      before rollout rather than producing a misleading video.
- [x] Record before/after `peg_head_at_hole` around insertion for the accepted
      `h5_continuous_insert/try04` case. Its `action_trace.json` records the
      DP finisher approach through the final success row, and `summary.json`
      records both best/final `peg_head_at_hole` and `peg_head_l2`.
- [x] Record commanded action trace through the Cosmos-control and finisher
      segments.
- [x] Save full annotated video showing target motion, Cosmos-control,
      finisher, and final insertion/failure.
- [x] Set `method_evidence_allowed=false`.
- [x] Reject any report that hides the snap / state intervention.
- [x] Reject any run where Oracle is applied before the DP prefix, causal
      trigger, and RGB Cosmos output exist.
- [x] Complete at least one visually confirmed validation-key single-case
      success.
- [ ] Expand successful coverage to forward/backward target motion.
- [ ] Expand successful coverage to left/right target motion.
- [ ] Expand successful coverage to peg/wooden-stick disturbance.
- [ ] Cover multiple keys from `experiments/maniskill/data/fix3_733/`.
- [ ] Only after all coverage above exists, mark the overall Oracle task
      complete.
- [ ] Add an explicit source-key / reset-key path for the approved
      `fix3_733/canonical_h5/<key>.fix3/<key>.h5` episodes. The current
      full-pipeline retry is seed-based and must not be reported as multi-key
      coverage.
- [ ] Use the approved 733-key data distribution when choosing multi-key
      coverage. Current filename-level counts under `canonical_h5/` are:
      `none` 160, `peg_drop` 119, `hole_late_fast_shift` 105,
      `hole_late_reverse` 99, `hole_late_continuous_insert` 96,
      `hole_late_sine` 60, `hole_late_constant` 48,
      `hole_late_move_stop` 44, and `peg_disturb` 2. These counts are planning
      context only until the full-pipeline script can reset from explicit
      approved keys.
- [ ] Use `TODO/03/multikey_candidates.md` as the initial approved-key
      candidate list once source-key reset support exists. This file is not
      coverage evidence by itself.
- [ ] Use `TODO/03/review_runbook.md` to review each full-pipeline run before
      describing any single case as successful.
- [ ] Use `TODO/03/archive_policy.md` after review to move invalid, misleading,
      or superseded Phase 03 runs out of active `experiments/`.
- [ ] Use
      `scripts/world_model/summarize_phase03_oracle_full_pipeline_artifacts.sh`
      as a read-only helper to locate the newest full-pipeline run and print
      key artifact statuses before the full review.

Current entry point status:

- `scripts/slurm/run_phase03_oracle_in_allocation.sh` is superseded archive /
  debug context only. It now refuses to run unless
  `ALLOW_SUPERSEDED_PHASE03_BOUNDARY=1` is set explicitly, and it must not be
  used for active Phase 03 evidence.
- `scripts/world_model/phase03_oracle_boundary_probe.py` no longer applies
  `peg.set_pose(base_env.goal_pose)` for final seating. It records an Oracle
  boundary / decision point and any after-state must come from an action-driven
  physics step.
- The corrected Oracle retry must not apply `peg.set_pose` to the peg. Oracle
  may only annotate the boundary / decision point or choose an action-driven
  DP-compatible continuation.
- The controlled target/hole perturbation is also logged as state
  intervention, so this run is an Oracle upper-bound boundary probe only.
- `scripts/slurm/run_phase03_oracle_video_trace_in_allocation.sh` and
  `scripts/training/eval_dp_oracle_boundary_trace.py` are also superseded
  archive / debug context only. The wrapper now refuses by default. The
  2026-07-03 `134040_server64` output from the earlier implementation is
  invalid for active evidence because it recorded a visible teleport into the
  hole after `peg.set_pose(base_env.goal_pose)`.
- Active full-pipeline entry point:
  `scripts/slurm/run_phase03_oracle_full_pipeline_in_allocation.sh` launches
  `scripts/training/eval_dp_oracle_full_pipeline.py`.
- Directional coverage wrapper prepared for after the first full-pipeline case:
  `scripts/slurm/run_phase03_oracle_full_pipeline_direction_suite_in_allocation.sh`.
  It runs target `+Y`, `-Y`, `+X`, and `-X` cases with separate run IDs. This
  wrapper does not cover peg/wooden-stick disturbance or multiple approved 733
  keys by itself and therefore cannot complete the overall Oracle task. It
  records per-case exit codes in `case_status.tsv` and tries to run all
  directions even if an earlier direction fails.
- The full-pipeline script keeps one live ManiSkill env in memory, runs DP
  before target motion, calls official `cosmos_framework.scripts.inference` in
  policy mode for Cosmos RGB/action prediction, executes only extracted Cosmos
  robot actions during the dynamic stage, and permits a DP/manual finisher only
  after the near-target gate.
- The target-motion boundary is now strict: when the first target/hole motion
  increment is applied, the script records a
  `target_motion_trigger_no_robot_action` frame and switches to Cosmos. No
  `dp_static_prefix` action may cross into target-motion time.
- The full-pipeline script requires at least
  `MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER=4` Cosmos-derived dynamic actions
  before the DP/manual finisher can be treated as protocol-compliant. This
  prevents a near-target gate from skipping or minimizing the required Cosmos
  action-control stage; if the minimum is not met, the script records
  `blocked_insufficient_cosmos_dynamic_actions_no_finisher` and does not enter
  the finisher.
- The full-pipeline script now installs a runtime peg state guard immediately
  after reset. If `peg.set_pose`, `peg.set_state`, or `peg.set_state_dict` is
  called by the runner/glue after reset, the attempt must fail instead of
  writing another snap-insertion video.
- Each Cosmos policy call records its live RGB prefix, generated Cosmos RGB
  prediction video path, `sample_outputs.json`, and extracted robot action
  chunk path in `summary.json`, including explicit `stage_name` and
  `prefix_frame_index` metadata.
- The artifact audit now requires both at least one pre-motion Cosmos report
  and at least one post-motion Cosmos report. New runs use short `pre/NN` and
  `post/NN` stage names; old `premotion_` / `postmotion_` artifacts are only
  accepted for historical compatibility. A run cannot pass by only calling
  Cosmos after target motion has already begun.
- The artifact audit also rejects any `dp_static_prefix` row after the
  `target_motion_trigger_no_robot_action` row, or any DP prefix row with
  `target_motion_applied=true`.
- The artifact audit rejects runs where
  `cosmos_dynamic_action_count < min_cosmos_dynamic_actions_before_finisher`
  and also rejects any active run whose configured minimum is below 4.
- The full-pipeline run writes both raw and annotated attempt videos under the
  run's `videos/` directory. The annotated video labels DP prefix,
  Cosmos-control, target-motion cumulative displacement, and finisher stages.
- After the full-pipeline script finishes, the wrapper runs
  `scripts/world_model/audit_phase03_oracle_full_pipeline_outputs.py` inside
  the same allocation and writes `artifact_audit.json`. This audit checks
  required files and protocol stages only; it does not replace visual review
  and cannot promote a run to physical success by itself.
- Visual review must use
  `scripts/world_model/phase03_oracle_visual_review_checklist.md`. A simulator
  metric or artifact audit cannot establish single-case success without this
  video review.
- The wrapper preserves `pipeline_exit_code` and `artifact_audit_exit_code` in
  `classification.txt`. The artifact audit is attempted even if the main
  pipeline exits nonzero, so failure runs still leave inspectable evidence.
- The full-pipeline script records partial evidence if a premotion or
  postmotion Cosmos policy call fails. `blocked_*` classifications return
  nonzero and `invalid_*` classifications return nonzero, so wrappers and
  suites cannot silently treat them as successful runs.
- If checkpoint / environment / agent initialization fails, the full-pipeline
  script writes `summary.json` and `classification.txt` with
  `blocked_initialization_failed_no_rollout` instead of leaving only a raw log.
- If DP-prefix rollout, rendering, or early environment stepping raises a
  runtime exception after initialization, the full-pipeline script records
  `blocked_runtime_exception_partial_evidence_written` with partial evidence.
- The full-pipeline script now applies controlled target/hole motion as
  multi-step small increments toward `TARGET_MOTION_X/Y/Z` using
  `TARGET_MOTION_PER_STEP` (default `0.00125`) instead of a single one-frame
  jump. The cumulative target motion vector is recorded in each action-trace
  row.
- During the static-target prefix, the full-pipeline script can request
  repeated pre-motion Cosmos predictions using
  `PREMOTION_COSMOS_STEP`, `PREMOTION_COSMOS_INTERVAL`, and
  `MAX_PREMOTION_COSMOS_PREDICTIONS`. Default settings start at step 20 and
  repeat every 16 steps until target motion starts; `MAX_PREMOTION_COSMOS_PREDICTIONS=0`
  means no fixed count cap, not zero predictions.
- The full-pipeline script refuses to execute Cosmos actions without an active
  WAM `normalization_stats.json`, because denormalizing the official Cosmos
  policy output is required before sending `pd_ee_delta_pose` actions.
- The full-pipeline script now writes simulator success separately from
  physical success: `physical_insertion_success=false` remains false until
  video review confirms a continuous physical insertion.
- The full-pipeline script clips Cosmos-predicted actions to the ManiSkill
  action-space bounds before `env.step`, and records both raw and executed
  actions in the trace.
- The full-pipeline script audits consecutive peg and target/hole displacement
  at the end of the run. Any discontinuous object jump marks the result invalid
  and prevents it from being treated as usable Oracle evidence.

Current resource status:

- Invalid metric-only Oracle boundary run, archived:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/p03_oracle_boundary_20260703_122133_162545_server30/`.
- Result: `oracle_boundary_diagnostic_complete_not_method_success`;
  `method_evidence_allowed=false`; `physical_insertion_success=false`;
  `target_motion_trigger_frame=84`; `dp_prefix_steps=85`;
  `oracle_jump_distance=0.3545860179864771`.
- This run used a controlled target/hole perturbation and explicit Oracle
  `peg.set_pose(base_env.goal_pose)`. It is not active evidence for a corrected
  no-teleport Oracle.
- Invalid video-bearing Oracle boundary run, archived:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/p03_oracle_video_trace_20260703_134040_162568_server64/`.
- Video result, now classified invalid:
  `oracle_boundary_video_diagnostic_complete_not_method_success`;
  `method_evidence_allowed=false`;
  `physical_insertion_success_claimed=false`;
  `target_motion_trigger_frame=84`; `dp_prefix_steps=85`;
  `oracle_jump_distance=0.35089600382963243`;
  before-oracle `peg_head_l2=0.3485887092784015`;
  after-oracle `peg_head_l2=0.003389940740797422`.
- The video shows a direct teleport/snap into the hole at the Oracle frame.
  This repeats the documented `docs/oracle_snap_error_review.md` failure mode
  and must not be treated as Phase 03 progress.
- Archived invalid video:
  `experiments/maniskill/runs/03_oracle/p03_oracle_video_trace_20260703_134040_162568_server64/videos/0.mp4`.
- Archived location:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/p03_oracle_video_trace_20260703_134040_162568_server64/`.
- The matching active log was archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/p03_oracle_video_trace_20260703_134040_162568_server64.log`.
- The metric-only `122133_server30` output was also archived because it used
  the same invalid final-seat `peg.set_pose` path.
- Corrected no-teleport boundary-only video run, archived as partial /
  superseded evidence because it does not insert:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/p03_oracle_no_teleport_trace_20260703_151937_162757_mgmtserver02/`.
- Corrected result:
  `oracle_boundary_no_teleport_video_diagnostic_complete_not_method_success`;
  `method_evidence_allowed=false`;
  `physical_insertion_success_claimed=false`;
  `oracle_state_intervention_used=false`;
  `oracle_peg_state_intervention_used=false`;
  `oracle_set_pose_used=false`;
  `snap_detected=false`;
  `target_motion_trigger_frame=84`; `dp_prefix_steps=85`;
  `oracle_jump_distance=0.009504186208756801`;
  before-oracle `peg_head_l2=0.3398252486823192`;
  after-oracle `peg_head_l2=0.3314545521039792`.
- Corrected rendered video:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/p03_oracle_no_teleport_trace_20260703_151937_162757_mgmtserver02/videos/0.mp4`.
- Corrected annotated video:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/p03_oracle_no_teleport_trace_20260703_151937_162757_mgmtserver02/videos/0_annotated_no_teleport_oracle.mp4`.
- This run is not a completed Oracle experiment: it stops around target-motion
  onset / decision and does not execute Cosmos-derived actions through the
  dynamic stage or attempt a physical insertion finisher.
- Matching log archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/p03_oracle_no_teleport_trace_20260703_151937_162757_mgmtserver02.log`.
- The matching WAM `normalization_stats.json` was found under the checkpoint's
  archived training condition root and copied into the active checkpoint
  directory:
  `experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam/normalization_stats.json`.
- Provenance is recorded in
  `experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam/normalization_stats_provenance.md`.
- Current corrected full-pipeline status as of `2026-07-04T21:25:00+08:00`:
  the old visible-snap video remains invalid and archived; the active reruns
  no longer use peg state editing and do not show snap in the displacement
  audit, but all completed attempts are physical failures, not success.
- The long run ids listed in this historical block are old failure / diagnostic
  paths only. They are not naming examples. New attempts must use short grouped
  paths such as `h5_constant/try01` or `peg_disturb/try03`.
- `salloc` job `164971` ran in tmux session `reflex_oracle_servo_0704` on
  `server56` for full-pipeline reruns with manual Oracle servo finisher
  variants. Completed active evidence runs:
  - `p03_oracle_full_pipeline_servo_20260704_171152_164971_mgmtserver02`:
    4 premotion Cosmos reports, 4 postmotion Cosmos reports, 28 Cosmos dynamic
    actions, manual finisher started at step 113, `snap_detected=false`,
    final `peg_head_l2=0.3119160743678719`, physical failure.
  - `p03_oracle_full_pipeline_servo_signed_20260704_173356_164971_mgmtserver02`:
    4 premotion Cosmos reports, 4 postmotion Cosmos reports, 28 Cosmos dynamic
    actions, signed manual finisher started at step 113,
    `snap_detected=false`, final `peg_head_l2=0.1406997740206978`, physical
    failure.
  - `p03_oracle_full_pipeline_servo_signed_yaw_20260704_175249_164971_mgmtserver02`:
    4 premotion Cosmos reports, 4 postmotion Cosmos reports, 32 Cosmos dynamic
    actions, no finisher because near gate `0.14` was not reached,
    `snap_detected=false`, final `peg_head_l2=0.14859554020073992`, physical
    failure. A contact-sheet review of annotated frames showed approach without
    visible teleport, but no insertion.
  - `p03_oracle_full_pipeline_servo_gate016_20260704_181951_164971_mgmtserver02`:
    4 premotion Cosmos reports, 3 postmotion Cosmos reports, 24 Cosmos dynamic
    actions, manual finisher started at step 109 with near gate `0.16`,
    `snap_detected=false`, final `peg_head_l2=0.2766077715587936`, physical
    failure.
- All four completed full-pipeline reruns have `method_evidence_allowed=false`,
  `physical_insertion_success_claimed=false`, `visual_full_insertion_confirmed=false`,
  `forbidden_peg_state_intervention_used=false`, `set_pose_used_on_peg=false`,
  and `artifact_audit.ok=true`.
- `salloc` job `165186` on `server10` attempted
  `p03_oracle_full_pipeline_servo_combo_20260704_185322_165186_mgmtserver02`
  but failed before rollout evidence with `vk::Device::waitForFences:
  ErrorDeviceLost`. It wrote only partial blocked evidence and must not be
  counted as protocol evidence.
- `salloc` job `165189` ran in tmux session `reflex_oracle_combo_0704b` on
  `server28` for additional no-snap full-pipeline reruns:
  - `p03_oracle_full_pipeline_servo_combo_retry_20260704_190600_165189_mgmtserver02`:
    4 premotion Cosmos reports, 3 postmotion Cosmos reports, 24 Cosmos dynamic
    actions, manual finisher started at step 109, `snap_detected=false`,
    final `peg_head_l2=0.750433`, physical failure.
  - `p03_oracle_full_pipeline_servo_no_yaw_20260704_193814_165189_mgmtserver02`:
    4 premotion Cosmos reports, 3 postmotion Cosmos reports, 24 Cosmos dynamic
    actions, no-yaw manual finisher started at step 109,
    `snap_detected=false`, final `peg_head_l2=0.469944`, physical failure.
  - `p03_oracle_full_pipeline_hole_frame_20260704_200855_165189_mgmtserver02`:
    4 premotion Cosmos reports, 4 postmotion Cosmos reports, 26 Cosmos dynamic
    actions, hole-frame manual finisher started at step 111,
    `snap_detected=false`, final `peg_head_l2=0.2513050137923124`, physical
    failure.
- The hole-frame trace showed the finisher initially reduced distance from
  `0.1525` to `0.1044`, then pushed the peg away again. The controller was
  therefore replaced by a staged hole-frame finisher that aligns local y/z
  before slow physical insertion attempts.
- `salloc` job `165288` ran in tmux session `reflex_oracle_staged_0704` on
  `server56` for staged finisher reruns:
  - `p03_oracle_full_pipeline_staged_hole_20260704_204110_165288_mgmtserver02`:
    4 premotion Cosmos reports, 3 postmotion Cosmos reports, 24 Cosmos dynamic
    actions, staged manual finisher started at step 109,
    `snap_detected=false`, final `peg_head_l2=0.10468034418100798`, physical
    failure. The trace held y/z close to the hole but stalled at
    `peg_head_at_hole[0] ~= -0.104`.
  - `p03_oracle_full_pipeline_staged_loose_20260704_210113_165288_mgmtserver02`:
    4 premotion Cosmos reports, 4 postmotion Cosmos reports, 25 Cosmos dynamic
    actions, staged manual finisher started at step 110,
    `snap_detected=false`, final `peg_head_l2=0.10549092663220139`, physical
    failure. Looser alignment improved early approach but still stalled around
    `peg_head_at_hole[0] ~= -0.103`.
- The staged finisher is a no-teleport protocol improvement, not a success:
  it prevents the earlier snap/final-seat error and stabilizes lateral/vertical
  alignment, but it does not yet solve physical insertion/contact.
- `salloc` job `165308` ran in tmux session `reflex_oracle_dpfin_0704` on
  `server56` for a DP-finisher full-pipeline run:
  - `p03_oracle_full_pipeline_dpfin_20260704_212324_165308_mgmtserver02`:
    4 premotion Cosmos reports, 4 postmotion Cosmos reports, 26 Cosmos dynamic
    actions, DP finisher started at step 111 after the near-target gate,
    `artifact_audit.ok=true`, `snap_detected=false`, simulator success metric
    true, final `peg_head_l2=0.01369317842884192`, final
    `peg_head_at_hole=[-0.013571381568908691, 0.0017894953489303589,
    -0.00034415721893310547]`, and maximum peg step displacement
    `0.01175413308574014`.
  - The source run's default video was partially occluded, so close-up visual
    review was generated by replaying the recorded action trace from reset
    with the same target-motion increments and no peg state intervention:
    `visual_review_replay_v2/front_replay_annotated.mp4`,
    `visual_review_replay_v2/oblique_replay_annotated.mp4`, and
    `visual_review_replay_v2/top_replay_annotated.mp4`.
  - After review, this run is downgraded to synthetic seed-motion diagnostic
    only because it used runner-specified `TARGET_MOTION_Y=0.0125` rather than
    an approved `fix3_733` validation/canonical H5 key and recorded
    disturbance trajectory. `visual_review_replay_v2/visual_review_verdict.json`
    records `single_case_success_confirmed=false`,
    `synthetic_seed_motion_diagnostic_only=true`, and
    `invalid_as_validation_set_success=true`.
  - This run must not be counted as validation-set success, even though it
    showed the DP-finisher handoff can reach simulator success without snap in
    the synthetic diagnostic setting.
- Current valid Phase 03 Oracle validation-key single-case success count is
  `1` under the stricter active-insertion visual standard:
  `h5_continuous_insert/try04`.
- Implementation correction after visual rejection: the active full-pipeline
  runner now accepts `SOURCE_H5_PATH` / `SOURCE_KEY`, loads that approved H5
  key's seed, target-motion frame, target-motion vector, and per-step motion,
  records `target_motion_source=fix3_733_source_h5_protocol`, and marks
  wrapper-created motion as `synthetic_runner_args_diagnostic_only`.
- To prevent the repeated "hole moves before wooden stick approach" error, H5
  target motion is now gated by both the approved key's motion step and a live
  preinsert gate derived from that key's source trigger thresholds. If the live
  rollout has not reached the intended approach state, the runner records
  `source_motion_gate_rejections` and classifies the attempt as
  `protocol_mismatch_source_h5_motion_gate_never_reached` instead of moving
  the target early.
- Validation audits can now be run with `--require-source-h5`; the Slurm
  wrapper exposes this as `REQUIRE_SOURCE_H5_PROTOCOL=true`, so synthetic
  target motion cannot pass as validation-key evidence.
- Validation-key attempt `h5_fastshift/try03` completed on Slurm job `165341`
  / `server46` using approved source key
  `hole_late_fast_shift_seed10300001_idx5000`. It is valid protocol evidence
  but not a success: `artifact_audit.ok=true`,
  `target_motion_source=fix3_733_source_h5_protocol`,
  `target_motion_trigger_frame=99`, 4 premotion Cosmos reports, 4 postmotion
  Cosmos reports, 27 Cosmos dynamic actions, DP finisher from step 127,
  `simulator_success_metric=false`, final `peg_head_l2=0.12979192961772668`,
  and `snap_detected=false`. The trace confirms DP steps 95-98 had zero target
  motion and the first target-motion increment occurred only at step 99 after
  the live H5 preinsert gate passed.
- Validation-key retry `h5_fastshift/try04` completed on Slurm job `166830` /
  `server57` using the same approved source key and stricter current protocol.
  It is valid protocol evidence but not a success: `artifact_audit.ok=true`,
  `target_motion_source=fix3_733_source_h5_protocol`,
  `target_motion_trigger_frame=97`, 4 premotion Cosmos reports, 4 postmotion
  Cosmos reports, 29 Cosmos-derived dynamic actions, original DP finisher from
  step 127, `simulator_success_metric=false`, final `peg_head_l2` about
  `0.1219`, best finisher `peg_head_l2` about `0.0994`,
  `snap_detected=false`, and `peg_state_guard.ok=true`. The run and matching
  log were moved to
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_fastshift/try04`
  and
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/h5_fastshift/try04.log`
  so failed attempts do not clutter active `experiments/`.
- Phase 03 active-folder cleanup completed after `h5_fastshift/try04`: the
  active run/log tree now keeps only
  `experiments/maniskill/runs/03_oracle/h5_continuous_insert/try04/` and
  `logs/03_oracle/h5_continuous_insert/try04.log` as accepted strict
  single-case evidence. Failed H5 retries, peg-disturbance retries,
  render/probe diagnostics, stale summary/check logs, and legacy long-name
  outputs were moved to `/public/home/yanhongru/ICLR2027/archive/Reflex/`
  while preserving repository-relative paths.
- Validation-key attempt `h5_constant/try01` completed under the same short
  grouped naming convention using approved key
  `hole_late_constant_seed10250253_idx5009`. It is now rejected as a strict
  Phase 03 Oracle single-case validation-key success:
  `artifact_audit.ok=true`,
  `target_motion_source=fix3_733_source_h5_protocol`,
  `target_motion_trigger_frame=120`, DP steps 116-119 had zero target motion,
  first target-motion increment occurred at step 120 after the live H5
  preinsert gate passed, one Cosmos-derived dynamic action executed at step
  121, DP finisher started at step 122, simulator success became true at step
  135, final `peg_head_l2=0.011706860340233998`, and
  `snap_detected=false`. However, visual review raised a target-assisted
  insertion concern: the moving target/hole appears to move onto the peg /
  wooden stick, so metric true does not prove active robot insertion. Visual
  review verdict:
  `experiments/maniskill/runs/03_oracle/h5_constant/try01/visual_review_verdict.json`.
- Current valid Phase 03 Oracle validation-key single-case success count is
  now `1`, from `h5_continuous_insert/try04`.
- Validation-key attempt `h5_reverse/try01` completed using approved key
  `hole_late_reverse_seed1040017_idx1250`. It is valid protocol evidence but
  not a success: `target_motion_source=fix3_733_source_h5_protocol`,
  `target_motion_trigger_frame=106`, one Cosmos-derived dynamic action before
  the DP finisher, `simulator_success_metric=false`, final
  `peg_head_l2=0.05669456492755592`, and `snap_detected=false`.
- Validation-key retry `h5_reverse/try02` completed on Slurm job `166477` /
  `server63` using the same approved key but requiring 4 Cosmos-derived dynamic
  actions before the finisher. It produced 6 pre-motion Cosmos reports, 1
  post-motion Cosmos report, executed 4 dynamic Cosmos actions, wrote
  `videos/raw.mp4` and `videos/annotated.mp4`, passed artifact audit, and had
  `snap_detected=false`. It is still not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, final `peg_head_l2` about `0.1865`, best
  finisher `peg_head_l2` about `0.1400`. The manual staged finisher worsened
  the state after the Cosmos segment, so this does not count as reverse
  coverage.
- Validation-key retry `h5_reverse/try03` completed on Slurm job `166502` /
  `server63` using the same approved key, 6 pre-motion Cosmos reports, 1
  post-motion Cosmos report, 4 dynamic Cosmos actions, and original DP as the
  near-target finisher. It passed artifact audit and had `snap_detected=false`,
  but still failed: `simulator_success_metric=false`, final `peg_head_l2` about
  `0.0220`, best about `0.0210`. This is the best reverse-key distance so far,
  but it is not a success because insertion is not complete and visual review
  has not confirmed full insertion.
- After `h5_reverse/try03`, the runner gained `dp_then_manual_close`, an
  action-level Oracle diagnostic finisher that keeps DP until a close-band live
  distance threshold and then switches to the existing staged physical action
  controller. It must remain `method_evidence_allowed=false` and cannot count
  unless it produces continuous full insertion without snap / wall / object
  disappearance and passes visual review.
- Validation-key retry `h5_reverse/try04` completed on Slurm job `166542` /
  `server63` using `dp_then_manual_close` with a `0.03m` close threshold. It
  passed artifact audit and had `snap_detected=false`, but failed:
  `simulator_success_metric=false`, final `peg_head_l2` about `0.0327`, best
  about `0.0302`. The trace shows 0 `manual_close_after_dp_gate` rows, so this
  run did not actually test the intended close-band manual push; the threshold
  was too tight.
- Validation-key retry `h5_reverse/try05` also completed without triggering the
  intended close-band manual push: final `peg_head_l2` about `0.0347`, best
  about `0.0332`, artifact audit ok, `snap_detected=false`, but 0
  `manual_close_after_dp_gate` rows. Although the launch environment set
  `MANUAL_DP_TO_MANUAL_L2=0.04`, the run manifest did not include the new field,
  so the runner still used the old Python default. The default is now `0.04m`
  for the next retry.
- Validation-key retry `h5_reverse/try06` completed on Slurm job `166613` /
  `server63` with the same approved key and short grouped output path
  `experiments/maniskill/runs/03_oracle/h5_reverse/try06/`. It passed artifact
  audit, produced `videos/raw.mp4` and `videos/annotated.mp4`, recorded 6
  premotion Cosmos reports, 1 postmotion Cosmos report, and executed 4
  Cosmos-derived dynamic actions before the finisher. The close-band handoff
  was actually exercised: 200 `diffusion_policy_before_manual_close_gate` rows
  followed by 210 `manual_close_after_dp_gate` rows. It still failed:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.0398`, best about `0.0278`,
  `snap_detected=false`. This is useful negative diagnostic evidence only and
  does not count as reverse-key success.
- Validation-key retry `h5_reverse/try07` targeted `server46` with a narrower
  `MANUAL_DP_TO_MANUAL_L2=0.025` close threshold and gentler manual insert
  settings, but it did not enter Oracle rollout. The render canary hung at
  `render_rgb_array_start` and timed out with exit code 124, leaving only
  `manifest.txt` and `classification.txt`. Treat it as infrastructure failure
  only, not Oracle evidence. The same settings were retried as
  `h5_reverse/try08` on `server63`.
- Validation-key retry `h5_reverse/try08` ran on Slurm job `166651` /
  `server63` with the same narrower `0.025m` close threshold. It completed the
  source-H5 full rollout with 6 premotion Cosmos reports, 1 postmotion Cosmos
  report, 4 Cosmos-derived dynamic actions, and 510 DP close-band finisher rows.
  The close handoff did not trigger: there were 0 `manual_close_after_dp_gate`
  rows because the best live distance only reached about `0.0334m`. Final
  `peg_head_l2` was about `0.0350`, `simulator_success_metric=false`,
  `visual_full_insertion_confirmed=false`, and summary discontinuity audit had
  `snap_detected=false`, `set_pose_used_on_peg=false`,
  `forbidden_peg_state_intervention_used=false`. It wrote summary, action trace,
  and videos, but no external `artifact_audit.json`; record this as negative
  rollout evidence with missing wrapper audit, not success.
- Completed / failed reverse attempts `h5_reverse/try01` through `try08` have
  been archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_reverse/`,
  with matching logs under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/h5_reverse/`.
- Validation-key retry `h5_reverse/try09` is queued as Slurm job `166816` on
  `server57` with approved key `hole_late_reverse_seed1040017_idx1250`,
  original DP finisher, `MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER=4`,
  `MAX_FINISHER_STEPS=520`, and `MAX_EPISODE_STEPS=650`. As of
  `2026-07-05T13:52+08:00`, it was pending for `Priority` and had no active
  run files yet.
- Validation-key retry `h5_reverse/try09` then completed on Slurm job `166816`
  / `server57`. It passed artifact audit with 4 premotion Cosmos reports,
  1 postmotion Cosmos report, 4 Cosmos-derived dynamic actions, and 540
  DP-finisher rows, with `snap_detected=false` and `peg_state_guard.ok=true`,
  but failed physical insertion: `simulator_success_metric=false`, final
  `peg_head_l2` about `0.1054`, best about `0.0981`. The longer pure-DP
  finisher worsened reverse-key performance and does not count as directional
  success.
- Validation-key retry `h5_reverse/try10` completed on Slurm job `166852` /
  `server63` using the same approved key and a `dp_then_manual_close` finisher
  intended to reproduce the close `try03` DP state, then switch to physical
  manual close at `MANUAL_DP_TO_MANUAL_L2=0.024`. It is valid protocol evidence
  but not success: `artifact_audit.ok=true`, 6 premotion Cosmos reports,
  1 postmotion Cosmos report, 4 Cosmos-derived dynamic actions, videos,
  `simulator_success_metric=false`, final `peg_head_l2` about `0.1678`, best
  about `0.1037`, `snap_detected=false`, and `peg_state_guard.ok=true`. The
  action trace has 410 `diffusion_policy_before_manual_close_gate` rows and
  zero `manual_close_after_dp_gate` rows, so the close-band manual stage did
  not trigger. Archive the run/log after classification; it is not
  reverse-direction coverage.
- Validation-key retry `h5_reverse/try11` targeted a second approved reverse
  key, `hole_late_reverse_seed1040025_idx0003`, with the closest prior
  DP-finisher settings from `try03`. It did not enter rollout: Slurm job
  `166896` / `server63` stopped during compute-node preflight with
  `ValueError: source code string cannot contain null bytes`. Subsequent
  byte-level inspection of the listed Python files did not find NUL bytes, so
  this is infrastructure / preflight failure evidence only. The run/log were
  archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_reverse/try11`
  and
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/h5_reverse/try11.log`.
- Follow-up `render_probe/try15` attempted to find a render-capable GPU and
  launch `h5_reverse/try12` on the same second reverse key. It landed on
  server36, timed out at `render_rgb_array_start`, and wrote
  `blocked_no_render_capable_visible_gpu`; no full Oracle rollout started. The
  probe/log were archived under the external archive and do not count as
  reverse-direction evidence.
- Validation-key retry `h5_reverse/try12` then ran the same second reverse key
  directly on render-capable server57 with `RUN_PREFLIGHT_PY_COMPILE=false`.
  It completed on Slurm job `166912` with 6 premotion Cosmos reports,
  1 postmotion Cosmos report, 4 Cosmos-derived dynamic actions, original DP
  finisher from step 121, videos, and `artifact_audit.ok=true`. It is valid
  negative protocol evidence, not success: `simulator_success_metric=false`,
  final `peg_head_l2` about `0.1113`, best about `0.0895`,
  `snap_detected=false`, and `peg_state_guard.ok=true`.
- Validation-key attempt `h5_move_stop/try01` completed using approved key
  `hole_late_move_stop_seed1080064_idx0000`. It is valid protocol evidence
  but not a success: `target_motion_source=fix3_733_source_h5_protocol`,
  `target_motion_trigger_frame=97`, one Cosmos-derived dynamic action before
  the DP finisher, `simulator_success_metric=false`, final
  `peg_head_l2=0.12060387172064195`, and `snap_detected=false`.
- Validation-key attempt `h5_continuous_insert/try01` completed on Slurm job
  `165423` / `server57` using approved key
  `hole_late_continuous_insert_seed10241044_idx5004`. It is valid protocol
  evidence but not a confirmed success: `target_motion_source=fix3_733_source_h5_protocol`,
  `target_motion_trigger_frame=140`, 4 premotion Cosmos reports, 1 postmotion
  Cosmos report, one Cosmos-derived dynamic action before the DP finisher,
  `artifact_audit.ok=true`, `simulator_success_metric=true`, final original
  `peg_head_l2=0.012049451865591558`, and `snap_detected=false`. However,
  close-up replay from the recorded action trace and target-motion increments
  ended with `success=false` and `peg_head_l2=0.18020458787514929`; therefore
  `visual_review_verdict.json` records `validation_key_single_case_success_confirmed=false`.
- Continuous-insert retry `h5_continuous_insert/try04` completed on Slurm job
  `166789` / `server57` and is the first accepted strict validation-key
  single-case Oracle success. It used approved key
  `hole_late_continuous_insert_seed10241044_idx5004`, `NEAR_TARGET_L2=0.16`,
  original DP finisher, and `MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER=4`.
  It produced 4 premotion Cosmos reports, 1 postmotion Cosmos report,
  4 Cosmos-derived dynamic actions, `videos/raw.mp4`, `videos/annotated.mp4`,
  `artifact_audit.ok=true`, `simulator_success_metric=true`, final
  `peg_head_l2` about `0.01446`, and `snap_detected=false`. The action trace
  has 140 DP static-prefix rows, 1 target-motion no-action row, 4
  `cosmos3_policy_output` rows, and 115 DP-finisher rows. Visual review of the
  annotated/raw videos and extracted keyframes confirmed active robot insertion
  rather than target-assisted insertion; verdict:
  `experiments/maniskill/runs/03_oracle/h5_continuous_insert/try04/visual_review_verdict.json`.
  This remains Oracle diagnostic evidence only with `method_evidence_allowed=false`
  and does not complete the overall Oracle task.
- Continuous-insert retry `h5_continuous_insert/try06` completed on Slurm job
  `166874` / `server57` using a second approved key,
  `hole_late_continuous_insert_seed10241574_idx5018`. It reached simulator
  metric true with `artifact_audit.ok=true`, 4 premotion Cosmos reports,
  1 postmotion Cosmos report, 4 Cosmos-derived dynamic actions, original DP
  finisher from step 122, final `peg_head_l2` about `0.01298`,
  `snap_detected=false`, and `peg_state_guard.ok=true`. Visual review rejected
  it as strict success: frames around 118-180 show the moving target/hole
  reaching and covering the held peg before a visually confirmed active
  robot-driven insertion, and frame 214 is metric-true but occluded. Verdict:
  `experiments/maniskill/runs/03_oracle/h5_continuous_insert/try06/visual_review_verdict.json`.
  This is a metric-true counterexample and does not increase the validation-key
  success count.
- Older continuous-insert attempts `try01`, `try02`, `try03`, and infrastructure
  failure `try05` have been archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/h5_continuous_insert/`,
  with matching logs under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/h5_continuous_insert/`.
- Validation-key attempt `h5_constant/try02` completed on Slurm job `165423`
  / `server57` using approved key `hole_late_constant_seed21250444_idx11004`.
  It is valid protocol evidence but not a success:
  `target_motion_trigger_frame=76`, 4 premotion Cosmos reports, 1 postmotion
  Cosmos report, one Cosmos-derived dynamic action before the DP finisher,
  `artifact_audit.ok=true`, `simulator_success_metric=false`, final
  `peg_head_l2=0.18421349769288659`, and `snap_detected=false`.
- Constant-key retry `h5_constant/try03` completed on Slurm job `166682` /
  `server63` using another approved key
  `hole_late_constant_seed10250338_idx5012`. It ran 4 premotion Cosmos reports,
  1 postmotion Cosmos report, 1 Cosmos-derived dynamic action, and a DP
  finisher from step 100. Artifact audit passed and `snap_detected=false`, but
  it failed physical insertion: `simulator_success_metric=false`, final
  `peg_head_l2` about `0.1303`, best about `0.0749`. It does not increase the
  success count.
- Move-stop retry `h5_move_stop/try02` completed on Slurm job `166700` /
  `server63` using approved key `hole_late_move_stop_seed1080064_idx0000` and
  the short source-H5 launcher. It ran 4 premotion Cosmos reports, 1 postmotion
  Cosmos report, 1 Cosmos-derived dynamic action, and `manual_align_then_dp` as
  the physical finisher. Artifact audit passed and `snap_detected=false`, but
  it failed physical insertion: `simulator_success_metric=false`, final
  `peg_head_l2` about `0.4783`, best about `0.1252`. Trace counts show 260
  `manual_staged_align_before_dp_gate` rows and no
  `diffusion_policy_after_manual_align_gate` rows, so this setting never
  reached the DP handoff and worsened the state. It is not a success and should
  not be reused.
- Move-stop retry `h5_move_stop/try03` was first queued as fixed-`server63`
  Slurm job `166727`, but that pending job was canceled because Slurm estimated
  a next-day start. It has been replaced by probe-backed Slurm job `166729`.
  It keeps the same approved key and short grouped path, requires at least 4
  Cosmos-derived dynamic actions before any finisher, and switches back to the
  original DP finisher instead of `manual_align_then_dp`. Job `166729` landed
  on `server36` and failed the render canary at `render_rgb_array_start` with
  exit code 124 before any Oracle rollout. This is infrastructure failure only,
  not Oracle evidence. A follow-up probe-backed attempt, Slurm job `166734` on
  `server46`, also failed the render canary at `render_rgb_array_start` with
  exit code 124 before any Oracle rollout. The next probe-backed attempt,
  Slurm job `166742` on `server02`, also failed the same render canary with
  exit code 124. These are infrastructure failures only. A reduced-resource
  `server63` retry was queued as Slurm job `166746` and then replaced by
  shorter-walltime Slurm job `166748` with `2 CPU`, `40G` memory, the same
  approved key, short path `h5_move_stop/try03`, at least 4 required
  Cosmos-derived dynamic actions, and original DP finisher. It is still pending
  with Slurm-estimated start `2026-07-06T07:19:22+08:00` and has produced no
  rollout evidence yet. Rechecked on `2026-07-05T12:53+08:00`: job `166748`
  was still pending for `Priority`, and
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
- Validation-key attempt `h5_sine/try01` completed on Slurm job `165423` /
  `server57` using approved key `hole_late_sine_seed1050232_idx0015`. It is
  valid protocol evidence but not a success:
  `target_motion_trigger_frame=101`, 4 premotion Cosmos reports, 1 postmotion
  Cosmos report, one Cosmos-derived dynamic action before the DP finisher,
  `artifact_audit.ok=true`, `simulator_success_metric=false`, final
  `peg_head_l2=0.10940540554847082`, and `snap_detected=false`.
- Peg-disturbance attempt `peg_disturb/try02` on Slurm job `165517` /
  `server36` was interrupted after manifest creation. It produced no rollout
  trace, no summary, no video, and no artifact audit; it is classified as
  `invalid_interrupted_no_rollout_evidence` and does not count as Oracle
  evidence.
- Peg-disturbance attempt `peg_disturb/try03` on Slurm job `165545` /
  `server57` was interrupted before the Python runner reached `main_start`.
  It produced no summary, action trace, video, or artifact audit; it is
  classified as `invalid_interrupted_before_main_no_rollout_evidence`.
- Peg-disturbance attempt `peg_disturb/try04` on Slurm job `165545` /
  `server57` reached DP initialization, reset, peg-state guard installation,
  initial render, and the first premotion Cosmos input-build command, but was
  interrupted after that build stage made no further progress. It produced no
  summary, action trace, postmotion Cosmos output, finisher attempt, final
  attempt video, or artifact audit; it is classified as
  `invalid_interrupted_premotion_cosmos_input_build_no_rollout_evidence`.
- Peg-disturbance attempt `peg_disturb/try05` on Slurm job `165572` /
  `server57` used the new progress markers and Cosmos subcommand timeout
  arguments. It again reached DP initialization, reset, peg-state guard,
  initial render, and `cosmos_build_start` for `premotion_00_step0020`, but
  did not produce the expected Cosmos input JSONL or any downstream Cosmos
  RGB/action output before interruption. It produced no summary, action trace,
  postmotion Cosmos output, dynamic Cosmos-control actions, perturbation
  trigger, finisher attempt, final attempt video, or artifact audit; it is
  classified as
  `invalid_interrupted_premotion_cosmos_input_build_no_oracle_evidence`.
- After `peg_disturb/try05`, the live-prefix Cosmos build path was patched so
  the runner passes `--expected-prefix-video-frames len(frames)` and the build
  script can avoid re-decoding the just-written prefix MP4 for frame counting.
  It still checks the asserted frame count against `prefix_frame_index+1` to
  prevent future-frame leakage.
- On `2026-07-05`, Slurm job `165607` / `server57` verified that patch inside
  the allocation: `py_compile` passed and the build-only probe
  `peg_disturb/build_probe/try01` wrote
  `input/inputs/live_prefix_wam_policy_samples.jsonl`.
- Peg-disturbance attempt `peg_disturb/try06` then ran with the same approved
  key `peg_disturb_seed1051032_idx0008` and short grouped naming. It loaded the
  source-H5 protocol and installed the peg-state guard, but failed during
  initial rendering with `vk::DeviceLostError`. It wrote `summary.json`,
  `action_trace.json`, and `classification.txt` with
  `classification=blocked_runtime_exception_partial_evidence_written`,
  `video=null`, `cosmos_dynamic_actions_executed=false`, and
  `visual_full_insertion_confirmed=false`. This is not Oracle evidence and
  does not increase the success count.
- The full-pipeline Slurm wrapper now runs compute-node preflight gates before
  Oracle rollout by default: `py_compile` for active Phase 03 scripts and
  `render_min_canary.py` with `shader_pack=default`. If either gate fails, the
  wrapper writes `classification.txt` as
  `blocked_preflight_py_compile_failed_no_rollout` or
  `blocked_render_canary_failed_no_rollout` and exits before rollout, so a bad
  Vulkan/render node is not mislabeled as Oracle evidence.
- Peg-disturbance attempt `peg_disturb/try07` on Slurm job `165620` /
  `server53` exercised those new preflight gates. `py_compile` passed, but the
  default-shader render canary hung at `render_rgb_array_start` and timed out
  with exit code `124`, so the wrapper wrote
  `blocked_render_canary_failed_no_rollout` and did not enter Oracle rollout.
  A separate minimal-shader canary on the same node also timed out at
  `render_rgb_array_start`; `server53` should be excluded from the next
  render-bearing Oracle attempt. This is infrastructure evidence only, not
  Oracle evidence.
- Peg-disturbance attempt `peg_disturb/try08` on Slurm job `165642` /
  `server57` also failed before rollout in the render canary. The canary
  reached `render_rgb_array_start` and then raised
  `vk::Device::waitForFences: ErrorDeviceLost`; the Slurm step had to be
  cancelled. Because this exposed that a C++ render abort could prevent the
  wrapper from writing `classification.txt`, the wrapper was hardened to write
  `render_canary_in_progress_no_rollout` before starting the canary and to run
  canary timeout with a `-k` kill window. `try08` is infrastructure failure
  evidence only, not Oracle evidence.
- Peg-disturbance attempt `peg_disturb/try09` on Slurm job `165653` /
  `server46` completed the full source-H5-gated Oracle pipeline for approved
  key `peg_disturb_seed1051032_idx0008`. Render canary passed, DP prefix ran,
  the peg-state guard was OK, four premotion Cosmos RGB/action predictions
  completed under `cosmos_policy/pre/00` through `pre/03`, one postmotion
  Cosmos RGB/action prediction completed under `cosmos_policy/post/00`, one
  Cosmos-derived dynamic action executed after the peg perturb trigger, and the
  DP finisher started only after the near-target gate. The run wrote
  `videos/raw.mp4`, `videos/annotated.mp4`, `summary.json`,
  `action_trace.json`, and `artifact_audit.json`.
- `peg_disturb/try09` is a completed pipeline / physical failure diagnostic,
  but it is not sufficient validation-matched peg-disturbance coverage:
  the source key's intended peg perturb is `[0, -0.04, 0.02]`, while the run
  recorded only about `[0, 0.00188, 0.00158]` immediately after the physical
  force trigger. This under-strength / wrong-direction physical perturb means
  the next peg-disturbance attempt must calibrate physical force/steps while
  still avoiding any peg state edit.
- `peg_disturb/try09` is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  `physical_insertion_success=false`, `cosmos_dynamic_action_count=1`, and
  `discontinuity_audit.snap_detected=false`. The artifact audit was updated to
  recognize the short `pre/` and `post/` stage names and to reject
  peg-disturbance runs whose observed physical peg perturb does not match the
  approved source key's perturb direction/magnitude. This run does not increase
  the validation-key success count or completed peg-disturbance coverage.
- Peg-disturbance calibration diagnostic `peg_disturb/calib01` ran on Slurm
  job `165697` / `server36` after the missing `PYTHONPATH` wrapper issue was
  fixed. This is not Oracle evidence because it skips Cosmos and final
  insertion; it only calibrates how to reproduce the approved source-H5 peg
  perturb through physical force with no peg state edit. Best trial:
  `force_scale=25.0`, `force_steps=8`, observed delta about
  `[0.0119, -0.0371, 0.00993]`, observed fraction about `0.898`, cosine about
  `0.936`, output `peg_disturb/calib01/calibration.json`.
- After the calibration, the full runner and stricter audit were patched to
  record and validate the isolated force-window peg displacement rather than
  an immediate or robot-confounded fallback measurement. The patched scripts
  were `py_compile` checked inside the compute allocation before the next
  rollout attempt.
- Peg-disturbance retry `peg_disturb/try10` on Slurm job `165697` / `server36`
  used the calibrated physical-force settings and short grouped naming, but it
  did not enter Oracle rollout. The wrapper wrote
  `blocked_render_canary_failed_no_rollout` because the default-shader render
  canary hung at `render_rgb_array_start` and timed out with exit code `124`.
  Files written: `manifest.txt` and `classification.txt`; no `summary.json`,
  action trace, Cosmos artifacts, or video exist. This is infrastructure
  failure evidence only, not Oracle evidence. Exclude `server36`, `server53`,
  and `server57` from the next render-bearing attempt unless a fresh canary
  proves otherwise.
- Peg-disturbance retry `peg_disturb/try11` on Slurm job `165729` / `server28`
  also failed before rollout in the default-shader render canary. It reached
  `render_rgb_array_start` and timed out with exit code `124`, then wrote
  `blocked_render_canary_failed_no_rollout`. Files written:
  `manifest.txt` and `classification.txt`; no `summary.json`, action trace,
  Cosmos artifacts, or video exist. This is infrastructure failure evidence
  only, not Oracle evidence. Exclude `server28` from the next render-bearing
  attempt as well.
- Peg-disturbance retry `peg_disturb/try12` on Slurm job `165737` / `server63`
  passed render canary and completed the full source-H5-gated pipeline. It
  wrote four premotion Cosmos RGB/action reports under `pre/00` through
  `pre/03`, one postmotion report under `post/00`, executed four
  Cosmos-derived dynamic actions, ran the DP finisher, and saved
  `videos/raw.mp4`, `videos/annotated.mp4`, `summary.json`,
  `action_trace.json`, and `artifact_audit.json`.
- `peg_disturb/try12` is not a success and not completed peg-disturbance
  coverage. It ended with
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  `physical_insertion_success=false`, and `artifact_audit.ok=false`. The audit
  failure is `peg_perturb_observed_delta_wrong_direction_for_source_key`:
  expected `[0, -0.04, 0.02]`, observed force-window delta about
  `[0.0191, -0.0007, 0.0184]`.
- Root cause for `try12`: the trigger step used zero robot action, but the
  remaining calibrated force steps were applied while Cosmos/DP robot actions
  were executing, so the measured force window was robot-action-confounded and
  no longer matched `peg_disturb/calib01`. The full runner has been patched to
  drain the entire calibrated peg force window with zero robot actions before
  starting postmotion Cosmos control. The next retry is `peg_disturb/try13`.
- Peg-disturbance retry `peg_disturb/try13` on Slurm job `165762` / `server63`
  used the patched zero-action force window and completed the artifact audit:
  4 premotion Cosmos RGB/action reports, 4 postmotion reports, 32
  Cosmos-derived dynamic actions, `videos/raw.mp4`, `videos/annotated.mp4`,
  `summary.json`, `action_trace.json`, and `artifact_audit.json`.
  `artifact_audit.ok=true`; observed perturb matched the source key with
  delta about `[0.0150, -0.0380, 0.0178]`, cosine about `0.942`, and fraction
  about `0.997`.
- `peg_disturb/try13` is not a success. It ended with
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  `near_target_before_finisher=false`, and `finisher_start_step=null`. The
  final `peg_head_l2` was about `0.1967`, so the current `NEAR_TARGET_L2=0.16`
  prevented the DP finisher from running after Cosmos control. The next retry
  should keep `MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER>=4` but loosen the
  near gate enough to test the physical DP finisher after Cosmos evidence.
- Peg-disturbance retry `peg_disturb/try14` used `NEAR_TARGET_L2=0.20` and
  kept `MIN_COSMOS_DYNAMIC_ACTIONS_BEFORE_FINISHER=4`. It passed audit with
  4 premotion Cosmos RGB/action reports, 1 postmotion report, 4
  Cosmos-derived dynamic actions, and matched source perturb. It then entered
  the DP finisher (`near_target_before_finisher=true`,
  `finisher_start_step=147`) and ran 273 `oracle_physical_dp_finisher` rows.
  It is not a success: `simulator_success_metric=false`,
  `visual_full_insertion_confirmed=false`, and final `peg_head_l2` worsened to
  about `0.2059`. The next retry should switch to an existing manual physical
  finisher, such as `manual_hole_frame_servo`, instead of reusing the failed DP
  finisher.
- Peg-disturbance retry `peg_disturb/try15` requested the existing
  `manual_hole_frame_servo` finisher, but it did not enter rollout. Slurm job
  `165816` / `server46` failed the render canary at `render_rgb_array_start`
  with exit code `124` and wrote `blocked_render_canary_failed_no_rollout`.
  Files written: `manifest.txt` and `classification.txt`; no `summary.json`,
  action trace, Cosmos artifacts, or video exist. This is infrastructure
  failure evidence only, not Oracle evidence. Exclude `server46` from the next
  render-bearing attempt.
- Peg-disturbance retry `peg_disturb/try16` reran the same
  `manual_hole_frame_servo` setting on Slurm job `165821` / `server63` and did
  complete rollout. It passed protocol artifact audit with 4 premotion Cosmos
  RGB/action reports, 1 postmotion report, 4 Cosmos-derived dynamic actions,
  and an approved physical peg perturb (`observed_delta` about
  `[0.0141, -0.0346, 0.0177]`; cosine about `0.940`; fraction about `0.924`).
  It entered the manual finisher at step 142 and ran 180
  `oracle_physical_manual_finisher` rows.
- `peg_disturb/try16` is not a success. It ended with
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  and `snap_detected=false`. Final `peg_head_l2` was about `0.1843`, with
  final `peg_head_at_hole` about `[-0.1043, 0.1498, -0.0255]`; the large
  lateral error shows this manual hole-frame finisher did not align the peg
  before insertion. The next retry should use an existing staged manual
  physical finisher that aligns before pushing, with short run name
  `peg_disturb/try17`.
- Peg-disturbance retry `peg_disturb/try17` requested
  `manual_staged_hole_servo`, but it did not enter rollout. Slurm job
  `165844` / `server27` failed the render canary at `render_rgb_array_start`
  with exit code `124` and wrote `blocked_render_canary_failed_no_rollout`.
  Files written: `manifest.txt` and `classification.txt`; no `summary.json`,
  action trace, Cosmos artifacts, or video exist. This is infrastructure
  failure evidence only, not Oracle evidence. Exclude `server27` from the next
  render-bearing attempt; retry the same staged manual controller under the
  next short run name `peg_disturb/try18`.
- Peg-disturbance retry `peg_disturb/try18` requested the same
  `manual_staged_hole_servo` setting on Slurm job `165851` / `server63`, but
  it also did not enter rollout. The render canary hung at
  `render_rgb_array_start`, timed out with exit code `124`, and wrote
  `blocked_render_canary_failed_no_rollout`. Files written:
  `manifest.txt` and `classification.txt`; no `summary.json`, action trace,
  Cosmos artifacts, or video exist. This is infrastructure failure evidence
  only, not Oracle evidence. The next attempt should keep the short name
  pattern and request a node that has previously produced Phase 03 video.
- `peg_disturb/try19` targeted `server64` and `peg_disturb/try20` targeted
  `server56`, but both Slurm allocation attempts were canceled while still
  pending. Neither wrote a manifest, action trace, summary, Cosmos artifacts,
  or video. They are scheduling attempts only, not Oracle evidence and not
  render failures.
- `peg_disturb/try21` was canceled while still pending, before allocation and
  before any manifest or rollout artifacts existed. It is a scheduling attempt
  only, not Oracle evidence.
- `peg_disturb/try22` requested `server63`, got Slurm job `165883`, wrote
  `manifest.txt`, and then failed before rollout in the render canary:
  `render_rgb_array_start` timed out with exit code `124`. Files written:
  `manifest.txt` and `classification.txt`; no `summary.json`, action trace,
  Cosmos artifacts, or video exist. This is infrastructure failure only.
- After `try22`, `scripts/slurm/run_phase03_oracle_full_pipeline_in_allocation.sh`
  and `scripts/training/eval_dp_oracle_full_pipeline.py` were patched so the
  render canary and full rollout use the same configurable
  `RENDER_SHADER_PACK`, defaulting to `minimal`. This keeps the required RGB
  video path and render canary, but avoids hard-coding the default shader that
  repeatedly hung at `render_rgb_array_start`.
- `peg_disturb/try23` used that patched `RENDER_SHADER_PACK=minimal` path on
  Slurm job `165900` / `server63`, but the render canary still hung at
  `render_rgb_array_start` and timed out with exit code `124`. Files written:
  `manifest.txt` and `classification.txt`; no `summary.json`, action trace,
  Cosmos artifacts, or video exist. This is infrastructure failure only, not
  Oracle evidence. Because `server63` has now failed both default and minimal
  shader canaries in this retry window, the next retry should use a different
  node with the minimal shader canary.
- `peg_disturb/try24` used `RENDER_SHADER_PACK=minimal` on Slurm job `165910`
  / `server27`, but the render canary also hung at `render_rgb_array_start`
  and timed out with exit code `124`. Files written: `manifest.txt` and
  `classification.txt`; no `summary.json`, action trace, Cosmos artifacts, or
  video exist. This is infrastructure failure only, not Oracle evidence.
- `peg_disturb/try25` used `RENDER_SHADER_PACK=minimal` on Slurm job `165918`
  / `server28`, but the render canary also hung at `render_rgb_array_start`
  and timed out with exit code `124`. Files written: `manifest.txt` and
  `classification.txt`; no `summary.json`, action trace, Cosmos artifacts, or
  video exist. This is infrastructure failure only, not Oracle evidence.
- After `try25`, `scripts/world_model/render_min_canary.py` was patched so the
  default canary render API is `env.render()`, matching the full pipeline's
  main video path, instead of directly calling
  `env.unwrapped.render_rgb_array("render_camera")`. The wrapper records
  `render_canary_api` in the manifest and still blocks before rollout unless a
  real RGB frame is rendered and written.
- `peg_disturb/try26` used the patched `env.render()` canary with
  `RENDER_SHADER_PACK=minimal` on Slurm job `165931` / `server28`, but it still
  hung at `render_gym_start` and timed out with exit code `124`. Files written:
  `manifest.txt` and `classification.txt`; no `summary.json`, action trace,
  Cosmos artifacts, or video exist. This is infrastructure failure only, not
  Oracle evidence.
- `peg_disturb/try27` used the patched `env.render()` canary with
  `RENDER_SHADER_PACK=minimal` on Slurm job `165942` / `server46`, but it still
  hung at `render_gym_start` and timed out with exit code `124`. Files written:
  `manifest.txt` and `classification.txt`; no `summary.json`, action trace,
  Cosmos artifacts, or video exist. This is infrastructure failure only, not
  Oracle evidence.
- `peg_disturb/try28` used the patched `env.render()` canary with
  `RENDER_SHADER_PACK=minimal` on Slurm job `165952` / `server30`, but it still
  hung at `render_gym_start` and timed out with exit code `124`. Files written:
  `manifest.txt` and `classification.txt`; no `summary.json`, action trace,
  Cosmos artifacts, or video exist. This is infrastructure failure only, not
  Oracle evidence. The available tested nodes in this retry window cannot
  produce the required first RGB render frame, so no Oracle rollout with valid
  video evidence can start until a render-capable node / render stack is
  available.
- Render GPU probe `render_probe/try01` on Slurm job `165968` / `server28`
  tested three visible GPUs, including bus ids `29:00`, `5C:00`, and `AA:00`,
  with `RENDER_SHADER_PACK=minimal` and `RENDER_CANARY_API=gym`. All three
  timed out at `render_gym_start`, so no full Oracle run was launched. This is
  infrastructure diagnosis only, not Oracle evidence. Because `try16` had
  succeeded with the old `default` shader plus direct `render_rgb_array`
  canary path, the next render probe should test that exact canary path before
  giving up on the currently available GPUs.
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
- The probe-launched full run at `render_probe/try29` completed the full
  protocol and passed artifact audit: 4 premotion Cosmos RGB/action reports,
  1 postmotion report, 4 Cosmos-derived dynamic actions, physical peg
  perturbation matching the source key (`cosine=0.9405`, fraction `0.9768`),
  180 `manual_staged_hole_servo` finisher rows, `videos/raw.mp4`, and
  `videos/annotated.mp4`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  and final `peg_head_l2` was about `0.1364`. The discontinuity audit reports
  `snap_detected=false`. The staged finisher aligned y/z but did not make
  enough insertion-axis progress; the next retry should use the corrected
  `peg_disturb` run group and stronger/longer staged insertion parameters.
- Render GPU probe `render_probe/try04` on Slurm job `166014` / `server63`
  used the corrected auto-launch path and wrote the full run to the intended
  short path `peg_disturb/try29`. The run completed the full protocol and
  passed artifact audit: 4 premotion Cosmos reports, 1 postmotion report,
  4 Cosmos dynamic actions, physical peg perturbation matching the source key
  (`cosine=0.9459`, fraction `0.9514`), 320 `manual_staged_hole_servo`
  finisher rows, `videos/raw.mp4`, and `videos/annotated.mp4`. It is not a
  success: `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.2032`, and final `peg_head_at_hole` about
  `[-0.0974, 0.1727, -0.0440]`. The discontinuity audit reports
  `snap_detected=false`, but the summary records
  `target_motion_state_intervention_used=true`, so this stays failed diagnostic
  evidence only and must not be reported as method progress. Trace review shows
  the finisher reached about `peg_head_l2=0.0831` near step 188 with y/z nearly
  aligned, then continued fixed-yaw/alignment actions swept the peg away.
- `scripts/training/eval_dp_oracle_full_pipeline.py` has been patched so
  target-motion residuals at or below `1e-6 m` are ignored. This is needed for
  the approved `peg_disturb_seed1051032_idx0008` key, whose source protocol has
  a `7e-9 m` target delta and should be treated as peg disturbance, not target
  state intervention.
- `render_probe/try05` on Slurm job `166046` launched `peg_disturb/try30` after
  real RGB canary frames. The run completed the full protocol and passed
  artifact audit: 4 premotion Cosmos reports, 1 postmotion report, 4 Cosmos
  dynamic actions, matched physical peg perturbation, 140
  `manual_staged_hole_servo` finisher rows, `videos/raw.mp4`, and
  `videos/annotated.mp4`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.0979`, and best finisher `peg_head_l2` about
  `0.0966`. The discontinuity audit reports `snap_detected=false`, and the
  target residual fix worked: `target_motion_state_intervention_used=false`.
  Compared with `try29`, y/z stayed aligned, but insertion-axis progress was
  still insufficient.
- `render_probe/try07` on Slurm job `166078` launched `peg_disturb/try32` after
  real RGB canary frames. The run completed the full protocol and passed
  artifact audit: 4 premotion Cosmos reports, 1 postmotion report, 4 Cosmos
  dynamic actions, matched physical peg perturbation, 220
  `manual_staged_hole_servo` finisher rows, `videos/raw.mp4`, and
  `videos/annotated.mp4`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.1170`, and best finisher `peg_head_l2` about
  `0.0991`. The discontinuity audit reports `snap_detected=false`; target
  residual accounting remains fixed with `target_motion_state_intervention_used=false`.
  Small continuous yaw did not solve insertion and worsened the final state.
- `render_probe/try08` on Slurm job `166117` launched `peg_disturb/try33` after
  real RGB canary frames. The run completed the full protocol and passed
  artifact audit: 4 premotion Cosmos reports, 1 postmotion report, 4 Cosmos
  dynamic actions, matched physical peg perturbation, 220
  `manual_staged_hole_servo` finisher rows, `videos/raw.mp4`, and
  `videos/annotated.mp4`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.1537`, and best finisher `peg_head_l2` about
  `0.1027`. The discontinuity audit reports `snap_detected=false`; target
  residual accounting remains fixed with `target_motion_state_intervention_used=false`.
  The yaw stop threshold `0.10` was too late to prevent late yaw damage.
- `render_probe/try09` on Slurm job `166157` found a render-capable GPU on
  `server63` and launched `peg_disturb/try34` with `MANUAL_YAW_ACTION=0.22`
  and `MANUAL_YAW_STOP_L2=0.13`. The run completed the full protocol and passed
  artifact audit: 4 premotion Cosmos reports, 1 postmotion report, 4 Cosmos
  dynamic actions, matched physical peg perturbation, 220
  `manual_staged_hole_servo` finisher rows, `videos/raw.mp4`, and
  `videos/annotated.mp4`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.1563`, and best finisher `peg_head_l2` about
  `0.1025`. The discontinuity audit reports `snap_detected=false`; target
  residual accounting remains fixed with `target_motion_state_intervention_used=false`.
  Raising the yaw stop threshold to `0.13` did not improve insertion.
- After `try34`, `scripts/training/eval_dp_oracle_full_pipeline.py` was patched
  so `manual_staged_hole_servo_action` actually honors `MANUAL_YAW_STOP_L2`.
  The previous yaw-stop option only applied to `manual_hole_frame_servo_action`,
  so staged finisher runs kept yawing through the near-hole band.
- `render_probe/try10` on Slurm job `166189` found a render-capable GPU on
  `server63` and launched `peg_disturb/try35` with `MANUAL_YAW_ACTION=0.22`
  and `MANUAL_YAW_STOP_L2=0.145`. The run completed the full protocol and
  passed artifact audit: 4 premotion Cosmos reports, 1 postmotion report, 4
  Cosmos dynamic actions, matched physical peg perturbation, 220
  `manual_staged_hole_servo` finisher rows, `videos/raw.mp4`, and
  `videos/annotated.mp4`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.1062`, and best finisher `peg_head_l2` about
  `0.0996`. The discontinuity audit reports `snap_detected=false`; target
  residual accounting remains fixed with `target_motion_state_intervention_used=false`.
  The yaw stop now works: 23 finisher rows used yaw `0.22`, then 197 rows used
  yaw `0`. The remaining failure is insufficient insertion-axis progress after
  alignment, so the next retry should keep yaw stop active and increase the
  pure insertion push.
- `render_probe/try11` on Slurm job `166208` found a render-capable GPU on
  `server63` and launched `peg_disturb/try36` with patched yaw stop,
  `MANUAL_INSERT_SPEED=0.12`, and `MANUAL_FORWARD_GAIN=3.0`. The run completed
  the full protocol and passed artifact audit: 4 premotion Cosmos reports, 1
  postmotion report, 4 Cosmos dynamic actions, matched physical peg
  perturbation, 220 `manual_staged_hole_servo` finisher rows, `videos/raw.mp4`,
  and `videos/annotated.mp4`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.1373`, and best finisher `peg_head_l2` about
  `0.0989`. The discontinuity audit reports `snap_detected=false`; target
  residual accounting remains fixed with `target_motion_state_intervention_used=false`.
  Stronger insertion push did not solve the peg-disturb case and worsened the
  final state, so do not keep blindly increasing force / forward gain.
- After `try36`, `scripts/training/eval_dp_oracle_full_pipeline.py` gained
  `manual_staged_twist_insert`, a physical-action finisher that applies
  configurable roll/pitch/yaw only during the insertion stage. This was
  motivated by the metric-true `h5_constant/try01` DP finisher trace, but that
  trace is now rejected as strict success because of target-assisted insertion
  concern.
- `render_probe/try12` on Slurm job `166228` found a render-capable GPU on
  `server63` and launched `peg_disturb/try37` with `manual_staged_twist_insert`,
  insert roll `-0.18`, and insert yaw `-0.28`. The run completed the full
  protocol and passed artifact audit: 4 premotion Cosmos reports, 1 postmotion
  report, 4 Cosmos dynamic actions, matched physical peg perturbation, 220
  manual finisher rows, `videos/raw.mp4`, and `videos/annotated.mp4`. It is not
  a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final `peg_head_l2` about `0.1117`, and best finisher `peg_head_l2` about
  `0.0950`. The discontinuity audit reports `snap_detected=false`; target
  residual accounting remains fixed with `target_motion_state_intervention_used=false`.
  The twist ran for 131 finisher rows and changed peg pose but still did not
  insert; do not use long continuous twist as a success path.
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
- `render_probe/try13` on Slurm job `166277` found a render-capable GPU on
  `server63` and launched `peg_disturb/try38` with `manual_staged_dp_rot`. The
  run completed the full protocol and passed artifact audit: 4 premotion Cosmos
  reports, 1 postmotion report, 4 Cosmos dynamic actions, matched physical peg
  perturbation, 220 hybrid finisher rows, `videos/raw.mp4`, and
  `videos/annotated.mp4`. It is not a success:
  `classification=physical_failure_not_inserted_full_pipeline_attempted`,
  `simulator_success_metric=false`, `visual_full_insertion_confirmed=false`,
  final and best `peg_head_l2` about `0.1503`. The discontinuity audit reports
  `snap_detected=false`; target residual accounting remains fixed with
  `target_motion_state_intervention_used=false`. The DP rotation components were
  too small to recover pose/axis mismatch.
- Use `scripts/slurm/phase03_peg_disturb.sh` as the short launcher for the
  next peg-disturbance retry inside a tmux-held allocation:
  `srun --pty bash scripts/slurm/phase03_peg_disturb.sh`. It sets the approved
  `peg_disturb_seed1051032_idx0008` source key, calibrated force perturb, short
  `RUN_GROUP=peg_disturb`, and `RUN_NAME` default, then delegates to the
  guarded full-pipeline wrapper. Do not paste long one-off command strings for
  this retry unless overriding a specific parameter.
- New Cosmos-call outputs must use the short grouped structure introduced after
  this retry: `cosmos_policy/pre/00/...` and `cosmos_policy/post/00/...`, with
  generic leaf names such as `prefix_rgb.mp4`, `actions/sample.json`, and
  `outputs/sample/vision.mp4`.
- New visual-review outputs must also group by folder instead of repeating the
  same prefix in filenames: use `review/oblique/raw.mp4`,
  `review/oblique/annotated.mp4`, and `review/oblique/trace.json`, not
  `oblique_replay_annotated.mp4`.
- Active Phase 03 wrappers now reject new long names containing `p03_`, date
  strings, host/job metadata, or `full_pipeline_<details>`. Rewrite them as
  short grouped paths such as `peg_disturb/try39` before launch.
- Next peg-disturb retry should use the approved source key with the logged
  soft-insert gate enabled, for example `MANUAL_SOFT_INSERT_THRESHOLD=0.025`
  and `MANUAL_SOFT_INSERT_SCALE=0.35`, so the physical finisher does not stall
  just outside the hard y/z alignment threshold. Keep `method_evidence_allowed=false`
  and reject any snap / wall / disappearance artifact.
- `peg_disturb/try40` showed the soft-insert gate improves `manual_staged_dp_rot`
  but does not solve insertion: best `peg_head_l2` was about `0.0973` and y/z
  stayed about `0.019m`. The next retry should remove DP rotation from the
  finisher and test soft-insert with pure `manual_staged_hole_servo` plus stronger
  lateral/vertical correction.
- `peg_disturb/try41` through `peg_disturb/try45` have now tested the next
  physical-finisher variants on the approved peg-disturb key: pure staged
  servo with stronger y/z correction, staged twist, direct pose servo, reversed
  smaller pose servo, and finally `manual_align_then_dp`. All completed
  protocol-compliant runs that reached rollout passed artifact audit and had
  `snap_detected=false`, but none inserted. The best current peg-disturb result
  is `try45`, with final `peg_head_l2` about `0.0950` and best about `0.0945`.
  Its trace confirms the allowed near-target DP handoff actually happened
  after manual alignment, so the remaining failure is a real physical/control
  plateau on this key rather than a skipped DP gate.
- `peg_disturb/try46` tested the second approved peg-disturbance key,
  `peg_disturb_seed40751016_pseed42751016_idx13000`, on Slurm job `166919` /
  `server57`. It completed the full protocol with 4 premotion Cosmos reports,
  4 postmotion Cosmos reports, 32 Cosmos-derived dynamic actions, videos, and
  `artifact_audit.ok=true`. The physical perturbation audit passed directionally
  (`peg_observed_delta_cosine` about `0.886`, observed fraction about `1.94`),
  and the run had `snap_detected=false` and `peg_state_guard.ok=true`. It still
  failed: `simulator_success_metric=false`, final `peg_head_l2` about `0.2229`,
  and `finisher_start_step=null`, meaning the near-target gate was never
  reached and the manual/DP finisher did not run. This is not completed
  peg-disturbance coverage.
- `peg_disturb/try47` was launched on Slurm job `166935` / `server57` to loosen
  the near-target gate on the same second peg-disturbance key, but the user
  stopped the repeated sweep before completion. The run was canceled before
  summary/action-trace/full-video artifacts were written and is classified as
  `canceled_by_user_request_partial_cosmos_evidence_only`. Do not launch more
  peg-disturbance retries without explicit user direction and a concrete new
  diagnosis.
- Overall Oracle task completion remains incomplete because successful
  direction coverage, peg/wooden-stick disturbance, and successful coverage
  across multiple approved 733 keys are still missing.
- The older archived video trace references the existing active Phase 02 RGB
  Cosmos output:
  `experiments/maniskill/runs/02_cosmos_imagination/p02_cosmos_rgb6_config_20260703_012531_162203_server44/cosmos_outputs/hole_late_fast_shift_seed10300001_idx5000/vision.mp4`.
  The active full-pipeline reruns above do rerun Cosmos inline from the live
  RGB history and record both premotion and postmotion Cosmos RGB/action
  artifacts.
- Earlier manual render and `RecordEpisode` rendering attempts hung under
  server30/server05 allocations or failed before the wrapper `PYTHONPATH` fix.
- Failed / interrupted Phase 03 Oracle attempts were archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/03_oracle/`
  and matching logs under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/logs/03_oracle/`.
- `salloc` job `162568` on `server64` completed and was released after the
  annotated video was written. Queue is clear for that job.
- Corrected no-teleport retry ran in `salloc` job `162757` on `server52` and
  was released after the annotated video was written. Queue is clear for that
  job.
