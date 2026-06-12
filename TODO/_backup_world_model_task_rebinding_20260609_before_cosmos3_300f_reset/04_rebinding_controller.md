# Rebinding Controller TODO

## First Controller

- [x] Implement candidate future `tau` sampler.
- [x] Build future pre-insert task frames from predicted hole pose.
- [x] Add reachability and basic collision checks.
- [x] Add servo/IK bridge to candidate task frame.
- [x] Add short-chunk receding-horizon replanning.
- [x] Add discrepancy-triggered rebind.
- [x] Add conservative DP handoff interface using `C_pi` model directories.
      Evaluation waits for compliant `C_pi` job `94471` to finish and inspect.
- [x] Add `C_pi` online feature-contract checks and ensemble disagreement gate.
      `HANDOFF_MODE=cpi` requires the score to exceed `CPI_THRESHOLD` and
      ensemble standard deviation to stay below `CPI_MAX_STD`.
- [x] Add calibrated `C_pi` no-video evaluation wrapper. It waits for a
      compliant `C_pi` inspection/calibration, reads the paired
      `threshold_std_gates` calibration row, exports both `CPI_THRESHOLD` and
      `CPI_MAX_STD`, and then calls the existing controller evaluator without
      changing the final dynamic success metric.
- [x] Add learned object-state world-model predictor path. `rebind_world_model`
      requires world-model checkpoint directories, verifies manifest contracts,
      builds the exact online feature history used in training, and gates
      bridge plans by ensemble uncertainty.
- [x] Make streaming discrepancy compare against a separate one-step
      prediction, so future-horizon bridge targets are not miscounted as
      immediate prediction failures.
- [x] Fix bridge candidate scoring to evaluate whether a future task-frame
      target can be physically approached within the candidate horizon. The old
      score minimized current distance and made `rebind_cv` collapse to
      `tau=0` in smoke jobs `94519` and `94521`.
- [x] Add infeasible reporting.
- [x] Summarize current state-controller evidence after the resource/smoke
      correction. Summary inputs and outputs:
      `experiments/world_model_task_rebinding/rebinding_controller/state_current_summary_inputs_20260602_2235.txt`,
      `state_current_summary_20260602_2235.json`, and
      `state_current_summary_20260602_2235.md`. The current state scaffold
      picture is mixed: DP-only and one CV task-frame run fail; one learned-WM
      task-frame no-video run succeeds; the full-shard learned-WM and C_pi
      guarded runs listed there fail; peg-drop/regrasp succeeds in no-video
      and video/contact-sheet scaffold runs. This is state/oracle scaffold
      evidence only, not RGB-D method evidence.
- [x] Rule out CPU-only controller rollout as an immediate smoke path. Debug
      job `99136` used `--no-cuda` and `SAVE_VIDEO=false` but failed during
      ManiSkill/SAPIEN environment construction with Vulkan
      `ErrorIncompatibleDriver`; controller rollout still needs a Slurm
      allocation with working Vulkan/GPU context even when video is disabled.
- [x] Localize current state insertion-axis bottleneck and prevent RGB-D
      controller auto-running a known-risk config. Offline H5/event-log
      diagnostic artifacts
      `state_axis_progress_diagnostic_20260602_2238.json` and
      `state_axis_bridge_plan_samples_20260602_2240.json` show seed-7400
      failures are not merely lateral error: full-shard WM stalls near
      `x=-0.1005` with `yz<0.005m` for many steps while still commanding
      `task_servo_axis_step_m=0.008`, and the C_pi guarded tcp-continuation
      config often blocks the insert axis with `lateral_outside_manifold`.
      Updated RGB-D controller defaults/submitter toward the state-supported
      move-stop config `phase_hybrid + task_frame_projected +
      tcp_continuation orientation + no insert guard`. This preserves the
      final inserted-state metric and is not a success claim.
- [x] Run focused state/oracle smoke for the seed-7400 insertion-manifold
      question, without changing the task metric. Submitted CV chain
      `99189 -> 99190 -> 99191` and learned-WM chain
      `99192 -> 99193 -> 99194` with `SCENARIO=hole_move_stop`,
      `SEED_START=7400`, `BRIDGE_SERVO_REFERENCE=phase_hybrid`,
      `BRIDGE_SERVO_MODE=task_frame_projected`,
      `BRIDGE_ORIENTATION_REFERENCE=peg_alignment`, and
      `TASK_SERVO_INSERT_MANIFOLD_GUARD=true`. First-principles reason: the
      known full-shard state failure is laterally aligned but stalls on
      insertion x, so this tests whether peg-alignment manifold control fixes
      the physical insertion failure or simply blocks the axis. This is
      state/oracle scaffold evidence only, not RGB-D method evidence. Evidence
      note:
      `docs/world_model_task_rebinding/2026-06-02_resource_probe_and_state_smoke_submission_2248.md`.
      Result at `2026-06-02T23:16+08:00`: both no-video state/oracle gates
      passed. CV job `99189` reached first success at step `177`, final
      peg-head hole-frame `x=-0.01113m`, `YZ=0.00366m`; learned-WM job
      `99192` reached first success at step `184`, final `x=-0.01102m`,
      `YZ=0.00368m`. This proves the seed-7400 state failure was an
      insertion-manifold/peg-orientation control issue. It does not prove
      learned-WM superiority because CV also passes, and it is not RGB-D
      evidence because `slot_source=oracle` and no video/contact sheet exists.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_front48_repair_and_state_smoke_2316.md`.
- [x] Apply the seed-7400 insertion-manifold lesson to pending RGB-D
      controller branches before they run. Future
      `evaluate_rgbd_rebinding_controller_video.sbatch` and
      `submit_auditable_rgbd_method_chain_job94676.sh` controller exports now
      default to `BRIDGE_ORIENTATION_REFERENCE=peg_alignment` and
      `TASK_SERVO_INSERT_MANIFOLD_GUARD=true`, while preserving the same
      final-state success metric, RGB-D slot source, RGB-D-derived
      world-model gates, and video review. Old repaired-chain pending
      controller tail `99220/99221/99222` was canceled before allocation and
      replaced by `99252/99253/99254` after the same world-model eval job
      `99219`. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_controller_tail_pegalign_replacement_2321.md`.
- [ ] Run compact state/oracle generalization smoke matrix after the
      move-stop seed-7400 physical insertion fix. Added
      `scripts/slurm/evaluate_state_rebinding_smoke_matrix.sbatch` and
      submitted fixed-path single-GPU job `99346` with no video and
      `SLOT_SOURCE=oracle`; earlier `99308` and `99331` were canceled before
      allocation. After the user explicitly requested active state-side smoke
      progress instead of waiting on the RGB-D visual gate, `99346` was
      corrected with `scontrol update JobId=99346 Dependency= Nice=0`; it is
      now dependency-free, immediately eligible, and normal-priority. It
      serially evaluates `hole_constant` CV/WM,
      `hole_reverse` CV/WM, `peg_disturb` CV regrasp, and `peg_drop` CV
      regrasp under `phase_hybrid + task_frame_projected + peg_alignment +
      insert_manifold_guard`. First-principles reason: prove or falsify that
      the same task-frame bridge can physically close continuous/reverse hole
      motion and peg-event recovery in state space before feeding it
      RGB-D-derived slots. This remains scaffold evidence only. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-03 00:14 state-smoke scheduling correction:
      `99346` remains the earliest available state/oracle matrix path after
      live probes; new 1GPU probes would start later. Its walltime was reduced
      from `01:30:00` to `00:45:00` to better fit backfill while preserving
      all six cases, state/oracle slot source, no-video setting, controller
      config, and final inserted-state success gate. The start estimate is
      still `2026-06-03T13:00:00`, so the current explicit state result is:
      move-stop seed `7400` passes for both CV (`99189`, first success step
      `177`, final hole-frame YZ `0.00366m`) and learned-WM (`99192`, first
      success step `184`, final YZ `0.00368m`), and peg-drop/regrasp V9 has
      video/contact-sheet scaffold success (`95107`, first success step
      `234`, final YZ `0.00328m`). Continuous, reverse, and peg-disturb
      state conclusions wait for `99346`; none of these are RGB-D method
      evidence.
- [x] 2026-06-03 00:36 short-backfill state-smoke source support:
      added `STATE_MATRIX_CASES` to
      `scripts/slurm/evaluate_state_rebinding_smoke_matrix.sbatch`, so future
      Slurm submissions can run one or several exact existing matrix cases
      without changing scenario seeds, state/oracle slot source, controller
      config, no-video setting, inspection, success gate, or final inserted
      metric. This supports active state-side smoke progress when only short
      one-GPU backfill slots are available. `bash -n` passed. Existing
      submitted job `99346` is unchanged because Slurm keeps the submitted
      script snapshot. This is scheduling/tooling support only, not state
      result evidence.
- [ ] 2026-06-03 00:46 state matrix is running:
      `99346` started on `server42` at `2026-06-03T00:44:53`. It is currently
      running the first case `constant_cv`; stderr is empty and `summary.tsv`
      has only the header so far. Short-backfill split dry-runs for
      `reverse_cv` were later than this running matrix and were not submitted.
      Await the completed `summary.tsv`, inspections, and gates before
      interpreting continuous, reverse, or peg-disturb state outcomes.
- [x] 2026-06-03 00:53 first state matrix row:
      `constant_cv` completed under oracle slots with unchanged final gate and
      failed: `eval_status=0`, `inspect_status=0`, `gate_status=65`,
      `success_at_end=false`, `first_success_step=-1`, final hole-frame
      `x=-0.11633`, `YZ=0.00508`. Interpretation: continuous constant
      target motion is not solved by the CV/task-frame bridge alone in this
      seed. This is state scaffold evidence only; the matrix is still running
      subsequent learned-WM, reverse, and peg-event cases.
- [x] 2026-06-03 01:00 additional state matrix rows:
      `constant_wm` succeeded (`first_success_step=211`, final
      `x=-0.01393`, `YZ=0.00374`), while `reverse_cv` and `reverse_wm` both
      succeeded (`first_success_step=146/147`, final `YZ≈0.0034`). Interim
      state interpretation: learned state WM adds value on continuous constant
      motion where CV failed, but reverse motion is currently solvable by both
      predictors in oracle state space. Peg-disturb and peg-drop rows are
      still pending. This remains state/oracle scaffold evidence only.
- [x] 2026-06-03 01:02 state matrix complete:
      `99346` completed all six rows. Final state/oracle scaffold results:
      `constant_cv` failed; `constant_wm` succeeded; `reverse_cv` succeeded;
      `reverse_wm` succeeded; `peg_disturb_cv_regrasp` failed with final
      `x=-0.03222`, `YZ=0.00352`; `peg_drop_cv_regrasp` succeeded with first
      success step `185`, final `x=-0.01296`, `YZ=0.00376`. This is a clear
      state result: state WM matters for continuous constant target motion;
      reverse target motion is easy for this state controller; peg drop is
      recoverable by regrasp; peg disturb in gripper is still not solved by
      the current CV regrasp branch. None of this is RGB-D method evidence.
- [x] 2026-06-03 03:02 peg-disturb insert-stall diagnosis and gated patch:
      inspected the failed state/oracle `peg_disturb` runs `99346` and
      `99714`. Both keep `grasped=true`, have final lateral error around
      `3-4mm`, and report manifold-ok insert commands, but x-axis progress
      stalls before the final inserted-state threshold. Offline `_choose_bridge`
      replay confirms the controller is still commanding positive insert-axis
      steps; this is a physical continuation/stalled-contact problem after a
      changed peg/TCP relation, not a scenario-specific regrasp-disabled bug.
      Added a default-off generic `insert_stall_regrasp` trigger: when
      insertion is aligned, manifold-ok, and repeatedly commanded but peg-head
      x progress over a rolling window is below threshold, the controller can
      call the existing regrasp primitive. This does not inspect scenario
      names and does not change the final success metric. `py_compile` passed
      for `evaluate_rebinding_controller.py`; `bash -n` passed for
      `evaluate_rebinding_controller.sbatch` and
      `evaluate_state_rebinding_smoke_matrix.sbatch`. Submitted low-priority
      state/oracle smoke `100224` with `INSERT_STALL_REGRASP=true` for only
      `peg_disturb_cv_regrasp` and `peg_disturb_wm_regrasp`; it is pending
      with no start forecast and is not RGB-D method evidence.
- [x] 2026-06-03 03:20 insert-stall smoke scheduling update:
      updated pending state/oracle diagnostic job `100224` to `Nice=0` and
      `ExcNodeList=server03` after probes showed the 1GPU state smoke could
      target non-`server03` backfill without directly occupying the active
      RGB-D slot-training target node. It remains `SLOT_SOURCE=oracle`, uses
      only `peg_disturb_cv_regrasp` and `peg_disturb_wm_regrasp`, and cannot
      support a RGB-D method claim. After scheduler refresh, `squeue` reports
      a forecast start of `2026-06-03T13:00:00`; the result still waits for
      Slurm and artifacts.
- [ ] 2026-06-03 03:24 insert-stall smoke forecast drift:
      a later `squeue` check moved `100224` to a forecast start of
      `2026-06-06T00:11:23`. Do not reinterpret this as a controller result
      or a reason to change the RGB-D method path. It is still a pending
      state/oracle diagnostic only.
- [x] 2026-06-03 03:26 insert-stall replacement probe canceled:
      after fresh non-executing probes suggested a new non-`server03` state
      smoke could start earlier, submitted one replacement probe job `100450`
      for the same two state/oracle cases. Actual Slurm forecast did not give
      a stable earlier start, so `100450` was canceled before allocation.
      Current insert-stall diagnostic remains `100224` only. No result was
      produced, and no RGB-D method claim is involved.
- [ ] 2026-06-03 03:27 insert-stall smoke current forecast:
      latest `squeue` reports `100224` forecast at `2026-06-03T07:00:00`.
      This is still only a forecast; no controller result exists yet.
- [x] 2026-06-03 03:47 insert-stall diagnostic logging preflight:
      inspected the pending `100224` submitted batch snapshot and current
      `evaluate_rebinding_controller.py` before the job runs. The diagnostic
      keeps `STATE_MATRIX_CASES=peg_disturb_cv_regrasp,peg_disturb_wm_regrasp`,
      `SLOT_SOURCE=oracle`, unchanged final-state gate, and
      `INSERT_STALL_REGRASP=true`. The controller writes `insert_stall_report`
      into each event log when applicable, including current/past hole-frame
      x, current lateral YZ, recent commanded insert steps, recent axis blocks,
      progress over the rolling window, trigger reason, and the stalled bridge
      plan. This means a future failure can be classified from artifacts
      instead of guessed. It is still only state/oracle diagnostic readiness,
      not RGB-D method evidence.
- [x] 2026-06-03 00:03 RGB-D controller static boundary audit:
      `slot_source=rgbd` requires a RGB-D slot ensemble, online control slots
      are predicted from synchronized RGB-D observations plus proprioception,
      and oracle metric slots are recorded only for perturbation scheduling
      and final-state success scoring. The controller also requires
      RGB-D-derived world-model inspection evidence and member RGB-D auxiliary
      feature evidence before using the learned world model. This is static
      contract evidence only; runtime method evidence still waits for the
      RGB-D chain and video review. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] Write the current state-scaffold result clearly:
      `docs/world_model_task_rebinding/2026-06-02_state_scaffold_current_result_2252.md`
      records the causal state-side picture. DP-only fails the dynamic
      move-stop task; CV and full-shard learned-WM task-frame projected runs
      on seed `7400` both reduce lateral error to millimeters but stall before
      final insertion; learned-WM seed `7500` reaches final no-video dynamic
      success; peg-drop/regrasp V9 has direct contact-sheet/video evidence for
      a state/oracle regrasp branch. The remaining state-side physical
      question is insertion-manifold control, not restoration or threshold
      relabeling.

Controller utilities:

- `scripts/world_model/evaluate_rebinding_controller.py`
- `scripts/world_model/inspect_rebinding_controller_run.py`
- `scripts/slurm/evaluate_rebinding_controller.sbatch`
- `scripts/slurm/inspect_rebinding_controller_run.sbatch`
- `scripts/slurm/gate_rebinding_controller_success.sbatch`
- Smoke job `94518` completed with `CONTROL_POLICY=rebind_cv`,
  `HANDOFF_MODE=never`, one `hole_move_stop` episode, and video enabled. It
  failed before bridge engagement and is controller/video plumbing evidence,
  not method success.
- Code-path smoke job `94519` completed with the same controller settings but
  `SAVE_VIDEO=false`. It exposed the horizon-scoring issue recorded below and
  is not method evidence.
- Added `CONTROL_POLICY=track_current` as the simple target-tracking baseline:
  same bridge mechanism, but only `tau=0` instead of future intercept.
- No-video comparison smokes `94521` (`track_current`) and `94522`
  (`dp_only`) completed and failed, as expected under the move-stop
  perturbation.
- Early smoke evidence note:
  `docs/world_model_task_rebinding/2026-06-02_controller_smoke_94518_94522.md`.
  It records `94518` as base-DP grasp-precondition failure, `94519` as
  horizon-0 bridge-score collapse, `94521` as `track_current` baseline
  failure, and `94522` as `dp_only` dynamic failure.
- `rebind_world_model` is implemented and dependent video smoke `94564` is
  completed after object-world-model inspection job `94620` passed, not raw
  training job `94442`. It used the four final `model.pt` files. Inspection
  `94696` passed and the contact sheet was opened directly. The run failed:
  final success was false, final YZ error was `0.1162m`, and the likely bug was
  immediate retreat on short `grasped=False` flickers.
- Additional dependent video smokes on the same compliant `94442` model also
  completed and failed. `94576` (`hole_constant`) was inspected by CPU job
  `94700`; final success was false, final YZ error was `0.1190m`, and the
  contact sheet shows the peg dropped/left near the box while the arm retreats.
  `94577` (`hole_reverse`) was inspected by CPU job `94717`; final success was
  false, final YZ error was `0.1026m`, and the contact sheet shows the peg
  carried to the box region but not inserted. Both runs have many
  `peg_not_grasped_regrasp_disabled` safe-retreat steps, so they reinforce the
  grasp-loss/regrasp controller gap rather than falsifying world-model task
  rebinding.
- Added `grasp_lost_grace_steps` to distinguish short contact/grasp predicate
  flicker from persistent peg loss. Same-seed no-video validation `94698`
  completed with `GRASP_LOST_GRACE_STEPS=4`; CPU inspection `94699` passed.
  The run still failed (`success_at_end=false`, final `YZ=0.1195m`) even
  though final grasp was true. Success gate `94740` failed by design, so stale
  gated video jobs were not used as evidence.
- Fixed-score same-seed CV validation `94591` completed and inspection `94741`
  passed. It had zero retreat and final grasp true, but still failed with
  final `YZ=0.0871m`. This shows the bridge target, not only retreat/grasp
  flicker, is a physical controller gap.
- Added `bridge_servo_reference=tcp_continuation` as the default bridge target.
  It uses hole-local TCP continuation poses measured from successful static
  rollouts: align `[-0.273, 0.0, 0.0]`, insert
  `[-0.169, 0.003, -0.004]`. The old peg-head bridge remains available as
  `bridge_servo_reference=peg_head`.
- Same-seed no-video validations for the new bridge target completed and
  failed final dynamic insertion. Initial 3-hour controller-smoke chain
  `94751`-`94760` was canceled before running to use shorter non-training
  walltime. Jobs `94763` (CV, seed `7200`) and `94764` (learned-WM `94442`,
  seed `7300`) completed; inspections `94765`/`94766` passed as failed-run
  inspections, success gates `94767`/`94768` failed by design, and dead gated
  video dependencies `94769`/`94770` plus inspections `94771`/`94772` were
  canceled.
- The TCP-continuation position fix is falsified as sufficient. In `94763` and
  `94773`, the TCP reached the measured hole-local continuation position with
  millimeter-level error, but the peg head remained `10-15cm` away in hole YZ.
  Static-success orientation analysis shows successful continuation frames keep
  hole-local peg/TCP orientation within a few degrees, while failed runs are
  tens of degrees off. The next controller change must servo the TCP pose in
  the hole task frame, including orientation, while keeping the original final
  inserted-state success metric.
- Implemented orientation-aware TCP-continuation bridge. The controller now
  supports `bridge_orientation_reference=tcp_continuation`, computes desired
  hole-local TCP rotation from static successful continuation frames, converts
  the world-frame rotation error to ManiSkill `XYZ` delta rotation, clips it by
  `max_bridge_delta_rot_rad`, and logs `bridge_delta_rot` plus rotation
  distance in each bridge plan. The Slurm wrapper records
  `BRIDGE_ORIENTATION_REFERENCE`, `ALIGN_TCP_RPY_XYZ`,
  `INSERT_TCP_RPY_XYZ`, and `MAX_BRIDGE_DELTA_ROT_RAD`.
- Inspector now reports bridge rotation behavior:
  `bridge_orientation_reference_counts`, clipped/unclipped rotation-delta norm
  stats, and `rot_distance_rad` stats. This makes orientation-aware runs
  diagnosable directly from inspection output without changing the final
  inserted-state success metric.
- Queued same-seed orientation-aware validations. CV chain:
  `94814 -> 94815 -> 94816 -> 94817 -> 94818`. Learned-WM `94442` chain:
  `94819 -> 94820 -> 94821 -> 94822 -> 94823`. Peg-drop/regrasp chain:
  `94824 -> 94825 -> 94826 -> 94827 -> 94828`. No-video jobs use 30-minute
  non-training walltime and success gates; video jobs only run if final
  dynamic success is observed.
- Added `summarize_rebinding_controller_inspections.py` and queued summary jobs
  for the same orientation-aware validation family: `94882` after no-video
  inspections `94815`/`94820`/`94825`, and video summary `94889` after video
  inspections `94818`/`94823`/`94828`. The original video summary `94881` was
  canceled before running because it required all gated-video inspections even
  though missing video artifacts are expected when no-video success gates fail.
  These summaries are comparison aids only; success claims still require the
  existing final-state gate and direct video/contact sheet review.
- Queue/config audit at `2026-06-02 06:44+08:00`: `sacct SubmitLine`
  confirms no-video jobs `94814`/`94819`/`94824` export
  `BRIDGE_SERVO_REFERENCE=tcp_continuation`,
  `BRIDGE_ORIENTATION_REFERENCE=tcp_continuation`,
  `MAX_BRIDGE_DELTA_ROT_RAD=0.08`, and `SAVE_VIDEO=false`. Their video jobs
  `94817`/`94822`/`94827` depend on success gates
  `94816`/`94821`/`94826`, export the same orientation-aware bridge settings,
  and set `SAVE_VIDEO=true`. This preserves the rule that video evidence is
  captured only after no-video final dynamic success is observed.
- Stale pending controller jobs `94578`, `94582`, `94583`, and `94590` were
  canceled before running because their run groups described pre-continuation
  controller variants and would pollute the evidence chain after the bridge
  target change.
- Full-shard learned-world-model controller validation is queued behind
  larger object-model inspections with the current v4
  `bridge_servo_mode=task_frame_projected` controller. Older position-only,
  orientation-aware, and phase-search full-shard chains were canceled before
  running once later diagnostics superseded them. Current chains are
  `95055 -> 95056 -> 95057 -> 95058 -> 95059` after strict inspection `94863`
  for job `94539`,
  `95060 -> 95061 -> 95062 -> 95063 -> 95064` after `94680` for job `94679`,
  `95065 -> 95066 -> 95067 -> 95068 -> 95069` after `94747` for job `94746`,
  and `95070 -> 95071 -> 95072 -> 95073 -> 95074` after `94810` for old
  alternate node-selection retry `94809`. The old video-job node-exclusion
  wording is superseded; future video jobs must use live Slurm state or
  targeted canaries for any job-local node decision, and must be opened
  directly if they produce contact sheets.
- Full-shard summary jobs are also queued for the v4 chains: `95075` after
  no-video inspections, `95076` for gated-video cleanup, and `95077` after
  gated-video inspections. They compare inspection JSONs across `94539`,
  `94679`, `94746`, and `94809` model-training attempts without changing
  success gates or replacing direct video review.
- Queue check at `2026-06-02 07:34+08:00`: orientation-aware no-video jobs
  `94814`/`94819`/`94824` are still pending on priority with forecast
  `2026-06-02T19:28:40`. No new orientation-aware metrics, inspection JSON, or
  contact sheet exists yet. Same-settings no-video `sbatch --test-only` probes
  with shorter 10/15-minute walltimes forecast `2026-06-09T19:27:41`; shorter
  walltime would not start earlier, so no duplicate controller jobs were
  submitted.
- Follow-up at `2026-06-02 07:36+08:00`: the same controller jobs were
  re-forecast by Slurm for `2026-06-02T10:00:12`. This improves the queue
  estimate but still provides no controller result until the jobs actually run
  and inspections/video gates produce artifacts.
- Evidence-chain preflight at `2026-06-02 07:41+08:00`: `94815`/`94820`/`94825`
  inspect the exact no-video run directories listed in
  `controller_smoke_v4_tcp_pose_continuation_novideo_inspections.txt`.
  `94818`/`94823`/`94828` inspect the exact gated-video run directories listed
  in `controller_smoke_v4_tcp_pose_continuation_video_inspections.txt`.
  Summary `94882` uses `REQUIRE_ALL=true` for no-video inspections, while
  video summary `94889` uses `REQUIRE_ALL=false`, so missing gated-video
  artifacts after failed success gates are reported rather than treated as
  summary failure. This does not alter the final-state success gate or direct
  video review requirement.
- Dead-dependency note from `2026-06-02 07:43+08:00`: previous chains
  `94769`/`94770`/`94771`/`94772` and `94776`/`94777` were manually canceled
  after success gates failed. For the current `94817`/`94822`/`94827` video
  chains, if no-video gates fail and Slurm leaves video/video-inspection jobs
  in dependency-dead pending states, cancel only those dead video branches and
  let or requeue the `REQUIRE_ALL=false` video summary. This is evidence-chain
  cleanup, not a change to evaluation or a success claim.
- Added automatic gated-video cleanup coverage at `2026-06-02 07:45+08:00`.
  Cleanup job `94929` waits on no-video gates `94816`/`94821`/`94826` and will
  cancel only failed-gate video branches before tolerant video summary `94889`
  runs. Full-shard cleanup job `94930` waits on gates
  `94867`/`94837`/`94842`/`94847` for the analogous later branches. If any gate
  passes, the cleanup job leaves that video branch alone, preserving the
  required direct video/contact-sheet review.
- Queue audit at `2026-06-02 07:50+08:00`: the same-seed orientation-aware
  no-video jobs `94814`/`94819`/`94824` are still pending on priority, now
  forecast for `2026-06-02T09:03:44` on `server31`. Their inspections, gates,
  cleanup job `94929`, strict no-video summary `94882`, and tolerant video
  summary `94889` remain dependency-pending. Artifact search found no new
  current orientation-aware inspection JSON, MP4, or contact sheet, so there is
  still nothing to claim or visually inspect.
- Static preflight at `2026-06-02 07:57+08:00`: before the queued
  orientation-aware jobs start, rechecked the TCP-pose rotation bridge against
  ManiSkill controller semantics. `PDEEPoseController` applies rotation deltas
  as Euler XYZ and left-multiplies them for
  `root_aligned_body_rotation`; `_choose_bridge` computes
  `target_tcp_rot_world @ current_tcp_rot_world.T`, so the frame direction is
  consistent. The normalized action scaling also matches ManiSkill's
  `clip_and_scale_action` path. A local small-angle round-trip check for the
  helper Euler functions had max error about `7e-9`, and
  `_load_runtime_modules()` completed successfully. This is execution
  preflight only, not controller evidence; the actual run must still show
  bridge rotation logs and final-state metrics.
- Queue/config audit at `2026-06-02 07:58+08:00`: `94814`/`94819`/`94824`
  remain priority-pending and are now forecast for `2026-06-02T09:12:45`.
  `sacct SubmitLine` confirms the no-video and gated-video jobs still export
  `BRIDGE_SERVO_REFERENCE=tcp_continuation`,
  `BRIDGE_ORIENTATION_REFERENCE=tcp_continuation`,
  `MAX_BRIDGE_DELTA_ROT_RAD=0.08`, and the intended `SAVE_VIDEO` setting.
  Inspection, summary, and cleanup inputs still point to the expected run
  directories; the learned-WM branch still sees all four `94442` member
  `model.pt`/`manifest.json`/`metrics.json` files. No new orientation-aware
  metrics, inspections, MP4s, or contact sheets exist yet.
- 2026-06-02 12:30 calibrated C_pi chain: added
  `scripts/slurm/evaluate_rebinding_controller_cpi_calibrated.sbatch` and
  submitted no-video chain `95392 -> 95393 -> 95394` after larger-label
  calibration `94660`. The first chain `95384 -> 95385 -> 95386` was canceled
  while still pending because its Slurm export stored the world-model member
  dirs as a space-separated environment value. The replacement uses
  `WORLD_MODEL_DIRS_FILE`:
  `object_state_world_model/ensemble_4gpu_from_rollout/job94539/world_model_member_dirs.txt`.
  Job `95392` uses the compliant larger object world model `94539`,
  `HANDOFF_MODE=cpi`, `CPI_TARGET_FPR=0.02`,
  `BRIDGE_SERVO_MODE=task_frame_projected`, `BRIDGE_PHASE_MODE=search`,
  `BRIDGE_ORIENTATION_REFERENCE=peg_alignment`,
  `TASK_SERVO_INSERT_MANIFOLD_GUARD=true`, and
  `TASK_SERVO_INSERT_AXIS_STEP_M=0.004`. It is no-video evidence only. If
  gate `95394` passes, submit a separate render-safe video job and directly
  inspect the resulting contact sheet/video.
- 2026-06-02 12:35 status check: no new controller video/contact sheet has
  appeared since the inspected CV peg-drop/regrasp video `95107 -> 95108`.
  Learned-WM move-stop video rerun `95191` and seed-7400 failure-debug video
  `95209` remain pending with render-safe exclusions and current scheduler
  forecast `2026-06-02T18:05:41` on `server27`. No-video insertion-manifold
  probe `95212` remains pending with forecast `2026-06-02T17:38:14`. The
  calibrated C_pi controller chain `95392 -> 95393 -> 95394` is still blocked
  on larger-label calibration `94660`, which itself waits for compliant
  inspection of running job `94542`. Do not claim video evidence or
  learned-WM dynamic insertion success until these jobs run, inspections
  complete, and the resulting video/contact sheet is opened directly.
- 2026-06-02 12:40 calibrated-controller preflight: `bash -n` passed for
  `evaluate_rebinding_controller_cpi_calibrated.sbatch`,
  `evaluate_rebinding_controller.sbatch`, inspection, and success-gate
  wrappers; `.venv/bin/python -m py_compile` passed for controller eval,
  inspection, C_pi calibration, and C_pi ensemble inspection scripts. The
  `job94539` world-model member-dir file contains the four inspected member
  directories, and each has `model.pt`, `manifest.json`, and `metrics.json`.
  Submit line for `95392` uses `WORLD_MODEL_DIRS_FILE` rather than a
  space-separated world-model export, so the previous pending-chain quoting
  issue is not present in the replacement chain.
- 2026-06-02 12:43 video-inspection preflight: controller video generation and
  inspection paths were rechecked while `95191`, `95209`, and `95212` remain
  pending. `evaluate_rebinding_controller.py` writes both `videos/*.mp4` and
  `videos/*_contact_sheet.png` when `SAVE_VIDEO=true`; the inspector records
  video/contact-sheet paths, final `success_at_end`, whether the dynamic event
  happened before success, and task-frame peg-head x/YZ errors. The inspector
  Markdown explicitly states that final dynamic manipulation success still
  requires both the episode condition and direct video/contact-sheet
  inspection. This preserves the evidence rule: if `95191` or `95209`
  produces a contact sheet, it must be opened directly before making any
  success or failure claim from the video.
- 2026-06-02 12:45 calibrated-C_pi video evidence chain: submitted gated video
  job `95433` with inspection `95437`. It depends on `afterok:95394`, so it
  only runs if the calibrated larger-label C_pi no-video controller chain
  passes the unchanged final dynamic success gate. The video job repeats the
  same seed `7400`, `hole_move_stop`, `job94539` world-model dirs,
  `HANDOFF_MODE=cpi`, `CPI_TARGET_FPR=0.02`,
  `BRIDGE_SERVO_MODE=task_frame_projected`, `BRIDGE_PHASE_MODE=search`,
  `BRIDGE_ORIENTATION_REFERENCE=peg_alignment`,
  `TASK_SERVO_INSERT_MANIFOLD_GUARD=true`, and
  `TASK_SERVO_INSERT_AXIS_STEP_M=0.004`, changing only
  `RUN_GROUP=cpi_calibrated_after94542_video` and `SAVE_VIDEO=true`.
  After submission, `95433`'s Slurm `ExcNodeList` was updated to the
  render-safe exclusion list including `server10` and `server58`, because the
  calibrated wrapper itself is also used for no-video jobs and only excludes
  drained nodes by default. This is rendering hygiene, not an evaluation
  change. If `95437` finds a contact sheet, it must be opened directly before
  any success claim.
- Cleanup for the same calibrated-C_pi branch was added as job `95442` with
  branches file
  `experiments/world_model_task_rebinding/rebinding_controller/cpi_calibrated_after94542_video_branches.txt`.
  It waits on `afterany:95394`: if gate `95394` passes, it leaves video job
  `95433` and inspection `95437` alone; if the gate fails, it cancels only
  that dead video branch. The cleanup script and wrapper passed syntax/compile
  checks.
- 2026-06-02 12:50 render-safe calibrated-Cpi video wrapper: added
  `scripts/slurm/evaluate_rebinding_controller_cpi_calibrated_video.sbatch`.
  It uses the same calibrated C_pi controller logic but declares the
  render-safe Slurm exclusion list at the `#SBATCH` layer and defaults
  `SAVE_VIDEO=true`. This prevents future calibrated-Cpi video jobs from
  relying on manual `scontrol update` to avoid nodes with earlier render failures. `bash -n`
  passed, and `sbatch --test-only` scheduled it on render-usable `server27`.
- 2026-06-02 14:15 RGB-D priority cleanup: canceled stale non-RGB-D video and
  inspection jobs `95191`, `95192`, `95209`, `95210`, `95215`, and `95216`
  while pending. They were old state/oracle learned-WM video or debug branches
  and no longer serve the current RGB-D method chain. This is queue/resource
  hygiene only; it does not change any existing recorded controller metric or
  evaluation gate.
- Video queue audit at `2026-06-02 10:54+08:00`: gated video jobs `95047`
  and `95107` remain pending with released dependencies and no artifacts, but
  Slurm now forecasts both for `2026-06-02T12:02:50`. Do not submit duplicate
  shorter video jobs while these have the earlier forecast; wait for the
  rendered artifacts and inspect the resulting contact sheets or frame grids
  directly before making any success claim.
- Video/artifact audit at `2026-06-02 11:20+08:00`: learned-WM move-stop
  video job `95047` failed during rendering on `server10` with Vulkan
  `ErrorDeviceLost`; it produced no metrics, H5, MP4, or contact sheet, and
  inspection `95048` found no evidence. Treat this as a rendering failure that
  requires rerun on a stable render node, not as controller evidence. Rerun
  `95191` with inspection `95192` was submitted with the same learned-WM
  move-stop video settings and `world_model_dirs_job94442.txt`; it only changes
  the output run group and adds `server10` to the render exclusion list.
  Peg-drop/regrasp video job `95107` completed, inspection `95108` found one
  video/contact sheet, and the contact sheet plus extracted last frame were
  opened directly. Visual evidence agrees with metrics:
  `success_at_end=True`, `first_success_step=234`, `regrasp_count=13`,
  final peg-head hole-frame `x=-0.013857m`, `yz=0.003281m`. This validates the
  CV peg-drop/regrasp branch only; it does not establish learned-WM advantage
  or moving-hole generality.
- Full-shard audit at `2026-06-02 11:16+08:00`: after larger-shard model
  `94539` passed strict inspection `94863`, no-video controller job `95055`
  ran using the four final `job94539/member_*` models. Inspection `95056`
  completed and success gate `95057` failed with exit `65`. The run had one
  dynamic trigger at step `90`, `234` bridge steps, no handoff/rebind/regrasp
  or retreat, world-model horizons `{'1': 4, '5': 230}`,
  `success_at_end=False`, final hole-frame peg-head `x=-0.10053m`, and
  `yz=0.00275m`. This localizes the controller gap to insertion-axis progress:
  lateral alignment is good, but the bridge stays around `10cm` short of final
  insertion. No video was produced because the no-video gate failed. Dead
  gated-video and failover branches were canceled as queue hygiene only.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_job94539_prediction_and_controller.md`.
- Seed `7400` insert-axis analysis at `2026-06-02 11:34+08:00`: comparing
  `95055`, `95139`, `95042`, and `95136` shows the failure is seed-linked and
  physical/controller-like, not specific to the larger `94539` world model.
  Both learned-WM runs fail at seed `7400`, while learned-WM and CV succeed at
  seed `7500`. In the failing seed, insert phase starts around
  `x=-0.145m`, lateral error is only a few millimeters, and the controller
  keeps commanding insertion-axis motion, but peg-head x stalls around
  `-0.101m`. The failed seed also has a much lower final hole height
  (`z≈0.0757m` versus `z≈0.1292m`). Debug video rerun `95209` with inspection
  `95210` was submitted to inspect the physical jam directly, preserving the
  `95055` controller settings and only enabling video plus excluding
  `server10`. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_seed7400_insert_axis_jam.md`.
- Submitted aligned insertion-manifold probe after the seed `7400` jam
  analysis: no-video `95212`, inspection `95213`, unchanged success gate
  `95214`, gated video `95215`, and video inspection `95216`. It keeps seed
  `7400`, `job94539` world-model dirs, `hole_move_stop`,
  `rebind_world_model`, and `HANDOFF_MODE=never`, while changing the physical
  insertion primitive to `BRIDGE_ORIENTATION_REFERENCE=peg_alignment`,
  `TASK_SERVO_INSERT_MANIFOLD_GUARD=true`, and
  `TASK_SERVO_INSERT_AXIS_STEP_M=0.004`. This is a falsifiable controller
  probe under the unchanged metric, not a success claim.
- The same `2026-06-02 06:44+08:00` audit confirms the full-shard no-video
  jobs `94830`/`94835`/`94840`/`94845` and video jobs
  `94833`/`94838`/`94843`/`94848` also export
  `BRIDGE_ORIENTATION_REFERENCE=tcp_continuation` and
  `BRIDGE_SERVO_REFERENCE=tcp_continuation`. The `94539` branch of that audit
  was later canceled before running because upstream inspection `94619` lacked
  the strict object-WM compliance gate; replacement jobs `94865`/`94868`
  preserve the same orientation-aware settings and remain gated by `afterok`
  on the corresponding success-gate jobs.
- Old no-video smokes `94519` and `94521` exposed a planner objective bug:
  `model_horizon_counts` was all `0`, so the bridge was effectively tracking
  the current frame. This is recorded as a scaffold failure, not method
  evidence. Same-seed fixed-score no-video smoke `94591` completed and was
  inspected by `94741`; it validated nonzero horizon selection but still
  failed insertion. Pending fixed-score v2 jobs `94582`, `94583`, and `94590`
  were canceled after the TCP-continuation bridge target replaced the old
  peg-head target.
- Orientation-aware TCP-pose bridge v1 same-seed jobs
  `94814`/`94819`/`94824` completed and failed the unchanged final dynamic
  insertion gate. The failure is now recorded in
  `docs/world_model_task_rebinding/2026-06-02_orientation_phase_search_failure.md`.
  Event logs show the planner mostly selected `align` (`242/244` CV bridge
  steps, `228/229` learned-WM bridge steps) and ended around pre-insert depth
  (`x=-0.10699` and `x=-0.11460`) instead of the official success threshold
  `x>=-0.015`. This is a phase-selection/planning failure, not an evaluation
  change and not evidence against world-model task rebinding.
- Phase-search v2 is implemented. `_choose_bridge` now scores `align` and
  `insert` phase candidates for each tau candidate, with continuous task
  progress and phase-entry costs. The inspector and summary now report
  `bridge_phase_counts`, `bridge_phase_mode_counts`,
  `phase_progress_shortfall_stats`, and `phase_entry_error_stats`.
  Lightweight static replay on the v1 failed terminal states chooses
  `insert, tau=4`, while an early large-y/z-error state still chooses `align`.
  Same-seed v2 Slurm chains are queued:
  `94981`/`94986`/`94991` for no-video rollout, inspections
  `94982`/`94987`/`94992`, success gates `94983`/`94988`/`94993`, gated
  videos `94984`/`94989`/`94994`, and video inspections
  `94985`/`94990`/`94995`. Summaries/cleanup are `94996`/`94997`/`94998`.
- Same-seed-ish v7 task-frame-projected no-video validations produced one
  learned-WM gate pass and one CV failure, but the paired controls prevent
  overclaiming. Initial jobs: CV seed `7400` `95039 -> 95040 -> 95041` failed;
  learned-WM seed `7500` `95042 -> 95043 -> 95044` passed. Paired controls:
  CV seed `7500` `95136 -> 95137 -> 95138` also passed, while learned-WM seed
  `7400` `95139 -> 95140 -> 95141` failed. Therefore this is evidence that
  the task-frame-projected bridge can solve some no-video move-stop cases, not
  evidence that the learned world model beats CV. Gated video `95047` and
  inspection `95048` are still required before any final dynamic manipulation
  claim.
- Regrasp v9 no-video validation `95104 -> 95105 -> 95106` passed the
  unchanged final dynamic success gate for `peg_drop` with regrasp enabled:
  `regrasp_count=13`, `bridge_count=150`, final peg-head hole-frame
  `x=-0.0138570`, `yz=0.0032815`, and dynamic event before success. This
  supports the peg-alignment/manifold-guard physical fix only at no-video
  metric level. Gated video `95107` and video inspection `95108` remain
  pending and must be opened directly if artifacts are produced.
- 2026-06-02 10:16 video queue audit: gated videos `95047` and `95107` both
  have dependencies released, keep `SAVE_VIDEO=true`, use conservative
  rendering exclusions, and are priority-pending with Slurm scheduling them on
  `server27` for `2026-06-02T11:12:03`. Artifact search found no
  `job95047` or `job95107` run directory yet, so no visual evidence exists.
- 2026-06-02 10:25 video queue audit: gated videos `95047` and `95107` still
  have `Dependency=(null)`, `SAVE_VIDEO=true`, conservative rendering
  exclusions, and one-GPU render allocations; Slurm currently schedules both
  on `server27` for `2026-06-02T11:37:05`. A direct artifact search found no
  `job95047` or `job95107` run directory, MP4, or contact sheet, so the
  no-video gate passes remain intermediate evidence only.
- 2026-06-02 10:45 video queue audit: gated videos `95047` and `95107` still
  have `Dependency=(null)`, but Slurm moved the forecast later to
  `2026-06-02T18:02:04` on `server58`. Direct artifact search still found no
  `job95047` or `job95107` run directory, MP4, or contact sheet. Backfill
  probes with the same evaluation settings and shorter 10-minute and 5-minute
  walltimes predicted `2026-06-03T13:46:05`, later than the existing jobs, so
  no duplicate video jobs were submitted. The no-video gate passes remain
  intermediate evidence until these videos run and are opened directly.
- Stale pending full-shard v1 controller branches were canceled before
  running to avoid evidence pollution. Replacement phase-search full-shard
  controller branches are queued behind the same strict object-WM inspections:
  `94999`-`95003` for `94539`, `95004`-`95008` for `94679`,
  `95009`-`95013` for `94746`, and `95014`-`95018` for `94809`.
  Summaries/cleanup are `95019`/`95020`/`95021`.
- Phase-search v2 same-seed jobs `94981`/`94986`/`94991` completed and failed
  final dynamic insertion. The no-video summary shows the moving-hole branches
  now select insertion phase (`221/244` CV bridge steps and `197/229`
  learned-WM bridge steps), but final x remains around `-0.11m`. State-log
  readout shows the TCP moved partway toward the insert target while the peg
  head did not advance into the hole. This is recorded in
  `docs/world_model_task_rebinding/2026-06-02_phase_search_v2_failure_and_hybrid.md`.
- Implemented `bridge_servo_reference=phase_hybrid`. It uses the measured TCP
  continuation pose for `align`, then uses peg-head task-frame position error
  for `insert` while keeping the TCP continuation orientation target. Static
  pure-function replay on v2 failed terminal states chooses `insert` with
  active servo reference `peg_head`. Hybrid v3 moving-hole chains are queued:
  CV `95026`-`95030`, learned-WM `95031`-`95035`, with summaries/cleanup
  `95036`/`95037`/`95038`.
- Hybrid v3 moving-hole no-video jobs `95026` and `95031` completed and failed
  final dynamic insertion. Inspections show phase switching and peg-head insert
  servo both happened, but direct 3D target chasing saturated root/world
  actions and allowed late align steps to retreat along the insertion axis.
  This falsifies direct-world phase-hybrid as sufficient, not the task-frame
  rebinding objective.
- Implemented `bridge_servo_mode=task_frame_projected`. The planner still
  searches future `tau` and phase, but the bridge now executes bounded local
  task-frame steps with one-step hole-motion feedforward. Insert phase advances
  the local hole axis by a small monotonic step while y/z are corrected with
  smaller bounded lateral steps; align no longer retreats along the insertion
  axis by default. Inspector/summary report servo mode, actual bridge-delta
  norms, and task-frame axis/lateral error and step stats. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_task_frame_projected_servo_v4.md`.
- Submitted task-frame projected v4 chains. No-video: CV
  `95039 -> 95040 -> 95041`, learned-WM `95042 -> 95043 -> 95044`, summary
  `95049`. Gated videos: CV `95045 -> 95046`, learned-WM `95047 -> 95048`,
  tolerant video summary `95050`. These are controller diagnostics; success
  still requires the unchanged final-state gate and direct video/contact-sheet
  inspection if videos run.
- Canceled stale pending full-shard phase-search chains `94999`-`95021` before
  they ran because they lacked `bridge_servo_mode=task_frame_projected`.
  Replacement full-shard v4 chains now wait on the same strict world-model
  inspections: `95055`-`95059` after `94863` for object-WM job `94539`,
  `95060`-`95064` after `94680` for `94679`, `95065`-`95069` after `94747`
  for `94746`, and `95070`-`95074` after `94810` for `94809`.
  Full-shard summaries/cleanup are `95075`/`95076`/`95077`.
- 2026-06-02 08:58 queue audit: same-seed v4 no-video jobs `95039` and
  `95042` are still priority-pending for `2026-06-02T10:00:12`. Their
  submit lines preserve `BRIDGE_SERVO_REFERENCE=phase_hybrid`,
  `BRIDGE_SERVO_MODE=task_frame_projected`,
  `BRIDGE_ORIENTATION_REFERENCE=tcp_continuation`,
  `BRIDGE_PHASE_MODE=search`, and `TASK_SERVO_ALLOW_AXIS_RETREAT=false`.
  Gated videos `95045`/`95047` depend on no-video gates `95041`/`95044`.
  Full-shard v4 submit lines for `95055`/`95060`/`95065`/`95070` preserve the
  same controller settings and wait only on strict object-WM inspections.
  Video branches inherit the live rendering exclusion policy from the
  Slurm wrapper.
- 2026-06-02 09:01 diagnostic-chain audit: `inspect_rebinding_controller_run.py`
  reports the fields needed to explain v4 behavior after a failed no-video
  run: `bridge_phase_counts`, `bridge_servo_mode_counts`,
  `active_servo_reference_counts`, bridge delta norms, task-frame axis/lateral
  errors and steps, final peg-head x, and final/min YZ after trigger. The
  success gate still requires final success after a prior dynamic event and
  does not alter the official final inserted-state metric. No new v4 artifact
  exists yet.
- 2026-06-02 09:03 evidence-chain hygiene: added same-seed v4 gated-video
  cleanup mapping
  `experiments/world_model_task_rebinding/rebinding_controller/controller_smoke_v7_task_frame_projected_video_branches.txt`
  and submitted cleanup job `95086` after no-video gates `95041`/`95044`.
  This mirrors the full-shard cleanup path: if a no-video gate fails, it
  cancels only the dead gated-video rollout/inspection branch; if a gate
  passes, it leaves the video branch intact for required direct inspection.
- 2026-06-02 09:05 peg-drop v4 extension: submitted
  `controller_regrasp_smoke_v8_task_frame_projected_novideo` with
  `SCENARIO=peg_drop`, `ALLOW_REGRASP=true`, seed `7300`,
  `BRIDGE_SERVO_REFERENCE=phase_hybrid`,
  `BRIDGE_SERVO_MODE=task_frame_projected`, `BRIDGE_PHASE_MODE=search`, and
  `TASK_SERVO_ALLOW_AXIS_RETREAT=false`. Chain:
  `95088 -> 95089 -> 95090`; gated video `95091 -> 95092`; cleanup `95093`;
  summaries `95094`/`95095`. This tests whether v4 task-frame lateral servo
  can recover from the prior align-only regrasp failure (`94991`, final
  `YZ≈0.062m`) without changing the regrasp primitive or success metric.
- 2026-06-02 09:30 peg-drop v8 result: no-video job `95088` completed and
  gate `95090` failed under the unchanged final inserted-state metric. It
  regained final grasp and improved lateral alignment (`min YZ=0.004605m`),
  but final insertion failed (`x=-0.109342`, `YZ=0.016672`). H5 diagnostics
  show peg-hole angle drifting from `0.07494rad` near the entry manifold to
  `0.34954rad` and then `0.93784rad` while x stayed around `-0.11m`. This is
  a physical insertion-manifold failure after regrasp, not a reason to change
  metrics or fall back to restoration.
- 2026-06-02 09:30 implemented and submitted v9 peg-alignment insertion
  manifold control. New explicit options:
  `BRIDGE_ORIENTATION_REFERENCE=peg_alignment` and
  `TASK_SERVO_INSERT_MANIFOLD_GUARD=true`. The guard blocks insertion-axis
  advance when peg-head lateral error or peg-hole angle leaves the static
  successful insertion manifold, while the rotation action tracks peg pose
  rather than only TCP static continuation orientation. Static checks passed
  and offline `_choose_bridge` replay on `95088` states showed axis advance
  only inside the manifold. Submitted chain: no-video
  `95104 -> 95105 -> 95106`, gated video `95107 -> 95108`, cleanup `95109`,
  summaries `95110`/`95111`. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_regrasp_v8_failure_v9_peg_alignment.md`.
- 2026-06-02 10:01 task-frame-projected no-video result: v7 learned-WM job
  `95042` passed CPU inspection `95043` and success gate `95044`; it reports
  `success_at_end=True`, `dynamic_event_before_success=True`, trigger step
  `90`, first success step `172`, final peg-head hole-frame
  `x=-0.0129687`, `YZ=0.0029474`, and prediction source `world_model`. CV job
  `95039` failed final dynamic success under its own seed and gate `95041`
  failed as expected. This is promising intermediate evidence only, because
  `95042` has no video/contact sheet yet and the CV/learned-WM no-video seeds
  differ (`7400` versus `7500`). Gated video job `95047` is pending after the
  passed gate; it must be inspected directly before any final success claim.
  Paired no-video controls were queued to check seed effects: CV@7500
  `95136 -> 95137 -> 95138` and learned-WM@7400
  `95139 -> 95140 -> 95141`. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_task_frame_projected_no_video_gate_pass.md`.

## Skill Primitives

- [x] Hold pose.
- [x] Move TCP to safe pose. Initial action-level primitive is implemented as
      `safe_retreat`; runtime validation is pending.
- [x] Servo to pre-insert frame.
- [x] Align peg head in hole frame.
- [x] Insert in moving task frame.
- [x] Retreat. Initial action-level primitive retreats TCP upward while keeping
      the gripper closed when the bridge target is infeasible.
- [x] Regrasp peg. Initial action-level primitive approaches the peg tail,
      descends, closes, and lifts; it is not success evidence until Slurm smoke
      and video inspection pass.
- [x] Report infeasible.

## Evaluation

- [x] Compare frozen DP only. Smoke `94522` completed and failed, as expected
      under the dynamic perturbation.
- [x] Compare simple target tracking. Smoke `94521` completed and failed; it
      is a baseline, not method evidence.
- [ ] Compare bridge without `C_pi`.
- [ ] Compare learned-world-model bridge against CV bridge using the same
      dynamic event and final inserted-state metric.
- [ ] Compare full controller.
- [x] Save event logs for rebinds, handoffs, and infeasible decisions.
- [x] Inspector reports evidence gates, regrasp/retreat counts, infeasible
      reasons, and `C_pi` mean/std stats. It still does not replace required
      video inspection.
- [x] Inspector now reports grasp-state timing (`ever_grasped`,
      `first_grasp_step`, `last_grasp_step`, `grasp_fraction`) so controller
      failures can be separated into "base DP never picked up peg" versus
      "bridge failed after grasp".
- [x] Inspector now reports task-frame error after the dynamic trigger
      (`final_peg_head_x`, final/min/mean peg-head YZ, and x range after
      trigger), so bridge behavior can be diagnosed without changing the
      final inserted-state metric.
- [x] Inspector now reports orientation-bridge stats from event logs, so
      pose-servo failures can be separated into "rotation command never
      applied" versus "rotation applied but physical insertion still failed".
- [ ] Save and inspect short videos or replay frames for representative cases.
      Learned-WM video smoke `94564` produced a contact sheet and was inspected
      directly; it failed and is recorded in
      `docs/world_model_task_rebinding/2026-06-02_learned_wm_controller_94564.md`.
      Learned-WM video smokes `94576` and `94577` also produced contact sheets
      and were inspected directly; both failed with peg-loss/safe-retreat
      behavior and are recorded in the same controller evidence family.
      2026-06-02 12:55 queue audit: the currently queued video/debug jobs have
      not produced new artifacts yet. Jobs `95212`, `95191`, and `95209` are
      priority-pending with forecasts around `2026-06-02T14:04:37` to
      `14:06:38`; calibrated `C_pi` video `95433` is still dependency-pending
      behind larger-label `C_pi` inspection/calibration and the no-video
      success gate. There is therefore no new video result or visually
      inspectable contact sheet at this time.
      2026-06-02 13:08 video evidence update: peg-drop/regrasp V9 video
      `95107` and inspection `95108` completed with final dynamic success and
      one contact sheet. The contact sheet and extracted last frame were
      opened directly; they are nonblank and consistent with peg regrasp,
      transport to the box, and final insertion. This is visual evidence for
      the `peg_drop` / `rebind_cv` / `peg_alignment` regrasp branch only, not
      a learned-world-model success claim. Learned-WM task-frame-projected
      video job `95047` produced no metrics, H5, video, or contact sheet
      because rendering failed immediately on `server10` with Vulkan
      `ErrorDeviceLost`; treat it as a render-node failure, not as policy
      evidence. Render-safe rerun `95191` was later canceled in the 14:15
      RGB-D priority cleanup because it is non-RGB-D state/oracle video
      evidence and no longer serves the current method chain. Failure note:
      `docs/world_model_task_rebinding/2026-06-02_task_frame_projected_video_device_lost.md`.

## Regrasp Smoke

- Job `94550` is submitted with `SCENARIO=peg_drop`,
  `CONTROL_POLICY=rebind_cv`, `HANDOFF_MODE=never`, `ALLOW_REGRASP=true`, and
  `SAVE_VIDEO=false`. It completed without runtime errors but failed: the peg
  was not grasped at the end, and the event log showed `212` regrasp actions
  cycling between `regrasp_approach` and `regrasp_descend` with no
  `regrasp_close_lift`. This is primitive-debug evidence, not dynamic-task
  success evidence.
- Failure note:
  `docs/world_model_task_rebinding/2026-06-02_regrasp_smoke_94550.md`.
- Fixed the stateless regrasp oscillation by allowing close/lift at the
  physically reachable TCP clearance once XY is aligned. Submitted same-seed
  no-video v2 smoke `94621`, then canceled it before running after offline
  replay showed `0.05m` clearance was too tight. Replacement v2 smoke `94622`
  uses `--regrasp-tcp-clearance-m 0.065`.
- Short-allocation v2 smoke `94626` completed and showed the next failure:
  `regrasp_close_lift` occurred, but final grasp stayed false. The gripper
  sign was correct; close and lift were coupled too early. The primitive now
  close-holds in place once XY/clearance is acceptable, and only normal
  `grasped=True` state exits the regrasp branch.
- Short-allocation v3 smoke `94633` completed after the close-hold change and
  still failed: the gripper closed and held but did not reacquire the peg.
  Slot analysis showed the TCP was closing about `5cm` above the peg, while
  the original successful grasp used peg-local TCP pose approximately
  `[-0.06295, -0.00005, 0.00068]`.
- Regrasp now targets that full local 3D grasp pose and closes only when the
  TCP is physically near it. Stale v3 video job `94634` was canceled before
  running. V4 no-video smoke `94645` completed with
  `RUN_GROUP=controller_regrasp_smoke_v4_empiricalgrasp_short`, and CPU
  inspection `94656` passed. It still failed: `regrasp_close` occurred at
  steps `103-104` but did not establish a stable grasp; later approach drifted
  away. This is primitive-debug evidence and is recorded in
  `docs/world_model_task_rebinding/2026-06-02_controller_followups_94591_94698_94645.md`.
- Same-seed v5 no-video regrasp validation `94773` completed with
  `bridge_servo_reference=tcp_continuation`, `SCENARIO=peg_drop`,
  `ALLOW_REGRASP=true`, and seed `7300`. It regained final grasp and logged
  regrasp actions, but still failed final dynamic insertion with final
  peg-head YZ about `0.156m`. Inspection `94774` passed as a failed-run
  inspection, gate `94775` failed by design, and dead gated video jobs
  `94776`/`94777` were canceled.
- Same-seed v8 task-frame-projected validation `95088` completed and failed
  final dynamic insertion. It answered the post-regrasp alignment question:
  bounded task-frame servo can recover lateral alignment after regrasp, but it
  does not preserve the peg insertion manifold during axis pushing. V9
  `95104` is queued with explicit peg-alignment rotation and insertion
  manifold guarding; this keeps the same peg-drop seed and final-state success
  gate.
- V9 peg-alignment regrasp chain completed. No-video `95104`, inspection
  `95105`, and final-state gate `95106` passed. Gated video `95107` and video
  inspection `95108` also passed, with `success_at_end=true`,
  `dynamic_event_before_success=true`, first success step `234`,
  `regrasp_count=13`, final peg-head hole-frame `x=-0.013857m`,
  `YZ=0.003281m`, one MP4, and one contact sheet. The contact sheet and last
  frame were opened directly at `2026-06-02 13:08+08:00`. This records a real
  peg-drop/regrasp controller success with visual evidence, while keeping the
  boundary clear: it uses oracle state slots and CV prediction, not learned
  world-model prediction or `C_pi` handoff.

Completion standard: full controller improves over simple tracking on
move-stop and continuously moving hole pilots, with interpretable rebind logs.

## 2026-06-03 Focused State Follow-Up

- [x] Added `peg_disturb_wm_regrasp` to
      `scripts/slurm/evaluate_state_rebinding_smoke_matrix.sbatch` as an
      exact selectable case via `STATE_MATRIX_CASES`. This keeps the same
      peg-disturb seed `7800`, oracle slot source, `phase_hybrid +
      task_frame_projected + peg_alignment + insert_manifold_guard`, no-video
      setting, final inserted-state gate, and regrasp allowance. The physical
      question is whether the peg-disturb failure is specific to CV/current
      frame rebinding or remains even when the state world model predicts the
      future task frame.
- [x] Submitted and completed state/oracle scaffold job `99714` for only
      `peg_disturb_wm_regrasp` under
      `experiments/world_model_task_rebinding/rebinding_controller/state_smoke_peg_disturb_wm_regrasp_20260603_0115`.
      It started immediately on `server42` and completed in `00:01:46`.
      Result: failed final dynamic gate (`gate_status=65`),
      `success_at_end=false`, `first_success_step=-1`, final hole-frame
      `x=-0.1131336`, `YZ=0.0037868`, `bridge_count=227`,
      `regrasp_count=0`, and `slot_source=oracle`. This clarifies that the
      peg-disturb gap is not specific to CV/current-frame rebinding; even with
      learned state-WM future prediction the controller stays laterally close
      but does not close insertion progress after the peg disturbance. This is
      state/oracle scaffold evidence only, not RGB-D method evidence.
