# RGB-D Pipeline TODO

## Hard Boundary

- [ ] Current cluster policy: no standing node exclusion list and no hard-coded
      default node exclusion in wrapper source. Past job/node observations are
      diagnostic records only. New submissions must use live `sinfo`/`scontrol`,
      `sbatch --test-only`, and targeted canaries; any job-local exclusion must
      be explicit in that job's manifest and tied to current evidence.
- [ ] Treat RGB-D world-model training as the primary method path, not an
      optional visual add-on. Current state/oracle-slot results are debugging
      scaffolds, diagnostic upper bounds, or failure-localization tools only.
      They must not be described as method success.
- [ ] Generate full synchronized RGB-D data before making further method
      claims: RGB, depth, state/proprio, actions, env states, camera
      parameters, and object/task labels.
- [ ] Train a RGB-D-derived representation for the world model. The minimum
      acceptable path is RGB-D -> object/task slots/confidence -> object-centric
      dynamics/continuability/rebinding. A stronger path may add latent RGB-D
      world-model prediction, but the controller must not depend on oracle
      slots for final evidence.
- [ ] Evaluate task-frame rebinding with RGB-D-derived slots/latents. Oracle
      state evaluation is an upper bound and debugging reference only.

## Current RGB-D Data Status

- [ ] Latest user override at `2026-06-02T21:55+08:00`: RGB-D rendering no
      longer waits for 2-GPU/4-GPU blocks. One-GPU disjoint RGB-D render shards
      are now allowed, and the previous 8-node / 64-GPU generation cap is
      superseded for RGB-D data generation only. This does not relax exact96
      inspection, visual gates, RGB-D-derived evidence, 4H200/3h training
      floors, controller metrics, or video inspection. One-GPU jobs are now
      allowed only as small rolling batches: estimate runtime from completed
      shards, submit a small batch, inspect artifacts, then scale.
- [x] Cancel the erroneous full96 1GPU96 fanout. Jobs `98929-99024` and
      dependent jobs `99027-99040` were canceled at
      `2026-06-02T22:07+08:00` before start; `sacct` shows zero allocated
      nodes and no GPU runtime. The branch root has zero RGB-D H5, preview
      videos, contact sheets, and Slurm logs. This was a scheduling error, not
      method evidence. It must not be treated as the current fastest path.
      Added a hard wrapper guard so future submissions reject more than 8
      render jobs by default and reject more than 4 one-GPU render jobs by
      default. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_full96_1gpu96_render_branch_2155.md`.
- [ ] Current earliest RGB-D render path is the existing 2-GPU aggregate path,
      not new one-GPU jobs. As of `2026-06-02T22:11+08:00`, jobs `98524` and
      `98525` are forecast for `2026-06-03T01:00:00` and `01:39:57`;
      `98526`, `98527`, and late-half jobs `98841-98844` forecast
      `2026-06-03T06:01:30`. Fresh 1x2 probes forecast `2026-06-04T00:54:08`
      and 1x4 probes forecast `2026-06-04T03:45:15`, so no later duplicate was
      submitted. Existing successful 2-GPU render shards `98209` and `98210`
      finished 12 RGB-D trajectories each in `4m34s` and `5m32s`, so once a
      2-GPU shard starts the expected runtime is minutes, not hours.
- [x] 2026-06-02 22:35 preflight while waiting instead of queue-spinning:
      `AGENTS.md` now records the approximately 30-minute queue/artifact
      check cadence and the CPU-only SAPIEN/Vulkan rollout boundary from
      failed job `99136`. Static preflight passed for the active aggregate,
      exact96 structural/visual gates, RGB-D slot training/export,
      RGB-D-derived world-model training/inspection/eval, RGB-D controller,
      and video-review wrappers. Current method-chain manifest
      `full96_aggregate_20260602_205201_visual98847/jobs.tsv` still requires
      `98907` visual gate before `98848`, exact96 exports/inspections,
      `REQUIRE_RGBD_DERIVED=true` for world-model inspection/eval, and
      `EXPECTED_SLOT_SOURCE=rgbd` plus video/nonblank review for controller
      evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_chain_preflight_and_agent_rules_2235.md`.
- [x] 2026-06-02 22:40 replace only the pending RGB-D controller branch with
      a phase-hybrid bridge config before it can run. Old dependency-pending
      jobs `98917/98918/98919` were canceled with zero allocation; replacement
      jobs are `99170/99171/99172` after `98916`, with
      `SLOT_SOURCE=rgbd`, `BRIDGE_SERVO_REFERENCE=phase_hybrid`,
      `BRIDGE_SERVO_MODE=task_frame_projected`,
      `BRIDGE_ORIENTATION_REFERENCE=tcp_continuation`,
      `TASK_SERVO_INSERT_MANIFOLD_GUARD=false`, `SAVE_VIDEO=true`,
      `EXPECTED_SLOT_SOURCE=rgbd`, and nonblank video review. No render,
      RGB-D data, slot, predicted-slot, world-model training, inspection, or
      evaluation job was changed. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_state_axis_progress_and_rgbd_controller_config_2240.md`.
- [x] 2026-06-02 22:48 resource probe, no passive waiting:
      visible `mix` nodes still do not imply immediate allocation for account
      `mayi`. Fresh probes show `cpu` 1/2 GPU forecasts
      `2026-06-04T00:54:08`, `cpu` 4 GPU/H200 forecasts
      `2026-06-04T02:55:53`, `gpu` forecasts `2026-06-06`, `debug` 1 GPU is
      later and 2/4 GPU is blocked by `MaxGRESPerAccount`, `gpux` is inactive
      or drained, and `gaosh`/`engram`/`test` reject the current
      account/partition combination. Do not pollute the RGB-D queue with later
      duplicate renders just to look active; continue half-hour
      queue/artifact checks and use the interval for aligned preflight,
      artifact inspection, and state-only scaffold smoke that is explicitly
      recorded as non-RGB-D evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_resource_probe_and_state_smoke_submission_2248.md`.
- [ ] 2026-06-02 23:16 repaired full96 RGB-D path:
      front-half retry branch `98524-98527` completed on `server42`; exact
      partial48 structural gate `98528` and visual gate `98530` both passed
      with zero warnings, and representative contact sheets were opened
      directly. Late-half jobs `98841`, `98843`, and `98844` completed, but
      `98842` on `server10` hit repeated Vulkan `ErrorDeviceLost` for
      shard5/traj5 task0 units and was canceled. The late root currently has
      `42` RGB-D H5/contact sheets, plus five recorded failed units and one
      canceled/missing unit. Submitted exact six-unit repair job `99208` with
      job-local `ExcNodeList=server10`, replacement aggregate `99209`,
      exact96 structural inspection `99210`, repaired visual gate `99211`, and
      downstream repaired RGB-D method chain `99212-99222` under
      `experiments/world_model_task_rebinding/rgbd_method_chains/full96_aggregate_repair98842_20260602_2312_visual99211`.
      The old dead branch `98909/98910/98907`, `98848/98849`,
      `98911-98916`, and `99170-99172` was canceled. Current blocker:
      `99208` is priority-pending with forecast
      `2026-06-03T04:00:00`; no exact96 RGB-D method evidence exists until
      `99208 -> 99209 -> 99210 -> 99211` passes.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_front48_repair_and_state_smoke_2316.md`.
- [x] 2026-06-02 23:30 resource probe without duplicate pollution:
      fresh `sbatch --test-only` probes found no earlier legal replacement
      for repair job `99208`: `cpu` 1/2 GPU and 4H200 jobs forecast
      `2026-06-04T00:23`, `gpu` jobs forecast `2026-06-05T06:09`,
      `debug` 1 GPU is later and larger debug jobs are blocked by
      `MaxGRESPerAccount`, `gpux` is inactive/drained, and
      `test`/`gaosh`/`engram` reject the current account/partition
      combination. No duplicate RGB-D repair/render job was submitted. While
      waiting on the half-hour cadence, submitted only a single fixed-path
      1-GPU state/oracle smoke matrix job `99308`; it was later canceled
      before allocation and replaced by low-priority dependency job `99331`
      after the new RGB-D visual gate, so scaffold work cannot preempt the
      core data blocker. It is not RGB-D evidence and does not relax
      exact96/RGB-D-derived gates. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [ ] 2026-06-02 23:36 remaining-three RGB-D repair path:
      `99208` failed on `server28` with Vulkan `ErrorDeviceLost` for three
      units but produced three valid repair RGB-D files. Added
      `scripts/slurm/repair_rgbd_late_shard5_traj5_remaining3_after99208_20260602_2345.sbatch`
      and submitted `99316`, a 1GPU serial repair of only the remaining
      `hole_constant`, `hole_reverse`, and `peg_disturb` traj5 units, with
      job-local `ExcNodeList=server10,server28` tied to current DeviceLost
      evidence. Canceled dead dependency branch `99209-99219` and
      `99252-99254`. A later aggregate preflight found and fixed a duplicate
      basename implementation bug, so the superseded pending chain
      `99317-99331` was canceled before allocation. Current exact96 path is
      `99316 -> 99332 -> 99333 -> 99334`; current RGB-D method chain is
      `99335-99345` under
      `experiments/world_model_task_rebinding/rgbd_method_chains/full96_aggregate_repair99208_remaining3_20260602_2345_aggfix_visual99334`.
      No RGB-D method evidence exists until `99316 -> 99334` passes and the
      RGB-D-derived training/eval/video gates complete. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-02 23:41 aggregate duplicate-basename bug fixed before run:
      existing source roots contain `93` RGB-D files but repeated basenames
      across per-trajectory renders, so the old aggregate destination naming
      would have failed. Patched `aggregate_rgbd_roots.sbatch` to write
      `trajectory_unit_parent__rgbd_basename` and still reject duplicate
      trajectory units. Preflight confirms no duplicate units/destination
      names and exactly three missing units, all assigned to `99316`.
      The three successful `99208` repair files passed local structure
      inspection (`3` files, `903` frames, `0` warnings) and contact sheets
      were opened directly. This is data-quality/implementation evidence only,
      not method evidence.
- [x] 2026-06-02 23:45 remove repair-wrapper source exclusions:
      the remaining-three and old six-unit repair wrapper sources no longer
      contain hard-coded `#SBATCH --exclude` lines. They record
      `EXCLUDED_NODES` in manifests instead, so future job-local exclusions
      must be passed at submission time with live evidence. Active `99316`
      was kept because canceling and resubmitting with command-line
      `--exclude=server10,server28` would move the forecast from
      `2026-06-03T06:01:30` to `2026-06-04T00:41:50`. Keeping the active job
      preserves the faster RGB-D repair path while source hygiene is fixed.
- [x] 2026-06-02 23:46 active chain snapshot audit:
      `scontrol` and chain manifest confirm `99316` is the only active repair
      blocker; `99332/99333/99334` are dependency-pending exact96 aggregate,
      structural, and visual gates; `99335` and `99340` are 4H200/3.5h
      training jobs with exact96 and `MIN_TRAIN_SECONDS=10800`; `99341/99342`
      require RGB-D-derived evidence; `99343-99345` require `SLOT_SOURCE=rgbd`
      controller video, expected-slot inspection, and nonblank video review.
      A `/tmp` aggregate unit test confirmed fixed destination naming accepts
      same basenames from different trajectory units and rejects duplicate
      units. This is preflight evidence only, not method evidence.
- [x] 2026-06-02 23:50 future aggregate stale-output guard:
      aggregate scripts now clean stale `files/*.rgbd.h5` after input count
      and duplicate-unit checks pass, and reject output roots inside source
      roots to prevent recursive self-inclusion. `/tmp` tests passed. Current
      active `99332` was kept because its aggregate root is absent, so stale
      cleanup is not needed for this run.
- [x] 2026-06-02 23:56 live GPU probe while preserving RGB-D priority:
      current-account probes show no earlier legal duplicate RGB-D or training
      allocation than active repair `99316`, now forecast at
      `2026-06-03T01:05:45`. New `cpu` 1/2/4GPU probes forecast around
      `2026-06-04T00:52`, `gpu` around `2026-06-05T14:33`, `debug` 1GPU
      around `2026-06-04T07:03`, `debug` 2/4GPU is blocked by
      `MaxGRESPerAccount`, `gpux`/`mgpu` are inactive or drained, and
      `gaosh`/`engram`/`test` reject the available account. Short 1GPU probes
      from 10 to 60 minutes and a 6CPU/32G shape did not reveal an earlier
      backfill slot. No duplicate RGB-D render was submitted. State scaffold
      job `99346` was made dependency-free and normal-priority so state
      physical smoke can proceed when Slurm can place it, without claiming
      RGB-D method evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-02 23:59 submitted snapshot audit:
      `scontrol write batch_script` and `sacct SubmitLine` confirm active
      submitted jobs preserve the exact96 aggregate/inspection gates,
      4H200/3h RGB-D slot training, exact96 predicted-slot export, 4H200/3h
      RGB-D-derived world-model training, `REQUIRE_RGBD_DERIVED=true`
      inspection/eval, `SLOT_SOURCE=rgbd` controller, and required video plus
      nonblank artifact review. `99316` submitted snapshot repairs only the
      three remaining traj5 units. No new RGB-D repair artifact exists yet.
- [x] 2026-06-03 00:02 RGB-D method contract static preflight:
      active RGB-D chain wrappers pass `bash -n`; RGB-D dataset/slot/export/
      world-model/controller/video Python entrypoints pass `py_compile`.
      Static contract check confirms exported `slots/*` are RGB-D-derived
      predictions, `oracle_slots/*` are inspection-only labels,
      world-model input uses `slots` with `oracle_slots_read=false`, the
      RGB-D-derived world-model wrapper validates predicted-slot uncertainty
      and probability datasets before training, and downstream
      inspection/eval/controller gates require RGB-D-derived evidence.
- [x] 2026-06-03 00:07 live duplicate-render/repair probe:
      `99316` is still the earliest aligned remaining-three repair path.
      Fresh `sbatch --test-only` probes for smaller 1GPU 6CPU/32G jobs found
      only later starts (`cpu` `2026-06-03T22:58:13`, `debug`
      `2026-06-04T07:03:05`, `gpu` `2026-06-05T14:43:13`), while
      `gpux`/`mgpu` remain unavailable and `gaosh`/`engram`/`test` reject the
      account. No duplicate repair/render was submitted because it would not
      produce exact96 RGB-D data earlier.
- [x] 2026-06-03 00:10 source-set preflight:
      existing source roots contain `93` RGB-D H5 files (`48 + 42 + 3`) with
      no duplicate trajectory units. The exact remaining gap is still the
      three traj5 units assigned to `99316`; the aggregate output root is
      absent and has no stale files. This keeps the next gate concrete:
      `99316` must add those three files before `99332 -> 99333 -> 99334`
      can produce exact96 inspected RGB-D data.
- [x] 2026-06-03 00:14 live resource probe without duplicate pollution:
      `sacctmgr` shows the user association is only account `mayi`. Partition
      checks show `gaosh`, `engram`, and `test` allow only account `null`,
      and `sbatch --test-only` rejects all tested account variants on those
      partitions. `gpux` and `mgpu` are unavailable because the partitions are
      drained/inactive. Fresh ten-minute probes for new `cpu` 1/2/4GPU jobs
      start at `2026-06-03T23:04:36`; `debug` 1GPU starts at
      `2026-06-04T07:03:05`; `gpu` starts at `2026-06-05T13:58:36`. Active
      remaining-three RGB-D repair `99316` is still the earliest aligned path
      with forecast `2026-06-03T01:05:45`, and there are still no repair logs
      or artifacts. No duplicate repair/render job was submitted.
- [x] 2026-06-03 00:20 post-repair readiness preflight:
      without queue-spinning, checked the downstream path that will run once
      `99316` produces the three remaining RGB-D files. Bash syntax passed for
      the repair, aggregate, exact96 structural/visual gates, RGB-D slot
      training/export, RGB-D-derived world-model training, RGB-D controller,
      and video review wrappers. Python compile passed for the RGB-D dataset
      inspector, visual artifact inspector, slot extractor/export, predicted
      slot inspection, object slot dataset, controller evaluator/inspector, and
      video artifact inspector. The active method-chain manifest still
      preserves exact96 file counts, RGB-D-derived boundaries, 4H200/3h
      training floors, `SLOT_SOURCE=rgbd`, and required nonblank video review.
      This does not prove any RGB-D result; it only prevents a known avoidable
      handoff failure when repair artifacts appear.
- [x] 2026-06-03 00:24 remaining-three repair input H5 preflight:
      read-only H5 checks confirm the `99316` worklist paths and trajectory
      indices are valid before the render job starts. The selected `traj_5`
      entries exist for `hole_constant_seed3000_n16`,
      `hole_reverse_seed4000_n16`, and `peg_disturb_seed5000_n16`; each has
      `actions` shape `(300, 7)`, `obs_stack` shape `(301, 2, 43)`,
      `env_states`, `slots/{hole_pose,peg_pose,tcp_pose,grasped,inserted}`,
      and `perturb/{trigger_step,hole_delta_cumulative,peg_delta_applied}`.
      This is input-availability preflight only; exact96 RGB-D evidence still
      waits for `99316 -> 99332 -> 99333 -> 99334`.
- [x] 2026-06-03 00:27 RGB-D visual-review tooling smoke:
      ran `inspect_rgbd_visual_artifacts.py` on the completed front48 RGB-D
      root with `expected_files=48`, `sample_files=1`, and `frames_per_file=2`.
      It produced
      `experiments/world_model_task_rebinding/rgbd_visual_tooling_preflight/20260603_002611/rgbd_visual_review_sheet.png`
      with `valid_visual_artifacts=true` and `num_warnings=0`. The sheet was
      opened directly and is nonblank for both RGB and depth views. This
      confirms the artifact review path can generate inspectable sheets before
      exact96 visual gate `99334` runs; it does not add RGB-D method evidence.
- [x] 2026-06-03 00:29 video artifact-review tooling smoke:
      ran `inspect_video_artifacts.py` on existing scaffold video job `95107`
      and wrote
      `experiments/world_model_task_rebinding/video_artifact_tooling_preflight/20260603_002814/peg_drop_seed7300_review_sheet.png`.
      The tool reports `valid_video_artifacts=true`, `num_videos=1`,
      `num_readable_videos=1`, and `num_nonblank_basic_videos=1`; the sheet
      was opened directly and is inspectable. This confirms the future RGB-D
      controller video artifact gate can generate readable review sheets, but
      it is not RGB-D method evidence because the source run is an oracle-slot
      scaffold video.
- [x] 2026-06-03 00:31 existing source companion-artifact preflight:
      filename-only scan of the current source roots confirms `93` existing
      RGB-D H5 files, `93` unique trajectory units, `0` duplicate units, and
      `0` missing companion-artifact rows. Each existing RGB-D unit has at
      least one contact sheet, MP4 preview, and JSON artifact in the same
      unit directory. Counts are still front48 `48`, latehalf `42`, and
      successful `99208` repair `3`; exact96 evidence still waits for `99316`
      to add the remaining three files and for `99332 -> 99333 -> 99334`.
- [x] 2026-06-03 00:34 incomplete aggregate refusal gate:
      ran `scripts/slurm/aggregate_rgbd_roots.sbatch` locally against the
      current `93` source RGB-D files with `EXPECTED_RGBD_FILES=96`. The
      wrapper refused the aggregate with exit code `1` and message
      `Found 93 RGB-D files across source roots; expected exactly 96`, leaving
      `0` RGB-D files in the aggregate output directory. Artifact root:
      `experiments/world_model_task_rebinding/rgbd_aggregate_gate_preflight/incomplete93_refusal_20260603_003417`.
      First-principles reason: the RGB-D slot extractor and RGB-D-derived
      world model must train only on the complete dynamic trajectory set, so
      missing repair units must block aggregation rather than becoming silent
      data skew. This is a data-gate smoke only; it does not produce RGB-D
      method evidence.
- [x] 2026-06-03 00:45 remaining-three repair `99316` failed as rendering:
      job `99316` ran on `server21` and wrote no RGB-D H5 files. Its
      `failed_units.tsv` contains all three remaining units
      (`hole_constant_seed3000_n16_traj_5`,
      `hole_reverse_seed4000_n16_traj_5`,
      `peg_disturb_seed5000_n16_traj_5`) with repeated SAPIEN/Vulkan
      `ErrorDeviceLost`. This is a render-node failure and does not falsify
      RGB-D world-model task rebinding. Downstream branch `99332-99345` was
      canceled after `99332` became dependency-never-satisfied.
- [ ] 2026-06-03 00:45 replacement remaining-three RGB-D path:
      submitted new repair job `99590` under
      `experiments/world_model_task_rebinding/rgbd_dynamic_distributed/from_rollout_dir/repair_remaining3_after99316_server21_20260603_0045`
      using the same three trajectory units and a job-local exclusion
      `server10,server28,server21` tied to current DeviceLost evidence. The
      replacement exact96 aggregate is `99591`, structural inspection `99592`,
      visual gate `99593`, and RGB-D method chain `99594-99604` under
      `experiments/world_model_task_rebinding/rgbd_method_chains/full96_aggregate_repair99590_20260603_0045_visual99593`.
      The aggregate root is
      `experiments/world_model_task_rebinding/rgbd_dynamic_distributed/from_rollout_dir/full96_aggregate_p48retry98524_98527_latehalf_repair99208_repair99590_20260603_0045`
      and must pass exact `96` RGB-D files before any RGB-D method claim.
- [ ] 2026-06-03 00:55 failed-only sensors-mode repair branch:
      while `99590` was still running on `server55`, its first unit had
      already failed with DeviceLost and no RGB-D H5. Because the failure
      occurs during initial camera capture before trajectory replay, added
      `scripts/slurm/repair_rgbd_failed_units_after99590_sensors_20260603_0055.sbatch`.
      It waits for `99590`, reads only `failed_units.tsv`, renders those units
      with `RENDER_MODE=sensors`, and leaves any successful `99590` units
      intact. Canceled the `99591-99604` branch that depended on `99590`
      afterok, then submitted sensors-mode replacement `99611 -> 99612 ->
      99613 -> 99614` plus RGB-D method chain `99615-99625`. This preserves
      RGB-D synchronized observations and exact96 gates; it is a rendering
      repair, not a method or evaluation change.
- [ ] 2026-06-03 01:00 sensors repair running:
      `99590` completed failed with `0/3` RGB-D files and all three failed
      units on `server55`. Reduced `99611` walltime from `30` to `20` minutes
      to fit backfill without changing the worklist or gates; it immediately
      started on `server58` at `2026-06-03T00:59:54`.
- [ ] 2026-06-03 01:03 sensors repair first failure:
      `99611` recorded a first DeviceLost failure on `server58` even with
      `RENDER_MODE=sensors`, so the failure is not only the human-render path.
      The job is still running; wait for the final failed/success unit list
      before submitting the next repair. Dry-runs for known-success single
      nodes and current-failure-node exclusions are later, so do not spam
      duplicate jobs before `99611` closes.
- [x] 2026-06-02 23:21 replace only the repaired-chain RGB-D controller tail:
      old pending tail `99220/99221/99222` still used
      `tcp_continuation + no insert guard`, which is weaker than the newly
      supported state-side insertion-manifold config. Those jobs were canceled
      before allocation and replaced with `99252/99253/99254` after the same
      `99219` dependency. Replacement config uses `SLOT_SOURCE=rgbd`,
      `CONTROL_POLICY=rebind_world_model`, `BRIDGE_SERVO_REFERENCE=phase_hybrid`,
      `BRIDGE_SERVO_MODE=task_frame_projected`,
      `BRIDGE_ORIENTATION_REFERENCE=peg_alignment`, and
      `TASK_SERVO_INSERT_MANIFOLD_GUARD=true`. This is a controller physics
      fix before execution, not an evaluation change or method claim.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_controller_tail_pegalign_replacement_2321.md`.
- [ ] Partial48 RGB-D generation has started producing real synchronized
      RGB-D data. Shard0 job `98209` completed on `server20` with `12` RGB-D
      H5 files, `12` contact sheets, `12` preview videos, and `0` failed work
      units. Local structure inspection found `12` files, `12` trajectories,
      `3612` frames, and `0` warnings; local visual inspection found `0`
      warnings and `valid_visual_artifacts=True`. Manual review of the contact
      grid showed nonblank base/hand RGB and depth with visible table,
      peg/hole, and gripper. This is shard data-quality evidence only, not
      method evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_live_node_policy_and_partial48_shard0.md`.
- [x] Close the first partial48 branch honestly. Shard0 `98209` produced good
      RGB-D data, but shard1/shard3 failed or were canceled after a concrete
      render failure, and strict inspection `98213` failed with only `24/48`
      RGB-D H5 files. Old visual gate `98215` and dead downstream chain
      `98422-98432` were canceled; incomplete data must not start training.
- [x] Supersede the old full96 branch `97778-97781 -> 97782 -> 98060`.
      It was canceled while pending after the earlier aggregate path
      `98524-98527 + 98841-98844 -> 98845 -> 98846 -> 98847` was submitted.
      Do not treat the canceled 4x4 branch as the current full96 input path.
- [x] Checked whether extra non-wasteful full96 shard jobs could start
      earlier after shard0 completed. Dry-run probes
      `dryrun_probe_8x2_full96_20260602_2011` and
      `dryrun_probe_4x4_full96_20260602_2011` both forecast
      `2026-06-05T12:20:16`, much later than the existing queued paths, so no
      extra render jobs were submitted.
- [x] Partial48 current branch failed strict inspection because
      shard1 job `98210` and shard3 job `98212` hit current rendering failure
      on `server28` and their incomplete outputs were moved out of the active
      RGB-D root. The failure exposed and fixed a wrapper bug:
      `render_dynamic_rgbd_dataset_task.sh` now requires a real `.rgbd.h5`
      before recording a success unit. Repair dry-runs excluding the concrete
      current failing render context forecast `2026-06-03T21:35:35` or later,
      so no repair job was submitted; retry and full96 remain queued.
- [ ] `98213` did fail with exit `64:0` because the active partial48 root had
      only `24/48` RGB-D H5 files. Retry jobs `98524-98527` were submitted with
      a job-local exclusion snapshot based on the concrete `98210/98212`
      render failure context. That submitted snapshot is scheduling history,
      not a reusable node list. Do not treat this as method evidence; it is
      data/rendering failure handling.
- [ ] Current earliest partial RGB-D path is retry branch
      `98524-98527 -> 98528 -> 98530` under
      `experiments/world_model_task_rebinding/rgbd_dynamic_distributed/from_rollout_dir/cpupart_4x2_partial48_retry_after98213_20260602_1958`.
      It requires exact `48/48` RGB-D H5 files, zero structural warnings, and
      nonblank visual artifacts before it can be used as data-quality
      evidence. The attached partial48 GPU method chain `98760-98770` was
      canceled while pending because partial48 must not become formal method
      evidence. Fresh post-chain dry-run probes
      did not find an earlier legal replacement: cpu 1x2 starts
      `2026-06-03T23:16:15`, cpu 1x4 starts `2026-06-04T01:16:27`, gpu
      1x2/1x4 start on `2026-06-05`, debug is blocked by `MaxGRESPerAccount`,
      and test/gaosh/engram are invalid account/partition combinations.
      Queue recheck moved `98524-98527` earlier to
      `2026-06-02T22:01:15` on `server03`; all four 2-GPU jobs are scheduled
      on the same 8-GPU node, so this is not sparse node use.
      Resume dry-runs for late-half shards 4-7 and new full96 1x2/1x4/1x8
      layouts were all later than the existing partial retry or full96 branch,
      so no duplicate render was submitted.
      Later at `2026-06-02T20:52+08:00`, this front-half branch became an
      input to the current full96 aggregate path together with late-half jobs
      `98841-98844`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_retry_chain_2029.md`.
- [ ] Current earliest full96 RGB-D path is now the aggregate branch submitted
      at `2026-06-02T20:52+08:00`: front-half render jobs `98524-98527`
      plus new late-half render jobs `98841-98844`, hardcoded aggregate job
      `98909`, hardcoded exact96 structural inspection `98910`, corrected
      visual gate `98907`, slot training/inspection `98848-98849`, and formal
      downstream RGB-D method chain `98911-98919`. The late-half jobs render disjoint
      shards `4-7` of `8`, so they do not duplicate the queued front half.
      The method chain depends on the visual gate, not merely structural
      inspection. Superseded old partial48 GPU method jobs `98760-98770` and
      old slower full96 4x4 jobs `97778-97781`, `97782`, `97811`,
      `98060-98071` were canceled while pending. All current pending `wm_`
      jobs have `ExcNodeList=(null)`. Visual gate `98847` was later canceled
      before running because its command used the old `job94676` default
      wrapper and the current RGB-D root was not auditable from `scontrol`;
      slot training `98848` now depends on hardcoded aggregate-root visual
      gate `98907`. Aggregate `98845` and strict inspection `98846` were also
      canceled before running because their submitted scripts required
      exported roots not visible from Slurm state; replacements `98909` and
      `98910` hardcode the current front/late/aggregate roots and exact96 gate.
      Old downstream jobs `98850-98858` were canceled before running because
      their predicted-slot metadata pointed to superseded RGB-D source job
      `98845`; replacements `98911-98919` point to aggregate source `98909`.
      This is queued work, not method evidence; current H5 counts are still
      zero. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_full96_aggregate_latehalf_chain_2052.md`.

## Data Export

- [x] Add dynamic RGB-D renderer:
      `scripts/world_model/render_dynamic_rgbd_dataset.py`.
- [x] Add single-GPU Slurm wrapper:
      `scripts/slurm/render_dynamic_rgbd_dataset.sbatch`.
- [x] Add official demo RGB-D replay wrapper:
      `scripts/slurm/replay_peg_official_rgbd.sbatch`.
- [x] Add dense multi-GPU RGB-D generation wrapper:
      `scripts/slurm/render_dynamic_rgbd_dataset_dense.sbatch`.
- [x] Enforce dense allocation in the large RGB-D wrapper. Use up to 8 nodes / 64 GPUs
      only as full-node or otherwise dense allocations; this rule was
      superseded for RGB-D generation by the `2026-06-02T21:55+08:00` user
      override, so current render scheduling may use disjoint one-GPU shards
      when they start sooner. Exact96, visual, RGB-D-derived evidence, and
      4H200/3h training gates are unchanged.
- [x] Save RGB, depth, state/proprio, actions, frame-aligned env states,
      camera parameters, and pose
      labels.
- [x] Add RGB-D dataset inspector:
      `scripts/world_model/inspect_rgbd_dataset.py`.
- [x] Add Slurm RGB-D dataset inspection wrapper:
      `scripts/slurm/inspect_rgbd_dataset.sbatch`.
- [x] Run RGB-D smoke export from the dynamic `hole_move_stop` rollout and
      inspect the contact sheet.
- [x] Inspect all dynamic RGB-D smoke H5s for required cameras, RGB/depth
      shapes, camera parameters, actions, slots, env states, and quantitative
      projection sanity for hole/peg/TCP labels.
- [ ] Run full-shard RGB-D export validation after larger dynamic rollout data
      exists. Former formal render job `94676` was canceled on
      `2026-06-02T19:21+08:00` after user direction to stop waiting on the
      2-node / 16-GPU block; it had produced zero RGB-D H5 files. Its old
      render-output audit `97571`, exact full96 strict inspection `96266`,
      visual gate `97676`, and downstream `96649`-series chain are no longer
      the active path. Old partial/full-shard
      attempts `94541`, `94858`, `94859`, `95235`, `95938`, and the old
      90-file chains were canceled or superseded before producing usable
      full-shard RGB-D method evidence.
- [ ] Run distributed non-wasteful RGB-D shard branch in parallel with `94676`
      instead of passively waiting for the 16-GPU block. Submitted correction
      branch at `2026-06-02T18:46+08:00`:
      render shards `97778`, `97779`, `97780`, and `97781`, each requesting
      one node / four GPUs for two hours and rendering disjoint trajectory
      worklist shards under
      `experiments/world_model_task_rebinding/rgbd_dynamic_distributed/from_rollout_dir/distributed_4x4_now_20260602_184650`.
      Exact-96 structural inspection is `97782`, initial visual no-warning
      audit was `97783`, render failure audit is `97811`, and the initial
      downstream RGB-D method chain was `97784 -> 97785 -> 97786 -> 97787 ->
      97789 -> 97790 -> 97791 -> 97792 -> 97793/97794`, with diagnostic
      sensitivity job `97788`. The initial visual/method chain was later
      canceled and superseded by hardcoded visual gate `98060` and method chain
      `98061`-`98071`. This is a scheduling correction only; at the time of
      submission the combined RGB-D H5 count was still zero, so there was still
      no RGB-D method evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_distributed_shard_correction_1850.md`.
- [x] Correct the distributed branch visual gate. The original visual job
      `97783` used a wrapper whose default root was `job94676` and was not
      auditable enough for the distributed branch. Added hardcoded wrapper
      `scripts/slurm/inspect_rgbd_visual_artifacts_distributed4x4_184650.sbatch`,
      submitted visual gate `98060` after structural inspection `97782`, and
      submitted replacement method chain `98061 -> 98062 -> 98063 -> 98064 ->
      98066 -> 98067 -> 98068 -> 98069 -> 98070/98071`, with diagnostic
      `98065`. Old pending chain `97783`-`97794` was canceled. Render shards
      `97778`-`97781`, structural inspection `97782`, and render audit `97811`
      remain active. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_distributed_visual_gate_correction_1914.md`.
- [x] While formal full RGB-D data are unavailable, run an aligned RGB-D smoke
      chain instead of making state/oracle claims. Slot smoke job `97927`
      trained from RGB-D images plus robot proprio only; predicted-slot export
      job `97944` wrote 8 RGB-D-derived slot H5 files; world-model smoke job
      `97986` trained on those predicted slots; inspection job `98008`
      confirmed `dataset_input_representation=rgbd_predicted_slots`,
      `dataset_world_model_input_group=slots`, `dataset_oracle_slots_read=false`,
      and 41 RGB-D probability/uncertainty auxiliary features in the world-model
      input. This is code-path smoke only, not method evidence and not
      4xH200/3h training evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_predicted_slot_world_model_smoke_1910.md`.
- [x] Re-audit the main `94676` RGB-D gate and legal small-render probes at
      `2026-06-02T19:18+08:00`. `94676` still had zero RGB-D H5 outputs but
      was dense 2-node / 16-GPU and used job-local render exclusions recorded
      in the submitted snapshot. It forecast earlier than fresh 1x2, 1x4, and
      1x8 test-only probes. Submitted snapshots for `96266`, `97571`, and
      `97676` are exact full96/no-warning gates; `96649` and `96654` preserve
      the 4xH200/3h training floor; controller/video inspection still requires
      RGB-D slot source and nonblank video. No duplicate render or replacement
      main chain was submitted. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_main_gate_reaudit_1918.md`.
- [x] Cancel the 16-GPU `94676` path and shift active RGB-D generation to
      smaller-card distributed jobs. `94676` was canceled before start with
      zero H5 outputs. Old `94676`-tied failover/main chains including
      `96278`, `96649`-`96670`, `97039`-`97052`, `97572`, `97676`, and `97677`
      were canceled so they cannot consume large GPU blocks or inspect empty
      data. Fresh probes for conservative 4x2/8x2/4x4, relaxed 1x2/1x4,
      node-directed 1x2, and debug 1x1/1x2 did not produce a runnable earlier
      full-data path; the debug tiny branch `98178 -> 98179 -> 98180` was
      canceled after `StartTime=Unknown`. Active render path is now the
      1-node/4-GPU distributed branch `97778`-`97781` with gates
      `97782`, `97811`, `98060`, and downstream `98061`-`98071`. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_cancel_94676_small_card_shift_1927.md`.
- [x] Submit a CPU-partition small-card partial RGB-D branch after discovering
      that the `cpu` partition accepts GPU GRES and can forecast earlier than
      the active GPU-partition branch. Submitted `98209`-`98212`, each
      1 node / 2 GPUs, on shard indices 0-3 of 8, expected to cover 48/96
      trajectory work units. Gates are `98213` partial strict inspection,
      `98214` render audit, and `98215` visual nonblank/no-warning review.
      This branch is partial48 only and has no downstream method training.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_cpu_partition_partial48_1930.md`.

Smoke record:

- Dynamic state source:
  `experiments/world_model_task_rebinding/smoke_dynamic_state/hole_move_stop_seed1/hole_move_stop_seed1_n1.h5`.
- RGB-D output:
  `experiments/world_model_task_rebinding/smoke_rgbd_dynamic/hole_move_stop_seed1/hole_move_stop_seed1_n1.rgbd.h5`.
- Slurm jobs:
  `94226` first RGB-D smoke, `94241` smoke after adding frame-aligned
  `env_states`; both completed on `server04`.
- Verified fields:
  `rgb: (12, 128, 128, 3) uint8`, `depth: (12, 128, 128, 1) int16`,
  camera params, actions, slots, frame indices, and actor/articulation
  env states.
- Official demo RGB-D wrapper smoke:
  `94247` failed with HDF5 `No locks available`; added
  `HDF5_USE_FILE_LOCKING=FALSE`; retry `94253` completed on `server04` and
  wrote one RGB-D official replay trajectory with 179 observations.
- All six dynamic smoke scenarios now have RGB-D/contact-sheet outputs. Initial
  renders for `none`, `hole_reverse`, and `peg_disturb` failed on `server04`
  with `vk::DeviceLost`; retry jobs `94280`, `94281`, and `94282` succeeded on
  `server27` after excluding `server04`.
- Overview image:
  `experiments/world_model_task_rebinding/smoke_rgbd_dynamic/all_contact_sheets_overview.png`.
- Bounded follow-up overview image:
  `experiments/world_model_task_rebinding/smoke_rgbd_dynamic_bounded/bounded_contact_sheets_overview.png`.
- RGB-D dataset inspection:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_dataset_inspection.md`.
  It checked eight dynamic RGB-D smoke files, 124 RGB-D frames, both
  `base_camera` and `hand_camera`, and reported zero warnings. Projection
  sanity is quantitative metadata, not a replacement for visual contact-sheet
  inspection.
- Full-shard RGB-D backfill:
  `94676` uses rollout source
  `experiments/world_model_task_rebinding/dynamic_state_rollouts/full4gpu/job94510`
  and writes to
  `experiments/world_model_task_rebinding/rgbd_dynamic_dense/from_rollout_dir/job94676`.
  It is queued as a dense 2-node/16-GPU render job; strict inspection `94859`
  will check required cameras and H5 alignment after the render exits.
- 2026-06-02 08:50 queue audit: existing `94541` 1-node/8-GPU render forecasts
  `2026-06-03T00:01:27`, while `94676` 2-node/16-GPU backfill forecasts
  `2026-06-04T23:27:44`. New `sbatch --test-only` probes for dense
  1-node/8-GPU 2h/4h, 2-node/16-GPU 2h, 4-node/32-GPU 2h, and 8-node/64-GPU
  2h allocations all forecast `2026-06-09` or later. No extra RGB-D job was
  submitted because it would not start earlier and would add queue pollution.
- 2026-06-02 09:41 preflight: current RGB-D chain is still pending with no
  new output artifacts. Submit lines and wrappers were rechecked:
  `94676` uses `--nodes=2 --ntasks-per-node=8 --gres=gpu:8`, so it is a
  dense 16-GPU render allocation, not a sparse many-node job. Runtime wrapper
  `render_dynamic_rgbd_dataset_from_rollout_dir_dense.sbatch` also refuses
  sparse multi-node RGB-D renders. Strict inspection `94859` waits on
  `afterany:94676` with `REQUIRE_NO_WARNINGS=true`; slot training `94860`
  waits on `afterok:94859`, requests one node / 4 H200 GPUs, and keeps
  `MIN_TRAIN_SECONDS=10800`. No queue or evaluation change was made.
- 2026-06-02 09:45 queue audit: `94541` remains a dense 1-node/8-GPU render
  job, currently forecast for `2026-06-03T06:01:30`; `94676` remains a dense
  2-node/16-GPU render job, currently forecast for `2026-06-04T18:00:00`.
  The dense wrapper was rechecked and still refuses sparse multi-node RGB-D
  renders. No extra RGB-D job was submitted because the existing jobs already
  use full-node GPU density and new dense probes are not expected to start
  earlier.
- 2026-06-02 09:52 queue audit: `scontrol` shows `94541` has
  `Dependency=(null)` and remains scheduled as a dense 1-node/8-GPU render;
  `94676` has `Dependency=(null)` and remains scheduled as a dense
  2-node/16-GPU render. No full-shard RGB-D output exists yet.
- 2026-06-02 10:07 queue audit: dense full-shard RGB-D job `94541` is running
  on `server58` with one node / 8 H200 GPUs, `obs_mode=rgb+depth`,
  `frame_stride=1`, and six rollout H5 inputs. The log is writing full
  301-frame RGB-D trajectories. The `CUDA_VISIBLE_DEVICES=0` line appears
  inside each single-GPU Slurm task and should be checked by the strict dataset
  inspector through output coverage, not interpreted as sparse allocation.
  Backfill `94676` remains dense at 2 nodes / 16 GPUs with
  `TresPerNode=gres:gpu:8`.
- 2026-06-02 10:12 artifact/queue audit: `94541` has written one complete
  RGB-D H5 so far,
  `task_3/none_seed1000_n16.rgbd/none_seed1000_n16.rgbd.h5`, with 16
  trajectories and `4816` RGB-D frames. Its stderr is empty. Dense
  `sbatch --test-only` probes for 16, 32, and 64 GPU RGB-D renders would start
  on `2026-06-09`, `2026-06-10`, and `2026-06-11` respectively, later than
  the running 8-GPU job and queued 16-GPU backfill. No extra RGB-D job was
  submitted because it would not accelerate data availability and would add
  queue pollution.
- 2026-06-02 10:16 wrapper hygiene: this old record has been superseded. Active
  wrapper source must not carry a standing node exclusion list; RGB-D render
  wrappers keep the dense allocation guard, and new node decisions must come
  from live Slurm state or targeted canaries.
- 2026-06-02 10:25 queue/artifact audit: `94541` remains running on
  `server58` as a dense one-node / 8-H200 render job with `TresPerNode=gres:gpu:8`.
  Its first completed output is still
  `task_3/none_seed1000_n16.rgbd/none_seed1000_n16.rgbd.h5` with 16
  trajectories; no additional RGB-D H5 has appeared yet and stderr remains
  empty. `94676` remains the dense 2-node / 16-GPU backfill, forecast for
  `2026-06-03T17:24:12`. No RGB-D dataset is usable for slot training until
  a render job exits and strict inspection `94858` or `94859` passes.
- 2026-06-02 10:30 queue/artifact audit: `94541` is still running and has not
  written a second RGB-D H5 yet; stderr remains empty. Dense 2-node / 16-GPU
  backfill `94676` is still pending but its scheduler forecast moved earlier
  to `2026-06-02T18:05:41` with reason `Resources`. This is useful queue
  movement only; downstream slot training remains blocked until strict RGB-D
  inspection passes.
- 2026-06-02 10:42 live render audit: `94541` remains running on `server58`
  as a dense one-node / 8-H200 render job. The first completed shard also
  wrote `rgbd_contact_sheet.png` and `rgbd_preview.mp4` under
  `task_3/none_seed1000_n16.rgbd/`. The contact sheet was opened directly and
  is nonblank: peg, box/hole, robot, and changing camera viewpoints are
  visible, so this shard is not a gross Vulkan black-frame or totally wrong
  camera failure. This is RGB-D pipeline evidence only and does not prove the
  full dataset is complete. `sstat` reports five active render tasks for
  `94541.0`, matching the five unfinished perturbation shards; two extra tasks
  had no input because there are six rollout H5s and eight allocated
  GPUs/tasks.
- 2026-06-02 10:49 scheduler/resource audit: `94541` remains running with
  one full 8-H200 node. `scontrol show step 94541.0` reports
  `TresBind=gpu:single:1` and `TresPerTask=cpu:6,gres:gpu:1`, so the repeated
  `CUDA_VISIBLE_DEVICES=0` render-log lines are per-task GPU remapping rather
  than sparse one-GPU-per-node allocation. Only the first RGB-D shard is
  visible so far and stderr is still empty. Backfill `94676` is still pending
  as a dense two-node / 16-GPU job, forecast for `2026-06-02T18:05:41`; since
  it uses the same rollout source as `94541` and the downstream slot training
  chain currently reads `job94676`, rewrite or resubmit the slot chain if
  `94541` completes and passes strict inspection before `94676` starts.
- 2026-06-02 10:54 live render audit: `94541` has still produced only the
  first RGB-D shard, but an in-allocation process/GPU check shows five
  unfinished render Python processes alive for `hole_constant`,
  `hole_move_stop`, `hole_reverse`, `peg_disturb`, and `peg_drop`. `nvidia-smi`
  on the allocated node reports five GPUs at `100%` utilization and about
  `301MiB` each, while three GPUs are idle because there are six input H5s and
  the `none` task already exited. Treat this as an active dense render; wait
  for shard outputs or stderr before intervening.
- 2026-06-02 11:25 live render audit: `94541` remains running on `server58`.
  The same five unfinished render processes are alive and five H200s report
  `100%` utilization. No second RGB-D H5 has appeared yet, so this remains
  slow active rendering rather than inspected data availability. Downstream
  slot training stays blocked until strict inspection `94858` passes.
- 2026-06-02 11:43 intervention: `94541` was canceled after `01:41:44` because
  it still had only the completed static `none_seed1000_n16` RGB-D shard while
  the five dynamic/perturbation shards had no first-trajectory progress line
  and continued to sit at about `99-100%` GPU utilization. Its partial-data
  inspection `94858` was also canceled so one static shard cannot be mistaken
  for a full-shard RGB-D dataset. This is recorded in
  `docs/world_model_task_rebinding/2026-06-02_rgbd_dense_trajectory_sharding.md`.
- 2026-06-02 11:49 dense-render fix: RGB-D dense rendering now shards by
  `(input_h5, trajectory)` rather than by H5 file, giving 96 work units for
  the current six-file full shard. The task script also has a per-unit timeout
  (`RGBD_TASK_TIMEOUT_SECONDS`, default `1800`) and progress events before
  reset, after reset, and after the first rendered frame. Pending dense render
  `94676` was kept because it calls the current task script at runtime; it was
  updated to exclude `server10` and `server58` and is expected to use all 16
  tasks on trajectory-level work units.
- 2026-06-02 11:57 queued-snapshot audit: `scontrol write batch_script 94676`
  shows the submitted main wrapper snapshot is older and does not itself write
  `input_worklist.tsv`, but it still calls
  `scripts/slurm/render_dynamic_rgbd_dataset_task.sh` from the current
  worktree. The task script now self-generates `input_worklist.tsv` when absent
  and appends `input_worklist`, `work_unit_count`,
  `rgbd_task_timeout_seconds`, and `rgbd_shard_trajectories=true` to the job
  manifest, so `94676` has auditable trajectory-level sharding despite being
  submitted before the main wrapper edit.
- 2026-06-02 12:01 queue audit: `94676` remains pending but its forecast moved
  earlier to `2026-06-03T00:01:27` on `server[20,42]`. Live `scontrol` still
  reports the updated render exclusions including `server10,server58`, and the
  allocation remains dense: two nodes / 16 GPUs with 8 GPUs per node. Strict
  inspection and slot-training chain remain `94676 -> 95235 -> 95236 -> 95237`.
- 2026-06-02 12:05 node-specific recovery attempt: added
  `scripts/slurm/render_node_rgbd_canary.sbatch` and submitted one-trajectory
  RGB-D render canaries to the rendering-suspect nodes `server58` (`95265`) and
  `server10` (`95266`). These jobs request one GPU only, render at most four
  frames with a 600-second timeout, and are node-health probes only. They do
  not count as RGB-D dataset evidence or training evidence. The formal dense
  RGB-D chain is still `94676 -> 95235 -> 95236 -> 95237`, with `95236`
  requiring 4 H200 GPUs and `MIN_TRAIN_SECONDS=10800`.
- 2026-06-02 12:07 historical-node-observation availability audit: most old nodes with earlier render failures
  currently have all eight GPUs allocated. `server39` and `server56` showed
  only four allocated GPUs, so additional one-GPU canaries were submitted as
  `95357` (`server39`) and `95358` (`server56`). These remain node-health
  probes only; if a canary passes, the node policy still needs to be updated
  before any large RGB-D or video job uses that node as evidence.
- 2026-06-02 12:15 canary inspection chain: checked existing RGB-D smoke and
  partial dense files and confirmed the expected cameras are
  `base_camera,hand_camera`. Added strict CPU RGB-D inspections after each
  node-specific canary: `95372` after `95265` (`server58`), `95370` after `95266`
  (`server10`), `95371` after `95357` (`server39`), and `95373` after
  `95358` (`server56`). These inspections are still node-health evidence
  only, not formal RGB-D dataset evidence.
- 2026-06-02 12:18 dense-scale scheduling check: used Slurm `--test-only` to
  compare larger dense RGB-D allocations before submitting anything. A 4-node /
  32-GPU dense job was forecast for `2026-06-05T16:39:26`; an 8-node /
  64-GPU dense job was forecast for `2026-06-06T17:12:44`. Existing 2-node /
  16-GPU job `94676` is still forecast for `2026-06-03T00:01:27`, so no
  32/64-GPU duplicate was submitted. This preserves the dense-allocation rule
  without wasting larger resources that would start later.
- 2026-06-02 12:35 user-requested RGB-D/video status check: no new full-shard
  RGB-D dataset exists yet. The only dense RGB-D artifact is the canceled
  `94541` static shard
  `rgbd_dynamic_dense/from_rollout_dir/job94541/task_3/none_seed1000_n16.rgbd`,
  which is not usable full-dataset evidence. Formal chain remains
  `94676 -> 95235 -> 95236 -> 95237`. Current Slurm forecast for `94676` is
  `2026-06-03T00:01:27` on dense nodes `server[20,42]`.
- 2026-06-02 12:35 expansion check: additional `sbatch --test-only` probes
  showed that increasing allocation does not start earlier right now. New
  dense 16-GPU, 32-GPU, and 64-GPU probes forecast `2026-06-04T17:46:27`,
  `2026-06-05T16:37:26`, and `2026-06-06T17:10:44` respectively; a 64-GPU
  30-minute probe is still forecast for `2026-06-06T17:10:44`. Therefore no
  larger duplicate was submitted: larger full-node blocks are harder for Slurm
  to fit than the already-queued 2-node / 16-GPU job.
- 2026-06-02 12:39 preflight fix for pending `94676`: the submitted main
  wrapper snapshot is older, but it calls the current
  `scripts/slurm/render_dynamic_rgbd_dataset_task.sh` at runtime. The task
  script now writes per-task `success_units.tsv` and `failed_units.tsv`; a
  single timed-out or failed trajectory is recorded and the task continues
  with its remaining trajectory work units. This avoids wasting a whole GPU's
  work queue after one bad unit. The strict downstream gate is unchanged:
  `95235` still requires at least 90 `.rgbd.h5` files and no RGB-D inspection
  warnings before slot training `95236` can run.
- 2026-06-02 12:39 preflight checks passed: `bash -n` passed for the dense
  render wrapper, task script, and RGB-D inspection wrapper; `.venv/bin/python
  -m py_compile` passed for RGB-D render/inspection and slot extractor
  scripts. The rollout source for `94676` contains six H5 files with 16
  trajectories each, so the expected trajectory-level worklist has 96 work
  units.
- 2026-06-02 12:55 queue/resource audit: formal RGB-D render `94676` is still
  pending, now forecast by Slurm for `2026-06-03T22:54:26` on dense
  full-node allocation `server[40,42]` with two nodes / 16 GPUs and
  `TresPerNode=gres:gpu:8`. No files exist yet under `job94676`; the only
  RGB-D files under the full-shard output root are from canceled partial
  `job94541`, so slot training has no inspected dataset. Current
  `sbatch --test-only` probes show new dense jobs would not start earlier:
  render-safe 1-node/8-GPU `2026-06-04T17:05:41`, 2-node/16-GPU
  `2026-06-04T23:01:27`, 4-node/32-GPU `2026-06-05T21:54:26`, and
  8-node/64-GPU `2026-06-06T22:27:44`; 30-minute 2/4/8-node probes remain
  no earlier. Relaxing the render exclusion policy also does not beat
  existing `94676` and would put formal data on nodes not yet proven healthy.
  No larger duplicate was submitted because it would consume more full-node
  blocks while starting later than the already-queued job.
- 2026-06-02 13:05 correction to resource policy: "dense" should not mean
  "must use all 8 GPUs on every node." The actual rule is to refuse wasteful
  one-GPU-per-node RGB-D jobs while allowing distributed partial-node shard
  jobs such as one node / 2 GPUs, one node / 4 GPUs, 4 nodes / 4 GPUs, or
  8 nodes / 2 GPUs when that helps throughput. Updated
  `render_dynamic_rgbd_dataset_from_rollout_dir_dense.sbatch`,
  `render_dynamic_rgbd_dataset_dense.sbatch`, and
  `render_dynamic_rgbd_dataset_task.sh` accordingly. Multi-node renders now
  require tasks to divide evenly across nodes and at least
  `MIN_RGBD_GPUS_PER_NODE=2` GPUs per node by default; they no longer require
  8 GPUs per node.
- 2026-06-02 13:05 distributed RGB-D sharding support: added
  `RGBD_JOB_SHARD_INDEX` / `RGBD_NUM_JOB_SHARDS` so multiple Slurm jobs can
  render disjoint trajectory-level worklist shards instead of duplicating the
  same trajectories. Added `OUTPUT_PARENT`, which makes each shard write to
  `OUTPUT_PARENT/shard_i_jobID/` so a single downstream RGB-D inspection can
  recursively validate the combined dataset. Added helper
  `scripts/slurm/submit_rgbd_distributed_shards.sh`; it submits N render
  shard jobs plus optional strict RGB-D inspection and optional slot training,
  and refuses one-GPU-per-node submissions.
- 2026-06-02 13:05 scheduling check after the correction: `bash -n` passed
  for the updated wrappers and submitter. Dry-run distributed schemes are
  valid but not currently faster than `94676`: 8 shards of one-node / 2-GPU
  jobs with 30-minute walltime, 4 shards of one-node / 4-GPU jobs with
  30-minute walltime, and 16 shards of one-node / 2-GPU jobs with 20-minute
  walltime all forecast `2026-06-04T18:36:41` from live scheduler state. Existing
  formal `94676` is still forecast earlier at `2026-06-03T00:01:27`, so no
  distributed backup was submitted in this audit. If `94676` slips or targeted
  canaries plus live Slurm state show a better path, use the distributed submitter
  rather than reimposing a full-node-only rule.

## Slot Extractor

- [x] Add RGB-D slot extractor dataset, training, inspection, and 4-GPU Slurm
      wrappers. These preserve the design rule that RGB-D predicts object
      slots, not actions.
- [ ] Train RGB-D to hole pose and peg pose.
- [ ] Predict confidence and visibility.
      Implementation note at `2026-06-02T17:41+08:00`: visibility predicates
      are already binary RGB-D slot targets, and the pending RGB-D-derived
      world-model path now preserves binary probabilities/std plus continuous
      ensemble std as input features. Formal training/evaluation is still
      pending because full96 RGB-D data are not available yet.
- [ ] Predict grasp/contact predicates if feasible from visual/proprio input.
      Implementation note at `2026-06-02T17:41+08:00`: grasped/inserted are
      already RGB-D slot binary targets and their probabilities/std now reach
      the RGB-D-derived world-model input. Additional contact predicates beyond
      grasped/inserted remain future work.
- [ ] Evaluate pose error in world frame and hole frame.

Slot extractor utilities:

- `scripts/world_model/rgbd_slot_dataset.py`
- `scripts/world_model/train_rgbd_slot_extractor.py`
- `scripts/world_model/inspect_rgbd_slot_extractor_ensemble.py`
- `scripts/slurm/train_rgbd_slot_extractor_ensemble_4gpu.sbatch`
- `scripts/slurm/inspect_rgbd_slot_extractor_ensemble.sbatch`
- Implementation note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_slot_extractor.md`
- RGB-D dataset inspection now defaults to `REQUIRE_NO_WARNINGS=true`; if
  camera, alignment, dtype, depth, or slot warnings are reported, the CPU
  inspection exits nonzero and prevents downstream slot training. This keeps
  corrupted RGB-D data from becoming visual model evidence.
- Earlier queued training chain after full-shard RGB-D backfill inspection:
  `94676 -> 95235 -> 95236 -> 95237`. Job `95235` requires at least 90 RGB-D
  H5 files and no inspection warnings. Job `95236` requests 4 H200 GPUs on one
  node, uses `MIN_TRAIN_SECONDS=10800`, requires at least 90 RGB-D H5 files,
  and explicitly reads RGB-D from `rgbd_dynamic_dense/from_rollout_dir/job94676`;
  it is not evidence until inspection `95237` passes. This pending chain was
  later canceled before running and replaced by the retry-hardened
  `95704 -> 95705 -> 95706` chain; `95704` was then canceled before running
  and replaced by stricter inspection `95938`; `95938` was later canceled
  before running and replaced by exact full96 gate `96266`. Earlier chains
  `94733`/`94734`, `94856`/`94857`, and stale full-shard chains through
  `94859`/`94860`/`94861`, `95235`, and `95938` were canceled before producing
  training evidence.
- 2026-06-02 13:31 correction: the RGB-D slot extractor is not enough by
  itself. Added and queued the missing RGB-D-derived world-model bridge:
  `scripts/world_model/export_rgbd_predicted_slots.py`,
  `scripts/slurm/export_rgbd_predicted_slots.sbatch`, and
  `scripts/slurm/train_rgbd_derived_object_world_model_ensemble_4gpu.sbatch`.
  The first submitted dependency chain was
  `94676 -> 95235 -> 95236 -> 95237 -> 95636 -> 95637 -> 95638 -> 95639`.
  Job `95636` exports RGB-D predicted slots from inspected slot ensemble
  `job95236`; the exported H5s use `slots/*` for RGB-D predictions and
  `oracle_slots/*` only for inspection. Job `95637` trains the object-centric
  world model on those RGB-D-derived slots, requests one node / four
  `NVIDIAH200` GPUs, and enforces `MIN_TRAIN_SECONDS=10800`. Jobs `95638`
  and `95639` inspect and evaluate that RGB-D-derived world-model ensemble.
  This chain was later canceled before running and replaced by the
  retry-hardened chain recorded below. It was not result evidence.
- 2026-06-02 13:40 pre-training export gate: added
  `scripts/world_model/inspect_rgbd_predicted_slot_export.py` and
  `scripts/slurm/inspect_rgbd_predicted_slot_export.sbatch`. Submitted
  export inspection job `95675` after `95636` and updated world-model
  training job `95637` to depend on `afterok:95675`. The formal chain at that
  time was therefore
  `94676 -> 95235 -> 95236 -> 95237 -> 95636 -> 95675 -> 95637 -> 95638 -> 95639`.
  This gate checks structure, frame alignment, finite values, predicted-slot
  uncertainty/probability fields, and the `slots/*` vs `oracle_slots/*`
  boundary before 4H200 world-model training begins. It does not introduce an
  arbitrary quality threshold or change the evaluation protocol.
- 2026-06-02 13:45 import-stability fix and requeue: repeated lightweight
  local imports showed intermittent `.venv` Python extension import failures
  involving `h5py/numpy` (`h5py._objects`, `numpy.lib.type_check`, and
  `numpy.fft._pocketfft_internal`). Added `scripts/slurm/python_retry.sh` and
  wired retry/preflight into RGB-D inspection, slot inspection, slot export,
  RGB-D-derived export inspection, RGB-D slot training, RGB-D-derived
  world-model training, world-model eval, and RGB-D render task worklist
  generation. This does not change evaluation or metrics; it prevents shared
  filesystem/Python import instability from falsely failing the experiment.
  Canceled the old not-yet-running downstream chain
  `95235 -> 95236 -> 95237 -> 95636 -> 95675 -> 95637 -> 95638 -> 95639`
  and submitted the retry-hardened chain
  `94676 -> 95704 -> 95705 -> 95706 -> 95707 -> 95708 -> 95709 -> 95710 -> 95711`.
  Job `95704` was later canceled before running and replaced by strict
  inspection `95938`; this intermediate chain was then superseded by the
  auditable post-RGB-D chain
  `94676 -> 95938 -> 95996 -> 95997 -> 95998 -> 95999 -> 96001 -> 96002 -> 96003`.
  `95996` and `96001` are both one-node / four-`NVIDIAH200` training jobs
  with `MIN_TRAIN_SECONDS=10800`. This 90-file-gated chain was later canceled
  before running and replaced by an intermediate exact-full96 chain
  `96266 -> 96267...96276/96421`, which was later canceled before running.
- 2026-06-02 13:37 node-health audit: `server10` RGB-D canary `95266` failed
  in ManiSkill camera capture with `vk::Device::waitForFences:
  ErrorDeviceLost`; dependent inspection `95370` failed because no RGB-D H5
  existed. This is render-node evidence only. It was job-local failure
  evidence at that time, not a standing exclusion rule; future RGB-D rendering
  must use live Slurm state, targeted canaries, and job-local manifests.
- 2026-06-02 14:30 node-health audit: `server58` RGB-D canary `95265` failed
  after `00:05:07` in ManiSkill camera capture with
  `vk::Device::waitForFences: ErrorDeviceLost`; dependent inspection `95372`
  failed because no RGB-D H5 existed. This is job-local render failure
  evidence, not a reusable bad-node list entry; future RGB-D rendering must
  revalidate the live node context rather than inheriting this exclusion.
- 2026-06-02 14:11 queue/resource audit: formal RGB-D render `94676` remains
  the earliest validated path, forecast for `2026-06-03T00:01:27` on
  `server[20,42]` with 2 nodes x 8 GPUs. Dry-run alternatives that respect
  the no-one-GPU-per-node rule were all later: 1x2 and 1x4 at
  `2026-06-04T16:03:41`, 2x4 at `2026-06-04T21:59:27`, 4x4 at
  `2026-06-05T23:16:29`, 1x8 at `2026-06-04T16:04:41`, 8x2 at
  `2026-06-05T23:17:29`, and 30-minute 1x8/2x8 probes no earlier than
  `2026-06-04T16:04:41`. No backup was submitted because it would not
  accelerate RGB-D data availability.
- 2026-06-02 14:20 render-failure localization: added
  `scripts/world_model/audit_rgbd_render_output.py` and
  `scripts/slurm/audit_rgbd_render_output.sbatch`; submitted audit job
  `95870` with dependency `afterany:94676`. It reads `job94676` manifests,
  worklist, task success/failed ledgers, RGB-D file counts, and Slurm logs.
  It is diagnostic only and does not replace strict RGB-D dataset inspection
  `95938`.
- 2026-06-02 14:26 strict-inspection dependency repair: Slurm does not expose
  pending job export values, so the previously queued inspection `95704` could
  not be proven from current state to carry `MIN_RGBD_FILES=90`. Canceled
  `95704`, submitted replacement `95914` with explicit `MIN_RGBD_FILES=90`,
  `REQUIRE_NO_WARNINGS=true`, and the same `job94676` RGB-D root, then
  replaced it with more auditable wrapper job `95938`. Job `95938` uses
  `scripts/slurm/inspect_full_rgbd_dataset_strict.sbatch`, whose submitted
  batch script itself hardcodes `MIN_RGBD_FILES=90` and
  `REQUIRE_NO_WARNINGS=true`; `95705` now depends on `afterok:95938`. This is
  gate hardening, not evaluation relaxation.
- 2026-06-02 14:39 queue/resource audit after re-reading `AGENTS.md` and the
  active TODO: formal RGB-D render `94676` is still the earliest path and is
  pending for `2026-06-03T00:01:27` on `server[20,42]` with 2 nodes x 8 GPUs.
  The current `job94676` output directory does not exist, so there is no full
  RGB-D dataset and no downstream visual-model evidence yet. Fresh
  `sbatch --test-only` probes that obey the no-one-GPU-per-node rule were all
  later than `94676`: 1x8 at `2026-06-04T16:02:41`, 2x8 at
  `2026-06-04T21:58:27`, 4x4 at `2026-06-05T23:15:29`, 8x2 at
  `2026-06-05T23:15:29`, 4x8 at `2026-06-05T23:15:29`, and 8x8 at
  `2026-06-07T20:44:17`. No duplicate render was submitted because it would
  not accelerate RGB-D availability and would add queue pollution.
- 2026-06-02 14:47 auditable-chain replacement: Slurm does not expose pending
  job `--export` values, so the previous downstream chain after `95938` could
  not be proven from current Slurm state to carry the exact RGB-D root and
  hard thresholds. Added
  `scripts/slurm/submit_auditable_rgbd_method_chain_job94676.sh`, submitted
  replacement chain
  `95996 -> 95997 -> 95998 -> 95999 -> 96001 -> 96002 -> 96003`, plus
  diagnostic `96000` and RGB-D controller/video review `96004 -> 96005/96006`,
  then canceled old pending jobs `95705`-`95711`, `95764`, `95765`, `95781`,
  and `95811`. The new submit manifest is
  `experiments/world_model_task_rebinding/rgbd_method_chains/job94676_auditable_20260602_144737/submit_manifest.txt`.
  It records `RGBD_ROOT=.../job94676`, `MIN_RGBD_FILES=90`,
  `MIN_TRAIN_SECONDS=10800`, `PREDICTED_SLOT_DIR` for world-model training,
  and `SLOT_SOURCE=rgbd` for controller evaluation. This is gate/audit
  hardening only; no method result exists until full RGB-D data, RGB-D slot
  training, RGB-D-derived world-model training, and inspected controller video
  complete. This chain was later canceled before running because a 90-file
  gate is too weak for the 96-trajectory source.
- 2026-06-02 14:52 live-chain audit: after re-reading `AGENTS.md`,
  `00_active.md`, and this file, `94676` remains pending with forecast
  `2026-06-03T00:01:27` on `server[20,42]`. There are no `94676` Slurm logs,
  no files under `rgbd_dynamic_dense/from_rollout_dir/job94676`, and no full
  RGB-D dataset. The auditable downstream chain
  `95996 -> 95997 -> 95998 -> 95999 -> 96001 -> 96002 -> 96003`, plus
  `96000` and `96004/96005/96006`, remains dependency-pending. This is an
  unchanged data-generation blocker, not a method failure and not evidence for
  or against RGB-D task rebinding. That 90-file downstream chain was later
  canceled before running and replaced by an intermediate exact-full96 chain
  `96266 -> 96267...96276/96421`, which was later canceled before running.
- 2026-06-02 14:55 queued-render preflight: `94676` is still pending and its
  submitted batch snapshot predates the main-wrapper Python retry hardening:
  it directly calls Python for rollout validation before launching render
  tasks. The runtime task script is current and does trajectory-level worklist
  rendering with per-unit timeout and retry. Updated the current
  `scripts/slurm/render_dynamic_rgbd_dataset_from_rollout_dir_dense.sbatch`
  so future submissions use `python_retry.sh` for both worklist generation and
  rollout validation. `bash -n` passed. A fresh 2x8 `sbatch --test-only` probe
  with the fixed wrapper forecast `2026-06-05T02:29:27`, later than existing
  `94676`, so no duplicate render was submitted and `94676` remains the
  earliest RGB-D data path. If `94676` fails before rendering because of
  Python/import instability, resubmit using the fixed wrapper instead of
  weakening the data or evaluation gate.
- 2026-06-02 14:59 queue/resource audit: `94676` remains pending on
  `server[20,42]`, but its forecast slipped from `2026-06-03T00:01:27` to
  `2026-06-03T03:00:00`. No `94676` logs or `job94676` files exist. Fresh
  non-wasteful replacement probes are still later: 1x8
  `2026-06-04T20:27:41`, 2x8 `2026-06-05T02:23:27`, and 4x4/8x2
  `2026-06-06T03:40:29`. No duplicate render was submitted. At that moment
  diagnostic audit `95870` and strict RGB-D gate `95938` both remained
  `afterany:94676`, while slot training `95996` remained `afterok:95938`.
  Audit `95870` was later replaced by auditable audit `96069`, which was then
  canceled before running and replaced by full96 audit `96311`.
- 2026-06-02 15:03 audit-gate hardening: Slurm does not expose pending job
  `--export` values, so the generic render audit job `95870` could not be
  proven from current Slurm state to carry `OUTPUT_ROOT=.../job94676` and
  `RENDER_JOB_ID=94676`. Added
  `scripts/slurm/audit_rgbd_render_output_job94676.sbatch`, which hardcodes
  `OUTPUT_ROOT=.../job94676`, `RENDER_JOB_ID=94676`, and
  `MIN_EXPECTED_RGBD_FILES=90`; `bash -n` and `sbatch --test-only` passed.
  Canceled old pending `95870` and submitted auditable replacement `96069`
  with `afterany:94676`. Strict RGB-D data gate `95938` remained unchanged at
  that time and blocked slot training `95996`; it was later canceled before
  running and replaced by exact full96 gate `96266`. Audit `96069` was also
  canceled before running and replaced by full96 audit `96311`, whose submitted
  script uses `MIN_EXPECTED_RGBD_FILES=96`.
- 2026-06-02 15:09 queue/resource audit: `94676` remains pending with
  `StartTime=2026-06-03T03:00:00`, `TresPerNode=gres:gpu:8`, and scheduled
  nodes `server[20,42]`. There are still no `94676` logs, no files under
  `rgbd_dynamic_dense/from_rollout_dir/job94676`, and no RGB-D slot,
  RGB-D-derived world-model, or controller-video evidence. Distributed shard
  `--test-only` probes were checked because they could render disjoint
  trajectory subsets without oracle/state degradation: 8 shards x 1 node x
  2 GPU forecast `2026-06-04T18:46:25`; 4 shards x 1 node x 4 GPU forecast
  `2026-06-04T20:29:41`; 2 shards x 2 nodes x 4 GPU and 1 shard x
  2 nodes x 8 GPU both forecast `2026-06-05T02:25:27`. Since all are later
  than current `94676`, no duplicate render was submitted.
- 2026-06-02 15:19 strict-gate failover chain: inspection revealed that
  render tasks can record failed trajectory units and continue, so render job
  `94676` could exit 0 while strict RGB-D gate `95938` fails because the
  dataset has too few files or warnings. The earlier render-exit failover
  `96155 -> 96156/96157 -> 96163...96173` was canceled to avoid duplicate
  rendering. Replacement failover render `96180` depends on
  `afternotok:95938`, uses the current hardened RGB-D render wrapper, dense
  2-node / 16-GPU allocation, the same rollout source `job94510`, and writes
  to
  `rgbd_dynamic_failover/from_rollout_dir/after95938_gatefail_20260602_1519/shard_0_job96180`.
  Strict RGB-D inspection `96181` and render audit `96182` wait on
  `afterany:96180`. The failover method chain
  `96183 -> 96184 -> 96185 -> 96186 -> 96188 -> 96189 -> 96190`, with
  diagnostic `96187` and video review `96191 -> 96192/96193`, waits on
  `afterok:96181`. This preserves the same strict data gate and only changes
  failure contingency, not evaluation.
- 2026-06-02 15:22 failed-unit hardening: updated
  `scripts/slurm/render_dynamic_rgbd_dataset_from_rollout_dir_dense.sbatch`
  so future RGB-D render submissions aggregate `success_units.tsv` and
  `failed_units.tsv` after `srun`. The wrapper now exits nonzero if any
  trajectory unit failed or if fewer success units exist than worklist units.
  This surfaces partial render failures earlier but keeps strict RGB-D
  inspection as the data gate. Because pending `96180` had already captured
  the older batch snapshot, canceled `96180 -> 96181/96182 -> 96183...96193`
  and submitted hardened gate failover `96215` with dependency
  `afternotok:95938`. Its submitted batch script contains
  `success_unit_count`, `failed_unit_count`, and the refusal checks. Strict
  inspection `96216` and audit `96217` wait on `afterany:96215`; method chain
  `96218 -> 96219 -> 96220 -> 96221 -> 96223 -> 96224 -> 96225`, diagnostic
  `96222`, and video review `96226 -> 96227/96228` wait behind
  `afterok:96216`.
- 2026-06-02 15:25 render-audit hardening: updated
  `scripts/world_model/audit_rgbd_render_output.py` to report
  `missing_success_units` and missing success-output directories, and to
  classify `incomplete_missing_success_units`, `success_outputs_missing`, and
  `failed_units_recorded` before generic missing-output states. This is
  diagnostic only; strict RGB-D dataset inspection remains the data gate. The
  pending audit jobs from that point called this script from the current
  worktree at runtime. Later, the old 90-file audits `96069` and `96217` were
  canceled before running; the next full96 primary audit was `96311`, which
  was later replaced by exact-argument audit `97571`.
- 2026-06-02 15:28 downstream compliance hardening: audited the RGB-D slot
  export, RGB-D-derived world-model training wrapper, and RGB-D controller
  path. Export writes RGB-D predictions under `slots/*` and oracle labels
  only under `oracle_slots/*`; predicted-slot inspection checks the boundary
  string, frame alignment, required prediction uncertainty/probability
  datasets, and compliant slot-model requirement. World-model training already
  refuses inputs missing `slots/rgbd_pred_cont_std` or
  `slots/rgbd_pred_bin_prob` and records `oracle_slots_not_used=true`.
  Added runtime compliance gates to
  `scripts/slurm/evaluate_rebinding_controller.sbatch`: RGB-D slot controller
  runs now fail early unless `RGBD_SLOT_ENSEMBLE_DIR/inspection.json` reports
  `compliant_training_evidence=true`, and world-model controller runs fail
  early unless the ensemble inspection reports compliant training evidence.
  `scripts/slurm/evaluate_rgbd_rebinding_controller_video.sbatch` now
  explicitly exports these require flags for future submissions. Queued video
  jobs `96004` and `96226` execute the current generic wrapper at runtime, so
  the added gates will apply.
- 2026-06-02 15:37 full96 correction: the source rollout directory for
  formal render `94676` contains exactly 96 trajectories, so any 90-file RGB-D
  gate is too weak for method evidence. Added
  `scripts/slurm/inspect_full_rgbd_dataset_job94676_full96.sbatch`, whose
  submitted job `96266` waits on `afterany:94676` and requires exactly 96
  `.rgbd.h5` files, both required cameras, and zero inspection warnings. The
  old 90-file main chain `95938 -> 95996...96006` and old 90-file failover
  `96215 -> 96216...96228` were canceled while pending. The intermediate
  replacement main chain was
  `96267 -> 96268 -> 96269 -> 96270 -> 96272 -> 96273 -> 96274`,
  with diagnostic `96271` and controller video/review `96275 -> 96276/96421`.
  Intermediate replacement failover triggered on `afternotok:96266`: render
  `96278` uses dense 2-node / 16-GPU allocation, inspection `96279` also
  requires exact 96, audit `96280` diagnoses render output, and failover
  method jobs were
  `96281 -> 96282 -> 96283 -> 96284 -> 96286 -> 96287 -> 96288`, with
  diagnostic `96285` and video/review `96289 -> 96290/96422`. Those
  intermediate method chains were later canceled before running. There is
  still no RGB-D method evidence because `job94676` currently has zero
  `.rgbd.h5` files.
- 2026-06-02 15:47 queue/resource audit: formal render `94676` remains the
  earliest RGB-D data path. It is pending for `2026-06-03T00:01:27` on
  scheduled nodes `server[20,42]`, with dense 2-node / 16-GPU allocation and
  `TresPerNode=gres:gpu:8`. `job94676` has zero `.rgbd.h5` files and no
  render logs yet. The submitted `94676` main-wrapper snapshot is older and
  lacks `python_retry.sh`, but it calls the current
  `render_dynamic_rgbd_dataset_task.sh`, whose trajectory-level worklist would
  produce 96 work units from the six source H5 files. If old pre-render
  validation fails, exact full96 gate `96266` will fail and hardened failover
  `96278` will trigger; `96278`'s submitted batch script was inspected and
  includes `python_retry.sh`, `INPUT_WORKLIST`, `work_unit_count`,
  `success_unit_count`, `failed_unit_count`, and the non-wasteful allocation
  guard. Fresh allowed-layout `sbatch --test-only` probes were all later than
  `94676`: 1x2/1x4/1x8 at `2026-06-04T21:47:41`, 2x4/2x8 at
  `2026-06-05T03:43:27`, and 4x4/8x2/8x8 at `2026-06-06T10:44:14`. No
  duplicate render was submitted.
- 2026-06-02 15:51 distributed-inspection hardening: updated generic
  `scripts/slurm/inspect_rgbd_dataset.sbatch` to accept optional
  `EXPECTED_RGBD_FILES` and fail if the actual file count is not exact. Updated
  `scripts/slurm/submit_rgbd_distributed_shards.sh` so its default is
  `MIN_RGBD_FILES=96`, `EXPECTED_RGBD_FILES=96`, and its inspection export
  carries both values. `bash -n`, CPU `sbatch --test-only`, and a dry-run
  distributed submitter check passed. No distributed render was submitted
  because current allowed-layout probes start later than `94676`.
- 2026-06-02 15:53 formal-chain default hardening: audited formal RGB-D
  method-chain wrappers and removed remaining weak default file-count gates.
  The current queued jobs already carry explicit 96-file exports, so no
  dependency-pending job was canceled or resubmitted. The hardening changes
  future standalone/resubmission behavior: `submit_auditable_rgbd_method_chain_job94676.sh`
  now defaults to strict gate `96266` and `MIN_RGBD_FILES=96`;
  slot extraction/export/inspection/sensitivity/world-model wrappers now
  default to 96 formal files; failover and render-audit wrappers default to
  96; and `train_rgbd_slot_extractor_ensemble_4gpu.sbatch` now gathers only
  `*.rgbd.h5` from `INPUT_RGBD_DIR` so state/oracle H5 files cannot enter the
  RGB-D slot training path by directory glob. `bash -n` and targeted
  `sbatch --test-only` checks passed.
- 2026-06-02 16:02 downstream RGB-D method-evidence hardening: audited slot
  extractor and world-model inspection reports. Plain
  `compliant_training_evidence` only proves 4H200/3h, so added explicit method
  evidence fields. `inspect_rgbd_slot_extractor_ensemble.py` now reports
  `full96_rgbd_input_evidence` and `rgbd_slot_training_evidence`; the slot
  inspection now also reports `rgbd_perception_input_evidence`, requiring
  RGB-D image channels from `base_camera` and `hand_camera` plus robot-only
  qpos/qvel proprio with no object-oracle proprio names. The wrapper refuses
  to pass `REQUIRE_COMPLIANT=true` unless `rgbd_slot_training_evidence` is
  true. `export_rgbd_predicted_slots.py` now requires
  `rgbd_slot_training_evidence` and writes a slot inspection summary into
  `export_report.json`; `inspect_rgbd_predicted_slot_export.py` refuses
  exports whose report lacks that evidence. `inspect_world_model_ensemble.py`
  now reports `rgbd_predicted_slot_input_evidence` and
  `rgbd_derived_training_evidence`, requiring RGB-D-predicted `slots/*`,
  `oracle_slots_not_used=true`, full96 predicted-slot inputs, 4H200, and
  >=3h. `evaluate_rebinding_controller.sbatch` now refuses `SLOT_SOURCE=rgbd`
  controller runs unless both the RGB-D slot inspection and world-model
  inspection carry these new method-evidence fields. `bash -n` and
  `py_compile` passed.
- 2026-06-02 16:14 downstream snapshot cleanup: the old pending full96
  world-model inspection/eval/controller jobs were submitted before every
  downstream gate explicitly required RGB-D-derived method evidence. They were
  canceled while dependency-pending and replaced without rerunning render,
  slot training, export, or world-model training. The intermediate replacement
  main downstream was
  `96272 -> 96528 -> 96529 -> 96530 -> 96531/96532`, where `96528` and
  `96529` both require `REQUIRE_RGBD_DERIVED=true`, `96531` requires
  `EXPECTED_SLOT_SOURCE=rgbd`, and `96532` requires readable nonblank video
  artifacts. The intermediate exact-full96 failover downstream was
  `96286 -> 96533 -> 96534 -> 96535 -> 96536/96537` with the same gates.
  Old jobs `96273`, `96274`, `96275`, `96276`, `96421`, `96287`, `96288`,
  `96289`, `96290`, and `96422` were canceled. `job94676` still has zero
  `.rgbd.h5` files, so the correct status remains no RGB-D method evidence.
- 2026-06-02 16:18 queue/resource audit: formal render `94676` remains the
  earliest RGB-D data path, pending for `2026-06-03T00:01:27` on dense
  `server[20,42]` with 16 GPUs and zero `.rgbd.h5` outputs. Fresh allowed
  layouts all start later: 1x2/1x4/1x8 on `2026-06-05T01:46:41`, 2x4/2x8 on
  `2026-06-05T07:42:27`, and 4x4/8x2/8x8 on
  `2026-06-06T14:43:14`; no duplicate render was submitted. Node-risk canary
  evidence does not support relaxing render exclusions: `server10` `95266`
  and `server58` `95265` both failed with Vulkan `ErrorDeviceLost`;
  `server39` `95357` and `server56` `95358` remain pending and start too late
  to accelerate current RGB-D data.
- 2026-06-02 16:28 current-snapshot requeue: the pending `96267...96272`,
  `96528...96532`, `96281...96286`, and `96533...96537` method jobs were
  canceled before running because their submitted batch snapshots still had
  weak formal defaults such as `MIN_RGBD_FILES=2` or
  `MIN_PREDICTED_SLOT_FILES=90`, despite manifests recording 96-file exports.
  Current formal main chain is
  `96649 -> 96650 -> 96651 -> 96652 -> 96654 -> 96655 -> 96656 -> 96657 -> 96658/96659`,
  with diagnostic `96653` after `96652`. Current failover method chain is
  `96660 -> 96661 -> 96662 -> 96663 -> 96665 -> 96666 -> 96667 -> 96668 -> 96669/96670`,
  with diagnostic `96664` after `96663`. The render/data jobs `94676`,
  `96266`, `96311`, `96278`, `96279`, and `96280` were kept. There is still no
  RGB-D method evidence because formal RGB-D generation has not produced any
  `.rgbd.h5` files.
- 2026-06-02 16:35 queue/resource audit: formal render `94676` is still
  pending for `2026-06-03T00:01:27` on `server[20,42]`, dense 2-node /
  16-GPU, with no Slurm logs and zero formal `job94676` `.rgbd.h5` files.
  Fresh allowed replacement probes were all later than the current render:
  1x2/1x4/1x8 at `2026-06-05T01:36:41`, 2x4/2x8 at
  `2026-06-05T07:32:27`, and 4x4/8x2/8x8 at `2026-06-06T14:33:14`. No
  duplicate render was submitted. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_queue_resource_audit_1635.md`.
  Submitted-snapshot audit: `94676` calls the current task script, so it should
  still create per-trajectory worklists and success/failed ledgers, but the
  old batch snapshot lacks final success/failed-unit aggregation. Therefore
  `94676` exit code alone is not data evidence; strict gate `96266` and
  current audit `97571` must classify the output before any RGB-D downstream
  claim.
- 2026-06-02 16:46 RGB-D gate/audit hardening: updated
  `inspect_rgbd_dataset.py` so formal RGB-D inspection warns on
  non-single-trajectory files, non-contiguous `source_frame_indices`, and
  action/frame count mismatches, and now recursively checks nested
  `env_states` frame counts. Also fixed an implementation bug in camera
  projection sanity: the 3x3 intrinsic matrix must project camera xyz, not a
  homogeneous 4-vector. Updated `audit_rgbd_render_output.py` to report
  expected RGB-D count, duplicate success units, success directories without
  `.rgbd.h5`, multiple `.rgbd.h5` in a success directory, and too many/too few
  RGB-D outputs. This is stricter data-alignment/failure-localization coverage,
  not an evaluation relaxation. `py_compile`, `bash -n`, empty-audit smoke,
  and minimal valid RGB-D H5 inspection smoke passed. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_gate_audit_hardening_1646.md`.
- 2026-06-02 16:50 queue recheck: formal render `94676` remains pending with
  reason `Resources`, the same forecast `2026-06-03T00:01:27`, and zero formal
  `job94676` `.rgbd.h5` files. Downstream `96266`, then-current audit
  `96311`, and `96649...96670` remain dependency-pending, so the correct
  status is still no RGB-D method evidence.
- 2026-06-02 16:52 RGB-D slot boundary hardening: fixed
  `rgbd_slot_dataset.py` visibility projection to use camera xyz with the
  3x3 intrinsic matrix. Added explicit member-manifest boundary metadata:
  `input_modality=rgbd_images_plus_robot_proprio`,
  `proprio_source_paths=observations/agent/qpos,qvel`,
  `proprio_boundary=robot_qpos_qvel_only_no_hole_peg_tcp_oracle_state`,
  `target_boundary=object_slots_are_training_labels_not_inference_inputs`,
  and `expected_rgbd_channels`. Hardened
  `inspect_rgbd_slot_extractor_ensemble.py` so
  `rgbd_perception_input_evidence` requires that boundary metadata in addition
  to base/hand RGB-D channels and robot-only proprio names. Positive and
  negative fake-ensemble smoke checks passed. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_slot_boundary_hardening_1652.md`.
- 2026-06-02 16:58 RGB-D world-model boundary hardening: updated
  `export_rgbd_predicted_slots.py` so exported predicted-slot H5 files and
  trajectory groups carry attrs proving `input_representation=rgbd_predicted_slots`,
  `world_model_input_group=slots`, `slot_source=rgbd_slot_extractor_ensemble_prediction`,
  and `oracle_slots_not_used=true`. `inspect_rgbd_predicted_slot_export.py`
  now requires those attrs. `object_slot_dataset.py` records
  `oracle_slots_read=false` and `rgbd_predicted_slot_input_evidence`; the
  world-model trainer refuses predicted-slot-like H5s without complete
  boundary attrs. `inspect_world_model_ensemble.py` now requires per-member
  dataset manifests proving RGB-D-predicted slots before
  `rgbd_derived_training_evidence` can be true. Positive/negative predicted
  export and fake world-model ensemble smoke checks passed. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_world_model_boundary_hardening_1658.md`.
- 2026-06-02 17:11 queue/snapshot audit: formal render `94676` remains
  pending for `2026-06-03T00:01:27` on `server[20,42]`, 2 nodes / 16 GPUs,
  with zero formal `job94676` `.rgbd.h5` files. Fresh legal replacement
  probes were all later: distributed 8x2, 4x4, and 2x8 shards at
  `2026-06-05T01:03:41`; single 2x8 at `2026-06-05T06:57:27`; single 4x4,
  8x2, and 4x8 at `2026-06-06T13:58:14`. No duplicate render was submitted.
  Submitted snapshots were checked: `96266` and then-current audit `96311`
  hardcoded full96/audit gates; failover render `96278` has worklists,
  timeout, success/failed-unit aggregation, and non-wasteful allocation guard;
  current main and failover method-chain `jobs.tsv` records prove 96-file gates,
  4H200/3h training, `REQUIRE_RGBD_DERIVED=true`, `SLOT_SOURCE=rgbd`,
  `EXPECTED_SLOT_SOURCE=rgbd`, and nonblank video artifact gates. This is
  queue/gate evidence only, not RGB-D method evidence. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_queue_snapshot_audit_1711.md`.
- 2026-06-02 17:20 formal RGB-D visual artifact audit added:
  `scripts/world_model/inspect_rgbd_visual_artifacts.py` samples RGB and
  depth frames from formal RGB-D H5s and writes a padded
  `rgbd_visual_review_sheet.png` plus JSON/Markdown reports. Submitted
  primary audit job `97006` with `afterok:96266`; it uses the hardcoded/default
  `job94676` RGB-D root and exact 96-file/nonblank checks. Initial failover
  audit `97007` was canceled before running because its failover root depended
  on hidden `--export`; replacement `97009` uses hardcoded wrapper
  `inspect_rgbd_visual_artifacts_failover96278.sbatch` and waits on
  `afterok:96279`. A smoke run on existing small RGB-D data produced
  `/tmp/reflex_rgbd_visual_smoke/rgbd_visual_review_sheet.png` with zero
  warnings, and the sheet was opened directly. This is visual data-quality
  evidence only; it does not change full96 data gates, training gates, or
  controller evaluation.
- 2026-06-02 17:22 visual-audit training dependency tightened:
  `scontrol update JobId=96649 Dependency=afterok:97006` and
  `scontrol update JobId=96660 Dependency=afterok:97009`. This means main RGB-D
  slot training now waits for the primary RGB-D visual artifact audit, and
  failover RGB-D slot training waits for the failover visual artifact audit.
  Chain-local dependency notes were written under the corresponding main and
  failover method-chain directories. This tightens RGB-D data-quality gating
  before training and does not change full96 inspection, evaluation, or success
  metrics.
- 2026-06-02 17:28 visual-gate failover added:
  if primary exact full96 gate `96266` passes but primary visual artifact audit
  `97006` fails, render failover `97039` triggers with `afternotok:97006`.
  Strict inspection `97040` and audit `97572` wait on `afterany:97039`;
  failover visual artifact audit `97053` waits on `afterok:97040`.
  Failover RGB-D slot training `97042` was updated in Slurm to depend on
  `afterok:97053` instead of the originally recorded `afterok:97040`, so this
  branch also requires exact full96 structure and visual RGB-D data-quality
  review before training. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_visual_gate_failover_1728.md`.
- 2026-06-02 18:25 visual no-warning gate hardening:
  `inspect_rgbd_visual_artifacts.py` now fails `valid_visual_artifacts` when
  any visual-audit warning is present by default. This means missing required
  cameras, non-finite depth, non-single-trajectory sampled files, low RGB
  variation, or zero-positive depth cannot pass the visual data-quality gate
  just because a review sheet was generated. Pending visual audit snapshots
  `97006` and `97053` were canceled and replaced by explicit
  `--require-no-warnings` jobs `97676` and `97677`; current training
  dependencies are `96649 afterok:97676` and `97042 afterok:97677`, and
  visual-failover render `97039` now depends on `afternotok:97676`.
  Lightweight smoke accepted a complete base/hand RGB-D H5 and rejected
  missing hand-camera and non-finite-depth cases. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_visual_no_warning_gate_1825.md`.
- 2026-06-02 17:31 queue recheck:
  `94676` remains pending for `2026-06-03T00:01:27` on `server[20,42]`,
  still 2 nodes / 16 GPUs with `TresPerNode=gres:gpu:8`, and formal
  `job94676` still has zero `.rgbd.h5` files. Fresh legal replacement probes
  would all start later than `94676`: 1x2/1x4/1x8 at
  `2026-06-05T03:03:52`, 2x4/2x8 at `2026-06-05T06:08:27`, and
  4x4/8x2/4x8/8x8 at `2026-06-06T13:09:14`. No duplicate RGB-D render was
  submitted. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_queue_recheck_1731.md`.
- 2026-06-02 17:41 RGB-D uncertainty/probability features:
  fixed the RGB-D predicted-slot to world-model boundary so the world model
  does not discard visual confidence. Export now names continuous and binary
  prediction fields; predicted-slot inspection validates the names; the
  world-model dataset appends continuous ensemble std, binary probabilities
  including visibility/grasp/inserted, and binary ensemble std as features;
  training manifests record actual feature names; and online world-model
  controller features are constructed from the trained manifest contract.
  Lightweight smoke used a synthetic predicted-slot H5 and verified 78-D
  features: 37 base features plus 41 RGB-D auxiliary features. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_world_model_uncertainty_features_1741.md`.
- 2026-06-02 17:45 RGB-D auxiliary-feature evidence gate:
  `inspect_world_model_ensemble.py` now refuses RGB-D-derived world-model
  evidence unless every member manifest proves the actual trained feature
  contract contains RGB-D auxiliary confidence features. Positive/negative
  fake-ensemble smokes showed the gate accepts a 4-member compliant fake
  ensemble with aux features and rejects the same provenance when aux features
  are absent. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_world_model_aux_gate_1745.md`.
- 2026-06-02 17:48 world-model training preflight hardening:
  future RGB-D-derived world-model training wrapper submissions now check for
  `rgbd_pred_cont_std`, `rgbd_pred_bin_prob`, `rgbd_pred_bin_std`, and the
  prediction-name attrs before launching 4H200 training. Lightweight smoke
  accepted a good minimal H5 and rejected missing-bin-std and missing-attrs
  cases. Already submitted `96654/97047` saved batch snapshots predate this
  wrapper edit, but their chain is still protected by export inspection
  `96652/97045` and downstream auxiliary-feature inspection `96655/97048`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_world_model_train_preflight_1748.md`.
- 2026-06-02 17:50 RGB-D controller auxiliary-feature gate:
  RGB-D controller/video wrappers now make the world-model inspection path
  explicit for future submissions and reject `slot_source=rgbd` controller
  execution unless the world-model inspection proves RGB-D auxiliary features
  entered training. Pending `96657/97050` saved video-wrapper snapshots predate
  the explicit export edit, but they execute the current generic evaluator,
  so the aux-feature runtime check applies. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_controller_aux_gate_1750.md`.
- 2026-06-02 17:56 RGB-D eval feature-contract gate and queue audit:
  `evaluate_object_world_model_ensemble.py` now refuses prediction evaluation
  if eval `dataset_meta.feature_names` do not exactly match the training
  member manifests, and `--require-rgbd-derived` additionally requires RGB-D
  predicted-slot evidence plus RGB-D uncertainty/probability auxiliary feature
  evidence in the eval dataset. Lightweight helper smoke accepted the positive
  contract and rejected feature-mismatch and missing-aux cases. Queue recheck
  still shows `94676` pending for `2026-06-03T00:01:27` with zero formal RGB-D
  H5 files; legal replacement probes for 1x4, 1x8, 2x8, 4x4, and 8x2 all
  start later, so no duplicate render was submitted. `server10` and
  `server58` canaries failed with Vulkan `ErrorDeviceLost`; `server39` and
  `server56` canaries remain pending. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_eval_feature_contract_queue_1756.md`.
- 2026-06-02 18:03 RGB-D controller Python-level evidence gate:
  `evaluate_rebinding_controller.py` now enforces the RGB-D method boundary
  even when called directly instead of through the Slurm wrapper. With
  `slot_source=rgbd`, world-model inspection must prove
  `rgbd_derived_training_evidence` and RGB-D auxiliary-feature input evidence;
  loaded world-model member manifests must prove RGB-D-predicted slots,
  `oracle_slots_read=false`, and RGB-D uncertainty/probability aux features;
  RGB-D slot source creation now requires `rgbd_slot_training_evidence` and
  `rgbd_perception_input_evidence`, not just generic compliant training.
  The generic controller sbatch now passes the resolved world-model inspection
  path and compliance switch into Python. Lightweight smokes accepted the
  positive contracts and rejected missing RGB-D-derived world-model evidence,
  missing world-model aux features, and missing RGB-D slot-training evidence.
  Formal RGB-D render `94676` is still pending with zero `.rgbd.h5` files.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_controller_python_gate_1803.md`.
- 2026-06-02 18:08 RGB-D controller inspection evidence gate:
  `inspect_rebinding_controller_run.py` now treats `slot_source=rgbd` as
  insufficient by itself. Candidate RGB-D controller input evidence now
  requires H5 control `slots` to contain frame-aligned finite
  `rgbd_pred_cont_std`, `rgbd_pred_bin_prob`, and `rgbd_pred_bin_std`;
  requires `metric_slots` to be stored separately from control `slots`; and
  requires event-log RGB-D slot uncertainty fields. With
  `--expected-slot-source rgbd`, the inspector exits nonzero if these evidence
  fields are incomplete. `evaluate_rebinding_controller.py` also now writes
  string arrays with UTF-8 HDF5 string dtype so RGB-D slot name fields do not
  break rollout H5 saving. Fake H5 smoke accepted a good RGB-D controller H5
  and rejected a `slot_source=rgbd` H5 missing RGB-D control aux tensors.
  Queue recheck: `94676` is still pending, now with reason `Priority`,
  scheduled for `2026-06-03T00:01:27`, and formal `.rgbd.h5` count remains
  zero. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_controller_inspection_gate_1808.md`.
- 2026-06-02 18:10 pending RGB-D controller snapshot audit:
  audited submitted batch snapshots and method-chain manifests for main
  controller/video jobs `96657/96658/96659` and visual-failover
  `97050/97051/97052`. The RGB-D video wrappers execute the current generic
  controller evaluator, so the Python-level RGB-D world-model/slot gates apply
  at runtime. The controller inspection wrappers pass `--expected-slot-source`
  when exported, and current `jobs.tsv` records `EXPECTED_SLOT_SOURCE=rgbd`.
  Video artifact review wrappers have `REQUIRE_VIDEO=true` and
  `REQUIRE_NONBLANK=true`. No queue change was made. Formal `94676` remains
  pending with zero `.rgbd.h5` files. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_pending_controller_snapshot_audit_1810.md`.
- 2026-06-02 18:19 render-output failure-classification hardening:
  `audit_rgbd_render_output.py` now records failed-unit exit/status counts,
  Slurm/Vulkan/OOM/timeout log pattern samples, and high-level failure
  classes such as `vulkan_device_lost`, `timeout`, and `oom_or_killed`.
  The audit wrappers now pass exact `--expected-rgbd-files 96` to the Python
  audit instead of only a minimum count. `inspect_rgbd_dataset.py` now warns
  when sampled depth is effectively blank. Pending weak audit snapshots
  `96311` and `97041` were canceled and replaced with `97571` and `97572`.
  This is failure-localization/data-quality hardening only; formal `94676`
  still has zero `.rgbd.h5` files, and there is still no RGB-D method evidence.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_render_failure_classification_1819.md`.
- 2026-06-02 18:22 queue/resource probe:
  formal render `94676` remains pending for `2026-06-03T00:01:27` with
  dense 2-node / 16-GPU allocation and zero formal `.rgbd.h5` files. Legal
  single-job probes and distributed-shard probes were all later than `94676`;
  the earliest 1x2/1x4/8-shard-1x2 layouts forecast `2026-06-05T04:04:24`.
  No duplicate render was submitted because it would not accelerate full
  RGB-D availability. Added a global `AGENTS.md` rule requiring `sbatch
  --test-only` or equivalent non-executing probes before duplicate RGB-D
  render submission. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_queue_probe_1822.md`.
- 2026-06-02 18:34 exact-count wrapper hardening:
  future formal slot training, RGB-D predicted-slot export, predicted-slot
  inspection, slot-sensitivity diagnostics, and RGB-D-derived world-model
  training now default to exact expected file-count checks rather than only
  `>=96` minimum checks. `submit_auditable_rgbd_method_chain_job94676.sh` now
  records and propagates expected counts. The current pending 4H200 jobs were
  not requeued because they are already protected by upstream exact data,
  visual, and predicted-slot export gates; requeueing would risk delaying
  training without improving the current evidence chain. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_exact_count_wrapper_hardening_1834.md`.

## RGB-D-Derived World Model

- [x] Add export script that converts RGB-D slot extractor ensemble
      predictions into object-slot trajectory H5s for world-model training.
- [x] Require compliant RGB-D slot extractor inspection before export by
      default.
- [x] Require frame-aligned RGB-D slot predictions by default; if the slot
      extractor was trained/exported with non-unit stride or non-unit history,
      the export refuses to create world-model inputs instead of silently
      training on time-misaligned data.
- [x] Require source RGB-D frames to be contiguous physical frames
      (`source_frame_indices == 0..N-1`) by default, so a rendered dataset
      with frame skipping cannot silently become world-model training data.
- [x] Add a dedicated 4H200/3h training wrapper for RGB-D-derived world-model
      training. The wrapper refuses inputs that do not contain
      `slots/rgbd_pred_cont_std` and `slots/rgbd_pred_bin_prob`, so raw oracle
      state H5s cannot pass as RGB-D-derived training data.
- [x] Add RGB-D predicted-slot sensitivity diagnostic:
      `scripts/world_model/evaluate_rgbd_slot_sensitivity.py` and
      `scripts/slurm/evaluate_rgbd_slot_sensitivity.sbatch`. It refuses to
      run unless the predicted-slot export inspection passed, then measures
      whether RGB-D slot errors flip controller-relevant gates. This is a
      perception-to-control diagnostic, not a changed evaluation protocol.
- [ ] Run export job `96651` after RGB-D slot inspection `96650` passes.
- [ ] Inspect RGB-D predicted slot export with job `96652` before training.
- [ ] Run RGB-D predicted-slot sensitivity diagnostic job `96653` after
      export inspection `96652` passes. This remains diagnostic only and cannot
      become method-success evidence.
- [ ] Run RGB-D-derived world-model training job `96654` after export `96651`
      and export inspection `96652` pass.
- [ ] Inspect RGB-D-derived world-model ensemble with job `96655`.
- [ ] Evaluate RGB-D-derived world-model predictions with job `96656`.

## Integration

- [x] Add `slot_source=rgbd` path to the online rebinding controller. In this
      mode the controller/world-model/handoff logic consumes RGB-D slot
      extractor predictions, while `metric_slots` are stored separately and
      used only for external perturbation scheduling and final success scoring.
- [x] Update controller inspection to prefer `metric_slots` for success and
      dynamic-event checks, so predicted `slots/inserted` cannot become the
      success metric.
- [x] Harden controller inspection so `slot_source=rgbd` is not enough by
      itself. The inspector now requires RGB-D aux tensors in control `slots`,
      separate `metric_slots`, and event-log slot uncertainty before
      `rgbd_controller_input_evidence=true`.
- [x] Add a dedicated RGB-D controller video wrapper:
      `scripts/slurm/evaluate_rgbd_rebinding_controller_video.sbatch`.
- [x] Add video artifact review tooling:
      `scripts/world_model/inspect_video_artifacts.py` and
      `scripts/slurm/inspect_video_artifacts.sbatch`.
      The artifact review now requires readable, basically nonblank videos by
      default and writes sampled padded review sheets. It remains an artifact
      quality gate, not semantic success judgment.
- [ ] Run RGB-D controller video smoke job `96657` after RGB-D-derived
      world-model eval `96656` passes.
- [ ] Inspect RGB-D controller run with job `96658`, then manually inspect the
      produced video/contact sheet before recording any success claim.
- [ ] Generate sampled video review sheets with job `96659` after `96657`;
      then manually open the generated sheet before any success claim.
- [ ] Replace oracle slots with predicted slots in offline evaluation.
- [ ] Interpret controller failures jointly with `96000` slot sensitivity
      results so perception/gate flips are separated from world-model planning
      failures.
- [ ] Add DDP-style baseline only after state rebinding is measurable.

Completion standard: a RGB-D world-model pipeline, trained on full inspected
RGB-D data, predicts task/object state well enough that the rebinding
controller can complete dynamic tasks using RGB-D-derived representations.
State-only success does not satisfy this completion standard.

## 2026-06-03 Repair Chain Update

- [x] `99611` sensors-mode failed-only repair completed as a render/scheduler
      failure, not as method evidence: node `server58`, exit `66:0`,
      elapsed `00:07:21`, `0/3` RGB-D H5, same Vulkan `ErrorDeviceLost`.
      Canceled dependency-dead jobs `99612-99625`.
- [x] Fresh resource probes after `99611`: current user association is only
      account `mayi`; `gaosh`, `engram`, and `test` reject the
      account/partition combination; `cpu` 1GPU/2GPU and 4GPU shapes are
      schedulable but forecast later, and with current render-failure nodes
      excluded the earliest 1GPU repair shape predicted `server42` around
      `2026-06-03T23:54:58`.
- [x] Submitted and completed next exact 3-unit RGB-D repair `99715` with
      `FAILED_SOURCE_ROOT=repair_failed_after99590_sensors_20260603_0055`,
      `OUTPUT_ROOT=repair_failed_after99611_exclude_current_sensors_20260603_0115`,
      `RENDER_MODE=sensors`, and job-local `ExcNodeList` tied only to current
      RGB-D render failures: `server10,server21,server28,server55,server58`.
      This is not a standing bad-node list. Result: `3/3` units rendered on
      `server42`, `failed_units.tsv` empty, three RGB-D H5 files plus preview
      videos/contact sheets written.
- [x] Submitted and completed downstream exact96 chain
      `99716 -> 99717 -> 99718`: aggregate root has exactly `96` RGB-D H5
      files; structural inspection reports `num_files=96`,
      `num_frames=28896`, `num_trajectories=96`, `num_warnings=0`; visual
      inspection reports `valid_visual_artifacts=true`, `num_sampled_files=16`,
      `num_sampled_frames=96`, `num_warnings=0`. The review sheet was opened
      directly and is nonblank.
- [ ] Submitted
      RGB-D method chain `99719-99729` under
      `rgbd_method_chains/full96_aggregate_repair99611_next_20260603_0115_visual99718`.
      Submitted snapshots preserve exact96 structural/visual gates, 4H200
      3.5h slot training `99719`, exact predicted-slot export/inspection,
      4H200 3.5h RGB-D-derived world-model training `99724`,
      `REQUIRE_RGBD_DERIVED=true`, `SLOT_SOURCE=rgbd`, and required video
      artifact review. `99719` started on `server20` at
      `2026-06-03T01:22:55` with one node / four H200 GPUs. No method result
      exists until RGB-D slot training, export, RGB-D-derived world-model
      training/eval, controller video, and video review pass.
- [x] 2026-06-03 01:26 downstream submitted-snapshot audit while `99719`
      trains: wrote submitted batch snapshots for `99720-99729` under
      `/tmp/reflex_sbatch_audit_99719` and checked their contracts. The chain
      preserves `REQUIRE_COMPLIANT=true` for slot inspection, exact
      `EXPECTED_RGBD_FILES=96` and `REQUIRE_FRAME_ALIGNED=true` for predicted
      slot export/inspection, exact `EXPECTED_PREDICTED_SLOT_FILES=96`,
      `MIN_TRAIN_SECONDS=10800`, and `rgbd_pred_*` input validation for
      RGB-D-derived world-model training, `REQUIRE_RGBD_DERIVED=true` for
      world-model inspection/eval, `SLOT_SOURCE=rgbd` for controller video,
      and `REQUIRE_VIDEO=true` plus `REQUIRE_NONBLANK=true` for artifact
      review. `bash -n` passed for downstream wrappers and `py_compile`
      passed for the actual Python entrypoints. This is preflight evidence,
      not model-quality evidence.
- [x] 2026-06-03 01:34 live RGB-D training check:
      `99719` remains running on `server20` with one node / four H200 GPUs.
      The submitted snapshot and job manifest confirm exactly `96` RGB-D H5
      inputs, `MIN_TRAIN_SECONDS=10800`, cameras `base_camera hand_camera`,
      and the intended inference boundary:
      `input_modality=rgbd_images_plus_robot_proprio`,
      `proprio_boundary=robot_qpos_qvel_only_no_hole_peg_tcp_oracle_state`,
      and `target_boundary=object_slots_are_training_labels_not_inference_inputs`.
      `sstat` shows the `99719.0` step has four tasks, about `5.5G` max RSS,
      `00:06:48` average CPU, and about `15GB` disk read, so it is doing
      dataset work even though no member checkpoints or metrics exist yet.
      Next useful action is a queue/artifact check around
      `2026-06-03T02:00+08:00`, unless the job starts writing metrics or
      exits earlier.
- [x] 2026-06-03 01:40 RGB-D controller coverage extension:
      the original method-chain tail had one RGB-D controller video scenario
      (`99727`, `hole_move_stop`). To avoid narrowing dynamic-scene transfer
      to a single move-stop case, submitted additional dependency-gated video
      branches after RGB-D-derived world-model eval `99726`, without changing
      any success metric or bypassing any RGB-D gate:
      `99802 -> 99803/99804` for continuous `hole_constant`,
      `99805 -> 99806/99807` for `hole_reverse`,
      `99808 -> 99809/99810` for `peg_drop` with `ALLOW_REGRASP=true`, and
      `99811 -> 99812/99813` for `peg_disturb` with
      `ALLOW_REGRASP=true`. All controller submit lines keep
      `SLOT_SOURCE=rgbd`, `HOLE_PREDICTOR=world_model`,
      `SAVE_VIDEO=true`, `BRIDGE_SERVO_REFERENCE=phase_hybrid`,
      `BRIDGE_SERVO_MODE=task_frame_projected`,
      `BRIDGE_ORIENTATION_REFERENCE=peg_alignment`, and
      `TASK_SERVO_INSERT_MANIFOLD_GUARD=true`; inspections require
      `EXPECTED_SLOT_SOURCE=rgbd`; video reviews require `REQUIRE_VIDEO=true`
      and `REQUIRE_NONBLANK=true`. Branch table:
      `experiments/world_model_task_rebinding/rgbd_method_chains/full96_aggregate_repair99611_next_20260603_0115_visual99718/generalization_jobs.tsv`.
- [x] 2026-06-03 01:53 live RGB-D slot-training liveness check:
      `99719` is still running, with downstream `99720-99729` and
      generalization video branches `99802-99813` dependency-pending. Current
      `scontrol` confirms the job owns one node, four tasks, and
      `gres/gpu:NVIDIAH200:4` on `server20`. A live overlapped
      `nvidia-smi` query inside the allocation shows all four H200 GPUs active
      (`25-33%` utilization, about `2291 MiB` each), so the per-member
      `cuda_visible_devices=0` lines are Slurm single-GPU task-local views,
      not evidence of a single-GPU run. Member manifests report the intended
      seeds `500-503`; each member has loaded `28896` RGB-D samples from the
      exact `96` RGB-D files. The training script continues epochs until
      `MIN_TRAIN_SECONDS=10800`, rather than sleeping after a short run. This
      is training-liveness and resource-use evidence only. It is still not
      method success; that requires `99720-99729` plus direct video/artifact
      inspection.
- [x] 2026-06-03 01:57 future RGB-D training walltime slack fix:
      found a concrete scheduling/implementation risk: both core RGB-D
      training wrappers enforce `MIN_TRAIN_SECONDS=10800` inside the Python
      trainer after input validation/dataset loading, so a `03:30:00` Slurm
      walltime can leave too little checkpoint/metrics slack when RGB-D input
      loading takes tens of minutes. Attempted to extend running `99719` to
      `04:00:00`, but Slurm returned `Access/permission denied`, so the
      current submitted snapshot remains unchanged. For future submissions,
      changed `train_rgbd_slot_extractor_ensemble_4gpu.sbatch` and
      `train_rgbd_derived_object_world_model_ensemble_4gpu.sbatch` to
      `#SBATCH --time=04:00:00` and added manifest `walltime_reason` lines.
      `bash -n` passed for both wrappers and `.venv` `py_compile` passed for
      the slot and object-world-model trainers. This is a scheduling
      correctness fix only; it does not change model, data, training floor,
      or evaluation. Current `99719` remains a live 3.5h snapshot and must be
      interpreted from its actual completion/inspection artifacts.
- [x] 2026-06-03 02:00 next-scale RGB-D data expansion preflight:
      checked whether existing dynamic rollout data can produce additional
      disjoint RGB-D trajectories while `99719` trains. Current rollout source
      `dynamic_state_rollouts/full4gpu/job94510` contains exactly six
      scenario H5s with `16` trajectories each, total `96`, matching the
      current full96 RGB-D aggregate. Therefore it has no unused disjoint
      dynamic trajectories for another RGB-D shard. A future larger RGB-D
      dataset must first collect new dynamic state rollouts through Slurm,
      then render/inspect RGB-D. Non-submitting 4GPU rollout probes forecast
      `cpu` start `2026-06-04T01:29:58` and `gpu` start
      `2026-06-05T02:01:58`; submitting such a job now could compete with
      `99721/99724`, so no expansion job was submitted. Syntax checks passed
      for `collect_dynamic_state_rollouts_4gpu.sbatch`,
      `render_dynamic_rgbd_dataset_from_rollout_dir_dense.sbatch`, and
      `submit_rgbd_distributed_shards.sh`.
- [x] 2026-06-03 02:05 RGB-D slot localization risk and quality-gate fix:
      live `99719` logs show the current old slot architecture is improving
      over a trajectory-level mean baseline but still has centimeter-scale
      hole/peg-head errors. Dataset checks show depth is stored as millimeter
      `int16`, so the existing `/1000` depth scaling is directionally correct;
      the likely implementation bottleneck is the old slot extractor's
      `AdaptiveAvgPool2d(1)` localization loss. Added a backward-compatible
      future architecture path in `train_rgbd_slot_extractor.py`: future
      training defaults to coordinate channels plus `4x4` spatial pooling,
      while old checkpoints without `coord_conv` still load with
      `coord_conv=false` and `spatial_grid_size=1`. Updated export loading
      and slot inspection reporting accordingly. Added a physical
      predicted-slot quality gate before RGB-D-derived world-model training:
      default max `hole_pos_rmse_m=0.03`, max
      `peg_head_hole_rmse_m=0.035`, max `peg_pos_rmse_m=0.04`, and min
      `binary_accuracy=0.95`. The predicted-slot inspection now separates
      `structural_export_valid` from `valid_export`, so quality failure blocks
      `99724` but still allows diagnostic sensitivity. Updated existing
      `99723` dependency to `afterany:99722`. `py_compile` and wrapper
      `bash -n` checks passed; a lightweight test confirmed new CoordConv
      forward works and old `99719` checkpoint loading remains strict.
- [x] 2026-06-03 02:07 CoordConv RGB-D slot failover chain submitted:
      submitted a dependency-gated RGB-D-only failover chain under
      `experiments/world_model_task_rebinding/rgbd_method_chains/coordconv_slot_failover_after99722_quality_20260603_020714`.
      It starts only if old predicted-slot inspection `99722` fails
      (`99929` dependency `afternotok:99722`), then retrains the RGB-D slot
      extractor with the future CoordConv/spatial-pooling architecture,
      re-exports predicted RGB-D slots, re-runs the physical slot-quality
      gate, trains/evaluates the RGB-D-derived object world model, and runs
      video-reviewed RGB-D controller branches for move-stop, continuous
      hole motion, reverse motion, peg drop/regrasp, and peg disturbance/
      regrasp. Jobs are `99929-99951`; the world-model training job `99934`
      keeps the 4-H200/3h floor with the future 4h walltime slack. This is a
      correctness failover, not a result: it does not relax the quality gate,
      does not use oracle slots, and does not claim method evidence unless
      the downstream RGB-D-derived metrics and video inspections pass.
- [x] 2026-06-03 02:12 old-slot-training notok failover submitted:
      found a concrete dependency gap: if running old-architecture slot job
      `99719` times out or exits nonzero before producing compliant
      `metrics.json/model.pt`, then `99722` never runs and the existing
      `afternotok:99722` CoordConv chain cannot trigger. Added a
      backward-compatible `SLOT_TRAIN_DEPENDENCY` override to
      `submit_auditable_rgbd_method_chain_job94676.sh` and submitted
      RGB-D-only failover chain
      `coordconv_slot_failover_after99719_notok_20260603_021233`.
      Its first job `99966` depends on `afternotok:99719`, uses the same
      exact96 inspected RGB-D data, the future CoordConv/spatial-pooling slot
      path, 4-H200/3h training floor with 4h walltime slack, strict
      predicted-slot quality gate, RGB-D-derived world-model training/eval,
      and video-reviewed controller branches. The chain jobs are
      `99966-99976`, with added generalization video branches
      `99979-99990` for continuous motion, reverse motion, peg drop/regrasp,
      and peg disturbance/regrasp. If `99719` succeeds, this chain remains
      dependency-blocked and does not consume GPUs.
- [x] 2026-06-03 02:18 predicted-slot visual diagnostic added:
      added `scripts/world_model/visualize_rgbd_predicted_slots.py` and
      `scripts/slurm/visualize_rgbd_predicted_slots.sbatch`. The tool draws
      RGB-D predicted `slots/*` and oracle `oracle_slots/*` inspection labels
      on sampled source RGB frames, reporting per-frame hole, peg, and
      peg-head-in-hole errors. This is diagnostic only: it does not change
      the predicted-slot quality gate, world-model input, controller success
      metric, or video evidence requirement. `py_compile`, `bash -n`, and a
      tiny `/tmp` synthetic H5 smoke passed, producing a nonempty review
      sheet with `num_rows=2` and `num_warnings=0`. Submitted CPU visual
      review jobs after possible predicted-slot exports: `100007` after
      `99721`, `100008` after `99931`, and `100009` after `99968`.
- [x] 2026-06-03 02:20 current dependency record correction:
      verified with `scontrol show job 99723` that the active slot
      sensitivity diagnostic now depends on `afterany:99722`, not the
      submit-time `afterok:99722` text in the current chain `jobs.tsv`.
      Updated the chain `jobs.tsv` to match the authoritative Slurm state.
      This keeps diagnostics available when the predicted-slot export is
      structurally valid but fails the physical quality gate. It does not
      change the world-model training dependency: `99724` remains
      `afterok:99722`, so quality failure still blocks RGB-D world-model
      training.
- [x] 2026-06-03 02:24 next-scale RGB-D data expansion queued:
      existing rollout source `job94510` had exactly the `96` trajectories
      already rendered into the current full96 RGB-D set, so broad dynamic
      coverage requires new dynamic rollouts. The old full96 rollout job
      completed in `00:11:14`; a new `64` episodes per scenario rollout is a
      reasonable next-scale data step and not a method shortcut. Added a
      generic visual artifact Slurm wrapper
      `scripts/slurm/inspect_rgbd_visual_artifacts.sbatch`, with `bash -n`
      and `py_compile` checks passing for the structural/visual validation
      path. Submitted low-priority (`Nice=10000`, Slurm priority `1`) data
      expansion chain
      `nextscale_state64_20260603_0225_data_expansion`:
      `100037` collects `384` dynamic state rollouts, `100038` renders
      synchronized RGB-D with dense 1-node/8-GPU allocation after state
      success, `100039` requires exactly `384` RGB-D H5 files with no
      structural warnings, and `100040` requires nonblank visual artifacts
      with no visual warnings. This chain is data expansion only; it is not
      current method evidence and does not preempt the active RGB-D method
      chain.
- [x] 2026-06-03 01:43 RGB-D slot training observability patch:
      inspected `scripts/world_model/train_rgbd_slot_extractor.py` and
      `rgbd_slot_dataset.py` to explain the current no-stdout interval from
      running job `99719`. The trainer prints only after
      `load_rgbd_slot_samples(...)` finishes constructing all image/proprio/
      target arrays, so a long silent startup can be normal for 96 RGB-D H5
      files. Added future-run JSON logs for `rgbd_slot_train_start` and
      `rgbd_slot_dataset_loaded` with sample/tensor shapes and load elapsed
      time. This changes observability only; it does not change data, model,
      loss, training floor, evaluation, or current running job `99719`.
      `.venv` `py_compile` passed for the patched trainer.
- [x] 2026-06-03 01:48 official-data check and data-boundary correction:
      local current RGB-D data are complete for the current full96 dynamic
      diagnostic/training set: aggregate root has exactly `96` RGB-D H5 files,
      structural inspection reports `num_frames=28896`,
      `num_trajectories=96`, `num_warnings=0`, and visual inspection reports
      `valid_visual_artifacts=true` with `16` sampled files and `96` sampled
      frames. This is not a broad "complete enough for generality" dataset; it
      is the current small dynamic RGB-D set for first RGB-D method-chain
      execution. Checked official HuggingFace sources:
      `haosulab/ManiSkill_PegInsertionSide` contains
      `PegInsertionSide-v1.zip` plus `sample.mp4`; the zip contains only
      `motionplanning/trajectory.h5/json`, `rl/trajectory.h5/json`, and sample
      videos, not pre-rendered RGB-D observations. The
      `haosulab/ManiSkill_Demonstrations` PegInsertionSide motionplanning path
      likewise contains `trajectory.h5`, `trajectory.json`, and `sample.mp4`.
      ManiSkill docs say raw demonstration files contain initial states,
      actions, and seeds but usually not observations; RGB/RGB-D must be
      produced by replay/conversion, preferably with `--use-env-states` for
      high-precision tasks. Therefore official data can seed additional
      static RGB-D replay baselines, but it does not remove the need for our
      dynamic RGB-D generation.
- [x] 2026-06-03 02:32 checkpoint-consistency and CoordConv controller fix:
      first-principles reason: the RGB-D method chain must evaluate the
      perception/world-model/controller actually selected by validation, and
      the online controller must instantiate the same slot-extractor
      architecture that was trained. Patched predicted-slot export,
      RGB-D controller, and world-model prediction eval to request
      `best_model.pt` by default, with explicit report fields for the actual
      checkpoint path and a fallback to `model.pt` only when the requested
      default best checkpoint is absent. Also patched the online RGB-D
      controller slot model to read `coord_conv` and `spatial_grid_size` from
      each slot member manifest, preventing a future weight-shape mismatch
      when current CoordConv slot models reach controller video jobs. Future
      Slurm wrappers now record/pass the checkpoint names. `.venv` compile
      passed for the three Python entrypoints, and `bash -n` passed for the
      four affected Slurm wrappers. This does not change RGB-D data gates,
      predicted-slot quality thresholds, world-model metrics, final-state
      controller success, or video review requirements.
- [x] 2026-06-03 02:35 CoordConv chain scheduling correction:
      running old-architecture slot job `99719` is healthy but still shows
      persistent cm-scale validation slot error in stdout, well above the
      strict physical slot-quality gate that must protect RGB-D-derived
      world-model training. A new independent 4H200 test-only submission
      forecast `2026-06-05T03:53:00`, so instead of creating another
      duplicate chain, released the already submitted CoordConv quality
      failover job `99929` from `afternotok:99722` to dependency-free pending.
      `99929` now has `Dependency=(null)`, `EligibleTime=2026-06-03T02:35:45`,
      and remains pending for priority/resources. Canceled duplicate notok
      chain jobs `99966-99976`, `99979-99990`, and `100009`; `sacct` reports
      zero elapsed/allocation for all canceled jobs. This is a queue-efficiency
      correction only: it does not change exact96 RGB-D data gates, 4H200/3h
      floors, strict predicted-slot quality thresholds, RGB-D-derived
      world-model gates, controller final-state metric, or video review.
- [x] 2026-06-03 02:40 submitted snapshot and walltime audit:
      audited already-submitted snapshots for the active and CoordConv chains.
      Old snapshots `99931` and `99936` do not pass explicit checkpoint flags,
      but they call the current Python entrypoints, whose defaults now request
      `best_model.pt` and record actual checkpoint use. Old controller video
      snapshot `99937` still prechecks `model.pt`, then execs the current
      `evaluate_rebinding_controller.sbatch`, so the online controller will
      request `best_model.pt` and still has a fallback report if needed.
      Checked training timing: `99719` is running with `TimeLimit=03:30:00`
      and cannot be extended by the current user (`Access/permission denied`),
      while its stdout shows data loading consumed substantial walltime before
      the 3h train-floor timer. This creates a real timeout risk for the old
      branch, not a reason to lower `MIN_TRAIN_SECONDS`. Successfully updated
      pending RGB-D world-model training job `99724` to `TimeLimit=04:00:00`.
      CoordConv slot job `99929` remains `TimeLimit=04:00:00` and now forecasts
      `2026-06-03T05:16:40` on `server58`.
- [x] 2026-06-03 02:44 afterany slot diagnostics added:
      submitted independent CPU diagnostics `100113` afterany `99719` and
      `100114` afterany `99929` using
      `inspect_rgbd_slot_extractor_ensemble.sbatch` with
      `REQUIRE_COMPLIANT=false`. Outputs are under
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/diagnostics/job99719_afterany`
      and
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/diagnostics/job99929_afterany`.
      These jobs are not method gates and do not feed downstream training;
      they exist so timeout/failure/completion leaves an auditable report
      instead of relying on memory or stdout scraping.
- [x] 2026-06-03 02:45 future slot-log attribution patch:
      patched `train_rgbd_slot_extractor.py` so future batch/epoch stdout
      rows include `event`, `output_dir`, and `seed`. This addresses a concrete
      diagnosis problem in 4-task ensemble training: without member identity,
      four interleaved JSON streams make it hard to tell which slot member has
      which physical error. The patch changes logging only; it does not change
      RGB-D inputs, model loss, optimizer, metrics, gates, training floors, or
      controller evidence. `.venv` `py_compile` passed. `99719` is already
      running and is unaffected; queued `99929` will use the patched Python
      when it starts. Current Slurm forecast for `99929` is
      `2026-06-03T16:00:00`, but this is a scheduler forecast, not evidence.
- [x] 2026-06-03 02:46 old slot branch canceled and classified:
      canceled running old-architecture slot job `99719` and dead downstream
      old-chain jobs `99720-99729`, `99802-99813`, and `100007`. Reason:
      current stdout showed training elapsed could not reach the required
      `MIN_TRAIN_SECONDS=10800` before the unextendable `03:30:00` walltime,
      and recent validation errors were still cm-scale. This avoids wasting
      4 H200 GPUs on a branch that cannot produce compliant RGB-D slot
      training evidence. Afterany diagnostic `100113` completed on `server54`
      and confirms exact96 RGB-D input and RGB-D perception input evidence
      were present, but `num_complete_members=0`,
      `compliant_training_evidence=false`, and
      `rgbd_slot_training_evidence=false`. This is a scheduling/training
      completeness failure of the old branch, not RGB-D method evidence and
      not a reason to lower gates. Active aligned path remains CoordConv
      slot job `99929`.
- [x] 2026-06-03 02:50 live CoordConv chain audit:
      re-read the active RGB-D TODO/PLAN files, checked live `squeue`,
      `squeue --start`, `sacct`, `scontrol show job`, and submitted batch
      snapshots for the active CoordConv RGB-D path. `99929` is dependency-free
      pending on `gpu`, requests one node / `4` H200 GPUs, has
      `TimeLimit=04:00:00`, `MIN_TRAIN_SECONDS=10800`, exact `96` RGB-D input
      requirement, no excluded nodes, and a current scheduler forecast of
      `2026-06-03T15:00:00` on `server34`. Old `99719` stdout shows the
      previous branch spent roughly `28` minutes in setup/load before the
      training-floor timer and had reached only about `3259` train seconds
      when it was canceled, so the 4h pending CoordConv walltime has enough
      margin for the 3h floor plus final checkpoint/metrics if the job runs
      normally. No duplicate 4H200 job was submitted because fresh probes made
      a duplicate later than `99929`, and increasing walltime would risk
      perturbing the earlier scheduled path without a concrete need.
- [x] 2026-06-03 02:52 submitted snapshot/source consistency audit:
      wrote and inspected submitted snapshots for `99929`, `99931`, `99934`,
      and `99937`. The `99929` snapshot calls the current
      `train_rgbd_slot_extractor.py` and does not hard-code the old
      no-CoordConv architecture, so the current CoordConv/spatial-pooling
      defaults and member-attributed logs will be used when the job starts.
      Export/eval/controller Python now default to `best_model.pt` and read
      CoordConv `coord_conv` / `spatial_grid_size` from member manifests.
      The already-submitted controller outer snapshot still checks world-model
      `model.pt`, but current world-model training writes both `best_model.pt`
      and `model.pt`; the inner wrapper then requests `best_model.pt`. This is
      a source/snapshot correctness check only. It does not change exact96
      data gates, slot-quality thresholds, RGB-D-derived world-model gates,
      controller final-state metrics, or video review.
- [x] 2026-06-03 02:53 nextscale expansion boundary audit:
      checked `100037-100040` and the
      `nextscale_state64_20260603_0225_data_expansion` manifest. This branch
      is a Nice=`10000` low-priority data-expansion chain: `100037` collects
      `384` dynamic state rollouts, `100038` renders them as RGB-D on one
      dense 8-GPU node, and `100039/100040` inspect exact `384` structural and
      visual artifacts. It is not current method evidence and should not be
      reported as success. It has no standing node exclusions in the submitted
      snapshots. Keeping it pending is acceptable because it is low priority
      and currently has no forecast; if it begins to compete with the active
      4H200 RGB-D method path, cancel or hold it before sacrificing the core
      `99929 -> 99931 -> 99934 -> controller video` chain.
- [x] 2026-06-03 02:56 RGB-D slot input-boundary source audit:
      inspected `rgbd_slot_dataset.py` and `train_rgbd_slot_extractor.py`.
      The slot extractor input is RGB-D image history from
      `observations/sensor_data/{camera}/{rgb,depth}` plus robot proprio from
      `observations/agent/qpos` and `observations/agent/qvel`. Hole, peg, TCP,
      peg-head-hole, radius, grasped, inserted, and visibility values are
      target labels or validation metrics, not inference inputs. The dataset
      metadata records `input_modality=rgbd_images_plus_robot_proprio`,
      `proprio_boundary=robot_qpos_qvel_only_no_hole_peg_tcp_oracle_state`,
      and `target_boundary=object_slots_are_training_labels_not_inference_inputs`.
      This preserves the RGB-D method boundary for `99929`; it is a source
      audit, not slot quality or method success evidence.
- [x] 2026-06-03 02:57 duplicate-training probe:
      after `squeue --start` moved active `99929` earlier to
      `2026-06-03T03:34:49`, ran non-submitting `sbatch --test-only` probes
      for the same 4H200/4h slot-training shape. A new `gpu` job would start
      `2026-06-05T01:35:02`; a new `cpu` job would start
      `2026-06-04T02:23:02`; `debug` is blocked by `MaxGRESPerAccount`.
      Therefore no duplicate/replacement slot training job was submitted.
      Keep `99929` as the earliest aligned RGB-D slot path.
- [x] 2026-06-03 03:02 controller risk patch before RGB-D video branch:
      while waiting for `99929`, inspected the known state/oracle
      `peg_disturb` failure that the RGB-D controller branch will later face.
      Added default-off insert-stall reseat/regrasp support and submitted
      low-priority state/oracle smoke `100224` to validate it before enabling
      it for RGB-D controller evidence. This is aligned controller debugging,
      not a RGB-D method claim. Active RGB-D controller jobs still keep their
      current behavior unless the smoke result later justifies changing the
      wrapper default or submitted controller environment.
- [x] 2026-06-03 03:07 slot-training partition correction:
      `squeue --start` moved active `99929` to `2026-06-04T03:00:00` on
      `gpu`. Fresh non-submitting probes for the identical 4H200/4h
      slot-training shape showed new `cpu` start `2026-06-04T02:06:36`, new
      `gpu` start `2026-06-05T02:54:36`, and `debug` blocked by
      `MaxGRESPerAccount`. Updated existing pending job `99929` to
      `Partition=cpu` instead of submitting a duplicate. The job remains one
      node / `4` H200, `TimeLimit=04:00:00`, `MIN_TRAIN_SECONDS=10800`,
      exact96 RGB-D input, no excluded nodes, and dependency-free. Current
      forecast after the update still reports `2026-06-04T03:00:00`, so keep
      monitoring; do not submit a duplicate unless it is clearly earlier and
      dependencies are safely redirected.
- [ ] 2026-06-03 03:17 CoordConv slot queue correction and current active
      chain:
      attempted to replace `99929` only after live probes suggested an
      equivalent `cpu` 4H200/4h slot job could start earlier and after adding
      an auditable CoordConv RGB-D method-chain submitter. The replacement
      handling exposed a scheduling/probe discrepancy: old chain `99929-99951`
      plus diagnostics `100008/100114` were canceled before allocation, but
      replacement `100274` then received a much later forecast and was
      canceled; direct probe job `100409` also did not get an earlier actual
      forecast and was canceled. Active RGB-D slot training is now only
      `100319`, pending on `cpu`, one node / `4` H200, `TimeLimit=04:00:00`,
      `MIN_TRAIN_SECONDS=10800`, exact `96` RGB-D input, and
      `ReqNodeList=server03`; it currently has no reliable `squeue --start`
      forecast. Downstream strict RGB-D chain `100411-100433` is dependency
      gated on `100319` and preserves slot inspection, exact96 RGB-D predicted
      slots, strict slot-quality thresholds, RGB-D-derived world-model
      inspection/eval, five RGB-D controller video branches, and video
      artifact review. This is scheduling repair and chain hygiene only, not
      RGB-D method evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_coordconv_slot_queue_correction_0317.md`.
- [ ] 2026-06-03 03:24 active-slot afterany diagnostic:
      submitted CPU diagnostic job `100441` with `Dependency=afterany:100319`
      and `REQUIRE_COMPLIANT=false`, writing to
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/diagnostics/job100319_afterany`.
      First-principles reason: if the active RGB-D slot training job fails or
      times out, strict gate `100411` (`afterok:100319`,
      `REQUIRE_COMPLIANT=true`) will not run, so an independent diagnostic is
      needed to classify whether artifacts are missing, incomplete,
      non-compliant, or usable only for debugging. `100441` does not feed
      predicted-slot export, world-model training, controller evaluation, or
      any method claim.
- [ ] 2026-06-03 03:25 RGB-D-derived world-model afterany diagnostic:
      submitted CPU diagnostic job `100445` with `Dependency=afterany:100415`,
      `REQUIRE_COMPLIANT=false`, and `REQUIRE_RGBD_DERIVED=true`, writing to
      `experiments/world_model_task_rebinding/rgbd_object_world_model/diagnostics/job100415_afterany`.
      First-principles reason: if RGB-D-derived world-model training fails or
      times out, strict gate `100416` (`afterok:100415`,
      `REQUIRE_COMPLIANT=true`, `REQUIRE_RGBD_DERIVED=true`) will not run.
      `100445` classifies artifacts but does not feed world-model eval,
      controller video, or method claims.
- [ ] 2026-06-03 03:29/04:23 RGB-D predicted-slot export afterany diagnostic:
      original CPU diagnostic job `100454` was canceled before allocation and
      replaced by corrected job `100535` with `Dependency=afterany:100412`,
      `MIN_FILES=0`, explicit no exact-count diagnostic expectation,
      `REQUIRE_REPORT=false`, and `REQUIRE_FRAME_ALIGNED=false`, writing to
      `experiments/world_model_task_rebinding/rgbd_predicted_slots/diagnostics/job100412_afterany`.
      First-principles reason: if predicted-slot export fails or partially
      writes files, strict gate `100413` (`afterok:100412`, exact96 files,
      strict frame alignment and slot-quality thresholds) will not run.
      `100535` only classifies zero/partial/malformed artifacts and does not
      feed RGB-D-derived world-model training or method claims.
- [x] 2026-06-03 03:35 current slot job queue-shape correction:
      inspected old slot job `99719` and afterany diagnostic `100113`. The
      old job is explicitly non-compliant: it was canceled after `01:22:48`,
      reports `num_complete_members=0`, `compliant_3h_training=false`, and
      `rgbd_slot_training_evidence=false`, even though its RGB-D input
      boundary metadata were present. It cannot seed the RGB-D method chain.
      Active slot job `100319` was pending with no forecast and stale
      `ReqNodeList=server03`; live `scontrol show node server03` showed all
      8 H200 GPUs allocated. Submitted one actual replacement probe `100465`
      using exact96 RGB-D input, one node / 4 H200, and
      `MIN_TRAIN_SECONDS=10800`, but its actual pending state also had no
      stable earlier forecast, so it was canceled before allocation.
      Temporarily clearing `100319` `ReqNodeList` and updating
      `TimeLimit=03:20:00` moved the actual scheduler candidate to a later
      `server13` slot, and the shorter walltime was rejected as a correctness
      risk because RGB-D data loading occurs before the
      `MIN_TRAIN_SECONDS=10800` training loop. A second actual `server03`
      replacement probe `100470` also failed to get a stable earlier forecast
      and was canceled before allocation. Final active state: `100319` has
      `TimeLimit=04:00:00`, `ReqNodeList=server03`, no stable start forecast,
      and all strict downstream RGB-D method jobs remain gated on `100319`.
      Current status remains pending slot training, with no compliant RGB-D
      slot/world-model/controller evidence yet.
- [x] 2026-06-03 03:42 canceled non-core nextscale expansion queue:
      canceled pending jobs `100037-100040` before allocation. They were a
      Nice=10000 expansion branch for `384` additional state/RGB-D files
      (`100037` 4GPU state rollout, `100038` 8GPU RGB-D render, and
      `100039/100040` inspections), not the current exact96 RGB-D
      slot/world-model/controller method chain. Canceling them prevents a
      non-core branch from occupying scarce GPUs while `100319` waits. This
      removes queue risk only; it produces no data, model, or controller
      evidence.
- [x] 2026-06-03 03:45 official ManiSkill RGB-D source lookup:
      checked official ManiSkill demonstration documentation and Hugging Face
      ManiSkill dataset listings. Official demos are downloadable and useful
      static-task sources, but ManiSkill documents that demos are typically
      compressed/minimal and often store env states rather than observations;
      replay/conversion is the intended way to add visual observations. No
      checked source replaces the current dynamic perturbation RGB-D dataset
      with moving/reversing/stopping holes and peg drop/disturbance events.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_official_maniskill_rgbd_lookup.md`.
- [x] 2026-06-03 04:00 active method-chain contract audit:
      added and ran
      `scripts/world_model/audit_rgbd_method_chain_contract.py` against
      `experiments/world_model_task_rebinding/rgbd_method_chains/coordconv_downstream_after_slot100319_20260603_031723`.
      Output:
      `contract_audit_20260603_0355.json` and
      `contract_audit_20260603_0355.md`. The audit passed `176/176` checks:
      exact96 full96 RGB-D files, structural gate, visual nonblank gate,
      slot training floor `MIN_TRAIN_SECONDS=10800`, strict slot inspection,
      exact96 predicted-slot export/inspection with frame alignment and
      quality thresholds, RGB-D-derived world-model gates, five RGB-D
      controller video branches, and required nonblank video review. This
      is contract/readiness evidence only. Current method evidence is still
      blocked on compliant `100319` slot training, RGB-D-derived world-model
      training/eval, controller metrics, and inspected videos.
- [x] 2026-06-03 04:02 pending-state and submitted-snapshot audit:
      current Slurm state remains pending, not failed. `100319` has
      `Reason=Priority`, `Elapsed=0`, `AllocNodes=0`, `ReqNodeList=server03`,
      one node / 4 H200 / `TimeLimit=04:00:00`, and no `squeue --start`
      forecast. `100411-100433` are dependency-pending; diagnostic
      `100441/100445/100454` are also pending; state/oracle diagnostic
      `100224` is pending and not RGB-D evidence. No slot, predicted-slot,
      world-model, controller, or video artifacts exist for the active chain.
      `scontrol write batch_script 100319` confirms the submitted slot wrapper
      still enforces exact RGB-D file count, one-node 4-GPU training,
      `MIN_TRAIN_SECONDS>=10800`, RGB-D image plus robot-proprio inputs, and
      object-slot labels as training targets only. No duplicate job was
      submitted because the last live replacement probes were minutes ago and
      there is no stable earlier actual allocation evidence.
- [x] 2026-06-03 04:07 RGB-D slot data-path smoke and venv repair:
      before `100319` started, a local lightweight single-H5 dataset smoke
      found `.venv` could not import NumPy because
      `numpy._typing._nested_sequence` was missing from the installed
      `numpy==1.26.4`. This was classified as a training-environment failure
      risk, not a data/model result. There was no repo-local `numpy` path
      shadowing, and `ldd` resolved NumPy shared-library dependencies, so
      NumPy was force-reinstalled in `.venv`. Verification passed:
      `import numpy, h5py, torch`, `pip check`, and the missing module file
      exists. The corrected single-file RGB-D dataset smoke passed with
      `301` samples, image tensor `(8,128,128)` for two RGB-D cameras,
      proprio `(18,)`, continuous target `(25,)`, and binary/visibility target
      `(8,)`. This keeps the RGB-D training path viable; it is not slot
      quality, world-model, controller, or method evidence.
- [x] 2026-06-03 04:11 RGB-D slot model forward smoke:
      with `100319` still pending and no active artifacts, verified the
      submitted environment and ran a one-thread CPU forward smoke on a
      two-sample batch from one RGB-D H5. Current `RgbdSlotExtractor`
      defaults `coord_conv=true` and `spatial_grid_size=4` accept RGB-D image
      batch `(2,8,128,128)` plus robot proprio `(2,18)` and produce
      continuous object/task-slot output `(2,25)` plus binary/visibility
      logits `(2,8)`. This directly checks that RGB-D tensors can reach the
      current CoordConv slot model interface. It is readiness evidence only,
      not training, validation, slot quality, or method evidence.
- [x] 2026-06-03 04:13 reusable RGB-D slot preflight:
      added `scripts/world_model/preflight_rgbd_slot_training_path.py`, a
      lightweight read-only preflight for exact RGB-D file count, imports,
      single-H5 dataset tensor construction, and one current slot-model
      forward pass. Ran it on the active full96 RGB-D root and wrote
      `rgbd_slot_preflight_20260603_0413.json` / `.md` under the active
      method-chain directory. Result: `status=pass`, `num_rgbd_files=96`,
      `num_samples=301`, batch image `(2,8,128,128)`, proprio `(2,18)`,
      prediction shapes `(2,25)` and `(2,8)`, `coord_conv=true`,
      `spatial_grid_size=4`. This creates a repeatable readiness check for
      future RGB-D slot submissions; it is not training or method evidence,
      and it does not relax `100319` or downstream gates.
- [x] 2026-06-03 04:17 RGB-D slot inspection gate audit:
      audited source and submitted snapshots for strict slot inspection
      `100411` and afterany diagnostic `100441`. The strict gate remains
      `afterok:100319` with `REQUIRE_COMPLIANT=true`; the diagnostic remains
      `afterany:100319` with `REQUIRE_COMPLIANT=false`, writes to
      `rgbd_slot_extractor/diagnostics/job100319_afterany`, and is not linked
      to predicted-slot export or method claims. Local logic checks confirmed
      an empty ensemble reports false evidence flags, while a fake
      4-member/3h/exact96/RGB-D-boundary ensemble reports true flags. This
      validates the inspection code path that will classify `100319` outputs
      or failures; it is not slot training or method evidence.
- [x] 2026-06-03 04:23 RGB-D predicted-slot export/inspection audit:
      audited the visual-to-world-model boundary after slot training:
      `export_rgbd_predicted_slots.py` writes RGB-D ensemble predictions under
      `slots/*`, stores oracle labels only under `oracle_slots/*`, and
      refuses non-compliant slot inspections unless explicitly diagnostic.
      Strict `100413` remains `afterok:100412`, exact96, report-required,
      frame-aligned, and quality-gated before RGB-D-derived world-model
      training can run. A diagnostic-only wrapper bug was fixed:
      `inspect_rgbd_predicted_slot_export.sbatch` now treats explicit empty,
      `none`, or `null` `EXPECTED_FILES` as no exact-count diagnostic check,
      while unset `EXPECTED_FILES` still defaults to `MIN_FILES` for strict
      use. Local wrapper smoke on an empty export wrote `expected_files=null`
      and exited `65`, preserving fail-closed semantics. Canceled old
      pending diagnostic `100454` with zero allocation and submitted corrected
      afterany diagnostic `100535`; this diagnostic is not a method gate and
      does not feed world-model training. Active contract audit
      `contract_audit_20260603_0423.json` passed `176/176`.
- [x] 2026-06-03 04:26 RGB-D-derived world-model input contract smoke:
      audited the predicted-slot to world-model path. The 4GPU training
      wrapper `100415` refuses incomplete predicted-slot inputs, undersized
      GPU allocations, sparse multi-node training, and
      `MIN_TRAIN_SECONDS<10800`. `object_slot_dataset.py` constructs both
      current features and future prediction targets from `slots/*`; it
      records `oracle_slots_read=false`, and `oracle_slots/*` stays
      inspection-only. `inspect_world_model_ensemble.py` requires four
      complete members, 3h elapsed, 4xH200, exact predicted-slot file count,
      `input_representation=rgbd_predicted_slots`,
      `world_model_input_group=slots`, and nonempty RGB-D aux
      uncertainty/probability features before `rgbd_derived_training_evidence`
      can be true. A local dataset smoke on old diagnostic predicted-slot
      files wrote
      `world_model_dataset_contract_smoke_20260603_0426.json` with
      `100` samples, feature shape `(100,2,78)`,
      `oracle_slots_read=false`, and RGB-D aux feature names present. This is
      code-path contract evidence only, not current full96 training or method
      evidence.
- [x] 2026-06-03 04:31 controller final-state evidence gate hardening:
      patched `inspect_rebinding_controller_run.py` so controller inspection
      recomputes `success_once` and `success_at_end` directly from
      `rollouts.h5/metric_slots/inserted`, recomputes the final peg-head
      task-frame value from H5 metric slots, and refuses inspection if those
      metric-derived values disagree with the controller summary. This
      preserves the original dynamic task-completion objective because final
      success must be measured from the real final metric state, not only
      from a JSON summary field or RGB-D control slot. Local fake-H5 checks
      passed: a consistent RGB-D run passed inspection, while a run whose
      summary claimed final success but whose metric slots ended uninserted
      exited nonzero. `audit_rgbd_method_chain_contract.py` now checks for
      this source-level guard; `contract_audit_20260603_0431.json` passed
      `178/178`. This is gate hardening only, not controller result or method
      evidence.
- [x] 2026-06-03 04:36 RGB-D method evidence review tail:
      added a post-controller/video aggregation script
      `review_rgbd_method_evidence.py` and CPU wrapper
      `review_rgbd_method_evidence.sbatch`. It reads the active chain
      `jobs.tsv` and controller branch manifest, then checks each branch for
      controller inspection, RGB-D controller input evidence, metric
      final-state summary consistency, final metric success after a prior
      dynamic event, valid video artifacts, and review-sheet paths. It does
      not judge semantics from pixels, does not change metrics, and explicitly
      sets `method_success_claim_allowed=false` until the agent directly
      inspects the videos/contact sheets and records a focused evidence note.
      Local no-result smoke correctly reported all five branches missing
      controller/video evidence. Submitted dependency-gated CPU job `100552`
      after all controller inspection/video review jobs; bad first submission
      `100551` was canceled with zero allocation because `OUTPUT_DIR` used
      literal `jobJOBID`. Added `100552` to `jobs.tsv` and
      `submit_manifest.txt`; latest chain audit
      `contract_audit_20260603_0436.json` passed `183/183`.
- [x] 2026-06-03 04:48 queue/resource and venv package fix while waiting
      for compliant RGB-D slot training:
      `100319` remains the active strict slot blocker and is still pending
      with no `job100319` artifacts. Equivalent exact96, 4H200,
      `MIN_TRAIN_SECONDS=10800` replacement probes did not produce an earlier
      legal path: `cpu` would start `2026-06-04T04:27:14`, `gpu` would start
      `2026-06-05T13:03:14`, `debug` is blocked by account GPU limits,
      `gaosh`/`engram`/`test` reject the account, and `gpux`/`mgpu` are
      unavailable. Live node accounting shows many `mix` nodes have all GPUs
      allocated; the only non-drained nodes with four free GPUs,
      `server39` and `server56`, have only about `10GB` free node memory, so
      they cannot satisfy the 256G training request. Existing `job99719`
      slot artifacts are not reusable: `99719` was canceled after
      `01:22:48`, diagnostic inspection reports zero complete members and no
      RGB-D slot training evidence, and it is the older non-CoordConv grid-1
      model. Fixed a controller/render import risk by adding the missing
      `diffusion_policy/__init__.py` under the ManiSkill baseline package and
      reinstalling it locally with `--no-deps`; `diffusion_policy.make_env`,
      `mani_skill`, `numpy/h5py/torch`, `pip check`, exact96 slot preflight,
      and post-fix chain audit all pass. This is readiness evidence only, not
      RGB-D slot quality or method evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 04:52 cancel non-core state/oracle diagnostic pending job:
      `100224` was a 1GPU state/oracle smoke matrix, not part of the active
      RGB-D method chain and not RGB-D method evidence. It had no artifacts
      and had drifted to a `2026-06-08T00:00:00` forecast, so it was canceled
      before allocation to reduce non-core GPU pending footprint. `sacct`
      reports `CANCELLED by 2059`, `Elapsed=00:00:00`, no assigned node.
      After cancellation, exact96 4H200 slot-training replacement probes were
      still not earlier than the active path: `cpu` forecasts
      `2026-06-04T04:44:18`, `debug` is still blocked by
      `MaxGRESPerAccount`, and `server03` still has all H200 GPUs allocated.
      `100319` remains the active compliant RGB-D slot blocker. No method
      gate, evaluation rule, or RGB-D chain dependency was changed. Evidence
      note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 04:55 submitted slot-training snapshot and memory analysis:
      inspected the submitted `100319` batch snapshot with
      `scontrol write batch_script`; the pending job still enforces exact96
      RGB-D input, one-node 4 H200 training, 256G memory, 4h walltime,
      `MIN_TRAIN_SECONDS=10800`, RGB-D image plus robot-proprio inputs, and
      object slots only as training labels. Live TRES shows no usable
      immediate 4H200/256G slot: `server03` has only 1 free H200,
      `server39` and `server56` each have 4 free H200 but only about `10GB`
      free node memory, and `server16/server29` are down/drain. Replacement
      probes remain later even at lower memory: 256G `cpu` forecasts
      `2026-06-04T04:40:17`, 256G `gpu` forecasts
      `2026-06-05T13:47:17`, and 128G/64G/32G `cpu` forecasts
      `2026-06-04T04:47:18`. The current training implementation loads full
      exact96 RGB-D arrays into memory before training; image tensors alone
      are about `14.1GiB` per member and `56.4GiB` across four members before
      overhead. Lowering memory to chase 10GB-free nodes would be a likely
      OOM shortcut, so no replacement was submitted and no gate was relaxed.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 14:35 strong-backup slot completion and strict inspection:
      `100730` completed normally on `server13` at
      `2026-06-03T14:31:03+08:00` after `03:40:28`; strict inspection
      `100731`, afterany inspection `100757`, and read-only triage `100760`
      also completed. The strict inspection in
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100730/inspection.md`
      records exact full96 RGB-D input, base/hand RGB+depth channels,
      robot-only qpos/qvel proprio, object slots only as training labels, four
      complete members, minimum elapsed `11489` seconds, and compliant
      4H200/3h RGB-D slot-training evidence. Aggregate validation remained
      physically large for task-frame slot geometry (`hole=0.06361m`,
      `peg_head_hole=0.08954m`, `peg=0.05735m`), so this does not prove
      controller-ready rebinding quality or method success. It only clears the
      compliant perception-component boundary and allows predicted-slot export
      `101009` to run from RGB-D observations. No threshold, gate, final
      metric, controller policy, or evaluation protocol changed.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 14:56 strong-backup predicted-slot handoff and world-model
      start:
      CPU-only export `101009` completed on `server52` at
      `2026-06-03T14:53:00+08:00`, writing exact `96` RGB-D-predicted slot
      H5 files plus `export_report.json/md`. Strict export inspection
      `101010` passed at `14:53:25` with `valid_export=True`,
      `quality_gate_passed=True`, `96` files, `96` trajectories, `0`
      warnings, and unchanged metrics `hole=0.01894m`,
      `peg_head_hole=0.02767m`, `peg=0.01462m`,
      `binary_accuracy=0.99834`. Visual review `101030` completed; the
      contact sheet was opened directly and is nonblank/readable with RGB-D
      frames and predicted/oracle slot overlays. Sensitivity `101011`
      completed and records diagnostic control-gate mismatch rates:
      handoff `0.00512`, insert-yz `0.04724`, bridge-close `0.03298`.
      4H200 RGB-D-derived world-model training `101012` is now running on
      `server13` with exact `96` predicted-slot inputs,
      `input_representation=rgbd_predicted_slots`,
      `world_model_input_group=slots`, `oracle_slots_not_used=true`, and
      `MIN_TRAIN_SECONDS=10800`. This is not completed world-model,
      controller, video, or method-success evidence.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:03 optional lazy RGB-D image loading:
      added a future-use lazy RGB-D image path for slot training. It preserves
      RGB-D images plus robot proprio as inputs and object slots as labels,
      but avoids keeping all exact96 images resident in host memory before
      training. The current submitted `100319` job remains eager because its
      Slurm snapshot is already fixed; future legal replacements can opt in
      with `LAZY_IMAGES=true` without changing method gates. Verification:
      Python compile passed, Slurm wrapper syntax passed, full exact96 lazy
      preflight passed with `96` H5 files, and a real single-H5 eager/lazy
      equality check matched `image`, `proprio`, `target_cont`, and
      `target_bin` exactly at indices `0`, `1`, and `10`. This is readiness
      and OOM-risk mitigation only, not RGB-D slot quality or method evidence.
      Post-change contract audit passed `183/183`. Same-objective lazy
      replacement probes were not earlier than the active path: 4h
      8G/12G/16G/32G and 3h15m 8G/16G/32G 4H200 probes all forecast
      `2026-06-04T05:08:30`, while `server39/server56` node-targeted probes
      were later. No replacement job was submitted.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:10 RGB-D slot-training triage tooling:
      added `scripts/world_model/triage_rgbd_slot_training_job.py`, a
      read-only diagnostic for slot-training Slurm state, stdout/stderr,
      ensemble artifacts, member completion, and concrete failure signatures.
      It exists to keep failure response aligned: inspect evidence, classify
      data/rendering/perception/training/scheduling implementation failure,
      then apply a same-objective fix. It does not approve method evidence or
      change any gate. Compile passed, and current `100319` triage output
      classifies only `scheduling_pending`, `pending_reason_priority`, and
      `no_ensemble_artifacts_yet`; no logs or artifacts exist yet.
      Submitted `100588` as an `afterany:100319` CPU-only diagnostic tail,
      wrote it into chain tracking, and reran contract audit; result remains
      `183/183`. This makes post-training failure analysis automatic without
      feeding any method gate.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:13 diagnostic tracking hygiene:
      added existing afterany diagnostics `100441`, `100535`, and `100445`
      to the active chain `jobs.tsv` as tracking-only diagnostic rows. Their
      Slurm submissions, dependencies, and outputs were confirmed from
      `sacct SubmitLine`; no job or method gate was changed. Contract audit
      after the bookkeeping fix still passed `183/183`. A no-result evidence
      review parsed the expanded tracking table and failed closed with exit
      `65` because controller/video artifacts are still absent, not because
      diagnostic rows corrupted the review.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:18 RGB-D-derived world-model training triage:
      added generic ensemble-training triage and submitted `100590`
      (`afterany:100415`) for the RGB-D-derived world-model training job.
      This automatically classifies log/artifact failures after `100415`
      without changing strict inspection/eval gates. Current read-only triage
      correctly reports `100415` as dependency-pending with no artifacts.
      Compile/syntax checks and post-tracking contract audit passed
      (`183/183`).
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:22 RGB-D controller job triage:
      added controller job triage and submitted `100591-100595` after the
      five controller video jobs. This is log/artifact failure localization
      only; strict controller inspections, video reviews, and method evidence
      review remain unchanged. Compile/syntax checks passed, a pending
      `100418` triage classified only dependency/no-artifacts, contract audit
      remained `183/183`, and no-result evidence review still failed closed
      because controller/video artifacts do not exist yet.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:28 RGB-D predicted-slot export triage:
      added predicted-slot export triage and submitted `100597` afterany
      `100412`. This is the diagnostic for the RGB-D perception-to-world-model
      handoff: if export fails, it classifies scheduling, environment/import,
      HDF5 locking, memory/CUDA OOM, exact96 RGB-D input/file-count, slot
      checkpoint/compliance, allocation, or export runtime errors before any
      interpretation. It does not change `100413`, quality thresholds,
      frame-alignment requirements, RGB-D-derived world-model gates,
      controller/video gates, or method evidence review. Compile/syntax
      checks passed, current read-only triage on pending `100412` classified
      only dependency/no-artifacts, `100597` is pending on `afterany:100412`,
      and post-tracking contract audit passed `183/183`. A no-result
      evidence-review smoke after the new row still found all five controller
      branches and failed closed only because controller/video artifacts do
      not exist yet.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:32 slot-training queue probe:
      active strict slot training `100319` is still priority-pending with no
      slot artifacts. Its `sacct SubmitLine` has no explicit `--nodelist`;
      the current `scontrol` `ReqNodeList=server03` is recorded as Slurm
      scheduling state, `scontrol --planned/--noplanned` show the same field,
      and `server03` currently has all `8` H200 GPUs allocated.
      Same-objective non-submitting probes preserving exact96
      RGB-D input, 4 H200 GPUs, and `MIN_TRAIN_SECONDS=10800` forecast later
      paths: `cpu` 256G/4h and lazy 64G/3h15m both at
      `2026-06-04T05:11:30`, `gpu` at `2026-06-05T13:01:30`. No replacement
      was submitted and no downstream RGB-D slot/world-model/controller gate
      was changed. After checking local `scontrol` documentation, cleared the
      stale/non-submitline required-node field with
      `scontrol update JobId=100319 ReqNodeList=`. This keeps the same
      `100319` job and downstream chain but restores `ReqNodeList=(null)`;
      the job remains priority-pending.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:36 slot-training forecast check:
      reread the relevant PLAN/TODO files, then rechecked the live chain.
      `100319` remains pending with `ReqNodeList=(null)`, no artifacts/logs,
      and Slurm forecast `StartTime=2026-06-03T17:35:25` on
      `SchedNodeList=server03`. Same-objective replacement probes were later:
      `cpu` 256G/4h and lazy 64G/128G 3h15m exact96 all forecast
      `2026-06-04T05:11:30`; `gpu` 256G/4h forecasts
      `2026-06-05T12:48:30`. No replacement was submitted. The next aligned
      action is to keep the active `100319` path, avoid queue churn, and
      inspect logs/artifacts immediately if the job starts or fails.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:39 downstream controller/video readiness audit:
      audited the RGB-D controller video wrapper, common controller wrapper,
      controller inspection wrapper, video artifact wrapper, and Python
      entrypoints after rereading the rebinding-controller PLAN. The path
      keeps `SLOT_SOURCE=rgbd`, requires compliant RGB-D slot and
      RGB-D-derived world-model inspections, exports Vulkan/HDF5 settings,
      saves videos, and inspects final success from H5 `metric_slots` with
      RGB-D control slots kept separate. Bash syntax and Python compile
      passed. This is readiness evidence only, not controller or method
      evidence.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:42 predicted-slot export/inspection readiness audit:
      audited the RGB-D slot-to-world-model handoff. Strict export/inspection
      still requires exact96 RGB-D files, compliant slot inspection, frame
      alignment, RGB-D predictions under `slots/*`, `oracle_slots/*` only as
      inspection labels, RGB-D uncertainty/probability tensors, and current
      quality thresholds before world-model training. Bash syntax and Python
      compile passed. No threshold or gate changed.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:43 RGB-D slot-training input boundary audit:
      audited the slot dataset/training/inspection code and wrappers. Slot
      training uses RGB-D camera observations plus robot-only `qpos/qvel` as
      input; object/task slots are prediction targets only. Strict inspection
      requires each member to prove `rgbd_perception_input_evidence`, exact
      full96 RGB-D input, requested 4xH200, and >=3h training before
      `rgbd_slot_training_evidence=true`. Bash syntax and Python compile
      passed. `100319` remained pending with forecast
      `2026-06-03T09:37:26` and no logs/artifacts. No code or gate changed.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:45 RGB-D-derived world-model readiness audit:
      audited the train/inspect/eval wrappers and dataset/training/eval
      entrypoints. Strict world-model training still requires exact96
      predicted-slot files, one-node 4H200, `MIN_TRAIN_SECONDS>=10800`,
      RGB-D aux tensors, `world_model_input_group=slots`, and
      `oracle_slots_read=false`. Inspection/eval require RGB-D-derived
      evidence before downstream controller gates. Bash syntax and Python
      compile passed. No code or gate changed.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:47 submitted slot snapshot and full96 RGB-D input check:
      verified the actual submitted `100319` batch snapshot, not only current
      source. It preserves exact96 RGB-D input, one-node 4 H200 allocation,
      `MIN_TRAIN_SECONDS=10800`, and the boundary
      `rgbd_images_plus_robot_proprio` with robot-only `qpos/qvel`; object
      slots remain training labels only. The current aggregate input has
      exactly `96` RGB-D H5 files, structural inspection `28896` frames with
      `0` warnings, visual inspection `valid_visual_artifacts=true`, and the
      review sheet was opened directly and found nonblank. `100319` is still
      pending with forecast `2026-06-03T09:37:26`; no slot logs/artifacts
      exist yet, so there is still no RGB-D method evidence.
      Post-record contract audit
      `contract_audit_after_submitted_snapshot_20260603_0547` passed
      `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:49 downstream submitted-snapshot audit:
      wrote and syntax-checked submitted snapshots for strict downstream
      jobs `100411-100433` and `100552`. The actual queued jobs preserve
      exact96 RGB-D export, predicted-slot inspection gates, exact96
      RGB-D-derived world-model training, `MIN_TRAIN_SECONDS=10800`,
      `REQUIRE_RGBD_DERIVED=true`, `SLOT_SOURCE=rgbd`, required videos,
      nonblank video checks, and final all-branch evidence review. Controller
      SubmitLines preserve the five intended scenarios and enable regrasp only
      for peg-drop/disturb branches. No submitted-snapshot state/oracle
      fallback was found.
      Post-audit contract check
      `contract_audit_after_downstream_snapshots_20260603_0549` passed
      `183/183`. Live blocker remains pending slot training `100319`; no
      `job100319` logs or artifacts exist yet.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:53 diagnostic tail audit:
      wrote and syntax-checked submitted diagnostic snapshots for
      `100441/100535/100445/100588/100590/100591-100595/100597`. These jobs
      are failure-localization only and are dependency-pending behind their
      strict upstream jobs. Current read-only triage for `100319`, `100412`,
      `100415`, and `100418` reports only priority/dependency pending plus
      no-artifact-yet conditions. No diagnostic path changes exact96, slot
      quality, 4H200/3h, RGB-D-derived world-model, controller metric, video,
      or visual-inspection gates, and no state/oracle fallback was found.
      Reread the relevant PLAN files after the audit; this remains aligned
      with the RGB-D slot -> RGB-D-derived world-model -> controller/video
      method path. Post-record contract audit
      `contract_audit_after_diagnostic_audit_20260603_0553` passed `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 05:56 same-objective resource probe:
      active `100319` remains the earliest aligned slot-training path with
      forecast `2026-06-03T09:37:26`. Legal non-submitting replacements
      preserving exact96, one-node 4H200, and `MIN_TRAIN_SECONDS=10800` were
      all later: `cpu` eager/lazy on `2026-06-04T05:11:30`, and `gpu`
      eager/lazy on `2026-06-05T12:51:30`. No replacement job was submitted,
      and no RGB-D slot/world-model/controller/video gate changed.
      Post-probe contract audit `contract_audit_after_resource_probe_20260603_0556`
      passed `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:03 slot-training runtime-risk audit:
      reread active rules/TODO/PLAN, then audited the still-pending `100319`
      slot-training path using only lightweight local checks. Corrected
      Python-only compile passed for slot training, RGB-D dataset, and
      ensemble inspection entrypoints; Bash syntax passed for the slot
      wrapper, actual submitted `100319` batch snapshot, downstream submitted
      snapshots, and diagnostic snapshots. The actual submitted `100319`
      batch is eager image loading, not the later source wrapper's lazy mode.
      It preserves exact96 RGB-D input, one-node 4H200,
      `MIN_TRAIN_SECONDS=10800`, robot-only `qpos/qvel` proprio input, and
      object/task slots as labels only. Read-only H5 shape audit found `96`
      RGB-D H5 files, `96` trajectories, `28896` frames, two 128x128 RGB-D
      cameras, qpos/qvel shape `(301, 9)`, and estimated eager float32 image
      memory `14.109 GiB` per member / `56.438 GiB` for four members,
      reasonable under the `256G` request. `100319` remains pending priority
      with forecast `2026-06-03T09:37:26` and no slot logs/artifacts. This is
      readiness evidence only; no RGB-D method evidence exists yet.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:06 single-H5 slot interface smoke:
      ran a bounded local dataset/model forward check on one RGB-D H5 with
      HDF5 locking disabled and CPU thread counts set to one. It did not
      train or optimize. The check produced image tensor
      `[301, 8, 128, 128]`, robot-only proprio `[301, 18]`, continuous target
      `[301, 25]`, binary target `[301, 8]`, and model forward outputs
      `[2, 25]` / `[2, 8]`, preserving the
      `rgbd_images_plus_robot_proprio` and slots-as-labels boundary. This is
      an interface smoke only, not slot quality, training, world-model,
      controller, or method evidence.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:10 live chain check and fail-closed review:
      reread active rules/TODO/PLAN, then checked the live RGB-D chain.
      `100319` remains `PENDING`, `Reason=Priority`, `ReqNodeList=(null)`,
      `SchedNodeList=server03`, forecast `2026-06-03T09:37:26`; `sacct`
      shows `AllocNodes=0`, `Elapsed=00:00:00`. There are still no slot logs
      or artifacts. Refreshed read-only triage under
      `experiments/world_model_task_rebinding/rgbd_method_chains/coordconv_downstream_after_slot100319_20260603_031723/current_readonly_triage_20260603_0610`;
      slot/export/world-model/controller triage all classify only priority or
      dependency pending plus no-artifact-yet conditions. A no-result
      evidence-review smoke under
      `.../no_result_evidence_review_smoke_20260603_0610` failed closed with
      exit code `65`, `method_success_claim_allowed=false`, zero candidate
      branches, and missing controller/video inspection for all five
      branches. This is scheduling/fail-closed evidence only; no method
      evidence or gate change.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:17 slot inspection fail-closed smoke:
      with `100319` still pending and no logs/artifacts, ran only a
      lightweight local strict-inspection smoke. Environment sanity showed
      `.venv` can import `typeguard._exceptions`; `typeguard=4.5.2`,
      `tyro=1.0.13`; both `train_rgbd_slot_extractor.py --help` and
      `review_rgbd_method_evidence.py --help` exit `0`. A first attempt under
      `no_result_slot_inspection_smoke_20260603_0614` is explicitly
      discarded as non-evidence because a transient import/command issue
      prevented `inspection.json` from being written. The corrected smoke
      under `no_result_slot_inspection_smoke_20260603_0617` wrote
      `inspection.json` with `num_members=0`,
      `compliant_training_evidence=false`, and
      `rgbd_slot_training_evidence=false`; the equivalent
      `REQUIRE_COMPLIANT=true` gate exits `65`. This is fail-closed gate
      evidence only; no method evidence, no code change, and no gate change.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:20 future wrapper CLI import preflight hardening:
      `100319` remains pending with no artifacts; this does not affect the
      already submitted `100319` batch snapshot and is not a replacement
      submission. Hardened only future source wrappers for RGB-D slot
      training, predicted-slot export, and RGB-D-derived world-model training
      so their Python preflight imports `tyro` and `typeguard._exceptions` in
      addition to `h5py/numpy/torch`. This prevents future GPU allocations
      from reaching the training/export entrypoint before discovering a CLI
      environment issue. Bash syntax passed for all three wrappers, and local
      venv import sanity passed. No training code, input modality, exact96
      gate, 4H200/3h floor, predicted-slot threshold, RGB-D-derived
      world-model requirement, controller metric, video review, or method
      evidence rule changed.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:22 submitted-batch drift audit:
      wrote the Slurm-stored `100319` batch script to
      `experiments/world_model_task_rebinding/rgbd_method_chains/coordconv_downstream_after_slot100319_20260603_031723/submitted_batch_100319_after_source_hardening_20260603_0622.sh`
      and compared it against the 05:46 submitted snapshot. `cmp=0`, so the
      active job is byte-identical to the earlier submitted batch: exact96
      eager RGB-D slot training, one-node 4H200, `MIN_TRAIN_SECONDS=10800`,
      RGB-D image plus robot-only `qpos/qvel` inputs, and object/task slots as
      labels only. The future source wrapper's added `tyro` /
      `typeguard._exceptions` import preflight and lazy-image path are not in
      the active `100319` job. This is submitted-job consistency evidence
      only, not slot quality, world-model, controller, or method evidence.
      Post-record contract audit
      `contract_audit_after_submitted_batch_drift_20260603_0622` passed
      `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:26 same-objective resource probe:
      active `100319` is still the earliest aligned RGB-D slot-training path:
      pending priority, `ReqNodeList=(null)`, `SchedNodeList=server03`,
      forecast `2026-06-03T09:37:26`, no slot logs, and no artifacts.
      Non-submitting replacement probes with exact96 RGB-D, one-node 4H200,
      and `MIN_TRAIN_SECONDS=10800` were later or invalid. `cpu` eager/lazy
      replacements forecast `2026-06-04T08:47:26`; `gpu` eager/lazy forecast
      `2026-06-05T16:19:26`; `debug` is blocked by `MaxGRESPerAccount`;
      `gpux`/`mgpu` are inactive or drained; `gaosh`/`engram`/`test` reject
      the current account/partition combination. Test-only IDs `100652-100656`
      are absent from `squeue`, so no job was actually submitted. No
      replacement was submitted because it would not complete the same RGB-D
      training objective earlier. This is resource evidence only, not method
      evidence.
      Post-probe contract audit
      `contract_audit_after_resource_probe_20260603_0626` passed `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:29 downstream dependency/output-path audit:
      checked actual queued Slurm dependencies and current output directories.
      The strict method path remains fail-closed on successful upstream gates:
      slot inspection `100411` waits `afterok:100319`, predicted-slot export
      `100412` waits `afterok:100411`, predicted-slot inspection `100413`
      waits `afterok:100412`, RGB-D-derived world-model training `100415`
      waits `afterok:100413`, world-model inspection/eval `100416/100417`
      wait `afterok`, and controller video jobs wait `afterok:100417`.
      Diagnostic, sensitivity, controller-inspection, video-review, and triage
      jobs are `afterany` only for fail-closed inspection or
      failure-localization. Predicted-slot output `job100412` and
      RGB-D-derived world-model output `job100415` each contain `0` files, so
      no stale downstream artifacts exist. This is dependency hygiene only,
      not method evidence.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:30 exact96 RGB-D data generation status:
      the RGB-D render/aggregate blocker is resolved. Aggregate
      `full96_aggregate_p48retry98524_98527_latehalf_repair99208_repair99611_next_20260603_0115`
      contains `96` `.rgbd.h5` files. Structural inspection reports
      `num_files=96`, `num_warnings=0`; visual artifact inspection reports
      `valid_visual_artifacts=true`, `num_files=96`, `num_warnings=0`.
      The visual review sheet was opened directly and is nonblank. Render
      shards and repair attempts included several SAPIEN/Vulkan `DeviceLost`
      failures, but aggregate job `99716` completed with exact96 inspected
      files. The current blocker is now RGB-D slot training `100319`, pending
      priority with forecast `2026-06-03T07:12:33`, no logs, and no slot
      artifacts. This proves data readiness only; RGB-D slot/world-model/
      controller method evidence is still missing.
      Post-record contract audit
      `contract_audit_after_rgbd_status_20260603_0630` passed `183/183`. At
      `2026-06-03T06:33:04+08:00`, `100319` was still pending priority with
      no slot logs/artifacts and downstream strict jobs remained
      dependency-pending. Recent `.pyc` files generated by lightweight local
      Python checks were deleted.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:35 scheduled-node sanity:
      `100319` remains pending priority with `SchedNodeList=server03`,
      forecast `2026-06-03T07:12:33`, and no slot logs/artifacts. Live node
      checks show `server03` is `MIXED`, not drained/down, has
      `Gres=gpu:NVIDIAH200:8`, and currently has all eight GPUs allocated
      (`AllocTRES` includes `gres/gpu=8`). `sinfo` reports reason `none`.
      This supports the current interpretation that the blocker is scheduling
      for a real H200 allocation, not a bad-node or wrong-resource artifact.
      No replacement or duplicate probe was submitted.
      Post-node-check contract audit
      `contract_audit_after_node_sanity_20260603_0635` passed `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:38 read-only slot triage refresh:
      refreshed `100319` triage under
      `current_readonly_triage_20260603_0638`. It still classifies only
      `scheduling_pending`, `pending_reason_priority`, and
      `no_ensemble_artifacts_yet`; stdout/stderr do not exist, the ensemble
      directory does not exist, member dirs are `0`, complete members are `0`,
      and there are no log pattern hits. Recommended next action is to wait on
      the current path and recheck, or probe only if a legal same-objective
      shape may start earlier. This is scheduling/failure-localization
      evidence only, not method evidence.
      Post-triage contract audit
      `contract_audit_after_readonly_triage_20260603_0638` passed `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:41 slot-training startup checklist:
      added chain-local checklist
      `slot100319_startup_and_completion_checklist_20260603_0641.md` so
      `100319` startup/completion checks are tied to the existing strict
      contract: exact96 RGB-D input, one-node four-H200 request,
      `MIN_TRAIN_SECONDS=10800`, RGB-D plus robot-proprio inputs, slots as
      labels, four complete members, and strict `100411` inspection before
      downstream method claims. This is operational hygiene only and changes
      no training code, threshold, resource, dependency, evaluation gate, or
      evidence rule.
      Post-checklist contract audit
      `contract_audit_after_slot100319_checklist_20260603_0641` passed
      `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:44 scheduling read-only check:
      `100319` remains pending priority with stable forecast
      `2026-06-03T07:12:33`, no slot logs, and no artifacts. `squeue --start`
      reports the same forecast. `server03` is still `MIXED` with
      `Gres=gpu:NVIDIAH200:8`, but live `AllocTRES` is now `gres/gpu=7`, so
      only one GPU is free while `100319` needs four H200 GPUs. No replacement
      or duplicate resource probe was submitted because the last same-objective
      probe was at `06:26`, below the 30-minute cadence. This is scheduling
      evidence only, not method evidence.
      Post-scheduling-check contract audit
      `contract_audit_after_scheduling_check_20260603_0644` passed `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 06:56 same-objective resource probe:
      `100319` remained pending priority with forecast
      `2026-06-03T07:12:33`, no slot logs, and no artifacts. Non-submitting
      exact96 one-node 4H200 replacement probes preserving
      `MIN_TRAIN_SECONDS=10800` were later or invalid: `cpu` eager/lazy
      forecasts `2026-06-04T06:22:33`, `gpu` eager/lazy forecasts
      `2026-06-05T13:55:33`, `debug` is blocked by `MaxGRESPerAccount`,
      `gpux`/`mgpu` are inactive or drained, and `gaosh`/`engram`/`test`
      reject the current account/partition combination. Test-only IDs
      `100671-100675` are absent from `squeue`; no replacement was submitted.
      Current `100319` remains the earliest aligned path. This is scheduling
      evidence only, not method evidence.
      Post-probe contract audit
      `contract_audit_after_resource_probe_20260603_0656` passed `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [ ] 2026-06-03 07:07 RGB-D slot training running:
      `100319` started on `server03` at `2026-06-03T07:02:39` with one node,
      four H200 GPUs, 32 CPUs, and four-hour walltime. Startup manifests
      confirm exact96 RGB-D H5 input, `MIN_TRAIN_SECONDS=10800`,
      RGB-D images plus robot qpos/qvel proprio, no hole/peg/TCP oracle state
      as input, object slots only as labels, and four ensemble members
      `member_0` through `member_3`. stdout has four
      `rgbd_slot_train_start` events with `num_paths=96`, `coord_conv=true`,
      `lazy_images=false`, cameras `base_camera` and `hand_camera`, and
      seeds `500-503`; stderr is empty at startup. `sstat` shows active step
      `100319.0` reading data. Current next gate is completed four-member
      artifacts followed by strict slot inspection `100411`; this is not yet
      RGB-D method evidence.
      Post-startup contract audit
      `contract_audit_after_slot100319_startup_20260603_0707` passed
      `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [ ] 2026-06-03 07:18 RGB-D slot training progress:
      `100319` remains running. stderr is empty; stdout has epoch/batch
      events for all four members; every member currently has `best_model.pt`
      and `manifest.json`; `sstat` reports active step `100319.0` with about
      `30.6GB` MaxRSS and `43.2GB` disk read. This is healthy running
      progress, not completion. The training script writes final `model.pt`,
      `metrics.json`, `metrics.md`, `training_history.json`, and
      `prediction_examples.json` only after the min-train-seconds loop exits.
      Therefore strict slot inspection `100411` remains the next evidence
      gate, and no RGB-D method claim is allowed yet.
      Post-progress contract audit
      `contract_audit_after_slot100319_running_progress_20260603_0718`
      passed `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [ ] 2026-06-03 07:27 RGB-D data and slot-training status:
      the full aggregate RGB-D dataset is no longer the blocker: the aggregate
      root contains exactly `96` `.rgbd.h5` files and the slot wrapper manifest
      records exact96 input, RGB-D images plus robot-proprio inputs, object
      slots as labels only, and `MIN_TRAIN_SECONDS=10800`. Job `100319` is
      running on `server03` with four H200 GPUs; runtime was `00:24:39` at the
      check. stderr is empty and `sstat` shows active training with about
      `30.6GB` MaxRSS and `43.2GB` disk read. Current parsed epoch diagnostics
      show `hole_pos_rmse_m` about `0.057-0.073` and
      `peg_head_hole_rmse_m` about `0.074-0.109`, so perception accuracy is a
      live risk. Do not interpret this as failure before the required training
      floor and strict inspections complete, and do not relax gates. Next
      authoritative evidence remains final member artifacts plus `100411`
      and exact96 predicted-slot inspection `100413`.
      Post-status contract audit
      `contract_audit_after_slot100319_0727_status_20260603_0730` passed
      `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 07:40 stronger slot risk mitigation, superseded:
      submitted a strictly same-objective backup branch
      `strongslot128_grid8_risk_mitigation_20260603_073418` because early
      `100319` geometry errors were still above the unchanged predicted-slot
      gate. This branch was later audited against actual Slurm submission
      records and found to have a submission mismatch: the manifest described
      the intended stronger slot capacity, but submitted job `100700` did not
      explicitly export all of those settings. It was therefore not kept as an
      active method path. Jobs `100700-100724` were canceled with zero elapsed
      time and no allocation, and the corrected branch
      `strongslot128_grid8_stratified_resubmit_20260603_0820` was submitted as
      jobs `100730-100754` with explicit strong-slot exports and stratified
      validation. This is recorded as a scheduling/submission-integrity repair,
      not method evidence and not a gate change.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 07:21 downstream submitted snapshot audit:
      captured `36` Slurm submitted batch snapshots in the chain directory and
      ran `bash -n` successfully. The submitted downstream contract still
      requires strict `100411` RGB-D slot inspection, exact96 predicted-slot
      export and inspection, predicted-slot quality gates
      (`MAX_HOLE_POS_RMSE_M=0.03`,
      `MAX_PEG_HEAD_HOLE_RMSE_M=0.035`,
      `MIN_BINARY_ACCURACY=0.95`), four-H200 RGB-D-derived world-model
      training with `MIN_TRAIN_SECONDS=10800`,
      `REQUIRE_RGBD_DERIVED=true` inspection/eval, `SLOT_SOURCE=rgbd`
      controller videos, `EXPECTED_SLOT_SOURCE=rgbd` controller inspection,
      and required nonblank video review. This prevents downstream drift but
      does not prove method success.
      Post-downstream-snapshot contract audit
      `contract_audit_after_downstream_snapshot_20260603_0721` passed
      `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 07:52 future RGB-D slot split fix:
      found and fixed a validation-split metadata issue while diagnosing live
      `100319` perception risk. The old fallback scenario parser used the
      aggregate filename too greedily, so validation splits could miss dynamic
      event categories. Future training now parses canonical scenario names
      and defaults to trajectory validation split stratified by scenario.
      Exact96 metadata validation confirms seeds `500-503` and `700` each
      receive `12` train and `4` val trajectories for all six categories.
      This is aligned because checkpoint selection should judge the RGB-D slot
      interface on every dynamic event before world-model task rebinding uses
      those slots. It does not change evaluation thresholds or allow oracle
      slots into the method path.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 07:59 predicted-slot scenario diagnostics:
      added diagnostic-only per-scenario oracle-error reporting to the RGB-D
      predicted-slot export and inspection scripts. This addresses the
      physical failure-localization problem: if RGB-D perception binds the
      task frame incorrectly, the next fix must know whether the dominant
      error comes from moving-hole cases, reverse motion, peg disturbance, or
      static cases. The existing global predicted-slot gate is unchanged:
      `quality_gate_passed` still depends only on the original global
      `hole_pos_rmse_m`, `peg_head_hole_rmse_m`, `peg_pos_rmse_m`, and
      `binary_accuracy` thresholds. Oracle slots remain inspection-only and
      the world model still reads RGB-D-derived `slots/*`.
      Python compile, small synthetic diagnostics, and both main/backup
      contract audits passed (`183/183`).
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 08:05 live slot-training progress triage:
      `100319` remains live training, not evidence: it is running on
      `server03`, stderr is empty, and all four members have only
      intermediate `best_model.pt` plus manifests, with no final
      `model.pt`/`metrics.json`/history/examples yet. Extended the read-only
      `triage_rgbd_slot_training_job.py` tool to parse stdout epoch events and
      summarize latest/min observed hole, peg-head-hole, and peg-position
      RMSEs by member. The generated
      `triage_live_progress_20260603_0804.md` parsed `2954` epoch events and
      shows live geometry risk remains high relative to downstream reference
      thresholds. This is diagnostic only: it does not make a pass/fail
      decision, does not change thresholds, and does not replace strict
      `100411/100413` inspections or RGB-D-derived downstream gates.
      Contract audits for main and strong backup chains passed `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 08:10 active validation-split triage:
      extended the read-only RGB-D slot triage tool to summarize canonical
      scenario counts in each active member's validation split. This is needed
      because `100319` was submitted before the canonical scenario parser and
      stratified validation fix; its live validation metrics should be read as
      old-split diagnostics, not balanced dynamic-event evidence. Generated
      `triage_live_progress_split_20260603_0809.md`, which shows `member_0`
      has no `hole_reverse` validation trajectory, while `member_1-3` cover
      all six scenarios but with imbalanced counts. This does not alter
      training, checkpoint selection, predicted-slot thresholds,
      `100411/100413`, RGB-D-derived world-model gates, controller metrics, or
      video evidence requirements. Main and backup contract audits passed
      `183/183`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 08:20 strong-slot backup submit mismatch repair:
      the first strong-slot backup submission was not auditable as the
      intended capacity branch. Its manifest claimed `CNN_CHANNELS=128`,
      `HIDDEN_DIM=512`, `SPATIAL_GRID_SIZE=8`, `DROPOUT=0.05`, and
      `BATCH_SIZE=96`, but the submitted `100700` job export lacked those
      values and its wrapper snapshot predated the
      `STRATIFY_VAL_BY_SCENARIO` export. This was classified as a
      submission/scheduling implementation issue, not RGB-D method evidence.
      Patched the submitter and contract audit so future chains must
      self-certify slot hyperparameters in `jobs.tsv` and `sacct SubmitLine`.
      Canceled old pending jobs `100700-100724` with zero elapsed time and no
      allocation, then submitted corrected chain
      `strongslot128_grid8_stratified_resubmit_20260603_0820` with jobs
      `100730-100754`. The corrected slot job `100730` explicitly exports
      exact96 input, `MIN_TRAIN_SECONDS=10800`, `CNN_CHANNELS=128`,
      `HIDDEN_DIM=512`, `SPATIAL_GRID_SIZE=8`, `DROPOUT=0.05`,
      `BATCH_SIZE=96`, `SEED=700`, and
      `STRATIFY_VAL_BY_SCENARIO=true`; downstream gates remain strict
      RGB-D-derived world-model/controller/video gates. Submitted snapshots
      were captured and syntax-checked; corrected chain contract audit passed
      `191/191`. Current `100730` forecast is `2026-06-09T04:00:00`, so the
      backup remains queued risk mitigation only.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 08:24 live slot-training recheck:
      `100319` is still formal RGB-D slot training, not method evidence. It is
      running on `server03` with about `01:21:28` elapsed, stderr empty, and
      no final member `model.pt`, `metrics.json`, history, or examples.
      Generated read-only diagnostic
      `triage_live_progress_split_20260603_0824.md/json`; it parsed `4026`
      epoch events and still shows live geometry risk above the unchanged
      downstream reference limits for hole and peg-head-hole errors. The
      diagnostic classification `incomplete_ensemble_members` reflects that
      the job has not completed the three-hour floor and final writeout; it is
      not a method failure. No gate or evaluation protocol changed, and no
      new heavy job was submitted.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 08:26 strong-slot backup diagnostics:
      added the same failure-localization coverage to corrected strong-slot
      backup chain `strongslot128_grid8_stratified_resubmit_20260603_0820`
      that the main chain already has. Submitted afterany CPU diagnostics
      `100757-100767` for slot inspection/triage, predicted-slot
      inspection/triage, RGB-D-derived world-model inspection/triage, and all
      five controller branches. They are dependency-pending with zero elapsed
      time and allocation. Captured `11` submitted snapshots and ran
      `bash -n`; corrected strong backup contract audit still passes
      `191/191`. These jobs do not change any gate, threshold, controller
      metric, video review, or evidence requirement.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 08:36 RGB-D slot label distribution audit:
      added reusable read-only audit
      `scripts/world_model/audit_rgbd_slot_label_distribution.py` and ran it
      on active `100319`, writing
      `label_distribution_audit_20260603_0830.md/json` under the job output
      directory. The physical purpose is to localize whether live slot errors
      are caused by label/split scale, a train-mean predictor, or an RGB-D
      perception generalization gap before downstream task-frame rebinding
      consumes the slots. The audit confirms exact96 labels are balanced
      across the six scenarios and quaternion targets are unit-normalized.
      Current live metrics beat train-mean label baselines but remain above
      the unchanged hole and peg-head-hole reference limits, so this is
      perception-risk localization only. It does not load RGB-D images, train,
      export slots, change thresholds, change `100411/100413`, or provide
      method evidence. Py compile passed; post-record contract audits still
      pass for the main chain (`183/183`) and corrected strong-slot backup
      (`191/191`).
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 08:39 live slot-training recheck after label audit:
      reran `triage_rgbd_slot_training_job.py` for active `100319`, writing
      `triage_live_progress_split_20260603_0839.md/json`. The job remains
      running on `server03` with stderr empty and no final member artifacts.
      The triage parsed `4898` epoch events; latest and minimum observed
      hole/peg-head-hole errors remain above unchanged reference limits. This
      confirms the current state is still live RGB-D perception-risk
      localization, not a completed slot result, not method evidence, and not
      a reason to change any gate.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 08:43 RGB-D slot label projection review:
      added `scripts/world_model/visualize_rgbd_slot_labels.py` and ran it on
      the exact96 RGB-D set. The output
      `visual_artifact_review/slot_label_projection_20260603_0842` contains a
      JSON/Markdown report and `rgbd_slot_label_projection_review.png`. I
      opened the contact sheet and checked the sampled overlays: base-camera
      hole/peg/TCP labels land on the expected target, peg, and end-effector
      image regions; hand-camera hole is mostly out of frame while peg/TCP
      remain visible, matching the projection rates. Warnings were `0`. This
      is a source-data/perception alignment diagnostic only. It does not
      train, export predicted slots, modify gates, or provide RGB-D method
      evidence, but it makes a gross camera/label projection bug less likely
      as the explanation for the current live `100319` slot errors.
      Post-record contract audits still pass for the main chain (`183/183`)
      and corrected strong-slot backup (`191/191`).
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 08:50 RGB-D slot training curve audit:
      added `scripts/world_model/audit_rgbd_slot_training_curves.py` and ran
      it on live `100319` stdout, writing
      `training_curve_audit_20260603_0848.md/json`. It parsed `5521` epoch
      events. Train loss keeps falling to roughly `0.3%` of the first parsed
      epoch, but every member's best-observed hole and peg-head-in-hole-frame
      errors remain above the unchanged reference limits; members `1` and `2`
      also show recent validation geometry worse than their best. This is
      failure-localization evidence for RGB-D slot generalization/capacity
      risk only. It does not alter checkpoint selection, thresholds, exports,
      downstream gates, or method evidence requirements.
      Post-record contract audits still pass for the main chain (`183/183`)
      and corrected strong-slot backup (`191/191`).
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 08:53 main-chain diagnostic coverage recheck:
      before adding more queue entries, rechecked the active main chain. It
      already has afterany slot diagnostics (`100441`, `100588`),
      predicted-slot export/inspection diagnostics (`100535`, `100597`),
      RGB-D-derived world-model diagnostics (`100445`, `100590`), controller
      branch triage (`100591-100595`), predicted-slot visual review
      (`100433`), and final evidence review (`100552`). `100319` is still
      running and has no final member artifacts, so the current missing
      evidence is strict downstream completion rather than missing diagnostics.
      No new Slurm job was submitted; this avoids duplicate queue pollution
      while preserving the same RGB-D method/evaluation contract.
      Post-record contract audits still pass for the main chain (`183/183`)
      and corrected strong-slot backup (`191/191`).
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 08:56 checkpoint contract audit tightened:
      patched `scripts/world_model/audit_rgbd_method_chain_contract.py` to
      make checkpoint semantics explicit. The audit now checks that
      predicted-slot export uses `SLOT_CHECKPOINT=best_model.pt` or the
      wrapper default, world-model eval uses `CHECKPOINT=best_model.pt` or
      default, and every RGB-D controller branch uses
      `RGBD_SLOT_CHECKPOINT=best_model.pt` and
      `WORLD_MODEL_CHECKPOINT=best_model.pt` or defaults. This is needed
      because live curves show validation geometry can regress from the best
      checkpoint while training continues. This change only strengthens
      readiness auditing; it does not change training, checkpoint selection,
      thresholds, exports, controller behavior, or method evidence. Py compile
      passed; main chain audit passes `195/195`, corrected strong-slot backup
      passes `203/203`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 09:01 live slot-training triage refresh:
      reran `triage_rgbd_slot_training_job.py` for active `100319`, writing
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_split_20260603_0901.md/json`.
      `100319` remains running on `server03` with stderr empty and no final
      member `model.pt`/`metrics.json`; strict `100411/100412/100413/100415`
      remain dependency-pending. The triage parsed `6159` live epoch events.
      Latest hole/peg-head-hole errors are still above the unchanged
      reference limits for all members, and best-observed values remain above
      those limits as well. This is RGB-D perception-risk localization only
      while the job is incomplete; it does not provide method evidence, does
      not classify a final failure, and does not justify changing any gate.
      No new Slurm job was submitted and no active job was canceled.
      Post-record contract audits still pass for the main chain (`195/195`)
      and corrected strong-slot backup (`203/203`).
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 09:05 downstream handoff gate audit:
      re-read active TODO/PLAN and audited the handoff from active slot job
      `100319` into submitted downstream jobs. `100411` still requires
      compliant exact96 RGB-D slot-training evidence and is not a slot-quality
      success claim by itself. `100412` still exports RGB-D-derived `slots/*`
      from wrapper-default `best_model.pt` and keeps `oracle_slots/*`
      inspection-only. `100413` still enforces the unchanged global
      predicted-slot task-frame quality gates before world-model training;
      per-scenario diagnostics are localization only. `100415` still refuses
      non-exact predicted-slot inputs, requires `96/96` files, one-node
      `4xNVIDIAH200`, `MIN_TRAIN_SECONDS>=10800`, and RGB-D prediction
      uncertainty/probability datasets. No new Slurm job was submitted, no
      active job was canceled, and no gate changed. Post-record contract
      audits still pass for the main chain (`195/195`) and corrected
      strong-slot backup (`203/203`).
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 09:08 live slot triage and backup submission recheck:
      refreshed Slurm/artifact state and generated
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_split_20260603_0908.md/json`.
      `100319` is still running on `server03`, stderr is empty, `sstat`
      reports active 4-task training with stable memory, and no final
      `model.pt`/`metrics.json` artifacts exist. The triage parsed `6583`
      live epoch events and still shows best-observed hole and
      peg-head-in-hole errors above unchanged reference limits for all four
      members. Rechecked `100730`: it is pending priority with null
      Req/ExcNodeList, one-node 4xH200 request, exact96 input, strong-slot
      hyperparameter exports, stratified validation, and `MIN_TRAIN_SECONDS=10800`.
      No new Slurm job was submitted, no active job was canceled, and no
      threshold/gate/controller protocol changed. Post-record contract audits
      still pass for the main chain (`195/195`) and corrected strong-slot
      backup (`203/203`).
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 09:16 strong-slot backup replacement probe refused by real
      scheduling:
      because `sbatch --test-only` probes under
      `strongslot128_grid8_stratified_resubmit_20260603_0820/resource_probe_20260603_0912`
      forecast a legal `cpu` 4H200 slot allocation at
      `2026-06-04T10:22:39`, submitted one complete replacement chain
      `100780-100804` under
      `strongslot128_grid8_stratified_cpu_replacement_20260603_0917` rather
      than a partial slot job. The replacement preserved exact96 RGB-D inputs,
      `MIN_TRAIN_SECONDS=10800`, strong-slot hyperparameters, stratified
      validation, unchanged predicted-slot gates, RGB-D-derived world-model
      training/inspection/eval, `SLOT_SOURCE=rgbd` controller video branches,
      and required video review. Its contract audit passed `203/203` and
      `25` submitted snapshots were saved. Real submitted `scontrol` then set
      slot job `100780` to `2026-06-09T03:00:00`, later than existing backup
      `100730` at `2026-06-09T01:00:00`, so replacement jobs `100780-100804`
      were canceled before allocation. `sacct` confirms all replacement jobs
      had `Elapsed=00:00:00`, `AllocNodes=0`, and `Start=None`. Existing
      backup `100730-100754` plus diagnostics `100757-100767` remain active;
      post-cancel audits pass for the main chain (`195/195`) and corrected
      backup (`203/203`). This is scheduling evidence only and does not change
      RGB-D evidence requirements.
- [x] 2026-06-03 09:22 live main-slot triage:
      with `100319` still running on `server03`, performed a read-only
      triage
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_split_20260603_0922.md/json`.
      The job has empty stderr, active 4-task training, no final
      `model.pt`/`metrics.json` artifacts, and strict downstream gates are
      still dependency-pending. The triage parsed `7360` epoch events and
      reports only `incomplete_ensemble_members`; latest live slot geometry
      remains above unchanged task-frame reference limits for hole and
      peg-head-in-hole. No new Slurm job was submitted and no active job was
      canceled. Main/backup contract audits still pass (`195/195` and
      `203/203`). This is perception-risk localization while training is
      incomplete, not RGB-D method evidence and not a threshold change.
- [x] 2026-06-03 09:25 downstream coverage recheck while waiting:
      `100319` remains running with no final slot artifacts, so checked
      submitted downstream coverage rather than adding queue churn. Main chain
      still has strict slot inspection, predicted-slot export/inspection,
      RGB-D-derived world-model train/inspect/eval, sensitivity, predicted
      slot visual review, controller video/inspection/video-review branches,
      afterany diagnostics, and final evidence review. Strong backup still has
      the same strict path plus diagnostics. No new Slurm job was submitted,
      no active job was canceled, and no gate changed. Contract audits pass
      for main and backup (`195/195`, `203/203`). This is readiness evidence
      only; method evidence still requires completed RGB-D-derived slots,
      world model, controller metrics, and inspected video.
- [x] 2026-06-03 09:28 final evidence/video review smoke:
      checked the final evidence path while no controller results exist.
      `inspect_rebinding_controller_run.py` requires RGB-D controller input
      evidence, separate metric slots, final-state H5 metric consistency, and
      dynamic event before success. `inspect_video_artifacts.py` creates
      readable nonblank review sheets without judging semantics.
      `review_rgbd_method_evidence.py` aggregates those reports and never
      allows a method success claim by itself. A `.venv` no-result smoke
      exited `65` with zero candidate branches because controller/video
      inspections are missing. No code, job, threshold, or gate changed; this
      is readiness evidence that metric-only or missing-video success is
      refused.
- [x] 2026-06-03 09:34 predicted-slot to RGB-D-derived world-model boundary
      audit:
      re-read the active TODO and RGB-D/world-model plans before checking the
      handoff. `100319` is still running with empty stderr and no final slot
      artifacts, so this is an interface-readiness check rather than result
      interpretation. The export path writes RGB-D slot-extractor ensemble
      predictions to `slots/*`, copies simulator labels only to
      `oracle_slots/*` for inspection, records boundary attrs, requires frame
      alignment, and reports oracle error only as quality-gate/diagnostic
      evidence. Predicted-slot inspection requires exact files, report
      presence, unchanged global quality gates, and RGB-D prediction
      uncertainty/probability datasets. `object_slot_dataset.py` builds
      features and targets from `group["slots"]`, records
      `oracle_slots_read=false`, and the world-model train/eval/inspection
      path refuses incomplete RGB-D-derived evidence. The 4H200/3h training
      wrapper validates exact96 predicted-slot files and required RGB-D
      prediction auxiliary datasets before launching members. Local syntax
      checks passed and post-audit contract audits pass for main/backup
      (`195/195`, `203/203`). No code, job, gate, or protocol changed; no
      RGB-D method evidence exists yet.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 09:36 live slot-training triage refresh:
      ran read-only `triage_rgbd_slot_training_job.py` for active `100319`,
      writing
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_split_20260603_0936.md/json`.
      The job is still running on `server03` with 4xH200, exact96 input,
      empty stderr, and no final member artifacts. The triage parsed `8184`
      epoch events and still reports only `incomplete_ensemble_members`.
      Latest hole and peg-head-in-hole errors remain above unchanged
      reference limits for all four members; peg RMSE is below the reference
      only for member_1. Member_0's validation split misses the
      `hole_reverse` scenario, so the existing stratified strong backup remains
      the aligned hedge rather than a reason to reinterpret this live job.
      This is live perception-risk localization only and no gate or protocol
      changed. Post-record contract audits pass for main/backup (`195/195`,
      `203/203`) at
      `contract_audit_after_0936_live_triage_20260603_0938.md/json`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 09:41 live artifact health check:
      refreshed Slurm and artifact state while `100319` was still running. A
      first `find` traversal reported `member_2` missing, but immediate direct
      checks showed `member_2` exists with `member_manifest.txt`,
      `manifest.json`, and `best_model.pt`, and stdout continues to report
      live `member_2` training. All four members have `best_model.pt`; none
      has final `model.pt` or `metrics.json`. `sstat` reports active 4-task
      training and stderr is empty. This is classified as a transient
      directory-traversal/filesystem-read anomaly, not a training failure or
      method evidence. No Slurm job, gate, threshold, or protocol changed.
      Post-record contract audits pass for main/backup (`195/195`, `203/203`)
      at `contract_audit_after_0941_artifact_health_20260603_0943.md/json`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 09:45 live slot-training triage refresh:
      ran read-only `triage_rgbd_slot_training_job.py` for active `100319`,
      writing
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_split_20260603_0945.md/json`.
      The job is still running on `server03` with exact96 RGB-D input,
      4xH200 allocation, empty stderr, active 4-task `sstat`, and no final
      member `model.pt` or `metrics.json`; downstream jobs remain
      dependency-pending. The triage parsed `8698` epoch events and reports
      only `incomplete_ensemble_members`. Latest hole and peg-head-in-hole
      errors remain above unchanged reference limits for all four members;
      peg RMSE is below reference only for member_1. Minimum observed hole and
      peg-head-in-hole errors remain above reference limits for every member.
      Member_0's validation split still misses `hole_reverse`, so the existing
      stratified strong backup remains the aligned hedge. This is live
      perception-risk localization only and no gate, threshold, job, or
      protocol changed. Post-record contract audits pass for main/backup
      (`195/195`, `203/203`) at
      `contract_audit_after_0945_live_triage_20260603_0946.md/json`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 09:50 dependency coverage and running-health recheck:
      re-read `AGENTS.md`, active/focused TODOs, and RGB-D/world-model/
      controller plans before interpreting the live state. `100319` remains
      running on `server03` with exact96 RGB-D input, 4xH200 allocation,
      empty stderr, active 4-task `sstat`, and no final member `model.pt` or
      `metrics.json`. Submitted snapshots confirm strict `100411` remains
      `afterok:100319`, diagnostics `100441` and `100588` remain
      `afterany:100319`, and final review `100552` still depends on afterany
      controller/video inspections. The current missing evidence is completed
      slot artifacts and strict downstream results, not missing diagnostics.
      No duplicate Slurm job was submitted and no gate, threshold, or protocol
      changed. Post-record contract audits pass for main/backup (`195/195`,
      `203/203`) at
      `contract_audit_after_0950_dependency_coverage_20260603_0951.md/json`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 09:55 live slot-training triage refresh:
      ran read-only `triage_rgbd_slot_training_job.py` for active `100319`,
      writing
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_split_20260603_0955.md/json`.
      `100319` remains running on `server03` with exact96 RGB-D input,
      4xH200 allocation, empty stderr, and no final member `model.pt` or
      `metrics.json`; downstream jobs remain dependency-pending. The triage
      parsed `9237` epoch events and reports only
      `incomplete_ensemble_members`. Latest and minimum-observed hole and
      peg-head-in-hole errors remain above unchanged reference limits for all
      four members; peg RMSE is below reference only for member_1. Member_0's
      validation split still misses `hole_reverse`, so the existing
      stratified strong backup remains the aligned hedge. This is live
      perception-risk localization only and no gate, threshold, job, or
      protocol changed. Post-record contract audits pass for main/backup
      (`195/195`, `203/203`) at
      `contract_audit_after_0955_live_triage_20260603_0956.md/json`; a
      misplaced main audit JSON was removed and generated
      `scripts/world_model/__pycache__` was cleaned.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 10:03 3h training-floor boundary check:
      waited until active `100319` crossed `MIN_TRAIN_SECONDS=10800`, then
      refreshed Slurm/log/artifact state. The job was still running on
      `server03` at `03:01:16/04:00:00`, with exact96 RGB-D input, 4xH200
      allocation, empty stderr, active 4-task `sstat`, and no final member
      `model.pt` or `metrics.json`; strict `100411` and afterany diagnostics
      `100441/100588` remained dependency-pending. This is training-floor
      compliance and running-health evidence only. It does not prove slot
      quality, predicted-slot export, RGB-D-derived world-model/controller
      behavior, video evidence, or method success/failure. No job, gate,
      threshold, or protocol changed. Post-record contract audits pass for
      main/backup (`195/195`, `203/203`) at
      `contract_audit_after_3h_floor_health_20260603_1004.md/json`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 10:10 post-floor live slot-training triage:
      ran read-only `triage_rgbd_slot_training_job.py` for active `100319`,
      writing
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_split_20260603_1010.md/json`.
      The job remains running on `server03` after the 3h floor with exact96
      RGB-D input, 4xH200 allocation, empty stderr, active 4-task training,
      all four `best_model.pt` files, and no final member `model.pt` or
      `metrics.json`. The triage parsed `10072` epoch events and still
      reports only `incomplete_ensemble_members`. Latest and minimum-observed
      hole and peg-head-in-hole errors remain above unchanged reference
      limits for every member; peg RMSE is below reference only for member_1.
      Member_0 validation still misses `hole_reverse`, so the existing
      stratified strong backup remains the aligned hedge. This is live
      perception-risk localization only and no gate, threshold, job, or
      protocol changed. Contract audits pass for main/backup (`195/195`,
      `203/203`) at
      `contract_audit_after_1010_live_triage_20260603_1012.md/json`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 10:21 slot completion and predicted-slot export failure:
      `100319` completed normally on `server03` at
      `2026-06-03T10:14:12+08:00` after `03:11:33`; all four members wrote
      final `model.pt` and `metrics.json`, and `100411/100441` slot training
      inspections completed. This proves exact96 RGB-D slot-training
      compliance, not method success: final aggregate slot geometry remained
      high (`hole_pos_rmse_m_mean=0.0686`,
      `peg_head_hole_rmse_m_mean=0.0877`, `peg_pos_rmse_m_mean=0.0498`).
      Predicted-slot export `100412` failed before writing any predicted-slot
      H5 files because importing `train_rgbd_slot_extractor.py` hit a Python
      `SyntaxError`; local `py_compile` and import checks now pass, so this is
      classified as export/import implementation runtime failure. Diagnostic
      jobs `100535/100597` confirmed zero predicted-slot H5 files, and dead
      old pending jobs `100413-100432`, `100445`, `100552`, and
      `100590-100595` were canceled. Added scheduling-only
      `PRED_EXPORT_PARTITION` support to the chain submitter, with no change
      to quality gates, thresholds, data, controller policy, or evidence
      rules. Submitted replacement chain
      `coordconv_downstream_after_slot100319_exportfix_20260603_1025`
      (`100847-100870`); `100847` completed, `100848` is pending on `cpu`
      after being moved off unavailable `debug`, and the replacement contract
      audit passed `203/203`. There is still no RGB-D method evidence.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 10:26 CPU-only export replacement correction:
      because the first replacement export `100848` had no useful start time
      after `debug` became unusable and GPU-backed `cpu` forecast later than
      the strong backup, canceled pending `100848-100870` and kept completed
      `100847` only as scheduling history. Added CPU-only export controls to
      the submitter/export manifest (`PRED_EXPORT_GRES=none`,
      `PRED_EXPORT_CUDA_EXPORT=false`, `REQUESTED_GRES_LABEL=none`) so
      predicted-slot export can run as Slurm compute inference without a GPU.
      This does not change RGB-D inputs, slot checkpoints, frame alignment,
      quality thresholds, downstream world-model gates, controller metrics, or
      video review. Submitted active replacement chain
      `coordconv_downstream_after_slot100319_exportfix_cpuonly_20260603_1028`
      (`100876-100899`); `100876` completed, `100877` is CPU-only export
      pending on `cpu`, and the chain contract audit passed `203/203`.
      CPU-only probes found no earlier legal shape than about
      `2026-06-04T09:46:28`. No predicted-slot or method evidence exists yet.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 10:42 predicted-slot export/inspection passed:
      CPU-only RGB-D slot export `100877` completed on `server54` with exact
      `96/96` predicted-slot H5 files and `28896` frames. Strict inspection
      `100878` passed unchanged gates with `0` warnings:
      `hole_pos_rmse_m=0.02067`, `peg_head_hole_rmse_m=0.02801`,
      `peg_pos_rmse_m=0.01485`, `binary_accuracy=0.99772`. Visual review
      `100898` passed and the contact sheet was opened directly; it is
      nonblank/readable and shows RGB-D frames with predicted/oracle slot
      overlays. Sensitivity job `100879` completed and records diagnostic
      controller-risk flips, including aggregate `insert_yz_gate` mismatch
      `0.05177` and `hole_move_stop` insert-gate mismatch `0.11524`; this
      does not change any controller gate or success metric. Downstream
      4H200/4h RGB-D-derived world-model training `100880` is now
      dependency-free and pending priority with forecast
      `2026-06-04T14:00:00`, while contract audit after predicted-slot pass
      remains `203/203`. This is RGB-D-derived slot handoff evidence, not
      world-model, controller, video, or method-success evidence.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 10:49 world-model queue correction:
      original post-predicted-slot world-model tail `100880` was canceled
      before allocation after its forecast moved to `2026-06-08T01:00:00`.
      Added submitter and audit support for reusing already passed
      predicted-slot artifacts/jobs (`100877/100878/100879/100898`) without
      rerunning export or changing gates. A replacement tail `100935-100953`
      was submitted and audited cleanly (`203/203`), but it forecast
      `2026-06-08T03:00:00` while strong backup `100730` moved earlier to
      `2026-06-03T11:57:34`; therefore `100935-100953` was canceled before
      allocation as queue pollution. Completed predicted-slot evidence remains
      under job `100877`; current earliest aligned RGB-D path is the strong
      backup chain starting at `100730`, and its post-correction contract
      audit passed `203/203`. No world-model, controller, video, or
      method-success evidence exists yet.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 11:10 strong-backup slot-training live health:
      `100730` started at `2026-06-03T10:50:35+08:00` on `server13` with
      exact96 RGB-D input, 4xH200 allocation, and the strong slot config
      (`CNN_CHANNELS=128`, `HIDDEN_DIM=512`, `SPATIAL_GRID_SIZE=8`,
      `BATCH_SIZE=96`, stratified validation). Startup `nvidia-smi` reported
      `Unable to determine the device handle for GPU1`, but the 4-task Slurm
      step remains running and all four members have written `best_model.pt`.
      Read-only triage at `11:10` classifies only
      `incomplete_ensemble_members`; members 0/2/3 are progressing quickly
      while member 1 is a slow straggler. A non-executing replacement probe
      with `--exclude=server13` starts only at `2026-06-04T10:36:40`, so the
      aligned action is to keep the running same-day job and monitor for
      concrete CUDA/runtime failure. Live metrics are diagnostic only and
      remain above unchanged slot reference limits; no final slot,
      world-model, controller, video, or method-success evidence exists yet.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 11:26 strong-backup straggler triage:
      read-only triage for `100730` wrote
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100730/triage_live_progress_split_20260603_1125.md/json`.
      The job is still running on `server13` with exact96 RGB-D input,
      4xH200 allocation, `EPOCHS=50`, `MIN_TRAIN_SECONDS=10800`, and only
      the startup GPU1 handle warning in stderr. Members 0/2/3 are far
      beyond epoch 300; member 1 is slow but has advanced to epoch 4 and has
      written `best_model.pt`. Same-shape replacement probes are later:
      `2026-06-04T10:32:10` if `server13` is allowed and
      `2026-06-04T14:38:13` with `--exclude=server13`. Therefore this is a
      hardware/performance risk to monitor, not a concrete failure or a
      reason to cancel the running same-day aligned path. Live errors remain
      diagnostic only; no RGB-D world-model/controller/video/method evidence
      exists yet.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 11:50 strong-backup still-live triage:
      `100730` is still running after about one hour on `server13`.
      Read-only triage wrote
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100730/triage_live_progress_split_20260603_1150.md/json`
      and still classifies only `incomplete_ensemble_members`. Members
      0/2/3 are beyond epoch `640`; member 1 is slow but reached epoch `10`
      and updated `best_model.pt` at `11:43:59`. Stderr still has only the
      startup GPU1 handle warning. There are no final `model.pt` or
      `metrics.json` files yet, and strict slot inspection `100731` remains
      dependency-pending. This is live scheduling/perception-risk evidence
      only; no gate, threshold, controller condition, queue branch, or method
      claim changed.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [x] 2026-06-03 12:15 strong-backup downstream tail scheduling fix:
      the pending strong-backup downstream tail still had GPU predicted-slot
      export `100732` and GPU-only world-model training `100735`. Non-running
      probes showed a faster legal same-objective path: CPU-only export
      `2026-06-04T11:20:08` versus GPU export `2026-06-06T10:03:04`, and
      4H200 world-model on `cpu,gpu` `2026-06-04T12:25:04` versus `gpu`
      `2026-06-06T10:03:04`. Submitted replacement chain
      `strongslot128_grid8_stratified_cpu_export_wmcpugpu_tail_20260603_1215`
      reusing active slot job `100730` and strict slot inspection `100731`.
      New active downstream jobs are CPU-only predicted-slot export `101009`,
      strict export inspection `101010`, diagnostic sensitivity `101011`,
      4H200 RGB-D-derived world-model `101012`, inspection/eval
      `101013/101014`, controller/video/review jobs `101015-101031`. Audit
      `contract_audit_after_submit_20260603_1215.md/json` passed `203/203`.
      Old pending tail `100732-100754` and old afterany diagnostics
      `100758-100767` were canceled with zero allocation; slot inspection
      `100731` and slot afterany diagnostics `100757/100760` remain. This
      changes only scheduling resources for inference/training placement, not
      RGB-D inputs, slot checkpoints, thresholds, final metrics, controller
      policy, or video requirements.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- [ ] 2026-06-03 18:30 RGB-D online controller-slot failure after
      `101012-101031`: strong slot extractor `100730`, predicted-slot export/
      inspection `101009/101010`, visual slot review `101030`, and
      RGB-D-derived world-model training/inspection/eval `101012/101013/101014`
      are valid component evidence, but RGB-D controller/video branches
      `101015`, `101018`, `101021`, `101024`, and `101027` all failed final
      dynamic success; final review `101031` correctly rejected method evidence
      with `success_after_dynamic_event_count=0`. Direct video review matched
      the metrics. H5 analysis showed online controller `slots` diverge from
      `metric_slots`, especially peg/TCP and `peg_head_at_hole`, while online
      RGB normalization matched training/export. Default-off online diagnostics
      completed for all five branches (`101580`, `101602`, `101603`, `101604`,
      replacement `101618`; first move-stop attempt `101601` was a
      `server60` DeviceLost rendering failure only). Direct contact-sheet
      inspection showed valid/nonblank controller-time RGB-D frames, but large
      post-perturb peg/TCP slot errors, so the failure class is slot-model
      bridge-state distribution coverage rather than a blank/stale image
      contract. Converted the five diagnostics into a standard RGB-D slot
      dataset (`5` files, `1505` frames, warnings `0`) and combined with full96
      into exact `101` H5s. Augmented 4H200 slot job `101635` started on
      `server13` with all four members loading the exact `101` H5s, but
      `member_3` was far slower than members `0/1/2`; the job and its
      downstream chain (`101638`, `101641-101646`, `101652-101667`) were
      canceled to avoid a likely timeout/dead dependency tail. `sacct` records
      `101635` as `CANCELLED by 2059`, elapsed `00:29:18`. This is
      scheduling/device-risk evidence only, not perception quality evidence.
      Current active path is backup chain `101697-101720`, excluding `server13`
      for slot/world-model training, with the same exact `101` H5s, gates,
      4H200/3h training requirements, RGB-D-derived world-model requirement,
      controller settings, and video review; contract audit passed `138/138`.
      At `2026-06-03T19:13+08:00`, `101697` is pending with forecast
      `2026-06-03T19:54:28` on `server34`. At `19:19`, non-submitting
      same-data/same-exclusion 4H200 probes for `03:45:00` and `04:00:00`
      both forecast `2026-06-03T22:39:28`, so no later duplicate was
      submitted. Do not change gates or replace this with oracle/state control;
      inspect slot/world-model artifacts and then videos before making any
      method claim. Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_controller_online_slot_failure_1830.md`.
- [ ] 2026-06-03 19:38 full1000 RGB-D generation branch:
      User correction: exact96/full96 and full96-plus-online5 are only
      small-scope validation; complete experiments require exact `1000` RGB-D
      demos. Added exact-spec state rollout wrapper
      `scripts/slurm/collect_dynamic_state_rollouts_specs_4gpu.sbatch` and
      dependency/visual-review support in
      `scripts/slurm/submit_rgbd_distributed_shards.sh`. Submitted exact
      `1000` dynamic state source job `101832` with specs
      `none:700000:166`, `hole_move_stop:701000:167`,
      `hole_constant:702000:167`, `hole_reverse:703000:167`,
      `peg_disturb:704000:166`, `peg_drop:705000:167`. Submitted dependent
      RGB-D render shards `101833-101840`, exact structural inspection `101841`,
      and visual review `101842` under
      `experiments/world_model_task_rebinding/rgbd_dynamic_distributed/from_rollout_dir/full1000_rgbd_from_state101832_8x4_exrendernodes_20260603_1938`.
      Render shards require `afterok:101832`, `EXPECTED_RGBD_FILES=1000`,
      `REQUIRE_NO_WARNINGS=true`, and visual review samples `24` files x `3`
      frames. At submit check, `101832` was pending; render/inspection/visual
      jobs were dependency-pending. `101832` later ran on `server04` from
      `2026-06-03T19:47:29` to `21:28:19`, completed `0:0`, and direct HDF5
      reading verified exact per-scenario counts
      `166/167/167/167/166/167`, total `1000`. This completes only the state
      source gate; RGB-D data are still incomplete until render shards
      `101833-101840`, exact structural inspection `101841`, and direct visual
      review `101842` pass. This branch generates the full data only; do not
      treat it as method evidence until exact inspection and direct visual
      review pass. Canceled the full96-plus-online5 small-chain
      `101697-101720` after the user correction; `101697` had run `00:20:47`
      on `server04` with no `rgbd_slot_dataset_loaded` event, and downstream
      jobs were canceled before allocation. This frees resources for full1000
      and does not count as perception, world-model, controller, or method
      evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_full1000_rgbd_generation_submission.md`.
- [ ] 2026-06-03 22:02 full1000 render shard0 server04 repair:
      Render shard `101833` on `server04` showed a concrete SAPIEN/Vulkan
      `ErrorDeviceLost` failure after about `30` minutes: direct accounting
      found `0/125` successful RGB-D H5 files and `14` failed units before
      cancellation. This is a rendering/node failure, not an evaluation or
      data-objective change. Canceled `101833` and old dependency gates
      `101841/101842`, then submitted disjoint shard0 replacement `102001`
      with `RGBD_JOB_SHARD_INDEX=0`, `RGBD_NUM_JOB_SHARDS=8`,
      `1 node x 2 GPU`, and job-local exclusion
      `server04,server10,server13,server21,server28,server55,server58,server60`.
      Submitted replacement structural gate `102002` after
      `101834:101835:101836:101837:101838:101839:101840:102001`, and visual
      review `102003` after `102002`. Full1000 RGB-D remains incomplete until
      the repaired render, exact structural inspection, and direct visual
      review pass. Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_full1000_rgbd_generation_submission.md`.
- [ ] 2026-06-03 22:24 full1000 shard7 failed62 repair:
      Render shard `101840` on `server53` completed only `63/125` units and
      failed `62` units with SAPIEN/Vulkan `ErrorDeviceLost`, exiting `66:0`.
      This is a rendering/node failure. A full shard7 rerun would duplicate the
      valid `63` H5 files and break exact `1000`, so generated failed-unit-only
      worklist
      `experiments/world_model_task_rebinding/rgbd_dynamic_distributed/from_rollout_dir/full1000_rgbd_from_state101832_8x4_exrendernodes_20260603_1938/repair_worklists/shard7_failed62_after101840_20260603_222230/input_worklist.tsv`
      with `62` unique units and zero overlap with shard7 successes. Patched
      `scripts/slurm/render_dynamic_rgbd_dataset_dense.sbatch` to aggregate
      success/failed unit ledgers and refuse failed or incomplete dense repair
      jobs. Submitted repair `102032` (`1 node x 4 GPU`) excluding
      `server04,server10,server13,server21,server28,server53,server55,server58,server60`.
      Canceled gates `102002/102003` before allocation and submitted exact
      structural gate `102033` plus visual gate `102034` after render jobs
      `101834-101840`, shard0 repair `102001`, and shard7 repair `102032`.
      `102001` completed `125/125`, `102032` completed `62/62`, `102033`
      passed exact `1000` RGB-D files with `301000` frames and `0` warnings,
      `102034` passed visual review on `24` sampled files x `3` frames x `2`
      cameras with `0` warnings, and the contact sheet was opened directly and
      found nonblank with visible robot/table/peg/hole content. Local unit
      audit found expected `1000`, actual `1000`, missing `0`, extra `0`,
      duplicates `0`. This is full1000 RGB-D data evidence only, not method
      success. Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_full1000_rgbd_generation_submission.md`.
- [ ] 2026-06-03 23:17 full1000 method-chain replacement after wrapper bug:
      Submitted full1000 RGB-D method chain
      `full1000_coordconv_rgbd_method_20260603_231525`, but slot job
      `102144` failed before training with `srun: Argument list too long`.
      Log inspection shows this was caused by exporting all `1000` long RGB-D
      H5 paths into the `srun` environment. This is an implementation wrapper
      failure and does not count as slot, world-model, controller, physics, or
      evaluation evidence. Fixed the aligned path by passing path-list files
      into slot/world-model member tasks, adding stream-per-file predicted-slot
      export, bounding lazy H5 open-file cache, and making the method-chain
      audit count distributed RGB-D roots recursively. Canceled dead branch
      `102145-102168`, then submitted replacement chain
      `full1000_coordconv_rgbd_method_pathfilefix_20260603_231732`:
      `102169` slot training, `102170` slot inspection, `102171-102173` and
      `102192` predicted-slot export/inspection/diagnostics/visual review,
      `102174-102176` RGB-D-derived world model, `102177-102191` controller
      and video branches, and `102193` evidence review. Contract audit passed
      `206/206`. `102169` started on `server34` with exact `1000` RGB-D H5
      paths, `LAZY_IMAGES=true`, `EPOCHS=1`, 4 H200 GPUs, and
      `MIN_TRAIN_SECONDS=10800`. It loaded full1000 metadata (`301000`
      samples, `[301000, 8, 128, 128]` RGB-D shape) in about `591` seconds
      and all four members emitted epoch-0 batch-0 training events. This is
      startup/readiness evidence only; do not claim full1000 method success
      until all downstream RGB-D-derived gates and direct video/contact-sheet
      inspection complete. By batch `100`, `102169` was progressing but the
      4h walltime looked tight for full-epoch validation/checkpoint. Direct
      walltime extension was denied, so submitted non-concurrent 6h backup
      chain
      `full1000_coordconv_rgbd_method_slot6h_afternotok102169_20260603_233525`
      with slot job `102292` on dependency `afternotok:102169`; audit passed
      `206/206`, and it is dependency-pending without consuming GPUs. If
      `102169` succeeds, cancel backup branch `102292-102316` as unneeded.
      Downstream preflight found predicted-slot export still expanded all
      full1000 RGB-D paths as Python argv; patched the export script/wrapper
      to use `--rgbd-paths-file` and the existing `rgbd_input_h5s.txt`, with
      `py_compile`, `--help`, `bash -n`, and contract audit passing. This
      preserves data, labels, stream-per-file export, metrics, and gates. At
      `2026-06-04T01:36:54+08:00`, `102169` remained running on `server34` at
      about `02:18:44/04:00:00` with all four members at epoch-0 batch `1750`
      but no epoch/checkpoint artifact yet; the long epoch-end gap is
      consistent with full1000 validation and increases walltime risk. Updated
      pending backup slot job `102292` from `06:00:00` to `08:00:00`, still
      on `afternotok:102169(unfulfilled)` and not consuming GPUs. This is
      scheduling slack only for the same exact `1000` RGB-D chain, not a
      dataset, gate, metric, or method change.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_full1000_rgbd_generation_submission.md`.
- [ ] 2026-06-04 01:56 full1000 visualization-budget correction:
      render/visual demo artifact generation is limited to `10` diagnostic
      samples per the user instruction, but the method chain remains exact
      `1000` RGB-D for slot training, predicted-slot export, world-model
      training, controller inputs, and strict gates. Patched the chain
      submitter defaults to `SLOT_VIS_SAMPLE_FILES=10`,
      `SLOT_VIS_FRAMES_PER_FILE=4`, and `VIDEO_REVIEW_SAMPLE_COUNT=10`, with
      an explicit manifest boundary that this changes only diagnostic rendered
      artifacts. Canceled pending old visual-only jobs
      `102179/102182/102185/102188/102191/102192/102193` and submitted
      replacements `102464-102470`; `102464` is the 10-file predicted-slot
      visual review, `102465-102469` are the 10-sample controller-video
      artifact reviews, and `102470` is the evidence review depending on the
      unchanged controller metric jobs plus the new video review jobs.
      Contract audit
      `experiments/world_model_task_rebinding/rgbd_method_chains/full1000_coordconv_rgbd_method_pathfilefix_20260603_231732/contract_audit_after_visual10_replacement.md`
      passed `206/206`, confirming the exact1000 RGB-D method contract was not
      weakened. Live slot job `102169` is still running on exact1000; it has
      produced four epoch-0 `best_model.pt` checkpoints and entered epoch 1.
      Walltime extension on the live job was denied, so exact1000 backup
      `102292` remains pending on `afternotok:102169` with `08:00:00`
      walltime. Continue with the strict full1000 chain; do not report the
      10-sample visual artifacts as a smaller training/evaluation subset.
- [ ] 2026-06-04 02:03 full1000 backup chain takeover:
      inspected the trainer loop and confirmed `MIN_TRAIN_SECONDS` is checked
      only at epoch boundaries. Main job `102169` had entered epoch 1 and was
      only at batch `300/1750` by elapsed `02:46:14/04:00:00`, so it could not
      complete epoch 1 plus final `model.pt/metrics.json` before walltime and
      would not trigger `afterok` inspection. Canceled `102169`, dead main
      tail jobs `102170-102191`, and main visual replacements `102464-102470`.
      This is a scheduling/walltime correction, not a method or slot-quality
      result. Backup job `102292` started on `server62` with exact `1000`
      RGB-D input, 4 H200, `MIN_TRAIN_SECONDS=10800`, and `08:00:00` walltime.
      Replaced backup visual-only jobs
      `102302/102305/102308/102311/102314/102315/102316` with 10-sample jobs
      `102483-102489`; backup chain contract audit after that replacement
      passed `206/206`. The active full1000 path is now backup chain
      `102292-102301/102303/102304/102306/102307/102309/102310/102312/102313`
      plus visual/evidence jobs `102483-102489`.
- [ ] 2026-06-04 02:32 full1000 backup slot load passed:
      `102292` on `server62` completed full1000 metadata load for all four
      slot members with exact `301000` samples and
      `[301000, 8, 128, 128]` RGB-D image shape. Load took `1646-1684`
      seconds, slower than the canceled `server34` main job but active and
      stderr-clean. Epoch-0 training has started and reached batch `50` for
      all four members. This is startup/readiness evidence only; continue to
      strict slot inspection `102293` before allowing predicted-slot export
      `102294` to serve the world-model/controller chain.
- [ ] 2026-06-04 continuation memory hygiene:
      `AGENTS.md` now explicitly records the current visualization budget:
      limit human-readable rendered/visual demo artifacts to `10` diagnostic
      samples, while preserving exact `1000` RGB-D for slot training,
      predicted-slot export, RGB-D-derived world-model training, controller
      inputs, controller metrics, and all strict gates.
- [ ] 2026-06-04 full1000 predicted-slot export repair:
      exact1000 slot training `102292` completed and strict slot inspection
      `102293` passed. Export `102294` produced only `598/1000` predicted-slot
      files before timing out on `server13`; stderr showed
      `Unable to determine the device handle for GPU0`, so this is classified
      as scheduling/node-GPU plus export walltime failure, not method evidence
      or a reason to relax gates. Patched
      `export_rgbd_predicted_slots.py` with `--resume-existing`, which skips
      only valid already-written RGB-D-derived predicted-slot H5s and computes
      the missing files, then writes the final `predicted_slot_h5s.txt` and
      `export_report.json` for exact `1000` inputs. Added CPU Slurm resume
      wrapper `export_rgbd_predicted_slots_cpu_resume.sbatch`. Submitted
      `103815` with `RESUME_EXISTING=true`, `CUDA_EXPORT=false`, exact
      `1000`, and the same `job102294` output directory; initially chained
      strict export inspection `102295` to that primary resume. Partial visual review
      `102483` is invalid for full1000 evidence. Because `103815` used the
      pre-optimization row-wise existing-H5 resume path after it started, it
      was canceled after `00:53:42` before making new files. Submitted
      optimized exact1000 CPU resume job `103904`, still writing the same
      `job102294` output directory with `RESUME_EXISTING=true` and
      `CUDA_EXPORT=false`; strict export inspection `102295` and the
      replacement 10-sample visual review `103816` now wait on
      `afterok:103904`. This remains the full `1000` RGB-D training/export
      chain; only rendered human-readable visual diagnostics are capped at
      `10` samples.
- [ ] 2026-06-04 full1000 predicted-slot quality failure and aligned retry:
      optimized resume export `103904` completed the structurally exact
      `1000` RGB-D predicted-slot export (`301000` samples; `598` reused and
      `402` recomputed), but strict inspection `102295` failed the unchanged
      task-frame quality gate: `hole_pos_rmse_m=0.031467>0.030000` and
      `peg_head_hole_rmse_m=0.054956>0.035000`. Sensitivity job `102296`
      confirms binary predicates are strong but continuous task-frame geometry
      remains too noisy (`peg_head_at_hole_rmse_m=0.042367`,
      `insert_yz_gate` mismatch `0.064791`). Replacement 10-sample visual
      review `103816` completed with `0` warnings and was opened directly;
      the sheet is nonblank, with visible target/prediction separation in
      several rows, consistent with the strict failure. This is a RGB-D
      perception quality failure, not method evidence and not a reason to
      train the world model on failed slots or relax gates. Patched future
      slot training with explicit task-frame loss weights that default to old
      behavior, then submitted exact1000 replacement chain
      `full1000_strongslot128_grid8_taskloss_20260604_1848`: slot job
      `105236` uses 4 H200, `16:00:00`, `MIN_TRAIN_SECONDS=10800`,
      `CNN_CHANNELS=128`, `HIDDEN_DIM=512`, `SPATIAL_GRID_SIZE=8`,
      `HOLE_POS_LOSS_WEIGHT=2.0`, `PEG_HEAD_HOLE_LOSS_WEIGHT=4.0`,
      `EPOCHS=2`, and `LAZY_IMAGES=true`. Downstream jobs `105237-105260`
      preserve exact `1000`, strict predicted-slot gates, RGB-D-derived world
      model/controller boundaries, and 10-sample visual diagnostics. Contract
      audit passed `206/206`. Old failed-branch dead dependencies
      `102297-102313` and `102484-102489` were canceled before allocation.
      Current active Slurm state: `105236` is pending with reason `Priority`
      and no authoritative `StartTime` yet. Evidence note:
      `docs/world_model_task_rebinding/2026-06-04_full1000_predicted_slot_quality_failure_and_strong_retry.md`.
- [ ] 2026-06-04 18:29 task-loss contract audit hardening:
      patched `audit_rgbd_method_chain_contract.py` so future chain audits
      verify `HOLE_POS_LOSS_WEIGHT` and `PEG_HEAD_HOLE_LOSS_WEIGHT` match the
      submitted manifest. Refreshed audit
      `full1000_strongslot128_grid8_taskloss_20260604_1848/contract_audit_after_taskloss_weight_check.md`
      passed `208/208`, including the two new checks. `squeue --start` now
      forecasts slot job `105236` for `2026-06-04T23:00:00` on `server08`,
      while `scontrol` still shows it pending with reason `Priority`. This is
      submission/readiness evidence only; wait for real Slurm allocation and
      slot logs before interpreting training quality.
      Fresh read at `2026-06-04T18:30:29+08:00` moved the forecast earlier to
      `2026-06-04T22:00:00`, with `105236` still `PENDING` for `Priority`.
      Treat live Slurm state as authoritative and do not submit a duplicate
      while this exact1000 aligned path is forecast to start soon.
- [ ] 2026-06-04 18:32 slot-inspection task-loss evidence hardening:
      patched `inspect_rgbd_slot_extractor_ensemble.py` so future strict slot
      inspection records task-frame loss weights and a non-gating
      `task_frame_loss_weight_evidence` field. This changes only evidence
      reporting; it does not alter compliant training gates, exact RGB-D
      gates, predicted-slot quality gates, world-model inputs, controller
      metrics, or visual budgets. `py_compile`, `bash -n` for the inspection
      wrapper, and a `.venv` read-only smoke on old exact1000 slot job
      `102292` passed. The old smoke still reports
      `rgbd_slot_training_evidence=True` but
      `task_frame_loss_weight_evidence=False`, preserving the old failure
      classification while preparing `105237` to prove the new repair
      metadata after `105236` completes. Latest `squeue --start` forecast for
      `105236` moved earlier again to `2026-06-04T21:00:00`; the job remains
      `PENDING` for `Priority`, so do not duplicate the exact1000 training
      path.
- [ ] 2026-06-04 18:37 waiting-period dependency recheck:
      `105236` remains `PENDING` for `Priority`, with `squeue --start`
      forecasting `2026-06-04T21:00:00`; no `wm_rgbd_slot4-105236.out/err`
      logs or `job105236` artifacts exist yet. Read-only `jobs.tsv`,
      `scontrol`, and refreshed audit checks confirm the chain still preserves
      the full method gates: slot training is exact `1000` with task-frame
      loss weights, strict predicted-slot inspection `105239` keeps the
      original thresholds, RGB-D-derived world-model training `105241`
      depends on `afterok:105239`, and visual review `105259` is only the
      10-sample diagnostic branch after export. Temporary audit
      `/tmp/full1000_chain_wait_audit.md` passed `208/208`. This is
      waiting-period readiness evidence only, not slot quality or method
      evidence; do not submit a duplicate while the aligned exact1000 path is
      forecast to start soon.
- [ ] 2026-06-04 18:42 user visual-budget confirmation and pending-job triage:
      user clarified that only rendered/visual demo artifacts should stay
      capped at `10` diagnostic samples; RGB-D slot training, predicted-slot
      export, strict inspections, RGB-D-derived world-model training,
      controller inputs, controller metrics, and exact-count gates must still
      use all `1000` RGB-D demos. Fresh Slurm state still has `105236`
      `PENDING` for `Priority`, with `squeue --start` and `scontrol`
      forecasting `2026-06-04T21:00:00` and `SchedNodeList=server08`; no logs
      or `job105236` artifacts exist. `jobs.tsv` confirms `105259` visual
      review is `SAMPLE_FILES=10` while export/inspection gates remain
      `MIN_FILES=1000` and `EXPECTED_FILES=1000`, and `105241` still depends
      on `afterok:105239`. Refreshed contract audit passed `208/208`.
      Lightweight pending-job triage wrote
      `experiments/world_model_task_rebinding/rgbd_method_chains/full1000_strongslot128_grid8_taskloss_20260604_1848/slot105236_pending_triage_20260604_184216.{json,md}`
      and classified the state as `scheduling_pending`,
      `pending_reason_priority`, and `no_ensemble_artifacts_yet`. This is
      readiness evidence only; keep the current exact1000 path queued and
      recheck when it starts or on the normal cadence.
- [ ] 2026-06-04 18:45 submitted snapshot recheck:
      after re-reading the active and focused RGB-D TODO files, live Slurm
      state still shows `105236` `PENDING` for `Priority` with
      `squeue --start` forecasting `2026-06-04T21:00:00`,
      `SchedNodeList=server08`, `16:00:00`, and `gres/gpu=4` H200; no
      stdout/stderr logs or `job105236` artifacts exist. `sacct SubmitLine`
      confirms the submitted environment, not just wrapper defaults: `105236`
      uses exact `1000` RGB-D files, `MIN_TRAIN_SECONDS=10800`, task-frame
      loss weights `2.0/4.0`, and `EPOCHS=2`; `105238` exports exact `1000`
      frame-aligned predicted slots; `105239` keeps the unchanged strict
      thresholds; `105241` requires exact `1000` predicted-slot files and
      depends on `afterok:105239`; `105259` is the diagnostic visual branch
      with `SAMPLE_FILES=10`, `FRAMES_PER_FILE=4`, and exact `1000` file
      gates. Submitted-chain audit `/tmp/full1000_chain_snapshot_audit.md`
      passed `208/208`, including RGB-D controller/video/nonblank/
      oracle-exclusion checks. This remains readiness evidence only, not
      method evidence.
- [ ] 2026-06-04 18:47 queue hygiene recheck:
      filtered live queue shows only the active replacement RGB-D method chain
      `105236-105260` among relevant `wm_*` jobs. Old failed-branch jobs
      `102297-102313` and `102484-102489` are all `CANCELLED by 2059` with
      `Elapsed=00:00:00` and `NodeList=None assigned`, so they are not
      consuming GPU allocations or dependency execution slots. This is
      resource hygiene evidence only; it does not prove RGB-D slot quality or
      method success.
- [ ] 2026-06-04 18:48 input/output readiness recheck:
      `105236` remained pending, so the waiting interval was used for non-GPU
      readiness checks. `AGENTS.md` still records the full1000 boundary:
      visual demo artifacts are capped at `10` diagnostic samples, but exact
      `1000` RGB-D remains required for slot training, predicted-slot export,
      world-model training, controller inputs/metrics, strict gates, and the
      full1000 data requirement. The submitted RGB-D input root still contains
      exactly `1000` `.rgbd.h5` files. The `job105236` output directory is
      absent and `wm_rgbd_slot4-105236.out/err` are absent, so there are no
      stale checkpoints/logs to contaminate the replacement run. This is
      clean-surface readiness evidence only, not method evidence.
- [ ] 2026-06-04 18:51 afterany slot-triage diagnostic:
      submitted CPU diagnostic job `105316` with dependency `afterany:105236`,
      `cpu=2`, `mem=4G`, and `time=00:20:00`, writing to
      `full1000_strongslot128_grid8_taskloss_20260604_1848/slot105236_afterany_triage`.
      Added `diagnostic_jobs.tsv` in the chain directory to keep this separate
      from method jobs. `scontrol` confirms `105316` is dependency-pending on
      `105236`; `sacct SubmitLine` confirms it only exports
      `TARGET_JOB_ID=105236`, the `job105236` ensemble dir, and the diagnostic
      output dir. This is read-only failure-localization only; it does not
      approve slot quality, replace strict inspection `105237`, trigger
      world-model training, or change any gate.
- [ ] 2026-06-04 18:53 PLAN drift recheck after diagnostic:
      re-read `PLAN/README.md`, `00_overview.md`, `05_rebinding_controller.md`,
      and `06_rgbd_and_baselines.md`. The active chain still matches the plan:
      RGB-D is required method evidence, oracle/state slots are only
      scaffolds, the RGB-D path trains object/task slots rather than a pixel
      world model from scratch on 1000 demos, and the controller must use
      object slots, predicted task frames, uncertainty, and conservative
      handoff rather than case-name branches. Refreshed contract audit
      `/tmp/full1000_chain_after_diagnostic_audit.md` passed `208/208` after
      adding diagnostic job `105316`; the method chain still preserves
      exact1000 inputs, strict predicted-slot gates, RGB-D-derived
      world-model/controller boundaries, video review branches, and
      oracle-source exclusion. This is plan-hygiene/readiness evidence only,
      not method evidence.
- [ ] 2026-06-04 18:55 same-shape scheduling probe:
      `105236` remains `PENDING` for `Priority` with
      `StartTime=2026-06-04T21:00:00`, `SchedNodeList=server08`, and
      `gres/gpu=4` H200. `server08` is currently `ALLOCATED` with
      `Gres=gpu:NVIDIAH200:8` and `AllocTRES=...gres/gpu=6`, consistent with
      waiting for resources to free. A non-submitting same-objective
      `sbatch --test-only` probe using the same exact1000 slot-training
      environment forecast `2026-06-05T00:23:29`, later than active `105236`;
      `squeue` and `sacct` confirm the probe id was not an actual submitted
      job. Decision: do not submit a duplicate/replacement; keep the active
      exact1000 path queued. This is scheduling evidence only.
- [ ] 2026-06-04 19:00 slot-wrapper manifest boundary metadata fix:
      static wrapper check found only a stale manifest text field in
      `train_rgbd_slot_extractor_ensemble_4gpu.sbatch`: the `boundary=` line
      still mentioned `formal_full96_dataset`. Patched the source wrapper to
      say `formal_inspected_rgbd_dataset_with_exact_expected_count` instead.
      This is a metadata wording fix only; it does not change Slurm resources,
      data paths, training loss, thresholds, dependencies, export behavior,
      world-model inputs, controller metrics, or visual budgets. Because
      `105236` was already queued, `scontrol write batch_script 105236` still
      contains the old string in the submitted snapshot, but the authoritative
      `sacct SubmitLine` remains exact `1000` via `MIN_RGBD_FILES=1000` and
      `EXPECTED_RGBD_FILES=1000`. `bash -n` passed and refreshed audit
      `/tmp/full1000_chain_after_manifest_boundary_fix_audit.md` passed
      `208/208`. At `2026-06-04T19:00:10+08:00`, `105236` was still pending
      with forecast `2026-06-04T20:56:37` and no logs/artifacts. This is
      metadata/readiness hygiene only, not method evidence.
- [ ] 2026-06-04 19:21 one-H200 RGB-D training/resource correction:
      user replaced the old training floor with `>=1xH200` and `>=10800s`;
      4 H200 is no longer mandatory. Created reusable tmux allocation
      `h200_1gpu_pool` / Slurm `105385` (`1` H200, `1-00:00:00`, no
      exclusions), granted on `server62`, and verified repeated `srun`
      commands can reuse the allocation with H200/PyTorch CUDA visible. Kept
      running exact1000 slot job `105236` because it already holds a valid
      4-H200 allocation. Patched resource defaults/checks so future slot and
      RGB-D-derived world-model training default to 1 H200/1 day while still
      requiring exact `1000` RGB-D inputs, strict predicted-slot gates,
      RGB-D-derived world-model/controller boundaries, and 10-sample visual
      diagnostics only. Started a full1000 1-H200 backup slot run at
      `rgbd_slot_extractor/ensemble_1h200_pool/job105385_strongslot_seed1600`.
      Submitted 1-H200 downstream tail
      `full1000_strongslot128_grid8_taskloss_1h200_tail_20260604_192147`
      (`105429-105447`) reusing upstream `105236-105239`; contract audit
      passed `208/208`. This is resource/readiness evidence only, not method
      success.
- [ ] 2026-06-04 19:26 1-H200 tail cleanup and live slot progress:
      canceled old 4-H200 downstream branch `105241-105258` plus old evidence
      review `105260` after the audited 1-H200 tail `105429-105447` was in
      place. `sacct` shows all canceled old-tail jobs had zero elapsed time and
      no allocated nodes/TRES. Active upstream remains `105236-105240`, with
      10-sample visual diagnostic `105259` and slot triage `105316`. `105236`
      is running on `server08`, loaded exact1000 data for all four members, and
      reached epoch-0 batch 50 for all members. The 1-H200 backup slot run in
      allocation `105385` started member `0` under
      `rgbd_slot_extractor/ensemble_1h200_pool/job105385_strongslot_seed1600`.
      By `2026-06-04T19:32+08:00`, `105236` had reached epoch-0 batch `150`
      for all four members. The 1-H200 backup had not emitted
      `dataset_loaded` yet, but read-only `sstat`/overlap checks showed the
      Python process alive and reading HDF5 data, so it was left running.
      No slot-quality claim exists until strict slot/predicted-slot inspection
      passes unchanged gates.
- [ ] 2026-06-04 19:41 second 1-H200 request and path-file transport fix:
      active/current submissions are not blocked by stale bad-node policy:
      `105236`, `105385`, and dependency-pending `105429` all show
      `ExcNodeList=(null)`. No-exclusion `1`-H200 probes on `cpu` forecast
      `2026-06-04T22:55:25` for `03:00:00`, `08:00:00`, and `1-00:00:00`;
      the `gpu` partition forecast was `2026-06-10T15:08:25`. Submitted
      second real tmux allocation `h200_1gpu_pool2` / Slurm `105502`
      (`1` H200, `8` CPU, `64G`, `1-00:00:00`, no exclusions) and preloaded
      the pane to run `scripts/slurm/run_rgbd_slot_backup_in_allocation.sh`
      when granted. That backup is exact1000 RGB-D strongslot seed `1700`, not
      oracle/state evidence. Patched RGB-D slot training transport so
      `train_rgbd_slot_extractor.py` accepts `--paths-file` and the slot
      wrapper passes the path file to member processes instead of expanding
      all `1000` H5 paths in argv. `bash -n`, `py_compile`, and CLI help
      checks passed. This does not change data, labels, losses, thresholds,
      RGB-D-derived downstream gates, controller metrics, or the visual cap.
      Follow-up at `2026-06-04T19:43+08:00`: `105502` forecast moved to
      `2026-06-04T20:53:39` with `ExcNodeList=(null)`, and `105236` reached
      epoch-0 batch `400` for all four members with empty stderr.
- [ ] 2026-06-04 19:49 105502 CUDA visibility failure:
      `105502` started on `server13` and launched the intended exact1000
      seed `1700` RGB-D slot backup, but live allocation canaries found
      `nvidia-smi` could not see a device and `.venv` PyTorch reported
      `torch_cuda_available False` / `torch_device_count 0`. Because CPU-only
      slot training is not compliant RGB-D method evidence, canceled `105502`
      after `00:02:38` before dataset load or checkpoint. Classified this as
      scheduling/node GPU visibility failure, not data/perception/world-model/
      controller/physics/evaluation evidence. Added default `require_cuda` to
      `train_rgbd_slot_extractor.py`, propagated `REQUIRE_CUDA` through the
      slot wrapper and allocation runner, and submitted replacement tmux
      allocation `105535` with only the current failed `server13` excluded.
      This preserves exact1000 inputs, strict gates, RGB-D-derived downstream
      requirements, controller metrics, and the 10-sample visual cap.
- [ ] 2026-06-04 20:01 CUDA-gated downstream replacement:
      submitted and audited replacement method tail
      `full1000_strongslot128_grid8_taskloss_1h200_tail_cudagate_20260604_1953`
      (`105553-105571`) so the RGB-D-derived world-model training snapshot
      also carries `REQUIRE_CUDA=true` and path-file transport. The chain
      reuses exact1000 upstream `105236-105240`, keeps strict predicted-slot
      thresholds, requests `1` H200 / `10800s` for world-model training, and
      keeps all controller/video branches at `SLOT_SOURCE=rgbd`.
      `contract_audit.md/json` passed `208/208`. Superseded old tail
      `105429-105447` was canceled only after the replacement audit passed;
      every old-tail job had zero elapsed time and no allocation. Current
      running branches are `105236` on `server08` with no exclusion, backup
      `105385` on `server62` with no exclusion, and backup `105535` on
      `server28` with only job-local `server13` excluded from live CUDA
      evidence. No RGB-D method evidence exists until strict slot,
      predicted-slot, RGB-D-derived world-model, controller metrics, and
      video/contact review pass.
- [ ] 2026-06-04 20:14 resource/diagnostic correction:
      fixed triage-only false positives around `MIN_TRAIN_SECONDS` and added
      a pre-dataset CUDA canary to the reusable allocation runner. Live jobs do
      not carry a standing bad-node list: main slot training `105236`, backup
      `105385`, and downstream tail `105553` have no exclusions; `105535`
      excludes only `server13` from the immediately observed CUDA visibility
      failure. Submitted one no-exclusion 1-H200 allocation `105646` in tmux
      `h200_1gpu_pool4`, forecast for `2026-06-04T21:31:41` on `server43`, to
      run exact1000 RGB-D strongslot seed `1900` after CUDA canary. No change
      to exact1000 data, slot labels, strict gates, RGB-D-derived downstream
      requirements, controller metrics, or 10-sample visual cap.
- [ ] 2026-06-04 20:21 CPU-first active chain scheduling:
      active dependency-pending GPU jobs now prefer `cpu,gpu` instead of slow
      `gpu`/`gpu,cpu` partition order: predicted-slot export `105238`,
      RGB-D-derived world-model `105553`, and controller-video jobs
      `105556/105559/105562/105565/105568`. Patched future current-chain
      defaults in the export, world-model, controller-video wrappers and
      coordconv submitter. This does not change any data, labels, strict
      thresholds, RGB-D-derived boundaries, controller settings, metrics, or
      video evidence rules.
- [ ] 2026-06-04 20:25 allocation-runner canary repair:
      allocation `105646` failed before dataset loading because the reusable
      allocation runner tested CUDA in the direct `salloc` shell instead of the
      `srun` step used by training. Correct `srun` canaries saw H200/CUDA on
      both `server62` and `server28`; classify this as runner implementation,
      not RGB-D method evidence or a standing node exclusion. Patched the
      runner to use `srun --gpus-per-task=1` for the pre-dataset CUDA canary
      and submitted replacement no-exclusion allocation `105707` for exact1000
      strongslot seed `1900`.
- [ ] 2026-06-04 20:34 one-H200 allocation correction:
      fixed-runner allocation `105707` started on `server13` with no submitted
      exclusion and the `srun` CUDA canary failed before dataset loading
      (`nvidia_smi_returncode=255`, `torch_cuda_available=false`). This is a
      repeated current server13 GPU-visibility scheduling failure, not RGB-D
      data/perception/method evidence. Submitted tmux allocation `105743`
      (`h200_1gpu_pool6`, `1` H200, `1-00:00:00`, exact1000 seed `2000`) and,
      after the repeated current failure, set only job-local
      `ExcNodeList=server13`; it then started earlier than forecast on
      `server43` at `2026-06-04T20:34:44`, passed the fixed `srun` CUDA canary,
      and launched exact1000 seed `2000` training at `20:35:10`. Main `105236`
      still trains exact1000 on 4 H200 and backups `105385`/`105535`/`105743`
      are running on one H200 each. No slot checkpoint/metrics have passed
      strict inspection yet, so there is still no full RGB-D method evidence.
      The 10-sample visualization cap is unchanged and applies only to
      human-readable visual diagnostics, not slot training/export/world-model/
      controller data.
- [ ] 2026-06-04 20:36 visualization-budget audit:
      active `105259` predicted-slot visual review submit line exports
      `SAMPLE_FILES=10,FRAMES_PER_FILE=4`, and the active chain manifest
      records `slot_visual_sample_files=10` and
      `video_review_sample_count=10`. Patched future defaults for predicted
      slot visual review, video artifact review, and future RGB-D distributed
      visual review to default to `10` diagnostic samples. This preserves the
      user's visual cap while leaving the exact1000 RGB-D training/export/
      world-model/controller/evaluation data unchanged.
- [ ] 2026-06-04 20:50 small one-H200 request override:
      user directed more aggressive small GPU allocation requests and warned
      not to attribute current queue behavior to old bad-node lists. Added
      `scripts/slurm/launch_rgbd_1h200_allocation_tmux.sh` to reproducibly
      launch tmux-backed one-H200 `salloc` jobs that run the exact1000 RGB-D
      strongslot backup after a fixed `srun` CUDA canary and keep the
      allocation reusable. Initial one-day requests `105826/105827` were
      canceled while pending with zero allocation after shorter probes showed
      earlier backfill. Replacement 12-hour requests are `105868`
      (`h200_1gpu_pool7`, no exclusions, seed `2100`) and `105867`
      (`h200_1gpu_pool8`, only job-local `ExcNodeList=server13`, seed
      `2200`). These are RGB-D perception-gate backups for the same exact1000
      dataset, not small-scope method evidence. They do not change slot
      labels, exact-count gates, predicted-slot export, RGB-D-derived
      world-model training, controller metrics, or the 10-sample visual
      diagnostic cap. Current strict method evidence remains pending:
      `105236` reached epoch-0 batch `1250`, `105535` loaded exact1000 and
      started training, `105385`/`105743` are alive, but no slot
      checkpoint/metrics have passed strict inspection.
- [ ] 2026-06-04 20:56 strict one-H200 backup assembly preflight:
      added `scripts/world_model/assemble_rgbd_slot_backup_ensemble.py` as a
      fallback-only tool for combining completed one-H200 exact1000 RGB-D slot
      backup members into an inspectable ensemble. It refuses assembly unless
      every source member has metrics, manifest, checkpoint, exact1000 paths,
      one-H200 source manifest, `>=10800s` elapsed time, and RGB-D image plus
      robot-only proprio input boundary. `py_compile` passed. Current dry-run
      over `105385`, `105535`, and `105743` correctly returned status `65`
      because no backup member has completed metrics/checkpoints or the 3h
      floor yet. This prevents a future fallback from becoming a shortcut; it
      does not create RGB-D method evidence.
