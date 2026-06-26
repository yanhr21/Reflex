# Failed-State Recovery Supplement Plan

## Boundary

This is a new proposed recovery-data path after the 733-only clean/dense export
and the scripted hard-teacher supplement both failed to cover live closed-loop
failure states.

Do not mix this with the old targeted hard-teacher variants. Do not start data
generation, merge, WAM export, SFT, or controller training from this path
without explicit user approval.

## Plain Diagnosis

The current short overfit is fine. It proves the export, SFT, checkpoint
restore, generation, action sidecar, and visual inspection path can learn two
rows.

The current full run is blocked by data coverage, not by the overfit/training
code. The 733 clean/dense source contains many successful rows, but the live
failed closed loop asks the model what to do when the target has moved and the
held peg is already several centimeters away from the new hole in y/z.

The first targeted hard-teacher idea also failed: moving the target while the
peg waits near the old hole often creates wall-sweep/contact geometry that the
teacher cannot pass. Four pilots produced `0` accepted rows. Repeating those
variants is not useful.

## What The Existing Failed Artifacts Can And Cannot Do

Inspected failed closed-loop root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen2_20260614/cosmos_iter2700_puredp_fail_4_9_10_11_12_13`

Observed:

- each sample has many `iter_##_prefix_f###` query folders;
- each query folder has `observed_prefix.mp4` and
  `live_history_raw_action_state.json`;
- `live_history_raw_action_state.json` contains only a `300 x 32` WAM
  action/state condition array;
- the summary records real relative error such as
  `peg_head_pos_at_hole`, success/failure, controller mode, and action chunk
  paths;
- it does not contain full live `env_state` or `get_state_dict()` snapshots.

Consequence:

The old failed artifacts are enough to diagnose the missing training
distribution. They are not enough to directly restore the simulator at a failed
post-motion contact state.

## Code Prepared

The live receding loop now has an opt-in state-recording switch:

```bash
SAVE_LIVE_STATE_SNAPSHOTS=true
```

When enabled, `run_cosmos3_live_receding_loop.py` writes, for each receding
iteration:

- `live_state_before_controller.h5`
- `live_state_after_controller.h5`

Each H5 stores:

- `state/`: real `base_env.get_state_dict()` converted to numpy;
- `observation/`: real `base_env.get_obs()` converted to numpy;
- attributes: `prefix_frame_index`, `iteration`, `label`, and boundary text.

These snapshots are not controller inputs and not method evidence by
themselves. They are only restore points for constructing recovery training
rows from real failed states.

The direct snapshot-enabled rerun exposed an implementation/runtime blocker:
the source-restore diagnostic reached `live_source_restore_before_render_prefix`
and then stalled in prefix rendering. To avoid confusing that render-prefix
stall with the data question, a separate no-render replay tool was added:

`scripts/world_model/replay_live_history_state_snapshots.py`

That tool replays the recorded live-loop robot actions from the old summary,
applies the same source target motion, and writes simulator snapshots at the
old query frames. It does not call Cosmos and does not render video.

## Snapshot Replay Result

Replay evidence from 2026-06-14:

- sample_09:
  `experiments/world_model_task_rebinding/cosmos3/replay_state_snapshots_sample09_20260614_141229`
  reproduced the old failed final state exactly:
  `peg_head_pos_at_hole =
  [-0.1239014864, -0.0626012236, -0.0517419614]`, old/new L2 `0.0`,
  `28` query snapshots.
- samples 10-13:
  `experiments/world_model_task_rebinding/cosmos3/replay_state_snapshots_hard2_remaining_20260614_141435`
  reproduced each old final failure with old/new L2 `0.0`.
- sample_04 also replayed exactly, but it was `success=True`; do not use it as
  a negative failed-state recovery seed.
- Total written query-state H5 files across 04 and 09-13: `163`. Usable failed
  snapshots are from 09-13.

This proves the old summaries can be converted into faithful restore-state
candidates without rerunning Cosmos or relying on source trajectory states.

## Recovery Teacher Smoke Result

A narrow failed-state recovery generator was added:

`scripts/world_model/generate_cosmos3_failed_state_recovery_teacher.py`

Its intended semantics:

- restore a real `frame_***_live_state.h5`;
- keep the real robot/peg/hole failed geometry;
- mark `perturb.triggered` from step 0 so the later WAM export treats the row
  as post-motion recovery context rather than a fresh static episode;
- ask a motion-planning teacher to regrasp, re-align, and insert;
- save only accepted `301/300` rows for later RGB review.

Smoke result:

`experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_smoke_20260614_142222`

Accepted `0/1` requested rows after `10` attempts over frames `296`, `288`,
`280`, and `272` from samples 09, 10, and 13.

Observed reject classes:

- `planner_preinsert_failed`
- `preinsert_gate_failed`
- `final_insert_wall_collision_risk`
- `planner_final_insert_refine_failed`

Plain interpretation:

The restore-state problem is solved enough for smoke testing. The current new
blocker is the recovery teacher itself: the motion-planning teacher cannot
reliably re-align and insert from the real failed closed-loop poses. This is
not a Cosmos training failure, and it does not justify starting full SFT.

Follow-up variants:

- staged preinsert smoke:
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_stage_smoke_20260614_143157`,
  accepted `0/1` after `20` attempts;
- existing-grasp-first smoke:
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_existing_grasp_smoke_20260614_143904`,
  accepted `0/1` after `24` attempts;
- multi-stage stronger refinement smoke:
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_multi_stage_smoke_20260614_144327`,
  accepted `0/1` after `24` attempts.
- release/regrasp-after-stage smoke:
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_release_regrasp_smoke_20260614_1510`,
  accepted `0/1` after `9` attempts;
- shallow release/regrasp smoke:
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_release_regrasp_shallow_smoke_20260614_1516`,
  accepted `0/1` after `9` attempts;
- counter-fix shallow release/regrasp smoke:
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_release_regrasp_shallow_counterfix_smoke_20260614_1525`,
  accepted `0/1` after `3` attempts;
- single close-case stronger-refine smoke:
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_release_regrasp_sample13_refine_smoke_20260614_1532`,
  accepted `0/1` after `1` attempt.
- task-axis release-regrasp smoke:
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_goal_y_regrasp_smoke_20260614_1542`,
  accepted `0/1` after `3` attempts;
- planned task-axis regrasp smoke:
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_planned_goal_y_regrasp_smoke_20260614_1548`,
  accepted `0/1` after `3` attempts.

Plain interpretation of the follow-ups:

The problem is not that no failed states exist. The replay produced real
failed-state snapshots and the teacher tried them. The problem is that the
planner teacher cannot turn those states into a clean positive recovery row.
Release/regrasp after staging is the right physical direction for the bad-grasp
diagnosis, and it improved some cases, but still produced no valid row. The
closest case put the peg head near the hole with no wall risk, but the whole
peg line still missed the strict centerline gate and live success stayed false.
Task-axis regrasp then tested whether the problem was simply that the regrasp
was inheriting the current TCP closing direction. That also failed: explicit
`goal_y` closing could not create an accepted row, and when forced before
staging it sometimes made the staged peg line worse. More reruns of the same
offset/regrasp/task-axis/refine style are unlikely to create valid data.

Implementation note:

The release/regrasp tests exposed and fixed a counter bug in
`generate_cosmos3_failed_state_recovery_teacher.py`: `raw_steps` and
`final_insert_start_step` must be measured relative to the current attempt,
not the global `RecordEpisode` step counter accumulated across rejected
attempts. The counterfix smoke confirmed that this bug could create false
`raw_steps_exceed_limit` rejects, but after the fix the remaining failures were
physical planner/insertion failures, not accounting failures.

## Proposed Flow

1. Keep the no-render replay snapshots as the current faithful restore-state
   source; do not confuse them with visual/method evidence.
2. Fix the recovery teacher strategy before any broader data generation. The
   next fix must be a different teacher mechanism for getting a physically
   valid grasp/axis/alignment from the failed pose, not another sweep over the
   same planner offsets, staged release/regrasp, task-axis closing modes, or
   final-refine counts.
3. Once a teacher accepts at least a small smoke set, save accepted recovery
   rows as H5 with `301` state/video frames and `300`
   action steps, with provenance marking them as failed-state recovery
   supplements.
4. Render RGB review sheets with the approved ManiSkill default-style camera.
5. Visually reject any row where the peg is not actually held/recovered or the
   final insertion is only a metric artifact.
6. Only after visual approval, merge accepted supplement rows with the frozen
   733 source, rerun WAM export, strict preflight, and live-query coverage.
7. Run a short 1-2 GPU, 50-100 step overfit sanity only if the merged preflight
   passes.
8. Start full 4-GPU SFT only if coverage and overfit both pass.

## Approval-Gated Snapshot Command

Do not run this on the login node. Do not use `sbatch`. Run only inside the
held tmux/Slurm allocation, or an equivalent tmux-held compute allocation:

```bash
srun --overlap --jobid=127723 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=140G \
  bash -lc 'cd /public/home/yanhongru/ICLR2027/Reflex && \
  STAMP=failed_state_snapshots_20260614 \
  SAMPLE_INDICES=9,10,11,12,13 \
  MAX_SAMPLES=5 \
  SAVE_LIVE_STATE_SNAPSHOTS=true \
  OUTPUT_ROOT=experiments/world_model_task_rebinding/cosmos3/failed_state_snapshots_iter2700_hard2_20260614 \
  bash scripts/slurm/run_cosmos3_live_receding_panel_in_allocation.sh'
```

This command reruns the live closed-loop evaluation to record restore states.
It does not create supplement data, merge data, export WAM conditions, or train.

## What Would Prove This Direction Is Worth Continuing

- The rerun writes state snapshots for the undercovered late post-motion query
  frames.
- The saved states restore the real failed geometry: target moved, peg still
  live, peg-head offset from the moved hole, and no fake future target pose as
  controller input.
- A recovery teacher can finish from those states while preserving visual
  grasp/recovery and real final insertion.
- After merging visually approved rows, the live-query coverage audit improves
  on the specific `58/173` undercovered queries.

## What Would Falsify It

- The replay cannot save valid restore states.
- Restored states diverge from the old failed summaries.
- The recovery teacher cannot finish from the real failed states without
  unrealistic resets or future-state leakage.
- The merged preflight still misses the same live-query gap.

Current status: the first two falsifiers are cleared for samples 09-13, but
the third falsifier is active after several teacher variants. Do not start full
SFT from this path. Record any future teacher fix as a new mechanism, not as a
rerun of the same failed planner.
