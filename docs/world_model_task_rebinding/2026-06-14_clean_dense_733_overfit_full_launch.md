# 2026-06-14 Clean/Dense 733 Overfit To Full Launch

## Boundary

This note records the launch of the user-approved clean/dense repair chain.
It is not method success evidence yet.

Goal:

1. Start from the frozen 733 RGB source data.
2. Build a denser clean-role condition root with `prefix_role_source=physical_mode`.
3. Check source visuals before training.
4. Run a short two-sample overfit gate, about 100 steps on 1-2 GPUs.
5. Start full 4-GPU training only if preflight and overfit/eval gates pass.

## Visual Source Check Before Launch

Opened source review sheets:

- `experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612/review_sheets/0000_hole_late_move_stop_seed1080087_idx1760.fix3_traj_0_review_sheet.png`
- `experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612/review_sheets/0001_hole_late_move_stop_seed1080097_idx1761.fix3_traj_0_review_sheet.png`

Visual interpretation:

- camera/framing is the approved 512x512 ManiSkill default-style human view;
- the robot keeps the peg visible;
- the final frames show real peg-in-hole/block insertion rather than a metric-only
  or off-camera success.

This only validates the source visual examples used for the short overfit
gate. Generated overfit review sheets still must be inspected after eval.

## Launched Compute

- tmux session: `cosmos3_clean_dense_4gpu_20260614`
- Slurm job: `127723`
- node: `server54`
- allocation: 4 GPUs, 2 days
- launch command inside allocation:
  `srun --overlap --ntasks=1 --gres=gpu:4 --cpus-per-task=32 --mem=220G bash scripts/slurm/run_cosmos3_clean_dense_733_overfit_then_full_in_allocation.sh`

Run root:

`experiments/world_model_task_rebinding/cosmos3/clean_dense_733_overfit_then_full_20260614_115421`

Planned artifact roots:

- full clean/dense condition:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_rgb_300step_20260614_115421`
- full clean/dense preflight:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_preflight_20260614_115421`
- overfit2 condition:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_overfit2_rgb_300step_20260614_115421`
- overfit2 preflight:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_preflight_20260614_115421`
- overfit SFT:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_rgb_300step_fix1recipe_2gpu_20260614_115421`
- full SFT:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_rgb_300step_fix1recipe_4gpu_20260614_115421`

## Current Status

Full clean/dense preflight finished export but failed the live-query coverage
gate, so full SFT was not launched.

Full preflight artifacts:

- condition root:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_rgb_300step_20260614_115421`
- preflight root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_preflight_20260614_115421`
- source episodes: `733`
- dense condition rows: `9223`
- role/mode mismatch: `0`
- strict full-episode preflight: pass
- receding distribution audit: pass
- live Cosmos queries checked: `173`
- undercovered live queries: `63` (`0.3642`)
- `clean_dense_preflight_summary.json`: `ready_for_overfit=false`

Plain interpretation: the export is structurally correct and denser than the
old root, but it still does not contain enough local neighbors for the failed
closed-loop recovery states. The undercovered states are mostly late
`target_post_motion` queries and `hole_late_fast_shift` `peg_recovery` queries
from six already-failed closed-loop videos. In those live failures, the peg is
several centimeters off in y/z after the target has moved; the accepted 733
source data mostly contain successful trajectories that have already reached
`insert_resume` or insertion by the same time. This is a data-distribution
coverage failure, not a frame-length/truncation failure and not a visual-camera
failure.

Because this gate failed, full 4-GPU training remains blocked.

## Late Prefix Diagnostic

After the first coverage failure, the export wrapper was updated to expose
`MIN_PREFIX_FRAMES` and `MAX_PREFIX_FRAMES`. This tested whether the failure
was caused by the old default `max_prefix_frames=260` cap.

Diagnostic run:

- run root:
  `experiments/world_model_task_rebinding/cosmos3/clean_dense_733_late299_stride8_20260614_130050`
- condition root:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050`
- preflight root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_preflight_20260614_130050`
- `MAX_PREFIX_FRAMES=299`
- `DENSE_RECEDING_PREFIX_STRIDE=8`
- rows: `9271`
- strict full-episode preflight: pass
- train prefix max: `294`
- role/mode mismatch: `0`
- strict length/action/sidecar audit: pass
- live Cosmos queries checked: `173`
- undercovered live queries: `58` (`0.3353`)
- `clean_dense_preflight_summary.json`: `ready_for_overfit=false`

Remaining undercovered queries:

- `50` `target_post_motion`
- `8` `peg_recovery`
- by scenario: `25` `hole_late_sine`, `14` `hole_late_constant`,
  `11` `hole_late_continuous_insert`, `8` `hole_late_fast_shift`

Interpretation: allowing later prefixes recovered only `5` of the `63`
previously undercovered queries. The old `260` prefix cap was a real but small
export issue. It is not the main blocker. The remaining gap is that the current
733 accepted source data still lack enough examples of the failed closed-loop
recovery states where the peg/head is already far from the moved hole in y/z.

Boundary: full SFT remains blocked. Starting a 4-GPU full run from this root
would knowingly train on a distribution that still misses one third of the
states queried by the failed closed loop.

## Short Overfit Diagnostic

The first overfit-only rerun exposed a wrapper bug: selecting the first two
source episodes produced only train rows and an empty val JSONL. That preflight
failed before training with:

`RuntimeError: empty jsonl: .../val/video_action_dataset_file.jsonl`

Fix:

- added `scripts/slurm/run_cosmos3_clean_dense_733_overfit_only_in_allocation.sh`;
- changed the overfit sanity root to sample two valid rows from the full 733
  condition root and write the same rows to train and val;
- changed the default two rows to `target_post_motion/hole_late_move_stop` and
  `peg_recovery/peg_drop`.

Completed short overfit run:

- run root:
  `experiments/world_model_task_rebinding/cosmos3/clean_dense_733_overfit_only_20260614_122020`
- overfit condition root:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_overfit2_rgb_300step_20260614_122020`
- overfit SFT root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_rgb_300step_fix1recipe_2gpu_20260614_122020`
- selected rows: `2` train + same `2` val
- selected roles: `target_post_motion`, `peg_recovery`
- preflight: `ready_for_overfit=true`
- training: 2 GPUs under Slurm job `127723`
- initial validation loss at iteration `0`: `3.545426`
- iter50 checkpoint:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_rgb_300step_fix1recipe_2gpu_20260614_122020/outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/checkpoints/iter_000000050`
- validation loss at iteration `50`: `0.469846`
- the run was stopped with Ctrl-C after iter57 to keep the short sanity within
  the user-approved 50-100 step boundary; the held Slurm allocation was not
  released.

Manual iter50 eval:

- eval root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_rgb_300step_fix1recipe_2gpu_20260614_122020/eval_full_episode_wam_iter_000000050`
- eval input roles after fix: `target_post_motion`, `peg_recovery`
- strict artifact inspection: `strict_eval_artifacts_ok=true`
- generated samples: `2`
- both predicted videos: `301` frames at `30 fps`
- action arrays: `300x32`, finite and shape-aligned
- aggregate future-video PSNR: `26.9853 dB`
- aggregate robot-action prefix RMSE: `0.00155`
- aggregate robot-action future RMSE: `0.71685`
- overfit pass gate:
  `experiments/world_model_task_rebinding/cosmos3/clean_dense_733_overfit_only_20260614_122020/overfit_pass_gate_iter50.json`

Visual review:

- opened
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_rgb_300step_fix1recipe_2gpu_20260614_122020/eval_full_episode_wam_iter_000000050/review_sheets/00_target_post_motion_hole_late_move_stop_hole_late_move_stop_seed3280649_idx2518.fix3_traj_0__target_post_motion_f130_ref_pred_sheet.png`
- opened
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_overfit2_rgb_300step_fix1recipe_2gpu_20260614_122020/eval_full_episode_wam_iter_000000050/review_sheets/01_peg_recovery_peg_drop_peg_drop_seed705165_idx1029.fix3_traj_0__peg_recovery_f129_ref_pred_sheet.png`

Interpretation: the predicted videos are nonblank, have the same viewpoint as
the references, and visually track the two overfit references through the full
episode. This supports the narrow claim that the clean/dense export, short SFT,
iter50 checkpoint restore, generation, action sidecar output, and strict
inspection path are working.

Eval bug fixed during this check:

- first manual eval selected the same `peg_recovery` row twice even though the
  overfit val JSONL contained one `target_post_motion` row and one
  `peg_recovery` row;
- root cause: `build_cosmos3_full_episode_wam_eval_inputs.py` could reuse the
  same row when a desired role/scenario pair appeared twice and the candidate
  list had one item;
- fix: the eval input builder now avoids duplicate row UUIDs and directly
  preserves all strict candidates when the candidate count is less than or
  equal to the requested sample count;
- the corrected eval was rerun and passed.

Boundary: this overfit run is only a pipeline sanity check. Passing it does not
unblock full 4-GPU SFT while the full 733 live-query coverage gate still has
`63` undercovered queries.

## Current Blocker In Plain Language

The current code/training path is not the main blocker. It can train, save,
reload, generate, inspect, and make visually sane overfit outputs.

The current data distribution is the blocker. Both the original clean/dense
root and the late-prefix `MAX_PREFIX_FRAMES=299` diagnostic have many valid
rows, but not enough rows that look like the live failed closed-loop states:
late after target motion, peg still held or needing recovery, and peg-head
several centimeters off the moved hole in y/z. Training a full model from this
root would mostly teach successful-source behavior, not the states where the
closed loop actually failed.

Next aligned action: add or otherwise recover targeted training coverage for
those failed live-query states, then rerun the clean/dense preflight and
live-query coverage audit. Because this changes the data boundary beyond
733-only export repair, get user approval before launching a targeted
hard-teacher/supplement generation path. Do not start full 4-GPU SFT from the
current root as method evidence.

Prepared follow-up artifacts without running data generation:

- targeted gap manifest:
  `experiments/world_model_task_rebinding/cosmos3/targeted_recovery_gap_manifest_20260614_from_late299.json`
- targeted supplement plan:
  `PLAN/cosmos3_lowfreq_wm_executor/01_targeted_recovery_supplement.md`
- approval-gated wrapper:
  `scripts/slurm/run_cosmos3_targeted_recovery_supplement_after_approval_in_allocation.sh`
- supplement structural checker:
  `scripts/world_model/inspect_cosmos3_targeted_recovery_supplement.py`
- generator extension:
  optional `post_motion_release_regrasp_scenarios`, default off, so
  `hole_late_fast_shift` can produce real `peg_recovery` physical-mode rows
  if the targeted supplement is approved.

The approval-gated supplement wrapper now stops after H5 generation, RGB
rendering, review sheet creation, and structural inspection. The inspection
checks count/quota, `301/300` length, source kind, regrasp metadata for
`hole_late_fast_shift`, RGB manifest/JSONL/video consistency, and review sheet
presence. A passing inspection is only permission to visually inspect the
review sheets. It is not permission to merge, export WAM conditions, or start
SFT.

## Targeted Supplement Attempts

After the user continued the plan, the approval-gated supplement path was
started inside the held tmux/Slurm allocation `127723` on `server54`.
All foreground runs were stopped with Ctrl-C in tmux; the allocation was not
released.

Attempts:

- `targeted_recovery_supplement_after_approval_20260614_131231`:
  intended 112 rows, stopped after `57` attempts, `0` accepted.
  Main reject reasons: `sample_final_pose_failed=25`,
  `target_motion_swept_wall_collision_risk=19`,
  `target_self_insert_after_motion_gate_failed=7`,
  `planner_final_insert_failed=4`.
- `targeted_recovery_supplement_after_approval_20260614_131700_pilot8`:
  8-row pilot after smaller target-motion range and fast-shift anti-gate
  exemption, stopped after `22` attempts, `0` accepted.
  Main reject reasons: `target_motion_swept_wall_collision_risk=10`,
  `planner_final_insert_failed=5`,
  `target_self_insert_after_motion_gate_failed=4`.
- `targeted_recovery_supplement_after_approval_20260614_132100_pilot8_retreat`:
  8-row pilot after retreating the initial old-hole wait pose by `0.06 m`,
  stopped after `23` attempts, `0` accepted.
  Main reject reason: `initial_preinsert_gate_failed=20`.
- `targeted_recovery_supplement_after_approval_20260614_132400_pilot8_waitgate`:
  8-row pilot after adding a separate initial wait-pose line-yz gate of `0.008 m`,
  stopped after `28` attempts, `0` accepted.
  Main reject reasons: `target_motion_swept_wall_collision_risk=11`,
  `planner_final_insert_failed=11`,
  `target_self_insert_after_motion_gate_failed=4`.

Interpretation:

This is a data/teacher-geometry blocker, not a Cosmos training blocker. The
current scripted hard-teacher path cannot construct the targeted moving-hole
recovery supplement: when the peg is near the old hole, target motion frequently
sweeps the moved wall through the peg line; moving the waiting peg farther back
changes the failure to initial wait-pose or final insertion geometry. Since no
H5 row was accepted, no RGB review sheet exists and there is nothing visually
approvable to merge.

Boundary:

Do not rerun these same variants. No supplement rows, RGB review sheets,
merged source, WAM export, or full SFT resulted from these attempts. Full
4-GPU SFT remains blocked by the original live-query coverage failure.

## Failed-State Snapshot Repair

After the hard-teacher zero-acceptance result, the next question was whether
the existing failed closed-loop artifacts could be used directly as restore
states for recovery-teacher data.

Inspected root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/hard_case_screen2_20260614/cosmos_iter2700_puredp_fail_4_9_10_11_12_13`

Findings:

- the failed samples contain per-query `iter_##_prefix_f###` folders;
- each query folder has `observed_prefix.mp4` and
  `live_history_raw_action_state.json`;
- `live_history_raw_action_state.json` contains only an `action` array with
  `300` rows and `32` WAM dimensions;
- the loop summary records real relative errors, controller mode, action chunk
  paths, and final success/failure;
- no full live simulator `env_state`, `state_dict`, or restore snapshot exists
  in the old failed artifacts.

Plain interpretation:

The existing artifacts explain the data gap, but they cannot directly restore
the simulator at the failed post-motion contact states. The source H5 files do
contain `env_states`, but those are the original source trajectory states, not
the real Cosmos-closed-loop failed states.

Code prepared:

- `scripts/world_model/run_cosmos3_live_receding_loop.py` now supports
  `--save-live-state-snapshots`;
- `scripts/world_model/run_cosmos3_live_receding_panel.py` passes the option
  through to each sample loop;
- `scripts/slurm/run_cosmos3_live_receding_loop_in_allocation.sh` and
  `scripts/slurm/run_cosmos3_live_receding_panel_in_allocation.sh` expose
  `SAVE_LIVE_STATE_SNAPSHOTS=true`;
- when enabled, each receding iteration writes
  `live_state_before_controller.h5` and `live_state_after_controller.h5`;
- each snapshot stores real `base_env.get_state_dict()` under `state/` and
  real `base_env.get_obs()` under `observation/`, with prefix frame,
  iteration, label, and boundary attributes.

Verification:

- `python -m py_compile` passed for the modified Python scripts;
- `bash -n` passed for the modified Slurm wrappers.

Boundary:

This is not training, not method success, and not a controller input. It is a
recording repair so the next approved run can capture true failed-state restore
points. After that, a separate failed-state recovery teacher can be written to
start from those real states, produce visually inspected recovery rows, and
only then feed a merged preflight/coverage gate.

New plan:

`PLAN/cosmos3_lowfreq_wm_executor/02_failed_state_recovery_supplement.md`

## No-Render Failed-State Replay

The direct `SAVE_LIVE_STATE_SNAPSHOTS=true` rerun was tried first, but the
source-restore diagnostic stalled at `live_source_restore_before_render_prefix`.
That localizes the immediate runtime issue to prefix rendering, not H5 loading,
env reset, or Cosmos inference.

To avoid depending on that render path, a no-render replay tool was added:

`scripts/world_model/replay_live_history_state_snapshots.py`

It replays the recorded live-loop robot actions from the old failed summaries,
applies the same source target motion, and saves simulator states at the old
query frames. It does not run Cosmos and does not render videos.

Replay outputs:

- sample_09:
  `experiments/world_model_task_rebinding/cosmos3/replay_state_snapshots_sample09_20260614_141229`
- samples 04 and 10-13:
  `experiments/world_model_task_rebinding/cosmos3/replay_state_snapshots_hard2_remaining_20260614_141435`

Replay fidelity:

- sample_09 old/new final `peg_head_pos_at_hole` matched exactly:
  `[-0.1239014864, -0.0626012236, -0.0517419614]`, L2 `0.0`,
  `28` query snapshots.
- sample_10 old/new final matched exactly:
  `[-0.0833214521, -0.1241385639, -0.0428964943]`, L2 `0.0`.
- sample_11 old/new final matched exactly:
  `[-0.1171780229, -0.0385717675, -0.0095921084]`, L2 `0.0`.
- sample_12 old/new final matched exactly:
  `[-0.0963682532, -0.0224182308, -0.0030372739]`, L2 `0.0`.
- sample_13 old/new final matched exactly:
  `[-0.0807159841, -0.0016521737, 0.0043166727]`, L2 `0.0`.
- sample_04 also matched exactly but was `success=True`, so it is not a
  negative recovery seed.

Total written `frame_***_live_state.h5` files across 04 and 09-13: `163`.
Usable failed-state restore candidates are from 09-13.

Boundary:

This is restore-state evidence only. It does not prove a controller, a world
model, or a recovery teacher works. It only proves that the old failed summaries
can be converted into faithful simulator restore states without using the
source H5 as a fake live state.

## Failed-State Recovery Teacher Smoke

A narrow smoke generator was added:

`scripts/world_model/generate_cosmos3_failed_state_recovery_teacher.py`

It restores a real `frame_***_live_state.h5`, marks the row as already
post-motion via `perturb.triggered` from step 0, then asks a motion-planning
teacher to regrasp, re-align, and insert.

Smoke output:

`experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_smoke_20260614_142222`

Result:

- requested accepted demos: `1`;
- accepted demos: `0`;
- attempts: `10`;
- tried frames: `296`, `288`, `280`, `272`;
- tried samples: 09, 10, and 13;
- no standardized H5 row, no RGB review sheet, no merge, no WAM export, and no
  SFT were produced.

Reject classes observed in the pane:

- `planner_preinsert_failed`
- `preinsert_gate_failed`
- `final_insert_wall_collision_risk`
- `planner_final_insert_refine_failed`

Interpretation:

The current blocker is now concrete: the recovery teacher cannot reliably
re-align and insert from the real failed closed-loop poses. This is not an
overfit/training failure and not a Cosmos SFT launch problem. Full 4-GPU SFT
remains blocked because there are still no visually approvable recovery rows to
merge and re-audit.

Follow-up teacher variants:

- staged preinsert smoke
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_stage_smoke_20260614_143157`:
  accepted `0/1` after `20` attempts;
- existing-grasp-first smoke
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_existing_grasp_smoke_20260614_143904`:
  accepted `0/1` after `24` attempts;
- multi-stage stronger refinement smoke
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_multi_stage_smoke_20260614_144327`:
  accepted `0/1` after `24` attempts over real failed snapshots from samples
  09, 10, and 13.
- release/regrasp-after-stage smoke
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_release_regrasp_smoke_20260614_1510`:
  accepted `0/1` after `9` attempts.
- shallow release/regrasp smoke
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_release_regrasp_shallow_smoke_20260614_1516`:
  accepted `0/1` after `9` attempts.
- counterfix shallow release/regrasp smoke
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_release_regrasp_shallow_counterfix_smoke_20260614_1525`:
  accepted `0/1` after `3` attempts.
- close-case stronger-refine smoke
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_release_regrasp_sample13_refine_smoke_20260614_1532`:
  accepted `0/1` after `1` attempt.
- task-axis release-regrasp smoke
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_goal_y_regrasp_smoke_20260614_1542`:
  accepted `0/1` after `3` attempts.
- planned task-axis regrasp smoke
  `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_planned_goal_y_regrasp_smoke_20260614_1548`:
  accepted `0/1` after `3` attempts.

Plain interpretation of the follow-ups:

The restore-state source is usable, but the positive-data teacher is not. The
multi-stage run can sometimes move the peg head nearer to the hole, but the
whole peg line/axis remains misaligned or the planner cannot complete a safe
preinsert/final insert. So the next blocker is not "train Cosmos longer"; it is
"build a different recovery teacher/data mechanism that produces visually real
held-peg insertion from these failed states."

The release/regrasp-after-stage path was added to address the bad-grasp
diagnosis directly: stage the peg, release the old grasp, regrasp, then insert.
This improved some geometry but still produced no accepted row. The closest
counterfix case had `peg_head_at_hole ~= [-0.0176, 0.00284, 0.00296]`, no wall
risk, and peg-axis cosine `~0.9955`, but the full peg line missed the strict
centerline gate (`~0.0156 m` versus `~0.003 m`) and live success was false.

Implementation fix during this test:

`generate_cosmos3_failed_state_recovery_teacher.py` now measures `raw_steps` and
`final_insert_start_step` relative to the current recovery attempt rather than
the global `RecordEpisode` counter. This fixed a false `raw_steps_exceed_limit`
path, but did not produce valid rows.

Additional task-axis regrasp test:

The teacher now has default-off `planned-regrasp-mode` and
`release-regrasp-mode` options so a smoke can use the hole/task-frame closing
axis instead of the live TCP closing axis. This tested whether the old regrasp
was just preserving a bad gripper orientation. It did not solve the data
blocker: the `goal_y` release-regrasp smoke and the forced planned `goal_y`
regrasp smoke both accepted `0` rows. The forced planned `goal_y` path failed
early regrasp for two sine cases and made the fast-shift staged peg line worse,
so this is not a viable patch by itself.

Verification:

- `python -m py_compile` passed for
  `scripts/world_model/replay_live_history_state_snapshots.py`,
  `scripts/world_model/generate_cosmos3_failed_state_recovery_teacher.py`, and
  the modified live-loop code.
- All compute above ran inside tmux-held Slurm allocation `127723`; no
  training/data generation was run on the login node.
