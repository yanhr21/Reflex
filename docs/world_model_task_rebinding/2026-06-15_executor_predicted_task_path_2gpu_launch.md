# 2026-06-15 Executor Predicted Task-Path 2-GPU Launch

## Boundary

The user changed the formal full-training floor from `4` GPUs for `3` hours to
`2` GPUs for `3` hours. If a valid `2` GPU allocation is available first, use it
instead of waiting for `4` GPUs.

This note is for the low-frequency Cosmos WM plus high-frequency
executor/DP-prior residual branch. It is not a new error-detection/recovery
case-mining branch, and it is not a repeat of the failed direct raw-Cosmos-action
closed loop.

## Why This Step Exists

The dense full Cosmos SFT already trained through `iter_000001500` and passed
strict generated-artifact checks, but live closed-loop execution stayed `0/4`.
The concrete failure is that raw Cosmos action chunks plus threshold DP handoff
do not place the held peg into a stable DP-continuable insertion state after the
target moves.

The next method step is to use Cosmos at low frequency as a task/world predictor
and train a smaller executor to follow a causal predicted task path while staying
near the frozen DP action prior. Full executor training cannot use
`gt_state_targets_debug`; it needs Cosmos-predicted sidecars from current/prefix
conditions.

## Resource Evidence

- Slurm job: `128023`
- tmux session: `cosmos3_executor_formal_2gpu_20260615`
- node: `server54`
- allocation: `gres:gpu:2`, time limit `1-00:00:00`
- CUDA canary: `.venv` reported `cuda_available=True`, `device_count=2`, both
  devices `NVIDIA H200`

This satisfies the updated resource floor for later formal training. The current
running step is prediction/data preparation for that training, not final method
evidence.

## Input Manifest

Executor-targeted Cosmos input root:

`experiments/world_model_task_rebinding/cosmos3/executor_wam_eval_inputs_20260615_executor_wam_inputs64_diverse`

Manifest summary:

- `strict_eval_input_ok=true`
- `num_selected_samples=64`
- `selection_policy=role_scenario_round_robin`
- role counts: `insert_resume=25`, `target_motion_observed=18`,
  `target_post_motion=21`
- scenario counts: `hole_late_constant=9`,
  `hole_late_continuous_insert=9`, `hole_late_fast_shift=9`,
  `hole_late_move_stop=9`, `hole_late_reverse=9`, `hole_late_sine=8`,
  `peg_disturb=4`, `peg_drop=5`, `none=2`

The previous first-row selector produced only `hole_late_move_stop` coverage for
the first 32 train rows. The round-robin selector fixes that input-preparation
problem before spending GPU time on Cosmos prediction.

## Running Prediction Pass

Dense SFT checkpoint:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/checkpoints/iter_000001500`

Executor-targeted prediction output root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/executor_wam_eval_iter_000001500_train64_diverse_20260615_server54`

Launch settings:

- `SKIP_BUILD_EVAL_INPUTS=true`
- `EVAL_INPUT_JSONL=.../executor_wam_eval_inputs_20260615_executor_wam_inputs64_diverse/inputs/train_executor_wam_policy_samples.jsonl`
- `NPROC_PER_NODE=2`
- `INFERENCE_NUM_STEPS=30`

Startup evidence so far:

- wrapper manifest written
- prebuilt input accepted
- distributed runtime initialized with `2` GPUs
- Cosmos inference loaded `64` samples

Completed 64-row evidence:

- `sample_outputs.json`: `64/64`
- strict inspection: `strict_eval_artifacts_ok=true`
- strict failures: `[]`
- predicted task-path dataset:
  `experiments/world_model_task_rebinding/cosmos3/executor_dataset_cosmos_predicted_task_path_iter1500_20260615_pred_task_path_train64_diverse`
- rows written: `64`
- task path source: `cosmos_predicted_action_sidecar`
- diagnostic GT RMSE: mean `0.0479531`, max `0.222928`
- matched DP-prior export:
  `experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_smoke_20260615_pred_task_path_dp_prior_train64_diverse`
- DP-prior rows: `64`
- no-GT two-sample gate:
  `experiments/world_model_task_rebinding/cosmos3/executor_overfit_smoke_20260615_executor_overfit2_pred_task_path_train64_diverse`
- no-GT gate final MSE: `1.74878e-08`

This proves the causal executor interface can be built without GT debug task
paths. It is still too small to be the formal training set.

## 512-Row Expansion

The same `2` GPU allocation completed a broader executor-targeted Cosmos
prediction pass:

Input root:

`experiments/world_model_task_rebinding/cosmos3/executor_wam_eval_inputs_20260615_executor_wam_inputs512_diverse`

Input manifest:

- `strict_eval_input_ok=true`
- `num_selected_samples=512`
- role counts: `insert_resume=169`, `target_motion_observed=150`,
  `target_post_motion=193`
- scenario counts: `hole_late_constant=75`,
  `hole_late_continuous_insert=75`, `hole_late_fast_shift=75`,
  `hole_late_move_stop=65`, `hole_late_reverse=75`, `hole_late_sine=75`,
  `peg_disturb=20`, `peg_drop=50`, `none=2`

Output root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/executor_wam_eval_iter_000001500_train512_diverse_20260615_server54`

Completed 512-row evidence:

- strict inspection: `strict_eval_artifacts_ok=true`
- generated samples inspected: `512`
- strict failures: `0`
- predicted task-path dataset:
  `experiments/world_model_task_rebinding/cosmos3/executor_dataset_cosmos_predicted_task_path_iter1500_20260615_pred_task_path_train512_diverse`
- rows written: `512`
- task path source: `cosmos_predicted_action_sidecar`
- diagnostic GT RMSE: mean `0.0567245`, max `0.261960`
- matched DP-prior export:
  `experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_smoke_20260615_pred_task_path_dp_prior_train512_diverse`
- DP-prior rows: `512`
- DP-prior failures: `{}`

The dataset summary still says `ready_for_formal_executor_training=false`
because that builder only checks the sidecar dataset before the DP-prior join.
The actual formal trainer accepted the sidecar dataset plus the matched
DP-prior JSONL, and its manifest records `task_path_sources=[
"cosmos_predicted_action_sidecar"]`.

## Formal Trainer Entry

New entry points:

- `scripts/world_model/train_cosmos3_executor_residual.py`
- `scripts/slurm/run_cosmos3_executor_residual_train_in_allocation.sh`

The trainer uses causal Cosmos-predicted task paths plus frozen-DP action prior,
rejects GT debug task paths, supports `torchrun`/DDP, and records the updated
formal floor of `2` GPUs for at least `3` hours.

## Formal Training Launch

Formal residual-executor training is now running in the same tmux-held Slurm
allocation:

- Slurm job: `128023`
- tmux session: `cosmos3_executor_formal_2gpu_20260615`
- node: `server54`
- output root:
  `experiments/world_model_task_rebinding/cosmos3/executor_residual_train_20260615_executor_residual_train512_formal_2gpu_server54`
- `NPROC_PER_NODE=2`
- `MIN_WALL_SECONDS=10800`
- `MAX_WALL_SECONDS=14400`
- `MIN_STEPS=1000`
- `BATCH_SIZE=512`
- `HIDDEN_DIM=4096`
- `NUM_LAYERS=6`

The training manifest records `world_size=2`, `formal_min_gpus=2`,
`min_wall_seconds=10800`, `num_samples=512`, `num_train=435`, `num_val=77`,
and `task_path_sources=["cosmos_predicted_action_sidecar"]`.

Early metric snapshot:

- step `10400`
- train action MSE: `5.74266e-06`
- validation action MSE: `0.00989754`
- frozen-DP-prior validation MSE: `0.00156083`

Plain interpretation: the run has started correctly and is using the right
causal inputs, but it has not yet shown that the residual improves over the
frozen DP prior on held-out rows. The next gate is to let this run cross the
required `3` hour training floor, inspect the saved summary/checkpoint, and
only then decide whether closed-loop eval is justified. If validation remains
worse than the DP prior, the blocker is likely executor target/generalization,
not Slurm startup, not 301/300 length accounting, and not the old raw-action
closed-loop gate.

## Closed-Loop Interface Prepared

While the formal training is running, the live loop/panel was extended for the
executor branch:

- `scripts/world_model/run_cosmos3_live_receding_loop.py`
- `scripts/world_model/run_cosmos3_live_receding_panel.py`
- `scripts/slurm/run_cosmos3_live_receding_loop_in_allocation.sh`
- `scripts/slurm/run_cosmos3_live_receding_panel_in_allocation.sh`

New controls:

- `--controller-action-source=residual_executor`
- `--executor-checkpoint <checkpoint_final.pt or checkpoint_latest.pt>`
- wrapper env: `CONTROLLER_ACTION_SOURCE=residual_executor`
- wrapper env: `EXECUTOR_CHECKPOINT=<checkpoint>`

In this branch, Cosmos still predicts from the causal observed prefix, but its
raw robot-action columns are not executed. The loop extracts the Cosmos
predicted task-state sidecar, computes a frozen-DP prior chunk from the current
live observation history, and lets the residual executor output the robot
action chunk. The review video timeline marks these frames as
`EXECUTOR_ACTIVE`.

Syntax checks for the modified Python files and wrappers passed inside Slurm
allocation `128023` on `server54`.

This only prepares the evaluation path. It is not permission to run live eval
before the formal training summary exists and the `2` GPU / `3` hour floor is
met.

## Mid-Run Monitor

At `2026-06-15T07:25:57+08:00`, training was still active and below the
required `3` hour floor.

- latest observed step: `48200`
- latest train action MSE: `3.58452e-06`
- latest validation action MSE: `0.0117111`
- frozen-DP-prior validation MSE: `0.00156083`
- best observed validation action MSE so far: `0.00493207` at step `23000`
- latest GPU utilization probe: `76%` / `23%`

This is not a completed result. The current risk is that the residual executor
is fitting the training rows but still hurting held-out action prediction
relative to simply using the frozen DP prior. If that remains true after the
post-3-hour summary, the correct blocker is executor training/generalization,
not missing closed-loop video.

## Split Diagnostic

A compute-node split diagnostic was written to:

`experiments/world_model_task_rebinding/cosmos3/executor_residual_train_20260615_executor_residual_train512_formal_2gpu_server54/split_baseline_diagnostic.json`

Key facts:

- train split DP-prior/teacher MSE: `0.00639773`
- validation split DP-prior/teacher MSE: `0.00156083`
- validation `target_motion_observed` DP-prior MSE: `0.000144541`
- validation `peg_drop` DP-prior MSE: `0.00959632`
- train `target_post_motion` DP-prior MSE: `0.0127695`
- train `peg_drop` DP-prior MSE: `0.0418070`

Plain interpretation: the training split contains many rows where the teacher
differs substantially from the frozen DP prior, especially `target_post_motion`
and peg perturbation rows. The validation split is much easier for the DP prior
overall. A residual model trained to change the DP prior can therefore fit
train rows while hurting held-out rows where the best action is close to "do
almost nothing beyond DP." This is a likely executor generalization/training
objective problem if the post-3-hour summary remains below gate.

Follow-up monitor at `2026-06-15T07:35:35+08:00`: training was still below the
formal `3` hour floor. Latest step `83000` had train action MSE `3.52990e-06`,
validation action MSE `0.0129802`, and the same DP-prior validation baseline
`0.00156083`. Best validation MSE remained `0.00493207` at step `23000`.

## Residual-Scale Gate Prepared

Added post-floor calibration entry:

`scripts/world_model/calibrate_cosmos3_executor_residual_scale.py`

The live loop/panel now also accept:

- `--executor-residual-scale`
- wrapper env `EXECUTOR_RESIDUAL_SCALE`

Purpose: check after training whether the residual direction is useful but too
large. The calibration sweeps scales in `[0, 1]` on the held-out split. Scale
`0` is the frozen-DP prior baseline and must not be reported as residual
executor method success. A live residual-executor panel is allowed only if a
positive scale beats the DP-prior validation baseline after the formal training
floor is met.

Compute-node syntax checks passed for the calibration script and modified
live-loop wrappers inside allocation `128023`.

A lightweight watcher was started in tmux session
`cosmos3_executor_residual_calibrate_watch_20260615`, Slurm step `128023.51`.
It waits for `training_summary.json` and `checkpoint_final.pt`, then writes
calibration outputs under:

`experiments/world_model_task_rebinding/cosmos3/executor_residual_train_20260615_executor_residual_train512_formal_2gpu_server54/residual_scale_calibration_final`

It does not launch closed-loop eval.

## Formal 2-GPU Training Result

At `2026-06-15T10:16:21+08:00`, the formal residual-executor training summary
existed and recorded:

- `world_size=2`
- `elapsed_seconds=10800.012`
- `formal_training_floor_met=true`
- `steps=642452`
- final validation action MSE: `0.0183239`
- frozen-DP-prior validation MSE: `0.00156083`
- `ready_for_closed_loop_eval=false`

Plain interpretation: the current user floor was satisfied, but the unscaled
residual executor is bad offline. It fits train rows and changes held-out DP
actions too strongly. The unscaled checkpoint must not be used as method
evidence.

The residual-scale calibration then completed under:

`experiments/world_model_task_rebinding/cosmos3/executor_residual_train_20260615_executor_residual_train512_formal_2gpu_server54/residual_scale_calibration_final`

Calibration result:

- scale `0`: DP-prior baseline validation MSE `0.00156083`
- best positive scale: `0.05`
- validation MSE at scale `0.05`: `0.00151089`
- delta versus DP prior: `-4.99e-05`
- `ready_for_scaled_closed_loop_eval=true`

This is a very small offline improvement, not a strong result. It only permits
one conservative residual-executor live panel with `EXECUTOR_RESIDUAL_SCALE=0.05`.
The live panel still needs real final-state metrics plus video/contact-sheet
inspection before it can count as dynamic task evidence.

## First Residual-Executor Live Panel

Launched inside held Slurm allocation `128023` on `server54`, not on the login
node:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_executor_residual_iter1500_panel4_20260615_1017_residual_scale005`

Configuration:

- `controller_action_source=residual_executor`
- `executor_residual_scale=0.05`
- `sample_indices=0,1,3,4`
- `run_cosmos_inference=true`
- `prefix_start_mode=target_motion_onset`
- `pretrigger_control_mode=frozen_dp_until_target_motion`
- `panel_full_episode_contract_ok=true`
- `failed_process_count=0`
- `final_success_count=2`
- every final observed rollout video decoded to `301` frames at `30 fps`

Per-sample final real-state results:

- sample00 `hole_late_move_stop`: success, final peg-head-at-hole
  `[-0.00677, 0.00248, -0.00142]`
- sample01 `hole_late_constant`: failure, final peg-head-at-hole
  `[-0.07927, -0.00030, 0.00461]`
- sample03 `hole_late_fast_shift`: success, final peg-head-at-hole
  `[0.00876, -0.00125, 0.00262]`
- sample04 `hole_late_sine`: failure, final peg-head-at-hole
  `[-0.10365, 0.02230, -0.02717]`

Visual review:

Opened
`live_receding_panel_contact_sheet.png`. The contact sheet matches the metrics:
sample00 and sample03 visually finish near/through the hole area after
executor-active rebind and DP handoff; sample01 remains short in insertion
depth despite good lateral alignment; sample04 is visibly misaligned late in
the sine-motion rollout. This is a real improvement over the recorded direct
raw-Cosmos-action `0/4` panel, but it is not full method evidence. The summary
itself records `method_evidence_allowed=false` because this is still a small
panel.

Immediate next check: run the same calibrated residual-executor protocol on
unused eval samples `2,5,6,7` to test whether the improvement is more than the
fixed four-sample comparison.

## Second Residual-Executor Live Panel

Launched inside the same held allocation `128023` on `server54`:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_executor_residual_iter1500_panel4b_20260615_1224_residual_scale005_samples2_5_6_7`

Configuration stayed the same as the first panel:

- `controller_action_source=residual_executor`
- `executor_residual_scale=0.05`
- `sample_indices=2,5,6,7`
- `run_cosmos_inference=true`
- `prefix_start_mode=target_motion_onset`
- `pretrigger_control_mode=frozen_dp_until_target_motion`
- `panel_full_episode_contract_ok=true`
- `failed_process_count=0`
- `final_success_count=1`
- every final observed rollout video decoded to `301` frames

Per-sample final real-state results:

- sample02 `hole_late_reverse`: success, final peg-head-at-hole
  `[0.00699, -0.00299, -0.00297]`
- sample05 `hole_late_continuous_insert`: failure, final peg-head-at-hole
  `[-0.20720, 0.03044, -0.01952]`
- sample06 `hole_late_continuous_insert`: failure, final peg-head-at-hole
  `[-0.01650, -0.00081, 0.00262]`
- sample07 `peg_drop`: failure, final peg-head-at-hole
  `[-0.10438, -0.00380, -0.01246]`

Visual review:

Opened
`live_receding_panel_contact_sheet.png`. The contact sheet matches the metrics.
Sample02 visibly reaches the hole. Sample05 drifts far from the hole during the
continuous-insert motion. Sample06 gets close but does not visibly complete
insertion. Sample07 never leaves the DP scan/monitor behavior:
`wm_active_frame_count=0`, so the peg-drop failure is a trigger/coverage
failure rather than a measured executor insertion attempt.

Two-panel summary:

- first panel: `2/4`
- second panel: `1/4`
- combined calibrated residual-executor small-panel result: `3/8`

Plain interpretation: the calibrated residual executor is a real improvement
over direct raw-Cosmos actions on the fixed panel (`0/4` -> `2/4` there), and
it also succeeds on one new unused sample. It is not stable enough to be the
final executor. The repeated failure mode is contact/insertion progress after
task-frame rebinding: the controller can sometimes reduce lateral hole error
but does not reliably maintain the right insertion-axis/contact behavior.
`peg_drop` also needs a perturbation trigger path, because current target-motion
detection does not activate the WM/executor when the target is static but the
peg state changes.

## Contact/Progress Label Exporter Smoke

Following the DDP/HDP-inspired repair plan, added:

`scripts/world_model/export_cosmos3_contact_progress_labels.py`

Purpose:

- export per-frame contact phase;
- export peg-head-at-hole, lateral progress, insertion progress, and combined
  contact progress;
- export grasp/inserted/robust-held labels;
- export DP-continuability labels using the existing static-DP continuability
  stats;
- export per-row prefix/chunk labels for future contact-conditioned executor
  training.

Boundary:

These labels are supervision, scoring, and diagnosis targets. They must not be
fed as future ground-truth controller conditions during live evaluation.

Smoke run:

`experiments/world_model_task_rebinding/cosmos3/contact_progress_labels_smoke_20260615_executor_repair`

Run location: inside held Slurm allocation `128023` on `server54`.

Result:

- `num_episodes=2`
- `num_row_labels=16`
- `row_labels_by_split={"train": 8, "val": 8}`
- `ready_for_contact_executor_dataset=true`
- phase counts include `far`, `lateral_align`, `preinsert_aligned`,
  `dp_continuable`, `inserted`, and `lost_grasp`

The smoke used only a small capped subset, and the two unique episodes came
from duplicated early rows. It proves the exporter can read the current H5
schema and produce the labels needed for the next executor dataset gate.

Full export:

`experiments/world_model_task_rebinding/cosmos3/contact_progress_labels_full_20260615_executor_repair`

The first full attempt hit an HDF5 read-only file-locking failure:
`No locks available`. This was a filesystem locking issue, not a data-schema
failure. Rerunning the same read-only export with
`HDF5_USE_FILE_LOCKING=FALSE` succeeded inside the same held allocation.

Full result:

- `num_episodes=733`
- `num_row_labels=9271`
- `row_labels_by_split={"train": 8438, "val": 833}`
- `num_success_episodes=733`
- `ready_for_contact_executor_dataset=true`
- scenario coverage:
  - `hole_late_constant=48`
  - `hole_late_continuous_insert=96`
  - `hole_late_fast_shift=105`
  - `hole_late_move_stop=44`
  - `hole_late_reverse=99`
  - `hole_late_sine=60`
  - `none=160`
  - `peg_drop=119`
  - `peg_disturb=2`

This completes the first label gate for the contact/progress-conditioned
executor repair. It does not train the new executor yet.

## Contact Executor Dataset Join

Join script:

`scripts/world_model/build_cosmos3_contact_executor_dataset.py`

The script joins three causal sources by `uuid`:

- the 512-row Cosmos-predicted task-path executor dataset;
- the matched frozen static-DP prior chunks;
- the full contact/progress row labels from the 733 full-episode source set.

Output root:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_iter1500_train512_scale005_repair_20260615`

Run location: inside held Slurm allocation `128023` on `server54`.

Result:

- `num_executor_rows=512`
- `num_joined_rows=512`
- `missing_counts={}`
- `ready_for_contact_executor_training=true`
- phase counts: `far=94`, `lateral_align=142`,
  `preinsert_aligned=76`, `dp_continuable=200`
- future inserted within chunk: `185`
- future DP-continuable within chunk: `319`
- role counts: `insert_resume=169`, `target_motion_observed=150`,
  `target_post_motion=193`
- scenario counts include all active dynamic buckets, including
  `peg_drop=50` and `peg_disturb=20`

Plain interpretation: this removes the immediate data-join blocker for the
DDP/HDP-inspired executor repair. The available 512-row causal dataset has
both non-contact alignment phases and contact/DP-continuable phases. The next
real test is whether a contact/progress-conditioned executor can improve
offline progress/contact gates and then live insertion, not whether labels or
DP priors are missing.

## Contact Executor Trainer Debug Gate

Added:

`scripts/world_model/train_cosmos3_contact_executor.py`

The trainer keeps the causal boundary explicit. Its inputs are current state,
Cosmos-predicted task path, frozen-DP prior actions, and current contact
context. Future contact/progress labels are targets only. The trainer also
recomputes future inserted/DP-continuable/progress targets from the episode
contact label arrays at the actual executable action horizon. In the current
DP-prior interface that horizon is `8` steps, so it does not train against a
misaligned `24`-step future label.

Short overfit/debug run:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_overfit_smoke_20260615_train64_progress_horizon8`

Run location: inside held Slurm allocation `128023` on `server54`.

Result:

- `num_samples=64`
- train/eval split: `48/16`
- steps: `100`
- `ready_for_debug_gate=true`
- train action MSE: `4.91476e-06`
- train DP-prior action MSE baseline: `0.00890316`
- train progress MSE: `0.0003792`
- train inserted accuracy: `1.0`
- train DP-continuable accuracy: `1.0`
- eval action MSE: `0.00133105`
- eval DP-prior action MSE baseline: `0.000372038`
- eval progress MSE: `0.0287098`
- eval inserted accuracy: `1.0`
- eval DP-continuable accuracy: `0.875`

Plain interpretation: the contact/progress executor interface is trainable and
the labels/actions are wired correctly, but this is not generalization
evidence. On the small held-out split, action MSE is still worse than the
frozen DP prior. This supports proceeding to a formal contact/progress training
test only with the correct `2` GPU / `3` hour floor and no live eval unless the
formal offline gate justifies it.

DDP smoke:

- first `4`-step `2`-rank smoke wrote a valid summary but returned torchrun
  failure because `ready_for_debug_gate=false` was encoded as exit code `64`;
  that was fixed because experimental unready is a recorded result, not a
  program crash;
- second `2`-rank smoke completed with exit `0` under
  `experiments/world_model_task_rebinding/cosmos3/contact_executor_ddp_smoke_20260615_train16_2step_exitfix`.

## Formal Contact Executor Training Launch

Wrapper:

`scripts/slurm/run_cosmos3_contact_executor_train_in_allocation.sh`

tmux session:

`cosmos3_contact_executor_formal_2gpu_20260615`

Slurm allocation:

- job `128023`
- node `server54`
- `2` H200s

Output root:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54`

Launch boundary:

- no `sbatch`;
- launched through the existing tmux-held allocation;
- formal floor requested as `2` GPUs for `10800` seconds;
- closed-loop live eval remains gated on the post-floor training summary.

Startup evidence:

- CUDA canary passed: `torch 2.5.1+cu121`, `cuda_available=True`,
  `device_count=2`, both devices `NVIDIA H200`;
- `training_manifest.json`, `training_history.json`, and
  `checkpoint_latest.pt` were written;
- step `3200` was reached within the first minute, so the run is executing
  rather than waiting in model/data startup.

Early metrics at step `3200`:

- train action MSE: `1.07773e-05`
- train DP-prior baseline MSE: `0.00639773`
- eval action MSE: `0.00561778`
- eval DP-prior baseline MSE: `0.00156083`
- eval progress MSE: `0.00853074`
- eval inserted accuracy: `0.961039`
- eval DP-continuable accuracy: `0.87013`

Plain interpretation: the early formal run is learning the training rows and
its progress/contact heads are useful on the held-out split, but its held-out
action prediction is still worse than the frozen DP prior. This is the exact
watchpoint for the repair: if the same pattern holds after the required
`2` GPU / `3` hour floor, the next action should be analysis or a stronger
candidate/diffusion executor design, not a live-eval success claim.

Restart note:

The first formal launch above was intentionally interrupted inside the tmux
session before the formal floor because the initial `SAVE_EVERY_STEPS=1000`
would write a `~200MB` checkpoint every few seconds on this small model/data
loop. This was an implementation/resource hygiene issue, not a model result.
The held allocation was preserved and `scancel` was not used.

Current formal root:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k`

Changes:

- `SAVE_EVERY_STEPS=20000`
- `EVAL_EVERY_STEPS=2000`
- same `2` GPU / `10800` second formal floor

Current-root startup evidence:

- CUDA canary passed again with `2` H200s;
- Slurm step `128023.63` is active on `server54`;
- `training_manifest.json`, `training_history.json`, and
  `checkpoint_latest.pt` were written;
- step `4000` was reached within the first minute.

Early current-root metrics at step `4000`:

- train action MSE: `1.11474e-05`
- train DP-prior baseline MSE: `0.00639773`
- eval action MSE: `0.00597767`
- eval DP-prior baseline MSE: `0.00156083`
- eval progress MSE: `0.00861553`
- eval inserted accuracy: `0.961039`
- eval DP-continuable accuracy: `0.87013`

The same watchpoint remains: contact/progress readouts are learning, but
held-out action prediction is still worse than the frozen DP prior. No live
eval should run unless the post-floor summary gives a defensible offline gate.

Mid-run check before formal floor:

- step `44000`;
- elapsed `577.85` seconds;
- eval action MSE `0.008588`;
- eval DP-prior baseline MSE `0.001561`;
- eval progress MSE `0.008822`;
- eval inserted accuracy `0.948`;
- eval DP-continuable accuracy `0.870`.

This is still not a final formal result because the `10800` second floor has
not been met. It only reinforces the current watchpoint.

Later mid-run check before formal floor:

- step `92000`;
- elapsed `1238.76` seconds;
- eval action MSE `0.010402`;
- eval DP-prior baseline MSE `0.001561`;
- eval progress MSE `0.009180`;
- eval inserted accuracy `0.948`;
- eval DP-continuable accuracy `0.909`.

This still is not a final formal result, but it confirms the action-head
problem is persistent across many evaluations rather than one noisy checkpoint.

## Preliminary Group Inspection

Added:

`scripts/world_model/inspect_cosmos3_contact_executor_training.py`

This script reloads a contact-executor checkpoint, reconstructs the same eval
split, and reports action/progress/contact metrics by scenario, prefix role,
and current contact phase. It is offline inspection only and does not prove
live task success.

Preliminary run:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k/post_training_group_metrics.json`

Checkpoint inspected:

`.../contact_executor_train_20260615_formal_2gpu_server54_save20k/checkpoint_latest.pt`

Overall at the inspected checkpoint:

- eval rows: `77`
- action MSE: `0.005276`
- DP-prior action MSE: `0.001561`
- action MSE minus prior: `+0.003715`
- progress MSE: `0.008699`
- inserted accuracy: `0.948`
- DP-continuable accuracy: `0.857`

Worst phase groups:

- `lateral_align`: action MSE `0.01519` versus prior `9.24e-05`
- `preinsert_aligned`: action MSE `0.00530` versus prior `0.000245`

Better-than-prior phase:

- `far`: action MSE `0.00262` versus prior `0.00444`

Worst scenario groups:

- `peg_disturb`: action MSE `0.07018` versus prior `0.000256`
- `hole_late_move_stop`: action MSE `0.00936` versus prior `0.000822`

Better-than-prior scenario:

- `peg_drop`: action MSE `0.00545` versus prior `0.00960`

Plain interpretation: the contact/progress supervision is giving useful
readouts, but the deterministic action head is not yet a safe executor. It
overrides the frozen DP in phases where DP is already very accurate. This
supports the DDP/HDP lesson: the next executor should use DP as a candidate or
regularized prior and select/generate action chunks by predicted progress, not
replace the prior with one deterministic residual everywhere.

Scale-sweep extension:

The inspection script now also evaluates the learned residual at scales
`0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0`.

Latest checkpoint diagnostic:

- unscaled action head: action MSE `0.008319`;
- DP prior / scale `0`: action MSE `0.001561`;
- best global positive scale: `0.05`, action MSE `0.001526`;
- best phase scales:
  - `far=1.0`;
  - `dp_continuable=0.2`;
  - `preinsert_aligned=0.01`;
  - `lateral_align=0.0`;
- best scenario scales:
  - `hole_late_move_stop=0.0`;
  - `peg_disturb=0.0`;
  - `peg_drop=1.0`.

Plain interpretation: the action residual has some useful phase-specific
signal, but unscaled direct execution is unsafe. A tiny global scale only
barely beats the DP prior offline. This reinforces the next method direction:
DP must remain a candidate/regularized prior, with progress/contact scoring or
phase-aware selection deciding when any learned residual is allowed.

Formal gate correction:

The first contact-executor trainer implementation could mark
`ready_for_formal_eval=true` from progress/contact readouts alone. That is not
safe for this branch because the current failure is exactly that progress
readouts can look good while the action head is worse than the DP prior.

Fixes:

- `scripts/world_model/train_cosmos3_contact_executor.py` now requires
  `eval_action_mse <= eval_baseline_dp_prior_mse` before future summaries can
  set `ready_for_formal_eval=true`;
- added `scripts/world_model/check_cosmos3_contact_executor_formal_gate.py`
  as an independent post-final gate for the current already-running training
  process;
- updated
  `scripts/slurm/watch_cosmos3_contact_executor_formal_inspect.sh` so that
  after final summary and group inspection it writes
  `formal_live_eval_gate.json`.

Dry gate check before final summary correctly reports
`live_eval_allowed=false` because `training_summary.json` is still missing.
The watcher session was restarted after this change:
`cosmos3_contact_executor_formal_watch_20260615`.

This prevents the pipeline from running closed-loop videos just because the
contact/progress classifier is good while the actual action executor is bad.

History trend summary:

Added:

`scripts/world_model/summarize_cosmos3_contact_executor_history.py`

Current mid-run output:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k/training_history_summary.json`

Facts from the current history:

- `52/52` eval points have action MSE worse than the DP-prior baseline;
- best eval action MSE is still `1.74x` worse than DP prior;
- latest point at the time of the summary is `6.76x` worse than DP prior.

The formal watcher now runs this trend summary before final group inspection
and formal gate writing. This is to make the post-floor blocker report concrete
if the final gate remains false.

## Formal Completion Watcher

Added:

`scripts/slurm/watch_cosmos3_contact_executor_formal_inspect.sh`

tmux session:

`cosmos3_contact_executor_formal_watch_20260615`

Log:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k/formal_watch.log`

Behavior:

- poll every `300` seconds for `training_summary.json` and
  `checkpoint_final.pt`;
- once they exist, run
  `scripts/world_model/inspect_cosmos3_contact_executor_training.py` inside
  held allocation `128023`;
- do not launch live eval.

This is to make the post-floor result immediately inspectable. If the final
summary keeps the same action-generation failure, the correct next step is to
stop and report the concrete blocker rather than run another closed-loop panel.

## 14:45-14:52 Mid-Run Contact-Executor Status

The formal contact/progress executor run is still below the required
`10800` second floor, so this is not the final decision and it does not permit
live eval.

Current active root:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k`

Latest parsed trend at `2026-06-15T14:52+08:00`:

- latest history point: step `158000`, elapsed `2147.69` seconds;
- eval action MSE: `0.011609`;
- frozen-DP-prior eval MSE: `0.0015608`;
- latest action MSE ratio to DP prior: `7.44x` worse;
- eval progress MSE: `0.009928`;
- inserted / DP-continuable accuracies: `0.948` / `0.909`;
- all `80/80` eval history points so far have action MSE worse than DP prior;
- best point so far is still `1.74x` worse than DP prior.

A separate mid-run checkpoint inspection was written to:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k/post_training_group_metrics_midrun_1445.json`

That inspection confirms the phase-specific failure:

- overall action MSE `0.010988` versus DP prior `0.0015608`;
- best global positive residual scale is only `0.05`, action MSE `0.001508`;
- `far` improves over DP prior;
- `lateral_align` is much worse than DP prior:
  `0.03224` versus `0.0000924`;
- `preinsert_aligned` is much worse than DP prior:
  `0.01693` versus `0.0002445`.

Plain interpretation: the model is learning useful contact/progress readouts
and has some phase-specific residual signal, but the deterministic action head
is not a safe executor. It damages phases where the DP prior is already nearly
correct. If the final post-floor gate remains false, the concrete blocker is
not missing labels or closed-loop eval; it is action-generation structure. The
next aligned method should be phase/contact-conditioned candidate or diffusion
action generation with DP as a candidate/regularizer and progress/value
scoring before live execution.

Gate hardening added during this wait:

- `scripts/world_model/check_cosmos3_contact_executor_formal_gate.py` now
  rejects stale group metrics unless `post_training_group_metrics.json` was
  produced from `checkpoint_final.pt`;
- py-compile passed;
- a mid-run dry gate still correctly reports `live_eval_allowed=false` because
  `training_summary.json` is missing.

Selector diagnostic added to the inspection script:

`scripts/world_model/inspect_cosmos3_contact_executor_training.py`

The script now reports oracle offline selector diagnostics: what validation
action MSE would be if a candidate residual scale were selected by a grouping
such as current contact phase. This is not live-controller evidence because it
uses eval-set group best scales, but it tests whether the next
DDP/HDP-style candidate-selection direction has signal.

Mid-run latest-checkpoint output:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k/post_training_group_metrics_midrun_selector_1452.json`

Numbers:

- unscaled action MSE: `0.011475`;
- DP prior: `0.0015608`;
- best global positive scale: `0.05`, action MSE `0.001505`;
- oracle current-phase scale selector: action MSE `0.001107`;
- phase choices: `far=1.0`, `dp_continuable=0.2`,
  `lateral_align=0.0`, `preinsert_aligned=0.01`.

Plain interpretation: a universal contact-phase selector can preserve the
strong DP prior near insertion while using learned residual signal in phases
where it helps. This supports the next-method direction, but it still does not
permit live eval from the current deterministic action head before the formal
final gate.

Stricter train-calibrated selector diagnostic:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k/post_training_group_metrics_midrun_traincal_selector_1457.json`

This variant selects the residual scale on the train split and evaluates it on
the validation split. It is meant to distinguish true selector generalization
from validation-set oracle signal.

Result:

- naive train-calibrated global selector chooses `scale=1.0`;
- naive train-calibrated current-phase selector also chooses `scale=1.0` for
  all phases;
- validation action MSE remains `0.011726`, much worse than DP prior
  `0.0015608`;
- reason: the deterministic model has nearly memorized train actions, so train
  action MSE selects the full residual even though the full residual fails
  held-out action execution.

Plain interpretation: the oracle phase-selector signal should not be turned
into a naive train-MSE scale calibration. A usable next selector needs
regularization and held-out progress/value/contact-continuability scoring. The
next method cannot be "pick the scale that fits train actions"; that recreates
the same overfit action-head failure.
