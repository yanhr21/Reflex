# World Model TODO

## Active Foundation-WAM Boundary

- [ ] 2026-06-09 pickup-aware Cosmos/DROID WAM is the active world-model line.
      The self-trained object-slot Transformer entries below are historical
      diagnostics/interface scaffolding only and must not be used as active
      publishable WM evidence. The active SFT must observe the complete
      pickup/holding segment before rollout, then predict short equal-length
      chunks with RGB video, action rows, and task-state readout targets
      aligned. Current active root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full1000_maniskill_rgb_chunked_pickup_wam_joint_policy_droid_policy_20260609_pickup_chunked_joint_policy`.
      Current condition root:
      `experiments/world_model_task_rebinding/cosmos3/action_state_conditions_full1000_maniskill_default_regen_chunked_pickup_wam_joint_policy_20260609_pickup_chunked_joint_policy`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-09_pickup_chunked_cosmos3_wam_method_correction.md`.
      Before any controller handoff, produce generated chunk demos and metrics
      over the exact `129`-frame / `128`-action contract, then run receding
      DP integration only after visual review shows target reconstruction,
      peg/gripper continuity, and plausible action chunks. A CPU Cosmos
      inference preflight on 2026-06-09 confirmed the active eval path
      preserves `condition_frame_indexes_vision=[0..20]` and
      `condition_frame_indexes_action=[0..79]` through Cosmos' internal
      `sequence_plan`; this is required plumbing evidence, not model-quality
      evidence. The post-SFT evidence path now also includes
      `scripts/slurm/watch_cosmos3_chunked_wam_readout_after_action_eval.sh`,
      which waits for the `10` strict chunk demos and decodes generated RGB
      into target-motion onset, final target pose, future hole path, peg/TCP,
      grasp, and insertion diagnostics under
      `experiments/world_model_task_rebinding/cosmos3/task_state_readout_chunked_pickup_wam_joint_policy_20260609`.
      Preflight
      `scripts/world_model/preflight_cosmos3_chunked_readout_alignment.py`
      now verifies the selected val demo chunks before readout; the initial
      val indices `0..9` report `strict_alignment_ok=true`, exact
      `129`-frame chunks, `128x32` action rows, `129x56` state targets, and no
      source H5/video/window mismatch.
      Live check `2026-06-09T16:18+08:00`: job `122177` is still holding the
      H200 on `server38` and training has reached at least `iter 414`. The
      only checkpoint is `iter_000000300` with val loss `3.669465`; there is no
      `sft_completed`/`sft_failed` marker yet. The 10 generated chunk demos and
      readout metrics have not started because the action-eval watcher is
      correctly waiting for SFT completion.
      The waiting post-SFT demo/readout watcher sample set is now
      `0 1 2 7 8 9 48 49 14 24` rather than raw val indexes `0..9`, so the 10
      demos will cover target pre-motion, target-motion-observed/post-motion,
      `hole_reverse`, `hole_move_stop`, and peg recovery/disturbance. The
      readout H5 preflight runs with `HDF5_USE_FILE_LOCKING=FALSE` and has
      passed for this diverse set.
      Checkpoint update `2026-06-09T17:15+08:00`: `iter_000000600` exists, but
      its validation loss is `4.106751`, worse than `iter_000000300` at
      `3.669465`; current best-val selection remains `iter_000000300`.
      Training continues on the held H200 past `iter 612`.
      Checkpoint update `2026-06-09T18:36+08:00`: `iter_000000900` exists and
      reports validation loss `3.380800`, improving over `iter_000000300` and
      `iter_000000600`; current best-val selection is now `iter_000000900`.
      Live check `2026-06-09T18:55+08:00`: job `122177` remains on
      `server38` with one H200 held and `100%` GPU util; training has reached
      at least `iter 968`. No SFT completion/failure marker exists, so the
      10 generated chunk demos/readout metrics have not started yet.
      Controller integration must stay paused until those exact-length
      generated-video/action/readout artifacts exist and are visually reviewed.

## Dataset Loader

- [x] Add object-slot sequence dataset loader.
- [x] Normalize positions, quaternions, velocities, and actions.
- [x] Build task-frame features such as peg head in hole frame.
- [x] Add train/val split by trajectory. Full train/val/test seed-family split
      waits for larger shards.

Dataset utilities:

- `scripts/world_model/object_slot_dataset.py`
- `scripts/world_model/inspect_object_slot_dataset.py`
- `scripts/world_model/inspect_world_model_ensemble.py`
- Smoke inspection:
  `experiments/world_model_task_rebinding/object_state_world_model/dataset_inspect/smoke_plus_bounded.json`

## Baselines

- [x] Implement static-hole predictor.
- [x] Implement constant-velocity predictor.
- [x] Implement last-observation-with-uncertainty baseline.

Baseline evaluator:

- `scripts/world_model/evaluate_hole_prediction_baselines.py`
- Smoke results:
  `experiments/world_model_task_rebinding/hole_prediction_baselines/original_smoke/baseline_eval.md`
- Bounded pilot results:
  `experiments/world_model_task_rebinding/hole_prediction_baselines/bounded_smoke/baseline_eval.md`
- Summary:
  `docs/world_model_task_rebinding/2026-06-02_hole_prediction_baselines.md`
- Last-observation uncertainty rerun:
  `experiments/world_model_task_rebinding/hole_prediction_baselines/with_lastobs_uncertainty/baseline_eval.md`

## Model

- [x] Implement first causal Transformer model.
- [x] Add multi-horizon prediction heads.
- [x] Add grasped/inserted predicate heads.
- [x] Add dropout uncertainty for validation-time calibration checks.
- [x] Save final and best checkpoints, validation summaries, and prediction
      examples.
- [x] Complete compliant Slurm training run and inspect metrics. Job `94371`
      completed but is undersized under the current 4-H200/3-hour rule and is
      not method evidence. Job `94442` completed as a compliant 4-GPU H200
      ensemble run; post-training inspection job `94620` passed with
      `compliant_training_evidence=true`.
- [x] Complete larger-shard Slurm training run after dynamic shard `94510`
      completes. Primary job `94539` used
      `scripts/slurm/train_object_state_world_model_from_rollout_dir_4gpu.sbatch`,
      validated discovered rollout H5s, enforced `MIN_TRAIN_SECONDS=10800`,
      completed on one node / 4 H200 GPUs, and passed strict post-training
      inspection `94863` with `compliant_training_evidence=true`. Prediction
      evaluation `94864` also completed. The first inspection `94619` was
      canceled before running because its submitted script snapshot lacked the
      compliance gate. Backfill duplicate `94679`, fast-fit duplicate `94746`,
      historical node-observation retry `94809`, and their attached inspections/evals
      were canceled after primary strict inspection `94863` passed. This avoids
      spending additional 4-H200 allocations on duplicate training evidence.

Training utilities:

- `scripts/world_model/train_object_state_world_model.py`
- `scripts/world_model/evaluate_object_world_model_ensemble.py`
- `scripts/slurm/train_object_state_world_model.sbatch` is now smoke-only and
  refuses by default unless `ALLOW_UNDERSIZED_SMOKE=true`.
- `scripts/slurm/train_object_state_world_model_ensemble_4gpu.sbatch`
- `scripts/slurm/train_object_state_world_model_from_rollout_dir_4gpu.sbatch`
- `scripts/slurm/inspect_world_model_ensemble.sbatch`
- `scripts/slurm/evaluate_object_world_model_ensemble.sbatch`
- 2026-06-02 17:56 RGB-D-derived eval feature-contract hardening:
  `scripts/world_model/evaluate_object_world_model_ensemble.py` now refuses
  prediction evaluation when eval `dataset_meta.feature_names` do not exactly
  match the training member manifests. With `--require-rgbd-derived`, it also
  requires eval RGB-D predicted-slot evidence and RGB-D auxiliary
  uncertainty/probability feature evidence. This is a refusal gate only; it
  does not change metrics or make any RGB-D method claim. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_eval_feature_contract_queue_1756.md`.
- 2026-06-02 18:03 online controller world-model contract hardening:
  `scripts/world_model/evaluate_rebinding_controller.py` now refuses
  `slot_source=rgbd` world-model control unless the inspected ensemble proves
  RGB-D-derived training and every loaded member manifest preserves the
  RGB-D-predicted-slot feature contract with RGB-D uncertainty/probability
  auxiliary inputs. This prevents a direct Python run from pairing RGB-D slot
  predictions with a state-trained world model and presenting it as RGB-D
  controller evidence. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_controller_python_gate_1803.md`.
- Undersized code-path smoke: `94371`, completed in 23 seconds on `server27`;
  do not use it to judge the research direction.
- Compliant training job: `94442`, completed on `server53`, 4 H200 GPUs,
  `MIN_TRAIN_SECONDS=10800`, elapsed `03:00:34`, inspection `94620` passed.
- Larger-shard primary training job: `94539`, completed on `server21`,
  4 H200 GPUs, `MIN_TRAIN_SECONDS=10800`, Slurm elapsed `03:01:17`,
  strict inspection `94863` passed, and prediction evaluation `94864`
  completed.
- 2026-06-02 09:45 queue audit: `94539` has actually started on `server21`
  with 4 H200 GPUs and elapsed `01:34:43`; it remains below the 3-hour
  evidence floor and has only intermediate `best_model.pt` files so far, not
  final `model.pt`/`metrics.json`.
- 2026-06-02 09:52 queue audit: `94539` remains running with Slurm elapsed
  `01:41:58`, still below `MIN_TRAIN_SECONDS=10800`. Log output continues to
  print member training progress; stderr contains only the expected Transformer
  nested-tensor warnings. No final `model.pt` or `metrics.json` exists yet.
- 2026-06-02 10:07 queue audit: `94539` is still running on `server21` with
  `JOB_GRES=gpu:NVIDIAH200:4`, Slurm elapsed `01:57:21`, and `TimeLimit=03:30:00`.
  Node `server21` reports `Gres=gpu:NVIDIAH200:8`. This is the right hardware
  class, but it is still below the 3-hour training-evidence floor, so it
  cannot be used to judge the method until final checkpoints exist and strict
  inspection `94863` reports `compliant_training_evidence=true`.
- 2026-06-02 10:12 queue audit: `94539` remains running on `server21` with
  4 H200 GPUs and Slurm elapsed `02:01:20`. Member logs are still advancing
  and the largest printed per-member elapsed time is about `7238s`, still
  below `10800s`. The output directory has intermediate `best_model.pt` files
  only, not final `model.pt`/`metrics.json`, so no training conclusion is
  allowed.
- 2026-06-02 10:25 queue audit: `94539` remains running on `server21` with
  one node / 4 H200 GPUs and Slurm elapsed `02:14:52`. The member logs are
  advancing, stderr only contains the expected Transformer nested-tensor
  warnings, and the output directory still has only intermediate
  `best_model.pt` files. It is still below `MIN_TRAIN_SECONDS=10800`, so
  neither the validation traces nor the temporary checkpoints can be used to
  judge the world-model direction.
- 2026-06-02 10:49 queue audit: `94539` remains running on `server21` with
  one node / 4 H200 GPUs and Slurm elapsed `02:39:08`. The latest printed
  member elapsed time is about `9442s`, still below the required `10800s`.
  Only intermediate `member_*/best_model.pt` files exist; no final
  `model.pt`/`metrics.json` exists yet. Stderr still contains only the
  expected Transformer nested-tensor warnings. This is healthy liveness, not
  compliant training evidence.
- 2026-06-02 10:54 queue audit: `94539` remains running on `server21` with
  one node / 4 H200 GPUs and Slurm elapsed `02:44:02`. The newest printed
  member elapsed time is about `9765s`, still below `MIN_TRAIN_SECONDS=10800`;
  no final checkpoints or metrics exist. The run is close to the floor but is
  still not compliant evidence.
- 2026-06-02 11:12 completion audit: `94539` completed on `server21` after
  Slurm elapsed `03:01:17`. Strict inspection `94863` passed with four final
  `model.pt`/`metrics.json` members, min elapsed `10870.0s`,
  `compliant_3h_training=True`, `compliant_4x_h200_request=True`, and
  `compliant_training_evidence=True`.
- 2026-06-02 11:14 prediction evaluation: CPU job `94864` completed and wrote
  `prediction_eval.md/json`. Overall learned hole-prediction RMSE beats CV at
  all horizons: horizon `1` `0.00014` vs `0.00032`, horizon `5` `0.00069` vs
  `0.00198`, horizon `10` `0.00159` vs `0.00482`, horizon `20` `0.00367` vs
  `0.01242`, horizon `40` `0.00741` vs `0.02880`. This is prediction evidence,
  not manipulation success evidence.
- Larger-shard backfill training job `94679`, fast-fit duplicate `94746`, and
  old alternate node-selection retry `94809` were canceled with their downstream inspections
  and evals after primary job `94539` passed strict inspection. They are not
  separate evidence.
- 2026-06-02 10:07 resource rule check: downstream compliant training jobs
  `94542` (`C_pi`), `94683` (`C_pi` backfill), and `94860` (RGB-D slot
  extractor) all request one node / 4 GPUs with Slurm time limits above three
  hours. This old node-exclusion wording is superseded: active wrapper source
  must not carry a standing node exclusion list. Shorter training is not
  allowed to count as evidence for or against the direction.
- 2026-06-02 10:16 wrapper hygiene: superseded by the no-node-list policy.
  Rendering and non-rendering wrappers must use live Slurm state, targeted
  canaries, and job-local manifest evidence rather than any inherited
  conservative render exclusion policy.
- Current scheduling note from `2026-06-02 07:31+08:00`: `94539` remains the
  earliest object-WM training forecast at `2026-06-03T00:01:27`. Additional
  compliant normal-route and alternate node-selection route probes would start at
  `2026-06-09T21:01:12`; drain-node-only probes for
  `server16`/`server29`/`server61` are rejected by Slurm, so no extra duplicate
  was submitted.
- Current scheduling note from `2026-06-02 09:45+08:00`: the earlier forecast
  resolved and `94539` is running. `94863` remains `afterany:94539` with
  `REQUIRE_COMPLIANT=true`, and `94864` remains `afterok:94863`. Failovers
  `94679`, `94746`, and alternate node-selection route `94809` stay dependency-gated behind
  failed inspections rather than producing uncontrolled duplicate evidence.
- Current resource note from `2026-06-02 09:52+08:00`: full-node scan shows no
  schedulable node with four free H200 GPUs. `server16`/`server29` have
  nominal free GPUs but remain drained; `server31` has only three free GPUs;
  `server61` is recovered to `mix` but all eight GPUs are allocated. No extra
  compliant duplicate was submitted.
- Dependent inspection jobs: `94620` for `94442` completed, strict `94863` for
  `94539`, `94680` for `94679`, `94747` for `94746`, and `94810` for
  `94809`. They run on the CPU
  partition with `afterany` so failure or non-compliance is exposed instead of
  silently skipped.
- Dependent prediction-evaluation jobs: `94672` for `94442` completed after
  `afterok:94620`, strict `94864` for `94539` after `afterok:94863`, and `94681` for
  `94679` after `afterok:94680`, and `94748` for `94746` after
  `afterok:94747`, and `94811` for `94809` after `afterok:94810`. They compare
  learned ensemble predictions against
  static-hole and constant-velocity baselines by horizon, scenario, and trigger
  phase; the wrapper refuses to run unless inspection reports
  `compliant_training_evidence=true`.
- Full-shard learned-world-model controller evaluation waits on successful
  post-training inspection and now uses the v4 task-frame-projected bridge.
  Older full-shard controller branches without
  `bridge_servo_mode=task_frame_projected` were canceled before running.
  Current replacement chains are `95055 -> 95056 -> 95057 -> 95058 -> 95059`
  after `94863` for primary job `94539`,
  `95060 -> 95061 -> 95062 -> 95063 -> 95064` after `94680` for failover
  job `94679`, `95065 -> 95066 -> 95067 -> 95068 -> 95069` after `94747` for
  failover job `94746`, and `95070 -> 95071 -> 95072 -> 95073 -> 95074` after
  `94810` for old node-specific failover `94809`. Gated video jobs must be
  visually inspected if they run.
- 2026-06-02 10:42 downstream audit: the current full-shard controller chains
  are still dependency-gated in the intended order. Primary chain `95055` only
  starts after strict object-WM inspection `94863`, then `95056` inspects the
  no-video run, `95057` gates final dynamic success, and only then can
  `95058` produce video for direct visual inspection by `95059`. The failover
  chains remain attached to `94680`, `94747`, and `94810`. This keeps raw
  training completion or no-video metrics from becoming final dynamic
  manipulation evidence.
- Post-run inspection command:
  `scripts/world_model/inspect_world_model_ensemble.py --ensemble-dir experiments/world_model_task_rebinding/object_state_world_model/ensemble_4gpu/job94442 ...`
  The inspector marks the run non-compliant if any member trained for less than
  10800 seconds.

## Evaluation

- [x] Report future hole pose error for non-learned baselines.
- [x] Report peg-head-in-hole-frame prediction error for non-learned baselines.
- [x] Report learned-model ensemble uncertainty after first compliant training.
      Evaluation `94672` reports nonzero ensemble uncertainty and high
      uncertainty-error correlation, but this is not a calibrated controller
      handoff rule yet.
- [x] Compare learned model against constant-velocity baseline for first
      compliant training. CPU evaluation `94672` completed for `94442`;
      strict eval `94864` for `94539` completed for the larger-shard primary.
      Backup eval `94681` for `94679` was canceled after the primary passed
      strict inspection.
- [x] Save example predicted versus actual trajectories in the training script.
- [x] Add online inference path for the controller with strict manifest,
      feature-name, target-name, and normalizer checks.
- [x] Queue a dependent `rebind_world_model` controller smoke that uses final
      `model.pt` paths after compliant post-training inspection. Job `94564`
      was updated from raw `afterok:94442` to `afterok:94620`, so it waits for
      successful object-model inspection rather than merely training-process
      completion.
- [x] After `94442` completes, inspect the ensemble before treating `94564` as
      world-model controller evidence. Inspection `94620` passed; `94564` is
      now running, and controller-run inspection `94696` waits on it.
- [x] After `94539` completes and strict inspection `94863` passes, inspect
      v4 no-video run `95055`/`95056` and, only if gate `95057` passes,
      inspect the gated video run `95058`/`95059` before
      treating full-shard learned-world-model controller behavior as evidence.
      `95055` ran after `94863`; inspection `95056` completed and gate
      `95057` failed with exit `65`, so no gated video was produced. The
      failure is recorded as insertion-axis controller/bridge failure, not
      world-model training failure.
- [x] If failover job `94679`, `94746`, or `94809` becomes active because the
      earlier strict inspection failed, inspect its corresponding v4 gated
      controller chain before drawing controller conclusions. This branch is
      closed: primary strict inspection `94863` passed, so failover trainings
      and controller chains were canceled before running and produced no
      evidence.

Completion standard: the learned model beats constant velocity on task-frame
prediction in at least one nontrivial dynamic family and has usable uncertainty.
