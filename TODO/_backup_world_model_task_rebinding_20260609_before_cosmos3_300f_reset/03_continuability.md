# Continuability TODO

## Label Generation

- [x] Add candidate-state sampler.
- [x] Add reset-or-bridge rollout evaluator for frozen DP.
- [x] Save candidate features, rollout result, and failure reason.
- [ ] Balance positives and hard negatives.
- [x] Add Slurm wrapper for label shards.
- [x] Add rollout-dir 4-GPU label wrapper for larger dynamic shards. It
      discovers H5s on the compute node, validates them, generates 4 label
      shards, writes `label_h5s.txt`, and runs per-shard inspection.
- [x] Complete first Slurm smoke label shard and inspect outputs.
- [x] Complete larger pilot label shard for `C_pi` training data.

Label utilities:

- `scripts/world_model/label_continuability_rollouts.py`
- `scripts/world_model/inspect_continuability_labels.py`
- `scripts/slurm/label_continuability_rollouts.sbatch`
- `scripts/slurm/label_continuability_rollouts_4gpu.sbatch`
- `scripts/slurm/label_continuability_from_rollout_dir_4gpu.sbatch`
- Smoke job: `94432` completed on 2026-06-02.
- Smoke inspection:
  `experiments/world_model_task_rebinding/continuability_labels/smoke/job94432/inspection.md`
- Larger pilot job: `94445` completed with `RUN_GROUP=pilot128`,
  `MAX_CANDIDATES=128`, `MAX_ROLLOUT_STEPS=120`.
- Pilot inspection:
  `experiments/world_model_task_rebinding/continuability_labels/pilot128/job94445/inspection.md`
- Canceled malformed output-path attempt: `94444`, recorded under
  `experiments/world_model_task_rebinding/continuability_labels/pilot128/jobmanual/slurm_manifest.txt`.
- Added candidate sharding flags and a dense 4-GPU label wrapper for larger
  follow-up shards if `94445` is insufficient or imbalanced.
- Dependent larger-shard label job: `94540`, completed after dynamic shard
  `94510`, with `MAX_CANDIDATES_PER_SHARD=128`, `CANDIDATES_PER_TRAJ=96`,
  `CANDIDATE_STRIDE=2`, and `MAX_ROLLOUT_STEPS=160`. It wrote four inspected
  shards and `label_h5s.txt` under
  `experiments/world_model_task_rebinding/continuability_labels/full4gpu_from_job94510/job94540`.
- Larger-shard label backfill job `94682` was canceled after primary label job
  `94540` completed and primary `C_pi` training job `94542` started, because
  it would have generated duplicate labels from the same rollout directory.

Current label definition:

- Reset simulator to a saved candidate `env_state`.
- Synchronize the frozen DP observation history from saved `obs_stack`.
- Roll out frozen DP for the remaining budget.
- By default, replay the source trajectory's recorded external perturbation
  deltas during the continuation rollout. This avoids silently freezing a
  still-moving target and producing labels that are too easy.
- Save empirical success/failure labels, not geometric-threshold labels.
- Inspect every shard for success rates, post-trigger coverage, failure reasons,
  and rollout trace lengths before using it for `C_pi` training.

Smoke result:

- 8 candidates, 50% post-trigger, 1/8 success.
- Failure reasons: 4 not-grasped/dropped, 3 timeout, 1 success.
- This validates the label path but is intentionally too small and imbalanced
  for `C_pi` training.

Pilot128 result:

- 128 candidates, 60.9% post-trigger, 28/128 success.
- Scenario counts: 23 hole_constant, 23 hole_move_stop, 20 hole_reverse,
  13 none, 23 peg_disturb, 26 peg_drop.
- Failure reasons: 94 timeout, 6 not-grasped/dropped, 28 success.
- This is enough to start the first `C_pi` training job, but still pilot-scale
  and not final label coverage.

## Model

- [x] Add first `C_pi` classifier/regressor training script and compliant
      4-GPU H200 ensemble wrapper.
- [x] Train first `C_pi` classifier/regressor. Job `94471` used inspected
      pilot128 labels, reserved 4 H200 GPUs, ran for `03:00:20`, and completed
      on `server04`. Post-training inspection `94618` passed with four
      complete members, min member elapsed `10817s`, and
      `compliant_training_evidence=true`.
- [ ] Train larger-label `C_pi` classifier/regressor. Job `94542` is running
      on `server28` with one node / 4 H200 GPUs, `MIN_LABELS=512`, and
      `MIN_TRAIN_SECONDS=10800`. Post-training inspection job `94658` is
      queued with `afterany:94542`; calibration `94660` waits on successful
      inspection. Canceled duplicate backfill chain
      `94682 -> 94683 -> 94684 -> 94685`.
      Queue audit at `2026-06-02 10:49+08:00`: `94542` remains running on
      `server28` with one node / 4 H200 GPUs and Slurm elapsed `00:29:48`.
      The output directory has only intermediate `best_model.pt` files.
      Early validation AUPRC/AUROC/BCE traces are liveness diagnostics only;
      no direction-level conclusion is allowed until the run reaches
      `MIN_TRAIN_SECONDS=10800` and inspection `94658` reports
      `compliant_training_evidence=true`.
      Queue audit at `2026-06-02 10:54+08:00`: `94542` remains live on
      `server28` with one node / 4 H200 GPUs and Slurm elapsed `00:34:42`.
      It is still far below `MIN_TRAIN_SECONDS=10800`; stdout is advancing
      but no final `model.pt`/`metrics.json` files exist.
      Queue audit at `2026-06-02 11:23+08:00`: `94542` remains live on
      `server28` with one node / 4 H200 GPUs and Slurm elapsed `01:03:26`.
      It is still below the 3-hour evidence floor; intermediate
      `best_model.pt` files and validation traces are liveness only until
      inspection `94658` reports compliant training evidence.
      Queue audit at `2026-06-02 11:55+08:00`: `94542` remains live on
      `server28` with one node / 4 H200 GPUs and Slurm elapsed `01:35:33`.
      The output directory still has only `best_model.pt` intermediates and
      member manifests, no final `model.pt`/`metrics.json`. Stdout continues
      to advance, but this is still below `MIN_TRAIN_SECONDS=10800`, so no
      larger-label `C_pi` conclusion is allowed.
- [ ] Include task-frame features, grasp state, phase, remaining budget, and
      world-model uncertainty.
- [x] Add post-hoc threshold calibration tool for conservative handoff.
- [x] Add online `C_pi` feature-contract checks and ensemble-disagreement
      logging in the controller. Handoff with `HANDOFF_MODE=cpi` now requires
      both `CPI_THRESHOLD` and `CPI_MAX_STD`.
- [x] Calibrate threshold for conservative handoff after compliant `94471`
      finishes and passes inspection. First calibration job `94659` failed
      after writing JSON because Markdown report generation reused the
      `threshold` format key. The report writer was fixed and CPU job `94716`
      completed. Held-out thresholds for target FPR `0.0`, `0.02`, `0.05`,
      and `0.1` all select `1.000001` with zero true positives, so this pilot
      `C_pi` must not be used as a useful DP handoff gate.

Training utilities:

- `scripts/world_model/train_continuability_model.py`
- `scripts/world_model/inspect_continuability_model_ensemble.py`
- `scripts/world_model/calibrate_continuability_threshold.py`
- `scripts/slurm/train_continuability_model_ensemble_4gpu.sbatch`
- `scripts/slurm/inspect_continuability_model_ensemble.sbatch`
- `scripts/slurm/calibrate_continuability_threshold.sbatch`
- Wrapper refuses undersized training (`<4` tasks/GPUs or
  `MIN_TRAIN_SECONDS<10800`) and refuses to train on uninspected or tiny label
  shards by default (`MIN_LABELS=128`).
- Dependent inspection job: `94618` for `94471`. It completed on the CPU
  partition and passed compliance; failure or non-compliance would have been
  exposed instead of silently skipped.
- Dependent larger-label inspection job: `94658` for `94542`, also on the CPU
  partition with `afterany`.
- Canceled backfill larger-label chain: label job `94682`, training job
  `94683`, inspection `94684`, and calibration `94685`.
- Post-run inspection command:
  `scripts/world_model/inspect_continuability_model_ensemble.py --ensemble-dir experiments/world_model_task_rebinding/continuability_model/ensemble_4gpu/job94471 ...`
  The inspector only marks the run compliant if all four members wrote
  `model.pt`/`metrics.json` and the shortest member elapsed time is at least
  10800 seconds.
- Post-inspection threshold calibration command:
  `scripts/world_model/calibrate_continuability_threshold.py --ensemble-dir experiments/world_model_task_rebinding/continuability_model/ensemble_4gpu/job94471 ...`
  The calibrator uses held-out validation predictions and chooses thresholds
  that satisfy target false-positive-rate constraints. It refuses incomplete
  members by default.
- Calibration jobs: `94659` waited on successful pilot inspection `94618` but
  failed only in Markdown report writing. Fixed rerun `94716` completed and
  wrote both `threshold_calibration.json` and `threshold_calibration.md`.
  `94660` waits on successful larger-label inspection `94658`. Backfill
  calibration `94685` waits on successful inspection `94684`. The Slurm
  wrapper refuses to calibrate unless `inspection.json` reports
  `compliant_training_evidence=true`.
- 2026-06-02 09:41 preflight: larger-label chains are still pending with no
  new label/model artifacts. `94540` has dependency released and is
  priority-pending as a one-node / 4-GPU label job over the six H5 files in
  `dynamic_state_rollouts/full4gpu/job94510`. Training `94542` waits on
  `afterok:94540` and exports `MIN_LABELS=512` plus
  `MIN_TRAIN_SECONDS=10800`. Backfill `94682 -> 94683` has the same 4-GPU
  label/training structure, with `94683` also enforcing `MIN_LABELS=512` and
  `MIN_TRAIN_SECONDS=10800`. Inspection jobs `94658`/`94684` require
  compliant training evidence before calibration jobs `94660`/`94685` can
  run. No training conclusion is allowed until these inspections pass.
- 2026-06-02 09:45 queue audit: `94540` is priority-pending with scheduler
  forecast `2026-06-02T10:51:00` on `server31`; backfill label job `94682` is
  priority-pending with forecast `2026-06-02T11:40:47` on `server21`.
  Training jobs `94542` and `94683` still request one node / 4 H200 GPUs with
  `MIN_TRAIN_SECONDS=10800` and remain blocked on `afterok` label dependencies,
  so no C_pi training can be judged before label generation and compliant
  post-training inspection.
- 2026-06-02 09:52 queue audit: `scontrol` shows `94540` has
  `Dependency=(null)`, a then-current job-local exclusion snapshot, and a
  forecast on `server31`; `94682` likewise has no residual dependency and a
  forecast on `server21`. There are no new label artifacts yet, so
  `94542`/`94683` remain correctly blocked on label completion.
- 2026-06-02 10:12 queue/artifact audit: `94540` is running on `server28`
  with one node / 4 GPUs and elapsed about `00:10`. Its true output directory
  is
  `experiments/world_model_task_rebinding/continuability_labels/full4gpu_from_job94510/job94540`.
  It currently contains input validation, `slurm_manifest.txt`, and four
  shard manifests, but no `continuability_labels.h5` yet. Therefore no
  `C_pi` training result exists, and downstream `94542` remains correctly
  blocked on label completion.
- 2026-06-02 10:16 wrapper hygiene: superseded by the no-standing-node-list
  policy. `C_pi` label/training/inspection and calibration wrappers should not
  inherit rendering-failure observations; any node exclusion must be job-local
  and backed by live Slurm evidence.
- 2026-06-02 10:21 queue/artifact audit: `94540` completed in `00:17:21` on
  `server28` and wrote all four `continuability_labels.h5` shards plus
  `label_h5s.txt`; total label count is 512. Primary training `94542` started
  at `2026-06-02T10:20:07+08:00` on `server28` with one node / 4 H200 GPUs and
  a `03:30:00` time limit. It has only intermediate `best_model.pt` files so
  far. No result can be judged before `94542` exceeds 10800 seconds and CPU
  inspection `94658` passes. Duplicate backfill jobs
  `94682`/`94683`/`94684`/`94685` were canceled to avoid redundant GPU work.
- 2026-06-02 10:25 queue/artifact audit: `94542` remains running on
  `server28` with one node / 4 H200 GPUs and elapsed about `00:05`. Its
  stdout is advancing, the output directory has only intermediate
  `best_model.pt` files, and no final `model.pt`/`metrics.json` exists. This
  is liveness evidence only; no larger-label `C_pi` claim is allowed before
  the 3-hour floor and inspection `94658`.
- 2026-06-02 11:55 queue/artifact audit: `94542` remains running on
  `server28` with one node / 4 H200 GPUs and elapsed `01:35:33`, still below
  `MIN_TRAIN_SECONDS=10800`. Latest stdout rows are liveness diagnostics only;
  no final per-member `model.pt`/`metrics.json` exists yet.
- 2026-06-02 12:00 liveness audit: `94542` remains running on `server28` with
  Slurm elapsed `01:39:59`, still below the 3-hour evidence floor. An
  in-allocation process check shows four
  `train_continuability_model.py` member processes alive with
  `--min-train-seconds 10800` and seeds `300`-`303`. The manifest records
  `requested_gres=gpu:4`, `ntasks=4`, `label_count=512`, and
  `min_train_seconds=10800`. GPU utilization is low for the tiny classifier,
  but stdout is advancing and processes are alive; this is liveness evidence
  only, not a result.
- 2026-06-02 12:26 calibration hygiene: local regression exposed that the
  shared `.venv` had a mixed/broken NumPy install (`numpy` import failed before
  CPU calibration could run). Reinstalled `numpy==1.26.4` in the same venv,
  then verified `import numpy`, `h5py`, `torch`, and `pip check`. The
  calibration script now reports paired `threshold_std_gates` in addition to
  probability-only thresholds, because online `HANDOFF_MODE=cpi` gates both
  `CPI_THRESHOLD` and `CPI_MAX_STD`. Pilot `94471` calibration was rerun to a
  temporary output and wrote JSON/Markdown successfully; the pilot held-out
  gate still has zero true positives, so this is only a code-path regression
  check, not new evidence for using the pilot `C_pi`.
- 2026-06-02 12:35 liveness audit: `94542` is still running on `server28`
  with one node / 4 H200 GPUs and Slurm elapsed `02:15:32`, below the
  `MIN_TRAIN_SECONDS=10800` evidence floor. The output directory contains
  only per-member `best_model.pt` and manifests, not final `model.pt` or
  `metrics.json`. No larger-label `C_pi` training result or controller
  handoff conclusion is allowed until `94658` reports compliant training
  evidence and `94660` completes calibration.
- 2026-06-02 12:43 compliance preflight: `inspect_continuability_model_ensemble.py`
  and its Slurm wrapper were rechecked. The inspection gate requires at least
  four complete member directories with `model.pt` and `metrics.json`, a
  shortest member elapsed time of at least `10800` seconds, and a 4-GPU H200
  training request before it writes `compliant_training_evidence=true`; the
  wrapper exits nonzero if that flag is false. Calibration `94660` has
  `afterok:94658`, so it cannot run after a non-compliant inspection. Current
  running job `94542` is on `server28`; `scontrol show node server28` reports
  `Gres=gpu:NVIDIAH200:8`, and the job stdout records four `NVIDIA H200`
  devices from `nvidia-smi`. This is compliance/liveness evidence only, not a
  C_pi result.
- 2026-06-02 12:55 liveness audit: larger-label `C_pi` training job `94542`
  is still running on `server28` with one node / 4 H200 GPUs and Slurm elapsed
  `02:35:00`, still below the 3-hour evidence floor. Inspection `94658` and
  calibration `94660` remain dependency-pending, so calibrated controller
  chain `95392 -> 95393 -> 95394 -> 95433 -> 95437` has not been released.
  Do not treat current intermediate traces as a training result.
- 2026-06-02 13:07 liveness audit: `94542` remains running on `server28` with
  one node / 4 H200 GPUs and Slurm elapsed `02:47:08`, still below
  `MIN_TRAIN_SECONDS=10800`. The output directory still contains only
  per-member `best_model.pt` and manifests, not final `model.pt` or
  `metrics.json`. Inspection `94658`, calibration `94660`, and calibrated
  controller/video chain `95392 -> 95393 -> 95394 -> 95433 -> 95437` remain
  blocked on compliant training completion.

## Evaluation

- [x] Report AUROC/AUPRC for the first compliant pilot. Inspection `94618`
      reports validation aggregate AUROC `0.7766 +/- 0.0662` and AUPRC
      `0.4652 +/- 0.2356`; held-out calibration `94716` reports AUROC
      `0.6350` and AUPRC `0.2334`.
- [x] Report calibration curve for the first compliant pilot. Calibration
      exists in
      `experiments/world_model_task_rebinding/continuability_model/ensemble_4gpu/job94471/threshold_calibration.json`;
      it shows severe held-out overconfidence, including 15 samples in the
      `0.9-1.0` probability bin with zero success.
- [x] Report false-positive rate at chosen threshold for the first compliant
      pilot. The only held-out thresholds satisfying target FPR up to `0.1`
      have FPR `0.0` and TPR `0.0`, so they are conservative but useless for
      handoff.
- [ ] Report gated DP handoff success rate.
- [ ] Compare against hand threshold rules.

Completion standard: `C_pi` reduces false handoffs compared with simple
geometric thresholds while preserving enough handoff opportunities to finish
tasks.
