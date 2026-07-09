# Dataset Plan

Objective: build the dataset needed for Cosmos-guided moving-frame Diffusion
Policy.

The dataset must support this method:

```text
RGB Cosmos-3 predicts future target / hole frames.
A readout extracts future pose, velocity, tau, and uncertainty.
A moving-frame adapter conditions a protected state-based DP skill.
The robot executes DP-compatible actions in the predicted future target frame.
```

## Stage 0: Inventory And Rendering Gap

Known facts:

- Official DP checkpoint is state-based and uses `pd_ee_delta_pose`.
- Official 1000 demos exist as H5/JSON state/action data.
- Official 1000 demos do not currently have active RGB/mp4/jsonl exports.
- Old `fix3_733` is not enough for final method evidence.

Required work:

- define a new active dataset root after the exact collection run starts;
- render official static demos into RGB/video-action format for Cosmos
  alignment;
- preserve H5 state/action and RGB/video timestamps in one manifest;
- do not reuse legacy long `world_model_task_rebinding` paths as active
  dataset roots.

## Stage 1: Static Expert Set

Build from official 1000 demos.

Current smoke entry point:

- launcher: `scripts/slurm/launch_dataset_static_rgb_smoke_tmux.sh`
- allocation runner:
  `scripts/slurm/run_dataset_static_rgb_smoke_in_allocation.sh`
- default output:
  `experiments/maniskill/runs/01_dataset/static_rgb/smoke05/`
- default log: `logs/01_dataset/static_rgb/smoke05.log`
- default sample count: `1` for smoke review; scale only after approval.
- production gate: `human_review_required=true`; do not scale until the user
  approves the rendered videos / frames.
- current status: `smoke05` completed on 2026-07-07 with one RGB video,
  `summary.json`, `manifest.txt`, and three review frames. Full static RGB
  production is still blocked until explicit human approval is recorded.
- resource policy for post-approval rendering: request `1 GPU` by default,
  lower CPU/memory first if the scheduler cannot allocate, wait on a valid
  queued tmux-held allocation, and use previously bad nodes only as
  smoke-only diagnostics with node evidence recorded.

Outputs:

- state/action H5 reference;
- RGB render for each accepted episode;
- target-centric relative trajectory fields;
- static insertion phase and final success labels.

Training use:

- DP action BC / distillation;
- Cosmos static future prediction;
- insertion phase extraction;
- regression test that new method does not break static skill.

## Stage 2: Dynamic Observation Set

Generate moving target / moving hole / peg disturbance episodes. Success is
not required.

This stage begins only after the Stage 1 smoke render path is visually
approved, because dynamic data also depends on the same RGB render stack.

Current launch status:

- guarded launcher exists:
  `scripts/slurm/launch_dataset_dynamic_rgb_smoke_tmux.sh`;
- shared Slurm/tmux smoke launcher exists:
  `scripts/slurm/launch_dataset_stage_smoke_tmux_common.sh`;
- in-allocation runner is still missing:
  `scripts/slurm/run_dataset_dynamic_rgb_smoke_in_allocation.sh`;
- current readiness remains blocked by Stage 1 human review.

Motion coverage:

- constant left/right and forward/backward motion;
- reverse motion;
- move-stop;
- sine or nonconstant velocity;
- continuous moving target that does not stop before insertion;
- peg / wooden-stick disturbance.

Outputs:

- RGB history and future RGB;
- H5 state/action trace if robot acts;
- target/hole pose trajectory;
- velocity, acceleration, tau candidates, and uncertainty labels;
- physical-validity audit.

Training use:

- Cosmos future RGB / latent prediction;
- future target-frame readout;
- trajectory consistency;
- uncertainty estimation.

Not training use:

- do not use failed actions as positive DP expert actions.

## Stage 3: Frozen-DP Dynamic Outcome Set

Run frozen official DP in dynamic scenes to record its outcome distribution and
failure modes. Success and failure are both labels; the collector must not
force the frozen policy to fail or reject a physically valid success.

Current launch status:

- guarded launcher exists:
  `scripts/slurm/launch_dataset_frozen_dp_dynamic_smoke_tmux.sh`;
- shared Slurm/tmux smoke launcher exists:
  `scripts/slurm/launch_dataset_stage_smoke_tmux_common.sh`;
- in-allocation runner is still missing:
  `scripts/slurm/run_dataset_frozen_dp_dynamic_smoke_in_allocation.sh`;
- current readiness remains blocked by Stage 1 human review.

Outputs:

- action trace;
- RGB video;
- before/after distances;
- labels for miss, jam, late contact, bad relative velocity,
  target-assisted insertion, and success if any.

Training use:

- negative / failure labels;
- real-imagination discrepancy;
- infeasible / unsafe / no-progress classifier;
- contrastive loss against insertion-corridor futures.

Research use:

- measures when the base static DP does or does not solve the target dynamic
  setting by itself.

## Stage 4: Future-Frame Cooperation Teacher Set

Create legal controller rollouts that use known future target trajectory to
teach the moving-frame adapter.

Current launch status:

- guarded launcher exists:
  `scripts/slurm/launch_dataset_future_frame_teacher_smoke_tmux.sh`;
- shared Slurm/tmux smoke launcher exists:
  `scripts/slurm/launch_dataset_stage_smoke_tmux_common.sh`;
- in-allocation runner is still missing:
  `scripts/slurm/run_dataset_future_frame_teacher_smoke_in_allocation.sh`;
- current readiness remains blocked by Stage 1 human review and by the missing
  GT future-frame interface runner.

This is the missing cooperation data. It should answer:

```text
If the method knows where the hole will be at contact time,
can a protected static DP skill plus adapter actively insert?
```

Inputs:

- real current robot/peg state;
- ground-truth future target / hole trajectory during data generation;
- static insertion relative trajectory or DP action prior.

Outputs:

- legal `pd_ee_delta_pose` action trace;
- target-frame and world-frame trajectories;
- phase / tau / relative velocity at contact;
- success/failure labels.

Rules:

- ground-truth future is allowed only for teacher generation and must be
  recorded as teacher-only;
- no simulator-state edits;
- no target-assisted success;
- no hidden manual finisher.

Training use:

- adapter / residual action supervision;
- moving-frame conditioning;
- phase and timing supervision.

## Stage 5: Cosmos-Predicted Cooperation Set

Replace ground-truth future target frame with Cosmos-predicted future frame.

Current launch status:

- guarded launcher exists:
  `scripts/slurm/launch_dataset_cosmos_predicted_coop_smoke_tmux.sh`;
- shared Slurm/tmux smoke launcher exists:
  `scripts/slurm/launch_dataset_stage_smoke_tmux_common.sh`;
- in-allocation runner is still missing:
  `scripts/slurm/run_dataset_cosmos_predicted_coop_smoke_in_allocation.sh`;
- current readiness remains blocked by Stage 1 human review and by missing
  B/D/Cosmos-readout validation.

Purpose:

- close the train-test gap;
- train robustness to Cosmos error;
- evaluate the actual deployed method.

Training/evaluation use:

- adapter robustness;
- uncertainty-conditioned action;
- live rollout under receding-horizon updates.

## Dataset Manifest Schema

Each sample manifest must include:

- sample id, source class A/B/C/D/E, source H5/RGB path;
- whether the future target frame is ground truth or Cosmos-predicted;
- controller type and action contract;
- DP checkpoint path if DP is used;
- Cosmos checkpoint path if Cosmos is used;
- RGB frame timing and state/action timing;
- `T_hole(t)`, future `T_hole(t+tau)`, `v_hole(t+tau)`, uncertainty;
- peg/ee poses and target-frame relative poses;
- action chunk and executed action rows;
- success, miss, jam, target-assisted, snap, state-intervention labels;
- allowed training losses for this sample;
- whether the sample is method evidence, teacher evidence, negative evidence,
  or diagnostic-only evidence.

## Minimal First Experiment

Before training Cosmos into the loop, run an oracle-future-data check inside a
valid Slurm allocation:

```text
real current state
+ ground-truth future target frame
+ protected DP / adapter
-> moving-target insertion attempt
```

If this fails, the future-frame interface is insufficient and the adapter /
representation must be changed before training Cosmos. If this succeeds, then
the next task is to train Cosmos/readout to predict the same future target
frame from RGB history.

## Smoke-First RGB Gate

Before any large data production:

1. render a small RGB smoke slice;
2. write `manifest.txt`, `summary.json`, videos, and review frames;
3. mark `human_review_required=true`;
4. stop and wait for user visual approval;
5. only then launch larger shards.

If a render fails or hangs, inspect the run log and node evidence from that
specific job. Follow `docs/legacy/RENDERING_CLUSTER_VULKAN.md`: use the system
NVIDIA Vulkan ICD, empty `DISPLAY`, and `HDF5_USE_FILE_LOCKING=FALSE`.

Output layout:

- `experiments/maniskill/runs/01_dataset/<set>/<try>/`
- logs under `logs/01_dataset/<set>/<try>.log`
- accepted dataset artifacts under a short active dataset root selected at
  collection time.
