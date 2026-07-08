# Dynamic Dataset Collection Plan

Date: 2026-07-06

This plan covers dataset classes B, C, D, and E. Do not launch dynamic
production until the Stage 1 static RGB smoke has passed human review.

## Preconditions

- Stage 1 static RGB smoke exists.
- User has approved smoke RGB visual quality.
- `scripts/world_model/require_dataset_smoke_approved.sh` passes.
- `scripts/world_model/require_dataset_stage_ready.sh` passes for the
  requested stage.
- Active data registry exists under `experiments/maniskill/data/active/`.

Current gate status on 2026-07-07: `static_rgb/smoke05` exists, but B/C/D/E
remain blocked because the smoke has not yet been human-approved. Do not
launch dynamic data production while this approval is missing.

Additional dynamic-scene gate on 2026-07-07: B/C/D/E new production also
requires an audited active adapter at
`scripts/world_model/active_dynamic_peg_adapter.py`. The source-audited adapter
now exists, but it is not a runner and is not yet runtime-validated. It must be
used only inside a compute-node Slurm smoke to prove that runtime SAPIEN
accepts continuous kinematic-target commands, produces RGB, and writes a
motion trace. This gate is required because the original `PegInsertionSide-v1`
target box is kinematic and the old recoverable generators used direct
pose/state paths that are not valid active data-construction mechanisms.

The current implementation gap list is maintained in:

- `docs/dataset_runner_implementation_gaps.md`

## B: Dynamic RGB Observation

Purpose: train Cosmos-3 future RGB / latent prediction and target-frame
readout.

Episodes do not need successful insertion. They must show continuous,
auditable motion.

Initial target after smoke approval: at least 1000 episodes.

Motion families:

- `constant_lr`: target moves left/right at constant velocity.
- `constant_fb`: target moves forward/backward at constant velocity.
- `reverse`: target reverses direction after a short window.
- `move_stop`: target moves then stops.
- `sine`: target follows nonconstant velocity.
- `continuous`: target does not stop before the insertion window.
- `peg_disturb`: peg / wooden stick is disturbed.

Required outputs:

- RGB video and review frames;
- state/action trace if robot acts;
- target/hole pose trajectory;
- velocity and acceleration;
- future pose labels at candidate `tau`;
- uncertainty or ambiguity labels if multiple futures are plausible;
- physical-validity audit.

Allowed losses:

- Cosmos dynamic future;
- target-frame readout;
- trajectory consistency;
- uncertainty.

Disallowed:

- treating failed robot actions as positive DP expert actions.

## C: Frozen-DP Dynamic Failure

Purpose: prove and label the gap between static DP and dynamic scenes.

Initial target after smoke approval: at least 500 rollouts balanced across B
motion families.

Current source status on 2026-07-07: the C smoke collector and in-allocation
runner exist:

- `scripts/world_model/collect_frozen_dp_dynamic_failure_smoke.py`
- `scripts/slurm/run_dataset_frozen_dp_dynamic_smoke_in_allocation.sh`

They are source-ready only. They have not been runtime-smoked, have not
produced data, and must not be counted as training artifacts until a
compute-node Slurm run produces RGB, action traces, target-motion traces, and
human-reviewed smoke evidence.

Required outputs:

- frozen DP checkpoint path;
- action trace;
- RGB video and review frames;
- target/hole motion trace;
- before/after distances;
- final simulator state;
- labels: miss, jam, no-progress, late contact, bad relative velocity,
  target-assisted, active robot-driven success if any.

Allowed losses:

- negative classification;
- discrepancy / OOD;
- infeasible/no-progress;
- contrastive against insertion-corridor futures.

Disallowed:

- positive BC from failed chunks;
- target-assisted success as success data.

## D: Future-Frame Cooperation Teacher

Purpose: train the moving-frame adapter with known future target trajectory.

Initial target: 100 smoke/overfit-quality rollouts, then 500-1000 if the
future-frame interface works.

Teacher input may include ground-truth future target trajectory, but only for
teacher generation. This must be recorded as teacher-only.

Current source status on 2026-07-07: the D smoke collector and in-allocation
runner exist:

- `scripts/world_model/collect_future_frame_teacher_smoke.py`
- `scripts/slurm/run_dataset_future_frame_teacher_smoke_in_allocation.sh`

They are source-ready only. They have not been runtime-smoked, have not
produced data, and must not be counted as teacher data until a compute-node
Slurm run produces RGB, legal action traces, target-motion traces, and
human-reviewed smoke evidence.

Required outputs:

- teacher future target source;
- legal controller action trace;
- target-frame trajectory and world-frame trajectory;
- phase / tau;
- relative velocity at contact;
- insertion corridor progress;
- RGB video and review frames;
- no state-edit audit.

Allowed losses:

- adapter residual;
- moving-frame conditioning;
- phase/timing;
- relative velocity at contact.

Disallowed:

- presenting teacher-only ground-truth future use as deployed method success.

## E: Cosmos-Predicted Cooperation

Purpose: close the train-test gap by replacing ground-truth future target
frames with Cosmos/readout predictions.

Initial target: 100-300 rollouts after B/D validation, then scale if stable.

E is explicitly gated by:

- `scripts/world_model/validate_dataset_production_run.sh b_dynamic_production`
- `scripts/world_model/validate_dataset_production_run.sh d_future_teacher_production`
- `scripts/world_model/require_dataset_cosmos_predicted_prereqs_ready.sh`

The Cosmos/readout validation summary is expected by default at
`experiments/maniskill/runs/02_joint_training/cosmos_readout/val01/summary.json`.
It must be held out on B-style dynamic data and must mark
`future_target_frame_readout_ready=true` before E smoke can start.

Current source status on 2026-07-07: the E smoke collector and in-allocation
runner exist:

- `scripts/world_model/collect_cosmos_predicted_coop_smoke.py`
- `scripts/slurm/run_dataset_cosmos_predicted_coop_smoke_in_allocation.sh`

They are source-ready only. They have not been runtime-smoked and must not be
run until B production, D production, and held-out Cosmos/readout validation
all pass. The E runner intentionally consumes Cosmos/readout prediction
JSONL; it must not silently fall back to ground-truth future target labels.

Required outputs:

- Cosmos checkpoint path;
- future prediction artifact;
- readout target frame, velocity, tau, uncertainty;
- adapter action trace;
- RGB video and review frames;
- final physical outcome;
- discrepancy between predicted and observed future.

Allowed losses:

- adapter robustness;
- uncertainty-conditioned control;
- live method evaluation.

Disallowed:

- hidden future labels as controller input for deployed-method claims.

## Production Rule

Each class must pass a small smoke and human visual review before production.
Do not scale B/C/D/E just because A passed. Each class can have distinct
failure modes.
