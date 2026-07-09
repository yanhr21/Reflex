# Dataset Collection Targets

Date: 2026-07-06

This document records target dataset sizes and production order. The exact
counts may be adjusted after smoke review, but production must follow the
class roles in `docs/dataset_construction.md`.

## Current Status

As of 2026-07-07, Stage 1 static RGB smoke `static_rgb/smoke05` completed with
one video, three review frames, and a summary file. Full static RGB production
is still blocked until the user approves the smoke visual quality by writing
the required approval file. B/C/D/E smoke launchers must also remain blocked
until this Stage 1 review gate passes.

Smoke video:

- `experiments/maniskill/runs/01_dataset/static_rgb/smoke05/0.mp4`

Review gate helpers:

- `scripts/world_model/require_dataset_smoke_approved.sh`
- `scripts/world_model/require_dataset_stage_ready.sh`
- `scripts/world_model/dataset_next_stage_status.sh`
- `scripts/world_model/dataset_post_approval_plan.sh`

## Production Order

1. A static expert RGB smoke from official demos.
2. Human visual review of smoke videos / frames.
3. A static expert RGB full render from official 1000 demos.
4. B dynamic RGB observation smoke.
5. Human visual review.
6. B dynamic RGB observation production.
7. C frozen-DP dynamic failure collection.
8. D future-frame cooperation teacher collection.
9. E Cosmos-predicted cooperation collection.

Do not start full production before the relevant smoke is approved.

## Target Sizes

Initial training-scale target after smoke approval:

- A static expert: render all official 1000 demos if render quality is
  approved. Use this for static skill preservation and Cosmos static future.
- B dynamic RGB observation: at least 1000 episodes across motion families.
  Success is not required. This is the main data for Cosmos dynamic future
  prediction.
- C frozen-DP dynamic outcome: at least 500 rollouts, balanced across motion
  families, to record frozen-DP successes, failures, and discrepancy labels.
  Success is an allowed outcome label, not a reason to reject or archive C.
- D future-frame cooperation teacher: start with 100 smoke/overfit-quality
  rollouts, then scale toward 500-1000 if the GT future-frame interface works.
- E Cosmos-predicted cooperation: start only after Cosmos/readout predicts
  future target frames well enough on held-out B data. Initial target is
  100-300 rollouts for robustness/adaptation, then scale if successful.

Do not treat these numbers as success claims. They are data production targets.
After A full RGB and B/C/D production finish, the first joint-training
readiness gate is `joint_overfit_abcd`. This gate is for the required
overfit experiment before full training and intentionally does not require E,
because E depends on Cosmos/readout predicted future target frames.

The `joint_overfit_abcd` readiness gate requires these production validators
to pass:

- `scripts/world_model/validate_dataset_production_run.sh b_dynamic_production`
- `scripts/world_model/validate_dataset_production_run.sh c_frozen_dp_production`
- `scripts/world_model/validate_dataset_production_run.sh d_future_teacher_production`

It also requires B/C/D to be indexed into the active data registry:

- `scripts/world_model/build_dataset_production_index.sh b_dynamic_production`
- `scripts/world_model/build_dataset_production_index.sh c_frozen_dp_production`
- `scripts/world_model/build_dataset_production_index.sh d_future_teacher_production`

The overfit training guard is:

```bash
scripts/world_model/require_dataset_training_inputs_ready.sh joint_overfit_abcd
```

E remains a later dataset class. After Cosmos/readout can predict future
target frames well enough, E production should pass:

- `scripts/world_model/validate_dataset_production_run.sh e_cosmos_predicted_production`

and be indexed with:

- `scripts/world_model/build_dataset_production_index.sh e_cosmos_predicted_production`

The later all-class training guard is:

```bash
scripts/world_model/require_dataset_training_inputs_ready.sh full_joint
```

For B/C/D shard production under `prod01/<family>`, use the shard index
builder after the shard-aware validator passes:

- `scripts/world_model/build_dataset_production_shard_index.sh b_dynamic_production`
- `scripts/world_model/build_dataset_production_shard_index.sh c_frozen_dp_production`
- `scripts/world_model/build_dataset_production_shard_index.sh d_future_teacher_production`

The read-only index status helper is:

- `scripts/world_model/dataset_production_index_status.sh`

The indexes live under `experiments/maniskill/data/active/` as
`b_dynamic_production/`, `c_frozen_dp_production/`,
`d_future_teacher_production/`, and `e_cosmos_predicted_production/`.
The index builder refuses to write an index unless the source `summary.json`
and `manifest.txt` match the requested stage, dataset class, run group, run
name, output directory, method/teacher evidence flags, and no-state-
intervention contract.

Both guards require production validation and production training indexes, not
just output-directory existence. `joint_overfit_abcd` remains the immediate
gate for the Stage 2 overfit route; `full_joint` remains closed until E is
actually available under its reviewed role and indexed into
`train_samples.jsonl` / `val_samples.jsonl`.

Current limited bootstrap:

- B dynamic legacy bootstrap already has `900` train and `100` val samples
  registered under
  `experiments/maniskill/data/active/b_dynamic_legacy_bootstrap/`.
- Its training entry files are `train_samples.jsonl` and
  `val_samples.jsonl`.
- It may support early Cosmos/readout diagnostics only. It is not new B
  production, not positive DP BC, and not final method evidence.

## Motion Family Balance

B/C/D/E should cover:

- constant left/right target motion;
- constant forward/backward target motion;
- reverse motion;
- move-stop;
- sine or nonconstant velocity;
- continuously moving target that does not stop before insertion;
- peg / wooden-stick disturbance.

For B and C, include failures and hard cases. For D and E, require active
robot-driven insertion attempts through legal controller actions.

Current B/C/D production shard plan after smoke approval:

- B `dynamic_rgb/prod01`: `lr=170`, `fb=170`, `reverse=165`,
  `stop=165`, `sine=165`, `cont=165`, total `1000`.
- C `frozen_dp_dynamic/prod01`: `lr=84`, `fb=84`, `reverse=83`,
  `stop=83`, `sine=83`, `cont=83`, total `500`.
- D `future_teacher/prod01`: `lr=84`, `fb=84`, `reverse=83`,
  `stop=83`, `sine=83`, `cont=83`, total `500`.

The active shard families are target/hole motion families:
`constant_lr`, `constant_fb`, `reverse`, `move_stop`, `sine`, and
`continuous`. `peg_disturb` remains a desired future family but is not counted
in the active production balance until a real peg-disturbance runner exists.

The shard-aware validator is still invoked through
`scripts/world_model/validate_dataset_production_run.sh`. If a single
`prod01/summary.json` is absent, it aggregates `prod01/<family>/summary.json`
and requires the six active families above.

## Loss Role By Class

- A: DP BC / distillation, Cosmos static future, phase extraction.
- B: Cosmos future RGB / latent, target-frame readout, uncertainty,
  trajectory consistency.
- C: outcome labels, negative / discrepancy labels, infeasible/no-progress,
  contrastive. Physically valid frozen-DP success remains a label, not a
  production failure.
- D: moving-frame adapter / residual, phase/timing, relative velocity at
  contact.
- E: adapter robustness to Cosmos prediction, uncertainty-conditioned live
  control.

No failed action chunk should become positive DP expert action just because it
appears in a trajectory.

## Review Gates

Every smoke run must produce:

- `manifest.txt`
- `summary.json`
- rendered videos
- review frames
- `human_review_required=true`

The user must approve the visual quality before production starts for that
class. If approval is pending, the dataset goal is waiting on human review, not
complete.
