# Dataset Review Policy

Date: 2026-07-06

Every generated sample must be reviewed before it can enter training.

## Positive Training Data

Positive DP or adapter data must show:

- legal controller actions;
- active robot-driven insertion or a valid teacher action trace;
- aligned RGB and state/action timestamps;
- no state intervention;
- no target-assisted self-insertion;
- no snap, teleport, wall penetration, or disappearing objects.

## Diagnostic / Negative Data

The following are useful but must not become positive DP BC:

- frozen-DP dynamic misses;
- jams;
- no-progress rollouts;
- target-assisted insertion;
- plausible Cosmos future that fails execution;
- bad relative velocity at contact;
- moving target that outruns the robot.

These can support negative classification, discrepancy, contrastive, and
infeasible/no-progress losses.

## Teacher Data

Future-frame teacher data may use ground-truth future target trajectory only
for data generation. It must be marked:

```text
teacher_only=true
method_evidence_allowed=false
```

It can train adapter/residual/moving-frame conditioning, but cannot be reported
as deployed method success.

## Human Review

RGB smoke review is mandatory before production. New RGB dataset videos should
default to `30 FPS`. Full production cannot begin until the corresponding
smoke has:

- non-empty videos;
- review frames;
- `summary.json`;
- user approval recorded in `human_review_approved.txt`.

Approval content:

```text
approved=true
```

Current Stage 1 review target:

- `experiments/maniskill/runs/01_dataset/static_rgb/smoke05/0.mp4`

Current review request:

- `experiments/maniskill/runs/01_dataset/static_rgb/smoke05/review_request.md`

Status helper:

- `scripts/world_model/dataset_review_status.sh`
- `scripts/world_model/record_dataset_smoke_review_decision.sh`
- `scripts/world_model/dataset_class_review_status.sh`
- `scripts/world_model/require_dataset_class_smoke_approved.sh`

The status helper must report `goal_blocked_on_human_review=true` until the
approval file exists and contains `approved=true`. Agents must not create this
approval file without explicit user approval.

The decision helper may record either `approved` or `rejected`, but approving
must only happen after explicit user instruction. A rejection writes
`human_review_rejected.txt` and keeps production blocked.

## Batch Smoke Review

B/C/D/E smoke artifacts should be generated as a batch when their guards allow
it, then sent for one combined human visual review. This avoids stopping after
every individual class or scenario. The batch rule does not waive per-class
production gates: each class still needs its own smoke output, review request,
and approval before production for that class starts.

## Per-Class Smoke Gates

Stage 1 static RGB approval does not approve B/C/D/E production. Each data
class must have its own smoke output, review request, and approval file.

Read all class gates with:

```bash
scripts/world_model/dataset_class_review_status.sh
```

Require one class gate with:

```bash
scripts/world_model/require_dataset_class_smoke_approved.sh a_static_full
scripts/world_model/require_dataset_class_smoke_approved.sh b_dynamic_smoke
scripts/world_model/require_dataset_class_smoke_approved.sh c_frozen_dp_smoke
scripts/world_model/require_dataset_class_smoke_approved.sh d_future_teacher_smoke
scripts/world_model/require_dataset_class_smoke_approved.sh e_cosmos_predicted_smoke
```

The B/C/D/E gates should report `summary_missing` until those class smoke runs
exist. They must not be treated as approved just because A static RGB is
approved.
