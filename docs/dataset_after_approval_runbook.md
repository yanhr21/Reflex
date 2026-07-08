# Dataset After-Approval Runbook

Date: 2026-07-07

This runbook starts only after the Stage 1 RGB smoke is explicitly approved by
the user. It does not approve the smoke and does not launch anything by
itself.

## Current Approval Target

- Smoke run: `static_rgb/smoke05`
- Video: `experiments/maniskill/runs/01_dataset/static_rgb/smoke05/0.mp4`
- Review request:
  `experiments/maniskill/runs/01_dataset/static_rgb/smoke05/review_request.md`
- Approval file:
  `experiments/maniskill/runs/01_dataset/static_rgb/smoke05/human_review_approved.txt`

Approval must be explicit. The approval file must contain:

```text
approved=true
```

The structured helper can record a user decision:

```bash
scripts/world_model/record_dataset_smoke_review_decision.sh \
  --decision approved \
  --reviewer <name> \
  --notes <text>
```

Agents must not run this approval command unless the user explicitly approves
the smoke visual quality.

For the B/C/D batch smoke, use the batch helper only after the user explicitly
approves those dynamic smoke videos:

```bash
scripts/world_model/record_dataset_bcd_smoke_review_decision.sh \
  --decision approved \
  --reviewer <name> \
  --notes <text>
```

This writes class-local approval files for `dynamic_rgb/smoke01`,
`frozen_dp_dynamic/smoke01`, and `future_teacher/smoke01`.

## First Post-Approval Step

After approval, run the full static RGB render before dynamic production:

```bash
scripts/world_model/require_dataset_smoke_approved.sh
scripts/slurm/launch_dataset_static_rgb_full_tmux.sh
```

Expected output:

- either legacy single-run output
  `experiments/maniskill/runs/01_dataset/static_rgb/full01/`;
- or shard output such as
  `experiments/maniskill/runs/01_dataset/static_rgb/full_s00a/`,
  `full_s00b`, ... when scheduler windows require shorter runs;
- matching logs under `logs/01_dataset/static_rgb/`.

The full run manifest must pass:

```bash
scripts/world_model/validate_dataset_run_manifest.sh \
  experiments/maniskill/runs/01_dataset/static_rgb/full01
scripts/world_model/require_dataset_static_full_ready.sh
```

If the single `full01` run is replaced by shards, validate each shard manifest
and use the shard-aware full gate:

```bash
scripts/world_model/dataset_static_full_shards_status.sh
scripts/world_model/require_dataset_static_full_ready.sh
```

The shard gate accepts only full-production shards with
`dataset_smoke_only=false`, `human_review_required=false`,
`large_scale_production_allowed=true`, manifest validation, rendered videos,
review frames, and a cumulative `count >= 1000`. Short shard names such as
`full_s00a` should be used instead of long metadata-heavy run names.

Default resource policy:

- `1 GPU`;
- render RGB at `30 FPS` unless a manifest explicitly documents a diagnostic
  exception;
- reduce CPU/memory/walltime first if resources are hard to obtain;
- keep the job in a tmux-held Slurm allocation;
- if the request is valid but not immediately assigned, keep monitoring the
  queued tmux request instead of repeatedly cancelling for estimate movement;
- use render canary and minimal shader settings;
- retry previously bad render nodes only as smoke/canary diagnostics, and
  record the node evidence before accepting larger production on that node;
- record node evidence in the manifest/log.
- inspect `scripts/world_model/dataset_render_risk_status.sh` before launching
  a full render or dynamic smoke so current canary, Vulkan, HDF5, shader, and
  node-risk evidence are visible in one place.

## Dynamic Smoke Order

Do not start B/C/D/E production immediately after A full starts. Dynamic stages
must go through their own smoke and review flow. To reduce review overhead,
generate the available B/C/D/E smoke artifacts as a batch and request one
combined human review instead of stopping after every individual smoke.

Order after A full static RGB exists:

1. Launch the available B/C/D/E smoke jobs as a batch.
2. Human review of the batch.
3. Start production only for classes whose smoke has been explicitly approved.
4. E Cosmos-predicted cooperation remains skipped until B/D and
   Cosmos/readout validation exist.

Dynamic smoke launchers must not open immediately after Stage 1 approval.
They also require A full static RGB to be ready:

```bash
scripts/world_model/require_dataset_static_full_ready.sh
```

They also require the active dynamic-scene adapter to pass:

```bash
scripts/world_model/dataset_dynamic_adapter_status.sh
```

Current status on 2026-07-08: A full static RGB is ready as 20 shards. The
first B/C/D smoke batch was rejected by human visual review and archived under
`experiments/legacy/01_dataset/invalid_bcd_smoke_20260707/` because the videos
were about 2 seconds instead of complete 300-frame episodes, and C moved the
holed target during robot grasp / initial approach. Later invalid retries are
also archived under `experiments/legacy/01_dataset/`, including short-frame
outputs, `server53` render failures / diagnostic C success, and the B
`server36` canary timeout. The regenerated B/C/D smoke batch is now available
for one combined review:
`experiments/maniskill/runs/01_dataset/review/bcd_smoke_review_20260708.md`.
B and C are complete 300-frame, 30 FPS smoke videos; D is four complete
300-frame, 30 FPS teacher-smoke videos. C is a frozen-DP dynamic failure
(`success_once=false`, `success_at_end=false`) with target motion delayed to
step 120. D is complete smoke evidence but not a successful teacher claim
because its summary has `success_once=false`. Do not start B/C/D production
until the regenerated class smoke videos are explicitly approved. E remains
blocked until B production, D production, and held-out Cosmos/readout
validation exist. Existing legacy dynamic/bootstrap data may still be used
only for the already documented diagnostic/readout scope.

Read-only status:

```bash
scripts/world_model/dataset_post_approval_plan.sh
scripts/world_model/dataset_next_stage_status.sh
scripts/world_model/dataset_goal_status.sh
scripts/world_model/dataset_training_inputs_status.sh
scripts/world_model/dataset_class_review_status.sh
scripts/world_model/dataset_bcd_review_block_status.sh
```

Guarded launchers:

```bash
scripts/slurm/launch_dataset_batch_smoke_tmux.sh
scripts/slurm/launch_dataset_dynamic_rgb_smoke_tmux.sh
scripts/slurm/launch_dataset_frozen_dp_dynamic_smoke_tmux.sh
scripts/slurm/launch_dataset_future_frame_teacher_smoke_tmux.sh
scripts/slurm/launch_dataset_cosmos_predicted_coop_smoke_tmux.sh
```

The B/C/D smoke launchers must default to complete `300` frame, `30 FPS`
episodes / rollouts. A shorter dynamic smoke is invalid as visual review
evidence. For C frozen-DP dynamic failure, the holed target must remain static
during robot grasp / initial approach; target motion must be delayed until
after the configured motion start.

Each launcher must refuse if the Stage 1 review gate is still closed, if the
Stage A full RGB render is missing, if the stage runner is missing, or if the
stage runner fails source audit.

Each dynamic class also needs its own class smoke approval:

```bash
scripts/world_model/require_dataset_class_smoke_approved.sh b_dynamic_smoke
scripts/world_model/require_dataset_class_smoke_approved.sh c_frozen_dp_smoke
scripts/world_model/require_dataset_class_smoke_approved.sh d_future_teacher_smoke
scripts/world_model/require_dataset_class_smoke_approved.sh e_cosmos_predicted_smoke
```

These gates should remain closed until the corresponding smoke run exists and
has explicit human approval.

After class smoke approval, production launchers are:

```bash
scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh
scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh
scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh
```

Production targets:

- B dynamic RGB observation: `1000` episodes under
  `experiments/maniskill/runs/01_dataset/dynamic_rgb/prod01/<family>/`.
- C frozen-DP dynamic failure: `500` rollouts under
  `experiments/maniskill/runs/01_dataset/frozen_dp_dynamic/prod01/<family>/`.
- D future-frame cooperation teacher: `500` rollouts under
  `experiments/maniskill/runs/01_dataset/future_teacher/prod01/<family>/`.

The read-only production shard plan is:

```bash
scripts/world_model/dataset_bcd_production_shard_plan.sh
scripts/world_model/dataset_bcd_production_next_shard.sh
```

After regenerated B/C/D smoke approval is recorded, the preferred guarded
launcher submits only the next missing shard:

```bash
scripts/slurm/launch_dataset_bcd_next_production_shard_tmux.sh --execute
```

Without `--execute`, it is a dry-run and submits no Slurm jobs. If the smoke
approval files are missing, it prints the review blocker and exits before
Slurm. With `--execute`, it still submits at most one shard.

The lower-level guarded batch launcher is:

```bash
scripts/slurm/launch_dataset_bcd_production_shards_tmux.sh --execute
```

Without `--execute`, it is a dry-run and submits no Slurm jobs. With
`--execute`, it defaults to `--max-launches 1` so only one shard is queued at a
time. Use `--stage B|C|D` and `--family lr|fb|reverse|stop|sine|cont` for an
explicit shard, for example:

```bash
scripts/slurm/launch_dataset_bcd_production_shards_tmux.sh --execute --stage B --family lr --max-launches 1
```

Increase `--max-launches` only when intentionally queueing multiple reviewed
production shards.

The launcher skips shards that already validate as complete. If a matching
output directory exists but does not validate, it refuses to relaunch over it;
archive or diagnose the incomplete shard before continuing.

Production launchers still require A full RGB readiness, corresponding class
smoke approval, dynamic adapter source readiness, runner source audit, and
short output names. They must not launch while those gates are closed.
The E production launcher remains a guarded entrypoint until its prereqs and
real Cosmos-predicted runner are ready:

```bash
scripts/slurm/launch_dataset_cosmos_predicted_production_tmux.sh
```

Production default node exclusions should follow the current render-risk list
from `scripts/world_model/dataset_render_risk_status.sh` and the shard
launcher, currently:

```text
server10,server28,server30,server35,server36,server39,server43,server44,server46,server56,server57,server58,server59,server63
```

After production finishes, validate the run before treating it as a training
input candidate:

```bash
scripts/world_model/validate_dataset_production_run.sh b_dynamic_production
scripts/world_model/validate_dataset_production_run.sh c_frozen_dp_production
scripts/world_model/validate_dataset_production_run.sh d_future_teacher_production
```

The validator checks production flags, target counts, nonempty RGB videos,
motion/action trace files, and forbidden positive-policy/state-intervention
labels. A launched production job is not sufficient evidence by itself.

For B/C/D shard production, build registry indexes with:

```bash
scripts/world_model/build_dataset_production_shard_index.sh b_dynamic_production
scripts/world_model/build_dataset_production_shard_index.sh c_frozen_dp_production
scripts/world_model/build_dataset_production_shard_index.sh d_future_teacher_production
```

The full joint-training input gate also calls these validators:

```bash
scripts/world_model/require_dataset_training_inputs_ready.sh full_joint
```

Directory existence is not sufficient for full joint training. B/C/D/E
production must satisfy the validator targets before the full joint gate can
open.

## Existing Bootstrap Use

The B legacy bootstrap data may be used before new B production only for early
Cosmos/readout diagnostics:

- `experiments/maniskill/data/active/b_dynamic_legacy_bootstrap/train_samples.jsonl`
- `experiments/maniskill/data/active/b_dynamic_legacy_bootstrap/val_samples.jsonl`

It is not positive DP BC, not new production data, and not final method
evidence.

Before any training launch, call the training input guard:

```bash
scripts/world_model/require_dataset_training_inputs_ready.sh diagnostic_b_bootstrap
scripts/world_model/require_dataset_training_inputs_ready.sh full_joint
```

`diagnostic_b_bootstrap` is only for early Cosmos/readout checks on the legacy
bootstrap split. `full_joint` must remain blocked until A full RGB and new
B/C/D/E data exist.
