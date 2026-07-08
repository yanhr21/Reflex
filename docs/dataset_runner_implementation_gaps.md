# Dataset Runner Implementation Gaps

Date: 2026-07-07

This document tracks what is still missing before B/C/D/E data collection can
produce training-scale data. It is not a substitute for data collection.

## Current Gate

Stage 1 static RGB smoke exists at:

- `experiments/maniskill/runs/01_dataset/static_rgb/smoke05/0.mp4`

All later stages remain blocked until `smoke05` is human-approved. The gate
file is:

- `experiments/maniskill/runs/01_dataset/static_rgb/smoke05/human_review_approved.txt`

The required content is:

```text
approved=true
```

## Existing Bootstrap Data

An older dynamic RGBD dataset has been registered for limited bootstrap use:

- `experiments/maniskill/data/active/b_dynamic_legacy_bootstrap/`

Read-only inspection on 2026-07-07 found 1000 RGBD H5 files and 1000 MP4
files with filename scenario counts:

- `hole_constant`: 167
- `hole_move_stop`: 167
- `hole_reverse`: 167
- `none`: 166
- `peg_disturb`: 166
- `peg_drop`: 167

This bootstrap data can support early Cosmos dynamic future/readout checks and
schema inspection, but it does not replace new B/C/D/E production data and
cannot provide positive DP BC or final method evidence without revalidation.

## Existing Launch Entrypoints

These launchers are guarded Slurm/tmux entrypoints. They refuse before
creating tmux or submitting Slurm if the Stage 1 review gate or runner check
fails. Once Stage 1 is approved and a stage runner exists and passes audit,
they submit the stage smoke through:

- `scripts/slurm/launch_dataset_stage_smoke_tmux_common.sh`

- `scripts/slurm/launch_dataset_dynamic_rgb_smoke_tmux.sh`
- `scripts/slurm/launch_dataset_frozen_dp_dynamic_smoke_tmux.sh`
- `scripts/slurm/launch_dataset_future_frame_teacher_smoke_tmux.sh`
- `scripts/slurm/launch_dataset_cosmos_predicted_coop_smoke_tmux.sh`

The shared readiness guard is:

- `scripts/world_model/require_dataset_stage_ready.sh`

Dynamic B/C/D/E stage readiness also depends on the full static RGB render:

- `scripts/world_model/require_dataset_static_full_ready.sh`

This prevents B/C/D/E smoke launchers from opening immediately after Stage 1
smoke approval. A full `static_rgb/full01` render must exist and pass manifest
validation first.

The read-only post-approval status helper is:

- `scripts/world_model/dataset_post_approval_plan.sh`

It reports each B/C/D/E command, common launcher, output/log path, and the
exact readiness reason returned by `require_dataset_stage_ready.sh`. It must
remain read-only and must not create tmux sessions, Slurm jobs, run
directories, or log directories.

The runner source audit guard is:

- `scripts/world_model/audit_dataset_runner_source.sh`

The collector source audit guard is:

- `scripts/world_model/audit_dataset_collector_source.sh`

Stage readiness requires both runner and collector source audits. The runner
guard verifies allocation/output/log/review behavior, while the collector guard
checks the actual data-generation source for active dataset class, legal
`env.step` execution, RGB/video writing, trace/summary writing, loss-role
fields, method/teacher evidence flags, and forbidden state-edit / legacy
Oracle paths.

The shared runtime-context guard for future in-allocation runners is:

- `scripts/world_model/require_dataset_runtime_context.sh`

The dynamic-scene adapter guard is:

- `scripts/world_model/dataset_dynamic_adapter_status.sh`

It checks the active dynamic adapter path:

- `scripts/world_model/active_dynamic_peg_adapter.py`

As of 2026-07-07 the source-audited adapter exists and exposes only a
kinematic-target command path plus motion-trace helpers. It is not data and is
not a runner. It still needs runtime validation inside a compute-node Slurm
allocation because the original `PegInsertionSide-v1` target box is a
kinematic actor and ManiSkill's public wrapper does not document a stable
kinematic-target helper. If runtime SAPIEN does not expose that command, the
adapter is required to fail instead of falling back to direct pose writes.

The recoverable old dynamic scripts moved the target through direct pose/state
paths. Active dynamic data production must not silently revive that pattern. A
reviewed adapter must provide continuous, logged target / hole / peg motion,
RGB evidence, manifest fields, and explicit `state_intervention` /
`snap_or_teleport` labels without per-step pose-edit or state-restore
shortcuts.

Future B/C/D/E runners should source this guard after setting `RUN_GROUP`,
`RUN_NAME`, and any output/log overrides. The guard refuses login-node
execution, requires a non-extern Slurm step, enforces active output/log roots,
checks short run names, validates smoke review flags, and exports the render
environment.

Once a runner exists, readiness still fails unless this audit passes. The audit
rejects old routes, state edits, source-state restore, future-label controller
shortcuts, hidden manual finishers, unreadable long names, and missing
allocation/output/review guard fields.

For B/C/D/E, readiness also fails unless
`scripts/world_model/dataset_dynamic_adapter_status.sh` passes. This only means
the active source path is present and auditable; it does not replace the
required compute-node smoke that proves target motion is continuous, rendered,
and logged.

After a production run validates, it still must be indexed before training.
The production index builder is:

- `scripts/world_model/build_dataset_production_index.sh`

The read-only index status helper is:

- `scripts/world_model/dataset_production_index_status.sh`

`full_joint` training readiness requires both the production run validators and
these registry indexes.
The index builder also checks the source `summary.json` and `manifest.txt`
before writing JSONL, so a copied run directory, wrong class, wrong run group,
wrong method/teacher flag, or state-intervention-expected mismatch cannot
silently enter the training registry.

The common launcher defaults are intentionally conservative:

- `1 GPU`;
- `1 CPU / 8G / 00:30:00` unless overridden;
- `PARTITION=cpu`;
- render canary enabled;
- `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`;
- `DISPLAY=`;
- `HDF5_USE_FILE_LOCKING=FALSE`;
- short paths under `experiments/maniskill/runs/01_dataset/<class>/smokeNN`
  and `logs/01_dataset/<class>/smokeNN.log`.

## Runner Status

The active dynamic adapter must be implemented and audited before B/C/D/E
in-allocation runners can become meaningful.

Current source status on 2026-07-07:

- B dynamic RGB observation runner source exists:
  `scripts/slurm/run_dataset_dynamic_rgb_smoke_in_allocation.sh`
- B dynamic RGB observation collector source exists:
  `scripts/world_model/collect_dynamic_rgb_observation_smoke.py`
- C frozen-DP dynamic failure runner source exists:
  `scripts/slurm/run_dataset_frozen_dp_dynamic_smoke_in_allocation.sh`
- C frozen-DP dynamic failure collector source exists:
  `scripts/world_model/collect_frozen_dp_dynamic_failure_smoke.py`
- D future-frame cooperation teacher:
  `scripts/slurm/run_dataset_future_frame_teacher_smoke_in_allocation.sh`
- E Cosmos-predicted cooperation:
  `scripts/slurm/run_dataset_cosmos_predicted_coop_smoke_in_allocation.sh`

The B, C, D, and E runners are source-ready only. They have not been
runtime-smoked and have not produced data. They will remain blocked until
Stage 1 smoke approval and A full RGB readiness. E has additional prereqs:
B production validation, D production validation, and held-out Cosmos/readout
validation. Do not create placeholder runners that write fake success
artifacts. A runner is ready as data collection only if it steps the active
ManiSkill task inside a compute-node Slurm allocation, records legal
controller-facing actions or explicit zero-action observation traces, writes
RGB evidence, and writes manifests matching `docs/dataset_manifest_schema.md`.

## Missing Legacy Source Code

Read-only source inspection on 2026-07-07 found pycache entries for several
old dynamic generators, but not their active `.py` source files under
`scripts/world_model/`:

- `generate_cosmos3_fix3_hard_dynamic_teacher`
- `generate_cosmos3_fix3_late_trigger_dynamic_experts`
- `generate_cosmos3_fix3_successful_dynamic_dataset`
- `render_cosmos3_maniskill_sft_dataset`

Because the source files are missing, these old generators must not be wrapped
as active B/C/D/E runners from pycache. Either recover the original source
from the external archive / git history, or implement new reviewed runners
against the active ManiSkill route and manifest schema.

Git history now confirms the files are recoverable from the parent of commit
`852976723d813352cabd5690f0acaab910f86c4e`, but they contain old-route
defaults and state-edit/render-from-state paths. The audit is recorded in:

- `docs/legacy_dynamic_source_recovery.md`

The recovered source is reference material only until adapted and reviewed.

## Runner Rules

Every B/C/D/E runner must:

- refuse login-node execution;
- run only inside a Slurm `srun` step held by tmux;
- source `scripts/world_model/require_dataset_runtime_context.sh` or implement
  equivalent checks before doing any data generation, rendering, rollout, or
  Python import;
- request only 1 GPU for smoke unless the user explicitly changes this;
- use short output paths under
  `experiments/maniskill/runs/01_dataset/<class>/<try_or_tag>/`;
- write logs under `logs/01_dataset/<class>/<try_or_tag>.log`;
- set `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`, `DISPLAY=`, and
  `HDF5_USE_FILE_LOCKING=FALSE`;
- run the minimal render canary before any RGB replay or render;
- write `manifest.txt`, `summary.json`, videos, and review frames;
- set `human_review_required=true` and
  `large_scale_production_allowed=false` for smoke;
- never call `set_pose`, `set_state`, `set_state_dict`, saved-state replay,
  source-state restore, geometric final placement, or hidden future labels as
  deployed controller inputs.

## Stage-Specific Implementation Notes

B dynamic RGB observation can include failures and does not need successful
insertion. It must record continuous target / hole / peg motion, RGB video,
state/action timing, and future target labels for Cosmos future learning.
Failed action chunks cannot receive positive DP BC loss.

C frozen-DP dynamic failure must run the official DP checkpoint unchanged in
dynamic scenes and record why it fails or succeeds. Target-assisted
self-insertion must be labeled negative/diagnostic, not success.

D future-frame cooperation teacher may use ground-truth future target
trajectory only for data generation. The source-ready smoke runner is:

- `scripts/slurm/run_dataset_future_frame_teacher_smoke_in_allocation.sh`

The collector is:

- `scripts/world_model/collect_future_frame_teacher_smoke.py`

It must execute legal controller actions and mark artifacts as teacher-only
with `teacher_evidence_allowed=true` and `method_evidence_allowed=false`. It
still needs runtime validation inside a compute-node Slurm allocation and
human visual review before production.

E Cosmos-predicted cooperation can start only after B/D and Cosmos/readout
validation. It replaces ground-truth future frames with Cosmos/readout
predictions and records prediction uncertainty and final physical outcome.
The explicit readiness guard is:

- `scripts/world_model/require_dataset_cosmos_predicted_prereqs_ready.sh`

This guard is read-only and must fail until B production, D production, and a
held-out Cosmos/readout validation summary all pass.

Current source status on 2026-07-07: the E smoke collector and in-allocation
runner exist:

- `scripts/world_model/collect_cosmos_predicted_coop_smoke.py`
- `scripts/slurm/run_dataset_cosmos_predicted_coop_smoke_in_allocation.sh`

Both source audits pass. This does not make E runnable yet. The runner is
designed to consume precomputed Cosmos/readout predictions and to fail if the
prediction JSONL or prereq validation summary is missing. Do not bypass the
prereq guard by substituting ground-truth future labels or synthetic
predictions.
