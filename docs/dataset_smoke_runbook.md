# Dataset RGB Smoke Runbook

Date: 2026-07-06

Purpose: render a small official static demo slice before any large data
production. The smoke output must be reviewed by the user before scaling.

## Entry Point

Launch from the login node:

```bash
scripts/slurm/launch_dataset_static_rgb_smoke_tmux.sh
```

If immediate GPU allocation is unavailable and the user wants progress to
continue, queue the same smoke in a tmux-held allocation:

```bash
QUEUE_IF_NEEDED=true CPUS_PER_TASK=2 MEMORY=12G TIME_LIMIT=00:20:00 \
  scripts/slurm/launch_dataset_static_rgb_smoke_tmux.sh
```

This is still not one-shot `sbatch`: the allocation is held by tmux, and the
actual render runs only after Slurm grants the interactive allocation.

If queued `salloc` is repeatedly revoked before node assignment, use the direct
`srun` tmux launcher:

```bash
scripts/slurm/launch_dataset_static_rgb_smoke_srun_tmux.sh
```

This is still not `sbatch`; the foreground `srun` is held by tmux and compute
starts only after Slurm grants resources.

This launcher only creates a tmux session and requests a Slurm allocation. The
actual render runs through:

```bash
scripts/slurm/run_dataset_static_rgb_smoke_in_allocation.sh
```

The runner refuses execution unless `SLURM_JOB_ID` and a non-extern
`SLURM_STEP_ID` are present.

## Default Output

```text
experiments/maniskill/runs/01_dataset/static_rgb/smoke05/
logs/01_dataset/static_rgb/smoke05.log
```

The run writes:

- `manifest.txt`
- `summary.json`
- rendered `*.mp4`
- `review/frames/*.png`

After the smoke finishes, prepare the human-review package:

```bash
scripts/world_model/prepare_dataset_smoke_review.sh
```

This writes `review_request.md` in the smoke output directory and lists the
videos / frames for review. It does not approve production.

Both manifest and summary mark:

```text
human_review_required=true
large_scale_production_allowed=false
```

## Render Environment

Required render env:

```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
DISPLAY=
HDF5_USE_FILE_LOCKING=FALSE
```

If render fails or hangs, inspect the specific job log and node evidence. The
old render note is `docs/legacy/RENDERING_CLUSTER_VULKAN.md`.

## Scaling Rule

Do not launch full official-1000 rendering or dynamic dataset rendering until
the user approves the smoke videos / frames by human visual review.

When waiting for approval, the goal is blocked on human review, not complete.

## Current Attempt Log

- 2026-07-06 20:07 CST: launched `dset_static_rgb_smoke01` with
  `salloc --immediate=60`. Slurm job name was `dset_rgb_smoke`.
- No GPU node was allocated in the immediate window.
- No render artifacts were produced.
- Visible idle GPU nodes were drain-state only at the time of check.
- 2026-07-06 20:12 CST: retried with lighter resources
  `2 CPU / 12G / 00:20:00`; no GPU node was allocated in the immediate
  window, and no render artifacts were produced.
- 2026-07-06 20:15 CST: queued `dset_static_rgb_smoke01` with
  `QUEUE_IF_NEEDED=true`, `2 CPU / 12G / 00:20:00`; Slurm job `168507`
  is pending with estimated start `2026-07-06T21:38:25` on `server44`.
- 2026-07-06 20:20 CST: job `168507` remains pending for priority. No
  manifest, summary, video, or review frames exist yet.
- 2026-07-06 20:23 CST: job `168507` remains pending for priority. No
  output directory exists yet; production gate still fails with
  `reason=summary_missing`.
- 2026-07-06 21:23 CST: job `168507` remains pending for priority. Scheduler
  estimate moved to `2026-07-06T23:15:35` on `server02`. Per user instruction,
  dataset smoke may use 1 GPU with reduced CPU/memory and may try previously
  bad render nodes as smoke-only; failures must be classified before scaling.
- 2026-07-06 21:26 CST: queued job `168507` was cancelled before node
  assignment; no render started and no artifacts were produced.
- 2026-07-06 21:29 CST: relaunched smoke with lower resources
  `1 CPU / 8G / 00:15:00`, still `1 GPU`, no node exclusions. New Slurm job
  is `168562`; `salloc` output is now captured in the smoke log.
- 2026-07-06 21:34 CST: job `168562` was revoked before node assignment.
  Added a direct `srun` tmux launcher as fallback for this cluster behavior.
- 2026-07-06 21:37 CST: launched direct `srun` tmux smoke. New Slurm job is
  `168581`, pending for priority; log records `srun: job 168581 queued and
  waiting for resources`.
- 2026-07-06 21:39 CST: direct `srun` job `168581` was cancelled before
  resources were assigned. No render started and no artifacts were produced.
- 2026-07-06 21:43 CST: relaunched direct `srun` tmux smoke on the `cpu`
  partition with `1 GPU / 1 CPU / 8G / 00:15:00` and no node exclusions, per
  the reduced-resource strategy. New Slurm job is `168586`, pending for
  priority with scheduler estimate `2026-07-06T23:03:41` on `server44`.
- 2026-07-06 22:03 CST: direct `srun` job `168586` was cancelled before any
  node was assigned. `scontrol` showed `SchedNodeList=server44` but runtime
  `00:00:00`; no render started and no artifacts were produced. Treat this as
  a scheduler/allocation failure, not a render failure.
- 2026-07-06 22:05 CST: scheduler `--test-only` for `cpu + 1 GPU + 1 CPU`
  moved to `2026-07-06T22:18:18` on `server44`. Relaunched reduced-resource
  direct `srun` smoke as job `168603` so pending time stays close to the
  predicted start window.
- 2026-07-06 22:36 CST: job `168603` started on `server39` and entered the
  real RGB render path. It failed after `00:01:20` with Vulkan
  `vk::Device::waitForFences: ErrorDeviceLost` / exit `134`, before any video
  or summary was produced. This is classified as a node / Vulkan device-loss
  smoke failure, not a dataset/protocol success and not a logic success.
- 2026-07-06 22:40 CST: archived failed `smoke01` run and log under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/`. Relaunched active smoke as
  `smoke02`, Slurm job `168635`, excluding `server39`; resource request remains
  `cpu` partition, `1 GPU / 1 CPU / 8G / 00:15:00`.
- 2026-07-06 22:47 CST: `smoke02` also ran on `server39` and failed with the
  same Vulkan `ErrorDeviceLost`. Diagnosis: the direct `srun` launcher appended
  `--exclude` after the executable script, so Slurm did not receive the exclude
  option. Patched `scripts/slurm/launch_dataset_static_rgb_smoke_srun_tmux.sh`
  so `--exclude` is added before the executable.
- 2026-07-06 22:50 CST: archived failed `smoke02` run and log under the
  external archive. Relaunched active smoke as `smoke03`, Slurm job `168659`,
  with corrected `--exclude=server39`. Test-only predicted `server30` around
  `2026-07-06T23:10:53`.
- 2026-07-06 23:01 CST: `smoke03` job `168659` remained pending but its start
  estimate slipped to `2026-07-07T00:04:08` on `server46`; a new test-only
  estimate predicted an earlier `server02` slot. Cancelled the not-yet-running
  job, archived its pending-only log as superseded, and patched the direct
  `srun` launcher to support explicit `NODELIST`.
- 2026-07-06 23:02 CST: relaunched `smoke03` as job `168691` with
  `NODELIST=server02` and `EXCLUDE_NODES=server39`. The log records both
  fields before `srun_output_start`.
- 2026-07-06 23:18 CST: `server02` targeting pushed job `168691` to
  `2026-07-07T03:16:46`; unconstrained test-only with only
  `EXCLUDE_NODES=server39` predicted an earlier `server46` slot. Cancelled the
  pending-only `168691`, archived its log, and relaunched `smoke03` as job
  `168759` with only `EXCLUDE_NODES=server39`.
- 2026-07-07 00:19 CST: job `168759` slipped to
  `2026-07-07T06:11:58` with scheduler reason
  `Nodes required for job are DOWN, DRAINED or reserved...` on `server57`.
  Test-only with `EXCLUDE_NODES=server39,server57` predicted the earlier
  `2026-07-07T04:24:50` slot on `server02`. Cancelled the pending-only
  `168759`, archived its log, and relaunched `smoke03` as job `168807` with
  `EXCLUDE_NODES=server39,server57`.

## 2026-07-08 B/C/D Correction Log

- Rejected and archived the regenerated B/C/D smoke batch after visual review
  showed the videos did not contain credible robot grasp / active peg
  manipulation. The invalid batch is under
  `experiments/legacy/01_dataset/invalid_bcd_smoke_20260708_no_grasp_task_motion/`.
- B was regenerated with legal official `pd_ee_delta_pose` demo actions plus
  dynamic target motion. The active B smoke is
  `experiments/maniskill/runs/01_dataset/dynamic_rgb/smoke01`, job `171244`
  on `server57`, 300 frames at 30 FPS, with `grasp_once=true`,
  `task_motion_quality_gate_passed=true`, `state_intervention=false`, and
  `snap_or_teleport=false`. B remains dynamic observation / failure data, not
  positive insertion data.
- C/D retry on `server28` failed render canary timeout before collection and
  was archived under
  `experiments/legacy/01_dataset/invalid_cd_smoke_20260708_server28_canary_timeout/`.
- C retry on `server60` failed render canary timeout before collection; D was
  cancelled after being assigned to the same newly diagnosed bad smoke node.
  The attempt is archived under
  `experiments/legacy/01_dataset/invalid_cd_smoke_20260708_server60_canary_timeout/`.
- C retry on `server34` failed render canary with Vulkan `ErrorDeviceLost`.
  The same retry also exposed stale D runner arguments after the D collector
  switch. The attempt is archived under
  `experiments/legacy/01_dataset/invalid_cd_smoke_20260708_server34_and_d_args/`.
- B/C/D/E smoke default exclusions now include `server28`, `server34`, and
  `server60`. Production defaults also exclude `server34` and `server60`
  until newer render evidence justifies changing that policy.
- 2026-07-07 00:46 CST: job `168807` remains pending for priority. Current
  `squeue --start` estimate is `2026-07-07T02:12:58` on `server02`. No output
  directory, manifest, summary, video, or review frame exists for the active
  `smoke03` attempt yet.
- 2026-07-07 01:40 CST: job `168807` started on `server59` and wrote the
  smoke manifest / copied working H5 and JSON, but made no video or summary.
- 2026-07-07 01:55 CST: job `168807` hit the 15 minute walltime while still
  at the beginning of `traj_0`; Slurm state is `TIMEOUT`, step exit was
  `143`, and no review video/frame was produced. Classify this as insufficient
  smoke resources / too-large smoke count (`COUNT=4`, `1 CPU`, `00:15:00`),
  not as a visual-quality success.
- 2026-07-07 01:56 CST: archived incomplete `smoke03` run/log under the
  external archive. Relaunched active smoke as `smoke04`, Slurm job `168881`,
  with `COUNT=1`, `1 GPU / 4 CPU / 16G / 00:45:00`, and
  `EXCLUDE_NODES=server39,server57,server59`.
- 2026-07-07 02:29 CST: job `168881` started on `server43` and wrote the
  manifest / working H5 / JSON.
- 2026-07-07 02:33 CST: job `168881` failed with Vulkan
  `vk::Device::waitForFences: ErrorDeviceLost` / exit `134`, before any video
  or summary was produced. This is another current node / Vulkan device-loss
  failure, not a data success.
- 2026-07-07 02:35 CST: archived failed `smoke04` run/log under the external
  archive. Patched `scripts/slurm/run_dataset_static_rgb_smoke_in_allocation.sh`
  to run `scripts/world_model/render_min_canary.py` before replay and to pass
  `--shader minimal` into `mani_skill.trajectory.replay_trajectory`.
- 2026-07-07 02:35 CST: relaunched active smoke as `smoke05`, Slurm job
  `168993`, with `COUNT=1`, `1 GPU / 4 CPU / 16G / 00:45:00`,
  `RUN_RENDER_CANARY=true`, `RENDER_SHADER_PACK=minimal`,
  `REPLAY_SHADER=minimal`, and
  `EXCLUDE_NODES=server39,server43,server57,server59`.
- 2026-07-07 02:47 CST: job `168993` remained pending on an unavailable
  `server63` candidate. `sinfo` showed `server63` in `comp` state. Cancelled
  the pending-only job, archived its log, and relaunched `smoke05` as job
  `169001` with `EXCLUDE_NODES=server39,server43,server57,server59,server63`.
  The matching test-only estimate is `2026-07-07T06:32:01` on `server02`.
- 2026-07-07 02:59 CST: job `169001` remains pending. `scontrol` now shows
  `SchedNodeList=server02` and start estimate `2026-07-07T03:43:01`; no active
  smoke output directory or manifest exists yet.
- 2026-07-07 03:46 CST: job `169001` missed the earlier `03:43` estimate and
  remains pending. `scontrol` now shows `SchedNodeList=server44` with start
  estimate `2026-07-07T05:07:13`. A fresh matching test-only estimate was later
  (`2026-07-07T06:35:08` on `server02`), so the active queued job is kept.
- 2026-07-07 04:48 CST: job `169001` started on `server44`. The minimal
  render canary passed and wrote `render_canary/frame.png`.
- 2026-07-07 04:49 CST: job `169001` completed successfully with exit `0`.
  The later A static RGB full-shard attempts also showed that `server20` can
  render valid 30 FPS shards, while `server58` is not safe for active RGB
  production.
- 2026-07-07 17:32 CST: A static RGB shard `full_s01b`, job `170132`, started
  on `server58` and failed in the render canary before replay with Vulkan
  `vk::Device::waitForFences: ErrorDeviceLost` / exit `134`. No video or
  summary was produced. Treat `server58` as a current device-loss node for RGB
  production and keep it in the active exclude list.
- 2026-07-07 17:56 CST: A static RGB shard `full_s02b`, job `170191`, started
  on `server56`. The render canary reached `render_gym_start` and then timed
  out after `3m` with exit `124`, before any replay, video, or summary. Treat
  `server56` as a current canary-timeout render-risk node for RGB production
  and keep it in the active exclude list.
- 2026-07-07 18:02 CST: A static RGB shard `full_s02b`, job `170201`, started
  on `server28`. The render canary reached `render_gym_start` and then timed
  out after `3m` with exit `124`, before replay, video, or summary. Treat
  `server28` as a current canary-timeout render-risk node for RGB production
  and keep it in the active exclude list.
  Active smoke `smoke05` produced one RGB video, three review frames,
  `summary.json`, and `review_request.md`. The summary keeps
  `human_review_required=true` and `large_scale_production_allowed=false`.
  Production remains blocked until the user approves the review artifacts.
- 2026-07-08 11:44 CST: B dynamic RGB smoke job `171196` started on
  `server36`. The render canary reached `render_gym_start` and timed out with
  exit `124`; collection did not start. The invalid active output was archived
  under
  `experiments/legacy/01_dataset/invalid_b_smoke_20260708_server36_canary_timeout/`.
  Treat `server36` as a current canary-timeout render-risk node for RGB smoke
  and keep it in the default smoke exclude list.
- 2026-07-08 11:44 CST: C frozen-DP dynamic smoke job `171197` started on
  `server57` and completed a 300-frame failure smoke with
  `success_once=false`, `state_intervention=false`, and
  `snap_or_teleport=false`. This is valid smoke evidence, but `server57`
  remains conservative production-risk evidence because earlier scheduler /
  resource issues were observed.
- 2026-07-08 11:44 CST: D future-frame teacher smoke job `171198` started on
  `server10` and completed four 300-frame teacher-smoke videos. This is valid
  smoke evidence, but the summary has `success_once=false`, so it is not a
  successful teacher-controller claim. `server10` remains in production
  default excludes because active production launchers are conservative after
  prior render-risk evidence.
- 2026-07-08 11:49 CST: B dynamic RGB smoke job `171210` started on
  `server10` and completed one 300-frame smoke video with
  `state_intervention=false` and `snap_or_teleport=false`.
- 2026-07-08: Earlier B/C/D retries on `server53` were archived under
  `experiments/legacy/01_dataset/invalid_bcd_smoke_20260708_server53_c_success/`:
  B hit render-canary timeout, D hit Vulkan `ErrorDeviceLost`, and C was a
  300-frame diagnostic success instead of the required frozen-DP failure.
  Treat `server53` as both current canary-timeout and device-loss
  render-risk evidence.

## Production Approval Guard

Before any full render wrapper is added or launched, call:

```bash
scripts/world_model/require_dataset_smoke_approved.sh
```

It currently defaults to the active smoke run `static_rgb/smoke05`. It requires
`summary.json` from a completed smoke run and an explicit
`human_review_approved.txt` containing:

```text
approved=true
```

Without that file, large-scale production must stay blocked.

The guarded full static RGB launcher is:

```bash
scripts/slurm/launch_dataset_static_rgb_full_tmux.sh
```

It calls `require_dataset_smoke_approved.sh` on the active smoke run
`static_rgb/smoke05` before creating tmux or requesting Slurm resources. Until
the smoke exists and the user-approved file is present, this launcher must
refuse to run. The full launcher uses the direct `srun` tmux launcher rather
than the older queued `salloc` path, and defaults to `PARTITION=cpu`,
`GPUS=1`, render canary enabled, and `REPLAY_SHADER=minimal`.

When it is eventually allowed, full production must set
`dataset_smoke_only=false`, `human_review_required=false`, and
`large_scale_production_allowed=true` in its manifest / summary. Smoke runs
must keep `dataset_smoke_only=true`, `human_review_required=true`, and
`large_scale_production_allowed=false`.

Use `scripts/world_model/dataset_full_static_status.sh` to inspect the
eventual static RGB production artifact status. It reports both the legacy
single-run `static_rgb/full01` path and the current shard-aware
`static_rgb/full_s*` aggregate gate.

The active read-only render risk helper is:

```bash
scripts/world_model/dataset_render_risk_status.sh
```

It summarizes the current smoke evidence, required
`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` /
`HDF5_USE_FILE_LOCKING=FALSE` / canary / minimal-shader settings, and the
current smoke-era node observations. It is not a static cluster blacklist:
future production must still use job-local canaries and manifest evidence
before treating a node as good or bad for a run.
