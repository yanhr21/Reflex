# Slurm And Hygiene TODO

## Slurm

- [ ] Every heavy rollout/training job has an sbatch wrapper.
- [ ] Node policy: there is no standing bad-node list. Do not maintain,
      consult, or recreate one. Use live Slurm state and job-local canaries for
      each submission; older node-name entries in this file are historical
      records only and must not be reused as default exclusions.
- [x] Current rendering jobs export `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`.
- [x] Current rendering jobs clear `DISPLAY`.
- [x] Current HDF5 replay/render jobs export `HDF5_USE_FILE_LOCKING=FALSE`.
- [x] Controller-run inspector now defaults `HDF5_USE_FILE_LOCKING=FALSE` so
      lightweight post-run inspection does not fail on shared-filesystem lock
      errors.
- [x] Current rendering jobs use live node policy defaults; no static render
      blacklist is maintained.
- [x] Large RGB-D generation uses dense GPU allocation. Up to 8 nodes / 64 GPUs
      is allowed only when using full-node or otherwise dense allocations; do
      not occupy many servers with one GPU each.
- [x] Current data-generation jobs write a manifest with command, environment, node, seed, and
      output path.
- [x] 2026-06-03 00:14 live account/resource probe:
      `sacctmgr` reports current user `yanhongru` is associated only with
      account `mayi`. `gaosh`, `engram`, and `test` partitions currently
      allow account `null`, not `mayi`, and `sbatch --test-only` rejects
      default, `mayi`, and `null` submissions for this user on those
      partitions. `gpux` and `mgpu` are drained/inactive. New 1/2/4GPU
      ten-minute probes on usable partitions start later than active repair
      `99316`; therefore do not submit duplicate render/state jobs just to
      occupy the queue. Keep the approximately half-hour queue/artifact check
      cadence unless a job starts or artifacts appear.
- [x] 2026-06-03 00:36 short state-smoke submission support:
      `evaluate_state_rebinding_smoke_matrix.sbatch` now accepts
      `STATE_MATRIX_CASES` for exact-case subsets, allowing future
      `sbatch --test-only` probes and submissions to target shorter one-GPU
      state smoke jobs when Slurm has short backfill holes. The wrapper still
      records the selected cases in the manifest and preserves the state-only
      scaffold boundary.
- [x] 2026-06-03 00:45 resource/action check:
      queue check found `99316` had run and failed on `server21`, while
      `99346` had started on `server42`. Short one-GPU split-state dry-runs
      and replacement repair dry-runs were not earlier than the active/running
      paths until the post-failure replacement repair was submitted. Canceled
      dead dependency branch `99332-99345` and submitted `99590-99604`.
      Current next useful checks are artifact/log checks for running `99346`
      and queued `99590`, not repeated resource probing before about
      `2026-06-03T01:10+08:00` unless a job starts or artifacts change.

Current wrappers:

- `scripts/slurm/collect_dynamic_state_rollouts.sbatch`
- `scripts/slurm/render_dynamic_rgbd_dataset.sbatch`
- `scripts/slurm/render_dynamic_rgbd_dataset_dense.sbatch`
- `scripts/slurm/replay_peg_official_rgbd.sbatch`
- `scripts/slurm/render_state_trajectory_frames.sbatch`
- `scripts/slurm/inspect_world_model_ensemble.sbatch`
- `scripts/slurm/inspect_continuability_model_ensemble.sbatch`
- `scripts/slurm/evaluate_object_world_model_ensemble.sbatch`
- `scripts/slurm/calibrate_continuability_threshold.sbatch`
- `scripts/slurm/inspect_rebinding_controller_run.sbatch`
- `scripts/slurm/inspect_rgbd_dataset.sbatch`
- `scripts/slurm/train_rgbd_slot_extractor_ensemble_4gpu.sbatch`
- `scripts/slurm/inspect_rgbd_slot_extractor_ensemble.sbatch`
- `scripts/slurm/summarize_rebinding_controller_inspections.sbatch`
- `scripts/slurm/cleanup_gated_video_branches.sbatch`

Post-training inspection jobs:

- `94620`: CPU-partition inspection for object world model job `94442`,
  dependency `afterany:94442`, completed and passed.
- `94619`: canceled before running because its submitted Slurm script snapshot
  lacked the strict `REQUIRE_COMPLIANT` post-training gate.
- `94863`: strict CPU-partition inspection for larger-shard object world model
  job `94539`, dependency `afterany:94539`, with `REQUIRE_COMPLIANT=true`.
- `94680`: CPU-partition inspection for larger-shard object world model
  backfill job `94679`, dependency `afterany:94679`.
- `94747`: CPU-partition inspection for larger-shard object world model
  fast-fit duplicate `94746`, dependency `afterany:94746`.
- `94810`: CPU-partition inspection for alternate node-selection route larger-shard object
  world model retry `94809`, dependency `afterany:94809`.
- `94618`: CPU-partition inspection for `C_pi` job `94471`, dependency
  `afterany:94471`, completed and passed.
- `94658`: CPU-partition inspection for larger-label `C_pi` job `94542`,
  dependency `afterany:94542`.
- `94684`: canceled CPU-partition inspection for larger-label `C_pi` backfill
  job `94683`; the whole backfill chain was canceled after primary training
  `94542` started.

Post-inspection world-model prediction evaluation jobs:

- `94672`: CPU-partition learned-vs-static/CV prediction evaluation for object
  world model job `94442`, dependency `afterok:94620`, completed.
- `94673`: canceled before running with old non-strict inspection chain.
- `94864`: CPU-partition learned-vs-static/CV prediction evaluation for
  larger-shard object world model job `94539`, dependency `afterok:94863`,
  with `REQUIRE_INSPECTION=true` and `REQUIRE_COMPLIANT=true`.
- `94681`: CPU-partition learned-vs-static/CV prediction evaluation for
  larger-shard object world model backfill job `94679`, dependency
  `afterok:94680`.
- `94748`: CPU-partition learned-vs-static/CV prediction evaluation for
  larger-shard object world model fast-fit duplicate `94746`, dependency
  `afterok:94747`.
- `94811`: CPU-partition learned-vs-static/CV prediction evaluation for old
  alternate node-selection route larger-shard object world model retry `94809`, dependency
  `afterok:94810`.

Inspection elapsed-time rule:

- Ensemble inspectors require final checkpoints/metrics plus at least
  `10800s` elapsed. They use `metrics.json` `total_elapsed_seconds` when
  available, otherwise the latest history row and the member
  `member_manifest.txt` start/complete timestamps. Running or incomplete jobs
  still report non-compliant.
- Current user rule override from `2026-06-04`: model-training evidence must
  be at least 1 H200 GPU and at least 3 hours. Do not force 4 H200 GPUs as the
  minimum. If a shorter/non-H200 run produces weak results, it is a
  smoke/code-path result only and cannot be used to call the direction failed.
  Current and future compliant training jobs must use live Slurm state rather
  than a standing node exclusion list; any node-specific exclusion is
  job-local evidence only. Prefer a reusable 1-H200 allocation when it reduces
  queue churn; 1 day is the default long request unless live forecasts and
  experiment duration justify longer.
- `2026-06-04 19:21+08:00` one-H200 allocation audit: tmux session
  `h200_1gpu_pool` submitted Slurm allocation `105385` with `1` H200,
  `8` CPUs, `64G`, `1-00:00:00`, no node exclusions. It started on `server62`
  at `2026-06-04T19:18:38+08:00`; repeated `srun --jobid=105385` commands
  confirmed H200 visibility and PyTorch CUDA availability, so the same
  allocation can run multiple experiments without releasing. The existing
  exact1000 4-H200 slot job `105236` was kept running because it already held
  useful resources. Future training wrappers/inspectors were patched to accept
  `>=1xH200/>=3h` evidence while preserving exact-count and RGB-D method
  gates.
- `2026-06-04 19:41+08:00` second one-H200 allocation request:
  live `scontrol` shows current relevant jobs `105236`, `105385`, and
  dependency-pending `105429` have `ExcNodeList=(null)`, so the current delay
  is not caused by a standing bad-node list. No-exclusion probes for `1` H200
  on `cpu` forecast the same start for `03:00:00`, `08:00:00`, and
  `1-00:00:00`; submitted real tmux allocation `h200_1gpu_pool2` / Slurm
  `105502` with `1` H200, `8` CPUs, `64G`, `1-00:00:00`, and no exclusions.
  The pane is preloaded to run an exact1000 RGB-D slot backup when granted, so
  the allocation is not intentionally left idle. Also added `--paths-file`
  support for RGB-D slot training and changed the wrapper to use it, avoiding
  fragile argv expansion of all `1000` H5 paths while preserving all exact
  data and evaluation gates.
- `2026-06-04 19:49+08:00` current node GPU-visibility reroute:
  allocation `105502` started on `server13` but live canaries inside the
  allocation found no usable CUDA device (`nvidia-smi` no devices, PyTorch
  `torch_cuda_available False`). Canceled `105502` after `00:02:38` to avoid
  invalid CPU-only RGB-D slot training. This is a scheduling/node GPU
  visibility failure, not method evidence. Added a default CUDA-required gate
  to RGB-D slot training. Submitted replacement `105535` with only
  `server13` excluded, tied to this current evidence; this does not create a
  standing node exclusion list.
- `2026-06-04 20:01+08:00` CUDA-gated tail cleanup and live node policy:
  replacement chain
  `full1000_strongslot128_grid8_taskloss_1h200_tail_cudagate_20260604_1953`
  passed contract audit `208/208`, then superseded tail `105429-105447` was
  canceled with zero elapsed time, no allocated TRES, and no assigned nodes.
  Current relevant jobs still do not use a standing bad-node list:
  `105236` has `ExcNodeList=(null)`, `105385` has `ExcNodeList=(null)`,
  `105553` has `ExcNodeList=(null)`, and `105535` has only
  job-local `ExcNodeList=server13` from the immediately observed CUDA failure
  in `105502`. `105535` is now running on `server28`, and its runner emitted
  `cuda_available=true` with `REQUIRE_CUDA=true`. Patched
  `run_rgbd_slot_backup_in_allocation.sh` so future allocation-runner
  manifests record live `job_local_excluded_nodes` and describe the resource
  policy as `live_node_policy_no_standing_blacklist`.
- `2026-06-04 20:14+08:00` no-standing-bad-node correction:
  live `scontrol` shows `105236`, `105385`, and dependency-pending `105553`
  all have `ExcNodeList=(null)`; `105535` has only job-local `server13`
  because allocation `105502` just showed no visible CUDA device on that node.
  Fixed RGB-D slot/generic triage false positives so normal
  `MIN_TRAIN_SECONDS` manifest text no longer classifies a running job as an
  undersized-training refusal, and updated the diagnostic boundary text to the
  current exact-count / `>=1` H200 / `>=3h` policy. Added an immediate CUDA
  canary to the allocation runner before exact1000 H5 scanning. Submitted one
  additional no-exclusion 1-H200/1-day tmux allocation
  `h200_1gpu_pool4` / Slurm `105646`; it is pending with
  `StartTime=2026-06-04T21:31:41`, `SchedNodeList=server43`, and
  `ExcNodeList=(null)`. Do not submit a large fanout of more allocation
  requests unless live forecasts or failures justify it.
- `2026-06-04 20:21+08:00` CPU-first GPU partition ordering:
  `sbatch --test-only` showed `--partition=cpu,gpu` for 1-H200 jobs forecasts
  tonight, while `--partition=gpu,cpu` forecasts `2026-06-10`; therefore
  active dependency-pending GPU jobs were updated to prefer `cpu,gpu`:
  `105238`, `105553`, and `105556/105559/105562/105565/105568`. No job-local
  node exclusions were added. Future current-chain defaults were patched in
  predicted-slot export, RGB-D-derived world-model, controller-video wrappers,
  and the coordconv submitter; `bash -n` passed. This is queue-placement
  hygiene only, not a method/evaluation change.
- `2026-06-04 20:25+08:00` allocation-runner CUDA canary repair:
  `105646` failed on `server62` after `00:00:12` because the runner canary
  checked CUDA in the direct `salloc` shell, but training runs under `srun`.
  Correct `srun` canaries in `105385` and `105535` saw H200/CUDA, so do not
  add server62 to any standing exclusion list. Patched the runner to perform
  CUDA canary through `srun --gpus-per-task=1`, then submitted no-exclusion
  replacement allocation `105707` in tmux `h200_1gpu_pool5`. Final live
  `scontrol` snapshot shows `StartTime=2026-06-04T23:00:00`,
  `SchedNodeList=server18`, and `ExcNodeList=(null)`.
- `2026-06-04 20:34+08:00` one-H200 rolling allocation correction:
  user clarified that small jobs should be requested promptly and that old
  bad-node history must not explain current queue behavior. Live audit found
  no standing exclusions on the main path: `105236`, `105385`, `105238`,
  `105553`, and controller-video jobs have `ExcNodeList=(null)`;
  `105535` has only job-local `server13` from current CUDA failure `105502`.
  Replacement allocation `105707` then started on `server13` with no submitted
  exclusions and the fixed `srun` CUDA canary failed before dataset load
  (`nvidia-smi_returncode=255`, `torch_cuda_available=false`,
  `failure_class=scheduling_node_gpu_visibility`). This repeats the current
  server13 GPU-visibility issue and is scheduling evidence only, not method
  evidence. Fresh one-H200 probes showed `03:00:00`, `08:00:00`, and
  `1-00:00:00` all forecast the same start under `cpu`/`cpu,gpu`, while
  `gpu,cpu` forecast `2026-06-10`; submitted tmux allocation
  `h200_1gpu_pool6` / Slurm `105743` with `1` H200, `8` CPUs, `64G`,
  `1-00:00:00`, exact1000 seed `2000`, and fixed `srun` CUDA canary. After
  the repeated `server13` canary failure, updated only this pending replacement
  to job-local `ExcNodeList=server13`. It then started earlier than forecast
  on `server43` at `2026-06-04T20:34:44`; fixed `srun` CUDA canary passed
  with one visible H200 and `torch_cuda_available=true`, and exact1000 seed
  `2000` training started at `20:35:10`. Current training remains incomplete:
  `105236` is running with all four members at epoch-0 batch `1100`, `105385`
  is a running one-H200 exact1000 backup, `105535` is a running one-H200
  exact1000 backup still in dataset loading/scanning, `105743` is a running
  one-H200 exact1000 backup just started scanning/loading, and no
  checkpoint/metrics have passed strict RGB-D slot inspection yet.
- `2026-06-04 20:36+08:00` visualization-budget default hygiene:
  active predicted-slot visual review `105259` already exports
  `SAMPLE_FILES=10`; the active chain manifest records
  `slot_visual_sample_files=10`, `video_review_sample_count=10`, and the
  boundary that visual sample counts limit diagnostic rendered artifacts only.
  Patched current default wrappers so future direct submissions also default
  to `10`: `visualize_rgbd_predicted_slots.sbatch`,
  `inspect_video_artifacts.sbatch`, and `submit_rgbd_distributed_shards.sh`.
  `bash -n` passed. This is visual-artifact budget hygiene only, not a data,
  training, export, controller, metric, or evaluation-gate change.
- `2026-06-04 20:39+08:00` persistent allocation manifest hygiene:
  patched `run_rgbd_slot_backup_in_allocation.sh` to record
  `keep_allocation_after_run` and an explicit `allocation_reuse_reason` in
  future allocation-runner manifests. This documents that successful one-H200
  allocations are intended to remain reusable for strict follow-up commands
  instead of releasing and requeueing. The later `20:50` user scheduling
  override superseded the temporary no-new-allocation decision made at this
  timestamp.
- `2026-06-04 20:50+08:00` user scheduling override and one-H200 rolling
  requests:
  added `scripts/slurm/launch_rgbd_1h200_allocation_tmux.sh` to submit
  reproducible tmux-backed one-H200 `salloc` requests that immediately run the
  exact1000 RGB-D slot backup after the fixed `srun` CUDA canary and keep the
  allocation open after completion. `bash -n` passed. Initial one-day
  requests `105826/105827` were canceled while pending with zero allocation
  after shorter `03:30:00` through `12:00:00` probes forecast earlier
  backfill. Replacement 12-hour requests are `105868` (`h200_1gpu_pool7`,
  no exclusions, seed `2100`) and `105867` (`h200_1gpu_pool8`, only
  job-local `ExcNodeList=server13`, seed `2200`). This is a small rolling
  batch, not a broad fanout. No standing node exclusion list is in use:
  `105236`, `105385`, `105238`, `105553`, and controller-video jobs have
  no exclusions; `server13` is only current job-local evidence from repeated
  pre-dataset CUDA visibility failures in `105502` and `105707`.
- `2026-06-04 20:56+08:00` strict one-H200 backup assembly preflight:
  added `scripts/world_model/assemble_rgbd_slot_backup_ensemble.py` to assemble
  completed one-H200 backup members only after exact1000, H200, `>=10800s`,
  RGB-D/proprio-boundary, metrics, manifest, and checkpoint checks pass.
  Current dry-run over `105385`, `105535`, and `105743` returned status `65`
  because all backups are still incomplete, which is the intended guard.
  This is fallback hygiene only; it does not change live allocation strategy,
  exact1000 gates, or downstream RGB-D-derived evidence requirements.
- `2026-06-04 21:05+08:00` live forecast versus test-only correction:
  after the user challenged the earlier queue interpretation, re-ran live
  scheduling checks instead of relying on old node history. Current active
  training/downstream jobs are not using a standing bad-node list:
  `105236`, `105385`, `105238`, `105553`, and controller-video jobs have
  `ExcNodeList=(null)`; `105535`, `105743`, and pending `105867` use only
  job-local `server13` exclusion from current pre-dataset CUDA visibility
  failures in `105502` and `105707`; pending `105868` has no exclusions and
  is intentionally left as a live canary. One-H200 `sbatch --test-only`
  probes for `03:00:00`, `06:00:00`, `12:00:00`, and `1-00:00:00` all
  forecast `2026-06-05T00:46:04` on `server18`, while `gpu` partition
  forecast `2026-06-09T22:33:28` and `debug` was unavailable. A real
  no-exclusion `12:00:00` tmux request `105936` was submitted to test whether
  the probe produced an actual earlier path; `scontrol` gave a worse
  `StartTime=2026-06-05T12:00:00` on `server62`, so it was canceled before
  allocation. `sacct` confirms `105936` had `Elapsed=00:00:00`, no
  `AllocTRES`, and no assigned node. Keep `105867` and `105868` because both
  now have real `StartTime=2026-06-04T23:00:00`. This is scheduling hygiene
  and perception-gate risk reduction only, not method evidence.
- `2026-06-04 21:09+08:00` no-exclusion server13 canary result and queue
  cleanup:
  `105868` started on `server13` with `ExcNodeList=(null)` and failed after
  `00:00:17` before RGB-D dataset load. The allocation runner recorded
  `failure_class=scheduling_node_gpu_visibility`, `nvidia_smi_returncode=255`,
  `nvidia_smi_stdout="Unable to determine the device handle for gpu
  0000:AB:00.0: Unknown Error"`, `torch_cuda_available=false`, and
  `torch_cuda_device_count=0`. This validates the user's concern by testing a
  no-exclusion request live instead of assuming old node state; the result
  supports only job-local `server13` exclusion for current follow-up requests.
  Real replacement requests `105949` (`1-00:00:00`) and `105950`
  (`06:00:00`) were submitted with job-local `server13` exclusion but canceled
  before allocation when they had no stable earlier forecast; `sacct` shows
  both with `Elapsed=00:00:00`, no `AllocTRES`, and no assigned node.
  Existing `105867` then started on `server18` at
  `2026-06-04T21:07:52`, the fixed `srun` CUDA canary passed, and exact1000
  RGB-D slot training started for seed `2200`. Current allocation state is
  useful work only: `105236` plus one-H200 backups `105385`, `105535`,
  `105743`, and `105867` are running; no extra one-H200 allocation is left
  queued.
- `2026-06-04 21:12+08:00` running-allocation and handoff readiness audit:
  `squeue`/`sacct` show no extra one-H200 allocation is queued; useful GPU
  work is `105236` on 4 H200 plus one-H200 backups `105385`, `105535`,
  `105743`, and `105867`. The main slot job has no stderr output and is still
  training, with visible batch progress through about `1400`; backups are
  alive but incomplete. No checkpoint, metrics, or strict slot inspection
  artifact exists, so downstream jobs correctly remain dependency-pending.
  The active chain contract audit is still valid (`208/208` pass), and a
  dry-run of `assemble_rgbd_slot_backup_ensemble.py` over `105385`, `105535`,
  `105743`, and `105867` returned status `65` because none is complete or
  above the `10800` second floor. Lightweight handoff checks passed:
  `bash -n` for the downstream Slurm wrappers and `py_compile` for the
  downstream Python entrypoints. This is readiness evidence only; it does not
  count as RGB-D method evidence.
- `2026-06-04 21:15+08:00` live RGB-D slot triage:
  ran `triage_rgbd_slot_training_job.py` on the running main slot job
  `105236`. Output:
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/diagnostics/job105236_live_triage_20260604_2114/triage.{json,md}`.
  The diagnostic boundary is explicit: it does not approve slot quality or
  alter gates. It reports `JobState=RUNNING`, `ExcNodeList=(null)`,
  stderr size `0`, member dirs `4`, complete members `0`, and failure class
  only `incomplete_ensemble_members`. The validation split audit confirms
  all four members use trajectory-stratified validation with all six canonical
  scenarios represented (`42` validation trajectories per scenario). This is
  failure-localization/readiness evidence only; strict slot inspection remains
  the next authoritative gate.
- `2026-06-04 21:20+08:00` user scheduling clarification and rolling
  allocation expansion:
  user pointed to the cluster guidance that small jobs should usually wait
  about `10-30` minutes and asked whether current delays came from excessive
  bad-node exclusions. Re-audited live jobs: `105236`, `105385`, `105238`,
  `105553`, and all dependency-pending controller-video jobs have
  `ExcNodeList=(null)`. Current `server13` exclusions exist only on
  one-H200 backup/fallback jobs after three live pre-dataset CUDA visibility
  failures on `server13` (`105502`, `105707`, and no-exclusion canary
  `105868`), so this remains job-local scheduling evidence rather than a
  standing bad-node list. Probes for one-H200 `03:00:00`, `06:00:00`,
  `12:00:00`, and `1-00:00:00` initially forecast the same start for each
  duration; therefore submitted two tmux-backed `1-00:00:00` requests to
  reduce future requeue churn while still staying small: `106000`
  (`h200_1gpu_pool9`, no exclusions, seed `2300`) and `106001`
  (`h200_1gpu_pool10`, only job-local `ExcNodeList=server13`, seed `2400`).
  Follow-up `scontrol` shows both pending at `StartTime=2026-06-05T04:00:00`:
  `106000` has `ExcNodeList=(null)` and `SchedNodeList=server07`; `106001`
  has only `ExcNodeList=server13` and `SchedNodeList=server21`. One-H200
  CPU-count probes (`4`, `6`, `8` CPUs) and four-H200 probes did not produce
  an earlier useful forecast, so no extra weaker request or 4-H200 duplicate
  was submitted. Keep this as a bounded rolling batch, not a broad fanout.
- `2026-06-04 21:27+08:00` test-only replacement discipline:
  a new one-H200 `sbatch --test-only` probe showed a possible earlier start,
  so submitted exactly one bounded replacement `106053` in tmux
  `h200_1gpu_pool11` with exact1000 seed `2500` and only job-local
  `ExcNodeList=server13`. Real `scontrol` then showed `106053` had
  `StartTime=Unknown`, while the older pending requests `106000` and `106001`
  had moved earlier to `StartTime=2026-06-05T01:00:00`. Following the rule
  that test-only probes are not authoritative replacement evidence, canceled
  `106053` before allocation and verified its tmux session exited. Current
  bounded pending allocation requests are `106000` with no exclusions and
  `106001` with only job-local `server13` exclusion; useful running GPU work
  remains `105236`, `105385`, `105535`, `105743`, and `105867`.
- `2026-06-04 21:34+08:00` no-exclusion allocation follow-through:
  `106000` started early on `server43` with `ExcNodeList=(null)`. The
  allocation runner's fixed `srun` CUDA canary passed, and exact1000 RGB-D
  slot backup seed `2300` started with `cuda_available=true`,
  `require_cuda=true`, and `num_paths=1000`. This is current live evidence
  that no standing node exclusion list is blocking useful work. `106001`
  remains the only extra one-H200 pending request, with only job-local
  `ExcNodeList=server13`; keep the pending batch bounded unless live failures
  or earlier real forecasts justify another replacement.
- `2026-06-04 21:36+08:00` final pending allocation follow-through:
  `106001` also started early on `server07` with only job-local
  `ExcNodeList=server13`. Its fixed `srun` CUDA canary passed and exact1000
  seed `2400` slot backup began with `cuda_available=true`,
  `require_cuda=true`, and `num_paths=1000`. There are now no extra one-H200
  requests pending. Do not submit another allocation unless live failures,
  completion/failure of an existing path, or an earlier real forecast justifies
  a bounded replacement. Running backups are fallback training only; they
  cannot bypass the strict RGB-D slot inspection/export/world-model/controller
  gates.
- `2026-06-02 10:36+08:00` audit after the user reinforced the floor again:
  active object-WM training `94539` and larger-label `C_pi` training `94542`
  are already running on one node with 4 H200 GPUs each and 3.5-hour time
  limits. Their wrappers hard-refuse `MIN_TRAIN_SECONDS < 10800`, sparse
  multi-node training, or fewer than four tasks/GPUs. The RGB-D slot extractor
  training job `94860` is still gated behind inspected RGB-D data and has the
  same 4-H200/3-hour floor. No duplicate training was submitted because the
  primary jobs already have cards; fallback object-WM jobs remain dependency
  gated and include recovered-node-friendly non-rendering retries. `sinfo`
  shows `server61` is currently `mixed` rather than drained, while
  `server16` and `server29` are still drained.
- `2026-06-02 11:12+08:00` audit: primary object-WM training `94539`
  completed on `server21` after `03:01:17`, and strict inspection `94863`
  passed on `server52` with `compliant_training_evidence=True`. Failover
  training branches `94679`, `94746`, and `94809` became dependency-dead and
  were canceled with their dependent inspection/evaluation/controller branches
  after the primary passed. This avoids duplicate 4-H200 training and
  dead-pending evidence branches; it does not change evaluation.
- `2026-06-02 11:16+08:00` controller hygiene: primary full-shard no-video
  controller job `95055` completed, inspection `95056` passed as a failed-run
  inspection, and gate `95057` failed with exit `65`. The dead gated-video
  branch `95058`/`95059` and blocked summary/cleanup branches `95075`-`95077`
  were canceled. No video was produced for this failed no-video gate.
- `2026-06-02 11:20+08:00` video hygiene: learned-WM move-stop video job
  `95047` failed with Vulkan `ErrorDeviceLost` on `server10` before metrics or
  video artifacts were written; inspection `95048` found no evidence. This is
  a render-node failure to rerun around, not a control/evaluation result.
  Submitted same-settings rerun `95191` with inspection `95192`; the rerun
  keeps the learned-WM move-stop video parameters and adds `server10` to the
  render exclusion list. Peg-drop/regrasp video job `95107` and inspection
  `95108` completed; the contact sheet and extracted final frame were opened
  directly and match the final-success metrics. Record this as CV regrasp
  visual evidence only.
- `2026-06-02 11:34+08:00` failure-video hygiene: submitted debug video
  rerun `95209` with inspection `95210` for the failed full-shard seed `7400`
  run. This is not a gated success-video branch; it exists to visually inspect
  the insertion-axis jam observed in no-video run `95055`. It preserves the
  `95055` controller settings and only enables video plus excludes `server10`
  after the Vulkan failure.
- `2026-06-02 11:36+08:00` controller-probe hygiene: submitted no-video
  insertion-manifold probe `95212`, inspection `95213`, unchanged success gate
  `95214`, gated video `95215`, and video inspection `95216`. The old node
  exclusion details in those submitted snapshots are scheduling history only;
  future no-video and video jobs must not inherit a standing node list.
- 2026-06-02 10:16 template audit: superseded. Active `scripts/slurm/` wrapper
  source must not carry standing node exclusions. Rendering templates keep the
  dense allocation guard, and any node-specific decision must be made from
  live Slurm state or a targeted canary and recorded in the job manifest.

Post-inspection calibration jobs:

- `94659`: CPU-partition threshold calibration for pilot `C_pi` job `94471`,
  dependency `afterok:94618`, failed after writing JSON because Markdown
  report generation had a `threshold` format-key collision.
- `94716`: fixed CPU-partition threshold calibration rerun for pilot `C_pi`
  job `94471`, completed and wrote both JSON and Markdown. Held-out
  conservative thresholds have zero recall, so the pilot is not a usable
  handoff gate.
- `94660`: CPU-partition threshold calibration for larger-label `C_pi` job
  `94542`, dependency `afterok:94658`.
- `94685`: canceled CPU-partition threshold calibration for larger-label
  `C_pi` backfill job `94683`; the whole backfill chain was canceled after
  primary training `94542` started.

Controller post-run inspection jobs:

- `94696`: inspect learned-world-model video smoke `94564`, dependency
  `afterany:94564`, completed.
- `94700`: inspect learned-world-model video smoke `94576`, dependency
  `afterany:94576`, completed.
- `94717`: inspect learned-world-model video smoke `94577`, submitted after
  the completed run was noticed, completed.
- `94699`: inspect same-seed learned-world-model no-video debounce validation
  `94698`, dependency `afterany:94698`, completed.
- `94740`: fixed CPU success gate after debounce inspection `94699`; it failed
  with exit `65` by design because observed final dynamic success was `0`.
- `94656`: inspect v4 regrasp smoke `94645`, dependency `afterany:94645`,
  completed.
- `94751`-`94760`: initial 3-hour TCP-continuation controller validation
  chain, canceled before running because these are controller smokes rather
  than model training and should use shorter backfill-friendly walltime.
- `94765`: inspect TCP-continuation CV no-video validation `94763`, dependency
  `afterany:94763`.
- `94766`: inspect TCP-continuation learned-WM no-video validation `94764`,
  dependency `afterany:94764`.
- `94767` and `94768`: CPU success gates after `94765`/`94766`. They exit
  success only if inspection reports final success after a prior dynamic event.
- `94769` and `94770`: gated video jobs after `94767`/`94768`, canceled after
  the matching success gates failed by design.
- `94771` and `94772`: CPU inspections after gated video jobs, canceled with
  the dead gated video dependencies.
- `94773`: TCP-continuation peg-drop/regrasp no-video validation, 30-minute
  non-training walltime, seed `7300`.
- `94774`: CPU inspection after `94773`.
- `94775`: success gate after `94774`, failed by design because the no-video
  run had no final dynamic success.
- `94776`: gated video after `94775`, canceled after the gate failed.
- `94777`: CPU inspection after `94776`, canceled with the dead gated video
  dependency.
- `94661`-`94664`: canceled before running because the jobs were submitted
  before the full-shard controller chain carried explicit
  `bridge_servo_reference=tcp_continuation` metadata and success gating.
- `94779`-`94783`, `94791`-`94795`, and `94796`-`94800`: canceled before
  running because they were full-shard position-only controller chains. They
  were replaced after the orientation-gap analysis showed the next full-shard
  controller validation must use task-frame TCP pose servo, not position-only
  TCP continuation.
- `94814`: orientation-aware TCP-pose CV no-video validation, 30-minute
  non-training walltime, seed `7200`.
- `94815`: CPU inspection after `94814`.
- `94816`: success gate after `94815`.
- `94817`: gated video after `94816`.
- `94818`: CPU inspection after `94817`.
- `94819`: orientation-aware TCP-pose learned-WM no-video validation using
  compliant model job `94442`, seed `7300`.
- `94820`: CPU inspection after `94819`.
- `94821`: success gate after `94820`.
- `94822`: gated video after `94821`.
- `94823`: CPU inspection after `94822`.
- `94824`: orientation-aware TCP-pose peg-drop/regrasp no-video validation,
  seed `7300`.
- `94825`: CPU inspection after `94824`.
- `94826`: success gate after `94825`.
- `94827`: gated video after `94826`.
- `94828`: CPU inspection after `94827`.
- `94882`: CPU summary after no-video inspections `94815`/`94820`/`94825`;
  it reads `controller_smoke_v4_tcp_pose_continuation_novideo_inspections.txt`
  and writes a JSON/Markdown comparison table under
  `experiments/world_model_task_rebinding/rebinding_controller/summaries/`.
- `94881`: canceled before running because it required all gated-video
  inspections even though missing video inspections are expected when no-video
  success gates fail.
- `94889`: CPU summary after video inspections `94818`/`94823`/`94828`;
  it reads `controller_smoke_v4_tcp_pose_continuation_video_inspections.txt`
  with `REQUIRE_ALL=false` and writes the corresponding video-artifact
  comparison table. This summary still does not replace direct
  video/contact-sheet inspection.
- `94830`-`94834`: canceled before running because their upstream object-WM
  inspection `94619` lacked the strict compliance gate.
- `94865`: full-shard orientation-aware learned-WM no-video validation for

## 2026-06-03 Resource Probe And Submission Hygiene

- [x] Fresh association and partition probe: `sacctmgr` reports user
      `yanhongru` has account `mayi`; `sinfo` shows many mixed nodes, but
      `sbatch --test-only` is the authoritative availability check.
      `gaosh`, `engram`, and `test` still reject the account/partition
      combination for this user. `gpux`/`mgpu` remain unusable for current
      submissions. Usable `cpu`/`gpu`/`debug` probes are schedulable but not
      immediate.
- [x] Current non-executing probes:
      `cpu` 1GPU 10/20min forecast `2026-06-03T23:53:53` on `server21`;
      `cpu` 2GPU 20min forecast the same time on `server58`; `gpu` 1GPU
      20min forecast `2026-06-04T21:22:53`; `debug` 1GPU 20min forecast
      `2026-06-04T07:03:05`; `cpu` 4GPU 3h forecast
      `2026-06-03T23:53:53` on `server58`. With current RGB-D render-failed
      nodes `server10,server21,server28,server55,server58` excluded, the
      1GPU 20min repair shape forecasts `2026-06-03T23:54:58` on `server42`,
      2GPU is later, and 4GPU is later still.
- [x] Submitted only two aligned jobs after the probe: focused state smoke
      `99714`, which started on `server42`, and exact failed-only RGB-D
      repair `99715` with job-local exclusions from current render evidence.
      No broad duplicate render fanout was submitted.
- [x] RGB-D repair and gate follow-through: `99715` completed `3/3` repair
      units on `server42`; `99716` exact aggregate completed with `96` files;
      `99717` structural inspection completed with `num_warnings=0`; `99718`
      visual inspection completed with `valid_visual_artifacts=true` and the
      review sheet was opened directly. RGB-D slot training `99719` then
      started on `server20` with one node / four H200 GPUs and `03:30:00`
      walltime. This is compliant training liveness, not model evidence until
      post-training inspection passes.
  object-WM job `94539`, dependency `afterok:94863`.
- `94866`: CPU inspection after `94865`.
- `94867`: success gate after `94866`.
- `94868`: gated video after `94867`.
- `94869`: CPU inspection after `94868`.
- `94835`: full-shard orientation-aware learned-WM no-video validation for
  object-WM backfill `94679`, dependency `afterok:94680`.
- `94836`: CPU inspection after `94835`.
- `94837`: success gate after `94836`.
- `94838`: gated video after `94837`.
- `94839`: CPU inspection after `94838`.
- `94840`: full-shard orientation-aware learned-WM no-video validation for
  object-WM fast-fit duplicate `94746`, dependency `afterok:94747`.
- `94841`: CPU inspection after `94840`.
- `94842`: success gate after `94841`.
- `94843`: gated video after `94842`.
- `94844`: CPU inspection after `94843`.
- `94845`: full-shard orientation-aware learned-WM no-video validation for
  old alternate node-selection retry `94809`, dependency `afterok:94810`.
- `94846`: CPU inspection after `94845`.
- `94847`: success gate after `94846`.
- `94848`: gated video after `94847`.
- `94849`: CPU inspection after `94848`.
- `94885`: full-shard no-video controller summary after inspections
  `94866`/`94836`/`94841`/`94846`, reading
  `fullshard_tcp_pose_continuation_novideo_inspections.txt`.
- `94886`: canceled before running for the same gated-video partial-artifact
  reason as `94881`.
- `94890`: full-shard gated-video controller summary after inspections
  `94869`/`94839`/`94844`/`94849`, reading
  `fullshard_tcp_pose_continuation_video_inspections.txt` with
  `REQUIRE_ALL=false`.

RGB-D post-run inspection jobs:

- `94665`: canceled before running because its submitted Slurm script snapshot
  did not include the strict no-warning RGB-D gate.
- `94677`: canceled before running because its submitted Slurm script snapshot
  did not include the strict no-warning RGB-D gate.
- `94858`: inspect dense full-shard RGB-D export `94541`, dependency
  `afterany:94541`, with `REQUIRE_NO_WARNINGS=true`.
- `94859`: inspect dense 2-node/16-GPU RGB-D backfill export `94676`,
  dependency `afterany:94676`, with `REQUIRE_NO_WARNINGS=true`.

RGB-D slot extractor jobs:

- `94732`: canceled before running because its manually supplied output
  directory would have used `jobmanual`; recorded as a path-hygiene correction.
- `94733`: canceled before running after preflight found it lacked an
  `INPUT_RGBD_DIR`/`RGBD_H5S_FILE` export.
- `94734`: canceled before running after preflight found it lacked
  `ENSEMBLE_DIR`.
- `94856`: canceled before running because it depended on old RGB-D inspection
  job `94677`.
- `94857`: canceled before running because it inspected canceled training job
  `94856`.
- `94860`: corrected 4-H200 RGB-D slot extractor training job, dependency
  `afterok:94859`, exporting
  `INPUT_RGBD_DIR=.../rgbd_dynamic_dense/from_rollout_dir/job94676` and
  `MIN_TRAIN_SECONDS=10800`.
- `94861`: CPU inspection for `94860`, dependency `afterany:94860`, exporting
  `ENSEMBLE_DIR=.../rgbd_slot_extractor/ensemble_4gpu/job94860` and requiring
  `compliant_training_evidence=true`.

Historical cluster observations:

The entries below record why specific old jobs were submitted, canceled, or
rerun. They are not an active bad-node list, and future jobs must not inherit
their node names without fresh live evidence.

- Old RGB-D jobs `94270`, `94272`, and `94273` failed with `vk::DeviceLost`;
  follow-up jobs used a job-local render exclusion and retries succeeded. This
  is historical failure evidence only, not current render policy.
- 2026-06-02 05:32 node scan: non-rendering training jobs already exclude only
  currently down/drain nodes `server16`, `server29`, and `server61`, so old
  nodes with earlier rendering failures are eligible for training. `server04`, `server09`, and
  `server53` were recovered enough for Slurm but already had all 8 GPUs
  allocated; `server03` had only 3 GPUs free; no current node had 4 free H200
  GPUs. Fast-fit job `94746` was submitted to wait for a smaller compliant
  3:10 scheduling window without lowering the 4-H200/10800-second floor.
- 2026-06-02 05:42 controller queue cleanup: stale pending controller jobs
  `94578`, `94582`, `94583`, and `94590` were canceled after the
  TCP-continuation bridge target replaced the old peg-head bridge default.
  This cleanup did not touch training, label, RGB-D, or full-shard dependent
  jobs.
- 2026-06-02 05:45 controller walltime cleanup: initial TCP-continuation chain
  `94751`-`94760` was canceled before running and replaced by shorter
  controller-smoke chain `94763`-`94772`. This does not lower model-training
  standards; it only avoids reserving 3 hours for one-episode controller
  validations that historically finish in about one minute.
- 2026-06-02 05:55 queue check: all usable H200 GPUs are currently allocated.
  The only nodes with at least four unallocated GPUs are `server16`,
  `server29`, and `server61`, all down/drain, so normal Slurm jobs cannot use
  them. Non-rendering training jobs already exclude only those down/drain nodes,
  so recovered nodes with earlier rendering failures remain eligible for training.
- 2026-06-02 05:59 partition probe: `sbatch --test-only` on `test`, `gaosh`,
  and `engram` failed because the current account cannot use those partition
  combinations; `debug` rejected the 4-GPU request with `MaxGRESPerAccount`.
  No real probe jobs were submitted, and the earliest compliant training
  candidate remains `94539`.
- 2026-06-02 06:01 controller-chain coverage: added world-model directory
  files and strict gated controller chains for object-WM scheduling duplicates
  `94679` and `94746`. These chains depend on successful compliant inspections
  and do not create controller evidence unless the corresponding training job
  passes inspection.
- 2026-06-02 06:13 user reaffirmed the training floor: model training must not
  be smaller than 4 H200 GPUs or shorter than 3 hours, and a short bad result
  must not be used to reject the direction. Fresh node scan showed every
  schedulable H200 GPU allocated; nodes with earlier rendering failures such as `server03`,
  `server04`, `server09`, and `server53` are schedulable but have
  `AllocTRES gres/gpu=8`. Submitted alternate node-selection route retry `94809` by
  excluding the complementary good node set; it still requests 4 H200 GPUs on
  one node, uses `MIN_TRAIN_SECONDS=10800`, and has compliance inspection
  `94810` plus prediction evaluation `94811`.
- 2026-06-02 06:25 orientation-aware controller validation jobs were queued
  after the TCP-position-only failure was traced to rotation error. Slurm
  forecast put the three 1-GPU no-video jobs `94814`/`94819`/`94824` on
  `server31` at `2026-06-02T07:23:25`; the earliest 4-GPU object training job
  `94539` remained forecast on `server42` at `2026-06-03T00:01:27`, so the
  controller smokes were not forecast to steal the training node. If this
  changes, training jobs take priority over controller smokes.
- 2026-06-02 06:30 queue hygiene correction: canceled pending full-shard
  position-only controller chains `94779`-`94783`, `94791`-`94795`, and
  `94796`-`94800` before they could run. Submitted orientation-aware
  replacements `94830`-`94849` behind successful compliant object-WM
  inspections, including coverage for old alternate node-selection retry `94809`. The
  `94830`-`94834` branch was later canceled before running and replaced by
  `94865`-`94869` after strict inspection `94863`.
- 2026-06-02 06:31 forecast update after queue correction: orientation-aware
  one-GPU no-video jobs `94814`/`94819`/`94824` moved to
  `2026-06-02T10:00:12` on `server28`; earliest 4-GPU object training
  `94539` remains forecast for `2026-06-03T00:01:27` on `server42`. Controller
  smokes are still not forecast to steal the training node.
- 2026-06-02 06:35 inspector update: added bridge-orientation summaries to
  `inspect_rebinding_controller_run.py`; local compatibility check on old run
  `94763` passed and produced empty rotation stats instead of failing.
- 2026-06-02 06:38 training queue audit: no user jobs are running. Slurm still
  forecasts the earliest compliant object-WM run `94539` for
  `2026-06-03T00:01:27` on `server42`, which reports `Gres=gpu:NVIDIAH200:8`.
  The object-WM backfills `94679` and `94746` also request 4 H200 GPUs and
  `MIN_TRAIN_SECONDS=10800`; old alternate node-selection retry `94809` is still pending on
  the restricted alternate node-selection route and is forecast on `server14`, also an
  `NVIDIAH200:8` node. C_pi and RGB-D slot training wrappers were rechecked:
  both refuse `MIN_TRAIN_SECONDS < 10800` and refuse fewer than 4 GPUs/tasks.
- 2026-06-02 06:38 RGB-D allocation audit: `94541` requests one full 8-GPU
  node and `94676` requests two nodes with 8 GPUs per node. This satisfies the
  dense-allocation rule for RGB-D generation; no multi-node one-GPU-per-node
  render job is pending.
- 2026-06-02 06:41 node scan: every schedulable `NVIDIAH200:8` node has
  `gres/gpu=8` allocated. The only nodes with apparent free GPUs are
  `server16`, `server29`, and `server61`, all down/drain. A same-spec
  `sbatch --test-only` 4-H200, 3:10 object-WM training submission would start
  at `2026-06-09T21:23:12`, later than the existing compliant jobs. No extra
  duplicate was submitted; the active path remains `94539`, `94679`, `94746`,
  and alternate node-selection retry `94809`.
- 2026-06-02 06:44 queue/config audit: no active result has completed.
  Same-seed orientation-aware controller no-video jobs
  `94814`/`94819`/`94824` remain forecast for `2026-06-02T10:00:12` and their
  `SubmitLine` exports include both
  `BRIDGE_SERVO_REFERENCE=tcp_continuation` and
  `BRIDGE_ORIENTATION_REFERENCE=tcp_continuation`. Their gated video jobs
  `94817`/`94822`/`94827` depend on the success gates and export
  `SAVE_VIDEO=true`. Full-shard controller chains `94830`-`94849` were also
  checked with `sacct SubmitLine` and are orientation-aware, but the
  `94830`-`94834` branch was later canceled before running when its upstream
  inspection was found non-strict. Node-risk retry `94809` forecast moved
  earlier to `2026-06-04T13:40:25`; dense RGB-D
  backfill `94676` moved to `2026-06-04T05:00:00`.
- 2026-06-02 06:47 RGB-D slot preflight: canceled pending `94733`/`94734`
  before running because the training job would not receive RGB-D inputs and
  the inspection job would not know the ensemble directory. Requeued
  intermediate chain `94856 -> 94857`, later canceled before running when
  the upstream RGB-D inspection snapshot issue was found.
- 2026-06-02 06:53 RGB-D inspection gate: `inspect_rgbd_dataset.sbatch` now
  defaults to `REQUIRE_NO_WARNINGS=true`. If dataset inspection reports any
  warnings, the wrapper exits nonzero. Because pending `94665`/`94677` had
  already snapshotted the old script, they were canceled before running and
  replaced by strict inspections `94858`/`94859`. Corrected downstream chain
  is now `94676 -> 94859 -> 94860 -> 94861`, preserving the
  4-H200/10800-second training floor. `bash -n` passed; a smoke RGB-D H5 with
  `base_camera` and `hand_camera` produced `num_warnings=0` under the same
  required-camera check.
- 2026-06-02 07:01 object-WM gate audit: pending inspection `94619` for
  `94539` used an old submitted Slurm script snapshot without the strict
  `REQUIRE_COMPLIANT` gate. Canceled `94619`, downstream eval `94673`, and
  dependent controller jobs `94830`-`94834` before running. Requeued strict
  post-training inspection/eval `94863 -> 94864` and replacement
  orientation-aware controller chain `94865 -> 94866 -> 94867 -> 94868 ->
  94869`. Training jobs themselves were not modified and still keep the
  4-H200/10800-second floor.
- 2026-06-02 07:08 queue/resource audit: no user jobs are running. `sacct`
  confirms all stale non-strict chains canceled before allocation and current
  strict replacements pending: RGB-D `94858`-`94861`, object-WM inspection/eval
  `94863`/`94864`, and controller chain `94865`-`94869`. `squeue --start`
  forecasts controller smokes `94814`/`94819`/`94824` at
  `2026-06-02T08:16:34`, first 4-H200 object-WM training `94539` at
  `2026-06-03T00:01:27`, dense RGB-D export `94541` at
  `2026-06-03T06:01:30`, dense 2-node RGB-D backfill `94676` at
  `2026-06-04T05:00:00`, and node-specific/limited-route retry `94809` at
  `2026-06-04T13:40:25`. H200 node scan still has `server16`, `server29`, and
  `server61` in drain/down-like states; no recovered node-specific node is currently idle
  and usable.
- 2026-06-02 07:10 submitted-script audit: no new result has completed.
  `sacct SubmitLine` and frozen batch scripts confirm the pending
  orientation-aware controller jobs export
  `BRIDGE_SERVO_REFERENCE=tcp_continuation`,
  `BRIDGE_ORIENTATION_REFERENCE=tcp_continuation`,
  `MAX_BRIDGE_DELTA_ROT_RAD=0.08`, and the intended `SAVE_VIDEO` setting.
  Gated video jobs `94817`/`94822`/`94827` and `94868` depend on success gates,
  not raw no-video completion. RGB-D inspections `94858`/`94859` include
  `REQUIRE_NO_WARNINGS=true`; RGB-D slot training `94860` keeps one-node
  4-H200 allocation and `MIN_TRAIN_SECONDS=10800`. Object-WM strict inspection
  `94863` and eval `94864` carry `REQUIRE_COMPLIANT=true`. The old node-specific
  retry `94809` intentionally writes to its historical output directory; its training,
  inspection, evaluation, and controller dir file all point to that same output
  directory, so the non-job-id path is consistent rather than a mismatch.
- 2026-06-02 07:13 scheduling probe: all key jobs are still pending; no new
  controller, training, RGB-D, or video result exists. Re-read the active
  `PLAN/world_model_task_rebinding` overview, object-WM, controller, RGB-D, and
  experiment-matrix files before acting. `sbatch --test-only` probes for new
  one-node 4-H200 object-WM duplicates with `MIN_TRAIN_SECONDS=10800` and
  walltimes `03:05:00`/`03:10:00` forecast `2026-06-09T21:24:12` for both the
  normal down/drain-only route and the alternate node-selection route. This is much later
  than existing job `94539` at `2026-06-03T00:01:27`, so no duplicate was
  submitted. Test-only pseudo job IDs `94874`-`94877` do not appear in
  `squeue` or `sacct`.
- 2026-06-02 07:16 controller evidence-chain audit: all key jobs remain
  pending. Re-read the active TODO, controller TODO, and experiment matrix.
  `evaluate_rebinding_controller.py` writes both MP4 and
  `*_contact_sheet.png` whenever `SAVE_VIDEO=true`; controller inspection
  reports video/contact-sheet paths and keeps final dynamic success separate
  from video evidence. `gate_rebinding_controller_success.py` gates automatic
  video jobs only on inspected no-video final success after a prior dynamic
  event and exits `65` otherwise, so failed gates do not become hidden method
  failures. `bash -n` passed for the controller evaluation, inspection, and
  gate wrappers. Updated forecasts: `94814`/`94819`/`94824` remain at
  `2026-06-02T08:16:34`; `94539` remains at `2026-06-03T00:01:27`; dense RGB-D
  backfill `94676` moved to `2026-06-03T22:54:26`; live-node retry `94809`
  moved to `2026-06-05T22:48:07`.
- 2026-06-02 07:18 added controller-inspection summary utility:
  `scripts/world_model/summarize_rebinding_controller_inspections.py` plus
  Slurm wrapper `scripts/slurm/summarize_rebinding_controller_inspections.sbatch`.
  The tool only summarizes existing inspection JSONs; it does not alter success
  gates, metrics, or video requirements. Local lightweight validation on old
  failed no-video inspections `94763`/`94764`/`94773` produced a table with
  zero final dynamic successes, and a missing-inspection test exited `65` while
  still writing a report. Submitted summary jobs `94882` after no-video
  inspections `94815`/`94820`/`94825` and initially `94881` after video
  inspections `94818`/`94823`/`94828`.
- 2026-06-02 07:23 extended the same summary mechanism to full-shard
  orientation-aware controller chains. Added input lists
  `fullshard_tcp_pose_continuation_novideo_inspections.txt` and
  `fullshard_tcp_pose_continuation_video_inspections.txt`, then submitted CPU
  summaries `94885` after no-video inspections `94866`/`94836`/`94841`/`94846`
  and initially `94886` after gated-video inspections
  `94869`/`94839`/`94844`/`94849`.
  Frozen batch-script audit confirms both use the fixed `PY_ARGS` bool handling
  for `--require-all`; wrapper syntax and Python compile checks passed.
- 2026-06-02 07:25 gated-video summary dependency audit: canceled pending
  video-summary jobs `94881` and `94886` before allocation. They had
  `REQUIRE_ALL=true`, but missing video inspections are expected when a no-video
  success gate fails and therefore should not make the artifact summary fail.
  Requeued replacements `94889` and `94890` with identical dependencies and
  outputs but `REQUIRE_ALL=false`. No-video summaries `94882` and `94885`
  remain strict because those inspections should exist for every no-video run.
- 2026-06-02 07:29 training-floor audit: the active standard remains one-node
  4-H200 training with `MIN_TRAIN_SECONDS>=10800`. Pending object-WM jobs
  `94539`, `94679`, `94746`, and live-node retry `94809` request
  `gres/gpu=4`; dependent `C_pi` training `94542`/`94683` and RGB-D slot
  training `94860` also request `gres/gpu=4` and keep the same 10800-second
  guard. Controller and render jobs may be shorter because they are not model
  training evidence.
- 2026-06-02 07:31 resource probe: `AllocTRES` shows all schedulable H200 nodes
  already have `gres/gpu=8` allocated except `server27` with one free GPU and
  `server31` with two free GPUs, both below the required four. `server16` and
  `server29` are `DOWN+DRAIN+NOT_RESPONDING`; `server61` is `IDLE+DRAIN`.
  Slurm rejects drain-node-only attempts for those nodes, so they cannot be
  used as a recovery queue until the administrator clears drain/down state.
  `gpux` and `mgpu` are drained, while `gaosh`/`engram` are invalid for this
  account. Additional compliant normal-route and alternate node-selection route
  `sbatch --test-only` probes would start at `2026-06-09T21:01:12`, so no
  extra duplicate was submitted.
- 2026-06-02 07:34 queue continuation audit: `squeue` shows no current
  method job running. `94814`/`94819`/`94824` remain pending on priority with
  forecast `2026-06-02T19:28:40`; `94539` remains the earliest formal
  object-WM training forecast at `2026-06-03T00:01:27`. Artifact search found
  only the older `94763`/`94764`/`94773` failed TCP-continuation results within
  the recent window. Same-settings controller probes with 10/15-minute
  non-training walltime would start at `2026-06-09T19:27:41`, so shortening
  walltime does not improve the queue and no duplicate was submitted.
- 2026-06-02 07:36 follow-up queue check: `squeue --start` moved
  `94814`/`94819`/`94824` earlier to `2026-06-02T10:00:12`. It also moved
  `94676` to `2026-06-03T22:07:26`, `94746` to `2026-06-04T04:00:00`, and
  `94809` to `2026-06-04T16:00:00`. No current-method job is running, so this
  is scheduling metadata only.
- 2026-06-02 07:37 external-state audit: artifact search found no new
  current-method metrics, inspections, MP4s, or contact sheets after the
  already-recorded `94763`/`94764`/`94773` failures. `scontrol show job`
  confirms key pending jobs have `Reason=Priority` and no unexpected
  dependency. All schedulable H200 nodes report `AllocTRES gres/gpu=8`; the
  only unallocated H200 nodes are still `server16`/`server29` down/drain and
  `server61` drain.
- 2026-06-02 07:39 status check: key current-method jobs are unchanged:
  `94814`/`94819`/`94824` pending at `2026-06-02T10:00:12`; formal
  object-WM training `94539` pending at `2026-06-03T00:01:27`; dense RGB-D
  backfill `94676` pending at `2026-06-03T22:07:26`. No current-method job is
  running and no new artifact requires inspection.
- 2026-06-02 07:40 artifact-search hygiene: a recursive `find` intermittently
  warned that `controller_smoke/...job94518/inspection.json` and
  `inspection.md` were missing, but direct `ls` and `find -ls` immediately
  confirmed both files exist alongside metrics and video/contact-sheet
  artifacts. Treat this as a transient shared-filesystem/stat warning, not as
  evidence loss.
- 2026-06-02 07:41 controller evidence-chain preflight: `sacct SubmitLine`
  confirms no-video inspection jobs `94815`/`94820`/`94825` export `RUN_DIR`
  paths that match the no-video summary input list exactly. Gated-video
  inspection jobs `94818`/`94823`/`94828` likewise match the video summary
  input list. Summary `94882` remains strict with `REQUIRE_ALL=true`; summary
  `94889` remains partial-artifact tolerant with `REQUIRE_ALL=false`.
- 2026-06-02 07:43 dead-gated-video dependency note: `sacct` on the prior
  TCP-continuation chains shows failed gates `94767`/`94768`/`94775` were
  followed by manually canceled video jobs and inspections
  `94769`/`94770`/`94771`/`94772` and `94776`/`94777`. If the current
  orientation-aware no-video gates fail, inspect the dependency states before
  summarizing; cancel only dead gated-video branches if Slurm leaves them
  pending, then run the tolerant video summary. Do not cancel no-video
  inspections or change success gates.
- 2026-06-02 07:45 added gated-video cleanup helper:
  `scripts/world_model/cleanup_gated_video_branches.py` and
  `scripts/slurm/cleanup_gated_video_branches.sbatch`. The helper reads
  `<label> <gate_job> <video_job> <inspection_job>` branches and only cancels
  video/video-inspection jobs when the no-video success gate is terminal and
  did not pass. It leaves branches untouched when gates are pending or passed.
  `py_compile`, `bash -n`, dry-run on old failed gates, and dry-run on current
  pending gates passed.
- 2026-06-02 07:45 submitted cleanup jobs: `94929` waits on
  `afterany:94816:94821:94826` for the same-seed orientation-aware controller
  family and uses
  `controller_smoke_v4_tcp_pose_continuation_video_branches.txt`; `94930`
  waits on `afterany:94867:94837:94842:94847` for the full-shard controller
  family and uses `fullshard_tcp_pose_continuation_video_branches.txt`. These
  jobs are evidence-chain hygiene only and do not change metrics, gates, or
  video review requirements.
- 2026-06-02 07:50 queue/resource audit: no current-method job is running.
  `94814`/`94819`/`94824` remain priority-pending with Slurm forecast
  `2026-06-02T09:03:44`; `scontrol show job` confirms each requests one GPU
  and has no unexpected dependency. `94929` is correctly blocked only on the
  three no-video success gates, and `94889` remains blocked only on the three
  gated-video inspections with tolerant summary settings. Artifact search
  found only older controller videos/contact sheets from already-recorded
  failed runs, not new orientation-aware results.
- 2026-06-02 07:57 static execution preflight: `py_compile` passed for the
  controller evaluator, controller inspector, and gated-video cleanup helper.
  The temporary `__pycache__` files from that check were removed. A direct
  helper-level Euler round-trip check passed, and
  `collect_dynamic_state_rollouts._load_runtime_modules()` eventually loaded
  successfully in the current virtualenv. Direct ad hoc `mani_skill` import
  briefly exposed import-chain noise through IPython/trimesh, but the actual
  project runtime entrypoint loaded, so no Slurm job was canceled or weakened.
  If a queued controller job nevertheless fails at import time, treat that as
  an execution issue to fix, not as controller evidence.
- 2026-06-02 07:58 queue/config audit: no current-method job is running and no
  new controller artifact exists. Same-seed orientation-aware jobs
  `94814`/`94819`/`94824` remain priority-pending with forecast
  `2026-06-02T09:12:45`. Their `sacct SubmitLine` entries preserve the
  orientation-aware TCP-continuation settings, no-video/video split, and
  success-gated video dependencies. The learned-WM directory list for job
  `94442` still points to four complete model member directories.
- 2026-06-02 08:01 resource audit: `94814`/`94819`/`94824` remain
  priority-pending with `StartTime=2026-06-02T09:12:45` and scheduled node
  `server31`. `scontrol show nodes` reports only `server27` and `server31`
  with one unallocated H200 each (`AllocTRES gres/gpu=7`); `server16` and
  `server29` remain `DOWN+DRAIN+NOT_RESPONDING`, and `server61` remains
  `IDLE+DRAIN`. There is no currently valid 4-H200 training slot, so no new
  compliant training duplicate was submitted and no standard was lowered.
- 2026-06-02 08:48 controller evidence-chain hygiene: canceled stale pending
  full-shard phase-search controller jobs `94999`-`95021` before execution
  because v3/v4 diagnosis superseded their bridge mode. Replacement
  task-frame-projected full-shard chains are `95055`-`95059`,
  `95060`-`95064`, `95065`-`95069`, and `95070`-`95074`, with summaries and
  gated-video cleanup `95075`/`95076`/`95077`. This did not touch training,
  object-WM inspections/evals, RGB-D, or `C_pi` jobs.
- 2026-06-02 08:48 same-seed v4 scheduling probe: no-video jobs `95039` and
  `95042` inherited the rendering exclusion list but already forecast
  `2026-06-02T10:00:12` on `server54`. Minimal-exclusion `sbatch --test-only`
  alternatives forecast `2026-06-09T17:06:54`, so no cancel/requeue was made.
- 2026-06-02 08:50 RGB-D dense-render scheduling probe: existing full-shard
  render `94541` remains earliest with forecast `2026-06-03T00:01:27`.
  Dense 8/16/32/64-GPU `sbatch --test-only` alternatives forecast
  `2026-06-09` or later, so no additional RGB-D render job was submitted.
- 2026-06-02 08:53 object-WM training failover hygiene: backup jobs were
  converted from ordinary priority-pending duplicates to failover-only
  dependencies. `94679` now has `Dependency=afternotok:94863`, `94746` has
  `Dependency=afternotok:94680`, and `94809` has
  `Dependency=afternotok:94747`. This preserves the node-specific recovery path and
  compliant backup routes without spending another 4 H200 GPUs unless an
  earlier strict inspection fails.
- 2026-06-02 08:55 training liveness audit: `94539` is still running on
  `server21` with one node and 4 H200 GPUs; `94679`, `94746`, and `94809`
  are dependency-pending failovers. Intermediate member manifests and
  `best_model.pt` files exist for `94539`, but runtime is still below
  `MIN_TRAIN_SECONDS=10800`, so no training conclusion is allowed yet.
- 2026-06-02 08:58 Slurm chain audit: `94539.0` is an active 4-task srun step
  under a one-node, 4-GPU allocation. The repeated
  `cuda_visible_devices=0` in member manifests is consistent with
  `srun --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1`, where each task
  sees its assigned GPU remapped to local device 0; it is not evidence that
  the job is using only one GPU. Same-seed v4 controller jobs still forecast
  `2026-06-02T10:00:12`; no new v4 metrics, inspection JSON, MP4, or contact
  sheet exists yet.
- 2026-06-02 09:00 static syntax check: `bash -n` passed for the controller
  and training Slurm wrappers, and `.venv/bin/python -m py_compile` passed for
  the controller evaluator/inspector/summary plus object-WM inspection/eval
  scripts. The generated `scripts/**/__pycache__` directories were removed.
- 2026-06-02 09:01 queue/artifact audit: `95039`/`95042` remain
  priority-pending for `2026-06-02T10:00:12`; `94539` remains running on
  `server21` with 4 H200 GPUs and about 50 minutes elapsed; RGB-D and larger
  `C_pi` jobs remain queued. Artifact search found no new
  task-frame-projected inspection JSON, metrics, MP4, or contact sheet, so
  there is no new result to interpret or visually inspect yet.
- 2026-06-02 09:03 same-seed v4 cleanup coverage: added
  `controller_smoke_v7_task_frame_projected_video_branches.txt` with
  `95041 -> 95045 -> 95046` and `95044 -> 95047 -> 95048`, then submitted
  CPU cleanup job `95086` with `Dependency=afterany:95041,95044`. This is
  dependency hygiene only; it does not change gates, metrics, or video-review
  requirements.
- 2026-06-02 09:05 peg-drop v4 scheduling: submitted no-video regrasp
  validation `95088`. The old node-exclusion wording is superseded; future
  gated video branches must use live Slurm state or targeted canaries rather
  than a wrapper-level node-specific render exclusion policy. Attached
  inspection/gate/cleanup/summary chain:
  `95089`/`95090`/`95093`/`95094`/`95095`.
- 2026-06-02 09:08 queue/liveness audit: `95088` remains priority-pending for
  `2026-06-02T09:26:54`, `95039`/`95042` remain priority-pending for
  `2026-06-02T10:00:12`, and no new task-frame-projected result artifacts
  exist. `94539.0` remains a 4-task Slurm step with training logs advancing;
  runtime is still below `MIN_TRAIN_SECONDS=10800`, so no object-WM
  conclusion is allowed.
- 2026-06-02 09:30 peg-drop v8 evidence-chain cleanup: `95088` completed,
  `95089` inspected it, gate `95090` failed, and cleanup `95093` canceled the
  dead gated-video branch `95091`/`95092`. Summaries `95094`/`95095`
  completed. No video/contact sheet exists because the no-video final-state
  gate did not pass.
- 2026-06-02 09:30 peg-alignment v9 scheduling: submitted no-video
  validation `95104` plus inspection/gate `95105`/`95106`; gated video
  branch `95107 -> 95108` depends on gate success; cleanup `95109` and
  summaries `95110`/`95111` use dedicated v9 branch/input files under
  `experiments/world_model_task_rebinding/rebinding_controller/`. This is a
  new controller validation, not a training job and not a metric change.
- 2026-06-02 09:33 no-video scheduling correction: updated pending no-video
  jobs `95039`, `95042`, and `95104` from a render-history-derived filter to a
  then-current live-state filter, so recovered old render-failure nodes could
  be used for non-rendering controller validation. Forecast still remained
  `2026-06-02T18:05:41`, so the delay was priority/resource related rather
  than a standing render exclusion list. The submitted node filters are
  historical snapshots only and must not be copied into future jobs.
- 2026-06-02 09:30 training liveness audit: object-WM training `94539` is
  still running on `server21` with one node, 4 H200 GPUs, and about
  `01:19:19` elapsed. This is below the 3-hour evidence floor, so no model
  quality conclusion is allowed yet. Failover jobs remain dependency-pending.
- 2026-06-02 09:35 queue audit: `94539` is still running on `server21` with
  4 H200 GPUs and about `01:24:06` elapsed, below the evidence floor. `94540`
  and `94682` are priority-pending 4-GPU `C_pi` label jobs forecast for
  `10:42:59` and `11:40:47`. Same-seed controller no-video jobs
  `95039`/`95042` and v9 `95104` are priority-pending and forecast for
  `2026-06-03T00:01:27`; no new metrics, inspection JSON, MP4, or contact
  sheet exists for them. RGB-D jobs remain dense allocations:
  `94541` requests one full 8-GPU node and `94676` requests two full 8-GPU
  nodes; neither is a sparse many-node/one-GPU allocation.
- 2026-06-02 09:38 liveness audit: `94539` remains running on `server21` with
  `gres/gpu=4`, `TresPerNode=gres:gpu:4`, `TimeLimit=03:30:00`, and about
  `01:27:04` elapsed. The Slurm stdout continues to print member epoch
  progress and stderr still only has the expected Transformer nested-tensor
  warnings. This is not training evidence until runtime exceeds the 3-hour
  floor and strict inspection `94863` passes. `94540` `C_pi` label forecast
  moved slightly to `10:51:00`; no controller/RGB-D artifact appeared.
- 2026-06-02 09:41 C_pi/RGB-D preflight: verified the pending downstream
  chains before they allocate. Larger C_pi label/training chains still use
  one-node 4-GPU jobs and training wrappers enforce `MIN_LABELS=512`,
  `MIN_TRAIN_SECONDS=10800`, one-node allocation, inspected label shards, and
  compliant post-training inspections before calibration. RGB-D render
  backfill `94676` remains dense (`2` nodes, `16` tasks/GPUs, `8` GPUs per
  node), and the dense render wrapper refuses sparse multi-node allocation at
  runtime. RGB-D inspection `94859` uses `REQUIRE_NO_WARNINGS=true`; slot
  training `94860` waits on that inspection and keeps 4 H200 / 10800-second
  training guards. No job was canceled or resubmitted.
- 2026-06-02 09:45 training-floor audit: object-WM `94539` is running on
  `server21` with one node, 4 H200 GPUs, and elapsed `01:34:43`, which is
  still below the 3-hour evidence floor. Only intermediate `best_model.pt`
  files exist; no final `model.pt`/`metrics.json` exists yet. Strict
  inspection/eval remain `94863 -> 94864`. C_pi label jobs `94540`/`94682` are
  priority-pending with concrete scheduler forecasts, and model training jobs
  `94542`/`94683` remain blocked on successful label generation while keeping
  4 H200 / `MIN_TRAIN_SECONDS=10800`. Old-alternate node-selection route object-WM retry
  `94809` remains queued as a failover behind failed inspections; since the
  primary object-WM job has already acquired 4 GPUs, no uncontrolled duplicate
  node-specific training job was submitted.
- 2026-06-02 09:49 recovered-node update: live `sinfo` showed a previously
  filtered node had recovered. Pending non-rendering jobs were updated from an
  older node-filter snapshot to a smaller then-current live-state filter:
  controller no-video jobs `95039`, `95042`, `95104`, C_pi label jobs
  `94540`, `94682`, full-shard no-video controller jobs `95055`, `95060`,
  `95065`, `95070`, RGB-D slot training `94860`, object-WM failovers `94679`,
  `94746`, and C_pi training jobs `94542`, `94683`. This is a historical
  scheduling correction, not a reusable node list. Old-alternate
  node-selection route object-WM failover `94809` was not modified because its
  complementary node-selection constraint was intentional for that submitted
  job only.
- 2026-06-02 09:52 queue/resource audit: `94539` continues running on
  `server21` with 4 H200 GPUs and elapsed `01:41:58`; it is still below the
  3-hour evidence floor and has no final checkpoints/metrics. `94540` and
  `94682` have no residual dependencies and are forecast for `10:51:00` and
  `11:40:47`. `94541` and `94676` also have no residual dependencies and
  remain dense RGB-D allocations. Node scan shows no schedulable 4-H200 slot:
  `server31` has only three free GPUs, `server61` has recovered but all GPUs
  are allocated, and the only nodes with eight nominal free GPUs are drained
  `server16`/`server29`. No extra duplicate job was submitted.
- 2026-06-02 10:01 controller evidence update: no-video v7 jobs `95039`
  and `95042` completed. Inspections `95040`/`95043` completed; CV gate
  `95041` failed as expected, while learned-WM gate `95044` passed with one
  final success after a prior dynamic event. Gated video job `95047` is
  pending on render-safe GPU resources; its walltime was shortened from
  `01:00:00` to `00:15:00` without changing evaluation settings, but the
  forecast remains `2026-06-04T07:02:14`. The result is intermediate until
  video/contact-sheet evidence from `95047`/`95048` is inspected. Because
  `95039` and `95042` used different seeds, paired no-video controls were
  queued: CV@7500 `95136 -> 95137 -> 95138` and learned-WM@7400
  `95139 -> 95140 -> 95141`.
- 2026-06-02 10:21 C_pi training/start audit: larger label job `94540`
  completed on `server28` after `00:17:21` and wrote four inspected shards plus
  `label_h5s.txt`. Larger-label training job `94542` started immediately on
  `server28` with one node / 4 H200 GPUs, `TimeLimit=03:30:00`, `MIN_LABELS=512`,
  and `MIN_TRAIN_SECONDS=10800`; inspection `94658` and calibration `94660`
  remain attached to the primary chain. Duplicate backfill jobs
  `94682`/`94683`/`94684`/`94685` were canceled after the primary chain became
  live, because keeping them would spend additional 4-GPU allocations on the
  same label/training input. Object-WM `94539` is still running below the
  3-hour evidence floor, and no model-quality conclusion is allowed until its
  strict inspection `94863` passes.
- 2026-06-02 10:25 queue/artifact audit: object-WM `94539` remains running on
  `server21` with one node / 4 H200 GPUs and elapsed `02:14:52`, below the
  evidence floor; no final checkpoint/metrics exist. Larger-label `C_pi`
  `94542` remains running on `server28` with one node / 4 H200 GPUs and
  elapsed about `00:05`, also far below the floor. Dense RGB-D render `94541`
  remains running on `server58` with one node / 8 H200 GPUs and only one H5
  output observed so far; dense 16-GPU backfill `94676` remains scheduled for
  `2026-06-03T17:24:12`. Gated video jobs `95047` and `95107` have
  dependencies released and are priority-pending on `server27` for
  `2026-06-02T11:37:05`; their old submitted node-exclusion snapshots are not
  reusable policy. No `job95047` or `job95107` visual artifact exists yet.
- 2026-06-02 10:30 queue/artifact audit: object-WM `94539` remains running
  below the 3-hour evidence floor, larger-label `C_pi` `94542` is still early
  in its 4-H200 run, and gated videos `95047`/`95107` still have no artifacts.
  Dense RGB-D render `94541` remains healthy but has only one observed H5 so
  far. Dense backfill `94676` moved earlier from `2026-06-03T17:24:12` to
  `2026-06-02T18:05:41` and is now pending for resources; it still requests
  2 nodes / 16 GPUs with 8 GPUs per node, so it remains compliant with the
  dense RGB-D allocation rule.
- 2026-06-02 11:49 RGB-D render hygiene update: dense render `94541` was
  canceled after `01:41:44` on `server58` because it had only one completed
  static shard while five dynamic/perturbation render processes showed
  `99-100%` GPU utilization and no first-trajectory progress. Its dependent
  partial-data inspection `94858` was also canceled. The renderer now shards
  by `(input_h5, trajectory)`, generating 96 work units for the full shard
  instead of six H5-level units, and each work unit has a default `1800s`
  timeout. Pending dense render `94676` was updated at that time with a
  job-local exclusion based on then-current render failures; that exclusion was
  not a reusable node policy and is superseded by the no-standing-node-list
  rule.
  Stale RGB-D chain `94859`/`94860`/`94861` was canceled and replaced with
  `95235`/`95236`/`95237`: inspection `95235` requires
  `MIN_RGBD_FILES=90` and no warnings; training `95236` requires one node /
  4 H200 GPUs, `MIN_TRAIN_SECONDS=10800`, and at least 90 RGB-D files; slot
  inspection `95237` requires compliant training evidence.
- 2026-06-02 11:50 render-node update for video evidence: video jobs `95191`,
  `95209`, and gated video job `95215` were updated to add `server58` to their
  rendering exclusion lists after the RGB-D hang. No-video controller probe
  `95212` was not changed because it is not a rendering job and the current
  policy allows non-rendering jobs to use recovered nodes with earlier rendering failures.
- 2026-06-02 11:58 queue audit: video evidence jobs `95191` and `95209` had
  job-local render exclusions from fresh Vulkan/render failures and were
  forecast for `2026-06-02T18:05:41`; gated video job `95215` carried the same
  submitted render filter while waiting on `95214`. No-video
  insertion-manifold probe `95212` kept a then-current non-rendering live-state
  filter and could schedule on nodes with old render-failure history because
  it did not request video/RGB-D rendering. This was a historical split between
  render jobs and no-video jobs, not a standing node list.
- 2026-06-02 14:15 queue cleanup: stale non-RGB-D video/debug branches
  `95191`, `95192`, `95209`, `95210`, `95215`, and `95216` were canceled while
  pending. The current queue is now focused on the RGB-D method chain
  (`94676 -> 95938 -> 95705 -> 95706 -> 95707 -> 95708 -> 95709 -> 95710 ->
  95711`, plus `95764/95765/95781` and diagnostic `95811`) and RGB-D
  render-node canaries. This avoids spending GPU time on stale state/oracle
  video evidence that cannot satisfy the method requirement.
- 2026-06-02 14:26 strict RGB-D inspection repair: canceled old pending
  inspection `95704` and submitted replacement `95914` with explicit
  `MIN_RGBD_FILES=90`, `REQUIRE_NO_WARNINGS=true`, and the same `job94676`
  RGB-D root. This intermediate repair was superseded by the 14:34 wrapper
  hardening below because the submitted batch script still did not show those
  strict values directly.
- 2026-06-02 14:34 strict RGB-D inspection wrapper hardening: because Slurm
  still does not show pending job export values, `95914` was superseded by
  `95938`, submitted from `scripts/slurm/inspect_full_rgbd_dataset_strict.sbatch`.
  That submitted batch script itself hardcodes `MIN_RGBD_FILES=90` and
  `REQUIRE_NO_WARNINGS=true`, and defaults to
  `rgbd_dynamic_dense/from_rollout_dir/job94676`. Updated `95705` to depend on
  `afterok:95938` and canceled `95914`.
- 2026-06-02 14:30 render-node canary result: `server58` canary `95265`
  failed with Vulkan `ErrorDeviceLost` before writing any RGB-D H5; dependent
  inspection `95372` failed with no files. This is job-local node-health
  evidence only, not method evidence and not a standing render exclusion rule.
- 2026-06-02 14:47 auditable RGB-D chain replacement: because Slurm does not
  expose pending job `--export` values, the old post-inspection chain
  `95705`-`95711`, `95764`, `95765`, `95781`, and `95811` was canceled while
  dependency-pending. Replacement chain
  `95996 -> 95997 -> 95998 -> 95999 -> 96001 -> 96002 -> 96003`, plus
  diagnostic `96000` and controller/video review `96004 -> 96005/96006`, was
  submitted through `scripts/slurm/submit_auditable_rgbd_method_chain_job94676.sh`.
  Its manifest records `RGBD_ROOT=job94676`, `MIN_RGBD_FILES=90`,
  `MIN_TRAIN_SECONDS=10800`, RGB-D predicted slots as world-model input, and
  `SLOT_SOURCE=rgbd` for the controller. This is audit/gate hardening and not
  method evidence.
- 2026-06-02 14:55 render-wrapper retry hardening: `scontrol write
  batch_script 94676` showed the queued `94676` snapshot still calls Python
  directly for main-wrapper rollout validation. The task script itself is
  current and already uses trajectory-level worklist rendering plus retry and
  timeout. Updated the current
  `render_dynamic_rgbd_dataset_from_rollout_dir_dense.sbatch` so future
  submissions source `python_retry.sh` and use retry for both worklist
  generation and validation. `bash -n` passed. A replacement 2x8 dry-run with
  the fixed wrapper forecast later than `94676`, so no replacement was
  submitted.
- 2026-06-02 15:03 render-audit wrapper hardening: old pending audit job
  `95870` used the generic `audit_rgbd_render_output.sbatch` and depended on
  hidden export values for `OUTPUT_ROOT` and `RENDER_JOB_ID`. Added auditable
  `audit_rgbd_render_output_job94676.sbatch`, canceled `95870`, and submitted
  replacement `96069` with `afterany:94676`. The submitted script itself
  records `OUTPUT_ROOT=.../job94676`, `RENDER_JOB_ID=94676`, and
  `MIN_EXPECTED_RGBD_FILES=90`. This is failure-localization hardening; strict
  RGB-D inspection `95938` remains the data gate. Later full96 correction:
  `96069` and `95938` were canceled while pending, because 90 files is too
  weak for the 96-trajectory source. The next audit/gate jobs became `96311`
  and exact full96 inspection `96266`. Later exact-argument correction:
  `96311` was canceled while pending and replaced by `97571`, because the old
  submitted snapshot passed only `MIN_EXPECTED_RGBD_FILES=96`; current primary
  render-output audit is `97571` with exact `EXPECTED_RGBD_FILES=96` passed to
  the Python audit.
- 2026-06-02 12:05 training-floor and node-specific audit: `94542` is the current
  active training job and still satisfies the floor with one node / 4 H200 GPUs,
  `TimeLimit=03:30:00`, and `MIN_TRAIN_SECONDS=10800`; it is not evidence until
  it reaches the floor and inspection `94658` passes. RGB-D slot training
  `95236` is queued behind data inspection and also requests 4 H200 GPUs with
  `MIN_TRAIN_SECONDS=10800`. To test whether nodes with earlier rendering failures have
  recovered without risking a dense dataset hang, submitted small RGB-D canary
  jobs `95265` on `server58` and `95266` on `server10` using
  `scripts/slurm/render_node_rgbd_canary.sbatch`. These are health probes,
  not training runs; if they pass, update the render-node policy before using
  those nodes for large RGB-D or video jobs.
- 2026-06-02 12:07 node-specific availability follow-up: `server10` and `server58`
  are `MIXED` but all eight GPUs are allocated, so their canaries may still wait
  despite the explicit `ReqNodeList`. Most old nodes with earlier render failures are also fully
  GPU-allocated. Submitted two additional one-GPU canaries on old render-failure-history
  nodes with apparent free GPUs: `95357` on `server39` and `95358` on
  `server56`. This implements the "try old node-specific nodes if cards are hard to get"
  policy without changing the strict training floor or sparse-allocation rule.
- 2026-06-02 12:15 canary gate hygiene: node-specific render canaries now have CPU
  inspection dependencies so a completed canary cannot be treated as healthy
  merely because the Slurm job exited. Inspection jobs are `95372` after
  `95265`, `95370` after `95266`, `95371` after `95357`, and `95373` after
  `95358`, all with `MIN_RGBD_FILES=1` and `REQUIRE_NO_WARNINGS=true`.
- 2026-06-02 12:18 large-allocation check: before submitting any bigger RGB-D
  renderer, ran Slurm `--test-only` for dense 4-node/32-GPU and 8-node/64-GPU
  versions of the existing full-shard RGB-D job. The forecasts were
  `2026-06-05T16:39:26` and `2026-06-06T17:12:44`, both later than existing
  2-node/16-GPU job `94676` at `2026-06-03T00:01:27`. No bigger duplicate was
  submitted.
- 2026-06-02 12:30 environment/controller hygiene: repaired the shared venv's
  broken mixed NumPy install by force-reinstalling `numpy==1.26.4`, then
  verified NumPy, h5py, torch imports and `pip check`. This protects queued
  CPU inspections/calibrations from failing for environment reasons. Added
  calibrated no-video C_pi controller chain `95392 -> 95393 -> 95394` behind
  larger-label calibration `94660`; the earlier pending chain
  `95384 -> 95385 -> 95386` was canceled before running and replaced with a
  `WORLD_MODEL_DIRS_FILE` submission to avoid space-delimited Slurm export
  ambiguity. The submitted eval job used a then-current no-video live-state
  filter, which is historical scheduling evidence only. Do not submit the video
  branch until gate `95394` passes; video must use fresh live render evidence
  and direct contact-sheet/video inspection.
- 2026-06-03 01:34 resource/action check: fresh non-executing probes were run
  because the user explicitly asked to verify whether cards are actually
  obtainable rather than infer from visible `mix` nodes. Current association
  remains `yanhongru|mayi`; `gpu`, `cpu`, and `debug` allow `mayi`, while
  `gaosh`, `engram`, and `test` list `AllowAccounts=null` and reject this user
  with `Invalid account or account/partition combination specified`. `gpux`
  and `mgpu` are drained or inactive. Ten-minute `sbatch --test-only` probes
  for new jobs report: `cpu` 1/2/4 GPU at `2026-06-04T00:18:15`, `gpu`
  1/2/4 GPU at `2026-06-04T21:25:15`, `debug` 1 GPU at
  `2026-06-04T07:03:05`, and `debug` 2/4 GPU blocked by `MaxGRESPerAccount`.
  Therefore the active useful GPU work is the already running `99719` 4-H200
  RGB-D slot training, and the next action is artifact/state analysis plus
  approximately half-hour queue/artifact checks, not duplicate submissions.
- `2026-06-04 21:43+08:00` bounded no-exclusion one-H200 follow-up:
  after the user clarified that small jobs should be requested promptly and
  old bad-node history should not explain current queue behavior, rechecked
  live `scontrol` and wrapper source. Current main/downstream RGB-D method
  jobs are not carrying a standing exclusion list: `105236`, `105385`,
  `106000`, `105238`, `105553`, and controller-video jobs have
  `ExcNodeList=(null)`. The only active node-specific exclusions are
  job-local `server13` on a subset of one-H200 fallback allocations, tied to
  same-day CUDA visibility failures in no-exclusion canaries. Fresh one-H200
  `sbatch --test-only` probes for `03:00:00`, `06:00:00`, `12:00:00`, and
  `1-00:00:00` forecast the same start around `2026-06-05T01:00:54` on
  `server07`, with or without `server13` excluded. Submitted one bounded
  no-exclusion one-H200/one-day tmux allocation `106147`
  (`h200_1gpu_pool11`, seed `2500`). Real `scontrol` now shows
  `StartTime=2026-06-04T22:56:00`, `SchedNodeList=server07`, and
  `ExcNodeList=(null)`, so this is a useful earlier rolling request. The
  physical reason is to reduce the risk that RGB-D slot perception remains
  the blocker for the RGB-D-derived world-model/controller chain while
  preserving exact1000 data, strict gates, and the no-standing-bad-node
  policy.
- `2026-06-04 21:48+08:00` one-H200 forecast and backup-assembly guard:
  `106147` later slipped to `StartTime=2026-06-05T01:00:00`,
  `SchedNodeList=server21`, still with `ExcNodeList=(null)`. Fresh
  non-executing probes for one-H200 `03:00:00`, `06:00:00`, `12:00:00`, and
  `1-00:00:00` all forecast later (`2026-06-05T02:02:42`) on `server21`,
  with or without `server13`, so no replacement request was submitted. Ran
  the backup assembly guard in dry-run mode over active one-H200 sources
  `105385`, `105535`, `105743`, `105867`, `106000`, and `106001`; it
  returned status `65` with `all_eligible=false` because the sources lack
  completed checkpoint/metrics artifacts and are below the `10800` second
  evidence floor. This confirms the fallback path cannot accidentally turn
  incomplete rolling allocations into slot evidence. The guard preserves the
  exact1000 RGB-D data, one-H200/3h floor, and strict downstream gates.
- `2026-06-04 21:52+08:00` no-exclusion fallback allocation actually started:
  `106147` / tmux `h200_1gpu_pool11` started early on `server62` with
  `ExcNodeList=(null)`. The fixed `srun` CUDA canary passed with one visible
  H200, and exact1000 seed `2500` emitted `rgbd_slot_train_start`. The
  manifest records `requested_gres=gpu:NVIDIAH200:1`,
  `min_train_seconds=10800`, `expected_rgbd_files=1000`,
  `require_cuda=true`, and the RGB-D-images-plus-robot-proprio/no-oracle-state
  boundary. This is bounded rolling fallback coverage for the slot perception
  blocker, not method evidence and not a replacement for the strict main
  dependency chain.
- `2026-06-04 22:01+08:00` user scheduling correction follow-up:
  re-audited live exclusions after the user questioned the bad-node story.
  The current main and downstream method path still has no standing bad-node
  list: `105236`, `105385`, `106000`, `106147`, `105238`, `105553`, and
  controller-video jobs show `ExcNodeList=(null)`. A subset of fallback jobs
  has only job-local `server13` exclusions from same-day no-exclusion CUDA
  visibility failures, and this is not reused as a global policy. Fresh
  no-exclusion one-H200 probes found `cpu`/`cpu,gpu` fit earlier than
  `gpu,cpu`; short and one-day requests forecast the same time for a given
  probe, so reducing below one day did not improve the forecast. Submitted
  bounded additional no-exclusion requests: tmux allocation `106237`
  (`h200_1gpu_pool12`, seed `2600`, one H200, one day) and ordinary Slurm
  fallback `106245` (`wm_rgbd_slot_1h200_sbatch13`, seed `2700`, one H200,
  one day). Both use the same `srun` CUDA canary and exact1000 RGB-D slot
  runner. Latest `scontrol` shows `106237` forecasting
  `2026-06-05T03:00:00` on `server13` with `ExcNodeList=(null)`; if it
  actually starts there, the runner will fail fast on the CUDA canary if the
  current server13 GPU-visibility issue persists. `106245` forecasts
  `2026-06-05T08:00:00` on `server07`, also with `ExcNodeList=(null)`.
  This is scheduling risk reduction for the RGB-D slot perception blocker;
  it does not change labels, exact-count gates,
  predicted-slot thresholds, RGB-D-derived world-model requirements,
  controller evaluation, or the 10-sample visualization budget.
- `2026-06-04 22:21+08:00` fast-start no-exclusion result and corrected
  queue interpretation:
  the user's cluster note says small jobs should normally wait about
  `10-30` minutes, so the live audit was repeated instead of relying on stale
  node-history explanations. The newest no-exclusion one-H200 requests started
  quickly, matching that guidance: `106237` started at
  `2026-06-04T22:17:10+08:00` on `server07`, and `106245` started at
  `2026-06-04T22:19:11+08:00` on `server21`; both have
  `ExcNodeList=(null)`, one visible H200, a passed CUDA canary, and exact1000
  RGB-D slot train-start events. Current broad method jobs are not constrained
  by a bad-node list: `105236`, `105238`, `105553`, and controller-video jobs
  all have `ExcNodeList=(null)`. Running no-exclusion fallbacks are
  `105385`, `106000`, `106147`, `106237`, and `106245`. Only
  `105535/105743/105867/106001` keep job-local `server13` exclusions from
  same-day no-exclusion CUDA visibility failures; this evidence must not be
  reused as a default policy. Future resource actions should request one-H200
  allocations with no exclusion first, use one-day or shorter walltime
  according to live `sbatch --test-only` and actual `squeue/scontrol`
  forecasts, and avoid extra fanout once enough H200 work is already running.
  This preserves the one-H200/three-hour training floor and all strict
  RGB-D-derived downstream gates.
- `2026-06-04 22:42+08:00` reusable-allocation correction after user
  escalation:
  audited all one-H200 allocation attempts with `squeue`, `sacct`, tmux
  panes, and allocation-runner status files. The current useful tmux/salloc
  jobs are still held and running: `105385`, `105535`, `105743`, `105867`,
  `106000`, `106001`, `106147`, and `106237`. The short releases were not
  completed training runs: `105646` failed after `00:00:12` because the
  first CUDA canary incorrectly ran outside `srun`; `105707` and `105868`
  failed after `00:00:11` and `00:00:17` on `server13` because the fixed
  `srun` canary saw no usable CUDA device before RGB-D dataset loading.
  `105826/105827` were canceled while pending with zero allocation to replace
  them with better forecast requests. These are scheduling/canary outcomes,
  not experiment conclusions and not RGB-D method evidence.

  Going forward, do not start new long RGB-D training fallbacks with ordinary
  one-shot `sbatch` if a reusable tmux/salloc path is available. The ordinary
  fallback `106245` is already running and should not be canceled just to
  enforce this policy retroactively, but it cannot be reused after completion.
  Future one-H200 requests should be tmux-backed `salloc` allocations with
  `KEEP_ALLOCATION_AFTER_RUN=true`, no default node exclusion, and an
  immediate `srun` CUDA canary. A successful CUDA-visible allocation must stay
  open after the training command so strict inspection/export/retry commands
  can be run inside the same allocation instead of returning to the queue.
  CUDA-invisible allocations are released before dataset loading because they
  cannot run H200 work and holding them would consume team resources without
  making progress.

## Repository Hygiene

- [ ] Keep new outputs under method-specific directories.
- [ ] Do not write new experimental results into old archived directories.
- [ ] Keep `PLAN/README.md` and `TODO/README.md` as stable entry points.
- [ ] Add small focused md files for new work instead of mixing unrelated notes.
- [ ] Keep failed pilots documented with the exact reason and whether they
      count as evidence.

Completion standard: a future reader can identify active work, archived work,
and usable assets without reading old restoration logs.
