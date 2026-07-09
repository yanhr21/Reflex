# Dataset TODO

- [x] Move old experiment outputs under `experiments/legacy/`.
- [x] Move old logs under `logs/legacy/`.
- [x] Move old plan / TODO / idea notes under `legacy`.
- [x] Record that the official DP checkpoint is state-based, not RGB-based.
- [x] Record that official 1000 demos currently need RGB rendering before
  Cosmos training.
- [x] Record that old `fix3_733` is not enough as final method data.

## Current Results 2026-07-07

- [x] Finish A static official RGB full render as 20 shards
  `static_rgb/full_s00a` ... `static_rgb/full_s09b`: 1000 videos, 480 review
  frames, 30 FPS, no invalid shards in the final aggregate.
- [x] Verify A static readiness with
  `scripts/world_model/require_dataset_static_full_ready.sh`:
  `dataset_static_full_ready=true`, `ready_source=shards`.
- [x] Verify full joint training readiness still correctly fails with
  `reason=full_joint_inputs_incomplete` because B/C/D/E production datasets
  and active indexes are not ready.
- [x] Batch-launch B/C/D smoke after A completion; E smoke correctly skipped
  with `reason=e_prereqs_not_ready`.
- [x] Reject and archive invalid B/C/D smoke runs after human visual review:
  `experiments/legacy/01_dataset/invalid_bcd_smoke_20260707/`. The rejected
  videos were about 2 seconds, did not show complete 300-frame episodes, and C
  moved the holed target during robot grasp / initial approach, which is
  strictly invalid.
- [x] Regenerate B dynamic RGB observation smoke as a complete 300-frame,
  30 FPS episode before any B production. The `170669` retry completed but
  was invalid because the collector did not pass `max_episode_steps=300` to
  ManiSkill and produced only 100 frames; it is archived under
  `experiments/legacy/01_dataset/invalid_bcd_smoke_20260708_short_frames/`.
  The later `171179` retry on `server53` failed render canary timeout and is
  archived under
  `experiments/legacy/01_dataset/invalid_bcd_smoke_20260708_server53_c_success/`.
  The `171196` retry on `server36` failed render canary timeout and is
  archived under
  `experiments/legacy/01_dataset/invalid_b_smoke_20260708_server36_canary_timeout/`.
  The active regenerated smoke is `dynamic_rgb/smoke01`: job `171210` on
  `server10`, 300 frames, 30 FPS manifest, `state_intervention=false`, and
  `snap_or_teleport=false`.
- [x] Regenerate C frozen-DP dynamic failure smoke as a complete 300-frame,
  30 FPS rollout. The holed target must remain static during grasp / initial
  approach and may only move after the delayed motion start. The `170671`
  retry completed but was invalid because it ended at 133 frames after frozen
  DP success; it is archived under
  `experiments/legacy/01_dataset/invalid_bcd_smoke_20260708_short_frames/`.
  The later `171177` retry produced 300 frames but still had
  `success_once=true`, so it is diagnostic only and archived under
  `experiments/legacy/01_dataset/invalid_bcd_smoke_20260708_server53_c_success/`.
  The active regenerated smoke is `frozen_dp_dynamic/smoke01`: job `171197`
  on `server57`, 300 frames, 30 FPS manifest, target motion starts at step
  120, `success_once=false`, `success_at_end=false`,
  `target_assisted=false`, `state_intervention=false`, and
  `snap_or_teleport=false`.
- [x] Archive failed B/D attempts instead of leaving them in active run paths:
  B `smoke01_cuda_tensor_failed_server44`,
  D `smoke01_server44_device_lost`,
  D `smoke01_server46_canary_timeout`,
  D `smoke01_server58_canary_timeout`,
  D `smoke01_cuda_tensor_failed_server60`,
  D `smoke01_server63_canary_timeout`.
- [x] Patch B and D collectors so rendered frames / info / summaries convert
  CUDA tensor values through CPU numpy before JSON or image writing.
- [x] Add `server46`, `server58`, and `server63` to default B/C/D/E smoke
  exclude lists after canary timeout evidence; keep failed attempts in legacy
  with logs.
- [x] Regenerate D future-frame cooperation teacher smoke as complete
  300-frame, 30 FPS teacher rollouts before any D production. The `170670`
  retry completed but was invalid because each rollout was about 100 frames;
  it is archived under
  `experiments/legacy/01_dataset/invalid_bcd_smoke_20260708_short_frames/`.
  The later `171176` retry on `server53` failed render canary with Vulkan
  `ErrorDeviceLost` and is archived under
  `experiments/legacy/01_dataset/invalid_bcd_smoke_20260708_server53_c_success/`.
  The active regenerated smoke is `future_teacher/smoke01`: job `171198` on
  `server10`, 4 videos, 1200 total frames, 30 FPS manifest,
  `teacher_evidence_allowed=true`, `state_intervention=false`, and
  `snap_or_teleport=false`. It is complete smoke evidence only;
  `success_once=false`, so it is not a successful teacher-controller claim.
- [x] Prepare a new B/C/D smoke review batch after regeneration. Do not start
  B/C/D production until each regenerated class smoke is approved.
  New batch review file:
  `experiments/maniskill/runs/01_dataset/review/bcd_smoke_review_20260708.md`.
  Current status is blocked on human review approval, not on generation.
- [x] Add run-local and review-level goal-blocked markers for regenerated
  B/C/D smoke:
  `experiments/maniskill/runs/01_dataset/review/goal_blocked.md`,
  `dynamic_rgb/smoke01/goal_blocked.md`,
  `frozen_dp_dynamic/smoke01/goal_blocked.md`, and
  `future_teacher/smoke01/goal_blocked.md`.
- [x] Reject and archive the regenerated B/C/D smoke after user visual review
  found the core task motion wrong: the videos did not show credible robot
  grasp / active peg manipulation, and checking only the first frame was
  insufficient. The invalid active batch was moved to
  `experiments/legacy/01_dataset/invalid_bcd_smoke_20260708_no_grasp_task_motion/`
  with matching logs under
  `logs/legacy/01_dataset/invalid_bcd_smoke_20260708_no_grasp_task_motion/`.
- [x] Patch B and D smoke generation to replay legal official
  `pd_ee_delta_pose` demo actions while applying the dynamic adapter, instead
  of producing zero-action / non-task-motion videos. The new collector is
  `scripts/world_model/collect_dynamic_demo_action_smoke.py`.
- [x] Patch C frozen-DP dynamic smoke with an explicit task-motion quality
  gate: TCP motion, peg motion, and at least one grasp must be recorded before
  a smoke can be accepted for review.
- [x] Regenerate B dynamic RGB smoke after the task-motion fix: job `171244`
  on `server57`, 300 frames at 30 FPS,
  `task_motion_quality_gate_passed=true`, `grasp_once=true`,
  `tcp_motion_m=0.30217133829062237`,
  `peg_motion_m=0.3225330658257392`, `success_once=false`,
  `state_intervention=false`, and `snap_or_teleport=false`. Visual frame
  checks now include first / grasp / middle / last frames, not only the first
  frame. This B smoke is dynamic observation / failure data, not positive
  insertion data.
- [x] Archive the C/D retry on `server28`: both failed the render canary with
  timeout before collection. Artifacts are under
  `experiments/legacy/01_dataset/invalid_cd_smoke_20260708_server28_canary_timeout/`
  with matching logs under
  `logs/legacy/01_dataset/invalid_cd_smoke_20260708_server28_canary_timeout/`.
- [x] Add `server28` to B/C/D/E smoke default node exclusions after repeated
  render-canary timeout evidence.
- [x] Archive the C/D retry on `server60`: C failed the render canary with
  timeout before collection, and D was cancelled after it was assigned to the
  same newly diagnosed bad smoke node. Artifacts are under
  `experiments/legacy/01_dataset/invalid_cd_smoke_20260708_server60_canary_timeout/`
  with matching logs under
  `logs/legacy/01_dataset/invalid_cd_smoke_20260708_server60_canary_timeout/`.
- [x] Add `server60` to B/C/D/E smoke default node exclusions after C
  render-canary timeout evidence on job `171261`.
- [x] Archive the C/D retry with two separate failures: C hit Vulkan
  `ErrorDeviceLost` on `server34`, and D exposed stale runner arguments after
  the collector was switched to `collect_dynamic_demo_action_smoke.py`.
  Artifacts are under
  `experiments/legacy/01_dataset/invalid_cd_smoke_20260708_server34_and_d_args/`
  with matching logs under
  `logs/legacy/01_dataset/invalid_cd_smoke_20260708_server34_and_d_args/`.
- [x] Remove stale D runner arguments `--future-tau-steps`,
  `--max-action-translation`, and `--approach-offset-*` from
  `scripts/slurm/run_dataset_future_frame_teacher_smoke_in_allocation.sh`.
- [x] Add `server34` to B/C/D/E smoke and production default node exclusions
  after C render-canary `ErrorDeviceLost` evidence on job `171273`.
- [x] Regenerate C and D again after the task-motion fixes and updated smoke
  exclusions. The stale pending jobs `171280` / `171281` were replaced after
  inspection. The successful active C smoke is job `172081` on `server10`:
  300 frames, 30 FPS, `success_once=false`, `success_at_end=false`,
  `state_intervention=false`, `snap_or_teleport=false`,
  `target_assisted=false`, and `task_motion_quality_gate_passed=true`. The
  successful active D smoke is job `172058` on `server27`: 4 videos, 1200
  total frames, 30 FPS, `teacher_future_target_source=ground_truth_future_motion_plan`,
  `teacher_action_adapter=official_demo_actions_plus_gt_future_residual`,
  `state_intervention=false`, `snap_or_teleport=false`, and
  `task_motion_quality_gate_passed=true`. The failed `server07` render-canary
  attempt was archived under
  `experiments/legacy/01_dataset/invalid_cd_smoke_20260708_server07_canary_timeout/`.
  The new B/C/D review batch is
  `experiments/maniskill/runs/01_dataset/review/bcd_smoke_review_20260708.md`.
  Do not start B/C/D production until explicit human review approval is
  recorded.
- [x] Add B/C/D batch review decision helper:
  `scripts/world_model/record_dataset_bcd_smoke_review_decision.sh`. It must
  be used for approval only after explicit user approval; `--dry-run` is safe
  for validation.
- [x] Add B/C/D review-block status helper:
  `scripts/world_model/dataset_bcd_review_block_status.sh`, summarizing review
  artifacts, approval files, and the next production command once approved.
- [x] Sync B/C/D/E production default node exclusions with current render-risk
  evidence:
  `server10,server28,server30,server35,server36,server39,server43,server44,server46,server56,server57,server58,server59,server63`.
- [x] Add `server36` to B/C/D/E smoke default node exclusions after B
  render-canary timeout evidence on job `171196`.
- [x] Clean and update the render-risk status helper after regenerated B/C/D
  smoke:
  `scripts/world_model/dataset_render_risk_status.sh` is ASCII text again,
  records B/C/D smoke success evidence on `server10` / `server57`, and records
  current render-risk evidence for `server36` and `server53`.
- [x] Update `docs/dataset_smoke_runbook.md` with 2026-07-08 B/C/D smoke
  evidence: `server36` canary timeout, `server53` canary timeout /
  `ErrorDeviceLost`, B success on `server10`, C failure-smoke success on
  `server57`, and D complete teacher-smoke generation on `server10`.
- [x] Add B/C/D multi-motion production shard plan:
  `scripts/world_model/dataset_bcd_production_shard_plan.sh`. The active
  production balance covers `constant_lr`, `constant_fb`, `reverse`,
  `move_stop`, `sine`, and `continuous`; `peg_disturb` is not counted until a
  real peg-disturbance runner exists.
- [x] Add guarded dry-run-by-default B/C/D shard launcher:
  `scripts/slurm/launch_dataset_bcd_production_shards_tmux.sh`. It supports
  `--stage`, `--family`, and `--max-launches`; execution defaults to one shard
  to keep resource use auditable.
- [x] Verify the B/C/D shard launcher dry-run submits no Slurm jobs, and
  `--execute --stage B --family lr --max-launches 1` stops while B/C/D smoke
  approval is absent or invalid.
- [x] Extend production validation to aggregate shard directories under
  `prod01/<family>` when a single `prod01/summary.json` is absent.
- [x] Add shard production index builder:
  `scripts/world_model/build_dataset_production_shard_index.sh`.
- [x] Add read-only B/C/D next production shard helper:
  `scripts/world_model/dataset_bcd_production_next_shard.sh`.
- [x] Update the B/C/D shard launcher to skip already-complete shards and to
  refuse relaunching over an incomplete existing shard, forcing archive or
  diagnosis first.
- [x] Add guarded dry-run-by-default next-shard launcher:
  `scripts/slurm/launch_dataset_bcd_next_production_shard_tmux.sh`. It
  refuses to launch before B/C/D human-review approval and submits at most one
  production shard after approval.
- [x] Regenerate the requested multi-motion B/C/D smoke matrix after user
  review rejected the single-family batch. The completed matrix covers
  `lr_pos`, `lr_neg`, `fb_pos`, `fb_neg`, `reverse`, `sine`, and
  `peg_disturb` for B/C/D. B has 14 videos / 4200 frames, C has 7 videos /
  2100 frames with `motion_trigger_mode=inserted`, and D has 14 videos /
  4200 frames with teacher-only ground-truth future residual labels. All
  manifests validate, `state_intervention=false`, and `snap_or_teleport=false`.
  Success/failure is recorded as a label only, especially for C. Review file:
  `experiments/maniskill/runs/01_dataset/review/bcd_smoke_matrix_review_20260708.md`.
  This matrix was later rejected and archived because C timing was wrong.
- [x] Reject and archive the 2026-07-08 multi-motion matrix after user visual
  review found the C timing wrong: C used `motion_trigger_mode=inserted`, so
  target/hole motion could begin only after insertion. That is not the desired
  replanning setting. The invalid matrix is archived under
  `experiments/legacy/01_dataset/20260709_bcd_matrix_invalid_c_trigger_after_insertion/`
  with logs under
  `logs/legacy/01_dataset/20260709_bcd_matrix_invalid_c_trigger_after_insertion/`.
- [x] Regenerate the B/C/D multi-motion smoke matrix with corrected C timing.
  C now uses `motion_trigger_mode=pre_insert_l2` and
  `motion_trigger_threshold_m=0.12`, so motion triggers only when the peg is
  near the hole and `inserted=false`. The C audit shows all seven families
  trigger before insertion: `lr_pos`, `lr_neg`, `fb_pos`, `fb_neg`, `reverse`,
  `sine`, and `peg_disturb`. Success/failure is recorded only as a label; C is
  not rejected merely because frozen DP still succeeds after a pre-insertion
  disturbance. Corrected review file:
  `experiments/maniskill/runs/01_dataset/review/bcd_smoke_matrix_review_20260709_preinsert.md`.
  This review gate was later cleared by explicit user approval on 2026-07-09.
- [x] Record explicit user approval for the corrected B/C/D smoke matrix and
  launch ABCD full production on 2026-07-09. A static RGB is already complete:
  20 shards, 1000 videos, 480 review frames, 30 FPS. B/C/D production was
  launched as 18 tmux-held Slurm shards under `prod01/<family>`:
  B target 1000 episodes, C target 500 rollouts, and D target 500 teacher
  rollouts. C success/failure remains only an outcome label, not a production
  blocker.
- [x] Archive and relaunch early production parameter/node failures without
  treating C success as invalid. B `reverse` / `stop` and C `reverse` first
  failed because `MAX_STEP_DELTA_M=0.004` was too tight for those continuous
  motion profiles; the failed attempts were archived under
  `experiments/legacy/01_dataset/20260709_b_reverse_stop_step_delta_too_tight/`
  and
  `experiments/legacy/01_dataset/20260709_c_reverse_step_delta_too_tight/`.
  C `stop` then failed render canary on `server07`; pending jobs were requeued
  after adding `server07` to the production exclude list, with archive record
  `experiments/legacy/01_dataset/20260709_requeue_pending_after_server07_canary_timeout/`.
  Relaunched `reverse` / `stop` shards use `MAX_STEP_DELTA_M=0.0045`.
- [x] Fix and relaunch C production after discovering the first production
  shards were not using the approved corrected C semantics. C production now
  defaults to `motion_trigger_mode=pre_insert_l2`, matching the approved
  2026-07-09 smoke matrix where target motion begins before insertion when
  `peg_head_l2 <= 0.12` and `inserted=false`. The C collector now separates
  rollout attempts from accepted rollouts: attempts that never reach the
  pre-insertion trigger or fail the task-motion quality gate are recorded as
  `skipped_attempts` and do not pollute accepted action / motion / task traces
  or videos. This does not reject C success; success remains an outcome label.
  The old C production shards launched with the wrong trigger / aborting
  collector were archived under
  `experiments/legacy/01_dataset/20260709_c_prod_old_trigger_and_abort_collector/`.
  The one-second C `lr` shell-manifest typo attempt was archived under
  `experiments/legacy/01_dataset/20260709_c_lr_shell_manifest_typo/`.
  C `lr`, `fb`, `reverse`, `stop`, `sine`, and `cont` were relaunched under
  active `frozen_dp_dynamic/prod01/<family>` with
  `MAX_STEP_DELTA_M=0.0045`; `fb` has already passed render canary and entered
  collection, and the remaining C shards are queued or running as tmux-held
  Slurm jobs.
- [x] Archive and relaunch D `reverse` / `stop` production failures. D
  `reverse` failed render canary on `server02`, so `server02` was added to
  the production default exclude list alongside `server07`; D `stop` failed
  because `MAX_STEP_DELTA_M=0.0045` was still too tight for the teacher
  residual path. Both old attempts were archived under
  `experiments/legacy/01_dataset/20260709_d_reverse_stop_canary_step_delta_requeue/`.
  D `reverse` and `stop` were relaunched with `MAX_STEP_DELTA_M=0.005` and
  the updated node exclusions.
- [x] Archive and relaunch B `reverse` / `stop` production failures from the
  same production pass. B `reverse` failed render canary on `server02`, and B
  `stop` failed the continuous-motion validation with
  `0.004592m > 0.004500m`. Both old attempts were archived under
  `experiments/legacy/01_dataset/20260709_b_reverse_stop_canary_step_delta_requeue/`.
  B `reverse` and `stop` were relaunched with `MAX_STEP_DELTA_M=0.005` and
  the updated production exclude list.
- [x] Fix B/D demo-action production collector after B `lr`, `fb`, `cont`,
  and `sine` all hit the same non-triggering source demo episode (`source
  episode 130`). The collector now separates source attempts from accepted
  episodes and records skipped source episodes when a source demo never
  reaches the configured motion trigger or fails the task-motion quality gate.
  Skipped source episodes do not enter accepted trace rows, accepted videos,
  or the production count. This preserves B as dynamic observation data and
  D as teacher-only data without treating an unhelpful source episode as a
  shard-wide failure. The old B failed/interrupted attempts were archived
  under
  `experiments/legacy/01_dataset/20260709_b_prod_source_episode_skip_requeue/`.
  B `lr`, `fb`, `reverse`, `stop`, `sine`, and `cont` were relaunched with
  `MAX_STEP_DELTA_M=0.005`; `lr` had one transient shell-manifest typo
  attempt archived under
  `experiments/legacy/01_dataset/20260709_b_lr_shell_manifest_typo/` and was
  relaunched again.
- [x] Add a guarded B/C/D expansion launcher for additional data generation
  after `prod01`. The new read-only plan is
  `scripts/world_model/dataset_bcd_expansion_shard_plan.sh`; the dry-run by
  default launcher is
  `scripts/slurm/launch_dataset_bcd_expansion_shards_tmux.sh`. The default
  expansion root is `prod02`, using the same approved multi-motion families,
  30 FPS, active adapter, B/C/D smoke approval gates, and one-GPU-per-shard
  Slurm/tmux production wrappers. The launcher refuses existing output dirs
  and does not overwrite `prod01`.
- [x] Add a guarded combined review-index builder for the expanded dataset:
  `scripts/world_model/build_dataset_bcd_prod02_review_index.sh`. Default
  mode writes one B/C/D `prod02` human-review markdown only after all 18
  shards are complete and pass summary quality gates. `--status-only` prints
  current counts and intentionally exits nonzero while shards are incomplete.
  The 2026-07-09 23:40 status-only check correctly refused to build the final
  review because only 322/2000 expansion videos existed and no shard had
  written `summary.json` yet.
- [ ] Monitor the first `prod02` expansion shards launched on 2026-07-09:
  B `dynamic_rgb/prod02/lr` job `173510` on `server13` passed render canary
  and entered collection for 170 RGB episodes; C
  `frozen_dp_dynamic/prod02/lr` job `173512` and D
  `future_teacher/prod02/lr` job `173513` are queued as tmux-held Slurm
  one-GPU shards. These are new data-generation shards, not Cosmos training
  evidence and not final method evidence.
  - 2026-07-09 23:14: D `future_teacher/prod02/lr` first attempt job
    `173513` on `server27` failed the render canary with exit code 124 before
    collection. The failed active attempt and log were archived under
    `experiments/legacy/01_dataset/20260709_d_prod02_lr_server27_canary_timeout/`.
    The expansion launcher now excludes `server27` by default for `prod02`,
    and D `future_teacher/prod02/lr` was relaunched as job `173524`.
  - 2026-07-09 23:16 status: B `dynamic_rgb/prod02/lr` job `173510` and C
    `frozen_dp_dynamic/prod02/lr` job `173512` are collecting on `server13`;
    D `future_teacher/prod02/lr` job `173524` is queued. Current partial file
    counts are B 28 videos, C 2 videos, D 0 videos.
  - 2026-07-09 23:31 status: all 18 `prod02` expansion shards are submitted
    for B/C/D across `lr`, `fb`, `reverse`, `stop`, `sine`, and `cont`.
    Running on `server13`: B `lr` job `173510`, B `fb` job `173543`,
    B `reverse` job `173547`, C `lr` job `173512`, C `fb` job `173544`,
    C `reverse` job `173548`, D `lr` job `173524`, and D `fb` job `173546`.
    Queued: B `stop` `173551`, B `sine` `173554`, B `cont` `173558`,
    C `stop` `173552`, C `sine` `173555`, C `cont` `173559`,
    D `reverse` `173549`, D `stop` `173553`, D `sine` `173556`, and
    D `cont` `173563`. Partial counts at this checkpoint: B `lr` 85/170,
    B `fb` 2/170, B `reverse` 2/165, C `lr` 29/84, D `lr` 54/84,
    D `fb` 2/84. No `summary.json` is present yet, so no shard is complete.
    C success remains an outcome label only and is not a rejection condition.
  - 2026-07-09 23:47 status: D `future_teacher/prod02/lr` completed with
    84/84 videos, 336 review frames, and `summary.json` present. The combined
    review-index status check counted it as the first complete shard after
    verifying its quality summary fields. Total current expansion count is
    375/2000 videos. D `future_teacher/prod02/reverse` job `173549` has
    started on `server13`; nine other submitted shards remain pending.
  - 2026-07-09 23:46-23:50: D `future_teacher/prod02/reverse` first attempt
    job `173549` on `server13` failed before collection:
    `render_canary_exit_code=137` and
    `dataset_smoke_status=blocked_render_canary_failed_no_collection`. The
    stuck srun was cancelled, the stale tmux session was removed after the
    Slurm job left the queue, and the failed active output/log were archived
    under
    `experiments/legacy/01_dataset/20260709_d_prod02_reverse_server13_canary_killed_137/`
    with matching logs under
    `logs/legacy/01_dataset/20260709_d_prod02_reverse_server13_canary_killed_137/`.
    D `future_teacher/prod02/reverse` was relaunched as job `173612`. This
    was treated as a failed canary/no-collection attempt, not as data.
  - 2026-07-09 23:53 status: running shards are still making progress after a
    short suspected-stall check. The combined review-index status check shows
    484/2000 videos and 1932 review frames. Current partial counts: B `lr`
    135/170, B `fb` 54/170, B `reverse` 53/165, C `lr` 55/84, C `fb` 25/84,
    C `reverse` 25/83, D `lr` 84/84 complete, and D `fb` 53/84. D `reverse`
    relaunch job `173612` remains queued; `stop`, `sine`, and `cont` shards
    remain queued for B/C/D.
  - 2026-07-10 00:04 status: combined review-index status check shows
    3/18 shards complete and 696/2000 expansion videos. Newly complete shards:
    B `dynamic_rgb/prod02/lr` 170/170 and D `future_teacher/prod02/fb`
    84/84. D `future_teacher/prod02/lr` was already complete. Current
    in-progress counts: B `fb` 96/170, B `reverse` 95/165, C `lr` 76/84,
    C `fb` 46/84, and C `reverse` 45/83. D `reverse` job `173612` and the
    remaining `stop` / `sine` / `cont` shards are still queued.
  - 2026-07-10 00:08 status: C `frozen_dp_dynamic/prod02/lr` completed with
    84/84 videos and 336 review frames. The combined review-index status
    check now shows 4/18 shards complete and 758/2000 expansion videos.
    Running shards: B `fb` 114/170, B `reverse` 113/165, C `fb` 55/84, and
    C `reverse` 54/83. D `reverse` job `173612` and the remaining queued
    shards are still waiting for resources.
  - 2026-07-10 user requested generation pause and script cleanup. All active
    `dset_*prod` Slurm jobs and `dset_[bcd]_prod02` tmux sessions were
    cancelled/removed, and the interrupted local `sleep 600` monitor process
    was cleared. No active production job remains in `squeue`. The stopped
    `prod02` state is 4/18 complete shards and 794/2000 videos; the combined
    review index correctly refuses to build because the expansion is
    incomplete.
  - 2026-07-10 active script cleanup: archived legacy Oracle / phase03 /
    fix3 / old replay-render entry points under
    `scripts/legacy/20260710_generation_cleanup/`. The current active script
    list is recorded in `scripts/ACTIVE_DATASET_GENERATION.md`. Use only the
    current B/C/D expansion plan/launcher, B/C/D production wrappers,
    B/C/D collectors, render canary, gates, validation, and combined review
    index for this dataset route. Do not use archived scripts for active data
    generation without re-auditing and deliberately moving them back.
- [x] Archive and relaunch C `stop` after `MAX_STEP_DELTA_M=0.0045` still
  failed the continuous-motion validation (`0.004592m > 0.004500m`). The old
  failed attempt is archived under
  `experiments/legacy/01_dataset/20260709_c_stop_step_delta_0045_too_tight/`;
  the active C `stop` shard was relaunched with `MAX_STEP_DELTA_M=0.005`.
- [x] Stop and archive the 2026-07-09 B/C/D dynamic production pass after
  user video review found that C still moved the target/hole too late:
  although `pre_insert_l2` required `inserted=false` at the trigger check, it
  did not require any lead time before the first inserted frame, so a sample
  could still have first motion on the same step as insertion. That is the
  wrong task setting; dynamic target/hole motion must begin before insertion
  while the peg is aligned/approaching the hole so replanning is meaningful.
  The running/pending B/C/D Slurm jobs were canceled, all active B/C/D smoke
  approvals and production shards were moved under
  `experiments/legacy/01_dataset/20260709_bad_dynamic_trigger_after_insertion_redo/`,
  and the active approval gate now points to a new review file
  `experiments/maniskill/runs/01_dataset/review/bcd_smoke_matrix_review_20260709_preinsert_lead8.md`.
  C success/failure remains an outcome label only; success is not rejected.
- [x] Add an explicit pre-insertion timing quality gate to B/C/D collectors.
  Accepted dynamic samples now record `first_motion_step`,
  `first_inserted_step`, `trigger_to_insert_steps`, and
  `min_trigger_to_insert_steps`. With the active default
  `min_trigger_to_insert_steps=8`, any sample whose target/hole motion starts
  too close to, on, or after the first inserted step is skipped and cannot
  enter accepted videos/traces. B/C/D runners and smoke/production launchers
  now default to `motion_trigger_mode=pre_insert_l2`,
  `motion_trigger_threshold_m=0.20`, and `min_trigger_to_insert_steps=8`.
- [x] Regenerate the full B/C/D smoke matrix from scratch under the new
  `preinsert_lead8` gate. The active matrix has 21 summaries and 35 videos:
  B has 14 videos, C has 7 videos, and D has 14 videos. Structure validation
  passed for `status=smoke_complete`, `motion_trigger_mode=pre_insert_l2`,
  `motion_trigger_threshold_m=0.20`, `min_trigger_to_insert_steps=8`,
  `state_intervention=false`, and `snap_or_teleport=false`. Any accepted
  sample with an inserted frame has `trigger_to_insert_steps >= 8`; accepted
  samples with `trigger_to_insert_steps=null` did not contain an inserted
  frame, so they are not post-insertion motion. Review file:
  `experiments/maniskill/runs/01_dataset/review/bcd_smoke_matrix_review_20260709_preinsert_lead8.md`.
- [x] Record explicit user approval for the new `preinsert_lead8` B/C/D smoke
  matrix and open the production gate. Approval was recorded with
  `scripts/world_model/record_dataset_bcd_smoke_review_decision.sh`; the
  review status now reports `all_approved=true` and `goal_blocked=false`.
  Production must keep the approved dynamic timing contract:
  `motion_trigger_mode=pre_insert_l2`, `motion_trigger_threshold_m=0.20`,
  `min_trigger_to_insert_steps=8`, and `MAX_STEP_DELTA_M=0.005`.
- [x] Launch the active B/C/D full production pass after approval as 18
  tmux-held Slurm shards under `prod01/<family>`: B `lr`, `fb`, `reverse`,
  `stop`, `sine`, `cont`; C `lr`, `fb`, `reverse`, `stop`, `sine`, `cont`;
  and D `lr`, `fb`, `reverse`, `stop`, `sine`, `cont`. Current shard targets
  are B 1000 dynamic-observation episodes, C 500 frozen-DP dynamic rollouts,
  and D 500 future-teacher rollouts. C success/failure remains an outcome
  label only and is not a production blocker.
- [x] Diagnose and requeue the first production jobs assigned to `server05`.
  B `cont`, all six C shards, and D `lr` failed in the render canary before
  collection with exit `124` / `137`; no videos or summaries from those
  attempts were accepted. The failed active dirs/logs, plus old pending D
  launch records made without the new node exclusion, were archived under
  `experiments/legacy/01_dataset/20260709_server05_render_canary_fail_requeue/`
  and
  `logs/legacy/01_dataset/20260709_server05_render_canary_fail_requeue/`.
  `server05` is now in the production default exclude list, and B `cont`, all
  six C shards, and all six D shards were relaunched as tmux-held Slurm jobs.
- [x] Diagnose and requeue the next production jobs assigned to `server52`.
  B `cont` and C `lr` / `fb` / `reverse` failed in the render canary before
  collection with exit `124` / `137`; several old pending C/D jobs were also
  cancelled or briefly assigned to the same node while the bad-node diagnosis
  was being applied. No active data from those attempts was accepted. The
  affected dirs/logs were archived under
  `experiments/legacy/01_dataset/20260709_server52_render_canary_fail_requeue/`
  and
  `logs/legacy/01_dataset/20260709_server52_render_canary_fail_requeue/`.
  `server52` is now in the production default exclude list, and B `cont`, all
  six C shards, and all six D shards were relaunched again as tmux-held Slurm
  jobs.
- [x] Clean up the pending-log race after the `server52` requeue. C `sine`,
  C `cont`, and D `lr` had new pending jobs with old `srun_failed=143` lines
  prepended by exiting stale sessions; those pending shards had not collected
  data. They were cancelled, archived under
  `experiments/legacy/01_dataset/20260709_clean_pending_logs_after_requeue_race/`
  with matching logs under
  `logs/legacy/01_dataset/20260709_clean_pending_logs_after_requeue_race/`,
  and relaunched with clean active logs.
- [x] Diagnose and requeue the production jobs assigned to `server51`. C
  `lr` / `fb` / `reverse` / `stop` and D `fb` / `reverse` failed render
  canary with Vulkan `ErrorDeviceLost`; D `stop` passed canary but the node
  was treated as untrusted for production after multiple device-lost failures.
  The affected B `cont`, all six C shards, and all six D shards were archived
  under
  `experiments/legacy/01_dataset/20260709_server51_render_device_lost_requeue/`
  with matching logs under
  `logs/legacy/01_dataset/20260709_server51_render_device_lost_requeue/`.
  `server51` is now in the production default exclude list, and B `cont`,
  all six C shards, and all six D shards were relaunched with clean active
  logs.
- [x] Diagnose and requeue the production jobs assigned to `server18`. C
  `sine` and C `cont` failed the render canary before collection with exit
  `124`; D `lr` and D `fb` failed the render canary with Vulkan
  `ErrorDeviceLost`. The C active directories contained only manifests and no
  accepted data; D `lr` / `fb` also stopped before collection. The failed
  attempts and stale pre-exclusion D pending logs were archived under
  `experiments/legacy/01_dataset/20260709_server18_render_canary_fail_requeue/`
  with matching logs under
  `logs/legacy/01_dataset/20260709_server18_render_canary_fail_requeue/`.
  `server18` is now in the production default exclude list; C `sine` /
  `cont` and all six D shards were relaunched as tmux-held Slurm jobs with
  clean active logs.
- [x] Diagnose and requeue D production jobs assigned to `server23`. D
  `reverse` and D `stop` failed the render canary with Vulkan
  `ErrorDeviceLost`; D `sine` and D `cont` had passed canary and started
  collection on the same node, so their partial videos were treated as
  untrusted. The affected D `reverse` / `stop` / `sine` / `cont` attempts
  were archived under
  `experiments/legacy/01_dataset/20260709_server23_render_device_lost_requeue/`
  with matching logs under
  `logs/legacy/01_dataset/20260709_server23_render_device_lost_requeue/`.
  `server23` is now in the production default exclude list, and those four D
  shards were relaunched with clean active logs.
- [x] Complete and verify ABCD full production after the approved
  `preinsert_lead8` smoke matrix. A static RGB remains ready as 20 shards,
  1000 videos, and 480 review frames. B dynamic observation production is
  complete as six shards / 1000 videos. C frozen-DP dynamic production is
  complete as six shards / 500 videos. D future-teacher production is complete
  as six shards / 500 videos. All B/C/D active production summaries report
  `status=production_complete`, `motion_trigger_mode=pre_insert_l2`,
  `motion_trigger_threshold_m=0.20`, `state_intervention=false`,
  `snap_or_teleport=false`, and pass the per-sample `min_trigger_to_insert`
  gate with no accepted sample moving on or after insertion. B/C/D shard
  indexes were refreshed under
  `experiments/maniskill/data/active/b_dynamic_production/`,
  `experiments/maniskill/data/active/c_frozen_dp_production/`, and
  `experiments/maniskill/data/active/d_future_teacher_production/`.

## Immediate Documentation / Design

- [x] Define dataset classes A-E by training role.
- [x] Define which samples may receive DP BC loss, Cosmos future loss,
  discrepancy loss, negative labels, and adapter loss.
- [x] Define target-frame manifest fields.
- [x] Update AGENTS with finalized data-class and smoke-first rules.
- [x] Record initial target data sizes and production order in
  `docs/dataset_collection_targets.md`.
- [x] Add dataset manifest schema in `docs/dataset_manifest_schema.md`.
- [x] Add read-only run manifest validator:
  `scripts/world_model/validate_dataset_run_manifest.sh`.
- [x] Add read-only smoke status helper:
  `scripts/world_model/dataset_smoke_status.sh`.
- [x] Add read-only overall dataset goal status helper:
  `scripts/world_model/dataset_goal_status.sh`.
- [x] Add smoke human-review package helper:
  `scripts/world_model/prepare_dataset_smoke_review.sh`.
- [x] Add smoke approval guard:
  `scripts/world_model/require_dataset_smoke_approved.sh`.
- [x] Add smoke review status helper:
  `scripts/world_model/dataset_review_status.sh`.
- [x] Add class-level smoke review status / approval helpers:
  `scripts/world_model/dataset_class_review_status.sh` and
  `scripts/world_model/require_dataset_class_smoke_approved.sh`.
- [x] Add explicit smoke review decision recorder:
  `scripts/world_model/record_dataset_smoke_review_decision.sh`.
- [x] Create active data registry for official static data and legacy 733
  references under `experiments/maniskill/data/active/`.
- [x] Register old 1000-sample dynamic RGBD data as limited B bootstrap under
  `experiments/maniskill/data/active/b_dynamic_legacy_bootstrap/`.
- [x] Generate B bootstrap path indexes and manifest:
  `rgbd_h5_paths.txt`, `mp4_paths.txt`, `image_paths.txt`,
  `scenario_counts.txt`, `manifest.txt`.
- [x] Add B bootstrap status helper:
  `scripts/world_model/dataset_bootstrap_status.sh`.
- [x] Add training input status / role validation helper:
  `scripts/world_model/dataset_training_inputs_status.sh`.
- [x] Add training input readiness guard:
  `scripts/world_model/require_dataset_training_inputs_ready.sh`.
- [x] Add production registry index builder and status helper for B/C/D/E
  training entry files:
  `scripts/world_model/build_dataset_production_index.sh` and
  `scripts/world_model/dataset_production_index_status.sh`.
- [x] Update `full_joint` training input guard so production data must pass
  both run validation and registry index validation before training can start.
- [x] Harden production index builder so it refuses to index if the source
  `summary.json` / `manifest.txt` do not match the expected stage, class,
  run group, run name, output directory, method/teacher evidence flags, and
  no-state-intervention contract.
- [x] Add B bootstrap index builder:
  `scripts/world_model/build_dataset_bootstrap_index.sh`.
- [x] Generate B bootstrap `samples.tsv` / JSONL and 900/100 train/val split.
- [x] Record B bootstrap training-side entry files:
  `train_samples.jsonl` and `val_samples.jsonl`, with allowed/disallowed loss
  roles carried per row.
- [x] Verify B bootstrap train/val JSONL role fields:
  `dataset_class`, `dataset_role`, `split`, `allowed_losses`,
  `disallowed_losses`, `method_evidence_allowed=false`,
  `positive_dp_bc_allowed=false`, and `replaces_new_production=false`.
- [x] Add dynamic dataset collection plan for classes B/C/D/E:
  `docs/dynamic_dataset_collection_plan.md`.
- [x] Add sample review template:
  `docs/dataset_sample_review_template.md`.
- [x] Add dataset review policy:
  `docs/dataset_review_policy.md`.
- [x] Add next-stage readiness guard:
  `scripts/world_model/require_dataset_stage_ready.sh`.
- [x] Add A full static RGB readiness guard:
  `scripts/world_model/require_dataset_static_full_ready.sh`.
- [x] Add next-stage status helper:
  `scripts/world_model/dataset_next_stage_status.sh`.
- [x] Add runner source audit guard:
  `scripts/world_model/audit_dataset_runner_source.sh`.
- [x] Add collector source audit guard and wire it into stage readiness /
  status reporting:
  `scripts/world_model/audit_dataset_collector_source.sh`.
- [x] Add shared runtime-context guard for future in-allocation runners:
  `scripts/world_model/require_dataset_runtime_context.sh`.
- [x] Add post-approval command/status helper:
  `scripts/world_model/dataset_post_approval_plan.sh`.
- [x] Add guarded B/C/D/E launch entrypoints that refuse before Slurm until
  Stage 1 review and runner checks pass.
- [x] Replace B/C/D/E placeholder launchers with a shared guarded Slurm/tmux
  smoke launcher that will submit only after Stage 1 approval and runner
  source audit pass:
  `scripts/slurm/launch_dataset_stage_smoke_tmux_common.sh`.
- [x] Verify B/C/D/E launchers currently exit at the Stage 1 review gate with
  `reason=stage1_smoke_review_pending`, create no tmux session, submit no
  Slurm job, and create no B/C/D/E run or log directory.
- [x] Verify class-level review gates report A static smoke as pending human
  approval and B/C/D/E smoke gates as missing `summary.json`.
- [x] Update `scripts/world_model/require_dataset_stage_ready.sh` so B/C/D/E
  smoke readiness also requires A full `static_rgb/full01` to exist and pass
  manifest validation after Stage 1 approval.
- [x] Verify `scripts/world_model/require_dataset_static_full_ready.sh`
  currently rejects with `reason=summary_missing`.
- [x] Verify `scripts/world_model/require_dataset_runtime_context.sh` refuses
  login-node execution with `reason=not_inside_compute_srun_step`.
- [x] Update `scripts/world_model/audit_dataset_runner_source.sh` so future
  runners may satisfy login/output/log/render requirements by sourcing
  `require_dataset_runtime_context.sh`.
- [x] Validate `smoke05` run manifest and add non-destructive
  `manifest_corrections.txt` for schema fields missing from the older manifest
  (`source_paths`, `disallowed_losses`).
- [x] Verify the post-approval plan is read-only and reports B/C/D/E command,
  common launcher, output/log path, and readiness gate without creating Slurm
  jobs or directories.
- [x] Update `scripts/world_model/dataset_post_approval_plan.sh` so B/C/D/E
  readiness uses `require_dataset_stage_ready.sh` and reports the exact gate
  reason, not only `runner_missing`.
- [x] Add post-approval runbook:
  `docs/dataset_after_approval_runbook.md`.
- [x] Record missing B/C/D/E in-allocation runners in
  `docs/dataset_runner_implementation_gaps.md`.
- [x] Record that old dynamic generator pycache exists but active source files
  are missing, so pycache must not be wrapped as an active runner.
- [x] Locate the deleted dynamic generator source in git history and record
  reuse risks in `docs/legacy_dynamic_source_recovery.md`.
- [x] Add source recovery status helper:
  `scripts/world_model/dataset_source_recovery_status.sh`.
- [x] Record resource monitoring state on 2026-07-07: no Reflex Slurm job is
  currently queued/running; because `static_rgb/smoke05` is still missing
  human approval, GPU allocation must not be requested yet. After the gate
  opens, launch through the guarded tmux Slurm entrypoint with `1 GPU`, reduce
  CPU / memory / walltime first if scheduling is slow, and keep monitoring a
  valid pending request instead of repeatedly canceling it.
- [x] Record 2026-07-07 06:46 CST monitor state: no Reflex Slurm job is queued
  or running, `static_rgb/smoke05` still lacks
  `human_review_approved.txt`, and the next compute action remains blocked by
  human review rather than scheduler availability.
- [x] Record 2026-07-07 06:49 CST Slurm check: jobs `169317 any_nofab` and
  `169319 g1_render` are under `WorkDir=/public/home/yanhongru/Curiosity`, not
  the active Reflex workspace, so they are not counted as Reflex progress or
  Reflex resource requests.
- [x] Add read-only Slurm workdir classifier:
  `scripts/world_model/dataset_slurm_status.sh`, and wire it into
  `scripts/world_model/dataset_goal_status.sh` so future status reports do
  not confuse Curiosity jobs with Reflex dataset jobs.
- [x] Record 2026-07-07 06:51 CST monitor state: `static_rgb/smoke05` still
  lacks `human_review_approved.txt`; `dataset_slurm_status.sh` reports
  `total_user_job_count=0`, `reflex_job_count=0`, and
  `non_reflex_job_count=0`.
- [x] Record 2026-07-07 06:52 CST monitor state: `static_rgb/smoke05` still
  lacks `human_review_approved.txt`; the only visible Slurm job is
  `169324 g1_fallback`, classified as `non_reflex` with
  `WorkDir=/public/home/yanhongru/Curiosity`.
- [x] Add run-local target-blocked artifact:
  `experiments/maniskill/runs/01_dataset/static_rgb/smoke05/goal_blocked.md`.
- [x] Record explicit user approval for `static_rgb/smoke05` with
  `scripts/world_model/record_dataset_smoke_review_decision.sh`.
- [x] Update active RGB defaults so A static replay and B/C/D/E collectors use
  `30 FPS`.
- [x] Update review policy so B/C/D/E smoke artifacts are generated as a
  batch for one combined human review when guards allow it, while production
  remains class-gated.
- [x] Add batch smoke launcher:
  `scripts/slurm/launch_dataset_batch_smoke_tmux.sh`, which launches available
  B/C/D/E smoke stages after A full readiness and skips stages whose prereqs
  are still closed.
- [x] Fix `scripts/world_model/dataset_slurm_status.sh` so it reports
  `ReqNodeList`, `ExcNodeList`, and `SchedNodeList` separately instead of
  mistaking excluded nodes for active candidate nodes.
- [x] Record `static_rgb/full01` attempt on `server36`: render canary timed
  out with exit `124`, no replay/video/summary produced; archived to
  `experiments/legacy/01_dataset/static_rgb/full01/server36_canary_timeout/`.
- [x] Relaunch `static_rgb/full01` with `30 FPS`, `1 GPU`, and
  `EXCLUDE_NODES=server36,server39,server43` as Slurm job `169686`.
- [x] Cancel pending-only job `169686` after scheduler diagnostics showed a
  better lower-resource request: `1 CPU / 8G / 4h` could start earlier on
  previously successful `server44`, while `169686` moved to risk node
  `server59`.
- [x] Archive pending-only `169686` log to
  `logs/legacy/01_dataset/static_rgb/full01/server59_pending_superseded.log`
  and relaunch `static_rgb/full01` with `30 FPS`, `1 GPU`, `1 CPU`, `8G`,
  `4h`, and `EXCLUDE_NODES=server36,server39,server43`.
- [x] Cancel pending-only job `169701` after scheduler diagnostics showed the
  no-nodelist request moving to risk node `server63`, while explicit
  `NODELIST=server44` had a concrete earlier test-only slot.
- [x] Archive pending-only `169701` log to
  `logs/legacy/01_dataset/static_rgb/full01/server63_pending_superseded.log`
  and relaunch `static_rgb/full01` with `NODELIST=server44`, `30 FPS`,
  `1 GPU`, `1 CPU`, `8G`, and `4h`.
- [x] Record 2026-07-07 12:02 CST status: `static_rgb/full01` job `169704`
  remains pending with `NODELIST=server44`; latest test-only estimates
  `server44` around `2026-07-07T12:32:36`, while `server02` is much later.
- [x] Record 2026-07-07 12:07 CST status: `static_rgb/full01` job `169704`
  remains pending with `NODELIST=server44`; Slurm reason changed to
  `Priority`, so this is no longer classified as a node-state hard failure.
- [x] Record 2026-07-07 14:02 CST diagnosis: `server44` is
  `IDLE+PLANNED`, so job `169704` remained pending without StartTime; `server35`
  is `MIXED`, not in the current device-lost list, and test-only gives the
  same earliest slot.
- [x] Cancel pending-only `169704`, archive its log to
  `logs/legacy/01_dataset/static_rgb/full01/server44_planned_superseded.log`,
  and relaunch `static_rgb/full01` on `NODELIST=server35` with `30 FPS`,
  `1 GPU`, `1 CPU`, `8G`, and `4h`.
- [x] Record 2026-07-07 14:08 CST status: `static_rgb/full01` job `169899`
  is pending with `ReqNodeList=server35`, `SchedNodeList=server35`, and reason
  `Priority`; keep the valid queued request.
- [x] Cancel pending-only job `169899` after `scontrol` showed stale
  `StartTime=2026-07-14T14:00:00` while same-config test-only showed
  `2026-07-07T15:23:18`; archive log to
  `logs/legacy/01_dataset/static_rgb/full01/server35_stale_starttime_superseded.log`
  and relaunch with the same `server35`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`,
  `4h` request.
- [x] Cancel pending-only job `169914` after the same stale StartTime pattern
  repeated for explicit `NODELIST=server35`; archive log to
  `logs/legacy/01_dataset/static_rgb/full01/server35_stale_starttime2_superseded.log`.
- [x] Relaunch `static_rgb/full01` without an explicit nodelist, but excluding
  known failed/risk nodes `server36,server39,server43,server44,server57,server59,server63`,
  because no-nodelist test-only selects `server35` for today while explicit
  nodelist jobs are being assigned stale one-week StartTimes.
- [x] Add compute-node shard builder
  `scripts/world_model/make_static_replay_shard.py` so short A static RGB
  production shards can render disjoint official-demo episode ranges instead
  of repeatedly replaying the first `COUNT` episodes.
- [x] Cancel pending-only full-run job `169921` and archive its log to
  `logs/legacy/01_dataset/static_rgb/full01/full01_superseded_by_shards.log`
  after the long 4h single-run strategy kept receiving stale/delayed scheduler
  plans.
- [x] Launch A static RGB shard `static_rgb/full_s00` with
  `EPISODE_START=0`, `COUNT=100`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:45:00`.
- [x] Cancel pending-only shard job `169940` after `scontrol` put it at
  `2026-07-07T21:00:00` while same-config test-only showed
  `2026-07-07T15:46:05`; archive log to
  `logs/legacy/01_dataset/static_rgb/full_s00/stale_2100_superseded.log` and
  relaunch `static_rgb/full_s00` with the same shard settings.
- [x] Cancel later pending-only `full_s00` attempts after Slurm repeatedly
  moved the 100-episode shard to late or bad-node windows; archive them under
  `logs/legacy/01_dataset/static_rgb/full_s00/` and
  `experiments/legacy/01_dataset/static_rgb/full_s00/`.
- [x] Split A static RGB production into 50-episode shards when test-only
  showed shorter walltime could start earlier without changing the scientific
  content.
- [x] Add shard-aware A full readiness:
  `scripts/world_model/dataset_static_full_shards_status.sh` plus updated
  `scripts/world_model/require_dataset_static_full_ready.sh`. The gate accepts
  legacy `static_rgb/full01` or ready `static_rgb/full_s*` shards totaling
  1000 episodes, with per-shard manifest validation and visual artifacts.
- [x] Update training/status guards so `full_joint` readiness uses the
  shard-aware A full RGB gate rather than only
  `static_rgb/full01/summary.json`.
- [x] Launch `static_rgb/full_s00a` as Slurm job `169991` with
  `EPISODE_START=0`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`; current scheduler window is `2026-07-07T16:55:03` on
  `server30`.
- [x] Record `static_rgb/full_s00a` job `169991` on `server30`: render canary
  hung at `render_gym_start`, timed out after `3m` with exit `124`, and
  produced no replay/video/summary. Archive to
  `experiments/legacy/01_dataset/static_rgb/full_s00a/server30_canary_timeout/`
  and `logs/legacy/01_dataset/static_rgb/full_s00a/server30_canary_timeout.log`.
- [x] Relaunch `static_rgb/full_s00a` as job `170001` with the same
  50-episode, 30 FPS, 1 GPU / 1 CPU / 8G / 30m settings and
  `server30` added to the current exclude list.
- [ ] Monitor `static_rgb/full_s00a` through render canary, replay, summary,
  and manifest validation; archive immediately if canary or visual artifacts
  fail.
- [ ] If `full_s00a` succeeds, continue sequential 50-episode A static RGB
  shards until the shard-aware full gate reaches 1000 episodes.
- [x] Validate `static_rgb/full_s00a` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=0`, node `server20`.
- [x] Confirm shard-aware gate counts `static_rgb/full_s00a` as ready and
  reports `total_count=50`, `target_count=1000`, so B/C/D/E remain blocked
  until the full A static RGB set is complete.
- [x] Launch `static_rgb/full_s00b` as job `170004` with
  `EPISODE_START=50`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Validate `static_rgb/full_s00b` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=50`, node `server20`.
- [x] Confirm shard-aware gate reports `total_count=100`,
  `total_video_count=100`, `ready_shard_count=2`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s01a` as job `170020` with
  `EPISODE_START=100`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Cancel pending-only `full_s01a` job `170020` after `squeue --start`
  assigned it to `2026-07-07T22:22:50` on `server46`, while same-config
  test-only showed `2026-07-07T18:16:05`; archive log to
  `logs/legacy/01_dataset/static_rgb/full_s01a/stale_2222_superseded.log`.
- [x] Relaunch `static_rgb/full_s01a` as job `170027` with the same
  50-episode, 30 FPS settings.
- [x] Cancel pending-only `full_s01a` job `170027` after the scheduler again
  assigned `2026-07-07T22:22:50` on `server46`; same-config test-only with
  `server46` excluded showed `2026-07-07T18:16:05`.
- [x] Relaunch `static_rgb/full_s01a` as job `170032` with `server46` added to
  the exclude list.
- [x] Validate `static_rgb/full_s01a` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=100`, node `server20`.
- [x] Confirm shard-aware gate reports `total_count=150`,
  `total_video_count=150`, `ready_shard_count=3`, and
  `invalid_shard_count=0`.
- [x] Add read-only next-shard helper:
  `scripts/world_model/dataset_static_full_next_shard.sh`. It reports
  `completed_count`, `remaining_count`, `next_run_name`,
  `next_episode_start`, and `next_count` so A static RGB shards can continue
  without hand-calculation errors.
- [x] Add guarded next-shard launcher:
  `scripts/slurm/launch_dataset_static_rgb_next_shard_tmux.sh`. It supports
  `--dry-run`, uses the read-only next-shard helper, keeps 30 FPS / 1 GPU /
  1 CPU / 8G defaults, and applies the current render-risk exclude list.
- [x] Launch `static_rgb/full_s01b` as job `170091` with
  `EPISODE_START=150`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`; current scheduled node is `server58` with start window
  `2026-07-07T19:10:11`.
- [x] Cancel pending-only `full_s01b` job `170091` after the scheduler moved
  it to `2026-07-07T20:13:09` on `server10`, while same-config test-only
  showed `2026-07-07T19:17:11` on `server58`; archive log to
  `logs/legacy/01_dataset/static_rgb/full_s01b/stale_2013_superseded.log`.
- [x] Relaunch `static_rgb/full_s01b` through
  `scripts/slurm/launch_dataset_static_rgb_next_shard_tmux.sh` as job
  `170106`.
- [x] Cancel pending-only `full_s01b` job `170106` after it repeated the
  `2026-07-07T20:13:09` `server10` assignment while same-config test-only
  still showed `2026-07-07T19:17:11` on `server58`; archive log to
  `logs/legacy/01_dataset/static_rgb/full_s01b/stale_2013_server10_superseded.log`.
- [x] Relaunch `static_rgb/full_s01b` as job `170110` with `server10` added
  to the current exclude list.
- [x] Cancel pending-only `full_s01b` job `170110` after it moved to
  `2026-07-07T20:14:09` on `server35`, while test-only with `server35`
  excluded still showed `2026-07-07T19:17:11` on `server58`; archive log to
  `logs/legacy/01_dataset/static_rgb/full_s01b/stale_2014_server35_superseded.log`.
- [x] Relaunch `static_rgb/full_s01b` as job `170119` with `server35` also
  excluded. If this job also receives a late moving window, keep the valid
  request instead of continuing blind scheduler churn.
- [x] Record `static_rgb/full_s01b` job `170119` scheduler status:
  `StartTime=2026-07-07T20:16:09`, `SchedNodeList=server02`; keep this valid
  low-resource request instead of continuing repeated pending-only relaunches.
- [x] Cancel pending-only `full_s01b` job `170119` after a later diagnostic
  showed explicit `NODELIST=server58` could start at `2026-07-07T17:44:01`
  while `170119` had drifted to `2026-07-07T20:38:11`; archive log to
  `logs/legacy/01_dataset/static_rgb/full_s01b/stale_2038_superseded.log`.
- [x] Relaunch `static_rgb/full_s01b` as job `170132` with explicit
  `NODELIST=server58`, after checking `server58` was `MIXED` with only
  `6/8` GPUs allocated.
- [x] Record `static_rgb/full_s01b` job `170132` scheduler mismatch:
  `scontrol` reports `StartTime=Unknown` for explicit `server58`, while
  same-node test-only reports `2026-07-07T19:30:11`. Keep the valid queued
  request instead of continuing scheduler churn.
- [x] Record `static_rgb/full_s01b` job `170132` on `server58`: render canary
  failed before replay with Vulkan `vk::Device::waitForFences:
  ErrorDeviceLost` / exit `134`; no video or summary was produced. Archive
  run/log to `experiments/legacy/01_dataset/static_rgb/full_s01b/server58_device_lost/`
  and `logs/legacy/01_dataset/static_rgb/full_s01b/server58_device_lost.log`.
- [x] Add `server58` to active render-risk device-lost reporting and the
  next-shard default exclude list.
- [x] Relaunch `static_rgb/full_s01b` as job `170171` with updated default
  exclude list including `server58`.
- [x] Validate `static_rgb/full_s01b` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=150`, node `server27`.
- [x] Confirm shard-aware gate reports `total_count=200`,
  `total_video_count=200`, `ready_shard_count=4`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s02a` as job `170183` with
  `EPISODE_START=200`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Validate `static_rgb/full_s02a` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=200`, node `server27`.
- [x] Confirm shard-aware gate reports `total_count=250`,
  `total_video_count=250`, `ready_shard_count=5`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s02b` as job `170191` with
  `EPISODE_START=250`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Record `static_rgb/full_s02b` job `170191` on `server56`: render canary
  hung at `render_gym_start`, timed out after `3m` with exit `124`, and
  produced no replay/video/summary. Archive run/log to
  `experiments/legacy/01_dataset/static_rgb/full_s02b/server56_canary_timeout/`
  and `logs/legacy/01_dataset/static_rgb/full_s02b/server56_canary_timeout.log`.
- [x] Add `server56` to active render-risk timeout reporting and the
  next-shard default exclude list.
- [x] Relaunch `static_rgb/full_s02b` as job `170201` with updated default
  exclude list including `server56` and `server58`.
- [x] Record `static_rgb/full_s02b` job `170201` on `server28`: render canary
  hung at `render_gym_start`, timed out after `3m` with exit `124`, and
  produced no replay/video/summary. Archive run/log to
  `experiments/legacy/01_dataset/static_rgb/full_s02b/server28_canary_timeout/`
  and `logs/legacy/01_dataset/static_rgb/full_s02b/server28_canary_timeout.log`.
- [x] Add `server28` to active render-risk timeout reporting and the
  next-shard default exclude list.
- [x] Relaunch `static_rgb/full_s02b` as job `170215` with updated default
  exclude list including `server28`, `server56`, and `server58`; request is
  `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and `00:30:00`.
- [x] Record `static_rgb/full_s02b` job `170215` scheduler status:
  `Reason=Priority`, `SchedNodeList=server02`, and
  `StartTime=2026-07-07T21:00:00`; keep this valid queued request rather than
  relaunching without a concrete invalid-resource or bad-node diagnosis.
- [x] Validate `static_rgb/full_s02b` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=250`, node `server02`.
- [x] Confirm shard-aware gate reports `total_count=300`,
  `total_video_count=300`, `ready_shard_count=6`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s03a` as job `170299` with
  `EPISODE_START=300`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Cancel pending-only `full_s03a` job `170299` after scheduler placed it
  on `server60` with start estimate `2026-07-08T02:14:47`, while a same
  resource `srun --test-only` on known-good `server02` estimated
  `2026-07-07T22:01:17`; archive the empty attempt to
  `logs/legacy/01_dataset/static_rgb/full_s03a/stale_server60_0214_superseded.log`
  and
  `experiments/legacy/01_dataset/static_rgb/full_s03a/stale_server60_0214_superseded/`.
- [x] Relaunch `static_rgb/full_s03a` as job `170307` with explicit
  `NODELIST=server02`, `EPISODE_START=300`, `COUNT=50`, `30 FPS`, `1 GPU`,
  `1 CPU`, `8G`, and `00:30:00`.
- [x] Record `static_rgb/full_s03a` job `170307` scheduler status:
  `Reason=Priority`, `StartTime=Unknown`; same-resource `srun --test-only`
  for `server02` estimates `2026-07-07T22:01:17`, so keep this valid queued
  request.
- [x] Validate `static_rgb/full_s03a` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=300`, node `server02`.
- [x] Confirm shard-aware gate reports `total_count=350`,
  `total_video_count=350`, `ready_shard_count=7`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s03b` as job `170367` with
  `EPISODE_START=350`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Record `static_rgb/full_s03b` job `170367` scheduler status:
  `Reason=Priority`, `SchedNodeList=server60`, and
  `StartTime=2026-07-07T21:18:19`; keep this valid queued request because
  `server60` has no current recorded RGB canary failure and this estimate is
  earlier than the known-good `server02` test-only estimate.
- [x] Validate `static_rgb/full_s03b` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=350`, node `server60`.
- [x] Confirm shard-aware gate reports `total_count=400`,
  `total_video_count=400`, `ready_shard_count=8`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s04a` as job `170383` with
  `EPISODE_START=400`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`,
  `00:30:00`, node `server60`.
- [x] Validate `static_rgb/full_s04a` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=400`, node `server60`.
- [x] Confirm shard-aware gate reports `total_count=450`,
  `total_video_count=450`, `ready_shard_count=9`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s04b` as job `170389` with
  `EPISODE_START=450`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`,
  `00:30:00`, node `server60`.
- [x] Validate `static_rgb/full_s04b` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=450`, node `server60`.
- [x] Confirm shard-aware gate reports `total_count=500`,
  `total_video_count=500`, `ready_shard_count=10`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s05a` as job `170406` with
  `EPISODE_START=500`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Validate `static_rgb/full_s05a` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=500`, node `server60`.
- [x] Confirm shard-aware gate reports `total_count=550`,
  `total_video_count=550`, `ready_shard_count=11`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s05b` as job `170413` with
  `EPISODE_START=550`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Validate `static_rgb/full_s05b` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=550`, node `server02`.
- [x] Confirm shard-aware gate reports `total_count=600`,
  `total_video_count=600`, `ready_shard_count=12`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s06a` as job `170420` with
  `EPISODE_START=600`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Validate `static_rgb/full_s06a` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=600`, node `server02`.
- [x] Confirm shard-aware gate reports `total_count=650`,
  `total_video_count=650`, `ready_shard_count=13`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s06b` as job `170427` with
  `EPISODE_START=650`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Validate `static_rgb/full_s06b` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=650`, node `server60`.
- [x] Confirm shard-aware gate reports `total_count=700`,
  `total_video_count=700`, `ready_shard_count=14`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s07a` as job `170432` with
  `EPISODE_START=700`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Validate `static_rgb/full_s07a` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=700`, node `server60`.
- [x] Confirm shard-aware gate reports `total_count=750`,
  `total_video_count=750`, `ready_shard_count=15`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s07b` as job `170446` with
  `EPISODE_START=750`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Validate `static_rgb/full_s07b` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=750`, node `server60`.
- [x] Record transient shard-status anomaly: one aggregate pass reported
  missing `review/` for `full_s02b`, `full_s03a`, and `full_s05b`, but direct
  file inspection found each had 50 mp4 files and 24 review png frames; a
  repeated aggregate pass returned `ready_shard_count=16` and
  `invalid_shard_count=0`, so no shard was rerun.
- [x] Confirm shard-aware gate reports `total_count=800`,
  `total_video_count=800`, `ready_shard_count=16`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s08a` as job `170460` with
  `EPISODE_START=800`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Validate `static_rgb/full_s08a` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=800`, node `server60`.
- [x] Confirm shard-aware gate reports `total_count=850`,
  `total_video_count=850`, `ready_shard_count=17`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s08b` as job `170470` with
  `EPISODE_START=850`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Validate `static_rgb/full_s08b` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=850`, node `server60`.
- [x] Confirm shard-aware gate reports `total_count=900`,
  `total_video_count=900`, `ready_shard_count=18`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s09a` as job `170483` with
  `EPISODE_START=900`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`,
  `00:30:00`, node `server60`.
- [x] Validate `static_rgb/full_s09a` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=900`, node `server60`.
- [x] Confirm shard-aware gate reports `total_count=950`,
  `total_video_count=950`, `ready_shard_count=19`, and
  `invalid_shard_count=0`.
- [x] Launch `static_rgb/full_s09b` as job `170498` with
  `EPISODE_START=950`, `COUNT=50`, `30 FPS`, `1 GPU`, `1 CPU`, `8G`, and
  `00:30:00`.
- [x] Validate `static_rgb/full_s09b` manifest: `dataset_run_manifest_valid=true`,
  `video_count=50`, `review_frame_count=24`, `video_fps=30`,
  `count=50`, `episode_start=950`, node `server60`.
- [x] Complete A static RGB full shard set: shard-aware gate reports
  `ready_shard_count=20`, `invalid_shard_count=0`, `total_count=1000`,
  `total_video_count=1000`, `total_review_frame_count=480`, and
  `dataset_static_full_shards_ready=true`.
- [x] Confirm `scripts/world_model/require_dataset_static_full_ready.sh`
  returns `dataset_static_full_ready=true` with `ready_source=shards`.
- [x] Confirm `scripts/world_model/dataset_static_full_next_shard.sh`
  returns `next_shard_needed=false` and `dataset_static_full_complete=true`.
- [x] Confirm full-joint training input guard is still intentionally blocked:
  A static RGB is ready, but B/C/D/E production datasets and active indexes are
  not ready yet, so `require_dataset_training_inputs_ready.sh` exits with
  `dataset_training_inputs_ready=false` / `reason=full_joint_inputs_incomplete`.
- [x] Run batch smoke dry-run after A static RGB full completion: B dynamic
  RGB smoke, C frozen-DP dynamic smoke, and D future-frame teacher smoke are
  ready to launch; E Cosmos-predicted smoke is correctly blocked by missing B
  production, D production, and Cosmos/readout held-out validation.
- [x] Launch batch smoke after A static RGB full completion:
  B dynamic RGB smoke `smoke01`, C frozen-DP dynamic smoke `smoke01`, and D
  future-frame teacher smoke `smoke01` were submitted; E Cosmos-predicted
  smoke was correctly skipped with `reason=e_prereqs_not_ready`.
- [x] Verify E Cosmos-predicted cooperation smoke runner and collector source
  audits pass:
  `scripts/slurm/run_dataset_cosmos_predicted_coop_smoke_in_allocation.sh`
  and `scripts/world_model/collect_cosmos_predicted_coop_smoke.py`.
- [x] Verify E is still correctly blocked by prereqs: B production, D
  production, and held-out Cosmos/readout validation do not exist yet.

## Stage 1: Static Expert Set

- [x] Create Stage 1 smoke launcher:
  `scripts/slurm/launch_dataset_static_rgb_smoke_tmux.sh`.
- [x] Create Stage 1 allocation-only runner:
  `scripts/slurm/run_dataset_static_rgb_smoke_in_allocation.sh`.
- [x] Attempt Stage 1 smoke launch with `salloc --immediate=60`; no node was
  allocated and no render artifacts were produced.
- [x] Retry Stage 1 smoke launch with lighter resources
  `2 CPU / 12G / 00:20:00`; no node was allocated and no render artifacts were
  produced.
- [x] Add queued tmux-held smoke option for when immediate allocation is not
  available.
- [x] Add guarded full static RGB launcher that refuses production until smoke
  approval: `scripts/slurm/launch_dataset_static_rgb_full_tmux.sh`.
- [x] Queue Stage 1 smoke render as tmux-held Slurm job `168507`
  (`dset_rgb_smoke`), pending priority with estimated start
  `2026-07-06T21:38:25` on `server44`.
- [x] Record 2026-07-06 20:20 CST status: job `168507` still pending, no
  smoke artifacts yet.
- [x] Record 2026-07-06 20:23 CST status: job `168507` still pending, no
  output directory or review artifacts yet.
- [x] Record user-approved resource strategy: dataset smoke uses 1 GPU, can
  reduce CPU/memory, can queue longer, and can try previously bad render nodes
  as smoke-only.
- [x] Add render risk status helper:
  `scripts/world_model/dataset_render_risk_status.sh`, summarizing current
  smoke render evidence, required Vulkan/HDF5/canary/minimal-shader settings,
  known current smoke risk nodes, and production guidance.
- [x] Record 2026-07-06 21:23 CST status: job `168507` still pending,
  scheduler estimate `2026-07-06T23:15:35` on `server02`.
- [x] Record 2026-07-06 21:26 CST status: job `168507` cancelled before node
  assignment; no render started.
- [x] Relaunch lower-resource smoke as Slurm job `168562`
  (`1 GPU / 1 CPU / 8G / 00:15:00`) with no node exclusions.
- [x] Record 2026-07-06 21:34 CST status: job `168562` revoked before node
  assignment; no render started.
- [x] Add direct `srun` tmux smoke launcher:
  `scripts/slurm/launch_dataset_static_rgb_smoke_srun_tmux.sh`.
- [x] Record 2026-07-06 21:39 CST status: direct `srun` job `168581`
  cancelled before resource assignment; no render started.
- [x] Relaunch reduced-resource direct `srun` smoke on `cpu` partition as job
  `168586` (`1 GPU / 1 CPU / 8G / 00:15:00`, no node exclusions), estimated
  start `2026-07-06T23:03:41` on `server44`.
- [x] Record 2026-07-06 22:03 CST status: job `168586` cancelled before node
  assignment; no render started and no artifacts were produced.
- [x] Relaunch close-to-start-window reduced-resource smoke as Slurm job
  `168603` after `--test-only` moved to `2026-07-06T22:18:18` on `server44`.
- [x] Record 2026-07-06 22:36 CST result: job `168603` started on `server39`
  but failed in RGB rendering with Vulkan `ErrorDeviceLost` / exit `134`; no
  video, summary, or review frame was produced.
- [x] Archive failed `smoke01` run/log under external archive and keep active
  experiment folder clean.
- [x] Relaunch active static RGB smoke as `smoke02`, Slurm job `168635`,
  excluding `server39` while keeping `1 GPU / 1 CPU / 8G / 00:15:00`.
- [x] Record 2026-07-06 22:47 CST result: `smoke02` incorrectly ran on
  `server39` and hit the same Vulkan `ErrorDeviceLost`.
- [x] Fix `scripts/slurm/launch_dataset_static_rgb_smoke_srun_tmux.sh` so
  `--exclude` is passed to Slurm before the executable script.
- [x] Archive failed `smoke02` run/log under the external archive.
- [x] Relaunch active static RGB smoke as `smoke03`, Slurm job `168659`, with
  corrected `--exclude=server39`.
- [x] Add `NODELIST` support to
  `scripts/slurm/launch_dataset_static_rgb_smoke_srun_tmux.sh` for explicit
  node targeting when Slurm test-only identifies a better slot.
- [x] Cancel superseded pending-only `smoke03` job `168659` after its estimate
  slipped, and archive its log.
- [x] Relaunch `smoke03` as Slurm job `168691` with
  `NODELIST=server02` and `EXCLUDE_NODES=server39`.
- [x] Cancel superseded pending-only `smoke03` job `168691` after its
  `server02` estimate slipped to `2026-07-07T03:16:46`, and archive its log.
- [x] Relaunch `smoke03` as Slurm job `168759` with only
  `EXCLUDE_NODES=server39`.
- [x] Cancel superseded pending-only `smoke03` job `168759` after its estimate
  slipped to `2026-07-07T06:11:58` on `server57` with down/drained/reserved
  reason, and archive its log.
- [x] Relaunch `smoke03` as Slurm job `168807` with
  `EXCLUDE_NODES=server39,server57`.
- [x] Record 2026-07-07 00:46 CST status: job `168807` remains pending for
  priority, with current estimate `2026-07-07T02:12:58` on `server02`; no
  active smoke artifacts exist yet.
- [x] Record 2026-07-07 01:40 CST status: job `168807` started on `server59`
  and wrote manifest / working H5 / JSON.
- [x] Record 2026-07-07 01:55 CST result: job `168807` timed out before any
  video, summary, or review frame; classify as insufficient smoke
  resources/count rather than visual success.
- [x] Archive incomplete `smoke03` run/log under external archive.
- [x] Relaunch active static RGB smoke as `smoke04`, Slurm job `168881`, with
  `COUNT=1`, `1 GPU / 4 CPU / 16G / 00:45:00`, and
  `EXCLUDE_NODES=server39,server57,server59`.
- [x] Record 2026-07-07 02:29 CST status: job `168881` started on `server43`
  and wrote manifest / working H5 / JSON.
- [x] Record 2026-07-07 02:33 CST result: job `168881` failed with Vulkan
  `ErrorDeviceLost` / exit `134`; no video, summary, or review frame.
- [x] Archive failed `smoke04` run/log under external archive.
- [x] Patch `scripts/slurm/run_dataset_static_rgb_smoke_in_allocation.sh` to
  run `render_min_canary.py` before replay and to use `--shader minimal` for
  replay.
- [x] Relaunch active static RGB smoke as `smoke05`, Slurm job `168993`, with
  canary gate, `COUNT=1`, `1 GPU / 4 CPU / 16G / 00:45:00`, and
  `EXCLUDE_NODES=server39,server43,server57,server59`.
- [x] Cancel pending-only `smoke05` job `168993` after it stayed on
  unavailable `server63`; archive its log.
- [x] Relaunch `smoke05` as Slurm job `169001` with
  `EXCLUDE_NODES=server39,server43,server57,server59,server63`.
- [x] Record 2026-07-07 02:59 CST status: job `169001` is pending with
  `SchedNodeList=server02` and start estimate `2026-07-07T03:43:01`; no active
  smoke artifacts yet.
- [x] Record 2026-07-07 03:46 CST status: job `169001` remains pending, now
  estimated `2026-07-07T05:07:13` on `server44`; keep it because fresh
  test-only is later.
- [x] Record 2026-07-07 04:48 CST status: job `169001` started on `server44`
  and the minimal render canary passed.
- [x] Record 2026-07-07 04:49 CST result: `smoke05` completed successfully
  with one RGB video, three review frames, and `summary.json`.
- [x] Run `scripts/world_model/prepare_dataset_smoke_review.sh` after smoke
  artifacts exist.
- [x] Regenerate `smoke05/review_request.md` with accept/reject criteria,
  target-blocked status, and post-approval next steps.
- [x] Fix `scripts/world_model/dataset_goal_status.sh` so it reports the
  active smoke run `smoke05` instead of stale `smoke01`.
- [x] Fix `scripts/world_model/dataset_smoke_status.sh`,
  `scripts/world_model/prepare_dataset_smoke_review.sh`, and
  `scripts/world_model/require_dataset_smoke_approved.sh` so their default
  active smoke is `smoke05`.
- [x] Fix `scripts/slurm/launch_dataset_static_rgb_full_tmux.sh` so production
  approval checks `smoke05` while production output remains `full01`, and so
  canary / minimal-shader settings propagate into the production launcher.
- [x] Fix `scripts/slurm/run_dataset_static_rgb_smoke_in_allocation.sh` so
  future `static_rgb/full01` and later static smoke/full runs write required
  manifest schema fields directly: `log_file`, `source_paths`,
  `allowed_losses`, and `disallowed_losses`.
- [x] Switch `scripts/slurm/launch_dataset_static_rgb_full_tmux.sh` to the
  direct `srun` tmux launcher by default, with `PARTITION=cpu`, `GPUS=1`, and
  current Vulkan device-loss nodes excluded.
- [x] Add full static RGB status helper:
  `scripts/world_model/dataset_full_static_status.sh`.
- [x] Verify full static RGB production launcher refuses before human approval
  without creating tmux or submitting Slurm.
- [x] Verify full static RGB status helper reports `full01` as not started
  while `smoke05` is waiting for approval.
- [x] Verify next-stage readiness is represented in
  `scripts/world_model/dataset_goal_status.sh`.
- [x] Connect `scripts/world_model/dataset_training_inputs_status.sh` into
  `scripts/world_model/dataset_goal_status.sh`.
- [x] Verify `require_dataset_training_inputs_ready.sh diagnostic_b_bootstrap`
  passes only as diagnostic/readout scope and forbids positive DP BC / final
  method evidence.
- [x] Verify `require_dataset_training_inputs_ready.sh full_joint` currently
  fails because Stage 1 smoke is unapproved, A full RGB is missing, and new
  B/C/D/E collection artifacts are missing.
- [x] Record 2026-07-07 status: `smoke05` has one RGB video and three review
  frames; no approval/rejection decision file exists; production gate still
  rejects with `human_review_approval_missing`.
- [x] Record resource policy update: post-approval dataset jobs should request
  `1 GPU` by default, reduce CPU/memory first when queueing is hard, wait on a
  valid queued tmux allocation, and use previously bad nodes only as
  smoke-only diagnostics with node evidence recorded.
- [x] Update AGENTS resource rules: do not request/hold a GPU while human
  review or another guard is closed; after a guarded valid run is allowed,
  keep monitoring the tmux-held request, default to `1 GPU`, and lower
  CPU/memory/walltime before changing scientific scope.
- [x] Update `scripts/world_model/dataset_post_approval_plan.sh` so the
  post-approval command summary reports the same resource policy.
- [x] Patch `scripts/slurm/run_dataset_static_rgb_smoke_in_allocation.sh` so
  future `DATASET_SMOKE_ONLY=false` static full runs write
  `status=render_complete` and production notes, while smoke runs keep
  `status=smoke_complete`.
- [x] Patch `scripts/world_model/dataset_full_static_status.sh` so full static
  readiness recognizes `render_complete` static RGB output.
- [x] Record 2026-07-07 05:48 CST status: no Reflex Slurm job is pending or
  running; only an unrelated `/public/home/yanhongru/Curiosity` job appears in
  `squeue`. Do not request a Reflex GPU while `smoke05` human review remains
  unapproved.
- [x] Add active dynamic-scene adapter status guard:
  `scripts/world_model/dataset_dynamic_adapter_status.sh`.
- [x] Wire B/C/D/E stage readiness through the dynamic adapter guard so that
  after A approval / A full RGB, dynamic production still refuses with
  `dynamic_adapter_not_ready` until a reviewed active adapter exists.
- [x] Record that `PegInsertionSide-v1` uses a kinematic target box and old
  dynamic scripts moved it through direct pose/state paths, so active B/C/D/E
  must not silently revive those scripts as production data collection.
- [x] Add source-audited dynamic adapter:
  `scripts/world_model/active_dynamic_peg_adapter.py`. It exposes continuous
  kinematic-target commands, motion trace rows, dataset-class loss-role
  manifest fields, and explicit `state_intervention=false` /
  `snap_or_teleport=false` trace labels.
- [x] Record that the dynamic adapter source is not itself data and not a
  runner; runtime validation must happen later inside a compute-node Slurm
  smoke because ManiSkill's public wrapper does not guarantee the kinematic
  target command.
- [x] Add B dynamic RGB smoke collector:
  `scripts/world_model/collect_dynamic_rgb_observation_smoke.py`.
- [x] Add B dynamic RGB in-allocation runner:
  `scripts/slurm/run_dataset_dynamic_rgb_smoke_in_allocation.sh`.
- [x] Fix B dynamic smoke quantity semantics: `COUNT` is the number of
  episodes, default smoke is `1` episode, and `STEPS_PER_EPISODE=300` is
  required for complete review evidence. A B smoke ending before a complete
  episode is invalid as dynamic evidence.
- [x] Add C frozen-DP dynamic failure collector:
  `scripts/world_model/collect_frozen_dp_dynamic_failure_smoke.py`.
- [x] Add C frozen-DP dynamic failure in-allocation runner:
  `scripts/slurm/run_dataset_frozen_dp_dynamic_smoke_in_allocation.sh`.
- [x] Fix C dynamic smoke quantity semantics: `COUNT` is the number of
  rollouts, default smoke is `1` rollout, and `MAX_EPISODE_STEPS=300` is
  required for complete review evidence. The target / holed object must not
  move during robot grasp or initial approach.
- [x] Add B/C production launcher common guard:
  `scripts/slurm/launch_dataset_stage_production_tmux_common.sh`.
- [x] Add B training-scale production launcher:
  `scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh`, target
  `1000` episodes under `dynamic_rgb/prod01`.
- [x] Add C training-scale production launcher:
  `scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh`, target
  `500` rollouts under `frozen_dp_dynamic/prod01`.
- [x] Add D/E guarded production launcher entrypoints:
  `scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh` and
  `scripts/slurm/launch_dataset_cosmos_predicted_production_tmux.sh`. These
  remain blocked until real in-allocation runners exist and pass audit.
- [x] Fix B/C production quantity semantics so production `COUNT=1000/500`
  means episodes/rollouts, not per-episode step count.
- [x] Add read-only production status helper:
  `scripts/world_model/dataset_production_status.sh`.
- [x] Add read-only production run validator:
  `scripts/world_model/validate_dataset_production_run.sh`. B/C production
  must pass this before it can be treated as a training input candidate.
- [x] Strengthen `scripts/world_model/require_dataset_training_inputs_ready.sh`
  so `full_joint` requires B/C production validators to pass; directory
  existence alone is no longer accepted as training-readiness evidence.
- [x] Extend production validation to D/E and require those validators in
  `full_joint` too. D/E directory existence alone is no longer accepted as
  training-readiness evidence.
- [ ] Wait for human review of smoke RGB videos / frames before production.
- [ ] Create an active short dataset root for official static expert data
  after smoke approval.
- [ ] Render official 1000 demo RGB/video under an active dataset root after
  smoke approval.
- [ ] Align official state/action H5 rows with rendered RGB timestamps.
- [ ] Export target-centric relative trajectories:
  `inv(T_hole) * T_peg`, `inv(T_hole) * T_ee`.
- [ ] Label static insertion phase and final success.
- [ ] Write sample manifests with allowed losses:
  DP BC, Cosmos static future, phase extraction.

## Stage 2: Dynamic Observation Set

- [ ] Define dynamic motion families:
  constant, reverse, move-stop, sine/nonconstant, continuous moving target,
  peg/wooden-stick disturbance.
- [x] Record B motion families and target scale in
  `docs/dynamic_dataset_collection_plan.md`.
- [x] Create guarded B dynamic RGB observation smoke launcher. It currently
  refuses because the Stage 1 review gate and B runner are not ready.
- [x] Wire B dynamic launcher to the shared Slurm/tmux smoke launcher for
  post-approval use.
- [x] Runtime-smoke B dynamic RGB observation after Stage 1 RGB smoke is
  visually approved and A full RGB is ready.
- [ ] Run B dynamic RGB production only after B smoke is human-approved.
- [ ] Validate B production with
  `scripts/world_model/validate_dataset_production_run.sh b_dynamic_production`.
- [ ] Runtime-smoke `scripts/world_model/active_dynamic_peg_adapter.py` inside
  a compute-node Slurm allocation after Stage 1 approval / A full RGB. It must
  prove continuous logged motion and RGB/manifest evidence without per-step
  pose-edit or state-restore shortcuts.
- [ ] Recover or rewrite the real dynamic generator source for B; do not use
  pycache or placeholder generators.
- [ ] Adapt B from recovered source only after removing old paths and
  state-edit/controller-evidence risks.
- [ ] Generate legal moving-target episodes with RGB and state/action logs.
- [ ] Include unsuccessful episodes; do not mark failed actions as expert.
- [ ] Export future target / hole pose trajectory, velocity, tau candidates,
  and uncertainty labels.
- [ ] Write manifests with allowed losses:
  Cosmos future, target-frame readout, uncertainty, trajectory consistency.

## Stage 3: Frozen-DP Dynamic Failure Set

- [x] Record C target scale, required outputs, and allowed losses in
  `docs/dynamic_dataset_collection_plan.md`.
- [x] Create guarded C frozen-DP dynamic failure smoke launcher. It currently
  refuses because the Stage 1 review gate and C runner are not ready.
- [x] Wire C frozen-DP dynamic launcher to the shared Slurm/tmux smoke
  launcher for post-approval use.
- [x] Runtime-smoke C frozen-DP dynamic failure after Stage 1 RGB smoke is
  visually approved and A full RGB is ready.
- [ ] Run C frozen-DP dynamic failure production only after C smoke is
  human-approved.
- [ ] Validate C production with
  `scripts/world_model/validate_dataset_production_run.sh c_frozen_dp_production`.
- [ ] Recover or rewrite the real frozen-DP dynamic rollout source for C; do
  not use pycache or placeholder generators.
- [ ] Adapt C with the official frozen DP checkpoint and active dynamic scene
  perturbation path; do not treat teacher/future labels as controller inputs.
- [ ] Run frozen official DP in dynamic scenes inside Slurm allocation.
- [ ] Record action trace, video, before/after distance, and final outcome.
- [ ] Label miss, jam, target-assisted, bad relative velocity, no progress,
  and any true robot-driven success.
- [ ] Write manifests with allowed losses:
  negative classification, discrepancy, infeasible/no-progress, contrastive.

## Stage 4: Future-Frame Cooperation Teacher Set

- [x] Record D teacher-only ground-truth future rules and target scale in
  `docs/dynamic_dataset_collection_plan.md`.
- [x] Create guarded D future-frame cooperation teacher smoke launcher. It
  currently refuses because the Stage 1 review gate and D runner are not
  ready.
- [x] Wire D future-frame teacher launcher to the shared Slurm/tmux smoke
  launcher for post-approval use.
- [x] Implement source-ready D future-frame cooperation teacher collector and
  in-allocation runner:
  `scripts/world_model/collect_future_frame_teacher_smoke.py` and
  `scripts/slurm/run_dataset_future_frame_teacher_smoke_in_allocation.sh`.
- [x] Runtime-smoke D future-frame cooperation teacher after Stage 1 RGB smoke
  is visually approved and A full RGB is ready.
- [ ] Validate D production with
  `scripts/world_model/validate_dataset_production_run.sh d_future_teacher_production`.
- [ ] Build teacher rollout that uses ground-truth future target trajectory
  only for data generation.
- [ ] Execute through legal controller actions; no state edit or snap.
- [ ] Record target-frame and world-frame trajectories.
- [ ] Record phase, tau, relative velocity at contact, and insertion corridor
  progress.
- [ ] Train/evaluate the moving-frame adapter on GT future frames before using
  Cosmos predictions.

## Stage 5: Cosmos-Predicted Cooperation Set

- [x] Record E Cosmos-predicted cooperation requirements and target scale in
  `docs/dynamic_dataset_collection_plan.md`.
- [x] Create guarded E Cosmos-predicted cooperation smoke launcher. It
  currently refuses because the Stage 1 review gate and E runner are not
  ready.
- [x] Wire E Cosmos-predicted launcher to the shared Slurm/tmux smoke launcher
  for post-approval use.
- [x] Add explicit E prereq guard so Cosmos-predicted cooperation cannot start
  before B production, D production, and held-out Cosmos/readout validation are
  ready:
  `scripts/world_model/require_dataset_cosmos_predicted_prereqs_ready.sh`.
- [ ] Implement E Cosmos-predicted cooperation in-allocation runner after D
  and Cosmos/readout validation.
- [ ] Validate E production with
  `scripts/world_model/validate_dataset_production_run.sh e_cosmos_predicted_production`.
- [ ] Train or evaluate Cosmos/readout to predict future target frame from RGB
  history.
- [ ] Replace GT future target frame with Cosmos-predicted future frame.
- [ ] Add noise/uncertainty augmentation to train adapter robustness.
- [ ] Evaluate receding-horizon insertion with predicted future target frames.

## Rejection Rules

- [ ] Reject or negative-label target-assisted self-insertion.
- [ ] Reject state intervention, snap, source-state restore, saved-state
  replay, geometric final placement, or hidden manual finisher.
- [ ] Reject samples whose RGB and state/action timestamps cannot be aligned.
- [ ] Keep old 733 examples as context/ablation unless revalidated under the
  new manifest and loss-role rules.
