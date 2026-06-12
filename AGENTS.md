# Agent Rules

These project rules apply to future agent work in this repository.

## Research Objective

The active objective is static-policy-to-dynamic-scene transfer through
world-model task rebinding:

> A policy trained on static manipulation tasks should finish the original task
> in dynamic test scenes by using streaming object/state perception,
> object-centric world prediction, task-frame rebinding, short-horizon physical
> bridging, and conservative policy-continuability handoff.

The goal is not to restore an old scene layout. The goal is task completion in
the changed world.

## Latest User Override: 2026-06-12

- Stop v7 data construction now. Do not continue generating or backfilling the
  v7 DP full1000 source set unless the user explicitly asks to resume data
  generation.
- The current frozen v7 DP source merge is the SFT source for the immediate
  Cosmos3 work:
  `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733`.
  It contains `733` unique H5 trajectories with all nine classes present, but
  it is intentionally not a full 1000-row quota-complete set.
- Proceed directly through Cosmos3 SFT preparation and training from this
  frozen source. Intermediate checks are agent-owned: run strict H5 audit,
  RGB render/preflight, WAM condition export, action/length audits, and visual
  sanity checks without stopping for user approval at each gate.
- Preserve the 301 RGB/state frame and 300 action-step contract. Do not use
  128/129-frame chunks, 93-frame crops, or any silent truncation.
- If a check fails, debug the render/export/SFT path itself first. Do not
  restart data construction or switch to hard-teacher data unless the user
  explicitly redirects to that path.

## Non-Negotiable Discipline

- Current continuation directive: execute the RGB-D world-model plan until the
  real objective is handled. Do not stop at a status recap while aligned work
  remains, and do not shrink success to a currently convenient subset.
- Do not degrade the method into hand-coded case splits, metric chasing, or
  threshold-only hacks.
- Continuation across turns is part of the same objective. Do not redefine
  success around the easiest completed subset, a currently passing scaffold,
  or a smaller experiment just because the full RGB-D method chain is waiting
  on cluster resources.
- Pending cluster resources, failed pilots, weak metrics, or inconvenient
  artifacts are not reasons to downgrade the method. Keep analyzing and fixing
  the aligned RGB-D data/perception/world-model/controller path; only stop for
  user direction after the failure is concrete, repeatedly confirmed, and not
  safely resolvable from current files and cluster state.
- Do not degrade the method into a state-only/oracle-slot controller result.
  The actual method must be a RGB-D world-model pipeline: RGB-D perception
  must produce object/task slots or latent state, the world model must predict
  future task state from those RGB-D-derived representations, and the
  controller must be evaluated using those predicted representations. Oracle
  state results are debugging scaffolds or upper bounds only, never the main
  method result.
- Do not change evaluation protocols to make a result look better.
- Do not tune, remove, relax, or replace evaluation gates because a current
  result is inconvenient. Evaluation changes are allowed only for documented
  implementation bugs or previously missing measurements, and the note must
  state why the change preserves the original dynamic task-completion
  objective.
- Latest user correction from 2026-06-04: continuous predicted-slot quality
  numbers such as hole-position RMSE and peg-head-in-hole RMSE are advisory
  diagnostics, not hard downstream blockers. Record them, use them to diagnose
  and rank perception attempts, and compare against previous runs, but do not
  stop RGB-D-derived world-model or controller training solely because those
  RMSE diagnostics miss a hand-written threshold. Hard requirements remain:
  exact full1000 files, structural validity, frame alignment, RGB-D-derived
  boundary/provenance, no oracle-state inference, controller metrics from the
  real final state, and inspected video/contact evidence.
- Latest user correction from 2026-06-05: stop the self-trained lightweight
  object-slot dynamics world-model method line. It may remain in archived notes
  only as historical diagnostic/interface evidence, but future agents must not
  train new in-repo object-slot dynamics WMs, submit object-WM controller runs,
  or present those models as the publishable method. The active world-model
  line must use a serious published/foundation video/world model, with Cosmos3
  as the current primary backbone, adapted to the RGB-D dynamic manipulation
  data and then connected to downstream controller evaluation.
- Cosmos3 data quality requirement from 2026-06-05: do not train or evaluate
  Cosmos3 from the old `sft_dataset_full1000_rgbd` preview-video links. Those
  videos are `320x192`, too fast, and use weak/non-default viewpoints. Future
  Cosmos3 data must use large readable videos from the same dynamic
  trajectories, rendered with the user-approved ManiSkill3
  `PegInsertionSide-v1` default human-render viewpoint at `30 fps` with no
  deliberate playback slowdown. Record camera/viewpoint, resolution, FPS,
  frame sampling, and visual inspection before training.
- Current 2026-06-05/06 Cosmos3 data boundary: the old full1000
  Cosmos/controller visual data are dirty for the foundation-WM path and must
  not be used as active training or controller input. The corrected 10-video
  preview using the ManiSkill3 default human-render viewpoint was approved by
  the user on 2026-06-05 after the
  `maniskill_default_preview10_20260605_103547` review. That preview approval
  was not an explicit instruction to regenerate the full dataset. The current
  6/6 regenerated full1000 path may continue only because the user explicitly
  allowed this specific in-progress chain on 2026-06-06. Future full-dataset
  regeneration, archival, replacement, or derived-chain reset must first
  report concrete evidence and proposed options to the user and wait for
  explicit approval. The current chain uses saved env-state trajectories with
  the approved camera/FPS, then runs Cosmos3 SFT with validation loss
  inspection, then proceeds to controller training/evaluation. Archived
  full1000 H5 files may be used only as env-state sources for re-rendering;
  old preview videos must not be reused.
- User correction from 2026-06-06: the current 6/6 regenerated Cosmos3
  full1000 path may continue, but it must not become a precedent for unilateral
  future data resets. If downstream results are bad, surprising, or suggest
  possible data problems, first inspect and report the concrete evidence and
  proposed options to the user. Do not move, archive, replace, or regenerate a
  full dataset or derived training chain merely because a result is poor unless
  the user explicitly approves that reset.
- Latest Cosmos3 fix3 data-construction correction from 2026-06-11: the
  approved v7 source protocol may be scaled to full1000, but class quotas may
  be nonuniform only if no class becomes tiny. Generate several classes or
  class groups in parallel instead of blocking sequentially on one low-pass
  class. Attempt/seed selection is an efficiency detail, not a research
  blocker; avoid duplicate seeds when merging, but do not spend repeated turns
  inventing seed tricks when generation quality is otherwise valid. If a class
  repeatedly cannot be generated after concrete log inspection and no safe
  aligned fix is available, stop for user direction instead of trying random
  variants. When the full1000 source set is complete, stop and wait for user
  approval before WAM export, rendering expansion, controller integration, or
  Cosmos3 SFT.
- Latest user correction from 2026-06-12 on fix3 data semantics: do not throw
  away, archive, or reclassify already accepted successful v7 H5 samples merely
  because the remaining construction strategy changes. Already generated valid
  samples stay in the full1000 candidate pool. The correction applies to the
  missing remainder: do not keep filling it by static-DP success filtering
  alone. Future added samples should make the original/static DP an explicit
  low-success baseline and should use scripted/manual/oracle-teacher
  construction when needed to create physically valid hard dynamic examples
  where target motion is genuinely outside the static DP's unaided competence.
  This follows the Dream Diffusion Policy lesson: OOD shifts must be
  deliberately constructed and baseline failure measured; expert/teacher data
  may be scripted, but the positive dataset must not be defined as "cases
  where the baseline happened to succeed." Preserve existing accepted H5 files,
  add provenance tags for old-v7 versus hard-teacher supplements, and do not
  waste time regenerating accepted rows unless a concrete structural/visual
  audit proves them invalid.
- Later 2026-06-12 user override on fix3 execution order: defer the
  hard-teacher supplement and do not let it block the current Cosmos3 chain.
  The active near-term data target is again the user-approved v7 DP-generated
  `full1000` source set. Continue filling the original full1000 quotas with
  the v7 DP generator, merge/deduplicate by scenario seed, run the strict
  source audit and rendered review gates, then proceed to Cosmos3 SFT only
  after the required approval/preflight evidence exists. Keep the hard-teacher
  script and notes as a later direction, not as the active blocker.
- Cosmos3 conditioning requirement from 2026-06-05: SFT must not be treated as
  single-frame I2V. Use a short video-prefix condition for future prediction,
  currently `conditioning_config={8: 1.0}` for both train and validation. The
  SFT data must also expose robot/object task-state grounding, currently by
  appending hole, peg, TCP, grasp, insertion, and perturbation summaries to
  the JSONL caption and metadata. Caption dropout must not erase this
  task-state text in the active manipulation SFT. Later controller integration
  should preserve this robot/object grounding rather than relying on pixels
  plus generic text alone.
- Latest Cosmos3 conditioning correction from 2026-06-05: caption-only
  robot/object grounding is not sufficient for the active method. The active
  Cosmos3 manipulation SFT/controller path must condition on a short video
  prefix plus structured causal action/proprio/object-state inputs whenever
  source rollout action/state records exist: robot action commands, TCP/hand
  state, peg state, hole state, task-frame relative geometry,
  grasp/insertion predicates, and perturbation summaries. If those structured
  manipulation inputs are missing, generate or recover them before method SFT;
  otherwise the run is only a diagnostic. Future rows may include
  candidate/recorded robot action commands, but must not include future
  ground-truth object poses as privileged conditions. Do not run or report a
  Cosmos3 SFT/controller path as method evidence if it is only single-image
  I2V, video-only V2V, or generic caption-conditioned video generation without
  these structured manipulation conditions.
- Single-image I2V is not an active Cosmos3 world-model/controller path. Old
  I2V scripts may be used only with an explicit diagnostic override and must
  not be reported as SFT/controller evidence. Post-SFT reconstruction and
  controller-facing rollout should use video-prefix conditioning plus
  robot/object state grounding. Do not rely on the Cosmos framework default
  video-conditioning indices `[0, 1]`; the active path must explicitly use
  `[0, 1, 2, 3, 4, 5, 6, 7]` to match the SFT video-prefix condition. The
  active post-SFT diagnostic entry point is
  `scripts/slurm/watch_cosmos3_sft_completion_v2v_eval.sh`; the old
  `watch_cosmos3_sft_completion_i2v_eval.sh` is guarded off by default and
  may only be used with `ALLOW_SINGLE_IMAGE_I2V_DIAGNOSTIC=true` for a
  non-evidence diagnostic.
- Controller-facing Cosmos conditioning must be causal. Reconstruction
  diagnostics may use the state-grounded JSONL caption to sanity-check the SFT
  video model, but controller evidence must condition on current/history
  RGB-D-derived robot/object state and candidate action/bridge context, not on
  future ground-truth object positions from the evaluated rollout.
- Latest user correction from 2026-06-08 on full-length RGB/RGB-D world-model
  training and inference: the active world model must support arbitrary
  rollout length through either a native full-length model call or an explicit
  chained/receding rollout that stitches short predictions without dropping
  frames. For every SFT/eval sample used as method evidence, predicted video,
  predicted action/state sequence, reference visual video, and GT action/state
  rollout must have the same intended frame/step length after any documented
  prefix offset. Silent truncation, `min(pred, ref)` comparison, fixed
  93-frame action eval, or PSNR on a cropped segment is diagnostic only and
  must not be reported as complete WM evidence. The current 6/6 Cosmos3
  action-eval video with `93` predicted frames versus a `301`-frame reference
  is therefore a short-horizon diagnostic, not a complete world-model result.
- Latest user correction from 2026-06-08 on Cosmos3 model selection and RGB
  inputs: before any further active Cosmos SFT run, audit the currently
  available Cosmos3 variants and local checkpoint/tooling. Do not assume
  `Cosmos3-Nano` is the best backbone merely because it is already downloaded.
  If a stronger applicable Cosmos3 variant such as `Cosmos3-Super` can be run
  or LoRA-SFTed on available H200 resources, switch to it or run the required
  download/preflight. If resource limits force `Nano`, record the concrete
  reason and treat it as a constrained baseline. Latest user override: depth
  is not required for the active ManiSkill Cosmos/DROID WAM SFT. Active
  SFT/evaluation must train from approved RGB visual inputs, not state-only
  trajectories and not depth. Simulator state
  may be used only for causal action/proprio metadata, labels, diagnostics,
  or visual readout supervision, never as the world model's visual
  input replacement.
- Latest user correction from 2026-06-08 on the invalid old ManiSkill
  Cosmos3 SFT chain: previous ManiSkill Cosmos3 SFT/eval results that used
  the 6/6 `total_video_frames=93` action/state export or compared 93-frame
  predictions against 301-frame references are not trustworthy and must not
  be continued, used as active checkpoints, used as controller input, or
  reported as method evidence. The ManiSkill world model must be restarted
  from a fresh full-length/equal-length SFT using either
  `Cosmos3-Nano-Policy-DROID` as a joint world-action-model backbone or
  `Cosmos3-Nano-DCP` as the constrained baseline. The fresh SFT must use RGB
  only, no depth input, strict pred/GT length accounting, and no accidental
  truncation. `Cosmos3-Nano-Policy-DROID` may and should be tested as a joint
  WAM, not only as an action prior: the adapted run should evaluate future RGB,
  task-state readout, and action/chunk prediction under the same length
  contract.
- Latest user correction from 2026-06-09: any currently trained ManiSkill
  Cosmos3 world-model SFT result is untrusted if its input video, predicted
  video, GT video, action targets, or state/readout targets are not exactly the
  intended same-length contract, or if accidental truncation/cropping is
  possible. Do not continue old ManiSkill SFT checkpoints/results after this
  failure mode. The active ManiSkill world model must be freshly trained with
  `Cosmos3-Nano-Policy-DROID` or the fresh `Cosmos3-Nano` constrained baseline,
  RGB visual input only, strict equal-length preflight, strict post-SFT
  prediction-vs-GT inspection, and no silent truncation in metrics or
  downstream controller inputs.
- Latest user correction from 2026-06-09 on the Cosmos3 training/test reset:
  the approved full1000 data should be kept and does not need regeneration,
  but the 129-frame / 128-action chunked construction is explicitly rejected.
  The intended "300 frames" boundary means the total episode horizon, not an
  extra 300-frame future prediction target. The active contract is a 300-step
  episode, with 301 RGB/state frames when frame 0 is included and 300 action
  steps. Future Cosmos3 training/testing must use the full episode/equal-length
  contract with causal prefix masks or remaining-horizon prediction over the
  same source episode; it must not physically slice samples into 129-frame
  clips or 128-action chunks. The rejected chunked SFT/checkpoints/watchers
  are historical only and must not be used as active method evidence.
  Controller/DP integration remains paused until the new full-episode plan is
  reviewed, a clean full-episode condition export passes strict preflight, and
  full-length generated-video/action/readout evidence with visual review
  exists.
- User correction from 2026-06-06 on Cosmos/controller integration: a short
  one-shot 93-frame Cosmos action-eval video from frame 0 is only a
  reconstruction/readout diagnostic, not a valid controller-facing world-model
  interface. Controller-facing Cosmos inference must be receding and
  teacher-forced by real observations: condition on the latest observed video
  prefix and RGB-D-derived robot/object state at or after the dynamic event,
  predict only a short future horizon, execute one/few controller steps, then
  refresh the prefix/state with the newly observed post-motion frames before
  predicting again. Inference must not rely on a stale pre-motion prefix or a
  single precomputed trajectory when the object has already moved. If this
  realtime teacher-forced interface is missing, the run is only a diagnostic
  and must not be reported as the active Cosmos controller method evidence.
- User correction from 2026-06-06 on controller purpose and training:
  a controller that takes over from the frozen DP must preserve the base DP's
  grasp-hold and insertion competence before it is allowed to claim dynamic
  task progress. A hand-coded geometric bridge that makes the robot lose or
  visually fail to hold the peg is a diagnostic failure, even if a numeric
  predicate is noisy or briefly favorable. Future controller training must not
  be "train a standalone controller" only. The active policy path should
  distill the official/static DP skill while adding world-model-conditioned
  takeover samples, so the learned policy can execute along imagined or
  predicted task trajectories without destroying the original grasp/insertion
  behavior. Failed takeover videos, including the 2026-06-06
  `cosmos_receding_teacher_forced_contact_preserve_20260606_132036` run where
  the peg is not visually held through insertion, must be treated as negative
  diagnostics and must not enter positive takeover distillation data.
- User correction from 2026-06-06 on Dream Diffusion Policy style controller
  fallback: if positive takeover distillation is not available or a first
  distillation implementation is stuck, do not keep waiting or return to
  threshold-only bridge tuning. The fallback controller direction must preserve
  the same causal structure: a serious world model such as Cosmos3 predicts or
  imagines a short future task trajectory from current/history observations,
  while a DP/expert policy prior or learned execution policy produces valid
  manipulation actions along that imagined path. The diagnostic
  `control_policy=wm_dp_prior_mpc` tried the minimal version of this idea
  around the frozen DP action prior and failed final success in oracle-slot
  no-video smoke tests; those failures are not method success and should push
  future work toward stronger expert/trajectory supervision or learned
  continuability, not more hand-written threshold hacks.
- 2026-06-07 DDP controller clarification: do not interpret DDP-style control
  as long open-loop video prediction, long open-loop simulator-copy planning,
  or waiting indefinitely for pure takeover distillation. The relevant
  execution contract is short action chunks, world-model/imagination scoring
  or conditioning, live execution of only a short prefix, then real
  re-observation and re-alignment. In contact-rich insertion, a restored
  `set_state_dict` planner env can report insertion that the live rollout does
  not achieve with identical actions, so restored planner contact success is
  diagnostic only. Future controller work should use DP/expert/Cosmos-
  conditioned short-chunk action generation with live re-observation as the
  authority, and must not keep tuning primitive gates after this failure mode
  is observed.
- Visual grasp/hold review is mandatory for takeover distillation data. A
  rollout cannot be used as a positive controller-teacher sample only because
  H5 metrics exist. It must have inspected video/contact-sheet evidence showing
  the peg is actually held, the robot follows the intended world-model-derived
  path, and the final task state is measured from the real simulator state.
  If visual review contradicts metrics, report the visual failure and keep the
  rollout out of positive training data.
- Latest visual correction from 2026-06-05: the candidate videos must use the
  ManiSkill3 `PegInsertionSide-v1` default human-render viewpoint, not a custom
  frontal/high-oblique guess. The official local/default camera is
  `look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])` with `fov=1` and a
  `512x512` camera; future preview/data videos may upscale this view for
  readability but must preserve the same camera pose/fov. Video FPS must be
  `30`, and frame sampling must not deliberately slow playback.
- Do not hide failed pilots. Record what failed, why it failed, and whether it
  counts as evidence.
- Do not reuse archived C/D1/restoration results as the new method.
- Do not claim success without authoritative evidence from current files,
  metrics, logs, and rendered/video artifacts.
- Do not replace a failed or inconvenient experiment with an easier objective.
  First analyze the physical or implementation failure; only stop for user
  guidance after the failure is concrete and cannot be resolved safely from the
  repo and cluster state.
- When an experiment fails, the required order is: inspect logs/artifacts,
  identify whether the failure is physical, implementation, data, scheduling,
  or evaluation-related, try the aligned fix, and record the evidence. Do not
  guess a story and move on. If the failure cannot be resolved from current
  files and cluster state, stop and ask for user direction instead of inventing
  a weaker substitute.
- Latest user correction from 2026-06-11: when the same data-generation,
  rendering, training, or debugging blocker repeats across multiple attempts
  without a concrete new diagnosis, stop and wait for user direction. Do not
  keep trying speculative variants, changing protocols, or burning held GPU
  time just to appear busy. Preserve any held allocation when possible, report
  the concrete blocker, the attempts already made, and the next options.
- Refuse degradation, shortcutting, and preference-driven reporting. Do not
  optimize for looking successful, pleasing the user, or matching a convenient
  metric if that conflicts with the original dynamic task-completion objective.
  A failed real experiment is better evidence than a passing easier test.
- Do not use apologies, confident prose, or user-pleasing summaries as a
  substitute for evidence and execution. When a correction exposes degradation,
  record the boundary, repair the pipeline or TODO state, and report the
  current evidence directly.
- Do not end a turn by repeating status when actionable work remains. If full
  RGB-D data are not available, keep making aligned progress with smoke checks,
  preflight validation, queue/resource probes, failure-localization tooling, or
  small Slurm jobs that verify the RGB-D pipeline direction without being
  reported as method evidence. Once full RGB-D data are available, shift
  immediately to the strict RGB-D slot, RGB-D-derived world-model, controller,
  and video-inspection chain.
- Refuse slope changes from the stated method into easier surrogates. If the
  RGB-D path fails, debug RGB-D data/perception/world-model/controller
  alignment first; do not silently replace it with oracle state, CV-from-state,
  hand-coded segmentation, or a narrower test.
- Do not run CPU-heavy or GPU-heavy experiments on the login node. Use Slurm
  compute nodes for rollout, replay, rendering, training, and large labeling.
- Do not spin on the Slurm queue. When RGB-D data are pending, check queue and
  artifacts on an approximately 30-minute cadence unless a job starts, a job
  finishes, a notification appears, or new artifacts/logs appear. Between
  checks, do aligned smoke validation, preflight, artifact inspection,
  failure-localization tooling, or documentation; do not wait passively.
- Do not use CPU-only ManiSkill/SAPIEN controller rollout as a substitute for
  a render-capable Slurm allocation. Even no-video controller rollout can
  construct a SAPIEN render system and fail without a working Vulkan/GPU
  context. A CPU-only path may be used only after a concrete preflight proves
  the simulator can construct the environment on that node class; otherwise it
  is a scheduling/rendering failure path, not controller evidence.
- When a result is bad or surprising, analyze the concrete logs, manifests,
  artifacts, and visuals first. Try an aligned fix when the current repo and
  cluster state support one; if the failure is concrete and not safely
  resolvable, stop for user direction instead of guessing or weakening the
  task.
- A major success requires metrics plus video/replay evidence inspected by the
  agent. If video exists, open it or a padded frame/contact-sheet grid before
  recording success. A metric-only result is not enough for dynamic
  manipulation claims.
- Write every plan, TODO, manifest, and evidence note so a smart non-expert can
  follow the causal chain: what changed in the world, what the robot observed,
  what the world model predicted, what bridge/handoff happened, and why the
  final state proves success or failure.
- Explain and record experiments from first principles, without assuming the
  reader will fill in missing causal steps. For every new module, wrapper,
  repair, or interpretation, make clear in plain language what physical
  problem it addresses, why it belongs to RGB-D task-frame rebinding rather
  than a shortcut, what evidence would prove it, what failure would falsify it,
  and whether the original dynamic task-completion evaluation is preserved.

## Experiment Execution Loop

For every substantial experiment or code change:

1. Re-read `TODO/world_model_task_rebinding/00_active.md` and the relevant
   focused TODO/PLAN file before acting.
2. State the first-principles reason for the action in the manifest, evidence
   note, TODO entry, or code metadata: what physical failure or capability it
   addresses and why it is part of RGB-D world-model task rebinding.
3. Run the aligned experiment through Slurm or a lightweight local syntax/path
   check only. Do not run heavy jobs on the login node.
4. If it fails, inspect logs, manifests, generated files, and visual artifacts
   before interpreting the result. Classify the failure as data, rendering,
   perception, world-model, controller, physics, scheduling, or evaluation
   implementation.
5. Try a fix only when the fix preserves the original objective. If current
   files and cluster state do not support a justified fix, stop and ask the
   user instead of guessing or weakening the task.
6. Record important outcomes under `docs/world_model_task_rebinding/`, with
   job IDs, artifact paths, metrics, visual/video evidence, and an explicit
   statement of what the result does and does not prove.

## First-Principles Check

Before adding a module or changing a method, answer in writing or in code
comments/metadata when appropriate:

1. What physical problem does this solve?
2. Why is it part of task-frame rebinding rather than a shortcut?
3. What evidence would prove it works?
4. What failure would falsify it?
5. Does it preserve the original evaluation objective?

If the answer is unclear, inspect the plan/TODO files before proceeding.

## Required Plan/TODO Hygiene

- Re-read the relevant files under `PLAN/world_model_task_rebinding/` and
  `TODO/world_model_task_rebinding/` before implementing a major change.
- During long experiment runs, re-check the active TODO and the relevant
  focused TODO before submitting follow-up jobs, changing code, interpreting
  results, or writing a final status. This is required to prevent drift from
  the original RGB-D world-model plan.
- Keep `PLAN/README.md` and `TODO/README.md` stable entry points.
- Add or update focused small files under the method-specific subdirectories
  instead of mixing unrelated notes.
- Keep new outputs under method-specific directories, not under archived paths.

## Evidence Rules

- Major successes require metrics plus a video or replay demo.
- If the full RGB-D dataset is missing, the correct status is "no RGB-D method
  evidence yet." Do not infer RGB-D performance from state/oracle runs,
  preview videos, partial static-only shards, or code-path smoke checks.
- Full-scale RGB-D method experiments require the exact 1000-demo RGB-D dataset
  to be generated, structurally inspected, and visually reviewed. The existing
  exact96/full96 data and full96-plus-online5 data are small-scope validation
  or debugging data only; they must not be presented as the complete
  experiment or as full-scale method evidence.
- Current full1000 RGB-D visualization budget: rendered/visual demo artifacts
  should be limited to 10 diagnostic samples unless the user explicitly changes
  this later. This limit is only for human-readable visual artifacts and must
  not reduce RGB-D slot training, predicted-slot export, RGB-D-derived
  world-model training, controller inputs, controller metrics, hard
  structural/evaluation gates, or the exact 1000-demo data requirement.
- Predicted-slot continuous RMSE thresholds are diagnostic quality references
  unless the user explicitly reinstates them as blockers. A structurally valid,
  exact1000, frame-aligned RGB-D-predicted slot export may be handed to the
  world model with those RMSE values recorded as advisory evidence.
- A major success is not accepted until the agent has opened the video or an
  equivalent replay-derived contact sheet/frame grid and verified that the
  visible behavior matches the metric: the dynamic event occurs, the robot
  reacts through the intended RGB-D-derived rebinding path, and final success
  is measured from the real final state.
- When a video is hard to inspect directly, create a padded contact sheet or
  frame grid and inspect that image before recording the result as a success.
- Major method claims require RGB-D world-model evidence. A result based only
  on simulator state, oracle object poses, CV-from-state predictors, or
  rendered video used only for visualization is not a RGB-D world-model result
  and must not be presented as one. It can only be recorded as a diagnostic,
  upper bound, or component sanity check.
- Major results, including failures that change the research direction, must
  be recorded in a focused evidence note under
  `docs/world_model_task_rebinding/` with job IDs, inputs, metrics, artifacts,
  and whether the result does or does not support the method.
- When a video demo is produced, the agent must inspect it directly. If useful,
  create a padded contact sheet or frame grid for fast review.
- Metrics without visual/replay evidence are intermediate evidence, not final
  evidence for dynamic manipulation claims.
- A result is not successful unless the dynamic event happened before task
  completion and the final task success is measured from the real final state.

## Slurm And Rendering

- Use Slurm for heavy rollout, replay, rendering, training, and label
  generation.
- Do not use `scancel` as the default way to stop a command running inside a
  tmux-held Slurm allocation. Preserve held GPU resources. Stop in-allocation
  commands by sending `Ctrl-C` to the tmux pane or otherwise interrupting the
  foreground process inside that allocation. Use `scancel` only when the user
  explicitly asks to release/cancel Slurm resources, or when a process is
  unreachable and the action is clearly documented as not releasing the held
  allocation.
- Latest training-resource override from 2026-06-04: model training evidence
  must use at least 1 H200 GPU and reserve at least 3 hours. Do not force
  4 H200 GPUs as the minimum. A 4-GPU job is still allowed when it starts
  earlier or is explicitly useful, but future RGB-D slot/world-model training
  should prefer a persistent 1-H200 allocation when that reduces queue churn.
  A 1-day allocation is the default long request; use longer only when live
  Slurm forecasts and experiment duration justify it.
  Shorter or non-H200 jobs may only be treated as code-path smoke checks, not
  as training evidence or evidence against a research direction.
- RGB-D data generation and RGB-D world-model training are required core work,
  not optional follow-up. Synchronized RGB, depth, state, actions, env states,
  camera parameters, and object-pose labels must be generated so that the
  world model can be trained and evaluated from RGB-D-derived representations.
  If full RGB-D data are missing, do not advance method claims; fix data
  generation first.
- Latest RGB-D render scheduling override: if multi-GPU blocks do not start
  promptly, submit disjoint one-GPU RGB-D render shards instead of waiting.
  The previous 8-node / 64-GPU cap and the previous refusal of one-GPU render
  shards are superseded for RGB-D data generation. This override does not relax
  exact expected-count inspection, visual review, RGB-D-derived evidence,
  training floors, controller metrics, or video inspection.
- One-GPU RGB-D rendering is a small rolling-batch fallback, not permission to
  submit one job per trajectory or a large fanout of single-GPU jobs. Estimate
  walltime from completed shards, submit a small batch, inspect runtime,
  failures, H5/video/contact artifacts, then decide whether to scale. The
  default wrapper guard rejects more than 8 render jobs per submission and
  more than 4 one-GPU render jobs per submission unless explicitly overridden
  with a documented reason.
- Large RGB-D generation should use whatever legal Slurm layout starts useful
  disjoint work soonest, including one-GPU shards. Full-node or multi-GPU
  allocations are still fine when they start earlier, but they are not required
  for rendering. Keep every shard disjoint and preserve the same strict
  inspection and downstream training gates.
- Before submitting a duplicate or replacement RGB-D render, run non-executing
  Slurm probes such as `sbatch --test-only` for the legal dense/partial-node
  layouts. Submit the replacement only if it is expected to produce the same
  strictly inspected RGB-D data earlier without wasteful allocation. If every
  legal probe starts later, do not pollute the queue; record that decision and
  keep waiting for the current earliest RGB-D path.
- `sbatch --test-only` is a probe, not authoritative replacement evidence.
  Do not cancel an existing active RGB-D data/training/method path merely
  because a test-only probe looks earlier. If a replacement is justified,
  first submit at most one complete or explicitly bounded replacement, compare
  the replacement's actual `squeue`/`scontrol` pending state and stable start
  forecast against the current path, verify downstream dependencies/output
  paths are safely redirected or intentionally absent, and only then cancel
  the old path. If the replacement does not get a stable earlier real
  forecast, cancel the replacement before allocation and record the failure as
  scheduling evidence, not method evidence.
- Do not treat a queued large RGB-D render as the only path just because its
  forecast is earliest. When full RGB-D data are missing, also submit disjoint
  shard jobs that can fit smaller backfill holes, including one-GPU shards
  under the latest override, while ensuring the merged output still passes
  exact expected-count strict inspection. The correct strategy is to get RGB-D
  work running early, then continue adding legal disjoint shards, not to wait
  passively for a single large block.
- For SAPIEN/ManiSkill rendering, export:
  - `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`
  - `DISPLAY=`
- For HDF5 replay/render jobs on the shared filesystem, export
  `HDF5_USE_FILE_LOCKING=FALSE`; otherwise `h5py` can fail with
  `No locks available` on compute nodes.
- Do not maintain or consult a standing node exclusion list. Old node failures are
  observations from specific jobs and times, not policy. If a bad-node-list
  file is reintroduced into the active repo, remove it before scheduling from
  it. Do not hard-code a default node exclusion in Slurm wrapper source. Use
  live `sinfo`/`scontrol`, `sbatch --test-only`, and targeted canaries for
  job-local node decisions; if a current drain/down node must be excluded, pass
  it explicitly at submission time and record the live evidence in the manifest
  or evidence note.
- Every heavy job should write a manifest with command, seed range, node,
  environment, checkpoint, and output path.

## Current Implementation Priority

- Treat oracle state/object slots as scaffolding only: they may debug
  task-frame geometry, label continuability, verify simulator state, and
  provide an upper bound, but they do not constitute the method.
- Do not present the current in-repo lightweight Transformer object-slot world
  model as the final publishable world-model contribution. It is an
  interface/diagnostic baseline for proving the RGB-D slot-to-controller path,
  checking data alignment, and preserving task-frame metrics while stronger
  world-model backbones are integrated. The publishable RGB-D method should
  use or compare against a serious pretrained world-foundation-model backbone
  when available, with Cosmos-family physical-AI world models as the preferred
  first target because they are explicitly designed for physical-world
  prediction/post-training and have open checkpoints/tooling. Wan-family video
  models may be considered for visual prediction or data augmentation
  baselines, but they are not a sufficient controller world model unless the
  implementation proves action conditioning, RGB-D/robot-state grounding,
  task-slot decoding, uncertainty, and closed-loop controller benefit under the
  unchanged dynamic task-completion evaluation.
- Prioritize full RGB-D dataset generation, RGB-D slot/latent extraction, and
  RGB-D-conditioned world-model training before claiming controller success as
  method evidence.
- A controller run is not a method success until it uses RGB-D-derived slots or
  latent state for world-model prediction and task rebinding. State-only video
  demos are useful sanity checks but not final evidence.
- Treat the frozen DP as a static skill prior. Query it only when
  continuability is high enough.

## Explanation Standard

- Write plans, TODOs, and evidence notes so the causal story is clear to a
  non-expert reader: what changed in the world, what the robot observed, what
  was predicted, what bridge or handoff was chosen, and why the final task
  success or failure follows from the real final state.
- Prefer concrete task-frame quantities, job IDs, artifact paths, and videos
  over vague labels such as "small move", "large move", or "looks robust".
- Treat explanations as part of the method. State the simple physical reason
  for each added module or experiment, the exact evidence needed to validate
  it, and the condition that would falsify it. If a smart non-expert cannot
  follow the causal chain from dynamic event to final success/failure, the note
  is not clear enough.
