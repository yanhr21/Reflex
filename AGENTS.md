# Agent Rules

These rules apply to all future work in this repository. If older notes,
archived plans, TODOs, or experiment artifacts conflict with this file, this
file wins.

## Priority 0: Node And Resource Rules

- Never run Python project tasks on the login node.
- Never run data generation, rendering, rollout, replay, training, evaluation,
  preflight, imports, syntax checks, smoke tests, or project-code debugging on
  the login node.
- Login-node work is limited to downloads, `git clone`, `s-get`, read-only
  text/status inspection, documentation edits, file moves, and git bookkeeping
  requested by the user.
- Even for read-only inspection on the login node, do not launch broad
  repository-wide file enumeration or index scans such as
  `rg --files --hidden --follow --no-ignore`, especially through VS Code /
  extension tooling. Use targeted file reads, narrow `find` paths, and known
  artifact paths instead.
- All experiment compute must run inside a tmux-held interactive Slurm
  allocation.
- Do not use one-shot `sbatch` for experiments unless the user explicitly
  overrides this rule.
- If a running experiment command must be replaced, interrupt the foreground
  process inside the tmux allocation and reuse the held allocation.
- Do not release a held allocation merely because the foreground command
  changes.
- Real training evidence must run for at least `1 GPU x 1 hour` on real data.
  Shorter runs are smoke checks or diagnostics only.
- If resources are unavailable, reduce CPU cores, memory, walltime, or GPU
  count when scientifically acceptable so a valid allocation starts sooner.
- Dataset smoke / render / diagnostic jobs should request only `1 GPU` by
  default unless a specific script proves it needs more. Prefer lowering CPU
  count and memory first, for example to `1 CPU / 8G`, before changing the
  scientific scope of the run.
- Previously bad render nodes may be retried only for smoke / diagnostic
  allocation tests, and the node/result must be recorded in the run manifest
  or TODO. Do not treat a successful bad-node smoke as proof that all larger
  production jobs are safe on that node.
- Once a tmux-held Slurm request is queued with scientifically acceptable
  resources, keep monitoring and wait through normal scheduler estimate
  movement. Do not repeatedly cancel and relaunch merely because the estimated
  start time changes. Relaunch only when there is a concrete reason such as a
  bad node diagnosis, an invalid resource request, an explicit user
  instruction, or a clearly better test-only result for the same valid job.
- Do not request a GPU allocation just to wait while a required guard is still
  closed. If the next run is blocked by human review, missing data, missing
  runner code, or another documented gate, keep preparing guards / docs on the
  login node and report the blocker instead of holding an idle GPU.
- After a guarded run is allowed and the allocation request is valid, do not
  give up only because the GPU is not assigned immediately. Keep monitoring the
  tmux-held request; if relaunching is justified, first lower CPU, memory, or
  walltime while keeping the scientific protocol intact and using `1 GPU` by
  default.
- Default resource order after all required guards are open: request `1 GPU`;
  for smoke / render / diagnostics start from the smallest scientifically
  acceptable CPU, memory, and walltime request; if scheduling is slow, reduce
  CPU / memory / walltime before changing the experiment; retry previously bad
  nodes only as smoke / diagnostic tests with canary and manifest evidence.
- Treat a pending but valid `1 GPU` tmux-held Slurm request as something to
  monitor patiently, not as a failure. Only replace it after a concrete
  diagnosis shows the request is invalid, the selected node is unusable, or a
  lower CPU / memory / walltime request would preserve the same experiment and
  likely start sooner.
- After resources are acquired, keep GPU utilization above the cluster release
  threshold; target more than `30%` utilization during active work.
- Do not leave held GPU resources idle. If blocked, either run aligned work in
  the allocation, reuse it for the next valid step, or report the blocker.

## Priority 1: Active Research Route

Oracle is no longer the main experiment.

The active route is now Dream Diffusion Policy style joint learning for
ManiSkill `PegInsertionSide-v1`:

1. Build a new dataset for dynamic peg insertion research.
2. Jointly train ordinary Diffusion Policy behavior and Cosmos-3 future-state
   imagination in one aligned training route.
3. Use Cosmos-3 imagination to guide or condition the policy, then test live
   control without Oracle.

The research question is whether a world model, here RGB Cosmos-3, can be
trained with or coupled to a normal Diffusion Policy so that:

- the policy still outputs normal rollout actions that physically drive the
  peg / wooden stick into the hole;
- Cosmos-3 can imagine future visual state / latent state for the same
  rollout;
- imagined future state can improve action selection or recovery when the
  scene changes.

Do not spend new mainline effort trying to complete Oracle coverage. Oracle
artifacts are legacy diagnostics and may be inspected only as failure context
or upper-bound context when the user explicitly asks.

## Priority 2: Active Assets And Dataset Reset

Active existing assets:

- Original DP checkpoint:
  `experiments/maniskill/dp_checkpoint/run_90201/checkpoints/`
- RGB Cosmos-3 checkpoint:
  `experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam/`

The previous `fix3_733` data remains available only as reference / audit
context until the new dataset design is specified. It is not the default
training source for the new main route.

The new main route requires a newly constructed dataset. The exact collection
protocol is pending user instruction, but these constraints are already fixed:

- Do not train the new method on examples where the original DP checkpoint
  already had an opportunity to solve moving-object cases.
- Do not count target-assisted self-insertion, where the target / hole moves
  onto a mostly stationary peg, as useful success data.
- Do not include samples whose final success depends on simulator state edits,
  target snap, peg snap, source-state restore, saved-state replay, geometric
  final placement, future labels used as controller inputs, or manual hidden
  intervention.
- Every accepted trajectory must preserve the controller-facing action
  contract and must have inspectable action traces, RGB evidence, and final
  physical outcome.

Active dataset classes:

- A static expert: official 1000 state/action demos plus newly rendered RGB.
  This class can support DP BC / distillation, Cosmos static future learning,
  and insertion phase extraction after RGB/state timestamps are aligned.
- B dynamic RGB observation: moving target / moving hole / peg disturbance
  episodes, including failures. This class trains Cosmos future prediction and
  target-frame readout, not positive DP action imitation from failed actions.
- C frozen-DP dynamic failure: frozen official DP in dynamic scenes. This
  class supplies negative / discrepancy / infeasible / miss / jam /
  target-assisted labels.
- D future-frame cooperation teacher: legal controller rollouts that use
  ground-truth future target trajectory only for teacher generation. This
  class trains the moving-frame adapter and must be recorded as teacher-only.
- E Cosmos-predicted cooperation: the cooperation setting with Cosmos/readout
  predicted future target frames. This class is the deployed-method data and
  comes after A-D.

Do not mix these classes into one generic success dataset. Every sample must
record its class and allowed losses in a manifest.

Active dynamic-scene adapter:

- `scripts/world_model/active_dynamic_peg_adapter.py` is the active
  source-audited adapter for B/C/D/E target / hole motion.
- The adapter source may use only continuous kinematic-target commands plus
  logged motion traces. If runtime SAPIEN does not expose a kinematic-target
  command, the dynamic smoke must fail; do not fall back to per-step direct
  pose edits or simulator-state restoration.
- The adapter is not data, not a runner, not a success result, and not runtime
  evidence. It must be validated later inside a compute-node Slurm smoke that
  renders RGB, writes the action / motion trace, and passes human review
  before production.

## Priority 3: Required Experiment Order

Use only these active phase folders:

- `IDEA/01_dataset`, `PLAN/01_dataset`, `TODO/01_dataset`
- `IDEA/02_joint_training`, `PLAN/02_joint_training`,
  `TODO/02_joint_training`
- `IDEA/03_imagination_policy`, `PLAN/03_imagination_policy`,
  `TODO/03_imagination_policy`
- `IDEA/arxiv/dream_diffusion_policy`,
  `PLAN/arxiv/dream_diffusion_policy`,
  `TODO/arxiv/dream_diffusion_policy`

Do not create new `00_overview.md` files. Do not revive numbered Oracle phase
folders as active work. Old `PLAN/01` ... `PLAN/05`, `TODO/01` ...
`TODO/05`, and old idea notes belong under `legacy`.

The active order is:

1. Dataset construction.
2. Joint training, starting with an overfit experiment and then moving to full
   training.
3. Cosmos-imagination-guided policy control.

## Priority 4: Dream Diffusion Policy Reference Rule

Use Dream Diffusion Policy as a research template, not as a claimed official
reproduction unless official code and weights are found and integrated.

Reference paper:

- `Dreaming the Unseen: World Model-regularized Diffusion Policy for
  Out-of-Distribution Robustness`, arXiv:2603.21017, 2026-03-22.

Current reading:

- The paper co-optimizes a diffusion policy and a diffusion world model through
  a shared encoder / aligned latent representation.
- The world model predicts future observation latents while the policy learns
  behavior cloning over action chunks.
- Inference uses real-imagination discrepancy and an imagination loop when
  observations become unreliable or out-of-distribution.
- Current search did not find an official DDP implementation. Until that
  changes, this repository should implement a DDP-inspired route using the
  real DP checkpoint / architecture and real Cosmos-3 checkpoint, clearly
  labeled as our adaptation.

Do not hand-write a toy VAE, toy MLP, toy Transformer, toy diffusion policy,
toy expert, or toy world model and present it as DDP, Cosmos-3, DP, or paper
method progress. Simplified checks may be used only when explicitly labeled as
diagnostics.

## Priority 5: Joint Training Standard

The joint-training implementation must start with an overfit experiment before
full training.

The overfit experiment must prove both objectives on a tiny real dataset slice:

- DP objective: the policy can output a coherent `pd_ee_delta_pose` rollout
  action chunk that physically controls the peg / wooden stick toward
  insertion.
- Cosmos objective: Cosmos-3 can imagine or predict future visual / latent
  state for the same rollout window.

Full training is allowed only after the overfit run has inspectable evidence:

- dataset manifest;
- model/config manifest;
- action loss or rollout-action sanity evidence;
- future-state imagination loss or visual/latent prediction evidence;
- at least one rendered review or equivalent artifact showing that the action
  and imagined future refer to the same physical episode.

## Priority 6: Imagination-Guided Policy Standard

The final method direction is not Oracle and not manual finishing. The method
must make Cosmos imagination controller-facing.

Acceptable controller-facing interfaces include:

- shared latent conditioning between Cosmos-3 and DP;
- Cosmos-predicted future latent or RGB features as policy condition;
- real-imagination discrepancy as a trust / OOD signal;
- autoregressive imagined latents that condition future policy action chunks;
- a documented Cosmos-derived adapter that outputs DP-compatible
  `pd_ee_delta_pose` action chunks.

Deployed method claims require:

- reset-to-end live ManiSkill rollout;
- RGB observation history and proprioception used causally;
- no simulator-state edits or hidden future controller inputs;
- DP-compatible controller actions or another explicitly approved controller;
- final success measured from live simulator state;
- rendered visual review for contact / insertion claims.

## Priority 7: Invalid Success Rule

Do not call state intervention success.

- `set_pose`, `set_state`, `set_state_dict`, source-state restore,
  saved-state replay, geometric final placement, future labels used by the
  controller, or hand-selected suffixes are not physical insertion success.
- Invalid physical behavior includes wall penetration, insertion into the wall
  instead of the hole, teleportation / snapping, disappearing objects, and
  discontinuous pose jumps.
- Teleportation / snapping is invalid for the peg, wooden stick, target/hole,
  robot, or any other task object.
- If the peg is far from the hole and then snaps into the hole, describe it as
  invalid state-intervention smoke.
- Do not count a moving-target sample as success if the target / hole moves
  onto the peg / wooden stick and creates insertion by itself. The robot must
  actively drive insertion through controller actions.
- Before reporting any insertion success, inspect the manifest, summary,
  command line, wrapper script, relevant Python code path, action trace, and
  video for state intervention or target-assisted self-insertion.

## Priority 8: Repository Hygiene

- Do not `git commit` or `git push` unless the user explicitly asks in the
  current conversation.
- Keep large checkpoints, generated data, cache directories, rendered media,
  and archived experiment dumps out of git unless explicitly requested.
- Do not delete important data or checkpoints without explicit user direction.
- Active assets live under `experiments/maniskill/`.
- Old experiment outputs live under `experiments/legacy/`.
- New active outputs go under
  `experiments/maniskill/runs/<stage>/<case_or_exp>/<try_or_tag>/`.
- New logs go under `logs/<stage>/<case_or_exp>/<try_or_tag>.log`.
- Legacy logs live under `logs/legacy/`.
- Use short, human-readable run paths. Put shared prefixes into directories
  and keep leaf names to only the differing role or attempt.
- Do not concatenate metadata into names. Put timestamps, job ids, hostnames,
  checkpoint paths, source keys, and command lines in `manifest.txt` or
  `summary.json`.
- If an experiment is proven useless, wrong, invalid, superseded, or
  misleading, move it under `experiments/legacy/` or the external archive path
  requested by the user before continuing.
- RGB dataset production must be smoke-first. Render a small review slice,
  mark `human_review_required=true`, and stop before large-scale production
  until the user approves the visual quality. Multiple B/C/D/E smoke runs may
  be generated before asking for review so the user can inspect them in one
  batch; do not scale any class to production until that class's smoke is
  approved. When waiting for visual review, report the goal as blocked on
  human review rather than silently scaling up.
- New RGB dataset videos must default to `30 FPS`. If a run uses any other
  FPS for a concrete downstream contract, record that reason in the manifest.
- For dataset RGB smoke, request only `1` GPU. If an immediate allocation is
  unavailable, reduce CPU cores, memory, and walltime when scientifically
  acceptable, then keep a tmux-held queued allocation rather than abandoning
  the smoke. Previously bad render nodes are not a permanent blacklist for
  dataset smoke; they may be tried on a small smoke only, and any failure must
  be classified from that run's manifest/log/artifacts before scaling.
- Dataset RGB smoke must run a minimal render canary before replaying official
  trajectories. The active canary is
  `scripts/world_model/render_min_canary.py` with `shader_pack=minimal`.
  The official replay smoke should pass `--shader minimal` unless a newer
  render document supersedes this. If the canary fails, do not start replay on
  that allocation; classify the render failure from the canary log/artifacts
  and archive the failed attempt before retrying.

## Priority 9: Main Code Map

Existing code that remains relevant:

- DP baseline / compatibility:
  `scripts/training/train_dp_state_ddp.py`,
  `scripts/training/dp_eval_compat.py`,
  `scripts/world_model/run_dp_full_episode_baseline_panel.py`
- Cosmos-3 input/action utilities:
  `scripts/world_model/build_cosmos3_live_prefix_wam_input.py`,
  `scripts/world_model/extract_cosmos3_policy_action_chunk.py`,
  `scripts/world_model/phase02_extract_rgb_v2v_inputs.py`
- Cosmos-3 dataset / executor training utilities:
  `scripts/world_model/build_cosmos3_contact_executor_dataset.py`,
  `scripts/world_model/build_cosmos3_executor_training_dataset.py`,
  `scripts/world_model/train_cosmos3_contact_executor.py`,
  `scripts/world_model/train_cosmos3_executor_overfit.py`
- Video / artifact inspection helpers:
  `scripts/world_model/video_contract_utils.py`,
  `scripts/world_model/inspect_video_artifacts.py`,
  `scripts/world_model/make_video_framebooks.py`

New mainline code should be added around the new route, not inside Oracle
wrappers:

- dataset builder for the new dynamic joint-training dataset;
- overfit trainer for joint DP action and Cosmos future-state objectives;
- full trainer after overfit evidence exists;
- evaluation wrapper for imagination-guided policy control.

Current active dataset smoke entry points:

- `scripts/slurm/launch_dataset_static_rgb_smoke_tmux.sh` starts the Stage 1
  official static RGB smoke from the login node by creating a tmux-held Slurm
  allocation.
- `scripts/slurm/run_dataset_static_rgb_smoke_in_allocation.sh` performs the
  actual render inside a Slurm `srun` step only and refuses login-node
  execution.

Oracle-specific scripts under `scripts/slurm/phase03_*`,
`scripts/slurm/run_phase03_oracle*`, and
`scripts/world_model/phase03_*` are legacy diagnostic context unless the user
explicitly asks to inspect or archive them.

## Priority 10: Script And Artifact Guard

- Before running any existing wrapper or experiment script, inspect it for old
  routes and state-intervention paths. If it references old non-active routes,
  Oracle-only launchers, `set_pose`, `set_state`, `set_state_dict`,
  source-state restore, saved-state replay, future labels used by the
  controller, geometric final-seat, scorer-only replay, or hand-selected
  suffixes, do not run it as an active method script until patched or
  explicitly approved as diagnostic-only.
- New active scripts must write outputs only under the new active run layout
  and logs only under the new active log layout.
- Every new run must write a manifest recording command, allocation, node,
  data source, checkpoint, controller/action contract, output path, evidence
  type, and whether the result is method evidence or diagnostic evidence.
- Active RGB render scripts must set
  `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`, `DISPLAY=`, and
  `HDF5_USE_FILE_LOCKING=FALSE` unless a newer render document supersedes
  these values.
