# Agent Rules

These rules apply to all future work in this repository. If older notes,
archived plans, or experiment artifacts conflict with this file, this file
wins.

## Priority 0: Node And Resource Rules

- Never run Python project tasks on the login node.
- Never run data generation, rendering, rollout, replay, training, evaluation,
  preflight, imports, syntax checks, smoke tests, or project-code debugging on
  the login node.
- Login-node work is limited to downloads, `git clone`, `s-get`, read-only
  text/status inspection, documentation edits, file moves, and git bookkeeping
  requested by the user.
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
- After resources are acquired, keep GPU utilization above the cluster release
  threshold; target more than `30%` utilization during active work.
- Do not leave held GPU resources idle. If blocked, either run aligned work in
  the allocation, reuse it for the next valid step, or report the blocker.

## Priority 1: Active Route

The active route is ManiSkill `PegInsertionSide-v1`.

Active experiment assets:

- Approved 733 ManiSkill data:
  `experiments/maniskill/data/fix3_733/`
- Original trained DP checkpoint:
  `experiments/maniskill/dp_checkpoint/run_90201/checkpoints/`
- RGB Cosmos-3 checkpoint:
  `experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam/`

Do not use or revive archived experiment routes unless the user explicitly
asks. Old failed runs, superseded non-active routes, geometric final-seat
smokes, source-state restore, saved-state replay, scorer diagnostics, and
stale executor variants are archive context only.

## Priority 2: Required Experiment Order

Run experiments in this order:

1. Reproduce whether the DP checkpoint works on static ManiSkill.
2. Check whether RGB Cosmos-3 can imagine future video and produce an
   action/task chart.
3. Combine DP and Cosmos to locate the exact failing step, with visualization.
4. Test the corrected Oracle boundary.
5. Test live control without Oracle.

Do not jump to Oracle or live control before Phases 01 and 02 are understood.

Phase docs must use short numbered folders only:

- `PLAN/01` ... `PLAN/05`
- `TODO/01` ... `TODO/05`

Do not create date-prefixed phase folders or long nested plan names.

## Priority 3: Oracle Boundary

The corrected Oracle boundary is:

1. start from reset;
2. execute the initial policy segment with the existing DP checkpoint;
3. detect target / hole motion causally from live observations;
4. run RGB Cosmos-3 imagination from live RGB history after target motion is
   detected;
5. only after Cosmos output exists, allow an explicitly labeled Oracle
   final-seat step.

Required Oracle evidence:

- command, allocation, node, checkpoint paths, and output path;
- target-motion trigger frame;
- Cosmos RGB input/output paths;
- before-oracle and after-oracle `peg_head_at_hole`;
- oracle jump distance;
- rendered video with the oracle moment annotated.

Oracle is an upper-bound / pipeline diagnostic only. It must be recorded with
`method_evidence_allowed=false` and must not be reported as deployed method
success.

## Priority 4: Invalid Success Rule

Do not call state intervention success.

- `set_pose`, source-state restore, saved-state replay, geometric final
  placement, future labels, or hand-selected suffixes are not physical
  insertion success.
- If the peg is far from the hole and then snaps into the hole, describe it as
  invalid state-intervention smoke.
- Specifically invalid as active success:
  `live_geom_seat_alltypes_rgb_success_20260701_server64`,
  `live_geometric_final_seat_alltypes_smoke_*`, and
  `oracle_final_seat_demo_f286_20260701_server29_fix`.
- Current valid physical reset-to-end live insertion success count is `0`
  unless a new run proves otherwise under the active protocol.
- Before reporting any insertion success, inspect the manifest, summary,
  command line, wrapper script, relevant Python code path, action trace, and
  video. If any controller-facing path uses `set_pose`, `set_state`,
  `set_state_dict`, source env-state restore, saved-state replay, geometric
  final placement, future labels, hand-selected suffixes, or manual takeover,
  the result is not physical success.
- A valid success report must include before/after task distances, the
  commanded action trace around insertion, whether there was a discontinuous
  pose jump, final simulator success state, and annotated video evidence. A
  metric flipping true after state editing is invalid.

## Priority 4.5: Physics / Git Attribution Rule

Do not blame physics, Git, or external tooling without direct evidence.

- Git is version control. It cannot physically move a peg, alter simulator
  dynamics at runtime, or make a video show a teleport unless the checked-out
  code being executed contains that behavior.
- ManiSkill physics is not considered failed when the code directly calls
  `set_pose`, `set_state`, `set_state_dict`, source-state restore, saved-state
  replay, or another simulator-state edit. In that case the cause is
  implementation / protocol misuse, not physics.
- The previous snap / suction artifact was caused by explicit simulator-state
  intervention in a diagnostic final-seat path. It must be recorded as an
  invalid state-intervention artifact, not as physical insertion, not as Git
  failure, and not as a ManiSkill physics failure.
- If a real physics-engine failure is suspected, prove it with a minimal
  controlled reproduction inside a compute-node allocation, logging the exact
  code path, state before and after, action vector, contact information, and
  video. Until that evidence exists, classify discontinuous seating as a
  state-intervention or evaluation-implementation failure.

## Priority 5: RGB Cosmos-3 Requirement

- Cosmos-3 must use RGB evidence.
- Do not replace RGB Cosmos-3 imagination with a state-only world model, toy
  dynamics model, simulator-state shortcut, or hidden future label.
- The DP checkpoint may be state-based; that is acceptable as the base /
  finisher controller.
- Do not spend time reproving DP by itself unless it is needed for the Oracle
  run or a later live protocol.

## Priority 6: Live Method Standard

After Oracle validation, deployed method claims require:

- reset-to-end live ManiSkill rollout;
- causal target-motion detection;
- RGB Cosmos-3 imagination from live RGB history;
- DP-compatible `pd_ee_delta_pose` actions or another explicitly approved live
  controller;
- final success measured from live simulator state;
- rendered visual review for contact/insertion claims;
- no `set_pose`, source-state restore, saved-state replay, future labels, or
  hand-selected suffixes.

## Priority 7: Official-Method Requirement

- Preserve the real DP checkpoint and `PegInsertionSide-v1` /
  `pd_ee_delta_pose` contract.
- Preserve the RGB Cosmos-3 checkpoint and official loading path.
- Do not hand-write a toy VAE, MLP, Transformer, diffusion policy, expert, or
  world model and present it as method progress.
- Simplified checks may be used only when explicitly labeled as diagnostics.

## Priority 8: Repository Hygiene

- Do not `git commit` or `git push` unless the user explicitly asks in the
  current conversation.
- Keep large checkpoints, generated data, cache directories, rendered media,
  and archived experiment dumps out of git unless explicitly requested.
- Do not delete important data or checkpoints without explicit user direction.
- Archive by moving under `/public/home/yanhongru/ICLR2027/archive/Reflex/`
  and preserving the original repository-relative structure. Do not create or
  use `/public/home/yanhongru/ICLR2027/Reflex/archive/`.
- Keep active Plan/TODO roots clean: `PLAN/00_overview.md` and
  `TODO/00_active.md`.
- Keep `experiments/` visually clean. Active assets live under
  `experiments/maniskill/`; new outputs go under
  `experiments/maniskill/runs/<phase>/`.
- Put all new run logs under `logs/<phase>/`.
- If an experiment is proven useless, wrong, invalid, or misleading, move it
  under `/public/home/yanhongru/ICLR2027/archive/Reflex/` after
  classification instead of leaving it in `experiments/`.

## Priority 9: Script And Artifact Guard

- Before running any existing wrapper or experiment script, inspect it for old
  routes and state-intervention paths. If it references
  `experiments/world_model_task_rebinding`, `experiments/dp_peg1000`,
  OpenPI / LIBERO / robosuite / truepeg paths, `set_pose`, `set_state`,
  `set_state_dict`, source-state restore, saved-state replay, future labels,
  geometric final-seat, scorer-only replay, or hand-selected suffixes, do not
  run it as an active method script until it is patched or archived.
- Stale scripts may remain as archive context only. They are not executable
  active protocol unless their paths, evidence rules, and state-intervention
  behavior have been reviewed and updated.
- New active scripts must write outputs only under
  `experiments/maniskill/runs/<phase>/<short_run_id>/` and logs only under
  `logs/<phase>/`.
- Every new run must write a manifest recording command, allocation, node,
  data source, checkpoint, controller/action contract, output path, evidence
  type, and whether `method_evidence_allowed` is true or false.
