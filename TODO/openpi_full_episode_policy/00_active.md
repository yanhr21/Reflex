# OpenPI Full-Episode Policy TODO

## Active Objective

Make official OpenPI/pi0.5 replace Diffusion Policy from episode step `0` for
PegInsertionSide. Static scenes must be full-episode OpenPI-only. Dynamic
scenes must start with OpenPI and then use a causal world model to provide
future scene/X-chat/task conditioning for OpenPI after observed scene change.

## Immediate Cleanup And Guardrails

- [ ] Keep `PLAN/legacy/` and `TODO/legacy/` as historical archives only.
- [ ] Confirm no active plan/TODO points to saved-snapshot takeover as the
      current method.
- [ ] Add or update run manifests so every OpenPI artifact is labeled as one
      of: full-episode live rollout, training, conversion/audit, offline
      inference diagnostic, or legacy takeover diagnostic.
- [ ] Keep checkpoint/data/cache/media out of git; commit only scripts,
      manifests, text logs, JSON summaries, plans, TODOs, and evidence notes.

## Data And OpenPI Compatibility

- [ ] Audit the 733 accepted ManiSkill PegInsertionSide source trajectories for
      301 RGB/state frames and 300 action steps.
- [ ] Identify the currently valid OpenPI/LeRobot export for the 733 data and
      record whether it preserves the full-episode contract.
- [ ] Verify the OpenPI action target convention for `pd_ee_delta_pose`.
- [ ] Prefer `pi05_base` with fresh normalization stats unless a concrete
      adapter proves another OpenPI checkpoint action convention is compatible.
- [ ] Record any blocker as a compatibility issue, not as permission to build
      a custom placeholder policy.

## Static OpenPI-Only Evaluation

- [ ] Train or resume official OpenPI/pi0.5 on the full 733-derived data inside
      a tmux-held interactive Slurm GPU allocation.
- [ ] Ensure the run reaches at least one GPU-hour before calling it training
      evidence.
- [ ] Run full-episode OpenPI rollouts from reset/state step `0`; no DP prefix.
- [ ] Save metrics, final simulator state, videos/contact sheets, and a run
      manifest.
- [ ] Inspect visual evidence for grasp, peg hold, alignment, insertion, and
      final task success.
- [ ] Produce a concise static result table separating success, near miss,
      physical failure, and implementation failure.

## Dynamic World-Model-Conditioned OpenPI

- [ ] Define the causal dynamic event interface: observed history/current
      perception only.
- [ ] Implement a causal target-motion detector for target/hole movement from
      observed history; do not use sample labels or future ground-truth state
      to decide controller mode.
- [ ] Implement a world-model activation/continuability gate: if no target
      motion is detected and OpenPI remains task-continuable, keep OpenPI
      running without world-model conditioning; after observed motion or
      uncertainty, activate world-model conditioning.
- [ ] Define legal prefix semantics: prefix frames must be real observations
      from the same live episode up to the current time. Different-frame
      prefix rollout is allowed for WM diagnostics and receding-loop testing,
      but not as full-episode OpenPI success evidence.
- [ ] Define the world-model output consumed by OpenPI: future scene summary,
      future X-chat/task text, and any allowed task-state condition.
- [ ] Build the receding execution loop: observe, world-model imagine,
      condition OpenPI, execute short horizon, re-observe.
- [ ] Verify that OpenPI remains the only robot action policy in the loop.
- [ ] Run dynamic live rollouts only after the static OpenPI-only path has
      working evidence or a concrete failure diagnosis.
- [ ] Save dynamic videos/contact sheets, metrics, manifests, and failure
      labels.

## Evidence Gates

- [ ] Do not report snapshot takeover as static or dynamic method success.
- [ ] Do not report arbitrary saved-prefix rollout as deployed dynamic
      success. A valid dynamic run must start with OpenPI at reset, observe the
      target motion online, activate the world model causally, and continue
      with OpenPI actions.
- [ ] Do not report offline action inference as live execution success.
- [ ] Do not report a world model as providing OpenPI actions; the world model
      provides future scene/X-chat/task conditioning.
- [ ] Do not report short smoke runs as method evidence.
- [ ] Every claimed success must include the exact checkpoint, data export,
      rollout command, frame/action contract, and visual artifact path.

## Next Execution Order

1. Re-audit current OpenPI data/export/training artifacts against the new
   full-episode protocol.
2. Identify the latest usable official OpenPI checkpoint and normalization
   state for 733-derived training.
3. Launch or resume the correct static OpenPI-only training/evaluation path in
   a held Slurm allocation.
4. Inspect and summarize static visual cases.
5. Only then implement the dynamic world-model-conditioned OpenPI loop.
