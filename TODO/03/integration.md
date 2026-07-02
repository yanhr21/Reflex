# Phase 03 TODO: DP Plus Cosmos Integration

- [ ] Open tmux-held interactive Slurm allocation.
- [ ] Create a run directory under `experiments/maniskill/runs/03_integration/`.
- [ ] Create a log file under `logs/03_integration/`.
- [ ] Start from reset.
- [ ] Run DP initial policy segment.
- [ ] Detect target/hole motion causally.
- [ ] Run RGB Cosmos-3 after target motion.
- [ ] Implement or select a first RGB task-state extractor.
- [ ] Extract from live RGB and Cosmos RGB:
      peg head, hole center, peg axis, hole axis, peg-to-hole vector, lateral
      error, axial error, angular error, near-hole flag, preinsert flag, and
      confidence.
- [ ] Save overlay images/video showing extracted task state on live frames and
      Cosmos future frames.
- [ ] Build a frame-aligned task-state chart with trigger frame and confidence.
- [ ] Generate DP-compatible candidate chunks:
      DP continuation, small task-frame residual, bounded insertion-axis push,
      bounded retreat-reapproach, and hold/reobserve.
- [ ] Reject any candidate that requires `set_pose`, state restore, future
      simulator labels, gripper-open finisher, or discontinuous peg motion.
- [ ] Score candidates by predicted progress, lateral/angular error,
      executability, trust, and discontinuity penalty.
- [ ] Save candidate table with selected and rejected candidates.
- [ ] Execute only a DP-compatible chunk if the trust gate permits execution.
- [ ] Reobserve after execution and compute prediction-observation discrepancy.
- [ ] Save live RGB, Cosmos future video, task-state overlays, candidate table,
      action chart, DP actions, selected action, rejected actions, trigger
      frame, trust score, and final state.
- [ ] Identify the first failing component:
      RGB future, extractor, candidate generator, scorer, trust gate,
      DP-compatible execution, or final contact/insertion.
- [ ] Do not apply Oracle in this phase.
