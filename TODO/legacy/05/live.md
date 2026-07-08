# Phase 05 TODO: Live Controller

- [ ] Open tmux-held interactive Slurm allocation.
- [ ] Create a run directory under `experiments/maniskill/runs/05_live/`.
- [ ] Create a log file under `logs/05_live/`.
- [ ] Start from reset.
- [ ] Run DP initial policy.
- [ ] Detect target/hole motion causally.
- [ ] Run RGB Cosmos-3 from live RGB history.
- [ ] Extract controller-facing task state:
      peg/hole displacement, insertion axis, near-hole/preinsert/contact gates,
      confidence, and trust score.
- [ ] Generate DP-compatible candidates:
      DP continuation, residual-DP, bounded insertion-axis push,
      retreat-reapproach, and hold/reobserve.
- [ ] Score candidates by task progress, executability, and trust.
- [ ] Select exactly one chunk and log all rejected chunks.
- [ ] Execute only logged `pd_ee_delta_pose` actions.
- [ ] Use adaptive chunk length:
      far from hole up to DP horizon, near-hole 1-2 steps, contact/insertion 1
      step and reobserve.
- [ ] Reobserve after each chunk and update prediction-observation discrepancy.
- [ ] Save live RGB video, Cosmos prediction video, task-state overlays,
      candidate table, trust-gate output, action chart, and final state.
- [ ] Compare live finisher action magnitudes against successful static DP
      insertion action magnitudes.
- [ ] Claim success only if final insertion is reset-to-end live physical
      success with no `set_pose`, no source-state restore, no saved-state
      replay, and no future labels.
