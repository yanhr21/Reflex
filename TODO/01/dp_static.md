# Phase 01 TODO: DP Static Check

- [ ] Open tmux-held interactive Slurm allocation.
- [ ] Create a run directory under `experiments/maniskill/runs/01_dp_static/`.
- [ ] Create a log file under `logs/01_dp_static/`.
- [ ] Record allocation, node, command, env, checkpoint path, and output path.
- [ ] Load
      `experiments/maniskill/dp_checkpoint/run_90201/checkpoints/best_eval_success_at_end.pt`.
- [ ] Instantiate static `PegInsertionSide-v1` with `pd_ee_delta_pose`.
- [ ] Run small rendered reset-to-end static rollout.
- [ ] Save final simulator state, metric JSON, video, and contact sheet.
- [ ] If a rollout succeeds, extract insertion-frame action statistics:
      xyz/rpy/gripper magnitudes, gripper channel, peg-to-hole distance before
      seating, and end-effector direction in task frame.
- [ ] Save the action statistics for Phase 03/05 executability checks.
- [ ] Mark pass/fail in the phase log.
- [ ] If failed, classify the blocker before starting Phase 02.
