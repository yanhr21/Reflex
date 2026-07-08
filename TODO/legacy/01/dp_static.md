# Phase 01 TODO: DP Static Check

- [x] Open tmux-held interactive Slurm allocation.
- [x] Create a run directory under `experiments/maniskill/runs/01_dp_static/`.
- [x] Create a log file under `logs/01_dp_static/`.
- [x] Record allocation, node, command, env, checkpoint path, and output path.
- [x] Load
      `experiments/maniskill/dp_checkpoint/run_90201/checkpoints/best_eval_success_at_end.pt`.
- [x] Instantiate static `PegInsertionSide-v1` with `pd_ee_delta_pose`.
- [x] Run small rendered reset-to-end static rollout.
- [x] Save final simulator state, metric JSON, video, and contact sheet.
- [x] If a rollout succeeds, extract insertion-frame action statistics:
      xyz/rpy/gripper magnitudes, gripper channel, peg-to-hole distance before
      seating, and end-effector direction in task frame.
- [x] Save the action statistics for Phase 03/04/05 executability checks.
- [x] Mark pass/fail in the phase log.
- [x] If failed, classify the blocker before starting Phase 02.

Current evidence:

- `experiments/maniskill/runs/01_dp_static/p01_static_trace3_20260703_003237_162153_server64/`
  is a partial pass: seed 2 succeeds physically at step 173, seeds 3 and 4
  fail by step 300.
- `experiments/maniskill/runs/01_dp_static/p01_static_trace_20260703_002848_162153_server64/`
  is a valid failed diagnostic for seed 1.
- The aborted 10-episode attempt was archived under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/experiments/maniskill/runs/01_dp_static/`.
