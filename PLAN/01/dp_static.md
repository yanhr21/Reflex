# Phase 01: DP Static Check

Goal: verify the original DP checkpoint still works on static ManiSkill
`PegInsertionSide-v1`.

Assets:

- DP checkpoint:
  `experiments/maniskill/dp_checkpoint/run_90201/checkpoints/best_eval_success_at_end.pt`
- DP run manifest:
  `experiments/maniskill/dp_checkpoint/run_90201/manifest.json`
- Data reference:
  `experiments/maniskill/data/fix3_733/`

Steps:

1. Start a tmux-held interactive Slurm allocation.
2. Record allocation, node, conda/env, repo revision, command, and GPU state
   under `logs/`.
3. Load the DP checkpoint through the real DP code path.
4. Instantiate static `PegInsertionSide-v1` with `pd_ee_delta_pose`.
5. Run a small rendered static rollout panel from reset.
6. Save metrics, rendered video/contact sheet, and final simulator state.
7. Extract successful insertion action statistics if any static rollout
   succeeds:
   - insertion frame range;
   - `pd_ee_delta_pose` xyz/rpy/gripper action magnitudes;
   - gripper channel values near insertion;
   - peg-to-hole distance before seating;
   - end-effector direction in task frame.
8. Classify the result as pass/fail before any Cosmos or Oracle work.

Pass condition:

- reset-to-end static rollout reaches real simulator success;
- no source-state restore, saved-state replay, future label, or `set_pose`;
- rendered video is visually consistent with physical insertion.
- successful insertion action statistics are saved for Phase 03/05
  executability checks.

Failure handling:

- classify as checkpoint loading, observation/action mapping, physics,
  rendering, or environment issue;
- do not proceed to Phase 02 as a formal chain until this is understood.
