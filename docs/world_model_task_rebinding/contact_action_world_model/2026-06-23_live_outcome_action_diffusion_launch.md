# 2026-06-23 Live-Outcome Action Diffusion Launch

## Purpose

This is the next experiment after the deterministic source-suffix MLP failed.
It trains from real saved live candidate outcomes rather than only successful
source suffixes. The physical target is the missing contact/insertion action
coverage: generate short action residuals from the current live state, task
path, DP prior, and contact context, while learning consequence/value heads
from real `candidate + DP96` labels.

## Launch

Launched inside the tmux-held interactive Slurm allocation:

- Slurm job: `146658`
- Slurm step: `150`
- host: `server56`
- wrapper:
  `scripts/slurm/run_live_outcome_action_diffusion_train_in_allocation.sh`
- trainer:
  `scripts/world_model/train_live_outcome_action_diffusion.py`
- output:
  `experiments/world_model_task_rebinding/cosmos3/live_outcome_action_diffusion_full_live_union_1gpu1h_20260623_175110_alloc146658`

## Data

Input live outcome union:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_source_suffix_union_plus_panel0245_plus0204fixuuid_dp96_20260623_alloc146658`

Training manifest reports:

- candidate outcome rows: `9349`
- live-state groups: `127`
- train/val rows: `7736/1613`
- positive rows: `146`
- hard positive rows: `138`
- feature dim: `554`
- action residual dim: `168`
- candidate families:
  - `checkpoint_model`: `8053`
  - `dp_prior`: `127`
  - `retrieval_success_residual`: `318`

Read-only data audit after launch:

- base live rows: `127`
- distinct `source_uuid`: `4`
- scenarios: `hole_late_continuous_insert=20`,
  `hole_late_move_stop=24`, `hole_late_reverse=40`,
  `hole_late_sine=43`
- `current_phase=no_grasp` for all `127` base rows
- `task_path_source=cosmos_predicted_action_sidecar` for all `127` base rows
- outcome rows: `9349`
- direct final successes: `0`
- direct inserted-pose positives: `0`
- `candidate + DP96` handoff successes: `138`
- live-state groups with any handoff success: `44/127`
- live-state groups where `dp_prior` already succeeds: `25/127`

This means the run is learning from DP-handoff/continuability labels, not from
direct insertion-completion positives. It can still be useful as a diagnostic
generator, but it does not by itself solve the "generate an inserted action"
target. The data split is also weak for broad generalization because the live
states come from only four source trajectories.
  - `teacher_scale`: `851`

## Boundary

This is not a live result. It is a one-GPU-hour training gate for a
live-outcome-conditioned action generator. A useful result must still pass
saved-snapshot generated-candidate replay, then full `301/300` live panels with
video/contact-sheet and final-state inspection.

Early metrics show that the selector/value head has not yet beaten DP prior on
held-out live groups; the run must be allowed to finish before interpretation.

At `2026-06-23 18:03 CST`, the run had reached `680` seconds and was still
running. Held-out group selection was tied with DP prior rather than better:
selected handoff success fraction `0.32`, DP prior `0.32`, oracle over the
existing candidate pool `0.44`. This remains an in-progress training run, not
a result.

At `2026-06-23 18:13 CST`, the run had reached `1290` seconds. Held-out
selection was still tied with DP prior at `0.32`, with oracle still `0.44`.
The diffusion loss had decreased (`0.159`), but the consequence/value selection
had not yet improved over DP.

At `2026-06-23 18:27 CST`, the run had reached `2109` seconds. Held-out
selection had briefly improved to `0.36` versus DP prior `0.32`, with oracle
still `0.44`. This is not yet a result because the run has not reached the
one-GPU-hour floor, and the metric is still selection over an existing replayed
candidate pool rather than generated-action replay.

At `2026-06-23 18:35 CST`, the run had reached `2630` seconds. Held-out
selection returned to a tie with DP prior: selected `0.32`, DP `0.32`, oracle
`0.44`.

At `2026-06-23 18:45 CST`, the run had reached `3245` seconds. Held-out
selection remained tied with DP prior at `0.32`, while value MSE had worsened
to `14.83`.

## Result

The run completed at `2026-06-23 18:54 CST` and wrote:

`experiments/world_model_task_rebinding/cosmos3/live_outcome_action_diffusion_full_live_union_1gpu1h_20260623_175110_alloc146658/training_summary.json`

Summary:

- `formal_one_gpu_hour_floor_met=true`
- `elapsed_seconds=3660.23`
- `steps=224776`
- `stop_reason=min_wall_and_min_steps`
- `ready_for_saved_snapshot_replay_gate=false`
- final held-out groups: `25`
- final selected handoff success fraction: `0.32`
- final DP-prior handoff success fraction: `0.32`
- final oracle handoff success fraction over existing candidates: `0.44`
- final selected-minus-DP: `0.0`
- final value MSE: `14.38`
- best value-MSE checkpoint was at step `1`

This is a formal negative/limited result. The one-GPU-hour training floor was
met, but the learned selector/generator did not beat DP prior on held-out
live-state groups and did not pass the saved-snapshot replay readiness gate.
The checkpoint should not be promoted to generated-action replay or live panel
evaluation unless it is explicitly needed as a negative diagnostic.

The result also reinforces the data audit: this label union contains no direct
inserted/successful live positives, so training on it alone is not enough to
teach physical insertion. The next aligned repair is stronger contact-action
supervision or a real WAM/base-policy post-training path, not another scalar
scorer threshold.
