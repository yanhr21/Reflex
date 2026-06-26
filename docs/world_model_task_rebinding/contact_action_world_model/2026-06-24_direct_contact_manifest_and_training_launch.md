# 2026-06-24 Direct Contact Manifest And Training Launch

## Scope

This is the first active data/training step after the Policy-DROID
same-prefix replay confirmed handoff-positive but direct-insertion-negative
behavior. It follows the contact-action reset: train or adapt an executor that
can generate insertion/contact actions, not another scalar scorer.

No project compute was run on the login node. Manifest construction and
training were launched inside held Slurm allocation `148732` on `server24`
through tmux session
`policy_replay_cuda_repair_gpu_request_20260624_1801`.

## Cleanup

Moved `39` stale scorer/smoke/retrieval/oracle/debug experiment directories
from the active Cosmos3 tree to:

`experiments/_archive_20260623_contact_action_reset/cosmos3_stale_after_policy_droid_audit_20260624`

The move list was appended to:

`experiments/_archive_20260623_contact_action_reset/MOVED_DIRS.txt`

Active Cosmos3 top-level directories after this cleanup: `99`.

## Clean Manifest

Added:

`scripts/world_model/build_direct_contact_executor_manifest.py`

Output:

`experiments/world_model_task_rebinding/cosmos3/direct_contact_executor_manifest_h24_sourcepos_livehard_20260624_alloc148732`

Summary:

- action horizon: `24`
- total rows: `2906`
- source direct positives: `2905`
- Policy-DROID live hard negatives: `1`
- source split labels and future labels are targets only, not controller
  inputs
- summary says `ready_for_direct_contact_executor_training=true`

The direct positives are source insertion suffixes where the horizon-24 chunk
can reach insertion/contact-positive state. This directly targets the missing
physical behavior.

## Hard-Negative Gap

The builder attempted to include the `192` causal-suffix diffusion replay
labels from:

`live_snapshot_replay_causal_suffix_diffusion_panel0134_offsets64_48_32_24_16_8_s2_exec8_dp96_fix1_20260623_201146_alloc146658`

Those rows could not be converted into action-negative rows because the replay
labels point back to the original `candidate_action_bank.npz`, but the
synthetic `causal_suffix_diffusion_o*` action chunks were generated at replay
time and were not persisted in that bank.

Consequence: the labels remain valid consequence evidence, but they are not
clean action-training rows unless the synthetic chunks are regenerated or the
replay tool is rerun with generated chunks persisted.

## Training Launch

Added:

`scripts/world_model/train_direct_contact_executor_diffusion.py`

Output:

`experiments/world_model_task_rebinding/cosmos3/direct_contact_executor_diffusion_h24_sourcepos_1gpu1h_20260624_alloc148732`

Training inputs:

- manifest rows used: `2905` primary direct-positive source rows
- split: `2469` train / `436` val by source
- action target: horizon-24 robot action chunk
- feature count: `21`
- no scenario label, source id, future insertion frame, end contact state, or
  DP96 outcome is used as an input feature
- CUDA device: `NVIDIA H200`

The trainer is guarded against non-Slurm execution and uses
`min_wall_seconds=3660`. Until `training_summary.json` reports
`formal_one_gpu_hour_floor_met=true`, intermediate metrics are liveness only.

First launch status:

- stopped at `500000` steps;
- elapsed only `1312s`;
- `formal_one_gpu_hour_floor_met=false`;
- `stop_reason=max_steps`;
- this is an execution-parameter failure and is not valid training evidence.

Repair launch:

`experiments/world_model_task_rebinding/cosmos3/direct_contact_executor_diffusion_h24_sourcepos_1gpu1h_fix1maxsteps_20260624_alloc148732`

The repair uses the same manifest and training settings but raises
`--max-steps` to `5000000`, so the run should be stopped by the one-GPU-hour
wall-clock gate rather than the step cap.

Repair result:

- `formal_one_gpu_hour_floor_met=true`
- elapsed: `3660.05s`
- steps: `1396013`
- stop reason: `min_wall_and_min_steps`
- `ready_for_saved_snapshot_replay_gate=true`
- best validation `x0_action_mse_mid_t`: `0.02960`
- final validation `x0_action_mse_mid_t`: `0.04845`

The model clearly overfits if allowed to run to the final wall-clock point.
Therefore replay must use `checkpoint_best_eval.pt`, not
`checkpoint_final.pt`. This is valid training evidence, but not method
evidence until sampled chunks pass saved dynamic snapshot replay and visual
review.

## Saved-Snapshot Replay

Added:

`scripts/world_model/sample_direct_contact_executor_chunk.py`

First sampling issue:

- naive iterative DDIM sampling from the best checkpoint produced `1e4`-scale
  actions;
- the live replay label reported the query as `grasped=false`, while the
  source-positive training distribution is grasped;
- this was treated as a sampler/OOD query failure, not a replay result.

Repair sampling:

- mode: `x0_mid`
- forced query grasped: true
- action clipping: true
- sampled chunks: o24, o16, o8
- bounded action stats: mean abs about `0.20`, max abs `<=1`

Replay outputs:

- `direct_contact_executor_replay_best_x0mid_o24_sample00_iter00_f106_20260624_alloc148732`
- `direct_contact_executor_replay_best_x0mid_o8_sample00_iter00_f106_20260624_alloc148732`

Results:

- o24: no direct success, no inserted live pose, no contact-stable state, no
  conservative gate pass, DP96 failed; `delta_abs_yz_sum=+0.1305`
- o8: no direct success, no inserted live pose, no contact-stable state, no
  conservative gate pass, DP96 failed; `delta_abs_yz_sum=+0.1225`
- both preserved grasp

Conclusion: the source-positive-only direct-contact executor is not a valid
controller candidate for this saved dynamic snapshot. It appears to learn
source insertion thrust but mis-handles lateral alignment in the dynamic
post-motion state. The next data repair must persist generated live candidate
actions as hard negatives and add live/contact-corrective positives; otherwise
the executor will keep imitating source suffixes that are not aligned with the
actual changed task frame.

## Next Gate

When training completes, inspect `training_summary.json`. If the one-GPU-hour
floor is met and the checkpoint is sane, the next required step is saved
live-snapshot replay of sampled generated chunks. No live panel or method
claim is allowed before that replay and visual/final-state review.
