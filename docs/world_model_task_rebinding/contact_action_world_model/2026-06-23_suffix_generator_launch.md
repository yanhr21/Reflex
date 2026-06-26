# 2026-06-23 Source-Suffix Contact-Action Generator Launch

## Purpose

This is the first experiment after the contact-action reset. It does not try
to make a better scorer. It trains a generator for insertion suffix action
chunks mined from the accepted 733 H5 source set, so the next controller branch
has an actual contact/insertion action source to test on saved live failure
states.

## Launch

Training was launched inside the tmux-held interactive GPU allocation:

- Slurm job: `146658`
- Slurm step: `149`
- host: `server56`
- wrapper:
  `scripts/slurm/run_contact_action_suffix_generator_train_in_allocation.sh`
- output:
  `experiments/world_model_task_rebinding/cosmos3/contact_action_suffix_generator_full733_1gpu1h_20260623_163847_alloc146658`

The run uses the full source insertion suffix bank:

- source bank:
  `experiments/world_model_task_rebinding/cosmos3/source_insertion_suffix_bank_20260622_alloc145920/source_insertion_suffix_bank.npz`
- suffix rows: `4711`
- train/val split: `4001/710`, split by source UUID
- action horizon: `96`
- visible GPU count: `1`
- minimum wall time: `3660` seconds

This satisfies the new execution boundary only if it actually reaches the
minimum wall time and writes `training_summary.json`. Until then, it is a
running experiment, not a completed result.

## Result

The run completed at `2026-06-23 17:40 CST` and wrote:

`experiments/world_model_task_rebinding/cosmos3/contact_action_suffix_generator_full733_1gpu1h_20260623_163847_alloc146658/training_summary.json`

Summary:

- `formal_one_gpu_hour_floor_met=true`
- `elapsed_seconds=3661.45`
- `steps=567001`
- `stop_reason=min_wall_and_min_steps`
- `ready_for_saved_snapshot_replay_gate=false`
- final `eval_action_mse=0.0162008`
- final mean-action baseline MSE: `0.0155657`
- final `train_action_mse=1.3e-5`

This is a formal negative diagnostic for the deterministic source-suffix MLP:
it overfit the train split and finished worse than the mean-action baseline on
held-out source UUIDs. It should not be promoted to live panel evaluation.

## Early Observation

The first process started successfully and wrote metrics from the compute
node. Early validation MSE briefly improved over the mean-action baseline, then
the model overfit the train split. Because the initial process had not saved a
best-eval checkpoint, the trainer was patched afterward to save:

`checkpoint_best_eval.pt`

The completed process did not pick up that code change and did not write
`checkpoint_best_eval.pt`. A follow-up run should only use the patched trainer
if the training target is also repaired; repeating the same deterministic
MLP/MSE setup is not an aligned fix.

## Boundary

This is a source-suffix action generator baseline. It is not RGB-D/Cosmos
controller evidence, and it is not a live task-completion result. It becomes
useful only after generated chunks are evaluated on saved live failure states,
then in full `301/300` live panels with visual/final-state inspection.
