# 2026-06-23 Causal Source-Suffix Diffusion Launch

## Purpose

This is the next experiment after two negative reset runs:

- deterministic source-suffix MLP overfit and underperformed mean-action
  baseline;
- live-outcome diffusion tied DP prior and had no direct inserted/successful
  live positives.

The physical target is still insertion action generation. This run trains on
direct inserted source suffixes from the accepted 733 H5 source set, but it
removes the non-causal `scenario_onehot` and future `first_insert_frame`
features used by the first MLP. The model is a conditional diffusion
generator, not a single MSE regressor, so it should better represent the
multimodal contact-action suffix distribution.

## Launch

Launched inside the tmux-held interactive Slurm allocation:

- Slurm job: `146658`
- Slurm step: `151`
- host: `server56`
- tmux window: `causal_suffix_diffusion_190108`
- wrapper:
  `scripts/slurm/run_causal_contact_action_suffix_diffusion_train_in_allocation.sh`
- trainer:
  `scripts/world_model/train_causal_contact_action_suffix_diffusion.py`
- output:
  `experiments/world_model_task_rebinding/cosmos3/causal_contact_action_suffix_diffusion_full733_1gpu1h_20260623_190108_alloc146658`

## Data

Input source suffix bank:

`experiments/world_model_task_rebinding/cosmos3/source_insertion_suffix_bank_20260622_alloc145920/source_insertion_suffix_bank.npz`

Read-only bank summary:

- source H5 trajectories seen: `733`
- inserted suffix rows: `4711`
- horizon: `96`
- robot action dim: `7`
- offsets before insertion: `96,64,48,32,24,16,8,0`
- end relative `abs_yz_sum` mean: `0.00449`
- end relative `x` median: `-0.00088`

The trainer keeps inserted-and-grasped rows and uses a source-UUID train/val
split. Conditions are limited to causal or controllable quantities:

- current/start peg-head-in-hole relative `x,y,z`;
- absolute/lateral task-frame error summaries;
- requested offset-before-insertion token;
- current frame / remaining episode;
- grasped-at-start flag;
- valid suffix horizon.

It explicitly does not condition on scenario labels or future first-insert
frame. Scenario counts remain diagnostic metadata only.

## Boundary

This is training evidence only. Even if the model beats source-action
baselines, it does not prove dynamic task completion. Required next gates are:

1. generated-action saved-snapshot replay from real live failure states;
2. DP96/final-state consequence labels from replay;
3. only then a full `301/300` live panel with video/contact-sheet inspection.

Early metrics are allowed only as liveness and training-shape checks. The
formal result requires the one-GPU-hour floor and `training_summary.json`.

## Early Status

At step `800`, elapsed `5.96` seconds, the run had written training metrics.
The teacher-forced mid-noise reconstruction metric was already below the
mean-action baseline (`eval_x0_action_mse_mid_t=0.00110` versus baseline
`0.01557`), but this is not replay evidence. The wrapper enforces
`min_wall_seconds=3660`, so the run must continue to the one-GPU-hour floor
before interpretation.

At `2026-06-23 19:13 CST`, the run was still active on Slurm step `146658.151`
at `669` elapsed training seconds. Metrics remained healthy for the
teacher-forced diffusion training objective:

- `eval_denoise_mse=0.282`
- `eval_zero_noise_baseline_mse=0.999`
- `eval_x0_action_mse_mid_t=0.000286`
- `eval_mean_action_baseline_mse=0.01557`

This is still in-progress training only. It does not bypass the one-GPU-hour
floor or the generated-action replay gate.

## Result

The run completed at `2026-06-23 20:05 CST` and wrote:

`experiments/world_model_task_rebinding/cosmos3/causal_contact_action_suffix_diffusion_full733_1gpu1h_20260623_190108_alloc146658/training_summary.json`

Summary:

- `formal_one_gpu_hour_floor_met=true`
- `elapsed_seconds=3660.22`
- `steps=638567`
- `stop_reason=min_wall_and_min_steps`
- `ready_for_saved_snapshot_replay_gate=true`
- best step: `148400`
- best `eval_denoise_mse=0.285` versus zero-noise baseline `0.999`
- best `eval_x0_action_mse_mid_t=0.000283` versus mean-action baseline
  `0.01557`
- final `eval_denoise_mse=0.318`
- final `eval_x0_action_mse_mid_t=0.000314`

This is the first post-reset training run that passes a source-training gate.
It still proves only that the model learned the direct source-suffix action
distribution under teacher-forced noising. It does not prove live dynamic
task completion. The next required step is generated-action replay from saved
live failure snapshots using the sampled diffusion chunks.
