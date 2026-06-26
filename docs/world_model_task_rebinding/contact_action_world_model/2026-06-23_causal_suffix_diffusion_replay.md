# 2026-06-23 Causal Suffix Diffusion Saved-Snapshot Replay

## Purpose

The causal source-suffix diffusion generator passed its one-GPU-hour
source-training gate, but that only proves teacher-forced source action
modeling. This replay tests the next required question: when sampled actions
are restored into real saved live failure snapshots from the latest strict
panel, do they move the peg into a better inserted/contact-continuable state,
and can DP continue afterward?

## Launch

Launched inside the tmux-held interactive Slurm allocation:

- Slurm job: `146658`
- Slurm step: `152`
- host: `server56`
- tmux window: `replay_causal_suffix_200828`
- replay tool:
  `scripts/world_model/replay_cosmos3_live_action_bank_from_snapshots.py`
- panel root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_safegate1_source_suffix_offsets64_48_32_24_panel0134_20260623_alloc146658`
- checkpoint:
  `experiments/world_model_task_rebinding/cosmos3/causal_contact_action_suffix_diffusion_full733_1gpu1h_20260623_190108_alloc146658/checkpoint_best_eval.pt`
- output:
  `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_causal_suffix_diffusion_panel0134_offsets64_48_32_24_16_8_s2_exec8_dp96_20260623_200828_alloc146658`

Replay configuration:

- samples: up to `4` failed panel samples
- iter dirs per sample: up to `4`
- generated offsets: `64,48,32,24,16,8`
- generated samples per offset: `2`
- execute steps: `8`
- DP continuability rollout: `96` steps
- saved bank candidates are not replayed in this run; this is generated
  causal-suffix-diffusion replay only.

## Boundary

This is saved-snapshot label replay, not a live controller result. It restores
real saved live simulator snapshots, executes generated short chunks, then
computes consequence labels. A positive replay result would justify a live
panel; it would not by itself be method success.

Early process status: Slurm step `146658.152` started on `server56`.

## First Launch Failure

The first replay launch failed before producing labels. Failure type:
implementation/checkpoint-loading bug.

Cause: the replay loader reconstructed
`CausalSuffixDiffusionNet` with `dropout=0.0`, but the training checkpoint was
created with dropout modules present. Because the network is built as
`torch.nn.Sequential`, removing dropout changes module indices and makes
`load_state_dict` fail.

Fix: reconstruct the replay model with the checkpoint's saved `dropout`
argument, then call `model.eval()` so dropout is disabled at inference without
changing the state-dict schema.

This failure is not physical evidence and not a replay result. A new replay
run must be launched after the loader fix.

## Fix1 Result

Relaunched after the loader fix:

- Slurm step: `146658.153`
- output:
  `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_causal_suffix_diffusion_panel0134_offsets64_48_32_24_16_8_s2_exec8_dp96_fix1_20260623_201146_alloc146658`

Summary:

- `records=192`
- `valid_records=192`
- `failure_counts={}`
- generated causal suffix diffusion records: `192`
- direct `after_success_count=0`
- direct `after_gate_ok_count=0`
- `dp_rollout_label_count=192`
- `dp_rollout_success_count=55`
- `dp_rollout_continuable_count=59`
- `dp_rollout_final_contact_stable_count=59`
- `improved_abs_yz_sum_count=50`
- `worsened_abs_yz_sum_count=142`

Group-level coverage:

- live snapshot groups: `16`
- groups with at least one generated candidate whose `candidate + DP96`
  succeeds: `16/16`
- groups with at least one generated candidate marked DP-continuable: `16/16`
- groups with direct generated-chunk success: `0/16`
- groups passing the post-chunk continuability gate before DP rollout: `0/16`

Interpretation:

The generated causal suffix diffusion model has real candidate coverage in
saved live failure snapshots: every tested snapshot has at least one sampled
chunk that lets DP finish within the DP96 label rollout. However, the generated
8-step chunks do not directly insert and do not satisfy the current
post-chunk continuability gate by themselves. The next blocker is candidate
selection/value modeling over generated chunks, plus possibly longer or
receding generated execution. This is not yet a live-panel-ready controller.

The best single fixed candidate family in this bounded replay was
`causal_suffix_diffusion_o8_s0`, with `8/16` DP96 successes. That is not enough
for a hand-picked live policy, and selecting it by name would be a brittle
case split. The aligned next step is to convert these generated replay labels
into the consequence/value training format and train a value head over sampled
generated chunks.
