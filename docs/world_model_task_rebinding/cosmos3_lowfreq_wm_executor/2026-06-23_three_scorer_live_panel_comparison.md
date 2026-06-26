# 2026-06-23 Three-Scorer Live Panel Comparison

## Boundary

This note compares three formal-qualified candidate outcome scorers on the same
live closed-loop panel: samples `0,2,4,5`, source-suffix offsets
`64,48,32,24`, source-suffix `k=2`, 8-step candidate execution, DP96 handoff,
and the full `301` RGB/state frames plus `300` actions contract.

The result is diagnostic only. Each panel has four samples and
`method_evidence_allowed=false`.

## Results

- h2048:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h2048_scorer_source_suffix_offsets64_48_32_24_panel0245_20260623_alloc146658`
  succeeded on sample02 and sample04, failed on sample00 and sample05.
- h8192:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_scorer_source_suffix_offsets64_48_32_24_panel0245_20260623_alloc146658`
  succeeded on sample00 and sample05, failed on sample02 and sample04.
- h16384 positive-weight:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h16384_posw_scorer_source_suffix_offsets64_48_32_24_panel0245_20260623_alloc146658`
  succeeded on sample00 and sample05, failed on sample02 and sample04.

All three panels completed the full contract with no failed process. Contact
sheets were opened for h8192 and h16384_posw after completion and matched the
metrics.

## Interpretation

The candidates and low-frequency live interface are not useless: different
scorers can finish different dynamic samples. But a single scorer is not yet
stable enough to choose the correct short action and DP handoff across the
panel.

The common failure mode is physical and controller-facing: the policy often
aligns lateral `y/z` error, but does not push far enough in the insertion `x`
direction, or a frozen-DP handoff is triggered from a state that the live DP
rollout cannot actually finish. This is a candidate/handoff scoring failure,
not a reason to add hand-coded recovery cases or relax final-state success.

## Next Work

Build an offline scorer-comparison table across h2048, h8192, and h16384_posw
on saved live candidate banks and real DP96 labels. The useful question is
whether a calibrated ensemble or risk-aware rule can select the union of the
observed successes without per-sample branches. If the saved labels are not
enough, generate broader real DP-rollout labels for the ambiguous live states.

## Offline Ensemble Follow-Up

The follow-up comparison root is:
`experiments/world_model_task_rebinding/cosmos3/three_scorer_ensemble_compare_union_plus_panel0245_seed20260725_20260623_alloc146658`.

On the same 21 validation groups, DP handoff is `3/21` and the handoff oracle
is `7/21`. Single h2048 and h16384_posw reach `6/21`; h8192 reaches `5/21`.
Simple `mean_raw_score` and `mean_delta_vs_dp` reach only `5/21`.
`max_delta_vs_dp` reaches `6/21`, so it does not beat the best single scorer.

This means simple ensemble averaging/maxing is not enough to justify a live
controller run. The next repair should create broader true DP96 labels for the
ambiguous live states and/or add scorer features that distinguish lateral
alignment from actual insertion-direction progress.

## Targeted 02/04 DP96 Replay

The targeted replay root is:
`experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_three_scorer_02_04_sourcesuffix_dp96_20260623_alloc146658`.

It replayed saved candidate banks from the failed or ambiguous 02/04 live
states and then ran DP96 from the candidate end state. Results:

- h2048: `39` valid rows, `9` DP96 successes.
- h8192: `161` valid rows, `32` DP96 successes.
- h16384_posw: `133` valid rows, `9` DP96 successes.

The important conclusion is narrow: the failed live panels still contain
DP-finishable candidates in their saved banks. The current blocker is therefore
selection/feature calibration over real DP96 continuability, not the complete
absence of executable short chunks.

## Conversion Bug And Fix

The first conversion output must not be used:
`experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_three_scorer_02_04_sourcesuffix_dp96_20260623_alloc146658`.

It collapsed h2048/h8192/h16384 states with the same sample, iteration, and
prefix into the same uuid. That mixes different live base features and labels.
The converter was fixed to namespace uuid by the source replay panel.

The fixed output is:
`experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_three_scorer_02_04_sourcesuffix_dp96_fixuuid1_20260623_alloc146658`.

Fixed conversion summary: `44` groups, `333` valid candidate rows, `50` DP96
successes, `10` source-suffix DP96-success groups, no uuid spanning multiple
input JSONLs, and no duplicate `(uuid,candidate_name)` pairs.

## Augmented Scorer Data

The fixed 02/04 labels were merged with the previous union+panel0245 data:
`experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_source_suffix_union_plus_panel0245_plus0204fixuuid_dp96_20260623_alloc146658`.

Merged summary: `127` live-state groups, `9349` candidate rows, DP prior
handoff success `25/127`, handoff oracle `44/127`, and source-suffix handoff
success `32/127`.

A 100-step h2048 sanity run loaded this merged data:
`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h2048_rank_sanity100_seed20260725_20260623_alloc146658`.

It is debug-only. It learned train ordering but did not improve held-out
handoff selection in 100 steps: train selected handoff `32/95` versus DP
`17/95`; validation selected handoff `8/32`, DP `8/32`, oracle `12/32`.

The h8192 formal attempt was started here:
`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h8192_formal3h_seed20260725_20260623_alloc146658`.

It was interrupted after about four minutes because GPU utilization was about
`22%`, below the cluster release-risk threshold. It is not formal evidence and
must not be used for live evaluation.

The h16384 positive-weight formal attempt was then started here:
`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h16384_posw_formal3h_seed20260725_20260623_alloc146658`.

It uses the same merged data and split seed, h16384 capacity, and binary
positive weights `1,4,1,1,2`. Slurm step was `146658.137`; checked GPU
utilization was `52%`.

This run was interrupted after about 20 minutes. Reason: with
`--allow-handoff-only-gate`, the trainer saved `checkpoint_best_gate.pt` at
step `2000` because selected handoff was `10/32` versus DP `8/32`, but weighted
error and contact progress were not safely better. That is too permissive for
this project because it can pass a checkpoint that only improves the handoff
count while slightly worsening the physical state quality. The interrupted run
is not formal evidence and must not be used for live evaluation.

The active formal run is now:
`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h16384_posw_safegate1_formal3h_seed20260725_20260623_alloc146658`.

It removes `--allow-handoff-only-gate`, so a gate checkpoint must improve
handoff while also respecting the existing weighted-error and progress safety
constraints. Slurm step is `146658.141`. It must meet the `10800` second formal
floor before any live eval is allowed.

A concurrent h8192 safegate formal run was also started:
`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h8192_safegate1_formal3h_seed20260725_20260623_alloc146658`.

It uses the same merged data, same split seed, and same strict gate. Reason:
h16384 GPU utilization was bursty; the h8192 run is also a useful capacity
comparison while keeping the held 1-GPU allocation meaningfully active. Slurm
step is `146658.144`. It also requires the `10800` second formal floor before
any live eval is allowed.

Interim safegate result at 11:24+08: both safegate runs produced strict gate
checkpoints, but neither had met the formal time floor yet.

- h16384_posw safegate step `8000`: validation selected handoff `10/32`, DP
  `8/32`, oracle `12/32`, handoff delta `+0.0625`, weighted-error delta
  `-0.00108`, progress delta `+0.00006`.
- h8192 safegate step `3000`: validation selected handoff `11/32`, DP `8/32`,
  oracle `12/32`, handoff delta `+0.09375`, weighted-error delta `-0.00122`,
  progress delta `+0.00311`.

These checkpoints are cleaner than the interrupted handoff-only gate because
handoff, weighted error, and progress all move in the right direction. They
still cannot be used for live evaluation until the `10800` second formal floor
is met.

## H16384 Safegate Formal Result

At 13:25+08, h16384_posw safegate completed the formal floor:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h16384_posw_safegate1_formal3h_seed20260725_20260623_alloc146658/training_summary.json`.

Summary:

- `elapsed_seconds=10802`
- `formal_training_floor_met=true`
- `ready_for_formal_live_eval=true`
- exact checkpoint:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h16384_posw_safegate1_formal3h_seed20260725_20260623_alloc146658/checkpoint_best_gate.pt`
- best gate step `7000`
- validation selected handoff `11/32`, DP `8/32`, oracle `12/32`
- handoff delta `+0.09375`
- weighted-error delta `-0.00590`
- progress delta `+0.00261`

This is offline live-eval eligibility, not live method evidence.

## H8192 Safegate Formal Result

At 13:56+08, h8192 safegate also completed the formal floor:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h8192_safegate1_formal3h_seed20260725_20260623_alloc146658/training_summary.json`.

Summary:

- `elapsed_seconds=10800`
- `formal_training_floor_met=true`
- `ready_for_formal_live_eval=true`
- exact checkpoint:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h8192_safegate1_formal3h_seed20260725_20260623_alloc146658/checkpoint_best_gate.pt`
- best gate step `7000`
- validation selected handoff `11/32`, DP `8/32`, oracle `12/32`
- handoff delta `+0.09375`
- weighted-error delta `-0.00349`
- progress delta `+0.00404`

Both h16384_posw and h8192 strict safegate scorers are now live-eval eligible.
They are still offline selector evidence only. The next step is a strict full
`301/300` live panel with an exact `checkpoint_best_gate.pt`, followed by
direct video/contact-sheet inspection and final real simulator state review.

## Live Panel Launched

At 14:01+08, the h8192 safegate live panel was launched on allocation `146658`,
Slurm step `146658.148`, tmux window `live_h8192_safegate0134`.

Launcher:
`experiments/world_model_task_rebinding/cosmos3/launch_live_panel_seed20260725_h8192_safegate1_panel0134_20260623_alloc146658.sh`.

Exact scorer checkpoint:
`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h8192_safegate1_formal3h_seed20260725_20260623_alloc146658/checkpoint_best_gate.pt`.

The panel uses samples `0,1,3,4`, full `301` observed frames / `300` actions,
source-suffix offsets `64,48,32,24`, 8-step execution, and DP96 handoff. It is
not method evidence until final simulator metrics and direct video/contact
review are recorded.

Previous active formal path superseded by the low-utilization interruption:
`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h8192_formal3h_seed20260725_20260623_alloc146658`.
