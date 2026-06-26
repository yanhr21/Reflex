# 2026-06-18 Outcome Scorer Live Panel

## What Ran

Live closed-loop panel on allocation `140738` / `server54`:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260618_1510_scorer_margin003_h24_server54_defaulticd_alloc140738_server54_samples0_1_3_4`

Controller:

- Cosmos3 dense full-episode checkpoint `iter_000001500`;
- formal 24-step candidate executor checkpoint `checkpoint_best_gate.pt`;
- formal retrieval-union outcome scorer checkpoint
  `candidate_outcome_scorer_hardphase_retrieval_union320_norank_formal1gpu3h_20260618_alloc139754/checkpoint_best_gate.pt`;
- conservative outcome-scorer DP margin `0.03`;
- action execution horizon `24`.

The panel finished all requested samples. There were no process failures and
the full episode contract passed: `301` observed frames / `300` action steps.

## Result

Panel result: `1/4` final real simulator success.

| sample | scenario | success | executor frames | DP handoff steps | final peg head at hole |
| --- | --- | --- | ---: | ---: | --- |
| 0 | `hole_late_move_stop` | false | 194 | 0 | `[-0.0926, 0.0008, -0.0664]` |
| 1 | `hole_late_constant` | false | 182 | 24 | `[-0.0802, -0.0028, 0.0059]` |
| 3 | `hole_late_fast_shift` | true | 71 | 97 | `[0.0060, 0.0030, 0.0030]` |
| 4 | `hole_late_sine` | false | 112 | 72 | `[-0.0919, 0.0025, 0.0014]` |

The contact sheet exists and was opened:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260618_1510_scorer_margin003_h24_server54_defaulticd_alloc140738_server54_samples0_1_3_4/live_receding_panel_contact_sheet.png`

Visual read: sample 3 is consistent with the metric success; samples 0, 1,
and 4 are not visually completed insertions. The panel summary still records
`method_evidence_allowed=false` because it requires direct review status to be
resolved; this result must therefore be reported as partial live evidence, not
final method success.

## Follow-Up Negative Diagnostic: Simple Outcome Filter

A follow-up live panel tested a simple live filter over the same four samples:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260618_1630_scorer_margin003_h24_progress_nonneg_cont005_server54_defaulticd_alloc140738_unknown_node_samples0_1_3_4`

Filter settings:

- outcome scorer margin `0.03`;
- non-DP candidate minimum predicted progress delta `0.0`;
- non-DP candidate minimum predicted DP-continuable probability `0.05`;
- non-DP candidate minimum inserted probability `0.0`.

Result: `0/4` final real simulator success. The full `301/300` contract still
passed and there were no process failures.

| sample | scenario | success | DP handoff steps | final peg head at hole |
| --- | --- | ---: | ---: | --- |
| 0 | `hole_late_move_stop` | false | 0 | `[-0.0925, 0.0006, -0.0584]` |
| 1 | `hole_late_constant` | false | 0 | `[-0.0929, -0.0277, -0.0641]` |
| 3 | `hole_late_fast_shift` | false | 42 | `[-0.1344, -0.0061, 0.0022]` |
| 4 | `hole_late_sine` | false | 22 | `[-0.1065, -0.0064, -0.0068]` |

The contact sheet was opened:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260618_1630_scorer_margin003_h24_progress_nonneg_cont005_server54_defaulticd_alloc140738_unknown_node_samples0_1_3_4/live_receding_panel_contact_sheet.png`

Visual read agrees with the metric: none of the four samples completed
insertion. The most important comparison is sample 3. The first panel solved
sample 3, but the simple filter changed the selected early chunks and the same
sample failed badly. This falsifies the idea that a scalar live threshold over
the current outcome scorer is the right repair.

Plain conclusion: the current optional filter code is useful only as a
diagnostic switch with defaults off. The formal path should not use
`min_progress_delta=0.0` and `min_continuable_prob=0.05` as the new controller.
It rejects candidates that can become useful after re-observation, while still
allowing bad DP fallback and bad near-contact handoff.

## What This Proves

This is the first useful live evidence for the DDP-style branch:

- Cosmos is no longer used as a raw high-frequency action controller.
- A candidate/diffusion executor plus outcome scorer can run in real receding
  closed loop.
- The system can solve at least one dynamic sample by using executor chunks to
  enter the DP-continuable region and then letting frozen DP finish insertion.

Sample 3 mechanism:

- early executor chunks moved the state from far/off-target into the handoff
  region;
- from prefix `203` onward, DP handoff stayed inside the real-state gate;
- DP handoff then completed insertion by frame `300`.

This is not a lucky generated sidecar: final success comes from the real live
simulator state and the contact sheet agrees with the metric.

## What Still Fails

The current blocker is not SFT startup, render, length accounting, or lack of
closed-loop wiring. The blocker is controller quality near contact:

1. Non-DP candidate selection is too willing to execute low-confidence chunks.
   On sample 3 and sample 4, the first selected `mean` chunk had negative
   predicted progress delta and very low predicted continuability, yet it was
   allowed to override DP because it exceeded the scalar margin. In sample 4
   this worsened the state immediately.
2. The scorer/executor do not model insertion-axis and contact stability
   tightly enough. They often improve x/y or pull the peg near the hole, but
   z/contact remains outside the real insertion manifold.
3. The current handoff gate is too instantaneous. It can pass for a single
   real state, but DP may still drift out over the next short chunk. Sample 1
   passed pre-handoff at prefix `228`, then DP moved z from about `0.0032m` to
   `0.0112m`, losing the insertion window.
4. The learned readouts are overconfident on near-hole bad states. In sample 4
   at prefix `188`, the executor readout predicted insert/continuable near
   `0.996/0.999` while the real state was still far from success.

Plain diagnosis: the method can rebind and approach the moved task frame, but
it does not yet robustly maintain the millimeter-level contact/insertion
conditions needed for final peg insertion.

## Next Fix

Do not add sample-specific rescue cases.

The next aligned fix is a general contact-stability repair, but not the simple
outcome-only filter above:

1. Train/evaluate the outcome scorer with explicit z/contact-stability targets
   instead of relying on a single weighted task error plus y/z lateral norm.
2. Replace the current single-frame continuability label with a short DP-rollout
   continuability label: a state is DP-continuable only if DP can keep the peg
   grasped and inside the insertion band over a short causal rollout, or finish.
3. Add a safe/adaptive executor fallback for low-confidence states: when all
   candidate chunks and DP look harmful, execute a shorter chunk or a hold/re-
   observe candidate rather than forcing a full 24-step DP or non-DP chunk.
4. Use current live failures as hard negative calibration evidence for the
   general continuability/scorer objective, not as positive rescue trajectories
   and not as enumerated error-recovery cases.

## Implemented After This Diagnostic

The outcome-label path was updated to support the next repair directly:

- `scripts/world_model/export_cosmos3_candidate_outcome_labels.py` now has an
  optional label-only short frozen-DP rollout after each candidate chunk:
  `--dp-rollout-continuability-horizon`. The rollout records whether DP can
  keep the state contact-stable or finish from the candidate's final state.
  Default horizon is `0`, so old calls remain compatible.
- The same exporter now records `final_contact_stable_proxy`,
  `dp_rollout_continuable_proxy`, `dp_rollout_success`, and
  `dp_rollout_stable_step_count`.
- `scripts/world_model/train_cosmos3_candidate_outcome_scorer.py` now uses
  `dp_rollout_continuable_proxy` as the preferred continuability target when
  it is present, falling back to the old single-frame proxy only for old rows.
- `scripts/slurm/run_cosmos3_hardphase_exploratory_candidate_replay_in_allocation.sh`
  and `scripts/slurm/run_cosmos3_hardphase_retrieval_smoke_in_allocation.sh`
  pass and record the new DP-rollout/contact-stability label settings.

Compute-node checks on allocation `140738` passed:

- `py_compile` for the outcome exporter, scorer trainer, and margin eval;
- `bash -n` for the edited replay wrappers;
- `--help` import/argument check for the exporter and trainer.

A one-row compute smoke ran on allocation `140738`:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval1_skip0_k0_20260618_1800_dp_rollout_label_smoke1_alloc140738`

It wrote `10` successful candidate-outcome rows, all with
`dp_rollout_continuability_horizon=8`. The first inspected record contains
the new nested `dp_rollout_continuability` label and correctly marks a bad DP
prior outcome as not DP-continuable.

The active follow-up shard is running on allocation `140738`:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval64_skip0_k8_20260618_1815_dpcont_skip0_alloc140738`

This shard uses the new DP-rollout labels and is data generation for the next
union/formal scorer. It is not method evidence and not live controller success.

## 2026-06-18 19:45+08 DP-Rollout Label Union

Two new-label retrieval shards have completed:

- skip `0`, allocation `140738`, output root
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval64_skip0_k8_20260618_1815_dpcont_skip0_alloc140738`;
- skip `64`, allocation `140717`, output root
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval64_skip64_k8_20260618_1820_dpcont_skip64_alloc140717`.

Each shard wrote `5440` successful candidate-outcome records with `0` row
failures. Skip `0` has `170` DP-rollout-continuable candidates and `118`
DP-rollout successes. Skip `64` has `231` DP-rollout-continuable candidates
and `167` DP-rollout successes.

The two-shard union on allocation `140738` passed the offline gate:

- `128` hard groups;
- DP success `7`, oracle success `14`;
- mean oracle-minus-DP weighted error `-0.0743`;
- meaningful improvements `95/128`;
- large improvements `76/128`;
- retrieval-family oracle candidates in `28` groups.

The formal 1-GPU/3-hour scorer run is active under
`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval_claim_union_rank1_canddesc_smoke500_20260618_1940_dpcont_union2_formal_alloc140738`.
This is offline scorer evidence only. It does not count as live controller
success until the trained scorer is used in a real closed-loop panel with final
state and video review.

## 2026-06-18 20:00+08 Full DP-Rollout Label Union

All five retrieval shards have completed: skip `0`, `64`, `128`, `192`, and
`256`. The complete union covers `312` hard groups. The final shard, skip
`128`, wrote `5440` successful records with `0` row failures, `206`
DP-rollout-continuable candidates, and `184` DP-rollout successes.

## 2026-06-19 h32/min16 Handoff Label Diagnostic

A later same-candidate comparison showed that the old DP-continuability label
was too short, not too loose. On the same `12` hard-phase groups and `444`
candidate records, `horizon=8,min_stable=4` produced `11` continuable
candidates and `10` DP-rollout successes, while `horizon=32,min_stable=16`
produced `31` continuable candidates and `21` DP-rollout successes. This means
some states need a longer DP handoff to finish insertion.

The runtime was also fixed to match this label: live DP handoff used to cap
`dp_handoff_horizon=32` by `action_exec_horizon=24`. The live loop, panel, and
Slurm wrapper now expose `dp_handoff_chunk_horizon`, so a live diagnostic can
run 32-step DP handoff without lengthening executor chunks.

A reduced64 h32/min16 label shard then completed on allocation `140738`:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval64_skip0_k4_20260619_0312_dpcont32_min16_reduced64_alloc140738`

It wrote `2368` candidate outcome rows with `0` failed rows. Headroom was
useful but sparse: `6/64` oracle-success groups versus `3/64` DP-success
groups, mean oracle-minus-DP weighted error `-0.0323`, `35/64` meaningful
improvements, and `21/64` large improvements.

A 100-step scorer sanity on that reduced64 shard finished here:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval64_skip0_k4_dpcont32_min16_reduced64_smoke100_20260619_050328_alloc140738`

Result: it did not pass as a usable selector. The train split improved over
DP (`selected-minus-DP=-0.0355`, progress delta `+0.0623`), but the held-out
split got worse (`selected-minus-DP=+0.0084`, progress delta `-0.0235`).
Conservative margin eval had `0` gate-passing eval margins.

Plain conclusion: the h32 label path runs and produces the right kind of
signal, but `64` groups are too small and sparse for a live controller scorer.
The next running chain is therefore the larger reduced128 h32/min16 export
plus 1-GPU/3-hour scorer on allocation `140738`, session
`cosmos3_dpcont32_reduced128_formal_140738_20260619`, log:

`experiments/world_model_task_rebinding/cosmos3/dpcont32_reduced128_formal_20260619_0510_alloc140738_tmux.log`

The reduced128 export completed at 2026-06-19 08:23+08:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval128_skip0_k4_20260619_0510_dpcont32_min16_reduced128_formal_alloc140738`

It wrote `4736` candidate outcome rows with `0` failed rows. The export
summary recorded `438` DP-rollout-continuable candidates and `401`
DP-rollout successes under `horizon=32,min_stable=16`.

The headroom gate passed both terminal and progress gates:

- `128` hard groups;
- `7` DP-success groups;
- `13` oracle-success groups;
- mean oracle-minus-DP weighted error `-0.0373`;
- `74/128` meaningful improvements;
- `44/128` large improvements;
- retrieval-family oracle candidates in `19` groups, with `1` terminal
  retrieval-success group.

The formal scorer started here:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval128_skip0_k4_rank1_canddesc_smoke500_20260619_0510_dpcont32_min16_reduced128_formal_alloc140738`

It has already written `checkpoint_best_gate.pt`, but this checkpoint is not
live-method evidence until the run reaches the required `10800` second
training floor and the final training summary marks
`ready_for_formal_live_eval=true`.

The five-shard union passed the offline scorer gate:

- DP success `23`, oracle success `35`;
- mean oracle-minus-DP weighted error `-0.0649`;
- meaningful improvements `219/312`;
- large improvements `171/312`;
- retrieval-family oracle candidates in `55` groups.

The preferred next scorer is now the five-shard formal run:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval_claim_union_rank1_canddesc_smoke500_20260618_2000_dpcont_union5_formal_alloc140717`

It is running on allocation `140717` with `SCORER_MIN_WALL_SECONDS=10800`,
`SCORER_MAX_WALL_SECONDS=10800`, `SCORER_FORMAL_MIN_GPUS=1`, and rank-loss
weight `0.0`. It is not live evidence yet. The next controller step should
use it only after the three-hour floor finishes, margin eval passes, and the
live closed-loop panel is run with final-state and video review.

## 2026-06-18 Late Formal DP-Rollout Scorers

Both formal DP-rollout-continuability scorers completed the required
1-GPU/3-hour floor.

Two-shard scorer:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval_claim_union_rank1_canddesc_smoke500_20260618_1940_dpcont_union2_formal_alloc140738`

- `formal_training_floor_met=true`
- `ready_for_formal_live_eval=true`
- best-gate checkpoint step `275`
- best-gate eval selected-minus-DP weighted error `-0.013472`
- margin eval selected margin `0.02`, selected-minus-DP `-0.014522`,
  progress delta `+0.018466`, non-DP fraction `0.500000`

Five-shard scorer:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval_claim_union_rank1_canddesc_smoke500_20260618_2000_dpcont_union5_formal_alloc140717`

- `formal_training_floor_met=true`
- `ready_for_formal_live_eval=true`
- best-gate checkpoint step `150`
- best-gate eval selected-minus-DP weighted error `-0.008783`
- margin eval selected margin `0.02`, selected-minus-DP `-0.010274`,
  progress delta `+0.009632`, non-DP fraction `0.192308`

In both runs the final checkpoint overfit relative to the best-gate checkpoint.
Only `checkpoint_best_gate.pt` should be used for live evaluation.

## 2026-06-19 00:20+08 Five-Shard Scorer Live Panel

The five-shard scorer was used in a real live panel on allocation `140738` /
`server54`:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260618_2258_dpcont_union5_margin002_h24_server54_alloc140738_server54_samples0_1_3_4`

Controller settings:

- Cosmos3 dense full-episode checkpoint `iter_000001500`;
- formal 24-step candidate executor `checkpoint_best_gate.pt`;
- five-shard DP-rollout-continuability scorer `checkpoint_best_gate.pt`;
- outcome-scorer DP margin `0.02`;
- action execution horizon `24`;
- no extra scalar progress/continuability filter.

The panel finished all four requested samples with no process failures and
the full `301` observed-frame / `300` action-step contract passed. Final live
success was again `1/4`.

| sample | scenario | success | DP handoff steps | final peg head at hole |
| --- | --- | ---: | ---: | --- |
| 0 | `hole_late_move_stop` | false | 0 | `[-0.0963, 0.0040, -0.0612]` |
| 1 | `hole_late_constant` | false | 0 | `[-0.0915, -0.0357, -0.0733]` |
| 3 | `hole_late_fast_shift` | true | 69 | `[0.0080, 0.0015, 0.0028]` |
| 4 | `hole_late_sine` | false | 0 | `[-0.1052, 0.0088, -0.0089]` |

The contact sheet was opened:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260618_2258_dpcont_union5_margin002_h24_server54_alloc140738_server54_samples0_1_3_4/live_receding_panel_contact_sheet.png`

Visual read agrees with the metrics: sample 3 is the only real insertion;
samples 0, 1, and 4 remain visibly outside final insertion. The panel summary
records `method_evidence_allowed=false`, so this is still partial live
evidence, not final method success.

## Updated Diagnosis After Union5 Live

The DP-rollout label repair was useful offline, but it did not fix the live
controller. The live blocker is now more specific:

1. The scorer still produces false-positive continuability near contact.
   Sample 1 and sample 4 had high predicted continuability late in the run,
   but the real handoff gate never accepted DP and both ended with large
   insertion-axis error.
2. Sample 3 proves the DDP-style mechanism can work: receding executor chunks
   reached a handoff region, then DP finished insertion from real state.
3. The three failures prove the mechanism is not reliable. The controller can
   align some axes while leaving x/insertion-axis or z/contact outside the
   true insertion manifold.

Plain conclusion: the current problem is not render, queueing, SFT startup,
data export, or length accounting. It is the mismatch between offline candidate
labels/scorer confidence and live physical continuability at contact.

The next aligned fix is not more enumerated rescue. Calibrate the scorer and
selector against the same live handoff condition required at runtime:

- make large insertion-axis error a hard negative for high continuability even
  when y/z look close;
- train on short live-like DP-handoff survival/failure labels rather than only
  restored candidate-end labels;
- add an adaptive short-chunk or hold/re-observe fallback when every full
  24-step candidate looks likely to leave the insertion manifold.

## 2026-06-19 02:25+08 State-Axis Calibration Live Panel

After the union5 live failure, the selector was given a general predicted
state-error penalty rather than a sample-specific recovery rule. The physical
reason was simple: the failed live samples were often close in y/z or contact
appearance but still far along the insertion axis, so high predicted
continuability needed to be punished when predicted final x error remained
large.

Implementation changes:

- the outcome scorer trainer, margin evaluator, live loop, live panel, and
  slurm wrappers now accept score-state abs-axis weights and target state;
- the live selector can subtract weighted predicted final-state error from the
  candidate score;
- wrapper manifests and per-candidate records include the active weights and
  predicted state.

Compute-node checks on allocation `140738` passed for the edited path:
`py_compile`, wrapper `bash -n`, and live/scorer `--help` import checks.

Calibration evidence before live:

- a strong state penalty `1.0,0.25,0.25` failed the offline gate because it
  collapsed selection toward DP and removed useful non-DP choices;
- an eval-only sweep over the formal union5 checkpoint kept the offline gate
  best around `0.1,0,0`, so the live test used that x-only weight.

Live panel:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260619_0102_statex010_dpcont_union5_margin002_h24_server54_alloc140738_samples0_1_3_4`

Settings:

- same Cosmos3 dense full-episode checkpoint `iter_000001500`;
- same formal 24-step candidate executor `checkpoint_best_gate.pt`;
- same five-shard DP-rollout-continuability scorer `checkpoint_best_gate.pt`;
- outcome-scorer DP margin `0.02`;
- `CANDIDATE_OUTCOME_SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS=0.1,0,0`;
- action execution horizon `24`.

The panel finished all four samples, had `0` process failures, and passed the
full `301` observed-frame / `300` action-step contract. Final live success
remained `1/4`.

| sample | scenario | success | DP handoff steps | final peg head at hole |
| --- | --- | ---: | ---: | --- |
| 0 | `hole_late_move_stop` | false | 0 | `[-0.0998, -0.0127, -0.0642]` |
| 1 | `hole_late_constant` | false | 24 | `[-0.0936, -0.0181, -0.0203]` |
| 3 | `hole_late_fast_shift` | true | 59 | `[0.0071, 0.0013, 0.0026]` |
| 4 | `hole_late_sine` | false | 0 | `[-0.1052, 0.0015, -0.0049]` |

The contact sheet was opened:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260619_0102_statex010_dpcont_union5_margin002_h24_server54_alloc140738_samples0_1_3_4/live_receding_panel_contact_sheet.png`

Visual read agrees with the metrics. Sample 3 reaches a real DP handoff and
finishes insertion. Samples 0, 1, and 4 remain visibly outside insertion at
the end. The run records `method_evidence_allowed=false`, so it is partial
live evidence, not final method success.

Plain conclusion: the light x-axis penalty did not hurt the one known success,
but it did not rescue the failures. The blocker is not that one scalar was
missing from the score. The blocker is that the scorer/selector still lacks a
reliable live notion of DP-continuable insertion manifold, and the candidate
set still does not reliably produce near-contact chunks that close the
remaining insertion-axis gap.

Next fix: stop adding scalar penalties. Train and evaluate the selector on
live-like handoff survival: after a candidate chunk, short causal DP handoff
must keep the peg inside the insertion manifold or finish. Also add
shorter-chunk or hold/reobserve candidates for uncertain near-contact states,
so the controller is not forced to commit a full 24-step action when x remains
unresolved.

## 2026-06-19 03:08+08 DP-Continuability Label Length Diagnostic

After the state-axis live panel failed to improve success, the next question
was whether the offline DP-continuability labels were too permissive. A
64-row / full-candidate `horizon=32,min_stable=16` attempt was stopped early:
it was only a label diagnostic and would have spent hours at low GPU
utilization before producing a summary.

A smaller same-row comparison was then run on allocation `140738` /
`server54`. Both exports used the same 12 hard-phase groups, `444` candidate
records, retrieval `k=4`, and `16` model candidate samples:

- old label:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval12_skip0_k4_20260619_0300_dpcont8_min4_rows12_compare_alloc140738`;
- longer label:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval12_skip0_k4_20260619_0250_dpcont32_min16_rows12_alloc140738`.

Result:

| label | continuable candidates | DP-rollout successes | final contact-stable |
| --- | ---: | ---: | ---: |
| `horizon=8,min_stable=4` | `11/444` | `10/444` | `10/444` |
| `horizon=32,min_stable=16` | `31/444` | `21/444` | `19/444` |

The two exports had the same `444` candidate keys. Shared-candidate comparison:

- `10` candidates were continuable under both labels;
- `1` candidate was continuable only under the 8-step label;
- `21` candidates were continuable only under the 32-step label;
- `412` candidates were continuable under neither label.

Plain conclusion: the old 8-step label is not too loose. It is too short. Many
candidates that still look bad after eight DP steps can finish after a longer
DP handoff. This does not solve the live failure, because the live controller
still must reach and recognize those longer-horizon DP-finishable states from
real observations. The next aligned experiment is a reduced-candidate 64-row
`horizon=32,min_stable=16` label shard, followed by scorer debug/retraining if
the gate remains useful.

Related implementation fix: the live loop accepted `dp_handoff_horizon=32`, but
the actual per-iteration DP handoff chunk was capped by `action_exec_horizon`.
In the current panel that meant DP handoff chunks were only `24` steps. The
live loop, panel entry point, and Slurm wrapper now expose
`dp_handoff_chunk_horizon`; default `0` keeps old behavior, while
`DP_HANDOFF_CHUNK_HORIZON=32` lets the DP handoff duration match the 32-step
continuability label without lengthening executor chunks. Compute-node
`py_compile`, wrapper `bash -n`, and `--help` checks passed on allocation
`140738`.

## 2026-06-19 12:45+08 Reduced128 h32/min16 Live Panel

The larger reduced128 h32/min16 scorer finished formal training on allocation
`140738`. The training summary records `formal_training_floor_met=true`,
`elapsed_seconds=10800.01`, `steps=222852`, `best_gate_step=1625`, and
`ready_for_formal_live_eval=true`. Conservative margin eval selected margin
`0.03`; the best eval margin had selected-minus-DP weighted error `-0.0163`,
selected-minus-DP progress delta `+0.0136`, and selected a non-DP candidate
for `16/32` eval groups.

Live panel output:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260619_1127_dpcont32_reduced128_margin003_h24_dphandoff32_server54_alloc140738_samples0_1_3_4`

Settings:

- Cosmos3 dense full-episode checkpoint `iter_000001500`;
- formal 24-step candidate executor `checkpoint_best_gate.pt`;
- reduced128 h32/min16 outcome scorer `checkpoint_best_gate.pt`;
- outcome-scorer DP margin `0.03`;
- `ACTION_EXEC_HORIZON=24`;
- `DP_HANDOFF_CHUNK_HORIZON=32`.

The panel completed all four samples, had `0` process failures, and passed the
full `301` observed-frame / `300` action-step contract.

| sample | scenario | success | WM frames | DP handoff steps | final peg head at hole |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | `hole_late_move_stop` | true | 97 | 97 | `[-0.0058, 0.0002, 0.0011]` |
| 1 | `hole_late_constant` | false | 206 | 0 | `[-0.0943, 0.0084, -0.0653]` |
| 3 | `hole_late_fast_shift` | false | 128 | 40 | `[-0.1174, 0.0013, -0.0002]` |
| 4 | `hole_late_sine` | false | 103 | 81 | `[-0.0983, -0.0044, -0.0029]` |

The contact sheet was opened:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260619_1127_dpcont32_reduced128_margin003_h24_dphandoff32_server54_alloc140738_samples0_1_3_4/live_receding_panel_contact_sheet.png`

Visual read agrees with the metrics. Sample 0 is a real insertion success.
Samples 1, 3, and 4 end near the block/hole but are not completed insertions.
The panel summary records `method_evidence_allowed=false`, so this is partial
live evidence and not final method success.

Plain diagnosis: the scorer training and live plumbing are no longer the main
blockers. The larger h32/min16 scorer improves offline candidate selection,
and the runtime now executes the 32-step DP handoff duration it was trained to
recognize. Live failure remains because the executor/selector still does not
reliably eliminate the remaining insertion-axis error before handoff. DP can
finish only when the live state is already physically insertable; in three of
four samples the chosen chunks leave about `9-12cm` x error.

Next useful fix: do not add enumerated rescue cases or more one-off scalar
thresholds. Improve the general near-contact action set and labels: add
shorter chunks or hold/reobserve candidates for uncertain states, and train
the selector on live-like handoff-survival negatives where DP from the
candidate end state fails to stay in the insertion manifold or finish.

## 2026-06-19 Short-Prefix Candidate Diagnostic

Implemented a default-off live candidate option:
`--candidate-executor-short-prefix-steps`. With values such as `8,12,16`, the
candidate executor builds extra candidates from the same 24-step prediction:
the first `N` steps use the candidate residual and the remaining steps use the
DP prior. The outcome scorer still sees a fixed 24-step feature vector, but if
that short candidate is selected, the live controller executes only the first
`N` steps and then reobserves.

Reason: the latest failures are not caused by missing closed-loop plumbing.
They are overcommitment/near-contact action failures: a full 24-step chunk can
leave `9-12cm` insertion-axis error before the next observation. Short-prefix
candidates test a general receding-horizon repair without adding sample
recovery branches.

Compute-node checks on allocation `140738` passed before launch:

- Python `py_compile` for `run_cosmos3_live_receding_loop.py` and
  `run_cosmos3_live_receding_panel.py`;
- `bash -n` for both live Slurm wrappers;
- `--help` showed the new option in both loop and panel entry points;
- a helper check confirmed short candidate descriptor mapping reuses the base
  candidate type;
- a targeted dummy candidate-function check selected `short12_scale_0.1` and
  returned a 12-step action chunk.

Launched diagnostic panel:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_shortprefix81216_20260619_1250_shortprefix81216_dpcont32_reduced128_margin003_h24_dphandoff32_server54_alloc140738_samples0_1_3_4`

Session:
`cosmos3_live_shortprefix81216_reduced128_handoff32_140738_20260619`.

This run is a diagnostic until final real simulator metrics and the contact
sheet are inspected.

Result: the diagnostic finished with `1/4` final success, `0` process
failures, and full `301/300` contract.

| sample | scenario | success | WM frames | DP handoff steps | final peg head at hole |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | `hole_late_move_stop` | true | 64 | 130 | `[-0.0079, -0.0008, 0.0017]` |
| 1 | `hole_late_constant` | false | 206 | 0 | `[-0.0951, 0.0142, -0.0607]` |
| 3 | `hole_late_fast_shift` | false | 136 | 32 | `[-0.1291, -0.0029, -0.0474]` |
| 4 | `hole_late_sine` | false | 184 | 0 | `[-0.1007, 0.0119, -0.0540]` |

The contact sheet was opened:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_shortprefix81216_20260619_1250_shortprefix81216_dpcont32_reduced128_margin003_h24_dphandoff32_server54_alloc140738_samples0_1_3_4/live_receding_panel_contact_sheet.png`

Visual read agrees with the metrics. Sample 0 is inserted. Samples 1, 3, and
4 are not completed insertions.

Plain conclusion: short-prefix execution worked mechanically. Sample 0
selected short candidates early, and the rollout reobserved after 8-step
chunks instead of always executing 24 steps. But success did not improve.
The failures still leave the peg roughly `9-13cm` short along the insertion
axis, often with z/contact error. So the current blocker is not simply dense
receding frequency. The blocker is the candidate action distribution plus
outcome scoring around near-contact insertion.

Next repair: add short-prefix and hold/reobserve candidates to the offline
outcome-label export and train/evaluate the scorer on their real outcomes.
Use this panel's failed end states as hard negative evidence for "looks near
the block but is not DP-continuable"; do not convert them into enumerated
recovery cases.
