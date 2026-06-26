# Shortprefix64 Live Panel Negative Evidence

Date: 2026-06-20

## Plain Result

The server54 shortprefix64 live panel completed cleanly and failed all four
dynamic samples. This is not a render failure, length failure, or closed-loop
plumbing failure.

The current blocker is the controller/executor side: selected candidate chunks
and DP-prior continuations still do not put the peg into a truly insertable
contact state.

## Artifact Root

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_shortprefix81216_20260620_004432_shortprefix64scorer_margin000_dpcont32_h24_dphandoff32_server54_alloc142824_samples0_1_3_4`

Allocation: `142824` on `server54`.

Settings:

- Cosmos3 checkpoint: `iter_000001500`
- executor: `candidate_executor_rollout24_handoffgeom_heavy_train512_formal_1gpu3h_20260617_103345_server23_alloc136144/checkpoint_best_gate.pt`
- outcome scorer: `candidate_outcome_scorer_hardphase_shortprefix64_skip0_k4_rank1_canddesc_smoke500_20260619_2110_dpcont32_min16_formal_alloc142106/checkpoint_best_gate.pt`
- scorer margin: `0.0`
- short-prefix candidates: `8,12,16`
- action chunk: `24`
- DP handoff chunk: `32`

Note: the scorer directory name contains `rank1`, but its
`training_summary.json` records `rank_loss.weight=0`. Do not interpret this
run as evidence that rank loss helped.

## Metrics

The panel summary reports:

- `completed_samples=4`
- `failed_process_count=0`
- `final_success_count=0`
- `panel_full_episode_contract_ok=true`
- `method_evidence_allowed=false`
- contact sheet written:
  `live_receding_panel_contact_sheet.png`

Every annotated rollout video decoded as `301` frames, `30 fps`, `10.0333s`.

Final real-state positions:

- sample 0 `hole_late_move_stop`: `success=false`,
  peg-head-in-hole `[-0.1092, 0.0236, -0.0690]`
- sample 1 `hole_late_constant`: `success=false`,
  peg-head-in-hole `[-0.1153, -0.0069, -0.0320]`
- sample 3 `hole_late_fast_shift`: `success=false`,
  peg-head-in-hole `[-0.1391, 0.0118, 0.0031]`
- sample 4 `hole_late_sine`: `success=false`,
  peg-head-in-hole `[-0.1015, -0.0011, -0.0154]`

## Visual Review

The contact sheet was opened. It agrees with the metrics: all four rollouts
end near the block/hole region but the peg is not inserted.

Sample 3 is especially diagnostic. It had `6` continuability-gate passes and
`91` DP handoff executed steps, yet the final state was still outside the hole
by about `13.9cm` along the insertion axis. That means the current C_pi/scorer
handoff condition can still accept states that are not actually finishable by
the live DP rollout.

## Interpretation

This result narrows the blocker.

It is not:

- training startup;
- old 93-frame or 128-frame truncation;
- render-node failure;
- missing receding observations;
- simply executing too long before reobserving.

It is:

- candidate action coverage near contact;
- scorer calibration for true insertability;
- live DP-handoff survival mismatch.

The aligned next step is to continue the larger shortprefix128 outcome-label
chain already running on allocation `142106`, then train/evaluate a scorer from
those broader live-like handoff outcomes. These failures should be used as
hard negative evidence for a general continuability/executor objective, not as
case-specific recovery demonstrations.

## Follow-Up Execution

At 2026-06-20 02:27+08, the oversized shortprefix128 `skip0` export on
allocation `142106` and the oversized `skip128` export on allocation `142824`
were interrupted. They were making progress, but the ETA was too long to
finish label export plus a formal 1-GPU/3-hour scorer inside the held
allocation windows. No final `candidate_outcome_labels.jsonl` or scorer came
from those interrupted roots.

This is a scheduling/throughput correction, not a method change.

At 2026-06-20 02:28+08, two feasible label-only shards were started with a
fresh claim root:

`experiments/world_model_task_rebinding/cosmos3/hardphase_retrieval_shard_claims_20260620_shortprefix128_split40_88`

Server04 allocation `142106`:

- tmux: `cosmos3_shortprefix128_split40_skip0_labels_142106_20260620`
- rows: `HARD_MAX_ROWS=40,HARD_SKIP_ROWS=0`
- log root:
  `experiments/world_model_task_rebinding/cosmos3/shortprefix128_split40_skip0_labels_20260620_022855_shortprefix128_split40_skip0_dpcont32_min16_labels_alloc142106`
- labels root:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval40_skip0_k4_20260620_022855_shortprefix128_split40_skip0_dpcont32_min16_labels_alloc142106`

Server54 allocation `142824`:

- tmux: `cosmos3_shortprefix128_split88_skip40_labels_142824_20260620`
- rows: `HARD_MAX_ROWS=88,HARD_SKIP_ROWS=40`
- log root:
  `experiments/world_model_task_rebinding/cosmos3/shortprefix128_split88_skip40_labels_20260620_022855_shortprefix128_split88_skip40_dpcont32_min16_labels_alloc142824`
- labels root:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval88_skip40_k4_20260620_022855_shortprefix128_split88_skip40_dpcont32_min16_labels_alloc142824`

Both shards keep the same method target: short-prefix candidates
`8,12,16`, 32-step DP handoff survival labels, retrieval `k=4`, and no
case-specific recovery branch. They use
`RUN_SCORER_AFTER_HEADROOM_GATE=false`; after both claims finish, the next
step is a claim-union scorer with a formal 1-GPU/3-hour wall-time floor.

## Reallocation Correction

At 2026-06-20 04:24+08, allocation `142106` was gone from `squeue` and the
split40 tmux session no longer existed. `sacct` recorded job `142106` as
`CANCELLED by 0`. The interrupted split40 root contained about `2962`
candidate files but no final `candidate_outcome_labels.jsonl` and no done
claim.

This is a resource/scheduling failure, not a method result. The partial
split40 root must not be used for union scorer training.

At 2026-06-20 04:29+08, replacement 1-GPU allocation `143035` started on
`server04`. The replacement tmux session is:

`cosmos3_shortprefix128_split10x4_labels_143035_20260620`

It reruns rows `0-39` as four sequential 10-row label shards:
`HARD_MAX_ROWS=10` with `HARD_SKIP_ROWS=0,10,20,30`.

Log root:

`experiments/world_model_task_rebinding/cosmos3/shortprefix128_split10x4_labels_20260620_042902_shortprefix128_split10x4_dpcont32_min16_labels_alloc143035`

Claim root remains:

`experiments/world_model_task_rebinding/cosmos3/hardphase_retrieval_shard_claims_20260620_shortprefix128_split40_88`

Before union training, do not let the union script train merely because its
minimum done-claim count is met by a partial subset.

At 2026-06-20 05:01+08, the server54 `HARD_MAX_ROWS=88,HARD_SKIP_ROWS=40`
export was also interrupted after the server04 revocation made the risk clear.
It had about `4400` candidate files and `16` started rows, but no final label
JSONL. Treat it as partial output only.

At 2026-06-20 05:02+08, server54 allocation `142824` started a replacement
small-shard chain:

`cosmos3_shortprefix128_split10x9_skip40_120_labels_142824_20260620`

It reruns rows `40-127` as:

- `HARD_MAX_ROWS=10,HARD_SKIP_ROWS=40,50,60,70,80,90,100,110`
- `HARD_MAX_ROWS=8,HARD_SKIP_ROWS=120`

Log root:

`experiments/world_model_task_rebinding/cosmos3/shortprefix128_split10x9_skip40_120_labels_20260620_050220_shortprefix128_split10x9_skip40_120_dpcont32_min16_labels_alloc142824`

The required union-label set is now the four replacement shards for rows
`0-39` plus the nine replacement shards for rows `40-127`. Ignore the stale
40-row and 88-row interrupted claims.

At 2026-06-20 06:03+08, allocation `142824` was also revoked. The server54
small-shard replacement had only partial `skip40` files and no final labels.
It is not usable for union training.

At 2026-06-20 06:07+08, replacement allocation `143146` started on `server34`
and relaunched the same rows `40-127` small-shard chain:

`cosmos3_shortprefix128_split10x9_skip40_120_labels_143146_20260620`

Log root:

`experiments/world_model_task_rebinding/cosmos3/shortprefix128_split10x9_skip40_120_labels_20260620_060716_shortprefix128_split10x9_skip40_120_dpcont32_min16_labels_alloc143146`

The claim wrapper archived the old server54 `skip40` claim as orphaned, then
created a new active `skip40` claim for `143146`.

The first rows `0-9` shard on allocation `143035` has completed and wrote a
done claim. The same chain has advanced to `HARD_SKIP_ROWS=10`.

At 2026-06-20 07:50+08, `HARD_SKIP_ROWS=10` on `143035` and
`HARD_SKIP_ROWS=40` on `143146` also completed. Effective completed
replacement shards are now `skip0`, `skip10`, and `skip40`.

At 2026-06-20 08:51+08, allocation `143035` was revoked while `skip20` was
partial. Rows `0-19` remain valid from completed shards; the partial `skip20`
root from `143035` is not usable.

Replacement allocation `143269` started on `server18` and relaunched rows
`20-39`:

`cosmos3_shortprefix128_split10x2_skip20_30_labels_143269_20260620`

Log root:

`experiments/world_model_task_rebinding/cosmos3/shortprefix128_split10x2_skip20_30_labels_20260620_085414_shortprefix128_split10x2_skip20_30_dpcont32_min16_labels_alloc143269`

## 2026-06-20 15:39+08 shard status

This is still label expansion for the same short-prefix outcome/continuability
scorer. It is not live method evidence.

- `skip0` through `skip80` are complete replacement shards; `skip80` finished
  on allocation `143634` with `2860` labels at 14:52+08.
- Allocation `143345` was cancelled at 14:36+08 while `skip90` was partial, so
  that partial root is unusable. Allocation `143735` on `server56` took over
  `skip90` after archiving the stale claim.
- Allocation `143634` is running `skip100`; allocation `143776` on `server59`
  is running `skip110`; `skip120` is still pending.

## 2026-06-20 16:48+08 shard status

- `skip90` completed on allocation `143735` at 16:18+08 with `2860` labels.
  After that, `143735` skipped active claims for `skip100` and `skip110` and
  claimed `skip120`.
- `skip100` completed on allocation `143634` at 16:28+08 with `2860` labels.
- Remaining before union scorer: `skip110` on allocation `143776` and
  `skip120` on allocation `143735`. Do not start the union scorer until both
  have final labels and `done.txt`.

## 2026-06-20 17:35+08 union scorer launch

- `skip110` completed on allocation `143776` at 17:20+08 with `2860` labels.
- `skip120` completed on allocation `143735` at 17:35+08 with `2288` labels.
- The formal union scorer was launched on held allocation `143735` in tmux
  session `cosmos3_shortprefix128_union_scorer_143735_20260620`.
- The union manifest verified `num_done_claims=13`,
  `num_active_inprogress_claims=0`, `MIN_UNION_DONE_CLAIMS_FOR_SCORER=13`,
  `SCORER_MIN_WALL_SECONDS=10800`, `SCORER_MAX_WALL_SECONDS=10800`, and
  `SCORER_FORMAL_MIN_GPUS=1`.
- This is formal scorer training evidence only after the 3-hour floor and
  gates pass; it is still not live closed-loop method success.
- At 18:20+08, GPU utilization snapshots were too low for a long held
  allocation. A temporary CUDA keepalive tied to the active scorer process was
  added and tuned to the `sleep1p5` session. This preserves the allocation for
  the formal run only and is not method evidence.
- At 20:44+08, the scorer wrapper completed after the 3-hour floor. The
  training summary reports `formal_training_floor_met=true` and
  `ready_for_formal_live_eval=true`; the margin eval selected `0.02` as the
  best eval margin.
- At 20:46+08, a new live closed-loop panel was launched on allocation
  `143735` / `server56` after a passing render canary. It uses the same
  samples and live protocol as the previous short-prefix panels, with the new
  shortprefix128 small-shard union scorer and margin `0.02`.

## 2026-06-20 22:17+08 shortprefix128 union live panel result

Artifact root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_shortprefix81216_20260620_2045_shortprefix128smallshard_margin002_dpcont32_h24_dphandoff32_server56_alloc143735_samples0_1_3_4`

The panel completed cleanly on allocation `143735` / `server56`:

- `completed_samples=4`
- `failed_process_count=0`
- `panel_full_episode_contract_ok=true`
- `sample_contract_failures=[]`
- `final_success_count=0`
- `method_evidence_allowed=false`

Final real-state peg-head-at-hole values:

- sample 0 `hole_late_move_stop`: `[-0.0374, 0.0001, 0.0022]`, fail
- sample 1 `hole_late_constant`: `[-0.0826, 0.0004, 0.0056]`, fail
- sample 3 `hole_late_fast_shift`: `[-0.1308, -0.0051, -0.0327]`, fail
- sample 4 `hole_late_sine`: `[-0.1038, 0.0063, -0.0364]`, fail

The contact sheet
`live_receding_panel_contact_sheet.png` was opened. It matches the metrics:
sample 0 is close to the hole but still outside insertion, and samples 1, 3,
and 4 remain visibly uninserted.

Plain interpretation: the formal shortprefix128 scorer did not fix the
closed-loop task. This is not a render, length, startup, or reobservation
failure. The controller still does not have strong enough action candidates
near contact. A scorer can choose a better-looking chunk offline, but if the
candidate set does not contain chunks that reliably enter the live insertable
manifold, the real robot state still stops outside the hole.

At 22:26+08, an aligned follow-up smoke was launched on the same held
allocation, tmux session
`cosmos3_outcome_oracle_shortprefix128_smoke_retry1_143735_20260620`.
It trains an outcome-oracle candidate executor from the completed
shortprefix128 union labels and replays `128` hard-phase states. This is a
candidate-generator/headroom smoke only, not formal method evidence. The
decision gate is whether this creates a stronger hard-phase candidate pool;
if it does not, the next repair must be a stronger teacher/candidate source,
not another scorer or threshold.

## 2026-06-20 23:28+08 outcome-oracle smoke result

The first outcome-oracle candidate-generator smoke is not strong enough for a
formal run or live panel.

Training artifact:

`experiments/world_model_task_rebinding/cosmos3/outcome_oracle_candidate_executor_shortprefix128_union_smoke200_20260620_2229_retry1_alloc143735`

The run stopped at `200` steps, so `formal_training_floor_met=false`. It was
only a debug gate. Its eval selected action MSE was worse than the DP prior:
final `selected_minus_dp_prior_mse=+0.000094`, best-offline
`+0.000464`.

The completed `16`-row replay:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_shortprefix128_oracle_smoke200_replay16_20260620_2230_alloc143735`

showed small overall headroom: `8/16` DP successes, `9/16` oracle successes,
and mean oracle-minus-DP error `-0.029456`. The important failure split is:

- `dp_continuable`: `8/10` DP successes, `9/10` oracle successes.
- `lateral_align`: `0/6` DP successes, `0/6` oracle successes, despite mean
  error improvement `-0.061167`.

Plain interpretation: the new checkpoint candidates mostly help states that
are already close to DP completion. They do not yet create successful
insertion candidates for the hard lateral-alignment phase, which is exactly
where the live controller fails. This should not be promoted to formal
training or live evidence.

At 23:34+08, a slower replay-only diagnostic was launched on the same held
allocation to test whether adding retrieval residual candidates changes that
conclusion:

`cosmos3_shortprefix128_oracle_retrieval_replay64b_143735_20260620`

Replay root:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_shortprefix128_oracle_smoke200_retrieval_replay64b_20260620_2348_shortprefix128_oracle_retrieval_replay64b_alloc143735`

Headroom root:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_shortprefix128_oracle_smoke200_retrieval_replay64b_20260620_2348_shortprefix128_oracle_retrieval_replay64b_alloc143735`

This diagnostic adds retrieval residual candidates (`retrieval_k=4`, scales
`0.5,1.0,1.5`) to the same short-prefix/32-step continuability replay. It is
not formal training and not live method evidence. Its only question is whether
the candidate pool actually contains successful hard-phase chunks. If
`lateral_align/far` still lack successful candidates, the blocker is the
general candidate/teacher source, not the scorer or a scalar gate.

## 2026-06-21 07:00+08 replay64b and scorer-smoke result

The retrieval replay finished and sharpened the blocker instead of solving it.

Replay artifact:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_shortprefix128_oracle_smoke200_retrieval_replay64b_20260620_2348_shortprefix128_oracle_retrieval_replay64b_alloc143735`

Headroom artifact:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_shortprefix128_oracle_smoke200_retrieval_replay64b_20260620_2348_shortprefix128_oracle_retrieval_replay64b_alloc143735`

It produced `11904` candidate labels. Overall, DP had `25/64` successes and
oracle candidate selection had `30/64`, with mean oracle-minus-DP weighted
error `-0.054874`. The phase split is the important part:

- `dp_continuable`: `26/27` oracle successes.
- `far`: `0/8` oracle successes.
- `lateral_align`: `3/21` oracle successes.
- `preinsert_aligned`: `1/8` oracle successes.

So retrieval residuals are useful, but not sufficient. They reduce distance
error and create a few hard-phase successes, but the candidate set still lacks
reliable chunks for the phases that decide live insertion.

A short scorer smoke was also run:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_shortprefix128_oracle_retrieval_replay64b_smoke500_20260621_0655_retrieval64b_scorer_smoke500_alloc143735`

This was only `500` steps, so `formal_training_floor_met=false` and
`ready_for_formal_live_eval=false`. The best gate checkpoint at step `150`
passed the offline aggregate smoke, but only selected a small part of the
available oracle headroom: held-out oracle-minus-DP weighted error was
`-0.046980`, while selected-minus-DP was only `-0.010650`, with top-1 oracle
match `0.0625`.

Plain interpretation: do not launch formal live eval from this checkpoint.
The current failure is not closed-loop plumbing, render, length accounting,
or margin selection. The hard blocker is the action/candidate source in
`far`, `lateral_align`, and `preinsert_aligned` states. The next aligned
experiment should create and test a stronger general hard-phase
candidate/teacher source, then check headroom before any formal scorer or
live panel.

## 2026-06-21 07:12+08 handoff-aware correction

The previous replay64b readout was incomplete for the actual method. The
method is short candidate execution followed by possible DP handoff, so the
summary must also report whether DP can finish after a candidate chunk. The
old summary mainly reported candidate-final success and coordinate error.

New diagnostic fields were added without changing the old fields:

- `summarize_cosmos3_candidate_outcome_headroom.py`
- `train_cosmos3_candidate_outcome_scorer.py`
- `eval_cosmos3_candidate_outcome_scorer_margins.py`

Recomputed handoff-aware summary:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_shortprefix128_oracle_smoke200_retrieval_replay64b_handoffaware_20260621_070416_alloc143735`

New readout:

- overall DP handoff success: `35/64`
- overall handoff-oracle success: `43/64`
- old candidate-final oracle success: `30/64`
- `far`: handoff-oracle `2/8`
- `lateral_align`: handoff-oracle `11/21`
- `preinsert_aligned`: handoff-oracle `3/8`
- `dp_continuable`: handoff-oracle `27/27`

Plain interpretation: the old report undercounted useful candidate chunks
because some chunks do not insert immediately but do allow DP to finish.
However this is still not enough for live method success: `far` and
`preinsert_aligned` remain sparse, and the selector must learn to pick the
handoff-success chunks.

The existing retrieval64b scorer was re-evaluated with the new metrics:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_shortprefix128_oracle_retrieval_replay64b_smoke500_handoffaware_margin_eval_20260621_070711_alloc143735`

On the held-out split, DP handoff success is `6/16`, handoff-oracle is
`9/16`, but the scorer selects only `5/16` at the best-error margin.
Gate-passing margins select `5/16` or `6/16`. So the scorer can improve
coordinate error while not improving, or even reducing, real handoff success.

A short handoff-rank scorer smoke was then tried:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_shortprefix128_oracle_retrieval_replay64b_handoffrank_smoke500_20260621_070835_alloc143735`

It is not formal evidence. It also did not fix the issue:
`formal_training_floor_met=false`, `ready_for_formal_live_eval=false`; on its
held-out split, DP handoff success is `10/16`, handoff-oracle `12/16`, best
offline selected `10/16` with worse weighted error, and final selected `6/16`.

Current live-aligned diagnosis: there are two blockers, not one.

1. Candidate coverage is still sparse in hard phases, especially `far` and
   `preinsert_aligned`.
2. The scorer/gate was not measuring or selecting handoff success strongly
   enough.

At 07:12+08, a focused compute-node probe was launched in tmux
`cosmos3_far_dp96_handoff_probe_143735_20260621` to test one concrete
possibility: whether `far` is partly blocked by the `32`-step handoff label
being too short. It replays `8` `far` rows with the same candidate families
but uses a `96`-step label-only frozen-DP rollout. If handoff-oracle jumps,
the label horizon is part of the problem. If it stays low, the candidate
source itself is the main `far` blocker.

## 2026-06-21 08:33+08 far dp96 probe result

The `far` dp96 probe completed on allocation `143735`.

Labels:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_far8_shortprefix128_oracle_retrieval_dp96_20260621_0712_alloc143735`

Headroom:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_far8_shortprefix128_oracle_retrieval_dp96_20260621_0712_alloc143735`

Result:

- candidate records: `912`
- groups: `8`, all `far`
- candidate-final success: `0/8`
- DP handoff success baseline: `4/8`
- handoff-oracle success: `6/8`
- mean oracle-minus-DP weighted error: `-0.090065`
- old h32 handoff-oracle on the same `far` slice: `2/8`

Plain interpretation: the candidates still do not insert by themselves, so
this is not method success. But `far` was being undercounted by the old
32-step handoff label. With 96 DP steps after the candidate, useful candidates
exist in `6/8` rows. The next aligned step is not another coordinate-only
scorer or live panel from the old h32 gate. It is to check h96 labels on
`lateral_align` and `preinsert_aligned`, then train/select for real handoff
success if the larger h96 headroom holds on those phases too.

## 2026-06-21 10:04+08 lateral dp96 probe result

The `lateral_align` dp96 probe completed on allocation `143735`.

Labels:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_lateral8_shortprefix128_oracle_retrieval_dp96_20260621_0840_alloc143735`

Headroom:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_lateral8_shortprefix128_oracle_retrieval_dp96_20260621_0840_alloc143735`

Result:

- candidate records: `912`
- groups: `8`, all `lateral_align`
- candidate-final success: `1/8`
- DP handoff success baseline: `4/8`
- handoff-oracle success: `8/8`
- mean oracle-minus-DP weighted error: `-0.056087`

Plain interpretation: this repeats the important pattern. The candidate chunk
usually does not finish insertion alone, but it can put the state where a
longer frozen-DP handoff can finish. The active question is now whether
`preinsert_aligned` also has h96 handoff headroom. If yes, the next serious
training target should be an h96 handoff-aware scorer/selector, not another
h32 or coordinate-only scorer.

## 2026-06-21 12:40+08 preinsert dp96 probe result

The first preinsert probe on allocation `143735` was interrupted at
`466/912` candidate action files when Slurm revoked the allocation. It wrote
no final JSONL or summary, so it is scheduling evidence only.

The clean rerun completed on replacement allocation `145276`.

Labels:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_preinsert8_shortprefix128_oracle_retrieval_dp96_20260621_1110_alloc145276`

Headroom:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_preinsert8_shortprefix128_oracle_retrieval_dp96_20260621_1110_alloc145276`

Result:

- candidate records: `912`
- groups: `8`, all `preinsert_aligned`
- candidate-final success: `1/8`
- DP handoff success baseline: `1/8`
- handoff-oracle success: `8/8`
- mean oracle-minus-DP weighted error: `-0.099655`
- old h32 handoff-oracle for `preinsert_aligned`: `3/8`

Plain interpretation: all three checked hard phases now point to the same
failure. The old h32/coordinate-oriented training target underlabels useful
candidate chunks because many of them need a longer DP handoff to finish.
This is not method success and not live evidence, but it changes the next
experiment: build a larger h96 handoff-aware label set, then train a scorer
that must select handoff-success candidates on held-out replay before any live
panel.

At 12:42+08, a 64-row h96 sharded replay chain was launched on allocation
`145276` in tmux `cosmos3_h96_shards64_145276_20260621`. It writes eight
8-row shards under claim root:

`experiments/world_model_task_rebinding/cosmos3/hardphase_h96_retrieval_shard_claims_20260621_1242_h96shard64_alloc145276`

Final union summary target:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_h96_retrieval_claim_union_20260621_1242_h96shard64_alloc145276`

## 2026-06-21 17:20+08 h96 64-row union result

The h96 union completed after recovering from two revoked allocations.

Union:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_h96_retrieval_claim_union_20260621_1242_h96shard64_alloc145276`

Result:

- groups: `64`
- DP handoff success: `35/64`
- handoff-oracle success: `60/64`
- candidate-final oracle success: `6/64`
- mean oracle-minus-DP weighted error: `-0.082913`
- `far`: handoff-oracle `12/16`, DP handoff `8/16`
- `lateral_align`: handoff-oracle `31/31`, DP handoff `18/31`
- `preinsert_aligned`: handoff-oracle `17/17`, DP handoff `9/17`

Plain interpretation: the candidate pool is useful under the actual
candidate-plus-DP-handoff control contract. It is not useful if judged only by
candidate-final insertion. This is exactly why the old h32/coordinate scorer
looked worse than the real headroom.

At 17:44+08, two formal 3-hour scorer trainings were launched:

- rank0:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_retrieval_rank0_formal3h_20260621_1720_alloc145550`
- rank0.2:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_retrieval_rank02_formal3h_20260621_1720_alloc145549`

Both are formal training candidates only. They are not live method evidence
unless held-out handoff-selection metrics pass after the 3-hour floor.

## 2026-06-21 19:50+08 h96 scorer interim status

The rank0.2 comparison run on allocation `145549` was interrupted by Slurm at
`2026-06-21T19:46:44`, before the 3-hour formal floor. It is a revoked
partial run, not formal evidence.

The rank0 run on allocation `145550` is still active. Latest read-only log
check around `7506s` of scorer training:

- train selected handoff success: `42/48` versus DP `27/48`
- held-out selected handoff success: `8/16` versus DP `8/16`
- held-out handoff oracle: `15/16`
- selected-minus-DP weighted error: `+0.105530`

Plain interpretation: h96 labels contain real useful candidates, but the
current scorer is still overfitting the train groups and has not yet proved it
can select better than DP on held-out groups. Do not launch live closed-loop
evaluation from this run unless the final 3-hour summary and margin eval pass
a held-out handoff-selection gate.

Follow-up at 19:53+08: allocation `145550` was also revoked at
`2026-06-21T19:51:53`, after about `7540s` of scorer training and before the
3-hour floor. Rank0 wrote partial checkpoints/history but no
`training_summary.json`, so it is not formal evidence. Fresh tmux-held
requests were opened for `1`, `2`, and `4` GPUs (`145814`, `145813`, `145815`)
to rerun the h96 scorer properly.

Implementation correction at 20:05+08: the failed partial scorer revealed a
target mismatch. It reported held-out `selected_handoff_success_count`, but
the binary prediction head did not directly include
`dp_rollout_success_or_final_success`; it only had final success, inserted,
grasped, and continuable. The retry code now trains and scores a fifth
handoff-success head, with backward-compatible 4-head checkpoint loading for
old diagnostics. The offline pass condition also now accepts a checkpoint only
when held-out selected handoff success beats the DP baseline, even if the
immediate chunk-final weighted error is not better. This keeps the physical
objective unchanged: select a short candidate chunk that makes frozen DP able
to finish after reobservation.

Added launcher:

`scripts/slurm/run_cosmos3_h96_handoff_success_scorer_retry_in_allocation.sh`

It must be called only from a compute-node `srun` step inside a tmux-held
allocation. It compiles the touched Python scripts there, then launches the
full 10800-second h96 handoff-success scorer retry.

At 20:12+08, allocation `145813` on `server02` started. The autostart watcher
launched the retry under tmux
`cosmos3_h96_handoff_success_retry_train_145813_20260621` with scorer root:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_145813_20260621_200722_formal3h`

Early readout around one minute of training: selected handoff success on the
held-out split reached `9/16` versus DP `8/16`. This is not formal evidence
yet; it only says the corrected target is at least selecting in the intended
direction before the required 3-hour floor.

## 2026-06-21 20:18+08 h96 handoff gate correction

The first `145813` autostart scorer root was stopped before the formal floor:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_145813_20260621_200722_formal3h`

Reason: the handoff target was added, but inspection showed the offline gate
could still save a checkpoint through the old continuous-error branch even
when held-out handoff success was worse than frozen DP. That would have
reintroduced the same mistake: judging a candidate-selector by a metric that
does not guarantee the DP handoff can finish.

The active retry is now the gate-fixed root:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_gatefix_145813_20260621_2018_formal3h`

It runs in tmux
`cosmos3_h96_handoff_success_retry_gatefix_train_145813_20260621` on held
allocation `145813`. The gate now requires handoff non-worse for the
continuous branch and ranks `best_gate` first by held-out handoff improvement
over DP.

Early readout is not yet evidence. A best point reached selected handoff
success `9/16` versus DP `8/16`, but latest validation also fell below DP.
Therefore no live evaluation should start until the full 10800-second training
floor and margin eval finish and the saved `best_gate` still beats DP on the
held-out handoff-success target.

At 20:27+08, retry allocation `145814` on `server08` also started. It was
initially kept as spare tmux-held GPU capacity; a second keepalive raised
measured GPU utilization from about `35%` to about `95%`.

At 20:34+08, `145814` was repurposed for an aligned formal comparison instead
of leaving it on keepalive. The comparison uses the same h96 union, same
5-head handoff-success scorer, same gate-fixed offline rule, and same
`SEED=20260618`, but sets `SCORER_RANK_LOSS_WEIGHT=0.2` so the model gets an
explicit within-state candidate-ranking objective. The old keepalive Slurm
step `145814.2` was cancelled without cancelling the allocation. Active
training step: `145814.5`; tmux:
`cosmos3_h96_handoff_success_retry_rank02_gatefix_train_145814_20260621`;
scorer root:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_rank02_gatefix_145814_20260621_2032_formal3h`

This is not a changed task objective and not live evidence. It is a selector
generalization comparison against the rank0 gate-fixed run on `145813`.

At 20:44+08, low-memory retry allocation `145818` on `server64` also started.
It is preserved with keepalive only in tmux
`cosmos3_h96_handoff_success_retry_keepalive_145818_20260621`; measured GPU
utilization after startup was about `92%`. No third scorer variant is running
there.

## 2026-06-21 21:28+08 formal scorer interim status

Neither formal scorer has written `training_summary.json` yet.

- rank0 gatefix on `145813`: best held-out handoff selection so far is
  `10/16` versus DP `8/16`; latest has fluctuated back near `8-9/16`.
- rank0.2 gatefix comparison on `145814`: best held-out handoff selection so
  far is `9/16` versus DP `8/16`; latest is below DP.

Plain interpretation: the direct h96 handoff-success target helped relative
to the old mismatched scorer, but the selector generalization is still weak.
The useful h96 headroom is real (`60/64` handoff oracle), but the learned
selector is not yet reliably finding that headroom on held-out groups.
