# Candidate Outcome Scorer Diagnosis

Date: 2026-06-17 04:15 +08:00

## Plain Result

The current blocker is the executor/candidate-action set, not the full-episode
video contract and not render scheduling on the current good nodes.

As of `2026-06-17 06:05 +08:00`, the repair being tested is no longer another
8-step scorer. The concrete old failure was that teacher/task-path labels had
24 steps, but the DP-prior chunks feeding the candidate executor had only
8 steps, so the trained executor collapsed to short nudges. The current run
therefore trains a 24-step executor, so it has a long enough physical action
chunk to align and start insertion before handing back to DP.

Current status:

- 2026-06-18 09:50 +08 resource probe update: the retrieval-residual replay
  still has no compute shard started. Five watched allocations remain pending
  (`139754`, `139841`, `139842`, `139861`, `139892`). A direct probe of
  apparently free GPUs on `server03`, `server23`, `server34`, and `server36`
  did not produce an immediate tmux allocation: the corrected single-node
  `salloc --immediate=30` probes made temporary jobs `139932`-`139935`, then
  all timed out and were revoked. This is a scheduling/resource blocker, not
  a replay, scorer, or code failure.
- 2026-06-18 10:03 +08 checked whether `cpu` partition could be used as a
  legal GPU backfill route because it is `UP`, allows account `mayi`, and
  reports GPU TRES. A tmux-held `salloc --immediate=60 -p cpu --gres=gpu:1`
  probe created temporary job `139943`, but Slurm kept it pending because the
  nodes are unavailable or reserved for higher-priority partitions, then
  revoked it on timeout. This route is closed for now.
- 2026-06-18 10:25 +08 obtained two short diagnostic backfills and ran the
  retrieval path on compute nodes. These are smoke/debug runs, not method
  evidence. `139951` on `server05` ran `16` rows at `skip=0`, producing `784`
  successful candidate-outcome records. `139967` on `server34` ran a
  diagnostic union/scorer smoke, a second `16` rows at `skip=16`, and a `32`
  row union/scorer smoke. The diagnostic claim root is isolated from the
  formal retrieval root:
  `experiments/world_model_task_rebinding/cosmos3/hardphase_retrieval_diag16_claims_20260618`.
  The `32`-row union had real candidate headroom but not enough single-chunk
  terminal success: DP success `1/32`, oracle success `2/32`, mean
  oracle-minus-DP error `-0.0586`, meaningful improvements `26/32`, large
  improvements `13/32`, far-phase mean delta `-0.1084`, and retrieval
  residuals as oracle candidates in `7` groups. This validates the
  retrieval-candidate export and progress/value scorer plumbing.
- 2026-06-18 10:25 +08 fixed the scorer gate to match the receding-control
  method. The previous gate trained a scorer only when a short hard-phase
  chunk already produced enough terminal successes. That is too strict for
  the intended closed loop: far/hard chunks may be correct if they make
  contact progress and leave a later chunk or DP handoff easier. The smoke
  and union runners now allow a documented progress-headroom gate in addition
  to the old terminal-success gate. This is not live-eval relaxation; it only
  prevents the progress/contact/value scorer from being skipped when the
  candidate set has the kind of short-horizon progress the method needs.
- 2026-06-18 10:25 +08 negative result: the `32`-row, `50`-step scorer smoke
  is not usable for live control. It improved the train split, but on held-out
  `8` groups it selected worse actions (`selected_minus_dp_prior_weighted_error_mean`
  `+0.0857`, contact-progress delta `-0.0287`). Conservative margin eval had
  `ready_for_conservative_offline_gate=false` and no gate-passing eval margin,
  so the diagnostic checkpoint must not be used. The next real evidence still
  requires the pending full retrieval shards/union and a non-overfit scorer.
- 2026-06-18 10:43 +08 expanded the same isolated diagnostic to `64` hard
  rows via short allocation `139986` on `server14`. The `64`-row union shows
  real candidate headroom: DP success `3/64`, oracle success `6/64`, mean
  oracle-minus-DP error `-0.0544`, meaningful improvements `52/64`, large
  improvements `34/64`, far mean delta `-0.0780`, retrieval residuals as
  oracle candidates in `19` groups, and `1` retrieval-success group. The
  candidate set is therefore not the immediate blocker in this diagnostic.
  The scorer objective is.
- 2026-06-18 10:43 +08 compared rank-loss versus no-rank scorer training on
  the same `64`-row diagnostic. Rank-loss training failed held-out selection
  (`selected_minus_dp_prior_weighted_error_mean` `+0.0693`, contact-progress
  delta `-0.0623`, no conservative eval margin passed). The no-rank control
  passed conservative offline margin eval: `ready_for_conservative_offline_gate=true`;
  best eval margin selected non-DP in `25%` of held-out groups, improved
  weighted error by `-0.0165`, improved contact-progress delta by `+0.0066`,
  and had `0` harmful switches / `3` improved switches. This checkpoint is
  still a short smoke, not live evidence, but it changes the next formal
  scorer default.
- 2026-06-18 10:43 +08 updated the retrieval smoke and retrieval union
  wrappers to default `SCORER_RANK_LOSS_WEIGHT=0.0` while keeping the env
  override. The reason is not to weaken the objective; it is to keep the
  scorer as a supervised progress/contact/value predictor and let the
  conservative margin selector decide whether any non-DP chunk is safe. The
  edited wrappers passed `bash -n` inside compute allocation `139986`.
- full rolling-24 DP-prior export:
  `experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_rollout24_train512_20260617_server54_alloc133179`
  wrote `512/512` records with zero failures;
- full joined training dataset:
  `experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_rollout24_predpath_train512_20260617_server23_alloc135069`
  joined `512/512` rows, with `future_inserted=185` and
  `future_dp_continuable=319`;
- full-512 short smoke:
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_predpath_train512_smoke200_20260617_server23_alloc135069`
  confirmed `action_horizon=24`. Its final held-out selected action MSE was
  `0.006749` versus DP prior `0.007762`, with non-DP selected fraction
  `0.727`, inserted accuracy `0.935`, and DP-continuable accuracy `0.896`.
  This is a training-direction gate, not method evidence;
- formal floor run now running:
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_predpath_train512_formal_1gpu3h_20260617_server23_alloc135069`,
  session `90757`, allocation `135069`, `--min-wall-seconds 10800`,
  `--formal-min-gpus 1`.

The first formal run evaluates every `1000` steps. Its step-1000 and step-2000
checkpoints improved action selection but did not pass the progress/next-state
gate, so a second same-protocol formal run was launched on allocation `133179`
with `--eval-every-steps 50`:

`experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_predpath_train512_formal_1gpu3h_eval50_20260617_server54_alloc133179`

This is not a new success criterion. It preserves the same offline gate and
3-hour floor, but avoids missing the early gate-passing checkpoint observed in
the smoke. The eval50 formal run wrote `checkpoint_best_gate.pt` at step `50`:
selected action MSE `0.007454` versus DP prior `0.007762`, inserted acc
`0.896`, DP-continuable acc `0.909`, progress MSE `0.0448`, and value MSE
`0.0866`. It still must run to the 3-hour floor before that checkpoint can be
used as the formal live-eval candidate.

The required next evidence is the formal 3h summary and then real live
closed-loop video/final-state review. The short smoke alone does not prove
dynamic task completion.

## Corrected 24-Step Live Panel

Latest corrected live root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260617_092327_alloc136144_unknown_node_samples0_1_3_4`

This run used the formal `checkpoint_best_gate.pt` from the eval50 1GPU/3h
run and explicitly passed `ACTION_EXEC_HORIZON=24`. It is the corrected test
of the rolling-24 repair, not the earlier accidental 8-step diagnostic.

Results:

- sample00 `hole_late_move_stop`: success `true`, `301` frames, final
  peg-head-at-hole `[-0.006579, -0.000144, -0.001397]`.
- sample01 `hole_late_constant`: success `false`, `301` frames, final
  peg-head-at-hole `[-0.121433, 0.009202, -0.019700]`.
- sample03 `hole_late_fast_shift`: success `false`, `301` frames, final
  peg-head-at-hole `[-0.103619, -0.002997, 0.004283]`.
- sample04 `hole_late_sine`: success `false`, `301` frames, final
  peg-head-at-hole `[-0.106229, -0.010824, -0.052037]`.

The panel contact sheet
`live_receding_panel_contact_sheet.png` was inspected. The failures are real
physical failures near the hole, not render failures or length-accounting
failures. Controller counts explain the pattern: sample00 got `85` DP handoff
steps, sample01 and sample04 got `0`, and sample03 got only `19`. The failed
rows therefore did not reliably reach the real DP-continuable insertion
geometry before the episode ended.

One concrete readout mismatch was also observed: the selected candidate on
sample04's last chunk predicted high inserted/DP-continuable probability, but
the real live contact context still had y/z error outside the handoff region.
The next aligned fix is a handoff-geometry executor objective: stronger
next-state y/z supervision and a higher DP-continuability score, while keeping
the same causal live interface and the same final-state/video authority.

The first handoff-geometry formal startup,
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_handoffgeom_train512_formal_1gpu3h_20260617_102938_server23_alloc136144`,
was stopped before the formal floor because 30-second GPU sampling stayed at
`11-19%`, below the user's utilization requirement. It is not evidence. The
replacement heavy formal run is active at
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_handoffgeom_heavy_train512_formal_1gpu3h_20260617_103345_server23_alloc136144`
with `FORMAL_MIN_GPUS=1`, `MIN_WALL_SECONDS=10800`, `HIDDEN_DIM=4096`,
`NUM_LAYERS=8`, `BATCH_SIZE=512`, stronger next-state loss/scoring, and higher
DP-continuability weight. GPU sampling showed utilization spikes from `32%`
to `98%`. This run still needs the full 3h floor and offline gate before any
live-eval claim.

The old formal candidate executor can run live. It succeeded on
`sample_00_hole_late_move_stop`, but `sample_01_hole_late_constant` completed
the full `300` actions / `301` frames and failed:

- final peg-head-at-hole: `[-0.080696, -0.001325, 0.009701]`
- final success: `false`
- video contract: `true`
- controller counts: `EXECUTOR_ACTIVE=198`, `DP_HANDOFF=8`,
  `DP_SCAN_TARGET=94`, `INIT_OBS=1`

The last eight executor iterations all selected `scale_0.2` and stayed near
the same x/y state while z stayed outside the insertion gate. The reviewed
framebook shows the peg is still held but not inserted. This is a real physical
failure, not a metric-only artifact.

`sample_04_hole_late_sine` later completed the full live contract too:

- final observed frames: `301`
- final prefix frame index: `300`
- final success: `false`
- final peg-head-at-hole: `[-0.077691, -0.002917, 0.000275]`
- controller counts: `EXECUTOR_ACTIVE=65`, `DP_HANDOFF=119`,
  `DP_SCAN_TARGET=116`, `INIT_OBS=1`

Framebook review of the annotated rollout shows the peg is still outside the
hole at frame `300`. The loop used `25` receding iterations. `scale_0.2`
executor chunks first made x worse to about `-0.25m`, recovered it near
`-0.10m`, then the real-state continuability gate allowed DP handoff because
the current x lower bound is `min_rel_x=-0.1342566`. Frozen DP only reached
about `-0.08m` by the end. Plain meaning: the handoff gate is too optimistic
for dynamic takeover and the current candidates do not provide a reliable
insertion-axis correction.

`sample_03_hole_late_fast_shift` then completed the full live contract:

- final observed frames: `301`
- final prefix frame index: `300`
- final success: `false`
- final peg-head-at-hole: `[-0.103819, -0.001383, 0.003959]`
- controller counts: `EXECUTOR_ACTIVE=160`, `DP_HANDOFF=8`,
  `DP_SCAN_TARGET=132`, `INIT_OBS=1`

Framebook review of frames `270..300` confirms the peg remains outside the
hole. This matches sample01 and sample04: the loop runs, video contract is
correct, but the 8-step executor never reaches a robust insertion state.

## Bad Next-State Formal Attempt

Output root:

`experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260617_formal_1gpu_alloc133179_nextstate_taskscore_diffusion_cap8`

This run was stopped before the formal floor and is not evidence. Reason:
the new next-state head was used in candidate scoring for arbitrary sampled
candidates, but the training labels only supervised the teacher candidate's
future state. Held-out next-state/progress metrics quickly exploded. Continuing
would only produce a formal failure from an ungrounded scoring term.

This is not an argument against action-conditioned outcome scoring. It is the
opposite: the scorer needs real candidate-outcome labels.

## Real Candidate Outcome Export

New script:

`scripts/world_model/export_cosmos3_candidate_outcome_labels.py`

It restores the saved prefix state, executes candidate short chunks in the
real simulator, and records the resulting peg-head-at-hole, grasp/insert state,
success, and per-step records. These labels are supervision/evaluation data;
they are not controller inputs.

Runs on `server54` allocation `133179`:

- smoke:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_smoke_20260617_server54_alloc133179`
  wrote `14/14` outcomes, `0` failures.
- hard32:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hard32_20260617_server54_alloc133179`
  wrote `224/224` outcomes, `0` failures.
- hard32 bigscale:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hard32_bigscale_20260617_server54_alloc133179`
  wrote `192/192` outcomes, `0` failures.

## What It Shows

For the hard32 subset, over true 8-step simulator outcomes:

- DP prior has the best average weighted task-frame error.
- Teacher/scale residuals only help a minority of lateral/preinsert rows.
- Large scales are worse on average and often clip actions.
- No hard32 candidate directly reaches final success in 8 steps.

Therefore the current candidate set does not contain a strong local corrective
move for many post-motion states. Training another scorer on the same
teacher-only labels is not the next useful step.

## Outcome-Scorer Smoke

New script:

`scripts/world_model/train_cosmos3_candidate_outcome_scorer.py`

It joins the original executor feature row with each replayed candidate action
and trains a scorer to predict the real 8-step outcome. Syntax was checked
inside allocation `133179`. A 5-step compute-node CPU smoke wrote:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_smoke_20260617_server54_alloc133179_cpu5step`

The code path works, but the result is negative for the current candidate set:

- joined rows: `288` candidates from `32` base rows
- held-out DP prior true weighted error: `0.2923`
- held-out oracle best over available candidates: `0.2914`
- held-out scorer-selected true weighted error: `0.2997`
- `selected_minus_dp=+0.0074`

Plain meaning: even an oracle over these candidates barely beats DP on the
held-out groups, so a better scorer alone is not the next bottleneck. The
candidate generator/expert chunks must be strengthened first.

## Checkpoint-Generated Candidate Replay

The outcome exporter now also supports checkpoint-generated candidates from
the formal candidate executor: `model_mean`, `model_scale_*`, and
`model_diffusion_*`.

One-row smoke:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_generated_model_smoke1_20260617_server54_alloc133179`

Result: `7/7` model candidates replayed with zero failures. On that row,
`model_diffusion_1` had lower true weighted error than DP (`0.3188` vs
`0.3312`), but it still did not succeed and it was marked
`selector_over_cap=true` by the old residual cap.

Plain meaning: there may be useful stochastic candidates that the old cap/scorer
never allows live to select, but the one-row smoke does not prove a successful
insertion candidate exists. The next diagnostic is a broader checkpoint-
generated candidate replay, judged by real 8-step simulator outcomes and oracle
headroom over DP.

8-row checkpoint-generated replay:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_generated_model_hard8_20260617_server54_alloc133179`

Result: `104/104` outcomes, `0` failures. On all `8` base rows, the oracle
best candidate was a stochastic `model_diffusion_*` chunk:

- mean DP weighted task-frame error: `0.326105`
- mean oracle-best weighted task-frame error: `0.308835`
- mean improvement: `0.017270`
- oracle successes: `0/8`
- all non-DP model candidates were `selector_over_cap`

Plain meaning: stochastic model candidates have real but small short-horizon
headroom over DP, and the old residual cap prevents live from selecting them.
However, even the oracle over these candidates did not reach insertion on the
8-row diagnostic, so cap repair alone is not enough evidence for a new formal
live run.

32-row checkpoint-generated replay:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_generated_model_hard32_20260617_server54_alloc133179`

Result: `672/672` outcomes, `0` failures.

- mean DP weighted task-frame error: `0.317845`
- mean oracle-best weighted task-frame error: `0.297342`
- mean improvement: `0.020503`
- successes: `0` total, `0` DP, `0` oracle-best
- all non-DP model candidates were `selector_over_cap` (`400/400`)

The best rows were still far from insertion. Example: the largest improvement
reduced a fast-shift row from `0.610269` to `0.563284`, but the final
peg-head-at-hole remained `[-0.305236, 0.110864, -0.009080]`. The best
`hole_late_sine` improvement reduced `0.259105` to `0.212745`, but still ended
at `[-0.092337, 0.052353, -0.003925]`.

Plain meaning: the model diffusion samples are directionally useful but not
strong enough. They mostly reduce error by centimeters in an 8-step chunk;
they do not generate a contact/insertion maneuver.

## Outcome-Scorer Gatefix

`train_cosmos3_candidate_outcome_scorer.py` was repaired so that both
`dp_prior` and `model_dp_prior` count as the DP baseline. It also now requires
at least `--min-eval-groups-for-gate` validation groups before saving a
gate-passing checkpoint. This avoids treating a tiny split as formal evidence.

Smoke with 50/50 split:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_generated_hard32_smoke100_val50_20260617_server54_alloc133179`

- joined candidate rows: `672`
- train groups: `16`
- val groups: `16`
- final val DP weighted error: `0.294614`
- final val selected weighted error: `0.289232`
- final selected-minus-DP: `-0.005381`
- final top-1 oracle match: `0.0625`
- formal live ready: `false`

Plain meaning: the scorer can learn a weak ranking signal, but it does not
solve the real blocker. The candidate set still has no successful insertion
candidate, so a formal scorer run on this same 8-step candidate set is not the
right next experiment.

## 24-Step Candidate Path

The 8-step bottleneck was traced to the DP-prior side of the executor dataset:
the teacher actions and predicted task path already have `24` steps, but the
old DP prior npz files have only `8x7` actions. Because the trainer uses the
minimum available horizon, the formal checkpoint became an 8-step executor.

`scripts/world_model/export_cosmos3_executor_dp_prior_chunks.py` now supports
`--rollout-horizon 24`. In that mode it executes frozen DP in the simulator,
reapplies the source target pose each step, and re-queries DP across chunks to
produce a 24-step prior.

Smoke:

`experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_rollout24_smoke2_20260617_server54_alloc133179`

Result: `2/2` records, both `24x7`, zero failures.

Full export running:

`experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_rollout24_train512_20260617_server54_alloc133179`

After this finishes, the next safe step is to join it with the existing
24-step contact labels and run a short 24-step candidate-executor smoke. This
tests whether the current failure is partly an 8-step horizon bottleneck
without changing the closed-loop evaluation objective.

64-row predicted-path smoke:

- DP prior:
  `experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_rollout24_predpath_train64_20260617_server23_alloc135069`
- Joined dataset:
  `experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_rollout24_predpath_train64_20260617_server23_alloc135069`
- Candidate-executor smoke:
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_predpath_train64_smoke100_20260617_server23_alloc135069`

The 64-row dataset has all four active phases, `27/64` rows with future
insertion in the 24-step chunk, and `40/64` rows with future DP-continuability.
The smoke confirms the trainer now uses `action_horizon=24` and `target_dim=168`.
Its best-gate checkpoint at step `60` passed the small held-out offline gate:

- selected action MSE: `0.004897`
- DP prior action MSE: `0.005832`
- selected-minus-DP: `-0.000936`
- non-DP selected fraction: `0.5`
- teacher inserted acc: `1.0`
- teacher DP-continuable acc: `1.0`
- progress MSE: `0.0445`
- value MSE: `0.0684`

This is not method evidence because it is a short 1GPU smoke and the held-out
split has only `10` rows. It is enough to justify the full 512-row 24-step
smoke/formal path after the rolling prior export finishes.

## Handoff-Geometry Live Failure And Ranking-Loss Check

The heavier handoff-geometry formal run completed the required 1GPU/3h floor,
but its corrected server61 live panel did not improve closed-loop behavior.

Live root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260617_205900_server61_icd_alloc139069_server61_samples0_1_3_4`

Result: `0/4` success with the full `301/300` contract and no process
failures. The contact sheet was inspected. The failures are physical
non-insertions, not missing render, old truncation, or closed-loop plumbing.

After this live failure, a 256-row real outcome replay was used to check
candidate selection. It has small but real oracle headroom on the held-out
split:

- DP weighted error: `0.194649`
- oracle weighted error: `0.156490`
- oracle-minus-DP: `-0.038158`

The old scorer failed to select that headroom: final selected-minus-DP was
`+0.030186`.

`train_cosmos3_candidate_outcome_scorer.py` was then extended with an opt-in
grouped ranking loss. The loss compares candidates only within the same
live-state `uuid`, so it directly trains the selection problem instead of only
regressing scalar outcomes. Syntax check passed inside allocation `139069`.

Two short smokes were run on `server61` allocation `139069`:

1. Standard score with success/inserted/grasped terms:
   `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_handoffgeom256_rank1_smoke1000_after_livefail_20260617_server61_alloc139069`
2. Error-only score with all binary score weights set to zero:
   `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_handoffgeom256_rank1_errorscore_smoke1000_after_livefail_20260617_server61_alloc139069`

Both failed the offline gate:

- standard rank loss: final selected-minus-DP `+0.072041`, best `+0.040366`;
- error-only rank loss: final selected-minus-DP `+0.077246`, best `+0.040184`;
- both learned the train split, around `-0.0286` selected-minus-DP, but did not
  generalize to held-out live states.

Plain meaning: this is not just "the scorer lacked a ranking term" and not
just binary readout overconfidence. The current candidate/features/data allow
memorizing which candidate wins on seen states, but do not predict which
candidate wins on new dynamic states. Do not run formal scorer training or
live eval from these checkpoints.

## Candidate Headroom Decomposition

New script:

`scripts/world_model/summarize_cosmos3_candidate_outcome_headroom.py`

Output:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_handoffgeom256_after_livefail_20260617_server61_alloc139069`

This replay diagnostic groups outcomes by `uuid` and asks whether any available
candidate beats the DP prior when executed from the same simulator state.

Overall `256`-state result:

- DP successes: `91`
- oracle-best successes: `101`
- any successful candidate: `103`
- mean DP weighted error: `0.168494`
- mean oracle weighted error: `0.137287`
- mean oracle-minus-DP: `-0.031208`
- meaningful improvements at `>=0.01`: `127`
- large improvements at `>=0.03`: `71`

Same train/val split as the scorer (`val_fraction=0.25`, seed `20260617`):

- train: `192` groups, DP success `74`, oracle success `79`,
  mean oracle-minus-DP `-0.028895`
- val: `64` groups, DP success `17`, oracle success `22`,
  mean oracle-minus-DP `-0.038145`

By phase:

- `dp_continuable`: `99` groups, DP success `83`, oracle success `87`,
  mean delta `-0.013037`
- `far`: `43` groups, DP success `0`, oracle success `1`,
  mean delta `-0.057829`
- `lateral_align`: `75` groups, DP success `6`, oracle success `8`,
  mean delta `-0.031611`
- `preinsert_aligned`: `39` groups, DP success `2`, oracle success `5`,
  mean delta `-0.047205`

Plain meaning: held-out candidate headroom exists; the current problem is not
"there is no candidate better than DP." The harder problem is selecting that
candidate from causal features, and for far/lateral/preinsert states the
candidate set still rarely contains a successful insertion maneuver.

## Candidate-Provenance Scorer Check

The scorer input was extended with candidate provenance/type features:
DP prior, teacher, teacher-scale, model mean, model-scale value, model
diffusion flag/index, and source flags. These are known at runtime and do not
use future object state.

Smoke:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_handoffgeom256_rank1_canddesc_smoke1000_after_livefail_20260617_server61_alloc139069`

Result:

- feature dim increased from `890` to `900`
- final selected-minus-DP: `+0.010995`
- best selected-minus-DP: `+0.003163` at step `100`
- best selected true error: `0.197812`
- DP true error: `0.194649`
- oracle true error: `0.156490`
- best top-1 oracle match: `0.203125`
- offline gate: `false`

Plain meaning: candidate provenance helps a lot compared with the previous
`+0.040` best, but it still does not beat DP. This checkpoint must not be used
for live control. The next aligned step is more real outcome replay and then a
new scorer smoke only if the expanded replay has no structural failures.

The full `512`-row replay has been launched on allocation `139069`:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_handoffgeom512_after_scorerfail_20260617_server61_alloc139069`

## 2026-06-18 Outcome-Oracle And Union Checks

The `512`-row replay finished cleanly:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_handoffgeom512_after_scorerfail_20260617_server61_alloc139069`

It wrote `14336/14336` successful outcome records with zero failed rows.
This rules out a replay/evaluation crash as the current explanation. The
outcomes show real but limited candidate headroom:

- DP successes: `172/512`
- oracle-best successes: `197/512`
- any successful candidate: `202/512`
- mean DP weighted error: `0.167720`
- mean oracle weighted error: `0.140742`
- mean oracle-minus-DP: `-0.026978`

The hard phases remain the weak part of the method:

- `far`: DP success `2/94`, oracle success `3/94`
- `lateral_align`: DP success `11/142`, oracle success `15/142`
- `preinsert_aligned`: DP success `10/76`, oracle success `15/76`
- `dp_continuable`: DP success `149/200`, oracle success `164/200`

Plain meaning: the easy/near-DP states have useful candidates, but the states
that actually need a strong dynamic bridge still rarely contain a successful
candidate.

The 512-row candidate-descriptor scorer still failed held-out selection:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_handoffgeom512_rank1_canddesc_smoke1000_after_scorerfail_20260617_server61_alloc139069`

- final held-out selected-minus-DP: `+0.032771`
- best offline selected-minus-DP: `+0.028280`
- top-1 oracle match: `0.10156`
- offline gate: `false`

The train split selected better-than-DP candidates, so this is overfitting,
not a broken training loop. A conservative DP-default margin sweep helped but
did not make the scorer live-ready:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_handoffgeom512_rank1_canddesc_margin_eval_20260618_server61_alloc139069`

Best margin result selected non-DP on only `0.078125` of held-out groups and
improved selected-minus-DP to `-0.002332`, still below the offline gate.

A new outcome-oracle candidate executor was then tested:

`scripts/world_model/train_cosmos3_outcome_oracle_candidate_executor.py`

It trains action chunks from real replay outcomes: for each observed state,
use the DP prior unless a replayed candidate beats DP by the configured
minimum improvement. This is a causal offline supervision source, not a
future-state controller input.

Short smoke:

`experiments/world_model_task_rebinding/cosmos3/outcome_oracle_candidate_executor_handoffgeom512_smoke1000_20260618_server61_alloc139069`

The smoke learned a small held-out action-MSE improvement, but real simulator
candidate replay showed it is not yet a stronger generator:

- no stochastic samples:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_handoffgeom256_outcomeoracle_exec_smoke_20260618_server61_alloc139069`
  gave oracle success `102/256`, mean oracle error `0.155483`, and mean
  oracle-minus-DP `-0.013015`;
- with `16` Gaussian samples:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_handoffgeom256_outcomeoracle_exec_samples16_smoke_20260618_server61_alloc139069`
  gave oracle success `100/256`, any success `103/256`, mean oracle error
  `0.139860`, and mean oracle-minus-DP `-0.028638`.

The old 256 replay was still slightly better on oracle error
(`0.137287`). Therefore outcome-oracle imitation alone did not create a
better hard-state candidate set.

Unioning old candidates with the outcome-oracle stochastic candidates gives
small complementarity:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_handoffgeom256_union_old_outcomeoracle_samples16_20260618_server61_alloc139069`

- DP success: `91/256`
- oracle success: `102/256`
- any success: `104/256`
- mean oracle error: `0.134673`
- mean oracle-minus-DP: `-0.033821`

But the union scorer still overfit:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_handoffgeom256_union_old_outcomeoracle_samples16_rank1_canddesc_smoke1000_20260618_server61_alloc139069`

- final held-out selected-minus-DP: `+0.053832`
- best offline selected-minus-DP: `+0.011095`
- offline gate: `false`

The best DP-default margin sweep on the union was close but still not enough:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_handoffgeom256_union_old_outcomeoracle_samples16_margin_eval_20260618_server61_alloc139069`

Best selected-minus-DP was `-0.004509` with non-DP fraction `0.09375`, below
the configured gate and too weak for method evidence.

Current diagnosis in plain terms:

- replay/eval is clean;
- candidate headroom exists, so the idea is not dead;
- the learned selector overfits and picks harmful candidates on held-out rows;
- the available candidate generator is still too weak in `far`,
  `lateral_align`, and `preinsert_aligned` states;
- the next useful repair is phase-balanced, outcome-supervised candidate
  generation for those hard states, not formal scorer training, a looser
  threshold, or live eval from the current checkpoints.

## 2026-06-18 Hard-Phase Outcome-Oracle Smoke

Allocation:

`139764`, `server12`, `1` H200, `1` hour. The job ended with Slurm
`TIMEOUT` after the planned compute steps had completed. This was a short
debug smoke, not formal training evidence.

New wrapper:

`scripts/slurm/run_cosmos3_hardphase_outcome_oracle_smoke_in_allocation.sh`

The wrapper runs only inside a compute-node `srun` step. It compiles the
relevant scripts, trains a short hard-phase balanced outcome-oracle candidate
executor, replays generated candidates in the simulator, and summarizes
headroom.

Training smoke:

`experiments/world_model_task_rebinding/cosmos3/outcome_oracle_candidate_executor_hardphase_balanced_smoke100_20260618_alloc139764`

Settings: `100` steps, phase-balanced sampling,
`hard_phase_oracle_min_improvement=0.0`, success-first target selection, and
hard phases `far,lateral_align,preinsert_aligned`. This makes the target
distribution more aligned with the current failure, but it is still only
distilling the best candidate already present in prior replay.

128-row replay:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_balanced_smoke100_replay128_20260618_alloc139764`

- records: `3456/3456`
- failed rows: `0`
- DP success: `45/128`
- new-generator oracle success: `51/128`
- mean oracle error: `0.136933`
- mean oracle-minus-DP: `-0.029957`

Same-128 old baseline:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_handoffgeom512_same128_as_hardphase_smoke_20260618_alloc139764`

- old-generator oracle success: `51/128`
- any success: `52/128`
- mean oracle error: `0.135108`
- mean oracle-minus-DP: `-0.031782`

Same-128 old+new union:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_union_old_hardphase_smoke_same128_20260618_alloc139764`

- oracle success: `51/128`
- any success: `52/128`
- mean oracle error: `0.131715`
- mean oracle-minus-DP: `-0.035175`
- `far` success remains `0/18`

Plain meaning: the new hard-phase balanced generator is complementary with
the old generator on error reduction, but it does not add successful hard
insertions. It improves union mean error, not the hard success bottleneck.

Union scorer smoke:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_old_hardphase_smoke_same128_rank1_canddesc_smoke1000_20260618_alloc139764`

Margin eval:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_old_hardphase_smoke_same128_margin_eval_20260618_alloc139764`

Best held-out margin was `0.2`, with selected-minus-DP `-0.003604`,
non-DP fraction `0.75`, and `0` gate-passing margins. This is directionally
better than harmful selection, but it still fails the `-0.005` improvement
gate and is not live-ready.

256-row replay:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_balanced_smoke100_replay256_20260618_alloc139764`

- records: `6912/6912`
- failed rows: `0`
- DP success: `91/256`
- new-generator oracle success: `101/256`
- any success: `102/256`
- mean DP error: `0.168494`
- mean oracle error: `0.139425`
- mean oracle-minus-DP: `-0.029069`

By phase:

- `dp_continuable`: oracle success `88/99`
- `far`: oracle success `1/43`
- `lateral_align`: oracle success `7/75`
- `preinsert_aligned`: oracle success `5/39`

This is not better than the previous old-generator 256 replay, which had
oracle mean error about `0.137287` and any success `103/256`. The current
hard-phase imitation target therefore did not solve the candidate-generation
problem.

Updated diagnosis:

- short phase-balanced outcome-oracle imitation is not enough;
- the hard-state data still lack action chunks that physically finish far and
  lateral/preinsert insertion cases;
- the scorer can only help after the candidate pool contains more reliable
  successful hard-phase chunks;
- next useful work should change the candidate distribution, not launch formal
  scorer/live eval from this checkpoint.

## Next Useful Step

For method repair, the next concrete step is a hard-phase candidate-generator
repair. Build phase-balanced outcome-supervised candidate data for
`far/lateral_align/preinsert_aligned` states, verify oracle headroom improves
on held-out real replay, and only then train a selector. Do not run formal
scorer training or live closed-loop eval from the current scorer checkpoints.

## 2026-06-18 Broad Hard-Phase Candidate Exploration

Purpose: test whether simply widening stochastic action candidates can create
successful hard-phase chunks. This is still offline replay/debug evidence, not
live controller evidence.

Code change:

- added `HARD_SKIP_ROWS` to
  `scripts/slurm/run_cosmos3_hardphase_exploratory_candidate_replay_in_allocation.sh`
  so later tmux-held allocations can cover disjoint hard-row windows.

Runs:

- `139778` on `server34`: first `64` hard rows at
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_explore64_samples48_20260618_053225_alloc139778`.
- `139778` replayed rows `64-127` at
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_explore64_skip64_samples48_20260618_alloc139778`;
  the allocation timed out before its summary, so `139781` later wrote the
  headroom summary at
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_explore64_skip64_samples48_20260618_alloc139778`.
- `139780` on `server14`: after cancelling a duplicate first-32-row step, it
  ran rows `128-159`, `160-191`, and `192-207` at the corresponding
  `candidate_outcome_headroom_hardphase_explore*_skip*_samples48_20260618_alloc139780`
  roots.

Aggregate over the `208` unique hard rows:

- DP success: `10/208`
- oracle-best success: `18/208`
- weighted mean oracle-minus-DP error: about `-0.051`
- meaningful-improvement rows: `158/208`
- large-improvement rows: `95/208`

By phase:

- `far`: `1/63` oracle successes
- `lateral_align`: `11/93` oracle successes
- `preinsert_aligned`: `6/52` oracle successes

Interpretation: wider stochastic sampling is not useless; it consistently
reduces task-frame error and adds a few lateral/preinsert successes. But it
does not solve the hard candidate distribution, especially `far`, where it
only finds `1` success in `63` rows. The next repair should create stronger
hard-phase teacher/candidate chunks rather than train another scorer on this
same weak pool.

## 2026-06-18 Pending Retrieval-Residual Repair

After the broad-sampling result, the next candidate-source test was prepared
but not yet compute-validated. `export_cosmos3_candidate_outcome_labels.py`
now has opt-in retrieval residual candidates:

- build a bank from rows with `future_inserted_within_chunk` or
  `future_dp_continuable_within_chunk`;
- retrieve phase/contact-neighbor residual chunks;
- add scaled versions of those residuals to the current DP prior as extra
  candidates. The pending runner uses scales `0.5,1.0,1.5` so one amplitude
  mismatch does not decide the whole retrieval test.

This is meant to test a general contact-phase action prior, not a failed-case
rescue table. The compute-node runner is:

`scripts/slurm/run_cosmos3_hardphase_retrieval_smoke_in_allocation.sh`

It first runs `py_compile` and `bash -n` inside the allocation, then runs a
64-row retrieval smoke. A watcher is pending on allocation `139754`.

2026-06-18 07:27+08 update: the same runner now has a post-replay
candidate-headroom gate before any scorer smoke. The gate writes
`retrieval_scorer_gate_summary.json` and only runs the short scorer if
retrieval creates real hard-phase headroom: oracle success must exceed DP
success by at least `2`, oracle success must be at least `4`, mean
oracle-minus-DP error must be `<= -0.025`, and `far` must have at least one
oracle-success candidate when present. If this fails, the result means the
retrieval candidate source is still too weak; the next step is stronger
teacher/candidate generation, not another scorer or margin sweep.

2026-06-18 08:00+08 update: shard workers now run replay/headroom only. They
do not train per-shard scorers. The scorer smoke is launched only by the
compute-node union gate after completed shard claims are combined. The gate
also checks candidate family contribution: `retrieval_success_residual` must
be an oracle family at least once or produce at least one successful-candidate
group. If legacy/model candidates are the only winners, the retrieval repair
is counted as not working. The union scorer also waits for at least two
completed shard claims by default, so the first one-GPU shard can summarize
headroom without triggering a scorer that would be superseded by a later
broader union.

2026-06-18 08:13+08 update: added short-backfill allocation request `139861`
for `1` GPU / `3` hours in tmux session
`cosmos3_hardphase_retrieval_1gpu_3h_20260618`, watched by
`cosmos3_hardphase_retrieval_1g3h_watch_139861_20260618`. It uses the same
shared shard-claim root and the same `skip=0` retrieval shard as the longer
one-GPU request, so whichever allocation starts first can claim the first
hard-phase window and the other will skip duplicates. This is a scheduling
fallback only; the experiment question and gates are unchanged.

2026-06-18 08:30+08 update: retargeted watcher `139861` from `skip=0` to
`skip=256` after a read-only count showed the current contact dataset has
`312` hard-phase rows. The existing 1/2/4-GPU watchers already cover
`0`, `0,64`, and `0,64,128,192`; the 3-hour one-GPU fallback is more useful
as tail coverage if it starts after the broader allocations. Shared shard
claims still prevent duplicates.

2026-06-18 08:34+08 update: upgraded the next outcome scorer to match the
intended progress/contact/value role. Candidate replay outcomes now include a
real-end-state contact-progress proxy, progress delta from the current
contact label, and a conservative continuability proxy. The scorer trains
separate heads for final task error, final peg-head state, contact
progress/delta, and success/inserted/grasped/continuable binaries, and the
selection score uses all of those terms. This is still offline candidate
selection, not live evidence, but it prevents the next scorer from degenerating
into a coordinate-error-only selector. The offline scorer gate now also
requires selected-vs-DP contact-progress delta to meet
`min_selected_progress_delta_improvement`, so a candidate that lowers a
coordinate error while losing physical contact progress does not pass by
default. The grouped rank loss now uses the same real composite value, rather
than choosing the minimum-coordinate-error candidate as the only target. Old
scorer checkpoints are architecture-incompatible and should not be reused for
the new gate.

2026-06-18 08:56+08 update: added a controller-facing entry point for the new
real-outcome scorer. The live loop and panel wrappers now accept
`candidate_outcome_scorer_checkpoint` and can use that checkpoint to re-rank
candidate/diffusion chunks generated by the candidate executor. The per-iter
artifact records whether selection came from the built-in candidate executor
scorer or the real-outcome scorer, plus per-candidate outcome scores. This
does not make any current checkpoint valid for live evidence; it only closes
the interface gap so a future retrieval-union scorer that passes the gate can
actually select live chunks.

2026-06-18 08:59+08 update: guarded that live entry point. If an outcome
scorer checkpoint is supplied to the live panel wrapper, the wrapper now
requires the scorer's `training_summary.json` to report
`ready_for_formal_live_eval=true` and to name that exact
`formal_live_eval_checkpoint`. A short retrieval scorer smoke can still be
used only as an explicit diagnostic; it cannot silently become method live
evidence.

2026-06-18 09:02+08 update: fixed the live candidate descriptor mapping for
the optional outcome scorer. The live candidate executor names generated
chunks as `dp_prior`, `mean`, `scale_*`, `sample_*`, and `diffusion_*`, while
the replay exporter trains the scorer on `model_dp_prior`, `model_mean`,
`model_scale_*`, `model_sample_*`, and `model_diffusion_*` under the
`checkpoint_model` source. The live feature builder now maps to those training
names before scoring. The live loop also refuses
`candidate_outcome_scorer_checkpoint` unless `controller_action_source` is
`candidate_executor`, because the scorer is only defined for ranking generated
candidate chunks.

2026-06-18 09:04+08 update: aligned the retrieval scorer margin diagnostic
with the new gate. The retrieval scorer wrappers now run margin eval on
`checkpoint_best_gate.pt` when that gate-passing checkpoint exists, and only
fall back to `checkpoint_best_offline.pt` otherwise. The done marker records
which checkpoint was evaluated. This avoids reporting a best-error-only
checkpoint as if it represented the progress/contact/value gate.

2026-06-18 09:08+08 scheduling update: the pending jobs are still blocked by
Slurm priority, not by watcher failure. Added one smaller legal tmux-held
backfill request, job `139892`, for `1` GPU / `3` hours with `4` CPU / `32G`
in session `cosmos3_hardphase_retrieval_1gpu_3h_min_20260618`. Watcher
`cosmos3_hardphase_retrieval_1g3h_min_watch_139892_20260618` polls it and
will run the tail hard-phase window (`SHARD_SKIPS=256`) if it starts. This
does not change the method or gates; the shared claim root still prevents
duplicated replay if another pending allocation covers that shard first.

2026-06-18 09:11+08 margin-report fix: the margin evaluator no longer treats
a valid selected-minus-DP value of exactly `0.0` as missing when choosing the
best diagnostic margin. This is a reporting fix only; it preserves the same
progress/contact/value gate.

2026-06-18 09:17+08 immediate scheduling probe: a one-off tmux-held
`salloc --immediate=60` request for the smallest legal shape (`1` GPU,
`3` hours, `4` CPU, `32G`) created temporary job `139895` but did not allocate
within the immediate window. It was cancelled with no node assigned and left no
pending job. Plain meaning: at that moment there was no immediate 1-GPU
3-hour backfill hole available; the active path remains the pending watched
requests.

2026-06-18 09:19+08 scheduler check: account/QOS inspection shows `yanhongru`
is under account `mayi` with QOS `user_yanhongru`, and all pending retrieval
requests have priority `1000`. No higher legal QOS or account route is visible
from the current account state. The concrete scheduling blocker is Slurm
priority, while the code path and watchers are prepared for the first request
that becomes `RUNNING`.

2026-06-18 09:21+08 concurrent-claim guard: the retrieval union runner now
distinguishes "no completed shard because another watched allocation currently
owns an in-progress claim" from "no completed shard and nothing is running."
The first case exits cleanly and records the active claims; the second still
returns a failure code. This prevents overlapping 1/2/4-GPU fallback
allocations from mislabeling normal shard-claim competition as a replay or
method failure.

2026-06-18 09:27+08 scorer descriptor update: stochastic checkpoint-model
samples now expose their candidate family, sample temperature, and sample index
to the real-outcome scorer in both training and live feature construction. This
is causal metadata known before execution. It should make the pending
retrieval-union scorer less dependent on raw action values alone when ranking
many `model_sample_t*_i` candidates. The descriptor names are written into the
training manifest, checkpoint payload, training summary, and live summary so
the larger feature width is auditable.

2026-06-18 09:33+08 live safety guard: the live loop now carries the same
candidate-outcome descriptor schema and refuses a saved scorer checkpoint when
its `candidate_descriptor_names` disagree with the live schema. This prevents
an old or mismatched scorer from silently assigning the wrong meaning to
candidate provenance dimensions during live chunk selection.

2026-06-18 09:34+08 margin-eval safety guard: the conservative margin
evaluator now applies the same descriptor-schema check and records
`candidate_descriptor_names` in its summary. This keeps offline margin
diagnostics aligned with what the live selector would be allowed to load.
