# DDP/HDP-Inspired Executor Repair Plan

## Why This Note Exists

Status update, 2026-06-22: the latest source insertion suffix replay gives the
clearest repair direction so far. The current live learned candidate family
can produce y/z improvements and weak gate positives, but its gate-positive
states still gave `0/211` DP96 successes. A diagnostic bank of successful
source insertion suffixes, retrieved by current live peg-hole state and
executed for `32` steps before DP96, gave `18/144` DP96 successes and
`19/144` DP-continuable/contact-stable outcomes. Therefore the next repair is
not another geometry threshold. It is to make insertion-suffix or retrieval-
residual contact chunks a causal candidate family, then train/select against
real DP96 success and contact-stable labels.

Status update, 2026-06-22 later: source-suffix candidates were wired into one
live receding smoke on sample 5. The controller generated `16` source-suffix
candidates in each of three iterations, but the old scorer chose `scale_0.2`,
then `dp_prior`, then `dp_prior`; final success remained false and visual
review showed the peg outside the hole. This means the new action source is
connected, but the old scorer has not learned when to trust it. The next repair
is a source-suffix-aware outcome scorer trained on live snapshot labels with
true DP96 success/continuability, not a repeated live panel with the old
scorer.

Status update, 2026-06-22 16:10 +08: the source-suffix-aware scorer converted
one live sample. The formal scorer trained from live snapshot source-suffix
labels for the required `10800` seconds and exposed `checkpoint_best_gate.pt`
as its live-eval checkpoint. In a full-episode live run on sample 5, the
selector chose `retrieval_resid_srcsuffix_r1_s1_o32` at the first live
decision; that short source-suffix chunk moved the real state onto a
DP-finishable insertion path, and frozen DP completed the episode with
`final_success=true`, `301` observed frames, and `full_episode_length_ok=true`.
This is the first positive conversion evidence for the DDP/HDP-style repair:
a short contact/insertion chunk plus live re-observation plus DP handoff. It
is still one sample, so the immediate next check is the same controller on the
previous failed set `0,2,4,5`, not a success claim.

Status update, 2026-06-22 17:20 +08: the broad source-suffix check exposed
two safety bugs and one deeper coverage blocker. Unsafe suffix reuse required
same-scenario matching plus a source-suffix start-distance cap of `0.02`.
Candidate execution also had to be limited to `8` steps before re-observation,
and pure `dp_prior` now cannot execute when the live continuability gate
`C_pi` is false. After those corrections, sample 00 iter0 replayed all `213`
available candidates from the exact live snapshot and got `0` success, `0`
after-gate states, `0` y/z-improving chunks, and `213/213` worsened y/z.
This means the next DDP/HDP borrow is not another selector pass over the same
pool. DDP works when the action generator/policy prior has learned useful
short chunks in the imagined/current task frame. Our next repair must train or
generate a stronger dense/source/live-state executor or action-candidate
family, then score it with real DP-continuability/contact labels.

Status update, 2026-06-18 09:50 +08: the active next test remains retrieval
residual candidate repair plus a progress/contact/value scorer before live
closed-loop reuse. No new replay/scorer artifact exists yet because every
tmux-held allocation is still pending. Extra one-node probes of apparent
free-GPU holes on `server03`, `server23`, `server34`, and `server36` timed out
and were revoked; this confirms the immediate blocker is scheduler access to a
compute GPU, not a change in the method plan.

Status update, 2026-06-18 10:25 +08: two short diagnostic backfills did run.
They confirm the important method point: retrieval residual and model
candidates create short-horizon progress headroom, especially in hard/far
states, but a tiny `32`-row scorer overfits and fails held-out conservative
selection. Therefore the next serious step is not to use the smoke checkpoint;
it is to let the pending full retrieval shards/union build a larger replay
set, train the progress/contact/value scorer on that larger set, and only then
try live closed loop if the formal offline margin gate passes.

Status update, 2026-06-18 10:43 +08: the `64`-row diagnostic sharpened the
plan. Candidate generation is producing useful headroom; rank-loss scorer
training is the unstable part. A no-rank supervised outcome scorer passed the
conservative offline margin smoke on held-out diagnostic groups, while the
rank-loss scorer failed. The next formal retrieval union should therefore
train the scorer with rank loss disabled by default, then rely on conservative
margin selection before any live closed-loop attempt.

Status update, 2026-06-21 07:00 +08: the larger shortprefix128 retrieval
replay confirms that scorer work is no longer the main bottleneck. Retrieval
residuals improve overall headroom (`25/64` DP successes versus `30/64`
oracle-candidate successes), but `far` remains `0/8`, `lateral_align` is only
`3/21`, and `preinsert_aligned` is only `1/8`. A `500`-step scorer smoke can
pick a small held-out improvement, but it captures only about `1cm` of about
`4.7cm` available oracle headroom and is not formal evidence. The next repair
must improve the general hard-phase action source before another formal live
panel: contact/phase-conditioned candidate generation, stronger teacher
chunks, or another DDP/HDP-style multimodal action source that can enter the
DP-continuable insertion manifold from `far/lateral/preinsert` states.

Status update, 2026-06-21 07:12 +08: the replay diagnosis needed one
correction. For the actual method, candidate-final success is not the only
valid offline headroom signal; a candidate may be valuable if it moves the
live state onto a manifold from which frozen DP can finish. A handoff-aware
summary of the same replay64b data shows DP handoff success `35/64` and
handoff-oracle success `43/64`, not only the old candidate-final oracle
success `30/64`. This does not make the method work, because hard phases are
still sparse (`far` `2/8`, `preinsert_aligned` `3/8`) and the current scorer
does not select handoff-success candidates reliably. The immediate diagnostic
is therefore to separate two issues: whether the `32`-step handoff label is
too short for `far`, and whether the selector objective must be made
handoff-success aware before any formal live attempt.

Status update, 2026-06-21 08:33 +08: the focused `far` dp96 probe answered
the first part. With the same 8 `far` rows and the same candidate families,
candidate-final success is still `0/8`, but handoff-oracle success rises from
the previous h32 `2/8` to h96 `6/8`; DP handoff baseline is `4/8`. Therefore
the old 32-step handoff label was too short for `far` and made useful
candidates look bad. This does not prove live success, because the selector
still has to choose those candidates and the candidate chunk still needs a
long DP handoff to finish. The next aligned plan is to test h96 on
`lateral_align` and `preinsert_aligned`, then train/select against
h96 handoff success only if the held-out handoff headroom is real.

Status update, 2026-06-21 10:04 +08: the h96 result also holds for the first
`lateral_align` slice. Candidate-final success is only `1/8`, but handoff
oracle is `8/8` with DP handoff baseline `4/8`. This strengthens the diagnosis:
the useful unit is "candidate chunk plus sufficiently long DP handoff", not
"candidate chunk inserts by itself." The plan now waits for the matching
`preinsert_aligned` h96 slice before deciding whether to relabel/retrain a
larger h96 handoff-aware scorer.

Status update, 2026-06-21 12:40 +08: the `preinsert_aligned` h96 slice also
supports the same conclusion. Candidate-final success is `1/8`, DP handoff
baseline is `1/8`, and handoff oracle is `8/8`; the previous h32 handoff
oracle for that phase was `3/8`. The next method step is therefore clear:
build a larger h96 hard-phase outcome set and train/select for handoff
success/continuability. A 64-row h96 sharded replay chain has started on
allocation `145276`; formal scorer training must wait for that union summary,
not use the 24-row diagnostic slices as method evidence.

Status update, 2026-06-21 17:20 +08: the larger h96 union confirms the
diagnosis. Across `64` hard-phase groups, DP handoff succeeds on `35/64`,
handoff oracle succeeds on `60/64`, but candidate-final oracle success is only
`6/64`. The immediate method target is therefore a scorer that selects
candidate chunks by h96 handoff success/continuability. Two 3-hour formal
scorer trainings are running: conservative rank0 and a rank0.2 comparison.
No live panel should start unless the held-out handoff-selection metrics pass.

Status update, 2026-06-21 19:50 +08: the rank0.2 comparison was revoked by
Slurm before the 3-hour floor and is not formal evidence. The rank0 run is
still active on allocation `145550`; the latest interim log shows strong train
fit but held-out selection only tied with DP (`8/16` versus `8/16`) and had
worse weighted error. Unless the final summary improves, the next blocker is
not lack of candidates but scorer generalization from h96 handoff labels.

Status update, 2026-06-21 19:53 +08: allocation `145550` was also revoked
before the 3-hour floor, so both h96 scorer trainings are partial-only and no
live eval is allowed from them. Fresh tmux-held allocation requests are queued
for `1`, `2`, and `4` GPUs. The next action remains the same: obtain a valid
allocation, rerun the h96 handoff scorer for the full formal duration, and
only proceed if held-out handoff selection beats DP.

Status update, 2026-06-21 20:05 +08: the retry should not repeat the exact old
objective. The old scorer measured h96 handoff success in evaluation, but did
not directly train or score a `dp_rollout_success`/handoff-success head. The
code now adds that handoff-success target as a fifth binary head and selector
weight. The offline gate now also has a matching handoff-success branch, but
only when selected held-out handoff success is strictly better than DP. This
is the DDP-style handoff correction: choose chunks that move the state onto a
DP-finishable manifold, not chunks that necessarily insert by themselves.

The calibrated residual executor produced real progress but also exposed its
limit. With `EXECUTOR_RESIDUAL_SCALE=0.05`, the first live panel improved over
the direct raw-Cosmos-action branch from `0/4` to `2/4`, but the executor still
failed `hole_late_constant` and `hole_late_sine`. The failures were not pure
task-frame perception failures: the controller often reduced lateral error, but
did not reliably finish contact-rich insertion.

Plain blocker: the current executor is too weakly coupled to the world model
and too conservative. It is a small residual on top of the frozen static DP,
trained by action MSE against teacher chunks. Full residual overfits and hurts
validation; tiny residual can help some cases but cannot reliably create the
last insertion/contact behavior.

## What DDP Actually Does Differently

Primary reference:
https://arxiv.org/html/2603.21017v1

Dream Diffusion Policy is not "a frozen old policy plus a tiny residual." It
works because:

- the policy and world model share the same 3D encoder, so the action policy is
  trained on representations that are already shaped by future-prediction loss;
- during inference, imagined latents can substitute for corrupted/OOD
  observations, and the policy was trained to act from that latent space;
- the OOD mechanism is anchored by real tracking and an initially
  in-distribution state, then uses imagination to bridge back toward an
  in-distribution physical state;
- it uses chunked execution and receding updates, but the chunked world-model
  prediction and action policy are synchronized;
- it still has limitations: it relies on external tracking, assumes the task
  starts in-distribution, and struggles with compounding OOD or low-level
  action disruptions such as slipping.

Direct implication for us: simply asking Cosmos to predict a task path and then
asking a weak residual MLP to nudge a static DP is not the DDP mechanism. The
executor must be trained to treat Cosmos-predicted task/contact latents as its
native input, not as a late sidecar correction.

## What AdaWorldPolicy Adds

Primary reference:
https://ar5iv.labs.arxiv.org/html/2602.20057v1

AdaWorldPolicy reinforces the same point: the world model is not only a video
preview. It is an active supervisor for the action model. The action model and
world model exchange features, and the system uses discrepancy between imagined
and real future observations as a self-supervised adaptation signal. For
contact-rich tasks, it also adds force prediction because visual prediction
alone misses physical interaction shifts.

Direct implication for us: if live real state disagrees with the imagined
Cosmos task path, that error should train or select the executor. Treating
Cosmos only as a one-shot generator and never using real/imagination
discrepancy to adapt or score actions wastes the world model.

## What DiWA/WAM Add

Primary references:

- DiWA: https://arxiv.org/html/2508.03645v1
- WAM: https://arxiv.org/abs/2603.28955

DiWA fine-tunes policies inside a learned world model using imagined rollouts
and a reward/success estimator, instead of only cloning teacher actions. WAM
explains why observation-only world-model latents can be bad for control:
latents trained only for reconstruction may drop action-relevant structure.
WAM adds inverse dynamics/action regularization so the world model encodes
what action caused a state transition.

Direct implication for us: action MSE to teacher chunks is not enough. The
executor needs a progress/success signal in the imagined or replayed task
state, and Cosmos/WAM representations must be action-relevant. Otherwise the
model learns visually plausible task paths that do not tell the executor how
to perform contact insertion.

## What HDP Adds For Contact-Rich Insertion

Primary reference:
https://arxiv.org/html/2411.12982v1

Hierarchical Diffusion Policy targets exactly the failure mode we now see:
end-to-end diffusion policies degrade on contact-rich manipulation because they
do not explicitly model robot-object interaction. HDP splits control into a
high-level contact predictor and a low-level action generator conditioned on
that contact. The contact goal reduces multimodal action ambiguity and gives
the low-level policy a concrete physical target.

Direct implication for us: "move peg toward moved hole" is not enough. The
executor should condition on a phase/contact goal:

- approach/alignment contact;
- peg-head at hole mouth;
- insertion axis alignment;
- insertion progress/depth;
- stable hold/contact preservation.

This is not enumerated failure recovery. It is a universal contact-phase
representation for peg insertion under any target motion composition.

## New Executor Direction

Replace the current main executor objective:

`action = frozen_DP_prior + small_residual(Cosmos_task_path, live_state)`

with a stronger, DDP/HDP-style executor:

```text
live RGB/state history
  -> Cosmos low-frequency task/contact imagination
       predicts: future task frame, peg-hole relative path,
                 insertion phase/contact goal, progress/risk
  -> contact-conditioned executor
       inputs: live state, short observation history, DP prior,
               Cosmos task/contact path, insertion phase/progress target
       outputs: multimodal short action chunk
  -> real execution for a short prefix
  -> real state/video/contact review
  -> progress/value/readout gate decides:
       continue executor, hand off to DP, or reobserve/replan
```

Key changes:

1. Train the executor as an action generator, preferably diffusion or another
   multimodal chunk policy, not only as a deterministic residual MLP.
2. Keep the DP prior as BC regularization or one candidate source, not as a
   hard center that every useful action must stay near.
3. Add phase/contact labels and progress readouts to the executor dataset.
   These can be derived from real simulator state for supervision and
   diagnostics, but controller-facing conditions must remain causal.
4. Train and validate with outcome/progress criteria, not only action MSE.
   A chunk that differs from the teacher but improves peg-hole progress should
   not be automatically punished.
5. Add action-relevant supervision to Cosmos/executor features:
   inverse-dynamics/readout losses, progress classifier, or success classifier
   over predicted task state.
6. Use real-imagination discrepancy as a scoring/adaptation signal: compare
   predicted task/contact progress after a chunk with real observed progress,
   then choose or update the executor accordingly.

2026-06-15 contact-executor mid-run diagnostic: an oracle validation-set
current-phase residual-scale selector has useful offline signal, but a stricter
train-calibrated selector overfits immediately. It chooses `scale=1.0` on the
train split because the deterministic action head memorizes train actions, and
then fails on validation. Therefore the next selector must not be "choose the
scale that minimizes train action MSE." It must be regularized and judged by
held-out progress/value/contact-continuability, with DP kept as a candidate or
regularizer.

## Immediate Implementation Path

This should not start by inventing case-specific rescues. The next aligned
steps are:

2026-06-17 update: the current 8-step candidate executor is not enough. Real
candidate-outcome replay shows stochastic model candidates have small oracle
headroom over DP but `0/672` successful insertions. The immediate repair is a
24-step candidate path: rolling 24-step DP prior export, join with existing
24-step contact/progress labels, short 24-step candidate-executor smoke, then
formal training only if held-out contact/progress and candidate selection beat
DP by a meaningful margin. This is not error recovery; it tests whether the
executor needs a longer contact/insertion chunk to bridge from dynamic target
motion back to the DP-continuable manifold.

2026-06-17 06:05+08 status: the first full 24-step path reached the formal
training stage. The rolling 24-step DP prior export wrote `512/512` records,
the joined contact-executor dataset has `512/512` rows, and the full-512 smoke
kept held-out action-selection signal (`0.006749` selected MSE versus
`0.007762` DP prior, non-DP fraction `0.727`). This smoke is only a gate, not
task evidence. The active formal run is
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_predpath_train512_formal_1gpu3h_20260617_server23_alloc135069`
on allocation `135069`, session `90757`, with `--min-wall-seconds 10800`.
If the formal summary passes the offline gate, the next step is real
closed-loop eval with video/final-state inspection; if it fails, do not return
to 8-step cap tuning.

2026-06-17 10:35+08 update after corrected `ACTION_EXEC_HORIZON=24` live
panel: the 24-step repair is necessary but not sufficient. It ran the full
closed-loop contract on samples `0,1,3,4` and produced `1/4` success.
sample00 succeeded, while sample01/sample03/sample04 ended near the hole but
outside insertion. The key diagnostic is handoff geometry: sample00 got
`85` DP handoff steps after executor progress; sample01/sample04 got `0`, and
sample03 got only `19`. This means the executor still does not reliably put
the real peg/hole state onto the DP-continuable manifold. The next plan is not
to loosen the gate. It is to train/score short chunks against DP-continuable
task-frame geometry, with stronger next-state y/z supervision and higher
DP-continuability value, then run the same live panel only after the formal
3h gate. A low-utilization startup was stopped; the active replacement formal
run is
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_handoffgeom_heavy_train512_formal_1gpu3h_20260617_103345_server23_alloc136144`.

2026-06-17 23:35+08 update after handoff-geometry and outcome-scorer checks:
the heavier handoff-geometry formal checkpoint did not improve live control.
The corrected server61 live panel ran the full `301/300` contract on samples
`0,1,3,4` and succeeded on `0/4`; the contact sheet was inspected and the
failures are physical non-insertions, not rendering or length failures. Real
candidate-outcome replay on `256` rows still shows oracle headroom, but two
short scorer smokes with grouped ranking loss failed to select it on held-out
states. Standard score: final `selected_minus_dp=+0.0720`, best `+0.0404`.
Error-only score: final `+0.0772`, best `+0.0402`. Both runs learned the train
split but not validation. Therefore the next useful work is not another formal
scorer or a looser gate. It is to measure where real candidate headroom exists
and then strengthen the candidate generator/data in those missing phases.

2026-06-17 23:55+08 candidate-headroom analysis shows held-out headroom exists
but the selector cannot exploit it yet. On the same `0.25` validation split,
DP succeeds on `17/64`, oracle-best succeeds on `22/64`, and mean
oracle-minus-DP is `-0.0381`. Candidate provenance/type features improved the
best held-out scorer result to `selected_minus_dp=+0.00316`, but still did not
beat DP or pass the gate. The next concrete step is expanded real outcome data:
a `512`-row replay is running at
`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_handoffgeom512_after_scorerfail_20260617_server61_alloc139069`.
Only after that replay passes should another short scorer smoke run; do not
start formal scorer training from the current `256` rows.

2026-06-18 02:20+08 update after 512 replay, margin checks, and
outcome-oracle executor smoke: the replay/eval path is clean, but the current
candidate generator and selector are both too weak for live control. The
`512`-row replay wrote `14336/14336` outcomes with zero failed rows and showed
oracle headroom (`172` DP successes versus `197` oracle successes,
mean oracle-minus-DP `-0.026978`). However the hard states still rarely have
successful candidates: `far` has `3/94` oracle successes, `lateral_align`
has `15/142`, and `preinsert_aligned` has `15/76`. The 512-row scorer still
overfits held-out rows (`+0.028280` best selected-minus-DP), and conservative
DP-default margin only reaches `-0.002332`, below gate. A new
outcome-oracle candidate executor trained from real replay targets did not
beat the old stochastic generator; unioning old and new candidates gave only
small extra headroom and the union scorer still failed. The DDP/HDP lesson is
therefore concrete now: do not treat Cosmos as a late critic and do not expect
a scalar scorer to rescue weak candidates. The next repair must improve the
action-candidate distribution itself, especially in `far`,
`lateral_align`, and `preinsert_aligned` states, using outcome-supervised and
phase-balanced short chunks before another formal scorer or live panel.

2026-06-18 03:40+08 update after hard-phase balanced smoke: phase-balanced
outcome-oracle imitation is a useful diagnostic but not the fix yet. A
100-step hard-phase balanced executor smoke on allocation `139764` replayed
cleanly, but the generated candidate pool did not beat the old pool as a
standalone source. On `256` replay states it had DP success `91`, oracle
success `101`, any success `102`, mean oracle error `0.139425`, and
oracle-minus-DP `-0.029069`; `far` still had only `1/43` oracle successes.
On the same first `128` rows, old+new union improved mean oracle error
(`0.131715`) but did not increase oracle success and still had `0/18` `far`
success. A union scorer smoke improved only to best held-out selected-minus-DP
`-0.003604`, below the `-0.005` gate. Direct implication: the next change must
create better hard-phase action candidates, not merely rebalance imitation of
the existing weak replay oracle or formalize the current scorer.

2026-06-18 07:10+08 update after broad stochastic hard-phase exploration:
simply widening the candidate samples is also not enough. Across `208` unique
hard rows, the broader pool reached DP success `10/208` and oracle-best success
`18/208`, with weighted mean oracle-minus-DP error about `-0.051`. This is real
error reduction, but the task-completion bottleneck remains: `far` has only
`1/63` oracle successes, while `lateral_align` has `11/93` and
`preinsert_aligned` has `6/52`. Therefore the next repair should not be another
scorer, another margin sweep, or another wider random sample. It must introduce
stronger successful hard-phase chunks, especially for far states, using a
teacher/candidate source that can actually bridge into DP-continuable insertion
geometry.

Prepared next repair: retrieval residual candidates. Instead of another random
sample or scalar scorer, retrieve residual action chunks from phase/contact
neighbors with `future_inserted_within_chunk` or
`future_dp_continuable_within_chunk`, then add those residuals to the current
DP prior. This tests whether successful contact-progress chunks from the same
task-frame phase provide a better candidate distribution for hard states. It
is still a candidate-source diagnostic until real replay proves improved hard
success, especially in `far`.

2026-06-18 07:27+08 execution guard for that repair: the retrieval runner now
uses a candidate-headroom gate before any scorer smoke. The gate is physical,
not cosmetic: oracle-best replay success must beat DP success, overall oracle
success must be nontrivial, mean oracle-minus-DP error must be negative enough,
and `far` rows must include at least one successful oracle candidate when
present. If this fails, the conclusion is that retrieval residual transfer did
not create enough successful hard-state actions; the next step is a stronger
teacher/candidate source, not another ranking loss or margin sweep. The
pending retrieval replay also tests modest residual scaling (`0.5,1.0,1.5`)
around each retrieved neighbor. This is not a special-case rescue; it is a
generic way to handle amplitude mismatch when reusing successful contact-phase
residual chunks from nearby task-frame states.

2026-06-18 07:43+08 execution detail for multi-GPU fallbacks: retrieval shards
use a shared shard-claim root so one-GPU, two-GPU, and four-GPU allocations do
not duplicate the same `HARD_SKIP_ROWS` window if they start in different
order. All retrieval watchers now run a compute-node union summary after their
shards finish, including the one-GPU watcher. The union gate answers the
actual question, "did the retrieval candidate source create enough successful
hard-phase actions overall?", before spending any scorer smoke on the combined
pool. The gate must also show that the `retrieval_success_residual` family
itself contributed an oracle or successful candidate; if legacy/model
candidates are the only winners, retrieval is not counted as repaired.
Shard workers now run replay/headroom only, and the scorer smoke is launched
only by the union gate to avoid per-shard scorer duplication. The union
scorer also requires at least two completed shard claims by default, so the
first one-GPU shard can report headroom without training a scorer that will be
superseded by the later broader union.

2026-06-18 08:34+08 scorer-interface correction: the next scorer is no longer
only a final coordinate-error ranker. Candidate replay outcomes now export
real-end-state contact-progress proxy, progress delta, and a conservative
continuability proxy. The scorer predicts final weighted task error, final
peg-head state, progress/delta, and success/inserted/grasped/continuable
binaries. The selection score combines those terms, and the offline gate
requires both weighted-error improvement and non-regression in contact-progress
delta versus DP. The grouped rank loss targets the same composite physical
value, not only the lowest final coordinate error. This is still an offline
selector gate; it does not replace live closed-loop rollout or video/contact
inspection.

2026-06-18 08:56+08 live-interface correction: the real-outcome scorer now has
an optional live selector path. The candidate/diffusion executor still
generates the action chunks from Cosmos task/contact imagination, live context,
and DP prior. If `--candidate-outcome-scorer-checkpoint` is provided, the live
loop re-ranks those generated chunks with the real-outcome scorer and records
the selector mode plus per-candidate outcome scores. This makes the offline
retrieval-union scorer a possible controller-facing selector after it passes
the gate, instead of leaving it as a report-only diagnostic.

2026-06-18 08:59+08 evidence guard: the live wrapper refuses to use an
outcome-scorer checkpoint as method evidence unless its own training summary
reports `ready_for_formal_live_eval=true` and points to the exact checkpoint.
Short retrieval scorer smokes therefore remain diagnostics until a formal
scorer run satisfies the active GPU/time and offline gate requirements.

2026-06-18 09:02+08 live selector compatibility guard: the optional
real-outcome scorer is only valid with `controller_action_source=
candidate_executor`, because it ranks generated candidate chunks. The live
feature builder maps live candidate names (`dp_prior`, `mean`, `scale_*`,
`sample_*`, `diffusion_*`) back to the replay-training names
(`model_dp_prior`, `model_mean`, `model_scale_*`, `model_sample_*`,
`model_diffusion_*`) before scoring. This prevents a train/eval descriptor
mismatch from becoming a false method failure.

2026-06-18 09:04+08 retrieval margin diagnostics now evaluate
`checkpoint_best_gate.pt` when that checkpoint exists, falling back to
`checkpoint_best_offline.pt` only when no gate checkpoint was saved. The point
is to inspect the checkpoint that actually passed the progress/contact/value
gate, not a best-error checkpoint that might not be admissible for live
selection.

2026-06-18 09:08+08 scheduling state: all retrieval allocations are still
pending on Slurm priority. A smaller legal 1-GPU/3-hour tmux-held backfill
request, job `139892`, was added with `4` CPU / `32G`; if it starts, its
watcher runs the tail hard-phase shard through the same shared claim path.
This is only a resource-shape attempt, not a method change.

2026-06-18 09:21+08 execution hygiene: the retrieval union runner now treats
"no completed shard yet, but another watched allocation owns an active claim"
as a clean scheduling race rather than a replay failure. This keeps the
multi-allocation fallback honest: duplicate shards are avoided, but the first
real completed shard still becomes the authority for headroom and scorer
gating.

2026-06-18 09:27+08 scorer feature hygiene: the real-outcome scorer descriptor
now encodes stochastic checkpoint-model sample metadata (`model_sample_t*_i`
family, temperature, and index) in both replay training and live selection.
This preserves the candidate-generator method while giving the selector causal
provenance that can matter when many sampled chunks are available.

2026-06-18 09:33+08 live scorer guard: live selection now checks a saved
outcome-scorer checkpoint's `candidate_descriptor_names` against the live
descriptor schema before loading it. This keeps future formal live eval tied
to the exact scorer feature contract used during replay training.

2026-06-18 09:34+08 margin-eval guard: conservative margin evaluation now
uses the same descriptor-schema check and records the accepted descriptor
names in its summary. Offline margin diagnostics and live selection therefore
refer to the same scorer feature contract.

1. Do not launch formal/live from the current hard-phase balanced smoke. Its
   256-row replay is clean but not stronger than the old generator, and the
   scorer still misses the held-out gate.
2. Build stronger hard-phase candidate data, not just reweighted imitation:
   add or generate candidate chunks that actually solve `far`,
   `lateral_align`, and `preinsert_aligned` states in real replay. The data
   should preserve DP as the safe default but must contain non-DP successful
   chunks often enough for the scorer to learn a real handoff choice.
3. Build a contact/progress label exporter for existing full-episode H5 rows:
   peg-head-at-hole, hole frame, TCP/peg relation, grasp predicate,
   hole-mouth contact phase, insertion depth/progress, and "DP-continuable in
   <=32 steps" label.
4. Rebuild the executor dataset so each row contains:
   causal Cosmos predicted task path, live/current state, DP prior action
   chunk, contact/progress target sequence, and real final/progress labels.
5. Add a stronger executor trainer:
   contact-conditioned action chunk model with BC regularization to DP/expert
   plus progress/value loss. The first implementation can be a deterministic
   MLP only as a syntax gate, but method evidence needs a multimodal
   diffusion-style or candidate-sampling executor.
6. Add an offline candidate-selection gate before live eval, but only after
   candidate headroom is real and selectable on held-out replay. The
   candidate-headroom diagnostic shows oracle candidates exist on held-out
   rows; the failed scorer smokes show the selector does not yet generalize.
   The `512` replay and union checks show the next precondition is stronger
   hard-phase candidates, not another scorer on the same candidate pool.
7. Current candidate/diffusion gate: before live eval, the formal final
   checkpoint must pass action selection plus scorer checks:
   progress MSE, value MSE, inserted accuracy, DP-continuable accuracy,
   selected-action MSE versus frozen DP, and non-DP candidate usage.
8. Run small overfit, then formal training on the first valid `1/2/4` GPU
   tmux-held allocation while preserving the `3` hour floor, then closed-loop
   panel with full video inspection.

## 2026-06-22 Strict Sample 05 Control

The strict sample 05 rerun keeps the same safety boundary as the sample 00
blocker: same-scenario suffixes, source-suffix start distance at most `0.02`,
8-step execution before re-observation, and no pure DP execution while live
`C_pi` is false.

Under that boundary, sample 05 succeeds:

- output root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_sample5_dist002_exec8_nodpunsafe_20260622_alloc145920`;
- the loop selected close source-suffix chunks for six 8-step receding
  iterations;
- real `C_pi` became true at frame `123`;
- frozen DP handoff completed insertion;
- final real peg-head-at-hole was
  `[0.030646, -0.003007, -0.002958]` and `success=true`;
- the `301`-frame annotated video review sheet and final frame were opened.

This matters because it separates two facts:

- the corrected source-suffix + real-state DP-handoff interface can work;
- the method is still not general because sample 00 has no usable action in
  its corrected first candidate bank.

Next diagnostic: replay all candidates from the strict sample 05 saved banks.
If unselected candidates also cross `C_pi`, prioritize selector training. If
only the selected chain works, prioritize action-generation coverage.

That diagnostic is now partly answered:

- replay root:
  `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_sample05_strict_all_candidates_dist002_exec8_nodpunsafe_20260622_alloc145920`;
- `1300` valid candidate replays;
- `0` direct 8-step success;
- `214` after-gate states;
- iteration `2` had `213/215` after-gate candidates;
- the live-selected source-suffix candidate had `0/6` one-step after-gate
  outcomes in replay.

Interpretation: sample 05 has action headroom; sample 00 does not. The next
distinction is whether sample 05's after-gate states really survive DP96
handoff. That replay is running under
`live_snapshot_replay_sample05_strict_aftergate_dp96_20260622_alloc145920`.

Completed DP96 result:

- `214/214` after-gate candidates replayed cleanly;
- `118/214` reached final DP96 success;
- `208/214` remained DP-continuable;
- `209/214` ended contact-stable;
- `214/214` improved y/z.

Interpretation: sample 05 is a selector/scoring failure, not an action-bank
coverage failure. Sample 00 remains an action-bank coverage failure. The next
repair must separate these two cases: train selection on real DP96
continuability labels where candidates exist, and train/generate better action
candidates where all current chunks move the state in the wrong direction.

## Success/Failure Criteria

Evidence that this repair is working:

- offline held-out progress/success scoring beats frozen DP prior by a clear
  margin, not a tiny `5e-05` action-MSE delta;
- live panel improves success beyond the current residual executor while
  preserving `301/300` contract and inspected videos;
- failures show meaningful contact-progress attempts rather than lateral
  alignment followed by no insertion.

Evidence that it is still blocked:

- contact/progress labels reveal the available dense data lacks successful
  dynamic insertion/contact phases;
- the executor improves lateral y alignment but repeatedly fails x/z insertion
  and contact stability across several panels;
- candidate scoring prefers chunks that look good in predicted readouts but
  fail in real simulator/video, indicating Cosmos/action readouts are not
  physically faithful enough.

If those blockers repeat with concrete evidence, stop and ask the user before
starting another speculative training run.
