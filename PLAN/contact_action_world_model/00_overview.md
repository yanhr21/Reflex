# Contact-Action World Model Plan

## 2026-06-26 Status

This branch is now historical diagnostic context. The active method is
OpenPI/pi0.5 full-episode execution from step `0`, with causal world-model
conditioning only in the dynamic setting. DP handoff and saved-snapshot
takeover are not main method evidence.

## Boundary

This was the reset after the 2026-06-23 diagnosis that scorer-only selection
is not the main solution. It now remains as archived reasoning for why
candidate selection and DP handoff were insufficient.

Do not delete or reinterpret older evidence. Older scorer, h96, source-suffix,
and dense-Cosmos artifacts remain historical diagnostics. New method claims
must come from this contact-action branch.

## Physical Problem

The robot often reaches a state that is visually and metrically near the hole
laterally, but it is not inside the insertion/contact manifold. Frozen DP was
trained for static scenes and often cannot recover from these dynamic contact
states. A scorer can only select among candidate actions; it cannot create the
missing insertion-axis/contact behavior.

## Method

1. Build a contact/insertion action dataset from accepted source trajectories,
   successful insertion suffixes, source-suffix live candidates, hard negative
   live failure states, and real `candidate chunk + DP rollout` labels.
2. Train a short-horizon contact-action generator that proposes executable
   chunks from current/history RGB, robot state, peg-hole task-frame state,
   action history, and predicted contact phase.
3. Keep world-model/task-frame conditioning causal. Future ground-truth object
   state may be a label/readout target but must not be a controller input.
4. Use a value/risk head only as a consequence model: it predicts success,
   DP96 continuability, contact stability, grasp preservation, and
   insertion-axis progress.
5. Evaluate in receding closed loop: execute only a short chunk, reobserve,
   regenerate/rescore, and hand off to DP or a stronger replacement policy only
   when the real state is physically continuable.

## Candidate Base Policies

- Source-suffix / contact-suffix diffusion executor: first local target because
  the repo already has 733 accepted H5/RGB trajectories and live DP96 labels.
- Cosmos Policy-style action/value/video head: strongest same-family pivot if
  local Cosmos tooling can be adapted cleanly, because it trains action
  latents, future state images, and value prediction inside the video model
  rather than using a separate scorer over external weak candidates.
- Octo fine-tune: open-source diffusion-policy generalist trained on large
  robot data; useful if local action/observation adaptation is tractable.
- pi0/OpenPI: flow-matching continuous-action generalist; useful if dependency
  and action-space integration are feasible.
- OpenVLA-style policy: possible stronger VLA baseline, but integration cost
  is higher and contact insertion still needs fine-tuning or RL.
- Residual RL / VLA-RL style post-training: useful when imitation cannot cover
  dynamic contact states and simulator rewards are reliable.
- Force/contact-aware action model: use simulator contact/force proxies when
  available to model insertion contact quality and compliance.

Frozen DP stays as baseline, static prior, and possible fallback, not as a
required final executor.

## Data To Preserve

Keep these active artifact classes under live experiment roots:

- accepted 733 H5 source set;
- approved RGB render/SFT data and full `301/300` condition exports;
- base static DP checkpoints and current Cosmos/WAM checkpoints;
- successful source insertion suffix banks;
- live snapshot replay labels with real DP96 outcomes;
- current strict-gate scorer checkpoints used only as diagnostic baselines;
- evidence notes and contact sheets for mixed/success/failure panels.

Move duplicate smoke, canary, failed exploratory, stale shard, and
scorer-only artifacts to archive with a manifest.

## Execution Rules

- Do not run project compute on the login node.
- Downloads, `git clone`, and Hugging Face downloads are allowed on the login
  node only while CPU stays under 300% and memory under 40G.
- All training, rollout, replay, rendering, preflight, and evaluation must run
  inside a tmux-held interactive Slurm allocation.
- Do not use one-shot `sbatch`. Reuse held allocations by interrupting
  foreground commands with `Ctrl-C` inside tmux when needed.
- New model training on the 733 RGB/contact-action data must run for at least
  one GPU-hour before it can be interpreted as a training result. Do not get
  stuck in repeated tiny smoke runs.

## First Formal Target

Train a contact-suffix action generator on the 733 source data plus current
live DP96 labels. The first valid training run must:

- use all structurally valid 733 RGB/H5 source rows or a documented full-data
  split;
- include hard negative live states from the failed panels;
- run for at least one GPU-hour;
- write a manifest with data roots, checkpoint path, action horizon, label
  definitions, GPU allocation, and exact command;
- evaluate on held-out live snapshot labels before any live panel;
- only then run a full `301/300` live panel with direct visual review.

## Current Known Negative Result

The old scorer-only path is not enough. h2048, h8192, and h16384 scorers solve
different samples on the same small panel, and simple ensemble rules do not
beat the best single scorer offline. This proves candidate/action generation
and real contact continuability must become first-class training targets.

The first source-suffix contact-action MLP trained for a formal one-GPU-hour
run on the full suffix bank, but finished worse than the mean-action baseline
on held-out source UUIDs (`eval_action_mse=0.0162008` versus baseline
`0.0155657`) and wrote `ready_for_saved_snapshot_replay_gate=false`. This
falsifies the simplest deterministic suffix-regression version of the reset;
the next aligned repair is a causal contact-action generator or Cosmos
Policy-style action/value/video training, not another scorer threshold.

The first live-outcome action diffusion run also trained for one GPU-hour, but
finished with `ready_for_saved_snapshot_replay_gate=false`: held-out selection
tied DP prior at `0.32`, while oracle over the existing candidate pool was
`0.44`. Its label source had no direct final-success or inserted-pose
positives, only `candidate + DP96` handoff successes, and only four source
trajectories. This makes it a negative/limited diagnostic rather than a replay
or live-panel candidate.

The next active repair is a causal source-suffix diffusion generator. It uses
the direct inserted suffixes from the 733 source bank but removes scenario
labels and future first-insert labels, and models the action suffix as a
conditional distribution rather than a deterministic MSE target. If it passes
the one-GPU-hour training gate, it still must be connected to generated-action
saved-snapshot replay before any live panel.

The causal source-suffix diffusion generator completed its first one-GPU-hour
run and passed the source-training replay-readiness gate. This shifts the
active blocker from "can it learn source insertion suffixes?" to "do sampled
chunks from the learned distribution work when restored into saved dynamic live
failure snapshots, and can DP continue afterward?"

## 2026-06-24 Current Gate

The saved-snapshot replay answered the first part partially. Sampled causal
suffix chunks generated from the one-GPU-hour checkpoint were valid on
`192/192` replay attempts and produced `55/192` `candidate + DP96` successes.
All `16/16` saved live snapshot groups had at least one generated candidate
that DP96 could finish from. This falsifies "there are no useful generated
candidates" as the main blocker.

The same replay also showed the remaining physical failure: the generated
short chunk alone produced no direct success and no direct post-chunk gate-ok
state. It is currently a handoff-state generator, not an insertion-completion
executor.

The value head trained on merged replay labels reached the current one-GPU-hour
floor, but the selector is not reliable enough for live evidence. The full
margin audit selected `8/29` handoff successes versus `7/29` for DP prior,
with `7` harmful switches. The tiny panel0134-only eval split selected `3/3`
generated causal candidates versus `0/3` DP prior, but the split has only
three groups. Therefore, do not run a live panel from this selector yet.

The next allowed diagnostic is selected saved-snapshot replay: use the value
head to choose one causal generated candidate per saved live state, replay
that selected chunk from the real snapshot, then run DP96 labeling. If this
selected replay cannot preserve the oracle headroom without many harmful
switches, stop the value-head line and move to a stronger action executor or
base policy.

The selected saved-snapshot replay has now failed to justify the value-head
line as the main method. It selected `11` causal generated candidates and
`5` DP-prior candidates, but authoritative replay reached only `9/16`
`candidate + DP96` successes versus `8/16` for DP prior, and still had
`0/16` direct post-chunk success/gate-ok. Scalar scorer/margin tuning is
therefore no longer the active repair direction.

The first stronger-base-policy pivot is Cosmos Policy-DROID because local
checkpoints and an action sidecar interface already exist. A 2026-06-24
input-only gate on allocation `148680` proved that the active 733 clean-dense
data can build a strict causal live-prefix Policy-DROID input: a `107`-frame
prefix-only video for prefix frame `106`, causal source-H5 history rows, and a
current prefix task-state prompt.

A follow-up 10-step inference probe completed and extracted a concrete
8-step robot action chunk from Policy-DROID for prefix frame `106`. This proves
the action-generation interface is operational, but not that the action is
physically useful. The immediate replay attempt exposed a compute-environment
blocker: in the current held allocation, `nvidia-smi` sees the H200, but torch
CUDA initialization fails for the ManiSkill/DP replay environment, so simulator
replay cannot be trusted from that step. CPU fallback is not evidence.

That CUDA blocker was repaired by moving replay to allocation `148732` on
`server24`. The first source-prefix replay of the extracted Policy-DROID chunk
showed a mixed result: direct post-chunk insertion/gate/contact-stability were
still false and the short chunk worsened lateral task error, but the chunk kept
the grasp and frozen DP succeeded within the DP96 label rollout. Therefore
Policy-DROID is currently a plausible handoff/action-prior source, not yet a
direct contact-completion executor.

The next base-policy step is to replay Policy-DROID chunks from saved dynamic
live failure snapshots before making any broader action-quality claim. In
parallel, the data target must shift toward direct insertion/contact-positive
labels; current live outcome labels contain handoff positives but no direct
post-chunk insertion positives.

The first saved live-snapshot diagnostic was run first with a mismatched
source-prefix Policy-DROID chunk replayed from a saved dynamic snapshot. That
diagnostic preserved grasp but did not directly insert, worsened short-horizon
task-frame error, and did not pass the conservative handoff gate; DP96 still
finished afterward in `67` steps.

The same-prefix Policy-DROID inference has now also completed on that saved
live prefix. It generated an executable 8-step action chunk from the real
observed prefix/history and replayed from the saved dynamic snapshot. The
direct post-chunk result is still negative: no success, no inserted live pose,
no contact-stable state, and no handoff-gate pass. It preserves grasp and
worsens y/z less than the mismatched diagnostic (`+0.00316` instead of
`+0.01257`), and DP96 finishes afterward in `63` steps. This confirms that
Policy-DROID is a better handoff/action-prior path than scalar scorer
selection, but it still does not solve the direct contact-action problem.

Current practical conclusion: frozen DP is still useful as a suffix finisher
from some handoff states, but the method needs an action generator/executor
that deliberately enters insertion/contact-continuable states. If
Policy-DROID remains only handoff-positive across a few saved live snapshots,
the next repair is contact-positive fine-tuning, a residual/contact-aware
executor, or a stronger base policy such as OpenPI/pi0/OpenVLA, not another
scalar scorer.

## 2026-06-24 Direct-Positive Data Audit

The old 2026-06-15 contact-executor join contains the kind of supervision the
reset now needs (`185/512` rows with future insertion inside the chunk), but it
is not directly reusable as an active method dataset. It comes from the older
`cosmos_predicted_action_sidecar` executor branch and its referenced DP-prior
jsonl/chunks are missing from the active tree after cleanup, while the
executor samples and contact labels still exist.

This reveals the current hidden gap: the active reset has plenty of handoff
labels, but too few clean direct-contact labels. The next implementation
should rebuild a clean direct-contact executor manifest from source insertion
suffix positives, contact-progress labels, saved live hard negatives, and
Policy-DROID/suffix replay outcomes. DP96 labels remain secondary handoff
labels; the primary target must be direct post-chunk insertion/contact-stable
or insertion-axis-continuable behavior.

The first clean manifest has now been built under
`direct_contact_executor_manifest_h24_sourcepos_livehard_20260624_alloc148732`.
It contains `2905` horizon-24 source direct positives and one Policy-DROID
live hard negative. This is enough to train a direct-positive action generator,
but not enough to train a strong value/risk head. The missing hard-negative
coverage is a concrete artifact bug: prior causal-suffix replay labels did not
persist the synthetic action chunks, so the labels cannot be cleanly converted
into action-negative rows after the fact.

The active experiment is therefore a direct-contact horizon-24 diffusion
executor trained from the clean direct-positive manifest. This is deliberately
not another scorer. It must run for at least one GPU-hour and then pass saved
snapshot replay before it can be treated as a candidate controller component.

The first valid direct-contact diffusion training run has now crossed the
one-GPU-hour floor:

- output:
  `direct_contact_executor_diffusion_h24_sourcepos_1gpu1h_fix1maxsteps_20260624_alloc148732`
- elapsed: `3660.05s`
- steps: `1396013`
- `formal_one_gpu_hour_floor_met=true`
- `ready_for_saved_snapshot_replay_gate=true`
- best validation `x0_action_mse_mid_t=0.02960`
- final validation `x0_action_mse_mid_t=0.04845`

Interpretation: the model learns the source direct-contact action distribution
well enough to deserve saved-snapshot replay, but the final checkpoint is
overfit. Use `checkpoint_best_eval.pt` for replay. Do not report this as
method success until generated chunks are replayed from saved dynamic live
snapshots and inspected with final-state/video evidence.

The first saved-snapshot replay of this checkpoint is negative. Conservative
`x0_mid` samples from `checkpoint_best_eval.pt` produced bounded actions, but
both tested chunks on
`sample_00_hole_late_move_stop/iter_00_prefix_f106` preserved grasp while
moving the peg laterally away from the hole. Direct insertion, contact-stable
state, conservative gate, and DP96 handoff all failed. This falsifies the
simple source-positive-only executor as a controller candidate for this saved
dynamic state.

Next repair should not be more scalar scoring or repeated replay of the same
source-only generator. The missing supervision is live dynamic hard negatives
and contact-corrective positives with persisted action chunks, or a stronger
base/contact-aware executor that can condition on live post-motion geometry.

Implementation repair after this negative replay: generated/synthetic replay
candidates are now persisted as standalone action-chunk JSON files by
`scripts/world_model/replay_cosmos3_live_action_bank_from_snapshots.py`, and
`scripts/world_model/build_direct_contact_executor_manifest.py` now loads those
persisted chunks before falling back to `candidate_action_bank.npz`. This
removes the previous blocker where causal-suffix replay outcomes could show
DP96 successes but could not become training rows because their generated
actions were lost. Existing old replay labels without persisted chunks remain
non-convertible; the next compute-node step is to rerun the selected
causal-suffix/live-candidate replay with chunk persistence enabled, then build
a new manifest containing real live hard negatives and handoff/action rows.

That compute-node rerun is now complete under
`live_snapshot_replay_selected_causal_suffix_value_head_panel0134_margin0_exec8_dp96_20260624_persistfix1_alloc148732`.
It produced `16/16` valid records, `11` persisted causal-suffix action chunks,
and no process failures. The execution result remains weak: `0/16` direct
success, `0/16` direct gate-ok, and `8/16` DP96 success. The positive outcome is
data plumbing, not controller quality. The rebuilt manifest
`direct_contact_executor_manifest_h24_sourcepos_persistedlivehard_20260624_alloc148732`
contains `2922` rows, including `16` live replay hard negatives, and confirms
`replay_action_loaded_from_persisted_chunk_json=11`. Use these rows as hard
negative/value or contrastive supervision; still collect real live
direct-contact positives before claiming an insertion executor improvement.
