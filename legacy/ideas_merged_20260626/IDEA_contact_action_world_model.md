# Contact-Action World Model Reset

Date: 2026-06-23

## One Sentence

Stop treating the controller as a scorer over weak candidates. Train or adapt a
contact/insertion action generator, conditioned by the world model and live
task state, so the robot can physically enter the insertion manifold before
handoff or completion.

## Why The Previous Line Is Not Enough

The latest live panels show that lateral alignment is not insertion. Many
candidate chunks reduce `y/z` error but leave the peg around `x=-0.09` to
`-0.11` in the hole frame, where frozen DP often cannot finish. The old
instantaneous `C_pi` gate is also not a valid training target: saved replays
found both `C_pi=false` states where `candidate + DP96` succeeds and
`C_pi=true` states where real DP handoff fails.

This means the physical blocker is contact/insertion action generation, not
only candidate ranking.

## External Basis

- Tau0-WM style WAMs jointly generate future visual latents and executable
  action chunks, then use an action-conditioned simulator/value interface to
  evaluate candidate consequences:
  `https://arxiv.org/html/2606.01027v1`.
- DreamZero-style WAMs treat world/action prediction as a single physical
  dynamics problem instead of only a semantic VLA policy:
  `https://arxiv.org/html/2602.15922v1`.
- Cosmos Policy-style video policies adapt a pretrained video model into a
  robot policy that emits action-latent frames, future state images, and value
  predictions in one post-training stage:
  `https://arxiv.org/abs/2601.16163`.
- Recent video-action alignment work also uses receding-horizon execution from
  fresh camera frames after each executed segment, which matches the dynamic
  rebinding requirement better than one stale open-loop rollout:
  `https://arxiv.org/html/2603.17808v1`.
- Unified World Models couple video and action diffusion so the same backbone
  can support policy, dynamics, inverse dynamics, and video prediction:
  `https://weirdlabuw.github.io/uwm/`.
- Hierarchical Diffusion Policy argues that ordinary Diffusion Policy is weak
  in contact-rich manipulation because it does not explicitly model robot-
  object contact, and adds contact guidance for trajectory generation:
  `https://arxiv.org/html/2411.12982v1`.
- Reactive Diffusion Policy and force-aware/compliant policies show that
  contact-rich tasks need fast contact/force feedback or compliance-aware
  execution, not only open-loop action chunks:
  `https://reactive-diffusion-policy.github.io/`,
  `https://arxiv.org/html/2409.11047v1`,
  `https://arxiv.org/html/2505.22159v3`,
  `https://arxiv.org/html/2410.19235v1`,
  `https://www.catalyzex.com/paper/fawam-force-aware-world-action-models-for`.

## Active Method Direction

The active method should have three coupled parts:

1. A low-frequency world/task model predicts the changed task frame, contact
   phase, and short future task-state consequences from RGB and causal
   robot/object metadata.
2. A contact-action generator proposes short insertion-aware action chunks.
   The first local target is a source-insertion-suffix or diffusion action
   model trained from the 733 accepted RGB/H5 source data plus live snapshot
   `candidate + DP96` labels. A stronger aligned target is a Cosmos
   Policy-style WAM/policy head that predicts actions, short future state, and
   value together from the same causal RGB/task context. Alternatives such as
   Octo, pi0/OpenPI, OpenVLA, residual RL, or force/contact-aware policies may
   replace frozen DP if DP remains unable to insert from dynamic contact
   states.
3. A value/risk head may rank or reject chunks, but it must be trained on real
   rollout consequences: final success, DP96 success/continuability, contact
   stability, grasp preservation, and insertion-axis progress.

## 2026-06-24 Method Correction

The latest saved-snapshot evidence shows that the causal suffix diffusion
generator has useful candidate coverage, but it is not yet a direct insertion
executor. On the panel0134 saved failure snapshots, generated causal suffix
candidates produced `55/192` `candidate + DP96` successes and every one of the
`16/16` live snapshot groups had at least one DP96-success generated candidate.
However, the generated short chunk itself produced `0/192` direct
`after_success` and `0/192` direct post-chunk gate-ok labels.

This means the next method step is not another scorer-only action selection
round. The current generator can sometimes move the state into a DP-continuable
region, but it does not yet learn the contact-completion behavior that pushes
the peg through the insertion axis while preserving grasp and contact.

The value head is therefore allowed only as a diagnostic selector. The 6/24
full merged margin audit improved DP handoff by only one eval group
(`8/29` versus `7/29`) and still made `7` harmful switches. The panel0134-only
split selected `3/3` handoff-positive causal candidates against `0/3` DP
baseline, but that split is too small to justify live-panel claims. This is
enough to run a selected saved-snapshot replay, not enough to claim method
progress or continue tuning scalar margins as the main line.

The action-generation repair should now prioritize:

1. A selected causal-suffix saved-snapshot replay that actually uses the value
   head to choose one generated candidate per live state, then replays only the
   selected chunk plus DP96 labeling.
2. A direct contact/insertion executor target with positive labels for
   candidate chunks that themselves enter insertion/contact-stable states, not
   only `candidate + DP96` positives.
3. A same-backbone Cosmos Policy-DROID style action/future-state/value
   post-training path, or OpenPI/pi0.5 if Cosmos Policy tooling cannot be
   adapted cleanly. Frozen DP remains the baseline/static prior, not the final
   required executor.
4. Contact/force/compliance-aware correction if direct visual/action chunking
   remains unable to maintain insertion contact.

## 2026-06-24 Later Update: Policy-DROID Action Exists, Replay Is Blocked

The selected causal-suffix value replay has now been run and it does not justify
continuing the scorer line as the main method. The selector chose `11` causal
suffix diffusion candidates and `5` DP-prior candidates, but authoritative
replay reached only `9/16` `candidate + DP96` successes versus `8/16` for the
DP-prior baseline. Direct post-chunk success and direct post-chunk gate-ok were
still `0/16`. The converted-label selector expectation (`15/16`) did not match
real replay (`9/16`), so converted-label summaries must not be treated as
execution evidence.

The same-day Policy-DROID pivot produced a more useful next direction:

- the active 733 clean-dense data can form a strict causal live-prefix
  Policy-DROID input with a prefix-only RGB video, source-H5 history rows, and
  current task-state prompt;
- a reduced `10`-step Cosmos inference completed and emitted a concrete
  `8`-step robot action chunk for prefix frame `106`;
- the extracted action is finite and executable-format, so the base-policy
  action interface is no longer just a plan.

The remaining blocker is replay/evaluation, not action extraction. A
source-prefix replay wrapper was added, but the current held allocation
developed a torch CUDA initialization failure for the ManiSkill/DP replay
environment: `nvidia-smi` sees the H200, while torch reports
`CUDA unknown error` and no usable CUDA device. CPU fallback is not valid
evidence for this project. Therefore the immediate next step is to restore a
CUDA-valid ManiSkill replay step, then evaluate the extracted Policy-DROID
chunk with direct success/gate/contact metrics and DP96 continuability.

That replay environment blocker was bypassed by moving to a new CUDA-valid
tmux-held allocation on `server24`. The first Policy-DROID source-prefix replay
is informative but not sufficient: the 8-step chunk preserved grasp but did not
directly insert, did not pass the conservative handoff gate, and worsened
`abs_y+abs_z`; nevertheless, frozen DP succeeded after the chunk within the
DP96 label rollout. This places Policy-DROID in the same category as a useful
handoff/action-prior source rather than a direct insertion executor until
saved-live replay or additional contact-positive adaptation proves otherwise.

The practical priority order is now:

1. Replay Policy-DROID chunks from saved dynamic live failure snapshots, not
   only the matching source prefix.
2. Use the replay outcome to decide whether Policy-DROID needs fine-tuning,
   contact-positive data, or a different base policy such as OpenPI/pi0.5.
3. Build direct insertion/contact-positive labels regardless, because the
   current live outcome labels mostly teach handoff, not insertion execution.

## 2026-06-24 Live-Snapshot Diagnostic Update

A replay wrapper for saved live snapshots now exists:

`scripts/world_model/replay_policy_droid_action_chunk_from_snapshot.py`

The first same-snapshot Policy-DROID inference attempt was not allowed to
finish: it built the causal live-prefix input and reached Cosmos model setup /
`Loaded 1 samples`, but was interrupted before producing `sample_outputs.json`
or an action chunk. That interruption is a run-control issue, not evidence
that Policy-DROID cannot generate actions from the saved live prefix.

As a fallback diagnostic, the already generated source-prefix Policy-DROID
chunk was replayed from the saved dynamic live snapshot
`sample_00_hole_late_move_stop/iter_00_prefix_f106`. This is intentionally
marked mismatched-prefix and cannot count as same-prefix WAM evidence.

Result:

- direct post-chunk success: false;
- direct post-chunk inserted/contact-stable/gate-ok: false;
- grasp preserved: true;
- near-term task-frame error worsened;
- DP96 after the chunk succeeded after `67` executed DP steps.

This answers the user-level question more concretely: the task is not failing
because insertion is physically impossible. The current action chunks simply
do not yet generate the contact/insertion motion themselves. They can leave a
state that DP sometimes finishes, but that is still a handoff prior, not a
contact-completion executor. The next method needs either same-backbone
Policy-DROID/contact-positive adaptation, a learned residual/contact-aware
executor, or a stronger base policy. Scorer-only selection is exhausted as the
main line.

The same-prefix rerun has now completed and makes the conclusion sharper. From
the exact saved live prefix/history, Cosmos Policy-DROID generated a finite
8-step robot chunk and replayed from the corresponding
`live_state_before_controller.h5`. The chunk preserves grasp but still does
not directly insert or reach a contact-stable/gate-ok state. It worsens
near-term y/z less than the mismatched replay and DP96 succeeds afterward in
`63` steps, but the action itself is still only handoff-positive.

Therefore the next idea is not "find a better scalar selector." The next idea
is to convert this into a contact-positive executor problem:

1. collect direct positives where a short chunk itself reaches inserted,
   contact-stable, or very near insertion-axis continuable state;
2. fine-tune Policy-DROID/action head or train a residual/contact executor on
   those positives plus hard negatives;
3. evaluate on saved live snapshots before any live panel;
4. if direct positives are too sparse or adaptation stalls, switch the base
   action model to OpenPI/pi0/OpenVLA-style policies with the same causal
   observation boundary and the same one-GPU-hour training floor for real
   data adaptation.

The 2026-06-24 direct-positive data audit found one important trap. An older
contact-executor dataset already has direct future-insertion labels
(`185/512`), but it is a historical predicted-task-path branch and its
DP-prior references are missing from the active tree. It should not be
revived as-is. Instead, use it as a schema/provenance warning and rebuild a
clean active manifest whose primary positives are chunks that directly enter
inserted/contact-stable or insertion-axis-continuable states.

The first clean active manifest exists now. It intentionally separates:

1. primary action positives: source suffix chunks that directly insert within
   a 24-step horizon;
2. hard negatives/value labels: live replay chunks that preserve grasp or
   produce handoff labels but fail direct insertion.

This exposed another hidden problem: the old causal-suffix replay labels kept
the consequences of synthetic generated actions, but not the generated action
arrays themselves. Future replay tools must persist every generated action
chunk next to the label. Otherwise the label is useful for reporting and value
calibration, but not for training a contrastive/action executor.

2026-06-24 implementation update: the live action-bank replay script now
persists generated/synthetic candidate action chunks as JSON and records the
path in each outcome label. The direct-contact manifest builder now reads this
persisted chunk before falling back to the original `candidate_action_bank.npz`.
This fixes the data plumbing for future causal-suffix/learned-candidate replay
runs. It does not repair old labels that were already written without actions;
those labels remain value/evidence records unless the replay is regenerated.

The first regenerated selected causal-suffix replay confirms the fix: `11`
generated chunks were persisted and converted into manifest rows. It also
confirms that the current candidates are still not direct insertion actions
(`0/16` direct success/gate-ok, `8/16` DP96 success). The next idea therefore
remains contact-corrective positive data or a stronger contact-aware base
executor, not another value-head margin search.

## What Counts As Evidence

Evidence requires full `301` observed frames and `300` action steps, real final
simulator success, inspected video/contact-sheet evidence, and a causal trace:
what moved, what the robot observed, what short action chunk was generated,
what the world/action model predicted, whether DP or a replacement policy took
over, and why the final state proves task completion.

Offline label replay, source-suffix replay, and oracle state are diagnostics
only. They are useful for training and failure localization, but not method
success.

## What Would Falsify This Direction

The direction is falsified if a contact-action generator trained on successful
source insertion suffixes and live DP96 labels still cannot produce any
candidate that reaches a real insertion/contact-continuable state on saved
live failure snapshots, or if live closed-loop panels continue to fail despite
oracle replay showing no remaining candidate coverage gap. In that case the
next aligned repair is a stronger base policy or contact/force/compliance
control, not another scalar scorer threshold.
