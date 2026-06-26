# Cosmos3 Low-Frequency WM Executor: Selector Blockers

Date: 2026-06-21

This note records the current blocker state for the low-frequency Cosmos3
world-model plus candidate/action-chunk executor line. It is separate from the
older live-panel negative evidence note so the current diagnosis and next
actions do not get mixed with previous run history.

## Current Plain Judgment

This is not "the method has succeeded and only needs scale-up."

The correct problem is now exposed:

- offline h96 replay contains real handoff-continuable action headroom;
- the current live candidate generator and learned selector do not yet
  reliably turn that headroom into real closed-loop insertion success.

Do not go back to direct long-horizon Cosmos actions, and do not tune scalar
thresholds as the main fix. The next work must isolate whether the bottleneck
is candidate generation, selector generalization, or a replay-to-live handoff
gap.

## Current Evidence

Completed h96 64-group union:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_h96_retrieval_claim_union_20260621_1242_h96shard64_alloc145276`

Key result:

- DP handoff success: `35/64`
- handoff oracle success: `60/64`
- candidate-final oracle success: `6/64`

Interpretation: the useful candidate is usually not a one-shot insertion
action. It is a short action chunk that moves the live state into a condition
where frozen DP can finish.

Completed formal scorer attempts:

- rank0 gatefix:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_gatefix_145813_20260621_2018_formal3h`
- rank0.2 gatefix comparison:
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_rank02_gatefix_145814_20260621_2032_formal3h`

Formal results:

- rank0 gatefix met the 10800-second floor. Best gate selected handoff
  `10/16` versus DP `8/16`, but selected-minus-DP weighted error was
  `+0.062411` and contact-progress delta was `-0.050287`. Margin eval also
  remained harmful: at margin `0.5`, weighted delta was still `+0.008804` and
  progress delta was `-0.013800`.
- rank0.2 gatefix met the 10800-second floor. Best gate selected handoff
  `9/16` versus DP `8/16`, but selected-minus-DP weighted error was
  `+0.038276` and contact-progress delta was `-0.017897`. Final checkpoint
  fell to `5/16` versus DP `8/16`. Margin eval stayed harmful: best margin
  `0.075` still had weighted delta `+0.024847` and progress delta
  `-0.013344`.

Interpretation: both formal scorer attempts are unsafe for live rollout. They
can slightly improve a binary handoff count while making the physical handoff
state worse.

Completed selector-blocker audit:

`experiments/world_model_task_rebinding/cosmos3/h96_selector_blocker_audit_20260621_233801_alloc145813`

Audit result:

- all-candidate handoff oracle: `60/64`;
- live-family-only handoff oracle: `58/64`;
- DP handoff baseline: `35/64`;
- all-candidate final-success oracle: `6/64`;
- live-family final-success oracle: `4/64`;
- no `source_uuid` train/val overlap;
- scenario and phase labels overlap, so validation is still small and weak.

Interpretation: live candidate coverage is not the immediate blocker on this
h96 union. The live-available candidate families still contain enough handoff
headroom. The main blocker is selector generalization and checkpoint gating.

Additional safe-headroom audit:

`experiments/world_model_task_rebinding/cosmos3/h96_live_family_safe_headroom_audit_20260621_235518_alloc145813`

Strict condition: candidate is live-family, handoff-successful, weighted error
is not worse than DP, and contact-progress delta is not worse than DP.

- strict safe live-family handoff exists: `46/64`;
- strict safe non-DP candidate exists: `16/64`;
- DP-failure groups: `29/64`;
- strict safe non-DP candidate when DP fails: `11/29`;
- validation split in this audit: strict safe non-DP when DP fails `2/7`.

Interpretation: the data have real safe handoff opportunities, but positive
non-DP examples are sparse. This explains why neural scorers quickly improve
training selection while held-out selection remains worse than DP.

## Main Blockers

1. Selector generalization is poor.

   Training selection improves strongly, but held-out groups do not improve
   reliably and can be worse than DP. This is the core blocker.

   Current live-family safety scorer attempts confirm the same early pattern:
   train selection improves, but held-out selection remains below DP and no
   safe `checkpoint_best_gate.pt` is written yet.

2. Replay oracle and live-executable candidate actions are now partially
   audited.

   Some all-candidate oracle winners are offline-only helper families such as
   `teacher_scale`, `legacy_teacher_scale`, and
   `retrieval_success_residual`. But after removing those families, live-family
   handoff oracle remains `58/64`. This points away from candidate coverage as
   the current first blocker.

3. Gate can still be too permissive.

   The current handoff gate mostly asks whether selected handoff success beats
   DP. It can allow a checkpoint whose handoff bit is slightly better while
   geometry error or contact progress is worse. For insertion, this is risky
   because DP handoff quality depends on pose, contact, and grasp state, not
   only a success bit.

4. Validation is too small.

   Current scorer split is `64` groups with only `16` validation groups. A
   `+1/16` or `+2/16` improvement is too noisy for a strong claim.

5. Source/scenario split leakage risk is bounded but validation remains weak.

   The audit found no `source_uuid` overlap between train and val. Scenario
   and phase names do overlap, which is expected in a 64-group split but means
   the `16`-group validation set is still not strong evidence.

6. Visual audit is not structured enough.

   The live panel note says the contact sheet was opened, but some summaries
   can still record review status as needing direct review. For failures this
   is mostly a bookkeeping problem. For any future claimed success, visual
   evidence must be written back structurally: contact sheet/video path,
   reviewer, final grasp/hold, insertion contact, and real simulator final
   state.

7. Repo state is dirty.

   There are many modified and untracked scripts, wrappers, plans, and docs.
   Until stable files are organized and committed, reproduction risk is high.

## Likely Hidden Problems

1. Live-candidate family coverage is not the first blocker for this union.

   The live-family-only audit keeps `58/64` handoff oracle successes. The next
   attempt should train a selector on the live-family distribution instead of
   spending a live panel on an unsafe scorer.

   The stricter safe-non-DP audit narrows this further: only `16/64` groups
   have a strict safe non-DP candidate, and only `11/29` DP-failure groups can
   be safely improved by non-DP. The next selector/data work must focus on
   those sparse positives.

2. DP handoff labels may not equal live DP handoff.

   Restored replay state can say DP succeeds, while the real closed loop after
   compounding execution error may still fail. The latest panel already showed
   DP handoff execution without final success.

3. Scorer input descriptors may be too abstract.

   In live eval, many generated candidates may appear as broad
   checkpoint-model descriptors. Offline training includes richer family
   labels. The selector may learn a family prior that does not transfer to the
   live candidate distribution.

4. Current live panel is too small for direction claims.

   A `0/4` or `1/4` panel is enough to reject a success claim, but not enough
   to prove a method direction is generally good. Use stricter offline/margin
   gates first, then run small live panels only from a formal passing
   checkpoint.

5. Simulator state is still used for labels and diagnostics.

   This is allowed for training labels, causal metadata, and diagnostics.
   Final controller evidence must still use RGB/RGB-D-derived state or slots
   as the controller input. Oracle/state-only results cannot be reported as
   the main method.

## Required Next Actions

1. Do not run live eval from the two current scorer checkpoints.

   They are formal but unsafe: the handoff bit improves slightly while
   geometry/contact progress worsen.

2. Train/evaluate a live-family-only safety selector.

   The selector should use only DP prior plus candidate families available to
   the live loop. It should rank against the group DP prior and live-family
   handoff oracle, and the gate must require held-out handoff improvement
   without worse weighted geometry error or contact progress.

   Current formal runs:

   - default-capacity safety scorer:
     `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_scorer_20260621_234650_alloc145814_formal3h`
   - low-capacity regularized safety scorer:
     `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_lowcap_20260621_234950_alloc145818_formal3h`

3. If both formal safety scorers fail, expand/rebalance labels around
   DP-failure states with safe non-DP live candidates.

   The target is not case-by-case recovery. It is a causal, reusable selector
   training set for the same task-frame rebinding interface: current observed
   state, live-generatable short chunk, safe transition to DP-continuable
   geometry/contact.

4. Only after that gate passes, run a small full `301/300` live panel on
   samples `0,1,3,4`.

   Record final real simulator state, DP handoff step count, candidate
   mode/family distribution, visual grasp/hold, and insertion conclusion.

5. Organize evidence and repo state.

   After formal summaries are available, update:

   - `TODO/cosmos3_lowfreq_wm_executor/00_active.md`;
   - this focused docs directory;
   - relevant stable scripts and wrappers.

   Then commit stable plan/script/doc changes so later work does not depend on
   a dirty worktree.

## Evidence Boundary

Do not claim method success unless there is:

- formal scorer summary;
- margin/offline gate evidence;
- full `301/300` live rollout;
- final real-state metric;
- opened and structured video/contact-sheet review;
- comparison against DP baseline.

If the live-family safety selector still cannot pass, the next method step is
selector data/feature/loss repair or larger balanced h96 labels, not scorer
threshold tuning.
