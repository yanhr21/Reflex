# Cosmos3 Low-Frequency WM Executor Current Instructions

Date: 2026-06-22

These instructions summarize the current method boundary, blocker judgment,
and next decision tree for the Cosmos3 low-frequency world-model plus
candidate/diffusion executor line.

## Plain Current Judgment

The method has one narrow live success, but not a broad success yet.

Latest 2026-06-22 update: the formal rank-loss source-suffix scorer finished
its 3-hour run and failed the live-eval gate. The sample00 offset diagnosis
also produced an important but narrower result: offsets `64,48,32,24` can make
source-suffix candidates available and produced one full-episode success, but
the same protocol inside panel `0,2,4,5` failed on sample00. The stable
conclusion is coverage improved, not that sample00 is solved.

The useful discovery is narrower and important: offline h96 replay shows
handoff-continuable action headroom, and the 2026-06-21 blocker audit shows
that most of this headroom still exists after filtering to live-available
candidate families. The 2026-06-22 live panel then showed that this offline
headroom still does not convert into live task completion.

Current plain status: the old h96/live-family scorer could sometimes move the
peg near the changed hole, but could not reliably make the real live state
finish insertion. The source-suffix repair changed the action source and the
selector target: successful source insertion suffixes are now candidate chunks,
and the selector is trained on real live-snapshot `candidate + DP96` outcomes.
On sample 5 this converted into a real full-episode live success. On sample00,
one run succeeded and one same-protocol panel rerun failed. The remaining
blocker is not just generality across samples; it is closed-loop stability of
source-suffix execution plus DP handoff/contact continuability.

Do not frame this as error detection and recovery. Do not return to direct
long-horizon Cosmos action rollout. Do not keep tuning scalar gates as the main
fix.

The active direction remains:

1. Cosmos provides low-frequency task/contact imagination.
2. The executor proposes short candidate or diffusion action chunks.
3. A progress/contact/value scorer selects the chunk.
4. The system executes only a short chunk, reobserves the real state, and
   hands off to DP only when the real state is actually continuable.

## Main Blockers

1. Selector generalization is poor.

   Training selected-handoff improves, but validation does not improve
   reliably and can be worse than DP. This is the core blocker.

2. Candidate coverage is not the main blocker in the current h96 union.

   Audit path:
   `experiments/world_model_task_rebinding/cosmos3/h96_selector_blocker_audit_20260621_233801_alloc145813`.

   All-candidate handoff oracle is `60/64`; live-family-only handoff oracle is
   still `58/64`. Therefore the immediate blocker is not "there are no useful
   live candidates." The blocker is that the selector/gate cannot choose safe
   candidates on held-out groups.

3. The gate may still allow bad checkpoints.

   A checkpoint can slightly beat DP on handoff count while having worse
   geometry error or contact progress. For peg insertion this is dangerous:
   DP handoff quality depends on pose, contact, and grasp state, not only a
   binary success bit.

4. Validation is too small.

   Current h96 scorer validation is only `16` groups. A `+1/16` or `+2/16`
   improvement is too noisy for a strong claim.

5. Source/scenario split leakage risk is partially bounded, but validation is
   still weak.

   The 2026-06-21 audit found no `source_uuid` overlap between train and val,
   but scenario and phase names overlap. That means this is not a duplicate
   source leak, but the validation set is still only `16` groups and does not
   prove robust generalization.

6. Live visual audit is not structured enough.

   Some notes say a contact sheet was opened, but summaries can still say
   `visual_review_status=needs_direct_agent_or_user_review`. A future success
   cannot count as a major result unless visual review is written back
   structurally with video/contact-sheet path, reviewer, grasp/hold status,
   insertion contact status, and final real simulator state.

7. Repo state is dirty.

   Many scripts, wrappers, plans, and docs are modified or untracked. Stable
   method logic must be organized and committed after current formal summaries
   are recorded.

8. Offline scorer predictions do not match live execution closely enough.

   The 2026-06-22 live panel showed repeated cases where the scorer chose
   `dp_prior`, `mean`, or short-prefix candidates that were predicted to be
   handoff-useful, but the real live rollout moved y/z away from the hole or
   failed to complete insertion. This is not a scheduling or length-contract
   failure. It is a live controller/scorer mismatch.

9. Geometry handoff is not sufficient for insertion.

   In `sample_03_hole_late_fast_shift`, the system reached states that passed
   the continuability geometry gate and executed DP handoff, but final success
   remained false. The final state was close
   `[-0.050987, 0.001507, 0.003059]`, yet insertion did not complete. The next
   target must include grasp/contact/insertion progress after handoff, not only
   near-hole pose.

9b. sample00 panel rerun shows C_pi-positive DP handoff can still fail.

   Evidence note:
   `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_panel0245_offsets64_sample00_resample_failure.md`.

   Panel root:
   `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658`.

   sample00 completed the full `301/300` contract but ended
   `success=false`, final peg-head-at-hole
   `[-0.095756, 0.016172, -0.065285]`. The visual review sheet was opened and
   shows no final insertion. In that run, source-suffix chunks made C_pi true
   at frame `168`, then DP96 moved the real state to
   `[-0.097337, 0.006050, -0.049492]` and broke continuability. Late frames
   had `0` source-suffix candidates.

   Plain conclusion: offsets `64,48,32,24` address an early candidate
   coverage miss, but the method still needs handoff scoring from real
   DP-rollout continuability/contact outcomes. Do not report the earlier
   sample00 standalone success as a stable fix.

10. Latest action-bank replay shows candidate-action coverage is also a live
    blocker at early decision states.

    Output:
    `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_allcand_iter0_20260622_0820_alloc145920`.

    Replayed all `213` saved candidates from the first action bank of sample 01
    and all `213` saved candidates from the first action bank of sample 03.
    Total result: `426` live labels, `0` replay failures, `0` after-gate ok,
    and `0` success. sample 01 had `159/213` y/z-improving candidates but none
    reached the handoff gate. sample 03 had `0/213` y/z-improving candidates;
    every candidate worsened y/z from the first live state.

    Plain conclusion: at the first live decision state, the problem is not only
    scorer selection. The current candidate action family itself can lack a
    live-continuable action consequence to select.

11. Latest direct visual review confirms metric failure.

    sample 03 contact sheet:
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_h96_fullunion_scorer_bestgate_formalhardphase_actionbank_samples1_3_20260622_074211_alloc145920/sample_3_single_panel/live_receding_panel_contact_sheet.png`.

    Visual readout: the rollout reached DP handoff mode, but the peg was still
    not inserted into the hole at the final frame. This is not a metric-only
    false negative.

12. Latest larger live panel still fails with a valid full-episode contract.

    Output:
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_h96_fullunion_scorer_bestgate_actionbank_samples0_2_4_5_20260622_090254_alloc145920`.

    Result: `0/4` final success with `panel_full_episode_contract_ok=true`.
    Samples 0, 2, 4, and 5 all produced full observed rollouts and real final
    simulator-state failures. Contact sheets for samples 0, 2, 4, and 5 were
    opened and agree with the metrics: the robot moves near the target in some
    cases, but the peg is not finally inserted. sample 5's single-sample sheet
    is:
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_h96_fullunion_scorer_bestgate_actionbank_samples0_2_4_5_20260622_090254_alloc145920/sample_05_hole_late_continuous_insert/live_observed_rollout_annotated_contact_sheet.png`.

13. Latest all-bank all-candidate replay separates scorer failure from action
    coverage failure.

    Outputs:

    - `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_samples0_2_allbanks_allcand_sharded_20260622_0952_alloc145920`;
    - `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_samples4_5_allbanks_allcand_sharded_20260622_110056_alloc145920`.

    Combined result over samples 0, 2, 4, and 5:

    - `8733` valid live replay labels;
    - `0` direct candidate successes;
    - `212` candidates passed the after-chunk handoff geometry gate;
    - `5468` candidates improved absolute y/z, `3265` worsened it;
    - the live scorer selected `41` candidates;
    - only `1/41` selected candidates passed the geometry gate;
    - that selected gate candidate still had final success false.

    Plain conclusion: there are two problems. First, direct candidate chunks
    still do not insert by themselves. Second, gate-passing chunks do exist in
    some live states, but the current scorer almost never selects them. The
    next test is whether a gate-passing chunk followed by DP h96 can actually
    finish insertion.

14. DP96 after gate candidates confirms the gate is a false-positive
    continuability target.

    Evidence note:
    `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_gate_dp96_false_positive_blocker.md`.

    Fair DP-rollout cases:

    - `hole_late_sine` iterations 4 and 5: `160` after-gate candidates,
      `0` DP96 success, `0` DP-continuable.
    - `hole_late_continuous_insert` iteration 8: `51` after-gate candidates,
      `0` DP96 success, `0` DP-continuable.

    Combined fair result: `211/211` after-gate candidates had real DP rollout
    horizon and produced `0` success, `0` continuable states, and `0`
    contact-stable final states. `211/211` still had the peg grasped, so the
    failure is not simply dropped-grasp detection; it is missing
    contact/insertion continuability.

    The `hole_late_move_stop` iteration 10 selected gate candidate is a late
    end-of-episode case: the candidate chunk reached step `300`, so no DP96
    horizon remained. It still ended in failure.

    Plain conclusion: scorer selection is not the only blocker. The current
    geometry gate itself is too weak as a proxy for DP-continuability. These
    gate-positive states should be treated as negative labels for handoff
    learning, not as positives.

15. Successful-source contact-manifold audit explains why the gate is false.

    Output:
    `experiments/world_model_task_rebinding/cosmos3/dp_success_contact_manifold_audit_20260622/dp_success_contact_manifold_summary.json`.

    All `733` source H5 files were read with `HDF5_USE_FILE_LOCKING=FALSE`.
    This is a read-only source-slot audit, not method evidence.

    Successful-source medians:

    - within `32` frames before first insertion: `x=-0.108`, abs y/z sum
      `0.0042`;
    - within `16` frames before first insertion: `x=-0.091`, abs y/z sum
      `0.0033`;
    - within `8` frames before first insertion: `x=-0.059`, abs y/z sum
      `0.0034`;
    - inserted frames: `x≈0`.

    Current false gate candidates are mostly around `x=-0.107` to `-0.115`.
    That is closer to a broad pre-insertion alignment state than to the real
    contact/insertion handoff state. The method needs candidates and scorer
    labels that target insertion-axis/contact progress, not y/z alignment
    alone.

16. Source insertion suffix replay shows the missing action source has real
    headroom.

    Evidence note:
    `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_source_suffix_replay_headroom.md`.

    Built diagnostic bank:
    `experiments/world_model_task_rebinding/cosmos3/source_insertion_suffix_bank_20260622_alloc145920`.

    Replay output:
    `experiments/world_model_task_rebinding/cosmos3/live_snapshot_source_suffix_replay_sine_cont_iter458_20260622_alloc145920`.

    The bank mined `4711` successful insertion suffixes from all `733` H5
    sources. Its start median is around `x=-0.110`, matching the false-gate
    live region, and its end median is near inserted state.

    Source suffix replay from live snapshots tested `144` diagnostic
    candidates. Direct 32-step candidate success was still `0/144`, but
    candidate chunk plus frozen DP96 produced `18/144` success,
    `19/144` DP-continuable states, and `19/144` final contact-stable states.
    Final grasp stayed true for `144/144`.

    Plain conclusion: the current learned/live candidate family is missing a
    useful contact/insertion action source. Successful source suffixes can
    move some live states into a DP-finishable region. The next repair should
    add causal insertion-suffix/retrieval-residual candidate families and train
    the scorer against real DP96 success/continuability. Do not turn this into
    scenario-label branching or a hand-written recovery table.

17. Source-suffix live smoke confirms the scorer-selection blocker.

    Output:
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_smoke_sample5_20260622_alloc145920`.

    The source-suffix candidates were generated inside the live loop:
    `16` source-suffix candidates per iteration and `229` candidates total.
    The panel completed `3` receding iterations without process failure, but
    final success was false. It ended at relative peg-head-in-hole
    `[-0.1175, 0.0199, -0.0159]`; the contact sheet shows the peg still
    outside the hole.

    The scorer never selected a source-suffix candidate:

    - iteration 0 selected `scale_0.2`;
    - iteration 1 selected `dp_prior`;
    - iteration 2 selected `dp_prior`.

    The panel is a diagnostic smoke, not method evidence: it intentionally ran
    only `3` iterations and `151` observed frames, so the full-episode contract
    is false by design. The useful conclusion is narrower: source suffixes are
    wired into the live candidate set, but the old outcome scorer is not
    calibrated to pick them. The next concrete action is to convert live
    snapshot labels into outcome-scorer training rows and train a
    source-suffix-aware scorer using real DP96 success/continuability labels.

18. Source-suffix-aware best-gate live sample 5 succeeded.

    Output:
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_sample5_20260622_alloc145920`.

    The formal source-suffix-aware outcome scorer was trained at:
    `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_live_snapshot_source_suffix_weighted_full_20260622_alloc145920_formal3h`.
    It met the `10800` second formal floor and marked
    `checkpoint_best_gate.pt` as the formal live-eval checkpoint.

    Live result:

    - sample `05`, scenario `hole_late_continuous_insert`;
    - completed `4` receding iterations;
    - video contract passed with `301` decoded frames at `30 fps`;
    - `full_episode_length_ok=true`;
    - final simulator success true;
    - final peg-head-in-hole state
      `[0.0365917, -0.0003869, -0.0028896]`;
    - controller frame counts: `DP_SCAN_TARGET=78`, `EXECUTOR_ACTIVE=5`,
      `DP_HANDOFF=217`, `INIT_OBS=1`.

    The important causal fact is that the scorer selected a source-suffix
    candidate in the first live decision:
    `retrieval_resid_srcsuffix_r1_s1_o32`. The old scorer did not select
    source-suffix candidates and failed this same direction. Here the
    source-suffix chunk moved the live state into a region from which DP96
    could finish insertion.

    Visual review: the panel contact sheet was opened. It shows the peg
    outside the moved hole initially, then aligned and inserted by the final
    frame. The generated summary still says
    `visual_review_status=needs_direct_agent_or_user_review`, so this success
    is recorded in docs as agent-reviewed evidence rather than silently
    overriding the generated JSON.

    This is not yet a broad method result. It is one-sample conversion
    evidence. The next active check is the same source-suffix-aware best-gate
    controller on the previous failed set `0,2,4,5`, preserving the same
    full-episode contract.

19. Formal rank-loss source-suffix scorer failed.

    Output:
    `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_live_snapshot_source_suffix_rank1_fullbatch_20260622_alloc145920_formal3h`.

    The run met the formal floor (`elapsed_seconds=10800.061`,
    `steps=55962`, `formal_training_floor_met=true`) but did not pass the
    live-eval gate. Final validation selected handoff was `0/8`, DP was
    `0/8`, oracle was `1/8`, and no `checkpoint_best_gate.pt` was produced.

    Plain conclusion: rank loss learned the train positives but did not
    generalize to the held-out positive. Do not run live eval from this
    scorer.

20. sample00 source-suffix offset audit narrowed the coverage miss.

    The corrected sample00 iter0 live bank used source suffix offsets `32,24`
    and had `source_suffix_candidate_count=0`. Read-only suffix-bank audit
    shows same-scenario offsets `32,24` had nearest distance `0.02073`, just
    outside the active `0.02` cap. The same scenario with all offsets had
    nearest distance `0.00652`, and offsets `48,64` enter the cap.

    Plain conclusion: this is an action-candidate coverage/offset issue, not
    evidence for loosening the distance threshold. The next compute-node
    diagnostic is a one-iteration sample00 run with offsets `64,48,32,24`.
    The first attempt did not run because allocation `145920` was revoked and
    step `145920.316` was cancelled after `4` seconds.

21. sample00 offsets `64,48,32,24` produced one full live success, but panel
    rerun failed.

    Standalone successful root:
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_sample00_full_vkfix_20260622_alloc146658`.

    Resource/render note: allocation `146639` on server27 failed twice at the
    first render frame with Vulkan `DeviceLost`. The rerun used allocation
    `146658` on server56 with
    `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`.

    Standalone result:

    - `completed_iterations=8`;
    - `final_observed_frames=301`;
    - `full_episode_length_ok=true`;
    - `video_file_contract_ok=true`;
    - final simulator `success=true`;
    - final peg-head-at-hole
      `[-0.007180, 0.000003, 0.001853]`;
    - controller frames: `DP_SCAN_TARGET=106`, `EXECUTOR_ACTIVE=44`,
      `DP_HANDOFF=150`, `INIT_OBS=1`.

    Causal chain: iterations `0-5` selected source-suffix candidates with
    offsets `48,48,32,32,24,24`. After iteration `5`, real state reached
    `[-0.131838, 0.007562, -0.000056]` and crossed C_pi. Frozen DP handoff
    then completed insertion. The contact sheet was opened and agrees with
    the final metric: the peg reaches the hole/box region and remains held.

    Panel rerun evidence:

    `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_panel0245_offsets64_sample00_resample_failure.md`

    Panel sample00 result:

    - full `301/300` contract passed;
    - final simulator `success=false`;
    - final peg-head-at-hole `[-0.095756, 0.016172, -0.065285]`;
    - visual sheet opened and agrees with failure.

    Plain meaning: sample00's old iter0 failure was partly an offset/action
    coverage miss, not a reason to loosen thresholds. But offsets `48,64` are
    not a stable fix by themselves. The current blocker is the closed-loop
    interface after source-suffix chunks: real C_pi/DP handoff/contact
    continuability can fail even when early candidate coverage exists.

22. sample00 replay also shows current C_pi is not a sufficient handoff
    target.

    One-iteration diagnostic root:
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_sample00_oneiter_vkfix_20260622_alloc146658`.

    The new offsets made source-suffix candidates enter sample00 iter0:
    `candidate_count=215`, `source_suffix_candidate_count=2`. The scorer
    selected `retrieval_resid_srcsuffix_r0_s1_o48`.

    Replay root:
    `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_sample00_offsets48_source_suffix_candidates_dp96_20260622_221837_alloc146658`.

    Replay result over the two source-suffix candidates:
    `after_gate_ok_count=0`, `improved_abs_yz_sum_count=0`,
    `worsened_abs_yz_sum_count=2`, but `dp_rollout_success_count=1/2`.
    The live-selected candidate was the positive one: after its 8-step chunk
    C_pi was false and y/z was worse, but DP96 succeeded in `48` steps with
    final peg-head-at-hole `[-0.003523, 0.002995, 0.002409]`.

    Plain conclusion: do not treat instantaneous C_pi as the only learning
    target. Handoff scoring should be trained from real DP-rollout
    continuability labels.

## Current Audit Result

Completed audit:

`experiments/world_model_task_rebinding/cosmos3/h96_selector_blocker_audit_20260621_233801_alloc145813`

Important facts:

- all-candidate handoff oracle: `60/64`;
- live-family-only handoff oracle: `58/64`;
- DP handoff baseline: `35/64`;
- live-family oracle final-success count: `4/64`;
- strict safe live-family handoff candidate exists for `46/64` groups;
- strict safe non-DP candidate exists for only `16/64` groups;
- among DP-failure groups, strict safe non-DP candidates exist for only
  `11/29` groups;
- no `source_uuid` train/val overlap;
- scenario and phase labels overlap, so validation is small and not a strong
  generalization proof.

Interpretation:

The current h96 data has broad live-family handoff headroom, but safe non-DP
positive examples are sparse. This explains the observed failure pattern:
training can find non-DP candidates, but validation often chooses candidates
that hurt geometry/contact. A live-family safety selector is the right test,
but if it keeps overfitting, the next fix is more balanced safe-non-DP labels,
not another scalar gate.

## Remaining Hidden Problems

1. DP handoff labels may not match live DP handoff.

   Restored replay state DP success does not guarantee live closed-loop DP
   success after compounding execution error. The latest live panel already
   showed DP handoff execution without final success.

2. Scorer descriptors may be too abstract.

   Live candidates may mostly appear as broad checkpoint-model descriptors,
   while offline labels include richer family information. The scorer may
   learn offline family priors that do not transfer to live.

3. The latest live panel is small but already rejects success.

   The 2026-06-22 panel is only four samples, so it is not a broad trend
   estimate. But it is enough to reject the current checkpoint/controller as a
   successful method because final success was `0/4` with full-length rollouts
   and visual evidence.

4. Simulator state use must stay bounded.

   Simulator state may be used for training labels, causal metadata, and
   diagnostics. Main controller evidence must still use RGB/RGB-D-derived
   state or slots as controller input. Do not package oracle/state-only
   results as the publishable method.

## Immediate Decision Tree

1. Formal scorer summaries are complete.

   - rank0 gatefix:
     `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_gatefix_145813_20260621_2018_formal3h`
   - rank0.2 gatefix:
     `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_rank02_gatefix_145814_20260621_2032_formal3h`

   Both meet the 10800-second floor, but both are unsafe for live evaluation:
   the best gates slightly improve held-out handoff count while worsening
   weighted geometry error and contact progress. Final rank0.2 also drops
   below DP.

2. Do not run live eval from either current scorer.

   A live panel here would mostly test a known-bad selector. It would not
   answer the next scientific question.

3. Current aligned experiment:

   Train/evaluate a live-family-only safety selector:

   - train and evaluate only on candidate families the live loop can generate,
     plus DP prior;
   - rank candidates against the group DP prior and the live-family handoff
     oracle;
   - gate checkpoints only when held-out handoff improves without worsening
     weighted geometry error or contact progress;
   - keep the same 3-hour formal training floor for method evidence.

   Running formal attempts:

   - default-capacity live-family safety selector on allocation `145814`:
     `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_scorer_20260621_234650_alloc145814_formal3h`
   - low-capacity regularized live-family safety selector on allocation
     `145818`:
     `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_lowcap_20260621_234950_alloc145818_formal3h`

   Latest read-only check at 2026-06-22 00:17+08 is still negative:
   `145814` validation selected `6/16` handoff successes versus DP `8/16`,
   and `145818` selected `7/16` versus DP `8/16`; both have worse weighted
   geometry error and worse contact progress. These are not live-eval
   checkpoints.

   Later status: the `145814` live-family safety selector step was cancelled
   after about `40` minutes, so that root is partial-only. The low-capacity
   `145818` run was also cancelled before the 10800-second formal floor and
   remained unsafe at its latest point. Neither live-family safety selector is
   usable for live eval.

4. If both live-family safety selectors fail their formal summaries:

   Treat h96 selector data balance as the confirmed blocker. Expand or
   rebalance labels around DP-failure states where live candidates can safely
   improve handoff. The target positive is not "any recovery case"; it is
   "same causal task-frame rebinding interface, DP would not finish, a
   live-generatable short chunk moves the real state to a DP-continuable
   geometry without hurting contact/progress."

   Replacement allocation `145919` started on `server24` at 2026-06-22
   00:13+08 and should be used for this live-family h96 label expansion.
   Allocation `145813` was lost during step cleanup and its partial label
   claim has no usable completed union; do not train from that partial.

   Allocation `145815` later started with four GPUs and is now being used to
   parallelize the same live-family h96 label expansion. It shares the
   `145919` claim root, skips already claimed shards, and has claimed
   `skip72`, `skip80`, `skip88`, and `skip96` while `145919` owns `skip64`.
   `skip64` completed first and `145919` then claimed `skip104`.

   Final status: all eight new live-family shards `skip64,72,80,88,96,104,
   112,120` completed. Final union over `64` groups has DP handoff `40/64`
   and live-family handoff oracle `57/64`; meaningful improvement is `44/64`.
   Final-success oracle is only `5/64` versus DP `4/64`, so this is useful
   handoff/progress headroom, not task success.

   Completed formal scorer results from this union:

   - default capacity on allocation `145920`:
     `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_fullunion_20260622_013500_fullunion_alloc145920_formal3h`
   - low-capacity regularized comparison on allocation `145815`:
     `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_fullunion_lowcap_20260622_014000_fullunion_lowcap_alloc145815_formal3h`

   The default-capacity run on `145920` completed the `10800` second formal
   floor. Its `checkpoint_best_gate.pt` is the only usable checkpoint:
   held-out selected handoff is `10/16` versus DP `9/16`, weighted geometry is
   better by `-0.0326008`, contact progress is better by `+0.0146695`, and
   non-DP selection is `0.4375`. The final/latest checkpoint is unsafe:
   `6/16` selected handoff versus DP `9/16`, with worse geometry and progress.
   Do not use `checkpoint_final.pt` or `checkpoint_latest.pt` for live eval.

   The conservative margin eval also passed only at margin `0.0`:
   `ready_for_conservative_offline_gate=true`, one gate-passing margin, with
   `6` improved switches and `1` harmful switch on `16` held-out groups. This
   is enough to justify a small live panel from `checkpoint_best_gate.pt`, but
   it is still weak evidence because the validation set is small and no
   positive score margin survives.

   The low-capacity `145815` comparison was cancelled before the formal floor
   and wrote no valid `training_summary.json` or `checkpoint_best_gate.pt`.
   Treat it as partial-only.

   New caution: the h96 live-family labels were generated with a hard-phase
   candidate-generator line whose original default was the short
   `outcome_oracle_candidate_executor_hardphase_balanced_smoke100` checkpoint.
   That smoke checkpoint is useful for diagnostic label expansion, but it is
   not formal method evidence. For live panels and future formal claims, prefer
   a formal 3-hour Gaussian hard-phase executor or regenerate the labels and
   scorer from the exact formal candidate generator used live.

5. Latest live panel result.

   The small formal-gated live panel was run on allocation `145920`:

   `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_h96_fullunion_scorer_bestgate_formalhardphase_panel4_20260622_0458_alloc145920_samples0_1_3_4`

   Facts:

   - final success `0/4`;
   - failed process count `0`;
   - panel full-episode contract passed;
   - every sample has `301` observed frames and `full_episode_length_ok=true`;
   - contact sheet exists at `live_receding_panel_contact_sheet.png` and was
     visually inspected;
   - this is not method evidence of success.

   Per-sample result:

   - sample 00 final `[-0.102740, -0.015748, -0.016932]`, no handoff, y/z
     still out of range;
   - sample 01 final `[-0.091294, -0.027579, -0.063382]`, no handoff, z badly
     out of range;
   - sample 03 final `[-0.050987, 0.001507, 0.003059]`, DP handoff for `59`
     steps, geometry close, but final insertion still false;
   - sample 04 final `[-0.102361, -0.020728, -0.028326]`, no handoff, y/z out
     of range.

   Interpretation: the method currently fails in two ways. Some samples never
   reach a reliable handoff state. One sample reaches a near-hole handoff
   state but still cannot complete insertion. This points to live consequence
   calibration plus contact/insertion execution, not more threshold tuning.

6. Latest live-real outcome extraction.

   Extracted selected live chunk outcomes:

   `experiments/world_model_task_rebinding/cosmos3/live_receding_selected_outcomes_h96_panel4_20260622_0729/live_selected_outcomes_summary.json`

   Facts:

   - `45` total executed records;
   - `37` candidate-executor records and `8` DP-handoff records;
   - final success samples `0/4`;
   - full episode length ok samples `4/4`;
   - selected candidate names are dominated by `dp_prior`, `short8_mean`,
     and `mean`;
   - `17` selected candidate chunks worsened absolute y/z error;
   - scorer predicted-state error on candidate chunks: y MAE `0.0370`,
     y RMSE `0.0522`, z MAE `0.0240`, z RMSE `0.0325`;
   - executor predicted-next-state error: x MAE `0.0686`, y MAE `0.0288`,
     z MAE `0.0198`;
   - gate failures are dominated by `rel_y_abs` and `rel_z_abs`.

   Plain conclusion: the offline scorer is not calibrated to the real live
   action consequence. It often thinks a selected chunk will be continuable,
   but after execution the real y/z contact geometry is still bad or worse.
   This is the concrete blocker to fix before another broad live panel.

7. Latest live outcome calibration audit.

   Audit output:

   `experiments/world_model_task_rebinding/cosmos3/live_outcome_calibration_h96_panel4_20260622_0745/live_outcome_calibration_summary.json`

   Facts:

   - audited `37` selected candidate-executor chunks from `4` samples;
   - selected chunks that worsened absolute y/z: `17/37`;
   - high handoff-confidence chunks, threshold `0.95`: `26`;
   - high-confidence chunks that worsened y/z: `13/26`;
   - high-confidence chunks that still failed the live gate: `21/26`;
   - scorer after-state abs-yz MAE: `0.03836`;
   - executor after-state abs-yz MAE: `0.03886`;
   - identity/no-motion after-state abs-yz MAE: `0.01828`.

   Plain conclusion: for these live states, the scorer/executor predicted
   next-state geometry is worse than assuming the peg-hole relation barely
   changes. That means the model is over-trusting imagined action effects.
   The next fix must collect and learn real live-state action consequences.

8. Required implementation change before broad reruns:

   Future candidate-executor live panels should enable candidate action-bank
   saving. This writes every generated candidate action/residual array for
   each live receding iteration to `candidate_action_bank.npz`. Without that
   sidecar, only the selected chunk can be calibrated after the fact, and
   unselected candidates cannot be replay-labeled from the same live state.

   Added controls:

   - `--save-candidate-action-bank` in
     `scripts/world_model/run_cosmos3_live_receding_loop.py`;
   - panel pass-through in
     `scripts/world_model/run_cosmos3_live_receding_panel.py`;
   - `SAVE_CANDIDATE_ACTION_BANK=true` in the Slurm live-panel wrappers.

9. Next aligned action:

   Build a live-real outcome calibration step from actual executed chunks:
   for every live receding iteration, pair the selected candidate, its scorer
   predictions, the real before state, and the real after state. Use that to
   audit and retrain/calibrate the scorer toward actual live outcomes:
   y/z displacement, grasp retention, post-chunk continuability, DP handoff
   result, and final insertion progress.

   This is still the Dream Diffusion Policy-style direction: short chunks,
   real re-observation, and a world-model/scorer loop that chooses actions by
   predicted task progress. It is not error-case enumeration. The fix is to
   make the learned scorer/executor match real contact consequences instead of
   hand-writing recoveries for each failure.

   Implementation update on 2026-06-22:

   - Added `scripts/world_model/replay_cosmos3_live_action_bank_from_snapshots.py`.
     It replays saved `candidate_action_bank.npz` candidates from
     `live_state_before_controller.h5` snapshots and records real live
     action-consequence labels.
   - A compute-node smoke test wrote
     `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_smoke_20260622_0805_alloc145920`.
     It replayed `dp_prior`, `mean`, and `scale_0.5` from one live state:
     all `3/3` worsened y/z, `0/3` passed the after-chunk gate, and `0/3`
     succeeded. This is calibration evidence, not method success.
   - A broader current-panel replay label job is running in tmux
     `cosmos3_live_actionbank_replay_labels_145920_20260622_0812`, output
     root
     `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_current_panel_20260622_0812_alloc145920`.
     It completed with `77` valid labels and `0` failures: `50/77`
     improved y/z, `27/77` worsened y/z, `11/77` passed the after-chunk
     continuability gate, and `0/77` succeeded. Among the live scorer's
     selected candidates, `7/11` improved y/z, `4/11` worsened y/z, only
     `2/11` passed the after-chunk gate, and `0/11` succeeded.

   Plain meaning: the next dataset should be real live consequence labels
   from the same snapshot, not more copies of the closed-loop panel. The
   useful signal is that many candidates can improve local y/z, but the
   current scorer/executor still does not reliably select or execute chunks
   that become insertion-completing states.

   Extra all-candidate check on 2026-06-22:

   - Output root:
     `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_allcand_iter0_20260622_0820_alloc145920`.
   - Replayed all `213` candidates from the first saved bank of sample 01 and
     all `213` candidates from the first saved bank of sample 03.
   - Total: `426` labels, `0` failures, `0` after-gate ok, `0` success.
   - sample 01: `159/213` improved y/z but none reached the handoff gate.
   - sample 03: `0/213` improved y/z; every candidate worsened y/z from the
     first live state.

   Plain meaning: for the first live decision state, the problem is not just
   that the scorer missed a hidden good candidate. The current generated
   candidate family itself often lacks a live action consequence that reaches
   a DP-continuable or insertion-success state.

   Larger all-bank replay on 2026-06-22:

   - Output roots:
     `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_samples0_2_allbanks_allcand_sharded_20260622_0952_alloc145920`
     and
     `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_samples4_5_allbanks_allcand_sharded_20260622_110056_alloc145920`.
   - Combined: `8733` valid labels, `0` direct candidate success, `212`
     after-gate ok candidates, `5468` y/z-improving candidates, and `3265`
     y/z-worsening candidates.
   - Gate-passing candidates were concentrated in `hole_late_sine` iterations
     4 and 5, `hole_late_continuous_insert` iteration 8, plus one
     `hole_late_move_stop` iteration 10 candidate.
   - The live scorer selected only `1` gate-passing candidate out of `41`
     selected records, and even that selected gate candidate did not finish.

   Completed follow-up:

   `experiments/world_model_task_rebinding/cosmos3/live_snapshot_gate_iters_dp96_hole_late_sine_iter4_5_gateonly_20260622_1145_alloc145920`.

   This replayed only the `160` candidates from gate-rich `hole_late_sine`
   iterations 4 and 5 that already passed the geometry handoff gate in the
   no-DP replay, then runs DP h96 handoff. The candidate filter file is:
   `experiments/world_model_task_rebinding/cosmos3/sine_iter45_gate_candidate_indices_20260622.tsv`.
   Result: `0/160` DP96 success and `0/160` DP-continuable.

   Follow-up on `hole_late_continuous_insert` iteration 8 adds `51` more fair
   DP96 cases, also `0/51` success and `0/51` DP-continuable. The selected
   `hole_late_move_stop` gate candidate occurred too late for DP rollout
   horizon and ended in failure.

   This answers the causal question: geometry-near-hole is still not enough
   for contact/insertion.

   Implementation update: `scripts/world_model/replay_cosmos3_live_action_bank_from_snapshots.py`
   now accepts `--candidate-filter-tsv`, whose first two columns are
   `iteration` and `candidate_index`. It was then hardened to also support
   `scenario iteration candidate_index`, because a combined multi-scenario
   replay showed that iteration-only filters can pull unintended candidates
   from another scenario. `py_compile` passed inside allocation `145920`.

10. Do not repeat the same live panel as the next main experiment.

   Running more samples with the same scorer/controller may provide a larger
   failure histogram, but it will not fix the confirmed blocker. The next
   useful compute should either collect live-real outcome labels for selected
   chunks or train/evaluate a scorer/executor that explicitly predicts live
   grasp/contact/insertion consequences.

11. Do not return to scalar threshold tuning as the main method.

12. Latest source-suffix safety and coverage result, 2026-06-22.

   The first source-suffix broad panel showed a real safety bug: the selector
   could choose far or cross-scenario suffixes that only looked good to the
   scorer. The active boundary now requires same-scenario suffix matching,
   source-suffix start distance at most `0.02`, 8-step execution before
   re-observation, and blocks pure `dp_prior` execution whenever the live
   continuability gate `C_pi` is false.

   After those fixes, sample 00 is a clear candidate-coverage blocker:

   - corrected live panel root:
     `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_panel0245_cap_scenario_k2_dist002_exec8_nodpunsafe_20260622_alloc145920`;
   - replay root:
     `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_sample00_iter0_all_candidates_dist002_exec8_nodpunsafe_20260622_alloc145920`;
   - `213` valid candidates replayed from the exact live snapshot;
   - `0` direct success, `0` after-gate states, `0` y/z-improving candidates;
   - `213/213` candidates worsened y/z.

   Plain conclusion: for this early live state, the failure is no longer only
   selector calibration. The current generated/retrieved action bank does not
   contain a good action to select. The DDP/HDP lesson therefore points to
   training or generating a stronger executor/action family from dense
   live-state and successful source/contact chunks, then selecting with real
   DP-continuability labels.

13. Strict source-suffix positive control on sample 05, 2026-06-22.

   Corrected run root:

   `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_sample5_dist002_exec8_nodpunsafe_20260622_alloc145920`.

   Boundary:

   - same-scenario source suffix only;
   - source-suffix start distance at most `0.02`;
   - execute only `8` steps before re-observation;
   - pure `dp_prior` is not executable while live `C_pi` is false;
   - full `301` RGB frames / `300` actions.

   Result:

   - trigger frame `78`;
   - iterations 0-5 all selected
     `retrieval_resid_srcsuffix_r0_s0p5_o32`;
   - after iteration 5 the real state reached
     `[-0.112408, 0.009808, -0.003034]` and `C_pi=true`;
   - frozen DP handoff ran from frame `123`;
   - final real peg-head-at-hole was
     `[0.030646, -0.003007, -0.002958]`;
   - final `success=true`;
   - generated and inspected
     `live_observed_rollout_annotated_review_sheet.png` plus frame `300`.

   Plain meaning: the corrected source-suffix/DP-handoff interface is not
   broken. It can convert at least one dynamic sample when close insertion
   suffix chunks exist and are selected repeatedly. This is still not broad
   method success because sample 00 under the same boundary has no usable
   candidate action in its first live state.

   Immediate diagnostic: replay all saved candidate banks from this strict
   sample 05 run. If many unselected candidates would also cross `C_pi`, the
   next bottleneck is selection/generalization. If only the selected suffix
   chain works, the bottleneck is still action-candidate coverage.

14. Strict sample 05 all-candidate replay result, 2026-06-22.

   Replay root:

   `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_sample05_strict_all_candidates_dist002_exec8_nodpunsafe_20260622_alloc145920`.

   Result:

   - `1300` valid candidate replays from the six saved action banks;
   - `0` direct 8-step insertion successes;
   - `214` after-chunk `C_pi` states;
   - `651` candidates improved y/z, `649` worsened y/z;
   - iteration 2 alone had `213/215` after-gate candidates;
   - the live scorer's six selected source-suffix chunks had `0/6`
     one-step after-gate outcomes in this replay.

   Plain meaning: sample 05 is not an action-coverage miss like sample 00.
   Its saved banks contain many short chunks that can reach the handoff gate.
   For sample 05, the bottleneck is selection/continuability scoring and
   handoff validation; for sample 00, the bottleneck remains missing actions.
   The active follow-up is DP96 replay on the `214` sample 05 after-gate
   candidates, because previous gate-only candidates from other samples were
   often false positives.

15. Strict sample 05 after-gate + DP96 replay, 2026-06-22.

   Filter:

   `experiments/world_model_task_rebinding/cosmos3/sample05_strict_after_gate_candidate_filter_20260622.tsv`

   Replay root:

   `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_sample05_strict_aftergate_dp96_20260622_alloc145920`.

   Result:

   - `214/214` filtered candidates were valid and after-gate;
   - `214/214` improved y/z;
   - `118/214` reached final DP96 success;
   - `208/214` remained DP-continuable;
   - `209/214` ended contact-stable;
   - the positives came from iteration `2`; the single iteration `4` gate
     candidate did not survive DP96.

   Plain meaning: sample 05 has real selectable short-chunk headroom. The
   selector missed a large set of actions that would have enabled an earlier
   DP handoff. Therefore sample 05 points to scorer/selector training. Sample
   00 still points to action-generation coverage. The next method repair must
   handle both: better selection where candidates exist, and stronger
   executor/action generation where they do not.

16. Short scorer sanity on mixed sample00/sample05 labels, 2026-06-22.

   Mixed label root:

   `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_sample00neg_sample05dp96_mix_20260622_alloc145920`.

   Data:

   - sample00 iter0 corrected all-candidate negatives;
   - sample05 non-gate candidates from all-candidate replay;
   - sample05 after-gate candidates with DP96 labels.

   Conversion produced `1513` valid candidate rows across `7` live states.
   It had `118` DP96-success candidate rows, but all of those positives came
   from one state: sample05 iteration 2, prefix frame `94`. In that same state,
   `dp_prior` was also DP96-success.

   Two short debug trainings were run inside allocation `145920`:

   - `scorer_overfit100_retry_boolfix` fixed the argparse flag and completed
     `100` steps, but used `rank_loss=0`; it learned target values but did not
     select the handoff-positive candidate.
   - `scorer_rank_overfit100_val0` used `rank_loss=1`, full-group batches, and
     no validation split. It drove rank CE to about `0.00069` and selected the
     one handoff-positive candidate. It still had selected DP96 success
     `1/7`, equal to DP prior `1/7`, so no offline gate/checkpoint was valid.

   Plain meaning: candidate ranking must include an explicit within-state
   rank loss. Plain regression is not enough. But the current tiny mixed label
   set is not adequate formal evidence because the only success-positive state
   is already DP-finishable by the DP prior. The next useful scorer work is a
   broader real replay label set with DP-failure states that contain
   non-DP-success candidates, plus the rank-loss objective. The sample00 fix
   still requires action-candidate/executor coverage, not just a better
   scorer.

17. Running formal rank-loss scorer on broader source-suffix labels.

   Allocation/step:

   `145920.315`

   Output root:

   `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_live_snapshot_source_suffix_rank1_fullbatch_20260622_alloc145920_formal3h`.

   Input root:

   `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_source_suffix_union_20260622_alloc145920`.

   Input data:

   - `41` live states;
   - `8877` candidate rows;
   - `3` states where DP prior fails but a non-DP candidate has DP96 success;
   - source-suffix candidates included, but still only as causal short chunks
     scored by real replay labels.

   Training change from the earlier formal run:

   - `rank_loss=1.0` instead of `0.1`;
   - `batch_size=8192` so rank batches contain full groups;
   - same `10800` second formal floor;
   - same held allocation and compute-node discipline.

   First eval at step 1 wrote successfully. This run tests whether the
   selector can use the broader DP-failure/non-DP-success replay labels. It
   cannot fix sample00 by itself, because sample00's corrected first bank has
   no useful candidate action.

   Early split audit while the run is active:

   - validation has `8` states;
   - its only DP96-success state is sample05 iteration `5`, frame `198`;
   - training positives include sample05 iteration `4`, frame `174`, from the
     same source/scenario, plus sample04 iteration `5`;
   - step `1000` and step `3000` selected `0/8` validation handoff successes
     while training stayed at `2/33`.
   - at about `46` minutes, step `13000` still selected `0/8` validation
     handoff successes while training stayed at `2/33`; validation oracle
     remains `1/8`, and selected weighted error/contact progress are still
     worse than DP.
   - at about `76` minutes, step `22000` had train rank CE near `0`, but the
     validation result was still selected handoff `0/8` versus oracle `1/8`.
   - final summary met the formal floor but had
     `ready_for_formal_live_eval=false`, `best_gate_metrics=null`, validation
     selected handoff `0/8`, DP `0/8`, oracle `1/8`.

   Plain meaning: the early failure is not caused by validation positives
   being completely unseen by source/scenario. The scorer is still not
   generalizing across nearby live states.

   Evidence note:

   `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_rank1_formal_selector_failure.md`

## Evidence Boundaries

Do not claim method success without:

- formal scorer summary;
- margin/offline gate evidence;
- full `301/300` live rollout;
- final real simulator state;
- structured video/contact-sheet review;
- DP baseline comparison.

Current concise evidence note:

`docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-21_selector_generalization_blockers.md`
