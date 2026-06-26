# Low-Frequency WM + Executor TODO

## Current Boundary

- [x] 2026-06-23 01:40+08 root Instruction refreshed with the latest user
      operating rule: top-level judgments stay under `Instruction/`, current
      tmux/Slurm/artifact state must be observed continuously, and routine
      updates should report facts, blockers, evidence status, and next action
      without pause/ending language.
- [ ] 2026-06-23 01:45+08 active formal scorer status: the first
      union+panel formal split was invalid as a live-eval gate, not merely
      inconclusive. Its validation split had DP handoff `4/21` and handoff
      oracle `5/21`; the maximum possible handoff improvement was therefore
      `1/21=0.0476`, below the strict `>0.05` gate. That run and its h8192
      capacity duplicate were interrupted inside the held allocation after this
      gate/split bug was confirmed. The corrected deterministic split is seed
      `20260725`, chosen only by oracle headroom and scenario balance: val DP
      `3/21`, val oracle `7/21`, val headroom `+4`, train headroom `+3`,
      scenario counts `5/5/6/5`. Current running roots on allocation `146658`:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_source_suffix_union_plus_panel0245_handoff_rank_seed20260725_h2048_formal3h_20260623_alloc146658`
      and
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_source_suffix_union_plus_panel0245_handoff_rank_seed20260725_h8192_formal3h_20260623_alloc146658`.
      Because h2048/h8192 left GPU utilization unstable, a third same-data,
      same-split h16384 positive-weight formal run was launched:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_source_suffix_union_plus_panel0245_handoff_rank_seed20260725_h16384_posw_formal3h_20260623_alloc146658`.
      All three runs completed the 3-hour floor and report
      `ready_for_formal_live_eval=true`. h2048 is the preferred live-panel
      checkpoint: best step `15000`, validation selected handoff `6/21`, DP
      `3/21`, oracle `7/21`, handoff delta `0.143`, weighted-error delta
      `-0.014`, progress delta `+0.0045`. h8192 reached `5/21`; h16384
      positive-weight also reached `6/21` but has a slightly negative best
      progress delta. Next action is a small live closed-loop panel from h2048
      `checkpoint_best_gate.pt`; offline scorer success alone is not method
      evidence.
- [x] 2026-06-23 06:10+08 h2048 best-gate live panel completed:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h2048_scorer_source_suffix_offsets64_48_32_24_panel0245_20260623_alloc146658`.
      Result: `2/4`, full contract ok, no failed process, visual contact sheet
      inspected. Successes were sample02 reverse and sample04 sine; failures
      were sample00 move_stop and sample05 continuous_insert. This is not a
      net improvement over the previous `2/4` panel: h2048 rescued sample02
      but lost sample05. The immediate aligned follow-up is the same live panel
      with the h8192 ready checkpoint, now running on held allocation `146658`,
      to check whether the regression is h2048-specific scorer selection or
      broader selector instability.
- [x] 2026-06-23 07:56+08 h8192 best-gate live panel completed:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_scorer_source_suffix_offsets64_48_32_24_panel0245_20260623_alloc146658`.
      Result: `2/4`, full contract ok, no failed process, visual contact sheet
      inspected. Successes were sample00 move_stop and sample05
      continuous_insert; failures were sample02 reverse and sample04 sine.
      Plain conclusion: this does not beat h2048, but it proves the scorer
      failure is complementary rather than a single fixed bad sample. h2048
      rescues 02/04; h8192 rescues 00/05. The issue is scorer selection
      stability over real DP-rollout continuability labels, not the lack of
      any useful candidate. Next aligned compute action is the same panel with
      the already formal-qualified h16384 positive-weight scorer.
- [x] 2026-06-23 current next action: launch h16384 positive-weight live panel
      on held allocation `146658`, same samples `0,2,4,5`, same source-suffix
      offsets `64,48,32,24`, same DP96 handoff and full `301/300` contract.
      Do not interpret it as method evidence unless final state and visual
      review agree. If it remains `2/4` with a different success set, repair
      should move to calibrated/ensemble scorer or better true DP-rollout
      labels, not hand-coded per-sample rules.
- [x] 2026-06-23 09:27+08 h16384 positive-weight live panel completed:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h16384_posw_scorer_source_suffix_offsets64_48_32_24_panel0245_20260623_alloc146658`.
      Result: `2/4`, full contract ok, no failed process, visual contact sheet
      inspected. Successes were sample00 move_stop and sample05
      continuous_insert; failures were sample02 reverse and sample04 sine.
      This matches h8192 and fails to merge h2048's sample02/sample04
      successes. Plain conclusion: running more single-capacity scorers is
      now low-value. The real blocker is unstable candidate/handoff scoring:
      failed samples often align y/z but do not push x into insertion, and
      some DP handoff chunks make the live state worse.
- [ ] 2026-06-23 next repair direction: build an offline scorer-comparison
      table across h2048, h8192, and h16384_posw on the saved live candidate
      banks and real DP96 labels. The question is whether a calibrated
      ensemble/risk rule can select the union of 00/02/04/05 successes without
      per-sample branches. If offline labels do not support that, generate
      broader true DP-rollout labels for the ambiguous live states rather than
      adding hand-written recovery cases.
- [x] 2026-06-23 09:35+08 offline scorer comparison and simple ensemble
      diagnostic completed on allocation `146658`. Outputs:
      `experiments/world_model_task_rebinding/cosmos3/three_scorer_margin_eval_union_plus_panel0245_seed20260725_20260623_alloc146658`
      and
      `experiments/world_model_task_rebinding/cosmos3/three_scorer_ensemble_compare_union_plus_panel0245_seed20260725_20260623_alloc146658`.
      Same validation split: 21 groups, DP handoff `3/21`, handoff oracle
      `7/21`. Single h2048 and h16384_posw reach `6/21`; h8192 reaches
      `5/21`. Simple `mean_raw_score` and `mean_delta_vs_dp` reach only
      `5/21`; `max_delta_vs_dp` reaches `6/21`, not above the best single
      model and still below oracle. Plain conclusion: do not launch a live
      ensemble from this result. The next useful repair is more true DP96
      labels and/or scorer features for ambiguous states, especially the
      cases where y/z aligns but x insertion progress and DP handoff quality
      are mis-scored.
- [ ] 2026-06-23 next compute action: identify saved live candidate banks for
      failed/ambiguous 02 and 04 states from h2048/h8192/h16384 panels, then
      replay a broader subset with DP96 labels under the held allocation. The
      purpose is not recovery-case enumeration; it is to give the scorer
      enough real labels to distinguish "looks aligned in y/z" from "actually
      pushed far enough in x for frozen DP to finish."
- [x] 2026-06-23 10:04+08 targeted 02/04 DP96 replay completed on held
      allocation `146658`. Output:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_three_scorer_02_04_sourcesuffix_dp96_20260623_alloc146658`.
      Summary: h2048 `39` valid rows with `9` DP96 successes; h8192 `161`
      valid rows with `32` DP96 successes; h16384_posw `133` valid rows with
      `9` DP96 successes. Plain conclusion: the failed 02/04 live panels still
      contain DP-finishable candidates in their saved banks, so the next repair
      is scorer/feature/label calibration, not enumerated recovery cases.
- [x] 2026-06-23 10:10+08 conversion bug found and fixed before training.
      The first conversion output
      `live_snapshot_outcome_scorer_training_three_scorer_02_04_sourcesuffix_dp96_20260623_alloc146658`
      collapsed different h2048/h8192/h16384 live states with the same
      sample/iteration/prefix into one uuid. That output must not be used for
      scorer training. The converter now namespaces uuid by the source replay
      panel. Fixed output:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_three_scorer_02_04_sourcesuffix_dp96_fixuuid1_20260623_alloc146658`.
      It has `44` groups, `333` valid rows, `50` DP96 successes, `10`
      source-suffix DP96-success groups, `bad_uuid_multi_jsonl=0`, and no
      duplicate `(uuid,candidate_name)` pairs.
- [x] 2026-06-23 10:13+08 merged fixed 02/04 labels into the previous
      union+panel0245 scorer set. Merged output:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_source_suffix_union_plus_panel0245_plus0204fixuuid_dp96_20260623_alloc146658`.
      It has `127` live-state groups and `9349` candidate rows. DP prior
      handoff succeeds on `25/127` groups; handoff oracle succeeds on `44/127`;
      source-suffix handoff succeeds on `32/127`. This is real training
      headroom, not live method evidence.
- [x] 2026-06-23 10:15+08 h2048 short sanity completed:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h2048_rank_sanity100_seed20260725_20260623_alloc146658`.
      This is debug-only. It loaded the fixed merged data and learned train
      ordering, but did not beat DP on held-out groups at 100 steps: train
      selected handoff `32/95` versus DP `17/95`; validation selected handoff
      `8/32`, DP `8/32`, oracle `12/32`.
- [x] 2026-06-23 10:16+08 h8192 formal scorer was launched on allocation
      `146658`, step `146658.135`, tmux window `train0204_h8192_formal`.
      Output:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h8192_formal3h_seed20260725_20260623_alloc146658`.
      It was interrupted after about `4` minutes because GPU utilization was
      only about `22%`, below the cluster release-risk threshold. This is not
      formal evidence and must not be used for live eval.
- [x] 2026-06-23 10:21+08 h16384 positive-weight formal scorer was launched
      on allocation `146658`, step `146658.137`, tmux window
      `train0204_h16384_formal2`. Output:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h16384_posw_formal3h_seed20260725_20260623_alloc146658`.
      It uses the same merged data, same split seed `20260725`, same score/gate
      as the earlier h16384 positive-weight formal, and binary positive weights
      `1,4,1,1,2`. GPU utilization checked at `52%`. This run was interrupted
      after about `20` minutes because the existing `--allow-handoff-only-gate`
      logic saved a `checkpoint_best_gate.pt` at step `2000` from handoff
      improvement `10/32` versus DP `8/32`, while weighted error and progress
      were not safely better. That is an over-permissive gate for this project
      and the interrupted run is not formal evidence.
- [x] 2026-06-23 10:42+08 h16384 positive-weight safegate formal completed
      active on allocation `146658`, step `146658.141`, tmux window
      `train0204_h16384_safegate`. Output:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h16384_posw_safegate1_formal3h_seed20260725_20260623_alloc146658`.
      It removes `--allow-handoff-only-gate`, so a saved gate checkpoint must
      improve handoff and also satisfy the existing weighted-error/progress
      safety constraints. This preserves the original objective and prevents a
      pure handoff-number checkpoint from entering live eval. Final summary:
      `training_summary.json` reports `elapsed_seconds=10802`,
      `formal_training_floor_met=true`, and `ready_for_formal_live_eval=true`.
      Exact checkpoint:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h16384_posw_safegate1_formal3h_seed20260725_20260623_alloc146658/checkpoint_best_gate.pt`.
      Best step `7000`: validation selected handoff `11/32`, DP `8/32`,
      oracle `12/32`, handoff delta `+0.09375`, weighted-error delta
      `-0.00590`, progress delta `+0.00261`. This is offline eligibility only,
      not live method evidence.
- [x] 2026-06-23 11:08+08 h8192 safegate formal completed on
      the same allocation, step `146658.144`, tmux window
      `train0204_h8192_safegate`. Output:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h8192_safegate1_formal3h_seed20260725_20260623_alloc146658`.
      Reason: the h16384 run is meaningful but GPU utilization was bursty and
      sometimes below the cluster threshold; h8192 safegate is the same data,
      same split, same strict gate, useful as a capacity comparison, and helps
      keep the held 1-GPU allocation active. Summary reports
      `elapsed_seconds=10800`, `formal_training_floor_met=true`, and
      `ready_for_formal_live_eval=true`. Exact checkpoint:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h8192_safegate1_formal3h_seed20260725_20260623_alloc146658/checkpoint_best_gate.pt`.
      Best step `7000`: validation selected handoff `11/32`, DP `8/32`,
      oracle `12/32`, handoff delta `+0.09375`, weighted-error delta
      `-0.00349`, progress delta `+0.00404`.
- [ ] 2026-06-23 11:24+08 both safegate formal runs have now produced strict
      offline gate checkpoints, but neither has met the formal time floor yet.
      h16384_posw safegate at step `8000`: validation selected handoff
      `10/32`, DP `8/32`, oracle `12/32`, handoff delta `+0.0625`,
      weighted-error delta `-0.00108`, progress delta `+0.00006`,
      `checkpoint_best_gate.pt` exists. h8192 safegate at step `3000`:
      validation selected handoff `11/32`, DP `8/32`, oracle `12/32`,
      handoff delta `+0.09375`, weighted-error delta `-0.00122`, progress
      delta `+0.00311`, `checkpoint_best_gate.pt` exists. Continue both to
      `10800` seconds before any live eval.
- [ ] 2026-06-23 13:56+08 both strict safegate formal scorers are live-eval
      eligible. h16384 has stronger weighted-error delta (`-0.00590`);
      h8192 has slightly stronger progress delta (`+0.00404`) with the same
      selected handoff `11/32`. Next action: run a strict full `301/300` live
      panel using an exact `checkpoint_best_gate.pt`, then inspect video/contact
      evidence before interpreting success.
- [ ] 2026-06-23 14:01+08 h8192 safegate live panel launched on held
      allocation `146658`, step `146658.148`, tmux window
      `live_h8192_safegate0134`. Launcher:
      `experiments/world_model_task_rebinding/cosmos3/launch_live_panel_seed20260725_h8192_safegate1_panel0134_20260623_alloc146658.sh`.
      Exact scorer checkpoint:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h8192_safegate1_formal3h_seed20260725_20260623_alloc146658/checkpoint_best_gate.pt`.
      It tests samples `0,1,3,4` with full `301/300`, source-suffix offsets
      `64,48,32,24`, 8-step execution, and DP96 handoff. This is the first live
      test from the augmented strict-gate scorer and still requires final
      simulator state plus direct video/contact-sheet inspection.
- [x] 2026-06-23 current blocker plan preserved under a new root instruction
      folder:
      `Instruction/cosmos3_current_blockers/2026-06-23_selector_live_headroom_plan.md`.
      It records the plain conclusion that the method is not proven successful:
      offline handoff-continuable headroom exists, but live candidate generation
      and selector/scorer generalization do not yet reliably convert that
      headroom into closed-loop insertion. The next aligned diagnostics are
      live-candidate-only headroom, source/scenario/phase split audit, selected
      family versus oracle-winner family comparison, and structured visual
      review writeback for any future success.
- [x] 2026-06-23 15:08+08 panel0134 sample00 completed inside the active
      h8192 safegate live panel. This is an interim panel fact, not the final
      panel conclusion. Output:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_safegate1_source_suffix_offsets64_48_32_24_panel0134_20260623_alloc146658/sample_00_hole_late_move_stop`.
      Result: full contract passed (`301` frames, `300` actions,
      `video_file_contract_ok=true`), final simulator `success=false`, final
      peg-head-at-hole `[-0.097125, 0.014449, -0.071617]`, and
      `continuability_gate_ok_count=0/43`. The sample never entered a safe DP
      handoff state. The panel has moved on to sample01.
- [x] 2026-06-23 15:38+08 panel0134 sample01 completed and failed. Output:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_safegate1_source_suffix_offsets64_48_32_24_panel0134_20260623_alloc146658/sample_01_hole_late_constant`.
      Result: full contract passed (`301` frames, `300` actions,
      `video_file_contract_ok=true`), final simulator `success=false`, final
      peg-head-at-hole `[-0.092171, 0.032896, -0.078168]`.
      Unlike sample00, this run did execute `96` DP handoff steps after
      `2/29` continuability-gate positives, but final insertion still failed.
      This directly supports the current blocker that replay/restored
      DP-handoff labels are not equivalent to real closed-loop handoff
      survival after compounding errors.
- [x] 2026-06-23 15:59+08 panel0134 sample03 completed and failed. Output:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_safegate1_source_suffix_offsets64_48_32_24_panel0134_20260623_alloc146658/sample_03_hole_late_fast_shift`.
      Result: full contract passed (`301` frames, `300` actions,
      `video_file_contract_ok=true`), final simulator `success=false`, final
      peg-head-at-hole `[-0.105599, 0.001399, 0.003859]`, and `96` DP handoff
      steps were executed after `2/19` gate positives. This is a clean failure
      mode: y/z became nearly aligned, but insertion-axis x remained about
      `10.6cm` short, so DP handoff could not finish. This supports the
      planned scorer repair: learn insertion/contact continuability, not just
      lateral alignment or handoff bit.
- [x] 2026-06-22 17:20+08 root-level instruction update:
      `Instruction/README.md` and
      `Instruction/cosmos3_lowfreq_wm_executor_current.md` are the current
      concise judgment entry points. They now record both sides of the latest
      source-suffix result: sample 5 has one real full-episode live success,
      while corrected sample 00 shows a hard action-candidate coverage miss.
- [x] 2026-06-22 Chinese top-level blocker document added under root
      `Instruction/`: `Instruction/cosmos3_selector_blockers_plain_zh_current.md`.
      It preserves the user-level judgment in plain Chinese: selector
      generalization is weak, replay oracle/live candidates can diverge, gate
      criteria can accept bad checkpoints, validation is small, split leakage
      must be audited, visual review must be structured, and repo state must
      be cleaned after the active formal scorer summary exists.
- [x] 2026-06-22 root-level watch instruction added:
      `Instruction/current_watch_items_zh.md`. It records the current
      action/observation loop in plain Chinese: no active Slurm job, rank-loss
      scorer failed, sample00 offset coverage is the immediate diagnostic, and
      the next compute action must run inside a fresh tmux-held allocation.
- [x] 2026-06-22 corrected source-suffix safety boundary was implemented and
      compile-checked inside allocation `145920`: same-scenario suffixes,
      source-suffix distance cap `0.02`, source-suffix execution `8` steps,
      candidate default execution `8` steps, and pure `dp_prior` blocked when
      live `C_pi` is false. These are safety/causality boundaries, not a
      metric-relaxing change.
- [x] 2026-06-22 sample 00 corrected all-candidate replay completed.
      Corrected live root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_panel0245_cap_scenario_k2_dist002_exec8_nodpunsafe_20260622_alloc145920`.
      Replay root:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_sample00_iter0_all_candidates_dist002_exec8_nodpunsafe_20260622_alloc145920`.
      Result: `213` valid candidates, `0` success, `0` after-gate states,
      `0` y/z improvements, and `213/213` worsened y/z. This means the current
      action bank lacks a usable chunk for this early live state; the next
      useful work is stronger dense/source/live-state executor or action
      candidate training, not rerunning the same candidate pool.
- [x] 2026-06-22 17:25+08 corrected sample 5 survival check completed inside
      held allocation `145920`, step `145920.307`, tmux
      `cosmos3_h96_live_family_labels_realloc2g_request_20260622_001307`.
      Output root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_sample5_dist002_exec8_nodpunsafe_20260622_alloc145920`.
      The run used same-scenario source suffixes, source-suffix start distance
      cap `0.02`, 8-step execution before re-observation, and no pure
      `dp_prior` execution while live `C_pi` was false. It preserved the full
      `301/300` contract. Iterations `0-5` selected
      `retrieval_resid_srcsuffix_r0_s0p5_o32`; after iteration `5`, real state
      reached `[-0.112408, 0.009808, -0.003034]` and crossed `C_pi=true`.
      Frozen DP handoff then ran from frame `123`; final real
      peg-head-at-hole was `[0.030646, -0.003007, -0.002958]` with
      `success=true`. Extracted review sheet
      `live_observed_rollout_annotated_review_sheet.png` and frame `300` were
      opened; the peg stayed held and visually entered the box/hole region.
      Plain conclusion: the corrected interface can work on sample 5. It does
      not erase the sample 00 blocker, where the corrected candidate bank had
      no usable action.
- [x] 2026-06-22 strict sample 5 all-candidate replay completed. Root:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_sample05_strict_all_candidates_dist002_exec8_nodpunsafe_20260622_alloc145920`.
      Result: `1300` valid candidates, `0` direct success, `214`
      after-gate states, `651` y/z-improving candidates, and `649`
      y/z-worsening candidates. Iteration `2` alone had `213/215`
      after-gate candidates. The six live-selected source-suffix candidates
      had `0/6` one-step after-gate outcomes in this replay. Plain conclusion:
      sample 5 is not a candidate-coverage miss like sample 00. Its action
      bank has many C_pi-reaching chunks, but the selector/handoff target is
      not choosing or validating the best short chunk efficiently.
- [x] 2026-06-22 sample 5 after-gate + DP96 replay completed inside
      allocation `145920`, step `145920.311`. Filter:
      `experiments/world_model_task_rebinding/cosmos3/sample05_strict_after_gate_candidate_filter_20260622.tsv`
      with `214` rows. Output:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_sample05_strict_aftergate_dp96_20260622_alloc145920`.
      Result: `214/214` valid after-gate candidates, `118/214` DP96 final
      success, `208/214` DP-continuable, `209/214` contact-stable, and
      `214/214` y/z-improving. Plain conclusion: sample 5 has real selectable
      short-chunk headroom; the selector missed many candidates that could
      have handed off to DP much earlier. This is different from sample 00,
      where no candidate reached the gate at all.
- [ ] Next immediate repair split:
      sample 5 needs selector/DP-continuability scoring from real replay
      labels; sample 00 needs stronger action-candidate generation or
      executor coverage. Do not collapse these two into one threshold tweak.
- [x] 2026-06-22 short mixed scorer sanity completed inside allocation
      `145920`. Mixed root:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_sample00neg_sample05dp96_mix_20260622_alloc145920`.
      Conversion produced `1513` valid rows over `7` live states. The first
      retry, `scorer_overfit100_retry_boolfix`, completed `100` steps but used
      `rank_loss=0` and did not select the handoff-positive candidate. The
      second retry, `scorer_rank_overfit100_val0`, used `rank_loss=1`,
      full-group batches, and no validation split; it learned the within-state
      ordering and selected the one handoff-positive candidate. Limitation:
      all `118` DP96-success rows came from sample05 iteration `2`, where
      `dp_prior` also succeeded, so selected success was `1/7`, equal to DP
      prior `1/7`, and no offline gate/checkpoint was valid. Plain conclusion:
      rank loss is necessary, but this label set is too small and not a formal
      selector proof. Need broader real replay labels with DP-failure states
      that contain non-DP-success candidates; sample00 still needs stronger
      action-candidate coverage.
- [x] 2026-06-22 launched formal rank-loss source-suffix scorer inside held
      allocation `145920`, step `145920.315`. Output:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_live_snapshot_source_suffix_rank1_fullbatch_20260622_alloc145920_formal3h`.
      It trains on
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_source_suffix_union_20260622_alloc145920`,
      which has `41` live states, `8877` candidates, and `3` states where DP
      prior fails but a non-DP candidate has DP96 success. Main changes versus
      the earlier formal source-suffix scorer: `rank_loss=1.0`, batch size
      `8192` for full-group ranking, and the same `10800` second formal floor.
      First eval wrote successfully at step `1`. This is selector-side work
      only; it does not solve sample00's action-candidate coverage miss.
- [x] 2026-06-22 early split audit for the running rank-loss scorer:
      validation has `8` states and only one DP96-success state, sample05
      iter05 frame `198`; training positives include sample05 iter04 frame
      `174` from the same source/scenario and sample04 iter05 frame `196`.
      At step `1000` and `3000`, training selected `2/33` handoff successes
      but validation selected `0/8` while validation oracle has `1/8`.
      Plain conclusion so far: even with nearby source/scenario positives in
      training, the scorer has not generalized to the validation positive.
- [x] 2026-06-22 rank-loss formal scorer interim at about `46` minutes:
      step `13000`, train selected handoff `2/33`, validation selected
      handoff `0/8`, validation oracle `1/8`, and validation selected
      weighted error/contact progress remain worse than DP. Continue to the
      `10800` second formal summary before treating it as final.
- [x] 2026-06-22 rank-loss formal scorer interim at about `76` minutes:
      step `22000`, train rank CE is near `0`, train selected handoff remains
      `2/33`, validation selected handoff remains `0/8`, validation oracle is
      still `1/8`, and validation weighted error/contact progress are still
      worse than DP. This strengthens the selector-generalization diagnosis,
      but final evidence still requires the formal summary.
- [x] 2026-06-22 rank-loss formal scorer interim at about `106` minutes:
      step `32000`, validation selected handoff is still `0/8` while oracle is
      `1/8`. Validation weighted error is slightly better than DP, but contact
      progress remains worse and the handoff positive is still missed. No live
      eval is justified from this interim state.
- [x] 2026-06-22 rank-loss formal scorer interim at about `134` minutes:
      step `42000`, validation selected handoff is still `0/8`, validation
      oracle remains `1/8`, and contact progress remains worse than DP. No
      live eval gate is available from this checkpoint.
- [x] 2026-06-22 rank-loss formal scorer interim at about `168` minutes:
      step `52000`, validation selected handoff is still `0/8` while
      validation oracle remains `1/8`. This is close to the formal `3` hour
      floor and currently points to selector generalization failure, not a
      live-eval-ready checkpoint.
- [x] 2026-06-22 rank-loss formal scorer final summary completed. Root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_live_snapshot_source_suffix_rank1_fullbatch_20260622_alloc145920_formal3h`.
      It met the formal floor: `formal_training_floor_met=true`, `steps=55962`,
      `stop_reason=max_wall_seconds`. It failed the method gate:
      `ready_for_formal_live_eval=false`, `final_ready_for_offline_gate=false`,
      `best_gate_metrics=null`, no `checkpoint_best_gate.pt`. Final validation:
      selected handoff `0/8`, DP `0/8`, oracle `1/8`, selected weighted error
      worse than DP by `+0.00246`, contact progress worse by `-0.02760`.
      Final train selected the two available train handoff positives (`2/33`).
      Plain conclusion: rank1 learned train positives but did not generalize
      to validation; do not run live eval from this scorer. Evidence note:
      `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_rank1_formal_selector_failure.md`.
- [x] 2026-06-22 follow-up offline audits completed for the failed rank1
      scorer. Evidence note:
      `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_rank1_failure_offline_audits.md`.
      Main results: all `18` DP96-success rows and all `3` success groups in
      this formal label union are source-suffix candidates; checkpoint-model
      non-DP candidates have `0/41` success groups; train/val share
      source/scenario/phase, so the validation miss is not because the
      positive came from a completely unseen setting; the final scorer selected
      one source-suffix candidate but missed the source-suffix handoff oracle.
      Decision: no live eval from this scorer. Next repair must address
      candidate/executor coverage and source-suffix/action scoring, not gate
      threshold tweaking.
- [x] 2026-06-22 sample00 source-suffix offset coverage audit completed.
      Evidence note:
      `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_sample00_source_suffix_offset_coverage.md`.
      Current failed sample00 iter0 used source-suffix offsets `32,24`, and
      the saved live action chunk had `source_suffix_candidate_count=0`.
      Read-only bank audit showed same-scenario offsets `32,24` nearest
      distance `0.02073`, just outside the active `0.02` cap, while
      same-scenario offsets `48,64` enter the cap with nearest distance
      `0.00652`. This makes sample00 an offset/action-candidate coverage
      miss, not a threshold-relaxation argument.
- [x] 2026-06-22 sample00 one-iteration offset diagnostic initially needed a fresh
      tmux-held allocation. The first launch attempted output root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_sample00_oneiter_20260622_alloc145920`,
      but Slurm revoked allocation `145920` and step `145920.316` was
      cancelled after `4` seconds. This produced no method result. It was
      superseded by the completed rerun on allocation `146658` with
      `--source-suffix-offsets 64,48,32,24`,
      `--max-receding-iterations 1`, same-scenario source suffixes, distance
      cap `0.02`, 8-step execution, and no unsafe DP prior while live
      `C_pi=false`.
- [x] 2026-06-22 sample00 offsets `64,48,32,24` one-iteration diagnostic
      completed on allocation `146658` after server27 allocation `146639`
      twice failed first-frame render with Vulkan `DeviceLost`. Output:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_sample00_oneiter_vkfix_20260622_alloc146658`.
      Result: sample00 iter0 candidate count increased from `213` to `215`;
      `source_suffix_candidate_count=2`; selected candidate was
      `retrieval_resid_srcsuffix_r0_s1_o48`. The 8-step execution alone did
      not pass C_pi and moved y/z worse, so this diagnostic was not method
      success by itself.
- [x] 2026-06-22 sample00 source-suffix-only DP96 replay completed. Root:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_sample00_offsets48_source_suffix_candidates_dp96_20260622_221837_alloc146658`.
      It replayed only candidate indices `213` and `214` from the new
      sample00 iter0 bank. Result: `2` valid rows, `0` after-gate, `0`
      y/z-improving, `2` y/z-worsening, but `1/2` DP96 success and `1/2`
      DP-continuable/contact-stable. The live-selected candidate
      `retrieval_resid_srcsuffix_r0_s1_o48` was the DP96-positive one:
      after chunk C_pi was false, but DP96 reached final peg-head-at-hole
      `[-0.003523, 0.002995, 0.002409]` and `success=true`. Plain conclusion:
      current instantaneous C_pi is not a reliable handoff label; train
      handoff scoring from real DP-rollout continuability labels.
- [x] 2026-06-22 sample00 full offsets `64,48,32,24` live run succeeded.
      Evidence note:
      `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_sample00_offsets64_full_success.md`.
      Root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_sample00_full_vkfix_20260622_alloc146658`.
      Result: `completed_iterations=8`, `final_observed_frames=301`,
      `full_episode_length_ok=true`, `video_file_contract_ok=true`,
      final simulator `success=true`, final peg-head-at-hole
      `[-0.007180, 0.000003, 0.001853]`. Iterations `0-5` selected
      source-suffix chunks; after iteration `5` real C_pi became true, and
      DP handoff finished insertion. Contact sheet
      `live_observed_rollout_annotated_sheet.jpg` was opened and matches the
      metric success. This is now a second strict full-episode live success
      after sample05, but it is still not broad method success.
- [x] 2026-06-22 panel-resample correction for sample00 completed.
      Evidence note:
      `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_panel0245_offsets64_sample00_resample_failure.md`.
      Root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658/sample_00_hole_late_move_stop`.
      Result: full contract passed (`301` frames, `300` actions,
      `video_file_contract_ok=true`) but final simulator `success=false`,
      final peg-head-at-hole `[-0.095756, 0.016172, -0.065285]`.
      Visual sheet
      `live_observed_rollout_annotated_review_sheet.jpg` was opened and
      confirms the peg is not inserted at the final frame. Causal trace:
      source-suffix chunks made C_pi true by frame `168`, DP96 then moved the
      real state to `[-0.097337, 0.006050, -0.049492]` and broke
      continuability; late frames had `0` source-suffix candidates and
      checkpoint-model chunks did not recover insertion. Plain conclusion:
      sample00 has one successful run but is not stably fixed. The active
      blocker is live handoff/contact continuability plus late candidate
      coverage, not a threshold relaxation issue.
- [x] 2026-06-22 same panel sample02 completed and failed. Evidence note:
      `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_panel0245_offsets64_sample02_failure.md`.
      Root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658/sample_02_hole_late_reverse`.
      Result: full contract passed (`301` frames, `300` actions,
      `video_file_contract_ok=true`) but final simulator `success=false`,
      final peg-head-at-hole `[-0.112931, 0.047073, -0.061056]`.
      Visual sheet
      `live_observed_rollout_annotated_review_sheet.jpg` was opened and
      confirms the peg is not inserted at the final frame. Causal trace:
      checkpoint-model `scale_1.5` chunks reached C_pi around frame `144`,
      then DP96 from frame `148` ended at
      `[-0.113029, 0.031689, -0.054556]` with no success. Source-suffix
      candidates existed briefly at frames `136/144` but were not selected;
      late frames had only checkpoint-model short chunks. Plain conclusion:
      sample02 reinforces the same handoff/contact-continuability blocker as
      sample00, plus a selector miss when source-suffix candidates are present.
- [x] 2026-06-22 same panel sample04 completed successfully. Evidence note:
      `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_panel0245_offsets64_sample04_success.md`.
      Root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658/sample_04_hole_late_sine`.
      Result: full contract passed (`301` frames, `300` actions,
      `video_file_contract_ok=true`) and final simulator `success=true`,
      final peg-head-at-hole `[-0.004514, -0.002941, -0.002943]`.
      Visual sheet
      `live_observed_rollout_annotated_review_sheet.jpg` was opened and does
      not contradict the metric success. Causal trace: source-suffix and
      scale chunks reached C_pi at frame `160`; DP96 from frame `162` reached
      `[-0.004445, -0.002970, -0.002971]`; DP42 from frame `258` preserved
      final success. Plain conclusion: sample04 proves the corrected interface
      can produce a real DP-continuable live state, but the panel is still
      mixed because sample00 and sample02 failed.
- [x] 2026-06-22 same panel sample05 completed successfully. Root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658/sample_05_hole_late_continuous_insert`.
      Result: full contract passed (`301` frames, `300` actions,
      `video_file_contract_ok=true`) and final simulator `success=true`,
      final peg-head-at-hole `[0.029410, -0.002664, -0.002634]`.
      Visual sheet
      `live_observed_rollout_annotated_review_sheet.jpg` was opened and
      matches the metric success. Causal trace: source-suffix chunks carried
      the live state to C_pi at frame `150`; DP96 from frame `152` reached
      `[0.031493, 0.000430, 0.000377]`; DP52 preserved final success.
- [x] 2026-06-22 source-suffix offsets `64,48,32,24` panel `0,2,4,5`
      completed on allocation `146658`. Evidence note:
      `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_panel0245_offsets64_final_2of4_mixed.md`.
      Panel root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658`.
      Summary: `completed_samples=4`, `failed_process_count=0`,
      `panel_full_episode_contract_ok=true`, and `final_success_count=2/4`.
      sample00 and sample02 are metric-and-visual failures; sample04 and
      sample05 are metric-and-visual successes. Plain conclusion: the
      corrected source-suffix/live-receding/DP-handoff interface can work, but
      it is not stable. Next repair target is real DP-rollout
      continuability/contact scoring plus late candidate coverage, not a
      threshold tweak or another identical panel repeat.
- [x] 2026-06-22 panel0245 targeted handoff-label replay completed on
      allocation `146658`. Evidence note:
      `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_panel0245_offsets64_handoff_label_replay_and_overfit.md`.
      Selected/source-suffix replay root:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_panel0245_offsets64_selected_sourcesuffix_dp96_20260622_alloc146658`.
      DP-prior replay root:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_panel0245_offsets64_dpprior_only_dp96_20260622_alloc146658`.
      Facts: selected/source-suffix replay produced `97` valid rows with
      `54` DP96 successes and `55` DP-continuable rows; source-suffix-name
      candidates had `53/76` DP96 success while live-selected candidates had
      `15/42`. DP prior had `16/42` DP96 success. The replay also found
      `45` cases where `C_pi=false` but DP96 succeeded, and `2` cases where
      `C_pi=true` but DP96 failed. Plain conclusion: the current handoff label
      must be real DP rollout continuability/contact quality, not
      instantaneous C_pi.
- [x] 2026-06-22 combined scorer dataset built from panel0245 handoff labels.
      Root:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_panel0245_offsets64_selected_sourcesuffix_dpprior_dp96_20260622_alloc146658`.
      Facts: `139` valid rows, `42` live-state groups, `42` groups with
      DP prior, `19` groups with source-suffix DP96 success, and `70` total
      DP96-success rows.
- [x] 2026-06-22 short handoff-rank overfit sanity completed. Root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_panel0245_offsets64_handoff_rank_overfit100_20260622_alloc146658`.
      Facts: `100` steps, `val_fraction=0`, formal floor false. It selected
      `20/42` handoff successes versus DP prior `16/42`, matching the
      handoff oracle count `20/42`. Plain conclusion: the new DP-rollout
      labels are learnable in overfit, but this is only a debug gate.
- [x] 2026-06-22 single-panel formal panel0245 handoff-rank scorer was
      interrupted because it underused the H200 allocation and validation was
      already below DP. Root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_panel0245_offsets64_handoff_rank_formal3h_20260622_alloc146658`.
      Config: `1` GPU, `min_wall_seconds=10800`, `max_wall_seconds=10800`,
      `32` train groups, `10` validation groups, `rank_loss_weight=1.0`,
      and handoff-success score weight `1.0`. Early result before interrupt:
      train selected handoff success was `13/32` versus DP `9/32`, but
      validation selected handoff success stayed below DP (`5-6/10` versus
      DP `7/10`). A larger single-panel retry
      `candidate_outcome_scorer_panel0245_offsets64_handoff_rank_formal3h_h4096_20260622_alloc146658`
      was also interrupted for the same reason. These are not formal results.
- [x] 2026-06-22 namespaced union+panel scorer dataset created. Root:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_source_suffix_union_plus_panel0245_offsets64_dp96_namespaced_20260622_alloc146658`.
      It combines the old source-suffix union (`8877` rows) and the new
      panel0245 labels (`139` rows) after prefixing UUIDs with `oldunion__`
      and `panel0245__` to avoid the `5` overlapping UUID strings. Final
      combined size: `83` base rows and `9016` outcome rows.
- [ ] 2026-06-22 union+panel handoff-rank formal scorer is running on
      allocation `146658`. Root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_source_suffix_union_plus_panel0245_handoff_rank_formal3h_20260622_alloc146658`.
      Config: `1` GPU, `min_wall_seconds=10800`, `max_wall_seconds=10800`,
      `62` train groups, `21` validation groups, `rank_loss_weight=1.0`,
      handoff-success score weight `8.0`, continuability weight `4.0`, and
      state penalty weights `[0.05,0.1,0.2]`. Step `1` is initialization only:
      validation selected handoff success equaled DP (`4/21`), train selected
      was `11/62` versus DP `12/62`. Wait for the 3-hour summary.
- [x] 2026-06-22 16:05+08 source-suffix-aware best-gate live sample 5
      completed successfully on allocation `145920`. Output:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_sample5_20260622_alloc145920`.
      Facts: `final_success=true`, `final_observed_frames=301`,
      `full_episode_length_ok=true`, `video_file_contract_ok=true`, and final
      peg-head-in-hole `[0.0365917, -0.0003869, -0.0028896]`. The first
      live decision selected source-suffix candidate
      `retrieval_resid_srcsuffix_r1_s1_o32`, then DP handoff completed
      insertion. This is one-sample conversion evidence only; do not call it
      broad method success.
- [ ] 2026-06-22 16:12+08 launched the next generality check inside held
      allocation `145920`, step `145920.290`, tmux
      `cosmos3_h96_live_family_labels_realloc2g_request_20260622_001307`.
      Output root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_panel0245_20260622_alloc145920`.
      It tests samples `0,2,4,5` with the same source-suffix-aware
      best-gate scorer, same source suffix bank, same DP96 handoff, and the
      same full `301/300` contract.
- [ ] 2026-06-21 current focused diagnosis is now maintained in:
      `Instruction/cosmos3_lowfreq_wm_executor_current.md`, with supporting
      evidence in
      `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-21_selector_generalization_blockers.md`.
      Treat the Instruction file as the concise entry point for the current
      blockers and next actions: both formal scorer retries met the 10800s
      floor but are unsafe for live evaluation because their small handoff
      gains come with worse weighted geometry/contact progress. The 2026-06-21
      audit shows live-family-only h96 handoff oracle remains high (`58/64`
      versus all-candidate `60/64`), so the next priority is a live-family-only
      safety selector, not live eval from the current checkpoints.
- [ ] 2026-06-21 prepared the offline selector-blocker audit entry point
      `scripts/slurm/run_cosmos3_h96_selector_blocker_audit_in_allocation.sh`
      and script `scripts/world_model/audit_cosmos3_h96_selector_blockers.py`.
      The audit was run inside allocation `145813` and wrote
      `experiments/world_model_task_rebinding/cosmos3/h96_selector_blocker_audit_20260621_233801_alloc145813`.
      Result: all-candidate handoff oracle `60/64`, live-family handoff oracle
      `58/64`, DP baseline `35/64`, no `source_uuid` train/val overlap, but
      scenario/phase overlap and only `16` validation groups. Treat this as an
      offline diagnosis only, not method evidence.
- [ ] 2026-06-21 23:45+08 next experiment: train/evaluate a live-family-only
      safety selector on the h96 union. It should filter training/eval to DP
      prior plus live-generatable candidate families, rank against the group DP
      prior and live-family handoff oracle, and pass only if held-out handoff
      improves without worse weighted geometry error or contact progress.
      Use the same tmux-held compute allocation rules and the full 3-hour
      formal training floor for method evidence.
- [ ] 2026-06-21 23:50+08 launched the default-capacity live-family safety
      scorer on held allocation `145814`, tmux
      `cosmos3_h96_live_family_safety_scorer_145814_20260621_234650`, root
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_scorer_20260621_234650_alloc145814_formal3h`.
      It uses live-family filtering and `allow_handoff_only_gate=false`; early
      history shows train selection improving while eval remains below DP, so
      this is testing the suspected selector generalization blocker.
- [ ] 2026-06-21 23:52+08 launched a low-capacity regularized live-family
      safety scorer on held allocation `145818`, tmux
      `cosmos3_h96_live_family_safety_lowcap_145818_20260621_235017`, root
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_lowcap_20260621_234950_alloc145818_formal3h`.
      It uses hidden dim `128`, `2` layers, dropout `0.30`, weight decay
      `0.001`, and the same safe gate. Let both formal runs reach their
      10800-second summaries before any live eval.
- [ ] 2026-06-21 23:55+08 safe-headroom audit on allocation `145813` wrote
      `experiments/world_model_task_rebinding/cosmos3/h96_live_family_safe_headroom_audit_20260621_235518_alloc145813`.
      Strict safe live-family handoff exists for `46/64`, but strict safe
      non-DP candidates exist for only `16/64`; among DP-failure groups, safe
      non-DP exists for only `11/29` overall and `2/7` in the audit val split.
      If the two live-family safety scorers fail, the next fix is to
      expand/rebalance labels around DP-failure states with safe non-DP live
      candidates, not to loosen the gate or run live from a harmful scorer.
- [ ] 2026-06-22 00:00+08 gate hygiene correction: the running low-capacity
      process saved an early `checkpoint_best_gate.pt` at a point that tied DP
      handoff (`8/16` vs `8/16`) while improving weighted error/progress. That
      is useful diagnostic signal but not enough for live eval under the
      current method boundary. Code has now been tightened so future scorer
      gates require held-out handoff to be strictly above DP and not worse on
      weighted error/contact progress. Interpret summaries from already
      running `145814`/`145818` with this manual stricter rule.
- [ ] 2026-06-22 00:15+08 project-level operating rules were separated into
      `Instruction/project_operating_rules_current.md`. This records the
      active research boundary, compute-node discipline, evidence rules, and
      reporting style so future sessions keep the same judgment frame instead
      of mixing it into method-specific TODO notes.
- [ ] 2026-06-22 00:15+08 resource correction: allocation `145813` was lost
      while trying to stop old foreground/keepalive steps. Treat this as a
      resource-handling/scheduling failure, not a method failure. Its partial
      live-family shard claim under
      `experiments/world_model_task_rebinding/cosmos3/hardphase_h96_live_family_shard_claims_20260622_000819_alloc145813`
      has no usable completed label union and must not be used for training.
- [ ] 2026-06-22 00:16+08 replacement allocation `145919` started on
      `server24` with 1 GPU, 8 CPUs, and 72G memory for the live-family h96
      label expansion. `145920` remains pending as a 2-GPU backup and `145815`
      remains pending as a 4-GPU backup. Use `145919` first for the
      live-family sharded replay because label expansion is the current aligned
      next action if the safety selectors keep failing.
- [ ] 2026-06-22 00:18+08 launched live-family h96 sharded replay inside
      allocation `145919`, tmux
      `cosmos3_h96_live_family_labels_145919_20260622_0018`, log
      `experiments/world_model_task_rebinding/cosmos3/h96_live_family_sharded_145919_20260622_001820.log`,
      claim root
      `experiments/world_model_task_rebinding/cosmos3/hardphase_h96_live_family_shard_claims_20260622_001820_alloc145919`,
      summary root
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_h96_live_family_claim_union_20260622_001820_alloc145919`.
      It is live-family-only, no legacy teacher candidates and no retrieval
      residual candidates, with skip shards `64,72,80,88,96,104,112,120`.
- [ ] 2026-06-22 00:35+08 resource hygiene for `145919`: early GPU use stayed
      near the release threshold because the label process is CPU-heavy while
      replay/render work is intermittent. Several experimental keepalive steps
      were started and then explicitly terminated by killing only
      `gpu_keepalive_until_process_exits.py` processes on `server24`; the main
      label process was left running. Current intended state is one keepalive
      session,
      `cosmos3_h96_live_family_labels_keepalive_145919_single_20260622_0035`,
      tied to the main sharded replay script. Do not use `scancel` on this
      allocation just to clean steps; if cleanup is needed, target only the
      keepalive process name inside the allocation.
- [ ] 2026-06-22 00:17+08 latest live-family safety scorer readout is still
      negative under the strict method rule. Default-capacity `145814` latest
      validation selected `6/16` handoff successes versus DP `8/16`, with
      worse weighted error and worse contact progress. Low-capacity `145818`
      latest validation selected `7/16` versus DP `8/16`, also with worse
      weighted error/contact progress. Do not run live eval from either unless
      the final formal summaries pass the strict manual gate.
- [ ] 2026-06-22 00:34+08 allocation `145814` disappeared from `squeue` after
      the live-family default-capacity safety selector had only run about
      `40` minutes. `sacct` shows the earlier rank0.2 gatefix step completed
      its 3-hour formal run, but the later live-family safety step
      `145814.24` was cancelled after `00:39:48`. Therefore the live-family
      default-capacity safety selector root
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_scorer_20260621_234650_alloc145814_formal3h`
      is partial-only and not method evidence.
- [ ] 2026-06-22 00:36+08 allocation `145815` started on `server46` with 4
      GPUs, 32 CPUs, and 220G memory. Rather than launching another scorer
      before fixing the suspected label-balance blocker, it was repurposed for
      parallel live-family h96 label expansion. Added
      `scripts/slurm/run_cosmos3_h96_live_family_sharded_replay_multiworker_in_allocation.sh`
      and a `RUN_UNION_AFTER_SHARDS` switch in
      `scripts/slurm/run_cosmos3_hardphase_h96_sharded_replay_in_allocation.sh`.
      Launched tmux
      `cosmos3_h96_live_family_labels_multi4_145815_20260622_0036`, sharing
      claim root
      `experiments/world_model_task_rebinding/cosmos3/hardphase_h96_live_family_shard_claims_20260622_001820_alloc145919`.
      The four workers successfully skipped active `skip64` and claimed
      `skip72`, `skip80`, `skip88`, and `skip96` without duplicate shards.
- [ ] 2026-06-22 00:40+08 `145815` multiworker label expansion is live and
      writing candidate action files in all four claimed shards. A multi-GPU
      keepalive session
      `cosmos3_h96_live_family_labels_multi4_keepalive_145815_20260622_0042_8192`
      is running with one keepalive process per GPU, tied to the multiworker
      driver process. GPU utilization is still bursty because label replay is
      CPU-heavy, but all four label workers are active. Do not use `scancel`
      for cleanup; if keepalive cleanup is needed, target only
      `gpu_keepalive_until_process_exits.py` inside the allocation.
- [ ] 2026-06-22 00:44+08 low-capacity safety scorer on `145818` reached a
      point with held-out selected handoff `9/16` versus DP `8/16`, but it
      still worsened weighted geometry error (`+0.0829`) and contact progress
      (`-0.0512`). This confirms the current unsafe-scorer pattern: a small
      handoff-count gain is not enough for live eval when the selected state is
      physically worse for insertion handoff. Continue the 10800-second formal
      run, but do not use this checkpoint for live evaluation under the strict
      manual gate.
- [ ] 2026-06-22 00:47+08 first new live-family label shard completed:
      `skip64` from allocation `145919` wrote
      `candidate_outcome_labels.jsonl` and `done.txt` under the shared claim
      root. `145919` then skipped active `skip72/80/88/96` claims owned by
      `145815` and claimed `skip104`. This confirms the shared atomic-claim
      parallelization is working and not duplicating shards.
- [ ] 2026-06-22 01:04+08 additional live-family label shards completed:
      `skip72`, `skip88`, and `skip96` from the `145815` multiworker run wrote
      final `candidate_outcome_labels.jsonl` files and `done.txt` claim
      markers. `skip80` is still active but near completion. Finished workers
      have advanced to later claims including `skip112` and `skip120`; `145919`
      continues `skip104`.
- [ ] 2026-06-22 01:05+08 interim union diagnostic over the completed live-
      family shards wrote
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_h96_live_family_claim_union_interim_done4_20260622_0105`.
      It covers `40` groups: DP handoff success `24/40`, live-family handoff
      oracle `36/40`, meaningful improvement `29/40`, large improvement
      `14/40`, but final success oracle is only `3/40` versus DP `3/40`.
      Interpretation: the added labels are increasing handoff-continuable
      candidate headroom, which is the current target, but they do not yet
      prove final insertion success. The old retrieval gate reports
      `passed=false` partly because it expects retrieval-family contribution;
      do not use that gate as the live-family-only decision rule.
- [ ] 2026-06-22 01:18+08 `skip104` from allocation `145919` completed and
      wrote final `candidate_outcome_labels.jsonl` plus `done.txt`. The
      `145919` driver step then failed with exit `127` while trying to advance
      to later shards; the log shows a truncated command name
      `ardphase_exploratory_candidate_replay_in_allocation.sh`. The checked
      script file is correct, so this is attributed to patching the shell
      script while a bash process was still executing the old file descriptor.
      Treat completed `skip64` and `skip104` labels as usable; treat the
      `145919` driver failure as an execution hygiene issue, not a label
      method failure.
- [ ] 2026-06-22 01:19+08 allocation `145919` was idle after its driver exit,
      so a claim-count keepalive was launched in tmux
      `cosmos3_h96_live_family_labels_keepalive_145919_claim8_20260622_0119`.
      It runs `gpu_keepalive_until_claims_done.py` against the shared claim
      root and stops when `8` shard `done.txt` markers exist. Reuse `145919`
      for final union/scorer work after `skip112/120` complete.
- [ ] 2026-06-22 01:22+08 corrected the live-family safety scorer launcher:
      `scripts/slurm/run_cosmos3_h96_live_family_safety_scorer_in_allocation.sh`
      now sets `MIN_RETRIEVAL_ORACLE_COUNT=0` and
      `MIN_RETRIEVAL_SUCCESS_GROUPS=0` by default. Reason: the current label
      expansion intentionally excludes retrieval residual candidates, so the
      old retrieval-family gate would block scorer training even when
      live-family handoff/progress headroom exists. This does not loosen the
      held-out scorer safety gate; it only makes the union pre-gate match the
      live-family-only candidate source.
- [ ] 2026-06-22 01:35+08 all eight live-family label shards completed:
      `skip64,72,80,88,96,104,112,120`. Final union in
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_h96_live_family_safety_fullunion_20260622_013500_fullunion_alloc145920`
      covers `64` groups: DP handoff `40/64`, live-family handoff oracle
      `57/64`, meaningful improvement `44/64`, large improvement `24/64`,
      final-success oracle `5/64` versus DP `4/64`. The progress/headroom
      pre-gate passed. This is still offline replay/headroom evidence, not
      live method success.
- [ ] 2026-06-22 01:35+08 launched default-capacity full-union live-family
      safety scorer on allocation `145920`, tmux
      `cosmos3_h96_live_family_safety_fullunion_145920_20260622_0135`, root
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_fullunion_20260622_013500_fullunion_alloc145920_formal3h`.
      It uses the final eight-shard union, 2 visible GPUs, and the required
      10800-second formal floor. Early validation is still unsafe
      (`8/16` selected handoff versus DP `9/16` with worse geometry/contact),
      so do not use it for live eval unless a later formal gate passes.
- [ ] 2026-06-22 03:31+08 important default-scorer update: read-only history
      audit found `checkpoint_best_gate.pt` and `12` strict-gate points in the
      default-capacity full-union run. The latest strict point is step `52400`
      at about `1426s`: selected handoff `10/16` versus DP `9/16`, weighted
      error delta `-0.0296`, contact progress delta `+0.0085`, and non-DP
      selected fraction `0.4375`. This is the first aligned sign that the
      full-union labels can train a selector that improves held-out handoff
      without hurting geometry/contact. It is not live-eval evidence yet:
      wait for the same run to finish the `10800s` formal floor and summary.
- [ ] 2026-06-22 04:36+08 default full-union live-family safety scorer on
      allocation `145920` completed the formal floor. Root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_fullunion_20260622_013500_fullunion_alloc145920_formal3h`.
      Summary has `formal_training_floor_met=true`,
      `ready_for_formal_live_eval=true`, and `best_gate_step=34275`.
      Only `checkpoint_best_gate.pt` is usable: held-out selected handoff
      `10/16` versus DP `9/16`, weighted error delta `-0.0326008`, contact
      progress delta `+0.0146695`, non-DP fraction `0.4375`. The final/latest
      checkpoint is unsafe (`6/16` versus DP `9/16`, worse geometry and
      worse progress), so do not use `checkpoint_final.pt` or
      `checkpoint_latest.pt` for live eval.
- [ ] 2026-06-22 04:36+08 conservative margin eval also completed at
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_margin_hardphase_h96_live_family_safety_fullunion_20260622_013500_fullunion_alloc145920`.
      It reports `ready_for_conservative_offline_gate=true`, but only margin
      `0.0` passes: selected handoff `10/16` versus DP `9/16`, weighted error
      delta `-0.0326008`, progress delta `+0.0146695`, `6` improved switches
      and `1` harmful switch. Plain meaning: there is enough signal for a
      small live panel, but no positive safety margin; the panel must be
      treated as conversion evidence, not success.
- [ ] 2026-06-22 04:40+08 low-capacity full-union scorer on allocation
      `145815` was cancelled before the 3-hour formal floor. Its root
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_fullunion_lowcap_20260622_014000_fullunion_lowcap_alloc145815_formal3h`
      has no `training_summary.json` and no `checkpoint_best_gate.pt`. Treat
      it as partial-only.
- [ ] 2026-06-22 04:48+08 new evidence-chain caution: the h96 live-family
      label expansion defaulted to the hard-phase outcome-oracle smoke
      checkpoint
      `experiments/world_model_task_rebinding/cosmos3/outcome_oracle_candidate_executor_hardphase_balanced_smoke100_20260618_alloc139764/checkpoint_best_offline.pt`.
      That checkpoint was explicitly a short smoke, not formal method
      training. It can justify diagnostic label expansion, but a formal live
      claim should either use a formal 3-hour Gaussian hard-phase generator in
      the live panel or regenerate the h96 label/scorer chain from the exact
      formal generator used live. The next panel should therefore bind
      `checkpoint_best_gate.pt` from the full-union scorer to a formal
      hard-phase candidate generator when possible, keep
      `CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS=8,12,16`, and record the
      generator mismatch plainly in the manifest/notes.
- [ ] 2026-06-22 04:49+08 launched the next small live panel inside held
      allocation `145920`, tmux
      `cosmos3_h96_live_panel_145920_20260622_0458`, log
      `experiments/world_model_task_rebinding/cosmos3/h96_fullunion_scorer_live_panel_145920_20260622_0458.log`,
      output root
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_h96_fullunion_scorer_bestgate_formalhardphase_panel4_20260622_0458_alloc145920_samples0_1_3_4`.
- [ ] 2026-06-22 07:23+08 live panel completed. Root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_h96_fullunion_scorer_bestgate_formalhardphase_panel4_20260622_0458_alloc145920_samples0_1_3_4`.
      Formal facts: `final_success_count=0/4`, `failed_process_count=0`,
      `panel_full_episode_contract_ok=true`, every sample has `301` observed
      frames and `full_episode_length_ok=true`. Contact sheet:
      `live_receding_panel_contact_sheet.png`, visually inspected; no row shows
      completed insertion. This is a real negative result, not a scheduling,
      render, or truncation failure.
- [ ] 2026-06-22 07:23+08 per-sample live conclusion:
      sample 00 ended at `[-0.102740, -0.015748, -0.016932]`, no DP handoff,
      y/z still outside the continuability band. Sample 01 ended at
      `[-0.091294, -0.027579, -0.063382]`, no DP handoff, z badly outside the
      band. Sample 03 ended at `[-0.050987, 0.001507, 0.003059]` after `59`
      DP handoff steps; geometry was close and gates often passed, but final
      insertion remained false. Sample 04 ended at
      `[-0.102361, -0.020728, -0.028326]`, no DP handoff, y/z outside the
      band.
- [ ] 2026-06-22 07:25+08 current blocker judgment: offline h96 scorer
      headroom does not yet transfer to live closed-loop control. The scorer
      repeatedly chose `dp_prior`, `mean`, or short-prefix candidates whose
      live execution pushed y/z away or failed to complete contact insertion.
      The most important new failure is sample 03: reaching near-hole geometry
      and DP handoff is still not enough for final insertion. Therefore the
      next fix is not another threshold gate or another identical panel; the
      next fix is live-real outcome calibration/training for selected chunks,
      including real after-state y/z, grasp retention, post-chunk
      continuability, DP handoff result, and final insertion progress.
- [ ] 2026-06-22 next aligned TODO: build or run a live-real selected-chunk
      outcome extraction/calibration step from the completed panel artifacts.
      Pair each executed candidate record with its scorer prediction, real
      before state, real after state, gate result, DP handoff result, and final
      sample outcome. Use this to decide whether the existing scorer can be
      recalibrated with live labels or whether the executor needs stronger
      contact/insertion teacher chunks. This preserves the DDP-style contract:
      short chunks, live re-observation, learned scoring of task progress, and
      no hand-written recovery enumeration.
      This run uses clean-dense Cosmos iter1500, full `301/300` contract,
      samples `0,1,3,4`, controller source `candidate_executor`, formal
      hard-phase Gaussian executor
      `outcome_oracle_candidate_executor_hardphase_retrieval_union320_formal1gpu3h_20260618_alloc139841/checkpoint_final.pt`,
      h96 full-union outcome scorer
      `candidate_outcome_scorer_hardphase_h96_live_family_safety_fullunion_20260622_013500_fullunion_alloc145920_formal3h/checkpoint_best_gate.pt`,
      and short-prefix candidates `8,12,16`. Purpose: test whether the
      formal offline h96 selector can transfer into real closed-loop
      reobservation. It is not success until the panel finishes, real final
      states are summarized, and video/contact sheets are inspected.
- [x] 2026-06-22 built diagnostic successful-source insertion suffix bank on
      allocation `145920`:
      `experiments/world_model_task_rebinding/cosmos3/source_insertion_suffix_bank_20260622_alloc145920`.
      It mined `4711` insertion suffix rows from all `733` accepted source
      H5 files, with start median around the false-gate live region
      (`x≈-0.110`) and end median near inserted state. This is only a
      candidate-source diagnostic artifact, not method evidence.
- [x] 2026-06-22 replayed source insertion suffix candidates from the latest
      saved live snapshots:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_source_suffix_replay_sine_cont_iter458_20260622_alloc145920`.
      Result: `144` valid labels, direct 32-step source-suffix success
      `0/144`, after-geometry-gate `15/144`, DP96 success `18/144`,
      DP96 continuable `19/144`, final contact-stable `19/144`, and final
      grasp `144/144`. This is an important diagnostic improvement over the
      current learned/live candidate family, whose gate-positive candidates
      produced `0/211` DP96 success.
- [ ] Next highest-priority repair: add a causal insertion-suffix or retrieval-
      residual candidate family to live candidate generation/label export, and
      train or recalibrate the scorer on real DP96 success/continuability and
      contact-stable labels. Do not use scenario labels for control flow and
      do not treat source-suffix replay as closed-loop method success. The
      physical target is insertion-axis/contact progress into a DP-finishable
      state, not another y/z geometry threshold.
- [x] 2026-06-22 wired source-suffix candidates into a live receding smoke:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_smoke_sample5_20260622_alloc145920`.
      Result: process completed, `0/1` success, `3` receding iterations, and
      `151` observed frames. This is not full-episode method evidence. The
      useful fact is that each iteration had `16` source-suffix candidates,
      but the old scorer selected `scale_0.2`, then `dp_prior`, then
      `dp_prior`; it never selected source suffix. Contact sheet review agrees
      with the metric failure: the peg remains outside the hole.
- [ ] Immediate next action: convert live snapshot labels into the exact
      outcome-scorer training contract used by live selection. Merge all-bank
      DP/model labels as baseline/negative candidates with source-suffix DP96
      labels as the new positive candidate source, then train a
      source-suffix-aware scorer. Do not rerun the same live panel before this
      scorer exists.
- [ ] 2026-06-22 01:40+08 launched low-capacity regularized full-union scorer
      on allocation `145815`, tmux
      `cosmos3_h96_live_family_safety_fullunion_lowcap_145815_20260622_0140`,
      root
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_fullunion_lowcap_20260622_014000_fullunion_lowcap_alloc145815_formal3h`.
      It uses hidden dim `128`, `2` layers, dropout `0.30`, weight decay
      `0.001`, lr `0.00015`, batch size `128`, 4 visible GPUs, and the same
      10800-second formal floor. Early validation is closer but still not
      safe: `10/16` selected handoff versus DP `9/16`, weighted error
      slightly better, but contact progress still slightly worse.
- [ ] 2026-06-22 00:52+08 low-capacity live-family safety scorer allocation
      `145818` was cancelled by Slurm before the required 10800-second formal
      floor. It wrote no `training_summary.json`; latest history was about
      `3552` seconds and still unsafe (`9/16` handoff versus DP `8/16`, but
      weighted error `+0.0899` and contact progress `-0.0619`). Treat
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_live_family_safety_lowcap_20260621_234950_alloc145818_formal3h`
      as partial-only and do not use its `checkpoint_best_gate.pt` for live
      evaluation.
- [ ] 2026-06-21 19:50+08 current h96 handoff-aware status: the 64-row
      hard-phase h96 replay union is complete and shows real offline headroom
      (`35/64` DP handoff success versus `60/64` handoff-oracle success), but
      this is still not live method evidence. The active blocker is now
      selector generalization: the scorer must choose handoff-success
      candidates on held-out groups before any closed-loop live panel.
- [ ] 2026-06-21 19:50+08 the rank0.2 formal scorer comparison on allocation
      `145549` was interrupted by Slurm at `2026-06-21T19:46:44` before the
      required 3-hour floor. Treat
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_retrieval_rank02_formal3h_20260621_1720_alloc145549`
      as a revoked partial run only unless a later audit proves a complete
      `training_summary.json` with `formal_training_floor_met=true`.
- [ ] 2026-06-21 19:50+08 the rank0 scorer on allocation `145550` is still
      the active formal run. Latest read-only log check around `7506s` of
      scorer training showed train selection improving (`42/48` selected
      handoff successes versus `27/48` DP), but held-out selection was only
      tied with DP (`8/16` versus `8/16`) and had worse weighted error. Do
      not start live eval from this checkpoint unless the final 3-hour summary
      and margin eval pass a held-out handoff-selection gate.
- [ ] 2026-06-21 19:53+08 follow-up correction: allocation `145550` was also
      revoked at `2026-06-21T19:51:53`, after about `7540s` of scorer
      training and before the required 3-hour floor. It wrote
      `checkpoint_latest.pt` and `training_history.json`, but no
      `training_summary.json`; therefore rank0 is also partial-only evidence.
      New tmux-held 1-day `salloc` requests were opened for retry resource
      acquisition: `145814` 1-GPU, `145813` 2-GPU, and `145815` 4-GPU. Use
      whichever valid allocation starts first for a fresh full 3-hour h96
      handoff scorer retry, and do not launch live eval from the revoked
      partial checkpoints.
- [ ] 2026-06-21 20:05+08 implementation correction for the retry: the old
      scorer evaluated `dp_rollout_success` as handoff success, but its binary
      training heads and selector score did not directly predict or reward
      that h96 handoff-success label. This explains why the h96 union had
      `60/64` handoff-oracle headroom but the partial scorer still selected
      poorly on held-out groups. The retry code now adds a fifth binary head
      `dp_rollout_success_or_final_success`, a
      `--score-handoff-success-weight` selector term, checkpoint
      `binary_dim` metadata, and backward-compatible 4-head checkpoint loading
      for margin eval/live scoring. The offline gate also now has a handoff
      branch: a checkpoint may pass without immediate weighted-error
      improvement only if held-out selected handoff success is strictly above
      the DP baseline and non-DP selection is nontrivial. This preserves the
      same receding candidate-plus-DP-handoff objective; it is not an
      error-recovery branch.
- [ ] 2026-06-21 20:08+08 because the existing 1/2/4-GPU retry requests were
      still pending on scheduler priority, an additional lower-resource
      1-GPU/4-CPU/32G 1-day tmux-held request was opened as `145818`
      (`cosmos3_h96_formal_retry1g_lowmem`). Keep all requests pending; use
      the first valid allocation that starts, then run compute-node
      `py_compile` followed by the full 3-hour h96 handoff-success scorer
      retry. Do not clear the user-owned `reflex` tmux session.
- [ ] 2026-06-21 20:10+08 added the compute-node launcher
      `scripts/slurm/run_cosmos3_h96_handoff_success_scorer_retry_in_allocation.sh`.
      It refuses login-node execution, compiles the touched scorer/live
      scripts inside the compute step, and then runs the h96 union scorer with
      `SCORER_SCORE_HANDOFF_SUCCESS_WEIGHT=4.0`,
      `SCORER_SCORE_CONTINUABLE_WEIGHT=1.0`,
      `SCORER_SCORE_SUCCESS_WEIGHT=0.5`, and the strict 10800-second formal
      wall-clock floor. Use this launcher for the next retry instead of
      manually reconstructing the long env command.
- [ ] 2026-06-21 20:12+08 allocation `145813` / `server02` started and the
      autostart watcher launched tmux session
      `cosmos3_h96_handoff_success_retry_train_145813_20260621` plus
      keepalive sessions. Scorer root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_145813_20260621_200722_formal3h`.
      The compute-node launcher reached training and wrote the 5-head manifest.
      Early non-formal metric around `59s` of training: held-out selected
      handoff success `9/16` versus DP `8/16`, selected-minus-DP handoff
      fraction `+0.0625`, top1 handoff-oracle match `0.25`, and selected
      weighted-error delta about `+0.0008`. This is promising relative to the
      revoked old objective, but not formal evidence until the 10800-second
      floor and final margin eval complete.
- [ ] 2026-06-21 20:16+08 correction: the first `145813` autostart scorer
      root above is also partial-only. It was stopped before the formal floor
      after inspection showed the offline gate could still save a checkpoint
      from the old continuous-error branch even when held-out handoff success
      was worse than frozen DP. Do not use
      `candidate_outcome_scorer_hardphase_h96_handoff_success_retry_145813_20260621_200722_formal3h`
      as method evidence.
- [ ] 2026-06-21 20:18+08 active scorer attempt is now the gate-fixed root
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_gatefix_145813_20260621_2018_formal3h`,
      running in tmux
      `cosmos3_h96_handoff_success_retry_gatefix_train_145813_20260621` on the
      same held allocation `145813`. The gate now requires handoff non-worse
      for the continuous branch and ranks `best_gate` by held-out handoff
      delta first. Early history is mixed: a best point reached `9/16`
      selected handoff successes versus DP `8/16`, but latest validation has
      also dropped below DP. Therefore the next allowed action is only to wait
      for the full 3-hour summary and margin eval; launch live eval only if
      `formal_training_floor_met=true`, `ready_for_formal_live_eval=true`, and
      the saved `best_gate` beats DP on held-out handoff success.
- [ ] 2026-06-21 20:27+08 retry allocation `145814` / `server08` also
      started. It was initially preserved as spare tmux-held GPU capacity with
      keepalive; a second keepalive raised measured GPU utilization from about
      `35%` to about `95%`.
- [ ] 2026-06-21 20:34+08 `145814` was repurposed for an aligned formal
      comparison instead of leaving it on keepalive: same h96 union, same
      5-head handoff-success scorer, same gate-fixed offline rule, same
      `SEED=20260618`, but `SCORER_RANK_LOSS_WEIGHT=0.2`. The old keepalive
      Slurm step `145814.2` was cancelled without cancelling the allocation,
      and the active training step is `145814.5` in tmux
      `cosmos3_h96_handoff_success_retry_rank02_gatefix_train_145814_20260621`.
      Scorer root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_handoff_success_retry_rank02_gatefix_145814_20260621_2032_formal3h`.
      This is a comparison for selector generalization, not a changed task
      objective and not live evidence until its own 10800-second floor and
      margin eval pass.
- [ ] 2026-06-21 20:44+08 low-memory retry allocation `145818` / `server64`
      started. It is being preserved with keepalive only
      (`cosmos3_h96_handoff_success_retry_keepalive_145818_20260621`);
      latest measured utilization was about `92%`. Do not launch a third
      scorer variant from it unless the current rank0/rank0.2 formal runs
      expose a concrete new implementation need. Keep it available for the
      next aligned action after summary/margin results.
- [ ] 2026-06-21 21:28+08 interim scorer readout: neither rank0 nor rank0.2
      has a formal summary yet. Rank0 gatefix on `145813` has reached a best
      held-out handoff selection of `10/16` versus DP `8/16`, but latest
      points fluctuate back near `8-9/16`. Rank0.2 gatefix on `145814` has
      reached only `9/16` best and latest is below DP. Treat this as weak
      selector generalization despite real h96 oracle headroom; wait for the
      10800-second summaries and margin eval before any live rollout.
- [ ] 2026-06-20 02:14+08 completed the server54 shortprefix64-scorer live
      panel on allocation `142824`, output root
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_shortprefix81216_20260620_004432_shortprefix64scorer_margin000_dpcont32_h24_dphandoff32_server54_alloc142824_samples0_1_3_4`.
      It used Cosmos3 `iter_000001500`, the formal 24-step executor, the
      shortprefix64 outcome scorer `checkpoint_best_gate.pt`, scorer margin
      `0.0`, `CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS=8,12,16`,
      `ACTION_EXEC_HORIZON=24`, and `DP_HANDOFF_CHUNK_HORIZON=32`.
- [ ] Result: the panel finished all four requested samples with `0` process
      failures, passed the full `301/300` contract, wrote four 301-frame
      annotated videos plus `live_receding_panel_contact_sheet.png`, and
      final real-state success was `0/4`. Final peg-head-in-hole states were
      sample 0 `[-0.1092, 0.0236, -0.0690]`, sample 1
      `[-0.1153, -0.0069, -0.0320]`, sample 3
      `[-0.1391, 0.0118, 0.0031]`, and sample 4
      `[-0.1015, -0.0011, -0.0154]`. The contact sheet was opened and agrees:
      all samples end near the block/hole but the peg is not inserted.
      `method_evidence_allowed=false`.
- [ ] Interpretation after this panel: the closed-loop implementation is
      working mechanically on render-capable `server54`; the failure is not
      render, video length, one-shot evaluation, or lack of reobservation.
      Short-prefix candidates and repeated DP-prior selections still leave
      about `10-14cm` insertion-axis error, and even the sample 3 DP handoff
      (`91` executed DP handoff steps, `6` real C_pi gate passes) did not
      finish insertion. The current blocker is therefore candidate/executor
      coverage plus scorer/continuability calibration on the true contact
      manifold, not another scalar gate or enumerated recovery branch.
- [ ] 2026-06-20 02:27+08 stopped the oversized shortprefix128 `skip0`
      export on allocation `142106` and the oversized `skip128` export on
      allocation `142824`. Both were cleanly interrupted after their ETA made
      it clear they were too large to finish label export plus a formal
      1-GPU/3-hour scorer inside the held allocation windows. No final
      `candidate_outcome_labels.jsonl` or scorer came from those interrupted
      roots, so they are scheduling/throughput evidence only, not method
      evidence.
- [ ] 2026-06-20 02:28+08 launched feasible non-overlapping shortprefix128
      label-only split shards with a fresh claim root
      `experiments/world_model_task_rebinding/cosmos3/hardphase_retrieval_shard_claims_20260620_shortprefix128_split40_88`.
      Server04 allocation `142106` runs `HARD_MAX_ROWS=40,HARD_SKIP_ROWS=0`
      in tmux session
      `cosmos3_shortprefix128_split40_skip0_labels_142106_20260620`, log root
      `experiments/world_model_task_rebinding/cosmos3/shortprefix128_split40_skip0_labels_20260620_022855_shortprefix128_split40_skip0_dpcont32_min16_labels_alloc142106`,
      labels root
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval40_skip0_k4_20260620_022855_shortprefix128_split40_skip0_dpcont32_min16_labels_alloc142106`.
      Server54 allocation `142824` runs `HARD_MAX_ROWS=88,HARD_SKIP_ROWS=40`
      in tmux session
      `cosmos3_shortprefix128_split88_skip40_labels_142824_20260620`, log root
      `experiments/world_model_task_rebinding/cosmos3/shortprefix128_split88_skip40_labels_20260620_022855_shortprefix128_split88_skip40_dpcont32_min16_labels_alloc142824`,
      labels root
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval88_skip40_k4_20260620_022855_shortprefix128_split88_skip40_dpcont32_min16_labels_alloc142824`.
      Both use `CANDIDATE_SHORT_PREFIX_STEPS=8,12,16`,
      `DP_ROLLOUT_CONTINUABILITY_HORIZON=32`,
      `DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS=16`,
      `RETRIEVAL_K=4`, `RETRIEVAL_RESIDUAL_SCALES=0.5,1.0,1.5`, and
      `RUN_SCORER_AFTER_HEADROOM_GATE=false`.
- [ ] 2026-06-20 04:24+08 detected that allocation `142106` was revoked while
      the `HARD_MAX_ROWS=40,HARD_SKIP_ROWS=0` split was still partial. It left
      about `2962` candidate action files, but no final
      `candidate_outcome_labels.jsonl` and no `done.txt` claim. Treat that
      root as an interrupted partial artifact only. Do not include it in union
      training.
- [ ] 2026-06-20 04:29+08 acquired replacement 1-GPU allocation `143035` on
      `server04` and launched tmux session
      `cosmos3_shortprefix128_split10x4_labels_143035_20260620`. This reruns
      rows `0-39` as four sequential 10-row label-only shards:
      `HARD_MAX_ROWS=10,HARD_SKIP_ROWS=0`, `10`, `20`, and `30`, under the
      same claim root
      `experiments/world_model_task_rebinding/cosmos3/hardphase_retrieval_shard_claims_20260620_shortprefix128_split40_88`.
      Log root:
      `experiments/world_model_task_rebinding/cosmos3/shortprefix128_split10x4_labels_20260620_042902_shortprefix128_split10x4_dpcont32_min16_labels_alloc143035`.
      This is still the same shortprefix128 label objective; only the shard
      granularity changed so completed labels are not lost if another
      allocation is revoked.
- [ ] 2026-06-20 05:01+08 also stopped the long `HARD_MAX_ROWS=88,
      HARD_SKIP_ROWS=40` shard on allocation `142824` before it could suffer
      the same partial-output loss. It had reached about `16` started rows and
      about `4400` candidate files, but had no final
      `candidate_outcome_labels.jsonl`. It is interrupted partial output only
      and must not be used for union scorer training.
- [ ] 2026-06-20 05:02+08 launched replacement server54 tmux session
      `cosmos3_shortprefix128_split10x9_skip40_120_labels_142824_20260620`.
      It reruns rows `40-127` as small sequential label-only shards:
      `HARD_SKIP_ROWS=40,50,60,70,80,90,100,110` with
      `HARD_MAX_ROWS=10`, then `HARD_SKIP_ROWS=120` with
      `HARD_MAX_ROWS=8`. Log root:
      `experiments/world_model_task_rebinding/cosmos3/shortprefix128_split10x9_skip40_120_labels_20260620_050220_shortprefix128_split10x9_skip40_120_dpcont32_min16_labels_alloc142824`.
      This preserves the same shortprefix128 label objective while making
      progress land shard-by-shard.
- [ ] 2026-06-20 06:03+08 detected that allocation `142824` was also revoked.
      The replacement server54 `skip40` small shard had only partial candidate
      files and no final labels. It is not usable for union training.
- [ ] 2026-06-20 06:07+08 acquired replacement allocation `143146` on
      `server34` and relaunched rows `40-127` as the same small-shard chain in
      tmux session
      `cosmos3_shortprefix128_split10x9_skip40_120_labels_143146_20260620`.
      Log root:
      `experiments/world_model_task_rebinding/cosmos3/shortprefix128_split10x9_skip40_120_labels_20260620_060716_shortprefix128_split10x9_skip40_120_dpcont32_min16_labels_alloc143146`.
      The old `skip40` claim from `142824` was archived as orphaned by the
      claim wrapper before the `143146` replacement claimed the shard.
- [ ] 2026-06-20 06:07+08 first replacement shard success: `143035` completed
      the `HARD_MAX_ROWS=10,HARD_SKIP_ROWS=0` shard and wrote `done.txt`; the
      same tmux chain advanced to `HARD_SKIP_ROWS=10`.
- [ ] 2026-06-20 07:50+08 two more small shards completed: `143035` finished
      `HARD_SKIP_ROWS=10` with `2860` labels and advanced to `HARD_SKIP_ROWS=20`;
      `143146` finished `HARD_SKIP_ROWS=40` with `2860` labels and advanced
      to `HARD_SKIP_ROWS=50`. Effective completed replacement shards are now
      `skip0`, `skip10`, and `skip40`.
- [ ] 2026-06-20 08:51+08 detected that allocation `143035` was revoked while
      `HARD_SKIP_ROWS=20` was partial. Completed rows `0-19` remain valid;
      the partial `skip20` output from `143035` must not be used. Replacement
      allocation `143269` on `server18` was acquired and launched in tmux
      session `cosmos3_shortprefix128_split10x2_skip20_30_labels_143269_20260620`
      to rerun `HARD_SKIP_ROWS=20` and then `30`. Log root:
      `experiments/world_model_task_rebinding/cosmos3/shortprefix128_split10x2_skip20_30_labels_20260620_085414_shortprefix128_split10x2_skip20_30_dpcont32_min16_labels_alloc143269`.
      The claim wrapper archived the old `skip20` claim as orphaned before
      `143269` took over.
- [ ] 2026-06-20 09:30+08 `143146` completed `HARD_SKIP_ROWS=50` with
      `2860` labels and advanced to `HARD_SKIP_ROWS=60`. Effective completed
      replacement shards are now `skip0`, `skip10`, `skip40`, and `skip50`.
- [ ] 2026-06-20 10:31+08 `143269` completed `HARD_SKIP_ROWS=20` with
      `2860` labels and advanced to `HARD_SKIP_ROWS=30`. Allocation `143146`
      was revoked while `HARD_SKIP_ROWS=60` was partial; replacement
      allocation `143345` on `server18` launched tmux session
      `cosmos3_shortprefix128_split10x7_skip60_120_labels_143345_20260620`
      to rerun `HARD_SKIP_ROWS=60,70,80,90,100,110` and final
      `HARD_SKIP_ROWS=120,HARD_MAX_ROWS=8`. Log root:
      `experiments/world_model_task_rebinding/cosmos3/shortprefix128_split10x7_skip60_120_labels_20260620_103242_shortprefix128_split10x7_skip60_120_dpcont32_min16_labels_alloc143345`.
      Effective completed replacement shards are now `skip0`, `skip10`,
      `skip20`, `skip40`, and `skip50`.
- [ ] 2026-06-20 12:10+08 `skip30` and `skip60` completed with `2860` labels
      each. The held `143269` allocation was reused to launch
      `cosmos3_shortprefix128_split10x5_skip80_120_labels_143269_20260620`
      for `skip80,90,100,110,120` while `143345` continues `skip70`.
      Effective completed replacement shards are now `skip0`, `skip10`,
      `skip20`, `skip30`, `skip40`, `skip50`, and `skip60`.
- [ ] 2026-06-20 13:11+08 allocation `143269` was revoked while `skip80` was
      partial. Replacement allocation `143634` on `server18` launched
      `cosmos3_shortprefix128_split10x5_skip80_120_labels_143634_20260620`
      to rerun `skip80,90,100,110,120`; the claim wrapper archived the partial
      `skip80` claim from `143269` as orphaned. `143345` continues `skip70`.
- [ ] 2026-06-20 13:48+08 `skip70` completed with `2860` labels. `143634`
      continues `skip80`; `143345` skipped the already-claimed `skip80` and
      advanced to `skip90`. Effective completed replacement shards are now
      `skip0`, `skip10`, `skip20`, `skip30`, `skip40`, `skip50`, `skip60`,
      and `skip70`.
- [ ] 2026-06-20 14:36+08 allocation `143345` was cancelled while its
      replacement `skip90` shard was partial. `skip60` and `skip70` remain
      valid because they have final labels and `done.txt`; the partial
      `skip90` root from `143345` must not be used. Replacement allocation
      `143735` on `server56` was acquired at 14:40 and launched tmux session
      `cosmos3_shortprefix128_split10x4_skip90_120_labels_143735_20260620`
      for `skip90,100,110,120`. The claim wrapper archived the old `skip90`
      claim as orphaned before `143735` took it.
- [ ] 2026-06-20 14:52+08 `skip80` completed on allocation `143634` with
      `2860` labels. The same allocation skipped the already-claimed `skip90`
      and advanced to `skip100`. Effective completed replacement shards are
      now `skip0`, `skip10`, `skip20`, `skip30`, `skip40`, `skip50`,
      `skip60`, `skip70`, and `skip80`.
- [ ] 2026-06-20 15:39+08 allocation `143776` on `server59` launched tmux
      session
      `cosmos3_shortprefix128_split10x2_skip110_120_labels_143776_20260620`
      for the remaining tail. Current live split is `143735` running
      `skip90`, `143634` running `skip100`, and `143776` running `skip110`;
      `skip120` remains pending behind whichever live chain reaches it first.
- [ ] 2026-06-20 16:18+08 `skip90` completed on allocation `143735` with
      `2860` labels. `143735` skipped the already-claimed `skip100` and
      `skip110`, then claimed the final `skip120` shard.
- [ ] 2026-06-20 16:28+08 `skip100` completed on allocation `143634` with
      `2860` labels. Effective completed replacement shards are now `skip0`,
      `skip10`, `skip20`, `skip30`, `skip40`, `skip50`, `skip60`, `skip70`,
      `skip80`, `skip90`, and `skip100`. Remaining before union scorer:
      `skip110` and `skip120`.
- [ ] 2026-06-20 17:20+08 `skip110` completed on allocation `143776` with
      `2860` labels. 2026-06-20 17:35+08 `skip120` completed on allocation
      `143735` with `2288` labels. All 13 required replacement shards are now
      complete.
- [ ] 2026-06-20 17:35+08 launched the formal shortprefix128 small-shard
      union scorer on held allocation `143735` / `server56`, tmux session
      `cosmos3_shortprefix128_union_scorer_143735_20260620`. Summary root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_retrieval_claim_union_shortprefix128_smallshard_20260620_1735_shortprefix128_smallshard_union_formal3h_alloc143735`.
      Scorer root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval_claim_union_shortprefix128_smallshard_formal3h_20260620_1735_shortprefix128_smallshard_union_formal3h_alloc143735`.
      Manifest verified `num_done_claims=13`,
      `num_active_inprogress_claims=0`,
      `MIN_UNION_DONE_CLAIMS_FOR_SCORER=13`,
      `SCORER_MIN_WALL_SECONDS=10800`,
      `SCORER_MAX_WALL_SECONDS=10800`, and
      `SCORER_FORMAL_MIN_GPUS=1`.
- [ ] 2026-06-20 18:20+08 the scorer process was alive but GPU utilization
      snapshots were near zero, which risks the same low-utilization
      allocation release seen during label export. A temporary CUDA keepalive
      was launched inside the same allocation and tied to the scorer process;
      after tuning, the active keepalive is
      `cosmos3_shortprefix128_union_scorer_gpu_keepalive_sleep1p5_143735_20260620`.
      This is resource preservation only, not experiment evidence or a model
      component.
- [ ] 2026-06-20 20:44+08 the union wrapper completed. Scorer training wrote
      `training_summary.json`, `checkpoint_final.pt`, and
      `checkpoint_best_gate.pt` after `10800.59s`; the summary has
      `formal_training_floor_met=true`,
      `best_gate_ready_for_offline_gate=true`, and
      `ready_for_formal_live_eval=true`. Margin eval wrote
      `margin_eval_summary.json`; best eval margin is `0.02` with selected
      weighted error improvement over DP on the held-out scorer split.
- [ ] 2026-06-20 20:46+08 launched the live closed-loop panel on allocation
      `143735` / `server56`, tmux session
      `cosmos3_shortprefix128_union_live_panel_143735_20260620`, after a
      passing render canary. It uses the same samples/protocol as the prior
      shortprefix panels (`sample_indices=0,1,3,4`, `iter1500`,
      `ACTION_EXEC_HORIZON=24`, `DP_HANDOFF_CHUNK_HORIZON=32`,
      `CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS=8,12,16`), with the new
      shortprefix128 small-shard union scorer and `candidate_outcome_scorer_dp_margin=0.02`.
- [ ] 2026-06-20 22:17+08 the shortprefix128 small-shard union live panel
      completed. It had `completed_samples=4`, `failed_process_count=0`,
      `panel_full_episode_contract_ok=true`, and `final_success_count=0`.
      Final peg-head-at-hole states were sample 0 `[-0.0374, 0.0001, 0.0022]`,
      sample 1 `[-0.0826, 0.0004, 0.0056]`, sample 3
      `[-0.1308, -0.0051, -0.0327]`, and sample 4
      `[-0.1038, 0.0063, -0.0364]`. The contact sheet was opened and agrees:
      sample 0 is closer than the shortprefix64 panel but still outside the
      hole; the other three are clearly not inserted. `method_evidence_allowed=false`.
- [ ] Interpretation after the shortprefix128 panel: larger short-prefix
      outcome labels plus a formal 3-hour scorer improved offline selection
      and helped sample 0 approach the hole, but it still did not solve live
      insertion. The current blocker remains the action-candidate/executor
      distribution near contact: the selector can only choose among weak
      chunks, and DP handoff either does not trigger or starts too far from a
      true insertable manifold.
- [ ] 2026-06-20 22:26+08 launched an aligned follow-up smoke on the same held
      allocation `143735`, tmux session
      `cosmos3_outcome_oracle_shortprefix128_smoke_retry1_143735_20260620`.
      It trains an outcome-oracle candidate executor from the completed
      shortprefix128 union labels, then replays `128` hard-phase states with
      short-prefix candidates and 32-step DP-continuability labels. This is
      only a candidate-generator/headroom smoke, not formal method evidence.
      If it shows stronger hard-phase candidate headroom, the next aligned
      step is a formal 1-GPU/3-hour candidate-generator training run followed
      by the same gated live panel; if it does not, stop treating scorer
      changes as the repair.
- [ ] Active next aligned work: monitor the outcome-oracle shortprefix128
      smoke, inspect its training summary and replay headroom. Do not launch a
      formal run unless the smoke shows that the candidate pool itself became
      stronger than the current retrieval/DP-prior pool.
- [ ] 2026-06-20 23:28+08 update: the first outcome-oracle smoke did not
      justify formal training or live eval. The 200-step training run wrote
      `training_summary.json` under
      `experiments/world_model_task_rebinding/cosmos3/outcome_oracle_candidate_executor_shortprefix128_union_smoke200_20260620_2229_retry1_alloc143735`.
      It is a short smoke only (`formal_training_floor_met=false`), and the
      selected eval action MSE was worse than the DP prior
      (`selected_minus_dp_prior_mse=+0.000094` final,
      `+0.000464` at best-offline).
- [ ] The matching `16`-row replay completed under
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_shortprefix128_oracle_smoke200_replay16_20260620_2230_alloc143735`.
      Overall it had `8/16` DP successes and `9/16` oracle successes, with
      mean oracle-minus-DP error `-0.029456`. The important split is the
      blocker: `dp_continuable` had `9/10` oracle successes, but
      `lateral_align` had `0/6` DP successes and `0/6` oracle successes even
      though error improved by about `0.061`. Plain meaning: the new
      checkpoint candidates mostly help already-near states; they still do
      not contain successful insertions for the hard lateral-alignment phase.
- [ ] 2026-06-20 23:34+08 launched a slower replay-only diagnostic on
      allocation `143735`, tmux session
      `cosmos3_shortprefix128_oracle_retrieval_replay64b_143735_20260620`.
      It reuses the same smoke checkpoint but also adds retrieval residual
      candidates (`retrieval_k=4`, scales `0.5,1.0,1.5`) with short prefixes
      `8,12,16` and the 32-step DP-continuability label. This is not formal
      training and not live method evidence. Its only question is whether the
      candidate pool contains successful hard-phase chunks once retrieval
      residuals are included. If `lateral_align/far` still have no success
      candidates, the next repair is a stronger general candidate/teacher
      source, not another scorer, threshold, or live panel.
- [ ] 2026-06-21 06:58+08 replay64b completed under
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_shortprefix128_oracle_smoke200_retrieval_replay64b_20260620_2348_shortprefix128_oracle_retrieval_replay64b_alloc143735`
      with `11904` candidate labels and no row-level export failure recorded.
      Headroom summary root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_shortprefix128_oracle_smoke200_retrieval_replay64b_20260620_2348_shortprefix128_oracle_retrieval_replay64b_alloc143735`.
      Overall, DP had `25/64` successes, oracle candidate selection had
      `30/64`, and mean oracle-minus-DP weighted error was `-0.054874`.
      The split that matters: `far` stayed `0/8` success candidates,
      `lateral_align` had only `3/21`, and `preinsert_aligned` had only
      `1/8`. Retrieval residuals improve distance error and add sparse hard
      successes, but they do not solve the hard contact-phase candidate
      coverage problem.
- [ ] 2026-06-21 07:00+08 a short retrieval64b scorer smoke completed under
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_shortprefix128_oracle_retrieval_replay64b_smoke500_20260621_0655_retrieval64b_scorer_smoke500_alloc143735`.
      It ran only `500` steps, so `formal_training_floor_met=false` and
      `ready_for_formal_live_eval=false`. The best gate checkpoint at step
      `150` passed the offline aggregate smoke but selected only a small part
      of the oracle headroom: held-out oracle-minus-DP weighted error was
      `-0.046980`, while selected-minus-DP was only `-0.010650` and
      top-1 oracle match was `0.0625`. Do not promote this checkpoint to
      formal live eval. It is useful only as evidence that the selector is
      weaker than the available oracle and that the candidate pool itself is
      still sparse in hard phases.
- [ ] Active next aligned work after replay64b: do not spend the held GPU on
      another margin-only or live-panel attempt from this smoke. The current
      blocker is the general hard-phase action source: `far`,
      `lateral_align`, and `preinsert_aligned` need candidate chunks that can
      actually enter a DP-continuable insertion manifold. The next useful
      compute should test a stronger general candidate/teacher source or a
      phase/contact-conditioned action generator, then measure headroom before
      any formal 3-hour scorer/live run.
- [ ] 2026-06-21 07:04+08 added handoff-aware diagnostic fields to
      `scripts/world_model/summarize_cosmos3_candidate_outcome_headroom.py`,
      `scripts/world_model/train_cosmos3_candidate_outcome_scorer.py`, and
      `scripts/world_model/eval_cosmos3_candidate_outcome_scorer_margins.py`.
      This preserves the old candidate-final-success/error metrics and only
      adds the missing method-relevant question: after a candidate chunk, can
      frozen DP finish or keep the state continuable?
- [ ] The recomputed handoff-aware replay64b summary is
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_shortprefix128_oracle_smoke200_retrieval_replay64b_handoffaware_20260621_070416_alloc143735`.
      It changes the readout: overall DP handoff success is `35/64` and
      handoff-oracle is `43/64`, compared with the old candidate-final oracle
      success `30/64`. Phase split still exposes the blocker:
      `far` handoff-oracle `2/8`, `lateral_align` `11/21`,
      `preinsert_aligned` `3/8`, and `dp_continuable` `27/27`.
      Plain meaning: the old summary underreported DP-handoff value, but hard
      phases are still not covered enough.
- [ ] 2026-06-21 07:07+08 reran margin eval for the existing retrieval64b
      scorer with the new handoff metrics:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_shortprefix128_oracle_retrieval_replay64b_smoke500_handoffaware_margin_eval_20260621_070711_alloc143735`.
      On its held-out split, DP handoff success is `6/16`,
      handoff-oracle is `9/16`, but the scorer selects only `5/16` at the
      best-error margin; gate-passing margins select `5/16` or `6/16`.
      Therefore the current scorer is not just weak; it can reduce real
      handoff success while improving coordinate error.
- [ ] 2026-06-21 07:09+08 ran a short handoff-rank scorer smoke:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_shortprefix128_oracle_retrieval_replay64b_handoffrank_smoke500_20260621_070835_alloc143735`.
      It is diagnostic only (`formal_training_floor_met=false`,
      `ready_for_formal_live_eval=false`) and did not fix selection. On its
      held-out split, DP handoff success is `10/16`, handoff-oracle `12/16`;
      best-offline selected `10/16` but with worse weighted error, and final
      selected only `6/16`. Do not run formal/live from this direction without
      a stronger selector objective and better candidate coverage.
- [ ] 2026-06-21 07:12+08 launched a focused compute-node probe in tmux
      `cosmos3_far_dp96_handoff_probe_143735_20260621` on held allocation
      `143735`. It replays only `8` `far` rows with the same short-prefix
      retrieval/model candidates but extends the label-only frozen-DP rollout
      from `32` to `96` steps. Roots:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_far8_shortprefix128_oracle_retrieval_dp96_20260621_0712_alloc143735`
      and
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_far8_shortprefix128_oracle_retrieval_dp96_20260621_0712_alloc143735`.
      Purpose: decide whether `far` is mostly a too-short handoff-label
      problem or a true candidate-source problem.
- [ ] 2026-06-21 08:33+08 the `far` dp96 probe completed with `912`
      candidate records and no method-evidence claim. Candidate-final
      insertion success stayed `0/8`, so the candidates still do not solve
      insertion by themselves. But the method-relevant handoff readout changed
      sharply: DP handoff baseline is `4/8`, handoff-oracle is `6/8`, versus
      only `2/8` handoff-oracle for the same `far` slice under the earlier
      32-step handoff label. Mean oracle-minus-DP error is `-0.090065`.
      Plain conclusion: for `far`, the 32-step handoff label was too short
      and undercounted useful candidates; the right next target is a
      handoff-aware 96-step label/selector check, not another coordinate-only
      scorer or a live panel from the old h32 scorer.
- [ ] Active next aligned work after the `far` dp96 result: run the same
      96-step handoff-label diagnostic on `lateral_align` and
      `preinsert_aligned` before spending a formal 3-hour scorer/live run.
      If those phases also gain handoff-oracle headroom, rebuild the scorer
      around h96 handoff success/continuability and require held-out
      handoff-success selection before live eval. If they do not, the
      remaining blocker is phase-specific candidate coverage, especially
      candidate chunks that enter a real DP-continuable contact manifold.
- [ ] 2026-06-21 08:40+08 launched the first follow-up phase diagnostic,
      `cosmos3_lateral8_dp96_handoff_probe_143735_20260621`, on the same
      tmux-held allocation `143735`. It uses `8` `lateral_align` rows, the
      same short-prefix/model/retrieval candidate families as the `far` probe,
      and `DP_ROLLOUT_CONTINUABILITY_HORIZON=96`,
      `DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS=16`. Roots:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_lateral8_shortprefix128_oracle_retrieval_dp96_20260621_0840_alloc143735`
      and
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_lateral8_shortprefix128_oracle_retrieval_dp96_20260621_0840_alloc143735`.
      A separate CUDA keepalive
      `cosmos3_lateral8_dp96_gpu_keepalive_143735_20260621` is tied to the
      lateral summary file and is only for preserving the allocation.
- [ ] 2026-06-21 10:04+08 the `lateral_align` dp96 probe completed with
      `912` candidate records. Candidate-final success was only `1/8`, but
      DP handoff baseline was `4/8` and handoff-oracle was `8/8`; mean
      oracle-minus-DP error was `-0.056087`. Handoff-success candidates came
      from retrieval residuals in all `8/8` groups, checkpoint-model
      candidates in `7/8`, and teacher/legacy families in `6/8`. Plain
      conclusion: for `lateral_align`, useful candidates exist when the label
      asks whether DP can finish after a longer handoff. This reinforces that
      the next scorer target must be handoff success/continuability, not
      candidate-final insertion alone.
- [ ] 2026-06-21 10:06+08 launched the matching
      `preinsert_aligned` dp96 diagnostic in tmux
      `cosmos3_preinsert8_dp96_handoff_probe_143735_20260621` on allocation
      `143735`, plus summary-tied keepalive
      `cosmos3_preinsert8_dp96_gpu_keepalive_143735_20260621`. Roots:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_preinsert8_shortprefix128_oracle_retrieval_dp96_20260621_1005_alloc143735`
      and
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_preinsert8_shortprefix128_oracle_retrieval_dp96_20260621_1005_alloc143735`.
- [ ] 2026-06-21 10:51+08 allocation `143735` was revoked while the first
      `preinsert_aligned` dp96 probe was partial. The log reports
      `STEP 143735.148 ON server56 CANCELLED AT 2026-06-21T10:51:36`; only
      `466/912` candidate action files existed and there was no final
      `candidate_outcome_labels.jsonl` or summary. Treat
      `...preinsert8_shortprefix128_oracle_retrieval_dp96_20260621_1005_alloc143735`
      as interrupted scheduling evidence only, not a data result.
- [ ] 2026-06-21 11:09+08 acquired replacement 1-GPU/1-day tmux-held
      allocation `145276` on `server63` via
      `cosmos3_preinsert_dp96_realloc_1gpu_request_20260621`. A temporary
      2-GPU pending fallback `145278` was submitted and then cancelled after
      `145276` started, so no extra held allocation was wasted. 2026-06-21
      11:10+08 launched clean rerun
      `cosmos3_preinsert8_dp96_rerun_145276_20260621`, with keepalive
      `cosmos3_preinsert8_dp96_rerun_keepalive_145276_20260621`. Roots:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_preinsert8_shortprefix128_oracle_retrieval_dp96_20260621_1110_alloc145276`
      and
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_preinsert8_shortprefix128_oracle_retrieval_dp96_20260621_1110_alloc145276`.
- [ ] 2026-06-21 12:40+08 the replacement `preinsert_aligned` dp96 probe
      completed with `912` candidate records. Candidate-final success was
      only `1/8`, DP handoff baseline was `1/8`, and handoff-oracle was
      `8/8`; mean oracle-minus-DP error was `-0.099655`. The earlier h32
      handoff-aware summary for the same `preinsert_aligned` phase had only
      `3/8` handoff-oracle. Plain conclusion: all three hard phases tested
      now support the same diagnosis. The current blocker is no longer
      "candidate source has no useful action"; it is that h32/coordinate-only
      scorer training underlabels the useful candidate+handoff behavior.
- [ ] 2026-06-21 12:42+08 added
      `scripts/slurm/run_cosmos3_hardphase_h96_sharded_replay_in_allocation.sh`
      and launched `cosmos3_h96_shards64_145276_20260621` on held allocation
      `145276`. This generates a larger h96 hard-phase replay set as `8`
      small completed shards of `8` rows each, using the same candidate
      families and a shared claim root
      `experiments/world_model_task_rebinding/cosmos3/hardphase_h96_retrieval_shard_claims_20260621_1242_h96shard64_alloc145276`.
      Final union root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_h96_retrieval_claim_union_20260621_1242_h96shard64_alloc145276`.
      A keepalive `cosmos3_h96_shards64_keepalive_145276_20260621` is tied to
      that final union summary. Do not train a formal scorer until this h96
      union exists and held-out handoff-aware headroom is inspected.
- [ ] 2026-06-21 14:13+08 the first h96 shard, `HARD_SKIP_ROWS=0`, completed
      and wrote a `done.txt` claim. The chain automatically advanced to
      `HARD_SKIP_ROWS=8`. This confirms the small-shard setup is preserving
      completed work if the allocation is interrupted later.
- [ ] 2026-06-21 15:11+08 allocation `145276` was revoked while the h96 chain
      was inside `HARD_SKIP_ROWS=8`. The completed `skip0` shard is valid and
      preserved by claim `done.txt`; `skip8` had only partial candidate action
      files and no final labels, so it must be rerun. 2026-06-21 15:18+08
      opened tmux-held replacement requests
      `cosmos3_h96_resume_1gpu_request_20260621_1518`,
      `cosmos3_h96_resume_2gpu_request_20260621_1518`, and
      `cosmos3_h96_resume_4gpu_request_20260621_1518`. Use the first valid
      allocation that starts, cancel the remaining pending requests, and
      resume with the same shard claim root so `skip0` is skipped and the
      orphaned `skip8` is rerun.
- [ ] 2026-06-21 15:55+08 all three fallback allocations started:
      `145548` 1-GPU on `server63`, `145549` 2-GPU on `server63`, and
      `145550` 4-GPU on `server04`. They were immediately repurposed rather
      than released: `skip8` runs on `145548`; `skip16` and `skip24` run on
      `145549`; `skip32`, `skip40`, `skip48`, and `skip56` run on `145550`.
      All seven workers share the original h96 claim root, so completed
      `skip0` is preserved and duplicate claims are avoided. 2026-06-21
      15:58+08 added claim-count keepalives
      `cosmos3_h96_resume_claim_keepalive_145548_20260621`,
      `cosmos3_h96_resume_claim_keepalive_145549_20260621`, and
      `cosmos3_h96_resume_claim_keepalive_145550_20260621`; instantaneous GPU
      utilization after keepalive was about `90%` on the 1-GPU allocation,
      `69/59%` on the 2-GPU allocation, and `36/77/68/66%` on the 4-GPU
      allocation.
- [ ] 2026-06-21 17:20+08 the h96 64-row union summary completed after all
      eight shards reached `done.txt`. Union root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_hardphase_h96_retrieval_claim_union_20260621_1242_h96shard64_alloc145276`.
      Overall DP handoff success is `35/64`, handoff-oracle success is
      `60/64`, and candidate-final oracle success is only `6/64`. By phase,
      handoff-oracle is `far 12/16`, `lateral_align 31/31`, and
      `preinsert_aligned 17/17`; DP handoff baseline is `far 8/16`,
      `lateral_align 18/31`, and `preinsert_aligned 9/17`. The progress
      headroom gate passed; terminal-success gate did not. Plain conclusion:
      h96 confirms the DDP-style unit is candidate chunk plus handoff, not
      candidate-final insertion.
- [ ] 2026-06-21 17:44+08 launched two formal 3-hour h96 scorer trainings:
      main `rank0` in tmux `cosmos3_h96_handoff_rank0_formal_145550_20260621`
      on allocation `145550`, root
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_retrieval_rank0_formal3h_20260621_1720_alloc145550`;
      and comparison `rank0.2` in tmux
      `cosmos3_h96_handoff_rank02_formal_145549_20260621` on allocation
      `145549`, root
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_h96_retrieval_rank02_formal3h_20260621_1720_alloc145549`.
      Both use `SCORER_MIN_WALL_SECONDS=10800`,
      `SCORER_MAX_WALL_SECONDS=10800`, `SCORER_BINARY_LOSS_WEIGHT=1.0`,
      `SCORER_SCORE_SUCCESS_WEIGHT=2.0`, and
      `SCORER_SCORE_CONTINUABLE_WEIGHT=3.0`. Keepalives were strengthened at
      17:47+08; instantaneous utilization after the stronger keepalives was
      about `32-39%` on `145550`, `31/47%` on `145549`, and `32%` on idle
      held allocation `145548`.
- [ ] 2026-06-19 12:45+08 completed the reduced128 h32/min16 formal-scorer
      live panel on allocation `140738` / `server54`, output root
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260619_1127_dpcont32_reduced128_margin003_h24_dphandoff32_server54_alloc140738_samples0_1_3_4`.
      It used Cosmos3 `iter_000001500`, the formal 24-step executor, the
      reduced128 h32/min16 scorer `checkpoint_best_gate.pt`, scorer margin
      `0.03`, `ACTION_EXEC_HORIZON=24`, and
      `DP_HANDOFF_CHUNK_HORIZON=32`.
- [ ] Result: the panel finished all four samples with `0` process failures,
      passed the full `301/300` contract, wrote a contact sheet, and final
      real-state success stayed `1/4`. Sample 0 succeeded at
      `[-0.0058, 0.0002, 0.0011]` after `97` DP handoff steps. Samples 1, 3,
      and 4 failed at `[-0.0943, 0.0084, -0.0653]`,
      `[-0.1174, 0.0013, -0.0002]`, and
      `[-0.0983, -0.0044, -0.0029]`. The contact sheet was opened and agrees:
      sample 0 is visually inserted; the others are near the block/hole but
      not completed insertions. `method_evidence_allowed=false`.
- [ ] Interpretation after this panel: the reduced128 h32/min16 scorer is a
      valid offline improvement over DP, and the runtime handoff duration now
      matches the 32-step label, but this still does not fix live control. The
      remaining blocker is not rendering, SFT startup, length truncation, or
      lack of closed-loop execution. The blocker is action generation and
      selection near contact: candidate chunks still often leave about
      `9-12cm` insertion-axis error, and DP handoff can only finish when the
      executor has already reached a truly insertable state.
- [ ] Next aligned step: stop treating more scalar scorer penalties as the
      main fix. The next useful experiment should improve candidate coverage
      and labels around the real near-contact manifold: shorter executable
      chunks or hold/reobserve candidates for uncertain states, plus live-like
      handoff-survival negatives where DP handoff from the candidate end state
      fails to stay insertable or finish. These failures are hard negatives for
      a general continuability/executor objective, not enumerated recovery
      cases.
- [ ] 2026-06-19 12:55+08 implemented and launched the first short-prefix
      candidate diagnostic on allocation `140738`, tmux session
      `cosmos3_live_shortprefix81216_reduced128_handoff32_140738_20260619`.
      The code change is default-off and adds
      `--candidate-executor-short-prefix-steps`; with `8,12,16`, the
      candidate executor scores a 24-step candidate-prefix plus DP-prior suffix
      but executes only the chosen short prefix before reobserving. This tests
      the general DDP-style short-chunk hypothesis, not a sample-specific
      recovery rule. Output root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_shortprefix81216_20260619_1250_shortprefix81216_dpcont32_reduced128_margin003_h24_dphandoff32_server54_alloc140738_samples0_1_3_4`.
      Log:
      `experiments/world_model_task_rebinding/cosmos3/live_shortprefix81216_reduced128_handoff32_20260619_1250_shortprefix81216_alloc140738_tmux.log`.
      Compute-node checks passed before launch: `py_compile`, wrapper
      `bash -n`, `--help`, helper descriptor check, and a targeted dummy
      candidate-function check that selected a `short12_*` candidate and
      returned a 12-step action chunk.
- [ ] Short-prefix diagnostic result: completed all four samples with `0`
      process failures and full `301/300` contract, but final success stayed
      `1/4`. Sample 0 succeeded at `[-0.0079, -0.0008, 0.0017]`; samples 1,
      3, and 4 failed at `[-0.0951, 0.0142, -0.0607]`,
      `[-0.1291, -0.0029, -0.0474]`, and
      `[-0.1007, 0.0119, -0.0540]`. The contact sheet was opened and agrees:
      only sample 0 is visually inserted. The diagnostic proves that denser
      reobservation alone is not the missing piece. Short candidates were
      actually selected, but the selected chunks still left the peg about
      `9-13cm` wrong along the insertion axis or damaged z/contact.
- [ ] Updated blocker after short-prefix diagnostic: the live failure is now
      narrowed further. It is not a full-episode length bug, not Cosmos
      startup, not lack of receding observations, and not simply executing too
      many steps before reobserving. The core blocker is that the candidate
      action distribution and outcome scorer are not trained/calibrated on the
      real near-contact insertion manifold. Next aligned repair is to add
      short-prefix/hold-reobserve candidates to the offline outcome-label
      export and train the scorer on their real handoff-survival outcomes,
      including these live failures as hard negatives.
- [ ] 2026-06-19 05:04+08 completed the short 100-step h32/min16 reduced64
      scorer sanity on allocation `140738` / `server54`. Input labels:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval64_skip0_k4_20260619_0312_dpcont32_min16_reduced64_alloc140738`.
      Output scorer:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval64_skip0_k4_dpcont32_min16_reduced64_smoke100_20260619_050328_alloc140738`.
      Margin eval:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval64_skip0_k4_dpcont32_min16_reduced64_margin_20260619_050328_alloc140738`.
      This was only a sanity run, not formal training evidence.
- [ ] Reduced64 scorer sanity result: the train split can learn useful
      non-DP selections, but held-out eval fails. Final train selected-minus-DP
      weighted error was `-0.0355` with progress delta `+0.0623`, while final
      eval selected-minus-DP weighted error was `+0.0084` with progress delta
      `-0.0235`. Conservative margin eval had `0` gate-passing eval margins;
      margin `0.2` safely selected only DP and gave no improvement. Plain
      interpretation: the h32 label path runs, but `64` groups are too small
      and too sparse to train a usable selector.
- [ ] 2026-06-19 05:05+08 launched the next aligned larger run on the same
      tmux-held allocation `140738`, session
      `cosmos3_dpcont32_reduced128_formal_140738_20260619`, log
      `experiments/world_model_task_rebinding/cosmos3/dpcont32_reduced128_formal_20260619_0510_alloc140738_tmux.log`.
      It exports `128` hard groups with reduced candidates, retrieval `k=4`,
      `DP_ROLLOUT_CONTINUABILITY_HORIZON=32`,
      `DP_ROLLOUT_CONTINUABILITY_MIN_STABLE_STEPS=16`, then runs a
      1-GPU/3-hour outcome scorer if the headroom gate passes. This is the
      current highest-priority running chain because the 64-row sanity showed
      the short h32 labels need more held-out coverage before live eval.
- [ ] 2026-06-19 08:23+08 reduced128 h32/min16 label export completed with
      `4736` candidate-outcome rows, `0` failed rows,
      `438` DP-rollout-continuable candidates, and `401` DP-rollout successes.
      Headroom gate passed both terminal and progress gates: `7/128` DP
      success groups, `13/128` oracle-success groups, mean oracle-minus-DP
      weighted error `-0.0373`, `74/128` meaningful improvements, and
      `44/128` large improvements. The formal 1-GPU/3-hour scorer has started
      under
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_hardphase_retrieval128_skip0_k4_rank1_canddesc_smoke500_20260619_0510_dpcont32_min16_reduced128_formal_alloc140738`.
      A `checkpoint_best_gate.pt` already exists, but it is not live-eligible
      until the scorer reaches the `10800` second formal floor and writes a
      final summary with `ready_for_formal_live_eval=true`.
- [ ] 2026-06-19 03:08+08 completed a fast DP-continuability label
      diagnostic on allocation `140738` / `server54` comparing the old
      `8`-step DP handoff label with a longer `32`-step handoff label on the
      same 12 hard-phase groups and 444 candidate records. The earlier
      64-row / full-candidate attempt was stopped because it was only a label
      diagnostic and would have burned hours at low GPU utilization before
      producing a summary.
- [ ] Diagnostic result: the old `8`-step label is not too loose. It is too
      short. With `horizon=8,min_stable=4`, only `11/444` candidates were
      DP-continuable and `10/444` succeeded during the DP label rollout. With
      `horizon=32,min_stable=16`, `31/444` were DP-continuable and `21/444`
      succeeded. The shared-candidate comparison had `21` candidates that were
      negative under 8-step labels but positive under 32-step labels, mostly
      because DP needed more than eight steps to close the insertion-axis gap.
- [ ] Updated blocker after the label diagnostic: the live failure is not
      explained by a simply over-permissive offline continuability label.
      The more likely failure is that restored-state labels and live
      closed-loop handoff are still mismatched, and the selector/candidate set
      does not reliably drive live states into the longer-horizon DP-finishable
      manifold. The next aligned experiment is to generate a reduced-candidate
      64-row `horizon=32,min_stable=16` label shard, then use it as a debug
      gate for retraining/calibrating the scorer before any larger formal
      run.
- [ ] Implementation note from the same diagnosis: live `dp_handoff_horizon=32`
      was previously capped by `action_exec_horizon=24`, so DP handoff chunks
      did not actually match the 32-step label horizon. Added optional
      `dp_handoff_chunk_horizon` plumbing through the live loop, panel, and
      Slurm wrapper; default `0` preserves old behavior. Compute-node
      `py_compile`, wrapper `bash -n`, and `--help` checks passed on
      allocation `140738`. The next live diagnostic may set
      `DP_HANDOFF_CHUNK_HORIZON=32` without lengthening executor chunks.
- [ ] 2026-06-19 02:25+08 completed the state-axis calibrated union5 live
      panel on allocation `140738` / `server54`, output root
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260619_0102_statex010_dpcont_union5_margin002_h24_server54_alloc140738_samples0_1_3_4`.
      It used the same formal executor and five-shard DP-rollout scorer as
      the previous panel, with selector penalty
      `CANDIDATE_OUTCOME_SCORER_SCORE_STATE_ABS_AXIS_WEIGHTS=0.1,0,0`.
      The run finished all four samples, had `0` process failures, passed the
      full `301/300` contract, and the contact sheet was opened. Final
      success stayed `1/4`: sample 3 succeeded at
      `[0.0071, 0.0013, 0.0026]` after `59` DP handoff steps; samples 0, 1,
      and 4 failed at `[-0.0998, -0.0127, -0.0642]`,
      `[-0.0936, -0.0181, -0.0203]`, and
      `[-0.1052, 0.0015, -0.0049]`. `method_evidence_allowed=false`.
- [ ] 2026-06-19 02:25+08 state-axis calibration result: a strong
      state-error penalty (`1.0,0.25,0.25`) collapsed offline selection toward
      DP and failed the gate. A lighter eval-only sweep showed
      `0.1,0,0` preserved the offline gate best, but the live panel proved it
      does not repair the real blocker. Penalizing predicted x error is useful
      as a diagnostic, not as the next method fix.
- [ ] Current blocker after state-axis panel: the failures are still
      insertion-axis/contact-manifold failures, not render, data, SFT length,
      or lack of closed-loop plumbing. Samples 0, 1, and 4 remain about
      `9-10cm` short along the insertion axis. Sample 3 still works, which
      means the DDP-style mechanism can succeed when the executor reaches a
      true DP-continuable state, but the selector/scorer still cannot reliably
      distinguish "near the block" from "physically insertable".
- [ ] Next aligned fix after state-axis panel: stop adding scalar penalties.
      Train/calibrate against live-like handoff survival and improve candidate
      coverage. A state should be counted as continuable only if a short
      causal DP handoff from the real/restored state stays inside the insertion
      manifold or finishes. The executor/selector also needs shorter
      chunk/hold-reobserve candidates for uncertain near-contact states, so it
      does not keep committing a full 24-step chunk that leaves x unresolved.
- [ ] 2026-06-19 00:20+08 completed the five-shard
      DP-rollout-continuability scorer live panel on allocation `140738` /
      `server54`. Output root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260618_2258_dpcont_union5_margin002_h24_server54_alloc140738_server54_samples0_1_3_4`.
      The run finished all four samples with no process failures, passed the
      full `301/300` contract, and the contact sheet was opened. Final live
      success is still only `1/4`. Sample 3 `hole_late_fast_shift` succeeded
      from real simulator state at `[0.0080, 0.0015, 0.0028]` after `69`
      DP handoff steps. Samples 0, 1, and 4 failed at
      `[-0.0963, 0.0040, -0.0612]`,
      `[-0.0915, -0.0357, -0.0733]`, and
      `[-0.1052, 0.0088, -0.0089]`. `method_evidence_allowed=false`;
      this is partial live evidence, not final method success.
- [ ] 2026-06-19 00:20+08 recorded the important new blocker. The
      DP-rollout labels and five-shard scorer improved offline selection, but
      they did not improve live panel success over the earlier `1/4` result.
      The scorer still overestimates DP-continuability for near-hole but
      non-insertable states: sample 1 and sample 4 reached high predicted
      continuability while the live gate never accepted handoff, and sample 3
      needed repeated receding plus DP handoff before succeeding. The current
      problem is therefore live near-contact/continuability calibration, not
      data generation, render, SFT startup, or length accounting.
- [ ] Next aligned fix after the union5 live panel: do not add enumerated
      recovery cases. Calibrate the selector against the same real handoff
      condition it must satisfy at runtime: penalize high-confidence
      continuability when x/insertion-axis error is still large, train on
      short live-like DP-handoff survival/failure labels rather than only
      restored candidate-end labels, and add an adaptive short-chunk or
      hold/re-observe fallback for states where all full 24-step candidates
      are likely to push the peg out of the insertion manifold.
- [ ] 2026-06-18 late completed both formal DP-rollout-continuability outcome
      scorers. The two-shard scorer met the 1-GPU/3-hour floor, used
      `checkpoint_best_gate.pt` from step `275`, and passed margin eval at
      margin `0.02` with selected-minus-DP weighted error `-0.014522`. The
      five-shard scorer met the 1-GPU/3-hour floor, used
      `checkpoint_best_gate.pt` from step `150`, and passed margin eval at
      margin `0.02` with selected-minus-DP weighted error `-0.010274`. Final
      checkpoints overfit relative to best-gate; only the best-gate
      checkpoints should be used for live eval.
- [ ] 2026-06-18 16:25+08 completed the first formal live panel using the
      retrieval-union outcome scorer on `server54` allocation `140738`.
      Output root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260618_1510_scorer_margin003_h24_server54_defaulticd_alloc140738_server54_samples0_1_3_4`.
      The panel finished all four requested samples with no process failures
      and the `301/300` contract passed, but final success was only `1/4`.
      Sample 3 `hole_late_fast_shift` succeeded from real simulator state
      (`[0.0060, 0.0030, 0.0030]` peg-head-in-hole frame) after executor
      chunks entered the handoff region and DP completed `97` handoff steps.
      The contact sheet was opened and agrees that this is a real partial
      success, not just a metric artifact. This is partial live evidence for
      the DDP-style executor/scorer direction, not final method success:
      `method_evidence_allowed=false` and the panel is only `1/4`.
- [ ] 2026-06-18 16:25+08 recorded the current blocker from the same live
      panel. The method can approach the moved task frame, but it cannot
      robustly hold the millimeter-level z/contact/insertion condition. Sample
      0 ended with y aligned but z off by `6.6cm`. Sample 1 briefly passed
      handoff at prefix `228`, but DP drifted from z about `0.0032m` to
      `0.0112m` and failed. Sample 4 entered handoff for `72` DP steps but
      still ended at x about `-9.2cm`, so current `C_pi` pass is not a
      reliable short-horizon continuability proof. This is a controller/scorer
      calibration failure, not a render, length, or SFT-startup failure.
- [ ] Next aligned fix: do not add enumerated rescue cases. Add a general
      contact-stability repair to the selector/scorer: non-DP candidates need
      a conservative validity check for positive progress/contact confidence
      before overriding DP; the outcome scorer needs explicit z/contact-stable
      targets; and continuability labels should come from short causal DP
      rollout survival/finish evidence, not only one instantaneous real-state
      threshold. Current live failures may be used only as hard negative
      calibration data for this general objective, not as positive
      error-recovery demonstrations.
- [ ] 2026-06-18 17:55+08 completed the follow-up simple outcome-filter live
      diagnostic on the same four samples:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260618_1630_scorer_margin003_h24_progress_nonneg_cont005_server54_defaulticd_alloc140738_unknown_node_samples0_1_3_4`.
      It finished all four samples with no process failures and passed the
      `301/300` contract, but final success fell to `0/4`. The contact sheet
      was opened and agrees that no sample completed insertion. The important
      negative result is sample 3: the first panel solved it, but this simple
      filter made it fail at final peg-head-in-hole
      `[-0.1344, -0.0061, 0.0022]`. Therefore do not promote
      `min_progress_delta=0.0` plus `min_continuable_prob=0.05` into the
      formal live controller. The optional filter remains a diagnostic switch
      with defaults off.
- [ ] Updated next fix after the 0/4 filter diagnostic: the real repair is
      not another scalar threshold. Add contact-stability and insertion-axis
      labels to the outcome scorer, replace instantaneous handoff labels with
      short DP-rollout continuability labels, and add a safe/adaptive executor
      fallback for low-confidence states so the controller can execute a
      shorter chunk or hold/re-observe instead of forcing DP or a full 24-step
      non-DP chunk when all candidates look harmful. Current failures are hard
      negatives for this general calibration problem, not positive
      sample-specific recovery demos.
- [ ] 2026-06-18 18:08+08 implemented the first part of that repair. The
      candidate outcome exporter can now optionally run a label-only short
      frozen-DP rollout after each candidate chunk and records
      `final_contact_stable_proxy`, `dp_rollout_continuable_proxy`,
      `dp_rollout_success`, and `dp_rollout_stable_step_count`. The outcome
      scorer trainer now prefers `dp_rollout_continuable_proxy` over the old
      single-frame `final_continuable_proxy` when the new label exists. The
      retrieval replay wrappers pass these settings and include them in shard
      claim keys so old labels are not silently mixed with new labels.
- [ ] 2026-06-18 18:08+08 compute-node verification on allocation `140738`
      passed for the edited path: Python `py_compile`, shell `bash -n`, and
      exporter/trainer `--help` import checks. A one-row smoke wrote `10`
      successful candidate-outcome rows under
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval1_skip0_k0_20260618_1800_dp_rollout_label_smoke1_alloc140738`.
      The inspected row contains the nested DP-rollout label and correctly
      marks a bad DP-prior continuation as not continuable. This is only an
      interface/label smoke, not method evidence.
- [ ] 2026-06-18 18:10+08 launched the first real new-label retrieval shard
      on allocation `140738` / `server54`, session
      `cosmos3_dpcont_retrieval64_skip0_140738_20260618`, log
      `experiments/world_model_task_rebinding/cosmos3/dpcont_retrieval64_skip0_20260618_1815_alloc140738_tmux.log`.
      Output root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_retrieval64_skip0_k8_20260618_1815_dpcont_skip0_alloc140738`.
      It uses `64` hard rows, retrieval `k=8`, model samples `48`, and
      DP-rollout continuability horizon `8`. Current observed status:
      process is active on `server54`, GPU utilization about `31%`, and
      candidate action files are being written. Do not train a new formal
      scorer until this or other DP-rollout-labeled shards are complete and
      unioned.
- [ ] 2026-06-18 19:45+08 completed the first two DP-rollout-labeled
      retrieval shards and started the next formal scorer. Completed shards:
      skip `0` on allocation `140738` and skip `64` on allocation `140717`.
      Each wrote `5440` successful candidate-outcome records with `0` row
      failures. Skip `0` recorded `170` DP-rollout-continuable candidates and
      `118` DP-rollout successes; skip `64` recorded `231` DP-rollout-
      continuable candidates and `167` DP-rollout successes. Their union has
      `128` hard groups, DP success `7`, oracle success `14`,
      mean oracle-minus-DP weighted error `-0.0743`, meaningful improvements
      `95/128`, large improvements `76/128`, and retrieval-family oracle
      candidates in `28` groups. Both terminal-success and progress-headroom
      gates passed.
- [ ] 2026-06-18 19:45+08 launched the 2-shard DP-rollout-continuability
      outcome scorer on allocation `140738`, session
      `cosmos3_dpcont_union2_scorer_140738_20260618`, log
      `experiments/world_model_task_rebinding/cosmos3/dpcont_union2_scorer_20260618_1940_alloc140738_tmux.log`.
      It uses the completed skip `0` and skip `64` outcome labels, rank-loss
      weight `0.0`, `SCORER_MIN_WALL_SECONDS=10800`,
      `SCORER_MAX_WALL_SECONDS=10800`, and `SCORER_FORMAL_MIN_GPUS=1`.
      The run is active and must be allowed to satisfy the three-hour floor
      before being treated as formal scorer evidence. A GPU keepalive follows
      the union wrapper because this scorer is too small to maintain stable
      GPU utilization by itself. Remaining shards skip `128`, `192`, and
      `256` are still running and should be unioned later for a fuller scorer.
- [ ] 2026-06-18 20:00+08 all five DP-rollout-labeled retrieval shards are
      complete. The final shard, skip `128` on allocation `139754`, wrote
      `5440` successful records with `0` failures, `206` DP-rollout-
      continuable candidates, and `184` DP-rollout successes. The complete
      hard set covers `312` groups across skip `0`, `64`, `128`, `192`, and
      `256`. The five-shard union gate passed: DP success `23`, oracle
      success `35`, mean oracle-minus-DP weighted error `-0.0649`,
      meaningful improvements `219/312`, large improvements `171/312`, and
      retrieval-family oracle candidates in `55` groups.
- [ ] 2026-06-18 20:00+08 launched the five-shard formal outcome scorer on
      allocation `140717`, session
      `cosmos3_dpcont_union5_scorer_140717_20260618`, log
      `experiments/world_model_task_rebinding/cosmos3/dpcont_union5_scorer_20260618_2000_alloc140717_tmux.log`.
      It uses the same scorer settings as the two-shard run:
      rank-loss weight `0.0`, `SCORER_MIN_WALL_SECONDS=10800`,
      `SCORER_MAX_WALL_SECONDS=10800`, and `SCORER_FORMAL_MIN_GPUS=1`.
      This five-shard scorer is the preferred next live-controller scorer if
      its three-hour training floor finishes and margin eval passes. The
      two-shard scorer remains useful as an early parallel run, but the
      five-shard scorer has the fuller DP-rollout-continuability label set.
- [ ] 2026-06-18 10:43+08 expanded the diagnostic retrieval set to `64`
      hard rows using short compute-node backfill `139986` on `server14`.
      Added `skip=32` and `skip=48` shards to the isolated diagnostic claim
      root, then ran a true `64`-row union. The `64`-row union shows the
      candidate set is useful: DP success `3/64`, oracle success `6/64`,
      mean oracle-minus-DP error `-0.0544`, meaningful improvements `52/64`,
      large improvements `34/64`, far-phase mean delta `-0.0780`, retrieval
      residuals were oracle candidates in `19` groups and had `1` successful
      group. This supports the DDP/HDP-inspired idea that retrieval/model
      chunks provide progress/contact headroom even when a single chunk does
      not finish insertion.
- [ ] 2026-06-18 10:43+08 found the actual scorer-training issue from the
      diagnostic: rank loss overfits/misranks on small hard-phase replay.
      The `64`-row rank-loss scorer trained well on train but failed held-out
      selection (`selected_minus_dp_prior_weighted_error_mean` `+0.0693`,
      contact-progress delta `-0.0623`, no conservative eval margin passed).
      A no-rank control on the same `64` rows passed conservative offline
      margin eval: `ready_for_conservative_offline_gate=true`; best eval
      margin selected non-DP in `25%` of held-out groups, improved weighted
      error by `-0.0165`, improved contact-progress delta by `+0.0066`, with
      `0` harmful switches and `3` improved switches. This is still only a
      smoke checkpoint, not live evidence.
- [ ] 2026-06-18 10:43+08 changed the default scorer training in both
      retrieval smoke and retrieval union wrappers from rank-loss weight `1.0`
      to `0.0`, with `SCORER_RANK_LOSS_WEIGHT` left as an explicit override.
      Plain reason: the method needs a progress/contact/value outcome scorer,
      but the grouped rank objective is currently less stable than supervised
      outcome heads plus conservative margin selection. Compute-node
      `bash -n` inside allocation `139986` passed for both edited wrappers.
- [ ] 2026-06-18 10:25+08 used two short tmux-held backfill allocations for
      diagnostic-only progress, without touching the formal retrieval claim
      root. Job `139951` on `server05` ran `16` hard rows at `skip=0` and
      produced `784` successful candidate-outcome records. Job `139967` on
      `server34` reused the allocation to run a diagnostic union/scorer smoke,
      a second `16` hard rows at `skip=16`, and then a `32`-row union/scorer
      smoke. The isolated diagnostic claim root is
      `experiments/world_model_task_rebinding/cosmos3/hardphase_retrieval_diag16_claims_20260618`.
      The first shard had DP success `1/16`, oracle success `1/16`, mean
      oracle-minus-DP error `-0.0489`, and retrieval residuals were oracle
      candidates in `5` groups. The second shard had DP success `0/16`,
      oracle success `1/16`, mean oracle-minus-DP error `-0.0683`, and
      retrieval residuals were oracle candidates in `2` groups. The `32`-row
      union gate passed only the new progress-headroom route: DP success
      `1/32`, oracle success `2/32`, mean oracle-minus-DP error `-0.0586`,
      meaningful improvements `26/32`, large improvements `13/32`, far-phase
      mean delta `-0.1084`, retrieval oracle count `7`. This proves the
      retrieval candidate export, progress gate, scorer training, descriptor
      schema, and margin-eval path run end-to-end on compute. It does not
      prove live control.
- [ ] 2026-06-18 10:25+08 fixed the scorer-gate interpretation exposed by
      the diagnostic. The old gate required single-chunk terminal success
      before training the outcome scorer, which is wrong for receding hard
      phases where a valid chunk may only make progress toward contact. Both
      `scripts/slurm/run_cosmos3_hardphase_retrieval_smoke_in_allocation.sh`
      and
      `scripts/slurm/run_cosmos3_hardphase_retrieval_claim_union_in_allocation.sh`
      now pass if either the old terminal-success gate passes or a
      documented progress-headroom gate passes. The progress gate requires
      mean oracle-minus-DP error improvement, enough meaningful/large
      improvement fraction, far-phase progress when far rows exist, and
      retrieval-family contribution. This is not a final eval relaxation; it
      only decides whether the progress/contact/value scorer is worth
      training.
- [ ] 2026-06-18 10:25+08 important negative diagnostic: the `32`-row,
      `50`-step scorer smoke is not usable as a live checkpoint. On the
      training split it selected non-DP candidates in `22/24` groups and
      improved weighted error by `-0.0410`, but on held-out `8` groups it
      selected worse actions (`selected_minus_dp_prior_weighted_error_mean`
      `+0.0857`, progress delta `-0.0287`). Margin eval therefore reported
      `ready_for_conservative_offline_gate=false`, with no gate-passing eval
      margin; the best conservative margin was DP-only. Conclusion: the path
      is wired correctly, but a tiny 32-row smoke overfits and cannot replace
      the pending full retrieval shards/union.
- [ ] 2026-06-18 10:03+08 tested the only remaining legal-looking partition
      detour. `cpu` partition is `UP`, allows account `mayi`, and reports GPU
      TRES, so a tmux-held `salloc --immediate=60 -p cpu --gres=gpu:1`
      1GPU/3h probe was attempted. Temporary job `139943` stayed pending with
      reason `Nodes required for job are DOWN, DRAINED or reserved for jobs in
      higher priority partitions`, then timed out and was revoked. Conclusion:
      `cpu` partition is not a practical GPU backfill route for this run.
- [ ] 2026-06-18 09:58+08 audited old tmux-held `salloc` shells before
      waiting longer. Many stale shells still contain old `SLURM_JOB_ID`
      values such as `133179`, `131564`, `128862`, `127819`, `139038`, and
      older generation/eval IDs, but `scontrol show job` rejects all scanned
      old IDs as invalid. There is therefore no hidden reusable held GPU
      allocation to repurpose for the retrieval-residual replay; only the five
      current watched pending jobs remain real.
- [ ] 2026-06-18 09:50+08 probed the apparent free-GPU holes on
      `server03`, `server23`, `server34`, and `server36` without running any
      project compute on the login node. A combined `-w server03,server23,...`
      probe was invalid because Slurm interpreted the nodelist as four
      required nodes (`invalid number of nodes (-N 4-1)`). Four corrected
      single-node tmux-held `salloc --immediate=30` probes created temporary
      jobs `139932`-`139935`, but all timed out and were revoked with
      `Unable to allocate resources: Connection timed out`. Conclusion:
      these mixed nodes expose accounting-level free GPUs, but the scheduler
      will not give this account an immediate 1-GPU/3h allocation there.
      Preserve the existing watched pending allocations; no retrieval replay
      shard has started yet.
- [ ] 2026-06-18 09:39+08 rechecked all watched retrieval allocations.
      `139754`, `139841`, `139842`, `139861`, and `139892` are all still
      `PENDING (Priority)`, and there are no running GPU jobs for `yanhongru`.
      The shared retrieval claim root is still empty, so no compute shard has
      started and no replay/scorer artifact is missing; it simply does not
      exist yet. Current earliest scheduler estimate is `139754` at
      `2026-06-19T13:06:17`, with the other retrieval fallbacks around
      `2026-06-20T00:21:57` or later.
- [ ] 2026-06-18 09:34+08 added the same candidate-descriptor schema check
      to `scripts/world_model/eval_cosmos3_candidate_outcome_scorer_margins.py`.
      Margin eval now imports the trainer's descriptor schema, refuses a
      mismatched scorer checkpoint when the checkpoint records descriptor
      names, and writes `candidate_descriptor_names` into the margin summary.
      Plain reason: the conservative margin report should describe the same
      scorer feature contract that live selection would accept.
- [ ] 2026-06-18 09:33+08 added a live-time descriptor-schema guard for the
      optional real-outcome scorer. The live loop now has the same
      `CANDIDATE_OUTCOME_DESCRIPTOR_NAMES` schema as the scorer trainer and
      refuses a scorer checkpoint whose saved `candidate_descriptor_names`
      disagree with the live schema. Plain reason: after adding stochastic
      sample temperature/index features, an old or mismatched scorer checkpoint
      must not be silently used to rank live candidate chunks with the wrong
      feature meaning.
- [ ] 2026-06-18 09:27+08 strengthened the candidate descriptor used by the
      next real-outcome scorer. `model_sample_t*_i` candidates now carry an
      explicit model-sample flag, sample temperature, and normalized sample
      index in both
      `scripts/world_model/train_cosmos3_candidate_outcome_scorer.py` and the
      live scorer feature builder in
      `scripts/world_model/run_cosmos3_live_receding_loop.py`. Plain reason:
      the pending retrieval replay/scorer will contain many stochastic
      checkpoint-model samples; the scorer should not have to infer the
      sampling family only from raw action values when candidate provenance is
      causally known at runtime. The descriptor names are now written into
      the scorer manifest, checkpoints, training summary, and live summary so
      the increased feature width is traceable.
- [ ] 2026-06-18 09:21+08 hardened the retrieval union runner for concurrent
      watched allocations. If a union step finds no completed shard yet but
      sees active in-progress shard claims owned by still-pending/running
      Slurm jobs, it now writes `active_inprogress_claims.txt` and exits
      cleanly instead of returning rc `20` as if the replay failed. If no
      completed shard and no active claim exists, it still returns rc `20`.
      Plain reason: overlapping 1/2/4-GPU allocations can legitimately skip
      duplicate shards because another allocation already owns the claim; that
      scheduling race should not be recorded as a method or replay failure.
- [ ] 2026-06-18 09:19+08 checked scheduler/account knobs after the
      immediate probe failed. `sacctmgr` shows user `yanhongru` under account
      `mayi` with QOS `user_yanhongru`; `sprio` reports the five pending
      retrieval jobs all at priority `1000`. No higher legal QOS/account path
      is visible from this account state, so do not keep opening speculative
      allocation variants just to look busy. Preserve the pending watched
      requests and wait for a real `RUNNING` transition.
- [ ] 2026-06-18 09:17+08 tried exactly one immediate tmux-held allocation
      probe for the smallest legal backfill shape: `1` GPU / `3` hours,
      `4` CPU / `32G`, session
      `cosmos3_hardphase_retrieval_1g3h_immediate_20260618`. Slurm created
      temporary job `139895`, did not allocate it within the `--immediate=60`
      window, and cancelled it with no node assigned. No pending job remains
      from this probe. Plain reason: this directly tests whether an immediate
      1-GPU hole exists without leaving another queue entry; the answer is no
      at this moment.
- [ ] 2026-06-18 09:11+08 fixed a margin-eval reporting edge case in
      `scripts/world_model/eval_cosmos3_candidate_outcome_scorer_margins.py`.
      The best-margin selector no longer uses `value or inf`, because a valid
      selected-minus-DP value of exactly `0.0` would be treated as missing and
      could make the diagnostic choose the wrong margin. Plain reason: if the
      retrieval-union scorer runs, the margin report must faithfully describe
      the scorer behavior instead of adding a reporting artifact.
- [ ] 2026-06-18 09:08+08 added one more legal tmux-held backfill request
      after confirming the existing pending jobs are blocked by Slurm
      priority, not by a bad watcher or impossible resource shape. New request
      `139892` asks for `1` GPU / `3` hours with a smaller `4` CPU / `32G`
      shape in session
      `cosmos3_hardphase_retrieval_1gpu_3h_min_20260618`; watcher
      `cosmos3_hardphase_retrieval_1g3h_min_watch_139892_20260618` polls it
      and will run `SHARD_SKIPS=256` with `CPUS_PER_SHARD=4` and
      `MEM_PER_SHARD=32G` if it starts. Existing requests remain pending:
      `139754`, `139841`, `139842`, and `139861`. Plain reason: this keeps
      the method unchanged but gives the scheduler a smaller valid 1-GPU
      3-hour shape that may fit a backfill slot; shared shard claims still
      prevent duplicated hard-phase replay.
- [ ] 2026-06-18 09:04+08 aligned the retrieval scorer margin eval with the
      new progress/contact/value gate. The single-shard and union retrieval
      scorer wrappers now evaluate `checkpoint_best_gate.pt` when it exists,
      falling back to `checkpoint_best_offline.pt` only if no gate checkpoint
      was written. The completion marker records `margin_checkpoint`. Plain
      reason: margin diagnostics should describe the checkpoint that actually
      satisfied the offline progress/contact/value gate, not a best-error-only
      checkpoint that might have failed the progress gate.
- [ ] 2026-06-18 09:00+08 fixed the live outcome-scorer descriptor mapping.
      The live candidate executor names candidates as `dp_prior`, `mean`,
      `scale_*`, `sample_*`, or `diffusion_*`, while the replay exporter trains
      the outcome scorer on `model_dp_prior`, `model_mean`, `model_scale_*`,
      `model_sample_*`, and `model_diffusion_*` with source
      `checkpoint_model`. The live outcome-scorer feature builder now maps
      those names back to the training names before building the candidate
      descriptor. Plain reason: the optional live scorer must see the same
      candidate-source features it was trained on; otherwise the selector could
      fail for an implementation reason rather than a physical-method reason.
- [ ] 2026-06-18 08:56+08 added an optional live entry point for the
      real-outcome progress/contact/value scorer. The live receding loop and
      panel now accept `--candidate-outcome-scorer-checkpoint` plus
      `--candidate-outcome-scorer-dp-margin`. Candidate/diffusion chunks are
      still generated by the candidate executor, but when this checkpoint is
      provided, the final selected chunk can be re-ranked by the real-outcome
      scorer trained from replay labels. Per-iteration JSON records
      `candidate_executor_selector_mode`,
      `candidate_outcome_scorer_checkpoint`, and per-candidate outcome scorer
      predictions/scores. The Slurm live-panel wrappers pass the new env vars
      `CANDIDATE_OUTCOME_SCORER_CHECKPOINT` and
      `CANDIDATE_OUTCOME_SCORER_DP_MARGIN`. Plain reason: the new scorer is
      now not only an offline diagnostic; after a retrieval-union scorer passes
      the gate, there is a causal live path to use it for selecting generated
      action chunks.
- [ ] 2026-06-18 08:59+08 added a live-evidence guard for the optional
      outcome scorer. If `CANDIDATE_OUTCOME_SCORER_CHECKPOINT` is set, the
      generic live-panel compute wrapper now looks for
      `CANDIDATE_OUTCOME_SCORER_SUMMARY` or
      `checkpoint_parent/training_summary.json` and requires
      `ready_for_formal_live_eval=true` plus matching
      `formal_live_eval_checkpoint`. Otherwise the run is marked diagnostic
      and is refused unless `ALLOW_LIVE_RECEDING_DIAGNOSTIC=true`. Plain
      reason: a short retrieval scorer smoke may be useful for debugging, but
      it must not silently become method live evidence.
- [ ] 2026-06-18 09:02+08 tightened the optional outcome scorer interface:
      `--candidate-outcome-scorer-checkpoint` now requires
      `--controller-action-source=candidate_executor` in the Python live loop,
      and the Slurm wrapper records the same mismatch as a diagnostic/refusal
      reason. Plain reason: the real-outcome scorer ranks generated candidate
      chunks; it must not be passed with raw-Cosmos/contact-executor modes and
      then silently ignored.
- [ ] 2026-06-18 08:48+08 made the retrieval scorer launchers explicit about
      the new progress/contact/value scorer semantics. Both the single-shard
      fallback scorer path and the union scorer path now pass explicit
      `--progress-loss-weight`, success/inserted/grasped/progress/progress-
      delta/continuable score weights, and
      `--min-selected-progress-delta-improvement` to training and margin eval.
      The union manifest also records these weights. Plain reason: when the
      compute allocation finally starts, the logs should prove the scorer is
      selecting chunks by physical progress/contact/value, not silently relying
      on old default coordinate-error behavior.
- [ ] 2026-06-18 08:34+08 upgraded the candidate outcome scorer interface
      from mostly error/success ranking to an explicit progress/contact/value
      scorer for the next retrieval-union run. The outcome exporter now writes
      `final_contact_progress_proxy`,
      `final_contact_progress_delta_proxy`, and `final_continuable_proxy`
      from the real replay end state. The scorer now has prediction heads for
      final weighted task error, final peg-head state, contact progress/delta,
      and binary success/inserted/grasped/continuable proxy; its score
      combines error reduction with success, insertion, grasp preservation,
      progress, progress delta, and continuability weights. Plain reason:
      candidate/diffusion chunks should be selected for physical insertion
      progress and DP-continuable handoff, not only for a low final coordinate
      error. The offline gate now also requires selected-vs-DP contact
      progress delta to be no worse than the threshold
      `min_selected_progress_delta_improvement` in addition to weighted-error
      improvement and non-DP usage. The grouped rank loss now targets the same
      real composite value instead of only the minimum final coordinate error.
      This intentionally makes old scorer checkpoints architecture-incompatible;
      future retrieval-union scorer smokes must train fresh checkpoints with
      the new heads.
- [ ] 2026-06-18 08:30+08 retargeted watcher
      `cosmos3_hardphase_retrieval_1g3h_watch_139861_20260618` from
      `SHARD_SKIPS=0` to `SHARD_SKIPS=256`. A light read-only count of the
      current 512-row contact dataset found `312` hard-phase rows
      (`far=94`, `lateral_align=142`, `preinsert_aligned=76`) and `319` rows
      usable for the retrieval-positive bank. Existing pending watchers cover
      `0`, `0,64`, or `0,64,128,192`; the short 1-GPU/3-hour fallback should
      add the tail window instead of duplicating the first shard if it starts
      late. The shared claim hash still prevents duplication if a future run
      covers the same tail window first.
- [ ] 2026-06-18 08:22+08 tried one alternate-partition tmux allocation
      request for `1` GPU / `3` hours on `gaosh,engram,test`. The tmux
      session exited immediately and no new Slurm job appeared in `squeue` or
      `sacct`, so this path did not create a usable allocation request. Do
      not keep spinning partition probes; keep the valid pending requests
      `139754`, `139841`, `139842`, and `139861` under their watchers.
- [ ] 2026-06-18 08:13+08 added a short-backfill tmux-held allocation
      request because the 1-day 1/2/4 GPU requests are still pending on
      scheduler priority. New request `139861` asks for `1` GPU / `3` hours
      in session `cosmos3_hardphase_retrieval_1gpu_3h_20260618`, with watcher
      `cosmos3_hardphase_retrieval_1g3h_watch_139861_20260618`. It uses the
      same sharded retrieval runner; it was first started with
      `SHARD_SKIPS=0` and was retargeted at `08:30+08` to `SHARD_SKIPS=256`,
      and the shared config-specific claim root, so it can contribute the
      first hard-phase retrieval shard if it starts before the 1-day request.
      Plain reason: this does not weaken the method; it only tries a shorter
      valid compute window so retrieval candidate-headroom evidence can begin
      instead of waiting only on the 1-day queue position.
- [ ] 2026-06-18 08:08+08 made shard claims configuration-specific. The
      shard claim directory now uses a short prefix plus a hash over the
      full candidate-replay configuration: contact dataset, checkpoint, hard
      phases, skip/window, horizon, legacy/model candidate scales/temps,
      retrieval source/fields/scales, and model sample count. The raw key and
      hash are written into `claim.txt`. Plain reason: a future run that
      changes checkpoint or candidate sampling must not accidentally reuse or
      skip a claim from an older candidate distribution. Added a `cksum`
      fallback if `sha1sum` is unavailable on a compute node.
- [ ] 2026-06-18 08:05+08 added a union scorer coverage floor. The
      compute-node union runner now defaults
      `MIN_UNION_DONE_CLAIMS_FOR_SCORER=2`: if only one shard has completed,
      it still writes the union headroom summary and gate, but skips scorer
      training with reason `union_done_claim_count_below_floor`. Plain reason:
      if `139754` one-GPU starts first, the first 64-row shard should not
      immediately create a scorer that will be superseded by the later
      2-/4-GPU union; scorer training should wait until at least two hard
      windows are available.
- [ ] 2026-06-18 08:00+08 disabled per-shard scorer smoke in the sharded
      retrieval launchers. The shard workers now pass
      `RUN_SCORER_AFTER_HEADROOM_GATE=false`, so each shard only generates
      replay outcomes and a local headroom summary. The only scorer smoke is
      launched by the union gate after completed shards are combined. Plain
      reason: otherwise every shard could train a scorer and the union could
      train another one, wasting GPU and confusing which candidate pool the
      scorer actually evaluated. Restarted all three pending watcher sessions
      at `07:59+08` with this change.
- [ ] 2026-06-18 07:57+08 fixed shard watcher return-code collection. The
      one-/two-/four-GPU drivers now save every background `srun` PID and
      `wait` each PID explicitly, instead of relying on a bare `wait` whose
      return status can miss one failed background shard. Restarted all three
      watcher sessions at `07:57+08`; pending jobs are still `139754`,
      `139841`, and `139842`.
- [ ] 2026-06-18 07:56+08 made retrieval contribution explicit in the
      headroom summaries and scorer gates. The headroom summarizer now reports
      oracle candidate families and successful-candidate family group counts
      in both JSON and markdown. The single-shard and union retrieval gates
      now require the retrieval family itself to contribute at least one oracle
      candidate or at least one successful-candidate group
      (`MIN_RETRIEVAL_ORACLE_COUNT=1` or
      `MIN_RETRIEVAL_SUCCESS_GROUPS=1`) before launching a scorer smoke. Plain
      reason: retrieval replay still includes legacy/model candidates; if
      those old families win, that is not evidence that the retrieval residual
      repair worked.
- [ ] 2026-06-18 07:53+08 fixed the orphaned in-progress claim case. If a
      shard leaves only `claim.txt` with no `done.txt` or `failed.txt`, a
      later allocation now reads the recorded `slurm_job_id` and checks
      `squeue`; if that owner job is no longer active, the claim is archived
      as orphaned and the shard can be re-claimed. If the owner job is still
      `PENDING/RUNNING/CONFIGURING/COMPLETING/SUSPENDED`, the later allocation
      still skips it to avoid duplicate GPU work. This handles hard node/job
      termination where the shell trap may not write `failed.txt`.
- [ ] 2026-06-18 07:50+08 fixed the shard-claim recovery path. A completed
      claim with `done.txt` still prevents duplicate GPU work, but a claim
      that ended with `failed.txt` is now archived and can be re-claimed by a
      later allocation by default (`ALLOW_RERUN_FAILED_SHARD=true`). Plain
      reason: the first compute attempt may expose a syntax/runtime bug; a
      failed first shard must not permanently block `139841` or `139842` from
      retrying that same hard window after the code is fixed.
- [ ] 2026-06-18 07:48+08 hardened the shard watcher driver logs. The
      one-/two-/four-GPU watchers now record both `shard_wait_rc` and
      `union_rc` in their `parallel_driver.log` instead of printing a plain
      done line that could hide a shard or union failure. Restarted all three
      watcher sessions so the pending jobs use this return-code reporting.
- [ ] 2026-06-18 07:46+08 corrected the four-GPU watcher resource request.
      The 4-GPU allocation request has `240G` total memory; four parallel
      shards at `64G` each could over-request `256G` and fail to place the
      steps even after the allocation starts. Restarted
      `cosmos3_hardphase_retrieval4_watch_139842_20260618` with
      `MEM_PER_SHARD=56G`, so four shards request `224G` total and keep
      `32` CPUs / `4` GPUs fully occupied without exceeding the allocation.
- [ ] 2026-06-18 07:45+08 restarted the one-GPU watcher
      `cosmos3_hardphase_retrieval_watch_139754_20260618` onto the same
      sharded/union path as the multi-GPU fallbacks. It now uses
      `scripts/slurm/watch_cosmos3_hardphase_retrieval_sharded_allocation.sh`
      with `SHARD_SKIPS=0`, so when `139754` starts it will claim/run the
      first retrieval shard and then run the compute-node union summary over
      whatever completed claims exist. This makes all three pending
      allocations use one resource-management rule: claim shard, run only if
      unclaimed, then summarize the union.
- [ ] 2026-06-18 07:43+08 added compute-only multi-shard retrieval union
      summary. New runner
      `scripts/slurm/run_cosmos3_hardphase_retrieval_claim_union_in_allocation.sh`
      scans completed shard claims under the shared claim root, collects their
      `candidate_outcome_labels.jsonl` files, runs the existing headroom
      summarizer over the union, writes
      `retrieval_union_scorer_gate_summary.json`, and only then runs a short
      union scorer smoke plus margin eval if the union candidate pool clears
      the same physical headroom gate. Updated the 2-GPU and 4-GPU watchers to
      launch this union runner via compute-node `srun` after their shard
      workers finish; the 2-GPU/4-GPU watcher sessions were restarted at
      `07:42+08` and the 1-GPU watcher was restarted at `07:45+08` so all
      pending jobs load the union logic.
- [ ] 2026-06-18 07:39+08 added shard-claim protection for the pending
      retrieval allocations. Because `139754`, `139841`, and `139842` may
      start in different order, the low-level replay wrapper now supports a
      shared `SHARD_CLAIM_ROOT` and writes one claim per `(HARD_MAX_ROWS,
      HARD_SKIP_ROWS, retrieval settings, model sample count)` shard. The
      retrieval runner defaults this root to
      `experiments/world_model_task_rebinding/cosmos3/hardphase_retrieval_shard_claims_20260618`.
      If a later allocation reaches a shard that was already claimed, it writes
      `shard_claim_skipped.txt` and exits cleanly instead of trying to read a
      missing headroom summary. Plain reason: if the 1-GPU allocation starts
      first and runs `skip=0`, the 2-GPU/4-GPU fallbacks must not burn GPU on
      the same shard again; they should only contribute new coverage.
- [ ] 2026-06-18 07:34+08 added a second tmux-held allocation request so the
      work is not blocked only on the existing one-GPU queue position.
      Allocation request `139841` asks for `2` GPUs / `1` day in session
      `cosmos3_hardphase_retrieval2_2h200_20260618`. Watcher
      `cosmos3_hardphase_retrieval2_watch_139841_20260618` runs
      `scripts/slurm/watch_cosmos3_hardphase_retrieval2_allocation.sh`; when
      the allocation starts it launches two compute-node `srun` shards in
      parallel: `HARD_SKIP_ROWS=0` and `HARD_SKIP_ROWS=64`. This is to keep a
      two-GPU allocation meaningfully used rather than running a one-GPU
      replay on two allocated GPUs. Current scheduler estimates:
      `139754` one-GPU starts around `2026-06-19T13:06:17`, while `139841`
      two-GPU starts around `2026-06-19T21:00:00`; estimates may move.
- [ ] 2026-06-18 07:36+08 added a four-GPU fallback request without changing
      the experiment protocol. Allocation request `139842` asks for `4` GPUs
      / `1` day in session
      `cosmos3_hardphase_retrieval4_4h200_20260618`. Watcher
      `cosmos3_hardphase_retrieval4_watch_139842_20260618` uses
      `scripts/slurm/watch_cosmos3_hardphase_retrieval_sharded_allocation.sh`
      and will run four one-GPU `srun` shards in parallel with skips
      `0,64,128,192` if the allocation starts. This prevents a 4-GPU
      allocation from being used as a one-GPU run. Current scheduler estimate
      for `139842` is `2026-06-19T21:00:00`, so it is a fallback rather than the expected
      first resource.
- [ ] 2026-06-18 07:27+08 tightened the pending retrieval run into a
      compute-only candidate-headroom gate before any scorer smoke. The same
      watcher still calls
      `scripts/slurm/run_cosmos3_hardphase_retrieval_smoke_in_allocation.sh`,
      but that runner now does: retrieval replay, summarize real headroom,
      write `retrieval_scorer_gate_summary.json`, and only then run a short
      outcome-scorer smoke plus conservative margin eval if the replay has
      materially better hard-phase candidates. Default gate: oracle success
      must beat DP by at least `2`, oracle success must be at least `4`,
      mean oracle-minus-DP error must be `<= -0.025`, and `far` must contain
      at least one oracle-success candidate when `far` rows are present. If
      this gate fails, the recorded next step is not another scorer; it is a
      stronger hard-phase teacher/candidate source. This keeps the current
      diagnosis plain: scorer overfitting is secondary; the main blocker is
      missing reliable successful hard-state action chunks. Also added
      generic retrieval residual scales (`0.5,1.0,1.5` in the pending runner)
      so the first replay tests whether successful neighbor residuals transfer
      with a modest under/over-correction, instead of betting on exactly one
      residual magnitude.
- [ ] 2026-06-18 07:25+08 prepared the next candidate-distribution repair:
      retrieval residual candidates. Updated
      `scripts/world_model/export_cosmos3_candidate_outcome_labels.py` with
      opt-in `--include-retrieval-residual-candidates`. It builds a bank from
      rows with `future_inserted_within_chunk` or
      `future_dp_continuable_within_chunk`, retrieves phase/contact-neighbor
      residual chunks, and adds scaled versions of them to the current DP
      prior. This is a general contact-phase candidate source, not failed-case
      recovery. Also updated the exploratory wrapper with retrieval env
      controls and added
      `scripts/slurm/run_cosmos3_hardphase_retrieval_smoke_in_allocation.sh`.
      This has not been compute-validated yet because no allocation is active;
      the runner will do `py_compile` and `bash -n` inside compute before
      replay.
- [ ] 2026-06-18 07:25+08 pending retrieval smoke: tmux watcher
      `cosmos3_hardphase_retrieval_watch_139754_20260618` is waiting for
      allocation `139754` and will run
      `scripts/slurm/run_cosmos3_hardphase_retrieval_smoke_in_allocation.sh`
      via `srun` once the 1-H200/1-day allocation starts. Current `squeue
      --start` estimate is `2026-06-19T13:06:17`, but this can
      shift. The smoke should be interpreted only as candidate-headroom
      evidence. If retrieval does not materially improve hard-phase success,
      especially `far`, the next step is a stronger teacher/candidate source,
      not another scorer.
- [ ] 2026-06-18 07:20+08 static data check for retrieval bank: the full
      `512`-row contact-executor dataset has positive residual-bank rows in
      every relevant phase when using
      `future_inserted_within_chunk OR future_dp_continuable_within_chunk`:
      `far=24/94`, `lateral_align=50/142`,
      `preinsert_aligned=47/76`, and `dp_continuable=198/200`. Therefore the
      pending retrieval smoke is not expected to fail from an empty bank. If
      it fails, the likely issue is residual transfer quality or simulator
      physics, not missing positive rows.
- [ ] 2026-06-18 07:25+08 updated the outcome scorer descriptor to recognize
      `retrieval_resid_*` candidates and the
      `retrieval_success_residual` source. This is only for future scorer
      training after retrieval replay proves real hard-phase headroom; it does
      not make the current weak candidate pool live-ready.
- [ ] 2026-06-18 07:30+08 updated the headroom summarizer so retrieval
      candidates are reported as `retrieval_success_residual` instead of being
      hidden under a generic fallback family. The retrieval smoke runner now
      compute-compiles the exporter, scorer, and summarizer before replay.
- [ ] 2026-06-18 07:10+08 broad hard-phase candidate exploration has a clear
      result and should not be promoted to scorer/formal/live as-is. Across
      `208` unique hard rows from the exploratory runs
      (`skip=0`, `64`, `128`, `160`, `192`) the widened candidate pool reached
      DP success `10/208` and oracle-best success `18/208`, with weighted
      mean oracle-minus-DP error about `-0.051`. This proves broad stochastic
      sampling is useful for reducing error, but not enough for task
      completion. By phase: `far` has only `1/63` oracle successes,
      `lateral_align` has `11/93`, and `preinsert_aligned` has `6/52`.
      Plain meaning: the current candidate distribution sometimes improves
      geometry and finds a few extra insertions, but it still almost never
      solves the hardest `far` states. Next repair must create stronger
      hard-phase teacher/candidate chunks, especially for `far`; do not spend
      the next real run on another scorer over this same weak pool.
- [ ] 2026-06-18 07:10+08 resource/execution record for the broad exploration:
      allocation `139778` on `server34` completed the first `64` hard rows at
      `candidate_outcome_labels_hardphase_explore64_samples48_20260618_053225_alloc139778`
      and replayed rows `64-127` at
      `candidate_outcome_labels_hardphase_explore64_skip64_samples48_20260618_alloc139778`
      before the allocation timed out. Allocation `139781` completed the
      small `16`-row duplicate smoke and was also used to summarize the
      `skip64` replay. Allocation `139780` on `server14` first began a
      duplicate `32`-row run; that step was cancelled and the allocation was
      reused for new rows `128-159`, `160-191`, and `192-207`. Added
      `HARD_SKIP_ROWS` to
      `scripts/slurm/run_cosmos3_hardphase_exploratory_candidate_replay_in_allocation.sh`
      so future tmux-held allocations can cover disjoint hard-row windows
      without rewriting the wrapper.
- [ ] 2026-06-18 03:42+08 prepared the next aligned test for candidate
      distribution, not another reweighted imitation run. Added
      `scripts/slurm/run_cosmos3_hardphase_exploratory_candidate_replay_in_allocation.sh`.
      It runs only inside a compute-node allocation, filters the contact
      executor rows to hard phases (`far,lateral_align,preinsert_aligned`),
      then replays a broader stochastic candidate set from the latest
      hard-phase checkpoint: default `64` hard rows, `48` model samples,
      temps `0.25,0.5,0.75,1.0,1.5,2.0`, model scales `0.2,0.5,1.0,1.5`,
      plus legacy teacher-scale candidates. Purpose: test whether expanding
      the action-candidate distribution can produce actual successful hard
      chunks, especially in `far`, before spending time on another scorer.
      A watcher is pending for short allocation `139778`; if it starts, it
      will run this wrapper via `srun` in tmux session
      `cosmos3_hardphase_dist_1h200_1h_20260618`.
- [ ] 2026-06-18 03:45+08 added a smaller backfill request because `139778`
      was pushed later by the scheduler. New request `139780` is `1` H200 for
      `30` minutes and currently has an earlier start estimate
      `2026-06-18T08:00:00` on `server54`. Its watcher
      `cosmos3_hardphase_watch_139780_20260618` will run the same exploratory
      wrapper with `HARD_MAX_ROWS=32` and output roots
      `candidate_outcome_labels_hardphase_explore32_samples48_20260618_alloc139780`
      and
      `candidate_outcome_headroom_hardphase_explore32_samples48_20260618_alloc139780`.
      This is only a short hard-candidate coverage diagnostic.
- [ ] 2026-06-18 03:48+08 added an even shorter backfill request `139781`
      because it currently starts earlier than `139780`:
      `2026-06-18T05:46:20` on `server34`. Its watcher
      `cosmos3_hardphase_watch_139781_20260618` will run the exploratory
      replay wrapper with `HARD_MAX_ROWS=16`, writing
      `candidate_outcome_labels_hardphase_explore16_samples48_20260618_alloc139781`
      and
      `candidate_outcome_headroom_hardphase_explore16_samples48_20260618_alloc139781`.
      This is not method evidence; it is a quick check whether broader
      stochastic samples produce any hard-phase success candidates at all.
- [ ] 2026-06-18 03:40+08 hard-phase outcome-oracle smoke on allocation
      `139764` completed as a debug run, not method evidence. Added
      `scripts/slurm/run_cosmos3_hardphase_outcome_oracle_smoke_in_allocation.sh`
      so the same chain can run inside a tmux-held compute allocation without
      fragile one-line shell quoting. The 1-H200/1h job ran on `server12` and
      timed out only after the planned steps had completed; `sacct` shows the
      training/replay steps completed. The 100-step hard-phase balanced
      outcome-oracle checkpoint is at
      `experiments/world_model_task_rebinding/cosmos3/outcome_oracle_candidate_executor_hardphase_balanced_smoke100_20260618_alloc139764`.
      It is a short smoke, not formal training.
- [ ] 2026-06-18 03:40+08 the new hard-phase balanced generator did not beat
      the old generator as a standalone candidate source. Its 256-row real
      replay
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_hardphase_balanced_smoke100_replay256_20260618_alloc139764`
      wrote `6912/6912` outcomes with zero failures. Headroom:
      DP success `91/256`, oracle success `101/256`, any success `102/256`,
      mean DP error `0.168494`, mean oracle error `0.139425`,
      mean oracle-minus-DP `-0.029069`. Hard phases remain weak:
      `far` `1/43`, `lateral_align` `7/75`, `preinsert_aligned` `5/39`
      oracle successes. This is close to but not better than the previous
      old-generator 256 replay (`oracle` mean about `0.137287`, any success
      `103`). Plain meaning: phase-balanced imitation of replay-oracle targets
      is not enough to create new successful hard-state chunks.
- [ ] 2026-06-18 03:40+08 the new hard-phase generator is complementary but
      still not enough. On the same first `128` rows, new-only headroom was
      DP success `45`, oracle success `51`, mean oracle error `0.136933`;
      old-only same-128 headroom was oracle success `51`, any success `52`,
      mean oracle error `0.135108`; old+new union improved mean oracle error
      to `0.131715` and oracle-minus-DP to `-0.035175`, but did not increase
      oracle success beyond `51/128` and still had `far` `0/18` success.
      Artifact roots:
      `candidate_outcome_headroom_hardphase_balanced_smoke100_replay128_20260618_alloc139764`,
      `candidate_outcome_headroom_handoffgeom512_same128_as_hardphase_smoke_20260618_alloc139764`,
      and
      `candidate_outcome_headroom_union_old_hardphase_smoke_same128_20260618_alloc139764`.
- [ ] 2026-06-18 03:40+08 the union outcome scorer still cannot be promoted to
      formal/live. The short scorer
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_old_hardphase_smoke_same128_rank1_canddesc_smoke1000_20260618_alloc139764`
      and margin eval
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_old_hardphase_smoke_same128_margin_eval_20260618_alloc139764`
      found best held-out margin `0.2`, selected-minus-DP `-0.003604`,
      non-DP fraction `0.75`, and `0` gate-passing margins. Plain meaning:
      the scorer is moving in the right direction on this tiny debug set, but
      it still misses the configured `-0.005` improvement gate and is too weak
      for live control.
- [ ] 2026-06-18 02:35+08 allocation `139069` was revoked after the latest
      outcome/scorer checks; `sacct` reports `CANCELLED by 0` after
      `05:20:47`, and the tmux pane printed `salloc: Job allocation 139069
      has been revoked.` No foreground experiment remains running in that
      allocation. New tmux-held interactive requests are pending:
      `139754` for `1` H200 / `1` day and `139758` for `1` H200 / `4` hours.
      The 2-GPU and 4-GPU pending requests were cancelled after `squeue
      --start` showed no earlier start advantage. Current resource blocker:
      all non-drained gpu nodes report `gres/gpu=8` allocated; only drained or
      down nodes have unallocated GPUs.
- [ ] 2026-06-18 02:50+08 prepared the next hard-phase repair in code, but it
      still needs compute-node syntax/smoke validation. Updated
      `scripts/world_model/train_cosmos3_outcome_oracle_candidate_executor.py`
      with opt-in phase-balanced or hard-boost sampling, hard-phase-specific
      oracle target margin, and optional success-first target selection. The
      default behavior is unchanged. Intended first compute-node command after
      allocation starts: run `py_compile`, then a short hard-phase balanced
      outcome-oracle smoke using the 512 replay and verify its real candidate
      replay before any scorer/formal/live step.
- [ ] 2026-06-18 02:20+08 latest blocker is now concrete: the real replay
      path is clean, but the selector and candidate generator are both still
      insufficient for live control. The `512`-row replay
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_handoffgeom512_after_scorerfail_20260617_server61_alloc139069`
      finished with `14336/14336` successful outcome records and zero failed
      rows. It shows real headroom: DP succeeds on `172/512`,
      oracle-best succeeds on `197/512`, and mean oracle-minus-DP error is
      `-0.026978`. But the hard phases remain weak: `far` has only
      `3/94` oracle successes, `lateral_align` has `15/142`, and
      `preinsert_aligned` has `15/76`. Plain meaning: evaluation/replay is
      not the current blocker; hard-state candidate quality is still poor.
- [ ] 2026-06-18 02:20+08 the best current scorer repair still cannot be used
      for live control. The 512-row candidate-descriptor/rank smoke
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_handoffgeom512_rank1_canddesc_smoke1000_after_scorerfail_20260617_server61_alloc139069`
      learned the train split but selected worse-than-DP candidates on held-out
      rows: final selected-minus-DP `+0.032771`, best offline
      selected-minus-DP `+0.028280`, gate `false`. A conservative DP-default
      margin sweep at
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_handoffgeom512_rank1_canddesc_margin_eval_20260618_server61_alloc139069`
      reduced harm but still only reached selected-minus-DP `-0.002332` with
      non-DP fraction `0.078125`. This is not enough for formal training or
      live eval.
- [ ] 2026-06-18 02:20+08 the outcome-oracle candidate executor smoke did not
      solve candidate generation. The new trainer
      `scripts/world_model/train_cosmos3_outcome_oracle_candidate_executor.py`
      trains from real replay outcomes by choosing the DP prior unless a
      replayed candidate beats it. Its short smoke at
      `experiments/world_model_task_rebinding/cosmos3/outcome_oracle_candidate_executor_handoffgeom512_smoke1000_20260618_server61_alloc139069`
      was only a debug gate. Real 256-state replay without stochastic samples
      was weaker than the old generator (`oracle` success `102`, mean
      oracle error `0.155483`), and replay with `16` Gaussian samples reached
      oracle success `100` and mean oracle error `0.139860`, still not a
      clear improvement over the old candidate replay.
- [ ] 2026-06-18 02:20+08 unioning old candidates with the outcome-oracle
      samples gives only small complementarity, not a solved method. The union
      headroom
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_handoffgeom256_union_old_outcomeoracle_samples16_20260618_server61_alloc139069`
      has DP success `91`, oracle success `102`, any success `104`, mean
      oracle error `0.134673`, and mean oracle-minus-DP `-0.033821`.
      The union scorer smoke still overfits: final held-out selected-minus-DP
      `+0.053832`; best offline `+0.011095`. A margin sweep nearly recovers
      DP-default behavior (`-0.004509`, non-DP fraction `0.09375`) but still
      fails the gate. Next work should target phase-balanced, outcome-supervised
      candidate generation for `far/lateral/preinsert` states, not another
      formal scorer or threshold tweak.
- [ ] 2026-06-17 23:55+08 candidate-headroom decomposition is complete and
      changes the diagnosis. Output:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_headroom_handoffgeom256_after_livefail_20260617_server61_alloc139069`.
      Over all `256` replay states, DP succeeds on `91`, oracle-best succeeds
      on `101`, and any candidate succeeds on `103`; mean oracle-minus-DP
      error is `-0.0312`. On the same scorer split, validation has `64`
      groups, DP success `17`, oracle success `22`, and mean
      oracle-minus-DP `-0.0381`. Plain meaning: held-out candidate headroom
      exists, so the immediate failure is selector generalization, not purely
      no candidate headroom. The hard phases are still clear: `far` has only
      `1/43` oracle successes, `lateral_align` has `8/75`, and
      `preinsert_aligned` has `5/39`.
- [ ] 2026-06-17 23:55+08 adding candidate provenance/type features to the
      scorer helped but still did not pass the offline gate. The descriptor
      smoke
      `candidate_outcome_scorer_handoffgeom256_rank1_canddesc_smoke1000_after_livefail_20260617_server61_alloc139069`
      improved best held-out selected-minus-DP from about `+0.040` to
      `+0.00316`, but it still did not beat DP and is not live-ready. This
      supports the next step: more real outcome replay / stronger selector
      supervision, not formal scorer training from the current 256 rows.
- [ ] 2026-06-17 23:55+08 launched a full `512`-row real outcome replay on the
      held `server61` allocation `139069`:
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_handoffgeom512_after_scorerfail_20260617_server61_alloc139069`.
      It uses the same handoff-geometry checkpoint, `24`-step horizon,
      legacy teacher-scale candidates, and `16` model diffusion candidates.
      When it finishes, rerun headroom summary with the same split, then only
      run another scorer smoke if the 512 replay passes without outcome
      failures. This is still offline replay/debug data, not method evidence.
- [ ] 2026-06-17 23:35+08 latest scorer repair result is negative.
      `scripts/world_model/train_cosmos3_candidate_outcome_scorer.py` now has
      an opt-in grouped ranking loss that compares candidates only inside the
      same live-state `uuid`. Syntax was checked inside allocation `139069`.
      Two 256-row short smokes ran on `server61`:
      `candidate_outcome_scorer_handoffgeom256_rank1_smoke1000_after_livefail_20260617_server61_alloc139069`
      and
      `candidate_outcome_scorer_handoffgeom256_rank1_errorscore_smoke1000_after_livefail_20260617_server61_alloc139069`.
      Both failed the offline gate. The first had final held-out
      `selected_minus_dp=+0.0720` and best `+0.0404`; the error-only scoring
      ablation had final `+0.0772` and best `+0.0402`. In both runs the train
      split selected better-than-DP candidates (`about -0.0286`) but held-out
      groups were worse than DP. Plain blocker: adding a ranking loss can
      memorize candidate ordering on seen states, but current features/data do
      not generalize candidate selection to new live states. Do not launch
      formal scorer training or live eval from these checkpoints.
- [ ] 2026-06-17 23:35+08 the current highest-priority debug is candidate
      headroom/coverage, not more scorer loss tuning. The 256-row outcome
      replay has real but small oracle headroom (`oracle_minus_dp=-0.0382` on
      the held-out split), while learned selectors choose worse candidates.
      Next action: decompose real replay outcomes by scenario, phase, start
      error, candidate family, and oracle success/improvement. If many states
      have no successful or meaningfully improving candidate, the fix must be
      stronger candidate generation / teacher chunks / broader outcome data.
      If headroom exists only in narrow phases, the executor needs phase/contact
      conditioned generation, not a global scalar scorer.
- [ ] 2026-06-17 10:35+08 latest priority is now
      **handoff-geometry 24-step executor**, not another 8-step cap/threshold
      tweak. The corrected `ACTION_EXEC_HORIZON=24` live panel ran on
      allocation `136144` / `server23` at
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260617_092327_alloc136144_unknown_node_samples0_1_3_4`.
      It preserved the full contract on all four samples (`301` frames,
      prefix `300`) but succeeded on only `1/4`: sample00 succeeded with
      peg-head-at-hole `[-0.006579, -0.000144, -0.001397]`; sample01 failed
      at `[-0.121433, 0.009202, -0.019700]`; sample03 failed at
      `[-0.103619, -0.002997, 0.004283]`; sample04 failed at
      `[-0.106229, -0.010824, -0.052037]`. The contact sheet
      `live_receding_panel_contact_sheet.png` was inspected. Plain blocker:
      24-step execution fixed the old horizon mismatch and can solve one row,
      but the executor still usually leaves the peg near the hole instead of
      reliably entering a DP-continuable insertion geometry.
- [ ] 2026-06-17 10:35+08 the concrete failure mode from the latest panel is
      not render, not length accounting, and not a missing closed-loop call.
      Successful sample00 had `85` DP handoff steps after executor progress.
      Failed sample01 and sample04 had `0` DP handoff steps, and sample03 had
      only `19`; the real-state gate stayed false because y/z or insertion
      geometry was still outside the conservative DP-continuability region.
      The last selected chunks also show scorer/readout mismatch: e.g.
      sample04 predicted high inserted/DP-continuable probability while the
      real state still had large y/z error. Next action is therefore to train
      the executor to reach DP-continuable task-frame geometry, not to loosen
      a hand-coded gate or add case-specific recovery.
- [ ] 2026-06-17 10:35+08 a low-utilization handoff-geometry formal startup
      at
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_handoffgeom_train512_formal_1gpu3h_20260617_102938_server23_alloc136144`
      was intentionally stopped before the formal floor because 30-second GPU
      sampling stayed around `11-19%`, which risks violating the user's
      allocation-utilization rule. It is not evidence. The replacement heavy
      formal run is active on the same allocation at
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_handoffgeom_heavy_train512_formal_1gpu3h_20260617_103345_server23_alloc136144`
      with `FORMAL_MIN_GPUS=1`, `MIN_WALL_SECONDS=10800`, `HIDDEN_DIM=4096`,
      `NUM_LAYERS=8`, `BATCH_SIZE=512`, stronger next-state loss/scoring
      (`NEXT_STATE_LOSS_WEIGHT=2.0`, `SCORE_NEXT_STATE_WEIGHT=3.0`,
      axis weights `1,4,8`), and higher DP-continuability weight
      (`SCORE_DP_CONTINUABLE_WEIGHT=1.2`). GPU sampling showed meaningful
      utilization spikes (`32-98%`). Do not use this run for live eval until
      it reaches the 3h floor and its best-gate checkpoint passes the existing
      offline gate.
- [ ] 2026-06-17 06:05+08 full `512` rolling-24 executor path has
      moved from idea to running formal training. The full 24-step DP prior
      export finished on `server54` allocation `133179` at
      `experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_rollout24_train512_20260617_server54_alloc133179`
      with `512/512` records and zero failures. It was joined with the
      Cosmos predicted-task-path rows and existing 24-step contact labels into
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_rollout24_predpath_train512_20260617_server23_alloc135069`
      with `512/512` joined rows, phase counts `dp_continuable=200`,
      `far=94`, `lateral_align=142`, `preinsert_aligned=76`,
      `future_inserted=185`, and `future_dp_continuable=319`. A short
      1GPU smoke at
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_predpath_train512_smoke200_20260617_server23_alloc135069`
      confirmed `action_horizon=24` and retained held-out signal on the full
      split: final selected action MSE `0.006749` versus DP prior `0.007762`,
      selected-minus-DP `-0.001013`, non-DP selected fraction `0.727`,
      inserted acc `0.935`, and DP-continuable acc `0.896`. This is still
      not live evidence, but it is enough to start the formal floor. A formal
      1GPU/3h run is now running on `server23` allocation `135069` at
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_predpath_train512_formal_1gpu3h_20260617_server23_alloc135069`
      with session `90757`, `--min-wall-seconds 10800`, and
      `--formal-min-gpus 1`. Next required gate: wait for the 3h summary,
      then run live closed-loop only if the formal final or best-gate
      checkpoint passes the offline gate.
- [ ] 2026-06-17 06:15+08 launched a second formal 1GPU/3h run with
      tighter eval cadence because the first formal run used
      `--eval-every-steps 1000` and missed the early step-50 gate seen in
      smoke. The second run is on `server54` allocation `133179`, session
      `48840`, at
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_predpath_train512_formal_1gpu3h_eval50_20260617_server54_alloc133179`
      with `--eval-every-steps 50`, `--min-wall-seconds 10800`, and
      `--formal-min-gpus 1`. It already wrote `checkpoint_best_gate.pt` from
      step `50`: selected action MSE `0.007454` versus DP prior `0.007762`,
      selected-minus-DP `-0.000308`, non-DP fraction `0.468`, inserted acc
      `0.896`, DP-continuable acc `0.909`, progress MSE `0.0448`, and value
      MSE `0.0866`. This checkpoint is not formal-live-ready until the same
      run reaches the 3h wall-clock floor and writes its final summary.
- [ ] 2026-06-17 05:35+08 24-step candidate-executor smoke on a
      non-toy predicted-path subset is positive as a code/data-direction gate,
      but not formal evidence. A clean `64`-row rolling-24 prior was exported
      on `server23` at
      `experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_rollout24_predpath_train64_20260617_server23_alloc135069`
      with `64/64` records and zero failures, then joined into
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_rollout24_predpath_train64_20260617_server23_alloc135069`
      (`future_inserted=27/64`, `future_dp_continuable=40/64`). The
      100-step 1GPU smoke at
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_rollout24_predpath_train64_smoke100_20260617_server23_alloc135069`
      confirmed `action_horizon=24` and `target_dim=168`. Its best-gate
      checkpoint at step `60` passed the offline gate on the small held-out
      split: selected action MSE `0.004897` versus DP prior `0.005832`,
      selected-minus-DP `-0.000936`, non-DP selected fraction `0.5`,
      teacher inserted acc `1.0`, DP-continuable acc `1.0`, progress MSE
      `0.0445`, and value MSE `0.0684`. Final step `100` drifted on progress
      MSE, and the run did not meet the formal wall-clock floor, so it is not
      live evidence. It does justify finishing the full `512` rolling-24
      prior export, running a full-512 short smoke, then launching formal
      1GPU/3h training only if the full smoke retains held-out gate signal.
- [ ] 2026-06-17 05:28+08 `sample_03_hole_late_fast_shift` also
      completed the full live contract and failed physically:
      `final_observed_frames=301`, `final_prefix_frame_index=300`,
      `video_file_contract_ok=true`, `full_episode_length_ok=true`,
      final success `false`, final peg-head-at-hole
      `[-0.103819, -0.001383, 0.003959]`, controller counts
      `EXECUTOR_ACTIVE=160`, `DP_HANDOFF=8`, `DP_SCAN_TARGET=132`,
      `INIT_OBS=1`. Framebook review of frames `270..300` confirms the peg
      remains outside the hole. The final live panel evidence now shows the
      same failure pattern on `sample_01`, `sample_03`, and `sample_04`:
      the loop is operational and the 301/300 contract is intact, but the
      8-step executor repeatedly selects small `scale_0.2` chunks and does
      not move the held peg into a DP-continuable insertion state.
- [ ] 2026-06-17 05:24+08 repaired and smoke-tested the next
      24-step executor data path. `train_cosmos3_candidate_outcome_scorer.py`
      now treats both `dp_prior` and `model_dp_prior` as the DP baseline and
      requires at least `--min-eval-groups-for-gate` groups before an offline
      checkpoint can pass, preventing tiny validation splits from creating a
      false gate. A generated-candidate outcome scorer smoke on the hard32
      labels with a 50/50 split wrote
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_generated_hard32_smoke100_val50_20260617_server54_alloc133179`.
      It selected non-DP candidates on held-out groups and reduced selected
      weighted error from DP `0.294614` to `0.289232`, but this is only a
      small diagnostic margin and the underlying candidate outcome replay
      still has `0/672` successes, so it must not launch formal live eval.
      The useful next direction is 24-step candidate generation, not another
      formal scorer on the 8-step weak candidates.
- [ ] 2026-06-17 05:23+08 added rolling-horizon support to
      `scripts/world_model/export_cosmos3_executor_dp_prior_chunks.py`.
      With `--rollout-horizon 24`, the exporter now executes the frozen DP in
      the real simulator, reapplies the source target pose each step, and
      re-queries DP across action chunks to produce a 24-step DP prior. A
      compute-node smoke at
      `experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_rollout24_smoke2_20260617_server54_alloc133179`
      wrote `2/2` priors with shape `24x7`. This addresses the concrete
      reason the old candidate executor was stuck at 8 steps: the teacher
      and task path already had 24 steps, but the DP prior npz had only
      `8x7`, so the trainer/checkpoint horizon collapsed to `8`. A full
      `512`-row rolling-24 prior export is running at
      `experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_rollout24_train512_20260617_server54_alloc133179`;
      after it finishes, join it with existing 24-step contact labels and run
      only a short 24-step candidate-executor smoke before considering any
      formal training.
- [ ] 2026-06-17 05:12+08 `sample_04_hole_late_sine` completed the
      full live contract and failed for a concrete physical reason, not a
      render/eval reason:
      `sample_panel_result.json` reports `final_observed_frames=301`,
      `final_prefix_frame_index=300`, `video_file_contract_ok=true`,
      `full_episode_length_ok=true`, final success `false`, and final
      peg-head-at-hole `[-0.077691, -0.002917, 0.000275]`. Framebook review
      of `live_observed_rollout_annotated` confirms the peg is still outside
      the hole at frame `300`. The loop used `25` receding iterations:
      early `scale_0.2` executor chunks first worsened x to about `-0.25m`,
      later recovered it near `-0.10m`, and then the real-state
      continuability gate repeatedly allowed DP handoff because the current
      x lower bound is `min_rel_x=-0.1342566`. Frozen DP then improved x only
      to about `-0.08m`, still far outside insertion. Plain blocker:
      the current `C_pi` handoff is too optimistic for dynamic takeover and
      the available executor candidates do not create a reliable insertion-axis
      corrective chunk. This is not a reason to hand-tune a looser/easier gate;
      it is evidence that handoff must be judged by real candidate outcomes
      and stronger contact/progress candidates.
- [ ] 2026-06-17 05:10+08 broader checkpoint-generated candidate replay
      finished on `server54` allocation `133179`.
      The 8-row generated-candidate replay at
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_generated_model_hard8_20260617_server54_alloc133179`
      wrote `104/104` outcomes with zero failures. On all `8` base rows,
      the oracle best candidate was a stochastic `model_diffusion_*` chunk,
      improving mean true weighted task-frame error from DP `0.326105` to
      `0.308835` (`0.017270` better), but no candidate reached final success
      (`0/8` oracle success). All non-DP model candidates were marked
      `selector_over_cap`, explaining why the current live selector keeps
      choosing tiny `scale_0.2` chunks instead of stochastic candidates.
      The 32-row replay with `--model-candidate-samples 16` at
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_generated_model_hard32_20260617_server54_alloc133179`.
      wrote `672/672` outcomes with zero failures. It improved mean true
      weighted task-frame error from DP `0.317845` to oracle-best
      `0.297342` (`0.020503` better), but again had zero successes:
      `success_total=0`, `dp_success=0`, `oracle_success=0`. All non-DP
      model candidates were still `selector_over_cap` (`400/400`). This
      means the old cap/scorer is too conservative, but cap repair alone is
      not enough because even an oracle over these stochastic chunks did not
      solve insertion. The next method work is a stronger contact/progress
      candidate generator or longer/phase-aware chunk supervision, with a
      short outcome-scorer smoke only as a code gate, not method evidence.
- [ ] 2026-06-17 04:50+08 extended
      `scripts/world_model/export_cosmos3_candidate_outcome_labels.py` so it
      can optionally load the formal candidate-executor checkpoint and replay
      the actual model-generated candidates (`model_mean`,
      `model_scale_*`, `model_diffusion_*`), not only DP/teacher residual
      scales. Compute-node syntax check passed. A one-row generated-candidate
      smoke at
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_labels_generated_model_smoke1_20260617_server54_alloc133179`
      replayed `7/7` model candidates with zero failures. On that row, the
      best true 8-step candidate was `model_diffusion_1`, improving weighted
      task error from DP `0.3312` to `0.3188`, but it was still not a task
      success and it was marked `selector_over_cap=true` by the old residual
      cap. This narrows the blocker: live currently selects small `scale_0.2`
      chunks because stochastic candidates are capped/scored out, while small
      scale chunks do not solve insertion. Before any new formal scorer/live
      run, replay a broader checkpoint-generated candidate set and check
      whether oracle model candidates have real task-frame headroom over DP;
      if not, the candidate generator/expert chunks must be strengthened.
- [ ] 2026-06-17 04:35+08 added
      `scripts/world_model/train_cosmos3_candidate_outcome_scorer.py`, an
      offline action-conditioned scorer trained from real replayed candidate
      outcomes instead of teacher-only future-state labels. Syntax was checked
      inside allocation `133179`, and a compute-node 5-step CPU smoke ran at
      `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_smoke_20260617_server54_alloc133179_cpu5step`.
      The smoke proves the join/train/eval path works (`288` candidate rows
      from `32` base rows), but it is negative method evidence: on held-out
      groups, DP prior true weighted error was `0.2923`, the oracle best over
      available candidates was only `0.2914`, and the learned selected
      candidates were worse at `0.2997` (`selected_minus_dp=+0.0074`). This
      means the current candidate set has almost no real headroom over DP on
      that split, so a formal scorer trained on the same candidate set should
      not be launched as the next method attempt. The next aligned repair is
      stronger dynamic task-frame/contact expert chunks or a broader
      candidate generator whose real 8-step outcomes actually improve over
      DP before live control.
- [ ] 2026-06-17 04:15+08 current blocker is now concrete, not a vague
      "training data vs eval" guess. The old formal candidate executor can
      execute full live rollouts on render-capable nodes, but it mostly selects
      `scale_0.2` DP-residual chunks. `sample_00_hole_late_move_stop`
      succeeded, but `sample_01_hole_late_constant` completed the full
      `300/301` contract and failed with final peg-head-at-hole
      `[-0.080696, -0.001325, 0.009701]`; the last eight iterations stayed
      near the same x/y state and failed only the z/insert-axis gate. Video
      framebook inspection showed the peg remained held but was not inserted,
      so this is not a metric-only artifact. A new next-state-scored formal
      attempt on allocation `133179` was intentionally stopped before the
      formal floor because it exposed an invalid training assumption: the
      next-state head was used to score arbitrary sampled candidates even
      though labels supervised only the teacher candidate outcome. Held-out
      progress/next-state metrics exploded, so continuing would only create a
      formal failure from an ungrounded scorer. The aligned repair is to train
      an action-conditioned outcome scorer from real candidate rollouts.
      Implemented `scripts/world_model/export_cosmos3_candidate_outcome_labels.py`
      and ran compute-node outcome exports on `server54`: a 2-row smoke wrote
      `14/14` candidate outcomes, hard32 wrote `224/224`, and hard32 bigscale
      wrote `192/192`, all with zero row failures. The hard32 evidence is
      negative for the current candidate set: over 8-step true simulator
      outcomes, DP prior has the best average weighted task-frame error;
      teacher/scale residuals only help a minority of lateral/preinsert rows,
      and large scales worsen average error with many clipped actions. This
      means the immediate blocker is not closed-loop plumbing and not the
      dense SFT data contract; it is that the current successful-DP-derived
      local candidate actions do not provide a strong corrective 8-step
      insertion move for hard post-motion states. Continue the running
      `sample_03`/`sample_04` diagnostics, but do not launch another formal
      executor training from the same ungrounded next-state scoring setup.
- [ ] 2026-06-16 14:20+08 current live blocker: the `4` GPU formal
      best-gate retry on allocation `131662` did complete the `10800` second
      floor on `server46`, wrote `training_summary.json`, and selected
      `checkpoint_best_gate.pt` as the valid live checkpoint. The gate source
      is `best_gate` at step `100`: selected action MSE
      `0.0015079 < 0.0015608` DP prior, `teacher_progress_mse=0.04146`,
      `teacher_value_mse=0.03786`, inserted acc `0.961`, and
      DP-continuable acc `0.857`. Gated live did launch, but no sample reached
      closed-loop evidence: `sample_00` died at the first render with
      `vk::Device::waitForFences: ErrorDeviceLost`, and parallel samples
      `01/03/04` failed before summaries with the same render/CUDA setup
      class. A render canary on `server46` with explicit
      `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` reached
      `render_rgb_array_start` and then timed out; a separate `server13`
      canary failed CUDA/Vulkan initialization. Therefore the current blocker
      is render-capable scheduling/runtime, not training data, not the offline
      gate, and not a measured closed-loop controller failure. A render-retry
      watcher was added:
      `scripts/slurm/watch_candidate_executor_live_render_retry_from_allocation.sh`.
      A general retry allocation `132888` reached `server61`, but its render
      canary also timed out. A later `server62` `1` GPU allocation `132981`
      passed live startup but the actual gated live wrapper still died on the
      first `env.render()` with `ErrorDeviceLost`, so the canary was not a
      false blocker. Current pending alternatives are `133177` (`server62`,
      `3` GPUs, to test whether another physical GPU on the same known
      previously live-capable node works), `133178` (`server24`, `1` GPU),
      `133179` (`server54`, `1` GPU), and `133180` (`server10`, `1` GPU).
      `server10`, `server24`, and `server54` all have previous live-summary
      evidence. First started allocation should run the
      same gated live path from `checkpoint_best_gate.pt`; cancel later
      pending alternatives to avoid idle resources. Do not change
      controller/eval gates to bypass video evidence. Log comparison found
      old successful `server62` live ran with the SAPIEN "failed to find
      Vulkan ICD" warning and still produced video, while the patched
      forced-ICD path failed; therefore live wrappers were changed again so
      they do not force `VK_ICD_FILENAMES` by default. Use
      `SET_NVIDIA_VK_ICD=true` only as an explicit diagnostic variant. A
      later general retry `133564` reached `server08`, but the canary again
      reached `render_rgb_array_start` and failed with
      `vk::Device::waitForFences: ErrorDeviceLost`; the allocation was
      released as render-unusable. Read-only comparison against the old
      successful `server62` path found another environment difference: the
      old path did not explicitly pass `DISPLAY=`, while the new retry wrapper
      did. The live/canary wrappers now preserve `DISPLAY` only if it is
      already non-empty; otherwise they leave it unset. This is a render
      runtime fix only, not a controller/eval change. A second general retry
      `133733` then reached `server52` with `display=null`, proving the
      `DISPLAY` fix took effect, but it still timed out at
      `render_rgb_array_start`; the job exited with `FAILED 124:0`. Therefore
      empty `DISPLAY` was not the sufficient cause of the render blocker.
      On 2026-06-17, fixed-node allocation `133178` did start on
      `server24`, proving the blocker was not simply "no resources". Its
      canary also timed out at first-frame render. A direct gated
      candidate-executor `sample_03` run first exposed a wrapper shell bug in
      `run_cosmos3_live_receding_panel_in_allocation.sh`; the
      `[[ ... || ... ]]` checkpoint guard was replaced by a `case`, and
      `bash -n` passed inside allocation `133178`. The rerun then cleared the
      formal/offline gate and entered real live eval, but stalled after
      `live_pretrigger_dp_loaded` for over 11 minutes with no new files while
      the GPU sat at 100%; this matches the canary first-render hang. Step
      `133178.3` was cancelled, and the `server24` allocation was released as
      render-unusable. Current pending attempts are `133177` (`server62`,
      `3` GPUs), `133179` (`server54`, `1` GPU), `133180` (`server10`,
      `1` GPU), plus new general tmux-held retries `135069` and `135070`
      excluding confirmed bad render nodes
      `server08,13,24,35,46,52,61,62`. The general retries auto-run the same
      render canary plus gated live path from `checkpoint_best_gate.pt` when
      Slurm grants them.
- [ ] 2026-06-16 10:10+08 current active state: the earlier `1` GPU
      allocation `131660` did start and completed the formal `10800` second
      floor, so the immediate blocker is no longer only Slurm. The 100-step
      diffusion smoke passed the offline gate
      (`teacher_progress_mse=0.0403`, `teacher_value_mse=0.0394`,
      inserted acc `0.961`, DP-continuable acc `0.870`, selected MSE
      `0.001508 < 0.001561` DP prior), but the 3-hour formal run's final
      checkpoint failed the scorer gate badly (`teacher_progress_mse=148981`,
      `teacher_value_mse=9217`, inserted acc `0.662`, DP-continuable acc
      `0.455`) even though selected action MSE was barely better than DP.
      JSONL train/val distribution is not obviously corrupted; progress
      labels remain bounded. Interpretation: the current failure is formal
      training/checkpoint-selection instability of the progress/contact/value
      scorer, not a closed-loop eval failure and not a known data-label
      outlier problem.
- [ ] 2026-06-16 10:10+08 repaired the formal-to-live gate without relaxing
      thresholds. `train_cosmos3_candidate_executor.py` now saves
      `checkpoint_best_gate.pt` whenever a held-out eval point passes the
      full offline gate, records `formal_live_eval_checkpoint`, and allows
      live only after the formal GPU/time floor is met plus either final or
      best-gate checkpoint satisfies the same metrics. The live launcher now
      loads the summary-selected checkpoint instead of hard-coding
      `checkpoint_final.pt`. The watcher formal config is reset to the
      smoke-validated stable model (`1024x4`, dropout `0.05`, lr `2e-4`,
      global batch about `128`) while still running for `10800` seconds.
      New active watcher:
      `cosmos3_candidate_diffusion_chain_watch_131662_4gpu_bestgate_20260616`.
      New formal root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_4gpu_alloc131662_diffusion_bestgate_stable_smokecfg`.
      It uses new best-gate chain markers so the old
      `candidate_executor_diffusion_chain_20260616.terminal` from the failed
      1GPU final-checkpoint run does not prematurely stop the 4GPU retry.
      Current Slurm state: only allocation `131662` remains, `PENDING
      (Priority)`, estimated start `2026-06-16T11:24:46` on `server38`.
- [ ] 2026-06-16 03:54+08 execution-boundary repair: moved the
      candidate-executor prelaunch audit out of the login-node watcher startup
      path. The watcher now only polls Slurm/tmux while the allocation is
      pending; once a tmux-held allocation becomes `RUNNING` and wins the
      shared launch mutex, it runs
      `audit_candidate_executor_diffusion_prelaunch.py` through
      `srun --overlap --jobid=...` inside the allocation before CUDA canary,
      smoke, formal training, or live eval. This preserves the user's rule
      that preflight/project-code checks run on compute nodes. Existing
      prelaunch audit files from the earlier login-node run are historical
      only and will be overwritten by the in-allocation audit before the chain
      launches. If `.formal_ready` already exists, a later live-only
      allocation skips the training prelaunch audit and proceeds to the live
      gate path.
- [ ] 2026-06-16 04:00+08 tightened the same execution boundary for status
      helpers. While allocations are pending, the watcher no longer invokes
      the Python chain summarizer on the login node; pending status is only
      the shell watch log plus Slurm state. After an allocation is `RUNNING`,
      `update_chain_status` calls
      `summarize_candidate_executor_diffusion_chain.py` through
      `srun --overlap --jobid=...` on the compute allocation. The post-live
      `watch_candidate_executor_post_gate_status.py` parser is also now run
      through `srun`, not directly on the login node. This keeps the login
      side limited to Slurm/tmux control-plane polling and local text logs.
- [ ] 2026-06-16 04:06+08 moved the remaining smoke/formal summary gate
      parser into the compute allocation as well. `summary_ready_field` now
      calls its JSON parser through `srun --overlap --jobid=...` after the
      allocation is `RUNNING`; if the allocation is still pending it returns
      `missing` without running Python. The three lightweight watcher sessions
      were restarted at `04:05:54+08`, and their latest logs again show only
      `prelaunch_audit_deferred_until_allocation_running` plus
      `allocation_wait state=PENDING reason=(Priority)`. Current estimates
      moved earlier: `131564` 2GPU at `2026-06-17T03:00:00`, `131660` 1GPU
      at `2026-06-17T04:00:00`, and `131662` 4GPU at
      `2026-06-17T20:00:00`.
      Existing `diffusion_chain_status.json` files with timestamp around
      `03:54` are historical artifacts from before the status-helper move;
      they are not refreshed while pending and will be overwritten only after
      a compute allocation starts.
- [ ] 2026-06-16 03:48+08 latest resource override confirmed in actual
      watcher state, not just notes. Formal candidate-executor training can
      start from the first valid tmux-held allocation among `1`, `2`, or `4`
      GPUs, but every formal run still requires `10800` seconds. Current
      pending jobs are `131660` (`1` GPU, estimated
      `2026-06-17T16:00:00`), `131564` (`2` GPUs, estimated
      `2026-06-17T11:00:00` at the latest check, but fluctuating), and
      `131662` (`4` GPUs, estimated
      `2026-06-18T04:00:00`), all `PENDING (Priority)`. The watcher logs
      show the real launch contracts:
      `formal_gpus=1 formal_nproc=1 formal_min_wall_seconds=10800`,
      `formal_gpus=2 formal_nproc=2 formal_min_wall_seconds=10800`, and
      `formal_gpus=4 formal_nproc=4 formal_min_wall_seconds=10800`.
      Therefore the current blocker is that no requested allocation is
      running yet, not that the code is still waiting only for two GPUs.
      If `server35` starts first, it may run the formal training floor, but
      live closed-loop eval is deferred through the `.formal_ready` marker
      because `server35` has prior live-render instability.
- [x] 2026-06-16 current continuation: hardened the candidate/diffusion
      formal-to-live gate so the executor branch actually matches the stated
      method: candidate/diffusion action chunks are selected by a
      progress/contact/value scorer. `train_cosmos3_candidate_executor.py`
      now includes `teacher_value_mse <= 0.25` in `offline_gate_from_eval`,
      alongside `teacher_progress_mse <= 0.05`,
      `teacher_inserted_acc >= 0.75`,
      `teacher_dp_continuable_acc >= 0.75`, selected-action MSE not worse
      than frozen DP, and non-DP selected candidates. The formal
      `training_summary.json` records these thresholds. The gated live
      launcher independently rechecks all these metrics before live rollout
      and writes `gate_metrics`/`gate_thresholds` into the live watch log;
      chain/post-gate status summaries now expose the scorer metrics directly.
      This is execution hardening only, not method evidence; the formal
      training allocations are still `PENDING (Priority)`.
- [ ] 2026-06-16 03:20+08 latest resource rule applied: formal
      candidate-executor training may now use `1`, `2`, or `4` GPUs, but it
      still must reserve/run for at least `10800` seconds. This removes the
      old "wait only for 2 GPUs" blocker without weakening the time floor.
      Requested tmux-held allocation alternatives are now:
      `131660` (`cand_diff_1gpu_0616`, `1` GPU),
      `131564` (`cand_diff_2gpu_0616`, `2` GPUs), and
      `131662` (`cand_diff_4gpu_0616`, `4` GPUs), all currently
      `PENDING (Priority)`. New watcher sessions
      `cosmos3_candidate_diffusion_chain_watch_131660_1gpu_20260616`,
      `cosmos3_candidate_diffusion_chain_watch_131564_2gpu_20260616`, and
      `cosmos3_candidate_diffusion_chain_watch_131662_4gpu_20260616` use a
      shared launch mutex, so whichever allocation starts first runs the
      smoke/formal/live chain and the others do not duplicate training. The
      restarted watchers also know their own allocation tmux session; if a
      later allocation starts after another watcher already holds the mutex,
      it sends `Ctrl-C` to its own allocation tmux session instead of idling
      below the GPU-utilization threshold.
      Prelaunch audit is ready for all three planned configs
      (`nproc_per_node=1/2/4`, `min_wall_seconds=10800`, `512` rows,
      feature contract `218 -> 56`). Current blocker is Slurm priority only,
      not data rows, static paths, or train/live feature mismatch.
- [x] 2026-06-16 current continuation: added a terminal marker to the
      direct diffusion watcher so later 1/2/4 GPU allocations do not repeat
      the same formal training after the first allocation reaches a formal
      conclusion. `candidate_executor_diffusion_chain_20260616.done` still
      means live success. New
      `candidate_executor_diffusion_chain_20260616.terminal` means the chain
      reached a terminal formal/live outcome, including formal gate failure or
      live completion with nonzero rc. Later watcher instances check both
      markers after their allocation starts and send `Ctrl-C` to their own
      allocation tmux session if the chain is already done/terminal. The
      three lightweight watcher sessions were restarted at `03:35+08` with
      this logic; allocation tmux sessions `131660`, `131564`, and `131662`
      remain pending.
- [x] 2026-06-16 current continuation: login-node read-only audit confirmed
      the live closed-loop path is wired to the intended method. When
      `controller_action_source=candidate_executor`, live execution loads
      `checkpoint_final.pt`, builds features from current state +
      Cosmos-predicted task path + frozen-DP prior chunk + live contact
      context, samples candidate chunks from `dp_prior`, mean/scale variants,
      and diffusion samples, then scores every candidate with the
      progress/contact/value scorer before executing the selected short chunk.
      Training and live selector caps use the same mean-squared residual
      quantity. The candidate live launcher manifest wording was updated from
      old `2GPU/3h` wording to the active `1/2/4-GPU plus 3-hour` rule.
      Static checks passed: live launcher `bash -n`, watcher `bash -n`, inline
      Python compile, `py_compile` for live loop/trainer/summarizers, and
      `git diff --check`.
- [x] 2026-06-16 01:50+08 closed the stale `128888` candidate-executor
      watcher state. Slurm now rejects allocation `128888` as expired/invalid,
      `squeue -u yanhongru` shows no active jobs, and the local watcher PIDs
      `522131`, `522145`, and `2110504` were stopped. The interrupted
      no-sample formal root
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_smallscales_nosample_direct_4096_finalgate`
      has `training_history.json` and `checkpoint_latest.pt`, but no
      `training_summary.json` and no `checkpoint_final.pt`. Last recorded
      step was `429000` with `elapsed_seconds=9133.67`, below the required
      `10800` second floor, followed by Slurm cancellation of step
      `128888.84`. This is a scheduling/resource failure and diagnostic
      trace only, not formal method evidence. Next action is a fresh
      tmux-held `2` GPU allocation, then the diffusion/candidate executor
      smoke/formal chain; do not resume stale non-diffusion live handoff from
      this invalid job.
- [ ] 2026-06-16 01:53+08 requested fresh tmux-held `2` GPU allocation
      `131564` (`cand_diff_2gpu_0616`, `1` day, excluding `server13`).
      Current state is `PENDING (Priority)`. Added and launched lightweight
      tmux watcher `cosmos3_candidate_diffusion_chain_watch_131564_20260616`
      using
      `scripts/slurm/watch_candidate_executor_diffusion_chain_from_allocation.sh`.
      The watcher does only login-safe polling until allocation `131564`
      becomes `RUNNING`; then it runs CUDA canary, diffusion smoke, formal
      `2` GPU / `3` hour diffusion candidate-executor training, and gated live
      panel only through `srun` inside that allocation. Roots are:
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_diffusion_smoke_20260616_alloc131564_rankcal`
      and
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_2gpu_alloc131564_diffusion_rankcal_finalgate`.
      Local checks passed: wrapper `bash -n`, related wrapper `bash -n`,
      Python `py_compile`, diffusion candidate-executor self-test, and
      `git diff --check` on the touched paths.
- [x] 2026-06-16 02:00+08 fixed diffusion-chain status wording for the new
      direct diffusion path. The existing summarizer still supported the old
      "wait for no-sample formal gate, then diffusion" chain, so when called
      with the direct 20260616 diffusion roots it reported
      `waiting_current_formal_gate`. It now detects
      `formal_root == formal_diffusion_root` as `chain_mode=direct_diffusion`
      and reports `waiting_diffusion_smoke` before smoke starts. This is
      evidence/status clarity only; it does not change training or gates.
- [ ] 2026-06-16 02:03+08 resource audit for `131564`: `scontrol show job`
      reports `JobState=PENDING`, `Reason=Priority`, `StartTime=2026-06-17T22:00:00`,
      and planned node `server34`. `server34` currently has all `8` H200 GPUs
      allocated. `gpux` is drained and `test` only allows `AllowAccounts=null`,
      so there is no obviously valid faster partition for the `mayi` account.
      Keep the tmux-held `gpu` allocation request and watcher active; do not
      start a 1-GPU smoke-only allocation that would finish quickly and then
      hold idle GPU time without satisfying the formal `2` GPU floor.
- [x] 2026-06-16 02:02+08 changed the new diffusion-chain watcher pending
      poll cadence from `60` seconds to `1800` seconds and restarted tmux
      session `cosmos3_candidate_diffusion_chain_watch_131564_20260616`.
      This keeps the login-node side from spinning on Slurm while allocation
      `131564` is pending. The allocation itself remains active in tmux; no
      Slurm resource was released or cancelled. Latest `scontrol show job`
      reports `JobState=PENDING`, `Reason=Priority`,
      `StartTime=2026-06-17T23:00:00`, and planned node `server03`.
- [x] 2026-06-16 02:04+08 completed login-safe prelaunch audit for the
      diffusion candidate-executor training input. Default
      `CONTACT_EXECUTOR_JSONL`
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_iter1500_train512_scale005_repair_20260615/contact_executor_dataset_file.jsonl`
      has `512` rows, all with
      `task_path_source=cosmos_predicted_action_sidecar`, no
      `gt_state_targets_debug`, and no missing `executor_sample_npz`,
      `dp_prior_npz`, `contact_label_npz`, or `source_h5` paths. Full
      `load_arrays` check produced `x=(512,218)`, `y=(512,56)`,
      `prior=(512,56)`, `progress=(512,2)`, and `binary=(512,2)`;
      the DP prior is loaded from `dp_prior_npz["dp_prior_actions"]`, so the
      empty `dp_prior_actions` field inside executor sample NPZs is not used
      by the trainer. This clears the input-path/shape gate before the 2GPU
      allocation starts.
- [x] 2026-06-16 02:08+08 completed login-safe prelaunch audit for the
      post-formal gated live panel dependencies. The clean-dense SFT root,
      formal iter1500 eval root, condition root, source H5 root, DP manifest,
      DP checkpoint, continuability stats, and SFT checkpoint directory
      `outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/checkpoints/iter_000001500`
      all exist. This means if the diffusion candidate-executor formal summary
      passes, the next expected blocker should be actual live execution quality
      or runtime execution errors, not missing static path inputs.
- [x] 2026-06-16 02:10+08 completed login-safe live/training feature-width
      audit for the candidate executor. Training `load_arrays` produced
      feature width `218`; live `load_candidate_executor_checkpoint` derives
      horizon from `target_dim / 7`, so the expected formal checkpoint with
      `target_dim=56` uses horizon `8`. Live feature construction then matches
      training composition: current state `35` + task path `8*14` + DP prior
      `8*7` + live contact context `15` = `218`. The live path also keeps the
      hard runtime guard that raises if feature width differs from checkpoint
      `feature_dim`.
- [x] 2026-06-16 02:48+08 converted the prelaunch checks into a reusable
      login-safe script:
      `scripts/world_model/audit_candidate_executor_diffusion_prelaunch.py`.
      It reads the same `load_arrays` path used by the formal diffusion
      candidate-executor trainer, checks the 512-row Cosmos-predicted task-path
      dataset, verifies no GT-debug task paths or missing sample/prior/contact
      files, checks the `218` feature-width contract against the live executor
      layout, verifies the clean-dense Cosmos/DP live dependencies, and checks
      the planned diffusion formal config (`2` GPU, `10800` seconds,
      `candidate_samples=8`, `candidate_rank_diffusion_count=1`). Latest
      output:
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_2gpu_alloc131564_diffusion_rankcal_finalgate/prelaunch_audit.json`
      and `.md`, with `ready_for_allocation_launch=true` and
      `blocker_class=slurm_resource_pending`. This is not method evidence; it
      means the current known blocker is resource availability, not bad rows,
      missing static paths, or train/live feature mismatch.
- [x] 2026-06-16 02:50+08 wired the reusable prelaunch audit into
      `scripts/slurm/watch_candidate_executor_diffusion_chain_from_allocation.sh`
      and restarted only the lightweight watcher session
      `cosmos3_candidate_diffusion_chain_watch_131564_20260616`. The watcher
      now refuses before waiting/launching compute if the audit fails. Latest
      watcher log shows `prelaunch_audit_ready=true`, then
      `allocation_wait state=PENDING reason=(Priority)`. The allocation tmux
      session `cosmos3_candidate_diffusion_2gpu_20260616` was not cancelled or
      released.
- [x] 2026-06-16 02:51+08 updated
      `scripts/world_model/summarize_candidate_executor_diffusion_chain.py` so
      `diffusion_chain_status.json/.md` reports the prelaunch audit result.
      Current status remains `overall_status=waiting_diffusion_smoke`, but it
      now also records `prelaunch_audit_ready=True`,
      `prelaunch_blocker_class=slurm_resource_pending`,
      `prelaunch_rows=512`, and `prelaunch_feature_contract_ok=True`.
- [x] 2026-06-16 02:53+08 added live post-gate status writeback to
      `scripts/slurm/watch_candidate_executor_diffusion_chain_from_allocation.sh`.
      After formal diffusion live panel finishes, the watcher now runs
      `scripts/world_model/watch_candidate_executor_post_gate_status.py`
      against `candidate_after_gate_live_watch.log` and writes
      `post_gate_status.json/.md`, so the chain status can see the live output
      root, panel summary, contract flag, final success count, and contact
      sheet path. Restarted only the lightweight watcher; latest log shows
      `prelaunch_audit_ready=true` and then
      `allocation_wait state=PENDING reason=(Priority)`. Slurm's estimated
      start time has fluctuated between `2026-06-17T20:00:00` and
      `2026-06-17T22:00:00`; latest `scontrol` at `02:53:40+08` reports
      `StartTime=2026-06-17T22:00:00` and scheduled node `server34`.
- [x] 2026-06-16 02:56+08 fixed a post-gate status hang in the direct
      diffusion watcher. The watcher wrote the live return code as
      `formal_diffusion_gated_live_rc` in the chain log, while
      `watch_candidate_executor_post_gate_status.py` looked for
      `candidate_after_gate_live_rc` in the live log. That could make the
      post-gate status loop wait forever after live completion. The watcher
      now appends `candidate_after_gate_live_rc=${live_rc}` to
      `candidate_after_gate_live_watch.log`, and the parser accepts both rc
      names. A temporary artifact test confirmed the parser reaches terminal
      states for both `live_panel_summary_available_needs_video_review` and
      `live_finished_without_panel_summary`. Restarted only the lightweight
      watcher; allocation `131564` remains `PENDING (Priority)` and current
      estimate is `2026-06-17T20:00:00`.
- [x] 2026-06-16 02:57+08 expanded post-gate/live evidence visibility. The
      post-gate status now records requested/completed live samples,
      final-success count, failed-process count, full-episode contract flag,
      contact-sheet path, visual-review status, and method-evidence flag from
      `live_receding_panel_summary.json`. The chain summarizer now exposes the
      live rc, success count, contract flag, and contact-sheet path in
      `diffusion_chain_status.md`. A temporary artifact test confirmed those
      fields are parsed from a live panel summary. Current chain status still
      correctly reports `overall_status=waiting_diffusion_smoke` because
      allocation `131564` has not started.
- [x] 2026-06-16 03:00+08 corrected the diffusion smoke gate semantics.
      The 50-100 step smoke run is an interface/runtime gate, not a method
      quality gate, so it should not block formal training under the active
      1/2/4-GPU plus 3-hour resource rule just
      because `ready_for_offline_gate=false` or selected-action MSE is still
      worse than frozen DP after the short smoke. The watcher now requires
      smoke artifacts to prove the diffusion path ran (`generator_type=diffusion`,
      positive `candidate_samples`, positive `candidate_rank_diffusion_count`,
      finite selected/DP MSE, positive step count, and multiple candidate
      sources). The strict quality gate remains on formal training:
      `formal_training_floor_met=true`, `ready_for_formal_live_eval=true`,
      selected MSE not worse than DP, and non-DP selected candidates. The chain
      summarizer now reports `diffusion_smoke_interface_ready` separately from
      `diffusion_smoke_ready_for_offline_gate`. A temporary summary test
      confirmed a smoke summary with `ready_for_offline_gate=false` but a valid
      diffusion interface produces
      `overall_status=diffusion_smoke_ready_waiting_formal_diffusion`.
      Restarted only the lightweight watcher; allocation `131564` remains
      `PENDING (Priority)`, estimated `2026-06-17T20:00:00`.
- [x] 2026-06-16 03:03+08 hardened the formal live launcher checkpoint gate.
      `run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`
      now verifies that `checkpoint_final.pt` agrees with the formal summary
      on `generator_type`, has positive diffusion candidate metadata, has
      `target_dim` divisible by robot action dim `7`, and satisfies the live
      feature contract `feature_dim = 35 + horizon*14 + target_dim + 15`
      (`218` when `target_dim=56`, horizon `8`). This prevents a mismatched or
      stale checkpoint from entering live closed-loop eval even if a summary
      file exists. Local temporary checkpoint test accepted `218/56` and
      rejected feature dim `217`; `bash -n` and `git diff --check` passed.
      Restarted only the lightweight watcher; allocation `131564` remains
      `PENDING (Priority)`, estimated `2026-06-17T20:00:00`.
- [x] 2026-06-16 03:05+08 hardened the formal candidate trainer's final
      checkpoint self-check. `train_cosmos3_candidate_executor.py` now uses
      `torch.load(..., weights_only=False)` when reloading
      `checkpoint_final.pt` and `checkpoint_best_offline.pt` immediately after
      saving. The checkpoint payload stores numpy normalization arrays, so an
      explicit non-weights-only load avoids a late formal-run failure if torch
      defaults change. Local torch is `2.5.1+cu121`; a temporary numpy-payload
      checkpoint load test passed. This is execution robustness only; no
      training or method evidence was produced.
- [x] 2026-06-16 03:08+08 added formal summary/checkpoint dimension
      consistency checks. `train_cosmos3_candidate_executor.py` now writes
      `feature_dim`, `target_dim`, and `action_horizon` into
      `training_summary.json`. The gated live launcher compares those summary
      values against `checkpoint_final.pt` when present, in addition to the
      checkpoint's own feature-contract check. `summarize_candidate_executor_diffusion_chain.py`
      now exposes these dimensions in `diffusion_chain_status.md`. A temporary
      summary/checkpoint test accepted `feature_dim=218`, `target_dim=56`,
      `action_horizon=8` and rejected mismatched `action_horizon=7`.
      Restarted only the lightweight watcher; allocation `131564` remains
      `PENDING (Priority)`, estimated `2026-06-17T20:00:00`.
- [x] 2026-06-16 03:11+08 hardened smoke/formal summary root matching against
      stale artifacts. `watch_candidate_executor_diffusion_chain_from_allocation.sh`
      now rejects a smoke/formal `training_summary.json` whose `output_root`
      does not resolve to the directory that contains that summary. This
      prevents an old copied or reused summary/checkpoint pair from allowing
      formal training or live evaluation in the current direct-diffusion chain.
      `summarize_candidate_executor_diffusion_chain.py` now reports
      `summary_output_root_matches` for the current formal and smoke roots. A
      temporary summary test accepted a matching root and rejected a mismatched
      root. Restarted only the lightweight watcher; allocation `131564`
      remains `PENDING (Priority)`, estimated `2026-06-17T20:00:00`.
- [x] 2026-06-15 23:50+08 restarted only the lightweight after-gate
      diffusion watcher so it now uses the latest diffusion/non-DP gate
      hardening. The previous tmux pane exited after `Ctrl-C`, so a new tmux
      session with the same name was created. New watcher PID is `1625010`;
      the log has fresh `diffusion_after_gate_watch_start` at
      `23:50:18+08`. Formal training step `128888.84` stayed active on
      `server33` and was not interrupted.
- [ ] 2026-06-15 23:52+08 current no-sample formal run is still pre-floor:
      latest step `376000`, elapsed `7918.49` seconds, about `2881.51`
      seconds remaining to the `10800` second floor. Latest selected-action
      MSE is `0.00152806` versus frozen-DP prior `0.00156083`; this is a
      better current point but not final evidence before `training_summary`
      and `checkpoint_final`. Diffusion smoke/formal roots remain gated and
      unlaunched. GPU spot check on allocation `128888` showed `62%/64%`
      utilization, so the held resource is active.
- [x] 2026-06-15 23:54+08 removed the remaining non-diffusion live
      auto-launch path. The old guarded-live watcher PID `271336` was stopped
      while leaving the post-gate status watcher and formal training active.
      `watch_candidate_executor_gate_then_diffusion_smoke.sh` now continues to
      diffusion smoke after the current post-floor formal decision even if the
      no-sample candidate baseline passes, instead of handing off to a
      non-diffusion live panel. Restarted lightweight diffusion watcher with
      PID `1774777`; formal training step `128888.84` stayed active.
- [x] 2026-06-15 23:59+08 removed duplicate diffusion watchers and restarted
      a single lightweight watcher, PID `2021025`. Formal training step
      `128888.84` stayed active.
- [x] 2026-06-16 00:02+08 removed another duplicate diffusion watcher pair
      and restarted a single lightweight watcher, PID `2110504`. Formal
      training step `128888.84` stayed active.
- [x] 2026-06-15 23:42+08 hardened the after-gate diffusion chain against
      stale non-diffusion summaries. `watch_candidate_executor_gate_then_diffusion_smoke.sh`
      now requires smoke/formal summaries to report `generator_type=diffusion`,
      `candidate_samples>0`, and `candidate_rank_diffusion_count>0` before
      treating them as ready. The gated live launcher now has
      `REQUIRE_DIFFUSION_GENERATOR=true` for the formal-diffusion live path
      and independently refuses summaries that are not diffusion or have
      diffusion candidates disabled. Local refusal checks passed for a
      gaussian summary (`reason=generator_type_not_diffusion`) and a diffusion
      summary with `candidate_samples=0`
      (`reason=diffusion_candidate_metadata_disabled`). This prevents a stale
      gaussian/no-sample candidate root from being mistaken for the intended
      diffusion action-chunk generator.
- [x] 2026-06-15 23:42+08 refreshed chain status after the diffusion-only
      gate hardening. Current no-sample formal run is still pre-floor:
      `waiting_current_formal_gate`, latest step `350000`, elapsed
      `7334.35` seconds, about `3465.65` seconds remaining to the `10800`
      second floor, selected-action MSE `0.00161744` versus frozen-DP prior
      `0.00156083`, recent window `24/24` worse than DP. Latest source counts
      are `mean=9`, `scale_0.05=3`, `scale_0.2=65`. Diffusion smoke/formal
      roots are still gated and unlaunched.
- [x] 2026-06-15 23:43+08 verification after diffusion-only gate hardening
      passed: py-compile for trainer/self-test/summarizer/live Python files,
      `bash -n` for the candidate watch/live/smoke/train wrappers, the
      diffusion candidate-executor self-test, the local gaussian/disabled
      diffusion refusal checks, and `git diff --check`. Allocation `128888`
      still has formal step `128888.84` active at about `123` minutes;
      overlap GPU spot check showed H200 utilization `29%/36%` and
      `4459 MiB` on both GPUs. The transient `128888.95 nvidia-s` Slurm step
      was only this spot-check command, not a new experiment.
- [x] 2026-06-15 23:36+08 hardened the candidate/diffusion offline and
      live gates against a pure-DP fake pass. `offline_gate_from_eval` now
      requires at least one selected non-`dp_prior` candidate in addition to
      progress/contact readout quality and `selected_action_mse <=
      dp_prior_action_mse`; candidate eval summaries now record
      `selected_non_dp_candidate_count`, fraction, and selected diffusion
      count. The after-gate watcher and gated live launcher independently
      reject summaries whose final `candidate_source_counts` collapse to pure
      `dp_prior`. A local fake-summary check confirmed the live launcher
      returns rc `44` with
      `reason=final_selected_candidate_collapsed_to_dp_prior`. This preserves
      the method boundary: a passing candidate-executor run must actually
      select a non-DP candidate somewhere offline before live evaluation.
- [x] 2026-06-15 23:36+08 refreshed chain status after the source-count
      summary repair. Current no-sample formal run remains pre-floor:
      `waiting_current_formal_gate`, latest step `333000`, about `3786.86`
      seconds remaining to the `10800` second floor, latest selected-action
      MSE `0.00163718` versus frozen-DP prior `0.00156083`, recent window
      `21/24` worse than DP. Latest source counts are
      `mean=7`, `scale_0.05=2`, `scale_0.1=1`, `scale_0.2=67`, so the current
      old no-sample run is not pure-DP fallback; it is still selecting
      non-DP candidates that are usually worse than DP.
- [x] 2026-06-15 23:39+08 verification after the non-DP gate hardening
      passed: py-compile for trainer/self-test/summarizer/live Python files,
      `bash -n` for the candidate watch/live/smoke/train wrappers, the
      diffusion candidate-executor self-test, and `git diff --check` for
      touched paths. Allocation `128888` still has formal step `128888.84`
      active at about `119` minutes; overlap GPU spot check showed H200
      utilization `17%/83%` and `4459 MiB` on both GPUs. No training process
      was interrupted.
- [x] 2026-06-15 23:31+08 improved diffusion-chain observability without
      changing gates. `summarize_candidate_executor_diffusion_chain.py` now
      reads `run_manifest.txt`, records `configured_min_wall_seconds`, and
      reports `formal_floor_remaining_seconds` in
      `diffusion_chain_status.json/.md`. Manual refresh shows the current
      no-sample formal run still `waiting_current_formal_gate`, latest step
      `319500`, elapsed `6710.30` seconds, about `4089.70` seconds remaining
      to the `10800` second floor, selected-action MSE `0.00164796` versus
      frozen-DP prior `0.00156083`, and `24/24` recent evals worse than DP.
      Verification passed: summarizer `py_compile` and `git diff --check`.
- [x] 2026-06-15 23:33+08 final verification for this continuation passed:
      py-compile for the summarizer/self-test/live/trainer Python files,
      `bash -n` for the candidate train/smoke/watch/live wrappers,
      the diffusion candidate-executor self-test, and `git diff --check` for
      touched paths. Current allocation `128888` still has formal step
      `128888.84` active; overlap `nvidia-smi` showed H200 utilization
      `33%/31%` and `4459 MiB` on both GPUs.
- [ ] 2026-06-15 23:27+08 current no-sample formal step `128888.84`
      remains active at about `108` minutes, still pre-floor. Refreshed chain
      status is `waiting_current_formal_gate`; latest step `306500` has
      selected-action MSE `0.00165272` versus frozen-DP prior `0.00156083`,
      with the latest trend window still `24/24` validation points worse than
      DP. Diffusion smoke/formal remain gated until this run writes final
      artifacts and the post-floor gate formally decides.
- [x] 2026-06-15 23:22+08 strengthened the login-safe diffusion
      candidate-executor self-test so it now exercises the actual live
      `candidate_executor_action_chunk` path, not only checkpoint loading.
      The test builds a shape-matched toy diffusion checkpoint with causal
      current state, Cosmos task path, frozen-DP prior chunk, and live contact
      context, then verifies finite selected actions, `diffusion_*` candidate
      records, rank metadata, and a `2x7` denormalized action chunk. Checks
      passed: `py_compile` for the self-test/live/trainer files,
      `.venv/bin/python scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py`,
      and `git diff --check` for the touched self-test path. This is interface
      evidence only; it does not use GPU and does not claim method success.
- [ ] 2026-06-15 23:22+08 current no-sample formal step `128888.84`
      remains active at about `103` minutes, still pre-floor. Refreshed chain
      status is `waiting_current_formal_gate`; latest step `291500` has
      selected-action MSE `0.00163363` versus frozen-DP prior `0.00156083`,
      and the latest trend window is still `24/24` validation points worse
      than DP. Diffusion smoke/formal roots remain gated and unlaunched until
      the current formal final artifacts exist and the post-floor gate
      formally refuses live.
- [x] 2026-06-15 23:24+08 allocation `128888` resource hygiene check:
      the formal training step is still active on `server33`; a corrected
      overlap `nvidia-smi` query with GPU visibility showed H200 utilization
      `24%/49%` and `4459 MiB` on each GPU. This is a spot check only, but it
      confirms the held allocation is not idle while waiting for the formal
      floor. A prior `--gres=gpu:0` query saw no devices by construction and
      is not a training failure.
- [x] 2026-06-15 23:26+08 audited the future candidate-executor live
      launcher against the formal-training discipline. Current
      `run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`
      defaults `EXECUTOR_CHECKPOINT` to the current formal root's
      `checkpoint_final.pt`, checks `formal_training_floor_met=true`,
      `ready_for_formal_live_eval=true`, verifies final selected-action MSE is
      no worse than the frozen DP prior, and loads that final checkpoint
      before launch. Updated the evidence note wording so early
      `checkpoint_best_offline.pt` files are clearly diagnostic only and
      cannot launch live evaluation.
- [x] 2026-06-15 23:28+08 closed the remaining candidate live-launcher
      override loophole. The launcher now refuses if `EXECUTOR_CHECKPOINT`
      resolves to anything other than `${FORMAL_ROOT}/checkpoint_final.pt`, so
      an environment override cannot substitute `checkpoint_best_offline.pt`
      after the formal gate. Verification passed: wrapper `bash -n`,
      `git diff --check`, and a local refusal-path check where a synthetic
      `checkpoint_best_offline.pt` returned rc `44` with
      `reason=executor_checkpoint_not_formal_final`.
- [x] 2026-06-15 23:12+08 added a post-failure handoff settle guard to
      the diffusion watcher. If the current formal gate fails after the
      required floor, `watch_candidate_executor_gate_then_diffusion_smoke.sh`
      now waits `POST_FAIL_SETTLE_SECONDS=75` before launching diffusion
      smoke, then rechecks allocation steps. This gives the older gated-live
      watcher time to run its expected refusal path and avoids racing a
      diffusion smoke against a short refused live-launch step. This changes
      scheduling hygiene only; it does not change any gate or metric.
      Verification passed: watcher `bash -n` and `git diff --check`. The
      lightweight watcher was restarted; active training step `128888.84` was
      not interrupted.
- [ ] 2026-06-15 23:12+08 current no-sample formal step `128888.84`
      remains active at about `93` minutes, still pre-floor. Refreshed chain
      status is `waiting_current_formal_gate`; latest step `263500` has
      selected-action MSE `0.00165816` versus DP `0.00156083`
      (`+9.73e-05`). Diffusion smoke/formal roots remain empty.
- [x] 2026-06-15 23:13+08 improved chain-status diagnostics with a recent
      eval trend window. `diffusion_chain_status.json/.md` now records the
      latest 24 validation deltas, including how many selected-action MSE
      points are worse than the frozen DP prior. Manual refresh at step
      `267000` showed `24/24` recent evals worse than DP, with mean selected
      minus DP MSE `+8.54e-05` and range `+5.25e-05` to `+1.48e-04`. This is
      still pre-floor evidence only; the final gate remains authoritative.
- [x] 2026-06-15 23:08+08 closed a remaining diffusion scorer
      train/eval mismatch before post-gate smoke starts. The future diffusion
      candidate executor's rank-calibration loss now includes
      `candidate_rank_diffusion_count` real denoised diffusion candidates
      when `generator_type=diffusion`, not only DP/mean/scale/random
      residual candidates. Smoke/formal diffusion watcher launches pass
      `CANDIDATE_RANK_DIFFUSION_COUNT=1`; live metadata and chain summaries
      preserve the field for audit. This is a generic scorer-alignment fix,
      not an enumerated recovery rule. Checks passed: `py_compile`,
      wrapper `bash -n`, CPU diffusion/gaussian self-test, and
      `git diff --check`. The lightweight diffusion watcher was restarted
      afterward; active training step `128888.84` was not interrupted.
- [ ] 2026-06-15 23:08+08 current no-sample formal step `128888.84`
      remains active at about `90` minutes, still pre-floor. Refreshed chain
      status is `waiting_current_formal_gate`; latest step `254000` has
      selected-action MSE `0.00170371` versus DP `0.00156083`
      (`+1.43e-04`). Diffusion smoke/formal roots are still empty, as
      expected before the current final gate.
- [x] 2026-06-15 23:00+08 improved the diffusion/candidate chain-status
      monitor and restarted only the lightweight watcher. The summarizer now
      records the latest `training_history.json` validation point when final
      `training_summary.json` is still missing and also embeds
      `post_gate_status.json`, so `diffusion_chain_status.json/.md` exposes
      the current selected-vs-DP trend and gate status instead of only
      `null` final metrics. Tmux session
      `cosmos3_candidate_diffusion_after_gate_watch_20260615` was restarted
      to load the current script; this did not touch the active training step
      `128888.84`. Verification passed: `py_compile` for the summarizer,
      watcher `bash -n`, and `git diff --check`.
- [ ] 2026-06-15 23:00+08 current no-sample formal step `128888.84`
      remains active at about `84` minutes, still pre-floor and without
      `training_summary.json` or `checkpoint_final.pt`. The refreshed chain
      status is `waiting_current_formal_gate`. Latest recorded validation
      point is step `236000`: selected-action MSE `0.00163644` versus DP
      `0.00156083` (`+7.56e-05`). The fixed diffusion smoke and formal
      diffusion roots are still empty, so no post-gate diffusion compute has
      launched yet.
- [x] 2026-06-15 22:48+08 added a lightweight chain-status summarizer:
      `scripts/world_model/summarize_candidate_executor_diffusion_chain.py`.
      It reads the current no-sample formal root, the fixed diffusion-smoke
      root, the fixed formal-diffusion root, and the diffusion watcher log,
      then writes `diffusion_chain_status.json/.md` under the current formal
      root. Current status is `waiting_current_formal_gate`; smoke/formal
      diffusion roots have no summaries/checkpoints, and the watcher log has
      not launched smoke/formal/live. This is observability only, not method
      evidence. Checks passed: `py_compile`, wrapper `bash -n`, and
      `git diff --check`.
- [ ] 2026-06-15 22:48+08 current no-sample formal step `128888.84`
      remains active at about `69` minutes, still pre-floor and without
      final summary/checkpoint. Latest step `195000` has selected-action MSE
      `0.00161908` versus DP `0.00156083` (`+5.82e-05`). The old
      Gaussian/no-rank-calibration selector remains worse than DP, but final
      gate evidence is still pending.
- [ ] 2026-06-15 22:42+08 current no-sample formal step `128888.84`
      remains active at about `63` minutes, still pre-floor. The latest 24
      validation evals are all worse than the frozen DP prior; latest step
      `178500` has selected-action MSE `0.00160406` versus DP `0.00156083`
      (`+4.32e-05`), with recent deltas ranging from `+1.57e-05` to
      `+1.54e-04`. This still cannot be called formal failure before
      `training_summary.json`/`checkpoint_final.pt`, but it makes a final
      gate refusal very likely. Light overlap GPU check on `server33` showed
      utilization `26%/39%` and memory `4459 MiB` on both GPUs, so the held
      allocation is still doing training work and should not be interrupted.
- [x] 2026-06-15 22:44+08 improved live audit metadata for the future
      diffusion/candidate executor path. `load_candidate_executor_checkpoint`
      now preserves `candidate_rank_loss_weight`,
      `candidate_rank_random_count`, and `candidate_rank_temperature` from
      checkpoint args, and each future `candidate_executor_action_chunk.json`
      records `candidate_samples` plus those rank-calibration settings along
      with `generator_type`, `diffusion_steps`, and `candidate_scales`. This
      does not change action selection; it makes later live videos/logs
      auditable for whether they actually used the rank-calibrated diffusion
      executor. Checks passed: `py_compile`, diffusion/gaussian CPU self-test,
      wrapper `bash -n`, and `git diff --check`.
- [ ] 2026-06-15 22:40+08 current no-sample formal step `128888.84`
      remains active at about `61` minutes and is still pre-floor. There is
      still no `training_summary.json` or `checkpoint_final.pt`;
      `post_gate_status.json` remains `waiting_training_final_artifacts`.
      The latest checked 16 validation points are all worse than the frozen
      DP prior; latest step `172500` has selected-action MSE `0.00163471`
      versus DP `0.00156083` (`+7.39e-05`). This is strong pre-floor
      evidence of selector drift in the old Gaussian/no-rank-calibration run,
      but not formal failure until the post-`10800` second final gate.
- [ ] 2026-06-15 22:40+08 fixed diffusion smoke/formal roots are still empty,
      so the after-gate diffusion watcher has not prematurely started compute:
      `candidate_executor_diffusion_smoke_after_nosample_gate_rankcal_20260615`
      and
      `candidate_executor_train_20260615_formal_2gpu_server33_diffusion_rankcal_after_nosample_gate`.
      Tmux watcher `cosmos3_candidate_diffusion_after_gate_watch_20260615`
      is alive and polling once per minute. The old gated-live watcher may
      briefly call the live launcher after the current final summary appears,
      but that launcher independently refuses if the formal gate fails; the
      diffusion watcher then proceeds after non-extern Slurm steps clear.
- [x] 2026-06-15 22:31+08 repaired the next diffusion/candidate trainer
      against the concrete selector-drift failure: the scorer is no longer
      trained only on the teacher residual and then asked to rank unseen
      `dp_prior`/`mean`/`scale`/random candidates at eval time. Added a
      generic candidate-ranking calibration loss to
      `scripts/world_model/train_cosmos3_candidate_executor.py`: each batch
      now builds DP-prior, generator-mean, small-scale, and random residual
      candidates, computes the same score shape used by offline/live
      selection, and applies cross-entropy toward the candidate closest to the
      teacher residual. This is not a case-specific recovery rule; it aligns
      the scorer's training task with the candidate-selection task. Future
      diffusion smoke/formal runs launched by the watcher explicitly pass and
      record `CANDIDATE_RANK_LOSS_WEIGHT=0.35`,
      `CANDIDATE_RANK_RANDOM_COUNT=4`, and
      `CANDIDATE_RANK_TEMPERATURE=1.0`. Checks passed: `py_compile`,
      diffusion/gaussian CPU self-test including the new rank loss,
      wrapper `bash -n`, and `git diff --check`.
- [x] 2026-06-15 22:35+08 hardened the after-gate watcher against duplicate
      restarts. Its default diffusion smoke output root is now the fixed path
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_diffusion_smoke_after_nosample_gate_rankcal_20260615`,
      and its full formal diffusion output root is the fixed path
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_diffusion_rankcal_after_nosample_gate`.
      This prevents a watcher restart from creating a new timestamped smoke
      or formal directory and wasting held GPU time. The watcher was restarted
      in tmux session
      `cosmos3_candidate_diffusion_after_gate_watch_20260615`; only one
      watcher process remains after removing the temporary foreground debug
      process.
- [x] 2026-06-15 22:38+08 added explicit summary metadata for the repaired
      diffusion/candidate executor path. Future
      `training_summary.json` files now record top-level `generator_type`,
      `diffusion_steps`, `candidate_samples`, `candidate_scales`,
      `candidate_rank_loss_weight`, `candidate_rank_random_count`, and
      `candidate_rank_temperature`, in addition to the manifest/checkpoint
      args. This makes the post-gate smoke/formal evidence auditable without
      loading the checkpoint. Checks passed: `py_compile`, diffusion/gaussian
      CPU self-test, wrapper `bash -n`, and `git diff --check`.
- [ ] 2026-06-15 22:31+08 current no-sample formal step `128888.84` is
      still active at about `52` minutes and remains pre-floor. Latest eval
      at step `150000` is still worse than the frozen DP prior:
      selected-action MSE `0.00163977` versus DP `0.00156083`
      (`+7.89e-05`). The current running process was launched before the
      candidate-ranking calibration loss existed, so this monitoring result
      should not be used to judge the repaired diffusion/candidate trainer.
- [ ] 2026-06-15 22:22+08 latest nosample formal-run check: Slurm step
      `128888.84` is still active at about `43` minutes, so it is still
      pre-floor and not formal failure evidence. The latest eval has now
      drifted clearly above the frozen DP prior: step `120500` selected-action
      MSE `0.00164551` versus DP `0.00156083` (`+8.47e-05`). The recent
      pattern is no longer a one-point fluctuation, so the likely final-gate
      outcome is refusal unless later training recovers. Do not launch live
      from this checkpoint unless the post-`10800` second final summary passes
      the hardened gate.
- [x] 2026-06-15 22:22+08 added and launched a lightweight after-gate
      diffusion-smoke watcher:
      `scripts/slurm/watch_candidate_executor_gate_then_diffusion_smoke.sh`,
      tmux session
      `cosmos3_candidate_diffusion_after_gate_watch_20260615`. It only polls
      `post_gate_status.json` and does not use GPU while the current formal
      run is active. If the current formal run reaches the 2GPU/3h floor and
      the final gate refuses live, it waits for non-extern Slurm steps in
      allocation `128888` to exit, then launches the compute-node-only
      diffusion candidate smoke wrapper with `GENERATOR_TYPE=diffusion`,
      `CANDIDATE_SAMPLES=8`, and `CANDIDATE_SCALES=0.05,0.1,0.2`. If the
      current final gate passes, it exits and lets the existing live watcher
      handle closed-loop eval.
- [x] 2026-06-15 22:27+08 upgraded and restarted that watcher so it does not
      leave the held GPUs idle after a passing smoke. New gate chain:
      current no-sample formal gate failure after floor -> diffusion 50-100
      step smoke; if smoke `ready_for_offline_gate=true` and final selected
      MSE is no worse than DP, launch a full `2` GPU / `10800` second formal
      diffusion candidate-executor run; if that formal final gate passes,
      launch the existing gated candidate-executor live panel. If any gate
      fails, the watcher stops and records the concrete reason. Current tmux
      pane confirms the new boundary text is active.
- [x] 2026-06-15 22:21+08 verified the diffusion launch path after the
      earlier candidate-scale propagation bug: the direct compute-node
      trainer wrapper now explicitly passes `--generator-type`,
      `--diffusion-steps`, `--diffusion-beta-*`,
      `--diffusion-loss-weight`, `--candidate-samples`, and
      `--candidate-scales` into
      `scripts/world_model/train_cosmos3_candidate_executor.py`. Checks
      passed: `bash -n` for the candidate train/live/diffusion-watch wrappers,
      `py_compile` for the trainer/live/self-test/status scripts,
      diffusion/gaussian CPU self-test, and `git diff --check`.
- [ ] 2026-06-15 22:14+08 latest nosample spot check: Slurm step
      `128888.84` is still active at about `35` minutes. Latest eval at step
      `98500` is slightly worse than DP prior: selected-action MSE
      `0.00156419` versus DP `0.00156083` (`+3.36e-06`). This reinforces the
      final-gate risk noted at 22:12, but it is still not post-floor formal
      failure evidence. Keep the run to the formal floor; the hardened final
      gate must refuse live if the final selected-action MSE is worse than DP.
      `post_gate_status.json` still reports `waiting_training_final_artifacts`.
- [ ] 2026-06-15 22:12+08 latest nosample spot check: Slurm step
      `128888.84` is still active at about `33` minutes. Latest eval at step
      `94500` has selected-action MSE `0.00153442` versus DP prior
      `0.00156083`, but the latest 10 evals include 3 selected-action MSE
      points worse than DP (`step 91000`, `93500`, `94000`). This is not a
      post-floor failure yet, but it raises final-gate risk; do not stop the
      formal run before the required floor, and let the hardened final gate
      decide live eligibility. No final summary/checkpoint or live panel
      exists yet.
- [x] 2026-06-15 22:13+08 extended
      `scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py`
      to cover both diffusion and Gaussian candidate-executor checkpoints in
      live loading. It now checks that a toy diffusion checkpoint loads with
      `generator_type=diffusion`, a toy Gaussian/no-sample checkpoint loads
      with `generator_type=gaussian` and `candidate_samples=0`, and both
      preserve `phase_residual_l2_caps`. Verification passed: `py_compile`,
      self-test, `bash -n`, and `git diff --check`.
- [ ] 2026-06-15 22:09+08 latest nosample spot check: Slurm step
      `128888.84` is still active at about `30` minutes. Latest eval at step
      `86000` has selected-action MSE `0.00155048` versus DP prior
      `0.00156083` (`-1.04e-05` delta), source counts
      `mean:1, scale_0.05:1, scale_0.2:75`, and `num_candidate_sources=5`.
      This is still below DP but the margin is now small; keep watching the
      final gate rather than declaring offline success. Spot GPU utilization
      via overlapping `nvidia-smi` was `60%/69%`. No final
      `training_summary.json`, `checkpoint_final.pt`, or live panel exists.
- [x] 2026-06-15 22:10+08 added compute-node-only diffusion smoke wrapper:
      `scripts/slurm/run_cosmos3_candidate_executor_diffusion_smoke_in_allocation.sh`.
      It sets `GENERATOR_TYPE=diffusion`, `CANDIDATE_SAMPLES=8`,
      `CANDIDATE_SCALES=0.05,0.1,0.2`, `MAX_STEPS=100`, and
      `MIN_WALL_SECONDS=0`, then delegates to the direct candidate-executor
      trainer. This is for a future 50-100 step smoke inside a tmux-held
      compute allocation only; it does not launch live eval and is not formal
      evidence. Verification passed: `bash -n`, diffusion self-test, and
      `git diff --check`.
- [ ] 2026-06-15 22:08+08 latest nosample spot check: Slurm step
      `128888.84` is still active at about `29` minutes. Latest eval at step
      `82500` has selected-action MSE `0.00151605` versus DP prior
      `0.00156083` (`-4.48e-05` delta), source counts
      `scale_0.05:1, scale_0.1:2, scale_0.2:74`, and
      `num_candidate_sources=5`. `post_gate_status.json` still reports
      `waiting_training_final_artifacts`; no final summary/checkpoint or live
      panel exists yet.
- [x] 2026-06-15 22:06+08 added a login-safe diffusion candidate-executor
      self-test:
      `scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py`.
      It constructs toy Gaussian and diffusion candidate executors on CPU,
      verifies diffusion candidate evaluation produces `diffusion_*`
      candidate sources under the same progress/contact/value scorer, and
      verifies `run_cosmos3_live_receding_loop.py` can load a toy diffusion
      checkpoint with `generator_type=diffusion`. The self-test also now
      checks that live loading preserves checkpoint `phase_residual_l2_caps`,
      so offline candidate caps and live selector caps stay consistent.
      Checks passed:
      `py_compile`, the self-test itself, and `bash -n` for candidate
      training/live wrappers. This is still code-path evidence only, not
      training or live method evidence.
- [ ] 2026-06-15 22:03+08 latest nosample spot check: Slurm step
      `128888.84` is still active at about `24` minutes, still far before
      the 3-hour formal floor. Latest eval at step `68500` has
      selected-action MSE `0.00152335` versus DP prior `0.00156083`
      (`-3.75e-05` delta), selected source counts
      `scale_0.05:1, scale_0.1:1, scale_0.2:75`, and
      `num_candidate_sources=5`. `post_gate_status.json` still reports
      `waiting_training_final_artifacts`: no `training_summary.json`,
      `checkpoint_final.pt`, or live panel exists yet.
- [ ] 2026-06-15 21:55+08 latest nosample spot check: Slurm step
      `128888.84` remains active and is still far before the 3-hour formal
      floor. Latest eval at step `60000` has selected-action MSE
      `0.00151335` versus DP prior `0.00156083`, selected source counts
      `scale_0.05:1, scale_0.1:1, scale_0.2:75`, and
      `num_candidate_sources=5`. No `training_summary.json`,
      `checkpoint_final.pt`, or live panel exists yet, so this is still only
      pre-gate offline evidence.
- [ ] 2026-06-15 21:55+08 added the next aligned implementation path while
      the Gaussian/no-sample formal run continues: the candidate executor can
      now be launched with `GENERATOR_TYPE=diffusion`, training a denoising
      action-residual generator under the same causal Cosmos task/contact
      feature and the same progress/contact/value scorer. The default remains
      `GENERATOR_TYPE=gaussian`, so the currently running checkpoint and its
      guarded live launcher stay compatible. Login-safe verification passed:
      `py_compile` for `train_cosmos3_candidate_executor.py` and
      `run_cosmos3_live_receding_loop.py`, `bash -n` for the candidate
      training/live wrappers, and `git diff --check` for the touched files.
      This is code-path readiness only; no diffusion executor training or
      live result exists yet.
- [ ] 2026-06-15 21:50+08 latest nosample spot check: Slurm step
      `128888.84` remains active. Latest eval at step `29000` has
      selected-action MSE `0.00150265` versus DP prior `0.00156083`, source
      counts `scale_0.05:2, scale_0.2:75`, and `num_candidate_sources=5`.
      Spot GPU utilization was `61%/37%`. No `training_summary.json`,
      `checkpoint_final.pt`, or live panel exists yet.
- [ ] 2026-06-15 21:48+08 latest nosample spot check: Slurm step
      `128888.84` remains active and non-idle. Latest eval at step `24500`
      has selected-action MSE `0.00148684` versus DP prior `0.00156083`,
      source counts `scale_0.05:2, scale_0.2:75`, and
      `num_candidate_sources=5`. Spot GPU utilization was `77%/32%`.
      `post_gate_status.json` still reports `waiting_training_final_artifacts`;
      no final summary/checkpoint or live panel exists yet.
- [ ] 2026-06-15 21:46+08 latest nosample monitor: Slurm step `128888.84`
      is still active, watcher and post-gate recorder are alive, no final
      summary/checkpoint yet. Latest eval at step `19500` has selected-action
      MSE `0.00149078` versus DP prior `0.00156083`; source counts are
      `scale_0.05:1, scale_0.1:1, scale_0.2:75` with
      `num_candidate_sources=5`. Spot GPU utilization was `36%/42%`. Continue
      to the 3-hour final gate; do not report method evidence until the final
      checkpoint gate and live panel/video review exist.
- [ ] 2026-06-15 21:39+08 active formal run is now the no-stochastic-sample
      smallscale direct run:
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_smallscales_nosample_direct_4096_finalgate`.
      Tmux session:
      `cosmos3_candidate_formal_phasecap_smallscales_nosample_direct_4096_2gpu_20260615`;
      Slurm step `128888.84` on `server33`. It uses
      `candidate_samples=0` and `candidate_scales=0.05,0.1,0.2`; manifest and
      first history rows confirm eval `num_candidate_sources=5`, so no
      stochastic sample, `scale_0.5`, or `scale_1` candidates are present.
      This is still the same low-frequency Cosmos task/contact conditioning
      plus learned candidate scorer path, but with a conservative action
      candidate family because stochastic samples caused repeated offline
      gate drift. Env-safe watcher:
      `cosmos3_candidate_after_gate_watch_smallscales_nosample_direct_env_4096_20260615`.
      Early check: step `500` selected MSE `0.00130318` vs DP `0.00156083`.
- [ ] 2026-06-15 21:42+08 nosample monitoring update: Slurm step
      `128888.84` is still running and remains pre-formal with no
      `training_summary.json` or `checkpoint_final.pt`. Through step `7500`,
      the latest 12 validation evals were all below DP-prior action MSE.
      Examples: step `2000` selected `0.00137006`, step `5500`
      `0.00137674`, step `7500` `0.00141868`, DP `0.00156083`. Candidate
      source counts are stable at `dp_prior:1, scale_0.2:76`, with
      `num_candidate_sources=5`. Spot GPU utilization was `44%/26%`; continue
      monitoring but keep this as early offline evidence only.
- [ ] 2026-06-15 21:45+08 added and launched a lightweight post-gate status
      recorder for the active nosample run:
      `scripts/world_model/watch_candidate_executor_post_gate_status.py`,
      tmux session `cosmos3_candidate_post_gate_status_nosample_20260615`.
      It does not run computation; it waits for `training_summary.json`,
      `checkpoint_final.pt`, the env-safe live watcher log, and any
      `live_receding_panel_summary.json`, then writes
      `post_gate_status.json` and `post_gate_status.md` under the formal root.
      Current recorded status is `waiting_training_final_artifacts`. This
      prevents the 3-hour result from being hidden in watcher logs; it is not
      a success claim or a substitute for video/contact-sheet inspection.
- [ ] 2026-06-15 21:38+08 the direct smallscale run with stochastic samples
      (`candidate_executor_train_20260615_formal_2gpu_server33_phasecap_smallscales_direct_4096_finalgate`)
      was interrupted inside tmux. It removed `scale_1`, but the stochastic
      sample family became the next selector-drift source: by step `18500`,
      the latest 12 validation evals had `6` points worse than DP, including
      step `15500` selected `0.00172562` vs DP `0.00156083`. This is not
      formal evidence and must not launch live eval.
- [ ] 2026-06-15 21:37+08 added one more generic trust-region knob:
      `candidate_samples=0` now means no stochastic samples are inserted in
      offline candidate selection or live candidate-executor selection. This
      preserves offline/live consistency and gives the next restart a clean
      `DP + scale_0.05/0.1/0.2` option if the current run later shows
      sustained stochastic-sample drift. Current active run was already loaded
      before this code change and still uses `CANDIDATE_SAMPLES=48`; do not
      reinterpret its checkpoint. Syntax check passed for trainer/live loop
      and direct/watch wrappers.
- [ ] 2026-06-15 21:31+08 active formal run is now the direct
      small-residual candidate-scale run:
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_smallscales_direct_4096_finalgate`.
      Tmux session:
      `cosmos3_candidate_formal_phasecap_smallscales_direct_4096_2gpu_20260615`;
      Slurm step `128888.80` on `server33`. It uses the direct compute-node
      wrapper
      `scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh`
      because the earlier generic wrapper path showed cache/env ambiguity for
      the new `CANDIDATE_SCALES` field. Manifest and first history row confirm
      the actual trainer args now contain `candidate_scales=0.05,0.1,0.2` and
      eval `candidate_scales=[0.05,0.1,0.2]`; candidate count is `53`, so
      `scale_0.5` and `scale_1` are no longer available. Guarded watcher:
      `cosmos3_candidate_after_gate_watch_smallscales_direct_env_4096_20260615`
      with log `candidate_after_gate_watch_envsafe.log`; it passes
      `FORMAL_ROOT` to the compute-node live launcher through `env` rather
      than relying on Slurm's implicit environment propagation. The earlier
      direct watcher without explicit `env` was stopped before it could launch
      anything.
      It still requires `checkpoint_final.pt`, `training_summary.json`,
      `formal_training_floor_met=true`, and the hardened final offline gate
      before any live panel. Early metrics are aligned but not evidence:
      through step `3500`, eval selected-action MSE stays below DP after the
      initial untrained step (`step500=0.00130318`,
      `step1000=0.00129563`, `step3500=0.00140141`, DP
      `0.00156083`), and selected candidates are DP, small scales, or a few
      stochastic samples under the small-scale candidate family.
- [ ] 2026-06-15 21:35+08 direct smallscale monitoring update: Slurm step
      `128888.80` is still running, no `training_summary.json` or
      `checkpoint_final.pt` yet. Through step `11000`, the latest 12
      validation evals were all below the DP-prior action MSE; examples:
      step `5500` selected `0.00139013`, step `10000` `0.00144018`, step
      `11000` `0.00148719`, DP `0.00156083`. Candidate sources remain inside
      DP/small-scale/stochastic small-residual family; no `scale_0.5` or
      `scale_1` are available. Spot GPU utilization was `31%/26%`; keep
      monitoring utilization and final gate, but do not treat these early
      metrics as evidence.
- [ ] 2026-06-15 21:27-21:30+08 the first smallscale restart root
      `candidate_executor_train_20260615_formal_2gpu_server33_phasecap_smallscales_4096_finalgate`
      is invalid and was interrupted. Although the tmux launch line intended
      `CANDIDATE_SCALES=0.05,0.1,0.2`, its manifest/history showed the default
      full scale set `[0.05,0.1,0.2,0.5,1.0]`. This is an implementation/
      launch-propagation failure, not method evidence. It was stopped inside
      tmux with Ctrl-C and replaced by the direct wrapper run above.
- [ ] 2026-06-15 21:26+08 the `phasecap_meanpen5_4096_finalgate` formal run
      was interrupted inside tmux because the selector failure became
      sustained, not merely noisy: in the latest 12 validation evals before
      stopping, 9 had selected-action MSE above the frozen DP prior, and
      `scale_1` dominated selected candidates. Example bad points:
      step `28000` selected `0.00163134` vs DP `0.00156083`; step `32000`
      selected `0.00163372`; step `34000` selected `0.00160710`. This run is
      not evidence and must not launch live eval. The failure diagnosis is
      selector/action trust-region drift: the scorer overuses large residual
      candidates even when the offline DP-prior gate would reject the final
      checkpoint.
- [ ] 2026-06-15 21:23+08 added a generic candidate-scale trust-region
      switch while the active `phasecap_meanpen5_4096_finalgate` run
      continues. Code now supports `--candidate-scales` /
      `CANDIDATE_SCALES`, records the scale set in training manifests and
      eval metrics, saves it in checkpoint args, and makes live
      `candidate_executor` selection use the same checkpoint scale set. This
      is not a case-specific recovery rule: it is a DP-prior trust region for
      the candidate/diffusion executor, matching the DDP-style idea that the
      world model should condition short action chunks, not replace the
      manipulation policy with unconstrained large residuals. Syntax checks
      passed:
      `.venv/bin/python -m py_compile scripts/world_model/train_cosmos3_candidate_executor.py scripts/world_model/run_cosmos3_live_receding_loop.py`
      and
      `bash -n scripts/slurm/run_cosmos3_candidate_executor_train_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`.
      The currently running checkpoint was launched before this option existed
      and therefore keeps the old full scale set by default. If its final gate
      fails or if sustained validation points drift above the DP prior, stop it
      inside tmux and restart with small-residual scales such as
      `CANDIDATE_SCALES=0.05,0.1,0.2` rather than repeating the same
      scale-1-heavy selector.
- [ ] 2026-06-15 21:22+08 monitoring update for the active
      `phasecap_meanpen5_4096_finalgate` run: it is still running on Slurm
      step `128888.73` and has not written `training_summary.json` or
      `checkpoint_final.pt`, so live eval remains blocked. Validation selected
      MSE stayed mostly below the DP prior through step `22500`, but one point
      at step `22000` was slightly worse (`0.00157423` versus DP
      `0.00156083`), and the selector is increasingly choosing `scale_1`
      candidates. Do not claim success from this. Continue monitoring; the
      hardened final gate still refuses live if final selected-action MSE is
      worse than DP.
- [ ] 2026-06-15 21:14+08 current active formal candidate-executor run is
      now the stronger mean-penalty selector run, after the previous
      `phasecap_sourcepen_margin0_4096_finalgate` run was interrupted inside
      tmux at about `10` minutes because late eval points drifted above the DP
      prior and would fail the hardened live gate. That interrupted run is not
      formal evidence. New active root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_meanpen5_4096_finalgate`;
      tmux session:
      `cosmos3_candidate_formal_phasecap_meanpen5_4096_2gpu_20260615`;
      Slurm step `128888.73` on `server33`. It keeps the same 2-GPU/10800s
      formal floor and phase residual-L2 caps, but increases selector
      penalties to `score_mean_source_penalty=5.0`,
      `score_large_scale_source_penalty=0.5`, and
      `score_stochastic_source_penalty=1.0`. This intentionally prevents raw
      unscaled generator mean from becoming the default action source; DP and
      cap-valid scale candidates still compete through the progress/contact/
      value scorer. Guarded watcher:
      `cosmos3_candidate_after_gate_watch_meanpen5_4096_20260615`. Early
      metrics are aligned but not formal evidence: through early evals,
      candidate sources no longer include raw `mean`, selected-action MSE
      remains below DP prior, and spot GPU utilization was about `41/42%`.
- [ ] 2026-06-15 21:02+08 formal candidate-executor training is active on
      the repaired selector path. Tmux session:
      `cosmos3_candidate_formal_phasecap_sourcepen_4096_2gpu_20260615`;
      Slurm allocation/step: `128888.58` on `server33`; output root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_sourcepen_margin0_4096_finalgate`.
      Config: `2` GPUs, `hidden_dim=4096`, `num_layers=6`, `dropout=0.2`,
      `lr=5e-5`, `weight_decay=2e-4`, `candidate_samples=48`,
      phase residual-L2 cap quantile `0.9`, cap max `0.02`, DP fallback
      phases `all`, fallback margin `0.0`, mean-source penalty `0.25`,
      large-scale penalty `0.25`, stochastic-source penalty `0.75`, formal
      wall floor `10800` seconds. Guarded watcher:
      `cosmos3_candidate_after_gate_watch_phasecap_4096_20260615`.
      It waits for this run's `training_summary.json` and
      `checkpoint_final.pt`, waits for the training step to exit, and then
      calls the gated live launcher inside allocation `128888`; the launcher
      still refuses live eval unless final `formal_training_floor_met=true`,
      `ready_for_formal_live_eval=true`, and the final checkpoint loads.
      Early formal metrics are aligned but not evidence: around step `8500`,
      eval selected-action MSE was `0.0012228` versus DP prior `0.0015608`,
      with contact/progress readouts still passing the offline thresholds.
      GPU utilization is fluctuating but not idle in spot checks; continue
      monitoring until the formal floor and final gate. The candidate
      live-gate was hardened after launch: future trainer summaries and the
      independent live launcher now require final selected-action MSE to be no
      worse than the frozen DP prior, not merely within the old `1.25x`
      diagnostic tolerance. This preserves the original objective by blocking
      a final checkpoint that damages the DP prior before any live rollout.
- [ ] 2026-06-15 20:49+08 active formal candidate-executor run was
      restarted on the CUDA-valid tmux-held Slurm allocation `128888` on
      `server33` after the heavier `4096` hidden-dim attempt failed before
      training. The failed root
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_globalfallback_heavy_finalgate`
      has no `training_history.json`; its console only reports
      `srun: error: Unable to create step for job 128888: Memory required by
      task is not available`. This is a scheduling/step-memory request
      failure, not method evidence. The stale heavy after-gate watcher was
      killed without releasing the held allocation. The active replacement is
      tmux session `cosmos3_candidate_formal_globalfallback_2048_2gpu_20260615`,
      Slurm step `128888.48`, output root
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_globalfallback_2048_finalgate`.
      It uses `2` GPUs, `hidden_dim=2048`, `num_layers=6`, `batch_size=256`,
      `candidate_samples=48`, global DP fallback, and the formal
      `10800` second floor. Live eval remains forbidden until this run writes
      a loadable `checkpoint_final.pt` and `training_summary.json` with
      `formal_training_floor_met=true` and `ready_for_formal_live_eval=true`.
- [ ] 2026-06-15 20:56+08 candidate selector was repaired before formal
      restart because the `2048` run repeated the same over-selection failure
      within minutes: by step `4000`, eval selected-action MSE was around
      `0.00517` versus DP prior `0.0015608`, with the selector mostly choosing
      raw `mean`. That run was interrupted inside tmux with Ctrl-C and is not
      evidence. The repair is a universal policy-prior barrier, not enumerated
      recovery: phase-specific residual-L2 caps are estimated from the train
      split's contact phases, raw `mean`, large-scale, and stochastic sources
      can receive source penalties, and live execution loads the same caps from
      the final checkpoint. Smoke gates on allocation `128888` show the desired
      selector shape. Best short smoke:
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_smoke512_100step_phasecap_sourcepen_margin0_4096`;
      it selected only DP plus small residual scales (`scale_0.05/0.1/0.2`),
      not raw mean or random samples, and reached eval selected-action MSE
      `0.00153759` versus DP prior `0.00156083`. This is only a 100-step smoke;
      the next formal run must still satisfy `2` GPUs and `10800` seconds with
      final-checkpoint gating.
- [ ] 2026-06-15 20:17+08 formal candidate-executor training is now
      running, not completed. The first formal attempt at
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_sourcepenalty`
      was interrupted inside tmux with Ctrl-C, not `scancel`, because it had no
      best-checkpoint support and the eval path was already overfitting badly:
      by step `48000`, selected eval action MSE was `0.00378292` versus
      frozen-DP-prior `0.00156083`, and the selector chose `mean` for `76/77`
      validation rows. This interrupted run is not formal evidence.
      The trainer now saves `checkpoint_best_offline.pt` from the same formal
      run and the guarded live launcher uses that checkpoint only if the final
      summary says `formal_training_floor_met=true` and
      `ready_for_formal_live_eval=true`.
- [ ] 2026-06-15 20:29+08 restarted formal candidate-executor training with
      best-offline checkpoint support and `EVAL_EVERY_STEPS=100` to preserve
      the early non-overfit window while still running the full `2` GPU /
      `3` hour floor. It uses tmux session
      `cosmos3_candidate_formal_best_2gpu_20260615`, Slurm allocation
      `128888` on `server33`, output root
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_bestoffline_sourcepenalty`.
      CUDA canary passed, `checkpoint_best_offline.pt` was written, and step
      `100` currently has selected eval action MSE `0.00155706` versus
      DP-prior `0.00156083`, progress MSE `0.0124811`, inserted acc
      `0.961039`, and DP-continuable acc `0.870130`. This is promising
      offline evidence but still not live evidence until the run reaches
      `10800` seconds and writes a passing `training_summary.json`.
- [ ] 2026-06-15 20:38+08 corrected the formal gate discipline again: using
      an early `checkpoint_best_offline.pt` for live eval would violate the
      user's "no short training as formal evidence" rule even if it came from
      a 3-hour run. The best-offline checkpoint remains diagnostic only. The
      active formal live gate now requires the post-floor `checkpoint_final.pt`
      and final `ready_for_formal_live_eval=true`. The bestoffline run above
      was interrupted inside tmux with Ctrl-C and replaced by
      `cosmos3_candidate_formal_globalfallback_2gpu_20260615`, output root
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_globalfallback_finalgate`.
      This run changes the candidate selector to global conservative DP
      fallback (`dp_fallback_phases=all`) so a non-DP chunk must beat the DP
      candidate by the recorded score margin in any phase before execution.
      Early evidence: step `500` selected MSE `0.00156168` versus DP
      `0.00156083`; step `1000` selected MSE `0.00186330`, still inside the
      1.25x offline gate, with progress MSE `0.00970525`, inserted acc
      `0.961039`, and DP-continuable acc `0.896104`. Formal summary is still
      missing until the run reaches `10800` seconds.
- [ ] A guarded after-gate watcher is active in tmux session
      `cosmos3_candidate_after_gate_watch_globalfallback_20260615`. It polls
      for `training_summary.json` and `checkpoint_final.pt`, then waits until
      the non-extern Slurm step for allocation `128888` exits, and only then
      launches
      `scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`
      inside the same held allocation. The launcher itself refuses live eval
      unless the formal summary has `formal_training_floor_met=true` and
      `ready_for_formal_live_eval=true`, and unless the final checkpoint loads.
      If the gate fails, this watcher records refusal in
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_globalfallback_finalgate/candidate_after_gate_watch.log`
      and does not run live.
- [x] 2026-06-15 candidate-executor short smoke on the `512` causal
      contact-executor rows found the useful selector boundary. The raw
      stochastic selector over-picked noisy sampled chunks; adding DP
      phase fallback, generator log-probability, and a stochastic-source
      penalty made the offline selector conservative enough for the short
      gate. Best smoke root:
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_candidate_smoke512_100step_excl13_sourcepenalty`.
      It is not formal evidence because it ran only `100` steps and
      `formal_training_floor_met=false`, but it passed the offline smoke gate:
      `ready_for_offline_gate=true`, selected eval action MSE `0.00156491`
      versus DP-prior eval MSE `0.00156083`, mean action MSE `0.00152223`,
      progress MSE `0.0127258`, inserted acc `0.987013`, DP-continuable acc
      `0.857143`. Plain interpretation: the candidate/scorer interface is
      wired, the generator mean has real signal, and the scorer needs
      conservative candidate selection before any live rollout.
- [ ] 2026-06-15 candidate/diffusion executor direction is now active by user
      instruction. Do not spend the next attempt on another direct
      deterministic contact-executor rerun unless it is only a debug baseline.
      The next method target is:
      `Cosmos low-frequency task/contact imagination -> stochastic
      candidate/diffusion action chunk generator -> action-conditioned
      progress/contact/value scorer -> short-prefix execution -> real
      re-observation`. This follows the DDP/HDP lesson: the executor must act
      from imagined task/contact structure as a native condition and must
      choose among multiple plausible chunks using progress/contact value, not
      single MSE residual regression.
- [x] Added the first code path for that direction:
      `scripts/world_model/train_cosmos3_candidate_executor.py`. It reads the
      causal contact executor dataset, trains a stochastic residual
      distribution over short action chunks, and trains an
      action-conditioned scorer that predicts contact progress, future
      insertion, future DP-continuability, and a scalar value for each
      candidate residual. Its offline candidate evaluation compares DP prior,
      generator mean, scaled residuals, and stochastic samples, then selects
      by predicted progress/contact/value with a DP-residual penalty. This is
      candidate-executor evidence only; live method evidence still requires a
      gated simulator rollout with videos/contact sheets inspected.
- [x] Added compute-only wrapper
      `scripts/slurm/run_cosmos3_candidate_executor_train_in_allocation.sh`.
      It refuses login-node execution and records a wrapper manifest. Syntax
      checks passed:
      `.venv/bin/python -m py_compile scripts/world_model/train_cosmos3_candidate_executor.py`
      and `bash -n scripts/slurm/run_cosmos3_candidate_executor_train_in_allocation.sh`.
- [ ] Requested a new tmux-held `2` GPU, `1` day allocation for candidate
      executor smoke/training using `salloc`, not `sbatch`: Slurm job
      `128862`, tmux session `cosmos3_candidate_exec2gpu_20260615`.
      It started on `server13`, but the short smoke did not enter training:
      `nvidia-smi` sees H200 devices and `/dev/nvidia*` permissions are
      present, yet PyTorch CUDA initialization fails with
      `CUDA unknown error` for both two-GPU and one-GPU steps. The failed
      smoke console is
      `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_candidate_smoke64_100step.console.log`.
      This is a scheduling/node CUDA preflight failure, not a candidate
      executor training result. The wrapper was also corrected to stop
      hard-defaulting `CUDA_VISIBLE_DEVICES=0,1`; it now lets Slurm provide
      the GPU mapping unless `CUDA_VISIBLE_DEVICES_OVERRIDE` is explicitly set.
      Since `server13` is unusable for PyTorch CUDA, requested a second
      tmux-held `2` GPU, `1` day allocation excluding `server13`: Slurm job
      `128888`, tmux session `cosmos3_candidate_exec2gpu_excl13_20260615`,
      currently pending with reason `Priority`. Once a CUDA-valid allocation
      starts, first run only the short overfit/smoke gate (`50-100` steps,
      small sample count) inside the allocation. Do not launch formal 3-hour
      training or live eval until the smoke output proves the
      candidate/scorer path is wired correctly.

- [ ] Latest status, 2026-06-15 early morning: the dense full SFT itself is
      no longer the immediate blocker. The current 4-GPU run satisfied the
      updated formal floor because it trained past the 2-GPU/3-hour minimum
      and saved checkpoints through `iter_000001500`. Generated eval strict
      checks pass for post-floor checkpoints: `iter900` and `iter1200` each
      passed 10/10 strict generated-artifact checks, `iter1500` passed the
      corrected generated strict gate, and an extra `iter1500` generated
      inspection produced `72` valid samples with
      `strict_eval_artifacts_ok=true` and no strict failures. The remaining
      problem is closed-loop real execution, not a 301/300 length or SFT
      startup failure.
- [ ] Closed-loop status from the formal dense run: `iter900` finished `0/4`
      real final-state successes, and `iter1200` also finished `0/4` with
      `panel_full_episode_contract_ok=true`, `failed_process_count=0`, and no
      sample contract failures. All four `iter1200` annotated rollout videos
      were converted to review sheets and opened. The failures are real
      simulator failures, not metric bookkeeping. In particular,
      `sample_01_hole_late_constant` executed `136` DP handoff steps and
      `sample_04_hole_late_sine` executed `68` DP handoff steps, but neither
      inserted the peg. The plain blocker is that raw Cosmos action chunks
      plus the DP handoff do not yet put the peg into a stable
      DP-continuable insertion state after target motion.
- [ ] `iter1500` closed-loop eval has now produced the same blocker. `server35`
      is not usable for live ManiSkill render evidence: parallel live eval hit
      Vulkan `DeviceLost`, and serial live eval stalled before useful progress.
      The final live evidence was therefore collected on render-capable
      `server62` and `server24` allocations using the same strict generated
      gate and the same live-receding protocol. The four iter1500 panel samples
      all failed real final-state success with full `301` observed frames:
      sample00 final peg-head-in-hole `[-0.0829, -0.0107, 0.0027]` with
      `30` DP handoff steps; sample01 `[-0.0822, -0.0015, 0.0051]` with
      `8` DP handoff steps; sample03 `[-0.1246, -0.0129, -0.0276]` with no
      handoff; sample04 `[-0.0424, 0.0026, -0.0030]` with `72` DP handoff
      steps. Review sheets were generated and opened for all four iter1500
      samples. Plain conclusion: after dense full SFT, generated strict eval
      passes but direct raw Cosmos action closed-loop remains `0/4` at
      iter900, `0/4` at iter1200, and `0/4` at iter1500. Stop treating raw
      Cosmos action chunks plus threshold handoff as the main controller.
      A user-facing evidence index is recorded at
      `docs/world_model_task_rebinding/2026-06-15_dense_closed_loop_failure_evidence_index.md`.
      Latest user continuation, 2026-06-15: start the planned low-frequency WM
      plus learned executor/DP-prior residual path under the updated `2` GPU /
      `3` hour formal-training floor. Do not rerun the same raw-action branch
      unless explicitly requested; it already has formal closed-loop failure
      evidence.
- [x] 2026-06-15 executor path start: added the executor interface/data gates
      and ran them on held compute allocation `128006` (`server62`), not on the
      login node. Full clean-dense executor preflight wrote `6969` debug
      samples under
      `experiments/world_model_task_rebinding/cosmos3/executor_dataset_clean_dense_late299_chunk24_debug_20260615_executor_preflight_full_debug`
      with `schema_ok=true` and `ready_for_debug_overfit=true`. This is not
      formal executor training because the task path is still
      `gt_state_targets_debug` and DP prior was initially absent.
- [x] 2026-06-15 DP-prior smoke: exported frozen static-DP proposal chunks for
      `2` executor rows under
      `experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_smoke_20260615_dp_prior_smoke2`.
      The DP checkpoint loaded on CUDA and produced two action-prior chunks
      with no export failures. This proves the DP-prior input interface works
      for the short gate; it is not evidence that frozen DP solves the dynamic
      task.
- [x] 2026-06-15 executor two-sample overfit: trained a small residual
      executor for `100` steps on the two matched rows under
      `experiments/world_model_task_rebinding/cosmos3/executor_overfit_smoke_20260615_executor_overfit2`.
      It used the short-overfit exception (`1` GPU, `100` steps, no
      `3`-hour requirement) and reduced MSE from `0.0035156261` at step `1`
      to final `3.46294e-07` with `ready_for_debug_gate=true`. This starts
      the executor branch but is still debug-only because the task path is GT
      debug, not causal Cosmos prediction.
- [x] Resolved the immediate GT-debug blocker for launch: executor training is
      no longer using `gt_state_targets_debug`. The causal input path is now
      Cosmos-predicted WAM sidecars with `task_path_source=
      cosmos_predicted_action_sidecar`, plus matched frozen-DP prior chunks.
      The `64`-row no-GT two-sample gate passed with final MSE
      `1.74878e-08`, proving the causal interface can be optimized before
      starting the larger formal run.
- [x] 2026-06-15 latest resource override is active in execution, not just in
      prose: formal full training may start with `2` GPUs for at least `3`
      hours; do not wait for `4` GPUs if a valid `2` GPU allocation is already
      available. A tmux-held interactive allocation was requested and granted
      as Slurm job `128023` on `server54` with `gres:gpu:2` for `1` day. CUDA
      canary passed: `.venv` sees `torch 2.5.1+cu121`, `cuda_available=True`,
      `device_count=2`, both devices are `NVIDIA H200`.
- [x] Built a broader executor-targeted Cosmos inference input set from the
      clean/dense train executor rows after replacing first-row selection with
      role/scenario round-robin selection:
      `experiments/world_model_task_rebinding/cosmos3/executor_wam_eval_inputs_20260615_executor_wam_inputs64_diverse`.
      It has `strict_eval_input_ok=true`, `64` selected samples, role counts
      `insert_resume=25`, `target_motion_observed=18`,
      `target_post_motion=21`, and covers nine scenario buckets instead of
      only `hole_late_move_stop`.
- [x] Started and completed on the `2` GPU `server54` allocation:
      executor-targeted
      Cosmos inference from dense SFT checkpoint `iter_000001500` using the
      64-row input JSONL above and output root
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/executor_wam_eval_iter_000001500_train64_diverse_20260615_server54`.
      This is the required causal predicted-task-path source for executor
      training. It is not a repeat of the failed raw-action closed loop.
      If this inference fails or stalls in model/checkpoint I/O, inspect
      `inference.log`, `inference/console.log`, and `inference/debug.log`
      before changing protocol.
- [x] The 64-row executor-targeted Cosmos prediction pass completed on `server54`
      and strict inspection passed: `strict_eval_artifacts_ok=true`,
      `num_samples=64`, and `strict_failures=[]`. The resulting causal
      predicted-task-path dataset is
      `experiments/world_model_task_rebinding/cosmos3/executor_dataset_cosmos_predicted_task_path_iter1500_20260615_pred_task_path_train64_diverse`
      with `64` rows and `task_path_source=cosmos_predicted_action_sidecar`.
      Diagnostic GT RMSE is advisory only: mean `0.0479531`, max `0.222928`.
- [x] Exported frozen-DP prior chunks for the same 64 selected rows under
      `experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_smoke_20260615_pred_task_path_dp_prior_train64_diverse`;
      `num_records=64`, `failure_counts={}`, `device=cuda`.
- [x] Ran the no-GT two-sample executor gate on the 64-row causal predicted
      dataset plus matched DP-prior rows:
      `experiments/world_model_task_rebinding/cosmos3/executor_overfit_smoke_20260615_executor_overfit2_pred_task_path_train64_diverse`.
      It used `task_path_sources=["cosmos_predicted_action_sidecar"]`, not GT
      debug paths, and reduced final MSE to `1.74878e-08` with
      `ready_for_debug_gate=true`.
- [x] Completed on the same `2` GPU `server54` allocation: the larger
      executor-targeted Cosmos prediction pass with `512` selected train rows,
      input root
      `experiments/world_model_task_rebinding/cosmos3/executor_wam_eval_inputs_20260615_executor_wam_inputs512_diverse`,
      and output root
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/executor_wam_eval_iter_000001500_train512_diverse_20260615_server54`.
      Its input manifest has `512` samples with role counts
      `insert_resume=169`, `target_motion_observed=150`,
      `target_post_motion=193`. Strict artifact inspection passed with
      `strict_eval_artifacts_ok=true`, `num_samples=512`, and `0` strict
      failures. The resulting executor dataset wrote `512` causal sidecar rows;
      diagnostic GT RMSE remains advisory only: mean `0.0567245`, max
      `0.261960`.
- [x] Exported frozen-DP prior chunks for the same `512` selected rows under
      `experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_smoke_20260615_pred_task_path_dp_prior_train512_diverse`;
      `num_records=512`, `failure_counts={}`, `device=cuda`.
- [x] Added the formal residual executor trainer and in-allocation wrapper:
      `scripts/world_model/train_cosmos3_executor_residual.py` and
      `scripts/slurm/run_cosmos3_executor_residual_train_in_allocation.sh`.
      The trainer refuses GT debug task paths, supports `torchrun`/DDP, records
      the formal `2` GPU / `3` hour floor, and trains a residual on top of the
      frozen DP prior using causal Cosmos-predicted task paths.
- [x] Formal residual-executor training completed in tmux session
      `cosmos3_executor_formal_2gpu_20260615`, Slurm job `128023` on
      `server54`, output root
      `experiments/world_model_task_rebinding/cosmos3/executor_residual_train_20260615_executor_residual_train512_formal_2gpu_server54`.
      It satisfied the current formal floor: `world_size=2`,
      `elapsed_seconds=10800.012`, and `formal_training_floor_met=true`.
      The unscaled final residual executor is not usable by itself:
      final validation action MSE is `0.0183239` versus frozen-DP-prior
      validation MSE `0.00156083`, so `ready_for_closed_loop_eval=false`.
      Plain interpretation: the model can fit train rows, but a full residual
      changes held-out DP-prior actions too aggressively.
- [ ] Mid-run monitor at `2026-06-15T07:25:57+08:00`: formal training is still
      active, not complete. Latest history entry around step `48200` has train
      action MSE `3.58e-06`, validation action MSE `0.01171`, and frozen-DP
      prior validation MSE `0.00156`. Best observed validation MSE so far is
      `0.00493` at step `23000`, still worse than DP prior. GPU utilization is
      fluctuating and was most recently `76%` / `23%`; keep monitoring because
      sustained low utilization can lose the allocation. Current plain status:
      the run is technically healthy but the learned residual has not yet shown
      held-out improvement over the DP prior, so no closed-loop launch yet.
- [x] Added a compute-node split diagnostic for the same formal run:
      `experiments/world_model_task_rebinding/cosmos3/executor_residual_train_20260615_executor_residual_train512_formal_2gpu_server54/split_baseline_diagnostic.json`.
      It shows the train split DP-prior/teacher MSE is `0.00639773`, while the
      validation split DP-prior/teacher MSE is only `0.00156083`. The easiest
      validation group is `target_motion_observed` with DP-prior MSE
      `0.000144541`; the hardest validation group is `peg_drop` with
      `0.00959632`. Plain implication: the residual is being trained on many
      rows where teacher differs strongly from DP, but validation contains many
      rows where DP is already nearly correct. If final validation remains
      worse than DP prior, the blocker is likely insufficiently conservative
      residual/generalization, not missing closed-loop video.
- [ ] Follow-up monitor at `2026-06-15T07:35:35+08:00`: formal training is
      still below the `3` hour floor. Latest step `83000` has train action MSE
      `3.52990e-06`, validation action MSE `0.0129802`, and DP-prior validation
      baseline `0.00156083`; best validation MSE remains `0.00493207` at step
      `23000`. The live panel remains gated off.
- [x] Prepared the residual-executor closed-loop interface while the formal
      training runs. `scripts/world_model/run_cosmos3_live_receding_loop.py`
      and `scripts/world_model/run_cosmos3_live_receding_panel.py` now accept
      `--controller-action-source=residual_executor` plus
      `--executor-checkpoint`. In that branch, Cosmos still runs at each
      reobserve point, but only its causal predicted task-state sidecar is used
      as the world/task path; robot actions come from the learned residual
      executor conditioned on current live state and frozen-DP prior actions.
      The raw Cosmos robot-action columns are recorded but not executed in this
      branch. The Slurm wrappers pass the same controls through
      `CONTROLLER_ACTION_SOURCE` and `EXECUTOR_CHECKPOINT`. Syntax checks for
      the two Python files and two wrappers passed inside compute allocation
      `128023`, not on the login node.
- [x] Added a conservative residual-scale calibration path for the post-floor
      checkpoint: `scripts/world_model/calibrate_cosmos3_executor_residual_scale.py`
      plus live-loop/panel support for `--executor-residual-scale` and wrapper
      env `EXECUTOR_RESIDUAL_SCALE`. This is not a shortcut to call DP a method:
      scale `0` is explicitly recorded as the frozen-DP baseline and cannot be
      reported as residual-executor success. Live eval remains allowed only if
      a positive residual scale beats the DP-prior validation baseline after
      the formal training floor. Py-compile and wrapper syntax checks passed
      inside allocation `128023`.
- [x] Started a lightweight compute-node watcher in tmux session
      `cosmos3_executor_residual_calibrate_watch_20260615`, Slurm step
      `128023.51` on `server54`. It waits for
      `training_summary.json` and `checkpoint_final.pt`, then runs the
      residual-scale calibration under
      `experiments/world_model_task_rebinding/cosmos3/executor_residual_train_20260615_executor_residual_train512_formal_2gpu_server54/residual_scale_calibration_final`.
      It does not launch closed-loop eval.
- [x] Post-floor residual-scale calibration completed under
      `experiments/world_model_task_rebinding/cosmos3/executor_residual_train_20260615_executor_residual_train512_formal_2gpu_server54/residual_scale_calibration_final`.
      Scale `0` is the frozen-DP baseline and is not method success. The best
      positive scale is `0.05`, with validation action MSE `0.00151089` versus
      DP prior `0.00156083` (`-4.99e-05`). This is only a tiny offline gain,
      but it clears the conservative gate for one small residual-executor live
      panel using `EXECUTOR_RESIDUAL_SCALE=0.05`. If the panel fails, do not
      reinterpret the unscaled executor as evidence; the likely blocker is
      still insufficiently robust executor generalization.
- [x] Ran the first residual-executor live panel with the calibrated small
      scale: output root
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_executor_residual_iter1500_panel4_20260615_1017_residual_scale005`.
      It completed `4/4` processes with `panel_full_episode_contract_ok=true`,
      `failed_process_count=0`, `final_success_count=2`, and `301` decoded
      frames for every final rollout video. Visual review opened
      `live_receding_panel_contact_sheet.png` and matched the metric-level
      result: sample00 `hole_late_move_stop` and sample03
      `hole_late_fast_shift` visibly reach/hold the hole area and are final
      real-state successes; sample01 `hole_late_constant` and sample04
      `hole_late_sine` remain visibly short/misaligned and fail real final
      state. This is a real improvement over the direct raw-Cosmos-action
      `0/4` panel, but it is still small-panel evidence only:
      `method_evidence_allowed=false` in the summary.
- [x] Run a second same-protocol residual-executor panel on unused eval samples
      `2,5,6,7` with `EXECUTOR_RESIDUAL_SCALE=0.05` before claiming the branch
      generalizes beyond the fixed four-sample comparison. Keep the same
      causal target-motion prefixing, full `301/300` contract, and video review
      requirement. Result root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_executor_residual_iter1500_panel4b_20260615_1224_residual_scale005_samples2_5_6_7`.
      It completed `4/4` processes with `panel_full_episode_contract_ok=true`,
      `failed_process_count=0`, and `final_success_count=1`. Visual review
      opened `live_receding_panel_contact_sheet.png` and matched the metrics:
      sample02 `hole_late_reverse` succeeds; sample05
      `hole_late_continuous_insert` visibly drifts away; sample06
      `hole_late_continuous_insert` reaches near the hole but does not insert;
      sample07 `peg_drop` never enters WM/executor active mode
      (`wm_active_frame_count=0`), so that failure is a trigger/coverage issue
      rather than a measured executor insertion attempt.
- [ ] Current residual-executor evidence after two panels: `3/8` live
      successes with full `301`-frame videos inspected (`2/4` first panel,
      `1/4` second panel). This is better than the direct raw-Cosmos-action
      `0/4`, but it is not final method evidence. Plain current blocker:
      small-scale residual executor can sometimes rebind to a moved hole, but
      it does not reliably maintain contact/insertion progress, and peg-drop
      perturbations are not covered by the current target-motion trigger.
- [x] Literature-driven repair note added after inspecting
      DDP/AdaWorldPolicy/DiWA/WAM/HDP:
      `PLAN/cosmos3_lowfreq_wm_executor/04_ddp_hdp_borrowed_executor_repair.md`.
      Plain conclusion: DDP-style methods work because policy and world model
      are tightly coupled through shared or jointly trained representations,
      and contact-rich methods work by making the low-level policy condition on
      explicit contact/phase goals. Our current residual executor is only a
      diagnostic bridge: full scale overfits and tiny `0.05` scale can improve
      some samples but is too weak for stable insertion. Do not keep training
      the same small residual MLP as the final method.
- [ ] Next method repair after the current second panel finishes: build a
      contact/progress label exporter and dataset for a contact-conditioned
      executor. Required labels include peg-head-at-hole, hole frame, grasp,
      insertion depth/progress, hole-mouth/contact phase, and
      DP-continuability. This is not enumerated error recovery; it is a
      universal contact-phase representation for peg insertion under dynamic
      task-frame rebinding.
- [x] Added the first contact/progress label exporter:
      `scripts/world_model/export_cosmos3_contact_progress_labels.py`. It reads
      the existing full-episode H5 slots and writes per-episode and per-row
      labels for contact phase, peg-head-at-hole, lateral/insertion/contact
      progress, grasp/inserted flags, perturb flags, and DP-continuability.
      These labels are explicitly supervision/scoring targets only and must not
      be used as privileged future controller inputs.
- [x] Ran a compute-node smoke for the label exporter inside held Slurm
      allocation `128023`, not on the login node:
      `experiments/world_model_task_rebinding/cosmos3/contact_progress_labels_smoke_20260615_executor_repair`.
      It wrote `2` unique episode label files and `16` row labels with
      `ready_for_contact_executor_dataset=true`. This proves the existing H5
      fields are sufficient for the first contact/progress dataset gate. The
      small episode count is because the first smoke rows duplicate the same
      source episodes, not because the exporter failed.
- [x] Ran the full contact/progress label export inside the same held compute
      allocation `128023`:
      `experiments/world_model_task_rebinding/cosmos3/contact_progress_labels_full_20260615_executor_repair`.
      First full attempt hit HDF5 read-only locking failure
      `No locks available`; rerun with `HDF5_USE_FILE_LOCKING=FALSE` succeeded.
      Final summary: `733` episodes, `9271` row labels (`8438` train, `833`
      val), all `733` source episodes marked successful at episode end, and
      `ready_for_contact_executor_dataset=true`. Scenario coverage includes
      hole motion, static/none, `peg_drop=119`, and `peg_disturb=2`.
- [x] Joined the 512-row causal Cosmos executor dataset, matched frozen-DP
      prior chunks, and full contact/progress labels inside held compute
      allocation `128023`:
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_iter1500_train512_scale005_repair_20260615`.
      The join matched `512/512` rows with `missing_counts={}` and
      `ready_for_contact_executor_training=true`. Phase coverage is
      `far=94`, `lateral_align=142`, `preinsert_aligned=76`, and
      `dp_continuable=200`; `185` chunks become inserted within the future
      window and `319` become DP-continuable. Plain implication: the immediate
      contact-executor data gate is open; the next blocker is not missing
      labels or uuid mismatch, but whether a stronger executor can use these
      labels to improve contact/insertion behavior rather than just action MSE.
- [x] Added the first contact/progress-conditioned executor trainer:
      `scripts/world_model/train_cosmos3_contact_executor.py`. It uses only
      causal inputs as features: current state, causal Cosmos-predicted task
      path, frozen-DP prior actions, and current contact/progress context.
      Future inserted/DP-continuable/progress values are training targets only,
      recomputed from `contact_label_npz` at the actual executable horizon
      (`8` steps here), not privileged live-controller inputs.
- [x] Ran a short contact-executor overfit/debug gate inside held allocation
      `128023`:
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_overfit_smoke_20260615_train64_progress_horizon8`.
      It used `64` rows, `48/16` train/eval split, `100` steps, and CUDA.
      Train action MSE dropped below the train DP-prior baseline
      (`4.91476e-06` versus `0.00890316`), train progress MSE reached
      `0.0003792`, and inserted/DP-continuable train accuracies reached `1.0`.
      This clears the contact-label/feature/action interface debug gate.
      Negative boundary: eval action MSE was still worse than the eval DP
      prior (`0.00133105` versus `0.000372038`), so this is not
      generalization evidence and must not trigger live eval by itself.
- [x] Added and smoke-tested `2`-rank DDP support for the contact executor
      trainer. The first `4`-step DDP smoke wrote a valid summary but returned
      a torchrun failure because the script used exit code `64` for
      `ready_for_debug_gate=false`; that was an implementation/reporting bug,
      not a DDP hang. The script now records readiness in
      `training_summary.json` and exits cleanly unless the program actually
      crashes. A second `2`-rank smoke completed with exit `0` under
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_ddp_smoke_20260615_train16_2step_exitfix`.
- [ ] Formal contact/progress executor training has started in tmux session
      `cosmos3_contact_executor_formal_2gpu_20260615`, inside the held
      Slurm allocation `128023` on `server54`, using wrapper
      `scripts/slurm/run_cosmos3_contact_executor_train_in_allocation.sh`.
      The first launch root
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54`
      was interrupted inside tmux before the formal floor because
      `SAVE_EVERY_STEPS=1000` would write a `~200MB` checkpoint every few
      seconds on this small dataset. The allocation was preserved; no
      `scancel` was used. The current formal root is
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k`
      with `SAVE_EVERY_STEPS=20000` and `EVAL_EVERY_STEPS=2000`.
      CUDA canary passed with `2` H200s and `torch 2.5.1+cu121`. Early current
      training is active, not complete: at step `4000` around `58` seconds,
      train action MSE was `1.11e-05` versus train DP-prior baseline
      `0.0063977`, but eval action MSE was still worse than the eval DP-prior
      baseline (`0.00598` versus `0.0015608`). Eval progress MSE was low
      (`0.00862`) and inserted/DP-continuable accuracies were about
      `0.961/0.870`. Plain current watchpoint: the model may be learning
      progress/contact readouts while still producing held-out actions that
      are worse than the frozen DP prior. Do not launch live eval unless the
      post-floor summary gives a defensible offline gate.
      Latest mid-run check before formal floor: at step `44000` around
      `578` seconds, eval action MSE was `0.008588` versus DP-prior baseline
      `0.001561`, while eval progress MSE was `0.008822`, inserted accuracy
      was `0.948`, and DP-continuable accuracy was `0.870`. This remains a
      trend record only, not a post-floor conclusion.
      Later mid-run check at step `92000` around `1239` seconds still shows
      the same failure direction: eval action MSE `0.010402` versus DP-prior
      `0.001561`, with eval progress MSE `0.009180`, inserted accuracy
      `0.948`, and DP-continuable accuracy `0.909`. The run still has not
      reached the formal `10800` second floor, so this is not the final gate,
      but the trend is no longer an isolated fluctuation.
      2026-06-15 14:45 mid-run check: the current formal run is still active
      and below the `10800` second floor. Latest parsed history has step
      `134000`, elapsed `1802` seconds, eval action MSE `0.011337` versus
      frozen-DP-prior eval MSE `0.0015608` (`7.26x` worse). Eval progress MSE
      remains low (`0.009897`) and inserted/DP-continuable accuracies are
      `0.961/0.883`. All `68/68` eval history points so far have action MSE
      worse than the DP prior; the best observed point is still `1.74x` worse
      than DP prior. Plain current status: contact/progress readouts are
      learnable, but the deterministic action head still cannot justify
      closed-loop execution. Keep the job running to the formal floor and let
      the independent final gate decide; do not run live eval from this
      checkpoint.
- [x] Added a post-training/group-inspection script for the contact executor:
      `scripts/world_model/inspect_cosmos3_contact_executor_training.py`.
      A preliminary inspection of the current `checkpoint_latest.pt` at about
      step `20000` confirms the watchpoint with more detail. Overall eval
      action MSE is `0.005276` versus DP-prior `0.001561`, but progress MSE is
      `0.008699`, inserted accuracy is `0.948`, and DP-continuable accuracy is
      `0.857`. The action regression problem is concentrated in
      `lateral_align` (`0.01519` action MSE versus `9.24e-05` prior) and
      `preinsert_aligned` (`0.00530` versus `0.000245`) phases, and in
      `peg_disturb` and `hole_late_move_stop` scenarios. `far` and `peg_drop`
      are actually better than the DP prior on action MSE. Plain implication:
      contact/progress readouts are useful, but the deterministic action head
      is overriding a strong DP prior in phases where the prior is already
      good. Let the run reach the formal floor before final judgment, but do
      not treat this as permission for live eval yet.
- [x] Extended the same inspection script with a residual-scale sweep. On the
      current latest checkpoint, the unscaled action head has overall eval
      action MSE `0.008319` versus DP-prior `0.001561`, but global residual
      scale `0.05` gives `0.001526`, only a tiny improvement over scale `0`.
      Phase best scales are `far=1.0`, `dp_continuable=0.2`,
      `preinsert_aligned=0.01`, and `lateral_align=0.0`. Scenario best scales
      keep `hole_late_move_stop=0.0` and `peg_disturb=0.0`, while `peg_drop`
      benefits from `1.0`. Plain implication: the residual contains some
      phase-specific signal, but a single deterministic executor is unsafe.
      This supports DP-as-candidate/regularized-prior plus phase/progress
      scoring, not direct execution of the learned action head.
- [x] Ran a separate mid-run checkpoint inspection at `2026-06-15 14:48`
      inside allocation `128023`, writing
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k/post_training_group_metrics_midrun_1445.json`.
      This did not overwrite the final watcher files and did not start live
      eval. Overall checkpoint-latest action MSE was `0.010988` versus
      DP-prior `0.0015608`; the best global positive residual scale was again
      tiny (`0.05`, action MSE `0.001508`). The phase split is the important
      result: `far` improves over DP prior, but `lateral_align` is much worse
      (`0.03224` versus `0.0000924`) and `preinsert_aligned` is much worse
      (`0.01693` versus `0.0002445`). Plain implication: the learned residual
      has phase-specific signal, but the deterministic action head destroys
      already-good DP behavior near insertion. If the final floor gate remains
      false, the aligned next method is phase/contact-conditioned candidate
      generation and scoring, not direct execution of this deterministic head.
      2026-06-15 15:05 mid-run check: the same formal run is still active and
      below the `10800` second floor. Latest parsed history has step `220000`,
      elapsed `2945` seconds, eval action MSE `0.012655` versus DP-prior
      `0.0015608` (`8.11x` worse). Eval progress MSE is `0.010286`, inserted
      accuracy is `0.948`, and DP-continuable accuracy is `0.883`. The latest
      refreshed history summary has all eval points worse than DP prior
      (`111/111` at this check, best ratio still `1.74x` worse). GPU use is
      active but imbalanced (`5-11%` on one H200, `100%` on the other in
      spot checks); do not stop the job because it is progressing and still
      must reach the formal floor. Live eval remains forbidden until final
      `training_summary.json`, `checkpoint_final.pt`, final inspection, and
      `formal_live_eval_gate.json` exist.
- [x] Extended `scripts/world_model/inspect_cosmos3_contact_executor_training.py`
      with oracle offline selector diagnostics. This reports what validation
      action MSE would be if residual scale were selected by a universal
      grouping such as current contact phase, without running live eval and
      without treating it as method evidence. A mid-run latest-checkpoint
      diagnostic wrote
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k/post_training_group_metrics_midrun_selector_1452.json`.
      Results: unscaled action MSE `0.011475`, DP prior `0.0015608`,
      best global positive scale `0.05` gives `0.001505`, while oracle
      current-phase scale selection gives `0.001107`. Phase choices were
      `far=1.0`, `dp_continuable=0.2`, `lateral_align=0.0`, and
      `preinsert_aligned=0.01`. Plain implication: candidate/scale selection
      keyed by universal contact phase has real offline signal and avoids
      damaging already-good DP phases, but this is still offline validation
      only. It must not authorize live eval before the final formal gate.
- [x] Added a stricter train-calibrated selector diagnostic to the same
      inspection script and ran it on the latest mid-run checkpoint:
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k/post_training_group_metrics_midrun_traincal_selector_1457.json`.
      This intentionally selects residual scales on the train split and
      evaluates them on the validation split. Result: naive train-calibrated
      global and phase selectors choose `scale=1.0` because the deterministic
      model has nearly memorized train actions, and validation MSE remains
      bad (`0.011726`, far worse than DP prior `0.0015608`). Plain implication:
      the oracle phase-selector signal is real as a diagnostic, but a naive
      train-MSE scale calibration overfits and is not a usable next method.
      Any candidate selector must be regularized and judged by held-out
      progress/value/contact continuability, not by train action MSE alone.
- [x] Fixed the formal contact-executor gate so progress/contact readout
      success cannot accidentally authorize live eval when actions are worse
      than the DP prior. `scripts/world_model/train_cosmos3_contact_executor.py`
      now requires `eval_action_mse <= eval_baseline_dp_prior_mse` for future
      `ready_for_formal_eval=true` summaries. Because the current training
      process was already running with the old in-memory code, also added an
      independent post-final gate:
      `scripts/world_model/check_cosmos3_contact_executor_formal_gate.py`.
      The watcher now runs group inspection and then writes
      `formal_live_eval_gate.json`; live eval is allowed only if the formal
      floor is met and action/progress/contact gates pass. A dry check before
      `training_summary.json` exists correctly reports
      `live_eval_allowed=false` with `missing_summary_or_group_metrics`.
      Later hardening: the gate now also rejects stale group metrics unless
      `post_training_group_metrics.json` was produced from
      `checkpoint_final.pt`. This prevents an old `checkpoint_latest.pt`
      inspection from accidentally authorizing live eval after the final
      summary appears. The check is path-strict: the group metrics checkpoint
      must resolve to the current training root's `checkpoint_final.pt`, not
      just any file named `checkpoint_final.pt`. Py-compile passed, and a
      mid-run dry gate still correctly reports `live_eval_allowed=false`.
      The final gate payload now also records oracle phase-selector and
      train-calibrated selector MSEs when the inspection file provides them.
      These numbers are diagnostics only and do not relax the action-MSE gate;
      they are included so a failed final gate can clearly distinguish
      "selector has held-out signal" from "selector overfits train actions."
- [x] Prepared the live-eval interface for the contact/progress executor
      without launching closed-loop eval. Before this fix,
      `run_cosmos3_live_receding_loop.py` and the panel wrappers only accepted
      `controller_action_source=residual_executor`, so a hypothetical
      successful contact-executor final gate would still have had no legal
      video path. Added `controller_action_source=contact_executor`, a
      compatible contact-checkpoint loader, causal live contact-context
      construction, and contact action-chunk JSON output with
      progress/inserted/DP-continuable readouts. The live contact context uses
      only current real simulator state plus causal continuability thresholds,
      not future contact labels. Updated the panel/loop Slurm wrapper
      checkpoint guards and video annotation so contact-executor chunks are
      marked `EXECUTOR_ACTIVE`. Verification: py-compile passed for the live
      loop/panel and gate scripts, `bash -n` passed for the two wrappers, a
      compute-node loader smoke loaded the current `checkpoint_latest.pt` as
      `horizon=8`, `feature_dim=218`, `target_dim=56`, and a dummy
      contact-action-chunk smoke produced finite `56` action values plus a
      progress readout. This is interface readiness only; live eval remains
      forbidden until the final formal gate allows it.
      2026-06-15 15:27 mid-run watch: the formal step is still active and
      below the `10800` second floor; `training_summary.json` and
      `checkpoint_final.pt` are still missing, so the formal gate remains
      closed with `missing_summary_or_group_metrics`. Refreshed history has
      `161/161` eval points worse than DP prior. Latest point: step `320000`,
      elapsed `4303.65` seconds, eval action MSE `0.014158` versus DP prior
      `0.0015608` (`9.07x` worse), progress MSE `0.011196`,
      inserted/DP-continuable accuracies `0.948/0.883`. GPU spot check inside
      allocation `128023` showed `100%/12%`; keep the allocation and do not
      stop the run before the formal floor.
      2026-06-15 17:25 finalization blocker: the run has now exceeded the
      formal floor (`128023.63` runtime about `3:09`) but did not produce a
      valid formal result. `training_summary.json` is still missing,
      `checkpoint_final.pt` exists but is only `15895` bytes and fails
      `torch.load` with `OSError(22, 'Invalid argument')`, while
      `checkpoint_latest.pt` is valid at step `780001` and about `199M`.
      The last history point is step `796000`, elapsed `10796.32` seconds,
      eval action MSE `0.015157` versus frozen-DP-prior MSE `0.0015608`
      (`9.71x` worse), and `399/399` eval points were worse than DP prior.
      A compute-node check showed both training ranks still alive at `100%`
      GPU after the final file stopped growing. Plain blocker: final
      checkpoint/summary writing or final DDP synchronization is stuck or
      corrupted, and the offline action trend would fail the live gate even if
      finalization were repaired. Do not run contact-executor live eval from
      this run. Stop for user direction before interrupting/restarting or
      substituting `checkpoint_latest.pt` for the formal final checkpoint.
      2026-06-15 17:28 final outcome: the stuck formal step exited by itself
      after NCCL watchdog timeout, so no manual `Ctrl-C` was sent and the held
      allocation remains available. Console evidence:
      rank1 timed out on `ALLREDUCE NumelIn=1`, rank0 timed out on
      `ALLREDUCE NumelIn=4323388`, then torchrun exited with
      `ChildFailedError` / rank0 `SIGABRT`, and Slurm step `128023.63`
      terminated with exit code `1`. `checkpoint_final.pt` grew to about
      `54M` but still fails `torch.load` with
      `PytorchStreamReader failed reading zip archive: failed finding central
      directory`; `training_summary.json` is still missing; both GPUs are now
      idle. Root cause is an implementation bug in the trainer: the DDP ranks
      can make different stop decisions near the wall-clock floor, causing one
      rank to enter final save while the other continues gradient allreduce.
      Code fix added in
      `scripts/world_model/train_cosmos3_contact_executor.py`: stop decisions
      are synchronized across ranks with a scalar all-reduce, JSON/checkpoint
      writes are atomic temp-file renames, and final checkpoint is loaded once
      after save before summary writing. `py_compile` passed. This repairs the
      repeatability bug for a future rerun but does not rescue this failed
      formal run or authorize live eval.
      2026-06-15 17:31 monitor hygiene: fixed the read-only decision summary
      and formal watcher so a failed run with an invalid final checkpoint is
      no longer reported as "waiting for summary." The refreshed decision
      summary now reports
      `status=failed_invalid_final_checkpoint_stop_for_user`, blockers
      `training_summary.json`, `invalid_checkpoint_final`, and
      `training_process_failed`, with `live_eval_allowed=false`. The old
      already-running watcher was stopped because it had loaded the previous
      wait-loop logic. The separate decision watcher received the same
      failed-state exit guard and its stale tmux process was stopped as well;
      the Slurm allocation was not cancelled.
- [x] Added a guarded contact-executor live-panel launcher without running it:
      `scripts/slurm/run_cosmos3_contact_executor_live_panel_after_gate_in_allocation.sh`.
      The script refuses login-node execution, requires the current formal
      `training_summary.json`, `checkpoint_final.pt`,
      `post_training_group_metrics.json`, and
      `formal_live_eval_gate.json`, and exits unless
      `live_eval_allowed=true`. It also binds the future live panel to the
      clean-dense Cosmos SFT `iter_000001500` root and the current formal
      contact-executor `checkpoint_final.pt`, avoiding the older wrapper
      defaults that still point at the stale 6/12 `iter2700` chain. `bash -n`
      passed. A refusal-path check with Slurm-like environment variables exited
      before live launch with `missing_training_summary=.../training_summary.json`.
      This is launch hygiene only; no video evidence exists for the contact
      executor until the final gate opens and a live panel is actually run and
      inspected.
- [x] Restarted watcher tmux session
      `cosmos3_contact_executor_formal_watch_20260615` after adding the gate.
      Its log currently shows it is polling the current formal root every
      `300` seconds. It still does not launch live eval.
- [x] Added a history trend summarizer:
      `scripts/world_model/summarize_cosmos3_contact_executor_history.py`.
      On the current mid-run history it reports `52/52` eval points with
      action MSE worse than the DP-prior baseline. The best observed eval
      action MSE is still `1.74x` worse than DP prior, and the latest point at
      the time of the summary is `6.76x` worse. This is still pre-floor trend
      evidence, not a final result, but it prepares a clear post-final blocker
      report if the final gate remains false. The formal watcher now runs this
      trend summary before final group inspection and gate writing.
- [x] Added and launched a formal-result watcher:
      `scripts/slurm/watch_cosmos3_contact_executor_formal_inspect.sh`, tmux
      session `cosmos3_contact_executor_formal_watch_20260615`. It polls
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k`
      every `300` seconds and, after `training_summary.json` plus
      `checkpoint_final.pt` exist, runs the group-inspection script inside the
      same held allocation `128023`. It does not launch live eval. Log:
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k/formal_watch.log`.
- [x] Added a read-only final-decision summarizer:
      `scripts/world_model/summarize_cosmos3_contact_executor_decision.py`.
      It reads the training summary, final gate, group inspection, and history
      summary and writes `formal_decision_summary.json/md` with one of three
      states: waiting for final files, gate open for guarded live eval, or
      gate closed and stop-for-user. It does not relax the formal gate and
      does not launch live eval. Py-compile passed. Running it on the current
      pre-final root wrote
      `experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k/formal_decision_summary.md`
      with `status=waiting_for_formal_floor_or_final_files` and blockers
      `training_summary.json, checkpoint_final.pt`.
- [x] Wired final-decision summarization into the watcher path without
      launching live eval. Updated
      `scripts/slurm/watch_cosmos3_contact_executor_formal_inspect.sh` so that
      after `formal_live_eval_gate.json` is written it also writes the plain
      `formal_decision_summary.json/md`. Because the already-running watcher
      may not reread the modified script while sleeping, added and launched a
      companion lightweight tmux watcher
      `cosmos3_contact_executor_decision_watch_20260615` using
      `scripts/slurm/watch_cosmos3_contact_executor_decision_summary.sh`. It
      polls every `300` seconds for the final summary, final checkpoint, and
      formal gate, then runs the read-only decision summarizer. Syntax checks
      passed. It does not use GPU and does not launch live eval. Later fixed
      both decision-summary watcher paths so a final `gate_closed_stop_for_user`
      return code is logged as `rc=2` rather than treated as a watcher crash;
      the companion watcher was restarted at `2026-06-15T15:42:27+08:00` with
      the updated script.
- [ ] After contact/progress labels exist, train a stronger executor where
      frozen DP is a prior/regularizer or candidate source, not a hard center.
      The offline gate should judge progress/contact/continuability and final
      predicted success, not just action MSE against a teacher chunk.
- [ ] Latest user override, 2026-06-14: current highest priority is to run
      full Cosmos3 SFT from the already generated dense 733 condition data,
      then immediately run closed-loop eval. The minimum formal training
      standard is now `2` GPUs for at least `3` hours; use `4` GPUs only if
      already available sooner. Do this before more compositional-teacher work.
- [ ] Preferred dense SFT input is the latest structurally valid dense root:
      `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050`
      with preflight root
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_preflight_20260614_130050`.
      It has `9271` rows, strict `301/300` preflight passed, and wider
      late-prefix coverage than the first `9223`-row dense root.
- [ ] The remaining live-query coverage gap (`58/173` undercovered) is now a
      recorded limitation for this run, not a startup blocker. Do not hide it
      and do not call the result final method success solely because full SFT
      runs. The purpose is to get the best current dense-data Cosmos3 baseline
      and then measure it in closed loop.
- [ ] If the current full-SFT wrapper refuses the late299 dense root because
      the old ready gate treats coverage as blocking, add an explicit
      documented user-override gate for this run only. The override must still
      require structural `301/300` preflight, action/sidecar audit, source
      provenance, and the recorded coverage-gap JSON/summary.
- [x] Implemented the explicit full-SFT-only live-query coverage override in
      `scripts/world_model/check_cosmos3_clean_dense_preflight_summary_ready.py`,
      `scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_full_fix1recipe_in_allocation.sh`,
      and `scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh`.
      The override accepts only the two live-query undercoverage failures and
      still requires the structural safe checks to be present and passing.
- [x] First replacement 4-GPU allocation `127817` started on `server13`, but
      PyTorch CUDA canaries failed in both `.venv_cosmos313` and `.venv`:
      `torch.cuda.is_available()=False` even though `CUDA_VISIBLE_DEVICES=0,1,2,3`
      and `nvidia-smi` saw four H200s. The dense full-SFT step was stopped with
      `Ctrl-C`; no checkpoint or usable training evidence was produced from
      `server13`.
- [x] Latest resource override: the formal full-training floor is now `2`
      GPUs for at least `3` hours, with `4` GPUs still preferred if already
      available sooner. A second tmux-held `2` GPU allocation was requested
      and granted as job `127821` on `server24`. CUDA canary passed, but the
      first 2-GPU SFT launch reached `torchrun` and then sat in
      model/config I/O with `0` useful GPU memory for more than eight minutes.
      When the 4-GPU allocation became available, the 2-GPU step was stopped
      with `Ctrl-C` and the empty allocation was released.
- [x] First `server35` 4-GPU launch exposed an implementation bug, not a data
      bug: the wrapper changed directory into `external/cosmos-framework` and
      passed relative JSONL paths to Cosmos, so Cosmos looked under
      `external/cosmos-framework/experiments/...` and failed with
      `FileNotFoundError`. Fixed
      `scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh` to
      canonicalize the dataset, condition, output, preflight-summary, train
      JSONL, and val JSONL paths to absolute paths before training.
- [ ] Current formal dense full SFT run is now the 4-GPU `server35` pathfix
      run:
      tmux session `cosmos3_clean_dense_4gpu_20260614`, Slurm job `127819`,
      output root
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299`.
      CUDA canary passed on `server35`; strict `301/300` preflight, action
      target audit, receding-distribution audit, and clean-dense readiness
      gate have passed. `torchrun --nproc_per_node=4` started after the
      pathfix; Cosmos successfully loaded absolute train/val JSONL metadata and
      dataloader prewarm completed. GPU memory rose to about `10-11GB` on all
      four H200s at `2026-06-14T17:57:19+08:00`; checkpoint load completed,
      training started, validation iteration `0` reported loss `3.585211`, and
      training reached at least iteration `21` by `2026-06-14T18:07:38+08:00`.
      Rank-0 loss fell from `3.9498` at iteration `1` to `1.5358` at
      iteration `21`; a GPU probe at `2026-06-14T18:07:44+08:00` showed all
      four GPUs at `100%` utilization with about `59-60GB` used each. This is
      still not method
      evidence: monitor until checkpoint logs appear and post-SFT generated
      video/action/readout eval plus closed-loop eval are completed.
- [ ] A separate 1-GPU eval allocation was requested through tmux session
      `cosmos3_clean_dense_eval_1gpu_20260614` and granted as Slurm job
      `127825` on `server24`. The eval watcher
      `scripts/slurm/watch_cosmos3_clean_dense_late299_iter300_eval_in_allocation.sh`
      is running in compute step `127825.0`, passed a CUDA canary
      (`torch=2.10.0+cu128`, `cuda_available=True`, `device_count=1`,
      `device0=NVIDIA H200`), and is waiting for
      `iter_000000300` from the current dense SFT. Its eval root is
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/eval_full_episode_wam_iter_000000300`.
      The watcher will only produce generated validation artifacts; this is
      not closed-loop method evidence until strict inspection, visual review,
      readout, and live-receding closed-loop eval are completed.
- [ ] The initial eval-only watcher in step `127825.0` was stopped without
      releasing the allocation because it would have required manual follow-up
      after generated eval. A replacement sequential pipeline is now running in
      compute step `127825.1`:
      `scripts/slurm/run_cosmos3_clean_dense_late299_iter300_eval_readout_live_pipeline_in_allocation.sh`.
      It waits for `iter_000000300`, runs strict generated full-episode eval,
      then runs the current-run readout wrapper, then launches the corrected
      live-receding closed-loop panel. The pipeline manifest is under
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/iter300_eval_readout_live_pipeline`.
      This still does not prove method success; video/contact-sheet inspection
      and real final-state metrics remain required after the panel finishes.
- [ ] Important evidence boundary: the `iter_000000300` pipeline is an early
      interface/visual diagnostic only. It is useful for catching generated
      video/action/readout/live-receding failures quickly, but it must not be
      reported as formal method evidence because the active training run has
      not yet satisfied the `2` GPU / `3` hour minimum at that checkpoint. A
      formal closed-loop claim needs a checkpoint produced after the minimum
      training-time floor is satisfied, plus the same generated-artifact
      inspection, readout diagnostics, live-receding final-state metrics, and
      inspected video/contact-sheet evidence.
- [ ] Formal-checkpoint rule for this run: the dense SFT log reports
      `Starting training...` at `2026-06-14T17:57:46+08:00`, so the minimum
      formal-training wall-clock boundary is `2026-06-14T20:57:46+08:00`.
      When later checkpoints appear, choose the first checkpoint whose actual
      save timestamp is after that boundary for formal closed-loop evidence
      (likely `iter_000000600` only if its save timestamp proves it crossed
      the boundary; otherwise use a later checkpoint such as `iter_000000900`).
      The same generated/readout/live-receding/video gates apply.
- [ ] The eval allocation has been switched from the single-stage iter300
      pipeline to the master pipeline in compute step `127825.2`:
      `scripts/slurm/run_cosmos3_clean_dense_late299_iter300_then_formal_pipeline_in_allocation.sh`.
      It first runs the `iter_000000300` early diagnostic pipeline, then waits
      for the first checkpoint whose actual `.metadata` mtime is after
      `2026-06-14T20:57:46+08:00` and runs the same generated/readout/live
      evidence chain for that formal checkpoint. The master manifest is under
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/iter300_then_formal_eval_readout_live_pipeline`.
- [ ] Latest live status at `2026-06-14T18:35:58+08:00`: the user-updated
      formal training floor is confirmed in `AGENTS.md`, this TODO, and the
      plan as `2` GPUs for at least `3` hours. The current run already has a
      valid earlier `4` GPU allocation, so it is continuing on that faster
      resource rather than waiting for another 2-GPU slot. Training job
      `127819` is still active on `server35`, reached iteration `119` by
      `2026-06-14T18:35:32+08:00`, and rank-0 loss was `0.2512`. A GPU probe
      showed the four H200s at `100%`, `98%`, `96%`, and `100%` utilization
      with about `59GB` memory used each. Eval job `127825`, step `127825.2`,
      is active on `server24` and waiting for `iter_000000300`. No checkpoint,
      generated videos, readout outputs, live-receding videos, or contact
      sheets exist yet for this run, so there is nothing visual to inspect at
      this moment.
- [ ] Follow-up status at `2026-06-14T18:38:23+08:00`: training continued to
      iteration `128`, rank-0 loss `0.2528`, and a GPU probe showed all four
      training H200s at `100%` utilization with about `59GB` memory used each.
      The eval allocation was alive but idle at `0%` GPU because it is waiting
      for the first saved checkpoint; this is expected at this stage and is
      not a method/data blocker. The master/eval/formal wrapper chain was
      inspected read-only: generated eval resolves and passes `CHECKPOINT_PATH`
      into Cosmos inference, and the formal stage selects a checkpoint whose
      saved metadata time is after `2026-06-14T20:57:46+08:00`, so there is no
      current evidence of a wrapper path mix-up between `iter_000000300` and a
      later formal checkpoint.
- [ ] Follow-up status at `2026-06-14T18:40:04+08:00`: training reached
      iteration `134`, rank-0 loss `0.3035`, with no checkpoint directory or
      video artifact yet. The saved Cosmos config was inspected and confirms
      `checkpoint.save_iter: 300`, `trainer.max_iter: 1500`, and
      `trainer.validation_iter: 300`; therefore the eval pipeline is waiting
      for the intended first checkpoint and not for a nonexistent save
      interval. No visual evidence exists yet to inspect.
- [ ] Follow-up status at `2026-06-14T18:43:06+08:00`: training job `127819`
      remained active on `server35`, reached iteration `144` by
      `2026-06-14T18:42:39+08:00`, and rank-0 loss was `0.1699`. A GPU probe
      showed all four training H200s at `100%` utilization with about `59GB`
      memory used each. Eval job `127825`, step `127825.2`, remained active on
      `server24` and was still waiting for `iter_000000300`; its GPU was idle
      because no checkpoint exists yet. The eval allocation has been idle for
      about 30 minutes, below the 3-hour low-utilization release window. No
      generated videos, live-receding videos, or contact sheets exist yet.
- [ ] Follow-up status at `2026-06-14T18:44:02+08:00`: training continued to
      iteration `148`, rank-0 loss `0.2491`. Slurm still showed training step
      `127819.7` and eval step `127825.2` active. The checkpoint directory was
      still empty, so the eval pipeline correctly remained in
      `checkpoint_not_ready` waiting for `iter_000000300`. No video/contact
      evidence exists yet.
- [ ] Follow-up status at `2026-06-14T18:45:20+08:00`: Slurm allocation
      limits were checked. Training job `127819` has `2-00:00:00` reserved
      until `2026-06-16T17:43:33`; eval job `127825` has `1-00:00:00`
      reserved until `2026-06-15T18:12:33`. The training GPUs remained at
      `100%` utilization on all four H200s. The eval GPU remained idle only
      because `iter_000000300` is not saved yet; this is still expected, not a
      data/model/eval failure.
- [ ] Follow-up status at `2026-06-14T18:46:34+08:00`: training continued to
      iteration `157`, rank-0 loss `0.1316`, and all four training H200s were
      still at `100%` utilization with about `59GB` memory used each.
      Checkpoint output was still empty, so eval step `127825.2` remained in
      `checkpoint_not_ready` for `iter_000000300`. No generated videos or
      contact sheets exist yet.
- [ ] Follow-up status at `2026-06-14T18:47:32+08:00`: training continued to
      iteration `161`, rank-0 loss `0.1408`. Slurm still showed training step
      `127819.7` and eval step `127825.2` active. Training GPUs remained at
      `100%` utilization; eval GPU remained idle only because the first
      checkpoint is not saved yet. Checkpoint output and generated visual
      artifacts are still absent.
- [ ] Follow-up status at `2026-06-14T18:49:05+08:00`: the updated training
      resource rule remains `2` GPUs / `3` hours minimum, not `4` GPUs. The
      current run is continuing on `4` GPUs only because that allocation is
      already running and faster. Training job `127819` remained active on
      `server35`, step `127819.7`, reached iteration `166` by
      `2026-06-14T18:48:55+08:00`, and rank-0 loss was `0.2013`. Eval job
      `127825`, step `127825.2`, remained active on `server24` and was still
      waiting for `iter_000000300` (`checkpoint_not_ready` through
      `1020` seconds). There is still no checkpoint, generated video,
      readout output, live-receding video, or contact sheet to inspect.
- [ ] Follow-up status at `2026-06-14T19:40:32+08:00`: early diagnostic
      checkpoint `iter_000000300` was saved at `2026-06-14T19:27:15+08:00`
      under the current `server35` SFT root. The eval master pipeline on
      `server24` waited for stable checkpoint files, then ran generated
      full-episode eval on 10 validation samples. Strict artifact inspection
      passed with `strict_eval_artifacts_ok=true`, `0` strict failures,
      `301` generated video frames per sample, and action shape `[300, 32]`.
      Mean diagnostics from `eval_artifact_inspection.md`: action RMSE
      `0.4346`, robot future RMSE `0.7215`, state-sidecar future RMSE
      `0.4517`, and future-video PSNR `20.34`. This is still early diagnostic
      evidence only because it was saved before the `2026-06-14T20:57:46+08:00`
      formal-training boundary.
- [ ] Visual review of the iter300 generated eval was completed by opening all
      10 generated-vs-reference review sheets under
      `eval_full_episode_wam_iter_000000300/review_sheets`. The videos are
      nonblank, use the expected viewpoint, and are frame-aligned. Visual
      conclusion: several insert/continuous-insert samples are qualitatively
      close to the reference, but pre-motion/static/peg-recovery samples show
      visible final peg/TCP pose errors. This proves the generated-video eval
      path is working; it does not prove closed-loop insertion success.
- [ ] Current readout status at `2026-06-14T19:40:32+08:00`: after strict
      generated artifacts passed, the eval pipeline advanced to
      `watch_cosmos3_clean_dense_late299_iter300_readout_in_allocation.sh`.
      It launched
      `scripts/world_model/train_cosmos3_task_state_readout.py` into
      `task_state_readout_fix3_v7_733_rgb_301f` with `2000` steps and
      `--require-cuda`. The process is currently active on `server24`; no
      readout metrics or live-receding panel artifacts exist yet.
- [ ] Follow-up status at `2026-06-14T19:52-20:01+08:00`: the readout
      diagnostic was not allowed to block the higher-priority closed-loop
      eval. It stayed in H5/model-data loading with no useful GPU work and no
      readout metrics, so step `127825.2` was interrupted inside the held tmux
      allocation without releasing job `127825`. This is an eval/readout
      implementation/IO blocker, not evidence that the dense SFT data or
      Cosmos checkpoint failed.
- [ ] The iter300 live-receding diagnostic panel was then launched in the
      same held eval allocation as step `127825.27`, output root
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_full300_panel_iter_000000300_clean_dense_20260614_195251`.
      It passed the strict generated-eval input check and entered real
      receding execution on sample `00_hole_late_move_stop`. Early partial
      status after two Cosmos chunks: target motion was detected at frame
      `106`, two 8-step chunks produced action JSON and observed-prefix video
      artifacts, but the real peg-head-in-hole-frame state moved farther from
      the hole (`y` about `0.030 -> 0.063m`), final/partial success is still
      false, and the real-state `C_pi` gate still rejects DP handoff. This is
      early iter300 diagnostic failure evidence only; let the panel finish and
      inspect video/contact sheets before recording a final closed-loop
      conclusion.
- [ ] Completed iter300 live-receding diagnostic sample
      `sample_00_hole_late_move_stop`: `25` Cosmos receding queries,
      `301` observed frames, `WM_ACTIVE=186` frames, `DP_HANDOFF=8` frames,
      `final_success=false`, final peg-head-in-hole-frame
      `[-0.1067, 0.0102, -0.0502]`, and
      `method_evidence_allowed=false`. The final annotated video passed file
      inspection (`301` frames, `30fps`) and a contact sheet was generated and
      opened:
      `live_receding_full300_panel_iter_000000300_clean_dense_20260614_195251/sample_00_hole_late_move_stop/live_observed_rollout_annotated_sheet.png`.
      Visual review agrees with the metric: the peg is not inserted at the
      end and remains visibly offset from the hole. This is an early
      checkpoint closed-loop failure, not formal post-3-hour method evidence.
- [ ] Completed iter300 live-receding diagnostic sample
      `sample_01_hole_late_constant`: `27` completed iterations, `301`
      observed frames, `WM_ACTIVE=36` frames, `DP_HANDOFF=170` frames,
      `final_success=true`, final peg-head-in-hole-frame
      `[0.0274, 0.0023, -0.0010]`, and `method_evidence_allowed=false`.
      A contact sheet was generated and opened:
      `live_receding_full300_panel_iter_000000300_clean_dense_20260614_195251/sample_01_hole_late_constant/live_observed_rollout_annotated_sheet.png`.
      Visual review agrees with the metric: WM gets the run to a DP-handoff
      state and the later DP segment keeps the peg at the hole. This is an
      early diagnostic positive sample only, not formal evidence.
- [ ] Completed iter300 live-receding diagnostic sample
      `sample_03_hole_late_fast_shift`: `301` observed frames,
      `final_success=false`, final peg-head-in-hole-frame
      `[-0.1465, -0.0190, 0.0064]`, and `method_evidence_allowed=false`.
      A contact sheet was generated and opened:
      `live_receding_full300_panel_iter_000000300_clean_dense_20260614_195251/sample_03_hole_late_fast_shift/live_observed_rollout_annotated_sheet.png`.
      Visual review agrees with the metric: the peg remains outside and offset
      from the hole at the end. This is an early checkpoint closed-loop
      failure, not formal post-3-hour method evidence.
- [ ] Completed iter300 live-receding diagnostic sample
      `sample_04_hole_late_sine`: `23` completed iterations, `301` observed
      frames, `final_success=false`, final peg-head-in-hole-frame
      `[-0.1084, -0.0209, -0.0019]`, and `method_evidence_allowed=false`.
      A contact sheet was generated and opened:
      `live_receding_full300_panel_iter_000000300_clean_dense_20260614_195251/sample_04_hole_late_sine/live_observed_rollout_annotated_sheet.png`.
      Visual review agrees with the metric: the peg approaches the hole but
      remains outside/offset at the end. This is an early checkpoint
      closed-loop failure, not formal post-3-hour method evidence.
- [ ] Iter300 live-receding diagnostic panel completed all `4` requested
      samples with `panel_full_episode_contract_ok=true`,
      `sample_contract_failures=[]`, `failed_process_count=0`,
      `final_success_count=1`, and `method_evidence_allowed=false`. The panel
      result is useful for failure diagnosis only: one early positive
      (`sample_01`) and three visually confirmed failures (`sample_00`,
      `sample_03`, `sample_04`).
- [ ] Formal post-3-hour generated-eval plus live-receding pipeline was started
      in the held eval allocation at `2026-06-14T22:09:11+08:00`, Slurm job
      `127825`, step `59`, node `server24`, via
      `scripts/slurm/run_cosmos3_clean_dense_late299_formal_eval_live_pipeline_in_allocation.sh`.
      It is waiting for the first checkpoint whose model `.metadata` mtime is
      after `2026-06-14T20:57:46+08:00`; current status at launch was
      `formal_checkpoint_not_ready`.
- [ ] Formal pipeline selected checkpoint `iter_000000900` with model
      `.metadata` mtime `2026-06-14T22:25:29+08:00`, which is after the
      formal boundary. Formal generated eval started at
      `2026-06-14T22:29:12+08:00` under
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/eval_full_episode_wam_iter_000000900_formal_after_3h`.
- [ ] Formal `iter_000000900` generated eval completed at
      `2026-06-14T22:34:54+08:00`: `10/10` samples inspected,
      `strict_eval_artifacts_ok=true`, `strict_failures=[]`, video frames
      `301`, action shape `[300, 32]`. Mean diagnostics: action RMSE
      `0.4523`, robot-action future RMSE `0.7967`, state-sidecar future RMSE
      `0.4393`, future video PSNR `22.5858`. Review sheets were opened via
      `review_sheets/combined_iter900_ref_pred_sheet.png` plus representative
      original sheets. Visual result: generated videos are nonblank and
      frame-aligned, but several samples show visible end-effector/peg offsets;
      this is valid generated-eval evidence, not closed-loop task success.
- [ ] Formal `iter_000000900` live-receding closed-loop panel started at
      `2026-06-14T22:34:54+08:00`, output root
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_full300_panel_iter_000000900_clean_dense_20260614_223454`.
      Early partial status for `sample_00_hole_late_move_stop` at frame `123`
      is still failure: peg-head-in-hole-frame `[-0.2273, 0.0672, 0.0033]`
      and `C_pi`/handoff gate false. Let it finish before making final
      closed-loop claims.
- [ ] Formal `iter_000000900` live-receding sample
      `sample_00_hole_late_move_stop` completed: `25` iterations,
      `301` observed frames, `final_success=false`, final
      peg-head-in-hole-frame `[-0.0173, -0.0029, -0.0017]`. Contact sheet
      generated and opened:
      `live_receding_full300_panel_iter_000000900_clean_dense_20260614_223454/sample_00_hole_late_move_stop/live_observed_rollout_annotated_sheet.png`.
      Visual review: the run improves substantially over iter300 and reaches a
      near-handoff/near-insertion state, but the authoritative real final-state
      metric remains false, so it is a near miss rather than success.
- [ ] Formal `iter_000000900` live-receding sample
      `sample_01_hole_late_constant` completed: `26` iterations,
      `301` observed frames, `WM_ACTIVE=38`, `DP_HANDOFF=168`,
      `final_success=false`, final peg-head-in-hole-frame
      `[-0.0198, 0.0026, 0.0025]`. Contact sheet generated and opened:
      `live_receding_full300_panel_iter_000000900_clean_dense_20260614_223454/sample_01_hole_late_constant/live_observed_rollout_annotated_sheet.png`.
      Visual review: the run reaches the hole region but remains a real-state
      failure. Unlike iter300, this sample is not a success at `iter900`.
      Formal closed-loop status after two samples is `0/2` success, with both
      failures near the threshold but still failures.
- [ ] Formal `iter_000000900` live-receding sample
      `sample_03_hole_late_fast_shift` completed: `21` iterations,
      `301` observed frames, `final_success=false`, final
      peg-head-in-hole-frame `[-0.1349, -0.0934, -0.0580]`. Contact sheet
      generated and opened:
      `live_receding_full300_panel_iter_000000900_clean_dense_20260614_223454/sample_03_hole_late_fast_shift/live_observed_rollout_annotated_sheet.png`.
      Visual review agrees with the metric: the peg remains outside and offset
      from the hole. Formal closed-loop status after three samples is `0/3`
      success.
- [ ] Formal `iter_000000900` live-receding sample
      `sample_04_hole_late_sine` completed: `23` iterations, `301` observed
      frames, `final_success=false`, final peg-head-in-hole-frame
      `[-0.3264, 0.1223, -0.0559]`. Contact sheet generated and opened:
      `live_receding_full300_panel_iter_000000900_clean_dense_20260614_223454/sample_04_hole_late_sine/live_observed_rollout_annotated_sheet.png`.
      Visual review agrees with the metric and is more severe than a small
      insertion-threshold miss: the peg is no longer held through the late
      WM-active phase, and the robot ends up pushing near the block without a
      valid insertion. This is a closed-loop execution/grasp-hold failure, not
      a generated-artifact length or preflight failure.
- [ ] Formal `iter_000000900` live-receding panel completed at
      `2026-06-15T00:12:52+08:00`: `completed_samples=4`,
      `final_success_count=0`, `method_evidence_allowed=false`,
      `panel_full_episode_contract_ok=true`, `sample_contract_failures=[]`,
      and `failed_process_count=0`. Plain conclusion: the dense SFT and
      generated-eval path ran under the corrected full-episode contract, but
      the closed-loop controller still fails in real simulator execution.
      Two cases are near misses around the hole (`sample_00`, `sample_01`);
      one remains clearly offset (`sample_03`); one loses/does not preserve
      grasp and ends far away (`sample_04`). The current blocker is therefore
      not "the data never started" or "the 301/300 eval is broken"; it is that
      raw Cosmos-predicted action chunks plus the current DP handoff do not
      reliably preserve grasp/contact and finish insertion after dynamic
      target motion.
- [ ] Follow-up formal eval for later checkpoint `iter_000001200` was started
      in the same held eval allocation at `2026-06-15T00:17:26+08:00`,
      Slurm job `127825`, step `70`, node `server24`, using the same
      generated-eval plus live-receding pipeline. The checkpoint model
      `.metadata` mtime is `2026-06-14T23:54:26+08:00`, after both the
      formal 3-hour boundary and the `iter_000000900` result. The first
      launch attempt requested `120G` inside a `100G` allocation and was
      rejected by Slurm before running; the active retry uses `90G` and
      selected `iter_000001200` correctly. Await generated-artifact
      inspection, video review, and closed-loop real final-state results
      before interpreting it.
- [ ] Formal `iter_000001200` generated eval completed at
      `2026-06-15T00:25:23+08:00`: `10/10` samples inspected,
      `strict_eval_artifacts_ok=true`, `strict_failures=[]`, video frames
      `301`, action shape `[300, 32]`. Mean diagnostics: action RMSE
      `0.4521`, robot-action future RMSE `0.7785`, state-sidecar future RMSE
      `0.4497`, future video PSNR `22.5841`. A combined review sheet was
      generated and opened:
      `eval_full_episode_wam_iter_000001200_formal_after_3h/review_sheets/combined_iter1200_ref_pred_sheet.png`.
      Representative original sheets `00`, `03`, `08`, and `09` were opened.
      Visual conclusion: outputs are nonblank and frame-aligned; some
      insertion/resume samples are close to reference, but visible pose errors
      remain in pre-motion/static-monitor cases. This is valid generated-eval
      evidence, not live task success.
- [ ] Formal `iter_000001200` live-receding panel started at
      `2026-06-15T00:25:23+08:00`, output root
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_full300_panel_iter_000001200_clean_dense_20260615_002523`.
      Current early artifact status: `sample_00_hole_late_move_stop` has
      entered `iter_00_prefix_f106`, meaning the first closed-loop WM query
      conditions on the real observed prefix through frame `106`; it has
      written `observed_prefix.mp4`, raw action/state JSON, and a pre-controller
      live state snapshot. Await final-state metrics and video/contact-sheet
      review before interpreting the checkpoint.
- [ ] Formal `iter_000001200` live-receding sample
      `sample_00_hole_late_move_stop` completed at
      `2026-06-15T01:12:56+08:00`: `25` completed iterations, `301` observed
      frames, `final_success=false`, final peg-head-in-hole-frame
      `[-0.1085, -0.0441, -0.0400]`, controller frames
      `INIT_OBS=1`, `DP_SCAN_TARGET=106`, `WM_ACTIVE=194`, and
      `dp_handoff_executed_steps=0`. A readable `301`-frame annotated video
      was inspected via
      `sample_00_hole_late_move_stop/video_review/live_observed_rollout_annotated_review_sheet.png`.
      Visual review agrees with the metric: the peg is visibly not inserted
      and the robot ends below/aside the hole block. Compared with `iter900`,
      this later checkpoint is worse on this sample because it never reaches
      the near-handoff state and never opens DP handoff.
- [ ] Training passed the 3-hour wall-clock floor at `2026-06-14T20:57:46+08:00`,
      but checkpoint `iter_000000600` was saved at `2026-06-14T20:56:15+08:00`
      with model `.metadata` at `2026-06-14T20:56:11+08:00`, so it is still
      before the formal boundary. Do not use `iter_000000600` as formal
      method evidence. Continue training and use the next checkpoint whose
      actual metadata timestamp is after the boundary, likely
      `iter_000000900`.
- [ ] Runtime/implementation note from the same live panel: the current
      wrapper reloads Cosmos for each receding query. It is valid for
      diagnosis but not an efficient deployment implementation. A deployable
      version must keep the model resident or batch/cache the receding calls;
      do not confuse this wrapper overhead with the intended low-frequency WM
      runtime design.
- [x] Added
      `scripts/slurm/run_cosmos3_clean_dense_late299_formal_eval_live_pipeline_in_allocation.sh`
      so the formal post-3-hour stage can run generated eval and live-receding
      closed-loop eval without blocking on the non-authoritative readout
      diagnostic. The script still requires a checkpoint whose actual
      `.metadata` timestamp is after `2026-06-14T20:57:46+08:00`, waits for
      stable checkpoint files, runs strict generated eval, then runs the same
      live panel. `bash -n` passed.
- [ ] Current readout status: no existing `best_model.pt` task-state readout
      checkpoint was found under the active Cosmos3 experiment roots by
      `rg --files`. Do not block the iter300 generated-video/action eval on
      this. After strict generated eval artifacts exist, either locate a valid
      current readout checkpoint or train one from
      `experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612/manifest.json`
      inside a compute allocation, then run generated-RGB readout diagnostics.
- [ ] After full SFT starts, monitor checkpoints/validation loss and run
      closed-loop eval from the strongest available checkpoint. Closed-loop
      eval must include real final-state metrics, pure-DP comparison where
      applicable, controller annotations, and video/contact-sheet inspection.
- [ ] This is a new proposed method branch. Do not mix its results with the old
      `cosmos3_300f_world_model` direct raw-Cosmos-action evidence.
- [ ] Current iter2700 status: implementation contract is repaired, but method
      effectiveness is false. Use it as failure evidence and query-coverage
      input, not as a successful checkpoint.
- [x] Clean/dense branch launch started on 2026-06-14 in tmux session
      `cosmos3_clean_dense_4gpu_20260614`, Slurm job `127723`, run root
      `experiments/world_model_task_rebinding/cosmos3/clean_dense_733_overfit_then_full_20260614_115421`.
- [x] Full clean/dense preflight exported `733` source episodes into `9223`
      dense rows with `0` role/mode mismatches, but failed live-query coverage:
      `63/173` undercovered live Cosmos queries. This was previously treated
      as blocking full SFT. The latest user override makes dense full SFT the
      priority anyway, with the coverage gap recorded as a limitation.
- [x] Short overfit wrapper bug found and fixed: selecting the first two source
      episodes gave an empty val JSONL. The overfit-only wrapper now samples two
      valid rows from the full 733 condition root and writes the same rows to
      train and val.
- [x] Short 2-GPU overfit diagnostic passed as a pipeline sanity check from
      `experiments/world_model_task_rebinding/cosmos3/clean_dense_733_overfit_only_20260614_122020`.
      The iter50 checkpoint was saved, validation loss fell from `3.545426` to
      `0.469846`, corrected iter50 eval generated both overfit roles, strict
      artifact inspection passed, and both review sheets were opened.
      This is still only a pipeline sanity check, not method evidence. Under
      the latest user override, full SFT proceeds from the late299 dense root
      with the remaining coverage gap recorded instead of blocking launch.
- [x] Eval sampling bug fixed: small overfit eval was initially selecting the
      same `peg_recovery` row twice because duplicate desired role/scenario
      pairs could reuse one source row. The eval input builder now avoids
      duplicate row UUIDs and preserves all candidates when the candidate count
      is already at or below the requested panel size.
- [x] Late-prefix 733-only diagnostic completed after exposing
      `MAX_PREFIX_FRAMES` in the SFT export wrapper. With
      `MAX_PREFIX_FRAMES=299`, the condition root grew only from `9223` to
      `9271` rows. Structural preflight and action/sidecar audit passed, but
      live-query coverage still failed: `58/173` undercovered. This improved
      only `5` queries, so the main issue is not the old frame-260 cap. Under
      the latest user override, this late299 dense root is the preferred input
      for immediate full SFT despite the recorded coverage limitation.
- [x] Targeted gap manifest written:
      `experiments/world_model_task_rebinding/cosmos3/targeted_recovery_gap_manifest_20260614_from_late299.json`.
      It groups the remaining `58` misses into five concrete frame/geometry
      ranges.
- [x] Prepared, but did not run, the approval-gated supplement wrapper:
      `scripts/slurm/run_cosmos3_targeted_recovery_supplement_after_approval_in_allocation.sh`.
      It refuses login-node execution and refuses to run unless
      `ALLOW_TARGETED_RECOVERY_SUPPLEMENT=true`. It only generates targeted
      H5 rows plus RGB review sheets, then runs the structural supplement
      checker
      `scripts/world_model/inspect_cosmos3_targeted_recovery_supplement.py`;
      it does not merge, export WAM, or start SFT.
- [x] Added the targeted supplement checker. It verifies H5/RGB count, scenario
      quotas, `301/300` lengths, source kind, regrasp flags for
      `hole_late_fast_shift`, RGB manifest/JSONL/video consistency, and review
      sheet presence. Its `ready_for_merge` remains false unless a later visual
      approval marker exists.
- [x] Added optional hard-teacher support for moving-hole post-motion
      release/regrasp. Default behavior is unchanged. This is needed so
      `hole_late_fast_shift` can produce real `peg_recovery` physical-mode
      rows instead of relying on unrelated peg-drop geometry.
- [x] Targeted hard-teacher supplement launch was attempted inside held Slurm
      allocation `127723`, not on the login node. Four variants all produced
      `0` accepted rows and were stopped with Ctrl-C in tmux:
      initial 112-row attempt `57/0`, motion-focused pilot `22/0`,
      retreat pilot `23/0`, and wait-gate pilot `28/0`. The repeated blocker is
      the scripted hard-teacher geometry, not Cosmos training: target motion
      while the peg is near the old hole triggers swept-wall risk; retreating
      avoids some sweep risk but breaks/then shifts failures to final insert
      geometry. No supplement rows, RGB review sheets, merge, WAM export, or
      SFT were produced from these attempts.
- [x] Inspected the hard-screen-2 failed closed-loop artifacts. They contain
      per-query `iter_##_prefix_f###` folders, prefix videos, summary metrics,
      and `live_history_raw_action_state.json`, but the history JSON contains
      only a `300x32` WAM condition array. It does not contain full live
      simulator `env_state` / `get_state_dict()` snapshots, so old artifacts
      cannot be directly restored into the simulator for recovery-teacher data.
- [x] Prepared the next non-training repair: `run_cosmos3_live_receding_loop.py`
      and the panel/loop Slurm wrappers now support
      `SAVE_LIVE_STATE_SNAPSHOTS=true`. When enabled in a compute allocation,
      each receding iteration writes `live_state_before_controller.h5` and
      `live_state_after_controller.h5` with real simulator state and
      observation snapshots. This does not change controller logic and is not
      method evidence; it only creates restore points for a later
      failed-state recovery supplement.
- [x] Added the separate plan:
      `PLAN/cosmos3_lowfreq_wm_executor/02_failed_state_recovery_supplement.md`.
      The next data-boundary action is to rerun selected failed closed-loop
      samples with state snapshots enabled, then build recovery-teacher rows
      from those real failed states only after approval.
- [x] Direct rerun with `SAVE_LIVE_STATE_SNAPSHOTS=true` stalled before useful
      snapshots because source-restore mode blocked in
      `render_prefix_from_env_states`. A no-render replay path was added
      instead:
      `scripts/world_model/replay_live_history_state_snapshots.py`.
      It replays the recorded live-loop robot actions from old summaries,
      applies the same source target motion, and saves simulator states at the
      old query frames. This is restore-state reconstruction only, not method
      evidence.
- [x] Replayed the hard-screen-2 samples inside Slurm allocation `127723`.
      sample_09 reproduced the old final failed state exactly:
      old/new `peg_head_pos_at_hole =
      [-0.1239014864, -0.0626012236, -0.0517419614]`, L2 `0.0`,
      with `28` query snapshots.
      Remaining replay root:
      `experiments/world_model_task_rebinding/cosmos3/replay_state_snapshots_hard2_remaining_20260614_141435`.
      Samples 10-13 also reproduced old final failures with L2 `0.0`.
      sample_04 reproduced old final `success=True`, so it is not a negative
      recovery seed. Total saved query-state H5 files across 04 and 09-13:
      `163`; usable failed-query snapshots are from 09-13 only.
- [x] Added a narrow failed-state recovery teacher smoke script:
      `scripts/world_model/generate_cosmos3_failed_state_recovery_teacher.py`.
      It restores `frame_***_live_state.h5`, marks perturbation as already
      triggered from step 0 for post-motion coverage semantics, and asks a
      motion-planning teacher to regrasp/replan/insert from the real failed
      state. It does not merge, render, export WAM, or train.
- [x] Failed-state recovery teacher smoke was attempted from real snapshots:
      `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_smoke_20260614_142222`.
      It accepted `0/1` requested demos after `10` attempts over frames
      `296`, `288`, `280`, and `272` from samples 09, 10, and 13. Rejects were
      physical/planner failures: `planner_preinsert_failed`,
      `preinsert_gate_failed`, `final_insert_wall_collision_risk`, and
      `planner_final_insert_refine_failed`. Therefore the current new blocker
      is not Cosmos training or export; it is that the recovery teacher cannot
      reliably re-align and insert from the real failed closed-loop states.
      No accepted H5, RGB review sheet, merge, WAM export, or SFT resulted.
- [x] Follow-up failed-state recovery teacher variants also failed, all inside
      Slurm allocation `127723`: staged preinsert, existing-grasp-first, and
      multi-stage stronger refinement. The final multi-stage smoke
      `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_multi_stage_smoke_20260614_144327`
      tried `24` real failed snapshots from samples 09, 10, and 13 and
      accepted `0/1` requested rows. It executed multiple staged offsets and
      sometimes moved the peg head near the hole, but the whole peg line/axis
      still failed strict insertion geometry or planner preinsert/final-insert
      checks. This confirms the blocker is the teacher/data construction from
      real failed states, not short overfit training and not Cosmos SFT
      mechanics.
- [x] Added and tested an opt-in release/regrasp-after-stage teacher path in
      `scripts/world_model/generate_cosmos3_failed_state_recovery_teacher.py`.
      Physical reason: the live failed states can preserve a bad grasp/peg-axis
      relationship, so moving the old grasp toward the new hole moves the peg
      head but does not reliably align the whole peg. The new path stages the
      grasped peg, opens the gripper, regraspes, and then inserts. It is off by
      default and only used when `--release-regrasp-after-stage` is passed.
- [x] Release/regrasp smoke results, all inside Slurm allocation `127723`, all
      accepted `0` rows:
      `failed_state_recovery_teacher_release_regrasp_smoke_20260614_1510`
      tried `9` attempts; `failed_state_recovery_teacher_release_regrasp_shallow_smoke_20260614_1516`
      tried `9` attempts with shallower final insertion; after fixing a
      `raw_steps` counter bug,
      `failed_state_recovery_teacher_release_regrasp_shallow_counterfix_smoke_20260614_1525`
      retried the suspected false reject for `3` attempts; and
      `failed_state_recovery_teacher_release_regrasp_sample13_refine_smoke_20260614_1532`
      tried one close sample with stronger final refinement. No accepted H5,
      RGB review sheet, merge, WAM export, or SFT resulted.
- [x] Fixed a recovery-teacher implementation bug: `raw_steps` and
      `final_insert_start_step` are now measured relative to the current
      attempt start, not the global `RecordEpisode` step counter across
      previous rejected attempts. The counterfix smoke confirmed the bug could
      mislabel an attempt as `11801` steps, but after the fix the same path
      still failed for real physical reasons: final wall collision risk,
      planner final-insert failure, or strict final insertion failure.
- [x] Current failed-state recovery conclusion: release/regrasp improved some
      geometry but did not produce data. The closest sample reached
      `peg_head_at_hole ~= [-0.0176, 0.00284, 0.00296]` with no wall risk and
      peg-axis cosine `~0.9955`, but the full peg line still missed the strict
      centerline gate (`~0.0156 m` versus `~0.003 m`) and live success remained
      false. Do not continue this same planner-offset/regrasp/refine family
      without a genuinely new teacher mechanism.
- [x] Added task-axis regrasp options to the failed-state recovery teacher:
      `--release-regrasp-mode {current_tcp,goal_y,world_y,world_x}` and
      `--planned-regrasp-mode {current_tcp,goal_y,world_y,world_x}`. Physical
      reason: for round peg geometry, the grasp helper uses `target_closing` to
      choose the gripper closing axis. The old path used the current TCP axis,
      which can preserve the bad live grasp orientation. The new options are
      default-off and allow explicit task-frame closing directions.
- [x] Task-axis recovery smoke results also produced `0` accepted rows:
      `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_goal_y_regrasp_smoke_20260614_1542`
      accepted `0/1` after `3` attempts using `release-regrasp-mode=goal_y`;
      `experiments/world_model_task_rebinding/cosmos3/failed_state_recovery_teacher_planned_goal_y_regrasp_smoke_20260614_1548`
      accepted `0/1` after `3` attempts using
      `--no-use-existing-grasp-first --planned-regrasp-mode=goal_y` plus
      `release-regrasp-mode=goal_y`. The latter failed early regrasp for two
      sine cases and made the fast-shift sample's staged peg line worse
      (`peg_axis_cos_hole_x ~= 0.376`, large centerline miss), so task-axis
      closing is not a viable patch by itself.
- [x] Method correction after rereading `IDEA.md`: failed-state recovery is not
      the active method direction. It is diagnostic evidence only. The next
      aligned plan is
      `PLAN/cosmos3_lowfreq_wm_executor/03_compositional_rebinding_distribution.md`:
      build a compositional dynamic task distribution and an online
      task-frame rebinding teacher, instead of enumerating failed closed-loop
      rescue cases.

## Do Not Do

- [ ] Do not resume old sampled-role v7_733 SFT as the repair.
- [ ] Do not run more broad panels from iter2700 as method evidence.
- [ ] Do not interpret dense receding as a deployment requirement to call
      Cosmos every 8 actions forever.
- [ ] Do not relax `C_pi` or use generated sidecars as handoff authority.
- [ ] Do not use 128/129-frame chunks, 93-frame diagnostics, cropped metrics,
      or hidden/manual target-motion prefixes.
- [ ] Do not run compute work on the login node. Use the login node only for
      downloads, git clone/commit/push, and read-only file/status inspection.
- [ ] Do not use one-shot `sbatch` jobs for this branch. Use a tmux-held
      interactive compute allocation and preserve/reuse it unless the user
      explicitly asks to release it.
- [ ] Do not present short overfit training as formal method evidence. Full
      training evidence must now use at least `2` GPUs and reserve/run for at
      least `3` hours. Four GPUs are still preferred if already available, but
      two GPUs are valid for the immediate dense full SFT. Short
      overfit/sanity training is the exception: it may use 1-2 GPUs for about
      50-100 steps, has no 3-hour minimum, and is only a debug gate.
- [ ] Do not claim the existing hard-screen-2 failed artifacts are restorable
      simulator states. They currently diagnose the missing distribution but
      lack full live state snapshots.
- [ ] Do not start full SFT from failed-state recovery smoke roots or old
      sampled-role roots. They have no accepted/valid main-method rows. The
      latest user override explicitly allows full SFT from the generated dense
      733 condition root despite the recorded live-query coverage gap.
- [ ] Do not keep rerunning the same failed-state recovery smoke without a new
      teacher design. The repeated blocker is planner alignment/insertion from
      real failed poses, not missing compute time.
- [ ] Do not render, merge, export WAM, or start full SFT from any of the
      release/regrasp failed-state recovery smoke roots. They have no accepted
      rows.
- [ ] Do not start another small smoke by merely changing the same
      planner-offset, release/regrasp, task-axis closing, or final-refine
      knobs. That family has repeatedly produced zero accepted rows from the
      real failed states.
- [ ] Do not turn the method into "error detection plus recovery case mining."
      Dynamic target motion, peg offsets, contact drift, and phase changes must
      be handled as observations under the same task-frame rebinding loop, not
      as separate rescue classes.

## Step 1: Freeze The New Method Definition

- [x] Record the corrected branch boundary:
      `PLAN/cosmos3_lowfreq_wm_executor/03_compositional_rebinding_distribution.md`.
      Failed-state recovery roots are diagnostic only; they are not positive
      data and not the main method.
- [ ] Record this branch in evidence notes before running jobs:
      low-frequency Cosmos task WM plus high-frequency executor.
- [ ] Define runtime variables:
      Cosmos call interval, executor chunk length, replan triggers, and latency
      reporting fields.
- [ ] Define which signals Cosmos predicts:
      future target/hole path, peg-head-in-hole-frame path, TCP/peg relation,
      grasp/contact/insertion risk, and optional coarse action hints.
- [ ] Define executor inputs:
      current live observation/state, DP action prior, predicted task path,
      current peg-hole/TCP/grasp predicates, and recent actions.
- [ ] Define executor outputs:
      short executable robot action chunks with action-space clipping recorded.

## Step 2: Clean Dense Condition Preflight

- [ ] Preferred compute-node runner for the current user request:

      `bash scripts/slurm/run_cosmos3_clean_dense_733_overfit_then_full_in_allocation.sh`

      Run it only inside a tmux-held `srun` step. It performs full clean/dense
      preflight, overfit2 preflight, 100-step overfit, strict overfit eval, and
      then starts full training only if the gates pass. The current full-run
      minimum is `2` GPUs for at least `3` hours.
- [x] Source visual check before launch: opened the first two source review
      sheets for the overfit gate and confirmed normal camera/framing plus
      visible final insertion.
- [ ] After explicit user approval, run only the clean/dense preflight inside
      a compute allocation:

      `ALLOW_CLEAN_DENSE_PREFLIGHT=true bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_preflight_in_allocation.sh`

- [ ] Confirm preflight creates a new clean/dense condition root, not a new
      SFT checkpoint.
- [ ] Confirm every row preserves `301` frames and `300` action steps.
- [ ] Confirm `prefix_role` equals physical mode and sampled/curriculum role is
      stored only as provenance.
- [ ] Confirm dense receding prefixes cover target-motion onset, post-motion
      correction, late held-peg states, and insertion/handoff states.
- [x] Confirm live-query coverage audit passes or records exact undercovered
      query modes. It failed: `63/173` undercovered, mostly late
      `target_post_motion` and `hole_late_fast_shift` `peg_recovery` from six
      failed closed-loop videos. Do not start full SFT from this root.
- [x] Rechecked the same question with `MAX_PREFIX_FRAMES=299`:
      `58/173` still undercovered. Remaining gaps are
      `50` `target_post_motion` queries and `8` `peg_recovery` queries,
      concentrated in `hole_late_sine`, `hole_late_constant`,
      `hole_late_continuous_insert`, and `hole_late_fast_shift`.
      Latest user override makes this late299 dense root the immediate full
      SFT input anyway, with the coverage miss recorded as a limitation.
- [ ] Confirm future object states are target/readout supervision only, not
      controller-facing privileged conditions.

## Step 3: Two-Sample Sanity

- [x] Build a clean/dense two-source condition root from the approved data.
- [x] Run preflight only first; require a matching
      `clean_dense_preflight_summary.json` with `ready_for_overfit=true`.
- [x] After explicit user approval, run two-sample overfit SFT only inside the
      tmux-held compute allocation using the short-overfit exception:
      1-2 GPUs, about 50-100 steps, no 3-hour minimum. This is a sanity gate,
      not method evidence.
      Entry:
      `ALLOW_CLEAN_DENSE_OVERFIT_SFT=true bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_overfit2_fix1recipe_in_allocation.sh`
- [x] Inspect generated videos/contact sheets directly. Opened both iter50
      review sheets; predictions are nonblank, same viewpoint, and visually
      aligned with the references for `target_post_motion` and `peg_recovery`.
- [x] Check action chunks on both samples:
      predicted action rows are finite and shape-aligned (`300x32`) with very
      low prefix RMSE. Future robot-action RMSE is still diagnostic, not
      controller proof. Full SFT is now allowed by the latest user override
      despite the recorded live-query coverage gap.
- [ ] If two-sample sanity fails, debug export/training/action extraction. Do
      not move to full training.

## Step 4: Executor Interface

- [x] Add an executor training dataset builder from clean/dense rows:
      `scripts/world_model/build_cosmos3_executor_training_dataset.py` plus
      compute-node wrapper
      `scripts/slurm/run_cosmos3_executor_dataset_preflight_in_allocation.sh`.
- [x] Include DP prior action chunks from the frozen static DP as input, not as
      the only controller.
      Initial two-row smoke is
      `scripts/world_model/export_cosmos3_executor_dp_prior_chunks.py` with
      wrapper `scripts/slurm/run_cosmos3_executor_dp_prior_smoke_in_allocation.sh`.
- [ ] Include Cosmos-predicted task path or task-frame target as input. Current
      debug preflight uses `gt_state_targets_debug`, which is explicitly not
      formal method evidence.
- [x] Include current peg-hole/TCP/grasp state from causal history.
- [x] Train the executor to output short chunks that reduce real peg-hole error
      while preserving grasp, as a two-sample debug gate:
      `scripts/world_model/train_cosmos3_executor_overfit.py`.
- [x] Keep teacher/source action targets separated from diagnostic sidecar
      state. Robot action dims are `7`; executor overfit target dim is `56`
      because the matched DP prior horizon is `8` steps.
- [x] Add a two-sample executor overfit check before full executor training.

## Step 5: Full Training And Closed-Loop Eval

- [x] Dense full-SFT immediate action is complete and should not be repeated
      as the next step. The low-frequency Cosmos task WM/raw-action baseline
      was trained from the generated dense late299 condition root:

      `CONDITION_ROOT=experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050`

      `CLEAN_DENSE_PREFLIGHT_SUMMARY=experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_preflight_20260614_130050/clean_dense_preflight_summary.json`

      `OUTPUT_ROOT=experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_2gpu_20260614_priority_full_sft_max299`

      `NPROC_PER_NODE=2 DATA_PARALLEL_SHARD_DEGREE=2 MAX_PREFIX_FRAMES=299 ALLOW_CLEAN_DENSE_FULL_SFT=true ALLOW_LIVE_QUERY_COVERAGE_GAP_FOR_FULL_SFT=true bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_full_fix1recipe_in_allocation.sh`

      It completed as the 4-GPU `server35` pathfix run through
      `iter_000001500`, which exceeds the current `2` GPU / `3` hour floor.
      Generated strict eval passed, but closed-loop real execution failed
      `0/4` at `iter900`, `0/4` at `iter1200`, and `0/4` at `iter1500`.
- [x] Monitor full SFT validation loss and checkpoints. Do not interpret the
      run as final method evidence until closed-loop eval and visual review
      are complete.
- [x] Immediately after a usable checkpoint exists, run full-episode
      closed-loop eval with the same `301/300` contract. Record checkpoint,
      sample set, controller mode annotations, number of Cosmos calls, final
      real-state success, pure-DP comparison when available, and video/review
      sheets.
- [x] Extract the actual selected live chunk outcomes from the 2026-06-22
      h96 fullunion panel. Output:
      `experiments/world_model_task_rebinding/cosmos3/live_receding_selected_outcomes_h96_panel4_20260622_0729/live_selected_outcomes_summary.json`.
      Result: `45` executed records, `37` candidate-executor chunks,
      `8` DP-handoff chunks, full length ok `4/4`, final success `0/4`.
      `17` selected candidate chunks worsened absolute y/z error. This
      confirms the active blocker is live consequence calibration plus
      contact/insertion execution, not truncation or missing compute.
- [x] Run a live outcome calibration audit on the selected chunks:
      `experiments/world_model_task_rebinding/cosmos3/live_outcome_calibration_h96_panel4_20260622_0745/live_outcome_calibration_summary.json`.
      Key result: high handoff-confidence chunks were `26`, but `13/26`
      worsened y/z and `21/26` still failed the live gate. Scorer/executor
      after-state abs-yz MAE was about `0.038`, worse than identity/no-motion
      `0.018`. This means the current model over-trusts imagined action
      effects in live contact, so the next labels must be real live-state
      action consequences.
- [x] Add candidate action-bank saving for future candidate-executor live
      panels. New control: `SAVE_CANDIDATE_ACTION_BANK=true`, passed through
      the live-loop and panel wrappers. It writes `candidate_action_bank.npz`
      per iteration so unselected candidates can be replay-labeled from the
      same saved live state.
- [x] Started the small action-bank diagnostic panel on key failed live states
      `sample_01_hole_late_constant` and `sample_03_hole_late_fast_shift`
      under tmux `cosmos3_live_actionbank_panel_145920_20260622_074211`.
      Current output root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_h96_fullunion_scorer_bestgate_formalhardphase_actionbank_samples1_3_20260622_074211_alloc145920`.
      It has saved `candidate_action_bank.npz` sidecars for multiple live
      iterations. This is calibration/debug data, not method success.
- [x] Added live-snapshot action-bank replay labeling script:
      `scripts/world_model/replay_cosmos3_live_action_bank_from_snapshots.py`.
      It restores the saved real live snapshot, replays selected bank
      candidates from the same state, advances only the external target from
      the source H5, and records real y/z, grasp, gate, contact, and success
      consequences.
- [x] Smoke-tested the replay labeling script inside allocation `145920`.
      Output:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_smoke_20260622_0805_alloc145920`.
      Result: `3` valid labels (`dp_prior`, `mean`, `scale_0.5`), `0`
      failures, `3/3` worsened y/z, `0/3` after-gate ok, `0/3` success.
      This directly supports the current blocker diagnosis: common selected
      candidates are miscalibrated in real live contact.
- [x] Completed batch live-snapshot action-bank replay labels from the current
      panel in tmux
      `cosmos3_live_actionbank_replay_labels_145920_20260622_0812`.
      Output root:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_current_panel_20260622_0812_alloc145920`.
      Result: `77` valid labels, `0` failures, `50/77` improved y/z,
      `27/77` worsened y/z, `11/77` after-gate ok, `0/77` success. The live
      scorer's selected candidates were `7/11` y/z-improving, `4/11`
      y/z-worsening, only `2/11` after-gate ok, and `0/11` success. Use these
      labels to audit/retrain the live consequence scorer, not as closed-loop
      success evidence.
- [x] Completed an all-candidate first-bank replay check in tmux
      `cosmos3_live_actionbank_replay_allcand_iter0_145920_20260622_0820`.
      Output:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_allcand_iter0_20260622_0820_alloc145920`.
      Result: `426` labels, `0` failures, `0` after-gate ok, `0` success.
      sample 01 had `159/213` y/z-improving candidates but none reached the
      gate; sample 03 had `0/213` y/z-improving candidates and every candidate
      worsened y/z at the first live state. This means the early-state blocker
      is not only scorer selection. The current candidate action family often
      has no live-continuable action consequence to select.
- [x] Directly reviewed the sample 03 contact sheet from the action-bank panel:
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_h96_fullunion_scorer_bestgate_formalhardphase_actionbank_samples1_3_20260622_074211_alloc145920/sample_3_single_panel/live_receding_panel_contact_sheet.png`.
      Visual readout agrees with metrics: the rollout reached DP handoff mode,
      but the peg was not inserted at the final frame.
- [x] Wrote concise evidence note:
      `docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_live_action_bank_replay_blocker.md`.
- [ ] 2026-06-22 08:26+08 launched the next live-label expansion from the
      saved action banks, inside allocation `145920`, output root
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_allbanks_allcand_20260622_082600_alloc145920`.
      It is intended to replay all candidates from all saved banks in the
      sample 01/03 action-bank panel using
      `--all-candidates --no-save-step-records`, about `15 * 213 = 3195`
      labels. This is real live consequence labeling, not closed-loop method
      success evidence. A previous launch at
      `live_snapshot_action_bank_replay_allbanks_20260622_082358_alloc145920`
      produced only `15` selected-candidate labels because shell quoting broke
      the regex; treat that previous root as a launch diagnostic only, not the
      all-candidate sweep.
- [ ] 2026-06-22 08:39+08 replaced the single-process all-bank replay with a
      four-way candidate-sharded replay because the first all-candidate launch
      had no intermediate output and was too slow for the held allocation.
      Running roots:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_allbanks_allcand_sharded_20260622_083941_alloc145920/shard_0`
      through `shard_3`. Each shard uses the same protocol with
      `--all-candidates`, `--candidate-shard-count 4`, and
      `--no-save-step-records`. The interrupted single-process root should be
      treated as superseded unless it somehow writes a complete summary.
- [x] 2026-06-22 later update: the broader all-bank all-candidate replay was
      completed on the newer samples 0, 2, 4, and 5 action-bank panel, using
      candidate sharding inside allocation `145920`.
      Roots:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_samples0_2_allbanks_allcand_sharded_20260622_0952_alloc145920`
      and
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_samples4_5_allbanks_allcand_sharded_20260622_110056_alloc145920`.
      Combined result: `8733` valid labels, `0` direct candidate success,
      `212` after-gate-ok candidates, `5468` y/z-improving candidates, and
      `3265` y/z-worsening candidates. The live scorer selected only `1`
      gate-passing candidate out of `41` selected records, and that selected
      gate candidate still did not finish.
- [x] 2026-06-22 active causal check: replay gate-rich candidates followed by
      DP h96 handoff. Current running root:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_gate_iters_dp96_hole_late_sine_iter4_5_gateonly_20260622_1145_alloc145920`.
      It targets only the `160` `hole_late_sine` iter4/iter5 candidates that
      already passed the no-DP geometry handoff gate. Filter file:
      `experiments/world_model_task_rebinding/cosmos3/sine_iter45_gate_candidate_indices_20260622.tsv`.
      Shards are bound inside allocation `145920`: shard0 sees GPU0 and shard1
      sees GPU1. If `candidate + DP h96` succeeds here, the immediate blocker
      is scorer/handoff selection. If it still fails, the blocker is
      contact/insertion execution even after the geometry gate.
      Result: `160/160` after-gate candidates had fair DP rollout horizon,
      with `0` DP success, `0` DP-continuable, and `0` final contact-stable.
- [x] 2026-06-22 follow-up gate-only DP96 on the other gate sources:
      root
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_gate_iters_dp96_continuous_iter8_movestop_iter10_gateonly_20260622_1152_alloc145920`.
      Correct target-only stats, after filtering out unintended move_stop
      iter8 rows from an iteration-only filter: `hole_late_continuous_insert`
      iter8 has `51/51` after-gate candidates with `0` DP success and `0`
      DP-continuable. `hole_late_move_stop` iter10 has `1` selected gate
      candidate, but the candidate chunk ended at step `300`, leaving no DP
      rollout horizon; final success remained false.
- [x] 2026-06-22 blocker conclusion: combined fair DP-rollout target cases are
      `211` after-gate candidates with real DP horizon, `0` success, `0`
      continuable, and `0` contact-stable final states. These gate-positive
      states are negative DP-continuability labels, not positives.
- [ ] Next repair target: replace the current y/z-heavy geometry handoff proxy
      as the learning target. Mine or construct real positive
      DP-continuable/contact-insertion states from successful expert/static-DP
      suffixes or approved teacher data, and train the live outcome scorer and
      executor against DP-continuability/contact labels rather than geometry
      alone.
- [x] 2026-06-22 read-only positive-source contact-manifold audit completed on
      all `733` source H5 files with `HDF5_USE_FILE_LOCKING=FALSE`. Output:
      `experiments/world_model_task_rebinding/cosmos3/dp_success_contact_manifold_audit_20260622/dp_success_contact_manifold_summary.json`.
      Successful-source medians show the physical gap: within `8` frames of
      first insertion, relative `x` is about `-0.059`, while current false gate
      candidates are around `-0.107` to `-0.115`. The next candidate/executor
      target must push into the insertion-axis/contact manifold before DP
      handoff, not just reduce y/z error.
- [x] 2026-06-22 implementation update for the active causal check: added
      `--candidate-filter-tsv` to
      `scripts/world_model/replay_cosmos3_live_action_bank_from_snapshots.py`.
      It first supported `iteration candidate_index`; after the combined
      follow-up exposed cross-scenario leakage, it was hardened to also support
      `scenario iteration candidate_index`. `py_compile` passed inside
      allocation `145920`.
- [x] 2026-06-22 visual review completion for sample 5 from the latest
      action-bank panel: generated and opened
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_h96_fullunion_scorer_bestgate_actionbank_samples0_2_4_5_20260622_090254_alloc145920/sample_05_hole_late_continuous_insert/live_observed_rollout_annotated_contact_sheet.png`.
      Final frame shows the peg remains outside the hole, agreeing with the
      metric failure.
- [x] Added diagnostic live-consequence scorer entry point
      `scripts/world_model/train_cosmos3_live_action_consequence_scorer.py`.
      It trains from live action-bank replay labels and candidate action chunks
      to predict executed after-state, y/z delta, after-gate, grasp/contact,
      and success. This is a calibration/debug tool, not method success
      evidence. Run it for a short 50-100 step sanity after the all-bank
      replay summary exists.
- [ ] Build the next scorer/executor calibration step from live-real chunk
      outcomes. The target is not error-case enumeration. The target is a
      learned live consequence model that predicts whether a short chunk will
      actually improve y/z, preserve grasp/contact, become DP-continuable, and
      lead to insertion after handoff.
- [ ] Train the executor or DP-prior residual policy after its two-sample sanity
      passes with causal Cosmos-predicted task paths, not GT debug task paths.
- [ ] Record all config: prefix source, dense stride, late-rebind weighting,
      action-loss recipe, executor chunk length, and DP-prior source.
- [ ] Keep all export/render/rollout/evaluation/training compute on Slurm
      compute nodes through the tmux-held allocation, not on the login node
      and not through one-shot `sbatch`.
- [ ] After the immediate dense full-SFT plus closed-loop eval result, use the
      recorded coverage miss to decide whether targeted supplementation or a
      new compositional teacher distribution is needed. Do not silently relabel
      the current 733-only limitation as final method success.
- [ ] If the user approves targeted supplementation, run only:
      `ALLOW_TARGETED_RECOVERY_SUPPLEMENT=true bash scripts/slurm/run_cosmos3_targeted_recovery_supplement_after_approval_in_allocation.sh`
      inside the held tmux allocation. Stop after rendered review sheets and
      inspect them directly before any merge/export/SFT. The wrapper will write
      `targeted_recovery_supplement_inspection.json/.md`; `structural_ok=true`
      is only permission to visually inspect, not permission to train.
- [ ] Do not rerun the same targeted hard-teacher supplement variants above.
      The next data-boundary action needs an explicit choice: either build a
      different teacher that moves the target while the peg is safely outside
      the swept wall, relax only the target-motion sweep model with visual
      rejection, or abandon hard-teacher supplement for another coverage source.
- [ ] Preferred next repair after approval: rerun selected failed closed-loop
      samples only to record real state snapshots, not to train:

      `SAVE_LIVE_STATE_SNAPSHOTS=true SAMPLE_INDICES=9,10,11,12,13 MAX_SAMPLES=5 bash scripts/slurm/run_cosmos3_live_receding_panel_in_allocation.sh`

      Run it only inside a tmux-held compute allocation. The output is restore
      states for the failed-state recovery supplement plan, not method success
      evidence and not permission to merge/export/SFT.

## Step 6: Offline Gates

- [ ] Strict generated artifact check:
      full `301/300`, no hidden truncation, no future leakage.
- [ ] Task-state readout:
      target/hole path, peg-head-in-hole-frame path, grasp/contact/insertion
      predicates.
- [ ] Action/executor audit:
      compare predicted chunks to source/teacher chunks on late-rebind rows.
- [ ] Live-query coverage re-audit:
      failed iter2700 query states should now have local physical-mode
      neighbors in the clean/dense condition distribution.
- [ ] Runtime proxy:
      estimate how often Cosmos would be called under the low-frequency
      schedule versus every-8-step raw-action mode.

## Step 7: Closed-Loop Evaluation

- [ ] Start with a small val panel only after offline gates pass.
- [ ] Use the new runtime contract:
      low-frequency Cosmos update, high-frequency executor chunks, real-state
      `C_pi`, and same detector for moving/no-motion cases.
- [ ] Record for every sample:
      number of Cosmos calls, frames between calls, executor chunk count,
      DP handoff chunks, final success, and video contract.
- [ ] Compare against same-source full pure DP.
- [ ] Do not proceed to hard screens if val is degraded versus pure DP.
- [ ] Hard screen is valid only after val is not worse than pure DP and at
      least one moving-target class shows clear benefit.

## Step 8: Method Success Criteria

- [ ] Full `300` actions / `301` frames.
- [ ] Causal target-motion detection.
- [ ] No future privileged object state as controller input.
- [ ] Cosmos active on moving-target cases only when detector/replan triggers.
- [ ] Executor preserves grasp and improves peg-hole alignment.
- [ ] DP handoff occurs only from real-state `C_pi` pass.
- [ ] Same-source pure-DP comparison is not worse on val.
- [ ] Hard pure-DP failures are rescued at a useful fraction, not only `1/6`.
- [ ] Direct video/contact-sheet inspection agrees with metrics.
- [ ] Runtime report shows the number of Cosmos calls is plausible for
      deployment or clearly marks the result as a simulator-only diagnostic.

## Fallback Decision

- [ ] If clean/dense direct Cosmos raw actions still fail after a fair full
      run, stop treating raw Cosmos actions as the controller.
- [ ] Switch to executor-first control:
      Cosmos predicts task state/coarse plan at low frequency, and the learned
      executor or DP-prior residual policy owns high-frequency robot actions.
- [ ] If the executor also fails to preserve grasp/insertion on inspected
      videos, stop for user direction with concrete failure artifacts.
