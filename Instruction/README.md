# Instruction

This directory stores project-level operating instructions that should stay
visible across future agent sessions.

Current active instruction:

- `project_operating_rules_current.md`
- `cosmos3_lowfreq_wm_executor_current.md`
- `cosmos3_selector_blockers_plain_zh_current.md`
- `cosmos3_current_blockers/2026-06-23_selector_live_headroom_plan.md`
- `current_watch_items_zh.md`

Latest user-level operating instruction:

- Keep these top-level judgments in this root `Instruction/` directory, not
  only inside experiment notes.
- Keep observing active tmux/Slurm/artifact state and continue aligned work
  when evidence or resources change.
- Routine updates should state the next observation/action. If a real blocker
  appears, state the blocker and the next concrete options.
- Do not close routine user updates with pause/ending language. Report the
  factual state, blocker, evidence status, and next aligned action.
- Do not end the work by repeating a status recap while the next aligned
  observation or compute-node action is available.

Current top-level judgment:

- Latest completed source-suffix panel `0,2,4,5` with offsets
  `64,48,32,24` finished with `2/4` real final successes under the full
  `301/300` contract: sample04 and sample05 succeeded; sample00 and sample02
  failed. This is mixed evidence. The interface can work, but it is not stable
  enough to claim broad method success.
- Latest handoff-label replay shows the actionable blocker more sharply:
  source-suffix-name candidates had `53/76` DP96 successes, while live-selected
  candidates had `15/42` and DP prior had `16/42`. There were `45` cases where
  `C_pi=false` but DP96 succeeded and `2` cases where `C_pi=true` but DP96
  failed. Therefore `C_pi` is a diagnostic, not the training target.
- Current running compute: 1-GPU 3-hour union+panel formal handoff-rank scorer
  at
  `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_source_suffix_union_plus_panel0245_handoff_rank_formal3h_20260622_alloc146658`.
  It trains on `9016` namespaced outcome rows from the old source-suffix union
  plus the new panel0245 handoff labels. Single-panel formal attempts were
  interrupted because they were too small and underused the GPU.
- The old Cosmos3/h96 live-family panel was `0/4` final success with valid
  full-length rollouts. The source-suffix repair has now produced one real
  live closed-loop success on sample 5. This is important conversion evidence,
  not broad method success yet.
- Latest successful artifact:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_bestgate_sample5_dist002_exec8_nodpunsafe_20260622_alloc145920`.
  It completed `301` observed frames, passed the full-episode contract, and
  ended with real simulator `final_success=true`.
- Latest mixed panel artifact:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658`.
  It completed all requested samples with no process failure and
  `final_success_count=2/4`.
- The key change was not a looser threshold. A source-suffix-aware outcome
  scorer trained from live snapshot DP96 labels selected
  `retrieval_resid_srcsuffix_r1_s1_o32` in the first live decision. That
  source-suffix chunk moved the real state into a DP-finishable region, and
  frozen DP completed insertion by frame 300.
- The remaining question is generality and stability. sample 5 proves the
  corrected source-suffix/DP-handoff interface can work once. sample00 now has
  both a single successful run and a same-protocol panel rerun failure, so the
  stable claim is not "sample00 fixed"; it is "offsets `64,48,32,24` improve
  source-suffix candidate coverage, but closed-loop DP handoff/contact
  continuability is still unstable."
- The 2026-06-22 all-candidate replay over the latest saved banks strengthens
  this judgment: `8733` live replay labels, `0` direct candidate success, and
  `212` gate-passing candidate chunks. The live scorer selected only `1` of
  those gate-passing candidates, and that selected gate candidate still did
  not finish insertion.
- The active check is DP-handoff replay after gate-rich candidate chunks. If
  `candidate chunk + DP h96` succeeds, the immediate fix is scorer/handoff
  selection. If it still fails, the deeper blocker is contact/insertion
  execution even after near-hole geometry.
- Latest result: the deeper blocker is confirmed. In fair DP-rollout cases,
  `211/211` geometry-gate candidates were followed by DP96 and produced
  `0` success, `0` continuable states, and `0` contact-stable final states.
  Current "gate positive" states are negative DP-continuability examples.
- Read-only positive-source audit over all `733` H5 files shows the real
  successful insertion manifold is farther along the insertion axis: within
  `8` frames of insertion the median relative `x` is about `-0.059`, while
  current false gate candidates are around `-0.107` to `-0.115`.
- New 2026-06-22 source-suffix replay refines the blocker: successful source
  insertion suffixes, used only as diagnostic candidates from live snapshots,
  produced `18/144` DP96 successes and `19/144` DP-continuable/contact-stable
  outcomes, while direct 32-step candidate success stayed `0/144`. Therefore
  the current live candidate family is missing a useful contact/insertion
  action source; the next repair is to add causal insertion-suffix/retrieval
  candidates and score real DP96 continuability, not to keep tightening the
  old y/z geometry gate.
- Follow-up source-suffix live smoke confirms the wiring but not success:
  source-suffix candidates were present in all three receding iterations, but
  the old scorer selected `scale_0.2`, then `dp_prior`, then `dp_prior`; final
  success was false and the contact sheet shows the peg still outside the hole.
  The immediate fix is a source-suffix-aware outcome scorer trained from real
  DP96 success/continuability labels, not another repeat of the same live
  panel with the old scorer.
- Latest safety correction: the first broad source-suffix panel exposed unsafe
  suffix reuse. The active boundary now requires same-scenario suffixes,
  source-suffix start distance at most `0.02`, 8-step execution before
  re-observation, and no pure `dp_prior` execution when live `C_pi` is false.
- Latest hard blocker: under those corrected boundaries, sample 00 iter0 had
  `213/213` valid candidate replays with `0` success, `0` after-gate states,
  `0` y/z improvements, and `213` worsened y/z. This means the current action
  bank has no usable action for that early live state. The next repair is
  stronger dense/source/live-state executor or action-candidate training, not
  another repeat of the same candidate pool.
- Latest positive control: under the same corrected boundaries, sample 05
  completed successfully. It used six 8-step close source-suffix chunks, then
  crossed the real-state `C_pi` gate at frame `123`; frozen DP handoff then
  finished the episode. Final real simulator state was
  `[0.0306, -0.0030, -0.0030]` in peg-head-at-hole coordinates with
  `success=true`, and the extracted `301`-frame review sheet was inspected.
  This proves the corrected source-suffix/DP-handoff interface can work on
  one dynamic sample, but sample 00 still proves the current action bank does
  not cover all early live states.
- Latest scorer sanity: a short rank-loss overfit on the mixed
  sample00-negative/sample05-DP96-positive labels learned to rank the one
  handoff-positive sample05 state, but the data only contained one
  DP96-success state and DP prior also succeeded there. This proves rank loss
  is necessary for the scorer; it does not yet prove a selector that beats DP
  or fixes sample00.
- Current running formal check: allocation `145920`, step `145920.315`, is
  now complete. It met the 3-hour floor but failed the gate:
  validation selected handoff `0/8` while oracle was `1/8`, no
  `checkpoint_best_gate.pt`, and no live eval is justified from that scorer.
- Latest sample00 offset audit: the strict sample00 run used source suffix
  offsets `32,24` and therefore generated `0` source-suffix candidates at
  iter0. Read-only bank audit shows same-scenario offsets `48,64` would enter
  the same `0.02` distance cap. This is an offset/action-candidate coverage
  issue, not a reason to loosen thresholds.
- Latest sample00 update: one standalone server56 run with offsets
  `64,48,32,24` succeeded, but the same protocol inside panel `0,2,4,5`
  failed on sample00. The panel rerun completed `301` frames with the full
  contract intact, but final simulator `success=false` and final
  peg-head-at-hole `[-0.09576, 0.01617, -0.06529]`; the review sheet was
  opened and shows no final insertion.
- Causal interpretation: offsets `64,48,32,24` make source-suffix candidates
  available and can produce a successful trajectory, but they do not yet make
  sample00 stable. In the panel rerun, a C_pi-positive state at frame `168`
  led into DP96, but DP96 moved the real rollout to
  `[-0.09734, 0.00605, -0.04949]` and broke continuability; late frames had
  no source-suffix candidates. The current blocker is real DP-rollout
  continuability/contact quality, not simply whether a source-suffix candidate
  exists at iter0.
- Important extra finding: one-iteration replay showed a selected
  source-suffix candidate that failed the instantaneous C_pi gate and worsened
  y/z, yet `candidate + DP96` still succeeded. The handoff target must
  therefore be learned from real DP-rollout continuability, not only from the
  current geometry gate.
