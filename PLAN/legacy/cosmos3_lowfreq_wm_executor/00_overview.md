# Low-Frequency Cosmos WM + High-Frequency Executor Plan

## Boundary

This is a new method branch. It does not replace the recorded
`cosmos3_300f_world_model` evidence, and it must not be mixed with the old
direct raw-Cosmos-action checkpoint claims.

The current iter2700 evidence says the implementation contract is now mostly
repaired: full 300 actions / 301 frames, causal target-motion detection,
explicit controller annotation, and same-source pure-DP comparison exist. The
remaining failure is method effectiveness: direct raw Cosmos action chunks are
not reliable enough to put the peg into a DP-continuable insertion state.

This branch keeps the original objective: finish the changed manipulation task.
It does not restore old layouts, use future object poses as controller
conditions, relax final-state success, or report generated sidecars as the
authority for handoff.

## Plain Summary

The old active closed loop asks Cosmos to directly output robot actions every
short chunk. That is too brittle and likely too slow for real deployment.

The new proposal separates time scales:

- Cosmos is a low-frequency world/task model. It predicts where the task frame
  is going and what relative peg-hole state should be reached.
- A smaller executor runs high-frequency robot control. It follows the
  world-model task plan while staying close to the frozen DP skill prior.
- Real observations remain the authority. If the target moves, uncertainty
  grows, contact fails, or DP handoff is unsafe, the system replans.

Dense receding is a training-data repair, not a runtime requirement to call the
large world model every 8 simulator steps forever.

## Execution Resource Boundary

This branch follows the 2026-06-14 experiment rules:

- login node only for downloads, `git clone`, `git commit`, `git push`, and
  read-only file/status inspection;
- all data export, preflight, rendering, rollout, evaluation, and training
  compute runs on compute nodes;
- compute is acquired through a tmux-held interactive allocation, not one-shot
  `sbatch`;
- latest 2026-06-16 user override: formal full-training evidence may use
  `1`, `2`, or `4` GPUs depending on which tmux-held allocation starts first,
  but it must still reserve/run for at least `3` hours; do not keep the chain
  blocked only because a `2` GPU allocation is pending;
- prelaunch audits, chain summarizers, post-gate parsers, and other
  project-code checks run inside the tmux-held allocation through `srun`, not
  on the login node;
- short overfit/sanity training may use 1-2 GPUs for about 50-100 steps with
  no 3-hour minimum, but it is only a debug gate and not method evidence.

## Current Execution Status

2026-06-19 12:45+08 update: the larger reduced128 h32/min16 chain finished
both formal scorer training and live evaluation. The scorer met the required
1-GPU/3-hour floor on allocation `140738` (`elapsed_seconds=10800.01`,
`steps=222852`) and selected `checkpoint_best_gate.pt` from step `1625`.
Conservative margin eval chose margin `0.03`, with selected-minus-DP weighted
error `-0.0163`, progress delta `+0.0136`, and non-DP selection fraction
`0.5`.

That checkpoint was then tested in a real live receding panel with
`ACTION_EXEC_HORIZON=24` and `DP_HANDOFF_CHUNK_HORIZON=32`:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260619_1127_dpcont32_reduced128_margin003_h24_dphandoff32_server54_alloc140738_samples0_1_3_4`

The panel completed all four samples with `0` process failures, passed the
full `301` observed-frame / `300` action-step contract, and wrote/opened the
contact sheet. Final live success remained only `1/4`: sample 0 succeeded;
samples 1, 3, and 4 failed with roughly `9-12cm` remaining insertion-axis
error. This is partial live evidence only (`method_evidence_allowed=false`).

Plain conclusion: the offline scorer is now better calibrated than the 64-row
sanity and the handoff chunk length bug is fixed, but the method still fails
because the executor/selector does not reliably drive live states into a
physically insertable DP-continuable manifold. The next fix should be general,
not case recovery: add shorter/hold-reobserve near-contact candidates and
train/evaluate against live-like handoff survival failures so bad candidate
end states become hard negatives.

2026-06-19 12:55+08 update: the first part of that general candidate repair is
implemented and running as a diagnostic. The live candidate executor now has a
default-off `--candidate-executor-short-prefix-steps` option. A short-prefix
candidate is still scored over the existing 24-step checkpoint feature width:
candidate residual for the first `N` steps, DP-prior residual for the suffix.
If selected, the controller executes only those `N` steps and then reobserves.
This preserves the DDP-style contract: short action chunk, real re-observation
as authority, no enumerated failure cases, and no future privileged state.

The diagnostic panel uses `CANDIDATE_EXECUTOR_SHORT_PREFIX_STEPS=8,12,16`,
the same reduced128 h32/min16 formal scorer, `ACTION_EXEC_HORIZON=24`, and
`DP_HANDOFF_CHUNK_HORIZON=32` on allocation `140738`, session
`cosmos3_live_shortprefix81216_reduced128_handoff32_140738_20260619`.
This run is not formal method success unless the final real-state metrics and
contact sheet support it; it is meant to test whether the current 24-step
overcommitment is part of the live x-error failure.

2026-06-19 14:30+08 result: the short-prefix diagnostic finished and did not
improve live success. The panel had `0` process failures, preserved the full
`301/300` contract, and the contact sheet was opened. Final success stayed
`1/4`: sample 0 succeeded, while samples 1, 3, and 4 failed with final states
about `9-13cm` off in x and/or several centimeters off in z. This falsifies
the simple hypothesis that the main problem is only overcommitting to a
24-step chunk before reobserving.

The important narrowed conclusion is: more frequent reobservation is necessary
for the method, but it is not sufficient. The learned candidate distribution
and outcome scorer still choose or score chunks that do not enter the
physical insertion manifold. The next repair should move the short-prefix
idea into training, not just runtime selection: export real outcomes for
short-prefix/hold-reobserve candidates, label whether DP handoff from those
end states survives or finishes, and train the scorer to reject the live
failure states as hard negatives.

2026-06-19 05:05+08 update: the 64-row h32/min16 scorer sanity completed and
showed that the new label definition is runnable but not enough to support
live evaluation. The reduced64 label shard had useful headroom
(`6/64` oracle successes versus `3/64` DP successes, mean
oracle-minus-DP weighted error `-0.0323`), but a 100-step scorer trained on
those labels did not generalize. It improved the train split
(`selected-minus-DP=-0.0355`, progress delta `+0.0623`) while hurting the
held-out split (`selected-minus-DP=+0.0084`, progress delta `-0.0235`), and
no conservative eval margin passed. This is a data/coverage problem for the
selector, not a reason to change the task or move to live eval.

The active follow-up is now a larger reduced-candidate 128-row h32/min16 chain
on allocation `140738`, session
`cosmos3_dpcont32_reduced128_formal_140738_20260619`. It uses the same
general DP-rollout continuability label (`horizon=32,min_stable=16`) and, if
the headroom gate passes, runs the scorer for the required 1-GPU/3-hour
formal floor. Plain reason: the controller should only use a learned handoff
selector after it can improve over DP on held-out candidate states; the 64-row
sanity could not prove that.

2026-06-19 08:23+08 update: that 128-row export has now completed and passed
the headroom gate. It wrote `4736` outcome labels with `0` failed rows, `438`
DP-rollout-continuable candidates, and `401` DP-rollout successes. The gate
observed `7/128` DP-success groups, `13/128` oracle-success groups, mean
oracle-minus-DP weighted error `-0.0373`, `74/128` meaningful improvements,
and `44/128` large improvements. The formal scorer is now active on the same
allocation. A `checkpoint_best_gate.pt` exists, but it is not live-eligible
until the 1-GPU/3-hour floor finishes and the final training summary marks
`ready_for_formal_live_eval=true`.

2026-06-19 03:08+08 update: a short DP-continuability label diagnostic changed
the interpretation of the current live failure. On allocation `140738` /
`server54`, the same 12 hard-phase groups and 444 candidate records were
exported with the old `8`-step DP handoff label and a longer `32`-step label.
The 8-step label produced `11/444` DP-continuable candidates and `10/444`
DP-rollout successes. The 32-step label produced `31/444` continuable
candidates and `21/444` successes. In a shared-candidate comparison, `21`
candidates were negative under the 8-step label but positive under the
32-step label; examples show DP often needed more than eight steps to close
the insertion-axis gap.

This means the old label is not simply too permissive. It is too short to
describe longer DP handoff survival, while the live panel still fails because
the controller does not reliably reach or recognize these longer-horizon
DP-finishable states in the real closed loop. The next experiment should
therefore generate a reduced-candidate 64-row `horizon=32,min_stable=16`
label shard and use it to debug/retrain the scorer. This is still a diagnostic
step, not method evidence.

A matching runtime issue was also fixed: live `dp_handoff_horizon=32` was
previously capped by `action_exec_horizon`, so the current panel actually ran
only `24` DP handoff steps per handoff iteration. The live loop, panel entry
point, and Slurm wrapper now expose `dp_handoff_chunk_horizon`; default `0`
preserves old behavior, while `32` lets DP handoff match the 32-step label
without lengthening executor chunks. Compute-node syntax/help checks passed on
allocation `140738`.

2026-06-19 02:25+08 update: the state-axis calibrated live panel has now
finished, and it did not improve the method. A strong state-error penalty
(`1.0,0.25,0.25`) was first rejected offline because it made the selector too
conservative. A lighter eval-only sweep selected `0.1,0,0` as the least
damaging x-axis penalty, then the real panel ran on allocation `140738` /
`server54` with the same formal executor, five-shard DP-rollout scorer,
margin `0.02`, and 24-step chunks. Output root:
`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_renderretry_20260619_0102_statex010_dpcont_union5_margin002_h24_server54_alloc140738_samples0_1_3_4`.

The panel finished all four samples with no process failures, passed the full
`301/300` contract, and the contact sheet was opened. Final success stayed
`1/4`: sample 3 succeeded after `59` DP handoff steps; samples 0, 1, and 4
still failed with roughly `9-10cm` insertion-axis error. This means the
state-axis penalty is only a diagnostic. It did not fix the real physical
problem: the controller still confuses "near the hole/block" with "inside a
DP-continuable insertion manifold". The next step should stop adding scalar
penalties and instead train/evaluate live-like handoff survival plus better
near-contact candidates: short DP handoff from the candidate state must stay
inside the insertion manifold or finish, and uncertain states need shorter
chunk or hold/reobserve options instead of a forced 24-step commit.

2026-06-19 00:20+08 update: the five-shard DP-rollout-continuability scorer
has now been tested in the real live panel, and it did not solve the main
blocker. The formal five-shard scorer trained for the required 1-GPU/3-hour
floor on allocation `140717`, selected `checkpoint_best_gate.pt` from step
`150`, and passed conservative margin eval at margin `0.02`
(`selected-minus-DP=-0.010274`, progress delta `+0.009632`). It was then used
with the formal 24-step executor on allocation `140738` / `server54`. The live
panel finished all four samples with no process failures and passed the
`301/300` length contract, but final success stayed `1/4`. Sample 3
`hole_late_fast_shift` again succeeded after receding chunks plus DP handoff;
samples 0, 1, and 4 failed. The contact sheet was opened and matches the
metrics. The new conclusion is narrow and important: more DP-rollout-labeled
offline scorer data is not enough by itself. The remaining method blocker is
false-positive continuability near contact, especially states that look close
in y/z but still have large insertion-axis error or unstable contact.

The next method step should stay general, not become case-by-case recovery.
Train/calibrate the scorer and selector against live-like handoff survival:
states should count as continuable only if short DP handoff from that state
keeps the peg in the insertion manifold or finishes under the same real gate
used at runtime. The controller also needs an adaptive short-chunk or
hold/re-observe action when full 24-step candidates are all risky. This keeps
the Dream Diffusion Policy style contract: short imagined/executed chunks,
real re-observation as authority, and a DP/expert prior for execution, without
enumerating failure cases.

2026-06-18 16:25+08 update: the DDP-style branch has now produced its first
partial live success, but not a final method result. The formal
retrieval-union outcome scorer trained for the required floor on allocation
`139754`, passed conservative margin eval with margin `0.03`, and was used in
a real live panel on allocation `140738` / `server54` together with the formal
24-step candidate executor. The panel completed all four samples with no
process failures and the `301` frame / `300` action contract held. Final live
success was `1/4`: sample 3 `hole_late_fast_shift` succeeded after executor
chunks moved the peg into the DP-continuable region and DP handoff completed
insertion. The inspected contact sheet agrees with the metric for sample 3.
Samples 0, 1, and 4 failed. The important new conclusion is that the method
line is viable enough to solve one dynamic closed-loop sample, but the current
selector/scorer is not robust near contact. It can improve coarse alignment,
but z/contact stability and short-horizon DP continuability are unreliable.
The next fix is not a sample-specific rescue branch: add a general
contact-stability repair by filtering low-confidence non-DP candidates, adding
explicit z/contact-stable scorer targets, and training continuability from
short DP-rollout survival/finish evidence rather than one instantaneous
threshold pass.

2026-06-18 17:55+08 update: a follow-up simple live filter is now a negative
diagnostic, not the next method. The same four-sample panel was rerun with
non-DP candidates rejected unless the outcome scorer predicted nonnegative
progress and at least `0.05` DP-continuable probability. The run completed
without process failures, passed the full `301/300` contract, and produced a
contact sheet, but final success dropped to `0/4`. Most importantly, the same
sample 3 that succeeded in the first panel now failed. This means the blocker
is not solved by one scalar confidence threshold. The filter is allowed to
remain as an optional diagnostic switch with defaults off, but the formal
repair should move to better labels and candidates: explicit z/contact-stable
outcome targets, short DP-rollout continuability labels, and a safe/adaptive
fallback that can execute a shorter chunk or hold/re-observe when both DP and
non-DP candidates look harmful.

2026-06-18 18:10+08 update: the first concrete repair is now implemented and
running. The candidate outcome exporter can generate label-only short
frozen-DP rollouts after candidate chunks and record whether the state is
actually DP-continuable over that short rollout. The scorer trainer now uses
that DP-rollout continuability label when present. A one-row compute smoke on
allocation `140738` passed and wrote `10` labeled candidate outcomes. A new
`64`-row hard-phase retrieval shard with these labels is running on
allocation `140738` / `server54` under
`candidate_outcome_labels_hardphase_retrieval64_skip0_k8_20260618_1815_dpcont_skip0_alloc140738`.
This is the correct next data path for a new formal scorer. Do not reuse the
old scorer or the failed scalar live filter as the next formal controller.

2026-06-18 19:45+08 update: the DP-rollout-label repair has produced the
first useful new scorer data. Shards skip `0` and skip `64` completed with
`5440` successful records each and no row failures. The two-shard union covers
`128` hard groups and passes both the terminal-success and progress-headroom
gates: DP success `7`, oracle success `14`, mean oracle-minus-DP weighted
error `-0.0743`, `95/128` meaningful improvements, `76/128` large
improvements, and retrieval-family oracle candidates in `28` groups. A
formal 1-GPU/3-hour scorer run is now active on allocation `140738` under
`candidate_outcome_scorer_hardphase_retrieval_claim_union_rank1_canddesc_smoke500_20260618_1940_dpcont_union2_formal_alloc140738`.
This is still offline scorer training, not live controller evidence. Shards
skip `128`, `192`, and `256` continue running and should be unioned later for
a fuller all-hard-row scorer before the next live panel if time/resources
allow.

2026-06-18 20:00+08 update: all five DP-rollout-labeled retrieval shards are
now complete and the full union scorer is running. The complete union covers
`312` hard groups and passes both scorer gates: DP success `23`, oracle
success `35`, mean oracle-minus-DP weighted error `-0.0649`, `219/312`
meaningful improvements, `171/312` large improvements, and retrieval-family
oracle candidates in `55` groups. The preferred next scorer is the five-shard
formal run on allocation `140717` under
`candidate_outcome_scorer_hardphase_retrieval_claim_union_rank1_canddesc_smoke500_20260618_2000_dpcont_union5_formal_alloc140717`.
It must complete the `10800` second floor and pass margin eval before live use.
The two-shard scorer on `140738` is still running in parallel as early
evidence, but the five-shard scorer has the complete hard-row DP-rollout label
set.

2026-06-16 14:20+08 update: the candidate-executor branch has now cleared
formal training and is blocked at live rendering. Allocation `131662` ran the
4-GPU formal retry on `server46` for the required floor, wrote a summary, and
selected `checkpoint_best_gate.pt` from the same formal run for live eval. The
selected gate point satisfies the unchanged offline thresholds
(`teacher_progress_mse=0.04146`, `teacher_value_mse=0.03786`, inserted acc
`0.961`, DP-continuable acc `0.857`, selected action MSE below the DP prior).
The gated live panel did start, but it failed before any closed-loop sample
summary or video: `server46` lost the Vulkan device on the first render, and
an explicit render canary with `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`
hung until timeout. `server13` then failed the same canary during CUDA/Vulkan
initialization. This is a scheduling/render-runtime blocker, not evidence that
the trained executor or closed-loop method failed. The next step is unchanged:
run the same gated live panel only after a tmux-held allocation passes the
render canary. A general retry request `132888` reached `server61`, but its
render canary also timed out. A subsequent `server62` `1` GPU allocation
`132981` reached the actual gated live wrapper, but died at the first
`env.render()` with `ErrorDeviceLost`, confirming the canary was not merely a
false blocker. Current pending alternatives are `133177` (`server62`, `3`
GPUs, to test whether another physical GPU on that node can render), `133178`
(`server24`, `1` GPU), `133179` (`server54`, `1` GPU), and `133180`
(`server10`, `1` GPU). `server10`, `server24`, and `server54` have prior
live-summary evidence and are the preferred nodes. A log comparison also
showed that the old successful `server62` live path did not force
`VK_ICD_FILENAMES`; it emitted SAPIEN's missing-ICD warning but still rendered
and completed. The live launchers now match that default again and only set
the NVIDIA ICD when `SET_NVIDIA_VK_ICD=true` is explicitly requested for a
diagnostic variant. A follow-up general retry `133564` reached `server08`,
but the render canary again died at first-frame rendering with
`vk::Device::waitForFences: ErrorDeviceLost`, so that allocation was released.
The old successful path also did not explicitly pass an empty `DISPLAY`;
the retry wrappers now preserve `DISPLAY` only when it is already non-empty
and otherwise leave it unset. This is a render-runtime repair, not a change to
controller gates or success criteria. A second general retry `133733` reached
`server52` with `display=null`, so the `DISPLAY` repair took effect, but it
still timed out at `render_rgb_array_start`; empty `DISPLAY` was not the
sufficient cause. Fixed-node allocation `133178` later started on `server24`.
Its canary failed the same way. A direct gated `sample_03` run found and fixed
a shell wrapper bug before live eval, then the rerun cleared the formal gate
and reached real live execution but stalled after `live_pretrigger_dp_loaded`
with 100% GPU and no new files, matching the first-render hang. That allocation
was released as render-unusable. Current pending attempts are fixed-node
requests for `server62/server54/server10` plus two general `1` GPU retries
excluding confirmed bad render nodes; all run the same gated live path from
`checkpoint_best_gate.pt` when granted.

2026-06-16 10:10+08 update: the candidate-executor branch has moved past a
pure queue blocker. The `1` GPU allocation `131660` started, passed CUDA and
the diffusion smoke, and ran the formal training floor for `10804.9` seconds.
The smoke checkpoint passed the offline scorer/action gate at 100 steps, but
the formal final checkpoint failed because the progress/contact/value scorer
overfit or became numerically uncalibrated on held-out rows. This is not yet a
closed-loop failure: live eval was correctly not launched. The repair is to
keep the same offline gate thresholds but use standard best-validation
checkpoint selection after the formal 3-hour floor is satisfied. The trainer
now saves `checkpoint_best_gate.pt`, the summary records
`formal_live_eval_checkpoint`, and the live launcher loads only that
summary-selected checkpoint. The next active run is a 4-GPU best-gate formal
retry on allocation `131662`, using the smoke-validated stable model
configuration and the same `10800` second floor. Current blocker is Slurm
priority for `131662`, not missing data, not live eval, and not an already
validated method success.

Latest user override, 2026-06-16: the active candidate-executor formal
training standard is now `1/2/4` GPUs, whichever valid allocation starts
first, with the `3` hour floor preserved. The chain must not stay stuck
waiting for exactly `2` GPUs. Current allocation alternatives are `131660`
(`1` GPU), `131564` (`2` GPUs), and `131662` (`4` GPUs), all tmux-held and
currently pending on Slurm priority with prelaunch audits ready.
As of 03:48+08, the active watcher logs confirm the actual launch contracts:
`1x10800s`, `2x10800s`, and `4x10800s`. The first allocation that becomes
`RUNNING` takes a shared launch mutex and runs the diffusion smoke, formal
candidate-executor training, and gated live chain. If the first usable
training allocation is on `server35`, it may complete the formal training
floor, but live closed-loop eval is deferred to a later live-capable
allocation through the `.formal_ready` marker because previous live rendering
on `server35` was unstable.

Prior user override, 2026-06-14: the highest priority is now to run full
Cosmos3 SFT from the already generated dense 733 condition data, then run
closed-loop eval. At that point, the minimum formal training standard was
2 GPUs for at least 3 hours; this resource floor is superseded for the current
candidate-executor chain by the 2026-06-16 `1/2/4` GPU rule above. This
supersedes the
earlier execution ordering that treated live-query coverage as a full-SFT
startup blocker. The coverage gap remains a recorded limitation and must not be
hidden.

The 2026-06-14 clean/dense 733 preflight produced a structurally valid denser
condition root (`9223` rows, `0` role/mode mismatches), but it failed live-query
coverage with `63/173` undercovered queries. The blocker is not truncation,
camera view, or role-label drift. The blocker is that the accepted 733 source
data mostly show successful trajectories, while the failed closed-loop queries
ask for late recovery from states where the peg is already several centimeters
off the moved hole in y/z.

A 733-only late-prefix diagnostic then exposed `MAX_PREFIX_FRAMES` and reran
the same clean/dense preflight with `MAX_PREFIX_FRAMES=299`. That root had
`9271` rows, strict `301/300` preflight passed, and action/sidecar audit
passed, but live-query coverage still failed with `58/173` undercovered
queries. Only `5` queries were recovered by allowing later prefixes. This
confirms the main gap is physical state coverage, not a simple export cap. It
is also the preferred dense full-SFT input under the latest user override.

The short 2-GPU overfit sanity passed at iter50: validation loss fell from
`3.545426` to `0.469846`, corrected eval covered both overfit roles
(`target_post_motion` and `peg_recovery`), strict generated-artifact inspection
passed, and both review sheets were opened. This proves the clean/dense
export, short SFT, checkpoint restore, generation, and inspection chain can run.

Full training is no longer blocked by the coverage gap for the immediate
execution order. It should be run now as the best current dense-data Cosmos3
baseline on the first valid allocation allowed by the active resource rule,
with the coverage gap recorded as a limitation and then tested by closed-loop
eval. Execution update: a 2-GPU `server24` launch passed CUDA and
entry gates but remained in model/config I/O with no useful GPU memory, so it
was stopped when the 4-GPU request became available. The first `server35`
4-GPU launch then exposed a wrapper path bug: relative JSONL paths were passed
after `cd external/cosmos-framework`. That bug is fixed by canonicalizing the
training paths to absolute paths. The current formal run is Slurm job `127819`
on `server35`, output root
`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299`;
it has passed structural/data gates, loaded absolute train/val JSONL metadata,
completed dataloader prewarm, loaded the base checkpoint, reported validation
iteration `0` loss `3.585211`, reached at least training iteration `11`, and
showed all four H200s at `100%` utilization with about `59GB` memory used each.
It is not
method evidence until checkpoints, generated-video/action/readout eval, and
closed-loop eval are complete.

2026-06-15 update: that immediate dense full-SFT test is now complete enough
to decide the direct raw-action branch. The formal dense run trained through
`iter_000001500`, generated strict eval passed at post-floor checkpoints, and
the extra iter1500 generated inspection passed `72` samples. Closed-loop live
execution still failed: `iter900` was `0/4`, `iter1200` was `0/4`, and
iter1500 was also `0/4` across the same four panel samples after moving live
render off broken `server35` and onto `server24/server62`. Several failures
executed DP handoff (`iter1500` handoff steps: sample00 `30`, sample01 `8`,
sample04 `72`) and still did not insert. Therefore the current blocker is not
training startup, generated-video length, or a handoff gate that never fires.
The blocker is that direct raw Cosmos action chunks do not create a stable
DP-continuable insertion state. Stop tuning this raw-action loop as the main
method; the next aligned step is the low-frequency WM plus learned executor or
DP-prior residual controller described below.

The targeted hard-teacher and failed-state recovery supplement attempts are now
diagnostic only. They produced `0` accepted rows and, more importantly, they
are the wrong main abstraction: mining closed-loop failures and teaching one
rescue per failure would turn the project into error detection plus case
recovery. That conflicts with `IDEA.md`.

The corrected repair path is:

`PLAN/cosmos3_lowfreq_wm_executor/03_compositional_rebinding_distribution.md`

Build a compositional dynamic task distribution and an online task-frame expert
that uses the same rebinding loop for all combinations. Dense training rows
should come from successful dynamic task-frame rebinding trajectories, not from
enumerated failed-state rescue cases. This remains the next method repair
direction after the immediate dense full-SFT plus closed-loop eval priority.

After inspecting DDP/AdaWorldPolicy/DiWA/WAM/HDP, the executor repair direction
is now recorded in:

`PLAN/cosmos3_lowfreq_wm_executor/04_ddp_hdp_borrowed_executor_repair.md`

Plain lesson: DDP-style systems work because the policy is trained to act from
world-model/imagination representations, not because a frozen policy is nudged
by a tiny residual. Contact-rich methods such as HDP also show that the low
level policy needs explicit contact/phase guidance. Therefore the current
small residual executor is a useful diagnostic but should not become the final
method. The next repair is a contact/progress-conditioned executor, preferably
diffusion or candidate-sampling based, with DP prior as regularization rather
than as a hard center.

2026-06-15 execution update: after the user reaffirmed that formal training
may start with `2` GPUs for `3` hours, the branch moved from plan to interface
work instead of rerunning the already-failed raw-action loop. The executor
dataset builder produced `6969` clean-dense debug samples, a frozen-DP prior
smoke exported two proposal chunks, and a two-sample residual executor overfit
ran for `100` steps with final MSE `3.46294e-07`. This clears the immediate
executor interface smoke, but it is not formal evidence because the task path
conditioning is still `gt_state_targets_debug`. Full executor training must
wait until causal Cosmos-predicted task paths/readouts replace those GT debug
paths.

2026-06-15 latest launch update: the `2` GPU floor is now the active execution
rule. A tmux-held `2` GPU allocation was granted as Slurm job `128023` on
`server54`; CUDA canary passed on both H200s. That resource was used for the
causal executor path, not to rerun the failed raw-action controller. The
executor-targeted input builder now uses role/scenario round-robin selection.
The 64-row causal gate passed first: Cosmos prediction from dense checkpoint
`iter_000001500` completed with `strict_eval_artifacts_ok=true`, the
predicted-task-path dataset wrote `64` rows, matched frozen-DP prior export
wrote `64` rows, and the no-GT two-sample executor overfit used
`task_path_source=cosmos_predicted_action_sidecar` with final MSE
`1.74878e-08`.

Later 2026-06-15 update: the broader 512-row causal preparation also completed
on the same `2` GPU `server54` allocation. The 512-row Cosmos prediction pass
has `strict_eval_artifacts_ok=true`, `num_samples=512`, and no strict failures;
the executor dataset wrote `512` causal sidecar rows with role counts
`insert_resume=169`, `target_motion_observed=150`, and
`target_post_motion=193`; matched frozen-DP prior export wrote `512` rows with
no export failures. Formal residual-executor training is now running under
tmux session `cosmos3_executor_formal_2gpu_20260615`, Slurm job `128023`,
output root
`experiments/world_model_task_rebinding/cosmos3/executor_residual_train_20260615_executor_residual_train512_formal_2gpu_server54`.
The trainer entry points are
`scripts/world_model/train_cosmos3_executor_residual.py` and
`scripts/slurm/run_cosmos3_executor_residual_train_in_allocation.sh`; they use
DDP/`torchrun`, reject GT debug task paths, and encode the current `2` GPU /
`3` hour training floor. Early metrics show a real risk rather than success:
the model fits train rows quickly, but the validation action MSE is still worse
than the frozen-DP-prior validation MSE. The next decision point is the saved
post-3-hour training summary, not another raw-action closed-loop retry.

2026-06-15 post-floor update: the formal residual-executor run satisfied the
current `2` GPU / `3` hour floor (`world_size=2`,
`elapsed_seconds=10800.012`). The unscaled executor failed the offline gate:
final validation action MSE was `0.0183239` versus frozen-DP prior
`0.00156083`. A validation-only residual-scale calibration then found a small
positive scale, `0.05`, with validation action MSE `0.00151089`, beating the
DP prior by only `4.99e-05`. This is not strong method evidence, but it is the
first conservative offline justification for a small residual-executor live
panel. The live panel must be interpreted as a test of whether a tiny
Cosmos-task-path-conditioned correction can improve real execution; it must not
be reported as success unless real final-state metrics and inspected video
support it.

Closed-loop plumbing for the executor branch is prepared but gated. The live
loop/panel now support `controller_action_source=residual_executor`: Cosmos is
called from the current observed prefix as before, but the executed robot chunk
comes from the learned residual executor conditioned on the causal
Cosmos-predicted task sidecar, current live robot/object state, and frozen-DP
prior action chunk. This is the intended low-frequency WM plus high-frequency
executor contract. It must only be launched after a post-3-hour formal
executor checkpoint is available and the training summary justifies
closed-loop eval; otherwise the correct next evidence is the training blocker,
not another live video from a controller that already fails offline.

## Why The Current Direct Raw-Action Loop Fails

Current evidence:

- Val comparison: Cosmos closed loop succeeds on `1/3`, same-source pure DP on
  `3/3`.
- Hard-screen-2 comparison: among six full-episode pure-DP failures, Cosmos
  rescues only `1/6`.
- Hard failures are dominated by real-state `C_pi` y/z blocks, occasional
  grasp loss, and action direction/scale mismatch.
- The old condition root has `1193/2899` role/mode mismatches.
- Live-query coverage audit found `74/173` live Cosmos queries without a local
  role/mode-consistent training neighbor.

Interpretation:

The loop now runs, but the raw actions predicted by Cosmos are not reliable.
The model often sees a moving target and partly reacts, but it does not
consistently drive the grasped peg into the moved-hole insertion manifold. When
it is near the gate, DP handoff can still drift out of continuability.

## Proposed Architecture

```text
live RGB/state history
  -> causal target-motion / contact / uncertainty monitor
  -> low-frequency Cosmos task-world update
       predicts: future hole/task frame, desired peg-head path,
                 grasp/contact/insertion risk, optional coarse action hints
  -> high-frequency executor
       inputs: current observation, DP action prior, predicted task path,
               current peg-hole/TCP relation, contact state
       outputs: short executable action chunk
  -> real simulator / robot execution
  -> real-state C_pi handoff check
       if continuable: frozen DP short chunks
       if not continuable: executor continues or Cosmos replans
  -> reobserve and repeat
```

## Runtime Contract

Do not assume Cosmos must be called every 8 actions at deployment time.

Use a low-frequency WM schedule:

- Always call Cosmos when causal target motion is first detected.
- Call Cosmos again when:
  - target motion changes direction or speed;
  - the executor's measured peg-hole error stops improving;
  - contact/grasp state changes;
  - `C_pi` rejects DP handoff for too many consecutive chunks;
  - model uncertainty or readout disagreement is high.
- Otherwise let the lightweight executor run for a longer chunk, such as
  `16`, `24`, or `32` actions, with real observation checks between chunks.

The exact chunk length is an evaluation variable, not a fixed claim. Every run
must record:

- number of Cosmos calls;
- mean and max Cosmos-call interval in frames/actions;
- executor chunk length;
- wall-clock inference latency where available;
- final success and video evidence.

## What Dense Receding Means

Dense receding means the training/export code creates many causal prefix rows
from one full episode, for example:

```text
target motion starts near f094
training prefixes: f094, f102, f110, f118, ... through recovery/insertion/end
```

Each row remains a full `301`-frame / `300`-action episode record with a
different causal prefix mask. It is not a 128-action clip and not a short
video method sample.

Purpose:

- Match the states the real closed loop actually queries.
- Teach late/post-motion correction rather than only a few easy successful
  source-trajectory prefixes.
- Give the executor examples of "the peg is held but still outside the moved
  hole; what short action should reduce the error?"

Dense receding is not a runtime promise to call Cosmos at every dense prefix.
It is coverage for learning and for choosing robust low-frequency replanning
points.

## What Preflight Means

Preflight is a no-training validation pass over a proposed data/export root.

It must answer:

- Are all videos `301` frames and all action/state targets `300` steps?
- Are condition masks causal?
- Is the controller-facing role the actual physical mode, not a sampled
  curriculum label?
- Are future hole/peg/TCP states stored only as targets/diagnostics, not as
  privileged controller conditions?
- Do dense receding rows cover the live failed-query states?
- Are train/val splits and source paths valid?
- Are there enough late-rebind rows where the peg is grasped but not yet
  inserted?

Preflight does not train and does not claim method success. It prevents wasting
GPU time on a broken export.

## Training Plan

### Stage 1: Clean Dense Condition Root

Start from the frozen accepted v7_733 source/RGB/H5 data. Do not regenerate H5
data for this repair.

Export a new condition root with:

- `prefix_role_source=physical_mode`;
- sampled/curriculum role stored separately as provenance;
- dense receding prefixes around target motion, late correction, and insertion;
- late-rebind row weighting or repetition;
- full `301/300` records only;
- live-query coverage audit recorded for limitation/future repair, not as the
  immediate full-SFT blocker under the latest user override.

### Stage 2: Low-Frequency Cosmos Task WM

Immediate run priority:

- train from
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050`;
- use preflight summary
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_preflight_20260614_130050/clean_dense_preflight_summary.json`;
- write full SFT under a new priority output root;
- preserve the current full-training resource rule: use the first valid
  1/2/4-GPU tmux-held allocation, but keep at least the 3-hour formal floor;
- if the old wrapper rejects the root solely because coverage is under target,
  add an explicit user-override gate that keeps structural checks and records
  the `58/173` gap.

Use Cosmos to predict future task state and visual/task evolution from causal
history:

- target/hole path;
- peg-head-in-hole-frame trajectory;
- TCP/peg relation;
- grasp/contact/insertion predicates;
- optional coarse action hints.

Do not treat generated state sidecars as handoff authority. They are model
outputs to guide/scaffold the executor and to diagnose failures. Real state
still decides `C_pi`.

### Stage 3: High-Frequency Executor

Train or distill a smaller execution policy that consumes:

- current live observation/state;
- DP action prior or DP feature/action proposal;
- Cosmos-predicted task path or task-frame target;
- current peg-hole/TCP/grasp predicates;
- recent action history.

It outputs short executable robot action chunks. It should preserve the static
DP's grasp/insertion competence while adding the ability to track a moved task
frame.

Possible executor forms:

- DP-prior residual policy;
- learned short-chunk policy conditioned on task-frame targets;
- MPC-style selector over DP/expert action candidates scored by WM task-state
  predictions.

The executor must not be a hand-coded threshold bridge. Thresholds may guard
safety and handoff, but not define the positive method.

### Stage 4: Conservative Handoff

Use real-state `C_pi` to decide whether frozen DP can resume. The handoff gate
must check:

- peg is grasped;
- target motion is slow or settled;
- peg-head is inside the empirical DP-continuability band;
- y/z alignment is tight enough for insertion;
- recent action did not damage contact.

If handoff fails, the executor continues or the WM replans. Do not relax the
gate to make a result pass.

## Theoretical Basis

### Time-Scale Separation

A large video/world model is useful for slower task-frame reasoning, not
necessarily for 30 Hz robot actuation. A smaller executor should handle local
contact-rich control.

### Receding-Horizon Control

Only a short executable chunk should be committed before reobserving. This
limits compounding errors and keeps the real world as the authority.

### Distribution Matching

The model must be trained on the states it will query at runtime. Dense
receding prefixes and live-query coverage repair the mismatch between sparse
successful source prefixes and real closed-loop recovery states.

### Policy Prior Preservation

The frozen DP already knows grasp-hold and insertion in static scenes. The new
controller should reuse that competence rather than replace it with raw
geometry thresholds or direct video-model actions.

### Conservative Continuability

Generated predictions are not enough for handoff. Handoff is valid only when
the real observed state is inside the DP success manifold.

## Evaluation Plan

Evaluate in this order:

1. Clean/dense preflight passes with no role/mode mismatch and full `301/300`
   length; live-query coverage is recorded, and for the current priority run
   its `58/173` miss is a limitation rather than a launch blocker.
2. Two-sample sanity overfit passes for both generated video/task-state and
   executor action chunks, using the explicit short-overfit exception:
   1-2 GPUs, about 50-100 steps, no 3-hour minimum, and not counted as method
   evidence by itself.
3. Full SFT/executor training produces strict same-length generated artifacts.
4. Offline action/readout metrics improve specifically on late-rebind rows.
5. Closed-loop val panel compares against same-source pure DP.
6. Hard-screen panels evaluate only after val is not degraded.
7. Runtime report records Cosmos-call frequency and latency proxies.
8. Major success requires metrics plus direct video/contact-sheet inspection.

Method success requires:

- full `300/301` rollout;
- causal target-motion detection;
- no future privileged state conditions;
- nonzero WM use on moving-target cases;
- same-source pure-DP comparison;
- hard-case success fraction high enough to show broad usefulness, not one
  lucky rescue;
- final success from real simulator state plus inspected video.

## Stop Conditions

Stop and ask for direction if:

- clean/dense preflight cannot cover failed live-query states without breaking
  causality;
- two-sample executor/WM sanity cannot overfit clean data;
- direct raw Cosmos actions remain unstable after clean/dense repair;
- a learned executor also cannot preserve grasp/insertion on inspected videos;
- runtime requires Cosmos calls so frequent that the method is not deployable.

If direct raw Cosmos actions remain the blocker, switch to the executor-first
method instead of continuing to train the same raw-action head.
