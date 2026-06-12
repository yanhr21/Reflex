# 2026-06-10 Cosmos3 Two-Sample Overfit Code Audit

## Boundary

This note records a code/training-logic diagnostic for the active Cosmos3
300-step full-episode WAM path. It is not controller evidence and not a method
success claim.

The data contract remains unchanged: each sample is a full 300-step episode
with 301 RGB frames and 300 action steps. No 128-action / 129-frame chunked
training construction is used.

## Failed Baseline Overfit

- Run root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_overfit2_v1_4gpu_20260610_0908`
- Slurm allocation: job `123385`, 4 H200 GPUs.
- Dataset:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_overfit2_v1_20260610_0906`
- Samples:
  - `hole_move_stop_seed701000_n167_traj_74_traj_74__target_motion_observed_f096`
  - `peg_drop_seed705000_n167_traj_79_traj_79__peg_recovery_f099`
- Result:
  - `iter_000000300` saved.
  - Validation loss: `4.000712`.
  - Recent training loss remained around `3.8-4.5`.

Interpretation: this is negative code/training-logic evidence. A two-sample
4-GPU overfit on this simple scene should not remain at this loss scale if the
WAM action/video training path is correctly configured.

## Code Issues Found

1. The old optimizer allowlist did not explicitly include the Cosmos3 action
   adapter modules `action2llm`, `llm2action`, or `action_modality_embed`.
   This could freeze the action interface while the run expects joint
   world/action modeling.
2. The old training logs only printed total loss, hiding whether the failure
   was video or action dominated.
3. The old config used `normalize_loss_by_active=false`, allowing conditioned
   prefix tokens to dilute future-token supervision.
4. The old action schedule reused the vision schedule. The local robot
   world-model reference (`Genie-Envisioner`) uses memory-conditioned future
   chunks and an explicit action path with independent action noise/loss.
5. The old overfit sanity settings were too conservative for a two-sample code
   diagnostic: `lr=2e-5`, long warmup, cosine toward zero, and `clip=0.1`.

## Patches Made

- `scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh`
  now exposes optimizer keys, LR, scheduler, grad clip, action loss weight,
  active-token normalization, independent action schedule, and `shift_action`.
- `external/cosmos-framework/cosmos_framework/callbacks/iter_speed.py`
  now prints component losses for vision/action/sound.
- `scripts/world_model/inspect_cosmos3_full_episode_wam_eval_artifacts.py`
  now reports robot-action RMSE over the first 7 dimensions separately from
  structured state sidecar RMSE over the remaining dimensions.
- `AGENTS.md` now records that tmux-held Slurm allocations should be preserved;
  stop in-allocation commands with `Ctrl-C` rather than defaulting to
  `scancel`.

## Active Fix1 Run

- Run root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_overfit2_fix1_4gpu_20260610_1045`
- Slurm allocation: job `123385`, step `123385.6`, 4 H200 GPUs.
- Key settings:
  - `NPROC_PER_NODE=4`
  - `DATA_PARALLEL_SHARD_DEGREE=4`
  - `OPTIMIZER_KEYS_TO_SELECT=moe_gen,time_embedder,vae2llm,llm2vae,action2llm,llm2action,action_modality_embed`
  - `OPTIMIZER_LR=1.0e-4`
  - `SCHEDULER_WARMUP_STEPS=10`
  - `SCHEDULER_F_MIN=0.5`
  - `GRAD_CLIP_NORM=1.0`
  - `ACTION_LOSS_WEIGHT=2.0`
  - `NORMALIZE_LOSS_BY_ACTIVE=true`
  - `INDEPENDENT_ACTION_SCHEDULE=true`
  - `SHIFT_ACTION=1`

Startup evidence:

- Optimizer selected `410` tensors / `6,982,401,216` elements, confirming the
  action adapter modules are included.
- Early component losses show the failure is action/structured dominated:
  vision loss is already small (`~0.05-0.14`) while action loss drops from
  about `1.88` to about `1.42` by iteration `6`.

## Next Evidence Required

Fix1 produced strict full-episode generated artifacts at `iter_000000100`:

- Eval root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_overfit2_fix1_4gpu_20260610_1045/eval_full_episode_wam_iter_000000100`
- Strict artifact inspection: passed, no strict failures.
- Samples: `2`.
- Predicted/reference video length: `301/301` for both samples.
- Predicted/reference action shape: `300x32/300x32` for both samples.
- Mean future video PSNR: about `30.94` dB.
- Mean robot-action future RMSE: about `0.32654`.
- Mean state-sidecar future RMSE: about `0.21112`.
- Visual evidence opened by the agent:
  `review_sheets/00_target_motion_observed_hole_move_stop_hole_move_stop_seed701000_n167_traj_74_traj_74__target_motion_observed_f096_ref_pred_sheet.png`
  and
  `review_sheets/01_peg_recovery_peg_drop_peg_drop_seed705000_n167_traj_79_traj_79__peg_recovery_f099_ref_pred_sheet.png`.
- Visual interpretation: ref/pred frames are close through the full 301-frame
  rollout for the moving-target and peg-recovery examples. The user explicitly
  confirmed the overfit video result is acceptable and directed the run to move
  to full-data training.

Decision: treat the two-sample overfit sanity check as passed. Do not spend
more allocation time on later overfit checkpoints unless a future regression
requires it.

The two-sample overfit srun step was stopped with tmux `Ctrl-C`, preserving the
4-H200 allocation. The same Slurm allocation `123385` on `server32` is now used
for full1000 fix1 SFT:

- Tmux: `cosmos3_full_fix1_4gpu_20260610`.
- Slurm step: `123385.7`.
- Output root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_fix1_4gpu_20260610_1131`.
- Condition root:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_full1000_rgb_300step_20260609_203902`.
- Full-data fix1 settings: `NPROC_PER_NODE=4`,
  `DATA_PARALLEL_SHARD_DEGREE=4`, `lr=1e-4`, `warmup=10`,
  `scheduler_f_min=0.5`, `grad_clip_norm=1.0`,
  `optimizer.keys_to_select=moe_gen,time_embedder,vae2llm,llm2vae,action2llm,llm2action,action_modality_embed`,
  `normalize_loss_by_active=true`, `independent_action_schedule=true`,
  `shift_action=1`, and `action_loss_weight=2.0`.

Next required evidence is full-data checkpoint validation generation and visual
review, not additional two-sample overfit.

## Full1000 Fix1 Startup

The full-data run passed startup sanity on the same 4-H200 allocation:

- The run loaded `Cosmos3-Nano-Policy-DROID-DCP` successfully.
- The optimizer selected `410` tensors / `6,982,401,216` elements, including
  the action adapter modules.
- Iteration-0 validation loss was `4.279975`.
- Training iterations `1-4` logged finite losses and grad norms at about
  `17.4` seconds per step after startup.
- The component split remains action dominated while the vision loss is low,
  consistent with the overfit diagnosis and not with a frozen video-only path.

No checkpoint has been produced yet. The next evidence gate is still full-data
checkpoint eval/render under the strict `301` frame / `300x32` action contract,
followed by direct visual inspection of the generated review sheets.

A separate tmux-held 1-H200 allocation was acquired for that checkpoint
eval/render work: Slurm job `123499` (`cosmos3_aux_eval_0610`) on `server32`.
It is not used for SFT and should prevent evaluation from stealing GPUs from
the active 4-H200 training step.

An allocation-only watcher is now running in that aux allocation:

- Tmux: `cosmos3_full_fix1_iter300_eval_watch_20260610`.
- Slurm job/step: `123499.1`.
- Watched checkpoint:
  `sft_full_episode_wam_full1000_rgb_300step_fix1_4gpu_20260610_1131/outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/checkpoints/iter_000000300`.
- Eval root:
  `sft_full_episode_wam_full1000_rgb_300step_fix1_4gpu_20260610_1131/eval_full_episode_wam_iter_000000300`.
- Eval settings: `N_EVAL_SAMPLES=10`, `INFERENCE_NUM_STEPS=50`, strict
  inspection enabled.

Mid-run training sanity:

- Iteration-100 validation loss: `0.947078`, down from iteration-0 validation
  loss `4.279975`.
- Recent rank-0 training losses around iterations `100-113` were roughly
  `0.18-0.34`, with action component mostly `0.08-0.16`.
- The watcher had not yet seen `iter_000000300`, which is expected at this
  point. No checkpoint/eval/video evidence exists from the full1000 fix1 run
  yet.
- The full1000 fix1 SFT passed the minimum 1-hour training floor while still
  running on the 4-H200 allocation. Around iterations `170-185`, rank-0 loss
  was roughly `0.11-0.17`, with action component roughly `0.04-0.07` and
  finite grad norms. This remains training sanity only; generated eval/video
  evidence is still pending.
- Iteration-200 validation loss was `0.756682`, still improving from
  iteration-100 validation loss `0.947078` and iteration-0 validation loss
  `4.279975`. Recent rank-0 losses around iterations `201-213` were roughly
  `0.09-0.19`. Continue training and wait for the iter300 checkpoint/eval gate.
- Checkpoint `iter_000000300` was saved at `2026-06-10 13:17:25 CST`.
  The rank-0 checkpoint-step loss was `0.1193` (`vision=0.0232`,
  `action=0.0481`). The aux watcher detected the checkpoint and is waiting for
  stable files before launching eval/render.
- Iteration-300 validation loss was `0.787787`, slightly worse than the
  iteration-200 validation loss `0.756682`.
- Iteration-300 strict generated artifact eval passed for all 10 samples:
  generated/reference videos `301/301`, actions `300x32/300x32`, mean action
  RMSE `0.3643254212`, mean robot-action future RMSE `0.9349301391`, mean
  state-sidecar future RMSE `0.0999715768`, and mean future-video PSNR
  `21.2818860253`.
- Visual review: all 10 review sheets were opened. Compared with the previous
  failed full-data chain, iter300 is visually much healthier: no white/fog
  collapse, no transparent full-frame drift, coherent robot/peg/target
  geometry, and good ref/pred match on target-motion/static/insert-resume
  panels. Boundary: peg-drop recovery is not proven as reliable regrasp
  behavior, and this is still not controller evidence.
- Generated-RGB task-state readout/profile strict structure passed. Aggregate
  diagnostics: mean final hole error `0.0879152346` m, mean future hole RMSE
  `0.0369255092` m, mean future peg RMSE `0.0495181164` m, mean future TCP
  RMSE `0.0480951126` m, and mean future peg-head-hole RMSE `0.0430254817` m.
  This is a remaining controller-interface limitation.
- Iter600 watchers have been launched in aux allocation `123499`: eval watcher
  step `123499.3` and readout/profile watcher step `123499.4`.
- Iteration-400 validation loss was `0.733358`, improving beyond iteration-200
  `0.756682` and iteration-300 `0.787787`. This means the current full1000
  fix1 run is still improving by validation loss and should continue to later
  checkpoint gates. Validation loss remains diagnostic only; the next method
  evidence gate is iter600 strict generated artifacts, readout/profile, and
  direct visual review.
- Iteration-500 validation loss was `0.746084`, slightly worse than iteration
  400 but still near the best point so far. The run resumed at iteration 501
  with low finite loss, so this is not a reason to stop before the iteration
  600 full-episode generated artifact/readout/video gate.
- Checkpoint `iter_000000600` was saved at `2026-06-10 14:56:48 CST`.
  Iteration-600 validation loss was `0.718841`, the best validation point so
  far, and training resumed at iteration 601 with low finite loss. The aux
  eval watcher in allocation `123499` detected the checkpoint and passed
  strict eval input construction for 10 samples with `301` expected RGB frames,
  `300` expected action steps, and action dim `32`. Generated eval/readout and
  visual review are still pending.
- Iteration-600 generated artifact inspection passed for all 10 samples:
  generated/reference videos `301/301`, actions `300x32/300x32`, mean action
  RMSE `0.3673552634`, mean robot-action future RMSE `0.9428412691`, mean
  state-sidecar future RMSE `0.0746684971`, and mean future-video PSNR
  `22.6652227928`. Visual review of all 10 sheets is materially better than
  the old failed full-data chain: target-motion/static panels are coherent and
  no global white/fog/transparent collapse appears. Peg-drop/regrasp and
  controller-ready handoff remain unresolved until readout/profile evidence.
- Iteration-600 generated-RGB readout/profile completed with strict structure:
  mean final hole error `0.0613948691` m, mean future hole RMSE
  `0.0328006673` m, mean future peg RMSE `0.0435265400` m, mean future TCP
  RMSE `0.0433667297` m, and mean future peg-head-hole RMSE `0.0449233321` m.
  Compared with iter300, this improves final hole, future hole, future peg,
  and future TCP diagnostics. It still does not prove controller readiness:
  simple threshold target-motion onset fires tens of frames too early and
  static/peg-only samples still show false target-motion onset.
- Iteration-700 validation loss was `0.725782`. This is a small rebound from
  the iter600 best `0.718841`, but it remains better than iter400/iter500 and
  does not by itself prove saturation. The active decision is to continue the
  overfit-validated full1000 fix1 run to the already-launched iter900 strict
  generated-video/action/readout/visual gate before any stop or controller/DP
  integration decision.
- Iteration-800 validation loss was `0.641040`, a clear new best over
  iteration 600. This confirms the full1000 fix1 run is still improving by
  validation loss; continue training and use the iter900 generated
  video/action/readout/visual gate as the next method-evidence check.
- Checkpoint `iter_000000900` saved at `2026-06-10 16:36:08 CST`; iteration
  900 validation loss was `0.634057`, again a new best over iteration 800.
  The aux eval watcher detected the checkpoint, waited for stable files, and
  passed strict eval input construction for 10 diverse samples with `301`
  expected RGB frames and `300x32` actions. Inference is running in allocation
  `123499`; generated artifacts, readout/profile, and direct visual review
  remain pending.

Iter900 generated evidence is now complete:

- Strict artifact inspection passed for all 10 samples. Generated/reference
  videos are `301/301` frames and generated/reference actions are
  `300x32/300x32`.
- Mean action RMSE improved to `0.2946495338`.
- Mean robot-action future RMSE improved to `0.7637177886`.
- Mean state-sidecar future RMSE improved to `0.0684030772`.
- Mean future-video PSNR is `22.6316859914`.
- All 10 review sheets were opened. The generated rollouts remain visually
  coherent, with no global white/fog/transparent collapse. Static,
  target-motion, target-post-motion, and insert-resume panels are stable
  enough to count as real SFT progress relative to the old failed chain.
  Peg-drop/regrasp remains unresolved as executable handoff evidence.
- Generated-RGB readout/profile passed strict structure. Aggregate metrics:
  mean final hole error `0.0678502409` m, mean future hole RMSE
  `0.0335725300` m, mean future peg RMSE `0.0417441050` m, mean future TCP
  RMSE `0.0375097505` m, and mean future peg-head-hole RMSE
  `0.0416219164` m.

Interpretation: iter900 is the best current full1000 fix1 checkpoint by
validation loss and improves action/robot-action/state-sidecar diagnostics
over iter600. It is still not controller/DP integration evidence, because the
target-motion onset profile remains early on moving-target samples and still
false-fires on static/peg-only samples. The aligned decision is to keep the
same 4-H200 full1000 SFT running to later strict checkpoint gates rather than
returning to overfit or starting controller integration.

After iter900, iteration-1000 validation rebounded to `0.697886`. This is not
a saved checkpoint gate and the run resumed normally at iter1001/1002 with
finite losses, so it is recorded as a validation fluctuation rather than a
stop condition. Iter1200 strict eval/readout/profile watchers are staged in
aux allocation `123499`.

Iteration-1100 validation then improved to `0.607350`, a new best for the
fresh full1000 fix1 run. The run resumed normally at iter1101/1102, so the
aligned decision is to continue to iter1200 checkpoint generation and strict
generated-video/action/readout/visual review.

Checkpoint `iter_000001200` saved at `2026-06-10 18:15:30 CST`. Iter1200
validation was `0.675294`, worse than the iter1100 best, but the training step
resumed normally at iter1201. The aux watcher in allocation `123499` detected
the checkpoint, passed strict eval input construction for 10 samples under the
`301` frame / `300x32` action contract, and started generated-video inference.

Iter1200 generated evidence is now complete:

- Strict artifact inspection passed for all 10 samples. Generated/reference
  videos are `301/301` frames and generated/reference actions are
  `300x32/300x32`.
- Mean action RMSE is `0.3327156417`.
- Mean robot-action future RMSE is `0.8618907214`.
- Mean state-sidecar future RMSE is `0.0662339055`.
- Mean future-video PSNR is `22.7099074529`.
- All 10 review sheets were opened. The rollouts are still coherent and do not
  show the old global white/fog/transparent collapse. Target-motion, static,
  and insert-resume panels remain usable as SFT progress evidence. Peg-drop
  recovery is still not a reliable executable regrasp/handoff behavior.
- Generated-RGB readout/profile passed strict structure. Aggregate metrics:
  mean final hole error `0.0620453945` m, mean future hole RMSE
  `0.0323648744` m, mean future peg RMSE `0.0414618213` m, mean future TCP
  RMSE `0.0359076100` m, and mean future peg-head-hole RMSE
  `0.0419642333` m.

Interpretation: iter1200 improves future-video PSNR and future TCP/peg readout
relative to iter900, but action RMSE and robot-action future RMSE regress from
iter900 and the target-motion onset profile still fires too early or falsely on
no-target-motion samples. It is therefore not controller/DP integration
evidence. The aligned decision is to keep the overfit-validated full1000 fix1
SFT running to the configured final `iter_000001500` gate.

Iteration-1300 validation later reported `0.694085`, still worse than the
iter1100 best `0.607350` and slightly worse than iter1200. Training resumed
normally at iter1301/1302 with low finite losses, so this is a validation
rebound, not a crash or checkpoint evidence. The next evidence gate remains
the strict iter1500 generated-video/action/readout/visual review.

Iteration-1400 validation reported `0.625498`, recovering from iter1200/1300
but still worse than the iter1100 best `0.607350`. Training resumed normally
at iter1401 with finite low loss. This supports continuing to the final
checkpoint gate; it is not generated-video or controller evidence by itself.

Checkpoint `iter_000001500` saved at `2026-06-10 19:54:51 CST`. Final
validation was `0.662956`, worse than the iter1100 best, and the trainer
reported done. The final checkpoint is therefore a strict evaluation gate, not
a validation-only success claim.

The final iter1500 gate completed in the auxiliary 1-H200 allocation `123499`.
Strict artifact inspection passed for all 10 samples with `301/301` generated
and reference videos, `300x32/300x32` generated and reference actions, and no
strict failures. Aggregate metrics: mean action RMSE `0.3157882582`, mean
robot-action future RMSE `0.8203911022`, mean state-sidecar future RMSE
`0.0626562464`, and mean future-video PSNR `23.1641813684`.

Generated-RGB readout/profile also passed strict structure. Aggregate metrics:
mean final hole error `0.0538804681` m, mean future hole RMSE
`0.0301422454` m, mean future peg RMSE `0.0386116191` m, mean future TCP RMSE
`0.0347898968` m, and mean future peg-head-hole RMSE `0.0413906728` m. These
are improved readout diagnostics relative to iter1200, but they are not
controller success.

All 10 iter1500 review sheets were opened. The generated videos remain
readable and coherent, with no old global white/fog/transparent collapse.
Static and many target-motion panels are close enough to count as SFT progress
evidence. The controller-facing boundary remains negative: peg_drop,
peg_disturb, and insert_resume panels still do not prove reliable
peg-gripper-hole contact continuity, and the target-onset profile still fires
too early or falsely on static/peg-only samples. No Slurm allocation was
cancelled.

A calibrated target-motion head diagnostic was re-run on the current fresh
fix1 iter1500 generated-RGB readout in aux allocation `123499`, step `12`.
Output root:
`experiments/world_model_task_rebinding/cosmos3/target_motion_readout_calibration_fresh_fix1_iter1500_20260610`.
The reference-RGB held-out result remains AUROC `0.9115353604`, F1@0.5
`0.7669897596`, best F1 `0.7788987602`; the current fresh generated-RGB panel
reaches only AUROC `0.7808262378`, F1@0.5 `0.6005305040`, best F1
`0.6179090483`. This supports the same boundary as the visual review: a
calibrated switch is feasible with reference-like readout, but current Cosmos3
generated rollouts are not reliable enough for controller-facing switching.
