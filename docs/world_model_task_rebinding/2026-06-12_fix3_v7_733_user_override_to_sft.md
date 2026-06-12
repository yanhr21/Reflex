# Fix3 V7 733-Row User Override To SFT

Date: 2026-06-12.

Current authoritative state after the latest fix1-recipe restart:

- The current authoritative full-data v7_733 SFT root is
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`.
- It used the overfit-approved fix1 action recipe and ended at Slurm wall time
  after rank-0 iteration `743`. This was a Slurm time-limit stop, not an agent
  `scancel`.
- Saved checkpoints are only `iter_000000300` and `iter_000000600`; there is
  no iter900/iter1200 checkpoint or active watcher for this root.
- Both evaluated checkpoints pass the strict 301-frame / 300-action artifact
  contract, but neither is controller-ready. Iter300 is the best qualitative
  sanity checkpoint so far and still has imprecise handoff geometry; iter600 is
  worse on rollout/readout/visual evidence despite lower validation loss.
- Closed-loop DP/controller work remains gated off. The `normactive_clip1`
  iter900/iter1200 notes later in this chronological document are historical
  negative diagnostics from a rejected recipe, not the active review gate.
- A conservative closed-loop gate checker was added at
  `scripts/world_model/check_cosmos3_closed_loop_gate.py`. It reads the strict
  eval artifact JSON, generated-RGB readout summary, readout failure profile,
  and an explicit visual-review verdict. It blocks by default unless all three
  gates pass. Running it on the latest fix1-recipe iter300 and iter600 roots
  with the agent visual verdict set to `fail` returned
  `closed_loop_allowed=false` for both; the blocking reason is
  `explicit_visual_review_not_passed`.

Latest user instruction stopped data construction immediately and directed the
agent to proceed to Cosmos3 SFT. The old gate requiring exactly 1000 rows and
then stopping for user approval is superseded for this immediate run.

Frozen source:

- root:
  `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733`
- total unique H5 trajectories: `733`
- class counts:
  `hole_late_move_stop=44`, `hole_late_constant=48`,
  `hole_late_reverse=99`, `hole_late_sine=60`,
  `hole_late_continuous_insert=96`, `hole_late_fast_shift=105`,
  `none=160`, `peg_drop=119`, `peg_disturb=2`
- all nine classes are present, but this is not quota-complete full1000 data.

Generation stop:

- active v7 generator processes were interrupted with SIGINT inside held Slurm
  allocations;
- no allocation was cancelled by `scancel`;
- held allocations `126210`, `126219`, and `126223` remain available for
  render/export/SFT work.

Audit:

- original 1000-quota audit produced `num_failed_records=0` and failed only
  on path/count mismatch, which is expected under the latest user override;
- user-override quota audit passed:
  `fix3_v7_dp_user_override_sft_source_20260612_733/strict_source_h5_audit_user_override_quota733`,
  `strict_ok=true`, `num_failed_records=0`.

Current action:

- render frozen H5 paths to RGB SFT dataset root:
  `experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612`;
- render uses the approved ManiSkill default human camera, `512x512`, `30 fps`,
  frame stride `1`, expected `301` frames;
- after render, run WAM condition export, strict full-episode preflight,
  action-target audit, and Cosmos3 SFT under the 301-frame / 300-action
  contract.

Completed render/data checks:

- the rendered dataset contains `733` final videos split into `661` train and
  `72` val rows;
- dataset inspection passed with `valid=true`, expected `512x512`, `30 fps`,
  state metadata, captions, and video-prefix policy;
- video artifact inspection passed for both splits:
  train `661/661` readable and nonblank, val `72/72` readable and nonblank;
- agent visual review opened representative review sheets for moving-hole
  classes, `none`, `peg_drop`, and both `peg_disturb` rows; the user later
  reported the videos looked acceptable.

WAM export and SFT startup:

- condition root:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_rgb_300step_20260612_0245`;
- SFT output root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_4gpu_20260612_0245`;
- full-episode preflight and action target audit completed before SFT;
- action targets are `300x32`, use future-aligned state sidecars, and
  `strict_action_target_ok=true`;
- SFT tmux session:
  `cosmos3_sft_fix3_v7_733_4gpu_126210`;
- Slurm allocation: `126210` on `server56`, `4xH200`;
- startup evidence: `Cosmos3-Nano-Policy-DROID-DCP` checkpoint loaded,
  train/val dataloaders prewarmed, and trainer entered `Starting training...`.

SFT completion and first generated-eval gate:

- SFT completed normally in held Slurm allocation `126210` on `server56`;
- completion marker:
  `sft_full_episode_wam_fix3_v7_733_rgb_300step_4gpu_20260612_0245/sft_completed`,
  timestamp `2026-06-12T10:36:20+08:00`;
- final checkpoint:
  `outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/checkpoints/iter_000001500`;
- latest checkpoint pointer is `iter_000001500`;
- final logged training iteration `1500` had loss `3.5723`, vision `0.0343`,
  and action `0.3538`;
- validation loss sequence was finite and decreasing before plateau:
  iter `0=11.343520`, `300=4.372651`, `600=3.661371`, `900=3.519888`,
  `1200=3.496552`, `1500=3.526614`;
- strict generated-eval root:
  `sft_full_episode_wam_fix3_v7_733_rgb_300step_4gpu_20260612_0245/eval_full_episode_wam_iter_000001500`;
- strict artifact inspection passed:
  `strict_eval_artifacts_ok=true`, `strict_failures=[]`, `10` samples,
  generated/reference video length `301`, action tensor shape `300x32`;
- aggregate generated-eval diagnostics were weak:
  mean future video PSNR `19.6738554813`, mean action RMSE `0.6634088986`,
  mean robot-action future RMSE `1.0473393741`, and mean state-sidecar future
  RMSE `0.7528493327`.

Visual review of generated eval:

- the agent opened all `10` ref/pred review sheets under
  `eval_full_episode_wam_iter_000001500/review_sheets`;
- the generated videos are structurally complete but not controller-ready:
  after the prefix, many predictions show semi-transparent/ghosted robot
  geometry, poor peg/gripper/contact continuity, and unreliable insert/resume
  or peg-recovery behavior;
- this is negative SFT diagnostic evidence. It does not support closed-loop
  controller/DP handoff.

Readout status:

- generated-RGB task-state readout was started inside held allocation `126210`
  after the strict eval gate, with current v7_733 dataset manifest and eval
  root;
- at readout step `250`, reference-RGB validation was still poor:
  future hole RMSE `0.0880907178` m, future peg RMSE `0.0580688566` m,
  future TCP RMSE `0.0492440201` m, future peg-head-hole RMSE
  `0.1031634808` m, future grasped accuracy `0.8934742808`, and future
  inserted accuracy `0.7359374762`;
- final readout step `2000` reference-RGB validation improved but remained
  only moderate: future hole RMSE `0.0390807018` m, future peg RMSE
  `0.0338327438` m, future TCP RMSE `0.0326168649` m, future
  peg-head-hole RMSE `0.0579820797` m, future grasped accuracy `0.9406250119`,
  and future inserted accuracy `0.9428308606`;
- generated-RGB readout eval passed strict structure over `10/10` samples but
  was negative for controller use: mean final hole error `0.1211597189` m,
  mean future hole RMSE `0.0612715537` m, mean future peg RMSE
  `0.0640420842` m, mean future TCP RMSE `0.0652820180` m, and mean future
  peg-head-hole RMSE `0.0502333957` m;
- target-motion/onset diagnostics were also negative. Moving target samples
  predicted target motion at frames `5-6` while target onsets were frames
  `76-116` at the 2 mm threshold, producing `70-110` frame onset errors;
  static/peg-only samples also predicted false target motion around frames
  `5-8`;
- failure profile:
  `eval_full_episode_wam_iter_000001500/task_state_readout_v7_733/readout_failure_profile.md`.

Current conclusion:

- this run proves the current v7_733 SFT/export/eval/readout chain is
  structurally runnable under the 301-frame / 300-action contract;
- it does not provide controller-ready world-model evidence. The generated
  videos are visually unstable after the prefix, the action/readout metrics are
  weak, target motion fires far too early, and final target position is too
  inaccurate;
- closed-loop/receding DP/controller evaluation must remain gated off for this
  checkpoint.

Fix1-recipe restart after user-approved overfit:

- The bad `normactive_clip1` follow-up was rejected after user visual review
  matched the old "action adapter not really trained" failure. The concrete
  config bug was that the run selected action tensors but did not carry the
  overfit-validated fix1 recipe: it used `lr=2e-5`,
  `action_loss_weight=10.0`, `independent_action_schedule=false`, and
  `shift_action=None`.
- The default full-episode WAM SFT wrapper now enforces the fix1 action recipe:
  `lr=1e-4`, warmup `10`, `f_min=0.5`, `grad_clip_norm=1.0`,
  `action_loss_weight=2.0`, `normalize_loss_by_active=true`,
  `independent_action_schedule=true`, `shift_action=1`, and optimizer keys
  including `action2llm`, `llm2action`, and `action_modality_embed`.
- Two-sample overfit under this recipe passed user visual review at iter100.
  The full v7_733 SFT was then restarted at
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`,
  tmux `cosmos3_sft_v7_733_full_fix1recipe_4gpu_126210`, Slurm step
  `126210.41` on `server56`.
- Launch/startup sanity passed: `fix1_action_recipe_check=passed`, `410`
  trainable tensors selected, optimizer `lr=0.0001`,
  `action_loss_weight=2.0`, `independent_action_schedule=true`, and
  `shift_action=1`. Iter0 validation loss was `3.606580`.

Fix1-recipe iter300 eval/readout/profile:

- Checkpoint `iter_000000300` saved at `2026-06-12 20:51 CST`; validation loss
  was `0.155843`. Training then resumed normally and stayed finite.
- Strict generated-eval root:
  `sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_full_episode_wam_iter_000000300`.
- Strict eval passed structurally for `10` samples:
  `strict_eval_artifacts_ok=true`, `strict_failures=[]`,
  generated/reference video frames `301/301`, and action tensor shape
  `300x32`.
- Aggregate generated-eval metrics: mean future video PSNR `21.6543040597`,
  mean action RMSE `0.3543249375`, mean robot-action prefix RMSE
  `0.0017889231`, mean state-sidecar prefix RMSE `0.0015759602`,
  mean robot-action future RMSE `0.6354126001`, and mean state-sidecar future
  RMSE `0.3533818061`.
- Generated-RGB readout/profile used the existing v7_733 reference readout
  checkpoint
  `sft_full_episode_wam_fix3_v7_733_rgb_300step_4gpu_20260612_0245/task_state_readout_reference_rgb_301f_v7_733/best_model.pt`.
  It passed strict structure over `10/10` samples.
- Generated-RGB readout/profile metrics: mean final hole error
  `0.0655233613` m, mean future hole RMSE `0.0391585658` m, mean future peg
  RMSE `0.0400948399` m, mean future TCP RMSE `0.0399431949` m, and mean
  future peg-head-hole RMSE `0.0318345695` m.
- The agent opened all `10` ref/pred review sheets. Visual quality is much
  better than the rejected bad-recipe run: no old white/fog-like robot
  collapse, no obvious full-scene geometry blow-up, and moving-hole samples
  show readable target motion with plausible robot/peg motion. However, several
  final peg/hole relative poses remain imprecise, and target-onset diagnostics
  still false-fire on static samples or fire early at low thresholds on moving
  samples. This checkpoint is therefore a positive training/eval sanity result,
  but not controller-ready world-model evidence.
- The `iter_000000600` strict eval/readout/profile watcher completed in held
  auxiliary allocation `127120`; the result is recorded in the next section.
  Controller/DP integration remains gated off.

Fix1-recipe iter600 eval/readout/profile:

- Checkpoint `iter_000000600` saved at `2026-06-12 22:21 CST`; validation loss
  was `0.131243`. Training then resumed normally with finite losses.
- Strict generated-eval root:
  `sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_full_episode_wam_iter_000000600`.
- Strict eval passed structurally for `10` samples:
  `strict_eval_artifacts_ok=true`, `strict_failures=[]`,
  generated/reference video frames `301/301`, and action tensor shape
  `300x32`.
- Aggregate generated-eval metrics are worse than iter300 despite the lower
  validation loss: mean future video PSNR `20.2910293787`, mean action RMSE
  `0.6189765621`, mean robot-action prefix RMSE `0.0017889231`, mean
  state-sidecar prefix RMSE `0.0015759602`, mean robot-action future RMSE
  `0.9830810889`, and mean state-sidecar future RMSE `0.6805342149`.
- Generated-RGB readout/profile passed strict structure over `10/10` samples,
  but also degraded relative to iter300: mean final hole error
  `0.1057760996` m, mean future hole RMSE `0.0603418101` m, mean future peg
  RMSE `0.0795095738` m, mean future TCP RMSE `0.0761800148` m, and mean
  future peg-head-hole RMSE `0.0457403359` m.
- The agent opened all `10` iter600 ref/pred sheets. There is no old global
  white-fog or full-scene geometry-collapse failure, but the checkpoint is not
  controller-ready. Several sheets show target/robot relative-pose drift after
  the prefix, peg/contact discontinuity, or target-final-position errors.
  Examples: static samples `01` and `02` drift in block/robot relative pose;
  moving target sample `04` loses the final peg/block alignment; moving
  insert-resume sample `07` leaves unsafe peg/contact geometry. This visual
  evidence agrees with the higher action/state/readout errors.
- Conclusion: do not select checkpoints by validation loss alone. Iter300 is
  the better qualitative sanity checkpoint so far, but it is also not yet
  controller-ready. Controller/DP integration remains gated until a checkpoint
  passes generated video, action metrics, readout metrics, and direct visual
  review together.

End of current fix1-recipe full SFT allocation:

- The 4-H200 training step `126210.41` ended at `2026-06-12 23:06:27 CST`
  because Slurm hit the allocation time limit:
  `STEP 126210.41 ON server56 CANCELLED ... DUE TO TIME LIMIT`.
- This was not an agent `scancel`; the run was allowed to continue until the
  system stopped the step.
- The final rank-0 training log reached iteration `743` with finite losses
  (`Loss=0.1963`, `vision=0.0119`, `action=0.0922` at iteration `743`).
  No traceback, CUDA OOM, or NaN marker was found in the inspected tail.
- Saved checkpoints remain exactly `iter_000000300` and `iter_000000600`.
  No `iter_000000900` checkpoint or eval exists, and no `sft_completed`
  marker exists for this root.
- Current evidence state: iter300 is the best qualitative sanity checkpoint
  among evaluated checkpoints, iter600 is controller-negative, and the run has
  no later checkpoint evidence. Do not advance to DP/controller integration
  from this root without a new checkpoint that passes the same strict
  full-episode generated-video/action/readout/visual gate.

Post-SFT code-path audit:

- The SFT/export path writes multiple full-episode rows per source trajectory,
  one per causal prefix/mode. Every row still references the complete `301`
  RGB-frame video and a `300x32` action/state target file; it is not a
  128/129-frame clip and not a 93-frame crop.
- The JSONL rows carry per-row `condition_frame_indexes_vision` and
  `condition_frame_indexes_action`. The local Cosmos SFT dataset loader uses
  these per-row fields when `action_path` exists, overriding the global
  `conditioning_config={8: 1.0}` plan for action-conditioned samples. Thus the
  training/eval prefix masks are variable and causal, not fixed to eight latent
  frames.
- The action file contains the full normalized `300x32` target. The sequence
  plan marks only prefix action indexes as clean conditions; future action/state
  tokens are noised and supervised by the flow-matching loss. The current audit
  therefore did not find future-action leakage through the condition mask.
- The artifact inspector was extended to report prefix/future action metrics
  separately. Re-running it on the same `10` eval samples kept
  `strict_eval_artifacts_ok=true` and showed that conditioned history is
  preserved: mean robot-action prefix RMSE `0.0017889231` and mean state-sidecar
  prefix RMSE `0.0015759602`, versus future RMSE `1.0473393741` and
  `0.7528493327`. This localizes the failure to future prediction quality, not
  to missing/ignored prefix action-state conditions.
- The negative result should currently be attributed to data/model/interface
  capability rather than an obvious length truncation or prefix-mask mismatch:
  the source is DP-success-filtered and not hard-dynamic evidence, and the
  current checkpoint still fails generated video/contact/readout quality gates.

Current source-distribution diagnosis:

- The frozen source has `733` source episodes expanded into `2899` full-episode
  prefix rows (`2620` train, `279` val). Scenario coverage is skewed:
  `none=160`, `peg_drop=119`, `hole_late_fast_shift=105`,
  `hole_late_reverse=99`, `hole_late_continuous_insert=96`,
  `hole_late_sine=60`, `hole_late_constant=48`,
  `hole_late_move_stop=44`, and `peg_disturb=2` source episodes.
- Prefix-role counts are `insert_resume=733`, `target_pre_motion=573`,
  `target_motion_observed=573`, `target_post_motion=573`,
  `static_monitor=160`, `static_late_monitor=160`, and `peg_recovery=127`.
  `peg_recovery` is almost entirely from `peg_drop`; moving-hole recovery rows
  are only a handful, and `peg_disturb` has only `10` total prefix rows.
- This distribution can exercise the full-episode SFT interface, but it is too
  weak to support a claim that Cosmos3 learned robust peg disturbance/recovery
  or hard dynamic target rebinding.

Follow-up SFT diagnostic launched:

- A second v7_733 bootstrap SFT run was started in held Slurm allocation
  `126210` on `server56`, step `126210.38`, tmux session
  `cosmos3_sft_v7_733_normactive_clip1_4gpu_126210`.
- Output root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_normactive_clip1_4gpu_20260612_124500`.
- It reuses the existing frozen v7_733 rendered dataset and WAM condition root;
  no data generation, rendering expansion, 128/129-frame slicing, or source
  reset is involved.
- Changed settings relative to the first v7_733 SFT:
  `normalize_loss_by_active=true` and `grad_clip_norm=1.0` instead of
  `false` and `0.1`. The reason is concrete: prefix/future metric splitting
  showed conditioned history is preserved but future prediction is poor, while
  the first run diluted masked-prefix losses through the full tensor mean and
  logged global grad norms around `7.x` under a `0.1` clip threshold.
- This follow-up is still v7_733 DP-success-filtered bootstrap evidence only.
  It can test whether the future-rollout training objective was underweighted,
  but it cannot prove task-success gain over frozen DP.
- Startup status: model/checkpoint load and dataloader prewarm completed, and
  training entered `Starting training...`. Validation-on-start produced finite
  iter-0 loss `17.642225`. This scalar is not directly comparable to the first
  run's `11.343520` because the loss denominator changed from full tensor mean
  to active-token normalization. Early logged global grad norms remain around
  `7-8`, now under `grad_clip_norm=1.0` instead of `0.1`.
- Monitoring update at `2026-06-12 13:11 CST`: the 4-GPU SFT step is still
  live as Slurm step `126210.38` on `server56`; recent rank-0 logs reached
  iteration `63` with finite loss around `12.4`, vision loss around `0.075`,
  and action loss around `1.236`. No `sft_failed` marker and no checkpoint
  directory are present yet. Because the same allocation is occupied by
  training, strict generated eval is deferred until a checkpoint exists and GPU
  resources are available. A separate tmux-held 1-GPU eval allocation was
  requested as job `126985` (`cosmos3_v7_eval_aux_1h200`) and is currently
  pending; it will be used for checkpoint eval only if it starts without
  contending with the 4-GPU SFT.
- Monitoring update at `2026-06-12 13:14 CST`: auxiliary eval allocation
  `126985` started on `server40`. A compute-node watcher is running as Slurm
  step `126985.0` from tmux session
  `cosmos3_v7_733_iter300_eval_watch_126985`, watching
  `iter_000000300` and configured to write
  `eval_full_episode_wam_iter_000000300` with `10` validation samples after
  checkpoint files are stable. This does not use the login node for eval and
  does not share GPUs with the 4-GPU SFT step.
- The iter-300 generated-RGB readout/profile chain is queued behind that eval
  step in the same held 1-GPU allocation through tmux sessions
  `cosmos3_v7_733_iter300_readout_watch_126985` and
  `cosmos3_v7_733_iter300_profile_watch_126985`. The readout watcher now
  explicitly checks `strict_eval_artifacts_ok=true` in
  `eval_artifact_inspection.json` before running generated-RGB readout, so a
  failed 301-frame / 300-action artifact gate will not be pushed downstream as
  readout evidence.

- Monitoring update at `2026-06-12 13:27 CST`: the 4-GPU SFT remains healthy
  in step `126210.38`. Recent rank-0 logs reached iteration `125`, with
  stable throughput around `17.3` seconds/iteration. Loss has fallen from the
  iter-0 validation scale and early train losses to roughly the `6-8` train
  range; recent examples include iteration `123` loss `6.2342`, vision
  `0.0432`, action `0.6191`, and iteration `125` loss `8.0516`, vision
  `0.0597`, action `0.7992`. No checkpoint has been written yet, which is
  expected before iteration `300`. Eval watcher `126985.0` is still waiting for
  `iter_000000300`; queued readout/profile `srun` messages about busy nodes
  are expected because the eval watcher currently owns the auxiliary GPU step.
- Monitoring update at `2026-06-12 13:52 CST`: the same 4-GPU SFT step reached
  rank-0 iteration `212`, still around `17.3` seconds/iteration, with recent
  train losses fluctuating roughly from `4.6` to `7.5` and action loss roughly
  from `0.45` to `0.74`. Device monitor at iteration `200` reported healthy
  GPU memory headroom (`~58.6GB` used, `~81.8GB` free by NVML) and high GPU
  utilization. No checkpoint has been written yet because the first save point
  is `iter_000000300`; eval watcher `126985.0` remains correctly waiting.
- Monitoring update at `2026-06-12 14:35 CST`: `iter_000000300` was saved and
  strict same-length eval completed in the auxiliary allocation. The artifact
  gate passed structurally with `10` samples, generated/reference videos
  `301/301`, action tensors `300x32`, `strict_failures=[]`, mean future video
  PSNR `19.6801159248`, mean action RMSE `0.6560889600`, robot-action prefix
  RMSE `0.0017889231`, state-sidecar prefix RMSE `0.0015759602`,
  robot-action future RMSE `1.0274358407`, and state-sidecar future RMSE
  `0.7472363522`. This confirms the full-episode eval path is structurally
  intact, but future rollout quality is still weak.
- Generated-RGB readout/profile also completed for `iter_000000300` and is
  negative for controller handoff: mean final hole error `0.1143826719` m,
  mean future hole/peg/TCP RMSE `0.0548931954` / `0.0669841219` /
  `0.0657259292` m, and mean future peg-head-hole RMSE `0.0522709684` m.
  Target-onset diagnostics still fire far too early and false-fire on static
  samples; for example moving samples predict onset around frames `5-6` while
  targets start around frames `76-116`, and static/peg-only samples also
  report early target motion.
- Visual review opened all `10` iter-300 review sheets. This checkpoint is not
  controller-ready: several moving-hole and insert-resume panels show visible
  robot/peg divergence after the prefix, dropped or misplaced pegs, large
  white/fog-like ghosting around the robot/gripper in later frames, and
  unreliable peg-hole contact. In particular samples `04`, `07`, `08`, and
  `09` are clearly unsuitable as DP-resume future-state images.
- The SFT itself continues on the 4-GPU allocation past iteration `330`.
  A new iter-600 strict eval/readout/profile watcher chain was started in the
  held 1-GPU allocation `126985`: tmux sessions
  `cosmos3_v7_733_iter600_eval_watch_126985`,
  `cosmos3_v7_733_iter600_readout_watch_126985`, and
  `cosmos3_v7_733_iter600_profile_watch_126985`. The eval watcher is running
  on compute node `server40` as Slurm step `126985.3` and is waiting for
  `iter_000000600`; readout/profile are queued behind it in the same held
  allocation.
- Monitoring update at `2026-06-12 14:41 CST`: the follow-up SFT remains live
  on 4 GPUs as Slurm step `126210.38` and has reached rank-0 iteration `367`.
  Recent train losses are finite and mostly in the `4-6` range, with action
  loss around `0.4-0.6`. No new checkpoint beyond `iter_000000300` exists yet.
  The iter-600 eval watcher remains active on compute node `server40` as step
  `126985.3`; readout/profile watchers are still waiting behind that step and
  their `Requested nodes are busy` messages are expected while the eval watcher
  owns the held 1-GPU allocation.
- Monitoring update at `2026-06-12 14:45 CST`: the SFT is still healthy on
  `server56`, now at rank-0 iteration `379`. The latest logs show finite train
  loss in the same `4-6` range; no error or `sft_failed` marker was found.
  Checkpoints still contain only `iter_000000300`, so there is not yet a new
  generated eval artifact to inspect. The iter-600 eval watcher remains active
  as step `126985.3` on `server40` and is still printing
  `checkpoint_not_ready`; readout/profile remain queued behind it.
- Monitoring update at `2026-06-12 14:48 CST`: the SFT remains live and
  healthy at rank-0 iteration `390`. Recent train loss is still finite in the
  same range, with no traceback, runtime error, or `sft_failed` marker in the
  log. No checkpoint beyond `iter_000000300` exists yet, so there are still no
  `iter_000000600` eval/readout/profile artifacts to inspect. The iter-600
  watcher is still correctly waiting for the checkpoint on compute node
  `server40`; readout/profile stay queued behind it in the same held
  allocation.
- Monitoring update at `2026-06-12 14:51 CST`: the SFT reached rank-0
  iteration `400` with finite loss (`4.1276`, vision `0.0416`, action
  `0.4086`) and no error marker. Checkpoints still contain only
  `iter_000000300`; `iter_000000600` is still pending. The eval watcher on
  `server40` is healthy and still printing `checkpoint_not_ready`; readout and
  profile remain queued behind it. At the current `~17.3s/iter` pace,
  `iter_000000600` is roughly one hour away from this timestamp.
- Monitoring update at `2026-06-12 15:00 CST`: the follow-up SFT remains live
  on `server56` as Slurm step `126210.38` and has reached rank-0 iteration
  `435`. Recent train loss is finite and still weighted-action dominated; the
  latest checked line is `Loss=4.6974`, vision `0.0398`, action `0.4658`.
  Checkpoints still contain only `iter_000000300`, and there are no
  `iter_000000600` eval/readout/profile artifacts yet. The iter-600 eval
  watcher is active as step `126985.3` on `server40` and is correctly waiting
  for the checkpoint. Readout/profile watcher `Requested nodes are busy`
  messages are expected while that single held auxiliary GPU step is occupied
  by the eval watcher.
- Monitoring update at `2026-06-12 15:14 CST`: the follow-up SFT reached
  rank-0 iteration `480` with finite loss (`4.3913`, vision `0.0347`, action
  `0.4357`). Checkpoints still contain only `iter_000000300`; the iter-600
  eval watcher remains active and correctly reports `checkpoint_not_ready`.
  No `iter_000000600` eval/readout/profile artifact exists yet, so this is
  still training-health evidence only and not a closed-loop trigger.
- Monitoring update at `2026-06-12 15:30 CST`: the follow-up SFT reached
  rank-0 iteration `535` with finite loss (`4.6347`, vision `0.0418`, action
  `0.4593`). Checkpoints still contain only `iter_000000300`; no iter-600
  eval files exist. The held aux eval watcher continues to wait for
  `iter_000000600`, so no controller/closed-loop action is allowed yet.
- Monitoring update at `2026-06-12 16:05 CST`: checkpoint `iter_000000600`
  was saved and the aux allocation `126985` completed strict eval,
  generated-RGB readout, and readout failure profiling. The strict generated
  artifact gate passed structurally for `10` samples with generated/reference
  videos `301/301`, generated/reference actions `300x32`, and
  `strict_failures=[]`. Metrics improved only mildly over iter300: mean future
  PSNR `19.8479565666`, mean action RMSE `0.6165386443`,
  robot-action future RMSE `1.0047414112`, state-sidecar future RMSE
  `0.6844772048`, while prefix preservation stayed clean
  (`robot_action_prefix_rmse=0.0017889231`,
  `state_sidecar_prefix_rmse=0.0015759602`).
- The iter600 generated-RGB readout/profile passed strict structure but remains
  controller-negative. Mean final hole error is `0.0861941030` m; mean future
  hole/peg/TCP RMSE is `0.0478804191` / `0.0555931353` / `0.0568463799` m;
  mean future peg-head-hole RMSE is `0.0468595485` m. These values improve
  over iter300, but target onset is still unreliable: static/peg-only samples
  false-fire target motion around frames `5-17`, and moving samples often fire
  tens of frames before the true target onset at low displacement thresholds.
- Visual review opened all `10` iter600 review sheets. The checkpoint is not
  controller-ready: several predictions still show robot/peg divergence after
  the prefix, white/fog-like geometry collapse near the gripper/block, peg
  drops or contact discontinuities, and moving-hole insert/resume rollouts
  that are not usable as DP-resume future-state images. Closed-loop/receding
  DP/controller evaluation remains gated off for `iter_000000600`.
- An iter900 watcher chain was started in the held auxiliary allocation
  `126985`. The strict eval watcher runs as tmux session
  `cosmos3_v7_733_iter900_eval_watch_126985` and Slurm step `126985.6`,
  waiting for checkpoint `iter_000000900` on compute node `server40`.
  Generated-RGB readout and failure-profile watcher sessions
  `cosmos3_v7_733_iter900_readout_watch_126985` and
  `cosmos3_v7_733_iter900_profile_watch_126985` are queued behind the same
  allocation. No controller/closed-loop action is allowed until the iter900
  strict artifacts, readout/profile, and visual review are inspected.
- Monitoring update at `2026-06-12 16:14 CST`: the follow-up SFT remains live
  and healthy on `server56` as Slurm step `126210.38`. Rank-0 logs reached
  iteration `678` with finite loss (`3.9466`, vision `0.0319`, action
  `0.3915`), and no error marker was found in the checked log tail.
  Checkpoints still contain only `iter_000000300` and `iter_000000600`.
  The iter900 eval watcher remains active as Slurm step `126985.6` on
  `server40`, correctly waiting for `iter_000000900`; no iter900
  eval/readout/profile artifact exists yet.
- Monitoring update at `2026-06-12 16:26 CST`: the follow-up SFT remains live
  on `server56` as Slurm step `126210.38`. Rank-0 logs reached iteration
  `718` with finite loss (`3.7110`, vision `0.0336`, action `0.3677`), and
  checkpoints still contain only `iter_000000300` and `iter_000000600`.
  The iter900 eval watcher remains active as Slurm step `126985.6` on
  `server40`, with `checkpoint_not_ready` through `960` seconds. No
  `iter_000000900` strict eval/readout/profile artifact exists yet, so
  controller/closed-loop evaluation remains gated off.
- Monitoring update at `2026-06-12 16:32 CST`: the follow-up SFT remains live
  on `server56` as Slurm step `126210.38`. Rank-0 logs reached iteration
  `739` with finite loss (`4.0361`, vision `0.0451`, action `0.3991`), and
  checkpoints still contain only `iter_000000300` and `iter_000000600`. The
  iter900 eval watcher remains active as Slurm step `126985.6` on `server40`,
  with `checkpoint_not_ready` through `1320` seconds.
- A light active-script/TODO audit during the wait confirmed the current
  active script surface mainly supports strict SFT eval, generated-RGB
  task-state readout, and readout failure profiling. Old controller wrappers
  were intentionally moved by the archive boundary. If a future checkpoint
  passes generated-video/action/readout/visual review, the next controller
  step must implement or use the PLAN receding interface
  (observe, predict remaining horizon, execute short action prefix, reobserve,
  refresh) rather than restarting an archived shortcut controller.
- Monitoring update at `2026-06-12 16:39 CST`: the follow-up SFT remains live
  on `server56` as Slurm step `126210.38`. Rank-0 logs reached iteration
  `762` with finite loss (`4.0525`, vision `0.0292`, action `0.4023`), and
  checkpoints still contain only `iter_000000300` and `iter_000000600`. The
  iter900 eval watcher remains active as Slurm step `126985.6` on `server40`,
  with `checkpoint_not_ready` through `1740` seconds; readout/profile watcher
  `Requested nodes are busy` messages are expected because that single
  auxiliary GPU step is held by the strict eval watcher while waiting.
- Focused script audit at `2026-06-12 16:39 CST` found no active source
  controller or receding wrapper under `scripts/world_model` or
  `scripts/slurm`; only stale `__pycache__` entries mention receding/controller
  distillation. This does not block the current SFT/eval/readout chain, but it
  means a future passed SFT gate must be followed by an implemented/restored
  PLAN-compliant receding interface before any live DP/controller test is run.
- Monitoring update at `2026-06-12 16:43 CST`: SFT is still live and healthy
  on `server56` as Slurm step `126210.38`. Rank-0 logs reached iteration
  `777` with finite loss (`4.0125`, vision `0.0488`, action `0.3964`).
  Checkpoints remain `iter_000000300` and `iter_000000600`; the iter900
  watcher remains active as Slurm step `126985.6` on `server40`, with
  `checkpoint_not_ready` through `1980` seconds. No iter900
  eval/readout/profile artifacts exist yet.
- Closed-loop dependency audit during the wait: Cosmos eval inputs/outputs
  use a normalized full `300x32` sequence. The first `7` columns are robot
  actions and the remaining `25` are task-state sidecars. The condition root
  stores `normalization_stats.json`; any future live controller wrapper must
  de-normalize only the first `7` robot-action columns before `env.step` and
  must not use sidecar columns as simulator oracle state. The frozen DP
  checkpoint manifest confirms `PegInsertionSide-v1`, `pd_ee_delta_pose`,
  `obs_horizon=2`, `act_horizon=8`, and `max_episode_steps=300`. Therefore a
  future passed-gate controller test must execute short 8-step-or-smaller
  action prefixes, reobserve real RGB/state, and refresh the prefix; it must
  not execute a single open-loop 300-step Cosmos rollout as method evidence.
- Monitoring update at `2026-06-12 16:56 CST`: SFT remains live and healthy on
  `server56` as Slurm step `126210.38`. Rank-0 logs reached iteration `820`
  with finite loss (`4.4826`, vision `0.0420`, action `0.4441`). Checkpoints
  still contain only `iter_000000300` and `iter_000000600`; no
  `iter_000000900` eval/readout/profile artifact exists. The iter900 eval
  watcher remains active on `server40` as Slurm step `126985.6`, with
  `checkpoint_not_ready` through `2701` seconds. The readout/profile watcher
  busy-node messages are expected because the strict eval watcher holds the
  single auxiliary GPU step while waiting. Closed-loop evaluation remains
  gated off.

Follow-up SFT checkpoint monitoring:

- `iter_000000300` strict eval/readout/profile passed structural checks but
  failed the controller handoff gate. It preserved prefix conditions, but
  future action/state/video quality was weak and direct review of all `10`
  sheets showed robot/peg divergence, dropped or misplaced peg paths,
  white/fog-like geometry artifacts, and unreliable peg-hole contact.
- `iter_000000600` improved numerically but still failed the controller gate.
  Strict eval passed with `301/301` videos and `300x32` actions. Aggregate
  metrics were mean future PSNR `19.8479565666`, mean action RMSE
  `0.6165386443`, robot-action future RMSE `1.0047414112`, and state-sidecar
  future RMSE `0.6844772048`. Generated-RGB readout/profile mean final hole
  error was `0.0861941030` m. Direct visual review remained negative because
  multiple samples had robot/peg divergence, geometry collapse around the
  gripper/block, peg drops or contact discontinuities, and unusable
  moving-hole insert/resume predictions.
- `iter_000000900` was saved at `2026-06-12 17:18:58 CST` and validation loss
  was `3.923666` at `17:22:20`. Strict artifact inspection passed over `10`
  samples with generated/reference videos `301/301`, action tensors
  `300x32`, and `strict_failures=[]`. Aggregate metrics were mean future PSNR
  `20.0838192282`, mean action RMSE `0.5730889361`, robot-action prefix RMSE
  `0.0017889231`, state-sidecar prefix RMSE `0.0015759602`, robot-action
  future RMSE `0.8931446655`, and state-sidecar future RMSE `0.6542504528`.
- `iter_000000900` generated-RGB readout/profile also passed strict
  structure. Mean final hole error was `0.0662412508` m; mean future
  hole/peg/TCP RMSE were `0.0401320661` / `0.0537541486` /
  `0.0534759435` m; mean future peg-head-hole RMSE was `0.0396896749` m.
  Target-onset diagnostics remain negative: static/peg-only samples still
  false-fire target motion early, and moving samples often predict target
  motion tens of frames before the true onset at low thresholds.
- Direct review opened all `10` `iter_000000900` review sheets. The checkpoint
  is still not controller-ready: several samples show robot/peg divergence,
  dropped or table-contact peg paths, white/transparent geometry collapse
  around the gripper/block, and non-physical contact continuity. Closed-loop
  controller evaluation remains gated off.
- As of `2026-06-12 18:01 CST`, the follow-up SFT is still live on Slurm step
  `126210.38` on `server56`, at rank-0 iteration `1036`, with finite loss and
  no error marker. The active auxiliary watcher is Slurm step `126985.9` on
  `server40`, waiting for checkpoint `iter_000001200`; no iter1200
  eval/readout/profile artifacts exist yet. Active-script audit found no
  current receding closed-loop wrapper under `scripts/world_model` or
  `scripts/slurm`; if a future checkpoint passes all gates, the next step is
  to implement or deliberately restore the PLAN-compliant compute-node wrapper
  before running live controller tests.
- Monitoring update at `2026-06-12 18:08 CST`: the same SFT step is still
  live and healthy, now at rank-0 iteration `1060`, with recent loss around
  `3.8-4.2` and no NaN/OOM/error marker in the inspected logs. Checkpoints
  remain `iter_000000300`, `iter_000000600`, and `iter_000000900`; the
  `iter_000001200` checkpoint has not been written yet. The strict eval
  watcher `126985.9` is correctly waiting inside the held `server40`
  allocation. Two prematurely launched readout/profile watcher tmux panes were
  producing `Requested nodes are busy` messages because they competed for the
  same 1-GPU auxiliary allocation before strict eval existed; they were
  interrupted with tmux `Ctrl-C` without cancelling the Slurm allocation.
  Readout/profile should be relaunched only after the iter1200 strict eval
  root exists.

Cleanup:

- active cosmos3 root now keeps only:
  the frozen `733` source, rendered SFT dataset, WAM condition root, SFT output
  root, v7 Complete9 review root, original 6/6 baseline/audit, and small
  approval/provenance files;
- `93` old process directories and `84` scattered logs/lists were moved, not
  deleted, to
  `/public/home/yanhongru/ICLR2027/archived/reflex_cosmos3_process_artifacts_after_fix3_v7_733_sft_20260612`.

Data-semantics warning:

- this `733`-row source is DP-success-filtered bootstrap data. The stored
  positive rollouts came from repeated dynamic-scene attempts where frozen DP
  still succeeded and the physical gates accepted the episode;
- replaying the stored H5 action/state trajectory should therefore succeed by
  construction. Re-running the same frozen DP online on the same seeds and
  dynamic scripts should be treated as an explicit baseline measurement, but
  the expected result on the accepted subset is near-ceiling success if the
  environment reset, physics, and stochasticity match the recorded run;
- this creates selection bias: the source distribution is
  `successful dynamic episodes under frozen DP`, not the unfiltered hard
  dynamic test distribution where frozen DP may fail;
- therefore this run can validate the full-episode Cosmos3 SFT chain and teach
  coherent target/peg/robot/action-state prediction on successful dynamic
  examples, but it cannot by itself prove improvement over DP on hard dynamic
  scenes where the original DP fails;
- expected gains from this run are component-level only: stable 301-frame RGB
  future prediction, 300-step action/sidecar prediction, target-motion/final-
  target readout, and a working SFT/eval interface. It should not be reported
  as a task-success gain over DP on this same accepted subset, because DP is
  already at the acceptance ceiling there;
- later method evidence must compare against unfiltered dynamic DP-only
  baselines and/or hard teacher/manual dynamic data where frozen DP has low
  success, followed by same-length generated-video/action/readout review and
  closed-loop evaluation.

Boundary:

- this note does not approve any shortcut, 128/129 chunking, 93-frame crop, or
  state-only/oracle method;
- SFT loss alone is not method evidence. Generated same-length validation
  videos, action/readout metrics, and later DP closed-loop evidence are still
  required.

2026-06-12 18:30 CST correction:

- User visual review indicated the current videos looked like the earlier
  "action adapter not effectively trained" failure. Config audit confirmed a
  real training mistake: the `normactive_clip1` run selected the action
  adapter tensors but did not use the fix1 action-training recipe. It used
  `lr=2e-5`, `action_loss_weight=10.0`,
  `independent_action_schedule=false`, and `shift_action=None`.
- The `normactive_clip1` SFT and its iter1200 watcher were interrupted inside
  tmux with `Ctrl-C`, preserving the Slurm allocations. No `scancel` was used.
- The base full-episode WAM SFT wrapper now defaults to and enforces the
  overfit-validated fix1 recipe: `lr=1e-4`, warmup `10`, `f_min=0.5`,
  `grad_clip_norm=1.0`, `action_loss_weight=2.0`,
  `normalize_loss_by_active=true`, `independent_action_schedule=true`,
  `shift_action=1`, and action-adapter optimizer keys included.
- A fresh two-sample overfit gate was started before any full-data retrain.
  Condition root:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_overfit2_rgb_300step_20260612_1830`.
  Selected rows are `hole_late_move_stop / target_motion_observed` and
  `peg_drop / peg_recovery`; train and val are identical. Preflight and
  action-target audit passed under the `301` RGB-frame / `300x32` action-state
  contract.
- Active overfit SFT root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_overfit2_rgb_300step_fix1recipe_4gpu_20260612_1840`.
  It runs on Slurm step `126210.40` with 4 H200 GPUs. Manifest and Hydra logs
  confirm `action_loss_weight=2.0`,
  `independent_action_schedule=True`, `shift_action=1`, optimizer `lr=0.0001`,
  and optimizer selection of `410` tensors / `6,982,401,216` elements.
  Iteration-0 validation loss is `3.393175`; early train iterations are finite.
- A strict iter100 eval watcher is active on auxiliary step `126985.10` with
  `N_EVAL_SAMPLES=2`. Full-data SFT is gated until the two generated overfit
  videos are produced and approved by the user.

2026-06-12 19:05 CST overfit gate update:

- The two-sample overfit run saved checkpoint `iter_000000100` at
  `2026-06-12 19:00:04 CST`. Iter100 validation loss was `0.125564`, down from
  `3.393175` at iter0 and `0.390711` at iter50. The 4-H200 foreground training
  step was then interrupted inside tmux with `Ctrl-C` after the checkpoint and
  eval launch, preserving the held Slurm allocation. No `scancel` was used.
- Strict iter100 eval completed for both overfit validation rows under the
  full-episode contract. `eval_artifact_inspection.json` reports
  `strict_eval_artifacts_ok=true`, `strict_failures=[]`, generated/reference
  videos `301/301`, actions `[300,32]`, mean action RMSE `0.1921146387`, mean
  robot-action future RMSE `0.3211359675`, mean state-sidecar future RMSE
  `0.2256914438`, and mean future-video PSNR `28.3125916850`.
- Generated videos for user visual approval:
  `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_overfit2_rgb_300step_fix1recipe_4gpu_20260612_1840/eval_full_episode_wam_iter_000000100/inference/00_peg_recovery_peg_drop_peg_drop_seed705095_idx0004.fix3_traj_0__peg_recovery_f131/vision.mp4`
  and
  `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_overfit2_rgb_300step_fix1recipe_4gpu_20260612_1840/eval_full_episode_wam_iter_000000100/inference/01_target_motion_observed_hole_late_move_stop_hole_late_move_stop_seed1080087_idx1760.fix3_traj_0__target_motion_observed_f117/vision.mp4`.
- Agent opened both ref/pred review sheets:
  `review_sheets/00_peg_recovery_peg_drop_peg_drop_seed705095_idx0004.fix3_traj_0__peg_recovery_f131_ref_pred_sheet.png`
  and
  `review_sheets/01_target_motion_observed_hole_late_move_stop_hole_late_move_stop_seed1080087_idx1760.fix3_traj_0__target_motion_observed_f117_ref_pred_sheet.png`.
  The sampled panels show close ref/pred alignment without obvious blank video,
  global geometry collapse, or large trajectory mismatch. This is overfit gate
  evidence only; full-data SFT remains blocked until user visual approval.
- User then confirmed the overfit videos pass visual review. The active next
  step is to commit/push code and documentation while excluding large
  environment/data/video/image/checkpoint artifacts, then start the full
  v7_733 SFT using the enforced fix1 action recipe as the default. Controller
  work remains gated until full-data checkpoints pass strict generated-video,
  action/readout, and visual review.

2026-06-12 19:30 CST full-data SFT restart:

- Code, plans, TODOs, and documentation were committed and pushed to
  `https://github.com/yanhr21/Reflex` on branch `main` as commit `1bd4691`
  (`Set Cosmos3 full-episode WAM training defaults`). The commit intentionally
  excludes local environments, `data/`, `experiments/`, `checkpoints/`,
  `logs/`, third-party checkouts, videos, images, H5 files, NumPy arrays, and
  model checkpoint binaries.
- The full v7_733 SFT was restarted from the frozen 733-row full-episode WAM
  condition root using the enforced fix1 recipe. Output root:
  `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`.
  It runs in tmux session
  `cosmos3_sft_v7_733_full_fix1recipe_4gpu_126210` on Slurm step `126210.41`
  (`server56`, `4xH200`).
- The first launch attempt at `20260612_191630` did not start training because
  the requested step memory/CPU exceeded the held allocation (`480G/64 CPU`
  requested versus `256G/32 CPU` available). The corrected launch uses
  `--mem=240G --cpus-per-task=32 --gres=gpu:4` within the same held allocation.
- Startup evidence: the manifest reports `fix1_action_recipe_check=passed`,
  `optimizer_lr=1.0e-4`, `action_loss_weight=2.0`,
  `normalize_loss_by_active=true`, `independent_action_schedule=true`,
  `shift_action=1`, `301` RGB/state frames, and `300` action/state rows.
  The optimizer selected `410` tensors / `6,982,401,216` elements at
  `lr=0.0001`.
- Training reached `Starting training...` at `19:20:30 CST`. Iter0 validation
  loss was `3.606580`; by rank-0 iteration 7 the loss was `3.0716` with
  `vision=0.0514` and `action=1.5101`. After the first validation step,
  iteration speed is about `17.3s`.
- The previous auxiliary allocation `126985` was cancelled by Slurm after the
  first watcher started; no `scancel` was used by the agent. A fresh tmux-held
  1-H200 auxiliary allocation `127120` was acquired on `server40`, and step
  `127120.0` is waiting for `iter_000000300` with strict eval settings
  (`N_EVAL_SAMPLES=10`, full `301/300` contract). Watch log:
  `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/eval_iter_000000300_watch_aux_realloc.log`.

2026-06-12 19:38 CST monitor update:

- Training remains live on Slurm step `126210.41`. Rank-0 loss has dropped
  quickly from iter0 validation `3.606580` to iteration 45 train loss `0.9703`
  (`vision=0.0544`, `action=0.4580`), with stable per-step time around
  `17.3s` after startup. No OOM, NaN, traceback, or SFT failure marker was
  observed in the inspected log.
- The iter300 eval watcher remains live on auxiliary job `127120`, step
  `127120.0`, and is correctly reporting `checkpoint_not_ready`; no
  `iter_000000300` checkpoint or eval artifacts exist yet.
- The generated-RGB readout checkpoint to use after strict eval is the existing
  v7_733 reference readout:
  `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_4gpu_20260612_0245/task_state_readout_reference_rgb_301f_v7_733/best_model.pt`.
  Do not run readout before strict eval finishes, to avoid competing with the
  single auxiliary GPU while inference is running.
