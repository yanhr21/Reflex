# WM Policy Distillation TODO

## Boundary

- [ ] 2026-06-09 pickup-aware chunked WAM handoff boundary: controller and
      distillation work must wait for the active Cosmos/DROID WAM to produce
      valid chunk evidence. The prior full301 short-prefix forward-dynamics
      SFT is superseded because it did not observe the full pickup/holding
      segment before prediction. The next controller input must come from
      `joint_policy` or paired `forward_dynamics` chunks that condition on
      current pickup-complete RGB/proprio/action history and predict a short
      future action/video/task-state chunk. Execution remains DDP-style:
      execute only a short prefix, re-observe target/peg/gripper state, and
      regenerate the next chunk. Do not train positive takeover data, run old
      full301 readout/controller watchers, or report DP integration until the
      chunk demos show target reconstruction, peg/gripper continuity, and
      action chunks over the exact generated length. The current controller
      gate is therefore visual/metric chunk evidence from the active
      Cosmos/DROID WAM, not another standalone controller or object-slot WM.
- [ ] 2026-06-08 DP target visibility and selected-WAM retention
      diagnostic: the latest hard-case evidence says not to diagnose the
      frozen state-DP as literally blind to the moved hole. In
      `wam_retention_penalty_small_rgbd_holeconstant_seed702000_no_video_20260608_1616`,
      `policy_obs[35:42]` matches metric `box_hole_pose` after the trigger
      (`0.000346m` position RMSE, final error `0.000000m`), while RGB-D slot
      hole/head drift remains nonzero (`0.017237m` and `0.019595m`
      post-trigger RMSE). Metrics fail (`success_once=false`,
      `success_at_end=false`, `final_grasped=true`, final metric
      peg-head-at-hole `[-0.126763, 0.020247, -0.048345]`). The approved-view
      1024px/30fps replay and keyframe sheet show normal pickup/hold and
      approach, but the peg stays outside the hole. Event audit found every
      selected rollout-MPC chunk was `dp_bridge_blend`; WM-policy action
      generator candidates were not selected. Strict selected-WAM runs
      `wam_retention_selected_rgbd_holeconstant_seed702000_no_video_20260608_1605`
      and
      `wam_retention_selected_tailok_rgbd_holeconstant_seed702000_no_video_20260608_1606`
      failed because no candidate survived once selected chunks were required
      to have available WAM future-state coverage. Boundary: negative
      diagnostic only, and the WAM scorer checkpoint is still non-method smoke
      (`method_source_count=0`). The controller work should therefore move
      toward a Cosmos3/WAM action interface: current RGB-D/proprio history plus
      target hole/peg/TCP future trajectory plus candidate action chunk ->
      predicted future task state/action score, with short execution and
      live re-observation. Do not use this failure to train positives or to
      resume standalone direct-head/object-slot WM paths.
      Implementation follow-up: added explicit terminal-held Cosmos target
      future support (`cosmos_task_state_allow_terminal_hold`,
      `cosmos_task_state_terminal_hold_max_frames`) to the controller and
      allocation wrapper, default off. This is a diagnostic WAM target-path
      extrapolation from the latest causal receding Cosmos row, not a live
      refresh claim. A no-simulator functional smoke verified frame-200
      future coverage changes from unavailable to available with terminal
      provenance. RGB-D hard-case run
      `wam_terminal_hold_selected_rgbd_holeconstant_seed702000_no_video_20260608_1645`
      then ran in allocation `118088` with strict selected-WAM required. It
      still failed (`success_once=false`, `success_at_end=false`, final grasp
      true), but WAM future availability is no longer the blocker:
      all `488` candidate WAM reports were available,
      `future_state_sequence_unavailable=0`, and terminal hold appeared in
      `300` candidate reports / `15` selected reports. Selected chunks remain
      all `dp_bridge_blend=192`; `75` WM-policy action-generator candidates
      were generated and none selected. Visual review confirms held/approached
      but not inserted. Next work should improve the DP-preserving
      WM-conditioned action generator/scorer objective so it can execute the
      Cosmos target path, instead of only making the target path available.
- [ ] 2026-06-08 DDP task-frame policy diagnostic:
      evidence note
      `docs/world_model_task_rebinding/2026-06-08_ddp_taskframe_directhead_dynamic_retention_diagnostic.md`
      records the current answer to the user DP/Cosmos question. The state-DP
      path is not simply blind to the moved hole because its observation
      contains current `box_hole_pose`; the missing capability is a causal
      Cosmos/RGB-D task-frame trajectory plus executable short action chunks
      and retention scoring. Updated
      `scripts/world_model/train_wm_conditioned_policy_distillation.py` to
      build 64D static partial takeover conditions from current/future
      TCP/peg/hole, peg-head-at-hole progress, relative task geometry, teacher
      action mean, and hole radius, and to guard direct-action-head training.
      The 3000-step frozen-base run
      `experiments/world_model_task_rebinding/wm_policy_distillation/train_ddp_static_taskframe_directhead_freeze_base_3000iter_20260608_1525`
      produced a checkpoint, but static eval still rejects the direct action
      head (`success_at_end=0.0` over `10` episodes). The diffusion path on the
      same checkpoint preserves the base DP in a small static smoke
      (`success_at_end=0.8` over `10` episodes). The dynamic hard-case
      diagnostic
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_diffusion_candidate_taskframe3000_holeconstant_defaultbridge_seed702000_no_video_20260608_1535`
      reached only temporary insertion (`success_once=true`, inserted frames
      `148..195`) and failed final success (`success_at_end=false`) despite
      preserved grasp. The approved-camera 1024px/30fps replay and manual
      review show a retention failure rather than the old gross peg lift/lost
      grasp failure. Event audit: WM-policy action-generator candidates were
      available, but selected chunks were DP/bridge variants, so this is not
      evidence that the learned controller solved the task. Keep this run out
      of positive takeover data. Next work: keep direct-head candidates
      blocked, keep the diffusion DP prior, and train/score selected short
      chunks with Cosmos/WAM future-task and final-retention objectives under
      live re-observation.
- [ ] 2026-06-08 WAM interface correction:
      do not keep training/evaluating a controller that treats Cosmos3 as only
      a 64D bridge-summary side channel. Web/literature check and local audit
      are recorded in
      `docs/world_model_task_rebinding/2026-06-08_wam_interface_literature_and_failure_audit.md`.
      Added `scripts/world_model/audit_wam_controller_interface.py` and ran it
      on
      `wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658`.
      Audit artifacts:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658/wam_interface_audit.json`
      and
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658/wam_interface_audit.md`.
      The audit shows the state DP input does contain current `box_hole_pose`
      in this rollout (`0.000346m` post-trigger hole-position RMSE against the
      metric hole pose), so the failure is not simply blindness to the current
      hole. Cosmos/readout has future object/task trajectory rows, but the
      policy H5 only exposes `(T,64)` snapshot conditions and no explicit
      temporal future trajectory/action-state condition. RGB-D slot drift is
      also large near insertion: final hole L2 `0.043888m` and final
      peg-head-at-hole L2 `0.038503m`. Next training work should build a WAM
      temporal condition dataset containing current RGB-D/proprio state,
      Cosmos future `hole/peg/tcp/peg_head_at_hole` trajectory, action chunk,
      and future metric/task-state labels. Failed no-insertion rollouts remain
      negative diagnostics, not positive takeover distillation data. Added
      `scripts/world_model/build_wam_temporal_condition_dataset.py` and
      exported the first structural dataset:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658/wam_temporal_condition_dataset_h16_a8.h5`.
      It has `86` samples with policy history `[2,43]`, current slot state
      `30D`, Cosmos future state `[16,27]`, action chunk `[8,7]`, and future
      metric state `[16,27]`; report:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658/wam_temporal_condition_dataset_h16_a8.json`.
      The source rollout remains negative (`inserted_any=false`), and the
      Cosmos imagined future versus executed future gap is large:
      peg-head-at-hole mean/max L2 `0.399634m/0.431472m`. This is training
      interface evidence, not controller success. Added
      `scripts/world_model/train_wam_action_state_scorer.py` for the corrected
      WAM interface:
      `policy_obs_history + current_slot_state + cosmos_future_state +
      action_chunk -> future_metric_state + final grasp/inserted`.
      The script rejects failed-only datasets by default and requires
      `--allow-structural-negative-smoke` for explicit non-evidence smoke.
      Dry-run guard passed, explicit smoke dry-run loaded the dataset
      (`feature_dim=604`, `target_dim=432`, train/val `69/17`), and a 50-step
      smoke in allocation `117750` wrote
      `experiments/world_model_task_rebinding/wam_action_state_scorer/structural_negative_smoke_50iter_20260608`.
      The state-prediction RMSE decreased, but this proves only that the
      WAM action-state scorer path is wired. Because the source labels have
      no insertion success, this is not positive data and not controller
      evidence. Follow-up action-prior WAM export and mixed scorer smoke:
      added `scripts/world_model/build_wam_action_prior_temporal_dataset.py`
      and exported
      `experiments/world_model_task_rebinding/wm_policy_distillation/wam_action_prior_temporal_dataset_h16_a8_20260608.h5`
      plus JSON. It includes `19` visually reviewed successful DP
      action-prior groups (`997` samples) and excludes `2` non-candidate
      groups. The future condition is
      `metric_future_as_desired_action_prior`, stored with
      `candidate_action_prior_teacher_ok=true` but
      `positive_takeover_teacher_ok=false` and
      `method_evidence_allowed=false`; this prevents treating successful
      DP-only chunks as Cosmos/RGB-D takeover success. Updated
      `scripts/world_model/train_wam_action_state_scorer.py` so action-prior
      chunks require `--allow-action-prior-positive-smoke`; default admission
      still rejects any dataset without real positive takeover sources. A
      mixed action-prior plus failed-Cosmos 300-step scorer smoke ran inside
      held allocation `117750`:
      `experiments/world_model_task_rebinding/wam_action_state_scorer/action_prior_plus_failed_contrastive_smoke_300iter_20260608`.
      It loads `1083` samples at feature dim `604` and target dim `432`.
      Train/val state RMSE dropped from `0.406120/0.406248` to
      `0.050978/0.050628`, and final inserted/grasp accuracy reached
      `1.0/1.0`. Boundary remains non-method: manifest has
      `positive_source_count=0`, `action_prior_positive_smoke_source_count=1`,
      `method_source_count=0`. The causal takeaway is that state-DP is not
      simply blind to the moved hole; its observation contains the current
      hole pose in the audited rollout. The missing controller mechanism is a
      causal Cosmos/RGB-D future task-frame trajectory plus an action-state
      scorer/policy that selects executable short chunks, then re-observes.
      Runtime hook follow-up: added optional
      `--wam-scorer-checkpoint` support to
      `scripts/world_model/evaluate_rebinding_controller.py` for
      `control_policy=wm_dp_prior_rollout_mpc`. The hook loads the WAM
      action-state scorer, checks manifest identity, requires
      `--wam-scorer-allow-nonmethod-smoke` for the current action-prior smoke
      checkpoint, and adds an optional candidate-ranking term based on
      predicted head trajectory, inserted probability, and grasp probability.
      `CosmosTaskStateTrajectoryPredictor` now exposes
      `future_state_sequence(...)` so controller-time scoring can consume a
      temporal `[16,27]` task-state future rather than the old 64D snapshot.
      A local runtime probe on the failed partial-takeover frame `90` loaded
      the scorer and receding Cosmos future successfully; it predicted very
      low insertion probability (`~7e-06`) and high grasp probability
      (`0.999912`), matching the physical failure. A RGB-D runtime rollout on
      held allocation `117750`/`server21` failed before controller evaluation
      with Vulkan `ErrorDeviceLost` while constructing the RGB-D slot
      `vision_env`; this is a rendering/perception environment failure, not a
      controller result. A no-video oracle-slot code-path smoke succeeded
      metrically:
      `experiments/world_model_task_rebinding/rebinding_controller/wam_scorer_rollout_mpc_oracle_seed702000_no_video_20260608`
      has `success_at_end=true`, inserted frames `167..300`, final
      peg-head-at-hole `[0.009820, 0.001172, 0.003026]`, and `2543` WAM scorer
      reports (`435` available future-state reports). Boundary: this only
      proves the WAM scorer can enter rollout-MPC candidate ranking in an
      oracle scaffold. It is not RGB-D method evidence, not visual evidence,
      and not positive Cosmos/RGB-D takeover distillation data. RGB-D
      follow-up ran on held allocation `118088`/`server44` after RGB-D
      vision-env preflight passed there:
      `experiments/world_model_task_rebinding/rebinding_controller/wam_scorer_rollout_mpc_rgbd_seed702000_no_video_20260608_server44`.
      It used `slot_source=rgbd`, the full RGB-D slot ensemble, the receding
      Cosmos task-state trajectory, and the mixed action-prior/failed-Cosmos
      WAM scorer checkpoint. Metrics failed (`success_once=false`,
      `success_at_end=false`, inserted frames `0`) while preserving grasp
      through the end. Final metric peg-head-at-hole was
      `[-0.112917, 0.001173, -0.007083]` with head L2 `0.113145m`. Visual
      review of the full 1024px/30fps replay
      `experiments/world_model_task_rebinding/rebinding_controller/wam_scorer_rollout_mpc_rgbd_seed702000_no_video_20260608_server44/visual_review_envstate_default_view_1024_full301_30fps_20260608/state_replay.mp4`
      and metric keyframe sheet confirms normal pickup/hold and approach to
      the box, but no visible insertion. RGB-D drift around the post-move
      frames is large: hole-position max L2 `0.116639m`,
      peg-head-at-hole max L2 `0.129873m`, and frame-100 hole-position error
      `0.101763m`. WAM scorer reports were produced (`1172` candidate reports,
      `437` available future-state reports), but selected candidates still
      behaved as DP/bridge approaches rather than executable insertion chunks.
      Boundary: negative RGB-D/Cosmos/WAM-scorer controller diagnostic, not
      positive takeover data. The next controller work must stabilize the
      RGB-D-derived target state after the dynamic event and make WAM
      scoring/policy generation operate on selected short chunks with a live
      re-observation loop. Follow-up selected-WAM logging was added to
      `scripts/world_model/evaluate_rebinding_controller.py` so future
      reports include the WAM scorer report for the actually selected
      rollout-MPC chunk. A default `hole_move_stop` diagnostic
      `wam_scorer_rollout_mpc_rgbd_seed702000_selectedlog_no_video_20260608_server44`
      succeeded (`success_once=true`, `success_at_end=true`, inserted frames
      `130..300`, final head L2 `0.010951m`) and its 1024px/30fps replay
      visibly shows pickup, insertion, and retention. Boundary: it used
      default bridge settings and `scenario=hole_move_stop`, so it is a
      positive easier-setting diagnostic only, not proof that `hole_constant`
      is solved. The exact `hole_constant` selected-log reproduction
      `wam_scorer_rollout_mpc_rgbd_holeconstant_seed702000_selectedlog_no_video_20260608_server44`
      failed again (`success_once=false`, `success_at_end=false`, inserted
      frames `0`, final grasp true, final head L2 `0.113145m`). Selected-WAM
      audit: selected candidate kinds were `dp_bridge_blend=184` and
      `bridge_variant=16`; only `9` selected plans had available selected-WAM
      scoring, `15` were unavailable because the future sequence was
      unavailable, and `176` had no selected-WAM dict. This confirms that the
      current hard-case controller is not yet consistently using Cosmos/WAM to
      constrain the actual executed short chunk.
- [ ] 2026-06-08 WM-policy admission and direct-head static preservation:
      readiness artifact discovery now recognizes current `visual_review*`
      replay videos/contact sheets as well as legacy `videos/`, so admission
      no longer depends on stale path layout. The RGB-D default DP-fallback
      failure is still rejected after the fix for real reasons
      (`metric_or_slot_drift_check_failed`, `manual_visual_review_not_positive`);
      the oracle success scaffold is recognized as scaffold-only and still not
      positive takeover training data. Added `--use-direct-action-head` /
      `USE_DIRECT_ACTION_HEAD` to static preservation evaluation. On held
      allocation `117200`, checkpoint
      `train_ddp_direct_action_head_21chunk_insertwindow_staticpartial25_freeze_base_24000iter_prior55_20260608_0025/checkpoints/final.pt`
      has diffusion-path static smoke `success_at_end=0.6` with zero WM
      condition, but direct-head static smoke `success_at_end=0.0`. Boundary:
      the direct action head is not a safe independent takeover controller.
      A controller guard now blocks direct-head rollout candidates unless a
      direct-head static preservation report clears the requested floor
      (default `success_at_end>=0.5`); the current failed direct-head report is
      rejected by that guard. Retrain the head with a stronger
      DP-preservation objective before using it as direct takeover machinery.
- [ ] 2026-06-08 RGB-D/Cosmos WM-policy partial takeover visual check:
      diagnostic run
      `wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658`
      used the RGB-D slot ensemble, receding Cosmos trajectory, and
      WM-policy diffusion path with action blend `0.35`; direct-head
      candidates were disabled. The 1024px/30fps/301-frame replay is:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Keyframe sheet:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658/visual_review_envstate_default_view_1024_full301_30fps/keyframe_contact_sheet_pickup_to_final_512.png`.
      Manual visual review:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658/visual_review_envstate_default_view_1024_full301_30fps/manual_visual_review.json`.
      Metrics and visuals agree: `success_once=false`, `success_at_end=false`,
      inserted frames `0`, final grasp true, metric grasped frames `54..300`,
      peg z range about `0.024m..0.126m`, and final peg-head-at-hole
      `[-0.136800, 0.031347, -0.017658]`. The peg does not fly up or flip
      skyward and remains near/in the gripper during late approach, but it
      stays outside the hole. Boundary: this is a negative controller
      diagnostic. It shows the diffusion-path partial takeover is not
      destroying pickup on this seed, but it still cannot be admitted as
      positive takeover distillation data because insertion fails visibly and
      metrically.
- [ ] 2026-06-08 default RGB-D/Cosmos DP-fallback visual check:
      diagnostic run
      `wm_dp_prior_rollout_mpc_rgbd_dpfallback_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0605`
      used `slot_source=rgbd`, `cosmos_task_state_min_tau=0`, and
      `controller_infeasible_fallback_policy=dp_prior` under held allocation
      `117200` on `server13`. It failed task completion
      (`success_once=false`, `success_at_end=false`, inserted frames `0`) but
      preserved basic pickup and avoided the old lift-away failure:
      `retreat_count=0`, `infeasible_dp_fallback_count=10`, metric grasped
      frames `54..300`, final metric peg-head-at-hole
      `[-0.111758, 0.001683, -0.010846]`, and metric peg z range about
      `0.024m..0.148m`. Full video:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_dpfallback_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0605/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Manual review:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_dpfallback_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0605/manual_visual_review.json`.
      Boundary: this is a negative RGB-D/Cosmos controller diagnostic and
      DP-preservation check; it must not enter positive takeover distillation
      data.
- [ ] 2026-06-08 DP-prior fallback diagnostic:
      added `controller_infeasible_fallback_policy=dp_prior` so infeasible
      RGB-D/Cosmos bridge decisions return to the frozen DP action instead of
      the old upward `safe_retreat`. Diagnostic run
      `wm_dp_prior_rollout_mpc_rgbd_mintau4_dpfallback_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0555`
      used `slot_source=rgbd` and `cosmos_task_state_min_tau=4` under held
      allocation `117200` on `server13`. It failed task completion
      (`success_once=false`, `success_at_end=false`, inserted frames `0`) but
      fixed the previous lift-away failure: `retreat_count=0`,
      `infeasible_dp_fallback_count=122`, metric grasped frames `54..300`,
      final metric peg z about `0.113m`, and final metric peg-head-at-hole
      `[-0.110683, -0.041798, -0.025048]`. Full video:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_mintau4_dpfallback_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0555/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Manual review:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_mintau4_dpfallback_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0555/manual_visual_review.json`.
      Boundary: this is a DP-preservation fix and negative controller
      diagnostic; it must not enter positive takeover distillation data.
- [ ] 2026-06-08 RGB-D min-tau visual failure:
      the diagnostic run
      `wm_dp_prior_rollout_mpc_rgbd_mintau4_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0530`
      used `slot_source=rgbd`, `cosmos_task_state_min_tau=4`, and
      `cosmos_task_state_reject_unavailable=false` under tmux/salloc
      allocation `117200` on `server13`. It failed with
      `success_at_end=false`, `success_once=false`, inserted frames `0`,
      metric grasped frames `54..300`, `infeasible_count=58`,
      `retreat_count=58`, final metric peg-head-at-hole
      `[-0.142177, 0.088691, 0.478954]`, and final metric peg z about
      `0.585m`. The full 1024px/30fps replay and inspected keyframe sheet show
      normal pickup and early approach, then a visible lift-away failure from
      about frame `210`. Full video:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_mintau4_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0530/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Manual review:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_mintau4_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0530/manual_visual_review.json`.
      Boundary: this is negative RGB-D/Cosmos controller failure localization
      and must not enter positive takeover distillation data.
- [ ] 2026-06-08 rollout-MPC hold success visual check:
      the run
      `wm_dp_prior_rollout_mpc_episode_restore_localaxis_successrollout8x4_lightpost_completion_axis12_seed702000_no_video_20260608_0414`
      is the first visually inspected single-seed controller scaffold in this
      sequence with `success_at_end=true`. Metrics: `success_once=true`,
      final grasp true, inserted frames `167..300` (`134`), grasped frames
      `54..300`, and final peg-head-at-hole
      `[-0.001794, 0.000714, -0.002903]`. Render job `117052` produced the
      1024px/30fps/301-frame replay on `server13`; inspected frames `54`,
      `90`, `167`, `230`, and `300` show normal pickup, stable grasp, no
      peg fly-up or gross orientation failure, visible insertion, and final
      retention. Full video:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_episode_restore_localaxis_successrollout8x4_lightpost_completion_axis12_seed702000_no_video_20260608_0414/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Manual review:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_episode_restore_localaxis_successrollout8x4_lightpost_completion_axis12_seed702000_no_video_20260608_0414/manual_visual_review.json`.
      Boundary: this is useful oracle-slot controller localization for the
      short-chunk DDP-style retention design, but it is not RGB-D/Cosmos method
      evidence and must not be admitted as positive takeover distillation data
      unless a future RGB-D/Cosmos-driven run proves the same behavior with
      video review.
- [ ] 2026-06-08 RGB-D slot smoke after oracle scaffold success:
      the same short-chunk rollout-MPC hold configuration was rerun with
      `slot_source=rgbd` and RGB-D slot ensemble
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job102292`
      under job `117075`. It failed: `success_at_end=false`,
      `success_once=false`, inserted frames `0`, final grasp true, final
      peg-head-at-hole `[-0.111166, 0.000307, -0.010842]`. Render job `117169`
      produced the 1024px/30fps/301-frame replay:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_successrollout8x4_lightpost_completion_axis12_seed702000_no_video_20260608_0435/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Manual review confirms normal pickup and stable grasp with no peg
      fly-up or gross orientation failure, but no visible insertion. This is
      negative RGB-D-derived controller evidence and must not be used as
      positive takeover distillation data. The immediate bottleneck is
      RGB-D slot/controller drift: post-trigger RMSE is peg-head-at-hole
      `0.03175m`, hole position `0.02421m`, and peg position `0.01523m`.
- [ ] 2026-06-08 latest DDP-style controller visual/hold result:
      the newest rendered 1024px state replays show normal peg pickup and no
      peg fly-up. The local-axis rollout can transiently insert from frame
      `181`, but final success still fails. Post-success diagnostics show the
      frozen DP handoff pulls the peg out, zero hold is worse, 8mm servo is
      worse than 4mm, and deepest-target servo is the current near miss
      (`success_once=true`, `success_at_end=false`, final peg-head-at-hole
      `[-0.015494, -0.001827, -0.001792]`, final grasp true). This is not
      positive takeover teacher data and must not enter distillation as a
      success source. Evidence:
      `docs/world_model_task_rebinding/2026-06-08_controller_visual_and_hold_failure_localization.md`.
- [ ] 2026-06-08 insert-x hold visual check:
      render job `116967` produced the full 1024px/30fps replay for
      `wm_dp_prior_rollout_mpc_episode_restore_localaxis_successservo_insertx0_completion_axis12_seed702000_no_video_20260608_0335`.
      The peg is visibly picked up normally and does not fly up or tilt to the
      sky, but final success is still false (`success_once=true`,
      `success_at_end=false`, inserted frames `181..294`, final grasp true,
      final peg-head-at-hole `[-0.017480, 0.002951, 0.003044]`). The extra
      insert-x hold target increases inserted-frame count but does not solve
      final contact retention, and it must not be used as positive takeover
      distillation data.
- [ ] Do not use failed geometric-controller rollouts as positive takeover
      training data. The 2026-06-06
      `cosmos_receding_teacher_forced_contact_preserve_20260606_132036`
      video is a negative diagnostic: the robot approaches the moved hole but
      does not visibly hold/insert the peg.
- [ ] Positive takeover distillation requires both metric success and inspected
      video/contact-sheet evidence that the peg is held through the intended
      world-model-conditioned motion.
- [ ] Preserve official/static DP behavior. The learned controller policy must
      be evaluated on static DP-style task completion before dynamic claims.
- [ ] Cosmos/controller conditioning must be closed-loop at controller time.
      The current receding retest used later teacher-forced segments and is
      better than the old frame-0 one-shot diagnostic, but it still merged
      segment predictions into a precomputed trajectory. The next aligned
      controller interface must refresh from the real observed rollout after
      object motion starts, predict a short horizon, execute one or a few
      actions, then teacher-force again from newly observed frames before the
      next prediction. A stale pre-motion prefix or a long precomputed
      trajectory must not be treated as the final world-model/controller
      method.
- [ ] 2026-06-06 Dream Diffusion Policy correction:
      do not reduce the controller to waiting for positive distillation data
      or hand-written bridge thresholds. Added `control_policy=wm_dp_prior_mpc`
      as a diagnostic DDP-style executor: Cosmos/world-model predicted
      task-frame trajectory scores short-horizon candidates around the frozen
      DP action prior. Three no-video oracle-slot diagnostics under allocation
      `111343` all failed final success, so this is not method evidence:
      conservative prior reached minimum yz `0.0165m` but ended at
      `[-0.11067,-0.05019,-0.01131]`; aggressive blend collapsed toward
      geometry and worsened; early handoff used 18 DP handoff steps but still
      ended at `[-0.11068,-0.03711,-0.01196]`. Conclusion: minimal DP-prior MPC
      is insufficient. Added `control_policy=wm_dp_prior_sequence_mpc` as a
      stronger DDP-style test-time planning diagnostic that scores short DP
      action chunks against the Cosmos/world-model task frame. One no-video
      oracle-slot smoke under `111343` also failed final success but localized
      the failure: final grasp remained true and lateral yz reached
      `0.00011m`, yet insertion x only reached `-0.09875m`. A follow-up
      phase-scoring fix was worse (`final peg_head_at_hole=
      [-0.19478,0.01128,-0.01956]`) and was reverted. Conclusion: DDP-style
      action-prior control is not blocked at lateral alignment; the missing
      piece is contact/insertion-axis execution from a stronger teacher,
      trajectory optimizer, or learned policy, not more threshold-only tuning.
      Evidence:
      `docs/world_model_task_rebinding/2026-06-06_ddp_style_controller_and_motion_planner_probe.md`.
- [ ] 2026-06-07 DDP-style hybrid controller update:
      implemented and tested concrete DDP-style alternatives instead of
      waiting on pure distillation. Added
      `control_policy=wm_dp_prior_rollout_mpc`, which scores candidate
      DP/bridge chunks in a separate simulator planning env before executing
      the selected chunk. On seed `702000` it still failed final success
      (`success_once=false`, `success_at_end=false`, max insertion x only
      `-0.111066`), so the current DP/bridge blend candidate family is not a
      sufficient insertion executor. Searched from the failed distillation
      rollout's near-hole frames and found 6 successful DP-action insertion
      teacher chunks in
      `teacher_search_from_distill80_failure_seed702000_20260607.h5`
      (`299` actions, final grasp true). Training a frozen-base 160-step
      checkpoint with those chunks preserved static smoke performance
      (`success_at_end=0.6`) but still failed dynamic final success. Added a
      hybrid controller switch
      `--wm-policy-takeover-bridge-on-teacher-support` plus bridge latch. The
      best oracle/scaffold diagnostic reached transient success for 19 frames
      but not final success:
      `wm_policy_takeover_hybrid_teacher_support_bridge_latch70_close006_successholdfix_insert6_160iter_seed702000_diag_20260607_continue`
      has `success_once=true`, `success_at_end=false`, success frames
      `267..285`, final peg-head-at-hole
      `[-0.023310, 0.003024, 0.003008]`, final grasp false. DP, zero, and
      servo post-success holds all failed final retention because the peg was
      already not grasped during the success window. Current bottleneck:
      grasp-preserving insertion/retention, not absence of a local insertion
      action chunk. Action comparison shows the online bridge latch is far
      outside the successful teacher action distribution: first-12 mean xyz
      action is `[0.0224,0.0805,0.0390]` for the teacher but
      `[-0.0231,0.5404,-0.0933]` online, with the same `-1.0` gripper command.
      Added teacher-action replay and local-delta replay diagnostics. Both
      failed. A new exact-failure-state teacher search found 4 successful
      chunks from the replay failure states
      (`teacher_search_from_local_delta_replay_failure_seed702000_20260607.h5`,
      final grasp true), but direct replay of the selected
      `bridge_phase_hybrid_tcp_continuation` chunk still failed
      (`success_once=false`, `success_at_end=false`, final grasp false).
      TCP-local-delta replay preserved final grasp but also failed insertion
      (`success_once=false`, final
      `[-0.145975,0.035818,-0.043910]`). Conclusion: raw action replay and
      TCP-only local-delta replay are not robust to the online robot/TCP state;
      the next controller should follow a multi-state closed-loop teacher
      trajectory over TCP, peg body/head, and orientation, or use an online
      trajectory optimizer from the current state. Follow-up `state_follower`
      replay executed 56 closed-loop teacher-trajectory steps and preserved
      final grasp, but still failed insertion (`success_once=false`, final
      `[-0.119097,0.037095,0.010557]`, max insertion x `-0.108954`).
      Rollout-MPC was extended to score teacher action/local-delta/TCP-delta/
      state-follower candidates before executing chunks. Chunk-4 selected
      teacher-derived candidates for 32/210 steps but failed; chunk-16
      selected only DP/bridge candidates and failed. The active replacement
      direction is now optional MPPI-style sampled action-sequence candidates
      around the teacher chunk, scored by the planner env. Allocation `114164`
      on `server27` passed GPU preflight. MPPI selected sampled action
      sequences for 50/210 steps but failed final success. Added online
      bridge/local-axis variant candidates from `search_dp_insertion_teacher.py`;
      chunk32 selected bridge/local-axis variants for most steps but failed,
      and no simulated online candidate reached `inserted_any=true`. Offline
      search from that exact failed H5 found a recoverable state at frame 219:
      `bridge_phase_hybrid_peg_alignment` succeeded in 47 steps with final
      grasp true. Chunk64 online planning changed the earlier trajectory and
      still failed; fixed recovered-teacher replay triggered too early and
      failed. Conclusion: direct replay and pure distillation are exhausted
      diagnostics for this seed; the next aligned controller is online
      primitive search/optimization from the actual current state followed by
      immediate execution. Follow-up two-stage rollout-MPC debugging added
      support-gated primitive search, decoupled primitive horizon from the
      DP 8-step action horizon, applied future dynamic hole perturbations
      inside planner rollouts, and checked simulated insertion after the next
      perturbation. The planner can now select a simulated inserting chunk,
      but the real rollout still does not insert even though the selected
      planned actions and executed real actions match exactly (`diff_max=0.0`).
      Offline search from the same failed H5 still finds successful
      frame-135/136 primitives. Current bottleneck is planner-env fidelity or
      state-restoration/timing. Added
      `scripts/world_model/verify_planned_chunk_replay.py`; on the step-130
      selected chunk from
      `wm_dp_prior_rollout_mpc_two_stage_future_perturb_postcheck_seed702000_diag_20260607_continue`,
      replay also failed with `action_diff_max=0.0` and
      `planner_after_vs_replay_head_rmse=0.041394m`. Planner success must not
      be treated as controller evidence until selected chunks are
      replay-verified or planner/replay dynamics are unified.
      Follow-up DDP paper check and controller correction: DDP's relevant
      execution pattern is short action chunking plus repeated
      real-imagination alignment, not more pure takeover distillation or long
      open-loop video. Added
      `--wm-dp-prior-rollout-execute-chunk` default `8`, so rollout-MPC scores
      a longer imagined horizon but executes only a short live prefix before
      re-observing and replanning. Three allocation-`114164` diagnostics on
      seed `702000` all failed and are negative scaffold evidence:
      long-open conservative (`final [-0.086633,-0.001576,-0.000618]`,
      max x `-0.046175`, final grasp true), receding-8
      (`final [-0.113631,0.002675,-0.001677]`, max x `-0.048891`,
      final grasp true), and receding-8 with non-inserting primitive fallback
      (`final [-0.203388,0.037057,-0.066889]`, max x `-0.111281`,
      final grasp true). A step-210 verifier on the conservative run showed
      reset+`set_state_dict` replay succeeds and matches the planner
      (`replay_success_once=true`, `planner_after_vs_replay_head_rmse=0.0`)
      while the original live window with identical actions fails
      (`action_diff_max=0.0`, `replay_vs_original_head_rmse=0.012168m`).
      Conclusion: restored planner contact state is not live contact state,
      and local-axis fallback is worse. Do not continue tuning primitive gates
      or wait on pure distillation; the next controller must use a DDP-style
      short-chunk action generator from DP/expert/Cosmos-conditioned behavior
      with live re-observation as the authority. Added a guided-DP residual
      candidate family to test the simplest DP-preserving version of that
      idea: keep the frozen DP action chunk and mix in a small task-frame
      residual. The isolated run
      `wm_dp_prior_rollout_mpc_receding8_guided_dp_residual_seed702000_diag_20260607_continue`
      failed (`success_once=false`, final
      `[-0.123001,0.012271,-0.045794]`, max x `-0.110716`,
      final grasp true); all 12 guided candidates were non-inserting and were
      rejected, so selected actions remained `dp_bridge_blend`. Do not keep
      tuning residual gains as a substitute for positive takeover data or a
      stronger action model. Video follow-up
      `wm_dp_prior_rollout_mpc_receding8_guided_dp_residual_seed702000_video_20260607_continue`
      confirms the visible outcome: the peg remains outside the hole in the
      final frame, and `manual_visual_review.json` marks
      `positive_takeover_teacher_ok=false`, so this rollout is negative
      diagnostic evidence and must not enter positive takeover distillation.
      A small DP-only live action-prior scan
      `dp_only_dynamic_teacher_scan_seed702200_10eps_20260607_continue`
      then found sparse frozen-DP dynamic successes (`success_at_end=0.2`,
      `success_once=0.4`). Video follow-up admitted only `702204` as a
      DP-action-prior candidate: it visually holds and inserts, and its review
      marks `candidate_dp_action_teacher_ok=true` but
      `positive_takeover_teacher_ok=false`. Candidate `702207` failed video
      rerun and is rejected. This provides one live-executable DP action chunk
      source to analyze, but still no admitted WM-controller takeover teacher.
      Added `scripts/world_model/extract_dp_action_prior_chunks.py` and wrote
      `dp_only_seed702204_action_prior_chunks.h5` under
      `experiments/world_model_task_rebinding/wm_policy_distillation/dp_live_action_prior_chunks_20260607/`.
      It contains one `dp_only_live_action_prior` chunk with `192` actions,
      `policy_obs (193,43)`, `wm_policy_condition (193,64)`, and separate
      `metric_slots`; it is compatible with `TeacherActionChunkLibrary` but
      remains explicitly `dp_action_prior_candidate_not_wm_takeover`.
      Distillation manifests must not treat it as `positive_takeover_sources`
      unless a future controller run proves WM-conditioned takeover using it
      with video review. Added explicit controller args
      `--wm-policy-action-prior-h5` and
      `--wm-policy-action-prior-variant` to load such chunks as
      `action_source_role=dp_or_expert_action_prior`, avoiding ambiguous
      teacher-source provenance.
      Evidence:
      `docs/world_model_task_rebinding/2026-06-07_ddp_style_hybrid_controller_probe.md`.
- [ ] 2026-06-07 allocation-114496 action-prior controller follow-up:
      continued the Dream Diffusion Policy-inspired path inside held
      allocation `114496` on `server34`. The concrete DDP lesson used here is
      short action chunks plus repeated live re-observation, not long
      open-loop replay or waiting for pure distillation. A 30-episode DP-only
      action-prior scan on seeds `702210..702239` found `10/30`
      no-video final successes, but video reruns accepted only seed `702210`
      as an action-prior candidate with final grasp true. Seed `702237`
      reproduced metric insertion but had `final_grasped=false`; its manual
      review is now marked `candidate_dp_action_teacher_ok=false` and
      `rejected_after_metric_grasp_audit=true`. Extracted libraries now
      include DP-only chunks `702204` and `702210`, plus two converted
      official-planner chunks in
      `dp_official_4chunk_action_prior_library.h5`. Controller probes with
      one-, two-, and four-chunk libraries all failed on seed `702000`.
      Strict filtering selected only `dp_bridge_blend`; allowing
      non-inserting chunks selected a few teacher follower candidates but
      made the final state worse. No action-prior candidate reached
      `inserted_any=true` in the online planner records. Conclusion:
      fixed action-prior replay is exhausted as the controller mechanism.
      Added `scripts/world_model/build_ddp_action_generator_manifest.py` and
      generated
      `experiments/world_model_task_rebinding/wm_policy_distillation/ddp_action_generator_manifest_20260607_0830/manifest.json`;
      it accepts 4 action-prior chunks, records the empty/rejected `702237`
      H5 as rejected, and marks the path
      `ready_for_ddp_action_generator_smoke_not_method_evidence`.
      The next aligned controller must be a DDP-style short-horizon learned or
      optimized action generator conditioned on Cosmos/RGB-D task state plus
      current robot/object state, with live re-observation after every short
      chunk. Do not spend more allocation time tuning fixed replay gates or
      pure takeover distillation unless it is a code-path preflight for that
      generator/optimizer.
- [ ] 2026-06-07 action-generator follow-up scheduling/data:
      used the end of allocation `114496` for DP-only action-prior candidate
      scan
      `dp_only_dynamic_action_prior_scan_seed702240_12eps_20260607_0835`.
      It completed before the 3-hour walltime with `success_at_end=0.25`
      (`3/12`) and `success_once=0.4166667` (`5/12`). Final-success seeds
      needing video rerun are `702240`, `702242`, and `702245`; transient-only
      seeds include `702241` and `702243`, with `702243` ending
      `final_grasped=false`. This is DP-only action-prior candidate search,
      not WM-controller evidence. Allocation `114496` then ended by walltime,
      not manual release. New tmux session
      `wm_action_generator_h200_24h_20260607_084948` requested one H200 for
      24h as job `114529`; it is pending for `Priority`. Test-only probes for
      3h/6h/12h/24h all forecast `2026-06-08T04:43:37` on `server34`, so no
      shorter duplicate allocation was submitted.
- [ ] 2026-06-07 action-generator smoke result:
      job `114529` started on `server34` and is being held for follow-up
      controller work. Video reruns for DP-only candidate seeds `702240`,
      `702242`, and `702245` all reproduced metric final success with final
      grasp true and passed manual contact-sheet review as action-prior
      candidates only. Extracted chunks:
      `dp_only_seed702240_action_prior_chunks.h5`,
      `dp_only_seed702242_action_prior_chunks.h5`, and
      `dp_only_seed702245_action_prior_chunks.h5`. Merged
      `dp_official_7chunk_action_prior_library.h5` and rebuilt
      `ddp_action_generator_manifest_20260607_0900/manifest.json`
      (`7` accepted action-prior chunks, `1` rejected source,
      `ready_for_ddp_action_generator_smoke_not_method_evidence`). Patched
      `train_wm_conditioned_policy_distillation.py` with explicit
      `--allow-action-prior-smoke`, requiring `--freeze-base-policy` and
      refusing to mix this with full positive takeover training. Dry-run saw
      `base_samples=152291`, `takeover/action_prior_samples=7798`,
      `positive_source_count=0`, `action_prior_source_count=7`. A 1000-iter
      frozen-backbone smoke
      `train_ddp_action_generator_7chunk_freeze_base_1000iter_20260607_0910`
      completed; static preservation 10eps was `success_at_end=0.6`,
      `success_once=0.6`. Dynamic seed `702000` no-video tests still failed:
      blend `1.0` ended at
      `[-0.217053,0.192513,0.001095]` with min yz `0.102853`; blend `0.35`
      improved lateral error but still failed, final
      `[-0.139404,0.048157,-0.023760]`, min yz `0.028926`, final grasp true,
      `inserted_any=false`. Failure localization: learned action-generator
      outputs over-large lateral/vertical moves compared with successful
      chunks. This proves the action-generator training path runs but is not a
      working controller yet. Next fix should make the generator/optimizer
      explicitly state-conditioned over current peg/hole/TCP geometry, not
      just a global adapter/blend.
- [ ] 2026-06-07 learned generator inside rollout-MPC follow-up:
      implemented the concrete DDP-style controller integration rather than
      waiting on pure distillation. `WMPolicyTakeover` now exposes full action
      sequences, and `wm_dp_prior_rollout_mpc` can score
      `wm_policy_action_generator` candidates plus MPPI samples around the
      learned generator. Wrapper knobs were added for
      `WM_DP_PRIOR_ROLLOUT_INCLUDE_WM_POLICY_CANDIDATES`,
      `WM_DP_PRIOR_ROLLOUT_WM_POLICY_BLENDS`,
      `WM_DP_PRIOR_ROLLOUT_NOISE_CANDIDATES`,
      `WM_DP_PRIOR_ROLLOUT_NOISE_STD`, and
      `WM_DP_PRIOR_ROLLOUT_NOISE_GRIPPER`. No-video seed `702000` results are
      negative scaffold evidence: direct learned-generator MPC selected the
      learned generator for 48 steps but had no simulated inserted candidates
      and failed final success; MPPI around DP/learned bases selected sampled
      candidates for 166 steps and found 15 simulated inserted candidates, but
      live rollout still failed. The one-action execution-prefix MPPI variant
      was cancelled as too slow after writing only `manifest.json`, while
      keeping allocation `114529` alive. Current diagnosis: planner/live
      contact-state mismatch or insertion-execution fidelity is now the
      bottleneck; pure distillation, fixed replay, and larger MPPI candidate
      sets are not the next answer.

## Data

- [ ] Build a distillation manifest with two source families:
      official/static DP demos for `L_base_dp_bc`, and successful dynamic
      takeover teacher rollouts for `L_takeover_bc`.
- [x] Add an explicit scaffold-only admission boundary so old state/CV/oracle
      successes cannot silently become positive RGB-D/Cosmos teacher data.
      `95107` is visually successful and useful for studying action structure,
      but it is recorded only under `scaffold_takeover_sources`; it remains
      rejected as positive training data because it uses CV/state scaffold
      provenance and lacks separate `metric_slots`.
- [ ] Store world-model predicted short trajectories as causal conditioning:
      no future ground-truth object poses may be used as privileged inputs.
- [x] Patch future controller rollouts to save the policy-training contract
      instead of trying to synthesize it later. `evaluate_rebinding_controller.py`
      now has `--save-policy-observations` and
      `--save-wm-policy-condition`: future H5s can contain `policy_obs`
      `(T+1,43)`, `policy_obs_frame_stack` for audit, and 64D causal
      `wm_policy_condition` `(T+1,64)`. The Cosmos allocation wrapper defaults
      these saves on. Readiness now rejects any candidate positive takeover
      run that lacks this contract.
- [ ] Record failed controller rollouts as negative diagnostics with the exact
      reason: visual grasp failure, slot-vs-metric drift, task-state readout
      drift, physical bridge failure, or scheduling/render failure.
- [x] Test whether frozen/static DP can provide dynamic teacher trajectories.
      Under allocation `111343`, a no-video `dp_only` scan on
      `hole_constant` seeds `702000..702004` produced
      `success_at_end=0.2`, `success_once=0.6`; the only final-success seed
      `702004` had no video and did not reproduce when rerun with
      `controller_env_states`. A second scan on seeds `702100..702104` saved
      `policy_obs`, `wm_policy_condition`, and `controller_env_states`, but
      produced `success_at_end=0.0`, `success_once=0.0`. Readiness rejects it
      with `positive_controller_source_count=0`. Conclusion: DP-only is not a
      reliable positive dynamic teacher source for this setting. Evidence:
      `docs/world_model_task_rebinding/2026-06-06_dp_dynamic_teacher_scan.md`.
- [x] Add a success-filtered official ManiSkill motion-planner expert probe.
      `scripts/world_model/evaluate_motion_planner_moved_hole.py` now supports
      `--only-count-success`, `--max-attempts`, and H5 trace export. Static
      seeds `0..4` gave success rate `0.4`; moved-hole seeds `0..4` gave
      success rate `0.2`. A success-filtered moved-hole collection gathered
      three successful expert traces from seeds `2`, `6`, and `9` under
      `experiments/world_model_task_rebinding/motion_planner_moved_hole/moved_hole_collect_success3_seed0_max30_20260606_continue/expert_traces.h5`.
      Boundary: this is a simulator-state official-planner expert trajectory
      source with `pd_joint_pos` actions of dimension `8`, while the frozen DP
      uses `pd_ee_delta_pose` actions of dimension `7`. It can seed
      trajectory-level/continuability supervision or an action-contract
      conversion step, but it must not be inserted directly as positive
      DP-action takeover BC data.
- [x] Add a scaffold converter from official planner expert traces to the
      current DP takeover H5 contract. New script:
      `scripts/world_model/convert_motion_planner_expert_to_dp_takeover.py`.
      It restores saved official-planner env states inside the frozen DP
      `pd_ee_delta_pose` environment, records `policy_obs`, converts expert TCP
      motion to 7D delta-pose actions, and writes causal
      `wm_policy_condition`. Converted output:
      `experiments/world_model_task_rebinding/wm_policy_distillation/motion_planner_converted_dp_takeover_20260606_continue/converted_takeover.h5`
      with three trajectories: `policy_obs` dims `43`, converted action dim
      `7`, condition dim `64`, and `metric_slots`. Dry-run loading through
      `train_wm_conditioned_policy_distillation.py` reported
      `base_samples=152291`, `takeover_samples=532`,
      `full_training_allowed=false`. Boundary: this is scaffold smoke data,
      not admitted RGB-D/Cosmos positive takeover evidence.

## Training

- [x] Implement a world-model-conditioned DP-style policy head that can be
      initialized from or distilled against the official DP checkpoint.
- [x] Add the conservative training entry point:
      `scripts/world_model/train_wm_conditioned_policy_distillation.py`. It
      preserves the official DP architecture by keeping the original
      observation-conditioning dimension and injecting world-model trajectory
      features through a zero-initialized adapter. With zero WM condition it
      starts from the loaded static DP behavior. Full training refuses to start
      unless the manifest contains admitted positive takeover teacher sources.
- [x] Use a mixed objective:
      `L_base_dp_bc + L_takeover_bc + L_grasp_hold + L_task_frame_progress +
      L_switch`. `train_wm_conditioned_policy_distillation.py` now implements
      weighted base/takeover diffusion behavior cloning plus auxiliary
      grasp-hold, task-frame-progress, and switch losses. Positive takeover H5s
      must provide `policy_obs`, causal `wm_policy_condition`, `metric_slots`,
      and `event_log_json`; failed videos and scaffold-only runs are still
      rejected before training.
- [x] Add allocation runner:
      `scripts/slurm/run_wm_policy_distillation_in_allocation.sh`. It is meant
      for held `salloc`/tmux resources and records that base-only runs are smoke
      only, not method evidence.
- [x] Keep the training entry robust to optional diffusers/accelerate/boto3
      environment drift by using a local EMA fallback when
      `diffusers.training_utils.EMAModel` cannot be imported.
- [ ] Train on Slurm only. A local run may only compile or build a manifest.
      Current verification used compile, dry-run, and refusal checks only; no
      heavy training was run from login and no full training started because
      the manifest has zero positive takeover teachers.
- [x] Add a takeover-eval preflight:
      `scripts/world_model/audit_wm_policy_takeover_ready.py` checks that a
      dynamic `wm_policy_takeover` eval has a checkpoint, receding/live
      world-model interface, admitted positive takeover sources, and static
      preservation metrics. Strict preflight on the current artifacts correctly
      reports not ready because there is no positive takeover teacher and no
      full trained checkpoint.
- [x] Run a base-only smoke checkpoint inside Slurm allocation `111279` only
      to verify the code path. It trained for 2 iterations with
      `base_batch_fraction=1.0`, `takeover_batch_fraction=0.0`, produced
      `train/checkpoints/final.pt`, and strict preflight rejects it as method
      evidence because `source_positive_takeover_count=0`,
      `run_takeover_samples=0`, and full training was not allowed by the
      source manifest.
- [x] Fix mixed distillation sampling so takeover data are actually seen.
      Before this patch, `train_wm_conditioned_policy_distillation.py` used
      `ConcatDataset(..., shuffle=True)`, so tiny takeover datasets were almost
      never sampled beside 152k base DP samples. Patched it to use
      `WeightedRandomSampler` whenever takeover data exist; the requested
      `--base-sample-fraction` now controls the base/takeover mixture. Smoke
      with converted scaffold data and `--base-sample-fraction 0.5` produced
      `takeover_batch_fraction` values `0.5`, `0.75`, `0.5` over three steps
      with finite grasp/progress/switch auxiliary losses. This is a training
      plumbing fix, not method evidence.
- [x] Add a source-level scaffold guard in the training entry. A hand-edited
      manifest that places converted motion-planner scaffold data under
      `positive_takeover_sources` now refuses to run unless
      `--allow-scaffold-takeover-smoke` is explicitly set, and full training is
      still refused whenever such scaffold sources are present. Negative guard
      check refused the converted scaffold manifest without override; positive
      dry-run with override loaded `base_samples=152291`,
      `takeover_samples=532`, recorded `scaffold_positive_source_count=1`, and
      stayed `full_training_allowed=false`. This keeps scaffold expert data
      from silently becoming formal positive takeover data.
- [x] Add a frozen-base training option after full-parameter scaffold mixing
      proved unsafe. Longer scaffold mixed run
      `train_scaffold_mixed_800iter_20260606_continue` fit the scaffold loss
      (`0.3566 -> 0.0553`) but destroyed static DP preservation:
      `static_preservation_eval10_scaffold_mixed_800iter_20260606_continue`
      gave `success_at_end=0.1`. Patched
      `train_wm_conditioned_policy_distillation.py` with
      `--freeze-base-policy`, freezing the loaded DP `noise_pred_net` while
      training the WM adapter and auxiliary heads; the allocation wrapper
      exposes this as `FREEZE_BASE_POLICY`. Frozen-base scaffold run
      `train_scaffold_mixed_freeze_base_400iter_20260606_continue` preserved
      static skill on a 5-episode smoke with `success_at_end=1.0`. Boundary:
      this is still scaffold/debug only, not admitted RGB-D/Cosmos method
      training.
- [x] Fix the scaffold conversion condition contract. The old converted H5
      used `desired_rel=future_peg_head_at_hole` and condition mode
      `bridge_motion_planner_teacher`, while dynamic `wm_policy_takeover`
      receives controller bridge-mode commands with
      `desired_rel=[preinsert_x or insert_x,0,0]`. Patched
      `convert_motion_planner_expert_to_dp_takeover.py` with default
      `--condition-mode controller_goal` and legacy option
      `legacy_future_peg_head`. New corrected scaffold H5:
      `converted_takeover_controller_goal_full3.h5` with three trajectories,
      7D actions, 43D policy observations, 64D WM conditions, and event
      bridge-plan phase counts `align=391`, `insert=144`. New manifest:
      `scaffold_smoke_manifest_controller_goal.json`; dry-run loaded
      `base_samples=152291`, `takeover_samples=532`,
      `full_training_allowed=false`. Boundary: this fixes scaffold/debug
      training-interface alignment only; it is still not RGB-D/Cosmos method
      evidence.
- [x] Retest corrected controller-goal frozen-base scaffold distillation.
      Final 800-step checkpoint had static preservation `success_at_end=0.4`
      and is blocked. Earlier `iter_00000400.pt` had static preservation
      `success_at_end=0.8` but remains blocked by scaffold provenance. Dynamic
      pure `wm_policy_takeover` with iter400 failed and lost final grasp
      (`[-0.77223,-0.08019,-0.11366]`). Added diagnostic
      `--wm-policy-action-blend` / `WM_POLICY_ACTION_BLEND` to blend learned
      WM-policy action with frozen DP prior. Blend `0.25` preserved grasp and
      reached min yz `0.00308m`, but final yz was still `0.02668m` and no
      insertion; blend `0.10` was worse. Conclusion: DP-prior blending stops
      the worst learned-policy divergence, but scaffold learned-policy takeover
      still does not beat sequence-MPC and is not a usable controller result.
- [x] Localize the DDP-style controller failure beyond generic "no insertion".
      In held allocation `112302`, reran sequence-MPC with controller env-state
      saves and replayed the aligned state. The controller can preserve grasp
      and reach head yz around `0.002m`, but the peg center/body yz was still
      `0.0124m`, unlike successful official planner traces whose pre-insertion
      peg center yz is around `0.0003..0.005m`. Patched
      `evaluate_rebinding_controller.py` so `insert_ready` and handoff require
      peg head yz, peg center yz, and peg angle, added `peg_body` bridge
      reference support, and allowed bounded pre-insert peg-angle correction
      under `contact_preserve`. The body/angle-aware diagnostic improved final
      state to head yz `0.00137m`, body yz `0.00579m`, angle `0.036rad`, and
      x `-0.1033`, but still failed final insertion. Strict handoff gave
      `handoff_count=49` and x `-0.1020`, still far from the official success
      threshold `-0.015`. Always tracking `peg_body` hurt head centering and
      also failed. Conclusion: DDP-style DP-prior/bridge-MPC is useful for
      task-frame preparation but lacks a competent insertion execution layer;
      stop spending effort on scaffold-only distillation or threshold tuning
      as the main path. Next controller work needs a stronger positive
      execution source in the DP action contract or trajectory-level optimizer.
- [x] Test the Dream-Diffusion-Policy-style controller fallback with a DP
      success handoff instead of a hand-written hold. Patched
      `evaluate_rebinding_controller.py` with `success_hold_policy=dp`: after
      the learned world-model-conditioned takeover first reaches the inserted
      manifold, control returns to the frozen/static DP prior to test
      conservative continuability. This fixed the single-seed post-insertion
      instability that zero hold and task-frame success-hold servo could not
      fix. Visual-reviewed diagnostic:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_success3_teacher_freeze_base_120iter_success_dp_handoff_diag_20260606_continue`
      has `success_once=true`, `success_at_end=true`, first success step
      `160`, success frames `141`, final peg-head-at-hole
      `[-0.00176, 0.00302, 0.00294]`, final yz `0.00422m`, and contact-sheet
      review under `visual_review_traj_0/`. Boundary: this is still
      oracle-slot plus scaffold-teacher diagnostic evidence, not RGB-D/Cosmos
      method evidence; final `grasped=false`, so it is peg-left-in-hole
      success rather than strong grasp-hold evidence. A 3-seed no-video
      matrix under
      `wm_policy_takeover_success3_teacher_freeze_base_120iter_success_dp_handoff_seed3_diag_20260606_continue`
      reached only `success_at_end=1/3` and `success_once=2/3`, so the current
      teacher set/policy is too narrow for a stable controller.
- [ ] Next controller step: do not keep tuning post-success hold thresholds.
      The useful direction is grasp-preserving insertion execution: build a
      multi-state closed-loop teacher-trajectory follower or online trajectory
      optimizer from the current state, using teacher TCP, peg body/head, and
      orientation targets rather than raw action replay or TCP-only local
      deltas. Then re-evaluate the hybrid
      learned approach + insertion chunk controller. After the oracle/scaffold
      diagnostic is stable, port the same controller interface to RGB-D/Cosmos
      slots; do not report oracle transient success as method evidence.
      Broader positive execution data in the DP action contract or a
      trajectory-level optimizer may still be needed if the bridge primitive
      cannot preserve grasp.
      insertion execution across moved-hole starts, then re-train/evaluate the
      WM-conditioned policy with static-DP preservation and RGB-D/Cosmos
      controller inputs. The DP success-handoff is a diagnostic design point,
      not a stopping criterion.
- [x] Expand scaffold DP-action insertion teachers and fix zero-condition
      preservation. `teacher_traces_success12.h5` collected 12 successful
      DP-action insertion traces (`737` actions) from 137 failed candidates;
      four sampled contact sheets were opened and looked like normal grasped
      insertion, but this is still sample-reviewed scaffold data only. Initial
      success12 frozen-base training exposed a real implementation bug:
      `wm_condition=0` could still alter the frozen DP because the trainable
      adapter's biases changed after initialization. Patched
      `WMConditionedDiffusionPolicy._condition` to use
      `adapter(cond) - adapter(zeros_like(cond))`, enforcing exact
      zero-condition identity. The fixed success12 240iter checkpoint restored
      5-episode static preservation to `success_at_end=0.6` after the unfixed
      success12 80/240iter checkpoints both scored `0.2`.
- [x] Retest success12 zero-identity dynamic controller. The 3-seed oracle-slot
      diagnostic
      `wm_policy_takeover_success12_zero_identity_240iter_success_dp_handoff_seed3_diag_20260606_continue`
      reached only `success_at_end=1/3`, `success_once=1/3`; it did not
      improve over the success3 scaffold controller. The apparent successful
      seed `702002` did not reproduce when rerun alone with controller
      env-state saves, so there is no visual-positive success12 dynamic
      evidence. Conclusion: broader scaffold teacher plus zero-identity fixes
      preservation plumbing, but the controller is still unstable and needs
      stronger/diverse execution supervision or trajectory optimization.
- [x] Add teacher-support gated takeover as a failure-localization tool.
      Dynamic takeover was being queried far outside the success12 teacher
      support; seed `702000` first takeover was at
      `[-0.4436,-0.0916,-0.1137]`, while teacher starts are around
      x `[-0.128,-0.082]`. Patched `evaluate_rebinding_controller.py` with
      `--wm-policy-takeover-require-teacher-support` and
      `--wm-policy-takeover-prepare-policy`. Pure bridge preparation failed
      badly and never entered support. DP-prior MPC preparation was better
      but with the original strict body-yz gate never handed off. After
      matching the body-yz gate to the actual teacher H5 support
      (`0.065m`), learned takeover fired for 31 steps but still failed
      (`success_once=false`, final
      `[-0.11663,0.01488,-0.03316]`) because peg angle left support
      (`~0.307rad`). Conclusion: the missing stage is orientation/body
      alignment preparation into teacher support, not more post-success hold
      tuning or pretending the insertion-only policy is an approach policy.

## Evaluation

- [x] Evaluate static DP preservation first enough to expose the baseline:
      one-video seed failed, 10-episode no-video eval gave `0.7`, and
      50-episode no-video eval under allocation `110599` gave
      `success_at_end=0.58`, `success_once=0.58` (`29/50`). Future
      world-model-conditioned policies must be compared against this base
      preservation baseline before dynamic claims.
- [x] Add a static preservation evaluator for WM-conditioned policies:
      `scripts/world_model/evaluate_wm_policy_static_preservation.py` plus
      allocation runner
      `scripts/slurm/run_wm_policy_static_preservation_in_allocation.sh`.
      It evaluates a trained WM-policy checkpoint, or a base-DP-initialized
      WM policy with zero condition, on the original static task before any
      dynamic claim.
- [x] Add the dynamic execution interface for the corrected controller path:
      `scripts/world_model/evaluate_rebinding_controller.py` now supports
      `control_policy=wm_policy_takeover`. In this mode the frozen/static DP
      still handles the initial task until the configured takeover condition
      is met, while the learned `WMConditionedDiffusionPolicy` checkpoint
      executes takeover actions conditioned on causal world-model/task-frame
      features. The geometric bridge planner is used only to form that causal
      condition and diagnostics, not as the final action executor. The path
      refuses to run without `--wm-policy-checkpoint` and, by default, refuses
      non-receding/non-live world-model interfaces. The allocation wrapper
      `scripts/slurm/run_cosmos_task_state_controller_in_allocation.sh` can
      now launch this path from the held tmux allocation with
      `CONTROL_POLICY=wm_policy_takeover` plus `WM_POLICY_CHECKPOINT=...`.
      This is an execution-path patch only; there is still no admitted
      positive takeover teacher and no trained dynamic checkpoint to report as
      method evidence.
- [x] Run a base-initialized WM-policy static preservation baseline:
      50 no-video episodes under allocation `110599` produced
      `success_at_end=0.54`, `success_once=0.54` (`27/50`) with zero WM
      condition. The previous official DP 50-episode baseline was
      `success_at_end=0.58` (`29/50`), so future trained WM policies must be
      checked against both numbers before dynamic claims.
- [ ] Run a statistically meaningful static preservation eval for any trained
      WM-conditioned policy checkpoint. The current 50-episode run is for the
      base-DP-initialized WM policy, not a trained dynamic policy.
- [x] Verify the dynamic `wm_policy_takeover` execution interface on a usable
      allocation. Allocation `111279` on `server13` was rejected as a
      scheduling/rendering failure because `nvidia-smi` could not determine
      the GPU handle, torch reported `cuda_available=False`, and SAPIEN/Vulkan
      failed with `ErrorIncompatibleDriver`. Replacement allocation `111343`
      on `server31` passed GPU preflight and ran one no-video
      `wm_policy_takeover` smoke with the base-only checkpoint. Result:
      `success_at_end=false`, `wm_policy_takeover_count=210`,
      `bridge_count=0`, and H5 contract fields were present:
      `policy_obs (301,43)`, `policy_obs_frame_stack (301,2,43)`,
      `wm_policy_condition (301,64)`, `event_log_json` present. This proves
      the execution branch and data contract, not dynamic method success.
- [x] Add and smoke-test a DDP-style controller alternative:
      `control_policy=wm_dp_prior_mpc` executes a short-horizon candidate
      selector around the frozen DP action prior using Cosmos/world-model
      task-frame predictions as the imagined trajectory. It records
      `wm_dp_prior_mpc_count` separately from `bridge_count` and stores
      candidate reports in the event log. Compile/help checks passed and three
      no-video allocation smokes ran. All failed final success, so this is only
      negative diagnostic evidence, not a controller result.
- [x] Run a small static smoke for the base-only checkpoint:
      5 no-video static episodes under allocation `111343` gave
      `success_at_end=0.6`, `success_once=0.6`. This is only a small
      checkpoint smoke, not the required meaningful trained-policy static eval.
- [x] Add static-preservation thresholding to the takeover-ready audit.
      `audit_wm_policy_takeover_ready.py` now rejects checkpoints whose
      static `success_at_end` is below `--min-static-success-at-end` default
      `0.5`. This is an added preservation preflight, not a change to dynamic
      task success. The unsafe full-parameter scaffold checkpoint is now
      blocked by scaffold provenance and
      `static_preservation_success_at_end_below_minimum`. The frozen-base
      scaffold checkpoint passes static preservation but remains blocked by
      scaffold provenance and lack of admitted positive takeover teacher.
- [x] Run a dynamic oracle-slot diagnostic for the frozen-base scaffold
      checkpoint after static preservation passed. Output:
      `wm_policy_takeover_scaffold_freeze_base_400iter_diag_20260606_continue`.
      It failed: `success_at_end=false`, `wm_policy_takeover_count=210`,
      final grasp remained true, but final peg-head-at-hole was
      `[-0.76907,-0.03461,-0.11303]` and minimum yz was `0.11428m`.
      Conclusion: freezing the DP backbone preserves static DP, but current
      scaffold converted planner data still do not teach a usable dynamic
      WM-policy takeover.
- [ ] Evaluate dynamic Cosmos/RGB-D controller rollouts only after static
      preservation passes.
- [ ] Inspect controller video/contact sheets directly before reporting
      success.
- [ ] Do not continue to new controller variants if the failure is another
      visual peg-hold failure; localize perception/readout/training data first.
- [x] Inspect the 2026-06-07 9-chunk WM-conditioned action-generator videos
      for the user's peg-hold failure question. Reviewed contact sheets for
      `wm_dp_prior_rollout_mpc_wmgen_9chunk_mppi8_seed702000_video_20260607_1155`
      and
      `wm_dp_prior_rollout_mpc_wmgen_9chunk_insertprobe_adaptive_mppi8_seed702000_video_20260607_1255`.
      Both metrics fail (`success_once=false`, `success_at_end=false`) while
      final grasp remains true. Visual conclusion: the peg is picked up and
      held, not immediately tilted/flying upward; the failure is contact-rich
      insertion execution where the peg ends outside/below/to the side of the
      hole. Added `manual_visual_review.json` to both runs with
      `positive_takeover_teacher_ok=false`; neither can be used as positive
      takeover distillation or RGB-D method evidence.
- [x] Run the 2026-06-07 longer action-generator and direct-head follow-up
      without releasing allocation `115187`. First-principles reason: if the
      visible peg-hold failure is not immediate grasp loss, the next DDP-style
      test is whether a short-horizon action generator can produce a live
      insertion executor from current policy state plus WM task condition,
      rather than replaying fixed gates. A 12000-step frozen-base 9-chunk
      action generator with 55% action-prior sampling trained in
      `train_ddp_action_generator_9chunk_wmcondition_freeze_base_12000iter_prior55_20260607_1345`
      (`loss 0.2164 -> 0.01231`, min `0.00645`) and matched the base DP
      static same-seed smoke (`success_at_end=0.3` on 10 episodes), but the
      dynamic no-video run
      `wm_dp_prior_rollout_mpc_wmgen_9chunk_prior55_12000_mppi8_seed702000_no_video_20260607_1405`
      still failed (`success_at_end=false`, final grasp true, final
      peg-head-at-hole `[-0.149186,0.049725,-0.056830]`). Added an optional
      direct short-horizon action head to
      `train_wm_conditioned_policy_distillation.py` and rollout-MPC direct-head
      candidates in `evaluate_rebinding_controller.py`. The 6000-step
      direct-head run
      `train_ddp_direct_action_head_9chunk_wmcondition_freeze_base_6000iter_prior55_20260607_1415`
      fit action chunks (`direct_action_loss 0.1654 -> 0.0198`), but dynamic
      direct-head evaluation still failed:
      `wm_dp_prior_rollout_mpc_directhead_9chunk_prior55_6000_mppi8_seed702000_no_video_20260607_1430_retry`
      selected the direct head for only `4` steps and ended
      `[-0.137025,0.072565,-0.115241]`; adding insert-probe candidates in
      `wm_dp_prior_rollout_mpc_directhead_insertprobe_9chunk_prior55_6000_mppi8_seed702000_no_video_20260607_1440`
      selected a predicted inserting probe at step `126` but live execution
      still failed, ending `[-0.194064,0.024835,-0.068018]`. Conclusion:
      current action-prior/direct-head training can fit available chunks but
      does not solve live contact insertion. Do not promote these rollouts as
      positive teachers; the next aligned controller work should add
      visually admitted dynamic execution data or unify live contact-state
      trajectory optimization with replay-verified execution, not keep
      extending fixed-replay or threshold gates.
- [ ] Keep allocation-`115187` DP-only WM-condition scan outputs pending until
      visual review is possible. No-video scan
      `dp_only_wmcondition_action_prior_scan_seed702312_36eps_20260607_1510`
      found `success_at_end=11/36` and `success_once=14/36`. Nine seeds also
      have `final_grasped=true`: `702316`, `702321`, `702325`, `702332`,
      `702335`, `702340`, `702342`, `702344`, and `702346`. These are useful
      candidates for future visual admission because the rollout saved
      `policy_obs`, `policy_obs_frame_stack`, and `wm_policy_condition`, but
      they are not admitted action-prior chunks yet. The first video rerun on
      `server14` stalled after writing only `manifest.json`; a direct render
      preflight could import SAPIEN/ManiSkill, make the env, and reset, but
      hung at `env.render()`. Steps `115187.13` and `115187.15` were cancelled
      while keeping allocation `115187` alive. Boundary: do not train from
      these metric-only candidates and do not extract them into an
      action-prior manifest until videos/contact sheets pass manual review on
      a render-capable allocation. Follow-up detached tmux
      `wm_dp_scan_115187_702348` is using step `115187.16` for a 48-episode
      no-video scan from seed `702348`; this keeps the held allocation used
      for aligned metric-candidate discovery only. Patched
      `extract_dp_action_prior_chunks.py` so DP-only action-prior extraction
      now refuses metric-only rollouts unless `--manual-review-json` contains
      `candidate_dp_action_teacher_ok=true`; an explicit
      `--allow-unreviewed-action-prior-smoke` override exists only for
      non-evidence smoke. `py_compile` passed and a negative extraction check
      against the `702312..702347` scan correctly refused without review.
      Also patched `build_ddp_action_generator_manifest.py` so DP-only
      action-prior chunks are rejected unless their H5 summary records
      `candidate_dp_action_teacher_ok=true`, and unreviewed smoke chunks are
      never manifest-admitted. A temp rebuild of the existing 9-chunk inputs
      still accepted 9 reviewed/official chunks and rejected 1 bad source.
      Submitted render-admission canary job `115449` for the 9 final-grasp
      metric candidates, with `--exclude=server14` because current live
      evidence shows `env.render()` hangs there. It started on `server40` but
      stalled on the first seed after writing only `manifest.json`, with no
      metrics/video, and was cancelled. This is a render/scheduling failure;
      the 9 metric candidates remain pending visual review.
- [ ] Keep allocation `115483` as no-video/controller-debug capacity, not
      visual evidence. After `115187` was cancelled, a new tmux-held 1-H200
      allocation `115483` started on `server58`. Render diagnosis on this
      allocation shows SAPIEN device summary, env creation, and reset succeed,
      but the first `render_rgb_array("render_camera")` call hangs even for a
      256x256 minimal one-frame canary. Existing controller contact sheets
      from `11:55` and `12:55` were reopened and show the peg does not
      immediately fly upward; the failure is near-hole/contact insertion.
      Therefore use `115483` for no-video scans or aligned controller/debug
      work only. The no-video scan
      `dp_only_wmcondition_action_prior_scan_seed702365_48eps_20260607_1717`
      completed with `success_at_end=11/48`, `success_once=14/48`. Eight
      final-success episodes retained final grasp and are pending video review
      only: `702369`, `702371`, `702375`, `702383`, `702388`, `702390`,
      `702395`, and `702409`. Three final-success episodes are rejected for
      action-prior admission because final grasp is false: `702367`,
      `702392`, and `702398`. Submitted bounded render-admission job `115714`
      for the eight pending final-grasp candidates, writing to
      `dp_only_wmcondition_action_prior_scan_seed702365_48eps_20260607_1717_visual_review/`,
      with job-local exclusion of current render-failed nodes
      `server14,server40,server58`. It started on `server27` and completed
      successfully (`COMPLETED`, exit `0:0`, elapsed `00:03:18`), producing
      eight 1024x1024 contact sheets and eight `state_replay.mp4` files.
      Manual visual review accepted all eight final-grasp seeds as DP-only
      action-prior candidates and confirmed none show a peg fly-up or
      immediate post-grasp tilt-to-sky failure. Review file:
      `experiments/world_model_task_rebinding/rebinding_controller/dp_only_wmcondition_action_prior_scan_seed702365_48eps_20260607_1717_visual_review/manual_visual_review.json`
      with `candidate_dp_action_teacher_ok=true` and
      `positive_takeover_teacher_ok=false`. Boundary: these can support
      DP-only action-prior extraction/provenance only; they are not positive
      WM-controller takeover data and not RGB-D/Cosmos method evidence.
- [ ] 2026-06-07 17-chunk direct-action-head smoke:
      continued inside held allocation `115483` without releasing the card.
      Extracted the eight visually admitted DP-only candidates to
      `experiments/world_model_task_rebinding/wm_policy_distillation/dp_wmcondition_action_prior_chunks_20260607/dp_wmcondition_seed702369_702409_action_prior_chunks.h5`
      and rebuilt
      `experiments/world_model_task_rebinding/wm_policy_distillation/ddp_action_generator_manifest_20260607_1915/manifest.json`
      with `17` accepted action-prior chunks and `1` rejected source.
      Trained frozen-base direct-action-head smoke
      `train_ddp_direct_action_head_17chunk_wmcondition_freeze_base_24000iter_prior55_20260607_1925`
      for the full `24000` iterations. Final step `23999` logged total loss
      `0.017129648476839066`, direct-action loss
      `0.009156551212072372`, and wrote `checkpoints/final.pt` plus interval
      checkpoints through `iter_00024000.pt`. Static preservation no-video
      10eps immediately afterward was `success_at_end=0.7`,
      `success_once=0.7`, matching the earlier 10eps static DP baseline level
      and not showing an obvious base-skill regression in this smoke. Dynamic
      seed-`702000` no-video rollout-MPC with direct-head candidates still
      failed (`success_at_end=false`, `success_once=false`,
      `wm_dp_prior_rollout_mpc_count=210`). Final grasp remained true and the
      run achieved min lateral/vertical peg-head error `0.0012995906872674823`
      m, but `inserted_any=false`, `inserted_frames=0`, and max insertion-axis
      x was only `-0.11124828457832336` m. Interpretation: the latest
      controller failure is still insertion-axis/contact execution, not
      immediate peg fly-up or basic pickup loss. Boundary: negative
      oracle/Cosmos scaffold evidence only; no RGB-D method claim and no
      visual claim for this exact no-video run. Exact-run saved-state render
      job `115924` was submitted on `server60` and canceled after `2:33`
      because it hung at `render_start` for frame `0` with zero PNG frames
      written. This is rendering failure evidence only and does not change
      the controller metrics.
- [ ] 2026-06-07 static-partial direct-head render check:
      patched `train_wm_conditioned_policy_distillation.py` and
      `run_wm_policy_distillation_in_allocation.sh` with static partial
      takeover pretraining from official DP demos. The physical purpose is to
      expose the frozen-base policy to nonzero WM/task-state conditions
      without changing the DP action target, so conditioning does not destroy
      grasp-hold-insert behavior. Trained
      `train_ddp_direct_action_head_17chunk_staticpartial25_freeze_base_24000iter_prior55_20260607_2030`
      for `24000` iterations; final total loss `0.01756080985069275`, direct
      action loss `0.0090208500623703`, and static preservation 10eps was
      `success_at_end=0.7`, `success_once=0.7`. Dynamic seed-`702000`
      rollout-MPC still failed. The strict run ended with final grasp true
      but no insertion; the allow-noninsert run
      `wm_dp_prior_rollout_mpc_direct_head_staticpartial25_allow_noninsert_seed702000_no_video_20260607_2130`
      also failed with `success_once=false`, `success_at_end=false`,
      final grasp true, inserted frames `0`, and final peg-head-at-hole
      `[-0.07179862260818481, -0.0004356801509857178, 0.00246545672416687]`.
      Rendered the exact allow-noninsert saved states on held allocation
      `116026`/`server13` at 1024px default oblique ManiSkill view and 30fps:
      full video
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_staticpartial25_allow_noninsert_seed702000_no_video_20260607_2130/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`,
      key-frame contact sheet
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_staticpartial25_allow_noninsert_seed702000_no_video_20260607_2130/visual_review_envstate_default_view_1024/contact_sheet.png`,
      and manual review
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_staticpartial25_allow_noninsert_seed702000_no_video_20260607_2130/manual_visual_review.json`.
      Visual/manual conclusion: this latest run does not show immediate
      peg fly-up or post-grasp tilt-to-sky. Metric grasp is continuously true
      from frame `54` through frame `300`, but insertion never occurs; the
      real failure is insertion-axis/contact execution near the hole.
      Boundary: negative oracle/Cosmos scaffold evidence only, not RGB-D
      method evidence and not positive takeover distillation data.
- [ ] 2026-06-07 21-chunk action-prior follow-up:
      continued inside held allocation `116026` on `server13`, without
      releasing the H200. Ran a bounded DP-only/W-M-condition scan
      `dp_only_wmcondition_action_prior_scan_seed702413_48eps_20260607_2140`
      with `success_at_end=8/48` and `success_once=16/48`. Rendered and
      manually inspected the four final-success/final-grasp candidates
      `702440`, `702444`, `702447`, and `702451` at 1024px default oblique
      ManiSkill view. Manual conclusion: these four candidates also do not
      show immediate peg fly-up or tilt-to-sky after pickup; they are stable
      DP-only action-prior candidates only. Rejected metric final-success
      seeds `702418`, `702429`, `702431`, and `702456` because
      `final_grasped=false`. Manual review:
      `experiments/world_model_task_rebinding/rebinding_controller/dp_only_wmcondition_action_prior_scan_seed702413_48eps_20260607_2140_visual_review/manual_visual_review.json`.
      Extracted four chunks to
      `experiments/world_model_task_rebinding/wm_policy_distillation/dp_wmcondition_action_prior_chunks_20260607/dp_wmcondition_seed702440_702451_action_prior_chunks.h5`;
      each chunk is marked
      `source_role=dp_action_prior_candidate_not_wm_takeover` and
      `positive_takeover_teacher_ok=false`. Rebuilt
      `experiments/world_model_task_rebinding/wm_policy_distillation/ddp_action_generator_manifest_20260607_2330/manifest.json`
      with `21` accepted chunks and `1` rejected source, then started
      `train_ddp_direct_action_head_21chunk_staticpartial25_freeze_base_24000iter_prior55_20260607_2335`
      for `24000` iterations with frozen base, direct action head, and static
      partial takeover fraction `0.25`. The full training completed and wrote
      `checkpoints/final.pt`; final step `23999` has total loss
      `0.0207542572170496` and direct action loss
      `0.009112123399972916`. Static preservation 10eps remained
      `success_at_end=0.7`, `success_once=0.7` under
      `static_preservation_staticpartial25_final_21chunk_24000iter_10eps_20260608_0000`.
      Dynamic seed-`702000` allow-noninsert rollout-MPC still failed:
      `success_at_end=false`, `success_once=false`, final grasp true, final
      peg-head-at-hole
      `[-0.13852724432945251, 0.011310238391160965, -0.04001953452825546]`.
      Rollout-MPC selected `wm_policy_direct_action_head` candidates for
      `40/210` reports and `dp_bridge_blend` for `170/210`, but no selected
      or candidate direct-head chunk inserted. Rendered contact sheet/video
      and manual review under
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_staticpartial25_21chunk_24000iter_allow_noninsert_seed702000_no_video_20260608_0006/`;
      visual conclusion is again normal pickup/no peg fly-up, but failed
      insertion. Boundary: action-generator/controller scaffold only, not
      RGB-D/Cosmos method evidence and not positive takeover data.
- [ ] 2026-06-08 insert-window direct-head failure check:
      continued inside held allocation `116026` on `server13`, without
      releasing the H200. Cropped the `21` accepted action-prior chunks around
      first inserted frame into
      `experiments/world_model_task_rebinding/wm_policy_distillation/dp_wmcondition_action_prior_insert_window_20260608/dp_official_wmcondition_21chunk_insert_window_pre64_post16.h5`
      to reduce post-success hold-target dominance while keeping every chunk
      marked `positive_takeover_teacher_ok=false`. Trained
      `train_ddp_direct_action_head_21chunk_insertwindow_staticpartial25_freeze_base_24000iter_prior55_20260608_0025`
      for the full `24000` iterations. Final step `23999` had total loss
      `0.017874812707304955` and direct-action loss
      `0.00978800654411316`. Static preservation 10eps stayed
      `success_at_end=0.7`, `success_once=0.7` under
      `static_preservation_insertwindow21_staticpartial25_final_24000iter_10eps_20260608_0055`.
      Dynamic seed-`702000` allow-noninsert rollout-MPC still failed:
      `success_at_end=false`, `success_once=false`, final grasp true, final
      peg-head-at-hole
      `[-0.1636543720960617, 0.007629893720149994, -0.04722881317138672]`.
      Rollout-MPC selected `wm_policy_direct_action_head` for `24/210`
      reports and `dp_bridge_blend` for `186/210`, with `0` selected or
      candidate inserted chunks. Rendered the exact saved states at 1024px,
      30fps, default oblique ManiSkill view. Full video:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_insertwindow21_staticpartial25_24000iter_allow_noninsert_seed702000_no_video_20260608_0100/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Key-frame sheet:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_insertwindow21_staticpartial25_24000iter_allow_noninsert_seed702000_no_video_20260608_0100/visual_review_envstate_default_view_1024/contact_sheet.png`.
      Manual review:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_insertwindow21_staticpartial25_24000iter_allow_noninsert_seed702000_no_video_20260608_0100/manual_visual_review.json`.
      Visual conclusion: no peg fly-up/basic pickup explosion; the failure is
      still approach/alignment/contact insertion near the moved hole.
      Boundary: negative controller/action-generator scaffold evidence only,
      not RGB-D/Cosmos method evidence and not positive takeover data.
- [ ] 2026-06-08 action-prior replay / insert-probe failure localization:
      prior held allocation `116026` disappeared during continuation;
      `sacct` reports `CANCELLED by 0` after `04:02:26` on `server13`.
      Reacquired a live one-H200 allocation `116575` on `server40` under job
      `wm_controller_h200_hold_cpu1d_0608`; `srun --jobid=116575 nvidia-smi -L`
      confirmed `NVIDIA H200`. A mistaken replacement interrupted earlier
      allocation `116567` after it had just reached `server40`; this is
      scheduling/resource evidence and must not be hidden. With `116575`
      held, ran action-prior replay diagnostic
      `wm_dp_prior_rollout_mpc_action_prior_replay_insertwindow21_seed702000_no_video_20260608_0125`
      using the insert-window 21-chunk action-prior H5. It still failed
      (`success_at_end=false`, `success_once=false`, final grasp true, final
      peg-head-at-hole
      `[-0.11914539337158203, 0.009525928646326065, -0.034445807337760925]`).
      Rollout-MPC selected `teacher_tcp_local_delta` `8` times and direct-head
      `16` times, but no teacher/direct/DP candidate inserted. Teacher support
      was true for only `6` reports; common failures were head/body yz and
      head-x support. Since selected source chunks have first-success within
      the imagined 64-step window, this is not just chunk truncation; offline
      action-prior replay does not transfer cleanly to this live state.
      Server40 rendering of this run hung at frame `0`; step `116575.4` was
      canceled after `00:02:24`, with no PNG frames written. Then ran
      insert-probe diagnostic
      `wm_dp_prior_rollout_mpc_insert_probe_direct_head_seed702000_no_video_20260608_0135`.
      It also failed (`success_at_end=false`, `success_once=false`, final
      grasp true, final peg-head-at-hole
      `[-0.11089277267456055, 0.01614382490515709, -0.04000784456729889]`).
      Probe candidates were selected `16` times and none inserted. Best probe
      laterally aligned to `1.8mm` but stayed at x about `-0.089m`; blend-1
      probe still failed and worsened orientation. Interpretation: current
      bottleneck is coupled insertion-axis/contact/orientation control after
      stable grasp, not peg pickup/fly-up. Boundary: oracle/scaffold
      failure-localization only, not RGB-D/Cosmos method evidence.
- [ ] 2026-06-08 high-bridge-blend visual/metric check:
      ran
      `wm_dp_prior_rollout_mpc_high_bridge_blend_insert_probe_seed702000_no_video_20260608_0148`
      inside held allocation `116575` using larger bridge blends
      `0.0,0.20,0.35,0.50,0.75,1.0`, rollout horizon `80`, execute chunk
      `16`, direct-head candidates, and insert-probe candidates. It still
      failed (`success_at_end=false`, `success_once=false`) with final grasp
      true and final peg-head-at-hole
      `[-0.11413013935089111, 0.01104189082980156, -0.03415173292160034]`.
      Metric slots show grasp first true at frame `54`, grasp remains true
      through frame `300`, inserted frames `0`, max insertion-axis progress
      only about x `-0.1112m`, and final peg z `0.1068m`. This confirms the
      latest controller still parks near pre-insert instead of pushing through
      the hole. The already-rendered insert-window exact replay
      `visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`
      and contact sheet show normal pickup/no peg fly-up. A fresh server40
      render canary for allocation `116575` created and reset the env but
      timed out at `render_rgb_array_start`; no server40 render is visual
      evidence. Follow-up render job `116625` reached `server55` but failed
      immediately because the script rejected the explicit boolean
      `--make-video True`. Corrected render job `116629` reached `server55`
      and hung at first-frame `render_start`; it was canceled after
      `00:05:41` to avoid wasting resources. The high-blend saved states still
      need a render-capable path before visual claims can be made. Boundary:
      negative oracle/controller scaffold only.
- [ ] 2026-06-08 completion-axis scoring smoke:
      patched `evaluate_rebinding_controller.py` and
      `run_cosmos_task_state_controller_in_allocation.sh` with default-off
      `wm_dp_prior_rollout_completion_axis_weight`. This is not an evaluation
      gate change; it fixes the rollout-MPC objective so all candidate phases
      can be charged for remaining insertion-axis shortfall to `insert_x`.
      Smoke run
      `wm_dp_prior_rollout_mpc_completion_axis12_insert_probe_seed702000_no_video_20260608_0210`
      used weight `12.0`, high bridge blends, direct-head candidates, and
      insert-probe candidates. Result remained negative:
      `success_at_end=false`, `success_once=false`, final grasp true,
      inserted frames `0`, max x `-0.11125597357749939`, final
      `[-0.11236929893493652, 0.00887833908200264, -0.020312055945396423]`.
      It did change the selected candidate distribution from high-blend's
      `dp_bridge_blend=186`, `direct_head=8`, `insert_probe=16` to
      `dp_bridge_blend=145`, `direct_head=48`, `insert_probe=16`, but the
      real rollout still parked near pre-insert. Conclusion: objective
      scoring alone is not enough; the next aligned debugging target is
      planner/live contact-state fidelity or a stronger DDP-style action
      generator, not more scalar weight tuning.
- [ ] 2026-06-08 WAM scorer / default-bridge ablation:
      exported a visually reviewed successful DP action-prior chunk dataset
      `experiments/world_model_task_rebinding/wm_policy_distillation/wam_action_prior_temporal_dataset_h16_a8_20260608.h5`
      with `997` samples from `19` included groups, `2` excluded groups,
      history shape `[2,43]`, current slot `30D`, future state `[16,27]`, and
      action chunk `[8,7]`. The dataset is explicitly marked
      `positive_takeover_teacher_ok=false`,
      `candidate_action_prior_teacher_ok=true`, and
      `method_evidence_allowed=false`. Trained non-method mixed smoke scorer
      `experiments/world_model_task_rebinding/wam_action_state_scorer/action_prior_plus_failed_contrastive_smoke_300iter_20260608/checkpoints/final.pt`;
      train/val RMSE moved from `0.406120/0.406248` to
      `0.050978/0.050628`, but `positive_source_count=0` and
      `method_source_count=0`, so this checkpoint is not method evidence.
      Patched selected-WAM logging into the rollout-MPC controller. Exact
      `hole_constant` reproduction with the stricter task-frame-projected,
      contact-preserve settings still failed with final grasp true, final
      peg-head L2 `0.113145m`, selected candidate kinds
      `dp_bridge_blend=184`, `bridge_variant=16`, and selected WAM available
      for only `9/200` selected rollout-MPC steps. Matched `hole_constant`
      default-bridge ablation succeeded: `success_once=true`,
      `success_at_end=true`, first success frame `147`, inserted frames
      `147..300`, final grasp true, final metric peg-head-at-hole
      `[0.004329, -0.002598, -0.002590]`, final L2 `0.005674m`, selected
      candidate kinds `dp_bridge_blend=21`, `bridge_variant=16`,
      `insert_probe_residual=15`, candidate WAM available `184/184`.
      Inspected replay:
      `experiments/world_model_task_rebinding/rebinding_controller/wam_scorer_rollout_mpc_rgbd_holeconstant_defaultbridge_seed702000_selectedlog_no_video_20260608_server44/visual_review_envstate_default_view_1024_full301_30fps_20260608/state_replay.mp4`;
      inspection:
      `experiments/world_model_task_rebinding/rebinding_controller/wam_scorer_rollout_mpc_rgbd_holeconstant_defaultbridge_seed702000_selectedlog_no_video_20260608_server44/inspection.md`.
      Interpretation: the DP is not simply blind to the moved target hole; the
      bottleneck is converting the current/predicted target task frame into
      stable short action chunks while preserving grasp/contact. Next
      controller work should make Cosmos3/WAM supply causal target trajectory
      plus selected-chunk action-state scoring/generation under live
      re-observation. Do not treat this smoke scorer or default-bridge single
      seed success as positive takeover distillation data or final RGB-D
      method evidence.
