# Active TODO

Current active work for world-model task rebinding. Keep this file short and
move details into the numbered task files.

## Non-Negotiable Correction

- [ ] 2026-06-09T13:35+08:00 method correction to pickup-aware
      chunked WAM SFT. The previous full301 forward-dynamics SFT is now
      superseded as active method evidence because its visual condition is only
      `condition_frame_indexes_vision=[0..7]`, i.e. about source frames
      `0..28`, while the full1000 label audit shows stable grasp first occurs
      around median frame `56` and target motion onset around median frame
      `91`. That means it does not satisfy the user's required input of the
      complete pickup/holding segment before chunk rollout. Added
      `scripts/world_model/export_cosmos3_maniskill_chunked_wam_conditions.py`,
      `scripts/world_model/audit_cosmos3_chunked_wam_contract.py`, and
      `scripts/slurm/run_cosmos3_maniskill_chunked_wam_sft_in_allocation.sh`.
      The new contract uses approved 301-frame ManiSkill default-view videos
      through `t2w_windows`, with `129` frame chunks, `81` frame observed
      pickup prefix, `80` history action-condition steps, and `48` future
      action/video/state prediction steps. It supports both `joint_policy`
      and `forward_dynamics`; the immediate active run is `joint_policy`
      because action chunk prediction was missing. Smoke checks on 20 source
      trajectories produced `73` chunk rows and passed strict chunked audit
      plus SFT preflight for `129` video frames, `128x32` action rows, and
      `129x56` state targets. Full1000 chunk export now exists at
      `experiments/world_model_task_rebinding/cosmos3/action_state_conditions_full1000_maniskill_default_regen_chunked_pickup_wam_joint_policy_20260609_pickup_chunked_joint_policy`
      with `3211` rows (`2914` train / `297` val), `129` effective video
      frames, `128x32` actions, `129x56` state targets, and strict chunked
      contract/preflight pass under
      `experiments/world_model_task_rebinding/cosmos3/sft_full1000_maniskill_rgb_chunked_pickup_wam_joint_policy_droid_policy_20260609_pickup_chunked_joint_policy`.
      The old full301 SFT process was stopped at about `iter 526`; allocation
      `121995` then ended `FAILED|15:0` with step `121995.2 CANCELLED by 2059`,
      so the card was not preserved. This is recorded as a scheduling/
      execution failure, not method evidence. Replacement `122098` was
      cancelled during a partition switch and produced no useful training.
      Replacement `122128` ran on `server27`, exported/audited the full1000
      chunk rows, then failed in Cosmos distributed CPU-affinity setup; patched
      `external/cosmos-framework/cosmos_framework/utils/distributed.py` to
      treat affinity `OSError` like a non-fatal NVML affinity failure. Current
      held allocation is
      `122177|cosmos3_chunked_pickup_wam_joint_0609|RUNNING|server38|1xH200`,
      tmux
      `cosmos3_chunked_pickup_wam_joint_gpu_retry_20260609_1406`. It keeps the
      card after wrapper failure. Manual retry inside that allocation started
      SFT from `Cosmos3-Nano-Policy-DROID-DCP`, loaded `2914/297` train/val
      rows, passed strict audits, and is currently running GPU training; as of
      `2026-06-09T14:42+08:00` job `122177` is still running on `server38`;
      the log reached at least `iter 70` with roughly `16s/iter`. Startup
      validation at iteration `0` wrote `val_loss=7.944049`; no checkpoint or
      generated chunk demo exists yet. A strict CPU Cosmos inference preflight
      passed for the controller-facing action-history interface:
      `condition_frame_indexes_vision=[0..20]`,
      `condition_frame_indexes_action=[0..79]`, `129` video frames, and
      `128` action rows with the same action-condition indexes preserved in
      Cosmos' internal `sequence_plan`. This proves the eval wrapper can feed
      pickup-complete video prefix plus history action rows into inference;
      it is not model-quality evidence. Also directly sampled the active
      Cosmos SFT dataloader after initializing a one-process `gloo` group:
      despite the generic saved config line `conditioning_config={8:1.0}`, the
      action-conditioned sample's actual sequence plan is
      `condition_frame_indexes_vision=[0..20]` and
      `condition_frame_indexes_action=[0..79]`, with video tensor
      `(3,129,256,256)` and action tensor `(128,64)` carrying `raw_action_dim=32`.
      Therefore the current training dataloader is not silently using the old
      8-latent short prefix for these chunk samples. At `2026-06-09T15:47+08:00`
      the run saved a complete DCP checkpoint at `iter_000000300` and reported
      `Validation loss (iteration 300): 3.669465`; training then resumed past
      `iter 303` with the H200 back at `100%` util. Boundary: this is a
      recoverable intermediate checkpoint and finite validation evidence only,
      not generated-video quality or controller evidence. Patched
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh` so optional
      `USE_T2W_WINDOW=true` clips the exact `t2w_windows` source segment before
      inference instead of accidentally using frame `0..128` from the original
      301-frame MP4. Added and launched tmux
      `cosmos3_chunked_wam_action_eval_after_sft_20260609`, which waits for
      this SFT's `sft_completed` marker and then runs exactly `10` val chunk
      demos under
      `experiments/world_model_task_rebinding/cosmos3/action_eval_after_sft_full1000_maniskill_rgb_chunked_pickup_wam_joint_policy_droid_policy_20260609_pickup_chunked_joint_policy`.
      That watcher is diagnostic only and starts no controller. Follow-up
      `2026-06-09T16:00+08:00`: patched the action-eval sample manifest to
      preserve `source_uuid`, `segment_start_frame`, `prefix_boundary_frame`,
      `valid_prediction_start_frame`, `chunk_role`, and target-motion fields,
      and patched `watch_cosmos3_readout_predict.sh` to resolve the reference
      H5 through `source_uuid` and the reference frame range through the chunk
      `segment_start_frame`/`t2w_window.start_frame` instead of defaulting to
      frame `0`. Added and launched tmux
      `cosmos3_chunked_wam_readout_after_action_eval_20260609`, which waits
      for `chunked_action_eval_10_completed` and then trains/predicts a
      controller-facing task-state readout under
      `experiments/world_model_task_rebinding/cosmos3/task_state_readout_chunked_pickup_wam_joint_policy_20260609`.
      That readout watcher computes target-motion onset, final hole pose,
      future hole-path RMSE, peg-head/task geometry, and grasp/insertion
      diagnostics over the exact `129` generated frames; it also starts no
      controller. Added
      `scripts/world_model/preflight_cosmos3_chunked_readout_alignment.py` and
      connected it into the readout watcher before generated-video decoding.
      The val demo-index preflight `0..9` passed under
      `experiments/world_model_task_rebinding/cosmos3/task_state_readout_chunked_pickup_wam_joint_policy_20260609/chunked_readout_alignment_preflight_val10.json`:
      `strict_alignment_ok=true`, no failures, source H5/video spans are
      `301` frames, every chunk is exactly `129` frames, action sidecars are
      `128x32`, state targets are `129x56`, and the expected displacement-only
      state onset is consistently `+3` frames relative to the exporter
      velocity/displacement onset. This is alignment evidence only, not model
      quality. Verification: `bash -n` passed for the patched action-eval,
      readout-predict, and new chunked readout watcher; `py_compile` passed
      for the new preflight script. As of `2026-06-09T16:00+08:00`, job
      `122177` remains running on `server38` with the H200 at `100%` util and
      SFT has advanced to at least `iter 350`.
      Stopped the old full301 fresh-readout/matrix/event/status watchers so
      the superseded chain cannot wake up and run downstream.
      Follow-up `2026-06-09T16:18+08:00`: the same held allocation
      `122177|server38|1xH200` is still running with GPU util `100%`, about
      `99GB` memory used, and no card release. The SFT log has reached at
      least `iter 414`; only checkpoint `iter_000000300` exists so far, with
      validation loss `3.669465`, and there is still no `sft_completed`,
      `sft_failed`, or `val_loss_summary.json` marker. The chunk-demo watcher
      `cosmos3_chunked_wam_action_eval_after_sft_20260609` remains correctly
      waiting on `sft_completed`; the readout watcher
      `cosmos3_chunked_wam_readout_after_action_eval_20260609` has already
      passed its val-index `0..9` alignment preflight and remains correctly
      waiting on `chunked_action_eval_10_completed`. No controller or DP
      integration has been started.
      Follow-up `2026-06-09T16:26+08:00`: while the SFT continued to at least
      `iter 446`, audited the chunk exporter and found a non-core
      `state_target` indexing bug for `perturb`-derived fields
      (`hole_delta_cumulative`, `peg_delta_applied`, `triggered`) in chunks
      whose segment does not start at frame `0`. Patched
      `scripts/world_model/export_cosmos3_maniskill_chunked_wam_conditions.py`
      so future exports use the source trajectory perturb length rather than
      the local chunk length. The current active condition root was not
      regenerated and SFT was not restarted: readout training/inspection reads
      `hole_pose`, `peg_pose`, `tcp_pose`, `peg_head_at_hole`, `grasped`, and
      `inserted` directly from H5, and the action-eval motion window uses hole
      motion, so this bug does not invalidate the current core target-motion,
      final-target, peg/TCP, grasp, or action-chunk diagnostics. Verification:
      `py_compile` passed for the patched exporter.
      Also created GT reference visuals for the same 10 selected validation
      chunks under
      `experiments/world_model_task_rebinding/cosmos3/task_state_readout_chunked_pickup_wam_joint_policy_20260609/gt_reference_chunk_visuals`.
      These are 129-frame, 30fps clips/contact sheets cropped from the approved
      default-view source videos, not Cosmos outputs and not method success
      evidence. Manual inspection of sample `0` (pre-motion forecast), sample
      `1` (motion-observed rollout), and sample `7` (hole_reverse pre-motion)
      confirms the GT windows use the correct slanted overhead view, include
      pickup/holding before prediction, and show target motion/continuation in
      the expected chunk span.
      Follow-up `2026-06-09T16:35+08:00`: changed the post-SFT 10-demo
      watcher sample set from default val indexes `0..9` to the scenario-
      diverse set `0 1 2 7 8 9 48 49 14 24`, covering `hole_constant`,
      `hole_reverse`, `hole_move_stop`, `peg_drop` recovery, and `peg_disturb`
      recovery. Restarted only the two waiting watcher tmux sessions; the SFT
      allocation/training process was not touched. Patched
      `watch_cosmos3_chunked_wam_readout_after_action_eval.sh` to export
      `HDF5_USE_FILE_LOCKING=FALSE` before its H5 alignment preflight after a
      filesystem lock error. The new readout preflight passed with
      `strict_alignment_ok=true`. Created scenario-diverse GT reference clips
      and contact sheets under
      `experiments/world_model_task_rebinding/cosmos3/task_state_readout_chunked_pickup_wam_joint_policy_20260609/gt_reference_chunk_visuals_diverse`.
      Manual inspection of samples `14` and `24` confirms they are
      `peg_drop`/`peg_disturb` recovery chunks using the approved default
      slanted overhead view. SFT was still running and had reached at least
      `iter 476`.
      Follow-up `2026-06-09T17:15+08:00`: SFT saved complete checkpoint
      `iter_000000600` at `17:08:33`, then validation reported
      `val_loss=4.106751` at `17:11:34`. This is worse than
      `iter_000000300` (`val_loss=3.669465`), so the current `best_val`
      checkpoint remains `iter_000000300` unless a later validation improves.
      Training resumed past `iter 612` and `nvidia-smi` on `server38` returned
      to `100%` GPU util with about `99GB` memory used; job `122177` remains
      running and no `sft_completed`/`sft_failed` marker exists. The watcher
      chain is still waiting and has not started controller work.
      Follow-up `2026-06-09T17:18+08:00`: independently reparsed
      `sft_train.log` and verified the current best-val selection from
      existing DCP checkpoints is still `iter_000000300`:
      iteration `0` has loss `7.944049` but no checkpoint, iteration `300` has
      loss `3.669465` with model metadata, and iteration `600` has loss
      `4.106751` with model metadata. This confirms the post-SFT watcher's
      `CHECKPOINT_SELECTION=best_val` should not accidentally pick the worse
      `iter_000000600` checkpoint if SFT completed now.
      Follow-up `2026-06-09T17:49+08:00`: job `122177` is still running on
      `server38` after about `3h42m`; the SFT log has advanced to at least
      `iter 740`. No `sft_completed`/`sft_failed` marker exists and no new
      checkpoint beyond `iter_000000300` and `iter_000000600` exists yet. The
      action-eval/readout watchers are still waiting; no generated chunk demo
      or controller run has started.
      Follow-up `2026-06-09T18:15+08:00`: job `122177` is still running on
      `server38` after about `4h08m`; SFT has advanced to at least `iter 835`.
      No `iter_000000900` checkpoint or validation exists yet, no completion/
      failure marker exists, and the only checkpoints remain `iter_000000300`
      and `iter_000000600`.
      Follow-up `2026-06-09T18:36+08:00`: SFT saved complete checkpoint
      `iter_000000900` at `18:33:24`, then validation reported
      `val_loss=3.380800` at `18:36:28`. This is better than
      `iter_000000300` (`3.669465`) and `iter_000000600` (`4.106751`), so the
      current best-val checkpoint is now `iter_000000900`. A read-only
      best-val reparse confirms iterations `300`, `600`, and `900` all have
      DCP model metadata and `900` has the minimum validation loss. Job
      `122177` remains running on `server38` with no completion/failure marker
      and no generated chunk demo yet.
      Follow-up `2026-06-09T18:55+08:00`: job `122177` is still running on
      `server38` after about `4h48m` with one H200 allocated. `nvidia-smi`
      reports `100%` GPU util, about `99GB` memory used, and about `486W`
      power draw. The SFT log has advanced to at least `iter 968`; no
      `sft_completed`, `sft_failed`, or `val_loss_summary.json` exists yet.
      The action-eval watcher remains correctly waiting for `sft_completed`,
      and the readout watcher remains correctly waiting for
      `chunked_action_eval_10_completed`. Therefore there are still no
      generated Cosmos chunk demos/readout metrics, and no controller/DP
      integration has started.

## Historical Diagnostics Boundary

The entries below this line are retained as scheduling/debugging history only.
They must not be read as active world-model method progress unless a later
entry explicitly revalidates them under the pickup-complete chunked WAM
contract above.
- [ ] 2026-06-09T13:13+08:00 fresh status monitor started:
      added `scripts/world_model/monitor_cosmos3_full301_fresh_status.sh`
      and launched tmux
      `droid_policy_forward_fresh_status_refresh_20260609`. This is a
      read-only, no-GPU, no-controller monitor that refreshes
      `full301_wam_status_latest.{json,md}` and
      `full301_method_evidence_audit_latest.{json,md}` every `300s`, stopping
      only if the audit becomes complete or fresh SFT writes `sft_failed`.
      First refresh succeeded and reported fresh forward-dynamics SFT
      `iter 455/1500`, loss `0.0165`, no completion/failure marker, next
      checkpoint `iter 600`, and controller still paused. Purpose: prevent
      stale status files from making the run look stuck at an old checkpoint;
      this does not create method evidence. Also cleaned the status markdown
      wording from old `primary/follow-up` labels to `active/fresh SFT`
      labels; JSON structure is unchanged.
- [ ] 2026-06-09T13:09+08:00 live fresh-SFT and context-cleanup
      status: current active Slurm allocation is still only `121995` on
      `server40` with one H200; the card was not released. The fresh
      full301 DROID/Cosmos3-Nano forward-dynamics SFT is actively training and
      the refreshed status reports `iter 439/1500`, loss `0.0178`, no
      `sft_completed`, no `sft_failed`, and only checkpoint
      `iter_000000300` on disk because `save_iter=300`; the next checkpoint
      and validation are at `iter 600`, not a sign of being stuck at 300.
      Time remaining in the allocation is about `83614s`, while estimated
      SFT time to completion is about `18058s`, so the current allocation is
      sufficient. The fresh readout, matrix, and event-trigger watchers are
      alive and waiting on explicit fresh markers only:
      `sft_completed`, `fresh_forward_readout_chain_completed`, and
      `matrix_completed`; no controller is running. Removed the stale tmux
      session `wm_rgbd_wam_h200_0608_170956` after confirming its pane showed
      `Job allocation 118609 has been revoked` and `squeue` contained no such
      job. Patched/refreshed the method audit so the active required list
      contains only the fresh forward-dynamics evidence; old state/CV/oracle
      scaffolds, lightweight object-slot WM, old 93-frame artifacts, and the
      invalidated joint-policy clean-caption chain are now shown only under
      excluded historical diagnostics and do not pollute main progress.
- [ ] 2026-06-09T12:55+08:00 fresh-chain default-entry guard:
      current live allocation remains `121995` on `server40`, running the
      fresh full301 DROID/Cosmos3-Nano forward-dynamics SFT at about
      `iter 393/1500`; no `sft_completed` or `sft_failed` marker exists, and
      only checkpoint `iter_000000300` is currently saved. The fresh readout,
      matrix, and event-trigger tmux watchers are still waiting on explicit
      fresh markers and have not started controller work. Patched remaining
      default status/diagnostic entry points that still named the invalid
      joint-policy roots: `watch_cosmos3_sft_completion_v2v_eval.sh` now
      defaults to the fresh forward-dynamics condition/SFT job dirs, and
      `report_cosmos3_full301_wam_status.py` now defaults to the fresh
      forward-dynamics SFT/action-eval/readout roots, job `121995`, and the
      fresh watcher tmux sessions. Verification: `bash -n` passed for the V2V
      watcher, `py_compile` passed for the status helper, refreshed status
      reports the fresh chain with exact `301` video frames and `300x32`
      action targets in the condition contract, and refreshed audit still
      fail-closes with controller paused until fresh SFT/eval/readout/matrix/
      event-trigger evidence exists.
- [ ] 2026-06-09T12:44+08:00 fresh downstream chain prepared without
      reusing old readout. Added
      `scripts/slurm/watch_cosmos3_forward_dynamics_fresh_readout_after_sft.sh`;
      it waits for the fresh forward-dynamics `sft_completed` marker and the
      strict full301 forward action-eval marker, then trains a fresh
      task-state/external-target readout under
      `task_state_readout_droid_policy_full301_forward_dynamics_clean_caption_20260608`
      on the approved 301-frame RGB manifest and decodes the exact same
      301-frame generated video. It writes explicit no-controller markers and
      refreshes the method audit. Started tmux
      `droid_policy_forward_fresh_readout_after_sft_20260609`; it is waiting
      on the fresh SFT completion marker and is not using GPU yet. Also
      redirected the active required matrix/event-trigger roots away from the
      invalid old roots to
      `action_readout_matrix_droid_policy_full301_forward_dynamics_fresh_readout_20260609`
      and
      `event_trigger_equal_gt_after_forward_dynamics_fresh_readout_20260609`.
      Started waiting tmux sessions
      `droid_policy_forward_fresh_matrix_after_readout_20260609` and
      `droid_policy_forward_fresh_event_after_matrix_20260609` so the order is
      fresh readout -> multi-scenario full301 matrix -> event-trigger equal-GT
      refresh. These watchers use job `121995`, wait on explicit fresh
      markers, and do not launch controller. Verification:
      `bash -n` passed for the new/updated watchers and `py_compile` passed
      for the audit/status helpers. Active SFT is still running on `121995`
      and had reached `iter 347`; no fresh SFT completion/failure marker yet.
- [ ] 2026-06-09T12:38+08:00 latest retrain-boundary enforcement:
      do not continue or report the previous ManiSkill Cosmos3 joint-policy
      SFT/eval/readout chain. Added active-use invalid sentinels to
      `sft_full1000_maniskill_rgb_full301_joint_policy_droid_policy_clean_caption_20260608`,
      its action-eval root, its task-state readout root, the old
      forward-dynamics matrix root, and the pre-boundary event-trigger root.
      Stopped the stale matrix/event-trigger/status watcher tmux sessions so
      they cannot wake up after the fresh SFT and run old joint-policy
      readout/eval. Patched
      `watch_cosmos3_droid_forward_dynamics_after_policy_readout.sh` so any
      future attempt to attach the old joint-policy readout after fresh
      forward-dynamics action eval exits fail-closed with
      `forward_dynamics_readout_blocked_by_retrain_boundary` instead of
      producing controller-facing readout. Patched the method audit so those
      joint-policy artifacts are optional historical diagnostics only, while
      required active evidence is now the fresh DROID/Cosmos3-Nano
      forward-dynamics chain: SFT completion, strict full301 generated video
      eval, fresh external-target readout, matrix diagnostics, and
      event-trigger equal-GT refresh. Verification: `bash -n` and
      `py_compile` passed; refreshed audit remains fail-closed with
      `world_model_evidence_complete=false` and required incomplete items
      limited to the forward-dynamics evidence. Active training is job
      `121995` on `server40`, resumed from `iter_000000300` and continuing
      past `iter 326` at about `17s/iter`; no `sft_completed` or
      `sft_failed` marker exists. The old allocation `118609` disappeared
      from `squeue`; `sacct` records `CANCELLED by 0` at
      `2026-06-09T12:36:35`, not from an agent `scancel`. Current live H200
      work is the single fresh `121995` allocation; do not launch controller.
- [ ] 2026-06-09T12:18+08:00 retrain-boundary correction after the clean
      forward-dynamics interruption: the previous ManiSkill Cosmos3 paths
      with possible length mismatch/truncation remain invalid and must not be
      continued. The clean full301 DROID/Cosmos3-Nano forward-dynamics SFT in
      `sft_full1000_maniskill_rgb_full301_forward_dynamics_droid_policy_clean_caption_20260608`
      is not method evidence yet: allocation `120966` failed with Slurm
      `ExitCode=127:0`, the SFT wrapper wrote `sft_failed` with
      `exit_code=143`, and the training log shows interruption after
      `iter 303`. A complete `iter_000000300` DCP checkpoint exists with
      validation loss `0.026690`, so it may only be used for strict
      full301/equal-length diagnostics. Started tmux
      `droid_policy_forward_iter300_strict_eval_20260609` on live allocation
      `118609` for this diagnostic; CPU and Cosmos framework preflights both
      passed with `301` video frames, `300` action steps, action batch
      `300x64`, state target `301x56`, and condition frames `[0..7]`; the
      generated MP4 must still pass the post-inference `301` frame count
      before any diagnostic readout is trusted. Submitted exactly one fresh
      tmux/salloc continuation request, job `121989`
      (`cosmos3_fd_clean_full301_resume_0609`, 1 H200, 1 day), currently
      pending, to resume the same clean full301 SFT from the latest DCP
      rather than using any old 93-frame chain. Patched
      `watch_cosmos3_droid_forward_dynamics_after_policy_readout.sh` so an
      interrupted `sft_failed` marker can be archived and resumed only when a
      complete latest DCP checkpoint exists; controller remains paused.
- [ ] 2026-06-09T12:24+08:00 strict iter300 diagnostic completed, but it is
      not method evidence. The diagnostic root
      `action_eval_after_sft_full1000_maniskill_rgb_full301_forward_dynamics_iter300_strict_length_diagnostic_20260609`
      passed pre-inference and post-inference length checks: sample/config
      video `301`, action `300`, state target `301x56`, Cosmos framework
      batch video `[3,301,256,256]`, output safetensors vision
      `[3,301,256,256]`, action `[300,64]`, MP4 `301` frames at `30fps`, and
      `strict_contract_ok=true`. Pixel reconstruction is equal-length rather
      than cropped: all-frame PSNR `20.7859` over `301` frames and
      future-only PSNR `19.2148` over `272` frames. The readout diagnostic
      root
      `task_state_readout_droid_policy_full301_forward_dynamics_iter300_diagnostic_20260609`
      also completed with exact `301` frames; it reports all-frame
      hole/peg-head-hole/peg/TCP RMSE `0.07167m` / `0.16725m` / `0.14912m` /
      `0.11128m`, grasped/inserted accuracy `0.6977` / `0.5282`. Visual
      inspection of the reconstruction sheet confirmed a readable default
      oblique-overhead view and frame coverage through `300`, but the later robot/peg geometry
      has visible drift/artefacts. Therefore this checkpoint proves the
      full301 wrapper no longer truncates to the old 93-frame/3s output, but
      it does not prove the forward world model is good enough for controller
      use. Fresh continuation `121995` is now running on `server40`: it
      archived the interrupted failure marker, passed the 12h walltime guard,
      and is running strict audits before GPU SFT resume.
- [ ] 2026-06-09T11:48+08:00 near-handoff poll: both held H200
      allocations remain live. Primary task-state readout is still training,
      now at step `4600/5000`; latest completed eval is step `4500` with
      future hole/head/TCP/insert `0.0540m` / `0.0905m` / `0.0438m` /
      `0.8334`. The readout prediction marker and controller-skipped marker
      are still absent, so forward-dynamics watcher must keep waiting and no
      controller/post-SFT downstream stage was launched. Forward-dynamics
      DROID/Cosmos3-Nano SFT is still training at `235/1500`, loss `0.0298`,
      about `65` iterations from the first checkpoint/validation at `300`;
      no forward `sft_completed` or `sft_failed` marker exists. Method audit
      remains fail-closed with the same incomplete items.
- [ ] 2026-06-09T11:41+08:00 live clean-chain refresh: both held H200
      allocations remain live and no card was released. `120966` on
      `server03` is running forward-dynamics DROID/Cosmos3-Nano SFT at
      `210/1500`, loss `0.0188`, with about `90` iterations to the first
      checkpoint/validation at iter `300`; no forward `sft_completed` or
      `sft_failed` marker exists. `118609` on `server43` is running primary
      task-state readout at train step `4450/5000`; latest completed eval is
      still step `4250` with future hole/head/TCP/insert `0.0585m` /
      `0.0929m` / `0.0427m` / `0.8199`. The primary readout prediction and
      controller-skipped markers are still absent, so the forward watcher is
      correctly waiting on
      `readout_prediction_completed_controller_skipped`. Refreshed method
      audit still exits `1` with `world_model_evidence_complete=false` and
      `controller_should_remain_paused=true`; satisfied forward prerequisites
      include condition contract, action/external-target label audits, and
      runtime config, while incomplete required items remain primary
      external-target readout prediction, forward SFT completion, strict
      full301 forward video/action eval, forward external-target readout,
      motion matrix diagnostics, and event-trigger equal-GT refresh.
- [ ] 2026-06-09T11:38+08:00 forward label audits folded into method
      audit while training continues. Refreshed live state: `120966` on
      `server03` is still running clean full301 forward-dynamics
      DROID/Cosmos3-Nano SFT, now `198/1500`, loss `0.0457`; no checkpoint,
      no `sft_completed`, and no `sft_failed` yet. `118609` on `server43` is
      still running primary task-state readout; train reached step `4350`,
      latest eval step `4250` has future hole/head/TCP/insert `0.0585m` /
      `0.0929m` / `0.0427m` / `0.8199`, but no prediction/completion marker
      yet. Re-ran the current forward condition audits: contract passes with
      exact `1000` rows, train/val `912/88`, RGB video `301`, action
      `300x32`, state target `301x56`, and no failures; action-target audit
      passes with all rows nonzero/varying robot action and varying time
      fraction; external-target audit passes with val `47/88` moving-target
      rows, onset frames `68..91`, no bad rows, and finite final hole/insert
      label summaries. Patched the read-only method audit to require
      `forward_dynamics_action_and_external_target_label_audits`; verification
      `py_compile` passed and refreshed audit marks this new item
      `satisfied`. Overall audit still exits `1` with
      `world_model_evidence_complete=false` and
      `controller_should_remain_paused=true` because post-SFT generated
      full301 video/readout/matrix/event-trigger evidence is still missing.
- [ ] 2026-06-09T11:32+08:00 old-root and controller hardening:
      added invalid sentinels to additional old 2026-06-06 ManiSkill
      Cosmos3 action-eval/readout/receding diagnostic roots that predate the
      length/truncation boundary:
      `action_eval_after_sft_full1000_maniskill_default_regen_20260606_0126`,
      `task_state_readout_full1000_regen_framemode_20260606_0129`,
      `action_eval_trigger_segment_after_sft_20260606_0002`, and
      `action_eval_receding_segments_after_sft_20260606_0014`. Extended the
      read-only method audit so those sentinels are mandatory. Patched
      `watch_cosmos3_action_eval_readout_controller.sh` so even an explicitly
      enabled one-shot controller diagnostic must first pass the strict
      full301 method-audit evidence check, unless a separate diagnostic-only
      pre-strict override is intentionally set; the default remains controller
      skip. Verification: `bash -n` and `py_compile` passed; refreshed audit
      still exits `1` with `world_model_evidence_complete=false`,
      `controller_should_remain_paused=true`, old invalid status `satisfied`,
      and forward runtime status `satisfied`. Live progress: forward-dynamics
      DROID/Cosmos3-Nano SFT reached `175/1500`, loss `0.0231`, no checkpoint
      or completion marker yet; primary readout train reached step `4200`,
      latest eval remains step `4000`. Required incomplete method evidence is
      unchanged: primary external-target readout prediction, forward SFT
      completion, strict full301 forward video/action eval, forward readout,
      motion matrix diagnostics, and event-trigger equal-GT refresh. Cards
      remain held on `120966`/`server03` and `118609`/`server43`.
- [ ] 2026-06-09T11:25+08:00 forward runtime-config audit gap closed:
      inspected the forward post-SFT action-eval/readout wrappers. The
      action-eval wrapper already refuses short-horizon method runs by
      matching `num_frames` to the reference video frame count and
      `action_chunk_size` to `reference_frames - 1`, runs a pre-inference
      sample/config/action/state contract check, counts generated MP4 frames
      before reconstruction/readout, and calls reconstruction with
      `--require-equal-length --require-no-truncation`. The readout watcher
      uses `READOUT_REQUIRE_EXACT_VIDEO_FRAMES=true`, `READOUT_NUM_FRAMES=301`,
      and the inspection requires expected `301` frames plus external-target
      event metrics. Patched the read-only method audit so it now has a
      required `forward_dynamics_runtime_config_full301_droid_action_gen`
      item in addition to the existing forward data-contract and completion
      checks. Verification: `py_compile` passed; refreshed audit still exits
      `1`, but the new runtime-config item is `satisfied` with
      `job_name=vision_sft_droid_policy_full1000_rgb_full301_forward_dynamics_restart`,
      `Cosmos3-Nano-Policy-DROID-DCP`, `vision_gen=true`, `action_gen=true`,
      `state_t=300`, train/val `num_video_frames=301`, `force_one`,
      `conditioning_config={8:1.0}`, `wam_sft_mode=forward_dynamics`, and
      RGB-only manifest text. Live progress: forward SFT reached
      `151/1500`, loss `0.0229`, no checkpoint and no `sft_completed`;
      primary readout reached step `4000`, now best diagnostic eval with
      future hole/head/TCP/insert `0.0573m` / `0.0864m` / `0.0428m` /
      `0.8284`, but still no readout prediction/completion marker. Required
      incomplete evidence remains primary external-target readout prediction,
      forward SFT completion, strict full301 forward video/action eval,
      forward external-target readout, matrix diagnostics, and event-trigger
      equal-GT refresh. Controller remains paused.
- [ ] 2026-06-09T11:20+08:00 clean chain still running with no new
      consumable artifact: refreshed Slurm, status, marker checks, and method
      audit. Allocations remain live (`120966.7` on `server03` for
      forward-dynamics SFT, `118609.61` on `server43` for task-state
      readout), so no card was released. Forward-dynamics SFT is at
      iteration `134/1500`, loss `0.0199`, latest validation still
      iteration `0` loss `0.079465`; no checkpoint directory exists yet and
      both `sft_completed` and `sft_failed` are absent. Primary readout
      reached train step `3900`; latest eval is still step `3750` with
      future hole RMSE `0.0588m`, peg-head-hole RMSE `0.0944m`, TCP RMSE
      `0.0621m`, inserted accuracy `0.7668`, and no
      `readout_prediction_completed_controller_skipped` or
      `action_eval_task_state_prediction/prediction_completed` marker. The
      forward chain marker is also absent. Therefore there is no safe
      next-stage strict full301 eval/readout to launch in this pass. The
      audit still exits `1`, with
      `world_model_evidence_complete=false` and
      `controller_should_remain_paused=true`.
- [ ] 2026-06-09T11:17+08:00 clean full301 chain refreshed again:
      no new strict stage is complete, and controller remains paused.
      The old ManiSkill Cosmos3 roots with possible 93-frame/301-frame
      mismatch, cropped PSNR, stale one-shot controller input, future-caption
      contamination, or old actioncond data remain invalid for active method
      evidence. Live held allocations are still `120966` on `server03` and
      `118609` on `server43`; no GPU was released. Forward-dynamics SFT from
      `Cosmos3-Nano-Policy-DROID-DCP` is active at iteration `124/1500`,
      loss `0.0291`, latest validation still iteration `0` loss `0.079465`,
      with no checkpoint, no `sft_completed`, and no `sft_failed`; next
      checkpoint/validation is iteration `300`. The strict full301 condition
      contract remains valid: `1000` rows, train/val `912/88`, RGB video
      `301` frames, actions `300x32`, state targets `301x56`, no depth, no
      future caption/metadata hits. Primary task-state readout reached train
      step `3800`; latest eval is step `3750` with future hole RMSE
      `0.0588m`, peg-head-hole RMSE `0.0944m`, TCP RMSE `0.0621m`, and
      inserted accuracy `0.7668`. The readout prediction/completion marker is
      still absent, so this is diagnostic readout training progress only.
      The method audit still exits `1` with
      `world_model_evidence_complete=false` and required incomplete items:
      primary external-target readout prediction, forward SFT completion,
      strict full301 forward video/action eval, forward external-target
      readout, motion matrix diagnostics, and event-trigger equal-GT refresh
      after target motion.
- [ ] 2026-06-09T11:13+08:00 clean full301 chain still running:
      refreshed Slurm, tmux, logs, status helper, and method audit. Live held
      allocations remain `120966.7` on `server03` for fresh
      forward-dynamics SFT and `118609.61` on `server43` for primary
      task-state readout; no card was released and no old chain is running.
      Forward-dynamics SFT is at iteration `107/1500`, loss `0.0326`,
      latest validation still iteration `0` loss `0.079465`, with no
      checkpoint yet, no `sft_completed`, and no `sft_failed`; next checkpoint
      and validation remain iteration `300`. Primary readout reached train
      step `3750`; latest eval remains step `3500` with future
      hole/head/TCP/insert `0.0625m` / `0.0908m` / `0.0401m` / `0.7668`.
      The expected forward post-SFT action-eval/readout roots do not exist
      yet because forward SFT is not complete. Watchers are waiting on the
      correct clean markers:
      `task_state_readout_droid_policy_full301_joint_policy_clean_caption_20260608/readout_prediction_completed_controller_skipped`
      and
      `droid_policy_forward_dynamics_after_clean_policy_readout_20260608/forward_dynamics_chain_completed`.
      Refreshed method audit still exits `1` with
      `world_model_evidence_complete=false`, and controller remains paused.
- [ ] 2026-06-09T11:08+08:00 clean full301 DROID/Cosmos3-Nano status:
      rechecked the user boundary that previous ManiSkill Cosmos3 SFT/eval
      chains with possible input-vs-GT length mismatch, accidental
      truncation, cropped PSNR, contaminated future caption, stale one-shot
      controller input, or old 93-frame output remain invalid for active
      method evidence. Current live jobs are still only the clean full301
      jobs: `120966.7` on `server03` for forward-dynamics SFT and `118609.61`
      on `server43` for primary task-state readout. Forward-dynamics SFT is
      training normally at iteration `100/1500`, loss `0.0667`, latest
      validation iteration `0` loss `0.079465`, no checkpoint yet, no
      `sft_completed`, and no `sft_failed`; active ETA is about `23296s`,
      which fits the held allocation with a 2h eval buffer. Primary readout
      train log reached step `3650`, latest eval step `3500`: future
      hole/head/TCP/insert are `0.0625m` / `0.0908m` / `0.0401m` / `0.7668`.
      Confirmed the launch log has `strict_length_preflight=passed`: the
      pre-SFT full scan checked train/val `912/88`, video frames `[301]`,
      action lengths/dims `[300]x32`, and state lengths/dims `[301]x56`;
      config train/val `num_video_frames` is `301` and `state_t` is `300`.
      The Cosmos VAE `tokenizer.chunk_duration=93` is only an internal
      tokenizer setting here and is not accepted as method evidence unless
      post-SFT generated outputs also pass explicit `301`-frame checks.
      The readout prediction marker is still absent, so the forward watcher is
      correctly waiting; matrix and event-trigger watchers still wait on
      `forward_dynamics_chain_completed`. Refreshed method audit exits `1`
      with incomplete items: primary external-target readout prediction,
      forward SFT completion, strict full301 forward video/action eval,
      forward external-target readout, matrix diagnostics, and event-trigger
      equal-GT refresh. Controller remains paused.
- [ ] 2026-06-09T11:03+08:00 status-report hygiene and live progress:
      refreshed the clean full301 DROID/Cosmos3-Nano chain. Slurm allocations
      remain live and held: `120966.7` on `server03` for forward-dynamics SFT
      and `118609.61` on `server43` for primary task-state readout. Forward
      SFT reached iteration `73/1500`, loss `0.0332`, latest validation still
      iteration `0` loss `0.079465`, with no `sft_completed`, no
      `sft_failed`, and no checkpoint yet. Primary readout training log
      reached step `3500`, while the latest completed eval/metrics file is
      still step `3250`; no `readout_prediction_completed_controller_skipped`
      marker exists yet. Patched read-only
      `report_cosmos3_full301_wam_status.py` to stop probing stale fixed
      Slurm step `118609.41` by default and to parse readout `train.log` so
      reports distinguish current train step from latest eval metrics.
      Verification: `py_compile` passed, refreshed status shows
      `readout train log latest step/eval step: 3500 / 3250`, current running
      step `118609.61`, and method audit still exits fail-closed with the
      same required incomplete items. Controller remains paused.
- [ ] 2026-06-09T10:58+08:00 DROID/Cosmos3-Nano retrain status after user
      boundary: old ManiSkill Cosmos3 SFT/eval/controller artifacts with any
      possible 93-frame/301-frame mismatch, accidental truncation, cropped
      PSNR, contaminated future caption, or stale one-shot controller input
      remain invalid for active evidence. Current live Slurm state has only
      the clean full301 jobs: `118609` on `server43` for primary task-state
      readout and `120966` on `server03` for the fresh
      `Cosmos3-Nano-Policy-DROID-DCP` forward-dynamics SFT. The forward
      condition contract is strict (`1000` rows, train/val `912/88`, RGB
      video `301` frames, action `300x32`, state target `301x56`, no depth,
      prefix video condition plus action-state conditioning). Refreshed
      status: forward SFT reached iteration `55/1500`, loss `0.0265`, latest
      validation is still iteration `0` loss `0.079465`, estimated active
      remaining time about `24074s`, and the held allocation has about
      `77496s` remaining. Patched read-only
      `report_cosmos3_full301_wam_status.py` so follow-up walltime forecast
      uses the active forward SFT speed/remaining iterations instead of the
      completed primary SFT speed, and so completed SFT runs no longer report
      bogus next checkpoint/validation iterations. Verification:
      `py_compile` passed and refreshed report now records forecast source
      `active_forward_dynamics_sft`, enough time including a 2h buffer.
      Matrix/event-trigger watchers remain correctly waiting on
      `forward_dynamics_chain_completed`; controller remains paused and the
      method audit still fails closed.
- [ ] 2026-06-09T10:52+08:00 forward-SFT status visibility:
      patched read-only `report_cosmos3_full301_wam_status.py` to parse and
      report the active forward-dynamics SFT job dir/log separately from the
      completed primary SFT. It now recognizes Cosmos lines of the form
      `Iteration N: ... Loss ... Time ...`, reports latest forward iteration,
      latest validation, configured max/save/val iter, and ETA. Verification:
      `py_compile` passed and refreshed status reports forward-dynamics SFT
      iteration `33/1500`, loss `0.0688`, latest validation iter `0` loss
      `0.079465`, remaining `1467` iters, ETA about `24352s`. This is
      monitoring only; no controller launch and no method completion claim.
      Follow-up refresh at `10:55`: iteration `36/1500`, loss `0.0482`,
      actual forward ETA about `24317s`; matrix and event-trigger watchers
      are still waiting on
      `droid_policy_forward_dynamics_after_clean_policy_readout_20260608/forward_dynamics_chain_completed`,
      and the forward watcher is still waiting on the primary readout
      completion marker before post-SFT strict eval/readout. No old root or
      short-horizon output was launched.
- [ ] 2026-06-09T10:50+08:00 primary joint-policy visual/action weakness:
      inspected the completed clean primary full301 action-eval artifacts.
      This run fixes the user's original 3-second truncation concern for this
      root: framework preflight resolved `301` video frames and `300` action
      steps, post-inference frame counting found `301` predicted MP4 frames at
      `30fps`, `output.safetensors` contains `vision [3,301,256,256]` and
      `action [300,64]`, and reconstruction metrics compare all `301` frames
      / future `272` frames without cropping. However, direct visual inspection
      of `reconstruction_all/reconstruction_comparison_sheet.png` and
      `reconstruction_future/reconstruction_comparison_sheet.png` shows
      future-frame ghosting/blur around the robot and hole block after the
      prefix. Action prediction is also weak: normalized `32D` action RMSE
      `0.9041`, future-after-prefix RMSE `0.8796`, post-target-motion RMSE
      `0.8580`. Boundary: equal-length chain evidence only; not controller-
      ready WM evidence and not dynamic task success. Continue forward-
      dynamics SFT/readout and do not launch controller.
- [ ] 2026-06-09T10:45+08:00 clean chain refreshed status:
      primary joint-policy full301 SFT and strict post-SFT action/video eval
      are now complete, but primary task-state readout is still running and
      therefore not accepted as complete evidence. Latest readout log reached
      step `3000`, with future hole/head/TCP/insert `0.0638m` / `0.0917m` /
      `0.0409m` / `0.7500`; best remains step `2000` for the readout
      selection criterion. Forward-dynamics SFT is live in `120966.7`;
      iter-0 validation loss is `0.079465`, and training reached iteration
      `9` with about `16.5s` per post-warmup step. Patched read-only
      `report_cosmos3_full301_wam_status.py` so forward export progress
      checks the condition root's manifest/train-val JSONLs; it now reports
      `1000/1000 completed=True`. The forward-dynamics watcher is still
      waiting for the primary readout completion marker and will not launch
      controller. Refreshed method audit exits `1` with incomplete items:
      primary external-target readout, forward SFT completion, forward
      video/action eval, forward readout, matrix diagnostics, and event-trigger
      equal-GT refresh.
- [ ] 2026-06-09T10:42+08:00 forward-dynamics SFT compute-side hygiene:
      the clean full301 forward-dynamics condition export completed with
      train/val `912/88`, `301` RGB frames, `300x32` actions, and `301x56`
      state targets. A strict preflight/contract/action/target audit passed,
      and SFT is now running inside held allocation `120966` on `server03`
      as step `120966.7` using
      `Cosmos3-Nano-Policy-DROID-DCP`, `conditioning_config={8:1.0}`,
      RGB-only input, and the clean forward-dynamics root. The audit/preflight
      phase of this particular resume initially executed from the login-side
      wrapper before entering `srun`; that is a scheduling implementation
      problem, not a method result. Patched
      `run_cosmos3_full1000_sft_in_allocation.sh` so future SFT preflight and
      `audit_cosmos3_wam_full301_contract.py`,
      `audit_cosmos3_external_target_labels.py`, and
      `audit_cosmos3_action_targets.py` run through
      `srun --jobid=...` on the held allocation. Added
      `scripts/world_model/audit_cosmos3_sft_preflight.py` for that strict
      length scan. `bash -n` and `py_compile` passed. Controller remains
      paused; no forward-dynamics SFT completion/eval/readout evidence exists
      yet.
- [ ] 2026-06-09T10:18+08:00 remaining truncation/I2V guard audit:
      patched `run_cosmos3_receding_teacher_forced_controller_retest.sbatch`
      so event-trigger/receding reconstruction metrics now use
      `--require-equal-length --require-no-truncation` for both all-frame and
      future-frame comparisons. This closes a possible cropped-PSNR loophole
      after the segment preflight. Also patched old
      `run_cosmos3_i2v_recon_in_allocation.sh` and
      `watch_cosmos3_recon_then_sft_in_allocation.sh` so single-image I2V and
      I2V-reconstruction-triggered SFT are refused by default unless an
      explicit diagnostic override is set; the I2V wrapper now passes the
      inner Python `--allow-single-image-i2v-diagnostic` only after the outer
      override. `bash -n` passed. Guard tests: old recon->SFT watcher exits
      `34`; single-image I2V recon exits `30`. Refreshed status:
      forward-dynamics export `675/1000`; primary readout step `2250` eval
      landed but best remains step `2000`; no forward train/val JSONL or SFT
      train log yet; controller remains paused.
- [ ] 2026-06-09T10:13+08:00 forward-dynamics contract and wrapper audit:
      inspected the active full301 forward-dynamics export/SFT path. The
      exporter writes `model_mode=forward_dynamics`,
      `condition_frame_indexes_action=[0..299]`, action sidecars `300x32`,
      state targets `301x56`, and prefix-only causal task-state captions with
      no future GT object pose as condition. The generic SFT wrapper then
      full-scans train/val JSONL, checks every video has `301` frames, every
      action target has `300` rows, every state target has `301` rows, and
      runs `audit_cosmos3_wam_full301_contract.py`,
      `audit_cosmos3_external_target_labels.py`, and
      `audit_cosmos3_action_targets.py` before training. Patched
      `run_cosmos3_maniskill_full301_wam_sft_in_allocation.sh` to pass
      `--raw-action-dim 32`, `--write-state-targets`, and
      `--sanitize-future-caption` explicitly to avoid future default drift.
      `bash -n`, exporter help, and `.venv/bin/python -m py_compile` passed.
      Refreshed status: forward-dynamics export reached `600/1000`, primary
      readout reached train step `2200`, and method audit still correctly
      exits `1` with controller paused.
- [ ] 2026-06-09T10:12+08:00 clean full301 liveness:
      refreshed read-only status after adding the V2V guard. Both allocations
      are still running: `118609` on `server43` and `120966` on `server03`.
      Forward-dynamics export progressed to `550/1000` full301 records; it
      still has no train/val JSONL, no strict condition contract, no
      `sft_train.log`, and no `sft_completed`. Primary readout is still live
      and reached train step `2150`; latest/best completed eval remains
      step `2000`. Follow-up allocation time remaining is about `80427s`,
      still enough for the estimated full forward-dynamics SFT. Continue
      monitoring and do not launch controller.
- [ ] 2026-06-09T10:09+08:00 v2v diagnostic guard tightened:
      patched `watch_cosmos3_sft_completion_v2v_eval.sh` so post-SFT V2V
      diagnostic entry points also refuse any dataset/SFT/output root with
      `INVALID_DO_NOT_USE_20260609.md` and refuse non-`301` frame evals unless
      `ALLOW_SHORT_HORIZON_DIAGNOSTIC=true` is explicitly set. `bash -n`
      passed. Fast guard tests confirmed an old invalid root exits `24` before
      GPU inference and `NUM_FRAMES=93` exits `25` before inference. This is
      misuse prevention only; it does not change current clean full301 jobs
      or make any controller claim.
- [ ] 2026-06-09T10:07+08:00 user retrain boundary revalidated:
      the old ManiSkill Cosmos3 SFT/eval/controller outputs remain untrusted
      because any input-vs-GT length mismatch, accidental truncation, cropped
      PSNR, or 93-vs-301 artifact invalidates them for method evidence and
      controller input. `AGENTS.md` already records this 2026-06-09 rule.
      Current `squeue` has only the clean full301 DROID/Cosmos3-Nano jobs
      running: `118609` on `server43` and `120966` on `server03`; no old
      93-frame/default-actioncond/controller job is pending or running. The
      audit still exits fail-closed with
      `world_model_evidence_complete=false` and
      `controller_should_remain_paused=true`. Primary readout reached eval
      step `2000`, now best by the readout script, with future hole
      `0.0700m`, peg-head-hole `0.0894m`, TCP `0.0536m`, and inserted
      accuracy `0.7306`; this is still diagnostic, not controller evidence.
      Forward-dynamics prework in allocation `120966` is still condition
      export, not SFT training: the export log reached `500/1000`, train/val
      JSONL and strict contract are not written yet, and no `sft_train.log`
      exists. Continue the clean full301 chain and keep controller paused.
- [ ] 2026-06-09T10:02+08:00 live status before handoff:
      both allocations remain running with one non-extern step each:
      `118609.61` on `server43` and `120966.1` on `server03`. No
      `sft_failed` marker exists. Forward-dynamics export reached `400/1000`
      raw records; `sft_started` exists, `sft_completed` does not. Primary
      readout training reached step `1900`; latest completed eval is still
      step `1750`, with the next eval expected at step `2000`. The status
      helper reports follow-up time remaining about `80984s`, enough for the
      estimated full 1500-step forward-dynamics SFT with about `9284s` raw
      margin. Controller remains paused and method audit remains incomplete.
- [ ] 2026-06-09T10:00+08:00 forward-dynamics duplicate-start guard:
      found and fixed a scheduling race: the prestart tmux session already
      began clean forward-dynamics work in allocation `120966`, but the
      fresh-allocation handoff watcher could later see a walltime marker
      before `sft_completed` exists and start the same SFT root a second time.
      Added `sft_started` marker creation to
      `run_cosmos3_maniskill_full301_wam_sft_in_allocation.sh`, added current
      `sft_started` for the active forward-dynamics root, and patched
      `watch_cosmos3_droid_forward_dynamics_after_policy_readout.sh` to wait
      for an existing started SFT to complete/fail instead of duplicating it.
      `bash -n` passed for the run wrapper, watcher, and allocation launcher;
      status helper now reports `forward dynamics SFT started=True`. The
      active export reached `375/1000` raw records. Method audit still
      correctly reports `world_model_evidence_complete=false` and
      `controller_should_remain_paused=true`.
- [ ] 2026-06-09T09:56+08:00 clean WM live progress and follow-up budget:
      refreshed the read-only full301 WAM status. Primary task-state readout
      produced a step-1750 eval: future hole `0.0677m`, peg-head-hole
      `0.0966m`, TCP `0.0515m`, inserted accuracy `0.7490`. This improves
      hole/TCP over step 1500 but worsens peg-head/insertion, so it remains
      diagnostic and not controller evidence. Forward-dynamics condition
      export in `120966.1` reached `325/1000` raw records; train/val JSONL
      and strict contract are still absent, so forward-dynamics SFT has not
      started. Patched the read-only status helper to parse follow-up
      allocation remaining walltime and report a separate
      `followup_pipeline_budget`: `120966` has about `81222s` remaining,
      the estimated full 1500-step forward-dynamics SFT is `71700s`, so the
      held follow-up allocation fits with about `9522s` raw margin and about
      `2322s` after a 2h eval buffer. This is scheduling evidence only; no
      controller launch.
- [ ] 2026-06-09T09:52+08:00 clean DROID/Cosmos3-Nano retrain boundary
      rechecked:
      user reaffirmed that any ManiSkill Cosmos3 SFT result with possible
      input/GT length mismatch, accidental truncation, cropped PSNR, or old
      short-horizon controller attachment is not trustworthy and must not be
      continued. `AGENTS.md` already records this 2026-06-09 boundary. Current
      Slurm/tmux state matches it: only clean full301 DROID/Cosmos3-Nano jobs
      are active (`118609` on `server43`, `120966` on `server03`), with no
      pending/running old truncated jobs. Old roots have
      `INVALID_DO_NOT_USE_20260609.md` markers. The active primary chain used
      RGB-only 301-frame videos and 300 action rows with strict post-SFT
      checks; it remains diagnostic, not controller evidence. Primary readout
      has reached train `step=1750` and still runs toward 5000. Forward-
      dynamics clean condition export has reached `250/1000` records and has
      not yet started SFT because train/val JSONL/contract are not written.
      Controller remains paused.
- [ ] 2026-06-09T09:49+08:00 clean WM liveness:
      status refreshed after another short monitor interval. Forward-dynamics
      condition export is still active in `120966.1` and reached `200/1000`
      raw records; it has not written train/val JSONL or the strict condition
      contract yet, so forward-dynamics SFT has not started. Primary readout
      continues in `118609.61` and reached train `step=1650`; latest/best
      metrics remain the step-1500 eval. Both held allocations are still
      running (`118609` on `server43`, `120966` on `server03`). Controller
      remains paused.
- [ ] 2026-06-09T09:47+08:00 primary readout step-1500 eval and status
      visibility:
      `step=1500` became the new best readout checkpoint. All-frame values:
      hole `0.0764m`, peg-head-hole `0.0995m`, peg `0.0613m`, TCP `0.0621m`,
      grasp `0.8644`, inserted `0.7792`. Future values: hole `0.0761m`,
      peg-head-hole `0.0919m`, peg `0.0606m`, TCP `0.0631m`, grasp `0.8788`,
      inserted `0.7555`. This improves peg-head and peg readout and binary
      grasp/inserted accuracy, but hole localization is worse than step 1250
      and still not controller-ready. Patched read-only
      `report_cosmos3_full301_wam_status.py` to surface readout latest/best
      metrics and forward-dynamics condition export progress. `py_compile`
      passed; refreshed status reports readout best/latest step `1500` and
      forward-dynamics export `150/1000`.
- [ ] 2026-06-09T09:44+08:00 clean WM liveness:
      two held allocations remain running, not released: `118609` on
      `server43` and `120966` on `server03`. Active tmux sessions now include
      `droid_policy_clean_caption_forward_dynamics_prestart_20260609_0940` in
      addition to the readout, forward-dynamics handoff, matrix, and
      event-trigger watchers. Forward-dynamics condition export reached
      `100/1000` records. Primary readout reached training `step=1500`, but
      `metrics_latest.json`/`metrics_best.json` still point to the step-1250
      eval, so the step-1500 eval has not landed yet. Continue both processes;
      controller remains paused.
- [ ] 2026-06-09T09:40+08:00 parallel forward-dynamics prestart:
      to avoid leaving the already allocated follow-up H200 idle while primary
      readout continues, started clean forward-dynamics SFT prework in tmux
      `droid_policy_clean_caption_forward_dynamics_prestart_20260609_0940`,
      reusing Slurm allocation `120966` on `server03`. This does not submit a
      new job and does not release either current allocation. The command uses
      `ACTION_CONDITION_MODE=forward_dynamics`, `BACKBONE=droid_policy`,
      RGB-only full301 source videos, `301` video frames, `300` action rows,
      `Cosmos3-Nano-Policy-DROID-DCP`, and the clean
      `action_state_conditions_full1000_maniskill_default_regen_full301_forward_dynamics_clean_caption_20260608`
      / `sft_full1000_maniskill_rgb_full301_forward_dynamics_droid_policy_clean_caption_20260608`
      roots. Forward-dynamics condition export is active and has reached at
      least `25/1000` records in `export_full301_conditions.log`. The existing
      fresh-allocation watcher remains in place; after this SFT completes it
      will skip duplicate SFT and continue action-eval/readout once primary
      readout writes its done marker. This is aligned WM progress, not
      controller evidence.
- [ ] 2026-06-09T09:38+08:00 primary readout step-1250 eval:
      `step=1250` became the new best readout checkpoint. All-frame values:
      hole `0.0717m`, peg-head-hole `0.1109m`, peg `0.0682m`, TCP `0.0566m`,
      grasp `0.8551`, inserted `0.7658`. Future values: hole `0.0712m`,
      peg-head-hole `0.1047m`, peg `0.0671m`, TCP `0.0554m`, grasp `0.8447`,
      inserted `0.7407`. Peg/TCP/peg-head continuous readout improved over
      step 1000, while hole RMSE worsened slightly and insertion
      classification remains weak. Continue full readout training; do not
      launch controller.
- [ ] 2026-06-09T09:37+08:00 status-helper follow-up step visibility:
      patched `report_cosmos3_full301_wam_status.py` to report
      `followup_running_slurm_steps` for allocation `120966`. This is a
      read-only status change; `py_compile` passed. Refreshed status now
      shows primary readout running in `118609.61` on `server43` and
      follow-up prestart/export running in `120966.1` on `server03`.
- [ ] 2026-06-09T09:35+08:00 live-status hygiene:
      patched `report_cosmos3_full301_wam_status.py` so completed one-shot
      SFT/action-eval/failure-monitor tmux sessions are not reported as
      missing live sessions after their markers exist. This is a read-only
      reporting fix only; it does not change evidence or launch work.
      `py_compile` passed. Refreshed status now reports
      `expected live tmux sessions present=True`, running readout step
      `118609.61`, and completed-stage sessions not required:
      `droid_policy_clean_caption_joint_policy_20260608`,
      `droid_policy_clean_caption_sft_failure_monitor_20260609`, and
      `droid_policy_clean_caption_action_eval_20260608`. Readout training log
      has advanced to `step=1250`; the step-1250 eval line has not appeared
      yet.
- [ ] 2026-06-09T09:32+08:00 user boundary reaffirmed:
      previous ManiSkill Cosmos3 SFT/eval/controller outputs with any possible
      input/GT length mismatch, 93-vs-301 truncation, contaminated future
      captioning, or old controller attachment remain invalid and must not be
      continued. The only active ManiSkill world-model chain is the clean
      RGB-only `Cosmos3-Nano-Policy-DROID-DCP` / DROID-policy full301 chain:
      301 RGB frames, 300 action rows, video-prefix frames `[0..7]`, no depth,
      and strict post-inference frame/action inspection. Current status:
      clean SFT and primary full301 action-eval completed strict length
      checks, but the generated future video is visually degraded and is only
      diagnostic. Exact-frame task-state readout is still live in Slurm step
      `118609.61` on `server43`, with train log past `step=1150`; best eval
      remains `step=1000`. Forward-dynamics condition/SFT/eval, matrix, and
      event-trigger equal-GT evidence are still incomplete, and controller
      remains paused.
- [ ] 2026-06-09T09:28+08:00 primary readout step-1000 eval:
      `step=1000` became the new best readout checkpoint. All-frame values:
      hole `0.0689m`, peg-head-hole `0.1147m`, peg `0.0804m`, TCP `0.0645m`,
      grasp `0.8430`, inserted `0.7719`. Future values: hole `0.0681m`,
      peg-head-hole `0.1082m`, peg `0.0800m`, TCP `0.0603m`, grasp `0.8392`,
      inserted `0.7474`. This improves continuous state readout over step 500,
      but insertion classification is still weak, and the underlying Cosmos
      generated video still has visible future-frame degradation. Treat this
      as better diagnostic/readout progress, not controller-ready evidence.
      Training continued past `step=1050`.
- [ ] 2026-06-09T09:18+08:00 primary readout step-750 eval:
      `step=750` was mixed, so `best_model.pt` correctly remains the
      `step=500` checkpoint. Step-750 all-frame values: hole `0.0741m`,
      peg-head-hole `0.1348m`, peg `0.1130m`, TCP `0.0967m`, grasp `0.8155`,
      inserted `0.7894`. Future values: hole `0.0738m`, peg-head-hole
      `0.1309m`, peg `0.1149m`, TCP `0.0972m`, grasp `0.8023`, inserted
      `0.7668`. Hole localization improved, but task-relative peg/head and
      gripper/TCP state worsened versus step 500. Continue training; do not
      use step 750 for handoff.
- [ ] 2026-06-09T09:10+08:00 status-helper running-step fix:
      patched `scripts/world_model/report_cosmos3_full301_wam_status.py` to
      add a read-only `running_slurm_steps` field from `squeue -s`, while
      preserving the old configured SFT step field for compatibility. This
      prevents the status markdown from implying no work is running after
      primary SFT step `118609.41` completed. `py_compile` passed, and the
      refreshed status now shows current running step `118609.61` on
      `server43` for the readout while follow-up H200 allocation `120966`
      remains running on `server03`.
- [ ] 2026-06-09T09:08+08:00 primary readout step-500 eval:
      second readout validation improved substantially but still is not
      controller-ready. At `step=500`, all-frame RMSE/accuracy are:
      hole `0.0821m`, peg-head-hole `0.1232m`, peg `0.0983m`, TCP `0.0846m`,
      grasp `0.8011`, inserted `0.7894`. Future-frame values are:
      hole `0.0830m`, peg-head-hole `0.1118m`, peg `0.0901m`, TCP `0.0861m`,
      grasp `0.8603`, inserted `0.7668`. This is better than `step=250`
      for peg/TCP/peg-head readout, but hole localization and insertion
      classification remain weak. Training continues; no controller handoff.
- [ ] 2026-06-09T09:00+08:00 primary readout first eval:
      readout validation at `step=250` completed and wrote
      `metrics_latest.json`, `metrics_best.json`, `model_latest.pt`, and
      `best_model.pt`. Early metrics are weak and must not be treated as
      controller-ready evidence: all-frame hole RMSE `0.0859m`,
      peg-head-hole RMSE `0.1791m`, peg RMSE `0.1851m`, TCP RMSE `0.1790m`,
      grasp accuracy `0.7493`, insertion accuracy `0.7894`; future-frame
      hole RMSE `0.0873m`, peg-head-hole RMSE `0.1699m`, peg RMSE `0.1834m`,
      TCP RMSE `0.1844m`, grasp accuracy `0.8297`, insertion accuracy
      `0.7668`. Training continued to `step=300`, so this is an early
      diagnostic checkpoint only. Continue the full readout run and compare
      later evals; do not hand off controller from this weak first readout.
- [ ] 2026-06-09T08:50+08:00 primary readout live progress:
      exact-frame task-state readout is actively running in Slurm step
      `118609.61` on `server43`, not waiting in queue and not using an old
      chain. Compute-node inspection shows the Python process for
      `train_cosmos3_task_state_readout.py` is running with
      `--num-frames 301`, `--future-start-frame 29`,
      `--require-exact-video-frames`, `--frame-mode`, and full `5000` steps.
      `manifest.json` was written at `08:48`, proving the normalizer/data
      setup passed. First training output at `08:50` reached `step=50` with
      `loss=1.2381`, `loss_cont=1.0574`, and `loss_bin=0.9037`. The next hard
      evidence point is the first validation/eval at `step=250`; no controller
      marker exists and controller remains paused.
- [ ] 2026-06-09T08:44+08:00 clean full301 DROID/Cosmos3-Nano
      SFT/action-eval checkpoint:
      the old ManiSkill Cosmos3 SFT/eval/controller chains with possible
      93-vs-301 truncation remain invalid and were not continued. The active
      clean RGB-only `Cosmos3-Nano-Policy-DROID-DCP` SFT completed in held
      H200 allocation `118609` at `2026-06-09T08:32:04+08:00`, with
      checkpoints through `iter_000001500`; best validation was
      `iter_000001200` loss `6.484097` versus final `iter_000001500` loss
      `6.489505`, so post-SFT action eval selected `iter_000001200`. The
      first post-SFT inference attempt failed closed because
      `max_action_dim` was incorrectly written into the Cosmos sample override
      JSON, which Pydantic rejects as an extra field. Patched
      `watch_cosmos3_sft_completion_action_eval.sh` to keep `max_action_dim`
      in model/manifest/preflight metadata but not in the inference sample
      override; parser smoke and `bash -n` passed.
      The rerun completed at `2026-06-09T08:39:36+08:00`. Strict contracts
      now pass for the primary diagnostic sample: framework preflight resolved
      video batch `[3,301,256,256]`, action batch `[300,64]`,
      `num_frames=301`, `action_chunk_size=300`, and condition frames
      `[0..7]`; generated `vision.mp4` has `301` frames at `30fps`; predicted
      action has shape `300x32` against target `300x32`. Reconstruction
      metrics are all-frame PSNR `21.23` over `301/301` frames and
      future-only PSNR `19.71` over `272/272` frames, with no truncation.
      Action RMSE is `0.9041` overall, `0.8796` after prefix, and `0.8580`
      after target motion onset. Direct visual inspection of the reconstruction
      sheets shows the length/camera contract is fixed, but future frames have
      obvious degradation: the arm/object region becomes noisy and partly
      translucent after the early prefix. Therefore this is valid strict
      full-length WAM diagnostic evidence, not final controller-ready world
      model evidence. The exact-frame task-state readout has now started under
      the same allocation; controller remains paused.
- [ ] 2026-06-09T07:03+08:00 source-video contract check:
      inspected the source video dataset manifest and clean JSONL rows. The
      active SFT videos come from
      `sft_dataset_full1000_maniskill_default_regen_20260606_0055`, not the
      old dirty `sft_dataset_full1000_rgbd` preview root. The source manifest
      records the ManiSkill3 `PegInsertionSide-v1` default human-render camera
      (`eye=[0.5,-0.5,0.8]`, `target=[0.05,-0.1,0.4]`, `fov=1.0`), `1024x1024`,
      `30fps`, `frame_stride=1`, `num_videos=1000`, train/val `912/88`, and
      every inspected sample has `num_video_frames=301`, source frames
      `0..300`, and duration `10.0333s`. The clean JSONL `vision_path` fields
      point to this full301 default-camera root, with metadata
      `camera=PegInsertionSide-v1_default_human_render`,
      `source=maniskill_default_human_render_from_env_states`,
      `condition_frame_indexes_vision=[0..7]`, `height=width=1024`,
      `fps=30`, `visual_input=RGB only; depth is not used`, and
      causal-prefix-only task-state metadata. The archived H5 paths inside the
      dataset manifest are used only as saved env-state sources for this
      approved re-render; the old preview videos are not SFT input.
- [ ] 2026-06-09T06:57+08:00 SFT launch manifest contract check:
      inspected the active clean SFT launch manifests. `sft_manifest.txt`
      records `base_checkpoint_path=checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID-DCP`,
      `local_tokenizer_dir=checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID`,
      `wam_sft_mode=joint_policy`, `action_conditioned_sft=true`,
      `require_state_targets=true`, `conditioning_config={8:1.0}`,
      `task_state_conditioning=video_prefix_plus_structured_32d_causal_robot_object_action_state`,
      `visual_input=RGB only; depth is not used`, and
      `invalidated_previous_chain=old_93_frame_cosmos3_maniskill_sft_must_not_be_reused`.
      The same manifest records strict launch preflight passed with
      train/val rows `912/88`, video frames `[301]`, action lengths `[300]`
      and dims `[32]`, and state lengths `[301]` and dims `[56]`.
      `restart_manifest.txt` repeats `backbone=droid_policy`,
      `num_video_frames=301`, `action_chunk_size=300`,
      `sanitize_future_caption=true`, and the same invalidation boundary. This
      is launch-contract evidence only; generated post-SFT output still must
      pass strict 301-frame/300-action inspection before method evidence
      exists.
- [ ] 2026-06-09T06:55+08:00 downstream watcher static check:
      audited the active downstream watcher scripts while clean SFT continued.
      The primary action-eval watcher defaults to the clean full301 DROID
      roots with `NUM_FRAMES=301`, `ACTION_CHUNK_SIZE=300`,
      `ALLOW_SHORT_HORIZON_DIAGNOSTIC=false`,
      `WAIT_FOR_SFT_COMPLETED=true`, `CHECKPOINT_SELECTION=best_val`, and
      invalid-root sentinels. The readout watcher waits for
      `post_sft_action_eval_completed`, trains/evaluates exact `301`-frame
      readout, and defaults to skipping one-shot controller launch. The clean
      forward-dynamics watcher waits for the primary readout skipped marker,
      requires the clean forward-dynamics condition/SFT roots, refuses low
      remaining walltime unless explicitly overridden, and then reuses the
      strict action-eval/readout path. The matrix watcher currently waits on
      the clean forward-dynamics chain completion in the live tmux panel
      before running samples; its internal `WAIT_FOR_SFT_COMPLETED=false`
      only avoids re-waiting after the upstream chain has completed. The
      event-trigger watcher waits for the clean forward-dynamics chain, clean
      forward-dynamics SFT completion, primary readout model/metrics, and
      runs with `RUN_CONTROLLER=false` and
      `ALLOW_SHORT_HORIZON_DIAGNOSTIC=false`. Patched only an outdated comment
      in `watch_cosmos3_action_eval_readout_controller.sh` that still referred
      to "93-frame one-shot"; the logic was not changed. `bash -n` passed for
      the active action-eval, readout, forward-dynamics, matrix, and
      event-trigger watcher scripts. Latest training tail at this check:
      `iter 1227/1500` in allocation `118609`, with job `120966` still
      pending as a GPU/H200 follow-up allocation.
- [ ] 2026-06-09T06:53+08:00 runtime-length guard/status refresh:
      refreshed the clean DROID/Cosmos3-Nano full301 status and evidence
      audit while training continued in the held H200 allocation. Current SFT
      status is `iter 1222/1500`, latest validation is still iter `1200`
      loss `6.484097`, `sft_completed=false`, and `sft_failed=false`.
      Runtime config explicitly records `num_video_frames=[301, 301]`,
      temporal modes `force_one/force_one`, `state_t=300`,
      `action_gen=true`, and `vision_gen=true`; the DROID tokenizer
      `chunk_duration=93` is an internal tokenizer/window setting and is not
      accepted as output length evidence. The active post-SFT watcher remains
      blocked on the clean completion marker and must still run the framework
      full301 action-inference preflight plus generated MP4 frame counting
      before any result can count. The refreshed audit remains incomplete and
      still pauses controller work; incomplete required items are clean SFT
      finish, strict full301 video/action eval, external-target readout,
      forward-dynamics condition/SFT/eval/readout, external-target matrix, and
      event-trigger equal-GT refresh. Follow-up job `120966` is still pending
      with `TresPerNode=gres:gpu:NVIDIAH200:1`, so it is a GPU follow-up even
      though `squeue` prints partition `cpu`.
- [ ] 2026-06-09T06:49+08:00 post-SFT readiness/guard check:
      while the clean SFT continued past `iter 1209/1500`, inspected the
      ordered post-SFT watcher and output roots without touching the running
      training process. The action-eval output root contains only watcher logs
      and a manifest; there is no `post_sft_action_eval_completed`, no
      `selected_checkpoint.json`, and no `inference/` directory, so no stale
      post-SFT artifact is available to be mistaken for evidence. The action
      watcher is still waiting for the clean `sft_completed` marker; readout
      is waiting for `post_sft_action_eval_completed`; forward-dynamics is
      waiting for the primary readout done marker. A read-only dry run of the
      same best-validation checkpoint selection logic over the active SFT logs
      currently selects clean checkpoint `iter_000001200` with validation loss
      `6.484097` and existing DCP metadata. Checked
      `run_cosmos3_full1000_sft_in_allocation.sh`: on normal torchrun exit it
      writes `val_loss_summary.json` from validation log lines and then writes
      `sft_completed`; on nonzero exit before completion its trap writes
      `sft_failed`. This is guard/readiness evidence only, not WM evidence.
- [ ] 2026-06-09T06:46+08:00 clean SFT iter-1200 milestone:
      the active clean full301 RGB `Cosmos3-Nano-Policy-DROID-DCP` SFT reached
      checkpoint `iter_000001200` and wrote model metadata; `latest_checkpoint`
      now points to `iter_000001200`. Validation at iter `1200` completed with
      loss `6.484097`, improving over iter `900` loss `6.597537`. Training
      has continued past `iter 1203/1500` in the same held H200 allocation
      `118609` / step `118609.41`; the allocation was not released. Refreshed
      status and method audit record this as a training milestone only:
      `sft_completed=false`, `sft_failed=false`, no strict full301 post-SFT
      video/action eval, no external-target readout, no forward-dynamics SFT,
      no event-trigger evidence, and no controller launch. The next
      checkpoint/validation is the final configured iter `1500`; action eval
      still waits for the clean `sft_completed` marker and must pass exact
      `301` generated frames plus `300` action-step inspection before it can
      count.
- [ ] 2026-06-09T06:29+08:00 read-only resource/status hardening:
      refreshed the clean full301 DROID/Cosmos3-Nano WAM status and method
      audit while SFT continued. Training is still live in allocation `118609`
      on `server43` / step `118609.41`; current status reports
      `iter 1162/1500`, latest checkpoint still `iter_000000900`, latest
      validation still iter `900` loss `6.597537`, no `sft_completed`, no
      `sft_failed`, no strict full301 post-SFT video/action eval, no readout,
      no forward-dynamics SFT, no event-trigger evidence, and no controller
      launch. Patched `scripts/world_model/report_cosmos3_full301_wam_status.py`
      so the follow-up allocation status no longer relies only on `squeue`:
      it now also parses `scontrol show job`, records
      `scontrol_tres_per_node`, `requests_gpu`, and `requests_h200`, and the
      markdown status explicitly shows `TresPerNode=gres:gpu:NVIDIAH200:1`
      with `requests_h200=True` for job `120966`. `py_compile` passed,
      status refresh passed, and method audit intentionally remains
      incomplete (`audit_rc=1`) because SFT completion and all required
      full301 post-SFT evidence are still missing.
- [ ] 2026-06-09T06:26+08:00 live watcher reload/status note:
      after the user's boundary that previous ManiSkill Cosmos3 SFT results
      are unreliable if input/GT length mismatch or accidental truncation is
      possible, only the clean full301 RGB
      `Cosmos3-Nano-Policy-DROID-DCP` chain remains active. The held H200
      allocation `118609` on `server43` was not released, and the live
      training step `118609.41` remains running. Waiting tmux watchers were
      restarted only to reload the stricter post-SFT framework preflight and
      readout/forward-dynamics guards; no old 93-frame, after-Nano,
      contaminated state-target, or controller watcher is active. Refreshed
      status at this point reports SFT `iter 1151/1500`, latest saved
      checkpoint still `iter_000000900`, latest validation still iter `900`
      loss `6.597537`, no `sft_completed`, no `sft_failed`, and no
      post-SFT demo/readout/forward-dynamics/event-trigger/controller
      evidence. Next checkpoint/validation is `iter 1200`; post-SFT inference
      remains blocked until the clean SFT completion marker exists and then
      must pass exact `301` generated frames plus `300` action-step checks.
      `scontrol show job 120966` was also checked because `squeue` reports its
      partition as `cpu`; the job still requests `TresPerNode=gres:gpu:NVIDIAH200:1`
      and is a GPU follow-up allocation for clean forward-dynamics continuation,
      not a CPU-only substitute.
- [ ] 2026-06-09T06:19+08:00 objective-to-artifact audit refresh:
      mapped the user's current full-length WAM requirement into explicit
      evidence items. The active joint-policy data path is valid as a
      `Cosmos3-Nano-Policy-DROID-DCP` policy/WAM SFT input: a full contract
      audit over all `1000` rows passed with train/val `912/88`, `301` RGB
      frames, `300x32` action targets, `301x56` state/readout labels, model
      mode `policy`, video prefix frames `[0..7]`, no future-caption hits, no
      future-metadata hits, and no invalid-marker hits. The 32-D action target
      layout is `7` robot action dimensions plus causal prefix TCP/peg/hole,
      peg-head-at-hole, hole velocity, grasp/insert predicates, perturbation
      summaries, and time; the full future state sequence remains a label and
      readout target, not a privileged Cosmos input. For the active val sample,
      target hole motion begins at frame `84`, max displacement is
      `0.132002m`, and final hole/insertion geometry is available in the
      `301x56` state target. The forward-dynamics/action-conditioned side is
      now explicitly tracked as separate required evidence: its condition root
      is not generated yet (`missing_train_jsonl`, `missing_val_jsonl`,
      `0/1000` rows), because the watcher waits for primary policy SFT,
      strict full301 action-eval, and primary external-target readout before
      exporting/training that second SFT. Patched
      `audit_cosmos3_wam_full301_contract.py` so missing roots fail closed
      with clear contract failures instead of markdown `KeyError`; patched
      `run_cosmos3_full1000_sft_in_allocation.sh` so every future condition
      export/SFT copies `full301_wam_contract_audit.{json,md}` to the condition
      root as `*_latest`; patched
      `audit_cosmos3_full301_method_evidence.py` and
      `report_cosmos3_full301_wam_status.py` to require/report
      `forward_dynamics_full301_condition_contract`. `py_compile`, `bash -n`,
      refreshed status, and refreshed audit passed; the method audit still
      reports `world_model_evidence_complete=false` and
      `controller_should_remain_paused=true`. Runtime status at this refresh:
      allocation `118609` remains live on `server43`, step `118609.41` is
      running, clean SFT is at `iter 1134/1500`, latest validation remains
      iter `900` loss `6.597537`, no `sft_completed`, no post-SFT demo, no
      readout, no forward-dynamics SFT, no event-trigger WM evidence, and no
      controller launch.
- [ ] 2026-06-09T06:12+08:00 strict DROID/Cosmos3-Nano full301 guard
      refresh: re-read `AGENTS.md` and confirmed the latest 2026-06-09 rule is
      now explicit: any old ManiSkill Cosmos3 SFT/eval/controller result with
      possible input/GT length mismatch, 93-vs-301 comparison, or accidental
      truncation is invalid for active use. The active replacement remains the
      fresh RGB-only `Cosmos3-Nano-Policy-DROID-DCP` chain, not any old SFT
      checkpoint. Refreshed status/audit with `.venv/bin/python`; the audit
      still correctly reports `world_model_evidence_complete=false` and
      `controller_should_remain_paused=true` because clean SFT completion and
      post-SFT equal-length action/video/readout evidence do not exist yet.
      Allocation `118609` is still held on `server43`, step `118609.41` is
      live, and the clean SFT status refresh recorded `iter 1113/1500` while
      the log tail had reached `iter 1114`; latest validation remains
      iter `900` loss `6.597537`, with next checkpoint/validation at
      iter `1200`. The action-eval watcher is still waiting for the clean
      `sft_completed` marker and has not advanced from `iter_900` or old
      93-frame artifacts. Additional guard hardening is now active:
      `scripts/world_model/preflight_cosmos3_action_inference_contract.py`
      imports the Cosmos action inference helpers and constructs the actual
      framework action batch on CPU before GPU inference. Smoke output at
      `experiments/world_model_task_rebinding/cosmos3/framework_action_preflight_smoke_20260609_0602/framework_action_inference_contract.json`
      passed with video batch shape `[3,301,256,256]`, action batch shape
      `[300,64]`, resolved `num_frames=301`, resolved
      `action_chunk_size=300`, and condition frames `[0..7]`. The active
      action-eval watcher now runs this framework preflight inside the held
      Slurm allocation before expensive inference, and the strict inspector,
      status report, and method audit require its output before accepting
      `post_sft_action_eval_completed`.
- [ ] 2026-06-09T05:49+08:00 boundary implementation refresh:
      restated user correction that old ManiSkill Cosmos3 SFT/eval artifacts
      with possible input-vs-GT length mismatch or accidental truncation must
      not be continued. Verified old 93-frame, after-Nano, contaminated
      immediate-state-target, old matrix/readout, and superseded
      forward-dynamics roots have `INVALID_DO_NOT_USE_20260609.md` sentinels;
      the only active SFT is still the clean full301 RGB
      `Cosmos3-Nano-Policy-DROID-DCP` chain in held allocation `118609`.
      Refreshed status shows SFT still running, not completed, so no model
      evidence exists yet. Extended
      `scripts/world_model/audit_cosmos3_full301_method_evidence.py` to
      require `event_trigger_equal_gt_refresh_after_target_motion`, meaning
      frame-0 full301 reconstruction/readout alone cannot satisfy the active
      world-model evidence contract. Added WM-only mode to
      `run_cosmos3_receding_teacher_forced_controller_retest.sbatch` and
      launched tmux
      `droid_policy_event_trigger_equal_gt_after_forward_dynamics_20260609`,
      which waits for the clean forward-dynamics chain and will run only
      equal-GT event-triggered Cosmos/readout segments with
      `RUN_CONTROLLER=false`; it will not launch controller or reuse old
      checkpoints. Follow-up hardening in the same run: updated
      `scripts/world_model/report_cosmos3_full301_wam_status.py` so the
      read-only status report also tracks the event-trigger equal-GT root,
      segment counts, role counts, and the new tmux watcher. `py_compile`,
      status refresh, and audit refresh passed; latest status still shows the
      clean SFT running with no `sft_completed` marker, so there is still no
      world-model evidence yet. Follow-up hardening at `2026-06-09T05:56+08:00`:
      patched the event-trigger segment path so every segment writes
      `pre_inference_segment_contract.json` before Cosmos GPU inference. The
      preflight checks segment video frames, action rows, VAE temporal
      roundtrip, source H5 span, variable remaining-horizon equality, and
      refusal of short-horizon diagnostics. The method audit now requires
      this pre-inference contract for
      `event_trigger_equal_gt_refresh_after_target_motion`, and the status
      helper exposes the preflight result per segment. `bash -n`,
      `py_compile`, status refresh, and audit refresh passed without
      interrupting training.
      Runtime poll at `2026-06-09T05:57+08:00`: allocation `118609` remains
      held/running on `server43`, Slurm step `118609.41` is still live, and
      the clean DROID/Cosmos3-Nano SFT reached `iter 1070/1500`. Latest saved
      checkpoint is still `iter_000000900`; latest validation remains
      iter `900` loss `6.597537`; next checkpoint/validation is `iter 1200`.
      No `sft_completed`, `sft_failed`, post-SFT action eval, readout,
      forward-dynamics, matrix, event-trigger, or controller evidence exists.
      Refreshed status/audit still report
      `world_model_evidence_complete=false` and
      `controller_should_remain_paused=true`.
      Runtime/guard refresh at `2026-06-09T06:00+08:00`: allocation `118609`
      remains running on `server43`; clean SFT reached `iter 1079/1500`, with
      latest validation still iter `900` loss `6.597537` and no completion or
      failure marker. Re-audited the active post-SFT watcher path while the
      SFT is running: `watch_cosmos3_sft_completion_action_eval.sh` still
      points only at the clean full301 DROID condition/SFT/action-eval roots,
      uses the DROID tokenizer, `condition_frame_indexes_vision=0..7`,
      `num_frames=301`, `action_chunk_size=300`,
      `allow_short_horizon_diagnostic=false`, and `checkpoint_selection=best_val`.
      Its inference preflight will reject non-301 sample/config/action/state
      contracts before GPU inference, the post-inference length precheck will
      reject any generated video whose counted frames differ from `301`, and
      the strict artifact inspector requires all/future reconstruction metrics
      with equal length/no truncation plus `300` predicted action steps for
      policy mode before writing `post_sft_action_eval_completed`. Readout is
      exact-frame `301` and still skips controller by default. `bash -n`,
      `py_compile`, status refresh, and method audit refresh passed; the
      method audit remains incomplete because no final post-SFT artifacts exist.
- [ ] 2026-06-09T02:43+08:00 user boundary restated:
      any previous ManiSkill Cosmos3 SFT/action-eval/controller artifact that
      may involve input-vs-GT length mismatch, accidental truncation, stale
      93-frame action eval, contaminated future caption/state-target
      conditioning, or a superseded pre-clean restart must not be continued or
      reported as method evidence. The only active training chain remains the
      clean full301 RGB `Cosmos3-Nano-Policy-DROID-DCP` SFT already running
      inside held Slurm allocation `118609` / step `118609.41`; do not cancel
      or release this H200 merely to restart the same compliant chain. Current
      manifest evidence says this active chain uses `301` RGB frames, `300 x
      32` action targets, `301 x 56` state labels, `conditioning_config={8:
      1.0}`, RGB-only visual input, clean captions without future endpoint
      text, and the DROID Policy checkpoint
      `checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID-DCP`. The earlier plain
      Nano, after-Nano DROID, immediate state-target, old matrix, old
      forward-dynamics, and 6/6 93-frame roots are invalid for active method
      use. The one older `state_targets/actions` root referenced by the clean
      JSONL is sidecar-label-only and must not be used directly for SFT; it is
      not a checkpoint/result continuation.
      Runtime poll at `2026-06-09T03:13+08:00`: clean DROID/Cosmos3-Nano
      SFT remains live in allocation `118609` / step `118609.41` on
      `server43`; it reached `iter 605`, wrote checkpoint
      `iter_000000600`, and validation improved from iter-300 loss
      `7.984733` to iter-600 loss `6.861973`. There is still no
      `sft_completed` or `sft_failed` marker. Action eval, task-state readout,
      forward-dynamics SFT, and controller remain waiting; no post-SFT demo,
      PSNR/readout, action metric, or controller evidence exists yet.
      Runtime poll at `2026-06-09T05:05+08:00`: the active clean full301 RGB
      DROID/Cosmos3-Nano SFT remains live in held allocation `118609` on
      `server43`; Slurm step `118609.41` is still running and the allocation
      has not been released. Training reached iteration `920`; checkpoint
      `iter_000000900` is present with model metadata, `latest_checkpoint.txt`
      points to `iter_000000900`, and validation improved across saved
      checkpoints from iter `300` loss `7.984733` to iter `600` loss
      `6.861973` to iter `900` loss `6.597537`. There is still no
      `sft_completed` or `sft_failed` marker, so this remains an intermediate
      training milestone, not world-model evidence. The strict action-eval
      watcher is still waiting for the clean `sft_completed` marker with
      `301` frames, `300` action steps, DROID tokenizer, and
      `allow_short_horizon_diagnostic=false`; readout waits for
      `post_sft_action_eval_completed`; forward-dynamics waits for the clean
      readout completion marker; matrix waits for the forward-dynamics chain
      completion marker. Refreshed `full301_wam_status_latest.{json,md}` and
      `full301_method_evidence_audit_latest.{json,md}` keep
      `world_model_evidence_complete=false` and
      `controller_should_remain_paused=true`. A static wrapper scan found old
      93-frame/after-Nano/immediate-state-target references only in
      refusal/audit guards or explicit diagnostic override paths; active
      defaults remain the clean full301 DROID chain. Follow-up allocation
      `120966` remains pending with `BeginTime=2026-06-09T08:29:42` for clean
      forward-dynamics continuation if the primary allocation becomes tight.
      Readout strictness hardening at `2026-06-09T05:12+08:00`: while the
      clean SFT continued, audited the downstream task-state readout path that
      decodes strict full301 Cosmos videos into external-target/controller
      state. `watch_cosmos3_action_eval_readout_controller.sh` already passes
      `READOUT_NUM_FRAMES=301` and `--require-exact-video-frames`, and
      `watch_cosmos3_readout_predict.sh` uses the same exact-frame setting.
      Patched `scripts/world_model/train_cosmos3_task_state_readout.py` so
      slot-label reads no longer clamp out-of-range frame indices to the last
      state frame; any requested target frame outside the H5 range now fails
      with `slot target frame range mismatch`. Patched
      `scripts/world_model/inspect_cosmos3_task_state_prediction.py` so
      strict readout inspection also rejects prediction JSON that was not
      produced with exact video-frame decode, and rejects reference spans that
      differ from the expected `301` frames. `py_compile` passed, a sampled
      clean-manifest smoke read `(301,25)` continuous and `(301,2)` binary
      labels, an intentional `302`-frame request failed closed, and a full
      `1000/1000` H5 slot-frame audit found every row has exactly `301` slot
      frames. This is downstream contract hardening only; it does not create
      post-SFT video/readout evidence before the clean SFT completes.
      Runtime-config audit at `2026-06-09T05:18+08:00`: extended
      `scripts/world_model/audit_cosmos3_full301_method_evidence.py` to
      require the active SFT runtime config itself, not only the JSONL
      contract. The refreshed audit checks that the live config loads
      `Cosmos3-Nano-Policy-DROID-DCP`, uses the DROID tokenizer, sets
      `model.config.action_gen=true`, uses clean-caption `ai_caption`,
      trains/validates with `num_video_frames=301`,
      `temporal_interval_mode=force_one`, `conditioning_config={8:1.0}`,
      and points train/val `jsonl_paths` at the clean full301 condition root.
      The audit item
      `clean_joint_policy_runtime_config_full301_droid_action_gen` is now
      `satisfied`; the method audit still intentionally reports
      `world_model_evidence_complete=false` because SFT completion and all
      post-SFT full301 action/video/readout/forward-dynamics/matrix evidence
      are not present yet. This is a guardrail fix only; it does not turn the
      current mid-training checkpoint into method evidence.
      Watcher/status refresh at `2026-06-09T05:20+08:00`: live tmux sessions
      are only the clean full301 DROID/Cosmos3-Nano chain plus read-only
      monitors. The action-eval watcher waits for the clean-caption
      `sft_completed` marker; readout waits for the clean
      `post_sft_action_eval_completed` marker; forward-dynamics waits for
      primary readout completion; the external-target matrix waits for the
      clean forward-dynamics completion marker. No old 93-frame, after-Nano,
      immediate-state-target, or controller watcher is active. Status refresh
      reports allocation `118609` still running on `server43`, step
      `118609.41` still live, SFT latest status `iter 956/1500`, latest
      validation `iter 900 loss 6.597537`, `sft_completed=false`,
      `sft_failed=false`, next checkpoint/validation `iter 1200`, and
      estimated completion still within the held allocation. Follow-up
      allocation `120966` remains pending for clean forward-dynamics
      continuation. The method audit remains incomplete only because final
      SFT/action-eval/readout/forward-dynamics/matrix artifacts do not exist
      yet.
      Status/resource refresh at `2026-06-09T05:22+08:00`: clean DROID/Cosmos3-
      Nano SFT is still live in allocation `118609` on `server43`, step
      `118609.41`, latest status `iter 961/1500`, latest validation still
      `iter 900 loss 6.597537`, no `sft_completed` or `sft_failed` marker,
      next checkpoint/validation `iter 1200`, and estimated final SFT
      completion still inside the held allocation. The strict action-eval
      manifest points only at the clean full301 DROID root with
      `num_frames=301`, `action_chunk_size=300`,
      `condition_frame_indexes_vision=0..7`,
      `allow_short_horizon_diagnostic=false`, and future-GT prompt text
      removed. The readout watcher is exact-frame full301 and has
      `ALLOW_ONE_SHOT_COSMOS_CONTROLLER_DIAGNOSTIC=false`, so it will write
      `readout_prediction_completed_controller_skipped` instead of launching
      a stale one-shot controller. The forward-dynamics watcher and matrix
      watcher are also waiting on clean ordered markers. Follow-up allocation
      `120966` is an H200 request (`gres:gpu:NVIDIAH200:1`) with an automatic
      handoff command, not an empty shell: after it starts it waits for
      `forward_dynamics_needs_new_allocation` or completion markers and then
      runs `watch_cosmos3_droid_forward_dynamics_after_policy_readout.sh` for
      the clean full301 forward-dynamics chain if needed.
      Live boundary refresh at `2026-06-09T05:26+08:00`: re-checked the user
      correction that any ManiSkill Cosmos3 SFT result with possible
      input/GT length mismatch or accidental truncation is untrusted and must
      not be continued. `AGENTS.md` already records this as a hard
      2026-06-09 boundary. The active allocation is still the fresh clean
      full301 RGB `Cosmos3-Nano-Policy-DROID-DCP` chain, not an old
      checkpoint/root: Slurm `118609` remains running on `server43`, step
      `118609.41` is live, the refreshed status helper reports SFT at
      `iter 983/1500`, latest validation remains iter `900` loss `6.597537`,
      and no `sft_completed` or `sft_failed` marker exists. The strict
      action-eval watcher is still
      waiting for the clean completion marker with `num_frames=301`,
      `action_chunk_size=300`, DROID tokenizer, explicit video-prefix frames
      `0..7`, `fps=30`, and `allow_short_horizon_diagnostic=false`; readout,
      forward-dynamics, and matrix watchers are still waiting on ordered
      clean markers. The evidence audit remains
      `world_model_evidence_complete=false` with old invalid roots marked by
      sentinels and the clean full301 data/runtime contract satisfied, but
      final SFT/action-eval/readout/forward-dynamics/matrix evidence
      incomplete. Follow-up allocation `120966` is a pending 1-H200 clean
      forward-dynamics handoff with an automatic wait-and-run command; it is
      not an old experiment. No controller has been launched.
      Event-trigger equal-length refresh prep at `2026-06-09T05:40+08:00`:
      audited the gap between full frame-0 action eval and the user-required
      "detect target motion then refresh/predict" interface. Existing
      `write_cosmos3_action_segment_sample.py` and
      `plan_cosmos3_receding_refresh_segments.py` treated any non-301 segment
      as short-horizon diagnostic, which prevents strict event-triggered spans
      inside a 301-frame episode. Patched them to support an explicit
      `--allow-variable-length-equal-gt-method` path with
      `--use-remaining-horizon`: each segment starts from a real observed
      prefix and runs to the real episode end, so predicted and GT lengths are
      still identical for that intended segment. The old arbitrary-short path
      remains refused unless `--allow-short-horizon-diagnostic` is explicitly
      set. Patched the receding script to pass variable segment lengths and
      readout frame counts when explicitly enabled; it still refuses
      controller execution by default and still requires full world-model
      evidence before controller use. Smoke on `hole_constant` val sample 0
      planned trigger/equal-GT segments `[16,52,64,76,88,100,112]` from
      source trigger step `80`; the first observed-motion refresh segment is
      `52..300`, `249` video frames and `248` action rows. A generated local
      sample verified `video_frames=249`, `action_shape=[248,32]`,
      `allow_short_horizon_diagnostic=false`,
      `allow_variable_length_equal_gt_method=true`, and
      `full_length_contract_ok=true`. A refusal smoke without the variable
      method flag exits `1`. This is CPU/local tooling readiness only; it has
      not run Cosmos inference, does not change the active SFT, and is not
      world-model evidence.
      Runtime poll at `2026-06-09T03:15+08:00`: clean SFT remains live at
      `iter 609`, latest validation remains iter-600 loss `6.861973`, and
      allocation `118609` still has about `51392s` remaining. The
      action-eval watcher is still waiting for the clean `sft_completed`
      marker with `num_frames=301`, `action_chunk_size=300`, and
      `allow_short_horizon_diagnostic=false`; it has not run on iter-600.
      The readout watcher is still waiting for
      `post_sft_action_eval_completed`. The clean forward-dynamics watcher is
      still waiting for the primary readout completion marker, and its
      manifest uses the clean roots with `301` frames / `300` actions plus a
      `43200s` walltime guard before starting the second full SFT. Follow-up
      allocation `120966` remains pending with BeginTime
      `2026-06-09T08:29:42`. Controller remains paused.
      Added read-only tmux status monitor
      `droid_policy_clean_caption_status_monitor_20260609` at
      `2026-06-09T03:16+08:00`. It refreshes
      `experiments/world_model_task_rebinding/cosmos3/full301_wam_status_latest.{json,md}`
      every `600s` and logs to
      `experiments/world_model_task_rebinding/cosmos3/full301_status_monitor_20260609/status.log`.
      It does not use GPU, start training, evaluate, or run controller; it
      exists only to keep the long clean SFT/post-SFT chain observable without
      releasing the held allocation.
      Evidence-audit hardening at `2026-06-09T03:18+08:00`: extended
      `scripts/world_model/audit_cosmos3_full301_method_evidence.py` so the
      required invalid-root sentinel check covers the superseded 6/6
      93-frame roots, early plain Nano root, after-Nano DROID roots,
      immediate contaminated roots, old matrix root, old readout root, and old
      forward-dynamics watcher roots. Added a separate required
      `sidecar_label_roots_are_not_direct_sft_roots` item for
      `action_state_conditions_full1000_maniskill_default_regen_full301_joint_policy_state_targets_20260608_after_nano_state_targets`,
      which is allowed only as equal-length action/state-label sidecars
      referenced by the clean JSONL, not as a direct SFT root. `py_compile`
      passed, and refreshed
      `experiments/world_model_task_rebinding/cosmos3/full301_method_evidence_audit_latest.{json,md}`
      reports those two guard items `satisfied` while correctly keeping the
      world-model evidence incomplete until final SFT/action-eval/readout/
      forward-dynamics/matrix artifacts exist.
      Runtime poll at `2026-06-09T03:20+08:00`: clean SFT remains live in
      allocation `118609` / step `118609.41` on `server43`; status helper
      reports latest iteration `624`, latest validation still iter-600 loss
      `6.861973`, `sft_completed=false`, and `sft_failed=false`. The action
      eval watcher is still waiting for final clean `sft_completed`; no
      prediction video, length precheck, action metric, readout prediction,
      forward-dynamics SFT, matrix diagnostic, or controller evidence exists
      yet. Follow-up allocation `120966` remains pending at BeginTime
      `2026-06-09T08:29:42`.
      Runtime poll at `2026-06-09T03:31+08:00`: clean SFT remains live in
      allocation `118609` / step `118609.41`; status helper reports latest
      iteration `657`, latest validation still iter-600 loss `6.861973`,
      `sft_completed=false`, and `sft_failed=false`. Next checkpoint and
      validation are iter `900` in about `243` iterations. Action eval remains
      waiting for the final clean completion marker, so there is still no
      trusted full301 prediction video, length precheck, action metric,
      readout, forward-dynamics, matrix, or controller evidence.
      Watcher race repair at `2026-06-09T03:36+08:00`: patched
      `scripts/slurm/run_cosmos3_full1000_sft_in_allocation.sh` so the SFT
      wrapper writes `val_loss_summary.json` before the final
      `sft_completed` marker, and patched
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh` so
      `CHECKPOINT_SELECTION=best_val` waits up to `300s` for that summary
      after `sft_completed` before selecting a checkpoint. `bash -n` passed
      for both scripts. Only the waiting tmux action-eval watcher was
      restarted; the live SFT step `118609.41` was not interrupted, and the
      allocation was not released. The restarted watcher remains in
      `waiting_for_sft_completed` with `num_frames=301`,
      `action_chunk_size=300`, `condition_frame_indexes_vision=0..7`, and
      `allow_short_horizon_diagnostic=false`. Current Slurm state: allocation
      `118609` still running on `server43`, follow-up allocation `120966`
      still pending for the clean forward-dynamics continuation, and no
      post-SFT demo/readout/controller evidence exists yet.
      Status/audit refresh at `2026-06-09T03:38+08:00`: refreshed
      `full301_wam_status_latest.{json,md}` and
      `full301_method_evidence_audit_latest.{json,md}`. The clean SFT is at
      latest iteration `675`, latest validation remains iter `600` loss
      `6.861973`, and `sft_completed=false` / `sft_failed=false`. The audit
      correctly reports `world_model_evidence_complete=false`: old invalid
      roots and the clean full301 data contract are satisfied, but joint-policy
      SFT completion, strict full301 action/video eval, external-target
      readout, forward-dynamics SFT/eval/readout, and moving-target matrix
      evidence remain incomplete. Controller remains paused.
      Short-horizon entry cleanup at `2026-06-09T03:43+08:00`: removed the
      remaining active-code `--total-video-frames 93` SFT entry points.
      `scripts/slurm/run_cosmos3_full1000_maniskill_render_sft_after_alloc.sh`
      now refuses unapproved full1000 regeneration by default, exports
      `TOTAL_VIDEO_FRAMES=301` / `REQUIRE_VIDEO_FRAMES=301`, verifies
      `300` action steps and `301` state-target frames, and passes
      `NUM_VIDEO_FRAMES=301`, `REQUIRE_STATE_TARGETS=true`,
      `STRICT_FULL_PREFLIGHT=true`, and `WAM_SFT_MODE` into SFT.
      `scripts/slurm/run_cosmos3_regen_metadata_action_sft.sbatch` now uses
      the same full301 export/SFT contract and defaults
      `RUN_POST_SFT_CONTROLLER_CHAIN=false`. The standalone refresh/sample
      helpers now default to full301, with short horizons requiring explicit
      diagnostic overrides. `bash -n` and `py_compile` passed. Smoke
      `experiments/world_model_task_rebinding/cosmos3/full1000_regen_guard_smoke_20260609_stderr.log`
      exits `20` before render/export/SFT when explicit regeneration approval
      is absent. Refreshed status shows the live clean SFT at iteration `691`,
      latest validation iter `600` loss `6.861973`, still no completion or
      failure marker; audit remains incomplete only because post-SFT artifacts
      do not exist yet.
      Pre-inference length guard at `2026-06-09T03:49+08:00`: audited the
      Cosmos action inference path. In `inference/action.py`, action-mode
      generation builds `target_frames = action_chunk_size + 1`, loads the
      reference video with `max_frames=action_chunk_size + 1`, and builds the
      sequence plan with `video_length=target_frames` / `action_length`.
      The VAE temporal mapping preserves `301` as `latent_frames=76` and
      `roundtrip_frames=301`; tokenizer `chunk_duration=93` is not accepted as
      an excuse for a short method artifact. Patched
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh` to write
      `pre_inference_length_contract.json` before any GPU `torchrun`, checking
      sample frames `301`, action chunk `300`, action sidecar `300 x 32`,
      state target `301` frames, config train/val `num_video_frames=301`,
      `temporal_interval_mode=force_one`, `model.config.action_gen=true`,
      condition frames `[0..7]`, and the VAE temporal round trip. Patched
      `scripts/world_model/inspect_cosmos3_action_eval_artifacts.py` and
      `scripts/world_model/audit_cosmos3_full301_method_evidence.py` so a
      strict action-eval artifact must include this pre-inference contract
      plus the post-inference frame-count precheck. `bash -n`/`py_compile`
      passed. The action-eval watcher was restarted at
      `2026-06-09T03:47:56+08:00` and is waiting for `sft_completed` with
      `301`/`300`/`allow_short_horizon_diagnostic=false`. Smoke inspection of
      the old 6/6 93-frame action-eval root still exits `1`, now also
      reporting `missing_pre_inference_length_contract`. Config smoke reports
      `yaml_ok 301 True`. Refreshed status shows the live SFT at iteration
      `705`, latest validation iter `600` loss `6.861973`, and still no
      post-SFT method evidence.
      User boundary re-confirmed at `2026-06-09T03:52+08:00`: the old
      ManiSkill Cosmos3 world-model SFT/action-eval/controller chain remains
      untrusted because input video, generated video, GT video, action
      targets, or readout targets may have been unequal length or accidentally
      truncated. Do not continue those checkpoints or derived controller
      inputs. The admissible path is still a fresh equal-length RGB-only
      ManiSkill world-model SFT using `Cosmos3-Nano-Policy-DROID` or, if
      DROID concretely fails, a fresh `Cosmos3-Nano` constrained baseline.
      Runtime status at this boundary: allocation `118609` remains running on
      `server43`, active clean SFT step `118609.41` reached iteration `717`,
      latest validation remains iter `600` loss `6.861973`, and no
      `sft_completed` / `sft_failed` marker or post-SFT demo/action/readout
      evidence exists yet. Controller remains paused until strict full301
      world-model evidence exists.
      Watcher hygiene at `2026-06-09T03:57+08:00`: inspected the live
      action-eval, readout, forward-dynamics, and external-target matrix
      watchers. The primary action-eval watcher is still waiting for the
      clean `sft_completed` marker with `301` frames, `300` action steps, and
      no short-horizon diagnostic override. The readout and forward-dynamics
      watchers are still waiting on the correct ordered markers and do not run
      controller. The matrix watcher had an early outer-`tee` directory-order
      warning, so `scripts/slurm/watch_cosmos3_full301_action_readout_matrix_after_main.sh`
      now attaches its own `${MATRIX_ROOT}/watch.log` after creating the
      matrix directories. `bash -n` passed, the matrix watcher was restarted
      in tmux `droid_policy_full301_external_target_matrix_after_core_20260609`,
      and it is waiting for
      `droid_policy_forward_dynamics_after_clean_policy_readout_20260608/forward_dynamics_chain_completed`.
      Refreshed status reports the clean SFT at iteration `730`, latest
      validation iter `600` loss `6.861973`, no completion/failure marker,
      allocation `118609` still running on `server43`, and follow-up
      allocation `120966` still pending. Full method evidence remains
      incomplete because no full301 generated video/action metric/readout/
      forward-dynamics/matrix artifacts exist yet.
      Motion/readout/action readiness at `2026-06-09T04:01+08:00`: refreshed
      the full clean contract audit and added
      `full301_motion_readout_action_readiness_latest.{json,md}` under the
      clean condition root. The readiness audit is data/evaluation-target
      evidence only, not model success: train/val remain exact `912/88`,
      every row has `301` video frames, `300 x 32` action labels, and
      `301 x 56` state/readout labels. It also verifies the objective-specific
      labels exist: train has `454` moving-target rows and val has `47`;
      active primary eval sample `val index 0`
      `hole_constant_seed702000_n167_traj_0_traj_0` is a moving-target sample
      with target-hole motion onset frame `84`, final hole displacement
      `0.132002m`, final hole xyz
      `[0.02204445, 0.38065955, 0.13779245]`, final peg-head-at-hole xyz
      `[-0.11419287, 0.00261620, -0.00564982]`, action shape `[300, 32]`,
      and state target shape `[301, 56]`. Matrix-selected val samples cover
      `hole_constant`, `hole_reverse`, `hole_move_stop`, and `none`. Patched
      `scripts/world_model/audit_cosmos3_full301_method_evidence.py` so the
      full-chain audit now requires this readiness report as
      `clean_full301_motion_readout_action_readiness`; `py_compile` passed,
      and refreshed audit marks it satisfied while correctly keeping method
      evidence incomplete until strict post-SFT full301 video/action/readout/
      forward-dynamics/matrix artifacts exist. Live SFT status at refresh:
      iteration `743`, latest validation iter `600` loss `6.861973`, no
      completion/failure marker, controller still paused.
      Forward-dynamics interface smoke at `2026-06-09T04:06+08:00`: patched
      `scripts/world_model/audit_cosmos3_wam_full301_contract.py` so the
      contract audit explains different boundaries for `policy` and
      `forward_dynamics` mode. The forward-dynamics mode conditions on the
      short video prefix plus the full `300`-row recorded/candidate
      action-state sequence, trains future RGB prediction, and keeps the
      `301`-frame state sidecar as readout/evaluation supervision rather
      than privileged visual/text conditioning. A local 4-row smoke export at
      `experiments/world_model_task_rebinding/cosmos3/forward_dynamics_export_smoke_20260609_0403`
      passed strict contract audit with train/val `2/2`, `301` video frames,
      `300 x 32` action rows, `301 x 56` state/readout labels, vision
      condition frames `[0..7]`, and action condition frames `[0..299]`.
      The root contains `DIAGNOSTIC_SMOKE_NOT_METHOD_EVIDENCE.md` and must
      not be treated as full1000 data, SFT output, controller input, or method
      evidence. Its only purpose is to catch length/conditioning mistakes
      before the ordered watcher generates the full forward-dynamics data
      after the primary readout completes.
      Controller/dryrun guard hardening at `2026-06-09T04:13+08:00`:
      static reference scan found that
      `scripts/slurm/run_cosmos3_receding_teacher_forced_controller_retest.sbatch`
      still defaulted to the old 6/6 action-condition, SFT, and readout
      roots. Patched it to default to the clean full301 DROID/Cosmos roots,
      the DROID tokenizer, and
      `experiments/world_model_task_rebinding/cosmos3/full301_method_evidence_audit_latest.json`;
      with `REQUIRE_WORLD_MODEL_EVIDENCE_COMPLETE=true` it now refuses
      controller retest until the strict full301 world-model evidence audit is
      complete. `bash -n` passed. Lightweight tests confirm default refusal
      exits `30`, and even with `ALLOW_RECEDING_CONTROLLER_DIAGNOSTIC=true`
      it exits `32` while the required incomplete items remain
      clean SFT/action-eval/readout/forward-dynamics/matrix evidence. Also
      patched `scripts/slurm/install_cosmos3_train_env_after_torch.sh` so its
      dryrun defaults no longer point at dirty `sft_dataset_full1000_rgbd` or
      plain Nano; it now points at the clean full301 RGB DROID condition root
      and DROID-DCP checkpoint. `bash -n` passed. The remaining old-root
      references under `scripts/` are only explicit refusal/audit guards.
      Post-SFT action metric parser hardening at
      `2026-06-09T04:19+08:00`: while clean SFT continued, audited
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh` and
      downstream inspectors. The action-eval path already refuses short
      generated videos before reconstruction/readout and requires policy-mode
      predicted actions to match the full `300`-step target. Patched the
      action-metric parser to load action sidecars in either direct-list form
      or `{"action": ...}` form, matching the existing pre-inference
      contract reader. This prevents a wrapper-format false failure after SFT
      completion without relaxing any evidence standard. `bash -n` passed, and
      a local `/tmp` smoke using current val sample
      `hole_constant_seed702000_n167_traj_0_traj_0` plus simulated predicted
      action verified `pred_shape=[300,32]`, `target_shape=[300,32]`, and
      `rmse=0.0`.
      Watcher default cleanup at `2026-06-09T04:25+08:00`: patched active
      post-SFT watcher defaults so direct invocations no longer drift back to
      old plain-Nano or immediate-state roots. The action-eval watcher now
      defaults to the clean full301 DROID joint-policy condition/SFT/job-dir,
      DROID tokenizer, and clean action-eval root. The readout watcher now
      defaults its output to the clean DROID full301 readout root. The
      forward-dynamics watcher now defaults to the clean primary readout root,
      clean forward-dynamics condition/SFT/action-eval/readout roots, and the
      active clean watch root. Generic full301 SFT wrappers now default to
      `Cosmos3-Nano-Policy-DROID-DCP` / DROID tokenizer and `BACKBONE=droid_policy`;
      `BACKBONE=nano` remains available only as an explicit fallback. The old
      after-Nano handoff watcher now refuses by default with exit `33`, and
      the old I2V reconstruction-triggered allocation/SFT watcher refuses by
      default with exit `34`. V2V eval defaults were also moved to clean
      full301 DROID roots. `bash -n` passed for the touched wrappers; refusal
      smoke tests passed for after-Nano and old I2V allocation watchers. The
      remaining old/default strings are either in explicit Nano fallback
      branches, guarded I2V diagnostic code, or invalid-root refusal/audit
      guards, not active defaults.
      Iter-900 training milestone at `2026-06-09T05:01+08:00`: clean
      full301 DROID/Cosmos SFT remained live in allocation `118609` on
      `server43` and wrote checkpoint
      `iter_000000900` with `model/.metadata`; `latest_checkpoint.txt` now
      points to `iter_000000900`. Validation loss improved again:
      iter `300` was `7.984733`, iter `600` was `6.861973`, and iter `900`
      is `6.597537`. Training continued past iter `908` with no
      `sft_completed` or `sft_failed` marker. This is only an intermediate
      training milestone: the strict full-chain audit remains incomplete
      because completed SFT, full301 generated video/action eval,
      external-target readout, forward-dynamics SFT/eval/readout, and the
      moving-target matrix do not exist yet. The action-eval/readout watchers
      remain waiting for the final clean completion markers; controller
      remains paused.

- [ ] 2026-06-09 current ManiSkill WAM boundary:
      old ManiSkill Cosmos3 SFT/action-eval/controller outputs remain stopped
      and invalid. Do not continue any checkpoint/result that can have
      input-vs-GT length mismatch, accidental truncation, or the old 93-frame
      action-eval contract. The only active ManiSkill world-model training
      chain is the fresh clean-caption full301 RGB `Cosmos3-Nano-Policy-DROID`
      joint-policy SFT under
      `experiments/world_model_task_rebinding/cosmos3/sft_full1000_maniskill_rgb_full301_joint_policy_droid_policy_clean_caption_20260608`,
      with condition root
      `experiments/world_model_task_rebinding/cosmos3/action_state_conditions_full1000_maniskill_default_regen_full301_joint_policy_clean_caption_20260608`.
      Current resource status at `2026-06-09T00:04+08:00`: Slurm allocation
      `118609` is still running on `server43`, active step `118609.41` is live
      inside tmux `droid_policy_clean_caption_joint_policy_20260608`, H200
      utilization is `100%` with about `105025 MiB / 143771 MiB` used, and
      `run.log`/`sft_train.log` were modified at `00:04`. There is still no
      `sft_completed`, `sft_failed`, `val_loss_summary.json`,
      `latest_checkpoint.txt`, or checkpoint `.metadata`; this means no new
      post-SFT method evidence exists yet. The strict action-eval watcher,
      exact-frame readout watcher, and forward-dynamics watcher remain waiting
      for the clean SFT chain; no controller is running.
      Corrected full-row audit at `2026-06-09T00:08+08:00` checked all
      `1000` clean JSONL rows plus reused action/state sidecars: train/val
      `912/88`, `model_mode=policy`, action targets `300 x 32`, state targets
      `301 x 56`, top-level vision condition frames `[0..7]`,
      action-condition frames `[]`, missing action/state files `0`, future
      caption-pattern hits `0`, and future `task_state_condition` metadata
      hits `0`. The saved Cosmos config also has `conditioning_config={8:
      1.0}`, `num_video_frames=301`, and `temporal_interval_mode=force_one`.
      This verifies the current clean SFT input contract; it is still
      component training, not method evidence until post-SFT full301 artifacts
      pass strict inspection.
      Script hardening at `2026-06-09T00:13+08:00`: added
      `INVALID_DO_NOT_USE_20260609.md` root refusal checks to
      `scripts/slurm/run_cosmos3_full1000_sft_in_allocation.sh`,
      `scripts/slurm/run_cosmos3_maniskill_full301_wam_sft_in_allocation.sh`,
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh`,
      `scripts/slurm/watch_cosmos3_action_eval_readout_controller.sh`,
      `scripts/slurm/watch_cosmos3_droid_forward_dynamics_after_policy_readout.sh`,
      and `scripts/slurm/watch_cosmos3_readout_predict.sh`. `bash -n` passed
      for all six scripts, and a lightweight refusal test confirmed the old
      invalid 93-frame condition root exits with code `24` before waiting or
      inference. Runtime liveness at `00:13`: allocation `118609` / step
      `118609.41` is still live on `server43`; H200 utilization is `100%` with
      about `105025 MiB / 143771 MiB` used. Still no checkpoint/completion
      marker exists, so post-SFT evidence remains pending.
      Action-inference contract audit at `2026-06-09T00:15+08:00`: the
      post-SFT action-eval sample writer passes `action_chunk_size=300`,
      `num_frames=301`, and `condition_frame_indexes_vision=[0..7]` and
      refuses any sample where the reference video frame count is not `301`.
      Local Cosmos action-mode inference uses `target_frames =
      action_chunk_size + 1`, so the planned action-mode prediction target is
      `301` frames, not the old 93-frame path. Domain mapping confirms
      `maniskill_peg_insertion` has `raw_action_dim=32` and domain id `21`,
      matching the clean train target. This is a code-path audit; final proof
      still requires the post-SFT generated MP4 and action JSON to pass the
      strict inspector.
      Runtime/watch correction at `2026-06-09T00:22+08:00`: allocation
      `118609` remains running on `server43` and is the only active user
      Slurm job; no old invalid Cosmos/controller jobs are queued or running.
      In-allocation `nvidia-smi` reports the H200 at `100%` utilization with
      about `105025 MiB / 143771 MiB` used. The clean DROID SFT step
      `118609.41` reached iteration `137` with recent loss about `10.9996`,
      still before the first `SAVE_ITER=300` / `VALIDATION_ITER=300`
      checkpoint. The action-eval watcher remains waiting for `sft_completed`.
      The readout watcher initially exited on a transient `.venv` torch import
      failure, but a direct preflight of `.venv/bin/python` passed for
      `h5py`, `imageio`, `torch`, and `tyro`; the watcher was relaunched in
      tmux `droid_policy_clean_caption_task_state_readout_20260608`, passed
      `readout_python_preflight_ok`, and is now waiting for
      `post_sft_action_eval_completed`. The forward-dynamics watcher remains
      waiting for primary readout completion. No controller is running and no
      new full301 method evidence exists yet.
      Length-contract hardening at `2026-06-09T00:28+08:00`: added an
      immediate post-inference `length_contract_precheck.json` in
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh`. It counts
      the generated `vision.mp4` before PSNR/readout/completion and hard-fails
      unless `predicted_frames == reference_video_frames`. Updated
      `scripts/world_model/inspect_cosmos3_action_eval_artifacts.py` so strict
      artifact inspection requires that precheck and fails if it is missing or
      false. The old 6/6 93-frame action-eval root was smoke-inspected under
      `experiments/world_model_task_rebinding/cosmos3/strict_length_inspector_smoke_20260609`
      and correctly exits `1`, with failures including missing full301
      reference contract, missing length precheck, old all-frame length
      mismatch, and no-truncation false. Added invalid-root refusal to the
      optional full301 matrix watcher. Syntax/compile checks passed for the
      patched scripts. Runtime liveness at `00:28`: allocation `118609` is
      still running on `server43`, clean DROID SFT step `118609.41` reached
      iteration `154`, and the action-eval/readout/forward-dynamics watchers
      remain live. No controller is running.
      External-target readout hardening at `2026-06-09T00:32+08:00`: patched
      `scripts/world_model/train_cosmos3_task_state_readout.py` so prediction
      metrics validate the reference H5 frame range instead of relying on
      clamped slot indices. Patched
      `scripts/world_model/inspect_cosmos3_task_state_prediction.py` to expose
      strict readout failures for expected frame count, finite predictions,
      required reference metrics, and required external-target event metrics.
      Patched `scripts/slurm/watch_cosmos3_readout_predict.sh` so the active
      readout prediction requires `301` frames, finite values, reference
      metrics, and event metrics for motion onset, final hole position, and
      insertion geometry. `bash -n`/`py_compile` passed, and strict perfect-GT
      smoke
      `experiments/world_model_task_rebinding/cosmos3/task_state_readout_event_metric_smoke_20260608/inspection_strict_20260609.json`
      reports `strict_readout_ok=true`, `num_frames=301`, motion onset frame
      `84`, zero onset error, and zero final hole / insertion-geometry error.
      Runtime poll at `00:31`: allocation `118609` remains held, clean DROID
      SFT reached iteration `164`, and no checkpoint/completion marker exists
      yet. No controller is running.
      Full301 WAM data-contract audit at `2026-06-09T00:37+08:00`: added
      `scripts/world_model/audit_cosmos3_wam_full301_contract.py` and ran it
      on the active clean joint-policy root. Output
      `experiments/world_model_task_rebinding/cosmos3/action_state_conditions_full1000_maniskill_default_regen_full301_joint_policy_clean_caption_20260608/full301_wam_contract_audit_20260609.json`
      reports `strict_contract_ok=true`, failures `[]`, train/val/total
      `912/88/1000`, all `model_mode=policy`, all videos `301` frames, all
      action sidecars `300 x 32`, all state-target sidecars `301 x 56`,
      vision condition frames `[0..7]`, empty action condition frames,
      future caption hits `0`, future metadata hits `0`, invalid marker hits
      `0`, and finite sidecars. The audit explicitly records the method
      boundary: current joint-policy DROID/Cosmos SFT directly trains future
      RGB plus robot action chunk from video prefix and causal prefix
      task-state text/metadata; `state_target_path` is label/readout
      supervision and evaluation target, not a privileged Cosmos input or
      direct state head. The paired forward-dynamics SFT queued after primary
      readout is the action-conditioned side that conditions on full
      recorded/candidate action rows and predicts future RGB/task state
      through the strict video readout. Patched
      `scripts/slurm/run_cosmos3_full1000_sft_in_allocation.sh` to invoke
      this audit automatically for strict action-conditioned full301 SFT
      launches, including the future forward-dynamics chain. `bash -n` and
      `py_compile` passed. Runtime poll at `00:37`: allocation `118609`
      remains held, clean DROID SFT reached iteration `181`, and no
      checkpoint/completion marker exists yet. No controller is running.
      Runtime poll at `2026-06-09T00:46+08:00`: allocation `118609`
      remains running on `server43`; active clean DROID/Cosmos3-Nano SFT step
      `118609.41` reached iteration `207` at about `20.6s/iter`, still before
      the first `SAVE_ITER=300` / `VALIDATION_ITER=300` checkpoint. There is
      still no `sft_completed`, `sft_failed`, `val_loss_summary.json`,
      `latest_checkpoint.txt`, or checkpoint `.metadata`; therefore no new
      post-SFT full301 demo, PSNR, action metric, readout, or controller
      evidence exists yet. A manual forward-dynamics condition export overlap
      step `118609.54` was cancelled after about five minutes because it had
      produced no files or manifest and was only a foreground sidecar export
      attempt; the held H200 allocation and training step were not cancelled.
      The existing forward-dynamics watcher remains live and will run the
      action-conditioned full301 side in order after the primary joint-policy
      readout completion marker. No controller is running.
      Forward-dynamics interface preflight at `2026-06-09T00:50+08:00`:
      patched `scripts/world_model/export_cosmos3_maniskill_action_conditions.py`
      to print observable export progress and patched
      `scripts/slurm/run_cosmos3_maniskill_full301_wam_sft_in_allocation.sh`
      to pass `--progress-every 25` for future full1000 condition exports.
      Syntax/compile checks passed. A local two-record smoke root
      `experiments/world_model_task_rebinding/cosmos3/action_state_conditions_forward_dynamics_progress_smoke_20260609`
      exported in about `5.6s` with progress logs, train/val `1/1`,
      `model_mode=forward_dynamics`, video frames `301`, action condition
      rows `300 x 32`, state targets `301 x 56`, action-condition frame
      indexes `[0..299]`, vision condition `[0..7]`, future caption hits `0`,
      and future metadata hits `0`. Strict audit output
      `.../full301_wam_contract_audit.json` reports
      `strict_contract_ok=true`. This verifies the action-conditioned WAM
      sidecar structure for the later forward-dynamics chain; it is not
      post-SFT model evidence. Runtime poll at `00:50`: allocation `118609`
      remains held, clean DROID SFT step `118609.41` reached iteration `220`,
      no checkpoint/completion marker exists yet, and no controller is
      running.
      SFT config audit at `2026-06-09T00:52+08:00`: active saved config
      `.../outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_full301_joint_policy_clean_caption/config.yaml`
      has `model.config.action_gen=true`, `action_loss_weight=10.0`,
      train/val `num_video_frames=301`, `temporal_interval_mode=force_one`,
      `max_action_dim=64`, `conditioning_config={8:1.0}`, and train/val JSONL
      paths under the clean full301 joint-policy condition root. The SFT
      manifest full preflight already checked all train/val rows (`912/88`)
      with video frames `[301]`, action lengths `[300]`, action dims `[32]`,
      state lengths `[301]`, and state dims `[56]`. This confirms the active
      training process is an equal-length full301 action-generating WAM SFT
      from the DROID backbone; final proof still requires post-SFT generated
      artifacts to pass the same strict length checks.
      Post-SFT eval guard audit at `2026-06-09T00:54+08:00`: inspected
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh` and local
      Cosmos inference docs/code. The sample writer refuses method eval unless
      `NUM_FRAMES` equals the reference video frame count and
      `ACTION_CHUNK_SIZE` equals `reference_frames - 1`; the active values are
      `301` and `300`. Cosmos docs list the 256p max video length as `400`
      frames, and `301` stays unchanged under the temporal compression
      rounding formula, so the requested full301 inference is within the
      framework's stated 256p range. The watcher then writes
      `length_contract_precheck.json` and hard-fails if the generated MP4 is
      not exactly `301` frames before PSNR/readout/completion. Policy action
      metrics also require predicted action length `300` and a dimension that
      covers the `300 x 32` target. Patched the old
      `scripts/world_model/write_cosmos3_action_segment_sample.py` so its
      previous 93-frame trigger-aligned diagnostic default now raises unless
      `--allow-short-horizon-diagnostic` is explicitly supplied; py_compile
      passed and smoke output
      `experiments/world_model_task_rebinding/cosmos3/short_horizon_guard_smoke_20260609_stderr.log`
      shows the default 93-frame path is refused. Runtime poll at `00:53`:
      allocation `118609` remains held, clean SFT reached iteration `228`,
      and no checkpoint/completion marker exists yet.
      Readout-predict path guard at `2026-06-09T00:57+08:00`: removed the old
      default action-eval root and default `PREDICTION_VIDEO` from
      `scripts/slurm/watch_cosmos3_readout_predict.sh`. The script now
      requires explicit `DATASET_MANIFEST`, `ACTION_EVAL_ROOT`,
      `PREDICTION_VIDEO`, and `SAMPLE_MANIFEST`, and checks those files before
      waiting for a readout model. `bash -n` passed. Smoke
      `experiments/world_model_task_rebinding/cosmos3/readout_predict_required_path_smoke_20260609_stderr.log`
      exits early with `missing_sample_manifest=...`, confirming the script no
      longer falls back to the old 93-frame path. Runtime poll at `00:56`:
      allocation `118609` remains held, clean SFT reached iteration `237`,
      no checkpoint/completion marker exists yet, and no controller is
      running.
      Future SFT failure-marker guard at `2026-06-09T00:59+08:00`: patched
      `scripts/slurm/run_cosmos3_full1000_sft_in_allocation.sh` to write
      `${OUTPUT_ROOT}/sft_failed` if the wrapper exits before
      `sft_completed`, and patched
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh` to stop
      with exit `31` when that marker exists instead of waiting forever.
      `bash -n` passed, and smoke
      `experiments/world_model_task_rebinding/cosmos3/sft_failed_wait_guard_smoke_20260609/stderr.log`
      confirms the action-eval watcher prints the marker and exits `31`.
      This mainly protects later forward-dynamics SFT and future launches;
      the already-running joint-policy SFT process was not restarted or
      interrupted. Runtime poll at `00:58`: allocation `118609` remains held,
      clean SFT reached iteration `243`, no checkpoint/completion/failure
      marker exists yet, and no controller is running.
      Live current-run failure monitor at `2026-06-09T01:00+08:00`: because
      the just-added `sft_failed` trap is not inherited by the already-running
      joint-policy shell, started tmux
      `droid_policy_clean_caption_sft_failure_monitor_20260609`. It only
      monitors Slurm step `118609.41` and markers under the active SFT root,
      with a `180s` grace window after the step disappears before writing
      `sft_failed`; it does not use GPU, train, evaluate, or run controller.
      Monitor manifest/log:
      `experiments/world_model_task_rebinding/cosmos3/sft_full1000_maniskill_rgb_full301_joint_policy_droid_policy_clean_caption_20260608/live_failure_monitor/`.
      Initial log reports `step_live step=118609.41`. Runtime poll at
      `01:00`: allocation `118609` remains held, clean SFT reached iteration
      `249`, no checkpoint/completion/failure marker exists yet, and the
      action-eval/readout/forward-dynamics watchers remain waiting on the
      strict full301 chain.
      Readiness status helper at `2026-06-09T01:02+08:00`: added
      `scripts/world_model/report_cosmos3_full301_wam_status.py`, a read-only
      reporter for the active full301 WAM chain. It checks SFT
      completion/failure markers, latest train/validation log entries,
      checkpoint metadata, strict action-eval length/action artifacts,
      readout completion, and forward-dynamics SFT markers without launching
      Slurm, GPU work, inference, or controller. `py_compile` passed. Current
      report output:
      `experiments/world_model_task_rebinding/cosmos3/full301_wam_status_20260609_0102.json`
      and `.md`. It reports latest SFT iteration `257`, latest validation
      still iteration `0` with val loss `16.758268`, no checkpoint directory,
      no `sft_completed`, no `sft_failed`, no action-eval sample/length
      precheck, no readout, and no forward-dynamics SFT. Therefore
      `ready_for_action_eval=false`, `ready_for_readout=false`, and
      `ready_for_forward_dynamics=false`. This is a readiness/status artifact
      only, not model evidence.
      Runtime/status poll at `2026-06-09T01:05+08:00`: allocation `118609`
      remains held on `server43`, active step `118609.41` remains live, clean
      SFT reached iteration `261` with latest visible loss `8.0814`, and
      there is still no checkpoint directory, `latest_checkpoint.txt`,
      checkpoint `.metadata`, `sft_completed`, `sft_failed`, or
      `val_loss_summary.json`. Refreshed read-only status report:
      `experiments/world_model_task_rebinding/cosmos3/full301_wam_status_latest.json`
      and `.md`; it still reports latest validation only at iteration `0`,
      no action-eval sample/length precheck, no readout, no forward-dynamics
      SFT, and all readiness flags false. The action-eval watcher is waiting
      for `sft_completed`, the readout watcher is waiting for strict
      `post_sft_action_eval_completed`, the forward-dynamics watcher is
      waiting for primary readout completion, and the live failure monitor is
      logging `step_live step=118609.41`. No controller is running.
      User boundary re-confirmed at `2026-06-09T01:08+08:00`: because the old
      ManiSkill Cosmos3 SFT/action-eval results may have input/GT length
      mismatch or accidental truncation, they must not be continued. The only
      admissible current ManiSkill WM training is a fresh DROID Policy or
      fresh Nano full301 RGB SFT with strict equal-length accounting. Current
      active run satisfies the allowed class: `Cosmos3-Nano-Policy-DROID`,
      clean-caption `joint_policy`, RGB-only, `num_video_frames=301`,
      `action_chunk_size=300`, vision prefix `[0..7]`, and no depth input.
      Re-run full1000 audit
      `experiments/world_model_task_rebinding/cosmos3/action_state_conditions_full1000_maniskill_default_regen_full301_joint_policy_clean_caption_20260608/full301_wam_contract_audit_latest.json`
      reports `strict_contract_ok=true`, failures `[]`, train/val/total
      `912/88/1000`, video frames `301`, action targets `300 x 32`, state
      targets `301 x 56`, future caption hits `0`, future metadata hits `0`,
      and invalid marker hits `0`. Old invalid roots have
      `INVALID_DO_NOT_USE_20260609.md` markers. Runtime poll at `01:08`:
      allocation `118609` is still the only active Slurm job, running on
      `server43`; step `118609.41` is live and the SFT log reached iteration
      `272` at about `20.7s/iter`. There is still no checkpoint/completion/
      failure marker, no action-eval artifact, no readout, no forward-dynamics
      SFT, and no controller evidence.
      First checkpoint status at `2026-06-09T01:22+08:00`: the active clean
      DROID SFT wrote
      `.../checkpoints/iter_000000300/model/.metadata` and
      `latest_checkpoint.txt=iter_000000300`; validation at iteration `300`
      completed with val loss `7.984733`, down from the iteration-0 val loss
      `16.758268`. Allocation `118609` and step `118609.41` remain live on
      `server43`. This is encouraging SFT progress for the fresh full301
      chain, but still not method evidence: no `sft_completed`, no generated
      301-frame prediction, no 300-step action metric, no readout, no
      forward-dynamics result, and no controller evidence exist yet.
      Watcher reload at `2026-06-09T01:25+08:00`: the action-eval, readout,
      and forward-dynamics watcher tmux sessions were restarted without
      touching allocation `118609`, training step `118609.41`, or the failure
      monitor, because the original waiting shells were created before the
      later strict length-precheck hardening. New watcher creation time is
      `01:25`, and the active action-eval manifest again records
      `num_frames=301`, `action_chunk_size=300`, condition frames `[0..7]`,
      `allow_short_horizon_diagnostic=false`, and `checkpoint_selection=best_val`.
      Script checks passed for the watcher/readout scripts and status/
      inspection helpers. Runtime status after reload: SFT reached iteration
      `311` with latest train loss `6.7787`; latest validation remains
      iteration `300` with val loss `7.984733`; no `sft_completed`,
      `sft_failed`, action-eval length precheck, readout, forward-dynamics
      result, or controller evidence exists yet.
      Completion handoff audit at `2026-06-09T01:27+08:00`: the active
      ManiSkill wrapper delegates to `run_cosmos3_full1000_sft_in_allocation.sh`,
      whose normal path writes `${OUTPUT_ROOT}/sft_completed` after the SFT
      `srun/torchrun` returns. The already-running shell may not inherit the
      later `sft_failed` trap patch, so the separate live failure monitor
      remains necessary and continues to report `step_live step=118609.41`.
      Latest visible SFT iteration at this audit was `313`; no post-SFT
      method artifact exists yet.
      External-target label audit at `2026-06-09T01:28+08:00`: added/read
      audit output
      `experiments/world_model_task_rebinding/cosmos3/external_target_label_audit_20260609/summary.json`.
      It checks the active clean full301 condition root's state-target labels,
      not model outputs. Result: `strict_external_target_label_ok=true`,
      failures `[]`, train/val rows `912/88`, no bad state-target rows, all
      audited rows have `301` state frames and `300` action steps. Validation
      set has `47/88` rows with target-hole motion, onset frames `68..91`
      with mean `87.319`, final hole displacement mean `0.060523m` and max
      `0.139999m`, and final peg-head-to-hole distance labels are present
      for insertion geometry. This confirms the post-SFT readout can evaluate
      the user's external-target requirement: detect hole motion, estimate
      final target-hole position, and estimate insertion geometry from the
      generated full301 video. It is label/readout readiness evidence only,
      not model quality or controller evidence.
      Reproducibility update: added
      `scripts/world_model/audit_cosmos3_external_target_labels.py` and reran
      the audit with `--strict`; `py_compile` passed and the script reproduced
      the same `strict_external_target_label_ok=true` result. This is now a
      reusable preflight for any future full301 condition root before claiming
      external-target readout evidence.
      Action-target audit at `2026-06-09T01:34+08:00`: added
      `scripts/world_model/audit_cosmos3_action_targets.py` and ran it on the
      same active condition root. Output
      `experiments/world_model_task_rebinding/cosmos3/action_target_audit_20260609/summary.json`
      reports `strict_action_target_ok=true`, failures `[]`, train/val
      action shapes `300x32` for `912/88` rows, no bad rows, and all train/val
      rows have nonzero robot action, robot action variation, and time-fraction
      variation. Validation robot-action absolute mean is `0.689190`, mean
      per-row robot-action std is `0.882922`, and max normalized robot-action
      magnitude is `12.986966`. This verifies the joint WAM action target is
      complete and non-degenerate before post-SFT action prediction metrics.
      It is action-label readiness evidence only, not model quality or
      controller evidence.
      Future-SFT preflight hardening at `2026-06-09T01:35+08:00`: patched
      `scripts/slurm/run_cosmos3_full1000_sft_in_allocation.sh` so future
      strict action-conditioned full301 SFT launches automatically run the
      full contract audit, the external-target label audit, and the action
      target audit before training. `bash -n` passed for the wrapper and
      `py_compile` passed for both audit scripts. This does not affect the
      already-running joint-policy SFT, which continues in step `118609.41`;
      latest visible iteration during this check was `338`.
      Runtime/user-boundary poll at `2026-06-09T01:39+08:00`: re-confirmed
      that `AGENTS.md` already carries the 2026-06-09 rule that old
      length-mismatched ManiSkill SFT/checkpoint/controller outputs must not
      be continued. The held allocation `118609` remains running on
      `server43`; tmux sessions are the clean DROID full301 SFT, strict
      action-eval watcher, exact-frame readout watcher, forward-dynamics
      watcher, failure monitor, and allocation shell. The active SFT log
      reached iteration `348` with latest loss `7.3886`; latest validation is
      iteration `300` with val loss `7.984733`, and
      `latest_checkpoint.txt=iter_000000300` with model metadata exists.
      There is still no `sft_completed`, no `sft_failed`, no 301-frame
      generated MP4, no 300-step action metric, no readout prediction, no
      forward-dynamics SFT result, and no controller evidence. The correct
      next action is to keep the current DROID/Cosmos3-Nano full301 SFT
      running in the held allocation and let the strict post-SFT watcher
      reject any non-301 prediction before metrics.
      Runtime/watcher poll at `2026-06-09T01:42+08:00`: allocation `118609`
      remains running on `server43`; clean DROID SFT log reached iteration
      `356` with latest loss `7.3308`, latest validation remains iteration
      `300` with val loss `7.984733`, and only checkpoint
      `iter_000000300` exists. The clean action-eval root contains only
      `post_sft_action_eval_manifest.txt` and watcher logs, with no generated
      MP4, no `length_contract_precheck.json`, no action metrics, and no
      completion marker. The action-eval watcher is waiting for the clean SFT
      `sft_completed`; readout is waiting for clean post-SFT action eval; and
      forward-dynamics is waiting for primary readout. The immediate-state
      DROID SFT/action-eval/readout roots have `INVALID_DO_NOT_USE_20260609.md`
      sentinels, and the lower 2026-06-08 handoff item was annotated as
      historical so its stale `118609.35` wording cannot be mistaken for the
      active run.
      Status-helper ETA update at `2026-06-09T01:44+08:00`: patched
      `scripts/world_model/report_cosmos3_full301_wam_status.py` to report
      `max_iter`, `save_iter`, `validation_iter`, remaining iterations to the
      next checkpoint/validation, and estimated seconds to next checkpoint and
      completion. `py_compile` passed. Refreshed
      `experiments/world_model_task_rebinding/cosmos3/full301_wam_status_latest.md`
      reports latest iteration `366`, latest validation `300` / `7.984733`,
      `max/save/val=1500/300/300`, next checkpoint/validation at iter `600`
      in `234` iterations, estimated next-checkpoint ETA `4855.5s`, and
      completion ETA `23530.5s`. This is read-only scheduling/readiness
      evidence only; it did not touch the active H200 SFT, which remains
      running on allocation `118609` with no post-SFT artifacts yet.
      Slurm walltime margin update at `2026-06-09T01:47+08:00`: extended the
      same read-only status helper to query `squeue` for active job `118609`
      and report elapsed time, time limit, remaining walltime, and whether
      the current completion ETA fits inside the allocation. `py_compile`
      passed. Refreshed status reports latest iteration `374`, latest
      validation `300` / `7.984733`, next checkpoint/validation iter `600` in
      `226` iterations with ETA `4691.8s`, completion ETA `23375.8s`, Slurm
      elapsed `8:16:05`, time limit `1-00:00:00`, walltime remaining
      `56635.0s`, and `eta_completion_within_time_limit=true` with margin
      `33259.2s`. No `sft_completed`, `sft_failed`, or post-SFT artifacts
      exist yet. This confirms the held H200 allocation should be sufficient
      for the current full301 SFT to finish without another queue cycle if
      training speed remains similar.
      Watcher-liveness status update at `2026-06-09T01:50+08:00`: extended
      `scripts/world_model/report_cosmos3_full301_wam_status.py` to query
      expected tmux sessions for the clean SFT, strict action-eval watcher,
      exact-frame readout watcher, forward-dynamics watcher, live failure
      monitor, and allocation shell. `py_compile` passed. Refreshed status
      reports latest iteration `381`, latest validation `300` / `7.984733`,
      next checkpoint/validation iter `600` in `219` iterations, completion
      ETA `23230.4s`, Slurm remaining `56485.0s`,
      `eta_completion_within_time_limit=true`, and
      `all_expected_sessions_present=true` with missing sessions `[]`. There
      is still no `sft_completed`, `sft_failed`, `length_contract_precheck`,
      action metrics, readout, forward-dynamics result, or controller
      evidence.
      User correction/status poll at `2026-06-09T01:54+08:00`: re-confirmed
      the active boundary that the old ManiSkill Cosmos3 SFT/eval/controller
      outputs cannot be continued because any input/GT length mismatch,
      accidental truncation, or old 93-frame action-eval contract makes them
      untrustworthy. The current live training is already the required fresh
      full-length/equal-length path: clean-caption RGB-only
      `Cosmos3-Nano-Policy-DROID-DCP` joint-policy SFT with `301` video
      frames, `300 x 32` action targets, `301 x 56` state/readout labels, and
      strict post-SFT length precheck before metrics. Slurm job `118609`
      remains the only active Slurm job and is running on `server43`; expected
      tmux sessions are all present. Refreshed
      `experiments/world_model_task_rebinding/cosmos3/full301_wam_status_latest.json`
      and `.md` report latest iteration `393` with loss `7.3548`, latest
      validation iteration `300` with val loss `7.984733`, only checkpoint
      `iter_000000300`, next checkpoint/validation at iter `600` in `207`
      iterations with ETA `4293.2s`, completion ETA `22959.2s`, Slurm
      remaining `56241.0s`, and `eta_completion_within_time_limit=true`.
      There is still no `sft_completed`, `sft_failed`, generated 301-frame
      MP4, action metric, readout, forward-dynamics result, or controller
      evidence. Do not restart this clean full301 DROID run unless the strict
      checks fail or the step actually fails; keeping it running preserves the
      user's instruction to avoid queue churn while satisfying the fresh SFT
      requirement.
      Controller-pause guard at `2026-06-09T01:59+08:00`: found one stale
      Slurm entry point,
      `scripts/slurm/run_cosmos3_receding_teacher_forced_controller_retest.sbatch`,
      whose defaults still pointed at the invalid 6/6 action-condition SFT
      and readout roots with `NUM_FRAMES=93`. Patched it so controller retest
      is refused by default while fresh full301 DROID/Cosmos WAM
      action/readout evidence is missing; it now requires
      `ALLOW_RECEDING_CONTROLLER_DIAGNOSTIC=true`, refuses any root carrying
      `INVALID_DO_NOT_USE_20260609.md`, and treats non-`301` frames as an
      explicit short-horizon diagnostic requiring
      `ALLOW_SHORT_HORIZON_DIAGNOSTIC=true`. Verification:
      `bash -n` passed; default smoke
      `experiments/world_model_task_rebinding/cosmos3/receding_controller_guard_smoke_20260609_stderr.log`
      exits `30` with `controller_is_paused...`; explicit diagnostic smoke
      with the old defaults exits `24` on the invalid action-condition root.
      This does not run controller or GPU work; it only prevents future
      accidental reuse of the invalid 93-frame/controller chain while the
      current full301 SFT continues. Final poll after this guard reports SFT
      iteration `409`, loss `6.7965`, latest validation still iteration `300`
      / `7.984733`, Slurm `118609` still running on `server43`, all expected
      tmux sessions present, no `sft_completed`, no `sft_failed`, no generated
      full301 MP4, no action metrics, no readout, and no forward-dynamics
      result.
      Evidence-text/readiness update at `2026-06-09T02:04+08:00`: patched
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh` so the
      generated sample manifest boundary is mode-specific. The active
      `policy`/joint-WAM eval will now be recorded as full301 future RGB plus
      `300`-step action-chunk prediction from video prefix and causal
      robot/object task-state grounding, rather than mislabeled as
      forward-dynamics. Patched
      `scripts/world_model/inspect_cosmos3_action_eval_artifacts.py` with the
      same mode-neutral wording, and extended
      `scripts/world_model/report_cosmos3_full301_wam_status.py` to surface
      action-eval sample contract fields and external-target event readout
      summary when those artifacts exist. Verification:
      `bash -n scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh`
      passed; `py_compile` passed for both Python helpers; refreshed
      `experiments/world_model_task_rebinding/cosmos3/full301_wam_status_latest.json`
      and `.md`. Since the action-eval watcher was already running from the
      old script body, it was safely restarted in tmux at `02:04:02` after
      archiving the old log as
      `watch_pre_boundary_text_refresh_20260609_0202.log`. It is again only
      waiting for the clean SFT `sft_completed` marker and has not launched
      inference. The SFT continued during the reload and reached iteration
      `422`, with no completion/failure/post-SFT artifact yet.
      Step-liveness status update at `2026-06-09T02:06+08:00`: added a
      read-only Slurm step check to
      `scripts/world_model/report_cosmos3_full301_wam_status.py`, defaulting
      to active training step `118609.41`. It calls `scontrol show step` and
      records `State`, `NodeList`, `StartTime`, and TRES so a held allocation
      is not confused with a live training step. Verification:
      `.venv/bin/python -m py_compile scripts/world_model/report_cosmos3_full301_wam_status.py`
      passed and refreshed `full301_wam_status_latest.json/.md`. Current
      status reports allocation `118609` running on `server43`, training step
      `118609.41` `RUNNING` on `server43`, latest SFT iteration `428` with
      loss `7.0426`, latest validation still iteration `300` / `7.984733`,
      next checkpoint/validation at iter `600` in `172` iterations, no
      `sft_completed`, no `sft_failed`, and no post-SFT action/readout/
      forward-dynamics artifact.
      Runtime poll at `2026-06-09T02:07+08:00`: refreshed status reports
      latest SFT iteration `432` with loss `6.2471`, latest validation still
      iteration `300` / `7.984733`, checkpoint `iter_000000300` remains the
      only checkpoint with metadata, next checkpoint/validation is iter `600`
      in `168` iterations, allocation `118609` and training step `118609.41`
      remain `RUNNING` on `server43`, and all expected tmux watcher sessions
      are present. There is still no `sft_completed`, no `sft_failed`, no
      generated full301 MP4, no action metric, no readout, and no
      forward-dynamics result.
      Runtime poll at `2026-06-09T02:10+08:00`: allocation `118609` remains
      `RUNNING` on `server43`, and training step `118609.41` remains
      `RUNNING` with TRES `cpu=12,gres/gpu=1,mem=96G,node=1`. Refreshed
      `experiments/world_model_task_rebinding/cosmos3/full301_wam_status_latest.json`
      and `.md`: latest SFT iteration is `439` with loss `7.7619`; latest
      validation is still iteration `300` with val loss `7.984733`;
      checkpoint `iter_000000300` is still the only checkpoint with metadata;
      next checkpoint/validation is iter `600`, `161` iterations away with
      ETA about `3344s`; completion ETA is about `22037s`, within the
      remaining Slurm limit. The live failure monitor still reports
      `step_live step=118609.41`, all expected tmux sessions are present, and
      no failure/completion/action-eval/readout artifacts were found. This is
      still active SFT training only, not method evidence or a controller
      result.
      External-target matrix watcher at `2026-06-09T02:16+08:00`: started
      tmux `droid_policy_full301_external_target_matrix_after_core_20260609`
      to cover the user's external-world requirement after the core WAM chain,
      without competing with the current SFT or the queued forward-dynamics
      SFT. It waits on
      `experiments/world_model_task_rebinding/cosmos3/droid_policy_forward_dynamics_after_clean_policy_readout_20260608/forward_dynamics_chain_completed`
      before launching any action-eval/readout work. Matrix root:
      `experiments/world_model_task_rebinding/cosmos3/action_readout_matrix_droid_policy_full301_clean_caption_after_forward_dynamics_20260609`;
      samples `0 2 15 3`; all use `NUM_FRAMES=301`,
      `ACTION_CHUNK_SIZE=300`, and
      `ALLOW_SHORT_HORIZON_DIAGNOSTIC=false`. Selected-sample audit
      `.../selected_sample_motion_audit.json` reports
      `strict_selection_ok=true`: index `0` `hole_constant` has target-hole
      displacement `0.132002m`, index `2` `hole_reverse` has `0.087600m`,
      index `15` `hole_move_stop` has `0.140000m`, and index `3` `none` has
      `0.000000m` as the static negative control. Patched the read-only
      status helper to include this watcher in expected tmux sessions;
      `py_compile` passed. Refreshed status at `02:16` reports allocation
      `118609` and training step `118609.41` still running on `server43`,
      latest SFT iteration `456`, loss `7.9753`, latest validation still
      iteration `300` / `7.984733`, next checkpoint/validation iter `600` in
      `144` iterations, and all seven expected tmux sessions present. Still
      no `sft_completed`, `sft_failed`, generated full301 MP4, action metric,
      readout, forward-dynamics result, matrix result, or controller evidence.
      Evidence audit and resource-handoff update at `2026-06-09T02:30+08:00`:
      added `scripts/world_model/audit_cosmos3_full301_method_evidence.py`, a
      read-only full-chain auditor that requires invalid old roots, clean
      full301 DROID condition contract, completed clean SFT, strict full301
      action/video eval, external-target readout, forward-dynamics SFT/eval/
      readout, moving-target matrix, and controller pause while evidence is
      incomplete. Current output
      `experiments/world_model_task_rebinding/cosmos3/full301_method_evidence_audit_latest.json`
      has `world_model_evidence_complete=false`: invalid sentinels and clean
      input contract are satisfied, but all generated post-SFT evidence items
      remain incomplete. Added a walltime forecast to
      `scripts/world_model/report_cosmos3_full301_wam_status.py`; refreshed
      status at `02:30` reports latest SFT iteration `496`, latest validation
      `300` / `7.984733`, completion still fits allocation `118609`, but the
      estimated remaining walltime after primary SFT is only about `31130.7s`
      while a full forward-dynamics SFT is estimated at about `34230.0s`
      before eval margin. Patched
      `scripts/slurm/watch_cosmos3_droid_forward_dynamics_after_policy_readout.sh`
      to refuse starting that long forward-dynamics SFT when less than
      `43200s` remains, writing
      `.../droid_policy_forward_dynamics_after_clean_policy_readout_20260608/forward_dynamics_needs_new_allocation`
      instead of wasting the tail of the current allocation.
      Submitted exactly one delayed tmux/salloc follow-up allocation, not a
      fanout: session
      `droid_policy_clean_caption_forward_dynamics_fresh_alloc_20260609`,
      job `120966`, partition `cpu`, one `gpu:NVIDIAH200:1`, `12` CPUs,
      `128G`, `1-00:00:00`, `BeginTime` eligible/start
      `2026-06-09T08:29:42`. Non-executing probes showed `gpu` partition
      one-H200 requests forecast around `2026-06-15T10:35`, while `cpu`
      partition with the same H200 gres and `--begin=now+6hours` forecasts
      `2026-06-09T08:27-08:29`. Added
      `scripts/slurm/launch_cosmos3_forward_dynamics_allocation_tmux.sh` so
      the follow-up allocation waits for the walltime-handoff marker or an
      already-completed forward-dynamics chain before doing anything, then
      resumes the same clean full301 forward-dynamics output roots and shared
      completion marker that the matrix watcher is waiting on. `bash -n` for
      the new launcher and `py_compile` for the status helper passed. This is
      scheduling/evidence-boundary work only; no controller is running, and no
      old 93-frame or contaminated output has been continued.
      Objective-evidence hardening at `2026-06-09T02:35+08:00`: patched
      `scripts/world_model/inspect_cosmos3_task_state_prediction.py` so strict
      external-target readout fails on a moving-target sample if the reference
      has `target_motion_onset_frame` but the prediction has no
      `predicted_motion_onset_frame`. Patched
      `scripts/world_model/inspect_cosmos3_action_eval_artifacts.py` so
      `policy`/`inverse_dynamics` samples require
      `action_is_prediction_target=true`, `action_prediction_required=true`,
      predicted action length `300`, target action length `300`, and predicted
      action dimension coverage. Patched
      `scripts/world_model/audit_cosmos3_full301_method_evidence.py` so the
      primary WAM evidence must be `model_mode=policy`, the
      forward-dynamics evidence must be `model_mode=forward_dynamics`, and
      both primary/forward readout evidence must include non-null predicted
      and target motion-onset frames plus finite final target-hole/insertion
      geometry errors. `py_compile` passed for all three scripts, and the
      refreshed audit still correctly reports
      `world_model_evidence_complete=false`. Latest status during this check:
      allocation `118609` and step `118609.41` remain running on `server43`,
      clean SFT reached iteration `507`, latest validation is still
      `300` / `7.984733`, follow-up allocation `120966` remains pending with
      reason `BeginTime`, and there are still no generated full301 MP4/action
      metrics/readout artifacts or controller evidence.
      Post-motion action metric hardening at `2026-06-09T02:39+08:00`:
      inspected the action target layout and confirmed the final action-target
      column is `prefix_perturb_triggered`, not a future trigger label; future
      target motion/trigger must therefore be evaluated through the
      external-target readout, not misread from this action column. Patched the
      post-SFT action-metric block inside
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh` so the
      generated `action_prediction_metrics.json` also records
      `future_after_prefix` and `post_target_motion` action-error windows,
      with the post-motion window sliced from `state_target_path` hole-motion
      onset using a `0.002m` threshold. Patched
      `scripts/world_model/inspect_cosmos3_action_eval_artifacts.py` and
      `scripts/world_model/audit_cosmos3_full301_method_evidence.py` so moving
      `policy` samples require a non-empty post-target-motion action window
      when action prediction evidence exists. Syntax/compile checks passed.
      Restarted tmux `droid_policy_clean_caption_action_eval_20260608` at
      `02:38:55` to load the updated watcher; the previous wait log is
      archived as
      `.../action_eval_after_sft_full1000_maniskill_rgb_full301_joint_policy_droid_policy_clean_caption_20260608/watch_pre_post_motion_action_metrics_20260609_023854.log`.
      The restarted watcher is only waiting for `sft_completed`; it did not
      launch inference. Runtime status after restart: allocation `118609` and
      step `118609.41` remain running on `server43`, clean SFT reached
      iteration `518`, latest validation `300` / `7.984733`, follow-up
      allocation `120966` remains `PENDING (BeginTime)`, no generated full301
      MP4/action metrics/readout artifacts exist, and controller remains
      paused.

- [ ] 2026-06-08 DROID Policy full301 SFT active handoff:
      Superseded boundary from `2026-06-09T01:40+08:00`: the older
      immediate-state-target DROID handoff records in this item are historical
      and must not be treated as the active run. The current active run is the
      clean-caption full301 RGB `Cosmos3-Nano-Policy-DROID` chain recorded in
      the 2026-06-09 boundary item above, with active step `118609.41` and
      output root
      `experiments/world_model_task_rebinding/cosmos3/sft_full1000_maniskill_rgb_full301_joint_policy_droid_policy_clean_caption_20260608`.
      Any reference below to step `118609.35`, immediate-state-target roots, or
      their action-eval/readout watchers is retained only as history/negative
      provenance and must not be resumed or used as method evidence.
      previous 6/6 ManiSkill Cosmos3 SFT/action-eval/controller outputs remain
      invalid as method evidence because they used the 93-frame short-horizon
      chain against 301-frame references. Do not resume those checkpoints,
      videos, PSNR numbers, readouts, or controller inputs. Immediate handoff
      to the published `Cosmos3-Nano-Policy-DROID` backbone has completed the
      checkpoint conversion to DCP at
      `checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID-DCP` after forcing the VLM
      processor/tokenizer to load from the local DROID checkpoint and clearing
      proxy metadata inside the compute step. The held H200 allocation
      `118609` on `server43` was kept alive; the constrained Nano baseline
      step was cancelled only to free the same card for DROID. Active training
      step is `118609.35`, output root
      `experiments/world_model_task_rebinding/cosmos3/sft_full1000_maniskill_rgb_full301_joint_policy_droid_policy_restart_20260608_immediate_state_targets`.
      This run uses RGB-only full301 `joint_policy` data from
      `action_state_conditions_full1000_maniskill_default_regen_full301_joint_policy_state_targets_20260608_after_nano_state_targets`,
      with strict full preflight passed on all `912` train and `88` val rows:
      video frames `[301]`, action lengths `[300]`, action dims `[32]`, state
      target lengths `[301]`, state dims `[56]`. Initial validation loss at
      iteration `0` is `16.758633`; by `2026-06-08T22:43+08:00` it reached
      iteration `14`, still training and not final evidence. Post-SFT strict
      full301 action/video eval watcher is active in tmux session
      `droid_policy_full301_action_eval_20260608`, output root
      `experiments/world_model_task_rebinding/cosmos3/action_eval_after_sft_full1000_maniskill_rgb_full301_joint_policy_droid_policy_20260608_immediate_state_targets`;
      it waits for `sft_completed`, selects best validation checkpoint, and
      hard-fails if predicted/reference video length is not `301` or predicted
      action length is not `300`. Controller work remains paused until this
      full-length RGB WAM contract produces inspected eval artifacts.
      External-target/task-state follow-up is also queued without running the
      controller: tmux session
      `droid_policy_full301_task_state_readout_20260608` waits for the DROID
      full301 action eval completion marker, then trains a strict exact-frame
      task-state readout on the approved `301`-frame regenerated videos and
      decodes the Cosmos-generated video into hole, peg, TCP,
      peg-head-at-hole, grasp, and insertion trajectories. Output root:
      `experiments/world_model_task_rebinding/cosmos3/task_state_readout_droid_policy_full301_joint_policy_20260608_immediate_state_targets`.
      It sets `ALLOW_ONE_SHOT_COSMOS_CONTROLLER_DIAGNOSTIC=false`; controller
      remains skipped until full-length WAM RGB/action and external
      target-state diagnostics exist and are inspected. Readout watcher
      hardening at `2026-06-08T22:55+08:00` added explicit
      `READOUT_PYTHON` recording and dependency import preflight; the active
      restarted watcher passed `readout_python_preflight_ok`. The task-state
      inspection now reports advisory external-target event metrics:
      target-hole motion onset frame, onset error, final hole-position error,
      final insertion-geometry error, future hole-path RMSE, and future
      insertion-geometry RMSE. Smoke artifact
      `experiments/world_model_task_rebinding/cosmos3/task_state_readout_event_metric_smoke_20260608`
      verified those fields on a perfect GT prediction with zero final errors
      and motion onset frame `84` at a `0.002m` diagnostic threshold.
      The earlier optional multi-scenario follow-up watcher
      `droid_policy_full301_matrix_after_main_20260608` was stopped before it
      started compute after the method-interface audit below, because the core
      action-conditioned `forward_dynamics` WAM side should run before optional
      multi-scenario diagnostics on the single held H200. That matrix can be
      relaunched later after the two primary WAM sides have strict full301
      evidence; it is not controller evidence.
      Latest poll at `2026-06-08T23:05+08:00`: allocation `118609` and step
      `118609.35` remain live on `server43`; the DROID SFT log reached
      iteration `77` with loss `14.6218`. No `sft_completed`,
      `val_loss_summary.json`, `latest_checkpoint.txt`, or checkpoint metadata
      exists yet under the active DROID output root. Action eval, readout, and
      matrix watchers remain in their intended wait states, and no controller
      has been run.
      Interface audit at `2026-06-08T23:10+08:00`: the active row is
      `model_mode=policy` with vision prefix `[0..7]`, empty action-condition
      frames, `action_chunk_size=300`, and `state_target_frame_count=301`, so
      it covers the action-generating WAM side but not the action-conditioned
      forward-dynamics/scoring side. Added and launched tmux watcher
      `droid_policy_forward_dynamics_after_policy_readout_20260608`, script
      `scripts/slurm/watch_cosmos3_droid_forward_dynamics_after_policy_readout.sh`.
      It waits for the current joint-policy full301 action eval and
      exact-frame external-target readout to finish, then runs DROID
      `forward_dynamics` full301 SFT/eval/readout on the same approved RGB
      dataset with `301` frames, `300` action rows, and `301` state targets.
      The optional multi-scenario matrix watcher was stopped before compute to
      prevent it from racing this core forward-dynamics chain for the single
      H200. Latest SFT poll at `2026-06-08T23:12+08:00`: step `118609.35`
      reached iteration `98`, no checkpoint or completion marker yet. The
      action-eval watcher is still waiting for `sft_completed`, the readout
      watcher is still waiting for `post_sft_action_eval_completed`, and the
      forward-dynamics watcher is still waiting for the primary
      `readout_prediction_completed_controller_skipped` marker.
      Follow-up data-leak audit at `2026-06-08T23:18+08:00`: the active
      DROID `joint_policy` JSONL carried correct structured `300 x 32`
      action/state targets, but its `t2w_windows.caption` still contained
      future ground-truth end poses such as hole/peg/TCP `moves from ... to
      ...`. Cosmos SFTDataset does tokenize `t2w_windows[*].caption`, so step
      `118609.35` was contaminated by privileged future text and cannot be
      method evidence. It was cancelled at iteration about `113` without
      releasing allocation `118609`. Updated
      `scripts/world_model/export_cosmos3_maniskill_action_conditions.py` so
      future exports default to sanitized prefix-only captions and metadata:
      prefix TCP/peg/hole/relative geometry, observed prefix velocity,
      prefix perturbation, grasp/insert predicates, and scenario only; future
      state remains only in `state_target_path` labels. Smoke export
      `action_state_conditions_clean_caption_smoke_20260608` verified
      `sanitize_future_caption=true` and no future-caption patterns. New clean
      DROID full301 `joint_policy` SFT is running in tmux
      `droid_policy_clean_caption_joint_policy_20260608`, step `118609.36`,
      output
      `experiments/world_model_task_rebinding/cosmos3/sft_full1000_maniskill_rgb_full301_joint_policy_droid_policy_clean_caption_20260608`,
      condition root
      `experiments/world_model_task_rebinding/cosmos3/action_state_conditions_full1000_maniskill_default_regen_full301_joint_policy_clean_caption_20260608`.
      Clean action-eval/readout/forward-dynamics watchers were restarted under
      `droid_policy_clean_caption_action_eval_20260608`,
      `droid_policy_clean_caption_task_state_readout_20260608`, and
      `droid_policy_clean_caption_forward_dynamics_after_policy_readout_20260608`.
      Lightweight `srun --overlap` check confirmed the clean exporter is
      running on `server43`; allocation was not released and no controller was
      run.
      Current clean-caption poll at `2026-06-08T23:37+08:00`: allocation
      `118609` remains held on `server43`; active Slurm step is `118609.41`
      in tmux session `droid_policy_clean_caption_joint_policy_20260608`.
      Strict full preflight has passed on the clean data root with train/val
      `912/88`, `301` RGB frames, `300 x 32` action targets, and `301 x 56`
      state targets. The clean SFT loaded
      `checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID-DCP`, uses local DROID
      tokenizer override, and reported iteration-0 validation loss
      `16.758268`. It reached iteration `9` with step time about `20.6s`;
      GPU utilization on the held H200 was `100%`, so the run is live rather
      than queued or released. No `sft_completed`, checkpoint metadata,
      `latest_checkpoint.txt`, or `val_loss_summary.json` exists yet. The
      strict clean action-eval watcher remains waiting for `sft_completed`;
      the readout watcher remains waiting for exact full301 action-eval
      completion; the forward-dynamics watcher remains waiting for the clean
      primary readout. No controller has been run.
      Clean-condition full audit at `2026-06-08T23:40+08:00` parsed every
      train/val JSONL row and sidecar, not only the manifest: train/val
      counts are `912/88`, every row is `model_mode=policy`, action sidecars
      are exactly `300 x 32`, state sidecars are exactly `301 x 56`, vision
      prefix is `[0..7]`, action-condition frames are empty, caption future
      endpoint patterns (`moves from`, `] to [`, `xyz_end`,
      `source_frame_end`, `grasped changes`, `inserted changes`) occur `0`
      times, and metadata contains no future `task_state_condition`. This is
      the current valid full301 RGB DROID WAM SFT input contract; the
      contaminated `118609.35` run and old 93-frame chain remain invalid.
      Resolution audit at `2026-06-08T23:44+08:00`: the approved source MP4s
      are still `1024x1024/301f/30fps`, but
      `scripts/slurm/run_cosmos3_full1000_sft_in_allocation.sh` currently
      hard-codes `model.config.resolution='256'` for this DROID/Nano SFT.
      The DROID checkpoint config has native `resolution="720"` and shift /
      encode settings for `256`, `480`, and `720`. Therefore the current run
      is a valid equal-length/no-leak full301 WAM-contract repair attempt, but
      its generated video clarity must not be presented as the final visual
      quality ceiling. If the full301 contract and action/readout diagnostics
      pass, the next controlled SFT/eval variable should be a higher
      resolution setting such as `480`/`720` if memory permits, or at minimum
      upsampled/padded demos for human review. Do not interrupt the current
      live clean SFT solely for this audit finding before the first full301
      checkpoint/eval evidence exists.
      Runtime poll at `2026-06-08T23:52+08:00`: allocation `118609` is still
      held on `server43`; active training step `118609.41` reached iteration
      `50` with loss `15.3362` and continues at roughly `20.7s/step`. The
      output root still has no checkpoint metadata, `latest_checkpoint.txt`,
      `sft_completed`, `val_loss_summary.json`, or `sft_failed`; this is
      expected before the first `SAVE_ITER=300` / `VALIDATION_ITER=300`
      checkpoint. Active tmux sessions are only the held allocation, clean
      DROID SFT, clean full301 action eval watcher, clean exact-frame readout
      watcher, and clean forward-dynamics watcher. No old 93-frame path or
      controller path is running.
      Action-eval hardening at `2026-06-08T23:56+08:00`: patched
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh` so
      policy/inverse-dynamics action metrics no longer compare
      `min(pred_dim, target_dim)` silently. Predicted action rows must now
      have at least the target `32` dimensions and exactly `300` steps, while
      reconstruction still requires `301` predicted/reference frames with no
      truncation. The sample manifest now records
      `causal_task_state_condition` and explicitly records whether a future
      `task_state_condition` appeared. The clean action-eval tmux watcher was
      restarted to load this script change, with prior log archived as
      `watch_pre_action_dim_hardening_20260608_235535.log`; the live
      `watch.log` is NUL-free and waits for the same clean
      `sft_completed` marker. The training step `118609.41` was not touched.
      Liveness check at `2026-06-08T23:57+08:00`: in-allocation
      `nvidia-smi` on `server43` showed the H200 at `100%` utilization with
      about `105025 MiB / 143771 MiB` used. The main
      `cosmos_framework.scripts.train` process is still running, and both
      `run.log` and `sft_train.log` had fresh modification times after
      `23:56`. The absence of post-iteration-50 `Iteration` lines is not
      treated as a stall while GPU/process/log liveness remain positive.
      Artifact-inspection hardening at `2026-06-08T23:59+08:00`: patched
      `scripts/world_model/inspect_cosmos3_action_eval_artifacts.py` so the
      final action-eval inspection is strict by default. It now fails if the
      prediction video is not exactly the sample manifest's reference frame
      count, if all/future reconstruction metrics are missing or not
      equal-length/no-truncation, if compared all frames are not `301`, if the
      future segment length does not match `301 - reference_start`, or if a
      policy/inverse-dynamics action prediction is missing, not `300` steps,
      or does not cover the target action/state-row dimension. `py_compile`
      and `bash -n` passed. This check runs before
      `post_sft_action_eval_completed` is written, so a short or
      under-dimensioned prediction will block downstream readout rather than
      becoming method evidence.
      Poll at `2026-06-09T00:01+08:00`: only Slurm job `118609` is active
      for this user, still running on `server43`; step `118609.41` is still
      active. No `sft_completed`, `sft_failed`, `val_loss_summary.json`,
      `latest_checkpoint.txt`, or checkpoint `.metadata` exists yet under the
      clean DROID SFT root. `run.log` and `sft_train.log` were both modified
      at `2026-06-09T00:01+08:00`, so the run remains live. The clean
      action-eval watcher is waiting for `sft_completed`; the exact-frame
      readout watcher is waiting for `post_sft_action_eval_completed`; the
      forward-dynamics watcher is waiting for the primary joint-policy readout
      completion marker. Controller remains paused.

- [ ] 2026-06-08 full-length RGB Cosmos/DROID WAM restart:
      the current Cosmos3 SFT/eval chain has a concrete length-contract
      failure. The approved regenerated full1000 ManiSkill Cosmos videos are
      `301` frames at `1024x1024`, `30 fps`, but the current action/state
      condition export and post-SFT action-eval path are configured for
      `total_video_frames=93`, `NUM_FRAMES=93`, and `ACTION_CHUNK_SIZE=92`.
      The audited prediction
      `experiments/world_model_task_rebinding/cosmos3/action_eval_after_sft_full1000_maniskill_default_regen_20260606_chain/inference/hole_constant_seed702000_n167_traj_0_traj_0_sft_full1000_forward_action_prefix8/vision.mp4`
      is `93` frames, `256x256`, while its reference
      `experiments/world_model_task_rebinding/cosmos3/sft_dataset_full1000_maniskill_default_regen_20260606_0055/val/videos/0000_hole_constant_seed702000_n167_traj_0_traj_0.mp4`
      is `301` frames, `1024x1024`. Therefore existing all/future PSNR values
      and the previous 6/6 ManiSkill Cosmos3 SFT/action-eval/controller chain
      are cropped short-horizon diagnostics only, not trustworthy world-model
      evidence and not reusable active checkpoints/controller inputs. Next
      active WM work must restart SFT and guarantee pred/ref equal intended
      length after any prefix offset, support arbitrary rollout length via
      native full-length or chained/receding predictions, and train/evaluate
      on RGB video inputs only; depth is not an active input. Controller work
      is paused until this full-length RGB WAM contract is repaired. Before
      further active SFT, audit current Cosmos3 variants and local
      checkpoints; do not default to `Cosmos3-Nano` unless the stronger
      applicable variant is infeasible and that constraint is recorded.
      Immediate variant audit: `nvidia/Cosmos3-Nano-Policy-DROID` is not only
      an action-prior candidate. It should be tried as a joint WAM backbone:
      video prefix plus causal proprio/action/task metadata in, future RGB,
      task-state readout, and action/chunk prediction out. Local
      `external/cosmos-framework` already supports action modes:
      `forward_dynamics` (`RGB/video + action -> future RGB`), `policy`
      (`RGB/video -> action + future RGB`), and `inverse_dynamics`
      (`RGB/video -> action`). Because the released checkpoint is
      DROID/RoboLab-oriented (`droid_lerobot`, 480-resolution, 15 fps
      conditioning, 32-step chunks, 8D served `joint_pos` actions; local
      ManiSkill domain is 32D), it must be downloaded/preflighted and then
      SFT/adapted on approved full301 ManiSkill RGB videos/action records.
      `Cosmos3-Nano-DCP` is allowed only as the fresh constrained baseline if
      DROID conversion/download is not immediately ready. Do not use oracle
      simulator state as visual input; simulator state may appear only as
      causal proprio/task metadata, labels, readout supervision, or diagnostics.
      Current execution status: old 93-frame ManiSkill Cosmos3 SFT/action-eval
      artifacts are invalidated and must not be reused. Fresh full301 RGB
      `joint_policy` Nano baseline is running in held allocation `118609` on
      `server43` under
      `experiments/world_model_task_rebinding/cosmos3/sft_full1000_maniskill_rgb_full301_joint_policy_nano_restart_20260608_210653`;
      strict preflight passed with `301` video frames and RGB-only input.
      DROID Policy download is being retried in filtered tmux session
      `cosmos3_droid_policy_download_filtered_20260608`; DROID SFT is pending
      successful checkpoint download and DCP conversion. Length-contract guard
      follow-up made
      `scripts/world_model/evaluate_cosmos3_rollout_reconstruction.py` strict
      by default: pred/ref length mismatch and `--max-frames` truncation now
      fail unless explicitly overridden for a non-method diagnostic. Smoke
      against the old invalid 93-vs-301 artifact failed as intended with
      `reference_available=301, prediction_available=93`; no metrics were
      written. Active action-eval wrapper now defaults to full301 paths and
      uses each JSONL row's `model_mode`; current `joint_policy` rows evaluate
      `model_mode=policy`, so the action chunk is a prediction target rather
      than a clean future condition. Allocation watcher
      `droid_policy_full301_after_nano_20260608` is running in tmux; it waits
      for Nano full301 SFT completion, then runs strict full301 Nano eval,
      converts DROID Policy to DCP when weights are complete, and launches the
      DROID full301 `joint_policy` SFT in allocation `118609` without sbatch or
      releasing the card.
      Runtime check at `2026-06-08T21:50+08:00`: `squeue -u yanhongru`
      shows only allocation `118609` active, so the old 6/5-6/6
      Cosmos/controller lines are not continuing as Slurm jobs. Stale old
      tmux sessions for preview, regeneration, readout, controller, and policy
      pilots were killed. After the state-target export completed, the only
      remaining tmux sessions are the held allocation, DROID checkpoint
      download, and DROID-after-Nano watcher. Follow-up poll at
      `2026-06-08T22:15+08:00`: the fresh Nano full301 SFT is alive in step
      `118609.28` and reached iteration `182` with loss `11.3883`; initial
      validation was `16.721653`. The filtered DROID Policy download directory
      is about `30G`; all seven transformer shards and
      `vision_encoder/model.safetensors` are present. A repository VAE cache
      download is still `.incomplete`, but the active DCP/SFT path uses the
      separate local `Wan2.2_VAE.pth` and the converter loads only
      transformer/vision weights from the diffusers checkpoint, so this cache
      artifact must not block DROID handoff. Do not interpret these as final
      evidence; they only prove the restart is active and the old 93-frame
      chain is not being advanced.
      Follow-up schema repair at `2026-06-08T21:38+08:00`: audited the fresh
      `joint_policy` JSONL and found that it trains future RGB plus action
      chunk, but had no explicit full-length future task-state target sidecar.
      Updated `scripts/world_model/export_cosmos3_maniskill_action_conditions.py`
      to write `state_target_path`/`task_state_target_path` files containing
      `301 x 56` raw target sequences for hole pose, peg pose, TCP pose,
      qpos/qvel, peg-head-at-hole, grasp/insert predicates, and perturbation
      summaries. These future simulator states are labels/evaluation targets
      only, not clean conditions. Smoke export
      `experiments/world_model_task_rebinding/cosmos3/action_state_conditions_full301_state_targets_smoke_20260608`
      verified `model_mode=policy`, vision condition `[0..7]`, empty action
      condition, `action_chunk_size=300`, and state target length `301`.
      `scripts/slurm/run_cosmos3_full1000_sft_in_allocation.sh` now supports
      `REQUIRE_STATE_TARGETS=true` and validates `state_target_path` length and
      required task fields before active method SFT. Restarted watcher
      `droid_policy_full301_after_nano_20260608` with stamp
      `20260608_after_nano_state_targets`; it will use
      `action_state_conditions_full1000_maniskill_default_regen_full301_joint_policy_state_targets_20260608_after_nano_state_targets`
      for DROID SFT. CPU/IO pre-export for that full1000 state-target root
      completed inside allocation `118609`; Nano SFT remains active in step
      `118609.28`.
      Follow-up eval repair: audited Cosmos inference docs/output and confirmed
      `model_mode=policy` writes predicted actions to `sample_outputs.json` in
      addition to `vision.mp4`. Updated
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh` to hard-fail
      policy/inverse-dynamics eval if predicted actions are missing or not
      exactly `300` steps, and to write normalized action-target MAE/MSE/RMSE
      in `action_prediction_metrics.json`. Future full301 method eval must
      inspect both equal-length video reconstruction and equal-length predicted
      action metrics.
      Full1000 state-target export completed at `2026-06-08T21:55+08:00`.
      Contract check
      `experiments/world_model_task_rebinding/cosmos3/action_state_conditions_full1000_maniskill_default_regen_full301_joint_policy_state_targets_20260608_after_nano_state_targets_export/contract_check.json`
      passed with `1000` rows, train/val `912/88`, action lengths `[300]`,
      action dims `[32]`, state lengths `[301]`, state dims `[56]`, required
      task-state fields present, and `0` errors. The current Nano baseline
      remains a fresh full301 constrained baseline because it was launched
      before the state-target sidecar repair; the DROID Policy SFT must use
      the validated state-target root with `REQUIRE_STATE_TARGETS=true`.
      Runtime handoff repair at `2026-06-08T22:00+08:00`: updated
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh` to support
      strict checkpoint eval without waiting for `sft_completed`, and updated
      `scripts/slurm/watch_droid_policy_full301_after_nano_in_allocation.sh`
      so the held allocation can switch from the constrained Nano baseline to
      DROID once both conditions hold: Nano has at least
      `iter_000000300/model/.metadata` and the DROID diffusers checkpoint is
      complete. The watcher was restarted with
      `ALLOW_NANO_EARLY_HANDOFF=true`; latest watcher state reports
      `checkpoint_ready=false` and `droid_checkpoint_ready=false`, so no
      training step has been interrupted yet. If both become true, it will
      cancel only the Nano Slurm step, keep allocation `118609`, run strict
      full301 Nano checkpoint eval, convert DROID to DCP, and launch DROID
      full301 RGB `joint_policy` SFT from the validated state-target root.
      Readout hardening at `2026-06-08T22:03+08:00`: task-state readout
      training/prediction now defaults to strict exact video-frame count via
      `require_exact_video_frames=true`, and readout watchers explicitly pass
      `--require-exact-video-frames`. This prevents a short Cosmos prediction
      from being silently padded into a fake 301-frame task-state trajectory.
      Smoke check accepted a regenerated `301`-frame, `30 fps`, `1024x1024`
      validation video and rejected the old invalid 93-frame prediction with
      `video frame count mismatch ... decoded=93, expected=301`. This extends
      the no-truncation/no-padding contract from video PSNR/action eval to the
      external target-hole/insert-point readout path.
      SFT preflight hardening at `2026-06-08T22:08+08:00`: updated
      `scripts/slurm/run_cosmos3_full1000_sft_in_allocation.sh` so
      `REQUIRE_STATE_TARGETS=true` defaults to `STRICT_FULL_PREFLIGHT=true`
      and checks every train/val row rather than only a small sample. It now
      records checked row counts, video frame counts, action lengths/dims, and
      state target lengths/dims in `strict_length_preflight_report`. Local
      preflight-equivalent check on the DROID state-target root passed for
      train `912` and val `88`: video frames `[301]`, action lengths `[300]`,
      action dims `[32]`, state lengths `[301]`, state dims `[56]`.
- [ ] 2026-06-08 DP target-visibility/WAM-retention follow-up:
      the current answer to the user question is: the frozen state-DP is not
      literally missing the moved hole in this controller stack. In the
      latest audited hard-case rollout, `policy_obs[35:42]` contains the
      metric `box_hole_pose` with post-trigger position RMSE `0.000346m` and
      final error `0.000000m`. The failure is that the static DP prior and
      current bridge/scorer path do not convert the moved target frame into a
      contact-feasible short insertion trajectory. New run
      `experiments/world_model_task_rebinding/rebinding_controller/wam_retention_penalty_small_rgbd_holeconstant_seed702000_no_video_20260608_1616`
      used `slot_source=rgbd`, `scenario=hole_constant`, final-success bonus,
      WAM scorer retention penalty, and the non-method WAM smoke checkpoint.
      It failed (`success_once=false`, `success_at_end=false`,
      `final_grasped=true`, final metric peg-head-at-hole
      `[-0.126763, 0.020247, -0.048345]`). Visual replay
      `.../visual_review_envstate_default_view_1024_full301_30fps_20260608/state_replay.mp4`
      and keyframe sheet
      `.../keyframe_sheet_000_054_090_130_170_210_250_280_300.png`
      show pickup/hold and approach to the box, but no stable insertion. Event
      audit: selected chunks were all `dp_bridge_blend`; WM-policy action
      generator candidates were never selected, and many selected chunks had
      no available WAM future. Strict selected-WAM variants
      `wam_retention_selected_rgbd_holeconstant_seed702000_no_video_20260608_1605`
      and
      `wam_retention_selected_tailok_rgbd_holeconstant_seed702000_no_video_20260608_1606`
      failed with no surviving candidate because the receding Cosmos future
      sequence did not cover the required selected chunks. Boundary: negative
      RGB-D/Cosmos/WAM diagnostic, not method success. Next aligned work is to
      make Cosmos3 provide a causal target hole/peg/TCP future trajectory with
      coverage for every selected short chunk, train/score action chunks on
      that WAM interface, execute only a short prefix, and re-observe; do not
      reduce this to more scalar gate/bridge tuning.
      Follow-up implementation added explicit
      `cosmos_task_state_allow_terminal_hold` and
      `cosmos_task_state_terminal_hold_max_frames` support to
      `scripts/world_model/evaluate_rebinding_controller.py` and
      `scripts/slurm/run_cosmos_task_state_controller_in_allocation.sh`.
      Default is off. When explicitly enabled, the controller can use the last
      causal receding Cosmos task-state row as a diagnostic held target for
      WAM future scoring after the final segment, and every future report
      records terminal-hold provenance. Functional smoke at frame `200`
      changed WAM future availability from missing `15/16` frames to
      available with `15` terminal-held frames. RGB-D hard-case diagnostic
      `experiments/world_model_task_rebinding/rebinding_controller/wam_terminal_hold_selected_rgbd_holeconstant_seed702000_no_video_20260608_1645`
      ran inside held allocation `118088` on `server44` with strict
      selected-WAM required. It failed task completion
      (`success_once=false`, `success_at_end=false`, final grasp true, final
      metric peg-head-at-hole
      `[-0.142047, -0.062097, -0.027473]`) but no longer had missing WAM
      futures: candidate WAM reports were all available and
      `future_state_sequence_unavailable=0`; terminal hold was used for `300`
      candidate reports and `15` selected reports. The remaining failure is
      action selection: selected chunks were `dp_bridge_blend=192`; WM-policy
      action-generator candidates were present (`75`) but selected `0` times.
      Visual review of
      `.../visual_review_envstate_default_view_1024_stride8_30fps_20260608/state_replay.mp4`
      shows pickup/hold and approach, but no insertion. Boundary: negative
      RGB-D/Cosmos/WAM diagnostic only; next aligned step is to train/score a
      stronger DP-preserving WM-action generator or scorer so WM-policy chunks
      can beat DP bridge under the same causal target trajectory.
      Follow-up H5 audit for the current user question checked three
      hard-case rollouts directly. In all three,
      `policy_obs[35:38]` tracks the metric moved-hole position after the
      dynamic event with post-trigger RMSE `0.000599m` and final error
      `0.000000m`; frame `300` is `[0.0220, 0.3747, 0.1378]` in both DP input
      and metric state. This confirms the failure is not simply missing the
      moved target coordinate. Cosmos3 should provide the causal post-move
      target hole/peg/TCP/task path, but the controller still needs a learned
      or scored DP-preserving short action chunk that can physically insert
      and retain along that path, then re-observe.
      Implementation follow-up added default-off
      `save_wam_candidate_action_chunks` logging in
      `scripts/world_model/evaluate_rebinding_controller.py` plus
      `SAVE_WAM_CANDIDATE_ACTION_CHUNKS` plumbing in
      `scripts/slurm/run_cosmos_task_state_controller_in_allocation.sh`.
      When explicitly enabled, future rollout-MPC event logs include every
      imagined candidate action chunk and simulated step reports for
      WAM/action-generator hard-negative or ranking dataset construction.
      This does not change candidate selection or success metrics. Syntax
      checks passed. New tmux-backed one-H200 allocation request
      `118609`/`wm_rgbd_wam_h200_0608_170956` is pending on `gpu` for
      `1-00:00:00` and must be reused once it starts instead of submitting
      short one-shot jobs.
      Added `scripts/world_model/build_wam_candidate_ranking_dataset.py` to
      export those logged candidates into a separate
      `dataset_kind=wam_candidate_ranking` H5. It stores current policy/RGB-D
      slot state, controller-time Cosmos future from the WAM report, each
      candidate action chunk, simulated candidate step outcomes, selected
      labels, and insertion/retention scores. It deliberately writes
      `positive_takeover_teacher_ok=false` and `method_evidence_allowed=false`
      because these are planner-imagined ranking/hard-negative labels, not
      visually verified live takeover positives. `py_compile` passed. A
      negative check on the old terminal-held hard-case H5 failed as expected
      with "not created with --save-wam-candidate-action-chunks", proving the
      exporter does not silently fabricate missing candidate actions.
      Prepared allocation-only run script
      `experiments/world_model_task_rebinding/rebinding_controller/wam_candidate_logging_rgbd_holeconstant_seed702000_no_video_20260608/run_in_allocation.sh`.
      It reproduces the RGB-D `hole_constant` seed `702000` terminal-held
      WAM hard case, keeps video off, enables
      `SAVE_WAM_CANDIDATE_ACTION_CHUNKS=true`, and requires an active
      `SLURM_JOB_ID`. `bash -n` and key input path checks passed. Once
      allocation `118609` starts, run this inside the tmux/salloc shell, then
      it automatically exports
      `wam_candidate_ranking_dataset_h16_a8.h5` and `.json` via
      `scripts/world_model/build_wam_candidate_ranking_dataset.py`.
      Added `scripts/world_model/train_wam_candidate_ranker.py` for the next
      step after candidate export. It trains
      `current policy/RGB-D slot state + Cosmos future task state + candidate
      action chunk -> physical quality / insertion / grasp-retention`
      predictors from the ranking H5. The training target is derived from
      simulated candidate outcome, not from the current bad selector alone,
      and the selected-label head has low diagnostic weight. The script
      refuses non-method candidate-ranking data by default; explicit
      `--allow-nonmethod-candidate-ranking-smoke` is required because these
      labels are planner-imagined hard negatives, not visually verified live
      positives. `py_compile` and `--help` passed.
      At `2026-06-08T17:24:27+08:00`, allocation `118609` was still pending.
      Started tmux watcher `wm_candidate_logging_watch_0608_172427`; it polls
      only `118609` and will send the one-shot allocation command to
      `wm_rgbd_wam_h200_0608_170956` once the allocation is running. It does
      not submit Slurm jobs or release the allocation.
      Runtime follow-up wired optional WAM candidate-ranker checkpoints into
      `scripts/world_model/evaluate_rebinding_controller.py` and the allocation
      wrapper. New args/env include
      `wam_candidate_ranker_checkpoint`,
      `wam_candidate_ranker_score_weight`, and
      `WAM_CANDIDATE_RANKER_CHECKPOINT`. The runtime builds the same current
      state + Cosmos future + action chunk feature, predicts quality/insertion/
      grasp retention, and adds `-score_weight * quality` to rollout-MPC
      candidate scoring. Non-method ranker checkpoints require explicit
      allow-nonmethod flags. The report records selected ranker availability
      and weighted score. `py_compile` and wrapper `bash -n` passed. This is
      still a candidate-selection hook only; live metric success and visual
      review remain authoritative.
      Runtime follow-up inside allocation `118609` on `server43` checked the
      user hypothesis that DP may only need a Cosmos target hole/path plus
      insertion-window action-prior chunks:
      `experiments/world_model_task_rebinding/rebinding_controller/wam_action_prior_candidate_rgbd_holeconstant_seed702000_no_video_20260608`.
      It used RGB-D slots, receding Cosmos task state, the mixed WAM scorer,
      candidate ranker, and
      `dp_official_wmcondition_21chunk_insert_window_pre64_post16.h5`.
      The run failed (`success_once=false`, `success_at_end=false`, final
      grasp true, final metric peg-head-at-hole
      `[-0.141985, -0.064468, -0.030080]`). Candidate audit:
      selected kind was `dp_bridge_blend=195`; action-prior teacher
      candidates were present only in a few support frames
      (`teacher_action/local_delta/tcp_local_delta/state_follower=3` each),
      and no candidate family had `inserted_any` or `final_inserted`.
      Visual review of
      `.../visual_review_envstate_default_view_1024_stride8_30fps_20260608/contact_sheet.png`
      and `frame_00300.png` shows the peg is held and approaches the box, but
      the peg head remains outside/below the hole. Boundary: negative
      RGB-D/Cosmos/WAM diagnostic, not method evidence. Updated conclusion:
      this is not solved by merely showing/creating the target hole for DP or
      adding a small static action-prior chunk library. The next aligned work
      is stronger causal WAM action generation/scoring from current RGB-D
      state + Cosmos receding target trajectory, with positive/negative
      short-chunk supervision admitted by real visual/contact evidence,
      followed by short-prefix execution and re-observation.
      Follow-up after the defaultbridge positive/negative ranker-guided
      diagnostic wrote an explicit manual visual review artifact at
      `experiments/world_model_task_rebinding/rebinding_controller/wam_defaultbridge_posneg_ranker_guided_rgbd_holeconstant_seed702000_no_video_20260608/manual_visual_review.json`
      and ran the readiness/admission check at
      `experiments/world_model_task_rebinding/wm_policy_distillation/readiness_checks/defaultbridge_posneg_ranker_guided_20260608/readiness.json`.
      Readiness reports `positive_controller_source_count=0`,
      `scaffold_controller_source_count=1`, and
      `rejected_controller_source_count=1`: policy contract is present,
      metric final success and visual hold/insertion are true, but the source
      remains scaffold-only because selected chunks include diagnostic
      bridge/probe variants, the ranker was trained from planner-imagined
      non-method labels, final RGB-D/control inserted is false, and Cosmos
      has no live closed-loop refresh. This preserves the current conclusion:
      DP is not simply missing the moved target hole in the audited state
      stack. Cosmos3/WAM must provide a causal post-move target
      hole/peg/TCP/task trajectory, but method progress also requires a
      DP-preserving learned or scored short-action generator that physically
      realizes and retains insertion along that trajectory.
      Follow-up `2026-06-08T19:56+08:00`: trained scaffold-only freeze-base
      WM-policy distillation for `6000` iterations on allocation `118609`
      (`server43`) from the defaultbridge/ranker diagnostic source. Static
      preservation smoke with the diffusion path reached `success_at_end=0.7`
      over `10` episodes; direct-head candidates remain blocked. Dynamic
      diagnostic
      `experiments/world_model_task_rebinding/rebinding_controller/wam_scaffold_policy_candidate_ranker_guided_rgbd_holeconstant_seed702000_no_video_20260608`
      used RGB-D slots, receding Cosmos task state, the scaffold policy
      checkpoint, WAM scorer, and candidate ranker. Metrics are
      `success_once=true`, `success_at_end=true`, final metric grasp true,
      final metric `peg_head_at_hole`
      `[-0.012992, -0.002199, -0.002863]`, but
      `world_model_closed_loop_refresh=false` and final RGB-D/control inserted
      is false. Visual review of
      `.../visual_review_envstate_default_view_1024_stride8_30fps_20260608/contact_sheet.png`
      and `frame_00300.png` shows pickup, grasp retention, approach, visible
      insertion/hold-success behavior, and no peg fly-up/lost-grasp failure.
      Candidate export has `sample_count=364`, `selected_count=12`,
      `wm_policy_action_generator=48` candidates, but selected learned
      WM-policy action-generator chunks `0`; selected chunks were
      bridge/scaffold actions. Manual review artifact:
      `.../manual_visual_review.json`. Boundary: non-method scaffold
      diagnostic only. The answer to the user hypothesis is therefore precise:
      Cosmos3 must supply the causal target hole/peg/TCP/task path, but the
      current DP failure is not solved by target visibility alone; the missing
      method component is a DP-preserving WAM/action generator or scorer that
      turns that path into executable short chunks, executes only a prefix,
      and re-observes.
      Follow-up limited diagnostic
      `experiments/world_model_task_rebinding/rebinding_controller/wam_scaffold_policy_limited_vs_dpbridge_rgbd_holeconstant_seed702000_no_video_20260608`
      removed bridge variants, insert-probe candidates, and action-prior
      teacher candidates, leaving only base `dp_bridge_blend` and scaffold
      `wm_policy_action_generator` candidates under the same RGB-D/Cosmos
      inputs. It failed (`success_once=false`, `success_at_end=false`, final
      metric grasp true, final metric `peg_head_at_hole`
      `[-0.110701, -0.015715, -0.019758]`). Candidate export:
      `sample_count=165`, `dp_bridge_blend=143`,
      `wm_policy_action_generator=22`, selected chunks `11`, selected
      WM-policy chunks `3`, but `inserted_any_count=0` and
      `final_inserted_count=0`. Visual review shows the peg is held and
      approaches the moved box/hole, but remains outside/below the hole at
      the end. Boundary: negative scaffold action-generator evidence, not
      method success. Updated next step: train/score a stronger WAM/action
      generator on current RGB-D-derived state plus Cosmos receding target
      trajectory plus positive/negative short chunks; do not reduce the issue
      to target-hole visibility or a prompt/video-only Cosmos output.
- [ ] 2026-06-08 DDP task-frame policy diagnostic after user DP/Cosmos
      question: documented in
      `docs/world_model_task_rebinding/2026-06-08_ddp_taskframe_directhead_dynamic_retention_diagnostic.md`.
      The audited state-DP observation already contains the current
      `box_hole_pose`, so the hard failure is not simply "DP cannot see the
      moved hole." The physical gap is converting the changed target task
      frame into executable short action chunks that keep the peg grasped,
      insert, and retain final insertion. Updated
      `scripts/world_model/train_wm_conditioned_policy_distillation.py` so
      static partial takeover samples use task-frame-aware 64D conditions
      derived from current/future TCP, peg, hole, peg-head-at-hole, relative
      geometry, teacher action mean, and hole radius; also added direct-head
      admission guards. The 3000-step frozen-base run
      `experiments/world_model_task_rebinding/wm_policy_distillation/train_ddp_static_taskframe_directhead_freeze_base_3000iter_20260608_1525`
      completed on allocation `118088`/`server44`. Static preservation:
      direct action head still fails (`success_at_end=0.0`, `10` episodes),
      while the diffusion path preserves the base DP in a small smoke
      (`success_at_end=0.8`, `10` episodes). Dynamic RGB-D/Cosmos hard-case
      diagnostic
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_diffusion_candidate_taskframe3000_holeconstant_defaultbridge_seed702000_no_video_20260608_1535`
      has `success_once=true` but `success_at_end=false`, inserted metric
      frames `148..195`, final grasp true, and final metric peg-head-at-hole
      `[-0.057878, -0.002710, 0.002529]`. Its 1024px/30fps default-view replay
      shows no gross peg fly-up/lost-grasp failure; the peg is briefly
      inserted, then final retention is lost. The event audit found WM-policy
      action-generator candidates were produced but not selected, so this run
      does not prove learned WM-policy control. Boundary: negative diagnostic,
      not method success and not positive takeover distillation data. Next
      aligned step is not threshold-gate tuning; make selected short chunks
      actually use a causal Cosmos/WAM objective and include final-retention
      scoring/training, while keeping direct-head candidates blocked until
      static preservation passes.
- [ ] 2026-06-08 WAM interface correction after literature/local audit:
      user correction is right that the active controller should use Cosmos3
      as a world/action model, not as a loose video/trajectory side channel.
      Literature check recorded in
      `docs/world_model_task_rebinding/2026-06-08_wam_interface_literature_and_failure_audit.md`
      supports a WAM-style interface: current RGB-D/proprio state plus
      Cosmos-predicted future object/task trajectory plus candidate/teacher
      action chunk should predict future task state and produce executable
      short action chunks. Added read-only audit script
      `scripts/world_model/audit_wam_controller_interface.py` and ran it on
      `wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658`.
      Audit artifacts:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658/wam_interface_audit.json`
      and
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658/wam_interface_audit.md`.
      Findings: DP state obs contains the current metric hole pose in this
      rollout (post-trigger RMSE `0.000346m`, final error `0.000000m`), so the
      state-DP failure is not simply because it cannot see the current hole.
      Cosmos trajectory has `465` rows with `hole_pose`, `peg_pose`,
      `tcp_pose`, and `peg_head_at_hole`, but the policy H5 only has a
      `(301,64)` snapshot `wm_policy_condition` and no temporal future
      trajectory/WAM condition dataset. RGB-D slot drift near insertion is
      large: final hole-position L2 `0.043888m`, final
      peg-head-at-hole L2 `0.038503m`, with no inserted frames. Conclusion:
      stop treating the 64D bridge summary as the WAM interface. Next aligned
      implementation step is to construct explicit temporal WAM conditions
      from Cosmos future task trajectory plus action chunks and future
      object-state labels, then train/evaluate a DP-preserving short-chunk
      policy/scorer on that interface. The audited failed rollout remains
      negative evidence and not positive takeover data. Implemented the first
      structural WAM export script
      `scripts/world_model/build_wam_temporal_condition_dataset.py` and
      exported
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658/wam_temporal_condition_dataset_h16_a8.h5`
      plus JSON report. It contains `86` post-trigger samples with policy
      history `[2,43]`, current slot state `30D`, Cosmos future state
      `[16,27]`, action chunk `[8,7]`, and future metric state `[16,27]`.
      The source rollout remains failed (`inserted_any=false`,
      `final_inserted=false`, `final_grasped=true`), so
      `positive_takeover_teacher_ok=false`. Cosmos imagined future versus
      actually reached future has large gap: hole-position mean/max L2
      `0.068791m/0.099756m`, peg-head-at-hole mean/max L2
      `0.399634m/0.431472m`. This quantifies that current actions do not
      execute the Cosmos imagined path and motivates WAM action-state
      training/scoring rather than more old-controller threshold tuning.
      Added `scripts/world_model/train_wam_action_state_scorer.py`; it
      predicts future metric state/grasp/inserted from
      `policy_obs_history + current_slot_state + cosmos_future_state +
      action_chunk`. Default dry-run correctly rejects the failed-only WAM
      dataset, and explicit `--allow-structural-negative-smoke` loads it
      (`feature_dim=604`, `target_dim=432`, train/val `69/17`). A 50-step
      structural-negative smoke ran inside held allocation `117750` on
      `server21`:
      `experiments/world_model_task_rebinding/wam_action_state_scorer/structural_negative_smoke_50iter_20260608`.
      Train/val state RMSE dropped from `0.415898/0.413177` to
      `0.127003/0.119454`, proving only that the WAM interface/model path
      fits. Since all inserted labels are negative, `inserted_acc=1.0` is
      trivial and has no method meaning. Next required step: construct/admit
      positive WAM temporal datasets from visually verified successful
      teacher/takeover chunks, or train a contrastive scorer over successful
      versus failed candidate chunks, then use it for short-chunk candidate
      selection. Follow-up completed the first non-method action-prior WAM
      dataset and mixed scorer smoke: added
      `scripts/world_model/build_wam_action_prior_temporal_dataset.py` and
      exported
      `experiments/world_model_task_rebinding/wm_policy_distillation/wam_action_prior_temporal_dataset_h16_a8_20260608.h5`
      plus JSON report. It admits only visually reviewed successful DP
      action-prior groups (`19` included, `2` excluded) and stores `997`
      samples with the same `604`-dim scorer feature contract. The future
      condition source is explicitly
      `metric_future_as_desired_action_prior`, so this is teacher/scorer
      structure, not controller-time Cosmos evidence:
      `positive_takeover_teacher_ok=false`,
      `candidate_action_prior_teacher_ok=true`,
      `method_evidence_allowed=false`. Updated
      `scripts/world_model/train_wam_action_state_scorer.py` so these chunks
      are accepted only under `--allow-action-prior-positive-smoke`; default
      admission still rejects them as non-positive-takeover data. A mixed
      action-prior plus failed-Cosmos 300-step scorer smoke ran inside held
      allocation `117750`:
      `experiments/world_model_task_rebinding/wam_action_state_scorer/action_prior_plus_failed_contrastive_smoke_300iter_20260608`.
      It used `1083` samples and reduced train/val state RMSE from
      `0.406120/0.406248` to `0.050978/0.050628`, with final train/val
      inserted and grasp accuracy `1.0/1.0`. Manifest identity remains
      non-method: `positive_source_count=0`,
      `action_prior_positive_smoke_source_count=1`,
      `method_source_count=0`. Interpretation to preserve: the DP failure is
      not simply state-DP blindness to the moved hole; the missing mechanism is
      a causal Cosmos/RGB-D target task-frame trajectory plus learned
      action-state scoring/short-chunk execution. Next controller work should
      connect this scorer to candidate selection over causal Cosmos future
      trajectories, execute only short chunks, and re-observe. Runtime hook
      follow-up: extended `scripts/world_model/evaluate_rebinding_controller.py`
      with optional `--wam-scorer-checkpoint` support for
      `control_policy=wm_dp_prior_rollout_mpc`. Cosmos task-state predictor now
      exposes `future_state_sequence(...)`, and
      `WAMActionStateScorerRuntime` builds the same `604`-dim feature contract
      for candidate ranking. Non-method action-prior checkpoints require
      `--wam-scorer-allow-nonmethod-smoke`; the current scorer manifest has
      `method_source_count=0`, `positive_source_count=0`, and
      `action_prior_positive_smoke_source_count=1`. A runtime probe on failed
      frame `90` produced a valid receding future sequence with no missing
      frames and predicted low insertion probability (`~7e-06`) but high grasp
      probability (`0.999912`), matching the observed failure type. A RGB-D
      rollout attempt on held allocation `117750`/`server21` failed before
      controller evaluation during RGB-D slot `vision_env` construction with
      Vulkan `ErrorDeviceLost`; this is a rendering/perception environment
      failure, not method evidence. The failed step was killed while keeping
      allocation `117750` running. A no-video oracle-slot code-path smoke did
      run:
      `experiments/world_model_task_rebinding/rebinding_controller/wam_scorer_rollout_mpc_oracle_seed702000_no_video_20260608`.
      It reached metric `success_at_end=true`, inserted frames `167..300`, and
      final peg-head-at-hole `[0.009820, 0.001172, 0.003026]`, with `2543` WAM
      scorer reports and `435` available future-state scorer reports in
      candidate summaries. Boundary: oracle/no-video scaffold only, not RGB-D
      method evidence and not positive Cosmos/RGB-D takeover data. RGB-D
      follow-up used a new held allocation `118088` on `server44` after
      server44 passed RGB-D vision-env preflight, without releasing the held
      server21 allocation. Run
      `experiments/world_model_task_rebinding/rebinding_controller/wam_scorer_rollout_mpc_rgbd_seed702000_no_video_20260608_server44`
      used the RGB-D slot ensemble, receding Cosmos trajectory, and the mixed
      WAM scorer checkpoint with `--wam-scorer-allow-nonmethod-smoke`. It
      failed task completion: `success_once=false`, `success_at_end=false`,
      inserted frames `0`, final grasp true, final metric peg-head-at-hole
      `[-0.112917, 0.001173, -0.007083]`, and final head L2 `0.113145m`.
      Video review from the full 1024px/30fps replay
      `.../wam_scorer_rollout_mpc_rgbd_seed702000_no_video_20260608_server44/visual_review_envstate_default_view_1024_full301_30fps_20260608/state_replay.mp4`
      confirms the robot keeps the peg and approaches the box, but the peg
      head remains outside the hole. RGB-D slot drift is a concrete blocker:
      hole-position max L2 `0.116639m`, peg-head-at-hole max L2 `0.129873m`,
      and at frame `100` shortly after the dynamic event the hole-position
      error is `0.101763m`. Boundary: negative RGB-D/Cosmos/WAM-scorer
      controller diagnostic only. It is not success and not positive takeover
      data. Post-run logging fix added `selected_wam_scorer_report` and
      compact selected-WAM step fields to
      `scripts/world_model/evaluate_rebinding_controller.py`; syntax check
      passed. Selected-log follow-up produced two important diagnostics.
      First, a default `hole_move_stop` run
      `wam_scorer_rollout_mpc_rgbd_seed702000_selectedlog_no_video_20260608_server44`
      succeeded with RGB-D slots and visual replay (`success_once=true`,
      `success_at_end=true`, inserted frames `130..300`, final head L2
      `0.010951m`). Its 1024px/30fps replay shows real pickup, insertion, and
      retention, but it accidentally used default bridge settings and
      `scenario=hole_move_stop`, so it must not be used as proof that the
      hard `hole_constant` case is solved. Second, the exact
      `hole_constant` selected-log reproduction
      `wam_scorer_rollout_mpc_rgbd_holeconstant_seed702000_selectedlog_no_video_20260608_server44`
      failed the same way as before (`success_once=false`,
      `success_at_end=false`, inserted frames `0`, final grasp true, final
      head L2 `0.113145m`). Selected-WAM evidence in that exact failure:
      selected candidate kinds `dp_bridge_blend=184`, `bridge_variant=16`;
      selected WAM reports `available=9`,
      `future_state_sequence_unavailable=15`, and `none=176`; candidate WAM
      reports `1172` total with `437` available. Conclusion: the active hard
      failure is not simply that DP cannot see the moved hole. The current
      controller mostly selects DP/bridge chunks without a consistently
      available WAM future-state constraint, while RGB-D target slots drift
      around the critical post-move frames. Next required step is to reduce
      post-move RGB-D target drift and make WAM scoring/action generation
      apply to the actually selected short chunks under live re-observation,
      so the controller follows a stable Cosmos/RGB-D target trajectory
      instead of approaching the box without insertion.
- [ ] 2026-06-08 WM-policy admission and direct-head preservation correction:
      fixed `scripts/world_model/inspect_wm_policy_distillation_readiness.py`
      so readiness discovers current compliant visual artifacts from
      `visual_review*` replay directories and manual review paths, not only
      legacy `videos/`. Rechecked the RGB-D default DP-fallback failure:
      `readiness_checks/default_rgbd_dpfallback_20260608_0635_after_artifactfix/readiness.json`
      now finds the replay video and keyframe sheet but still rejects it for
      real reasons (`metric_or_slot_drift_check_failed`,
      `manual_visual_review_not_positive`). Rechecked the oracle success
      scaffold:
      `readiness_checks/oracle_success_scaffold_20260608_0640_after_artifactfix/readiness.json`
      finds its video and marks it `scaffold=True`, `positive=False`. Also
      added direct-head static preservation support to
      `scripts/world_model/evaluate_wm_policy_static_preservation.py` and
      `scripts/slurm/run_wm_policy_static_preservation_in_allocation.sh`.
      Under held allocation `117200` on `server13`, checkpoint
      `train_ddp_direct_action_head_21chunk_insertwindow_staticpartial25_freeze_base_24000iter_prior55_20260608_0025/checkpoints/final.pt`
      gets `success_at_end=0.6` for the diffusion path with zero WM condition
      (`static_preservation_eval_21chunk_insertwindow_diffusion_after_directfix_20260608_0652`)
      but `success_at_end=0.0` for direct action head execution
      (`static_preservation_eval_21chunk_insertwindow_directhead_20260608_0650`).
      Added a controller guard: enabling
      `wm_dp_prior_rollout_include_wm_policy_direct_head_candidates=true` now
      requires `--wm-policy-direct-head-static-preservation-json` whose report
      evaluated the direct head and whose `success_at_end` clears
      `--wm-policy-direct-head-min-static-success-at-end` (default `0.5`).
      Validation rejects the current direct-head report (`success_at_end=0.0`)
      under the default floor and passes only when the floor is explicitly
      lowered to `0.0`. Conclusion: the frozen DP diffusion path still has
      small-sample base preservation, but the direct action head cannot be an
      independent takeover controller or unguarded controller candidate until
      retrained with stronger preservation constraints. Evidence:
      `docs/world_model_task_rebinding/2026-06-08_wm_policy_admission_and_static_preservation.md`.
- [ ] 2026-06-08 RGB-D/Cosmos WM-policy partial takeover visual check:
      diagnostic
      `wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658`
      used the RGB-D slot ensemble, receding Cosmos trajectory, and
      WM-policy diffusion path as a partial takeover controller with action
      blend `0.35`; direct-head candidates were disabled. Rendering on
      allocation `117755`/`server27` produced a full 1024px/30fps/301-frame
      replay:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Keyframe sheet:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658/visual_review_envstate_default_view_1024_full301_30fps/keyframe_contact_sheet_pickup_to_final_512.png`.
      Manual review:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_rgbd_diffusion_blend035_seed702000_no_video_20260608_0658/visual_review_envstate_default_view_1024_full301_30fps/manual_visual_review.json`.
      Metrics: `success_once=false`, `success_at_end=false`, inserted frames
      `0`, final grasp true, metric grasped frames `54..300`, peg z range
      about `0.024m..0.126m`, final peg-head-at-hole
      `[-0.136800, 0.031347, -0.017658]`, `wm_policy_takeover_count=201`,
      `dp_count=99`, and `infeasible_dp_fallback_count=9`. Visual review:
      the peg does not fly up or flip skyward and remains near/in the gripper
      during late approach, but the peg head stays outside the hole. Boundary:
      negative RGB-D/Cosmos controller evidence only; the current
      diffusion-path partial takeover preserves pickup better than the old
      failed takeover but still lacks insertion alignment/contact execution
      and must not enter positive takeover distillation data.
- [ ] 2026-06-08 default RGB-D/Cosmos DP-fallback visual check:
      diagnostic
      `wm_dp_prior_rollout_mpc_rgbd_dpfallback_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0605`
      ran inside held allocation `117200` on `server13` with
      `slot_source=rgbd`, `cosmos_task_state_min_tau=0`, and
      `controller_infeasible_fallback_policy=dp_prior`. It failed task
      completion (`success_once=false`, `success_at_end=false`, inserted
      frames `0`) but did not show the earlier peg lift-away/fly-up:
      `retreat_count=0`, `infeasible_dp_fallback_count=10`, metric grasped
      frames `54..300`, final metric peg-head-at-hole
      `[-0.111758, 0.001683, -0.010846]`, and metric peg z range about
      `0.024m..0.148m`. The full 1024px/30fps/301-frame replay and inspected
      keyframe sheet are:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_dpfallback_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0605/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`
      and
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_dpfallback_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0605/visual_review_envstate_default_view_1024_full301_30fps/keyframe_contact_sheet_pickup_to_final_512.png`.
      Manual visual review confirms normal pickup/approach and preserved grasp
      without gross peg orientation failure or lift-away through frames
      `210..300`; the visible failure is non-insertion with the peg remaining
      outside the hole. Boundary: negative RGB-D/Cosmos controller evidence and
      DP-preservation fix only; not method success and not positive takeover
      distillation data.
- [ ] 2026-06-08 DP-prior fallback fix for infeasible WM/controller states:
      added `controller_infeasible_fallback_policy` to
      `scripts/world_model/evaluate_rebinding_controller.py` and wired
      `CONTROLLER_INFEASIBLE_FALLBACK_POLICY` through
      `scripts/slurm/run_cosmos_task_state_controller_in_allocation.sh`.
      The default is now `dp_prior`; old upward `safe_retreat` remains
      available as an explicit diagnostic. Physical reason: the controller is
      meant to improve the frozen static DP with RGB-D/Cosmos rebinding, not
      destroy the DP's basic pickup behavior when the WM/perception bridge is
      unavailable. This changes controller action selection under infeasible
      WM/slot conditions only; it does not change the success metric or make
      fallback-only behavior method success. Syntax checks passed with
      `python3 -m py_compile`, `.venv/bin/python -m py_compile` inside
      allocation `117200`, and `bash -n` on the wrapper. The diagnostic
      `wm_dp_prior_rollout_mpc_rgbd_mintau4_dpfallback_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0555`
      ran inside held allocation `117200` on `server13` with
      `slot_source=rgbd`, `cosmos_task_state_min_tau=4`, and
      `controller_infeasible_fallback_policy=dp_prior`. It still failed
      (`success_once=false`, `success_at_end=false`, inserted frames `0`), but
      it eliminated the lift-away failure: `retreat_count=0`,
      `infeasible_dp_fallback_count=122`, metric grasped frames `54..300`,
      final metric peg z about `0.113m`, and final metric peg-head-at-hole
      `[-0.110683, -0.041798, -0.025048]`. Full 1024px/30fps replay and
      inspected keyframe sheet:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_mintau4_dpfallback_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0555/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`
      and
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_mintau4_dpfallback_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0555/visual_review_envstate_default_view_1024_full301_30fps/keyframe_contact_sheet_pickup_to_final.png`.
      Manual visual review confirms normal pickup and preserved grasp without
      fly-up/lift-away, but no insertion. Boundary: negative RGB-D/Cosmos
      controller localization and DP-preservation fix only; not method success
      and not positive takeover distillation data. Evidence note:
      `docs/world_model_task_rebinding/2026-06-08_controller_visual_and_hold_failure_localization.md`.
- [ ] 2026-06-08 current held allocation and RGB-D min-tau visual failure:
      previous held allocation `116575` on `server40` is no longer running;
      `sacct` reports `CANCELLED by 0` at `2026-06-08T05:21:34+08:00` after
      `04:06:20`. A new tmux/salloc hold allocation `117200` is running on
      `server13` (`wm_h200_tmux_hold_0608b`) and must not be released. Added
      default-off diagnostics `cosmos_task_state_min_tau` and
      `cosmos_task_state_reject_unavailable`. A first attempt exposed and
      fixed an implementation bug where `min_tau` was incorrectly applied to
      explicit internal `tau_candidates=(0,)` rollout-MPC bridge generation.
      The corrected RGB-D diagnostic
      `wm_dp_prior_rollout_mpc_rgbd_mintau4_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0530`
      ran with `slot_source=rgbd`, `cosmos_task_state_min_tau=4`, and
      `cosmos_task_state_reject_unavailable=false`. It failed:
      `success_at_end=false`, `success_once=false`, inserted frames `0`,
      metric grasped frames `54..300`, controller stdout
      `infeasible_count=58`, `retreat_count=58`, final metric
      peg-head-at-hole `[-0.142177, 0.088691, 0.478954]`, and final metric
      peg z about `0.585m`. Full 1024px/30fps replay and inspected keyframe
      sheet:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_mintau4_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0530/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`
      and
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_mintau4_successrollout8x4_completion_axis12_seed702000_no_video_20260608_0530/visual_review_envstate_default_view_1024_full301_30fps/keyframe_contact_sheet_pickup_to_final.png`.
      Visual review: pickup and early approach are normal, but from about
      frame `210` the grasped peg is lifted far above the hole. Interpretation:
      the default RGB-D failure is not immediate pickup/peg fly-up, but simply
      excluding `tau=0` without enough receding Cosmos coverage causes a new
      lift/retreat failure. Boundary: negative RGB-D/Cosmos controller
      diagnostic only, not method evidence and not positive distillation data.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-08_controller_visual_and_hold_failure_localization.md`.
- [ ] 2026-06-08 rollout-MPC hold visual success localization:
      no-video rollout step `116575.30` completed inside held allocation
      `116575` on `server40` in `00:12:03`, writing
      `metrics.json` and `rollouts.h5` for
      `wm_dp_prior_rollout_mpc_episode_restore_localaxis_successrollout8x4_lightpost_completion_axis12_seed702000_no_video_20260608_0414`.
      Metrics are `success_at_end=true`, `success_once=true`, final grasp
      true, inserted frames `167..300` (`134`), grasped frames `54..300`,
      and final peg-head-at-hole
      `[-0.001794, 0.000714, -0.002903]`. Render job `117052` completed on
      `server13` in `00:01:55` and produced the 1024px, 30fps, 301-frame
      replay:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_episode_restore_localaxis_successrollout8x4_lightpost_completion_axis12_seed702000_no_video_20260608_0414/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Inspected frames `54`, `90`, `167`, `230`, and `300` show normal pickup,
      stable grasp, no fly-up/large flip/tilt-to-sky, visible insertion, and
      final retention. Manual review:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_episode_restore_localaxis_successrollout8x4_lightpost_completion_axis12_seed702000_no_video_20260608_0414/manual_visual_review.json`.
      Boundary: this is the first positive single-seed oracle-slot controller
      scaffold/localization result in this sequence, not RGB-D/Cosmos method
      evidence and not positive takeover distillation data by itself.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-08_controller_visual_and_hold_failure_localization.md`.
- [ ] 2026-06-08 RGB-D slot smoke after oracle hold success:
      job `117075` ran the same short-chunk rollout-MPC hold configuration on
      `server13`, but with `slot_source=rgbd` and RGB-D slot ensemble
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job102292`.
      It completed in `00:19:10` and failed:
      `success_at_end=false`, `success_once=false`, inserted frames `0`,
      final grasp true, final peg-head-at-hole
      `[-0.111166, 0.000307, -0.010842]`. Post-trigger slot-vs-metric RMSE is
      peg-head-at-hole `0.03175m`, hole position `0.02421m`, and peg position
      `0.01523m`. Render job `117169` completed on `server13` in `00:01:42`
      and produced the 1024px, 30fps, 301-frame replay:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_successrollout8x4_lightpost_completion_axis12_seed702000_no_video_20260608_0435/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Inspected frames `54`, `90`, `167`, `230`, and `300` show normal pickup
      and stable grasp with no fly-up/large flip/tilt-to-sky, but the peg stays
      outside/short of insertion through the final frame. Manual review:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_rgbd_successrollout8x4_lightpost_completion_axis12_seed702000_no_video_20260608_0435/manual_visual_review.json`.
      Boundary: negative RGB-D-derived controller evidence; next work should
      analyze slot/controller drift and candidate selection, not retune oracle
      hold thresholds.
- [ ] 2026-06-08 controller visual and hold localization:
      render job `116796` completed on `server13` and produced a 1024px,
      30fps, 301-frame replay for
      `wm_dp_prior_rollout_mpc_episode_restore_localaxis_completion_axis12_seed702000_no_video_20260608_0245`.
      Keyframe sheet:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_episode_restore_localaxis_completion_axis12_seed702000_no_video_20260608_0245/visual_review_envstate_default_view_1024_keyframes_30fps/contact_sheet.png`.
      Full video:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_episode_restore_localaxis_completion_axis12_seed702000_no_video_20260608_0245/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Manual review says the peg is picked up normally, stays grasped, and
      does not fly up or tilt to the sky. It visibly reaches/enters the hole
      around frames `181..201`, but `success_hold_policy=dp` pulls it back
      out; final success is false. Follow-up hold diagnostics in held
      allocation `116575` on `server40` show: DP handoff inserted `21` frames,
      servo 4mm first-latch inserted `93` frames, servo 8mm/gain1 inserted
      `85` frames, zero hold inserted `18` frames, and servo 4mm deepest-latch
      inserted `62` frames with final x `-0.015494m` versus success threshold
      `-0.015m`. Render job `116902` completed on `server27` for the deepest
      near miss; full video:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_episode_restore_localaxis_successservo_deepest_completion_axis12_seed702000_no_video_20260608_0325/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Conclusion: current failure is not basic pickup/peg fly-up. The controller
      can transiently insert and keep grasp, but final contact retention still
      fails under the original metric. Boundary: oracle-slot controller
      scaffold only, not RGB-D/Cosmos method evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-08_controller_visual_and_hold_failure_localization.md`.
- [ ] 2026-06-08 insert-x hold render check:
      follow-up insert-x hold run
      `wm_dp_prior_rollout_mpc_episode_restore_localaxis_successservo_insertx0_completion_axis12_seed702000_no_video_20260608_0335`
      completed with `success_once=true`, `success_at_end=false`, final grasp
      true, inserted frames `181..294` (`114`), and final peg-head-at-hole
      `[-0.017480, 0.002951, 0.003044]`. Render job `116967` completed on
      `server13` in `00:02:06` and produced a 1024px, 30fps, 301-frame replay:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_episode_restore_localaxis_successservo_insertx0_completion_axis12_seed702000_no_video_20260608_0335/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Keyframe sheet:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_episode_restore_localaxis_successservo_insertx0_completion_axis12_seed702000_no_video_20260608_0335/visual_review_envstate_default_view_1024_keyframes_30fps/contact_sheet.png`.
      Manual review:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_episode_restore_localaxis_successservo_insertx0_completion_axis12_seed702000_no_video_20260608_0335/manual_visual_review.json`.
      Visual conclusion: peg pickup is normal, the peg stays stably grasped,
      and there is no fly-up, large flip, or tilt-to-sky failure. The failure
      remains final insertion/contact retention; insert-x hold increased
      inserted-frame count but ended slightly worse than the deepest-latch
      near miss. This is negative oracle-slot controller localization only,
      not RGB-D/Cosmos method evidence.
- [ ] 2026-06-08 latest controller failure localization:
      historical held allocation `116575` on `server40`
      (`wm_controller_h200_hold_cpu1d_0608`) was verified with `NVIDIA H200`
      during this diagnostic sequence, but it is no longer current; see the
      2026-06-08 allocation update above for the active `117200` server13
      hold. Previous allocation `116026` disappeared and
      `sacct` reports `CANCELLED by 0` after `04:02:26` on `server13`. During
      replacement, allocation `116567` briefly reached `server40` but was
      interrupted; record this as scheduling/resource evidence. Action-prior
      replay diagnostic
      `wm_dp_prior_rollout_mpc_action_prior_replay_insertwindow21_seed702000_no_video_20260608_0125`
      failed with final grasp true and final peg-head-at-hole
      `[-0.11914539337158203, 0.009525928646326065, -0.034445807337760925]`;
      no teacher/direct/DP candidate inserted. Server40 render of that run
      hung at frame `0` and wrote no PNG frames, so it is not visual evidence.
      Insert-probe diagnostic
      `wm_dp_prior_rollout_mpc_insert_probe_direct_head_seed702000_no_video_20260608_0135`
      also failed with final grasp true and final peg-head-at-hole
      `[-0.11089277267456055, 0.01614382490515709, -0.04000784456729889]`.
      Probe candidates could laterally align, best about `1.8mm`, but stayed
      around insertion-axis x `-0.089m`; no inserted candidate. Current
      high-bridge-blend follow-up
      `wm_dp_prior_rollout_mpc_high_bridge_blend_insert_probe_seed702000_no_video_20260608_0148`
      also failed with final grasp true, no inserted frames, and final
      peg-head-at-hole
      `[-0.11413013935089111, 0.01104189082980156, -0.03415173292160034]`.
      Metric slots across the 6/8 diagnostics show grasp first true around
      frame `54` and final grasp true at frame `300`; the exact rendered
      insert-window video/contact sheet shows normal pickup and no peg
      tilt-to-sky. Therefore the current failure is stable-grasp
      insertion-axis/contact/orientation control, not peg fly-up. A fresh
      server40 render canary in allocation `116575` again hung at
      `render_rgb_array_start`, so server40 is not accepted as a visual
      evidence node until rendering is revalidated. Follow-up render job
      `116625` reached `server55` but failed immediately due to a command-line
      boolean argument error; corrected job `116629` reached `server55` and
      again hung at first-frame `render_start`, then was canceled after
      `00:05:41` to avoid wasting resources. The high-bridge-blend run
      therefore remains metric-only failure localization, not visual evidence.
      Boundary: oracle/scaffold failure localization only, not RGB-D/Cosmos
      method evidence.
- [ ] 2026-06-08 completion-axis rollout-MPC smoke:
      added a default-off rollout-MPC score term
      `wm_dp_prior_rollout_completion_axis_weight` so candidate scoring can
      penalize remaining distance from the real insert target `insert_x`
      across all phases. Physical reason: the planner was rewarding
      align-phase candidates for parking at pre-insert x around `-0.11m`,
      which is not task completion. This changes only candidate scoring, not
      the final success metric. Single-seed smoke
      `wm_dp_prior_rollout_mpc_completion_axis12_insert_probe_seed702000_no_video_20260608_0210`
      ran in held allocation `116575` with weight `12.0`; it still failed
      (`success_at_end=false`, `success_once=false`) with final grasp true,
      inserted frames `0`, max x about `-0.1113m`, and final
      peg-head-at-hole
      `[-0.11236929893493652, 0.00887833908200264, -0.020312055945396423]`.
      Candidate selection changed (`wm_policy_direct_action_head` increased
      from `8` to `48` reports, `dp_bridge_blend` decreased from `186` to
      `145`), but live insertion did not improve. Do not keep weight-tuning
      this as a substitute for a real action generator or planner/live contact
      fix. Boundary: negative oracle/controller scaffold evidence only.
- [ ] 2026-06-08 latest insert-window render check:
      exact saved-state render for
      `wm_dp_prior_rollout_mpc_direct_head_insertwindow21_staticpartial25_24000iter_allow_noninsert_seed702000_no_video_20260608_0100`
      completed inside held allocation `116026` on `server13`, using the
      default oblique ManiSkill view at 1024px/30fps. Full video:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_insertwindow21_staticpartial25_24000iter_allow_noninsert_seed702000_no_video_20260608_0100/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Key-frame sheet:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_insertwindow21_staticpartial25_24000iter_allow_noninsert_seed702000_no_video_20260608_0100/visual_review_envstate_default_view_1024/contact_sheet.png`.
      Manual review:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_insertwindow21_staticpartial25_24000iter_allow_noninsert_seed702000_no_video_20260608_0100/manual_visual_review.json`.
      Conclusion: not a peg fly-up/basic-pickup explosion. The peg is picked
      up and held, but insertion still fails. Metrics:
      `success_at_end=false`, `success_once=false`, final grasp true, final
      peg-head-at-hole
      `[-0.1636543720960617, 0.007629893720149994, -0.04722881317138672]`.
      Rollout-MPC selected `wm_policy_direct_action_head` only `24/210`
      times, selected `dp_bridge_blend` `186/210` times, and no selected or
      candidate chunk ever inserted. Current bottleneck is still
      approach/alignment/contact execution near the moved hole. This remains
      negative controller/action-generator scaffold evidence only, not
      RGB-D/Cosmos method evidence and not positive takeover distillation data.
- [ ] 2026-06-07 latest rendered controller check:
      exact saved-state render for
      `wm_dp_prior_rollout_mpc_direct_head_staticpartial25_allow_noninsert_seed702000_no_video_20260607_2130`
      completed inside held allocation `116026` on `server13`, using the
      default oblique ManiSkill view at 1024px/30fps. Full video:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_staticpartial25_allow_noninsert_seed702000_no_video_20260607_2130/visual_review_envstate_default_view_1024_full301_30fps/state_replay.mp4`.
      Key-frame sheet:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_staticpartial25_allow_noninsert_seed702000_no_video_20260607_2130/visual_review_envstate_default_view_1024/contact_sheet.png`.
      Manual review:
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_staticpartial25_allow_noninsert_seed702000_no_video_20260607_2130/manual_visual_review.json`.
      Conclusion: not a peg fly-up/basic-pickup failure. Metric grasp is true
      from frame `54` through frame `300`, but inserted frames are `0` and
      final peg-head-at-hole is
      `[-0.07179862260818481, -0.0004356801509857178, 0.00246545672416687]`.
      Current bottleneck is insertion-axis/contact execution near the hole.
      This remains negative oracle/Cosmos scaffold evidence only, not RGB-D
      method evidence and not positive takeover distillation data.
- [ ] 2026-06-07 latest DP-only visual admission:
      inside held allocation `116026` on `server13`, the additional DP-only
      action-prior scan
      `dp_only_wmcondition_action_prior_scan_seed702413_48eps_20260607_2140`
      found `8/48` final successes and `16/48` success-once episodes. Four
      final-success/final-grasp seeds were rendered and manually inspected:
      `702440`, `702444`, `702447`, and `702451`. The 1024px contact sheets
      show normal pickup and no immediate peg fly-up/tilt-to-sky. These are
      admitted only as DP action-prior candidates, not positive takeover data
      and not RGB-D/Cosmos method evidence. Extracted four action-prior chunks
      to
      `experiments/world_model_task_rebinding/wm_policy_distillation/dp_wmcondition_action_prior_chunks_20260607/dp_wmcondition_seed702440_702451_action_prior_chunks.h5`
      and rebuilt
      `experiments/world_model_task_rebinding/wm_policy_distillation/ddp_action_generator_manifest_20260607_2330/manifest.json`
      with `21` accepted chunks. A 21-chunk frozen-base direct-action-head
      training run completed under
      `experiments/world_model_task_rebinding/wm_policy_distillation/train_ddp_direct_action_head_21chunk_staticpartial25_freeze_base_24000iter_prior55_20260607_2335`.
      Final step `23999`: total loss `0.0207542572170496`, direct action
      loss `0.009112123399972916`. Static preservation 10eps stayed at
      `success_at_end=0.7`, `success_once=0.7`. Dynamic seed-`702000`
      allow-noninsert rollout-MPC still failed; final grasp remained true but
      final peg-head-at-hole was
      `[-0.13852724432945251, 0.011310238391160965, -0.04001953452825546]`.
      The exact rendered 1024px contact sheet under
      `experiments/world_model_task_rebinding/rebinding_controller/wm_dp_prior_rollout_mpc_direct_head_staticpartial25_21chunk_24000iter_allow_noninsert_seed702000_no_video_20260608_0006/visual_review_envstate_default_view_1024/contact_sheet.png`
      again shows normal pickup/no peg fly-up, but no insertion. This is
      negative controller scaffold evidence only.
- [ ] The main method must be a RGB-D world-model pipeline. State/oracle-slot
      experiments are only scaffolding, upper bounds, or debugging evidence.
      Do not claim method success from state-only world models, CV-from-state
      predictors, or oracle-slot controller videos.
- [ ] 2026-06-06 user correction on dataset regeneration authority:
      the current 6/6 regenerated Cosmos3 full1000 path may continue, but it
      must not become a future default. The corrected 10-video viewpoint
      approval was not an instruction to regenerate a full dataset. If
      downstream results are bad, surprising, or suggest a possible data issue,
      first inspect the concrete logs/artifacts/visuals and report the
      evidence plus options to the user. Do not move, archive, replace, or
      regenerate a full dataset or derived training chain merely because a
      result is poor unless the user explicitly approves that reset.
- [ ] 2026-06-06 user correction on Cosmos/controller interface:
      the current post-SFT `vision.mp4` under
      `action_eval_after_sft_full1000_maniskill_default_regen_20260606_chain`
      is only a 93-frame, 256px, 3.1-second one-shot diagnostic from the
      trajectory start. It is not a valid full rollout world-model interface
      and not controller method evidence. Future controller-facing Cosmos3
      inference must be receding and teacher-forced: condition on the latest
      real observed video prefix and RGB-D-derived robot/object state at or
      after the object starts moving, predict a short future horizon, execute
      one/few controller steps, then refresh with newly observed post-motion
      frames before predicting again. Do not use a stale pre-motion prefix or a
      single precomputed trajectory as the main Cosmos controller method. Also
      inspect and repair the bridge orientation interface: the current
      `bridge_orientation_reference=peg_alignment` run visibly rotates the peg
      after controller takeover, has max peg rotation error about `0.676` rad,
      and all insert-manifold checks false, so it is diagnostic failure
      evidence, not an acceptable controller design.
- [ ] 2026-06-06 receding Cosmos3/controller retest result:
      job/allocation `110599` in tmux
      `cosmos3_receding_ctrl_20260606_132036` ran five
      teacher-forced Cosmos3/readout segments (`26`, `62`, `74`, `86`, `98`)
      and fed the merged trajectory into the RGB-D-slot controller with
      `bridge_orientation_reference=contact_preserve`. It did not regenerate
      full1000 data or retrain SFT. Result is a failure, not method success:
      `success_at_end=false`, `success_once=false`, `handoff_count=0`,
      `bridge_count=54`, `infeasible_count=192`, `retreat_count=192`.
      Manual contact-sheet inspection shows the robot grasps the peg and
      approaches the moved hole but does not insert. The explicit commanded
      rotation issue is reduced (`bridge_delta_rot_norm max=0`,
      `rot_servo_reference=contact_preserve_hold_orientation`), but the
      closed-loop control slots drift badly after trigger: post-trigger
      slot-vs-metric mean errors are hole `0.0548m`, peg `0.2636m`, TCP
      `0.2765m`, peg-head-in-hole `0.2357m`; final control peg-head estimate
      `[-0.2026,-0.0371,0.0002]` disagrees with true metric
      `[-0.2462,0.1185,0.1977]` by `0.2552m`. The controller also believes
      grasp is lost from frame `120` while the true metric slot remains
      grasped through frame `300`. Classification: RGB-D
      perception/readout/controller-interface failure after receding Cosmos
      integration, not DP handoff success and not a reason to relax gates.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-06_receding_teacher_forced_cosmos3_controller_retest.md`.
- [ ] 2026-06-06 user correction on controller purpose/training:
      a takeover controller must preserve the frozen/static DP's grasp-hold
      and insertion competence before it can claim dynamic progress. The
      failed receding Cosmos3 controller video is visually a peg-hold/insert
      failure and must not be used as positive takeover distillation data.
      Updated `AGENTS.md`,
      `PLAN/world_model_task_rebinding/05_rebinding_controller.md`, added
      `PLAN/world_model_task_rebinding/08_wm_policy_distillation.md`, and
      added `TODO/world_model_task_rebinding/07_wm_policy_distillation.md`.
      Added admission scripts
      `scripts/world_model/inspect_wm_policy_distillation_readiness.py` and
      `scripts/world_model/build_wm_policy_distillation_manifest.py`. Current
      failed run has manual visual review artifact
      `experiments/world_model_task_rebinding/rebinding_controller/cosmos_receding_teacher_forced_contact_preserve_20260606_132036/manual_visual_review.json`
      with `positive_takeover_teacher_ok=false`. Distillation readiness
      preflight
      `experiments/world_model_task_rebinding/wm_policy_distillation/preflight_20260606_145907/readiness.json`
      rejects it as positive teacher:
      `metric_or_slot_drift_check_failed`,
      `manual_visual_review_not_positive`, peg-head slot/metric drift mean
      `0.2357m`, grasp disagreement `0.7393`. Manifest
      `experiments/world_model_task_rebinding/wm_policy_distillation/preflight_20260606_145907/manifest/manifest.json`
      has `training_allowed=false` because no positive takeover teacher exists.
      Static DP preservation checks inside held allocation `110599` found a
      one-seed video failure under
      `static_dp_preservation_eval_20260606_150058` and a 10-episode no-video
      baseline success rate `0.7` under
      `static_dp_preservation_eval10_20260606_150410`. Added
      `scripts/world_model/train_wm_conditioned_policy_distillation.py`, a
      DP-preserving diffusion policy that loads compatible official DP
      checkpoint weights and injects world-model trajectory conditioning
      through a zero-initialized adapter so base behavior is preserved at
      initialization. Added
      `scripts/slurm/run_wm_policy_distillation_in_allocation.sh` for held
      allocation training. Dry-run in allocation `110599` wrote
      `experiments/world_model_task_rebinding/wm_policy_distillation/allocation_dryrun_20260606_152154`
      with `base_samples=152291`, `takeover_samples=0`,
      `full_training_allowed=false`; a negative full-training check correctly
      refused to train from the current manifest. Evidence note:
      `docs/world_model_task_rebinding/2026-06-06_wm_policy_distillation_correction.md`.
- [ ] 2026-06-06 DDP-style controller follow-up:
      the aligned controller lesson is world-model-conditioned takeover plus
      frozen-DP policy-prior handoff, not more hand-written hold thresholds.
      In held allocation `112302`, `success_hold_policy=dp` produced one
      visually reviewed oracle/scaffold diagnostic success on seed `702000`
      under
      `experiments/world_model_task_rebinding/rebinding_controller/wm_policy_takeover_success3_teacher_freeze_base_120iter_success_dp_handoff_diag_20260606_continue`
      (`success_at_end=true`, first success step `160`, final
      `[-0.00176, 0.00302, 0.00294]`), with contact sheet and mp4 in
      `visual_review_traj_0/`. A three-seed diagnostic under
      `wm_policy_takeover_success3_teacher_freeze_base_120iter_success_dp_handoff_seed3_diag_20260606_continue`
      reached only `success_at_end=1/3`, `success_once=2/3`. Boundary: this is
      not RGB-D/Cosmos method evidence and not a stable controller; it proves
      there is useful policy-prior signal and that the next step is broader
      positive DP-action execution data or trajectory optimization, followed
      by RGB-D/Cosmos controller evaluation.
- [ ] 2026-06-06 success12/zero-identity controller update:
      expanded DP-action scaffold insertion teachers to
      `teacher_traces_success12.h5` (`12` successes, `737` actions) and sample
      visual-reviewed four contact sheets. Initial success12 frozen-base
      training broke static preservation (`success_at_end=0.2`) because the
      WM adapter changed the DP condition even when `wm_condition=0`. Patched
      `WMConditionedDiffusionPolicy._condition` to subtract
      `adapter(zeros_like(cond))`, enforcing zero-condition identity. The
      fixed checkpoint
      `train_freeze_base_success12_240iter_zero_identity` restored 5-episode
      static preservation to `success_at_end=0.6`, but dynamic oracle-slot
      retest was still unstable:
      `wm_policy_takeover_success12_zero_identity_240iter_success_dp_handoff_seed3_diag_20260606_continue`
      got only `success_at_end=1/3`, and the apparent successful seed
      `702002` did not reproduce when rerun alone with env-state saves.
      Boundary: useful implementation fix, but no stable controller and no
      RGB-D/Cosmos method evidence yet.
- [ ] 2026-06-06 teacher-support gated takeover update:
      patched `evaluate_rebinding_controller.py` with
      `--wm-policy-takeover-require-teacher-support` and
      `--wm-policy-takeover-prepare-policy`. The diagnostic showed the learned
      insertion policy was being queried far outside its teacher support
      (seed `702000` first takeover at
      `[-0.4436,-0.0916,-0.1137]`). Pure bridge preparation never entered
      support and worsened peg orientation; DP-prior MPC preparation reached
      x/head closer to support but failed unless body-yz support was matched
      to the actual teacher H5 distribution. With `body_yz_max=0.065`, learned
      takeover ran for 31 steps but still failed because peg angle left
      support, final `[-0.11663,0.01488,-0.03316]`. Boundary: useful
      failure localization only. Next aligned work is an orientation/body
      alignment preparation teacher or DP-action trajectory optimizer before
      learned insertion, then RGB-D/Cosmos evaluation.
- [ ] 2026-06-07 current DDP-style controller boundary:
      allocation `114496` on `server34` continued the DDP-inspired controller
      work with short-chunk action-prior probes. The useful borrowed idea is
      receding short-horizon action generation with live re-observation, not
      long open-loop replay. New DP-only action-prior scan found seed `702210`
      as a visually reviewed DP-only candidate with final grasp true; seed
      `702237` is now rejected because final grasp is false. Merged
      DP-only/official converted libraries with up to four chunks were tested
      on seed `702000`; strict filtering always selected only
      `dp_bridge_blend`, allow-non-inserting selected a few teacher follower
      steps but worsened final state, and no online candidate reached
      `inserted_any=true`. Fixed action-prior replay is therefore exhausted
      as the controller mechanism. Added
      `scripts/world_model/build_ddp_action_generator_manifest.py` and
      generated
      `experiments/world_model_task_rebinding/wm_policy_distillation/ddp_action_generator_manifest_20260607_0830/manifest.json`
      with 4 accepted action-prior chunks and 1 rejected empty candidate file.
      Continue with a DDP-style short-horizon
      learned or optimized action generator conditioned on Cosmos/RGB-D task
      state plus current robot/object state, refreshing after each short
      chunk. Do not keep tuning fixed replay gates or pure distillation.
- [ ] 2026-06-07 action-generator scheduling follow-up:
      DP-only scan
      `dp_only_dynamic_action_prior_scan_seed702240_12eps_20260607_0835`
      finished inside allocation `114496` with `success_at_end=0.25`
      (`3/12`) and `success_once=0.4166667`; final-success candidate seeds
      are `702240`, `702242`, `702245` and require video rerun before action
      prior extraction. Allocation `114496` ended by its 3-hour walltime, not
      by manual release. New tmux session
      `wm_action_generator_h200_24h_20260607_084948` requested one H200 for
      24h as Slurm job `114529`; current state is pending `Priority`.
      Test-only probes for 3h/6h/12h/24h all forecast the same start
      `2026-06-08T04:43:37` on `server34`, so keep one pending allocation and
      do not add duplicate short jobs unless live queue evidence changes.
- [ ] 2026-06-07 latest action-generator smoke:
      allocation `114529` started on `server34` and should be kept for
      follow-up work. Video reruns accepted DP-only seeds `702240`, `702242`,
      and `702245` as action-prior candidates only; extracted three chunks and
      merged a 7-chunk library
      `dp_official_7chunk_action_prior_library.h5`. New manifest
      `ddp_action_generator_manifest_20260607_0900/manifest.json` has
      7 accepted action-prior chunks and 1 rejected source. Patched
      `train_wm_conditioned_policy_distillation.py` with
      `--allow-action-prior-smoke` gated by `--freeze-base-policy`. The
      1000-iter frozen-backbone smoke checkpoint
      `train_ddp_action_generator_7chunk_freeze_base_1000iter_20260607_0910/checkpoints/final.pt`
      completed. Static preservation 10eps is `success_at_end=0.6`.
      Dynamic seed `702000` still fails: blend `1.0` final
      `[-0.217053,0.192513,0.001095]`, blend `0.35` final
      `[-0.139404,0.048157,-0.023760]`, both `inserted_any=false` with final
      grasp true. Diagnosis: action-generator smoke runs, but its actions
      over-shoot lateral/vertical motion and are not sufficiently
      state-conditioned. Next aligned fix is a state-conditioned short-horizon
      generator or online optimizer over peg/hole/TCP geometry, not more
      fixed replay or pure distillation.
- [ ] 2026-06-07 DDP action-generator MPC follow-up:
      patched `evaluate_rebinding_controller.py` so the trained
      WM-conditioned action generator can be used as a short-horizon candidate
      inside `control_policy=wm_dp_prior_rollout_mpc`, and patched
      `run_cosmos_task_state_controller_in_allocation.sh` to pass the
      checkpoint/candidate/noise controls from held allocation `114529`. On
      seed `702000`, direct learned-generator rollout-MPC
      `wm_dp_prior_rollout_mpc_wmgen_7chunk_seed702000_no_video_20260607_0958`
      still failed (`success_once=false`, `success_at_end=false`, final
      `[-0.170336,0.017695,-0.046041]`, final grasp true). The learned
      generator was selected for 48 executed steps, but no simulated candidate
      inserted. MPPI around DP plus learned-generator candidates
      `wm_dp_prior_rollout_mpc_wmgen_mppi8_seed702000_no_video_20260607_1012`
      also failed (`success_once=false`, final
      `[-0.170446,0.074166,-0.101969]`, final grasp true) even though planner
      summaries contained 15 simulated inserted candidates. This localizes the
      current controller gap to planner/live contact-state mismatch or
      insertion-execution fidelity, not simply absence of an imagined
      inserting candidate. The stricter `execute_chunk=1` MPPI run
      `wm_dp_prior_rollout_mpc_wmgen_mppi4_exec1_seed702000_no_video_20260607_1025`
      was cancelled after writing only `manifest.json` because per-step MPPI
      was too slow; this is a cost/implementation failure, not controller
      evidence. Allocation `114529` remains running on `server34`; only step
      `114529.10` was cancelled. Do not go back to pure distillation or fixed
      replay as the controller answer. Next aligned fix must either unify
      planner/live state restoration for contact-rich chunks or train/use a
      stronger state-conditioned insertion action model with live execution
      validation.
- [ ] 2026-06-07 latest visual peg-hold check:
      9-chunk WM-conditioned action-generator/MPPI controller videos were
      inspected directly under
      `wm_dp_prior_rollout_mpc_wmgen_9chunk_mppi8_seed702000_video_20260607_1155`
      and
      `wm_dp_prior_rollout_mpc_wmgen_9chunk_insertprobe_adaptive_mppi8_seed702000_video_20260607_1255`.
      Both are negative oracle-slot scaffold diagnostics:
      `success_once=false`, `success_at_end=false`, final grasp true. Visual
      conclusion: the peg is picked up and held roughly horizontally; it is
      not an immediate peg-flying/sky-tilt failure. The failure is the
      insertion/contact stage: the peg remains outside, below, or to the side
      of the moved hole. Both runs now have `manual_visual_review.json` with
      `positive_takeover_teacher_ok=false` and
      `method_evidence_allowed=false`. A short-feedback strong-z no-video
      retest in allocation `115187` also failed
      (`[-0.191184,0.033923,-0.048477]`, final grasp true). Two render
      attempts for that no-video H5 on `server14` stalled before producing
      frames and were cancelled as render/scheduling failures while keeping
      allocation `115187` alive. Evidence note:
      `docs/world_model_task_rebinding/2026-06-07_ddp_style_hybrid_controller_probe.md`.
- [ ] 2026-06-07 direct-head action-generator follow-up:
      after the visual peg-hold check, continued inside allocation `115187`
      without releasing the H200. A longer 12000-step frozen-base 9-chunk
      action-generator smoke with action-prior sampling at `55%` completed at
      `train_ddp_action_generator_9chunk_wmcondition_freeze_base_12000iter_prior55_20260607_1345`;
      it remains action-prior smoke with zero positive takeover sources.
      Same-seed static smoke gave `0.3/10`, matching same-seed base DP
      `0.3/10`, but dynamic seed `702000` still failed and was not better
      than the 3000-step baseline. Patched an optional direct action head into
      `train_wm_conditioned_policy_distillation.py`, exposed it in
      `evaluate_rebinding_controller.py` and the allocation wrappers, and
      trained
      `train_ddp_direct_action_head_9chunk_wmcondition_freeze_base_6000iter_prior55_20260607_1415`
      (`direct_action_loss` `0.1654 -> 0.0198`). Direct-head dynamic no-video
      selected the direct head for only `4` steps and still failed
      (`[-0.137025,0.072565,-0.115241]`, final grasp true). Direct-head plus
      insert-probe also failed
      (`[-0.194064,0.024835,-0.068018]`, final grasp true) even though the
      planner selected one predicted inserted insert-probe chunk at step
      `126`. Boundary: action-prior/direct-head training can fit chunks but
      does not solve live contact insertion. Next aligned work should be more
      visually admitted dynamic execution data or a planner/live
      contact-state-unified trajectory optimizer, not more fixed replay or
      longer adapter training.
- [ ] 2026-06-07 allocation-115187 DP-only WM-condition action-prior scan:
      to expand live executable action-prior coverage without releasing the
      held H200, ran no-video `dp_only` seeds `702312..702347` with
      `save_wm_policy_condition=true` and
      `wm_policy_condition_log_bridge_for_dp_only=true` under
      `dp_only_wmcondition_action_prior_scan_seed702312_36eps_20260607_1510`.
      Result: `success_at_end=11/36`, `success_once=14/36`. Only 9 final
      successes also have `final_grasped=true`: `702316`, `702321`,
      `702325`, `702332`, `702335`, `702340`, `702342`, `702344`, and
      `702346`; `702312` and `702314` are rejected for action-prior
      admission because final grasp is false. Video admission is still
      pending: the first video rerun
      `dp_only_wmcondition_action_prior_video_seed702316_20260607_1615`
      stalled after `manifest.json`, and a direct SAPIEN/ManiSkill preflight
      in the same allocation hung at `env.render()` after import/env/reset
      succeeded. Cancelled only steps `115187.13` and `115187.15`; allocation
      `115187` remains alive. Boundary: these 9 seeds are metric-only pending
      candidates, not admitted action-prior chunks and not controller/method
      evidence until a render-capable allocation produces videos/contact
      sheets and manual review passes. To keep the held allocation doing
      aligned no-render work, started detached tmux session
      `wm_dp_scan_115187_702348` as Slurm step `115187.16` for a follow-up
      no-video scan `702348..702395` under
      `dp_only_wmcondition_action_prior_scan_seed702348_48eps_20260607_1625`;
      it is also metric-candidate discovery only, not admission/training.
      Added a guard in `extract_dp_action_prior_chunks.py`: DP-only
      action-prior extraction now requires manual review with
      `candidate_dp_action_teacher_ok=true` by default, so metric-only pending
      candidates cannot be accidentally extracted into a training manifest.
      Also patched `build_ddp_action_generator_manifest.py` to reject
      DP-only chunks without admitted review and reject unreviewed smoke
      chunks; a temp rebuild of existing 9-chunk inputs still accepted 9 and
      rejected the known bad source.
      Submitted one bounded visual-review canary job `115449`
      (`wm_pending_video_review`, 1 GPU, 2.5h, `--exclude=server14`) for the
      9 pending final-grasp metric candidates; it started on `server40`, but
      the first seed `702316` again stalled after writing only `manifest.json`
      with low CPU and no metrics/video, so `115449` was cancelled to avoid
      wasting the GPU. This is a render/scheduling failure, not video
      admission and not controller evidence.
- [ ] 2026-06-07 allocation-115483 render diagnosis and continuation:
      allocation `115187` later became `CANCELLED` at
      `2026-06-07T17:01:33`; its follow-up scan step `115187.16` wrote only
      `manifest.json`, so it is not a completed scan. Immediately requested a
      new held one-H200 allocation in tmux
      `wm_controller_h200_hold_20260607_1710`; Slurm job `115483` is running
      on `server58` with a 24h time limit and should not be released after
      short probes. Patched controller/replay rendering to use the same named
      slanted overhead `render_camera` as the approved Cosmos3 videos and
      added `scripts/world_model/render_min_canary.py`. On `server58`, a
      multi-frame env-state render probe timed out, and the minimal 256x256
      canary showed SAPIEN device summary OK, `gym_make_done`, `reset_done`,
      then hung exactly at `render_rgb_array_start` until timeout. Therefore
      this allocation can be used for no-video rollout/training/debugging but
      not for human visual rendering. Existing 11:55 and 12:55 contact sheets
      were reopened: peg pickup/hold is roughly normal and not a fly-to-sky
      failure; the negative controller failure is near-hole/contact insertion.
      Completed no-video DP-only metric-candidate scan
      `dp_only_wmcondition_action_prior_scan_seed702365_48eps_20260607_1717`
      inside `115483`: `success_at_end=11/48`, `success_once=14/48`.
      Eight final-success episodes retained final grasp and are pending
      visual review only (`702369`, `702371`, `702375`, `702383`, `702388`,
      `702390`, `702395`, `702409`); three final-success episodes are
      rejected for action-prior admission because final grasp is false
      (`702367`, `702392`, `702398`). Submitted bounded visual-admission
      render job `115714` for the eight pending candidates, with job-local
      exclusion of current render-failed nodes `server14,server40,server58`;
      it started on `server27` and completed successfully (`COMPLETED`,
      exit `0:0`, elapsed `00:03:18`), writing contact sheets/videos under
      `dp_only_wmcondition_action_prior_scan_seed702365_48eps_20260607_1717_visual_review/`.
      All eight rendered contact sheets were inspected and accepted as
      DP-only action-prior candidates only; none show the peg being picked up
      and immediately tilted upward or thrown away. Manual review:
      `dp_only_wmcondition_action_prior_scan_seed702365_48eps_20260607_1717_visual_review/manual_visual_review.json`
      (`candidate_dp_action_teacher_ok=true`,
      `positive_takeover_teacher_ok=false`). Also patched
      `evaluate_rebinding_controller.py` to record
      `planner_restore_report` for each rollout-MPC candidate and
      `selected_planner_restore_report` for the chosen plan. This does not
      alter scoring or success; it records whether
      `reset + set_state_dict(start_state)` reconstructed the live
      hole/peg/TCP/qpos/qvel/grasp/inserted state before imagined rollout
      scoring. Use it to diagnose planner/live contact-state mismatch instead
      of treating simulated `inserted_any` as live execution evidence.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-07_ddp_style_hybrid_controller_probe.md`.
- [ ] 2026-06-07 latest held-allocation action-head check:
      still inside allocation `115483`, extracted the eight visually admitted
      DP-only candidates and rebuilt
      `experiments/world_model_task_rebinding/wm_policy_distillation/ddp_action_generator_manifest_20260607_1915/manifest.json`
      with `17` accepted action-prior chunks. The frozen-base direct-action
      head smoke
      `train_ddp_direct_action_head_17chunk_wmcondition_freeze_base_24000iter_prior55_20260607_1925`
      completed all `24000` iterations and wrote `checkpoints/final.pt`.
      Final logged step `23999`: total loss `0.017129648476839066`, direct
      action loss `0.009156551212072372`. Immediate no-video static
      preservation 10eps:
      `static_preservation_final_17chunk_24000iter_10eps_20260607_2000`
      was `success_at_end=0.7`, `success_once=0.7`. Dynamic seed-`702000`
      no-video rollout-MPC with direct-head candidates:
      `wm_dp_prior_rollout_mpc_direct_head_17chunk_24000iter_seed702000_no_video_20260607_2000`
      failed (`success_at_end=false`, `success_once=false`). Final grasp was
      true, min lateral/vertical peg-head error reached `0.0012995906872674823`
      m, but `inserted_any=false`, `inserted_frames=0`, and max insertion-axis
      x was only `-0.11124828457832336` m. Current conclusion is unchanged:
      the failure is contact-rich insertion-axis execution and retention, not
      peg pickup flying upward. Boundary: this remains negative oracle/Cosmos
      scaffold evidence, not RGB-D method evidence and not a visual result for
      this exact no-video run. Exact-run saved-state render job `115924` was
      submitted on `server60` and canceled after `2:33` because it hung at
      `render_start` for frame `0` with zero PNG frames written. This is
      rendering failure evidence only and does not change the controller
      metrics.
- [ ] 2026-06-07 DDP-style hybrid controller update:
      the controller work has now borrowed the concrete Dream Diffusion Policy
      pattern instead of only waiting on distillation: imagined/action-prior
      rollout scoring, failure-state teacher search, and learned-approach plus
      physical insertion-chunk switching were implemented and tested. Added
      `control_policy=wm_dp_prior_rollout_mpc`, but on seed `702000` it still
      failed (`success_once=false`, `success_at_end=false`, max insertion x
      `-0.111066`). Searched from failed distillation near-hole states and
      found 6 successful DP-action insertion chunks
      (`teacher_search_from_distill80_failure_seed702000_20260607.h5`,
      `299` actions, final grasp true). Training with those chunks preserved
      static smoke (`success_at_end=0.6`) but did not make pure learned
      takeover succeed. The best hybrid diagnostic
      `wm_policy_takeover_hybrid_teacher_support_bridge_latch70_close006_successholdfix_insert6_160iter_seed702000_diag_20260607_continue`
      reached transient success (`success_once=true`, success frames
      `267..285`) but failed final success because grasp was already false and
      DP/zero/servo holds all let the peg drift out. Boundary: this is
      oracle/scaffold evidence only, not RGB-D/Cosmos method evidence. Current
      bottleneck is grasp-preserving insertion/retention. Action comparison
      shows the online bridge latch is not reproducing the successful teacher
      primitive: teacher first-12 mean xyz action was
      `[0.0224,0.0805,0.0390]`, while online latch was
      `[-0.0231,0.5404,-0.0933]` with the same `-1.0` gripper command. The
      next fix is to align the online insertion action generator with the
      teacher action distribution or replay a teacher-style local insertion
      chunk, not more distillation or post-success hold tuning. Follow-up
      teacher-action replay was implemented, but raw action replay and
      peg-head local-delta replay still failed. A new exact-failure-state
      teacher search found 4 successful chunks from the same support states
      (`teacher_search_from_local_delta_replay_failure_seed702000_20260607.h5`,
      final grasp true), yet replaying those raw actions online also failed
      (`tcp_continuation` selected run: `success_once=false`,
      `success_at_end=false`, final grasp false). TCP-local-delta replay from
      the same teacher preserved grasp but also failed insertion
      (`success_once=false`, final `[-0.145975,0.035818,-0.043910]`).
      Conclusion: raw action replay and TCP-only local-delta replay are not
      robust enough; the next aligned fix is a multi-state closed-loop
      teacher-trajectory follower or online trajectory optimizer over TCP, peg
      body/head, and orientation. Follow-up state-follower replay preserved
      final grasp but still failed insertion
      (`wm_policy_takeover_teacher_state_follower_tcp_continuation_exact_failure_seed702000_diag_20260607_continue`,
      `success_once=false`, final
      `[-0.119097,0.037095,0.010557]`, max insertion x `-0.108954`).
      Rollout-MPC was then extended to score teacher action/local-delta/
      TCP-delta/state-follower candidates in the planner env. Chunk-4 selected
      teacher candidates for 32/210 steps but still failed
      (`final [-0.119670,0.013228,0.010310]`, max x `-0.111457`);
      chunk-16 selected only DP/bridge candidates and also failed
      (`final [-0.113406,0.018828,-0.012211]`, max x `-0.110885`).
      Patched an optional MPPI-style sampled action-sequence candidate path
      around teacher chunks; MPPI selected sampled actions for 50/210 steps
      but still failed (`final [-0.123545,0.025668,-0.116508]`, max x
      `-0.111267`). Added online bridge/local-axis variant candidates from
      `search_dp_insertion_teacher.py`; chunk32 selected variants for most
      steps but failed and no online planner candidate had `inserted_any`.
      Offline search from that exact failed H5 found a recoverable state at
      frame 219: `bridge_phase_hybrid_peg_alignment` succeeded in 47 steps
      with final grasp true. Chunk64 online planning did not reach that same
      state and still failed; fixed recovered-teacher replay triggered too
      early and failed. Conclusion: the next controller change is online
      primitive search/optimization from the actual current state followed by
      immediate execution, not pure distillation or fixed teacher-H5 replay.
      Follow-up two-stage online primitive search fixed several implementation
      issues: variants are support-gated, primitive horizon is no longer
      capped by the DP 8-step action horizon, future dynamic hole perturbation
      is applied in planner rollouts, and simulated insertion is checked after
      the next perturbation. The planner can now select a simulated inserting
      chunk, but the real rollout still does not reproduce it even though the
      selected planned actions and real executed actions match exactly
      (`diff_max=0.0`). Offline search from the same failed H5 still finds
      successful frame-135/136 primitives with final grasp true. Current
      localized bottleneck: planner-env/state-restoration/timing fidelity.
      Added planned-chunk replay verifier
      `scripts/world_model/verify_planned_chunk_replay.py`. On the step-130
      selected chunk from
      `wm_dp_prior_rollout_mpc_two_stage_future_perturb_postcheck_seed702000_diag_20260607_continue`,
      verifier replay also failed despite `action_diff_max=0.0`;
      planner predicted final
      `[-0.013769,-0.001972,-0.002892]`, replay ended at
      `[-0.105999,-0.038823,-0.027090]`, and
      `planner_after_vs_replay_head_rmse=0.041394m`. The immediate next fix is
      replay-verified online primitive search or planner/replay dynamics
      unification before executing a chunk.
      Follow-up Dream Diffusion Policy paper check: the relevant lesson is
      short action chunks with repeated real-imagination alignment, not long
      open-loop video or more pure distillation. Patched
      `evaluate_rebinding_controller.py` with
      `--wm-dp-prior-rollout-execute-chunk` default `8`, so rollout-MPC may
      score a longer imagined horizon but executes only a short prefix before
      re-observing live state. The long-open conservative run still failed
      (`final [-0.086633,-0.001576,-0.000618]`, max x `-0.046175`,
      final grasp true). Step-210 reset/replay succeeded with the same actions
      (`replay_success_once=true`, final
      `[-0.014533,0.001570,0.002781]`) while the original live window failed
      (`[-0.046175,-0.003229,0.000712]`), proving restored planner contact
      state is not live contact state. Receding-8 also failed
      (`final [-0.113631,0.002675,-0.001677]`, max x `-0.048891`,
      final grasp true), and allowing non-inserting local-axis fallback was
      worse (`final [-0.203388,0.037057,-0.066889]`, max x `-0.111281`).
      Conclusion: current primitive/planner family is exhausted for this
      diagnostic. Next aligned controller work is a DDP-style short-chunk
      action generator from DP/expert/Cosmos-conditioned behavior with live
      re-observation, not more primitive-gate tuning or standalone
      distillation. Added a guided-DP residual candidate family as a final
      light-weight DDP-style fallback: it preserves the frozen DP action chunk
      and mixes in only a small task-frame residual. The isolated run
      `wm_dp_prior_rollout_mpc_receding8_guided_dp_residual_seed702000_diag_20260607_continue`
      also failed (`success_once=false`, final
      `[-0.123001,0.012271,-0.045794]`, max x `-0.110716`, final grasp true);
      all 12 guided-DP candidates were non-inserting and rejected, and the
      selected actions stayed `dp_bridge_blend`. This rules out simple
      residual gain tuning. The next source must be live-executable action
      chunks from admitted positive takeover data, official/expert DP-contract
      conversion verified live, or a stronger Cosmos-conditioned action model.
      Visual follow-up at
      `wm_dp_prior_rollout_mpc_receding8_guided_dp_residual_seed702000_video_20260607_continue`
      saved 30fps video/contact sheet and manual review; last frame visibly
      leaves the peg outside the hole, matching the failed metrics, and
      `manual_visual_review.json` marks
      `positive_takeover_teacher_ok=false`.
      Follow-up DP-only live action-prior scan
      `dp_only_dynamic_teacher_scan_seed702200_10eps_20260607_continue`
      found sparse frozen-DP dynamic successes: `success_at_end=0.2`,
      `success_once=0.4` over seeds `702200..702209`, with policy contract
      fields saved (`policy_obs`, `wm_policy_condition`,
      `controller_env_states`). No-video final-success candidates were
      `702204` and `702207`; video rerun confirmed `702204` visually holds and
      inserts the peg, while `702207` failed to reproduce and is rejected.
      `702204` is marked `candidate_dp_action_teacher_ok=true` but
      `positive_takeover_teacher_ok=false` because it is `dp_only`, not a
      world-model-controller takeover. It may be used to study/live-verify
      action-prior chunks, but not silently as positive takeover data. Added
      `scripts/world_model/extract_dp_action_prior_chunks.py` and extracted
      the visually approved `702204` run to
      `experiments/world_model_task_rebinding/wm_policy_distillation/dp_live_action_prior_chunks_20260607/dp_only_seed702204_action_prior_chunks.h5`.
      The chunk is `dp_only_live_action_prior`, source frames `108..300`,
      `192` actions, `policy_obs (193,43)`, `wm_policy_condition (193,64)`,
      and separate `metric_slots`; it is compatible with
      `TeacherActionChunkLibrary` while remaining explicitly
      `dp_action_prior_candidate_not_wm_takeover`. Next aligned step is to
      test this H5 as an explicit action-prior/chunk library in a
      WM-conditioned controller, not as positive takeover distillation data.
      Added explicit controller args `--wm-policy-action-prior-h5` and
      `--wm-policy-action-prior-variant` so this source can be loaded as
      `action_source_role=dp_or_expert_action_prior` instead of through the
      ambiguous teacher-action path. Allocation `114164` naturally expired at
      its 3h walltime after the last checks; it was not manually released.
      Evidence:
      `docs/world_model_task_rebinding/2026-06-07_ddp_style_hybrid_controller_probe.md`.
- [ ] 2026-06-06 held allocation continuation:
      previous allocation `112302` naturally expired at its 3h walltime; it was
      not manually released. Started replacement tmux-backed one-GPU allocation
      request `113466` in session
      `wm_policy_h200_3h_teacher_support_20260606_234658`
      (`partition=gpu`, `--gres=gpu:1`, `--cpus-per-task=12`,
      `--time=03:00:00`) to continue the teacher-support/controller work.
- [ ] 2026-06-06 scaffold teacher admission boundary:
      audited archived controller video `95107`
      (`peg_drop_rebind_cv_never_job95107`). The video/review sheet and last
      frame were opened directly and are visually consistent with the metrics:
      `success_at_end=true`, first success step `234`, final
      `peg_head_at_hole=[-0.013857, 0.002249, -0.002390]`, `regrasp_count=13`.
      Boundary: this is old state/CV/oracle scaffold evidence only, not
      RGB-D/Cosmos method evidence, because it uses `hole_predictor=cv` and
      the H5 lacks separate `metric_slots`. Added
      `manual_visual_review.json` under the archived run with
      `positive_takeover_teacher_ok=false`, `scaffold_teacher_ok=true`, and
      `method_evidence_allowed=false`. Patched
      `inspect_wm_policy_distillation_readiness.py` to support explicit
      `--allow-scaffold-only-sources` and patched
      `build_wm_policy_distillation_manifest.py` to keep these in
      `scaffold_takeover_sources`, never `positive_takeover_sources`. The
      scaffold audit manifest under
      `experiments/world_model_task_rebinding/wm_policy_distillation/scaffold_audit_20260606_1605/manifest/manifest.json`
      has `positive_takeover_sources=0`, `scaffold_takeover_sources=1`,
      `world_model_condition_sources=1`, and `training_allowed=false`.
      Dry-run after adding a local EMA fallback loaded base DP with
      `base_samples=152291`, `takeover_samples=0`,
      `full_training_allowed=false`; non-dry-run still refuses to train with
      rc `1`. A longer static DP preservation baseline was then run inside
      held allocation `110599`, not via sbatch:
      `experiments/world_model_task_rebinding/wm_policy_distillation/static_dp_preservation_eval50_20260606_1612`
      with `success_at_end=0.58`, `success_once=0.58`, `29/50` final
      successes. This is a base-skill preservation baseline, not dynamic
      method evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-06_scaffold_teacher_admission_boundary.md`.
- [ ] 2026-06-06 controller policy-observation contract patch:
      future controller rollouts intended for DP-preserving policy
      distillation must save the real policy input contract. Patched
      `scripts/world_model/evaluate_rebinding_controller.py` with
      `--save-policy-observations` and `--save-wm-policy-condition`, writing
      raw current `policy_obs` `(T+1,43)`, full
      `policy_obs_frame_stack` for audit, and causal 64D
      `wm_policy_condition` built from current control slots/bridge-plan
      fields before each action. Patched the Cosmos allocation controller
      wrapper and watcher to default these saves on for future runs. Patched
      `inspect_wm_policy_distillation_readiness.py` so a candidate positive
      takeover run is rejected if it lacks this contract. Rechecked the failed
      receding Cosmos controller and archived `95107`: both still have
      `positive_controller_source_count=0`; `95107` remains scaffold-only.
      New reports:
      `experiments/world_model_task_rebinding/wm_policy_distillation/contract_preflight_20260606_1650/readiness_cosmos_failed.json`
      and
      `experiments/world_model_task_rebinding/wm_policy_distillation/contract_preflight_20260606_1650/readiness_scaffold_95107.json`.
      Contract manifest
      `experiments/world_model_task_rebinding/wm_policy_distillation/contract_preflight_20260606_1650/manifest/manifest.json`
      still has `training_allowed=false` with
      `positive_takeover_sources=0`, `scaffold_takeover_sources=1`,
      and `world_model_condition_sources=1`. Dry-run reported
      `base_samples=152291`, `takeover_samples=0`; non-dry-run refused full
      training with rc `1`.
- [ ] 2026-06-06 mixed objective implementation:
      patched `scripts/world_model/train_wm_conditioned_policy_distillation.py`
      so the corrected controller policy objective is no longer just action
      diffusion BC. Future admitted positive takeover H5s now train
      `L_base_dp_bc + L_takeover_bc + L_grasp_hold +
      L_task_frame_progress + L_switch`. Auxiliary labels are derived from
      admitted rollout supervision only: grasp retention and task-frame
      progress from `metric_slots`, switch class from `event_log_json`, while
      policy inputs remain `policy_obs` plus causal `wm_policy_condition`.
      Patched `inspect_wm_policy_distillation_readiness.py` to reject positive
      takeover candidates that lack metric label fields or event logs, and
      patched `scripts/slurm/run_wm_policy_distillation_in_allocation.sh` to
      record/pass the auxiliary loss weights. Regression reports under
      `experiments/world_model_task_rebinding/wm_policy_distillation/mixed_objective_preflight_20260606_1715`
      still have `positive_takeover_sources=0`,
      `scaffold_takeover_sources=1`, `training_allowed=false`; dry-run
      reported `base_samples=152291`, `takeover_samples=0`, and non-dry-run
      refused full training with rc `1`. A temporary synthetic positive-H5
      smoke test verified label extraction and finite mixed-loss forward.
- [ ] 2026-06-06 one-shot Cosmos controller guard:
      patched `scripts/world_model/evaluate_rebinding_controller.py` with
      `--cosmos-task-state-require-receding`. When this is set, a single
      93-frame one-shot `cosmos_action_eval_controller_trajectory.json` is
      rejected before controller rollout; only a payload with teacher-forced
      receding `segments` is accepted. Patched
      `scripts/slurm/run_cosmos_task_state_controller_in_allocation.sh` to
      default `COSMOS_TASK_STATE_REQUIRE_RECEDING=true`; patched
      `scripts/slurm/run_cosmos3_receding_teacher_forced_controller_retest.sbatch`
      to pass it explicitly. Patched
      `scripts/slurm/watch_cosmos3_action_eval_readout_controller.sh` so the
      old 3-second action-eval/readout path now skips controller launch by
      default and writes
      `controller_skipped_one_shot_not_closed_loop.txt`; running that old path
      as a diagnostic now requires
      `ALLOW_ONE_SHOT_COSMOS_CONTROLLER_DIAGNOSTIC=true`, which explicitly
      disables the receding requirement. Smoke check: the current one-shot
      trajectory is rejected under `require_receding=True`; the merged
      receding retest JSON loads with `5` segments and first prefix boundary
      `54`.
- [ ] 2026-06-06 WM-policy static preservation evaluator:
      added `scripts/world_model/evaluate_wm_policy_static_preservation.py`
      and
      `scripts/slurm/run_wm_policy_static_preservation_in_allocation.sh`.
      This evaluates a WM-conditioned policy checkpoint, or a base-DP-
      initialized WM policy, on the original static ManiSkill task with zero
      or explicitly provided WM condition. It is a required preservation check
      before dynamic claims, not dynamic method evidence. Compile/shell checks
      passed, CLI help works, and a one-episode no-video smoke ran inside
      allocation `110599` on `server13`:
      `experiments/world_model_task_rebinding/wm_policy_distillation/static_wm_policy_preservation_smoke_20260606_1735/metrics.json`.
      The smoke used the official DP checkpoint to initialize the WM policy,
      loaded `148` compatible keys, used zero WM condition, and got
      `success_at_end=1.0`, `success_once=1.0` for one episode. This only
      proves the evaluator path runs and can preserve behavior on that seed;
      it is not a statistical preservation claim.
      Follow-up 50-episode no-video baseline:
      `experiments/world_model_task_rebinding/wm_policy_distillation/static_wm_policy_preservation_eval50_20260606_1750/metrics.json`
      ran inside the same allocation `110599` on `server13`, loaded the same
      `148` compatible official-DP keys with zero WM condition, and produced
      `success_at_end=0.54`, `success_once=0.54` (`27/50`). The earlier
      official-DP 50-episode baseline was `0.58` (`29/50`), so this is a
      static preservation reference for the WM-policy wrapper, not dynamic
      method evidence.
- [ ] 2026-06-06 WM-policy dynamic execution interface:
      patched `scripts/world_model/evaluate_rebinding_controller.py` with
      `control_policy=wm_policy_takeover`. This is the corrected controller
      execution path requested by the user: frozen/static DP handles the
      initial base skill until the configured takeover condition, then a
      trained `WMConditionedDiffusionPolicy` checkpoint executes actions from
      the current DP observation plus the causal 64D
      `wm_policy_condition`. The existing bridge planner is used only to build
      task-frame/world-model conditioning and diagnostics; it is not the
      final action executor in this mode. The path refuses to run without
      `--wm-policy-checkpoint`, rejects CV-only predictors, and by default
      requires a receding teacher-forced or live-refresh world-model
      interface. Patched
      `scripts/slurm/run_cosmos_task_state_controller_in_allocation.sh` so
      the held tmux allocation can launch it with
      `CONTROL_POLICY=wm_policy_takeover` and `WM_POLICY_CHECKPOINT=...`.
      Light checks passed: `py_compile`, `bash -n`, and CLI help exposing the
      new `--wm-policy-*` flags. Boundary: this only connects the correct
      dynamic execution interface. It is not dynamic method evidence until a
      policy is trained from admitted positive takeover data, static
      preservation passes, and a RGB-D/Cosmos dynamic rollout has metrics plus
      inspected video showing the peg remains held and inserted.
      Resource status: allocation `110599` ended as `CANCELLED by 0` at
      `2026-06-06T17:26:36+08:00` after `04:05:42`; it was not intentionally
      released by the agent. A replacement 12h one-card H200 `salloc`
      (`111260`) was pending only and was replaced by shorter 3h one-card H200
      tmux allocation request `111279` in session
      `wm_policy_h200_3h_20260606_173017`. Current reason:
      `Nodes required for job are DOWN, DRAINED or reserved for jobs in higher
      priority partitions`; no start forecast yet.
      Follow-up: added
      `scripts/world_model/audit_wm_policy_takeover_ready.py` and evidence
      note
      `docs/world_model_task_rebinding/2026-06-06_wm_policy_takeover_preflight_and_smoke.md`.
      Strict preflight on current artifacts remains not ready because there is
      no admitted positive takeover teacher and no full trained checkpoint.
      Allocation `111279` later started on `server13`; live preflight showed
      `nvidia-smi` GPU handle failure, torch `cuda_available=False`, and
      SAPIEN/Vulkan `ErrorIncompatibleDriver`, so it was canceled as a
      scheduling/rendering failure and replaced with targeted current-node
      exclusion. Replacement allocation `111343` started on `server31`, passed
      GPU preflight, and ran a one-episode no-video base-only
      `wm_policy_takeover` smoke. Result: `success_at_end=false`,
      `wm_policy_takeover_count=210`, `bridge_count=0`, H5 contract present
      (`policy_obs (301,43)`, `policy_obs_frame_stack (301,2,43)`,
      `wm_policy_condition (301,64)`, `event_log_json`). Readiness rejects it
      with `positive_controller_source_count=0`; this proves execution-path
      plumbing only, not method success. A 5-episode static smoke for the same
      base-only checkpoint under `111343` gave `success_at_end=0.6`.
      DP-only teacher acquisition scan under `111343` did not produce admitted
      positive teacher data. No-video seeds `702000..702004` gave
      `success_at_end=0.2`, but the only final-success seed had no video and
      did not reproduce when rerun with env states. A second env-state scan
      on seeds `702100..702104` gave `success_at_end=0.0`, `success_once=0.0`
      despite saving `policy_obs`, `wm_policy_condition`, and
      `controller_env_states`. Readiness report
      `experiments/world_model_task_rebinding/wm_policy_distillation/dp_dynamic_teacher_scan_envstates_20260606_1813_readiness.json`
      has `positive_controller_source_count=0`. Evidence:
      `docs/world_model_task_rebinding/2026-06-06_dp_dynamic_teacher_scan.md`.
      Follow-up DDP-style controller probe: added
      `control_policy=wm_dp_prior_mpc`, which uses Cosmos/world-model
      task-frame prediction as the imagined short-horizon trajectory while
      selecting candidates around the frozen DP action prior. This directly
      addresses the Dream Diffusion Policy controller idea, but current
      results are negative diagnostics only because they use `slot_source=oracle`
      and no inspected video. Under allocation `111343`, conservative
      DP-prior MPC reached a minimum yz error `0.0165m` but ended failed at
      peg-head `[-0.11067,-0.05019,-0.01131]`; aggressive blending collapsed
      to geometry and worsened; early handoff made 18 DP handoff steps but
      still ended failed at `[-0.11068,-0.03711,-0.01196]`. Conclusion:
      minimal action-prior MPC is insufficient; next controller work should
      use a stronger expert/trajectory or policy-learning execution layer, not
      more threshold-only bridge tuning. Follow-up `wm_dp_prior_sequence_mpc`
      was added as a stronger DDP-style short action-chunk planner around the
      DP prior. One no-video oracle-slot smoke under `111343` also failed
      final success but localized the failure: final grasp stayed true and yz
      reached `0.00011m`, yet insertion x only reached `-0.09875m`. A
      phase-scoring fix was worse (`final peg_head_at_hole=
      [-0.19478,0.01128,-0.01956]`) and was reverted. Current conclusion:
      the controller is not primarily blocked on lateral alignment; it needs
      reliable contact/insertion-axis execution from a stronger teacher,
      trajectory optimizer, or learned policy.
      Follow-up inside held allocation `112302` checked the actual Dream
      Diffusion Policy boundary and localized insertion failure further. DDP
      itself uses shared policy/world-model perception latents plus
      inference-time imagined latent rollouts; the current repo only has the
      weaker fallback "Cosmos/readout task trajectory + frozen DP action prior"
      and must not present it as full DDP. A sequence-MPC rerun with
      controller env states reproduced failure at final peg-head
      `[-0.10677,-0.00176,-0.00147]` while preserving grasp and achieving
      head yz `0.00229m`. Replay from the aligned state showed pure bridge,
      action blends, local insertion-axis sign probes, z offsets, and bounded
      orientation correction all failed insertion. Comparing official planner
      successes exposed a real controller bug: head yz alone was aligned, but
      peg center/body yz was `0.0124m`, whereas successful planner
      pre-insertion body yz is about `0.0003..0.005m`. Patched
      `evaluate_rebinding_controller.py` so `insert_ready` and handoff require
      peg head yz, peg center yz, and peg angle; added `peg_body` reference and
      bounded pre-insert peg-angle correction under `contact_preserve`. The
      body/angle-aware diagnostic improved final head yz to `0.00137m`, body
      yz to `0.00579m`, angle to `0.036rad`, and x to `-0.1033`, but still
      failed. Strict DP handoff made 49 handoff steps and ended at x
      `-0.1020`; always tracking `peg_body` worsened head centering. Current
      controller conclusion: task-frame preparation is partially solved, but
      the active execution layer still cannot generate the physical insertion
      push. Next work must use a stronger positive insertion execution source
      in the DP action contract or an explicit trajectory optimizer; do not
      spend more mainline effort on scaffold-only distillation or
      threshold-only bridge tuning. Added official motion-planner
      teacher-source probe
      `scripts/world_model/evaluate_motion_planner_moved_hole.py`. Seed
      `702000` static/moved-hole failed, but broader static seeds `0..4`
      succeeded at `0.4` and moved-hole seeds `0..4` succeeded at `0.2`.
      Success-filtered moved-hole collection gathered three official-planner
      expert traces from seeds `2`, `6`, and `9` under
      `experiments/world_model_task_rebinding/motion_planner_moved_hole/moved_hole_collect_success3_seed0_max30_20260606_continue/expert_traces.h5`.
      Boundary: these are simulator-state `pd_joint_pos` expert trajectories
      with action dim `8`, not direct positive DP-action takeover BC for the
      current `pd_ee_delta_pose` action dim `7` policy. They require
      trajectory-level training or action-contract conversion before use.
      Added action-contract converter
      `scripts/world_model/convert_motion_planner_expert_to_dp_takeover.py`.
      It restores official-planner env states inside the frozen DP
      `pd_ee_delta_pose` environment and writes scaffold takeover H5 with
      `policy_obs`, 7D converted actions, `wm_policy_condition`, and
      `metric_slots`. Converted output:
      `experiments/world_model_task_rebinding/wm_policy_distillation/motion_planner_converted_dp_takeover_20260606_continue/converted_takeover.h5`
      has three trajectories and dry-run loads through
      `train_wm_conditioned_policy_distillation.py` with
      `base_samples=152291`, `takeover_samples=532`,
      `full_training_allowed=false`. Also fixed a real mixed-training bug:
      the distillation loader now uses `WeightedRandomSampler` when takeover
      data exist, so `--base-sample-fraction` actually controls base/takeover
      sampling. A 3-step scaffold smoke with `base_sample_fraction=0.5` saw
      takeover batch fractions `0.5`, `0.75`, `0.5` with finite auxiliary
      losses. Added a source-level guard so scaffold/non-method-evidence
      sources placed in `positive_takeover_sources` are rejected unless the run
      explicitly passes `--allow-scaffold-takeover-smoke`, and full training is
      still refused with scaffold sources. Negative guard check refused the
      converted scaffold manifest without override; override dry-run loaded
      `base_samples=152291`, `takeover_samples=532`, and recorded
      `scaffold_positive_source_count=1`. This is code-path/scaffold evidence
      only, not RGB-D method success. Follow-up scaffold mixed distillation:
      full-parameter `train_scaffold_mixed_800iter_20260606_continue` fit the
      scaffold loss but destroyed static DP preservation (`success_at_end=0.1`
      over ten static episodes), so unrestricted mixed fine-tuning is not an
      acceptable controller training default. Patched
      `train_wm_conditioned_policy_distillation.py` with
      `--freeze-base-policy` and the allocation wrapper with
      `FREEZE_BASE_POLICY`; frozen-base
      `train_scaffold_mixed_freeze_base_400iter_20260606_continue` preserved
      static DP on a 5-episode smoke (`success_at_end=1.0`) but is still
      blocked from method eval by scaffold provenance. Dynamic oracle-slot
      diagnostic
      `wm_policy_takeover_scaffold_freeze_base_400iter_diag_20260606_continue`
      failed badly (`success_at_end=false`, final peg-head-at-hole
      `[-0.76907,-0.03461,-0.11303]`, min yz `0.11428m`). Current conclusion:
      freezing the DP backbone is necessary for base-skill preservation, but
      current converted scaffold planner data do not teach a usable dynamic
      WM-policy takeover. Follow-up analysis found a concrete condition
      contract bug in the old converted H5: it trained on
      `desired_rel=future_peg_head_at_hole` and a non-takeover mode one-hot,
      while inference uses controller-goal preinsert/insert commands in
      `wm_policy_takeover` mode. Patched
      `convert_motion_planner_expert_to_dp_takeover.py` with default
      `--condition-mode controller_goal`; generated corrected scaffold H5
      `converted_takeover_controller_goal_full3.h5` and manifest
      `scaffold_smoke_manifest_controller_goal.json`. Dry-run loads
      `base_samples=152291`, `takeover_samples=532`, still
      `full_training_allowed=false`.
      Evidence:
      `docs/world_model_task_rebinding/2026-06-06_ddp_style_controller_and_motion_planner_probe.md`.
- [ ] 2026-06-06 receding refresh segment planner:
      added `scripts/world_model/plan_cosmos3_receding_refresh_segments.py`
      and patched
      `scripts/slurm/run_cosmos3_receding_teacher_forced_controller_retest.sbatch`
      so `SEGMENT_STARTS=auto` derives refresh segments from the selected
      sample H5 trigger timing rather than hard-coded starts. The planner
      writes a pre-motion context prefix, a first-observed-motion prefix, and
      configurable post-trigger refresh prefixes with explicit
      `prefix_boundary_frame` / `valid_prediction_start_frame` causal
      metadata. Smoke output:
      `experiments/world_model_task_rebinding/cosmos3/receding_refresh_plan_smoke_20260606_1700/refresh_segment_plan.json`
      for `hole_constant_seed702000_n167` found `trigger_step=80` and planned
      segment starts `16 52 64 76 88 100 112` with prefix boundaries
      `44,80,92,104,116,128,140`. This is still offline teacher-forced
      diagnostic planning, not the final live online refresh loop.
- [ ] 2026-06-06 visual/provenance admission tooling:
      patched `scripts/world_model/evaluate_rebinding_controller.py` so new
      controller manifests and episode summaries record a top-level
      `world_model_interface_report` / `world_model_interface_mode`
      distinguishing `one_shot_task_state_trajectory`,
      `receding_teacher_forced_segments`, and future `live_online_refresh`.
      Patched `scripts/world_model/inspect_rebinding_controller_run.py` to
      surface this interface report. Added
      `scripts/world_model/write_takeover_visual_review.py` to write required
      manual visual review records; positive review requires visible dynamic
      event, held peg through takeover, visible insertion, task behavior ok,
      and non-scaffold provenance. Patched
      `scripts/world_model/inspect_wm_policy_distillation_readiness.py` so
      positive takeover training sources are rejected if the WM interface is
      not receding or live-refresh. Smoke checks:
      `experiments/world_model_task_rebinding/wm_policy_distillation/visual_review_helper_smoke_20260606_1715/manual_visual_review_failed_cosmos.json`
      records the failed receding Cosmos video as visual-negative, and
      `experiments/world_model_task_rebinding/wm_policy_distillation/interface_visual_preflight_20260606_1715/readiness_cosmos_failed.json`
      keeps `positive_controller_source_count=0` with rejection reasons
      `metric_or_slot_drift_check_failed`,
      `missing_policy_obs_or_wm_condition_contract`,
      `world_model_interface_not_receding_or_live_refresh`, and
      `manual_visual_review_not_positive`. Predictor smoke confirmed the
      current one-shot trajectory reports `one_shot_task_state_trajectory`
      while the merged receding retest reports
      `receding_teacher_forced_segments`.
      Follow-up patch on 2026-06-06 fixed legacy provenance parsing in
      `inspect_wm_policy_distillation_readiness.py`: old manifests without a
      top-level `world_model_interface_report` now infer
      `receding_teacher_forced_segments` from per-segment
      `world_model_checkpoint_reports` when they contain prefix-boundary /
      valid-prediction-start metadata. Regression report
      `experiments/world_model_task_rebinding/wm_policy_distillation/interface_legacy_inferred_preflight_20260606_1810/readiness_cosmos_failed.json`
      still has `positive_controller_source_count=0`, but no longer rejects
      the failed receding run for interface provenance. Remaining rejection
      reasons are the actual failures:
      `metric_or_slot_drift_check_failed`,
      `missing_policy_obs_or_wm_condition_contract`, and
      `manual_visual_review_not_positive`; post-trigger peg-head drift mean is
      `0.2357m` and grasp disagreement is `0.7393`.
- [ ] 2026-06-06 00:50+08:00 Cosmos3/controller recovery status:
      this was not a clean restart from scratch. Allocation `107677` was
      canceled by system/root (`CANCELLED by 0`) at
      `2026-06-05T23:36:54+08:00`, killing readout step `107677.25` after
      `03:32:30`; the readout reached step `4675`, so there is no 5000-step
      eval. The best checkpoint remains step `4250`; step `4500` was not
      better. I then incorrectly canceled recovery allocation `109215`
      (`CANCELLED by 2059`) after it had started for 8 seconds. Current held
      allocation is `109227` on `server13`; do not intentionally release it
      while aligned Cosmos/controller work can still run. The recovered
      readout export under
      `experiments/world_model_task_rebinding/cosmos3/task_state_readout_full1000_framemode_20260605_200425/action_eval_task_state_prediction`
      completed from the step-4250 checkpoint and has future hole/peg/TCP/
      peg-head-in-hole RMSE `0.019377` / `0.017163` / `0.017730` /
      `0.016601` m on the 93-frame action-eval trajectory.
- [ ] 2026-06-06 00:50+08:00 Cosmos controller diagnostic status:
      patched `evaluate_rebinding_controller.py` so a finite precomputed
      Cosmos task-state trajectory is not reused past its covered frames, the
      controller preserves the observed static hole orientation when using
      Cosmos task-state hole-position predictions, and `search` phase mode
      enters `insert` once the existing physical insert-readiness condition
      is met. These are implementation/controller fixes, not evaluation-gate
      changes. The newest controller runs still fail final inserted-state
      success and all use `slot_source=oracle`, so they are diagnostic
      evidence only, not RGB-D method evidence. The merged receding Cosmos
      trajectory `62..218` with orientation fix improved lateral alignment
      but parked before the hole: `insertready` final peg-head-at-hole
      `[-0.110804, -0.000830, -0.004040]`, `success_at_end=false`,
      `bridge_phase_counts={align:182, insert:64}`. Higher insert-axis and
      guarded variants also failed (`axis04`, `guardedinsert`, `guard006`).
      Inspected contact sheets show the peg held near the hole mouth but not
      inserted. Current failure classification: controller/physics bridge
      failure after Cosmos readout export, not a Cosmos video-view failure and
      not task success.
- [ ] 2026-06-06 00:59+08:00 full1000 Cosmos3 data reset boundary:
      the new render did intentionally start at sample index `0` because the
      previously active full1000 Cosmos3 derived dataset, action-condition
      export, SFT outputs, readout outputs, and eval outputs were moved out of
      the active tree to
      `experiments/_archive/world_model_task_rebinding/20260606_superseded_original_cosmos3_full1000`.
      Reason: after the user approved the corrected ManiSkill-default 10-video
      preview, the active method must not reuse any old full1000 video/action/
      SFT artifacts whose provenance was mixed with the earlier dirty Cosmos
      data path. The only reused inputs are the exact `1000` saved env-state
      H5 source trajectories listed in
      `experiments/world_model_task_rebinding/cosmos3/full1000_env_state_source_h5s_regen_20260606.txt`;
      these are simulator state sources for re-rendering, not old Cosmos
      videos. New outputs are under
      `experiments/world_model_task_rebinding/cosmos3/sft_dataset_full1000_maniskill_default_regen_20260606_0055`,
      `experiments/world_model_task_rebinding/cosmos3/action_state_conditions_full1000_maniskill_default_regen_20260606_0055`,
      and
      `experiments/world_model_task_rebinding/cosmos3/sft_full1000_maniskill_default_actioncond_regen_20260606_0055`.
      The tmux render/SFT session is
      `cosmos3_regen_full1000_20260606_0055` inside held allocation `109227`
      on `server13`. A separate `salloc` request `109651`
      (`cosmos3_regen6h`) started early on `server13`, but a targeted shard
      canary inside that allocation failed before writing any shard videos:
      `torch` could not initialize NVML, SAPIEN/Vulkan raised
      `vk::createInstanceUnique: ErrorIncompatibleDriver`, and a direct
      `nvidia-smi` step failed with `Unable to determine the device handle for
      GPU0: 0000:AB:00.0`. This is a current job-local rendering/GPU failure,
      not a standing bad-node policy. `109651` was canceled for concrete
      allocation failure before it could become the render continuation. A
      no-dependency 24-hour request `109755` had also been canceled while
      still pending because it had no stable earlier start and could have
      overlapped with `109651`. Dependent request `109763`
      (`cosmos3_regen24h_after6h`) then started on `server13`, but the same
      direct `nvidia-smi` preflight failed with the same GPU handle error, so
      `109763` was canceled before it could render or write. The active long
      continuation is now `109804` (`cosmos3_regen24h_excl13`) in tmux
      `cosmos3_regen_full1000_resume24h_excl13_20260606_0121`, submitted with
      explicit `--exclude=server13` for this live GPU-handle failure only.
      It forecasts `2026-06-06T04:14:42+08:00`. The current single-writer
      chain is `109227` -> `109804`. At the latest check the new root had
      `186` finalized MP4s and the active render log was around sample `172`.
      Follow-up `2026-06-06T01:22+08:00`: `109804` forecast improved to
      `2026-06-06T02:47:41+08:00` on `server43`. Added
      `scripts/slurm/run_cosmos3_maniskill_render_shard.sbatch`, a bounded
      one-GPU shard wrapper with `nvidia-smi` preflight and
      `--no-write-canonical-metadata`, so disjoint high-index videos can be
      rendered without overwriting the canonical full1000 JSONL/manifest.
      Submitted only two high-index shard jobs, both with live-failure
      `--exclude=server13`: `109833` for indices `750..874` (forecast
      `2026-06-06T04:16:42+08:00` on `server42`) and `109834` for
      `875..999` (forecast `2026-06-06T05:00:00+08:00` on `server28`).
      These are far ahead of the sequential writer and are intended only to
      accelerate tail rendering; the main full wrapper must still do the final
      exact-1000 canonical manifest/JSONL inspection and action-condition/SFT
      handoff.
      Follow-up post-SFT automation: patched
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh` so it waits
      for the regenerated action JSONL and SFT config instead of exiting
      before those files exist. Started tmux
      `cosmos3_regen_post_sft_action_eval_20260606_0126`, which waits on the
      new action-condition root and new SFT root, selects `best_val`, and will
      run action-conditioned forward-dynamics reconstruction after SFT
      completes. Added and started
      `scripts/slurm/watch_cosmos3_action_eval_readout_controller.sh` in tmux
      `cosmos3_regen_readout_controller_watch_20260606_0129`; it waits for the
      regenerated canonical dataset manifest and post-SFT action-eval marker,
      then trains a new frame-mode task-state readout on the regenerated
      dataset (`5000` steps, `batch_size=128`, `eval_every=250`,
      `max_eval_batches=80`), decodes the generated action-eval video, and
      launches the existing Cosmos task-state controller diagnostic. This is a
      controller-interface continuation, not an RGB-D task-success claim.
      Follow-up `2026-06-06T01:52+08:00`: the apparent restart was intentional
      data-provenance reset after rejecting the dirty old Cosmos3 visual data,
      not reuse of old full1000 videos and not a stopped/released GPU cycle.
      Current Slurm state contains only regenerated-data jobs `109804`
      (main 24h continuation on `server21`), `109833` (disjoint shard
      `750..874` on `server55`), and `109834` (disjoint shard `875..999` on
      `server43`). The new dataset root has `461/1000` finalized MP4s;
      `109804` is writing fresh low/mid-index videos after skipping existing
      regenerated samples, `109834` is producing `09xx` videos, and `109833`
      has passed GPU preflight and is rendering index `0750` but has not yet
      finalized it. No old object-slot WM or superseded controller job is
      active in `squeue`.
      Follow-up `2026-06-06T02:11+08:00`: `109834` completed `875..999`.
      The held `109804` allocation was reused directly, not requeued, but
      index `0410` hit `vk::Device::waitForFences: ErrorDeviceLost` twice on
      `server21`; a follow-up `0411..749` shard in the same allocation also
      made no first-frame progress, so `109804` was canceled as a current
      render allocation failure. Index `0750` also failed to produce a first
      frame on `109833`/`server55` and replacement `110084`/`server09`, so it
      is isolated instead of blocking `751..874`. H5 metadata checks for
      `0410`, neighbors `0409/0411`, and `0750` found finite actions, slots,
      actor poses, normalized quaternions, and no obvious H5 corruption. The
      duplicate-source search found no alternate/repair H5 for the same
      trajectories, so do not silently swap `0410` or `0750` to an easier
      source. The current dataset has `535/1000` finalized MP4s. Active queued
      bounded shards are `110134` for `411..749` and `110170` for `751..874`;
      samples `0410` and `0750` still require render-specific
      isolation/repair before exact-1000 inspection, action-condition export,
      SFT, and controller continuation.
      Follow-up `2026-06-06T02:19+08:00`: the apparent restart is only the
      approved data-provenance reset, not reuse of the old dirty full1000
      videos/action/SFT/readout path. The regenerated active root now has
      `584/1000` finalized MP4s. `110134` is running `411..749` on
      `server28` and has reached sample `0435`; `110170` is running
      `751..874` on `server09` from sample `0751`. Added optional
      `MAX_FRAMES` and `RENDER_TIMEOUT_SECONDS` controls to the shard wrapper
      without changing default full-shard behavior. Submitted bounded
      diagnostics `110240` for `0410` and `110241` for `0750`, each 1GPU/20m,
      one-frame render with a 900s timeout, writing only to
      `experiments/world_model_task_rebinding/cosmos3/diagnostics/render_singletons_0410_0750_20260606`.
      These diagnostic videos are not full1000 training data and must not be
      counted for exact-1000 inspection.
      Follow-up `2026-06-06T02:21+08:00`: canceled `110170` on `server09`
      because it reached `sample_start` for `0751` but produced no first-frame
      log, no temporary MP4, and only `00:00:10` average CPU in `sstat`,
      matching the earlier `server09` no-first-frame behavior on `0750`.
      Submitted replacement `110251` for `751..874` with 1GPU/1h and
      job-local `--exclude=server13,server21,server09`; the 1h and 3h probes
      forecast the same start, so the shorter request reduces reserved
      resource cost. Current active regenerated root count is `604/1000`;
      `110134` continues rendering `411..749` on `server28`.
      Follow-up `2026-06-06T02:24+08:00`: single-frame diagnostics `110240`
      and `110241` completed successfully on `server44`, proving `0410` and
      `0750` can render at least frame `0` with the approved camera. These
      diagnostic videos remain outside the training root. The active root
      still lacked complete `0410`/`0750` MP4s, so submitted full-frame repair
      jobs `110301` and `110302`, each 1GPU/20m with `MAX_FRAMES=0`, writing
      to the active regenerated dataset root and no canonical metadata.
      Current active regenerated root count is `610/1000`; exact1000
      inspection remains pending.
      Follow-up `2026-06-06T02:34+08:00`: added metadata-only canonicalization
      to `render_cosmos3_maniskill_sft_dataset.py` and verified it on a
      `/tmp` three-sample smoke. Added
      `run_cosmos3_regen_metadata_action_sft.sbatch`, a 1-H200/1-day exact
      postprocess chain that refuses to run unless the active root has exactly
      `1000` final MP4s and zero tmp MP4s, then runs canonical manifest/JSONL,
      strict inspection, action-condition export, Cosmos3 SFT, action eval,
      readout, and controller diagnostic in one allocation. `110251`,
      `110301`, and `110302` all started but failed at first-frame Vulkan
      `ErrorDeviceLost` on `server31`/`server60`; `0410`, `0750`, and
      `751..874` remain missing. Added `run_cosmos3_maniskill_render_ranges`
      and submitted `110333` for `410:411,750:751,751:875` with job-local
      exclusions tied to current failures only. `110134` continues on
      `server28`; current active root count is `704/1000`.
      Follow-up `2026-06-06T02:38+08:00`: `110333` completed on `server27`
      and successfully wrote full active-root `0410`, but the Slurm
      comma-separated `--export` syntax truncated `RANGES` to only `410:411`.
      Patched the ranges wrapper to accept semicolon-separated ranges and
      resubmitted `110339` with `RANGES='750:751;751:875'` via environment
      variables plus `--export=ALL`. Submitted dependent 1-H200/1-day chain
      `110336` for exact1000 metadata/action/SFT/action-eval/readout/
      controller and updated its dependency to `afterok:110134:110339` so it
      cannot run before both the middle shard and tail repair succeed.
      Current active root count is `734/1000`; `110134` has reached `0608`.
      Follow-up `2026-06-06T02:49+08:00`: this is not a second restart from
      zero. Completed regenerated MP4s remain in the active root; current
      work is only filling missing disjoint ranges left by render allocation
      failures. `110134` is rendering `0411..0749` on `server28`, `110339`
      is rendering `0750` and `0751..0874` on `server27`, and `110336`
      remains pending on `afterok:110134:110339`. The active root has
      `891/1000` finalized MP4s, and the postprocess/SFT chain cannot start
      until exact-count preflight sees exactly `1000` final MP4s with zero
      temporary MP4s.
      Follow-up `2026-06-06T02:57+08:00`: exact regenerated full1000 render
      is complete. `110339` completed `0750` and `0751..0874` on `server27`
      with `ExitCode=0:0`; `110134` completed `0411..0749` on `server28`
      with `render_status=0`. The active root has exactly `1000` finalized
      MP4s and zero temporary MP4s. Dependent job `110336` started on
      `server43` and passed its exact-count preflight, then began
      metadata-only canonicalization from the new dataset root. At
      `2026-06-06T03:01+08:00`, metadata-only was around index `316`; strict
      inspection, action-condition export, SFT, action eval, readout, and
      controller continuation remain pending.
      Follow-up `2026-06-06T03:12+08:00`: metadata-only canonicalization
      completed with `manifest.json` `num_videos=1000`. Strict inspection
      passed with train/val split `912/88`, readable nonblank 1024 videos,
      matching FPS/sizes, required state captions/metadata, and zero tmp MP4s.
      Action-condition export completed with exactly `1000` records. Cosmos3
      action-conditioned SFT started in the same `110336` allocation using
      `train/video_action_dataset_file.jsonl` and
      `val/video_action_dataset_file.jsonl`, `num_video_frames=93`,
      `conditioning_config={8:1.0}`, and structured causal robot/object state
      plus future action commands. This is foundation-WM component training,
      not controller success.
      Follow-up `2026-06-06T03:20+08:00`: SFT loaded
      `Cosmos3-Nano-DCP` successfully and started training. The data loader
      decoded the regenerated action-conditioned JSONL with `912` train
      videos/windows and `88` val videos/windows. Iteration `0` validation
      loss is `0.030806`; training then advanced normally at about
      `5.8s/iter`, with observed losses around `0.02..0.06` through
      iteration `21`. This is only the pre-finetune/early-training baseline;
      the next meaningful comparison point is validation/checkpoint at
      iteration `300`.
      Follow-up `2026-06-06T03:52+08:00`: user corrected the regeneration
      boundary: this 6/6 regenerated path may continue, but future dataset
      reset/regeneration must not be done unilaterally because a downstream
      result is bad; first report evidence and options to the user. Recorded
      the rule in `AGENTS.md` and this TODO. In active job `110336`, checkpoint
      `iter_000000300` saved successfully with model/optimizer/scheduler/
      trainer DCP files and `model/.metadata`. Validation loss at iteration
      `300` is `0.016981`, down from iteration `0` `0.030806`. Job `110336`
      remains running on `server43`; continue to later checkpoints and
      post-SFT action-eval/readout/controller chain without releasing the GPU.
      Follow-up `2026-06-06T03:57+08:00`: preflight found that the regenerated
      post-SFT controller watcher still defaulted to `slot_source=oracle`,
      which would make the automatic controller run diagnostic-only even after
      Cosmos3 SFT/readout. Patched
      `scripts/slurm/watch_cosmos3_action_eval_readout_controller.sh` to
      default the regenerated chain to `slot_source=rgbd`, using the compliant
      exact1000 RGB-D slot ensemble
      `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job102292`
      and saving RGB-D observations. Patched
      `scripts/slurm/run_cosmos_task_state_controller_in_allocation.sh` to
      pass RGB-D slot ensemble/checkpoint/obs-mode/compliance arguments into
      `evaluate_rebinding_controller.py`. `bash -n` passed for both wrappers.
      The `job102292` inspection reports exact `1000` RGB-D inputs, `4`
      complete members, `4xH200`, elapsed time above `3h`, and
      `compliant_training_evidence=True`,
      `rgbd_slot_training_evidence=True`,
      `rgbd_perception_input_evidence=True`. Its continuous slot RMSEs remain
      diagnostics, not downstream blockers. This patch does not change the
      evaluation protocol and does not claim controller success; it prevents
      oracle-slot controller output from being mistaken for RGB-D method
      evidence.
      Follow-up `2026-06-06T04:01+08:00`: `110336` remains running on
      `server43`; SFT reached iteration `401` with normal `5.8s/iter`
      training after the iteration-300 validation/checkpoint, and no
      crash/NaN marker was found. The action-conditioned val JSONL sample has
      `condition_frame_indexes_vision=[0..7]`, action condition frames
      `0..91`, `action_condition_policy=video_prefix_plus_causal_prefix_robot_object_state_and_future_action_commands`,
      and metadata `conditioning_policy=video_prefix_not_single_image_i2v`.
      Its causal boundary states that prefix object/task state is repeated
      through future action rows and future ground-truth object poses are not
      written into the action condition. The post-SFT action-eval watcher uses
      `CHECKPOINT_SELECTION=best_val`, selected from validation loss records
      with existing DCP checkpoint metadata, then computes all-frame and
      future-only reconstruction artifacts. Next meaningful SFT checkpoint is
      iteration `600`; do not release the allocation while it is progressing.
      Follow-up `2026-06-06T04:03+08:00`: `110336` is still running on
      `server43`. Latest SFT log reached iteration `427`; only checkpoint
      `iter_000000300` exists so far, and `latest_checkpoint.txt` points to
      `iter_000000300`. No post-SFT action-eval/readout/controller files have
      appeared yet, as expected before `sft_completed`. Structured provenance
      check over `train`/`val` action JSONL counted `912/88` rows and found
      `bad_count=0`: all `vision_path` entries point to the regenerated
      dataset root and all `action_path` entries point to the regenerated
      action-condition root. The SFT config has `conditioning_config: {8:
      1.0}`, `cfg_dropout_rate: 0.0`, `num_video_frames: 93`, and
      `temporal_interval_mode: force_one` for both train and val. Controller
      CLI help confirms the patched flags exist:
      `--rgbd-slot-ensemble-dir`,
      `--rgbd-slot-require-compliant/--no-rgbd-slot-require-compliant`, and
      `--save-rgbd-observations/--no-save-rgbd-observations`.
      Follow-up `2026-06-06T04:07+08:00`: `110336` remains running on
      `server43`; SFT reached iteration `463` with no `sft_completed` marker
      yet. The only checkpoint remains `iter_000000300`, so the next required
      evidence point is iteration `600` validation/checkpoint. No post-SFT
      action-eval/readout/controller files exist yet, as expected.
      Follow-up `2026-06-06T04:24+08:00`: user explicitly clarified that
      future full dataset regeneration must not be inferred from poor
      downstream results; evidence and options must be reported before any
      reset. The current 6/6 path is allowed to continue. In job `110336`,
      checkpoint `iter_000000600` saved successfully with complete DCP
      metadata and `latest_checkpoint.txt` now points to `iter_000000600`.
      Validation loss at iteration `600` is `0.017195`, slightly worse than
      iteration `300` `0.016981`, so the current best validation checkpoint
      remains `iter_000000300`. This is normal checkpoint-selection evidence,
      not a reason to alter or regenerate data. SFT resumed after validation
      and reached iteration `608` with normal step time; keep the allocation
      running toward later checkpoints and the configured post-SFT
      action-eval/readout/RGB-D-slot controller chain.
      Follow-up `2026-06-06T04:56+08:00`: `110336` remains running on
      `server43` without releasing the allocation. Checkpoint
      `iter_000000900` saved successfully with complete DCP files and
      metadata for model, optimizer, scheduler, and trainer. Validation loss
      at iteration `900` is `0.014009`, improving over `300` `0.016981` and
      `600` `0.017195`, so current best validation checkpoint is now
      `iter_000000900`. SFT resumed after validation and reached iteration
      `903`; `sft_completed` is still absent, so post-SFT action-eval/
      readout/controller files are still expected to wait. This is SFT
      component evidence only, not controller success.
      Follow-up `2026-06-06T05:29+08:00`: `110336` remains running on
      `server43`. Checkpoint `iter_000001200` saved successfully with
      complete DCP files and metadata for model, optimizer, scheduler, and
      trainer. Validation loss at iteration `1200` is `0.015577`, worse than
      iteration `900` `0.014009`, so current best validation checkpoint
      remains `iter_000000900`. SFT resumed after validation and reached
      iteration `1207`; `sft_completed` is still absent and post-SFT
      action-eval/readout/controller artifacts remain absent as expected.
      Follow-up `2026-06-06T06:02+08:00`: `110336` remains running on
      `server43`. Checkpoint `iter_000001500` saved successfully with
      complete DCP files and metadata for model, optimizer, scheduler, and
      trainer. Validation loss at iteration `1500` is `0.015305`, better
      than `1200` but still worse than iteration `900` `0.014009`; current
      best validation checkpoint remains `iter_000000900`. SFT resumed after
      validation and reached iteration `1507`; `sft_completed` is still
      absent and post-SFT action-eval/readout/controller artifacts remain
      absent as expected.
      Follow-up `2026-06-06T06:35+08:00`: SFT completed without releasing
      job `110336`. Checkpoint `iter_000001800` saved successfully with
      complete DCP files and metadata for model, optimizer, scheduler, and
      trainer. Full regenerated SFT validation losses are `0`: `0.030806`,
      `300`: `0.016981`, `600`: `0.017195`, `900`: `0.014009`, `1200`:
      `0.015577`, `1500`: `0.015305`, and `1800`: `0.015911`. The
      `sft_completed` marker was written at `2026-06-06T06:35:27+08:00`.
      Post-SFT action eval started in the same allocation at
      `2026-06-06T06:35:31+08:00`, selected `iter_000000900` by
      best-validation loss with existing DCP metadata, and uses the
      regenerated action-condition val JSONL with vision prefix frames
      `[0..7]`, action frames `0..91`, and regenerated condition video. The
      action-eval inference is running; reconstruction/readout/controller
      evidence is still pending.
      Follow-up `2026-06-06T06:40+08:00`: directly inspected the post-SFT
      action-eval reconstruction contact sheets. The generated video uses the
      approved oblique ManiSkill-default view and is nonblank; the robot, peg,
      block/hole, and table remain readable. The prediction tracks the broad
      manipulation geometry, but the future segment shows visible late
      peg/gripper pose error and edge ghosting around contact, so this is
      usable Cosmos3 reconstruction/readout input evidence, not controller
      success. Metrics from regenerated artifacts are all-frame mean PSNR
      `28.652080371675584` over `93` frames and future-only mean PSNR
      `25.539106680378573` over `64` frames; all-frame/future MSE are
      `0.0022117838489457524` and `0.003086370921209891`. The selected
      checkpoint is confirmed by `selected_checkpoint.json` as
      `iter_000000900`, not a later checkpoint. Readout training has started
      inside the still-running `110336` allocation with `slot_source=rgbd`;
      after the initial regenerated-MP4 decode it printed step `50` with loss
      `0.5515707731246948`, `loss_cont=0.4675872027873993`, and
      `loss_bin=0.4199177622795105`. No dataset reset, archival, evaluation
      change, or new Slurm submission was made.
      Follow-up `2026-06-06T06:51+08:00`: readout reached its first eval at
      step `250` in the same running allocation. `metrics_latest.json`,
      `metrics_best.json`, `model_latest.pt`, and `best_model.pt` were
      written. Early future RMSE is still poor: hole `0.07427370548248291` m,
      peg `0.08806054294109344` m, TCP `0.06410109251737595` m, and
      peg-head-in-hole `0.13923463225364685` m; future grasped accuracy is
      `0.482421875`, inserted accuracy `1.0`. This is early readout evidence,
      not a data-quality reset trigger and not controller success. Continue
      the configured `5000`-step readout/controller chain without releasing
      job `110336`.
      Follow-up `2026-06-06T07:02+08:00`: readout step `500` improved over
      step `250` and became the current `best_model.pt`. Future RMSE is now
      hole `0.059404805302619934` m, peg `0.04453597962856293` m, TCP
      `0.03888651728630066` m, and peg-head-in-hole
      `0.12904539704322815` m; future grasped accuracy improved to
      `0.6681463122367859`, inserted accuracy remains `1.0`. Peg/TCP
      readout improved substantially, hole improved moderately, and
      peg-head-in-hole remains weak. This supports continuing the same
      readout training rather than resetting data or changing evaluation.
      Follow-up `2026-06-06T07:13+08:00`: readout step `750` improved again
      and is the current `best_model.pt`. Future RMSE is hole
      `0.0514037199318409` m, peg `0.04213445633649826` m, TCP
      `0.031081074848771095` m, and peg-head-in-hole
      `0.11707198619842529` m; future grasped accuracy is
      `0.6844815611839294`, inserted accuracy remains `1.0`.
      Peg-head-in-hole remains the weak readout component, but all future
      continuous metrics improved over step `500`. Job `110336` remains
      running on `server43`; continue toward the configured `5000` steps and
      downstream RGB-D-slot controller run.
      Follow-up `2026-06-06T07:24+08:00`: readout step `1000` improved
      again and is the current `best_model.pt`. Future RMSE is hole
      `0.045761361718177795` m, peg `0.038946352899074554` m, TCP
      `0.03261781111359596` m, and peg-head-in-hole
      `0.09443950653076172` m; future grasped accuracy is
      `0.7212358117103577`, inserted accuracy remains `1.0`.
      Peg-head-in-hole improved materially from step `750` `0.117071986` m
      to `0.094439507` m, but this is still readout evidence only. Keep
      job `110336` running toward full `5000` steps and then RGB-D-slot
      controller evaluation.
      Follow-up `2026-06-06T07:35+08:00`: readout step `1250` improved
      again and is the current `best_model.pt`. Future RMSE is hole
      `0.04293438047170639` m, peg `0.03206409886479378` m, TCP
      `0.02734413556754589` m, and peg-head-in-hole
      `0.08476512134075165` m; future grasped accuracy is
      `0.7837358117103577`, inserted accuracy remains `1.0`.
      All future continuous metrics improved over step `1000`. This remains
      readout evidence only, not controller success; continue the full
      readout/controller chain in job `110336`.
      Follow-up `2026-06-06T07:46+08:00`: readout step `1500` completed.
      `metrics_latest.json` is now step `1500`, but `metrics_best.json`
      remains step `1250` because step `1500` improved some terms and
      worsened others under the current readout selection rule. Future step
      `1500` RMSE is hole `0.04084068164229393` m, peg
      `0.03133416920900345` m, TCP `0.033429957926273346` m, and
      peg-head-in-hole `0.08762893825769424` m; future grasped accuracy is
      `0.8002485632896423`, inserted accuracy remains `1.0`. Compared with
      step `1250`, hole/peg/grasped improved, while TCP and peg-head-in-hole
      worsened, so this is not a new best checkpoint. Continue the same
      `110336` chain toward `5000` steps and downstream RGB-D-slot controller
      evaluation; do not reset data or change evaluation.
      Follow-up `2026-06-06T07:57+08:00`: readout step `1750` completed and
      became the current `best_model.pt`. Future RMSE is hole
      `0.040771082043647766` m, peg `0.03325921669602394` m, TCP
      `0.032188303768634796` m, and peg-head-in-hole
      `0.0688704177737236` m; future grasped accuracy is
      `0.7979403138160706`, inserted accuracy remains `1.0`.
      Compared with step `1250`, peg-head-in-hole improved materially
      `0.08476512134075165` -> `0.0688704177737236` m and
      `loss_cont_norm_mean` improved `14.909246062486636` ->
      `14.197815336673024`, while peg/TCP/grasped are mixed. This is still
      readout evidence only; continue the same `110336` chain toward `5000`
      steps and downstream RGB-D-slot controller evaluation.
      Follow-up `2026-06-06T08:08+08:00`: readout step `2000` completed.
      `metrics_latest.json` is step `2000`, but `metrics_best.json` remains
      step `1750` because the training script selects best by
      `future.peg_head_hole_rmse_m` only. Step `2000` future metrics are:
      hole `0.044307589530944824` m, peg `0.03053366020321846` m, TCP
      `0.0302606038749218` m, peg-head-in-hole `0.07069564610719681` m,
      grasped accuracy `0.8364701867103577`, inserted accuracy `1.0`, and
      `loss_cont_norm_mean=13.824633231960084`. Compared with step `1750`,
      peg/TCP/grasped and normalized continuous loss improved, but
      peg-head-in-hole worsened `0.0688704177737236` ->
      `0.07069564610719681` m and hole worsened, so the current best
      controller-facing checkpoint remains step `1750`. Continue job
      `110336`; no data reset, no evaluation change.
      Follow-up `2026-06-06T08:18+08:00`: readout step `2250` completed and
      became the current `best_model.pt`. Future metrics are: hole
      `0.03909970074892044` m, peg `0.027749789878726006` m, TCP
      `0.024479370564222336` m, peg-head-in-hole
      `0.05806586146354675` m, grasped accuracy `0.8425071239471436`,
      inserted accuracy `1.0`, and
      `loss_cont_norm_mean=12.137639397167158`. Compared with step `1750`,
      peg-head-in-hole improved `0.0688704177737236` ->
      `0.05806586146354675` m, and hole/peg/TCP/grasped also improved.
      This is the best controller-facing readout checkpoint so far, but still
      not dynamic task success. Keep job `110336` running toward `5000`
      steps and downstream RGB-D-slot controller evaluation.
      Follow-up `2026-06-06T08:29+08:00`: readout step `2500` completed and
      became the current `best_model.pt`. Future metrics are: hole
      `0.040213026106357574` m, peg `0.029873337596654892` m, TCP
      `0.025539740920066833` m, peg-head-in-hole
      `0.05561114847660065` m, grasped accuracy `0.8627485632896423`,
      inserted accuracy `1.0`, and
      `loss_cont_norm_mean=11.999509891573402`. Compared with step `2250`,
      peg-head-in-hole improved `0.05806586146354675` ->
      `0.05561114847660065` m and grasped accuracy improved, while
      hole/peg/TCP slightly worsened. This is still readout evidence only;
      keep job `110336` running toward `5000` steps and downstream RGB-D-slot
      controller evaluation.
      Follow-up `2026-06-06T08:40+08:00`: readout step `2750` completed, but
      `metrics_best.json` remains step `2500`. Future metrics are: hole
      `0.03938684239983559` m, peg `0.030739478766918182` m, TCP
      `0.028346864506602287` m, peg-head-in-hole
      `0.05911800637841225` m, grasped accuracy `0.8675426244735718`,
      inserted accuracy `1.0`, and
      `loss_cont_norm_mean=11.787130596351064`. Compared with step `2500`,
      hole, grasped accuracy, and normalized continuous loss improved, but
      peg, TCP, and the controller-facing peg-head-in-hole metric worsened
      `0.05561114847660065` -> `0.05911800637841225` m. Keep step `2500` as
      best and continue the same `110336` chain; this is not a data reset,
      evaluation change, or dynamic task-success claim.
      Follow-up `2026-06-06T08:51+08:00`: readout step `3000` completed, and
      `metrics_best.json` still remains step `2500`. Future metrics are: hole
      `0.04399019479751587` m, peg `0.02864331193268299` m, TCP
      `0.02398841269314289` m, peg-head-in-hole
      `0.06199472397565842` m, grasped accuracy `0.8641690611839294`,
      inserted accuracy `1.0`, and
      `loss_cont_norm_mean=11.757421652009876`. Compared with step `2500`,
      TCP and normalized continuous loss improved, peg is slightly better,
      but hole and peg-head-in-hole worsened, so step `2500` remains best for
      handoff/readout selection. Continue job `110336`; no data reset or
      evaluation change.
      Follow-up `2026-06-06T09:01+08:00`: readout step `3250` completed, but
      `metrics_best.json` still remains step `2500`. Future metrics are: hole
      `0.03945852816104889` m, peg `0.026644907891750336` m, TCP
      `0.025204021483659744` m, peg-head-in-hole
      `0.057094987481832504` m, grasped accuracy `0.8819246888160706`,
      inserted accuracy `1.0`, and
      `loss_cont_norm_mean=11.175591723310633`. Compared with step `2500`,
      hole, peg, grasped accuracy, and normalized continuous loss improved,
      but peg-head-in-hole remains slightly worse `0.05561114847660065` ->
      `0.057094987481832504` m and TCP is slightly worse. Keep step `2500`
      as the selected readout checkpoint and continue job `110336`.
      Follow-up `2026-06-06T09:12+08:00`: readout step `3500` completed, and
      `metrics_best.json` remains step `2500`. Future metrics are: hole
      `0.04120657965540886` m, peg `0.025055307894945145` m, TCP
      `0.02387113869190216` m, peg-head-in-hole
      `0.059890106320381165` m, grasped accuracy `0.8689630627632141`,
      inserted accuracy `1.0`, and
      `loss_cont_norm_mean=11.271757731805915`. Peg and TCP improve over
      step `2500`, but hole, grasped accuracy, and peg-head-in-hole are worse.
      Keep step `2500` as the selected readout checkpoint and continue the
      same allocation toward downstream RGB-D-slot controller evaluation.
      Follow-up `2026-06-06T09:23+08:00`: readout step `3750` completed and
      became the current `best_model.pt`. Future metrics are: hole
      `0.0374438613653183` m, peg `0.02463959902524948` m, TCP
      `0.022496521472930908` m, peg-head-in-hole
      `0.054317083209753036` m, grasped accuracy `0.8860085010528564`,
      inserted accuracy `1.0`, and
      `loss_cont_norm_mean=10.814533774337693`. Compared with step `2500`,
      peg-head-in-hole improved `0.05561114847660065` ->
      `0.054317083209753036` m, and hole, peg, TCP, grasped accuracy, and
      normalized continuous loss also improved. This is the best
      controller-facing readout checkpoint so far. Continue job `110336`
      toward `5000` steps and downstream RGB-D-slot controller evaluation.
      Follow-up `2026-06-06T09:34+08:00`: readout step `4000` completed and
      became the current `best_model.pt`. Future metrics are: hole
      `0.03905976936221123` m, peg `0.022460995241999626` m, TCP
      `0.019693810492753983` m, peg-head-in-hole
      `0.054067693650722504` m, grasped accuracy `0.8876065611839294`,
      inserted accuracy `1.0`, and
      `loss_cont_norm_mean=10.862390294452567`. Compared with step `3750`,
      peg-head-in-hole improved `0.054317083209753036` ->
      `0.054067693650722504` m, and peg, TCP, and grasped accuracy improved;
      hole and normalized continuous loss slightly worsened. Continue job
      `110336`; this is still readout/controller-interface evidence only.
      Follow-up `2026-06-06T09:45+08:00`: readout step `4250` completed and
      became the current `best_model.pt`. Future metrics are: hole
      `0.0373920276761055` m, peg `0.025213023647665977` m, TCP
      `0.021134328097105026` m, peg-head-in-hole
      `0.050709620118141174` m, grasped accuracy `0.9021661877632141`,
      inserted accuracy `1.0`, and
      `loss_cont_norm_mean=10.654650955722246`. Compared with step `4000`,
      peg-head-in-hole improved `0.054067693650722504` ->
      `0.050709620118141174` m, and hole, grasped accuracy, and normalized
      continuous loss improved; peg and TCP worsened. This is the best
      controller-facing readout checkpoint so far. Continue job `110336`
      toward `5000` steps and downstream RGB-D-slot controller evaluation.
      Follow-up `2026-06-06T09:56+08:00`: readout step `4500` completed and
      became the current `best_model.pt`. Future metrics are: hole
      `0.03787080943584442` m, peg `0.022331496700644493` m, TCP
      `0.018644850701093674` m, peg-head-in-hole
      `0.04810020327568054` m, grasped accuracy `0.8979048132896423`,
      inserted accuracy `1.0`, and
      `loss_cont_norm_mean=10.520152124608135`. Compared with step `4250`,
      peg-head-in-hole improved `0.050709620118141174` ->
      `0.04810020327568054` m, and peg, TCP, and normalized continuous loss
      improved; hole and grasped accuracy slightly worsened. This is the best
      controller-facing readout checkpoint so far. Continue job `110336`
      toward `5000` steps and downstream RGB-D-slot controller evaluation.
      Follow-up `2026-06-06T10:08+08:00`: readout step `4750` completed but
      did not replace the step `4500` `best_model.pt`. Future metrics are:
      hole `0.04096656292676926` m, peg `0.025051651522517204` m, TCP
      `0.02279924973845482` m, peg-head-in-hole
      `0.04927215725183487` m, grasped accuracy `0.9005681872367859`,
      inserted accuracy `1.0`, and
      `loss_cont_norm_mean=10.548804763242064`. Compared with step `4500`,
      grasped accuracy improved slightly, but peg-head-in-hole, hole, peg,
      TCP, and normalized continuous loss worsened, so the controller-facing
      selection remains step `4500`. Continue the same allocation to the
      planned `5000` readout completion and RGB-D-slot controller; do not
      reset data, archive data, or change evaluation because of this latest
      non-best eval.
      Follow-up `2026-06-06T10:20+08:00`: readout step `5000` completed but
      did not replace the step `4500` `best_model.pt`. Future metrics are:
      hole `0.03836413845419884` m, peg `0.022073090076446533` m, TCP
      `0.02127661369740963` m, peg-head-in-hole
      `0.05461423099040985` m, grasped accuracy `0.9005681872367859`,
      inserted accuracy `1.0`, and
      `loss_cont_norm_mean=10.410361389726955`. Compared with step `4500`,
      hole, peg, grasped accuracy, and normalized continuous loss improved,
      but the controller-facing peg-head-in-hole metric worsened, so
      selection remains step `4500`. The watcher then exported
      `action_eval_task_state_prediction/cosmos_action_eval_controller_trajectory.json`
      and ran the RGB-D-slot Cosmos controller in the same job.
      Follow-up `2026-06-06T10:23+08:00`: controller job `110336` completed
      naturally after the chained controller run; it did not remain allocated
      after the script ended. Controller artifacts:
      `experiments/world_model_task_rebinding/rebinding_controller/cosmos_task_state_controller_regen_20260606_chain/metrics.json`,
      `rollouts.h5`, `inspection.json`, `inspection.md`, and
      `videos/hole_constant_seed702000.mp4` plus
      `videos/hole_constant_seed702000_contact_sheet.png`. This run used
      `slot_source=rgbd`, `control_policy=rebind_cosmos`,
      `hole_predictor=cosmos_task_state`, RGB-D slot ensemble
      `job102292`, and no fallback checkpoint. It failed: `success_at_end=0`,
      `success_once=0`, `triggered=1`, `rebind_count=228`,
      `bridge_count=24`, `retreat_count=222`, and `infeasible_count=222`.
      Directly inspected the contact sheet: the dynamic hole/box moved, but
      the robot spent most of the post-trigger episode near the peg with
      repeated retreat/rebind behavior and never inserted the peg. Inspection
      found RGB-D method-input evidence present and frame-aligned, but the
      failure is concrete. H5 comparison shows the root symptom: true metric
      grasp remained `true` from frame `54` to `300`, while RGB-D control
      `slots/grasped` was `true` only through frame `101`; from frame `102`
      onward the controller believed the peg was not grasped. At frame `300`,
      true metric peg-head-to-hole yz norm was `0.12192532420158386` m, while
      control slots estimated `0.00903400219976902` m. Failure localization:
      RGB-D control-state/perception or interface calibration drift after the
      trigger, not evidence that data should be regenerated. Do not reset,
      regenerate, archive, or replace the dataset from this result without a
      user-approved plan.
      Follow-up `2026-06-06T10:32+08:00`: deeper non-destructive H5/code
      audit confirms the controller uses RGB-D ensemble `bin_prob[0] >= 0.5`
      as the online `grasped` control state. Frames `90..101` have metric and
      control grasp both true, grasp probability mean
      `0.8937607407569885`, and peg-head error mean
      `0.03871897980570793` m. Frames `102..300` have true metric grasp
      still true but control grasp always false, grasp probability mean
      `0.05690806731581688`, peg position error mean
      `0.25849029421806335` m, TCP position error mean
      `0.29750555753707886` m, and peg-head error mean
      `0.2114732563495636` m. The generated diagnostic sheet
      `videos/rgbd_control_drift_frames_096_107.png` shows base/hand RGB
      inputs remain nonblank and the peg/gripper are visible when grasp
      probability falls below threshold. The existing RGB-D slot ensemble
      inspection already had weak continuous slot metrics
      (`peg_head_hole_rmse_m_mean=0.11620820047049721`,
      `peg_pos_rmse_m_mean=0.09115196972005164`), so the next aligned fix is
      not data regeneration but an explicitly approved RGB-D perception/state
      estimator repair: improve slot extraction, temporal fusion,
      contact/gripper-state handling, or controller-interface calibration
      while preserving real metric-slot scoring and video inspection. Do not
      make that repair or submit new heavy runs until the user approves one
      option, because the user explicitly required stopping for discussion
      after bad downstream results.
- [ ] 2026-06-05 Cosmos3 action-conditioned SFT/eval status:
      action-conditioned Cosmos3 SFT on the approved full1000 ManiSkill
      default-view data completed in allocation `107677` without releasing the
      H200. Validation losses across the original run and restart were
      `0`: `0.022813`, `300`: `0.017178`, `600`: `0.013811`, `900`:
      `0.018228`, `1200`: `0.016808`, `1500`: `0.015549`, and `1800`:
      `0.015494`. The post-SFT action eval selected `iter_000000600` by
      best validation loss, not the latest checkpoint. Do not directly
      interpret Cosmos3 diffusion/flow `val loss` as pixel MSE or PSNR; if
      `0.01` were literal `[0,1]` pixel MSE it would imply `20 dB`, but the
      active evidence is sampled reconstruction plus visual inspection. The
      selected action-conditioned eval produced all-frame mean PSNR
      `28.069774627572667` and future-only mean PSNR `24.6968681807353`.
      Directly inspected reconstruction sheets showed correct approved view
      and nonblank/readable manipulation geometry, with visible late
      contact/pose error. Debug tensors are available with `action [92,64]`,
      `vision [3,93,256,256]`, and `vision_latent [48,24,16,16]`. This
      justifies proceeding to a Cosmos prediction/latent -> task-state or
      uncertainty readout for controller integration. It is not controller
      success and not permission to resume the lightweight object-slot WM as
      the main method.
- [ ] 2026-06-05 controller-facing Cosmos readout continuation:
      added `scripts/world_model/train_cosmos3_task_state_readout.py` to train
      a supervised task-state decoder from Cosmos/reference video frames to
      hole, peg, TCP, peg-head-in-hole, grasped, and inserted state labels
      from the same trajectory H5 slots. This is a controller interface head,
      not a replacement world model and not controller success by itself.
      `py_compile` passed and tiny CPU smokes passed. The first clip-mode
      runs were canceled after proving per-clip decode was too slow; they are
      not evidence. The active run is frame-mode in tmux `cosmos_readout`,
      Slurm step `107677.25`, under
      `experiments/world_model_task_rebinding/cosmos3/task_state_readout_full1000_framemode_20260605_200425`.
      It uses all approved full1000 videos, `num_frames=93`,
      `future_start_frame=29`, `image_size=160`, `steps=5000`, and
      `batch_size=128`. Step `250` eval was poor: future hole/peg/TCP/
      peg-head-in-hole RMSE `0.074272` / `0.085055` / `0.060090` /
      `0.139875` m. Step `500` improved but is still not controller success:
      future hole/peg/TCP/peg-head-in-hole RMSE `0.060259` / `0.043135` /
      `0.036955` / `0.128709` m, future grasped accuracy `0.669922`.
      Step `750` improved again but is still controller-interface evidence,
      not controller success: future hole/peg/TCP/peg-head-in-hole RMSE
      `0.054460` / `0.040220` / `0.031032` / `0.120627` m, future grasped
      accuracy `0.759233`. Step `1000` improved the controller-critical
      relative geometry further: future hole/peg/TCP/peg-head-in-hole RMSE
      `0.045219` / `0.036967` / `0.030799` / `0.093520` m. Step `1250`
      improved again: future hole/peg/TCP/peg-head-in-hole RMSE `0.043610` /
      `0.031994` / `0.028372` / `0.084743` m, future grasped accuracy
      `0.775391`. Step `1500` is the current best checkpoint, with future
      hole/peg/TCP/peg-head-in-hole RMSE `0.041377` / `0.032977` /
      `0.032642` / `0.084226` m and future grasped accuracy `0.803267`.
      Step `1750` improved the controller-critical relative geometry further:
      future hole/peg/TCP/peg-head-in-hole RMSE `0.041268` / `0.033543` /
      `0.033780` / `0.066345` m, future grasped accuracy `0.797585`.
      Step `2000` was a mixed update from the same uninterrupted Slurm step
      `107677.25`, not a restarted run: future grasped accuracy, peg RMSE,
      and TCP RMSE improved to `0.839489`, `0.030425` m, and `0.027635` m,
      while future hole RMSE and peg-head-in-hole RMSE were worse than step
      `1750` at `0.043946` m and `0.068325` m. The best checkpoint therefore
      remains step `1750` for now, while the original 5000-step run continues.
      Step `2250` became the new best checkpoint from the same uninterrupted
      run: future hole/peg/TCP/peg-head-in-hole RMSE improved to `0.039351` /
      `0.027498` / `0.023971` / `0.057434` m, future grasped accuracy was
      `0.839666`, and `loss_cont_norm_mean=12.041701`. Continue the original
      5000-step run in allocation `107677`; do not restart or release the
      card.
      Step `2500` became the new best by the run's validation score:
      future hole/peg/TCP/peg-head-in-hole RMSE was `0.039723` /
      `0.031125` / `0.024518` / `0.056128` m, future grasped accuracy was
      `0.859197`, and `loss_cont_norm_mean=12.032602`. This improves
      peg-head-in-hole and grasp over step `2250`, while peg/TCP are slightly
      worse; keep the original run going to `5000` and preserve the best
      checkpoint.
      Steps `2750` and `3000` lowered the normalized continuous loss but did
      not replace the best checkpoint because peg-head-in-hole and/or hole
      geometry regressed. Step `3250` became the new best checkpoint:
      future hole/peg/TCP/peg-head-in-hole RMSE was `0.039515` /
      `0.025735` / `0.024587` / `0.055130` m, future grasped accuracy was
      `0.877308`, and `loss_cont_norm_mean=11.150957`. Continue the same run
      to `5000`.
      Step `3500` was not a new best, but step `3750` became the next best:
      future hole/peg/TCP/peg-head-in-hole RMSE was `0.037879` /
      `0.023525` / `0.021112` / `0.052700` m, future grasped accuracy was
      `0.883878`, and `loss_cont_norm_mean=10.801970`. The long run is still
      improving, so continue to `5000` before export/controller diagnostics.
      Step `4000` was not a new best, but step `4250` became the next best:
      future hole/peg/TCP/peg-head-in-hole RMSE was `0.037481` /
      `0.025772` / `0.020701` / `0.049435` m, future grasped accuracy was
      `0.899503`, and `loss_cont_norm_mean=10.596867`. The peg-head relative
      geometry has now improved below `5cm`; continue to the planned `5000`
      before export.
      Added
      `scripts/world_model/inspect_cosmos3_task_state_prediction.py` and
      patched `scripts/slurm/watch_cosmos3_readout_predict.sh` so the later
      best readout export writes a finite/metric inspection and a compact
      controller trajectory JSON. Patched
      `scripts/world_model/evaluate_rebinding_controller.py` with a
      `rebind_cosmos` / `hole_predictor=cosmos_task_state` code path that
      reads this compact trajectory as a controller diagnostic prediction
      source. Added
      `scripts/slurm/run_cosmos_task_state_controller_in_allocation.sh` to run
      that diagnostic from the held allocation after the trajectory exists.
      This preserves old CV/world-model/RGB-D-slot behavior and is not a
      task-success claim. Also added future-use task-state loss weighting
      args to the readout script; the active run is unchanged, but if the
      current 5000-step run remains weak, a weighted replacement can prioritize
      hole/TCP/peg-head geometry without releasing allocation `107677`.
      Continue training without releasing allocation `107677`; use the later
      best readout to decode the Cosmos-generated action-eval video before
      tying predictions to controller decisions.
- [ ] 2026-06-05 reconstruction-video continuation directive:
      After diagnostic video generation, use reconstruction error and direct
      visual inspection only as a sanity check that the generated rollout is
      finite, nonblank, physically interpretable, and not worse than the
      previous diagnostic baseline. If that sanity check is reasonable, proceed
      directly to the next RGB-D-derived training/evaluation step without
      asking the user or exiting. This is not a hard downstream gate and does
      not replace the required RGB-D slot, world-model, controller metrics, or
      inspected controller video evidence.
- [ ] 2026-06-05 user override, now approved for full1000 continuation:
      the previous continuation directive was paused for the
      Cosmos3/controller data reset because the old full1000 Cosmos/controller
      visual data were dirty for the foundation-WM path. The old videos were
      too small, too fast, and used the wrong/weak viewpoint. The required
      10-video human-inspection preview has now been approved by the user, so
      the active path is to regenerate the exact full1000 Cosmos3 SFT video
      dataset from saved env-state trajectories with the approved ManiSkill3
      default human-render camera, then run Cosmos3 SFT with validation-loss
      inspection, then proceed to downstream controller training/evaluation.
      2026-06-05 10:09+08:00: generated exactly 10 candidate preview videos
      and 10 review sheets under
      `experiments/world_model_task_rebinding/cosmos3/overhead_preview10_20260605_100816`
      from saved dynamic RGB-D H5 `env_states`, with `960x540`, `10 fps`,
      `120` selected frames, and source-frame coverage `0..300`. Evidence note:
      `docs/world_model_task_rebinding/2026-06-05_cosmos3_preview10_human_inspection_stop.md`.
      That candidate was stopped for user inspection and was later superseded
      by the accepted ManiSkill default-view candidate below.
      2026-06-05 follow-up correction: the 10:09 candidate view is still not
      accepted. Re-render exactly 10 candidates using the ManiSkill3
      `PegInsertionSide-v1` default human-render camera
      `look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])`, `fov=1`, `fps=30`, and
      no deliberate playback slowdown. Stop again after the 10 videos for user
      inspection.
      2026-06-05 10:37+08:00: generated the replacement 10 candidates under
      `experiments/world_model_task_rebinding/cosmos3/maniskill_default_preview10_20260605_103547`.
      Each video is `1024x1024`, `30 fps`, `301` frames, and `10.033333s`;
      source-frame coverage is `0..300`. Added local review index
      `review_index.html`. Evidence note:
      `docs/world_model_task_rebinding/2026-06-05_maniskill_default_preview10_and_archive.md`.
      The user then approved this preview. The old active full1000 RGB-D H5
      root was moved to
      `experiments/_archive/world_model_task_rebinding/20260605_wrong_cosmos_object_slot_wm/source_full1000_rgbd_env_states/`
      as an env-state source backup, and the active worklist
      `experiments/world_model_task_rebinding/cosmos3/full1000_env_state_source_h5s_archived_20260605.txt`
      contains exact `1000` source H5 paths. Do not reuse old preview videos.
      Regenerate the new full1000 SFT dataset with the approved view and then
      run Cosmos3 SFT/controller.
      Follow-up `2026-06-05T11:24+08:00`: started corrected full1000 render/SFT
      under
      `experiments/world_model_task_rebinding/cosmos3/sft_dataset_full1000_maniskill_default_20260605_1114`
      and
      `experiments/world_model_task_rebinding/cosmos3/sft_full1000_maniskill_default_20260605_1114`.
      Initial working allocation `107612` on `server13` produced the first
      complete videos, then was stopped at the render step only because it had
      insufficient time for full render plus SFT. One-day replacement
      allocation `107670` on `server13` was canceled after concrete
      GPU/Vulkan failure (`nvidia-smi` unknown device handle and
      `ErrorIncompatibleDriver`). Current active allocation is `107677` on
      `server43`, tmux `cosmos3_full1000_1h200_repl_1121`; `nvidia-smi` and
      a ManiSkill render preflight passed, and the full1000 render/SFT wrapper
      is running there. Evidence note:
      `docs/world_model_task_rebinding/2026-06-05_full1000_maniskill_default_render_sft_start.md`.
      Follow-up conditioning correction: Cosmos3 SFT must use video-prefix
      conditioning, not single-frame I2V. The active train/val config is
      `conditioning_config={8: 1.0}`, and the corrected JSONL records append
      hole, peg, TCP, grasp, insertion, and perturbation summaries to the
      caption/metadata so the foundation WM sees robot/object task-state
      grounding in addition to pixels. A lightweight OmegaConf check showed
      that command-line dotlist override syntax for this dict is unsafe, so
      the SFT wrapper now relies on the patched source config instead of
      passing `conditioning_config={8:1.0}` through Hydra CLI. Caption dropout
      is set to `0.0` on this active SFT path so robot/object state text is not
      randomly erased during training. Added pre-SFT dataset inspection
      `scripts/world_model/inspect_cosmos3_sft_dataset.py`; the current wrapper
      must pass exact-count, no-tmp, JSONL state-caption/metadata, video-prefix
      policy, and sampled readable 1024x1024/30fps checks before SFT starts.
      The old `run_cosmos3_i2v_rollout.py` single-image path is now protected
      behind an explicit diagnostic override and must not be used for active
      SFT/controller evidence. For post-SFT reconstruction/controller-facing
      inference, use Cosmos framework video conditioning with explicit
      `condition_frame_indexes_vision=[0,1,2,3,4,5,6,7]`; the framework default
      for video conditioning is only `[0,1]` and is too weak for this path.
      Follow-up `2026-06-05T13:17+08:00`: corrected full1000 render completed
      in allocation `107677` with exact `1000` final MP4s and `0` tmp MP4s.
      `cosmos3_sft_dataset_inspection.json` passed all checks: exact JSONL
      count, exact MP4 count, 1024x1024/30fps metadata, state caption/metadata,
      video-prefix policy, and sampled readable/nonblank videos. The same
      allocation entered the Cosmos3 SFT wrapper; wait for validation loss
      before interpreting the SFT result.
      First SFT attempt failed at initial validation because released Cosmos3
      `OmniMOTModel.validation_step` was an empty hook returning `None`. Logs
      are preserved under
      `sft_full1000_maniskill_default_20260605_1114/failed_validation_empty_hook_20260605_132041`.
      Patched validation to reuse `training_step` under `model.eval()` and
      `torch.no_grad()`, then restarted SFT in the same `107677` allocation at
      `sft_full1000_maniskill_default_valfix_20260605_1322`.
      The `valfix` restart failed in validation callback logging because
      `wandb_log_eval.py` asserted a batched `dataset_name` list length of one.
      Logs are preserved under
      `sft_full1000_maniskill_default_valfix_20260605_1322/failed_validation_dataset_name_assert_20260605_132624`.
      Patched callback logging to handle list-valued dataset names, then
      restarted SFT again in the same allocation at
      `sft_full1000_maniskill_default_valfix2_20260605_1327`.
      Follow-up `2026-06-05T13:34+08:00`: the `valfix2` SFT run is active.
      Initial validation succeeded with
      `Validation loss (iteration 0): 0.033790`, and training reached at least
      iteration `18` at about `6.1s/iter` after the startup validation. Code
      inspection confirms `conditioning_config={8: 1.0}` is consumed by
      `SFTDataset.process_one_sample()` as
      `SequencePlan(condition_frame_indexes_vision=list(range(num_cond)))`,
      so the training condition is a multi-frame video prefix, not a single
      I2V frame. JSONL inspection confirms each record carries
      `conditioning_policy=video_prefix_not_single_image_i2v` plus
      robot/object task-state caption and metadata for hole, peg, TCP,
      grasp/insertion, and perturbation.
      Follow-up post-SFT entrypoint correction: added
      `scripts/slurm/watch_cosmos3_sft_completion_v2v_eval.sh` for the next
      reconstruction diagnostic. It waits for the SFT completion marker,
      reuses the same running allocation, samples an approved full1000 MP4 as
      a condition video, writes `model_mode=video2video` with explicit
      `condition_frame_indexes_vision=[0,1,2,3,4,5,6,7]`, keeps official
      `30fps`, and evaluates both all-frame and future-only reconstruction
      error after the first `29` condition pixel frames. The old
      `watch_cosmos3_sft_completion_i2v_eval.sh` now refuses to run unless
      `ALLOW_SINGLE_IMAGE_I2V_DIAGNOSTIC=true`; it must not be used as active
      SFT/controller evidence.
      Follow-up `2026-06-05T13:46+08:00`: iteration `100` passed both
      checkpoint and validation. Checkpoint
      `iter_000000100` has model/optimizer/scheduler/trainer metadata and
      `latest_checkpoint.txt`; validation loss improved from iteration `0`
      `0.033790` to iteration `100` `0.026050`. The same tmux session now has
      a waiting `v2v_watch` window with output root
      `experiments/world_model_task_rebinding/cosmos3/v2v_eval_after_sft_full1000_maniskill_default_valfix2_20260605_134724`.
      It is waiting on `sft_completed` and has not started a GPU step while
      SFT is still running.
      Follow-up `2026-06-05T13:58+08:00`: iteration `200` also passed
      checkpoint and validation. Checkpoint `iter_000000200` saved
      successfully, and validation loss improved again to `0.018302`.
      Training continued past iteration `205` in the same allocation.
      Follow-up `2026-06-05T14:11+08:00`: iteration `300` checkpoint
      `iter_000000300` saved with model/optimizer/scheduler/trainer metadata.
      Validation loss was `0.020307`, a small increase from iteration `200`
      but still below iteration `100` and the initial validation. Continue to
      `MAX_ITER=500`; do not stop on this non-failing single-point fluctuation.
      Follow-up `2026-06-05T14:24+08:00`: iteration `400` checkpoint
      `iter_000000400` saved successfully. Validation loss was `0.023049`,
      higher than iterations `200` and `300` but still below iteration `100`
      and the initial validation. Training continued past iteration `408`;
      keep running to `500` and then let the waiting V2V diagnostic run.
      Follow-up `2026-06-05T14:40+08:00`: `valfix2` SFT completed inside the
      same allocation `107677` and wrote `sft_completed` at
      `2026-06-05T14:36:50+08:00`. Final checkpoint
      `iter_000000500` contains model, optimizer, scheduler, and trainer
      state. Validation losses were iteration `0`: `0.033790`, `100`:
      `0.026050`, `200`: `0.018302`, `300`: `0.020307`, `400`: `0.023049`,
      and `500`: `0.021407`. The waiting V2V diagnostic then ran in the same
      allocation with `model_mode=video2video`,
      `condition_frame_indexes_vision=[0,1,2,3,4,5,6,7]`,
      `condition_video_keep=first`, official `30fps`, and the robot/object
      state-grounded JSONL caption. It generated
      `experiments/world_model_task_rebinding/cosmos3/v2v_eval_after_sft_full1000_maniskill_default_valfix2_20260605_134724/inference/hole_constant_seed702000_n167_traj_0_traj_0_sft_full1000_v2v_prefix8/vision.mp4`
      as a valid `93`-frame, `30fps`, `256x256` clip. Reconstruction metrics:
      all frames mean MAE/RMSE/PSNR `0.0257569` / `0.0552557` / `27.5316`;
      future-only after condition frames mean MAE/RMSE/PSNR `0.0329914` /
      `0.0737908` / `23.2981` over `64` frames. Direct inspection of
      `reconstruction_all/reconstruction_comparison_sheet.png` and
      `reconstruction_future/reconstruction_comparison_sheet.png` shows the
      approved ManiSkill angled-overhead viewpoint, readable robot/table/peg/
      hole structure, nonblank output, and visible future contact/pose errors.
      This supports that the Cosmos3 video-prefix conditioning path is wired
      correctly enough to proceed to controller-facing integration; it is not
      controller evidence, not task success, and not permission to revert to
      single-image I2V or the lightweight object-slot WM.
      Follow-up action/state conditioning correction:
      the user explicitly rejected single-image I2V and weak caption-only
      grounding for Cosmos3 SFT. Added structured causal manipulation
      conditions under
      `experiments/world_model_task_rebinding/cosmos3/action_state_conditions_full1000_maniskill_default_20260605_145157`
      from the exact approved full1000 ManiSkill-default dataset. The manifest
      records exact `1000` samples, train/val split `912/88`,
      `domain_name=maniskill_peg_insertion`, `domain_id=21`,
      `raw_action_dim=32`, z-score normalization, vision prefix condition
      frames `[0,1,2,3,4,5,6,7]`, and action condition steps `0..91`. The 32D
      causal vector contains future recorded/candidate action commands plus
      prefix-observed TCP, peg, hole, peg-head-to-hole, hole velocity,
      grasp/insertion, perturbation, and time features; it must not contain
      future ground-truth object poses as privileged prediction targets.
      Patched the Cosmos3 SFT dataset loader to preserve `action_path` metadata,
      load and pad action/state arrays to `max_action_dim=64`, set
      `has_action=True`, and keep the eight-frame video-prefix plan. A loader
      smoke check on the active compute allocation returned
      `video_shape=(3,93,256,256)`, `action_shape=(92,64)`,
      `raw_action_dim=32`, and `domain_id=21`.
      Follow-up user reminder: future Cosmos3 SFT must keep this conditioning
      form. Single-image I2V, video-only V2V, or caption-only SFT is too weak
      for manipulation prediction and is not method evidence. When action/env
      state exists, method SFT must use a video prefix plus structured robot
      action, TCP/hand, peg, hole, and task-frame relative-geometry inputs; if
      those inputs are missing, recover/generate them before method SFT rather
      than treating the weak run as publishable evidence.
      Follow-up `2026-06-05T15:05+08:00`: started action-conditioned Cosmos3
      SFT in the existing allocation `107677` on `server43`, tmux
      `cosmos3_full1000_1h200_repl_1121`, without `sbatch` or releasing the
      GPU. Output root:
      `experiments/world_model_task_rebinding/cosmos3/sft_full1000_maniskill_default_actioncond_20260605_150550`.
      It resumes model weights from the previous video-prefix SFT checkpoint
      `iter_000000500`, uses
      `ACTION_CONDITIONED_SFT=true`, `MAX_ITER=1800`,
      `VALIDATION_ITER=300`, `SAVE_ITER=300`, `MAX_VAL_ITER=20`,
      the 912/88 action JSONLs, `num_video_frames=93`,
      `temporal_interval_mode=force_one`, and
      `conditioning_config={8:1.0}`. This run is the active foundation-WM
      conditioning path for controller-facing integration; it is still SFT
      component evidence only until followed by reconstruction/controller
      diagnostics with inspected videos and metrics.
      Follow-up `2026-06-05T15:11+08:00`: action-conditioned SFT passed the
      startup validation and first training steps. Validation loss at
      iteration `0` was `0.022813`; iteration `1`, `2`, and `3` logged losses
      `0.0130`, `0.0163`, and `0.0230`. The first training step took
      `245.97s` because it included startup validation/cache overhead; the
      next two steps were about `7.2s/iter`. Keep the allocation and continue
      to the first checkpoint/validation milestone at iteration `300`.
      Follow-up `2026-06-05T15:17+08:00`: the run passed iteration `50`;
      iter-speed callback at iterations `52..55` reported `7.21..7.24s/iter`
      with losses `0.0232`, `0.0165`, `0.0190`, and `0.0143`. No NaN, OOM,
      dataloader failure, callback failure, or allocation release occurred.
      Follow-up `2026-06-05T15:23+08:00`: reached iteration `101`; iteration
      `100` logged `7.21s/iter` and loss `0.0196`, and iteration `101` logged
      `7.44s/iter` after the norm-monitor callback. Continue toward iteration
      `300`.
      Follow-up `2026-06-05T15:35+08:00`: reached iteration `203`;
      iteration `200` logged `7.22s/iter` and loss `0.0333`. Device monitor at
      iteration `200` showed GPU utilization `100%`, peak GPU memory
      `79.512590GB`, NVML used GPU memory `84.464661GB`, and free GPU memory
      `55.936707GB`. Continue to iteration `300`.
      Follow-up `2026-06-05T15:52+08:00`: action-conditioned SFT passed the
      first checkpoint/validation milestone. Checkpoint `iter_000000300` saved
      successfully with model, optimizer, scheduler, and trainer state; save
      time was `37.90s`. Validation loss at iteration `300` was `0.017178`,
      improved from action-conditioned iteration `0` validation loss
      `0.022813`. Training resumed afterward and reached at least iteration
      `315` at about `7.24s/iter`. Continue the active run toward
      `MAX_ITER=1800`; meanwhile prepare an action-conditioned diagnostic that
      uses video-prefix plus action/state arrays, not video-only V2V or
      single-image I2V.
      Follow-up action-inference wiring: patched Cosmos3 action inference so
      `forward_dynamics` samples can carry explicit
      `condition_frame_indexes_vision`. The default released action builder
      conditions only the first latent vision frame (`[0]`), which is too weak
      for this manipulation path. The patched path preserves official
      forward-dynamics action conditioning (`condition_frame_indexes_action`
      `0..91`) while allowing the active sample JSON to set the same eight
      vision-prefix latent frames used in SFT (`[0..7]`). A CPU batch-builder
      smoke check returned `has_action=True`,
      `condition_frame_indexes_vision=[0,1,2,3,4,5,6,7]`,
      `condition_frame_indexes_action=0..91`, action shape `(92,64)`, and
      `domain_id=21`.
      Added
      `scripts/slurm/watch_cosmos3_sft_completion_action_eval.sh`, a post-SFT
      watcher that waits for `sft_completed`, reuses the same running
      allocation, loads the latest action-conditioned checkpoint, and runs
      `model_mode=forward_dynamics` with the approved video-prefix MP4 plus
      the structured action/state JSON. It then computes all-frame and
      future-only reconstruction metrics and sheets. Started this watcher in
      tmux window `action_eval_watch` at output root
      `experiments/world_model_task_rebinding/cosmos3/action_eval_after_sft_full1000_maniskill_default_20260605_1559`.
      It is currently only polling the completion marker and has not launched
      a GPU step while SFT is still training.
      Follow-up `2026-06-05T16:06+08:00`: action-conditioned SFT reached at
      least iteration `428`. Iteration `400` logged `7.26s/iter` and loss
      `0.0208`; device monitor showed GPU utilization `100%`, peak GPU memory
      `79.511139GB`, NVML used GPU memory `87.300598GB`, and free GPU memory
      `53.100769GB`. Continue to the next checkpoint/validation milestone at
      iteration `600`.
      Follow-up `2026-06-05T16:32+08:00`: action-conditioned SFT passed the
      iteration `600` checkpoint/validation milestone. Checkpoint
      `iter_000000600` saved successfully with model, optimizer, scheduler,
      and trainer state; save time was `28.06s`. Validation loss improved to
      `0.013811` from iteration `300` validation loss `0.017178` and
      iteration `0` validation loss `0.022813`. Training resumed afterward and
      reached at least iteration `614` at about `7.24s/iter`. Continue toward
      `900`, then `1200`, `1500`, and `1800`.
      Follow-up `2026-06-05T16:46+08:00`: the post-SFT action eval watcher was
      restarted in the same tmux session and same output root after enabling
      Cosmos inference `--debug` output. This preserves input/output tensors
      for future controller-facing readout/debugging while keeping the same
      action-conditioned forward-dynamics sample, checkpoint wait, metrics,
      and non-controller-evidence boundary. The restart did not touch or stop
      the active training step. The SFT reached at least iteration `731`, with
      latest checkpoint still `iter_000000600`.
      Follow-up `2026-06-05T16:49+08:00`: patched Cosmos inference debug mode
      to save `vision_latent` alongside the decoded vision output tensor.
      This is controller/readout artifact preservation only; it does not alter
      sampling, conditioning, reconstruction metrics, or any evaluation
      protocol. `py_compile` passed for the patched inference/action modules,
      and `bash -n` passed for the action-eval watcher.
      Added
      `scripts/world_model/inspect_cosmos3_action_eval_artifacts.py` to inspect
      the action eval output root after completion: prediction MP4 metadata,
      all/future reconstruction summaries, sample manifests, debug tensor
      files, and saved `vision_latent` shapes. A pre-completion dry run
      correctly reports no completed marker/video yet. This inspector is
      artifact readiness/debugging only, not a controller gate or task success
      metric.
      Follow-up `2026-06-05T17:10+08:00`: action-conditioned SFT passed the
      iteration `900` checkpoint/validation milestone. Checkpoint
      `iter_000000900` saved successfully; save time was `27.93s`.
      Validation loss was `0.018228`, worse than iteration `600` `0.013811`
      but still better than the action-conditioned initial validation
      `0.022813`. This single validation fluctuation is not a stop condition
      or a gate; training resumed and reached at least iteration `910`.
      Continue to `1200`, `1500`, and `1800`, then run the waiting
      action-conditioned forward-dynamics diagnostic.
      Follow-up `2026-06-05T17:44+08:00`: after the earlier scheduler-cycle
      implementation failure at the 1000-step boundary, the run was restarted
      in the same allocation from `iter_000000900` with `MAX_ITER=1800` and
      `scheduler.cycle_lengths=[1800]`. The resumed training crossed the real
      failure boundary: logs show iteration `1000` at `7.22s/iter`, device
      monitor output at iteration `1000`, and continued iterations `1001`
      through at least `1007` without a scheduler exception. Slurm step
      `107677.19` remains active on `server43`; continue toward `1200`,
      `1500`, and `1800` without releasing the allocation.
      Follow-up `2026-06-05T17:50+08:00`: tightened the waiting action-eval
      watcher prompt. The structured `action_path` condition is causal
      prefix-state plus future action commands, but the inherited JSONL
      captions include full-trajectory start/end summaries. The watcher no
      longer copies that caption into the inference prompt; it now gives a
      causal prompt that refers to the eight-frame video prefix and structured
      robot/object/action condition only. Restarted tmux window
      `action_eval_watch` in the same allocation/output root. It is again
      waiting for `sft_completed`, does not use GPU while SFT is running, and
      will not run single-image I2V or video-only V2V.
      Follow-up `2026-06-05T18:13+08:00`: action-conditioned SFT passed the
      iteration `1200` checkpoint/validation milestone. Checkpoint
      `iter_000001200` saved successfully with model, optimizer, scheduler,
      and trainer metadata; save time was `40.89s`. Validation loss at
      iteration `1200` was `0.016808`, better than iteration `900`
      (`0.018228`) and the initial action-conditioned validation (`0.022813`),
      but worse than iteration `600` (`0.013811`). This remains a recorded
      validation fluctuation, not a stop condition; training resumed and
      reached at least iteration `1217` in the same Slurm step `107677.19`.
      Continue to `1500` and `1800`.
      Follow-up `2026-06-05T18:30+08:00`: patched the waiting post-SFT
      action-eval watcher to default to `CHECKPOINT_SELECTION=best_val`
      instead of blindly using the latest checkpoint. The selector parses all
      active SFT logs and only considers validation rows whose corresponding
      DCP checkpoint has `model/.metadata`; a dry run currently selects
      `iter_000000600` because it has the best existing validation loss
      (`0.013811`). If `1500` or `1800` improves, the same selector will use
      that better checkpoint. Restarted only tmux window `action_eval_watch`
      in the same allocation/output root; the `action_sft` training window and
      Slurm allocation `107677` were not stopped or released.
      Follow-up `2026-06-05T18:53+08:00`: action-conditioned SFT passed
      iteration `1500` checkpoint/validation. Checkpoint `iter_000001500`
      saved successfully; save time was `29.05s`. Validation loss was
      `0.015549`, better than iteration `1200` (`0.016808`) and `900`
      (`0.018228`) but still worse than iteration `600` (`0.013811`).
      The best-val selector therefore still selects `iter_000000600` unless
      iteration `1800` improves. Training resumed afterward and reached at
      least iteration `1523` in the same Slurm step `107677.19`; continue to
      `1800` and then let action-conditioned forward-dynamics eval run.
      Follow-up `2026-06-05T18:55+08:00`: patched the action-eval watcher to
      run `inspect_cosmos3_action_eval_artifacts.py` automatically after the
      reconstruction metrics complete, writing `artifact_inspection.json/md`
      and checking prediction video metadata, all/future PSNR summaries, debug
      tensor keys/shapes, and saved `vision_latent`. Restarted only
      `action_eval_watch`; the running SFT step and allocation were untouched.
- [ ] 2026-06-05 readable video-review directive:
      Future controller/video review sheets should use large readable frames
      (`thumb_width=480` unless explicitly overridden) so the dynamic event,
      robot reaction, contact state, and final task state can be inspected
      directly. This changes only human-readable review artifacts; it must not
      change evaluation metrics, scenario definitions, RGB-D inputs, world-model
      inputs, or the 10-demo visualization budget. The controller run-group
      summary should include both controller inspection metrics and
      `video_review/video_artifact_inspection.json` review-sheet paths so the
      agent can open the large visual artifacts before making any dynamic
      manipulation claim.
- [ ] 2026-06-04 foundation world-model selection correction:
      The current in-repo Transformer world model is an RGB-D-derived
      object-slot interface baseline, not a publishable final world-model
      choice. Keep the running exact1000 jobs as diagnostic/controller
      scaffolding because they validate the data boundary and downstream
      interface, but do not present them as the main method. The preferred
      serious backbone is a Cosmos-family physical-AI world foundation model
      because it is designed for future physical-world prediction and
      post-training. Wan-family video models may be used as visual
      prediction/augmentation baselines, but they are not sufficient as the
      main controller world model unless action conditioning, RGB-D/robot-state
      grounding, task-slot decoding, uncertainty, and closed-loop controller
      benefit are all proven under the unchanged dynamic task-completion
      evaluation. Evidence note:
      `docs/world_model_task_rebinding/2026-06-04_world_model_choice_and_code_structure.md`.
- [ ] 2026-06-04 23:10 predicted-slot quality advisory correction and tmux
      world-model handoff:
      User corrected the prior hard-blocking predicted-slot RMSE gate. The
      exact1000 export
      `experiments/world_model_task_rebinding/rgbd_predicted_slots/from_rgbd_job102034_slot_job102292/job102294`
      has `1000` predicted-slot H5s and `301000` samples with RGB-D-derived
      `slots/*`; its old strict inspection failed only on advisory geometry
      diagnostics (`hole_pos_rmse_m=0.031467` and
      `peg_head_hole_rmse_m=0.054956`). Those numbers must remain recorded and
      used for diagnosis/ranking, but they must not block RGB-D-derived
      world-model/controller progress by themselves. Canceled the old strict
      slot/dependency branch and mistaken ordinary sbatch attempts
      (`105236`, `105237`, `105238`, `105239`, `105240`, `105259`,
      `105316`, `105553-105571`, `106245`, `106402`, `106406`) and stopped
      old slot-backup steps inside reusable allocations. Repurposed the live
      tmux H200 allocations directly for full1000 RGB-D-derived world-model
      training from export `102294`, without submitting a new ordinary sbatch:
      output root
      `experiments/world_model_task_rebinding/rgbd_object_world_model/ensemble_4gpu/from_export_job102294/tmux_reuse_20260604_2303`,
      sessions `h200_1gpu_pool`, `h200_1gpu_pool3`, `h200_1gpu_pool8`,
      `h200_1gpu_pool9`, `h200_1gpu_pool10`, `h200_1gpu_pool11`,
      `h200_1gpu_pool12`, `h200_1gpu_pool13`, and `h200_1gpu_pool14`,
      seeds `800/900/1000/1100/1200/1300/1400/1500/1600`. Each started with
      exact `1000` predicted-slot files, `HISTORY=8`, `EPOCHS=50`,
      `BATCH_SIZE=512`, `D_MODEL=512`, `N_LAYERS=6`, `N_HEADS=8`,
      `HORIZONS=1 5 10 20 40`, `REQUIRE_CUDA=true`, and
      `oracle_slots_not_used=true`. Current status at launch: all nine Slurm
      steps are running inside tmux allocations and have written wrapper
      manifests; no world-model metrics/checkpoints yet. This is aligned
      progress, not method evidence. Next required action is to monitor/fix
      those tmux runs, then assemble/inspect the RGB-D-derived world-model
      ensemble and run controller metrics plus at most 10 diagnostic videos.
      Follow-up `2026-06-04T23:17+08:00`: the first launch exposed an
      implementation bottleneck in `object_slot_dataset.py` where target HDF5
      arrays were re-read for every sequence sample. Patched the loader to
      read targets once per trajectory, added explicit dataset load events,
      canceled only the old srun steps (`105385.14`, `105535.6`, `105867.3`,
      `106000.3`, `106001.3`, `106147.3`, `106237.3`, `106359.3`,
      `106360.3`) while keeping the tmux allocations alive, and restarted all
      nine runs under
      `rgbd_object_world_model/ensemble_4gpu/from_export_job102294/tmux_reuse_20260604_2317_fastload`.
      New steps are `105385.16`, `105535.8`, `105867.5`, `106000.5`,
      `106001.5`, `106147.5`, `106237.5`, `106359.5`, and `106360.5`.
      Each loaded exact1000 RGB-D-derived predicted slots in about `54-58`
      seconds, produced `254000` sequence samples with
      `rgbd_predicted_slot_input_evidence=true`, wrote `manifest.json` and
      `best_model.pt`, and reached at least epoch `1`; several reached epoch
      `10`. They remain running toward the `10800` second training floor. This
      is world-model training progress only, not controller or method success.
      Follow-up `2026-06-04T23:21+08:00`: Slurm revoked allocation `105385`
      (`CANCELLED by 0`) after `04:03:00`, killing fastload step `105385.16`
      after `00:04:36`. This was not an agent release and not method evidence;
      seed `800` is incomplete and must not be assembled as a compliant member.
      The remaining eight fastload steps `105535.8`, `105867.5`, `106000.5`,
      `106001.5`, `106147.5`, `106237.5`, `106359.5`, and `106360.5` remain
      running and are the current world-model ensemble candidates.
      Follow-up `2026-06-04T23:28+08:00`: rechecked the queue after the user
      correction to stop stale tasks and use the tmux cards directly. `squeue`
      shows only the eight intended `wm_rgbd_1h200_pool{3,8,9,10,11,12,13,14}`
      allocations; no old strict-gated or mistaken ordinary sbatch tasks remain
      queued/running. The stale tmux shells `h200_1gpu_pool` and
      `h200_1gpu_pool6` were just revoked-allocation leftovers, so they were
      killed to avoid accidental reuse. The controller-in-allocation launcher
      passed `bash -n`; downstream controller/video must be launched from one
      of the eight live tmux allocations after world-model assembly/inspection,
      not by ordinary sbatch.
      Follow-up `2026-06-04T23:31+08:00`: added and launched detached tmux
      watcher `rgbd_wm_controller_watch`, running
      `scripts/slurm/watch_rgbd_wm_assemble_and_controller_from_allocations.sh`.
      The watcher does not submit sbatch. It polls the eight exact1000
      RGB-D-derived WM member dirs until all have final `metrics.json`,
      `model.pt`, and `complete=` manifest entries, symlink-assembles them
      into
      `rgbd_object_world_model/ensemble_4gpu/from_export_job102294/tmux_reuse_20260604_2317_fastload_assembled8`,
      runs `inspect_world_model_ensemble.py`, then launches at most five
      RGB-D controller/video branches through `srun --jobid` on a still-live
      allocation. Updated the watcher and restarted it at
      `2026-06-04T23:32+08:00` so each completed controller branch also runs
      local `inspect_rebinding_controller_run.py` and
      `inspect_video_artifacts.py` to create controller inspection JSON/MD and
      readable nonblank video review sheets without CPU sbatch. Current
      watcher log:
      `logs/slurm/rgbd_wm_controller_watch_20260604_233245.log`, initial
      status `world_model_members_complete=0/8`.
      Follow-up `2026-06-04T23:34+08:00`: live monitor still shows exactly
      the eight intended H200 allocations running and no extra queued/running
      ordinary sbatch jobs. The eight fastload steps are alive with latest
      visible epochs around `90-105`, validation
      `val_mean_hole_delta_rmse_m` around `0.019-0.020`, and active CPU/RSS
      in `sstat`. Final compliant artifacts are still absent (`metrics.json`
      and `model.pt` missing) because the runs have not reached the `10800`
      second training floor. This is continued RGB-D-derived world-model
      training progress only, not controller/video or method evidence.
- [ ] 2026-06-03 19:38 full-scale RGB-D data correction:
      exact96/full96 and full96-plus-online5 are small-scope validation only.
      Full-scale method experiments must wait for exact `1000` synchronized
      RGB-D demos generated from dynamic state rollouts and passing structural
      plus visual gates. Added
      `scripts/slurm/collect_dynamic_state_rollouts_specs_4gpu.sbatch` so the
      dynamic state source can be exactly `1000` episodes with per-scenario
      counts (`166/167`) instead of the old uniform `n16` shard. Patched
      `submit_rgbd_distributed_shards.sh` with `RENDER_DEPENDENCY` and optional
      visual review so render shards wait for state generation and then run
      exact-count inspection and nonblank visual review. Submitted state source
      job `101832` (`4` H200, `EXPECTED_TOTAL_EPISODES=1000`,
      `ExcNodeList=server13` due current slot-training straggler evidence) and
      dependent RGB-D render shards `101833-101840` (`8` shards, each
      `1 node x 4 GPU`, exact `1000`, render exclusions tied to recent
      DeviceLost/straggler evidence), structural inspection `101841`, and
      visual review `101842`. This is data generation only; no full-scale
      method claim is allowed until `101841/101842` pass and the agent opens
      the resulting visual contact sheet. Because this correction makes the
      full96-plus-online5 chain small-scope validation only, canceled running
      small-chain slot job `101697` after `00:20:47` with no
      `rgbd_slot_dataset_loaded` event, and canceled its downstream
      `101698-101720` before allocation to free the 4H200 allocation for the
      full1000 data path. This cancellation is prioritization/resource hygiene,
      not method evidence. State source job `101832` later ran on `server04`
      from `2026-06-03T19:47:29` to `21:28:19`, completed `0:0`, and direct
      HDF5 reading verified exact per-scenario counts
      `166/167/167/167/166/167`, total `1000`. This completes only the state
      source gate; RGB-D data are still incomplete until render shards
      `101833-101840`, structural inspection `101841`, and visual review
      `101842` pass. Render shard `101833` on `server04` then produced a
      concrete SAPIEN/Vulkan `ErrorDeviceLost` failure (`0/125` successful H5,
      `14` failed units before cancellation), so it was canceled and replaced
      by disjoint shard0 job `102001` (`1 node x 2 GPU`, excluding
      `server04` plus existing render-risk nodes). Old gates `101841/101842`
      were canceled before allocation and replaced by exact structural gate
      `102002` and visual gate `102003`, dependent on completed/active render
      shards `101834-101840` plus `102001`. Full RGB-D data are still
      incomplete until `102001/102002/102003` pass and the contact sheet is
      opened. Render shard `101840` then failed partially on `server53`
      (`63/125` valid H5, `62` failed units, `ErrorDeviceLost`). Generated a
      failed62 repair worklist with zero overlap against successful shard7
      units, patched `render_dynamic_rgbd_dataset_dense.sbatch` so dense repair
      jobs refuse failed/incomplete unit ledgers, submitted shard7 repair
      `102032` excluding `server53/server04` plus prior render-risk nodes, and
      replaced old gates `102002/102003` with `102033/102034`. Full RGB-D data
      are still incomplete until `102001`, `102032`, `102033`, and `102034`
      pass and the contact sheet is opened. `102001` completed `125/125`,
      `102032` completed `62/62`, structural inspection `102033` passed exact
      `1000` RGB-D files with `301000` frames and `0` warnings, visual review
      `102034` passed `24` sampled files x `3` frames x `2` cameras with `0`
      warnings, and the contact sheet was opened directly and found nonblank
      with visible robot/table/peg/hole content. A local unit audit found
      expected `1000`, actual `1000`, missing `0`, extra `0`, duplicates `0`.
      This is full1000 RGB-D data evidence only, not method success. Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_full1000_rgbd_generation_submission.md`.
- [ ] 2026-06-03 23:17 full1000 RGB-D method-chain startup:
      After full1000 structural/visual gates passed, submitted
      `full1000_coordconv_rgbd_method_20260603_231525` with exact `1000`
      RGB-D inputs, 4H200/`MIN_TRAIN_SECONDS=10800` slot training,
      RGB-D predicted-slot export, RGB-D-derived world-model training, and
      five `SLOT_SOURCE=rgbd` controller/video branches. Slot job `102144`
      failed before training with `srun: Argument list too long` because the
      wrapper exported all `1000` long H5 paths into the `srun` environment.
      Classified this as an implementation wrapper failure, not data,
      perception, world-model, controller, physics, or evaluation evidence.
      Patched slot and world-model training wrappers to pass path-list files
      into member tasks, added stream-per-file predicted-slot export for
      full1000, bounded the lazy H5 cache, and fixed contract audit recursive
      file counting for distributed RGB-D roots. Canceled dead dependency
      branch `102145-102168`. Submitted replacement chain
      `full1000_coordconv_rgbd_method_pathfilefix_20260603_231732` with jobs
      `102169-102193`; contract audit passed `206/206`. Slot training
      `102169` started on `server34` with `1000` RGB-D paths,
      `LAZY_IMAGES=true`, `EPOCHS=1`, 4 H200 GPUs, and the same 3h training
      floor. It then loaded exact full1000 metadata (`301000` samples,
      `[301000, 8, 128, 128]` RGB-D shape) in about `591` seconds and all
      four members emitted epoch-0 batch-0 training logs. This is
      startup/readiness evidence only; no full1000 RGB-D method evidence
      exists until slot inspection, predicted-slot quality, RGB-D-derived
      world model, controller metrics, and direct video/contact review
      complete. By batch `100`, `102169` was progressing but the 4h walltime
      looked tight for a full epoch plus checkpoint; direct `scontrol` walltime
      extension was denied. Submitted non-concurrent backup chain
      `full1000_coordconv_rgbd_method_slot6h_afternotok102169_20260603_233525`
      with slot backup `102292` depending on `afternotok:102169` and
      `SLOT_TRAIN_TIME=06:00:00`; audit passed `206/206`, and Slurm shows it
      pending on dependency, not consuming GPUs. If `102169` succeeds, cancel
      backup branch `102292-102316` as unneeded. Downstream preflight found
      predicted-slot export still expanded all RGB-D paths as Python argv, the
      same implementation-risk class as the earlier `srun` argument-length
      failure. Patched `export_rgbd_predicted_slots.py` and
      `export_rgbd_predicted_slots.sbatch` to use `--rgbd-paths-file` and the
      already-written input path file; `py_compile`, `--help`, `bash -n`, and
      contract audit passed. This is argument-transport hygiene only, not a
      data/gate/evaluation change. At `2026-06-04T01:36:54+08:00`, `102169`
      remained running on `server34` at about `02:18:44/04:00:00` with all
      four members at epoch-0 batch `1750`, but no epoch/checkpoint artifact
      yet; the long epoch-end gap is consistent with full1000 validation and
      increases walltime risk. Updated pending backup slot job `102292` from
      `06:00:00` to `08:00:00`, still on
      `afternotok:102169(unfulfilled)` and not consuming GPUs. This is
      scheduling slack only for the same exact `1000` RGB-D chain, not a
      dataset, gate, metric, or method change. Evidence note:
      `docs/world_model_task_rebinding/2026-06-03_full1000_rgbd_generation_submission.md`.
- [ ] 2026-06-03 18:30 RGB-D controller online slot failure:
      compliant RGB-D slot extraction/export (`100730/101009/101010`),
      RGB-D-derived world-model training/inspection/eval (`101012/101013/101014`),
      and five RGB-D controller video branches (`101015`, `101018`, `101021`,
      `101024`, `101027`) completed, but final review `101031` failed because
      every branch had `success_after_dynamic_event_count=0`. Direct video
      review confirmed no final insertion. H5 analysis localizes the failure
      to online RGB-D controller slots: peg/TCP and peg-head-in-hole drift far
      from `metric_slots` after perturbation while hole position is better
      preserved. The online RGB preprocessing already matches training/export
      (`/255.0`). Default-off controller diagnostics saved raw control-time
      RGB-D observations and controller env states for jobs `101580`, `101602`,
      `101603`, `101604`, and replacement `101618`; `101601` failed with
      SAPIEN/Vulkan `ErrorDeviceLost` on `server60` and is scheduling/rendering
      evidence only. Direct contact-sheet inspection showed valid/nonblank
      online RGB-D frames but large post-perturb peg/TCP slot errors, so the
      aligned fix is perception coverage for controller-time bridge states, not
      gate changes or oracle fallback. Converted the five diagnostics into a
      standard RGB-D slot-training dataset (`5` files, `1505` frames, structural
      warnings `0`) and combined it with full96 into exact `101` RGB-D H5s.
      Augmented 4H200 slot training `101635` started on `server13`, but live
      monitoring found `member_3` far slower than members `0/1/2` after all
      four loaded the exact `101` H5s. The job and its downstream main chain
      (`101638`, `101641-101646`, `101652-101667`) were canceled to avoid a
      likely timeout/dead tail; `sacct` records `101635` as
      `CANCELLED by 2059`, elapsed `00:29:18`. This is scheduling/device-risk
      evidence, not perception quality or method evidence. Current active path
      is backup chain `101697-101720`, excluding `server13` for slot/world-model
      training, with the same data, gates, 4H200/3h requirements, controller
      settings, and video review; its contract audit passed `138/138`.
      At `2026-06-03T19:13+08:00`, backup job `101697` was pending with
      forecast `2026-06-03T19:54:28` on `server34`. It later started on
      `server04` but produced only `rgbd_slot_train_start` events and no
      `rgbd_slot_dataset_loaded` event after `00:20:47`. After the full1000
      correction, `101697` and downstream `101698-101720` were canceled as
      small-scope validation/resource hygiene, not method evidence. Evidence
      note:
      `docs/world_model_task_rebinding/2026-06-03_rgbd_controller_online_slot_failure_1830.md`.
- [x] 2026-06-02 22:35 state/resource correction:
      fresh Slurm probes show current-account schedulability, not visible
      mixed-node count, is the limiting factor. `gaosh`, `engram`, and `test`
      reject account `mayi`; `gpux`/`mgpu` are drained; new `cpu` 1/2 GPU
      probes start on `2026-06-04`, later than the active RGB-D render path.
      A CPU-only debug controller smoke job `99136` failed with SAPIEN/Vulkan
      `ErrorIncompatibleDriver`, so CPU-only simulator rollout is a
      rendering/scheduling failure path, not controller evidence. Existing
      state evidence was summarized at
      `experiments/world_model_task_rebinding/rebinding_controller/state_current_summary_20260602_2235.md`;
      evidence note:
      `docs/world_model_task_rebinding/2026-06-02_state_resource_probe_and_current_summary_2235.md`.
      Next queue/artifact check should be no earlier than about
      `2026-06-02T23:00+08:00` unless a job starts or artifacts appear first.
- [x] 2026-06-02 22:35 agent-rule and RGB-D chain preflight:
      `AGENTS.md` now explicitly records the approximately 30-minute
      queue/artifact check cadence and the CPU-only ManiSkill/SAPIEN rollout
      boundary from failed debug job `99136`. Static preflight passed for the
      active full96 aggregate chain wrappers and Python scripts. The current
      method-chain manifest still requires exact96 RGB-D data, visual no-blank
      no-warning gate `98907`, 4H200/3h RGB-D slot training `98848`, exact96
      RGB-D predicted-slot export/inspection `98911/98912`, RGB-D-derived
      world-model training/inspection/eval `98914/98915/98916`, and
      `SLOT_SOURCE=rgbd` controller video/inspection/review
      `98917/98918/98919`. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_chain_preflight_and_agent_rules_2235.md`.
- [x] 2026-06-02 22:40 controller config correction before RGB-D branch runs:
      state-axis diagnostics found that the old queued RGB-D controller config
      (`tcp_continuation + peg_alignment + insert_guard`) could inherit an
      insertion-axis progress failure: seed-7400 state runs were laterally
      close but either stalled near `x=-0.1005` or blocked insert axis steps.
      Updated future RGB-D controller defaults/submitter to
      `BRIDGE_SERVO_REFERENCE=phase_hybrid`,
      `BRIDGE_ORIENTATION_REFERENCE=tcp_continuation`, and
      `TASK_SERVO_INSERT_MANIFOLD_GUARD=false`, preserving the same
      final-state metric and RGB-D gates. Canceled old pending controller/video
      jobs `98917/98918/98919` before start and replaced them with
      `99170/99171/99172` after `98916`. This is a controller-configuration
      fix, not method evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_state_axis_progress_and_rgbd_controller_config_2240.md`.
- [x] 2026-06-02 22:48 resource probe and focused state smoke:
      fresh `sbatch --test-only` probes still show no immediately allocatable
      GPU for account `mayi`: `cpu` 1/2 GPU forecasts
      `2026-06-04T00:54:08`, `cpu` 4 GPU/H200 forecasts
      `2026-06-04T02:55:53`, `gpu` forecasts `2026-06-06`, `debug` 1 GPU is
      later and 2/4 GPU is blocked by `MaxGRESPerAccount`, `gpux` is inactive
      or drained, and `gaosh`/`engram`/`test` are invalid account/partition
      combinations. Submitted two small state/oracle scaffold smoke chains to
      test the physical insertion-manifold question on move-stop seed `7400`,
      without changing the final-state gate or RGB-D requirements:
      CV `99189 -> 99190 -> 99191` and learned-WM `99192 -> 99193 -> 99194`
      under
      `experiments/world_model_task_rebinding/rebinding_controller/state_smoke_move_stop_pegalign_guard_seed7400_20260602_2248`.
      They use `phase_hybrid + task_frame_projected + peg_alignment +
      insert_manifold_guard`. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_resource_probe_and_state_smoke_submission_2248.md`.
- [x] 2026-06-02 23:16 RGB-D render progress, repair, and state-smoke result:
      front-half RGB-D retry jobs `98524-98527` completed on `server42` with
      exact `48/48` RGB-D H5/contact sheets, `0` failed units, structural
      gate `98528` passed with `num_warnings=0`, and visual gate `98530`
      passed with `valid_visual_artifacts=true`. Representative contact
      sheets were opened directly and were nonblank. Late-half jobs
      `98841`, `98843`, and `98844` completed, but `98842` on `server10`
      hit repeated Vulkan `ErrorDeviceLost` on shard5/traj5 task0 units and
      was canceled after task1 completed. Submitted exact six-unit repair
      `99208` with job-local `ExcNodeList=server10`, replacement aggregate
      `99209`, exact96 structural/visual gates `99210/99211`, and repaired
      RGB-D method chain `99212-99222`. Old dead branch `98909/98910/98907`,
      `98848/98849`, `98911-98916`, and `99170-99172` was canceled. The
      focused state/oracle smoke also completed: CV `99189->99190->99191`
      and learned-WM `99192->99193->99194` both passed no-video final dynamic
      gates on seed `7400`, showing the state failure was insertion-manifold
      control and not learned-WM advantage. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_front48_repair_and_state_smoke_2316.md`.
- [x] 2026-06-02 23:21 RGB-D controller tail replacement before allocation:
      because state/oracle seed-7400 smokes `99189` and `99192` both showed
      `peg_alignment + insert_manifold_guard` resolves the insertion-axis
      stall without changing the final-state gate, updated future RGB-D
      controller defaults/submitter to `BRIDGE_ORIENTATION_REFERENCE=peg_alignment`
      and `TASK_SERVO_INSERT_MANIFOLD_GUARD=true`. Canceled old pending
      controller/video tail `99220/99221/99222` with zero allocation and
      replaced it with `99252 -> 99253/99254` after the same RGB-D-derived
      world-model eval dependency `99219`. Data gates, slot training,
      predicted-slot export, RGB-D-derived world-model training/eval,
      final-state metric, and video artifact review are unchanged. Evidence
      note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_controller_tail_pegalign_replacement_2321.md`.
- [x] 2026-06-02 23:30 resource probe and compact state matrix:
      fresh non-submitting probes still found no earlier legal GPU path than
      active repair job `99208`, which is now priority-pending with forecast
      `2026-06-03T04:00:00`. New `cpu` 1/2 GPU and 4H200 probes forecast
      `2026-06-04T00:23`, `gpu` forecasts `2026-06-05T06:09`, `debug`
      1 GPU is later and larger debug allocations are blocked by
      `MaxGRESPerAccount`, `gpux` is inactive/drained, and
      `test`/`gaosh`/`engram` reject the current account/partition
      combination. Therefore no duplicate RGB-D repair/render was submitted.
      Added a single 1-GPU serial state/oracle smoke matrix wrapper and
      submitted fixed-path job `99308` under
      `state_smoke_generalization_pegalign_guard_20260602_2330` to test
      continuous hole motion, reverse motion, and peg event/regrasp cases
      using the seed-7400-supported `phase_hybrid + task_frame_projected +
      peg_alignment + insert_manifold_guard` controller. Initial job `99307`
      was canceled before allocation only to fix ambiguous runtime-timestamp
      output naming. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_resource_probe_and_state_matrix_2330.md`.
- [x] 2026-06-02 23:36 repair failure classification and remaining3 chain:
      repair job `99208` actually started on `server28` and failed with
      SAPIEN/Vulkan `ErrorDeviceLost` on three task0 units
      (`hole_constant`, `hole_reverse`, `peg_disturb` traj5), while producing
      three valid RGB-D repair files (`hole_move_stop`, `none`, `peg_drop`).
      Canceled dead downstream branch `99209-99219` and `99252-99254`.
      Added and submitted remaining-three 1GPU repair `99316` with job-local
      `ExcNodeList=server10,server28`, then submitted replacement exact96
      chain `99317 -> 99318 -> 99319` and RGB-D method chain `99320-99330`.
      State matrix `99308` was canceled before allocation so scaffold work
      would not run before the RGB-D blocker; it was requeued as low-priority
      job `99331` after visual gate `99319`. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-02 23:41 aggregate implementation bug preflight:
      existing RGB-D sources before remaining repair contain `93` files with
      repeated `.rgbd.h5` basenames across trajectories, so the old aggregate
      script would have failed even after `99316`. Fixed
      `scripts/slurm/aggregate_rgbd_roots.sbatch` to name outputs by
      `trajectory_unit_parent__rgbd_basename` while still refusing duplicate
      trajectory units. Local preflight found no duplicate units or destination
      names, and the missing units are exactly the three that `99316` repairs.
      The three successful `99208` repair H5 files passed a local structural
      check (`3` files, `903` frames, `0` warnings) and their contact sheets
      were opened directly. Canceled old pending branch `99317-99331` before
      allocation and submitted fixed chain `99332 -> 99333 -> 99334` plus
      RGB-D method jobs `99335-99345`; state matrix is now low-priority
      `99346` after visual gate `99334`. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-02 23:45 node-exclusion source hygiene:
      removed hard-coded `#SBATCH --exclude` lines from the one-off repair
      wrapper sources and changed their manifests to record
      `EXCLUDED_NODES`. Active job `99316` was not canceled because a
      command-line-exclude replacement dry-run would start on
      `2026-06-04T00:41:50`, much later than active `99316`
      (`2026-06-03T06:01:30`). The active exclusion remains job-local and
      tied to current `98842/99208` DeviceLost evidence; future submissions
      must pass exclusions at submit time. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-02 23:46 downstream snapshot audit:
      audited active jobs with `scontrol`, submitted `jobs.tsv`, wrapper
      source, and a `/tmp` aggregate unit test. Current chain remains
      `99316 -> 99332 -> 99333 -> 99334 -> 99335...99345`, with slot and
      world-model training each on 4 H200 GPUs for 3.5h, exact96 file gates,
      RGB-D-derived inspection/eval requirements, `SLOT_SOURCE=rgbd`
      controller video, expected-slot inspection, and required nonblank video
      artifact review. Aggregate naming fix accepts same basenames only when
      trajectory-unit parents differ and rejects duplicate trajectory units.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-02 23:50 future aggregate stale-output guard:
      added source-side cleanup of stale aggregate `files/*.rgbd.h5` and a
      self-recursive source/output-root guard to aggregate scripts. `/tmp`
      tests confirm stale files are removed, duplicate trajectory units are
      rejected, and output-root-inside-source is rejected. Current pending
      aggregate `99332` was not canceled because the active aggregate root is
      absent, so this future guard is not needed for the current run and
      canceling would only churn the queue. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-02 23:56 live GPU probe and state-matrix priority correction:
      repeated `sbatch --test-only` probes for `cpu`, `gpu`, `debug`, `gpux`,
      `mgpu`, `gaosh`, `engram`, and `test` under live account constraints.
      Current user association is account `mayi`; `gaosh`/`engram`/`test`
      reject the available account and matching-account attempts, `gpux`/`mgpu`
      are inactive or drained, `debug` 2/4GPU hits `MaxGRESPerAccount`, and
      new `cpu` 1/2/4GPU probes forecast around `2026-06-04T00:52`, later
      than core repair `99316` now forecast at `2026-06-03T01:05:45`. Short
      10-60 minute 1GPU probes and a 6CPU/32G shape did not find earlier
      backfill. Corrected state/oracle matrix `99346` with
      `scontrol update JobId=99346 Dependency= Nice=0`; it is now eligible,
      normal-priority, and dependency-free. Wrapper syntax and controller
      Python compile preflights passed. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-02 23:59 submitted snapshot audit:
      verified the active Slurm snapshots, not only source files. `99316`
      repairs exactly the three remaining units; `99332` has the aggregate
      duplicate-unit guard and fixed naming; exact96 structural/visual gates,
      4H200/3h slot/world-model training, RGB-D-derived inspection/eval,
      `SLOT_SOURCE=rgbd` controller video, expected slot-source inspection,
      and required nonblank video review remain present in submitted jobs.
      `99316` and `99346` are still pending with no new artifacts yet.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-03 00:02 RGB-D method contract static preflight:
      while no new artifacts existed, shell syntax and Python compile checks
      passed for the active RGB-D data, slot, predicted-slot, world-model,
      controller, and video-review chain. A corrected static contract check
      confirmed RGB-D predictions flow through `slots/*`, oracle labels stay
      in `oracle_slots/*` for inspection, world-model training reads `slots`
      with `oracle_slots_read=false`, the RGB-D-derived wrapper validates
      predicted-slot uncertainty/probability datasets before training, and
      inspection/eval/controller gates require RGB-D-derived evidence. This is
      preflight only, not method evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-03 00:07 live queue and duplicate-submission probe:
      `99316` remains pending with forecast `2026-06-03T01:02:45` and no new
      repair artifacts. `99346` remains dependency-free but forecast moved to
      `2026-06-03T15:00:00`. Fresh `sbatch --test-only` probes for smaller
      1GPU 6CPU/32G 10-60 minute jobs found no earlier legal path:
      `cpu` starts `2026-06-03T22:58:13`, `debug` starts
      `2026-06-04T07:03:05`, `gpu` starts `2026-06-05T14:43:13`,
      `gpux`/`mgpu` are unavailable, and `gaosh`/`engram`/`test` reject the
      account. No duplicate repair or split state job was submitted. Evidence
      note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-03 00:34 incomplete RGB-D aggregate refusal preflight:
      without re-querying the queue before the half-hour cadence, ran the
      aggregate wrapper locally on the current three source roots
      (`48 + 42 + 3 = 93` files) with `EXPECTED_RGBD_FILES=96`. It exited
      nonzero with `Found 93 RGB-D files across source roots; expected exactly
      96` and created `0` aggregate RGB-D files under
      `experiments/world_model_task_rebinding/rgbd_aggregate_gate_preflight/incomplete93_refusal_20260603_003417`.
      This proves the current chain refuses incomplete RGB-D inputs before
      slot/world-model training. It is gate evidence only, not method
      evidence; the missing three units still wait for `99316`.
- [x] 2026-06-03 00:45 RGB-D repair failure and replacement chain:
      `99316` ran on `server21` from `00:25:50` to `00:42:26` and failed
      with exit `66` after `3/3` units recorded SAPIEN/Vulkan
      `vk::Device::waitForFences: ErrorDeviceLost`; it produced `0` RGB-D
      H5 files. This is a rendering/scheduling failure, not RGB-D method
      evidence. Canceled dead downstream branch `99332-99345`. Added new
      output-root repair wrapper
      `scripts/slurm/repair_rgbd_remaining3_after99316_20260603_0045.sbatch`
      and submitted job `99590` with job-local
      `--exclude=server10,server28,server21`, tied to current DeviceLost
      evidence only. Submitted replacement exact96 chain
      `99591 -> 99592 -> 99593` and RGB-D method chain `99594-99604`.
      `99590` is pending with forecast `2026-06-03T01:29:53`; state matrix
      `99346` is running on `server42`.
- [x] 2026-06-03 00:10 source-set preflight:
      current RGB-D source roots still contain exactly `93` files before
      `99316`: front48 retry `48`, latehalf `42`, and successful `99208`
      repair `3`. No duplicate trajectory-unit parent directories were found.
      The only absent units are still
      `hole_constant_seed3000_n16_traj_5`,
      `hole_reverse_seed4000_n16_traj_5`, and
      `peg_disturb_seed5000_n16_traj_5`, which are exactly the `99316`
      worklist. The target aggregate root is still absent, so no stale
      aggregate files exist. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-03 00:14 resource probe and state-smoke execution check:
      live account/partition checks show the current user has only account
      `mayi`. `gaosh`, `engram`, and `test` partitions advertise
      `AllowAccounts=null`, and `sbatch --test-only` fails on them for
      account `mayi`, `null`, and the default account. `gpux`/`mgpu` remain
      drained or inactive. New `cpu` 1/2/4GPU ten-minute probes forecast
      `2026-06-03T23:04:36`, `debug` 1GPU forecasts
      `2026-06-04T07:03:05`, and `gpu` forecasts
      `2026-06-05T13:58:36`, all later than active RGB-D repair `99316`
      (`2026-06-03T01:05:45`). State/oracle matrix job `99346` is
      dependency-free and eligible; its walltime was reduced from `90` to
      `45` minutes to improve backfill fit without changing scenarios,
      controller settings, or metrics, but the current start estimate remains
      `2026-06-03T13:00:00`. No duplicate GPU job was submitted. Current
      completed state scaffold evidence remains move-stop seed `7400` CV and
      learned-WM successes plus peg-drop/regrasp visual success; continuous,
      reverse, and peg-disturb state matrix evidence waits for `99346`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-03 00:20 post-repair readiness preflight:
      because the last queue/artifact check was only at `00:18`, no new queue
      probe was run. Instead, the exact post-`99316` path was checked without
      running heavy work: `repair_rgbd_late_shard5_traj5_remaining3...`,
      `aggregate_rgbd_roots`, exact96 structural inspection, exact96 visual
      inspection, RGB-D slot training/export, RGB-D-derived world-model
      training, RGB-D controller video, and video artifact review all pass
      shell syntax or Python compile checks as appropriate. The method-chain
      manifest still requires exact96 RGB-D files, `MIN_TRAIN_SECONDS=10800`
      for both 4H200 training jobs, `REQUIRE_RGBD_DERIVED=true` for world-model
      inspection/eval, `SLOT_SOURCE=rgbd` for the controller, and required
      nonblank video review. This is readiness/preflight evidence only, not
      RGB-D data or method evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-03 00:24 remaining-three input H5 preflight:
      without re-querying Slurm, verified the three source state-rollout H5s
      used by `99316`. Each source file exists, has `16` trajectory groups,
      selects `traj_5` for the repair unit, and the selected trajectory has
      `actions`, `env_states`, `obs_stack`, `slots`, and `perturb` with the
      expected controller/RGB-D replay fields. Summaries are
      `hole_constant` seed `3005`, `hole_reverse` seed `4005`, and
      `peg_disturb` seed `5005`. A local first attempt at this read-only check
      had an HDF5 group/dataset scripting bug, and a transient numpy import
      failure was checked with `pip check` plus repeated import; numpy/h5py now
      import correctly, so no reinstall was performed. This proves only that
      the repair inputs are structurally available, not that RGB-D repair or
      the method has succeeded. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-03 00:27 RGB-D visual-review tooling smoke:
      because it was still too early to re-check the queue, ran a lightweight
      visual artifact inspector smoke on the already completed front48 RGB-D
      root. The tool sampled `1` H5 file and `2` frames from base/hand cameras,
      produced `rgbd_visual_review_sheet.png`, reported
      `valid_visual_artifacts=true` and `num_warnings=0`, and the generated
      sheet was opened directly. The sheet is nonblank and shows RGB/depth
      views of the table, peg/hole, and gripper. This validates the review
      tooling and manual-inspection path only; it is not new RGB-D method
      evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-03 00:29 video artifact-review tooling smoke:
      still before the next queue check, ran `inspect_video_artifacts.py` on
      the existing state/oracle peg-drop scaffold video `95107` to validate the
      future RGB-D controller video review path. The tool found `1` readable
      nonblank MP4 with `301` frames, sampled `10` frames, and generated
      `peg_drop_seed7300_review_sheet.png`. The generated sheet was opened
      directly and is nonblank, showing peg drop/regrasp/approach frames. This
      is video-review tooling evidence only, not RGB-D controller or method
      evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [x] 2026-06-03 00:31 existing source companion-artifact preflight:
      still before the next queue check, scanned only filenames under the three
      current RGB-D source roots. The existing source set has `93` RGB-D H5
      files and `93` unique trajectory-unit parent directories, with no
      duplicate units. Every existing RGB-D file has local companion artifacts:
      at least one contact sheet, one MP4 preview, and one JSON manifest/report.
      Root counts remain `48 + 42 + 3`. This confirms current source artifact
      hygiene before `99316` adds the last three files; it is not exact96 data
      evidence or method evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- [ ] Full RGB-D generation, strict RGB-D inspection, RGB-D representation
      training, and RGB-D world-model/controller evaluation are blocking core
      tasks, not follow-up polish.
- [x] Add the missing RGB-D-derived world-model bridge:
      RGB-D slot extractor predictions are exported into trajectory H5 files
      whose `slots/*` are RGB-D-derived values; oracle labels are copied only
      to `oracle_slots/*` for inspection.
- [x] Submit the first formal RGB-D world-model chain:
      Current auditable full96 chain is
      `94676 -> 96266 -> 97676 -> 96649 -> 96650 -> 96651 -> 96652 -> 96654 -> 96655 -> 96656`.
      Diagnostic slot-sensitivity job `96653` waits on `96652`; RGB-D
      controller video/inspection/review are `96657 -> 96658/96659`. Jobs
      `96649` and `96654` each request one node / four `NVIDIAH200` GPUs and
      enforce `MIN_TRAIN_SECONDS=10800`. Gate `96266` requires exactly 96
      RGB-D trajectory files and zero structural inspection warnings; visual
      gate `97676` requires exact 96 RGB-D files, base/hand cameras, nonblank
      sampled RGB/depth, and zero visual-audit warnings before any training
      starts. The earlier `95938 -> 95996...` chain was canceled before
      running because `MIN_RGBD_FILES=90` was too weak for the 96-trajectory
      source and would allow incomplete RGB-D data to become method evidence.
      Old pending downstream jobs `96273`, `96274`, `96275`, `96276`, and
      `96421` were later canceled because their submitted exports did not
      require `REQUIRE_RGBD_DERIVED=true` at world-model inspection/eval time.
      The later pending method jobs `96267...96272` and `96528...96532` were
      also canceled before running because their submitted batch snapshots
      still had weak defaults such as `MIN_RGBD_FILES=2` or
      `MIN_PREDICTED_SLOT_FILES=90`; current jobs `96649...96659` were
      submitted after full96 and RGB-D perception-input default hardening.
- [ ] Wait for full RGB-D data, RGB-D slot inspection, RGB-D-derived slot
      export, 4H200/3h RGB-D world-model training, strict inspection, and
      eval before making any method claim.
- [x] Add RGB-D controller slot-source path and queue the first real
      RGB-D-controller video smoke:
      current auditable video chain is `96656 -> 96657 -> 96658`, with video
      artifact review job `96659` after `96657`. Job `96657` uses
      `slot_source=rgbd`; job `96658` requires `EXPECTED_SLOT_SOURCE=rgbd`;
      `96659` requires `REQUIRE_VIDEO=true` and `REQUIRE_NONBLANK=true`.
      Any success claim still requires manual video/contact-sheet inspection.
      Old artifact job `96277` was canceled while pending and replaced by
      `96421`, then the whole old downstream chain was superseded by
      `96528...96532`, then by `96649...96659`, to require RGB-D-derived
      world-model evidence at inspection/eval time and strict current
      full96/perception-input defaults in submitted snapshots.
- [x] Add and queue a RGB-D predicted-slot sensitivity diagnostic:
      main chain `96652 -> 96653` and failover chain `96663 -> 96664`.
      The diagnostic compares RGB-D-derived `slots/*` against
      `oracle_slots/*` inspection labels and reports whether visual slot error
      flips handoff/insert/bridge gates. This is diagnostic only; it does not
      change controller success scoring and is not method-success evidence.
- [x] Correct root idea notes and global allocation rule:
      `idea.md`, `IDEA.md`, and `AGENTS.md` now state that state/oracle slots
      are scaffold only, method evidence must be RGB-D-derived, and RGB-D
      generation may use non-wasteful partial-node shard layouts such as 4x4
      or 8x2. The earlier refusal of one-GPU render shards was superseded by
      the `2026-06-02T21:55+08:00` user override; current RGB-D generation may
      use disjoint one-GPU shards when they get RGB-D data running sooner.
- [x] Harden future distributed RGB-D fallback against the same 90-file
      downgrade: `scripts/slurm/inspect_rgbd_dataset.sbatch` now supports
      optional `EXPECTED_RGBD_FILES`, and
      `scripts/slurm/submit_rgbd_distributed_shards.sh` defaults to
      `MIN_RGBD_FILES=96` and `EXPECTED_RGBD_FILES=96` for formal distributed
      RGB-D shard inspection.
- [x] Harden formal RGB-D method-chain defaults against accidental weak
      resubmission: `submit_auditable_rgbd_method_chain_job94676.sh`,
      `train_rgbd_slot_extractor_ensemble_4gpu.sbatch`,
      `export_rgbd_predicted_slots.sbatch`,
      `inspect_rgbd_predicted_slot_export.sbatch`,
      `evaluate_rgbd_slot_sensitivity.sbatch`,
      `train_rgbd_derived_object_world_model_ensemble_4gpu.sbatch`,
      `submit_rgbd_failover_after_job.sh`, `audit_rgbd_render_output.sbatch`,
      and `inspect_full_rgbd_dataset_strict.sbatch` now default to full96
      gates where they are formal RGB-D file-count gates. The slot training
      wrapper also gathers only `*.rgbd.h5` from `INPUT_RGBD_DIR`, so state H5
      files cannot silently enter RGB-D slot training through a directory
      argument. Follow-up at `2026-06-02T18:34+08:00`: future formal
      slot-training, RGB-D predicted-slot export/inspection/sensitivity, and
      RGB-D-derived world-model training wrappers now use exact expected file
      counts by default, not only minimum counts. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_exact_count_wrapper_hardening_1834.md`.
- [x] Harden downstream RGB-D method-evidence gates:
      slot inspection now reports and requires `rgbd_slot_training_evidence`
      for full96 RGB-D input, RGB-D image channels from base/hand cameras,
      robot-only qpos/qvel proprio with no object-oracle proprio names, and
      4H200/3h; RGB-D predicted-slot export now requires that evidence;
      world-model inspection now reports
      `rgbd_derived_training_evidence` for RGB-D-predicted `slots/*`,
      `oracle_slots_not_used=true`, full96 predicted-slot inputs, and 4H200/3h.
      The RGB-D controller wrapper runs the current generic evaluator, which
      now refuses `SLOT_SOURCE=rgbd` unless both RGB-D method-evidence fields
      are present.
- [x] Harden RGB-D-derived world-model input boundary:
      `export_rgbd_predicted_slots.py` now writes H5 file/group attrs proving
      `slots/*` are RGB-D predictions and `oracle_slots/*` are inspection-only;
      `inspect_rgbd_predicted_slot_export.py` requires those attrs;
      `object_slot_dataset.py` records `oracle_slots_read=false` and
      `rgbd_predicted_slot_input_evidence`; `train_object_state_world_model.py`
      refuses RGB-D-predicted-slot-like files without complete boundary attrs;
      and `inspect_world_model_ensemble.py` requires per-member dataset
      manifests proving RGB-D-predicted slots before
      `rgbd_derived_training_evidence` can be true. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_world_model_boundary_hardening_1658.md`.
- [x] Harden RGB-D slot input boundary evidence:
      `rgbd_slot_dataset.py` now records `input_modality`,
      `proprio_source_paths`, `proprio_boundary`, `target_boundary`,
      `target_source_paths`, and `expected_rgbd_channels` in each member
      manifest; `inspect_rgbd_slot_extractor_ensemble.py` requires those
      boundary fields before `rgbd_perception_input_evidence` can be true.
      Visibility projection was also fixed to project camera xyz through the
      3x3 intrinsic matrix. Positive/negative fake-ensemble smoke checks
      confirm that missing boundary metadata prevents RGB-D evidence. Evidence
      note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_slot_boundary_hardening_1652.md`.
- [x] Harden RGB-D data gate and render audit implementation:
      `inspect_rgbd_dataset.py` now warns on non-single-trajectory formal
      files, non-contiguous `source_frame_indices`, and action/frame count
      mismatches; recursively checks nested `env_states` frame counts; and
      fixes projection sanity to project camera xyz rather than a homogeneous
      4-vector through a 3x3 intrinsic matrix. The render audit now reports
      expected RGB-D counts, duplicate success units, success directories
      without `.rgbd.h5`, and too many/too few RGB-D outputs.
      This tightens missing data-alignment checks; it does not change task
      success scoring or allow state/oracle evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_gate_audit_hardening_1646.md`.
- [x] Cancel stale non-RGB-D video/debug jobs to protect the current method
      path: `95191`, `95192`, `95209`, `95210`, `95215`, and `95216` were
      canceled while pending. They were old state/oracle learned-WM video or
      debug branches and are not valid RGB-D method evidence.
- [x] Add and queue a post-render RGB-D output audit:
      current auditable audit job is `94676 -> 97571` with dependency
      `afterany:94676`. It summarizes worklist coverage, success/failed render
      units, RGB-D file counts, and Slurm log error patterns for failure
      localization. The audit script now also reports missing success units
      and missing success-output directories so partial trajectory-level
      renders are classified before generic missing-output states. This is
      diagnostic only; exact full96 strict RGB-D dataset inspection `96266`
      remains the current data gate. Old audit job `96311` was canceled while
      pending and replaced by `97571` because its submitted snapshot did not
      pass exact `EXPECTED_RGBD_FILES=96` into the Python audit. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_render_failure_classification_1819.md`.
- [x] Replace ambiguous strict RGB-D inspection dependency:
      old inspection job `95704` was canceled because its exported
      `MIN_RGBD_FILES=90` setting could not be verified from current Slurm
      state. Intermediate inspection `95914` used explicit submission export
      but was superseded by auditable wrapper job `95938`, whose submitted
      batch script itself hardcodes `MIN_RGBD_FILES=90` and
      `REQUIRE_NO_WARNINGS=true`. This was later found insufficient because
      the rollout source has exactly 96 trajectories. Jobs `95938` and
      `95996...96006` were canceled while pending and replaced by exact
      full96 gate `96266` and an intermediate `96267...96276/96421` chain,
      which was later canceled before running.
- [x] Clean stale "oracle first" wording from active plans and scripts:
      `PLAN/world_model_task_rebinding/00_overview.md`,
      `PLAN/world_model_task_rebinding/06_rgbd_and_baselines.md`,
      `PLAN/world_model_task_rebinding/07_experiment_matrix.md`,
      `scripts/world_model/evaluate_rebinding_controller.py`, and
      `scripts/world_model/collect_dynamic_state_rollouts.py` now state that
      state/oracle paths are scaffold/debug only and RGB-D-derived
      representations are required for method evidence.
- [ ] Current RGB-D data blocker at `2026-06-02T14:39+08:00`: formal render
      job `94676` is still pending for `2026-06-03T00:01:27` on
      `server[20,42]` with dense 2-node / 16-GPU allocation. The `job94676`
      output directory does not exist yet, so there is still no full RGB-D
      data, no RGB-D slot training result, no RGB-D-derived world model, and
      no RGB-D controller video evidence. Fresh non-wasteful `sbatch
      --test-only` probes for 1x8, 2x8, 4x4, 8x2, 4x8, and 8x8 all start
      later than `94676`; no duplicate render was submitted.
      Recheck at `2026-06-02T14:52+08:00`: `94676` is still pending for the
      same forecast on `server[20,42]`, no Slurm logs or `job94676` output
      files exist, and the auditable downstream chain remains dependency
      blocked.
      Recheck at `2026-06-02T14:59+08:00`: `94676` remains pending on
      `server[20,42]`, but the forecast slipped to `2026-06-03T03:00:00`.
      No logs or `job94676` files exist. Fresh non-wasteful replacement probes
      were still later, so no duplicate render was submitted.
      Recheck at `2026-06-02T15:09+08:00`: `94676` is still pending with the
      same `2026-06-03T03:00:00` forecast, no logs or `job94676` files exist,
      and every downstream RGB-D slot/world-model/controller job is still
      dependency-pending. Distributed non-wasteful shard probes were also
      later: 8 shards x 1 node x 2 GPU forecast `2026-06-04T18:46:25`,
      4 shards x 1 node x 4 GPU forecast `2026-06-04T20:29:41`, 2 shards x
      2 nodes x 4 GPU forecast `2026-06-05T02:25:27`, and 1 shard x
      2 nodes x 8 GPU forecast `2026-06-05T02:25:27`. No duplicate render was
      submitted because it would not accelerate RGB-D data availability.
      Recheck at `2026-06-02T15:37+08:00`: `94676` is still pending and
      `job94676` contains zero RGB-D H5 files, so there is still no RGB-D
      method evidence. The source rollout set has exactly 96 trajectories;
      old 90-file gates/chains `95938 -> 95996...96006` and
      `96215 -> 96216...96228` were canceled while pending. At that moment the
      exact full96 inspection gate was `96266` after `94676`, followed by an
      intermediate method chain
      `96267 -> 96268 -> 96269 -> 96270 -> 96272 -> 96273 -> 96274`, plus
      diagnostic `96271` and video review `96275 -> 96276/96421`. The
      then-current exact-full96 gate failover was
      `96278 -> 96279/96280 -> 96281 -> 96282 -> 96283 -> 96284 -> 96286 -> 96287 -> 96288`,
      plus `96285` and `96289 -> 96290/96422`. Those intermediate method
      chains were later canceled before running.
      Recheck at `2026-06-02T15:47+08:00`: `94676` is still pending with
      `StartTime=2026-06-03T00:01:27`, `SchedNodeList=server[20,42]`,
      dense 2-node / 16-GPU allocation, and `TresPerNode=gres:gpu:8`.
      `job94676` still has zero `.rgbd.h5` files and no Slurm logs, so there
      is still no RGB-D method evidence. Full96 audit/gate/downstream jobs
      remain dependency-pending: `96311`, `96266`, and the then-current
      `96267...96276/96421` chain, which was later canceled before running.
      Fresh `sbatch --test-only` probes for allowed non-wasteful layouts were
      all later than `94676`: 1x2/1x4/1x8 at `2026-06-04T21:47:41`, 2x4/2x8
      at `2026-06-05T03:43:27`, and 4x4/8x2/8x8 at
      `2026-06-06T10:44:14`; no duplicate render was submitted. The pending
      failover render `96278` was inspected from its submitted batch script
      and contains the current hardened wrapper features: `python_retry.sh`,
      trajectory `INPUT_WORKLIST`, `work_unit_count`, `success_unit_count`,
      `failed_unit_count`, and non-wasteful multi-node allocation guard.
      Recheck at `2026-06-02T16:14+08:00`: `94676` is still pending with
      `StartTime=2026-06-03T00:01:27`, dense 2-node / 16-GPU allocation, and
      `job94676` still has zero `.rgbd.h5` files. There was still no RGB-D
      method evidence. Old pending weak downstream snapshots were canceled
      and replaced so world-model inspection/eval both required
      `rgbd_derived_training_evidence=true`; that intermediate `965xx`
      downstream replacement was later also canceled before running because
      submitted defaults were still not strict enough for formal full96
      evidence.
      Recheck at `2026-06-02T16:18+08:00`: `94676` is still pending for
      `2026-06-03T00:01:27`, scheduled on dense 2-node / 16-GPU
      `server[20,42]`, with zero `.rgbd.h5` files and no stdout/stderr logs
      yet. Fresh allowed-layout `sbatch --test-only` probes were all later
      than `94676`: 1x2/1x4/1x8 at `2026-06-05T01:46:41`, 2x4/2x8 at
      `2026-06-05T07:42:27`, and 4x4/8x2/8x8 at
      `2026-06-06T14:43:14`; no duplicate render was submitted. Node-risk
      canaries do not justify relaxing render exclusions: `server10` job
      `95266` and `server58` job `95265` both failed with Vulkan
      `ErrorDeviceLost`, while `server39` job `95357` and `server56` job
      `95358` are still pending for `2026-06-05T22:00:00`.
      Requeue at `2026-06-02T16:28+08:00`: old pending method jobs
      `96267...96272`, `96528...96532`, `96281...96286`, and
      `96533...96537` were canceled while dependency-pending because their
      submitted batch snapshots still had weak formal defaults even though the
      manifests recorded 96-file exports. Current main chain is
      `96649 -> 96650 -> 96651 -> 96652 -> 96654 -> 96655 -> 96656 ->
      96657 -> 96658/96659`; current exact-full96 failover method chain is
      `96660 -> 96661 -> 96662 -> 96663 -> 96665 -> 96666 -> 96667 ->
      96668 -> 96669/96670`. The render/data jobs `94676`, `96266`, `96311`,
      `96278`, `96279`, and `96280` were not canceled.
      Recheck at `2026-06-02T16:35+08:00`: `94676` remains pending for
      `2026-06-03T00:01:27` with dense 2-node / 16-GPU allocation on
      `server[20,42]`, no Slurm logs, and zero formal `job94676` `.rgbd.h5`
      files. Allowed replacement layout probes are all later than `94676`:
      1x2/1x4/1x8 at `2026-06-05T01:36:41`, 2x4/2x8 at
      `2026-06-05T07:32:27`, and 4x4/8x2/8x8 at
      `2026-06-06T14:33:14`. No duplicate render was submitted. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_queue_resource_audit_1635.md`.
      Submitted-snapshot audit: `94676` calls the current task script but its
      batch snapshot lacks final success/failed-unit aggregation, so `94676`
      exit code alone is not evidence; strict gate `96266` and audit `97571`
      remain required before any downstream method claim.
      Recheck at `2026-06-02T16:50+08:00`: `94676` is still pending with
      reason `Resources`, the same `2026-06-03T00:01:27` forecast, and zero
      formal `job94676` `.rgbd.h5` files. Downstream `96266`, then-current
      audit `96311`, and `96649...96670` remain dependency-pending.
      Recheck at `2026-06-02T17:11+08:00`: `94676` is still pending for
      `2026-06-03T00:01:27` on `server[20,42]`, 2 nodes / 16 GPUs, with zero
      formal `job94676` `.rgbd.h5` files. Fresh legal replacement probes were
      later than `94676`: distributed 8x2, 4x4, and 2x8 shard layouts at
      `2026-06-05T01:03:41`; single 2x8 at `2026-06-05T06:57:27`; single
      4x4, 8x2, and 4x8 at `2026-06-06T13:58:14`. No duplicate render was
      submitted. Submitted-snapshot audit confirmed `96266` and then-current
      audit `96311` hardcoded full96/audit gates; failover render `96278` is the
      hardened wrapper with worklists, timeout, failed-unit refusal, and
      non-wasteful allocation guard. Current main and failover `jobs.tsv`
      records prove 96-file gates, 4H200/3h training, RGB-D-derived
      world-model inspection/eval gates, `SLOT_SOURCE=rgbd`,
      `EXPECTED_SLOT_SOURCE=rgbd`, and nonblank video artifact gates. This is
      queue/gate evidence only, not RGB-D method evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_queue_snapshot_audit_1711.md`.
      Follow-up at `2026-06-02T17:20+08:00`: added formal RGB-D visual
      artifact audit so full96 RGB-D data will also produce padded RGB/depth
      review sheets for direct inspection. Primary visual audit job `97006`
      waits on `afterok:96266`; failover visual audit `97009` waits on
      `afterok:96279` and uses a hardcoded failover wrapper. Hidden-export
      failover audit `97007` was canceled before running. The visual audit
      requires exact 96 files, `base_camera` and `hand_camera`, and basic
      nonblank RGB/depth samples. Smoke validation on existing small RGB-D
      data produced `/tmp/reflex_rgbd_visual_smoke/rgbd_visual_review_sheet.png`,
      which was opened directly. This remains data-quality evidence only and
      does not change full96, training, world-model, controller, or video
      success gates.
      Dependency update at `2026-06-02T17:22+08:00`: tightened training
      dependencies so visual data-quality audit must pass before RGB-D slot
      training starts. `96649` now depends on `afterok:97006` instead of
      `afterok:96266`; `96660` now depends on `afterok:97009` instead of
      `afterok:96279`. This prevents structurally valid but visually broken
      RGB-D data from silently entering slot training. It does not change
      full96 inspection, 4H200/3h training floor, downstream world-model eval,
      controller metrics, or success claims.
      Follow-up at `2026-06-02T17:28+08:00`: added a separate visual-gate
      failover branch because exact full96 gate `96266` can pass while visual
      artifact audit `97006` fails. New failover render `97039` depends on
      `afternotok:97006`; strict inspection `97040` and render audit `97572`
      wait on `afterany:97039`; visual artifact audit `97053` waits on
      `afterok:97040`. The failover method chain `97042...97052` was already
      submitted from this RGB-D root, and `97042` was updated to depend on
      `afterok:97053` instead of `afterok:97040`. This closes the gap without
      changing evaluation or allowing state/oracle fallback. Old audit
      job `97041` was canceled while pending and replaced by `97572` for the
      same exact-96 render-audit argument hardening. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_visual_gate_failover_1728.md`.
      Follow-up at `2026-06-02T18:25+08:00`: visual artifact inspection now
      treats any warning as a failed visual data gate by default. Old primary
      visual audit `97006` and failover visual audit `97053` were canceled
      while pending and replaced with explicit no-warning snapshots `97676`
      and `97677`. Current dependencies are `96649 afterok:97676`,
      `97039 afternotok:97676`, and `97042 afterok:97677`. Lightweight H5
      smoke accepted a complete base/hand RGB-D file and rejected missing
      hand-camera and non-finite-depth cases. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_visual_no_warning_gate_1825.md`.
      Recheck at `2026-06-02T17:31+08:00`: formal render `94676` is still
      pending with `Reason=Resources`, `StartTime=2026-06-03T00:01:27`, and
      scheduled dense 2-node / 16-GPU allocation on `server[20,42]`.
      The formal `job94676` root still contains zero `.rgbd.h5` files, so
      there is still no RGB-D data, RGB-D slot training, RGB-D-derived
      world-model, or RGB-D controller-video evidence. Fresh legal
      `sbatch --test-only` probes were later than `94676`: 1x2/1x4/1x8 at
      `2026-06-05T03:03:52`, 2x4/2x8 at `2026-06-05T06:08:27`, and
      4x4/8x2/4x8/8x8 at `2026-06-06T13:09:14`. No duplicate render was
      submitted because it would not accelerate RGB-D data availability.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_queue_recheck_1731.md`.
      Implementation hardening at `2026-06-02T17:41+08:00`: the RGB-D
      predicted-slot export/world-model input boundary now preserves visual
      uncertainty/probability features. Exported `slots` groups write
      `rgbd_pred_cont_names` and `rgbd_pred_bin_names`; export inspection
      validates those attrs; `object_slot_dataset.py` appends
      `rgbd_pred_cont_std`, all binary probabilities, and
      `rgbd_pred_bin_std` to RGB-D-derived world-model features; world-model
      manifests now record actual feature names; online controller feature
      construction follows the trained manifest. Lightweight smoke verified a
      synthetic RGB-D predicted-slot file produces 78-D features
      (37 base + 41 RGB-D aux) and inspector warnings remain zero. This is not
      RGB-D method evidence; it prevents the future RGB-D world model from
      discarding visual confidence/visibility information. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_world_model_uncertainty_features_1741.md`.
      Gate hardening at `2026-06-02T17:45+08:00`: world-model ensemble
      inspection now requires RGB-D auxiliary feature evidence before
      `rgbd_derived_training_evidence=true`. Each member manifest must prove
      the trained feature contract includes the RGB-D auxiliary features from
      `dataset_meta.rgbd_aux_feature_names`, and those aux features must cover
      continuous prediction std, binary probabilities, and binary prediction
      std. Positive/negative fake-ensemble smokes confirmed the gate passes
      when aux features are present and fails when provenance exists but aux
      features are missing. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_world_model_aux_gate_1745.md`.
      Wrapper hardening at `2026-06-02T17:48+08:00`: future
      `train_rgbd_derived_object_world_model_ensemble_4gpu.sbatch`
      submissions now preflight predicted-slot H5s for all three RGB-D
      uncertainty/probability datasets and their name attrs before reserving
      4 H200 GPUs. Smoke confirmed good input is accepted while missing
      `rgbd_pred_bin_std` or prediction-name attrs is rejected. Already
      submitted `96654/97047` snapshots do not include this new early wrapper
      preflight, but they are still guarded by upstream export inspection and
      downstream RGB-D auxiliary-feature ensemble inspection. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_world_model_train_preflight_1748.md`.
      Controller gate hardening at `2026-06-02T17:50+08:00`:
      RGB-D controller video wrapper now explicitly exports
      `WORLD_MODEL_ENSEMBLE_DIR` and `WORLD_MODEL_INSPECTION_JSON`, and the
      generic controller wrapper explicitly rejects `SLOT_SOURCE=rgbd` unless
      world-model inspection reports RGB-D auxiliary feature input evidence.
      Lightweight check accepted the positive fake inspection and rejected the
      negative one. Pending `96657/97050` snapshots predate the explicit
      export edit, but they still run the current generic wrapper, whose new
      aux-feature check applies at runtime. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_controller_aux_gate_1750.md`.
      Eval feature-contract hardening and queue recheck at
      `2026-06-02T17:56+08:00`: prediction evaluation now refuses mismatched
      eval/training `feature_names` and, when `--require-rgbd-derived`, also
      requires RGB-D predicted-slot evidence plus RGB-D aux feature evidence in
      the eval dataset. Lightweight smoke accepted the positive contract and
      rejected feature-mismatch and missing-aux cases. Formal RGB-D render
      `94676` is still pending for `2026-06-03T00:01:27` with dense
      2-node / 16-GPU allocation and zero formal `.rgbd.h5` files. Fresh legal
      replacement probes for 1x4, 1x8, 2x8, 4x4, and 8x2 all start later, so
      no duplicate render was submitted. Node-risk canaries on `server10` and
      `server58` failed with Vulkan `ErrorDeviceLost`; `server39` and
      `server56` canaries remain pending. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_eval_feature_contract_queue_1756.md`.
      Controller Python-level gate hardening at
      `2026-06-02T18:03+08:00`: direct Python controller evaluation now
      enforces the same RGB-D method boundary as the Slurm wrapper. With
      `slot_source=rgbd`, world-model inspection must prove RGB-D-derived
      training and RGB-D auxiliary-feature input evidence; member manifests
      must prove RGB-D-predicted slots, `oracle_slots_read=false`, and RGB-D
      uncertainty/probability aux features; RGB-D slot source creation now
      requires `rgbd_slot_training_evidence` and
      `rgbd_perception_input_evidence`. Lightweight smokes rejected missing
      RGB-D-derived world-model evidence, missing aux features, and missing
      RGB-D slot-training evidence. Formal `94676` remains pending with zero
      `.rgbd.h5` files. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_controller_python_gate_1803.md`.
      Controller inspection hardening at `2026-06-02T18:08+08:00`:
      controller-run inspection now requires H5 evidence that RGB-D control
      `slots` contain frame-aligned finite `rgbd_pred_cont_std`,
      `rgbd_pred_bin_prob`, and `rgbd_pred_bin_std`; `metric_slots` must be
      separate from control `slots`; and event logs must carry RGB-D slot
      uncertainty. `--expected-slot-source rgbd` now fails if those fields are
      missing, so a result cannot rely on the `slot_source=rgbd` string alone.
      The controller H5 writer also handles UTF-8 string arrays for RGB-D slot
      name fields. Fake H5 smoke accepted the good case and rejected a missing
      aux case. Formal `94676` is still pending with reason `Priority` and
      zero `.rgbd.h5` files. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_controller_inspection_gate_1808.md`.
      Pending controller snapshot audit at `2026-06-02T18:10+08:00`:
      submitted batch snapshots and method-chain manifests for main jobs
      `96657/96658/96659` and visual-failover jobs `97050/97051/97052` were
      inspected. Video jobs execute the current generic controller evaluator;
      inspection jobs pass `--expected-slot-source` when exported, and current
      `jobs.tsv` records `EXPECTED_SLOT_SOURCE=rgbd`; video artifact review
      snapshots have `REQUIRE_VIDEO=true` and `REQUIRE_NONBLANK=true`.
      Therefore these pending jobs will hit the current strict RGB-D
      controller evidence gates once dependencies clear. Formal `94676` is
      still pending with reason `Priority` and zero `.rgbd.h5` files. Evidence
      note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_pending_controller_snapshot_audit_1810.md`.
      Recheck at `2026-06-02T18:22+08:00`: formal render `94676` remains
      pending with `Reason=Resources`, dense 2-node / 16-GPU allocation on
      `server[20,42]`, and forecast `2026-06-03T00:01:27`. The formal
      `job94676` root still contains zero `.rgbd.h5` files. Fresh legal
      single-job probes and distributed-shard probes all start later than
      `94676`: earliest 1x2/1x4 and 8-shard/1x2 layouts start at
      `2026-06-05T04:04:24`; 2x8-like layouts start at
      `2026-06-05T08:28:27`; 4x4/8x2/4x8/8x8 layouts start at
      `2026-06-06T15:29:14`. No duplicate render was submitted because it
      would not accelerate RGB-D data availability. Current render-output
      audit jobs are `97571` after `94676` and `97572` after visual-failover
      render `97039`. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_queue_probe_1822.md`.
      Correction at `2026-06-02T18:50+08:00`: waiting only on the 16-GPU render
      block was the wrong execution strategy. Submitted a parallel distributed
      RGB-D branch with four disjoint one-node/four-GPU render shards
      `97778`, `97779`, `97780`, and `97781`. This branch writes to
      `experiments/world_model_task_rebinding/rgbd_dynamic_distributed/from_rollout_dir/distributed_4x4_now_20260602_184650`.
      Exact-96 structural inspection is `97782`, initial visual no-warning
      audit was `97783`, render-output audit is `97811`, and initial downstream
      RGB-D method jobs were `97784 -> 97785 -> 97786 -> 97787 -> 97789 ->
      97790 -> 97791 -> 97792 -> 97793/97794`, with diagnostic job `97788`.
      The initial visual/method chain was later canceled and superseded by
      hardcoded visual gate `98060` and method chain `98061`-`98071`. At this
      correction point both the original `94676` root and the distributed root
      still contain zero `.rgbd.h5` files, so this is scheduling correction
      only, not RGB-D method evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_distributed_shard_correction_1850.md`.
- [x] 2026-06-02 19:14 distributed visual-gate correction: old visual job
      `97783` used a `job94676` default wrapper and was not sufficiently
      auditable for the distributed RGB-D root, so old pending chain
      `97783`-`97794` was canceled. Added hardcoded distributed visual wrapper
      `scripts/slurm/inspect_rgbd_visual_artifacts_distributed4x4_184650.sbatch`,
      submitted visual gate `98060` after `97782`, and submitted replacement
      chain `98061 -> 98062 -> 98063 -> 98064 -> 98066 -> 98067 -> 98068 ->
      98069 -> 98070/98071`, with diagnostic `98065`. Render shards
      `97778`-`97781`, structural inspection `97782`, and audit `97811` remain
      active. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_distributed_visual_gate_correction_1914.md`.
- [x] 2026-06-02 19:10 RGB-D smoke chain while full data are still unavailable:
      slot extractor smoke `97927` used RGB-D images plus robot proprio only;
      predicted-slot export `97944` wrote 8 RGB-D-derived slot H5 files;
      RGB-D-derived world-model smoke `97986` completed on one H200; inspection
      `98008` passed with `dataset_input_representation=rgbd_predicted_slots`,
      `dataset_world_model_input_group=slots`, `dataset_oracle_slots_read=false`,
      and 41 RGB-D probability/uncertainty auxiliary features in the world-model
      input. This is not method evidence, not full-data evidence, and not
      4xH200/3h training evidence. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_predicted_slot_world_model_smoke_1910.md`.
- [x] 2026-06-02 19:18 main RGB-D gate re-audit: `94676` was pending with
      zero RGB-D H5 outputs, remained dense 2-node / 16-GPU, and used
      job-local render exclusions recorded in the submitted snapshot.
      Submitted snapshots confirm exact full96/no-warning gates `96266`, `97571`, and
      `97676`; downstream `96649` and `96654` keep 4xH200/3h; controller/video
      gates still require RGB-D slot source and nonblank video. Fresh legal
      1x2, 1x4, and 1x8 render probes all forecast `2026-06-05T06:18:41`,
      later than `94676`, so no duplicate render was submitted. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_main_gate_reaudit_1918.md`.
- [x] 2026-06-02 19:27 cancellation/resource correction: user directed canceling
      the pending 2-node / 16-GPU `94676` block. `94676` was canceled before
      start with zero H5 outputs. Old `94676`-tied failover/main chains were
      canceled, including the 2-node failover `96278` that had entered
      `Priority`, so the canceled path cannot consume large GPU blocks. Fresh
      small-card probes did not find a runnable earlier full-data path:
      conservative 4x2/8x2/4x4 and relaxed 1x2/1x4 forecast `2026-06-05` or
      later; directed 1x2 node probes were not earlier than `97778`; debug 1x2
      hit `MaxGRESPerAccount`; debug 1x1 branch `98178 -> 98179 -> 98180` was
      canceled after `StartTime=Unknown`. Active render path is now the
      1-node/4-GPU distributed branch `97778`-`97781` with strict/audit/visual
      gates `97782`, `97811`, `98060`, and downstream `98061`-`98071`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_cancel_94676_small_card_shift_1927.md`.
- [x] 2026-06-02 19:30 submitted CPU-partition small-card partial RGB-D branch:
      probes showed `cpu` partition GPU GRES could start earlier than the active
      GPU-partition `97778` branch. Submitted `98209`-`98212`, each 1 node /
      2 GPUs, shard indices 0-3 of 8, expected partial48 coverage. Attached
      partial strict inspection `98213`, render audit `98214`, and visual
      nonblank/no-warning review `98215`. This branch is not full96 method
      evidence and has no downstream method training. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_cpu_partition_partial48_1930.md`.
- [x] 2026-06-02 20:26 attached an auditable RGB-D method chain to the released
      partial48 retry gate, then superseded it before execution. First
      partial48 gate `98213` failed with only `24/48` RGB-D H5 files, so dead
      visual/training branch `98215` and `98422-98432` was canceled. Retry
      render/inspection/visual gate is `98524-98527 -> 98528 -> 98530`, exact
      `48/48` and no-warning/nonblank. The attached partial48 method chain
      `98760-98770` was later canceled while pending because partial48 must not
      become formal RGB-D method evidence. The front-half retry jobs
      `98524-98527` now serve only as data-generation input to the current
      full96 aggregate path. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_rgbd_retry_chain_2029.md`.
- [x] 2026-06-02 goal-resume execution check: AGENTS now explicitly records the
      current continuation directive, including no status-only stopping while
      aligned work remains, failed-result analysis before changes, and
      video/contact-sheet inspection before major success claims. Queue/artifact
      recheck found `partial_retry_h5=0`, `full96_h5=0`, `partial_now_h5=24`;
      `98524-98527` remain the earliest partial path at
      `2026-06-02T22:01:15`, while full96 `97778-97781` starts at
      `2026-06-03T02:31:15`. Non-executing late-half/full96 probes were later,
      so no extra duplicate render was submitted.
- [x] Replace the pending post-RGB-D chain with an auditable submit manifest:
      jobs `95705`-`95711`, `95764`, `95765`, `95781`, and `95811` were
      canceled while dependency-pending because their `--export` values could
      not be audited from Slurm state. Replacement chain
      `95996 -> 95997 -> 95998 -> 95999 -> 96001 -> 96002 -> 96003`, plus
      `96000` and `96004/96005/96006`, was submitted by
      `scripts/slurm/submit_auditable_rgbd_method_chain_job94676.sh`. The
      manifest records `RGBD_ROOT=job94676`, `MIN_RGBD_FILES=90`,
      `MIN_TRAIN_SECONDS=10800`, `PREDICTED_SLOT_DIR` for world-model input,
      and `SLOT_SOURCE=rgbd` for the controller. That 90-file chain was later
      canceled before running and replaced by the current full96 chain
      `96266 -> 96649 -> 96650 -> 96651 -> 96652 -> 96654 -> 96655 -> 96656`,
      plus diagnostic `96653` and video/inspection `96657 -> 96658/96659`.
- [x] Replace the pending render-output audit with an auditable wrapper:
      old audit job `95870` was canceled while dependency-pending because its
      generic wrapper depended on hidden `--export` values for `OUTPUT_ROOT`
      and `RENDER_JOB_ID`. Replacement `96069` used
      `scripts/slurm/audit_rgbd_render_output_job94676.sbatch`, whose
      submitted script hardcodes `OUTPUT_ROOT=.../job94676`,
      `RENDER_JOB_ID=94676`, and `MIN_EXPECTED_RGBD_FILES=90`. This was later
      canceled before running and replaced by full96 audit `96311`, whose
      submitted script hardcodes `MIN_EXPECTED_RGBD_FILES=96` and points at
      strict gate `96266`.
- [x] Queue a strict RGB-D failover path that only triggers if the primary
      strict RGB-D data gate fails:
      render-level failover `96155` was canceled because `afternotok:94676`
      would not cover the case where render exits 0 but strict RGB-D
      inspection fails due missing files or warnings. Gate failover `96180`
      was then canceled and replaced by hardened failover `96215` because the
      current render wrapper now refuses success when any trajectory unit fails
      or when success units are fewer than worklist units. That `96215`
      failover still inherited a 90-file inspection gate, so it was canceled
      before running. Current failover is triggered by exact full96 gate
      `96266`: render `96278` depends on `afternotok:96266`, uses dense
      2-node / 16-GPU allocation, and writes RGB-D to
      `rgbd_dynamic_failover/from_rollout_dir/after96266_full96_gatefail_hardened_20260602_153747/shard_0_job96278`.
      Strict failover inspection `96279` requires exactly 96 RGB-D files and
      zero warnings; diagnostic audit `96280` waits on `afterany:96278`.
      The failover RGB-D method chain
      `96660 -> 96661 -> 96662 -> 96663 -> 96665 -> 96666 -> 96667`, plus
      diagnostic `96664` and video/inspection `96668 -> 96669/96670`, waits
      on `afterok:96279`. Old failover artifact job `96291` was canceled
      while pending and replaced by `96422`, then the old downstream
      `96287 -> 96288 -> 96289 -> 96290/96422` was canceled and replaced by
      `96533 -> 96534 -> 96535 -> 96536/96537`, then by
      `96660 -> 96661 -> 96662 -> 96663 -> 96665 -> 96666 -> 96667 ->
      96668 -> 96669/96670`, to require RGB-D-derived world-model evidence
      and strict current full96/perception-input defaults in submitted
      snapshots. This covers render nonzero and render-zero-but-invalid data
      without weakening the RGB-D world-model path.

## Next Concrete Steps

- [x] Create dynamic state rollout generator.
- [x] Add Slurm wrapper for dynamic rollout generation.
- [x] Add RGB-D companion renderer and Slurm wrappers.
- [x] Validate saved state traces by replay-rendering several episodes.
- [x] Implement constant-velocity hole prediction baseline.
- [x] Train first object-state world model. Old job `94371` completed but is
      undersized under the current rule and is not training evidence. Compliant
      4-GPU H200 ensemble job `94442` completed on `server53` after
      `03:00:34`; CPU inspection `94620` passed with
      `compliant_training_evidence=true`. Prediction evaluation `94672`
      completed and compares the learned ensemble against static/CV baselines.
      This is world-model prediction evidence, not controller success evidence.
- [x] Generate first frozen-DP continuability labels. Smoke job `94432`
      completed and inspection passed as a label-chain smoke, not as a dataset
      sufficient for `C_pi` training.
- [x] Train first `C_pi` model. Pilot label shard `94445` completed and was
      inspected; compliant 4-GPU H200 training job `94471` completed on
      training-usable `server04` after `03:00:20`. CPU inspection `94618`
      passed with four complete members, min member elapsed `10817s`, and
      `compliant_training_evidence=true`. Threshold calibration first failed
      in report writing (`94659`), then completed after a tool fix (`94716`).
      The held-out conservative thresholds have zero recall, so this pilot is
      training evidence and overfit/coverage evidence, not a usable handoff
      gate.
- [x] Generate larger dynamic state rollout shard for non-smoke training and
      labeling. Dense 4-GPU job `94510` completed on `server60` with all six
      perturbation families, 16 episodes per family, and zero validator
      warnings.
- [x] Train object-state world model on the larger dynamic shard after `94510`
      completes. Primary 4-GPU H200 job `94539` completed after more than
      3 hours, strict CPU inspection `94863` passed with
      `compliant_training_evidence=true`, and prediction evaluation `94864`
      completed. This is larger-shard world-model training/prediction evidence,
      not controller success evidence. Old inspection `94619` was canceled
      before running because its submitted script snapshot lacked the
      compliance gate; old eval `94673` was canceled before running. Stale full-shard
      controller jobs `94661`-`94664` were canceled before running because
      their run group predated the explicit TCP-continuation bridge metadata.
      Backups `94679`, `94746`, and live-node retry `94809`, plus their
      inspections/evals and dependent controller branches, were canceled after
      primary inspection `94863` passed. Current full-shard controller evidence
      is therefore the primary `95055 -> 95056 -> 95057` chain only; it failed
      the unchanged final dynamic success gate and produced no gated video.
      Queue audit at `2026-06-02 09:45+08:00`: primary `94539` is running on
      `server21` with 4 H200 GPUs and elapsed `01:34:43`, so it is still
      below the 3-hour evidence floor. Strict inspection/eval remain
      `94863 -> 94864`; the alternate node-selection route retry `94809` remains a
      failover, not a duplicate result source.
      Queue audit at `2026-06-02 09:52+08:00`: `94539` is still running on
      `server21` with elapsed `01:41:58`; stdout is advancing, stderr only has
      the expected PyTorch Transformer nested-tensor warnings, and the output
      directory still has intermediate `best_model.pt` files but no final
      `model.pt`/`metrics.json`. No conclusion is allowed before the
      3-hour floor and strict inspection `94863`.
      Queue audit at `2026-06-02 10:07+08:00`: `94539` is confirmed as
      `JOB_GRES=gpu:NVIDIAH200:4` on `server21` with elapsed `01:57:21`.
      It is correctly running but still below the required 3-hour evidence
      floor, so any later failure or weak metric before strict inspection is
      not allowed to become a direction-level conclusion.
      Queue audit at `2026-06-02 10:12+08:00`: `94539` remains running on
      4 H200 GPUs with elapsed `02:01:20`; member logs are advancing but still
      below `10800s`, and final `model.pt`/`metrics.json` files do not exist.
      No training conclusion is allowed yet. Non-rendering sbatch templates
      were also updated to avoid inheriting the earlier node-exclusion snapshot,
      so recovered nodes are no longer excluded by default.
      Queue audit at `2026-06-02 10:25+08:00`: `94539` is still running on
      `server21` with one node / 4 H200 GPUs and elapsed `02:14:52`, below the
      3-hour evidence floor. The output directory still has only intermediate
      `best_model.pt` files, not final `model.pt`/`metrics.json`; stderr only
      has the expected PyTorch Transformer nested-tensor warnings. Its live
      submitted snapshot still lists `server61` in `ExcNodeList`, but this is
      not blocking allocation because the job is already running.
      Queue audit at `2026-06-02 10:36+08:00`: `94539` remains live on
      `server21` with 4 H200 GPUs, dependency cleared, elapsed about
      `02:24`, and `MIN_TRAIN_SECONDS=10800` in the submitted wrapper. Member
      logs are advancing but still below the 3-hour floor, so no weak
      validation trace or intermediate checkpoint can be used to judge the
      direction. The compliant fallback chain is still present but inactive:
      `94679` and `94746` retry normal non-rendering allocations that only
      avoid inherited rendering-node exclusions, while live-node retry `94809`
      remains behind failure dependencies. Since the primary job already has
      cards, no duplicate 4-GPU training was submitted.
      Queue audit at `2026-06-02 10:49+08:00`: `94539` remains live on
      `server21` with one node / 4 H200 GPUs and Slurm elapsed `02:39:08`.
      The latest member stdout has advanced to about `9442s`, still below the
      required `10800s`; the output directory still has only intermediate
      `best_model.pt` files, not final `model.pt`/`metrics.json`. Stderr is
      still only the expected Transformer nested-tensor warnings. This remains
      a healthy running job, not evidence for or against the method.
      Queue audit at `2026-06-02 10:54+08:00`: `94539` is still running on
      `server21` with one node / 4 H200 GPUs and Slurm elapsed `02:44:02`,
      still below the 3-hour evidence floor. The newest member stdout has
      reached about `9765s`, still below `10800s`; no final
      `model.pt`/`metrics.json` files exist. This remains live training only.
      Completion audit at `2026-06-02 11:12+08:00`: `94539` completed on
      `server21` after `03:01:17`; strict inspection `94863` completed on
      `server52` with exit `0`. Inspection reports four complete members,
      min elapsed `10870.0s`, compliant 3-hour training `True`, compliant
      4xH200 request `True`, and `compliant_training_evidence=True`.
      Prediction evaluation `94864` then completed and wrote
      `prediction_eval.md/json`. This is larger-shard world-model training
      and prediction evidence only, not controller success.
- [x] Generate larger `C_pi` label shards from the larger dynamic shard after
      `94510` completes. 4-GPU H200 label job `94540` completed on `server28`
      and wrote four inspected shards plus `label_h5s.txt` under
      `continuability_labels/full4gpu_from_job94510/job94540`. The completed
      shards contain 512 candidates total; per-shard final success rates are
      `0.21875`, `0.2109375`, `0.203125`, and `0.1328125`. Backfill label job
      `94682` and its downstream chain `94683`/`94684`/`94685` were canceled
      after the primary label chain completed and primary training started, to
      avoid spending 4 GPUs on duplicate labels/training.
- [ ] Train `C_pi` on the larger label shards after `94540` completes.
      4-GPU H200 job `94542` started on `server28` at
      `2026-06-02T10:20:07+08:00` with `MIN_LABELS=512` and
      `MIN_TRAIN_SECONDS=10800`. CPU inspection job `94658` is queued with
      `afterany:94542`, and calibration job `94660` waits on successful
      inspection. No larger-label `C_pi` training conclusion is allowed until
      `94542` reaches the 3-hour evidence floor and `94658` reports
      `compliant_training_evidence=true`.
      Queue audit at `2026-06-02 09:45+08:00`: label jobs `94540` and `94682`
      are priority-pending with concrete scheduler forecasts, while model
      training jobs `94542` and `94683` correctly remain blocked on
      successful label generation.
      Queue audit at `2026-06-02 09:52+08:00`: `94540` has no residual
      dependency and is forecast for `2026-06-02T10:51:00`; `94682` is
      forecast for `2026-06-02T11:40:47`. No label artifacts exist yet.
      Queue audit at `2026-06-02 10:07+08:00`: `94540` is running on
      `server28` with 4 H200 GPUs and is generating labels. Backfill label
      job `94682` remains scheduled for `2026-06-02T11:40:47` on `server31`.
      Downstream `C_pi` training jobs `94542` and `94683` both request
      4 GPUs with time limits above 3 hours and avoid inherited rendering-node
      exclusions.
      Queue audit at `2026-06-02 10:12+08:00`: `94540` output directory
      `continuability_labels/full4gpu_from_job94510/job94540` contains input
      validation and shard manifests but no `continuability_labels.h5` yet, so
      downstream `C_pi` model training remains correctly blocked.
      Queue audit at `2026-06-02 10:21+08:00`: `94540` completed all four
      label shards and wrote `label_h5s.txt`; `94542` is running on `server28`
      with one node / 4 H200 GPUs and a `03:30:00` time limit. Duplicate
      backfill chain `94682`/`94683`/`94684`/`94685` was canceled after the
      primary chain became live.
      Queue audit at `2026-06-02 10:25+08:00`: `94542` remains running on
      `server28` with one node / 4 H200 GPUs, elapsed about `00:05`, and only
      intermediate `best_model.pt` files. It is far below the 3-hour evidence
      floor, so current validation traces are liveness diagnostics only.
      Queue audit at `2026-06-02 10:36+08:00`: `94542` remains live on
      `server28` with 4 H200 GPUs, elapsed about `00:15`, `MIN_LABELS=512`,
      and `MIN_TRAIN_SECONDS=10800`. Early AUPRC/AUROC swings in stdout are
      not evidence for or against the handoff idea; only the strict inspection
      `94658` after a complete 3-hour run can make this a compliant training
      result.
      Queue audit at `2026-06-02 10:49+08:00`: `94542` remains live on
      `server28` with one node / 4 H200 GPUs and Slurm elapsed `00:29:48`.
      Its early validation AUPRC/AUROC/BCE traces are liveness diagnostics
      only. No conclusion is allowed until the strict 3-hour compliance
      inspection `94658` passes.
      Queue audit at `2026-06-02 10:54+08:00`: `94542` remains running on
      `server28` with one node / 4 H200 GPUs and Slurm elapsed `00:34:42`.
      The output directory still has only intermediate `best_model.pt` files.
      The stdout is advancing, but the job is far below the 3-hour evidence
      floor, so no handoff-model conclusion is allowed.
      Queue audit at `2026-06-02 11:23+08:00`: `94542` remains running on
      `server28`, still one node / 4 H200 GPUs, elapsed `01:03:26`. It is
      still below `MIN_TRAIN_SECONDS=10800`; early `best_model.pt` files and
      validation traces are liveness only.
- [x] Implement first receding-horizon rebinding controller scaffold. Slurm
      smoke `94518` is submitted to test closed-loop event logs and video.
- [x] Wire learned object-state world-model prediction into the controller.
      The new `rebind_world_model` path uses checked model manifests,
      online features matching `object_slot_dataset.py`, ensemble uncertainty,
      and model-trained horizons rather than the CV tau list.
- [x] Queue learned-world-model controller smoke behind compliant object model
      job `94442`. Dependent video smokes now wait on inspection job `94620`
      rather than raw training job `94442`; `94564` and `94576` started only
      after final `model.pt` files existed and post-training compliance
      inspection passed.
- [x] Inspect `94442` before treating dependent learned-world-model controller
      smoke `94564` as evidence. Inspection `94620` passed after final
      `model.pt` and `metrics.json` files existed for all four members.
      Controller video smoke `94564` then ran, failed, was inspected by CPU job
      `94696`, and its contact sheet was opened directly.
- [x] Inspect additional `94442` learned-world-model controller video smokes.
      `94576` (`hole_constant`) and `94577` (`hole_reverse`) both completed,
      failed final dynamic insertion, were inspected by CPU jobs `94700` and
      `94717`, and their contact sheets were opened directly. Both runs show
      bridge activity followed by many safe-retreat steps after
      `grasped=False` with regrasp disabled; they are controller/grasp
      failure evidence, not evidence against world-model task rebinding.
- [x] Validate dropped-peg regrasp primitive. No-video code-path smoke `94550`
      completed and exposed a regrasp state-machine oscillation, not success.
      The fix is implemented; same-seed no-video v2 job `94621` was canceled
      before running after offline replay showed its clearance was too tight.
      Short-allocation v2 job `94626` completed and showed close/lift happens
      but still fails to grasp, so the primitive now close-holds before lift.
      Short-allocation v3 job `94633` showed close-hold still failed because
      the primitive closed about `5cm` above the peg. Regrasp now uses the
      empirical peg-local TCP grasp pose from successful frames and a small
      vertical close window. Stale v3 video job `94634` was canceled; v4
      no-video smoke `94645` completed and inspection `94656` passed, but the
      run still failed: close happened at steps `103-104` without stable
      reacquisition, then regrasp approach drifted away. This is regrasp
      primitive failure evidence, not method failure. Because the run briefly
      reacquired grasp before the old bridge target pushed it away, same-seed
      v5 tcp-continuation no-video validation `94773` completed with inspection
      `94774`. It regained final grasp but still failed final dynamic
      insertion with peg-head YZ about `0.156m`; gate `94775` failed by design,
      and dead gated video dependencies `94776`/`94777` were canceled. Later
      v9 peg-alignment/manifold-guard run `95104 -> 95105 -> 95106` passed the
      unchanged no-video dynamic success gate for `peg_drop`; gated video
      `95107` and inspection `95108` completed. The contact sheet and an
      extracted final frame were opened directly and agree with the metrics.
      This validates the CV peg-drop/regrasp branch only; it is not a
      learned-WM success or broad generality claim.
- [ ] Run full-shard RGB-D export after larger dynamic rollout shards exist.
      Dependent dense 8-GPU render job `94541` is queued after `94510`; it uses
      rendering node-specific exclusions and refuses sparse multi-node allocation.
      CPU RGB-D dataset inspection job `94858` is queued with `afterany:94541`
      and strict `REQUIRE_NO_WARNINGS=true`.
      Because `94541` is scheduled far out, dense 2-node/16-GPU backfill job
      `94676` was also queued with the same input rollout directory and a
      shorter 2-hour limit. It still uses 8 GPUs per node and is not a sparse
      many-node/one-GPU allocation. After `94541` exposed a full-shard render
      hang on `server58`, dense RGB-D rendering was changed to trajectory-level
      work units and per-unit timeout. Pending `94676` was kept because it
      calls the current task script at runtime; it now excludes
      `server10,server58` and should create 96 trajectory work units from the
      six source H5 files. Current exact full96 strict CPU inspection `96266`
      waits on `afterany:94676` and requires exactly 96 RGB-D H5 files with
      no warnings. Earlier inspections `94665`/`94677`/`94858` and stale
      chains through `95235`, `95938`, and `96215` were canceled or
      superseded before producing usable full-shard RGB-D or slot-training
      evidence.
- [ ] Train first RGB-D slot extractor after inspected full-shard RGB-D exists.
      Dataset/model/inspection code and 4-H200 Slurm wrappers are implemented.
      Correct dependent chain is now `94676 -> 96266 -> 96649 -> 96650`,
      where `96266` requires exactly 96 RGB-D H5 files with no warnings and
      `96649` requires 4 H200 GPUs, `MIN_TRAIN_SECONDS=10800`, and at least
      96 RGB-D H5 files from the inspected RGB-D backfill directory
      `job94676`. Mistaken path-hygiene submission `94732`, missing-input chain
      `94733`/`94734`, stale-script chain `94856`/`94857`, and old full-shard
      chains through `94859`/`94860`/`94861`, `95235`, and `95938` were
      canceled before producing training evidence.
      Queue audit at `2026-06-02 09:45+08:00`: RGB-D render jobs remain
      dense: `94541` is 1 node / 8 GPUs, and `94676` is 2 nodes / 16 GPUs
      with `8` GPUs per node. Slot training `94860` stays behind strict
      dataset inspection `94859`.
      Queue audit at `2026-06-02 09:52+08:00`: full-shard RGB-D jobs remain
      priority-pending with no output artifacts. `94541` has no residual
      dependency and keeps the dense 1-node/8-GPU allocation; `94676` keeps
      the dense 2-node/16-GPU allocation.
      Queue audit at `2026-06-02 10:07+08:00`: `94541` is running on
      `server58` as a dense 1-node/8-H200 RGB-D job and is writing synchronized
      RGB-D frames. `94676` remains a dense 2-node/16-GPU backfill, not a
      sparse many-node allocation. Slot training `94860` remains gated behind
      strict dataset inspection and requests 4 GPUs for at least 3 hours.
      Queue audit at `2026-06-02 10:12+08:00`: `94541` has written one
      complete RGB-D H5 with 16 trajectories and 4816 frames; stderr is empty.
      Dense `sbatch --test-only` probes for 16/32/64 GPU RGB-D jobs all start
      days later than the running/queued jobs, so no extra RGB-D job was
      submitted.
      Queue audit at `2026-06-02 10:25+08:00`: `94541` remains running on
      `server58` as a dense one-node / 8-H200 render job. Only the first
      `none_seed1000_n16` RGB-D H5 has been observed so far, and stderr remains
      empty. Backfill `94676` remains a dense 2-node / 16-GPU job scheduled for
      `2026-06-03T17:24:12`.
      Queue audit at `2026-06-02 10:30+08:00`: `94541` is still running with
      the same single observed RGB-D H5 and empty stderr. Dense 16-GPU backfill
      `94676` moved earlier in the scheduler forecast to
      `2026-06-02T18:05:41` with reason `Resources`; it remains dense
      2 nodes / 16 GPUs and still waits for allocation.
      Queue audit at `2026-06-02 10:36+08:00`: `94541` is still running as a
      dense 1-node / 8-H200 render job and has one observed RGB-D H5 so far.
      `94676` remains a dense 2-node / 16-GPU backfill, not a sparse
      many-node allocation. RGB-D slot training `94860` is still correctly
      blocked on strict dataset inspection `94859`, requests 4 H200 GPUs for
      `03:30:00`, and keeps `MIN_TRAIN_SECONDS=10800`.
      Queue audit at `2026-06-02 10:42+08:00`: the first RGB-D shard from
      `94541` also wrote `rgbd_contact_sheet.png` and `rgbd_preview.mp4`. The
      contact sheet was opened directly and is nonblank: peg, box/hole, robot,
      and camera viewpoint changes are visible, so this is not a Vulkan
      black-frame failure. This is RGB-D pipeline evidence only, not controller
      success. `sstat` shows five active render tasks remain, matching the
      five unfinished perturbation shards, while two extra tasks had no input
      and task `3` completed `none_seed1000_n16`.
      Queue audit at `2026-06-02 10:49+08:00`: `94541` remains live on
      `server58` with one full 8-H200 node, `TresBind=gpu:single:1`, and
      `TresPerTask=cpu:6,gres:gpu:1`. This confirms the `CUDA_VISIBLE_DEVICES=0`
      lines in the render log are per-task remapping, not sparse
      one-GPU-per-node allocation. Only the first completed RGB-D shard has
      been observed so far; stderr is empty. Backfill `94676` is still pending
      as a dense 2-node / 16-GPU job and is tied to the downstream slot
      training chain. If `94541` completes and passes strict inspection before
      `94676` starts, the chain should be rewritten or resubmitted to avoid
      spending 16 GPUs on duplicate RGB-D export.
      Queue audit at `2026-06-02 10:54+08:00`: `94541` remains live on
      `server58` with elapsed `00:52:45`. There is still only one completed
      RGB-D shard, but an in-allocation check shows five render Python
      processes still active for the unfinished perturbation inputs, and
      `nvidia-smi` reports five GPUs at `100%` utilization with about `301MiB`
      memory each. The other three GPUs are idle because there are six input
      H5 files total and the `none` shard already completed. This is a live
      dense render, not a sparse allocation or obvious render hang.
      Queue audit at `2026-06-02 11:25+08:00`: `94541` is still running on
      `server58`; the five unfinished render processes remain alive and five
      H200s are at `100%` utilization. No second RGB-D H5 is visible yet, so
      downstream RGB-D slot training remains correctly blocked behind strict
      inspection `94858`.
      Intervention at `2026-06-02 11:43+08:00`: `94541` was canceled after
      `01:41:44` because it still had only the completed static
      `none_seed1000_n16` shard while five dynamic/perturbation render
      processes had no first-trajectory progress and stayed at about
      `99-100%` GPU utilization. Partial-data inspection `94858` was canceled.
      Infrastructure fix at `2026-06-02 11:49+08:00`: dense RGB-D rendering now
      shards by `(input_h5, trajectory)`, producing 96 work units for this
      full shard, and each unit has a default `1800s` timeout. Pending backfill
      `94676` was updated to exclude `server10,server58` and is forecast on
      `server[40,42]`; the strict replacement chain is
      `94676 -> 95235 -> 95236 -> 95237`.
- [ ] Validate data-measured TCP continuation bridge target. Completed
      follow-ups `94591`, `94698`, and `94645` showed that the old bridge
      stayed outside the static policy continuation manifold even when retreat
      was absent. The controller now defaults to
      `bridge_servo_reference=tcp_continuation`, using static-success
      hole-local TCP poses. Initial 3-hour no-video validation chain
      `94751`-`94760` was canceled before running because controller smokes
      are not model training and shorter walltime improves backfill. Current
      same-seed no-video validations `94763` (CV) and `94764` (learned-WM)
      completed and failed final dynamic insertion. Inspections
      `94765`/`94766` passed as failed-run inspections; gates `94767`/`94768`
      failed by design, and dead gated video dependencies
      `94769`/`94770`/`94771`/`94772` were canceled. Slot/orientation analysis
      shows the TCP reached the measured continuation position, but the
      hole-local TCP/peg orientations were tens of degrees outside successful
      static continuation frames, so the next bridge fix must be
      orientation-aware task-frame servo rather than threshold tuning. The
      orientation-aware bridge is implemented and same-seed no-video validation
      chains are queued: CV `94814 -> 94815 -> 94816 -> 94817 -> 94818`,
      learned-WM `94442` chain `94819 -> 94820 -> 94821 -> 94822 -> 94823`,
      and peg-drop/regrasp chain `94824 -> 94825 -> 94826 -> 94827 -> 94828`.
      Full-shard post-training controller chains were also replaced with
      orientation-aware variants: current `94539` branch `94865`-`94869`, plus
      `94835`-`94849` for the other object-WM duplicates.
      Stale pre-continuation controller jobs `94578`, `94582`, `94583`, and
      `94590` were canceled.
      Same-seed v7 task-frame-projected no-video validations `95039` and
      `95042` completed. CV job `95039` failed final dynamic success; learned
      WM job `95042` passed inspection `95043` and success gate `95044` with
      dynamic event before success, but has no video evidence yet. Gated video
      job `95047` is pending; it was shortened from 1 hour to 15 minutes for
      backfill without changing evaluation settings. Because the CV and
      learned-WM no-video jobs used different seeds, paired no-video controls
      were submitted: CV@7500 `95136 -> 95137 -> 95138` and learned-WM@7400
      `95139 -> 95140 -> 95141`.
      Paired controls completed: CV@7500 `95136 -> 95137 -> 95138` passed the
      no-video gate, while learned-WM@7400 `95139 -> 95140 -> 95141` failed.
      Therefore the current no-video task-frame-projected result is not
      evidence that learned WM beats CV; it only shows that the controller can
      complete some move-stop seeds. Video `95047`/inspection `95048` remain
      required before any final success claim.
      Video queue audit at `2026-06-02 10:45+08:00`: gated videos `95047` and
      `95107` still have released dependencies, but Slurm moved both forecasts
      later to `2026-06-02T18:02:04` on `server58`. Same-settings
      `sbatch --test-only` probes with 10-minute and 5-minute walltimes
      forecast `2026-06-03T13:46:05`, so no duplicate shorter video jobs were
      submitted. No `job95047`/`job95107` video artifact exists yet.
      Video queue audit at `2026-06-02 10:54+08:00`: Slurm moved video jobs
      `95047` and `95107` earlier to `2026-06-02T12:02:50`, still pending on
      priority with no artifacts. This is useful scheduling movement only;
      no success can be claimed until the videos run, inspections pass, and
      the contact sheets or frame grids are opened directly.
      Video/artifact audit at `2026-06-02 11:20+08:00`: learned-WM move-stop
      video job `95047` ran on `server10` but failed with Vulkan
      `ErrorDeviceLost` before writing metrics, H5, MP4, or contact sheet;
      inspection `95048` correctly found no video evidence. This is a render
      failure, not an evaluation change. Same-settings rerun `95191` was
      submitted with inspection `95192`; it keeps the original learned-WM
      move-stop video parameters, seed `7500`, and `world_model_dirs_job94442`,
      and only changes the output run group plus a render exclusion adding
      `server10`. Peg-drop regrasp video job `95107` completed and inspection
      `95108` found one MP4/contact sheet. I opened the contact sheet and an
      extracted final frame directly; they agree with `success_at_end=True`,
      `regrasp_count=13`, final peg-head hole-frame `x=-0.013857m`,
      `yz=0.003281m`. This is visual evidence for the CV peg-drop/regrasp
      branch only, not a learned-WM or dynamic-hole generality claim.
      Full-shard controller audit at `2026-06-02 11:16+08:00`: after
      `94539` passed strict inspection, no-video controller job `95055`
      ran with the full-shard world model and current
      `task_frame_projected` bridge. Inspection `95056` passed as a failed-run
      inspection, and success gate `95057` failed with exit `65`. The run had
      `success_at_end=False`, `success_once=False`, trigger step `90`, bridge
      steps `234`, prediction source `world_model`, final hole-frame
      peg-head `x=-0.1005m`, `yz=0.00275m`. Lateral alignment is good but
      insertion-axis progress is insufficient. Dead gated video branch
      `95058`/`95059` and dead failover branches
      `94679`-`94681`, `94746`-`94748`, `94809`-`94811`, and
      `95060`-`95077` were canceled as queue hygiene after the primary
      inspection passed and the primary gate failed. Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_job94539_prediction_and_controller.md`.
      Follow-up seed `7400` jam analysis at `2026-06-02 11:34+08:00` compared
      `95055`, `95139`, `95042`, and `95136`: both learned-WM runs fail at
      seed `7400`, while learned-WM and CV succeed at seed `7500`. The failing
      seed enters insert phase and commands axis motion, but peg-head x stalls
      around `-0.101m`; final hole height is much lower than seed `7500`.
      Debug video rerun `95209 -> 95210` was submitted with the same `95055`
      controller settings, only enabling video and excluding `server10`.
      Evidence note:
      `docs/world_model_task_rebinding/2026-06-02_seed7400_insert_axis_jam.md`.
      Aligned insertion-manifold probe `95212 -> 95213 -> 95214 -> 95215 ->
      95216` was also submitted for the same seed and `job94539` model. It
      uses peg-axis alignment, insertion-manifold guard, and slower insert-axis
      steps, while preserving the unchanged success gate and video requirement.

## Current Rule

Do not run heavy rollout, replay, rendering, or training on the login node.
Use Slurm compute nodes. There is no node exclusion list in this repo. New
submissions must use live `sinfo`/`scontrol`, `sbatch --test-only`, and
targeted canaries for job-local node decisions; if a node is excluded for a
specific job, record the current evidence in that job's manifest or evidence
note. Older entries that mention concrete node names are historical scheduling
evidence only and must not be reused as a bad-node list.

## Latest Progress

- 2026-06-02 21:55+08:00: User override superseded the previous RGB-D render
  scheduling cap and the previous refusal of one-GPU render shards. Updated
  `AGENTS.md`, `submit_rgbd_distributed_shards.sh`, and
  `render_dynamic_rgbd_dataset_from_rollout_dir_dense.sbatch` so RGB-D data
  generation can use disjoint 1GPU shards while preserving exact96, visual,
  RGB-D-derived, 4H200/3h training, controller, and video gates. Submitted
  full96 1GPU render branch `98929-99024` under
  `experiments/world_model_task_rebinding/rgbd_dynamic_distributed/from_rollout_dir/full96_1gpu96_20260602_2155`.
  Added hardcoded exact96 inspection `99027`, hardcoded visual gate `99028`,
  render audit `99029`, and downstream RGB-D method chain `99030-99040`.
  `squeue --start` forecasts the 1GPU render shards for
  `2026-06-03T00:01:27` on `server42`; H5 count is still zero, so there is
  still no RGB-D method evidence. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_full96_1gpu96_render_branch_2155.md`.
- 2026-06-02 22:00+08:00: Rechecked one-card startup paths after submitting
  `98929-99024`. The branch is still the earliest Slurm forecast at
  `2026-06-03T00:01:27`. Short low-memory one-card canary probes were later:
  generic `cpu` `2026-06-04T00:47:47`, `gpu` `2026-06-06T05:40:47`,
  `debug` `2026-06-04T07:33:05`; `test`/`gaosh`/`engram` are invalid for the
  current account. Live node parsing found only schedulable unallocated GPUs on
  `server39` and `server56` (4 each), but directed one-card probes to those
  nodes forecast `2026-06-05T13:08:08`; `server16`/`server29` remain
  down/drain and unavailable. Current full96 1GPU branch H5/video/log counts
  remain zero. This is scheduling evidence only, not RGB-D method evidence.
- 2026-06-02 22:04+08:00: Slurm forecasts moved again. Current earliest RGB-D
  render start is the aggregate front-half retry `98524` at
  `2026-06-03T01:00:00`, followed by `98525`, `98526`, and `98527` hourly
  through `04:00`. Late-half jobs `98841-98844` and the full96 one-card branch
  `98929-99024` now forecast `2026-06-03T06:01:30`. Fresh replacement probes
  on `cpu` for 1x1 15m, 1x1 45m, 1x2 45m, and 1x4 45m all forecast
  `2026-06-04` or later, so submitting another duplicate would not start
  RGB-D work earlier. Current full96/aggregate H5 and video counts remain
  zero.
- 2026-06-02 22:07+08:00: Canceled the erroneous 96-job one-GPU fanout before
  it ran. Render jobs `98929-99024` and dependent jobs `99027-99040` show
  `CANCELLED by 2059`, zero elapsed time, zero allocated nodes, and no
  generated RGB-D H5/video/contact/log artifacts. The mistake was scheduling
  one job per trajectory instead of using a small rolling batch. This is not
  method evidence and must not be treated as an active path.
- 2026-06-02 22:10+08:00: Rechecked partition/account access. Current user
  `yanhongru` has account `mayi` only. `gaosh`, `engram`, and `test`
  partitions reject the current account; `gpux` and `mgpu` are drained;
  `debug` is limited by `MaxGRESPerAccount`. Fresh `cpu` 2-GPU probes forecast
  `2026-06-04T00:54:08`, and `gpu` 2-GPU probes forecast
  `2026-06-06T09:35:08`, both later than existing jobs. Directed 2-GPU probes
  to the only schedulable nodes with apparent free GPUs (`server39`,
  `server56`) forecast `2026-06-05T11:44:08`. Existing successful 2-GPU
  render shards `98209` and `98210` each produced 12 RGB-D trajectories in
  about 5 minutes, so the bottleneck is Slurm start time, not render runtime.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_resource_account_and_rolling_policy_2218.md`.
- 2026-06-02 22:12+08:00: Added a hard guard to
  `scripts/slurm/submit_rgbd_distributed_shards.sh`: default max 8 render jobs
  per submission, default max 4 one-GPU render jobs per submission, and reject
  multi-node one-GPU-per-node sparse shards. `bash -n` passed. This preserves
  the RGB-D objective while preventing another accidental large one-GPU fanout.
- 2026-06-02 22:17+08:00: Verified the new guard with local non-Slurm checks.
  `NUM_JOB_SHARDS=96 GPUS_PER_NODE=1` exits `64` before `sbatch`; 9 render
  jobs also exit `64`; sparse `NODES_PER_SHARD=2 GPUS_PER_NODE=1` exits `64`.
  This is a scheduling-safety check only; it does not produce RGB-D data.
- 2026-06-02 22:16+08:00: Queue/artifact recheck: active RGB-D render path is
  still the existing 2-GPU aggregate path. `98524` forecasts
  `2026-06-03T01:00:00`; `98525`, `98526`, `98527`, and `98841-98844` forecast
  `2026-06-03T06:01:30`. Current aggregate, partial retry, and canceled
  1GPU96 roots all have zero new RGB-D H5, preview videos, contact sheets, and
  Slurm logs.
- 2026-06-02 21:47+08:00: Current formal chain after audit-boundary repair is
  `98524-98527 + 98841-98844 -> 98909 -> 98910 -> 98907 -> 98848 -> 98849 ->
  98911 -> 98912 -> 98914 -> 98915 -> 98916 -> 98917`, with diagnostic
  `98913` and video checks `98918/98919`. Old `98845/98846` and
  `98850-98858` are canceled before running. `98909`, `98910`, and `98907`
  use hardcoded current aggregate paths; `98914` requests one node / four H200
  GPUs for `03:30:00`. Render jobs are still priority-pending, front/late/
  aggregate H5 counts are still zero, and there is still no RGB-D method
  evidence.
- 2026-06-02 21:48+08:00: `squeue --start` shows late-half jobs `98841-98844`
  forecast for `2026-06-03T00:01:27` on `server42`, while front-half jobs
  `98524-98527` forecast sequentially on `server03` from `03:00` to `06:00`.
  Ran non-executing replacement probes for front shards `0-3` with legal 1x2
  and 1x4 CPU-partition GPU layouts; every probe forecast
  `2026-06-04T01:15:27`, later than the current front jobs, so no replacement
  front root was submitted. Current H5 counts remain zero.
- 2026-06-02 21:34+08:00: Post-correction check confirms the current formal
  chain at that time was `98845 -> 98846 -> 98907 -> 98848`, not old `98847`.
  This snapshot was superseded at `21:43+08:00` by hardcoded
  `98909 -> 98910 -> 98907` and replacement downstream `98911-98919`.
- 2026-06-02 21:31+08:00: Corrected an auditable-root risk in the full96 visual
  gate before it could run. Submitted job `98847` used the old
  `inspect_rgbd_visual_artifacts_job94676.sbatch` command; because `scontrol`
  does not expose exported `RGBD_ROOT`, the current aggregate root could not be
  verified from the submitted snapshot. Added hardcoded aggregate-root visual
  gate wrapper `scripts/slurm/inspect_rgbd_visual_artifacts_full96_aggregate_205201.sbatch`,
  submitted replacement visual gate `98907` after `98846`, updated slot
  training job `98848` to depend on `afterok:98907`, and canceled old `98847`
  before it ran. The new wrapper requires exact96, base/hand cameras,
  nonblank samples, and no warnings; this preserves the original RGB-D visual
  gate and does not change evaluation or training criteria. Manifests and
  ledgers were updated; a post-correction pre-render audit records
  `visual_gate_job_id=98907` and `superseded_visual_gate_job_id=98847`.
- 2026-06-02 21:27+08:00: Current exact96 render branch is still pending:
  `98524-98527` and `98841-98844` remain priority-pending, `squeue --start`
  forecasts `2026-06-03T00:01:27` on `server42`, and front/late/aggregate H5
  counts plus success/failed ledgers remain zero. Ran a lightweight local
  pre-render smoke of `audit_rgbd_multi_root_render.py` into
  `pre_render_multi_root_audit_2127.json/.md`; it reports zero files/ledgers
  as expected before render starts and confirms the audit path can read the
  configured roots. It is diagnostic smoke only, not render failure evidence
  or method evidence. Fresh no-execute probes remain later (`cpu 1x2 20m` at
  `2026-06-03T23:15:15`, `cpu 1x4 20m` at
  `2026-06-04T01:15:27`), so no replacement render was submitted.
- 2026-06-02 21:23+08:00: Added and submitted lightweight multi-root render
  failure-localization audit `98902` afterany all eight current render shards
  `98524-98527` and `98841-98844`. It writes
  `multi_root_render_audit.json` under the aggregate root and summarizes
  front/late/aggregate H5 counts, success/failed ledgers, worklist counts, and
  render log error patterns if any shard fails. This is diagnostic only:
  exact96 structural inspection `98846`, visual gate `98847`, 4H200/3h slot
  and world-model training, RGB-D-derived evidence, controller evaluation, and
  video review remain unchanged. Static checks passed for the new Python and
  Slurm wrapper; `scontrol show job -dd 98902` reports `ExcNodeList=(null)`.
- 2026-06-02 21:25+08:00: Rechecked current external state. Representative
  front job `98524` and late job `98841` are still priority-pending for
  `2026-06-03T00:01:27` on `server42`, both with `ExcNodeList=(null)` and
  `TresPerNode=gres:gpu:2`. Multi-root audit `98902` is dependency-pending on
  all eight render shards. Front/late/aggregate RGB-D H5 counts remain zero.
  Fresh no-execute probes are still later (`cpu 1x2 20m` starts
  `2026-06-03T23:15:15`; `cpu 1x4 20m` starts `2026-06-04T01:15:27`), so no
  duplicate render was submitted. There is still no RGB-D method evidence.
- 2026-06-02 21:43+08:00: Repaired the current full96 chain's audit boundary
  before it could run. Submitted hardcoded aggregate `98909` and hardcoded
  exact96 structural inspection `98910`, updated visual gate `98907` to depend
  on `afterok:98910`, and canceled old aggregate/inspection jobs
  `98845/98846` before running because their submitted scripts required
  exported roots that are not visible from Slurm state. Then canceled old
  downstream jobs `98850-98858` before running because predicted-slot export
  metadata pointed to superseded RGB-D source job `98845`, and resubmitted
  downstream chain `98911 -> 98912 -> 98914 -> 98915 -> 98916 -> 98917` with
  diagnostic `98913` and video checks `98918/98919`. Slot training `98848`
  still depends on visual gate `98907`; `98914` requests one node / four H200
  GPUs for `03:30:00` with `MIN_TRAIN_SECONDS=10800`. This does not change
  data gates, visual gates, training floors, controller metrics, or success
  criteria; H5 counts are still zero and there is still no RGB-D method
  evidence.
- 2026-06-02 21:15+08:00: Current full96 RGB-D render branch remains pending
  with no RGB-D H5 files, no success/failed ledgers, and no Slurm logs for the
  current shard jobs. The eight render jobs `98524-98527` and `98841-98844`
  still have `ExcNodeList=(null)`, but the scheduler forecast slipped to
  `2026-06-03T00:01:27` on `server42`. Live node scan found most mixed nodes
  had all 8 GPUs allocated; only two nodes had four unallocated GPUs, and
  targeted no-execute probes on those nodes forecast `2026-06-05T11:44:08`.
  Generic no-execute probes were also later or invalid (`cpu 1x2 20m` starts
  `2026-06-03T23:24:15`, `cpu 1x4 20m` starts
  `2026-06-04T01:24:27`, debug hits `MaxGRESPerAccount`, test/gaosh/engram
  are invalid account/partition combos). No replacement render was submitted
  because it would not accelerate exact96 RGB-D data. There is still no RGB-D
  method evidence.
- 2026-06-02 21:18+08:00: Submitted snapshot audit completed. Actual Slurm
  fields for current render jobs are 1 node x 2 GPU with
  `ExcNodeList=(null)`; source wrappers have no fixed node exclusion. Frozen
  `scontrol write batch_script` text can still contain stale old `#SBATCH`
  defaults from submitted snapshots, so future scheduling must rely on current
  Slurm fields/source wrappers, not frozen historical script text. Added a
  retroactive submission-audit `submit_manifest.txt` and `submitted_jobs.tsv`
  under the late-half root so shard jobs `98841-98844` are auditable before
  they run. Method-chain `jobs.tsv` confirms formal downstream gates remain
  exact96 RGB-D, 4H200/3h for slot and world-model training, RGB-D-derived
  world-model evidence, `SLOT_SOURCE=rgbd`, and nonblank video review.
- 2026-06-02 21:11+08:00: Rechecked and reinforced the no-standing-bad-node
  rule after user direction. `docs/CLUSTER_NODES.md` is absent, `scripts/slurm`
  has no fixed `#SBATCH --exclude=...` node policy, `bash -n
  scripts/slurm/*.sbatch scripts/slurm/*.sh` passes, and old node-name entries
  in active TODOs are now explicitly historical scheduling records rather than
  reusable exclusions. This is scheduling hygiene only; RGB-D H5 counts remain
  zero and there is still no RGB-D method evidence. `scontrol show job` reports
  `ExcNodeList=(null)` for current render jobs `98524-98527` and
  `98841-98844`, aggregate/inspection/visual gates `98845-98847`, formal
  slot/world-model/video jobs `98848`, `98853`, `98856`, and audit `98884`.
  Fresh no-execute probes found no earlier legal replacement than the current
  `2026-06-02T22:01:15` render forecast: cpu 1x2 starts
  `2026-06-03T23:15:15`, cpu 1x4/1x8 start `2026-06-04T01:15:27`, and gpu
  1x2/1x4 start on `2026-06-06`, so no duplicate render was submitted.
- 2026-06-02 20:52+08:00: Submitted an earlier full96 aggregate RGB-D path:
  front-half `98524-98527` plus late-half `98841-98844`, aggregate `98845`,
  exact96 inspect `98846`, visual gate `98847`, and formal RGB-D method chain
  `98848-98858`. Cleared old `ExcNodeList` snapshots from current pending
  `wm_` jobs and verified `remaining_with_exclude=0`. Canceled superseded
  partial48 GPU method jobs `98760-98770` and old slower full96 4x4 jobs
  `97778-97781`, `97782`, `97811`, `98060-98071`. Current H5 counts remain
  zero, so there is still no RGB-D method evidence. Recheck at
  `2026-06-02T20:55+08:00`: all eight render jobs remained pending for
  `2026-06-02T22:01:15` on `server03`, with `ExcNodeList=(null)` and zero H5
  files; a tiny server03 canary dry-run forecast later than the render jobs, so
  it was not submitted. Recheck at `2026-06-02T20:59+08:00`: zero H5 files and
  no success/failed unit files; a non-executing 30-minute 8-shard 1x2-GPU
  dry-run forecast `2026-06-03T23:15:15`, so no replacement was submitted and
  the existing 45-minute jobs were left unchanged. Recheck at
  `2026-06-02T21:01+08:00`: all eight render jobs still pending, zero H5 files,
  no success/failed unit files. Targeted 1x2-GPU dry-runs on `server39` and
  `server56` forecast `2026-06-05T11:44:08`, so no replacement shard was
  submitted. A lightweight source-H5 metadata preflight confirmed 6 source H5
  files, 96 total work units, 8 shards x 12 units, front/late halves of 48
  each, and zero overlap. Added late-half render diagnostic audit job `98884`
  afterany `98841-98844`, matching front-half audit `98529`; this is
  failure-localization only and does not alter gates. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_full96_aggregate_latehalf_chain_2052.md`.
- 2026-06-02 20:46+08:00: Deleted the standing node-list file
  `docs/CLUSTER_NODES.md`, removed fixed wrapper-source
  `#SBATCH --exclude=server16,server29` defaults, and changed
  `submit_rgbd_distributed_shards.sh` so node exclusions are empty by default
  and only passed when explicitly set from live evidence. This is scheduling
  hygiene only; submitted Slurm snapshots are unchanged and there is still no
  RGB-D method evidence. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_delete_standing_node_list.md`.
- 2026-06-02 20:07+08:00: Removed the static render-node exclusion policy from
  active entry points; future scheduling uses live Slurm state plus targeted
  node canaries, with no default node exclusion list in wrapper source.
  Partial48 RGB-D shard0 job `98209` completed on `server20` with 12
  RGB-D H5 files, 12 contact sheets, 12 preview videos, and 0 failed work
  units. Local structure and visual checks passed with 0 warnings, and the
  review sheet was opened directly. This is shard0 RGB-D data-quality evidence
  only, not method evidence. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_live_node_policy_and_partial48_shard0.md`.
- 2026-06-01: Added `scripts/world_model/collect_dynamic_state_rollouts.py`.
- 2026-06-01: Added `scripts/slurm/collect_dynamic_state_rollouts.sbatch`.
- 2026-06-01: Added RGB-D data-generation and dense multi-GPU allocation
  constraints to `AGENTS.md` and the data TODOs.
- 2026-06-01: Lightweight checks passed: Python compile, sbatch syntax, and
  CLI help. No heavy rollout was run on the login node.
- 2026-06-01: Slurm smoke `94220` completed on `server04`; dynamic
  `hole_move_stop` moved the hole by `0.14m` in y and frozen base DP failed
  after the perturbation, giving expected OOD data.
- 2026-06-02: Added dynamic RGB-D renderer, single-GPU and dense Slurm RGB-D
  wrappers, plus official demo RGB-D replay wrapper.
- 2026-06-02: RGB-D smoke `94241` completed on `server04`; output contains RGB,
  depth, camera params, actions, slots, and frame-aligned env states. Contact
  sheet was inspected.
- 2026-06-02: Official demo RGB-D wrapper smoke first failed on HDF5 file
  locking (`94247`), then passed after adding `HDF5_USE_FILE_LOCKING=FALSE`
  (`94253`).
- 2026-06-02: Dynamic smoke rollouts for `none`, `hole_constant`,
  `hole_reverse`, `peg_disturb`, and `peg_drop` completed. Validator and RGB-D
  contact sheets were generated for all six smoke scenarios.
- 2026-06-02: RGB-D render jobs exposed `server04` `vk::DeviceLost`; retrying
  on `server27` succeeded, and `server04` was added to render exclusions.
- 2026-06-02: Added static-hole and constant-velocity future hole prediction
  baselines. Results show CV is strong on bounded constant motion, overshoots
  move-stop at long horizons, and leaves room for learned stop/reversal and
  uncertainty modeling.
- 2026-06-02: Added object-slot dataset utilities and the first lightweight
  Transformer world-model training script with trajectory-level split,
  validation dropout uncertainty, best/final checkpoints, and prediction
  examples. Slurm smoke job `94371` was submitted; do not count it as evidence
  until metrics are written and inspected.
- 2026-06-02: Added first frozen-DP continuability labeler and Slurm wrapper.
  It resets candidate env states, synchronizes saved DP obs history, optionally
  replays recorded external perturbation deltas, and writes empirical labels.
  Slurm smoke job `94432` was submitted; do not count it as label evidence
  until output H5/JSON are inspected.
- 2026-06-02: Added training minimum to `AGENTS.md`: model training must use at
  least 4 H200 GPUs and reserve at least 3 hours unless explicitly overridden.
  Job `94371` is therefore recorded only as an undersized code-path smoke, not
  as training evidence. Submitted compliant 4-GPU H200 ensemble job `94442`,
  which later completed on recovered historical node-observation member `server53` and
  passed compliance inspection.
- 2026-06-02: `94432` continuability smoke completed on `server27`; inspection
  found 8 candidates, 50% post-trigger coverage, and 1/8 success. This proves
  the label path works but is too small and imbalanced for `C_pi` training.
- 2026-06-02: Added `C_pi` training script and 4-GPU H200 ensemble wrapper.
  The wrapper refuses uninspected/tiny label shards by default, preventing the
  8-sample smoke from becoming a misleading model result.
- 2026-06-02: Submitted larger continuability label pilot `94445` with
  `RUN_GROUP=pilot128`, `MAX_CANDIDATES=128`, and `MAX_ROLLOUT_STEPS=120`.
  Malformed path attempt `94444` was canceled and recorded, not hidden.
- 2026-06-02: `94445` completed on `server27` with 128 labels, 60.9%
  post-trigger coverage, and 28/128 successes. Inspection passed as pilot data.
  Submitted compliant 4-GPU H200 `C_pi` training job `94471`; it is pending
  due to priority at last update.
- 2026-06-02: Added dense 4-GPU dynamic state rollout wrapper and submitted job
  `94510` for a larger non-smoke multi-scenario shard. This prepares enough H5
  sources for larger `C_pi` labels and later RGB-D export.
- 2026-06-02: Added first oracle-slot receding-horizon rebinding evaluator,
  controller-run inspector, and Slurm wrapper. Submitted video smoke `94518`.
  The smoke uses CV prediction and no learned `C_pi` handoff, so it validates
  plumbing and logs rather than final method performance.
- 2026-06-02: Added `last_observation_uncertainty` no-future-leakage baseline
  and reran baseline evaluation under
  `experiments/world_model_task_rebinding/hole_prediction_baselines/with_lastobs_uncertainty`.
- 2026-06-02: Added `CONTROL_POLICY=track_current` for simple current-target
  tracking. Submitted no-video comparison smokes `94521` (`track_current`) and
  `94522` (`dp_only`).
- 2026-06-02: Added `C_pi` threshold calibration tool. It uses held-out member
  validation predictions and refuses incomplete ensemble members by default, so
  it cannot turn an unfinished `94471` run into handoff evidence.
- 2026-06-02: Added rollout-dir Slurm wrappers for downstream scale-up:
  `train_object_state_world_model_from_rollout_dir_4gpu.sbatch`,
  `label_continuability_from_rollout_dir_4gpu.sbatch`, and
  `render_dynamic_rgbd_dataset_from_rollout_dir_dense.sbatch`.
- 2026-06-02: Queued dependency chain after dynamic shard `94510`: object-state
  training `94539`, larger `C_pi` labels `94540`, RGB-D export `94541`, and
  larger-label `C_pi` training `94542`.
- 2026-06-02: Added explicit safe-retreat and dropped-peg regrasp primitives to
  the controller scaffold. Submitted no-video Slurm smoke `94550` for runtime
  validation; it is not method success evidence.
- 2026-06-02: Hardened online `C_pi` use in the controller: model manifests
  must match the expected feature names, normalizer dimensions are checked, and
  ensemble disagreement is logged/gated by `CPI_MAX_STD`.
- 2026-06-02: Added `scripts/world_model/inspect_rgbd_dataset.py` and inspected
  all existing dynamic RGB-D smoke files. The smoke RGB-D dataset has 8 files,
  8 trajectories, 124 RGB-D frames, both base/hand cameras, camera parameters,
  slots, actions, and env states with zero inspection warnings.
- 2026-06-02: Added `rebind_world_model` controller path. It loads an
  object-state world-model ensemble only from complete checkpoint directories,
  checks feature/target contracts, predicts future hole task frames online, and
  logs prediction source, model horizon, and uncertainty in controller events.
  Streaming discrepancy now compares against a separate one-step prediction
  rather than the chosen future bridge horizon.
- 2026-06-02: Enforced the user-requested training floor operationally:
  alternate node-selection route retry `94809` is queued with 4 H200 GPUs,
  `MIN_TRAIN_SECONDS=10800`, and compliance inspection `94810`. No current
  schedulable node has 4 free H200 GPUs; recovered nodes with earlier rendering failures are
  eligible for non-rendering training retries.
- 2026-06-02: Added orientation-aware TCP-continuation bridge. It keeps
  position-only behavior as `bridge_orientation_reference=none`, and new
  validation jobs explicitly set `BRIDGE_ORIENTATION_REFERENCE=tcp_continuation`.
- 2026-06-02: Queued dependent learned-world-model video smoke `94564`
  (`afterok:94442`). A lightweight offline load check passed using undersized
  smoke model `94371`; this only validates the interface and is not method
  evidence.
- 2026-06-02: After hardening inspection gates, updated pending
  learned-world-model controller dependencies: `94564`, `94576`, and `94577`
  now depend on successful object-model inspection `94620`, while `94578`
  depends on both object-model inspection `94620` and `C_pi` inspection
  `94618`.
- 2026-06-02: Dynamic shard `94510` completed and passed validation: 96
  trajectories across `none`, moving-hole, peg-disturb, and peg-drop families;
  all dynamic cases triggered before first insertion, with zero validator
  warnings. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_dynamic_shard_94510.md`.
- 2026-06-02: Controller smokes `94519`/`94521` exposed a bridge-scoring
  failure: the planner always selected horizon `0`, so `rebind_cv` collapsed
  to current-frame tracking. The score was changed to account for physical
  travel budget over each candidate horizon. Same-seed no-video smoke `94591`
  later completed and provided the scoring validation; pending v2 jobs
  `94582`, `94583`, and `94590` were canceled after the bridge target changed
  to TCP-continuation.
- 2026-06-02: Added root `idea.md` as a stable lowercase entry point pointing
  to the high-level method note, active TODOs, and current evidence notes.
- 2026-06-02: Added focused evidence note for early controller smokes
  `94518`/`94519`/`94521`/`94522`, separating base-DP grasp precondition
  failure, horizon-0 bridge collapse, `track_current`, and `dp_only` failures:
  `docs/world_model_task_rebinding/2026-06-02_controller_smoke_94518_94522.md`.
- 2026-06-02: Queue check confirmed compliant training started, not just
  submitted: object world model `94442` ran on `server53` and `C_pi` `94471`
  ran on `server04`, both with 4 H200 GPUs and `MIN_TRAIN_SECONDS=10800`.
  Both later completed and passed compliance inspection. Status note:
  `docs/world_model_task_rebinding/2026-06-02_training_and_scaleup.md`.
- 2026-06-02: Added CPU-partition post-training inspection wrappers and queued
  dependent inspection jobs: `94620` after `94442`, `94618` after `94471`, and
  originally `94619` after `94539`. The `94619` submitted snapshot was later
  replaced by strict inspection `94863`. These wrappers expose
  missing-output/non-compliance failures but do not count as manipulation
  success evidence.
- 2026-06-02: Hardened the object-state and `C_pi` ensemble inspectors so
  `compliant_training_evidence=true` requires at least four complete members,
  final checkpoints/metrics, minimum member elapsed time `>=10800` seconds,
  and a Slurm manifest request for at least four `NVIDIAH200` GPUs. The CPU
  inspection wrappers now default to failing nonzero when that gate is false,
  so downstream controller jobs can depend on inspection success instead of
  raw training-job completion.
- 2026-06-02: Hardened elapsed-time accounting: future training writes
  `total_elapsed_seconds`, and inspectors also read member start/complete
  timestamps so a compliant run is not falsely rejected because the final
  periodic history row was logged before `10800s`.
- 2026-06-02: Dropped-peg regrasp smoke `94550` completed and failed with a
  clear primitive issue: `212` regrasp actions cycled between approach and
  descend, never close/lift, and final grasp was false. Fixed the regrasp rule
  and queued same-seed v2 smoke `94622` after canceling too-tight pending job
  `94621` before it ran. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_regrasp_smoke_94550.md`.
- 2026-06-02: Short-allocation v2 job `94626` ran immediately and exposed the
  next primitive issue: close/lift occurred but did not create a grasp because
  close and lift were coupled. The primitive now close-holds in place until
  `grasped=True`; this still requires Slurm smoke and video evidence.
- 2026-06-02: Short-allocation v3 job `94633` showed close-hold alone was not
  enough: the gripper closed while the TCP was roughly `5cm` above the peg.
  Updated regrasp to target the full peg-local TCP pose measured from
  successful grasp frames and queued v4 smoke `94645`.
- 2026-06-02: Added `scripts/slurm/inspect_rebinding_controller_run.sbatch`
  and queued CPU inspection job `94656` after v4 regrasp smoke `94645`, so the
  no-video controller result is inspected automatically without using login
  CPU for H5 analysis.
- 2026-06-02: Added `scripts/slurm/calibrate_continuability_threshold.sbatch`.
  Queued threshold calibration `94659` after pilot `C_pi` inspection `94618`,
  larger `C_pi` inspection `94658` after training `94542`, and larger
  calibration `94660` after `94658`.
- 2026-06-02: Created
  `experiments/world_model_task_rebinding/rebinding_controller/fullshard_world_model_dirs_job94539.txt`
  and queued full-shard learned-world-model controller smokes behind
  inspection `94619`. The initial pending `94661`-`94664` chain was later
  canceled before running and replaced by an explicit TCP-continuation gated
  chain, which was itself canceled before running after orientation analysis.
  The first orientation-aware `94539` chain `94830`-`94834` was also canceled
  before running because its upstream inspection `94619` lacked the strict
  compliance gate. Current full-shard controller chains are orientation-aware
  jobs `94865`-`94869` for `94539`, plus `94835`-`94849` for the other
  object-WM duplicates.
- 2026-06-02: Added `scripts/slurm/inspect_rgbd_dataset.sbatch` and queued
  CPU inspection job `94665` after dense full-shard RGB-D export `94541`.
  The old submitted snapshot was later replaced by strict no-warning inspection
  `94858`.
- 2026-06-02: Added
  `scripts/world_model/evaluate_object_world_model_ensemble.py` and
  `scripts/slurm/evaluate_object_world_model_ensemble.sbatch`; queued
  learned-vs-CV/static prediction evaluation job `94672` after `94620`. The
  first larger-shard eval `94673` after `94619` was later canceled before
  running with the old non-strict inspection chain; current larger-shard eval
  is `94864` after strict inspection `94863`.
- 2026-06-02: Queued dense RGB-D backfill `94676` for the full dynamic shard:
  2 nodes, 16 H200 GPUs total, 8 GPUs per node, 2-hour limit, same rollout
  source as `94541`. Strict inspection job `94859` waits on `afterany:94676`;
  old inspection `94677` was canceled before running because its submitted
  script snapshot did not include the no-warning gate.
- 2026-06-02: Queued full-shard object-world-model backfill `94679`: 4 H200
  GPUs on one node, `MIN_TRAIN_SECONDS=10800`, 3:15 Slurm limit to improve
  backfill chances without going below the training-evidence floor.
  Inspection `94680` and prediction evaluation `94681` depend on it.
- 2026-06-02: Queued larger `C_pi` backfill chain: label job `94682`, 4-GPU
  compliant training job `94683` with `MIN_LABELS=512` and
  `MIN_TRAIN_SECONDS=10800`, inspection `94684`, and threshold calibration
  `94685`. This chain was later canceled at `10:21+08:00` after primary label
  job `94540` completed and primary 4-H200 training job `94542` started, so it
  would have been duplicate GPU work.
- 2026-06-02: Object world model `94442` completed as a compliant 4-H200,
  3-hour training run; inspection `94620` passed and prediction evaluation
  `94672` completed. It beats static and CV on `hole_move_stop` and beats CV at
  long horizons overall, but does not beat CV at short horizons. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_object_wm_94442.md`.
- 2026-06-02: Learned-world-model controller video smoke `94564` completed and
  failed after direct contact-sheet inspection. The likely controller bug is
  treating three 2-frame `grasped=False` flickers as persistent peg loss,
  causing safe retreats before insertion. Added `grasp_lost_grace_steps` and
  queued same-seed no-video validation `94698` with CPU inspection `94699`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_learned_wm_controller_94564.md`.
- 2026-06-02: Added CPU success gate
  `scripts/slurm/gate_rebinding_controller_success.sbatch`. Queued gate
  `94724` after debounce inspection `94699`, same-seed video job `94725` after
  the gate, and inspection `94726` after video completion. This preserves the
  evaluation metric and only automates required video evidence when the
  no-video dynamic run actually succeeds.
- 2026-06-02: First compliant `C_pi` training job `94471` completed on 4 H200s
  after `03:00:20`; inspection `94618` passed. Calibration `94659` exposed a
  Markdown report bug; fixed rerun `94716` completed. Held-out conservative
  thresholds have zero recall, so the pilot proves the path but must not be
  used as a useful handoff gate. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_cpi_94471.md`.
- 2026-06-02: Learned-world-model video smokes `94576` (`hole_constant`) and
  `94577` (`hole_reverse`) completed, failed final insertion, were inspected
  by CPU jobs `94700`/`94717`, and contact sheets were opened directly. Both
  reinforce the peg-loss/safe-retreat controller gap. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_learned_wm_controller_94576_94577.md`.
- 2026-06-02: Queue/node-specific check at `05:01+08:00` found no usable node with
  at least four free H200 GPUs; only drained/down nodes had four or more free
  GPUs. Pending training jobs already allow recovered nodes with earlier rendering failures
  for non-rendering training and do not lower the 4-H200/3-hour standard.
- 2026-06-02: Queue check at `05:10+08:00` found no new completed controller,
  RGB-D, label, or training results. `94591`, `94645`, and `94698` are still
  pending on priority with relaxed non-rendering exclusions; `94539` remains
  the earliest scheduled compliant larger-shard object-world-model training
  run.
- 2026-06-02: Added RGB-D slot extractor data/model/inspection path and
  4-H200 Slurm wrapper. Lightweight compile, wrapper syntax, loader shape
  check, and zero-epoch CPU forward/eval dry-run passed. Canceled mistaken
  path-hygiene submission `94732`; the initially queued chain
  `94676 -> 94677 -> 94733 -> 94734` was later canceled before running after
  preflight found missing RGB-D input/inspection exports. Current corrected
  chain is `94676 -> 94859 -> 94860 -> 94861`, where `94860` trains only after
  inspected full-shard RGB-D exists and still requires 4 H200 GPUs plus
  `MIN_TRAIN_SECONDS=10800`. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_rgbd_slot_extractor.md`.
- 2026-06-02: Follow-up controller runs `94591`, `94698`, and `94645`
  completed and were inspected. They are failure/diagnostic evidence, not
  success evidence. `94591` showed fixed-score CV bridge no longer retreats
  but still fails with final `YZ=0.0871m`; `94698` showed learned-WM debounce
  still fails with final `YZ=0.1195m`; `94645` showed regrasp close does not
  create a stable grasp. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_controller_followups_94591_94698_94645.md`.
- 2026-06-02: Added `bridge_servo_reference=tcp_continuation` using
  data-measured hole-local TCP continuation poses from successful static
  rollouts. This preserves the final inserted-state metric and keeps the
  bridge in task-frame rebinding rather than scenario branching. Rationale:
  `docs/world_model_task_rebinding/2026-06-02_bridge_continuation_pose_fix.md`.
- 2026-06-02: Canceled stale pre-continuation controller jobs `94578`,
  `94582`, `94583`, and `94590`. Initial 3-hour controller-smoke chain
  `94751`-`94760` was also canceled before running and replaced by shorter
  non-training validation chains `94763`-`94772` and peg-drop chain
  `94773`-`94777`. At `05:47+08:00`, these new controller validations were
  still pending on Slurm priority and had produced no new metrics or videos.
- 2026-06-02: Training-floor check at `06:38+08:00` found no user jobs running
  yet. Active model-training jobs still meet the floor: object-WM jobs
  `94539`, `94679`, `94746`, and alternate node-selection retry `94809` request one node with
  4 `NVIDIAH200` GPUs and `MIN_TRAIN_SECONDS=10800`; C_pi training
  `94542`/`94683` and RGB-D slot training `94860` use the same 4-H200,
  10800-second hard gate. The earliest object-WM forecast remains `94539` on
  `server42` at `2026-06-03T00:01:27`. Dense RGB-D generation is queued as
  one 8-GPU node (`94541`) and two 8-GPU nodes (`94676`), avoiding one-GPU-per-
  node waste.
- 2026-06-02: Node scan at `06:41+08:00` found no schedulable node with free
  H200 GPUs; all non-drain H200 nodes report `gres/gpu=8` allocated, while
  `server16`, `server29`, and `server61` are down/drain. A new same-spec
  training duplicate would start later than existing jobs, so no lower-quality
  or extra noisy submission was added.
- 2026-06-02: Queue/config audit at `06:44+08:00` found no new completed
  results. Same-seed orientation-aware no-video jobs `94814`/`94819`/`94824`
  are still forecast for `2026-06-02T10:00:12`; their `SubmitLine` exports
  confirm `BRIDGE_SERVO_REFERENCE=tcp_continuation`,
  `BRIDGE_ORIENTATION_REFERENCE=tcp_continuation`, and `SAVE_VIDEO=false`.
  Their gated video jobs `94817`/`94822`/`94827` depend on success gates and
  export `SAVE_VIDEO=true`. Full-shard controller no-video jobs
  `94830`/`94835`/`94840`/`94845` and gated video jobs
  `94833`/`94838`/`94843`/`94848` were also confirmed orientation-aware at that
  time; the `94830`-`94834` branch was later canceled before running and
  replaced by `94865`-`94869` after strict inspection `94863`.
  Node-risk object-WM retry `94809` forecast improved to
  `2026-06-04T13:40:25`.
- 2026-06-02: Preflight at `06:47+08:00` found RGB-D slot chain
  `94733`/`94734` would fail because the training job lacked `INPUT_RGBD_DIR`
  and the inspection job lacked `ENSEMBLE_DIR`. Both were canceled before
  running and initially replaced by `94856 -> 94857`. That intermediate chain
  was also canceled before running after discovering `94677` used an old Slurm
  script snapshot without the no-warning gate. Current chain is
  `94859 -> 94860 -> 94861`: `94860` depends on successful strict RGB-D
  inspection `94859`, exports
  `INPUT_RGBD_DIR=/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/rgbd_dynamic_dense/from_rollout_dir/job94676`,
  requests 4 H200 GPUs for `03:30:00`, and keeps
  `MIN_TRAIN_SECONDS=10800`; `94861` inspects
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job94860`
  with `REQUIRE_COMPLIANT=true`. A fresh `squeue --start` after the requeue
  showed forecast volatility: `94676` was back to `2026-06-03T22:54:26` and
  `94809` was back to `2026-06-05T22:48:07`, so start-time forecasts remain
  scheduling metadata only.
- 2026-06-02: RGB-D data gate preflight at `06:53+08:00` tightened
  `inspect_rgbd_dataset.sbatch`: it now defaults to
  `REQUIRE_NO_WARNINGS=true`. Because pending `94665`/`94677` had already
  snapshotted the old script, they were canceled before running and replaced
  by strict inspections `94858`/`94859`. `94859` exits nonzero on RGB-D camera,
  alignment, render, depth, or slot warnings and will not release `94860`.
  `bash -n` passed, and existing smoke RGB-D data with `base_camera` and
  `hand_camera` reports `num_warnings=0`.
- 2026-06-02: Object-WM gate preflight at `07:01+08:00` found pending
  `94539` inspection `94619` used an old Slurm script snapshot without the
  `REQUIRE_COMPLIANT` post-training gate. Canceled `94619`, downstream eval
  `94673`, and dependent controller chain `94830`-`94834` before running.
  Requeued strict chain `94863 -> 94864` for prediction evaluation and
  orientation-aware controller chain `94865 -> 94866 -> 94867 -> 94868 ->
  94869`, all still depending on successful compliant inspection of
  `94539`.
- 2026-06-02: Queue/resource audit at `07:08+08:00`: no user jobs are running.
  `sacct` confirms old non-strict jobs `94619`, `94673`, `94830`-`94834`,
  `94665`, `94677`, and `94856`/`94857` were canceled before allocation, while
  strict replacements `94858`-`94861` and `94863`-`94869` remain pending.
  `squeue --start` forecasts same-seed orientation-aware controller jobs
  `94814`/`94819`/`94824` on `2026-06-02T08:16:34`, first 4-H200 object-WM
  training `94539` on `2026-06-03T00:01:27`, dense RGB-D export `94541` on
  `2026-06-03T06:01:30`, dense 2-node RGB-D backfill `94676` on
  `2026-06-04T05:00:00`, and node-specific/limited-route object-WM retry `94809` on
  `2026-06-04T13:40:25`. Current H200 node scan still shows `server16`,
  `server29`, and `server61` in drain/down-like states, so there is no recovered
  node-specific node being missed.
- 2026-06-02: Submitted-script audit at `07:10+08:00`: no new result has
  completed. Frozen Slurm scripts and `SubmitLine` exports confirm the pending
  orientation-aware controller jobs use TCP-position and TCP-orientation
  continuation, success-gated video jobs, strict RGB-D no-warning inspection,
  and 4-H200/10800-second RGB-D slot training. The old alternate node-selection retry `94809`
  uses a historical output directory name consistently across training,
  inspection, evaluation, and controller dir files.
- 2026-06-02: Scheduling probe at `07:13+08:00`: all key jobs remain pending.
  Re-read the active PLAN files for overview, object-WM, controller, RGB-D, and
  experiment matrix. New `sbatch --test-only` probes for additional compliant
  one-node 4-H200 object-WM duplicates with `03:05:00`/`03:10:00` walltime and
  `MIN_TRAIN_SECONDS=10800` forecast `2026-06-09T21:24:12` on both the normal
  route and the alternate node-selection route, later than existing `94539`, so no extra
  duplicate was submitted.
- 2026-06-02: Controller evidence-chain audit at `07:16+08:00`: no key job has
  started or completed. The controller writes MP4 plus contact sheet when video
  is enabled; inspection exposes both paths and separately reports
  `success_after_dynamic_event_count`; success gates only release videos after
  inspected no-video final dynamic success. This preserves the rule that final
  manipulation success still requires direct video/contact-sheet inspection if
  a video job runs.
- 2026-06-02: Added controller-inspection summary utility at `07:18+08:00` and
  queued CPU summary jobs: no-video summary `94882` after
  `94815`/`94820`/`94825`, and initially video summary `94881` after
  `94818`/`94823`/`94828`. `94881` was later canceled before running and
  replaced by `94889` with `REQUIRE_ALL=false`, because gated-video inspection
  files are expected to be absent when no-video success gates fail. The
  summaries organize inspection JSONs into
  comparison tables only; they do not replace success gates or direct
  video/contact-sheet review.
- 2026-06-02: Added full-shard controller summary inputs at `07:23+08:00` and
  queued CPU summaries `94885` after no-video inspections
  `94866`/`94836`/`94841`/`94846`, and initially `94886` after gated-video
  inspections `94869`/`94839`/`94844`/`94849`. `94886` was later canceled
  before running and replaced by `94890` with `REQUIRE_ALL=false`. These are
  evidence organization jobs only.
- 2026-06-02: Training-floor audit at `07:29+08:00`: all model-training
  evidence must still reserve one node with at least 4 H200 GPUs and
  `MIN_TRAIN_SECONDS>=10800`; shorter runs are smoke/debug only and cannot be
  used to reject the direction. `94539`, `94679`, `94746`, `94809`, dependent
  `C_pi` training `94542`/`94683`, and RGB-D slot training `94860` all request
  `gres/gpu=4` and satisfy this floor. `squeue --start` forecasts the earliest
  object-WM training `94539` at `2026-06-03T00:01:27`; no training job is
  currently running.
- 2026-06-02: Node-risk recovery probe at `07:31+08:00`: `AllocTRES` shows no
  schedulable node with four free H200 GPUs. `server27` has only one free GPU,
  `server31` has two free GPUs, and `server16`/`server29`/`server61` are
  down/drain or drain, so Slurm rejects drain-node-only probes with
  `Requested node configuration is not available`. Normal-route and old
  alternate node-selection route `sbatch --test-only` probes for extra compliant object-WM
  duplicates both forecast `2026-06-09T21:01:12`, later than existing `94539`;
  no lower-standard or later duplicate training was submitted.
- 2026-06-02: Queue continuation audit at `07:34+08:00`: no new current-method
  result or video artifact has appeared since the prior TCP-continuation
  failures. `94814`/`94819`/`94824` are still pending but Slurm moved their
  forecast to `2026-06-02T19:28:40`; `94539` remains the earliest formal
  object-WM training forecast at `2026-06-03T00:01:27`. Shorter 10/15-minute
  same-settings no-video controller `sbatch --test-only` probes forecast
  `2026-06-09T19:27:41`, much later than the existing queued controller jobs,
  so no duplicate controller backfill was submitted.
- 2026-06-02: Follow-up queue check at `07:36+08:00`: Slurm moved
  `94814`/`94819`/`94824` back earlier to `2026-06-02T10:00:12`, dense RGB-D
  backfill `94676` to `2026-06-03T22:07:26`, fast-fit object-WM `94746` to
  `2026-06-04T04:00:00`, and live-node retry `94809` to
  `2026-06-04T16:00:00`. No user job is running; these are volatile forecasts,
  not results.
- 2026-06-02: External-state audit at `07:37+08:00`: no new current-method
  inspection, metrics, MP4, or contact sheet has appeared; recent artifacts are
  still the earlier `94763`/`94764`/`94773` TCP-continuation failures. All
  schedulable H200 nodes now report `gres/gpu=8` allocated; `server16` and
  `server29` remain down/drain, and `server61` remains drain. Pending jobs are
  waiting on Slurm priority, not on a local dependency or script error.
- 2026-06-02: Status check at `07:39+08:00`: `94814`/`94819`/`94824` remain
  pending for the orientation-aware controller validation, still forecast at
  `2026-06-02T10:00:12`. Larger object-WM training `94539` remains pending at
  `2026-06-03T00:01:27`. No current-method artifact newer than the recorded
  TCP-continuation failures exists, so there is no result to interpret yet.
- 2026-06-02: Added gated-video cleanup helper and queued cleanup jobs at
  `07:45+08:00`: `94929` for the current orientation-aware controller family
  after gates `94816`/`94821`/`94826`, and `94930` for full-shard controller
  branches after gates `94867`/`94837`/`94842`/`94847`. They only cancel dead
  gated-video branches after failed no-video gates; they do not touch
  no-video evidence, success gates, metrics, or video review requirements.
- 2026-06-02: Larger-shard object-WM training `94539` started at
  `08:10:47+08:00` on `server21`. `scontrol` reports one node, 4 H200 GPUs,
  `TimeLimit=03:30:00`, and `TRES=...gres/gpu=4`; the job manifest records
  `MIN_TRAIN_SECONDS=10800` and four ensemble members. This is the active
  formal training candidate; it is not evidence until strict inspection
  `94863` passes after completion.
- 2026-06-02: Rechecked nodes with earlier rendering failures after the training-floor
  reminder. They are either down/drain (`server16`, `server29`) or have all
  GPUs already allocated; test-only 4-H200 probes on recovered old node-specific nodes
  forecast starts much later than the already-running `94539`. No undersized
  or later duplicate training job was submitted.
- 2026-06-02: Orientation-aware bridge v1 same-seed jobs `94814`/`94819`/`94824`
  completed and failed the unchanged final dynamic success gate. Evidence note
  `docs/world_model_task_rebinding/2026-06-02_orientation_phase_search_failure.md`
  records the physical failure: CV/learned-WM reached near-center y/z but
  stayed at pre-insert x around `-0.11m`, with event-log phases dominated by
  `align`.
- 2026-06-02: Implemented phase-search rebinding v2. `_choose_bridge` now
  scores both `align` and `insert` phase candidates over the same tau
  candidates, with continuous progress and phase-entry costs. This implements
  the planned `argmax over tau, bridge, phase` and preserves the official
  final inserted-state metric. Inspector/summary now report bridge phase
  counts. Documentation:
  `docs/world_model_task_rebinding/2026-06-02_phase_search_v2.md`.
- 2026-06-02: Submitted v2 same-seed controller evidence chains:
  CV `94981 -> 94982 -> 94983 -> 94984 -> 94985`, learned-WM `94986 -> 94987
  -> 94988 -> 94989 -> 94990`, and peg-drop/regrasp `94991 -> 94992 -> 94993
  -> 94994 -> 94995`. Summary/cleanup jobs are `94996`/`94997`/`94998`.
- 2026-06-02: Canceled stale pending full-shard v1 controller branches
  `94865`-`94869`, `94835`-`94839`, `94840`-`94844`, `94845`-`94849`, plus old
  summaries/cleanup `94885`, `94890`, and `94930`. Training, strict
  object-WM inspections/evals, RGB-D, and `C_pi` jobs were not canceled.
  Replacement phase-search full-shard controller branches are
  `94999`-`95003` for `94539`, `95004`-`95008` for `94679`, `95009`-`95013`
  for `94746`, and `95014`-`95018` for `94809`; summaries/cleanup are
  `95019`/`95020`/`95021`.
- 2026-06-02: Phase-search v2 same-seed jobs `94981`/`94986`/`94991` completed
  and failed final dynamic insertion. This is recorded in
  `docs/world_model_task_rebinding/2026-06-02_phase_search_v2_failure_and_hybrid.md`.
  V2 fixed phase selection (`insert` dominated in moving-hole branches), but
  the insertion primitive did not move the peg head to the official
  `x>=-0.015` success region. Implemented `bridge_servo_reference=phase_hybrid`:
  align uses TCP continuation, insert closes the position loop on peg-head
  task-frame error while retaining TCP orientation control.
- 2026-06-02: Submitted hybrid v3 moving-hole chains: CV
  `95026 -> 95027 -> 95028 -> 95029 -> 95030`, learned-WM
  `95031 -> 95032 -> 95033 -> 95034 -> 95035`, with summary/cleanup
  `95036`/`95037`/`95038`. Peg-drop is not resubmitted for hybrid yet because
  v2 evidence shows it remains in align after regrasp due to y/z around
  `0.062m`; that is a regrasp/alignment issue, not an insert-servo issue.
- 2026-06-02: Hybrid v3 no-video jobs `95026`/`95031` completed and failed the
  unchanged final dynamic insertion gate. They did enter insert and used
  peg-head servo, but direct 3D target chasing still saturated actions and did
  not make monotonic hole-axis progress. Added
  `bridge_servo_mode=task_frame_projected`: planner still searches tau/phase,
  while the bridge executes bounded task-frame local steps with one-step
  hole-motion feedforward and no default insertion-axis retreat. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_task_frame_projected_servo_v4.md`.
- 2026-06-02: Submitted task-frame projected v4 controller chains. No-video:
  CV `95039 -> 95040 -> 95041`, learned-WM `95042 -> 95043 -> 95044`,
  summary `95049`. Gated video: CV `95045 -> 95046`, learned-WM
  `95047 -> 95048`, tolerant video summary `95050`. These are controller
  diagnostics, not model-training evidence; success still requires final
  dynamic insertion, passed gate, and direct video/contact-sheet inspection if
  videos are produced.
- 2026-06-02: Canceled stale pending full-shard phase-search controller chains
  `94999`-`95021` before execution because they did not include v4
  `bridge_servo_mode=task_frame_projected`. Replaced them with full-shard v4
  task-frame projected chains behind the same strict object-WM inspections:
  `95055`-`95059` for `94539`, `95060`-`95064` for `94679`,
  `95065`-`95069` for `94746`, and `95070`-`95074` for `94809`; summaries and
  cleanup are `95075`/`95076`/`95077`.
- 2026-06-02 08:53: Converted backup object-WM training jobs to strict
  failover dependencies instead of parallel H200 consumption. Primary job
  `94539` remains running on 4 H200 GPUs; backup `94679` now waits on
  `afternotok:94863`, backup `94746` waits on `afternotok:94680`, and old
  alternate node-selection backup `94809` waits on `afternotok:94747`. This keeps the
  compliant backups available if an inspection fails while avoiding wasted
  4-H200 training when the primary run passes.
- 2026-06-02 08:55: Slurm audit confirms `94539` is still running on
  `server21` with `gres/gpu=4`, `TresPerNode=gres:gpu:4`, and
  `TimeLimit=03:30:00`; runtime is still below the 3-hour evidence floor.
  The job directory has four member manifests and intermediate `best_model.pt`
  files, and the Slurm log continues to print epoch progress. This is
  liveness only, not success/failure evidence.
- 2026-06-02 08:58: Re-read `AGENTS.md`, active TODOs, and current v4
  controller entries before touching the queue. `94539` remains running below
  the 3-hour evidence floor; `95039`/`95042` remain priority-pending with
  forecast `2026-06-02T10:00:12`; RGB-D dense jobs `94541`/`94676` and larger
  `C_pi` jobs remain queued. `sacct SubmitLine` confirms same-seed and
  full-shard v4 controller jobs use
  `BRIDGE_SERVO_MODE=task_frame_projected`, `BRIDGE_PHASE_MODE=search`, and
  `TASK_SERVO_ALLOW_AXIS_RETREAT=false`. Video jobs are gated after success
  jobs and inherit the conservative rendering node-specific exclusions from
  `evaluate_rebinding_controller.sbatch`.
- 2026-06-02 09:01: No new v4 task-frame-projected metrics, inspection JSON,
  MP4, or contact sheet exists yet. `94539` is still running at about 50
  minutes, so it is still below the training evidence floor. Rechecked the
  v4 inspection/gate path: the success gate requires final success after a
  prior dynamic event, while the inspector records bridge phase, servo mode,
  active servo reference, axis/lateral task-frame errors and steps, final
  peg-head x, and final/min YZ after trigger. This is sufficient for the next
  no-video failure analysis without changing the final inserted-state metric.
- 2026-06-02 09:03: Added same-seed v4 gated-video cleanup job `95086` and
  branch file
  `experiments/world_model_task_rebinding/rebinding_controller/controller_smoke_v7_task_frame_projected_video_branches.txt`.
  It waits on gates `95041`/`95044` and only cancels dead gated-video branches
  after failed gates; passing gates still preserve the video jobs for direct
  visual inspection.
- 2026-06-02 09:05: Submitted peg-drop/regrasp v4 task-frame-projected
  validation because prior evidence had already regained final grasp but
  remained outside the hole frame (`YZ≈0.062m`, align-only in `94991`). New
  chain: no-video `95088 -> 95089 -> 95090`, gated video `95091 -> 95092`,
  cleanup `95093`, summaries `95094`/`95095`. It keeps `peg_drop`, seed
  `7300`, `ALLOW_REGRASP=true`, `BRIDGE_SERVO_MODE=task_frame_projected`,
  `BRIDGE_PHASE_MODE=search`, and the unchanged final inserted-state gate.
- 2026-06-02 09:08: Queue/artifact audit found no completed new v4 or regrasp
  v8 result yet. `95088` is priority-pending for `2026-06-02T09:26:54`;
  `95039`/`95042` remain pending for `2026-06-02T10:00:12`. `94539` is still
  running on `server21` with four tasks and about 58 minutes elapsed; the
  Slurm log continues to print training epochs and `stderr` only has the
  expected PyTorch Transformer warning. This is liveness only, not training
  evidence.
- 2026-06-02 09:30: Regrasp v8 no-video job `95088` completed, inspection
  `95089` passed, and gate `95090` failed under the unchanged final
  inserted-state metric. It regained final grasp and entered insert phase but
  stalled at `x≈-0.109m`; H5 diagnostics show the peg-hole angle leaving the
  static insertion manifold while axis pushing continued. This is recorded in
  `docs/world_model_task_rebinding/2026-06-02_regrasp_v8_failure_v9_peg_alignment.md`.
- 2026-06-02 09:30: Implemented v9 peg-alignment insertion manifold control.
  Existing v4 pending jobs remain behavior-compatible because the new guard is
  explicitly enabled only by `TASK_SERVO_INSERT_MANIFOLD_GUARD=true`, and
  `peg_alignment` is selected only by
  `BRIDGE_ORIENTATION_REFERENCE=peg_alignment`. Static py_compile, `bash -n`,
  and offline `_choose_bridge` replay on `95088` states passed. Submitted
  regrasp v9 chain: no-video `95104 -> 95105 -> 95106`; gated video
  `95107 -> 95108`; cleanup `95109`; summaries `95110`/`95111`.
- 2026-06-02 09:33: Updated no-video controller jobs `95039`, `95042`, and
  `95104` from a render-history-derived node filter to a then-current
  live-state filter, allowing recovered old render-failure nodes for
  non-rendering validation. Their forecast was still
  `2026-06-02T18:05:41`; all old node-filter snapshots are scheduling history
  only, not reusable policy.
- 2026-06-02 09:35: Current queue audit found no new controller artifact after
  v8. `94539` is still running on 4 H200 GPUs but below 3 hours. `94540` and
  `94682` `C_pi` label jobs are forecast for `10:42:59` and `11:40:47`.
  Controller no-video jobs `95039`/`95042`/`95104` are now forecast for
  `2026-06-03T00:01:27`. Dense RGB-D jobs still request full nodes
  (`94541`: 1 node/8 GPUs, `94676`: 2 nodes/16 GPUs), so the RGB-D queue
  remains consistent with the no-sparse-allocation rule.
- 2026-06-02 09:38: No new result artifact exists. `94539` continues on
  `server21` with 4 H200 GPUs and about `01:27` elapsed; stdout advances and
  stderr only contains the known Transformer warning, but this remains
  liveness only. `94863` strict inspection is still the first point where the
  large-shard object-WM can become compliant training evidence.
- 2026-06-02 09:41: Preflighted pending C_pi and RGB-D chains. C_pi label and
  training jobs still use one-node 4-GPU allocations, `MIN_LABELS=512`, and
  `MIN_TRAIN_SECONDS=10800` where applicable. RGB-D `94676` is dense
  2-node/16-GPU with 8 GPUs per node; strict RGB-D inspection `94859` blocks
  slot training `94860`, which itself keeps the 4-H200/10800-second training
  guard. No new artifact or result appeared, and no queue change was needed.
- 2026-06-02 12:05: Re-audited the training floor after the explicit user
  constraint. Active training `94542` is running on one node / 4 H200 GPUs with
  `MIN_TRAIN_SECONDS=10800`; it remains below the 3-hour evidence floor and
  cannot be interpreted yet. RGB-D slot training `95236` is queued behind
  dense RGB-D inspection and also requires 4 H200 GPUs and
  `MIN_TRAIN_SECONDS=10800`. Added and submitted controlled node-specific RGB-D
  canaries `95265` (`server58`) and `95266` (`server10`) to check whether old
  nodes with earlier rendering failures recovered; these one-GPU canaries are only node-health
  probes and do not count as training/data evidence.
- 2026-06-02 12:07: Since `server10` and `server58` currently have all eight
  GPUs allocated, also submitted historical-node-observation canaries `95357` (`server39`) and
  `95358` (`server56`), the only checked old nodes with earlier render failures with apparent
  free GPUs. This is a recovery/availability probe only; formal training
  remains 4 H200 GPUs for at least 10800 seconds.
- 2026-06-02 12:15: Added CPU RGB-D inspection jobs for the node-specific canaries:
  `95372` waits on `95265` (`server58`), `95370` waits on `95266`
  (`server10`), `95371` waits on `95357` (`server39`), and `95373` waits on
  `95358` (`server56`). These use `MIN_RGBD_FILES=1` and
  `REQUIRE_NO_WARNINGS=true`, and only validate node/render health.
- 2026-06-02 12:18: Checked whether a larger dense RGB-D job should be
  submitted. Slurm `--test-only` forecasts were later for 32 GPUs
  (`2026-06-05T16:39:26`) and 64 GPUs (`2026-06-06T17:12:44`) than the active
  16-GPU dense job `94676` (`2026-06-03T00:01:27`), so no larger duplicate
  was submitted.
- 2026-06-02 12:30: Fixed an environment blocker before downstream CPU jobs
  hit it: `.venv` had a mixed NumPy install and local calibration import failed.
  Reinstalled `numpy==1.26.4`, verified NumPy/h5py/torch imports plus
  `pip check`, and reran pilot calibration to temporary outputs. Added paired
  `threshold_std_gates` calibration for `C_pi`. Initial no-video C_pi
  controller chain `95384 -> 95385 -> 95386` was canceled while pending because
  the submit line stored world-model member dirs in one space-separated env
  value. Replacement chain `95392 -> 95393 -> 95394` uses
  `WORLD_MODEL_DIRS_FILE` after `94660`. This chain is not video evidence and
  cannot become a major success claim unless the no-video gate passes and a
  separate render-safe video branch is run and inspected directly.
- 2026-06-03 01:13: Current resource/action correction. `99611` sensors-mode
  remaining-three RGB-D repair finished as a rendering/scheduling failure
  (`0/3` RGB-D files, `server58`, Vulkan `ErrorDeviceLost`), so dead
  dependency jobs `99612-99625` were canceled. Fresh non-executing probes show
  current account `mayi`; `gaosh`, `engram`, and `test` still reject the
  account/partition combination; usable `cpu` 1GPU/2GPU and 4GPU shapes are
  schedulable but not immediately allocatable. Submitted focused state/oracle
  smoke `99714` for `peg_disturb_wm_regrasp`, which started immediately on
  `server42`, and submitted 3-unit RGB-D repair `99715` with current-evidence
  job-local exclusions for RGB-D render-failed nodes
  `server10,server21,server28,server55,server58`. Exact96 aggregate and
  RGB-D method chain are `99716-99729`; slot and world-model training jobs
  remain 4 H200 / 3.5h, and controller evidence remains `SLOT_SOURCE=rgbd`.
  This is not RGB-D method evidence until exact96 data, visual gate, RGB-D
  slots, RGB-D-derived world model, controller video, and video review pass.
- 2026-06-03 01:22: RGB-D data blocker cleared. `99715` completed the exact
  three missing RGB-D units on `server42` (`3/3`, `failed_units.tsv` empty),
  `99716` aggregated exactly `96` files, `99717` structural inspection passed
  with `num_frames=28896` and `num_warnings=0`, and `99718` visual inspection
  passed with `valid_visual_artifacts=true`, `num_sampled_files=16`,
  `num_sampled_frames=96`, and `num_warnings=0`. The generated review sheet
  was opened directly and is nonblank. RGB-D slot training `99719` then
  started on `server20` at `2026-06-03T01:22:55` with one node / 4 H200 GPUs
  for a 3.5h time limit. This is RGB-D data and training-liveness evidence,
  not method success; method evidence still requires trained RGB-D slots,
  RGB-D-derived world model, controller video, and video review.
- 2026-06-03 01:34 resource probe and state scaffold outcome:
  `sbatch --test-only` confirms no new immediately allocatable GPU shape for
  the current user while `99719` is running. The user association is account
  `mayi`; `gaosh`, `engram`, and `test` still reject this account,
  `gpux`/`mgpu` are drained/inactive, `debug` 2/4 GPU is blocked by
  `MaxGRESPerAccount`, and new usable `cpu` 1/2/4 GPU probes forecast
  `2026-06-04T00:18:15`. Do not submit duplicate queue pollution right now.
  Existing state/oracle scaffold jobs did complete and give a clear physical
  result: continuous constant hole motion fails with CV chasing but succeeds
  with state learned-WM future prediction; reverse motion succeeds with both
  CV and state learned-WM; peg-drop regrasp succeeds; peg-disturb/regrasp
  remains a failure with final lateral alignment but insufficient insertion
  axis progress and no regrasp event. These are oracle-state scaffold results,
  not RGB-D method evidence. Evidence note:
  `docs/world_model_task_rebinding/2026-06-02_repair99208_failure_and_remaining3_chain_2336.md`.
- 2026-06-03 01:40 RGB-D controller coverage correction:
  added dependency-gated RGB-D controller video/inspection/review branches
  after `99726` for `hole_constant` (`99802-99804`), `hole_reverse`
  (`99805-99807`), `peg_drop` with regrasp (`99808-99810`), and
  `peg_disturb` with regrasp (`99811-99813`). They preserve
  `SLOT_SOURCE=rgbd`, RGB-D-derived world-model dependency, unchanged
  final-state evaluation, and required video artifact review. This avoids
  narrowing the method to one `hole_move_stop` demo, but it is not evidence
  until the jobs run and videos are inspected.
- 2026-06-03 03:17 active RGB-D slot chain correction:
  current active RGB-D slot training is `100319` only: one node / 4 H200,
  `TimeLimit=04:00:00`, `MIN_TRAIN_SECONDS=10800`, exact96 RGB-D input,
  pending on `cpu` with `ReqNodeList=server03` and no reliable start forecast.
  Downstream strict RGB-D method jobs are `100411-100433`, all gated on
  `100319`, with strict predicted-slot inspection, RGB-D-derived world-model
  gates, five controller video branches, and video artifact review. Earlier
  chain `99929-99951` and replacement attempts `100274` and `100409` were
  canceled before allocation after Slurm actual forecasts contradicted
  `--test-only` probes. This is a scheduling/queue correction, not RGB-D
  method evidence. Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_coordconv_slot_queue_correction_0317.md`.
- 2026-06-03 03:24 active slot failure-path diagnostic:
  submitted `100441` with `Dependency=afterany:100319` and
  `REQUIRE_COMPLIANT=false` to
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/diagnostics/job100319_afterany`.
  This is only a diagnostic for failed/timeout slot training; strict RGB-D
  method gate `100411` remains `afterok:100319` with
  `REQUIRE_COMPLIANT=true`.
- 2026-06-03 03:25 active world-model failure-path diagnostic:
  submitted `100445` with `Dependency=afterany:100415`,
  `REQUIRE_COMPLIANT=false`, and `REQUIRE_RGBD_DERIVED=true` to
  `experiments/world_model_task_rebinding/rgbd_object_world_model/diagnostics/job100415_afterany`.
  This is only a diagnostic for failed/timeout RGB-D-derived world-model
  training; strict RGB-D-derived gate `100416` remains `afterok:100415` with
  `REQUIRE_COMPLIANT=true` and `REQUIRE_RGBD_DERIVED=true`.
- 2026-06-03 03:29 active predicted-slot export failure-path diagnostic:
  submitted `100454` with `Dependency=afterany:100412`,
  `MIN_FILES=0`, `REQUIRE_REPORT=false`, and
  `REQUIRE_FRAME_ALIGNED=false` to
  `experiments/world_model_task_rebinding/rgbd_predicted_slots/diagnostics/job100412_afterany`.
  This is only a diagnostic for failed/partial predicted-slot export; strict
  RGB-D predicted-slot gate `100413` remains `afterok:100412` with exact96
  files and strict quality thresholds.
- 2026-06-03 03:35 active slot scheduling correction without evidence
  downgrade:
  inspected prior slot job `99719` and diagnostic `100113`; it had exact96
  RGB-D input and RGB-D/proprio boundary metadata, but was canceled after
  `01:22:48`, with `num_complete_members=0`, `compliant_3h_training=false`,
  and `rgbd_slot_training_evidence=false`. It must not be used as a method
  checkpoint. Active job `100319` was still pending and had stale
  `ReqNodeList=server03` while `server03` had all 8 H200 GPUs allocated.
  Submitted one true replacement probe `100465` with the same exact96 / 4H200
  / `MIN_TRAIN_SECONDS=10800` contract and `03:20:00` walltime; actual Slurm
  state did not produce a stable earlier forecast, so `100465` was canceled
  before allocation. Temporarily cleared `100319` `ReqNodeList` and reduced
  its walltime to `03:20:00`; the actual scheduled candidate moved to a later
  `server13` slot, and the shorter walltime was rejected as a correctness risk
  because RGB-D data loading happens outside the `MIN_TRAIN_SECONDS=10800`
  training loop. A second actual `server03` replacement probe `100470` also
  failed to get a stable earlier forecast and was canceled before allocation.
  Final active state: `100319` has `TimeLimit=04:00:00`,
  `ReqNodeList=server03`, no stable start forecast, and all downstream RGB-D
  method jobs remain gated on `100319`. There is still no compliant RGB-D
  slot, RGB-D-derived world-model, or RGB-D controller evidence.
- 2026-06-03 03:42 non-core expansion queue cleanup:
  canceled pending nextscale expansion jobs `100037-100040` before allocation
  (`0` elapsed, no nodes assigned). That branch requested 4GPU state rollout
  plus 8GPU RGB-D rendering for a 384-file expansion dataset, but it is not
  required for the current exact96 RGB-D slot/world-model/controller method
  chain and could consume scarce GPU resources without producing current
  method evidence. Current queue now keeps the active RGB-D method chain
  `100319 -> 100411...100433` plus state/oracle diagnostic `100224`.
- 2026-06-03 03:45 active chain lightweight preflight:
  with `100319` and `100224` still pending and no new artifacts, `bash -n`
  passed for the active slot, slot-inspection, predicted-slot export/
  inspection, RGB-D-derived world-model train/inspect/eval, RGB-D controller,
  video-review, and state-smoke wrappers. `.venv` `py_compile` passed for the
  corresponding Python entrypoints. Submitted downstream contract still points
  to `job100319`, exact96 predicted slots, `MIN_TRAIN_SECONDS=10800`,
  `REQUIRE_RGBD_DERIVED=true`, and five `SLOT_SOURCE=rgbd` video branches.
  This is readiness evidence only, not method evidence.
- 2026-06-03 03:52 live GRES probe and replacement refusal:
  `scontrol show nodes -d` showed most nodes with all 8 H200 GRES used; only
  `server39` and `server56` appeared to have 4 free H200 GRES, while
  `server16/server29` were drained. Non-executing probes for exact96 4H200/4h
  slot training on `server39` and `server56` forecast
  `2026-06-05T09:38:08`, but an actual bounded `server56` replacement probe
  `100485` did not receive a stable start forecast and was canceled before
  allocation. Active RGB-D slot path remains `100319`; no downstream
  dependency was changed and no method evidence was produced.
- 2026-06-03 04:00 active RGB-D chain contract audit:
  added lightweight read-only contract checker
  `scripts/world_model/audit_rgbd_method_chain_contract.py` and ran it on
  current chain
  `experiments/world_model_task_rebinding/rgbd_method_chains/coordconv_downstream_after_slot100319_20260603_031723`.
  The audit passed `176/176` checks and wrote
  `contract_audit_20260603_0355.json` / `.md`. It verifies exact96 full96
  RGB-D structural/visual gates, slot job `100319` with `MIN_TRAIN_SECONDS=10800`,
  strict slot inspection, exact96 RGB-D predicted-slot export/inspection,
  RGB-D-derived world-model inspection/eval, five `SLOT_SOURCE=rgbd`
  controller video branches, and required nonblank video artifact review.
  This is submitted-chain readiness evidence only; it is not RGB-D slot
  quality, world-model performance, controller success, or method evidence.
- 2026-06-03 04:02 current pending-state audit:
  re-read `AGENTS.md` plus active RGB-D/controller TODOs, then checked
  current Slurm state and artifacts. `100319` is still `PENDING` with
  `Reason=Priority`, `Elapsed=0`, `AllocNodes=0`, one node / 4 H200 /
  `TimeLimit=04:00:00`, `ReqNodeList=server03`; `squeue --start` gives
  `N/A`. Downstream `100411-100433`, diagnostics `100441/100445/100454`,
  and state/oracle diagnostic `100224` are still pending or dependency
  pending with no produced slot, predicted-slot, world-model, controller, or
  video artifacts. `server03` currently reports `GresUsed=8/8` H200 and
  `AllocTRES=cpu=80,mem=1256G,gres/gpu=8`, while `squeue -w server03` shows
  no visible jobs for this account. This is scheduling/resource evidence only,
  not training failure. No duplicate training job was submitted and no
  dependency/evaluation gate was changed.
- 2026-06-03 04:07 RGB-D slot data-path environment fix:
  a lightweight single-file RGB-D dataset smoke initially exposed a real venv
  problem: `.venv` NumPy import failed because the installed package was
  missing `numpy._typing._nested_sequence`, even though no repo-local
  `numpy.py`/`numpy` path shadowing existed and NumPy shared-library
  dependencies resolved. Since pending slot training `100319` uses the same
  `.venv`, this was a data/training environment issue that could have wasted
  a 4H200 allocation. Reinstalled `numpy==1.26.4`; verified
  `import numpy, h5py, torch`, `pip check`, and existence of the missing
  NumPy module. Re-ran the single-file RGB-D dataset smoke successfully:
  301 samples from one H5, image shape `(8,128,128)`, proprio `(18,)`,
  continuous targets `(25,)`, binary/visibility targets `(8,)`. This is
  code-path readiness only, not slot training or method evidence. `100319`
  remained pending throughout and no Slurm dependencies/evaluation gates were
  changed.
- 2026-06-03 04:11 RGB-D slot forward-path smoke:
  checked current Slurm state again before running local smoke: `100319` and
  `100224` were still `PENDING`, with no `job100319` artifacts. `sacct`
  submit line for `100319` still exports the exact96 RGB-D root,
  `MIN_RGBD_FILES=96`, `EXPECTED_RGBD_FILES=96`, and
  `MIN_TRAIN_SECONDS=10800`. Ran a one-thread CPU forward smoke on two samples
  from one RGB-D H5 using the current `RgbdSlotExtractor` defaults
  (`coord_conv=true`, `spatial_grid_size=4`). The model accepted image batch
  `(2,8,128,128)` plus proprio `(2,18)` and produced continuous slot output
  `(2,25)` plus binary/visibility logits `(2,8)`. This proves the current
  RGB-D slot data tensor shape and model forward interface are compatible
  before the 4H200 job starts. It is not training, slot quality, world-model,
  controller, or method evidence.
- 2026-06-03 04:13 reusable RGB-D slot preflight:
  added `scripts/world_model/preflight_rgbd_slot_training_path.py` to make the
  import/file-count/single-H5 dataset/forward smoke reproducible and
  evidence-backed. Ran it on the active exact96 RGB-D root with output
  `experiments/world_model_task_rebinding/rgbd_method_chains/coordconv_downstream_after_slot100319_20260603_031723/rgbd_slot_preflight_20260603_0413.json`
  and `.md`; result `status=pass`, `num_rgbd_files=96`, `num_samples=301`,
  batch image `(2,8,128,128)`, proprio `(2,18)`, predicted continuous slots
  `(2,25)`, predicted binary/visibility logits `(2,8)`,
  `coord_conv=true`, `spatial_grid_size=4`. This is reusable code-path
  readiness only, not training, validation, slot-quality, world-model,
  controller, or method evidence. `100319` remained pending and no Slurm
  dependency/evaluation gate was changed.
- 2026-06-03 04:17 RGB-D slot inspection gate audit:
  audited `inspect_rgbd_slot_extractor_ensemble.py`, strict inspection job
  `100411`, and diagnostic job `100441`. `100411` remains
  `afterok:100319` with `REQUIRE_COMPLIANT=true`; `100441` remains
  `afterany:100319`, `REQUIRE_COMPLIANT=false`, writes only to
  `rgbd_slot_extractor/diagnostics/job100319_afterany`, and its manifest
  states diagnostic-only/not method gate. Local gate-logic checks passed:
  an empty `/tmp` ensemble reports `num_members=0`,
  `compliant_training_evidence=false`, `rgbd_slot_training_evidence=false`;
  a fake `/tmp` 4-member/3h/exact96/RGB-D-boundary fixture reports both
  evidence flags true. This verifies the inspection gate can distinguish no
  artifacts from compliant evidence, but it is only gate-logic readiness, not
  training or method evidence. `100319` remained pending and no dependency or
  evaluation gate changed.
- 2026-06-03 04:23 RGB-D predicted-slot diagnostic semantics fix:
  audited `export_rgbd_predicted_slots.py`, strict predicted-slot inspection
  `100413`, and afterany diagnostic `100454`. The strict path remains
  `100412 -> 100413`, exact96, frame-aligned, `REQUIRE_REPORT=true`,
  `REQUIRE_FRAME_ALIGNED=true`, and quality thresholds
  `hole_pos<=0.03m`, `peg_head_hole<=0.035m`, `peg_pos<=0.04m`,
  `binary_accuracy>=0.95`. Found a diagnostic-only wrapper ambiguity:
  explicit empty `EXPECTED_FILES=` was interpreted as `0`, even though the
  manifest meant "no exact-count check for afterany failure classification".
  Patched `inspect_rgbd_predicted_slot_export.sbatch` so unset
  `EXPECTED_FILES` still defaults to `MIN_FILES`, while explicit empty,
  `none`, or `null` disables only the exact-count diagnostic check. Local
  wrapper smoke wrote `expected_files=null` and exited nonzero on an empty
  export, so diagnostics still cannot be mistaken for success. Canceled old
  pending diagnostic `100454` with `AllocNodes=0` and submitted corrected
  afterany diagnostic `100535`; it is not linked to world-model training or
  method claims. Re-ran the active method-chain audit:
  `contract_audit_20260603_0423.json` passed `176/176`. `100319` remained
  pending and no strict evaluation gate changed.
- 2026-06-03 04:26 RGB-D-derived world-model input contract smoke:
  audited `object_slot_dataset.py`,
  `train_rgbd_derived_object_world_model_ensemble_4gpu.sbatch`,
  `inspect_world_model_ensemble.py`, and the RGB-D controller loader. The
  training wrapper refuses fewer than 96 predicted-slot files, fewer than
  4 H200 tasks, multi-node sparse training, or `MIN_TRAIN_SECONDS<10800`.
  The dataset reads current and future values from `slots/*`, records
  `oracle_slots_read=false`, and uses `oracle_slots/*` only as inspection
  evidence. The world-model inspection requires exact predicted-slot input,
  RGB-D aux uncertainty/probability features, 4xH200, and at least 3h before
  setting `rgbd_derived_training_evidence=true`. A local smoke using old
  diagnostic predicted-slot files wrote
  `world_model_dataset_contract_smoke_20260603_0426.json`: `100` samples,
  feature shape `(100,2,78)`, `input_representation=rgbd_predicted_slots`,
  `world_model_input_group=slots`, `oracle_slots_read=false`, and RGB-D aux
  feature names present. This is representation contract readiness only, not
  full-data training or method evidence.
- 2026-06-03 04:31 controller final-state evidence gate hardening:
  patched `inspect_rebinding_controller_run.py` so controller inspection
  recomputes `success_once` and `success_at_end` directly from
  `rollouts.h5/metric_slots/inserted`, recomputes final peg-head task-frame
  state from H5 metric slots, and refuses inspection if those metric-derived
  values disagree with the JSON summary. First-principles reason: final
  success for dynamic task completion must be measured from the real final
  metric state after the dynamic event, not from RGB-D control slots or a
  potentially inconsistent summary field. Local fake-H5 checks passed:
  consistent RGB-D controller evidence exited `0`, while a run with summary
  final success but metric final uninserted exited nonzero. Added matching
  source-level checks to `audit_rgbd_method_chain_contract.py`; rerun
  `contract_audit_20260603_0431.json` passed `178/178`. This is strict
  evidence-gate hardening only, not controller success or method evidence.
- 2026-06-03 04:36 RGB-D method evidence review tail:
  added `review_rgbd_method_evidence.py` plus Slurm wrapper
  `review_rgbd_method_evidence.sbatch` as a post-controller/video evidence
  aggregator. First-principles reason: even if individual controller metrics
  and video artifacts exist, a method claim still needs one place that checks
  RGB-D controller input evidence, metric final-state consistency, dynamic
  event before final success, valid video review sheets, and then explicitly
  states that direct agent video/contact-sheet inspection is still required.
  The script never changes metrics and always sets
  `method_success_claim_allowed=false`; it only identifies whether branches
  are ready for direct visual review. Local no-result smoke on the active
  chain reported all five branches missing controller/video evidence, as
  expected. Submitted dependency-gated CPU job `100552` after all controller
  inspection/video review jobs. Initial job `100551` was canceled with
  `AllocNodes=0` because its `OUTPUT_DIR` had literal `jobJOBID`; `100552`
  uses the wrapper default `job${SLURM_JOB_ID}` path. Added this tail job to
  `jobs.tsv` and manifest; latest contract audit
  `contract_audit_20260603_0436.json` passed `183/183`. This is evidence
  hygiene only, not method evidence.
- 2026-06-03 04:48 slot queue/resource and environment correction:
  current RGB-D method evidence remains blocked on compliant slot training
  job `100319`, which is still `PENDING` with `Reason=Priority`,
  `Elapsed=0`, one node / 4 H200 / 32 CPU / 256G, and no `job100319`
  artifacts. Equivalent `sbatch --test-only` replacements that preserve
  exact96 RGB-D input and `MIN_TRAIN_SECONDS=10800` are not earlier:
  `cpu` forecasts `2026-06-04T04:27:14`, `gpu` forecasts
  `2026-06-05T13:03:14`, `debug` is blocked by `MaxGRESPerAccount`,
  `gaosh`/`engram`/`test` reject the account, and `gpux`/`mgpu` are
  unavailable. Live node accounting shows the only non-drained nodes with
  four free GPUs, `server39` and `server56`, each have only about `10GB`
  node memory free, below the 256G training request. Old slot artifacts in
  `job99719` are not reusable method input: the job was canceled after
  `01:22:48`, diagnostic inspection reports zero complete members and
  `RGB-D slot training evidence: False`, and it is the non-CoordConv grid-1
  path. Fixed a real venv/controller import risk by adding
  `deps/ManiSkill_clean/examples/baselines/diffusion_policy/diffusion_policy/__init__.py`
  and reinstalling the local editable package with `--no-deps`;
  `diffusion_policy.make_env`, `mani_skill`, `numpy/h5py/torch`, and
  `pip check` now pass. RGB-D slot preflight after the package fix passed on
  exact96 files, and chain audit
  `contract_audit_after_envfix_20260603_0506.json` passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 04:52 non-core pending state job cleanup:
  `100319` remains pending with no slot artifacts or Slurm logs. Canceled
  state/oracle diagnostic job `100224` before allocation because it is not
  part of the active RGB-D CoordConv chain, had no artifacts, and its
  `squeue --start` forecast had moved to `2026-06-08T00:00:00`. `sacct`
  confirms `100224` is `CANCELLED by 2059`, `Elapsed=00:00:00`, with no
  assigned node. This removes a non-method 1GPU pending footprint and does not
  change evidence gates or delete method evidence. After cancellation,
  exact96 4H200 slot-training probes still did not reveal an earlier legal
  path: `cpu` replacement forecasts `2026-06-04T04:44:18` on `server03`,
  `debug` remains blocked by `MaxGRESPerAccount`, and `server03` still has
  `AllocTRES=...,gres/gpu=8`. No replacement training job was submitted.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 04:55 submitted snapshot and memory-boundary check:
  wrote and inspected the submitted batch snapshot for `100319`; it still
  preserves exact96 input, one-node 4 H200 training, 256G memory,
  `TimeLimit=04:00:00`, `MIN_TRAIN_SECONDS=10800`, RGB-D image plus
  robot-proprio inputs, and object-slot labels only as training targets. Live
  TRES shows `server03` has only 1 free H200, while `server39` and `server56`
  each have 4 free H200 but only about `10GB` free node memory; down/drain
  `server16/server29` are not usable. Equivalent 256G replacement probes are
  still later (`cpu` `2026-06-04T04:40:17`, `gpu`
  `2026-06-05T13:47:17`), and lower-memory `128G/64G/32G` probes still
  forecast `2026-06-04T04:47:18`, not an earlier path. Inspected the slot
  dataset implementation: `load_rgbd_slot_samples` creates full in-memory
  arrays and `RgbdSlotDataset` converts them to torch tensors; exact96 image
  tensors alone are about `14.1GiB` per ensemble member and `56.4GiB` across
  four members before overhead. Therefore lowering memory to chase 10GB-free
  nodes would be an OOM-prone scheduling shortcut, not an aligned fix. No
  replacement job was submitted and no training/evaluation gate changed.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:03 optional lazy RGB-D image loading for future aligned
  retries/replacements:
  added `load_rgbd_slot_metadata`, `LazyRgbdSlotDataset`,
  `--lazy-images`, and Slurm `LAZY_IMAGES=true` support so future slot
  training can keep RGB-D images as the perception input without materializing
  all exact96 image tensors in host memory. The default remains eager, so the
  submitted `100319` snapshot is unchanged. Compile and wrapper syntax checks
  passed; full exact96 lazy preflight passed with `96` RGB-D H5 files; a
  single-H5 real-data equality check confirmed eager and lazy outputs match
  exactly for `image`, `proprio`, `target_cont`, and `target_bin` on indices
  `0`, `1`, and `10`. This is input-loading readiness only, not slot-quality
  evidence or method evidence, and no training/evaluation gate was relaxed.
  Post-change contract audit passed `183/183`. Legal lazy replacement probes
  preserving exact96 input, 4 H200, one node, and
  `MIN_TRAIN_SECONDS=10800` were not earlier: 4h 8G/12G/16G/32G and 3h15m
  8G/16G/32G probes forecast `2026-06-04T05:08:30`; node-targeted probes on
  `server39/server56` were later. No lazy replacement job was submitted.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:10 RGB-D slot-training triage tooling:
  added `triage_rgbd_slot_training_job.py` so future `100319` failures can be
  classified from Slurm state, logs, member artifacts, and concrete error
  signatures instead of guessed. The tool is diagnostic only and does not
  change exact96, 4H200/3h, slot-quality, RGB-D-derived WM, controller, or
  video gates. Compile passed. Running it on current `100319` produced
  `slot_training_triage_100319_20260603_0508.json` and classified only
  `scheduling_pending`, `pending_reason_priority`, and
  `no_ensemble_artifacts_yet`; no logs or artifacts exist and `sacct` still
  shows elapsed `00:00:00`. This is readiness/failure-localization evidence,
  not method evidence.
  Submitted diagnostic tail job `100588` with dependency `afterany:100319`
  and output
  `rgbd_slot_extractor/diagnostics/job100319_triage/triage.json`; it is
  CPU-only diagnostic, not a method gate. Added it to active chain tracking
  and reran contract audit, which still passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:13 diagnostic tracking hygiene:
  existing afterany diagnostics `100441`, `100535`, and `100445` were
  submitted and documented in TODOs, but were missing from the active
  chain `jobs.tsv`. Added tracking-only rows with labels containing
  `diagnostic`, preserving their original dependencies and outputs:
  slot `afterany:100319`, predicted-slot export `afterany:100412`, and
  RGB-D-derived world-model `afterany:100415`. No Slurm job, strict gate,
  or method dependency was changed. Post-tracking contract audit still passed
  `183/183`. A no-result evidence-review smoke parsed the expanded
  `jobs.tsv`, found all five controller branches, and exited `65` only
  because controller/video artifacts are still missing, which is expected
  fail-closed behavior before method evidence exists.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:18 RGB-D-derived world-model training triage:
  added generic read-only ensemble training triage plus a Slurm wrapper, then
  submitted `100590` with `Dependency=afterany:100415` to classify future
  world-model training logs/artifacts. It detects scheduling, environment,
  HDF5 locking, memory/CUDA OOM, predicted-slot input/file-count errors,
  undersized-wrapper refusal, `srun` allocation errors, and incomplete
  members. This is diagnostic only: it does not replace `100416/100417`,
  approve RGB-D-derived training evidence, or touch controller/video gates.
  Compile/syntax checks passed; current read-only triage on pending `100415`
  classified only dependency-pending/no-artifacts; post-tracking contract
  audit passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:22 RGB-D controller job triage:
  added read-only controller Slurm/log/artifact triage and submitted
  `100591-100595` with dependencies after controller video jobs
  `100418/100421/100424/100427/100430`. The diagnostics classify
  rendering/Vulkan, missing RGB-D slot/world-model inputs, HDF5 locking,
  memory/CUDA OOM, Slurm allocation, import/environment, controller runtime,
  missing `rollouts.h5`, and missing videos. They do not change controller
  metric gates, video review gates, or method evidence review. Compile/syntax
  checks passed; current pending `100418` triage reports only
  dependency-pending/no-artifacts; chain audit remains `183/183`; no-result
  evidence review still fails closed only because controller/video artifacts
  are absent.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:28 RGB-D predicted-slot export triage:
  added read-only predicted-slot export triage plus Slurm wrapper and
  submitted `100597` with `Dependency=afterany:100412`. This diagnoses the
  RGB-D perception-to-world-model handoff if `100412` fails or exits without
  export artifacts; it classifies scheduling, environment/import, HDF5 lock,
  memory/CUDA OOM, exact96 RGB-D/file-count, slot checkpoint/compliance,
  allocation, and export runtime errors. It is diagnostic only and does not
  change `100413`, predicted-slot quality thresholds, RGB-D-derived
  world-model gates, controller/video gates, or method evidence review.
  Local read-only triage on pending `100412` classified only
  dependency-pending/no-artifacts; `100597` is pending on `afterany:100412`;
  chain audit after tracking passed `183/183`. A no-result evidence-review
  smoke after the new row still found all five controller branches and failed
  closed only because controller/video artifacts do not exist yet.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:32 slot-training queue probe:
  `100319` remains `PENDING` with `Reason=Priority`, no start forecast, no
  slot artifacts, and downstream `100411/100412/100597` still dependency
  pending. `sacct SubmitLine` confirms `100319` was not submitted with an
  explicit `--nodelist`; the current `ReqNodeList=server03` in `scontrol` is
  recorded as Slurm scheduling state, and `scontrol --planned/--noplanned`
  show the same field, while `server03` currently has
  `AllocTRES=...,gres/gpu=8`. Same-objective 4H200 exact96 probes were not
  better enough to justify a replacement: `cpu` 256G/4h and lazy 64G/3h15m
  both forecast `2026-06-04T05:11:30`, and `gpu` forecasts
  `2026-06-05T13:01:30`. No replacement was submitted and no downstream gate
  changed. After reading local `scontrol` documentation, cleared the
  non-submitline required-node field with
  `scontrol update JobId=100319 ReqNodeList=`. This preserved the same job,
  output path, exact96 input, 4H200/256G/4h request, `MIN_TRAIN_SECONDS=10800`,
  and all downstream dependencies. `100319` is still priority-pending, but
  now with `ReqNodeList=(null)`, so it is no longer pinned to `server03`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:36 slot-training forecast check:
  after rereading active TODO and relevant PLAN files, `100319` is still the
  active strict RGB-D slot blocker: `PENDING`, `Reason=Priority`,
  `ReqNodeList=(null)`, no artifacts/logs, but now with Slurm forecast
  `StartTime=2026-06-03T17:35:25`, `SchedNodeList=server03`. Downstream
  strict and diagnostic jobs remain dependency-pending. Same-objective
  replacement probes preserving exact96, 4 H200 GPUs, and
  `MIN_TRAIN_SECONDS=10800` were later: `cpu` 256G/4h and lazy 64G/128G
  3h15m all forecast `2026-06-04T05:11:30`, while `gpu` 256G/4h forecasts
  `2026-06-05T12:48:30`. No replacement was submitted because the active
  path is earlier and already attached to the downstream RGB-D-derived chain.
  This is scheduling evidence only, not slot/model/controller evidence.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:39 downstream controller/video readiness audit:
  reread the rebinding-controller PLAN and audited the RGB-D controller video
  wrapper, common controller wrapper, controller inspection wrapper, video
  artifact wrapper, and Python entrypoints. The wrappers keep
  `SLOT_SOURCE=rgbd`, require compliant RGB-D slot and RGB-D-derived
  world-model inspections, export Vulkan/HDF5 settings, save videos, and
  inspect final success from H5 `metric_slots` with RGB-D control slots kept
  separate. Bash syntax and Python compile passed. No code/gate changed; this
  is readiness evidence only.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:42 predicted-slot export/inspection readiness audit:
  audited the RGB-D slot-to-world-model handoff. The strict path still
  refuses non-exact96 RGB-D inputs, requires compliant slot inspection and
  frame alignment, writes RGB-D predictions under `slots/*`, copies
  `oracle_slots/*` only for inspection, preserves uncertainty/probability
  tensors, and enforces current quality gates before world-model training.
  Bash syntax and Python compile passed. No threshold/gate changed.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:43 RGB-D slot-training input boundary audit:
  audited `rgbd_slot_dataset.py`, `train_rgbd_slot_extractor.py`,
  `inspect_rgbd_slot_extractor_ensemble.py`, and the slot train/inspection
  wrappers. The training input is RGB-D images from camera observations plus
  robot-only `qpos/qvel`; object/task slots are targets only and not model
  inputs. Member manifests record the proprio/target boundaries, and strict
  slot inspection requires exact full96 RGB-D input, RGB-D perception input
  evidence, 4xH200, and >=3h training before `rgbd_slot_training_evidence`
  can be true. Bash syntax and Python compile passed. `100319` was still
  pending with forecast `2026-06-03T09:37:26`, `ReqNodeList=(null)`, and no
  logs/artifacts. No code/gate changed.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:45 RGB-D-derived world-model readiness audit:
  audited train/inspect/eval wrappers and dataset/training/eval Python
  entrypoints. The strict path refuses non-exact96 predicted-slot files,
  fewer than four H200 tasks, sparse multi-node training, and
  `MIN_TRAIN_SECONDS<10800`; dataset features come from `slots/*` plus RGB-D
  aux tensors with `oracle_slots_read=false`; inspection requires 4xH200/3h,
  RGB-D predicted-slot input, and RGB-D aux feature evidence before
  `rgbd_derived_training_evidence=true`. Bash syntax and Python compile
  passed. No code/gate changed.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:47 submitted slot snapshot and full96 RGB-D input check:
  pulled the actual submitted batch script for `100319` with
  `scontrol write batch_script` and verified it preserves exact96 RGB-D
  input, one-node 4 H200 training, `MIN_TRAIN_SECONDS=10800`, and the
  RGB-D image plus robot-only `qpos/qvel` input boundary. The full96 aggregate
  root contains exactly `96` `.rgbd.h5` files; structural inspection reports
  `28896` frames and `0` warnings; visual artifact inspection reports
  `valid_visual_artifacts=true`; the review sheet was opened directly and is
  nonblank. `100319` remains pending with forecast
  `2026-06-03T09:37:26`, `ReqNodeList=(null)`, and no logs/artifacts. This is
  readiness evidence only; no RGB-D method evidence exists yet.
  Post-record contract audit
  `contract_audit_after_submitted_snapshot_20260603_0547` passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:49 downstream submitted-snapshot audit:
  wrote and syntax-checked submitted snapshots for 24 queued downstream jobs
  `100411-100433` and `100552`. The snapshots preserve exact96 RGB-D slot
  export, exact96 predicted-slot world-model training,
  `MIN_TRAIN_SECONDS=10800`, `REQUIRE_RGBD_DERIVED=true`,
  `SLOT_SOURCE=rgbd`, video/nonblank review, and final
  `REQUIRE_ALL_BRANCHES=true` evidence review. Actual controller SubmitLines
  preserve branch scenarios `hole_move_stop`, `hole_constant`,
  `hole_reverse`, `peg_drop`, and `peg_disturb`; `ALLOW_REGRASP=true` is only
  on peg-drop/disturb branches. Negative grep found no state/oracle
  slot-source fallback. This is submitted-job correctness evidence only.
  Post-audit contract check
  `contract_audit_after_downstream_snapshots_20260603_0549` passed
  `183/183`. Live blocker remains `100319`, pending priority with forecast
  `2026-06-03T09:37:26` and no logs/artifacts.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:53 diagnostic tail audit:
  wrote and syntax-checked submitted snapshots for 11 diagnostic jobs
  `100441/100535/100445/100588/100590/100591-100595/100597`. All are
  dependency-pending behind the strict upstream jobs they diagnose. Regenerated
  read-only triage for `100319`, `100412`, `100415`, and `100418`; the outputs
  classify only priority/dependency pending and no-artifact-yet conditions.
  The diagnostic boundaries explicitly do not change exact96, 4H200/3h,
  RGB-D-derived world-model, controller metric, video, or visual-inspection
  gates. Negative grep found no state/oracle fallback or method-success
  override.
  Reread the relevant PLAN files after the audit; the current path still
  matches the original plan boundary. Post-record contract audit
  `contract_audit_after_diagnostic_audit_20260603_0553` passed `183/183`.
  Live blocker remains `100319`, pending priority with forecast
  `2026-06-03T09:37:26`, no logs, and no slot artifacts.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 05:56 same-objective resource probe:
  checked whether a legal exact96, one-node 4H200, `MIN_TRAIN_SECONDS=10800`
  replacement slot-training job could start earlier than active `100319`.
  Current `100319` remains pending priority with forecast
  `2026-06-03T09:37:26`. Non-submitting probes were later: `cpu` eager
  256G/4h, `cpu` lazy 64G/4h, and `cpu` lazy 64G/3h15m all forecast
  `2026-06-04T05:11:30`; `gpu` eager/lazy forecasts
  `2026-06-05T12:51:30`. No replacement was submitted; no gate changed.
  Post-probe contract audit `contract_audit_after_resource_probe_20260603_0556`
  passed `183/183`; live `100319` still has no stdout/stderr or slot manifest.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:03 slot-training runtime-risk audit:
  reread active rules/TODO/PLAN, then audited the still-pending `100319`
  slot-training path with lightweight local checks only. Corrected Python-only
  compile passed for slot training, RGB-D dataset, and ensemble inspection
  entrypoints; Bash syntax passed for the slot wrapper, actual submitted
  `100319` snapshot, downstream submitted snapshots, and diagnostic
  snapshots. The actual submitted `100319` batch is eager image loading, not
  the later source wrapper's lazy mode. It preserves exact96 RGB-D input,
  one-node 4H200, `MIN_TRAIN_SECONDS=10800`, robot-only `qpos/qvel` proprio
  input, and object/task slots as labels only. Read-only H5 shape audit found
  `96` RGB-D H5 files, `96` trajectories, `28896` frames, two 128x128 RGB-D
  cameras, qpos/qvel shape `(301, 9)`, and estimated eager float32 image
  memory `14.109 GiB` per member / `56.438 GiB` for four members, reasonable
  under the `256G` request. `100319` remains pending priority with forecast
  `2026-06-03T09:37:26` and no slot logs/artifacts. This is readiness
  evidence only; no RGB-D method evidence exists yet.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:06 single-H5 slot interface smoke:
  ran a bounded local dataset/model forward check on one RGB-D H5 with HDF5
  locking disabled and CPU thread counts set to one. It did not train or
  optimize. The check produced image tensor `[301, 8, 128, 128]`, robot-only
  proprio `[301, 18]`, continuous target `[301, 25]`, binary target
  `[301, 8]`, and model forward outputs `[2, 25]` / `[2, 8]`, preserving the
  `rgbd_images_plus_robot_proprio` and slots-as-labels boundary. This is an
  interface smoke only, not slot quality, training, world-model, controller,
  or method evidence.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:10 live chain check and fail-closed review:
  reread active rules/TODO/PLAN, then checked the live chain. `100319`
  remains `PENDING`, `Reason=Priority`, `ReqNodeList=(null)`,
  `SchedNodeList=server03`, forecast `2026-06-03T09:37:26`; `sacct` shows
  `AllocNodes=0`, `Elapsed=00:00:00`. There are still no slot logs or
  artifacts. Refreshed read-only triage under
  `experiments/world_model_task_rebinding/rgbd_method_chains/coordconv_downstream_after_slot100319_20260603_031723/current_readonly_triage_20260603_0610`;
  slot/export/world-model/controller triage all classify only priority or
  dependency pending plus no-artifact-yet conditions. A no-result
  evidence-review smoke under
  `.../no_result_evidence_review_smoke_20260603_0610` failed closed with
  exit code `65`, `method_success_claim_allowed=false`, zero candidate
  branches, and missing controller/video inspection for all five branches.
  This is scheduling/fail-closed evidence only; no method evidence or gate
  change.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:17 slot inspection fail-closed smoke:
  with `100319` still pending and no logs/artifacts, ran only a lightweight
  local strict-inspection smoke. Environment sanity showed `.venv` can import
  `typeguard._exceptions`; `typeguard=4.5.2`, `tyro=1.0.13`; both
  `train_rgbd_slot_extractor.py --help` and
  `review_rgbd_method_evidence.py --help` exit `0`. A first attempt under
  `no_result_slot_inspection_smoke_20260603_0614` is explicitly discarded as
  non-evidence because a transient import/command issue prevented
  `inspection.json` from being written. The corrected smoke under
  `no_result_slot_inspection_smoke_20260603_0617` wrote `inspection.json`
  with `num_members=0`, `compliant_training_evidence=false`, and
  `rgbd_slot_training_evidence=false`; the equivalent
  `REQUIRE_COMPLIANT=true` gate exits `65`. This is fail-closed gate evidence
  only; no method evidence, no code change, and no gate change.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:20 future wrapper CLI import preflight hardening:
  `100319` remains pending with no artifacts; this does not affect the
  already submitted `100319` batch snapshot and is not a replacement
  submission. Hardened only future source wrappers for RGB-D slot training,
  predicted-slot export, and RGB-D-derived world-model training so their
  Python preflight imports `tyro` and `typeguard._exceptions` in addition to
  `h5py/numpy/torch`. This prevents future GPU allocations from reaching the
  training/export entrypoint before discovering a CLI environment issue.
  Bash syntax passed for all three wrappers, and local venv import sanity
  passed. No training code, input modality, exact96 gate, 4H200/3h floor,
  predicted-slot threshold, RGB-D-derived world-model requirement,
  controller metric, video review, or method evidence rule changed.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:22 submitted-batch drift audit:
  `100319` remains pending priority with forecast `2026-06-03T09:37:26`,
  no logs, and no slot artifacts. Because the last same-objective resource
  probe was at `05:56`, no new probe or replacement was submitted before the
  cadence. Wrote the Slurm-stored `100319` batch script to
  `submitted_batch_100319_after_source_hardening_20260603_0622.sh`; it is
  byte-identical to the 05:46 submitted snapshot (`cmp=0`). The active job
  therefore still does not include the future source wrapper's `tyro` /
  `typeguard._exceptions` preflight or lazy-image path, and still preserves
  exact96 eager RGB-D slot training with `MIN_TRAIN_SECONDS=10800`,
  RGB-D+robot-proprio inputs, and object slots as labels only. This is
  submitted-job consistency evidence only, not method evidence.
  Post-record contract audit
  `contract_audit_after_submitted_batch_drift_20260603_0622` passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:26 same-objective resource probe:
  `100319` remains pending priority with forecast
  `2026-06-03T09:37:26`, no slot logs, and no artifacts. Non-submitting
  exact96 one-node 4H200 replacement probes preserving
  `MIN_TRAIN_SECONDS=10800` were all later or invalid: `cpu` eager/lazy
  replacements forecast `2026-06-04T08:47:26`, `gpu` eager/lazy forecast
  `2026-06-05T16:19:26`, `debug` is blocked by `MaxGRESPerAccount`,
  `gpux`/`mgpu` are inactive or drained, and `gaosh`/`engram`/`test` reject
  the current account/partition combination. Test-only IDs `100652-100656`
  are not in `squeue`; no replacement was submitted. This is scheduling
  evidence only, not method evidence.
  Post-probe contract audit `contract_audit_after_resource_probe_20260603_0626`
  passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:29 downstream dependency/output-path audit:
  verified the actual queued strict chain still fail-closes on upstream gates:
  `100411` is `afterok:100319`, `100412` is `afterok:100411`,
  `100413` is `afterok:100412`, `100415` is `afterok:100413`,
  `100416` is `afterok:100415`, `100417` is `afterok:100416`, and
  controller video jobs are `afterok:100417`. Diagnostic/visual/triage jobs
  use `afterany` only for inspection or failure-localization paths. Predicted
  slot output `job100412` and RGB-D-derived world-model output `job100415`
  both currently contain `0` files, so there are no stale downstream artifacts.
  This is dependency hygiene only, not method evidence.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:30 RGB-D data status:
  exact96 RGB-D data are available under
  `experiments/world_model_task_rebinding/rgbd_dynamic_distributed/from_rollout_dir/full96_aggregate_p48retry98524_98527_latehalf_repair99208_repair99611_next_20260603_0115`.
  The aggregate has `96` `.rgbd.h5` files; structural inspection reports
  `num_files=96`, `num_warnings=0`; visual artifact inspection reports
  `valid_visual_artifacts=true`, `num_files=96`, `num_warnings=0`. The review
  sheet was opened directly and is nonblank. Current blocker is slot training
  `100319`, now pending priority with forecast `2026-06-03T07:12:33` and no
  slot logs/artifacts. This proves RGB-D data readiness, not RGB-D method
  success.
  Post-record contract audit
  `contract_audit_after_rgbd_status_20260603_0630` passed `183/183`. At
  `2026-06-03T06:33:04+08:00`, `100319` was still pending priority with the
  same `2026-06-03T07:12:33` forecast and no slot logs/artifacts; downstream
  strict jobs remain dependency-pending. Recent `.pyc` files generated by
  lightweight local Python checks were deleted.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:35 scheduled-node sanity:
  `100319` remains pending priority with `SchedNodeList=server03`, forecast
  `2026-06-03T07:12:33`, and no slot logs/artifacts. Live node checks show
  `server03` is `MIXED`, not drained/down, has `Gres=gpu:NVIDIAH200:8`, and
  currently has `AllocTRES=...gres/gpu=8`, so its GPUs are fully allocated at
  the moment. `sinfo` reports reason `none`. This supports the current
  interpretation that the blocker is scheduling/priority for a real H200
  allocation, not a bad-node or wrong-resource artifact. No replacement or
  duplicate probe was submitted.
  Post-node-check contract audit
  `contract_audit_after_node_sanity_20260603_0635` passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:38 read-only slot triage refresh:
  refreshed `100319` triage under
  `experiments/world_model_task_rebinding/rgbd_method_chains/coordconv_downstream_after_slot100319_20260603_031723/current_readonly_triage_20260603_0638`.
  It still classifies only `scheduling_pending`, `pending_reason_priority`,
  and `no_ensemble_artifacts_yet`; stdout/stderr do not exist, the ensemble
  directory does not exist, member dirs are `0`, complete members are `0`, and
  there are no log pattern hits. Recommended next action is to wait on the
  current path and recheck, or probe only if a legal same-objective shape may
  start earlier. This is scheduling/failure-localization evidence only, not
  method evidence.
  Post-triage contract audit
  `contract_audit_after_readonly_triage_20260603_0638` passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:41 slot-training startup checklist:
  added
  `experiments/world_model_task_rebinding/rgbd_method_chains/coordconv_downstream_after_slot100319_20260603_031723/slot100319_startup_and_completion_checklist_20260603_0641.md`
  so the first running/completed `100319` evidence is checked against the
  existing strict contract: exact96 RGB-D input, one-node four-H200 request,
  `MIN_TRAIN_SECONDS=10800`, RGB-D plus robot-proprio inputs, slots as labels,
  four complete members, and strict `100411` inspection before downstream
  method claims. This is operational hygiene only and changes no gate.
  Post-checklist contract audit
  `contract_audit_after_slot100319_checklist_20260603_0641` passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:44 scheduling read-only check:
  `100319` remains pending priority with stable forecast
  `2026-06-03T07:12:33`, no slot logs, and no artifacts. `squeue --start`
  reports the same forecast. `server03` is still `MIXED` with
  `Gres=gpu:NVIDIAH200:8`, but live `AllocTRES` is now `gres/gpu=7`, so only
  one GPU is free while `100319` needs four H200 GPUs. No replacement or
  duplicate resource probe was submitted because the last same-objective probe
  was at `06:26`, below the 30-minute cadence. This is scheduling evidence
  only, not method evidence.
  Post-scheduling-check contract audit
  `contract_audit_after_scheduling_check_20260603_0644` passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 06:56 same-objective resource probe:
  `100319` remained pending priority with forecast `2026-06-03T07:12:33`,
  no slot logs, and no artifacts. Non-submitting exact96 one-node 4H200
  replacement probes preserving `MIN_TRAIN_SECONDS=10800` were later or
  invalid: `cpu` eager/lazy forecasts `2026-06-04T06:22:33`, `gpu`
  eager/lazy forecasts `2026-06-05T13:55:33`, `debug` is blocked by
  `MaxGRESPerAccount`, `gpux`/`mgpu` are inactive or drained, and
  `gaosh`/`engram`/`test` reject the current account/partition combination.
  Test-only IDs `100671-100675` are absent from `squeue`; no replacement was
  submitted. Current `100319` remains the earliest aligned path. This is
  scheduling evidence only, not method evidence.
  Post-probe contract audit
  `contract_audit_after_resource_probe_20260603_0656` passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 07:07 slot training startup health check:
  `100319` is now running on `server03`, started
  `2026-06-03T07:02:39`, with one node, four H200 GPUs, 32 CPUs, and
  four-hour walltime. Startup manifests confirm exact96 RGB-D H5 input,
  `MIN_TRAIN_SECONDS=10800`, `input_modality=rgbd_images_plus_robot_proprio`,
  robot qpos/qvel-only proprio, object slots as labels only, coord-conv
  RGB-D slot extractor, and four members/seeds `500-503`. stderr is empty at
  startup; `sstat` reports active step `100319.0` with about `6.8GB` MaxRSS
  and `18GB` disk read. This is startup evidence only; no RGB-D slot,
  world-model, controller, or method success is claimed until four member
  checkpoints/metrics and strict downstream inspections complete.
  Post-startup contract audit
  `contract_audit_after_slot100319_startup_20260603_0707` passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 07:18 slot training running progress:
  `100319` is still running on `server03` with runtime `00:16:04`. stderr
  remains empty; stdout now contains regular epoch/batch events for all four
  members; all four member dirs have `best_model.pt` and `manifest.json`.
  `sstat` shows active step `100319.0` with about `30.6GB` MaxRSS and
  `43.2GB` disk read. This proves the RGB-D slot job is past startup and
  training, but not complete: final `model.pt`, `metrics.json`, completion
  lines, and strict `100411` inspection are still required before any slot
  evidence claim. Downstream RGB-D-derived world-model and controller/video
  jobs remain dependency-pending.
  Post-progress contract audit
  `contract_audit_after_slot100319_running_progress_20260603_0718` passed
  `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 07:21 downstream submitted snapshot audit:
  captured `36` submitted batch scripts under
  `submitted_batch_snapshots_20260603_0721_jobs_100411_100433_100552`;
  `bash -n` passed. The submitted chain still preserves strict `afterok`
  gates from `100411` through `100417`, exact96 RGB-D predicted-slot export
  and quality gate, RGB-D-derived world-model training with
  `MIN_TRAIN_SECONDS=10800`, `REQUIRE_RGBD_DERIVED=true`, `SLOT_SOURCE=rgbd`
  controller videos, expected-slot-source inspection, and nonblank video
  review. No stale downstream artifacts were found in checked roots. This is
  hygiene evidence only, not method evidence.
  Post-downstream-snapshot contract audit
  `contract_audit_after_downstream_snapshot_20260603_0721` passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 07:27 RGB-D data and slot-training progress:
  exact96 RGB-D data are present in the aggregate root (`96` `.rgbd.h5` files).
  Formal RGB-D slot training job `100319` is running on `server03` with four
  H200 GPUs and `MIN_TRAIN_SECONDS=10800`; runtime was `00:24:39` at
  `2026-06-03T07:27:18+08:00`. stderr is empty, `sstat` shows the active
  training step with about `30.6GB` MaxRSS and `43.2GB` disk read, and stdout
  has active batch/epoch events for all four members. Latest parsed per-member
  epochs are about `204-210`. Current diagnostic slot RMSEs are still
  centimeter-level (`hole_pos_rmse_m` about `0.057-0.073`,
  `peg_head_hole_rmse_m` about `0.074-0.109`), so slot quality remains an
  open risk and must be judged by unchanged downstream inspections `100411`
  and `100413` after the 3-hour floor completes. No RGB-D-derived
  world-model/controller/video evidence exists yet.
  Post-status contract audit
  `contract_audit_after_slot100319_0727_status_20260603_0730` passed
  `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 07:40 stronger RGB-D slot risk-mitigation branch:
  early `100319` validation curves are not final evidence but show a real
  perception-risk pattern: best hole RMSE is still about `0.052-0.070m` and
  best peg-head-hole RMSE is about `0.072-0.104m`, far from the unchanged
  predicted-slot gate. Patched the slot training wrapper to explicitly record
  and pass `SPATIAL_GRID_SIZE`, and patched the coordconv submitter to
  explicitly record/export slot capacity settings plus auto-submit final
  method-evidence review. Submitted controlled backup chain
  `strongslot128_grid8_risk_mitigation_20260603_073418` with slot job
  `100700` and downstream `100701-100724`: exact96 RGB-D, 4H200,
  `MIN_TRAIN_SECONDS=10800`, `CNN_CHANNELS=128`, `HIDDEN_DIM=512`,
  `SPATIAL_GRID_SIZE=8`, `DROPOUT=0.05`, `BATCH_SIZE=96`, strict unchanged
  predicted-slot/world-model/controller/video gates. `100700` is pending
  priority with a tentative `2026-06-09T03:00:00` start, so it is not a faster
  replacement for running `100319`; it is queued risk mitigation and can be
  canceled if the current branch proves sufficient. Main and backup chain
  contract audits both pass `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 07:52 RGB-D slot split metadata fix:
  live `100319` remains running and is not yet formal evidence. During
  perception-risk diagnosis, found that the fallback scenario parser was
  treating aggregate filenames with trajectory suffixes as distinct scenarios,
  so old validation splits were not guaranteed to cover all dynamic event
  types; `member_0` had no `hole_reverse` validation trajectory. Patched
  future source to parse canonical scenario names and default to trajectory
  validation split stratified by scenario. Exact96 metadata validation now
  gives each tested seed (`500-503`, `700`) six balanced categories with `12`
  train and `4` val trajectories each. This is a checkpoint-selection and
  diagnostic fix for future queued/retry slot jobs, not a change to RGB-D
  inputs, labels, exact96 gates, thresholds, 4H200/3h floor,
  RGB-D-derived world-model/controller gates, or video evidence requirements.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 07:59 predicted-slot scenario diagnostics:
  added diagnostic-only per-scenario oracle-error reporting for RGB-D
  predicted-slot export/inspection. The physical purpose is to localize a
  future strict `100413` failure to a dynamic event type, such as moving-hole
  or peg-disturb cases, rather than guessing from a global RMSE. This does not
  change the predicted-slot quality gate, thresholds, RGB-D-derived world-model
  requirement, controller protocol, or video evidence rule; `quality_gate_passed`
  still uses the original global thresholds and oracle slots remain
  inspection-only. Python compile and synthetic diagnostic checks passed.
  Main and strong backup contract audits
  `contract_audit_after_predslot_scenario_diag_20260603_0759` passed
  `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 08:05 live RGB-D slot training progress triage:
  `100319` is still running on `server03` with about `01:02:43` elapsed,
  stderr size `0`, four H200 GPUs allocated, and downstream
  `100411/100412/100413/100415` still dependency-pending. The output directory
  still has only intermediate `best_model.pt` files and manifests; no final
  `model.pt`, `metrics.json`, history, or prediction examples exist. Extended
  `triage_rgbd_slot_training_job.py` to parse live `rgbd_slot_train_epoch`
  events and write latest/min observed geometry diagnostics without changing
  any gate. Diagnostic report
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_20260603_0804.md`
  parsed `2954` epoch events and shows current/min hole and peg-head-hole
  RMSEs remain far above the downstream reference thresholds, but this is
  live risk only. Formal evidence still requires completed member artifacts
  and strict `100411 -> 100412 -> 100413`. Main and strong backup contract
  audits `contract_audit_after_slot_triage_progress_20260603_0805` passed
  `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 08:10 active validation-split triage:
  `100319` remains running with no final member `model.pt`/`metrics.json` and
  downstream `100411/100412/100413/100415` still dependency-pending. Extended
  the read-only slot triage tool to report canonical scenario counts in each
  member validation split from `val_keys`. This matters because active
  `100319` was submitted before the canonical scenario/stratified validation
  fix, so its live/final validation metrics must not be over-interpreted as
  balanced dynamic-event evidence. Generated
  `triage_live_progress_split_20260603_0809.md`; it shows `member_0` has no
  `hole_reverse` validation trajectory and the other members cover all six
  scenarios but with imbalanced counts. This does not change training,
  checkpoint selection, thresholds, downstream gates, or video evidence
  rules. Main and strong backup contract audits
  `contract_audit_after_valsplit_triage_20260603_0810` passed `183/183`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 08:20 strong-slot backup submit mismatch repair:
  audited the actual submitted Slurm record for the earlier backup chain
  `strongslot128_grid8_risk_mitigation_20260603_073418` and found a
  submission mismatch: the manifest described `CNN_CHANNELS=128`,
  `HIDDEN_DIM=512`, `SPATIAL_GRID_SIZE=8`, `DROPOUT=0.05`, and
  `BATCH_SIZE=96`, but `jobs.tsv`, `sacct SubmitLine`, and submitted
  `100700.sbatch` did not explicitly carry those exports. Added contract
  checks so manifest slot hyperparameters must match submitted job export
  values, and added explicit `STRATIFY_VAL_BY_SCENARIO` export to the
  coordconv submitter. The mismatch audit failed as expected (`183/188`), so
  old pending jobs `100700-100724` were canceled with zero elapsed time and no
  `AllocTRES`. Submitted corrected backup chain
  `strongslot128_grid8_stratified_resubmit_20260603_0820`, jobs
  `100730-100754`, with exact96 RGB-D, 4H200/3h slot and world-model floors,
  strong slot capacity, stratified validation, unchanged predicted-slot/
  world-model/controller/video gates, and final evidence review. `sacct
  SubmitLine` for `100730` now explicitly includes all strong-slot and
  stratification exports; submitted snapshots were saved and `bash -n`
  passed. Corrected contract audit passed `191/191`. `100730` is pending
  priority with current tentative start `2026-06-09T04:00:00`, so this is
  readiness/risk-mitigation queue work, not method evidence and not a
  replacement for running `100319`.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 08:24 live slot-training recheck:
  `100319` remains running on `server03` with runtime about `01:21:28`,
  stderr empty, and no final member `model.pt`, `metrics.json`,
  `training_history.json`, or `prediction_examples.json`. Generated
  read-only diagnostic
  `triage_live_progress_split_20260603_0824.md/json`, which parsed `4026`
  epoch events. Latest live validation geometry is still far above the
  unchanged downstream reference gates for hole and peg-head-hole errors
  (`member_0` `0.0589/0.0715m`, `member_1` `0.0606/0.0758m`,
  `member_2` `0.0697/0.0777m`, `member_3` `0.0773/0.1101m`). The triage
  classifies `incomplete_ensemble_members` only because training is not done.
  This is a live perception-risk diagnostic, not failure evidence and not a
  gate change. Existing afterany diagnostics are already queued for slot
  training, predicted-slot export, RGB-D-derived world-model training, and
  controller branches; no new heavy job was submitted.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 08:26 strong-slot backup diagnostics:
  the corrected strong-slot backup chain
  `strongslot128_grid8_stratified_resubmit_20260603_0820` had strict gates
  but lacked the main chain's afterany failure-localization diagnostics.
  Submitted lightweight CPU diagnostics `100757-100767`: non-strict slot
  inspection after `100730`, non-strict predicted-slot export inspection after
  `100732`, non-strict RGB-D-derived world-model inspection after `100735`,
  slot/world-model/export triage after `100730/100735/100732`, and controller
  triage after the five video jobs `100738/100741/100744/100747/100750`.
  These jobs are dependency-pending with zero elapsed time/allocation. Saved
  `11` submitted snapshots under the strong backup chain directory and
  `bash -n` passed. Strong backup contract audit
  `contract_audit_after_diagnostics_added_20260603_0826` passed `191/191`.
  This is failure-localization readiness only, not a gate change or method
  evidence.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_rgbd_slot_queue_envfix_0448.md`.
- 2026-06-03 08:36 RGB-D slot label distribution audit:
  added and ran read-only diagnostic
  `scripts/world_model/audit_rgbd_slot_label_distribution.py` for active
  `100319`, producing
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/label_distribution_audit_20260603_0830.md/json`.
  The audit reads labels, robot proprio metadata, visibility metadata, and
  member split manifests only; it does not materialize RGB-D image arrays,
  train, export predicted slots, change thresholds, or affect any downstream
  gate. It confirms the exact96 label set is dataset-balanced (`16`
  trajectories and `4816` samples for each of the six scenarios) and
  quaternion targets are unit-normalized. The live `100319` slot metrics are
  better than train-mean label baselines, so the current high live errors are
  not explained by a trivial mean predictor; however hole and peg-head-hole
  errors remain above the unchanged downstream reference limits. This keeps
  the classification as perception-quality risk while `100319` is still
  running, not method evidence and not a reason to relax gates. Py compile
  passed; post-record contract audits still pass for the main chain
  (`183/183`) and corrected strong-slot backup (`191/191`); evidence note
  updated.
- 2026-06-03 08:39 live slot-training recheck after label audit:
  reran read-only triage
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_split_20260603_0839.md/json`.
  `100319` remains running on `server03` with runtime about `01:36:34`,
  stderr empty, and no final member `model.pt`, `metrics.json`, history, or
  examples. The triage parsed `4898` epoch events; latest live geometry is
  still above the unchanged hole and peg-head-hole reference limits
  (`member_0` `0.0593/0.0723m`, `member_1` `0.0610/0.0766m`,
  `member_2` `0.0699/0.0785m`, `member_3` `0.0778/0.1104m` for
  hole/peg-head-hole). Minimum observed hole and peg-head-hole values are
  also still above limits. This remains perception-risk localization only;
  formal status waits for completed artifacts and strict downstream gates.
- 2026-06-03 08:43 RGB-D slot label projection review:
  added and ran lightweight visual diagnostic
  `scripts/world_model/visualize_rgbd_slot_labels.py`, producing
  `experiments/world_model_task_rebinding/rgbd_dynamic_distributed/from_rollout_dir/full96_aggregate_p48retry98524_98527_latehalf_repair99208_repair99611_next_20260603_0115/visual_artifact_review/slot_label_projection_20260603_0842/slot_label_projection_review.md/json`
  and contact sheet
  `.../slot_label_projection_20260603_0842/rgbd_slot_label_projection_review.png`.
  The diagnostic samples one trajectory from each of the six exact96
  scenarios and overlays simulator hole/peg/TCP labels onto source RGB-D
  frames. I opened the contact sheet: base-camera markers land on the
  target/peg/end-effector regions, while the hand camera mostly sees peg/TCP
  and usually not the hole, matching projection rates and expected viewpoint.
  Warnings were `0`. This reduces the likelihood that live `100319` slot risk
  is caused by gross camera/label misalignment; it is visual debugging only,
  not slot quality evidence, not method evidence, and not a gate change.
  Post-record contract audits still pass for the main chain (`183/183`) and
  corrected strong-slot backup (`191/191`). `100319` remains running; strict
  downstream inspections are still pending.
- 2026-06-03 08:50 RGB-D slot training curve audit:
  added and ran read-only stdout diagnostic
  `scripts/world_model/audit_rgbd_slot_training_curves.py`, producing
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/training_curve_audit_20260603_0848.md/json`.
  It parsed `5521` live epoch events from `100319`. Train loss is down to
  roughly `0.3%` of the first parsed epoch for all members, but every member's
  best-observed hole and peg-head-in-hole-frame errors remain above the
  unchanged task-frame reference limits. `member_1` and `member_2` also show
  validation geometry regressing from their best while train loss keeps
  falling. This strengthens the classification as RGB-D slot perception
  generalization/capacity risk, not label-scale, mean-baseline, or gross
  projection failure. It is not final failure evidence while `100319` is still
  running and strict `100411/100413` have not completed; no gate changed.
  Post-record contract audits still pass for the main chain (`183/183`) and
  corrected strong-slot backup (`191/191`).
- 2026-06-03 08:53 main-chain diagnostic coverage recheck:
  `100319` remains running with stderr empty and no final member artifacts.
  Rechecked the active chain's existing afterany coverage before submitting
  anything else. The main path already has afterany slot inspection/triage
  (`100441`, `100588`), predicted-slot export/inspection diagnostics
  (`100535`, `100597`), RGB-D-derived world-model diagnostics (`100445`,
  `100590`), controller branch triage (`100591-100595`), predicted-slot visual
  review (`100433`), and final evidence review (`100552`). No new job was
  submitted because the missing evidence is completed `100319` artifacts and
  strict downstream outcomes, not a missing diagnostic. This avoids queue
  pollution and preserves the unchanged RGB-D method chain. Post-record
  contract audits still pass for the main chain (`183/183`) and corrected
  strong-slot backup (`191/191`).
- 2026-06-03 08:56 checkpoint contract audit tightened:
  patched `scripts/world_model/audit_rgbd_method_chain_contract.py` so
  readiness audits now require RGB-D predicted-slot export, RGB-D world-model
  eval, and all RGB-D controller branches to resolve slot/world-model
  checkpoints to `best_model.pt`, either explicitly or through wrapper
  defaults. This prevents drift where downstream could silently use final
  `model.pt` after validation geometry has regressed from the best checkpoint.
  It is anti-drift readiness only: no training, export, gate, threshold, or
  controller protocol changed. Py compile passed; main chain audit now passes
  `195/195`, corrected strong-slot backup passes `203/203`.
- 2026-06-03 09:01 live slot-training triage refresh:
  reran read-only triage for active `100319`, writing
  `triage_live_progress_split_20260603_0901.md/json`. `100319` is still
  running on `server03`, stderr is empty, and final member artifacts remain
  incomplete (`0/4` members with final `model.pt` plus `metrics.json`). The
  triage parsed `6159` live epoch events; latest and best-observed hole and
  peg-head-in-hole errors remain above the unchanged reference limits. This
  keeps the current classification as RGB-D slot perception
  generalization/capacity risk while training is incomplete. It is not method
  evidence, not a final slot failure, and not a reason to change gates. No new
  Slurm job was submitted and no active job was canceled. Post-record
  contract audits still pass for the main chain (`195/195`) and corrected
  strong-slot backup (`203/203`).
- 2026-06-03 09:05 downstream handoff gate audit:
  while `100319` remained running with no final artifacts, audited the source
  and submitted snapshots for strict `100411`, export `100412`, predicted-slot
  gate `100413`, and RGB-D-derived world-model training `100415`. The handoff
  still blocks the world model unless slots come from compliant exact96 RGB-D
  training, export uses validation-selected `best_model.pt`, predicted slots
  pass unchanged global task-frame quality gates, and world-model training
  receives exact96 RGB-D-predicted `slots/*` with uncertainty/probability
  datasets under the 4H200/3h floor. Per-scenario predicted-slot diagnostics
  remain localization only, not separate gates. This is readiness evidence,
  not method evidence, and no Slurm job or gate changed. Post-record contract
  audits still pass for the main chain (`195/195`) and corrected strong-slot
  backup (`203/203`).
- 2026-06-03 09:08 live slot triage and backup recheck:
  refreshed live state for `100319` and corrected strong-slot backup `100730`.
  `100319` remains running on `server03`, stderr is empty, no final member
  artifacts exist, and strict downstream jobs remain dependency-pending. New
  read-only triage `triage_live_progress_split_20260603_0908.md/json` parsed
  `6583` epoch events; latest and best-observed hole/peg-head-hole errors are
  still above unchanged reference limits, so this remains live RGB-D
  perception-risk localization only. `100730` remains pending priority with
  the intended one-node 4xH200, exact96, strong-slot, stratified-validation,
  3h-floor submission. No job or gate changed. Post-record contract audits
  still pass for the main chain (`195/195`) and corrected strong-slot backup
  (`203/203`).
- 2026-06-03 09:16 strong backup replacement probe canceled:
  re-read active TODO/PLAN and submitted exactly one complete strong-slot
  replacement chain `100780-100804` after non-submitting probes suggested a
  `cpu` 4H200 shape could start on `2026-06-04T10:22:39` instead of current
  backup `100730` on `2026-06-09T01:00:00`. The replacement preserved exact96
  RGB-D input, strong-slot settings, stratified validation, unchanged
  predicted-slot gates, RGB-D-derived world-model/controller gates, 4H200/3h
  floors, and video review; its contract audit passed `203/203`. Real
  submitted scheduling then assigned `100780` to
  `2026-06-09T03:00:00`, later than `100730`, so `100780-100804` were
  canceled before allocation. `sacct` shows zero elapsed time, zero allocated
  nodes, and `Start=None` for all replacement jobs. Kept existing backup
  `100730-100754` and diagnostics `100757-100767`; main and backup contract
  audits after cancellation still pass (`195/195` and `203/203`). This is
  scheduling/readiness evidence only, not RGB-D method evidence and not a gate
  or protocol change.
- 2026-06-03 09:22 live slot triage while main job runs:
  re-read `AGENTS.md`, active/focused TODOs, and RGB-D/world-model plans
  before interpreting live state. `100319` remains running on `server03` with
  `4xNVIDIAH200`, runtime about `02:19:40`, empty stderr, active 4-task
  memory use, and no final member `model.pt`/`metrics.json` artifacts. New
  read-only triage
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_split_20260603_0922.md/json`
  parsed `7360` epoch events and classified only
  `incomplete_ensemble_members`. Latest hole/peg-head-hole/peg errors remain
  above or near unchanged reference limits (`member_0`
  `0.060101/0.073934/0.043116`, `member_1`
  `0.061389/0.077729/0.038566`, `member_2`
  `0.070601/0.080718/0.043772`, `member_3`
  `0.078045/0.112979/0.068811`). This is live perception-risk localization
  only, not final slot or method evidence. Strict downstream jobs remain
  dependency-pending; no Slurm job or gate changed. Post-record contract
  audits pass for the main chain (`195/195`) and corrected strong backup
  (`203/203`).
- 2026-06-03 09:25 downstream coverage recheck:
  refreshed Slurm/artifacts while `100319` was still running
  (`02:22:54/04:00:00`, empty stderr, no final member artifacts). Checked the
  submitted main-chain and strong-backup `jobs.tsv` coverage instead of
  submitting duplicate jobs. Main chain still has strict
  `100319 -> 100411 -> 100412 -> 100413 -> 100415 -> 100416 -> 100417`,
  `100414` sensitivity, `100433` predicted-slot visual review, afterany slot,
  predicted-slot, world-model, controller triage, video reviews, and final
  evidence review `100552`. Strong backup still has `100730-100754` plus
  diagnostics `100757-100767`. No Slurm job or gate changed. Main/backup
  contract audits pass (`195/195`, `203/203`). This is readiness evidence
  only; final RGB-D evidence still waits for completed slot artifacts and
  strict downstream results.
- 2026-06-03 09:28 final evidence review no-result smoke:
  while `100319` remained running with no final artifacts, audited the
  evidence/video review path. Controller inspection recomputes success from
  H5 metric slots, checks RGB-D controller input evidence, separate metric
  slots, summary consistency, and dynamic-event-before-success. Video artifact
  inspection creates readable nonblank review sheets but does not judge
  semantics. Final evidence review aggregates those reports and keeps
  `method_success_claim_allowed=false`. A no-result `.venv` smoke under
  `coordconv_downstream_after_slot100319_20260603_031723/no_result_evidence_review_smoke_20260603_0928`
  exited `65` because all five controller/video inspections are still missing;
  it reported zero candidate branches and no method success claim. No code,
  Slurm job, gate, or protocol changed. Py compile passed and post-smoke
  audits still pass for main/backup (`195/195`, `203/203`).
- 2026-06-03 09:34 predicted-slot to world-model boundary audit:
  re-read active TODO/PLAN while `100319` continued running
  (`02:31:05/04:00:00`) with empty stderr and no final member artifacts.
  Audited the export, predicted-slot inspection, object-slot dataset,
  world-model training, inspection, eval, and contract-audit sources. The
  handoff still writes RGB-D extractor ensemble predictions into `slots/*`,
  copies oracle labels only into `oracle_slots/*`, requires boundary attrs and
  frame alignment, blocks export if unchanged global task-frame quality gates
  fail, and trains/evaluates the world model from `slots/*` with
  `oracle_slots_read=false` plus RGB-D uncertainty/probability auxiliary
  features. The 4H200/3h world-model wrapper still requires exact96
  predicted-slot files and refuses missing RGB-D prediction datasets. Local
  py_compile and bash syntax checks passed; main and strong-backup contract
  audits pass (`195/195`, `203/203`) at
  `contract_audit_after_predslot_wm_boundary_20260603_0934.md/json`. No code,
  Slurm job, gate, threshold, or protocol changed. This is handoff readiness
  evidence only, not final slot quality, world-model performance, controller
  success, video evidence, or RGB-D method evidence.
- 2026-06-03 09:36 live slot triage refresh:
  ran read-only triage
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_split_20260603_0936.md/json`.
  `100319` is still running on `server03` (`02:34:00/04:00:00`) with empty
  stderr, exact96 RGB-D input, 4xH200 allocation, no final `model.pt` or
  `metrics.json`, and downstream jobs dependency-pending. The triage parsed
  `8184` live epoch events and again classified only
  `incomplete_ensemble_members`. Latest hole/peg-head-hole/peg RMSEs remain
  above or near unchanged reference limits: member_0
  `0.05955/0.07397/0.04265`, member_1 `0.06165/0.07806/0.03908`,
  member_2 `0.07128/0.08142/0.04495`, member_3
  `0.07897/0.11345/0.06866`. Member_0 validation split misses
  `hole_reverse`, reinforcing why the stratified strong backup exists. This
  is live RGB-D perception-risk localization only; no gate, job, or protocol
  changed and there is still no RGB-D method evidence. Post-record contract
  audits pass for main/backup (`195/195`, `203/203`) at
  `contract_audit_after_0936_live_triage_20260603_0938.md/json`.
- 2026-06-03 09:41 live artifact health check:
  re-read TODO/PLAN and refreshed Slurm/artifacts. `100319` remains running
  on `server03` (`02:38:52/04:00:00`) with strict downstream jobs still
  dependency-pending and no final `model.pt`/`metrics.json`. A first `find`
  traversal emitted `member_2: No such file or directory` while stdout was
  still writing `member_2` events, so treated it as an artifact anomaly to
  inspect before interpreting anything. Direct `ls`, `find -maxdepth 1`, and
  `test` checks immediately confirmed all four member directories exist and
  all four `best_model.pt` files are present; no member has final `model.pt`.
  `sstat` shows the 4-task step is active, stderr is empty, and stdout keeps
  writing epoch events. This is a transient directory-traversal/filesystem-read
  anomaly, not a training failure, not method evidence, and not a reason to
  submit duplicate work or change gates. Post-record contract audits pass for
  main/backup (`195/195`, `203/203`) at
  `contract_audit_after_0941_artifact_health_20260603_0943.md/json`.
- 2026-06-03 09:45 live slot triage refresh:
  ran read-only triage
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_split_20260603_0945.md/json`
  while `100319` remained running on `server03` (`02:42:40/04:00:00`) with
  empty stderr, active 4-task `sstat`, no final `model.pt`/`metrics.json`,
  and strict downstream jobs still dependency-pending. The triage parsed
  `8698` live epoch events and classified only `incomplete_ensemble_members`.
  Latest hole / peg-head-in-hole / peg RMSEs were member_0
  `0.06006/0.07458/0.04383`, member_1 `0.06163/0.07825/0.03889`,
  member_2 `0.07135/0.08139/0.04463`, and member_3
  `0.07966/0.11403/0.06805`; unchanged reference limits remain
  `0.03/0.035/0.04`. Minimum observed hole and peg-head-in-hole errors remain
  above reference limits for every member. Member_0 validation still misses
  `hole_reverse`, so the existing stratified strong backup remains the aligned
  hedge. This is live RGB-D perception-risk localization only: no completed
  slot artifact, predicted-slot export, world-model/controller metric, video,
  method evidence, gate change, threshold change, Slurm change, or protocol
  change. Post-record contract audits pass for main/backup (`195/195`,
  `203/203`) at
  `contract_audit_after_0945_live_triage_20260603_0946.md/json`.
- 2026-06-03 09:50 dependency coverage and running-health recheck:
  re-read `AGENTS.md`, active/focused TODOs, and RGB-D/world-model/controller
  plans, then refreshed Slurm and artifacts. `100319` is still running on
  `server03` (`02:47:08/04:00:00` at `2026-06-03T09:49:47+08:00`), stderr is
  `0` bytes, stdout is still writing all-member epoch events, `sstat` shows
  the 4-task step active, and no final member `model.pt`/`metrics.json`
  exists. Submitted snapshots confirm strict slot inspection `100411` remains
  `afterok:100319`, while failure-localization diagnostics `100441` and
  `100588` remain `afterany:100319`; final evidence review `100552` still
  waits on afterany controller and video-inspection branches. Therefore no
  duplicate Slurm job was submitted and no gate/threshold/protocol changed.
  This is dependency/readiness evidence only, not slot quality or RGB-D method
  evidence. Post-record contract audits pass for main/backup (`195/195`,
  `203/203`) at
  `contract_audit_after_0950_dependency_coverage_20260603_0951.md/json`.
- 2026-06-03 09:55 live slot triage refresh:
  ran read-only triage
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_split_20260603_0955.md/json`.
  `100319` remains running on `server03` (`02:52:30/04:00:00`) with exact96
  RGB-D input, 4xH200 allocation, empty stderr, no final member
  `model.pt`/`metrics.json`, and strict downstream jobs still
  dependency-pending. The triage parsed `9237` epoch events and reports only
  `incomplete_ensemble_members`. Latest hole / peg-head-in-hole / peg RMSEs
  were member_0 `0.06002/0.07469/0.04393`, member_1
  `0.06223/0.07885/0.03912`, member_2 `0.07149/0.08127/0.04721`, and
  member_3 `0.07974/0.11303/0.06716`; unchanged reference limits remain
  `0.03/0.035/0.04`. Minimum observed hole and peg-head-in-hole errors remain
  above reference limits for every member. Member_0 validation still misses
  `hole_reverse`, reinforcing the existing stratified strong backup as an
  aligned hedge. This is live RGB-D perception-risk localization only, not
  final slot evidence, method evidence, a gate change, or a reason to submit
  duplicate work. Post-record contract audits pass for main/backup
  (`195/195`, `203/203`) at
  `contract_audit_after_0955_live_triage_20260603_0956.md/json`; the
  accidentally misplaced main audit JSON was removed and generated
  `scripts/world_model/__pycache__` was cleaned.
- 2026-06-03 10:03 3h training-floor boundary check:
  waited until `100319` crossed the required `MIN_TRAIN_SECONDS=10800`
  boundary, then refreshed Slurm, logs, and artifacts. At
  `2026-06-03T10:03:55+08:00`, `100319` was still running on `server03`
  (`03:01:16/04:00:00`), stderr was `0` bytes, `sstat` still showed the
  4-task step active, stdout continued writing all four member epoch events,
  all four members still had `best_model.pt`, and no member had final
  `model.pt`/`metrics.json`. Strict `100411` and afterany diagnostics
  `100441/100588` remained dependency-pending. This confirms the run was not
  shortened below the required 3h floor, but it is not slot-quality evidence,
  not final failure/success evidence, not RGB-D method evidence, and not a
  reason to change gates or submit duplicate jobs. Post-record contract
  audits pass for main/backup (`195/195`, `203/203`) at
  `contract_audit_after_3h_floor_health_20260603_1004.md/json`.
- 2026-06-03 10:10 post-floor live slot triage:
  ran read-only triage
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100319/triage_live_progress_split_20260603_1010.md/json`.
  `100319` is still running on `server03` at about `03:06:58/04:00:00`,
  with exact96 RGB-D input, 4xH200 allocation, empty stderr, active 4-task
  training, all four `best_model.pt` files present, and no final member
  `model.pt` or `metrics.json`. The triage parsed `10072` epoch events and
  still reports only `incomplete_ensemble_members`. Latest hole /
  peg-head-in-hole / peg RMSEs were member_0 `0.06006/0.07441/0.04458`,
  member_1 `0.06197/0.07901/0.03960`, member_2
  `0.07215/0.08166/0.04603`, and member_3 `0.07886/0.11464/0.06957`;
  unchanged reference limits remain `0.03/0.035/0.04`, and minimum-observed
  hole plus peg-head-in-hole errors remain above reference for every member.
  Member_0 validation still misses `hole_reverse`; the existing stratified
  strong backup remains the aligned hedge. This is live perception-risk
  localization only, not final slot quality, RGB-D method evidence, a gate
  change, or a reason to submit duplicate work. Contract audits pass for
  main/backup (`195/195`, `203/203`) at
  `contract_audit_after_1010_live_triage_20260603_1012.md/json`.
- 2026-06-03 10:21 slot completion, export failure classification, and
  replacement chain:
  `100319` completed on `server03` at `2026-06-03T10:14:12+08:00`
  after `03:11:33` with exact96 RGB-D input, 4xH200 allocation, empty
  stderr, and all four final `model.pt`/`metrics.json` artifacts. Strict
  training-evidence inspection `100411` and afterany inspection `100441`
  completed, confirming RGB-D input/proprio boundary and 4H200/3h compliance;
  final aggregate slot geometry remained high (`hole_pos_rmse_m_mean=0.0686`,
  `peg_head_hole_rmse_m_mean=0.0877`, `peg_pos_rmse_m_mean=0.0498`), so this
  is not slot-quality or method evidence. Predicted-slot export `100412`
  failed before export with a Python import `SyntaxError` in
  `train_rgbd_slot_extractor.py`; current source now passes `py_compile` and
  import checks, so the failure is classified as export/import implementation
  runtime failure, not RGB-D data, physics, or method evidence. Diagnostic
  jobs `100535/100597` confirm zero predicted-slot H5 files. Canceled dead
  old pending branch jobs `100413-100432`, `100445`, `100552`, and
  `100590-100595`. Added scheduling-only submitter parameter
  `PRED_EXPORT_PARTITION` so replacement export need not default to the later
  `gpu` partition; this does not change any gate, threshold, input, output,
  or controller protocol. Submitted replacement chain
  `coordconv_downstream_after_slot100319_exportfix_20260603_1025`:
  `100847 -> 100848 -> 100849 -> 100851 -> 100852 -> 100853 ->
  controller/video/review 100854-100870`. `100847` already completed;
  `100848` was moved from unavailable `debug` to `cpu` partition and is
  pending priority. Replacement contract audit passed `203/203`. No RGB-D
  method evidence exists until predicted-slot export/inspection,
  RGB-D-derived world-model, controller metrics, and video inspection finish.
- 2026-06-03 10:26 CPU-only predicted-slot export replacement correction:
  the first replacement branch `100847-100870` was superseded because its
  export job `100848` had no useful start time after `debug` became unusable
  and a GPU-backed `cpu` partition move forecast later than the strong backup.
  Canceled pending `100848-100870` after `100847` had completed. Added
  submitter/export support for CPU-only predicted-slot export
  (`PRED_EXPORT_GRES=none`, `PRED_EXPORT_CUDA_EXPORT=false`,
  `REQUESTED_GRES_LABEL=none`) and manifest recording; this is an inference
  scheduling fix, not a method/gate change. Submitted active replacement
  chain
  `coordconv_downstream_after_slot100319_exportfix_cpuonly_20260603_1028`
  with jobs `100876-100899`; `100876` completed, `100877` is pending as a
  CPU-only export on `cpu` with exact96 RGB-D, the same slot ensemble, and
  unchanged strict predicted-slot quality gates. Contract audit passed
  `203/203`; CPU-only Slurm probes found no earlier legal shape than about
  `2026-06-04T09:46:28`. Still no predicted-slot H5, RGB-D-derived
  world-model, controller, video, or method evidence.
- 2026-06-03 10:42 predicted-slot export and inspection pass:
  CPU-only export `100877` ran on `server54` and completed at
  `2026-06-03T10:39:42+08:00` after `00:13:14`, writing exact `96`
  RGB-D-predicted slot H5 files plus `export_report.json/md`; stderr contains
  only the expected `torch.load` FutureWarning. Strict predicted-slot
  inspection `100878` completed at `10:40:35`, with `valid_export=True`,
  `quality_gate_passed=True`, `96` files, `96` trajectories, `0` warnings,
  and unchanged gate metrics:
  `hole_pos_rmse_m=0.02067`, `peg_head_hole_rmse_m=0.02801`,
  `peg_pos_rmse_m=0.01485`, `binary_accuracy=0.99772`. Visual review
  `100898` completed at `10:40:37`; the generated contact sheet was opened
  directly and is nonblank/readable with RGB-D frames and predicted/oracle
  slot overlays. Sensitivity diagnostic `100879` completed at `10:41:36`,
  showing aggregate `insert_yz_gate` mismatch `0.05177` and `hole_move_stop`
  insert-gate mismatch `0.11524`, which is controller-risk context only and
  not a gate change. Current world-model training job `100880` is
  dependency-free, 4xH200, 4h, pending priority with forecast
  `2026-06-04T14:00:00`; contract audit after predicted-slot pass is
  `203/203`. There is still no RGB-D-derived world-model, controller metric,
  controller video, or method-success evidence.
- 2026-06-03 10:49 world-model queue correction:
  after predicted-slot pass, Slurm forecasts moved the original current
  world-model tail `100880` from `2026-06-04T14:00:00` to
  `2026-06-08T01:00:00`; a test-only same-shape `cpu` probe initially showed
  a possible earlier window, so added submitter/audit support for reusing
  already-passed predicted-slot jobs `100877/100878/100879/100898` and
  submitted a replacement tail `100935-100953`. The replacement chain audit
  passed `203/203`, proving the reuse manifest still preserved RGB-D-derived
  world-model/controller contracts, but `100935` forecast then moved to
  `2026-06-08T03:00:00` while the strong backup slot job `100730` moved
  earlier to `2026-06-03T11:57:34`. Canceled old pending tail
  `100880-100897,100899` and replacement pending tail `100935-100953` with
  zero allocation, leaving completed predicted-slot evidence intact and
  avoiding duplicate far-future 4H200 queue pollution. Current earliest
  aligned path is strong backup `100730` and its existing strict downstream
  chain; post-correction strong-backup contract audit passed `203/203`. No
  RGB-D world-model/controller/video/method evidence exists yet.
- 2026-06-03 11:10 strong-backup slot training live health:
  strong backup RGB-D slot job `100730` started early on `server13` at
  `2026-06-03T10:50:35+08:00` with exact96 RGB-D input, 4xH200 request,
  `spatial_grid_size=8`, `CNN_CHANNELS=128`, `HIDDEN_DIM=512`, and
  `MIN_TRAIN_SECONDS=10800`. Slurm stderr contains a node/GPU warning
  (`Unable to determine the device handle for GPU1`), but all four member
  tasks are running and have written `best_model.pt`. Live triage at
  `11:10` reports only `incomplete_ensemble_members`; members 0/2/3 are
  around epochs `118/110/115`, while member 1 is a slow straggler at epoch
  `1`. A legal replacement probe excluding `server13` would start only on
  `2026-06-04T10:36:40`, so do not cancel the running same-day path unless
  concrete failure appears. Current live slot errors remain above unchanged
  task-frame reference limits and are diagnostic only, not final slot or
  method evidence.
- 2026-06-03 11:26 strong-backup straggler triage and no-cancel decision:
  read-only triage
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100730/triage_live_progress_split_20260603_1125.md/json`
  confirms `100730` is still running on `server13`, stderr still only has
  the GPU1 handle warning, and the only failure class remains
  `incomplete_ensemble_members`. The submitted contract is `EPOCHS=50` plus
  `MIN_TRAIN_SECONDS=10800`, exact96 RGB-D, 4xH200, RGB-D image plus
  robot-proprio input. Members 0/2/3 are far beyond epoch 300; member 1 is
  slow but has advanced to epoch 4 and has a `best_model.pt`, so this is a
  hardware/performance risk rather than a concrete failure. Same-shape
  replacement probes are later (`server13` allowed:
  `2026-06-04T10:32:10`; `--exclude=server13`:
  `2026-06-04T14:38:13`), so the aligned action is to keep monitoring the
  running job, not cancel or submit a duplicate. Live RMSE values remain
  diagnostics only; no RGB-D world-model, controller, video, or method
  evidence exists yet.
- 2026-06-03 11:50 strong-backup still-live triage:
  `100730` remains running on `server13` at about `01:00:08/04:00:00`.
  Read-only triage
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100730/triage_live_progress_split_20260603_1150.md/json`
  still reports only `incomplete_ensemble_members`; stderr still contains
  only the startup GPU1 handle warning. Members 0/2/3 are beyond epoch
  `640`, and member 1 has advanced to epoch `10` with a fresh
  `best_model.pt` at `11:43:59`, so the straggler remains a
  hardware/performance risk rather than a concrete completion failure. The
  job has not produced final `model.pt`/`metrics.json`, strict slot
  inspection `100731` remains dependency-pending, and all live RMSE values
  remain diagnostic only. No cancel, duplicate submission, threshold change,
  or method claim was made.
- 2026-06-03 12:15 strong-backup downstream CPU-export tail replacement:
  while `100730` remained running, submitted an aligned downstream
  replacement tail to remove a concrete scheduling risk in the pending
  strong-backup chain. The old predicted-slot export `100732` requested
  1 H200 on `gpu`; same-shape probes showed CPU-only export on `cpu` would
  start `2026-06-04T11:20:08`, while GPU export would start
  `2026-06-06T10:03:04`. World-model probes showed 4H200 `cpu,gpu`
  training starts `2026-06-04T12:25:04`, while `gpu` starts
  `2026-06-06T10:03:04`. Submitted replacement chain
  `strongslot128_grid8_stratified_cpu_export_wmcpugpu_tail_20260603_1215`
  with existing slot job `100730` and strict slot inspection `100731`,
  CPU-only predicted-slot export `101009`, strict predicted-slot inspection
  `101010`, 4H200 RGB-D-derived world-model `101012`, controller/video jobs
  `101015-101031`, and unchanged RGB-D gates. Contract audit passed
  `203/203`, then old pending tail `100732-100754` and old afterany
  diagnostics `100758-100767` were canceled before allocation. This is
  scheduling hygiene only, not RGB-D method evidence and not a gate change.
- 2026-06-03 14:35 strong-backup slot completion and strict inspection:
  `100730` completed on `server13` at `2026-06-03T14:31:03+08:00` after
  `03:40:28`, exit `0:0`. Strict slot inspection `100731` completed at
  `14:31:26`, and afterany inspection/triage `100757/100760` also completed.
  Inspection artifacts under
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job100730/`
  record exact `96/96` RGB-D files, base/hand RGB+depth input (`8x128x128`),
  robot-only qpos/qvel proprio, no oracle object/TCP inference input, four
  complete members, minimum elapsed `11489` seconds, requested
  `gpu:NVIDIAH200:4`, and compliant 3h/4H200 slot-training evidence.
  Validation aggregate is `hole_pos_rmse_m_mean=0.06361`,
  `peg_head_hole_rmse_m_mean=0.08954`, `peg_pos_rmse_m_mean=0.05735`,
  `tcp_pos_rmse_m_mean=0.04782`. This is compliant RGB-D perception component
  evidence only. Downstream CPU-only predicted-slot export `101009` started on
  `server52` at `14:32:22` and remains the active next gate; no RGB-D-derived
  world-model/controller/video/method evidence exists yet.
- 2026-06-03 14:56 strong-backup predicted-slot handoff and world-model start:
  CPU-only RGB-D slot export `101009` completed on `server52` at
  `2026-06-03T14:53:00+08:00` after `00:20:38`, writing exact `96`
  predicted-slot H5 files from RGB-D observations. Strict inspection `101010`
  completed at `14:53:25` with `valid_export=True`,
  `quality_gate_passed=True`, `96` files, `96` trajectories, `0` warnings,
  and unchanged gate metrics `hole_pos_rmse_m=0.01894`,
  `peg_head_hole_rmse_m=0.02767`, `peg_pos_rmse_m=0.01462`,
  `binary_accuracy=0.99834`. Slot visual review `101030` completed and the
  contact sheet was opened directly; it is nonblank/readable with RGB-D frames
  and predicted/oracle slot overlays. Sensitivity diagnostic `101011`
  completed with handoff mismatch `0.00512`, insert-yz mismatch `0.04724`,
  and bridge-close mismatch `0.03298`, diagnostic only. RGB-D-derived
  world-model training `101012` started on `server13` at `14:54:23` with
  exact `96` predicted-slot inputs, `input_representation=rgbd_predicted_slots`,
  `oracle_slots_not_used=true`, 4xH200, and `MIN_TRAIN_SECONDS=10800`.
  This is RGB-D-derived slot handoff evidence and world-model startup only;
  no completed world-model/controller/video/method evidence exists yet.
- 2026-06-04 01:56 full1000 visualization-budget correction:
  user clarified that rendered/visual demo artifacts should be limited to
  `10` samples, while all other work must continue and training must use all
  exact `1000` RGB-D demos. Applied this boundary only to diagnostic
  visualization jobs, not to data generation, slot training, predicted-slot
  export, world-model training, controller evaluation, or gates. Canceled
  pending old visual-only jobs `102179/102182/102185/102188/102191`,
  predicted-slot visual review `102192`, and evidence review `102193`;
  submitted replacements `102464` for predicted-slot visual review with
  `SAMPLE_FILES=10,FRAMES_PER_FILE=4`, video artifact review jobs
  `102465-102469` with `SAMPLE_COUNT=10`, and evidence review `102470`.
  Updated chain metadata and reran the static contract audit:
  `contract_audit_after_visual10_replacement.md/json` passed `206/206`,
  preserving exact `1000` RGB-D inputs, strict file-count gates, RGB-D
  predicted slots, RGB-D-derived world model, and `SLOT_SOURCE=rgbd`
  controller branches. Slot training `102169` remains the live main path on
  exact `1000`; it completed epoch-0 validation and wrote four `best_model.pt`
  checkpoints, then entered epoch 1 because `MIN_TRAIN_SECONDS=10800` was not
  yet satisfied. A direct walltime extension to `08:00:00` was denied for
  `102169`, so backup job `102292` remains dependency-pending on
  `afternotok:102169` with an `08:00:00` walltime and exact `1000` input. If
  `102169` succeeds, cancel backup `102292-102316`; if it times out, classify
  it as scheduling/walltime risk and let the exact1000 backup path continue.
  This is visualization and scheduling hygiene only, not method evidence.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_full1000_rgbd_generation_submission.md`.
- 2026-06-04 02:03 full1000 backup chain takeover:
  inspected `train_rgbd_slot_extractor.py` and confirmed the slot trainer
  checks `MIN_TRAIN_SECONDS` only at epoch boundaries:
  `while epoch < args.epochs or elapsed < args.min_train_seconds`. Main job
  `102169` had entered epoch 1 and reached only batch `300/1750` at
  `02:46:14/04:00:00`, so it could not finish epoch 1, write
  `model.pt/metrics.json`, and trigger `afterok` before the walltime. Canceled
  `102169` and dead main-chain tails `102170-102191` plus replaced visual jobs
  `102464-102470` to avoid wasting 4 H200 GPUs and dependency queue clutter.
  This is a scheduling/walltime correction, not RGB-D method failure or slot
  quality evidence. Backup job `102292` started on `server62` with exact
  `1000` RGB-D input and `08:00:00` walltime; strict downstream backup jobs
  remain dependency-pending. Also replaced backup visual-only jobs
  `102302/102305/102308/102311/102314/102315/102316` with 10-sample jobs
  `102483-102489`; backup contract audit after replacement passed `206/206`.
  Continue monitoring `102292` through strict slot inspection, predicted-slot
  export/inspection, RGB-D-derived world model, controller metrics, and direct
  video/contact-sheet inspection.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_full1000_rgbd_generation_submission.md`.
- 2026-06-04 02:32 full1000 backup slot load passed:
  backup slot job `102292` is running on `server62` and passed full1000
  metadata load for all four members. The member load events report
  `num_samples=301000`, `image_shape=[301000, 8, 128, 128]`,
  `proprio_shape=[301000, 18]`, `target_cont_shape=[301000, 25]`,
  `target_bin_shape=[301000, 8]`, and `lazy_images=true`. Load time was slow
  (`1646-1684` seconds) but active, consistent with shared-filesystem/H5
  metadata I/O, and no stderr errors appeared. Training has entered epoch 0
  and reached batch `50` for all four members. This is full1000 slot-training
  startup/readiness evidence only; keep monitoring until strict inspection
  `102293` passes before predicted-slot export `102294` can count as a valid
  RGB-D-derived handoff.
  Evidence note:
  `docs/world_model_task_rebinding/2026-06-03_full1000_rgbd_generation_submission.md`.
- 2026-06-04 continuation memory hygiene:
  updated `AGENTS.md` so future agent turns preserve the same boundary:
  current full1000 RGB-D visual/demo artifacts are limited to `10` diagnostic
  samples, but this must not reduce exact `1000` RGB-D training, predicted-slot
  export, RGB-D-derived world-model training, controller inputs, controller
  metrics, strict gates, or the full1000 data requirement. This records the
  user-requested limit globally and prevents future scope drift.
- 2026-06-04 full1000 predicted-slot export repair:
  `102292` completed exact1000 RGB-D slot training and `102293` strict slot
  inspection passed. Predicted-slot export `102294` on `server13` hit a
  concrete GPU/node scheduling failure (`Unable to determine the device handle
  for GPU0`) and timed out at `04:00:00` after writing only `598/1000`
  predicted-slot H5 files, with no `predicted_slot_h5s.txt` or
  `export_report.json`. This is a scheduling/node-GPU plus export walltime
  failure, not RGB-D perception quality, world-model, controller, physics, or
  evaluation evidence. Patched export with `--resume-existing` so valid partial
  H5s can be reused and only missing exact1000 files are computed; added CPU
  Slurm resume wrapper. Submitted resume export `103815`
  (`RESUME_EXISTING=true`, `CUDA_EXPORT=false`, same `job102294` output dir,
  `EXPECTED_RGBD_FILES=1000`) and initially chained strict export inspection
  `102295` to that primary resume without changing thresholds. Old visual review `102483`
  ran on the partial export and is not full1000 visual evidence. `103815`
  started on `server31` but used the pre-optimization row-wise existing-H5
  resume path and was canceled after `00:53:42` to avoid wasting the compute
  allocation. Submitted optimized exact1000 resume job `103904`
  (`RESUME_EXISTING=true`, `CUDA_EXPORT=false`, same `job102294` output dir)
  and updated strict export inspection `102295` plus replacement 10-sample
  visual review `103816` to `afterok:103904`. This preserves exact `1000`
  RGB-D predicted-slot export and the original quality gates; it only limits
  human-readable visual diagnostics to `10` samples.
- 2026-06-04 full1000 predicted-slot quality failure and aligned retry:
  optimized resume export `103904` completed the structurally exact `1000`
  RGB-D predicted-slot export (`301000` samples; `598` reused and `402`
  recomputed), but strict inspection `102295` failed the unchanged task-frame
  quality gate: `hole_pos_rmse_m=0.031467>0.030000` and
  `peg_head_hole_rmse_m=0.054956>0.035000`. Sensitivity job `102296` confirms
  binary predicates are strong but continuous task-frame geometry remains too
  noisy (`peg_head_at_hole_rmse_m=0.042367`, `insert_yz_gate` mismatch
  `0.064791`). Replacement 10-sample visual review `103816` completed with
  `0` warnings and was opened directly; the sheet is nonblank, with visible
  target/prediction separation in several rows, consistent with the strict
  failure. This is a RGB-D perception quality failure, not method evidence and
  not a reason to train the world model on failed slots or relax gates.
  Patched future slot training with explicit task-frame loss weights that
  default to old behavior, then submitted exact1000 replacement chain
  `full1000_strongslot128_grid8_taskloss_20260604_1848`: slot job `105236`
  uses 4 H200, `16:00:00`, `MIN_TRAIN_SECONDS=10800`, `CNN_CHANNELS=128`,
  `HIDDEN_DIM=512`, `SPATIAL_GRID_SIZE=8`, `HOLE_POS_LOSS_WEIGHT=2.0`,
  `PEG_HEAD_HOLE_LOSS_WEIGHT=4.0`, `EPOCHS=2`, and `LAZY_IMAGES=true`.
  Downstream jobs `105237-105260` preserve exact `1000`, strict
  predicted-slot gates, RGB-D-derived world-model/controller boundaries, and
  10-sample visual diagnostics. Contract audit passed `206/206`. Old
  failed-branch dead dependencies `102297-102313` and `102484-102489` were
  canceled before allocation. Current active Slurm state: `105236` is pending
  with reason `Priority` and no authoritative `StartTime` yet. Evidence note:
  `docs/world_model_task_rebinding/2026-06-04_full1000_predicted_slot_quality_failure_and_strong_retry.md`.
- 2026-06-04 18:29 task-loss contract audit hardening:
  patched `audit_rgbd_method_chain_contract.py` so future chain audits verify
  `HOLE_POS_LOSS_WEIGHT` and `PEG_HEAD_HOLE_LOSS_WEIGHT` match the submitted
  manifest. Refreshed audit
  `full1000_strongslot128_grid8_taskloss_20260604_1848/contract_audit_after_taskloss_weight_check.md`
  passed `208/208`, including the two new checks. `squeue --start` now
  forecasts slot job `105236` for `2026-06-04T23:00:00` on `server08`, while
  `scontrol` still shows it pending with reason `Priority`. This is
  submission/readiness evidence only; wait for real Slurm allocation and slot
  logs before interpreting training quality. Fresh read at
  `2026-06-04T18:30:29+08:00` moved the forecast earlier to
  `2026-06-04T22:00:00`, with `105236` still `PENDING` for `Priority`. Treat
  live Slurm state as authoritative and do not submit a duplicate while this
  exact1000 aligned path is forecast to start soon.
- 2026-06-04 18:32 slot-inspection task-loss evidence hardening:
  patched `inspect_rgbd_slot_extractor_ensemble.py` so future strict slot
  inspection records task-frame loss weights and a non-gating
  `task_frame_loss_weight_evidence` field. This changes only evidence
  reporting; it does not alter compliant training gates, exact RGB-D gates,
  predicted-slot quality gates, world-model inputs, controller metrics, or
  visual budgets. `py_compile`, `bash -n` for the inspection wrapper, and a
  `.venv` read-only smoke on old exact1000 slot job `102292` passed. The old
  smoke still reports `rgbd_slot_training_evidence=True` but
  `task_frame_loss_weight_evidence=False`, preserving the old failure
  classification while preparing `105237` to prove the new repair metadata
  after `105236` completes. Latest `squeue --start` forecast for `105236`
  moved earlier again to `2026-06-04T21:00:00`; the job remains `PENDING` for
  `Priority`, so do not duplicate the exact1000 training path.
- 2026-06-04 18:37 waiting-period dependency recheck:
  `105236` remains `PENDING` for `Priority`, with `squeue --start`
  forecasting `2026-06-04T21:00:00`; no `wm_rgbd_slot4-105236.out/err` logs
  or `job105236` artifacts exist yet. Read-only `jobs.tsv`, `scontrol`, and
  refreshed audit checks confirm the chain still preserves the full method
  gates: slot training is exact `1000` with task-frame loss weights, strict
  predicted-slot inspection `105239` keeps the original thresholds,
  RGB-D-derived world-model training `105241` depends on `afterok:105239`, and
  visual review `105259` is only the 10-sample diagnostic branch after export.
  Temporary audit `/tmp/full1000_chain_wait_audit.md` passed `208/208`. This
  is waiting-period readiness evidence only, not slot quality or method
  evidence; do not submit a duplicate while the aligned exact1000 path is
  forecast to start soon.
- 2026-06-04 18:42 user visual-budget confirmation and pending-job triage:
  user clarified that only rendered/visual demo artifacts should stay capped
  at `10` diagnostic samples; RGB-D slot training, predicted-slot export,
  strict inspections, RGB-D-derived world-model training, controller inputs,
  controller metrics, and exact-count gates must still use all `1000` RGB-D
  demos. Fresh Slurm state still has `105236` `PENDING` for `Priority`, with
  `squeue --start` and `scontrol` forecasting `2026-06-04T21:00:00` and
  `SchedNodeList=server08`; no logs or `job105236` artifacts exist. `jobs.tsv`
  confirms `105259` visual review is `SAMPLE_FILES=10` while export/inspection
  gates remain `MIN_FILES=1000` and `EXPECTED_FILES=1000`, and `105241` still
  depends on `afterok:105239`. Refreshed contract audit passed `208/208`.
  Lightweight pending-job triage wrote
  `experiments/world_model_task_rebinding/rgbd_method_chains/full1000_strongslot128_grid8_taskloss_20260604_1848/slot105236_pending_triage_20260604_184216.{json,md}`
  and classified the state as `scheduling_pending`, `pending_reason_priority`,
  and `no_ensemble_artifacts_yet`. This is readiness evidence only; keep the
  current exact1000 path queued and recheck when it starts or on the normal
  cadence.
- 2026-06-04 18:45 submitted snapshot recheck:
  after re-reading the active and focused RGB-D TODO files, live Slurm state
  still shows `105236` `PENDING` for `Priority` with `squeue --start`
  forecasting `2026-06-04T21:00:00`, `SchedNodeList=server08`, `16:00:00`,
  and `gres/gpu=4` H200; no stdout/stderr logs or `job105236` artifacts exist.
  `sacct SubmitLine` confirms the submitted environment, not just wrapper
  defaults: `105236` uses exact `1000` RGB-D files, `MIN_TRAIN_SECONDS=10800`,
  task-frame loss weights `2.0/4.0`, and `EPOCHS=2`; `105238` exports exact
  `1000` frame-aligned predicted slots; `105239` keeps the unchanged strict
  thresholds; `105241` requires exact `1000` predicted-slot files and depends
  on `afterok:105239`; `105259` is the diagnostic visual branch with
  `SAMPLE_FILES=10`, `FRAMES_PER_FILE=4`, and exact `1000` file gates.
  Submitted-chain audit `/tmp/full1000_chain_snapshot_audit.md` passed
  `208/208`, including RGB-D controller/video/nonblank/oracle-exclusion
  checks. This remains readiness evidence only, not method evidence.
- 2026-06-04 18:47 queue hygiene recheck:
  filtered live queue shows only the active replacement RGB-D method chain
  `105236-105260` among relevant `wm_*` jobs. Old failed-branch jobs
  `102297-102313` and `102484-102489` are all `CANCELLED by 2059` with
  `Elapsed=00:00:00` and `NodeList=None assigned`, so they are not consuming
  GPU allocations or dependency execution slots. This is resource hygiene
  evidence only; it does not prove RGB-D slot quality or method success.
- 2026-06-04 18:48 input/output readiness recheck:
  `105236` remained pending, so the waiting interval was used for non-GPU
  readiness checks. `AGENTS.md` still records the full1000 boundary: visual
  demo artifacts are capped at `10` diagnostic samples, but exact `1000` RGB-D
  remains required for slot training, predicted-slot export, world-model
  training, controller inputs/metrics, strict gates, and the full1000 data
  requirement. The submitted RGB-D input root still contains exactly `1000`
  `.rgbd.h5` files. The `job105236` output directory is absent and
  `wm_rgbd_slot4-105236.out/err` are absent, so there are no stale
  checkpoints/logs to contaminate the replacement run. This is clean-surface
  readiness evidence only, not method evidence.
- 2026-06-04 18:51 afterany slot-triage diagnostic:
  submitted CPU diagnostic job `105316` with dependency `afterany:105236`,
  `cpu=2`, `mem=4G`, and `time=00:20:00`, writing to
  `full1000_strongslot128_grid8_taskloss_20260604_1848/slot105236_afterany_triage`.
  Added `diagnostic_jobs.tsv` in the chain directory to keep this separate
  from method jobs. `scontrol` confirms `105316` is dependency-pending on
  `105236`; `sacct SubmitLine` confirms it only exports `TARGET_JOB_ID=105236`,
  the `job105236` ensemble dir, and the diagnostic output dir. This is
  read-only failure-localization only; it does not approve slot quality,
  replace strict inspection `105237`, trigger world-model training, or change
  any gate.
- 2026-06-04 18:53 PLAN drift recheck after diagnostic:
  re-read `PLAN/README.md`, `00_overview.md`, `05_rebinding_controller.md`,
  and `06_rgbd_and_baselines.md`. The active chain still matches the plan:
  RGB-D is required method evidence, oracle/state slots are only scaffolds,
  the RGB-D path trains object/task slots rather than a pixel world model from
  scratch on 1000 demos, and the controller must use object slots, predicted
  task frames, uncertainty, and conservative handoff rather than case-name
  branches. Refreshed contract audit
  `/tmp/full1000_chain_after_diagnostic_audit.md` passed `208/208` after
  adding diagnostic job `105316`; the method chain still preserves exact1000
  inputs, strict predicted-slot gates, RGB-D-derived world-model/controller
  boundaries, video review branches, and oracle-source exclusion. This is
  plan-hygiene/readiness evidence only, not method evidence.
- 2026-06-04 18:55 same-shape scheduling probe:
  `105236` remains `PENDING` for `Priority` with
  `StartTime=2026-06-04T21:00:00`, `SchedNodeList=server08`, and `gres/gpu=4`
  H200. `server08` is currently `ALLOCATED` with `Gres=gpu:NVIDIAH200:8` and
  `AllocTRES=...gres/gpu=6`, consistent with waiting for resources to free.
  A non-submitting same-objective `sbatch --test-only` probe using the same
  exact1000 slot-training environment forecast `2026-06-05T00:23:29`, later
  than active `105236`; `squeue` and `sacct` confirm the probe id was not an
  actual submitted job. Decision: do not submit a duplicate/replacement; keep
  the active exact1000 path queued. This is scheduling evidence only.
- 2026-06-04 19:00 slot-wrapper manifest boundary metadata fix:
  static wrapper check found only a stale manifest text field in
  `train_rgbd_slot_extractor_ensemble_4gpu.sbatch`: the `boundary=` line still
  mentioned `formal_full96_dataset`. Patched the source wrapper to say
  `formal_inspected_rgbd_dataset_with_exact_expected_count` instead. This is a
  metadata wording fix only; it does not change Slurm resources, data paths,
  training loss, thresholds, dependencies, export behavior, world-model inputs,
  controller metrics, or visual budgets. Because `105236` was already queued,
  `scontrol write batch_script 105236` still contains the old string in the
  submitted snapshot, but the authoritative `sacct SubmitLine` remains exact
  `1000` via `MIN_RGBD_FILES=1000` and `EXPECTED_RGBD_FILES=1000`. `bash -n`
  passed and refreshed audit
  `/tmp/full1000_chain_after_manifest_boundary_fix_audit.md` passed `208/208`.
  At `2026-06-04T19:00:10+08:00`, `105236` was still pending with forecast
  `2026-06-04T20:56:37` and no logs/artifacts. This is metadata/readiness
  hygiene only, not method evidence.
- 2026-06-04 19:21 one-H200 resource override and live allocation:
  user replaced the old training floor with at least `1` H200 GPU and at least
  `10800` seconds; do not force 4 H200 as the minimum. Kept already-running
  exact1000 slot job `105236` on `server08` because canceling a live 4-H200
  allocation would waste the active full1000 repair. Created tmux session
  `h200_1gpu_pool` and submitted reusable allocation `105385` (`1` H200,
  `8` CPU, `64G`, `1-00:00:00`, no node exclusions); Slurm granted it on
  `server62` at `2026-06-04T19:18:38+08:00`. Repeated `srun --jobid=105385`
  checks confirmed H200 visibility and `torch.cuda.is_available()==True`, so
  the allocation can run multiple experiments without releasing. Patched
  AGENTS, active training wrappers, and strict inspectors so future training
  evidence accepts `>=1xH200/>=3h` while still requiring exact data/gates.
  `bash -n`, `py_compile`, and source text scans passed. Started an exact
  full1000 1-H200 backup slot training inside `105385` at
  `rgbd_slot_extractor/ensemble_1h200_pool/job105385_strongslot_seed1600`
  with the same strongslot/task-loss settings. Submitted bounded 1-H200
  downstream tail
  `full1000_strongslot128_grid8_taskloss_1h200_tail_20260604_192147`
  reusing upstream jobs `105236-105239`; new world-model/controller branch is
  `105429-105447` and contract audit passed `208/208`. This is resource/
  readiness progress only, not RGB-D method evidence until strict
  predicted-slot, RGB-D-derived world-model, controller, and video/contact
  gates pass.
- 2026-06-04 19:26 one-H200 tail cleanup and slot-training progress:
  canceled superseded old 4-H200 downstream jobs `105241-105258` and old
  evidence review `105260`; `sacct` confirms every canceled job had
  `Elapsed=00:00:00`, no `AllocTRES`, and `NodeList=None assigned`. Kept
  upstream `105236-105240`, 10-sample visual diagnostic `105259`, slot triage
  `105316`, reusable allocation `105385`, and audited 1-H200 method tail
  `105429-105447`. Live `105236` logs show exact1000 dataset load for all
  four members (`301000` samples, `[301000, 8, 128, 128]` RGB-D tensors) plus
  epoch-0 batch-0 and batch-50 training events for every member, with losses
  dropping into roughly `0.61-0.85` at batch 50. The 1-H200 backup
  `job105385_strongslot_seed1600` has started member `0` with exact1000
  strongslot/task-loss settings but has no checkpoint or strict inspection yet.
  At `2026-06-04T19:32+08:00`, `105236` had reached epoch-0 batch `150` for
  all four members, with losses about `0.35-0.51`. The 1-H200 backup had not
  emitted `dataset_loaded` yet, but a read-only overlap check on `server62`
  showed the Python training process alive and `sstat` showed about `2GB` disk
  reads, consistent with CPU/HDF5 scanning before GPU training. It was left
  running.
  This is training-progress/queue-hygiene evidence only, not slot-quality or
  method evidence.
- 2026-06-04 19:41 second one-H200 request and path-file transport fix:
  live `scontrol` shows the active resource path is not using a standing
  bad-node list: `105236`, `105385`, and dependency-pending `105429` all have
  `ExcNodeList=(null)`. No-exclusion `1`-H200 probes on `cpu` forecast the
  same start, `2026-06-04T22:55:25`, for `03:00:00`, `08:00:00`, and
  `1-00:00:00`; `gpu` partition forecast `2026-06-10T15:08:25`, so it was
  ignored. Submitted second real tmux allocation `h200_1gpu_pool2` / Slurm
  `105502` (`1` H200, `8` CPU, `64G`, `1-00:00:00`, no exclusions) and
  preloaded the tmux pane so that once it starts it runs
  `scripts/slurm/run_rgbd_slot_backup_in_allocation.sh` rather than idling.
  Patched `train_rgbd_slot_extractor.py` plus the slot wrapper to support
  `--paths-file`, avoiding member-process argv expansion of all `1000` H5
  paths. Added the allocation-runner script with an explicit RGB-D method
  boundary. `bash -n`, `py_compile`, and CLI help checks passed. This
  preserves exact1000 data, task-frame losses, strict gates, RGB-D-derived
  world-model/controller requirements, and the 10-sample visual cap; it is
  resource/implementation hygiene only, not method evidence.
  Follow-up `scontrol` at `2026-06-04T19:43+08:00` forecast `105502` at
  `2026-06-04T20:53:39` on `server62`, still with `ExcNodeList=(null)`.
  `105236` reached epoch-0 batch `400` for all four members with empty
  stderr; strict slot quality remains unproven.
- 2026-06-04 19:49 105502 live CUDA failure and replacement:
  `105502` started on `server13` and launched the intended exact1000 seed
  `1700` backup with `num_paths=1000` and `--paths-file`, but live canaries in
  the allocation found no usable CUDA device: `nvidia-smi` reported no devices
  and `.venv` PyTorch reported `torch_cuda_available False`,
  `torch_device_count 0`. Because the trainer previously fell back to CPU,
  canceled `105502` after `00:02:38` before dataset load/checkpoint and wrote
  `rgbd_slot_extractor/ensemble_1h200_pool2/job105502_strongslot_seed1700/cancel_reason.md`.
  This is scheduling/node GPU visibility evidence, not method evidence. Added
  default `require_cuda=True` to the RGB-D slot trainer, wrapper support for
  `REQUIRE_CUDA`, and runner `REQUIRE_CUDA=true`; `py_compile`, `bash -n`,
  and CLI help checks passed. Re-probes showed no-exclude would return to the
  current failed `server13`, while excluding only `server13` forecasts
  `server40`; submitted replacement tmux allocation `h200_1gpu_pool3` /
  Slurm `105535` (`1` H200, `1-00:00:00`, `ExcNodeList=server13`, seed
  `1800` command preloaded). `105236` reached epoch-0 batch `500` for all four
  members with empty stderr. `105385` still has no `dataset_loaded` event, but
  disk reads continue increasing, so it remains an active slow-scan branch.
- 2026-06-04 20:01 CUDA-gated 1-H200 tail replacement:
  submitted replacement downstream chain
  `full1000_strongslot128_grid8_taskloss_1h200_tail_cudagate_20260604_1953`
  with jobs `105553-105571`, reusing upstream `105236-105240` and visual
  diagnostic `105259`. The replacement world-model trainer requests `1` H200
  for `1-00:00:00`, keeps `MIN_TRAIN_SECONDS=10800`, uses exact `1000`
  predicted-slot files from RGB-D slot export, and sets `REQUIRE_CUDA=true`.
  Contract audit passed `208/208` at `contract_audit.md/json`, confirming
  exact1000 RGB-D inputs, unchanged strict predicted-slot gates,
  RGB-D-derived world-model/controller boundaries, `SLOT_SOURCE=rgbd`, video
  review, and oracle-source exclusion. Canceled superseded old tail
  `105429-105447` after audit; `sacct` showed every old-tail job had
  `Elapsed=00:00:00`, no allocated TRES, and `NodeList=None assigned`. Live
  state: `105236` is running on `server08` with no `ExcNodeList` and has
  reached epoch-0 batch `700`; `105385` remains an alive slow HDF5 scan on
  `server62` with no `ExcNodeList`; `105535` is running on `server28` with
  only job-local `ExcNodeList=server13`, `require_cuda=true`, and
  `cuda_available=true`; `105553` is dependency-pending on `105239` with no
  `ExcNodeList`. This is scheduling/contract/training-progress evidence only,
  not RGB-D method success.
- 2026-06-04 20:14 no-standing-bad-node resource correction:
  live `scontrol` confirmed the active path is not using a standing bad-node
  list: `105236`, `105385`, and `105553` have `ExcNodeList=(null)`, while
  `105535` has only job-local `server13` from the immediately observed CUDA
  visibility failure in `105502`. Fixed RGB-D slot/generic training triage so
  normal `MIN_TRAIN_SECONDS` manifest text no longer creates a false
  `undersized_training_refusal`; `py_compile` passed and corrected `105236`
  triage now reports only `incomplete_ensemble_members`. Added a pre-dataset
  CUDA canary to `run_rgbd_slot_backup_in_allocation.sh`. Submitted one more
  no-exclusion tmux allocation `h200_1gpu_pool4` / Slurm `105646`, requesting
  `1` H200 for `1-00:00:00`; it is pending with
  `StartTime=2026-06-04T21:31:41`, `SchedNodeList=server43`, and will run
  exact1000 RGB-D strongslot seed `1900` after CUDA canary. This is resource
  hygiene and perception-risk reduction only, not method evidence.
- 2026-06-04 20:21 CPU-first GPU scheduling fix:
  live probes showed `cpu,gpu` one-H200 jobs forecast tonight while `gpu,cpu`
  forecasts `2026-06-10`, so partition order materially affects latency.
  Updated dependency-pending active chain jobs `105238`, `105553`, and video
  jobs `105556/105559/105562/105565/105568` to `Partition=cpu,gpu`, all still
  with `ReqNodeList=(null)` and `ExcNodeList=(null)`. Patched current RGB-D
  method wrappers/submitter defaults for predicted-slot export,
  RGB-D-derived world-model training, and RGB-D controller video to prefer
  `cpu,gpu`; `bash -n` passed. This is scheduling-only and preserves exact1000
  inputs, strict slot/predicted-slot gates, RGB-D-derived world-model,
  controller metrics, and video-review requirements. `105236` remains running
  with latest member batches `950-1000` and no checkpoint/metrics; `105385`
  has now loaded exact1000 data and started batch `0`.
- 2026-06-04 20:25 allocation-runner canary bug and replacement:
  no-exclusion allocation `105646` started on `server62` and failed after
  `00:00:12` because the new runner canary tested CUDA in the direct `salloc`
  shell, where `nvidia-smi`/PyTorch CUDA were not visible. Correct `srun`
  canaries inside existing allocations `105385` on `server62` and `105535` on
  `server28` both saw one H200 and `torch_cuda_available=true`, so this is an
  allocation-runner implementation bug, not method evidence and not a standing
  server62 exclusion. Patched `run_rgbd_slot_backup_in_allocation.sh` to run
  the pre-dataset CUDA canary through `srun --gpus-per-task=1`; `bash -n` and
  manual `srun` canary passed. Submitted no-exclusion replacement
  `h200_1gpu_pool5` / Slurm `105707`, 1 H200 for `1-00:00:00`, seed `1900`.
  Final live `scontrol` snapshot shows `StartTime=2026-06-04T23:00:00`,
  `SchedNodeList=server18`, and `ExcNodeList=(null)`.
- 2026-06-04 20:34 one-H200 rolling allocation correction:
  after the user clarified that small GPU jobs should be requested promptly
  and old bad-node history should not explain current queue behavior, audited
  the live path again. `105236`, `105385`, `105238`, `105553`, and RGB-D
  controller-video jobs still have `ExcNodeList=(null)`; `105535` has only the
  job-local `server13` exclusion from current CUDA failure `105502`. Fixed
  `srun`-canary replacement `105707` then started on `server13` with
  `ExcNodeList=(null)` and failed before dataset load:
  `failure_class=scheduling_node_gpu_visibility`,
  `nvidia_smi_returncode=255`, `torch_cuda_available=false`. This repeats the
  current server13 GPU-visibility failure and is scheduling evidence only, not
  RGB-D method evidence or a standing bad-node policy. Fresh probes showed
  one-H200 `03:00:00`, `08:00:00`, and `1-00:00:00` requests all forecast the
  same start under `cpu`/`cpu,gpu`, while `gpu,cpu` forecasts `2026-06-10`, so
  one day does not cost earlier backfill here. Submitted tmux allocation
  `h200_1gpu_pool6` / Slurm `105743`, `1` H200 for `1-00:00:00`, exact1000
  seed `2000`, with fixed `srun` CUDA canary; after the repeated `server13`
  failure, updated only this pending job with job-local
  `ExcNodeList=server13`. It then started earlier than forecast on `server43`
  at `2026-06-04T20:34:44`; fixed `srun` CUDA canary passed with one visible
  H200 and `torch_cuda_available=true`, and exact1000 seed `2000` training
  started at `20:35:10`. Current training status remains incomplete: `105236`
  is running with all four members at epoch-0 batch `1100`; `105385` is
  training exact1000 on one H200; `105535` is running with CUDA visible and
  still loading/scanning exact1000; `105743` is running with CUDA visible and
  just started loading/scanning exact1000; no slot checkpoint/metrics have
  passed strict inspection yet.
- 2026-06-04 20:36 visualization-budget audit:
  verified active predicted-slot visual diagnostic job `105259` was submitted
  with `SAMPLE_FILES=10` and the active chain manifest records
  `slot_visual_sample_files=10` plus `video_review_sample_count=10`. Patched
  future defaults in `visualize_rgbd_predicted_slots.sbatch`,
  `inspect_video_artifacts.sbatch`, and `submit_rgbd_distributed_shards.sh`
  to default to `10` diagnostic samples. `bash -n` passed. This only limits
  human-readable diagnostic visual artifacts; it does not reduce exact1000
  RGB-D slot training, predicted-slot export/inspection, RGB-D-derived
  world-model training, controller inputs, controller metrics, or strict gates.
- 2026-06-04 20:39 persistent allocation manifest hygiene:
  patched `run_rgbd_slot_backup_in_allocation.sh` so future manifests record
  `keep_allocation_after_run` and an explicit reuse reason: keep successful
  H200 allocations available for follow-up strict inspection/export/retry
  commands instead of releasing and requeueing. `bash -n` passed. Current live
  resources are already `105236` on 4 H200 plus one-H200 backups `105385`,
  `105535`, and `105743`; the no-new-allocation decision at this timestamp was
  superseded by the later user scheduling override below.
- 2026-06-04 20:50 user scheduling override applied:
  user clarified that small GPU requests should be made more aggressively and
  should not be explained by old bad-node history. Added
  `scripts/slurm/launch_rgbd_1h200_allocation_tmux.sh` so one-H200 tmux
  allocation requests are reproducible and automatically run the exact1000
  RGB-D strongslot backup inside the allocation after the fixed `srun` CUDA
  canary. `bash -n` passed. Initial one-day requests `105826/105827` were
  canceled while still pending with zero allocation because shorter one-H200
  probes forecast earlier backfill; replacement 12-hour requests are `105868`
  (`h200_1gpu_pool7`, no exclusions, seed `2100`) and `105867`
  (`h200_1gpu_pool8`, only job-local `ExcNodeList=server13`, seed `2200`).
  No standing bad-node list is in use: the main path and downstream jobs have
  no exclusions, and `server13` is only current job-local evidence from
  repeated CUDA visibility failures before dataset load. Current active
  training remains incomplete but progressing: `105236` reached epoch-0 batch
  `1250` for all four members, `105535` loaded exact1000 and started training,
  and `105385`/`105743` are alive in HDF5/I-O heavy exact1000 paths. No slot
  checkpoint/metrics have passed strict inspection yet, so there is still no
  full RGB-D method evidence.
- 2026-06-04 20:56 strict one-H200 backup assembly preflight:
  added `scripts/world_model/assemble_rgbd_slot_backup_ensemble.py` so
  completed one-H200 backup members can be combined only after each source
  satisfies exact1000, one-H200, `>=10800s`, RGB-D/proprio boundary, metrics,
  manifest, and checkpoint requirements. `py_compile` passed. A dry-run over
  current backups `105385`, `105535`, and `105743` returned status `65` as
  expected because all are still incomplete. This is fallback hygiene only and
  cannot be used as RGB-D method evidence before strict inspection/export/
  world-model/controller/video gates pass.
- 2026-06-04 21:05 live scheduling correction after user challenge:
  user pointed out that small jobs are expected to queue faster and that old
  bad-node explanations should not be reused. Re-audited live Slurm state:
  `105236` is running on `server08` with `ExcNodeList=(null)`;
  one-H200 backups `105385` on `server62` has `ExcNodeList=(null)`, `105535`
  on `server28` has only job-local `ExcNodeList=server13`, and `105743` on
  `server43` has only job-local `ExcNodeList=server13`; downstream
  `105238`, `105553`, and RGB-D controller-video jobs remain dependency
  pending with no standing exclusions. Fresh one-H200 `sbatch --test-only`
  probes for `03:00:00`, `06:00:00`, `12:00:00`, and `1-00:00:00` all
  forecast `2026-06-05T00:46:04` on `server18`, but a real no-exclusion
  `12:00:00` tmux allocation `105936` received a worse actual forecast
  (`StartTime=2026-06-05T12:00:00`, `SchedNodeList=server62`) and was
  canceled before allocation; `sacct` records `Elapsed=00:00:00`, no
  allocated TRES, and no node. Existing bounded requests `105868`
  (no exclusions, `SchedNodeList=server13`) and `105867` (only job-local
  `server13` exclusion, `SchedNodeList=server18`) both now forecast
  `2026-06-04T23:00:00`, so they are kept. Current training remains
  incomplete: `105236` reached epoch-0 batch `1350` for all four members,
  `105385` reached batch `250`, `105535` reached batch `50`, and `105743`
  is still exact1000 loading/scanning. No checkpoint/metrics or strict RGB-D
  slot inspection has completed, so there is still no full RGB-D method
  evidence.
- 2026-06-04 21:09 one-H200 allocation follow-through:
  no-exclusion canary allocation `105868` actually started on `server13` and
  failed after `00:00:17` before RGB-D data loading with
  `failure_class=scheduling_node_gpu_visibility`,
  `nvidia_smi_returncode=255`, and `torch_cuda_available=false`. This is a
  third current server13 CUDA visibility failure and remains scheduling
  evidence only. Replacement probes and real requests were tested without
  keeping bad forecasts: `105949` (`1-00:00:00`, job-local
  `ExcNodeList=server13`) and `105950` (`06:00:00`, job-local
  `ExcNodeList=server13`) were both canceled before allocation when their
  actual forecasts were worse or unavailable; `sacct` records `Elapsed=00:00:00`
  and no assigned nodes for both. Pending `105867` then started immediately
  on `server18`, fixed `srun` CUDA canary passed, and exact1000 RGB-D
  strongslot seed `2200` emitted `rgbd_slot_train_start` with
  `cuda_available=true`, `num_paths=1000`, and `min_train_seconds=10800`.
  Current live GPU work is therefore `105236` plus one-H200 backups `105385`,
  `105535`, `105743`, and `105867`; no extra one-H200 allocation is left
  pending. This reduces perception-gate scheduling risk without changing the
  RGB-D method boundary or evaluation gates.
- 2026-06-04 21:12 live RGB-D slot gate audit:
  re-read AGENTS and focused TODOs, then checked Slurm/tmux/logs/artifacts.
  `105236` is still running on `server08` at about `01:59:11/16:00:00`,
  exact1000 loaded, and latest visible main-member batches are around
  epoch-0 batch `1400` for members `0/1/2` and batch `1350` for member `3`;
  no `model.pt`, `best_model.pt`, `metrics.json`, or `inspection.json` exists
  yet. One-H200 backups are also running: `105385` batch `250`, `105535`
  batch `100`, `105743` still exact1000 loading/scanning, and `105867`
  exact1000 training started on `server18` with `cuda_available=true`.
  Downstream jobs `105237 -> 105238 -> 105239/105240/105259 -> 105553 ->
  105554 -> 105555 -> controller/video/review` remain dependency-pending.
  Re-audited the active chain contract (`208/208` pass): exact1000 data,
  strict predicted-slot thresholds, RGB-D-derived world-model requirements,
  `SLOT_SOURCE=rgbd`, and video review sample count `10` are preserved.
  Re-ran one-H200 backup assembly dry-run including `105867`; it correctly
  returned status `65` because all sources lack completed metrics/checkpoints
  and the `10800` second floor. Lightweight downstream preflight passed:
  `bash -n` for slot inspection/export/world-model/controller/video wrappers
  and `py_compile` for the corresponding Python entrypoints. No method
  evidence exists yet because the slot checkpoint/strict inspection gate has
  not completed.
- 2026-06-04 21:15 live triage without gate changes:
  rechecked Slurm at `2026-06-04T21:14:23+08:00`: useful GPU work is still
  `105236` plus one-H200 backups `105385`, `105535`, `105743`, and `105867`;
  no extra one-H200 allocation is queued. `105236` is still running
  (`02:02:45/16:00:00`) and downstream `105237+` remain dependency-pending.
  Artifact search still found no checkpoint, metrics, or inspection outputs.
  `105236` stdout shows all four members at epoch-0 batch `1400`, with
  member `2` reaching batch `1450`; stderr is empty. Backup panes show
  `105385` at batch `300`, `105535` at batch `100`, and `105743`/`105867`
  still before `dataset_loaded` after their exact1000 path scans. Wrote
  read-only diagnostic triage under
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/diagnostics/job105236_live_triage_20260604_2114/`;
  it reports job state `RUNNING`, failure class only
  `incomplete_ensemble_members`, exact1000 RGB-D image plus robot-proprio
  inputs, and validation splits covering all six canonical scenarios with
  `42` validation trajectories each. This confirms the current blocker is
  simply unfinished slot training, not an observed scheduling/log/artifact
  failure. It is not method evidence and does not replace strict inspection.
- 2026-06-04 21:20 one-H200 rolling allocation expansion:
  after the user clarified that small jobs should queue sooner and old
  bad-node explanations should not be reused, re-audited live Slurm state and
  submitted two more tmux-backed one-H200 allocation requests. The active main
  and downstream chain still does not use a standing exclusion list:
  `105236`, `105385`, `105238`, `105553`, and all controller-video jobs have
  `ExcNodeList=(null)`. The only current exclusions are job-local
  `server13` on backups `105535`, `105743`, `105867`, and new fallback
  `106001`, tied to the current pre-dataset CUDA visibility failures in
  `105502`, `105707`, and no-exclusion canary `105868`. Fresh probes showed
  one-H200 `03:00:00`, `06:00:00`, `12:00:00`, and `1-00:00:00` requests had
  the same initial forecast; therefore the new requests use `1-00:00:00` to
  reduce requeue churn. Submitted `106000` / tmux `h200_1gpu_pool9` with no
  exclusions and seed `2300` as a live recovery canary, and `106001` / tmux
  `h200_1gpu_pool10` with only job-local `ExcNodeList=server13` and seed
  `2400` as fallback. Subsequent `scontrol` shows both pending with
  `StartTime=2026-06-05T04:00:00`; `106000` is scheduled to `server07` with
  no exclusions, and `106001` is scheduled to `server21` with only
  `server13` excluded. Four-H200 probes forecast only
  `2026-06-05T06:09:15`, so no extra 4-H200 duplicate was submitted. Reducing
  one-H200 CPU count to `4` or `6` did not produce an earlier real forecast
  after the new queue state, so no weaker CPU request was added. This is
  scheduling/resource-risk control for the exact1000 RGB-D perception gate,
  not method evidence; no checkpoint, metrics, or strict slot inspection has
  completed yet.
- 2026-06-04 21:27 live queue replacement check:
  rechecked active jobs and artifacts. `105236` remains running on `server08`
  with no exclusions at about `02:12:35/16:00:00`; strict slot inspection
  `105237` is still `afterok:105236(unfulfilled)`, and there are still no
  `model.pt`, `best_model.pt`, `metrics.json`, or `inspection.json` artifacts.
  Parsed main stdout confirms exact1000 loaded (`301000` RGB-D samples) and
  the latest member progress is seed `1200` batch `1500`, seed `1201` batch
  `1450`, seed `1202` batch `1500`, and seed `1203` batch `1450`; stderr is
  empty. One-H200 backups remain alive: `105385` batch `350`, `105535` batch
  `150`, `105743` completed exact1000 dataset load after about `2967s`, and
  `105867` is still in normal early exact1000 loading. A fresh non-executing
  probe found a potentially earlier one-H200 start, so submitted bounded
  replacement `106053` / tmux `h200_1gpu_pool11`; real `scontrol` did not give
  it a stable earlier forecast, while existing `106000` and `106001` both
  moved earlier to `StartTime=2026-06-05T01:00:00`. Canceled `106053` before
  allocation and verified the tmux session exited. Kept `106000` (no
  exclusions, `SchedNodeList=server21`) and `106001` (only job-local
  `ExcNodeList=server13`, `SchedNodeList=server62`). This is queue hygiene
  and perception-risk reduction only; no RGB-D method evidence exists yet.
- 2026-06-04 21:29 final live check for this turn:
  no leftover diagnostic `srun nvidia-smi` query remains. `105236` stdout now
  shows all four main members reached epoch-0 batch `1500`; `105743` advanced
  past dataset loading and emitted batch `0`. `106000` and `106001` remain the
  only extra pending one-H200 allocations. The strict gate state is unchanged:
  `105237` is still dependency-pending, and no slot checkpoint, metrics, or
  RGB-D method/video evidence exists yet.
- 2026-06-04 21:30 bounded resource probe:
  rechecked live state at `2026-06-04T21:30:37+08:00`: `105236` is still
  running at about `02:18:59/16:00:00`; `105237`, `105238`, and `105553`
  remain dependency-pending; backups `105385`, `105535`, `105743`, and
  `105867` are running; pending `106000/106001` are still the only extra
  one-H200 allocations. Fresh one-H200 `sbatch --test-only` probes for
  `03:00:00`, `06:00:00`, `12:00:00`, and `1-00:00:00` on `cpu`/`cpu,gpu`
  forecast no earlier than `2026-06-05T01:39:26`, while `gpu` forecasts
  `2026-06-09T22:14:26`. Existing real pending requests `106000` and `106001`
  are earlier (`StartTime=2026-06-05T00:00:00`), so no new allocation was
  submitted. This preserves bounded queue usage and does not change any
  exact1000 RGB-D gate.
- 2026-06-04 21:31 short follow-up monitor:
  after a 30s read-only check, `105236` remains running at about
  `02:20:14/16:00:00`; `105237` is still dependency-pending and no checkpoint,
  metrics, or inspection artifact exists. Main slot stdout progressed to seed
  `1202` batch `1550`, with the other three members still last logged at
  batch `1500`. One-H200 backup panes are still alive (`105385` batch `350`,
  `105535` batch `200`, `105743` batch `0`, `105867` loading/scanning).
  This is live progress evidence only, not RGB-D method evidence.
- 2026-06-04 21:34 no-exclusion allocation started:
  pending `106000` / tmux `h200_1gpu_pool9` started early on `server43` at
  `2026-06-04T21:32:58` with `ExcNodeList=(null)`. The fixed `srun` CUDA
  canary completed successfully, and exact1000 RGB-D slot backup seed `2300`
  emitted `rgbd_slot_train_start` with `cuda_available=true`,
  `num_paths=1000`, `require_cuda=true`, and `min_train_seconds=10800`.
  Artifacts now include allocation/slot manifests and `rgbd_paths.txt` under
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_1h200_pool9/job106000_strongslot_seed2300/`.
  This is scheduling/progress evidence only. `106001` remains the only extra
  pending one-H200 allocation, forecast around `2026-06-04T22:19:42` with
  only job-local `ExcNodeList=server13`.
- 2026-06-04 21:36 final pending allocation started:
  `106001` / tmux `h200_1gpu_pool10` also started early on `server07` with
  only job-local `ExcNodeList=server13`. Its fixed `srun` CUDA canary passed
  (`torch_cuda_available=true`, one visible H200), and exact1000 RGB-D slot
  backup seed `2400` emitted `rgbd_slot_train_start` with
  `num_paths=1000`, `require_cuda=true`, and `min_train_seconds=10800`.
  There are now no extra one-H200 allocation requests pending. Running useful
  GPU work is the main 4-H200 slot job `105236` plus one-H200 exact1000
  backup allocations `105385`, `105535`, `105743`, `105867`, `106000`, and
  `106001`. This increases perception fallback coverage without changing
  exact1000 data, labels, strict slot inspection, predicted-slot gates,
  RGB-D-derived world-model requirements, or controller/video evidence rules.
- 2026-06-04 21:37 strict gate still pending:
  final read-only check for this turn found `105236` still running at about
  `02:25:49/16:00:00`; `105237`, `105238`, `105239`, and `105553` remain
  dependency-pending. Artifact search still finds no slot checkpoint, metrics,
  inspection, or allocation-runner completion status for the active main or
  backup slot runs. Main stdout now shows all four `105236` members at
  epoch-0 batch `1550`. No diagnostic `srun nvidia-smi` query remains.
- 2026-06-04 21:39 live gate and backup audit:
  rechecked Slurm at `2026-06-04T21:39:05+08:00`. `105236` is still running
  at about `02:27:27/16:00:00`, and `105237`, `105238`, `105239`, and
  `105553` remain dependency-pending. No checkpoint, metrics, inspection, or
  allocation-runner completion artifact exists for the main or backup slot
  runs. Main stdout progressed to seed `1202` batch `1600`, while seeds
  `1200`, `1201`, and `1203` are at batch `1550`. Backups continue to make
  progress where past loading: `105385` batch `400`, `105535` batch `250`,
  and `105743` batch `50`; `105867`, `106000`, and `106001` are still in
  their normal early exact1000 loading/scanning windows. There are no extra
  pending one-H200 allocations and no leftover diagnostic `srun` query.
  Downstream remains blocked only by the intended strict slot gate, not by a
  new observed failure.
- 2026-06-04 21:43 user resource correction follow-through:
  the user relayed current cluster guidance that small jobs should usually
  wait about `10-30` minutes and asked not to over-attribute delays to old
  bad-node history. Re-audited live Slurm state and wrappers before acting:
  main slot training `105236`, no-exclusion one-H200 backups `105385` and
  `106000`, and downstream GPU jobs `105238`, `105553`, and controller-video
  jobs all have `ExcNodeList=(null)`. Backups `105535`, `105743`, `105867`,
  and `106001` use only job-local `ExcNodeList=server13`, tied to current
  same-day CUDA visibility failures from no-exclusion allocation canaries
  (`105502`, `105707`, `105868`), not a standing bad-node list. Source
  wrappers expose only optional `EXCLUDE_NODES`; no fixed bad-node file was
  found. Fresh `sbatch --test-only` probes showed one-H200 `03:00:00`,
  `06:00:00`, `12:00:00`, and `1-00:00:00` requests all forecast the same
  start around `2026-06-05T01:00:54` on `server07`, with or without
  `server13` excluded. Because the forecast was identical, submitted one
  additional bounded no-exclusion one-H200 one-day tmux allocation
  `106147` / `h200_1gpu_pool11`, exact1000 seed `2500`, so a granted H200 can
  keep running follow-up strict RGB-D slot work without requeueing. Real
  `scontrol` reports `106147` pending with
  `StartTime=2026-06-04T22:56:00`, `SchedNodeList=server07`, and
  `ExcNodeList=(null)`; keep it bounded and do not submit a large fanout
  unless live failures or earlier real forecasts justify it. This resource action
  addresses the physical bottleneck that RGB-D slot perception gates are
  blocking RGB-D-derived world-model/controller experiments; it does not
  change exact1000 data, labels, strict slot thresholds, RGB-D-derived
  world-model requirements, controller inputs, video cap, or evaluation gates.
- 2026-06-04 21:48 strict gate/resource follow-up:
  rechecked main and fallback paths. `105236` remains running on `server08`
  at about `02:35:08/16:00:00` with `ExcNodeList=(null)`, stderr empty, and
  all four members last logged at epoch-0 batch `1600`; no `model.pt`,
  `best_model.pt`, `metrics.json`, or `inspection.json` exists yet. Strict
  gate `105237` and downstream `105238`, `105239`, and `105553` remain
  dependency-pending. One-H200 backups are still useful but incomplete:
  `105385` batch `450`, `105535` batch `250`, `105743` batch `100`, while
  `105867`, `106000`, and `106001` are still in their normal early full1000
  loading windows. The new no-exclusion allocation `106147` forecast slipped
  to `StartTime=2026-06-05T01:00:00`, `SchedNodeList=server21`,
  `ExcNodeList=(null)`; fresh probes for `03:00:00`, `06:00:00`,
  `12:00:00`, and `1-00:00:00` all forecast later
  (`2026-06-05T02:02:42`), with or without `server13`, so `106147` remains
  the bounded earliest pending request and no replacement was submitted.
  Contract audit remains pass `208/208`, and wrapper syntax/py_compile
  checks passed for slot inspection/export, predicted-slot inspection,
  RGB-D-derived world-model, controller-video, video-artifact inspection, and
  evidence review. Backup assembly dry-run over six active one-H200 sources
  returned status `65` with `all_eligible=false` because all sources are
  missing checkpoint/metrics and are below the `10800` second floor; this is
  the correct guard and prevents incomplete backups from becoming method
  evidence. No RGB-D method evidence or mp4 demo exists yet.
- 2026-06-04 21:49 final snapshot:
  `105236` is still running at about `02:37:46/16:00:00`; seed `1202`
  advanced to epoch-0 batch `1650`, while the other main members remain last
  logged at batch `1600`. Artifact search still finds no checkpoint, metrics,
  or inspection output. `105237`, `105238`, `105239`, and `105553` remain on
  the intended dependencies. `106147` remains pending; no new replacement was
  submitted after the later probes. This is live progress only, not RGB-D
  method evidence.
- 2026-06-04 21:51 continued live ETA/resource check:
  `105236` remains running with no exclusions and `EndTime=2026-06-05T11:11:38`.
  Parsed training logs show seed `1200`, `1201`, and `1202` at epoch-0 batch
  `1650`, with seed `1203` at batch `1600`. The member manifest gives
  `301000` samples, `VAL_FRACTION=0.25`, and `BATCH_SIZE=96`, so each epoch
  has about `2352` training batches; the main run is roughly `68-70%` through
  epoch 0 and still has enough walltime for the configured two epochs plus
  validation unless a new slowdown appears. No checkpoint, metrics, or
  inspection artifact exists. Backups continue as risk reduction only:
  `105385` batch `450`, `105535` batch `300`, `105743` batch `100`, while
  `105867`, `106000`, and `106001` have not yet emitted dataset-loaded lines
  in their visible panes. Pending no-exclusion allocation `106147` now
  forecasts `StartTime=2026-06-04T22:19:42` on `server21`, so it is kept; no
  additional job was submitted. This is scheduling/progress evidence only and
  cannot support a RGB-D method claim.
- 2026-06-04 21:52 no-exclusion one-H200 fallback started:
  `106147` / tmux `h200_1gpu_pool11` started earlier than forecast on
  `server62` with `ExcNodeList=(null)`. The fixed `srun` CUDA canary passed
  (`torch_cuda_available=true`, one visible H200), and exact1000 RGB-D slot
  backup seed `2500` emitted `rgbd_slot_train_start` with `num_paths=1000`,
  `require_cuda=true`, `min_train_seconds=10800`, and the same
  RGB-D-images-plus-robot-proprio/no-oracle-state boundary. Its manifests are
  under
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_1h200_pool11/job106147_strongslot_seed2500/`.
  This is useful perception fallback coverage only; it does not bypass
  `105236 -> 105237`, predicted-slot quality gates, RGB-D-derived
  world-model training, controller gates, or video evidence requirements.
- 2026-06-04 21:53 final live snapshot:
  `105236` is still running at about `02:41:46/16:00:00`; `105237`,
  `105238`, `105239`, `105553`, and controller/video jobs remain on their
  intended dependencies. `106147` is running at about `00:02:22/1-00:00:00`
  on `server62` as a no-exclusion one-H200 fallback. Artifact search still
  finds no checkpoint, metrics, strict inspection, RGB-D-derived world-model
  result, controller result, or mp4/video evidence. No method claim is
  supported yet.
- 2026-06-04 21:55 continued strict-gate monitor:
  re-read AGENTS/TODO, then checked Slurm, tmux, logs, and artifacts. `105236`
  remains running on `server08` with `ExcNodeList=(null)` at about
  `02:43:28/16:00:00`; stderr is still empty. Parsed stdout now shows all
  four main seeds `1200/1201/1202/1203` at epoch-0 batch `1650`. With the
  existing manifest-derived `2352` batches per epoch, the slowest member is
  about `70%` through epoch 0. No `model.pt`, `best_model.pt`,
  `metrics.json`, or `inspection.json` exists for the main or fallback slot
  runs, so strict gate `105237` and downstream `105238`, `105239`, `105553`,
  and controller/video jobs remain dependency-pending. One-H200 backups are
  still running only as perception fallback: `105385` batch `500`, `105535`
  batch `300`, `105743` batch `100`, and `105867/106000/106001/106147`
  started exact1000 slot training but have not emitted visible dataset-loaded
  lines yet. Contract audit remains pass `208/208`, and downstream wrapper
  syntax/py_compile checks still pass. This is progress/readiness evidence
  only; no RGB-D method evidence or mp4 demo exists.
- 2026-06-04 22:01 scheduling correction and bounded extra requests:
  after the user pointed out that small jobs should usually start quickly and
  that old bad-node history should not drive current scheduling, re-audited
  live Slurm state. The current main/downstream method jobs are not blocked by
  a broad exclusion list: main slot `105236`, running no-exclusion backups
  `105385/106000/106147`, predicted-slot export `105238`, RGB-D-derived
  world-model `105553`, and controller-video jobs all show
  `ExcNodeList=(null)`. Only some fallback jobs exclude `server13`, and only
  because same-day no-exclusion canaries found CUDA invisible before dataset
  loading; this remains job-local scheduling evidence. Submitted two bounded
  no-exclusion requests to increase useful one-H200 coverage without changing
  the method: tmux allocation `106237` (`h200_1gpu_pool12`, seed `2600`) and
  ordinary Slurm fallback `106245` (`wm_rgbd_slot_1h200_sbatch13`, seed
  `2700`). Both use the exact1000 RGB-D slot runner with `srun` CUDA canary,
  `EXPECTED_RGBD_FILES=1000`, `MIN_TRAIN_GPUS=1`, and
  `MIN_TRAIN_SECONDS=10800`. Latest `scontrol` shows `106237` forecasting
  `2026-06-05T03:00:00` on `server13` with `ExcNodeList=(null)`, so it is a
  no-exclusion test of current scheduler placement; if server13 still has the
  observed CUDA visibility issue, the runner will fail before RGB-D dataset
  loading. `106245` forecasts `2026-06-05T08:00:00` on `server07`, also with
  `ExcNodeList=(null)`. Main
  `105236` continues training and has reached at least epoch-0 batch `1700`
  for all four seeds, with stderr empty and no checkpoint/metrics yet. This is
  scheduling/progress evidence only; no RGB-D method evidence or mp4 demo
  exists until strict slot inspection, predicted-slot export/inspection,
  RGB-D-derived world-model training/eval, controller metrics, and visual
  review pass.
- 2026-06-04 22:05 strict-gate monitor:
  re-read AGENTS/TODO and checked live Slurm, main stdout/stderr, artifact
  directories, and tmux-backed one-H200 backups. `105236` remains running on
  `server08` at about `02:52:52/16:00:00`, with `ExcNodeList=(null)` and
  empty stderr. All four main seeds `1200/1201/1202/1203` have reached
  epoch-0 batch `1700`; no `model.pt`, `best_model.pt`, `metrics.json`,
  or `inspection.json` exists under `ensemble_4gpu/job105236`, so strict
  slot gate `105237` and all downstream RGB-D-derived jobs remain correctly
  dependency-pending. One-H200 fallbacks are still incomplete and cannot be
  evidence: `105385` reached batch `550`, `105535` batch `350`, `105743`
  batch `150`, `105867` finished exact1000 loading and reached batch `0`,
  while `106000/106001/106147` have started exact1000 slot training but have
  not yet emitted dataset-loaded lines in visible panes. New no-exclusion
  requests remain pending as scheduling risk reduction: `106237` forecasts
  `2026-06-05T03:00:00` on `server13` with CUDA canary required, and
  `106245` forecasts `2026-06-05T08:00:00` on `server07`. There is no RGB-D
  method evidence or mp4 demo yet.
- 2026-06-04 22:07 backup guard recheck:
  ran a dry-run-only `assemble_rgbd_slot_backup_ensemble.py` check over the
  seven active one-H200 fallback sources
  `105385/105535/105743/105867/106000/106001/106147`. The guard reported
  `all_eligible=false`, `num_sources=7`: every source is incomplete, missing
  member metrics/checkpoint artifacts, and below the `10800` second evidence
  floor; the newest three also lack member training manifests because they are
  still in the load/start window. This confirms the fallback path still cannot
  be used to bypass strict slot inspection, predicted-slot quality, or
  RGB-D-derived downstream gates.
- 2026-06-04 22:09 strict-chain dependency and artifact check:
  rechecked live dependencies and the submitted chain contract while main
  training is still running. `105237` remains `afterok:105236`,
  `105238` remains `afterok:105237`, `105239` remains `afterok:105238`, and
  `105553` remains `afterok:105239`; controller-video branches remain
  `afterok:105555` and still use `SLOT_SOURCE=rgbd`,
  `HOLE_PREDICTOR=world_model`, `SAVE_VIDEO=true`, and video review
  `SAMPLE_COUNT=10`. Contract audit is still pass `208/208`. Main `105236`
  is running on `server08` at about `02:56:51/16:00:00`, no exclusions, empty
  stderr, with latest visible batch `1750` for seeds `1200/1201/1202` and
  `1700` for seed `1203`. Artifact search still finds only member manifests,
  no checkpoint/metrics/inspection. One-H200 backups continue without
  evidence artifacts: `105385` batch `550`, `105535` batch `350`, `105743`
  batch `200`, `105867` batch `0`, and `106000/106001/106147` still before
  visible dataset-loaded lines. Pending no-exclusion jobs shifted forecast:
  `106237` now forecasts `2026-06-05T02:00:00` on `server07`, and `106245`
  forecasts `2026-06-05T08:00:00` on `server13`; both still require the
  CUDA canary before any RGB-D load. No failed gate or completed strict gate
  exists to act on yet, and no RGB-D method evidence or mp4 demo exists.
- 2026-06-04 22:11 training-floor boundary check:
  parsed `105236` stdout with the project venv and confirmed exact1000 slot
  training loaded `301000` samples after `588.038s`; with the configured
  `0.25` validation fraction and `BATCH_SIZE=96`, each epoch has about
  `2352` train batches. All four seeds `1200/1201/1202/1203` are at epoch-0
  batch `1750`, about `74.4%` through epoch 0. Live `scontrol` now shows
  `105236` has crossed the `03:00:00` training floor (`RunTime=03:00:04`),
  still running on `server08`, `ExcNodeList=(null)`, stderr empty. This does
  not create slot evidence because artifact search still finds no
  `model.pt`, `best_model.pt`, `metrics.json`, or strict `inspection.json`.
  `105237` and downstream remain dependency-pending.
- 2026-06-04 22:13 live artifact and fallback check:
  `105236` remains running on `server08` with `RunTime=03:01:55/16:00:00`,
  no exclusions, and dependency-free main training. Parsed stdout still shows
  all four seeds `1200/1201/1202/1203` at epoch-0 batch `1750`; no
  `rgbd_slot_epoch` or completion event has appeared. Artifact search under
  `ensemble_4gpu/job105236` still finds no checkpoints, metrics, history, or
  strict inspection output. The downstream chain remains structurally aligned:
  `105237 afterok:105236`, `105238 afterok:105237`,
  `105239 afterok:105238`, `105553 afterok:105239`. One-H200 fallback dirs
  `105385/105535/105743/105867/106000/106001/106147` still contain only
  member manifests, no metrics/checkpoints/status that could be assembled into
  evidence. Forecasts shifted but remain no-exclusion for the two pending
  requests: `106237` now forecasts `2026-06-05T00:00:00` on `server21`, and
  `106245` forecasts `2026-06-05T04:00:00` on `server13`; both keep the
  pre-load CUDA canary. No new job was submitted because useful main/fallback
  work is already running and no earlier bounded forecast/failure justified
  more queue pressure.
- 2026-06-04 22:16 contract drift check while waiting for slot artifacts:
  re-ran `audit_rgbd_method_chain_contract.py` against the active full1000
  chain and wrote
  `full1000_strongslot128_grid8_taskloss_1h200_tail_cudagate_20260604_1953/contract_audit_latest.{json,md}`.
  The current audit timestamp is `2026-06-04T22:16:05+08:00` and it passes
  `208/208`. The audit boundary explicitly says this is readiness evidence
  only, not slot quality, world-model performance, controller success, or
  method evidence. Main `105236` remains running with no checkpoint/metrics;
  latest parsed slot progress is seeds `1200/1202` at epoch-0 batch `1800`
  (`76.5%` of the derived `2352` epoch steps), and seeds `1201/1203` at
  batch `1750` (`74.4%`). Strict `105237` and all downstream gates remain
  dependency-pending.
- 2026-06-04 22:21 user scheduling correction follow-through:
  rechecked live Slurm state after the user relayed current cluster guidance
  that small jobs normally wait about `10-30` minutes and that old bad-node
  history should not explain current scheduling. The current main/downstream
  method path still has no broad bad-node exclusion: `105236` runs on
  `server08` with `ExcNodeList=(null)`, and downstream `105238`, `105553`,
  and controller-video jobs remain dependency-pending with no exclusions.
  No-exclusion one-H200 fallbacks `105385`, `106000`, `106147`, `106237`, and
  `106245` are running on `server62/server43/server62/server07/server21`.
  The only active exclusions remain job-local `server13` entries on
  `105535`, `105743`, `105867`, and `106001`, tied to same-day CUDA
  visibility canary failures rather than a standing node list. The two latest
  no-exclusion requests actually started quickly: `106237` started at
  `2026-06-04T22:17:10+08:00` on `server07`, and `106245` started at
  `2026-06-04T22:19:11+08:00` on `server21`. Both passed the pre-load CUDA
  canary and emitted `rgbd_slot_train_start` with `num_paths=1000`,
  `require_cuda=true`, `MIN_TRAIN_GPUS=1`, and
  `MIN_TRAIN_SECONDS=10800`. This corrects the scheduling interpretation:
  future requests should prefer no-exclusion one-H200 allocations and choose
  one day or a shorter walltime by live forecast, not by old bad-node history.
  The queue already has useful running H200 work, so no additional fanout was
  submitted at this moment. Main `105236` has crossed the three-hour floor and
  all four seeds are now at epoch-0 batch `1800/2352`; artifact search still
  finds no checkpoint, metrics, or strict inspection output, so this remains
  resource/progress evidence only. There is still no RGB-D method evidence or
  mp4 demo until strict slot inspection, predicted-slot export/inspection,
  RGB-D-derived world-model training/eval, controller metrics, and visual
  review pass.
- 2026-06-04 22:26 strict-gate and fallback guard check:
  re-read `AGENTS.md`, active TODO, and Slurm hygiene notes, then checked
  live jobs, logs, artifact directories, and dependency state. Main
  exact1000 RGB-D slot training `105236` remains running on `server08` with
  no exclusions, empty stderr, `RunTime` about `03:12:55/16:00:00`, and the
  exact1000 dataset loaded as `301000` samples. Derived epoch length remains
  `2352` train batches. Latest parsed progress is seed `1202` at epoch-0
  batch `1850` and seeds `1200/1201/1203` at batch `1800`; no epoch-complete,
  train-complete, checkpoint, metrics, or strict inspection artifact exists
  yet under
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job105236`.
  The strict dependency chain is unchanged and aligned:
  `105237 afterok:105236`, `105238 afterok:105237`,
  `105239 afterok:105238`, and `105553 afterok:105239`; controller-video
  jobs remain behind `105555` and have no node exclusions.

  The one-H200 fallback guard was re-run in dry-run mode over all nine active
  fallback sources `105385/105535/105743/105867/106000/106001/106147/106237/106245`
  and wrote
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/backup_assembly_guard_20260604_2225.json`.
  It returned status `65` with `all_eligible=false`, which is the correct
  protection: every source is still incomplete and missing member
  checkpoint/metrics artifacts, and none can prove the `10800` second
  training floor. This preserves the physical method boundary: partial
  one-H200 slot runs can reduce perception-training risk, but cannot bypass
  strict slot inspection, predicted-slot quality gates, RGB-D-derived
  world-model training, controller evaluation, or video evidence. No RGB-D
  method evidence and no mp4 demo exists yet.
- 2026-06-04 22:29 contract-audit path-equivalence repair:
  while waiting for slot artifacts, re-ran the active chain contract audit.
  The first invocation used a relative `--rgbd-root` for the same exact1000
  directory and produced two false failures:
  `slot_uses_exact_rgbd_root` and `predicted_export_uses_rgbd_root`. Concrete
  inspection of `jobs.tsv` and the manifests showed the submitted slot job
  and predicted-slot export still point at the correct absolute full1000
  RGB-D root; the failure was caused by
  `scripts/world_model/audit_rgbd_method_chain_contract.py` comparing path
  strings without normalizing equivalent relative and absolute paths. Fixed
  `_same_path` to compare `Path(...).expanduser().resolve(strict=False)`.
  This is an implementation repair to the audit tool only: it does not change
  the exact1000 data, thresholds, dependencies, slot/world-model/controller
  inputs, video requirements, or any evaluation gate. `py_compile` passed, and
  the same audit now passes `208/208` with timestamp
  `2026-06-04T22:29:22+08:00`. This remains contract/readiness evidence only,
  not slot-quality, world-model, controller, method, or mp4 evidence.
- 2026-06-04 22:31 live slot-gate monitor:
  re-read `AGENTS.md`, active TODO, and Slurm hygiene notes, then checked
  `squeue`, `scontrol`, main stdout/stderr, main slot artifacts, and all
  active one-H200 fallback panes/logs. Main exact1000 RGB-D slot training
  `105236` is still running on `server08`, `RunTime` about
  `03:19:46/16:00:00`, `ExcNodeList=(null)`, and stderr is empty. The loaded
  dataset is still exact1000 RGB-D with `301000` samples and about `2352`
  train batches per epoch. All four main seeds `1200/1201/1202/1203` are now
  at epoch-0 batch `1850` (`78.7%` of epoch 0). Artifact search under
  `experiments/world_model_task_rebinding/rgbd_slot_extractor/ensemble_4gpu/job105236`
  still finds no `model.pt`, `best_model.pt`, `metrics.json`,
  `history.csv`, or `inspection.json`, so strict inspection `105237` remains
  correctly pending on `afterok:105236`.

  The downstream chain is still strict and unchanged:
  `105238 afterok:105237`, `105239 afterok:105238`,
  `105553 afterok:105239`, and controller-video branches after world-model
  eval. No broad bad-node exclusion is present on the main/downstream path.
  One-H200 fallbacks remain useful risk reduction only and are still
  incomplete: visible panes show `105385` batch `650`, `105535` batch `500`,
  `105743` batch `300`, `105867` batch `150`, and `106000/106001` at batch
  `0`; `106147` and `106237` have emitted train-start but not visible
  dataset-loaded events, and ordinary fallback `106245` has passed its CUDA
  canary and emitted train-start but has no dataset-loaded line yet. Artifact
  search over all active one-H200 fallback dirs still finds no
  checkpoints/metrics/inspection/status that could be assembled into evidence.
  There is still no RGB-D method evidence and no mp4 demo.
- 2026-06-04 22:34 downstream readiness and slot-progress check:
  while waiting for the strict slot gate, parsed the active full1000 chain
  `jobs.tsv` and ran a non-executing syntax preflight over all unique
  submitted scripts. All `13` scripts exist and `bash -n` passed:
  slot training, slot inspection, predicted-slot export/inspection,
  sensitivity, RGB-D-derived world-model train/inspect/eval,
  controller-video, controller inspection, video artifact inspection,
  predicted-slot visual review, and final RGB-D method evidence review.
  This preflight addresses scheduling/implementation risk only; it does not
  train, render, export slots, evaluate controllers, change thresholds, or
  create method evidence.

  Final live check at `2026-06-04T22:34:04+08:00`: main `105236` is still
  running at about `03:22:26/16:00:00`, stderr remains empty, and strict
  inspection `105237` is still dependency-pending. Latest parsed slot
  progress is seed `1202` at epoch-0 batch `1900`, with seeds
  `1200/1201/1203` at batch `1850`; no epoch-complete event,
  checkpoint/metrics/history, strict inspection, RGB-D-derived world-model
  output, controller output, or mp4 exists. The next action remains concrete:
  when `105236` exits, inspect `105237` logs/artifacts immediately; if it
  fails, classify the failure from logs/artifacts and repair the RGB-D path
  without weakening exact1000 data, thresholds, RGB-D-derived inputs, or video
  evidence requirements.
- 2026-06-04 22:39 user queue-guidance recalibration and no-exclusion
  follow-up:
  after the user reiterated that small jobs are normally expected to wait
  about `10-30` minutes and that old bad-node history should not drive
  scheduling, rechecked the live Slurm state instead of relying on stale
  node explanations. The active method path still does not carry a standing
  bad-node list: `105236` is running on `server08` with
  `ExcNodeList=(null)`, and pending downstream jobs `105238`, `105553`, and
  controller-video jobs also have `ExcNodeList=(null)`. Some older fallback
  allocations still have job-local `server13` exclusions from same-day
  no-exclusion CUDA visibility failures, but the newer no-exclusion jobs
  `106000`, `106147`, `106237`, and `106245` show that future requests
  should test live nodes with the CUDA canary rather than pre-exclude from
  old history. Corrected a probe mistake: `gpu:h200:1` is not the actual
  GRES string here; using the live `gpu:NVIDIAH200:1` form, one-H200
  `cpu`/`cpu,gpu` `sbatch --test-only` probes for `03:00:00` through
  `1-00:00:00` all forecast about `2026-06-05T03:13:07` on `server07`,
  while `gpu,cpu` forecasts `2026-06-08T21:09:08` and should not be used.
  Submitted two additional bounded no-exclusion one-H200/one-day
  tmux-backed requests, `106360` (`h200_1gpu_pool13`, seed `2800`) and
  `106359` (`h200_1gpu_pool14`, seed `2900`). Both are pending for
  `Priority`, have `ExcNodeList=(null)`, and will immediately run the
  exact1000 RGB-D slot backup with the fixed `srun` CUDA canary if allocated.
  Real `squeue --start` currently reports `N/A` for these true pending jobs,
  so the test-only forecast is scheduling context only, not authoritative.
  Main `105236` still has no checkpoint/metrics/inspection artifacts; latest
  parsed progress is seeds `1200/1201/1202` at epoch-0 batch `1900` and seed
  `1203` at batch `1850` out of about `2352` train batches per epoch. There
  is still no RGB-D method evidence and no mp4 demo until strict slot
  inspection, predicted-slot export/inspection, RGB-D-derived world-model
  training/eval, controller metrics, and visual review pass.
- 2026-06-05 04:39 Cosmos3 full1000 RGB-D SFT and post-train video sanity:
  live allocation `107329` / `cosmos3_1h200_train` is running on `server62`
  and must be kept. Cosmos3 DCP conversion has succeeded, and SFT step
  `107329.16` has loaded the exact full1000 RGB-D SFT dataset (`918` train
  records, `82` val records in the prepared split), reached the training loop,
  saved checkpoint `iter_000000100`, and continued through iteration `150`
  with loss roughly in the `0.16-0.21` range. The post-train watcher
  `tmux cosmos3_post_sft_i2v_eval` is now waiting for `sft_completed`; when
  SFT finishes it will use the latest DCP checkpoint inside the same live
  allocation to generate a Cosmos3 I2V prediction for the `hole_move_stop`
  full1000 RGB-D diagnostic scene, compute reconstruction error against the
  reference preview, and write a comparison sheet. This follows the user's
  instruction that reconstruction error is a sanity check after video
  generation, not a blocking evaluation gate. If the video is nonblank and the
  reconstruction metrics are finite/physically interpretable, proceed to the
  next RGB-D-derived world-model/controller work without asking. This remains
  foundation world-model diagnostic evidence only; it is not final controller
  success or RGB-D task-rebinding method evidence.
- 2026-06-05 05:45 Cosmos3 SFT completed and downstream continuation:
  SFT step `107329.16` completed normally after `01:32:19`, saved final
  checkpoint `iter_000000500`, and wrote
  `sft_completed=2026-06-05T05:42:42+08:00` under
  `experiments/world_model_task_rebinding/cosmos3/sft_full1000_rgbd_after_i2v_recon_allocwatch_20260605_0305`.
  The same live allocation `107329` was reused for post-train I2V step
  `107329.17` and reconstruction step `107329.18`. The diagnostic video is
  `experiments/world_model_task_rebinding/cosmos3/i2v_eval_after_sft_full1000_rgbd_20260605_043900/inference/hole_move_stop_traj9_sft_full1000_i2v/vision.mp4`;
  reconstruction over `16` frames reported
  `mean_rmse_rgb01=0.3226327032678583`,
  `median_rmse_rgb01=0.27613693915821874`, and
  `mean_psnr_db=10.030909751148876`, improved over the earlier pretrained
  sanity (`mean_rmse_rgb01=0.34874007376007043`,
  `mean_psnr_db=9.2454`). Direct inspection of
  `reconstruction_comparison_sheet.png` found nonblank generated frames with
  table/gripper/peg/hole-block structure, but still substantial
  viewpoint/occlusion mismatch. This is foundation-WFM video sanity evidence
  only, not controller or method success. It is good enough to continue the
  RGB-D-derived downstream chain without asking, as requested. Next action:
  reuse the still-running H200 allocation for exact1000 RGB-D-derived slot /
  world-model / controller work; do not release the allocation after a short
  step.
- 2026-06-05 05:55 full1000 RGB-D downstream resumed in the live allocation:
  `job102294` predicted-slot export was rechecked as the current downstream
  input: exact `1000` RGB-D-derived slot H5 files and `301000` samples from
  the full1000 RGB-D root. Its old advisory geometry warnings
  (`hole_pos_rmse_m=0.031466636806726456`,
  `peg_head_hole_rmse_m=0.05495644360780716`) remain recorded but are not a
  hard blocker after the user correction. Existing fastload WM artifacts from
  `from_export_job102294/tmux_reuse_20260604_2317_fastload` only have two
  complete members (`h200_1gpu_pool13_seed1500` and
  `h200_1gpu_pool14_seed1600`); the old 8-member watcher is therefore stale
  and must not be treated as an active blocker. Added
  `scripts/slurm/run_rgbd_derived_wm_member_in_allocation.sh` and launched
  tmux `rgbd_wm_alloc107329_seed1700`, Slurm step `107329.19`, output
  `experiments/world_model_task_rebinding/rgbd_object_world_model/ensemble_4gpu/from_export_job102294/alloc107329_20260605_0550_seed1700`,
  log `logs/slurm/rgbd_wm_alloc107329_seed1700_20260605_0550.log`.
  The run is inside live allocation `107329` on `server62`, not an ordinary
  sbatch submission, and has already validated exact `1000` predicted-slot
  files, loaded `254000` sequence samples with
  `rgbd_predicted_slot_input_evidence=true`, and completed epoch `1` with
  `val_mean_hole_delta_rmse_m=0.018595176451771077`. It is configured for
  `MIN_TRAIN_SECONDS=10800`, `HISTORY=8`, `BATCH_SIZE=512`, `D_MODEL=512`,
  `N_LAYERS=6`, and `HORIZONS=1 5 10 20 40`. Added and launched watcher
  `rgbd_wm_alloc107329_assemble_controller`, log
  `logs/slurm/rgbd_wm_alloc107329_assemble_controller_20260605_0555.log`;
  when the seed-1700 member completes, it will assemble the two old complete
  members plus the new member into
  `from_export_job102294/alloc107329_20260605_0550_seed1700_assembled3`,
  require `rgbd_derived_training_evidence=true`, run learned-vs-CV evaluation,
  and then run at most five RGB-D controller/video branches inside allocation
  `107329`. This is continued RGB-D-derived downstream work, not final method
  evidence yet. Evidence note:
  `docs/world_model_task_rebinding/2026-06-05_full1000_rgbd_downstream_after_cosmos3.md`.
  Follow-up `2026-06-05T05:56+08:00`: `107329` remains running on `server62`
  and step `107329.19` remains active. The seed-1700 member has reached epoch
  `20` with latest visible `val_mean_hole_delta_rmse_m=0.01988275640299857`.
  Rechecked the training loop: it exits only after both `epoch >= EPOCHS` and
  elapsed time reaches `MIN_TRAIN_SECONDS`, so the fast first epochs will not
  release the allocation before the `10800` second floor. `bash -n` passed for
  the downstream wrappers, and project-venv `py_compile` passed for the
  assembly, inspection, evaluation, controller-run inspection, and video
  artifact inspection scripts. This is liveness and implementation readiness
  only; final `metrics.json`/`model.pt`, RGB-D-derived inspection,
  controller metrics, and inspected videos are still pending.
  Follow-up `2026-06-05T05:58+08:00`: the only Slurm job remains live
  allocation `107329` and step `107329.19` continues on `server62`. Cleaned
  stale old tmux sessions `h200_1gpu_pool3/8/9/10/11/12/13/14` after
  confirming they had no active Slurm allocations; retained only
  `cosmos3_h200_alloc`, `rgbd_wm_alloc107329_seed1700`, and
  `rgbd_wm_alloc107329_assemble_controller`. The seed-1700 member continued
  after cleanup and reached epoch `30` with
  `val_mean_hole_delta_rmse_m=0.019410628152972325`. This preserves the live
  allocation and reduces accidental stale-panel reuse; it is not method
  evidence.
  Follow-up `2026-06-05T06:00+08:00`: `107329.19` is still running and the
  seed-1700 member reached epoch `40` with
  `val_mean_hole_delta_rmse_m=0.019347243018496198`. The assemble target
  `alloc107329_20260605_0550_seed1700_assembled3` is still empty, so no stale
  outputs are being reused. The two intended old members
  `h200_1gpu_pool13_seed1500/member_0` and
  `h200_1gpu_pool14_seed1600/member_0` have final `metrics.json` and
  `model.pt`; the new member has `manifest.json` and `best_model.pt` but not
  final `metrics.json/model.pt` yet. The watcher remains correctly waiting
  and has not started controller work before the 3-hour member completes.
  Follow-up `2026-06-05T06:02+08:00`: seed1700 reached epoch `50` and then
  continued beyond the nominal epoch count because the `10800` second floor
  had not been reached. Latest visible epoch `55` has
  `val_mean_hole_delta_rmse_m=0.019317546188990876`. `sstat` for `107329.19`
  shows the task is still live with about `2.6 GB` RSS; no final
  `metrics.json/model.pt` exists yet. This confirms the allocation is being
  held for the intended long training instead of being released after a short
  run.
  Follow-up `2026-06-05T06:04+08:00`: `107329.19` remains active and seed1700
  reached epoch `60` with
  `val_mean_hole_delta_rmse_m=0.019375277269239554`. The new member still has
  only `best_model.pt`, `manifest.json`, and `member_manifest.txt`; final
  `metrics.json/model.pt` remain absent. The assembled WM target is empty and
  controller run group `rgbd_full1000_alloc107329_wm3_20260605_0555` does not
  exist yet, which is correct because controller work must wait for completed
  RGB-D-derived WM inspection.
  Follow-up `2026-06-05T06:06+08:00`: `107329` remains running on `server62`
  and step `107329.19` has run for about `12` minutes. Seed1700 reached epoch
  `70` with `val_mean_hole_delta_rmse_m=0.019300894198408285`; `sstat` RSS is
  about `2.6 GB`. Final `metrics.json/model.pt`, assembled ensemble, and
  controller run directory are still absent, so the watcher remains correctly
  waiting for the 3-hour completed member.
  Follow-up `2026-06-05T06:07+08:00`: `107329.19` has run for about `13`
  minutes and seed1700 reached epoch `75` with
  `val_mean_hole_delta_rmse_m=0.01953486391613537`. Final artifacts remain
  absent because the run is intentionally continuing toward the `10800`
  second floor. Watcher is still waiting; retained tmux sessions are only
  `cosmos3_h200_alloc`, `rgbd_wm_alloc107329_seed1700`, and
  `rgbd_wm_alloc107329_assemble_controller`.
  Follow-up `2026-06-05T06:08+08:00`: `107329.19` has run for about `14`
  minutes and seed1700 reached epoch `80` with
  `val_mean_hole_delta_rmse_m=0.01941447899795401`. Final
  `metrics.json/model.pt` remain absent, the assembled WM directory is empty,
  and controller run group is absent, as expected before the `10800` second
  floor.
  Follow-up `2026-06-05T06:09+08:00`: `107329` remains running and
  `107329.19` has run for about `15` minutes. Seed1700 reached epoch `85`
  with `val_mean_hole_delta_rmse_m=0.0193015849193661`; watcher remains in
  wait state and no final WM, assembled WM, or controller artifacts exist yet.
  Follow-up `2026-06-05T06:10+08:00`: `107329.19` has run for about `16`
  minutes and seed1700 reached epoch `90` with
  `val_mean_hole_delta_rmse_m=0.01934988992470442`. Final
  `metrics.json/model.pt` remain absent; assembled WM directory is empty and
  controller run group is absent. Still below the `10800` second floor.
  Follow-up `2026-06-05T06:12+08:00`: seed1700 reached epoch `100` with
  `val_mean_hole_delta_rmse_m=0.019525333678420954` while continuing toward
  the 3-hour floor. Checked downstream checkpoint defaults:
  learned-vs-CV eval, RGB-D controller video wrapper, and controller eval all
  default to `best_model.pt`, so the eventual 3-hour completion artifact will
  prove run duration while downstream evaluation still uses validation-best
  weights.
  Follow-up `2026-06-05T06:14+08:00`: `107329.19` has run for about `18`
  minutes and seed1700 reached epoch `110` with
  `val_mean_hole_delta_rmse_m=0.01929032061150795`. Final
  `metrics.json/model.pt` remain absent; assembled WM and controller
  artifacts remain absent. Resource usage remains normal.
  Follow-up `2026-06-08T15:12+08:00`: current WAM/RGB-D controller evidence
  says the DP failure is not simply "the policy cannot see the moved hole."
  The exact `hole_constant` reproduction with the stricter
  task-frame-projected/contact-preserve controller failed while preserving
  grasp (`success_at_end=false`, final head L2 `0.113145m`, selected WAM
  available for only `9/200` selected rollout-MPC steps). A matched
  `hole_constant` ablation with default direct-world bridge, default servo
  hold, and insert-probe residuals succeeded on the same seed:
  `success_once=true`, `success_at_end=true`, first success frame `147`, final
  grasp true, final head L2 `0.005674m`. Inspected video:
  `experiments/world_model_task_rebinding/rebinding_controller/wam_scorer_rollout_mpc_rgbd_holeconstant_defaultbridge_seed702000_selectedlog_no_video_20260608_server44/visual_review_envstate_default_view_1024_full301_30fps_20260608/state_replay.mp4`.
  Boundary: this is still a single-seed diagnostic because the WAM scorer is a
  non-method action-prior smoke checkpoint and Cosmos reports
  `closed_loop_refresh=false`. The active next step remains a causal
  Cosmos3/WAM interface that supplies target task-frame trajectory plus
  action-state scoring/generation for selected short chunks with live
  re-observation, not more hand-written gate tuning.
  Follow-up `2026-06-08T18:50+08:00`: reused tmux allocation `118609` on
  `server43` for
  `experiments/world_model_task_rebinding/rebinding_controller/wam_defaultbridge_posneg_ranker_guided_rgbd_holeconstant_seed702000_no_video_20260608`.
  It used RGB-D slots, the receding Cosmos task-state trajectory, the
  mixed WAM scorer, defaultbridge/insert-probe/local-axis candidates, and
  candidate ranker
  `experiments/world_model_task_rebinding/rebinding_controller/wam_candidate_ranker_defaultbridge_posneg_1000iter_20260608/checkpoints/final.pt`.
  Metrics: `success_once=true`, `success_at_end=true`, trigger step `90`,
  `wm_dp_prior_rollout_mpc_count=205`, `infeasible_count=5`, final metric
  grasp true and inserted true, final metric `peg_head_at_hole`
  `[-0.014226, -0.002979, -0.002977]`. Visual review artifact:
  `experiments/world_model_task_rebinding/rebinding_controller/wam_defaultbridge_posneg_ranker_guided_rgbd_holeconstant_seed702000_no_video_20260608/visual_review_envstate_default_view_1024_stride8_30fps_20260608/contact_sheet.png`;
  video:
  `.../state_replay.mp4`. Manual review: peg is held, approaches the moved
  box, visibly inserts around frames `192..200`, and stays in a hold-success
  posture through frame `300`; no gross peg fly-up/lost-grasp failure.
  Candidate export `wam_candidate_ranking_dataset_h16_a8.h5` has
  `sample_count=331`, `final_inserted_count=73`, selected chunks `11`, and
  selected final-inserted chunks `4` from `bridge_variant` or
  `insert_probe_residual`. This directly answers the DP/Cosmos question:
  the state-DP is not simply missing the moved target hole; target visibility
  alone and a small static action-prior library failed. The needed component
  is a causal Cosmos3/WAM trajectory plus DP-preserving short action
  generation/ranking that physically inserts and retains after re-observation.
  Boundary: this remains non-method diagnostic evidence because the ranker was
  trained from planner-imagined non-method candidate logs, RGB-D/control slot
  final inserted is still false, and Cosmos reports
  `world_model_closed_loop_refresh=false`.
