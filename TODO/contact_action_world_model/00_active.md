# Contact-Action World Model TODO

## 2026-06-26 Status

This TODO is historical diagnostic context after the OpenPI full-episode
protocol correction. Do not execute these items as the active method. The
active OpenPI TODO is `TODO/openpi_pi05_contact_action/00_active.md`.

## Current Boundary

- [x] 2026-06-23 reset written: direction was a contact/insertion
      action generator or WAM-conditioned executor, not scorer-only selection.
- [x] Frozen DP is demoted to baseline/static prior/fallback. It may be used
      for labels and comparison, but it is not required to remain the final
      base policy if Octo, pi0/OpenPI, OpenVLA, residual RL, or a local
      contact-action model is stronger.
- [x] Login-node policy refreshed in `AGENTS.md`: no project compute, rollout,
      replay, training, rendering, preflight, syntax/debug checks, or smoke
      tests on the login node.
- [x] Latest h8192 strict-gate live panel audited:
      `completed_samples=4`, `final_success_count=0`, full `301/300`
      contract OK, no process failure, method evidence not allowed. Visual
      contact sheet shows near-hole behavior without stable insertion/contact
      completion.
- [x] Current review/blocker note written:
      `docs/world_model_task_rebinding/contact_action_world_model/2026-06-23_current_review_and_blockers.md`.

## Immediate Cleanup

- [x] Preserve active artifacts:
      accepted 733 H5 source, approved RGB/SFT data, full `301/300` condition
      exports, base DP checkpoints, current SFT/WAM checkpoints, source suffix
      banks, DP96 live labels, current strict scorer baselines, and the most
      important evidence notes/contact sheets.
- [x] Move superseded smoke/canary/failed exploratory/duplicate shard/stale
      scorer artifacts under
      `experiments/_archive_20260623_contact_action_reset/` with a manifest.
      Do not move any directory currently being written by allocation `146658`.
- [x] Update the archive manifest after each move batch so paths remain
      recoverable.
      Current manifest: `experiments/_archive_20260623_contact_action_reset/MOVED_DIRS.txt`
      records `659` archived Cosmos3 canary/smoke/debug/scorer/retrieval
      directories after the second cleanup batch. The active Cosmos3 top-level
      tree now has `119` directories.

## Data Construction

- [x] Audit the old 2026-06-15 contact-executor dataset before reusing it.
      Result: the joined manifest has useful-looking direct-positive labels
      (`512` rows, `185` `future_inserted_within_chunk`, `319`
      `future_dp_continuable_within_chunk`), but it is a historical
      `cosmos_predicted_action_sidecar` branch and is now partially断链:
      executor samples and contact labels still exist, while the referenced
      DP-prior jsonl/chunks under
      `executor_dp_prior_smoke_20260615_pred_task_path_dp_prior_train512_diverse`
      are missing from the active tree. Do not train from this root directly.
      Use it only as an audit pointer for rebuilding a clean active manifest.
- [ ] Build a contact-action training manifest from the 733 accepted source
      trajectories. Required labels: insertion suffix start/end, peg-hole
      relative state, grasp/contact proxies, action chunk, and whether the
      suffix reaches inserted/contact-stable state.
- [ ] Rebuild a clean direct-contact executor manifest in the active branch.
      Minimum inputs: source insertion suffix positives from the 733 accepted
      H5 set, contact-progress labels, Policy-DROID/suffix saved-snapshot
      direct failures, and `candidate + DP96` handoff labels kept as secondary
      labels. The primary positive target must be direct post-chunk inserted,
      contact-stable, or insertion-axis-continuable behavior, not scalar
      scorer preference.
      2026-06-24 progress: added
      `scripts/world_model/build_direct_contact_executor_manifest.py` and ran
      it inside held allocation `148732` on `server24`. Output:
      `experiments/world_model_task_rebinding/cosmos3/direct_contact_executor_manifest_h24_sourcepos_livehard_20260624_alloc148732`.
      It built a horizon-24 manifest with `2906` rows: `2905`
      `source_direct_positive` rows and `1` Policy-DROID live hard negative.
      It is ready as a direct-positive action-generation manifest, but hard
      negatives remain incomplete because the `192` causal-suffix replay rows
      reference synthetic `causal_suffix_diffusion_o*` actions that were not
      persisted in the original `candidate_action_bank.npz`; the builder now
      records this as `replay_skip_causal_suffix_action_not_persisted`.
- [x] Fix the replay-to-manifest plumbing so future generated candidates keep
      their action chunks. 2026-06-24 update:
      `scripts/world_model/replay_cosmos3_live_action_bank_from_snapshots.py`
      now writes generated/synthetic replayed action chunks under
      `persisted_action_chunks/` and records `persisted_action_chunk_json` in
      each label. `scripts/world_model/build_direct_contact_executor_manifest.py`
      now loads that persisted chunk before falling back to
      `candidate_action_bank.npz`. This fixes future replay conversion, but it
      does not recover old labels that were already written without generated
      actions.
- [ ] Rerun the selected causal-suffix/live-candidate replay inside the held
      Slurm allocation with action persistence enabled, then rebuild the
      direct-contact manifest so generated DP96-positive and hard-negative
      live candidates become usable action rows rather than label-only
      evidence. Do not run this on the login node.
      2026-06-24 progress: reran the selected causal-suffix value replay inside
      allocation `148732` with stamp `20260624_persistfix1`. Output:
      `live_snapshot_replay_selected_causal_suffix_value_head_panel0134_margin0_exec8_dp96_20260624_persistfix1_alloc148732`.
      The replay produced `16/16` valid records, `0` failures, `11`
      causal-suffix records, and `11` persisted generated action-chunk JSON
      files. Execution metrics remain weak: direct success `0/16`, direct
      gate-ok `0/16`, DP96 success `8/16`, DP96 continuable `10/16`, improved
      `abs_y+abs_z` `4/16`, worsened `12/16`. Rebuilt manifest:
      `direct_contact_executor_manifest_h24_sourcepos_persistedlivehard_20260624_alloc148732`.
      It has `2922` rows: `2905` source positives, `16` live replay hard
      negatives, `1` Policy-DROID hard negative. The summary confirms
      `replay_action_loaded_from_persisted_chunk_json=11`. This fixes the
      action-loss blocker, but does not create direct live positives.
- [ ] Train a direct-contact executor with an auxiliary live-hard-negative
      risk head from the persisted-live-hard manifest. The action diffusion
      target remains primary direct-positive source chunks; the risk head sees
      source positives versus live hard negatives as value/contrastive
      supervision. 2026-06-24 launch notes: an initial run
      `direct_contact_executor_diffusion_h24_sourcepos_persistedlivehard_riskhead_1gpu1h_20260624_alloc148732`
      was interrupted after about one minute because the risk validation split
      did not reliably include negative examples. The trainer was patched to
      use label-stratified risk validation, then relaunched in the same held
      allocation as
      `direct_contact_executor_diffusion_h24_sourcepos_persistedlivehard_riskhead_1gpu1h_fix1stratval_20260624_alloc148732`.
      Early metadata: risk train `2469` positive / `14` negative, risk val
      `436` positive / `3` negative. This run must reach the one-GPU-hour
      floor before it can be interpreted, and even then it is training
      evidence only until sampled chunks pass saved-snapshot replay.
- [ ] Merge live snapshot labels from source-suffix and panel0245/0204 DP96
      replays. Required labels: `candidate + DP96` final success,
      continuability, contact stability, grasp preservation, and insertion-axis
      progress.
- [ ] Add hard negatives from current failures where y/z alignment or `C_pi`
      was positive but DP96 failed.
- [ ] Audit train/val split by source/scenario/phase before training so the
      model is not only memorizing one source suffix.

## Model Work

- [ ] First local model: train a short-horizon contact-suffix diffusion/action
      generator from current/history RGB or RGB-derived state, action history,
      peg-hole task-frame state, and contact phase. Minimum valid training:
      one GPU-hour on the 733 data or documented full-data split.
      2026-06-23 start: launched a first 1GPU/1h source-suffix contact-action
      generator run from the full 733-source insertion suffix bank
      (`4711` suffix rows, `4001/710` source-UUID train/val split) inside
      held allocation `146658`, step `149`, output
      `experiments/world_model_task_rebinding/cosmos3/contact_action_suffix_generator_full733_1gpu1h_20260623_163847_alloc146658`.
      This run is a baseline action-generator experiment, not live method
      evidence. The trainer was patched afterward to save
      `checkpoint_best_eval.pt` for follow-up runs because early metrics showed
      fast train overfit after a good early val point.
      2026-06-23 result: the run finished inside allocation `146658` with
      `formal_one_gpu_hour_floor_met=true`, `elapsed_seconds=3661.45`,
      `steps=567001`, and `stop_reason=min_wall_and_min_steps`, but it failed
      the saved-snapshot replay gate:
      `ready_for_saved_snapshot_replay_gate=false`. Final validation MSE
      (`0.0162008`) was worse than the mean-action baseline (`0.0155657`)
      while train MSE was `1.3e-5`, so this first deterministic source-suffix
      MLP is a formal negative diagnostic, not a usable live candidate.
      2026-06-23 repair launch: started
      `scripts/world_model/train_causal_contact_action_suffix_diffusion.py`
      via
      `scripts/slurm/run_causal_contact_action_suffix_diffusion_train_in_allocation.sh`
      inside held allocation `146658`, step `151`, output
      `experiments/world_model_task_rebinding/cosmos3/causal_contact_action_suffix_diffusion_full733_1gpu1h_20260623_190108_alloc146658`.
      This run trains a conditional diffusion generator on the same direct
      inserted source suffix bank, but removes non-causal `scenario_onehot`
      and future `first_insert_frame` features. It conditions on causal
      task-frame state plus a requested insertion-offset control token. The
      wrapper enforces `min_wall_seconds=3660`; early metrics are liveness
      only until the one-GPU-hour floor and `training_summary.json` exist.
      2026-06-23 result: completed with
      `formal_one_gpu_hour_floor_met=true`, `elapsed_seconds=3660.22`,
      `steps=638567`, and `ready_for_saved_snapshot_replay_gate=true`.
      Best source-training metrics were `eval_denoise_mse=0.285` versus
      zero-noise baseline `0.999`, and `eval_x0_action_mse_mid_t=0.000283`
      versus mean-action baseline `0.01557`. This is the first reset training
      run to pass a source-training gate, but it is not method evidence until
      sampled actions pass saved live-snapshot replay and full-panel visual
      gates.
- [x] Audit whether current local Cosmos/Cosmos-Policy tooling can support an
      action/value/video head on the 733 RGB trajectories without breaking the
      `301/300` contract. This is the preferred same-backbone replacement for
      scorer-only selection if dependency and checkpoint availability are
      feasible.
      Initial local audit found usable `Cosmos3-Nano-Policy-DROID` and DCP
      assets plus a live-prefix action extraction wrapper, but no ready
      contact-action/value/video post-training path. Defaults also need
      retargeting away from older 2026-06-12 roots before use.
- [ ] Adapt the Cosmos Policy-DROID live-prefix/action sidecar entrypoint to
      the active 733 clean-dense roots, then design the missing value/progress
      head training path using contact-action labels and DP96 outcomes.
      2026-06-24 progress: added
      `scripts/slurm/run_cosmos_policy_droid_active_prefix_input_gate_in_allocation.sh`
      and fixed the `SOURCE_H5` live-prefix builder path after `_prefix_payload`
      signature drift. Inside held allocation `148680` on `server13`, the
      input-only gate succeeded for
      `hole_late_move_stop_seed3280649_idx2518.fix3_traj_0__target_motion_observed_f106`:
      it wrote a `107`-frame prefix-only RGB video for frame `0..106` and
      produced `cosmos_live_prefix/live_prefix_input_manifest.json` under
      `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/cosmos_policy_droid_active_prefix_input_gate_iter_000001500_20260624_173015_alloc148680`.
      This proves the active 733 clean-dense data can form a strict causal
      Policy-DROID live-prefix input. It is not controller evidence.
      A follow-up `RUN_INFERENCE=true` probe under
      `.../cosmos_policy_droid_active_prefix_input_gate_iter_000001500_20260624_173040_alloc148680`
      was canceled after several minutes with no action output; after
      cancellation the Cosmos log showed it had only reached `Loaded 1 samples`.
      Treat this as an inference startup/run-to-completion integration blocker,
      not an action-quality result. Evidence note:
      `docs/world_model_task_rebinding/contact_action_world_model/2026-06-24_policy_droid_pivot_gate.md`.
      2026-06-24 update: reran the same sample with
      `INFERENCE_NUM_STEPS=10` as
      `.../cosmos_policy_droid_active_prefix_input_gate_iter_000001500_20260624_1735_10step_alloc148680`.
      This completed: Cosmos wrote `sample_outputs.json`, predicted
      `vision.mp4`, and extracted
      `cosmos_live_prefix/live_prefix_action_chunk.json`. The extracted
      chunk covers steps `106..114`, has `8` robot-action rows, all finite,
      and denormalized stats min about `-1.016`, max about `0.372`, mean abs
      about `0.211`. This proves the Policy-DROID action-generation path is
      operational on the active data, but it is still only a diagnostic until
      replay/evaluation.
      Added `scripts/world_model/replay_policy_droid_action_chunk_from_source.py`
      to replay the chunk from the matching source H5 prefix state. First run
      exposed a wrapper bug (`save_step_records` missing) and was fixed.
      The next run failed because torch CUDA initialization failed inside the
      current held allocation for the ManiSkill/DP replay venv, even though
      `nvidia-smi` saw the H200 and `CUDA_VISIBLE_DEVICES=0`. Do not treat this
      as an action-quality result; it is a compute-environment replay blocker.
      2026-06-24 resource update: repeated CUDA checks in allocation `148680`
      confirmed `torch.cuda.init()` fails on `server13` while `nvidia-smi`
      sees the H200. Tried CUDA visible-device variants, GPU UUID,
      `--gpus=1`, `--gpu-bind=none`, clearing `LD_LIBRARY_PATH`, and
      `CUDA_MODULE_LOADING=EAGER`; none repaired PyTorch CUDA. Released
      `148680` rather than keeping an unusable idle GPU. A tmux-held
      interactive request on the broader `gpu` partition is pending as job
      `148732`, session
      `policy_replay_cuda_repair_gpu_request_20260624_1801`, excluding
      `server13`.
      2026-06-24 replay result: the `gpu` partition request started as
      allocation `148732` on `server24`, where `.venv` torch CUDA was valid.
      Replayed the extracted Policy-DROID 8-step chunk from the matching
      source H5 prefix state into
      `experiments/world_model_task_rebinding/cosmos3/policy_droid_action_replay_source_prefix_server24_20260624_1810_alloc148732`.
      Direct post-chunk result was negative: `after_success=false`,
      `after_inserted_live_pose=false`, `after_contact_stable_proxy=false`,
      `after_continuability_gate.ok=false`, `after_grasped=true`, and
      `delta_abs_yz_sum=+0.0128`. DP96 label after the chunk was positive:
      `continuable=true`, `success=true`, DP needed `48` steps, and final
      peg-head-at-hole was approximately `[-0.0149, 0.0030, -0.0010]`.
      Interpretation: this single Policy-DROID chunk is not a direct insertion
      executor, but it is a valid handoff/action-prior diagnostic. Next replay
      should target saved dynamic live failure snapshots.
      2026-06-24 saved-snapshot diagnostic: added
      `scripts/world_model/replay_policy_droid_action_chunk_from_snapshot.py`.
      A same-snapshot Policy-DROID inference attempt on allocation `148732`
      wrote a valid live-prefix input but was manually interrupted during
      slow Cosmos model startup after reaching `Loaded 1 samples`; no action
      chunk was produced, so this is a run-control issue, not action-quality
      evidence. Then replayed the existing source-prefix Policy-DROID chunk
      from saved dynamic live snapshot
      `sample_00_hole_late_move_stop/iter_00_prefix_f106` as an explicitly
      marked mismatched-prefix diagnostic:
      `experiments/world_model_task_rebinding/cosmos3/policy_droid_action_replay_live_snapshot_sourceprefix_mismatch_sample00_iter00_f106_20260624_alloc148732`.
      Direct post-chunk result remained negative:
      `after_success=false`, `after_inserted_live_pose=false`,
      `after_contact_stable_proxy=false`, `after_continuability_gate.ok=false`,
      and `delta_abs_yz_sum=+0.01257`, while `after_grasped=true`.
      DP96 after the chunk was positive:
      `continuable=true`, `success=true`, with `67` executed DP steps and
      final peg-head-at-hole about `[-0.0058, 0.0030, 0.0029]`.
      Interpretation: the live snapshot is physically recoverable and the
      chunk preserves grasp, but Policy-DROID has not yet shown direct
      insertion/contact-completion behavior. The next valid probe is
      same-prefix Policy-DROID inference plus saved-snapshot replay to
      completion, not a live panel and not more scorer tuning.
      2026-06-24 same-prefix replay result: reran the saved live prefix to
      completion on allocation `148732` under
      `.../policy_droid_live_snapshot_sample00_iter00_f106_10step_rerun_complete_alloc148732/cosmos_live_prefix`.
      Cosmos wrote `sample_outputs.json` and extracted
      `live_prefix_action_chunk.json`; the chunk covers steps `106..114`,
      has `8` finite robot-action rows, denormalized min about `-1.032`,
      max about `0.437`, and mean abs about `0.234`. Replayed that same-prefix
      chunk from the saved dynamic snapshot into
      `experiments/world_model_task_rebinding/cosmos3/policy_droid_action_replay_live_snapshot_sameprefix_sample00_iter00_f106_20260624_alloc148732`.
      Direct post-chunk result is still negative:
      `after_success=false`, `after_inserted_live_pose=false`,
      `after_contact_stable_proxy=false`, `after_continuability_gate.ok=false`;
      grasp is preserved (`after_grasped=true`) and near-term y/z error
      worsens only slightly (`delta_abs_yz_sum=+0.00316`). DP96 after the
      chunk succeeds: `continuable=true`, `success=true`, `63` executed DP
      steps, final peg-head-at-hole about `[-0.0021, 0.0025, -0.0018]`.
      This improves over the mismatched-prefix diagnostic but remains a
      handoff/action-prior result, not direct insertion execution.
- [ ] Train the clean direct-contact horizon-24 diffusion executor from the
      new manifest for at least one GPU-hour before interpreting it.
      2026-06-24 launch: added
      `scripts/world_model/train_direct_contact_executor_diffusion.py` and
      started it inside allocation `148732`:
      `experiments/world_model_task_rebinding/cosmos3/direct_contact_executor_diffusion_h24_sourcepos_1gpu1h_20260624_alloc148732`.
      The trainer uses only primary direct-positive source rows as action
      imitation targets (`2905` rows; `2469/436` source split) and excludes
      scenario/source/future/end labels from input features. It is guarded
      against non-Slurm execution and uses `min_wall_seconds=3660`; until
      `training_summary.json` reports the one-GPU-hour floor, progress metrics
      are liveness only.
      First launch stopped early at `500000` steps after `1312s` with
      `formal_one_gpu_hour_floor_met=false` because `--max-steps` was too low;
      this root is invalid as training evidence. Relaunched immediately in
      the same held allocation with `--max-steps 5000000`:
      `experiments/world_model_task_rebinding/cosmos3/direct_contact_executor_diffusion_h24_sourcepos_1gpu1h_fix1maxsteps_20260624_alloc148732`.
      The fix1 run is the active training run to judge after the one-GPU-hour
      summary exists.
      2026-06-24 result: fix1 completed with
      `formal_one_gpu_hour_floor_met=true`, `elapsed_seconds=3660.05`,
      `steps=1396013`, `stop_reason=min_wall_and_min_steps`, and
      `ready_for_saved_snapshot_replay_gate=true`. Best validation
      `x0_action_mse_mid_t` was `0.02960`, while final validation worsened to
      `0.04845`, so use `checkpoint_best_eval.pt` for replay and treat the
      final checkpoint as overfit diagnostic only. This is valid training
      evidence, not method evidence until saved dynamic snapshot replay and
      visual/final-state review.
      2026-06-24 saved-snapshot replay: added
      `scripts/world_model/sample_direct_contact_executor_chunk.py`. A naive
      DDIM sampler with the live query marked `grasped=false` produced
      unusable `1e4`-scale actions, exposing an OOD/sampler issue. Repaired
      sampling with `x0_mid`, forced grasped query, and action clipping; sampled
      actions were finite and bounded. Replayed o24 and o8 chunks from
      `sample_00_hole_late_move_stop/iter_00_prefix_f106`. Both preserved
      grasp but failed direct insertion/gate/contact and failed DP96 handoff.
      o24 worsened `abs_y+abs_z` by `+0.1305`; o8 worsened it by `+0.1225`.
      This is a formal negative saved-snapshot diagnostic for the source-only
      direct-positive executor: it pushes the peg laterally far off the hole in
      the dynamic snapshot. Do not promote this checkpoint to live panel.
- [ ] Add a consequence/value head trained on real rollout labels, not
      instantaneous geometry gate: final success, DP96 success/continuability,
      contact stability, grasp, and insertion-axis progress.
- [ ] Train a live-outcome-conditioned contact-action generator from saved
      live candidate outcomes, not source suffixes alone.
      2026-06-23 start: launched
      `scripts/world_model/train_live_outcome_action_diffusion.py` via
      `scripts/slurm/run_live_outcome_action_diffusion_train_in_allocation.sh`
      inside held allocation `146658`, step `150`, on `server56`. Output:
      `experiments/world_model_task_rebinding/cosmos3/live_outcome_action_diffusion_full_live_union_1gpu1h_20260623_175110_alloc146658`.
      It uses the latest live outcome union with `9349` candidate outcome
      rows, `127` live-state groups, `146` positive rows, and `138` hard
      positive rows. The model conditions on causal base features/contact
      context and trains diffusion residual generation plus consequence/value
      heads. This is training evidence only if it reaches the one-GPU-hour
      floor, and it is not live method evidence without saved-snapshot replay
      and full-panel visual/final-state gates.
      Read-only data audit after launch found an important limitation:
      the outcome set has `0` direct final successes and `0` direct inserted
      positives; the `138` hard positives are `candidate + DP96` handoff
      successes. The `127` base live states come from only `4` source UUIDs.
      This run can test whether live outcome labels improve action selection
      or residual generation, but it is not a complete direct insertion-action
      dataset.
      2026-06-23 result: the run completed with
      `formal_one_gpu_hour_floor_met=true`, `elapsed_seconds=3660.23`,
      `steps=224776`, and `stop_reason=min_wall_and_min_steps`, but failed the
      replay gate: `ready_for_saved_snapshot_replay_gate=false`. Final
      held-out selection tied DP prior (`0.32` selected versus `0.32` DP;
      oracle over existing candidates `0.44`), and best value-MSE checkpoint
      was at step `1`. Treat this as a formal negative/limited diagnostic.
      Do not promote this checkpoint to saved-snapshot replay or live panels
      except as an explicitly marked negative diagnostic.
- [ ] If the local contact-action model cannot produce DP96-positive
      candidates on saved live failure snapshots, start an alternative base
      policy integration audit for Octo, pi0/OpenPI, and OpenVLA. Downloads or
      `git clone` may happen on the login node within CPU/memory limits; all
      adaptation/preflight/training must happen on compute nodes.
      2026-06-23 audit: local OpenPI exists at
      `/public/home/yanhongru/ICLR2027/openpi`, but no ManiSkill/PegInsertion
      LeRobot converter, normalization config, or eval wrapper exists in this
      repo yet. No local Octo/OpenVLA checkout or checkpoint was found by
      read-only inspection. Cosmos Policy-DROID remains the fastest
      same-family pivot because local checkpoints and live-prefix action
      extraction already exist, although post-training/value-head support is
      still missing.
- [ ] Consider residual RL only after the imitation/contact-action target is
      structurally valid and failure remains physical, not data/schema related.

## Evaluation

- [ ] Offline gate: on held-out live snapshot groups, selected generated
      candidate must beat DP prior on real DP96 success/continuability without
      worsening weighted task error or contact progress.
      2026-06-23 prep: `replay_cosmos3_live_action_bank_from_snapshots.py`
      now accepts `--suffix-generator-checkpoint` so the source-suffix action
      generator can be tested on saved live failure states before any live
      panel. This code path still needs compute-node execution after the
      1GPU-hour training run produces a usable checkpoint. Boundary issue:
      the first trainer/replay path uses `scenario_onehot` from source/sample
      metadata, so generated-suffix replay is diagnostic only. A method-valid
      generator must replace this with causal observed-history motion/contact
      features before live evidence can be claimed.
      2026-06-23 update: skip replay for the first MLP final checkpoint unless
      it is explicitly needed as a negative diagnostic, because the completed
      training summary already marks the checkpoint not ready for the replay
      gate.
      2026-06-23 live-outcome diffusion caveat: the current replay tool only
      supports saved banks, source-suffix retrieval, and the first suffix MLP.
      It does not yet support sampling actions from
      `train_live_outcome_action_diffusion.py`; if the training summary marks
      the checkpoint replay-ready, a diffusion extraction/replay path still
      has to be added and run inside the held compute allocation before any
      live panel.
      2026-06-23 causal suffix diffusion update: the source-training gate is
      now true for
      `causal_contact_action_suffix_diffusion_full733_1gpu1h_20260623_190108_alloc146658`.
      Next action is to add sampled diffusion candidates to
      `replay_cosmos3_live_action_bank_from_snapshots.py` and replay them from
      saved live failure snapshots inside allocation `146658`; do not jump
      directly to a live panel.
      2026-06-23 replay launch: added sampled causal-suffix-diffusion replay
      support to `replay_cosmos3_live_action_bank_from_snapshots.py` and
      started saved-snapshot replay in allocation `146658`, step `152`,
      output
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_causal_suffix_diffusion_panel0134_offsets64_48_32_24_16_8_s2_exec8_dp96_20260623_200828_alloc146658`.
      This replays generated candidates only on up to four failed panel0134
      samples, four iter dirs each, offsets `64,48,32,24,16,8`, two samples
      per offset, execute `8`, then DP96 continuability labeling. It is
      diagnostic replay, not live evidence.
      First launch failed before labels because the checkpoint loader rebuilt
      the model with `dropout=0.0`, changing Sequential state-dict indices.
      Fixed loader to use the checkpoint dropout value and relaunched as
      `..._fix1_20260623_201146_alloc146658`. Fix1 replay completed with
      `192/192` valid records and no process failures. Direct generated-chunk
      success was `0`, direct post-chunk gate-ok was `0`, but `candidate+DP96`
      success was `55/192` and DP-continuable was `59/192`; all `16/16` live
      snapshot groups had at least one generated candidate with DP96 success.
      This proves generated candidate coverage in saved snapshots, but not
      selection or live controller success.
      2026-06-23 DP baseline/control-label repair: replayed the same panel0134
      saved snapshots with `dp_prior` only inside allocation `146658`, step
      `155`, output
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_dp_prior_panel0134_exec8_dp96_20260623_204147_alloc146658`.
      The baseline replay produced `16/16` valid records, no process failures,
      direct success `0`, direct gate-ok `0`, and DP96 success/continuability
      `8/16`. Then converted DP + causal generated labels with a shared
      snapshot namespace to
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_causal_suffix_diffusion_panel0134_exec8_dp96_20260623_204543_alloc146658`.
      Conversion produced `208` valid rows, `16` base groups,
      `groups_with_dp_prior=16`, and
      `groups_with_causal_suffix_diffusion_dp96_success=16`.
      Merged this with the older live-outcome union as
      `experiments/world_model_task_rebinding/cosmos3/contact_value_training_union_plus_panel0134_causal_suffix_20260623_204657_alloc146658`;
      merged summary: `143` groups, `9557` joined outcome rows,
      `0` missing base rows, all `143` groups have DP prior, `16` groups have
      causal suffix diffusion candidates, and all `16` causal groups have at
      least one DP96-success generated candidate.
      2026-06-23 value-head launch: started formal 1GPU/1h consequence/value
      training inside allocation `146658`, step `158`, output
      `experiments/world_model_task_rebinding/cosmos3/contact_value_head_union_plus_panel0134_causal_suffix_1gpu1h_20260623_204725_alloc146658`.
      This is offline value/consequence training over real replay labels, not
      live method evidence. Do not jump to a live panel until selected
      generated candidates pass saved-snapshot replay and the run reaches the
      one-GPU-hour floor.
      2026-06-24 result: the run completed with `elapsed_seconds=3660.10`,
      `steps=141934`, `visible_cuda_device_count=1`, and
      `stop_reason=min_wall_and_min_steps`. The raw time evidence satisfies the
      current 1GPU/1h rule, but the trainer summary was written before the
      outcome-scorer script was repaired from the older `10800`-second formal
      floor to the current `3600`-second floor; the stale summary fields
      `formal_training_floor_met=false` and
      `ready_for_formal_live_eval=false` should be interpreted as an
      implementation-gate bug for this completed run, not as evidence that it
      ran too short. The trainer has now been patched to use `3600`.
      Offline result is weak: best-gate eval improved DP96 handoff from
      `7/29` to `8/29` (`+0.0345`), improved weighted task error by only
      `0.00324`, improved progress delta by `0.0103`, and matched the handoff
      oracle only `0.1724`. Final checkpoint regressed to DP parity on
      handoff success (`7/29`) and slightly worse weighted error. This is not
      live-controller evidence.
      Resource status: the held allocation `146658` has ended; do not attempt
      to reuse it and do not use `sbatch`.
      2026-06-24 follow-up: opened a new tmux-held interactive allocation
      request, session
      `contact_value_selected_replay_1gpu_request_20260624_170015`, Slurm job
      `148676`, requesting `1` GPU for `1` day. It stayed pending and was
      later canceled after `148680` started. The active allocation is now
      tmux session `contact_value_margin_gpux_1gpu_request_20260624_170413`,
      Slurm job `148680`, on `server13`.
      Margin audits in `148680` showed that the best-gate checkpoint produced
      only a weak full merged eval improvement: selected handoff `8/29` versus
      DP `7/29`, selected-minus-DP weighted error `-0.00324`,
      contact-progress delta `+0.01028`, but `7` harmful switches among
      `20` non-DP switches. The panel0134-only margin audit looked stronger on
      a tiny held-out split (`3/3` selected versus `0/3` DP), but that split
      has only three eval groups and cannot justify live-panel claims.
      Current boundary: value/margin selection is not strong enough to run a
      live panel. The only aligned next diagnostic is a selected
      saved-snapshot replay that actually uses the value head to choose one
      causal generated candidate per live snapshot and then measures real
      replay plus DP96 labels.

- [ ] Add and run selected causal-suffix saved-snapshot replay. Required
      behavior: construct causal suffix diffusion candidates from the same
      saved live snapshots, score them with the one-GPU-hour value head using
      the same feature contract as converted outcome rows, write a concrete
      candidate filter, replay only selected candidates from the simulator
      snapshots inside the held Slurm allocation, and label DP96
      continuability. This remains a diagnostic, not live method evidence.
      2026-06-24 result: completed inside allocation `148680` on `server13`.
      Selection output:
      `experiments/world_model_task_rebinding/cosmos3/selected_causal_suffix_value_head_panel0134_margin0_20260624_171802_alloc148680`.
      Replay output:
      `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_selected_causal_suffix_value_head_panel0134_margin0_exec8_dp96_20260624_171802_alloc148680`.
      The selector chose `11` causal generated candidates and `5` DP-prior
      candidates. Authoritative replay produced `16/16` valid records, no
      process failures, direct post-chunk success `0/16`, direct post-chunk
      gate-ok `0/16`, `candidate + DP96` success `9/16`, and continuability
      `11/16`. This is only a tiny gain over the same-state DP-prior baseline
      (`8/16` success and `8/16` continuability), and it is far below the
      converted-label selector expectation (`15/16`). Evidence note:
      `docs/world_model_task_rebinding/contact_action_world_model/2026-06-24_selected_causal_suffix_value_replay.md`.

- [ ] If selected replay fails to retain most of the oracle DP96 headroom or
      still produces many harmful switches, stop the value-head/scorer line and
      move to direct contact-action executor training or a stronger base policy
      integration.
      2026-06-24 decision: selected replay did not retain the oracle/converted
      label headroom and did not produce direct insertion/gate-ok states. Do
      not tune more scalar margins as the main path. Use value heads only as
      diagnostics while moving to direct contact-action executor labels or a
      stronger WAM/base-policy path.
      2026-06-24 base-policy follow-up: Policy-DROID is now input-schema
      feasible on the active data and has produced one concrete 8-step action
      chunk. The next useful Policy-DROID step is replaying this action chunk
      from a simulator state once torch CUDA works for ManiSkill/DP replay,
      not another scalar selector margin.
- [ ] Live gate: run a full `301/300` panel inside a tmux-held GPU allocation.
      Required evidence: final simulator state, controller timeline, selected
      action family, video/contact-sheet inspection, and explicit statement of
      what the result proves or falsifies.
- [ ] Do not report few-sample/tens-of-seconds training or smoke output as
      method progress. If a debug gate is needed, it must be explicitly marked
      as non-evidence and not repeated as the main work.

## Current Open Questions

- [ ] Does the active simulator expose reliable contact/force/wrench proxies
      that can be used as labels or policy observations without violating the
      causal RGB/RGB-derived method boundary?
- [ ] Which base-policy path has the best integration cost/benefit for this
      repo: local diffusion executor, Octo, pi0/OpenPI, OpenVLA, or residual
      RL?
- [ ] Is current source-suffix retrieval too scenario-specific? If yes, train
      a generator that imitates suffix behavior from causal task-frame inputs
      rather than relying on scenario/source lookup.
