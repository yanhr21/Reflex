# OpenPI pi0.5 Contact-Action TODO

## Current Status

- [x] User pivot recorded: official OpenPI/pi0.5 is now the active action
      model direction; DP is not the main method.
- [x] Local OpenPI checkout found at
      `/public/home/yanhongru/ICLR2027/openpi`, commit `650c5b0`.
- [x] Official pi0.5 paths identified in local code:
      `pi05_base`, `pi05_libero`, `Pi0Config(pi05=True, ...)`,
      `LeRobotLiberoDataConfig`, `CheckpointWeightLoader`.
- [x] Added OpenPI official-style config `pi05_maniskill_peg733` in the local
      OpenPI checkout. It uses official `Pi0Config`, official
      `LeRobotLiberoDataConfig`, and official `pi05_base` weight loading.
- [x] 733 source data located and audited: strict H5 audit reports `733`
      records and `0` failures.
- [x] Existing RGB/action/state export located:
      `full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050`
      with `733` episodes, `9271` prefix rows, `301/300` contract, RGB-only
      input, and raw action dimension `32`.
- [x] Current direct-action failure baseline recorded: old selected/generated
      candidates have `0` direct insertion/gate success on saved snapshot
      replay, despite some DP96 handoff success.
- [x] 2026-06-25 contact-suffix audit completed inside held Slurm allocation
      `150773`, step `19`. Evidence:
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix_audit_20260625_alloc150773/contact_suffix_audit.json`.
      It found `733/733` final-success-like episodes, `733/733` episodes with
      any inserted frame, `733/733` eligible for 16-step suffix extraction, and
      `71045` total eligible suffix windows. First inserted frame ranges from
      `95` to `300` with mean `172.77`. This is a strong positive data signal:
      the accepted 733 data contain insertion/contact actions; the current
      OpenPI failure is more likely suffix supervision/conditioning dilution
      than total absence of insertable actions.
- [x] 2026-06-26 prepared the next OpenPI-native near-contact branch after
      object17-video replay and q3x4 receding both stayed direct-insertion
      negative. Added config
      `pi05_maniskill_peg733_contact_suffix16_object17_video_nearcontact_20260626`
      in the local OpenPI checkout and mapped it in the Slurm training wrapper.
      It uses the same official `Pi0Config(pi05=True, action_horizon=16,
      discrete_state_input=False)` and official `pi05_base` loader. The
      intended repo id is
      `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_nearcontact_20260626`.
      This is a data-window distribution repair, not a new model family or
      scorer.
- [x] Added Slurm-only preparation wrapper
      `scripts/slurm/run_openpi_pi05_nearcontact_object17_video_prepare_in_allocation.sh`.
      It runs conversion, structural audit, and OpenPI-format norm-stat
      fallback from the near-contact repo. It refuses login-node execution and
      records the no-scorer/no-custom-model method boundary.
- [x] Build the near-contact object17/video LeRobot repo inside a tmux-held
      Slurm allocation using the accepted 733 source, `STATE_MODE=object_state17`,
      `CAMERA_STORAGE=video`, H.264 video storage, and offsets
      `16,12,8,4,2,1`. Do not use offset `0` as the main branch because that
      starts at the first already-inserted frame rather than the closest
      pre-insertion action. Evidence:
      `experiments/world_model_task_rebinding/openpi/object17_video_nearcontact_prepare_20260626_offsets16_12_8_4_2_1_alloc153455/convert/conversion_manifest.json`.
      Actual suffix count is `4375`, not the naive `733*6=4398`, because late
      first-insert samples cannot supply every close offset while preserving a
      16-step suffix.
- [x] Audit the near-contact repo structurally in Slurm: expected source
      episodes `733`, state dim `17`, action dim `7`, suffix length `16`, and
      no login-node conversion/audit. Evidence:
      `experiments/world_model_task_rebinding/openpi/object17_video_nearcontact_prepare_20260626_offsets16_12_8_4_2_1_alloc153455/audit/audit_summary.json`,
      `passed=true`, `4375` episodes, `70000` rows, `8750` mp4 files.
- [x] Compute or install fresh OpenPI-format norm stats for
      `pi05_maniskill_peg733_contact_suffix16_object17_video_nearcontact_20260626`.
      Evidence:
      `experiments/world_model_task_rebinding/openpi/object17_video_nearcontact_prepare_20260626_offsets16_12_8_4_2_1_alloc153455/norm_stats/norm_stats_fallback_summary.json`
      and OpenPI asset
      `/public/home/yanhongru/ICLR2027/openpi/assets/pi05_maniskill_peg733_contact_suffix16_object17_video_nearcontact_20260626/yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_nearcontact_20260626/norm_stats.json`.
- [x] Train the near-contact config with official OpenPI scripts and
      `pi05_base` for at least one GPU-hour before interpreting any result.
      Evidence:
      `experiments/world_model_task_rebinding/openpi/pi05_object17_video_nearcontact_direct1700_1gpu1h_pyav_20260626_alloc153455/training_walltime_summary.json`,
      `elapsed_seconds=5044`, `formal_one_gpu_hour_floor_met=true`,
      `train_return_code=0`. Preserved checkpoint:
      `experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733_contact_suffix16_object17_video_nearcontact_20260626/pi05_object17_video_nearcontact_direct1700_1gpu1h_pyav_20260626_alloc153455/1699`.
- [x] Replay the resulting checkpoint from saved dynamic snapshots and compare
      direct success/inserted/contact-stable/grasp preservation against the
      object17-video direct replay and q3x4 receding smoke. Evidence:
      `experiments/world_model_task_rebinding/openpi/20260626_object17_video_nearcontact_direct1700_replay4_exec16_alloc153455/openpi_pi05_snapshot_replay_summary.json`
      and
      `experiments/world_model_task_rebinding/openpi/20260626_object17_video_nearcontact_direct1700_replay4_exec16_alloc153455/contact_state_sheets/contact_state_sheets_manifest.json`.
      Result: direct success `0/4`, inserted `0/4`, contact-stable `0/4`,
      grasp `4/4`, DP96 success/continuable `1/4`. This is worse handoff than
      the previous clean object17-video checkpoint (`3/4` DP96), so near-contact
      window reweighting alone is not the missing insertion solution.

## Do Not Do

- [ ] Do not train new custom VAE/MLP/diffusion intermediate models as the
      main method.
- [ ] Do not present zero-shot pi0.5 inference, a tiny smoke run, or a
      seconds-long finetune as progress.
- [ ] Do not run OpenPI imports, conversion, norm stats, training, inference,
      replay, rendering, or evaluation on the login node.
- [ ] Do not feed future ground-truth hole/peg poses into policy observations.

## Data Adapter

- [x] Decided against DROID-style action semantics for the first run. The 733
      data use ManiSkill `pd_ee_delta_pose`, so the first formal path is
      Libero-style LeRobot plus official `pi05_base` weights and fresh norm
      stats.
- [x] Wrote a 733-to-LeRobot converter:
      `scripts/openpi/convert_maniskill_peg733_to_lerobot.py`. It preserves
      approved RGB frames, qpos-derived state, 7D source H5 robot actions,
      task prompt, source H5 provenance, split, and scenario. It refuses
      login-node execution by default.
- [x] Added Slurm-only conversion wrapper:
      `scripts/slurm/run_openpi_pi05_peg733_lerobot_convert_in_allocation.sh`.
- [x] Run converter inside a tmux-held Slurm allocation.
      Current run:
      `pi05_peg733_lerobot_20260624_existing_cosmos313_lerobot_source_h5lock_alloc149062`
      completed in held job `149062` on `server24`. It produced the
      OpenPI/LeRobot dataset at
      `experiments/world_model_task_rebinding/openpi/lerobot_home/yanhongru/maniskill_peg733_openpi_libero`
      and wrote conversion manifest
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_lerobot_20260624_existing_cosmos313_lerobot_source_h5lock_alloc149062/conversion_manifest.json`.
      This is data conversion only, not training evidence.
- [x] Audit produced LeRobot dataset counts, episode lengths, image shapes,
      action shapes, and train/val split inside Slurm.
- [x] Added Slurm-only LeRobot dataset audit tooling:
      `scripts/openpi/audit_maniskill_peg733_lerobot.py` and
      `scripts/slurm/run_openpi_pi05_peg733_lerobot_audit_in_allocation.sh`.
      It checks `733` episodes, `300` rows per episode, `219900` total rows,
      image/state/action feature shapes, and conversion-manifest consistency.
      Run it only after conversion has finished.
- [x] Audit passed after fixing a PyArrow type-string checker that was too
      strict about `item` versus `element` labels in `fixed_size_list`.
      Evidence:
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_lerobot_audit_20260625_after_h5lock_convert_auditfix_alloc149062/audit_summary.json`
      has `passed=true`, `parquet_file_count=733`,
      `total_parquet_rows=219900`, and `unique_episode_lengths=[300]`.
- [x] Added OpenPI-native contact-suffix converter:
      `scripts/openpi/convert_maniskill_peg733_contact_suffix_to_lerobot.py`.
      It selects short successful insertion windows from the accepted 733 H5
      data using inserted labels only as offline training-data selection
      metadata, while policy inputs remain causal RGB/state at each suffix
      timestep. Default extraction uses offsets
      `64,48,32,24,16,12,8,4` before first insertion with `16` action steps.
- [x] Added Slurm-only contact-suffix conversion wrapper:
      `scripts/slurm/run_openpi_pi05_contact_suffix_lerobot_convert_in_allocation.sh`.
      Smoke conversion completed in allocation `150773`, step `27`, producing
      `16` suffix episodes / `256` frames from `2` source episodes at
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix_lerobot_smoke2_reuseuv_20260625_alloc150773/conversion_manifest.json`.
- [x] Full contact-suffix conversion completed in allocation `150773`, step
      `28`, output root
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix_lerobot_full733_20260625_alloc150773`.
      It produced repo id
      `yanhongru/maniskill_peg733_openpi_contact_suffix16` under the project
      LeRobot home, with `5853` suffix episodes and `93648` total frames. The
      count is slightly below `733*8` because `11` very late insertion
      episodes cannot fit every fixed pre-insertion offset before the 300-step
      action boundary; every source episode still contributed at least one
      valid suffix window.
- [x] Contact-suffix LeRobot audit passed. Evidence:
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix_lerobot_full733_20260625_alloc150773/audit_summary.json`.
      It reports `episodes_jsonl_rows=5853`, `parquet_file_count=5853`,
      `total_parquet_rows=93648`, `unique_episode_lengths=[16]`, image/wrist
      features `256x256x3`, state dim `8`, action dim `7`, and `failures=[]`.

## OpenPI Config

- [x] Add an OpenPI config for the 733 LeRobot dataset using official
      `TrainConfig` style.
- [x] Prefer `Pi0Config(pi05=True, action_horizon=16,
      discrete_state_input=False)` with official `pi05_base` weights and fresh
      norm stats, because DROID action normalization is incompatible with the
      simulator action convention for the first run.
- [x] Keep all weights/checkpoints OpenPI-compatible; no hand-written
      intermediate architecture.
- [x] Added official OpenPI config
      `pi05_maniskill_peg733_contact_suffix16` in
      `/public/home/yanhongru/ICLR2027/openpi/src/openpi/training/config.py`.
      It uses `Pi0Config(pi05=True, action_horizon=16,
      discrete_state_input=False)`, `LeRobotLiberoDataConfig`, official
      `pi05_base` checkpoint loading, and repo id
      `yanhongru/maniskill_peg733_openpi_contact_suffix16`.

## Training

- [x] Compute norm stats with OpenPI's official script inside Slurm.
      Completed run:
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_1gpu1h_20260625_after_hf_sequence_metadata_repair_alloc150311`
      in tmux-held Slurm job `150311` on `server38`.
- [x] 2026-06-25 norm stats completed for `pi05_maniskill_peg733` in Slurm
      job `150311`. Evidence:
      `/public/home/yanhongru/ICLR2027/openpi/assets/pi05_maniskill_peg733/yanhongru/maniskill_peg733_openpi_libero/norm_stats.json`
      exists, and the completed `compute_norm_stats.log` wrote stats to that
      path after `3435/3435` batches. This is official OpenPI data
      normalization evidence, not model training evidence.
- [x] Launch one formal pi0.5 finetune on real 733-derived data inside a
      tmux-held Slurm allocation.
- [x] Added Slurm-only OpenPI norm-stats/training wrapper:
      `scripts/slurm/run_openpi_pi05_peg733_train_in_allocation.sh`.
- [x] Fixed the training wrapper to unset deprecated `LEROBOT_HOME` and use
      `HF_LEROBOT_HOME`, matching the converter-side official LeRobot
      requirement.
- [x] Enforce at least one GPU-hour before interpreting training result.
      Completed formal run:
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_1gpu1h_20260625_after_nfs_enolck_localckpt_1600_skipnorm_alloc150773`
      in tmux-held Slurm allocation `150773` on `server38`. The official
      OpenPI `scripts/train.py` run used the audited 733 LeRobot dataset,
      official `pi05_base/params` weights restored from the repaired OpenPI
      cache, existing official norm stats, W&B offline logging, and OpenPI
      TrainConfig CLI overrides only: `--num-train-steps=1600
      --save-interval=800`. `training_walltime_summary.json` reports
      `elapsed_seconds=4152`, `min_wall_seconds=3660`,
      `formal_one_gpu_hour_floor_met=true`, `train_return_code=0`,
      `slurm_step_id=2`. The train log has real optimization evidence:
      loss decreased from `0.1017` at step `0` to `0.0204` at step `1500`.
      This is a real pi0.5 finetune result, not a smoke run, but it is not yet
      insertion/replay success evidence.
- [x] Record train duration, GPU allocation, config, base checkpoint,
      norm-stat source, step count, and checkpoint path.
      Final checkpoint `1599` was first saved on compute-node local disk to
      avoid NFS/Orbax locks, then preserved under the active experiment tree:
      `experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733/pi05_peg733_1gpu1h_20260625_after_nfs_enolck_localckpt_1600_skipnorm_alloc150773/1599`.
      Copy verification matched source and destination exactly: `57` files and
      `44697898725` bytes. The checkpoint contains finalized `assets`,
      `params`, and `train_state` directories.
- [ ] 2026-06-25 after the weak saved-snapshot replay result below, a longer
      official OpenPI resume run was launched in the same held allocation
      `150773`, restoring the node-local checkpoint at step `1599` and
      continuing the same OpenPI experiment to `4000` train steps with
      `--resume --num-train-steps=4000 --save-interval=1200`. The run
      directory is
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_resume4000_20260625_after_panel4_weak_prefixobs_alloc150773`.
      Latest log evidence shows the checkpoint restore completed from
      `/tmp/openpi_pi05_checkpoints_yanhongru_150773_local/.../1599` and
      training progressed past step `1600`. The first resume save at step
      `2400` completed on node-local `/tmp` without `ENOLCK` or asynchronous
      save errors, and the second resume save at step `3600` also finalized
      cleanly. The run reached step `3999`, but the final checkpoint did not
      finalize and Slurm later marked step `150773.11` as `OUT_OF_MEMORY`
      with `MaxRSS=94234168K`; only an incomplete
      `3999.orbax-checkpoint-tmp-*` existed. Therefore the authoritative
      resume checkpoint is step `3600`, not `3999`.
- [x] Preserve the completed resume checkpoint.
      Step `3600` was copied from server38 node-local `/tmp` to
      `experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733/pi05_peg733_resume4000_20260625_after_panel4_weak_prefixobs_alloc150773/3600`.
      Copy verification matched source and destination exactly: `52` files and
      `44671728268` bytes.
- [x] Formal contact-suffix OpenPI/pi0.5 training completed after fixing the
      checkpoint-memory blocker. The first suffix run reached step `1000` but
      Slurm marked it `OUT_OF_MEMORY` during Orbax checkpoint save, leaving
      only `1000.orbax-checkpoint-tmp-*`; it is not a valid checkpoint. The
      successful run used the same official
      `pi05_maniskill_peg733_contact_suffix16` config with `ema_decay=None`
      to avoid duplicating current params and EMA params during checkpoint
      save, reused official suffix norm stats, skipped mid-run saves with
      `--save-interval=5000`, and trained for `1700` steps in allocation
      `150773`, step `44`. Evidence root:
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix16_noema_direct1700_1gpu1h_20260626_alloc150773`.
      `training_walltime_summary.json` reports `elapsed_seconds=4303`,
      `formal_one_gpu_hour_floor_met=true`, and `train_return_code=0`.
      Train loss decreased from `0.1038` at step `0` to `0.0373` at step
      `1600`, with a noisy peak around step `100`.
- [x] Preserve the completed contact-suffix checkpoint. Step `1699` finalized
      on server38 node-local `/tmp` and was copied to
      `experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733_contact_suffix16/pi05_peg733_contact_suffix16_noema_direct1700_1gpu1h_20260626_alloc150773/1699`.
      Copy verification matched source and destination: `31G` and `71` files.
- [x] Evaluate the contact-suffix checkpoint from saved dynamic snapshots.
      The first 4-sample exec16 panel completed inside allocation `150773`.
      Evidence root:
      `experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix16_noema1699_panel4_exec16_alloc150773_alloc150773`.
      Summary: direct success `0/4`, inserted `0/4`, contact-stable `0/4`,
      grasp preserved `4/4`, DP96 continuable `2/4`, DP96 success `1/4`.
      This is negative task evidence: contact-suffix training fixed the
      data/training/checkpoint path, but did not make the policy insert from
      these dynamic snapshots.
- [x] Build contact-state sheets for the contact-suffix replay inside the
      held Slurm allocation. Evidence:
      `experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix16_noema1699_panel4_exec16_alloc150773_alloc150773/contact_state_sheets/contact_state_sheets_manifest.json`.
      The sheets confirm the metric boundary: direct inserted `0/4`, direct
      success `0/4`, direct grasped `4/4`, DP96 success `1/4`. The f116 sheet
      is a positive handoff diagnostic only; it has no inserted step during
      the OpenPI chunk.
- [ ] Diagnose why the suffix checkpoint still misses insertion. Initial
      replay evidence shows the generated chunks preserve grasp but often
      drive the peg laterally or farther negative in the hole-frame x axis:
      f106 changes rel-x from `-0.148` to `-0.198` and abs(y)+abs(z) from
      `0.054` to `0.122`; f132 changes abs(y)+abs(z) from `0.013` to
      `0.089`; f116 stays DP-continuable but still has no direct insertion.
      The next fix should change the conditioning/object-frame alignment or
      training target distribution, not add a scorer-only selector.
- [x] First action/object-frame diagnosis completed inside Slurm using
      `scripts/openpi/diagnose_openpi_contact_suffix_replay.py`. Baseline
      replay diagnosis:
      `experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix1699_panel4_exec16_alloc150773/contact_suffix_replay_diagnosis.json`.
      It found `4/4` samples worsened `abs_yz_sum`, `3/4` worsened `abs_x`,
      mean `delta_abs_yz_sum=0.03767`, and mean `delta_abs_x=0.02172`.
      The tested prefixes were `50..66` frames before source first insertion,
      near but not exactly at the fixed training offsets.
- [x] Privileged prompt-phase diagnostic completed:
      `experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix1699_panel4_exec16_privprompt_diag_alloc150773`.
      It used source first-insert timing in the prompt and is explicitly not
      causal method evidence. Result stayed negative: direct success `0/4`,
      inserted `0/4`, contact-stable `0/4`, grasp `4/4`, DP96 continuable
      `1/4`. Its diagnosis still worsened `abs_x` in `4/4` and `abs_yz_sum`
      in `3/4`, so prompt-phase mismatch is not the primary blocker.
- [ ] Build the next OpenPI-native repair around causal object/task-frame
      conditioning. The policy state should expose current peg/hole/TCP
      relative geometry derived from allowed observations/metadata, with
      OpenPI normalization and official config transforms. Do not solve this
      by adding another scorer-only selector or by training a custom
      intermediate VAE/MLP/diffusion model.
- [x] Built the first object/task-frame conditioning diagnostic dataset as an
      OpenPI-compatible LeRobot rewrite, not a new model. Repo id:
      `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17`. State is
      17D simulator-derived current task state:
      `tcp_pose3, peg_pose3, hole_pose3, peg_head_at_hole3,
      hole_velocity_step3, grasped, inserted`. This is an upper-bound
      diagnostic for object binding; it is not yet the publishable RGB-derived
      perception path.
- [x] Object17 audit passed inside held Slurm allocation `150773`, step `77`.
      Evidence:
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix_lerobot_object17_rewrite_20260626_alloc150773/audit_summary.json`.
      It reports `episodes_jsonl_rows=5853`, `total_parquet_rows=93648`,
      `unique_episode_lengths=[16]`, image/wrist `256x256x3`, state dim `17`,
      action dim `7`, `record_start_grasped_count=5780`, and
      `record_end_inserted_count=2906`, with `failures=[]`.
- [x] Added official OpenPI config
      `pi05_maniskill_peg733_contact_suffix16_object17` in the local OpenPI
      checkout. It keeps official `Pi0Config(pi05=True, action_horizon=16,
      discrete_state_input=False)`, official `LeRobotLiberoDataConfig`, and
      official `pi05_base` loading. The change is data conditioning, not a
      custom VAE/MLP/diffusion/action model.
- [x] Official `compute_norm_stats.py` repeatedly stalled on the object17
      rewritten LeRobot dataset after Hugging Face split generation, so a
      data-loading-only fallback norm pass was run inside Slurm step `85`:
      `scripts/openpi/compute_lerobot_state_action_norm_stats.py`. It uses
      OpenPI `normalize.RunningStats` and `normalize.save` over LeRobot
      parquet `state/actions`; it does not change policy code or action
      representation. Evidence:
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix16_object17_norm_fallback_20260626_alloc150773/norm_stats_fallback_summary.json`.
      It wrote OpenPI-format stats for `93648` rows, state dim `17`, action
      dim `7`.
- [ ] Object17 training is currently blocked before first optimization step.
      Direct official `scripts/train.py` launch with existing object17 norm
      stats reached config/norm loading, allocated the pi0.5 model on GPU, and
      generated the `93648`-example train split, but then produced no `Step 0`
      log for about 26 minutes. The step was cancelled to avoid idle GPU burn:
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix16_object17_direct1700_skipnorm_1gpu1h_20260626_alloc150773`.
      `tmux_driver.log` ends with Slurm step `150773.87` cancelled and return
      code `137`. This is a dataloader/JAX/train-initialization blocker, not
      an insertion-performance result.
- [ ] Next compute-side action for object17: debug the official OpenPI data
      loader/training initialization inside the held allocation, likely by
      reducing or disabling dataset workers/prefetch/persistent loader behavior
      through official config-compatible knobs, or by fixing the LeRobot cache
      path. Do not interpret object17 until a real run passes `Step 0` and
      then satisfies the one-GPU-hour floor.
- [x] Extended the snapshot observation preparer for object17 replay
      preparation. `scripts/openpi/prepare_openpi_pi05_snapshot_observations.py`
      now builds 17D state by explicitly concatenating causal saved history
      columns `tcp 0:3`, `peg 3:6`, `hole 6:9`, `rel3 27:30`,
      `holevel 30:33`, and `flags 33:35`. This is a static code fix only;
      it still needs compute-side validation and replay after object17
      training exists. The replay would remain a simulator-state upper-bound
      diagnostic until RGB-derived object/task-state perception replaces these
      slots.
- [x] Found a previously hidden data-corruption issue in the object17 rewrite
      path. `scripts/openpi/rewrite_contact_suffix_lerobot_object_state.py`
      used `cp -al` hardlinks before editing `meta/info.json` and parquet
      state columns. Because hardlinks share inodes, the canonical qpos8
      contact-suffix repo was also mutated to object17 metadata/state. This
      means the old object17 `Step 0` blocker was diagnosed on a contaminated
      data state and must not be treated as a clean OpenPI/LeRobot conclusion.
- [x] Patched the object17 rewrite script to use real `shutil.copytree`
      copies, not hardlinks. Also patched both contact-suffix conversion
      wrappers to refuse destructive `OVERWRITE=true` on canonical repo id
      `yanhongru/maniskill_peg733_openpi_contact_suffix16` unless
      `ALLOW_DESTRUCTIVE_CANONICAL_OVERWRITE=true` is explicitly set.
- [ ] Canonical qpos8 contact-suffix repo is currently not usable. A repair
      conversion was launched in held allocation `150773`, step `92`, but the
      allocation entered `COMPLETING` and Slurm cancelled the step at about
      `97%`. Current observed metadata has only `5704` episodes / `91264`
      frames with state dim `8`, so it is a partial dataset. Do not train,
      compute norm stats, or replay from this repo until it is rebuilt and
      audited.
- [ ] Old object17 repo is also not usable as method evidence until rebuilt
      from a clean source without hardlinks. The next compute allocation
      should convert qpos8 and object17 into fresh noncanonical repo ids,
      audit both, then either update OpenPI configs to the fresh repo ids or
      perform a controlled replacement after successful audit. Avoid in-place
      overwrite of active canonical data.
- [x] Clean noncanonical qpos8 and object17 rebuilds were completed and
      audited under fresh repo ids, avoiding the hardlink-contaminated canonical
      trees:
      `yanhongru/maniskill_peg733_openpi_contact_suffix16_qpos8_clean_20260626`
      and
      `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_clean_20260626`.
      Both preserve the accepted 733 source trajectories, `5853` suffix
      episodes, `93648` rows, `16`-step windows, 7D actions, and the qpos8 or
      object17 state contract. These are data-repair results, not training
      results.
- [x] Diagnosed the clean image-backed object17 repo as structurally valid but
      operationally blocked for official OpenPI training. Official
      `LeRobotDataset(repo_id)` hung before returning `dataset[0]`. The likely
      bottleneck is embedded image storage: two `256x256` RGB camera streams
      are stored as `dtype=image` payloads across `5853` per-episode parquet
      files, roughly `18 GB` total.
- [x] Added a video-backed clean object17 repair path without changing model
      architecture or weights. The converter now supports
      `--args.camera-storage video` and a configurable official LeRobot video
      codec path; the active repaired repo id is
      `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_clean_20260626`.
      The OpenPI config
      `pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626`
      uses official `Pi0Config(pi05=True, action_horizon=16,
      discrete_state_input=False)`, official `LeRobotLiberoDataConfig`, and
      official `pi05_base` checkpoint loading.
- [x] H.264 video-backed object17 conversion completed in held Slurm
      allocation `152622` on `server60`.
      Evidence:
      `experiments/world_model_task_rebinding/openpi/object17_video_clean_rebuild_h264_20260626_alloc152622/convert_driver_rc.txt`
      reports `object17_video_h264_convert_rc=0`.
- [x] Strict video-backed object17 audit passed.
      Evidence:
      `experiments/world_model_task_rebinding/openpi/object17_video_clean_audit_20260626_alloc152622/audit_summary.json`.
      It reports `passed=true`, `parquet_file_count=5853`,
      `video_file_count=11706`, `total_parquet_rows=93648`,
      `unique_episode_lengths=[16]`, `info_total_episodes=5853`,
      `info_total_frames=93648`, state dim `17`, action dim `7`, and
      camera dtypes `image=video`, `wrist_image=video`.
- [x] OpenPI-format norm stats were computed for the video-backed object17
      config inside Slurm using the OpenPI normalize fallback over LeRobot
      `state/actions`.
      Evidence:
      `experiments/world_model_task_rebinding/openpi/object17_video_clean_pyav_debug_20260626_alloc152622/norm_stats_fallback_summary.json`
      and installed stats at
      `/public/home/yanhongru/ICLR2027/openpi/assets/pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626/yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_clean_20260626/norm_stats.json`.
      Summary: `93648` rows, state dim `17`, action dim `7`.
- [x] Video-backed object17 now passes the expensive split-generation stage
      that blocked the image-backed path. The staged debug generated the full
      `93648`-example train split, which is a positive dataloader signal but
      still not a training result.
- [x] Narrowed the video-backed object17 loader blocker. Direct H.264/pyav
      decode is not the cause: metadata, external image mp4 decode,
      wrist-image mp4 decode, and all-16-frame decode for episode `0` pass
      inside Slurm. Evidence:
      `experiments/world_model_task_rebinding/openpi/object17_video_decode_item_debug_20260626_alloc152622_reuseenv`.
      The failing path was Hugging Face `datasets.load_dataset` using the
      default shared cache; even a single episode generated the 16-row split
      and then did not return.
- [x] Confirmed the cache fix. With `HF_HOME`, `HF_DATASETS_CACHE`, and
      `XDG_CACHE_HOME` redirected to compute-node `/tmp`, single-episode
      manual constructor, no-delta item, and delta item all pass:
      `experiments/world_model_task_rebinding/openpi/object17_video_hf_tmpcache_debug_20260626_alloc152622`.
      Evidence: `manual_one_tmpcache_rc=0`,
      `dataset_one_nodelta_tmpcache_rc=0`,
      `dataset_one_delta_tmpcache_rc=0`, item retrieval about `0.05..0.06 s`.
- [x] Patched OpenPI Slurm wrappers to default Hugging Face datasets/cache
      paths to compute-node `/tmp`: `HF_HOME`, `HF_DATASETS_CACHE`, and
      `XDG_CACHE_HOME`. This is an operational cache repair only; it does not
      alter OpenPI model weights, architecture, action representation, or the
      LeRobot dataset schema.
- [x] With `/tmp` HF cache, full video-backed raw item now passes:
      `experiments/world_model_task_rebinding/openpi/object17_video_clean_tmpcache_postdebug_20260626_alloc152622`.
      `first_batch_rcs.txt` reports `raw_item_rc=0`; `raw_item_debug.json`
      reports `create_dataset_seconds=37.52`, `item_seconds=0.10`, image and
      wrist image tensors `3x256x256`, state dim `17`, action chunk `16x7`.
- [x] Video-backed object17 transformed item / first-batch gate passed after
      the `/tmp` Hugging Face cache repair and OpenPI-root working-directory
      patch. Evidence:
      `experiments/world_model_task_rebinding/openpi/object17_video_clean_tmpcache_postdebug_patched_20260626_alloc153455`.
      `first_batch_rcs.txt` reports `raw_item_rc=0`,
      `transformed_item_rc=0`, and `first_batch_rc=0`; the transformed item
      has padded state dim `32`, action chunk `16x32`, and `224x224` images.
- [x] Formal video-backed object17 OpenPI/pi0.5 training completed in held
      Slurm allocation `153455` on real 733-derived data. Evidence root:
      `experiments/world_model_task_rebinding/openpi/pi05_object17_video_clean_direct1700_1gpu1h_20260626_alloc153455`.
      It used official config
      `pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626`,
      official `Pi0Config(pi05=True, action_horizon=16,
      discrete_state_input=False)`, official `pi05_base` restore, audited
      video-backed object17 LeRobot data, and no custom VAE/MLP/diffusion
      executor. `training_walltime_summary.json` reports
      `elapsed_seconds=4748`, `formal_one_gpu_hour_floor_met=true`, and
      `train_return_code=0`. Final checkpoint `1699` was preserved at
      `experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626/pi05_object17_video_clean_direct1700_1gpu1h_20260626_alloc153455/1699`.

## Evaluation

- [x] Build an OpenPI saved-live-snapshot query/replay wrapper using official
      OpenPI policy inference.
      Added `scripts/openpi/replay_openpi_pi05_action_from_snapshots.py` and
      split-environment helpers
      `scripts/openpi/prepare_openpi_pi05_snapshot_observations.py` and
      `scripts/openpi/infer_openpi_pi05_from_prepared_observations.py`, with
      Slurm-only wrapper
      `scripts/slurm/run_openpi_pi05_snapshot_replay_in_allocation.sh`. The
      script restores saved dynamic live snapshots, renders the current RGB
      observation, builds the same 8D `qpos[:7] + mean(finger qpos)` state used
      by the 733 LeRobot conversion, calls official
      `policy_config.create_trained_policy(config, checkpoint_dir)` on the
      preserved pi0.5 checkpoint, then reuses the existing saved-snapshot
      ManiSkill replay evaluator for direct success/contact/DP96 labels. This
      is an evaluation bridge only; it introduces no custom VAE/MLP/diffusion
      intermediate model.
- [ ] Evaluate direct short-chunk replay on saved dynamic live snapshots.
      Next compute-node launch should start with the default 1-sample/1-iter
      replay smoke inside held allocation `150773`:
      `STAMP=20260625_pi05_snapshot_smoke MAX_SAMPLES=1 MAX_ITER_DIRS=1 srun --jobid=150773 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=120G bash scripts/slurm/run_openpi_pi05_snapshot_replay_in_allocation.sh`.
      Do not run this on the login node.
      2026-06-25 first attempt
      `openpi_pi05_snapshot_replay_20260625_pi05_snapshot_smoke_alloc150773`
      failed before inference because the OpenPI uv environment lacks
      `sapien`; this is an environment-composition blocker, not action-quality
      evidence. A compute-node dependency probe showed project `.venv` has
      `sapien` and `mani_skill` but lacks `jax/flax/orbax`, while the OpenPI uv
      env has OpenPI/JAX but lacks `sapien`, motivating the split-env wrapper.
      2026-06-25 split-env attempt
      `openpi_pi05_snapshot_replay_20260625_pi05_snapshot_smoke_splitenv_alloc150773`
      bypassed the `sapien` import failure, but the prepare phase produced only
      the SAPIEN Vulkan warning and no `prepared_observations_manifest.json`;
      Slurm showed the step with no useful progress, so step `150773.6` was
      cancelled while preserving allocation `150773`. This is a render/env
      initialization or Slurm step-launch blocker, not an OpenPI insertion
      result.
      2026-06-25 prefix-observation repair changed the default observation
      preparation to use the last frame from each saved `observed_prefix.mp4`
      and qpos reconstructed from `live_history_raw_action_state.json`, avoiding
      live ManiSkill rendering during OpenPI inference preparation while keeping
      replay itself inside the ManiSkill/SAPIEN project environment.
      The 1-sample smoke
      `openpi_pi05_snapshot_replay_20260625_pi05_snapshot_smoke_prefixobs_alloc150773`
      completed end to end: direct success `0/1`, inserted `0/1`,
      contact-stable `0/1`, grasp preserved `1/1`, DP96 success/continuability
      `0/1`.
      The 4-sample panel
      `openpi_pi05_snapshot_replay_20260625_pi05_snapshot_panel4_prefixobs_alloc150773`
      also completed end to end: direct success `0/4`, inserted `0/4`,
      contact-stable `0/4`, grasp preserved `4/4`, DP96
      success/continuability `1/4`. Per-label diagnostics show
      `openpi_pi05_0001_iter_000_f094` became DP96-continuable/final inserted
      after the baseline handoff, but the OpenPI chunk itself still did not
      directly insert. This is a positive signal for grasp preservation and
      one continuable state, not method success.
      2026-06-25 replay of the longer step-`3600` resume checkpoint completed
      in
      `openpi_pi05_snapshot_replay_20260625_pi05_snapshot_panel4_prefixobs_resume3600_alloc150773_alloc150773`.
      Summary: direct success `0/4`, inserted `0/4`, contact-stable `0/4`,
      grasp preserved `4/4`, DP96 success/continuability `2/4`.
      Per-label: `0000_f106` and `0001_f094` were DP96-success/continuable
      after the OpenPI chunk; `0002_f132` and `0003_f116` remained failures.
      This is a real positive movement over the `1600` checkpoint on
      continuability (`1/4` to `2/4`) while still failing the direct insertion
      objective.
      A follow-up run using the model's full native `16`-action horizon,
      `openpi_pi05_snapshot_replay_20260625_pi05_snapshot_panel4_prefixobs_resume3600_exec16_alloc150773_alloc150773`,
      did not improve the direct result: direct success `0/4`, inserted
      `0/4`, contact-stable `0/4`, grasp preserved `4/4`, DP96
      success/continuability `2/4`. Therefore the failure is not just that the
      earlier replay executed only 8 of the 16 OpenPI actions.
- [x] Add non-render contact-state evidence for the current OpenPI replay.
      Live simulator RGB rendering from the replay labels was attempted in
      Slurm step `150773.14`, but it produced no visual files after about
      `10:47` and was cancelled; GPU utilization stayed high while no output
      appeared, matching the earlier live-render/Vulkan blocker. This is a
      render evidence blocker, not an action-quality result.
      Added `scripts/openpi/build_openpi_pi05_contact_state_sheets.py`, which
      renders PNG sheets from authoritative saved replay `step_records`
      without rerunning policy inference, changing action selection, or using
      simulator render. Evidence root:
      `experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260625_pi05_snapshot_panel4_prefixobs_resume3600_exec16_alloc150773_alloc150773/contact_state_sheets`.
      Manifest confirms `4` sheets, direct success `0/4`, direct inserted
      `0/4`, direct grasped `4/4`, DP96 success `2/4`. The sheet for
      `openpi_pi05_0001_iter_000_f094` was visually opened and is readable:
      it shows grasp maintained for `16/16` OpenPI steps, no inserted step
      during the OpenPI chunk, and DP96 success/continuability. These sheets
      are visual diagnostics of state/contact metrics, not RGB simulator
      videos and not success claims.
- [x] Evaluate the video-backed object17 OpenPI/pi0.5 checkpoint from saved
      dynamic snapshots. Evidence root:
      `experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_direct1700_replay4_exec16_fixhistory_alloc153455`.
      The first attempt exposed a replay-preparation bug: old live-receding
      panels store action first and task fields in columns `7:24`, so
      `scripts/openpi/prepare_openpi_pi05_snapshot_observations.py` was
      patched to map that layout to the same object17 state used in training.
      The rerun completed with wrapper rc `0`: prepared observations `4`,
      OpenPI action chunks `4`, direct success `0/4`, direct inserted `0/4`,
      direct contact-stable `0/4`, grasp preserved `4/4`, DP96
      continuable/success `3/4`. This is the strongest OpenPI handoff signal
      so far, but still not direct insertion success.
- [x] Build contact-state sheets for the video-backed object17 replay inside
      held Slurm allocation `153455`. Evidence:
      `experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_direct1700_replay4_exec16_fixhistory_alloc153455/contact_state_sheets/contact_state_sheets_manifest.json`.
      Manifest confirms `4` sheets, direct success `0/4`, direct inserted
      `0/4`, direct grasped `4/4`, DP96 success `3/4`, and no inserted step
      during the OpenPI chunk. This supports the conclusion that object17
      improves continuability but still leaves the final contact/insertion
      action unsolved.
- [x] Report direct success, inserted, contact-stable, grasp preservation,
      insertion-axis progress, and optional DP96 continuability baseline for
      the current object17-video checkpoint in the active status document.
- [x] Add and run a minimal split-env receding OpenPI diagnostic. Added
      `scripts/openpi/run_openpi_pi05_receding_snapshot_rollout.py` and
      `scripts/slurm/run_openpi_pi05_receding_snapshot_rollout_in_allocation.sh`.
      The script keeps ManiSkill execution in the project environment and
      calls official OpenPI inference in a subprocess; it introduces no custom
      action model or scorer. Smoke evidence:
      `experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_receding1_q3_exec4_alloc153455`.
      Result on `sample_00_hole_late_move_stop`, prefix `106`: `3` queries,
      `4` steps each, `12` OpenPI steps total, direct success `0/1`, inserted
      `0/1`, contact-stable `0/1`, grasp preserved `1/1`, final
      peg-head-at-hole `[-0.2088, 0.1156, -0.0186]`, and
      `abs(y)+abs(z)` worsened from `0.0539` to `0.1341`. This is an
      upper-bound diagnostic using refreshed simulator object17 state plus a
      static observed-prefix image, not final RGB-derived method evidence.
- [ ] Inspect RGB video/contact-sheet evidence before claiming manipulation
      success. Current contact-state sheets are useful diagnostics, but RGB
      replay video remains blocked by the render/Vulkan path.
- [ ] Compare against the old direct-action baseline: `0/16` selected replay
      direct success/gate and `0/192` candidate direct success/gate.

## Resource State

- [x] Old allocation `148732` on `server24` is no longer usable. It was
      revoked by the cluster after the dependency-download retries; later
      `srun --jobid=148732` reported `Slurm job 148732 has expired`.
- [x] New tmux-held interactive allocation
      `openpi_pi05_peg733_1gpu_request_20260624` / Slurm job `149062` is
      allocated on `server24` and currently held with only `149062.extern`
      running. Preserve this allocation for conversion/training; do not
      release it just because the foreground command failed.
- [ ] First conversion launch attempt in `148732` failed before code execution
      because `--mem=80G` exceeded available allocation memory.
- [ ] Second conversion attempt entered compute step `148732.42` but failed
      during OpenPI `uv run` dependency setup because the inherited
      `http_proxy=https_proxy=http://127.0.0.1:37890` proxy was dead on the
      compute node. The wrappers now unset proxy variables before `uv`.
- [ ] Third conversion attempt entered compute step `148732.43`; proxy was
      fixed and many wheels downloaded, but the pinned official LeRobot git
      dependency failed with `GnuTLS recv error (-110)` while fetching
      `https://github.com/huggingface/lerobot` commit
      `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`. This is an OpenPI
      dependency-download blocker, not an OpenPI model/data result.
- [ ] Conversion retry in allocation `149062` also failed before data
      conversion because `uv` again could not fetch the official pinned
      LeRobot commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`, with the
      same `GnuTLS recv error (-110)`. This is still a dependency-download
      blocker, not a data/model/training result.
- [ ] Login-node dependency download attempts for the same official pinned
      LeRobot source have also failed at the network transport layer:
      GitHub codeload tarball stopped twice around `13M` and failed tar
      integrity with `gzip: stdin: unexpected end of file`; filtered
      `git clone --filter=blob:none --no-checkout` hung and then failed with
      `fetch-pack: unexpected disconnect while reading sideband packet` when
      interrupted. Do not use the incomplete tarballs as source.
- [ ] Next dependency action: obtain a valid local mirror/archive of the exact
      official LeRobot commit, then point OpenPI's dependency source to that
      local official checkout as a source-transport repair only. If this keeps
      failing, report the dependency mirror blocker before trying speculative
      model/protocol changes.
- [x] Dependency-source repair completed: downloaded a valid GitHub zip archive
      for official LeRobot commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`,
      verified it with `unzip -t`, extracted it to
      `/public/home/yanhongru/ICLR2027/external_sources/lerobot_0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`,
      and changed OpenPI's `pyproject.toml` `lerobot` source to that local
      path. This is a dependency transport repair only; OpenPI model
      architecture and pi0.5 checkpoint source remain unchanged.
- [x] The next conversion retry reached the other pinned git dependency
      `dlimp` and stalled there. Downloaded and verified the exact official
      `dlimp` commit `ad72ce3a9b414db2185bc0b38461d4101a65477a`, extracted it
      to
      `/public/home/yanhongru/ICLR2027/external_sources/dlimp_ad72ce3a9b414db2185bc0b38461d4101a65477a`,
      and changed OpenPI's `pyproject.toml` `dlimp` source to that local path.
      This is also only a dependency transport repair.
- [ ] Next allocation action: rerun 733-to-LeRobot conversion in held Slurm
      allocation `149062` using the local official LeRobot and dlimp sources.
- [x] `uv.lock` was also repaired to point `lerobot` and `dlimp` at the local
      official source mirrors. The conversion/norm/training wrappers now call
      `uv run --locked --python-platform x86_64-unknown-linux-gnu`.
- [x] Diagnosed an additional cluster/filesystem issue: uv cache/project envs
      under the shared experiment/NFS path fail with
      `No locks available (os error 37)`. The wrappers now set
      `UV_CACHE_DIR` and `UV_PROJECT_ENVIRONMENT` under compute-node `/tmp`
      and use `UV_LINK_MODE=copy`.
- [ ] Latest conversion retry with `/tmp` uv env reached
      `Creating virtual environment at: /tmp/openpi_uv_env_yanhongru_149062_9`,
      but Slurm step `149062.9` then showed `RUNNING` with no task PID from
      `scontrol listpids` and `0` average CPU. It was cancelled while
      preserving allocation `149062`. This is a Slurm step launch/accounting
      blocker, not a model/data/training result.
- [ ] Next operational action: retry conversion in the held allocation with a
      fresh Slurm step after the step-launch anomaly clears, or open an
      interactive shell step on the allocated node and run the wrapper there.
      Keep using `/tmp` uv cache/env and local official LeRobot/dlimp mirrors.
- [x] Opened interactive shell step `149062.10` on `server24`, avoiding the
      earlier no-task-PID Slurm step anomaly while preserving the held
      allocation.
- [x] Diagnosed the Python version issue: a Python 3.12 pip environment cannot
      install the official pinned `dlimp` dependency because it requires
      `tensorflow==2.15.0`, which has no Python 3.12 wheel. Created a Python
      3.11 environment at `/tmp/openpi_py311_env_yanhongru_149062` inside the
      compute node instead.
- [ ] Current allocation action: install official OpenPI plus local official
      LeRobot/dlimp mirrors into the Python 3.11 compute-node environment.
      This is still dependency setup, not a model/data/training result.
- [x] Conversion and training wrappers now accept `OPENPI_PYTHON=/path/to/python`
      so the same official OpenPI scripts can run through the compute-node
      Python 3.11 environment when `uv` is blocked by network/lock behavior.
- [x] Diagnosed that `dlimp` must not be installed into the main pi0.5 training
      environment: `dlimp -> tensorflow==2.15.0 -> ml-dtypes~=0.2.0`
      conflicts with OpenPI's `jax==0.5.3 -> ml_dtypes>=0.4.0`. `dlimp` is
      only an RLDS/source dependency, not required for the 733 LeRobot
      conversion or pi0.5 training path.
- [x] Added `scripts/openpi/openpi_py311_main_constraints.txt`, copied from the
      official OpenPI `uv.lock`, to keep pip from unbounded resolver backtracking
      when cluster `uv sync` is blocked.
- [ ] Latest main-environment blocker: constrained Python 3.11 pip install can
      resolve OpenPI's official dependencies, but PyPI download throughput on
      `server24` is unusably slow for large official wheels. `torch==2.7.1`
      began downloading at roughly `14-16 KB/s` with an estimated `14-16`
      hours for the single `821 MB` wheel, before additional CUDA/JAX wheels.
      This is a dependency-download/resource blocker, not a model, data, or
      training result.
- [ ] Existing local environments contain non-matching torch versions
      (`2.5.1+cu121`, `2.10.0+cu128`) and some unrelated LeRobot installs, but
      no observed OpenPI-compatible `torch==2.7.1` environment. They may be
      useful for one-off data conversion only if documented, but must not be
      reported as the formal OpenPI training environment.
- [ ] Next dependency options: use a faster wheel source/cache for the official
      OpenPI lock, pre-stage the official torch/JAX wheels outside the held GPU
      allocation, or run data conversion with a separate non-training converter
      environment while keeping formal pi0.5 training blocked until official
      OpenPI dependency/weight requirements are satisfied.
- [x] 2026-06-25 conversion state: job `149062` completed the full 733-row
      LeRobot conversion on `server24`; the HDF5 lock fix held through the
      complete run. Do not interpret this as pi0.5 training progress.
- [ ] 2026-06-25 official OpenPI norm-stats attempt after dataset audit:
      `scripts/slurm/run_openpi_pi05_peg733_train_in_allocation.sh` was
      launched in held job `149062` with `uv run --locked`, but it stopped
      before norm stats because OpenPI's `uv.lock` content hash was stale
      after the local official LeRobot/dlimp source-mirror repair:
      `The lockfile at uv.lock needs to be updated, but --locked was provided`.
      A compute-node `uv lock` attempt was then run for about `9.5` minutes;
      it stayed in low-CPU network/metadata work, grew
      `/tmp/openpi_uv_cache_yanhongru_149062_lock` only to about `103M`, and
      was interrupted to avoid burning the held GPU allocation. No norm stats
      or training started.
- [ ] Next dependency action: rerun official norm stats/training with
      `uv run --frozen` and native compute-node platform. Regenerating the
      lock is no longer the preferred first repair because the existing lock
      already encodes the official package versions, and `--frozen` avoids
      unnecessary dependency re-resolution. If a full dependency install is
      still slow, pre-stage official wheels/cache outside the held GPU
      allocation when possible; do not train from the non-matching existing
      local torch environments.
- [x] 2026-06-25 weight-download repair: installed user-level `gsutil==5.37`
      after verifying it can list
      `gs://openpi-assets/checkpoints/pi05_base/params`. The first official
      train launch fell back to `gcsfs` because `gsutil` was absent and was
      interrupted after only about `233 MiB / 11.6 GiB`, with ETA drifting to
      `6-8` hours. This was a dependency/download bottleneck, not training.
- [x] Added Slurm-only skip-norm wrapper
      `scripts/slurm/run_openpi_pi05_peg733_train_in_allocation_skipnorm.sh`.
      It refuses login-node execution, requires the existing official
      `norm_stats.json`, records the same OpenPI method/weight boundaries, and
      then calls official `scripts/train.py`; it does not introduce any custom
      VAE/MLP/diffusion/intermediate model.
- [x] 2026-06-25 skip-norm wrapper runtime repair: default `WANDB_MODE` is now
      `offline` so official OpenPI `scripts/train.py` does not fail on
      no-tty W&B API-key prompts inside Slurm. This changes logging mode only;
      it does not change OpenPI model code, official weight loading, data, or
      optimizer behavior.
- [ ] 2026-06-25 `gsutil` skip-norm train launch after the weight-download
      repair:
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_1gpu1h_20260625_after_gsutil_weight_download_skipnorm_alloc150311`
      in Slurm step `150311.35`. It successfully skipped duplicate norm stats,
      loaded the existing OpenPI stats, initialized batches with image/state
      and action shapes, and started official `gsutil -m cp` download from
      `gs://openpi-assets/checkpoints/pi05_base/params`. Latest read-only
      accounting check: Slurm job `150311` is no longer in `squeue`; `sacct`
      reports the allocation as `CANCELLED by 0` after `04:19:22`, and the
      train step `150311.35` as `CANCELLED` after `03:09:42`.
      The final log tail was still in the official checkpoint download stage,
      around `11.1 GiB / 11.6 GiB`, `96%`, `18/20` files, with ETA roughly
      `11` to `16` minutes. At cancellation time the weight cache still
      contained `params.partial` and `params.lock`, not final `params`. The
      checkpoint output directory still contains only `wandb_id.txt`, with no
      train-step loss, optimizer state, model checkpoint, or
      `training_walltime_summary.json`.
      This was an interrupted official-checkpoint download, not a training
      result. Do not count any of the `04:19:22` allocation wall time toward
      the one-GPU-hour floor; optimization never reached official train-step,
      loss, or checkpoint evidence on the 733-derived OpenPI data.
- [x] 2026-06-25 login-node official-weight cache repair completed as a
      download-only action, not project compute: compared the interrupted
      `params.partial` contents against
      `gs://openpi-assets/checkpoints/pi05_base/params`, found the two large
      `.gstmp` shards had bad CRCs, redownloaded only those two official GCS
      objects, verified their local CRC32C values matched GCS
      (`JyiHIA==` and `gQSvJQ==`), then atomically moved `params.partial` to
      final `params` and removed the stale lock. Final cache check reports
      `PARAMS_DONE`, `20` files, and `12441721931` bytes, matching the
      official GCS object listing. This makes the next Slurm train launch able
      to restore official `pi05_base/params` directly instead of spending GPU
      allocation time on checkpoint download.
- [ ] 2026-06-25 follow-up dependency diagnosis: switching the wrapper from
      `uv run --locked` to `uv run --frozen` correctly bypassed lock freshness
      re-resolution and used the existing lock, but the wrapper still forced
      `--python-platform x86_64-unknown-linux-gnu`. That made uv target
      `manylinux_2_28_x86_64`, while the official locked dependency
      `rerun-sdk==0.23.1` only has `manylinux_2_31_x86_64` Linux wheels.
      The wrapper now leaves `UV_PYTHON_PLATFORM` unset by default so uv uses
      the compute node's native platform, and only passes `--python-platform`
      if explicitly requested.
- [ ] Allocation `149062` was revoked by the cluster immediately after this
      dependency diagnosis; no norm stats or training had started. A new
      tmux-held interactive request
      `openpi_pi05_peg733_1gpu_request_20260625` / Slurm job `149491` is
      queued for one GPU and currently remains `PENDING (Priority)`.
      When allocated, run:
      `STAMP=20260625_official_uv_frozen_native_after_lerobot_audit CONFIG_NAME=pi05_maniskill_peg733 MIN_WALL_SECONDS=3660 bash scripts/slurm/run_openpi_pi05_peg733_train_in_allocation.sh`.
- [ ] To avoid blocking on exactly one resource shape, a second tmux-held
      interactive request
      `openpi_pi05_peg733_2gpu_request_20260625` / Slurm job `149497` was
      also submitted. It is currently `PENDING (Priority)`. Use whichever
      allocation starts first for official OpenPI norm stats/training, and do
      not interpret pending resources as a model result.
- [ ] A third tmux-held interactive request
      `openpi_pi05_peg733_4gpu_request_20260625` / Slurm job `149520` was
      submitted for the same reason and is also `PENDING (Priority)`. Use the
      first available 1/2/4-GPU allocation; keep other pending requests from
      being mistaken for experiment progress.
- [ ] 2026-06-25 01:51 +08 queue check: all three OpenPI allocation requests
      remain pending for the same scheduler reason, `PENDING (Priority)`:
      `149491` 1-GPU, `149497` 2-GPU, and `149520` 4-GPU. The next real action
      is still to use the first allocated tmux shell to run official OpenPI
      norm stats/training with the audited 733 LeRobot data.
- [ ] 2026-06-25 resumed queue check: the first wave allocation requests
      (`149491`, `149497`, `149520`) had all reached nodes but were later
      revoked by the cluster before official OpenPI norm stats/training was
      launched. This is still a scheduling/allocation failure, not pi0.5 model
      evidence and not a data failure.
- [ ] 2026-06-25 replacement tmux-held interactive requests were submitted with
      the same resource shapes: `150292` 1-GPU
      (`openpi_pi05_peg733_1gpu_request_20260625_r2`), `150291` 2-GPU
      (`openpi_pi05_peg733_2gpu_request_20260625_r2`), and `150293` 4-GPU
      (`openpi_pi05_peg733_4gpu_request_20260625_r2`). All three are currently
      `PENDING (Priority)`. Prefer the 1-GPU allocation if it starts first;
      otherwise use the first available allocation without changing the
      official OpenPI/pi0.5 model path.
- [ ] 2026-06-25 13:02 +08: the replacement 1-GPU allocation `150292` started
      on `server60`, and official OpenPI norm stats/training was launched via a
      compute-node `srun` step:
      `srun --jobid=150292 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=120G bash scripts/slurm/run_openpi_pi05_peg733_train_in_allocation.sh`.
      Run directory:
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_1gpu1h_20260625_130200_alloc150292`.
      The manifest records `slurm_step_id=0`, `host=server60`,
      `uv_python_platform=native`, official OpenPI pi0.5 method boundary, and
      the official checkpoint-loader weight boundary. As of this note, the run
      is still in `uv run --frozen` official wheel download/environment
      creation before `compute_norm_stats.py` has produced dataset statistics;
      it is not yet training evidence.
- [x] 2026-06-25 resource hygiene: once the 1-GPU allocation `150292` started,
      the unused 2-GPU/4-GPU race allocations `150291` and `150293` were
      cancelled to avoid holding idle GPUs. The active run remains `150292`.
- [ ] 2026-06-25 dependency/runtime repair after `150292`: two official
      `uv run --frozen` attempts in the 1-GPU allocation were stopped before
      norm stats because dependency download/cache creation was too slow for
      GPU time. The first attempt used node-local `/tmp`; the second used the
      shared cache plus TUNA PyPI mirror. Neither reached
      `compute_norm_stats.py`, so neither is training evidence.
- [x] 2026-06-25 wrapper repair: default `UV_CACHE_DIR` now points to the
      persistent project cache
      `experiments/world_model_task_rebinding/openpi/uv_cache`, the manifest
      records `UV_INDEX_URL` / `UV_DEFAULT_INDEX`, and wrapper supports
      optional `UV_RUN_PYTHON` so official `uv run --frozen` can use a known
      good Python interpreter without changing OpenPI model code or weights.
- [x] 2026-06-25 dependency prewarm diagnosis: the original uv-managed
      `cpython-3.11.15` install was corrupted/missing stdlib `encodings`, which
      caused local LeRobot editable build isolation to fail. Python 3.12 was
      rejected for this path because `mujoco==2.3.7` did not use a cp312 wheel
      and fell back to source build requiring `MUJOCO_PATH`. The uv Python
      3.11.15 runtime was reinstalled with `uv python install 3.11.15
      --reinstall`, then verified to import `encodings`.
- [x] 2026-06-25 dependency prewarm complete: using repaired uv Python 3.11,
      `UV_LINK_MODE=copy`, `UV_LOCK_TIMEOUT=1800`, shared `uv_cache`, and TUNA
      PyPI mirror, `uv run --frozen --python ... python -c ...` completed on
      the login node as a download/cache-prewarm operation and installed
      `97` packages into
      `experiments/world_model_task_rebinding/openpi/uv_prewarm_env_py311_repaired`.
      This is environment readiness evidence only, not norm stats or training
      evidence.
- [x] 2026-06-25 post-prewarm allocation: a fresh 1-GPU tmux-held interactive
      request `openpi_pi05_peg733_1gpu_request_20260625_r3` / Slurm job
      `150311` was submitted after dependency prewarm and later allocated on
      `server38`. The intended launch command was:
      `STAMP=20260625_after_py311_repair_shared_cache CONFIG_NAME=pi05_maniskill_peg733 MIN_WALL_SECONDS=3660 UV_LINK_MODE=copy UV_LOCK_TIMEOUT=1800 UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple UV_RUN_PYTHON=/public/home/yanhongru/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11 srun --jobid=150311 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=120G bash scripts/slurm/run_openpi_pi05_peg733_train_in_allocation.sh`.
- [ ] 2026-06-25 13:57 +08: allocation `150311` started on `server38` and the
      official wrapper was launched in Slurm step `150311.0`. Run directory:
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_1gpu1h_20260625_after_py311_repair_shared_cache_alloc150311`.
      The manifest records `host=server38`, `slurm_step_id=0`, repaired
      `UV_RUN_PYTHON`, shared `uv_cache`, TUNA index URLs, official OpenPI
      pi0.5 method boundary, and official checkpoint-loader weight boundary.
      Dependency install reused the cache and completed quickly:
      `Installed 242 packages in 24.99s`.
- [ ] 2026-06-25 13:58 +08: official `scripts/compute_norm_stats.py
      --config-name pi05_maniskill_peg733` is running inside the compute-node
      step. A light process probe on `server38` showed the actual Python
      process active with substantial CPU use. This proves the run passed the
      dependency/download blocker and entered official norm-stat computation,
      but it is still not pi0.5 training evidence until `scripts/train.py`
      starts and runs for the required wall time.
- [x] 2026-06-25 norm-stats blocker diagnosis: the apparent hang before
      `Computing stats` was not a model issue. A staged compute-node diagnostic
      showed `LeRobotDatasetMetadata` loaded the local 733 dataset successfully,
      but `datasets.load_dataset("parquet")` failed on the parquet footer
      HuggingFace feature metadata with
      `ValueError: Feature type 'List' not found`. The physical Arrow schema
      was valid fixed-size-list state/action data; only the HF metadata type
      name was incompatible with the installed `datasets` version.
- [x] 2026-06-25 data metadata repair: added
      `scripts/openpi/repair_lerobot_parquet_hf_sequence_metadata.py` and ran
      it inside allocation `150311`. It rewrote 733/733 parquet footers from
      legacy `_type=List` to `_type=Sequence` for `state` and `actions` only.
      Manifest:
      `experiments/world_model_task_rebinding/openpi/lerobot_home/yanhongru/maniskill_peg733_openpi_libero/maniskill_peg733_hf_sequence_metadata_repair_manifest.json`.
      Numeric data, image data, OpenPI model code, official checkpoint loader,
      and weight paths were not changed.
- [x] 2026-06-25 post-repair HF load verification: a compute-node
      `datasets.load_dataset("parquet")` verification loaded all `219900` rows
      and reported `state=Sequence(length=8)` and `actions=Sequence(length=7)`.
- [x] 2026-06-25 official norm stats after repair: wrapper run
      `pi05_peg733_1gpu1h_20260625_after_hf_sequence_metadata_repair_alloc150311`
      completed official OpenPI norm stats in allocation `150311`: the log
      reached `3435/3435` batches and wrote
      `/public/home/yanhongru/ICLR2027/openpi/assets/pi05_maniskill_peg733/yanhongru/maniskill_peg733_openpi_libero/norm_stats.json`.
      This is data-normalization evidence, not pi0.5 training evidence.
- [x] 2026-06-25 replacement r4 tmux-held allocation requests after official
      weight-cache repair: `150773` 1-GPU
      (`openpi_pi05_peg733_1gpu_request_20260625_r4`), `150774` 2-GPU
      (`openpi_pi05_peg733_2gpu_request_20260625_r4`), and `150775` 4-GPU
      (`openpi_pi05_peg733_4gpu_request_20260625_r4`) were submitted. The
      1-GPU allocation `150773` started on `server38`; unused requests
      `150774` and `150775` were cancelled for resource hygiene.
- [x] 2026-06-25 first formal OpenPI pi0.5 run after W&B/cache repair:
      allocation `150773` is running on `server38`; unused requests `150774`
      and `150775` were cancelled for resource hygiene. Current run directory:
      `experiments/world_model_task_rebinding/openpi/pi05_peg733_1gpu1h_20260625_after_wandb_offline_and_params_cache_repair_skipnorm_alloc150773`.
      Evidence: official wrapper skipped existing norm stats, W&B is
      offline, the full `733`-episode / `219900`-row HF train split generated,
      data loader initialized with RGB/state/action tensors
      `(64, 224, 224, 3)`, `(64, 32)`, and `(64, 16, 32)`, official
      `pi05_base/params` restored from local OpenPI cache in `8.34` seconds,
      train state initialized with OpenPI/PaliGemma/pi0.5 parameter tree, and
      progress reached about `1.08kit/20.0kit`. This was real optimization
      evidence, but not a completed saved model because checkpoint save failed
      on the NFS/project path.
- [x] 2026-06-25 NFS checkpoint blocker in the first active run:
      at step `1000`, Orbax/TensorStore attempted to write checkpoint
      `1000` under the project/NFS checkpoint directory and failed with
      `ENOLCK No locks available` while acquiring a local file lock for
      `manifest.ocdbt.__lock`. The run continued to about `1.08kit/20.0kit`,
      but only an incomplete `1000.orbax-checkpoint-tmp-0` directory exists,
      so it is not a usable checkpoint and must not be treated as a saved
      model. The step was interrupted inside the held allocation and the
      allocation was preserved for an aligned retry.
- [x] 2026-06-25 local-checkpoint retry in the same allocation:
      relaunched official OpenPI `scripts/train.py` in Slurm job `150773` with
      the same 733 data, same official `pi05_base/params` weights, and
      `CHECKPOINT_BASE_DIR=/tmp/openpi_pi05_checkpoints_yanhongru_150773_local`
      to avoid NFS file locks. Runtime overrides are official TrainConfig CLI
      fields only: `--num-train-steps=1600 --save-interval=800`, intended to
      produce a valid local checkpoint while running for roughly one GPU-hour.
      Evidence: CheckpointManager root is the `/tmp` local checkpoint
      path, W&B is offline, norm stats and data config loaded, and
      `local_batch_size: 64` was reached. The run restored official
      `pi05_base/params` in `8.71` seconds, initialized train state, and
      entered training. The first local checkpoint save at step `800`
      finalized successfully to
      `/tmp/openpi_pi05_checkpoints_yanhongru_150773_local/pi05_maniskill_peg733/pi05_peg733_1gpu1h_20260625_after_nfs_enolck_localckpt_1600_skipnorm_alloc150773/800`
      with `No errors found in background save thread`; final checkpoint
      `1599` also finalized successfully with `No errors found in background
      save thread` and `CheckpointManager Save Finalize is done on all hosts`.
      Slurm step `150773.2` completed with exit code `0:0` after `01:09:13`;
      `training_walltime_summary.json` records `elapsed_seconds=4152` and
      `formal_one_gpu_hour_floor_met=true`. The preserved checkpoint under the
      active experiment tree has `57` files and `44697898725` bytes, matching
      the compute-node local source exactly.
