# OpenPI/pi0.5 Active Status - 2026-06-26

## Boundary

Active action-model line is official OpenPI/pi0.5, not frozen DP, scorer-only
selection, or any in-repo placeholder action model. All training/eval/data
compute must run inside tmux-held interactive Slurm allocations.

## Established Evidence

- The previous qpos8 contact-suffix pi0.5 checkpoint trained for one GPU-hour
  from official `pi05_base`, but replay was negative for insertion: grasp held
  in the inspected panel, insertion success was 0/4. This is a baseline/negative
  diagnostic, not method success.
- The likely physical reason is not "Cosmos cannot imagine insertion" alone:
  qpos8 does not expose the moved target/task-frame geometry. A policy asked to
  insert from dynamic contact states needs current object/task-frame binding.
- Clean qpos8 and object17 LeRobot repo ids were rebuilt on 2026-06-26 after the
  hardlink-contamination issue:
  `yanhongru/maniskill_peg733_openpi_contact_suffix16_qpos8_clean_20260626`
  and
  `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_clean_20260626`.
  Both structural audits passed: 5853 suffix episodes, 93648 rows, 16-step
  windows, 733 source episodes.
- Clean object17 norm stats were computed and installed under the official
  OpenPI assets tree. State dim is 17, action dim is 7.

## Resolved Data-Repair Gates

The clean object17 repair has advanced beyond the earlier hardlink-contamination
state.

- Fresh noncanonical qpos8 and object17 repo ids were rebuilt and audited:
  `yanhongru/maniskill_peg733_openpi_contact_suffix16_qpos8_clean_20260626`
  and
  `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_clean_20260626`.
- The image-backed object17 repo is structurally valid, but official
  `LeRobotDataset(repo_id)` construction hung before `dataset[0]`. The likely
  cause is the storage layout: two `256x256` RGB streams stored as
  `dtype=image` columns across `5853` per-episode parquet files, about `18 GB`
  of image payload.
- To repair that bottleneck without changing model family or weights, a
  video-backed clean object17 repo was built:
  `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_clean_20260626`.
- H.264 video-backed conversion completed in allocation `152622` with
  `object17_video_h264_convert_rc=0`.
- Strict audit passed in allocation `152622`:
  `experiments/world_model_task_rebinding/openpi/object17_video_clean_audit_20260626_alloc152622/audit_summary.json`.
  It reports `passed=true`, `5853` parquet files, `11706` mp4 files,
  `93648` rows, `unique_episode_lengths=[16]`, state dim `17`, action dim `7`,
  `info_total_episodes=5853`, and `info_total_frames=93648`.
- OpenPI-format norm stats were computed for the video-backed repo and written
  to:
  `/public/home/yanhongru/ICLR2027/openpi/assets/pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626/yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_clean_20260626/norm_stats.json`.
- Official Hugging Face/LeRobot split generation now reaches all `93648`
  examples. This is a real positive signal versus the image-backed constructor
  hang, but it is not a training result.

## Loader Diagnosis Update

The first interpretation of the video-backed blocker was too broad. The
failure is not H.264/pyav decode itself.

A Slurm-only staged diagnostic in allocation `152622` found:

- metadata load succeeds;
- direct pyav decode of episode `0` external image mp4 succeeds;
- direct pyav decode of episode `0` wrist mp4 succeeds;
- direct pyav decode of all `16` frames from one mp4 succeeds in about
  `0.033 s`;
- `LeRobotDataset(..., episodes=[0])` with the default shared Hugging Face
  datasets cache hangs after printing `Generating train split: 16 examples`;
- a manual constructor-tail diagnostic also hangs inside
  `datasets.load_dataset(...)` after the split is generated, before timestamp
  checks or `dataset[0]`.

When `HF_HOME`, `HF_DATASETS_CACHE`, and `XDG_CACHE_HOME` are redirected to
compute-node `/tmp`, the same single-episode constructor and item retrieval
pass:

- diagnostic root:
  `experiments/world_model_task_rebinding/openpi/object17_video_hf_tmpcache_debug_20260626_alloc152622`;
- `manual_one_tmpcache_rc=0`;
- `dataset_one_nodelta_tmpcache_rc=0`;
- `dataset_one_delta_tmpcache_rc=0`;
- `dataset_construct_seconds` about `2.9..3.3 s`;
- `item_seconds` about `0.05..0.06 s`.

The current root cause is therefore the Hugging Face datasets cache/finalize
path on shared storage, not the mp4 encoding. The OpenPI Slurm wrappers were
patched to default `HF_HOME`, `HF_DATASETS_CACHE`, and `XDG_CACHE_HOME` to
compute-node `/tmp`.

## First-Batch Gate

With `/tmp` HF cache, the full patched pretraining loader gate now passes on
the video-backed repo:

- debug root:
  `experiments/world_model_task_rebinding/openpi/object17_video_clean_tmpcache_postdebug_patched_20260626_alloc153455`;
- `first_batch_rcs.txt`: `raw_item_rc=0`, `transformed_item_rc=0`,
  `first_batch_rc=0`;
- `postrebuild_debug_summary.json`: `passed=true`;
- transformed item has `224x224` model images, padded state dim `32`, and
  action chunk `16x32`;
- first batch has batch size `16`, three image streams, state `16x32`, and
  actions `16x16x32`.

This clears the data/backend/training-input gate for object17-video OpenPI. It
does not prove task performance.

## Training Result

Formal official OpenPI/pi0.5 training completed in allocation `153455` on
`server60`:

`experiments/world_model_task_rebinding/openpi/pi05_object17_video_clean_direct1700_1gpu1h_20260626_alloc153455`

Boundaries:

- config:
  `pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626`;
- model/weights: official `Pi0Config(pi05=True, action_horizon=16,
  discrete_state_input=False)` with official `pi05_base` checkpoint restore;
- data: audited 733-derived video-backed object17 contact-suffix LeRobot repo;
- no custom VAE, MLP, diffusion executor, or scorer-only selector;
- checkpoint base: server-local `/tmp`, then copied to project preserved
  checkpoint root.

Evidence:

- `training_walltime_summary.json`: `elapsed_seconds=4748`,
  `formal_one_gpu_hour_floor_met=true`, `train_return_code=0`;
- `tmux_driver_rc.txt`: `object17_video_direct_train_rc=0`;
- checkpoint finalized at step `1699`;
- preserved checkpoint:
  `experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626/pi05_object17_video_clean_direct1700_1gpu1h_20260626_alloc153455/1699`;
- copy summary: source and destination both `31G`, preserved checkpoint has
  `87` files;
- train log: step `0` loss `0.0950`; step `1600` loss `0.0319`.

This is now a valid object17-video OpenPI/pi0.5 training result, but training
loss is not task success evidence.

## Replay Result

Saved-snapshot replay with matching 17D causal object/task state preparation
completed in allocation `153455`:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_direct1700_replay4_exec16_fixhistory_alloc153455`

Important implementation repair: old live-receding `live_history_raw_action_state`
files store action first, then task fields in columns `7:24`. The snapshot
preparer now maps that layout to the same object17 state used in training:
`tcp, peg, hole, peg_head_at_hole, hole_velocity, grasped, inserted`. This is
glue/state-layout repair only; no model or action generator was changed.

Summary:

- wrapper return code: `0`;
- prepared observations: `4`;
- OpenPI action chunks: `4`, executed for `16` steps each;
- direct success: `0/4`;
- direct inserted: `0/4`;
- direct contact-stable: `0/4`;
- grasp preserved after OpenPI chunk: `4/4`;
- DP96 historical continuability/success after OpenPI chunk: `3/4`.

Contact-state sheets were generated in Slurm:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_direct1700_replay4_exec16_fixhistory_alloc153455/contact_state_sheets/contact_state_sheets_manifest.json`

They confirm all four OpenPI chunks keep grasp for `16/16` executed steps, but
none contains an inserted step during the OpenPI chunk. Three chunks leave a
state from which the historical frozen DP96 rollout can finish. This is a
stronger handoff/continuability signal than the earlier qpos8 contact-suffix
run, but it is still not OpenPI direct task completion.

## Receding Diagnostic

A split-env receding OpenPI diagnostic was added:

- `scripts/openpi/run_openpi_pi05_receding_snapshot_rollout.py`;
- `scripts/slurm/run_openpi_pi05_receding_snapshot_rollout_in_allocation.sh`.

The project Python process owns ManiSkill execution, while each OpenPI query
is delegated to the official OpenPI environment through the existing
`infer_openpi_pi05_from_prepared_observations.py` helper. This adds no custom
policy model, VAE, MLP, diffusion executor, or scorer. By default it reuses the
saved observed-prefix image and refreshes privileged simulator object17 state;
therefore it is an upper-bound diagnostic, not final RGB-derived evidence.

One smoke run completed in allocation `153455`:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_receding1_q3_exec4_alloc153455`

Settings/result:

- sample: `sample_00_hole_late_move_stop`, prefix frame `106`;
- `3` OpenPI queries, `4` executed steps per query, `12` total OpenPI steps;
- direct success `0/1`;
- inserted `0/1`;
- contact-stable `0/1`;
- grasp preserved `1/1`;
- final peg-head-at-hole `[-0.2088, 0.1156, -0.0186]`;
- `abs(y)+abs(z)` worsened from `0.0539` to `0.1341`.

Contact-state sheet:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_receding1_q3_exec4_alloc153455/contact_state_sheets/contact_state_sheets_manifest.json`

This small receding smoke shows that simply refreshing object17 state every
four actions does not fix the first tested hard snapshot. It should not be
overgeneralized, but it weakens the hypothesis that the failure is only
single-chunk open-loop drift.

## Current Conclusion

Important boundary: the OpenPI/pi0.5 maintenance/training objective has been
completed, but the research objective has not. The system still has no direct
dynamic insertion success from the current OpenPI action model, and this file
must not be read as claiming task completion.

Visual progress dashboard:
`docs/world_model_task_rebinding/openpi_pi05_contact_action/2026-06-26_progress_visual_dashboard.md`.

Object/task-frame conditioning helped the action prior reach DP-continuable
states more often, but it still does not generate the final insertion/contact
motion by itself on the tested dynamic snapshots. The remaining blocker is
direct contact/insertion execution, not official-weight loading, LeRobot video
decode, norm stats, or first-batch training input.

The current replay diagnostics do not blame Cosmos/world-model action
generation, because Cosmos is not the action generator in these tests. The
saved-snapshot replay isolates the action executor: it gives OpenPI the current
snapshot observation/state, runs official OpenPI inference, and executes the
predicted action chunk in ManiSkill. Since even privileged object17 state plus
official OpenPI finetuning stays at direct inserted `0/4`, the immediate
blocker is the action model/executor's contact-mode behavior, not a world model
failing to imagine the right action.

Concrete failure pattern:

- actions stay within the simulator action space and preserve grasp;
- no evaluated chunk contains an inserted step;
- several chunks worsen lateral `abs(y)+abs(z)` relative to the current hole
  frame;
- naive receding with refreshed object17 state also failed on the first hard
  snapshot;
- near-contact reweighting reduced DP96 handoff from `3/4` to `1/4`.

Therefore the unresolved issue is likely an action-coordinate/contact-mode
mismatch: the policy can keep holding and approach, but it has not learned a
robust final insertion correction in the robot action coordinates from these
dynamic takeover states.

Next work should keep the official OpenPI/pi0.5 path and focus on one of:

1. stronger contact-target supervision/action windows closer to insertion;
2. a broader receding panel only if it changes the query/execution contract
   beyond the small q3x4 smoke above;
3. replacing privileged object17 slots with causal RGB-derived task slots if
   object17 continues to look useful as an upper bound.

Do not return to scorer-only selection or custom VAE/MLP/diffusion executors as
the main method.

## Near-Contact Branch Result

The next concrete OpenPI-native repair was a near-contact object17/video
dataset/config branch:

- config:
  `pi05_maniskill_peg733_contact_suffix16_object17_video_nearcontact_20260626`;
- repo id:
  `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_nearcontact_20260626`;
- model/weights: official `Pi0Config(pi05=True, action_horizon=16,
  discrete_state_input=False)` and official `pi05_base` restore;
- intended conversion: same accepted 733 source and video-backed LeRobot path
  as object17-video, but with insertion-close offsets such as
  `16,12,8,4,2,1` instead of the old far-heavy `64,48,32,24,16,12,8,4`;
- Slurm-only preparation wrapper:
  `scripts/slurm/run_openpi_pi05_nearcontact_object17_video_prepare_in_allocation.sh`;
- reason: offset `1` is the closest pre-insertion causal action target, while
  offset `0` would already observe the first inserted frame and is therefore
  less useful for learning the act of insertion.

This branch changes only the data window distribution and official OpenPI
config/repo id. It does not add a scorer, DP controller, custom VAE, MLP,
diffusion executor, or non-OpenPI policy model.

Preparation result:

`experiments/world_model_task_rebinding/openpi/object17_video_nearcontact_prepare_20260626_offsets16_12_8_4_2_1_alloc153455`

- conversion/audit/norm stats completed in allocation `153455`;
- audit `passed=true`;
- `733` source episodes;
- `4375` suffix episodes, `70000` rows, `8750` mp4 files;
- state dim `17`, action dim `7`;
- norm stats installed under the OpenPI assets tree.

Training result:

`experiments/world_model_task_rebinding/openpi/pi05_object17_video_nearcontact_direct1700_1gpu1h_pyav_20260626_alloc153455`

- official config
  `pi05_maniskill_peg733_contact_suffix16_object17_video_nearcontact_20260626`;
- official `Pi0Config(pi05=True, action_horizon=16,
  discrete_state_input=False)`;
- official `pi05_base` restore;
- `training_walltime_summary.json`: `elapsed_seconds=5044`,
  `formal_one_gpu_hour_floor_met=true`, `train_return_code=0`;
- final checkpoint `1699` preserved at:
  `experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733_contact_suffix16_object17_video_nearcontact_20260626/pi05_object17_video_nearcontact_direct1700_1gpu1h_pyav_20260626_alloc153455/1699`.

Replay result:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_nearcontact_direct1700_replay4_exec16_alloc153455`

- wrapper return code `0`;
- prepared/action chunks `4`;
- executed `16` OpenPI actions per sample;
- direct success `0/4`;
- direct inserted `0/4`;
- direct contact-stable `0/4`;
- grasp preserved `4/4`;
- DP96 historical continuability/success `1/4`.

Contact-state sheets:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_nearcontact_direct1700_replay4_exec16_alloc153455/contact_state_sheets/contact_state_sheets_manifest.json`

The sheets confirm all four chunks preserve grasp for `16/16` steps but none
contains an inserted step. Compared with the previous clean object17-video
checkpoint, near-contact window reweighting did not solve direct insertion and
weakened historical DP96 handoff from `3/4` to `1/4`. The blocker is therefore
not just that the old suffix dataset had too many far approach windows.
