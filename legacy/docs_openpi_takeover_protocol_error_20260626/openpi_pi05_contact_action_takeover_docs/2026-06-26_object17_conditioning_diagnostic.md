# 2026-06-26 OpenPI pi0.5 Object17 Conditioning Diagnostic

## Summary

The qpos8 contact-suffix checkpoint proved that official OpenPI/pi0.5 can
train on the 733-derived insertion suffix data and preserve grasp, but it still
does not insert from saved dynamic snapshots. The object17 branch is the next
OpenPI-native diagnostic: expose current object/task-frame geometry to the
same official pi0.5 model family and test whether binding to the moved hole is
the missing piece.

This is data/config evidence only so far. It is not a trained model result.

## Dataset

- Source dataset: `yanhongru/maniskill_peg733_openpi_contact_suffix16`.
- Rewritten repo id:
  `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17`.
- Dataset root:
  `experiments/world_model_task_rebinding/openpi/lerobot_home/yanhongru/maniskill_peg733_openpi_contact_suffix16_object17`.
- Rewrite evidence:
  `experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix_lerobot_object17_rewrite_20260626_alloc150773/conversion_manifest.json`.
- Audit evidence:
  `experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix_lerobot_object17_rewrite_20260626_alloc150773/audit_summary.json`.
- Count: `5853` suffix episodes, `93648` rows, `16` rows per episode.
- Images: external and duplicated wrist images, `256x256x3`.
- Actions: original 7D ManiSkill `pd_ee_delta_pose` suffix actions.

The 17D state is:

`tcp_pose3, peg_pose3, hole_pose3, peg_head_at_hole3, hole_velocity_step3, grasped, inserted`

The audit passed with `failures=[]`, state dim `17`, action dim `7`,
`record_start_grasped_count=5780`, and `record_end_inserted_count=2906`.

## Method Boundary

This branch keeps the model official:

- config: `pi05_maniskill_peg733_contact_suffix16_object17`;
- model: official `Pi0Config(pi05=True, action_horizon=16,
  discrete_state_input=False)`;
- data config: official `LeRobotLiberoDataConfig`;
- weights: official `pi05_base`;
- no custom VAE, MLP, diffusion executor, or scorer-only action selection.

The object17 slots are simulator/source task-state slots. They are allowed only
as an upper-bound diagnostic for object binding. If this branch succeeds, the
publishable path still needs RGB-derived peg/hole/TCP/task-frame perception to
produce equivalent causal slots.

## Norm Stats

Official `scripts/compute_norm_stats.py` repeatedly stalled after Hugging Face
split generation on the object17 rewritten dataset. To avoid changing the
model while unblocking data normalization, a Slurm-only fallback script was
used:

`scripts/openpi/compute_lerobot_state_action_norm_stats.py`

It reads LeRobot parquet `state/actions` and writes OpenPI-format stats using
OpenPI `normalize.RunningStats` and `normalize.save`.

Evidence:

`experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix16_object17_norm_fallback_20260626_alloc150773/norm_stats_fallback_summary.json`

Summary: `93648` rows, state dim `17`, action dim `7`, Slurm allocation
`150773`, step `85`.

## Training Attempt

A direct official OpenPI training launch bypassed the wrapper's norm-stats
stage and called `scripts/train.py` directly:

`experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix16_object17_direct1700_skipnorm_1gpu1h_20260626_alloc150773`

Evidence:

- `run_manifest.txt`;
- `train.log`;
- `tmux_driver.log`.

Observed progress:

- ran on `server38`, Slurm step `150773.87`;
- loaded object17 norm stats from the OpenPI assets directory;
- printed the expected 17D state and 7D action stats in `data_config`;
- generated the full `93648`-example train split;
- allocated the pi0.5 model on GPU.

Initial blocker:

No `Step 0` training log appeared after roughly 26 minutes. A read-only
process/GPU inspection during the run showed high GPU memory allocation but
zero useful training progress. The step was cancelled to avoid idle GPU burn;
`tmux_driver.log` records return code `137`.

Therefore this is not a one-GPU-hour training result and not insertion
evidence.

## Data Contamination Found Later

After the first-batch investigation, the object17 rewrite path was found to be
unsafe. `scripts/openpi/rewrite_contact_suffix_lerobot_object_state.py` cloned
the qpos8 contact-suffix LeRobot repo with `cp -al` hardlinks before editing
`meta/info.json` and parquet `state` columns. Because those files were
hardlinked, the canonical qpos8 repo was also mutated to object17
metadata/state.

This invalidates the object17 data tree as clean evidence. The earlier
first-step stall may still describe a real LeRobot loading failure, but it was
observed on a contaminated data state and must not be interpreted as a clean
OpenPI training result.

First-batch diagnostic that led to this:

- `scripts/openpi/debug_openpi_lerobot_first_batch.py` was added as a
  Slurm-only diagnostic using official OpenPI `create_data_loader`.
- With `num_workers=0`, `first_batch` generated the full `93648`-example split
  and then timed out before returning a batch.
- A narrower `raw_item` mode also generated the split and timed out before
  returning `LeRobotDataset[0]`.
- This moved suspicion below OpenPI training/weight initialization and led to
  inspecting the LeRobot data tree, where the hardlink contamination was found.

Because the data tree was contaminated, these timeouts are diagnostic
breadcrumbs only, not clean evidence about official OpenPI behavior.

Repair attempt:

- `rewrite_contact_suffix_lerobot_object_state.py` now uses real
  `shutil.copytree`, not hardlinks.
- Contact-suffix conversion wrappers now refuse destructive overwrite of the
  canonical repo id unless explicitly acknowledged.
- A qpos8 rebuild was launched in allocation `150773`, step `92`, but Slurm
  cancelled it at about `97%` when the allocation entered `COMPLETING`.
- Current canonical qpos8 metadata is partial: `5704` episodes / `91264`
  frames, state dim `8`.

Therefore both the canonical qpos8 repo and the old object17 repo must be
treated as unusable until rebuilt and audited from clean source data.

## Clean Rebuild And Video-Backend Repair

Later on 2026-06-26, the clean-data repair moved to fresh noncanonical repo ids.
This keeps the OpenPI/pi0.5 model path unchanged and avoids reusing either the
partial canonical qpos8 repo or the hardlink-derived object17 repo.

Clean rebuilt repo ids:

- `yanhongru/maniskill_peg733_openpi_contact_suffix16_qpos8_clean_20260626`;
- `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_clean_20260626`;
- `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_clean_20260626`.

The image-backed clean object17 repo audited structurally but was not usable as
a training input: official `LeRobotDataset(repo_id)` hung before returning
`dataset[0]`. The likely bottleneck is storage layout. The repo stores two
`256x256` RGB streams as embedded `dtype=image` columns in `5853` per-episode
parquet files, producing roughly `18 GB` of image payload.

The active repair was to write the same object17 suffix data with LeRobot
`dtype=video` camera features. The H.264 video-backed conversion completed in
allocation `152622`:

`experiments/world_model_task_rebinding/openpi/object17_video_clean_rebuild_h264_20260626_alloc152622/convert_driver_rc.txt`

The strict audit passed:

`experiments/world_model_task_rebinding/openpi/object17_video_clean_audit_20260626_alloc152622/audit_summary.json`

Audit summary:

- `passed=true`;
- `5853` parquet files;
- `11706` mp4 files;
- `93648` total rows;
- `unique_episode_lengths=[16]`;
- state dim `17`, action dim `7`;
- camera dtypes `image=video`, `wrist_image=video`;
- `info_total_episodes=5853`, `info_total_frames=93648`.

OpenPI-format norm stats were also computed for the video-backed config:

`/public/home/yanhongru/ICLR2027/openpi/assets/pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626/yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_clean_20260626/norm_stats.json`

This is positive infrastructure evidence: the repaired video-backed repo avoids
the image-backed Hugging Face split bottleneck and can generate the full
`93648`-example train split.

The later loader diagnosis narrowed the failure. The mp4 files are decodable:
metadata, external image video, wrist image video, and all `16` frames from the
first episode decode successfully with pyav. The observed timeout came from
Hugging Face `datasets.load_dataset` using the default shared cache. Even a
single-episode load generated the 16-row split and then did not return before
the timeout.

With cache variables redirected to compute-node `/tmp`, the same
single-episode constructor and item access pass:

`experiments/world_model_task_rebinding/openpi/object17_video_hf_tmpcache_debug_20260626_alloc152622`

Evidence:

- `manual_one_tmpcache_rc=0`;
- `dataset_one_nodelta_tmpcache_rc=0`;
- `dataset_one_delta_tmpcache_rc=0`;
- item retrieval about `0.05..0.06 s`.

This means the current storage repair does not require re-encoding video. It
requires running OpenPI/LeRobot with `HF_HOME`, `HF_DATASETS_CACHE`, and
`XDG_CACHE_HOME` on compute-node local `/tmp`, matching the existing `uv`
environment discipline.

A full raw-item gate with `/tmp` HF cache also passed:

`experiments/world_model_task_rebinding/openpi/object17_video_clean_tmpcache_postdebug_20260626_alloc152622`

Evidence:

- `raw_item_rc=0`;
- `create_dataset_seconds=37.52`;
- `item_seconds=0.10`;
- image and wrist image tensors `3x256x256`;
- state dim `17`;
- action chunk `16x7`.

The remaining gate failure is transformed item / first batch. The first
tmp-cache full gate failed with `transformed_item_rc=1` because norm stats were
not found. This was not a missing stats file; it was a working-directory bug:
the debug wrapper used `OPENPI_PYTHON` without changing directory to the OpenPI
root, so OpenPI resolved relative `assets/` under the Reflex repo. The wrapper
and `debug_openpi_lerobot_first_batch.py` have been patched to load config
assets from the OpenPI root. Allocation `152622` expired before this patched
full gate could rerun.

Therefore object17 video-backed data are clean and audited, and raw item access
is now unblocked with `/tmp` HF cache. The first transformed-item/first-batch
gate still needs one more Slurm rerun before training. No object17 pi0.5
training result exists yet.

## Current Conclusion

Positive signals:

- The accepted 733 data do contain insertion suffix actions.
- Official OpenPI/pi0.5 qpos8 suffix training is valid and preserves grasp.
- The qpos8 replay failure is geometrically specific: actions often worsen
  hole-frame lateral error instead of entering contact.
- Clean object17 data/config/norm stats now directly target that object-binding
  failure while keeping OpenPI official.
- The video-backed object17 repair passed strict structural audit and full
  `93648`-example split generation.
- Direct pyav video decode and raw item access are now proven viable when HF
  datasets cache is moved to compute-node `/tmp`.

Current blocker:

Object17 training is blocked by the final transformed-item/first-batch gate,
not by video decode and not by model weights. A wrapper working-directory fix
has been applied after the `Normalization stats not found` failure, but the
patched gate still needs to be rerun in a valid Slurm allocation.

Next required gates:

1. Acquire/use a non-curiosity tmux-held Slurm allocation.
2. Rerun the patched post-rebuild debug wrapper with `/tmp` HF cache and valid
   OpenPI Python until raw item, transformed item, and first batch return.
3. Make object17 official OpenPI training reach `Step 0`.
4. Run at least one GPU-hour before interpreting training.
5. Replay the resulting checkpoint from saved dynamic snapshots with matching
   17D causal state preparation.
6. If object17 improves insertion, replace simulator task slots with
   RGB-derived object/task-frame perception for method evidence.

## Final Update From Allocation 153455

The remaining gate and training steps are now complete.

First-batch gate:

`experiments/world_model_task_rebinding/openpi/object17_video_clean_tmpcache_postdebug_patched_20260626_alloc153455`

Evidence:

- `raw_item_rc=0`;
- `transformed_item_rc=0`;
- `first_batch_rc=0`;
- transformed item has padded state dim `32`, action chunk `16x32`, and
  `224x224` model images;
- first batch has batch size `16`, three image streams, state `16x32`, and
  actions `16x16x32`.

Formal training:

`experiments/world_model_task_rebinding/openpi/pi05_object17_video_clean_direct1700_1gpu1h_20260626_alloc153455`

Evidence:

- official config
  `pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626`;
- official `Pi0Config(pi05=True, action_horizon=16,
  discrete_state_input=False)`;
- official `pi05_base` restore;
- audited 733-derived video-backed object17 LeRobot repo;
- `training_walltime_summary.json`: `elapsed_seconds=4748`,
  `formal_one_gpu_hour_floor_met=true`, `train_return_code=0`;
- final checkpoint `1699` preserved at:
  `experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626/pi05_object17_video_clean_direct1700_1gpu1h_20260626_alloc153455/1699`.

Replay preparation repair:

`scripts/openpi/prepare_openpi_pi05_snapshot_observations.py` now supports both
the 35-column executor-state layout and the older action-prefixed
`live_history_raw_action_state` layout. The replay panel used here stores
action first, then task fields in columns `7:24`, so object17 state is mapped
as `tcp=7:10`, `peg=10:13`, `hole=13:16`, `peg_head_at_hole=16:19`,
`hole_velocity=19:22`, `grasped/inserted=22:24`. This was validated by the
successful replay below.

Saved-snapshot replay:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_direct1700_replay4_exec16_fixhistory_alloc153455`

Summary:

- wrapper return code `0`;
- prepared observations `4`;
- OpenPI action chunks `4`;
- executed `16` OpenPI actions per sample;
- direct success `0/4`;
- direct inserted `0/4`;
- direct contact-stable `0/4`;
- grasp preserved `4/4`;
- DP96 historical continuability/success `3/4`.

Contact-state sheet evidence:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_direct1700_replay4_exec16_fixhistory_alloc153455/contact_state_sheets/contact_state_sheets_manifest.json`

The sheets confirm that all four OpenPI chunks preserve grasp for `16/16`
steps, but none contains an inserted step. Three chunks leave states from
which the historical DP96 rollout can finish. Therefore object17 conditioning
improves handoff/continuability versus the qpos8 suffix result, but it still
does not solve direct insertion.

Updated conclusion:

The object17-video OpenPI/pi0.5 branch is now a valid trained-result diagnostic,
not just data/config evidence. It shows a real positive signal for
grasp-preserving object-conditioned action generation, but the blocker has
moved to direct contact/insertion execution.

## Receding Smoke Update

A first split-env receding diagnostic was implemented and run after the direct
replay. The project-side process owns ManiSkill execution, and each action
query is delegated to the official OpenPI environment through the existing
inference helper. This adds no custom policy model, scorer, VAE, MLP, or
diffusion executor.

Scripts:

- `scripts/openpi/run_openpi_pi05_receding_snapshot_rollout.py`;
- `scripts/slurm/run_openpi_pi05_receding_snapshot_rollout_in_allocation.sh`.

Run root:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_receding1_q3_exec4_alloc153455`

Settings:

- sample: `sample_00_hole_late_move_stop`, prefix `106`;
- `3` OpenPI queries;
- `4` executed steps per query;
- `12` total OpenPI steps;
- refreshed simulator object17 state between queries;
- static observed-prefix image, so this is an upper-bound diagnostic rather
  than final RGB-derived method evidence.

Summary:

- wrapper return code `0`;
- direct success `0/1`;
- direct inserted `0/1`;
- direct contact-stable `0/1`;
- grasp preserved `1/1`;
- `abs(y)+abs(z)` worsened from `0.0539` to `0.1341`.

Contact-state sheet evidence:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_receding1_q3_exec4_alloc153455/contact_state_sheets/contact_state_sheets_manifest.json`

The sheet confirms grasp is preserved for all `12` executed steps, but there is
no inserted step. This weakens the hypothesis that direct insertion fails only
because a single 16-step chunk is open-loop. The stronger current hypothesis is
that the trained action distribution still lacks a reliable insertion-axis
contact mode from these dynamic snapshots.

Updated next direction:

Keep OpenPI/pi0.5 as the action model and shift the next repair toward stronger
near-contact insertion supervision or contact-mode action-distribution repair,
using official OpenPI configs, fresh norm stats, and the accepted 733-derived
data. A broader receding panel is useful only if the query/execution contract
changes meaningfully. Do not fall back to DP as the main method, scorer-only
selection, or any homemade intermediate action model.
