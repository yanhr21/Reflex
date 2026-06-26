# OpenPI pi0.5 Contact-Action Plan

## Boundary

This is the active method branch after the 2026-06-24 user pivot. It supersedes
DP-as-main, scorer-only selection, and custom in-repo action diffusion as the
main insertion executor. Those older branches remain evidence for why the
pivot is needed, not the path to continue.

No training, conversion, norm-stat computation, inference, replay, rendering,
or evaluation may run on the login node. All project compute must run inside a
tmux-held interactive Slurm allocation.

## Starting Point

Use official OpenPI/pi0.5 code and weights from:

`/public/home/yanhongru/ICLR2027/openpi`

Relevant local official entry points:

- `src/openpi/training/config.py`
- `scripts/compute_norm_stats.py`
- `scripts/train.py`
- `examples/droid/convert_droid_data_to_lerobot.py`
- `src/openpi/policies/droid_policy.py`

Initial config pattern:

- `pi05_maniskill_peg733`;
- `Pi0Config(pi05=True, action_horizon=16, discrete_state_input=False)`;
- `LeRobotLiberoDataConfig`;
- fresh OpenPI norm stats computed from the converted 733 LeRobot dataset;
- weights from `gs://openpi-assets/checkpoints/pi05_base/params`.

`pi05_droid` is not the first formal run because its action convention is
DROID Franka joint velocity plus gripper, while the 733 simulator action is
ManiSkill `pd_ee_delta_pose`. It can be revisited only with a documented
action-space adapter.

## Dataset Plan

Input roots:

- accepted H5 source:
  `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733`;
- RGB/action/state export:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050`;
- approved RGB videos:
  `experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612`.

Convert into a LeRobot dataset with OpenPI-native fields:

- external RGB image from the approved 512x512 default ManiSkill view;
- wrist image duplicated from the approved external view for the first run,
  because no real wrist camera exists in the 733 export;
- proprio/state from causal qpos/gripper state;
- `actions` as the 7D source H5 `pd_ee_delta_pose` robot action. Do not feed
  the 32-column Cosmos task-state sidecar as policy action;
- `task` prompt describing the actual goal: insert the peg into the moved
  target hole from the current observed scene.

The converter must not feed future target final poses as observations. Future
pose/readout data may stay in metadata for labels and audits only.

## Training Plan

1. Add an OpenPI-compatible data config and policy transform only if the
   existing DROID/Libero transforms cannot map the 733 dataset fields directly.
   Keep it inside OpenPI's official config/transform style.
2. Compute normalization stats with OpenPI's official script inside Slurm.
3. Finetune pi0.5 on the 733-derived LeRobot data for at least one GPU-hour
   before interpreting the result.
4. Save checkpoints under a clearly named OpenPI experiment root, separate
   from Cosmos/contact-diffusion checkpoints.
5. Record exact checkpoint source, config name, action dimension, action
   horizon, norm-stat source, train duration, step count, and data root.

## Near-Contact Repair Plan

The object17-video branch proved that official OpenPI/pi0.5 can train from the
733-derived object/task-frame data for more than one GPU-hour and preserve
grasp, but it still did not insert directly. The next repair stays within the
same official OpenPI model family and changes only the training-window
distribution:

- config:
  `pi05_maniskill_peg733_contact_suffix16_object17_video_nearcontact_20260626`;
- repo id:
  `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_nearcontact_20260626`;
- source: the same accepted 733 H5/RGB data;
- state/action: object17 causal task state and 7D source H5
  `pd_ee_delta_pose` actions;
- image storage: LeRobot video streams, H.264, as in the clean object17-video
  branch;
- offsets before first insertion: `16,12,8,4,2,1`.

The old object17-video data used far-heavy offsets
`64,48,32,24,16,12,8,4`. The near-contact branch removes the long approach
windows and adds offset `2` and offset `1`, where offset `1` is the closest
pre-insertion causal action target. Offset `0` is not the main choice because
it starts at the first already-inserted frame and can teach post-insertion
maintenance rather than the missing insertion action.

Required gates:

1. Convert the near-contact LeRobot repo inside a tmux-held Slurm allocation.
2. Audit the repo in Slurm: `733` source episodes, state dim `17`, action dim
   `7`, suffix length `16`, and video-backed camera keys.
3. Compute fresh OpenPI norm stats for the new config/repo id.
4. Train official pi0.5 from official `pi05_base` for at least one GPU-hour.
5. Replay saved dynamic snapshots and inspect contact sheets before any
   success claim.

This is not DP-as-main, scorer-only selection, or a homemade VAE/MLP/diffusion
executor. It is a supervision-distribution repair inside the official
OpenPI/pi0.5 path.

Near-contact outcome:

The branch was executed in allocation `153455`:

- preparation root:
  `experiments/world_model_task_rebinding/openpi/object17_video_nearcontact_prepare_20260626_offsets16_12_8_4_2_1_alloc153455`;
- training root:
  `experiments/world_model_task_rebinding/openpi/pi05_object17_video_nearcontact_direct1700_1gpu1h_pyav_20260626_alloc153455`;
- replay root:
  `experiments/world_model_task_rebinding/openpi/20260626_object17_video_nearcontact_direct1700_replay4_exec16_alloc153455`.

It met the method constraints: accepted 733 source, official OpenPI/pi0.5
config, official `pi05_base` weights, no custom intermediate model, and
`5044` seconds of training. Replay was still negative for direct insertion:
direct success `0/4`, inserted `0/4`, contact-stable `0/4`, grasp `4/4`, and
DP96 handoff `1/4`. This means simple near-contact window reweighting is not
the missing contact-action solution.

The next plan should not repeat the same offset-only change. It should address
action-coordinate/contact-mode mismatch more directly while staying inside the
official OpenPI path, or replace privileged object17 slots with causal
RGB-derived task slots only after the action distribution issue is better
understood.

## Evaluation Plan

Evaluate only after a valid one-GPU-hour run exists:

1. Serve or load the OpenPI policy using official OpenPI inference code.
2. Query from saved dynamic live snapshots with only causal prefix/current
   observations.
3. Execute short action chunks in the simulator inside Slurm.
4. Measure direct post-chunk success, inserted predicate, contact stability,
   grasp preservation, insertion-axis progress, and optional DP96
   continuability as a historical comparison.
5. Inspect videos/contact sheets before any success claim.

The main baseline to beat is not training loss. It is the old direct-action
failure: `0/16` selected replay direct gate/success and `0/192` generated
candidate direct gate/success.

## Current Evidence As Of 2026-06-26

The official OpenPI/pi0.5 path is now real rather than speculative:

- `pi05_maniskill_peg733` uses official OpenPI `Pi0Config(pi05=True, ...)`,
  official norm stats, official `pi05_base` weights, and the audited 733
  LeRobot conversion.
- The first formal finetune met the one-GPU-hour floor and preserved checkpoint
  step `1599`.
- A longer resume reached step `3999`, but the final checkpoint did not
  finalize because Slurm marked the step `OUT_OF_MEMORY`; the latest valid
  preserved resume checkpoint is step `3600`.

Saved-snapshot replay from the step-`3600` checkpoint shows a narrow but real
positive signal:

- 8-step replay: direct success `0/4`, inserted `0/4`, contact-stable `0/4`,
  grasp `4/4`, DP96 success/continuability `2/4`;
- 16-step replay using the model's native action horizon: direct success `0/4`,
  inserted `0/4`, contact-stable `0/4`, grasp `4/4`, DP96
  success/continuability `2/4`.

The current conclusion is therefore not "OpenPI solved insertion." It is:
OpenPI has learned a better grasp-preserving, partly DP-continuable action prior
than the old scorer-only branch, but it still does not generate the final
contact/insertion action itself. Executing all 16 native actions did not fix the
direct insertion failure.

The new contact-suffix audit gives a concrete repair direction rather than a
guess. In the accepted 733 H5/RGB data, `733/733` episodes are final-success-like
and `733/733` have eligible 16-step suffix windows around first insertion,
with `71045` total eligible windows. First inserted frame spans `95..300`
with mean `172.77`. Therefore the immediate blocker is not that the dataset
lacks insertion actions; it is that full-episode behavior cloning and current
snapshot evaluation are not concentrating enough probability mass on the
short contact/insertion suffix.

The active contact-suffix dataset is:

- repo id `yanhongru/maniskill_peg733_openpi_contact_suffix16`;
- official LeRobot fields `image`, duplicated `wrist_image`, 8D qpos/gripper
  state, and 7D source H5 `pd_ee_delta_pose` actions;
- suffix windows selected at offsets `64,48,32,24,16,12,8,4` before first
  insertion, length `16`;
- OpenPI config `pi05_maniskill_peg733_contact_suffix16`, still using official
  pi0.5 model/weights and no custom intermediate network.

Smoke conversion from two source trajectories wrote `16` suffix episodes /
`256` frames successfully. The full 733 contact-suffix conversion then
completed in held Slurm allocation `150773`, step `28`, producing
`5853` suffix episodes / `93648` frames at repo id
`yanhongru/maniskill_peg733_openpi_contact_suffix16`. The corresponding
LeRobot audit passed with `unique_episode_lengths=[16]`, state dim `8`, action
dim `7`, and no failures. The conversion count is `11` windows below `733*8`
only because some very late insertions cannot fit every fixed offset before
the 300-step action boundary; all `733` source episodes still contribute.

A formal contact-suffix OpenPI/pi0.5 training run now exists. The first suffix
attempt reached step `1000` but OOMed during Orbax checkpoint save, leaving
only a temporary checkpoint. The repaired run used the same official
`Pi0Config(pi05=True, action_horizon=16, discrete_state_input=False)` and
official `pi05_base` loader, but set `ema_decay=None` for this suffix config
to reduce checkpoint memory and used final-only saving. It ran `1700` steps in
allocation `150773`, step `44`, for `4303` seconds, satisfying the one-GPU-hour
floor. Step `1699` finalized and was preserved at:

`experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733_contact_suffix16/pi05_peg733_contact_suffix16_noema_direct1700_1gpu1h_20260626_alloc150773/1699`

Training loss is positive optimization evidence, not task success evidence:
it moved from `0.1038` at step `0` to `0.0373` at step `1600`, with noisy
intermediate values.

The first saved-snapshot replay from checkpoint `1699` is negative for direct
task completion:

- exec16 panel root:
  `experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix16_noema1699_panel4_exec16_alloc150773_alloc150773`;
- direct success `0/4`;
- inserted `0/4`;
- contact-stable `0/4`;
- grasp preserved `4/4`;
- DP96 continuable `2/4`;
- DP96 success `1/4`.

Therefore the current conclusion is sharper: OpenPI/pi0.5 contact-suffix
training is now technically valid, but the learned suffix policy still does
not insert from the tested dynamic snapshots. It preserves grasp but does not
bring the peg into the insertion/contact-stable state needed by the task.

Contact-state sheets generated from replay step records confirm this boundary:
all four tested chunks keep grasp for `16/16` executed steps, but no chunk has
an inserted step. The strongest sample, f116, is only a DP handoff signal:
OpenPI leaves the state DP96-successful, while the OpenPI chunk itself remains
direct success `0` and inserted `0`. The failure mode is geometric, not a
missing-checkpoint issue: f106 moves rel-x from `-0.148` to `-0.198` and
abs(y)+abs(z) from `0.054` to `0.122`; f132 moves abs(y)+abs(z) from `0.013`
to `0.089`. This points to object-frame/contact-target alignment and receding
execution as the next problems, not to another scalar scorer.

The first diagnosis points to object/task-frame conditioning, not just prompt
or training-time issues:

- baseline diagnosis:
  `experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix1699_panel4_exec16_alloc150773/contact_suffix_replay_diagnosis.json`;
- `4/4` samples worsened `abs_yz_sum`, `3/4` worsened `abs_x`;
- mean `delta_abs_yz_sum=0.03767`, mean `delta_abs_x=0.02172`;
- tested snapshots were `[60, 60, 50, 66]` frames before source first
  insertion, near but not exactly on the fixed training offsets;
- same-time source-action cosine is high, but the predicted chunks
  over-amplify lateral motion and fail to bind the action to the moved hole.

A privileged prompt-phase diagnostic that injected source first-insert timing
into the prompt stayed negative: direct success `0/4`, inserted `0/4`,
contact-stable `0/4`, grasp `4/4`, DP96 continuable `1/4`. Because this used
future/source timing, it is diagnostic only. Its failure suggests that the main
blocker is not merely missing suffix-offset text in the prompt.

The first object/task-frame conditioning diagnostic has now been constructed,
but it has not produced a training result yet. The object17 LeRobot rewrite:

- repo id `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17`;
- 17D state
  `tcp_pose3, peg_pose3, hole_pose3, peg_head_at_hole3,
  hole_velocity_step3, grasped, inserted`;
- `5853` suffix episodes / `93648` rows, same `16`-step action horizon;
- official OpenPI config
  `pi05_maniskill_peg733_contact_suffix16_object17`, still using
  `Pi0Config(pi05=True, action_horizon=16, discrete_state_input=False)`,
  `LeRobotLiberoDataConfig`, and `pi05_base`.

Audit passed with state dim `17`, action dim `7`, no failures, and the same
image/wrist shape as the qpos8 suffix dataset. This is a useful upper-bound
diagnostic because it exposes current object/task geometry directly to OpenPI.
It is not yet the final RGB-derived method, because those slots come from
simulator/source metadata rather than an RGB perception module.

The object17 branch then exposed a more fundamental data-management issue.
The rewrite script originally cloned the qpos8 contact-suffix LeRobot repo with
`cp -al` hardlinks and then edited metadata/parquet state columns. That
mutated the canonical qpos8 suffix repo as well, because hardlinked files share
inodes. A subsequent repair attempt rebuilt the qpos8 repo from the original
733 H5/RGB source inside allocation `150773`, but Slurm cancelled the step at
about `97%` when the allocation entered `COMPLETING`. The canonical qpos8 repo
is therefore currently partial (`5704` episodes / `91264` frames observed) and
must not be used for training, norm stats, replay, or further conclusions.

This changed the immediate blocker ordering. The object17 first-step stall is
still useful evidence that the old object17 path did not produce a clean
training result, but it was observed while the data tree was contaminated and
cannot be treated as a clean OpenPI/LeRobot diagnosis.

The clean-data repair has now progressed. Fresh noncanonical qpos8 and object17
repo ids were rebuilt and audited without hardlinks:

- `yanhongru/maniskill_peg733_openpi_contact_suffix16_qpos8_clean_20260626`;
- `yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_clean_20260626`.

The image-backed clean object17 repo is structurally valid but still not a
practical OpenPI training input: official `LeRobotDataset(repo_id)` hung before
returning `dataset[0]`, likely because two `256x256` RGB streams are embedded
as `dtype=image` payloads in `5853` per-episode parquet files.

The current active repair is therefore the video-backed clean object17 repo:

`yanhongru/maniskill_peg733_openpi_contact_suffix16_object17_video_clean_20260626`

H.264 conversion and strict audit passed in allocation `152622`. The audit
reports `5853` parquet files, `11706` mp4 files, `93648` rows,
`unique_episode_lengths=[16]`, state dim `17`, action dim `7`, and video camera
dtypes. OpenPI-format norm stats were installed for the corresponding official
config
`pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626`.

This is a positive data/backend signal, not a policy result. The later loader
diagnosis showed that direct H.264/pyav video decode works, and the old timeout
was caused by Hugging Face `datasets.load_dataset` using shared cache storage:
even a single-episode load generated the split and then did not return. Moving
`HF_HOME`, `HF_DATASETS_CACHE`, and `XDG_CACHE_HOME` to compute-node `/tmp`
makes single-episode constructor/item access pass and makes the full raw item
gate pass (`raw_item_rc=0`).

The patched loader gate, formal training, and first replay have now completed.
With `/tmp` Hugging Face cache and the OpenPI-root working-directory fix, the
video-backed object17 repo passed raw item, transformed item, and first-batch
debug in allocation `153455`.

Formal object17-video OpenPI/pi0.5 training then ran for `4748` seconds in the
same held allocation, satisfying the one-GPU-hour floor. It used official
`Pi0Config(pi05=True, action_horizon=16, discrete_state_input=False)`,
official `pi05_base` restore, audited 733-derived video-backed object17
LeRobot data, and no custom intermediate model. The preserved checkpoint is:

`experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733_contact_suffix16_object17_video_clean_20260626/pi05_object17_video_clean_direct1700_1gpu1h_20260626_alloc153455/1699`

Saved-snapshot replay from this checkpoint used matching 17D causal
object/task state preparation and executed `16` OpenPI actions per sample:

- replay root:
  `experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_direct1700_replay4_exec16_fixhistory_alloc153455`;
- direct success `0/4`;
- inserted `0/4`;
- contact-stable `0/4`;
- grasp preserved `4/4`;
- DP96 historical continuability/success `3/4`.

Contact-state sheets for that replay confirm that no OpenPI chunk inserted
during its own `16` steps, although all four kept grasp and three left states
from which the historical DP96 rollout could finish. This is the strongest
OpenPI/pi0.5 handoff signal so far, but not direct task completion.

The current blocker has therefore moved: it is no longer data conversion,
video decode, first-batch loading, official weight compatibility, or the
minimum training floor. The blocker is direct contact/insertion execution from
dynamic snapshots. The next plan should keep OpenPI/pi0.5 as the action model
and test receding/refreshed-observation execution or stronger contact-target
supervision, rather than adding another scorer over weak chunks.

A first split-env receding diagnostic has now been implemented and run:

- scripts:
  `scripts/openpi/run_openpi_pi05_receding_snapshot_rollout.py` and
  `scripts/slurm/run_openpi_pi05_receding_snapshot_rollout_in_allocation.sh`;
- run root:
  `experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_receding1_q3_exec4_alloc153455`;
- one sample, prefix `106`, `3` OpenPI queries, `4` executed steps per query;
- direct success `0/1`, inserted `0/1`, contact-stable `0/1`, grasp `1/1`;
- `abs(y)+abs(z)` worsened from `0.0539` to `0.1341`.

This diagnostic used refreshed simulator object17 state and a static
observed-prefix image. It is useful upper-bound evidence that a naive receding
wrapper alone does not fix the first hard snapshot, but it is not final
RGB-derived method evidence.

RGB replay video evidence remains blocked by the live-render/Vulkan path. The
contact-state sheets are authoritative state/contact diagnostics, not simulator
RGB video and not success claims.

## Execution Note

The preferred path remains OpenPI's official dependency lock and scripts. On
the current cluster, `uv` hit network fetch failures for the pinned official
LeRobot/dlimp sources and then NFS lock behavior; Python 3.12 pip also fails
because official `dlimp` pins `tensorflow==2.15.0`. The active operational
repair is therefore a compute-node Python 3.11 environment with local mirrors
of the exact official LeRobot/dlimp commits, launched through
`OPENPI_PYTHON=/tmp/openpi_py311_env_yanhongru_149062/bin/python`. This changes
only dependency transport and interpreter compatibility; model code, config
style, and pi0.5 checkpoint loading remain official OpenPI.

## Current Open Questions

- Does duplicated external-view-as-wrist hurt enough to require a custom
  OpenPI-style image mask transform?
- Does pi0.5-base with fresh simulator norm stats produce direct insertion
  chunks, or is a DROID/Franka action adapter worth building later?
- Can the current held `server24` allocation clear its stale step cleanly, or
  should the next compute action use a fresh tmux-held allocation?
- How should the causal object/task-frame geometry enter OpenPI while staying
  official-code compatible? The next repair should add peg/hole/TCP relative
  features through an OpenPI-native data/config transform and fresh norm stats,
  then rerun suffix training/replay. It must not become a custom
  intermediate model or scorer-only selector.
- Can clean qpos8/object17 LeRobot repos be rebuilt without hardlink
  contamination and without in-place partial overwrite? This is now the first
  blocker to clear before interpreting object17 training behavior.
- After clean rebuild, why does the object17 official OpenPI training path
  stall after split generation and before `Step 0`, if it still does? The next
  compute-side debug should then focus on dataloader/cache/prefetch/worker
  behavior or LeRobot dataset materialization while keeping the model and
  config OpenPI-native.
- Object17 reached a valid one-GPU-hour training result and improved
  DP-continuability to `3/4`, but direct insertion stayed `0/4`. A first
  q3x4 receding smoke on f106 also stayed direct `0/1`, so the stronger
  current hypothesis is that the learned action distribution still lacks a
  reliable insertion-axis contact mode.
- If object17 remains useful as an upper bound, how do we replace privileged
  simulator object slots with RGB-derived object/task-state perception while
  preserving the same OpenPI-native conditioning interface?
- Can replay RGB evidence be produced through an offline/default-camera render
  path that avoids the current live-render/Vulkan hang?
