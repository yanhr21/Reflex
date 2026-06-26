# 2026-06-26 OpenPI pi0.5 Near-Contact Object17-Video Diagnostic

## Boundary

This branch keeps the official OpenPI/pi0.5 action-model path:

- official config:
  `pi05_maniskill_peg733_contact_suffix16_object17_video_nearcontact_20260626`;
- official model:
  `Pi0Config(pi05=True, action_horizon=16, discrete_state_input=False)`;
- official weight loader:
  `gs://openpi-assets/checkpoints/pi05_base/params`;
- data source: accepted 733 ManiSkill PegInsertionSide trajectories;
- no DP-as-main controller, scorer-only selector, VAE, MLP, custom diffusion
  executor, or non-OpenPI policy model.

## Data

Preparation root:

`experiments/world_model_task_rebinding/openpi/object17_video_nearcontact_prepare_20260626_offsets16_12_8_4_2_1_alloc153455`

The data branch uses the same object17 causal task state and video-backed
LeRobot camera keys as the clean object17-video branch, but changes the suffix
offsets to `16,12,8,4,2,1`.

Evidence:

- audit `passed=true`;
- `733` source episodes;
- `4375` suffix episodes;
- `70000` rows;
- `8750` mp4 files;
- state dim `17`, action dim `7`;
- norm stats written to the OpenPI assets tree.

The suffix count is `4375`, not `733*6`, because late first-insertion episodes
cannot provide every near-contact offset while preserving a 16-step suffix.

## Training

Training root:

`experiments/world_model_task_rebinding/openpi/pi05_object17_video_nearcontact_direct1700_1gpu1h_pyav_20260626_alloc153455`

Evidence:

- `training_walltime_summary.json`: `elapsed_seconds=5044`;
- `formal_one_gpu_hour_floor_met=true`;
- `train_return_code=0`;
- final checkpoint `1699` preserved at:
  `experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733_contact_suffix16_object17_video_nearcontact_20260626/pi05_object17_video_nearcontact_direct1700_1gpu1h_pyav_20260626_alloc153455/1699`.

The first train attempt failed before optimization because LeRobot defaulted to
`torchcodec`, and the compute environment lacked compatible FFmpeg shared
libraries. The successful run explicitly used `LEROBOT_VIDEO_BACKEND=pyav`.
This was an environment/backend repair, not a model change.

## Replay

Replay root:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_nearcontact_direct1700_replay4_exec16_alloc153455`

Result:

- wrapper return code `0`;
- labels/action chunks `4`;
- executed `16` OpenPI actions per sample;
- direct success `0/4`;
- direct inserted `0/4`;
- direct contact-stable `0/4`;
- grasp preserved `4/4`;
- DP96 historical continuability/success `1/4`.

Contact-state sheets:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_nearcontact_direct1700_replay4_exec16_alloc153455/contact_state_sheets/contact_state_sheets_manifest.json`

The sheets show no inserted step in any OpenPI chunk and grasp preserved for all
executed steps.

## Conclusion

This diagnostic is a negative result for insertion. It completes the
near-contact OpenPI training/evaluation check, not the overall dynamic
insertion research objective.

Near-contact window reweighting did not solve direct insertion. It preserved
grasp but did not produce reliable insertion-axis contact behavior, and it
reduced historical DP96 handoff from the previous object17-video result
(`3/4`) to `1/4`.

The blocker is therefore not only far-window dilution in the suffix dataset.
The replay does not test Cosmos-generated actions; it isolates official OpenPI
as the action executor from saved dynamic snapshots. Because the policy still
fails with privileged object17 task-state conditioning, the immediate blocker
is the OpenPI action policy/contact executor, specifically its inability to
produce the final insertion-axis correction in the right action coordinates
from dynamic takeover states. The next repair should target
action-coordinate/contact-mode mismatch or a stronger official
OpenPI-compatible contact-action formulation, while keeping the
no-custom-intermediate-model boundary.
