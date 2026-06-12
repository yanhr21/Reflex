# 2026-06-11 Fix2 Official-Insertion Reproduction Baseline

## Status

Rejected after user direct video review on 2026-06-11. This static
fix2/official ManiSkill insertion reproduction smoke is not a baseline for
future fix3 construction, not user-approved fix3 review data, and not SFT
evidence.

No fix3 SFT was started.

The user-identified rejected video was:

`experiments/world_model_task_rebinding/cosmos3/fix2_official_insert_repro_smoke6_20260611_server56_padded301/render_512_serverserver56/train/videos/0001_none_fix2_official_seed1002002_idx0001.fix2_traj_0.mp4`

Failure: severe peg/target penetration. Therefore this repro cannot be used as
proof that the current reproduction/gate stack is physically valid.

## User Correction Recorded

The user clarified that penetration, physical-gate failure, wall insertion, and
peg self-drilling cannot be fixed by changing protocol wording. Since the
official/fix2 path can work but the modified fix3 path fails, the failure must
be treated as damage from the added modifications or evidence gates.

The rejected modifications now include:

- late-trigger/final-pose variants that visually inserted into the wall;
- Qwen-approved direct-video gates that missed penetration and misalignment;
- the constrained-insert v8 projection, which caused visible penetration and
  peg self-drilling artifacts;
- strict v8 centerline gates used as a standardizer for official fix2
  reproduction, which rejected an official-success trajectory and therefore
  cannot be treated as the fix2 reproduction authority.

## Code Boundary

Added:

`scripts/world_model/generate_cosmos3_fix2_official_insert_smoke.py`

Added allocation-only runner:

`scripts/slurm/run_fix2_official_insert_smoke_in_allocation.sh`

The generator reproduces the official ManiSkill solver structure:

- grasp the peg;
- compute `insert_pose = goal_pose * peg_init_pose.inv() * grasp_pose`;
- move to `-0.01 - peg_half_length` pre-insert pose;
- run three official pre-insert refinements;
- execute the final `+0.05 m` insertion target.

It applies no target motion enlargement and no constrained peg projection.

## Slurm Evidence

Allocation:

- job `125642`;
- node `server56`;
- one H200 via tmux-held `salloc`;
- wrapper ran inside `srun` step, not on the login node.

Archived rejected root:

`/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate/fix2_repro_user_rejected_penetration_20260611/fix2_official_insert_repro_smoke6_20260611_server56_padded301`

Archived noncontract/intermediate roots:

`/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate/fix2_repro_failed_or_noncontract_20260611`

The archived roots include:

- `fix2_official_insert_repro_smoke6_20260611_server56`: official solver
  accepted 6 samples, but the later v8 strict centerline standardizer rejected
  the first accepted trajectory. This is evidence against reusing that v8 gate
  as a fix2 authority.
- `fix2_official_insert_repro_smoke6_20260611_server56_rawsplit`: rendered
  valid videos, but they used raw unpadded frame counts (`121-166` frames), so
  they are noncontract diagnostics only.

## Rejected Artifacts

Padded root before archival:

`experiments/world_model_task_rebinding/cosmos3/fix2_official_insert_repro_smoke6_20260611_server56_padded301`

Generation:

- accepted `6` official-solver success trajectories from `8` attempts;
- output paths file:
  `fix2_h5_paths.txt`;
- H5 records are padded to `301` states/video frames and `300` action steps
  for renderer contract compatibility.

Rendering:

- render root:
  `render_512_serverserver56`;
- `6` mp4 videos;
- every video has `301` frames at `30 fps`;
- duration is `10.033333s`;
- camera is the approved ManiSkill default human-render camera;
- videos are readable and nonblank per
  `render_512_serverserver56/train_dense_video_review/video_artifact_inspection.json`.

Dense review sheets opened by the agent:

- `0000_none_fix2_official_seed1002001_idx0000.fix2_traj_0_review_sheet.png`
- `0001_none_fix2_official_seed1002002_idx0001.fix2_traj_0_review_sheet.png`
- `0002_none_fix2_official_seed1002003_idx0002.fix2_traj_0_review_sheet.png`
- `0003_none_fix2_official_seed1002004_idx0003.fix2_traj_0_review_sheet.png`
- `0004_none_fix2_official_seed1002005_idx0004.fix2_traj_0_review_sheet.png`
- `0005_none_fix2_official_seed1002007_idx0005.fix2_traj_0_review_sheet.png`

The prior agent dense-sheet review is invalidated by the user's direct video
inspection. The failed direct-video evidence is authoritative here: this
package has visible penetration and must not be reused.

## Next Valid Fix3 Step

Return to the originally effective 2026-06-06 full1000 dataset/protocol:

`experiments/world_model_task_rebinding/cosmos3/sft_dataset_full1000_maniskill_default_regen_20260606_0055`

The next construction must use the original physical gate as the baseline,
then resample/regenerate accepted demos so that every sample has true final
insertion, late target motion after peg alignment, larger target displacement,
and continuous moving-target cases. If the physical gate rejects a candidate,
the candidate is failed and resampled; insertion must never be forced by
state projection or penetration.

Stop at small rendered smoke review before any 60-video package or SFT.
