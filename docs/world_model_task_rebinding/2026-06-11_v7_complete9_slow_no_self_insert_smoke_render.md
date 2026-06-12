# 2026-06-11 V7 Complete9 Slow No-Self-Insert Smoke Render

## Boundary

This note records the v7 complete-nine smoke render after the user rejected
v4/current6 videos for two concrete semantic failures:

- `hole_late_move_stop` and `hole_late_fast_shift` were too fast and collided
  into the peg/hole region.
- `hole_late_constant` and `hole_late_sine` showed target motion toward the
  peg, making it possible for the target to self-insert without meaningful
  robot rebinding.

V7 is still only a smoke-review package. It is not user approval, not a
60-video package, and not permission to start Cosmos3 SFT.

## First-Principles Check

Physical problem: the dataset must show the robot reacting to a late target
motion, not the target block solving the task by moving into the peg or by
allowing wall/penetration artifacts.

Why this is task-frame rebinding: the target should move after the peg is held
and near the old/current hole, then the robot must redirect the held peg toward
the new/future hole before final insertion. The policy challenge is the changed
task frame, not an easier collision/self-insertion trajectory.

Evidence required: every accepted row must have `301` observation/render
frames, `300` actions, true final insertion, target motion before insertion
for moving-hole classes, first insertion after the target-motion window, and
visible video evidence for the nine-class smoke set.

Falsification: user video review finds target self-insertion, direct collision,
wall insertion, penetration, target motion before peg pickup/prealignment, or
any final failure.

Objective preservation: the original 300-step full-episode contract is
preserved. No 128/129-frame chunking, no cropped evaluation, and no forced
state projection is used.

## Code Change

The copied fix3 generator was patched only in the fix3-specific script:

`scripts/world_model/generate_cosmos3_fix3_original_protocol_large_motion_dp.py`

The original 2026-06-06 full1000 dataset/protocol remains read-only.

V7 adds a stricter rejection gate for moving-hole candidates:

- counterfactual future-target path gate: reject a candidate if the future
  moving-hole path would already align with the trigger-time peg head;
- post-motion self-insert gate: reject if the target moved at least `0.08m`,
  the TCP moved less than `0.04m` before first insertion, and the peg-head YZ
  error is already within `0.015m` before insertion.

The rejected v7 fast-shift search confirmed the gate is active: bad
`hole_late_fast_shift` seeds were rejected for
`target_self_insert_without_robot_rebind_motion` before the accepted
replacement was kept.

## Rendered Smoke Package

Combined root:

`experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611`

Rendered root:

`experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21`

The render ran inside Slurm allocation `125951` on `server21` with the
approved ManiSkill default human camera, `512x512`, `30 fps`, and full
`301`-frame videos. No rollout/render task ran on the login node.

Structural audits:

- H5 gate audit:
  `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/combined_gate_audit.json`
- video frame audit:
  `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21/video_frame_audit.json`
- render manifest:
  `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21/manifest.json`

Both audits have `failures=[]`. The render manifest has exactly one video for
each required class:

`hole_late_move_stop`, `hole_late_constant`, `hole_late_reverse`,
`hole_late_sine`, `hole_late_continuous_insert`,
`hole_late_fast_shift`, `none`, `peg_drop`, and `peg_disturb`.

## H5 Gate Snapshot

All nine rows have `inserted_end=true`, `301` observation frames, and `300`
action steps. Moving-hole rows have first insertion after the target-motion
window and meaningful TCP motion before insertion:

| scenario | first motion | last motion | first insert | target motion | TCP motion before insert |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 97 | 127 | 147 | `0.2247m` | `0.3515m` |
| `hole_late_constant` | 96 | 124 | 154 | `0.2208m` | `0.3317m` |
| `hole_late_reverse` | 102 | 133 | 197 | `0.2255m` | `0.3367m` |
| `hole_late_sine` | 92 | 132 | 168 | `0.2238m` | `0.2207m` |
| `hole_late_continuous_insert` | 88 | 142 | 179 | `0.2341m` | `0.1491m` |
| `hole_late_fast_shift` | 82 | 93 | 173 | `0.2327m` | `0.3314m` |

This does not by itself prove visual acceptability; it only proves the H5 gate
is no longer accepting the concrete v4/v5 failure pattern where insertion
happens during target motion with little robot rebinding.

## Rendered Videos

- `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21/val/videos/0000_hole_late_move_stop_seed1080064_idx0000.fix3_traj_0.mp4`
- `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21/train/videos/0001_hole_late_constant_seed1050118_idx0001.fix3_traj_0.mp4`
- `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21/train/videos/0002_hole_late_reverse_seed1040017_idx0002.fix3_traj_0.mp4`
- `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21/train/videos/0003_hole_late_sine_seed1050127_idx0003.fix3_traj_0.mp4`
- `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21/train/videos/0004_hole_late_continuous_insert_seed1040042_idx0004.fix3_traj_0.mp4`
- `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21/train/videos/0005_hole_late_fast_shift_seed1100041_idx0000.fix3_traj_0.mp4`
- `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21/train/videos/0006_none_seed700107_idx0006.fix3_traj_0.mp4`
- `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21/train/videos/0007_peg_drop_seed705095_idx0007.fix3_traj_0.mp4`
- `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21/train/videos/0008_peg_disturb_seed1051032_idx0008.fix3_traj_0.mp4`

All-frame framebooks were also generated under:

`experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21/all_frames_framebooks_20fpp`

The agent then inspected every page of the all-frame framebooks for all nine
videos. Each video has 16 pages: `15` pages with `20` consecutive frames and a
final page for frame `300`, so the inspection covered every rendered frame
from `0` through `300` rather than a sparse key-frame sheet.

All-frame visual inspection outcome:

- `hole_late_move_stop`: target stays still during pickup/approach, moves late,
  then the robot redirects and inserts after the target-motion window. No
  visible target self-insertion, wall insertion, or penetration.
- `hole_late_constant`: target moves after the peg is held/near, not by
  driving into a static peg. Final insertion is robot-followed and stable.
- `hole_late_reverse`: target reverse motion completes before insertion; the
  robot follows to the final hole pose and holds insertion.
- `hole_late_sine`: the peg is close during target motion, so this remains a
  visually high-risk case, but the framebook does not show stable insertion
  before the target-motion window ends, and the final insertion is stable.
- `hole_late_continuous_insert`: another high-risk close-peg case, but the
  target-motion window ends before stable insertion and the robot continues
  adjusting before insertion. No visible side-wall insertion.
- `hole_late_fast_shift`: the target shifts quickly and stops; the peg is not
  carried into the hole by the target. Frames after the shift show robot
  rebinding before insertion.
- `none`: static baseline insertion remains stable with no target motion or
  visual physics-gate failure.
- `peg_drop`: the peg visibly drops to the table, the robot re-approaches and
  re-grasps it, then inserts and holds.
- `peg_disturb`: the peg is visibly perturbed while held; the robot recovers
  alignment and inserts without target motion or a detached/self-inserting peg.

This all-frame smoke inspection passes the current user continuation gate for
scaling to full1000 data generation. It is still smoke evidence, not Cosmos3
SFT evidence or controller evidence.

## Decision

The latest user continuation directive allows full1000 generation only after
the all-frame smoke videos truly pass. That condition is now satisfied by the
agent all-frame inspection above, so full1000 data generation was started in a
tmux-held Slurm allocation:

- allocation: `126029`
- node: `server52`
- output root:
  `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_full1000_20260611`
- log:
  `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_full1000_20260611.generate.log`

Do not start Cosmos3 SFT from this data until the full1000 source itself passes
strict H5/source audits, rendered video/frame checks, full-episode WAM
condition export, preflight, and action-target audits.
