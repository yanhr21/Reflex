# 2026-06-11 V5b Complete9 Smoke Frame Review

## Superseded Boundary

This v5b smoke package is superseded by the later v7 smoke package and must
not be used as approval evidence for full fix3 generation or Cosmos3 SFT.

The reason is concrete: after the user rejected the v4/current6 videos for
excessive speed and target self-insertion semantics, the subsequent audit found
that the old v5b `hole_late_move_stop` row would fail the stricter
post-motion self-insert gate. The `hole_late_fast_shift` class was regenerated
again because earlier fast-shift candidates could insert during target motion
with little TCP rebinding. The v5b note remains only as historical smoke
evidence for the complete-nine taxonomy and framebook tooling.

## Scope

This note records the agent-side framebook review of the v5b Complete9 smoke
package. It is smoke/data-quality evidence only. It is not user approval, not
a 60-video package, and not permission to start Cosmos3 SFT.

Combined H5 paths:

`experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v5b_complete9_combined_20260611/fix3_h5_paths.txt`

Rendered videos:

`experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v5b_complete9_combined_20260611/render_512_server56/train/videos`

All-frame framebooks:

`experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v5b_complete9_combined_20260611/render_512_server56/all_frames_framebooks_20fpp`

The render manifest has `9` videos, each `301` frames, `30 fps`, `512x512`,
and the approved ManiSkill default human camera. The framebook manifest has
`9` records, `16` pages per video, and `144` page images total, covering every
frame from `0` through `300`.

## Geometry Audit Snapshot

The H5 state audit found final physical insertion for all nine accepted rows.
For the six moving-hole rows, target motion is about `0.22-0.234m`, and the
counterfactual anti-self-insert gate checks that the current trigger-time peg
head is not already aligned with the future moving-hole path.

| id | scenario | seed | trigger | first move | target motion | counterfactual min YZ | final inserted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0000 | hole_late_move_stop | 1050023 | 103 | 104 | 0.222 | 0.0636 | true |
| 0001 | hole_late_constant | 1050118 | 95 | 96 | 0.221 | 0.0687 | true |
| 0002 | hole_late_reverse | 1040017 | 101 | 102 | 0.225 | 0.0825 | true |
| 0003 | hole_late_sine | 1050127 | 91 | 92 | 0.224 | 0.1014 | true |
| 0004 | hole_late_continuous_insert | 1040042 | 87 | 88 | 0.234 | 0.0583 | true |
| 0005 | hole_late_fast_shift | 1051012 | 119 | 120 | 0.224 | 0.0781 | true |
| 0006 | none | 700107 | -1 | -1 | 0.000 | n/a | true |
| 0007 | peg_drop | 705095 | 110 | -1 | 0.000 | n/a | true |
| 0008 | peg_disturb | 1051032 | 127 | -1 | 0.000 | n/a | true |

`0004` is the closest accepted moving-hole counterfactual case, with min YZ
`0.0583m` against the v5b hard floor of about `0.055m`. It should remain a
human-review item even though it passed the current gate.

## Framebook Review

The agent opened all `144` framebook pages before writing this note.

- `0000_hole_late_move_stop_seed1050023_idx0000`: target is static during the
  beginning; motion starts after peg pickup/near-hole approach; the final
  segment shows stable insertion rather than visible side-wall insertion.
- `0001_hole_late_constant_seed1050118_idx0001`: target is static at the
  beginning and moves after the peg is held. Frames around `120-160` still make
  the target motion visually close to the robot/peg region, but the
  counterfactual path gate shows the future hole path is not already on the
  trigger-time peg centerline.
- `0002_hole_late_reverse_seed1040017_idx0002`: target stays static through
  the early approach, then makes a late larger reverse-style motion; the robot
  visibly redirects and finishes insertion.
- `0003_hole_late_sine_seed1050127_idx0003`: target stays static early; after
  the late trigger the target moves away/down in the image and the robot
  follows before stable final insertion.
- `0004_hole_late_continuous_insert_seed1040042_idx0004`: target moves after
  peg pickup/near-hole approach and final insertion is stable in the framebook.
  This remains a borderline counterfactual-margin sample because its accepted
  min YZ margin is the smallest of the moving-hole set.
- `0005_hole_late_fast_shift_seed1051012_idx0005`: v5b slowed this class
  relative to v4, and final insertion looks physically stable. It is still the
  highest-risk semantic sample for user review because frames around `120-139`
  show a quick target shift toward the robot/peg region. The current gate says
  the trigger-time peg is not already aligned with the future hole path, but
  this sample should not be treated as user-approved without direct video
  review.
- `0006_none_seed700107_idx0006`: static target, normal pickup and final
  insertion; no moving-target semantics are expected.
- `0007_peg_drop_seed705095_idx0007`: peg drop/perturbation is visible in the
  middle; the robot later recovers and the final segment shows stable
  insertion.
- `0008_peg_disturb_seed1051032_idx0008`: target remains static; peg pose is
  visibly disturbed before the robot re-aligns and inserts.

## Decision Boundary

V5b is better aligned than the rejected v4 package in taxonomy and basic
render structure, but it is not the active approval candidate anymore. The
stricter v7 gate and replacement rows supersede this package.

No full1000/fix3-scale generation and no Cosmos3 SFT should start from v5b.
