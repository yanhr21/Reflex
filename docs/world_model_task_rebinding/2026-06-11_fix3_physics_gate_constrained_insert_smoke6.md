# 2026-06-11 Rejected Fix3 Constrained-Insert Smoke6

## Status

This package is rejected historical diagnostic evidence only. It must not be
used for fix3 data generation, 60-video review, Cosmos3 SFT, controller input,
or method evidence.

No fix3 SFT was started from it.

## User Correction

The user rejected the previous
`fix3_late_trigger_smoke60_20260611_official_insert_vlm_gate` package after
direct video inspection. The local Qwen2.5 direct-video gate is therefore not
approval evidence. User-identified failures included initial penetration,
peg/hole misalignment, and visual side/wall insertion.

After that, the constrained-insert v8 smoke was generated and rendered:

`experiments/world_model_task_rebinding/cosmos3/fix3_physics_gate_constrained_insert_smoke6_20260611_v8`

The user then rejected v8 by direct video inspection. In particular:

- `0000_hole_late_move_stop_seed999000_idx0000.fix3_traj_0.mp4` visibly
  penetrates/misaligns;
- the peg appears to drill/crawl into the hole instead of being physically
  inserted by robot motion;
- the previous key-frame-only agent inspection is not valid evidence.

The active root was moved outside Reflex under:

`/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate/user_rejected_constrained_insert_v8`

## Rejected Implementation Boundary

The constrained insert projection changed the standardized final insertion
segment after the planner's raw motion. Even though the H5 audit recorded line
clearance and no wall-risk flags, the rendered video showed a non-physical
insertion artifact. Therefore this projection is not an active data
construction method.

The generator now disables `constrained_insert_projection` by default and
raises unless `ALLOW_REJECTED_CONSTRAINED_INSERT_DIAGNOSTIC=true` is explicitly
set for archived diagnostics.

## Next Valid Action

The next fix3 data step is to first reproduce the fix2/official ManiSkill
insertion protocol as a small Slurm-side smoke, without target-motion
enlargement and without constrained projection. That smoke must be inspected
from actual videos or dense insertion-frame sequences. Only after the fix2
protocol is visibly and physically correct should target displacement be
increased while preserving the same insertion protocol.
