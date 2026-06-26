# Fix3 Successful Large-Motion Data Plan

## Current Plan Boundary

The earlier widened-final-pose/overlay construction below is historical and
was rejected by user video review. The active fix3 plan is now the
copy-on-write original-protocol rebuild:

- keep the original 2026-06-06 full1000 dataset/protocol read-only;
- generate new fix3 candidates with the original DP-driven late-target
  protocol rather than post-hoc state overlay or forced insertion;
- target stays static while the peg is picked and brought near the old/current
  hole, then target motion starts late;
- final accepted rows must pass real simulator insertion and slot insertion;
- moving-hole rows must show enough target displacement to require rebinding,
  but must not let the target move into the peg or self-insert;
- smoke taxonomy is the complete nine classes:
  `hole_late_move_stop`, `hole_late_constant`, `hole_late_reverse`,
  `hole_late_sine`, `hole_late_continuous_insert`,
  `hole_late_fast_shift`, `none`, `peg_drop`, and `peg_disturb`;
- the current smoke approval candidate is v7:
  `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21`.

The v7 all-frame smoke inspection passed the continuation gate:
the agent opened every framebook page covering all `301` frames for each of the
nine videos and did not find target self-insertion, wall insertion, visible
penetration, early target motion, or final failure. Full1000 data generation
began from the same v7 copied original-protocol generator.

Latest 2026-06-12 execution override: the hard-teacher supplement and 1500-row
expansion are deferred. The active near-term source target is the original
v7 DP-generated `full1000` set. Keep all already accepted v7 rows, finish the
remaining original full1000 quotas with the v7 DP generator, then run merge,
strict source audit, rendered per-class review, and user approval before WAM
export or Cosmos3 SFT. The hard-teacher script remains as later work, not as a
current blocker.

## Why Fix3 Exists

The fresh fix1/fix2 line localized a data-quality failure that invalidates
continued SFT on the current full1000 source. Many reference/validation videos
do not end with a real insertion, and source metadata confirms poor
`inserted_end` rates in several scenarios. This is not a training-duration
problem.

The physical objective for fix3 is therefore:

1. every source demo must finish inserted;
2. target-hole motion must be large and varied enough to be a real dynamic
   challenge, not a small displacement that the frozen static DP can absorb;
3. some demos must require aiming at the future target location while the hole
   is still moving;
4. the full 301 RGB/state frame / 300 action-step contract remains unchanged.

The hard dynamic remainder has one extra requirement: the original/static DP
must be evaluated on the same dynamic family and shown to be unable to solve it
reliably. This is the reason to use a world model. The positive training rows
may use a teacher that manually/scriptedly tracks the target, moves the held
peg, or hands off to DP only for the final local insertion when that preserves
physical correctness, but they must not be selected solely because the original
DP already solved the dynamic episode.

## Historical Invalid Construction To Avoid

Do not generate fix3 by taking a successful static expert trajectory and making
the moving target return to that same old static terminal pose. That only
guarantees insertion by bringing the task back to the expert's ID final state.
It would teach "restore the old static endpoint", not dynamic task-frame
rebinding.

The postprocessor
`scripts/world_model/generate_cosmos3_fix3_successful_dynamic_dataset.py`
therefore refuses, by default, to run directly on the official static replay
source. It may only consume a source set whose final target poses were already
sampled for fix3 and solved successfully.

This approach is not the active construction after the later physics-gate
failures. The active construction is the copied original-protocol generator:

`scripts/world_model/generate_cosmos3_fix3_original_protocol_large_motion_dp.py`

## Historical Overlay Construction

Generate a new source expert set first:

- sample widened reachable final hole poses, including positions outside the
  old static DP target-position support where feasible;
- use a scripted/motion-planning/manual expert to solve insertion at each new
  final target pose;
- reject any source attempt whose final simulator state is not inserted;
- record robot, peg, target, TCP, action, and env-state traces.

Then overlay dynamic target trajectories:

- the target starts far from its final pose or follows a large path;
- path families include move-stop, constant movement, reversal, curved/sine
  movement, late shift, and continuous motion until insertion;
- the final target pose is the new sampled fix3 target pose, not the original
  static replay terminal pose;
- robot/peg states come from the expert solution to that new final target, so
  the visual rollout shows successful insertion in the changed world.

## Required Gates

Before any SFT:

- strict H5 audit: every trajectory has `301` states, `300` actions, final
  `slots/inserted[-1] == true`, and target motion norm above the fix3 floor;
- provenance audit: retained v7 rows are merged by scenario/seed uniqueness,
  with hard-teacher rows excluded until the deferred later direction resumes;
- source-video render: default ManiSkill human camera, 30 fps, no slowdown;
- render `10` videos per fix3 scenario for user human inspection;
- stop at a user-approval gate after those per-class videos are ready. Do not
  start SFT until the user approves the rendered demos;
- full-episode WAM condition export and preflight with no 128/129/93-frame
  artifacts;
- action target audit under `future_aligned_state`.

## Evidence Boundary

Fix3 source generation is data construction, not controller success. Cosmos3
SFT evidence only starts after the fixed data passes source metrics, rendered
visual review, explicit user approval, strict condition preflight, and then
generated-video/action/readout evaluation.
