# Cosmos3 Fix3 Data Reset Evidence

## Trigger

The user observed that many validation/reference videos under the fresh
full1000 SFT eval root do not end with successful insertion. This makes the
current source unsuitable for world-model SFT, regardless of later training
loss.

## Source Metadata Audit

The old full1000 source dataset was audited inside Slurm allocation `123385`
before the allocation expired. Source root:

`experiments/world_model_task_rebinding/cosmos3/sft_dataset_full1000_maniskill_default_regen_20260606_0055`

The audit found `1000` rows split across:

- `hole_constant=167`
- `hole_move_stop=167`
- `hole_reverse=167`
- `none=166`
- `peg_disturb=166`
- `peg_drop=167`

Final insertion is not reliable:

- `hole_constant`: `33/167` inserted at end
- `hole_move_stop`: `21/167` inserted at end
- `hole_reverse`: `125/167` inserted at end
- `none`: `104/166` inserted at end
- `peg_disturb`: `2/166` inserted at end
- `peg_drop`: `43/167` inserted at end

Moving-target displacement is also narrow:

- `hole_constant`: median about `0.126m`, max about `0.143m`
- `hole_move_stop`: median/max about `0.140m`
- `hole_reverse`: median about `0.088m`, max about `0.105m`

This supports the user's diagnosis: the old source is not a valid fix3 SFT
curriculum.

## Fix3 Boundary

Fix3 must regenerate source demonstrations. Filtering the old data is
insufficient because the remaining target-motion distribution is too narrow and
does not cover continuous moving-target insertion.

The invalid shortcut is also explicitly rejected: do not make the moving target
return to the old static expert terminal pose just to guarantee insertion.
Instead, first generate successful expert/manual insertion at new widened final
target poses, then use those final target poses as the end of dynamic target
paths.

## Current Code State

Added
`scripts/world_model/generate_cosmos3_fix3_successful_dynamic_dataset.py` as a
postprocessor, but it now refuses direct official-static replay input by
default. It can only be used after a proper fix3 expert source set exists.

Added
`scripts/world_model/generate_cosmos3_fix3_final_pose_expert_sources.py` to
create that proper source set. It samples widened final hole poses first,
solves insertion at those poses with the ManiSkill motion-planning expert, and
keeps only source trajectories whose final simulator state is inserted.

Renderer/eval helper scenario lists now recognize the new large-motion
families:

- `hole_move_stop_large`
- `hole_constant_large`
- `hole_reverse_large`
- `hole_sine_large`
- `hole_continuous_insert_large`
- `hole_late_shift_large`

No new fix3 SFT has started.

The normal full-episode SFT wrapper now has two guardrails for this reset:

- it refuses the invalid 6/6 full1000 source by default because the audit found
  failed final insertions and weak target-motion diversity;
- when a fix3 source/condition/output path is used with `RUN_SFT=true`, it
  requires an approval file containing `approved_for_sft=true` or
  `user_approved=true`.

This is a code-level backstop for the user's instruction that per-class videos
must be approved before SFT.

## Fix3 Smoke Evidence

The first smoke chain ran inside Slurm allocation `123499`, not on the login
node:

- generated 2 successful widened-final-pose expert source trajectories;
- replayed those trajectories into `pd_ee_delta_pose` action space with
  `2/2` successful replays;
- produced 6 dynamic overlay smoke trajectories, one for each fix3
  large-motion scenario.

Smoke output roots:

- `experiments/world_model_task_rebinding/cosmos3/fix3_final_pose_source_smoke_20260610_2`
- `experiments/world_model_task_rebinding/cosmos3/fix3_dynamic_overlay_smoke_20260610_6`

The strict smoke audit found no structural/source failures. Every sample has
`301` state frames, `300x7` actions, final insertion, and large target motion:

- `hole_move_stop_large`: `0.2394m`, first inserted near step `277`
- `hole_constant_large`: `0.2087m`, first inserted near step `282`
- `hole_reverse_large`: `0.3210m`, first inserted near step `277`
- `hole_sine_large`: `0.2857m`, first inserted near step `280`
- `hole_continuous_insert_large`: `0.2170m`, first inserted near step `285`
- `hole_late_shift_large`: `0.2852m`, first inserted near step `280`

This is only source/data smoke evidence. It does not satisfy the visual review
gate yet.

## Render Blocker And Resolution

Default-view video rendering initially failed on two nodes. Attempts on
allocation `123499` / `server32` failed with SAPIEN/Vulkan device loss at
`1024x1024` and then hung during first-frame rendering at `512x512`. Attempts
on allocation `125318` / `server13` were also unhealthy: torch CUDA
initialization failed and SAPIEN render-device creation failed even when CUDA
was hidden.

Allocation `125385` / `server35` passed the render canary and was used for the
successful fix3 smoke and review rendering. The failed node attempts are
scheduling/rendering evidence only; they are not counted as data success or
method failure.

The separate tmux-held 1-H200 render allocation request `125351` later started
on `server38`. No SFT was started from it while the user approval gate is
closed.

## Rejected Fix3 Review60 Overlay Package

The first per-scenario human-review package was rendered, but the user rejected
it on 2026-06-10. It is historical negative evidence only and must not be used
for SFT.

Source root:

`experiments/world_model_task_rebinding/cosmos3/fix3_final_pose_source_review60_20260610`

The source generator accepted `60/127` widened-final-pose expert attempts with
true final insertion. Replaying to `pd_ee_delta_pose` saved `57/60`
successful demos; the three replay failures were excluded from the dynamic
overlay review package.

Dynamic overlay root:

`experiments/world_model_task_rebinding/cosmos3/fix3_dynamic_overlay_review60_20260610_6x10`

Strict source audit:

- `num_paths=60`
- six scenarios with exactly `10` paths each
- `failures=[]`
- target-motion norm min/mean/max: `0.188646` / `0.281455` / `0.378151` m
- first insertion between steps `165` and `294`
- every inspected H5 keeps the full contract: `301` state frames and `300x7`
  actions

Render root:

`experiments/world_model_task_rebinding/cosmos3/fix3_dynamic_overlay_review60_render_server35_20260610_6x10`

Render and video-index evidence:

- `60` rendered videos
- `10` videos per scenario for:
  `hole_move_stop_large`, `hole_constant_large`, `hole_reverse_large`,
  `hole_sine_large`, `hole_continuous_insert_large`, and
  `hole_late_shift_large`
- ManiSkill default human-render camera, `1024x1024`, `30 fps`, `301` frames
- approval index:
  `fix3_review60_approval_index.md` and
  `fix3_review60_approval_index.json`
- OpenCV verification found all `60/60` videos openable with `301` frames,
  `30 fps`, `1024x1024`, existing review sheets, and no strict failures

The agent opened one representative review sheet from each fix3 large-motion
family. The later user review found that this was not sufficient: the videos
show the wrong task semantics and possible visual insertion into the block wall.

Rejection reasons:

- target motion starts at the beginning; the intended dynamic event is after
  the robot has picked up the peg and is nearly aligned with the current hole;
- visual insertion can look like the peg goes into the block wall rather than
  into the side hole, so the physical correctness gate is not satisfied;
- the construction is still an overlay/amplitude change rather than a proper
  fix2-style late-rebinding demonstration;
- the target motion is too slow and does not create a sharp prediction/handoff
  challenge;
- the target must not simply move into the peg; the robot must have to redirect
  or aim at the future target pose.

This package is not a user-approved data source. It does not prove Cosmos3 SFT,
controller handoff, full training-scale data completion, or valid fix3 source
quality.

## Corrected Fix3 Boundary

The corrected rebuild must be based on the fix2/full-episode physical data
logic, not on a post-hoc kinematic overlay that changes object slots after the
robot/peg trajectory is already fixed.

Required corrected behavior:

- the target remains static while the robot grasps the peg and approaches the
  current hole;
- the target begins moving only at a late trigger point, when the peg is held
  and near pre-insertion alignment;
- the final target pose remains inside the static DP/expert reachable domain
  so final insertion is physically possible;
- the target motion is fast and hard enough to break the original static DP,
  but the block must not simply come to the peg;
- success filtering must use the real ManiSkill episode success plus a stricter
  visual/geometric insertion check that rejects apparent wall insertion;
- render `10` videos per corrected late-trigger scenario and stop again for
  user approval before any SFT.

## Corrected Late-Trigger Review60 Evidence

Corrected source generator:

`scripts/world_model/generate_cosmos3_fix3_late_trigger_dynamic_experts.py`

The corrected generator does not post-hoc overlay object motion onto a
completed robot trajectory. It records real ManiSkill episodes: the robot
grasps the peg, moves near the current hole, then the target block moves
quickly inside the live episode. The expert then replans to the final reachable
target pose and inserts there. Sources are rejected unless real episode success
and stricter peg-head-in-hole geometry both pass.

Implementation fixes found during smoke:

- `RecordEpisode._elapsed_record_steps` is cumulative across episodes, so the
  corrected trigger/raw-step metadata uses episode-relative counts.
- `transforms3d.quat2axangle` returns `axis, angle`; the action-label export was
  fixed before accepting the smoke/review H5s.

Smoke root:

`experiments/world_model_task_rebinding/cosmos3/fix3_late_trigger_smoke2_20260610_6`

Smoke result:

- one sample per corrected scenario;
- `audit_failures=[]`;
- target-motion norm range `0.1038-0.1574m`;
- trigger range `66-97`;
- move duration range `12-31`;
- first insertion range `129-173`;
- rendered on `server35` under `render_512_server35`;
- agent opened all six smoke review sheets and verified late target motion plus
  final side-hole insertion.

Review60 source root:

`experiments/world_model_task_rebinding/cosmos3/fix3_late_trigger_review60_20260610_6x10`

Review60 source audit:

- accepted `60/99` physical attempts;
- six corrected scenarios with exactly `10` sources each:
  `hole_late_move_stop`, `hole_late_constant`, `hole_late_reverse`,
  `hole_late_sine`, `hole_late_continuous_insert`, and
  `hole_late_fast_shift`;
- `failures=[]`;
- all accepted sources keep the full `301` state-frame / `300x7` action-label
  contract;
- target-motion norm min/mean/max:
  `0.0935517` / `0.1296621` / `0.1689105` m;
- trigger min/max: `62` / `159`;
- move duration min/max: `7` / `52` steps;
- first insertion min/max: `127` / `240`.

Review60 render root:

`experiments/world_model_task_rebinding/cosmos3/fix3_late_trigger_review60_20260610_6x10/render_512_server35`

Render evidence:

- `60` rendered videos and `60` review sheets;
- exactly `10` videos per corrected scenario;
- ManiSkill3 `PegInsertionSide-v1` default human-render camera
  `look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])`, `fov=1`;
- `512x512`, `30 fps`, `301` frames, `10.033s`;
- review index: `review_index.md`;
- per-scenario overview sheets: `overview_sheets/`.

Visual inspection performed by the agent:

- opened all six smoke review sheets;
- opened six scenario overview sheets, each summarizing the `10` corrected
  demos at frames `55`, `109`, and `300`;
- did not observe the rejected patterns: target motion is not visible from the
  first frame, the visible target shift occurs after peg pickup/near alignment,
  and final frames show side-hole insertion or the peg hidden by the block at
  the side hole rather than obvious wall insertion.

Rendering resource note:

- Allocation `125351` on `server38` generated corrected H5 data successfully
  but failed rendering: SAPIEN saw the H200, env creation/reset worked, yet
  `render_rgb_array("render_camera")` hung even for a `256x256` canary.
- New tmux-held allocation `125516` on `server35` passed
  `render_min_canary.py` and rendered the corrected smoke/review60 package.

## 2026-06-11 Invalidation Of Late-Trigger Review60

The late-trigger review60 package above is rejected by user visual review and
must not be used for SFT or method evidence.

Concrete user-identified failures:

- `0001_hole_late_constant_seed974001_idx0001.fix3_traj_0.mp4`
- `0002_hole_late_reverse_seed974002_idx0002.fix3_traj_0.mp4`

Both were judged to insert into the block side/wall rather than strictly into
the visible hole, and the trigger/alignment timing did not match the successful
fix2-style "about to insert" protocol. Direct inspection of the first review
sheet confirmed that the final frames look like side/wall insertion, despite
the H5 `has_peg_inserted`/peg-head metrics passing. A Slurm-side state
diagnostic showed why the previous gate was insufficient: for the rejected
samples the peg head, center, tail, and peg axis can be numerically aligned to
the hole frame while the default-view visual still fails the intended physical
protocol.

Additional implementation issue found during the correction: the rejected
generator allowed final target box positions outside the original static
PegInsertionSide target support. Defaults were
`x=[-0.22,0.22]`, `y=[0.06,0.56]`, while the environment initializes boxes in
approximately `x=[-0.05,0.05]`, `y=[0.20,0.40]`. The rejected samples include
`0002` with final `y=0.4257` and `0001` outside the x lower bound. This
violates the requirement that the challenge come from late motion/prediction,
not from a final insertion pose outside the static expert domain.

Code repair started in
`scripts/world_model/generate_cosmos3_fix3_late_trigger_dynamic_experts.py`:

- final target defaults are restricted to the original static box-pose support;
- trigger now occurs only after official pre-insertion refinement and a strict
  peg-head/peg-center YZ preinsert gate;
- the final repaired state restores the official/fix2 insertion protocol:
  refined pre-insert alignment followed by the official `0.05m` insertion
  target, with real ManiSkill episode success as the live success authority.

Replacement data must be regenerated, rendered, inspected, and user-approved
before any fix3 SFT.

## User Approval Gate

Latest user instruction: after fix3 data construction, render `10` videos per
scenario for human inspection and stop before SFT. The correct state at that
point is a blocked approval gate, not automatic training. Cosmos3 SFT may start
only after the user explicitly approves the rendered demos.

Current gate state: the previous late-trigger `6x10` rendered review package,
the Qwen-approved replacement package, and the later constrained-insert v8
smoke are all rejected. No fix3 Cosmos3 SFT has been started. The next valid
step is to reproduce the successful fix2/official ManiSkill insertion protocol
as a small smoke before increasing target displacement again.

## 2026-06-11 Replacement Official-Insert Smoke/VLM Gate

After the user rejected the previous late-trigger package for visible
side/wall insertion, the generator was repaired to restore the official/fix2
insertion protocol: refined pre-insert alignment, real target motion only
after the peg is held and near the current hole, final reachable target pose in
the static expert support, and final `0.05m` official insertion judged by real
ManiSkill episode success.

User inspection rule for this gate changed: every video must be reviewed
directly by a VLM from the mp4, not by flattened contact sheets. The pass
criteria require late target motion, pre-motion peg/hole alignment, visible
target movement, visible red peg-head entry into the hole, no side/wall
insertion, final insertion, and peg grasp retention.

Six-video smoke:

- source/render root:
  `experiments/world_model_task_rebinding/cosmos3/fix3_late_trigger_smoke6_20260611_official_insert_6`;
- render: `512x512`, `30 fps`, `301` frames, approved ManiSkill default
  human-render camera;
- direct-video VLM result:
  `render_512_server35/vlm_video_review_qwen25_7b_v3/summary.json`;
- model:
  `/public/home/yanhongru/ICLR2027/predictive-interruption-v2/checkpoints/Qwen2.5-VL-7B-Instruct`;
- result: `6/6` pass, `all_pass=true`.

Sixty-video approval package:

- source root:
  `experiments/world_model_task_rebinding/cosmos3/fix3_late_trigger_smoke60_20260611_official_insert_vlm_gate`;
- source audit: `num_paths=60`, exactly `10` demos per scenario,
  `failures=[]`, target-motion norm min/mean/max
  `0.0901/0.1137/0.1665m`, trigger min/max `72/164`, move duration
  min/max `6/51`, first insertion min/max `138/248`;
- render root:
  `experiments/world_model_task_rebinding/cosmos3/fix3_late_trigger_smoke60_20260611_official_insert_vlm_gate/render_512_server35`;
- render manifest: `60` videos, `512x512`, `30 fps`, `301` frames,
  approved ManiSkill default camera;
- direct-video VLM result:
  `render_512_server35/vlm_video_review_qwen25_7b/summary.json`;
- model:
  `/public/home/yanhongru/ICLR2027/predictive-interruption-v2/checkpoints/Qwen2.5-VL-7B-Instruct`;
- result: `60/60` pass, `all_pass=true`.

This section is now historical only. On 2026-06-11 the user rejected the
package by direct video inspection and identified failures that the local
Qwen2.5 direct-video gate missed: initial penetration, misalignment, and
side/wall insertion. The package must not be used for SFT or method evidence.
No fix3 SFT was started from it.

The later constrained-insert v8 replacement after that rejection is also
rejected by user direct video inspection. It is documented as historical
negative evidence in
`docs/world_model_task_rebinding/2026-06-11_fix3_physics_gate_constrained_insert_smoke6.md`.
The active data action is fix2/official insertion protocol reproduction first,
then careful late-motion enlargement only after physical video verification.
