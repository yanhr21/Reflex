# 2026-06-11 Original Dataset Reset After Fix2 Repro Penetration

## Boundary

The fix1/fix2/fix3 repro and modified generation roots are untrusted after
user visual review found penetration and wall-insertion failures. The active
physical baseline is reset to the originally effective dataset/protocol:

`experiments/world_model_task_rebinding/cosmos3/sft_dataset_full1000_maniskill_default_regen_20260606_0055`

The intended behavior to preserve from that source is: target static at the
start, peg is grasped and aligned first, target moves after alignment, and the
episode ends in true physical insertion.

The original generation environment, generator code, source H5 tree, and
rendered dataset root are read-only baseline assets. The rebuild must not
modify them in place. Any change to target-motion amplitude, acceptance, or
sampling must be made in a copied fix3-specific generator/config, with new
outputs written under a new experiment root.

## Required Rebuild

The old source has two known problems for fix3 training: target displacement is
too small, and many rendered rows have `inserted_end=false`. The rebuild must
therefore generate or resample from the original physical protocol until every
accepted sample has:

- true final insertion from the simulator/physical gate;
- no visible penetration, wall insertion, or self-drilling;
- target motion after peg alignment, not at episode start;
- larger target displacement and continuous-motion cases that require
  prediction/rebinding;
- 301 RGB/state frames and 300 action steps.

If the physical gate rejects a trajectory or final insertion fails, that sample
is failed and regenerated. No state projection or forced insertion is allowed.

## Current Action

Audit the original source H5s and manifests, then implement an allocation-only
small smoke generator from the original protocol. Render the smoke videos and
stop for user review before any SFT.

## Read-Only Audit Result

Audit script:

`scripts/world_model/audit_original_full1000_readonly.py`

Output:

`experiments/world_model_task_rebinding/cosmos3/original_full1000_readonly_audit_20260611`

The audit ran inside Slurm allocation `125642` and read `1000/1000` source H5s
with no read failures. It did not modify the source H5 tree, original dataset
root, or original generation code.

Final inserted counts:

- `hole_constant`: `33/167`
- `hole_move_stop`: `21/167`
- `hole_reverse`: `125/167`
- `none`: `104/166`
- `peg_disturb`: `2/166`
- `peg_drop`: `43/167`

Moving-hole target displacement in the original source is small:

- `hole_constant`: mean `0.1295m`
- `hole_move_stop`: mean `0.1400m`
- `hole_reverse`: mean `0.0910m`

This confirms the user boundary: the original protocol is the physical
baseline, but the original rows are not directly usable as fix3 SFT data
because many do not end inserted and the moving-hole range is too small.

## Active Workspace Cleanup

Rejected fix2 roots were archived under:

`/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate/fix2_chain_user_rejected_after_original_reset_20260611`

Old fix1/fix2-era SFT, overfit, and readout roots were archived under:

`/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate/old_sft_and_readout_untrusted_after_original_reset_20260611`

After cleanup and the copied smoke rebuild, the active cosmos3 experiment
directory retains the original 6/6 full1000 dataset, the read-only audit,
render canaries, and the current copied fix3 smoke outputs described below.

## Copy-On-Write Fix3 Generator

The large-motion rebuild is implemented in a copied fix3-specific script:

`scripts/world_model/generate_cosmos3_fix3_original_protocol_large_motion_dp.py`

The script reconstructs the original DP-driven late-target protocol from the
H5 provenance instead of editing the original generation chain. It refuses to
write into the original 6/6 full1000 dataset root. The original full1000 data,
source H5 tree, generation environment, and original generator remain read-only
baseline assets.

The only renderer-side change in the active code is a metadata/provenance bug
fix: late-trigger samples are now marked `triggered=true` if any timestep is
triggered, not only if frame 0 is triggered. This does not modify trajectories
or relax physical gates.

## Smoke6 V1 Rejected For Timing And Motion Scale

Smoke root:

`experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_smoke6_20260611_v1`

Render root:

`experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_smoke6_20260611_v1/render_512_server56`

The smoke was generated and rendered inside Slurm allocation `125642` on
`server56`. No rollout or render task ran on the login node.

The six accepted records all have `success_at_end=true`, `301` state/source
frames, `300` action steps, and target motion between `0.1807m` and `0.1918m`.
The rendered videos use the approved ManiSkill default human camera at
`512x512`, `30 fps`, and `301` frames.

- `0000 hole_late_move_stop`: seed `1012018`, motion `0.1860m`, trigger `90`,
  first insertion `172`
- `0001 hole_late_constant`: seed `1012053`, motion `0.1807m`, trigger `90`,
  first insertion `161`
- `0002 hole_late_reverse`: seed `1012056`, motion `0.1845m`, trigger `90`,
  first insertion `240`
- `0003 hole_late_sine`: seed `1012089`, motion `0.1918m`, trigger `90`,
  first insertion `210`
- `0004 hole_late_continuous_insert`: seed `1012093`, motion `0.1819m`,
  trigger `61`, first insertion `109`
- `0005 hole_late_fast_shift`: seed `1012113`, motion `0.1817m`, trigger `90`,
  first insertion `201`

Agent visual check opened dense frame grids for all six videos, but user direct
video review on 2026-06-11 is authoritative. The user reported that insertion
quality is acceptable, but V1 is still rejected because:

- `0002_hole_late_reverse...mp4` and
  `0005_hole_late_fast_shift...mp4` begin target motion before peg pickup.
- `0001_hole_late_constant...mp4`, `0003_hole_late_sine...mp4`, and
  `0004_hole_late_continuous_insert...mp4` have target motion that is too
  small and too slow.

V1 is not approved for 60-video generation or Cosmos3 SFT.

## Complete9 Patch Boundary

The fix stays inside the copied fix3 generator. The original 6/6 full1000
dataset, source H5 tree, original environment, and original generator remain
read-only.

The first replacement attempt after V1 incorrectly replaced the six previous
moving-hole variants with the six original top-level classes. That was wrong:
the smoke taxonomy must be additive. The wrong `v2_smoke6` and
`v3_complete_smoke6` partial outputs were archived under:

`/public/home/yanhongru/ICLR2027/archived/reflex_bad_fix3_20260611_user_rejected_physics_gate/wrong_taxonomy_smoke6_replaced_by_complete9_20260611`

Complete9 changes:

- fallback time alone cannot trigger target motion;
- target motion requires stable peg hold/lift, a delay after first robust hold,
  and preinsert geometry;
- fallback is only a relaxed preinsert geometry fallback after robust hold, not
  an unconditional step-90 trigger;
- moving-target schedules complete larger displacement in short windows rather
  than spreading motion over the remaining episode;
- default target displacement is raised to `0.22-0.30m`.
- smoke classes are the previous six moving-hole variants plus the original
  static/peg classes: `hole_late_move_stop`, `hole_late_constant`,
  `hole_late_reverse`, `hole_late_sine`, `hole_late_continuous_insert`,
  `hole_late_fast_shift`, `peg_disturb`, `peg_drop`, and `none`.

The partial v4 `current6` render is rejected by user direct video review. The
observed failures are:

- `hole_late_move_stop` and `hole_late_fast_shift` move too fast and collide
  into the target;
- `hole_late_constant` and `hole_late_sine` show the target moving toward the
  peg / self-inserting, which would let the original policy stay near the old
  peg trajectory instead of requiring rebinding.

The copied generator is patched for v5 only: `move_stop` and `fast_shift`
motion windows are doubled to reduce speed by 50%, and a counterfactual
anti-self-insert path gate rejects moving-hole candidates where the current
peg head would already align with any sampled future target path pose at the
trigger. This is a data-quality gate, not an evaluation relaxation.

V5 also adds sampling hygiene for the added static/peg classes. The full smoke
taxonomy remains nine classes, but `none`, `peg_drop`, and `peg_disturb` first
try seed/trajectory indices that the read-only original audit already found to
end inserted. These candidates still must pass the current live final-success
and slot-insertion gates. The generator now writes partial paths and a partial
manifest after each accepted sample, so accepted smoke videos can be rendered
for inspection even if a later low-success class stalls.

V5b adds a continuation-only CLI (`scenario_sequence` and
`accepted_index_offset`) so already accepted smoke rows can be kept while
remaining classes are generated in a separate root and then rendered from a
combined paths file. This preserves the nine-class smoke requirement. The
anti-self-insert hard YZ floor is set to `0.055m` while retaining the `2.4x`
hole-radius multiplier; this keeps rejecting target paths that run into the
current peg, but avoids rejecting borderline paths whose sampled future hole
is already more than about 5.5 cm away from the peg-hole centerline.

The next evidence gate is a new nine-scenario Complete9 smoke render. It must be
generated and rendered inside Slurm, then stopped for user direct video review
before any 60-video package or SFT.

## V5b Complete9 Smoke Render

Generated inside Slurm allocation `125865` on `server56`; no rollout or render
ran on the login node. The final smoke is a combined nine-class paths file:

`experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v5b_complete9_combined_20260611/fix3_h5_paths.txt`

Source roots:

- first four moving-hole rows:
  `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v5b_complete9_no_self_insert_probe_delta_smoke_20260611`;
- remaining five rows:
  `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v5b_remaining5_no_self_insert_probe_delta_20260611`.

Rendered review root:

`experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v5b_complete9_combined_20260611/render_512_server56`

Render structure check: `9` videos, each `301` frames, `30 fps`, `512x512`,
default ManiSkill human camera. Scenarios are `hole_late_move_stop`,
`hole_late_constant`, `hole_late_reverse`, `hole_late_sine`,
`hole_late_continuous_insert`, `hole_late_fast_shift`, `none`, `peg_drop`,
and `peg_disturb`. All accepted source rows have final `inserted_end=true` in
the render manifest/task-state metadata. This is still only smoke evidence:
user direct-video approval is pending, and no 60-video package or SFT should
start before that approval.

Agent all-frame review:

`docs/world_model_task_rebinding/2026-06-11_v5b_complete9_smoke_frame_review.md`

This v5b package is now superseded and must not be used as approval evidence.
The later v7 pass added a stricter post-motion self-insert gate and replaced
the weak `move_stop` / `fast_shift` rows.

## V7 Complete9 Slow No-Self-Insert Smoke Render

V7 generator patch:

`scripts/world_model/generate_cosmos3_fix3_original_protocol_large_motion_dp.py`

The patch remains copy-on-write and does not modify the original 2026-06-06
full1000 baseline. It adds two moving-hole rejection checks: a counterfactual
future-path gate at the trigger-time peg head, and a post-motion self-insert
gate that rejects cases where the target has moved substantially but TCP
rebinding is too small before first insertion.

Combined root:

`experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611`

Rendered root:

`experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/render_512_server21`

Evidence note:

`docs/world_model_task_rebinding/2026-06-11_v7_complete9_slow_no_self_insert_smoke_render.md`

The H5 audit has `failures=[]` for nine classes, all rows have `301` frames,
`300` actions, and `inserted_end=true`. The render audit has `failures=[]` for
nine openable `512x512`, `30 fps`, `301`-frame videos. Moving-hole first
insertions are after each target-motion window, and TCP motion before
insertion is at least `0.149m`.

This is still only a smoke approval candidate. Full-scale generation and SFT
remain blocked pending user direct-video review.
