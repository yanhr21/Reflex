# Targeted Recovery Supplement Plan

## Boundary

This is a proposed data-boundary change after the 733-only dense export failed
live-query coverage. It must not run without explicit user approval.

It does not replace the accepted 733 rows. It adds targeted hard-teacher rows
only for the states that the current 733 distribution does not cover.

## Evidence That Motivates It

Latest 733-only late-prefix diagnostic:

- condition root:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050`
- preflight root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_preflight_20260614_130050`
- rows: `9271`
- strict `301/300` preflight: pass
- action/sidecar audit: pass
- live-query coverage: fail, `58/173` undercovered

Gap manifest:

`experiments/world_model_task_rebinding/cosmos3/targeted_recovery_gap_manifest_20260614_from_late299.json`

The remaining gaps are:

- `25` `hole_late_sine / target_post_motion`
- `14` `hole_late_constant / target_post_motion`
- `11` `hole_late_continuous_insert / target_post_motion`
- `8` `hole_late_fast_shift / peg_recovery`

Plain meaning: the model is queried in states where the target has moved and
the peg is still physically off the moved hole. The current 733 successful
source rows do not provide enough local examples of those recovery states.

## 2026-06-14 Attempt Result

The approval-gated supplement path was launched inside held Slurm allocation
`127723` and stopped with Ctrl-C in tmux after repeated zero-acceptance pilots.
No login-node compute or `sbatch` was used.

Results:

- initial 112-row wrapper attempt:
  `targeted_recovery_supplement_after_approval_20260614_131231`,
  `57` attempts, `0` accepted;
- motion-focused 8-row pilot:
  `targeted_recovery_supplement_after_approval_20260614_131700_pilot8`,
  `22` attempts, `0` accepted;
- old-hole preinsert retreat pilot:
  `targeted_recovery_supplement_after_approval_20260614_132100_pilot8_retreat`,
  `23` attempts, `0` accepted;
- retreat plus initial wait-gate pilot:
  `targeted_recovery_supplement_after_approval_20260614_132400_pilot8_waitgate`,
  `28` attempts, `0` accepted.

Interpretation:

The current scripted hard-teacher generator is the blocker. With the peg held
near the old hole, target motion often sweeps the moved block wall through the
peg line. Retreating the peg before target motion avoids some swept-wall
failures, but then the scripted replan/final insertion still often violates the
strict final line gate or wall-risk check. This is a physical/teacher-geometry
failure before any RGB render, Cosmos SFT, or overfit training.

Do not rerun these same variants. A useful next data action needs a different
teacher design or an explicit user-approved boundary change.

## Proposed Supplement

Generate a small targeted hard-teacher supplement, not a new full dataset:

- `hole_late_sine=40`
- `hole_late_constant=24`
- `hole_late_continuous_insert=24`
- `hole_late_fast_shift=24`

Total: `112` candidate H5 rows.

For `hole_late_fast_shift`, enable the new optional
`post_motion_release_regrasp_scenarios=hole_late_fast_shift` path. That creates
moving-hole rows where the teacher briefly releases/regrasps after target
motion, so the export can produce real `peg_recovery` physical-mode prefixes
instead of relying on unrelated `peg_drop` rows.

## Run Command After Approval

Historical command shape, kept for provenance only. Do not rerun the same
variant without a new fix. Run only inside the held tmux allocation, not on the
login node:

```bash
srun --overlap --jobid=127723 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --mem=120G \
  bash -lc 'cd /public/home/yanhongru/ICLR2027/Reflex && \
  ALLOW_TARGETED_RECOVERY_SUPPLEMENT=true \
  bash scripts/slurm/run_cosmos3_targeted_recovery_supplement_after_approval_in_allocation.sh'
```

The wrapper stops after:

1. hard-teacher H5 generation;
2. RGB render at `512x512`, `30 fps`;
3. review sheet creation;
4. structural supplement inspection:
   `scripts/world_model/inspect_cosmos3_targeted_recovery_supplement.py`.

It does not merge into the 733 source, export WAM conditions, or start SFT.

The inspection must pass before visual review. It checks:

- exactly `112` supplement rows;
- scenario quotas match the proposed gap-focused counts;
- every H5 has `301` state/video frames and `300` action steps;
- source kind is
  `hard_dynamic_teacher_targeted_recovery_gap_20260614`;
- `hole_late_fast_shift` rows record the post-motion release/regrasp phase;
- rendered RGB manifest, JSONL records, MP4 paths, `512x512` size, and `30 fps`
  metadata are consistent;
- at least the configured review sheets exist.

Passing this inspection only means "open the review sheets now." It does not
approve merge/export/SFT. The checker writes `ready_for_merge=false` unless a
later visual approval marker is provided.

## Visual Gate

Before any merge or SFT:

- open the generated review sheets;
- verify the camera is the approved ManiSkill default-style view;
- verify target motion is visible but not absurdly out of distribution;
- verify the peg is visibly held or visibly regrasped as intended;
- verify the final frames show real insertion, not only a metric flag;
- reject any row whose visual result contradicts the H5 success metric.

If the supplement looks too visually biased or unnatural, do not merge it.

## Next Gate After Visual Approval

Only after visual approval:

1. merge accepted supplement rows with the frozen 733 source while preserving
   provenance;
2. render/manifest the merged source as RGB SFT data;
3. export full-episode WAM conditions with dense receding prefixes;
4. rerun strict preflight and live-query coverage;
5. run short overfit only if the preflight is ready;
6. start full 4-GPU SFT only if coverage and overfit both pass.

## What Would Falsify This Direction

- Hard-teacher rows frequently fail strict insertion.
- Rendered videos show unnatural target jumps, off-camera motion, dropped peg
  not recovered, or metric-only success.
- The merged coverage audit still leaves the same `target_post_motion` or
  `peg_recovery` gaps.
- Short overfit fails after the merged preflight passes.

In those cases, do not train full SFT. Report the concrete failure evidence.
