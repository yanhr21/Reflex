# V7 DP Full1000 Resume After Hard-Teacher Deferral

Date: 2026-06-12.

Latest user instruction: defer the hard-teacher supplement and return to the
previous approved v7 DP data-generation chain. The active target is the
original `full1000` source set, not the 1500-row supplement.

Current corrected live merge:

- merge root:
  `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_full1000_live_merge_20260612_status_now`
- total selected: `723/1000`
- duplicate scenario/seed count: `28`
- the merge includes the approved v7 complete-nine
  `fix3_h5_paths.txt`, because that combined review root references H5 files
  stored under earlier smoke/search roots
- selected counts:
  `move_stop=43/70`, `constant=48/90`, `reverse=97/100`,
  `sine=60/90`, `continuous_insert=95/120`, `fast_shift=105/120`,
  `none=160/160`, `peg_drop=113/150`, `peg_disturb=2/100`
- missing counts:
  `27/42/3/30/25/15/0/37/98` in the same scenario order

Hard-teacher smoke attempts were stopped by interrupting the generator process;
no Slurm allocation was cancelled for that stop. The hard-teacher script stays
as later work only.

Active resume jobs:

- `126223`, tmux `cosmos3_fix3_v7_resume_nonpeg_a_126223`:
  4-GPU non-peg bundle for constant, continuous, sine, and move-stop.
- `126210`, tmux `cosmos3_fix3_v7_resume_mixed_peg_a_126210`:
  4-GPU bundle for reverse, fast-shift, peg-drop, and peg-disturb.
- `126219`, tmux `cosmos3_fix3_v7_resume_pegdist_b_126219`:
  1-GPU peg-disturb supplement with independent policy RNG seeds.
- `126174`, tmux `cosmos3_fix3_v7_resume_pegdrop_b_126174`:
  1-GPU peg-drop supplement with independent policy RNG seeds.
- `126175`, tmux `cosmos3_fix3_v7_resume_nonpeg_b_126175`:
  1-GPU mixed non-peg supplement.

New helper wrapper:
`scripts/slurm/run_fix3_v7_resume_full1000_bundle_20260612_in_allocation.sh`.

Merge helper update:
`scripts/world_model/merge_fix3_full1000_unique_h5.py` now accepts
`--source-paths-file` so approved combined-review path lists can be included
without copying or moving H5 files.

Early startup evidence:

- all five held allocations entered running `bash` steps;
- the 4-GPU bundles reached the v7 generator and loaded the DP checkpoint;
- early accepted rows were observed for `hole_late_reverse` and `peg_drop`;
- `peg_disturb` was still rejecting early attempts with `final_insert_failed`
  and must be monitored for real accepted rows rather than assumed solved.

Monitoring update:

- after adding explicit `--source-paths-file` support, the live merge reached
  `647/1000`;
- after the first resume window, it reached `677/1000`;
- a fresh status merge at
  `fix3_original_protocol_large_motion_dp_v7_full1000_live_merge_20260612_status_now`
  reached `723/1000`;
- selected counts at `677/1000`:
  `move_stop=43/70`, `constant=47/90`, `reverse=85/100`,
  `sine=57/90`, `continuous_insert=92/120`, `fast_shift=104/120`,
  `none=160/160`, `peg_drop=87/150`, `peg_disturb=2/100`;
- selected counts at `723/1000`:
  `move_stop=43/70`, `constant=48/90`, `reverse=97/100`,
  `sine=60/90`, `continuous_insert=95/120`, `fast_shift=105/120`,
  `none=160/160`, `peg_drop=113/150`, `peg_disturb=2/100`;
- `peg_disturb` has one new accepted row from
  `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260612_resume_pegdist_b_seedbase40751000`,
  but remains the clear bottleneck.

Next gate: once the running roots finish or enough rows accumulate, rerun
`scripts/world_model/merge_fix3_full1000_unique_h5.py` over v7 roots only.
If it reaches exactly `1000/1000`, run
`scripts/world_model/audit_fix3_merged_source_h5.py`, render per-class review
videos/framebooks, and stop for user approval before WAM export or Cosmos3
SFT.
