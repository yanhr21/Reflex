# Fix3 1500 Hard-Teacher Supplement

Date: 2026-06-12.

Status update later on 2026-06-12: this direction is deferred by the latest
user instruction. The active near-term chain is v7 DP-generated `full1000`.
Do not use this note as permission to block the current data/SFT path on
hard-teacher generation.

Current retained v7 count is `630/1500` unique H5 candidates. These rows stay
in the final candidate pool.

The target total is now `1500`, preserving the original class proportions:

| scenario | target | retained | supplement |
| --- | ---: | ---: | ---: |
| `hole_late_move_stop` | 105 | 41 | 64 |
| `hole_late_constant` | 135 | 46 | 89 |
| `hole_late_reverse` | 150 | 76 | 74 |
| `hole_late_sine` | 135 | 53 | 82 |
| `hole_late_continuous_insert` | 180 | 88 | 92 |
| `hole_late_fast_shift` | 180 | 99 | 81 |
| `none` | 240 | 161 | 79 |
| `peg_drop` | 225 | 66 | 159 |
| `peg_disturb` | 150 | 0 | 150 |

Supplement total: `870`.

New generator:
`scripts/world_model/generate_cosmos3_fix3_hard_dynamic_teacher.py`.

It keeps retained v7 rows, then fills the supplement with
`source_kind=hard_dynamic_teacher` rows. Moving-hole rows use a motion-planning
teacher to grasp/pre-align, inject late target motion, replan to the moved
hole, and insert. `none` rows are static teacher controls. `peg_drop` and
`peg_disturb` rows perturb/drop the peg after pre-insertion alignment, then use
the teacher to regrasp and insert. This avoids defining new positives as
"frozen DP happened to solve it."

No WAM export, render expansion, controller run, or Cosmos3 SFT may start until
the 1500-row set is merged, audited, visually reviewed, and approved.
