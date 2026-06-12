# Fix3 V7 Full1000 Generation Status

Date: 2026-06-11.

This note records source-data generation state only. It is not Cosmos3 SFT
evidence, controller evidence, or proof of the downstream method.

## Current Objective

Generate a complete fix3 V7 full1000 source set under the strict 300-step
episode contract: `301` RGB/state frames, `300` actions, final insertion for
every accepted row, late target motion after peg grasp/pre-insertion alignment,
and no forced state projection or penetration. The final combined set still
must pass source audits, rendered inspection, full-episode condition preflight,
and user approval before WAM export or Cosmos3 SFT.

Target nonuniform quotas remain:

`hole_late_move_stop=70`, `hole_late_constant=90`,
`hole_late_reverse=100`, `hole_late_sine=90`,
`hole_late_continuous_insert=120`, `hole_late_fast_shift=120`,
`none=160`, `peg_drop=150`, and `peg_disturb=100`.

## Latest Count And User Direction

Latest merged count reported by the user: `276/1000` unique H5 rows.

User correction: do not let peg/peg-disturb be the active bottleneck. Finish
the non-peg classes first; handle peg-drop/peg-disturb last.

## Resource Handling

Peg-focused local `srun` steps were stopped by interrupting/terminating the
local step processes only. The held Slurm allocations were preserved; no
`scancel` was used.

Preserved allocations repurposed to non-peg generation:

- `126127` on `server52`
- `126089` on `server24`
- `126125` on `server03`
- `126052` on `server38`

Allocation-side syntax check passed before restart:

`py_compile` passed for
`scripts/world_model/generate_cosmos3_fix3_original_protocol_large_motion_dp.py`
and `scripts/world_model/merge_fix3_full1000_unique_h5.py`; `bash -n` passed
for `scripts/slurm/run_fix3_v7_approved_full1000_generation_in_allocation.sh`.

## New Non-Peg Buffer Roots

The following roots were started inside the preserved Slurm allocations:

- `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_cfs_aux330_seedshift4`
- `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_mrs_aux260_seedshift3`
- `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_cfs_aux330_seedshift5`
- `experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_mrs_aux260_seedshift4`

`cfs` roots generate `hole_late_constant`,
`hole_late_continuous_insert`, and `hole_late_fast_shift`. `mrs` roots
generate `hole_late_move_stop`, `hole_late_reverse`, and `hole_late_sine`.
These are supplemental work buffers; final merge must deduplicate and cap each
scenario at the approved quota.

Early log evidence: the `nonpeg_cfs_aux330_seedshift4` root accepted at least
one `hole_late_fast_shift` row with `target_motion_norm_m=0.2369` and
`first_insert_step=180`. Other early rows are being rejected by the physical
and anti-self-insert gates, which is expected filtering rather than method
evidence.

## Follow-Up Count

Later file-name-level count over the active/credible work-buffer roots reached
`280/1000` unique scenario/seed rows. This is a progress estimate from H5
filenames, not a substitute for structural H5 audit or video inspection.

Scenario counts at this check:

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 11 | 10 | 70 | 60 |
| `hole_late_constant` | 6 | 6 | 90 | 84 |
| `hole_late_reverse` | 20 | 19 | 100 | 81 |
| `hole_late_sine` | 16 | 14 | 90 | 76 |
| `hole_late_continuous_insert` | 5 | 5 | 120 | 115 |
| `hole_late_fast_shift` | 19 | 19 | 120 | 101 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 48 | 46 | 150 | 104 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

All nine held allocations had an active generation step at the check, so no
additional non-peg root was started from an idle allocation. The current
action is to let the active non-peg roots run through a meaningful window,
then recount. The `peg_disturb=0` state remains deferred by user instruction
until the non-peg classes are filled.

## Expanded Non-Peg Parallelism

A subsequent scan over all active, non-archived v7 full1000 group roots reached
`305/1000` unique scenario/seed rows. This count includes valid already
accepted `peg_drop` rows from the stopped pegmix root, but still leaves the
six non-peg moving-hole classes far from quota:

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 11 | 10 | 70 | 60 |
| `hole_late_constant` | 6 | 6 | 90 | 84 |
| `hole_late_reverse` | 22 | 21 | 100 | 79 |
| `hole_late_sine` | 16 | 14 | 90 | 76 |
| `hole_late_continuous_insert` | 6 | 6 | 120 | 114 |
| `hole_late_fast_shift` | 21 | 21 | 120 | 99 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

Because all nine existing held allocations were busy and the non-peg missing
count was still `512`, two additional tmux-managed 1-H200 allocations were
requested and started:

- `126174` on `server52`, running
  `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_cfs_aux330_seedshift6`
  for `hole_late_constant`, `hole_late_continuous_insert`, and
  `hole_late_fast_shift`.
- `126175` on `server38`, running
  `fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_mrs_aux260_seedshift5`
  for `hole_late_move_stop`, `hole_late_reverse`, and `hole_late_sine`.

Both new roots reached the generator after model load. Early log lines show the
same physical gates rejecting unsuitable rows (`no_large_final_target_in_bounds`,
`counterfactual_final_target_self_insert_gate_failed`, `final_insert_failed`,
and `never_triggered_after_robust_hold_and_preinsert`). This is expected source
filtering, not SFT evidence.

## Later Live Check

A later active-root filename scan reached `314/1000` unique scenario/seed rows.
This is still file-level progress only; it does not replace the required H5
structural audit, rendered review, or user approval.

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 11 | 10 | 70 | 60 |
| `hole_late_constant` | 7 | 7 | 90 | 83 |
| `hole_late_reverse` | 24 | 23 | 100 | 77 |
| `hole_late_sine` | 17 | 15 | 90 | 75 |
| `hole_late_continuous_insert` | 8 | 8 | 120 | 112 |
| `hole_late_fast_shift` | 24 | 24 | 120 | 96 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

The non-peg moving-hole missing count is `503`. Eleven held allocations have
active generation steps. Log checks found no `Traceback`, `CUDA error`,
`DeviceLost`, `OutOfMemory`, `No space`, or `srun: error` in the active non-peg
logs. Some older roots had older log mtimes but still had active `srun`
processes, so they were not interrupted; re-check on the next monitoring pass
before deciding they are stalled.

## Latest Live Count

The next active-root filename scan reached `323/1000` unique scenario/seed
rows. Recent H5 mtimes show current production from several non-peg roots, so
the held allocations were left running rather than interrupted.

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 11 | 10 | 70 | 60 |
| `hole_late_constant` | 7 | 7 | 90 | 83 |
| `hole_late_reverse` | 26 | 25 | 100 | 75 |
| `hole_late_sine` | 18 | 16 | 90 | 74 |
| `hole_late_continuous_insert` | 9 | 9 | 120 | 111 |
| `hole_late_fast_shift` | 29 | 29 | 120 | 91 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

The non-peg moving-hole missing count is now `494`. Recent nonzero roots
include `constant_cont_fast_aux330_seedshift1/2/3`,
`move_sine_aux160_seedshift2`, `nonpeg_cfs_aux330_seedshift4/5/6`, and
`nonpeg_mrs_aux260_seedshift3/4/5`.

## Subsequent Live Count

The next active-root filename scan reached `326/1000` unique scenario/seed
rows. This remains file-level progress only; structural and visual checks are
still required before approval.

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 11 | 10 | 70 | 60 |
| `hole_late_constant` | 7 | 7 | 90 | 83 |
| `hole_late_reverse` | 27 | 26 | 100 | 74 |
| `hole_late_sine` | 18 | 16 | 90 | 74 |
| `hole_late_continuous_insert` | 10 | 10 | 120 | 110 |
| `hole_late_fast_shift` | 30 | 30 | 120 | 90 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

The non-peg moving-hole missing count is `491`. Log scan still found no
`Traceback`, `CUDA error`, `DeviceLost`, `OutOfMemory`, `No space`, or
`srun: error` in the active non-peg logs, and no active non-peg root had a
`fix3_original_protocol_large_motion_done` marker. All eleven held allocations
remained running, so no allocation was interrupted or repurposed.

## Non-Peg Priority Reconfirmed

After the user reported the latest merged count as `276/1000`, the scheduling
priority was reconfirmed: leave `peg_drop` and especially `peg_disturb` for
last, and finish the non-peg moving-hole quotas first. A fresh active-root
filename scan saw `340/1000` unique rows already present across the
non-archived v7 group roots. The gap between `276` merged rows and `340`
active-root rows is expected because some accepted H5 files have landed in
work-buffer roots but have not yet been folded into the merged path list.

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 12 | 11 | 70 | 59 |
| `hole_late_constant` | 8 | 8 | 90 | 82 |
| `hole_late_reverse` | 28 | 27 | 100 | 73 |
| `hole_late_sine` | 19 | 17 | 90 | 73 |
| `hole_late_continuous_insert` | 12 | 12 | 120 | 108 |
| `hole_late_fast_shift` | 38 | 38 | 120 | 82 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

The non-peg moving-hole missing count is now `477`. Eleven 1-H200 allocations
remain running with generation steps. Active non-peg logs and tmux panes show
continued source filtering and accepted rows, not environment failure. The
dominant reject causes are still physical/source-quality gates such as
`final_insert_failed`, `counterfactual_final_target_self_insert_gate_failed`,
`no_large_final_target_in_bounds`, and
`never_triggered_after_robust_hold_and_preinsert`. These rejects preserve the
fix3 data-quality contract rather than relaxing the gate to inflate counts.

## Focused Non-Peg Resource Attempt

A later active-root filename scan reached `346/1000` unique rows with
`471` non-peg moving-hole rows still missing. The gap is concentrated in
`hole_late_constant`, `hole_late_continuous_insert`, `hole_late_move_stop`,
`hole_late_sine`, and `hole_late_reverse`, while existing CFS roots are
accepting `hole_late_fast_shift` more often. To avoid wasting attempts on a
class that may fill earlier, a focused 4-GPU bundle script was added:

`scripts/slurm/run_fix3_v7_nonpeg_focus4_bundle_in_allocation.sh`

The script only launches the existing approved v7 generator wrapper in four
parallel `gpu:1` steps inside a held allocation. It does not modify the
generator, source protocol, original data, physical gates, or SFT gate. The
four focused roots target `hole_late_continuous_insert`,
`hole_late_constant`, `hole_late_move_stop`, and
`hole_late_sine/hole_late_reverse`.

The script passed `bash -n` inside held Slurm allocation `126174`. A tmux
allocation request was started as
`cosmos3_fix3_full1000_nonpeg_focus4_4h200_0611`; Slurm job `126188` first
queued briefly, then started on `server42`.

Startup exposed an allocation-internal step-management issue: invoking
`salloc ... bash script` created an outer bash step, so non-overlap nested
`srun` calls could serialize. The approved generation wrapper was updated with
an optional `SRUN_EXTRA_ARGS` field, defaulting to unchanged behavior; the
focus bundle now passes `SRUN_EXTRA_ARGS=--overlap`. This preserves the
physical/data protocol and only changes how steps are scheduled inside a held
allocation. Both Slurm scripts passed `bash -n` inside allocation `126188`.

The running `126188` allocation now has the original focused
`hole_late_continuous_insert` root plus three additional `focus2` roots started
with overlap for `hole_late_constant`, `hole_late_move_stop`, and
`hole_late_sine/hole_late_reverse`. Early evidence confirms the new resources
are doing useful work: `nonpeg_focus_continuous...` accepted
`2` `hole_late_continuous_insert` rows and `nonpeg_focus2_move_stop...`
accepted `1` `hole_late_move_stop` row. Focus-root error scan found no
`Traceback`, `CUDA error`, `DeviceLost`, `OutOfMemory`, `No space`,
`srun: error`, `RuntimeError`, `Killed`, or `Segmentation`.

After including focused roots, the active-root filename scan reached
`366/1000` unique rows, with `451` non-peg moving-hole rows still missing:

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 17 | 16 | 70 | 54 |
| `hole_late_constant` | 8 | 8 | 90 | 82 |
| `hole_late_reverse` | 35 | 34 | 100 | 66 |
| `hole_late_sine` | 19 | 17 | 90 | 73 |
| `hole_late_continuous_insert` | 18 | 18 | 120 | 102 |
| `hole_late_fast_shift` | 46 | 46 | 120 | 74 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

## Focus Allocation Cleanup

A subsequent check found that the original pre-fix `focus4` bundle still had
three local non-overlap `srun` commands waiting on step creation for the
superseded roots `nonpeg_focus_constant...`, `nonpeg_focus_move_stop...`, and
`nonpeg_focus_sine_reverse...`. These were not producing data and had already
been replaced by the running `focus2` overlap roots. Only those local waiting
processes and their corresponding tee/wrapper processes were terminated by PID.
No `scancel` was used, allocation `126188` was preserved, and the useful
`nonpeg_focus_continuous...` step plus the three `focus2` steps kept running.

The next active-root filename scan reached `368/1000` unique rows, with
`449` non-peg moving-hole rows still missing:

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 17 | 16 | 70 | 54 |
| `hole_late_constant` | 8 | 8 | 90 | 82 |
| `hole_late_reverse` | 36 | 35 | 100 | 65 |
| `hole_late_sine` | 19 | 17 | 90 | 73 |
| `hole_late_continuous_insert` | 18 | 18 | 120 | 102 |
| `hole_late_fast_shift` | 47 | 47 | 120 | 73 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

Focus and non-peg logs still show physical-gate filtering and occasional
accepted rows, not environment failure. Error scans found no traceback, CUDA,
Vulkan/device-loss, out-of-memory, filesystem, or segmentation fault failure.

## Focus GPU Pinning Correction

A follow-up GPU-process check on `126188` showed that the original overlap
focus steps all landed on the same logical GPU. That wasted the rest of the
4-H200 allocation even though the steps were running. The allocation was kept
alive and no running useful step was cancelled. A new allocation-internal
script was added:

`scripts/slurm/run_fix3_v7_nonpeg_focus3_pinned_in_allocation.sh`

The script uses one `gpu:4` overlap step and explicitly launches three
additional focused generator processes with `CUDA_VISIBLE_DEVICES=1`, `2`, and
`3`. These new roots target the still-scarce non-peg classes:
`hole_late_constant`, `hole_late_sine`, and
`hole_late_continuous_insert`. The generator, physical gates, episode length,
and SFT approval gate are unchanged.

The script passed `bash -n` inside allocation `126188` and was started as tmux
session `cosmos3_fix3_focus3_pinned_126188`. `nvidia-smi` then confirmed the
new `.venv/bin/python` processes were distributed onto GPU UUIDs corresponding
to logical GPUs `1`, `2`, and `3`, while the earlier focus processes remained
on logical GPU `0`. Early focus3 logs reached the generator and produced only
expected physical-gate rejects (`final_insert_failed`,
`no_large_final_target_in_bounds`, and
`counterfactual_final_target_self_insert_gate_failed`), with no traceback or
device/runtime errors.

The active-root filename scan before the pinned roots had accepted rows reached
`378/1000` unique rows, with `439` non-peg moving-hole rows still missing:

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 17 | 16 | 70 | 54 |
| `hole_late_constant` | 8 | 8 | 90 | 82 |
| `hole_late_reverse` | 40 | 39 | 100 | 61 |
| `hole_late_sine` | 20 | 18 | 90 | 72 |
| `hole_late_continuous_insert` | 21 | 21 | 120 | 99 |
| `hole_late_fast_shift` | 49 | 49 | 120 | 71 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

## Latest Non-Peg-First Live Check

After the user reconfirmed that peg classes can be handled last, the active
policy is: keep all useful held allocations focused on the six non-peg
moving-hole classes, leave `none` capped in the final merge, and return to
`peg_drop`/`peg_disturb` only after the non-peg holes are filled.

A fresh active-root filename scan reached `403/1000` unique scenario/seed rows
across non-archived v7 group roots. This is not the final merged count; it
includes work-buffer H5 files that still need strict merge, deduplication,
H5 structural audit, visual/render review, and user approval. The latest user
reported merged count remains `276/1000`.

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 19 | 18 | 70 | 52 |
| `hole_late_constant` | 12 | 12 | 90 | 78 |
| `hole_late_reverse` | 46 | 45 | 100 | 55 |
| `hole_late_sine` | 22 | 20 | 90 | 70 |
| `hole_late_continuous_insert` | 24 | 24 | 120 | 96 |
| `hole_late_fast_shift` | 57 | 57 | 120 | 63 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

The six non-peg moving-hole classes still have `414` rows missing. Recent
accepted rows came from both the older non-peg CFS/MRS roots and the focused
roots, including `hole_late_fast_shift`, `hole_late_continuous_insert`,
`hole_late_reverse`, `hole_late_sine`, and `hole_late_constant`. The observed
slowdown is therefore low physical-gate pass rate, not an idle queue or
obvious runtime failure. Allocation names that still mention `static`,
`pegmix`, or `pegdist` may be misleading because their tmux panes were
repurposed to non-peg roots; they should not be interpreted as active peg
debugging unless the log root confirms it.

## Extra Focus4b Allocation

Because all useful held allocations were busy and the non-peg moving-hole
classes still had a large gap, an additional focused 4-H200 allocation was
requested rather than waiting passively. A new allocation-internal launcher was
added:

`scripts/slurm/run_fix3_v7_nonpeg_focus4_pinned2_in_allocation.sh`

It only starts four disjoint-seed v7 generator roots on a held allocation and
pins them to separate logical GPUs. The generator, physics gates, source
protocol, episode length, and SFT approval gate are unchanged. The focused
classes are `hole_late_constant`, `hole_late_continuous_insert`,
`hole_late_sine`, and `hole_late_move_stop`, with seed bases `19250000`,
`19241000`, `19251000`, and `19280000`.

The script passed `bash -n` inside allocation `126188`. A tmux-held allocation
was then started as `cosmos3_fix3_focus4b_pinned_4h200_0611`; Slurm job
`126210` first queued briefly, then started on `server56`. The four new logs
recorded their intended `CUDA_VISIBLE_DEVICES=0/1/2/3` and output roots.
Early error scan found no traceback, CUDA/Vulkan, out-of-memory, filesystem,
or segmentation error. A follow-up check confirmed one `.venv/bin/python`
generator process on each of the four GPUs, and all four focus4b logs reached
the generator after checkpoint load. The first accepted focus4b row was a
`hole_late_sine` trajectory with `target_motion_norm_m=0.2315` and
`first_insert_step=160`, so `126210` is now productive source generation.

A fresh active-root filename scan after the new allocation became productive
reached `421/1000` unique scenario/seed rows:

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 21 | 20 | 70 | 50 |
| `hole_late_constant` | 12 | 12 | 90 | 78 |
| `hole_late_reverse` | 47 | 46 | 100 | 54 |
| `hole_late_sine` | 26 | 24 | 90 | 66 |
| `hole_late_continuous_insert` | 28 | 28 | 120 | 92 |
| `hole_late_fast_shift` | 64 | 64 | 120 | 56 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

The six non-peg moving-hole classes still have `396` rows missing. This remains
file-level buffer progress only and is not a substitute for the final merge,
strict H5 audit, rendered/video review, or user approval.

## Follow-Up Productive Check

A follow-up check confirmed there was still no `done` marker and no active log
hit for traceback, CUDA/Vulkan/device loss, out-of-memory, filesystem, or
segmentation failure. Eleven held allocations were running with active Slurm
steps, including `126210` and `126188`.

The next active-root filename scan reached `425/1000` unique scenario/seed
rows:

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 21 | 20 | 70 | 50 |
| `hole_late_constant` | 12 | 12 | 90 | 78 |
| `hole_late_reverse` | 48 | 47 | 100 | 53 |
| `hole_late_sine` | 26 | 24 | 90 | 66 |
| `hole_late_continuous_insert` | 29 | 29 | 120 | 91 |
| `hole_late_fast_shift` | 66 | 66 | 120 | 54 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

The six non-peg moving-hole classes still have `392` rows missing. The current
decision is to keep running the non-peg focused source generation and not start
peg-focused work yet. No extra allocation was started after `126210`, because
the current held resources are active and producing accepted rows.

## Merge And Strict Source Audit Tooling

The existing merge script
`scripts/world_model/merge_fix3_full1000_unique_h5.py` already performs
file-level selection by scenario quota and scenario/seed uniqueness, and writes
`fix3_h5_paths.txt` plus `manifest.json`. It correctly records that file-level
merge does not replace H5 structural or physical audits.

A new read-only strict source audit was added:

`scripts/world_model/audit_fix3_merged_source_h5.py`

It verifies the merged path list before render expansion, WAM export, or SFT:
scenario quotas, duplicate paths, one trajectory per H5, `301` frame arrays,
`300x7` actions, source frame indices `0..300`, required `slots`/`perturb`/
`env_states`, finite arrays, final inserted state from slots, summary
`success_at_end`/`inserted_end`/`live_success_end`, moving-hole target motion
at least `0.22m`, no hole motion in `none`/peg scenarios, no peg deltas in
moving-hole scenarios, and event timing after grasp/robust hold where
applicable. Passing this script is still not video approval; it is only the
structural/metadata gate before rendered review.

The script passed `py_compile` inside Slurm allocation `126188` and passed a
canary audit on the approved v7 complete-nine smoke path list:

`experiments/world_model_task_rebinding/cosmos3/fix3_original_protocol_large_motion_dp_v7_complete9_combined_20260611/strict_source_h5_audit_20260611`

Canary result: `strict_ok=true`, `num_paths=9`, `num_failed_records=0`, with
all nine scenarios present at `1/1`.

## Latest Active-Root Count

A later active-root filename scan reached `449/1000` unique scenario/seed rows:

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 25 | 24 | 70 | 46 |
| `hole_late_constant` | 19 | 19 | 90 | 71 |
| `hole_late_reverse` | 51 | 50 | 100 | 50 |
| `hole_late_sine` | 30 | 28 | 90 | 62 |
| `hole_late_continuous_insert` | 35 | 35 | 120 | 85 |
| `hole_late_fast_shift` | 66 | 66 | 120 | 54 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

The six non-peg moving-hole classes still have `368` rows missing. Eleven
held allocations remain running. The current action is unchanged: keep
resources focused on non-peg source generation first and defer peg-only work
until the non-peg quotas are filled or a concrete repeated blocker appears.

## Latest Continuation Check

A later read-only active-root scan reached `455/1000` unique scenario/seed
rows:

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 26 | 25 | 70 | 45 |
| `hole_late_constant` | 19 | 19 | 90 | 71 |
| `hole_late_reverse` | 52 | 51 | 100 | 49 |
| `hole_late_sine` | 31 | 29 | 90 | 61 |
| `hole_late_continuous_insert` | 38 | 38 | 120 | 82 |
| `hole_late_fast_shift` | 66 | 66 | 120 | 54 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

The six non-peg moving-hole classes still have `362` rows missing. All eleven
held allocations remain running with active steps. Error scan found no
traceback, CUDA/Vulkan/device-loss, out-of-memory, filesystem, Slurm-step, or
segmentation failure, and no `fix3_original_protocol_large_motion_done` marker
exists yet. Recent tmux panes show the generators are still filtering by the
expected physical gates and accepting rows for `hole_late_continuous_insert`,
`hole_late_sine`, and `hole_late_reverse`.

## Non-Peg Continuation After User Count

The user reported the latest merged count as `276/1000` unique H5 rows and
explicitly allowed handling peg classes last. A fresh active-root filename scan
over the work-buffer roots reached `493/1000` unique scenario/seed rows. This
is higher than the merged count because buffer roots have not yet been folded
into the final capped path list, and it is still only file-level progress, not
structural/video approval.

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 29 | 28 | 70 | 42 |
| `hole_late_constant` | 24 | 24 | 90 | 66 |
| `hole_late_reverse` | 59 | 58 | 100 | 42 |
| `hole_late_sine` | 36 | 34 | 90 | 56 |
| `hole_late_continuous_insert` | 49 | 49 | 120 | 71 |
| `hole_late_fast_shift` | 73 | 73 | 120 | 47 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

The six non-peg moving-hole classes still have `324` rows missing. The current
action remains to use all productive resources on those classes first, then
return to `peg_drop` and `peg_disturb`.

Allocation `126065` was lost by Slurm revocation (`CANCELLED by 0`) rather than
agent cancellation. A replacement tmux-held 1-H200 allocation `126219` started
on `server40` and is running the disjoint constant/continuous buffer root
`fix3_original_protocol_large_motion_dp_v7_approved_full1000_20260611_group_nonpeg_cc_repl_aux210_seedbase202x`.
GPU/process and log checks show an active `.venv/bin/python` generator. Its
early rows are being rejected by the expected physical-quality gates
(`final_insert_failed`, `counterfactual_final_target_self_insert_gate_failed`,
and `no_large_final_target_in_bounds`), with no traceback, CUDA/device-loss,
OOM, filesystem, Slurm-step, or segmentation error found in the active logs.

Because the non-peg gap remains large and all existing held allocations are
productive, one additional tmux-held 4-H200 allocation was requested for
disjoint non-peg focus work. New launcher:
`scripts/slurm/run_fix3_v7_nonpeg_focus4_pinned3_in_allocation.sh`; it passed
`bash -n` and uses the unchanged v7 generator/physical gates with new seed
bases `21250000`, `21241000`, `21251000`, and `21280000`. Slurm job `126223`
started immediately on `server38`, with roots
`nonpeg_focus4c_constant_aux90_seedbase21250000`,
`nonpeg_focus4c_continuous_aux120_seedbase21241000`,
`nonpeg_focus4c_sine_aux90_seedbase21251000`, and
`nonpeg_focus4c_move_stop_aux70_seedbase21280000`. Process-tree and
`nvidia-smi` inspection on the compute node confirmed four active
`.venv/bin/python` generators distributed across four GPUs. The logs reached
the reject/accept loop, and the first focus4c accepted row was
`hole_late_sine_seed21251007_idx11220` with `target_motion_norm_m=0.2266`,
`trigger_step=92`, and `first_insert_step=159`. This is acceleration for the
approved source-generation objective, not a change to the dataset protocol.

## Live Count After Focus4c Startup

A later active-root filename scan reached `606/1000` unique scenario/seed H5
rows. The count remains a work-buffer estimate before final capped merge,
strict source audit, rendered review, and user approval.

| scenario | raw | unique | quota | missing |
| --- | ---: | ---: | ---: | ---: |
| `hole_late_move_stop` | 38 | 37 | 70 | 33 |
| `hole_late_constant` | 41 | 41 | 90 | 49 |
| `hole_late_reverse` | 76 | 75 | 100 | 25 |
| `hole_late_sine` | 54 | 52 | 90 | 38 |
| `hole_late_continuous_insert` | 81 | 81 | 120 | 39 |
| `hole_late_fast_shift` | 93 | 93 | 120 | 27 |
| `none` | 172 | 161 | 160 | 0 |
| `peg_drop` | 68 | 66 | 150 | 84 |
| `peg_disturb` | 0 | 0 | 100 | 100 |

The six non-peg moving-hole classes still have `211` rows missing, so the
correct next action is still to keep current productive resources on non-peg
source generation. The tmux sessions whose names include `pegdist` or `pegmix`
were checked and are currently running repurposed non-peg roots, not peg-only
generation. Queue/step checks show eleven running allocations after `126074`
was revoked by Slurm; log mtimes show recent non-peg activity; the error scan
found no traceback, CUDA/Vulkan/device loss, out-of-memory, filesystem,
Slurm-step, or segmentation failure. No done marker exists yet. The count
continued to grow from `510` to `523`, `529`, `538`, `550`, `564`, `570`,
`582`, `598`, and then `606` across successive monitoring passes, so the running non-peg
allocations are still productive. Allocation `126089` was later revoked by
Slurm after accepting additional non-peg reverse rows; ten allocations remain
running, and no further allocation was requested at this moment because the
current production rate is still useful.

## Gate

Do not start WAM export, render-scale expansion, or Cosmos3 SFT when the count
reaches full1000. Stop and wait for user approval.
