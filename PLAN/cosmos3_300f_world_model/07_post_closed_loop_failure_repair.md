# Post Closed-Loop Failure Repair Plan

## Evidence Boundary

The corrected live-receding closed-loop eval was advanced after the user
explicitly stopped further smoke training. It failed on moving-target sample
`00` even after `12` receding Cosmos calls through frame `202`, while the
source teacher inserts by frame `166`. This is negative corrected closed-loop
evidence for the current checkpoint/interface, not a reason to launch more
training immediately.

The read-only distribution audit is:

`docs/world_model_task_rebinding/2026-06-13_receding_training_distribution_audit.json`

It reports:

- `2899` SFT rows from `733` source episodes;
- `1193/2899` role/mode mismatches (`41.15%`);
- no action condition-mask errors;
- `1287` late-rebind proxy rows where target motion has been observed, the
  peg is grasped, insertion is not done, and peg-head-to-hole error is outside
  a small local band.

## First-Principles Diagnosis

The physical failure is not "Cosmos did not see a target move." In the
corrected live run, the target moves, the loop reobserves, and the robot partly
follows. The failure is that the generated short action chunks do not drive the
peg into a DP-continuable insertion manifold after the target settles.

This remains task-frame rebinding, not a shortcut, because the required fix is
to make the model/controller reason from causal current target, peg, TCP,
grasp, and relative geometry. The fix must not provide future target poses as
conditions, restore old layouts, or relax the live simulator success gate.

The current training interface weakens that capability in two ways:

1. `prefix_role` is often a sampled intent rather than the actual physical
   mode at the prefix. The live controller uses observed physical history, so
   its conditioning distribution is cleaner than the training caption
   distribution but no longer exactly matched to it.
2. Each source episode contributes only a few hand-picked full-episode prefix
   masks. Live receding control quickly asks for recovery from its own miss,
   but the current rows mostly teach source-teacher trajectories, not dense
   short-horizon corrections from late/off-source states.

## Repair Track A: Clean Receding Condition Export

Do not regenerate or truncate source data for this repair. Start from the
accepted v7 H5/RGB sources and create a new condition export only.

Required exporter behavior:

- Preserve the `301` RGB/state frame and `300` action-step contract.
- Keep `condition_frame_indexes_action == range(prefix_frame_index)` for
  policy-history rows.
- Store the actual observed physical mode as the controller-facing role in
  captions and metadata.
- If a sampled curriculum role is still needed, store it separately as
  `sampled_prefix_role`, not as the role the model should obey at inference.
- Add dense receding prefix masks around the dynamic segment, e.g. every
  8 frames from causal target-motion onset through insertion or episode end.
- Include static monitor and peg recovery masks, but keep their physical modes
  clean as well.
- Record prefix histograms and role/mode mismatch counts as hard preflight
  diagnostics for any new condition root.

This is still full-episode SFT: repeated rows are full `301/300` samples with
different causal masks, not 128-action clips.

Implementation status:

- `export_cosmos3_maniskill_full_episode_wam_conditions.py` now supports
  `--prefix-role-source physical_mode` and
  `--dense-receding-prefix-stride N`.
- Default behavior remains `--prefix-role-source sampled` and
  `--dense-receding-prefix-stride 0`, preserving old-root reproduction.
- In clean mode, `prefix_role` is the physical mode used by the live
  controller; `sampled_prefix_role` records curriculum provenance such as
  `dense_receding`.
- A two-episode `/tmp` probe with physical-mode roles and 8-frame dense
  prefixes produced `23` full-episode rows and `0` role/mode mismatches. This
  probe was read/write file I/O only, not training, rendering, or controller
  evidence.
- Late-rebind row repetition is implemented as an optional JSONL-only sampler.
  A `/tmp` probe with `--late-rebind-weight 3` repeated the existing full rows
  from `23` to `59`, with no slicing, relabeling, rendering, or H5 changes.
- The preflight-only wrapper
  `scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_preflight_in_allocation.sh`
  is prepared for later user-approved condition export. It defaults to
  `RUN_SFT=false`.
- `summarize_cosmos3_clean_dense_preflight.py` combines condition manifest,
  strict full-episode preflight, receding-distribution audit, and weighted
  train JSONL metadata into `clean_dense_preflight_summary.json` with a single
  `ready_for_overfit` flag.

Approval-time command, to be run only inside a held compute allocation:

```bash
ALLOW_CLEAN_DENSE_PREFLIGHT=true bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_preflight_in_allocation.sh
```

This command should produce a new condition root and preflight output root,
but no SFT checkpoint, because `RUN_SFT=false` by default.

Before using the command, the login-safe dry run may be used to inspect the
exact paths and flags:

```bash
DRY_RUN_CONFIG_ONLY=true bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_preflight_in_allocation.sh
```

This dry run must not be confused with data export; it prints configuration
only and exits before calling the compute-node wrapper.

The same pattern exists for a two-source clean/dense overfit preflight:

```bash
DRY_RUN_CONFIG_ONLY=true bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_overfit2_preflight_in_allocation.sh
```

After approval, the non-dry command can be run inside a held compute
allocation to create only the overfit condition root and preflight summary.
It defaults `MAX_RECORDS=2` and `RUN_SFT=false`, and also requires
`ALLOW_CLEAN_DENSE_PREFLIGHT=true` for real execution.

## Repair Track B: Late-Rebind Action Coverage

The failed live rollout reaches states where the peg is grasped but outside
the moved-hole frame. Those states must be overrepresented or explicitly
supervised.

Allowed near-term repair:

- Use row repetition or sampling weights for physically labeled
  `target_motion_observed`, `target_post_motion`, and `insert_resume` rows
  that satisfy a late-rebind proxy: target motion observed, peg held, not
  inserted, and peg-head-to-hole error outside the local DP manifold.
- Report teacher action statistics by physical mode and prefix frame.
- Keep robot action dims `0..6` separate from sidecar state diagnostics in
  every metric.

If the clean/dense export still underreacts after an approved overfit and
full-run check, the evidence should be treated as a limitation of direct raw
Cosmos action-token execution. The next aligned controller would then be a
learned short-chunk executor or DP-prior policy conditioned on Cosmos-predicted
task state, not a threshold-only bridge and not blind long DP takeover.

## Repair Track C: Hard Dynamic Data, Later

The v7 accepted rows are kept. They are not thrown away. But they are
DP-success-filtered source trajectories, so they cannot by themselves prove
the original DP fails on the task family.

A later hard-teacher supplement should deliberately create physically valid
dynamic scenes where the static DP has low success, then solve them with a
manual/scripted/oracle teacher. This supplement is not launched by this plan
unless the user explicitly resumes it.

## Required Evidence Before Another Closed-Loop Claim

Before another controller-facing claim:

- new condition root passes strict `301/300` preflight;
- role/mode mismatch report is clean or explicitly justified;
- dense receding prefix distribution covers target onset through post-motion
  recovery;
- `clean_dense_preflight_summary.json` records `ready_for_overfit=true`;
- two-sample overfit is visually accepted only after the user approves
  resuming training;
- generated eval passes strict length, action/readout, and visual review;
- live closed-loop uses only the corrected receding interface with real
  re-observation and real-state `C_pi`.

No broad smoke, no extra SFT, and no full closed-loop panel should be launched
from the current failed checkpoint without a user-approved repair direction.
