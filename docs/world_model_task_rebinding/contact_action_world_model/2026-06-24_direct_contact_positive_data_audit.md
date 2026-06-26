# 2026-06-24 Direct Contact-Positive Data Audit

## Scope

This was a login-node read-only audit after the Policy-DROID same-prefix
snapshot replay showed grasp preservation and DP96 handoff success, but no
direct post-chunk insertion/contact-stable state.

No project-code compute, replay, rendering, training, import check, or smoke
test was run on the login node.

## Current Evidence

Policy-DROID can now generate executable-format 8-step chunks from the active
causal RGB/live-prefix interface. On the saved dynamic snapshot
`sample_00_hole_late_move_stop/iter_00_prefix_f106`, the same-prefix chunk:

- preserved grasp;
- did not directly insert;
- did not reach inserted live pose, contact-stable state, or gate-ok state;
- slightly worsened near-term y/z task-frame error;
- allowed frozen DP to finish afterward in the DP96 label rollout.

The earlier causal suffix diffusion replay has the same shape at larger scale:
useful handoff coverage but `0` direct post-chunk insertion/gate positives in
the saved replay panel.

Conclusion: the main missing behavior is the contact/insertion action itself,
not scalar selection over the current candidate pool.

## Old Contact-Executor Dataset Audit

Existing historical dataset:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_iter1500_train512_scale005_repair_20260615`

Summary fields:

- joined rows: `512`
- `future_inserted_within_chunk_count`: `185`
- `future_dp_continuable_within_chunk_count`: `319`
- task path source: `cosmos_predicted_action_sidecar`
- scenarios include hole-late variants plus peg disturbance/drop cases

Path/provenance check:

- executor sample npz files referenced by the first rows still exist;
- contact-label npz files referenced by the first rows still exist;
- referenced DP-prior jsonl/chunks under
  `executor_dp_prior_smoke_20260615_pred_task_path_dp_prior_train512_diverse`
  are missing from the active tree;
- this root belongs to the older 2026-06-15 predicted-task-path executor
  branch, not the 2026-06-23 contact-action reset.

Therefore this dataset is not a clean active training root. It is useful
evidence that direct insertion/contact-positive supervision exists somewhere
in the repo, but it must not be trained from directly without rebuilding the
manifest and provenance.

## Hidden Problem Found

The active live outcome labels mostly encode "candidate chunk plus frozen DP
can finish" rather than "candidate chunk itself enters insertion/contact."
This explains why value heads and scorer selection can look promising offline
but fail as direct executors: they are learning handoff likelihood, not the
physical insertion motion.

The old contact-executor data had a better direct-positive label shape, but it
was not carried forward cleanly into the reset and now has broken DP-prior
references. Training it blindly would mix incompatible branches and make the
next result hard to interpret.

## Next Action

Build a new active direct-contact executor manifest, then train only after the
manifest is verified inside the held Slurm allocation.

Required manifest labels:

- causal current/history state and RGB-derived task context;
- current peg-head-at-hole and contact phase as input/context;
- short action chunk;
- direct post-chunk inserted/contact-stable/gate state as primary target;
- grasp preservation and insertion-axis progress;
- DP96 success/continuability as secondary handoff labels;
- hard negatives from Policy-DROID, causal suffix, and scorer-selected chunks
  that preserve grasp but fail direct insertion.

This keeps DP as a baseline/static suffix prior while shifting the learnable
target to contact execution.
