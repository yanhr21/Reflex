# 2026-06-22 Gate-DP96 False Positive Blocker

## Plain Result

The current handoff gate is not a reliable continuability test.

Across the latest gate-only replay, `211` candidate chunks that had already
passed the geometry handoff gate were followed by real frozen-DP rollout for
up to `96` steps. Result:

- `211/211` passed the after-chunk geometry gate;
- `0/211` reached task success after DP rollout;
- `0/211` were DP-continuable;
- `0/211` ended in contact-stable state;
- `211/211` still had the peg grasped.

This means the current candidates can make y/z look close while still not
putting the peg into a physical state from which DP can insert.

## Evidence Paths

Sine gate-only DP96 replay:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_gate_iters_dp96_hole_late_sine_iter4_5_gateonly_20260622_1145_alloc145920`

Continuous/move-stop follow-up replay:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_gate_iters_dp96_continuous_iter8_movestop_iter10_gateonly_20260622_1152_alloc145920`

Filter files:

- `experiments/world_model_task_rebinding/cosmos3/sine_iter45_gate_candidate_indices_20260622.tsv`
- `experiments/world_model_task_rebinding/cosmos3/continuous_insert_iter8_gate_candidate_indices_20260622.tsv`
- `experiments/world_model_task_rebinding/cosmos3/move_stop_iter10_gate_candidate_indices_20260622.tsv`

## Breakdown

Fair DP-rollout cases:

- `hole_late_sine`, iterations 4 and 5: `160` gate candidates, `0` DP success,
  `0` DP-continuable.
- `hole_late_continuous_insert`, iteration 8: `51` gate candidates, `0` DP
  success, `0` DP-continuable.

Late end-of-episode case:

- `hole_late_move_stop`, iteration 10: `1` selected gate candidate, but after
  the candidate chunk the episode was already at step `300`, so no DP rollout
  horizon remained. It still ended in failure.

## Physical Interpretation

The false-positive states are not random. They are typically y/z-close but
still physically wrong for insertion.

For sine gate candidates:

- after candidate chunk, median relative position was roughly
  `x=-0.107`, `y=-0.0086`, `z=-0.0014`;
- after DP96, median moved to roughly `x=-0.104`, `y=0.0093`,
  `z=-0.0379`;
- stable contact steps stayed `0`.

For continuous-insert gate candidates:

- after candidate chunk, median relative position was roughly
  `x=-0.115`, `y=0.0062`, `z=-0.0003`;
- after DP96, median was roughly `x=-0.119`, `y=0.0098`,
  `z=-0.0001`;
- stable contact steps stayed `0`.

Plain meaning: matching y/z near the hole is not enough. The peg remains too
far from a real insertion/contact manifold, and frozen DP either fails to
advance insertion or drifts the peg away in contact.

## Positive Contact-Manifold Audit

Read-only source-slot audit:

`experiments/world_model_task_rebinding/cosmos3/dp_success_contact_manifold_audit_20260622/dp_success_contact_manifold_summary.json`

Markdown summary:

`experiments/world_model_task_rebinding/cosmos3/dp_success_contact_manifold_audit_20260622/dp_success_contact_manifold_summary.md`

The audit covered all `733` accepted source H5 files after setting
`HDF5_USE_FILE_LOCKING=FALSE`.

Key successful-source medians:

- within `32` frames before first insertion: `x=-0.108`, abs y/z sum
  `0.0042`;
- within `16` frames before first insertion: `x=-0.091`, abs y/z sum
  `0.0033`;
- within `8` frames before first insertion: `x=-0.059`, abs y/z sum
  `0.0034`;
- inserted frames: `x≈0`, abs y/z sum `0.0048`.

Comparison:

- sine false gate candidates: after-candidate median `x≈-0.107`, abs y/z sum
  `0.0089`;
- continuous-insert false gate candidates: after-candidate median `x≈-0.115`,
  abs y/z sum `0.0066`.

Plain meaning: many false gate states look like a broad "within 32 frames"
pre-insert alignment state, but the current DP handoff needs a much more
advanced insertion-axis/contact state. The old gate accepted "pointed at the
hole" as if it were "ready for insertion."

## What This Does Not Prove

This is not method success. It is label/evaluation evidence for the blocker.

It also does not mean "just tighten a threshold." A tighter hand-written gate
would only hide this batch of false positives. The method needs a learned or
teacher-supported notion of DP-continuable contact/insertion state, and a
candidate generator that can actually produce that state.

## Next Direction

The next aligned repair is:

1. Treat current geometry-gate positives as negative DP-continuability labels.
2. Mine or construct real positive DP-continuable insertion states from
   successful expert/static-DP suffixes or approved teacher data.
3. Train/evaluate the live outcome scorer on real DP-continuability/contact
   labels, not on y/z geometry alone.
4. Change candidate generation/executor training so candidates aim for the
   insertion/contact manifold before handoff, not just y/z alignment.

This remains the Dream Diffusion Policy-style contract: short candidate
chunks, scored by predicted real task consequence, execute a short prefix,
reobserve, and hand off only when the real state is physically continuable.
