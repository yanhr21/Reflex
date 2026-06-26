# 2026-06-22 Source Insertion Suffix Replay Headroom

## Plain Result

Successful source insertion suffixes create live DP-handoff headroom.

This is not closed-loop method success. It is a diagnostic replay from saved
live snapshots. But it answers an important physical question: a stronger
insertion/contact action source can move some live states into a region where
frozen DP can finish.

## Artifacts

Source suffix bank:

`experiments/world_model_task_rebinding/cosmos3/source_insertion_suffix_bank_20260622_alloc145920`

Bank NPZ:

`experiments/world_model_task_rebinding/cosmos3/source_insertion_suffix_bank_20260622_alloc145920/source_insertion_suffix_bank.npz`

Replay output:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_source_suffix_replay_sine_cont_iter458_20260622_alloc145920`

Replay labels:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_source_suffix_replay_sine_cont_iter458_20260622_alloc145920/live_snapshot_action_bank_outcome_labels.jsonl`

## What Was Tested

The source bank mined all `733` accepted successful H5 trajectories and
extracted insertion suffixes with horizon `96` from offsets
`96,64,48,32,24,16,8,0` before first insertion.

The bank has `4711` suffix rows. Its start relative position median is about
`x=-0.110`; its end relative position median is near inserted state
(`x=-0.00088`). This covers the same x region where the current false geometry
gate was stuck.

The replay used live snapshots from the latest failed action-bank panel:

`live_receding_candidate_executor_iter1500_h96_fullunion_scorer_bestgate_actionbank_samples0_2_4_5_20260622_090254_alloc145920`

It tested `hole_late_sine` and `hole_late_continuous_insert`, iterations
`4,5,8`. Per live snapshot, it retrieved the nearest `12` source suffixes by
current peg-head-at-hole state, with no scenario match by default, blended
them with the live DP prior at `0.5` and `1.0`, executed the first `32` steps,
then ran frozen DP for up to `96` steps as a label-only test.

## Counts

Total valid labels: `144`.

- direct after-candidate success: `0/144`;
- after-candidate geometry gate ok: `15/144`;
- DP96 success after the source suffix chunk: `18/144`;
- DP96 continuable: `19/144`;
- DP96 final contact-stable: `19/144`;
- final grasp after DP96: `144/144`;
- y/z improved after the source suffix chunk: `85/144`.

Breakdown:

- `hole_late_continuous_insert`: `16/72` DP96 success;
- `hole_late_sine`: `2/72` DP96 success;
- iteration `4`: `9/48` DP96 success;
- iteration `5`: `9/48` DP96 success;
- iteration `8`: `0/48` DP96 success.

Offset signal:

- offset `24`: `10/60` DP96 success;
- offset `32`: `8/72` DP96 success;
- offsets `8` and `16`: `0/12` DP96 success.

## Interpretation

This changes the blocker from "no candidate can help" to a more specific
failure:

The current live candidate family does not contain enough contact/insertion
action source. It can often improve y/z or pass a weak geometry gate, but it
does not reliably push along the insertion/contact axis. Successful source
suffixes do provide such a source in some live states.

This also confirms that the previous geometry gate is the wrong training
target. The right label is real DP96 success/continuability or contact-stable
progress after the candidate chunk.

## What This Does Not Prove

This is not a closed-loop method result.

It used saved live snapshots, source-mined candidate suffixes, and label-only
DP rollout. No video/contact sheet was produced for a method success claim.
It only proves that the action-source distribution has recoverable headroom.

## Next Repair

The next aligned repair is to convert this into a causal candidate family:

1. Add insertion-suffix or retrieval-residual candidates to live candidate
   generation using current/history task state, not scenario labels.
2. Train or recalibrate the scorer on real DP96 success/continuability and
   contact-stable labels, not geometry-gate positives.
3. Keep execution receding: execute a short prefix, reobserve, then rescore or
   hand off to DP only when real state is DP-continuable.

This follows the DDP/HDP lesson: the policy needs a serious action source for
the contact phase, while the world model/scorer decides which short chunk is
physically useful from the current live state.
