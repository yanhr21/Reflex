# 2026-06-22 Rank Overfit Sanity

Purpose: check whether the scorer failure on strict sample05 is partly caused
by training the outcome heads without an explicit within-state candidate
ranking loss.

Mixed label root:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_sample00neg_sample05dp96_mix_20260622_alloc145920`

Input labels:

- sample00 corrected iter0 all-candidate replay negatives;
- sample05 all-candidate replay rows that did not pass the handoff gate;
- sample05 after-gate rows with DP96 replay labels.

Conversion result:

- `1513` valid candidate rows;
- `7` live states;
- `118` DP96-success candidate rows;
- all DP96-success rows were from sample05 iteration 2, prefix frame `94`;
- in that same state, `dp_prior` also had DP96 success.

Training checks:

- `scorer_overfit100_retry_boolfix`: `100` steps, `rank_loss=0`.
  It completed after the boolean CLI fix, but selected `0/7` handoff-success
  candidates while DP prior had `1/7`.
- `scorer_rank_overfit100_val0`: `100` steps, `rank_loss=1`, full-group batch,
  `val_fraction=0`. Rank CE fell to about `0.00069`; selected handoff success
  became `1/7`, equal to DP prior `1/7`.

Interpretation:

Rank loss is required for this scorer. Regression losses alone can fit target
values while still selecting the wrong candidate in a state. The current mixed
label set is too small for method evidence because it contains only one
DP96-success live state, and DP prior succeeds in that same state. A formal
selector run should use broader real replay labels that include DP-failure
states with non-DP-success candidates. sample00 remains an action-candidate
coverage problem, not a scorer-only problem.
