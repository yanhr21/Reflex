# 2026-06-22 Rank1 Failure Offline Audits

This note follows the failed formal rank1 scorer:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_live_snapshot_source_suffix_rank1_fullbatch_20260622_alloc145920_formal3h`

Input label union:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_source_suffix_union_20260622_alloc145920`

## Candidate-Family Headroom

Total:

- `41` live states;
- `8877` candidate rows;
- `3/41` states have any DP96/final-success candidate;
- DP prior success groups: `0/41`;
- checkpoint-model non-DP success groups: `0/41`;
- source-suffix success groups: `3/41`.

Candidate rows by family:

- `dp_prior`: `41`;
- `model_mean`: `41`;
- `model_scale_full`: `164`;
- `model_sample_full`: `1968`;
- `model_short8`: `2173`;
- `model_short12`: `2173`;
- `model_short16`: `2173`;
- `source_suffix`: `144`.

Success rows by family:

- `source_suffix`: `18`;
- all other families: `0`.

Continuable groups by family:

- `source_suffix`: `4`;
- all other families: `0`.

Interpretation: in this formal rank1 label union, usable DP96 headroom comes
only from source-suffix candidates. The checkpoint-model candidate bank has no
DP96-success group. This supports the action-candidate coverage diagnosis:
ordinary live checkpoint executor candidates are not yet enough in this data
slice.

## Split Audit

Split reproduced from the scorer defaults:

- seed: `20260617`;
- validation fraction: `0.2`;
- train groups: `33`;
- validation groups: `8`.

Overlap:

- train and validation share `source_uuid`;
- train and validation share scenarios;
- both train and validation are `current_phase=no_grasp`;
- both train and validation use `prefix_role=target_motion_observed`.

Validation positive:

- sample05 iter05, frame `198`;
- `7` source-suffix DP96-success candidates.

Training positives:

- sample04 iter05, frame `196`, with `2` source-suffix DP96-success candidates;
- sample05 iter04, frame `174`, with `9` source-suffix DP96-success candidates.

Interpretation: the failed validation positive is not from a completely
unseen source/scenario/phase. The scorer saw a nearby sample05 positive in
training but still did not select the validation positive. This strengthens
the selector-generalization diagnosis.

## Selected Family Versus Oracle Family

Final validation selected counts:

- `dp_prior`: `3`;
- `source_suffix`: `1`;
- `model_scale_full`: `1`;
- `model_short12`: `1`;
- `model_short8`: `2`.

Final validation handoff oracle:

- `none`: `7`;
- `source_suffix`: `1`.

Final validation weighted-error oracle:

- `source_suffix`: `2`;
- `model_scale_full`: `4`;
- `model_short8`: `2`.

Interpretation: the final scorer did not simply avoid source suffixes; it
selected one source-suffix candidate, but the wrong one. The single validation
handoff-positive family is source suffix, specifically
`retrieval_resid_srcsuffix_r10_s0p5_o24`, while the final scorer selected
`retrieval_resid_srcsuffix_r2_s1_o32` once and missed the handoff-positive
state.

## Decision

Do not run live eval from this scorer. The next repair should not be another
threshold change. It should either:

- improve candidate generation/executor coverage, because checkpoint-model
  candidates have no DP96-success groups in this union; or
- improve source-suffix/action scoring features and split robustness, because
  the scorer still fails on a nearby validation positive.
