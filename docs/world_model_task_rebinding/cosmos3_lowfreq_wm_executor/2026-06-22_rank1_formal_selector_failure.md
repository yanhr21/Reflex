# 2026-06-22 Rank1 Formal Selector Failure

Run:

`experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_live_snapshot_source_suffix_rank1_fullbatch_20260622_alloc145920_formal3h`

Input labels:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_source_suffix_union_20260622_alloc145920`

Boundary:

- held allocation `145920`;
- 1 visible GPU;
- `10800` second formal floor;
- `41` live states;
- `8877` candidate rows;
- `3` states where DP prior fails but a non-DP candidate has DP96 success;
- `rank_loss=1.0`;
- full-group ranking batch size `8192`.

Final summary:

- `formal_training_floor_met=true`;
- `stop_reason=max_wall_seconds`;
- `steps=55962`;
- `ready_for_formal_live_eval=false`;
- `final_ready_for_offline_gate=false`;
- `best_gate_metrics=null`;
- no `checkpoint_best_gate.pt`.

Final eval:

- validation groups: `8`;
- DP prior handoff success: `0/8`;
- validation handoff oracle: `1/8`;
- selected handoff success: `0/8`;
- selected minus DP handoff: `0`;
- selected minus DP weighted error: `+0.00246`;
- selected minus DP contact-progress delta: `-0.02760`;
- selected candidates: `dp_prior` three times, one source-suffix candidate,
  one `scale_1.5`, one `short12_scale_1.5`, one `short8_scale_1`, and one
  `short8_scale_1.5`.

Final train:

- train groups: `33`;
- DP prior handoff success: `0/33`;
- train handoff oracle: `2/33`;
- selected handoff success: `2/33`;
- selected minus DP weighted error: `-0.04329`.

Interpretation:

The rank objective learned the training positives but did not select the
single validation handoff-positive state. The validation positive is not from
a completely unrelated distribution: it is sample05 iter05, while training
contains sample05 iter04 from the same source/scenario plus a sample04
positive. This is therefore selector generalization failure, not evidence that
the method is ready for live eval.

This result does not solve sample00. sample00's corrected candidate bank has
no usable early action candidate, so that path still requires stronger
action-candidate/executor coverage.

Next aligned audits:

- live-candidate-only/source-suffix headroom under deployment candidate
  families;
- source/scenario/phase split audit;
- selected-family versus oracle-winner-family comparison.
