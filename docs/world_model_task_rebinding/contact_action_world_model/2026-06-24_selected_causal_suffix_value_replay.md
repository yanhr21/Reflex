# 2026-06-24 Selected Causal-Suffix Value Replay

## Scope

This diagnostic used the one-GPU-hour value head to choose one candidate per
panel0134 saved live snapshot, then replayed only the selected candidate from
the simulator snapshot inside tmux-held Slurm allocation `148680` on `server13`.
It is not live-controller method evidence.

## Code And Inputs

- Selector:
  `scripts/world_model/select_cosmos3_live_snapshot_candidates_with_value_head.py`
- Replay wrapper:
  `scripts/slurm/run_selected_causal_suffix_value_replay_in_allocation.sh`
- Replay script patch:
  `scripts/world_model/replay_cosmos3_live_action_bank_from_snapshots.py`
  now applies candidate-filter TSVs after synthetic causal candidates are
  generated, so generated candidate ids from converted outcome rows can be
  replayed directly.
- Value checkpoint:
  `experiments/world_model_task_rebinding/cosmos3/contact_value_head_union_plus_panel0134_causal_suffix_1gpu1h_20260623_204725_alloc146658/checkpoint_best_gate.pt`
- Causal suffix generator:
  `experiments/world_model_task_rebinding/cosmos3/causal_contact_action_suffix_diffusion_full733_1gpu1h_20260623_190108_alloc146658/checkpoint_best_eval.pt`
- Selection output:
  `experiments/world_model_task_rebinding/cosmos3/selected_causal_suffix_value_head_panel0134_margin0_20260624_171802_alloc148680`
- Replay output:
  `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_selected_causal_suffix_value_head_panel0134_margin0_exec8_dp96_20260624_171802_alloc148680`

## Results

The selector chose `16` candidates:

- `11` causal suffix diffusion candidates
- `5` DP-prior candidates

The selection summary, evaluated against the previously converted labels, said
`15/16` selected rows were DP96-success rows. This was not accepted as final
evidence. The selected filter was replayed again from the real saved simulator
snapshots.

Authoritative replay summary:

- valid records: `16/16`
- failure counts: `{}`
- direct post-chunk success: `0/16`
- direct post-chunk gate-ok: `0/16`
- DP96 labels: `16`
- `candidate + DP96` success: `9/16`
- DP-continuable: `11/16`
- final contact-stable after DP96: `11/16`
- improved `abs_y + abs_z`: `4/16`
- worsened `abs_y + abs_z`: `12/16`

Baseline from the same saved states:

- DP-prior-only replay had `8/16` DP96 success and `8/16` continuability.

## Conclusion

The selected value-head replay improves DP96 success only from `8/16` to
`9/16` and continuability from `8/16` to `11/16`. It still produces no direct
post-chunk insertion or gate-ok states. This is too weak for live-panel
promotion and confirms that the current value head is not a reliable controller
selector.

The mismatch between the selector's converted-label expectation (`15/16`) and
the authoritative replay (`9/16`) is itself a blocker. It may come from replay
nondeterminism, subtle candidate-index/action reconstruction differences, or
DP96 rollout sensitivity. Until this is diagnosed, old converted-label
selection summaries must not be treated as execution evidence.

## Next Direction

Do not continue scalar margin tuning as the main method. The aligned next
method work is a direct contact/insertion action executor or a stronger base
policy/WAM path:

1. Train or adapt a model whose positive labels include direct insertion or
   contact-stable post-chunk states, not only `candidate + DP96` handoff.
2. Add an explicit candidate family descriptor for causal suffix diffusion in
   future value-head training, because the current checkpoint descriptor schema
   has no causal-specific bit and must infer that family indirectly.
3. Prioritize Cosmos Policy-DROID style action/future-state/value
   post-training or OpenPI/pi0.5 integration if the local executor remains a
   handoff-state generator rather than an insertion executor.
