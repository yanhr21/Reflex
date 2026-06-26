# 2026-06-24 Generated Action Persistence Repair

## Context

The contact-action reset found a concrete data plumbing blocker. Some replay
labels preserved the outcome of synthetic generated candidates, including DP96
handoff success/failure, but did not preserve the generated robot action arrays.
Those labels were useful as evidence/value labels, but could not be converted
into action-training rows for a direct contact executor.

## Change

Two scripts were updated on the login node as text/code edits only:

- `scripts/world_model/replay_cosmos3_live_action_bank_from_snapshots.py`
  now persists generated/synthetic replay candidates as standalone action-chunk
  JSON files and records `persisted_action_chunk_json` in each outcome label.
  By default it persists generated candidates, not original saved-bank actions.
- `scripts/world_model/build_direct_contact_executor_manifest.py` now tries
  `persisted_action_chunk_json` first, then falls back to the original
  `candidate_action_bank.npz`.

## Boundary

This is not a training, replay, or controller result. It does not retroactively
recover old replay labels that were already written without action chunks.

## Next Compute Step

Inside the held tmux/Slurm allocation, rerun the selected causal-suffix or live
candidate replay with action persistence enabled, then rebuild the
direct-contact manifest. The expected improvement is data availability:
generated DP96-positive candidates and live hard negatives should become usable
action rows instead of label-only evidence.

## Compute Follow-Up

This was run inside tmux-held Slurm allocation `148732` on `server24`, not on
the login node.

Replay output:
`experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_selected_causal_suffix_value_head_panel0134_margin0_exec8_dp96_20260624_persistfix1_alloc148732`

Replay summary:

- records: `16`
- valid records: `16`
- process failures: `{}`
- causal-suffix records: `11`
- persisted generated action chunks: `11`
- direct success: `0/16`
- direct gate-ok: `0/16`
- DP96 success: `8/16`
- DP96 continuable: `10/16`
- improved `abs_y+abs_z`: `4/16`
- worsened `abs_y+abs_z`: `12/16`

Rebuilt manifest:
`experiments/world_model_task_rebinding/cosmos3/direct_contact_executor_manifest_h24_sourcepos_persistedlivehard_20260624_alloc148732`

Manifest summary:

- total rows: `2922`
- source direct positives: `2905`
- live replay hard negatives: `16`
- Policy-DROID hard negatives: `1`
- causal-suffix family rows: `11`
- `replay_action_loaded_from_persisted_chunk_json=11`
- `ready_for_direct_contact_executor_training=true`

Interpretation: the action-loss blocker is fixed for new replay labels. The
execution result is still not a method success and still has no direct live
contact-positive rows.
