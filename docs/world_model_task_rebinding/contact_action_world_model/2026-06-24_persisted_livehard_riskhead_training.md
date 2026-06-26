# 2026-06-24 Persisted-Live-Hard Risk-Head Training

## Purpose

Use the newly persisted generated action chunks as actual training rows. The
action diffusion head still imitates only primary source direct-contact
positives. An auxiliary risk head trains on source positives versus failed live
replay hard negatives so future sampling can reject action chunks that resemble
known live failures.

This is not scorer-only control. The risk head is a value/risk component around
an action generator, and sampled actions still require saved-snapshot replay.

## Inputs

- Manifest:
  `experiments/world_model_task_rebinding/cosmos3/direct_contact_executor_manifest_h24_sourcepos_persistedlivehard_20260624_alloc148732/direct_contact_executor_manifest.jsonl`
- Actions:
  `experiments/world_model_task_rebinding/cosmos3/direct_contact_executor_manifest_h24_sourcepos_persistedlivehard_20260624_alloc148732/direct_contact_executor_manifest_arrays.npz`
- Manifest summary:
  `2922` rows, `2905` source direct positives, `16` live replay hard negatives,
  `1` Policy-DROID hard negative, and
  `replay_action_loaded_from_persisted_chunk_json=11`.

## Launch

The first launch
`direct_contact_executor_diffusion_h24_sourcepos_persistedlivehard_riskhead_1gpu1h_20260624_alloc148732`
was interrupted after about one minute because risk validation did not reliably
contain negative examples. That partial run is invalid as training evidence.

The trainer was patched to use label-stratified risk validation, then relaunched
inside held allocation `148732`:

`experiments/world_model_task_rebinding/cosmos3/direct_contact_executor_diffusion_h24_sourcepos_persistedlivehard_riskhead_1gpu1h_fix1stratval_20260624_alloc148732`

Early metadata:

- risk train positives: `2469`
- risk train negatives: `14`
- risk val positives: `436`
- risk val negatives: `3`

## Boundary

Do not interpret this as a training result until
`formal_one_gpu_hour_floor_met=true` appears in `training_summary.json`. Even
then it remains training evidence only. Controller evidence requires sampled
chunks replayed from saved dynamic live snapshots, followed by final-state and
visual review.
