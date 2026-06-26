# Workspace And Assets TODO

## Completed Cleanup

- [x] Kept Cosmos3 weights under `checkpoints/cosmos3/`.
- [x] Kept base static DP checkpoints under
      `experiments/dp_peg1000/run_90201/checkpoints/`.
- [x] Kept approved full1000 RGB dataset under
      `experiments/world_model_task_rebinding/cosmos3/sft_dataset_full1000_maniskill_default_regen_20260606_0055`.
- [x] Moved full1000 source H5/specs to
      `data/cosmos3/full1000_rgbd_env_states_20260603_1938`.
- [x] Updated approved full1000 dataset manifests/source list away from the
      old `experiments/_archive/...` source path.
- [x] Moved old experiments/docs/logs to
      `/public/home/yanhongru/ICLR2027_archive/reflex_20260609_cosmos3_300f_reset/`.
- [x] Moved explicit chunked/128-action/129-frame scripts to the same archive
      root.
- [x] Moved old SFT/eval/watch/controller wrappers that could restart rejected
      branches.
- [x] Moved remaining legacy `rgbd`/`full96` script entry points out of the
      active script tree.

## Still Required Before Training

- [ ] Run a source-list preflight proving `1000` active H5s exist and match the
      full1000 RGB manifest.
- [ ] Confirm the active source list has no archived path references.
- [ ] Confirm there are no active `chunked`, `128`, `129`, or `93-frame`
      condition roots under `experiments/world_model_task_rebinding/cosmos3/`.
- [ ] Record the preflight under `docs/world_model_task_rebinding/` after user
      review.
