# Joint Training TODO

- [ ] Inspect existing DP and Cosmos data interfaces inside a Slurm allocation
  before writing runnable training code.
- [ ] Add `scripts/world_model/build_joint_dp_cosmos_dataset.py`.
- [ ] Add `scripts/world_model/inspect_joint_dp_cosmos_batch.py`.
- [ ] Add `scripts/training/train_joint_dp_cosmos_overfit.py`.
- [ ] Run the overfit experiment on a tiny real slice.
- [ ] Verify DP action loss and Cosmos future-state / latent loss separately.
- [ ] Verify action chunks and future-state chunks are temporally aligned.
- [ ] Add `scripts/training/train_joint_dp_cosmos_full.py` only after overfit
  evidence is correct.
- [ ] Run full training for at least `1 GPU x 1 hour` on real data before
  calling it training evidence.
