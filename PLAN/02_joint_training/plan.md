# Joint Training Plan

Objective: train a DDP-inspired model route where policy action learning and
Cosmos future-state imagination are aligned.

Stage 1: overfit experiment.

- Use a tiny real slice from the new dataset.
- Verify observation windows, action chunks, and future-state targets line up.
- Train until the model can reconstruct or closely match the tiny action
  target and future-state target.
- Produce inspectable evidence before full training.

Stage 2: full training.

- Scale only after the overfit run is correct.
- Keep the real DP action contract.
- Keep real Cosmos-3 loading and RGB evidence.
- Log policy loss, world-model/future-state loss, and any shared-latent or
  adapter loss separately.

Required evidence:

- dataset manifest;
- model/config manifest;
- action-target sanity report;
- future-state imagination report;
- at least one visual or latent alignment review showing that the policy action
  target and Cosmos future target refer to the same rollout.

Candidate new main code:

- `scripts/world_model/build_joint_dp_cosmos_dataset.py`
- `scripts/training/train_joint_dp_cosmos_overfit.py`
- `scripts/training/train_joint_dp_cosmos_full.py`
- `scripts/world_model/inspect_joint_dp_cosmos_batch.py`

These files do not replace Cosmos-3 or DP with toy models. They should be glue,
adapters, and training entry points around the real assets.
