# Imagination-Guided Policy TODO

- [ ] Define the controller-facing interface from Cosmos imagination to policy:
  shared latent, conditioning feature, discrepancy gate, or adapter.
- [ ] Add real-imagination discrepancy computation.
- [ ] Add imagination-conditioned policy evaluation wrapper.
- [ ] Run reset-to-end live rollout only after joint training evidence exists.
- [ ] Produce action trace, Cosmos future evidence, trust signal, final outcome,
  and rendered review for each claim.
- [ ] Reject any rollout with state intervention, snap, target-assisted
  self-insertion, manual hidden finisher, or future-label controller input.
