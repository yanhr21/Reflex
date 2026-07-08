# Imagination-Guided Policy Plan

Objective: use Cosmos imagination to guide policy action generation without
Oracle.

Inference concept:

- encode live RGB/proprio history;
- generate a DP-compatible action chunk;
- ask Cosmos-3 for future state or future latent prediction for the same
  rollout window;
- compare real observation with imagined state to detect scene shift or
  unreliable perception;
- condition or adjust later policy chunks using imagined future state or a
  documented Cosmos-derived adapter;
- execute only valid controller actions in the simulator.

Required evidence:

- reset-to-end live rollout;
- action trace showing controller commands;
- Cosmos future-state evidence for the same time window;
- real-imagination discrepancy or equivalent trust signal;
- final physical outcome and rendered review;
- audit proving no state edit, snap, target-assisted self-insertion, or manual
  hidden finisher.

Candidate new main code:

- `scripts/world_model/eval_imagination_guided_policy.py`
- `scripts/world_model/compute_real_imagination_discrepancy.py`
- `scripts/world_model/apply_cosmos_policy_conditioning.py`

Oracle is not a prerequisite for this stage. Old Oracle outputs can only be
used as failure context.
