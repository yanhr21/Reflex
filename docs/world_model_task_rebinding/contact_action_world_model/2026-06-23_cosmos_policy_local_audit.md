# 2026-06-23 Cosmos Policy Local Audit

## Scope

This was a read-only local audit while the source-suffix contact-action
generator was running in Slurm allocation `146658`. No Cosmos inference,
training, replay, or import check was run on the login node.

## Available Local Assets

- `checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID`
- `checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID-DCP`
- `checkpoints/cosmos3/Cosmos3-Nano-DCP`
- `checkpoints/cosmos3/Cosmos3-Nano`

The DROID checkpoint README describes `Cosmos3-Nano-Policy-DROID` as a policy
model that generates robot action trajectories from language instructions and
visual observations on DROID-style robot data. Local files include transformer,
vision encoder, tokenizer, VAE, and converted DCP model shards.

## Existing Repo Entrypoints

- `scripts/slurm/run_cosmos3_live_prefix_policy_inference_in_allocation.sh`
- `scripts/world_model/extract_cosmos3_policy_action_chunk.py`

These support a live-prefix inference shape: build a causal prefix input, run
Cosmos inference in a compute step, then extract a short denormalized robot
action chunk from the predicted `300 x 32` action sidecar.

## Gap

The current local entrypoint is not yet a Cosmos Policy-style action/value/video
post-training pipeline:

- it is inference/extraction, not training;
- its defaults point to older 2026-06-12 paths and must be retargeted to the
  active 733 clean-dense/root before use;
- it extracts action chunks but does not train a value/reward/progress head;
- it does not yet merge contact-action labels, DP96 replay outcomes, or saved
  live failure states into action/value training;
- it does not by itself solve the insertion-action coverage gap found in the
  latest live panel.

## Conclusion

The strongest same-family direction is feasible enough to keep active:
local Cosmos3 Policy-DROID assets exist, and the repo already has a causal
live-prefix action extraction wrapper. The missing work is adaptation into a
full contact-action/value training path on the accepted 733 RGB/action data.
Until that exists, the current source-suffix generator is a practical local
baseline, while Cosmos Policy-DROID remains the preferred backbone for a
cleaner action/value/video head.
