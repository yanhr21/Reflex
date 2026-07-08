# Cosmos Imagination Guides Policy

The final active method should use Cosmos imagination to guide policy control.

The desired form follows the useful part of Dream Diffusion Policy:

- policy actions are still generated through a DP-compatible controller path;
- Cosmos-3 predicts future state or future latent state from the same recent
  history;
- real observations can be compared with imagined future state to detect when
  the scene has shifted;
- during disturbed or uncertain periods, imagined latents or Cosmos-derived
  features condition the next policy action chunk;
- the robot physically recovers by executing controller actions, not by
  simulator-state edits.

This stage is only meaningful after the new dataset and joint overfit/full
training stages exist. It should not depend on Oracle finishing or manual
physical takeover.
