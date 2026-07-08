# Dream Diffusion Policy Note

Paper:

- `Dreaming the Unseen: World Model-regularized Diffusion Policy for
  Out-of-Distribution Robustness`
- arXiv:2603.21017
- Date: 2026-03-22
- URL: https://arxiv.org/abs/2603.21017

Relevant idea:

- DDP co-optimizes a diffusion policy and a diffusion world model through a
  shared visual/geometric encoder.
- The policy learns behavior cloning over action chunks.
- The world model predicts future observation latents over aligned temporal
  chunks.
- At inference, DDP uses real-imagination discrepancy to detect unreliable or
  out-of-distribution observations, then uses autoregressive imagined latents
  to keep generating actions.

Current repository adaptation:

- Use real Cosmos-3 as the world-model side.
- Use the real DP checkpoint / DP action contract as the policy side.
- Build a new dataset where action chunks and future-state imagination targets
  are aligned.
- Treat this as a DDP-inspired route unless official DDP code is found and
  integrated.

Current code status:

- Search did not find an official DDP implementation. Do not claim official
  reproduction.
- Do not replace DDP/Cosmos/DP with toy models.
