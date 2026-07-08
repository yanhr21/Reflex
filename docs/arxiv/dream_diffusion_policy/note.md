# Dream Diffusion Policy

Paper:

- `Dreaming the Unseen: World Model-regularized Diffusion Policy for
  Out-of-Distribution Robustness`
- arXiv:2603.21017
- Date: 2026-03-22
- URL: https://arxiv.org/abs/2603.21017

Main points for this project:

- Jointly train policy behavior and world-model prediction instead of treating
  Oracle as the main experiment.
- Align action chunks and future-state chunks.
- Use shared or adapter-aligned representation between policy and world model.
- Use real-imagination discrepancy as a trust / OOD signal.
- Let imagination condition policy behavior during scene disturbance.

Implementation status:

- No official DDP code was found in the current web search.
- This repository should use the DDP idea as a design reference and clearly
  label the method as DDP-inspired unless official code is later integrated.
