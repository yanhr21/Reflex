# Dream Diffusion Policy Plan Note

Reference:

- `Dreaming the Unseen: World Model-regularized Diffusion Policy for
  Out-of-Distribution Robustness`, arXiv:2603.21017.
- URL: https://arxiv.org/abs/2603.21017

What to borrow:

- co-train policy and world-model objectives;
- align action chunks with future-state chunks;
- use a shared or adapter-aligned representation;
- use real-imagination discrepancy as an OOD / trust signal;
- use imagined future state to keep action generation stable when observation
  is disturbed.

What not to claim:

- Do not claim official DDP reproduction unless official implementation is
  found and integrated.
- Do not claim Oracle success as method success.
- Do not use a toy world model as a substitute for Cosmos-3.

Repository adaptation:

- policy side: existing DP action contract and checkpoint context;
- world-model side: RGB Cosmos-3 checkpoint and RGB/future-state evidence;
- bridge: joint dataset and training code that puts both objectives into an
  aligned training route.
