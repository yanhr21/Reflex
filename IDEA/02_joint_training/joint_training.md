# Joint Training

The main method direction is Dream Diffusion Policy style joint learning:
ordinary Diffusion Policy behavior and Cosmos-3 future-state imagination should
be trained in the same aligned space.

The goal is not to build a separate Oracle controller. The goal is to make the
policy and world model learn from the same episodes so that a policy rollout
and a Cosmos imagined future describe the same physical process.

Required outcome:

- the policy can output normal rollout actions that drive the peg / wooden
  stick into the hole through valid controller actions;
- Cosmos-3 can imagine future visual or latent state for that rollout;
- the shared or aligned representation lets future imagination regularize,
  condition, or diagnose policy action generation.

Training must start with an overfit experiment. The overfit experiment is not
final evidence; it is the first correctness check that the dataset format,
action chunks, observation windows, and Cosmos future target are synchronized.

Only after the overfit experiment produces inspectable evidence should full
training begin.
