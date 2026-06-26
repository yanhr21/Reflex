# Training Plan

## Stage 0: No-Training Preflight

Before any Slurm training job:

1. Rebuild a clean full-episode condition manifest from the approved RGB
   dataset and active source H5s.
2. Refuse the manifest if any row has a 129/128, 93-frame, cropped, or
   mismatched length contract.
3. Write one manifest per split with explicit fields:
   `sample_id`, `scenario`, `source_h5`, `rgb_video`, `frame_start=0`,
   `frame_end=300`, `num_rgb_frames=301`, `num_action_steps=300`,
   `prefix_policy`, `causal_condition_fields`, and `label_fields`.
4. Inspect a small validation set of RGB videos/contact sheets only as data
   sanity evidence; do not launch SFT during this reset.

## Stage 1: Full-Episode Conditioned SFT

Train Cosmos3 as a full-episode world/action model rather than as a short clip
generator.

Inputs may include:

- RGB video prefix sampled from the same full episode,
- causal robot action/proprio history up to the prefix boundary,
- causal task-state summaries up to the prefix boundary,
- target object motion history up to the prefix boundary.

Targets may include:

- future RGB frames within the same 300-step episode,
- action sequence targets for the remaining episode or for an execution-head
  mask aligned to the same full record,
- readout targets for target pose/path/onset, peg pose, TCP pose, grasp,
  inserted predicate, and peg-head-in-hole geometry.

The prefix policy should cover the physical regimes needed by the user:

- pre-target-motion monitoring,
- observed target-motion response,
- post-motion continuation,
- peg disturbance while held,
- peg drop/regrasp before continuation.

The prefix policy must not turn into separate 128-action windows. A sampled
prefix is a mask over the full episode record, not a truncated training sample.

Training and all nontrivial export/preflight/debug work must run inside a
Slurm allocation, not on the login node. Prefer a tmux-held `salloc` allocation
so the same GPU resource can be reused for debugging, export, training,
validation generation, video review preparation, and monitoring. A run only
counts as training evidence if it uses at least one GPU for at least one hour
and is monitored until validation no longer clearly improves. Prefer 4 or 8
GPUs when those allocations start promptly; use a 1-GPU allocation to make
progress instead of waiting indefinitely for larger shapes. If a larger
allocation up to 8 nodes / 64 GPUs is available and actually useful, it may be
used, but the method must not wait passively for it.

## Stage 2: Monitor And Joint WAM Heads

The trained model should expose two controller-facing outputs:

1. Target monitor/readout:
   - target motion onset probability,
   - target path and final pose,
   - uncertainty or consistency score,
   - DP-continuation warning when the target state leaves the static manifold.
2. Joint world/action rollout:
   - future RGB reconstruction,
   - robot/peg/TCP state readout,
   - executable action sequence aligned to the full episode contract,
   - grasp/regrasp and insertion progress predicates.

The readout can be a decoder on generated RGB/latent features, but it is not a
replacement world model. Controller-facing evidence still requires generated
RGB/latent rollout, action prediction, readout metrics, and visual review.

## Stage 3: Training Evidence

A training run is not method evidence by validation loss alone. Active evidence
requires:

- strict preflight manifest,
- Slurm/H200 training record,
- saved checkpoint and config,
- generated validation rollouts under the full-episode/equal-length contract,
- action/state/readout metrics over exactly the intended frame/action lengths,
- inspected visual contact sheets or videos showing the target event, peg
  continuity/recovery, and final state.
- generated validation videos/contact sheets saved for human review backup.

Controller/DP integration remains out of scope until the full-episode SFT
evidence exists.
