# Object-State World Model

## Model Choice

Updated 2026-06-04: the lightweight object-centric state world model is only a
baseline and interface scaffold. It is useful for quickly validating exact1000
RGB-D-derived slot files, task-frame targets, uncertainty plumbing, and
controller integration, but it should not be the final publishable backbone.

Primary method direction: integrate a pretrained world-foundation-model
backbone, preferably Cosmos-family physical-AI world models, then train the
task-frame head/controller interface on RGB-D manipulation data:

```text
inputs:
  multi-view RGB-D video history
  robot proprio/action history
  optional current RGB-D-derived object/task slots

foundation backbone:
  pretrained physical-AI video/world model or tokenizer
  post-trained or adapted on the exact RGB-D manipulation dataset

outputs for controller:
  future latent/video predictions
  decoded future hole/peg/TCP task slots
  grasp/insert predicates
  calibrated uncertainty or non-conformity scores
```

Keep the current lightweight object-slot Transformer as a diagnostic baseline:

```text
inputs:
  history of hole, peg, TCP, qpos, qvel, grasp/contact predicates
  recent actions or bridge primitive parameters

outputs:
  future hole pose distribution
  future peg/TCP/qpos prediction
  future insertion and grasp predicates
  uncertainty
  feasibility/value estimates
```

Recommended first architecture: causal Transformer or RSSM with an ensemble.

Workability check: this is small enough to train from simulated rollouts and
rich enough to predict future task frames, but it is not enough for the final
research claim. A stronger foundation-world-model branch must be added or used
as the main method/comparison before claiming publishable RGB-D world-model
evidence.

## Prediction Targets

Train multi-horizon predictions:

```text
x_{t+1:t+K}
hole_pose_{t+1:t+K}
peg_head_in_future_hole_frame_{t+1:t+K}
grasped_{t+1:t+K}
inserted_{t+1:t+K}
```

Use task-frame errors as primary metrics, not just absolute pose errors.

Workability check: predicting absolute hole pose alone is too easy and not
enough. The controller needs future relative geometry and uncertainty.

## Losses

Use:

- pose regression loss in world frame and hole frame
- velocity/acceleration consistency loss
- binary cross entropy for grasped/inserted predicates
- negative log likelihood or ensemble variance for uncertainty
- optional contrastive real-vs-imagined discrepancy calibration

Workability check: every loss should support an online decision. If a loss does
not affect prediction, uncertainty, feasibility, or continuability, leave it
out of the first version.

## Discrepancy Signal

Use real-imagination discrepancy as a rebind trigger:

```text
D(t) = distance(observed object slots, predicted object slots)
```

This is not the final success criterion. It means "the current binding is no
longer reliable; replan in task-frame space."

Workability check: this is more useful than a generic failure detector because
it points to the model component that became wrong: target motion, peg state,
grasp state, or robot state.

## Failure Criteria

Stop investing in this model variant if:

- future hole-frame error is not better than constant-velocity baseline
- uncertainty is not calibrated against prediction error
- the model cannot distinguish grasped from dropped episodes
- predictions do not improve controller decisions in ablation

Workability check: a world model is only useful if it changes action selection
or rebind timing. Pretty rollouts are not enough.
