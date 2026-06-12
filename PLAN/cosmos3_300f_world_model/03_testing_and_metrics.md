# Testing And Metrics Plan

## Structural Tests

Every generated validation or test artifact must carry:

- source `sample_id`,
- source H5 path,
- source RGB video path,
- split,
- scenario,
- frame span,
- prefix boundary,
- intended RGB/state frame count,
- intended action step count,
- generated RGB frame count,
- predicted action step count,
- readout target frame count.

Fail the artifact as method evidence if any intended length differs from the
generated/measured length without a documented full-episode convention.

## Validation Generation

Start with a fixed scenario-diverse validation panel after training approval:

- static/none,
- pre-motion target forecast,
- observed moving target,
- move-stop,
- reverse,
- peg disturbance,
- peg drop/regrasp.

The panel should generate full-episode or remaining-episode rollouts from
causal prefixes. It must not generate 129-frame clips as the method sample.

## Metrics

Target object:

- onset classification AUROC/F1,
- onset timing error in frames,
- target path RMSE over the supervised future,
- final target pose error,
- target-motion false positives in `none` cases.

Robot/peg:

- predicted action MSE or diffusion negative log-likelihood over intended
  action steps,
- TCP and peg pose readout error,
- peg-head-in-hole-frame error,
- grasp/hold classification accuracy,
- inserted predicate accuracy,
- failure labeling for peg-drop cases where regrasp is absent or visually bad.

Video/world reconstruction:

- generated-vs-reference RGB diagnostics over the exact intended frame range,
- contact-sheet review for every major claim,
- explicit rejection of cropped comparisons.

Policy resume readiness:

- DP resume-state distance to the static policy manifold,
- whether the generated/readout state would allow DP continuation,
- live simulator success only after the SFT/readout evidence passes review.

## Receding Test Interface

After review and after SFT evidence exists, test a receding interface:

1. observe live prefix,
2. generate the remaining full-episode-consistent rollout or a causal
   remaining-horizon rollout,
3. execute only a short action prefix,
4. reobserve real RGB/state,
5. refresh the prefix and repeat.

This is an execution strategy, not a permission to train on 128-action chunks.
The authority for success is the live simulator final state plus video review.
