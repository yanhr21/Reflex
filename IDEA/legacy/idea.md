# Current Idea: Static Training, Dynamic Deployment

Date: 2026-07-02

## Core Claim

The paper idea is not Oracle.

The core claim is a bridge: train the manipulation skill in static scenes, then
deploy that skill in dynamic scenes by adding a mechanism that connects the
static skill to the changed scene.

For the current route, the base skill is the existing DP checkpoint for static
ManiSkill `PegInsertionSide-v1`. The dynamic evidence comes from RGB Cosmos-3.
The bridge is the part that turns dynamic-scene evidence into a deployment
decision for the static skill.

## What The Bridge Means

The bridge is not a separate toy policy and not a new simplified world model.
It should preserve the real DP checkpoint and the real RGB Cosmos-3 checkpoint.

At the current level, the bridge only needs to answer the central deployment
question:

```text
Given a policy trained in static peg insertion, and a dynamic scene observed
through RGB/Cosmos evidence, when and where can the static skill still be used?
```

Detailed mechanisms such as geometry, IK/servo, candidate scoring, trust gates,
and continuability checks are implementation choices inside the bridge. They
should not distract from the main claim, but the bridge itself is the method
direction and must be built before Oracle is meaningful.

## Role Of Oracle

Oracle is not success and not the method.

Oracle is only an upper-bound diagnostic for the bridge. It may be used after:

1. the static DP checkpoint has been checked;
2. RGB Cosmos-3 has produced dynamic-scene evidence;
3. the bridge has connected that evidence to a static-skill deployment
   decision.

The Oracle step tests whether the bridge has found a meaningful finish
condition. It must be labeled with `method_evidence_allowed=false` and cannot
be reported as physical insertion success.

## Active Route

Use only ManiSkill `PegInsertionSide-v1` for the current implementation.

Active assets:

- Approved 733 ManiSkill data:
  `experiments/maniskill/data/fix3_733/`
- Original DP checkpoint:
  `experiments/maniskill/dp_checkpoint/run_90201/checkpoints/`
- RGB Cosmos-3 checkpoint:
  `experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam/`

## Required Order

1. Verify the original DP checkpoint on static ManiSkill.
2. Verify RGB Cosmos-3 imagination from RGB history on the approved data.
3. Build the first bridge from Cosmos dynamic evidence to static-skill
   deployment.
4. Run the corrected Oracle boundary diagnostic for that bridge.
5. Replace Oracle with live physical control only after the diagnostic is
   understood.

## Evidence Rules

Do not call state intervention success.

Any result that uses `set_pose`, `set_state`, `set_state_dict`, source-state
restore, saved-state replay, geometric final placement, future labels, or a
hand-selected suffix is not physical insertion success. If the peg is far from
the hole and then snaps into place, the result must be reported as invalid
state intervention.

The current valid reset-to-end live insertion success count remains `0` until a
new run proves otherwise under the active protocol.

## Workspace Rule

Keep active work under the current ManiSkill layout:

- active assets: `experiments/maniskill/`
- new outputs: `experiments/maniskill/runs/<phase>/`
- logs: `logs/<phase>/`
- invalid or obsolete experiment artifacts:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/`

Do not revive old `experiments/world_model_task_rebinding`,
`experiments/dp_peg1000`, OpenPI, LIBERO, robosuite, truepeg, source-restore,
saved-state replay, suffix, or geometric final-seat routes as active work.
