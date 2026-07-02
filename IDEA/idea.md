# Current Idea: ManiSkill Five-Phase Route

Date: 2026-07-02

## Global Direction

Use ManiSkill `PegInsertionSide-v1` as the only active route.

Active assets:

- 733 approved data: `experiments/maniskill/data/fix3_733/`
- DP checkpoint: `experiments/maniskill/dp_checkpoint/run_90201/checkpoints/`
- RGB Cosmos-3 checkpoint:
  `experiments/maniskill/cosmos3_checkpoint/vision_sft_droid_policy_full1000_rgb_300step_wam/`

## Required Order

1. Reproduce whether the DP checkpoint works on static ManiSkill.
2. Check whether RGB Cosmos-3 can imagine future video and produce an
   action/task chart.
3. Convert Cosmos output into controller-facing signals and combine them with
   DP-compatible execution.
4. Test the corrected Oracle boundary.
5. Test live control without Oracle.

## Controller-Facing World Model Direction

The research review in
`docs/controller_facing_world_model_research_20260702.md` is now part of the
active idea. The key conclusion is:

```text
RGB future / latent future
  -> task-state extraction
  -> candidate action or local primitive proposal
  -> world-model / geometry / executability scoring
  -> trust gate
  -> DP-compatible live execution
```

Do not optimize only for better-looking Cosmos videos. Cosmos-3 must become a
source of explicit controller-facing variables:

- target / hole displacement;
- task-frame and insertion-axis estimate;
- peg-head / hole-center / peg-axis / hole-axis chart;
- lateral, axial, and angular error;
- near-hole, preinsert, contact, and continuability gates;
- prediction-observation trust score;
- executable-action score.

The active bridge should follow the lowest-risk literature pattern for our
assets: GPC-style frozen-policy candidate ranking plus Feedback-WM-style online
prediction-observation correction. That means:

1. the frozen DP and bounded primitives propose real `pd_ee_delta_pose`
   candidate chunks;
2. Cosmos RGB futures plus geometric extraction score task progress and
   executability;
3. the controller executes only short chunks;
4. the system reobserves, updates the trust gate, and replans.

Recommended active modules:

1. RGB task-state extractor: converts live/Cosmos RGB into peg/hole/task-frame
   variables and saves overlays.
2. Candidate generator: DP continuation, small task-frame residuals, bounded
   insertion-axis push, retreat-reapproach, and hold/reobserve.
3. Scorer: ranks candidates by predicted progress, lateral/angular error,
   executability, trust, and discontinuity penalties.
4. Trust gate: decides whether to trust Cosmos, chunk length, handoff mode, and
   whether the result can be method evidence.
5. Physical finisher: executes only logged DP-compatible actions, with gripper
   closed by default and no state intervention.

Full WAM retraining, toy world models, or action-free video-only success claims
are not the next step.

## Non-Negotiable Lesson

A video where the peg is far from the hole and then snaps into the hole is not
success. `set_pose`, source-state restore, saved-state replay, geometric final
placement, and future labels are state interventions. They are diagnostics only
and must be labeled that way.

The new literature reinforces the same point under the name "executability
gap": visually plausible futures can still imply impossible robot actions. Any
Cosmos future must be checked against feasible `pd_ee_delta_pose` action
magnitudes, successful DP insertion action statistics, and discontinuity
checks before it is allowed to influence a live controller.

## Workspace Rule

Keep `experiments/` clean. Use:

- active assets: `experiments/maniskill/`
- new outputs: `experiments/maniskill/runs/<phase>/`
- logs: `logs/<phase>/`
- useless or invalid experiments:
  `/public/home/yanhongru/ICLR2027/archive/Reflex/`

Before a result is treated as success, rule out state intervention by checking
the wrapper, manifest, summary, relevant Python path, action trace, and video.
Git is not a physical cause, and ManiSkill physics is not blamed when the code
path directly overwrites simulator state.
