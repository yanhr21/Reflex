# Oracle Snap Error Review

Date: 2026-07-02

## What Went Wrong

The previous Oracle / geometric final-seat videos were incorrectly treated too
close to success. In the reviewed video, the peg was still far from the hole,
then suddenly appeared seated. That was not physical insertion.

The direct cause was state intervention:

- the old diagnostic path computed a target peg pose from the current hole pose;
- it then used simulator pose setting, such as `set_pose`, to place the peg;
- the rendered final frame was replaced after the state edit;
- the final metric became true after this state edit.

This created the visible snap / suction effect.

## What It Was Not

This was not a MySQL issue. No database behavior caused the peg to move.

This was not a Git issue. Git did not corrupt physics, checkpoints, or videos.
Git only records which code and files are checked out. It cannot cause a peg to
teleport during a rollout unless the executed code path itself contains a state
edit or other behavior that does that.

This was not evidence that ManiSkill physics spontaneously pulled the peg into
the hole. The motion came from explicit simulator-state editing in the
diagnostic code path.

This was not Cosmos-3 success. The reviewed invalid sample had no valid Cosmos
imagination stage for the claimed behavior.

This was not DP success. The final insertion was not produced by DP actions.

## Physics / Git Failure Ruling

The reviewed snap is classified as an implementation and protocol failure, not
as a physics-engine failure and not as a Git failure.

The physical reason is simple: `set_pose`, `set_state`, `set_state_dict`,
source-state restore, saved-state replay, and geometric final placement bypass
ordinary action-driven contact dynamics. If those calls are used in the active
controller-facing path, the simulator is being told to overwrite the state.
The resulting video can look like suction because the object state changes
discontinuously between rendered frames.

A real ManiSkill physics failure would require separate evidence: the exact
action vector should be applied without any state-setting calls, the peg should
still move discontinuously, and the run should log state before / after,
contacts, code path, and video from a compute-node allocation. That evidence
does not exist for the invalid Oracle videos. Therefore the correct cause is
state-intervention misuse plus a reporting failure.

## Why The Mistake Happened

The mistake was a protocol and reporting failure:

1. A diagnostic state-intervention path existed to force a final seated state.
2. The output used names and summaries that contained `success`.
3. The report did not make the state intervention visually and verbally
   impossible to confuse with real insertion.
4. The artifact was allowed to sit near active experiments instead of being
   archived immediately after it was known to be invalid.

## Required Future Handling

- Any `set_pose`, source-state restore, saved-state replay, or geometric final
  placement is diagnostic only.
- If a video contains a snap into the hole, it must be labeled invalid state
  intervention.
- Before any result is called success, inspect the wrapper, manifest, summary,
  relevant Python path, action trace, and video for state-setting calls or
  discontinuous pose jumps.
- Such artifacts must be moved under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/` after they are classified,
  preserving the original repository-relative structure.
- New logs go under `logs/`.
- `experiments/` must contain only current active assets and current active
  experiment outputs.
