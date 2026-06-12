# World-Model-Conditioned Policy Distillation

## Physical Problem

The current geometric controller can move toward a predicted task frame, but it
does not reliably preserve the base DP manipulation skill. The 2026-06-06
Cosmos3 receding-controller video shows the robot approaching the moved hole
while the peg is not visually held through insertion. That is not a publishable
controller: dynamic task completion requires the robot to keep the peg, align
it, and insert it in the changed world.

## Method

Train a policy, initialized from or distilled against the official/static DP,
that receives:

```text
current RGB-D-derived slots/proprio/history
+ world-model predicted short task trajectory
+ candidate action or bridge context
-> action sequence in the original control mode
```

The policy must be trained on two data types:

- base-skill preservation data from official/static DP demos, so the new policy
  does not forget grasping, holding, and inserting in the original task.
- positive takeover teacher data from dynamic scenes, but only when metric
  success and inspected video show that the peg is physically held and inserted.

Failed takeover runs are not positive data. They are negative diagnostics for
perception drift, bad world-model readout, or bad policy conditioning.

The dynamic execution entry for this corrected path is
`control_policy=wm_policy_takeover` in
`scripts/world_model/evaluate_rebinding_controller.py`. In that mode the
frozen/static DP keeps responsibility for the initial skill until the configured
takeover condition, and a trained `WMConditionedDiffusionPolicy` checkpoint
executes takeover actions. The bridge planner is not the final controller in
this mode; it only constructs the causal world-model/task-frame condition that
the policy consumes. By default this path refuses stale one-shot Cosmos
interfaces and requires receding teacher-forced or live-refresh world-model
provenance.

If admitted positive takeover distillation data are not available, the
fallback should still follow the same physical idea rather than reverting to a
standalone bridge. The diagnostic entry
`control_policy=wm_dp_prior_mpc` implements the minimal Dream-Diffusion-Policy-
style alternative: Cosmos/world-model prediction supplies an imagined
short-horizon task-frame trajectory, while the frozen DP action remains the
manipulation prior and a small candidate set is selected by task-progress
scoring. `control_policy=wm_dp_prior_sequence_mpc` extends this fallback to
short DP action chunks scored against the task-frame target before rolling
execution. These paths are diagnostic until they use RGB-D-derived
slots/latents and pass metric plus visual evidence.

## Losses

- `L_base_dp_bc`: imitate official/static DP action sequences.
- `L_takeover_bc`: imitate successful world-model-conditioned takeover
  actions.
- `L_grasp_hold`: penalize action/trajectory choices that lose the peg or make
  the visual grasp inconsistent with metric grasp.
- `L_task_frame_progress`: prefer actions that reduce predicted and real
  peg-head-in-hole error without violating contact/orientation constraints.
- `L_switch`: imitate when to keep DP, when to execute a takeover chunk, and
  when to retreat/replan.

## Dream Diffusion Policy Connection

The relevant idea is not to report video prediction alone. The useful pattern
is to use a learned world model to imagine or predict a short future trajectory
and train/control the policy around that imagined path. For this project, the
world-model path is Cosmos3 plus task-state readout or a better published
world-model backbone; the manipulation policy must still be trained to execute
valid peg-in-hole actions.

2026-06-06 diagnostic result: the minimal DP-prior MPC fallback is not enough.
Conservative weighting preserved grasp and temporarily reduced lateral error
to `0.0165m`, but did not insert; aggressive weighting collapsed toward
geometric bridge behavior and got worse; an early handoff diagnostic also
failed. The next controller path should therefore use a stronger expert or
trajectory-level teacher and learned continuability, not more threshold-only
tuning of the bridge.

Follow-up sequence-MPC diagnostic also failed final success but sharpened the
failure: it preserved grasp and reached lateral yz error `0.00011m`, while
insertion x only reached `-0.09875m`. A phase-scoring follow-up was worse and
was reverted. This points to contact/insertion-axis execution as the controller
gap, not merely lateral task-frame rebinding.

2026-06-07 continuation: fixed action-prior replay is now also exhausted as
the controller mechanism. DP-only and converted official-planner insertion
chunks can be useful action priors, but one-, two-, and four-chunk replay
libraries failed on the dynamic seed `702000`; no online planner candidate
from those libraries reached `inserted_any=true`, and allowing non-inserting
teacher followers made the final state worse. The next DDP-style controller is
therefore not another replay gate or pure distillation attempt. It should be a
short-horizon learned or optimized action generator conditioned on
Cosmos/RGB-D task state plus current robot/object state, execute only a short
prefix, then re-observe and regenerate. This preserves the original
world-model rebinding objective while moving past the failed fixed-replay
scaffold.

Follow-up smoke result: a 7-chunk action-prior manifest and a frozen-DP
1000-iteration action-generator smoke ran successfully, and small static
preservation remained `success_at_end=0.6`. Dynamic seed `702000` still failed:
the learned actions over-shot lateral/vertical motion and never inserted, even
with a conservative `0.35` action blend. Therefore the next generator cannot
be only a global WM adapter over the frozen DP condition. It must explicitly
condition action generation or online optimization on current peg/hole/TCP
geometry and regenerate after short live chunks.

Follow-up controller integration: the learned action generator is now connected
to `wm_dp_prior_rollout_mpc` as a scored short-horizon candidate, and MPPI can
sample around the learned generator as well as the frozen DP prior. Direct
learned-generator MPC on seed `702000` selected the learned generator for 48
executed steps but failed and had no simulated inserted candidate. MPPI around
DP/learned bases found 15 simulated inserted candidates, but live execution
still failed final success. A stricter `execute_chunk=1` MPPI variant was too
slow and cancelled before metrics, while the allocation was kept. Current
controller bottleneck is planner/live contact-state mismatch or
insertion-execution fidelity; more fixed replay, larger MPPI fanout, or pure
distillation is not the next justified method step.

Policy distillation must protect the base DP backbone. A scaffold full-parameter
mixed run fit losses but dropped static preservation to `success_at_end=0.1`;
the corrected training option is to freeze the loaded DP `noise_pred_net` and
train only WM adapter/auxiliary modules unless there is stronger evidence that
full fine-tuning preserves base skill. Frozen-base scaffold training preserved
static DP in a 5-episode smoke, but still failed dynamic takeover and remains
scaffold-only. Therefore freezing is a preservation requirement, not method
success.

## Evidence Required

A distilled controller result is not accepted unless all are true:

- the policy preserves static DP capability on official/static evaluation;
- dynamic rollout uses RGB-D-derived slots or latents at controller time;
- the world model supplies a causal short-horizon predicted task trajectory;
- the video/contact sheet is opened and shows the peg remains held through
  approach/insertion;
- final success is measured from real simulator metric state;
- failed rollouts and slot/readout drift are recorded instead of hidden.

## Failure That Falsifies The Path

The current path is falsified for a run if the controller visibly loses the peg,
the policy cannot reproduce static DP behavior, the world-model-conditioned
features are ignored, or final metric success is absent after the dynamic
event. These failures should block positive distillation data, not be papered
over by threshold changes.
