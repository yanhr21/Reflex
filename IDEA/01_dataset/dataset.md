# Dataset Construction Idea

The first active task is to build the dataset for the new method. The method
setting is:

- the official Diffusion Policy checkpoint is a state-based static insertion
  skill;
- Cosmos-3 remains RGB-based and imagines future visual state;
- the interface between them is a future target / hole frame, plus timing,
  velocity, uncertainty, and an adapter that makes the prediction usable by the
  state-based policy.

The intended control loop is not "Cosmos outputs actions" and not "Diffusion
Policy samples random actions for Cosmos to score." The intended loop is:

```text
RGB history
-> Cosmos future RGB / latent
-> future hole frame readout
-> real robot state + predicted future hole state
-> frozen or protected state-based DP + moving-frame adapter
-> DP-compatible action
```

## Confirmed Asset Facts

- The official DP checkpoint was trained from
  `trajectory.state.pd_ee_delta_pose.physx_cpu.h5`; its manifest records
  `capture_video=false`, `control_mode=pd_ee_delta_pose`, `obs_horizon=2`,
  `pred_horizon=16`, and `act_horizon=8`. It should be treated as
  state-based, not RGB-based.
- The official 1000 demo exists locally as state/action H5 and JSON under
  `data/official_replay/PegInsertionSide-v1/motionplanning/`.
- The official 1000 demo does not currently have a ready RGB/mp4/jsonl export
  in the active workspace.
- The old `fix3_733` set contains H5 files and audits, but it is not a clean
  final training set for the new claim because many samples came from or gave
  opportunity to the original DP checkpoint in moving-object scenes.
- The current Cosmos-3 checkpoint is useful as previous training context, but
  its old RGB/video-action jsonl paths are legacy and not a sufficient active
  dataset definition.

## Required Dataset Principle

The base policy may learn precise insertion from static data. The world model
must learn dynamic future evolution from RGB. The adapter must learn how to
use a predicted future target frame without corrupting the base static skill.

Therefore data must be split by training role, not simply by success/failure.

## Data Classes

### A. Official Static Expert

Source: official 1000 ManiSkill demos.

Purpose:

- preserve static insertion skill;
- train / validate state-based DP action behavior;
- derive static insertion phase and target-centric relative trajectories;
- render RGB for Cosmos static alignment.

Allowed losses:

- DP behavior-cloning loss;
- static future RGB / latent loss;
- target-centric trajectory / phase supervision.

Not enough for:

- dynamic target generalization;
- moving-target insertion claims.

### B. Dynamic RGB Observation

Source: newly generated moving-target / moving-object episodes. These episodes
do not need to succeed.

Purpose:

- train Cosmos to predict moving target / hole future state from RGB history;
- learn target velocity, acceleration, time-to-contact candidates, and
  uncertainty;
- expose continuous target motion, not just final static position.

Allowed losses:

- RGB future prediction;
- future latent prediction;
- future target / hole pose and velocity readout;
- uncertainty / trajectory consistency.

Not allowed:

- treating failed robot actions as positive DP expert actions.

### C. Frozen-DP Dynamic Failure

Source: frozen official DP checkpoint deployed in dynamic scenes where it
often fails.

Purpose:

- document the gap between static DP and dynamic deployment;
- train negative / infeasible / miss / jam / target-assisted labels;
- train real-imagination discrepancy and failure diagnostics;
- teach which future predictions are not enough for insertion.

Allowed losses:

- failure classification;
- contrastive separation between insertion-corridor futures and miss/jam
  futures;
- discrepancy / OOD signal;
- executability or feasibility labels.

Not allowed:

- direct positive action imitation from failed action chunks;
- counting target-assisted insertion as success.

### D. Future-Frame Cooperation Teacher

Source: newly generated episodes where a legal controller uses a known future
target / hole trajectory to produce physically valid moving-frame insertion
actions.

Purpose:

- train the adapter that connects future target frames to the static DP skill;
- produce examples where the robot actively inserts while the target is moving;
- create the missing DP-Cosmos cooperation distribution.

Important distinction:

- This is not simulator-state editing.
- The target motion must be continuous and logged.
- The robot must act through `pd_ee_delta_pose` or another approved controller.
- The teacher may use ground-truth future target trajectory during data
  generation, but that fact must be recorded and must not be presented as a
  deployed method.

Allowed losses:

- adapter / residual action supervision;
- moving-frame state conditioning;
- phase / timing supervision;
- relative velocity at contact;
- insertion corridor progress.

### E. Cosmos-Predicted Cooperation

Source: rollouts where the future target frame is supplied by Cosmos/readout,
not by ground truth.

Purpose:

- close the train-test gap;
- teach the adapter to tolerate Cosmos prediction error;
- test the actual method.

Allowed losses:

- adapter fine-tuning with predicted future target frames;
- robustness to target-frame noise;
- uncertainty-conditioned behavior;
- live rollout evaluation.

This class comes after A-D exist. It should not be the first dataset.

## Representation

Every usable sample should expose both world-frame and target-frame fields:

```text
T_hole(t)
T_hole(t + tau)
v_hole(t + tau)
T_peg(t)
T_ee(t)
relative_peg = inv(T_hole) * T_peg
relative_ee  = inv(T_hole) * T_ee
action_pd_ee_delta_pose
phase / tau / uncertainty
success, failure, miss, jam, target_assisted, state_intervention
```

The future target should not be only a final position. It should include at
least a future pose, timing `tau`, velocity, and uncertainty. Continuous moving
targets require a future trajectory window, not one isolated endpoint.

## Acceptance Rules

Accept as positive policy / adapter data only if:

- the robot actively drives the peg / wooden stick into the hole;
- target / hole motion is continuous and logged;
- the action trace is legal controller action;
- no `set_pose`, `set_state`, state restore, snap, target-assisted
  self-insertion, or hidden manual finisher is involved;
- RGB evidence and state/action evidence refer to the same time window.

Use as negative or diagnostic data if:

- DP misses because the target moved;
- insertion is target-assisted;
- the peg jams or approaches with bad relative velocity;
- Cosmos future looks plausible but does not yield executable insertion.

Reject from active training if:

- state intervention or snap creates the outcome;
- source labels or future labels are used as hidden controller inputs in a
  deployed-method claim;
- timestamps/action windows cannot be aligned.
