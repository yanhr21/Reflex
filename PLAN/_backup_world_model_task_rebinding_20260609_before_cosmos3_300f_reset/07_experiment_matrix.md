# Experiment Matrix

## Main Metrics

Report:

- success_once
- success_at_end
- final peg-head error in hole frame
- dynamic event handled before insertion completion
- number of rebinds
- handoff count and handoff success
- infeasible fraction
- model prediction error and uncertainty calibration
- continuability false-positive rate

Workability check: success alone is not enough. We need to know whether the
world model, bridge, and handoff each did their job.

## Core Conditions

Use shared seeds where possible:

```text
S0 static baseline
D1 hole moves then stops
D2 hole keeps moving
D3 hole reverses near insertion
D4 peg disturbed but still grasped
D5 peg dropped and regrasp required
D6 unreachable or too-fast infeasible cases
```

Workability check: these are evaluation conditions, not controller branches.
The controller should receive only object slots and dynamics.

## Ablations

Compare:

- frozen DP only
- frozen DP with simple tracking offset
- servo/IK bridge without `C_pi`
- `C_pi` gate without learned world model
- world model without `C_pi`
- full rebinding controller
- later: RGB-D slot extractor version
- later: DDP-style baseline

Workability check: this isolates whether gains come from prediction, physical
bridging, continuability, or perception.

## Pilot Gates

Do not scale experiments until:

- dynamic perturbations are visually and numerically verified
- state traces can be replay-rendered
- constant-velocity baseline is implemented
- first world model beats constant velocity on task-frame prediction
- first `C_pi` model is calibrated enough to reduce false handoffs
- first controller improves over simple tracking on at least stopped and moving
  target cases
- full RGB-D data pass strict inspection
- RGB-D-derived slots/latents feed the world model and controller without
  using oracle slots as method inputs

Workability check: scaling before these gates would only produce expensive
unclear failures.
