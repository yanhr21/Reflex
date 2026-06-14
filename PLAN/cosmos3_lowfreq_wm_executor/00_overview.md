# Low-Frequency Cosmos WM + High-Frequency Executor Plan

## Boundary

This is a new method branch. It does not replace the recorded
`cosmos3_300f_world_model` evidence, and it must not be mixed with the old
direct raw-Cosmos-action checkpoint claims.

The current iter2700 evidence says the implementation contract is now mostly
repaired: full 300 actions / 301 frames, causal target-motion detection,
explicit controller annotation, and same-source pure-DP comparison exist. The
remaining failure is method effectiveness: direct raw Cosmos action chunks are
not reliable enough to put the peg into a DP-continuable insertion state.

This branch keeps the original objective: finish the changed manipulation task.
It does not restore old layouts, use future object poses as controller
conditions, relax final-state success, or report generated sidecars as the
authority for handoff.

## Plain Summary

The old active closed loop asks Cosmos to directly output robot actions every
short chunk. That is too brittle and likely too slow for real deployment.

The new proposal separates time scales:

- Cosmos is a low-frequency world/task model. It predicts where the task frame
  is going and what relative peg-hole state should be reached.
- A smaller executor runs high-frequency robot control. It follows the
  world-model task plan while staying close to the frozen DP skill prior.
- Real observations remain the authority. If the target moves, uncertainty
  grows, contact fails, or DP handoff is unsafe, the system replans.

Dense receding is a training-data repair, not a runtime requirement to call the
large world model every 8 simulator steps forever.

## Why The Current Direct Raw-Action Loop Fails

Current evidence:

- Val comparison: Cosmos closed loop succeeds on `1/3`, same-source pure DP on
  `3/3`.
- Hard-screen-2 comparison: among six full-episode pure-DP failures, Cosmos
  rescues only `1/6`.
- Hard failures are dominated by real-state `C_pi` y/z blocks, occasional
  grasp loss, and action direction/scale mismatch.
- The old condition root has `1193/2899` role/mode mismatches.
- Live-query coverage audit found `74/173` live Cosmos queries without a local
  role/mode-consistent training neighbor.

Interpretation:

The loop now runs, but the raw actions predicted by Cosmos are not reliable.
The model often sees a moving target and partly reacts, but it does not
consistently drive the grasped peg into the moved-hole insertion manifold. When
it is near the gate, DP handoff can still drift out of continuability.

## Proposed Architecture

```text
live RGB/state history
  -> causal target-motion / contact / uncertainty monitor
  -> low-frequency Cosmos task-world update
       predicts: future hole/task frame, desired peg-head path,
                 grasp/contact/insertion risk, optional coarse action hints
  -> high-frequency executor
       inputs: current observation, DP action prior, predicted task path,
               current peg-hole/TCP relation, contact state
       outputs: short executable action chunk
  -> real simulator / robot execution
  -> real-state C_pi handoff check
       if continuable: frozen DP short chunks
       if not continuable: executor continues or Cosmos replans
  -> reobserve and repeat
```

## Runtime Contract

Do not assume Cosmos must be called every 8 actions at deployment time.

Use a low-frequency WM schedule:

- Always call Cosmos when causal target motion is first detected.
- Call Cosmos again when:
  - target motion changes direction or speed;
  - the executor's measured peg-hole error stops improving;
  - contact/grasp state changes;
  - `C_pi` rejects DP handoff for too many consecutive chunks;
  - model uncertainty or readout disagreement is high.
- Otherwise let the lightweight executor run for a longer chunk, such as
  `16`, `24`, or `32` actions, with real observation checks between chunks.

The exact chunk length is an evaluation variable, not a fixed claim. Every run
must record:

- number of Cosmos calls;
- mean and max Cosmos-call interval in frames/actions;
- executor chunk length;
- wall-clock inference latency where available;
- final success and video evidence.

## What Dense Receding Means

Dense receding means the training/export code creates many causal prefix rows
from one full episode, for example:

```text
target motion starts near f094
training prefixes: f094, f102, f110, f118, ... through recovery/insertion/end
```

Each row remains a full `301`-frame / `300`-action episode record with a
different causal prefix mask. It is not a 128-action clip and not a short
video method sample.

Purpose:

- Match the states the real closed loop actually queries.
- Teach late/post-motion correction rather than only a few easy successful
  source-trajectory prefixes.
- Give the executor examples of "the peg is held but still outside the moved
  hole; what short action should reduce the error?"

Dense receding is not a runtime promise to call Cosmos at every dense prefix.
It is coverage for learning and for choosing robust low-frequency replanning
points.

## What Preflight Means

Preflight is a no-training validation pass over a proposed data/export root.

It must answer:

- Are all videos `301` frames and all action/state targets `300` steps?
- Are condition masks causal?
- Is the controller-facing role the actual physical mode, not a sampled
  curriculum label?
- Are future hole/peg/TCP states stored only as targets/diagnostics, not as
  privileged controller conditions?
- Do dense receding rows cover the live failed-query states?
- Are train/val splits and source paths valid?
- Are there enough late-rebind rows where the peg is grasped but not yet
  inserted?

Preflight does not train and does not claim method success. It prevents wasting
GPU time on a broken export.

## Training Plan

### Stage 1: Clean Dense Condition Root

Start from the frozen accepted v7_733 source/RGB/H5 data. Do not regenerate H5
data for this repair.

Export a new condition root with:

- `prefix_role_source=physical_mode`;
- sampled/curriculum role stored separately as provenance;
- dense receding prefixes around target motion, late correction, and insertion;
- late-rebind row weighting or repetition;
- full `301/300` records only;
- live-query coverage audit as a readiness gate.

### Stage 2: Low-Frequency Cosmos Task WM

Use Cosmos to predict future task state and visual/task evolution from causal
history:

- target/hole path;
- peg-head-in-hole-frame trajectory;
- TCP/peg relation;
- grasp/contact/insertion predicates;
- optional coarse action hints.

Do not treat generated state sidecars as handoff authority. They are model
outputs to guide/scaffold the executor and to diagnose failures. Real state
still decides `C_pi`.

### Stage 3: High-Frequency Executor

Train or distill a smaller execution policy that consumes:

- current live observation/state;
- DP action prior or DP feature/action proposal;
- Cosmos-predicted task path or task-frame target;
- current peg-hole/TCP/grasp predicates;
- recent action history.

It outputs short executable robot action chunks. It should preserve the static
DP's grasp/insertion competence while adding the ability to track a moved task
frame.

Possible executor forms:

- DP-prior residual policy;
- learned short-chunk policy conditioned on task-frame targets;
- MPC-style selector over DP/expert action candidates scored by WM task-state
  predictions.

The executor must not be a hand-coded threshold bridge. Thresholds may guard
safety and handoff, but not define the positive method.

### Stage 4: Conservative Handoff

Use real-state `C_pi` to decide whether frozen DP can resume. The handoff gate
must check:

- peg is grasped;
- target motion is slow or settled;
- peg-head is inside the empirical DP-continuability band;
- y/z alignment is tight enough for insertion;
- recent action did not damage contact.

If handoff fails, the executor continues or the WM replans. Do not relax the
gate to make a result pass.

## Theoretical Basis

### Time-Scale Separation

A large video/world model is useful for slower task-frame reasoning, not
necessarily for 30 Hz robot actuation. A smaller executor should handle local
contact-rich control.

### Receding-Horizon Control

Only a short executable chunk should be committed before reobserving. This
limits compounding errors and keeps the real world as the authority.

### Distribution Matching

The model must be trained on the states it will query at runtime. Dense
receding prefixes and live-query coverage repair the mismatch between sparse
successful source prefixes and real closed-loop recovery states.

### Policy Prior Preservation

The frozen DP already knows grasp-hold and insertion in static scenes. The new
controller should reuse that competence rather than replace it with raw
geometry thresholds or direct video-model actions.

### Conservative Continuability

Generated predictions are not enough for handoff. Handoff is valid only when
the real observed state is inside the DP success manifold.

## Evaluation Plan

Evaluate in this order:

1. Clean/dense preflight passes with no role/mode mismatch, full `301/300`
   length, and live-query coverage.
2. Two-sample sanity overfit passes for both generated video/task-state and
   executor action chunks.
3. Full SFT/executor training produces strict same-length generated artifacts.
4. Offline action/readout metrics improve specifically on late-rebind rows.
5. Closed-loop val panel compares against same-source pure DP.
6. Hard-screen panels evaluate only after val is not degraded.
7. Runtime report records Cosmos-call frequency and latency proxies.
8. Major success requires metrics plus direct video/contact-sheet inspection.

Method success requires:

- full `300/301` rollout;
- causal target-motion detection;
- no future privileged state conditions;
- nonzero WM use on moving-target cases;
- same-source pure-DP comparison;
- hard-case success fraction high enough to show broad usefulness, not one
  lucky rescue;
- final success from real simulator state plus inspected video.

## Stop Conditions

Stop and ask for direction if:

- clean/dense preflight cannot cover failed live-query states without breaking
  causality;
- two-sample executor/WM sanity cannot overfit clean data;
- direct raw Cosmos actions remain unstable after clean/dense repair;
- a learned executor also cannot preserve grasp/insertion on inspected videos;
- runtime requires Cosmos calls so frequent that the method is not deployable.

If direct raw Cosmos actions remain the blocker, switch to the executor-first
method instead of continuing to train the same raw-action head.
