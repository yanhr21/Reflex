# Controller-Facing World Model Research

Date: 2026-07-02

Question:

> Cosmos-3 can generate RGB futures, but the output is not yet a stable
> controller-facing task-frame / near-hole / insertion-axis / gate signal. What
> should we do?

This note reviews the most relevant recent literature available by
2026-07-02 and converts it into an execution plan for the ManiSkill
`PegInsertionSide-v1` route.

## Bottom Line

Do not make the next step "better Cosmos videos." The literature points to a
different target:

```text
RGB future / latent future
  -> task-state extraction
  -> candidate action or local primitive proposal
  -> world-model / geometry / executability scoring
  -> trust gate
  -> DP-compatible live execution
```

For our repo, Cosmos-3 should become a producer of explicit control variables:

- target / hole displacement;
- task-frame and insertion-axis estimate;
- near-hole / preinsert / continuability gate;
- short-horizon action-chart candidates;
- trust score comparing imagined future against live observation;
- executable-action score that rejects discontinuous or impossible finishes.

The immediate plan should be GPC-style and Feedback-WM-style, not a full new
WAM training project. We already have a frozen DP checkpoint and a Cosmos-3
checkpoint, so the lowest-risk bridge is:

1. sample or construct several DP-compatible candidate chunks / local finishers;
2. use Cosmos RGB futures plus geometric extraction to score task progress and
   executability;
3. execute only short chunks;
4. reobserve and update the gate after every chunk.

## Literature Findings

### 1. GPC: Frozen Policy + Action-Conditioned World Model

Generative Predictive Control (GPC) is directly relevant because it augments a
frozen diffusion policy at inference time. The policy proposes multiple action
chunks, an action-conditioned world model predicts their consequences, and the
system either ranks candidates or optimizes an action proposal through the world
model.

Key lesson for us:

- keep DP frozen;
- do not ask Cosmos to output the final action by itself;
- ask the active policy / local primitive to propose candidate chunks;
- use the world model as a lookahead evaluator;
- rank by predicted task progress and executability.

This maps naturally to ManiSkill:

```text
candidate chunks:
  DP continuation
  DP with small task-frame residuals
  bounded insertion-axis push
  bounded retreat-reapproach
  hold/reobserve

scores:
  predicted peg-head-to-hole distance
  predicted insertion-axis alignment
  predicted contact/preinsert state
  action magnitude smoothness
  gripper closed and no discontinuity
```

### 2. Feedback World Model: Close the Prediction-Observation Loop

Feedback-WM observes that open-loop world-model predictions drift under
distribution shift. Its core idea is to maintain a lightweight feedback state:
after each action, compare predicted next state with observed next state and use
that mismatch to correct future predictions. It also introduces action-aware
guidance to emphasize controllable components and suppress irrelevant visual
change.

Key lesson for us:

- after each executed chunk, compare Cosmos-predicted target / hole / peg state
  with the newly observed RGB-derived state;
- use this discrepancy as a trust score;
- shorten or stop the rollout when prediction-observation mismatch grows;
- score only action-controllable elements: peg, hole, gripper, contact region,
  not background or renderer artifacts.

This is the right answer to "when do we trust Cosmos?" Trust should be measured
online, not assumed from a nice-looking video.

### 3. When to Trust Imagination: Adaptive Chunk Length

The "When to Trust Imagination" paper frames WAM execution as
future-reality verification. A verifier checks whether predicted future actions
and predicted visual dynamics remain consistent with actual observations, and
then adapts how many actions to execute before replanning.

Key lesson for us:

- do not execute long chunks near insertion;
- use longer chunks only when prediction-observation consistency is high;
- force short chunks or immediate reobserve during contact, near-hole, or high
  uncertainty;
- make the chunk length an output of the trust gate.

For peg insertion, the policy should become more conservative near contact:

```text
far from hole: chunk 4-8
near-hole/preinsert: chunk 1-2
contact/insertion: chunk 1 and reobserve every step
```

### 4. WoG: Predict in the Action Condition Space

World Guidance (WoG) argues that future observations are too redundant, while
over-compressed latents may lose fine control. Their solution is to discover a
condition space that is explicitly useful for action generation: future
observations are first injected into the action pipeline, compressed into
conditions, and then the policy learns to predict those conditions.

Key lesson for us:

- the useful Cosmos output is not the full RGB frame;
- it should be compressed into a condition that directly changes actions;
- for peg insertion, the condition space should be small and explicit.

Recommended condition vector for the current route:

```text
c_t = {
  hole_center_cam_or_world,
  hole_axis,
  peg_head,
  peg_axis,
  peg_to_hole_vector,
  axial_error,
  lateral_error,
  angular_error,
  near_hole_probability,
  preinsert_probability,
  insertion_axis_push_allowed,
  trust_score
}
```

If a learned head is added later, train it to predict this vector from current
RGB plus Cosmos future RGB. For the next smoke demo, a deterministic extractor
from simulator-rendered RGB / known camera geometry may be acceptable only as a
diagnostic, with no hidden future state in method claims.

### 5. LaWAM / VPP / Fast-WAM: Latents Beat Pixels for Control

LaWAM predicts compact latent visual subgoals instead of reconstructed future
videos, then conditions the action generator on those subgoals. VPP similarly
uses predictive visual representations from a video prediction model. Fast-WAM
finds that much of WAM's benefit may come from video modeling during training,
not necessarily explicit future generation at every test step.

Key lesson for us:

- future RGB is evidence and visualization, but the controller should consume a
  compact latent / structured state;
- if test-time Cosmos is too slow or unstable, use it for short-horizon
  checkpoints and condition extraction, not continuous full-episode video;
- for the current frozen DP setting, the controller-facing output should be a
  structured condition / gate, not a generated video stream.

### 6. τ0-WM: Proposal, Evaluation, Revision

τ0-WM combines a Video Action Model that proposes executable actions and an
Action-Conditioned Video Simulator that predicts futures and dense progress
scores for candidate action chunks. At inference, it samples candidates, ranks
them, and rectifies low-quality candidates.

Key lesson for us:

- split the problem into proposal and evaluation;
- do not rely on one action proposal;
- score every candidate with task-progress and safety metrics;
- let the world model reject bad DP continuations or bad insertion pushes.

We do not have a trained τ0-WM, but the interface is useful:

```text
proposal: DP / residual / primitive candidates
evaluation: Cosmos future + extractor + progress score
revision: choose next chunk or switch to reobserve
```

### 7. EVA / VERA: The Executability Gap

EVA names the failure mode that matters here: video rollouts can be visually
coherent but not executable. It trains an inverse dynamics model and uses it as
a reward to penalize videos that imply impossible, jerky, out-of-bounds, or
kinematically inconsistent actions. VERA takes a decoupled route: a video
planner produces visual lookahead, and a Jacobian inverse-dynamics bridge turns
visual motion into robot action.

Key lesson for us:

- visually plausible Cosmos futures are not enough;
- every predicted future must be checked against feasible `pd_ee_delta_pose`
  action magnitudes and the successful DP insertion action distribution;
- reject any future or candidate whose implied motion requires a teleport, a
  discontinuous peg pose jump, gripper opening by accident, or action magnitude
  outside the DP / controller bounds.

This directly prevents the previous snap error from reappearing.

### 8. MinD: Slow Imagination, Fast Control

MinD uses a low-frequency video generator and a high-frequency diffusion policy,
with an alignment module connecting video prediction features to actions. It
also uses predicted futures as risk signals.

Key lesson for us:

- run Cosmos sparsely, not every simulator step;
- use DP or a local controller at high frequency;
- use Cosmos to update gates, target-frame estimates, and risk only when the
  scene changes or the controller is uncertain.

## Recommended Design For This Repo

Use a five-module bridge:

### Module A: RGB Task-State Extractor

Input:

- live RGB prefix;
- Cosmos future RGB;
- camera metadata / known rendering contract when available.

Output:

- peg head estimate;
- hole center estimate;
- hole axis;
- peg axis;
- peg-to-hole vector;
- near-hole and preinsert flags;
- confidence.

Rule:

- method claims may not use future simulator state;
- diagnostic charts may compare against simulator state after the fact;
- every extractor output must be saved as an overlay on live/Cosmos frames.

### Module B: Candidate Generator

Generate several DP-compatible candidates:

- frozen DP continuation;
- DP action chunk plus small lateral residual toward predicted task frame;
- bounded insertion-axis push with gripper closed;
- bounded retreat then reapproach;
- hold/reobserve.

All candidates must be real `pd_ee_delta_pose` vectors. No pose setting.

### Module C: World-Model / Geometry Scorer

For each candidate, produce a score:

```text
score =
  task_progress
  - lateral_error
  - angular_error
  - executability_penalty
  - trust_penalty
  - discontinuity_penalty
```

The first implementation can be ranking-only. Do not implement gradient
optimization until ranking is understood.

### Module D: Trust Gate

Inputs:

- prediction-observation discrepancy;
- extractor confidence;
- target/hole motion state;
- near-hole/preinsert/contact state;
- action magnitude / smoothness;
- whether Cosmos and live observations agree.

Outputs:

- `trust_cosmos`: true/false;
- `execute_chunk_len`: 1, 2, 4, or 8;
- `handoff_mode`: DP, residual-DP, primitive, reobserve, or fail-safe stop;
- `method_evidence_allowed`: true only for live reset-to-end physical rollouts.

### Module E: Physical Finisher

The physical finisher may be:

- frozen DP if near-hole continuability is high;
- bounded axis-aligned or task-frame insertion primitive;
- retreat-reapproach primitive if contact is jammed.

Rules:

- gripper stays closed by default;
- every action vector is logged;
- action magnitudes must be compared to successful DP insertion magnitudes;
- raw world-axis push is diagnostic unless justified by live task-frame axis;
- no `set_pose`, no saved state, no source suffix.

## Concrete Experiment Sequence

### Step 1: Static DP Action Distribution

From successful static DP rollouts, extract:

- insertion frame range;
- action magnitudes near insertion;
- gripper channel values;
- end-effector direction in task frame;
- peg-to-hole errors before successful seating.

Purpose:

- define safe primitive bounds;
- define what a plausible insertion action looks like.

### Step 2: Cosmos-to-State Chart

For forward, backward, continuous, and static target/hole motion cases:

- run RGB Cosmos;
- overlay extracted peg/hole/task-frame variables on predicted frames;
- create a chart of lateral error, axial error, angular error, confidence, and
  trust score over frames.

Purpose:

- prove Cosmos output can be converted into a stable control condition.

### Step 3: Offline Candidate Scoring

For recorded live prefixes:

- generate candidate chunks without executing them;
- score each candidate using Cosmos / geometry;
- compare chosen candidate against actual next success/failure labels only as
  offline diagnostic.

Purpose:

- debug the scoring function before risking live claims.

### Step 4: Oracle Diagnostic With Correct Boundary

Start from reset:

1. DP prefix;
2. causal motion detection;
3. Cosmos RGB imagination;
4. task-state extraction and candidate scoring;
5. only then an Oracle final-seat diagnostic.

Required output:

- evidence that the gate picked a physically meaningful finish condition;
- explicit Oracle moment;
- jump distance;
- `method_evidence_allowed=false`.

Purpose:

- test if the bridge identifies the right finish state without pretending the
  Oracle action is physical success.

### Step 5: Live Ranking Controller

Replace Oracle with candidate execution:

- execute one selected chunk;
- reobserve;
- update trust gate;
- repeat.

Success requires reset-to-end physical insertion with no state intervention.

## What Not To Do

- Do not spend effort only improving RGB video aesthetics.
- Do not train a toy world model and call it Cosmos / WAM progress.
- Do not use future simulator state for controller-facing decisions.
- Do not claim success from metric flips after state edits.
- Do not execute long open-loop chunks near contact.
- Do not run old scripts that still contain `world_model_task_rebinding`,
  `dp_peg1000`, `set_pose`, saved-state replay, or geometric final-seat paths
  as active method scripts.

## Sources

- GPC, "Inference-Time Enhancement of Generative Robot Policies via Predictive
  World Modeling": https://arxiv.org/html/2502.00622v4
- DDP, "Dreaming the Unseen": https://arxiv.org/abs/2603.21017
- Feedback-WM: https://arxiv.org/abs/2605.15705
- When to Trust Imagination: https://arxiv.org/abs/2605.06222
- World Guidance / WoG: https://arxiv.org/abs/2602.22010
- Fast-WAM: https://arxiv.org/abs/2603.16666
- VPP: https://video-prediction-policy.github.io/
- LaWAM: https://arxiv.org/html/2606.15768
- τ0-WM: https://arxiv.org/html/2606.01027v1
- EVA: https://arxiv.org/abs/2603.17808
- VERA: https://arxiv.org/abs/2605.27817
- MinD: https://arxiv.org/abs/2506.18897
- "From World Models to World Action Models: A Concise Tutorial for Robotics":
  https://arxiv.org/html/2607.00836v1
