# Phase 04: Oracle Boundary

Goal: test whether the corrected Oracle pipeline can succeed after DP and RGB
Cosmos have both run and after the controller-facing bridge has produced a
task-state / candidate-score / trust-gate output.

Boundary:

1. Start from reset.
2. DP executes the initial policy segment.
3. Target/hole motion is detected causally.
4. RGB Cosmos-3 imagines future video or task-frame state.
5. Phase 03 bridge extracts task state, scores candidates, and emits a trust
   gate.
6. Only then may Oracle final seating be applied.

Required evidence:

- before-oracle `peg_head_at_hole`;
- after-oracle `peg_head_at_hole`;
- oracle jump distance;
- trigger frame;
- Cosmos RGB input/output path;
- extracted task-state chart and overlays;
- candidate list and selected candidate score;
- trust gate output;
- video with Oracle moment annotated;
- `method_evidence_allowed=false`.

Invalid result:

- any report that hides the snap;
- any run where Oracle is applied before Cosmos;
- any run where Oracle is applied before task-state extraction and candidate
  scoring;
- any run where the final metric is described as physical controller success;
- any run that uses future labels as if they were deployed observations.

This phase is an upper-bound diagnostic only. Its purpose is to test whether
the bridge identifies a meaningful finish condition, not to claim live physical
insertion.
