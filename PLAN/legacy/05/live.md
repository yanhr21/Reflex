# Phase 05: Live Controller

Goal: replace Oracle final seating with a real live controller.

Prerequisites:

- Phase 01 DP static check is understood.
- Phase 02 Cosmos imagination is understood.
- Phase 03 Oracle boundary has been validated or rejected.
- Phase 04 integration failure point is known.

Steps:

1. Start from reset.
2. DP executes initial policy.
3. Detect target/hole motion causally.
4. Run RGB Cosmos-3 from live RGB history.
5. Extract controller-facing task state:
   peg/hole displacement, insertion axis, near-hole/preinsert/contact gates,
   confidence, and trust score.
6. Generate DP-compatible candidates:
   DP continuation, residual-DP, bounded insertion-axis push,
   retreat-reapproach, and hold/reobserve.
7. Score candidates by task progress, executability, and trust.
8. Execute only the selected `pd_ee_delta_pose` chunk.
9. Reobserve and repeat.
10. Reduce chunk length near contact:
    far from hole can use up to DP horizon chunks; near-hole uses 1-2 steps;
    contact/insertion uses 1 step and reobserve.
11. Measure final success from live simulator state.
12. Save rendered visual review and logs.

Success condition:

- no `set_pose`;
- no source-state restore;
- no saved-state replay;
- no future labels;
- reset-to-end live success with rendered video.

Required logged evidence:

- live RGB and Cosmos RGB;
- task-state overlays;
- candidate list and scores;
- selected and rejected action chunks;
- trust-gate output and chunk length;
- action magnitudes compared against successful DP insertion actions;
- final simulator success state.
