# Phase 03: DP Plus Cosmos Integration

Goal: convert RGB Cosmos output into controller-facing signals and determine
which bridge component fails before introducing an Oracle final seat.

Inputs:

- Phase 01 DP static result.
- Phase 02 Cosmos video/action-chart result.
- Literature guidance:
  `docs/controller_facing_world_model_research_20260702.md`.

Required modules:

1. RGB task-state extractor:
   - live RGB prefix and Cosmos future RGB in;
   - peg head, hole center, peg axis, hole axis, peg-to-hole vector, lateral
     error, axial error, angular error, near-hole flag, preinsert flag,
     confidence out;
   - save overlays on live and Cosmos frames.
2. Candidate generator:
   - frozen DP continuation;
   - DP action plus small task-frame residual;
   - bounded insertion-axis push with gripper closed;
   - bounded retreat-reapproach;
   - hold/reobserve.
3. Scorer:
   - rank candidates by predicted task progress, lateral/angular error,
     executability, trust, and discontinuity penalty;
   - first version is ranking-only, not gradient optimization.
4. Trust gate:
   - compare predicted task state with reobserved task state;
   - output `trust_cosmos`, chunk length, handoff mode, and evidence label.
5. Physical execution interface:
   - only real `pd_ee_delta_pose` actions;
   - no `set_pose`, no state restore, no future simulator labels.

Steps:

1. Start from reset in ManiSkill inside a tmux-held Slurm allocation.
2. Run DP initially and log actions.
3. Detect target/hole motion causally from live observations.
4. Run RGB Cosmos-3 from live RGB history after motion detection.
5. Run the task-state extractor on live RGB and Cosmos future RGB.
6. Build the frame-aligned chart:
   - target/hole displacement;
   - insertion axis;
   - peg-to-hole vector;
   - lateral, axial, and angular error;
   - near-hole/preinsert/contact gates;
   - extractor confidence and trust score.
7. Generate DP-compatible candidate chunks without applying Oracle.
8. Score and rank candidates offline first.
9. Execute only an allowed DP-compatible chunk if the trust gate permits it.
10. Reobserve and update the prediction-observation discrepancy.
11. Save synchronized visualization:
    live RGB, Cosmos future video, extracted task state overlays, candidate
    chart, chosen action, rejected actions, trigger frame, trust score, and
    final simulator state.
12. Identify the first failing component.

Failure classes:

- DP cannot preserve static skill.
- Cosmos future video is wrong.
- Cosmos output is visually plausible but task-state extraction is wrong.
- Task-state extraction is correct but candidate generation is missing the
  required physical finish.
- Candidate scoring chooses a bad action.
- Trust gate executes too long, too early, or with low confidence.
- DP-compatible execution moves the peg out of the basin.
- Final contact/insertion fails after near-hole alignment.
- Candidate requires impossible action magnitude or discontinuous state change.

Pass condition:

- the integrated run provides a clear, visual, frame-aligned diagnosis of the
  first failing step;
- every Cosmos output used by the controller is converted into an explicit
  saved signal;
- every executed action is a logged DP-compatible action.

Do not use Oracle in this phase.
