# Compositional Rebinding Distribution Plan

## Current Priority Override

Latest user override, 2026-06-16: formal training may use the first valid
1/2/4-GPU tmux-held allocation, but it must still run/reserve for at least
3 hours. Do not block the current candidate-executor chain only because an
exact 2-GPU allocation is pending.

Prior user override, 2026-06-14: before implementing more compositional
teacher work, first run full Cosmos3 SFT from the already generated dense 733
condition data, then run closed-loop eval. This plan remains the next method
repair direction after that dense full-SFT/eval result is measured.

## Why This Replaces Failed-State Recovery

The failed-state recovery supplement path is not the method. It is diagnostic
evidence only.

The method in `IDEA.md` is not "detect an error, then learn a rescue case."
That would create an endless list of cases: missed hole, late hole, fast hole,
bad peg axis, weak grasp, wall contact, and then all combinations of those.
That is not task rebinding.

The correct target is one closed-loop behavior:

1. observe the current task state;
2. predict the future task frame;
3. move in task-frame coordinates;
4. keep checking real observations;
5. hand back to the frozen DP only when the relative state is continuable.

Different dynamic scenes are different values of the same variables, not
different recovery modes.

## Correct Data Principle

Do not build data by mining closed-loop failures and teaching one rescue per
failure. Build data by sampling a broad dynamic task distribution from the
start and using an online expert that always acts in the current task frame.

The 733 accepted rows remain useful as the frozen base source and visual/static
skill reference. They are not enough by themselves, because successful-source
trajectories do not cover the full compositional dynamic state distribution.

New rows should be generated from task factors, not from failure labels:

- hole motion: start time, velocity, acceleration, stop/reverse/continuous
  behavior, amplitude, direction;
- peg state: still grasped, small pose drift, contact offset, partial
  misalignment;
- robot state: phase along the static skill, TCP/peg relation, gripper/contact
  state;
- task relation: peg-head position and peg axis in the live hole frame;
- feasibility: reachable workspace, visible objects, insertion bandwidth.

These factors can combine. The teacher must not branch on "case name" to
choose a special rescue. It should run the same task-frame rebinding loop.

## Teacher Semantics

The teacher is an online task-frame expert, not a recovery script.

At each control step or short chunk:

1. read current peg, hole, TCP, qpos, grasp/contact;
2. predict or use the current/future hole task frame;
3. compute a desired peg-head and peg-axis trajectory in the hole frame;
4. use IK/servo/MPC to reduce the task-frame error while preserving grasp;
5. run the frozen DP only when `C_pi` says the real relative state is
   continuable;
6. if the target changes, simply update the task frame and continue.

If the peg is not grasped or the task is physically infeasible, the row should
be labeled infeasible or excluded from positive insertion data. Do not force it
into a "rescue" class.

## Model/Controller Split

Cosmos should not be trained as a direct raw-action oracle for every failure.

Cosmos should predict:

- future hole/task-frame path;
- peg-head and peg-axis relation in the task frame;
- grasp/contact/insertion risk;
- continuability/readiness signals;
- optional coarse action hints.

The high-frequency executor should consume:

- live observation/state history;
- frozen DP action prior;
- Cosmos task-frame prediction;
- current peg/hole/TCP relation;
- contact/grasp predicates.

It outputs short executable action chunks. Runtime uses real observations as
authority and calls Cosmos only when needed, not every dense training prefix.

## Data Flow

1. Keep the 733 source as the base visual/static skill reference.
2. Stop using failed closed-loop restore states as positive-data targets.
   Preserve them as diagnostics showing what the current system cannot handle.
3. Build a new compositional dynamic teacher generator:
   - sample dynamic task factors before or during the episode;
   - run the same online task-frame expert for all samples;
   - record successful physical insertions only when video/replay and final
     simulator state agree;
   - record infeasible cases separately, not as positive rows.
4. Render a small visual review first. The final frames must show real
   insertion, not metric-only success.
5. Export dense causal prefixes from these successful dynamic expert rows.
   Dense prefixes are training coverage, not a runtime call schedule.
6. Run strict preflight:
   - `301` RGB/state frames;
   - `300` action steps;
   - causal masks;
   - no future privileged object state in controller conditions;
   - task-factor provenance;
   - live-query coverage as a diagnostic, not as a reason to invent rescue
     classes.
7. Run short overfit on a tiny mixed subset:
   - 1-2 GPUs;
   - about 50-100 steps;
   - visual review of generated reference/prediction sheets.
8. Only if visual review, strict preflight, and short overfit pass, start full
   training under the current minimum resource rule: first valid 1/2/4-GPU
   allocation, at least 3 hours.

## What Would Prove The Direction

- A small generated set contains visually real insertions under combined
  dynamic factors, not only single named cases.
- The same teacher loop handles moving, stopping, reversing, continuous motion,
  and mild peg/contact offsets without switching to per-case scripts.
- Dense export covers late task-frame states because the distribution itself
  contains them, not because we mined failed artifacts.
- A held-out combination test succeeds without adding a new case-specific
  branch.

## What Would Falsify It

- The online task-frame expert cannot produce valid rows except by
  case-specific rescue logic.
- Visual review shows grasp loss, wall penetration, or metric-only insertion.
- Full rows need privileged future state as controller input.
- Held-out combinations require new hand-written branches.

## Immediate Next Work

1. Mark failed-state recovery supplement as diagnostic only.
2. Implement or refactor toward a compositional dynamic teacher generator.
3. Run only a tiny compute-node smoke first, with several combined dynamic
   factors and no full training.
4. If smoke rows exist, render review sheets and inspect them.
5. If visual review passes, export dense prefixes and rerun the overfit gate.
