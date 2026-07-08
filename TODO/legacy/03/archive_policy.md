# Phase 03 Archive Policy

Use this after reviewing a Phase 03 full-pipeline run. Active experiment
directories must not keep misleading or invalid evidence.

Keep under `experiments/maniskill/runs/03_oracle/` only if:

- the run is still under active review; or
- it is valid partial evidence needed for the next active step; or
- it is a reviewed single-case success candidate that still needs broader
  coverage.

Move under `/public/home/yanhongru/ICLR2027/archive/Reflex/` if any of the
following are true:

- peg / wooden stick teleports or snaps;
- target / hole teleports;
- object disappears;
- wall penetration or insertion into wall occurs;
- controller-facing path uses `set_pose`, `set_state`, `set_state_dict`,
  source-state restore, saved-state replay, geometric final placement, future
  labels, or hand-selected suffixes;
- dynamic target-motion stage is controlled by Diffusion Policy instead of
  Cosmos-derived actions;
- run claims physical success from simulator metric without visual review;
- artifact audit fails for missing required evidence and the failure is no
  longer being debugged;
- video stops at trigger / boundary and is later superseded by a complete
  full-pipeline attempt.

Archive destination rule:

- Preserve the original repository-relative structure under
  `/public/home/yanhongru/ICLR2027/archive/Reflex/`.
- Archive matching logs from `logs/03_oracle/` alongside run artifacts.
- Do not create `/public/home/yanhongru/ICLR2027/Reflex/archive/`.

Every archived run should have a classification explaining why it is invalid,
failed, superseded, or partial-only.
