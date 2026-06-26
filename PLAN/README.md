# Active Plan

The active plan is now only:

- `PLAN/openpi_full_episode_policy/00_overview.md`

All previous plan content has been moved under `PLAN/legacy/`. Legacy files are
historical context only and must not drive execution.

## Current Protocol Boundary

- OpenPI/pi0.5 replaces Diffusion Policy as the action policy.
- OpenPI must run from episode step `0`; no DP prefix, no DP handoff, and no
  saved-snapshot takeover as method evidence.
- Static scene evaluation is full-episode OpenPI-only insertion.
- Dynamic scene evaluation starts with OpenPI and, after observed scene change,
  uses a causal world model to provide future scene/X-chat conditions for
  OpenPI; OpenPI still produces the robot actions.
- The official OpenPI architecture, transforms, and checkpoint loading path
  must be preserved. Do not replace OpenPI with custom MLP/VAE/diffusion
  executors.
- Use 733 accepted ManiSkill PegInsertionSide trajectories as the first real
  adaptation data source, preserving the 301 RGB/state frame and 300 action
  step contract.
