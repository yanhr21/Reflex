# 2026-06-26 OpenPI Full-Episode Protocol Correction

## Correction

The main OpenPI method must start at episode step `0`. Saved dynamic takeover
snapshot replay is no longer an active protocol for method evidence.

## Correct Evaluation

Static:

- OpenPI executes the entire static episode.
- No DP handoff and no takeover snapshot.
- Success requires final simulator success plus video/contact review.

Dynamic:

- OpenPI executes from the start of the episode.
- After observed target/object motion, the world model may provide causal
  future scene/task-state or future `x_t` conditioning.
- OpenPI completes the task under that conditioning.
- DP is a baseline only.

## Archived Material

Old OpenPI takeover docs and dashboards were moved to:

`legacy/docs_openpi_takeover_protocol_error_20260626/openpi_pi05_contact_action_takeover_docs`

Old takeover experiment outputs were moved to:

`legacy/experiments_openpi_takeover_protocol_error_20260626/`

These artifacts are negative/diagnostic history only.
