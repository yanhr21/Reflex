# TODO

Active TODO entry point after the 2026-06-09 Cosmos3 reset.

## Active Directory

- `TODO/cosmos3_300f_world_model/`

The previous `TODO/world_model_task_rebinding/` directory was moved to:

- `TODO/_backup_world_model_task_rebinding_20260609_before_cosmos3_300f_reset/`

## Proposed Method Branch

- `TODO/cosmos3_lowfreq_wm_executor/`

This branch records the new low-frequency Cosmos world-model plus
high-frequency executor direction. It is separate from the current direct
raw-Cosmos-action TODOs and should only be advanced after explicit user
approval for the new repair path.

## Historical Contact-Action Reset

- `TODO/contact_action_world_model/`

This is historical diagnostic context after the 2026-06-26 OpenPI full-episode
correction. It records why scorer-only and DP-handoff approaches were
insufficient. It is not the active OpenPI method TODO.

## Active OpenPI/pi0.5 Full-Episode Pivot

- `TODO/openpi_pi05_contact_action/`

This is the active TODO branch after the 2026-06-26 protocol correction.
OpenPI/pi0.5 must execute from episode step `0`; DP handoff and saved-snapshot
takeover are legacy diagnostics, not main method evidence. The previous
takeover-oriented TODO is archived under:

- `legacy/plan_todo_openpi_takeover_protocol_error_20260626/`

## Current Override

The 2026-06-09 stop gate was superseded by later 2026-06-12 user instructions:
the approved 733-row v7 source should proceed through Cosmos3 SFT, strict eval,
readout, and guarded closed-loop preflight. Do not resume v7 data construction
unless the user explicitly asks.

The old 128-action / 129-frame chunked SFT chain is rejected and archived.
Future work must use the approved full1000 data as a 300-step episode contract,
not as truncated chunk samples.

Live DP/controller rollout remains gated by the focused closed-loop TODO. A
checkpoint must pass strict same-length artifacts, generated-RGB readout/profile,
and direct visual review before any live simulator control is started.
