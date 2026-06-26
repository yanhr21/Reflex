# OpenPI Full-Episode Active Status - 2026-06-26

## Boundary

The active OpenPI/pi0.5 protocol is full-episode OpenPI execution, not
saved-snapshot takeover.

Previous OpenPI takeover replay outputs and object17/near-contact diagnostics
were moved to:

`legacy/experiments_openpi_takeover_protocol_error_20260626/`

The previous docs/dashboard were moved to:

`legacy/docs_openpi_takeover_protocol_error_20260626/`

## Current Evidence

There is currently no accepted evidence that OpenPI has completed:

- static PegInsertionSide full-episode insertion from step `0`; or
- dynamic PegInsertionSide full-episode insertion from step `0`; or
- dynamic OpenPI + world-model full-episode insertion.

Existing official OpenPI conversion, training, checkpoint preservation, and
inference tooling remain useful infrastructure. Existing takeover replay
metrics are diagnostic only and must not be cited as method success.

## Active Assets To Keep

- `lerobot_home/`
- `openpi_data_home/`
- `checkpoints_local_preserved/`
- full-episode `pi05_peg733_*` conversion/audit/training roots that are not
  contact-suffix takeover outputs
- accepted 733 source trajectories under the Cosmos experiment tree

## Next Required Gate

Run OpenPI-only static full-episode evaluation:

1. reset static ManiSkill PegInsertionSide;
2. execute OpenPI actions from step `0` through the full horizon;
3. save final metrics, video/contact sheets, and action traces;
4. do not use DP execution or saved-snapshot takeover.

If this fails, diagnose OpenPI action-space, observation, normalization,
prompt, checkpoint, and dataset alignment before dynamic WM integration.
