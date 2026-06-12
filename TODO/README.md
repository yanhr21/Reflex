# TODO

Active TODO entry point after the 2026-06-09 Cosmos3 reset.

## Active Directory

- `TODO/cosmos3_300f_world_model/`

The previous `TODO/world_model_task_rebinding/` directory was moved to:

- `TODO/_backup_world_model_task_rebinding_20260609_before_cosmos3_300f_reset/`

## Stop Gate

Do not start a new Slurm training, generation, readout, controller, or DP
integration job until the user reviews the new plan/TODO.

The old 128-action / 129-frame chunked SFT chain is rejected and archived.
Future work must use the approved full1000 data as a 300-step episode contract,
not as truncated chunk samples.
