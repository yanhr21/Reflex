# 2026-07-10 Generation Script Cleanup

This archive contains legacy generation / diagnostic entry points removed from
the active script directories.

Archived groups:

- Phase03 Oracle / bridge / forward-backward launchers and helpers.
- Old fix3 / 733 render and merge utilities.
- Old direct replay / Cosmos render sbatch entry points that are not used by
  the current B/C/D dataset generation route.

Current active dataset generation remains under `scripts/slurm/` and
`scripts/world_model/` around the `01_dataset` B/C/D route, especially
`launch_dataset_bcd_expansion_shards_tmux.sh`,
`dataset_bcd_expansion_shard_plan.sh`, the B/C/D production launchers, the
B/C/D collectors, and `build_dataset_bcd_prod02_review_index.sh`.

Do not launch scripts from this archive as active method or active dataset
generation without first re-auditing them and moving them back deliberately.
