# Active Dataset Generation Scripts

Current status on 2026-07-10: generation is paused by user request. Do not
launch new production or expansion jobs until the user explicitly asks.

Use these active entry points for the current ManiSkill dataset route:

- B/C/D expansion plan:
  `scripts/world_model/dataset_bcd_expansion_shard_plan.sh`
- B/C/D expansion launcher:
  `scripts/slurm/launch_dataset_bcd_expansion_shards_tmux.sh`
- B/C/D combined review index:
  `scripts/world_model/build_dataset_bcd_prod02_review_index.sh`

B/C/D launcher dependencies:

- Shared production launcher:
  `scripts/slurm/launch_dataset_stage_production_tmux_common.sh`
- B production wrapper:
  `scripts/slurm/launch_dataset_dynamic_rgb_production_tmux.sh`
- C production wrapper:
  `scripts/slurm/launch_dataset_frozen_dp_dynamic_production_tmux.sh`
- D production wrapper:
  `scripts/slurm/launch_dataset_future_frame_teacher_production_tmux.sh`
- B in-allocation runner:
  `scripts/slurm/run_dataset_dynamic_rgb_smoke_in_allocation.sh`
- C in-allocation runner:
  `scripts/slurm/run_dataset_frozen_dp_dynamic_smoke_in_allocation.sh`
- D in-allocation runner:
  `scripts/slurm/run_dataset_future_frame_teacher_smoke_in_allocation.sh`

Current collectors and guards:

- Dynamic adapter:
  `scripts/world_model/active_dynamic_peg_adapter.py`
- B/D demo-action collector:
  `scripts/world_model/collect_dynamic_demo_action_smoke.py`
- C frozen-DP collector:
  `scripts/world_model/collect_frozen_dp_dynamic_failure_smoke.py`
- Render canary:
  `scripts/world_model/render_min_canary.py`
- Smoke approval gate:
  `scripts/world_model/require_dataset_class_smoke_approved.sh`
- Source audits:
  `scripts/world_model/audit_dataset_runner_source.sh`
  and `scripts/world_model/audit_dataset_collector_source.sh`
- Production validation:
  `scripts/world_model/validate_dataset_production_run.sh`

Static A RGB render scripts remain active for the static dataset route:

- `scripts/slurm/launch_dataset_static_rgb_smoke_tmux.sh`
- `scripts/slurm/launch_dataset_static_rgb_full_tmux.sh`
- `scripts/slurm/launch_dataset_static_rgb_next_shard_tmux.sh`
- `scripts/slurm/run_dataset_static_rgb_smoke_in_allocation.sh`
- `scripts/world_model/make_static_replay_shard.py`
- `scripts/world_model/require_dataset_static_full_ready.sh`

Legacy Oracle / phase03 / fix3 / old replay-render entry points were archived
under `scripts/legacy/20260710_generation_cleanup/` and must not be used for
active data generation without a new source audit.
