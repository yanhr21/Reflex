# Dynamic Data TODO

## Generator

- [x] Add `scripts/world_model/collect_dynamic_state_rollouts.py`.
- [x] Add `scripts/slurm/collect_dynamic_state_rollouts.sbatch`.
- [x] Add dense 4-GPU rollout wrapper for larger multi-scenario shards:
      `scripts/slurm/collect_dynamic_state_rollouts_4gpu.sbatch`.
- [x] Add rollout-dir downstream wrappers so larger shards can feed training,
      `C_pi` labeling, and RGB-D export without manual path reconstruction.
- [x] Keep RGB-D companion generation in the data plan; state-only traces are
      not the final dataset.
- [x] Support frozen DP rollout from `best_eval_success_at_end.pt`.
- [x] Support oracle slot extraction from simulator state.
- [x] Save resettable env states for every timestep.
- [x] Save actions, object slots, predicates, event metadata, and final metrics.

## Perturbations

- [x] Implement hole move-and-stop perturbation.
- [x] Implement continuously moving hole perturbation.
- [x] Implement hole reversal perturbation near pre-insertion.
- [x] Implement peg disturbance event.
- [x] Implement peg drop event.
- [ ] Implement regrasp-required episodes.
- [ ] Implement infeasible cases for unreachable or too-fast target motion.

## Validation

- [x] Render a small sample from saved env states.
- [x] Confirm RGB-D export can be generated from the same seeds and task
      families, even if large rendering is deferred to a dense multi-GPU Slurm
      job.
- [x] Verify movement starts before insertion completion in smoke rollouts.
- [x] Smoke-render all perturbation families and inspect contact sheets.
- [x] Validate every `traj_*` group inside multi-episode H5 files rather than
      checking only the first trajectory.
- [x] Quantitatively project hole/peg/TCP slots into RGB-D camera frames.
      `inspect_rgbd_dataset.py` reports per-camera in-image and positive-depth
      rates. This is projection sanity, not a substitute for visual inspection.
- [x] Verify grasp/drop labels from simulator state in smoke rollouts.
- [x] Verify success metrics use saved final `inserted` state in smoke rollouts.
- [x] Write smoke validation manifest with command, seed range, node, git state, and
      source checkpoint.
- [x] Write full dataset manifest after larger shards exist. Job `94510`
      produced six 16-episode H5 shards and validation artifacts under
      `experiments/world_model_task_rebinding/dynamic_state_rollouts/full4gpu/job94510`.

Completion standard: a small Slurm pilot produces valid H5/JSON traces for all
dynamic families with replayable states and correct event labels.

## Notes

- The first generator is intentionally single-env inside each job because it
  directly manipulates simulator state and writes full env-state traces. Scale
  should come from Slurm shards until the trace semantics are validated. The
  dense 4-GPU wrapper keeps this single-env trace semantics while avoiding
  sparse many-node one-GPU allocation.
- Rollout-dir downstream wrappers:
  `scripts/slurm/train_object_state_world_model_from_rollout_dir_4gpu.sbatch`,
  `scripts/slurm/label_continuability_from_rollout_dir_4gpu.sbatch`, and
  `scripts/slurm/render_dynamic_rgbd_dataset_from_rollout_dir_dense.sbatch`.
  Each discovers H5s on the compute node and runs validation before consuming
  the shard. The RGB-D wrapper defaults to one full 8-GPU node and keeps the
  dense-allocation guard for larger runs.
- Perturbation families are data-generation conditions, not online controller
  branches. The future controller should consume object slots and uncertainty,
  not scenario labels.
- The peg disturbance/drop events are implemented as direct simulator
  perturbations and still require replay/video validation before they count as
  reliable experimental evidence.
- Smoke `94241` rendered synchronized RGB-D from the dynamic `hole_move_stop`
  env-state trace and wrote frame-aligned env states into the RGB-D H5.
- Smoke report:
  `docs/world_model_task_rebinding/2026-06-02_dynamic_data_smoke.md`.
- Validation summary:
  `experiments/world_model_task_rebinding/smoke_dynamic_state/validation_summary.md`.
- Visual overview:
  `experiments/world_model_task_rebinding/smoke_rgbd_dynamic/all_contact_sheets_overview.png`.
- `hole_reverse_seed13` moved the hole by about `1.044m`, so it is currently
  evidence for reachability/infeasibility handling rather than a clean
  reachable rebinding case.
- Updated default continuous/reverse hole velocity to `0.001m/step` and reran
  bounded pilots:
  `hole_constant_seed22` moved about `0.216m` and failed;
  `hole_reverse_seed23` moved about `0.162m` and succeeded. The latter should
  be treated as a `C_pi` boundary sample, not as dynamic generalization.
- Full 4-GPU shard `94510` completed on `server60` with 96 trajectories and
  zero validation warnings. Summary note:
  `docs/world_model_task_rebinding/2026-06-02_dynamic_shard_94510.md`.
