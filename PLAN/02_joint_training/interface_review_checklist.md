# Interface Review Checklist

Date: 2026-07-09

Review this after the Slurm-side interface inspection writes:

- `project_interface_summary.json`
- `cosmos_interface_summary.json`
- `classification.txt`

The current inspection run is launched by:

```bash
scripts/slurm/launch_joint_dp_cosmos_interface_inspect_tmux.sh
```

and writes under:

```text
experiments/maniskill/runs/02_joint_training/interface_inspect/inspect01/
```

## Required Pass Conditions

Project/data interface:

- `status=ok` in `project_interface_summary.json`;
- `joint_overfit_abcd` guard passed inside the allocation;
- A static H5/JSON paths exist;
- B/C/D active indexes exist with expected train/val counts;
- sampled B/C/D rows have matching dataset class and `positive_dp_bc_allowed=false`;
- sampled trace files parse as JSON and expose action / motion / task row
  counts;
- sampled action rows have `pd_ee_delta_pose` 7D action vectors;
- DP checkpoint exists and loads with real `agent` / `ema_agent` state dicts;
- DP checkpoint args expose the expected action contract and horizons.

Cosmos interface:

- `status=ok` in `cosmos_interface_summary.json`;
- Cosmos Python is `.venv_cosmos313/bin/python`;
- `cosmos_framework` import spec is found;
- active Cosmos root exists;
- `config.yaml`, `normalization_stats.json`, and latest DCP checkpoint
  metadata exist;
- local tokenizer directory and WAN VAE path exist.

Slurm/runtime:

- manifest records `method_evidence_allowed=false`;
- manifest records `uses_toy_model=false`;
- job ran inside an `srun` compute step, not on the login node;
- no data generation, training, or method success claim is made by the
  interface inspection.

## Failure Handling

If project/data interface fails:

- inspect missing or malformed A/B/C/D index rows;
- do not write runnable trainer code until the sampled real rows expose a
  stable action / motion / task schema;
- do not use C rows as positive action targets to make the check pass.

If Cosmos interface fails:

- fix environment path, `PYTHONPATH`, tokenizer, VAE, or checkpoint layout;
- do not replace Cosmos-3 with a toy placeholder;
- rerun the same interface inspection in a tmux-held Slurm allocation.

If the job remains pending:

- keep monitoring the valid queued request;
- replace it only if a concrete test-only resource request preserves the same
  diagnostic and is demonstrably earlier;
- do not hold a second duplicate GPU request for the same inspection.

## Gate To Implementation

Only after both summaries pass should the next files become runnable:

- `scripts/world_model/build_joint_dp_cosmos_dataset.py`
- `scripts/world_model/inspect_joint_dp_cosmos_batch.py`
- `scripts/training/train_joint_dp_cosmos_overfit.py`

Until then, implementation work should stay at docs, guards, launchers, and
non-executed scaffolding.
