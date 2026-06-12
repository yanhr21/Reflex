# Archive Boundary

## Why The Workspace Was Cleaned

The previous workspace contained several method lines that are now explicitly
invalid for the active Cosmos3 plan:

- self-trained object-state world models,
- state/oracle controller branches,
- old RGB-D slot/controller chains,
- old single-image/short-video Cosmos3 diagnostics,
- old 93-frame exports,
- old full301 SFT results with untrusted contracts,
- rejected 129-frame / 128-action chunked WAM SFT roots.

Keeping these roots under active `experiments/`, `docs/`, `PLAN/`, or `TODO/`
made it too easy for future work to resume a rejected method. They were moved
out of the current `/public/home/yanhongru/ICLR2027` tree.

## External Archive

Archived content is under:

```text
/public/home/yanhongru/ICLR2027_archive/reflex_20260609_cosmos3_300f_reset/
```

This archive is for historical inspection only. It is not an active input root
for future training or evaluation.

## Kept In Active Workspace

The active workspace keeps only assets needed for the next reviewed Cosmos3
run:

- local Cosmos3 checkpoints,
- base DP checkpoints,
- official replay data,
- approved full1000 RGB videos,
- full1000 source H5/specs,
- code and external dependencies.

## Current Stop Point

No new experiment has been started after this cleanup. The next action is user
review of `PLAN/cosmos3_300f_world_model/` and
`TODO/cosmos3_300f_world_model/`.
