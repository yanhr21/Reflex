# Legacy Dynamic Source Recovery

Date: 2026-07-07

This note records the current source-recovery status for old dynamic data
generators. It is for audit and adaptation planning only. It does not make the
old route active.

## Git Source

The following files were deleted in commit
`852976723d813352cabd5690f0acaab910f86c4e` on 2026-07-03:

- `scripts/world_model/generate_cosmos3_fix3_hard_dynamic_teacher.py`
- `scripts/world_model/generate_cosmos3_fix3_late_trigger_dynamic_experts.py`
- `scripts/world_model/generate_cosmos3_fix3_successful_dynamic_dataset.py`
- `scripts/world_model/render_cosmos3_maniskill_sft_dataset.py`

Their last pre-deletion source is available at:

```bash
git show 852976723d813352cabd5690f0acaab910f86c4e^:<path>
```

Line counts from the recoverable source:

- hard dynamic teacher: 2210 lines
- late-trigger dynamic experts: 1482 lines
- successful dynamic postprocessor: 520 lines
- RGB renderer from env states: 658 lines

## Useful Pieces

The recoverable source contains useful implementation references:

- scenario sets for late moving target, reverse motion, continuous motion,
  peg drop, and peg disturb;
- target motion path helpers;
- motion-planner based physical teacher attempts;
- H5 writing conventions with `summary_json`, `env_states`, actions, slots,
  perturb traces, and source audit files;
- RGB rendering from saved env states into Cosmos-style video datasets.

These are relevant to B/C/D/E runner design, but only after active-route
review.

## Risks That Block Direct Reuse

The recoverable source is not ready to run as active code:

- default outputs point under `experiments/world_model_task_rebinding/...`;
- hard dynamic and late-trigger teacher scripts call `box.set_pose`, and the
  hard teacher also calls `peg.set_pose`;
- the renderer calls `env.unwrapped.set_state_dict(...)` to render saved env
  states;
- generated teacher data is teacher/diagnostic data, not deployed method
  success;
- source was intentionally deleted during the 2026-07-03 cleanup, so it must
  not be silently revived as active route code.

Because of these risks, do not restore the files into active
`scripts/world_model/` and do not wrap them as B/C/D/E runners without a
reviewed adaptation.

## Adaptation Direction

For B dynamic RGB observation:

- reuse scenario naming, H5 manifest conventions, perturb traces, and future
  target labels;
- build a new active runner that writes under
  `experiments/maniskill/runs/01_dataset/dynamic_rgb/<try_or_tag>/`;
- execute inside Slurm allocation only;
- mark failures and target-assisted events as diagnostic/negative, never
  positive DP BC.

For C frozen-DP dynamic failure:

- use the official DP checkpoint unchanged;
- add dynamic scene perturbations through a reviewed active environment path;
- record action traces, outcome labels, and RGB review artifacts;
- do not use teacher future labels as controller inputs.

For D future-frame teacher:

- use ground-truth future target trajectory only for teacher generation;
- record `teacher_only=true` and `method_evidence_allowed=false`;
- require legal controller actions, no snap or final state edit.

For rendering:

- prefer active renderer code with the current Vulkan/canary rules;
- saved-state rendering is acceptable as RGB dataset reconstruction only when
  clearly labeled as rendering from recorded states, not physical rollout
  success.
