# Executor Path Start After Dense Failure

Date: 2026-06-15

## Plain Result

The dense Cosmos3 SFT is no longer the startup blocker. It trained past the
current formal floor and closed-loop still failed. Repeating the same raw
Cosmos-action loop is not the next step.

The executor branch has now started:

- build executor samples from dense rows;
- export frozen static-DP action proposals as prior input;
- run a two-sample residual-executor overfit.

This is a real start of the next method path, but it is not formal method
evidence yet.

## Resource Rule

The active formal-training floor is now:

- at least `2` GPUs;
- at least `3` hours;
- `4` GPUs only if already available sooner.

Short overfit/sanity training remains the exception:

- `1-2` GPUs;
- about `50-100` steps;
- no `3` hour minimum;
- debug gate only.

All work below ran inside tmux-held Slurm compute allocation `128006` on
`server62`, not on the login node.

## What Ran

Executor dataset preflight:

`experiments/world_model_task_rebinding/cosmos3/executor_dataset_clean_dense_late299_chunk24_debug_20260615_executor_preflight_full_debug`

Summary:

- `schema_ok=true`
- `ready_for_debug_overfit=true`
- `samples_written_total=6969`
- train samples: `6359`
- val samples: `610`
- roles: `target_motion_observed=2572`, `target_post_motion=4037`,
  `insert_resume=360`

Formal blockers recorded by the preflight:

- `gt_task_path_debug_not_formal_evidence=6969`
- `missing_dp_prior_actions=6969`

DP-prior smoke:

`experiments/world_model_task_rebinding/cosmos3/executor_dp_prior_smoke_20260615_dp_prior_smoke2`

Summary:

- `num_records=2`
- `failure_counts={}`
- DP checkpoint:
  `experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt`
- `ready_for_debug_executor_overfit=true`

Executor overfit smoke:

`experiments/world_model_task_rebinding/cosmos3/executor_overfit_smoke_20260615_executor_overfit2`

Summary:

- `num_samples=2`
- residual executor trained for `100` steps on CUDA;
- DP-prior baseline MSE: `0.0002738447`
- step-1 residual MSE: `0.0035156261`
- final MSE: `3.46294e-07`
- max absolute action error: `0.00203459`
- `ready_for_debug_gate=true`

## What This Proves

The executor data path is not just a written plan anymore:

1. Dense rows can be converted into executor samples.
2. Frozen DP can be loaded and queried for prior action chunks from restored
   source states.
3. A small residual executor can overfit the matched two-row interface.

So the immediate code/interface blocker is cleared.

## What It Does Not Prove

This is not a dynamic-task success claim.

The executor overfit still uses `gt_state_targets_debug` for the task path.
That is future ground-truth task state. It is acceptable for an interface
smoke, but it is not a legal controller-facing input for formal evidence.

Full executor training must not start until this input is replaced by causal
Cosmos-predicted task paths/readouts.

## Current Blocker

The current blocker is precise:

`executor needs causal Cosmos task-path predictions as input, not GT future
task paths`.

DP prior is no longer a pure unknown because the two-row smoke worked, but it
still needs to be exported at the scale selected for training.

## Next Step

1. Export Cosmos-predicted task paths/readouts from the trained dense
   `iter_000001500` checkpoint for executor rows.
2. Join those predicted paths with DP-prior chunks and teacher action targets.
3. Rerun the two-sample executor overfit without `gt_state_targets_debug`.
4. If that passes, request/use a `2` GPU allocation for at least `3` hours and
   start full executor or DP-prior residual training.

Do not spend a 2-GPU formal-training allocation before step 3 passes, because
that would train against an oracle input and would not answer the research
question.
