# Cosmos3 SFT And Closed-Loop Current Status

Date: 2026-06-13 CST

## Scope

This note records the current state after the user-directed corrected
closed-loop eval. It is a monitoring/status artifact only. No SFT, rendering,
Cosmos inference, or simulator rollout was launched to create it.

## Cluster State

- Slurm job `127350` is still held on `server10`, but only as
  `127350.extern`.
- There is no active Slurm step for training or eval.
- Process checks found no active `torchrun`, Cosmos inference, SFT training,
  or live-receding eval process.

The held allocation therefore exists, but it is not currently doing useful GPU
work.

## Active SFT Root

Root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`

Condition root:

`experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_rgb_300step_20260612_0245`

Checkpoints present:

- `iter_000000300`
- `iter_000000600`
- `iter_000000900`
- `iter_000001200`
- `iter_000001500`
- `iter_000001800`
- `iter_000002100`
- `iter_000002400`
- `iter_000002700`

Training status:

- `sft_completed` exists with timestamp `2026-06-13T12:23:29+08:00`.
- `sft_train.log` has a later mtime, `2026-06-13 12:41:51 +0800`, and the
  latest visible training line is around iteration `2735`.
- Therefore `sft_completed` is not sufficient evidence that the latest resumed
  training segment finished a planned `MAX_ITER`; the authoritative current
  state is: training is stopped now, latest checkpoint is `iter_000002700`,
  and no training process is active.

Validation-loss diagnostics:

- iteration `2400`: `0.118837`
- iteration `2700`: `0.123388`

These are training diagnostics only. They do not override generated-video,
action/readout, visual-review, or live-simulator evidence.

## Generated Eval Status

Eval roots exist through `eval_full_episode_wam_iter_000002700`.

The latest generated eval gate is not controller-positive:

- `closed_loop_allowed=false`
- `visual_review_status=fail`
- reason: `explicit_visual_review_not_passed`
- visual note: `manual_visual_review.json: 4 fail / 4 pass /
  2 pass_with_caution; unsafe late robot/peg/hole handoff geometry remains`

Key `iter_000002700` generated-eval metrics:

- `num_eval_samples=10`
- mean future video PSNR: `22.3724`
- mean robot-action future RMSE: `0.6147`
- mean state-sidecar future RMSE: `0.4238`
- mean final hole position error: `0.0669 m`

The task-state readout summary says all 10 rows are structurally strict, but
the readout boundary explicitly states it is not controller success evidence.

## Corrected Live Closed-Loop Status

The corrected live-receding eval was advanced after the user explicitly said
not to run more smoke training. It failed.

Primary corrected live panel:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_panel10_corrected_iter2100_20260613_161006`

Observed failures:

- sample `00`, `hole_late_move_stop`: 3 receding chunks, final
  peg-head-at-hole `[-0.2652, 0.0979, 0.0020]`, success `false`
- sample `01`, `hole_late_constant`: 3 receding chunks, final
  peg-head-at-hole `[-0.2281, 0.0420, 0.0020]`, success `false`
- sample `02`, `hole_late_reverse`: already failing after partial execution

Long-horizon corrected check:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_promptfix_sample0_longhorizon_iter2100_20260613`

Result:

- sample `00`, `hole_late_move_stop`
- target-motion-onset prefix `f106`
- 12 receding Cosmos chunks through frame `202`
- source teacher inserts by frame `166`
- live final success `false`
- live final peg-head-at-hole `[-0.1423, 0.0004, 0.0101]`
- DP handoff: `0` steps, because real-state `C_pi` correctly blocked handoff

Directly opened contact sheets show the same qualitative failure: the target
moves, but robot/peg do not rebind into the moved hole and the final state is
not DP-continuable.

## Current Failure Classification

This is corrected closed-loop negative evidence for the current checkpoint and
condition root. The failure is not currently explained by:

- stale source-H5 future rows in live WAM conditioning;
- missing external target motion in live eval;
- prompt-only schema drift;
- blind long frozen-DP takeover;
- a simple action-row temporal offset.

The stronger diagnosis remains:

- direct Cosmos action chunks underreact in late dynamic rebind states;
- generated video/readout can be optimistic and cannot be the authority for
  DP handoff;
- current SFT condition export has role/mode drift (`1193/2899` mismatches);
- current prefixes are sparse source-teacher masks rather than dense receding
  recovery states from late/off-source miss regimes.

## Next Allowed Work

Do not continue training from the current condition root as if more iterations
alone are the answer. Do not launch more broad panels from the failed
checkpoint.

The prepared next step is a user-approved repair preflight only:

```bash
bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_preflight_in_allocation.sh
```

This must be run only inside a compute-node allocation and defaults
`RUN_SFT=false`. It should produce a clean-role/dense-receding condition root
and preflight summary, not a new checkpoint.

Training may resume only after:

1. the clean/dense condition preflight writes
   `clean_dense_preflight_summary.json`;
2. that summary has `ready_for_overfit=true`;
3. the user explicitly approves the next overfit SFT step.
