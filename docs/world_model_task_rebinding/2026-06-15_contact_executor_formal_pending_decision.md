# Contact Executor Formal Pending Decision

## Boundary

This note is a waiting-state decision record, not a final result. The current
formal contact/progress executor training has not yet reached the required
`10800` second floor, and no `training_summary.json` or `checkpoint_final.pt`
exists yet. Therefore no closed-loop video evaluation is allowed from this
checkpoint.

Active run:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k`

Slurm allocation:

- job `128023`
- node `server54`
- `2` H200 GPUs
- tmux session `cosmos3_contact_executor_formal_2gpu_20260615`
- watcher tmux session `cosmos3_contact_executor_formal_watch_20260615`

## Current Mid-Run Evidence

Latest checked state around `2026-06-15T15:08+08:00`:

- latest step: `234000`
- elapsed training time: `3127` seconds
- eval action MSE: `0.013708`
- frozen-DP-prior eval MSE: `0.0015608`
- action MSE ratio: `8.78x` worse than DP prior
- eval progress MSE: `0.011277`
- inserted accuracy: `0.961`
- DP-continuable accuracy: `0.883`
- all eval history points so far are worse than the DP prior on action MSE

Plain reading: the contact/progress readouts are learnable, but the
deterministic action head is not a safe executor. This remains mid-run
evidence only until the formal floor is reached.

## What Is Not Allowed Yet

- Do not launch closed-loop live eval from `checkpoint_latest.pt`.
- Do not report any video evidence for this formal contact executor; none has
  been produced because the gate is still closed.
- Do not treat oracle phase-selector diagnostics as controller evidence.
- Do not use train-action-MSE scale calibration as the next method; the
  stricter diagnostic already showed it overfits.

## Final Gate Branches

When `training_summary.json` and `checkpoint_final.pt` exist, the watcher will:

1. refresh the history summary;
2. run final group inspection from the current root's `checkpoint_final.pt`;
3. write `formal_live_eval_gate.json`.

If `formal_live_eval_gate.json` has `live_eval_allowed=true`, the next action
is a closed-loop panel with full `301/300` contract, saved videos/contact
sheets, and direct visual inspection before any claim.

Interface readiness check: the live loop and panel wrappers now accept
`controller_action_source=contact_executor`. The contact checkpoint loader and
dummy contact action chunk were smoke-tested inside allocation `128023` using
the current `checkpoint_latest.pt`: the loader reported `horizon=8`,
`feature_dim=218`, and `target_dim=56`, and the dummy chunk produced finite
`56` action values plus progress readout. This only proves the execution
interface is wired; it does not allow live eval before the final gate.

15:17 update: the formal run is still active and still has no final summary.
Latest checked training point:

- latest step: `274000`
- elapsed training time: `3662.78` seconds
- remaining time to formal floor: about `7137` seconds
- eval action MSE: `0.013864`
- frozen-DP-prior eval MSE: `0.0015608`
- action MSE ratio: `8.88x` worse than DP prior
- eval progress MSE: `0.011028`
- inserted / DP-continuable accuracies: `0.948` / `0.883`
- all eval history points so far are worse than DP prior

This strengthens the same mid-run reading, but it is still not a final result
until the formal floor is reached and the final gate is written.

If `live_eval_allowed=false`, stop and report the final blocker to the user.
The likely blocker, if the current trend persists, is:

> The model learns contact/progress readouts but the deterministic action head
> remains much worse than the frozen DP prior on held-out actions. The next
> aligned method is a regularized candidate/diffusion executor with held-out
> progress/value/contact-continuability scoring, not direct execution of this
> deterministic head.

## 15:23 Guarded Live Launcher Prepared

Added:

`scripts/slurm/run_cosmos3_contact_executor_live_panel_after_gate_in_allocation.sh`

Purpose:

- refuse login-node execution;
- require the formal final files:
  `training_summary.json`, `checkpoint_final.pt`,
  `post_training_group_metrics.json`, and `formal_live_eval_gate.json`;
- refuse live evaluation unless `formal_live_eval_gate.json` has
  `live_eval_allowed=true`;
- bind the future contact-executor live panel to the clean-dense
  `iter_000001500` Cosmos SFT root and the current formal contact-executor
  `checkpoint_final.pt`, rather than the stale generic wrapper defaults that
  still point at the older `iter2700` chain.

Verification:

- `bash -n scripts/slurm/run_cosmos3_contact_executor_live_panel_after_gate_in_allocation.sh`
  passed.
- Refusal-path check with Slurm-like environment variables and the current
  missing final summary exited before live launch:
  `rc=2`,
  `missing_training_summary=.../contact_executor_train_20260615_formal_2gpu_server54_save20k/training_summary.json`.

Boundary:

This is launch hygiene only. It does not produce video evidence and does not
open the gate. The current formal run still has no final summary or final
checkpoint, so live evaluation remains forbidden.

## 15:27 Mid-Run Watch

The formal training step is still active and has not reached the required
`10800` second floor.

Current Slurm state:

- allocation `128023` is still running on `server54`;
- formal training step `128023.63` has run about `1:11:55`;
- `training_summary.json` is still missing;
- `checkpoint_final.pt` is still missing;
- the existing gate remains closed with
  `failure_reasons=["missing_summary_or_group_metrics"]`.

Refreshed history summary:

- latest step: `320000`;
- elapsed training time: `4303.65` seconds;
- eval action MSE: `0.014158`;
- frozen-DP-prior eval MSE: `0.0015608`;
- action MSE ratio: `9.07x` worse than DP prior;
- eval progress MSE: `0.011196`;
- inserted / DP-continuable accuracies: `0.948` / `0.883`;
- `161/161` eval history points are worse than the DP prior on action MSE.

GPU spot check inside the held allocation:

- H200 `0`: `100%`;
- H200 `1`: `12%`.

Plain reading remains unchanged: the run is using the allocation and must
continue to the formal floor, but the trend strongly suggests the final gate
will fail on action generation unless the final phase changes sharply. No live
eval is allowed from this mid-run checkpoint.

## 15:30 Mid-Run Watch

The formal step is still active:

- allocation `128023` remains running on `server54`;
- formal training step `128023.63` has run about `1:15:16`;
- `training_summary.json` is still missing;
- `checkpoint_final.pt` is still missing;
- `formal_live_eval_gate.json` remains closed with
  `failure_reasons=["missing_summary_or_group_metrics"]`.

Latest parsed training point:

- step `332000`;
- elapsed training time `4479.05` seconds;
- eval action MSE `0.014241`;
- frozen-DP-prior eval MSE `0.0015608`;
- action MSE ratio `9.12x` worse than DP prior;
- eval progress MSE `0.011084`;
- inserted / DP-continuable accuracies `0.948` / `0.896`;
- train action MSE `1.12e-06`.

Boundary remains the same: no live evaluation is allowed until the final
summary, final checkpoint, final inspection, and a positive formal gate exist.

## 15:32 Mid-Run Watch

The run still has not reached the formal gate point:

- allocation `128023` remains running on `server54`;
- formal training step `128023.63` has run about `1:17:12`;
- `training_summary.json` is still missing;
- `checkpoint_final.pt` is still missing;
- `formal_live_eval_gate.json` remains closed with
  `failure_reasons=["missing_summary_or_group_metrics"]`.

Refreshed history summary:

- latest step `340000`;
- elapsed training time `4592.97` seconds;
- eval action MSE `0.014300`;
- frozen-DP-prior eval MSE `0.0015608`;
- action MSE ratio `9.16x` worse than DP prior;
- eval progress MSE `0.010837`;
- inserted / DP-continuable accuracies `0.948` / `0.883`;
- `171/171` eval history points are worse than DP prior on action MSE.

GPU spot check inside the held allocation showed H200 utilization `11%/100%`,
consistent with the earlier alternating imbalance. The allocation is active;
do not stop it before the formal floor.

## 15:34 Mid-Run Watch

The run remains pre-final:

- allocation `128023` is still running on `server54`;
- formal training step `128023.63` has run about `1:18:37`;
- `training_summary.json` is still missing;
- `checkpoint_final.pt` is still missing;
- the gate is still closed with
  `failure_reasons=["missing_summary_or_group_metrics"]`;
- no `live_receding_contact_executor_iter1500_panel*` output directory exists.

Refreshed history summary:

- latest step `348000`;
- elapsed training time `4707.75` seconds;
- eval action MSE `0.014705`;
- frozen-DP-prior eval MSE `0.0015608`;
- action MSE ratio `9.42x` worse than DP prior;
- eval progress MSE `0.011102`;
- inserted / DP-continuable accuracies `0.948` / `0.896`;
- `175/175` eval history points are worse than DP prior on action MSE.

GPU spot check inside the held allocation showed H200 utilization `12%/100%`.
The correct action remains to preserve the allocation and wait for the formal
floor and final gate; no live eval is allowed from `checkpoint_latest.pt`.

## 15:37 Decision Summary Helper

Added:

`scripts/world_model/summarize_cosmos3_contact_executor_decision.py`

Purpose:

- read `training_summary.json`, `formal_live_eval_gate.json`,
  `post_training_group_metrics.json`, and `training_history_summary.json`;
- write a plain `formal_decision_summary.json/md`;
- state exactly one next action:
  waiting for final files, launch guarded live eval, or stop and report the
  final blocker;
- never relax the formal gate and never launch live eval.

Verification:

- `python -m py_compile` passed;

## 17:25 Finalization Blocker

The run reached the formal wall-clock region but did not produce a usable
formal result.

Current evidence:

- Slurm allocation `128023` is still running on `server54`.
- Formal step `128023.63` has run about `3:09`.
- `training_summary.json` is missing.
- `checkpoint_final.pt` exists but is only `15895` bytes.
- Loading `checkpoint_final.pt` with `.venv/bin/python` and `torch.load` fails
  with `OSError(22, 'Invalid argument')`.
- `checkpoint_latest.pt` is valid and loads, step `780001`, size about `199M`.
- The last training-history point is step `796000`, elapsed `10796.32`
  seconds, eval action MSE `0.015157`, DP-prior eval MSE `0.0015608`, action
  MSE ratio `9.71x` worse than the DP prior, eval progress MSE `0.012391`,
  inserted accuracy `0.948052`, and DP-continuable accuracy `0.883117`.
- The refreshed decision summary records `399/399` eval points worse than the
  DP prior.
- A compute-node process check showed both training ranks still alive at
  `100%` GPU after `checkpoint_final.pt` stopped growing.

Plain interpretation:

This is now a concrete implementation/finalization blocker, not permission to
run live evaluation. The formal run is stuck or corrupted while writing the
final checkpoint/summary. Separately, the offline action trend says the
deterministic contact-executor action head remains far worse than the frozen
DP prior, so the live gate would remain closed even if the final file were
repaired.

Boundary:

- Do not launch contact-executor live eval from this run.
- Do not substitute `checkpoint_latest.pt` for the required formal
  `checkpoint_final.pt` as method evidence.
- Stop for user direction before interrupting/restarting this formal run or
  moving to the next candidate/diffusion executor implementation.

## 17:28 Final Failure And Code Repair

The formal Slurm step exited after the NCCL watchdog fired. No manual
`Ctrl-C` was sent.

Final state:

- Slurm step `128023.63` terminated with exit code `1`.
- The held allocation `128023` remains alive on `server54`.
- Both GPUs are idle after the crash.
- `training_summary.json` is still missing.
- `checkpoint_final.pt` is about `54M` but still invalid:
  `torch.load` fails with
  `PytorchStreamReader failed reading zip archive: failed finding central directory`.
- `checkpoint_latest.pt` remains valid at step `780001`.

Console root cause:

- rank1 timed out in NCCL `ALLREDUCE` with `NumelIn=1`;
- rank0 timed out in NCCL `ALLREDUCE` with `NumelIn=4323388`;
- torchrun reported `ChildFailedError`, rank0 `SIGABRT`;
- this is consistent with different ranks making different stop decisions near
  the wall-clock floor, so one rank entered finalization while another rank
  was still in/back near gradient synchronization.

Code repair applied:

- `scripts/world_model/train_cosmos3_contact_executor.py` now synchronizes the
  stop decision across ranks with a scalar all-reduce.
- JSON and checkpoint writes now use temp-file plus atomic rename.
- The final checkpoint is loaded once after saving; if that load fails, the
  trainer records `training_final_checkpoint_error.json` and raises instead of
  silently leaving a bad final checkpoint plus missing summary.
- Syntax check passed:
  `.venv/bin/python -m py_compile scripts/world_model/train_cosmos3_contact_executor.py`.

Boundary:

This repair only prevents the same finalization bug in a future rerun. It does
not make the failed formal run usable. It also does not change the scientific
blocker: the last held-out action metric was still about `9.71x` worse than
the frozen DP prior, so the deterministic contact-executor action head remains
unfit for live closed-loop evaluation.

## 17:31 Watcher/Decision Hygiene

The old decision summary still said "waiting" because it only checked whether
`training_summary.json` was missing. That was wrong for this state: the formal
process had already failed and the final checkpoint was invalid.

Fixes applied:

- `scripts/world_model/summarize_cosmos3_contact_executor_decision.py` now
  attempts to load `checkpoint_final.pt` and scans the console tail for
  fatal training errors.
- The current failed root now produces
  `status=failed_invalid_final_checkpoint_stop_for_user`.
- The blockers are now explicit:
  `training_summary.json`, `invalid_checkpoint_final`, and
  `training_process_failed`.
- The script exits with return code `2` for this failed/stop state.
- `scripts/slurm/watch_cosmos3_contact_executor_formal_inspect.sh` now calls
  the decision summary if `checkpoint_final.pt` exists without
  `training_summary.json`; if the summary returns nonzero, the watcher exits
  instead of waiting forever.
- `scripts/slurm/watch_cosmos3_contact_executor_decision_summary.sh` now exits
  when the refreshed decision summary returns a failed/stop state instead of
  polling forever.
- The stale already-running watcher tmux sessions were stopped after these
  fixes because they had loaded the previous wait-loop logic. The Slurm
  allocation was not cancelled.

Verification:

- `.venv/bin/python -m py_compile` passed for the decision summary and trainer.
- `bash -n` passed for both watcher scripts.
- Running the refreshed decision summary on the failed root wrote the corrected
  `formal_decision_summary.json/md` and returned `rc=2`.
- running it on the current pre-final root wrote
  `formal_decision_summary.json` and `formal_decision_summary.md`;
- current status is `waiting_for_formal_floor_or_final_files`;
- current blockers are `training_summary.json` and `checkpoint_final.pt`;
- next action is to keep the held allocation and not run live eval from
  `checkpoint_latest.pt`.

## 15:40 Decision Watcher

Updated:

`scripts/slurm/watch_cosmos3_contact_executor_formal_inspect.sh`

so that after it writes `formal_live_eval_gate.json`, it also runs the
read-only decision summarizer and writes `formal_decision_summary.json/md`.

Because the already-running watcher may not reliably reread modified lines
after its current sleep loop, added and launched a companion lightweight
watcher:

`scripts/slurm/watch_cosmos3_contact_executor_decision_summary.sh`

tmux session:

`cosmos3_contact_executor_decision_watch_20260615`

Current log:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_train_20260615_formal_2gpu_server54_save20k/formal_decision_watch.log`

Status:

- syntax checks passed for both watcher scripts;
- the decision watcher is polling every `300` seconds for
  `training_summary.json`, `checkpoint_final.pt`, and
  `formal_live_eval_gate.json`;
- it does not use GPU;
- it does not launch live eval.

15:40 refresh:

- `formal_decision_summary.md` was refreshed manually once after adding the
  watcher;
- current status remains `waiting_for_formal_floor_or_final_files`;
- latest reflected step is `376000`;
- elapsed training time is `5074.31` seconds;
- eval action MSE is `0.014480` versus DP prior `0.0015608`
  (`9.28x` worse);
- `189/189` eval points are worse than DP prior;
- no `live_receding_contact_executor_iter1500_panel*` output directory exists.

## 15:42 Watcher Return-Code Fix

The decision summarizer intentionally returns nonzero when the final state is
`gate_closed_stop_for_user`. That is a valid experimental conclusion, not a
watcher crash. Updated both watcher paths:

- `scripts/slurm/watch_cosmos3_contact_executor_formal_inspect.sh`;
- `scripts/slurm/watch_cosmos3_contact_executor_decision_summary.sh`.

They now record the decision summarizer return code in the log and continue to
write a clear completion line. The companion decision watcher was restarted as:

`cosmos3_contact_executor_decision_watch_20260615`

Current restart time:

`2026-06-15T15:42:27+08:00`

Current decision watcher log again shows it is waiting for final decision
inputs. GPU spot check at the same time was `14%/100%`, and no contact-executor
live output directory exists.

## 15:44 Blocked-On-External-State Check

The same blocking condition has now repeated across multiple continuation
checks:

- formal training step `128023.63` is still running, about `1:28:41`;
- `training_summary.json` is still missing;
- `checkpoint_final.pt` is still missing;
- the formal gate remains closed because final files do not exist;
- latest parsed point is step `390000`, elapsed `5267.55` seconds;
- eval action MSE is `0.014497` versus DP prior `0.0015608`
  (`9.29x` worse);
- no `live_receding_contact_executor_iter1500_panel*` output directory exists.

All aligned preparation work is in place: formal watcher, decision watcher,
guarded live launcher, and decision summary writer. The remaining change must
come from the external training process reaching the formal floor and writing
the final files. Until those files exist, running live eval is forbidden.
