# Candidate Executor Start

Date: 2026-06-15

## Why This Exists

The deterministic contact/progress executor failed as a main controller path.
It learned contact/progress readouts, but its held-out action head stayed far
worse than the frozen DP prior. The next method direction is therefore not
another single residual head. The active direction is:

```text
Cosmos low-frequency task/contact imagination
  -> stochastic candidate action chunk generator
  -> action-conditioned progress/contact/value scorer
  -> execute only a short prefix
  -> real re-observation and video/final-state review
```

This is the local DDP/HDP borrowing point: DDP tightly couples imagined
representations with chunked policy execution, and HDP makes contact phase a
first-class control condition. For this peg-insertion task, the executor must
choose among multiple plausible contact chunks instead of averaging them by
MSE.

## Code Added

- `scripts/world_model/train_cosmos3_candidate_executor.py`
  - reads the causal contact executor dataset;
  - trains a stochastic residual distribution over short action chunks;
  - trains an action-conditioned scorer for progress, insertion,
    DP-continuability, and scalar value;
  - evaluates candidate selection over DP prior, generator mean, scaled
    residual candidates, and stochastic samples;
  - writes manifests, checkpoints, training history, and summary.
- `scripts/slurm/run_cosmos3_candidate_executor_train_in_allocation.sh`
  - refuses login-node execution;
  - runs the trainer under `torchrun` inside a compute-node Slurm step;
  - does not launch live evaluation.
- `scripts/world_model/run_cosmos3_live_receding_loop.py` and
  `scripts/world_model/run_cosmos3_live_receding_panel.py`
  - now accept `--controller-action-source=candidate_executor`;
  - load a candidate-executor checkpoint;
  - build the same causal feature as training from live state, Cosmos
    task path, frozen-DP prior, and live contact context;
  - sample/scale residual candidates and select one with the
    progress/contact/value scorer;
  - write `candidate_executor_action_chunk.json` for each receding iteration.
- `scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`
  - refuses login-node execution;
  - refuses live eval unless the formal candidate-executor summary has
    `formal_training_floor_met=true` and `ready_for_formal_live_eval=true`;
  - uses the same formal root's post-floor `checkpoint_final.pt`; early
    `checkpoint_best_offline.pt` files remain diagnostic and cannot launch
    live evaluation;
  - refuses an explicit `EXECUTOR_CHECKPOINT` override unless it resolves to
    `${FORMAL_ROOT}/checkpoint_final.pt`.

## Verification So Far

- Python syntax check passed:
  `.venv/bin/python -m py_compile scripts/world_model/train_cosmos3_candidate_executor.py`
- Wrapper syntax check passed:
  `bash -n scripts/slurm/run_cosmos3_candidate_executor_train_in_allocation.sh`
- 23:22 CST diffusion live-interface self-test passed:
  `.venv/bin/python scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py`.
  This now covers the actual
  `run_cosmos3_live_receding_loop.candidate_executor_action_chunk` path with a
  toy diffusion checkpoint whose feature width matches live inputs:
  current state, causal Cosmos task path, frozen-DP prior, and live contact
  context. It verifies finite selected actions, `diffusion_*` candidate
  records, rank metadata, and a `2x7` denormalized robot action chunk. This is
  only interface evidence; it does not replace formal training, final gates,
  or closed-loop video review.
- 23:28 CST gated live-launcher discipline was hardened:
  `run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh` now
  refuses an explicit `EXECUTOR_CHECKPOINT` override unless it resolves to the
  same formal root's `checkpoint_final.pt`. A local refusal-path check with a
  synthetic `checkpoint_best_offline.pt` returned rc `44` with
  `reason=executor_checkpoint_not_formal_final`. This prevents an early
  best-offline checkpoint from bypassing the post-floor final-checkpoint gate.
- 23:31 CST diffusion-chain status reporting now includes the current formal
  run's configured wall-clock floor and remaining seconds to that floor. This
  is observability only; it does not alter any gate. Manual refresh of the
  current no-sample formal root reported about `4089.70` seconds remaining to
  the `10800` second floor, latest step `319500`, selected-action MSE
  `0.00164796`, frozen-DP prior MSE `0.00156083`, and `24/24` recent eval
  points worse than DP.
- 23:36 CST candidate-executor gates were hardened against a pure-DP fake
  pass. Offline readiness now requires at least one selected non-`dp_prior`
  candidate, and the after-gate watcher/live launcher independently reject a
  final eval whose `candidate_source_counts` collapse entirely to
  `dp_prior`. A local fake-summary check confirmed the live launcher refuses
  that case with `reason=final_selected_candidate_collapsed_to_dp_prior`.
  This keeps a passing run tied to the actual candidate/diffusion executor
  objective rather than silently becoming the frozen-DP baseline.
- 23:42 CST after-gate diffusion readiness was hardened against stale
  non-diffusion summaries. The diffusion watcher now requires
  `generator_type=diffusion`, `candidate_samples>0`, and
  `candidate_rank_diffusion_count>0` before treating smoke/formal summaries as
  ready. The formal-diffusion live handoff sets
  `REQUIRE_DIFFUSION_GENERATOR=true`, and local refusal checks verified that a
  gaussian summary and a diffusion summary with disabled samples are rejected.
  This keeps the automatic chain tied to the requested diffusion action-chunk
  generator rather than any older candidate-executor root.
- 23:50 CST restarted only the lightweight after-gate diffusion watcher to
  load the latest hardened gate code. The old watcher tmux pane exited after
  `Ctrl-C`, so a new `cosmos3_candidate_diffusion_after_gate_watch_20260615`
  session was created. The fresh log starts at `23:50:18+08`, watcher PID is
  `1625010`, and formal training step `128888.84` remained active.
- 23:54 CST removed the non-diffusion live auto-launch branch. The old
  guarded-live watcher PID `271336` was stopped, but formal training and the
  post-gate status watcher were left running. The diffusion watcher now
  continues to diffusion smoke after the current post-floor formal decision
  even if the no-sample candidate baseline passes. Fresh watcher PID is
  `1774777`, with log start `23:54:05+08`.
- Short `512`-row smoke with conservative source penalty passed the offline
  smoke gate:
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_candidate_smoke512_100step_excl13_sourcepenalty`.
  It ran only `100` steps, so `formal_training_floor_met=false` and it is not
  live-eval evidence. The important numbers are:
  DP-prior eval action MSE `0.00156083`, generator-mean action MSE
  `0.00152223`, selected action MSE `0.00156491`, progress MSE `0.0127258`,
  inserted accuracy `0.987013`, DP-continuable accuracy `0.857143`, and
  `ready_for_offline_gate=true`.
- Interpretation of the smoke: the candidate/scorer interface is wired and
  the generator mean has useful held-out signal. The scorer must stay
  conservative; without source/logprob/fallback penalties it over-selects
  noisy stochastic samples.

## Resource State

A tmux-held interactive allocation was requested with `salloc`, not `sbatch`:

- Slurm job: `128862`
- tmux session: `cosmos3_candidate_exec2gpu_20260615`
- request: `2` GPUs, `1` day
- current state: started on `server13`, but not usable for PyTorch CUDA

The first short smoke did not enter model training. The CUDA canary failed:

- `nvidia-smi` sees H200 GPUs;
- `/dev/nvidia*` permissions are present;
- Slurm sets `CUDA_VISIBLE_DEVICES=0,1`;
- PyTorch fails CUDA initialization with `CUDA unknown error`;
- one-GPU and two-GPU checks both fail.

Failed smoke console:

`experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_candidate_smoke64_100step.console.log`

This is a scheduling/node CUDA preflight failure, not a candidate-executor
training result.

Wrapper hygiene fix:

- `scripts/slurm/run_cosmos3_candidate_executor_train_in_allocation.sh` no
  longer defaults `CUDA_VISIBLE_DEVICES=0,1`;
- it lets Slurm provide GPU visibility unless `CUDA_VISIBLE_DEVICES_OVERRIDE`
  is explicitly set.

A second tmux-held allocation was requested, excluding `server13`:

- Slurm job: `128888`
- tmux session: `cosmos3_candidate_exec2gpu_excl13_20260615`
- request: `2` GPUs, `1` day
- current state: running on `server33`; CUDA canary passed

## Next Gate

The short smoke gate has passed. The next gate is the formal `2` GPU / `3`
hour candidate-executor training. Formal live evidence must use the post-floor
final checkpoint. Early best checkpoints are diagnostic only; they cannot be
used to dodge the user's minimum training rule.

- Slurm job: `128888`
- node: `server33`
- tmux session: `cosmos3_candidate_formal_globalfallback_2gpu_20260615`
- output root:
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_globalfallback_finalgate`
- boundary: no live eval unless final `training_summary.json` has
  `formal_training_floor_met=true` and `ready_for_formal_live_eval=true`

Startup evidence:

- wrapper CUDA canary passed with `2` H200 GPUs;
- `training_history.json`, `training_manifest.json`,
  `checkpoint_latest.pt`, and `checkpoint_best_offline.pt` were written;
- the active selector now uses global conservative DP fallback
  (`dp_fallback_phases=all`), so a non-DP candidate must beat the DP candidate
  by the saved margin in any phase before execution;
- early metric at step `1000`: selected eval action MSE `0.00186330` versus
  DP-prior eval action MSE `0.00156083`, progress MSE `0.00970525`,
  inserted accuracy `0.961039`, and DP-continuable accuracy `0.896104`. This
  is still pre-formal-floor evidence only.

Closed-loop videos come only after the formal gate is valid.

23:22 CST current-gate status: the active no-sample formal run is still
pre-floor at about `103` minutes. Latest chain summary is
`waiting_current_formal_gate`; latest step `291500` selected-action MSE is
`0.00163363` versus frozen-DP prior `0.00156083`, and the recent 24-eval
window is still all worse than DP. This running checkpoint was launched before
the diffusion rank-calibration repair, so it remains a gate that must finish
or formally fail before the repaired diffusion candidate smoke can use the
held allocation.

Guarded continuation:

- tmux session `cosmos3_candidate_after_gate_watch_globalfallback_20260615` is
  waiting for the formal summary and final checkpoint;
- it calls
  `scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`
  through `srun --overlap` inside allocation `128888`;
- the launcher refuses live eval unless the formal `2` GPU / `3` hour gate and
  final-checkpoint offline gate both pass.

## Restart Note

The first formal attempt
`candidate_executor_train_20260615_formal_2gpu_server33_sourcepenalty` was
interrupted inside tmux with Ctrl-C. It did not have best-checkpoint support,
and by step `48000` validation selected-action MSE had degraded to `0.00378292`
versus DP prior `0.00156083`, with the selector choosing `mean` on `76/77`
validation rows. That interrupted run is not formal evidence. The restarted
run keeps the same method but records the best held-out offline checkpoint
inside the full-duration formal run.

The second attempt
`candidate_executor_train_20260615_formal_2gpu_server33_bestoffline_sourcepenalty`
was also interrupted before formal completion. The reason was not a training
crash: it exposed a discipline problem. Its early best checkpoint looked good,
but using that checkpoint for live eval would make the formal result depend on
a few-second checkpoint, which violates the user's minimum-training rule. The
active replacement is therefore the global-fallback final-checkpoint run above.

2026-06-15 20:49 CST update: the first heavier global-fallback final-gate
attempt,
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_globalfallback_heavy_finalgate`,
failed before training because the Slurm step could not allocate the requested
memory: `srun: error: Unable to create step for job 128888: Memory required by
task is not available`. It has no `training_history.json`, so this is a
scheduling request failure rather than candidate-executor evidence. The stale
heavy after-gate watcher was stopped without releasing allocation `128888`.

The active formal replacement is
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_globalfallback_2048_finalgate`
in tmux session `cosmos3_candidate_formal_globalfallback_2048_2gpu_20260615`,
Slurm step `128888.48` on `server33`. It keeps the same method boundary:
global conservative DP fallback, final-checkpoint-only live gate, `2` GPUs,
and the `10800` second formal floor. It reduces the model to
`hidden_dim=2048`, `num_layers=6`, `batch_size=256`, and uses
`candidate_samples=48` so the run fits the current allocation while remaining
stronger than the short smoke. Live eval is still gated off until the final
summary and `checkpoint_final.pt` prove the formal gate.

2026-06-15 20:56 CST selector repair: the `2048` formal restart was
interrupted inside tmux before the formal floor because it immediately repeated
the selector overfit pattern. By early evals, the scorer again selected raw
`mean` on most validation rows; around step `4000`, selected-action MSE was
about `0.00517` while the frozen-DP prior remained `0.0015608`. Continuing
that run would only burn the allocation on a known failed selector.

The candidate selector now has a policy-prior barrier derived from the training
distribution rather than hand-written failure cases: residual-L2 caps are
estimated per universal contact phase (`far`, `lateral_align`,
`preinsert_aligned`, `dp_continuable`), and raw `mean`, large-scale, and
stochastic sources can be penalized so the selector prefers DP or small
residual scales unless the progress/contact/value scorer gives a better
candidate. The live candidate-executor path loads the same caps from
`checkpoint_final.pt`, so offline gating and live action selection use the same
rule.

Short compute-node smoke with the repaired selector:
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_smoke512_100step_phasecap_sourcepen_margin0_4096`.
It used `2` GPUs for `100` steps only, so it is not formal evidence. It did
verify the selector shape needed before formal training: validation selected
only DP plus small residual scales (`scale_0.05`, `scale_0.1`, `scale_0.2`),
not raw mean or random samples, and selected-action MSE was `0.00153759`
versus DP-prior `0.00156083`. The next formal run should use this repaired
selector, `hidden_dim=4096`, final-checkpoint-only gating, and the `10800`
second floor.

2026-06-15 21:02 CST formal run launched with the repaired selector:
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_sourcepen_margin0_4096_finalgate`.
It is running in tmux session
`cosmos3_candidate_formal_phasecap_sourcepen_4096_2gpu_20260615`, Slurm step
`128888.58` on `server33`, with the same held allocation discipline. The
guarded after-gate watcher is
`cosmos3_candidate_after_gate_watch_phasecap_4096_20260615`; it waits for
`training_summary.json`, `checkpoint_final.pt`, and the training step exit
before calling the gated live launcher inside allocation `128888`.

Early formal metrics are aligned but not final evidence. Around step `8500`,
validation selected-action MSE was about `0.0012228` versus frozen-DP prior
`0.0015608`, while inserted and DP-continuable readouts remained above the
offline gate thresholds. Some selected candidates are still named `mean`, but
they are now selected only after the phase residual-L2 cap and source penalty;
the previous unrestricted mean-overfit failure was blocked by the selector
barrier. Formal evidence still requires this same final checkpoint to pass the
post-`10800` second summary gate and then produce real closed-loop video/final
state evidence.

2026-06-15 21:10 CST gate hardening: the candidate-executor formal/live gate
now requires the final selected-action MSE to be no worse than the frozen-DP
prior. The original `1.25x` tolerance was useful as an early diagnostic, but
it is too weak as a live-launch rule after the observed mean-overselection
failure. The running training process may still write the older summary field,
so the gated live launcher independently checks `final_metrics.eval` and
refuses live eval if `selected_action_mse > dp_prior_action_mse`. This does not
claim action MSE is the final task metric; it prevents a final checkpoint that
already damages the DP prior offline from consuming live evaluation.

2026-06-15 21:14 CST restart: the `phasecap_sourcepen_margin0_4096_finalgate`
formal run was interrupted inside tmux before the formal floor because the
latest validation points drifted just above the DP-prior action MSE. Under the
hardened gate, this checkpoint would not be allowed to run live, so continuing
it would only burn allocation time. The new active formal root is
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_meanpen5_4096_finalgate`.
It keeps the same phase residual-L2 cap and final-checkpoint-only gate, but
uses `score_mean_source_penalty=5.0`,
`score_large_scale_source_penalty=0.5`, and
`score_stochastic_source_penalty=1.0`. Plain reason: raw unscaled generator
mean repeatedly becomes an unsafe attractor during longer training; this run
keeps the candidate executor stochastic but makes the selector pay a stronger
cost before it can leave the DP-prior neighborhood.

2026-06-15 21:23 CST trust-region code update: while the mean-penalty formal
run continues, the candidate executor trainer/live path now supports an
explicit candidate scale set. The default remains backward-compatible
(`0.05,0.1,0.2,0.5,1.0`), so already-running checkpoints are not silently
reinterpreted. New runs can set `CANDIDATE_SCALES=0.05,0.1,0.2` to remove
large residual candidates from both offline selection and live execution.
This is a generic DP-prior trust region, not a hand-coded case split: Cosmos
still supplies the low-frequency task/contact path, but the action generator
must stay inside short-chunk policy corrections unless the learned scorer can
win within that allowed candidate family. Syntax checks passed for the
trainer, live loop, and candidate-executor train/live wrappers.

The same monitoring pass found the active mean-penalty run still pre-formal:
Slurm step `128888.73` is running, `training_summary.json` and
`checkpoint_final.pt` are absent, and live eval is still blocked. Through step
`22500`, validation selected-action MSE was mostly below the DP prior, but
step `22000` briefly rose to `0.00157423` versus DP `0.00156083`, with the
selector increasingly choosing `scale_1`. This is not failure evidence yet
because the final gate has not been reached, but it is the concrete reason the
next aligned restart should restrict candidate scales if the final gate fails.

2026-06-15 21:26 CST update: the mean-penalty run was stopped inside tmux
before the formal floor because the risk became a repeated failure pattern. In
the last 12 validation evals inspected before stopping, 9 were worse than the
frozen DP prior, and the selected candidate distribution was dominated by
`scale_1`. Representative validation points were step `28000`
selected-action MSE `0.00163134` versus DP `0.00156083`, step `32000`
`0.00163372`, and step `34000` `0.00160710`. This run is not method evidence
and must not feed live evaluation.

2026-06-15 21:27-21:30 CST restart correction: the first attempted smallscale
root,
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_smallscales_4096_finalgate`,
was also stopped because its manifest/history proved it did not actually use
the intended small candidate scale set. The tmux launch intended
`CANDIDATE_SCALES=0.05,0.1,0.2`, but the run recorded the default full set
`[0.05,0.1,0.2,0.5,1.0]`. This is a launch/propagation failure, not a model
result.

2026-06-15 21:31 CST active replacement: launched the direct smallscale formal
run
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_smallscales_direct_4096_finalgate`
on Slurm step `128888.80` in tmux session
`cosmos3_candidate_formal_phasecap_smallscales_direct_4096_2gpu_20260615`.
It uses the new direct compute-node wrapper
`scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh`
so `--candidate-scales` is passed explicitly to the trainer. The run manifest
and first training-history row confirm `candidate_scales=0.05,0.1,0.2` and
eval candidate count `53`, with no `scale_0.5` or `scale_1` candidates.
The paired gated watcher is
`cosmos3_candidate_after_gate_watch_smallscales_direct_env_4096_20260615`; it
passes `FORMAL_ROOT` to the compute-node live launcher through `env` rather
than relying on implicit Slurm environment propagation. It will not run live
unless the post-3-hour final checkpoint passes the hardened formal gate. Early
metrics are aligned but not evidence: after the untrained step-1 eval, steps
`500`, `1000`, `1500`, `2000`, `2500`, `3000`, and `3500` all selected
actions below the DP-prior MSE (`0.00156083`), with step `3500` at
`0.00140141`. Candidate sources are now DP, `scale_0.05`, `scale_0.1`,
`scale_0.2`, and a few stochastic samples; `scale_0.5` and `scale_1` are not
available in this run.

2026-06-15 21:35 CST monitoring update: the direct smallscale run remains
active on Slurm step `128888.80`; no final summary/checkpoint exists yet.
The latest 12 validation evals through step `11000` were all below the frozen
DP-prior action MSE. Representative points: step `5500` selected MSE
`0.00139013`, step `10000` `0.00144018`, step `11000` `0.00148719`, versus
DP `0.00156083`. This is still early offline training evidence only, but it
shows the immediate `scale_1` selector-drift failure has been removed by the
small candidate-scale trust region. Spot GPU utilization was `31%/26%`.

2026-06-15 21:37 CST code update: the candidate-executor trainer and live
selector now treat `candidate_samples=0` as "no stochastic samples." This is
not used by the currently running direct smallscale formal run, which already
loaded with `CANDIDATE_SAMPLES=48`. It is a prepared fallback if the current
run later shows sustained stochastic-sample drift: the next aligned run can
evaluate only DP plus the small deterministic residual scales while preserving
the same Cosmos task/contact conditioning and final live gate.

2026-06-15 21:38 CST update: the direct smallscale run with stochastic samples
was stopped inside tmux before the formal floor because stochastic candidates
became the new selector-drift source. It successfully removed `scale_1`, but
by step `18500` the latest 12 validation evals contained 6 points above the
DP prior, including step `15500` selected MSE `0.00172562` versus DP
`0.00156083`. This run is not formal evidence and must not launch live.

2026-06-15 21:39 CST active replacement: launched
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_smallscales_nosample_direct_4096_finalgate`
on Slurm step `128888.84` in tmux session
`cosmos3_candidate_formal_phasecap_smallscales_nosample_direct_4096_2gpu_20260615`.
It uses `candidate_samples=0` and `candidate_scales=0.05,0.1,0.2`. The
manifest and first history rows confirm `num_candidate_sources=5`, so the
candidate set is DP prior, generator mean, and the three small residual
scales only. The paired env-safe watcher is
`cosmos3_candidate_after_gate_watch_smallscales_nosample_direct_env_4096_20260615`.
Step `500` selected MSE was `0.00130318` versus DP `0.00156083`; this is
only early offline training evidence.

2026-06-15 21:42 CST monitoring update: the nosample direct run remains active
on Slurm step `128888.84`. It is still pre-formal, with no final summary or
final checkpoint. Through step `7500`, the latest 12 validation evals were all
below the frozen DP-prior action MSE: step `2000` selected `0.00137006`, step
`5500` `0.00137674`, and step `7500` `0.00141868`, versus DP `0.00156083`.
The selector is stable and simple here: `dp_prior:1, scale_0.2:76` on the
validation set, with only 5 candidate sources total. Spot GPU utilization was
`44%/26%`.

2026-06-15 21:45 CST post-gate recorder: added
`scripts/world_model/watch_candidate_executor_post_gate_status.py` and
launched it in tmux session
`cosmos3_candidate_post_gate_status_nosample_20260615` for the active
nosample formal root. It writes `post_gate_status.json` and
`post_gate_status.md` under the formal root. Current status is
`waiting_training_final_artifacts`. This is only artifact tracking; any live
success still requires final-state metrics plus direct contact-sheet/video
inspection.

2026-06-15 21:46 CST monitoring update: nosample formal step `128888.84` is
still active. No final summary/checkpoint exists yet. Latest eval at step
`19500` has selected-action MSE `0.00149078` versus DP prior `0.00156083`,
with source counts `scale_0.05:1, scale_0.1:1, scale_0.2:75` and
`num_candidate_sources=5`. The env-safe live watcher and post-gate status
recorder are both alive. Spot GPU utilization was `36%/42%`.

2026-06-15 21:48 CST monitoring update: nosample formal step `128888.84`
remains active and non-idle. Latest eval at step `24500` has selected-action
MSE `0.00148684` versus DP prior `0.00156083`, with source counts
`scale_0.05:2, scale_0.2:75` and `num_candidate_sources=5`. Spot GPU
utilization was `77%/32%`. `post_gate_status.json` still reports
`waiting_training_final_artifacts`; no final summary/checkpoint or live panel
exists yet.

2026-06-15 21:50 CST monitoring update: nosample formal step `128888.84`
remains active. Latest eval at step `29000` has selected-action MSE
`0.00150265` versus DP prior `0.00156083`, with source counts
`scale_0.05:2, scale_0.2:75` and `num_candidate_sources=5`. Spot GPU
utilization was `61%/37%`. No final summary/checkpoint or live panel exists
yet.

2026-06-15 21:55 CST diffusion-candidate readiness update: the current active
formal run is the no-stochastic-sample small-scale Gaussian candidate executor
at
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_smallscales_nosample_direct_4096_finalgate`.
It is still pre-floor and has no final summary/checkpoint or live panel.
Latest spot check at step `60000` has selected-action MSE `0.00151335` versus
DP prior `0.00156083`, with selected sources `scale_0.05:1`,
`scale_0.1:1`, and `scale_0.2:75`.

While that run continues, the next DDP-style implementation gap was repaired:
`scripts/world_model/train_cosmos3_candidate_executor.py` now supports
`--generator-type diffusion`, training a denoising residual action generator
under the same causal Cosmos task/contact feature. The live receding loop now
loads diffusion candidate-executor checkpoints separately from the old
Gaussian checkpoint type and samples `diffusion_*` action candidates for the
same progress/contact/value scorer. The direct and generic candidate training
wrappers pass `GENERATOR_TYPE`, `DIFFUSION_STEPS`,
`DIFFUSION_BETA_START`, `DIFFUSION_BETA_END`, and
`DIFFUSION_LOSS_WEIGHT`.

Boundary: the default remains `GENERATOR_TYPE=gaussian`, so this does not
reinterpret the currently running no-sample checkpoint. No diffusion executor
training has been run yet. This is readiness for the next aligned attempt if
the current conservative candidate gate or live panel fails.

Verification:

- `.venv/bin/python -m py_compile scripts/world_model/train_cosmos3_candidate_executor.py scripts/world_model/run_cosmos3_live_receding_loop.py`
- `bash -n scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_train_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`
- `git diff --check -- scripts/world_model/train_cosmos3_candidate_executor.py scripts/world_model/run_cosmos3_live_receding_loop.py scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_train_in_allocation.sh`

2026-06-15 22:03 CST monitoring update: the active no-sample formal run is
still running on Slurm step `128888.84` at about `24` minutes, so it is still
far before the formal 3-hour floor. Latest eval at step `68500` has
selected-action MSE `0.00152335` versus DP prior `0.00156083`, with selected
sources `scale_0.05:1`, `scale_0.1:1`, and `scale_0.2:75`.
`post_gate_status.json` still reports `waiting_training_final_artifacts`; no
final summary/checkpoint or live panel exists yet.

2026-06-15 22:06 CST diffusion self-test update: added
`scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py` as a
login-safe check for the new diffusion candidate path. It runs entirely on CPU
with toy arrays: diffusion candidate eval produces `7` sources
(`dp_prior`, `mean`, two small scales, and three diffusion candidates),
Gaussian/no-sample eval remains at `4` sources, and the live receding loop can
load a toy checkpoint whose args say `generator_type=diffusion`.
The same test now checks that live checkpoint loading preserves
`phase_residual_l2_caps`, which keeps the offline residual-cap selector and
live residual-cap selector aligned.

Verification passed:

- `.venv/bin/python -m py_compile scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py scripts/world_model/train_cosmos3_candidate_executor.py scripts/world_model/run_cosmos3_live_receding_loop.py`
- `.venv/bin/python scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py`
- `bash -n scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_train_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`

Boundary: this confirms the code path, not controller performance. No
diffusion executor training or live panel has been run yet; the active
evidence chain remains the no-sample Gaussian formal run until its final gate.

2026-06-15 22:08 CST monitoring update: no-sample formal step `128888.84`
remains active at about `29` minutes. Latest eval at step `82500` has
selected-action MSE `0.00151605` versus DP prior `0.00156083`, with selected
sources `scale_0.05:1`, `scale_0.1:2`, and `scale_0.2:74`.
`post_gate_status.json` still reports `waiting_training_final_artifacts`; no
final summary/checkpoint or live panel exists yet.

2026-06-15 22:09 CST monitoring update: no-sample formal step `128888.84`
remains active at about `30` minutes. Latest eval at step `86000` has
selected-action MSE `0.00155048` versus DP prior `0.00156083`, with selected
sources `mean:1`, `scale_0.05:1`, and `scale_0.2:75`. This is still below
DP-prior action MSE, but the margin is small, so the final gate remains the
authority. An overlapping `nvidia-smi` probe showed GPU utilization
`60%/69%`; the run is not idle. No final summary/checkpoint or live panel
exists yet.

2026-06-15 22:10 CST diffusion smoke wrapper update: added
`scripts/slurm/run_cosmos3_candidate_executor_diffusion_smoke_in_allocation.sh`.
It is compute-node-only and delegates to the direct candidate-executor trainer
with `GENERATOR_TYPE=diffusion`, `CANDIDATE_SAMPLES=8`,
`CANDIDATE_SCALES=0.05,0.1,0.2`, `MAX_STEPS=100`, and
`MIN_WALL_SECONDS=0`. Boundary: this is a future 50-100 step smoke launcher,
not formal evidence and not a live-eval launcher.

Verification passed:

- `bash -n scripts/slurm/run_cosmos3_candidate_executor_diffusion_smoke_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_train_in_allocation.sh`
- `.venv/bin/python scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py`
- `git diff --check -- scripts/slurm/run_cosmos3_candidate_executor_diffusion_smoke_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_train_in_allocation.sh scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py scripts/world_model/train_cosmos3_candidate_executor.py scripts/world_model/run_cosmos3_live_receding_loop.py`

2026-06-15 22:12 CST monitoring update: no-sample formal step `128888.84`
remains active at about `33` minutes. Latest eval at step `94500` has
selected-action MSE `0.00153442` versus DP prior `0.00156083`. However, the
latest 10 evals already include three selected-action points worse than DP:
step `91000`, `93500`, and `94000`. This is not post-floor failure evidence
yet, but it means the final gate risk is real. The aligned action is to keep
the formal run until the required floor and let the hardened final gate decide
whether live eval is allowed.

2026-06-15 22:13 CST self-test hardening: extended
`scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py` so it
now verifies both checkpoint families. The toy diffusion checkpoint loads in
the live loop as `generator_type=diffusion`; the toy Gaussian/no-sample
checkpoint loads as `generator_type=gaussian` with `candidate_samples=0`.
Both paths preserve `phase_residual_l2_caps`, so the live selector can use the
same residual-cap barrier as offline evaluation.

Verification passed:

- `.venv/bin/python -m py_compile scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py scripts/world_model/train_cosmos3_candidate_executor.py scripts/world_model/run_cosmos3_live_receding_loop.py`
- `.venv/bin/python scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py`
- `bash -n scripts/slurm/run_cosmos3_candidate_executor_diffusion_smoke_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_train_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`
- `git diff --check -- scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py scripts/world_model/train_cosmos3_candidate_executor.py scripts/world_model/run_cosmos3_live_receding_loop.py scripts/slurm/run_cosmos3_candidate_executor_diffusion_smoke_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_train_in_allocation.sh`

2026-06-15 22:14 CST monitoring update: no-sample formal step `128888.84`
remains active at about `35` minutes. Latest eval at step `98500` is slightly
worse than DP prior: selected-action MSE `0.00156419` versus DP
`0.00156083`. This reinforces the final-gate risk, but it is still not a
post-floor formal failure. Keep the run to the formal floor; the hardened
final gate must refuse live if the final selected-action MSE is still worse
than DP.

2026-06-15 22:22 CST update: the active no-stochastic-sample direct run
`experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_smallscales_nosample_direct_4096_finalgate`
is still running as Slurm step `128888.84` on `server33`. It is still
pre-floor at about `43` minutes, so it is not formal failure evidence.
However, the recent validation pattern now supports the same selector-drift
diagnosis: step `120500` has selected-action MSE `0.00164551` versus the
frozen-DP prior `0.00156083` (`+8.47e-05`). The last checked window had many
points above DP despite removing stochastic samples and large residual scales.
This means the conservative Gaussian candidate family still does not reliably
produce a final checkpoint that is safer than the DP prior.

This failure mode is exactly why the next aligned path is the DDP-style
diffusion candidate action generator, not another hand-tuned residual scale:
the executor needs to generate and score multiple short contact chunks under
the Cosmos task/contact path, while the DP prior remains a fallback and
regularizer. The diffusion code path is now wired and login-safe tested:

- the direct compute-node trainer wrapper explicitly passes
  `--generator-type`, `--diffusion-steps`, `--diffusion-beta-*`,
  `--diffusion-loss-weight`, `--candidate-samples`, and
  `--candidate-scales` into
  `scripts/world_model/train_cosmos3_candidate_executor.py`;
- `scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py`
  verifies both diffusion and Gaussian checkpoint loading in the live loop,
  including `phase_residual_l2_caps`;
- checks passed: `bash -n`, `py_compile`, the CPU diffusion/gaussian
  self-test, and `git diff --check`.

To avoid manual delay after the current formal gate, a lightweight watcher was
added and launched:
`scripts/slurm/watch_candidate_executor_gate_then_diffusion_smoke.sh`, tmux
session `cosmos3_candidate_diffusion_after_gate_watch_20260615`. It only polls
status on the login node. If the current no-sample formal run reaches the
2GPU/3h floor and the final gate refuses live, it waits for allocation
`128888` to have no non-extern Slurm step, then launches the compute-node-only
diffusion smoke wrapper with `GENERATOR_TYPE=diffusion`,
`CANDIDATE_SAMPLES=8`, and `CANDIDATE_SCALES=0.05,0.1,0.2`. If the final
gate passes, it exits and leaves closed-loop evaluation to the existing gated
live watcher.

2026-06-15 22:27 CST watcher hardening: the same watcher was upgraded and
restarted so that a passing diffusion smoke does not leave the held GPUs idle.
The active gate chain is now:

```text
current no-sample formal gate fails after the 2GPU/3h floor
  -> diffusion 50-100 step smoke
  -> if smoke ready_for_offline_gate=true and selected MSE <= DP MSE:
       launch full 2GPU/10800s formal diffusion candidate-executor training
  -> if that formal final gate passes:
       launch the gated candidate-executor live panel
  -> if any gate fails:
       stop and record the reason
```

This preserves the user's training standard: the smoke is only a code/path
and short-overfit gate, not formal method evidence. The formal diffusion run
still uses `NPROC_PER_NODE=2`, `MIN_WALL_SECONDS=10800`, final-checkpoint-only
offline gating, and then real closed-loop evaluation only after the final gate
passes. The tmux pane for
`cosmos3_candidate_diffusion_after_gate_watch_20260615` confirms the upgraded
boundary text is running.

2026-06-15 22:31 CST scorer-training repair: the current no-sample Gaussian
formal run is still pre-floor, but its late validation trend has stayed worse
than DP. At step `150000`, selected-action MSE was `0.00163977` versus the
frozen-DP prior `0.00156083`. This is not formal failure evidence yet, but it
re-confirms the concrete selector-drift failure: the scorer can rank a
candidate above DP even when that candidate is worse under the held-out action
gate.

The next diffusion/candidate trainer was repaired for this specific training
mismatch. Before this change, the scorer was trained only on the teacher
residual chunk, but during eval/live it had to rank DP-prior, generator mean,
scaled residuals, and stochastic or diffusion candidates. That is a train-test
objective mismatch for the scorer. `scripts/world_model/train_cosmos3_candidate_executor.py`
now adds a generic candidate-ranking calibration loss: each batch builds
DP-prior, generator-mean, small-scale, and random residual candidates, scores
them with the same progress/contact/value expression used by offline/live
selection, and applies a cross-entropy target toward the candidate closest to
the teacher residual. This does not make a method-success claim and does not
replace the real closed-loop gate; it makes the scorer learn the same kind of
candidate-selection task it will perform at inference.

Wrappers now pass and record:

- `CANDIDATE_RANK_LOSS_WEIGHT=0.35`
- `CANDIDATE_RANK_RANDOM_COUNT=4`
- `CANDIDATE_RANK_TEMPERATURE=1.0`

This affects future diffusion smoke/formal runs launched by the watcher. It
does not reinterpret the already-running no-sample checkpoint, which was
started before this loss existed.

Verification passed:

- `.venv/bin/python -m py_compile scripts/world_model/train_cosmos3_candidate_executor.py scripts/world_model/run_cosmos3_live_receding_loop.py scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py`
- `.venv/bin/python scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py`
- `bash -n scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_diffusion_smoke_in_allocation.sh scripts/slurm/watch_candidate_executor_gate_then_diffusion_smoke.sh scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`
- `git diff --check -- scripts/world_model/train_cosmos3_candidate_executor.py scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_diffusion_smoke_in_allocation.sh scripts/slurm/watch_candidate_executor_gate_then_diffusion_smoke.sh`

2026-06-15 22:35 CST watcher restart hygiene: the after-gate watcher was
changed from timestamped diffusion output roots to fixed rank-calibrated roots:

- smoke:
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_diffusion_smoke_after_nosample_gate_rankcal_20260615`
- formal diffusion:
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_diffusion_rankcal_after_nosample_gate`

This prevents a watcher restart from creating duplicate smoke/formal runs and
wasting the held allocation. The watcher was restarted in tmux session
`cosmos3_candidate_diffusion_after_gate_watch_20260615`; the temporary
foreground debug watcher was stopped, leaving one active watcher process.

2026-06-15 22:38 CST summary-audit update: future candidate-executor
`training_summary.json` files now record the key diffusion/rank-calibration
configuration at top level:

- `generator_type`
- `diffusion_steps`
- `candidate_samples`
- `candidate_scales`
- `candidate_rank_loss_weight`
- `candidate_rank_random_count`
- `candidate_rank_temperature`

This makes the post-gate diffusion smoke/formal evidence auditable without
loading the checkpoint or relying only on wrapper logs. Verification passed:
`py_compile`, the diffusion/gaussian CPU self-test, wrapper `bash -n`, and
`git diff --check`.

2026-06-15 22:40 CST monitoring update: the active no-sample Gaussian formal
step `128888.84` has reached about `61` minutes, still before the 2GPU/3h
formal floor. It has not produced `training_summary.json` or
`checkpoint_final.pt`, and `post_gate_status.json` remains
`waiting_training_final_artifacts`. The latest checked 16 validation points
are all worse than the frozen DP prior; the latest point, step `172500`, has
selected-action MSE `0.00163471` versus DP `0.00156083`. This is strong
pre-floor evidence of the old selector drift, but it is not formal failure
until the final post-floor checkpoint gate.

The fixed diffusion smoke/formal roots are still empty, so the after-gate
diffusion watcher has not prematurely launched compute. The old gated-live
watcher and the new diffusion watcher can both wake up after the current
summary appears. This is acceptable: the old live launcher independently
refuses if the current final gate fails, and the diffusion watcher waits for
non-extern Slurm steps to clear before launching the rank-calibrated diffusion
smoke.

2026-06-15 22:42 CST monitoring update: the same no-sample formal step is at
about `63` minutes and remains pre-floor. The latest 24 validation evals are
all worse than the frozen DP prior; latest step `178500` has selected-action
MSE `0.00160406` versus DP `0.00156083`, with recent selected-minus-DP deltas
from `+1.57e-05` to `+1.54e-04`. This still cannot be called formal failure
before `training_summary.json` and `checkpoint_final.pt`, but it reinforces
that the old Gaussian/no-rank-calibration selector is likely to fail the final
gate. A light overlap GPU spot check on `server33` showed utilization
`26%/39%` and `4459 MiB` memory on both GPUs, so the held allocation is still
doing training work and should not be interrupted.

2026-06-15 22:44 CST live-audit update: the future candidate-executor live
path now records the rank-calibrated diffusion configuration in each
`candidate_executor_action_chunk.json`. The live checkpoint loader preserves
`candidate_rank_loss_weight`, `candidate_rank_random_count`, and
`candidate_rank_temperature`; the action-chunk JSON records those fields plus
`candidate_samples`, `generator_type`, `diffusion_steps`, and
`candidate_scales`. This does not alter action selection, but it makes later
closed-loop videos and per-iteration action logs auditable for whether the
rank-calibrated diffusion executor was actually used. Verification passed:
`py_compile`, the diffusion/gaussian CPU self-test, wrapper `bash -n`, and
`git diff --check`.

2026-06-15 22:48 CST chain-status update: added
`scripts/world_model/summarize_candidate_executor_diffusion_chain.py` to
summarize the current no-sample formal root, the fixed diffusion-smoke root,
the fixed formal-diffusion root, and the diffusion watcher log. It writes
`diffusion_chain_status.json` and `diffusion_chain_status.md` under the
current formal root. The current overall status is
`waiting_current_formal_gate`: the current no-sample root still lacks final
summary/checkpoint, the diffusion smoke/formal roots are empty, and the
watcher log has not launched smoke/formal/live. This is observability only;
method evidence still requires final closed-loop metrics and inspected video.

The same check found the current no-sample step still pre-floor at about
`69` minutes. Latest step `195000` has selected-action MSE `0.00161908`
versus DP `0.00156083`, so the old Gaussian/no-rank-calibration selector
remains worse than DP, but final gate evidence is still pending.

2026-06-15 23:00 CST chain-status repair: the diffusion/candidate chain
summarizer now reads the latest `training_history.json` entry when final
`training_summary.json` is still absent and embeds `post_gate_status.json`.
This makes `diffusion_chain_status.json/.md` show the current selected-vs-DP
trend and gate status while the formal no-sample run is still pre-floor. The
lightweight watcher tmux session
`cosmos3_candidate_diffusion_after_gate_watch_20260615` was restarted to load
the current watcher script; the active training step `128888.84` was not
interrupted.

Current status after the restart and one automatic watcher refresh:

- Slurm step `128888.84` remains active on `server33` at about `84` minutes.
- `post_gate_status.json` is still `waiting_training_final_artifacts`.
- There is still no `training_summary.json` or `checkpoint_final.pt`.
- `diffusion_chain_status.json` reports `waiting_current_formal_gate`.
- Latest recorded validation point: step `236000`, selected-action MSE
  `0.00163644` versus frozen-DP MSE `0.00156083`.
- Fixed diffusion smoke/formal roots remain empty, so no post-gate diffusion
  compute has launched.

Verification passed:

- `.venv/bin/python -m py_compile scripts/world_model/summarize_candidate_executor_diffusion_chain.py`
- `bash -n scripts/slurm/watch_candidate_executor_gate_then_diffusion_smoke.sh`
- `git diff --check -- scripts/world_model/summarize_candidate_executor_diffusion_chain.py scripts/slurm/watch_candidate_executor_gate_then_diffusion_smoke.sh`

2026-06-15 23:08 CST diffusion rank-alignment repair: one remaining mismatch
was found before the post-gate diffusion smoke starts. The diffusion executor
would evaluate/live-select real denoised diffusion candidates, but the
candidate-rank calibration loss only showed the scorer DP-prior, generator
mean, scaled mean residuals, and random Gaussian residual candidates. That is
weaker than the DDP-style contract where the action scorer/policy should be
trained on the same imagined/candidate action space it uses at inference.

The trainer now supports `candidate_rank_diffusion_count`. When
`generator_type=diffusion`, the rank-calibration loss can include real
denoised diffusion candidates, detached from the denoising path, so the scorer
learns to compare actual diffusion chunks against DP/mean/scale candidates
without turning the rank loss into an expensive full denoiser objective.
Watcher-launched smoke/formal diffusion runs pass
`CANDIDATE_RANK_DIFFUSION_COUNT=1`. Live checkpoint loading and
`candidate_executor_action_chunk.json` metadata preserve the field, and the
chain summarizer records it when summaries exist.

This is not an error-recovery case split. It is a generic scorer-alignment fix
for the candidate/diffusion executor objective:

```text
Cosmos task/contact imagination -> diffusion/candidate action chunks
  -> progress/contact/value scorer trained on the same candidate family
  -> short execution -> real re-observation
```

Verification passed:

- `.venv/bin/python -m py_compile scripts/world_model/train_cosmos3_candidate_executor.py scripts/world_model/run_cosmos3_live_receding_loop.py scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py scripts/world_model/summarize_candidate_executor_diffusion_chain.py`
- `bash -n scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_diffusion_smoke_in_allocation.sh scripts/slurm/watch_candidate_executor_gate_then_diffusion_smoke.sh scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`
- `.venv/bin/python scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py`
- `git diff --check -- scripts/world_model/train_cosmos3_candidate_executor.py scripts/world_model/run_cosmos3_live_receding_loop.py scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py scripts/world_model/summarize_candidate_executor_diffusion_chain.py scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh scripts/slurm/run_cosmos3_candidate_executor_diffusion_smoke_in_allocation.sh scripts/slurm/watch_candidate_executor_gate_then_diffusion_smoke.sh`

The lightweight diffusion watcher was restarted after this repair and the
active no-sample formal training step was not interrupted. Current chain
status after the restart: `waiting_current_formal_gate`; latest recorded
validation point step `254000`, selected-action MSE `0.00170371` versus
frozen-DP MSE `0.00156083`; fixed diffusion smoke/formal roots still empty.

2026-06-15 23:12 CST watcher handoff guard: the diffusion after-gate watcher
now waits `POST_FAIL_SETTLE_SECONDS=75` after a post-floor formal gate failure
before launching diffusion smoke. Plain reason: the older gated-live watcher
may briefly start the live launcher after the current formal summary appears;
if the formal gate fails, that launcher should refuse quickly. The short
settle wait gives that expected refusal path time to clear so the diffusion
smoke does not race it for the same held allocation. This is scheduling
hygiene only and does not change any training, offline gate, live gate, metric,
or success criterion.

The lightweight watcher was restarted after the guard was added; active
training step `128888.84` was not interrupted. Current chain status:
`waiting_current_formal_gate`; latest recorded validation point step `263500`,
selected-action MSE `0.00165816` versus frozen-DP MSE `0.00156083`; fixed
diffusion smoke/formal roots still empty.

2026-06-15 23:13 CST trend-window observability: the chain-status summarizer
now records the latest validation-window selected-vs-DP deltas, not only the
single latest point. Manual refresh at step `267000` showed all `24/24` recent
evals worse than the frozen DP prior. Mean selected-minus-DP MSE was
`+8.54e-05`, with range `+5.25e-05` to `+1.48e-04`. This strengthens the
pre-floor diagnosis that the old no-sample Gaussian selector is drifting, but
it is still not a formal failure until the post-`10800` second summary and
`checkpoint_final.pt` are written.
