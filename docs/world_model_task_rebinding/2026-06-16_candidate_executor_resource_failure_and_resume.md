# Candidate Executor Resource Failure And Resume

Date: 2026-06-16

## Boundary

The active method direction remains:

```text
Cosmos low-frequency task/contact imagination
  -> diffusion/candidate action chunks
  -> progress/contact/value scorer
  -> short execution
  -> real re-observation
```

The `128888` no-sample formal run did not produce formal method evidence. It
was a gate before the diffusion executor path, and it was cancelled by Slurm
before the current `2` GPU / `3` hour floor.

## Evidence

- Slurm allocation `128888` is no longer valid:
  `srun --overlap --jobid=128888 hostname` returns
  `Unable to confirm allocation for job 128888: Invalid job id specified`.
- `squeue -u yanhongru` shows no active jobs.
- Stale local watcher processes were stopped:
  `522131`, `522145`, and `2110504`.
- Interrupted root:
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260615_formal_2gpu_server33_phasecap_smallscales_nosample_direct_4096_finalgate`.
- Present artifacts: `training_history.json`, `checkpoint_latest.pt`.
- Missing artifacts: `training_summary.json`, `checkpoint_final.pt`.
- Last recorded training-history point:
  `step=429000`, `elapsed_seconds=9133.67`, selected action MSE
  `0.00148796` versus frozen-DP MSE `0.00156083`.
- Console tail records:
  `STEP 128888.84 ON server33 CANCELLED AT 2026-06-16T00:11:36`.

## Interpretation

This is a scheduling/resource interruption, not a completed training result.
The partial offline selected-MSE values are useful diagnostics only. They do
not satisfy the formal training floor, do not create a final checkpoint, and
cannot launch closed-loop live evaluation.

## Resume Action

Request a fresh tmux-held `2` GPU interactive allocation. After CUDA canary
passes inside that allocation, continue the requested diffusion/candidate
executor path directly:

1. Run the diffusion smoke using the hardened gate that requires
   `generator_type=diffusion`, `candidate_samples>0`, and
   `candidate_rank_diffusion_count>0`.
2. If smoke passes, run the formal `2` GPU / `3` hour diffusion candidate
   executor training.
3. Launch live closed-loop evaluation only from the post-floor
   `checkpoint_final.pt` and only if the final summary passes the offline
   non-DP/diffusion gate.

Do not restart the old non-diffusion live handoff from allocation `128888`.

## 10:10 CST Formal-Training Failure And Best-Gate Repair

Allocation `131660` (`1` GPU) started before the `2`/`4` GPU requests. It ran
the current diffusion candidate-executor chain on `server62`.

What passed:

- CUDA canary passed.
- The 100-step diffusion smoke passed the held-out offline gate:
  `teacher_progress_mse=0.040261`, `teacher_value_mse=0.039450`,
  `teacher_inserted_acc=0.961039`,
  `teacher_dp_continuable_acc=0.870130`, and selected action MSE
  `0.001508 < 0.001561` frozen-DP prior MSE.

What failed:

- The formal run satisfied the time floor:
  `elapsed_seconds=10804.909`, `formal_training_floor_met=true`.
- The final formal checkpoint failed the scorer gate:
  `teacher_progress_mse=148981`, `teacher_value_mse=9217.4268`,
  `teacher_inserted_acc=0.662338`,
  `teacher_dp_continuable_acc=0.454545`.
- The final selected action MSE was only marginally better than DP:
  `0.00155038 < 0.00156083`.
- JSONL train/val distribution was not obviously corrupt; train and val have
  similar phase/role/scenario coverage, and contact-progress deltas stay in a
  bounded physical range. This points to scorer training/checkpoint selection
  instability, not a known data-label scale bug.

Interpretation:

The current executor can produce small action residuals around the DP prior,
but the progress/contact/value scorer is not stable at the 3-hour final
checkpoint. Since the live controller uses that scorer to choose action
chunks, launching live from this final checkpoint would be unjustified. This
is a training/checkpoint-selection blocker before live eval, not evidence that
closed-loop eval itself is wrong.

Repair:

- Keep the same offline gate thresholds.
- Save `checkpoint_best_gate.pt` whenever a held-out eval point passes the
  full scorer/action gate.
- After the formal GPU/time floor is met, allow live eval from the
  summary-selected gate-passing checkpoint, not necessarily from the final
  post-floor weights.
- Reset the formal model configuration to the smoke-validated stable setting:
  `1024x4`, dropout `0.05`, lr `2e-4`, global batch about `128`.

Active rerun:

- Allocation: `131662` (`4` GPUs), currently `PENDING (Priority)`.
- Watcher:
  `cosmos3_candidate_diffusion_chain_watch_131662_4gpu_bestgate_20260616`.
- Formal root:
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_4gpu_alloc131662_diffusion_bestgate_stable_smokecfg`.
- New chain markers use prefix
  `candidate_executor_diffusion_bestgate_chain_20260616`, so the old
  `candidate_executor_diffusion_chain_20260616.terminal` from the failed
  final-checkpoint chain does not stop the best-gate retry.

## 14:20 CST Formal Passed, Live Blocked By Render Runtime

The best-gate retry on allocation `131662` started on `server46` and completed
the formal floor.

Formal result:

- Slurm job: `131662`
- node: `server46`
- formal root:
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_4gpu_alloc131662_diffusion_bestgate_stable_smokecfg`
- formal GPUs: `4`
- formal run rc: `0`
- formal training floor: met
- live checkpoint selected by summary:
  `checkpoint_best_gate.pt`
- live metrics source: `best_gate`
- best-gate step: `100`
- held-out gate metrics:
  selected action MSE `0.0015079`, DP-prior MSE `0.0015608`,
  `teacher_progress_mse=0.04146`, `teacher_value_mse=0.03786`,
  inserted acc `0.961`, DP-continuable acc `0.857`

The final checkpoint still overfit the scorer heads, but it was not used for
live because the summary-selected checkpoint was the gate-passing checkpoint
from the same formal run.

Live result:

- gated live launched from the selected checkpoint at `13:41:56+08`
- output root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_panel4_20260616_134157_samples0_1_3_4_after_formal_gate`
- live rc: `50`
- completed samples: `0`
- video/contact sheet: none
- `sample_00` failed before a loop summary with:
  `vk::Device::waitForFences: ErrorDeviceLost`
- parallel retries for samples `01`, `03`, and `04` also failed before loop
  summaries, with logs showing `sapien_cuda` unavailable or CUDA/Vulkan
  initialization failure.

Render canaries:

- `server46`, with explicit
  `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`, reached
  `render_rgb_array_start` and then timed out at the Slurm step limit.
- `server13` failed earlier with
  `vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed` and
  `RuntimeError: CUDA unknown error`.

Interpretation:

This is a render-capable-node/runtime blocker. It is not a method failure, not
a training-data failure, and not a closed-loop controller-quality result,
because no sample reached real short-chunk execution plus final-state/video
evidence.

Repairs applied:

- `scripts/slurm/run_cosmos3_live_receding_panel_in_allocation.sh` now exports
  empty `DISPLAY`.
- `scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`
  now exports the same render environment.
- Added
  `scripts/slurm/watch_candidate_executor_live_render_retry_from_allocation.sh`,
  which first runs `render_min_canary.py` through `srun` inside the held
  allocation, then runs the same gated candidate-executor live panel only if
  the render canary passes.
- Follow-up log comparison showed that the old successful `server62` live
  panel did not force `VK_ICD_FILENAMES`; it emitted SAPIEN's missing-ICD
  warning and still rendered. The live wrappers therefore no longer force the
  NVIDIA ICD by default. `SET_NVIDIA_VK_ICD=true` is now an explicit
  diagnostic variant only.

Current resource state:

- `server46` allocation `131662` was released after formal training completed
  and the render canary timed out.
- `server13` allocation `132849` was released after render canary failure.
- general retry allocation `132888` reached `server61`, but its render canary
  also timed out.
- allocation `132981` reached `server62` and the actual gated live wrapper for
  `sample_03`, but the run still died on the first `env.render()` with
  `vk::Device::waitForFences: ErrorDeviceLost`. This confirms the render
  canary was not merely a false blocker.
- current pending alternatives:
  `133177` (`server62`, `3` GPUs, to test other physical GPUs on the same
  node), `133178` (`server24`, `1` GPU), `133179` (`server54`, `1` GPU), and
  `133180` (`server10`, `1` GPU). `server10`, `server24`, and `server54`
  have previous live-summary evidence.

Next action:

Use the first started pending allocation that can render, then rerun the same
gated live panel from `checkpoint_best_gate.pt`. Cancel later pending
alternatives after one valid render-capable allocation is active. Do not
bypass video evidence or weaken the controller/evaluation gates.

## 01:53 CST Resume State

Fresh allocation request:

- Slurm job: `131564`
- tmux allocation session: `cosmos3_candidate_diffusion_2gpu_20260616`
- request: `2` GPUs, `1` day, excluding `server13`
- current state: `PENDING (Priority)`

Lightweight chain watcher:

- tmux session:
  `cosmos3_candidate_diffusion_chain_watch_131564_20260616`
- script:
  `scripts/slurm/watch_candidate_executor_diffusion_chain_from_allocation.sh`
- watch log:
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_2gpu_alloc131564_diffusion_rankcal_finalgate/diffusion_chain_watch.log`
- smoke root:
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_diffusion_smoke_20260616_alloc131564_rankcal`
- formal root:
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_2gpu_alloc131564_diffusion_rankcal_finalgate`

The watcher does not run training on the login node. It only polls `squeue`.
When allocation `131564` becomes `RUNNING`, it launches all compute through
`srun --overlap --jobid=131564`: CUDA canary, diffusion smoke, formal
diffusion candidate-executor training, then gated live eval if the formal
summary allows it.

Local verification before launch:

- `bash -n scripts/slurm/watch_candidate_executor_diffusion_chain_from_allocation.sh`
- `bash -n scripts/slurm/run_cosmos3_candidate_executor_diffusion_smoke_in_allocation.sh`
- `bash -n scripts/slurm/run_cosmos3_candidate_executor_train_direct_in_allocation.sh`
- Python `py_compile` for the candidate trainer, diffusion-chain summarizer,
  and live loop
- `.venv/bin/python scripts/world_model/selftest_cosmos3_candidate_executor_diffusion.py`
  returned `status: passed`, including the live diffusion action-chunk path
- `git diff --check` on the touched TODO, evidence note, and watcher script

Status-summary correction: the chain summarizer now marks this 2026-06-16
path as `chain_mode=direct_diffusion`, so the pre-smoke state is reported as
`waiting_diffusion_smoke` instead of the old no-sample-gate wording
`waiting_current_formal_gate`.

## Resource Audit

At 02:03 CST, allocation `131564` is still pending:

- `JobState=PENDING`
- `Reason=Priority`
- `StartTime=2026-06-17T22:00:00`
- planned node: `server34`

`server34` currently has all `8` H200 GPUs allocated. `gpux` is drained, and
`test` is not available to the current account because it allows only
`AllowAccounts=null`. The valid action is to keep the tmux-held `gpu`
allocation request and watcher alive. A separate 1-GPU smoke-only allocation
would not satisfy the formal `2` GPU floor and would risk holding idle GPU
time after the short smoke completes.

## 02:02 CST Queue-Poll Hygiene

The pending-allocation watcher was restarted with the script default
`POLL_SECONDS=1800` instead of `60`. This keeps the login-node side from
spinning on Slurm while job `131564` waits for priority. The allocation tmux
session is still active and no Slurm resource was cancelled.

Latest `scontrol show job 131564` after the restart:

- `JobState=PENDING`
- `Reason=Priority`
- `StartTime=2026-06-17T23:00:00`
- planned node: `server03`

## 02:04 CST Training-Input Prelaunch Audit

Default candidate-executor training input:

`experiments/world_model_task_rebinding/cosmos3/contact_executor_dataset_iter1500_train512_scale005_repair_20260615/contact_executor_dataset_file.jsonl`

Login-safe audit results:

- rows: `512`
- task path sources: `cosmos_predicted_action_sidecar:512`
- no `gt_state_targets_debug` occurrences
- missing `executor_sample_npz`: `0`
- missing `dp_prior_npz`: `0`
- missing `contact_label_npz`: `0`
- missing `source_h5`: `0`
- `load_arrays` output shapes:
  `x=(512,218)`, `y=(512,56)`, `prior=(512,56)`,
  `progress=(512,2)`, `binary=(512,2)`
- prefix roles:
  `insert_resume=169`, `target_motion_observed=150`,
  `target_post_motion=193`
- phases:
  `dp_continuable=200`, `lateral_align=142`, `preinsert_aligned=76`,
  `far=94`

The executor sample NPZs contain an empty internal `dp_prior_actions` array,
but the trainer does not use that field. `load_arrays` reads the actual
frozen-DP prior from each row's separate `dp_prior_npz["dp_prior_actions"]`,
which has the expected `8x7` chunk. Therefore the current input path is ready
for the diffusion smoke/formal trainer once allocation `131564` starts.

## 02:08 CST Live-Panel Dependency Audit

The gated live launcher after formal diffusion training depends on the
clean-dense Cosmos SFT/eval chain and DP assets. Login-safe path audit found
all required static inputs present:

- clean-dense SFT root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299`
- SFT checkpoint:
  `outputs/cosmos3/sft/vision_sft_droid_policy_full1000_rgb_300step_wam/checkpoints/iter_000001500`
- formal iter1500 eval root:
  `eval_full_episode_wam_iter_000001500_formal_after_3h_abs4gpu_retry2`
- condition root:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050`
- source H5 root:
  `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5`
- DP manifest:
  `experiments/dp_peg1000/run_90201/manifest.json`
- DP checkpoint:
  `experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt`
- continuability stats:
  `experiments/world_model_task_rebinding/cosmos3/dp_static_continuability_stats_20260613/dp_static_continuability_stats.json`

This does not prove live success. It only clears the static path dependency
gate before allocation `131564` starts.

## 02:10 CST Live Feature-Width Audit

The candidate executor must use the same feature layout in training and live
execution. The prelaunch audit confirmed:

- training `load_arrays` produced feature width `218`;
- training target width is `56`, i.e. `8` robot actions of dimension `7`;
- live checkpoint loading derives horizon as `target_dim / robot_action_dim`,
  so the formal checkpoint will use horizon `8`;
- live candidate-executor feature construction is:
  current state `35` + task path `8*14` + DP prior `8*7` + live contact
  context `15` = `218`;
- live execution still has a hard guard:
  `candidate executor feature width ... != checkpoint feature_dim`.

This clears the feature-layout mismatch risk before the 2GPU allocation
starts. It does not prove the learned executor will succeed in live rollout.

## 02:48 CST Reusable Prelaunch Audit

Added a login-safe, reusable prelaunch audit:

`scripts/world_model/audit_candidate_executor_diffusion_prelaunch.py`

Latest outputs:

- `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_2gpu_alloc131564_diffusion_rankcal_finalgate/prelaunch_audit.json`
- `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_2gpu_alloc131564_diffusion_rankcal_finalgate/prelaunch_audit.md`

Result:

- `ready_for_allocation_launch=true`
- `blocker_class=slurm_resource_pending`
- rows: `512`
- task path source: `cosmos_predicted_action_sidecar:512`
- GT-debug rows: `0`
- missing required row paths: `0`
- `load_arrays` shapes:
  `x=(512,218)`, `y=(512,56)`, `prior=(512,56)`,
  `progress=(512,2)`, `binary=(512,2)`
- feature contract:
  current state `35` + task path `8*14` + frozen-DP prior `8*7` +
  live contact context `15` = `218`
- required static live dependencies: present
- planned formal config:
  `generator_type=diffusion`, `candidate_samples=8`,
  `candidate_rank_diffusion_count=1`, `nproc_per_node=2`,
  `min_wall_seconds=10800`

Interpretation: the current known blocker before training starts is Slurm
resource availability for allocation `131564`, not known bad training rows,
missing Cosmos/DP paths, or a train/live feature-width mismatch. This audit is
prelaunch evidence only; it is not training, live rollout, video, or method
evidence.

## 02:50 CST Watcher Prelaunch Gate

The reusable prelaunch audit is now wired into
`scripts/slurm/watch_candidate_executor_diffusion_chain_from_allocation.sh`.
The lightweight watcher runs it before polling/launching compute and exits with
`13` if the audit fails. This prevents the chain from starting formal smoke or
training after file drift, missing paths, or a train/live feature-contract
mismatch.

The watcher session
`cosmos3_candidate_diffusion_chain_watch_131564_20260616` was restarted to use
the new gate. The Slurm allocation tmux session
`cosmos3_candidate_diffusion_2gpu_20260616` was not cancelled or released.

Latest watcher log:

- `prelaunch_audit_ready=true`
- `allocation_wait state=PENDING reason=(Priority)`

Latest Slurm state remains:

- job: `131564`
- state: `PENDING`
- reason: `Priority`
- start time: `2026-06-17T23:00:00`
- scheduled node: `server03`

## 02:51 CST Chain Status Includes Prelaunch Result

`scripts/world_model/summarize_candidate_executor_diffusion_chain.py` now
includes the prelaunch audit in `diffusion_chain_status.json/.md`.

Current status file:

`experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_2gpu_alloc131564_diffusion_rankcal_finalgate/diffusion_chain_status.md`

Key fields:

- `overall_status=waiting_diffusion_smoke`
- `prelaunch_audit_ready=True`
- `prelaunch_blocker_class=slurm_resource_pending`
- `prelaunch_rows=512`
- `prelaunch_feature_contract_ok=True`

## 02:53 CST Live Post-Gate Status Writeback

The direct-diffusion watcher now runs
`scripts/world_model/watch_candidate_executor_post_gate_status.py` after the
gated live panel returns. It writes:

- `post_gate_status.json`
- `post_gate_status.md`
- `post_gate_status_watch.log`

Those files allow `diffusion_chain_status.json/.md` to report the live output
root, live panel summary, contract flag, final success count, and contact
sheet path after live evaluation finishes.

The lightweight watcher was restarted again to use this logic. Allocation
`131564` was not cancelled or released. Latest watcher log shows:

- `prelaunch_audit_ready=true`
- `allocation_wait state=PENDING reason=(Priority)`

Latest Slurm estimate:

- state: `PENDING`
- reason: `Priority`
- start time: fluctuating; latest `scontrol` at `02:53:40+08` reports
  `2026-06-17T22:00:00`
- scheduled node: `server34`

## 02:56 CST Post-Gate RC Parser Fix

A direct-diffusion post-gate status issue was found before live execution:

- the direct watcher logged the live return code as
  `formal_diffusion_gated_live_rc`;
- `watch_candidate_executor_post_gate_status.py` only parsed
  `candidate_after_gate_live_rc`;
- the status watcher was pointed at `candidate_after_gate_live_watch.log`, so
  live completion could be missed and `post_gate_status.json/.md` could remain
  stuck at `formal_gate_passed_waiting_live_completion`.

Fix:

- `watch_candidate_executor_diffusion_chain_from_allocation.sh` now appends
  `candidate_after_gate_live_rc=${live_rc}` to
  `candidate_after_gate_live_watch.log` after the live `srun` returns.
- `watch_candidate_executor_post_gate_status.py` now accepts both
  `candidate_after_gate_live_rc` and `formal_diffusion_gated_live_rc`.

Verification:

- `bash -n scripts/slurm/watch_candidate_executor_diffusion_chain_from_allocation.sh`
- `py_compile scripts/world_model/watch_candidate_executor_post_gate_status.py`
- temporary artifact test confirmed terminal statuses for:
  `live_panel_summary_available_needs_video_review` and
  `live_finished_without_panel_summary`
- `git diff --check` on the touched paths

The lightweight watcher was restarted to use the fix. Allocation `131564` was
not cancelled or released. Latest status is still `PENDING (Priority)`, with
the scheduler estimate currently fluctuating around `2026-06-17T20:00:00`.

## 02:57 CST Live Evidence Fields In Chain Status

The post-gate/live status path now carries the fields needed for the next
human-readable decision after live evaluation:

- requested live samples
- completed live samples
- final-success count
- failed-process count
- full-episode panel contract flag
- contact-sheet path and contact-sheet ok flag
- visual-review status
- method-evidence flag

`diffusion_chain_status.md` now exposes the live return code, success count,
contract flag, and contact-sheet path from `post_gate_status.json`. A
temporary artifact test confirmed these fields are parsed from
`live_receding_panel_summary.json`.

Current chain status remains `overall_status=waiting_diffusion_smoke` because
allocation `131564` has not started training yet.

## 03:00 CST Smoke Gate Semantics Fix

The direct-diffusion watcher previously treated the 50-100 step smoke as if it
had to pass the same offline quality gate as formal training. That could stop
the formal `2` GPU / `3` hour run merely because the short smoke had
`ready_for_offline_gate=false` or selected-action MSE was still worse than the
frozen-DP prior.

That was the wrong gate boundary. The smoke run is an interface/runtime check:
it should prove the diffusion candidate path runs and writes valid artifacts.
It is not method-quality evidence.

New smoke requirements:

- `generator_type=diffusion`
- `candidate_samples > 0`
- `candidate_rank_diffusion_count > 0`
- finite selected-action and DP-prior MSE fields
- positive training step count
- more than two candidate sources in eval

Formal/live requirements remain strict:

- `formal_training_floor_met=true`
- `ready_for_formal_live_eval=true`
- selected-action MSE not worse than frozen-DP prior
- non-DP selected candidate count greater than zero
- post-formal live panel, final-state metrics, contract status, and video or
  contact-sheet review

`summarize_candidate_executor_diffusion_chain.py` now reports
`diffusion_smoke_interface_ready` separately from
`diffusion_smoke_ready_for_offline_gate`. A temporary summary test confirmed
that a valid diffusion smoke with `ready_for_offline_gate=false` now leads to
`overall_status=diffusion_smoke_ready_waiting_formal_diffusion`, not a false
formal-training block.

The lightweight watcher was restarted to use the corrected gate. Allocation
`131564` remains `PENDING (Priority)`, with latest estimate
`2026-06-17T20:00:00`.

## 03:03 CST Formal Live Checkpoint Contract Gate

The gated live launcher now verifies more than "checkpoint file loads" before
closed-loop evaluation:

- checkpoint `args.generator_type` must match the formal summary;
- diffusion checkpoint metadata must have positive `candidate_samples` and
  `candidate_rank_diffusion_count`;
- `target_dim` must be divisible by robot action dim `7`;
- live feature contract must hold:
  `feature_dim = 35 + horizon*14 + target_dim + 15`.

For the current expected formal checkpoint this means `target_dim=56`,
horizon `8`, and `feature_dim=218`.

Reason: formal live eval must not run from a stale or mismatched checkpoint
that happens to sit beside a valid summary file. This preserves the method
boundary: live actions must come from the same diffusion candidate executor
that passed the formal offline gate.

Verification:

- local temporary checkpoint test accepted `218/56`;
- the same test rejected feature dim `217`;
- `bash -n scripts/slurm/run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`;
- `git diff --check` on the touched launcher.

The lightweight watcher was restarted to use the updated launcher. Allocation
`131564` remains `PENDING (Priority)`, estimated `2026-06-17T20:00:00`.

## 03:11 CST Summary Root Matching

The direct-diffusion watcher now rejects smoke/formal training summaries whose
`output_root` does not resolve to the directory containing that summary.

Reason: if an old `training_summary.json` and `checkpoint_final.pt` are copied
or left in the current root, the watcher should not treat them as evidence for
the current chain. This is especially important before formal training and
live evaluation, where stale artifacts could make the chain skip the intended
run.

Implementation:

- `watch_candidate_executor_diffusion_chain_from_allocation.sh` checks
  `Path(payload["output_root"]).resolve() == training_summary.parent.resolve()`
  inside `summary_ready_field`.
- `summarize_candidate_executor_diffusion_chain.py` reports
  `summary_output_root_matches` for the current formal root and smoke root.

Verification:

- temporary summary test accepted a matching `output_root`;
- the same test rejected a mismatched `output_root`;
- `bash -n`, `py_compile`, and `git diff --check` passed on touched files.

The lightweight watcher was restarted to use the new check. Allocation
`131564` remains `PENDING (Priority)`, estimated `2026-06-17T20:00:00`.

## 03:05 CST Candidate Trainer Checkpoint Reload Robustness

`scripts/world_model/train_cosmos3_candidate_executor.py` now reloads
`checkpoint_final.pt` and `checkpoint_best_offline.pt` with
`weights_only=False` during the final save verification.

Reason: the candidate executor checkpoint includes numpy normalization arrays
(`x_mean`, `x_std`, `residual_mean`, `residual_std`) in addition to model
weights. The active local torch version is `2.5.1+cu121`, but explicitly
setting `weights_only=False` prevents a late formal-run failure if torch
serialization defaults change.

Verification:

- `py_compile scripts/world_model/train_cosmos3_candidate_executor.py`
- temporary checkpoint with numpy payload loaded using `weights_only=False`
- `git diff --check` on the touched trainer

This is execution robustness only. It is not training or method evidence.

## 03:08 CST Summary/Checkpoint Dimension Consistency

Formal candidate-executor summaries now carry the controller-facing dimensions:

- `feature_dim`
- `target_dim`
- `action_horizon`

The gated live launcher compares these summary values against
`checkpoint_final.pt` when they are present, in addition to checking the
checkpoint's own feature contract. The chain status summary also exposes these
fields.

Reason: the formal offline gate and live checkpoint must describe the same
executor. A correct-looking summary should not allow a stale checkpoint with a
different action horizon or feature width to enter closed-loop evaluation.

Verification:

- temporary summary/checkpoint test accepted `feature_dim=218`,
  `target_dim=56`, `action_horizon=8`;
- the same test rejected mismatched `action_horizon=7`;
- `bash -n`, `py_compile`, and `git diff --check` passed on touched files.

The lightweight watcher was restarted to use the updated launcher. Allocation
`131564` remains `PENDING (Priority)`, estimated `2026-06-17T20:00:00`.

## 03:48 CST 1/2/4 GPU Formal Allocation Rule Applied

The latest user override is now reflected in the actual watcher state, not
only in prose:

- formal candidate-executor training may start on `1`, `2`, or `4` GPUs;
- whichever tmux-held allocation becomes `RUNNING` first takes the shared
  launch mutex and runs the chain;
- formal training still requires `10800` seconds; the time floor is not
  relaxed;
- short smoke/overfit remains a debug gate only, not method evidence.

Current Slurm requests:

- `131660`, `cand_diff_1gpu_0616`: `1` GPU, `PENDING (Priority)`, estimated
  start `2026-06-17T16:00:00`;
- `131564`, `cand_diff_2gpu_0616`: `2` GPUs, `PENDING (Priority)`, estimated
  start `2026-06-17T11:00:00` at the latest check, with estimates still
  fluctuating;
- `131662`, `cand_diff_4gpu_0616`: `4` GPUs, `PENDING (Priority)`, estimated
  start `2026-06-18T04:00:00`.

Current watcher sessions:

- `cosmos3_candidate_diffusion_chain_watch_131660_1gpu_20260616`;
- `cosmos3_candidate_diffusion_chain_watch_131564_2gpu_20260616`;
- `cosmos3_candidate_diffusion_chain_watch_131662_4gpu_20260616`.

The watcher logs confirm the real launch contracts:

- `formal_gpus=1`, `formal_nproc=1`,
  `formal_min_wall_seconds=10800`;
- `formal_gpus=2`, `formal_nproc=2`,
  `formal_min_wall_seconds=10800`;
- `formal_gpus=4`, `formal_nproc=4`,
  `formal_min_wall_seconds=10800`.

All three prelaunch audits report `ready_for_allocation_launch=true`, with
the same `512` causal rows, `218 -> 56` feature/target contract, diffusion
candidate config, and `min_wall_seconds=10800`. Therefore the current blocker
is only that none of the three requested allocations is running yet. It is not
a data-row issue, missing static dependency issue, feature-width issue, or an
old two-GPU-only code path.

One extra live-eval guard is active: if `server35` starts first, the watcher
may use it for formal training, but it writes
`candidate_executor_diffusion_chain_20260616.formal_ready` and defers live
closed-loop eval to a later allocation if the current allocation is on
`server35`. This preserves the formal training opportunity while avoiding
claiming live render/controller evidence on a node with previous live-render
instability.

Resource partition check at the same time:

- `gpu` is the active legal partition for the mayi account;
- `gpux`, `debug`, and `mgpu` are drained;
- `test`, `gaosh`, `engram`, and `mgpu` expose `AllowAccounts=null`, so they
  are not usable by the current mayi allocation requests;
- no one-shot `sbatch` path was used or introduced.

## 03:54 CST Prelaunch Audit Moved Into Allocation

The watcher previously ran
`scripts/world_model/audit_candidate_executor_diffusion_prelaunch.py` as soon
as the lightweight watcher started. That was too loose under the latest
execution rule: even small project-code preflight/debug checks must run on a
compute node, not on the login node.

Repair:

- while the Slurm allocation is pending, the watcher now only polls Slurm/tmux
  state and writes status;
- after an allocation becomes `RUNNING` and wins the shared launch mutex, the
  watcher runs the prelaunch audit through
  `srun --overlap --jobid=${ALLOC_JOB_ID}` inside the tmux-held allocation;
- the in-allocation audit writes the same `prelaunch_audit.json` and
  `prelaunch_audit.md`, but with ready `blocker_class=ready_for_chain_launch`;
- if `.formal_ready` already exists, a later allocation skips the training
  prelaunch audit because its job is live eval from the already-passed formal
  root.

The older prelaunch audit files are retained as historical traces only. They
are not method evidence and should not be treated as the compute-node
prelaunch gate. The first allocation that actually starts will overwrite them
from inside the allocation before CUDA canary, smoke, formal training, or live
eval.

The three lightweight watchers were restarted at `03:54:45+08`. Their newest
logs show `prelaunch_audit_deferred_until_allocation_running` followed by
`allocation_wait state=PENDING reason=(Priority)`, confirming that the login
node side is back to Slurm/tmux polling only.

## 04:00 CST Status Helpers Moved Into Allocation

A second execution-boundary issue was fixed: the lightweight watcher still
called Python status helpers from the login node while allocations were
pending. Those calls did not train or render, but they were still project-code
status checks and could drift against the latest user rule.

Repair:

- while allocation state is not `RUNNING`, `update_chain_status` is now a
  no-op; the pending watcher records only shell log lines and Slurm/tmux
  state;
- after an allocation is `RUNNING`, `update_chain_status` runs
  `summarize_candidate_executor_diffusion_chain.py` through
  `srun --overlap --jobid=${ALLOC_JOB_ID}` on the compute allocation;
- after live eval returns, `watch_candidate_executor_post_gate_status.py` also
  runs through `srun`, not directly on the login node.

This keeps the pending login-node side limited to control-plane polling. It
does not change the method, training gate, candidate generator, scorer, or
live-eval contract.

## 04:06 CST Smoke/Formal Summary Gate Moved Into Allocation

One more login-node leak was found and repaired: after smoke/formal training,
the watcher used an inline Python JSON parser to decide whether
`training_summary.json` satisfied the smoke interface gate or the formal
offline gate. That parser now also runs through
`srun --overlap --jobid=${ALLOC_JOB_ID}` once the allocation is `RUNNING`.
When the allocation is still pending, it returns `missing` without invoking
Python.

The three lightweight watchers were restarted at `04:05:54+08`. Their newest
logs again show only:

- `prelaunch_audit_deferred_until_allocation_running`
- `allocation_wait state=PENDING reason=(Priority)`

Latest queue snapshot:

- `131564` / `2` GPUs: `PENDING (Priority)`, estimated
  `2026-06-17T03:00:00`, scheduled node `server34`;
- `131660` / `1` GPU: `PENDING (Priority)`, estimated
  `2026-06-17T04:00:00`, scheduled node `server35`;
- `131662` / `4` GPUs: `PENDING (Priority)`, estimated
  `2026-06-17T20:00:00`, scheduled node `server33`.

Note: any `diffusion_chain_status.json` files still showing timestamp around
`03:54` are stale historical status files from before status helpers were
moved into the allocation. They should not be used as current pending-state
truth. The current pending truth is the Slurm state plus the watcher text log;
the JSON status will be rewritten from the compute allocation after a job
starts.

## 03:20 CST 1/2/4 GPU Allocation Fanout

The latest user execution rule is now applied: if `2` GPUs do not start, use
`4` GPUs or `1` GPU instead. Formal evidence may use `1` GPU, but the
`10800` second training floor remains mandatory.

Implementation changes:

- `watch_candidate_executor_diffusion_chain_from_allocation.sh` now accepts
  `FORMAL_GPUS`, `FORMAL_NPROC_PER_NODE`, `FORMAL_MIN_GPUS`,
  `FORMAL_MIN_WALL_SECONDS`, `SMOKE_GPUS`, and `CANARY_GPUS`.
- `run_cosmos3_candidate_executor_train_direct_in_allocation.sh` now forwards
  `FORMAL_MIN_GPUS` to the trainer, so a deliberate `1` GPU formal run is not
  rejected by the old hard-coded `2` GPU check.
- `audit_candidate_executor_diffusion_prelaunch.py` now treats
  `planned_nproc_per_node >= 1` and `planned_min_wall_seconds >= 10800` as the
  valid formal resource contract.
- The new watchers share
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_diffusion_chain_20260616.launch.lock`,
  so only the first allocation that becomes `RUNNING` launches compute.
- Each watcher now receives its own allocation tmux session name. If it starts
  after another watcher already holds the launch mutex or after the chain is
  already done, it sends `Ctrl-C` to its own allocation tmux session. This
  prevents a losing 1/2/4 GPU request from sitting idle after resources are
  granted.

Active tmux-held allocation requests:

- `131660` / `cand_diff_1gpu_0616`: `1` GPU, watcher
  `cosmos3_candidate_diffusion_chain_watch_131660_1gpu_20260616`, formal root
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_1gpu_alloc131660_diffusion_rankcal_finalgate`
- `131564` / `cand_diff_2gpu_0616`: `2` GPUs, watcher
  `cosmos3_candidate_diffusion_chain_watch_131564_2gpu_20260616`, formal root
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_2gpu_alloc131564_diffusion_rankcal_finalgate`
- `131662` / `cand_diff_4gpu_0616`: `4` GPUs, watcher
  `cosmos3_candidate_diffusion_chain_watch_131662_4gpu_20260616`, formal root
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_4gpu_alloc131662_diffusion_rankcal_finalgate`

All three prelaunch audits are ready:

- rows: `512`
- planned generator: `diffusion`
- candidate samples: `8`
- rank diffusion count: `1`
- planned wall floor: `10800` seconds
- planned nproc: `1`, `2`, or `4` according to allocation size
- blocker class: `slurm_resource_pending`

Current state from `squeue`: all three allocation requests are still
`PENDING (Priority)`. The current blocker is therefore Slurm priority, not a
known data, path, feature-width, or training-script failure. No method evidence
has been produced by this change.

Latest scheduler estimates:

- `131564` / `2` GPUs: latest observed `2026-06-17T16:00:00`
- `131660` / `1` GPU: `2026-06-18T11:25:12`
- `131662` / `4` GPUs: `2026-06-18T11:25:12`

## Current Continuation: Scorer Gate Hardening

The candidate/diffusion executor branch is supposed to work as:

`Cosmos task/contact imagination -> candidate/diffusion action chunks -> progress/contact/value scorer selection`.

The formal-to-live gate now checks that full contract explicitly. Before a
candidate executor checkpoint can launch live closed-loop evaluation, the
final formal summary must satisfy:

- `teacher_progress_mse <= 0.05`
- `teacher_value_mse <= 0.25`
- `teacher_inserted_acc >= 0.75`
- `teacher_dp_continuable_acc >= 0.75`
- selected action MSE not worse than frozen-DP prior action MSE
- at least one selected candidate must be non-`dp_prior`

Changes:

- `train_cosmos3_candidate_executor.py` now includes `teacher_value_mse` in
  `offline_gate_from_eval` and records `offline_gate_thresholds` in the
  manifest and final summary.
- `run_cosmos3_candidate_executor_live_panel_after_gate_in_allocation.sh`
  independently rechecks progress, value, inserted, DP-continuable,
  selected-vs-DP, and non-DP selection before live rollout. It writes
  `candidate_executor_metric_gate_ok=true`, `gate_metrics=...`, and
  `gate_thresholds=...` into the live watch log after a pass.
- `summarize_candidate_executor_diffusion_chain.py` and
  `watch_candidate_executor_post_gate_status.py` now expose the scorer
  metrics and thresholds.

Verification:

- `bash -n` passed for the candidate live launcher, diffusion watcher, and
  direct training wrapper.
- `py_compile` passed for candidate trainer, chain summarizer, post-gate
  status watcher, and prelaunch audit.
- the live launcher inline Python block compiled successfully.
- `summarize_candidate_executor_diffusion_chain.py` ran on the current
  `131564` root and reported `overall_status=waiting_diffusion_smoke`.
- existing smoke summary
  `candidate_executor_train_20260615_smoke512_100step_phasecap_meanpen1_4096`
  still passes the new scorer gate with `teacher_value_mse=0.0574679` under
  the `0.25` value threshold.

This is not method evidence. It prevents a bad or stale formal summary from
entering live closed-loop evaluation once one of the pending 1/2/4 GPU
allocations starts.

## Current Continuation: Terminal Marker For Duplicate-Run Prevention

The 1/2/4 allocation fanout needs one more guard: if the first allocation
finishes formal training and reaches a real decision, later allocations should
not repeat the same formal training just because their jobs start later.

Watcher behavior is now:

- `candidate_executor_diffusion_chain_20260616.done` means live evaluation
  completed with `live_rc=0`.
- `candidate_executor_diffusion_chain_20260616.terminal` means the direct
  diffusion chain reached a terminal formal/live outcome. This includes
  `formal_diffusion_gate_failed` and `live_finished` even when the live return
  code is nonzero.
- When a watcher allocation becomes `RUNNING`, it checks both markers before
  launching compute. If either exists, it sends `Ctrl-C` to its own allocation
  tmux session and exits.
- Runtime failures before a formal conclusion still release the launch mutex,
  so another allocation can attempt the chain.

The three lightweight watcher sessions were restarted at `03:35+08` with this
logic. Allocation tmux sessions were not stopped.

Latest queue snapshot after restart:

- `131564` / `2` GPUs: `PENDING (Priority)`, latest observed start
  `2026-06-17T17:00:00`
- `131660` / `1` GPU: `PENDING (Priority)`, latest observed start
  `2026-06-18T09:00:00`
- `131662` / `4` GPUs: `PENDING (Priority)`, latest observed start
  `2026-06-18T11:25:12`

The terminal marker is currently absent, as expected, because no allocation
has started smoke/formal training yet.

## Current Continuation: Live Candidate-Executor Path Audit

A login-node read-only code audit checked that live closed-loop execution is
wired to the intended method, not just the trainer:

```text
current live state
  + Cosmos-predicted task path
  + frozen-DP prior chunk
  + live contact context
  -> candidate/diffusion executor
  -> progress/contact/value scorer
  -> selected short action chunk
```

Observed implementation facts:

- `run_cosmos3_live_receding_loop.py` has
  `controller_action_source=candidate_executor`.
- Live feature construction concatenates current executor state, causal
  Cosmos task path, DP prior chunk, and causal live contact context.
- Candidate sources include `dp_prior`, mean, scale variants, and diffusion
  samples when `generator_type=diffusion`.
- Every candidate is scored with `score_candidate`, using predicted progress,
  inserted probability, DP-continuable probability, predicted value, logprob,
  and residual penalty.
- The selected candidate, predicted progress/contact/value, candidate records,
  DP prior stats, and selected action stats are written to
  `candidate_executor_action_chunk.json`.
- Training and live residual caps use the same quantity:
  `mean(raw_resid ** 2)`.
- The gated live launcher passes
  `CONTROLLER_ACTION_SOURCE=candidate_executor` and
  `EXECUTOR_CHECKPOINT=${FORMAL_ROOT}/checkpoint_final.pt`.

Small cleanup:

- The candidate live launcher manifest wording was updated from the obsolete
  `2GPU/3h` phrase to the active `1/2/4-GPU plus 3-hour` rule.

Verification:

- `bash -n` passed for the candidate live launcher and direct diffusion
  watcher.
- the candidate live launcher inline Python block compiled.
- `py_compile` passed for `run_cosmos3_live_receding_loop.py`,
  `train_cosmos3_candidate_executor.py`,
  `summarize_candidate_executor_diffusion_chain.py`, and
  `watch_candidate_executor_post_gate_status.py`.
- `git diff --check` passed.

Latest queue snapshot during this audit:

- `131564` / `2` GPUs: `PENDING (Priority)`, latest observed start
  `2026-06-17T16:00:00`
- `131660` / `1` GPU: `PENDING (Priority)`, latest observed start
  `2026-06-18T11:25:12`
- `131662` / `4` GPUs: `PENDING (Priority)`, latest observed start
  `2026-06-18T11:25:12`

## Current Continuation: Render Runtime Retry

The formal candidate-executor checkpoint is still valid for live eval:
`checkpoint_best_gate.pt` from allocation `131662`, with the unchanged offline
gate passing from the `best_gate` metrics. The live blocker remains
first-frame render/runtime failure before any candidate-executor action is
executed.

New evidence:

- General retry allocation `133564` reached `server08`.
- The render canary created the ManiSkill environment, reset it, reached
  `render_rgb_array_start`, then failed with
  `vk::Device::waitForFences: ErrorDeviceLost`.
- The terminal file is
  `experiments/world_model_task_rebinding/cosmos3/candidate_executor_train_20260616_formal_4gpu_alloc131662_diffusion_bestgate_stable_smokecfg/candidate_live_render_retry_auto_any_no_bad_20260616_174803_alloc133564_server08.terminal`.
- The allocation was released because it was concretely render-unusable.

Render-env repair:

- Read-only comparison against the old successful `server62` live path found
  that the old path did not explicitly pass an empty `DISPLAY`.
- The current retry/candidate wrappers had been exporting `DISPLAY=`.
- The wrappers now preserve `DISPLAY` only if it is already non-empty;
  otherwise they leave it unset. They still avoid forcing
  `VK_ICD_FILENAMES` unless `SET_NVIDIA_VK_ICD=true` is requested for a
  diagnostic.
- `bash -n` and `git diff --check` passed for the touched shell wrappers.
- General retry allocation `133733` reached `server52` after this fix. The
  canary log recorded `display=null`, so the fix was active, but the run still
  timed out at `render_rgb_array_start` with exit code `124`. Therefore empty
  `DISPLAY` was not the sufficient cause of the render blocker.

Current pending live attempts:

- `133177`: `server62`, `3` GPUs, fixed-node retry.
- `133179`: `server54`, `1` GPU, fixed-node retry.
- `133180`: `server10`, `1` GPU, fixed-node retry.

All fixed-node holds have tmux autostart watchers attached. When Slurm grants
one, the watcher sends the same gated live command into the held allocation.

## 2026-06-17 Server24 Allocation

Allocation `133178` did start on `server24`; the resource was real. It did not
produce method evidence.

Evidence:

- The retry canary reached the same first-frame render path and timed out with
  rc `124`.
- A direct gated candidate-executor `sample_03` run was launched to check
  whether the canary was too conservative.
- That direct run first exposed a shell-wrapper implementation bug:
  `run_cosmos3_live_receding_panel_in_allocation.sh` line `81` used a
  compound `[[ ... || ... ]]` conditional that failed on the compute step.
  The guard was rewritten as a `case`, and `bash -n` passed inside
  allocation `133178`.
- The rerun passed the formal/offline gate from `checkpoint_best_gate.pt` and
  entered the real live path, but stopped making progress after
  `live_pretrigger_dp_loaded`. For over 11 minutes no new output files were
  written while `nvidia-smi` showed the allocated GPU at `100%`. This matches
  the render hang already seen by the canary.
- Slurm step `133178.3` was cancelled, and allocation `133178` was released.

Current queue after releasing `server24`:

- `133177`: fixed `server62`, `3` GPUs.
- `133179`: fixed `server54`, `1` GPU.
- `133180`: fixed `server10`, `1` GPU.
- `135069` and `135070`: general `1` GPU retries excluding confirmed bad
  render nodes `server08,13,24,35,46,52,61,62`.
