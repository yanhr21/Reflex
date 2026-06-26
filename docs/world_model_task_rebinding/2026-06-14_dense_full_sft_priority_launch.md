# 2026-06-14 Dense Full SFT Priority Launch

## User Priority

The current highest priority is to run full Cosmos3 SFT from the already
generated dense 733 late299 condition root, then continue to closed-loop eval.
The minimum formal training standard is now 2 GPUs for at least 3 hours; use
4 GPUs only if already available sooner. The remaining live-query coverage gap
is recorded as a limitation, not a launch blocker for this run.

## Dense Input

- condition root:
  `experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050`
- preflight summary:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_preflight_20260614_130050/clean_dense_preflight_summary.json`
- structural status: `301` RGB/state frames, `300` actions, strict preflight
  passed, role/mode mismatch `0`.
- known limitation: live-query coverage still misses `58/173` queries.

## Gate Change

The old SFT-entry validator rejected this root because `ready_for_overfit=false`
was caused by the two live-query coverage checks:

- `live_query_coverage_undercovered_count`
- `live_query_coverage_undercovered_fraction`

The validator now has an explicit full-SFT override path that accepts only this
coverage-gap failure mode. It still requires the structural safe checks to be
present and passing, including the condition manifest, full-episode preflight,
receding audit, role/mode cleanliness, weighted manifest, and coverage audit
condition-root match. Other failures still block SFT.

Changed files:

- `scripts/world_model/check_cosmos3_clean_dense_preflight_summary_ready.py`
- `scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_full_fix1recipe_in_allocation.sh`
- `scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh`
- `scripts/slurm/watch_cosmos3_clean_dense_late299_iter300_eval_in_allocation.sh`
- `scripts/slurm/watch_cosmos3_clean_dense_late299_iter300_readout_in_allocation.sh`
- `scripts/slurm/watch_cosmos3_clean_dense_late299_iter300_live_receding_panel_in_allocation.sh`
- `scripts/slurm/run_cosmos3_clean_dense_late299_iter300_eval_readout_live_pipeline_in_allocation.sh`
- `scripts/slurm/run_cosmos3_clean_dense_late299_formal_eval_readout_live_pipeline_in_allocation.sh`
- `scripts/slurm/run_cosmos3_clean_dense_late299_iter300_then_formal_pipeline_in_allocation.sh`
- `scripts/world_model/selftest_cosmos3_clean_dense_preflight_summary_ready.py`

## Resource Status

The previous allocation `127723` was revoked after the failed-state recovery
teacher attempts. A new tmux-held allocation request was submitted from
`cosmos3_clean_dense_4gpu_20260614`:

```text
salloc --partition=gpu --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=32 --mem=220G --time=2-00:00:00 --job-name=cosmos3_dense_full_sft_4gpu_0614
```

Slurm allocation request:

- job: `127817`
- state: `PENDING`
- reason: `Priority`
- submit time: `2026-06-14T17:01:38+08:00`
- requested resources: `4` GPUs, `32` CPUs, `220G`, `2-00:00:00`

Allocation `127817` then started on `server13`. The SFT entry gates passed, but
the actual training launch could not use CUDA. Evidence:

- `sft_train.log` contains repeated PyTorch CUDA initialization warnings:
  `Setting the available devices to be zero`.
- compute-node canary in `.venv_cosmos313`: `CUDA_VISIBLE_DEVICES=0,1,2,3`,
  `nvidia-smi` saw four H200s, but `torch.cuda.is_available()` was `False`.
- compute-node canary in `.venv`: same CUDA failure.
- the foreground SFT step was stopped with `Ctrl-C`; no checkpoint or usable
  training evidence was produced on `server13`.

The bad allocation was released, and a new tmux-held 4-GPU request excluding
`server13` was submitted:

- job: `127819`
- state: `PENDING`
- reason: `Priority`
- submit time: `2026-06-14T17:15:09+08:00`
- requested resources: `4` GPUs, `32` CPUs, `220G`, `2-00:00:00`
- excluded node: `server13`
- queue forecast at `2026-06-14T17:21:56+08:00`: start
  `2026-06-15T00:28:32+08:00` on `server54`

Latest resource override: do not wait for 4 GPUs if a valid non-`server13`
2-GPU allocation can start first. A 2-GPU full SFT is valid method training
only if it reserves/runs for at least 3 hours and passes the same structural,
CUDA, checkpoint, generated-video, and closed-loop eval gates.

Execution update at `2026-06-14T17:51:14+08:00`:

- 2-GPU allocation `127821` started on `server24`. CUDA canary passed
  (`torch.cuda.is_available()=True`, `device_count=2`, H200 visible).
- The 2-GPU SFT entry checks passed and `torchrun --nproc_per_node=2` started,
  but it stayed in model/config I/O with only `4 MiB` per GPU and no training
  step after more than eight minutes. Because the 4-GPU request became
  available first as a usable formal run, the 2-GPU step was stopped with
  `Ctrl-C` and the allocation was released.
- 4-GPU allocation `127819` started on `server35`. CUDA canary passed
  (`torch.cuda.is_available()=True`, `device_count=4`, H200 visible).
- The first active formal run was:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_priority_full_sft_max299`.
- This 4-GPU run has passed strict `301/300` preflight, SFT JSONL audit, action
  target audit, receding-distribution audit, and the clean-dense readiness gate.
  `torchrun --nproc_per_node=4` started at `2026-06-14T17:48:46+08:00`.
  It then failed before training because the wrapper passed relative JSONL
  paths after changing directory into `external/cosmos-framework`, causing
  Cosmos to look for
  `external/cosmos-framework/experiments/.../video_action_dataset_file_role_weighted.jsonl`.
  This is an implementation path bug, not a dense data failure.
- The wrapper was fixed to convert the condition root, output root, preflight
  summary, train JSONL, and val JSONL to absolute paths before launching
  Cosmos. Syntax check passed with `bash -n`.
- The active formal run after the path fix is:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299`.
  It has passed the same strict gates, started `torchrun --nproc_per_node=4`,
  successfully loaded absolute train/val JSONL metadata, completed dataloader
  prewarm, and reached about `10-11GB` GPU memory per H200 at
  `2026-06-14T17:57:19+08:00`. Checkpoint loading completed at
  `2026-06-14T17:57:46+08:00`, validation iteration `0` reported loss
  `3.585211`, training reached at least iteration `11` by
  `2026-06-14T18:04:47+08:00`, and a GPU probe at
  `2026-06-14T18:04:03+08:00` showed all four GPUs at `100%` utilization with
  about `59GB` memory used each. It is not method evidence until checkpoints,
  generated validation artifacts, and closed-loop eval exist.
- Follow-up status at `2026-06-14T18:07:44+08:00`: training was still active
  in Slurm job `127819`, step `127819.7`, and the log had reached iteration
  `21` by `2026-06-14T18:07:38+08:00`. Rank-0 loss fell from `3.9498` at
  iteration `1` to `1.5358` at iteration `21`; all four H200s were at `100%`
  utilization with about `59-60GB` memory used each. No checkpoint had been
  written yet; checkpoint/eval evidence is still pending.
- Follow-up status at `2026-06-14T18:17:44+08:00`: training reached iteration
  `56`, with rank-0 loss `0.5252` at iteration `56`. A separate tmux-held
  1-GPU eval allocation was granted as Slurm job `127825` on `server24`.
  The watcher
  `scripts/slurm/watch_cosmos3_clean_dense_late299_iter300_eval_in_allocation.sh`
  is running in compute step `127825.0`, passed CUDA canary
  (`torch=2.10.0+cu128`, `cuda_available=True`, `device_count=1`,
  `device0=NVIDIA H200`), and is waiting for checkpoint
  `iter_000000300`. Eval output root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/eval_full_episode_wam_iter_000000300`.
  No existing `best_model.pt` task-state readout checkpoint was found under
  the active Cosmos3 experiment roots by `rg --files`; generated-video/action
  eval should proceed first, then a current readout checkpoint must be located
  or trained from
  `experiments/world_model_task_rebinding/cosmos3/sft_dataset_fix3_v7_user_override_733_rgb_512_20260612/manifest.json`
  before generated-RGB readout diagnostics are claimed.
- Follow-up status at `2026-06-14T18:24:47+08:00`: the eval-only watcher
  step `127825.0` was stopped without releasing the held allocation because it
  would have required manual follow-up after generated eval. The replacement
  sequential pipeline
  `scripts/slurm/run_cosmos3_clean_dense_late299_iter300_eval_readout_live_pipeline_in_allocation.sh`
  is running as compute step `127825.1` on `server24`. It waits for
  `iter_000000300`, then runs strict generated full-episode eval, generated-RGB
  readout diagnostics, and the corrected live-receding closed-loop panel in
  sequence. Training job `127819` continued on `server35`; at
  `2026-06-14T18:24:47+08:00` it had reached iteration `81`, rank-0 loss
  `0.3485`, and all four H200s were at `94-100%` utilization.
  Evidence boundary: the `iter_000000300` pipeline is only an early
  interface/visual diagnostic, not formal method evidence, because the
  checkpoint is expected before the active training run satisfies the `2` GPU /
  `3` hour minimum. A formal closed-loop claim needs a checkpoint produced
  after that training-time floor, plus strict generated-artifact inspection,
  readout diagnostics, live-receding final-state metrics, and inspected
  video/contact-sheet evidence.
- Formal checkpoint boundary: the SFT log reports `Starting training...` at
  `2026-06-14T17:57:46+08:00`, so the earliest acceptable formal-training
  wall-clock boundary is `2026-06-14T20:57:46+08:00`. Use the first checkpoint
  whose actual save timestamp is after that boundary for formal closed-loop
  evidence; if `iter_000000600` is saved before the boundary, do not treat it
  as formal and use a later checkpoint such as `iter_000000900`.
- Follow-up status at `2026-06-14T18:31:38+08:00`: the eval allocation was
  switched to the master pipeline
  `scripts/slurm/run_cosmos3_clean_dense_late299_iter300_then_formal_pipeline_in_allocation.sh`
  in compute step `127825.2`. This runs the `iter_000000300` diagnostic
  pipeline first, then automatically waits for a checkpoint whose `.metadata`
  mtime is after `2026-06-14T20:57:46+08:00` and runs the same
  generated/readout/live-receding evidence chain for the formal checkpoint.
  Training job `127819` was still active at iteration `105`, rank-0 loss
  `0.3017`.
- Follow-up status at `2026-06-14T18:33:00+08:00`: training job `127819`
  remained active on `server35`, reached iteration `110`, and rank-0 loss was
  `0.2271`. Eval job `127825`, step `127825.2`, remained active on `server24`
  and was still waiting for `iter_000000300`. No checkpoint, generated videos,
  readout outputs, live-receding videos, or contact sheets existed yet for this
  run, so there was no visual artifact to inspect at that time.
- Follow-up status at `2026-06-14T18:35:58+08:00`: the updated formal
  training floor is confirmed as `2` GPUs for at least `3` hours in
  `AGENTS.md`, the active TODO, and the focused plan. The active run is still
  using the already-available `4` GPU allocation because it is valid and faster
  than waiting for another 2-GPU slot. Training job `127819` remained active on
  `server35`, reached iteration `119` by `2026-06-14T18:35:32+08:00`, and
  rank-0 loss was `0.2512`. A GPU probe showed the four H200s at `100%`,
  `98%`, `96%`, and `100%` utilization with about `59GB` memory used each.
  Eval job `127825`, step `127825.2`, remained active on `server24` and was
  still waiting for `iter_000000300`. No checkpoint or visual artifact existed
  yet.
- Follow-up status at `2026-06-14T18:38:23+08:00`: training reached iteration
  `128`, rank-0 loss was `0.2528`, and all four training H200s were at `100%`
  utilization with about `59GB` memory used each. The eval allocation was alive
  but idle at `0%` GPU because it was still waiting for the first checkpoint.
  Read-only inspection of the master/eval/formal wrapper chain found no current
  checkpoint-path mix-up: generated eval consumes `CHECKPOINT_PATH`, and the
  formal stage selects a checkpoint saved after `2026-06-14T20:57:46+08:00`.
  No generated videos, live-receding videos, or contact sheets existed yet.
- Follow-up status at `2026-06-14T18:40:04+08:00`: training reached iteration
  `134`, rank-0 loss was `0.3035`, and no checkpoint directory or generated
  video artifact existed yet. The saved Cosmos config confirms
  `checkpoint.save_iter: 300`, `trainer.max_iter: 1500`, and
  `trainer.validation_iter: 300`, so the eval pipeline is waiting for the
  intended first checkpoint.
- Follow-up status at `2026-06-14T18:43:06+08:00`: training job `127819`
  remained active on `server35`, reached iteration `144`, and rank-0 loss was
  `0.1699`. A GPU probe showed all four training H200s at `100%` utilization
  with about `59GB` memory used each. Eval job `127825`, step `127825.2`,
  remained active on `server24` and was still waiting for `iter_000000300`.
  No checkpoint, generated video, live-receding video, or contact sheet existed
  yet.
- Follow-up status at `2026-06-14T18:44:02+08:00`: training continued to
  iteration `148`, rank-0 loss was `0.2491`, and both Slurm steps
  `127819.7` and `127825.2` remained active. The checkpoint directory was still
  empty, so no generated eval or visual evidence existed yet.
- Follow-up status at `2026-06-14T18:45:20+08:00`: allocation limits were
  checked. Training job `127819` is reserved until
  `2026-06-16T17:43:33`; eval job `127825` is reserved until
  `2026-06-15T18:12:33`. Training GPUs remained at `100%` utilization; eval
  was still idle only because the checkpoint had not been saved.
- Follow-up status at `2026-06-14T18:46:34+08:00`: training reached iteration
  `157`, rank-0 loss was `0.1316`, and all four training H200s were still at
  `100%` utilization. The checkpoint directory remained empty, so eval was
  still correctly waiting for `iter_000000300`; no visual artifacts existed.
- Follow-up status at `2026-06-14T18:47:32+08:00`: training reached iteration
  `161`, rank-0 loss was `0.1408`, and both Slurm steps `127819.7` and
  `127825.2` remained active. Training GPUs were still at `100%` utilization.
  No checkpoint or generated visual artifact existed yet.
- Follow-up status at `2026-06-14T18:49:05+08:00`: the formal training
  resource rule is `2` GPUs for at least `3` hours; the active run is staying
  on `4` GPUs only because that allocation is already live and faster. Training
  job `127819` remained active on `server35`, step `127819.7`, and reached
  iteration `166` by `2026-06-14T18:48:55+08:00`; rank-0 loss was `0.2013`.
  Eval job `127825`, step `127825.2`, remained active on `server24` and was
  still waiting for `iter_000000300` (`checkpoint_not_ready` through
  `1020` seconds). No checkpoint, generated video, readout output,
  live-receding video, or contact sheet existed yet.
- Follow-up status at `2026-06-14T19:40:32+08:00`: early diagnostic checkpoint
  `iter_000000300` was saved at `2026-06-14T19:27:15+08:00`. The eval master
  pipeline waited for stable checkpoint files, then ran generated full-episode
  eval on `10` validation samples. Strict generated-artifact inspection passed:
  `strict_eval_artifacts_ok=true`, `0` strict failures, `301` video frames per
  sample, and action shape `[300, 32]`. Mean diagnostics were action RMSE
  `0.4346`, robot future RMSE `0.7215`, state-sidecar future RMSE `0.4517`,
  and future-video PSNR `20.34`.
- Manual visual review was performed by opening all `10` generated-vs-reference
  sheets under
  `eval_full_episode_wam_iter_000000300/review_sheets`. The generated videos
  are nonblank, same-viewpoint, and frame-aligned. Visual conclusion: several
  insert/continuous-insert samples are qualitatively close to the reference,
  but pre-motion/static/peg-recovery samples still show visible final peg/TCP
  pose errors. This is a working generated-eval diagnostic, not closed-loop
  success and not formal method evidence.
- After generated-artifact inspection passed, the pipeline advanced to the
  generated-RGB task-state readout diagnostic and launched
  `scripts/world_model/train_cosmos3_task_state_readout.py` under
  `task_state_readout_fix3_v7_733_rgb_301f`. Readout metrics and live-receding
  panel artifacts are still pending.

- Follow-up status at `2026-06-14T19:52-20:01+08:00`: the readout diagnostic
  stayed in H5/model-data loading with no useful GPU work and produced no
  metrics, so it was interrupted inside the held eval allocation. The
  allocation `127825` was preserved. This is a readout/eval implementation or
  IO blocker, not a dense-data or SFT-training failure.
- The iter300 live-receding panel was launched next in the same eval
  allocation as step `127825.27`, output root
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_full300_panel_iter_000000300_clean_dense_20260614_195251`.
  It passed the strict generated-eval input check and entered sample
  `00_hole_late_move_stop`. Partial evidence after two 8-step Cosmos chunks:
  target motion was detected at frame `106`, Cosmos produced action chunks and
  observed-prefix artifacts, but the real peg-head-in-hole-frame `y` error
  worsened from about `0.030m` to `0.063m`; success remained false and `C_pi`
  did not allow DP handoff. This is early iter300 closed-loop diagnostic
  failure evidence only, not a formal post-3-hour method result.
- Completed iter300 live-receding diagnostic sample `sample_00_hole_late_move_stop`.
  The run made `25` Cosmos receding queries over the full `301` observed
  frames, with `WM_ACTIVE=186` frames and `DP_HANDOFF=8` frames. Final
  real-state success was false; final peg-head-in-hole-frame was
  `[-0.1067, 0.0102, -0.0502]`. The final annotated video passed file
  inspection (`301` frames, `30fps`). A contact sheet was generated and opened:
  `live_receding_full300_panel_iter_000000300_clean_dense_20260614_195251/sample_00_hole_late_move_stop/live_observed_rollout_annotated_sheet.png`.
  Visual review agrees with the metric: the peg is visibly not inserted at the
  end. This is an early checkpoint closed-loop failure, not formal method
  evidence.
- Completed iter300 live-receding diagnostic sample `sample_01_hole_late_constant`.
  The run made `27` completed iterations over `301` observed frames, with
  `WM_ACTIVE=36` frames and `DP_HANDOFF=170` frames. Final real-state success
  was true; final peg-head-in-hole-frame was `[0.0274, 0.0023, -0.0010]`.
  A contact sheet was generated and opened:
  `live_receding_full300_panel_iter_000000300_clean_dense_20260614_195251/sample_01_hole_late_constant/live_observed_rollout_annotated_sheet.png`.
  Visual review agrees with the metric: WM brings the run to DP handoff and
  the DP segment keeps the peg at the hole. This is an early diagnostic
  positive sample only, not formal evidence.
- Completed iter300 live-receding diagnostic sample `sample_03_hole_late_fast_shift`.
  The run reached the full `301` observed frames. Final real-state success was
  false; final peg-head-in-hole-frame was `[-0.1465, -0.0190, 0.0064]`.
  A contact sheet was generated and opened:
  `live_receding_full300_panel_iter_000000300_clean_dense_20260614_195251/sample_03_hole_late_fast_shift/live_observed_rollout_annotated_sheet.png`.
  Visual review agrees with the metric: the peg remains outside and offset from
  the hole at the end. This is an early checkpoint closed-loop failure, not
  formal post-3-hour method evidence.
- Completed iter300 live-receding diagnostic sample `sample_04_hole_late_sine`.
  The run made `23` completed iterations over the full `301` observed frames.
  Final real-state success was false; final peg-head-in-hole-frame was
  `[-0.1084, -0.0209, -0.0019]`.
  A contact sheet was generated and opened:
  `live_receding_full300_panel_iter_000000300_clean_dense_20260614_195251/sample_04_hole_late_sine/live_observed_rollout_annotated_sheet.png`.
  Visual review agrees with the metric: the peg approaches the hole but remains
  outside/offset at the end. This is an early checkpoint closed-loop failure,
  not formal post-3-hour method evidence.
- The iter300 live-receding diagnostic panel completed all `4` requested
  samples with `panel_full_episode_contract_ok=true`,
  `sample_contract_failures=[]`, `failed_process_count=0`,
  `final_success_count=1`, and `method_evidence_allowed=false`. This panel is
  useful failure/implementation evidence only: one early positive sample and
  three visually confirmed failures.
- The formal post-3-hour generated-eval plus live-receding pipeline was started
  in the held eval allocation at `2026-06-14T22:09:11+08:00`, Slurm job
  `127825`, step `59`, node `server24`, using
  `scripts/slurm/run_cosmos3_clean_dense_late299_formal_eval_live_pipeline_in_allocation.sh`.
  It is waiting for the first checkpoint whose model `.metadata` mtime is
  after `2026-06-14T20:57:46+08:00`; launch status was
  `formal_checkpoint_not_ready`.
- The formal pipeline selected checkpoint `iter_000000900`; its model
  `.metadata` mtime is `2026-06-14T22:25:29+08:00`, after the formal boundary.
  Formal generated eval started at `2026-06-14T22:29:12+08:00` under
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/eval_full_episode_wam_iter_000000900_formal_after_3h`.
- Formal `iter_000000900` generated eval completed at
  `2026-06-14T22:34:54+08:00`: `10/10` samples inspected,
  `strict_eval_artifacts_ok=true`, `strict_failures=[]`, video frames `301`,
  action shape `[300, 32]`. Mean diagnostics: action RMSE `0.4523`,
  robot-action future RMSE `0.7967`, state-sidecar future RMSE `0.4393`, and
  future video PSNR `22.5858`. Review sheets were opened through
  `review_sheets/combined_iter900_ref_pred_sheet.png` and representative
  original sheets (`00`, `03`, `07`, `08`). Visual review: generated videos are
  nonblank and frame-aligned, but several samples show visible end-effector/peg
  offsets, so this is valid generated-eval evidence only, not closed-loop task
  success.
- Formal `iter_000000900` live-receding closed-loop panel started at
  `2026-06-14T22:34:54+08:00`, output root
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_full300_panel_iter_000000900_clean_dense_20260614_223454`.
  Early partial status for `sample_00_hole_late_move_stop` at frame `123` is
  still failure: peg-head-in-hole-frame `[-0.2273, 0.0672, 0.0033]` and the
  handoff/continuability gate is false. This is only a partial status; wait for
  full final-state/video evidence.
- Formal `iter_000000900` live-receding sample
  `sample_00_hole_late_move_stop` completed: `25` iterations, `301` observed
  frames, `final_success=false`, final peg-head-in-hole-frame
  `[-0.0173, -0.0029, -0.0017]`. A contact sheet was generated and opened:
  `live_receding_full300_panel_iter_000000900_clean_dense_20260614_223454/sample_00_hole_late_move_stop/live_observed_rollout_annotated_sheet.png`.
  Visual review: the run improves substantially over the iter300 same-sample
  failure and reaches a near-handoff/near-insertion state, but authoritative
  real final-state success is still false. Record it as a near miss, not a
  closed-loop success.
- Formal `iter_000000900` live-receding sample
  `sample_01_hole_late_constant` completed: `26` iterations, `301` observed
  frames, `WM_ACTIVE=38`, `DP_HANDOFF=168`, `final_success=false`, final
  peg-head-in-hole-frame `[-0.0198, 0.0026, 0.0025]`. A contact sheet was
  generated and opened:
  `live_receding_full300_panel_iter_000000900_clean_dense_20260614_223454/sample_01_hole_late_constant/live_observed_rollout_annotated_sheet.png`.
  Visual review: the run reaches the hole region but remains a real-state
  failure. Unlike iter300, this sample is not a success at `iter900`. Formal
  closed-loop status after two samples is `0/2` success, with both failures
  close to the threshold but still failures.
- Formal `iter_000000900` live-receding sample
  `sample_03_hole_late_fast_shift` completed: `21` iterations, `301` observed
  frames, `final_success=false`, final peg-head-in-hole-frame
  `[-0.1349, -0.0934, -0.0580]`. A contact sheet was generated and opened:
  `live_receding_full300_panel_iter_000000900_clean_dense_20260614_223454/sample_03_hole_late_fast_shift/live_observed_rollout_annotated_sheet.png`.
  Visual review agrees with the metric: the peg remains outside and offset
  from the hole. Formal closed-loop status after three samples is `0/3`
  success.
- Formal `iter_000000900` live-receding sample `sample_04_hole_late_sine`
  completed: `23` iterations, `301` observed frames, `final_success=false`,
  final peg-head-in-hole-frame `[-0.3264, 0.1223, -0.0559]`. A contact sheet
  was generated and opened:
  `live_receding_full300_panel_iter_000000900_clean_dense_20260614_223454/sample_04_hole_late_sine/live_observed_rollout_annotated_sheet.png`.
  Visual review agrees with the metric: this is not a small final-threshold
  miss. The peg is not preserved in the grasp through the late WM-active phase,
  and the robot ends up pushing near the block without valid insertion.
- Formal `iter_000000900` live-receding panel completed at
  `2026-06-15T00:12:52+08:00`: `completed_samples=4`,
  `final_success_count=0`, `method_evidence_allowed=false`,
  `panel_full_episode_contract_ok=true`, `sample_contract_failures=[]`, and
  `failed_process_count=0`. Plain conclusion: generated-video/action artifacts
  are structurally valid under the corrected `301` frame / `300` action
  contract, but closed-loop real execution still fails. The failure is now
  concrete: current raw Cosmos action chunks plus the DP handoff can bring some
  cases near the hole, but they do not reliably preserve grasp/contact and
  finish insertion after dynamic target motion.
- Follow-up formal eval for `iter_000001200` started in the same held eval
  allocation at `2026-06-15T00:17:26+08:00`, Slurm job `127825`, step `70`,
  node `server24`. It uses the same generated-eval plus live-receding pipeline
  and selected checkpoint `iter_000001200`; the model `.metadata` mtime is
  `2026-06-14T23:54:26+08:00`. The first retry attempted `120G` inside a
  `100G` eval allocation and was rejected by Slurm before running; the active
  retry uses `90G`. No result is claimed yet for `iter_000001200`.
- Formal `iter_000001200` generated eval completed at
  `2026-06-15T00:25:23+08:00`: `10/10` samples inspected,
  `strict_eval_artifacts_ok=true`, `strict_failures=[]`, video frames `301`,
  action shape `[300, 32]`. Mean diagnostics: action RMSE `0.4521`,
  robot-action future RMSE `0.7785`, state-sidecar future RMSE `0.4497`, and
  future video PSNR `22.5841`. A combined review sheet was generated and
  opened at
  `eval_full_episode_wam_iter_000001200_formal_after_3h/review_sheets/combined_iter1200_ref_pred_sheet.png`;
  representative original sheets `00`, `03`, `08`, and `09` were also opened.
  Visual review: generated videos are nonblank and frame-aligned; some
  insert/resume rows track the reference closely, but visible pose errors
  remain in pre-motion/static-monitor rows. This is generated-eval evidence
  only.
- Formal `iter_000001200` live-receding panel started at
  `2026-06-15T00:25:23+08:00`, output root
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_full300_panel_iter_000001200_clean_dense_20260615_002523`.
  Early artifact status: `sample_00_hole_late_move_stop` reached
  `iter_00_prefix_f106`, which means the first WM call is conditioned on the
  real observed prefix through frame `106`. Final live metrics and video review
  are still pending.
- Formal `iter_000001200` live-receding sample `sample_00_hole_late_move_stop`
  completed at `2026-06-15T01:12:56+08:00`: `25` completed iterations,
  `301` observed frames, `final_success=false`, final peg-head-in-hole-frame
  `[-0.1085, -0.0441, -0.0400]`, controller frames `INIT_OBS=1`,
  `DP_SCAN_TARGET=106`, `WM_ACTIVE=194`, and `dp_handoff_executed_steps=0`.
  A readable `301`-frame annotated video was inspected through
  `sample_00_hole_late_move_stop/video_review/live_observed_rollout_annotated_review_sheet.png`.
  Visual review agrees with the metric: the peg is not inserted, and the robot
  ends below/aside the hole block. Compared with `iter900`, this checkpoint is
  worse on this sample because it never reaches the near-handoff state.
- The formal 3-hour wall-clock boundary was reached at
  `2026-06-14T20:57:46+08:00`, but `iter_000000600` was saved at
  `2026-06-14T20:56:15+08:00` and its model `.metadata` timestamp is
  `2026-06-14T20:56:11+08:00`, before the boundary. Therefore `iter_000000600`
  is not formal method evidence. Continue training and use the next checkpoint
  whose actual metadata timestamp is after the boundary, likely
  `iter_000000900`.
- Runtime note: this live wrapper currently reloads Cosmos for each receding
  query. That is acceptable for a diagnostic panel but too inefficient for the
  intended deployment design. The intended method still requires a resident or
  cached low-frequency WM call path plus a faster executor.
- Added
  `scripts/slurm/run_cosmos3_clean_dense_late299_formal_eval_live_pipeline_in_allocation.sh`
  for the formal post-3-hour stage. It selects the first checkpoint whose
  actual `.metadata` timestamp is after `2026-06-14T20:57:46+08:00`, waits for
  stable checkpoint files, runs strict generated eval, and then runs
  live-receding closed-loop eval. It intentionally skips readout because
  readout is diagnostic and already blocked on IO/model loading; real
  final-state metrics plus inspected video remain the authority. `bash -n`
  passed.

For a 4-GPU launch, the relevant command settings are:

```text
ALLOW_CLEAN_DENSE_FULL_SFT=true
ALLOW_LIVE_QUERY_COVERAGE_GAP_FOR_FULL_SFT=true
MAX_PREFIX_FRAMES=299
CONDITION_ROOT=experiments/world_model_task_rebinding/cosmos3/full_episode_wam_conditions_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_20260614_130050
CLEAN_DENSE_PREFLIGHT_SUMMARY=experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_preflight_20260614_130050/clean_dense_preflight_summary.json
OUTPUT_ROOT=experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299
bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_full_fix1recipe_in_allocation.sh
```

For a 2-GPU launch, use the same inputs and set:

```text
NPROC_PER_NODE=2
DATA_PARALLEL_SHARD_DEGREE=2
OUTPUT_ROOT=experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_2gpu_20260614_priority_full_sft_max299
```

## 2026-06-15 Early Status

- The formal dense SFT completed past the updated `2` GPU / `3` hour minimum
  and saved checkpoints through `iter_000001500` in:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299`.
  This is a valid formal training run because the already-held 4-GPU resource
  exceeds the new floor.
- Generated eval is not the current blocker. `iter900` and `iter1200` each
  passed the strict 10-sample generated-artifact check with full `301` video
  frames and `[300, 32]` action outputs. `iter1500` also passed the corrected
  strict generated gate, and an extra `iter1500` generated inspection produced
  `72` valid samples with `strict_eval_artifacts_ok=true` and no strict
  failures.
- Closed-loop real execution is still failing. Formal `iter900` completed
  `0/4` real final-state successes. Formal `iter1200` also completed `0/4`
  with `panel_full_episode_contract_ok=true`, `failed_process_count=0`, and no
  sample contract failures. All four `iter1200` annotated rollout videos were
  converted to review sheets and opened:
  `sample_00_hole_late_move_stop/video_review/live_observed_rollout_annotated_review_sheet.png`,
  `sample_01_hole_late_constant/video_review/live_observed_rollout_annotated_review_sheet.png`,
  `sample_03_hole_late_fast_shift/video_review/live_observed_rollout_annotated_review_sheet.png`,
  and
  `sample_04_hole_late_sine/video_review/live_observed_rollout_annotated_review_sheet.png`.
  The visual evidence agrees with the metrics: the peg is brought near the
  hole/box in several cases but remains outside the insertion manifold. One
  important diagnostic: `iter1200/sample_01_hole_late_constant` executed `136`
  DP handoff steps and `iter1200/sample_04_hole_late_sine` executed `68` DP
  handoff steps, but both still failed. The problem is not just that the
  handoff gate never fires.
- `server35` should not be used for live ManiSkill render evidence. Parallel
  `iter1500` live eval failed with Vulkan `DeviceLost`, and serial live eval
  stalled before useful progress. This is a node/render scheduling problem,
  not evidence against the trained checkpoint.
- A replacement tmux-held live allocation is active as Slurm job `128006` on
  `server62`, output root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_full300_panel_iter_000001500_clean_dense_server62_20260615_0224`.
  The first live-prefix Cosmos query completed and the first executed chunk
  progressed through frame `154` after six executed chunks with real
  final-state success still false. Keep the panel running to get the final
  `iter1500` closed-loop result.

Plain conclusion at this point: training data export, SFT launch, checkpoint
selection, generated strict eval, and the `301/300` length contract are not the
immediate failure. The current blocker is execution physics in closed loop:
raw Cosmos action chunks plus the current DP handoff can move toward the hole,
but they do not reliably create a stable, grasp-preserving, DP-continuable
insertion state after the target moves.

## 2026-06-15 Iter1500 Closed-Loop Stop Point

The `iter_000001500` generated gate passed, and the extra generated inspection
also passed `72` samples with `strict_eval_artifacts_ok=true`. The remaining
failure is the live closed loop.

Because `server35` live render was unreliable, the final `iter1500` live
evidence was collected on render-capable held allocations:

- sample00 on `server62`:
  `live_receding_full300_panel_iter_000001500_clean_dense_server62_20260615_0224/sample_00_hole_late_move_stop`.
  Result: `final_success=false`, `301` frames, `DP_HANDOFF=30`,
  final peg-head-in-hole `[-0.0829, -0.0107, 0.0027]`.
- sample01 on `server24`:
  `live_receding_full300_panel_iter_000001500_clean_dense_server24_sample01_20260615_025555/sample_01_hole_late_constant`.
  Result: `final_success=false`, `301` frames, `DP_HANDOFF=8`,
  final peg-head-in-hole `[-0.0822, -0.0015, 0.0051]`.
- sample03 on `server62`:
  `live_receding_full300_panel_iter_000001500_clean_dense_server62_sample03_20260615_033427/sample_03_hole_late_fast_shift`.
  Result: `final_success=false`, `301` frames, no DP handoff,
  final peg-head-in-hole `[-0.1246, -0.0129, -0.0276]`.
- sample04 on `server24`:
  `live_receding_full300_panel_iter_000001500_clean_dense_server24_sample04_20260615_041209/sample_04_hole_late_sine`.
  Result: `final_success=false`, `301` frames, `DP_HANDOFF=72`,
  final peg-head-in-hole `[-0.0424, 0.0026, -0.0030]`.

Review sheets were generated and opened for all four samples:

- `sample_00_hole_late_move_stop/video_review/live_observed_rollout_annotated_review_sheet.png`
- `sample_01_hole_late_constant/video_review/live_observed_rollout_annotated_review_sheet.png`
- `sample_03_hole_late_fast_shift/video_review/live_observed_rollout_annotated_review_sheet.png`
- `sample_04_hole_late_sine/video_review/live_observed_rollout_annotated_review_sheet.png`

Visual review agrees with the metrics: the robot often approaches the moved
hole/box, and DP handoff sometimes executes, but the peg remains outside the
insertion manifold. This repeats the `iter900` and `iter1200` failure pattern.

Stop point: direct raw Cosmos action chunks plus threshold/continuability
handoff should no longer be treated as the main controller. The next aligned
method step is the planned low-frequency world model plus learned executor or
DP-prior residual controller. Continuing to tune thresholds or add enumerated
error-recovery cases would violate the project boundary.

A compact evidence index for user review is recorded at
`docs/world_model_task_rebinding/2026-06-15_dense_closed_loop_failure_evidence_index.md`.
