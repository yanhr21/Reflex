# Joint Training TODO

- [ ] Inspect existing DP and Cosmos data interfaces inside a Slurm allocation
  before writing runnable training code.
  - 2026-07-09: added `joint_overfit_abcd` as the immediate readiness guard
    for the required A/B/C/D overfit route. `full_joint` remains a later
    all-class guard that still requires E after Cosmos/readout prediction is
    available.
  - 2026-07-09: added
    `scripts/world_model/inspect_joint_dp_cosmos_interfaces.py`,
    `scripts/slurm/run_joint_dp_cosmos_interface_inspect_in_allocation.sh`,
    and `scripts/slurm/launch_joint_dp_cosmos_interface_inspect_tmux.sh` for
    the required Slurm-side interface inspection before runnable training code.
  - 2026-07-09 15:44: launched tmux session `joint_interface_inspect01`
    with Slurm job `173059` for the interface inspection. Requested
    `1 GPU / 1 CPU / 8G / 1h`; Slurm reports pending on priority with
    scheduled node `server27` and estimated start `2026-07-09T19:52:07`.
  - 2026-07-09: added
    `PLAN/02_joint_training/overfit_contract.md` to lock A/B/C/D loss roles:
    C success/failure is an outcome label only and C must not feed positive
    DP behavior cloning.
  - 2026-07-09: added read-only status helper
    `scripts/world_model/joint_training_status.sh`. Current status:
    `joint_overfit_abcd_ready=true`, `full_joint_ready=false` because E is
    not ready, interface inspection job `173059` still pending on priority.
  - 2026-07-09 15:53: compared shorter `10m` / `30m` test-only GPU requests.
    Slurm estimated those alternatives no earlier than 2026-07-15, while the
    existing valid job `173059` remains estimated for 2026-07-09 19:52 on
    `server27`; keep monitoring the existing tmux-held request.
  - 2026-07-09: added
    `PLAN/02_joint_training/interface_review_checklist.md`; runnable overfit
    dataset/trainer scripts remain gated on passing
    `project_interface_summary.json` and `cosmos_interface_summary.json`.
  - 2026-07-09: added
    `scripts/world_model/require_joint_interface_inspect_ready.sh` and wired it
    into `scripts/world_model/joint_training_status.sh`. It currently reports
    `joint_interface_inspect_ready=false` because the Slurm inspection has not
    started and has not written manifest / summaries.
  - 2026-07-09 15:58: `scripts/world_model/joint_training_status.sh` still
    reports `joint_overfit_abcd_ready=true`,
    `joint_interface_inspect_ready=false`, and Slurm job `173059` pending on
    priority with estimated start `2026-07-09T19:52:07`. No runnable overfit
    code has been started before this interface gate.
  - 2026-07-09 20:03: Slurm job `173059` ran on `server27` and failed fast
    with exit code 62 before Python interface inspection. Diagnosis:
    `RUN_GROUP=interface_inspect` and `RUN_NAME=inspect01` leaked into the
    A/B/C/D dataset readiness validators, causing them to check
    `01_dataset/interface_inspect/inspect01` instead of the real production
    runs. Patched `scripts/world_model/require_dataset_training_inputs_ready.sh`
    to isolate validator environment. Rechecked
    `joint_overfit_abcd` under the polluted outer environment and it now
    reports A=`static_rgb/full01`, B=`dynamic_rgb/prod01`,
    C=`frozen_dp_dynamic/prod01`, D=`future_teacher/prod01`,
    `dataset_training_inputs_ready=true`, and `failure_count=0`.
  - 2026-07-09 20:05: relaunched as tmux session
    `joint_interface_inspect02`, Slurm job `173286`, run
    `interface_inspect/inspect02`. The job completed on `server27` with exit
    code 0 in 52 seconds. `project_interface_summary.json` and
    `cosmos_interface_summary.json` both report `status=ok` and
    `failure_count=0`; this opens the interface gate for writing the A/B/C/D
    overfit dataset and batch-inspection code.
- [x] Add `scripts/world_model/build_joint_dp_cosmos_dataset.py`.
  - 2026-07-09: builder added for the tiny real A/B/C/D overfit manifest.
    It refuses login-node execution by default, reruns the
    `joint_overfit_abcd` and interface-inspect gates inside Slurm, writes
    `joint_overfit_samples.jsonl`, and records C success as a valid outcome
    label rather than positive DP BC.
- [x] Add `scripts/world_model/inspect_joint_dp_cosmos_batch.py`.
  - 2026-07-09: batch inspector added. It validates real video decoding,
    action chunks with shape `(H, 7)`, A static H5 actions, B/C/D trace
    actions, and class-specific loss permissions without training or method
    success claims.
- [x] Run the first A/B/C/D joint overfit batch inspection.
  - 2026-07-09 20:18: launched tmux session `joint_batch_inspect01`,
    Slurm job `173294`, run `joint_overfit_dataset/overfit01`.
    The job completed on `server27` with exit code 0 in 32 seconds.
    Builder selected 16 real samples, A/B/C/D = 4/4/4/4, with success
    outcome counts A=4, B=1, C=2, D=1. Batch inspection wrote
    `batch_inspection.json` with `status=ok`, `failure_count=0`, decoded
    videos at 30 FPS, and verified all inspected action chunks as `(16, 7)`.
- [x] Build and preflight the Cosmos SFT condition root for the joint overfit.
  - 2026-07-09: added
    `scripts/world_model/build_joint_dp_cosmos_sft_condition_root.py`,
    `scripts/slurm/run_joint_dp_cosmos_sft_condition_in_allocation.sh`, and
    `scripts/slurm/launch_joint_dp_cosmos_sft_condition_tmux.sh`.
    The converter uses only B/D rows from `joint_overfit_dataset/overfit01`
    because they have real RGB videos plus trace `task_rows`; A remains the
    protected DP action objective, and C remains outcome/discrepancy labels.
    It writes 300-frame / 300-action / 32-channel Cosmos SFT JSONL records
    derived from recorded actions and task traces, with causal prefix metadata
    and no future target labels in conditions.
  - 2026-07-09 20:24: launched tmux session `joint_cosmos_condition01`,
    Slurm job `173298`, run `joint_cosmos_condition/overfit01`.
    Requested `1 GPU / 1 CPU / 8G / 1h`; it ran on `server27` and failed
    preflight with exit code 1. The converter succeeded and wrote 8 B/D rows,
    but B and D reused raw sample ids such as `cont_episode_000001`, so
    `source_uuid` collided across classes and preflight saw only 4 unique
    source rows instead of 8. Patched the converter to prefix UUID/source UUID
    with the source class, e.g. `B_cont_episode_...` and
    `D_cont_episode_...`.
  - 2026-07-09 20:26: relaunched as tmux session
    `joint_cosmos_condition02`, Slurm job `173301`, run
    `joint_cosmos_condition/overfit02`. The job completed on `server27`
    with exit code 0 in 10 seconds. It wrote
    `condition_root/train/video_action_dataset_file.jsonl` and matching val
    JSONL with 8 B/D rows, `num_video_frames=300`,
    `num_action_steps=300`, `raw_action_dim=32`, and preflight
    `strict_alignment_ok=true`. Follow-up code inspection found that the
    existing Cosmos SFT dataloader accepts `.npy` action paths or bare JSON
    arrays, while `overfit02` wrote action paths as JSON objects accepted only
    by the local preflight. Patched the converter to write `.npy` action
    arrays before using the condition root for real Cosmos training.
  - 2026-07-09 20:27: launched tmux session
    `joint_cosmos_condition03`, Slurm job `173302`, run
    `joint_cosmos_condition/overfit03`, to regenerate the same B/D condition
    root with `.npy` action arrays compatible with the real Cosmos SFT
    dataloader. The job completed on `server27` with exit code 0 in 10
    seconds. It wrote 8 train + 8 val B/D rows, action paths ending in
    `action_state_32.npy`, `num_video_frames=300`,
    `num_action_steps=300`, `raw_action_dim=32`, and preflight
    `strict_alignment_ok=true` with 8 unique source rows and 0 row failures.
    Follow-up inspection of the real Cosmos SFT dataloader found two more
    hard requirements not covered by the local preflight: each JSONL row must
    include `width` / `height`, and action condition length must equal
    `video_frames - 1`. Patched the converter to write video metadata and to
    export 299 transition rows for each 300-frame B/D video.
  - 2026-07-09 20:31: launched tmux session
    `joint_cosmos_condition04`, Slurm job `173304`, run
    `joint_cosmos_condition/overfit04`. It wrote 300-frame / 299-action rows
    with video metadata, but preflight failed because state targets and task
    labels were still exported at action length 299 instead of video length
    300. Patched the converter so action condition arrays are 299 transition
    rows while state targets and task labels remain 300 video-frame rows.
  - 2026-07-09 20:33: relaunched as tmux session
    `joint_cosmos_condition05`, Slurm job `173305`, run
    `joint_cosmos_condition/overfit05`. The job completed on `server27`
    with exit code 0 in 11 seconds. It wrote 8 B/D train rows and matching
    val rows, 300 video frames, 299
    action-transition rows, `.npy` action arrays, 32 raw action/state
    channels, video `width` / `height` metadata, default
    `video_dataset_file.jsonl` files for the Cosmos recipe, and preflight
    `strict_alignment_ok=true`.
  - 2026-07-09 20:39: after the first real Cosmos dataloader attempt failed
    on missing JSONL `duration`, patched the converter to add
    `duration=nb_frames/framerate` and regenerated
    `joint_cosmos_condition/overfit06` with schema
    `joint_dp_cosmos_sft_condition_root_v2`. Slurm job `173332` completed
    on `server27`; `condition_preflight.json` reports
    `strict_alignment_ok=true`, 8 train rows, 8 val rows, B/D = 4/4, and no
    failures. This is the current trainable condition root for Cosmos SFT.
  - 2026-07-09 20:44: after the second real Cosmos dataloader attempt showed
    that the official loader temporally truncates the 300-frame source video
    to 297 frames (`4*N+1`) and expects 296 action/state rows, patched the
    converter and preflight to separate `source_video_frames=300` from
    `num_video_frames=297`. Regenerated
    `joint_cosmos_condition/overfit07` on CPU Slurm job `173336`; preflight is
    strict OK with 8 B/D train rows, 8 val rows, `num_action_steps=296`,
    `raw_action_dim=32`, and schema
    `joint_dp_cosmos_sft_condition_root_v3`. This supersedes overfit06 for
    real Cosmos SFT.
- [ ] Add `scripts/training/train_joint_dp_cosmos_overfit.py`.
  - 2026-07-09: added real-Cosmos launcher scaffolding
    `scripts/slurm/run_joint_cosmos_sft_overfit_in_allocation.sh` and
    `scripts/slurm/launch_joint_cosmos_sft_overfit_tmux.sh`. These call
    `.venv_cosmos313/bin/torchrun -m cosmos_framework.scripts.train` with the
    active DCP checkpoint and the `joint_cosmos_condition/overfit07`
    condition root. They do not implement a toy model.
- [ ] Run the overfit experiment on a tiny real slice.
  - 2026-07-09 20:35: launched tmux session
    `joint_cosmos_sft_overfit01`, Slurm job `173330`, run
    `cosmos_sft_overfit/overfit01`. The real Cosmos train entry point
    loaded config, initialized distributed runtime, loaded the local
    tokenizer/VAE, and reached the official SFT dataloader, then failed with
    `KeyError: 'duration'` from
    `cosmos_framework/data/vfm/local_datasets/sft_dataset.py`. This was a
    condition-root schema issue, not a toy-model fallback.
  - 2026-07-09 20:39: launched tmux session
    `joint_cosmos_sft_overfit02`, Slurm job `173333`, run
    `cosmos_sft_overfit/overfit02`, using
    `joint_cosmos_condition/overfit06` and Cosmos job name
    `joint_cosmos_sft_overfit02`. It reached the official SFT dataloader and
    loaded 8 metadata rows, proving the `duration` fix, but every sample was
    skipped because action/state arrays had 299 rows while the dataloader's
    truncated 297-frame video expected 296. The job was cancelled with exit
    status 143 to avoid wasting GPU time.
  - 2026-07-09 20:44: launched tmux session
    `joint_cosmos_sft_overfit03`, Slurm job `173337`, run
    `cosmos_sft_overfit/overfit03`, using
    `joint_cosmos_condition/overfit07` and Cosmos job name
    `joint_cosmos_sft_overfit03`. It loaded train/val metadata without
    action-length mismatch, loaded the active checkpoint, started real FSDP
    training, and completed iteration 1 with
    `Loss=8.3241`, `vision=0.0132`, `action=0.8311`. It then failed on the
    next training step with CUDA OOM on H200 (`139.39 GiB` in use, `405 MiB`
    free). This proves the data path and first real backward step, but does
    not provide a completed/saved smoke checkpoint.
  - 2026-07-09 20:51: patched the Cosmos SFT runner so validation can be
    disabled by environment variables. Next smoke should use
    `MAX_ITER=1`, `SAVE_ITER=1`, `RUN_VALIDATION=false` to get a completed
    real-Cosmos checkpoint before attempting longer overfit or LoRA/memory
    reductions.
  - 2026-07-09 20:52: launched tmux session
    `joint_cosmos_sft_overfit04`, Slurm job `173346`, run
    `cosmos_sft_overfit/overfit04`, with `MAX_ITER=1`, `SAVE_ITER=1`,
    `RUN_VALIDATION=false`, and condition root
    `joint_cosmos_condition/overfit07`. It failed before training with a
    runner shell syntax error after the validation-toggle patch. Patched the
    runner to simplify nested quoting in `cosmos_nvidia_lib_dirs` and patched
    the launcher to pass `MAX_ITER`, `SAVE_ITER`, `VALIDATION_ITER`,
    `MAX_VAL_ITER`, `RUN_VALIDATION`, and `RUN_VALIDATION_ON_START` through
    to the runner.
  - 2026-07-09 20:53: relaunched the 1-iter no-validation smoke as tmux
    session `joint_cosmos_sft_overfit05`, Slurm job `173349`, run
    `cosmos_sft_overfit/overfit05`, using condition root
    `joint_cosmos_condition/overfit07`. It allocated `server27`, but did not
    produce a valid training result: the log stopped after the Cosmos config
    debug line, no `torchrun`/Python train process remained, GPU usage was
    only `4MiB / 0%`, and the held Slurm step was manually cancelled at
    21:03 to avoid idle GPU occupancy. `classification.txt` records
    `cosmos_sft_overfit_status=cancelled_idle_orphaned`; this is not a saved
    checkpoint and not method evidence.
  - 2026-07-09: next run must reduce the real Cosmos memory footprint rather
    than retrying the same 297-frame full fine-tune. Candidate is a shorter
    B/D condition root that still uses real recorded videos/actions and the
    official Cosmos SFT entry point, with the same strict preflight and no toy
    model fallback.
  - 2026-07-09 21:06: generated short-window condition root
    `joint_cosmos_condition/overfit09` on CPU Slurm job `173380`
    (`server13`). It uses the same real B/D videos and traces, but exports
    `num_video_frames=93` and `num_action_steps=92`; strict preflight passed
    with train/val rows 8 each, B/D=4/4, and zero failures. The interrupted
    `joint_cosmos_condition/overfit08` is a partial cancelled build and must
    not be used.
  - 2026-07-09 21:03-21:39: attempted multiple real-Cosmos short-window
    startup controls using `joint_cosmos_condition/overfit09`. None produced
    a completed checkpoint. `overfit06` and `overfit07` reproduced the idle
    startup pattern on `server27`; `overfit08` added PID instrumentation and
    showed torchrun/worker reaching only the Cosmos config debug line before
    staying in `D` state at `epc_proc_wait_request` with GPU idle at 4MiB.
    `overfit10` reproduced the same on `gpux/server13`; `overfit11` reproduced
    it with direct `python -m cosmos_framework.scripts.train`, ruling out the
    torchrun/elastic layer; `overfit13` reproduced it with no LoRA, compile
    enabled, and grad accumulation 2, ruling out LoRA and the short-window
    memory-reduction settings as the immediate blocker. All idle GPU holds
    were manually cancelled and classified; none are method evidence.
  - 2026-07-09: patched
    `scripts/slurm/run_joint_cosmos_sft_overfit_in_allocation.sh` and
    `scripts/slurm/launch_joint_cosmos_sft_overfit_tmux.sh` to record
    `cosmos_train_invocation`, `cosmos_train_pid`, `USE_TORCHRUN`, LoRA
    settings, compile setting, and grad accumulation. The next useful step is
    not another blind retry; inspect the Cosmos train startup after config
    loading, especially the dependency/file wait behind
    `epc_proc_wait_request`, before requesting another GPU.
  - 2026-07-09: added CPU-only startup import diagnostic runners
    `scripts/slurm/run_cosmos_startup_import_diag_in_allocation.sh` and
    `scripts/slurm/launch_cosmos_startup_import_diag_tmux.sh`. Run
    `cosmos_startup_diag/import_trace01` timed out before `CONFIG_LOADED`
    without starting training. `-X importtime` and `strace` show the config
    path is stuck inside `make_config()` imports, with `diffusers` import
    triggering `importlib.metadata.packages_distributions()` and a slow
    all-distribution metadata walk over the shared `.venv_cosmos313`
    filesystem. This rules out C data, LoRA, torchrun, server27, and
    short-window memory settings as the immediate startup blocker.
  - 2026-07-09: added
    `scripts/world_model/cosmos_fast_import_sitecustomize/sitecustomize.py`
    and wired it into both the startup diagnostic runner and the real Cosmos
    SFT overfit runner. The patch is startup-only: it returns an empty map for
    `packages_distributions()` so `diffusers` does not globally scan package
    metadata, while preserving normal per-package `version()` checks and
    leaving Cosmos model / data / loss code unchanged. Launched CPU Slurm
    diagnostic `cosmos_startup_diag/import_trace02` as tmux session
    `cosmos_startup_diag02`, job `173415`, to verify `CONFIG_LOADED` before
    any new GPU request.
  - 2026-07-09: `cosmos_startup_diag/import_trace02` is invalid diagnostic
    evidence because the runner heredoc was parsed incorrectly and the script
    never reached `diag_start`; it is classified as
    `invalid_runner_heredoc_parse`. Patched the diagnostic runner to write a
    `diag_load_config.py` file and execute that file instead of passing a
    multi-line `-c` string.
  - 2026-07-09: `cosmos_startup_diag/import_trace03` ran on CPU Slurm job
    `173416` (`server13`) and proved the startup patch changes the failure
    mode: `make_config()` returned after about `44.6s` instead of hanging
    before the `importlib.import_module` debug line. The run then failed in
    Hydra override application because the runner used stale TOML paths such
    as `model.compile.enabled`; the structured TOML remaps VFM model fields
    to `model.config.*`. Patched the diagnostic and real SFT runners to use
    `model.config.compile.enabled`, `model.config.lora_*`, and
    `model.config.ema.enabled`. Launched `cosmos_startup_diag/import_trace04`
    as tmux session `cosmos_startup_diag04`, job `173417`, to verify
    `CONFIG_LOADED` before requesting GPU.
  - 2026-07-09: `cosmos_startup_diag/import_trace04` is also invalid guard
    evidence. It launched before the corrected diagnostic body was actually in
    effect and still generated `model.compile.enabled=true`; after Python
    returned, the shell runner also hit a classification quote bug. Patched
    the classification block to avoid nested quotes and confirmed the runner
    now generates `model.config.compile.enabled=false`. Launched
    `cosmos_startup_diag/import_trace05`, tmux session
    `cosmos_startup_diag05`, job `173418`, as the current CPU guard.
  - 2026-07-09: `cosmos_startup_diag/import_trace05` completed on CPU Slurm
    job `173418` (`server13`) with exit status 0. It printed
    `CONFIG_LOADED diag_import_trace`; `load_config` reported total config
    load time about `16.2s`. This opens the guard for the next GPU smoke.
    Next GPU run should use the corrected startup patch and corrected
    `model.config.*` overrides, with short-window
    `joint_cosmos_condition/overfit09`, `MAX_ITER=1`,
    `RUN_VALIDATION=false`, `MODEL_COMPILE_ENABLED=false`, and LoRA enabled
    as a real Cosmos memory-control setting, not a toy replacement.
  - 2026-07-09 21:57: launched GPU smoke
    `cosmos_sft_overfit/overfit14` as tmux session
    `joint_cosmos_sft_overfit14`, Slurm job `173420`. Request:
    `1 GPU / 8 CPU / 96G / 1h`, condition root
    `joint_cosmos_condition/overfit09`, `MAX_ITER=1`, `SAVE_ITER=1`,
    `RUN_VALIDATION=false`, `GRAD_ACCUM_ITER=1`,
    `MODEL_COMPILE_ENABLED=false`, `ENABLE_LORA=true`. It is pending on
    priority at launch time; keep monitoring the held request.
  - 2026-07-09: `cosmos_sft_overfit/overfit14` ran on `server27` and failed
    before training iteration. It passed the slow import stage
    (`importlib.import_module` returned after about `33.5s`) but Hydra rejected
    `model.config.lora_target_modules=q_proj_moe_gen,k_proj_moe_gen,...` as
    an ambiguous sweep because the comma-separated string was not quoted.
    Patched the runner to emit
    `model.config.lora_target_modules='q_proj_moe_gen,...'` as a string.
  - 2026-07-09 22:05: `cosmos_sft_overfit/overfit15` completed the first
    saved real-Cosmos short-window GPU smoke on Slurm job `173440`
    (`server27`). It used `joint_cosmos_condition/overfit09`,
    `MAX_ITER=1`, `SAVE_ITER=1`, `RUN_VALIDATION=false`,
    `MODEL_COMPILE_ENABLED=false`, and LoRA enabled against the real Cosmos
    checkpoint. The run loaded the active DCP checkpoint, started FSDP
    training, completed iteration 1 with `Loss=1.6022`, `vision=0.0050`, and
    `action=0.1597`, then saved
    `cosmos_output/cosmos3/sft/joint_cosmos_sft_overfit15/checkpoints/iter_000000001`.
    `classification.txt` records `cosmos_sft_overfit_status=complete`. This
    proves the real short-window Cosmos SFT path can train and save a
    checkpoint, but it is still a one-iteration smoke, not a converged overfit,
    not full training evidence, and not closed-loop method evidence.
  - 2026-07-09 22:06: launched the next multi-step real-Cosmos tiny overfit
    attempt as `cosmos_sft_overfit/overfit16`, tmux session
    `joint_cosmos_sft_overfit16`, Slurm job `173443`. It uses the same
    strict short-window B/D condition root `joint_cosmos_condition/overfit09`
    with `MAX_ITER=5`, `SAVE_ITER=5`, validation disabled, compile disabled,
    and LoRA enabled. The goal is to move from a one-iteration saved smoke to
    a short multi-step training trace with separate `vision` and `action`
    losses plus a saved checkpoint. It is not full training evidence or
    closed-loop method evidence.
  - 2026-07-09 22:10: `cosmos_sft_overfit/overfit16` completed with exit
    status 0 on `server27`. It loaded the active real Cosmos checkpoint,
    trained five iterations on the strict B/D short-window condition root, and
    saved
    `cosmos_output/cosmos3/sft/joint_cosmos_sft_overfit16/checkpoints/iter_000000005`.
    Logged losses were:
    iter 1 `Loss=1.6022`, `vision=0.0050`, `action=0.1597`;
    iter 2 `Loss=1.9469`, `vision=0.0065`, `action=0.1940`;
    iter 3 `Loss=1.7392`, `vision=0.0096`, `action=0.1730`;
    iter 4 `Loss=1.6557`, `vision=0.0066`, `action=0.1649`;
    iter 5 `Loss=1.5949`, `vision=0.0064`, `action=0.1588`.
    The checkpoint includes model / optimizer / scheduler / trainer DCP
    metadata and `classification.txt` records
    `cosmos_sft_overfit_status=complete`. This is the first multi-step
    real-Cosmos tiny overfit smoke with separate action and vision losses, but
    it still does not satisfy full training or reset-to-end closed-loop
    evidence.
  - 2026-07-09 22:24: added active forward-dynamics eval wrappers
    `scripts/slurm/run_joint_cosmos_forward_dynamics_eval_in_allocation.sh`
    and `scripts/slurm/launch_joint_cosmos_forward_dynamics_eval_tmux.sh`.
    Ran `cosmos_forward_eval/eval01` on Slurm job `173460` (`server27`) using
    the real `overfit16` checkpoint at
    `checkpoints/iter_000000005`, its saved `config.yaml`, and condition root
    `joint_cosmos_condition/overfit09`. The eval selected
    `B_cont_episode_000002`, converted its `[92, 32]` `.npy` action/state
    tensor to official Cosmos JSON, ran official `forward_dynamics`
    inference, and wrote
    `inference_output/forward_B_cont_episode_000002/vision.mp4`.
    Reconstruction diagnostics compared 93 generated frames against the
    current reference-window contract and wrote
    `reconstruction_metrics/metrics.json` plus
    `reconstruction_comparison_sheet.png`: mean MSE `0.0008612`, mean MAE
    `0.0177664`, mean PSNR `30.8112`, mean SSIM `0.85176`.
    `classification.txt` records `cosmos_forward_eval_status=complete`,
    `method_evidence_allowed=false`, and `closed_loop_evidence=false`. This
    proves the saved real-Cosmos overfit checkpoint can generate a
    forward-dynamics video through the official inference path, but it is still
    only diagnostic future-state evidence. It does not prove closed-loop
    control, insertion success, or a fully corrected prefix-after-motion
    future-window dataset contract.
  - 2026-07-09 22:28: fixed the short-window condition-root contract after
    `cosmos_forward_eval/eval01` exposed that `overfit09` still referenced the
    full 300-frame source video and used absolute `window_start` as a capped
    relative prefix. Patched
    `scripts/world_model/build_joint_dp_cosmos_sft_condition_root.py` to:
    rebase action/task/motion trace rows to the selected source window,
    materialize per-sample `window_rgb.mp4`, keep the original
    `source_vision_path` and `source_window_start/end_frame`, and set the
    causal condition prefix to the relative `obs_horizon` (`2` frames) rather
    than the absolute trigger step. Patched
    `preflight_cosmos3_full_episode_wam_contract.py` and the condition-root
    Slurm runner so manifest `source_video_frames=300` remains the original
    source-video contract while exported `vision_path` videos are checked at
    `93` frames. Rebuilt `joint_cosmos_condition/overfit10` on CPU Slurm job
    `173468`; `condition_preflight.json` reports `strict_alignment_ok=true`,
    prefix frame min/max `2`, 8 train rows, 8 val rows, B/D=4/4, and zero
    failures. This supersedes `overfit09` for subsequent Cosmos training.
  - 2026-07-09 22:33: reran the real Cosmos short overfit on corrected
    `joint_cosmos_condition/overfit10` as `cosmos_sft_overfit/overfit17`,
    Slurm job `173482` on `server27`. The run completed with exit status 0,
    loaded the active real Cosmos base checkpoint, trained five iterations,
    and saved
    `cosmos_output/cosmos3/sft/joint_cosmos_sft_overfit17/checkpoints/iter_000000005`.
    Logged losses were:
    iter 1 `Loss=11.2537`, `vision=0.0582`, `action=1.1196`;
    iter 2 `Loss=11.2894`, `vision=0.0633`, `action=1.1226`;
    iter 3 `Loss=11.2967`, `vision=0.0683`, `action=1.1228`;
    iter 4 `Loss=11.1195`, `vision=0.0518`, `action=1.1068`;
    iter 5 `Loss=11.2869`, `vision=0.0553`, `action=1.1232`.
    `classification.txt` records `cosmos_sft_overfit_status=complete`.
    This supersedes `overfit16` as the current corrected-window real-Cosmos
    tiny overfit checkpoint, but it is still not full training or closed-loop
    method evidence.
  - 2026-07-09 22:36: reran forward-dynamics inference on the corrected
    checkpoint/root as `cosmos_forward_eval/eval02`, Slurm job `173484` on
    `server27`. It used `cosmos_sft_overfit/overfit17` checkpoint
    `iter_000000005`, corrected condition root `joint_cosmos_condition/overfit10`,
    selected `B_cont_episode_000002`, and confirmed the inference sample uses
    `window_rgb.mp4` with `condition_prefix_frames=2` rather than the old
    full-source video. Official Cosmos `forward_dynamics` inference completed
    and wrote
    `inference_output/forward_B_cont_episode_000002/vision.mp4`.
    Reconstruction metrics compared against the corrected 93-frame window
    video with length match true: mean MSE `0.0120718`, mean MAE `0.0530525`,
    mean PSNR `19.4171`, mean SSIM `0.652853`. This is the current valid
    future-state diagnostic for the corrected-window route. It is still not
    reset-to-end closed-loop control evidence or insertion success evidence.
  - 2026-07-09 23:08: completed a larger corrected-window condition root
    preparation from `joint_training_dataset/train_slice01`. The dataset
    slice contains 704 inspected A/B/C/D samples with zero batch-inspection
    failures; the Cosmos condition root uses the B/D subset only:
    `joint_cosmos_condition/train_slice01`, 512 RGB window rows
    (B=256, D=256), 93 frames, 92 action/state steps, raw action/state dim
    32, `strict_alignment_ok=true`. This is a larger trainable RGB condition
    root, but the next Cosmos training run is intentionally deferred while
    `01_dataset` expands new `prod02` B/C/D generation.
- [ ] Verify DP action loss and Cosmos future-state / latent loss separately.
- [ ] Verify action chunks and future-state chunks are temporally aligned.
- [ ] Add `scripts/training/train_joint_dp_cosmos_full.py` only after overfit
  evidence is correct.
- [ ] Run full training for at least `1 GPU x 1 hour` on real data before
  calling it training evidence.
