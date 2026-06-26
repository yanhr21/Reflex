# 2026-06-24 Cosmos Policy-DROID Pivot Gate

## Scope

This is a compute-node diagnostic for the post-scorer pivot. It checks whether
the current 733 clean-dense full-episode WAM data can drive a causal
Cosmos Policy-DROID live-prefix input, and whether the existing local
Policy-DROID inference wrapper can start from that input.

This is not controller or method evidence.

## Code Changes

- Added
  `scripts/slurm/run_cosmos_policy_droid_active_prefix_input_gate_in_allocation.sh`.
  The wrapper refuses login-node execution, targets the active 733 clean-dense
  roots by default, selects a `target_motion_observed` full-episode JSONL row,
  writes a prefix-only video inside the Slurm step, and calls the existing
  live-prefix builder.
- Fixed `scripts/world_model/build_cosmos3_live_prefix_wam_input.py` for the
  current `_prefix_payload` signature on the `SOURCE_H5` path. The previous
  call was stale and crashed before building the input when no live-history
  JSON was supplied.
- The wrapper passes the selected row's task caption as `PROMPT`, so the model
  receives the current prefix hole/peg/TCP state instead of a generic fallback
  prompt.

## Allocation

- tmux session:
  `contact_value_margin_gpux_1gpu_request_20260624_170413`
- Slurm job: `148680`
- Node: `server13`

No project-code compute was run on the login node.

## Input Gate Result

Successful input-only gate:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/cosmos_policy_droid_active_prefix_input_gate_iter_000001500_20260624_173015_alloc148680`

Selected row:

- dataset split/source: current 733 clean-dense validation JSONL
- UUID:
  `hole_late_move_stop_seed3280649_idx2518.fix3_traj_0__target_motion_observed_f106`
- prefix frame: `106`
- prefix role: `target_motion_observed`
- source H5:
  `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5/hole_late_move_stop_seed3280649_idx2518.fix3/hole_late_move_stop_seed3280649_idx2518.h5`
- prefix-only RGB video: `107` frames, frame `0..106`, `30 fps`
- builder output:
  `cosmos_live_prefix/live_prefix_input_manifest.json`

The key positive result is narrow but real: current active data can produce a
strict causal Policy-DROID live-prefix input without future RGB frames.

## Inference Probe

Probe output root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/cosmos_policy_droid_active_prefix_input_gate_iter_000001500_20260624_173040_alloc148680`

The probe was launched with `RUN_INFERENCE=true`, `INFERENCE_NUM_STEPS=30`, and
`ACTION_EXEC_HORIZON=8`.

Observed behavior:

- input construction succeeded;
- `live_prefix_inference.log` stayed empty for several minutes;
- after cancellation, Cosmos had just initialized logs and printed
  `Loaded 1 samples`;
- no `sample_outputs.json` or action chunk was produced.

The step was canceled with `scancel 148680.5`; the held allocation `148680` was
preserved.

This is a startup/runtime integration blocker, not an action-quality result.
It does not prove that Policy-DROID actions are bad. It only proves the current
single-sample inference wrapper needs a cleaner run-to-completion strategy
before action chunks can be evaluated.

## Completed 10-Step Inference Probe

Second probe output root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/cosmos_policy_droid_active_prefix_input_gate_iter_000001500_20260624_1735_10step_alloc148680`

This run used `RUN_INFERENCE=true`, `INFERENCE_NUM_STEPS=10`, and
`ACTION_EXEC_HORIZON=8`.

Result:

- input construction succeeded;
- Cosmos loaded the single sample;
- the 10-step sampler completed in about `56` seconds after startup;
- outputs were written:
  - `cosmos_live_prefix/inference/hole_late_move_stop_seed3280649_idx2518.fix3_traj_0__target_motion_observed_f106/sample_outputs.json`
  - `cosmos_live_prefix/inference/hole_late_move_stop_seed3280649_idx2518.fix3_traj_0__target_motion_observed_f106/vision.mp4`
  - `cosmos_live_prefix/live_prefix_action_chunk.json`

Extracted action chunk:

- chunk start/end: `106..114`
- chunk steps: `8`
- robot action dim: `7`
- all values finite
- denormalized action stats: min about `-1.016`, max about `0.372`,
  mean absolute value about `0.211`

This proves the active Policy-DROID path can generate a concrete live-prefix
robot action chunk. It still does not prove the action is useful; simulator
replay is required.

## Source-Prefix Replay Attempt

Added:

`scripts/world_model/replay_policy_droid_action_chunk_from_source.py`

Purpose: replay the extracted Policy-DROID chunk from the matching source H5
prefix state, then optionally run DP96 continuability labeling.

First attempt:

- output root:
  `experiments/world_model_task_rebinding/cosmos3/policy_droid_action_replay_source_prefix_20260624_1740_alloc148680`
- failed after loading the DP agent because the wrapper was missing
  `save_step_records`, which `replay_bank_candidate()` expects.

Second attempt:

- output root:
  `experiments/world_model_task_rebinding/cosmos3/policy_droid_action_replay_source_prefix_fix1_20260624_1742_alloc148680`
- failed before env construction completed because torch CUDA initialization
  failed inside the held allocation step:
  `CUDA unknown error ... Setting the available devices to be zero`.

Follow-up checks inside allocation `148680`:

- `nvidia-smi` sees the H200 GPU.
- `CUDA_VISIBLE_DEVICES=0`.
- `.venv/bin/python -c 'import torch; ...'` reports
  `torch.cuda.is_available() == False` with the same CUDA unknown error.
- `.venv_cosmos313` reports the same CUDA initialization warning.
- Changing `CUDA_VISIBLE_DEVICES`, `--gpus=1`, and `--gpu-bind=none` did not
  fix torch CUDA initialization.
- `SLURM_STEP_GPUS=2` while `CUDA_VISIBLE_DEVICES=0`; setting
  `CUDA_VISIBLE_DEVICES` to `2` or to the GPU UUID did not fix the failure.
- Clearing `LD_LIBRARY_PATH` and using `CUDA_MODULE_LOADING=EAGER` did not fix
  the failure.
- Explicit `torch.cuda.init()` fails in `.venv` with:
  `RuntimeError: CUDA unknown error ... Setting the available devices to be zero`.

This is a compute-environment blocker for ManiSkill/DP replay in the current
held allocation, not an action-quality result. CPU fallback is not accepted as
controller evidence for this project.

Because `gpux` only showed usable resources on `server13` while `server42` was
down, a fresh tmux-held request was opened on the broader `gpu` partition,
excluding `server13`: job `148732`, session
`policy_replay_cuda_repair_gpu_request_20260624_1801`. It was pending on
priority at the time of this note. No one-shot `sbatch` was used.

Later resource update: the bad `server13` allocation `148680` was released
after repeated torch-CUDA initialization failures made it unusable for the
Policy-DROID replay path. The broader `gpu` request `148732` remained pending
on priority. The next replay attempt should start from `148732` or another
CUDA-valid tmux-held interactive allocation, not from the released `148680`.

## Source-Prefix Replay Result On Server24

The broader `gpu` request started as allocation `148732` on `server24`.
Torch CUDA was valid there:

- `.venv` torch: `2.5.1+cu121`
- `torch.cuda.is_available() == True`
- GPU: `NVIDIA H200`

Policy-DROID chunk replay output:

`experiments/world_model_task_rebinding/cosmos3/policy_droid_action_replay_source_prefix_server24_20260624_1810_alloc148732`

Inputs:

- action chunk:
  `.../cosmos_policy_droid_active_prefix_input_gate_iter_000001500_20260624_1735_10step_alloc148680/cosmos_live_prefix/live_prefix_action_chunk.json`
- source H5:
  `experiments/world_model_task_rebinding/cosmos3/fix3_v7_dp_user_override_sft_source_20260612_733/canonical_h5/hole_late_move_stop_seed3280649_idx2518.fix3/hole_late_move_stop_seed3280649_idx2518.h5`
- prefix frame: `106`
- executed Policy-DROID steps: `8`
- DP rollout label horizon: `96`

Direct post-chunk result:

- `after_success=false`
- `after_inserted_live_pose=false`
- `after_grasped=true`
- `after_contact_stable_proxy=false`
- `after_continuability_gate.ok=false`
- before weighted error: `0.2354`
- after weighted error: `0.2906`
- `delta_abs_yz_sum=+0.0128`

DP96 label after the Policy-DROID chunk:

- `continuable=true`
- `success=true`
- DP executed `48` steps before success
- final peg-head-at-hole:
  `[-0.0149, 0.0030, -0.0010]`
- final grasped/inserted/contact-stable all true

Interpretation:

This single source-prefix replay does not show that Policy-DROID directly
inserts. The short chunk alone made the conservative gate worse and did not
enter the direct insertion/contact-stable state. However, the chunk preserved
grasp and led to a state from which frozen DP could finish within the DP96
label rollout. It is therefore useful as a handoff/action-prior diagnostic,
not yet as a contact-completion executor.

The next meaningful replay should use saved dynamic live failure snapshots,
not only the matching source H5 prefix state, because the research objective is
dynamic-scene task completion after reobservation.

## Saved Live-Snapshot Replay Diagnostic

Added:

`scripts/world_model/replay_policy_droid_action_chunk_from_snapshot.py`

Purpose: replay one Policy-DROID action chunk from a saved dynamic
`live_state_before_controller.h5` with the matching live action/state history,
then run the same DP96 continuability label. This is still diagnostic unless
the action chunk was generated from that exact live prefix and visual evidence
is reviewed.

An attempted same-snapshot live-prefix inference was launched on allocation
`148732`:

`.../policy_droid_live_snapshot_sample00_iter00_f106_10step_alloc148732/cosmos_live_prefix`

Input construction succeeded from the saved prefix video and live history for
`sample_00_hole_late_move_stop/iter_00_prefix_f106`. Cosmos inference was
mistakenly interrupted while it was still loading: logs reached model setup and
`Loaded 1 samples`, but no `sample_outputs.json` or action chunk was produced.
This is an operator run-control error / slow-startup observation, not a
Policy-DROID action-quality failure.

To avoid losing the replay gate, a clearly marked mismatched-prefix diagnostic
was then run: the previously generated source-prefix Policy-DROID chunk was
replayed from the saved dynamic live snapshot.

Output:

`experiments/world_model_task_rebinding/cosmos3/policy_droid_action_replay_live_snapshot_sourceprefix_mismatch_sample00_iter00_f106_20260624_alloc148732`

Direct post-chunk result from the live snapshot:

- `after_success=false`
- `after_inserted_live_pose=false`
- `after_contact_stable_proxy=false`
- `after_continuability_gate.ok=false`
- `after_grasped=true`
- before weighted error: `0.2874`
- after weighted error: `0.3402`
- `delta_abs_yz_sum=+0.01257`

DP96 label after the chunk:

- `continuable=true`
- `success=true`
- DP executed `67` steps before success
- final peg-head-at-hole:
  `[-0.0058, 0.0030, 0.0029]`
- final grasped/inserted/contact-stable all true

Interpretation:

This mismatched-prefix replay is not evidence that Policy-DROID generated the
right action for the saved live prefix. It does show that the live snapshot is
physically recoverable and that the Policy-DROID-style chunk does not
necessarily destroy grasp. The short chunk still does not directly insert and
actually worsens near-term task-frame error. The current blocker remains direct
contact/insertion action generation or adaptation, not impossibility of the
task.

## Same-Prefix Live-Snapshot Replay Result

The same saved live prefix was rerun to completion on allocation `148732`:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/policy_droid_live_snapshot_sample00_iter00_f106_10step_rerun_complete_alloc148732/cosmos_live_prefix`

Cosmos inference completed with `INFERENCE_NUM_STEPS=10`:

- input construction succeeded from the saved observed prefix/history;
- `sample_outputs.json` was written under
  `inference/policy_droid_snapshot_sample00_iter00_f106_sameprefix/`;
- extracted action chunk:
  `cosmos_live_prefix/live_prefix_action_chunk.json`;
- chunk frame range: `106..114`;
- robot action steps: `8`;
- all action values finite;
- denormalized action stats: min about `-1.032`, max about `0.437`,
  mean absolute value about `0.234`.

Same-prefix snapshot replay output:

`experiments/world_model_task_rebinding/cosmos3/policy_droid_action_replay_live_snapshot_sameprefix_sample00_iter00_f106_20260624_alloc148732`

Direct post-chunk result:

- `after_success=false`
- `after_inserted_live_pose=false`
- `after_contact_stable_proxy=false`
- `after_continuability_gate.ok=false`
- `after_grasped=true`
- before weighted error: `0.2874`
- after weighted error: `0.3052`
- `delta_abs_yz_sum=+0.00316`

DP96 label after the same-prefix chunk:

- `continuable=true`
- `success=true`
- DP executed `63` steps before success
- final peg-head-at-hole:
  `[-0.0021, 0.0025, -0.0018]`
- final grasped/inserted/contact-stable all true

Comparison to the earlier mismatched-prefix replay:

- same-prefix direct post-chunk result is still negative;
- same-prefix worsened y/z less (`+0.00316` vs `+0.01257`);
- same-prefix DP96 succeeded in fewer steps (`63` vs `67`);
- neither result is a direct insertion executor result.

Interpretation:

Policy-DROID from the real saved live prefix is operational and produces a
physically executable action chunk that preserves grasp. It improves the
handoff-prior picture relative to the mismatched replay, but it still does not
enter the insertion/contact-stable manifold by itself. The next method repair
must train/adapt a direct contact/insertion executor or switch to a stronger
contact-capable base policy; the Policy-DROID chunk can be used as action-prior
evidence, not as method success.

## Current Conclusion

The scorer/value-head line remains stopped as the main method. The active
problem is still action generation/execution:

- causal suffix diffusion can create some DP-continuable handoff states, but
  direct insertion/gate-ok remains `0`;
- selected value replay only improved DP96 success from `8/16` to `9/16`;
- Cosmos Policy-DROID is locally feasible at the input/action-schema level,
  but current source-prefix and saved live-snapshot replays show handoff
  usefulness rather than direct insertion execution.

## Next Options

1. Define direct
   insertion/contact-positive labels, because the
   existing live candidate labels contain no direct final-success positives and
   cannot train a real insertion executor by themselves.
2. Use the same-prefix replay outcome to decide whether Policy-DROID needs
   contact-positive fine-tuning, a residual/contact executor, or replacement
   by OpenPI/pi0/OpenVLA-style action models. Do not return to scorer-only
   margin tuning.
3. If Policy-DROID remains only handoff-positive after a few saved live
   snapshots, use local OpenPI as the next
   base-policy audit target, but first build a ManiSkill/PegInsertion data
   converter and action-normalization bridge; no such bridge currently exists
   in this repo.
