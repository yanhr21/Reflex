# 2026-06-26 OpenPI pi0.5 Contact-Suffix Training Evidence

## Summary

The OpenPI/pi0.5 contact-suffix branch now has a valid trained checkpoint.
This is training/checkpoint evidence only, not replay insertion success.

## Data

- Source: accepted 733 ManiSkill PegInsertionSide H5 trajectories.
- Contact-suffix LeRobot repo id:
  `yanhongru/maniskill_peg733_openpi_contact_suffix16`.
- Conversion output:
  `experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix_lerobot_full733_20260625_alloc150773`.
- Count: `5853` suffix episodes, `93648` frames, suffix length `16`.
- Audit:
  `experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix_lerobot_full733_20260625_alloc150773/audit_summary.json`.
- Audit result: passed, `failures=[]`, `unique_episode_lengths=[16]`,
  image/wrist `256x256x3`, state dim `8`, action dim `7`.

## Training

- Config: `pi05_maniskill_peg733_contact_suffix16`.
- Model: official OpenPI `Pi0Config(pi05=True, action_horizon=16,
  discrete_state_input=False)`.
- Base weights: official `pi05_base/params`.
- Runtime: held Slurm allocation `150773`, step `44`, host `server38`.
- Steps: `1700`.
- Walltime: `4303` seconds.
- One GPU-hour floor: met.
- Return code: `0`.
- Summary:
  `experiments/world_model_task_rebinding/openpi/pi05_peg733_contact_suffix16_noema_direct1700_1gpu1h_20260626_alloc150773/training_walltime_summary.json`.
- Loss evidence: step `0` loss `0.1038`; step `1600` loss `0.0373`.

## Checkpoint

- Final step: `1699`.
- Node-local source:
  `/tmp/openpi_pi05_contact_suffix_checkpoints_yanhongru_150773_local_noema/pi05_maniskill_peg733_contact_suffix16/pi05_peg733_contact_suffix16_noema_direct1700_1gpu1h_20260626_alloc150773/1699`.
- Preserved checkpoint:
  `experiments/world_model_task_rebinding/openpi/checkpoints_local_preserved/pi05_maniskill_peg733_contact_suffix16/pi05_peg733_contact_suffix16_noema_direct1700_1gpu1h_20260626_alloc150773/1699`.
- Copy verification: source and destination both `31G`, `71` files.

## Blocker Fixed

The first suffix training attempt reached step `1000` but Slurm reported
`OUT_OF_MEMORY` during Orbax checkpoint save, leaving only an incomplete
`1000.orbax-checkpoint-tmp-*` directory. The successful run set
`ema_decay=None` for the suffix config and used `--save-interval=5000`, so the
run skipped mid-training saves and checkpointed only at the final step. This
keeps the model official OpenPI/pi0.5 while reducing checkpoint memory.

## First Replay Gate

Saved-snapshot replay from preserved checkpoint `1699` completed inside Slurm:

- Output:
  `experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix1699_panel4_exec16_alloc150773`.
- Execute steps: `16`.
- Labels: `4`.
- Direct success: `0/4`.
- Inserted: `0/4`.
- Contact-stable proxy: `0/4`.
- Grasp preserved: `4/4`.
- DP96 continuable: `1/4`.
- DP96 success: `0/4`.

This is negative task evidence. The suffix policy can be trained and loaded,
and it preserves grasp on this panel, but it still does not produce direct
insertion/contact-stable states from the tested dynamic snapshots.

## Next Gate

The first mismatch diagnosis is complete:

- Diagnostic script:
  `scripts/openpi/diagnose_openpi_contact_suffix_replay.py`.
- Baseline replay diagnosis:
  `experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix1699_panel4_exec16_alloc150773/contact_suffix_replay_diagnosis.json`.
- Result: `4/4` samples worsened `abs_yz_sum`; `3/4` worsened `abs_x`.
  Mean `delta_abs_yz_sum=0.03767`; mean `delta_abs_x=0.02172`.
- Tested snapshot offsets to source first insertion were `[60, 60, 50, 66]`.
  None exactly matched the fixed training offsets
  `[64, 48, 32, 24, 16, 12, 8, 4]`, but they are near the trained range, so
  offset mismatch alone is not a sufficient explanation.
- Predicted actions have high same-time source-action cosine
  (`0.966..0.981`) but usually over-amplify lateral motion; the model is
  reproducing a source-style action direction without reliably binding it to
  the live dynamic hole frame.

A privileged prompt-phase diagnostic was also run:

- Output:
  `experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix1699_panel4_exec16_privprompt_diag_alloc150773`.
- Boundary: diagnostic only; prompts used source first-insert timing and are
  not causal method evidence.
- Result: direct success `0/4`, inserted `0/4`, contact-stable `0/4`, grasp
  `4/4`, DP96 continuable `1/4`, DP96 success `0/4`.
- Diagnosis JSON:
  `experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix1699_panel4_exec16_privprompt_diag_alloc150773/contact_suffix_replay_diagnosis.json`.
- It still worsened `abs_x` in `4/4` and `abs_yz_sum` in `3/4`, so merely
  matching the training prompt's suffix-phase text does not fix insertion.

Current diagnosis: the main missing piece is not just training duration,
checkpoint validity, or prompt phase. The OpenPI observation/action interface
does not expose enough causal object/task-frame geometry for the policy to
bind its learned insertion suffix to the moved hole. The likely next repair is
an OpenPI-native data/config transform that adds causal RGB-derived or
state-derived object/task-frame features to the policy state, with matching
normalization and replay evaluation. This should stay inside official OpenPI
model/config machinery and must not become a custom VAE/MLP/diffusion executor
or a scorer-only selector.

## Follow-Up Object-State Diagnostic

The first object/task-frame conditioning follow-up is recorded separately in:

`docs/world_model_task_rebinding/openpi_pi05_contact_action/2026-06-26_object17_conditioning_diagnostic.md`

Short version: object17 data/config/norm stats now exist and pass audit, but
official OpenPI training stalls after split generation and before `Step 0`.
It is therefore not yet a trained model or insertion result.

Later correction: the object17 rewrite path was found to have used hardlinks
and contaminated the canonical qpos8 contact-suffix repo. A qpos8 repair
conversion was then cancelled near completion, leaving the current canonical
repo partial. The preserved qpos8/contact-suffix checkpoints remain historical
training artifacts, but the current canonical LeRobot repo on disk must not be
used for new norm stats, training, or replay until rebuilt and audited.
