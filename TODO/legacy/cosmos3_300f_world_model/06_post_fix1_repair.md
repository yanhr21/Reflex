# Post-Fix1 Repair TODO

## Current Evidence Boundary

- [x] Fresh full1000 fix1 completed strict full-episode SFT/eval/readout/visual
      review through `iter_000001500`.
- [x] The final checkpoint is SFT progress evidence, not controller evidence.
      Target switching and peg/contact continuity remain negative.
- [x] Current fresh target-motion head diagnostic was run in Slurm allocation
      `123499`, step `12`, not on the login node. Generated RGB reached AUROC
      `0.7808262378`, F1@0.5 `0.6005305040`, best F1 `0.6179090483`; held-out
      reference RGB remains AUROC `0.9115353604`, F1@0.5 `0.7669897596`, best
      F1 `0.7788987602`.

## Failure Localization

- [x] Summarize role-wise errors in
      `docs/world_model_task_rebinding/2026-06-10_cosmos3_fresh_fix1_failure_localization.md`.
- [x] Identify current dominant failures:
      target_pre_motion final-hole error, target_post_motion robot-action
      error, peg_recovery contact/peg-head-hole error, and static false target
      onset.
- [x] Inspect the current full-episode condition exporter and SFT wrapper to
      determine where explicit target-onset/final-target, robot-action, and
      contact/grasp losses can be added without breaking the full 301/300
      contract.
- [x] Audit role sampling counts in the current full1000 condition root and
      decide whether role-balanced weighting can be added without regenerating
      source data or creating 128-action chunks.
- [x] Draft the concrete fix2 training config only after the two audits above:
      loss heads/weights, optimizer keys, role weights, validation gates, and
      expected evidence. Do not start fix2 SFT until the config is explicit.

## 2026-06-10 Code Audit Result

- [x] The full-episode SFT dataloader does honor JSONL
      `condition_frame_indexes_vision` and `condition_frame_indexes_action`.
      This means fix1 did use causal prefix masks; the bad full result is not
      explained by a silent fallback to single-frame I2V.
- [x] The dominant training-target bug is in the full-episode exporter:
      the 32-dim action/state token used real robot action in dims `0..6`, but
      dims `7..31` repeated the prefix TCP/peg/hole/contact state for every
      future step. Therefore Cosmos was not directly supervised to output
      future target/peg/TCP/contact state in its action branch.
- [x] Patched
      `scripts/world_model/export_cosmos3_maniskill_full_episode_wam_conditions.py`
      with `sidecar_target_mode=future_aligned_state`. Action row `i` now
      aligns to video frame `i+1`: history rows before the causal prefix are
      clean conditions, and rows after the prefix are generated future
      action/task-state targets.
- [x] Patched
      `scripts/slurm/run_cosmos3_300f_full_episode_wam_in_allocation.sh` to
      pass `SIDECAR_TARGET_MODE=future_aligned_state` explicitly and record it
      in the SFT manifest.
- [x] Patched `scripts/world_model/audit_cosmos3_action_targets.py` to fail
      future-aligned condition roots if task-state sidecars are still
      non-varying constants.
- [x] Added
      `scripts/world_model/build_cosmos3_role_weighted_sft_jsonl.py` for
      optional role-balanced row repetition. This does not regenerate data or
      create clips; it only repeats full 301/300 JSONL rows.
- [x] Local lightweight checks passed:
      `python3 -m py_compile` on the modified Python scripts,
      `bash -n` on the Slurm wrapper, and `.venv` import check that both
      vector layouts are 32 dim with robot actions in dims `0..6`.

## Role Audit

- [x] Current fix1 condition root has `3808` train rows and `384` val rows.
      Train roles: `target_pre_motion=753`, `target_motion_observed=753`,
      `target_post_motion=753`, `insert_resume=909`, `peg_recovery=322`,
      `static_monitor=159`, `static_late_monitor=159`.
- [x] The rare/failed roles are exactly the ones needing more sampling:
      `peg_recovery` for peg/contact continuity and static monitor rows for
      false target-onset suppression.

## Fix2 Candidate Config

- [x] Data source: keep the approved full1000 RGB 301-frame dataset; do not
      re-render or regenerate source videos.
- [x] Condition export: new condition root using the patched exporter,
      `sidecar_target_mode=future_aligned_state`, `301` RGB/state frames,
      `300x32` action/state rows, `joint_policy_history_action`, causal
      history action mask `0..prefix_frame-1`, and causal vision latent prefix.
- [x] Optional train weighting:
      `target_pre_motion=2,target_motion_observed=1,target_post_motion=2,insert_resume=2,peg_recovery=3,static_monitor=3,static_late_monitor=3`.
      Validation stays unweighted.
- [x] SFT hyperparameters should start from the full1000 fix1 stable values:
      4 GPUs, `MAX_ITER=1500`, `SAVE_ITER=300`, validation every `100` if
      affordable or `300` otherwise, `LR=1e-4`, warmup `10`,
      `ACTION_LOSS_WEIGHT=2.0`, `NORMALIZE_LOSS_BY_ACTIVE=true`,
      `INDEPENDENT_ACTION_SCHEDULE=true`, `SHIFT_ACTION=1`, optimizer keys
      include `action2llm,llm2action,action_modality_embed`.
- [ ] Slurm-side preflight required before SFT:
      strict 301/300 length check, action target audit with
      `future_aligned_state_rows_with_task_sidecar_variation == future_aligned_state_rows`,
      and role-weighted manifest if weighting is enabled.
- [ ] Do not use fix2 as controller evidence until generated validation passes
      equal-length video/action inspection, generated-RGB readout/profile,
      target-onset calibration, and visual peg/contact review.

## Controller Gate

- [ ] Do not start controller/DP integration from fresh fix1 `iter_000001500`.
- [ ] Reconsider controller only after a future checkpoint passes generated-RGB
      target-monitor calibration, role-wise action metrics, and visual
      peg/contact continuity.
