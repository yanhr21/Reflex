# 2026-06-23 Current Review And Blockers

## Scope

This review follows the contact-action reset: scorer-only action selection is
not the active method. Login-node work in this review was limited to read-only
inspection and documentation edits. The first reset training run completed
inside the tmux-held Slurm allocation `146658` on `server56`.

## Current Progress

- New active plan/TODO/idea branch exists under
  `PLAN/contact_action_world_model/`, `TODO/contact_action_world_model/`, and
  `IDEA_contact_action_world_model.md`.
- Superseded Cosmos3 smoke/canary/debug/scorer/retrieval directories were
  moved under `experiments/_archive_20260623_contact_action_reset/`; the
  manifest records `659` moved directories. The active Cosmos3 top-level tree
  is down to `119` directories.
- Local Cosmos Policy-DROID assets exist:
  `checkpoints/cosmos3/Cosmos3-Nano-Policy-DROID`,
  `Cosmos3-Nano-Policy-DROID-DCP`, `Cosmos3-Nano-DCP`, and `Cosmos3-Nano`.
- Local `openpi` source exists at `/public/home/yanhongru/ICLR2027/openpi`.
  It offers pi0/pi0.5-style policy fine-tuning, but this repo does not yet
  have a ManiSkill PegInsertion data adapter, normalization config, or eval
  wrapper for it.
- The first contact-action reset training run completed:
  `experiments/world_model_task_rebinding/cosmos3/contact_action_suffix_generator_full733_1gpu1h_20260623_163847_alloc146658`.
  It used `4711` source-suffix rows with a `4001/710` source-UUID train/val
  split, reached the one-GPU-hour floor, and failed the saved-snapshot replay
  readiness gate.

## Current Experimental Results

Latest strict safegate live panel:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_safegate1_source_suffix_offsets64_48_32_24_panel0134_20260623_alloc146658`

- `completed_samples=4`
- `failed_process_count=0`
- `panel_full_episode_contract_ok=true`
- `final_success_count=0`
- `method_evidence_allowed=false`
- Each inspected final video has `301` frames at `30 fps`.

Final peg-head positions in the hole frame:

- sample 0, `hole_late_move_stop`: `[-0.0971, 0.0144, -0.0716]`
- sample 1, `hole_late_constant`: `[-0.0922, 0.0329, -0.0782]`
- sample 3, `hole_late_fast_shift`: `[-0.1056, 0.0014, 0.0039]`
- sample 4, `hole_late_sine`: `[-0.1040, 0.0306, -0.0325]`

This is a physical insertion failure: the controller often gets laterally near
the hole but remains roughly `9-10.5 cm` short along the insertion axis.

The previous panel0245 scorer variants show partial but unstable success:

- h2048: `2/4`, success on `hole_late_reverse` and `hole_late_sine`.
- h8192: `2/4`, success on `hole_late_move_stop` and
  `hole_late_continuous_insert`.
- h16384 positive-weight: `2/4`, success on `hole_late_move_stop` and
  `hole_late_continuous_insert`.

The different scorers solve different samples, so this is not a robust
selection rule.

Offline scorer diagnostics over saved labels:

- eval groups: `21`
- DP-prior handoff successes: `3`
- handoff oracle successes in the current candidate pool: `7`
- selected handoff successes: `5-6`, depending on scorer
- harmful switches remain high: `7-10`
- top-1 oracle match is low: about `4.8%-14.3%`

This confirms two limits at once: DP alone is weak, and the available candidate
pool/scorer pair still lacks reliable insertion-action coverage.

## Current Training Status

At `2026-06-23 17:40 CST`, the active source-suffix generator run completed
and wrote `training_summary.json`.

Observed training behavior:

- best validation MSE so far occurred at step `200`:
  `eval_action_mse=0.012315` versus mean-action baseline `0.015566`.
- later training strongly overfits: train action MSE drops near `2e-5`, while
  eval action MSE returns to or worsens past the mean-action baseline.
- by `1950` seconds, `eval_action_mse=0.015930` while the mean-action
  baseline is `0.015566`, with `train_action_mse=1.7e-5`.
- by `2559` seconds, `eval_action_mse=0.016008`, still worse than the
  mean-action baseline, with `train_action_mse=2.7e-5`.
- by `3176` seconds, `eval_action_mse=0.016166`, worse still, with
  `train_action_mse=1.4e-5`.
- final summary: `formal_one_gpu_hour_floor_met=true`,
  `elapsed_seconds=3661.45`, `steps=567001`,
  `stop_reason=min_wall_and_min_steps`, and
  `ready_for_saved_snapshot_replay_gate=false`.
- final validation MSE was `0.0162008`, worse than the mean-action baseline
  `0.0155657`, while final train action MSE was `1.3e-5`.

This is a formal negative diagnostic for the first deterministic source-suffix
MLP. It should not be promoted to live evaluation. Saved-snapshot replay should
be skipped unless a negative diagnostic is explicitly useful, because the
completed training summary already marks the checkpoint not ready for the
replay gate.

The second reset training run also completed:

`experiments/world_model_task_rebinding/cosmos3/live_outcome_action_diffusion_full_live_union_1gpu1h_20260623_175110_alloc146658`

It trained a live-outcome-conditioned diffusion residual generator plus
consequence/value heads. A read-only data audit found `9349` outcome rows over
`127` live states, but only four distinct source trajectories. The labels have
`0` direct final successes and `0` inserted-pose positives; the positive rows
are `candidate + DP96` handoff successes. At `2026-06-23 18:27 CST`, the run
was still below the one-GPU-hour floor and the held-out selection metric was
`0.36` versus DP prior `0.32`, with oracle `0.44`. This is an in-progress
training signal only, not method evidence.

At `2026-06-23 18:54 CST`, the live-outcome diffusion run completed. It met
the formal one-GPU-hour floor (`elapsed_seconds=3660.23`) but failed the replay
readiness gate: `ready_for_saved_snapshot_replay_gate=false`. Final held-out
selection tied DP prior (`0.32` selected versus `0.32` DP), while oracle over
the existing replayed candidates was `0.44`. The best value-MSE checkpoint was
at step `1`, and final value MSE was `14.38`, so the value/rank head did not
learn a stable held-out selector. This checkpoint should not be promoted to
saved-snapshot replay or live panel evaluation except as an explicitly marked
negative diagnostic.

## Main Blockers

1. Candidate coverage is the core blocker. A scorer can only choose among
   existing chunks; it cannot synthesize the missing insertion-axis/contact
   behavior.
2. Frozen DP is not reliably continuable after dynamic target motion. It can
   finish some handoff states but fails many states that look geometrically
   close.
3. The current `C_pi`/geometry gate is not a sufficient training target.
   Saved replay already shows both false positives and false negatives.
4. Contact labels are too weak. The repo has final success, weighted geometry,
   and some proxy progress labels, but not enough contact stability,
   insertion-force/compliance, or grasp-hold supervision.
5. The current source-suffix bank may be scenario/timing specific rather than
   a reusable task-frame insertion primitive.
6. The first suffix generator is not yet method-valid: it conditions on
   `scenario_onehot` from source/sample metadata, which is acceptable only for
   diagnostic replay. A real controller must replace this with causal
   observed-history motion/contact features.
7. The first suffix generator is also a deterministic MLP/MSE regressor over a
   multimodal action distribution. That tends to average incompatible suffixes
   and then overfit, instead of generating diverse contact-feasible chunks.
8. Cosmos Policy-DROID is locally available but only has an inference/action
   extraction path in this repo. The missing work is post-training on the 733
   contact-action data plus a value/progress head.
9. OpenPI/pi0.5 is a credible fallback, but the current local repo lacks the
   ManiSkill observation/action mapping, LeRobot or custom dataset conversion,
   normalization stats, and evaluation wrapper.
10. Some summary schemas differ across scripts. Panel summaries are reliable,
    but single-sample keys must be read from each schema rather than assumed.
11. The live-outcome diffusion data currently has no direct insertion-positive
    rows. It optimizes DP handoff/continuability, not direct insertion
    completion. That is aligned as a diagnostic but insufficient as the final
    answer to why the peg will not go in.
12. The live-outcome data comes from only four source trajectories, so a
    UUID-level split over live states is not a strong source-level
    generalization test.
13. The current saved-snapshot replay script does not yet know how to sample
    from the live-outcome diffusion checkpoint. A positive offline training
    metric would still require a new generated-action replay path before any
    live panel.
14. Both reset training attempts are now negative/limited: the deterministic
    source-suffix MLP overfit and underperformed the mean-action baseline, and
    the live-outcome diffusion run tied DP prior after one GPU-hour. This makes
    it unlikely that another scorer-sized model over the same weak labels will
    solve physical insertion.

## Next Actions

1. Do not promote the first MLP final checkpoint to live evaluation; its
   completed summary says `ready_for_saved_snapshot_replay_gate=false`.
2. Run saved-snapshot replay for that checkpoint only if an explicit negative
   diagnostic is needed. It should not be a method gate or live precursor.
3. Treat any replay from the current checkpoint as diagnostic only because of
   the current `scenario_onehot` feature. It can tell whether the generator has
   any action coverage; it cannot be method evidence.
4. Replace non-causal scenario conditioning with causal observed-history
   descriptors: target-motion type inferred from state/history, task-frame
   relative motion, contact phase, grasp proxy, and recent action history.
5. Move from deterministic MLP/MSE to a diffusion/flow/contact-action
   generator or Cosmos Policy-style action/value/video post-training.
6. Add hard negative and positive labels from `candidate + DP96` replay:
   final success, DP96 success/continuability, contact stability, grasp
   preservation, and insertion-axis progress.
7. Retarget the Cosmos Policy-DROID live-prefix/action sidecar path to the
   active 733 clean-dense roots, then add missing action/value training.
8. If saved-snapshot replay shows no generated candidate can beat DP prior,
   start the fallback base-policy integration audit for OpenPI/pi0.5 first,
   then Octo/OpenVLA only if downloads and adapters are justified.
9. After the two reset training negatives, prioritize either a Cosmos
   Policy-style action/value/video post-training path or a new contact-action
   dataset with direct insertion/contact positives. Do not spend the next turn
   tuning scalar scorer thresholds on the same candidate pool.
