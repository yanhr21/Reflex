# Low-Frequency WM + Executor TODO

## Current Boundary

- [ ] This is a new proposed method branch. Do not mix its results with the old
      `cosmos3_300f_world_model` direct raw-Cosmos-action evidence.
- [ ] Current iter2700 status: implementation contract is repaired, but method
      effectiveness is false. Use it as failure evidence and query-coverage
      input, not as a successful checkpoint.
- [ ] No Slurm export, training, rendering, or rollout has been launched for
      this new branch yet.

## Do Not Do

- [ ] Do not resume old sampled-role v7_733 SFT as the repair.
- [ ] Do not run more broad panels from iter2700 as method evidence.
- [ ] Do not interpret dense receding as a deployment requirement to call
      Cosmos every 8 actions forever.
- [ ] Do not relax `C_pi` or use generated sidecars as handoff authority.
- [ ] Do not use 128/129-frame chunks, 93-frame diagnostics, cropped metrics,
      or hidden/manual target-motion prefixes.

## Step 1: Freeze The New Method Definition

- [ ] Record this branch in evidence notes before running jobs:
      low-frequency Cosmos task WM plus high-frequency executor.
- [ ] Define runtime variables:
      Cosmos call interval, executor chunk length, replan triggers, and latency
      reporting fields.
- [ ] Define which signals Cosmos predicts:
      future target/hole path, peg-head-in-hole-frame path, TCP/peg relation,
      grasp/contact/insertion risk, and optional coarse action hints.
- [ ] Define executor inputs:
      current live observation/state, DP action prior, predicted task path,
      current peg-hole/TCP/grasp predicates, and recent actions.
- [ ] Define executor outputs:
      short executable robot action chunks with action-space clipping recorded.

## Step 2: Clean Dense Condition Preflight

- [ ] After explicit user approval, run only the clean/dense preflight inside
      a compute allocation:

      `ALLOW_CLEAN_DENSE_PREFLIGHT=true bash scripts/slurm/run_cosmos3_300f_full_episode_wam_fix3_v7_733_clean_dense_preflight_in_allocation.sh`

- [ ] Confirm preflight creates a new clean/dense condition root, not a new
      SFT checkpoint.
- [ ] Confirm every row preserves `301` frames and `300` action steps.
- [ ] Confirm `prefix_role` equals physical mode and sampled/curriculum role is
      stored only as provenance.
- [ ] Confirm dense receding prefixes cover target-motion onset, post-motion
      correction, late held-peg states, and insertion/handoff states.
- [ ] Confirm live-query coverage audit passes or records exact undercovered
      query modes. If it fails, do not train.
- [ ] Confirm future object states are target/readout supervision only, not
      controller-facing privileged conditions.

## Step 3: Two-Sample Sanity

- [ ] Build a clean/dense two-source condition root from the approved data.
- [ ] Run preflight only first; require a matching
      `clean_dense_preflight_summary.json` with `ready_for_overfit=true`.
- [ ] After explicit user approval, run two-sample overfit SFT.
- [ ] Inspect generated videos/contact sheets directly.
- [ ] Check action chunks on both samples:
      predicted robot action rows must be finite, correctly aligned, and able
      to reproduce the teacher direction/scale on the overfit samples.
- [ ] If two-sample sanity fails, debug export/training/action extraction. Do
      not move to full training.

## Step 4: Executor Interface

- [ ] Add an executor training dataset builder from clean/dense rows.
- [ ] Include DP prior action chunks from the frozen static DP as input, not as
      the only controller.
- [ ] Include Cosmos-predicted task path or task-frame target as input.
- [ ] Include current peg-hole/TCP/grasp state from causal history.
- [ ] Train the executor to output short chunks that reduce real peg-hole error
      while preserving grasp.
- [ ] Keep teacher/source action targets separated from diagnostic sidecar
      state. Robot action dims must be reported separately.
- [ ] Add a two-sample executor overfit check before full executor training.

## Step 5: Full Training

- [ ] Train the low-frequency Cosmos task WM on clean/dense conditions only
      after preflight and two-sample sanity pass.
- [ ] Train the executor or DP-prior residual policy after its two-sample sanity
      passes.
- [ ] Record all config: prefix source, dense stride, late-rebind weighting,
      action-loss recipe, executor chunk length, and DP-prior source.
- [ ] Keep training on Slurm GPU nodes. Do not run heavy export/render/training
      on the login node.

## Step 6: Offline Gates

- [ ] Strict generated artifact check:
      full `301/300`, no hidden truncation, no future leakage.
- [ ] Task-state readout:
      target/hole path, peg-head-in-hole-frame path, grasp/contact/insertion
      predicates.
- [ ] Action/executor audit:
      compare predicted chunks to source/teacher chunks on late-rebind rows.
- [ ] Live-query coverage re-audit:
      failed iter2700 query states should now have local physical-mode
      neighbors in the clean/dense condition distribution.
- [ ] Runtime proxy:
      estimate how often Cosmos would be called under the low-frequency
      schedule versus every-8-step raw-action mode.

## Step 7: Closed-Loop Evaluation

- [ ] Start with a small val panel only after offline gates pass.
- [ ] Use the new runtime contract:
      low-frequency Cosmos update, high-frequency executor chunks, real-state
      `C_pi`, and same detector for moving/no-motion cases.
- [ ] Record for every sample:
      number of Cosmos calls, frames between calls, executor chunk count,
      DP handoff chunks, final success, and video contract.
- [ ] Compare against same-source full pure DP.
- [ ] Do not proceed to hard screens if val is degraded versus pure DP.
- [ ] Hard screen is valid only after val is not worse than pure DP and at
      least one moving-target class shows clear benefit.

## Step 8: Method Success Criteria

- [ ] Full `300` actions / `301` frames.
- [ ] Causal target-motion detection.
- [ ] No future privileged object state as controller input.
- [ ] Cosmos active on moving-target cases only when detector/replan triggers.
- [ ] Executor preserves grasp and improves peg-hole alignment.
- [ ] DP handoff occurs only from real-state `C_pi` pass.
- [ ] Same-source pure-DP comparison is not worse on val.
- [ ] Hard pure-DP failures are rescued at a useful fraction, not only `1/6`.
- [ ] Direct video/contact-sheet inspection agrees with metrics.
- [ ] Runtime report shows the number of Cosmos calls is plausible for
      deployment or clearly marks the result as a simulator-only diagnostic.

## Fallback Decision

- [ ] If clean/dense direct Cosmos raw actions still fail after a fair full
      run, stop treating raw Cosmos actions as the controller.
- [ ] Switch to executor-first control:
      Cosmos predicts task state/coarse plan at low frequency, and the learned
      executor or DP-prior residual policy owns high-frequency robot actions.
- [ ] If the executor also fails to preserve grasp/insertion on inspected
      videos, stop for user direction with concrete failure artifacts.
