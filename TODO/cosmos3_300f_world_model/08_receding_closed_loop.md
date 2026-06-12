# Receding Closed-Loop TODO

## Current Gate

- [ ] Do not start live DP/controller evaluation from the current v7_733
      follow-up SFT until a checkpoint passes all three gates:
      strict same-length generated artifacts, generated-RGB readout/profile,
      and direct visual review of all validation sheets/videos.
- [x] The latest fix1-recipe root
      `sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745`
      ended at Slurm wall time after rank-0 iteration `743`. It saved only
      `iter_000000300` and `iter_000000600`; no iter900/iter1200 checkpoint or
      watcher exists for this root.
- [x] The already evaluated latest fix1-recipe `iter_000000300` and
      `iter_000000600` checkpoints are not controller-ready. They preserve
      prefix conditions and pass the 301-frame / 300-action structural gate,
      but future action/state/video quality and visual peg/contact continuity
      remain negative. Iter300 is the best qualitative sanity checkpoint so
      far; iter600 is worse despite lower validation loss.
- [x] Historical `normactive_clip1` iter900/iter1200 notes are not the active
      closed-loop gate. That run was rejected because it did not use the
      overfit-approved fix1 action recipe.

## Required Execution Contract

- [x] Add a conservative closed-loop gate checker:
      `scripts/world_model/check_cosmos3_closed_loop_gate.py`. It reads a
      Cosmos eval root, strict artifact inspection, generated-RGB
      readout/profile, and an explicit visual-review verdict. It blocks by
      default unless all three gates pass and should be called by any future
      live closed-loop wrapper before it touches DP or the simulator.
- [ ] The future closed-loop wrapper must run only inside a Slurm compute-node
      allocation. It must refuse login-node execution, like the current
      Cosmos eval/readout watchers.
- [ ] Use the frozen static DP checkpoint only through its real ManiSkill
      state-policy interface:
      `experiments/dp_peg1000/run_90201/checkpoints/best_eval_success_at_end.pt`.
      Its manifest fixes `PegInsertionSide-v1`, `pd_ee_delta_pose`,
      `obs_horizon=2`, `act_horizon=8`, and `max_episode_steps=300`.
- [ ] Observe live RGB/state, build a causal Cosmos prefix from the latest
      observed frames/state, predict the remaining or short future horizon,
      execute only a short action prefix, then reobserve and refresh. The
      default action execution prefix should be `<=8` steps to match DP
      `act_horizon`; never execute a one-shot 300-step open-loop Cosmos
      trajectory as method evidence.
- [ ] The authority for success is the live simulator final state plus video
      review. A restored planner state, generated RGB, or readout-only success
      is diagnostic, not controller evidence.

## Cosmos Action Contract

- [ ] Cosmos WAM eval output is a normalized `300x32` sequence. Columns
      `0..6` are robot actions; columns `7..31` are predicted task-state
      sidecars.
- [ ] Before live execution, load
      `full_episode_wam_conditions_fix3_v7_733_rgb_300step_20260612_0245/normalization_stats.json`
      and de-normalize only columns `0..6` with the matching vector-name
      stats. Clip/validate against the ManiSkill action space before
      `env.step`.
- [ ] Treat columns `7..31` as predicted task-state diagnostics and controller
      scoring/readout context only. They must not be written into simulator
      state, used as oracle object poses, or used to bypass RGB/state
      re-observation.
- [ ] Preserve the 301 RGB/state frame and 300 action-step accounting in all
      manifests. A receding controller may execute short prefixes, but the
      generated/evaluated sample contract must not become 128/129-frame clips.

## Implementation Steps After A Passed Gate

- [ ] Add a compute-node-only wrapper, e.g.
      `scripts/slurm/run_cosmos3_receding_closed_loop_in_allocation.sh`, that
      records job id, checkpoint path, condition root, DP checkpoint, action
      normalization stats, validation seeds/scenarios, and evidence boundary.
      Current active-script audit found no existing active receding wrapper;
      this must be implemented or deliberately restored only after an SFT
      checkpoint passes the generated artifact/readout/visual gate.
- [ ] Add a Python entry point, e.g.
      `scripts/world_model/run_cosmos3_receding_closed_loop.py`, with an
      initial one-env smoke mode:
      reset dynamic ManiSkill episode, maintain real observation history,
      call the Cosmos full-episode policy inference for a causal prefix,
      de-normalize and execute at most `8` robot actions, reobserve, and
      repeat until termination or `300` steps.
- [ ] Save for every rollout: live RGB video, per-step executed robot action,
      generated action/state sidecars, real simulator metrics, target/peg/TCP
      readout trajectory, final success predicates, and a review sheet.
- [ ] Run a tiny compute-node smoke first. It passes only if length accounting,
      action de-normalization, live `env.step`, video recording, and final
      metrics all complete without using sidecar/oracle state.
- [ ] Only after the smoke passes, run the fixed scenario-diverse validation
      panel from the testing plan: static/none, pre-motion target forecast,
      observed target motion, move-stop, reverse, peg disturbance, and
      peg-drop/regrasp.

## Negative Cases To Preserve

- [ ] If the passed SFT gate never happens, keep this as an implementation
      plan only. Do not force a closed-loop run from a weak checkpoint.
- [ ] If a live closed-loop rollout loses the peg, visibly inserts into the
      wall, relies on target self-insertion, or disagrees with its metric, mark
      it negative and keep it out of positive controller evidence.
- [ ] If the same implementation blocker repeats after concrete log/artifact
      inspection and no safe aligned fix is clear, preserve held resources when
      possible and stop for user direction rather than trying random variants.
