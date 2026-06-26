# 2026-06-23 Causal Suffix Value-Head Launch

## Purpose

The causal suffix diffusion generator produced candidates that can make saved
live failure snapshots DP-continuable, but the raw generated 8-step chunks did
not directly insert or pass the post-chunk continuability gate. The next
physical problem is therefore candidate choice under the same causal live
state: choose a generated contact-action prefix that puts the peg into a state
where the real frozen DP can finish, without claiming that a scorer alone is
the method.

This value head is a consequence model over real replay labels. It is not live
controller evidence. A controller claim still requires selected-candidate
saved-snapshot replay, then full live closed-loop execution with final-state
and video/contact review.

## Inputs

- Causal generated replay:
  `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_causal_suffix_diffusion_panel0134_offsets64_48_32_24_16_8_s2_exec8_dp96_fix1_20260623_201146_alloc146658`
- Same-snapshot DP-prior baseline replay:
  `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_dp_prior_panel0134_exec8_dp96_20260623_204147_alloc146658`
- Same-namespace DP + causal conversion:
  `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_causal_suffix_diffusion_panel0134_exec8_dp96_20260623_204543_alloc146658`
- Broader old live-outcome union:
  `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_source_suffix_union_plus_panel0245_plus0204fixuuid_dp96_20260623_alloc146658`
- Final merged training root:
  `experiments/world_model_task_rebinding/cosmos3/contact_value_training_union_plus_panel0134_causal_suffix_20260623_204657_alloc146658`

## Label Evidence Before Training

DP-prior baseline replay on the same 16 panel0134 live snapshot groups:

- records: `16`
- valid records: `16`
- failure counts: `{}`
- direct post-chunk success: `0`
- direct post-chunk gate-ok: `0`
- DP96 success: `8/16`
- DP96 continuable/contact-stable: `8/16`

Merged DP + causal conversion:

- valid rows: `208`
- base groups: `16`
- DP-prior groups: `16`
- causal suffix diffusion rows: `192`
- causal suffix diffusion groups with DP96 success: `16/16`
- total DP96 successes in the converted panel set: `63/208`

Merged training root:

- groups: `143`
- joined outcome rows: `9557`
- missing base rows: `0`
- groups with DP prior: `143`
- groups with causal suffix diffusion: `16`
- groups with causal suffix diffusion DP96 success: `16`
- groups with DP-prior DP96 success: `33`

## Launch

Started inside the held tmux/Slurm allocation:

- Slurm job: `146658`
- Slurm step: `146658.158`
- host: `server56`
- tmux window: `train_value_20260623_204725`
- wrapper:
  `scripts/slurm/run_contact_value_head_train_in_allocation.sh`
- output:
  `experiments/world_model_task_rebinding/cosmos3/contact_value_head_union_plus_panel0134_causal_suffix_1gpu1h_20260623_204725_alloc146658`

The wrapper enforces `min_wall_seconds=3660` and `formal_min_gpus=1`. Early
metrics before the one-GPU-hour floor are liveness only, not a result.

## Boundary

This run addresses the current selector blocker without changing the task
objective: it uses real simulator replay labels to predict final task error,
DP96 handoff success, continuability, grasp preservation, contact progress,
and final peg-head-in-hole state. It does not replace the world-model/action
generator with a hand-coded scorer, and it does not count as task success
until selected actions are executed in saved snapshots and then in the live
closed-loop dynamic task with video/final-state review.

## Result

The training run completed:

- elapsed seconds: `3660.10`
- steps: `141934`
- stop reason: `min_wall_and_min_steps`
- visible CUDA devices: `1`
- final checkpoint:
  `experiments/world_model_task_rebinding/cosmos3/contact_value_head_union_plus_panel0134_causal_suffix_1gpu1h_20260623_204725_alloc146658/checkpoint_final.pt`
- best-gate checkpoint:
  `experiments/world_model_task_rebinding/cosmos3/contact_value_head_union_plus_panel0134_causal_suffix_1gpu1h_20260623_204725_alloc146658/checkpoint_best_gate.pt`

Important implementation note: the original training summary was written
before the outcome-scorer trainer was updated from the older 3-hour formal
floor to the current user-approved 1GPU/1h floor. Therefore
`formal_training_floor_met=false` and `ready_for_formal_live_eval=false` in
that summary are stale with respect to the 2026-06-16/23 execution rule. The
script has been patched so future runs use `min_wall_seconds >= 3600` instead
of `10800`. The raw evidence still shows this run met the 1GPU/1h wall-clock
training requirement.

Offline value-head quality is weak but nonzero:

- best-gate eval groups: `29`
- DP-prior handoff success: `7/29 = 0.2414`
- selected handoff success at best gate: `8/29 = 0.2759`
- selected minus DP handoff success: `+0.0345`
- selected weighted task error improvement: `-0.00324`
- selected contact-progress delta improvement: `+0.0103`
- selected non-DP fraction: `0.6897`
- top-1 handoff-oracle match: `0.1724`
- top-1 weighted-error-oracle match: `0.3103`

The final checkpoint regressed to DP parity on handoff success:

- selected handoff success: `7/29 = 0.2414`
- selected minus DP handoff success: `0.0`
- selected weighted task error was slightly worse than DP:
  `+0.00172`

## Conclusion

This run does not prove a live controller. It does show that a value head can
sometimes choose a non-DP generated candidate that improves the DP96 handoff
label, but the margin is only one additional validation group and the oracle
match is low. The next aligned step is not a full live success claim. If GPU
resources are available, the cautious next experiment is a saved-snapshot
selected-candidate replay using the best-gate checkpoint and conservative
DP-default margins. If that replay does not clearly beat DP on the same saved
snapshots, stop this value-head line and improve the action generator or base
policy rather than tuning scorer thresholds.

Resource status after completion: the held Slurm job `146658` is no longer
available, so any selected replay must start a new tmux-held interactive
allocation. Do not use one-shot `sbatch`.

## Follow-Up Resource Request

On 2026-06-24, opened a new tmux-held interactive allocation request:

- tmux session: `contact_value_selected_replay_1gpu_request_20260624_170015`
- Slurm job: `148676`
- request: `gpu`, `1` GPU, `8` CPUs, `80G`, `1-00:00:00`
- state at launch check: `PENDING`, reason `Priority`

Prepared but not yet run:

- wrapper:
  `scripts/slurm/run_contact_value_head_margin_eval_in_allocation.sh`
- purpose: offline DP-default margin audit for
  `checkpoint_best_gate.pt` over the merged replay-label dataset.

This wrapper must be run only inside the new compute allocation. If the margin
audit does not show a clear DP-default improvement, do not run a live panel.

Because job `148676` remained pending, a second tmux-held request was opened
on `gpux`:

- tmux session: `contact_value_margin_gpux_1gpu_request_20260624_170413`
- Slurm job: `148680`
- request: `gpux`, `1` GPU, `8` CPUs, `80G`, `1-00:00:00`
- state at launch check: `PENDING`, reason `Priority`

Do not add more queue requests unless both remain pending for a long interval
and a concrete alternative partition/resource choice is justified. If either
request starts, use it for the margin audit and cancel the other still-pending
request to avoid tying up resources.
