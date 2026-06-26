# 2026-06-23 Contact-Action Reset Audit

## Scope

This audit follows the user correction that scorer-only action selection is
not an elegant or sufficient solution for insertion. No project compute was
run on the login node. The actions here were read-only inspection, web
literature review, documentation, and conservative artifact cleanup.

## Current Local Result

Latest inspected live panel:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_safegate1_source_suffix_offsets64_48_32_24_panel0134_20260623_alloc146658`

Panel summary:

- completed samples: `4`
- failed processes: `0`
- full episode contract: `true`
- final success count: `0/4`
- method evidence allowed: `false`
- videos: each inspected video reports `301` decoded frames at `30 fps`

Final peg-head positions in hole frame:

- sample 0, `hole_late_move_stop`: `[-0.0971, 0.0144, -0.0716]`, failure.
- sample 1, `hole_late_constant`: `[-0.0922, 0.0329, -0.0782]`, failure.
- sample 3, `hole_late_fast_shift`: `[-0.1056, 0.0014, 0.0039]`, failure.
- sample 4, `hole_late_sine`: `[-0.1040, 0.0306, -0.0325]`, failure.

Contact sheet:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_safegate1_source_suffix_offsets64_48_32_24_panel0134_20260623_alloc146658/live_receding_panel_contact_sheet.png`

Visual inspection is consistent with the metrics: the controller often gets
near the block/hole but does not produce a stable insertion-axis/contact
sequence. This is a physical policy failure, not a length-contract failure.

## Conclusion

The current blocker is not only candidate ranking. A scorer can reject or rank
existing chunks, but it cannot create the missing contact-rich insertion
behavior. The repeated failure mode is that the peg remains roughly
`9-10.5 cm` short along the hole-frame insertion axis, even when lateral error
is sometimes small. Frozen DP can be kept as a baseline/static prior/fallback,
but it should not be treated as the required final executor.

The active method should pivot to a contact/insertion action generator or
WAM-conditioned executor: generate short executable chunks that enter and stay
on the insertion/contact manifold, then use value/risk only as a consequence
model.

## External Evidence Reviewed

- Tau0-WM: unified video-action model that can emit action chunks and roll out
  visual latents in receding deployment: `https://arxiv.org/html/2606.01027v1`.
- DreamZero: jointly models video and action as a world-action policy, with
  real-time closed-loop deployment: `https://arxiv.org/html/2602.15922v1`.
- Cosmos Policy: fine-tunes a pretrained video model to emit action latent
  frames, future state images, and values in one stage:
  `https://arxiv.org/abs/2601.16163`.
- Video-action alignment work uses receding-horizon execution from fresh
  camera frames after each segment: `https://arxiv.org/html/2603.17808v1`.
- Hierarchical Diffusion Policy explicitly notes that ordinary diffusion
  policy is weak in contact-rich tasks without contact guidance:
  `https://arxiv.org/html/2411.12982v1`.
- Reactive Diffusion Policy and force-aware policies support the same lesson:
  insertion needs fast contact/tactile/force or compliance-aware feedback, not
  just open-loop visual chunks:
  `https://reactive-diffusion-policy.github.io/`,
  `https://arxiv.org/html/2409.11047v1`,
  `https://arxiv.org/html/2505.22159v3`.

## Problems And Blockers

1. Action coverage gap: current candidates do not reliably contain a chunk
   that drives the peg through the insertion axis after dynamic motion.
2. DP continuability target is unreliable: previous DP96 replay found both
   false-negative and false-positive `C_pi` states, so instantaneous geometry
   or old continuability gates cannot be the main target.
3. Scorer variance is high: h2048, h8192, and h16384 helped different samples;
   simple ensemble rules did not fix the physical missing-action problem.
4. Contact evidence is thin: the current labels emphasize final success and
   geometry more than contact stability, insertion force/compliance, and grasp
   preservation.
5. Source-suffix retrieval may be too scenario-specific: replayed suffixes can
   help some saved states, but the live panel still failed all four samples.
6. Current WAM use is not yet a real action generator. It is still mostly
   supporting candidate selection/rebinding, not training the video model to
   generate contact-aware actions and values as in Cosmos Policy-style methods.

## Likely Hidden Problems To Check Next

- The action representation may not expose the right insertion primitive:
  position-only chunks can jam or stop short without compliance/force control.
- The dataset may underrepresent the exact dynamic contact states where DP
  fails, because accepted v7 source rows were generated by successful static
  DP behavior.
- Contact/force proxies may exist in simulator state but are not yet promoted
  to labels or causal observations for the action model.
- The current suffix bank may encode source-specific timing and pose rather
  than reusable task-frame insertion behavior.
- Value labels may be delayed too far: final success alone may not teach the
  short-horizon model how to stay in contact, preserve grasp, and advance
  along insertion x.
- Existing checkpoints may be overfit to candidate families rather than
  learning a general action distribution around insertion.
- If Octo/pi0/OpenPI/OpenVLA are used, action-space and embodiment adaptation
  may dominate the schedule; this needs a clean integration audit before
  training, not speculative tiny runs.

## Next Work

1. Build a contact-action training manifest from the 733 source trajectories:
   insertion suffix start/end, peg-hole relative state, grasp/contact proxies,
   action chunks, and final insertion/contact-stable labels.
2. Merge live `candidate + DP96` labels as positive/negative consequence data,
   especially the hard negatives where y/z or `C_pi` looked favorable but real
   DP handoff failed.
3. Train a short-horizon contact-suffix diffusion/action generator for at
   least one GPU-hour on a full documented data split inside the tmux-held
   Slurm allocation.
4. Audit local Cosmos/Cosmos-Policy feasibility for an action/value/video head.
   This is the most direct same-backbone replacement for scorer-only
   selection if checkpoint/tooling constraints allow it.
5. Use the scorer only as a value/risk head over real consequences: final
   success, DP96 continuability, contact stability, grasp preservation, and
   insertion-axis progress.
6. Run live full `301/300` panels only after offline generated candidates beat
   the DP prior on held-out saved live states.

## Cleanup Performed

Moved `659` superseded canary/smoke/debug/scorer/retrieval directories from
the active Cosmos3 experiment tree to:

`experiments/_archive_20260623_contact_action_reset/cosmos3_superseded/`

Recoverable manifest:

`experiments/_archive_20260623_contact_action_reset/MOVED_DIRS.txt`

The cleanup intentionally preserved active data, checkpoints, live labels,
current panel outputs, and evidence notes.
