# 2026-06-22 Live Action-Bank Replay Blocker

## Plain Result

The current closed-loop failure is not a length-contract issue and not a
simple evaluation-script crash. The latest live panel produced valid
full-length rollouts, but final success stayed at `0`.

The current blocker is live action consequence: the generated action chunks do
not reliably move the real simulator state into an insertion-continuable
contact state.

## Evidence Paths

Live action-bank panel:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_h96_fullunion_scorer_bestgate_formalhardphase_actionbank_samples1_3_20260622_074211_alloc145920`

Common-candidate replay labels:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_current_panel_20260622_0812_alloc145920`

All-candidate first-bank replay labels:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_action_bank_replay_allcand_iter0_20260622_0820_alloc145920`

sample 03 contact sheet:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_h96_fullunion_scorer_bestgate_formalhardphase_actionbank_samples1_3_20260622_074211_alloc145920/sample_3_single_panel/live_receding_panel_contact_sheet.png`

## Numbers

Common replay over selected panel candidates:

- `77` valid replay labels.
- `0` replay failures.
- `50/77` improved absolute y/z.
- `27/77` worsened absolute y/z.
- `11/77` passed the after-chunk continuability gate.
- `0/77` achieved task success.
- Live-scorer selected candidates: `7/11` improved y/z, `4/11` worsened y/z,
  `2/11` passed the after-chunk gate, `0/11` succeeded.

All-candidate first-bank replay:

- `426` valid replay labels.
- `0` replay failures.
- `0/426` passed the after-chunk gate.
- `0/426` succeeded.
- sample 01 first bank: `159/213` improved y/z, but `0/213` passed the gate.
- sample 03 first bank: `0/213` improved y/z; all `213/213` worsened y/z.

sample 03 live panel:

- `301` observed frames, full contract ok.
- `99` executor-active frames.
- `69` DP-handoff frames.
- Final `peg_head_pos_at_hole`:
  `[-0.0951820016, 0.0027006269, 0.0033096075]`.
- Final success: `false`.
- Visual review: final frame still not inserted.

## Interpretation

This does not prove the whole method is wrong. It proves the current executor
and scorer are not yet grounded in the real consequences of their own action
chunks.

The important distinction:

- If only the scorer were wrong, all-candidate replay should reveal hidden
  good candidates that the scorer missed.
- In the first-bank all-candidate replay, no candidate reached the gate, and
  sample 03 had no y/z-improving candidate at all.

Therefore the next useful step is not another repeat of the same closed-loop
panel. The next useful step is to improve the candidate action distribution
and live consequence model using replay labels from real live snapshots.

This remains the DDP-style direction: short action chunks, real re-observation,
and selection by predicted task progress. It is not error-case enumeration.
