# 2026-06-22 Panel0245 Offsets64 Final: 2/4 Mixed Evidence

## Run

- Allocation: `146658` on `server56`
- Panel root:
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658`
- Samples: `0,2,4,5`
- Source suffix offsets: `64,48,32,24`
- Boundaries: same-scenario suffixes, source-suffix start distance cap `0.02`,
  source-suffix execute steps `8`, no unsafe pure DP prior while live `C_pi`
  is false, full episode contract `301` observed frames / `300` action steps.

## Panel Summary

`live_receding_panel_summary.json` reports:

- `completed_samples=4`
- `requested_samples=4`
- `failed_process_count=0`
- `panel_full_episode_contract_ok=true`
- `sample_contract_failures=[]`
- `final_success_count=2`
- `method_evidence_allowed=false`

The panel contact sheet was opened:

`live_receding_panel_contact_sheet.png`

The sheet matches the metrics: sample00/sample02 are not inserted at the end;
sample04/sample05 end at the hole/box region with no visual contradiction to
the success metrics.

## Per-Sample Results

| sample | scenario | final success | final peg-head-at-hole | visual review |
|---|---|---:|---|---|
| 00 | `hole_late_move_stop` | false | `[-0.095756, 0.016172, -0.065285]` | review sheet opened; no insertion |
| 02 | `hole_late_reverse` | false | `[-0.112931, 0.047073, -0.061056]` | review sheet opened; no insertion |
| 04 | `hole_late_sine` | true | `[-0.004514, -0.002941, -0.002943]` | review sheet opened; supports success |
| 05 | `hole_late_continuous_insert` | true | `[0.029410, -0.002664, -0.002634]` | review sheet opened; supports success |

## Causal Reading

This is mixed evidence, not broad method success.

The corrected source-suffix/live-receding/DP-handoff interface can work:
sample04 and sample05 both reached a real DP-continuable state and finished
with frozen DP handoff.

It is not stable:

- sample00 has both a standalone success and a same-protocol panel failure.
- sample02 reached `C_pi` around frames `144/148`, but DP96 still moved the
  real rollout away from insertion.
- sample00 panel rerun reached `C_pi` by frame `168`, but DP96 moved the real
  state to `[-0.097337, 0.006050, -0.049492]` and did not finish.

The current handoff gate is therefore not the right target by itself.
There are two contradictory but useful signals:

- A sample00 one-iteration replay had `C_pi=false` after the candidate chunk,
  yet `candidate + DP96` succeeded.
- The panel failures had `C_pi=true`, yet DP96 failed in the real closed loop.

## Next Repair Target

Do not repeat the same panel as the next main action.

The next useful repair is to train/evaluate a scorer on real
`candidate chunk + DP rollout` continuability/contact labels from the saved
live snapshots and candidate banks. The label should distinguish:

- `C_pi=true` states where DP96 still fails;
- `C_pi=false` states where DP96 succeeds;
- contact/insertion progress that predicts whether frozen DP will actually
  preserve grasp and finish insertion in the live rollout.

This preserves the task-frame rebinding objective: Cosmos/source-suffix
candidates propose short physical bridges, the robot executes only short
chunks, the loop re-observes, and DP handoff is allowed only when the real
closed-loop state is continuable.
