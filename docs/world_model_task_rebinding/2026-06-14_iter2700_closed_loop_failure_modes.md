# Cosmos3 Closed-Loop Failure Modes

- implementation_contract_ok: `True`
- method_effectiveness_ok: `False`
- val Cosmos: `1/3`
- val pure DP: `3/3`
- hard Cosmos on pure-DP failures: `1/6`
- hard matched pure-DP success: `0/6`
- primary_current_failure: `direct_raw_cosmos_action_rebind_and_dp_continuability_are_unreliable`

## Method Effectiveness Failures

- `val_cosmos_underperforms_same_source_pure_dp:1/3<pure_dp:3/3`
- `hard_case_not_broadly_reliable:1/6`

## Hard Failure Gate Blocks

- `rel_y_abs`: `203`
- `rel_z_abs`: `195`
- `rel_x_min`: `36`
- `grasped`: `18`
- `hole_speed`: `9`

## Hard Failure Action Flags

- `pred_x_scale_gt_2x_teacher`: `2`
- `pred_x_sign_agreement_lt_0.5`: `2`
- `pred_y_scale_gt_2x_teacher`: `2`
- `pred_z_scale_gt_2x_teacher`: `2`
- `pred_y_sign_agreement_lt_0.5`: `1`
- `pred_z_sign_agreement_lt_0.5`: `1`

## Not The Current Primary Failure

- `video_length_or_missing_301_frames`
- `manual_target_onset_disclosure`
- `missing_cosmos_takeover_annotation`
- `static_no_motion_special_branch`

## Next Aligned Action

Do not keep broad-evaluating the old checkpoint as evidence. Use the clean-role/dense-receding condition repair path, starting with preflight and two-sample overfit after explicit user approval; if direct raw Cosmos actions remain unstable, move to a learned short-chunk executor or DP-prior policy conditioned on Cosmos-predicted task state.
