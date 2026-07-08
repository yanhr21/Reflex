# Dataset Manifest Schema

Date: 2026-07-06

Every active dataset sample or run must write a manifest. This schema is the
minimum required structure for dataset construction.

For legacy smoke runs that predate this schema, use a non-destructive
`manifest_corrections.txt` next to `manifest.txt` instead of rewriting the
original run manifest. New runners must write the required fields directly in
`manifest.txt`.

## Run-Level Manifest

Required run fields:

```text
phase
dataset_class
run_group
run_name
job_id
step_id
node_list
output_dir
log_file
source_paths
controller
action_contract
rgb_required
human_review_required
large_scale_production_allowed
method_evidence_allowed
teacher_evidence_allowed
allowed_losses
disallowed_losses
forbidden_state_intervention_expected
```

Dataset class must be one of:

```text
A_static_expert
B_dynamic_rgb_observation
C_frozen_dp_dynamic_failure
D_future_frame_cooperation_teacher
E_cosmos_predicted_cooperation
```

## Sample-Level Manifest

Required sample fields:

```text
sample_id
dataset_class
source_h5
source_json
rgb_video
rgb_frames
state_action_timing
rgb_timing
controller
action_contract
dp_checkpoint
cosmos_checkpoint
future_target_source
future_target_is_ground_truth
future_target_is_cosmos_predicted
allowed_losses
evidence_type
human_review_required
```

Required task-state fields when available:

```text
T_hole_t
T_hole_t_plus_tau
v_hole_t_plus_tau
target_uncertainty
T_peg_t
T_ee_t
relative_peg_in_hole_frame
relative_ee_in_hole_frame
tau
phase
relative_velocity_at_contact
```

Required outcome labels:

```text
success
miss
jam
late_contact
bad_relative_velocity
target_assisted
snap_or_teleport
state_intervention
hidden_manual_finisher
timestamp_alignment_ok
positive_policy_data_allowed
positive_adapter_data_allowed
negative_or_diagnostic_allowed
```

## Loss Permissions

Allowed by class:

- `A_static_expert`: `dp_bc`, `dp_distillation`,
  `cosmos_static_future`, `phase_extraction`
- `B_dynamic_rgb_observation`: `cosmos_dynamic_future`,
  `target_frame_readout`, `trajectory_consistency`, `uncertainty`
- `C_frozen_dp_dynamic_failure`: `negative_classification`,
  `discrepancy`, `infeasible_no_progress`, `contrastive`
- `D_future_frame_cooperation_teacher`: `adapter_residual`,
  `moving_frame_conditioning`, `phase_timing`,
  `relative_velocity_at_contact`
- `E_cosmos_predicted_cooperation`: `adapter_robustness`,
  `uncertainty_conditioned_control`, `live_method_evaluation`

Failed action chunks must not be marked as positive DP expert actions.

## Human Review Gate

Every RGB smoke run must set:

```text
human_review_required=true
large_scale_production_allowed=false
```

Only after user approval may a later production run set:

```text
human_review_required=false
large_scale_production_allowed=true
```

## Validation

Use the read-only validator:

```bash
scripts/world_model/validate_dataset_run_manifest.sh <run_dir>
```

The validator reads:

- `manifest.txt`;
- `summary.json`;
- optional `manifest_corrections.txt` for legacy overlays.

It must not submit Slurm jobs or modify experiment artifacts.
