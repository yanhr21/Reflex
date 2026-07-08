# Dataset Sample Review Template

Use this template when reviewing any candidate dataset sample before it enters
training.

## Identity

```text
sample_id:
dataset_class: A_static_expert | B_dynamic_rgb_observation | C_frozen_dp_dynamic_failure | D_future_frame_cooperation_teacher | E_cosmos_predicted_cooperation
source_h5:
source_json:
rgb_video:
review_frames:
manifest:
summary:
```

## Source And Controller

```text
controller:
action_contract:
dp_checkpoint:
cosmos_checkpoint:
future_target_source:
future_target_is_ground_truth:
future_target_is_cosmos_predicted:
teacher_only:
method_evidence_allowed:
```

## RGB / State Alignment

```text
rgb_present:
state_trace_present:
action_trace_present:
rgb_state_timestamps_aligned:
frame_rate:
num_frames:
num_action_rows:
```

Reject if RGB and state/action rows cannot be aligned.

## Motion And Task State

```text
target_motion_family:
target_motion_continuous:
T_hole_t_available:
T_hole_t_plus_tau_available:
v_hole_t_plus_tau_available:
tau_available:
target_uncertainty_available:
relative_peg_in_hole_frame_available:
relative_ee_in_hole_frame_available:
relative_velocity_at_contact_available:
```

For moving-target data, reject positive labels if target motion is discontinuous
or unlogged.

## Outcome Labels

```text
success:
active_robot_driven_insertion:
target_assisted:
miss:
jam:
late_contact:
bad_relative_velocity:
no_progress:
snap_or_teleport:
state_intervention:
hidden_manual_finisher:
wall_penetration:
disappearing_objects:
```

Positive policy / adapter data requires `active_robot_driven_insertion=true`
and all invalidity flags false.

## Allowed Losses

```text
dp_bc:
dp_distillation:
cosmos_static_future:
cosmos_dynamic_future:
target_frame_readout:
trajectory_consistency:
uncertainty:
negative_classification:
discrepancy:
contrastive:
adapter_residual:
moving_frame_conditioning:
phase_timing:
uncertainty_conditioned_control:
```

Failed actions are never positive DP BC just because they are present in an
episode.

## Review Decision

```text
accepted_for_training:
accepted_for_diagnostic_only:
rejected:
rejection_reason:
requires_human_review:
reviewer:
review_time:
notes:
```
