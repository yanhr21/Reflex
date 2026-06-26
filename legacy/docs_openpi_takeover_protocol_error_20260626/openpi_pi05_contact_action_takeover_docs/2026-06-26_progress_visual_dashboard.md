# 2026-06-26 OpenPI pi0.5 Progress Visual Dashboard

## What This Shows

This dashboard visualizes the current OpenPI action-policy progress. It does
not visualize a successful world-model closed loop, because the current replay
experiments intentionally bypass the world model to isolate the action
executor.

The tested loop here is:

1. saved dynamic ManiSkill snapshot;
2. current observation/state prepared for OpenPI;
3. official OpenPI/pi0.5 checkpoint inference;
4. execute the predicted action chunk in ManiSkill;
5. measure grasp, insertion, contact stability, and optional DP96 handoff.

Therefore, the current failure is not evidence that Cosmos generated the wrong
action. Cosmos is not the action generator in these replays.

## What Is Object17?

`object17` is a 17-dimensional observation/state vector added to OpenPI so the
policy can see the current task geometry. It is not a different task and not a
new action. The task is still "insert the grasped peg into the current target
hole"; `object17` only changes what state the policy observes.

Layout:

- `tcp_xyz`: 3 dims;
- `peg_xyz`: 3 dims;
- `hole_xyz`: 3 dims;
- `peg_head_at_hole_xyz`: 3 dims, peg-head position expressed relative to the
  current hole frame;
- `hole_velocity_step_xyz`: 3 dims;
- `grasped`: 1 dim;
- `inserted`: 1 dim.

This is privileged simulator/source state in the current diagnostics. It is an
upper-bound conditioning signal, not yet the final publishable RGB-derived
perception interface.

## Summary Table

| Branch | What Changed | Direct Inserted | Direct Success | Grasp | DP96 Handoff | Diagnosis |
|---|---|---:|---:|---:|---:|---|
| Baseline OpenPI qpos8 suffix | Official pi0.5, qpos/gripper state only | 0/4 | 0/4 | 4/4 | 0/4 success | Holds peg, does not bind action to moved hole |
| Object17-video | Adds current object/task-frame state | 0/4 | 0/4 | 4/4 | 3/4 success | Better handoff, still no direct insertion |
| Object17 receding smoke | Refreshes object17 state every 4 executed steps | 0/1 | 0/1 | 1/1 | not main | Re-observation alone did not fix hard snapshot |
| Near-contact object17-video | Trains closer-to-insertion offsets `16,12,8,4,2,1` | 0/4 | 0/4 | 4/4 | 1/4 success | Offset-only reweighting did not create contact mode |

## Baseline OpenPI qpos8

Run root:

`experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix1699_panel4_exec16_alloc150773`

Result:

- direct inserted `0/4`;
- direct success `0/4`;
- grasp `4/4`;
- DP96 success `0/4`.

Visuals:

![baseline f106](/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix1699_panel4_exec16_alloc150773/contact_state_sheets/openpi_pi05_0000_iter_000_f106/contact_state_sheet.png)

![baseline f094](/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix1699_panel4_exec16_alloc150773/contact_state_sheets/openpi_pi05_0001_iter_000_f094/contact_state_sheet.png)

![baseline f132](/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix1699_panel4_exec16_alloc150773/contact_state_sheets/openpi_pi05_0002_iter_000_f132/contact_state_sheet.png)

![baseline f116](/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/openpi/openpi_pi05_snapshot_replay_20260626_contact_suffix1699_panel4_exec16_alloc150773/contact_state_sheets/openpi_pi05_0003_iter_000_f116/contact_state_sheet.png)

## Object17-Video OpenPI

Run root:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_direct1700_replay4_exec16_fixhistory_alloc153455`

Result:

- direct inserted `0/4`;
- direct success `0/4`;
- grasp `4/4`;
- DP96 success `3/4`.

This is the strongest positive signal so far: object/task-frame conditioning
makes OpenPI produce states that the old DP can often finish from. It still
does not insert directly.

Visuals:

![object17 f106](/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_direct1700_replay4_exec16_fixhistory_alloc153455/contact_state_sheets/openpi_pi05_0000_iter_000_f106/contact_state_sheet.png)

![object17 f094](/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_direct1700_replay4_exec16_fixhistory_alloc153455/contact_state_sheets/openpi_pi05_0001_iter_000_f094/contact_state_sheet.png)

![object17 f132](/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_direct1700_replay4_exec16_fixhistory_alloc153455/contact_state_sheets/openpi_pi05_0002_iter_000_f132/contact_state_sheet.png)

![object17 f116](/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_direct1700_replay4_exec16_fixhistory_alloc153455/contact_state_sheets/openpi_pi05_0003_iter_000_f116/contact_state_sheet.png)

## Receding Object17 Smoke

Run root:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_receding1_q3_exec4_alloc153455`

Result:

- one hard sample;
- `3` OpenPI queries;
- `4` executed steps per query;
- direct inserted `0/1`;
- direct success `0/1`;
- grasp `1/1`;
- final `abs(y)+abs(z)` worsened by `0.08026`.

Visual:

![receding f106](/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/openpi/20260626_object17_video_clean_receding1_q3_exec4_alloc153455/contact_state_sheets/sample_00_hole_late_move_stop_iter000/contact_state_sheet.png)

## Near-Contact Object17-Video

Run root:

`experiments/world_model_task_rebinding/openpi/20260626_object17_video_nearcontact_direct1700_replay4_exec16_alloc153455`

Result:

- direct inserted `0/4`;
- direct success `0/4`;
- grasp `4/4`;
- DP96 success `1/4`.

This falsifies the narrow explanation that the old dataset failed mainly
because it included far approach windows. It did not produce the missing final
contact/insertion mode.

Visuals:

![nearcontact f106](/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/openpi/20260626_object17_video_nearcontact_direct1700_replay4_exec16_alloc153455/contact_state_sheets/openpi_pi05_0000_iter_000_f106/contact_state_sheet.png)

![nearcontact f094](/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/openpi/20260626_object17_video_nearcontact_direct1700_replay4_exec16_alloc153455/contact_state_sheets/openpi_pi05_0001_iter_000_f094/contact_state_sheet.png)

![nearcontact f132](/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/openpi/20260626_object17_video_nearcontact_direct1700_replay4_exec16_alloc153455/contact_state_sheets/openpi_pi05_0002_iter_000_f132/contact_state_sheet.png)

![nearcontact f116](/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/openpi/20260626_object17_video_nearcontact_direct1700_replay4_exec16_alloc153455/contact_state_sheets/openpi_pi05_0003_iter_000_f116/contact_state_sheet.png)

## Where The Closed Loop Fails

The failure happens at the final insertion action, not at gross grasp
preservation:

- every evaluated OpenPI branch keeps grasp on all tested chunks;
- no branch produces an inserted step in its own OpenPI chunk;
- qpos8 often moves in source-like directions without binding to the moved
  target;
- object17 improves handoff but still stops short of insertion;
- receding state refresh alone does not fix the hard snapshot;
- near-contact offset reweighting does not fix it either.

Current best diagnosis: the action policy has not learned a robust
hole-frame-to-robot-action contact correction. The next visual diagnostic
should compare OpenPI-predicted action vectors against successful source
near-contact actions in TCP/hole frames, not just compare success counters.
