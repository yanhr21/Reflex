# Data Preflight TODO

## Full-Episode Export

- [ ] Create a new clean condition root name, e.g.
      `action_state_conditions_full1000_maniskill_default_regen_full_episode_300step_YYYYMMDD`.
- [ ] Export one row per full episode, not one row per 128-step chunk.
- [ ] Store `num_rgb_frames=301` and `num_action_steps=300` when frame 0 is
      included.
- [ ] Store causal prefix metadata as masks/indexes over the full episode,
      not as physically sliced 129-frame videos.
- [ ] Include action/proprio/task-state histories only up to each prefix.
- [ ] Store future target/peg/TCP states only as supervision/readout targets,
      not controller-facing conditions.

## Refusal Checks

- [ ] Refuse any row with `129` video frames or `128` action steps.
- [ ] Refuse any row with `93` generated/reference frames.
- [ ] Refuse missing or stale `input_h5` paths.
- [ ] Refuse train/val leakage by source sample id.
- [ ] Refuse rows where RGB video, H5 state, action tensor, and task labels do
      not refer to the same scenario and trajectory.

## Visual Sanity

- [ ] Build a small scenario-diverse contact sheet panel after export.
- [ ] Verify default view, 30 fps, full pickup/holding, target motion, and peg
      disturbance/drop visibility.
- [ ] Keep this as data sanity evidence only, not model evidence.
