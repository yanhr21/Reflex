# Post-Fix1 Repair Plan

## First-Principles Reason

Fresh full1000 fix1 proved the full-episode data path and action-adapter
training are no longer broken, but it did not prove controller handoff. The
remaining physical failures are target-motion switching, final target
prediction, executable robot-action chunks, and peg contact/recovery.

## Repair Objective

Produce a Cosmos3 world/action model interface that can:

1. detect or predict target motion without static false positives,
2. predict final target/hole pose accurately enough for rebinding,
3. output robot action chunks with acceptable robot-action error,
4. preserve or recover peg contact before insertion resumes.

This is still full-episode/equal-length training and evaluation. Do not return
to 128-action clips, 93-frame exports, state-only WMs, or controller shortcuts.

## Candidate Repairs

1. Add explicit target-monitor supervision:
   train a temporal target-motion/onset and final-target readout head from
   RGB-derived or latent trajectories, with calibration reported separately on
   reference RGB and generated RGB.
2. Improve role balance:
   oversample or upweight pre-motion target forecast, observed/post-motion
   target continuation, insert-resume, and peg-recovery masks without slicing
   the episode into short samples.
3. Separate executable robot-action evaluation:
   keep reporting first-7-dim robot action metrics separately from state
   sidecar metrics; consider a dedicated action/chunk head or distillation
   target if generic WAM action tokens remain high-RMSE.
4. Add contact/grasp recovery supervision:
   expose grasp, peg-head-in-hole, and contact-continuity readouts as explicit
   losses/diagnostics; visual review remains mandatory.

## Fix2 Training-Target Repair

The first concrete repair is to fix the structured target carried by Cosmos3's
action branch. Fresh fix1 only supervised true robot actions in dims `0..6`.
Dims `7..31` repeated the prefix TCP/peg/hole/contact state at every step, so
the model did not directly learn a future target/peg/TCP/contact rollout in
the action/state tokens.

Fix2 keeps the same full1000 source videos and the same full-episode length
contract, but changes the sidecar target:

- `sidecar_target_mode=future_aligned_state`;
- action row `i` corresponds to video frame `i+1`;
- rows before `prefix_frame` are clean observed history conditions;
- rows from `prefix_frame` onward are generated future targets containing
  robot action plus TCP, peg, hole, peg-head-relative-to-hole, target velocity,
  grasp/insert flags, perturb deltas, time fraction, and perturb trigger;
- no future target pose is added to the prompt or to conditioned history rows.

This directly addresses the user-facing requirements:

- target onset/final pose can be learned as future hole/velocity/contact state,
  not only as a post-hoc readout from generated pixels;
- generated action chunks now carry both executable robot action and
  controller-facing future task state;
- static false positives can be trained against with static monitor rows whose
  future hole state remains stationary;
- peg recovery has explicit peg/TCP/peg-head/contact sidecar targets rather
  than relying only on RGB reconstruction.

## Fix2 Sampling

Use role-balanced row repetition only at the JSONL level:

`target_pre_motion=2,target_motion_observed=1,target_post_motion=2,insert_resume=2,peg_recovery=3,static_monitor=3,static_late_monitor=3`

This does not slice episodes, change frame counts, or regenerate data.
Validation remains unweighted.

## Fix2 Execution Gate

Before SFT, the Slurm-side condition audit must prove:

- every row remains `301` RGB frames and `300x32` action/state rows;
- `sidecar_target_mode=future_aligned_state` is recorded in row metadata;
- action audit reports all future-aligned rows have varying task-state
  sidecars, preventing a repeat of the prefix-repeated bug;
- role-weighted train JSONL manifest exists if weighting is enabled.

## Gate Before Controller

Controller/DP integration remains blocked until a future run passes:

- strict `301/301` video and `300x32/300x32` action accounting;
- generated-RGB target-monitor metrics close enough to reference-RGB
  calibration;
- final target/hole error and target path RMSE recorded on moving target
  samples;
- robot-action future RMSE by role, not only aggregate action RMSE;
- visual peg/hand/contact continuity on peg_drop, peg_disturb, and
  insert_resume samples.
