# Phase 03 Oracle Visual Review Checklist

Use this checklist after a full-pipeline run writes an annotated video. This
checklist is required because artifact audits and simulator metrics alone do
not prove physical insertion.

Required files:

- `manifest.json`
- `summary.json`
- `classification.txt`
- `artifact_audit.json`
- `action_trace.json`
- `videos/raw.mp4`
- `videos/annotated.mp4`
- optional replay views under `review/<view>/raw.mp4`,
  `review/<view>/annotated.mp4`, and `review/<view>/trace.json`
- every Cosmos report's `prefix_rgb.mp4`
- every Cosmos report's `vision.mp4`
- every Cosmos report's `sample_outputs.json`
- every Cosmos report's `cosmos_action_chunk.json`

Single-case success requires all of the following:

- the annotated video shows DP static prefix before target motion;
- the target / hole motion is gradual and visible, not a one-frame jump;
- Cosmos RGB prediction is generated from live RGB history;
- Cosmos action prediction is generated and used during the dynamic stage;
- the dynamic target-motion stage is controlled by `cosmos3_policy_output`,
  not by Diffusion Policy;
- the finisher starts only after the near-target gate;
- final insertion is visually complete and continuous;
- final insertion is active robot insertion: the robot / gripper drives the
  peg or wooden stick into the hole through controller actions after Cosmos
  evidence exists;
- target / hole motion does not create success by moving onto a mostly
  stationary peg / wooden stick;
- the final insertion window is reviewed frame-by-frame around the first
  simulator-success frame or the final lowest-distance frame. In that window,
  the robot / gripper and peg / wooden stick must show active insertion motion
  toward / into the hole; the target / hole moving toward the held peg is not
  sufficient;
- compare the action trace around the insertion window with the video. The
  trace must show controller actions that plausibly drive the robot/peg into
  the hole after Cosmos evidence exists, not only target/hole motion creating
  a metric success;
- action-trace stage order is valid: first target-motion increment, then
  Cosmos dynamic-control rows, then DP/manual physical finisher rows. A trace
  with the right row counts in the wrong order is invalid;
- the first `live_eval.success=true` row in `action_trace.json` occurs in
  `oracle_physical_dp_finisher` or `oracle_physical_manual_finisher`, after the
  finisher has started. If simulator success appears before the physical
  finisher, reject the run as target-assisted / pre-finisher success unless a
  separate user-approved diagnostic explicitly says otherwise;
- the peg / wooden stick does not teleport, snap, disappear, or penetrate a
  wall;
- the target / hole does not teleport or disappear;
- the run has no `set_pose`, `set_state`, `set_state_dict`, source-state
  restore, saved-state replay, geometric final placement, future labels, or
  hand-selected suffix controller path;
- `method_evidence_allowed=false`;
- `physical_insertion_success_claimed=false` until this review is completed.

Invalid immediately:

- peg snaps from far away to inside the hole;
- peg inserts into the wall rather than the hole;
- target / hole jumps in one frame;
- target / hole motion is the main reason the peg / wooden stick becomes
  inserted, rather than robot-controlled insertion;
- object disappears or passes through a solid surface;
- action trace shows Diffusion Policy actions during the dynamic Cosmos stage;
- action trace has DP static-prefix rows after target motion starts, or has the
  finisher before the Cosmos dynamic-control stage;
- simulator success flips true after any state edit;
- only a boundary, trigger, or no-insertion video is produced.

Verdict file requirements:

- write `single_case_oracle_success_confirmed=true` only after every required
  item above passes;
- write `active_robot_insertion_confirmed=true` only when the insertion-window
  video and action trace both support robot-driven insertion;
- write `full_sequence_video_reviewed=true` only after reviewing a rendered
  video that covers DP prefix, target-motion trigger, Cosmos dynamic-control
  rows, and the DP/manual physical finisher through insertion or physical
  failure;
- write `video_covers_cosmos_control_and_finisher=true` only when the reviewed
  rendered video visibly includes both the Cosmos-control stage and the
  finisher stage;
- write `video_covers_final_insertion_or_physical_failure=true` only when the
  reviewed rendered video reaches final insertion or a clearly logged physical
  failure; a boundary-only or target-motion-start video must set this false;
- write `target_assisted_insertion_rejected=true` when target/hole motion is
  the main insertion mechanism or when the video is too occluded to rule that
  out;
- write `overall_task_complete=false` unless directional coverage,
  peg/wooden-stick disturbance, and multiple approved `fix3_733` keys are all
  already successful.

Known counterexamples:

- `h5_continuous_insert/try06` reached simulator metric true but was rejected
  because target/hole motion appeared to move onto the held peg before a
  visually confirmed robot-driven insertion.
- `h5_constant/try01` reached simulator metric true but was rejected as
  target-assisted / passive insertion and archived.

Overall task remains incomplete after one reviewed success. Continue with:

- forward/backward target motion;
- left/right target motion;
- peg / wooden-stick disturbance;
- multiple approved keys from `experiments/maniskill/data/fix3_733/`.
