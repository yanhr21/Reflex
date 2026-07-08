# Phase 03 Full-Pipeline Review Runbook

Use this after `scripts/slurm/run_phase03_oracle_full_pipeline_in_allocation.sh`
finishes. This runbook is evidence review only; it does not mark the goal
complete.

1. Locate the newest run directory:

   `experiments/maniskill/runs/03_oracle/<case_group>/<tryNN>/`

   Optional read-only status helper:

   `scripts/world_model/summarize_phase03_oracle_full_pipeline_artifacts.sh`

2. Read required text artifacts:

   - `manifest.txt`
   - `manifest.json`
   - `classification.txt`
   - `summary.json`
   - `artifact_audit.json`
   - `action_trace.json`

3. Required protocol checks:

   - `method_evidence_allowed=false`
   - `physical_insertion_success=false` in `classification.txt`
   - `physical_insertion_success_claimed=false` in `summary.json`
   - `cosmos_dynamic_actions_executed=true`
   - at least one `cosmos_policy_reports[*].stage_name` starts with `pre/`
     for new runs; `premotion_` is accepted only for old historical artifacts
   - at least one `cosmos_policy_reports[*].stage_name` starts with `post/`
     for new runs; `postmotion_` is accepted only for old historical artifacts
   - `target_motion_trigger_no_robot_action` exists before the Cosmos-control
     rows
   - no `dp_static_prefix` trace row has `target_motion_applied=true`
   - no `dp_static_prefix` row appears after the target-motion trigger row
   - `cosmos_dynamic_action_count >= min_cosmos_dynamic_actions_before_finisher`
     before any finisher can be accepted
   - every dynamic-stage action trace row has
     `stage=cosmos_dynamic_control`
   - every dynamic-stage action trace row has
     `action_source=cosmos3_policy_output`
   - no dynamic-stage row uses Diffusion Policy action source
   - stage order is valid: first target-motion increment before first
     `cosmos_dynamic_control`, first `cosmos_dynamic_control` before first
     DP/manual physical finisher, and no `dp_static_prefix` after target motion
     starts
   - `peg_state_guard.ok=true`
   - `discontinuity_audit.snap_detected=false`
   - `artifact_audit.ok=true`

4. Required Cosmos evidence:

   For every report in `summary.json` `cosmos_policy_reports`, check:

   - `prefix_video` exists and is from live RGB history;
   - `cosmos_rgb_prediction_video` exists;
   - `sample_output_json` exists;
   - `cosmos_action_chunk_json` exists;
   - action chunk `ok=true`.

5. Required video review:

   Use `scripts/world_model/phase03_oracle_visual_review_checklist.md`.
   Review `videos/annotated.mp4` before accepting any
   single-case success. Artifact audit and simulator metrics are insufficient.
   For moving target / moving hole validation keys, explicitly check that the
   robot / gripper actively drives the peg or wooden stick into the hole after
   Cosmos evidence exists. If the target / hole moves onto a mostly stationary
   peg / wooden stick and creates insertion, reject the run as target-assisted
   and do not count it as success.
   Review the insertion window frame-by-frame around the first simulator
   success frame or the lowest final-distance frame. Cross-check those frames
   against nearby `action_trace.json` rows: the trace must show controller
   actions that plausibly drive robot/peg insertion after Cosmos evidence, not
   only target/hole motion. If occlusion prevents ruling out target-assisted
   insertion, do not accept the run as success.
   Also check the first `live_eval.success=true` row in `action_trace.json`:
   it must be in `oracle_physical_dp_finisher` or
   `oracle_physical_manual_finisher`, after the finisher has started. If
   success appears before the physical finisher, reject it as pre-finisher /
   target-assisted metric success.

6. Invalid immediately:

   - no annotated video;
   - no Cosmos RGB prediction video;
   - no Cosmos action chunk;
   - no dynamic Cosmos-control stage;
   - fewer Cosmos dynamic actions than the configured minimum before finisher;
   - Diffusion Policy controls the dynamic target-motion stage;
   - Diffusion Policy executes any action after target motion has started;
   - peg / wooden stick teleports, snaps, disappears, penetrates a wall, or
     inserts into wall;
   - target / hole teleports or disappears;
   - target / hole motion creates insertion by moving onto a mostly stationary
     peg / wooden stick;
   - `peg_state_guard.ok` is missing or false;
   - final success appears only after simulator-state edit;
   - run stops at trigger / boundary without insertion attempt.

7. If a single case passes:

   - record it as one reviewed single-case success only;
   - write a verdict file that explicitly records
     `active_robot_insertion_confirmed`, `target_assisted_insertion_rejected`,
     `visual_full_insertion_confirmed`, `full_sequence_video_reviewed`,
     `video_covers_cosmos_control_and_finisher`,
     `video_covers_final_insertion_or_physical_failure`, and
     `overall_task_complete`;
   - do not mark overall Oracle complete;
   - continue with directional suite, peg/wooden-stick disturbance, and
     multiple approved `fix3_733` keys.
