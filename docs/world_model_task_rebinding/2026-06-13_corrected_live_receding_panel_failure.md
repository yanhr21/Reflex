# Corrected Live-Receding Panel Failure

Date: 2026-06-13 CST

## Trigger

The user instructed to stop further SFT/smoke work and directly advance
closed-loop eval. If closed-loop eval failed, the instruction was to stop and
analyze the failure instead of continuing speculative runs.

## Run

Output root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_panel10_corrected_iter2100_20260613_161006`

Slurm:

- allocation/job: `127350`, node `server10`, 1 GPU;
- eval step: `127350.46`;
- step was stopped with foreground process termination, not `scancel`, after
  corrected closed-loop failure was evident on the first moving-target samples.

Configuration:

- checkpoint: `iter_000002100`;
- eval manifest: `eval_full_episode_wam_iter_000002100/eval_input_manifest.json`;
- requested samples: `0,1,2,3,4,5,6,7,8,9`;
- live interface: `run_cosmos3_live_receding_panel.py` ->
  `run_cosmos3_live_receding_loop.py`;
- pretrigger: frozen DP until causal target-motion onset;
- external target mode: replay source `box_with_hole` actor pose only;
- receding iterations: `3`;
- action chunk: `8` steps;
- optional DP handoff horizon: `32`, gated by real-state `C_pi`;
- no SFT training was running or resumed.

## Protocol Evidence

The corrected live-history bug was not present in this run. Every inspected
live-prefix JSONL used:

`source_summary.source=provided_live_history_action_rows_only`

Observed prefixes:

- sample `00`, prefixes `106`, `114`, `122`;
- sample `01`, prefixes `94`, `102`, `110`;
- sample `02`, prefixes `104`, `112` before the stopped run.

The run therefore exercised the intended corrected loop:

1. frozen DP produced the observed pre-motion prefix;
2. target motion was detected causally;
3. Cosmos generated a short robot-action chunk;
4. only that chunk was executed in the live simulator;
5. the next Cosmos call conditioned on real live history;
6. frozen DP was not allowed because `C_pi` did not pass.

This is not the old one-shot `8 + 96` DP takeover diagnostic.

## Metrics

Sample `00` (`hole_late_move_stop`):

- target trigger: frame `106`, first moving frame `105`;
- completed iterations: `3`;
- final live success: `false`;
- total DP handoff steps: `0`;
- final peg-head-at-hole: `[-0.2652, 0.0979, 0.0020]`.

Per iteration:

- after iter 0: `[-0.1788, 0.0540, -0.0055]`;
- after iter 1: `[-0.2491, 0.0910, 0.0012]`;
- after iter 2: `[-0.2652, 0.0979, 0.0020]`.

Sample `01` (`hole_late_constant`):

- target trigger: frame `94`, first moving frame `93`;
- completed iterations: `3`;
- final live success: `false`;
- total DP handoff steps: `0`;
- final peg-head-at-hole: `[-0.2281, 0.0420, 0.0020]`.

Per iteration:

- after iter 0: `[-0.1934, 0.0445, -0.0177]`;
- after iter 1: `[-0.2262, 0.0494, -0.0094]`;
- after iter 2: `[-0.2281, 0.0420, 0.0020]`.

Sample `02` (`hole_late_reverse`) was interrupted after failure was already
visible:

- target trigger: frame `104`, first moving frame `103`;
- partial iterations: `1`;
- partial live success: `false`;
- total DP handoff steps: `0`;
- partial peg-head-at-hole: `[-0.3051, 0.1426, -0.0081]`.

## Visual Evidence

Contact sheet opened directly:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_panel10_corrected_iter2100_20260613_161006/closed_loop_failure_sample00_01_live_rollout_sheet.png`

Visual review agrees with the live metrics. In samples `00` and `01`, the
target moves, but the gripper/peg trajectory does not rebind to the moved hole.
The final frames show the peg outside the hole rather than an insertion or a
DP-continuable state.

Predicted-vs-live sheet opened directly:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_panel10_corrected_iter2100_20260613_161006/closed_loop_failure_live_vs_cosmos_predictions_sheet.png`

This sheet compares the real live rollout against each per-iteration
`vision.mp4`. The generated videos show the target/hole moving, but they do
not show a clear robot/peg rebind trajectory toward the moved hole. The
failure is therefore not just an action-extractor converting a visually
correct predicted robot trajectory into bad actions; the video prediction and
the executable action chunk both lack a convincing task-frame rebind.

Additional numeric artifact:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_panel10_corrected_iter2100_20260613_161006/closed_loop_failure_action_sidecar_analysis.json`

This records predicted action chunks versus source-teacher actions and the
predicted sidecar state versus the real post-execution state.

## Failure Analysis

This failure is not explained by the earlier eval bug:

- source-H5 future rows are not used as live WAM conditions;
- the target is replayed as an external moving object while robot/peg remain
  live;
- DP observation after target replay is repaired;
- DP is not doing a blind long takeover.

The failure is therefore a controller-facing model/action failure under the
corrected protocol:

1. Cosmos action chunks do not steer the robot/peg into the moved target frame.
   The relative peg-head-at-hole error worsens across receding iterations in
   the first moving-target sample and remains far outside the static-DP success
   manifold in the second.
2. `C_pi` correctly blocks frozen DP handoff because the real state is not
   close enough to the hole and the peg is not in a policy-continuable pose.
3. More SFT is not the immediate next action without first inspecting why the
   action head predicts chunks that are inconsistent with the required
   task-frame rebinding.

Failure localization from the action/sidecar analysis:

- sample `00`, iter 0: predicted robot-action RMSE versus the source teacher
  rows `106:114` is `0.0601`; real post-chunk peg-head-at-hole is
  `[-0.1788, 0.0540, -0.0055]`, while the predicted sidecar at the same end
  row is more optimistic at `[-0.1331, 0.0343, -0.0079]`.
- sample `00`, iter 2: predicted mean absolute x/y/z action is
  `[0.0393, 0.0494, 0.0200]`, while the source teacher rows require
  `[0.1299, 0.1217, 0.0323]`; the executed live state remains
  `[-0.2652, 0.0979, 0.0020]`.
- sample `01` also fails all three receding chunks. The action RMSEs versus
  source teacher rows are `0.0666`, `0.0392`, and `0.0441`; the final real
  peg-head-at-hole remains `[-0.2281, 0.0420, 0.0020]`.

The sidecar rows show that Cosmos often predicts the moving hole itself
reasonably, but the generated robot/peg trajectory does not become a valid
task-frame rebind. In some rows the sidecar is more optimistic than the live
simulator result, so generated state readout cannot be used as the authority
for DP handoff.

One remaining interface bug was found during this analysis. The SFT/eval
caption includes current target/hole, peg, TCP, peg-head-at-hole, observed
hole velocity, grasp, and insertion state. The live-prefix builder's
`history_action_path` branch had fallen back to a generic prompt because it no
longer had a source-H5 prefix payload. The structured sidecar still contained
the current live task state, so this does not invalidate the live-history
repair, but it is a train/eval schema drift. The builder is now repaired to
reconstruct the same style of causal geometry caption from the live history
row `prefix_frame_index - 1`. A local builder-only probe on sample `00`,
prefix `106`, produced a prompt with the live target/hole xyz
`[-0.0526, 0.2546, 0.0940]`, peg-head-at-hole
`[-0.1508, 0.0388, -0.0170]`, and observed target velocity
`[0.0016, 0.0029, 0.0000]`. No Cosmos inference or simulator rollout was
rerun after this fix.

An additional metadata-only bug was fixed for future exports: the
full-episode exporter computed correct z-score statistics in the correct
column order, but wrote old `prefix_*` vector names in
`normalization_stats.json`. Future full-episode exports now publish the
matching `task_*` vector names. The current root's numeric normalization is
unchanged and remains usable because the column order matches.

## Prompt-Fixed Minimal Recheck

After repairing the live-prefix prompt, a single-sample corrected live-receding
check was run, without any SFT continuation:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_promptfix_sample0_iter2100_20260613`

It used the same checkpoint `iter_000002100`, the same target-motion onset
trigger at frame `106`, the same 3 receding iterations, and the same 8-step
Cosmos chunks. The only intended interface change was that the live-prefix
JSONL prompt now contained current live geometry from the observed history:
current target/hole xyz `[-0.0526, 0.2546, 0.0940]`, current peg xyz
`[-0.0774, 0.0082, 0.0765]`, current TCP xyz
`[-0.0692, -0.0463, 0.0773]`, peg-head-at-hole
`[-0.1545, 0.0406, -0.0180]`, and observed target velocity
`[0.0016, 0.0029, 0.0000]`.

Result:

- final live success: `false`;
- final peg-head-at-hole: `[-0.2666, 0.0985, 0.0029]`;
- DP handoff: still blocked by real-state `C_pi` on every iteration;
- visual sheet opened directly:
  `live_receding_promptfix_sample0_iter2100_20260613/live_receding_panel_contact_sheet.png`;
- predicted-vs-live sheet opened directly:
  `live_receding_promptfix_sample0_iter2100_20260613/promptfix_live_vs_cosmos_predictions_sheet.png`;
- action/sidecar numeric artifact:
  `live_receding_promptfix_sample0_iter2100_20260613/promptfix_action_sidecar_analysis.json`.

The prompt-fixed run is essentially the same failure as before. Per-iteration
real post-chunk peg-head-at-hole values were:

- iter 0: `[-0.1815, 0.0550, -0.0055]`;
- iter 1: `[-0.2518, 0.0918, 0.0018]`;
- iter 2: `[-0.2666, 0.0985, 0.0029]`.

The action analysis again shows under-reaction in the late dynamic segment:
sample `00`, iter `2`, predicted mean absolute xyz action is
`[0.0380, 0.0538, 0.0211]` versus source-teacher
`[0.1299, 0.1217, 0.0323]`.

Therefore the live prompt mismatch was a real interface bug and is now fixed,
but it is not the main cause of the corrected closed-loop failure. With the
prompt fixed, this checkpoint still does not produce executable rebind chunks
for the moving target.

## Long-Horizon Closed-Loop Recheck

The initial prompt-fixed 3-chunk check was useful but not by itself enough to
declare sample `00` fully failed, because the source teacher inserts later.
In the source H5,
`hole_late_move_stop_seed3280649_idx2518.fix3_traj_0`, the target starts
moving at frame `105`, the live/Cosmos prefix is frame `106`, and the first
source insertion frame is `166`.

Per the user's instruction to push closed-loop eval rather than keep training
or smoke-testing, a longer single-sample closed-loop run was executed:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_rgb_300step_fix1recipe_4gpu_20260612_191745/live_receding_promptfix_sample0_longhorizon_iter2100_20260613`

Configuration:

- checkpoint: `iter_000002100`;
- sample: `00`, `hole_late_move_stop`;
- prefix start: causal target-motion onset, frame `106`;
- pretrigger: live frozen DP from source frame `0` until target motion;
- external target mode: source `box_with_hole` actor pose only;
- receding iterations: `12`;
- action chunk: `8` steps;
- final live frame: `202`;
- optional DP handoff: `32` steps, still gated by real-state `C_pi`;
- SFT/training was not resumed.

Result:

- final live success: `false`;
- final live peg-head-at-hole: `[-0.1423, 0.0004, 0.0101]`;
- completed iterations: `12`;
- DP handoff executed steps: `0`;
- final observed video:
  `sample_00_hole_late_move_stop/live_observed_rollout.mp4`;
- panel sheet opened directly:
  `live_receding_panel_contact_sheet.png`;
- dense 18-frame rollout sheet opened directly:
  `sample_00_hole_late_move_stop/live_observed_rollout_dense_sheet.png`.

Visual review of the dense sheet agrees with the metrics. The target moves,
the robot follows after it, but at the frame where the source teacher has
already inserted (`f166`) the live peg is still visibly outside the hole. At
the final frame `f202`, the peg still has not entered the hole.

The decisive source-versus-live comparison is:

- source first inserted frame: `166`;
- live frame `170` after receding iter `7`: `[-0.1525, 0.0002, 0.0102]`,
  `success=false`;
- live final frame `202`: `[-0.1423, 0.0004, 0.0101]`, `success=false`;
- source same-frame state at `202`: about
  `[-0.0056, 0.0005, -0.0022]`, `inserted=true`.

The final `C_pi` block is also justified by real state:

- grasped: `true`;
- target speed: `0`, so the target is settled;
- y is within the static-DP continuability band;
- x fails `min_rel_x=-0.1342566` because live x is `-0.1423`;
- z fails `max_abs_z=0.0038843` because live z is `0.0101`.

Therefore the DP handoff did not fail because of an overly stale observation
or because the code forgot to run DP. It was deliberately blocked because the
live state was not on the static-DP success manifold.

Action/source alignment in this long run explains the physical failure. The
first two chunks roughly match the teacher's coarse direction, but after the
target settles the model does not produce the sustained correction/insertion
actions required by the source:

- iter `2`, frames `122:130`: Cosmos mean absolute xyz action
  `[0.0386, 0.0487, 0.0191]`; source teacher
  `[0.1299, 0.1217, 0.0323]`.
- iter `7`, frames `162:170`: Cosmos mean absolute xyz action
  `[0.0478, 0.0105, 0.0226]`; source teacher
  `[0.0204, 0.2894, 0.0288]`; source is already inserted by frame `170`,
  while live remains at `[-0.1525, 0.0002, 0.0102]`.
- iter `11`, frames `194:202`: Cosmos mean absolute xyz action
  `[0.0047, 0.0310, 0.0239]`; source teacher
  `[0.0390, 0.1247, 0.0187]`.

This makes the current failure classification stronger:

1. The corrected closed-loop eval path runs the intended causal mechanism:
   live observation, target-motion trigger, short Cosmos action chunks, real
   re-observation, target-only replay, and real-state DP gate.
2. The current checkpoint can keep the peg held and can partially chase the
   moved target, but it stalls outside the final insertion manifold.
3. The immediate issue is executable action/rebind capability, especially the
   late post-motion insertion correction, not the old source-H5 conditioning
   bug, not a prompt-only mismatch, and not insufficient eval horizon.

## Additional Root-Cause Audit

After the long-horizon failure, a no-training, no-rollout audit checked whether
the negative result could be explained by a simple code mismatch.

### Action Indexing

No action-row temporal shift bug was found.

- The SFT config uses `shift_action=1`, but Cosmos3 implements this as the
  action diffusion/noise schedule shift, not as an action index offset.
- The Cosmos sequence packer states that `action[0]` aligns with vision
  frame `1`, i.e. action `i` advances video frame `i` to `i+1`.
- The full-episode evaluator uses the same convention: if the observed prefix
  ends at frame `f`, actions `[0, f)` are history and action `f` is the first
  future action.
- The live extractor therefore selects `[prefix_frame_index,
  prefix_frame_index + horizon)` correctly.

This does not rule out action quality problems; it only rules out the simple
"one-frame extractor offset" explanation.

### SFT Role/Mode Schema Drift

The exported data have a real semantic weakness. `prefix_role` is the intended
sampled training row type, while `mode` is the actual physical mode at that
prefix frame. These often disagree:

- `insert_resume`: `733` rows total, but only part are physically insert-ready.
  It includes `228` rows whose actual `mode` is `target_pre_motion`, `255`
  rows whose actual `mode` is `target_motion_observed`, and `81` rows whose
  actual `mode` is `target_post_motion`.
- `target_motion_observed`: `573` rows total, including `108` rows whose
  actual `mode` is already `target_post_motion`.
- `target_post_motion`: `573` rows total, including `55` rows whose actual
  `mode` is `insert_resume` and `110` rows whose actual `mode` is
  `peg_recovery`.

For the failed sample
`hole_late_move_stop_seed3280649_idx2518.fix3_traj_0`, the exported rows are:

- `target_pre_motion_f097`, actual mode `target_pre_motion`;
- `insert_resume_f108`, actual mode `target_motion_observed`;
- `target_motion_observed_f113`, actual mode `target_motion_observed`;
- `target_post_motion_f133`, actual mode `target_post_motion`.

The live loop, by contrast, infers the role from observed physical history and
therefore calls Cosmos with `target_motion_observed` at frames `106`, `114`,
and `122`, then `target_post_motion` from frame `130` onward. This is closer
to the true physical mode than the exported row name, but it means the live
conditioning text is not drawn from a clean, consistently labeled role
distribution. The model was trained on captions that sometimes say
`prefix_role=insert_resume mode=target_motion_observed`, and sometimes say
`prefix_role=target_motion_observed mode=target_post_motion`. That weakens the
mode-switch signal the controller is trying to use.

### Prefix Distribution And Off-Policy Receding Drift

The training export gives each source episode only a few hand-picked full
episode prefix masks, not a dense receding-state dataset. Prefix distributions
from the active condition root:

- `target_motion_observed`: min `65`, p50 `103`, p90 `128`, max `171`;
- `target_post_motion`: min `75`, p50 `132`, p90 `166`, max `210`;
- `insert_resume`: min `76`, p50 `97`, p90 `110`, max `135`.

The long-horizon live eval queries frames `106` through `194`, but after the
first few chunks the robot/peg state is no longer on the source teacher
trajectory. The model is then asked to recover from its own under-reaction.
The v7_733 data are DP-success-filtered teacher trajectories, not off-policy
receding corrections. They contain successful examples, but not enough
examples of "the model is late/outside the hole frame; recover in the next
8-step chunk." This is the exact regime reached in the failed live rollout.

### Later Checkpoints

The main root contains later checkpoints/evals up to `iter_000002700`. This
does not overturn the negative conclusion:

- `iter_000002700` has strict artifacts, but
  `closed_loop_gate_visual_review.json` records
  `closed_loop_allowed=false`, `visual_review_status=fail`;
- its manual visual review is `4` fail, `4` pass, `2` pass-with-caution;
- gate metrics remain controller-negative despite improvement over
  `iter_000002400`: mean robot-action future RMSE `0.6147`, state-sidecar
  future RMSE `0.4238`, mean final hole error `0.0669` m, and residual unsafe
  late robot/peg/hole geometry.

Therefore the long-horizon closed-loop failure on `iter_000002100` is not
best explained as "the agent stopped one checkpoint too early." The later
checkpoint evidence remains insufficient for controller handoff as well.

### Read-Only Training Distribution Audit

A reproducible no-training audit script was added:

`scripts/world_model/audit_cosmos3_receding_training_distribution.py`

It was run on the active condition root and wrote:

`docs/world_model_task_rebinding/2026-06-13_receding_training_distribution_audit.json`

Key findings:

- total rows: `2899`;
- role/mode mismatches: `1193` rows, `0.4115` fraction;
- action condition-mask errors: `0` examples found;
- late-rebind proxy rows: `1287`.

This makes the code/data boundary sharper. The action history masks are not
the observed failure. The training interface does contain many rows whose
conditioning role text does not match the actual physical mode, and the
current source rows are still sparse full-episode prefix masks rather than a
dense receding recovery curriculum.

## Boundary

This is negative corrected closed-loop evidence for the current
`iter_000002100` checkpoint/interface. It does not prove the overall method is
impossible, and it is not a full 10-sample completion. It does prove that, on
the inspected moving-target sample where the source teacher succeeds by
frame `166`, the current closed-loop Cosmos action path fails even when run
through frame `202` with repeated live re-observation.

The next aligned debugging step is failure localization, not more SFT:

- treat this as a model/action capability failure rather than a remaining
  live-prefix prompt bug;
- inspect whether the action objective underweights late dynamic rebind rows,
  whether the source data overrepresents DP-easy trajectories, or whether the
  controller needs a learned short-chunk executor instead of direct raw Cosmos
  actions;
- keep DP handoff gated by real state. Do not use generated sidecar optimism
  as a handoff authority.
