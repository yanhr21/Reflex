# Cosmos3 Fresh Fix1 Failure Localization

## Scope

This note diagnoses the current fresh full1000 fix1 `iter_000001500` result.
It is not a controller claim. The physical question is why the generated
rollout still cannot safely trigger a target-motion mode switch or hand off an
executable peg/robot continuation despite passing strict full-length artifact
accounting.

## Inputs

- SFT root:
  `/public/home/yanhongru/ICLR2027/Reflex/experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_full1000_rgb_300step_fix1_4gpu_20260610_1131`
- Eval root:
  `.../eval_full_episode_wam_iter_000001500`
- Target-motion head diagnostic:
  `.../target_motion_readout_calibration_fresh_fix1_iter1500_20260610`

All generated/reference videos are `301/301` frames and all action tensors are
`300x32/300x32`; strict failures are empty.

## Aggregate Evidence

- Validation: best was iter1100 `0.607350`; final iter1500 was `0.662956`.
- Strict eval: action RMSE `0.3157882582`, robot-action future RMSE
  `0.8203911022`, state-sidecar future RMSE `0.0626562464`, future-video PSNR
  `23.1641813684`.
- Generated-RGB readout/profile: final hole error `0.0538804681` m, future
  hole/peg/TCP RMSE `0.0301422454` / `0.0386116191` / `0.0347898968` m,
  peg-head-hole RMSE `0.0413906728` m.
- Calibrated target-motion head: held-out reference RGB AUROC `0.9115353604`,
  F1@0.5 `0.7669897596`, best F1 `0.7788987602`; current fresh generated RGB
  AUROC `0.7808262378`, F1@0.5 `0.6005305040`, best F1 `0.6179090483`.

## Role-Wise Failure Table

| id | role | scenario | action | robot-act | state | psnr | final-hole | fut-hole | fut-peg | fut-tcp | peg-head-hole | onset@1cm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 00 | target_pre_motion | hole_move_stop | 0.2709 | 0.6745 | 0.0261 | 21.0239 | 0.1277 | 0.0723 | 0.0519 | 0.0454 | 0.0270 | 11 vs 89 |
| 01 | target_motion_observed | hole_move_stop | 0.3275 | 0.8346 | 0.0543 | 23.1214 | 0.0165 | 0.0167 | 0.0298 | 0.0262 | 0.0444 | 12 vs 85 |
| 02 | target_post_motion | hole_reverse | 0.5622 | 1.5997 | 0.0413 | 23.4305 | 0.0837 | 0.0386 | 0.0216 | 0.0360 | 0.0226 | 25 vs 107 |
| 03 | insert_resume | hole_move_stop | 0.2837 | 0.7413 | 0.0465 | 21.5299 | 0.0415 | 0.0239 | 0.0459 | 0.0471 | 0.0345 | 14 vs 92 |
| 04 | peg_recovery | peg_drop | 0.2559 | 0.6426 | 0.0973 | 21.4808 | 0.0541 | 0.0361 | 0.0671 | 0.0550 | 0.1064 | 91 vs None |
| 05 | peg_recovery | peg_disturb | 0.3473 | 0.8974 | 0.0699 | 21.4759 | 0.0679 | 0.0313 | 0.0440 | 0.0476 | 0.0404 | 48 vs None |
| 06 | static_monitor | none | 0.1707 | 0.4200 | 0.0386 | 23.4140 | 0.0228 | 0.0175 | 0.0445 | 0.0220 | 0.0515 | 19 vs None |
| 07 | static_late_monitor | none | 0.2724 | 0.7093 | 0.1930 | 30.7831 | 0.0227 | 0.0147 | 0.0167 | 0.0060 | 0.0327 | 16 vs None |
| 08 | target_pre_motion | hole_constant | 0.3115 | 0.7639 | 0.0275 | 23.1462 | 0.0603 | 0.0289 | 0.0357 | 0.0265 | 0.0233 | 19 vs 97 |
| 09 | insert_resume | none | 0.3558 | 0.9206 | 0.0319 | 22.2361 | 0.0416 | 0.0213 | 0.0288 | 0.0362 | 0.0310 | 24 vs None |

## Diagnosis

1. Target switch is not solved by threshold tuning. Low-threshold readout
   displacement fires tens of frames early and false-fires on no-target-motion
   samples. A calibrated temporal head is better on reference RGB, but drops
   sharply on current generated RGB/readout, so the generated rollout remains
   outside the switch calibration regime.
2. Target final pose is still a real rollout/readout problem in pre-motion and
   post-motion cases. Sample 00 has `0.1277` m final-hole error and sample 02
   has `0.0837` m, despite coherent-looking video.
3. The robot/action branch is still too inaccurate for executable chunks.
   Robot-action future RMSE is especially high on `target_post_motion`
   (`1.5997`) and remains high on insert-resume/recovery roles.
4. Peg recovery/contact is the hardest physical failure. Peg-drop has
   peg-head-hole RMSE `0.1064` m and the visual sheets do not show reliable
   regrasp/contact continuity. This cannot be fixed by a mode switch alone.

## Next Repair Direction

Do not start controller/DP integration from this checkpoint. The next repair
should preserve the full `301` RGB frame / `300` action contract and focus on:

- explicit target-motion/onset and final-target readout supervision over
  generated/latent features, not raw displacement thresholding;
- role-balanced or role-weighted SFT emphasis for pre-motion target forecast,
  post-motion continuation, and peg recovery;
- a controller-facing action/chunk head or action-distillation objective whose
  robot action quality is evaluated separately from the 25-D state sidecar;
- contact/grasp/peg-head-in-hole supervision and visual review for peg recovery
  before any DP handoff.

Evidence required for the next attempt: strict artifact accounting, generated
RGB/readout metrics, calibrated target-monitor metrics on generated RGB, robot
action metrics, and visual review showing peg contact continuity.

## 2026-06-10 Code Audit Addendum

The failure is not only "not enough training." The full-episode SFT dataloader
does use the JSONL causal `condition_frame_indexes_vision/action`, so fix1 was
not silently falling back to single-frame I2V. The concrete target-construction
bug is that the 32-D action/state token supervised true robot action only in
dims `0..6`; dims `7..31` repeated the prefix TCP/peg/hole/contact state at
every step. Future target/peg/TCP/contact state therefore was not directly
supervised in the Cosmos action branch.

Implemented repair:

- `export_cosmos3_maniskill_full_episode_wam_conditions.py` now supports
  `sidecar_target_mode=future_aligned_state`, where action row `i` aligns to
  video frame `i+1`. History rows before the causal prefix remain clean
  conditions; rows after the prefix are generated future action/task-state
  targets.
- The Slurm wrapper now records and passes
  `SIDECAR_TARGET_MODE=future_aligned_state`.
- The action-target audit now checks that future-aligned task-state sidecars
  vary over time, which should catch a repeat of the prefix-repeated bug.
- A role-weighted JSONL helper was added for optional full-episode row
  repetition, not data regeneration or clip slicing.

Fix2 should export a new condition root from the same approved full1000 RGB
source, run strict preflight/audit in a Slurm allocation, then SFT with the
same 4-GPU fix1 hyperparameters plus optional role weighting. Controller
integration remains blocked until generated validation videos/action/readout
and visual peg/contact review pass.
