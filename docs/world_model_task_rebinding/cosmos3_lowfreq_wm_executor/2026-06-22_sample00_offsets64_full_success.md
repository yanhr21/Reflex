# sample00 Offsets64 Full Success

Date: 2026-06-22

## Question

Was sample00 still a hard action-coverage failure, or did the source-suffix
offset audit identify a real missing candidate family?

## Resource Notes

The first rerun attempt on allocation `145920` did not execute because Slurm
revoked the allocation. A later rerun on allocation `146639` / server27 failed
twice at first-frame ManiSkill render with Vulkan `DeviceLost`.

The successful run used allocation `146658` / server56 and explicitly set:

`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`

## One-Iteration Diagnostic

Root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_sample00_oneiter_vkfix_20260622_alloc146658`

Compared with the old strict sample00 bank:

- old offsets `32,24`: `213` candidates, `0` source-suffix candidates;
- new offsets `64,48,32,24`: `215` candidates, `2` source-suffix candidates.

The scorer selected:

`retrieval_resid_srcsuffix_r0_s1_o48`

The selected chunk alone did not pass C_pi. It moved from
`[-0.1393, 0.0321, -0.0125]` to about
`[-0.1826, 0.0442, -0.0035]`, so y/z got worse and the one-iteration run was
not method success by itself.

## Candidate + DP96 Replay

Root:

`experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_sample00_offsets48_source_suffix_candidates_dp96_20260622_221837_alloc146658`

It replayed only the two new source-suffix candidates from sample00 iter0.

Summary:

- records: `2`;
- after-gate ok: `0/2`;
- y/z-improving: `0/2`;
- y/z-worsening: `2/2`;
- DP96 success: `1/2`;
- DP-continuable: `1/2`;
- final contact-stable: `1/2`.

The live-selected candidate was the positive one. Even though C_pi was false
after the 8-step chunk, frozen DP reached success in `48` steps with final
peg-head-at-hole:

`[-0.003523, 0.002995, 0.002409]`

This means current instantaneous C_pi is not a sufficient handoff learning
target. Handoff scoring needs real DP-rollout continuability labels.

## Full Live Run

Root:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_sample00_full_vkfix_20260622_alloc146658`

Result:

- `completed_iterations=8`;
- `final_observed_frames=301`;
- `full_episode_length_ok=true`;
- `video_file_contract_ok=true`;
- final simulator `success=true`;
- final peg-head-at-hole:
  `[-0.007180, 0.000003, 0.001853]`;
- controller frames:
  `DP_SCAN_TARGET=106`, `EXECUTOR_ACTIVE=44`, `DP_HANDOFF=150`, `INIT_OBS=1`.

Causal chain:

- iterations `0-5` selected source-suffix chunks;
- selected offsets were `48,48,32,32,24,24`;
- after iteration `5`, the real state reached
  `[-0.131838, 0.007562, -0.000056]` and crossed C_pi;
- frozen DP handoff then finished insertion.

Visual evidence:

`experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_sample00_full_vkfix_20260622_alloc146658/sample_00_hole_late_move_stop/live_observed_rollout_annotated_sheet.jpg`

The sheet was opened. It agrees with the metric: the robot keeps the peg and
the final frames show the peg at the moved hole/box region.

## Interpretation

sample00 was not an unsolved impossible state. The previous failure was a
source-suffix action-coverage miss caused by using only offsets `32,24`.
Adding offsets `64,48` under the same scenario match and the same `0.02`
distance cap made the required candidate family available.

This is still not broad method success. A later same-protocol panel rerun of
sample00 failed under the full `301/300` contract:

`docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_panel0245_offsets64_sample00_resample_failure.md`

Therefore the stable claim is narrower: offsets `64,48,32,24` improve
source-suffix candidate coverage and can produce success, but the closed-loop
handoff/contact behavior is not yet stable. Do not turn this into a
hand-written recovery rule or a looser success definition.
