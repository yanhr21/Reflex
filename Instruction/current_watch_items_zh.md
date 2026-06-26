# 当前观察项

Date: 2026-06-23

这个文件只保存当前要盯住的事情，避免后续会话又从旧结论绕回去。

## 当前最重要判断

1. 最新 panel `0,2,4,5` 已经跑完，结果是 `2/4`：sample04、sample05 成功，
   sample00、sample02 失败。它证明 source-suffix + live receding + DP
   handoff 这条接口“能成”，但还不能稳定。
1. sample 5 已经证明 corrected source-suffix + DP handoff 可以在一个真实
   full-episode live 样本上成功。
2. sample 00 有过一次 full-episode 成功，但同协议 panel 复跑已经失败。
   现在不能再说 sample00 已经稳定修复。offsets `64,48,32,24` 只能说明候选
   覆盖变好，不等于闭环稳定完成。
3. formal rank-loss scorer 已经跑满 3 小时但失败：验证集 selected handoff
   `0/8`，oracle `1/8`，没有 live-eval checkpoint。
4. server27 allocation `146639` 两次在第一帧 render 报 Vulkan `DeviceLost`；
   server56 allocation `146658` 设置系统 ICD 后跑通。
5. sample00 单次成功 run:
   `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_sample00_full_vkfix_20260622_alloc146658`。
   结果：`301` frames、full contract ok、`success=true`、final
   peg-head-at-hole `[-0.00718, 0.000003, 0.00185]`。
6. sample00 panel 复跑失败:
   `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658/sample_00_hole_late_move_stop`。
   结果：`301` frames、full contract ok、`success=false`、final
   peg-head-at-hole `[-0.09576, 0.01617, -0.06529]`。视觉 sheet 已打开，最后
   没插进去。
7. sample02 panel 也失败:
   `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658/sample_02_hole_late_reverse`。
   结果：`301` frames、full contract ok、`success=false`、final
   peg-head-at-hole `[-0.11293, 0.04707, -0.06106]`。视觉 sheet 已打开，最后
   没插进去。关键过程：frame `144/148` 达到 C_pi，但 DP96 后变成
   `[-0.11303, 0.03169, -0.05456]`，真实闭环 handoff 失败。
8. sample04 panel 成功:
   `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658/sample_04_hole_late_sine`。
   结果：`301` frames、full contract ok、`success=true`、final
   peg-head-at-hole `[-0.00451, -0.00294, -0.00294]`。视觉 sheet 已打开，
   没有和 metric success 冲突。关键过程：frame `160` 的 source-suffix
   chunk 把状态带到 C_pi，之后 DP96/DP42 完成。
9. sample05 panel 成功:
   `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_source_suffix_offsets64_48_32_24_panel0245_vkfix_20260622_alloc146658/sample_05_hole_late_continuous_insert`。
   结果：`301` frames、full contract ok、`success=true`、final
   peg-head-at-hole `[0.02941, -0.00266, -0.00263]`。视觉 sheet 已打开，
   和 metric success 一致。关键过程：多轮 source-suffix chunk 把状态带到
   C_pi，之后 DP96/DP52 完成。
10. panel0245 的 targeted handoff replay 已完成。核心数字：
    selected/source-suffix replay `97` rows，其中 `54` 个接 DP96 成功；
    source-suffix 名字候选 `53/76` 成功，live-selected 候选 `15/42` 成功；
    DP prior `16/42` 成功。最关键的是 `45` 个 `C_pi=false` 但 DP96 成功，
    以及 `2` 个 `C_pi=true` 但 DP96 失败。这确认当前 C_pi 只能做诊断，
    不能做 handoff 训练目标。
11. combined scorer dataset 已生成：
    `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_panel0245_offsets64_selected_sourcesuffix_dpprior_dp96_20260622_alloc146658`。
    它有 `139` rows、`42` live-state groups、全部 group 有 DP prior、`19`
    个 group 有 source-suffix DP96 正例。
12. short overfit sanity 已完成：
    `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_panel0245_offsets64_handoff_rank_overfit100_20260622_alloc146658`。
    它选出 `20/42` handoff success，DP prior 是 `16/42`，等于 handoff oracle
    的 `20/42`。这说明标签可学，但不是正式证据。
13. 两个 single-panel formal scorer 已中断：`h512` 和 `h4096` 都太小，
    GPU 利用率低，且早期 validation handoff success 低于 DP。它们不是正式
    结果。
14. 2026-06-23 发现一个 gate/split 配置问题：原
    union+panel formal scorer 的 validation split 是 DP `4/21`、handoff
    oracle `5/21`，最大可能提升只有 `1/21=0.0476`，但 gate 要求
    handoff_delta `>0.05`，所以这条 handoff gate 数学上不可达。这个结果不能
    被解释成训练收敛后的方法失败。
15. 当前正在跑的是修正 split 后的 union+panel 1-GPU 3-hour formal scorer
    对照，seed `20260725`，validation 是 DP `3/21`、handoff oracle `7/21`、
    headroom `+4`，train headroom `+3`。两个输出根：
    `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_source_suffix_union_plus_panel0245_handoff_rank_seed20260725_h2048_formal3h_20260623_alloc146658`
    和
    `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_source_suffix_union_plus_panel0245_handoff_rank_seed20260725_h8192_formal3h_20260623_alloc146658`。
    因为 h2048/h8192 的 GPU 利用率不稳定，又启动了同数据、同 split、同 gate
    的 h16384 positive-weight 对照：
    `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_source_suffix_union_plus_panel0245_handoff_rank_seed20260725_h16384_posw_formal3h_20260623_alloc146658`。
    要点：validation selected 需要至少 `5/21`，才会严格超过 DP 并满足
    `>0.05` handoff gate。
16. 这三条 scorer formal 都已经完成 3 小时并且
    `ready_for_formal_live_eval=true`。最佳结果：
    - h2048：best step `15000`，validation selected `6/21`、DP `3/21`、
      oracle `7/21`，handoff delta `0.143`，weighted error delta `-0.014`，
      progress delta `+0.0045`。
    - h8192：best step `9500`，selected `5/21`、DP `3/21`、oracle `7/21`。
    - h16384 positive-weight：best step `25900`，selected `6/21`、DP `3/21`、
      oracle `7/21`，但 progress delta 略负。
    当前优先使用 h2048 `checkpoint_best_gate.pt` 做小 live panel，因为它更小、
    best 指标更平衡。这个仍然只是 offline scorer 资格；方法证据必须来自
    live closed-loop final state 和视频/contact-sheet。
17. h2048 best-gate live panel 已完成：
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h2048_scorer_source_suffix_offsets64_48_32_24_panel0245_20260623_alloc146658`。
    结果仍是 `2/4`，full `301/300` contract ok，0 process failure。成功：
    sample02 reverse、sample04 sine。失败：sample00 move_stop、sample05
    continuous_insert。相对上一轮，sample02 被救回，但 sample05 从成功退化为
    失败，所以这不是净方法成功。
18. h8192 best-gate live panel 已完成：
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_scorer_source_suffix_offsets64_48_32_24_panel0245_20260623_alloc146658`。
    结果仍是 `2/4`，full `301/300` contract ok，0 process failure，contact
    sheet 已打开。成功：sample00 move_stop、sample05 continuous_insert。失败：
    sample02 reverse、sample04 sine。它和 h2048 是互补的：h2048 救 02/04，
    h8192 救 00/05。这说明候选/闭环接口有真实可用性，但单个 scorer 的选择
    泛化不稳，不能称为整体方法成功。
19. h16384 positive-weight best-gate live panel 已完成：
    `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h16384_posw_scorer_source_suffix_offsets64_48_32_24_panel0245_20260623_alloc146658`。
    结果仍是 `2/4`，full `301/300` contract ok，0 process failure，contact
    sheet 已打开。成功：sample00 move_stop、sample05 continuous_insert。失败：
    sample02 reverse、sample04 sine。它没有合并 h2048 的 02/04 成功，反而和
    h8192 一样只保住 00/05。
20. 三 scorer 离线组合诊断已完成：
    `experiments/world_model_task_rebinding/cosmos3/three_scorer_ensemble_compare_union_plus_panel0245_seed20260725_20260623_alloc146658`。
    在同一 union+panel DP96 标签验证集上，`mean_raw_score`/`mean_delta_vs_dp`
    只有 `5/21` handoff success；`max_delta_vs_dp` 到 `6/21`，和单个
    h2048/h16384_posw 的最佳水平一样，仍低于 handoff oracle `7/21`。简单
    ensemble 不能直接解决 live 互补失败。
21. targeted 02/04 DP96 replay 已完成：
    `experiments/world_model_task_rebinding/cosmos3/live_snapshot_replay_three_scorer_02_04_sourcesuffix_dp96_20260623_alloc146658`。
    三个 panel 的标签合计显示：h2048 有 `39` rows、`9` 个 DP96 success；
    h8192 有 `161` rows、`32` 个 DP96 success；h16384_posw 有 `133` rows、
    `9` 个 DP96 success。最关键的是 h8192/h16384 在 live panel 里 02/04 失败，
    但 saved candidate bank 里仍有可接 DP96 成功的候选。这把卡点进一步收紧到
    selector/feature/label，而不是“完全没有动作路”。
22. 第一次 02/04 转换结果不能用：
    `live_snapshot_outcome_scorer_training_three_scorer_02_04_sourcesuffix_dp96_20260623_alloc146658`
    只有 `24` 个 group，因为 converter 把 h2048/h8192/h16384 里同名
    sample/iter/prefix 合成了同一个 uuid。这个会混掉不同 live state 的 base
    feature 和候选标签，属于实现 bug。
23. converter 已修复 uuid namespace：`scripts/world_model/convert_cosmos3_live_snapshot_labels_for_outcome_scorer.py`
    现在把 panel 名写进 uuid，例如 `h2048__...`。干净输出是
    `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_three_scorer_02_04_sourcesuffix_dp96_fixuuid1_20260623_alloc146658`。
    它有 `44` 个 group、`333` 条有效候选、`50` 个 DP96 success、
    `10` 个 source-suffix DP96-success group；审计结果是
    `bad_uuid_multi_jsonl=0`、重复 `(uuid,candidate)` 为 `0`。
24. fixuuid 的 02/04 标签已经和旧 union+panel0245 数据合并：
    `experiments/world_model_task_rebinding/cosmos3/live_snapshot_outcome_scorer_training_source_suffix_union_plus_panel0245_plus0204fixuuid_dp96_20260623_alloc146658`。
    合并后是 `127` 个 group、`9349` 条候选；DP prior handoff success
    `25/127`，handoff oracle `44/127`，source-suffix handoff success group
    `32/127`。这说明训练集有真实 headroom，但不代表 scorer 已经会选。
25. 100-step h2048 sanity 已完成：
    `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h2048_rank_sanity100_seed20260725_20260623_alloc146658`。
    它不是 formal 证据。结果：train selected handoff `32/95`，DP `17/95`；
    val selected handoff `8/32`，DP `8/32`，oracle `12/32`。结论：数据可读、
    训练排序可学，但 100 step 没有 held-out 提升。
26. h8192 formal 已中断，不是 evidence：
    `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h8192_formal3h_seed20260725_20260623_alloc146658`。
    它只跑了约 `4` 分钟，GPU 利用率约 `22%`，低于集群释放风险阈值，所以用
    Ctrl-C 取消。不能 live eval，不能当 formal 结果。
27. h16384 positive-weight formal 已中断，不是 evidence：
    `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h16384_posw_formal3h_seed20260725_20260623_alloc146658`。
    这条跑到约 `20` 分钟后被取消，因为 `--allow-handoff-only-gate` 会让 step
    `2000` 的纯 handoff 提升 `10/32` vs DP `8/32` 保存 best gate，但 weighted
    error/progress 没有安全变好。这种 checkpoint 不能进 live eval。
28. h16384 positive-weight safegate formal 已完成并通过 strict offline gate：
    `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h16384_posw_safegate1_formal3h_seed20260725_20260623_alloc146658`。
    它去掉了 `--allow-handoff-only-gate`，所以 gate checkpoint 必须同时满足
    handoff 提升和 error/progress 安全约束。summary 显示
    `elapsed_seconds=10802`、`formal_training_floor_met=true`、
    `ready_for_formal_live_eval=true`。exact checkpoint 是
    `checkpoint_best_gate.pt`。best step `7000`：selected `11/32`、DP `8/32`、
    oracle `12/32`，error delta `-0.00590`，progress delta `+0.00261`。
    这只是 offline 资格，不是 live 方法证据。
29. h8192 safegate formal 也已完成并通过 strict offline gate：
    `experiments/world_model_task_rebinding/cosmos3/candidate_outcome_scorer_union_panel0245_plus0204fixuuid_h8192_safegate1_formal3h_seed20260725_20260623_alloc146658`。
    原因是 h16384 GPU 利用率有波动，且 h8192 是同数据同 split 同 strict gate
    的有用容量对照；不是换目标。summary 显示 `elapsed_seconds=10800`、
    `formal_training_floor_met=true`、`ready_for_formal_live_eval=true`。
    best step `7000`：selected `11/32`、DP `8/32`、oracle `12/32`，
    error delta `-0.00349`，progress delta `+0.00404`。
30. 11:24+08 两条 safegate 都已经有 strict gate checkpoint，但还没有 formal
    资格：
    h16384_posw step `8000` 是 selected `10/32`、DP `8/32`、oracle `12/32`，
    error delta `-0.00108`，progress delta `+0.00006`；
    h8192 step `3000` 是 selected `11/32`、DP `8/32`、oracle `12/32`，
    error delta `-0.00122`，progress delta `+0.00311`。这两个比之前的
    handoff-only gate 更干净。现在 h16384 和 h8192 都已满足 `10800` 秒 floor，
    两者都只提供 offline live-eval 资格，不是 live 方法证据。下一步要用 exact
    `checkpoint_best_gate.pt` 跑严格 full `301/300` live panel，并做视觉审计。
31. h8192 safegate live panel 已启动：
    `experiments/world_model_task_rebinding/cosmos3/launch_live_panel_seed20260725_h8192_safegate1_panel0134_20260623_alloc146658.sh`。
    Slurm step 是 `146658.148`，tmux window 是 `live_h8192_safegate0134`。
    它用 h8192 safegate exact `checkpoint_best_gate.pt`，跑 samples `0,1,3,4`，
    full `301/300`，source-suffix offsets `64,48,32,24`，8-step execute，
    DP96 handoff。结果必须等 final simulator state 和 contact sheet/video 审计，
    不能只看 process 成功。

## 下一步观察/动作

1. 当前最高优先级转为补真实 DP-rollout 标签和改 scorer 特征。三个已满足
   3 小时门槛的 scorer 都只做到 `2/4`：h2048 救 02/04，h8192 和
   h16384_posw 救 00/05。简单 ensemble 离线也没有超过单模型上限；继续单独
   换容量或直接平均没有新信息。
   现在已经补了一批 02/04 真实 DP96 标签，并启动合并后的 h8192 formal。
   下一步只看这条 formal 的 held-out handoff 是否超过 DP；未过 gate 就不能
   live eval。
2. 同步记录一个更深判断：当前 C_pi 不是可靠 continuability 标签。sample00
   iter0 的 selected source-suffix 候选 C_pi=false 且 y/z 变差，但
   `candidate + DP96` replay 成功；panel 复跑里 sample00 和 sample02 都出现
   C_pi=true 后 DP96 把真实状态带坏。后续 scorer/handoff 必须学真实
   DP-rollout continuability 和 contact/insertion 质量，而不是继续手调几何
   gate。
3. 旧 panel、h2048 panel、h8192 panel 都是 `2/4`，但成功样本不同。当前最
   直接的问题不是“Cosmos 全量 SFT 完全没学”，而是 scorer 在相同候选池上对
   哪个短动作能被 DP 接住判断不稳。
4. 失败样本的共同物理形态：y/z 往往能对齐，但 x 方向没有把 peg 推进孔内，
   或者短暂触发 DP 后 DP rollout 把 peg 带坏。这是“可完成短动作/handoff
   质量判断”问题，不是靠更密集 re-observation 或手写错误恢复能直接解决。

## 不能做的事

- 不能把这个改成手写错误恢复表。
- 不能放宽成功定义或 final-state gate。
- 不能把 state-only/oracle replay 包装成主方法结果。
- 不能在登录节点跑 rollout、render、preflight、training 或 eval。
- 不能把 sample00 单次成功和 sample05 成功说成广泛成功；sample00 panel 复跑
  已经失败。
