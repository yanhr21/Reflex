# Cosmos3 当前卡点与下一步计划

Date: 2026-06-23

这个文件保存当前 selector/executor 线最重要的判断。它是根目录
`Instruction/` 下的长期可见入口，不替代具体 TODO、实验 summary 或 evidence
note。

## 一句话判断

现在不是“方法已经成功，只差扩大实验”。正确问题已经暴露出来：离线存在
handoff-continuable 动作 headroom，但当前 live 候选生成和 scorer 还不能稳定
把这个 headroom 转成真实闭环插入成功。

下一步必须围绕 live-candidate-only headroom 和 selector 泛化定位，不能继续换
阈值、做手写恢复分支，或者退回 direct Cosmos action。

## 主要卡点

1. selector 泛化差。

   训练集 selected handoff 很高，验证集没有稳定提升，甚至更差。这是当前最核心
   的问题。

2. replay oracle 和 live 可执行动作之间有落差。

   h96 union 的 oracle 候选来源包含 `teacher_scale`、
   `legacy_teacher_scale`、`retrieval_success_residual` 等候选族。它们不一定
   就是 live loop 里真实能生成、能执行、会被当前 checkpoint executor 稳定提出
   的候选。

   换句话说，`60/64` oracle headroom 可能部分来自“离线知道的好动作族”，而不
   是当前 live executor 真能稳定拿出来的动作。

3. gate 可能太容易放过坏 checkpoint。

   如果训练脚本主要看 selected handoff 是否超过 DP，就可能允许“handoff 数字
   略好一点，但几何误差或接触进度更差”的 checkpoint。插孔任务里这很危险，
   因为 DP handoff 前的位置、接触姿态、插入轴进度会直接决定后续能不能完成。

4. validation 太小。

   当前一些 scorer 数据只有几十个 groups，例如 `64` groups、验证 `16` groups。
   `+1/16` 就会显得像提升，但统计方差很大，不能支撑强结论。

5. 可能存在 source/scenario split 泄漏风险。

   scorer 主要按 `uuid` split。JSONL 里还有 `source_uuid`、`scenario`、
   `current_phase`。如果同一 source/scenario 的不同 prefix 同时进入 train/val，
   val 可能高估泛化。当前结果已经差，说明真实问题只会更严重；这个必须专门
   audit。

6. live panel 视觉审计没有被结构化写回。

   文档里有“contact sheet opened”的记录，但 summary 里仍可能是
   `visual_review_status=needs_direct_agent_or_user_review`。失败时影响不大；
   以后如果出现成功，必须把视觉审计证据结构化记录清楚，否则不能作为 major
   success。

7. repo 状态很脏。

   有很多 modified/untracked 脚本、文档和实验 wrapper。当前逻辑如果不整理提交，
   后续复现风险很高。整理时不能回滚用户或已有 agent 的无关改动。

## 可能还没被充分发现的问题

1. live candidate family 覆盖不足。

   必须做 live-candidate-only h96 headroom。如果只保留 live 实际可用候选后
   oracle 从 `60/64` 掉很多，问题就不是 scorer 阈值，而是 candidate
   generator/executor 本身不够强。

2. DP handoff label 可能不等价于 live DP handoff。

   replay/restored 状态下 DP 能接上，不代表真实闭环 compounding error 后 DP 还
   能接上。已经出现过 DP_HANDOFF 发生但最终失败的 panel。

3. scorer 输入描述可能过于抽象。

   live 里很多候选以 `checkpoint_model` 描述进入 scorer，但离线数据里 source
   family 更丰富。scorer 可能学到的 family prior 和 live 分布不一致。

4. 当前 sample panel 太小，容易误判方向。

   `1/4`、`0/4`、`2/4` 都不足以证明方法有效，但足以说明不能宣称方法成功。下
   一步要先用更严格 offline/margin gate 缩小试错，再跑 live。

5. 当前方法仍依赖 simulator state 做 label/诊断。

   这允许用于训练标签、causal metadata 和诊断，但最终 controller evidence 仍
   需要 RGB/RGB-D-derived state/slot 输入。不能把 oracle/state-only 结果包装成
   主方法。

## 之后怎么办

1. 先确认 formal 训练自然跑满 3 小时并写出 summary，不要提前启动 live eval。

   只有 `training_summary.json` 显示 `ready_for_formal_live_eval=true`，并且
   margin/safegate eval 没有明显几何或 contact 退化，才允许跑小 panel。

2. 如果有 scorer 通过 formal gate：

   用 summary 指定的 exact `checkpoint_best_gate.pt` 跑 full `301/300` live panel。
   必须打开 contact sheet/video，记录最终 real simulator state、DP handoff step、
   candidate mode 分布和视觉结论。

3. 如果 scorer 都失败：

   不要继续烧 live GPU。先做三个离线审计：

   - live-candidate-only h96 headroom；
   - source/scenario/phase held-out split audit；
   - best_gate selected candidates 的 family 分布和 oracle-winner family 对比。

4. 如果 live-candidate-only headroom 低：

   问题在 candidate generator/executor。下一步应把 teacher/retrieval residual
   winner 蒸馏进 live executor，或者训练 phase/contact-conditioned executor，
   而不是继续调 scorer 阈值。

5. 如果 live-candidate-only headroom 高但 scorer 选不出来：

   问题在 scorer。需要改 split、features、loss/gate，要求 handoff improvement
   同时不恶化几何/contact progress，不能只看 handoff bit。

6. 整理证据和 repo 状态。

   当前 formal/live 结果出最终 summary 后，更新
   `TODO/cosmos3_lowfreq_wm_executor/00_active.md` 和对应
   `docs/world_model_task_rebinding/` evidence note。稳定脚本和文档应提交，避免
   后续 agent 在脏 worktree 上丢失实验逻辑。

## 当前执行口径

- 用户原始判断里提到的 `145813`、`145814` formal 训练已经是历史上下文，不能再
  当作当前等待对象。
- 当前有效的 formal gate 来自 allocation `146658` 上的 strict safegate runs：
  h16384 positive-weight safegate 和 h8192 safegate 都已跑满 3 小时，并且
  `ready_for_formal_live_eval=true`。
- 当前最高优先级是用 exact safegate checkpoint 做 full `301/300` live panel，
  然后以最终真实 simulator state 加 contact sheet/video 审计为准。
- 当前正在跟踪的 live panel 是 h8192 safegate panel0134：
  `experiments/world_model_task_rebinding/cosmos3/sft_full_episode_wam_fix3_v7_733_clean_dense_late299_stride8_rgb_300step_fix1recipe_4gpu_20260614_server35_pathfix1_priority_full_sft_max299/live_receding_candidate_executor_iter1500_seed20260725_h8192_safegate1_source_suffix_offsets64_48_32_24_panel0134_20260623_alloc146658`
- 这个 panel 仍只是小 panel。即使有成功，也必须写清楚视觉审计状态；如果失败，
  下一步优先做 live-candidate-only headroom 和 split/family 审计，而不是继续换
  阈值。
- 15:08+08 中期事实：panel0134 的 sample00 已完成，合同通过但最终失败：
  `301` frames、`300` actions、final simulator `success=false`、
  peg-head-at-hole `[-0.097125, 0.014449, -0.071617]`，
  `continuability_gate_ok_count=0/43`。这支持当前卡点判断：当前 live short
  chunks 没把状态推进到 DP-continuable insertion manifold。完整 panel 结论仍要
  等 sample01、sample03、sample04 全部完成。
- 15:38+08 中期事实：panel0134 的 sample01 也已完成并失败：
  `301/300` 合同通过，final simulator `success=false`，peg-head-at-hole
  `[-0.092171, 0.032896, -0.078168]`。它和 sample00 的失败类型不同：
  sample01 触发了 `96` 步 DP handoff，但仍没有最终插入。这说明当前
  continuability/handoff 判断仍会放过真实闭环里 DP 接不住的状态，不能只看
  handoff bit 或 restored replay 成败。
- 15:59+08 中期事实：panel0134 的 sample03 已完成并失败：
  `301/300` 合同通过，final simulator `success=false`，peg-head-at-hole
  `[-0.105599, 0.001399, 0.003859]`，且执行了 `96` 步 DP handoff。这个失败
  特别重要：y/z 几乎对齐，但 x 仍差约 `10.6cm`，所以 DP 接不上。这说明
  selector/scorer 必须学“插入轴推进 + contact continuability”，不能把 y/z
  对齐或 handoff bit 当成真正成功。
