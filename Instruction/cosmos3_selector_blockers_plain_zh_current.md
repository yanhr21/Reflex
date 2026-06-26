# Cosmos3 Selector/Executor 当前卡点中文入口

Date: 2026-06-22

这个文件保存用户要求长期可见的总体判断。它是给后续 agent
快速读懂当前局面的中文入口，不替代具体 evidence note、TODO 或实验
summary。

## 一句话判断

现在不是“方法已经成功，只差扩大实验”。现在暴露出来的核心问题是：
离线 replay 里确实存在一些能把状态带到 DP 可接手区域的动作 headroom，
但当前 live 候选生成和 scorer 还不能稳定把这些 headroom 转成真实闭环插入
成功。

下一步必须围绕两个问题定位：

- live 可生成/可执行候选里到底有没有好动作；
- 如果有，selector 为什么选不出来。

不要回到换阈值、手写错误恢复、direct Cosmos 长动作 rollout，或者把
state-only/oracle 结果包装成主方法。

## 主要卡点

1. selector 泛化差。

   训练集 selected handoff 可以很高，但验证集没有稳定提升，甚至更差。
   这是当前 selector 线最核心的问题。

2. replay oracle 和 live 可执行动作之间有落差。

   旧 h96 union 的 oracle 候选来源包含 `teacher_scale`、
   `legacy_teacher_scale`、`retrieval_success_residual` 等候选族。它们不一定
   都是 live loop 里真实能生成和执行的候选。因此旧 `60/64` oracle headroom
   可能部分来自离线知道的好动作族，而不是当前 live checkpoint executor
   能稳定提出的动作。

3. gate 可能太容易放过坏 checkpoint。

   如果 gate 主要看 selected handoff 是否超过 DP，就可能放过“handoff 数字
   略好，但几何误差或接触进度更差”的 checkpoint。插孔任务里这很危险，
   因为 DP handoff 前的位置、姿态、接触质量会直接决定后续能不能插进去。

4. validation 太小。

   旧 scorer 只有 `64` groups、验证 `16` groups。`+1/16` 或 `+2/16`
   看起来像提升，但统计方差很大，不能支撑强结论。

5. source/scenario split 泄漏风险需要继续防。

   scorer 主要按 `uuid` split。JSONL 里还有 `source_uuid`、`scenario`、
   `current_phase`。如果同一 source 或同类 scenario 的不同 prefix 同时进入
   train/val，验证可能高估泛化。2026-06-21 audit 没发现 `source_uuid`
   overlap，但 scenario/phase 仍有重合，且验证集小，所以不能放松。

6. live panel 视觉审计必须结构化写回。

   仅仅说打开过 contact sheet 不够。以后如果出现成功，summary/manifest
   必须写清楚 review sheet/video 路径、review 状态、peg 是否持住、是否真
   插入、最终真实 simulator state。否则不能算 major success。

7. repo 状态很脏。

   当前有很多 modified/untracked 脚本、文档、wrapper 和实验输出。稳定方法
   逻辑需要整理并提交，否则后续复现风险很高。不能因为 worktree 脏就改动或
   回滚用户已有内容。

## 还要继续查的问题

1. live candidate family 覆盖不足。

   必须做 live-candidate-only headroom。旧离线 oracle 里赢的候选不一定来自
   live executor 会生成的 family。如果只保留 live 可用候选后 oracle 大幅
   下降，问题就是候选生成器/executor，不是 scorer 阈值。

2. DP handoff label 不一定等价于真实闭环 handoff。

   replay/restored 状态下 DP 能接上，不代表真实闭环累积误差后 DP 还能接上。
   最新 panel 已经出现 DP handoff 发生但最终失败的情况，所以必须以真实
   closed-loop final state 和视觉证据为准。

3. scorer 输入描述可能过于抽象。

   live 中大量候选以 `checkpoint_model` 描述进入 scorer；离线数据里 source
   family 更丰富。scorer 可能学到和 live 分布不一致的 family prior。

4. panel 太小，容易误判方向。

   `1/4` 和 `0/4` 不能证明方法有效，但足够说明不能宣称成功。先用更严格
   offline/margin gate 缩小试错，再跑 live。

5. state/simulator 信息只能做标签和诊断。

   simulator state 可以用于训练标签、causal metadata、诊断和视觉读出监督。
   最终 controller evidence 必须来自 RGB/RGB-D-derived state/slot 输入和真实
   live rollout，不能把 oracle/state-only 结果当主方法。

## 当前已更新的事实

- 旧 `145813`/`145814` formal scorer 已经是历史。它们没有给出可直接用于
  live eval 的强证据。
- 2026-06-22 source-suffix 修复产生了一个严格边界下的 sample05 full-episode
  live 成功。这证明接口能工作一次，但不是广泛成功。
- 同一严格边界下，sample00 iter0 的 `213/213` 候选全部无效，`0` after-gate，
  `0` y/z improvement。这证明 sample00 是 action-candidate coverage 问题。
- sample05 严格 all-candidate replay 有很多可 handoff 候选，且 after-gate
  + DP96 中 `118/214` 成功。这证明 sample05 是 selector/continuability
  scoring 问题。
- 7-state short sanity 显示 plain regression 不够，必须有 within-state
  rank loss。但那个小数据集只有一个 DP96-success 状态，而且 DP prior 也成功，
  所以不是正式 selector 证据。
- 当前正在 allocation `145920` step `145920.315` 上跑正式 rank-loss scorer：
  `41` live states、`8877` candidates、其中 `3` 个状态是 DP prior 失败但
  non-DP candidate DP96 成功。这个 run 只验证 selector 侧；它不能解决
  sample00 的 action coverage 缺口。
- 当前 rank-loss scorer 的 split 还有一个重要信号：验证集唯一 DP96-success
  状态是 sample05 iter05；训练集里已经有同一 source/scenario 的 sample05
  iter04 正例，以及 sample04 iter05 正例。也就是说，当前验证失败不是因为
  验证正例完全来自陌生场景；即便有相近正例在训练集里，selector 仍没有稳定
  选出验证集正候选。
- 运行约 `46` 分钟、step `13000` 时，中期读数仍是训练集 selected handoff
  `2/33`，验证集 selected handoff `0/8`，验证 oracle `1/8`，且验证
  weighted error/contact progress 相对 DP 仍为负面。这个中期趋势继续支持
  selector 泛化差，但正式判断仍要等 `10800` 秒 summary。
- 运行约 `76` 分钟、step `22000` 时，训练 rank loss 已经接近 `0`，训练集
  仍 selected handoff `2/33`，验证集仍 `0/8`。这说明 rank objective 已经把
  训练排序学进去了，但没有迁移到验证的相邻 live state。
- 运行约 `106` 分钟、step `32000` 时，验证 selected handoff 仍是 `0/8`。
  weighted error 相对 DP 略好，但 contact progress 仍差，且仍未选中验证
  oracle 的 handoff 正例。
- 运行约 `134` 分钟、step `42000` 时，验证 selected handoff 仍是 `0/8`，
  验证 oracle 仍是 `1/8`，contact progress 仍差。到这个时间点还没有可用
  live eval gate。
- 运行约 `168` 分钟、step `52000` 时，验证 selected handoff 仍是 `0/8`。
  这已经非常接近正式 `3` 小时 floor；除非最后 summary 意外改变，否则这个
  run 将是 selector 泛化失败证据，而不是 live eval 入口。
- 正式 summary 已出：`formal_training_floor_met=true`，`ready_for_formal_live_eval=false`，
  `best_gate_metrics=null`，最终验证 selected handoff `0/8`，DP `0/8`，
  oracle `1/8`。这个 rank1 formal run 失败，不能进入 live eval。
- sample00 offset 覆盖审计完成：当前失败的 sample00 iter0 使用
  source-suffix offsets `32,24`，因此 `source_suffix_candidate_count=0`。
  同场景 source bank 里 offsets `32,24` 的最近距离是 `0.02073`，刚好超过
  `0.02` cap；但同场景全 offset 最近距离是 `0.00652`，且 offsets `48,64`
  有多个候选进入 `0.02`。这说明 sample00 当前不是“阈值太严”，而是使用的
  suffix offset 覆盖不够。
- sample00 offset 诊断重跑已经尝试启动，但 allocation `145920` 被回收，
  step `145920.316` 只运行 `4` 秒并取消。这个没有产生方法结果，只是资源
  /调度失败。
- sample00 offset 修复在 allocation `146658` 上有过一次 full run 成功。
  这说明 offsets `64,48,32,24` 能让必要 source-suffix 候选进入候选集。
  但同协议 panel 复跑 sample00 已经失败：完成 `301` frames，合同有效，
  最终真实 simulator `success=false`，final peg-head-at-hole
  `[-0.09576, 0.01617, -0.06529]`，视觉 sheet 确认没有插入。关键过程是
  frame `168` C_pi=true 后执行 DP96，但 DP96 把真实状态带到
  `[-0.09734, 0.00605, -0.04949]`，后续 continuability 断掉。
  所以现在不能说 sample00 稳定修复；只能说早期 source-suffix 覆盖改善。
- 同时，sample00 one-iteration replay 暴露了更深问题：selected source-suffix
  candidate 在 8 步后 C_pi=false 且 y/z 变差，但接 DP96 仍成功。这说明当前
  C_pi/几何 gate 不是足够好的 handoff 标签；panel 复跑又说明 C_pi=true
  也不保证 DP96 能完成。后续应该学真实 DP-rollout continuability 和
  contact/insertion 质量，而不是继续手调 threshold。
- 同协议 panel 的 sample02 也失败：完整 `301` frames，最终真实 simulator
  `success=false`，final peg-head-at-hole `[-0.11293, 0.04707, -0.06106]`。
  过程上 frame `144/148` 达到 C_pi 后执行 DP96，但 DP96 后状态为
  `[-0.11303, 0.03169, -0.05456]`，没有插入；视觉 sheet 已打开并确认失败。
  sample02 还暴露了 selector 问题：frame `136/144` 已有 `2` 个 source-suffix
  候选，但 scorer 仍选 checkpoint-model `scale_1.5`。这把当前卡点进一步收敛
  到真实 handoff/contact continuability 标签和 late candidate coverage。
- 同协议 panel 的 sample04 是正例：完整 `301` frames，最终真实 simulator
  `success=true`，final peg-head-at-hole `[-0.00451, -0.00294, -0.00294]`。
  过程上 frame `160` 的 source-suffix chunk 把状态带过 C_pi，DP96/DP42 后
  完成插入；视觉 sheet 已打开，没有和 metric success 冲突。这个正例说明
  当前接口不是完全无效，但它不能抵消 sample00/sample02 的失败。当前结论应
  写成 mixed evidence：能成功，不能稳定成功。
- 同协议 panel 的 sample05 也是正例：完整 `301` frames，最终真实 simulator
  `success=true`，final peg-head-at-hole `[0.02941, -0.00266, -0.00263]`。
  过程上多轮 source-suffix chunk 把状态带过 C_pi，随后 DP96/DP52 保持成功；
  视觉 sheet 已打开并支持 metric success。最终 panel 结果是
  `sample00=false`、`sample02=false`、`sample04=true`、`sample05=true`，
  即 `2/4`。这不是失败到底，也不是方法成功；它把问题定位成
  handoff/continuability 不稳定。
- panel0245 的 handoff-label replay 已完成，数字更直接：source-suffix 名字
  候选 `53/76` 接 DP96 成功，live-selected 候选只有 `15/42` 成功，DP prior
  是 `16/42`。同时有 `45` 个 `C_pi=false` 但 DP96 成功，`2` 个 `C_pi=true`
  但 DP96 失败。结论很明确：当前 C_pi 不是可靠训练目标；必须学真实
  `candidate chunk + DP rollout` 的成功/continuability/contact 标签。
- 新的 combined scorer dataset 已生成：`139` rows、`42` live-state groups、
  每个 group 有 DP prior，`19` 个 group 有 source-suffix DP96 正例。短
  overfit sanity 能选到 `20/42` handoff success，超过 DP prior `16/42`，
  等于 handoff oracle `20/42`。这只证明标签可学，不证明泛化。
- 两个 single-panel formal scorer 已经中断，因为数据太小、GPU 利用率低，
  且早期 validation handoff success 低于 DP。当前有效 formal 是 old
  source-suffix union + new panel0245 labels 的 namespaced 合并训练：
  `9016` rows、`83` groups。step `1` 只是初始化，validation selected
  handoff success 和 DP 持平为 `4/21`。最终要等 3 小时 summary。

Evidence note:

`docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_rank1_formal_selector_failure.md`

Follow-up offline audit:

- in the formal rank1 label union, all `18` success rows and all `3` success
  groups are source-suffix candidates;
- checkpoint-model non-DP candidates have `0/41` DP96-success groups;
- train/val share source/scenario/phase, so the validation miss is not caused
  by a completely unseen scenario;
- the final scorer selected one source-suffix candidate but missed the
  source-suffix handoff oracle.

Audit note:

`docs/world_model_task_rebinding/cosmos3_lowfreq_wm_executor/2026-06-22_rank1_failure_offline_audits.md`

## 下一步决策

1. 把 sample00 单次成功、sample00 panel 失败、sample02 失败、sample04 成功、
   sample05 成功作为同一组 mixed evidence 记录，避免单边解读。

2. 不重复同一 panel 作为下一步。第一批真实 `candidate chunk + DP rollout`
   标签已经生成；当前要看 union+panel formal scorer 能不能在 held-out live
   states 上选出比 DP prior 更好的 handoff candidate。

3. 后续 scorer/handoff 改法：训练真实 DP-rollout continuability target，
   并加入 contact/insertion progress 质量。不能再把 instantaneous C_pi 当唯一
   handoff 标签。
   不能再把 instantaneous C_pi 当唯一 handoff 标签。
