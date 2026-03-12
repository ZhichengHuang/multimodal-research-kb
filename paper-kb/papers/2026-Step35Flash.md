---
title: "Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters"
authors: [StepFun AI]
date: 2026-02
venue: arxiv
url: "https://arxiv.org/abs/2602.10604"
tags: [architecture, moe, scaling, rl, pretraining, posttraining, training-stability, reasoning]
category: "pretraining/moe-architecture"
level: 2
status: read
importance: medium
problem_tree_nodes: [RL-2a, RL-3c]
aliases: ["Step35Flash", "Step-3.5-Flash", "Step3.5Flash"]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结
196B 参数（11B 激活）稀疏 MoE 语言模型，通过 S3F1 混合注意力（3:1 SWA:Full）+ Head-wise Gating + MTP 多 token 预测实现高效推理，采用 MIS-PO（Metropolis Independence Sampling 过滤策略优化）替代传统重要性加权实现稳定大规模 RL，在 IMO-AnswerBench 85.4% / LiveCodeBench-v6 86.4% 达到开源 SOTA。

## 核心 Insight
1. **S3F1 混合注意力 + Head-wise Gating**: 3 层 SWA（W=512）+ 1 层 Full Attention 的 3:1 交替布局，通过增加 SWA 层 query head 数量（64→96）和数据依赖的 head-wise gating 补偿 SWA 信息丢失，在预训练质量上超越全 Full Attention 基线（55.7 vs 54.1），推理 FLOPs 仅为全注意力的 ~1.01×。
2. **稀疏 MoE 训练稳定性的系统诊断**: 三大不稳定模式——Muon Polar Express bfloat16 数值敏感、Expert Collapse（激活归零而非 routing collapse）、深层激活爆炸——及对应的系统性解决方案（float16 cast、per-expert activation/parameter norm 监控、activation clipping），实现全程仅 1 次 loss spike 的训练稳定性。
3. **MIS-PO 离散过滤替代连续重要性加权**: 在 token 和轨迹层面用离散分布过滤替代连续重要性权重，将优化限制在稳定信赖域内，大幅降低梯度方差，适用于长时推理 RL。

## 与已有工作的关系
- **继承自**: DeepSeek-R1（GRPO 算法基础）、DeepSeek-V3（MoE 架构模式）
- **扩展**: [[2025-KimiK2.5]] MoE 架构和多 token 预测——Step 3.5F 以 1/3 参数规模实现相当性能; [[2026-GLM-5]] MoE 训练稳定性诊断——Expert Collapse 监测、Activation Clipping 等机制
- **对比**:
  - vs [[2026-GLM-5]]: 都是大规模 MoE RL 基础模型。GLM-5: 744B/40B 激活, 聚焦 Agentic RL + Slime 框架; Step 3.5F: 196B/11B 激活, 聚焦纯文本推理 + MIS-PO。Step 3.5F 展示了更激进的参数效率
  - vs [[2025-Qwen3-VL]]: 都是大规模 AR 模型但方向不同。Step 3.5F 纯文本+编码推理; Qwen3-VL 多模态视觉理解。不同焦点, 难以直接性能对比
  - vs [[2026-Beyond-LM]]: 追求 Frontier 性能但路线不同。Step 3.5F: AR LLM + MoE 稀疏激活（成熟范式优化）; Beyond-LM: 从零训练 dLLM + flow matching（新范式探索）
- **互补**:
  - 与 [[2026-EBPO]] 正交互补: EBPO 改进 advantage baseline 估计方差 vs MIS-PO 改进 importance weight 方差，可叠加应用
  - 与 [[2026-LFPO]] 正交互补: LFPO 无似然度梯度优化 vs MIS-PO 离散重要性采样过滤，两者都是 GRPO 改进但维度不同
  - 与 [[2025-KimiK2.5]] 方法论互补: K2.5 的 Toggle RL（25-30% 减少输出 token）与 Step 3.5F 的 MTP 位置感知 loss 都在降低生成成本

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 架构
- **规模**: 196B 总参数, 11B 激活参数
- **结构**: 45 层 Transformer（3 dense + 42 MoE 层），每 MoE 层 288 routed experts + 1 shared expert，k=8 专家激活/token
- **注意力**: GQA-8（8 个 KV head），混合布局 S3F1（3 SWA W=512 + 1 Full Attention）
- **Head-wise Gating**: SWA 层 query head 从 64 增至 96 + 可学习数据依赖 gating 机制，补偿 SWA 信息丢失
- **MTP**: 3 个轻量 MTP head（SWA + dense FFN），仅增加 0.81B 参数（~0.41%），主训练仅优化 MTP-1，MTP-2/3 从 MTP-1 克隆后轻量联合微调

### 并行策略
- **硬件**: 4,096 × NVIDIA H800 GPU，NVLink/NVSwitch 内节点 + 8×200 Gbps RoCE 跨节点
- **并行**: 8-way Pipeline Parallelism + 8-way Expert Parallelism + ZeRO-1 Data Parallelism
- **解耦并行**: Attention 和 MoE 模块独立并行策略，各自数据并行组内梯度归约
- **通信优化**: Fabric-aware scheduling（NVLink/RoCE 分阶段）+ Communication-aware placement（最小化跨交换机跳数），共减 ~5% 迭代时间
- **Muon ZeRO-1 resharding**: 将完整参数分配到 DP rank（而非拆分），消除梯度 unsharding 冲突，减 ~5% 迭代开销

### 预训练
- **总 token**: 17.2T 高质量 tokens
- **数据组成**: 通用知识（web/books/documents）、代码（纯代码 + PR/issue/commit）、数学/STEM、工具使用和推理数据
- **训练阶段**:
  1. 预训练: 4K context, 初始数据
  2. 中间训练: 上下文扩展至 128K, 合成 agentic 数据强化
  3. 后训练: SFT + RL

### MoE 负载均衡
- **EP-level balancing**: ℒ_EP = G ∑(g=1→G) f_g p_g，显式促进 rank-group 级别利用率均衡
- **全局批次级别统计**: 优于微批次约束，避免局部不均衡传播

### 训练稳定性系统诊断
1. **Muon 数值敏感**: Polar Express 迭代（Newton-Schulz 正交化）在 bfloat16 下产生极端异常值 → 仅将 Polar Express 迭代 cast 到 float16，其余保持混合精度
2. **Expert Collapse（非 routing collapse）**: 专家激活归零但 routing 统计稳定 → 显式 scaling 校准共享/路由专家贡献 + 全局批次平衡 + per-expert activation norm（RMS/mean at FFN intermediate）和参数 norm 监控
3. **深层激活爆炸**: 深 MoE 层重尾激活分布 → Activation clipping（FFN intermediate 逐元素裁剪，优于 weight clipping）

**结果**: 全程仅 1 次 loss spike

### 后训练: MIS-PO
- **核心思想**: 用离散分布过滤替代连续重要性加权，在 token 和轨迹层面限制优化在稳定信赖域内
- **优势**: 大幅降低梯度方差，同时保留学习信号，适用于长时推理 RL
- **Reward 集成**:
  - Verifiable rewards（数学正确性、代码执行）
  - Non-verifiable rewards（偏好反馈）
  - Agent rewards（任务完成度）
  - GenRM + MetaRM 组合训练

### 上下文扩展
- 4K → 32K → 64K (SFT) → 128K（最终部署）

### 推理优化
- **Speculative Decoding**: SWA 保留标准注意力语义，天然支持 KV masking 并行验证；GQA-8 使注意力更 memory-bandwidth bound，创造计算余量吸收 speculative drafting
- **MTP 加速**: 3 个 MTP head 用于 speculative decoding，acceptance length 未在此论文报告，但设计思路与 GLM-5 的 MTP 2.76 类似
- **部署性能**: Hopper GPU 上 ~170 tokens/s（OpenRouter 首周数据）

## Building Blocks（可复用组件）

### Block 1: S3F1 混合注意力 + Head-wise Gating
- **做法**: 3 层 SWA（W=512）交替 1 层 Full Attention 的 3:1 布局。SWA 层 query head 数从 64 增至 96（增加 50%），引入可学习数据依赖的 head-wise gating 机制
- **机制 (WHY it works)**: SWA 以窗口外信息丢失为代价换取 O(L·W) 线性复杂度。关键洞察: 不是所有 attention head 都需要全局信息——通过增加 SWA head 数量扩大局部信息捕获带宽，head-wise gating 让模型学习哪些 head 需要全局 vs 局部信息。3:1 比例使每 4 层有一层提供全局信息整合机会（类似金字塔结构）
- **适用条件**: 长上下文推理场景（128K+），需要低延迟推理的 agent 部署; 与 speculative decoding 兼容（SWA 支持 KV masking 验证）
- **什么时候会 break**: 任务强依赖全局注意力（如需要全序列 token 间交互的任务），3:1 比例中 SWA 层可能丢失关键远距离依赖; W=512 的窗口对极短序列可能过大（浪费计算）
- **可组合方向**: 可与 dLLM 的 Prefix-DLM KV cache 方案结合——SWA 天然与 KV cache 兼容; 与 Sparse-LaViDa 的 step-causal mask 正交组合; 与 Beyond-LM 的 Hybrid Attention Masking（帧内双向+跨序列因果）对比——两者在不同层面处理注意力效率

### Block 2: 稀疏 MoE 训练稳定性诊断框架
- **做法**: 系统监控三层信号——(1) per-expert activation norm（RMS/mean at FFN intermediate）及 max-to-median ratio（最可靠的 expert death 早期指标）; (2) 参数 Frobenius norm; (3) loss spike 模式归因（Muon 数值/expert collapse/激活爆炸）。对应三种干预: float16 cast / scaling 校准 / activation clipping
- **机制 (WHY it works)**: Expert collapse 的根因链: 高频 bi-gram 触发专家专门化 → pre-norm 允许单专家主导 → SwiGLU 稀疏激活放大幅度 → Muon 低秩更新加速 collapse。理解因果链才能精确干预（如 activation clipping 直接切断放大环节）
- **适用条件**: 大规模 MoE 模型（100B+ 参数）的长时间预训练; 使用 Muon 优化器的场景
- **什么时候会 break**: 小规模 MoE（10B 以下）可能不出现这些不稳定模式; 非 Muon 优化器的数值敏感模式可能不同
- **可组合方向**: 对 Beyond-LM 的 MoE per-modality shared experts 训练有直接参考——多模态 MoE 可能面临更复杂的 expert collapse（模态不平衡加剧）; 对 dLLM 统一模型未来的 MoE scaling 提供稳定性工具箱

### Block 3: MIS-PO（离散过滤策略优化）
- **做法**: Metropolis Independence Sampling 过滤——在 token 和轨迹层面用离散分布过滤替代连续重要性加权，仅对信赖域内的样本进行优化更新
- **机制 (WHY it works)**: 传统重要性加权（如 PPO/GRPO 的 importance ratio clipping）在 off-policy 程度大时权重方差爆炸。MIS-PO 将连续权重离散化为"接受/拒绝"，完全消除高方差权重项。离散过滤比连续加权更鲁棒，虽然会丢弃部分可用数据，但保证了优化方向的可靠性
- **适用条件**: 长时推理 RL（agent 任务、多步推理）中策略漂移严重的场景; 大规模异步 RL 训练
- **什么时候会 break**: 数据极度稀缺时过滤会丢弃过多样本（需要足够大的 rollout 批次）; 信赖域阈值过严导致学习停滞，过松则失去稳定性优势
- **可组合方向**: 与 GLM-5 IcePop（连续重要性校正）形成两种范式对比——离散过滤 vs 连续校正; 可替代 dLLM RL 中的 GRPO clipping（UniGRPO、answer-forcing GRPO）作为更稳定的 off-policy 处理; 与 EBPO shrinkage baseline 正交（MIS-PO 处理 off-policy 问题，EBPO 降低 advantage 方差）

### Block 4: MTP 多 token 预测（Speculative Decoding 适配）
- **做法**: 3 个轻量 MTP head（各含 SWA + dense FFN），仅增加 0.81B 参数（~0.41%）。主训练仅优化 MTP-1（预测 t+2），MTP-2/3 从 MTP-1 克隆后位置依赖 loss 重加权轻量联合微调
- **机制 (WHY it works)**: MTP head 共享主模型的表示，仅需学习"偏移映射"（从 t 到 t+1+h），参数开销极小。位置依赖 loss 重加权防止远距离预测（t+3、t+4）过度优化导致近距离预测退化。克隆+微调避免从零训练 MTP-2/3 的不稳定性
- **适用条件**: 需要低延迟推理的 agent 部署; 与 GQA 和 SWA 兼容的推理框架
- **什么时候会 break**: 生成任务中下一 token 高度不确定时（如开放式创意写作），speculative decoding 的 acceptance rate 可能很低; MTP head 与主模型表示耦合，模型更新后需重新微调
- **可组合方向**: 与 GLM-5 的 MTP 3 层参数共享方案对比（Step 3.5 Flash 克隆+微调 vs GLM-5 共享参数训练）; 对 dLLM 不适用（dLLM 并行去噪不需要 speculative decoding），但 MTP 的损失设计思想（位置依赖加权）可用于 dLLM 的 multi-step denoising loss 设计

### Block 5: EP-level Load Balancing
- **做法**: 在 Expert Parallel 级别引入 ℒ_EP = G ∑ f_g p_g 损失，显式促进 rank-group 级别利用率均衡; 结合 loss-free 全局批次统计平衡
- **机制 (WHY it works)**: 传统 token-level 或 micro-batch-level 平衡可能在局部均衡但跨 GPU rank 不均，导致分布式训练中 straggler 问题。EP-level 损失直接优化跨 rank 均衡，消除通信等待瓶颈
- **适用条件**: 大规模分布式 MoE 训练（100+ GPU）; Expert Parallelism 部署时
- **什么时候会 break**: 单机训练不需要 EP-level 平衡; 过强的 EP 损失可能牺牲专家专业化程度
- **可组合方向**: 对 Beyond-LM 的 MoE per-modality shared experts 训练有参考——多模态场景下 EP 平衡需考虑模态路由偏好; 与 Qwen3-VL 32B-MoE（64 experts, top-8）的负载均衡策略对比

## Anti-patterns / 已知失败模式
- **SWA 简单交替布局（S1F1/S3F1 without Head scaling）性能退化**: 朴素 S3F1 在代码和长上下文上分别退化 0.7/1.3 分，必须配合 head 数增加 + gating 机制才能超越 Full Attention
- **Expert Collapse ≠ Routing Collapse**: 传统监控 routing 统计（gate probability, dispatch counts）不足以发现 expert death，必须监控 activation norm——这是 GLM-5 同样观察到的问题（两篇独立验证）
- **Muon bfloat16 Polar Express 数值爆炸**: Muon 优化器的 Newton-Schulz 正交化在 bfloat16 下有极端尾部异常值，必须对该特定操作 cast 到 float16（不影响整体混合精度）
- **深层 MoE 激活重尾分布**: Weight clipping 不足以防止激活爆炸复发，必须使用 Activation clipping（直接在 FFN intermediate 层裁剪）
- **Micro-batch balancing 局部均衡但全局不均**: 微批次级别的 MoE 负载均衡可能导致跨 rank 负载失衡，必须在全局批次级别统计

## 实验关键发现
- **S3F1+Head 超越 Full Attention**: 预训练综合分数 55.7 vs Full Attention 54.1，仅 1.01× FLOPs 开销（vs Full Attention 2.68×）
- **Head-wise Gating**: 在 100B-A10B 规模将平均性能从 62.46 提至 64.43（+1.97）
- **预训练 vs 更大模型**: 1/3 参数（11B 激活）在 SimpleQA 上达 31.6%，超越更大模型
- **IMO-AnswerBench**: 85.4%
- **LiveCodeBench-v6**: 86.4%（2024.08-2025.05）
- **τ²-Bench**: 88.2%
- **BrowseComp**: 69.0%（with context management）
- **Terminal-Bench 2.0**: 51.0%
- **推理速度**: Hopper GPU ~170 tokens/s（OpenRouter 首周）
- **训练稳定性**: 17.2T tokens 全程仅 1 次 loss spike

## Relations (结构化)
- `extends` → [[2025-KimiK2.5]]: 同为大规模 MoE 模型，Step 3.5F 以 196B/11B（vs K2.5 1.04T/32B）实现 frontier 性能; MTP 位置依赖 loss 重加权扩展 K2.5 的 MTP 范式
- `extends` → [[2026-GLM-5]]: Expert Collapse 诊断（activation norm 监控而非 routing stats）与 GLM-5 独立验证; Activation Clipping 方案一致; MTP clone+finetune vs GLM-5 shared-parameter 两种设计
- `extends` → [[DeepSeek-R1]]: GRPO 算法基础，MIS-PO 是在 GRPO 重要性加权基础上的离散过滤改进
- `alternative_to` → [[2026-GLM-5]]: MIS-PO（离散过滤）vs IcePop（连续校正）——两种 off-policy 处理范式; Slime 异步解耦框架 vs MIS-PO 内部信赖域过滤——两种 RL 稳定性策略
- `alternative_to` → [[2026-Beyond-LM]]: AR MoE 稀疏激活（成熟范式效率极致）vs 从零训练 dLLM + flow matching（新范式探索），代表两条 frontier 路线
- `alternative_to` → [[2025-Qwen3-VL]]: 大规模 AR 模型但方向不同——纯文本推理 vs 多模态视觉理解; 11B 激活 MoE vs 32B-MoE/8B dense
- `alternative_to` → [[2025-LaViDa-O]]: 容量分配策略不同——均匀 MoE routing 自然分化 vs Elastic-MoT 显式非对称分支（8B 理解+2.4B 生成）
- `combines_with` → [[2026-EBPO]]: MIS-PO（off-policy 方差降低）与 EBPO shrinkage baseline（advantage 方差降低）正交互补，可叠加使用——先 MIS-PO 过滤样本，再 EBPO 收缩 advantage
- `combines_with` → [[2026-LFPO]]: MIS-PO（离散过滤 RL）与 LFPO（无似然度 velocity-based RL）都是 GRPO 改进但维度不同，理论上可组合
- `combines_with` → [[2025-MMaDA]]: MIS-PO 离散过滤可替代 UniGRPO 的重要性 ratio clipping，为 dLLM RL 提供更稳定的 off-policy 处理
- `combines_with` → [[2026-LaViDa-R1]]: MIS-PO 在轨迹级过滤 + answer-forcing 在信号恢复层面互补——MIS-PO 过滤过激时 answer-forcing 注入正样本
- `alternative_to` → [[2026-Seed2.0]]: 同期 frontier 模型——Step 3.5 Flash 196B/11B MoE 纯文本公开 MIS-PO/S3F1，Seed2.0 多模态但闭源；数学推理和编码能力互为参考

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- **长时推理 RL 中的 off-policy 梯度方差爆炸**: MIS-PO 用离散过滤替代连续重要性加权，在 IMO 85.4% / LiveCodeBench 86.4% 的 frontier 难度下保持训练稳定
- **大规模 MoE 训练稳定性的系统诊断**: 建立了"Expert Collapse ≠ Routing Collapse"的认知——routing 统计正常但专家激活归零，提供 activation norm 监控 + activation clipping 的完整诊断-干预框架
- **SWA 信息丢失与注意力效率的 tradeoff**: S3F1 + Head-wise Gating 首次证明 SWA 可通过 head 数量增加 + 数据依赖 gating 超越 Full Attention 基线（55.7 vs 54.1），FLOPs 仅为 1.01×

### 未解决的问题
- 问题: MIS-PO 信赖域阈值的任务适应性
  - 为什么难: 固定阈值假设所有任务有相似的策略漂移速度，但数学推理（低熵策略）vs 创意生成（高熵策略）的 off-policy ratio 分布差异巨大
  - 潜在思路: 自适应阈值——基于当前策略熵或 rollout age 动态调整; per-task 或 per-reward-type 阈值

- 问题: S3F1 3:1 比例的最优性缺乏理论支撑
  - 为什么难: 论文未消融 1:1 vs 2:1 vs 5:1 比例; 不清楚信息需求在深度方向如何分布
  - 潜在思路: 可学习的 SWA/Full 比例分配; 内容感知的动态比例（图像 patch 更需要局部注意力 vs 文本 token 更需要全局注意力）

- 问题: Expert Collapse 诊断框架对非 Muon 优化器的泛化性
  - 为什么难: 因果链中"Muon 低秩更新加速 collapse"是 Muon 特有的；AdamW 等优化器是否有不同的 collapse 模式未知
  - 潜在思路: 在 AdamW/LAMB 等优化器上复现 expert collapse 实验，建立优化器无关的诊断框架

- 问题: MIS-PO 离散过滤 vs IcePop 连续校正的系统对比
  - 为什么难: 两篇论文使用不同架构（196B vs 744B）、不同任务、不同评估基准
  - 潜在思路: 控制变量实验——在同一模型和任务上对比 MIS-PO vs IcePop vs 标准 GRPO clipping

### 对问题树的推进
- 推进了 [[problem-tree#RL-2a]] 🟡（补充新范式）: MIS-PO 离散过滤为 off-policy RL 提供第五种范式（vs UniGRPO 结构化 mask / complementary masking 似然度估计 / EBPO shrinkage baseline / LFPO 无似然度），特别适用于策略漂移严重的长时推理场景
- 推进了 [[problem-tree#RL-3c]] 🟡（补充 AR→dLLM 技术迁移）: MIS-PO 是架构无关的 RL 改进组件（与 EBPO shrinkage 类似），可直接迁移到 dLLM RL；Expert Collapse 诊断框架可迁移到 Beyond-LM 的多模态 MoE
- 新增问题 [RL-2l]: MIS-PO 离散过滤 vs 连续重要性加权的理论统一——何时离散过滤优于连续加权？是否存在最优离散化粒度？
- 新增问题 [RL-2m]: MIS-PO 在异构 Reward 分布（verifiable vs non-verifiable vs agent）下的阈值收缩——单一阈值是否适用于 reward 分布差异极大的多模态场景？
- 新增问题 [PT-8]: 稀疏 MoE 训练稳定性诊断向多模态推广——模态不平衡（Beyond-LM 51:1）下 expert collapse 是否呈现新模式？是否需要 per-modality 诊断策略？

## 个人深度评注

> 以下由 Critic + Ideator Agent 联合分析

### 一、对 KB 的核心价值判定

Step 3.5 Flash 是纯文本 AR LLM，不是多模态论文。对本 KB（聚焦多模态 dLLM）的价值在于**三项可迁移方法论组件**:

| 组件 | 迁移目标 | 价值判定 |
|---|---|---|
| MIS-PO 离散过滤 | dLLM RL 的 off-policy 稳定性 | **中高** — 架构无关，可嵌入 UniGRPO/answer-forcing GRPO，但需适配 dLLM 高熵分布 |
| Expert Collapse 诊断框架 | 多模态 MoE scaling 的训练稳定性 | **高** — 与 GLM-5 独立验证，是迄今最完整的 MoE 稳定性工具箱 |
| S3F1 混合注意力 | dLLM 长序列推理效率 | **低中** — SWA 与 dLLM 双向注意力不兼容（SWA 假设因果性），但 head-wise gating 思想可迁移 |
| MTP clone+finetune | dLLM 不适用 | **低** — dLLM 并行去噪不需要 speculative decoding，但位置依赖 loss 思想可借鉴 |
| EP-level Load Balancing | 多模态 MoE 分布式训练 | **中** — 需适配模态路由偏好 |

### 二、MIS-PO vs IcePop: 离散 vs 连续的根本性 tradeoff

| 维度 | MIS-PO（离散过滤） | IcePop（连续校正） |
|---|---|---|
| 机制 | 接受/拒绝二元决策 | 平滑重要性权重校正 |
| 方差降低 | 极致（仅 in-distribution 样本参与） | 有效但不彻底（仍有权重项） |
| 信息保留 | 丢弃 off-policy 样本 | 重加权而非丢弃 |
| 超参敏感度 | 信赖域阈值（过严→学习停滞） | β, ε_low, ε_high 组合 |
| 最适场景 | 长时 RL（off-policy 严重） | 短-中时 RL（部分 off-policy 样本仍有价值） |

**判断**: 对 dLLM RL，image token NLL>6 导致 off-policy 严重，MIS-PO 的激进过滤可能更适合。但需解决阈值自适应问题（per-modality 阈值：text τ=0.9, image τ=0.7）。

### 三、Expert Collapse 诊断对多模态 MoE 的预警

Step 3.5 + GLM-5 独立验证了"Expert Collapse ≠ Routing Collapse"。对 KB 中的多模态 MoE 工作（Beyond-LM per-modality shared experts, Qwen3-VL 32B-MoE, K2.5 384 experts）有三层警示:

1. **不能只监控 routing 统计**: gate probability 和 dispatch count 正常不代表专家健康，必须监控 per-expert activation norm
2. **多模态不平衡加剧 collapse 风险**: Beyond-LM 51:1 视觉:语言数据比可能导致语言专家过度专化而视觉专家饿死，需要 per-modality activation norm 监控
3. **Activation clipping 可能需要 per-modality 阈值**: 视觉 FFN intermediate 和语言 FFN intermediate 的激活分布可能不同

### 四、S3F1 对 dLLM 的有限但启发性价值

S3F1 的 SWA 假设因果注意力（AR 模型），与 dLLM 的双向注意力不兼容。但两个思想可迁移:
1. **Head-wise Gating**: dLLM 可用类似机制让不同 head 学习"何时需要全局双向 vs 局部窗口"——这与 Sparse-LaViDa 的 step-causal mask（clean token 间双向 + mask token 仅看 clean token）在哲学上相似
2. **深度异构注意力**: 3:1 比例的启示是"不是所有层都需要相同注意力模式"——dLLM 可在浅层用廉价局部注意力、深层用昂贵全局注意力

### 五、Critic 分析的隐含失败模式

1. **MIS-PO 训练冷启动**: 早期训练策略接近随机，几乎所有样本 off-policy ratio 超出信赖域 → 大量拒绝 → 训练信号饥饿。论文未报告早期学习曲线
2. **Head-wise Gating 初始化 collapse**: 可学习 gating 在训练初期缺乏区分全局/局部需求的梯度信号，可能 collapse 到"所有 head 局部"的局部最优
3. **Activation clipping 引入梯度不连续**: 裁剪边界处梯度跳变（非零→零），长期训练可能导致优化轨迹震荡
4. **Expert Collapse 可能是 feature 而非 bug**: 适度的专家专化（"分号专家""缩进专家"）是期望行为，论文未区分"合理专化"和"过度 collapse"的边界
