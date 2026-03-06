---
title: "Beyond Language Modeling: Multimodal Pretraining"
authors: []
date: 2026-03
venue: arxiv
url: "https://arxiv.org/html/2603.03276v1"
tags: [pretraining, unified-model, architecture, moe, diffusion, generation, understanding]
category: "pretraining/unified-multimodal"
level: 3
status: read
importance: high
problem_tree_nodes: [PT-1a, PT-1c, PT-2, PT-3b, Tok-2a, Uni-2a, Uni-2b, Diff-1a]
aliases: ["Beyond-LM", "Multimodal-Pretraining-Scratch"]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结
从零训练统一多模态模型，系统性研究视觉-语言混合预训练的关键设计因素，证明 RAE 统一视觉表示 + MoE 架构可实现高效多模态扩展。

## 核心 Insight
视觉和语言数据是互补而非竞争关系，统一多模态预训练自然涌现世界建模能力；视觉比语言更数据饥渴（在万亿参数规模差距达 51×），MoE 通过模态专家自然分化解决容量需求不对称问题。

## 与已有工作的关系
<!-- 简要标注和知识库中已有论文的关系，使用 [[论文文件名]] wiki link 格式 -->
- **继承自**: 使用 RAE (SigLIP 2) 作为统一视觉编码器，继承自 SigLIP 系列；与 [[2025-LLaDA-V]] 共享 SigLIP2 技术
- **对比**:
  - vs [[2025-MMaDA]]: 从零训练 AR+连续扩散混合 vs LLaDA 初始化纯离散扩散
  - vs [[2025-Lumina-DiMOO]]: 从零训练+连续扩散+MoE vs LLaDA 初始化+离散扩散+大规模数据
  - vs [[2025-LaViDa-O]]: MoE 自动分化 vs Elastic-MoT 手动设计非对称分支
  - vs [[2025-Muddit]]: 从零训练（第四条路线）vs Vision-first 初始化
- **互补**:
  - 与 [[2026-LaViDa-R1]]: 预训练发现（视觉:语言 51:1）可指导 RL 数据配比
  - 与 [[2025-LLaDA-V]]: 共享 SigLIP2 编码器技术

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

**架构**: 统一 decoder-only Transformer，使用 modality-specific FFN layers（语言和视觉各自独立的 FFN）。

**视觉表示**: 使用 Representation Autoencoder (RAE) 作为统一视觉编码器，同时支持理解和生成任务。具体使用 SigLIP 2 作为语义编码器。

**训练目标**:
- 语言: next-token prediction (自回归)
- 视觉: flow matching (连续扩散过程)

**注意力机制**: Hybrid attention masking - 帧内双向注意力 + 跨序列因果注意力，平衡视觉理解和生成需求。

**MoE 设计**:
- Granularity G=16 (每 16 层共享一组专家)
- Per-modality shared experts (每个模态独立的共享专家，优于全局共享)
- 自然涌现模态专家分化

**数据配比**: 研究发现视觉数据需求远超语言，在万亿参数规模下视觉:语言数据比达 51:1。

## Building Blocks（可复用组件）

### Block 1: RAE 统一视觉表示
- **做法**: 使用 Representation Autoencoder (RAE) 作为单一视觉编码器，同时服务理解和生成任务。具体实现使用 SigLIP 2 semantic encoder。
- **机制 (WHY it works)**: RAE 学习的是语义级别的表示空间，既保留了判别性特征（用于理解），又保持了足够的重建信息（用于生成）。通过 autoencoding 目标强制表示的完整性。
- **适用条件**: 需要同时支持视觉理解和生成的统一模型；视觉数据量充足时效果更好。
- **什么时候会 break**: 当理解和生成任务对表示空间的要求严重冲突时（如需要极高分辨率生成细节 vs 高度抽象的语义理解）；数据量不足时可能无法学到平衡的表示。
- **可组合方向**: 可与不同的解码器架构组合（AR、Diffusion、Flow Matching）；可扩展到视频、3D 等其他视觉模态。

### Block 2: Modality-Specific FFN
- **做法**: 在统一 Transformer 中为不同模态使用独立的 FFN 层，共享 attention 层。
- **机制 (WHY it works)**: Attention 层捕获跨模态交互和序列依赖，FFN 层处理模态特定的特征变换。分离 FFN 允许每个模态学习专门的非线性映射，同时保持跨模态对齐。
- **适用条件**: 多模态数据分布差异较大时；需要在统一架构中保持模态特异性时。
- **什么时候会 break**: 模态间需要深度耦合的特征变换时；参数预算极度受限时（增加参数量）。
- **可组合方向**: 可与 MoE 结合实现更细粒度的专家分化；可扩展到更多模态（音频、3D 等）。

### Block 3: MoE with Per-Modality Shared Experts
- **做法**: 使用 Mixture-of-Experts，granularity G=16，每个模态有独立的 shared expert（而非全局共享）。
- **机制 (WHY it works)**: Per-modality shared experts 为每个模态提供基础能力保障，routing experts 实现任务级专业化。G=16 的粒度平衡了专家复用和专业化。自然涌现的模态专家分化减少了跨模态干扰。
- **适用条件**: 多模态数据容量需求不对称时（如视觉 vs 语言）；需要高效扩展参数规模时。
- **什么时候会 break**: 模态间数据量极度不平衡导致某些专家训练不足；routing 机制失效导致负载不均。
- **可组合方向**: 可与 modality-specific FFN 叠加使用；可调整 granularity 适应不同规模。

### Block 4: Flow Matching for Vision
- **做法**: 使用 flow matching 作为视觉生成目标，替代传统的像素级重建或离散 token 预测。
- **机制 (WHY it works)**: Flow matching 学习从噪声到数据的连续变换路径，比离散化保留更多视觉细节，比像素重建更高效。与 next-token prediction 统一在同一 Transformer 框架下。
- **适用条件**: 需要高质量视觉生成时；有足够计算资源训练连续扩散模型时。
- **什么时候会 break**: 极低分辨率或高度风格化的视觉任务可能不需要连续建模；推理速度要求极高时（需要多步采样）。
- **可组合方向**: 可与其他扩散变体（DDPM、Rectified Flow）替换；可扩展到视频、3D 生成。

### Block 5: Hybrid Attention Masking
- **做法**: 帧内使用双向注意力（bidirectional），跨序列使用因果注意力（causal）。
- **机制 (WHY it works)**: 双向注意力允许视觉 token 在帧内充分交互（类似 BERT），提升理解能力；因果注意力保持序列生成的自回归特性。混合设计兼顾两种需求。
- **适用条件**: 同时需要视觉理解（需要全局上下文）和序列生成（需要因果性）的任务。
- **什么时候会 break**: 纯生成任务可能不需要双向注意力（增加计算）；纯理解任务不需要因果约束。
- **可组合方向**: 可扩展到更复杂的 attention pattern（如 sliding window + bidirectional）；可与 sparse attention 结合。

## Anti-patterns / 已知失败模式
- **分离编码器陷阱**: 为理解和生成使用不同的视觉编码器会增加复杂度，RAE 证明统一表示更优。
- **全局共享专家**: MoE 中使用全局 shared expert（跨所有模态）不如 per-modality shared experts，因为无法提供模态特定的基础能力。
- **忽视数据比例**: 简单的 1:1 视觉-语言数据配比会导致次优结果，视觉需要显著更多数据（尤其在大规模时）。
- **过早引入领域数据**: 世界建模能力在通用预训练中自然涌现，过早引入大量领域特定数据（如 VQA）反而可能降低泛化性（饱和点在 ~1%）。
- **[RAE] 单一表示空间的生成-理解冲突**: 当理解任务需要高度抽象语义、生成任务需要细粒度纹理时，单一编码器可能无法同时最优化两者。
- **[RAE] 低级视觉任务瓶颈**: 语义编码器丢失像素级细节，super-resolution/dehazing 等任务表现弱（与 DiMOO 相同问题）。
- **[Modality-Specific FFN] 高层共享 attention 的表达能力限制**: 如果高层确实需要模态专用 attention（LaViDa-O 假设），���制共享可能次优。
- **[MoE] Routing collapse under extreme data imbalance**: 51:1 的视觉:语言数据比例可能导致 routing 退化为单模态主导。
- **[Flow Matching] 推理速度 vs 质量 tradeoff**: 连续扩散的多步 ODE 求解无法享受 masked diffusion 的并行采样和 KV 缓存加速。
- **[Hybrid Attention] 跨序列因果约束破坏多图推理**: 与 Prefix-DLM 相同问题——第一张图无法 attend 到第二张图。

## 实验关键发现
- **数据协同效应**: 多模态预训练在使用更少领域数据的情况下，VQA 性能超过单模态预训练，证明视觉-语言数据互补。
- **世界建模涌现**: 通用预训练自然产生世界建模能力，领域数据需求极少（~1% 即饱和）。
- **扩展律不对称**: Dense 模型中，视觉和语言的参数扩展指数差距为 0.10；MoE 将差距缩小到 0.05，有效协调了模态间容量需求。
- **MoE 效率**: MoE 模型在相同激活参数下，性能匹配或超过 dense 模型，同时自然实现模态专家分化。
- **RAE 优越性**: RAE 在理解和生成任务上均达到最优，无需为不同任务切换编码器。

## Relations (结构化)
<!-- Agent 用于构建关系网络。type: extends | alternative_to | combines_with | motivated_by | enables | conflicts_with -->
<!-- 格式: `type` → [[论文文件名]]: 说明 -->
- `alternative_to` → [[2025-MMaDA]]: 统一模型路线之争——从零训练 AR+连续扩散混合 vs LLaDA 初始化纯离散扩散；modality-specific FFN + MoE vs 模态无关全共享
- `alternative_to` → [[2025-Lumina-DiMOO]]: 同为大规模统一模型——从零训练+连续扩散+MoE vs LLaDA 初始化+离散扩散+~110M 数据；"视觉数据饥渴"发现与 DiMOO 的大规模数据工程策略呼应
- `alternative_to` → [[2025-LaViDa-O]]: 容量分配哲学对立——MoE per-modality shared experts 自动分化 vs Elastic-MoT 手动设计非对称分支（8B 理解+2.4B 生成）
- `alternative_to` → [[2025-Muddit]]: 初始化路线对立——从零训练（第四条路线）vs Vision-first（从 T2I 模型出发）
- `combines_with` → [[2025-LLaDA-V]]: 共享 SigLIP2 视觉编码器技术；Beyond-LM 用 RAE (SigLIP 2) 统一理解+生成，LLaDA-V 用 SigLIP2 纯理解
- `combines_with` → [[2026-LaViDa-R1]]: 预训练发现（视觉:语言 51:1，MoE 缩小扩展律差距）可指导 LaViDa-R1 的多任务 RL 数据配比和架构改进
- `conflicts_with` → [[2025-MMaDA]] / [[2025-Lumina-DiMOO]] / [[2026-LaViDa-R1]]: 根本性路线冲突——连续扩散 (flow matching) vs 离散扩散 (masked diffusion)；代表 Diffusion 原生路线中的"连续 vs 离散"分歧
- `conflicts_with` → [[2025-DiffusionVL]]: 路线对立——Beyond-LM 从零训练证明扩散训练本身的价值 vs DiffusionVL 用 AR 初始化"免费"继承知识再做扩散微调；Beyond-LM 提供了 DiffusionVL 无法提供的"公平对比基准"

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- **系统性研究从零训练统一多模态模型的设计空间**: 首次在万亿 token 规模系统性研究视觉-语言混合预训练的关键设计因素，填补了 KB 中"从零训练"的空白（现有工作均基于 LLaDA 初始化）
- **量化视觉与语言的数据饥渴度差异**: 发现在万亿参数规模下视觉数据需求是语言的 51 倍，为 [[problem-tree#PT-2]] 提供首个大规模实证依据
- **验证 RAE 统一视觉表示的可行性**: 证明单一视觉编码器（SigLIP 2 作为 RAE）可同时服务理解和生成，为 [[problem-tree#Tok-2a]] 提供第三条路径（SigLIP+VQ 双路、纯 VQ、RAE 统一）
- **MoE 缩小模态间扩展律差异**: MoE 架构（G=16, per-modality shared experts）将视觉-语言扩展律指数差距从 0.10（dense）降至 0.05
- **多模态预训练自然涌现世界建模能力**: 通用预训练即可产生世界建模能力，领域数据需求极少（~1% 即饱和）

### 未解决的问题
- **问题**: RAE 统一表示在极高分辨率生成（如 2048²）时的信息瓶颈
  - **为什么难**: SigLIP 2 的语义表示可能无法保留足够的像素级细节；flow matching 需要更多采样步数，推理成本高
  - **潜在思路**: 多尺度 RAE（低分辨率语义+高分辨率细节）；与 VQ tokenizer 混合（RAE 做语义引导，VQ 做细节生成）

- **问题**: Flow matching 与 masked diffusion 在统一模型中的性能对比
  - **为��么难**: 本文用 flow matching（连续扩散），KB 中其他统一模型均用 masked diffusion（离散扩散），缺乏同等条件对比
  - **潜在思路**: 在相同架构和数据下对比两种扩散范式的生成质量、训练效率、推理速度

- **问题**: Modality-specific FFN 与 MoE 的最优组合方式
  - **为什么难**: 本文同时使用两者但未消融独立贡献；LaViDa-O 的 Elastic-MoT 是任务级路由，本文是 token 级路由，粒度不同
  - **潜在思路**: 消融实验——仅 modality-specific FFN vs 仅 MoE vs 两者组合；与 LaViDa-O 的任务级路由做对比

- **问题**: 51:1 视觉-语言数据比在不同规模下的泛化性
  - **为什么难**: 51:1 是在万亿 token 规模观察到的，小规模（百亿-千亿 token）下最优比例可能不同
  - **潜在思路**: 多规模 scaling study（10B/100B/1T tokens），绘制数据比例-性能 Pareto 前沿

- **问题**: Hybrid attention masking 的最优设计
  - **为什么难**: 本文用"帧内双向+跨序列因果"，但未系统探索其他混合模式；与 LLaDA-V 的全双向、MMaDA 的全双向缺乏对比
  - **潜在思路**: Attention pattern 搜索空间探索；与 KB 中其他 attention 策略做消融对比

### 对问题树的推进
<!-- 这篇论文推进了哪些问题树节点？打开了什么新问题？使用 [[problem-tree#节点标题]] 链接 -->
- **推进了 [[problem-tree#PT-1a]] 🔴→🟡**: 验证了"先冻结后解冻"策略在从零训练场景同样有效（与 LLaDA-V 在 LLaDA 初始化场景的发现一致）
- **推进了 [[problem-tree#PT-1c]] 🔴→🟡**: 验证 RAE + 简单连接器（类似 MLP）即可达 competitive 性能，进一步支持"dLLM 不需要复杂连接器"的结论
- **推进了 [[problem-tree#PT-2]] 🔴→🟡**: 首次为多模态数据配比提供大规模实证——视觉:语言 = 51:1（万亿规模），但仅一个数据点
- **推进了 [[problem-tree#PT-3b]] 🔴→🟡**: 提供了视觉-语言扩展律的首个系统性研究——dense 模型指数差距 0.10，MoE 缩小至 0.05
- **推进了 [[problem-tree#Tok-2a]] 🟡→🟢（部分）**: RAE 统一表示证明"理解和生成可共享同一视觉编码器"，提供第三条路径
- **推进了 [[problem-tree#Uni-2a]] 🟡（强化）**: MoE with per-modality shared experts 自然涌现模态专家分化，证明"共享参数下能力不冲突"
- **推进了 [[problem-tree#Uni-2b]] 🟡（新维度）**: 提供了 token 级 MoE routing（G=16）的实证，与 LaViDa-O 的任务级 Elastic-MoT 形成对比
- **推进了 [[problem-tree#Diff-1a]] 🔴→🟡**: 首次在统一模型中使用 flow matching（连续扩散），与 KB 中其他工作的 masked diffusion（离散扩散）形成对比
- **新增问题 [PT-6]**: 从零训练 vs 基于预训练 LLM 初始化的 tradeoff——哪种策略在什么条件下更优？
- **新增问题 [PT-7]**: 世界建模能力的涌现机制——为什么通用预训练即可产生世界建模能力？领域数据饱和点（~1%）的理论解释是什么？
- **新增问题 [Uni-6]**: RAE 统一表示 vs 双路方案（SigLIP+VQ）vs 纯 VQ 的系统对比——三种方案在理解、生成、训练效率、推理速度上的 Pareto 前沿是什么？

## 个人深度评注
<!-- 你自己的判断、直觉、和延伸思考。这是知识库中最有价值的信息。 -->
- (待用户审阅后补充)
