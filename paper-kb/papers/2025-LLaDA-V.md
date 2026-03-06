---
title: "LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning"
authors: [Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, Chongxuan Li]
date: 2025-05
venue: arxiv
url: "https://arxiv.org/abs/2505.16933"
tags: [diffusion, architecture, pretraining, understanding]
category: diffusion-foundation/dllm-understanding
level: 2
status: read
importance: medium
problem_tree_nodes: [Diff-1b, Uni-1a, PT-1c, PT-1a, PT-3b]
aliases: [LLaDA-V]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结
提出 LLaDA-V，首个基于纯扩散语言模型（LLaDA-8B-Instruct）的多模态理解模型，采用 SigLIP2 视觉编码器 + MLP 连接器 + 三阶段视觉指令微调，在 18 个基准中 11 个超越同训练管线的 LLaMA3-V，验证了离散扩散 LLM 作为多模态理解骨干的可行性和数据扩展优势。

## 核心 Insight
离散扩散语言模型（dLLM）的双向注意力机制天然适配多模态理解任务——相比 AR 模型的 causal attention，bidirectional attention 能捕获完整上下文关系，使 dLLM 在知识推理（MMMU +3.2）和数学推理（MMMU-Pro +6.9）等需要全局信息的任务上具有系统性优势。但在需要逐步细粒度解析的任务（图表理解、文档理解）上不如 AR 模型。

## 与已有工作的关系
- **继承自**: [[LLaDA]]（LLaDA-8B-Instruct 预训练权重，masked diffusion 训练范式），[[LLaVA]]（LLaVA-style 架构模板和训练管线，从 AR LLM 迁移到 dLLM），[[SigLIP2]]（SigLIP2-so400m-patch14-384 视觉编码器，KB 中唯一使用 SigLIP2 的工作）
- **对比**: [[2025-MMaDA]]（同基于 LLaDA，但 LLaDA-V 纯理解 vs MMaDA 统一模型，MMMU 48.6 >> 30.2），[[2025-Lumina-DiMOO]]（同 LLaDA 初始化，DiMOO 纯 VQ 统一模型 MMMU 58.6% 但用 ~110M 数据 vs ~16M），[[2025-LaViDa-O]]（同用 SigLIP，MMMU 48.6 > 45.1 但 LaViDa-O 是 10.4B 统一模型），[[LLaMA3-V]]（控制变量直接对比，dLLM vs AR，11/18 胜出），[[Qwen2-VL]]（SOTA AR 基线，LLaDA-V 仍有差距），[[2025-Muddit]]（路线完全相反，Vision-first vs LLM-first）
- **互补**: [[2026-LaViDa-R1]]（RL 后训练框架可迁移到 LLaDA-V 解决偏好对齐缺失），[[MAmmoTH-VL]]（核心训练数据来源），[[VisualWebInstruct]]（Stage 3 推理增强数据）

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 整体架构
- **语言骨干**: LLaDA-8B-Instruct（离散 masked diffusion LLM，双向 attention）
- **视觉编码器**: SigLIP2-so400m-patch14-384（冻结 → Stage 2 起解冻，lr=2×10⁻⁶）
- **连接器**: 两层 MLP（随机初始化）
- **注意力**: 全双向（no mask），经消融验证优于 dialogue causal mask

### 三阶段训练
| 阶段 | 数据 | 训练参数 | 学习率 | 批次 | 长度 |
|------|------|----------|--------|------|------|
| Stage 1: 图文对齐 | LLaVA-Pretrain (558K) | 仅 MLP 连接器 | 1×10⁻³ | 64 | — |
| Stage 2a: 单图指令微调 | MAmmoTH-VL-SI (10M) | 全模型 | vision 2×10⁻⁶, LM+proj 1×10⁻⁵ | 256 | 8192 |
| Stage 2b: 多模态指令微调 | MAmmoTH-VL diverse (2M) | 全模型 | 同上 | 256 | 16384 |
| Stage 3a: 推理增强 | VisualWebInstruct (900K) | 全模型 | 同上 | 256 | 8192 |
| Stage 3b: 均衡推理训练 | MAmmoTH-VL + VisualWebInstruct (3M) | 全模型 | 同上 | 256 | 16384 |

### 训练目标
扩展 masked diffusion 到多轮多模态对话：仅对 response token 做随机 mask，图像特征和 prompt 保持 clean（不 mask）。模型学习在给定视觉上下文和对话历史条件下恢复被 mask 的 response token。

### 推理流程
- 初始化为全 [MASK] 的 response 序列
- 迭代去噪：时间步从 t=1 降至 t=0
- Low-confidence remasking：每步保留高置信度预测��re-mask 低置信度 token（比例 s/t）
- 继承 LLaDA 的采样策略

## Building Blocks（可复用组件）

### Block 1: dLLM 作为多模态理解骨干（LLaDA-as-Backbone）
- **做法**: 用 LLaDA-8B-Instruct（masked diffusion LLM）替换标准 AR LLM（如 LLaMA3-8B）作为多模态理解的语言骨干，搭配 SigLIP2 视觉编码器和 MLP 连接器，完全沿用 LLaVA-style 训练管线
- **机制 (WHY it works)**: dLLM 的 bidirectional attention 允许每个 token 访问完整上下文（包括前后文），对需要全局推理的多模态理解任务有结构性优势。与 BERT 的 MLM 类似，双向注意力在理解类任务上天然优于单向 causal attention。此外，dLLM 的 mask-predict 训练目标迫使模型学习更 robust 的表征（需要从部分信息重建完整语义）
- **适用条件**: 需要有高质量的 dLLM 预训练权重（如 LLaDA-8B）；多模态理解场景；可直接复用 LLaVA-style 训练管线和数据
- **什么时候会 break**: (1) 图表/文档理解等需要逐行逐区域解析的任务——这类任务的顺序处理与 AR 的 causal 生成更匹配（AI2D 77.8 vs LLaMA3-V 81.1）；(2) dLLM 推理效率不如 AR+KV-cache——多步迭代去噪 vs 单次前向传播；(3) LLaDA 缺乏 PPO/DPO 等偏好对齐，限制了指令遵循能力上界
- **可组合方向**: 与 RL 后训练（UniGRPO/LaViDa-R1 框架）结合提升推理能力；与统一生成能力结合构建 unified model（如 MMaDA、LaViDa-O 路线）

### Block 2: Bidirectional Attention for Multimodal Understanding（双向注意力选择）
- **做法**: 在多轮对话中使用全双向注意力（no mask）而非 dialogue causal mask。消融实验对比两种方案
- **机制 (WHY it works)**: dLLM 的训练目标本身是双向的（mask-predict），施加 causal mask 等于人为限制了模型的信息访问范围，与预训练目标不一致。Bidirectional attention 在 12 个基准中 7 个胜出（MMMU +1.78, MuirBench +5.19），尤其在需要综合多个对话轮次信息的任务上优势明显
- **适用条件**: 纯理解任务（不需要自回归生成）；多轮对话场景
- **什么时候会 break**: (1) 需要自回归文本生成时，bidirectional attention 无法直接生成（需要迭代去噪）；(2) 在部分基准上 causal 更好（MME、SeedBench、ChartQA），暗示某些任务受益于有序处理
- **可组合方向**: 混合 attention 策略——理解阶段 bidirectional + 生成阶段 causal（类似 Transfusion 的思路）

### Block 3: 多阶段推理增强训练（Reasoning Enhancement Pipeline）
- **做法**: 在标准视觉指令微调后增加推理增强阶段——Stage 3a 用 VisualWebInstruct (900K) 训练推理链，Stage 3b 混合指令数据 + 推理数据 (3M) 均衡训练，用 `/think` 和 `/no_think` 标签控制是否激活推理模式
- **机制 (WHY it works)**: 分阶段引入推理能力——先用纯推理数据激活 CoT 推理模式，再混合训练防止灾难遗忘。`/think`-`/no_think` 标签让模型学会在需要推理的问题上展开推理链、在简单问题上直接回答，避免不必要的计算开销
- **适用条件**: 有高质量推理数据（如 VisualWebInstruct）；需要在推理能力和通用能力间平衡
- **什么时候会 break**: (1) 推理数据质量不高时引入噪声推理链；(2) 混合比例不当导致推理能力或通用能力退化；(3) dLLM 的并行去噪对 CoT 的因果有效性存疑（与 MMaDA 面临相同问题）
- **可组合方向**: 与 RL 后训练结合——推理增强 SFT 作为 RL 冷启动（参考 P-RL-03）；与 answer-forcing（LaViDa-R1）结合提升困难推理问题的训练效果

## Anti-patterns / 已知失败模式
- **dLLM 在图表/文档理解上弱于 AR**: AI2D 77.8 vs 81.1, DocVQA 83.9 vs 86.2——这类任务需要顺序解析结构化信息，与 AR 的 causal 生成更匹配。根源在于 dLLM 并行去噪缺乏序列/结构推理的归纳偏置，与 MMaDA GenEval Position 0.20 的空间推理弱点属同一机制缺陷
- **dLLM 缺乏偏好对齐**: LLaDA 未做 PPO/DPO/SimPO 等对齐，限制了指令遵循能力上界，是与 Qwen2-VL 差距的重要原因之一
- **图像处理策略简陋**: 用 split+resize 处理高分辨率图像（非动态分辨率），降低效率和精度。dLLM 的并行 attention 下 split 打破空间连续性可能比 AR 更有害
- **[Critic] 直接移植 AR 训练管线未利用 dLLM 独有特性**: 完全沿用 LLaVA 管线，未利用 dLLM 的 inpainting、并行去噪、中间状态可获取性等独有优势（对比 LaViDa-R1 的 answer-forcing、DiMOO 的 ML-Cache）
- **[Critic] `/think`-`/no_think` 的因果有效性存疑**: dLLM 并行预测所有 masked token，reasoning→result 的因果关系不严格成立。缺乏 faithfulness 实验验证 CoT 效果是因果推理还是模式匹配
- **[Critic] "数据扩展性优越"的过早泛化**: 仅基于 MMMU-Pro 一个基准的 scaling curve，不足以得出一般性结论。在 dLLM 弱势任务（图表/文档）上 scaling 行为可能相反

## 实验关键发现
- **11/18 基准超越 LLaMA3-V**: 在完全控制变量（同视觉编码器、同连接器、同训练数据、同学习率）下，dLLM 骨干 > AR 骨干
- **MMMU 48.6 vs 45.4 (+3.2)**: 多学科知识推理，dLLM 双向注意力的优势最显著
- **MMMU-Pro 35.2 vs 28.3 (+6.9)**: 困难推理任务上差距更大
- **数据扩展性优越**: MMMU-Pro 上 1M 数据的 LLaDA-V > 9M 数据的 LLaMA3-V，暗示 dLLM 骨干有更高的数据效率
- **Bidirectional > Causal**: 消融显示全双向注意力在 7/12 基准上胜出（MMMU +1.78, MuirBench +5.19）
- **多图/视频理解有优势**: MuirBench 48.3 vs 47.4, MLVU 59.5 vs 57.5, VideoMME 56.1 vs 55.8
- **在 diffusion 类模型中 SOTA**: 超越 MetaMorph、Show-o、D-DiT、JanusFlow 等混合/纯扩散模型
- **与 Qwen2-VL 仍有差距**: 主要归因于 LLaDA 语言骨干弱于 Qwen2-7B（缺乏偏好对齐 + 预训练规模差距）

## Relations (结构化)
<!-- type: extends | alternative_to | combines_with | motivated_by | enables | conflicts_with -->
- `extends` → [[LLaDA]]: 基于 LLaDA-8B-Instruct 预训练权重，首次将 dLLM 扩展为多模态视觉理解模型
- `extends` → [[LLaVA]]: 沿用 LLaVA-style 架构和训练管线，将该范式从 AR LLM 迁移到 dLLM
- `alternative_to` → [[2025-MMaDA]]: 同基于 LLaDA 但路线分歧——LLaDA-V 纯理解 (SigLIP2+MLP) vs MMaDA 统一模型 (MAGVIT-v2+UniGRPO)；MMMU 48.6 >> 30.2 验证专注理解在理解维度远优于统一模型（同数据量级）
- `alternative_to` → [[2025-Lumina-DiMOO]]: LLaDA-V 用 SigLIP2 做纯理解 vs DiMOO 纯 VQ token 做统一模型；DiMOO MMMU 58.6% > 48.6% 但用了 ~110M 数据 vs ~16M
- `alternative_to` → [[LLaMA3-V]]: 严格控制变量的直接替代——dLLM 骨干 vs AR 骨干，11/18 基准胜出，是"dLLM 理解优于 AR 理解"的最强证据
- `motivated_by` → [[LLaDA]]: dLLM bidirectional attention 天然适配理解任务的假设来自 LLaDA 架构设计
- `enables` → [[2026-LaViDa-R1]]: LLaDA-V 验证的"bidirectional attention 理解优势"是 LaViDa-R1 将 RL 应用于 dLLM 理解任务的前提假设
- `combines_with` → [[SigLIP2]]: 使用 SigLIP2-so400m-patch14-384 视觉编码器，KB 中唯一使用 SigLIP2 的工作
- `combines_with` → [[MAmmoTH-VL]]: 核心训练数据来源 (Stage 2a 10M + Stage 2b 2M)
- `combines_with` → [[VisualWebInstruct]]: Stage 3 推理增强数据来源 (900K)
- `enables` → [[2025-ReDiff]]: LLaDA-V 验证的 dLLM 理解优势为 ReDiff 的主动精炼框架提供了理解端的质量保证前提
- `enables` → [[2025-SDAR-VL]]: LLaDA-V 的 dLLM 多模态理解验证和 SigLIP2 视觉编码器选型为 SDAR-VL 的块状扩散 VLM 提供了参考

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- 首次在控制变量条件下验证 dLLM 骨干 vs AR 骨干在多模态理解上的系统性差异（同 SigLIP2 + MLP + 训练数据），11/18 基准 dLLM 胜出。这是 KB 中**唯一一篇做到真正公平对比的论文**（MMaDA/DiMOO/LaViDa-O 均无同等条件的 AR 基线），为 [[problem-tree#Diff-1b]] 提供迄今最干净的"离散扩散 vs AR"理解端证据
- 刻画了 dLLM 理解任务的 fine-grained 优劣势分布：知识/数学推理 (MMMU +3.2, MMMU-Pro +6.9) 系统性优于 AR；图表/文档理解 (AI2D -3.3, DocVQA -2.3) 系统性弱于 AR
- 验证了 dLLM 骨干在多模态理解上可能具有更高的数据效率（MMMU-Pro: 1M > 9M）
- 验证了 LLaVA-style 训练管线可直接迁移到 dLLM，降低了 dLLM 多模态化的工程门槛

### 未解决的问题
- 问题: dLLM 在图表/文档理解上的系统性劣势
  - 为什么难: 图表/文档需要逐行逐区域顺序解析，dLLM 并行去噪缺乏序列/结构推理的归纳偏置，这是架构级缺陷。与 MMaDA GenEval Position 0.20 的空间推理弱点属同一根源
  - 潜在思路: block-wise 半自回归推理（按逻辑区块顺序去噪）；layout-aware masking schedule；混合 attention（底层双向+顶层因果）
- 问题: dLLM 缺乏偏好对齐 (no PPO/DPO)
  - 为什么难: dLLM 的 DPO log-ratio 计算需要额外 likelihood estimation (P-RL-02)；KL 正则化有效性存争议 (P-RL-04)；KB 中 LaViDa-R1 方案仅在统一模型验证，纯理解场景未测试
  - 潜在思路: 迁移 LaViDa-R1 统一 PG 框架到纯理解场景（纯文本 response 无高熵 image token 问题，KL 可能稳定）；探索 SimPO 等不依赖 reference model 的方法
- 问题: CoT 在 dLLM 并行去噪框架中的因果有效性存疑
  - 为什么难: dLLM 并行预测所有 masked token，reasoning→result 因果关系不严格成立。LLaDA-V、MMaDA、LaViDa-R1 共同面临此问题，均无 faithfulness 实验验证
  - 潜在思路: faithfulness 消融（扰乱推理链看 result 是否变化）；对比 block-wise 生成 vs 全并行生成；分析注意力图中推理 token 对 result token 的影响
- 问题: 数据扩展性优越的泛化性未验证
  - 为什么难: 仅 MMMU-Pro 一个基准的单条 scaling curve，可能是任务特定而非 dLLM 固有属性
  - 潜在思路: 在多个任务类型上建立独立的 dLLM vs AR scaling curve，验证数据效率优势的普遍性

### 对问题树的推进
- 推进了 [[problem-tree#Diff-1b]]: 提供迄今最干净的控制变量对比——理解端 dLLM 在知识/数学推理上系统性优于 AR，在图表/文档理解上系统性弱于 AR。量化了优劣势边界，不仅验证可行性，更刻画了适用范围
- 推进了 [[problem-tree#Uni-1a]]: 为统一模型理解端提供间接但重要的证据——dLLM 骨干做纯理解已优于 AR 骨干，统一模型中 dLLM 的理解能力不应是瓶颈
- 推进了 [[problem-tree#PT-1c]]: 验证了 dLLM 骨干 + 简单两层 MLP 连接器即可达 competitive 性能
- 推进了 [[problem-tree#PT-1a]]: 验证了"先冻结后解冻"视觉编码器策略在 dLLM 骨干上同样有效
- 推进了 [[problem-tree#PT-3b]] (新增证据): MMMU-Pro 1M dLLM > 9M AR——dLLM 数据效率在推理密集型任务上可能显著高于 AR，是 dLLM Scaling Law 的首个间接证据
- 暴露了 [[problem-tree#RL-2]] / [[problem-tree#Post-3]]: dLLM 纯理解模型的偏好对齐是尚未被任何论文充分解决的 open problem
- 新增问题: [PT-4] dLLM 骨干的任务偏好分布——哪些理解子任务天然适合 dLLM、哪些不适合？需要更系统的 benchmark 分析
- 新增问题: [PT-5] dLLM 多模态理解的数据效率机制——双向注意力为何在少数据条件下更高效？

## 个人深度评注
<!-- 留待用户审阅后补充 -->

### [Critic] 增量贡献评估
| Block | 新颖性 | 理由 |
|-------|--------|------|
| Block 1: dLLM-as-Backbone | 低中 | 本质是 LLaVA 管线的"骨干替换实验"。KB 中 MMaDA (2025-05 同期) 已在统一模型场景验证 LLaDA 多模态可行性，LLaDA-V 收窄到纯理解场景。核心价值是控制变量实验设计而非方法创新 |
| Block 2: Bidirectional Attention | 低 | dLLM 使用全双向是预训练目标的自然推论。消融验证"不限制优于限制"是预期结果。KB 中 LaViDa-R1、MMaDA 已隐含此结论。贡献在于逐基准消融数据 |
| Block 3: 推理增强训练 | 低中 | 分阶段引入推理数据 + `/think`-`/no_think` 是 DeepSeek-R1 在 AR 上已广泛使用的做法。在 dLLM 上首次应用，但未分析有效性机制 |

**总体判断**: 核心价值是**经验验证**而非方法创新——通过严格控制变量实验证明 dLLM 骨干在多模态理解上可以超越 AR 骨干。三个 Block 均为已有技术的组合/迁移。

### [Critic] 关键隐含假设
1. **LLaVA 管线对 dLLM 同样最优**: 为 AR 设计的三阶段训练流程可能非 dLLM 最优方案——dLLM 的表征空间结构与 AR LLM 不同，最优对齐策略未必相同。未做管线设计消融
2. **dLLM vs AR 性能差异可归因于 attention 机制**: LLaDA-8B 和 LLaMA3-8B 的差异远不止 attention mask——预训练数据/质量、tokenizer、训练 loss、偏好对齐等都不同。骨干本身的质量差异无法完全控制
3. **数据扩展性是 dLLM 固有属性**: 仅一个基准一条 curve，可能是任务特定的。统计效力不足以得出一般性结论
4. **`/think` 标签在 dLLM 中触发因果推理而非模式匹配**: 缺乏 faithfulness 实验。迭代去噪的隐式序列性（high-confidence 先确定）可能提供了弱因果链，但论文未分析置信度分布

### [Critic] 机制层深度分析

**双向注意力优势的三个层次**:
1. 表层: 视觉 token（序列前部）和问题 token（后部）可在同一前向传播中互相参照（causal 下视觉 token 无法"看到"后面的问题）
2. 深层: mask-predict 训练目标迫使模型从**部分信息重建完整语义**——这与理解任务需求（从不完整线索推断答案）高度匹配。AR 的 next-token prediction 优化**序列延续**能力而非信息**整合**能力
3. [推测] 视觉 token 全局一致性: bidirectional attention 让所有视觉 token 对等参与全局表征构建，causal attention 下第一个视觉 token 只能看到自己

**图表/文档理解弱势的深层原因**:
不仅是"逐行解析"，而是**结构化信息的序列依赖性**——表格行列标题与数据单元有固定拓扑结构，AR 的 causal 生成自然编码了"先标题后数据"的层次。dLLM 并行预测将所有位置视为等权，丢失结构先验。与 MMaDA GenEval Position 0.20 属同一机制缺陷——dLLM 缺乏空间/结构推理归纳偏置。

**LLaDA-V (纯理解) vs 统一模型在理解上的 tradeoff**:
| 模型 | MMMU | 类型 | 训练数据 |
|------|------|------|----------|
| LLaDA-V | 48.6 | 纯理解, 8B | ~16M |
| MMaDA | 30.2 | 统一, 8B | ~数 M |
| LaViDa-O | 45.1 | 统一, 10.4B | 未公开 |
| DiMOO | 58.6 | 统一, 8B | ~110M |

LLaDA-V 在 ~16M 数据下 MMMU 48.6 优于同数据量级统一模型（MMaDA 30.2, LaViDa-O 45.1），但弱于大数据量 DiMOO (58.6)。暗示: (a) 中等数据下专注理解比统一训练更高效；(b) 数据量足够大时统一训练的跨模态协同可反超

### [Connector] 技术谱系定位
```
LLaDA (masked diffusion LM, 基础, 2024)
  │
  ├── [路线 D: 纯理解]
  │   └── LLaDA-V (2025-05) ← 本文
  │       SigLIP2+MLP, LLaVA-style, 纯理解
  │
  ├── [路线 A: 统一, 模态无关全共享]
  │   ├── MMaDA (2025-05, NeurIPS) — 方法论开创
  │   └── DiMOO (2025-10) — 数据工程 scaling
  │
  └── [路线 B: 统一, 非对称架构]
      └── LaViDa → LaViDa-O (2025-09) → LaViDa-R1 (2026-02)
```
LLaDA-V 是 dLLM 多模态理解能力的"纯净验证"——唯一一个刻意放弃生成、专注测量 dLLM 理解上界的工作。

### [Ideator] 潜在研究方向
1. **dLLM 纯理解模型的偏好对齐 (LLaDA-V + RL)**: 迁移 LaViDa-R1 统一 PG 框架到 LLaDA-V。纯文本 response 无高熵 image token，KL 应稳定（P-RL-04 中的问题不存在）。用 `/think` 推理增强作 SFT 冷启动 (P-RL-03)，加 answer-forcing 解决困难推理。**风险低、可行性高**。预期 MMMU/MathVista 提升 2-5 分
2. **dLLM 理解优势的任务级机制解析**: probing study 对比 LLaDA-V 和 LLaMA3-V 不同层的表征质量→设计"底层双向+顶层因果"混合 attention→layout-aware masking schedule。**风险中等、可行性中等**
3. **dLLM 数据效率的 Scaling Law 建立**: 固定架构，在 {0.5M-20M} 数据规模下训练 dLLM vs AR，在多个任务上绘制 scaling curve，验证数据效率优势的普遍性。**风险低、可行性中等偏高**。若证明 dLLM 在 5-10M 系统性匹配 AR 50M+，将有重大影响

### [Ideator] Pattern 候选
- **候选 P-Diff-02: dLLM 双向注意力在全局推理任务上系统性优于 AR，在顺序解析任务上系统性弱于 AR**
  - 支撑: [[2025-LLaDA-V]]（MMMU +3.2, MMMU-Pro +6.9; AI2D -3.3, DocVQA -2.3）、[[2026-LaViDa-R1]]（Lisa-Grounding 超越 AR specialist）
  - 需第三篇独立控制变量验证
- **候选 P-Diff-03: dLLM 骨干在多模态理解上具有数据效率优势**（证据不足，暂记为初步观察，待 Scaling Study 验证）

### [Ideator] 对已有 Pattern 的影响
- **P-Diff-01**: 强化——从"统一模型"扩展到"纯理解模型"，增加 LLaDA-V 控制变量对比证据。建议补充"图表/文档理解系统性弱于 AR"的例外
- **P-RL-03**: 间接强化——LLaDA-V Stage 3 推理增强是 SFT 冷启动的变体，MMMU-Pro 提升主要来自 SFT 而非 RL（与 MMaDA 发现一致）
- **P-RL-04**: 提供新视角——纯理解场景（无 image token）是 KL 稳定性的天然对照组。若 LLaDA-V + RL 时 KL 稳定，可细化 P-RL-04 为"仅在含 image token 的 dLLM RL 中 KL 有争议"
- **P-Uni-02**: 需修订——LLaDA-V 数据效率发现暗示 dLLM 骨干所需"足够数据"门槛可能低于 AR 骨干
