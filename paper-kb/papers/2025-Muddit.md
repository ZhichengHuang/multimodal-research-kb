---
title: "Muddit: Unified Discrete Diffusion for Multimodal Generation"
authors: [Qingyu Shi, Jinbin Bai, Zhuoran Zhao, Wenhao Chai, Kaidong Yu, Jianzong Wu, Shuangyong Song, Yunhai Tong, Xiangtai Li, Xuelong Li, Shuicheng Yan]
date: 2025-05
venue: arxiv
url: "https://arxiv.org/abs/2505.23606"
tags: [diffusion, unified-model, architecture, generation, understanding]
category: unified-model/diffusion-native
level: 2
status: read
importance: medium
problem_tree_nodes: [Uni-1, Uni-2a, Diff-1b, Diff-2b]
aliases: [Muddit, Meissonic-Unified]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结
提出 Muddit，一个基于预训练 Meissonic（MM-DiT text-to-image 模型）初始化的 1B 统一离散扩散模型，通过添加轻量文本解码头和统一 masked token prediction 训练，在仅 3.5M 图文对训练下实现文本生成和图像生成的双向统一，推理速度达到同类模型的 4-11 倍加速。

## 核心 Insight
与主流"从语言模型出发扩展到视觉"的路线（LLaDA→MMaDA/DiMOO/LaViDa）相反，Muddit 提出**从视觉先验出发扩展到文本**——利用预训练 T2I 模型（Meissonic）的强图像生成能力作为骨干，通过轻量适配获得文本理解/生成能力。这验证了一条"视觉优先"的统一模型构建路线：强视觉先验 + 轻量文本适配，以极小参数量（1B）和极少数据（3.5M）实现 competitive 的多模态统一。

## 与已有工作的关系
- **继承自**: [[Meissonic]]（预训练 T2I 模型，提供 MM-DiT 骨干权重和强图像生成先验）、[[FLUX]]（MM-DiT dual-/single-stream 架构设计的来源）
- **对比**: [[2025-MMaDA]]（同为 dLLM 统一模型，但初始化路径相反：MMaDA 从 LLaDA-8B 语言模型出发，Muddit 从 Meissonic T2I 模型出发；8B vs 1B）、[[2025-Lumina-DiMOO]]（同为 dLLM 统一模型但路线对立：DiMOO 从 LLM 出发用 8B+110M 数据达 GenEval 88%，Muddit 从 T2I 出发用 1B+3.5M 达 61%）、[[Show-o]]（同参数量级 ~1B 统一模型，AR+离散扩散混合 vs Muddit 纯离散扩散，GenEval 0.68 vs 0.61 但 Muddit 5.6x 更快）、[[UniDisc]]（1.4B 统一离散扩散，GenEval 0.42，Muddit 显著超越）
- **互补**: [[2026-LaViDa-R1]]（RL 后训练方法论可迁移到 Muddit 骨干上）、[[2025-MMaDA]]（UniGRPO 可作为 Muddit 后训练增强路径）

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 整体架构
- **骨干**: MM-DiT（Multimodal Diffusion Transformer），初始化自预训练 Meissonic 权重，遵循 FLUX 的 dual-/single-stream 设计
- **图像 Tokenizer**: 预训练 VQ-VAE（冻结），将 512×512 图像编码为离散 token
- **文本编码器**: CLIP tokenizer + encoder（冻结），最大 77 token
- **文本解码器**: 轻量线性头（linear head），将 MM-DiT 预测转换回文本 token 空间
- **参数量**: ~1B（远小于 LLaDA-based 的 8-10B 方案）

### 统一离散扩散目标
- 连续时间马尔科夫链：dpt/dt = Qt·pt
- 前向过程：q(x_t|x) = Cat(x_t | α_t·x + (1-α_t)·m)，其中 m 为 mask token
- 统一训练 loss：L_unified = E[∫₀¹ (α'_t)/(1-α_t) log(G(x_t,α_t,c)·x) dt]
- Cosine masking schedule：γ_t = (2/π)(1-(1-t)²)^(-1/2)
- 关键：文本→图像和图像→文本方向使用完全相同的 loss

### 两阶段训练
| 阶段 | 步数 | 数据 | 内容 |
|------|------|------|------|
| Pretraining | 70K steps | 2M 图文对（Qwen2.5-VL-3B re-caption） | 50% T2I + 50% I2T balanced sampling |
| Fine-tuning | — | 额外 1.5M (LLaVA-Instruct-150K, MG-LLaVA, 500K 精选) | VQA + 多任务 |

- Batch size: 1024（effective via gradient accumulation）
- Learning rate: 1×10⁻⁴
- Weight decay: 1×10⁻²
- 图像分辨率: 512×512

### 推理采样
- 时间反演后验采样：未 mask token 保持不变，mask token 按 α_s/α_t 加权的凸组合解码
- 默认 32 步（论文测试 8-64 步）
- CFG scale: 9.0（T2I 和 I2T 通用）

### 支持任务
1. **T2I (Text→Image)**: 文本编码器产生嵌入 → MM-DiT 迭代去噪 masked image tokens
2. **I2T (Image→Text)**: 图像编码器产生 VQ tokens → MM-DiT 预测文本 tokens
3. **VQA**: 拼接图像和问题嵌入作为双条件输入
4. **Image-guided text completion**: 扩展的 I2T 变体

## Building Blocks（可复用组件）

### Block 1: 视觉先验骨干策略（Visual-Prior Backbone）
- **做法**: 使用预训练 T2I 模型（Meissonic, MM-DiT 架构）作为统一模型的骨干，保留其强图像生成能力，仅添加轻量线性头适配文本生成。与主流的"LLM 骨干 + 视觉适配器"路线相反
- **机制 (WHY it works)**: Meissonic 的 MM-DiT 已通过 T2I 预训练学习了 text-conditioned visual generation 的联合表征空间。CLIP conditioning 迫使 MM-DiT 在每一层维护 text-image 对应关系，这些对应关系虽是为 T2I 训练的，但 cross-attention 权重隐式编码了双向映射。从视觉先验出发使图像生成的"冷启动代价"为零，统一训练阶段仅需学习文本解码
- **适用条件**: 需要高质量预训练 T2I 模型作为初始化；适用于视觉生成为主、文本理解为辅的应用场景；数据/计算资源受限的场景（1B + 3.5M 即可 competitive）
- **什么时候会 break**: (1) 文本理解/推理深度受限——CLIP 77 token 限制是架构硬约束，无法做 CoT 推理；(2) T2I 模型的文本处理仅是浅层语义匹配而非深层推理，cross-attention 的"可逆性"缺乏理论保证；(3) 1B 参数中大部分已被 T2I 预训练分配给视觉生成，给文本能力的"剩余容量"有限
- **可组合方向**: 在 Muddit 基础上添加 LLM 文本分支（类似 Elastic-MoT 的反向应用：视觉骨干+文本分支）；与更强的 VQ tokenizer 组合提升生成上界

### Block 2: 轻量文本解码头（Lightweight Text Decoder）
- **做法**: 在 MM-DiT 骨干顶部添加简单线性层将 hidden states 映射到文本 token 空间（hidden_dim → vocab_size），参数量仅约数十 M
- **机制 (WHY it works)**: MM-DiT 的 hidden states 在 T2I 预训练中已被迫编码文本语义信息（CLIP embedding 通过 cross-attention 注入后必须在 hidden states 中保留以指导去噪）。如果文本信息在 hidden states 中是线性可分的（合理假设，因 CLIP embedding 是线性注入的），线性头即可将其投射到文本 token 空间
- **适用条件**: 文本输出相对简单（短描述、VQA 答案）；不需要复杂语言推理
- **什么时候会 break**: (1) 线性头只能做点估计，不能建模 token 间依赖关系（依赖 MM-DiT self-attention 处理）；(2) 需要生成长文本/复杂推理时表达能力不足；(3) CLIP tokenizer 词表非为理解任务设计，可能不适合专业领域回答
- **可组合方向**: 渐进增加文本解码器复杂度（线性→小 transformer→大 LM）

### Block 3: Balanced Joint Training（均衡联合训练）
- **做法**: 预训练阶段 50% T2I + 50% I2T 均衡采样，同时优化两个方向；文本 loss 权重最优值为 0.6（高于图像 loss 权重 0.4）
- **机制 (WHY it works)**: 均衡采样确保 MM-DiT 骨干同时接收双向梯度信号。消融实验（移除联合训练 → GenEval 61.6 → 28.3）揭示了深层机制：T2I 方向梯度优化"文本如何引导图像特征"，I2T 方向梯度优化"图像特征如何投射到文本空间"，两者在 cross-attention 层产生互补效应。text loss weight 0.6 > 0.4 的解读：图像生成能力已由 Meissonic 预训练提供，fine-tuning 主要学习任务是文本理解，因此文本 loss 权重更大以补偿能力缺口
- **适用条件**: 统一模型的预训练/微调阶段；两个方向有可比数据量
- **什么时候会 break**: (1) 移除联合训练的暴跌可能更多归因于 catastrophic forgetting（仅用 T2I 在新数据上 fine-tune 导致遗忘 Meissonic 原始生成分布）而非纯粹的"协同缺失"；(2) 50/50 样本比不等于 token 级梯度均衡——1024 image tokens vs ~77 text tokens 的不对称被忽略
- **可组合方向**: 引入 curriculum learning 动态调整方向比例；结合 RL 阶段的联合优化（参考 DiMOO Self-GRPO）

### Block 4: MM-DiT 双流/单流混合架构（Dual-/Single-Stream MM-DiT）
- **做法**: 遵循 FLUX 设计——前半部分双流（文本/图像各自 self-attention），后半部分单流（联合 attention）。从 Meissonic 预训练权重初始化
- **机制 (WHY it works)**: 双流部分避免低层 attention 被跨模态噪声污染——训练早期文本和图像 token 处于不同分布空间，强制联合可能导致 attention 退化。高层 single-stream 在模态特定特征已被提炼后做跨模态融合更有效。Meissonic 预训练权重直接提供了最优的分界层位置先验
- **适用条件**: 多模态生成任务；有可用的 MM-DiT 预训练权重
- **什么时候会 break**: (1) T2I 预训练确定的 dual/single 分界对 I2T 方向可能不是最优——I2T 可能在更早的层就需要跨模态融合；(2) DiMOO 的全共享（single-stream from layer 1）在更多数据下达到 GenEval 88%，暗示 dual-stream 的好处可被数据量替代
- **可组合方向**: 与 Elastic-MoT 的非对称分支思路结合；扩展到视频的三流设计

## Anti-patterns / 已知失败模式
- **CLIP 77 token 硬限制导致功能缺失**: 不是"性能差"而是"能力不存在"——无法做 CoT 推理、长文本生成、数学推导。KB 中 MMaDA/DiMOO/LaViDa-R1 均支持长 CoT，Muddit 在此维度完全不可比
- **VQ-VAE 冻结导致生成质量天花板**: 生成质量上界由 Meissonic 使用的 VQ-VAE 决定，且 VQ 离散化在高频细节（皮肤纹理、光影渐变）上天然有信息损失
- **CLIP text encoder 冻结的语义瓶颈**: CLIP 表征是为对比学习优化的全局语义向量，不擅长细粒度语义区分，无法通过训练改善文本理解基础
- **CFG 9.0 在 I2T 方向的效果存疑**: CFG 放大条件信号对生成的引导，但文本生成仅有线性头，CFG 的放大效应主要作用在 MM-DiT hidden states 层面。KB 中 MMaDA 仅对 T2I 使用 CFG 3.5，更为审慎
- **64 步性能下降 (61.9→61.1) 暗示 over-denoising**: 过多步数导致已正确的 token 被重新 mask 和错误重预测——与 DiMOO ML-Cache 的动机相呼应（已确定 token 不应重复计算）
- **Catastrophic forgetting 风险**: 从 Meissonic fine-tune 时 3.5M 新数据与原始 T2I 训练数据的分布差异可能导致遗忘原有生成分布

## 实验关键发现
- **T2I 生成 (GenEval 512×512)**: Overall 0.61，超越基座 Meissonic (0.54) 和 UniDisc (0.42)，接近 SD3 (0.62)，仅 1B 参数
- **I2T 理解**: CIDEr 59.9，VQAv2 68.2%（超越 D-DiT 60.1%），MME 1107.4，GQA 57.5
- **推理速度**: 平均延迟 1.49s，4.2× faster than Qwen-2.5-VL，5.6× than Show-O，8.1× than BLIP-2，10.9× than LLaVA-1.6
- **联合训练关键**: 移除联合训练 → GenEval 61.6 → 28.3（暴跌 >50%），是 KB 中最极端的协同效应证据
- **联合训练提升基座 T2I**: Meissonic 基座 GenEval 0.54 → Muddit 0.61（+7pp），说明 I2T 方向的梯度反哺了生成质量
- **文本 loss 权重最优 0.6**: 图像生成能力已有预训练先验，文本理解是主要学习任务，因此文本 loss 权重更大
- **步数消融**: 32 步性能趋于饱和（GenEval 61.9），64 步略降（61.1，可能 over-denoising），8 步差（51.6）

## Relations (结构化)
- `extends` → [[Meissonic]]: 基于 Meissonic MM-DiT 预训练权重初始化，继承 T2I 生成能力，添加轻量文本解码头实现双向统一
- `alternative_to` → [[2025-MMaDA]]: 统一 dLLM 路线之争——"视觉先验骨干 + 轻量文本适配"(1B) vs "语言先验骨干 + 视觉扩展"(8B)；Muddit 参数效率极高但文本推理深度受限
- `alternative_to` → [[2025-Lumina-DiMOO]]: 路线对立——Muddit 从 T2I 出发(1B, 3.5M data) vs DiMOO 从 LLM 出发(8B, ~110M data)；代表"视觉先验 + 极简数据" vs "语言先验 + 大规模数据"的 tradeoff 两端
- `alternative_to` → [[2025-LaViDa-O]]: 架构哲学对立——Muddit 全参数共享 MM-DiT(1B) vs LaViDa-O Elastic-MoT 非对称分支(10.4B)
- `alternative_to` → [[Show-o]]: 同参数量级(~1B)——Muddit 纯离散扩散 vs Show-o AR+离散扩散混合；Muddit 5.6x 更快但 GenEval 略低(0.61 vs 0.68)
- `motivated_by` → [[FLUX]]: MM-DiT dual-/single-stream 架构的直接来源
- `combines_with` → [[2026-LaViDa-R1]]: Muddit 的视觉先验骨干可与 LaViDa-R1 的 RL 后训练框架（answer-forcing, tree search）结合
- `conflicts_with` → [[2025-MMaDA]]: 关于最优初始化方向的根本性分歧——Muddit 认为视觉先验是更好起点(T2I→unified)，MMaDA 系列认为语言先验更好(LLM→unified)
- `enables` → [[2025-dMLLM-TTS]]: dMLLM-TTS 在 Muddit 基座上应用 test-time scaling，GenEval 0.53→0.67 (+26.4%)
- `alternative_to` → [[2026-Beyond-LM]]: 初始化路线对立——Vision-first (Meissonic T2I) vs 从零训练（无 LLM/T2I 初始化）
- `alternative_to` → [[2025-SDAR-VL]]: 初始化路线对立——Vision-first (Meissonic T2I, 1B) vs LLM-first (dLLM 块状扩散)
- `alternative_to` → [[2025-LaViDa]]: 初始化路线对立——Vision-first (Meissonic 1B, 3.5M 数据) vs LLM-first (LLaDA 8B, 1.6M 数据)；代表 dLLM 统一模型初始化路线的两个对称端点

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- 首次验证了"从 T2I 模型出发构建统一多模态模型"的可行性，开辟了区别于 LLM-first 路线的**第三条路线**（Vision-first），以极低资源（1B 参数、3.5M 数据）实现 competitive 的多模态统一
- 证明强视觉先验初始化可以大幅降低统一模型的训练门槛——GenEval 0.61 接近 SD3 (0.62)，且超越了 Meissonic 自身基座 (0.54)，说明联合训练反哺了原始 T2I 能力
- 提供了 dLLM 推理速度的新基准——1.49s 延迟，4-11x 加速，在 1B 规模实现实用级推理速度
- 为 [[problem-tree#Diff-2b]] (MM-DiT 设计空间) 提供了首个"MM-DiT 用于统一理解+生成"的实证

### 未解决的问题
- 问题: 文本理解/推理能力严重受限（CLIP 77 token 硬约束）
  - 为什么难: CLIP text encoder 是架构硬约束，替换需重新设计 MM-DiT cross-attention 结构。不是微调能解的问题
  - 潜在思路: CLIP+T5 双编码器（类 SD3 triple text encoder）；在 MM-DiT 上方添加小型 transformer decoder 绕过 CLIP 限制
- 问题: 视觉先验路线的 scaling 行为完全未知
  - 为什么难: Muddit 仅验证了 1B 规模。当参数量足够大时 LLM 骨干的文本/推理优势可能远超视觉先验初始化优势，导致两条路线在大规模下收敛。无任何 >3B 的 Vision-first 统一模型数据点
  - 潜在思路: 做 scaling study——固定架构，在 1B/3B/7B 对比"视觉先验"vs "LLM 先验"初始化的训练效率和最终性能
- 问题: 无 RL/CoT 阶段，推理能力天花板低
  - 为什么难: 1B 模型容量有限 + 77 token 限制 = CoT 无法展开、RL 探索空间受限
  - 潜在思路: 跳过 CoT，直接用 Self-GRPO（DiMOO Block 3）做 entity-level 联合优化，不依赖长文本推理
- 问题: GenEval 暴跌消融的因果归因未明
  - 为什么难: 移除联合训练 → GenEval 61.6→28.3 可能是 catastrophic forgetting（遗忘 Meissonic 原始分布）而非"协同缺失"，两种因果解读的政策含义不同
  - 潜在思路: 加入 EWC/L2 regularization 防遗忘后重做消融；测量仅 T2I 训练后在 Meissonic 原始评测上的表现

### 对问题树的推进
- 推进了 [[problem-tree#Uni-1]] (补充新路线): 在 Diffusion 原生路线中定义了新子路线"Vision-first + T2I 初始化"，证明不仅 LLM→统一可行 (MMaDA/DiMOO)，T2I→统一也可行，将讨论从"哪种架构"扩展到"哪种先验初始化"
- 推进了 [[problem-tree#Uni-2a]] (强化 evidence): GenEval 暴跌消融是 KB 中最极端的"双向协同必要性"证据——在小模型+少数据 regime 下协同不是"锦上添花"而是"生存条件"
- 推进了 [[problem-tree#Diff-1b]] (补充架构数据点): 证明离散扩散不限于标准 transformer——MM-DiT (FLUX-style) 也可做 masked diffusion 统一模型
- 推进了 [[problem-tree#Diff-2b]] 🔴→🟡 (首个实证): KB 中首个将 MM-DiT 用于统一理解+生成的工作，证明 dual-stream→single-stream 设计在统一场景下可行，但也暴露了 CLIP 编码器瓶颈
- 新增问题: [Uni-1e] 视觉先验 vs 语言先验初始化的 scaling crossover——在什么规模/数据量下两种策略的优势发生交叉？
- 新增问题: [Diff-2b-sub] MM-DiT 在统一模型中的文本处理瓶颈——dual-stream 的前半部分模态分离对 I2T 方向是结构性约束吗？

## 个人深度评注

### [Critic] 增量贡献评估
| Block | 新颖性 | 理由 |
|-------|--------|------|
| Block 1: Visual-Prior Backbone | **中** | "从 T2I 出发构建统一模型"作为路线验证有价值——KB 中所有其他方法都从 LLM 出发，Muddit 提供了对称实验点。但 Meissonic 本身非本文贡献，"用预训练模型初始化"是标准做法。结果(GenEval 0.61)远未达到 LLM-first 路线水平，减弱了路线说服力 |
| Block 2: Lightweight Text Decoder | 低 | 线性头是最简单的 baseline，无设计创新。成功归因于 MM-DiT hidden states 质量而非解码器设计。未对比更复杂解码器 |
| Block 3: Balanced Joint Training | 低中 | 50/50 均衡采样是标准超参搜索。联合训练协同已被 P-Uni-01 充分验证。GenEval 暴跌是极端数据点但可能混淆了 catastrophic forgetting |
| Block 4: MM-DiT dual-/single-stream | 低 | 完全继承自 FLUX/Meissonic，无架构修改。贡献仅在于验证该架构可用于双向任务 |

### [Critic] 核心判断: 路线验证 > 方法创新
Muddit 的核心学术价值不在于任何单个 Block 的技术新颖性，而在于**路线验证**——证明从 T2I 模型出发是一条可走通的统一模型路线。但在当前结果下（1B, GenEval 0.61, VQAv2 68.2, 无 CoT），LLM-first 路线在绝对性能（8B, GenEval 0.88, MMMU 58.6）和能力覆盖（CoT、数学推理、长文本）上严格优于 T2I-first 路线。

关键对比:

| 维度 | Vision-first (Muddit) | LLM-first (MMaDA/DiMOO/LaViDa) |
|------|----------------------|-------------------------------|
| 骨干来源 | Meissonic (T2I, 1B) | LLaDA (LLM, 8B) |
| 编码的知识 | 视觉生成分布 + CLIP text-image 对齐 | 语言语义/推理/世界知识 |
| 参数效率 | 极高（1B 即 competitive） | 低（需 8-10B） |
| 数据效率 | 极高（3.5M 即可） | 低（需 数M-110M） |
| 文本推理 | 几乎不可能（CLIP 77 token） | 强（GSM8K 73.4, MMMU 58.6） |
| 生成质量 | 中等 (GenEval 0.61) | 高 (GenEval 0.88) |
| 推理速度 | 极快 (1.49s, 4-11x speedup) | 较慢 |
| 最佳定位 | 资源受限、视觉为主场景 | 通用多模态智能 |

### [Critic] 关键隐含假设
1. **T2I cross-attention 可逆性**: MM-DiT 的 cross-attention 在 T2I 训练中学到的 text→image 信息流能被反向用于 I2T。缺乏理论保证
2. **1B 参数有足够"剩余容量"给文本能力**: 视觉生成是参数密集型任务，T2I 预训练后模型可能接近满容量
3. **Cosine masking schedule 对双向任务同样最优**: 图像去噪是粗→细，文本恢复可能是核心→外围，最优 schedule 可能不同
4. **联合训练暴跌是"协同缺失"而非"灾难遗忘"**: 两种因果解读的政策含义完全不同（前者要求更多联合训练、后者要求正则化防遗忘）
5. **3.5M 数据足以激活视觉先验的文本理解潜力**: DiMOO 用 110M 数据达到更高性能，Muddit 的数据是否足以真正验证 T2I-first 路线的上界？

### [Connector] Muddit 在 dLLM 统一模型谱系中的定位
```
路线 A: LLM-first, 模态无关全共享
  LLaDA-8B → MMaDA (2025-05, NeurIPS)     方法论开创
  LLaDA-8B → Lumina-DiMOO (2025-10)       数据工程 scaling

路线 B: LLM-first, 非对称专用架构
  LaViDa-8B → LaViDa-O (2025-09)          架构设计
            → LaViDa-R1 (2026-02)          RL 方法论

路线 C: Vision-first, 全参数共享  ← Muddit 开辟的新路线
  Meissonic (T2I) → Muddit (2025-05)       视觉先验 + 轻量适配
```
Muddit 独立于路线 A/B，开辟路线 C。核心学术价值在于提供对照实验点，对 [[problem-tree#Uni-1]] 有重要补充——在"哪条路线最可能 scale"的讨论中，T2I→unified 也是一条有基础的路线，特别适合参数受限、推理速度优先的场景。

### [Ideator] 潜在研究方向
1. **Visual-Prior Backbone + LLM Text Branch**: 以 Muddit 的 MM-DiT 视觉骨干为主，添加中等规模 LLM 分支（如 Qwen2.5-3B）做文本理解/生成（Elastic-MoT 的反向应用）。动机: 解决 Muddit 最大短板（文本能力弱）同时保留视觉先验优势。依赖: Muddit Block 1 + LaViDa-O Block 1 (Elastic-MoT)。风险中等、可行性中等偏高。总参数量 ~4B，计算门槛可控
2. **Self-GRPO for Small Unified Models (1B)**: 跳过 CoT，直接用模型自身 VQA 能力（68.2%）做 entity-level T2I 评估，应用 Self-GRPO 联合优化。动机: Muddit 缺乏 RL 阶段，DiMOO Self-GRPO 方法论可迁移。依赖: Muddit Block 3 + DiMOO Block 3 + LaViDa-R1 Block 4。风险中偏高（小模型 bootstrapping 偏差更严重），但若成功将是首个 1B 规模 dLLM RL 验证
3. **MM-DiT + VQ-VAE 联合语义增强**: 解冻 Muddit 的 VQ-VAE decoder，添加 CLIP contrastive loss + SigLIP 语义蒸馏，使 VQ token 同时保留重建质量和语义信息。动机: 解决 Tok-3a/Tok-2c。依赖: Muddit Block 1/4 + MM-DiT 与 VQ-VAE 的紧密耦合优势。风险中等（多目标训练可能导致 codebook collapse）

### [Ideator] 新 Pattern 候选
- **候选 P-Uni-03: 联合训练的协同效应在小规模下更关键**: Muddit 移除联合训练暴跌 >50%（vs MMaDA 的温和提升）。在 1B+3.5M regime 下协同是"生存条件"而非"锦上添花"。需要第二篇小规模统一模型验证
- **候选: 视觉先验初始化在数据稀缺下的效率优势**: Muddit 3.5M 数据达 GenEval 0.61，数据效率极高。暗示 P-Uni-02 需要修订为"全共享 + (大规模数据 **OR** 强先验初始化) 可匹配非对称分支"。需第二篇 Vision-first 工作验证
