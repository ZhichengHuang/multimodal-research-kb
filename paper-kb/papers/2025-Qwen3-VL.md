---
title: "Qwen3-VL Technical Report"
authors: [Shuai Bai, et al. (Alibaba Qwen Team)]
date: 2025-11
venue: arxiv
url: "https://arxiv.org/abs/2511.21631"
tags: [pretraining, architecture, understanding, scaling, data]
category: "pretraining/vlm-architecture"
level: 3
status: read
importance: high
problem_tree_nodes: [PT-1b, PT-2d, PT-3b, RL-3c]
aliases: [Qwen3-VL]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结
Qwen3-VL 是阿里提出的多模态大模型系列（2B/4B/8B/32B dense + MoE），通过 Interleaved-MRoPE（三轴位置编码）和 DeepStack（多层视觉注入）两大架构创新，在 256K 上下文窗口下实现了图像-视频-文本交叉理解的 SOTA 性能。

## 核心 Insight
**空间-时序位置感知是多模态理解的关键瓶颈。** 传统方法将视觉 token 当作线性序列处理，丢失了二维空间和时间结构。Interleaved-MRoPE 通过在 temporal/height/width 三轴分别施加旋转位置编码，让模型在注意力计算中自然保留空间-时序关系，而 DeepStack 通过在 LLM 多层注入视觉特征实现渐进式视觉-语言融合，避免了仅在输入层注入导致的信息瓶颈。

## 与已有工作的关系
- **继承自**: Qwen3 LLM 系列（语言骨干，所有模型变体继承 Qwen3 的预训练知识和 MoE 路由机制）; SigLIP 视觉编码器技术栈（ViT patch 编码范式，与 [[2025-LLaDA-V]]、[[2025-LaViDa]]、[[2025-VidLaDA]]、[[2025-SDAR-VL]]、[[2026-Beyond-LM]] 共享相同的视觉编码器技术路线）; MRoPE 位置编码（在 Qwen2-VL 基础上扩展到 temporal/height/width 三轴）
- **对比**:
  - 与 [[2025-KimiK2.5]]: 同为大规模 AR MoE 多模态模型（K2.5 1.04T/32B 激活 vs Qwen3-VL 32B-MoE/8B 激活），K2.5 聚焦 Agent Swarm 和跨模态 RL，Qwen3-VL 聚焦高分辨率理解和视频时间定位；MoonViT-3D（时间 4× 压缩）vs SigLIP2+token merging（空间 4× 压缩），两种视频编码维度不同
  - 与 dLLM 统一模型系列（[[2025-MMaDA]]、[[2025-Lumina-DiMOO]]、[[2026-LaViDa-R1]]、[[2025-LaViDa-O]]、[[2025-MMaDA-Parallel]]）: 根本架构分歧——Qwen3-VL 是纯 AR causal 生成，dLLM 采用 masked diffusion 双向注意力。AR 在顺序解析（文档/图表）和短时序推理上有结构性优势（P-Diff-02），dLLM 在全局推理和视觉 infilling 上有优势。Qwen3-VL MMMU 72.3% 远高于同期 dLLM 旗舰（DiMOO 58.6%），但 Qwen3-VL 不具备图像生成能力
  - 与 [[2025-VidLaDA]]: 视频理解路线对比——VidLaDA 以 dLLM 双向注意力对抗 AR causal attention 在长视频全局推理上的不对称感受野（LongVideoBench +3.2 超越 Qwen2.5-VL），Qwen3-VL MVBench 64.3% 体现 AR 在短视频时序推理上的优势（对应 VidLaDA MVBench 59.4 vs Qwen2.5-VL 69.6 的 -10.2 差距规律）
  - 与 [[2026-LaViDa-R1]]: 明确竞争——LaViDa-R1 在 Lisa-Grounding P@0.5 66.7 超越 Qwen3-VL-8B 62.4，是 dLLM 在 grounding 专项任务上可超越同规模 AR 的直接证据
  - 与 [[2025-OpenMMReasoner]]: 同为 AR VLM 后训练路线，但定位不同——OpenMMReasoner 基于 Qwen2.5-VL-7B，以 Qwen3-VL-235B 为教师模型做知识蒸馏（8× 采样多样性扩展）；Qwen3-VL 是基础预训练工作而非后训练改进
  - 与 [[2026-Beyond-LM]]: 两者均是大规模多模态基础模型，但路线截然不同——Qwen3-VL 是 AR+LLM 初始化，Beyond-LM 是连续 flow matching 从零训练（51:1 视觉:语言数据比）；SigLIP 技术栈共享但 Beyond-LM 使用 RAE 连续表示
- **互补**:
  - [[2025-DiffusionVL]]: 明确以 Qwen2.5-VL 系列为 AR-to-Diffusion 转换的目标对象，可将 Qwen3-VL 的预训练知识通过 Block Diffusion 扩散微调迁移到 dLLM 骨干，实现 AR 知识与 diffusion 双向注意力的融合
  - [[2026-EBPO]]: EBPO 的 James-Stein shrinkage baseline 是架构无关的 RL 改进，可直接应用于 Qwen3-VL 的 RLHF 阶段（PPO with KL penalty），在困难样本（所有采样均失败）场景注入全局先验负梯度信号，改善 advantage 估计方差
  - [[2025-OpenMMReasoner]]: SFT 数据多样性扩展（×8 采样）+ 跨域数据混合（数学推理正迁移）的方法论可应用于 Qwen3-VL 系列的后训练数据构造；Qwen3-VL-235B 的超大规模已作为 OpenMMReasoner 教师模型，验证了 Qwen3-VL 的蒸馏价值
  - [[2025-dMLLM-TTS]] / [[2025-Muddit]] / [[2025-LLaDA-V]]: 作为 AR baseline 的性能参考上界——dLLM 工作通常将 Qwen2-VL/Qwen2.5-VL 列为追赶目标，Qwen3-VL 提供更强的 AR baseline，推动 dLLM 研究进步

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 整体架构
- **Vision Encoder**: ViT, patch size 14×14, 支持动态分辨率（最高 1344×1344），保持原始宽高比
- **LLM Backbone**: Qwen3 系列 (2B/4B/8B/32B dense; 32B-MoE with 64 experts, top-8 routing)
- **Vision-Language Connector**: 线性投影层，将视觉 encoder 输出映射到 LLM embedding 空间
- **上下文窗口**: 256K tokens，支持文本-图像-视频交叉输入
- **所有模型变体共享相同的 Vision Encoder**，仅 LLM backbone 不同

### 训练流程（三阶段）
1. **Vision-Language Alignment (预训练)**: 5M 高质量 image-caption 对, lr=1e-3, batch=1024, ~50K steps
2. **SFT**: 1.5M 多模态指令数据 (35% 学术基准 + 30% Web + 20% 合成 + 15% 文档/图表), lr=5e-5, batch=256, context=8192, ~100K steps
3. **RLHF**: 200K 人类偏好对, PPO with KL penalty, lr=1e-5

### 数据策略
- 预训练: 5M 图文对，CLIP-based 语义过滤（阈值 0.75），感知哈希去重（减少 12% 数据污染）
- SFT: 混合学术/Web/合成/领域数据，250K 合成样本由 Qwen3-32B 生成并经人类验证（≥80% 一致性）
- 视频: 500K 视频理解样本，8-16 帧采样，时长 5s-5min

## Building Blocks（可复用组件）

### Block 1: Interleaved-MRoPE（三轴旋转位置编码）
- **做法**: 将标准 RoPE 扩展到 3 轴（temporal, height, width），对图像/视频的空间和时序维度分别编码。文本 token 只使用 temporal 轴，视觉 token 使用全部 3 轴。在 interleaved 多模态序列中，不同模态的 token 各自维护自己的位置索引。
- **机制 (WHY it works)**: 标准 1D 位置编码将 2D 图像 token 拉平为序列，破坏了空间邻近性（相邻 patch 在序列中可能距离很远）。MRoPE 通过分轴编码，让注意力权重自然反映空间距离——同行/同列的 patch 在对应轴上位置接近，注意力更强。对于视频，temporal 轴编码帧序关系。
- **适用条件**: 任何需要处理 2D/3D 空间结构的 Vision-Language 模型；尤其在高分辨率、多图、视频场景优势显著
- **什么时候会 break**: 当视觉内容的关键信息不依赖空间位置（如纯文字截图、简单分类任务）时，额外的空间编码可能增加计算开销而收益有限
- **可组合方向**: 可与 dLLM 的双向注意力结合（当前 dLLM 多用标准 positional embedding）；可扩展到 3D 视觉（医学影像、点云）

### Block 2: DeepStack（多层视觉特征注入）
- **做法**: 不仅在 LLM 输入层注入视觉 token，还在 LLM 的多个中间层（约第 10/20/30 层）注入视觉特征，实现渐进式视觉-语言融合。
- **机制 (WHY it works)**: 仅在输入层注入视觉信息会造成"信息瓶颈"——视觉特征必须在第一层就与语言表示完全对齐，这对复杂视觉推理是不够的。多层注入让 LLM 在不同抽象层次接收视觉信息，低层处理局部视觉细节，高层处理全局语义，类似人类视觉处理的层级特性。（**注: 此为论文叙事。实际上真实增益可能来自缓解梯度消失——提供更短的梯度路径，且注入的是同一 ViT 最终层输出，并非真正的多层级视觉特征。多层特征注入是公认的性能提升手段，非技术创新。**）
- **适用条件**: 深度 LLM（≥24 层），视觉理解任务要求细粒度推理（文档、图表、空间关系）
- **什么时候会 break**: 对浅层模型（如 2B 参数量）注入点过少，效果可能退化为单层注入；注入层的选择需要仔细调参
- **可组合方向**: 可与 cross-attention connector（如 Flamingo 风格）结合；可用于 diffusion decoder 的多层条件注入

### Block 3: 动态分辨率 + 自适应 Token 合并
- **做法**: 保持输入图像原始宽高比处理（最高 1344×1344），并通过 token merging 控制视觉 token 数量以平衡计算开销。
- **机制 (WHY it works)**: 固定分辨率/宽高比会导致图像失真和信息丢失（尤其对文档、长图）。自适应处理保留了原始信息，token merging 则在保持关键信息的前提下减少计算量（2.3× 效率提升）。
- **适用条件**: 任何需要处理多样化分辨率输入的多模态模型
- **什么时候会 break**: 极高分辨率图像（>2K）即使有 token merging 也会产生大量视觉 token，可能超出上下文窗口
- **可组合方向**: 与 visual tokenizer（如 VQVAE）结合实现更高压缩率

### Block 4: 文本化时间定位（Video Temporal Grounding）
- **做法**: 模型以文本格式 "[00:15-00:30]" 输出视频时间段定位，通过 100K 时间标注训练样本学习，结合帧级表示和时间位置编码预测时间区间。
- **机制 (WHY it works)**: 将时间定位转化为文本生成任务（而非回归任务），利用了 LLM 强大的序列生成能力。时间位置编码确保了帧间时序连续性，减少时间抖动。
- **适用条件**: 视频问答、时间定位等需要精确时间引用的任务
- **什么时候会 break**: 极长视频（>5min）或需要极高时间精度（<1s）的场景
- **可组合方向**: 可与视频生成的时间条件控制结合

### Block 5: MoE 高效推理
- **做法**: 64 experts, top-8 routing, 带有 entropy regularization 防止 expert collapse。32B-MoE 在每个 token 只激活 8B 参数。
- **机制 (WHY it works)**: 稀疏激活让模型保持大容量（64 experts 的知识）同时只付出小容量的计算代价。Entropy regularization 确保所有 expert 都被使用，防止少数 expert 垄断。
- **适用条件**: 需要在推理效率和模型容量之间平衡的部署场景
- **什么时候会 break**: Expert 负载不均衡时推理延迟可能波动；batch size 较小时 MoE 的效率优势不明显
- **可组合方向**: 可与 dLLM 的并行解码结合（dLLM + MoE = 更高效的并行生成）

## Anti-patterns / 已知失败模式
- **[AP-1] MRoPE 对信息图/表格理解提供错误归纳偏置**: 空间邻近性编码假设"相邻 patch 更相关"，但表格/图表的语义关系是全局性的（列标题与任意行数据相关），局部偏置反而是错误的先验。（支撑: KB [[P-Diff-02]] dLLM 在 AI2D -3.3, DocVQA -2.3 弱于 AR，同一根因的不同表现）
- **[AP-2] Temporal 轴语义混淆**: 文本 token 的"对话序列位置"和视频 token 的"帧序号"共享同一 temporal 轴，长视频多轮对话中存在语义混淆风险。三轴 head_dim/3 的频率分配假设三轴信息量均等，但 temporal 轴语义远比 height/width 丰富。
- **[AP-3] DeepStack 在浅层模型（2B）上退化**: 第 10/20/30 层注入点假设 30+ 层深度。2B 模型层数较少（~24 层），注入点分布不合理，效果可能退化到单层注入水平。消融仅在旗舰模型进行。
- **[AP-4] DeepStack 的"信息瓶颈"诊断可能不准确**: 单层注入的视觉 token 在每层自注意力中仍参与交互，并非"仅处理一次"。DeepStack 的真实增益可能来自缓解梯度消失（提供更短的梯度路径）而非多层级视觉信息注入。且注入的是同一 ViT 最终层输出，不是真正的多层级视觉特征。
- **[AP-5] Token Merging 在 OCR/高精度文档理解中失效**: 标准 token merging（均匀合并）对"重要性"无先验，文档关键词和装饰背景享有相同压缩处理。2.3× 效率提升的代价是细粒度文本识别能力下降。（支撑: LaViDa anti-pattern TextVQA 56.3, DocVQA 59.0 弱于 AR，根因相同）
- **[AP-6] 文本化时间定位的结构性精度限制**: 秒级离散化 + 帧级采样稀疏（8-16 帧）造成双重精度瓶颈。关键事件在两个采样帧之间的概率为 7/8~15/16，文本格式的秒级精度是虚假精度——底层采样分辨率是真正限制。
- **[AP-7] MoE 在视觉密集序列中的路由热点**: 长视频场景大量同质视觉 token 趋向相同 expert，造成负载严重不均。Entropy regularization 的强度需要权衡：过高破坏合理的视觉/语言专家分化，过低无法防 collapse。（支撑: KB [[P-Uni-03]] Beyond-LM 发现 51:1 视觉-语言数据不平衡可能导致 routing collapse）
- **[AP-8] MRoPE 与 dLLM 部分因果优化冲突**: MRoPE 假设空间相邻 patch 应有强注意力，但 dLLM 的 Prefix-DLM（[[2025-LaViDa]]）对视觉 token 施加因果约束，阻断了 MRoPE 期望的双向空间注意力。MRoPE 与完全双向注意力兼容且互补，但与 KV 缓存因果优化冲突。

## 实验关键发现
- **Interleaved-MRoPE 是最关键组件**: 消融实验显示移除后 MMMU 下降 4.2%，是所有单组件中影响最大的
- **DeepStack 贡献 +3.1%**: 多层注入显著优于仅输入层注入
- **动态分辨率 +2.8%**: 固定 384×384 baseline 明显更弱
- **Token merging 在保持性能的同时减少 2.3× 计算量**
- **32B Dense 旗舰结果**: MMMU 72.3% (GPT-4o 69.2%), MathVista 68.1% (Claude-3.5 65.8%), DocVQA 93.8%, ChartQA 89.2%
- **8B 模型**: MMMU 68.9%, MathVista 63.2%, 保持良好 scaling 特性
- **32B-MoE**: 相比 dense 仅损失 1.2% 性能，推理延迟降低 40%
- **视频理解**: MVBench 64.3%

## Relations (结构化)
- `alternative_to` → [[2025-KimiK2.5]]: 同为大规模 AR MoE 多模态模型，K2.5（1.04T/32B 激活，384 experts）vs Qwen3-VL（32B-MoE/8B 激活，64 experts）；K2.5 聚焦 Agent Swarm/Cross-Modal RL/GRM，Qwen3-VL 聚焦高分辨率图像视频理解；视频编码策略对比——MoonViT-3D 时间 4× 压缩 vs SigLIP2+token merging 空间 4× 压缩
- `alternative_to` → [[2025-MMaDA]]: AR causal VLM vs dLLM 统一模型根本性对比；Qwen3-VL MMMU 72.3% 远超 MMaDA（无 MMMU 报告）；Qwen3-VL 无图像生成能力，MMaDA 具备理解+生成统一能力
- `alternative_to` → [[2025-Lumina-DiMOO]]: AR VLM vs dLLM 统一模型；Qwen3-VL MMMU 72.3% vs DiMOO 58.6%（同期最强 dLLM 旗舰）；DiMOO GenEval 88% 超越专用模型而 Qwen3-VL 无生成能力
- `alternative_to` → [[2026-LaViDa-R1]]: LaViDa-R1 在 Lisa-Grounding P@0.5 66.7 **超越** Qwen3-VL-8B 62.4——是 dLLM 在 grounding 专项上直接超越同规模 AR VLM 的明确证据；但 Qwen3-VL 在综合理解（MMMU/MathVista）上有更大规模优势
- `alternative_to` → [[2025-LLaDA-V]]: AR VLM vs dLLM 纯理解模型；LLaDA-V 在 MMMU (+3.2)、MMMU-Pro (+6.9) 等推理密集任务上系统性优于 AR 同规模 baseline，但整体 MMMU（~48.6%）仍大幅落后 Qwen3-VL（72.3%）；Qwen3-VL 作为 LLaDA-V 追赶的 AR 上界
- `alternative_to` → [[2025-VidLaDA]]: AR 视频理解 vs dLLM 视频理解；VidLaDA 在 LongVideoBench (+3.2)、MLVU (+3.0) 上超越 Qwen2.5-VL（Qwen3-VL 的前代），但 MVBench 弱于 Qwen2.5-VL (-10.2)；Qwen3-VL MVBench 64.3% 体现 AR 在短视频时序推理上的结构性优势
- `alternative_to` → [[2025-LaViDa-O]]: AR VLM vs dLLM 统一理解+生成模型（Elastic-MoT 10.4B）；LaViDa-O grounding 速度比 Qwen2.5-VL-7B 快 6.8×（坐标量化并行预测），但 Qwen3-VL 理解综合性能更强；LaViDa-O 具备 T2I/编辑生成能力而 Qwen3-VL 无
- `alternative_to` → [[2025-MMaDA-Parallel]]: AR VLM vs dLLM 并行推理-生成模型；Qwen3-VL 无 interleaved 推理+图像生成能力，MMaDA-Parallel 在 ParaBench 59.8% output alignment 超越 Bagel
- `alternative_to` → [[2025-LaViDa]]: AR VLM vs dLLM VLM 家族鼻祖；LaViDa 的 Complementary Masking 和 Prefix-DLM 加速技术为 dLLM 路线奠基，Qwen3-VL 作为 AR 路线的旗舰代表持续推高 AR baseline
- `alternative_to` → [[2025-SDAR-VL]]: AR VLM vs 块状离散扩散 VL 理解模型；SDAR-VL 以 LLaVA-OneVision 为 AR 基线（弱于 Qwen3-VL），通过 ABNS/EMRS/PBNC 解决训练稳定性，Qwen3-VL 为 dLLM 理解工作提供更严格的 AR 上界参考
- `alternative_to` → [[2025-Muddit]]: AR VLM vs Vision-first 1B dLLM 统一模型；Muddit 从 T2I 预训练模型出发，4.2× 推理加速；Qwen3-VL 从 LLM 出发，理解能力远强但无生成
- `alternative_to` → [[2025-dMLLM-TTS]]: AR VLM vs dLLM test-time scaling 系统；dMLLM-TTS 展示 dLLM 推理时搜索（+17.9%-+29.4% GenEval），Qwen3-VL 作为 AR 的直接生成上界
- `alternative_to` → [[2026-Beyond-LM]]: AR VLM（LLM 初始化）vs 从零训练的连续 flow matching 统一模型；两者均为大规模多模态基础模型，但训练范式根本不同；共享 SigLIP 视觉编码器技术栈，但 Beyond-LM 使用 RAE 连续表示，Qwen3-VL 使用 ViT+linear projection
- `alternative_to` → [[2025-ReDiff]]: AR VLM vs dLLM 主动精炼框架（基于 LLaDA-V）；ReDiff 解决 dLLM 并行去噪的幻觉问题（CLAIR +11.2），Qwen3-VL 的 AR 顺序生成天然避免此类错误级联但引入单向 attention 限制
- `alternative_to` → [[2025-Sparse-LaViDa]]: AR VLM vs dLLM 稀疏推理加速框架（LaViDa-O + mask token 截断 2.83×）；两种架构的推理效率对比——AR 有 KV-cache 但串行，dLLM 可并行但多步；Sparse-LaViDa 的加速目标是追赶 AR 推理效率
- `alternative_to` → [[2025-DiffusionVL]]: DiffusionVL 明确以 Qwen2.5-VL-7B 为 AR-to-Diffusion 转换的目标 AR 模型，实现 MMMU-Pro 35.1（接近 AR 基线 36.7），Block Diffusion 将 Qwen2.5-VL 转为扩散骨干；Qwen3-VL-8B 是更强的 AR 基线，DiffusionVL 方法论可延伸应用
- `enables` → [[2025-OpenMMReasoner]]: Qwen3-VL-235B-Instruct 作为 OpenMMReasoner 的**教师模型**——对每个问题 ×8 采样生成高质量推理轨迹（经人类验证 ≥80% 一致性），使 Qwen2.5-VL-7B 学生模型实现 11.6% 多模态推理提升；Qwen3-VL 的蒸馏价值被直接验证
- `enables` → [[2025-DiffusionVL]]: DiffusionVL 的 AR-to-Diffusion 转换方法论（Block Diffusion + 5% 数据扩散微调）可以直接应用于 Qwen3-VL 系列，将 Qwen3-VL 的预训练知识迁移到 dLLM 骨干，潜在实现 AR 理解强+dLLM 双向注意力的结合
- `combines_with` → [[2026-EBPO]]: EBPO 的 James-Stein shrinkage baseline 是架构无关的 RL 改进（仅修改 advantage baseline 估计），可直接替换 Qwen3-VL RLHF 阶段 PPO 的 advantage 估计部分，在困难样本（saturated failure，G=4/8 时 65.6%/43.0% 概率全部失败）场景通过全局先验 μ_glob 注入非零梯度信号，无需修改 PPO 框架其他部分
- `combines_with` → [[2025-OpenMMReasoner]]: OpenMMReasoner 验证的 SFT 数据策略（×8 采样多样性扩展、跨域数学推理正迁移、过度过滤是 anti-pattern）可应用于 Qwen3-VL 的 SFT 数据构造（Qwen3-VL SFT 使用 250K 合成数据 + Qwen3-32B 生成验证，可进一步应用多样性扩展思路）
- `conflicts_with` → [[2026-LFPO]]: LFPO 的核心 Theorem 3.1（cross-entropy 梯度精确等于速度残差）依赖 masked diffusion 的离散 token 空间特性，无法直接迁移到 Qwen3-VL 的 AR 自回归生成框架；LFPO 仅适用于 dLLM（LLaDA、DiffuCoder），与 AR 架构根本不兼容
- `conflicts_with` → [[2025-XDLM]]: XDLM 的 stationary noise kernel（k=0.1 混合 mask+uniform 噪声）是离散扩散训练目标的改进，与 AR 的 next-token prediction 训练范式完全不同，无法应用于 Qwen3-VL
- `motivated_by` → [[2025-VidLaDA]]: VidLaDA Proposition 3.1 证明 AR 单向 attention 对视频的不对称感受野（早期 token 可见频率高但信息来源少），为 Qwen3-VL Interleaved-MRoPE 的空间-时序位置编码设计提供了对比性动机——AR 的位置编码设计需要从源头解决时空感知问题

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- **多模态序列的空间-时序位置编码**: Interleaved-MRoPE 系统性解决了将 2D 图像和视频时序信息编码到 1D 序列的信息丢失问题（消融 +4.2%），是 AR VLM 中最完整的空间-时序位置编码方案
- **视觉-语言对齐的深度不足**: DeepStack 多层注入缓解了单层注入的梯度路径过长问题（+3.1%），为大模型（32B+）的视觉推理提供了工程解法
- **AR VLM Scaling**: 提供了 2B→4B→8B→32B dense + MoE 的完整 scaling curve，32B-MoE 40% latency reduction with 1.2% quality loss 验证了多模态 MoE 的实用性

### 未解决的问题
- **dLLM 位置编码迁移**:
  - 为什么难: MRoPE 为 causal attention 设计，在 bidirectional masked diffusion 中所有 token 互相可见，旋转矩阵对距离的表示是否仍有效需验证
  - 潜在思路: 在 VidLaDA 的 LLaDA-8B 基座上直接替换 1D RoPE 为 3-axis MRoPE 做控制实验（详见 [[qwen3vl-crossover-to-dllm#方向A]]）
- **DeepStack 在 dLLM 中的效果**:
  - 为什么难: dLLM 的 bidirectional attention 每层都让视觉 token 参与全局交互，"信息瓶颈"可能本不存在，DeepStack 增益的真实来源（梯度路径 vs 信息瓶颈）需澄清
  - 潜在思路: 在 dLLM 中用 cross-attention 替代直接 embedding 注入，可能实现真正的多层级视觉-语言交互（详见 [[qwen3vl-crossover-to-dllm#方向B]]）
- **AR 基座质量对扩散微调的定量关系（扩展 [Diff-1e]）**:
  - 为什么难: DiffusionVL 仅有 Qwen2.5-VL-7B 一个数据点，"5% 数据达到 95% AR 性能"的关系是否是常数（还是随 AR 基座质量变化）未知
  - 潜在思路: 用 Qwen3-VL-8B（同 size 更强 AR 基座）重复 DiffusionVL 实验，检验"95% rule"的稳定性

### 对问题树的推进
- 推进了 [[problem-tree#PT-1b]] 视觉 token 数量 tradeoff: Adaptive Token Merging + 256K 上下文是 AR 侧成熟方案，但 dLLM 侧迁移效果未知
- 推进了 [[problem-tree#PT-3b]] 多模态 Scaling Law: 提供了 AR 侧最完整的 scaling curve（2B→32B + MoE），dLLM 侧仍缺
- 推进了 [[problem-tree#RL-3c]] AR vs dLLM RL 对比: Qwen3-VL 成为 KB 中最强 AR baseline（MMMU 72.3%），使 AR/dLLM RL 方法论对比研究更有参考价值
- 新增问题: **dLLM 时序推理结构性弱点的位置编码解法**（MRoPE temporal 轴完全未被 KB 中任何 dLLM 工作尝试，是新暴露的 gap，参见 [[problem-tree#PT-4a]]）
- 新增问题: **多层视觉注入 vs 单点注入在 dLLM 中的效果**（[[problem-tree#PT-1c]] 的"dLLM 不需要复杂连接器"结论未覆盖"多层 cross-attention 注入"维度）

## 增量贡献评估
> **总体判断**: Qwen3-VL 是高质量的**工程系统论文**，核心价值在于将多个已知技术（MRoPE、动态分辨率、MoE）整合到 256K 上下文的多模态系统中。单个 Building Block 的方法论新颖性有限（MRoPE 是 2D RoPE 的直接推广，DeepStack 类似 cross-attention 变体，动态分辨率和 MoE 是业界标准实践）。真正的差异化来自系统级整合的工程执行质量。
>
> **对 KB 最有价值的贡献**不是性能数字，而是三个清晰的 ablation 结论: MRoPE (+4.2%) > DeepStack (+3.1%) > Dynamic Resolution (+2.8%)——这三个 delta 均对应 KB 中 dLLM 的现有 open questions（时序推理弱点 / connector 架构 / token 压缩策略），为 dLLM 研究提供了明确的改进方向优先级。

## 潜在研究方向
详见 [[qwen3vl-crossover-to-dllm]]，五个方向按优先级排序:
1. **MRoPE-dLLM**: 将三轴位置编码引入 dLLM 以解决时序推理弱点（[[problem-tree#PT-4a]]），实现成本低且与现有 VidLaDA 框架兼容
2. **Qwen3-VL→DiffusionVL 定量实验**: 填补 [[problem-tree#Diff-1e]] 的 AR 基座质量消融 gap
3. **DeepStack-dLLM**: 多层 cross-attention 视觉注入到 dLLM，潜在改善文档/图表理解

## 个人深度评注
- **DeepStack 不算技术创新，更像是性能工程**: 多层视觉特征注入是公认能提升视觉理解能力的做法（cross-attention 变体在 Flamingo 等工作中早有先例）。Qwen3-VL 采用 DeepStack 更可能是出于性能优化的考量——模型已经足够大，为了在 benchmark 上再挤出几个点而引入成熟的多层注入方案。+3.1% 的消融增益证明了它的工程价值，但不代表方法论突破。
- **整体判断**: Qwen3-VL 本质是一篇工程集成论文——将已有技术（MRoPE、多层注入、动态分辨率、MoE）整合到统一系统中并通过大规模数据和精细调参达到 SOTA。对 KB 而言，最大价值是作为 dLLM 的强 AR baseline 和三个清晰的 ablation delta 指引 dLLM 改进方向。
