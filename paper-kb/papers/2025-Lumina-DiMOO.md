---
title: "Lumina-DiMOO: An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding"
authors: [Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, Jinbin Bai, Qian Yu, Dengyang Jiang, Yuandong Pu, Haoxing Chen, Le Zhuo, Junjun He, Gen Luo, Tianbin Li, Ming Hu, Jin Ye, Shenglong Ye, Bo Zhang, Chang Xu, Wenhai Wang, Hongsheng Li, Guangtao Zhai, Tianfan Xue, Bin Fu, Xiaohong Liu, Yu Qiao, Yihao Liu]
date: 2025-10
venue: arxiv
url: "https://arxiv.org/abs/2510.06308"
tags: [diffusion, unified-model, pretraining, posttraining, rl, architecture, generation, understanding, data]
category: unified-model/diffusion-native
level: 3
status: read
importance: high
problem_tree_nodes: [Uni-1, Uni-2a, Uni-2c, Uni-3, Uni-5, Diff-1b, Diff-1c, Tok-2a, Tok-2d]
aliases: [Lumina-DiMOO, DiMOO, Self-GRPO, ML-Cache]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结
提出 Lumina-DiMOO，一个基于 LLaDA 初始化的 8B 全离散扩散统一模型，通过四阶段训练管线（预训练→中间训练→SFT→Self-GRPO）、任意分辨率支持（`<end-of-line>` token）、ML-Cache 推理加速，在 T2I 生成（GenEval 88%）、image-to-image（editing/style transfer/controllable generation）和图像理解（MMMU 58.6%）上全面超越 MMaDA，是迄今为止最完整的开源 dLLM Omni 模型。

## 核心 Insight
dLLM 统一模型的性能瓶颈不在架构（模态无关共享即可），而在**训练数据规模与管线设计**——80M 预训练 + 3M 中间训练 + 30M SFT 的四阶段管线 + 纯 VQ token（无独立视觉编码器）的极简架构在 8B 规模即可超越 FLUX.1-dev (GenEval) 和 GPT-4o 等。Self-GRPO 联合优化生成和理解任务的 RL 方案验证了"一个 loss 同时提升两种能力"。ML-Cache（基于 max logit 的选择性缓存）提供了 dLLM 独有的无训练推理加速方案。

## 与已有工作的关系
- **继承自**: [[LLaDA]]（离散扩散 LLM，提供 8B 预训练初始化权重）、[[aMUSEd]]（VQ tokenizer，codebook 8192, 16×16 下采样）
- **对比**: [[2025-MMaDA]]（同为 LLaDA-based dLLM 统一模型；DiMOO 在 GenEval 88% vs 63%、DPG 86 vs 70、UniGenBench 71 vs 41 全面超越）、[[2025-LaViDa-O]]（非对称 MoT vs DiMOO 全共享）
- **互补**: [[2026-LaViDa-R1]]（LaViDa-R1 聚焦 RL 方法论创新，DiMOO 聚焦系统级工程和任务覆盖广度）

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 整体架构
- **骨干**: 基于 LLaDA-Base 8B 初始化的 masked diffusion transformer（模态无关全共享，无独立分支）
- **图像 tokenizer**: aMUSEd-VQ, 16×16 下采样, codebook size=8192
- **视觉编码器**: 无（纯 VQ token，不使用 SigLIP/CLIP 等连续视觉编码器）
- **词表**: 126,345 LLaDA text tokens + 8,192 visual tokens + 特殊标记 = 134,537+
- **任意分辨率**: `<end-of-line>` token 在每行图像 token 末尾插入，保留 2D 结构信息，搭配 1D RoPE
- **统一目标**: cross-entropy mask prediction loss，同时对文本和图像 token 做 masked diffusion

### 四阶段训练管线
| 阶段 | 数据规模 | 内容 | 分辨率 |
|------|----------|------|--------|
| Stage I: 预训练 | 80M 图文对 | 多模态 mask-predict 预训练 | 渐进 ~256² → ~512² |
| Stage II: 中间训练 | 3M 专用图像 | image-to-image 任务（editing, subject-driven, controllable, style transfer） | 512 (I2I) / 1024 (T2I) |
| Stage III: SFT | 15M 理解 + 15M 生成 | system/user/answer 格式，masked answer prediction | 混合 |
| Stage IV: Self-GRPO | 文本 prompt | 联合 T2I+理解 RL，步轨迹跟随 | 1024 |

学习率: 2.0×10⁻⁴ → 3.0×10⁻⁶（逐阶段递减）

### Self-GRPO 机制
- 对每个 prompt 采样 G 张候选图像，用 N 个问题评估每张图的质量
- 联合目标: L(θ) = -∑w(g)(ℓ_T2I(g) + ℓ_MMU(g)) + β·KL(p_θ ∥ p_θ_ref)
- w(g) 基于理解正确率的 normalized softmax 加权
- **Step Trajectory Following**: 保留完整采样轨迹但仅对选定时间步 𝒯_sel 计算梯度，解决 1024² 图像（4096+ tokens）的显存瓶颈
- 保留 KL regularization（与 LaViDa-R1 的"去 KL"设计不同）

### 推理加速
- **并行采样**: 4 阶段 cosine masking schedule，所有 masked 位置同步预测
- **Block-Wise 理解**: 半自回归策略 + `</answer>` 早停
- **ML-Cache**: 基于 max logit 值识别稳定 token，复用其 KV 表征和 logits，额外 2× 加速
  - 三个超参: cache_ratio（复用比例）、warmup_ratio（预热步数）、refresh_interval（刷新频率）

## Building Blocks（可复用组件）

### Block 1: 纯 VQ Token 统一架构（无独立视觉编码器）
- **做法**: 不使用 SigLIP/CLIP 等连续视觉编码器，理解和生成都基于 aMUSEd-VQ 离散 token。通过大规模 SFT 数据（15M 理解样本）弥补 VQ token 缺乏语义信息的不足
- **机制 (WHY it works)**: 极简架构消除了"理解用连续编码器 vs 生成用 VQ token"的表示不一致问题。aMUSEd-VQ 虽然语义信息不如 CLIP 丰富，但通过大规模训练数据（80M+30M）让 LLM 骨干学会从 VQ token 中提取语义。无额外视觉编码器意味着整个模型共享同一套参数，训练/部署更简单
- **适用条件**: 需要大规模高质量训练数据弥补 tokenizer 语义缺失；模型规模足够大（8B+）以在参数中隐式学习视觉语义
- **什么时候会 break**: (1) 小规模模型和有限数据下，VQ token 的语义不足会严重限制理解性能；(2) 论文自己承认"低级视觉任务（super-resolution, dehazing, denoising）表现一般"——VQ token 丢失了像素级细节；(3) 精细视觉理解（如细粒度 OCR）可能不如有 SigLIP 编码器的方案
- **可组合方向**: 可与更强语义的 VQ tokenizer（如 SEED-tokenizer, 多目标训练的 tokenizer）组合；或探索"VQ token + 少量连续 token"的混合方案

### Block 2: `<end-of-line>` 任意分辨率机制
- **做法**: 在每行图像 token 末尾插入 `<end-of-line>` 特殊 token，配合 1D RoPE 位置编码，使模型能处理任意宽高比的图像。无需修改位置编码方案或引入 2D 位置嵌入
- **机制 (WHY it works)**: `<end-of-line>` 显式标记了行边界，使 1D 序列保留了 2D 空间结构信息。这比简单将 2D token 铺平为 1D（丢失行列信息）更有效。RoPE 的相对位置编码在同行内 token 间自然建立局部关联，跨行通过 `<end-of-line>` 的"语义断点"信号建立全局结构
- **适用条件**: 基于 1D RoPE 的 transformer 处理 2D 图像数据；需要支持可变宽高比
- **什么时候会 break**: (1) 极端宽高比（如 1:10 的长条形）时，行间信息传递路径过长；(2) 增加了 token 数量（每行多一个 token），对高分辨率大图造成额外开销
- **可组合方向**: 扩展到 3D 数据（`<end-of-line>` + `<end-of-frame>` 用于视频）；与 LaViDa-O 的 Stratified Sampling 结合（利用行结构信息指导采样）

### Block 3: Self-GRPO（自评估联合 RL）
- **做法**: 利用统一模型自身的理解能力评估 T2I 生成质量。对每个 prompt 采样 G 张图像，用 N 个自动生成的理解问题（基于 prompt 的 entity-relation-value 三元组）评估每张图。联合优化 T2I loss 和理解 loss，用 KL 正则化防止偏离。Step Trajectory Following 仅对选定时间步计算梯度，解决高分辨率显存问题
- **机制 (WHY it works)**: 利用 dLLM 的理解能力作为 T2I 的隐式 reward model——理解正确率高的图像被认为是更好的生成结果。联合优化 T2I+理解使两种能力互相强化。Step Trajectory Following 将显存开销从 O(T) 降到 O(|𝒯_sel|)，使 1024² RL 训练可行
- **适用条件**: 统一模型（同时具备理解和生成）；理解能力足够准确以作为 reward signal；有大量 text prompt 数据
- **什么时候会 break**: (1) 理解能力本身有偏差时，reward signal 也有偏差（bootstrapping 循环偏差问题）；(2) 保留 KL 正则化——LaViDa-R1 已证明 KL 在 image token 高熵分布下可能导致训练不稳定（NLL>6）；(3) entity-relation-value 三元组的自动生成质量直接影响 reward 质量
- **可组合方向**: 与 LaViDa-R1 的 answer-forcing 结合解决困难 prompt 的训练信号消失；与外部 VLM reward (如 GPT-4V) 组合做 reward 交叉验证；将 step trajectory following 应用到其他 dLLM RL 方法

### Block 4: ML-Cache（Max Logit 选择性缓存）
- **做法**: 在 MDM 多步推理中，基于 max logit 值识别"稳定"token（高 logit 意味着模型对预测很确定），复用这些 token 的 KV 表征和 logits，跳过其前向计算。三个超参控制: cache_ratio、warmup_ratio（避免早期误估）、refresh_interval（防止误差累积）
- **机制 (WHY it works)**: MDM 的多步去噪中，大量 token 在中后期已经确定（max logit 很高），对它们的重复前向传播是冗余计算。与 AR 模型的 KV-cache 不同（AR 天然支持因果缓存），MDM 的 bidirectional attention 需要特殊策略——ML-Cache 通过"选择性"缓存高确定性 token 的表征来实现近似加速
- **适用条件**: 基于 bidirectional attention 的 MDM 推理；token 确定性在步骤间逐步增加的场景
- **什么时候会 break**: (1) 过高的 cache_ratio 导致误差累积（已确定的 token 可能因上下文变化而需要更新）；(2) warmup 阶段过短导致早期错误缓存传播；(3) 对理解任务（文本序列）的适用性不如图像生成
- **可组合方向**: 与 Stratified Sampling (LaViDa-O) 结合——在空间分散的 unmask 模式下选择性缓存；与 consistency distillation 结合进一步减少步数

### Block 5: 四阶段训练管线（大规模数据驱动）
- **做法**: Stage I 预训练（80M 图文对，渐进分辨率）→ Stage II 中间训练（3M 专用 I2I 数据，引入 controllable/editing/subject-driven）→ Stage III SFT（15M+15M 理解+生成）→ Stage IV Self-GRPO（联合 RL）
- **机制 (WHY it works)**: 四阶段渐进课程让模型先学基础的多模态对齐（Stage I），再学复杂的 I2I 条件生成（Stage II），接着通过大规模指令数据对齐用户意图（Stage III），最后用 RL 自评估精炼质量（Stage IV）。总计 ~110M 训练样本远超 MMaDA（~数百 K-M 级），是 DiMOO 性能领先的主要原因之一
- **适用条件**: 需要大量多样化高质量数据；计算资源充足
- **什么时候会 break**: (1) 数据质量不均匀时，某阶段引入的噪声可能在后续阶段放大；(2) 四阶段的超参调优（学习率/数据配比/阶段长度）复杂度高
- **可组合方向**: 将 Stage II 扩展到更多 I2I 任务（视频编辑、3D 生成）；用 DiMOO 的四阶段管线 + LaViDa-O 的 Elastic-MoT 架构

## Anti-patterns / 已知失败模式
- **纯 VQ token 在低级视觉任务上弱**: super-resolution、dehazing、denoising 表现一般——VQ tokenizer 的 16×16 下采样丢失了像素级细节
- **半自回归理解对采样步数和生成长度高度敏感**: block-wise 生成策略在步数/长度选择不当时性能不稳定（论文自述），虽然 `</answer>` 早停缓解了此问题
- **VQ tokenizer 语义信息缺乏**: aMUSEd-VQ 从重建目标训练，不包含语义信息——论文承认"这对理解任务构成挑战"，需要大规模数据弥补
- **Self-GRPO 保留 KL 正则化**: LaViDa-R1 已发现 KL 在 image token NLL>6 时导致方差过大和训练发散，DiMOO 的 Self-GRPO 仍保留 KL，在更大分辨率/更长序列下的稳定性存疑

## 实验关键发现
- **T2I SOTA**: GenEval 88%（超越 FLUX.1-dev 82%、GPT-4o 84%）；DPG 86.04；UniGenBench 第一名
- **全面超越 MMaDA**: GenEval 88% vs 63%（+25 pp）、DPG 86 vs 70（+16 pp）、UniGenBench 71 vs 41（+30 pp）——同为 LLaDA-based dLLM，差距巨大
- **强理解能力**: MMMU 58.6%（匹配 Uniworld-V1），MMB 84.5%（超越 Janus-Pro 79.2%），SEED 83.1%
- **广泛 I2I 支持**: controllable generation（canny/depth/pose/HED）、style transfer（比 OmniGen 提升 5% text alignment）、subject-driven（DINOv2 +3.97%）、image editing（object add/replace 3.82/3.83 vs OmniGen 3.47/2.94）
- **32× 推理加速**: 相比 Lumina-mGPT 2.0（AR 模型），T2I 生成速度提升 32×
- **ML-Cache 额外 2× 加速**: 无训练推理加速方法，与并行采样正交
- **零样本 inpainting/extrapolation**: masked diffusion 范式天然支持，无需额外微调
- **LLaDA 初始化关键**: 从预训练 dLLM 初始化显著减少训练需求（vs 从头训练）

## Relations (结构化)
<!-- type: extends | alternative_to | combines_with | motivated_by | enables | conflicts_with -->
- `extends` → [[LLaDA]]: 基于 LLaDA-Base 8B 初始化，保持模态无关全共享架构
- `extends` → [[aMUSEd]]: 使用 aMUSEd-VQ tokenizer (codebook 8192, 16×16 下采样)
- `alternative_to` → [[2025-MMaDA]]: 同为 LLaDA-based dLLM 统一模型；DiMOO 在所有可比基准上大幅超越 MMaDA（GenEval +25pp, DPG +16pp）；DiMOO 用更大规模数据（~110M vs ~数 M）和更完整的训练管线
- `alternative_to` → [[2025-LaViDa-O]]: 同为 dLLM 统一模型；DiMOO 用模态无关全共享 vs LaViDa-O 的 Elastic-MoT 非对称分支；DiMOO 纯 VQ token vs LaViDa-O 的 SigLIP+VQ 双路
- `motivated_by` → [[DeepSeek-R1]]: GRPO 算法的原始来源，Self-GRPO 是其 dLLM 适配
- `combines_with` → [[Lumina-mGPT]]: 同一团队前作（AR 统一模型），DiMOO 将其从 AR 转为 dLLM 并实现 32× 加速
- `conflicts_with` → [[2026-LaViDa-R1]]: KL 正则化设计冲突——DiMOO Self-GRPO 保留 KL，LaViDa-R1 发现 image token NLL>6 时 KL 导致训练发散而移除 KL
- `enables` → [[2025-dMLLM-TTS]]: dMLLM-TTS 在 DiMOO 基座上应用 test-time scaling，GenEval 0.78→0.92 (+17.9%)
- `combines_with` → [[2025-Sparse-LaViDa]]: ML-Cache（缓存稳定 token）与 Sparse-LaViDa（截断 mask token）正交互补，可叠加实现更高加速比
- `combines_with` → [[2026-EBPO]]: EBPO 可替换 Self-GRPO 的 baseline 估计；Self-GRPO 自评估 reward 噪声较高，EBPO 方差降低价值更大
- `alternative_to` → [[2026-Beyond-LM]]: 同为大规模统一模型——DiMOO 从 LLaDA 初始化+离散扩散+~110M 数据 vs Beyond-LM 从零训练+连续扩散+MoE；"视觉数据饥渴"发现互相呼应
- `combines_with` → [[2025-SDAR-VL]]: DiMOO 的 ML-Cache（推理加速）与 SDAR-VL 的块状处理（训练稳定性）正交互补
- `alternative_to` → [[2025-KimiK2.5]]: 统一模型路线分歧——DiMOO 离散扩散统一架构 vs K2.5 纯 AR MoE 多模态 + agentic；GRM（外部评估器）vs Self-GRPO（自评估）reward 策略对比
- `combines_with` → [[2025-VTP]]: VTP 多目标语义增强 tokenizer 可替换 DiMOO 的 aMUSEd-VQ（纯重建），解决低级视觉任务语义不足问题，潜在提升 T2I 生成质量上界
- `combines_with` → [[2026-Omni-Diffusion]]: Omni-Diffusion 的三模态扩展验证 masked diffusion 统一架构可 scale 到更多模态；DiMOO 的 ML-Cache 推理加速和 Self-GRPO 自评估可迁移到三模态场景

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- 首次在 dLLM 统一模型中证明**纯 VQ token（无独立视觉编码器）+ 大规模数据**可同时达到 SOTA 生成（GenEval 88%）和 competitive 理解（MMMU 58.6%），为 [[problem-tree#Uni-1]] 提供极简架构方案
- 通过四阶段训练管线（80M+3M+30M）将 dLLM 统一模型的 T2I 质量推至超越 FLUX.1-dev 和 GPT-4o（GenEval），为 [[problem-tree#Uni-3a]] 提供首个「dLLM > 专用连续扩散模型」的实证
- Self-GRPO 验证了「一个 loss 同时提升生成和理解」的联合 RL 方案，为 [[problem-tree#Uni-5]] 提供统一 RL 的新范式
- ML-Cache 提供了 dLLM 独有的无训练推理加速方案（额外 2×），为 [[problem-tree#Diff-1c]] 开辟新方向

### 未解决的问题
- 问题: 纯 VQ token 在低级视觉任务（super-resolution, dehazing, denoising）上表现弱
  - 为什么难: VQ tokenizer 的 16×16 下采样天然丢失像素级细节，这是离散化本身的信息瓶颈
  - 潜在思路: 多尺度 VQ（低分辨率语义 token + 高分辨率细节 token）；混合连续-离散 token 方案
- 问题: Self-GRPO 的 bootstrapping 偏差——模型用自身理解能力评估生成质量，理解偏差会传播到 reward signal
  - 为什么难: 统一模型的生成和理解共享参数，评估偏差与生成偏差耦合，形成正反馈循环
  - 潜在思路: 引入外部 VLM（如 GPT-4V）做 reward 交叉验证；定期用 anchor reward（如 CLIP/EditScore）校准自评估偏差
- 问题: ML-Cache 的超参敏感性——cache_ratio、warmup_ratio、refresh_interval 的最优组合可能是任务/分辨率的函数
  - 为什么难: 三维超参空间搜索成本高；不同分辨率/步数下最优值不同
  - 潜在思路: 自适应 cache_ratio（基于当前步的全局 max logit 分布动态调整）；学习型 cache policy
- 问题: KL 正则化在高分辨率长序列下的稳定性存疑
  - 为什么难: LaViDa-R1 已发现 image token NLL>6 时 KL 导致方差过大和训练发散，DiMOO 保留 KL 但未在更极端条件下验证
  - 潜在思路: 自适应 KL 系数（随序列长度/NLL 动态调整）；或采用 LaViDa-R1 的去 KL 策略
- 问题: 110M 数据规模下的数据质量控制——如何确保大规模训练数据的质量一致性？
  - 为什么难: 80M 图文对来源多样，质量参差不齐；数据筛选在此规模下计算成本高
  - 潜在思路: 基于模型自身的 data quality scoring + curriculum learning；主动学习策略选择高信息量样本

### 对问题树的推进
- 推进了 [[problem-tree#Uni-1]] (强化 🟡): 纯 VQ token 极简架构 + 大规模数据证明「无需独立视觉编码器」也可达 SOTA 生成+competitive 理解，为全共享架构提供最强实证
- 推进了 [[problem-tree#Uni-3a]] 🟡→🟢: GenEval 88% 超越 FLUX.1-dev (82%) 和 GPT-4o (84%)，首次证明 dLLM 在 T2I 指标上可超越专用连续扩散模型
- 推进了 [[problem-tree#Uni-2c]] (补充数据维度): 四阶段 110M 数据管线提供了迄今最详细的 dLLM 统一模型训练 recipe
- 推进了 [[problem-tree#Uni-5]]: Self-GRPO 联合优化生成+理解的 RL 方案，且验证了 Step Trajectory Following 使 1024² RL 训练可行
- 推进了 [[problem-tree#Diff-1c]] 🟡→🟢: ML-Cache 提供了与并行采样正交的无训练加速方案，总计 ~64× 加速（32× 并行 + 2× cache）
- 推进了 [[problem-tree#Tok-2a]]: 纯 aMUSEd-VQ (codebook 8192) 在极大规模数据下可达 MMMU 58.6%，证明 VQ token 的语义不足可被数据弥补
- 新增问题: [Tok-3a] 纯 VQ token 在低级视觉任务上的信息瓶颈——如何在保持离散化优势的同时恢复像素级细节？
- 新增问题: [RL-4a] Self-GRPO 的 bootstrapping 偏差量化——如何测量和纠正自评估偏差对 RL 收敛的影响？
- 新增问题: [Diff-3b] ML-Cache 的自适应策略——能否学习最优 cache policy 而非手动调超参？

## 个人深度评注

### [Critic] 增量贡献评估
| Block | 新颖性 | 理由 |
|-------|--------|------|
| Block 1: 纯 VQ Token 统一架构 | 低中 | 「不用独立视觉编码器」本身不新（SEED-X 等已探索），但在 dLLM 框架下首次做到 SOTA 生成有实证价值 |
| Block 2: `<end-of-line>` 任意分辨率 | 中 | 在 1D RoPE + MDM 框架中处理 2D 结构的实用方案，有设计价值但非概念突破 |
| Block 3: Self-GRPO | 中 | GRPO 适配 dLLM 并非首创（MMaDA UniGRPO 在先），自评估联合优化有新意但有 bootstrapping 偏差问题 |
| Block 4: ML-Cache | 中高 | MDM 独有的无训练推理加速方案，max logit 作为稳定性指标简洁有效，是最具独立价值的贡献 |
| Block 5: 四阶段训练管线 | 低 | 工程贡献为主，各阶段设计合理但无方法论创新 |

### [Critic] 核心判断: 数据 > 架构

DiMOO 相比 MMaDA 的性能领先（GenEval +25pp, DPG +16pp）**主要归因于数据规模差异**（~110M vs ~数 M），而非架构创新。两者都基于 LLaDA-8B + 模态无关全共享架构，核心区别在于:
- DiMOO 80M 预训练 + 30M SFT vs MMaDA 数量级更小的训练数据
- DiMOO 的四阶段渐进课程 vs MMaDA 的两阶段（SFT+RL）

这意味着 DiMOO 的成功更多是**工程 scaling 的胜利**，而非方法论的突破。但这本身是有价值的信号——证明了在当前 dLLM regime 下，数据规模可能比架构选择更重要。

### [Critic] 关键隐含假设
1. **aMUSEd-VQ 语义缺失可被数据弥补**: 假设大规模训练数据足以让 LLM 从 VQ token 中隐式学习视觉语义。MMMU 58.6% 部分验证了这一点，但与使用 SigLIP 的模型（如 LaViDa-O MMMU 45.1，但其训练数据更少）的 fair comparison 缺失
2. **Self-GRPO 的自评估无系统性偏差**: 用同一套参数生成和评估，可能存在自我接受偏差（sycophancy）。entity-relation-value 三元组的自动生成质量直接影响 reward 质量
3. **KL 正则化在 DiMOO 规模下稳定**: 与 LaViDa-R1 去 KL 的设计冲突——DiMOO 在 1024² 分辨率（4096+ tokens）下保留 KL 的稳定性未被充分验证
4. **ML-Cache 的误差累积可控**: 高 cache_ratio 下被缓存 token 的表征可能因上下文变化而过时，refresh_interval 的设定是否足以防止累积误差影响生成质量

### [Connector] DiMOO 在 dLLM 统一模型谱系中的定位
```
路线 A (北大+Princeton: 模态无关全共享)
LLaDA + MAGVIT-v2 → MMaDA (2025-05, NeurIPS) — 方法论开创
                                         ↘
路线 A' (上科大+港中大: 数据工程驱动)
LLaDA + aMUSEd-VQ → Lumina-DiMOO (2025-10) — 工程 scaling 实证

路线 B (Adobe+UCLA: 非对称专用架构)
LaViDa (NeurIPS 2025) → LaViDa-O (2025-09) → LaViDa-R1 (2026-02)
                        方法论创新           RL 方法论创新
```

DiMOO 代表「数据工程驱动」的学派:
- vs MMaDA: 同为全共享 LLaDA 分支，DiMOO 用 50-100× 更多数据实现大幅超越
- vs LaViDa-O: 全共享 vs 非对称分支，纯 VQ vs SigLIP+VQ——是「用数据换简洁性」vs「用架构换数据效率」的 tradeoff
- vs LaViDa-R1: 在 KL 正则化问题上存在直接冲突（DiMOO 保留 vs LaViDa-R1 移除）

### [Connector] 三模型核心设计对比
| 维度 | DiMOO | MMaDA | LaViDa-O |
|---|---|---|---|
| 架构哲学 | 模态无关全共享 | 模态无关全共享 | 非对称 Elastic-MoT |
| 视觉编码器 | 无（纯 VQ） | 无（纯 VQ） | SigLIP + VQ |
| 图像 Tokenizer | aMUSEd-VQ 8192 | MAGVIT-v2 8192 | VQGAN |
| T2I 最高分辨率 | 1024² | 512² | 1024² |
| 训练数据规模 | ~110M 样本 | ~数 M 样本 | 未公开 |
| GenEval | 88% | 63% (84% w/ RL) | 89% (w/ reflection) |
| MMMU | 58.6% | 30.2% | 45.1% |
| 贡献重心 | 数据工程 + 系统设计 | 后训练 RL 方法论 | 架构设计 + 预训练 |
| RL 方法 | Self-GRPO（保留 KL） | UniGRPO（保留 KL） | N/A（LaViDa-R1 做 RL） |

### [Ideator] 潜在研究方向
1. **语义增强 VQ Tokenizer**: 在 aMUSEd-VQ 框架上增加 CLIP contrastive loss 或 knowledge distillation loss（从 SigLIP 蒸馏语义），使 VQ token 同时保留重建质量和语义信息。动机: 解决 [Tok-3a] 纯 VQ 语义不足的根本问题。依赖: Block 1 (DiMOO) + SigLIP。风险: 多目标训练可能导致 codebook collapse。可行性: 中等偏高
2. **ML-Cache + Consistency Distillation**: 将 ML-Cache 的选择性缓存与 consistency distillation（减少采样步数）结合，探索乘法加速效应。动机: ML-Cache 是步间加速，consistency distillation 是减步加速，两者正交。依赖: Block 4 (DiMOO) + Consistency Models。风险: consistency distillation 在 discrete diffusion 上不成熟。可行性: 中等
3. **Self-GRPO 偏差校准**: 引入少量外部 anchor reward（如 EditScore、CLIPScore 在简单 prompt 上的得分）周期性校准 Self-GRPO 的自评估偏差。动机: 解决 [RL-4a] bootstrapping 偏差问题。依赖: Block 3 (DiMOO) + LaViDa-R1 的 EditScore 经验。风险: anchor reward 本身有偏差（P-RL-01）。可行性: 中等偏高
