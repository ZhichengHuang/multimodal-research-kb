---
title: "MMaDA: Multimodal Large Diffusion Language Models"
authors: [Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, Mengdi Wang]
date: 2025-05
venue: NeurIPS
url: "https://arxiv.org/abs/2505.15809"
tags: [diffusion, unified-model, rl, posttraining, pretraining, architecture, alignment, reward-model]
category: unified-model/diffusion-native
level: 3
status: read
importance: high
problem_tree_nodes: [Uni-1, Uni-2a, Uni-5, RL-2, RL-3, Diff-1b]
aliases: [MMaDA, UniGRPO]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结
提出 MMaDA，一个基于离散扩散（masked diffusion）的多模态统一基础模型，通过 modality-agnostic 架构、混合长 CoT 微调和 UniGRPO（首个面向扩散模型的统一 RL 算法），在 8B 规模同时实现文本推理、多模态理解和图像生成的 SOTA 或可比性能。

## 核心 Insight
扩散模型可以通过统一的 mask-predict 范式（离散扩散）消除模态间的架构差异，而 RL 后训练（UniGRPO）可以直接适配非自回归模型——关键在于结构化噪声策略和高效 log-likelihood 近似，避免了 Monte Carlo 采样的高计算开销。混合长 CoT 微调提供了 RL 冷启动的有效途径。

## 与已有工作的关系
- **继承自**: [[LLaDA]]（离散扩散语言模型，提供预训练权重）、[[Show-o]]（图像 tokenizer MAGVIT-v2）
- **对比**: [[2026-LaViDa-R1]]（同为 dLLM RL，但方法设计不同）、[[d1]]（dLLM RL 先驱，仅单任务）、[[Janus]]（解耦编码统一模型）
- **互补**: [[Transfusion]]（AR+Diffusion 混合路线 vs MMaDA 的纯 Diffusion 路线）

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 整体架构
- **骨干**: 基于 [[LLaDA]]-8B-Instruct 的 masked diffusion transformer
- **图像 tokenizer**: MAGVIT-v2（来自 [[Show-o]]），downsampling factor=16, codebook size=8192, 512×512 图像 → 32×32=1024 离散 token
- **文本 tokenizer**: LLaDA tokenizer
- **统一目标**: 所有模态共享同一个 cross-entropy mask prediction loss

### 统一概率形式
前向过程: 以概率 t 将 token 替换为 [MASK]
训练目标: ℒ_unify(θ) = -𝔼[t,x₀,xₜ][(1/t)∑ I[xᵢₜ=[MASK]] log pθ(x⁰ᵢ|xₜ)]
关键: 跨模态对齐噪声腐蚀和语义恢复过程，促进跨模态交互

### 三阶段训练
| 阶段 | 步数 | 数据 | 目标 |
|------|------|------|------|
| Stage 1 | 200K+400K | RefinedWeb, ImageNet→diverse image-text | 基础预训练 |
| Stage 2 | 50K | Instruction + 混合长 CoT 推理数据 | Mixed Long-CoT SFT |
| Stage 3 | 50K | GSM8K, GeoQA, CLEVR | UniGRPO RL |

硬件: 64×A100 80GB GPUs, batch size 1280

### 推理采样
- **文本**: 半自回归（block-wise left-to-right），block size=64, 每步 unmask 2 个最低置信度 token
- **图像**: 并行非自回归，cosine schedule, 50 步去噪, CFG scale=3.5

## Building Blocks（可复用组件）

### Block 1: Modality-Agnostic Masked Diffusion 架构
- **做法**: 将文本和图像统一为离散 token 序列，共享同一个 mask-predict 目标函数。无需模态特定组件（如 AR 的 causal mask 或 diffusion 的 UNet/DiT）
- **机制 (WHY it works)**: 离散扩散（masked diffusion）天然与 LLM 的 token 框架对齐——前向过程（随机 mask）和反向过程（predict masked tokens）在文本和离散化图像上完全一致。统一的噪声-恢复过程使跨模态表示学习成为隐式对齐
- **适用条件**: 需要高质量的视觉离散 tokenizer（如 MAGVIT-v2）；图像分辨率受 tokenizer 限制（目前 512×512）
- **什么时候会 break**: (1) 视觉 tokenizer 信息损失大时，生成质量受限于 tokenizer 上界；(2) 高分辨率场景下 token 数爆炸（1024→4096+）使训练成本剧增；(3) 文本的 next-token prediction 和图像的全局去噪在语义粒度上不匹配
- **可组合方向**: 与更强的视觉 tokenizer（BSQ/LFQ）组合提升生成上界；扩展到视频 token 序列

### Block 2: Mixed Long Chain-of-Thought 微调（冷启动策略）
- **做法**: 为多种任务构造统一 CoT 格式 `|<special_token>|<reasoning_process>|<special_token>|<result>`，在 SFT 阶段混合文本推理（数学）、多模态推理（VQA）和图像生成（世界知识 T2I）的 CoT 数据。训练时保留原始 prompt 不 mask，仅对 result 部分做 masked diffusion
- **机制 (WHY it works)**: 统一 CoT 格式跨模态对齐推理范式——让模型学会"先推理再输出"的通用模式。混合训练促进跨模态知识迁移，特别是让图像生成任务也能利用文本推理能力。冷启动数据让 RL 阶段有合理的 policy 起点
- **适用条件**: 需要高质量的 CoT 推理数据；需要 LLM/VLM 辅助生成和验证推理链
- **什么时候会 break**: (1) CoT 数据质量不高时引入推理噪声；(2) 图像生成的"推理"本质上是 factual retrieval 而非逻辑推导，CoT 的增益可能有限；(3) 混合比例不当导致某些任务欠训练
- **可组合方向**: 与 [[2026-LaViDa-R1]] 的 answer-forcing 结合（CoT 冷启动 + answer-forcing 热启动）；扩展到视频/3D 生成的 CoT

### Block 3: UniGRPO（面向扩散模型的统一 RL 算法）
- **做法**: 将 GRPO 适配到非自回归扩散模型，三个关键设计:
  1. **结构化噪声策略**: 为每个 response 随机采样 mask ratio p∈[0,1]，跨梯度步骤变化随机种子，暴露模型于从全 mask 到近乎全显的各种去噪阶段
  2. **高效 log-likelihood 近似**: 在 masked 区域计算 per-token log prob，序列级 log-likelihood = masked tokens 的平均 log prob，避免 Monte Carlo 采样
  3. **PPO-style 目标**: Token 级 reward ratio + clipping + KL penalty

  目标函数: 𝒥_UniGRPO = 𝔼[min(r'·Â, clip(r',1-ε,1+ε)·Â) - β·D_KL]

- **机制 (WHY it works)**: 结构化噪声策略有效近似了 Monte Carlo 平均（LLaDA 需 128 次），同时保留了多步去噪的动态特性。与 d1 的固定 mask ratio 相比，随机采样增加了噪声多样性，更好地覆盖了 diffusion 过程的各个阶段
- **适用条件**: 适用于任何 masked diffusion model；需要 verifiable 或可评分的 reward signal
- **什么时候会 break**: (1) 当 mask ratio 集中在极端值（接近 0 或 1）时，log-likelihood 近似偏差大；(2) 非自回归特性使 token 间的 credit assignment 困难——不清楚是哪些 token 贡献了好/坏结果；(3) 计算开销仍然较大（需要为每个 sample 采多个 mask ratio）
- **可组合方向**: 与 [[2026-LaViDa-R1]] 的 complementary masking (w=1) 结合改进 likelihood estimation；与 process reward model 结合做更细粒度的 credit assignment

### Block 4: Diversified Reward Modeling（多任务奖励设计）
- **做法**: 为三类任务设计不同 reward:
  - 文本推理: correctness reward (2.0) + format reward (0.5, 要求 `<think>` 格式)
  - 多模态推理: correctness + format + CLIP reward (0.1·CLIP score)
  - T2I 生成: CLIP reward (0.1) + Image Reward (0.1)
- **机制 (WHY it works)**: 多维 reward 信号覆盖不同质量维度——正确性、格式规范、视觉-文本对齐、人类偏好。scale factor (0.1) 平衡不同 reward 的量级
- **适用条件**: 需要为每个任务有合适的 reward 函数；不同 reward 的量级需要手动对齐
- **什么时候会 break**: (1) CLIP reward 不具备 compositional reasoning 能力（与 [[2026-LaViDa-R1]] 发现的问题一致）；(2) reward hacking——模型可能过度优化简单 reward 而忽视真正的质量；(3) 手工设定的 reward scale 不一定泛化到不同数据分布
- **可组合方向**: 引入 VLM-as-judge 作为更强的 reward signal；使用 reward ensemble 减少 hacking 风险

## Anti-patterns / 已知失败模式
- **d1 的固定 mask + question masking 策略**: 减少噪声多样性，忽略多步去噪动态，不如 UniGRPO 的结构化随机策略
- **LLaDA 的 Monte Carlo 采样 (~128 ratios)**: 计算开销过大，不适合 on-policy RL 的频繁策略更新
- **完全随机 masking (baseline)**: 导致 reward 波动大、收敛慢（Figure 4），不如结构化策略的均匀时间步采样
- **位置信息（Position）在 GenEval 上表现较弱 (0.20)**: 说明 masked diffusion 在空间关系推理上仍有局限
- **CLIP-based reward 的局限**: 不支持 compositional reasoning，无法评估需要世界知识的 T2I prompt

## 实验关键发现
- **首个 diffusion-native 统一模型在三项任务均达 competitive 性能**: 文本推理接近 Qwen2-7B，多模态理解超越 Show-o/SEED-X，T2I 超越 SDXL/Janus
- **Mixed Long-CoT 带来巨大提升**: GSM8K 从 17.4→65.2 (+47.8), CLIP Score 从 23.1→29.4
- **UniGRPO 在 CoT 基础上进一步提升**: GSM8K 65.2→73.4, MATH500 26.5→36.0, ImageReward 0.84→1.15
- **跨模态协同效应**: Stage 2 训练中所有任务的指标同步提升（Figure 6），表明统一训练存在互利效应
- **采样效率**: 图像生成仅需 50 步（vs 1024）即可保持强性能；文本/理解任务可用 1/2-1/4 步数
- **WISE 文化知识基准大幅领先 (0.67 vs Janus 0.16)**: 得益于世界知识 CoT 数据 + RL 优化
- **天然支持 inpainting**: masked diffusion 范式无需额外训练即可做文本/图像 inpainting

## Relations (结构化)
<!-- type: extends | alternative_to | combines_with | motivated_by | enables | conflicts_with -->
- `extends` → [[LLaDA]]: 基于 LLaDA-8B-Instruct 预训练权重，扩展到多模态统一模型
- `extends` → [[Show-o]]: 使用 Show-o 的 MAGVIT-v2 图像 tokenizer
- `alternative_to` → [[2026-LaViDa-R1]]: 同为多模态 dLLM RL；UniGRPO 用结构化随机噪声+显式 KL vs LaViDa-R1 用 complementary masking w=1+SFT 正则化+answer-forcing；MMaDA 早 9 个月，是 LaViDa-R1 的对比基线
- `alternative_to` → [[d1]]: 同为 dLLM RL，但 UniGRPO 支持多模态多任务 vs d1 仅单任务数学
- `alternative_to` → [[Janus]]: 统一模型路线之争——纯 diffusion vs 解耦编码
- `alternative_to` → [[Transfusion]]: 统一模型路线之争——纯 diffusion vs AR+Diffusion 混合
- `motivated_by` → [[DeepSeek-R1]]: GRPO 算法的原始来源，CoT 推理范式
- `combines_with` → [[MAGVIT-v2]]: 使用其 tokenizer 作为视觉离散化方案
- `extends` → [[2025-MMaDA-Parallel]]: MMaDA-Parallel 是 MMaDA 的直接改进版本，从 sequential 推理-生成转为 parallel 架构
- `enables` → [[2025-dMLLM-TTS]]: dMLLM-TTS 在 MMaDA 上应用 test-time scaling，GenEval 0.51→0.66 (+29.4%)
- `alternative_to` → [[2025-SDAR-VL]]: 同为 dLLM 多模态模型——标准 masked diffusion+UniGRPO RL vs 块状扩散+ABNS/EMRS/PBNC 训练稳定性优化
- `combines_with` → [[2026-EBPO]]: EBPO 的 shrinkage baseline 可直接替换 UniGRPO 中的 GRPO baseline 估计，解决"训练信号消失问题未解决"
- `alternative_to` → [[2026-Beyond-LM]]: 统一模型路线之争——LLaDA 初始化纯离散扩散 + 模态无关全共享 vs 从零训练 AR+连续扩散混合 + modality-specific FFN + MoE
- `alternative_to` → [[2025-KimiK2.5]]: 架构路线对立——dLLM (masked diffusion) vs AR-based MoE (1.04T, 自回归)；同样做跨模态 RL 但实现不同（UniGRPO vs GRM+Toggle RL）；K2.5 在 AR 上独立验证跨模态协同 (P-Uni-01)

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- 首次证明纯 diffusion（masked diffusion）路线可以在 8B 规模同时实现文本推理、多模态理解和 T2I 生成的 competitive 性能，为 [[problem-tree#Uni-1a]] "Diffusion 原生"路线提供了最强的可行性证据
- 将 GRPO RL 算法从 AR 模型适配到非自回归扩散模型（UniGRPO），解决了 token 级 log-likelihood 估计、mask ratio 采样、策略优化三个核心挑战
- 通过 Mixed Long-CoT 微调建立了 dLLM 统一模型的后训练完整链路（预训练→SFT→RL）

### 未解决的问题
- 问题: dLLM 的空间关系推理能力极弱（GenEval Position 0.20）
  - 为什么难: MAGVIT-v2 的离散化在重建目标下训练，不显式建模空间关系；mask-predict 目标也无空间推理归纳偏置。这是架构级缺陷，RL 后训练无法修复
  - 潜在思路: 引入显式 2D 位置先验（如相对位置编码）到 tokenizer 或模型架构；训练空间关系感知的 visual tokenizer
- 问题: UniGRPO 的 log-likelihood 估计在高熵 image token 分布下方差偏大
  - 为什么难: i.i.d. 随机 mask 在单次估计中可能遗漏关键 token，高熵分布放大了估计方差；保留 KL penalty 在 image token 下可能导致训练不稳定（LaViDa-R1 已验证）
  - 潜在思路: 采用 [[2026-LaViDa-R1]] 的 complementary masking (w=1)；或用 importance sampling / control variate 降低方差
- 问题: CoT 在 dLLM 并行去噪框架中的因果有效性存疑
  - 为什么难: dLLM 并行预测所有 masked token，reasoning → result 的因果关系在训练层面不严格成立。模型可能学到的是 "answer 和 reasoning template 的 co-occurrence 模式" 而非因果推导
  - 潜在思路: 需要 faithfulness 实验（扰乱 reasoning 是否影响 result 质量）来验证；或引入 sequential generation 的归纳偏置（如 block-wise CoT 生成）
- 问题: 训练信号消失问题未解决——当 RL 阶段遇到"所有 rollout 都失败"的困难问题时，无正向梯度信号
  - 为什么难: GRPO 的 advantage 估计需要 group 内有正向样本；MMaDA 无对应机制
  - 潜在思路: 引入 [[2026-LaViDa-R1]] 的 answer-forcing（利用 dLLM inpainting 能力），或设计 curriculum RL（逐步增加任务难度）

### 对问题树的推进
- 推进了 [[problem-tree#Diff-1b 🔴 连续扩散 vs 离散扩散 (Masked Diffusion)|Diff-1b]] → 🟡: 与 LaViDa-R1 合计提供充分证据，离散 masked diffusion 是 dLLM 统一模型的主流可行路线
- 推进了 [[problem-tree#Uni-1 🔴 统一架构路线之争|Uni-1a]]: 为 "Diffusion 原生" 路线提供了首个 NeurIPS 级别的完整验证，证伪风险明显降低
- 推进了 [[problem-tree#Uni-2 🔴 理解与生成的能力关系|Uni-2a]] → 🟡: Figure 6 跨模态协同效应是 "共享参数下能力不冲突甚至互利" 的直接实验证据
- 推进了 [[problem-tree#Uni-2 🔴 理解与生成的能力关系|Uni-2c]]: 三阶段课程学习（预训练→Mixed CoT SFT→RL）是目前最完整的 dLLM 统一模型训练路径
- 推进了 [[problem-tree#Uni-5 🔴 统一模型的 Post-training 和 RL|Uni-5a]] → 🟡: 提供了首个 diffusion-native 全链路后训练方案（与 LaViDa-R1 合计）
- 推进了 [[problem-tree#Uni-5 🔴 统一模型的 Post-training 和 RL|Uni-5b]] → 🟡: Diversified Reward 设计提供了具体实践，但同时暴露了 CLIP reward 对 compositional reasoning 无能为力
- 推进了 [[problem-tree#Uni-5 🔴 统一模型的 Post-training 和 RL|Uni-5c]] → 🟡: UniGRPO 同时提升 GSM8K（推理）和 ImageReward（生成），初步验证 RL 促进理解-生成协同
- 推进了 [[problem-tree#RL-2 🔴 GRPO/PPO 如何适配多模态？|RL-2a]]: UniGRPO 的结构化随机策略替代 Monte Carlo 128-sample，显著降低采样成本
- 推进了 [[problem-tree#RL-3 🔴 RL 能提升多模态推理能力吗？|RL-3a]]: 在 GeoQA/CLEVR 上用 correctness+CLIP reward，验证 verifiable reward 对 dLLM 可行
- 新增问题: [RL-2c] 高熵 token 分布下的 RL 正则化——KL 在 image token 下失效（LaViDa-R1 验证），SFT 替代有效但理论不完备
- 新增问题: [RL-1d] T2I Compositional Reasoning Reward Model 空白——CLIP 和 VLM-as-judge 均不可靠
- 新增问题: [Tok-2d] MAGVIT-v2 的生成质量上界是否是当前 dLLM 统一模型的主要瓶颈？

## 个人深度评注
<!-- 留待用户审阅后补充 -->

### [Critic] 增量贡献评估
| Block | 新颖性 | 理由 |
|-------|--------|------|
| Block 1: 统一 Masked Diffusion 架构 | 中 | LLaDA + MAGVIT-v2 的工程集成，概念新颖性有限，但系统级验证价值高 |
| Block 2: Mixed Long CoT SFT | 低中 | 与 LaViDa-R1 Stage 1 SFT 同质；GSM8K +47.8 的跳跃更像 "灾难遗忘恢复" 而非 "能力涌现" |
| Block 3: UniGRPO | 中 | 随机 mask ratio 比 d1 的固定 mask 更合理；但 log-likelihood 估计的理论性不如 LaViDa-R1 的 complementary masking |
| Block 4: Diversified Reward | 低 | 标准多 reward 设计，CLIP reward 的局限在两篇论文中都已认识到 |

### [Critic] 关键隐含假设
1. **Tokenizer 语义对齐**: MAGVIT-v2 的 codebook 从重建目标而非语义相似性训练，与语言 token 的对齐方式根本不同
2. **噪声级别等价**: 统一 mask ratio t 等同对待文本和图像 token，但两者信息密度截然不同——mask 掉文本的 "not" 完全改变语义，mask 掉图像局部 patch 仅造成局部缺失
3. **CoT 因果有效性**: dLLM 并行预测使 reasoning→result 的因果关系在训练层面不严格成立，模型可能学到联合分布模式而非因果推导
4. **KL 稳定性**: 保留 KL penalty 在 image-heavy RL 场景下可能不稳定（LaViDa-R1 已证明），MMaDA 通过限制 RL 为文本推理任务间接回避了此问题

### [Critic] 新识别的失败模式
- **Token 数量不对称**: 1024 视觉 token vs ~100-300 文本 token，预训练实际以图像重建为主导，压制了文本推理能力（GSM8K Stage 1 仅 17.4 间接证实）
- **CoT 训练的 exposure bias**: 训练时 reasoning token 完整可见（不 mask），推理时需自行生成（mask→unmask），造成 train-inference gap
- **非自回归 credit assignment 困难**: 序列级 correctness reward 均匀分配到所有 token，关键推理步骤的 token 和格式 token 等权，信号被稀释
- **On-policy 偏差放大**: dLLM 的 50 步去噪使 policy 参数微小变化在多步去噪链中被放大，off-policy 偏差可能比 AR 更严重

### [Connector] MMaDA vs LaViDa-R1 核心设计对比
| 维度 | MMaDA / UniGRPO | LaViDa-R1 |
|------|-----------------|-----------|
| **Likelihood 估计** | 结构化随机 mask ratio, 平均 log prob (i.i.d. 覆盖) | Complementary masking w=1, 互补 mask 保证全 token 覆盖 |
| **RL 正则化** | 保留 KL penalty (β·D_KL) | 去掉 KL, SFT 隐式正则化 |
| **训练信号消失** | 未解决 | Answer Forcing: dLLM inpainting 反向填充推理链 |
| **CoT 冷启动** | 统一 CoT 格式 (含 T2I 世界知识 CoT), 50K steps | 混合 SFT + CoT 推理数据 7:3, 100K steps |
| **dLLM 特性利用** | 利用 multi-step denoising dynamics | 利用 bidirectional inpainting (AR 无法实现) |
| **时序** | 2025-05 (先发, 系统验证) | 2026-02 (基于 MMaDA 经验做针对性改进) |

### [Connector] 技术谱系定位
```
LLaDA (masked diffusion LM, 基础)
  ↓
d1 (首个 dLLM RL, 单任务, 固定 mask)
  ↓
MMaDA (NeurIPS 2025, 多模态统一 + UniGRPO)  ← 本文
  ↓
LaViDa → LaViDa-O → LaViDa-R1 (2026-02, 改进 RL + answer-forcing + tree search)
```
MMaDA 和 LaViDa-R1 共同构成了 dLLM 多模态 RL 从 "proof of concept" 到 "principled framework" 的演进。

### [Ideator] 潜在研究方向
1. **dLLM Process Reward Model + MCTS**: 训练评估 diffusion 中间状态质量的 PRM，将 LaViDa-R1 的 Tree Search 升级为 MCTS。动机: UniGRPO 等权对待所有中间状态，LaViDa-R1 的 step 8/64 分支是经验选择。风险: 中间状态语义结构不明确，PRM 监督信号难以构造
2. **dLLM 自评估 Reward for T2I**: 利用统一模型自身的理解能力评估生成图像质量（generation-and-verification 循环），bootstrapped reward。动机: CLIP-based reward 不支持 compositional reasoning。风险: 循环偏差、reward hacking
3. **基于 Inpainting 的多轮迭代 Image Editing RL**: 将 dLLM 的 inpainting 能力与多轮编辑结合，构建 agentic editing 循环。动机: LaViDa-R1 已证明 RL 对单步 editing 有效，dLLM 天然支持 inpainting。风险: 多轮 credit assignment 更困难
