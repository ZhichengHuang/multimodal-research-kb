---
title: "Stabilizing Reinforcement Learning for Diffusion Language Models"
authors: [Jianyuan Zhong, Kaibo Wang, Ding Ding, Zijin Feng, Haoli Bai, Yang Xiang, Jiacheng Sun, Qiang Xu]
date: 2026-03
venue: arxiv
url: "https://arxiv.org/abs/2603.06743"
tags: [rl, dllm, training-stability, grpo, policy-gradient]
category: rl/training-stability
level: 2
status: read
importance: high
problem_tree_nodes: [RL-2a, RL-2c, RL-2g, RL-3a, RL-3c]
aliases: [StableDRL]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结

StableDRL 识别出 GRPO 应用于 dLLM 时因噪声重要性比率导致的自强化不稳定循环（梯度尖峰 → 策略漂移 → 比率方差增大 → 更多尖峰），通过无条件裁剪（suppressing individual outliers）和自归一化（constraining group-level updates within convex hull）两个简洁修改打破该循环，首次实现 dLLM 的稳定全参数 RL 训练超过 1000 步，在 LLaDA-8B 和 SDAR-8B 上均达到 SOTA。

## 核心 Insight

dLLM RL 的训练不稳定不仅源于重要性比率的估计噪声（现有工作关注点），更来自 GRPO 公式本身对噪声比率的脆弱性。标准 GRPO 的两个设计在 AR 模型中安全但在 dLLM 中致命：(1) **条件裁剪**（A<0 时允许大 ρ 不裁剪以加速回归信赖域）在 dLLM 中被模型无关的估计噪声触发，产生梯度尖峰；(2) **固定 group-size 归一化**（除以 G）在高方差估计下使梯度幅度剧烈波动。这两者共同形成自强化循环：噪声 → 尖峰 → 策略漂移 → 更大噪声。StableDRL 的修复切中了 GRPO 公式层面的问题，与具体的 likelihood 估计方法正交。

## 与已有工作的关系
- **继承自**: [[LLaDA]]（基座模型 LLaDA-8B-Instruct）, [[2025-SDAR-VL]]（SDAR-8B 块状扩散基座）, [[2025-SPG]]（block-wise masking 作为 score surrogate）, [[DeepSeek-R1]]（GRPO 框架）
- **对比**: [[2025-SPG]]（SPG 改善 likelihood 估计质量 vs StableDRL 改善 GRPO 更新公式——前者修"数据"偏差，后者修"优化"不稳定）, [[ESPO]]（直接使用标准 GRPO，在 dLLM 上训练不稳定）
- **互补**: [[2026-EBPO]]（改进 advantage baseline 估计 vs StableDRL 改进梯度更新公式——作用于 RL 管线不同环节，正交可叠加）, [[2026-LaViDa-R1]]（answer-forcing 解决训练信号消失 vs StableDRL 解决梯度尖峰不稳定——两个不同的训练病理）

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 问题诊断：自强化不稳定循环

GRPO 在 dLLM 上的不稳定源于三阶段循环：

**Stage 1 — 重要性比率的噪声**: dLLM 的 log-likelihood 不可解析，需通过 ELBO 等 MC 估计。估计误差 η(x) 经 exp(·) 映射产生长尾分布，ρ̂(x) = exp(ΔL(x)) · exp(Δη(x))，单个 rollout 的 ρ̂ 可爆炸到 10⁵ 量级。

**Stage 2 — 梯度尖峰（两种机制）**:
- *个体异常*: GRPO 的条件裁剪在 A<0, ρ̂>1+ε 时不裁剪（设计初衷：加速回归信赖域）。但 dLLM 中大 ρ̂ 由噪声而非真实策略对齐驱动，导致不被裁剪的巨大梯度
- *群体异常*: 固定 group-size 归一化（÷G），当 group 内 ρ̂ 同时偏大或偏小时梯度幅度剧烈波动

**Stage 3 — 策略漂移**: 梯度尖峰导致策略 πθ 剧变，ΔL(x) 增大，进一步放大后续 ρ̂ 的方差，循环加剧

### StableDRL 核心设计

**无条件裁剪 (Unconditional Clipping)**:
- 将 GRPO 的条件裁剪（仅在 ρ 偏离信赖域且 A 同号时裁剪）替换为无条件裁剪——ρ̂ 始终限制在 [1-ε, 1+ε]，无论 advantage 符号
- 消除个体异常导致的梯度尖峰
- 但单独使用仍不够——裁剪后梯度在上下界之间高频震荡（boundary-hitting regime）

**自归一化 (Self-Normalization)**:
- 将固定归一化 1/G 替换为 1/Σ clip(ρ̂ᵢ)——按裁剪后权重的总和归一化
- 将梯度更新约束在 per-sample 梯度的凸包内：||∇θ J|| ≤ max||gᵢ|| ≤ B
- 解耦更新幅度与 group-level 权重波动

**更新公式**:
```
∇θ J_Ours = E[ (1/Σᵢ clip(ρ̂ᵢ)) · Σⱼ clip(ρ̂ⱼ)·Aⱼ·gⱼ ]
```

**数值稳定实现**: 在 log-space 计算——clip-then-softmax（裁剪 log-ratio → LogSumExp 归一化），避免 exp 溢出/下溢

### 块状扩散扩展：阶梯注意力 (Staircase Attention)

**问题**: 块状扩散的 RL 面临效率-泄露两难——逐块计算 ELBO 需 O(K) 前向传播，标准并行注意力导致目标块 attend 到自身的 ground truth

**阶梯注意力方案**: 双流输入（干净上下文 + 噪声目标），组合注意力掩码实现 O(1) 无泄露 ELBO 评估：
- 干净上下文流：标准因果掩码
- 目标 → 上下文：块下三角掩码（block k 只 attend block 1..k-1 的干净历史）
- 目标自注意力：块对角掩码（块内并行去噪，块间隔离）

### 实现细节
- 基座: LLaDA-8B-Instruct (full-attention) + SDAR-8B (block diffusion)
- Score surrogate: 使用 SPG 的 block-wise masking 作为 score function 代理
- ε = 5（默认裁剪阈值，较宽松以保留有效信号）
- lr = 1e-6, AdamW
- 全参数 RL（非 LoRA），训练超 1000 步

## Building Blocks（可复用组件）

### Block 1: 无条件裁剪 (Unconditional Clipping)
- **做法**: 将 GRPO/PPO 的条件裁剪替换为无条件裁剪——importance ratio ρ̂ 始终限制在 [1-ε, 1+ε]，不论 advantage 符号
- **机制 (WHY it works)**: 标准 GRPO 在 A<0, ρ>1+ε 时不裁剪，设计初衷是"加速策略回归信赖域"。在 AR 模型中有效，因为大 ρ 真实反映策略偏移。但 dLLM 的 ρ 来自 MC 估计，大 ρ 可能纯粹由噪声驱动（模型无关的 Δη），条件裁剪被噪声"欺骗"而放行异常梯度。无条件裁剪消除此漏洞，确保梯度严格有界
- **适用条件**: 任何使用 GRPO/PPO 风格裁剪且 importance ratio 含噪声的场景（dLLM RL 是典型场景）
- **什么时候会 break**: (1) 单独使用时，虽然消除了个体尖峰，但梯度在裁剪上下界之间高频震荡（boundary-hitting regime），仍可扰乱 AdamW 动量——必须与自归一化配合；(2) ε 过紧（如 ε=1）限制学习速度，过松（如 ε=1000）退化为无裁剪
- **可组合方向**: 必须与自归一化配合使用（消除 group-level 波动）；与 SPG/ESPO 等 score surrogate 正交（StableDRL 修改优化公式，不依赖特定 likelihood 估计方法）

### Block 2: 自归一化 (Self-Normalization)
- **做法**: 将 GRPO 的固定 group-size 归一化（÷G）替换为按裁剪后权重总和归一化（÷Σ clip(ρ̂ᵢ)），等价于 softmax-style 加权
- **机制 (WHY it works)**: 固定归一化下，当 group 内所有 ρ̂ 同时偏大（噪声相关），梯度幅度被放大 ~(1+ε) 倍；同时偏小则被压缩 ~(1-ε) 倍。这种 group-level 波动频率高且幅度大，破坏 AdamW 的一阶/二阶矩估计，导致次优更新方向。自归一化显式除去随机 group-scale 因子，将更新约束在 per-sample 梯度的凸包内（||∇|| ≤ B），结构性地消除 group-level 随机性
- **适用条件**: 与无条件裁剪配合使用（确保 w_ij > 0 以构成合法凸组合）；group size G ≥ 4
- **什么时候会 break**: (1) 无裁剪时，单个噪声异常值的权重 → 1（主导凸组合），退化为单样本更新；(2) 所有 ρ̂ 相同时（如全 on-policy），退化为标准 1/G 归一化，无额外收益
- **可组合方向**: 与 EBPO shrinkage baseline 正交（EBPO 改进 advantage 估计，自归一化改进梯度聚合）；可推广到任何需要聚合多个噪声估计的场景

### Block 3: 阶梯注意力 (Staircase Attention for Block Diffusion)
- **做法**: 双流输入（干净序列作为上下文 + 噪声序列作为目标），设计组合注意力掩码实现 O(1) 无信息泄露的 ELBO 评估。掩码由三部分组成：块下三角（目标 attend 历史干净上下文）+ 块对角（块内并行去噪）+ 上下文因果掩码
- **机制 (WHY it works)**: 块状扩散要求每个块严格条件于其干净历史进行去噪。朴素逐块实现需 O(K) 前向传播（K=块数），标准并行注意力则泄露当前块的 ground truth。阶梯注意力通过精心设计的掩码几何在单次前向传播中满足 ELBO 条件独立性要求，实现计算效率和数学正确性的统一
- **适用条件**: 块状扩散模型（如 SDAR、Fast-dLLM、DiffusionVL）的 RL 训练
- **什么时候会 break**: (1) 双流输入使序列长度翻倍（2n），显存需求增加；(2) 块数很少（K=2-3）时 O(K) 逐块实现可能更简单高效
- **可组合方向**: 可与 SDAR-VL 的 ABNS/EMRS/PBNC 训练稳定性技术叠加（StableDRL 解决 RL 层面的不稳定，SDAR-VL 解决预训练层面的不稳定）

## Anti-patterns / 已知失败模式
- **GRPO 条件裁剪在 dLLM 中的致命缺陷**: 条件裁剪（A<0 时允许大 ρ 通过）在 dLLM 中被估计噪声"欺骗"，产生梯度尖峰。论文通过理论和实验证明这是 reward collapse 的第一推动力
- **纯裁剪导致 boundary-hitting regime**: 无条件裁剪消除个体尖峰，但梯度频繁触碰裁剪边界，高频震荡破坏 AdamW 动量历史——这是"窄信赖域不稳定、宽信赖域尖峰"的两难（Figure 7: ε=100/1000 仍崩溃）
- **SPG 的 off-policy 偏差**: SPG 复用 rollout 而不做 importance sampling 校正（隐式假设 ρ=1），避免了权重爆炸但累积 off-policy 偏差。在 stress test 中 SPG 在 Normal 和 Exploding 条件下均崩溃（Figure 5），说明 off-policy 偏差比重要性比率噪声更致命
- **LoRA / Early Stopping 是"掩盖"而非"解决"不稳定**: ESPO 和 SPG 通过 LoRA 或提前停止约束不稳定，限制了模型利用全部参数容量的能力。StableDRL 直接全参数训练，better unlocking reasoning capability

## 实验关键发现
- **Full-attention dLLM (LLaDA-8B-Instruct) SOTA**:
  - GSM8K 84.2% (avg over 128/256/512), MATH500 41.8%, Countdown 83.5%, Sudoku 91.5%
  - 超越 ESPO 和 SPG（Countdown +13.7% over SPG at 256 tokens）
  - 首次实现全参数 RL 训练 >1000 步不崩溃

- **Block diffusion (SDAR-8B) SOTA**:
  - AIME'24 16.7% (Static), 超越 Qwen3-8B (10.0%) 和 Trado (13.3%)
  - StableDRL 在 dynamic sampling 下仍保持鲁棒（13.3%），Trado 下降到 11.0%

- **Stress Test 验证**:
  - Exploding Weight 测试中 StableDRL 保持稳定单调改善，ESPO 立即崩溃，SPG 在两种条件下均崩溃
  - 验证了 StableDRL 对重要性比率噪声的结构性鲁棒

- **消融关键发现**:
  - 去掉自归一化: 梯度幅度振荡导致 AdamW 动量损坏，最终崩溃
  - 去掉裁剪: 噪声异常值主导凸组合，快速崩溃
  - ε 敏感度: ε∈{1,5} 稳定（ε=5 更优），ε∈{100,1000} 崩溃——信赖域需要足够紧

- **长度泛化**: 训练在 seq_len=256 下进行，但 128/256/512 三个长度上性能一致——全参数 RL 比 LoRA 提供更好的长度泛化

## Relations (结构化)
- `extends` → [[DeepSeek-R1]]: 直接修改 GRPO 的更新公式——将条件裁剪替换为无条件裁剪，将固定 group-size 归一化替换为自归一化，针对 dLLM 的噪声 importance ratio 场景定制
- `extends` → [[2025-SPG]]: 使用 SPG 的 block-wise masking 作为 score surrogate，但修正了 SPG 的 off-policy 偏差（SPG 隐式假设 ρ=1，StableDRL 显式处理 ρ 的噪声）
- `alternative_to` → [[ESPO]]: 两者都基于 GRPO 做 dLLM RL，ESPO 用标准 GRPO 公式 + LoRA 约束不稳定，StableDRL 修改 GRPO 公式 + 全参数训练。StableDRL 在所有基准上超越 ESPO
- `alternative_to` → [[2025-SPG]]: SPG 改善 likelihood 估计的偏差（ELBO→三明治 bounds），StableDRL 改善 GRPO 更新公式对噪声的鲁棒性。两者改善 dLLM RL 的不同维度——SPG 修"输入质量"，StableDRL 修"公式鲁棒性"。SPG 在 stress test 中因 off-policy 偏差崩溃，StableDRL 稳定
- `combines_with` → [[2026-EBPO]]: EBPO 改进 advantage baseline 估计（shrinkage），StableDRL 改进梯度更新公式（无条件裁剪 + 自归一化）——分别作用于 RL 管线的不同环节（advantage 计算 vs 梯度聚合），正交可叠加
- `combines_with` → [[2026-LaViDa-R1]]: LaViDa-R1 的 answer-forcing 解决训练信号消失（全失败 group），StableDRL 解决梯度尖峰不稳定（高方差 importance ratio）——两个不同的训练病理，可同时存在且需同时解决
- `combines_with` → [[2025-SDAR-VL]]: SDAR-VL 的 ABNS/EMRS/PBNC 解决块状扩散预训练层面的不稳定，StableDRL 的阶梯注意力 + 无条件裁剪/自归一化解决块状扩散 RL 层面的不稳定——预训练和 RL 训练稳定性的双层优化
- `motivated_by` → V-trace/Retrace: 经典 off-policy RL 的截断重要性加权方法，StableDRL 将这些鲁棒性原则适配到 dLLM RL 的 proxy-ratio 场景
- `enables` → [[2025-SDAR-VL]]: 首次为块状扩散模型（SDAR-8B）提供可行的 RL 训练方案（阶梯注意力实现 O(1) ELBO 评估）

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- **GRPO 公式对 dLLM 噪声 importance ratio 的脆弱性**: 首次系统分析 GRPO 公式本身（而非 likelihood 估计方法）在 dLLM 中的两个设计缺陷——条件裁剪的噪声欺骗和固定归一化的幅度波动。提出理论框架（自强化不稳定循环）和简洁修复（无条件裁剪 + 自归一化）
- **dLLM 全参数 RL 训练的不可行性**: 此前所有 dLLM RL 工作使用 LoRA 或提前停止约束不稳定。StableDRL 首次实现 >1000 步的全参数 RL 训练，better unlock reasoning capability
- **块状扩散模型的 RL 训练**: 通过阶梯注意力实现 O(1) 无泄露 ELBO 评估，首次将 RL 训练扩展到块状扩散模型（SDAR-8B），并在 AIME'24 上超越 AR baseline Qwen3-8B

### 未解决的问题
- 问题: StableDRL 与 SPG 三明治 bounds 的组合效果
  - 为什么难: StableDRL 使用 SPG 的 block-wise masking 作为 score surrogate，但在 stress test 中发现 SPG 因 off-policy 偏差崩溃。将 StableDRL 的更新公式与 SPG 的三明治 ELBO+EUBO bounds 组合是否能实现"更好的 likelihood 估计 + 更稳定的优化"？
  - 潜在思路: 用 StableDRL 替换 SPG 的 GRPO 更新规则，保留 SPG 的三明治 likelihood 估计——StableDRL 提供优化稳定性，SPG 提供 likelihood 估计准确性
- 问题: 无条件裁剪对 on-policy 学习速度的影响
  - 为什么难: 无条件裁剪永久限制了 importance ratio 的范围，即使在 on-policy 阶段（ρ≈1）也施加约束。这可能在低噪声阶段不必要地减慢学习——ε=5 的宽松裁剪部分缓解了此问题，但在纯文本 LLM 上无条件裁剪可能引入不必要的约束
  - 潜在思路: 自适应 ε（根据 importance ratio 方差动态调整）；分阶段策略（前期无条件裁剪，训练稳定后放松为条件裁剪）
- 问题: 自归一化在极端 advantage 分布下的行为
  - 为什么难: 自归一化将更新约束在凸包内，当 advantage 分布极度不平衡（如 saturated failure，所有 A≈0）时，凸包退化为零向量附近——自归一化不解决训练信号消失问题（这是 EBPO 的领域）
  - 潜在思路: StableDRL + EBPO 叠加——StableDRL 解决梯度尖峰，EBPO 解决信号消失
- 问题: 多模态场景验证缺失
  - 为什么难: StableDRL 仅在纯文本推理任务（GSM8K, MATH500, Countdown, Sudoku）上验证。多模态场景中 image token NLL>6 导致 importance ratio 噪声更严重，ε=5 是否仍然合适？
  - 潜在思路: 在 MMaDA/LaViDa-R1 等多模态 dLLM 上验证；per-modality ε（text ε=5, image ε=2 更紧裁剪）

### 对问题树的推进
- 推进了:
  - [[problem-tree#[RL-2a] 多模态采样的成本远高于纯文本，如何降低？]]: StableDRL 提供第七种 dLLM RL 范式——**优化公式鲁棒化**。与前六种（似然度近似/似然度降方差/似然度 bounds/advantage 降方差/无似然度/off-policy 离散过滤）不同，StableDRL 不改善 likelihood 估计质量或 advantage 估计质量，而是修改 GRPO 更新公式本身以容忍噪声——是公式层面而非估计层面的改进
  - [[problem-tree#[RL-2c] 高熵 token 分布下的 RL 正则化]]: StableDRL 的无条件裁剪提供了一种隐式正则化——通过限制 importance ratio 范围间接限制策略偏移速度。与 KL 正则化不同，无条件裁剪不需要估计 KL（在高熵 image token 下方差大），而是直接在 ratio 空间施加硬约束
  - [[problem-tree#[RL-2g] Velocity-based vs Likelihood-based dLLM RL 的系统对比]]: StableDRL 显著提升了 likelihood-based 路线的实用性——首次证明 likelihood-based dLLM RL 可以全参数训练（之前 SPG/ESPO 需要 LoRA），缩小了与 LFPO velocity-based 路线的实用性差距
  - [[problem-tree#[RL-3a] 视觉推理的 verifiable reward]]: GSM8K 84.2%, MATH500 41.8%, AIME'24 16.7% (SDAR-8B, Static) 创下 dLLM 推理新 SOTA，AIME'24 超越 AR Qwen3-8B (10.0%)
  - [[problem-tree#[RL-3c] 传统 VLM RL vs dLLM RL 的方法论差异]]: StableDRL 的无条件裁剪+自归一化是**架构感知但非架构专有**的 PG 组件——它针对 dLLM 的噪声 importance ratio 设计，但其修改（无条件裁剪、自归一化）在数学上适用于任何含噪声 ratio 的 RL 场景。与 EBPO（完全架构无关）和 SPG/answer-forcing（dLLM 专有）之间
- 新增问题:
  - **[RL-2p] 🔴 GRPO 更新公式的全参数 RL 鲁棒性**: StableDRL 首次实现 dLLM 全参数 RL 训练（此前所有方法用 LoRA 或 early stopping）。全参数 RL 的长度泛化更好（128/256/512 一致），但计算成本更高。何时应该用全参数 vs LoRA？两者的 Pareto 前沿（性能 vs 计算成本）是什么形状？

## 个人深度评注

### [Critic] 贡献本质是诊断性而非算法性
- 无条件裁剪和自归一化分别对应 V-trace/Retrace 的截断重要性加权和自归一化重要性采样——都是经典 off-policy RL 的已知工具。StableDRL 的真正新颖性在于**首次正确诊断 GRPO 公式在 dLLM 中的两个具体漏洞**（条件裁剪被噪声欺骗 + 固定归一化放大 group-level 波动），并证明已知工具的组合足以修复。这种"诊断 > 算法"的贡献模式在实用性上非常有价值，但在学术新颖性评判上可能被低估
- ε=5 的实质是**异常值截断**而非紧致信赖域（标准 PPO ε∈[0.1, 0.3]）。"无条件裁剪"的标签有一定误导性——在 ε=5 下，ratio 范围 [0, 6] 非常宽，只有极端异常值被截断。这解释了为什么 ε=1 和 ε=5 都 work 但 ε=100/1000 崩溃——关键是截断足够严格的异常值，而非紧致约束

### [Critic] 自归一化的代价——丢失幅度信息
- 自归一化将更新约束在凸包内，丢失了"大 ρ 对应大策略偏移需要大修正"的幅度信号。当策略真的大幅偏移（legitimately large ρ）时，标准 GRPO 的幅度放大是正确行为，自归一化压制了这一信号。这是鲁棒性换效率的真实 tradeoff，非免费午餐
- 当 group 内 ρ 适度分散（非极端异常也非均匀）时，自归一化的重加权质量未被分析。与 median normalization、trimmed mean 等替代方案的比较缺失

### [Critic] vs SPG 的压力测试部分不公平
- 压力测试专为 importance ratio 噪声设计（StableDRL 的优势场景）。SPG 设计目标是改善 likelihood 估计质量，不是应对极端 ratio 噪声。测试 SPG 在 Exploding Weight 条件下的表现类似于在越野赛道上测试轿车——揭示了真实局限，但在 SPG 未设计应对的场景中。公平比较需要双向压力测试：(1) ratio 噪声（StableDRL 强项）+ (2) likelihood 估计偏差严重（SPG 强项）

### [Critic] vs LFPO 的结构性对比
- StableDRL GSM8K 84.2% vs LFPO 79.6%、MATH500 41.8% vs 37.6%——StableDRL 基准数据更优，可能因为：(1) token 级信用分配更精细（LFPO 在时间步级别操作，关键推理 token 仅 2-3 个时信号被稀释）；(2) 全参数训练提供更大容量（LFPO 可能用 LoRA）
- 但 LFPO 有结构性优势：完全消除 likelihood 估计噪声。随着序列/模态扩展（image token NLL>6），LFPO 的"无噪声底线"可能越来越有价值。StableDRL 的修复再好，也无法将方差降到 MC 估计的内禀方差之下——只能防止方差导致灾难性失败
- 长期看 LFPO 和 StableDRL 可能在不同 regime 各有优势：短序列低噪声下 StableDRL（保留 token 级精度），长序列高噪声下 LFPO（无噪声底线）

### [Connector] 在 dLLM RL 管线中的唯一定位
- StableDRL 是 KB 中唯一修改 GRPO 更新公式本身的工作。其他改进都作用于公式的"输入"（SPG/complementary masking 改善 likelihood 估计，EBPO 改善 advantage baseline，answer-forcing 改善样本质量）或"绕过"公式（LFPO 用速度场替代 PG）。这使 StableDRL 与其他所有方法正交可组合——是"管线出口"而非"管线入口"的改进
- **最有价值的组合预测**: StableDRL + EBPO + answer-forcing 三层栈，分别解决梯度尖峰/信号消失/探索失败——三个正交病理的三个正交修复。详见 `[[stabledrl-complete-three-layer-stack]]`

### [Ideator] 多模态验证是最关键的 gap
- 所有实验仅在纯文本推理任务上验证。image token NLL>6 使 importance ratio 噪声比文本场景严重得多——恰好是 StableDRL 最需要但最未被验证的场景。per-modality ε（text=5, image=2）是最自然的扩展方向
- SDAR-8B AIME'24 16.7% 超越 AR Qwen3-8B 10.0% 是 dLLM 在前沿推理基准上首次超越 AR 的里程碑结果——但仅限于 Static sampling 条件
