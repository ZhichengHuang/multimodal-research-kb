---
title: "dMLLM-TTS: Self-Verified and Efficient Test-Time Scaling for Diffusion Multi-Modal Large Language Models"
authors: [Yi Xin, Siqi Luo, Qi Qin, et al.]
date: 2025-12
venue: arxiv
url: "https://arxiv.org/html/2512.19433"
tags: [diffusion, unified-model, rl, posttraining, generation]
category: posttraining/test-time-scaling
level: 2
status: read
importance: high
problem_tree_nodes: [Uni-5, Diff-1c, RL-5]
aliases: [dMLLM-TTS, SVF, HTS]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结
提出 dMLLM-TTS，首个面向离散扩散多模态大模型的 test-time scaling 框架，通过 Self-Verified Feedback（模型自身理解能力评估生成质量）和 Hierarchical Trajectory Search（O(N+T) 复杂度的分层搜索），在 Lumina-DiMO/MMaDA/Muddit 上实现 +17.9%~+29.4% GenEval 提升，同时达到 5-6× 推理效率增益。

## 核心 Insight
Test-time scaling（推理时计算投入）是 AR 模型（如 o1）的核心能力，但在 dLLM 中尚未被系统探索。dMLLM-TTS 证明了 dLLM 的 **bidirectional attention + 统一理解-生成架构** 天然适合 test-time scaling——模型自身的理解能力可作为 verifier（无需外部 reward model），多轨迹并行采样可通过分层剪枝高效搜索。这为 dLLM 开辟了一条区别于训练时 scaling 的新路径。

## 与已有工作的关系
- **继承自**: [[2025-Lumina-DiMOO]]、[[2025-MMaDA]]、[[2025-Muddit]]（三个基座模型，提供统一理解-生成架构）
- **对比**: [[2025-ReDiff]]（训练时精炼 vs 推理时搜索）、[[2026-LaViDa-R1]]（Tree Search 在 diffusion 时间维度 vs HTS 在分辨率维度）
- **互补**: [[2025-Lumina-DiMOO]]（Self-GRPO 训练时自评估 + SVF 推理时自验证）、[[2026-LaViDa-R1]]（HTS breadth pruning + Tree Search depth exploration）

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 整体框架
dMLLM-TTS 在推理时沿两个轴进行 scaling:
1. **Trajectory Exploration（轨迹探索）**: 生成 N 个多样化候选输出
2. **Iterative Refinement（迭代精炼）**: 对候选进行 T 轮自验证和改进

三阶段流程:
- **Stage 1: Stochastic Exploration** — 初始采样 N 个候选轨迹
- **Stage 2: Hierarchical Thinning** — 分层剪枝，从 N 个候选中筛选出 top-k
- **Stage 3: Final Refinement** — 对存活候选进行迭代精炼

### Self-Verified Feedback (SVF)
- **做法**: 利用 dLLM 自身的图像理解能力评估生成质量。对每个生成的图像，模型生成一组 QA 对（基于 prompt 的关键信息），然后用模型自己回答这些问题，正确率作为质量分数
- **关键设计**:
  - QA 生成基于 prompt 的 entity-relation-value 三元组提取
  - 避免外部 verifier（如 CLIP/VLM-as-judge）的偏差和计算开销
  - 利用 dLLM 统一架构的独特优势（AR 模型无法做到）

### Hierarchical Trajectory Search (HTS)
- **做法**: 分层搜索策略，先在低分辨率/少步数下快速筛选，再在高分辨率/多步数下精炼
- **复杂度**: O(N+T) vs 线性搜索的 O(N×T)
- **关键机制**:
  - Coarse-to-fine generation: 256² → 512² → 1024²
  - Adaptive pruning: 每层根据 SVF 分数剪枝，仅保留 top-k 进入下一层
  - Early stopping: 低质量候选在早期被淘汰，避免浪费计算

### 支持的基座模型
- Lumina-DiMO (8B): GenEval 0.78 → 0.92 (+17.9%)
- MMaDA (8B): GenEval 0.51 → 0.66 (+29.4%)
- Muddit (1B): GenEval 0.53 → 0.67 (+26.4%)

## Building Blocks（可复用组件）

### Block 1: Self-Verified Feedback（自验证反馈）
- **做法**: 用模型自身的理解能力评估生成质量。生成图像后，模型基于 prompt 生成验证问题（如 "图中有几个苹果？""苹果是什么颜色？"），然后用模型回答这些问题，正确率作为质量分数
- **机制 (WHY it works)**: dLLM 统一架构使理解和生成共享参数，理解能力可直接用于评估生成质量。相比外部 verifier（CLIP/VLM），SVF 避免了模态 gap（CLIP 的 text-image embedding 不对齐）和幻觉问题（VLM-as-judge 的偏差）。QA 格式提供了细粒度的 compositional 评估（vs CLIP 的全局相似度）
- **适用条件**: 需要统一理解-生成模型（dLLM）；理解能力需足够准确；适用于可验证的生成任务（T2I、image editing）
- **什么时候会 break**: (1) 理解能力本身有偏差时，SVF 分数不可靠（bootstrapping 偏差）；(2) QA 生成质量依赖 prompt 的结构化程度——模糊 prompt 难以生成有效验证问题；(3) 对抽象/艺术风格图像，QA 验证可能不适用（难以定义"正确答案"）
- **可组合方向**: 与外部 reward（如 EditScore）组合做交叉验证；与 [[2025-Lumina-DiMOO]] 的 Self-GRPO 结合（训练时自评估 + 推理时自验证）

### Block 2: Hierarchical Trajectory Search（分层轨迹搜索）
- **做法**: 分层搜索策略——先在低成本条件下（低分辨率/少步数）生成大量候选并快速筛选，再在高成本条件下精炼 top-k 候选。每层根据 SVF 分数自适应剪枝
- **机制 (WHY it works)**: 利用"低分辨率生成质量与高分辨率生成质量正相关"的假设，在早期淘汰低质量候选，避免在它们身上浪费高分辨率计算。分层结构将 O(N×T) 的线性搜索降到 O(N+T)——N 个候选在 coarse 层并行，仅 top-k 进入 fine 层。Adaptive pruning 根据质量分布动态调整保留比例
- **适用条件**: 生成质量在不同分辨率/步数下单调或强相关；计算资源受限需要高效搜索；候选质量分布有明显分层（top 和 bottom 差距大）
- **什么时候会 break**: (1) 低分辨率质量与高分辨率质量不相关时（如某些细节密集型任务），早期剪枝可能误杀；(2) 所有候选质量接近时，剪枝收益小；(3) 分层数过多导致每层候选数过少，搜索空间不足
- **可组合方向**: 与 [[2025-Lumina-DiMOO]] 的 ML-Cache 结合（分层搜索 + 稳定 token 缓存）；与 [[2026-LaViDa-R1]] 的 Tree Search 结合（HTS 做 breadth pruning，Tree Search 做 depth exploration）

### Block 3: Iterative Refinement（迭代精炼）
- **做法**: 对筛选出的 top-k 候选进行多轮迭代改进。每轮用 SVF 识别错误（QA 回答错误的部分），然后用 dLLM 的 inpainting 能力修正这些区域
- **机制 (WHY it works)**: dLLM 的 bidirectional attention 和 masked diffusion 范式天然支持 inpainting——可以 mask 掉错误区域，保留正确区域，重新生成。SVF 提供了细粒度的错误定位（哪些 entity/attribute 错误），指导 inpainting 的 mask 策略。迭代精炼类似 self-correction，但基于外部验证信号（QA 正确性）而非模型内部置信度
- **适用条件**: 需要 dLLM 的 inpainting 能力；错误可被 QA 验证定位；迭代次数有限（避免过度精炼导致质量下降）
- **什么时候会 break**: (1) 错误定位不准确时，inpainting 可能破坏正确区域；(2) 过多迭代导致 over-refinement（类似 [[2025-Muddit]] 的 64 步性能下降）；(3) 某些错误是结构性的（如整体布局错误），局部 inpainting 无法修复
- **可组合方向**: 与 [[2025-ReDiff]] 的精炼训练结合（训练时学习纠错 + 推理时迭代精炼）；与 [[2025-MMaDA-Parallel]] 的并行生成结合（并行生成多个精炼版本）

## Anti-patterns / 已知失败模式
- **线性搜索（Baseline）**: O(N×T) 复杂度，在所有候选上执行完整精炼流程，计算开销过大
- **外部 Verifier 的局限**: CLIP-based reward 不支持 compositional reasoning（[[P-RL-01]]）；VLM-as-judge 有幻觉和偏差问题（[[2026-LaViDa-R1]] 报告）
- **固定剪枝比例**: 不考虑质量分布的动态性，可能在质量接近时过度剪枝，或在质量分散时保留过多低质量候选
- **单分辨率搜索**: 无法利用 coarse-to-fine 的计算效率优势

### 深层失败模式（Critic Agent 分析）
- **SVF Bootstrapping 偏差循环**: 模型用自身理解能力评估生成质量，系统性理解偏差会导致错误生成得高分。与 [[2025-Lumina-DiMOO]] Self-GRPO 的训练时偏差不同，SVF 的推理时偏差无梯度更新机制校准，可能更严重。缺乏外部 anchor reward（如 EditScore）交叉验证
- **HTS 单调性假设失效**: 在细节密集型任务（精细文字、复杂纹理）中，低分辨率（256²）的 SVF 分数不是高分辨率（1024²）质量的无偏估计，导致早期剪枝误杀。与 [[2026-LaViDa-R1]] Tree Search 的"从 step 16+ 分支无���"类似——过早决策基于信息不足的状态
- **QA 生成质量依赖 Prompt 结构化程度**: Entity-relation-value 提取在模糊/抽象 prompt（"一幅超现实主义画作"）上失效，无法生成有效验证问题。[[P-RL-01]] 指出 CLIP 在 compositional prompt 上失效（GenEval Position 0.20），SVF 是否能处理需要在 GenEval 细分指标上验证
- **Iterative Refinement 的 Over-Refinement**: 多轮迭代可能导致质量振荡——后续 inpainting 破坏前期修正。类似 [[2025-Muddit]] 的 64 步性能下降。缺乏显式收敛判断机制（vs [[2025-ReDiff]] 训练时学习的隐式停止策略）
- **训练-推理分布不匹配（Inpainting）**: Iterative Refinement 的 inpainting 面临 [[P-Diff-04]] 指出的问题——mask 掉错误区域后的分布与训练时干净数据不同。[[2025-ReDiff]] 通过两阶段精炼训练解决，dMLLM-TTS 未处理
- **计算开销的隐藏成本**: O(N+T) 分析忽略分辨率差异——256² 生成 N 个候选的成本远低于 1024² 生成 N 个（16× 像素差异）。真实复杂度 O(N·C_low + k·T·C_high)。与训练时方法（[[2025-ReDiff]]、[[2026-LaViDa-R1]]）的 tradeoff 不明确——如果推理成本等于"训练一个小型 RL 模型"，可能不如一次性训练

## 实验关键发现
- **Lumina-DiMO**: GenEval 0.78 → 0.92 (+17.9%)，在 compositional generation 任务上提升最显著
- **MMaDA**: GenEval 0.51 → 0.66 (+29.4%)，基座模型越弱，test-time scaling 收益越大
- **Muddit**: GenEval 0.53 → 0.67 (+26.4%)，1B 小模型也能从 test-time scaling 受益
- **效率增益**: HTS 相比线性搜索达到 5-6× 加速，O(N+T) vs O(N×T) 的理论优势得到验证
- **Compositional tasks 提升最大**: Position、Count、Attribute Binding 等需要精确控制的任务受益最多
- **SVF vs CLIP**: SVF 在 compositional 评估上显著优于 CLIP score（细粒度 QA vs 全局相似度）

## Relations (结构化)
<!-- Agent 用于构建关系网络。type: extends | alternative_to | combines_with | motivated_by | enables | conflicts_with -->
- `extends` → [[2025-Lumina-DiMOO]]: 在 DiMOO 基座上应用 test-time scaling，GenEval 0.78→0.92
- `extends` → [[2025-MMaDA]]: 在 MMaDA 基座上应用 test-time scaling，GenEval 0.51→0.66
- `extends` → [[2025-Muddit]]: 在 Muddit 基座上应用 test-time scaling，GenEval 0.53→0.67
- `motivated_by` → [[DeepSeek-R1]]: Test-time scaling 范式的灵感来源（o1-style inference-time compute）
- `combines_with` → [[2025-Lumina-DiMOO]]: SVF 与 DiMOO 的 Self-GRPO 互补——训练时自评估 RL + 推理时自验证搜索
- `combines_with` → [[2026-LaViDa-R1]]: HTS 与 LaViDa-R1 的 Tree Search 互补——HTS 做 breadth pruning，Tree Search 做 depth exploration
- `alternative_to` → [[2025-ReDiff]]: 两种提升 dLLM 生成质量的路径——训练时精炼（ReDiff）vs 推理时搜索（dMLLM-TTS）
- `combines_with` → [[2025-OpenMMReasoner]]: 训练时 RL 优化（OpenMMReasoner GSPO）+ 推理时 test-time scaling（dMLLM-TTS HTS）可组合——先用 RL 提升基座推理能力，再用 HTS 搜索最优输出

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- **首次将 test-time scaling 系统化应用于 dLLM**：填补了 dLLM 在推理时优化这一维度的空白，证明了"推理时计算投入"在 dLLM 中的可行性
- **Self-Verified Feedback（SVF）**: 利用 dLLM 统一架构的独特优势——模型自身的理解能力评估生成质量，无需外部 verifier（避免 CLIP 的 compositional reasoning 缺失和 VLM-as-judge 的幻觉问题）
- **Hierarchical Trajectory Search（HTS）**: O(N+T) 复杂度的分层搜索策略，通过 coarse-to-fine 剪枝实现 5-6× 效率增益
- **跨基座验证**: 在 Lumina-DiMO（GenEval 0.78→0.92, +17.9%）、MMaDA（0.51→0.66, +29.4%）、Muddit（0.53→0.67, +26.4%）三个不同架构上均验证有效

### 未解决的问题
- 问题: SVF 的 Bootstrapping 偏差
  - 为什么难: 模型用自身理解能力评估生成质量，理解偏差会传播到 reward signal，形成循环偏差。统一模型的生成和理解共享参数，评估偏差与生成偏差耦合
  - 潜在思路: 引入外部 VLM（GPT-4V）做交叉验证；定期用 anchor reward（EditScore）校准
- 问题: HTS 的低-高分辨率质量相关性假设
  - 为什么难: HTS 假设低分辨率质量与高分辨率质量正相关，但在细节密集型任务（如精细 OCR、纹理生成）中可能不成立。某些视觉特征在低分辨率下完全丢失，早期剪枝可能误杀
  - 潜在思路: 任务自适应剪枝策略；多尺度质量预测模型
- 问题: 迭代精炼的收敛性保证
  - 为什么难: 论文未提供精炼过程的收敛性分析，可能存在振荡或过度精炼导致质量下降。无显式收敛约束，bidirectional attention 的全局更新可能创建循环依赖
  - 潜在思路: 收敛正则化（惩罚 token 翻转）；自适应停止策略
- 问题: 计算成本的定量分析缺失
  - 为什么难: 论文展示质量提升但未量化 wall-clock time 或 FLOPs，无法判断 cost-benefit tradeoff。迭代精炼 + 多轮采样的实际计算开销可能远超理论分析
  - 潜在思路: 与训练时方法（ReDiff、LaViDa-R1 RL）做 compute-matched 对比

### 对问题树的推进
- 推进了 [[problem-tree#Uni-5]]: 开辟了"推理时优化"的第三条路径（vs 训练时 SFT/RL），证明 test-time scaling 在 dLLM 上可行且高效
- 推进了 [[problem-tree#Diff-1c]]: HTS 提供了与现有加速技术（Prefix-DLM KV 缓存、ML-Cache）正交的新方向，分层搜索 + 自适应剪枝的范式可推广到其他 dLLM 任务
- 新增问题 [RL-5] Test-time Scaling for dLLM: dMLLM-TTS 首次系统性探索 dLLM 的推理时计算投入，与 AR 模型的 o1-style reasoning 形成对比——dLLM 通过多轨迹搜索 + 自验证实现
- 新增问题 [RL-5a] dLLM Test-time Scaling 的理论上界: 在什么条件下推理时搜索的收益饱和？与训练时 scaling 的 tradeoff？
- 新增问题 [RL-5b] Self-Verification 的可靠性边界: 哪些任务类型适合自验证？抽象/艺术风格生成如何定义"正确"？
- 新增问题 [Diff-1c-3] 分层搜索的最优层次数和剪枝策略: 当前 coarse→fine 两层是否最优？能否推广到三层或动态层次？

## 个人深度评注

### [Critic] 增量贡献评估
| Block | 新颖性 | 理由 |
|-------|--------|------|
| Block 1: SVF | 中 | 推理时应用（vs DiMOO 训练时）降低偏差传播风险；QA 格式提供细粒度评估。但本质是"VLM-as-judge 的特例"，继承所有风险 |
| Block 2: HTS | 中 | 空间维度（分辨率）的分层 vs LaViDa-R1 时间维度（diffusion step），两者正交可组合。但单调性假设在细节密集型任务中脆弱 |
| Block 3: Iterative Refinement | 低中 | 无需额外训练，但质量上界受限于现有 inpainting 能力。缺乏 ReDiff 训练时学习的隐式停止策略 |

### [Critic] 核心判断: 系统化 > 方法创新
dMLLM-TTS 的核心价值不在于单个 Block 的技术新颖性，而在于**首次将 test-time scaling 系统化应用于 dLLM**，证明了"推理时计算投入"在 dLLM 中的可行性。这为 dLLM 开辟了区别于训练时 scaling 的新路径。

**关键发现**：
- **基座模型越弱，收益越大**（MMaDA +29.4% vs DiMOO +17.9%）——暗示 test-time scaling 更多是"弥补训练不足"而非"突破能力上界"
- **与训练时方法的 Tradeoff**：ReDiff 训练成本高但推理无额外开销；dMLLM-TTS 无需训练但推理成本高（5-6× 加速仍意味着比单次生成慢）

### [Connector] dMLLM-TTS 在知识库中的定位
```
dLLM 统一模型谱系:
├── 训练时优化
│   ├── 数据驱动: DiMOO (110M 数据)
│   ├── RL 优化: MMaDA (UniGRPO), LaViDa-R1 (answer-forcing)
│   └── 错误修正: ReDiff (精炼训练)
│
└── 推理时优化  ← dMLLM-TTS 开辟的新维度
    ├── 多轨迹搜索: dMLLM-TTS (SVF + HTS)
    ├── 中间状态搜索: LaViDa-R1 (Tree Search)
    └── 推理加速: DiMOO (ML-Cache), LaViDa (Prefix-DLM)
```

### [Ideator] 最值得探索的研究方向
1. **SVF + 外部 Reward 混合验证**（可行性：高）
   - 在简单 prompt 上用外部 reward 作为 anchor，校准 SVF 的评分尺度
   - 对复杂 prompt 用 SVF + 外部 reward 的加权组合（权重根据 prompt 复杂度自适应）
   - 依赖: dMLLM-TTS Block 1 + LaViDa-R1 Multi-Reward RL

2. **Test-time + Training-time 协同**（可行性：高）
   - 先用 ReDiff 两阶段训练提升基座模型的纠错能力
   - 在推理时应用 dMLLM-TTS 的 HTS + 迭代精炼
   - 用 ML-Cache 加速多轮精炼的计算开销
   - 依赖: ReDiff + dMLLM-TTS + DiMOO ML-Cache

3. **二维搜索组合**（可行性：中等偏高）
   - HTS 做 breadth pruning（分辨率层筛选）+ LaViDa-R1 Tree Search 做 depth exploration（时间步分支）
   - Stage 1: 256² 生成 N=16 个候选，SVF 筛选 top-4
   - Stage 2: 对 top-4 从 diffusion step 8/64 分支出 4 个新候选（共 16 个）
   - Stage 3: 提升到 1024²，SVF 筛选 top-1

### [Critic] 关键隐含假设
1. **SVF 的自洽性假设**: 模型的理解能力无系统性偏差；生成和理解的参数共享不引入冲突
2. **低分辨率质量单调性**: 全局布局错误在低分辨率下可见；细节质量与全局质量相关
3. **QA 生成质量假设**: Entity-relation-value 提取准确；prompt 足够结构化
4. **Inpainting 的修正能力**: 错误是局部的；mask 策略能准确覆盖错误区域

### [Critic] 与 o1-style Reasoning 的对比
| 维度 | o1 (AR 模型) | dMLLM-TTS (dLLM) |
|------|-------------|------------------|
| 计算投入方式 | 更长的 CoT 推理链 | 多轨迹并行采样 + 分层搜索 |
| Verifier | 外部 reward model | 模型自身理解能力（SVF）|
| 搜索空间 | Sequential reasoning steps | Parallel generation trajectories |
| 独特优势 | 因果推理链可解释 | Bidirectional attention 支持 inpainting |
| 主要风险 | CoT 幻觉 | Bootstrapping 偏差 |
