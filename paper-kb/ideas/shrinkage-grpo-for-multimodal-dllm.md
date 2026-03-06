---
title: "Shrinkage-GRPO for Multimodal dLLM RL"
status: draft
priority: high
origin: "[EBPO x KB RL methods] Ideator analysis"
related_papers: ["[[2026-EBPO]]", "[[2025-MMaDA]]", "[[2026-LaViDa-R1]]", "[[2025-Lumina-DiMOO]]", "[[2025-OpenMMReasoner]]"]
problem_tree_nodes: [RL-2, RL-2a, RL-2c, RL-3, RL-3c, Uni-5a]
date: 2026-03-05
---

# Shrinkage-GRPO for Multimodal dLLM RL

## 动机

多模态 dLLM RL（UniGRPO、Self-GRPO、LaViDa-R1 统一 PG）面临比纯文本 RL 更严重的 EBPO 所发现的两个问题:

1. **Saturated failure 更频繁**: 多模态推理（几何、图表、空间推理）的基线成功率远低于纯数学推理。dLLM 在 AI2D、DocVQA 等任务上系统性弱于 AR（[[P-Diff-02]]），意味着 rollout 全部失败的概率更高。同时视觉生成的 compositional reasoning 同样困难（MMaDA GenEval Position 仅 0.20）
2. **Group size 天然受限**: 多模态采样成本远高于纯文本（[[problem-tree#RL-2a]]）。UniGRPO 用结构化随机 mask ratio 替代 Monte Carlo 128-sample 正是为了降低成本，这意味着实际 group size 更小，正落入 EBPO 优势最大的区间

## 核心方案

**将 EBPO 的三个 building blocks 适配到多模态 dLLM RL:**

### 1. Shrinkage Baseline for dLLM Advantage Estimation

- **修改 UniGRPO/Self-GRPO 的 advantage 计算**: 将 Â_i = (r_i - μ_group) / σ_group 替换为 Â_i = (r_i - V^EB) / σ，其中 V^EB = (1-S_q) × μ_group + S_q × μ_glob
- **与 LaViDa-R1 框架的兼容性**: LaViDa-R1 将 SFT、GRPO、self-distillation 统一为加权 PG。Shrinkage baseline 仅修改 A_i^{GRPO} 的计算，不影响框架结构，A_i^{aggr} = 0.5 × A_i^{distill} + 0.5 × A_i^{GRPO-shrinkage} 直接可用
- **关键适配**: dLLM 的 reward 来自多种任务（correctness、IoU、EditScore），不同任务的 reward scale 不同。需要 per-task 维护 Welford 统计量，而非全局混合

### 2. Multimodal Topic-Coherent Sampling

- **聚类维度设计**: 不能简单照搬数学领域聚类。提议按**任务类型 x 视觉内容复杂度**二维聚类:
  - 任务类型: 数学推理 / VQA / grounding / image editing / T2I
  - 视觉复杂度: 简单自然图像 / 图表文档 / 多图场景 / 高分辨率细粒度
- **与 DiMOO Self-GRPO 的适配**: DiMOO 的联合优化 L(θ) = -Σw(g)(l_T2I + l_MMU) 中 T2I 和理解任务天然属于不同"主题"，topic-coherent sampling 可以让 Welford 估计器在每种任务类型内更准确
- **来源**: EBPO Proposition 3.8 证明随机 shuffle 引入额外方差 = 主题间均值方差。多模态场景主题间方差更大（T2I vs 数学推理的成功率分布完全不同），因此聚类带来的方差降低更显著

### 3. Answer-Forcing + Shrinkage 的协同效应

- **假说**: LaViDa-R1 的 answer-forcing 解决 saturated failure 的方式是"强制注入正向信号"（注入比例约 10%）。EBPO 的 shrinkage baseline 解决同一问题的方式是"让零成功的 group 获得负梯度信号"。**两者机制互补**:
  - Answer-forcing: 告诉模型"存在一条成功的路径"（正向引导）
  - Shrinkage baseline: 告诉模型"你当前的策略在这个难度级别上低于平均水平"（负向约束）
- **预测**: 两者组合可能优于任一单独使用。具体来说，answer-forcing 的注入比例可以在 shrinkage 的帮助下降低（因为 shrinkage 已经提供了部分信号），从而减少 answer-forcing 的 collapse 风险

## 机制层面的兼容性分析

| 组合 | 兼容性 | 风险 |
|------|--------|------|
| Shrinkage + Complementary Masking | 完全正交。Shrinkage 改 advantage 估计，CM 改 likelihood 估计 | 无已知冲突 |
| Shrinkage + SFT 正则化（替代 KL） | 兼容。Shrinkage 不依赖 KL 约束 | Shrinkage 可能与 SFT 正则化形成双重保守（需调参） |
| Topic-coherent + 多任务 RL | 需要 per-task Welford。多任务混合可能破坏聚类 | 极端 imbalanced 任务分布可能导致稀有任务的先验不准 |
| EBPO-diff 课程 + dLLM | 需要估计多模态任务难度。视觉内容难度估计不如数学成熟 | 视觉难度和推理难度可能不对齐 |

## 预期收益

1. **效率提升**: 在保持 G=8 的条件下达到 G=32 的训练效果，等效节省 4x rollout 计算
2. **困难任务提升**: 在 dLLM 的结构性弱点任务（图表/文档理解、空间推理）上获得更好的梯度信号
3. **理论贡献**: 首个将统计收缩估计引入 dLLM RL 的工作，提供 architecture-agnostic PG 组件的具体实例

## 验证路线图

1. **Phase 1** (2 weeks): 在 MMaDA UniGRPO 上实现 Shrinkage baseline，固定 G=8，对比 AIME/MATH-500 性能
2. **Phase 2** (2 weeks): 扩展到 LaViDa-R1 框架，测试 Shrinkage + answer-forcing + SFT 正则化的组合
3. **Phase 3** (3 weeks): 实现多模态 topic-coherent sampling，在 DiMOO 多任务 RL 上验证

## 风险与缓解

- **Gaussian 假设在高熵 image token 下失效**: 缓解——使用 non-parametric shrinkage（经验 Bayes 不依赖参数分布假设，只需二阶矩）
- **Welford 估计器在 RL 策略演化中滞后**: 缓解——使用指数加权移动平均（EWMA）替代全历史 Welford，窗口大小与 KL divergence 联动
- **多模态 reward scale 不统一**: 缓解——per-task 独立维护 Welford 统计量，batch 内标准化在 per-task 子集上进行
