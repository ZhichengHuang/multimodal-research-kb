# StableDRL-Complete: 三层 dLLM RL 稳定性栈

> 状态: 构想
> 来源: [[2026-StableDRL]] + [[2026-EBPO]] + [[2026-LaViDa-R1]] 的正交性分析
> 问题树节点: [[problem-tree#[RL-2a]]], [[problem-tree#[RL-2c]]], [[problem-tree#[RL-2p]]]

## 核心思想

dLLM RL 训练存在三种正交的失败模式，当前分别有对应的解决方案，但尚无工作将三者组合为统一系统：

| 层 | 失败模式 | 解决方案 | 管线位置 | 来源 |
|---|---|---|---|---|
| Layer 1 | **梯度尖峰** — 噪声 importance ratio 导致梯度爆炸 | 无条件裁剪 + 自归一化 | 梯度聚合 | [[2026-StableDRL]] |
| Layer 2 | **信号消失** — 全失败 group 导致 advantage 全为零 | James-Stein shrinkage baseline | Advantage 计算 | [[2026-EBPO]] |
| Layer 3 | **探索失败** — 难题上无正向样本可学习 | Answer-forcing (dLLM inpainting) | Rollout 生成 | [[2026-LaViDa-R1]] |

## 正交性分析

三层方案作用于 RL 管线的不同阶段，互不干涉：

```
Rollout 生成 → Likelihood 估计 → Advantage 计算 → 梯度聚合 → 参数更新
     ↑                                    ↑                ↑
  Layer 3                              Layer 2          Layer 1
  answer-forcing                       EBPO             StableDRL
  (注入正向样本)                       (shrinkage)       (裁剪+归一化)
```

**Layer 1 ↔ Layer 2 无冲突**: EBPO 修改 advantage 值（shrinkage baseline），StableDRL 修改梯度聚合方式（无条件裁剪 + 自归一化）。两者作用于不同的数学量。StableDRL 的无条件裁剪与 advantage 符号无关，因此 EBPO 改变 advantage 值不会影响裁剪行为——这实际上比标准 GRPO + EBPO 更安全（标准 GRPO 的条件裁剪依赖 advantage 符号，EBPO 改变 advantage 可能意外触发/抑制裁剪）。

**Layer 1 ↔ Layer 3 无冲突**: Answer-forcing 修改 rollout 样本（注入 ground-truth），StableDRL 修改梯度聚合。Answer-forced 样本的 importance ratio 可能与自然样本的 ρ 分布不同（模型未"自然"生成），但 StableDRL 的无条件裁剪确保无论 ρ 如何极端都被限制在 [1-ε, 1+ε]。

**Layer 2 ↔ Layer 3 兼容**: Answer-forcing 降低 saturated failure 率（注入正向样本），EBPO shrinkage 从全失败 group 提取负向信号。两者从不同方向解决相同问题（训练信号不足）——answer-forcing 增加正向信号，EBPO 保留负向信号。

## 潜在协同效应

1. **StableDRL 使 EBPO 更安全**: EBPO 的 shrinkage 可能在极端情况下产生大 advantage 值（当 group mean 与 global mean 差距大时），在标准 GRPO 下可能触发条件裁剪 bypass 导致梯度尖峰。StableDRL 的无条件裁剪消除此风险。

2. **StableDRL 使 answer-forcing 更稳健**: Answer-forced 样本可能产生异常 ρ（模型对这些"注入"样本的 likelihood 估计噪声可能更大）。StableDRL 确保这些异常 ρ 不会导致梯度灾难。

3. **三层叠加可能启用全参数多模态 dLLM RL**: 当前所有多模态 dLLM RL（MMaDA UniGRPO, LaViDa-R1, DiMOO Self-GRPO）都使用 LoRA 或有限步训练。三层稳定性栈可能是首次实现全参数多模态 dLLM RL 的关键。

## 多模态扩展

对于多模态场景（text + image tokens），建议扩展：

- **Layer 1 扩展**: Per-modality ε（text ε=5, image ε=2——image token NLL>6 导致 ratio 噪声更严重，需要更紧裁剪）
- **Layer 2 扩展**: Per-task Welford 统计（不同任务的 reward 分布差异大，per-task shrinkage 更精确）
- **Likelihood 估计**: 在三层栈基础上叠加 complementary masking（LaViDa-R1）或 SPG 三明治 bounds 改善 likelihood 估计质量（与三层栈正交）

## 实验设计建议

### 基线对比
| 配置 | 说明 |
|------|------|
| Vanilla GRPO | 标准 GRPO（预期快速崩溃） |
| + StableDRL only | Layer 1 alone（防止梯度尖峰但可能信号不足） |
| + EBPO only | Layer 2 alone（改善 advantage 但可能梯度不稳定） |
| + Answer-forcing only | Layer 3 alone（改善探索但可能梯度不稳定） |
| + StableDRL + EBPO | Layer 1+2（稳定 + 信号恢复） |
| + StableDRL + AF | Layer 1+3（稳定 + 探索改善） |
| + EBPO + AF | Layer 2+3（信号恢复 + 探索改善） |
| StableDRL-Complete | Layer 1+2+3（完整三层栈） |

### 关键指标
- 训练稳定性: 可训练步数、梯度范数方差、reward 曲线平滑度
- 推理性能: GSM8K, MATH500, AIME'24（text reasoning）; MathVista, GenEval（multimodal）
- 全参数 vs LoRA: 长度泛化（128/256/512/1024）
- 消融: 单独去掉每一层的性能退化

### 建议基座
- 纯文本: LLaDA-8B-Instruct（与 StableDRL 原文一致）
- 多模态: MMaDA-8B 或 LaViDa-O-10.4B（KB 中最成熟的多模态 dLLM）

## 风险分析

1. **EBPO Gaussian 假设 vs StableDRL 裁剪**: 裁剪截断了 ρ 分布的尾部，可能使 Welford 统计偏离 Gaussian 假设。**缓解**: 使用非参数 shrinkage（基于分位数而非均值/方差）
2. **Answer-forcing 注入比例与 StableDRL ε 的交互**: 过多 forced 样本改变 group 内 ρ 分布的形状。**缓解**: 限制 answer-forcing 比例 ≤ 25%
3. **计算成本**: 三层叠加增加 overhead（EBPO 的 Welford 在线估计 + answer-forcing 的额外 inpainting 推理）。**预估**: ~15-20% 训练时间增加（主要来自 answer-forcing 的额外推理）
4. **调参复杂度**: 三层栈引入多个超参数（ε, shrinkage 权重, answer-forcing 比例）。**建议**: 固定 StableDRL（ε=5）和 EBPO（默认 shrinkage）的超参数，仅调 answer-forcing 比例

## 与 KB 中其他方案的关系

- 与 LFPO 互斥: LFPO 绕过 likelihood 估计，StableDRL 假定使用 likelihood-based PG。两者代表不同的技术路线
- 与 SPG 三明治 bounds 正交可叠加: SPG 改善 likelihood 估计质量（管线更前端），三层栈中可替换 complementary masking
- 与 MIS-PO 离散过滤部分重叠: MIS-PO 的 off-policy 过滤和 StableDRL 的裁剪都处理异常 ρ，但机制不同（离散丢弃 vs 连续截断）。在三层栈中可考虑用 MIS-PO 替代 StableDRL，但丢失更多信息

## 参考论文

- [[2026-StableDRL]] — Layer 1: 无条件裁剪 + 自归一化
- [[2026-EBPO]] — Layer 2: James-Stein shrinkage baseline
- [[2026-LaViDa-R1]] — Layer 3: Answer-forcing + complementary masking
- [[2025-SPG]] — 可选 Layer 0: 三明治 ELBO+EUBO likelihood bounds
- [[2026-LFPO]] — 替代路线: 无似然度速度场修正（与三层栈互斥）
