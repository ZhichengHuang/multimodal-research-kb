---
title: "Qwen3-VL → dLLM 知识迁移研究方向"
status: draft
priority: high
origin: "Qwen3-VL Ideator Analysis"
related_papers: ["[[2025-Qwen3-VL]]", "[[2025-LLaDA-V]]", "[[2026-LaViDa-R1]]", "[[2025-MMaDA]]", "[[2025-Lumina-DiMOO]]", "[[2026-Beyond-LM]]", "[[2025-VidLaDA]]", "[[2025-XDLM]]"]
problem_tree_nodes: [PT-1, PT-1b, PT-3, PT-4, Diff-1b, Diff-2, Uni-1, Uni-2, Uni-4]
date: 2026-03-07
---

# Qwen3-VL 知识迁移研究方向

本文档由 Ideator Agent 基于 Qwen3-VL 论文关键事实与 KB 现有 Building Blocks 生成。

---

## 方向一：Interleaved-MRoPE → dLLM 的时空位置编码迁移

### 动机

Qwen3-VL 的 Interleaved-MRoPE 将位置编码分解为三个独立轴：`temporal`（时间帧 ID）、`height`（垂直空间）、`width`（水平空间），每个轴分配 1/3 的 head 维度。消融实验显示这是 Qwen3-VL 性能提升中最关键的单一因素（+4.2% MMMU）。

KB 现状：
- [[P-Diff-02]] 记录 dLLM 在长视频全局推理上优于 AR（+3.2 LongVideoBench），但在短视频时序任务上严重弱于 AR（MVBench -10.2）
- [[problem-tree#PT-4a]] 将"短视频时序因果推理是 dLLM 的结构性弱点"标记为 🔴 open
- VidLaDA 使用简单的 1D 时间位置编码，并未区分三轴
- dLLM 的 bidirectional attention 无法区分"A→B"和"B→A"序列，必须完全依赖位置编码区分时序方向

### 核心假设

MRoPE 的三轴分解，尤其是独立的 temporal 轴，可以：
1. **为 bidirectional dLLM 补偿缺失的时序方向性**——explicit temporal ID 区分帧顺序，部分弥补 causal mask 缺失导致的时序推理劣势
2. **在 masked diffusion 训练中提供稳定的时序锚点**——不同 mask 模式下 temporal embedding 始终保持不变，为去噪提供稳定的时序条件

### 方案：MRoPE-dLLM（三轴位置编码 dLLM）

**基础 Building Block（来源）**：
- Qwen3-VL Interleaved-MRoPE（时空三轴分解）
- LLaDA-V 的 bidirectional attention 基础架构（[[2025-LLaDA-V]]）
- VidLaDA 的三阶段视频课程训练策略（[[2025-VidLaDA]]）

**具体实现**：
1. 将 LLaDA-8B 的 1D RoPE 替换为 3-axis MRoPE（temporal/height/width），text tokens 保持原 1D RoPE
2. 在 SigLIP2 提取视觉特征后，根据帧 ID 和空间坐标分配三轴位置 ID
3. 沿用 VidLaDA 的 MARS-Cache 推理加速和三阶段视频课程训练

**预期效果**：
- MVBench（短视频时序任务）差距从 -10.2 缩小至 <-5（预测）
- LongVideoBench 优势保持（bidirectional attention 的全局推理优势不受影响）

**机制层面的兼容性**：
- MRoPE 修改的是位置编码而非 attention 模式，与 bidirectional mask、masked diffusion 训练目标完全正交
- 与 MARS-Cache 兼容——MARS-Cache 的 chunk attention 基于帧边界划分，与 MRoPE 的 temporal 轴在概念上一致

**风险**：
- MRoPE 的三轴分解是针对 AR causal 生成设计的，在 bidirectional masked diffusion 中的优化目标对齐性未验证
- dLLM 训练时每步 mask 的 token 分布不规则，三轴位置 ID 的编码稳定性需要检验

**验证路线图**：
1. 在 LLaDA-8B 上将 RoPE → MRoPE，仅在短视频分类任务（MVBench-style）上做 SFT，验证时序推理基线提升
2. 全量三阶段视频课程训练，与 VidLaDA 做控制变量对比

---

## 方向二：DeepStack → dLLM 多层视觉注入

### 动机

Qwen3-VL 的 DeepStack 在 LLM 中间层插入额外的视觉 cross-attention，让视觉信息在语言处理的多个语义层次上注入，而不仅在输入层通过 MLP 连接器一次性注入。消融显示 DeepStack 贡献 +3.1% MMMU。

KB 现状：
- [[problem-tree#PT-1c]] 记录所有 dLLM 工作均使用简单 MLP/线性投影连接器（LLaDA-V MLP、LaViDa 线性投影、Beyond-LM 简单 connector），认为"dLLM 不需要复杂连接器"
- 但这一结论的前提是"单点注入"——视觉 token 在 LLM 输入层对齐后就没有再注入
- PT-1c 当前状态为 🟡 partially-solved，问题"Connector 架构本质区别"仍未回答

### 核心假设

"dLLM 不需要复杂连接器"的结论仅针对"连接器架构复杂性"（MLP vs Q-Former），而非"注入位置数量"。
DeepStack 的多层注入从根本上改变了视觉信息在 LLM 计算图中的存在形式——从"输入条件"升级为"持续参与的跨层注意力源"，对 dLLM 的 bidirectional reasoning 可能有额外增益：bidirectional attention 在每层都可以利用跨层的视觉 cross-attention 信号，形成视觉引导的迭代精炼。

### 方案：DeepStack-dLLM

**基础 Building Block（来源）**：
- Qwen3-VL DeepStack（多层 cross-attention 视觉注入）
- LLaDA-V 的 dLLM 视觉理解框架（[[2025-LLaDA-V]]）
- LaViDa-O 的 Elastic-MoT（非对称参数分配），可参考其多层路由设计

**具体实现**：
1. 在 LLaDA-8B 的每 4 层插入一个 cross-attention 模块（32 层 → 8 个注入点），保持 bidirectional self-attention 不变
2. Cross-attention 的 K/V 来自 SigLIP2 的视觉特征（冻结），Q 来自 LLM 中间层隐状态
3. 额外参数量约 +15%（每个 cross-attn 模块 ~50M），在可控范围内

**与现有 KB 知识的兼容性分析**：

| 组合 | 兼容性 | 机制说明 |
|------|--------|----------|
| DeepStack + Complementary Masking | 完全兼容 | CM 作用于 masked token 的 loss 估计，与 cross-attn 层正交 |
| DeepStack + ML-Cache (DiMOO) | 部分兼容 | ML-Cache 缓存稳定 token 的 KV，cross-attn 的 KV（来自视觉）天然不变，可直接缓存 |
| DeepStack + MARS-Cache (VidLaDA) | 兼容 | 视觉 cross-attn KV 在帧间可复用（与 MARS-Cache 的视觉 token 慢刷新一致） |
| DeepStack + MRoPE (方向一) | 兼容 | MRoPE 作用于 self-attn 位置编码，cross-attn 的视觉端不使用 RoPE |

**预期收益**：
- MMMU/MMMU-Pro 上 +2-4%（类比 Qwen3-VL 消融的 +3.1%）
- 文档/图表理解（DocVQA, AI2D）等 dLLM 弱势任务可能改善——这些任务需要对结构化视觉信息的深层处理，多层注入可能比单层输入更有利

**风险**：
- "先冻结后解冻" 策略（[[P-PT-02]]）在有多层 cross-attention 时的最优解冻顺序未知
- 增加 ~15% 参数会降低训练效率，与 DiMOO 的全共享极简架构策略背道而驰

---

## 方向三：Adaptive Token Merging → dLLM 高效视频推理

### 动机

Qwen3-VL 的 Adaptive Token Merging 将相似视觉 token 合并（如相邻帧中静止区域），在动态分辨率（最高 1344×1344）下将 token 数量控制在可管理范围内，支持 256K 长上下文。

KB 现状：
- [[problem-tree#PT-1b]] 标记"视觉 token 数量与性能的 tradeoff"为 🔴 open
- VidLaDA 使用简单 2×2 average pooling（27²→14² per view，压缩率 75%），导致 OCR/文档任务明显弱化（TextVQA -7.25, DocVQA -15.5）
- LaViDa 同样用 2×2 pooling，受同样限制
- [[problem-tree#Uni-4a]] 标记"视频 token 数爆炸"为 🟡 partially-solved，训练层面仍未解决

### 核心假设

与 Qwen3-VL 的 AR 生成相比，dLLM 的 bidirectional attention 对 adaptive token merging 有独特优势：
- AR 模型合并 token 后，被合并的信息在 causal 生成中将永久丢失
- dLLM 的 bidirectional attention 允许所有时间步的 token 相互参考——即使某些区域 token 被合并，后续去噪步骤中仍可从保留 token 的双向 attention 中"借回"上下文

### 方案：Semantic-Aware Token Merging for dLLM (SAT-Merge)

**基础 Building Block（来源）**：
- Qwen3-VL Adaptive Token Merging（动态 token 合并策略）
- VidLaDA MARS-Cache 的模态异步刷新策略（[[2025-VidLaDA]]）——"视觉 token 时间稳定"是 token 合并的天然前提
- Sparse-LaViDa 的 register tokens（[[P-Diff-06]]）——被合并 token 的信息可通过 register token 压缩保存

**具体实现**：
1. 用 semantic hash（基于 SigLIP2 特征的 cosine 相似度）识别跨帧相似区域
2. 相似度 > 0.95 的 token 合并（保留一个代表 token，其余软合并到 register pool）
3. Register token pool（64-128 个可学习 token）作为被合并信息的全局摘要
4. Masked diffusion 去噪过程中，register token 参与 bidirectional attention，保证被合并信息仍可访问

**与 KB 知识的交叉验证**：
- [[P-Diff-06]] "Register Tokens 作为虚拟容量补偿"——SAT-Merge 是这一 pattern 的视频域扩展
- MARS-Cache 的"视觉 token 慢刷新"（视觉信息在帧间稳定）与 token 合并的假设（相邻帧相似区域可合并）在语义上一致，可联合优化

---

## 方向四：MoE + MRoPE → dLLM 的模态感知位置感知 MoE 专家

### 动机

Qwen3-VL 的 MoE 变体在 40% 延迟降低的代价下仅损失 1.2% 质量。Beyond-LM 发现 MoE（per-modality shared experts, G=16）将视觉-语言 scaling 指数差距从 0.10 缩小到 0.05（[[P-Uni-03]]）。

这两个发现的结合暗示：在 MoE 的 expert routing 中引入**位置感知**（spatial vs temporal vs text），可能同时解决多模态 MoE 的两个问题：
1. modality-aware routing（视觉 vs 文本 token 使用不同 expert）
2. spatial-aware routing（空间 token vs 时序 token 使用不同 expert）

### 方案：Position-Aware MoE for Multimodal dLLM (PasMoE)

**基础 Building Block（来源）**：
- Qwen3-VL MoE 架构（40% latency reduction with 1.2% quality loss）
- Beyond-LM per-modality shared experts（[[P-Uni-03]]，缩小 scaling 差距）
- MRoPE 三轴位置编码（方向一）

**具体实现**：
1. 在 Beyond-LM 的 per-modality MoE 基础上，增加 position-aware routing 维度
2. Expert 分组：text experts（RoPE 1D）/ spatial experts（height+width MRoPE 激活）/ temporal experts（temporal MRoPE 激活）
3. Routing 信号来自 MRoPE 的三轴位置 ID，不需要额外分类器

---

## 方向五：3-Stage Training × dLLM Backbone

### 动机

Qwen3-VL 采用三阶段训练：Stage 1（视觉-语言对齐）→ Stage 2（能力扩展，高分辨率 + 视频）→ Stage 3（指令微调）。

KB 中 dLLM 工作的训练阶段差异很大：LLaDA-V 三阶段（encoder冻结→解冻→推理增强），DiMOO 四阶段，VidLaDA 三阶段视频课程。但没有工作系统性对比不同训练阶段划分对 dLLM 的影响。

**关键问题**：Qwen3-VL Stage 2 的"能力扩展"阶段（高分辨率 + 视频）在 dLLM 中是否应该在 masked diffusion 训练稳定后才引入？还是从 Stage 1 开始就联合？

这与 [[problem-tree#PT-2c]] "视频数据的引入时机"和 [[Diff-1b]] "离散扩散在统一模型中的选择"直接相关。

### 方案：Staged Capability Expansion for dLLM

**基础 Building Block（来源）**：
- Qwen3-VL 三阶段训练 recipe
- VidLaDA 三阶段视频课程（短视频→长视频的渐进扩展）
- [[P-PT-02]] "先冻结后解冻"策略
- DiMOO 四阶段管线（[[2025-Lumina-DiMOO]]）

**核心假设**：对 dLLM，"能力扩展"阶段的最优时机比 AR 更晚——因为 masked diffusion 的训练需要更多步数才能稳定，过早引入高分辨率会增加方差导致训练不稳定（类比 SDAR-VL 发现的训练稳定性问题）。

**具体设计**：
1. Stage 1：低分辨率（224×224）图像对齐，encoder 冻结，稳定 masked diffusion 训练
2. Stage 2a：高分辨率（768×768）扩展，encoder 解冻（[[P-PT-02]]），引入 MRoPE 和 adaptive token merging
3. Stage 2b：短视频扩展（≤30s，参考 VidLaDA 阶段 1-2）
4. Stage 3：长视频 + SFT（VidLaDA 阶段 3 + Mixed CoT SFT）

---

## 附：问题树推进分析

### Qwen3-VL 推进的节点

| 节点 | 状态变化 | 说明 |
|------|---------|------|
| [PT-1b] 视觉 token 数量 tradeoff | 提供新数据点 | Adaptive token merging 在 256K 上下文下可行，但 dLLM 迁移效果未知 |
| [PT-3] 多模态 Scaling Law | 提供 AR baseline | 32B dense 和 MoE 变体的 scaling 数据丰富 dLLM scaling law 的对比基线 |
| [RL-3c] AR VLM vs dLLM RL 方法论差异 | 增强 AR 端证据 | Qwen3-VL 作为强 AR baseline，使 AR/dLLM RL 对比研究更有价值 |

### Qwen3-VL 暴露或新增的节点

| 节点 | 类型 | 描述 |
|------|------|------|
| [PT-1-NEW] MRoPE 位置编码迁移到 dLLM | 🔴 open | 三轴 MRoPE 能否改善 dLLM 的时序推理弱点（[[problem-tree#PT-4a]]） |
| [PT-1c-EXT] DeepStack 多层注入 vs 单点 MLP | 🔴 open | "dLLM 不需要复杂连接器"的结论是否排除多层注入 |
| [Diff-2-NEW] AR 的 position encoding 创新能否在 dLLM 中复现 | 🔴 open | MRoPE, NTK-RoPE 等 AR 位置编码技术在 masked diffusion 中的适用性 |
| [PT-6-EXT] Qwen3-VL 作为更强 AR 基座的 diffusion 微调 | 🔴 open | 用 Qwen3-VL-8B 替换 Qwen2.5-VL 做 DiffusionVL 类实验，AR 基座质量对迁移效果的定量影响（[[problem-tree#Diff-1e]]） |
