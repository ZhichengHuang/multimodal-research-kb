---
title: "DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models"
authors: [Lunbin Zeng, Jingfeng Yao, Bencheng Liao, Hongyuan Tao, Wenyu Liu, Xinggang Wang]
date: 2025-12
venue: arxiv
url: "https://arxiv.org/html/2512.15713"
tags: [diffusion, unified-model, posttraining, architecture]
category: "diffusion-foundation/ar-to-diffusion"
level: 2
status: read
importance: high
problem_tree_nodes: [Uni-1b, Uni-2a]
aliases: [DiffusionVL]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结
通过"扩散微调"将任意自回归视觉语言模型直接转换为扩散模型，无需架构修改，仅用5%训练数据即达到SOTA扩散模型性能，并提出Block Diffusion策略实现KV-cache复用和2×推理加速。

## 核心 Insight
AR模型和扩散模型之间的性能差距并非架构本质差异，而是训练范式差异——通过在预训练AR模型上进行扩散目标微调，可以直接继承AR模型的语言理解能力，同时获得扩散模型的并行解码优势。Block Diffusion通过块内双向注意力+块间因果注意力的混合设计，既保留了变长生成能力，又实现了KV-cache跨块复用。

## 与已有工作的关系
- **继承自**: [[2025-LLaDA-V]] (扩散VLM基线), [[2025-LaViDa]] (扩散VLM先驱)
- **对比**: [[2025-A2D-VL]] (同期工作，需要annealing策略), 传统AR-VLM (Qwen2.5-VL等)
- **互补**: 可与任意AR-VLM/AR-LM结合，提供AR→Diffusion转换通用方案
- **被诊断**: [[2026-NAP]]（NAP 诊断 Block Diffusion 半自回归为 Fast-DLM amplifying ARness 的实例）

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 两条转换路径

**路径1: AR-VLM → dVLM (范式转换)**
- 直接对已对齐的视觉语言模型进行端到端扩散微调
- 使用Block Diffusion目标替换原始AR目标
- 保持视觉编码器和连接器参数不变

**路径2: AR-LM → dVLM (模态+范式双转换)**
- 阶段1: 使用AR损失训练连接器实现视觉-文本对齐 (580K样本)
- 阶段2: 应用扩散微调转换为dVLM (738K样本)
- 适用于纯语言模型扩展到多模态场景

### 训练目标对比

**自回归损失 (AR)**:
```
ℒ_AR(x;θ) = -𝔼_x[∑(i=1 to L) log P_θ(x^i | x^<i)]
```
标准下一token预测，使用前序上下文。

**全序列扩散损失 (Full Diffusion)**:
```
ℒ_DM(x;θ) = -𝔼_{t,x_0,x_t}[(1/t) ∑(i=1 to L) log P_θ(x_0^i | x_t)]
```
对整个序列施加均匀mask概率t，仅在masked位置计算损失。

**块扩散损失 (Block Diffusion)**:
```
ℒ_BDM(x;θ) = -𝔼[∑(i=1 to L/D) α log P_θ(x_0^i | x_<i, x_D(i))]
```
将序列分为D大小的块，块内施加均匀噪声，α为损失缩放因子。

ffusion策略实现

**块划分与注意力模式**:
- 序列padding至块大小整数倍，切分为不重叠块
- 混合注意力: 块内双向注意力 + 块间因果注意力
- 噪声和干净embedding拼接，通过特殊mask强制注意力约束

**KV-Cache跨块复用**:
```
K_in, V_in = [K_cache; k], [V_cache; v]
```
当前块与缓存的前序上下文拼接，实现跨生成周期复用。

**块内解码策略**:
- 静态低置信度重mask (默认): 每步解码⌊B/S⌋个最高置信度token
- 动态重mask: 每步解码超过置信度阈值的所有token，实现可变速生成

## Building Blocks（可复用组件）

### Block 1: Diffusion Finetuning (扩散微调)
- **做法**: 在预训练AR模型上直接应用扩散训练目标，无需修改架构，仅改变训练损失和注意力mask
- **机制 (WHY it works)**: AR模型已学习强大的语言先验和视觉-语言对齐，扩散微调仅需教会模型"并行去噪"而非"从头学习语言"——本质是训练范式迁移而非知识重建
- **适用条件**:
  - 需要高质量预训练AR模型作为初始化
  - 适用于已有视觉-语言对齐的AR-VLM (直接微调) 或纯语言AR-LM (需先训练连接器)
  - 微调数据量可低至原始训练量的5% (738K vs 15M)
- **什么时候会 break**:
  - 基座AR模型质量差时，扩散微调无法弥补基础能力缺陷
  - 微调数据分布与下游任务严重不匹配时性能下降
- **可组合方向**:
  - 与任意AR-VLM/AR-LM结合 (Qwen, LLaMA, InternVL等)
  - 可叠加后训练技术 (SFT, DPO) 在扩散微调前后应用
  - 可与MoE、稀疏激活等效率优化组合

### Block 2: Block Diffusion Strategy (块扩散策略)
- **做法**: 将序列分为固定大小块，块内施加均匀噪声并行去噪，块间保持因果依赖；推理时复用前序块的KV-cache
- **机制 (WHY it works)**:
  - 块内并行: 同一块内token共享噪声水平，可同时预测，减少串行步骤
  - 块间因果: 保留自回归依赖，确保长文本生成连贯性和变长支持
  - KV-cache复用: 前序块的K/V可直接用于后续块，避免全序列重计算
- **适用条件**:
  - 适用于需要变长生成的场景 (vs 全序列扩散的固定长度限制)
  - 块大小需权衡: 小块(1-4)精度高但并行度低，大块(8-16)并行度高但精度略降
  - 推理时需要KV-cache支持的框架
- **什么时候会 break**:
  - 块大小过大(>16)时，块内token依赖关系被强制双向化，可能破坏自然的因果结构
  - 极短序列(<块大小)时无法发挥并行优势
  - 动态重mask策略在低置信度阈值下可能牺牲生成质量
- **可组合方向**:
  - 与speculative decoding结合: 用小模型预测块内token分布
  - 与分层生成结合: 粗粒度块规划 + 细粒度块内填充
  - 可扩展到视频生成: 时间维度分块 + 空间维度扩散

### Block 3: Hybrid Attention Pattern (混合注意力模式)
- **做法**: 在同一Transformer中实现块内双向注意力和块间单向因果注意力，通过特殊attention mask控制
- **机制 (WHY it works)**:
  - 双向注意力允许块内token相互参考，充分利用并行去噪能力
  - 因果注意力保证块间依赖单向传播，维持生成的自回归特性
  - 无需修改Transformer架构，仅通过mask实现行为切换
- **适用条件**:
  - 需要框架支持自定义attention mask (大多数现代框架已支持)
  - 块大小需与模型上下文窗口匹配
- **什么时候会 break**:
  - 块边界处可能出现不连贯 (但实验显示影响较小)
  - 与位置编码交互可能产生意外行为 (需验证RoPE等相对位置编码兼容性)
- **可组合方向**:
  - 与sliding window attention结合: 块内全注意力 + 块间滑窗
  - 与sparse attention结合: 块内稠密 + 块间稀疏
  - 可推广到encoder-decoder架构: encoder全双向, decoder块扩散

### Block 4: AR-to-Diffusion Knowledge Transfer (AR→扩散知识迁移)
- **做法**: 利用AR模型权重初始化扩散模型，跳过从零训练阶段，直接在高质量初始化上微调
- **机制 (WHY it works)**:
  - AR模型的FFN层已学习丰富的语言知识和推理能力
  - 注意力层已捕获视觉-语言交互模式
  - 扩散微调仅需调整"如何使用这些知识"(并行 vs 串行)，而非重新学习知识本身
- **适用条件**:
  - 源AR模型需在目标领域有良好表现
  - 微调数据需覆盖目标任务分布
  - 适用于资源受限场景 (相比从零训练节省95%数据)
- **什么时候会 break**:
  - 源模型和目标任务领域差异过大 (如代码模型→视觉任务)
  - 扩散训练超参数设置不当可能导致灾难性遗忘
- **可组合方向**:
  - 与LoRA/Adapter结合: 冻结AR权重，仅训练轻量适配层
  - 与知识蒸馏结合: 用AR模型作为teacher指导扩散student
  - 可扩展到其他模态: 音频AR模型 → 音频扩散模型

## Anti-patterns / 已知失败模式

- **全序列扩散的固定长度限制**: 传统扩散VLM无法支持变长生成，Block Diffusion通过块间因果依赖解决
- **KV-cache低复用效率**: 全序列扩散每步需重新计算所有token的K/V，Block Diffusion通过跨块复用提升2×速度
- **从零训练扩散VLM的数据饥渴**: LLaDA-V需15M样本，DiffusionVL仅需738K (5%)，证明AR初始化的重要性
- **块大小过大导致精度下降**: 块大小16时MMMU-Pro从32.1降至31.0，需权衡并行度和精度
- **动态重mask的质量-速度权衡**: 低置信度阈值虽加速但可能引入错误累积

## 实验关键发现

- **性能突破**: DiffusionVL-7B在MMMU-Pro上达36.9，超越同尺寸AR基线Qwen2.5-VL-7B (36.7)，同时比先前最佳扩散模型LLaDA-V-8B高1.7分
- **数据效率**: 仅用738K样本 (5% of LLaDA-V) 达到SOTA扩散性能，验证AR初始化的有效性
- **推理加速**: 在详细图像描述任务上实现2.0×加速，同时BERTscore比LLaDA-V-8B高2.02×
- **块大小影响**: 块大小1-16性能差异<1.1分 (MMMU-Pro: 31.0-32.1)，块大小8为最佳平衡点
- **去噪步数缩放律**: 推理时增加去噪步数持续提升质量，存在test-time scaling效应
- **路径等价性**: AR-LM→dVLM和AR-VLM→dVLM两条路径性能相当，证明方法通用性
- **超越并发工作**: 在相同数据量下超越A2D-VL (35.1 vs 35.0)，且无需annealing策略

## Relations (结构化)
- `extends` → [[2025-LLaDA-V]]: 继承扩散VLM框架，但通过AR初始化大幅降低训练成本（738K vs 15M+，仅5%数据量）。但存在混淆变量——base AR模型的预训练成本未计入
- `extends` → [[2025-LaViDa]]: Block Diffusion的混合注意力（块内双向+块间因果）是LaViDa Prefix-DLM的泛化——从"前缀因果"扩展到"任意块边界因果"，实现变长生成+KV缓存复用
- `alternative_to` → [[2025-A2D-VL]]: 同期AR-to-Diffusion转换工作，DiffusionVL更简单（无需annealing策略）且数据效率相当（MMMU-Pro 35.1 vs 35.0），证明AR-Diffusion差距主要是训练目标而非架构不兼容
- `enables` → 任意AR-VLM (Qwen2.5-VL, InternVL, LLaVA-Next等): 提供通用AR→Diffusion转换方案，使已部署AR-VLM可获得扩散优势（并行解码、test-time scaling、infilling）而无需从头训练
- `combines_with` → [[2025-LaViDa]]: Block Diffusion可作为LaViDa的推理加速后端，Prefix-DLM (3.9×) + Block Diffusion (2×) 正交组合，理论加速7-8×
- `combines_with` → [[2025-Lumina-DiMOO]]: Block Diffusion + ML-Cache形成四层缓存层次（前缀KV + 块间KV + 稳定token + register tokens），理论总加速10-15×
- `motivated_by` → AR-VLM的成功部署现状: 大量组织已有Qwen/LLaVA等AR-VLM生产部署，DiffusionVL提供低成本迁移路径而非全面替换
- `conflicts_with` → [[2026-Beyond-LM]]: 路线对立——AR初始化（DiffusionVL）vs 从零训练（Beyond-LM）。DiffusionVL无法证明扩散训练本身优于AR训练，Beyond-LM提供了更公平的对比基准
- `combines_with` → [[2025-SDAR-VL]]: 同为块状扩散 VLM——DiffusionVL 的 Block Diffusion 聚焦 AR→Diffusion 转换和 KV-cache 复用，SDAR-VL 聚焦从零训练的训练稳定性优化（ABNS/EMRS/PBNC）；Block Diffusion 策略可迁移到 SDAR-VL 架构

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- 扩散VLM性能显著落后AR-VLM，且需从零训练成本高昂
- 现有扩散VLM无法支持变长生成，限制实际应用
- 扩散模型推理时KV-cache复用效率低，速度慢

### 未解决的问题
- 问题: 扩散VLM在复杂推理任务 (如数学、代码) 上仍弱于顶级AR模型
  - 为什么难: 扩散训练可能削弱链式推理能力，块内并行可能破坏思维链结构
  - 潜在思路: 混合训练 (部分层保持AR，部分层扩散)，或设计推理专用的块划分策略
- 问题: 块边界处的连贯性保证缺乏理论分析
  - 为什么难: 块间因果依赖是否足以传递所有必要信息尚不明确
  - 潜在思路: 引入块间overlap或cross-block attention增强信息流动

### 对问题树的推进
- **推进了 [[problem-tree#Uni-1b]] AR+Diffusion融合细节** 🔴→🟡: 证明融合通过微调而非架构重设计实现，但违反第一性原理——无法证明扩散训练本身优于AR训练
- **推进了 [[problem-tree#Diff-1c]] 采样效率** 🟡→🟡强化: Block Diffusion提供第四个正交加速维度（块级并行+KV缓存）
- **推进了 [[problem-tree#Uni-2a]] 共享参数下能力冲突** 🟡→🟡强化: AR知识可无损迁移至Diffusion
- **新增问题 [[problem-tree#PT-6]]**: 从零训练 vs AR初始化的公平对比——需要在同等训练数据、同等计算资源下对比
- **新增问题 [[problem-tree#Diff-1e]]**: AR模型质量对扩散微调效果的定量关系
- **新增问题 [[problem-tree#Diff-1f]]**: 块大小的自适应选择理论
- **新增问题 [[problem-tree#Diff-1g]]**: 块间信息衰减的实验验证（需要长序列/多块场景测试）
- **新增问题 [[problem-tree#Diff-1h]]**: 扩散微调的数据效率上界（5%能否进一步压缩?）

## 个人深度评注

### 核心价值与局限
- **核心价值**: 这篇论文打破了"扩散VLM需从零训练"的认知，证明AR和扩散的差距是训练范式而非架构本质——这为快速构建高质量扩散VLM提供了捷径
- **Block Diffusion的巧妙性**: 混合注意力设计优雅地平衡了并行性和因果性，KV-cache复用是实用性的关键突破
- **潜在局限**: 块大小固定可能不适应所有任务 (短回复 vs 长文档)，未来可探索动态块大小或分层块结构

### 批判性反思: 违反第一性原理的实验设计

**核心问题**: 这篇论文**无法证明基于diffusion的模型能够完全依托自己的训练方式达到很好的性能**，因为它的成功可能主要来自AR模型的知识迁移，而非扩散训练本身的优越性。

**混淆变量分析**:
1. **数据量不对等**: DiffusionVL用738K样本达到35.1 MMMU-Pro，而LLaDA-V从零训练���15M+样本达48.6。但这个对比不公平——DiffusionVL的base AR模型(Qwen2.5-VL)本身就是用**数百万甚至上亿样本**预训练的，这些知识被"免费"继承了
2. **真正的对比应该是**: (Qwen预训练数据 + 738K扩散微调) vs (15M从零训练扩散)。如果算上Qwen的预训练成本，DiffusionVL的总数据量可能**远超**LLaDA-V
3. **AR模型质量是隐藏变量**: 论文未测试"弱AR模型 + 扩散微调"的下界。如果base AR模型仅30%准确率，扩散微调后可能仅28-32%——这说明性能主要来自AR基础，而非扩散训练的增益

**为什么现在AR transfer超过原生扩散?**
- **多模态dLLM训练数据太少**: LLaDA-V的15M样本相比AR-VLM的训练数据(通常50M-100M+)仍然是小规模。扩散模型可能需要**更多数据**才能从零学会视觉-语言对齐
- **AR模型已经"作弊"**: Qwen2.5-VL在大规模数据上已经学会了视觉理解、语言生成、推理能力。扩散微调只是"换个生成方式"，核心能力都是继承的
- **扩散训练的真正潜力未被验证**: 如果给扩散模型**同等规模**的训练数据(50M-100M)，是否能超越AR? 这个问题DiffusionVL无法回答

**Block Diffusion的块间信息衰减问题**:
- **理论隐患**: 块间因果依赖意味着第N块要访问第1块的信息需要经过N-1次"中继"，每次中继都可能损失信息
- **缺乏实验验证**: 论文未提供长序列(>2048 tokens)或多块(>16 blocks)场景下的性能曲线，无法判断信息衰减的实际影响
- **需要补充实验**: 在不同块数量(4/8/16/32/64)下测试性能，绘制"块数量 vs 性能"曲线，验证是否存在衰减拐点

**对统一模型研究的启示**:
- **AR和Diffusion可能不是对立选择**: 而是同一模型的不同"工作模式"——未来模型可能支持运行时切换范式
- **但需要更公平的对比**: 在**同等训练数据、同等计算资源**下对比从零训练的AR vs Diffusion，才能判断哪个范式本质上更优
- **当前结论**: DiffusionVL证明了"AR→Diffusion转换是可行的"，但**未证明**"Diffusion训练本身优于AR训练"

**与RL结合的可能性**: 扩散模型的迭代精炼特性天然适合RL微调，可探索在去噪过程中引入reward guidance
