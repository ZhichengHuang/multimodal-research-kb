---
title: "OpenMMReasoner: Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe"
authors: []
date: 2025-11
venue: arxiv
url: "https://arxiv.org/html/2511.16334"
tags: [posttraining, rl, reasoning, multimodal, sft, grpo]
category: "rl/multimodal-reasoning"
level: 2
status: read
importance: high
problem_tree_nodes: []
aliases: [OpenMMReasoner]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结
提出完全透明的两阶段训练框架（SFT + RL），通过教师模型蒸馏、答案多样性扩展和 GSPO 算法，在 Qwen2.5-VL-7B 基座上实现 11.6% 的多模态推理性能提升。

## 核心 Insight
**答案多样性比数据规模更重要** — 通过 ×8 采样扩展答案多样性 + 跨领域混合（数学推理数据集），比单纯增加数据量更有效；过度过滤会损害多样性，反而降低性能。

## 与已有工作的关系
- **继承自**: 数据来源继承自多个工作的数据集构建经验，包括多模态推理数据的采集和标注方法；教师模型蒸馏范式继承自大模型蒸馏的通用实践
- **对比**:
  - **vs dLLM RL 工作** ([[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]], [[2025-MMaDA-Parallel]]): OpenMMReasoner 是**传统 VLM 架构**（Qwen2.5-VL-7B 基座，AR 生成），而非 diffusion-native 统一模型。这是根本性的架构差异
  - **vs MMaDA/LaViDa-R1 的任务定位**: MMaDA/LaViDa-R1 聚焦**理解+生成统一**（T2I generation + visual reasoning），OpenMMReasoner 聚焦**纯推理任务**（数学推理、多模态 QA），不涉及图像生成
  - **GSPO vs GRPO 变体**: OpenMMReasoner 的 GSPO 算法与 MMaDA 的 UniGRPO、Lumina-DiMOO 的 Self-GRPO、LaViDa-R1 的统一 PG 框架形成对比
- **互补**:
  - **与 dLLM RL 工作的互补性**: OpenMMReasoner 在传统 VLM 上验证了"RL 提升多模态推理"的有效性，与 dLLM RL 工作形成**架构互补**——证明 RL 对多模态推理的提升不依赖特定架构
  - **与 [[2025-dMLLM-TTS]] 的互补**: OpenMMReasoner 是训练时 RL 优化，dMLLM-TTS 是推理时 test-time scaling，两者可组合

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 两阶段训练框架

**Stage 1: SFT (874K samples)**
- 数据来源: LLaVA-CoT, OpenVLThinker, We-Math2.0, MMR1, MiroMind-M1
- 教师模型蒸馏: 使用 Qwen3-VL-235B-Instruct 生成高质量推理轨迹
- 答案多样性扩展: 对每个问题进行 ×8 采样，保留多样化答案
- 跨领域混合: 引入数学推理数据集（We-Math2.0）增强跨域泛化
- 关键发现: 过度过滤（aggressive filtering）会降低答案多样性，反而损害性能

**Stage 2: RL (74K samples)**
- 算法选择: GSPO（Group Relative Policy Optimization）优于 DAPO 和 GRPO，训练更稳定
- 复合奖励函数: 平衡准确性（accuracy reward）和格式遵循（format adherence reward）
- Rollout 配置: ×16 rollout 确保稳定训练动态
- 长度惩罚: 防止过度冗长的推理过程
- 温度设置: 标准温度设置（未过度调参）

## Building Blocks（可复用组件）

### Block 1: 教师模型蒸馏 + 答案多样性扩展
- **做法**: 使用大规模教师模型（Qwen3-VL-235B）对每个问题生成 ×8 采样，保留多样化答案而非仅保留最优答案
- **机制 (WHY it works)**:
  - 多样性扩展让学生模型接触到多种推理路径，增强泛化能力
  - 避免过拟合单一推理模式，提升对不同问题类型的适应性
  - 教师模型的规模优势（235B vs 7B）提供高质量监督信号
- **适用条件**:
  - 有足够计算资源运行大规模教师模型
  - 任务具有多种合理解法（如数学推理、视觉问答）
- **什么时候会 break**:
  - 任务只有唯一正确答案且推理路径固定（如简单分类）
  - 教师模型本身在目标任务上表现不佳
  - 过度采样导致低质量样本混入
- **可组合方向**:
  - 可与主动学习结合，优先采样高不确定性样本
  - 可与课程学习结合，先学简单样本再学多样化样本

### Block 2: 跨领域数据混合
- **做法**: 在多模态推理数据基础上，混入纯数学推理数据集（We-Math2.0）
- **机制 (WHY it works)**:
  - 文本推理能力与多模态推理能力存在正迁移
  - 数学推理提供结构化的逻辑训练，增强模型的系统性思维
  - 跨域混合防止模型过拟合特定模态的表面特征
- **适用条件**:
  - 目标任务需要逻辑推理能力
  - 不同领域的推理模式存在共性（如链式推理、分步求解）
- **什么时候会 break**:
  - 领域差异过大（如视觉推理 vs 常识推理）
  - 混合比例不当导致某一领域性能下降
- **可组合方向**:
  - 可扩展到更多领域（代码推理、科学推理）
  - 可与多任务学习框架结合，动态调整领域权重

### Block 3: GSPO 算法
- **做法**: 使用 Group Relative Policy Optimization，相比 GRPO 和 DAPO 更稳定
- **机制 (WHY it works)**:
  - Group-based normalization 减少方差，提升训练稳定性
  - Relative advantage 计算避免绝对奖励尺度问题
  - 适配多模态场景的复合奖励函数
- **适用条件**:
  - 需要稳定的 RL 训练过程
  - 奖励信号存在噪声或方差较大
- **什么时候会 break**:
  - Group size 设置不当（过小或过大）
  - 奖励函数设计不合理
- **可组合方向**:
  - 可与 KL 正则化结合（本文未详细说明是否使用）
  - 可扩展到其他 policy gradient 方法

### Block 4: 复合奖励函数（准确性 + 格式遵循）
- **做法**: 平衡任务准确性奖励和格式遵循奖励，避免模型生成不符合要求的输出
- **机制 (WHY it works)**:
  - 单一准确性奖励可能导致格式混乱（如缺少推理步骤）
  - 格式奖励确保输出可解析、可评估
  - 两者联合优化兼顾内容质量和形式规范
- **适用条件**:
  - 任务对输出格式有明确要求（如 CoT 推理）
  - 需要可解析的结构化输出
- **什么时候会 break**:
  - 两个奖励目标冲突（如格式要求过严限制推理灵活性）
  - 权重设置不当导致偏向某一目标
- **可组合方向**:
  - 可扩展到更多维度（如简洁性、可读性）
  - 可与 reward shaping 技术结合

## Anti-patterns / 已知失败模式
- **过度过滤损害多样性**: 论文明确指出，aggressive filtering 会降低答案多样性，反而损害性能。应保留合理的多样性而非追求"完美"数据
- **忽视跨域迁移**: 仅使用多模态数据训练，忽视文本推理数据的迁移价值
- **算法选择不当**: DAPO 和 GRPO 在本任务上不如 GSPO 稳定，需要实验验证算法适配性

## 实验关键发现
- **11.6% 整体提升**: 在 9 个 benchmark 上平均提升 11.6%（相比 Qwen2.5-VL-7B 基座）
- **数学推理显著提升**: MathVista, MathVerse, WeMath 等数学推理任务提升明显
- **文本推理正迁移**: "textual reasoning transfers alongside strengthened multimodal reasoning" — 多模态推理增强的同时，纯文本推理能力也得到提升
- **答案多样性 > 数据规模**: 通过多样性扩展和跨域混合，比单纯增加数据量更有效
- **教师模型选择关键**: 教师模型质量显著影响蒸馏数据质量

## Relations (结构化)
`alternative_to` → [[2025-MMaDA]]: 架构路线不同——OpenMMReasoner 是传统 VLM (Qwen2.5-VL-7B) + AR 生成，MMaDA 是 dLLM 统一模型；任务不同——OpenMMReasoner 纯推理，MMaDA 理解+生成；RL 算法不同——GSPO vs UniGRPO

`alternative_to` → [[2026-LaViDa-R1]]: 架构路线不同——传统 VLM vs dLLM；RL 方法不同——GSPO vs 统一 PG 框架 + answer-forcing；OpenMMReasoner 无 answer-forcing 机制解决训练信号消失

`alternative_to` → [[2025-Lumina-DiMOO]]: 架构路线不同——传统 VLM vs dLLM；任务不同——OpenMMReasoner 纯推理，DiMOO 理解+生成；RL 算法不同——GSPO vs Self-GRPO（自评估联合优化）

`alternative_to` → [[2025-MMaDA-Parallel]]: 架构路线不同——传统 VLM vs dLLM 并行生成架构；RL 算法不同——GSPO vs ParaRL（轨迹级 reward）

`combines_with` → [[2025-dMLLM-TTS]]: 训练时 RL 优化（OpenMMReasoner）+ 推理时 test-time scaling（dMLLM-TTS）可组合——先用 RL 提升基座推理能力，再用 HTS 搜索最优输出
- `alternative_to` → [[2026-EBPO]]: 同为改进 GRPO 的工作——GSPO 改进 group 构造，EBPO 改进 baseline 估计；两者正交可叠加
- `combines_with` → [[2025-KimiK2.5]]: 跨模态 RL 正迁移的互相验证——OpenMMReasoner 在传统 VLM 上，K2.5 在 AR MoE 上；但两者证据强度均不足以建立因果

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- **传统 VLM 的多模态推理能力提升**: 通过 SFT + RL 两阶段训练，在 Qwen2.5-VL-7B 基座上实现 11.6% 提升
- **数据效率问题**: 答案多样性扩展（×8 采样）+ 跨域混合比单纯增加数据量更有效
- **推理能力迁移**: 验证了文本推理（数学推理数据）对多模态推理的正迁移
- **RL 算法稳定性**: GSPO 在多模态推理任务上比 GRPO/DAPO 更稳定

### 未解决的问题
- **GSPO 机制不明**: 论文仅报告 GSPO 比 GRPO/DAPO 更稳定，但未解释 WHY — Group-based normalization 的理论基础、与 dLLM RL 的差异、是否是工程优化而非理论创新
  - 为什么难: 论文未提供 GSPO 的详细算法描述和理论分析
  - 潜在思路: 理论推导 GSPO 的方差界；与 GRPO 的 ablation 对比；在 dLLM 上验证 GSPO 的迁移性
- **答案多样性的理论边界**: ×8 采样是经验选择，最优采样数量与任务复杂度、教师模型质量的关系未建模
  - 为什么难: 需要建立信息论框架量化"推理空间的分布覆盖度"
  - 潜在思路: 信息论视角建模多样性收益；主动学习选择高不确定性样本；课程学习先简单后多样
- **跨域迁移的泛化性**: 仅验证数学推理 → 多模态推理，其他领域（代码推理、科学推理）的迁移效果未知
  - 为什么难: 需要系统性实验构建多领域迁移矩阵，计算成本高
  - 潜在思路: 任务表征学习预测迁移效果；元学习框架自动发现可迁移模��
- **传统 VLM vs dLLM 的 RL 方法论差异**: OpenMMReasoner 在传统 VLM 上验证 RL 有效性，但与 dLLM RL 的 likelihood 估计、KL 正则化、rollout 策略存在本质差异，两者的方法能否互相借鉴？
  - 为什么难: 架构差异导致 RL 实现细节完全不同（AR vs masked diffusion）
  - 潜在思路: 抽象出与架构无关的 PG 组件；GSPO 适配到 masked diffusion

### 对问题树的推进
- **推进了 [[problem-tree#RL-3]] RL 能提升多模态推理能力吗？**: 在传统 VLM（Qwen2.5-VL-7B）上验证 RL（GSPO）对多模态推理的有效性，9 个 benchmark 平均提升 11.6%
- **推进了 [[problem-tree#Post-2]] 多模态 SFT 数据如何高效构造？**: 验证了答案多样性扩展（×8 采样）和跨域混合（数学推理数据）的有效性
- **新增问题**:
  - **[RL-3c] 传统 VLM RL vs dLLM RL 的方法论差异**: 传统 VLM 使用标准 policy gradient（GSPO/GRPO），dLLM 需要特殊 likelihood 估计（complementary masking）。两者的 RL 方法能否互相借鉴？
  - **[Post-2c] 答案多样性扩展的理论基础**: OpenMMReasoner 发现"答案多样性 > 数据规模"，但机制不明。最优采样数量与任务复杂度的关系未建模
  - **[Post-2d] 跨域迁移的泛化性边界**: 验证数学推理 → 多模态推理的正迁移，但其他领域的迁移效果未知。哪些领域组合存在负迁移？

## 个人深度评注
<!-- 用户审阅后补充 -->
