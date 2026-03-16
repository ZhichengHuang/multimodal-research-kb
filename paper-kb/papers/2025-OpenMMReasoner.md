---
title: "OpenMMReasoner: Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe"
authors: []
date: 2025-11
venue: arxiv
url: "https://arxiv.org/html/2511.16334"
tags: [posttraining, rl, reasoning, multimodal, sft, grpo, training-stability]
category: "rl/multimodal-reasoning"
level: 2
status: read
importance: high
problem_tree_nodes: [RL-3, RL-3a, RL-3c, Post-2, Post-2c, Post-2d]
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
  - **与 [[2026-EBPO]] 的互补**: EBPO 的 shrinkage baseline 是 GSPO advantage 估计的替代改进——GSPO 降低 ratio 方差，EBPO 降低 baseline 方差，两者正交可叠加

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 两阶段训练框架

**Stage 1: SFT (874K samples)**
- 数据演化: 103K 原始 QA → 583K 蒸馏验证推理轨迹 → 874K 混合最终数据
- 数据来源: LLaVA-CoT, OpenVLThinker, We-Math2.0, MMR1, MiroMind-M1
- 教师模型蒸馏: 使用 Qwen3-VL-235B-Instruct（比 baseline 在所有 benchmark 上平均提升 ≥4.5 分）
- 答案多样性扩展: ×8 采样（50.5→55.2 平均分）；采用 no-filtering 策略，过度过滤损害性能
- 跨领域混合: 引入 MMR1（图像数学）+ MiroMind-M1（纯文本数学），混合后 55.2→56.3
- 训练超参: lr=5e-5, AdamW, cosine scheduler, max_len=61440, 4300 steps, packing enabled

**Stage 2: RL (74K samples)**
- 数据: MMEureka, ViRL, TQA, We-Math, PuzzleVQA, AlgoPuzzleVQA, ThinkLiteVL
- 算法: GSPO（Group Sequence Policy Optimization）——使用序列级 importance ratio 替代 GRPO 的 token 级 ratio，配合更小 clipping threshold ε
- 复合奖励: R = (1−λ_fmt)·R_acc + λ_fmt·R_fmt, λ_fmt=0.1
- Rollout 配置: ×16 rollout（比 ×8 更稳定、更高 reward）
- 温度: 1.0（1.4 导致失败——窄稳定窗口）
- 训练超参: lr=1e-6, AdamW, global_batch=128, max_gen=28696, max_prompt=4096, 1232 steps

## Building Blocks（可复用组件）

### Block 1: 教师模型蒸馏 + 答案多样性扩展
- **做法**: 使用大规模教师模型（Qwen3-VL-235B, ~33× 学生规模）对每个问题生成 ×8 采样，保留全部多样化答案（no-filtering 策略）。性能从 ×1 的 50.5 提升到 ×8 的 55.2
- **机制 (WHY it works)**:
  - **隐式探索预播种**: ×8 采样近似覆盖教师推理分布的多个 mode（代数/几何/数值等不同解法），为下游 RL 提供多样化的策略初始化点，本质上是 SFT 阶段的"exploration pre-seeding"
  - **信息论视角**: 1 样本让学生学 delta 分布（点估计），8 样本学近似条件分布 P(reasoning|problem)，保留了 RL 探索所需的策略熵
  - **No-filtering 的隐式正则化**: 保留"非最优但正确"的推理路径，相当于 reasoning-level 的 label smoothing——防止过拟合单一推理模式
  - 教师模型的规模优势（235B vs 7B）确保采样质量下界
- **适用条件**:
  - 有足够计算资源运行大规模教师模型
  - 任务具有多种合理解法（推理路径分布有 K≥2 个有效 mode）
  - 教师模型在目标领域足够强（accuracy > 50%）
- **什么时候会 break**:
  - 任务只有唯一正确答案且推理路径固定（如简单分类）——多样性无增量信息
  - 教师模型在特定领域（如医学影像、3D 空间理解）能力薄弱时，×8 采样产生 8 种类似的错误
  - ×8 可能是经验最优——数学推理的有效 mode 约 2-5 个，×8 覆盖主要 mode 后收益递减
- **可组合方向**:
  - 可与课程学习结合（早期低多样性高质量→晚期全多样性展开）
  - No-filtering + EBPO shrinkage baseline 可叠加——SFT 保留多样性 + RL 降低 advantage 方差

### Block 2: 跨领域数据混合
- **做法**: 在多模态推理数据基础上，混入 MMR1（图像数学）+ MiroMind-M1（纯文本数学），从 55.2→56.3
- **机制 (WHY it works)**:
  - **共享推理原语**: 数学推理和视觉推理共享序列化分解、状态追踪、中间结果验证等推理原语。Transformer attention 以模态无关方式学习这些原语，数学数据提供密集可验证的训练信号
  - **双向迁移证据**: OpenMMReasoner 证明 math→multimodal 正迁移（文本推理 15.1→22.2→29.4）；[[2025-KimiK2.5]] 有弱证据 visual→text（MMLU-Pro +1.7%）。两者方向一致但因果性未建立
  - 跨域混合防止 modality-specific surface feature 过拟合
- **适用条件**:
  - 目标任务需要逻辑推理能力（分步求解、逻辑链构建）
  - 不同领域的推理模式存在共性
- **什么时候会 break**:
  - **Modality neglect**: 文本数据 loss 更低时主导梯度更新，模型可能"用文本思维解视觉问题"而忽视视觉信息——需监控 per-domain loss 曲线
  - 混合比例不当导致某一领域性能下降
  - 领域间推理模式差异过大（如纯视觉推理 vs 符号推理）时正迁移消失
- **可组合方向**:
  - 可扩展到更多领域（代码推理、科学推理）构建 N×N 迁移矩阵
  - 可结合动态混合权重（训练中根据验证集调整领域比例）

### Block 3: GSPO 算法（Group Sequence Policy Optimization）
- **做法**: 将 GRPO 的 **token 级 importance ratio**（∏ᵢ π(aᵢ|sᵢ)/π_ref(aᵢ|sᵢ)）替换为**序列级 importance ratio**（π(seq)/π_ref(seq) 作为单一标量），配合更小 clipping threshold ε
- **机制 (WHY it works)**:
  - **方差降低**: Token 级 importance ratio 的乘积方差随序列长度 L 指数增长（长 CoT 尤其严重）。序列级 ratio 等价于 token 级 ratio 的几何均值，个体波动被平滑
  - **更紧信赖域**: 更小 ε 限制策略更新幅度，在序列级 ratio 已较平滑的条件下成本低
  - 实证: 比 GRPO 和 DAPO "收敛更快、reward 更高、行为更稳定"
- **关键 tradeoff**: 序列级 ratio 对所有 token 施加相同 importance weight——**牺牲了 token 级信用分配精细度**。对长 CoT 中关键推理步骤（如 "therefore the triangle is isosceles"，占 <10% tokens）和填充文本（如 "let me think"）无差别对待
- **适用条件**:
  - 需要稳定 RL 训练、序列较长（>500 tokens）的 CoT 推理场景
  - 当 GRPO token 级 ratio 的方差导致训练不稳定时
- **什么时候会 break**:
  - 当信用分配精细度重要时（短序列且关键 token 占比小）
  - 论文未公开 GSPO 完整算法细节和理论分析——迁移到其他任务/架构时效果未知
- **可组合方向**:
  - **GSPO + EBPO**: 正交改进——GSPO 改 importance ratio 计算，EBPO 改 advantage baseline 估计，两者可直接叠加
  - **GSPO-MDM**: 将序列级 ratio 适配到 masked diffusion（via complementary masking 估计序列概率），避免 dLLM 高熵 token 的 ratio 爆炸问题

### Block 4: 复合奖励函数（准确性 + 格式遵循）
- **做法**: R = (1−λ_fmt)·R_acc + λ_fmt·R_fmt, λ_fmt=0.1（即 90% 准确性 + 10% 格式奖励）
- **机制 (WHY it works)**:
  - 单一准确性奖励可能导致格式混乱（缺少推理步骤/结构标记）
  - 格式奖励确保输出可解析、可评估（CoT 格式合规）
  - λ_fmt=0.1 低权重确保格式不喧宾夺主
- **适用条件**:
  - 任务对输出格式有明确要求（如 CoT 推理需要 `<think>...</think>` 标记）
  - 需要可解析的结构化输出
- **什么时候会 break**:
  - **Reward hacking 风险**: RL 训练中反思词频率增加（"wait"、"let me reconsider"）可能是格式奖励的表面关联而非真正推理改善——需验证反思词是否实际伴随答案修正
  - 两个奖励目标冲突（如格式要求过严限制推理灵活性）
- **可组合方向**:
  - 可与 Kimi K2.5 GRM 多维评估方案对比——GRM 提供更细粒度的多维度评分
  - 可扩展为 curriculum: λ_fmt 随训练衰减（模型学会格式后减少格式约束）

## Anti-patterns / 已知失败模式
- **过度过滤损害多样性**: aggressive filtering 降低答案多样性反而损害性能——机制: (1) 噪声正则化效应消失; (2) 训练分布出现系统性 gap（如删除所有含 backtracking 的推理链，导致模型不会从错误中恢复）; (3) 过度打磨的 SFT 数据使 RL 策略过于自信，抑制探索
- **忽视跨域迁移**: 仅使用多模态数据训练，忽视文本推理数据的迁移价值
- **算法选择不当**: DAPO 和 GRPO 在本任务上不如 GSPO 稳定，需实验验证算法适配性
- **序列级 importance ratio 破坏精细信用分配**: GSPO 对所有 token 施加相同 importance weight。长 CoT（>500 tokens）中关键推理步骤（<10% tokens）的梯度信号被填充文本稀释。检测: 监控 per-token gradient magnitude，关键 token vs 填充 token 梯度相近说明信用分配失效
- **反思词频率作为 reward hacking 信号**: RL 训练中反思词（"wait"、"let me reconsider"）频率增加，可能是格式奖励的表面关联而非真正推理改善。检测: 分析反思词是否实际伴随答案修正（functional reflection）还是仅作为风格标记（cosmetic reflection）
- **温度敏感性表明策略脆弱性**: 1.0 work 而 1.4 fail——窄稳定窗口暗示 RL 策略在探索-利用相变临界点附近。部署时温度微调可能导致性能剧烈变化。此发现可能特定于 Qwen2.5-VL-7B
- **教师-学生质量反转**: 在教师模型薄弱的领域（如专业医学影像、3D 空间理解），×8 采样 + no-filtering 注入系统性错误。检测: 运行蒸馏前计算 per-domain 教师 accuracy，<50% 的领域需过滤
- **跨域混合的 modality neglect**: 文本数据 loss 更低时主导梯度更新，模型偏向"文本思维"而忽视视觉信号。检测: per-domain loss 曲线——文本 loss 下降远快于多模态 loss 说明 modality neglect 正在发生

## 实验关键发现
- **11.6% 整体提升**: 在 9 个 benchmark 上平均提升 11.6%（相比 Qwen2.5-VL-7B 基座）
- **逐阶段提升**: MathVista 74.8→79.5（SFT→RL），MathVision 36.6→43.6, WeMath 63.8, CharXiv 79.0
- **答案多样性定量验证**: ×1 sampling 50.5 → ×8 sampling 55.2（+4.7），加跨域混合 56.3（+1.1）
- **文本推理正迁移**: AIME24 6.7→16.7→27.1（baseline→SFT→RL），整体 15.1→22.2→29.4——效应量显著，强于 Kimi K2.5 的 +1.7% MMLU-Pro
- **GSPO 优于 GRPO/DAPO**: 收敛更快、reward 更高、训练更稳定（具体数字未给出）
- **×16 rollout > ×8**: 更高 reward 更平滑训练动态
- **RL 增强反思行为**: 反思词频率在 RL 训练中增加（可能 genuine 也可能 reward hacking，见 Anti-patterns）
- **Token 效率**: 在更少 token 预算下优于 OpenVisionReasoner

## Relations (结构化)
`alternative_to` → [[2025-MMaDA]]: 架构路线不同——OpenMMReasoner 是传统 VLM (Qwen2.5-VL-7B) + AR 生成，MMaDA 是 dLLM 统一模型；任务不同——OpenMMReasoner 纯推理，MMaDA 理解+生成；RL 算法不同——GSPO vs UniGRPO

`alternative_to` → [[2026-LaViDa-R1]]: 架构路线不同——传统 VLM vs dLLM；RL 方法不同——GSPO vs 统一 PG 框架 + answer-forcing；OpenMMReasoner 无 answer-forcing 机制解决训练信号消失

`alternative_to` → [[2025-Lumina-DiMOO]]: 架构路线不同——传统 VLM vs dLLM；任务不同——OpenMMReasoner 纯推理，DiMOO 理解+生成；RL 算法不同——GSPO vs Self-GRPO（自评估联合优化）

`alternative_to` → [[2025-MMaDA-Parallel]]: 架构路线不同——传统 VLM vs dLLM 并行生成架构；RL 算法不同——GSPO vs ParaRL（轨迹级 reward）

`combines_with` → [[2025-dMLLM-TTS]]: 训练时 RL 优化（OpenMMReasoner）+ 推理时 test-time scaling（dMLLM-TTS）可组合——先用 RL 提升基座推理能力，再用 HTS 搜索最优输出
- `alternative_to` → [[2026-EBPO]]: 同为改进 GRPO 的正交维度——GSPO 改 importance ratio 计算（token→sequence level），EBPO 改 advantage baseline 估计（local→shrinkage）；两者可直接叠加。**注**: EBPO 有 James-Stein 理论基础，GSPO 理论不完备
- `combines_with` → [[2025-KimiK2.5]]: 跨模态 RL 正迁移的互相验证——OpenMMReasoner 在传统 VLM 上，K2.5 在 AR MoE 上；但两者证据强度均不足以建立因果

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- **传统 VLM 的多模态推理能力提升**: 通过 SFT + RL 两阶段训练，在 Qwen2.5-VL-7B 基座上实现 11.6% 提升
- **数据效率问题**: 答案多样性扩展（×8 采样）+ 跨域混合比单纯增加数据量更有效
- **推理能力迁移**: 验证了文本推理（数学推理数据）对多模态推理的正迁移
- **RL 算法稳定性**: GSPO 在多模态推理任务上比 GRPO/DAPO 更稳定

### 未解决的问题
- **GSPO 机制不完备**: 论文描述 GSPO 使用序列级 importance ratio + 更小 ε，但未提供完整算法、理论分析或方差界推导。核心 tradeoff（方差降低 vs 信用分配损失）未被量化
  - 为什么难: 序列级 ratio 在不同任务/架构上的行为缺乏理论预测框架
  - 潜在思路: 理论推导 GSPO 方差界与序列长度关系；与 GRPO/EBPO 的严格 ablation；在 dLLM 上验证 GSPO 迁移性（GSPO-MDM）
- **答案多样性的理论边界**: ×8 是经验选择。推理分布有 K 个有效 mode 时，最优采样数 k* 应满足 mode 覆盖条件。k* 与任务复杂度、教师模型质量的关系未建模
  - 为什么难: 需要估计任务分布熵 H(Task_distribution|Path_set)，本身是开放问题
  - 潜在思路: 信息论框架 k* = argmin_k [H(Task|Path_k)] subject to compute budget；embedding 空间覆盖度近似
- **跨域迁移的泛化性**: 仅验证数学推理 → 多模态推理，其他领域（代码推理、科学推理）的迁移效果未知。双向迁移的因果性未建立
  - 为什么难: 需要系统性 N×N 迁移矩阵，计算成本高；confounding variables（训练数据重叠、continued training effect）难控制
  - 潜在思路: 任务表征学习预测迁移效果；元学习框架自动发现可迁移模式
- **传统 VLM vs dLLM 的 RL 方法论差异**: GSPO 使用标准 AR log p(y|x)，无需互补掩码。两者核心差异: likelihood 估计、KL 正则化、rollout 策略。GSPO 的稳定性优势是来自更简单的 likelihood 估计还是序列级 ratio 本身？
  - 为什么难: 架构差异导致不可控变量过多
  - 潜在思路: GSPO-MDM（将 GSPO 序列级 ratio 适配到 masked diffusion）；抽象架构无关 PG 组件
- **教师-学生最优规模比**: 235B→7B (~33×)。最优比例未知，是否存在最小有效比例（如 32B→7B 约 4.6× 是否足够）？
  - 潜在思路: 系统消融不同规模教师；信息论分析蒸馏效率与规模比的关系

### 对问题树的推进
- **推进了 [[problem-tree#RL-3]] RL 能提升多模态推理能力吗？**: 在传统 VLM（Qwen2.5-VL-7B）上验证 RL（GSPO）对多模态推理的有效性，9 个 benchmark 平均提升 11.6%
- **推进了 [[problem-tree#Post-2]] 多模态 SFT 数据如何高效构造？**: 验证了答案多样性扩展（×8 采样）和跨域混合（数学推理数据）的有效性
- **新增问题**:
  - **[RL-3c] 传统 VLM RL vs dLLM RL 的方法论差异**: 传统 VLM 使用标准 policy gradient（GSPO/GRPO），dLLM 需要特殊 likelihood 估计（complementary masking）。两者的 RL 方法能否互相借鉴？
  - **[Post-2c] 答案多样性扩展的理论基础**: OpenMMReasoner 发现"答案多样性 > 数据规模"，但机制不明。最优采样数量与任务复杂度的关系未建模
  - **[Post-2d] 跨域迁移的泛化性边界**: 验证数学推理 → 多模态推理的正迁移，但其他领域的迁移效果未知。哪些领域组合存在负迁移？

## 个人深度评注
<!-- 用户审阅后补充 -->
