---
title: "LaViDa-R1: Advancing Reasoning for Unified Multimodal Diffusion Language Models"
authors: [Shufan Li, Yuchen Zhu, Jiuxiang Gu, Kangning Liu, Zhe Lin, Yongxin Chen, Molei Tao, Aditya Grover, Jason Kuen]
date: 2026-02
venue: arxiv
url: "https://arxiv.org/abs/2602.14147"
tags: [rl, diffusion, unified-model, posttraining, alignment, reward-model]
category: rl/dllm-reasoning
level: 2
status: read
importance: high
problem_tree_nodes: [RL-2, RL-3, Uni-5, Diff-1b]
aliases: [LaViDa-R1]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结
提出 LaViDa-R1，一个面向多模态 Diffusion LLM (dLLM) 的统一后训练框架，通过将 SFT、online GRPO 和 self-distillation 统一为加权 policy gradient 形式，并引入 answer-forcing 和 tree search 解决 RL 训练信号消失问题，在 visual math reasoning、grounding 和 image editing 上均取得显著提升。

## 核心 Insight
dLLM 的双向生成能力（bidirectional inpainting）不仅是推理时的优势，更可以在 RL 训练中被利用——通过 answer-forcing 将 ground-truth answer 注入 masked sequence 末端，让 dLLM 反向填充 reasoning trace，从而在困难问题上也能获得有效训练信号。这本质上将 dLLM 的 inpainting 能力转化为一种 guided exploration 机制，是 AR 模型无法实现的。

## 与已有工作的关系
- **继承自**: [[LaViDa]] (NeurIPS 2025 Spotlight) → [[LaViDa-O]] (Elastic MoT 统一理解生成) → [[2026-LaViDa-R1|LaViDa-R1]] (后训练推理增强)
- **对比**: [[d1]] (dLLM RL, 单任务数学推理), [[2025-MMaDA|MMaDA]] (多模态 dLLM RL, UniGRPO), [[DiffuGRPO]] (语言 dLLM RL)
- **互补**: [[VLM-R1]] (AR VLM 的 RL reasoning), [[EditScore]] (image editing reward model), [[GoT]] (编辑推理数据)

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 整体流程
两阶段训练，基于 LaViDa-O (10.4B 参数) 作为基础模型：
- **Stage 1 (SFT)**: 100k steps, lr=5e-6, 在生成/理解/推理数据上做 SFT，重点引入推理数据 (CoT reasoning, 编辑推理, reason-intensive grounding)，推理数据与原始数据采样比为 7:3
- **Stage 2 (Unified Post-training)**: 5k steps, lr=5e-7, 统一 SFT + online GRPO + self-distillation，使用多种 reward (correctness, IoU, EditScore)

### 核心公式
统一目标函数：$J_{Unified}(\theta) = \frac{1}{N}\sum_{i=1}^{N} A_i \log \pi_\theta(y^i|x^i)$

不同的 $(y^i, x^i)$ 来源和 $A_i$ 权重定义了不同的训练目标：
- **SFT**: $(y^i, x^i) \sim D$, $A_i = 1$
- **Online GRPO**: $y^i \sim \pi_\theta(\cdot|x^i)$, $A_i = A_i^{GRPO}$ (normalized reward)
- **Self-distillation**: $y^i \sim \pi_\theta(\cdot|x^i)$, $A_i = 1$ 当 $i = \arg\max r$, 否则 $A_i = 0$
- **最终设计**: $A_i^{aggr} = 0.5 A_i^{distill} + 0.5 A_i^{GRPO}$, 同时混合 SFT batch

关键：去掉 KL regularization，用 SFT loss 作为隐式正则化替代品。

## Building Blocks（可复用组件）

### Block 1: 统一 Policy Gradient 后训练框架
- **做法**: 将 SFT、GRPO、self-distillation、Online DPO、SLiC 等后训练方法统一为加权 policy gradient 形式 $\sum A_i \log \pi_\theta(y^i|x)$，通过拼接不同来源的 (y, x, A) 三元组实现混合训练
- **机制 (WHY it works)**: 所有这些 loss 在 on-policy 设置下的梯度形式等价于 policy gradient。SFT 项替代 KL regularization 的作用——允许模型在 RL 阶段充分探索（不被 suboptimal reference model 约束），同时通过高质量 SFT 数据防止 collapse。去掉 reference model 还减少了 GPU 内存需求
- **适用条件**: 适用于任何可以表示为 weighted log-likelihood 的训练目标；需要有多种数据来源和 reward 信号
- **什么时候会 break**: (1) KL estimator 对高熵分布（如图像 token）不稳定，NLL > 6 导致 KL 项方差过大 → GRPO+KL 会 diverge；(2) 当 SFT 数据质量差时，SFT 正则化可能引入噪声
- **可组合方向**: 可扩展到更多 loss 类型（如 contrastive loss），可应用于 AR 模型的多目标后训练。参考 [[DeepSeek-R1]] 的 GRPO 设计

### Block 2: Answer Forcing (基于 dLLM Inpainting 的 Guided Exploration)
- **做法**: 当 online sampling 的 N 个 rollout 全部失败（reward < threshold τ=0.5）时，构造一个预填充 ground-truth answer 的 masked sequence "[M]...[M] \<answer\> z* \</answer\>"，利用 dLLM 的 inpainting 能力反向生成 reasoning trace。实际实现中总是并行生成 N+1 个 sample（1 个 answer-forced），按条件决定是否使用
- **机制 (WHY it works)**: dLLM 的双向上下文使其天然支持 text infilling——给定答案在末尾，模型可以利用全局信息填充推导过程。这解决了 RL 中"困难问题全部得零分导致训练信号消失"的核心难题。answer-forced sample 提供了正向训练信号，即使其推理链不完美
- **适用条件**: 需要 verifiable reward（0-1 correctness, IoU 等）且有 ground-truth answer 的任务
- **什么时候会 break**: (1) 注入比例过高（50%+）会导致 collapse——因为 forced sample 总是得高分，扭曲 advantage 分布；最优比例约 10%。(2) dLLM 的 infilling 质量有限时，生成的 reasoning trace 可能 ill-formed，产生误导性信号
- **可组合方向**: 可结合 [[IGPO]]（注入部分 reasoning trace）；可扩展到代码生成（给定测试用例输出反填代码推理过程）

### Block 3: Tree Search (基于 Diffusion 中间状态的分支搜索)
- **做法**: 利用 diffusion 的中间去噪状态实现搜索树。先生成 N 个 i.i.d. sample，找到 reward 最高的 sample，从其早期 diffusion state（如第 8 步/共 64 步）分支出 N 个新 sample，重复 k 次得到 Nk 个 sample。用于没有 ground-truth answer 但有 real-valued reward 的任务（如 image editing）
- **机制 (WHY it works)**: Diffusion 过程的中间状态保留了高层结构信息，从高 reward 轨迹的早期状态重启相当于"在有前景的方向上做更多探索"。与 AR 模型的 beam search 不同，dLLM 的 tree search 可以在全局结构层面分支。紧凑存储：只保存 (y_0, unmasking order v) 即可恢复任意时刻状态，开销 O(NL)
- **适用条件**: 需要 real-valued reward function（不适用于 0-1 reward，因所有样本为 0 时无法选最优）；diffusion 步数需足够多以提供有意义的中间状态
- **什么时候会 break**: (1) 从太晚的 diffusion step 分支（如 step 16/64）引入的不确定性太小，几乎等于复制原样本；最优分支点约 step 8/64。(2) 步骤过多（[0,8,16,32]）不比简单 [0,8] 更好——边际收益递减
- **可组合方向**: 可结合 [[MCTS]] 以引入更智能的节点选择策略；可推广到视频生成任务中在时序维度做分支

### Block 4: Complementary-Masking Likelihood Estimator (w=1)
- **做法**: 用两个互补 mask pattern 估计 dLLM 的数据 log-likelihood。采样 $t_1 \sim U[0,1]$，令 $t_2 = 1-t_1$，构造互补 mask：$y_{t_1}$ 和 $y_{t_2}$ 的 mask 位置互补（一个被 mask 的位置在另一个中未被 mask）。关键创新：使用 $w(t)=1$ 而非传统的 $w(t)=1/t$
- **机制 (WHY it works)**: 三个优势：(1) 互补 mask 保证所有 token 被覆盖一次，避免 i.i.d. sampling 遗漏重要 token；(2) $w(t)=1$ 避免了 $w(t)=1/t$ 导致的权重不平衡（当 $t_1=0.9, t_2=0.1$ 时权重比为 9:1，但哪些 token 被 mask 是随机的，不应有如此大差异）；(3) 相比 d1 的全 mask ($t=1$)，降低了 train-inference gap
- **适用条件**: 适用于任何基于 ELBO 的 dLLM 训练；尤其在视觉生成任务（大量 image token）中优势明显
- **什么时候会 break**: 当序列非常短时，互补 mask 与 i.i.d. sampling 差异不大
- **可组合方向**: 可作为任何 dLLM RL 方法（[[d1]], [[UniGRPO]], [[DMPO]] 等）的 drop-in likelihood estimator 替换

### Block 5: Multi-Task Multi-Reward RL for dLLMs
- **做法**: 在一个统一框架中同时对多种任务（数学推理、VQA、object grounding、image editing）做 RL，每个任务使用适当的 reward（0-1 correctness reward, IoU reward, EditScore VLM reward），配合各任务特定的生成长度（数学 512 tokens, grounding 128 tokens, editing 256 tokens）
- **机制 (WHY it works)**: 统一的 policy gradient 形式使得混合多种 task/reward 只需拼接 (y, x, A) 三元组。不同任务互相提供正则化效应——理解任务帮助维持推理能力，生成任务也从推理链中获益
- **适用条件**: 需要为每种任务设计合适的 reward；不同任务的序列长度和 token 类型（text vs image）需分别处理
- **什么时候会 break**: (1) 任务间的梯度冲突可能降低某些任务的性能；(2) image editing 的 reward evaluation 是瓶颈（EditScore 评估 256 张图需 70-140s），限制了训练吞吐
- **可组合方向**: 扩展到视频理解/生成、多轮对话、code generation 等更多模态和任务

## Anti-patterns / 已知失败模式
- **Answer Forcing 注入比例过高导致 collapse**: 100% 注入时模型 collapse（MathVista 仅 4.1），因为被强制的 sample 总是高 reward，其它 sample 的 advantage 被系统性压低。最优注入率约 10%。[Critic 注] 10% 几乎必然是 task/model-specific 的经验值——取决于任务难度分布（困难问题占比）、group size N、infilling 质量。更 principled 的方案是 adaptive injection + importance weight 修正
- **KL regularization 在高熵 image token 下失效**: image generation 的 NLL > 6（text < 2），导致 KL estimator 方差极大，GRPO+KL 训练发散。[Critic 注] 根本原因不仅是估计质量，而是 KL 约束本身对 dLLM 不合适——reference model 是 suboptimal starting point，约束 policy 不偏离它反而限制探索
- **w(t)=1/t 权重在 image editing 任务中严重失衡**: 因 image editing 产生 4096 image tokens + 256 text tokens，大量 token 的权重不平衡被放大
- **T2I generation 的现有 reward model 不支持 reasoning**: PickScore (CLIP-based) 无推理能力，UnifiedReward-Qwen-7B 会幻觉出错误评判标准，无法评估需要逻辑推理的 T2I prompt
- **从过晚的 diffusion step 分支无效**: tree search 从 step 16+ 分支时不确定性太低，生成的新 sample 基本重复。[Critic 注] step 8/64 的选择几乎必然是 task-dependent——image editing 的"全局→局部"符合 12.5%，但数学推理中关键转折可能在中间步骤
- **[Critic 新增] SFT 正则化的隐含风险**: SFT 替代 KL 假设 SFT 数据分布是"好的"正则化目标。若 SFT 数据有系统性偏差（hallucination/格式偏差），SFT 正则化会将模型拉向这些有偏方向。此外 SFT 正则化强度间接取决于 batch 比例，不如 KL 的 β 系数灵活
- **[Critic 新增] Answer Forcing 可能教出 post-hoc rationalization**: dLLM 给定答案反填推理链时，可能生成"看起来对但推理链不严谨"的跳跃式推理，本质是 pattern matching 而非真正推导

## 实验关键发现
- **LaViDa-R1 在所有任务上超越 SFT baseline**: MathVista 60.0 (+2.4 over SFT), Lisa-Grounding mIoU 60.0 (+23.1 over SFT), ImgEdit 3.90 (+0.09 over SFT)
- **SFT 在 image editing 上接近饱和**: SFT 只带来 +0.01 提升，RL 带来 +0.10，说明 RL 能探索 SFT 数据覆盖不到的模式
- **语言任务的提升最大**: GSM8K 81.5 (+10.9 over base), Math500 38.6 (+15.2 over base)，因为 base model 是视觉为主的预训练，语言能力有较大提升空间
- **LaViDa-R1 在 Lisa-Grounding 上超越 [[VLM-R1]] 和 specialist models**: mIoU 60.0 vs [[VLM-R1]] "–"，P@0.5 66.7 vs LISA-7B 49.4
- **统一 loss (GRPO+SFT) 比 GRPO+KL 和纯 GRPO 都更稳定**: 图5显示纯 GRPO 和 GRPO+KL 都发散，统一 loss 稳定收敛到更高 reward
- **Self-distillation 混合 (γ=0.5) 效果最佳**: 纯 GRPO (γ=0) 或纯 self-distillation (γ=1.0) 都不如混合
- **dLLM 与 SOTA AR MLLM 仍有差距**: 例如 Qwen3-VL-8B 在 Lisa-Grounding 上的 P@0.5 为 62.4 vs LaViDa-R1 的 66.7（LaViDa-R1 更优），但在 MathVista 等其他任务上 AR 模型更强

## Relations (结构化)
<!-- type: extends | alternative_to | combines_with | motivated_by | enables | conflicts_with -->
- `extends` → [[LaViDa-O]]: 在 LaViDa-O 基础上添加统一后训练框架，LaViDa-O 提供理解+生成基础能力
- `extends` → [[LaViDa]]: 继承 complementary masking 技术，从预训练拓展到 RL likelihood estimation
- `alternative_to` → [[d1]]: 同为 dLLM RL，但 LaViDa-R1 支持多任务多模态，d1 仅单任务数学推理
- `alternative_to` → [[2025-MMaDA]]: 同为多模态 dLLM RL；LaViDa-R1 引入 answer-forcing（解决训练信号消失）、tree search（diffusion 中间状态搜索）、统一 SFT-RL loss（替代 KL 正则化）三个 MMaDA 未解决的问题；LaViDa-R1 晚 9 个月，以 MMaDA 为基线
- `combines_with` → [[EditScore]]: 使用 EditScore 作为 image editing 的 reward model
- `motivated_by` → [[DeepSeek-R1]]: 将 LLM reasoning (GRPO) 的思路扩展到 dLLM
- `alternative_to` → [[2026-EBPO]]: 两者解决训练信号消失的方式正交——LaViDa-R1 用 answer-forcing 生成正向样本（dLLM inpainting），EBPO 用 shrinkage baseline 从全失败 group 提取负向梯度（统计方法）；可叠加使用
- `combines_with` → [[2025-dMLLM-TTS]]: dMLLM-TTS 的 test-time scaling（HTS 搜索）可在 LaViDa-R1 训练后进一步提升推理质量；LaViDa-R1 的 tree search 是训练时探索，dMLLM-TTS 是推理时搜索
- `combines_with` → [[2025-SDAR-VL]]: SDAR-VL 的块状扩散训练稳定性优化（ABNS/EMRS/PBNC）可迁移到 LaViDa-R1 骨干的预训练阶段
- `alternative_to` → [[2025-KimiK2.5]]: 多模态 RL 方案对比——dLLM RL (answer-forcing + complementary masking) vs AR RL (GRM + Toggle RL)；K2.5 在 AR MoE 上独立验证跨模态正迁移
- `alternative_to` → [[2025-SPG]]: 改善 dLLM likelihood 估计的不同维度——complementary masking 降低 MC 方差（覆盖范围），SPG block-wise masking 对齐推理结构（分布匹配）+ 三明治 bounds 解决负 advantage 偏差；两者可组合
- `combines_with` → [[2026-StableDRL]]: StableDRL 解决梯度尖峰不稳定（GRPO 公式鲁棒化），LaViDa-R1 的 answer-forcing 解决训练信号消失（探索失败）——两个正交训练病理的正交修复，可组合为三层稳定性栈的 Layer 1+3

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- dLLM 的 RL 训练如何扩展到多任务多模态场景（之前仅限于单任务数学推理）
- dLLM RL 中 KL regularization 对高熵 image token 失效的问题
- dLLM RL 中困难问题的训练信号消失问题
- dLLM 的 likelihood estimation 在 RL 训练中权重不平衡问题

### 未解决的问题
- 问题: dLLM 与 SOTA AR MLLM 在推理能力上仍有差距
  - 为什么难: 根源可能在预训练阶段——dLLM 的 pretraining data 和 compute 远不如 AR 模型充分
  - 潜在思路: 扩大 dLLM 预训练规模（LLaDA 2.0 已探索 100B）、改进 pretraining 数据质量
- 问题: T2I generation 缺乏支持 reasoning 的 reward model
  - 为什么难: 现有 reward model（CLIP-based 或 VLM-based）不具备 compositional reasoning 能力
  - 潜在思路: 训练专门针对 compositional/reasoning prompt 的 reward model，或使用 frontier VLM (GPT-4.1) 但成本高
- 问题: RL 训练效率瓶颈——reward evaluation 耗时
  - 为什么难: VLM-based reward model (EditScore) 评估 256 images 需 70-140s
  - 潜在思路: reward model 蒸馏、异步 reward 评估、batch 化

### 对问题树的推进
- 推进了 [[problem-tree#RL-2 🔴 GRPO/PPO 如何适配多模态？|RL-2]]: 展示了 GRPO 适配多模态 dLLM 的具体方案，解决了 KL 失效和训练信号消失的问题
- 推进了 [[problem-tree#RL-3 🔴 RL 能提升多模态推理能力吗？|RL-3]]: 证明 RL 能提升 dLLM 的视觉推理能力（MathVista, MathVerse, Lisa-Grounding）
- 推进了 [[problem-tree#Uni-5 🔴 统一模型的 Post-training 和 RL|Uni-5]]: 提供了统一模型的后训练和 RL 方案，同时提升理解和生成
- 推进了 [[problem-tree#Diff-1b 🔴 连续扩散 vs 离散扩散 (Masked Diffusion)|Diff-1b]]: 进一步验证了离散扩散 (Masked Diffusion) 在统一多模态模型中的可行性
- 新增问题: T2I reasoning reward model 的缺失是 dLLM RL 扩展到纯生成任务的关键瓶颈

## 个人深度评注
<!-- 留待用户审阅后补充 -->

### [Critic] 增量贡献评估
| Block | 新颖性 | 理由 |
|-------|--------|------|
| Block 1: 统一 PG 框架 | 中等 | 技术组合创新，SFT 替代 KL 在 AR 领域已有讨论 |
| Block 2: Answer Forcing | **高** | 利用 dLLM 独有 inpainting 能力的原创方法，AR 模型无法复制 |
| Block 3: Tree Search | 中高 | 将 diffusion 中间状态转化为搜索策略，概念新颖 |
| Block 4: Complementary Masking w=1 | 低中 | 继承自 LaViDa，w=1 更像实验发现非理论创新 |
| Block 5: Multi-Task RL | 低 | 标准多任务学习范式的首次 dLLM 应用 |

### [Critic] 关键隐含假设
1. **梯度尺度兼容性**: 不同来源的 (y,x,A) 三元组直接拼接，假设梯度量级兼容——但 GRPO 经 normalize 而 SFT 的 A_i=1，信噪比不同
2. **任务间正迁移 > 负干扰**: 多任务 RL 假设理解和生成互相帮助，但缺乏梯度冲突分析
3. **On-policy 近似**: dLLM 的多步去噪使 off-policy bias 可能比 AR 更严重
4. **Reward 尺度兼容性**: 0-1/IoU/EditScore 数值范围差异可能导致某些任务主导梯度

### [Ideator] 三线交汇定位
本文处于三条技术发展线的交汇点:
1. **dLLM 发展线**: [[MDLM]] → [[LLaDA]] → [[LaViDa]] → [[LaViDa-O]] → **LaViDa-R1** (补上 "后训练" 缺环)
2. **多模态 RL/Reasoning 线**: GRPO([[DeepSeek-R1]]) → [[VLM-R1]]/[[d1]] → [[UniGRPO]] → **LaViDa-R1** (多任务多模态 dLLM RL)
3. **统一模型后训练线**: MLLM SFT → MLLM DPO → Unified SFT → **LaViDa-R1** (首个统一模型 RL)

核心学术价值: **将 dLLM 的固有属性 (bidirectional inpainting, diffusion 中间状态) 转化为 RL 训练的独特优势**，而非简单移植 AR 方法。

### [Ideator] 新暴露的 Open Questions (建议加入问题树)
1. **[RL-2c] 高熵 token 分布下的 RL 正则化**: KL 对 image token 失效，SFT 替代有效但理论不完备
2. **[RL-2d] dLLM guided exploration 的理论与泛化**: Answer Forcing 注入率的理论解、无 ground-truth 场景扩展
3. **[Diff-3] Diffusion 中间状态的语义结构与利用**: 不同 step 对应什么层次的决策？MCTS for dLLM？dLLM PRM？
4. **[RL-1d] T2I Reasoning Reward Model**: 现有 reward model 均不支持 compositional reasoning

### [Ideator] 潜在研究方向
1. **dLLM Process Reward Model + MCTS**: 训练评估 diffusion 中间状态质量的 PRM，引入 MCTS 搜索 (风险: 中间状态语义结构不明)
2. **Answer Forcing → dLLM Self-Play 推理引擎**: 先猜答案 → 反填推理链 → 评估推理链质量验证答案可靠性，实现 abductive reasoning (风险: 推理质量不足导致 self-verification 不可靠)
3. **dLLM Self-Evaluation Reward for T2I**: 利用统一模型自身的理解能力评估生成质量，bootstrapped reward (风险: 循环偏差/reward hacking)

### [Ideator] dLLM 路线的根本性限制判断
- **非致命性限制**: 不存在阻止 dLLM 持续进步的根本障碍
- **结构性劣势**: (1) sequential reasoning 能力——全局并行去噪不匹配 step-by-step 推理; (2) 推理效率——多步大量前向传播 vs AR+KV-cache; (3) 生态不成熟
- **最佳定位**: 理解-生成统一 (unique selling point) + 利用 bidirectional 特性的新能力 + 图像/视频为主的任务
- **不建议**: 试图让 dLLM 在纯文本 sequential reasoning 上追赶 AR——以己之短攻彼之长
