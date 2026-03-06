---
title: "LaViDa-O: Elastic Large Masked Diffusion Models for Unified Multimodal Understanding and Generation"
authors: [Shufan Li, Jiuxiang Gu, Kangning Liu, Zhe Lin, Zijun Wei, Aditya Grover, Jason Kuen]
date: 2025-09
venue: arxiv
url: "https://arxiv.org/abs/2509.19244"
tags: [diffusion, unified-model, architecture, pretraining, moe, generation, understanding]
category: unified-model/diffusion-native
level: 3
status: read
importance: high
problem_tree_nodes: [Uni-1, Uni-2a, Uni-2b, Uni-3, Diff-1b, Diff-1c, Tok-2d]
aliases: [LaViDa-O, Lavida-O, Elastic-MoT]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结
提出 LaViDa-O，基于 Elastic Mixture-of-Transformers (Elastic-MoT) 架构的统一 masked diffusion 模型，通过将轻量生成分支 (2.4B) 与大容量理解分支 (8B) 弹性耦合，结合 modality-aware masking、stratified sampling、planning & reflection 机制，在 10.4B 规模实现图像理解、object grounding、1024² T2I 生成和图像编辑的多任务统一，是 LaViDa-R1 的基础模型。

## 核心 Insight
理解和生成对模型容量需求不对称——理解需要大模型（语义推理），生成需要的模型容量相对较小但需要专用参数。通过 Elastic-MoT 让不同任务激活不同参数子集（8B/6.4B/10.4B），既避免了标准 MoT 的参数翻倍问题，又保留了模态专用参数的好处。此外，利用统一模型自身的理解能力（planning + reflection）显式提升生成质量，形成理解→生成的正反馈循环。

## 与已有工作的关系
- **继承自**: [[LaViDa]]（NeurIPS 2025 Spotlight，提供 8B 理解基座 + Complementary Masking 技术）
- **直接竞争**: [[2025-MMaDA]]（同为 dLLM 统一模型；LaViDa-O 通过非对称 Elastic-MoT 解决容量不匹配，通过 Planning 显式解决 MMaDA 的 compositional generation 弱点；MMaDA 贡献重心在 UniGRPO RL，LaViDa-O 贡献重心在架构设计和预训练）
- **使能**: [[2026-LaViDa-R1]]（LaViDa-O 是 LaViDa-R1 的直接基础模型；Coordinate Quantization → IoU reward，Planning+Reflection → self-evaluation reward，Elastic-MoT → 更稳定的多任务 RL 基础）
- **灵感来源**: [[Transfusion]]（MoT 概念的原始提出者，LaViDa-O 改造为非对称 Elastic 版本）
- **技术序列**: LaViDa → **LaViDa-O** → LaViDa-R1（三级管线中的中间环节）

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 整体架构
- **骨干**: 基于 LaViDa 的 masked diffusion transformer
- **参数分配**: 理解分支 8B + 生成分支 2.4B = 总计 10.4B
- **图像 tokenizer**: VQ-Encoder（基于 VQGAN），支持离散化到 1024² 分辨率
- **视觉编码器**: SigLIP（来自 LaViDa，用于理解任务的视觉嵌入）
- **统一目标**: 所有模态共享 cross-entropy mask prediction loss

### Elastic-MoT 设计
- 32 层 transformer，前 16 层全模态联合 attention，后 16 层模态专用 self-attention
- 弹性参数激活:
  - 理解 only → 8B（全 32 层理解参数）
  - 生成 only → 6.4B（2.4B 生成参数 + 前 16 层 4B 共享参数）
  - 交叉任务 → 10.4B（全参数）
- 相比标准 MoT，训练加速 3.17×

### 三阶段训练
| 阶段 | 内容 | 数据 |
|------|------|------|
| Stage 1 | 理解能力扩展 | RefCOCO + image-level 理解数据 |
| Stage 2 | 生成预训练 | 大规模图文对，分辨率渐进 256→512→1024 |
| Stage 3 | 联合端到端训练 | 混合全任务数据 |

### 推理特性
- **分辨率**: 支持 1024×1024 图像生成（vs MMaDA 512×512, 其他 MDM 256/512）
- **推理加速**: Grounding 任务比 Qwen2.5-VL-7B 快 6.8×（得益于并行坐标解码）
- **Planning**: 先生成 layout bbox，再按 layout 生成图像
- **Reflection**: 利用理解能力自评生成结果，不满意则重新生成

## Building Blocks（可复用组件）

### Block 1: Elastic Mixture-of-Transformers (非对称模态分支)
- **做法**: 为理解和生成设计不同容量的 transformer 分支——理解分支 8B，生成分支仅 2.4B。前 16 层共享联合 attention（允许跨模态交互），后 16 层分离为模态专用 self-attention。不同任务激活不同参数子集
- **机制 (WHY it works)**: 理解任务需要大量语义推理能力（需要 8B 规模），而图像生成的去噪过程相对"简单"（条件扩散无需推理，2.4B 即可）。非对称设计避免了标准 MoT 的参数翻倍问题，同时保留了模态专用参数的好处。前半部分的联合 attention 提供必要的跨模态交互，后半部分的分离 attention 避免了模态间的梯度干扰
- **适用条件**: 需要理解和生成能力不对称的统一模型场景；需要前期有一个训好的理解基座（如 LaViDa）
- **什么时候会 break**: (1) 如果生成任务复杂度接近理解（如复杂 composional T2I），2.4B 可能容量不足；(2) 联合/分离 attention 的分界层（16/32）是超参，不同模型规模下最优值不同；(3) 弹性激活增加了工程实现复杂度
- **可组合方向**: 与 MoE routing 结合实现更细粒度的参数分配（per-token 级别）；扩展到视频模态（理解/生成/时序各一个分支）

### Block 2: Modality-Aware Masking (动态路由机制)
- **做法**: 引入特殊 `[exp]` text token 和对应时间步 t_exp。训练时在 t_exp 时将图像 VQ token 序列折叠为单个 `[exp]` token；推理时当模型生成 `[exp]`，将其替换为 L_img 个图像 mask token，后续步骤路由至生成分支
- **机制 (WHY it works)**: 解决了 MDM 中的关键问题——何时从纯文本预测切换到图像生成。`[exp]` token 作为"展开信号"，让模型自主决定图像生成的发起时机和位置，避免了手动指定输出模态的硬编码。训练时的序列长度变化（折叠/展开）创造了自然的变长序列训练信号
- **适用条件**: 需要交叉模态生成（interleaved generation）的场景；需要模型自主决定何时生成图像
- **什么时候会 break**: (1) 当图像 token 数量巨大（高分辨率/视频）时，展开后序列长度暴增；(2) `[exp]` 的位置预测不准确时导致图像生成位置错误
- **可组合方向**: 扩展到视频（`[exp_video]`）、音频等多模态展开；与 speculative decoding 结合加速展开决策

### Block 3: Stratified Random Sampling (空间分散去噪)
- **做法**: 将图像 token 划分为层次化网格 (2^d × 2^d, d=1,2,...,log₂N)，每步从每个网格单元中采样一个 token 进行 unmask，确保 unmask 在空间上均匀分散
- **机制 (WHY it works)**: MDM 的推理假设 token 间条件独立，但基于置信度的 unmasking 倾向于在空间上聚集（相邻 patch 语义相关→置信度相似），违反独立性假设。分层网格采样强制空间分散，更好地满足独立性假设。此外，空间分散提供了更均衡的全局上下文，避免了"先填角落后填中心"的偏差
- **适用条件**: 二维结构化数据（图像、视频帧）；token 间存在空间相关性的场景
- **什么时候会 break**: (1) token 网格不规则时（如 variable-resolution）分层采样不易定义；(2) 对一维序列（纯文本）无意义
- **可组合方向**: 与 confidence-based 方法组合（在每个网格内按置信度选择）；扩展到 3D volume（视频时空）

### Block 4: Planning + Reflection (理解驱动的生成增强)
- **做法**: Planning——模型先生成目标 layout（bounding boxes），再根据 layout 生成图像。Reflection——生成后用理解能力自评是否满足 prompt 要求，不满足则重生成。GenEval 从 0.77 → 0.85 (planning) → 0.89 (reflection)
- **机制 (WHY it works)**: 利用统一模型的理解能力显式指导生成过程，形成 generate → evaluate → refine 循环。Planning 将复杂的 composional 生成分解为两步——先规划空间布局（利用理解能力推理对象关系），再在约束下生成（降低生成难度）。Reflection 通过自评估提供无需外部 reward model 的质量检查
- **适用条件**: 统一模型（同时具备理解和生成能力）；composional 或需要空间推理的生成任务
- **什么时候会 break**: (1) 理解能力本身不足时，planning/reflection 产生的指导也不准确；(2) 多次 reflection 增加推理开销；(3) 简单 prompt 下 planning 的 layout 生成是不必要的额外开销
- **可组合方向**: 与 RL reward（[[2026-LaViDa-R1]] 的 self-evaluation reward）结合做训练时优化；扩展到多轮迭代编辑（plan-edit-reflect 循环）

### Block 5: Universal Text Conditioning (自然语言微参数)
- **做法**: 将传统的微条件嵌入（resolution、crop、aesthetic score、HPS、luminance、contrast）替换为自然语言字符串 "[KEY]: [VALUE]"，附加到 prompt 后。训练时随机 drop 各条件
- **机制 (WHY it works)**: 利用 LLM 的文本理解能力处理条件信息，无需设计专用的 condition embedding 模块。随机 drop 实现 classifier-free guidance 的效果，同时让模型学会处理不同条件子集的组合
- **适用条件**: 以 LLM 作为骨干的生成模型；条件参数可以用自然语言描述
- **什么时候会 break**: (1) 数值精度要求高的条件（如精确坐标）可能不如专用嵌入准确；(2) text token 数增加了序列长度成本
- **可组合方向**: 扩展到更多细粒度控制参数（style、LoRA weight 等）；与 prompt engineering 技术结合

### Block 6: Coordinate Quantization (并行 Bbox 预测)
- **做法**: 将 bbox 坐标归一化到 [0,1] 后量化为 1025 个离散 bin (0/1024 到 1024/1024)，每个 bbox 用 4 个 token 表示 (x_min, y_min, x_max, y_max)，支持并行解码多个 bbox
- **机制 (WHY it works)**: 将连续坐标转为离散 token 使其适配 MDM 的 mask-predict 框架，且多个 bbox 可并行预测而非逐个自回归生成。量化精度（1/1024 ≈ 0.001）对大多数 grounding 任务足够
- **适用条件**: 需要预测空间坐标的任务（grounding、detection）；在 MDM/dLLM 框架内
- **什么时候会 break**: (1) 极高精度需求（如像素级分割）时量化误差不可忽略；(2) 当 bbox 数量很大时（如密集检测）token 数线性增长
- **可组合方向**: 扩展到 segmentation mask 的离散化表示；与 set prediction（DETR-style）结合

## Anti-patterns / 已知失败模式
- **标准 MoT 参数翻倍**: 为每个模态创建同等大小的分支导致参数量和计算成本翻倍，Elastic-MoT 的非对称设计更高效
- **Confidence-based unmasking 的空间聚集**: 基于置信度选择 unmask token 导致空间上聚集，违反 MDM 独立性假设，degrading 生成质量
- **手动指定输出模态**: 硬编码何时生成图像/文本，限制了 interleaved 生成的灵活性，modality-aware masking 的 `[exp]` 机制更通用
- **单一输出分辨率训练**: 仅在固定分辨率训练限制模型泛化，渐进分辨率训练 (256→512→1024) 更有效
- **[Critic 新增] 非对称分支的梯度不平衡**: 联合训练时 8B 理解梯度与 2.4B 生成梯度在共享前 16 层存在量级差异，可能导致共享层优化偏向理解方向
- **[Critic 新增] Planning+Reflection 在创意任务上的过度约束**: 先固定 bbox 布局再生成，在需要整体美学统一的创意任务上可能导致风格不连贯
- **[Critic 新增] `[exp]` token 模态决策崩溃风险**: 如果训练数据中生成任务比例过低（理解:生成 > 10:1），模型可能永远不激活生成分支

## 实验关键发现
- **多任务 SOTA**: 在 MDM 类模型中首次同时达成 image understanding (MMMU 45.1)、grounding (RefCOCO P@0.5 92.3)、T2I (GenEval 0.89 w/ reflection)、image editing (Overall 3.80 w/ planning)
- **1024² 高分辨率生成**: 首个支持 1024×1024 像素生成的 MDM 统一模型
- **Planning 大幅提升 compositional 能力**: GenEval 0.77 → 0.85 (+10.4%)，说明理解能力可以显式提升生成
- **Reflection 进一步提升**: GenEval 0.85 → 0.89，自评估机制有效
- **推理加速**: Grounding 任务比 AR 模型 (Qwen2.5-VL-7B) 快 6.8×，得益于并行坐标解码
- **Elastic-MoT 训练加速 3.17×**: 验证了非对称分支设计的效率优势
- **超越 GPT-4o on 部分编辑任务**: Object replacement (4.39) 和 removal (3.98)

## Relations (结构化)
<!-- type: extends | alternative_to | combines_with | motivated_by | enables | conflicts_with -->
- `extends` → [[LaViDa]]: 在 LaViDa 8B 理解基座上添加 Elastic-MoT 生成分支和多任务能力
- `enables` → [[2026-LaViDa-R1]]: 作为 LaViDa-R1 的基础模型 (10.4B)，提供理解+生成预训练能力；Planning/Reflection 是 LaViDa-R1 self-evaluation reward 的前体
- `alternative_to` → [[2025-MMaDA]]: 同为 dLLM 统一模型；LaViDa-O 用 Elastic-MoT 非对称分支 + 1024² vs MMaDA 的模态无关架构 + 512²；LaViDa-O 通过 Planning 显式解决 MMaDA GenEval Position 弱点 (0.20)
- `motivated_by` → [[Transfusion]]: 受 MoT 概念启发，改造为非对称 Elastic 设计解决参数翻倍问题
- `motivated_by` → [[VQGAN]]: 使用基于 VQGAN 的 VQ-Encoder 支持 1024² 离散化
- `combines_with` → [[SigLIP]]: 使用 SigLIP 视觉编码器提供理解任务的视觉嵌入
- `enables` → [[2025-Sparse-LaViDa]]: Sparse-LaViDa 直接基于 LaViDa-O 10.4B checkpoint，通过稀疏化和 KV 缓存实现 1.95-2.83× 推理加速
- `alternative_to` → [[2026-Beyond-LM]]: 容量分配哲学对立——Elastic-MoT 手动设计非对称分支（理解 8B + 生成 2.4B）vs Beyond-LM MoE per-modality shared experts 自动分化

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- 首次在 dLLM 统一模型中引入 MoE/Routing 解耦设计（Elastic-MoT），证明非对称参数分配（理解 8B + 生成 2.4B）的可行性和效率优势（3.17× 训练加速），为 [[problem-tree#Uni-2b]] 提供首个实证回答
- 将 MDM 统一模型的生成分辨率从 512² 提升到 1024²，是首个达到此分辨率的 masked diffusion 统一模型
- 通过 Planning+Reflection 机制将理解能力显式转化为生成质量提升（GenEval 0.77→0.89），为"理解→生成正反馈循环"建立了方法论范式

### 未解决的问题
- 问题: Elastic-MoT 的最优非对称比例（8B:2.4B）是否为最优？
  - 为什么难: 需要在固定总参数量下系统扫描比例空间，计算成本高；最优比例可能是任务集合的函数
  - 潜在思路: 在较小规模（3B 总参）做预实验验证趋势，构建 understanding-generation Pareto 前沿
- 问题: `[exp]` token 的模态决策在歧义输入下的鲁棒性未经验证
  - 为什么难: 自然语言 prompt 的歧义性使"是否应该生成图像"的决策边界模糊
  - 潜在思路: 构造边界案例测试集，评测 `[exp]` 激活率分布
- 问题: Planning+Reflection 的自评估质量与人类偏好的对齐程度未知
  - 为什么难: 模型用同套参数生成和评估，存在生成-评估耦合（可能倾向接受自己的生成）
  - 潜在思路: 引入外部 VLM 做对比评估；LaViDa-R1 的 RL 可能部分通过外部 reward 修正此偏差
- 问题: 非对称分支的共享层（前 16 层）在联合训练中的梯度平衡问题
  - 为什么难: 8B 和 2.4B 分支的梯度量级不同，共享层优化可能被理解方向主导
  - 潜在思路: 梯度归一化、动态 loss 加权、分阶段冻结策略

### 对问题树的推进
- 推进了 [[problem-tree#Uni-2b]] 🔴→🟡: Elastic-MoT 是首个 dLLM 框架内的 MoE/Routing 解耦实证，证明非对称分支可行且高效
- 推进了 [[problem-tree#Uni-2a]] (补充新维度): LaViDa-O 表明"共享有益但非最优"——路由解耦可在不破坏协同的前提下提升效率（MMMU 45.1 vs MMaDA 30.2）
- 推进了 [[problem-tree#Uni-3d]]: Planning+Reflection 在图像编辑上部分超越 GPT-4o（replacement 4.39, removal 3.98），将"理解驱动生成"从理论推进到实证
- 推进了 [[problem-tree#Uni-3a]]/[[problem-tree#Uni-3b]]: 1024² 高分辨率 + FID 6.68 将 dLLM 与 FLUX/SD3 的差距进一步缩小
- 推进了 [[problem-tree#Diff-1b]] (强化): 离散 MDM 在 1024² 分辨率 + 多任务场景下仍可达 competitive
- 推进了 [[problem-tree#Diff-1c]]: Stratified Sampling 提供非"减少步数"的采样效率思路；Coordinate Quantization 使 grounding 加速 6.8×
- 推进了 [[problem-tree#Uni-2c]] (部分): 三阶段课程学习（理解扩展→渐进分辨率生成→联合端到端）为训练配比提供实践参考
- 新增问题: [Uni-2b-sub] 弹性激活的 routing collapse 风险——在什么条件下 `[exp]` 路由会退化？
- 新增问题: [Uni-3e] Planning+Reflection 的边界条件——哪类任务能提升、哪类无效甚至有害？
- 新增问题: [Diff-3a] Stratified Sampling 的理论基础——空间分散改善质量的机制是什么？能否推广到视频时空维度？

## 个人深度评注
<!-- 留待用户审阅后补充 -->

### [Critic] 增量贡献评估
| Block | 新颖性 | 理由 |
|-------|--------|------|
| Block 1: Elastic-MoT | 中高 | 非对称参数分配 + 弹性激活在 MDM 统一框架下是新颖的，但"共享底层+专用顶层"思路在 VL 模型中有先例 |
| Block 2: Modality-Aware Masking | 中 | `[exp]` token 路由本质是可学习的条件控制信号，但将其应用于 MDM 的 interleaved 生成有设计价值 |
| Block 3: Stratified Sampling | 中低 | 空间分散 unmask 启发式有效，但类似思路在 MAGE/MaskGIT 中已有体现 |
| Block 4: Planning+Reflection | 中高 | 理解驱动生成的系统性应用在 MDM 中较少见；GenEval 0.77→0.89 是最重要的实证贡献 |
| Block 5: Universal Text Conditioning | 低 | 编码格式选择，非算法创新 |
| Block 6: Coordinate Quantization | 低中 | 坐标离散化+并行解码在 grounding 任务中已有先例 (Pix2Seq 等) |

### [Critic] 关键隐含假设
1. **理解/生成容量非对称假设**: 8B 理解 + 2.4B 生成假设图像生成相对"简单"——但复杂 compositional T2I（多对象精确空间关系）可能需要更大生成分支
2. **前 16 层联合 attention 足以建立模态对齐**: 后 16 层分叉可能导致高层表征不一致，编辑任务 Overall 3.80 vs GPT-4o 4.20 的差距可能部分源于此
3. **Planning+Reflection 的串行依赖**: layout 规划错误会在生成阶段放大（error compounding），reflection 能否纠正 planning 错误是未知的
4. **Reflection 自评估无偏差**: 同一模型生成和评估，可能存在自我接受偏差（sycophancy）

### [Critic] 机制层深度分析

**Elastic-MoT vs MMaDA 的 tradeoff 本质:**

这是两种不同归纳偏置的选择——MMaDA 假设"理解与生成共享同一表征空间已足够"，LaViDa-O 假设"理解和生成在高层需要专用能力"。MMaDA GenEval Position 0.20 证明了全共享假设在空间推理上失败；LaViDa-O 的非对称分支通过 Planning 显式利用理解能力解决了此问题。

**Elastic-MoT vs 标准 MoE:**
| 维度 | Elastic-MoT | 标准 MoE |
|---|---|---|
| 路由粒度 | 任务级（硬路由，`[exp]` 触发） | Token 级（软路由/Top-K） |
| 专家数量 | 2 个 | 8-64 个 |
| 负载均衡 | 不需要（模态频率天然决定） | 需要 auxiliary loss |
| 参数激活比 | ~62-77% | ~25-50% |

[推测] "Mixture-of-Transformers" 命名更接近 Conditional Activation Network，与 MoE 的概念亲缘性有限。

**Planning+Reflection 与 Test-Time Scaling 的关系:**

Planning+Reflection 是 test-time scaling 的结构化形式。Reflection 使计算量约 2-3×（生成+评估+可能重生成），换来 GenEval +4.7%。LaViDa-R1 将其升级为 Tree Search（全局搜索而非局部贪心），说明作者自己已识别 Reflection 是 TTS 的中间状态。

### [Connector] LaViDa-O vs MMaDA 核心设计对比
| 维度 | LaViDa-O (Elastic-MoT) | MMaDA (模态无关) |
|---|---|---|
| 架构哲学 | 非对称专用分支，按模态分配容量 | 全参数共享，统一处理所有模态 |
| 图像 Tokenizer | VQGAN-based，支持 1024² | MAGVIT-v2，512² 上限 |
| 生成质量 | GenEval 0.89 (w/ reflection) | GenEval 0.84 |
| 理解能力 | MMMU 45.1 | MMMU 30.2 |
| 贡献重心 | 架构设计 + 预训练 | 后训练 RL (UniGRPO) |
| 空间推理 | Planning 显式解决 | Position 0.20，未解决 |

### [Connector] 技术谱系定位
```
路线 A (北大+Princeton, 模态无关全共享)
LLaDA + MAGVIT-v2 → MMaDA (2025-05, NeurIPS)

路线 B (Adobe+UCLA, 非对称专用架构)
LaViDa (NeurIPS 2025, 8B 理解基座)
  ↓
LaViDa-O (2025-09, arxiv) ← 本文
  ↓
LaViDa-R1 (2026-02, arxiv, RL 后训练)
```
两条路线汇聚: 共同确立 dLLM 统一模型是可扩展的 MLLM 路线。LaViDa-O 定位为"实用部署级统一模型"——Elastic-MoT 的效率优势和 1024² 能力是另两篇无法直接复制的。

### [Ideator] 潜在研究方向
1. **非对称 MoE Scaling Study**: 固定总参数量，系统扫描理解:生成分支比例（1:1、2:1、4:1、8:1），构建 Pareto 前沿。动机: LaViDa-O 的 8:2.4 比例是否最优未知。依赖: Elastic-MoT (LaViDa-O)、MMaDA 对称基线。风险: 搜索空间大但可用小规模预实验。可行性: 中等
2. **Compositional Reward Model 训练**: 用 Planning 推理链（显式分解 compositional 关系）作为监督信号训练 reward model，解决 [[problem-tree#RL-1d]] T2I Compositional Reasoning Reward 空白。依赖: Planning (LaViDa-O) + UniGRPO/Complementary Masking (MMaDA/LaViDa-R1)。风险: Planning 推理链质量不稳定。可行性: 中等偏高
3. **Stratified Temporal-Spatial Unmasking for Video MDM**: 将 Stratified Sampling 扩展到 3D 时空，时间轴也强制 unmask 分散，加入关键帧优先 unmask 策略。动机: Diff-1c + Uni-4a。风险: 视频 MDM 基础设施不成熟。可行性: 较低但高价值
