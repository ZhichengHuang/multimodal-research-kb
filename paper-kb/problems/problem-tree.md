# 研究问题树

> 每个节点格式: [ID] 问题描述
> 状态标记: 🔴 open | 🟡 partially-solved | 🟢 solved
> 连接标记: → 表示跨分支依赖

---

## [Root] 如何构建强大的多模态大模型？

---

### [PT] 预训练阶段

#### [PT-1] 🔴 视觉信息如何高效注入 LLM？
- [PT-1a] 🟡 Vision encoder 该冻结还是联合训练？
  - 冻结: 训练效率高，但 encoder 表征不一定对齐 LLM
  - 联合训练: 效果更好但成本高，scaling 行为不明确
  - LLaDA-V 验证了"先冻结后解冻"策略在 dLLM 骨干上同样有效（Stage 1 冻结 SigLIP2，Stage 2 起解冻 lr=2×10⁻⁶）
  - Beyond-LM 在从零训练场景验证"先冻结后解冻"同样有效，与 LLaDA-V 在 LLaDA 初始化场景的发现一致
  - 已有尝试:
  - 相关论文: [[2026-LaViDa-R1]], [[2025-LLaDA-V]], [[2026-Beyond-LM]]
- [PT-1b] 🔴 视觉 token 数量与性能的 tradeoff
  - 高分辨率需要大量 token → 计算成本
  - 压缩策略 (Perceiver, pooling) vs 保留细节
  - LaViDa 使用 2×2 average pooling（27²→14² per view, 共 980 token），但丢失 75% 空间分辨率导致 OCR/文档理解弱（TextVQA 56.3, DocVQA 59.0）。768² 高分辨率显著帮助 OCR 类任务（TextVQA +7.25, DocVQA +15.5），说明视觉 token 的空间细粒度对特定任务至关重要
  - 已有尝试:
  - 相关论文: [[2025-LaViDa]]
- [PT-1c] 🟡 Connector 架构选择的本质区别是什么？
  - MLP vs Q-Former vs Perceiver vs cross-attention
  - LLaDA-V 验证 dLLM 骨干 + 简单两层 MLP 连接器即可达 competitive 性能（MMMU 48.6）
  - LaViDa 使用更简单的单层线性投影即可在 LLaVa-1.6/Open-LLaVa-Next 上达 competitive，进一步验证 dLLM 不需要复杂连接器
  - Beyond-LM 验证 RAE + 简单连接器（类似 MLP）即可达 competitive 性能，进一步支持"dLLM 不需要复杂连接器"的结论
  - 已有尝试:
  - 相关论文: [[2025-LLaDA-V]], [[2025-LaViDa]], [[2026-Beyond-LM]]

#### [PT-2] 🟡 多模态预训练数据如何配比？
- [PT-2a] 🔴 图文对 vs interleaved 数据的最优比例
  - 已有尝试:
  - 相关论文:
- [PT-2b] 🔴 纯文本数据在多模态训练中的作用
  - 保留语言能力 vs 冲淡视觉学习
  - 已有尝试:
  - 相关论文:
- [PT-2c] 🔴 视频数据的引入时机和方式
  - 已有尝试:
  - 相关论文:
- [PT-2d] 🟡 视觉 vs 语言数据的最优比例
  - Beyond-LM 首次为多模态数据配比提供大规模实证: 在万亿 token 规模下，视觉数据需求是语言的 51 倍（51:1）
  - 发现视觉比语言更"数据饥渴"，且差距随规模扩大而增加
  - 但仅一个数据点（万亿规模），小规模（百亿-千亿 token）下最优比例未知
  - 不同任务组合（纯理解 vs 统一模型）的最优比例可能不同
  - 相关论文: [[2026-Beyond-LM]]

#### [PT-3] 🟡 多模态 Scaling Law 是什么形态？
- [PT-3a] 🔴 Vision encoder 和 LLM 的最优 size 比
- [PT-3b] 🟡 数据量 vs 模型 size 的 tradeoff
  - LLaDA-V 提供 dLLM Scaling Law 的首个间接证据: MMMU-Pro 上 1M 数据 dLLM > 9M 数据 AR (LLaMA3-V)，暗示 dLLM 双向注意力在推理密集型任务上数据效率显著高于 AR
  - Beyond-LM 提供视觉-语言扩展律的首个系统性研究: Dense 模型中视觉和语言的参数扩展指数差距为 0.10；MoE 架构将差距缩小到 0.05，有效协调了模态间容量需求
  - Beyond-LM 发现 MoE (per-modality shared experts, G=16) 在相同激活参数下性能匹配或超过 dense 模型
  - 但仅在预训练阶段验证，后训练和 RL 阶段的 scaling law 未知
  - 相关论文: [[2025-LLaDA-V]], [[2026-Beyond-LM]]

#### [PT-4] 🔴 dLLM 骨干的任务偏好分布
- LLaDA-V 首次揭示 dLLM 理解的 fine-grained 优劣势: 知识/数学推理 (MMMU +3.2, MMMU-Pro +6.9) 系统性优于 AR；图表/文档理解 (AI2D -3.3, DocVQA -2.3) 系统性弱于 AR
- 核心问题: 哪些理解子任务天然适合 dLLM、哪些不适合？18 个基准远不足以绘制完整"dLLM 能力地图"
- 与 MMaDA GenEval Position 0.20 的空间推理弱点可能属同一根源——dLLM 缺乏序列/结构推理归纳偏置
- 潜在思路: 更系统的 benchmark 分类分析；probing study 对比 dLLM vs AR 不同层表征质量
- 相关论文: [[2025-LLaDA-V]], [[2025-MMaDA]]

#### [PT-5] 🔴 dLLM 多模态理解的数据效率机制
- LLaDA-V 发现 MMMU-Pro 上 1M 数据 dLLM > 9M 数据 AR——双向注意力为何在少数据条件下更高效？
- 可能解释: (1) bidirectional attention 每个 token 利用完整上下文，有效信息流量更大; (2) mask-predict 目标的数据增强效应（不同 mask pattern = 不同训练 instance）; (3) 推理任务的全局依赖性与双向 attention 的天然匹配
- 需要多任务多规模的系统 Scaling Study 验证
- 相关论文: [[2025-LLaDA-V]]

#### [PT-6] 🔴 从零训练 vs AR初始化的公平对比
- **问题**: DiffusionVL用738K扩散微调达35.1 MMMU-Pro，超越LLaDA-V从零训练15M达48.6的效率。但这个对比不公平——DiffusionVL的base AR模型(Qwen2.5-VL)本身用数百万样本预训练，这些知识被"免费"继承
- **核心疑问**: 如果给扩散模型同等规模的训练数据(50M-100M)，是否能超越AR? 当前多模态dLLM训练数据(15M)相比AR-VLM(50M-100M+)仍是小规模，可能是扩散模型表现不如AR transfer的主要原因
- **需要验证**: (1) 在同等训练数据、同等计算资源下对比从零训练的AR vs Diffusion; (2) 测试"弱AR���型 + 扩散微调"的下界，判断性能主要来自AR基础还是扩散训练增益; (3) 大规模扩散训练(50M-100M样本)是否能超越AR
- **相关论文**: [[2025-DiffusionVL]] (AR初始化5%数据), [[2025-LLaDA-V]] (从零训练15M数据)

---

### [Post] 后训练阶段

#### [Post-1] 🔴 多模态幻觉的根因是什么？如何系统解决？
- [Post-1a] 🟡 幻觉来源: 预训练数据偏差 vs 对齐不足 vs 解码策略
  - ReDiff 揭示了 dLLM 幻觉的第三个来源：训练-推理分布不匹配。标准 masked diffusion 在干净数据上训练但从噪声中间输出生成，并行解码时早期 token 错误污染后续生成，触发复合错误和语义幻觉
  - ReDiff 通过两阶段精炼训练（合成错误 + 模型特定错误）显著减少幻觉（CLAIR +11.2），证明训练时显式引入错误分布建模可有效缓解幻觉
  - 但仅在详细图像描述任务上验证，其他任务类型效果未知
  - 已有尝试: ReDiff 两阶段精炼训练（合成错误 + 在线自我纠错）
  - 相关论文: [[2025-ReDiff]]
- [Post-1b] 🔴 SFT 阶段能解决多少幻觉？上界在哪？
  - 已有尝试:
  - 相关论文:
- [Post-1c] 🔴 需要 RL 介入吗？→ 连接到 [RL-1]
  - 已有尝试:
  - 相关论文:

#### [Post-2] 🟡 多模态 SFT 数据如何高效构造？
- [Post-2a] 🔴 合成数据 vs 人工标注的质量对比
- [Post-2b] 🔴 多模态 instruction 的多样性与覆盖度
- [Post-2c] 🔴 答案多样性扩展的理论基础
  - OpenMMReasoner 发现"答案多样性 > 数据规模"——×8 采样扩展答案多样性比单纯增加数据量更有效；过度过滤损害多样性反而降低性能
  - 核心机制不明: 是学习"推理空间的分布"还是"防止过拟合单一模式"？
  - 最优采样数量（×8）是经验选择，与任务复杂度、教师模型质量的关系未建模
  - "合理多样性"的边界在哪？过度多样性何时会损害性能？
  - 潜在思路: 信息论视角建模多样性收益（最大化 I(路径集合; 任务分布)）；主动学习选择高不确定性样本；课程学习先简单后多样
  - 相关论文: [[2025-OpenMMReasoner]]
- [Post-2d] 🔴 跨域迁移的泛化性边界
  - OpenMMReasoner 验证数学推理 → 多模态推理的正迁移（"textual reasoning transfers alongside strengthened multimodal reasoning"）
  - 边界未明: 需要共享推理模式（如分步求解、逻辑链构建）
  - 其他领域（代码推理、科学推理、常识推理）的迁移效果未知
  - 哪些领域组合存在负迁移？如何量化领域间的"推理模式相似度"？
  - 混合比例的最优值未知（OpenMMReasoner 未明说 We-Math2.0 占比）
  - 潜在思路: 构建多领域迁移矩阵（N×N）；基于任务表征的相似度度量；元学习框架自动发现可迁移模式；动态混合策略（训练中根据验证集调整领域权重）
  - 相关论文: [[2025-OpenMMReasoner]]

#### [Post-3] 🔴 DPO 类方法在多模态场景有效吗？
- [Post-3a] 🔴 偏好数据如何构造 (图文场景的偏好是什么？)
- [Post-3b] 🔴 Offline 方法的分布覆盖问题在多模态是否更严重

---

### [RL] RL 阶段

#### [RL-1] 🔴 多模态 Reward Signal 如何设计？
- [RL-1a] 🔴 图文场景的 reward model 该怎么训？
  - 已有尝试:
  - 相关论文:
- [RL-1b] 🔴 VLM-as-judge 的可靠性如何？
  - 已有尝试: LaViDa-R1 报告 UnifiedReward-Qwen-7B 会幻觉出错误评判标准
  - 相关论文: [[2026-LaViDa-R1]]
- [RL-1c] 🔴 生成任务 (图/视频) 的 reward 如何定义？→ 连接到 [Uni-5b]
  - 已有尝试: MMaDA 用 CLIP+ImageReward, LaViDa-R1 用 EditScore
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]]
- [RL-1d] 🔴 T2I Compositional Reasoning Reward Model 空白
  - CLIP-based reward 无法评估空间关系、属性绑定、多对象计数等 compositional 能力
  - VLM-as-judge 有幻觉风险，现有 reward model 均不支持需要逻辑推理的 T2I prompt
  - 已有尝试: MMaDA GenEval Position 仅 0.20 暴露了此缺口; LaViDa-R1 废弃 PickScore
  - 潜在思路: 训练 compositional reasoning-aware reward model, 或使用 frontier VLM (成本高)
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]]

#### [RL-2] 🔴 GRPO/PPO 如何适配多模态？
- [RL-2a] 🟡 多模态采样的成本远高于纯文本，如何降低？
  - 已有尝试: MMaDA UniGRPO 结构化随机 mask ratio 替代 Monte Carlo 128-sample; LaViDa-R1 complementary masking (w=1)——该技术源自 LaViDa 的 Complementary Masking 训练方法，后被扩展为 RL likelihood estimator
  - 相关论文: [[2025-LaViDa]], [[2025-MMaDA]], [[2026-LaViDa-R1]]
- [RL-2b] 🔴 Group 构造: 同一图不同回答 vs 不同图？
- [RL-2c] 🔴 高熵 token 分布下的 RL 正则化
  - MMaDA 保留 KL penalty, LaViDa-R1 发现 image token NLL>6 导致 KL estimator 方差极大、训练发散
  - LaViDa-R1 用 SFT loss 替代 KL 作为隐式正则化，有效但理论不完备
  - DiMOO Self-GRPO 保留 KL（与 LaViDa-R1 直接冲突），在 1024² 分辨率下稳定性未被充分验证
  - 根本问题: KL 约束对 dLLM 的高熵离散分布是否根本不合适？
  - 潜在思路: 分 token 类型加权 KL, entropy-aware clipping, 自适应 KL 系数, 理论推导最优正则化形式
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]]
- [RL-2d] 🔴 Shrinkage-based advantage 估计在 dLLM 高熵 token 分布下的表现
  - EBPO 证明 James-Stein 收缩 baseline 在纯文本 RL 中严格优于 GRPO 的局部 group mean（MSE 降低），G=8 时 +11.28%
  - 但其 Gaussian 近似假设（θ_q ~ N(μ_glob, τ²)）在 dLLM 的高熵 image token 分布下是否成立未知（NLL>6，分布严重偏斜）
  - 收缩因子 S_q 的计算依赖 σ² 和 τ² 的 Welford 估计，在非 Gaussian 分布下估计质量可能退化
  - 不同模态的 reward 分布特性差异大（text reward 近似正态，image reward 可能重尾），是否需要 per-modality 收缩策略？
  - 潜在思路: 非参数收缩（基于分位数而非均值/方差）；分 token 类型维护先验；直接在 dLLM 上做 EBPO 实验验证
  - 相关论文: [[2026-EBPO]], [[2026-LaViDa-R1]], [[2025-MMaDA]]
- [RL-2e] 🔴 多模态 RL 的 topic-coherent sampling 如何设计
  - EBPO 证明文本数学推理按领域聚类（代数/几何/概率等 9 类）可显著提升高难度 benchmark（AIME25 +6.04%）
  - 多模态场景的聚类维度设计是开放问题——按任务类型（VQA / grounding / T2I）？按视觉内容（自然图像 / 图表 / 文档）？按推理模式（空间推理 / 数学推理 / 常识推理）？
  - 与 [P-RL-07] "SFT 数据多样性比规模更重要"的关系: topic-coherent sampling 在 RL 阶段是否与 SFT 阶段的多样性原则冲突？
  - 潜在思路: 基于 embedding 相似度的自动聚类；多层次聚类（先按任务类型、再按难度）；与 DiMOO 多任务 RL 的 per-task 统计量结合
  - 相关论文: [[2026-EBPO]], [[2025-Lumina-DiMOO]]
- [RL-2f] 🔴 全局先验在非稳态 RL 训练中的滞后问题
  - EBPO 用 Welford 在线估计器维护全局先验（μ_glob, σ², τ²），但 RL 训练中策略持续演化（成功率随训练上升），全局先验可能系统性滞后
  - 在 dLLM RL 中策略变化可能更剧烈（masked diffusion 的随机 mask 引入更多探索性），滞后问题可能更严重
  - 训练早期策略快速提升时，mu_glob 偏低导致 saturated failure 的惩罚信号 (-S_q × mu_glob) 偏弱
  - 潜在思路: EWMA 替代全历史 Welford（窗口大小与 KL divergence 联动）；分阶段重置先验；在线学习 shrinkage 权重
  - 相关论文: [[2026-EBPO]]

#### [RL-4] 🔴 轨迹级 RL vs Token 级 RL 的 Tradeoff
- MMaDA-Parallel ParaRL 采用轨迹级优化（沿整个去噪轨迹应用 CLIP-based alignment reward）
- MMaDA UniGRPO 采用 token 级 reward（masked tokens 的平均 log prob）
- 核心 tradeoff: (1) 轨迹级 RL 的 credit assignment 更困难（哪个时间步的哪个 token 导致了最终结果？），但可能更好地建模多步依赖；(2) Token 级 RL 信号更密集，但可能稀释关键步骤的重要性
- 开放问题: 轨迹级 reward 在中间步骤的有效性——MMaDA-Parallel 用 CLIP 评估噪声图像，但 CLIP 在 out-of-distribution 输入上的可靠性未验证
- 潜在思路: Process Reward Model for dLLM（评估中间去噪状态质量）；混合策略（关键步骤用轨迹 reward，其他用 token reward）
- 相关论文: [[2025-MMaDA-Parallel]], [[2025-MMaDA]]

#### [RL-5] 🟡 Test-time Scaling for dLLM
- **核心问题**: 推理时计算投入（test-time scaling）在 dLLM 中的可行性和上界？与训练时 scaling 的 tradeoff？
- **已有尝试**: dMLLM-TTS 首次系统化探索 dLLM 的推理时优化——通过 Self-Verified Feedback（模型自身理解能力评估生成质量）和 Hierarchical Trajectory Search（O(N+T) 分层搜索），在 Lumina-DiMO/MMaDA/Muddit 上实现 +17.9%~+29.4% GenEval 提升
- **关键发现**: (1) 基座模型越弱，收益越大（MMaDA +29.4% vs DiMOO +17.9%）——暗示 test-time scaling 更多是"弥补训练不足"而非"突破能力上界"；(2) dLLM 的统一架构使模型自身理解能力可作为 verifier，这是 AR 模型难以实现的独特优势
- **核心局限**: (1) SVF 的 bootstrapping 偏差——模型评估自己的输出，理解偏差会循环放大；(2) HTS 的单调性假设——低分辨率质量与高分辨率质量正相关，在细节密集型任务中可能失效；(3) 计算成本的绝对量化缺失——与训练时方法（ReDiff、LaViDa-R1 RL）的 tradeoff 不明确
- **开放子问题**:
  - [RL-5a] 🔴 推理时搜索的收益饱和点？在什么条件下继续增加推理计算不再提升质量？
  - [RL-5b] 🔴 Self-Verification 的可靠性边界？哪些任务类型适合自验证？抽象/艺术风格生成如何定义"正确"？
  - [RL-5c] 🔴 Test-time vs Training-time 的成本对比？如果推理成本等于"训练一个小型 RL 模型"，是否应该优先训练时优化？
- **潜在思路**: (1) SVF + 外部 reward 混合验证（用 anchor reward 校准 bootstrapping 偏差）；(2) Test-time + Training-time 协同（ReDiff 训练时学习纠错 + dMLLM-TTS 推理时搜索）；(3) 二维搜索组合（HTS 分辨率维度 + LaViDa-R1 Tree Search 时间维度）
- **相关论文**: [[2025-dMLLM-TTS]]

#### [RL-3] 🟡 RL 能提升多模态推理能力吗？
- [RL-3a] 🟡 视觉推理 (图表/几何) 的 verifiable reward
  - 已有尝试: MMaDA 对 GeoQA/CLEVR 用 correctness+CLIP reward, UniGRPO 提升 GSM8K +8.2; LaViDa-R1 在 MathVista +2.4; OpenMMReasoner GSPO 在 9 个 benchmark 平均 +11.6%; EBPO 在数学推理上 G=8 时 +11.28%（验证改进 advantage 估计对困难推理任务的显著效果，机制可迁移到多模态）
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-OpenMMReasoner]], [[2026-EBPO]]
- [RL-3b] 🔴 RL 对 grounding 能力的影响
  - 相关论文: [[2026-LaViDa-R1]]
- [RL-3c] 🔴 传统 VLM RL vs dLLM RL 的方法论差异
  - OpenMMReasoner 在传统 VLM（Qwen2.5-VL-7B，AR 生成）上验证 RL（GSPO）有效性，与 dLLM RL 工作（MMaDA/LaViDa-R1）形成架构分野
  - 核心差异:
    - **Likelihood 估计**: 传统 VLM 直接用 AR 解码的 log p(y|x)；dLLM 需要特殊估计（complementary masking/随机 mask ratio）覆盖完整去噪时间步
    - **KL 正则化**: 传统 VLM 的 token 分布熵较低，KL estimator 方差小；dLLM 高熵分布下 KL 存在争议（LaViDa-R1 移除 KL，MMaDA/DiMOO 保留 KL）
    - **Rollout 策略**: 传统 VLM 直接 ×16 采样；dLLM 需要覆盖去噪时间步（complementary masking 或随机 mask ratio）
  - 开放问题: 两者的 RL 方法能否互相借鉴？GSPO 的稳定性优势能否迁移到 dLLM？GSPO 能否适配 masked diffusion？
  - 潜在思路: 抽象出与架构无关的 PG 组件（advantage 估计、reward shaping、正则化）；GSPO-MDM（将 GSPO 的 group-based advantage 适配到 masked diffusion）；统一 Policy Gradient 框架
  - **进展**: EBPO 的 shrinkage baseline 是首个与架构无关的 PG 改进组件——仅修改 advantage 估计中的 baseline 项，不依赖 AR 还是 masked diffusion 的 likelihood 估计方式，可直接移植到 GSPO/UniGRPO/Self-GRPO
  - 相关论文: [[2025-OpenMMReasoner]], [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]], [[2026-EBPO]]

---

### [Diff] Diffusion 基础方法论

#### [Diff-1] 🟡 扩散范式在多模态统一模型中的选择
- [Diff-1a] 🟡 Flow Matching vs DDPM: 在统一模型中哪个更优？
  - Flow Matching 训练更稳定, ���样更快
  - 但与 AR 框架的兼容性分别如何？
  - Beyond-LM 首次在统一模型中使用 flow matching（连续扩散），与 KB 中其他工作的 masked diffusion（离散扩散）形成对比
  - 但缺乏同等条件下的性能对比，无法判断连续 vs 离散哪个更优
  - 已有尝试:
  - 相关论文: [[2026-Beyond-LM]]
- [Diff-1b] 🟡 连续扩散 vs 离散扩散 (Masked Diffusion)
  - 离散扩散和 LLM 的 token 框架天然对齐
  - 连续扩散保留更多信息但需要架构改造
  - 已有证据: MMaDA、LaViDa 系列、DiMOO 均验证离散 masked diffusion 在统一模型中可行且 competitive；DiMOO 在 1024² 分辨率 + GenEval 88% 进一步巩固离散路线
  - LaViDa 是首个将 dLLM 用于多模态理解的 VLM 家族（NeurIPS 2025 Spotlight），在两种 dLLM 骨干（LLaDA-8B, Dream-7B）上均验证方法通用性。Complementary Masking 改进了 masked diffusion 训练的基本效率问题（ScienceQA +67% on 200K subset）
  - LaViDa-O 在 1024² 分辨率 + 多任务场景下仍达 competitive（FID 6.68）
  - LLaDA-V 提供迄今最干净的控制变量对比: 理解端 dLLM 在知识/数学推理上系统性优于 AR (MMMU +3.2, MMMU-Pro +6.9)，在图表/文档理解上系统性弱于 AR (AI2D -3.3, DocVQA -2.3)。不仅验证可行性，更量化了优劣势边界
  - ReDiff 提供 dLLM 训练范式的重要演进：从"被动去噪"到"主动精炼"。两阶段训练（合成错误 + 模型特定错误）为 masked diffusion 训练方法论提供新维度，补充 LaViDa 的 Complementary Masking（效率优化）和 DiMOO 的四阶段管线（数据规模）
  - 连续扩散在统一场景中尚无可比工作
  - Beyond-LM 的 Hybrid Attention Masking（帧内双向 + 跨序列因果）试图兼顾 AR 和 Diffusion 在注意力模式上的优势，是连续扩散方向的新数据点
  - SDAR-VL 在 21 个基准上达到与 AR 基线相当的性能，并通过 ABNS/EMRS/PBNC 解决块状离散扩散的训练稳定性问题
  - 相关论文: [[2025-LaViDa]], [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]], [[2025-LaViDa-O]], [[2025-LLaDA-V]], [[2025-ReDiff]], [[2026-Beyond-LM]], [[2025-SDAR-VL]]
- [Diff-1c] 🟡 采样效率: 如何在统一模型中减少 diffusion 步数
  - Consistency model / distillation 是否适用
  - Few-step generation 对理解能力的影响
  - 已有尝试: MMaDA 用 50 步（vs 1024）图像生成保持强性能，文本/理解可用 1/2-1/4 步数
  - LaViDa Prefix-DLM: 首个 dLLM KV 缓存方案——对 visual+prompt 施加 causal mask 使 KV 可跨去噪步复用，3.9× 加速（CIDEr 仅 -3.1%）。开辟了 dLLM 推理加速方向
  - LaViDa Timestep Shifting Schedule: 凸调度（α=1/3）在 NFE=25% 时 CIDEr 101.1 vs 线性 84.9（+19%），首个针对 dLLM 文本生成的少步调度方案
  - DiMOO ML-Cache: 基于 max logit 选择性缓存稳定 token，与并行采样正交，额外 2× 加速（总计 ~64×，vs AR 的 32× 加速）。无需训练的推理加速方案。与 Prefix-DLM 正交可叠加（前者加速生成部分，后者加速前缀部分）
  - LaViDa-O Stratified Sampling: 空间分散去噪，改善 MDM 独立性假设，提供非「减少步数」的效率思路
  - LaViDa-O Coordinate Quantization: 并行 bbox 解码使 grounding 加速 6.8×
  - ReDiff 主动精炼: 在并行化场景下保持质量（8 tokens/step +21.06），与现有加速技术（Prefix-DLM、ML-Cache）正交。开辟选择性精炼的新方向（仅精炼低置信度 token）
  - Sparse-LaViDa: 首个同时支持 KV 缓存和 token 截断且不牺牲双向上下文的 MDM 加速方案。通过稀疏参数化（动态截断 mask token）+ register tokens（容量补偿）+ step-causal attention mask（KV 缓存支持）实现 1.95-2.83× 加速，质量几乎无损（GenEval 0.78 vs 0.77）。与 Prefix-DLM 和 ML-Cache 正交可叠加，理论上可达 10-15× 总加速
  - SDAR-VL 的块状扩散策略（ABNS 异步噪声 + EMRS 掩码比例缩放 + PBNC 渐进课程）提供训练稳定性优化，与上述推理加速技术正交互补
  - DiffusionVL Block Diffusion: 块间 KV 缓存 + 块内并行去噪，2× 加速。块内双向 + 块间因果的混合注意力实现了 AR-Diffusion 融合
  - 相关论文: [[2025-LaViDa]], [[2025-MMaDA]], [[2025-Lumina-DiMOO]], [[2025-LaViDa-O]], [[2025-ReDiff]], [[2025-Sparse-LaViDa]], [[2025-DiffusionVL]], [[2025-SDAR-VL]]
- [Diff-1c-1] 🔴 Register tokens 的最优数量和设计原则
  - Sparse-LaViDa 使用固定 64 个 register 作为被截断 mask token 的全局摘要，但这是经验选择（0→64 时 GenEval 0.76→0.78，FID 9.32→7.63）
  - 核心问题: 最优数量是否与截断比例、序列长度、模型容量相关？如何理论推导最优值？
  - 潜在思路: (1) 动态 register 数量（根据当前 mask token 数量调整）；(2) 层次化 register（不同层使用不同数量）；(3) 建立 register 数量与截断比例的信息论关系
  - 相关论文: [[2025-Sparse-LaViDa]]
- [Diff-1c-2] 🔴 Step-causal attention mask 的理论基础和泛化性
  - Sparse-LaViDa 的 step-causal mask（clean token 间双向 + mask token 仅看 clean token）是经验设计，缺乏理论分析
  - 核心问题: 为什么这种非对称 attention 不破坏 mask token 预测质量？是否存在其他 attention pattern 实现更好的加速-质量 tradeoff？
  - 对需要 mask token 间交互的任务（如结构化生成）是否有系统性影响？
  - 潜在思路: (1) 推导 step-causal mask 下的 ELBO 界；(2) 分析不同 attention pattern 的加速-质量 tradeoff；(3) 系统评估对结构化生成任务的影响
  - 相关论文: [[2025-Sparse-LaViDa]]
- [Diff-1d] 🔴 dLLM 训练时的错误分布建模
  - ReDiff 暴露的核心问题：如何系统化建模推理时的错误分布？
  - 合成错误（Stage I 随机替换 + 事实错误注入）vs 模型特定错误（Stage II 在线自我纠错）的最优配比
  - 错误分布是否需要随训练动态调整？ReDiff Stage II 证明在线学习能适应模型错误分布的演化
  - 合成错误分布与真实错误的结构化对齐：随机替换创建均匀错误分布，但真实并行生成错误是结构化的（早期错误通过注意力传播到语义相关 token）
  - 潜在思路: 持续在线学习；多样化错误生成策略；adversarial error injection；基于注意力的结构化错误注入（模拟级联传播）
  - 相关论文: [[2025-ReDiff]], [[2025-SDAR-VL]]
- [Diff-1e] 🔴 AR 模型质量对扩散微调效果的定量关系
  - DiffusionVL 需要"高质量预训练 AR 模型"但未量化阈值
  - AR 模型质量是多维的(困惑度、下游任务、推理能力)，哪些维度对扩散迁移最关键?
  - 需要系统性消融: 3B/7B/13B 不同质量 AR 模型的扩散微调效果
  - 相关论文: [[2025-DiffusionVL]]
- [Diff-1f] 🔴 块大小的自适应选择理论
  - DiffusionVL Block Diffusion 块大小 1-16 性能差异 <1.1 分，但最优选择是任务相关的
  - 短回复适合小块(高精度)，长文档适合大块(高并行度)
  - 需要研究: 动态块大小策略、基于内容复杂度的自适应选择、层次化块结构
  - 相关论文: [[2025-DiffusionVL]]
- [Diff-1g] 🔴 块间信息衰减的实验验证
  - **理论隐患**: Block Diffusion 的块间因果依赖意味着第 N 块访问第 1 块信息需经过 N-1 次"中继"，每次可能损失信息
  - **缺乏验证**: DiffusionVL 未提供长序列(>2048 tokens)或多块(>16 blocks)场景下的性能曲线
  - **需要补充**: 在不同块数量(4/8/16/32/64)下测试性能，绘制"块数量 vs 性能"曲线，验证是否存在衰减拐点
  - **理论上限**: 有效序列长度可能在 4096-8192 之间，超过此长度块间依赖衰减严重
  - 相关论文: [[2025-DiffusionVL]]
- [Diff-1h] 🔴 扩散微调的数据效率上界
  - DiffusionVL 5% 数据达 95% 性能——能否进一步压缩?
  - 当前仅单一数据点(738K)，不清楚 100K/300K/500K 的性能曲线
  - 与 LaViDa Complementary Masking 结合能否达到 3-4% 数据效率?
  - 相关论文: [[2025-DiffusionVL]], [[2025-LaViDa]]
- [Diff-1i] 🔴 并行生成的 Error Propagation 系统性解决方案
  - MMaDA-Parallel 揭示 sequential reasoning-then-generation 的错误传播问题：推理阶段错误污染���成阶段条件输入
  - 已有方案: (1) MMaDA-Parallel 并行架构（interleaved token sequences + bidirectional attention）在架构层面避免错误传播；(2) ReDiff 精炼训练（合成错误 + 模型特定错误）在训练层面建模错误分布
  - 但仍缺乏系统理论框架：并行架构 vs 精炼训练的 tradeoff？两者能否组合？
  - 开放问题: Interleaving 策略如何系统化设计？如何量化错误传播的严重程度？
  - 潜在思路: 结构化错误注入训练（基于注意力的级联错误模拟）；并行生成 + 主动精炼组合；理论分析错误传播的图结构
  - 相关论文: [[2025-MMaDA-Parallel]], [[2025-ReDiff]]

#### [Diff-2] 🔴 Diffusion 骨干架构演进
- [Diff-2a] 🔴 DiT 成功的本质原因？Transformer 替代 UNet 的关键
- [Diff-2b] 🟡 MM-DiT (多模态 DiT) 的设计空间
  - Muddit 首次将 MM-DiT (FLUX dual-/single-stream) 用于统一理解+生成模型，证明 T2I 预训练的 MM-DiT 可通过轻量适配（线性头）实现双向统一
  - 但暴露了 CLIP text encoder 77 token 硬限制作为统一场景的结构性瓶颈
  - 开放问题: T2I 预训练确定的 dual/single 分界对 I2T 方向是否最优？MM-DiT vs 全共享 transformer 在统一模型中的 tradeoff？
  - 相关论文: [[2025-Muddit]]
- [Diff-2c] 🔴 Diffusion backbone 和 LLM backbone 能否共享参数？→ 连接到 [Uni-2a]

---

### [Tok] 视觉 Tokenizer

#### [Tok-1] 🔴 离散化的信息损失问题
- [Tok-1a] 🔴 VQGAN codebook 的利用率和表达力
  - Codebook collapse 问题
  - Codebook size 和 token 数的 tradeoff
  - 已有尝试:
  - 相关论文:
- [Tok-1b] 🔴 新量化方案 (FSQ/LFQ/BSQ) 能否根本解决信息损失？
  - 已有尝试:
  - 相关论文:
- [Tok-1c] 🔴 残差量化 (RQ) vs 单层量化的深度对比

#### [Tok-2] 🟡 理解和生成对 Tokenizer 的矛盾需求（核心瓶颈）
- [Tok-2a] 🟡 理解需要语义丰富表示 (CLIP-like) vs 生成需要像素级细节 (VQGAN-like)
  - 这是统一 tokenizer 的根本矛盾
  - DiMOO 证明纯 VQ token（aMUSEd-VQ，无独立视觉编码器）通过大规模数据（~110M）可达 MMMU 58.6%——语义不足可被数据弥补
  - LaViDa-O 采用 SigLIP+VQ 双路方案——理解用 SigLIP 语义，生成用 VQ 离散 token
  - Beyond-LM 提供第三条路径: RAE (Representation Autoencoder, SigLIP 2) 统一表示——单一连续编码器同时服务理解和生成，避免 VQ 离散化信息瓶颈
  - 三种策略: 「用数据换简洁性」(DiMOO 纯 VQ) vs 「用架构换数据效率」(LaViDa-O 双路) vs 「用连续表示换统一性」(Beyond-LM RAE)
  - 相关论文: [[2025-Lumina-DiMOO]], [[2025-LaViDa-O]], [[2026-Beyond-LM]]
- [Tok-2b] 🔴 Janus 的解耦方案是否是最优解？
  - 解耦解决了矛盾, 但失去了理解-生成的表示共享
  - 表示不共享是否限制了涌现能力？
  - 已有尝试:
  - 相关论文:
- [Tok-2c] 🔴 有没有可能训出同时满足两者的统一 Tokenizer？
  - 多目标训练: reconstruction + semantics
  - 分层表示: 高层语义 + 底层细节
  - 已有尝试:
  - 相关论文:
- [Tok-2d] 🟡 MAGVIT-v2 是否是当前 dLLM 统一模型生成质量的主要瓶颈？
  - MMaDA 使用 MAGVIT-v2 (codebook 8192, 512×512), T2I 与 FLUX/SD3 有明显差距
  - DiMOO 使用 aMUSEd-VQ (codebook 8192, 16×16) 达 GenEval 88%，说明 tokenizer 非唯一瓶颈——训练数据量同样关键
  - DiMOO 纯 VQ 在低级视觉任务（super-res, dehazing）表现弱——VQ 信息损失在细粒度任务上确实是瓶颈
  - 需区分: 多少差距来自 tokenizer 信息损失 vs 模型容量/训练数据不足
  - 潜在思路: 语义增强 VQ (CLIP contrastive + reconstruction dual loss)；多尺度 VQ
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]]

#### [Tok-3] 🔴 Tokenizer 对下游训练的影响
- [Tok-3a] 🔴 Tokenizer 质量对生成质量的上界限制
  - DiMOO 纯 VQ token 在低级视觉任务上的信息瓶颈——16×16 下采样天然丢失像素级细节
  - 潜在思路: 多尺度 VQ（低分辨率语义 + 高分辨率细节）；混合连续-离散 token 方案
  - 相关论文: [[2025-Lumina-DiMOO]]
- [Tok-3b] 🔴 Tokenizer 和主模型是否需要联合训练？

---

### [Uni] 理解生成一体化

#### [Uni-1] 🔴 统一架构路线之争
- [Uni-1a] 🟡 哪条路线最可能 scale？
  - AR+Diffusion (Transfusion, MetaMorph): 灵活, 各取所长, 但训练目标不统一
  - 纯AR离散化 (Chameleon, Emu3): 框架统一优雅, 但视觉离散化有损
  - 解耦编码 (Janus, SEED): 实用, 避免冲突, 但不够"统一"
  - Diffusion原生: MMaDA 首次在 NeurIPS 级别验证 8B 规模可行；DiMOO 纯 VQ 极简架构达 GenEval 88%（超 FLUX.1-dev），LaViDa-O Elastic-MoT 达 89%（w/ reflection），证明 Diffusion 原生路线可 scale
  - dLLM 纯理解验证: LLaDA-V 在控制变量下证明 dLLM 骨干做纯理解已优于 AR 骨干 (11/18 基准胜出)，为统一模型的理解端提供信心基础
  - Vision-first 子路线: Muddit 从预训练 T2I 模型（Meissonic MM-DiT）出发，1B+3.5M 达 GenEval 0.61 + 4-11x 推理加速，证明从视觉先验出发也可行，但文本能力受限（CLIP 77 token）→ 见 [Uni-1e]
  - AR-to-Diffusion 子路线: DiffusionVL 从预训练 AR VLM（Qwen2.5-VL-7B）扩散微调，738K 数据达 95% 性能。Block Diffusion 提供 AR-Diffusion 融合的中间范式。证明 AR 知识可高效迁移到扩散骨干，但自身训练增益 vs AR 知识继承的贡献未分离 → 见 [PT-6]
  - SDAR-VL 通过块状扩散 + 训练稳定性优化（ABNS/EMRS/PBNC），在 21 个基准上达到与 AR 基线相当的性能，进一步验证 dLLM 统一模型的可行性
  - 已有尝试: LaViDa (首个 dLLM VLM 家族，路线 B 起点), MMaDA (diffusion-native 三任务 competitive), LaViDa-R1 (grounding specialist level), DiMOO (GenEval 88% 超越专用模型), LaViDa-O (Elastic-MoT 10.4B 多任务 SOTA), Muddit (Vision-first 1B 路线验证), LLaDA-V (dLLM 纯理解验证 11/18 超越 AR), DiffusionVL (AR→Diffusion 迁移, Block Diffusion), SDAR-VL (块状扩散训练稳定性)
  - 相关论文: [[2025-LaViDa]], [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]], [[2025-LaViDa-O]], [[2025-Muddit]], [[2025-LLaDA-V]], [[2025-DiffusionVL]], [[2025-SDAR-VL]]
- [Uni-1b] 🔴 AR + Diffusion 的融合细节
  - 在哪一层融合？token level vs feature level
  - Loss 权重如何平衡？动态 vs 静态
  - Transfusion 的 causal mask + bidirectional mask 设计
  - 已有尝试:
  - 相关论文:
- [Uni-1c] 🔴 纯 AR 路线的生成质量天花板
  - 离散 token 的信息瓶颈是否可突破
  - 已有尝试:
  - 相关论文:
- [Uni-1d] 🔴 解耦方案 (Janus) 的扩展性
  - 解耦是否只是过渡方案？
  - 能否在解耦基础上逐步走向统一？
- [Uni-1e] 🔴 视觉先验 vs 语言先验初始化的 Scaling Crossover
  - Muddit 从 T2I 模型（Meissonic）出发，1B+3.5M 达 GenEval 0.61（接近 SD3 0.62），参数/数据效率极高
  - KB 中所有其他 dLLM 统一模型均从 LLM（LLaDA）出发，8-10B+数M-110M，GenEval 0.63-0.88
  - 核心问题: 在什么规模/数据量下两种初始化策略的优势发生交叉？视觉先验在小规模占优，但大规模下 LLM 骨干的文本/推理优势可能远超视觉先验初始化
  - 当前仅有 Muddit (1B) 一个 Vision-first 数据点，无法判断趋势
  - 潜在思路: 固定架构做 1B/3B/7B scaling study，对比两种初始化的训练效率和最终性能 Pareto
  - 相关论文: [[2025-Muddit]], [[2025-Lumina-DiMOO]], [[2025-MMaDA]]

#### [Uni-2] 🔴 理解与生成的能力关系
- [Uni-2a] 🟡 共享参数下能力是否冲突？
  - 实验证据: MMaDA Figure 6 显示 Mixed CoT 训练中三项任务指标同步提升，无 seesaw 效应
  - DiMOO 进一步证明模态无关全共享在 ~110M 数据下可达 GenEval 88% + MMMU 58.6%，无明显冲突
  - LaViDa-O 表明共享有益但路由解耦可在不破坏协同的前提下提升效率（MMMU 45.1 vs MMaDA 30.2）
  - 是否存在 mutual benefit？MMaDA 和 LaViDa-R1 均观察到跨模态正向协同
  - 但仅在 8-10B 规模验证，更大规模行为未知
  - → 连接到 [Diff-2c]
  - DiffusionVL 从 AR VLM 扩散微调后理解性能几乎无损（MMMU-Pro 35.1 vs base 36.7），说明共享参数从 AR 迁移到 Diffusion 后能力不冲突
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]], [[2025-LaViDa-O]], [[2025-DiffusionVL]]
- [Uni-2b] 🟡 MoE / Routing 解耦是否可行？
  - LaViDa-O Elastic-MoT 首次验证: 非对称参数分配（理解 8B + 生成 2.4B），前 16 层共享 + 后 16 层分离，训练加速 3.17×
  - 但 DiMOO 用全共享架构 + 更多数据达到接近性能（GenEval 88% vs 89%），说明解耦非必须
  - 开放问题: 最优非对称比例？routing collapse 风险？共享层梯度不平衡？
  - 相关论文: [[2025-LaViDa-O]], [[2025-Lumina-DiMOO]]
- [Uni-2c] 🟡 训练配比与课程学习
  - 理解数据 vs 生成数据 vs 交叉数据的配比
  - 先理解后生成 vs 联合训练 vs 课程策略
  - DiMOO 四阶段管线（预训练 80M→中间训练 3M→SFT 30M→RL）提供迄今最详细的 dLLM 训练 recipe
  - LaViDa-O 三阶段课程（理解扩展→渐进分辨率生成→联合端到端）提供另一范式
  - 相关论文: [[2025-Lumina-DiMOO]], [[2025-LaViDa-O]]
- [Uni-2d] 🔴 表示对齐: 理解和生成是否需要在同一语义空间？

#### [Uni-3] 🟡 生成质量追赶专用模型
- [Uni-3a] 🟡 当前差距分析: 统一模型 vs FLUX/SD3 级别
  - DiMOO GenEval 88% 超越 FLUX.1-dev (82%) 和 GPT-4o (84%)，首次证明 dLLM 在 compositional T2I 上可超越专用连续扩散模型
  - LaViDa-O GenEval 89% (w/ planning+reflection) 同样超越 FLUX.1-dev
  - 但 FID/美学质量等维度仍有差距，GenEval 主要衡量 compositional 能力
  - 相关论文: [[2025-Lumina-DiMOO]], [[2025-LaViDa-O]]
- [Uni-3b] 🔴 生成质量的瓶颈到底在哪？
  - Tokenizer? Diffusion 步数? 模型容量分配? 训练数据?
- [Uni-3c] 🔴 用 RL 优化生成质量 → 连接到 [RL-1c]
  - DPOK, DDPO 等方法在统一模型中的应用
- [Uni-3d] 🟡 图像编辑作为理解+生成的联合测试
  - LaViDa-O 在 editing 上部分超越 GPT-4o（replacement 4.39, removal 3.98）
  - DiMOO 广泛 I2I 支持（controllable/style transfer/subject-driven/editing），整体超越 OmniGen
  - 两者均证明理解能力可显式提升编辑质量（LaViDa-O planning, DiMOO subject-driven DINOv2 +3.97%）
  - 相关论文: [[2025-LaViDa-O]], [[2025-Lumina-DiMOO]]

#### [Uni-4] 🔴 扩展到视频
- [Uni-4a] 🔴 计算瓶颈: 视频 token 数爆炸
- [Uni-4b] 🔴 时序建模: AR 时序 + Diffusion 空间的混合
- [Uni-4c] 🔴 视频理解和生成的统一是否比图像更难？
- [Uni-4d] 🔴 世界模型视角: 视频生成 = 物理规律学习？

#### [Uni-5] 🟡 统一模型的 Post-training 和 RL
- [Uni-5a] 🟡 统一模型如何做 alignment？
  - 理解和生成分别对齐 vs 联合对齐
  - 已有方案: MMaDA 提供首个 diffusion-native 全链路（预训练→Mixed CoT SFT→UniGRPO RL）; LaViDa-R1 提供统一 PG 框架（SFT+GRPO+self-distillation）; DiMOO 提供四阶段管线 + Self-GRPO 联合 RL; EBPO 的 shrinkage baseline 可作为架构无关的 advantage 估计改进，直接嵌入任何 GRPO 变体
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]], [[2026-EBPO]]
- [Uni-5b] 🟡 统一模型的 reward 如何设计？→ 连接到 [RL-1c]
  - 理解正确性 + 生成质量 + 一致性 的多目标 reward
  - 已有方案: MMaDA Diversified Reward (correctness/format/CLIP/ImageReward), LaViDa-R1 Multi-Reward (correctness/IoU/EditScore), DiMOO Self-GRPO (模型自身理解能力作为隐式 reward)
  - DiMOO 的自评估方案: 用 entity-relation-value 三元组自动生成理解问题评估 T2I 质量，无需外部 reward model
  - 已发现局限: CLIP reward 不支持 compositional reasoning → 连接到 [RL-1d]; Self-GRPO 有 bootstrapping 偏差风险
  - "一致性" 维度仍完全未被探索
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]]
- [Uni-5c] 🟡 RL 是否能促进理解-生成的协同？
  - 假设: RL 优化"看图生图"任务可能同时提升两种能力
  - 已有证据: MMaDA UniGRPO 同时提升 GSM8K (推理) 和 ImageReward (生成); LaViDa-R1 多任务 RL 各任务均有提升; DiMOO Self-GRPO 验证「一个 loss 同时提升 T2I 和理解」
  - DiMOO 的联合目标 L(θ) = -∑w(g)(ℓ_T2I + ℓ_MMU) + β·KL，明确将 T2I 和理解 loss 绑定在同一优化中
  - 但机制不清晰: 是共享表示的迁移还是任务间正则化？
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]]

#### [Uni-6] 🔴 Thinking-Aware Image Synthesis 的评估标准
- MMaDA-Parallel 提出 ParaBench（300 challenging prompts，6 维度评估：text quality, text alignment, image quality, image alignment, image consistency, cross-modal output alignment）
- 首次系统化评估"推理感知图像生成"——需要先推理后生成的任务
- 核心问题: 如何量化 CoT 推理对生成质量的因果贡献？faithfulness 如何测量？
- 现有 GenEval/COCO 等基准主要评估 compositional 能力，不评估推理链质量
- 开放问题: (1) 推理链的 faithfulness（扰乱推理是否影响生成质量？）；(2) 推理链的必要性（哪些任务真正需要推理？）；(3) 推理链的可解释性（能否从推理链预测生成结果？）
- 潜在思路: 因果干预实验（mask/扰乱推理链观察生成变化）；推理链质量与生成质量的相关性分析；构建需要多步推理的 T2I 基准
- 相关论文: [[2025-MMaDA-Parallel]]
