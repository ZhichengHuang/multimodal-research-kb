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
  - Qwen3-VL 提供 AR 侧的成熟方案: 动态分辨率（最高 1344×1344, 保持原始宽高比）+ 自适应 Token Merging（2.3× 效率提升同时保持性能），但 Token Merging 对 OCR/细粒度文档理解有损（均匀合并无重要性先验）
  - dLLM 侧的 token 压缩策略（LaViDa 2×2 pooling、VidLaDA 帧级 pooling）均为粗暴全局压缩，Qwen3-VL 的内容感知自适应 merging 是否可迁移到 dLLM 尚未验证
  - 已有尝试:
  - 相关论文: [[2025-LaViDa]], [[2025-Qwen3-VL]]
- [PT-1c] 🟡 Connector 架构选择的本质区别是什么？
  - MLP vs Q-Former vs Perceiver vs cross-attention vs 多层注入（DeepStack）
  - LLaDA-V 验证 dLLM 骨干 + 简单两层 MLP 连接器即可达 competitive 性能（MMMU 48.6）
  - LaViDa 使用更简单的单层线性投影即可在 LLaVa-1.6/Open-LLaVa-Next 上达 competitive，进一步验证 dLLM 不需要复杂连接器
  - Beyond-LM 验证 RAE + 简单连接器（类似 MLP）即可达 competitive 性能，进一步支持"dLLM 不需要复杂连接器"的结论
  - Qwen3-VL DeepStack 在 AR VLM 中验证多层视觉注入（约第 10/20/30 层）贡献 +3.1% MMMU。但这是公认的性能提升手段（Flamingo cross-attention 的变体），而非方法论创新。对 dLLM 的启示: "dLLM 不需要复杂连接器"结论尚未覆盖"多层 cross-attention 注入"维度——dLLM 的 bidirectional attention 理论上在每层都能利用 visual cross-attention，效果可能与 AR 不同（需实验验证）
  - 已有尝试:
  - 相关论文: [[2025-LLaDA-V]], [[2025-LaViDa]], [[2026-Beyond-LM]], [[2025-Qwen3-VL]]

#### [PT-2] 🟡 多模态预训练数据如何配比？
- [PT-2a] 🔴 图文对 vs interleaved 数据的最优比例
  - 已有尝试:
  - 相关论文:
- [PT-2b] 🔴 纯文本数据在多模态训练中的作用
  - 保留语言能力 vs 冲淡视觉学习
  - 已有尝试:
  - 相关论文:
- [PT-2c] 🟡 视频数据的引入时机和方式
  - VidLaDA 提供 dLLM 视频训练的三阶段课程方案: (1) Short-clip temporal pre-training → (2) Temporal scaling warm-up → (3) Long-form video expansion（2-30 分钟）。配合时间分层采样 + LLM 指令合成 + text-bias 过滤 + MLLM 一致性投票的数据工程
  - 已有尝试: VidLaDA 三阶段视频课程
  - 相关论文: [[2025-VidLaDA]]
- [PT-2d] 🟡 视觉 vs 语言数据的最优比例
  - Beyond-LM 首次为多模态数据配比提供大规模实证: 在万亿 token 规模下，视觉数据需求是语言的 51 倍（51:1）
  - 发现视觉比语言更"数据饥渴"，且差距随规模扩大而增加
  - Kimi K2.5 提供另一数据点: 15T 规模 1.04T MoE 模型上恒定 10% 视觉 token 比例优于渐进提升策略，但消融不足（未扫描 5%-20% 范围），因果推断不成立
  - 两者从不同维度切入（数据总需求量 vs 混合训练比例），方向一致: 视觉数据"够用但不主导"
  - 但仅一个数据点（万亿规模），小规模（百亿-千亿 token）下最优比例未知
  - 不同任务组合（纯理解 vs 统一模型）的最优比例可能不同
  - 相关论文: [[2026-Beyond-LM]], [[2025-KimiK2.5]]

#### [PT-3] 🟡 多模态 Scaling Law 是什么形态？
- [PT-3a] 🔴 Vision encoder 和 LLM 的最优 size 比
- [PT-3b] 🟡 数据量 vs 模型 size 的 tradeoff
  - LLaDA-V 提供 dLLM Scaling Law 的首个间接证据: MMMU-Pro 上 1M 数据 dLLM > 9M 数据 AR (LLaMA3-V)，暗示 dLLM 双向注意力在推理密集型任务上数据效率显著高于 AR
  - Beyond-LM 提供视觉-语言扩展律的首个系统性研究: Dense 模型中视觉和语言的参数扩展指数差距为 0.10；MoE 架构将差距缩小到 0.05，有效协调了模态间容量需求
  - Beyond-LM 发现 MoE (per-modality shared experts, G=16) 在相同激活参数下性能匹配或超过 dense 模型
  - Qwen3-VL 提供 AR 侧最完整的 scaling curve: 2B→4B→8B→32B dense + 32B-MoE（64 experts, top-8 routing），32B-MoE 仅损失 1.2% 性能但降低 40% 推理延迟，验证了多模态 MoE 的实用性。但所有变体共享相同 Vision Encoder，仅 LLM backbone 不同——Vision encoder scaling 仍是空白
  - dLLM 侧的 scaling law 研究仍缺（LLaDA-V 8B 是唯一数据点），与 AR 侧 Qwen3-VL 的完整 scaling curve 形成鲜明对比
  - 但仅在预训练阶段验证，后训练和 RL 阶段的 scaling law 未知
  - 相关论文: [[2025-LLaDA-V]], [[2026-Beyond-LM]], [[2025-Qwen3-VL]]

#### [PT-4] 🔴 dLLM 骨干的任务偏好分布
- LLaDA-V 首次揭示 dLLM 理解的 fine-grained 优劣势: 知识/数学推理 (MMMU +3.2, MMMU-Pro +6.9) 系统性优于 AR；图表/文档理解 (AI2D -3.3, DocVQA -2.3) 系统性弱于 AR
- VidLaDA 新增视频维度数据点: 长视频全局推理 dLLM 优（LongVideoBench +3.2, MLVU +3.0），短视频时序定位 dLLM 弱（MVBench -10.2，差距远超图像域 -2 至 -3）
- 核心问题: 哪些理解子任务天然适合 dLLM、哪些不适合？18 个基准远不足以绘制完整"dLLM 能力地图"
- 与 MMaDA GenEval Position 0.20 的空间推理弱点可能属同一根源——dLLM 缺乏序列/结构推理归纳偏置
- 潜在思路: 更系统的 benchmark 分类分析；probing study 对比 dLLM vs AR 不同层表征质量
- 相关论文: [[2025-LLaDA-V]], [[2025-MMaDA]], [[2025-VidLaDA]]

- [PT-4a] 🔴 dLLM 骨干在短视频/细粒度时序任务上的结构性弱势
  - VidLaDA MVBench 59.4 vs Qwen2.5-VL 69.6（-10.2）——差距远超图像域（-2 至 -3），说明视频时序因果推理是 dLLM 更严重的结构性弱点
  - 根因: MVBench 包含动作顺序、因果关系、状态变化等天然顺序依赖任务。dLLM bidirectional attention 无法区分"A→B"和"B→A"序列（需依赖 position embedding 区分因果方向），AR 的 causal mask 强制编码顺序偏好
  - Qwen3-VL 的 Interleaved-MRoPE 消融显示 temporal 轴位置编码是最关键组件（+4.2% MMMU），从位置编码层面解决时序感知。但 MRoPE 为 AR causal attention 设计，**dLLM 中完全未尝试空间-时序位置编码**——这是 KB 中新暴露的 gap。MRoPE 与完全双向注意力理论上兼容（位置编码与 attention mask 正交），但与 dLLM 的部分因果优化（Prefix-DLM）冲突
  - 潜在思路: 时序感知 masking schedule（时间早的帧先去噪）；Temporal Block Diffusion（按时间块顺序去噪 + 块内并行）；混合 attention（理解全双向 + 解码时序因果）；**MRoPE-dLLM（将三轴位置编码引入 dLLM，仅改位置编码不触及 attention mask，详见 [[qwen3vl-crossover-to-dllm#方向A]]）**
  - 相关论文: [[2025-VidLaDA]], [[2025-Qwen3-VL]]
- [PT-4b] 🔴 dLLM 多模态理解的数据效率机制
  - LLaDA-V 发现 MMMU-Pro 上 1M 数据 dLLM > 9M 数据 AR——双向注意力为何在少数据条件下更高效？
- 可能解释: (1) bidirectional attention 每个 token 利用完整上下文，有效信息流量更大; (2) mask-predict 目标的数据增强效应（不同 mask pattern = 不同训练 instance）; (3) 推理任务的全局依赖性与双向 attention 的天然匹配
- 需要多任务多规模的系统 Scaling Study 验证
- 相关论文: [[2025-LLaDA-V]]

#### [PT-6] 🔴 从零训练 vs AR初始化的公平对比
- **问题**: DiffusionVL用738K扩散微调达35.1 MMMU-Pro，超越LLaDA-V从零训练15M达48.6的效率。但这个对比不公平——DiffusionVL的base AR模型(Qwen2.5-VL)本身用数百万样本预训练，这些知识被"免费"继承
- **核心疑问**: 如果给扩散模型同等规模的训练数据(50M-100M)，是否能超越AR? 当前多模态dLLM训练数据(15M)相比AR-VLM(50M-100M+)仍是小规模，可能是扩散模型表现不如AR transfer的主要原因
- **需要验证**: (1) 在同等训练数据、同等计算资源下对比从零训练的AR vs Diffusion; (2) 测试"弱AR���型 + 扩散微调"的下界，判断性能主要来自AR基础还是扩散训练增益; (3) 大规模扩散训练(50M-100M样本)是否能超越AR
- **相关论文**: [[2025-DiffusionVL]] (AR初始化5%数据), [[2025-LLaDA-V]] (从零训练15M数据)

#### [PT-8] 🔴 稀疏 MoE 训练稳定性诊断向多模态推广
- **问题**: Step 3.5 Flash + GLM-5 独立验证了 "Expert Collapse ≠ Routing Collapse"——routing 统计（gate probability, dispatch counts）正常但专家 activation norm 归零。诊断框架包含三层: (1) per-expert activation norm 监控（max-to-median ratio 是最可靠 early indicator）; (2) Muon Polar Express float16 数值修复; (3) FFN intermediate activation clipping（优于 weight clipping）
- **多模态 gap**: Beyond-LM 使用 per-modality shared experts + routing experts，51:1 视觉:语言数据不平衡下 expert collapse 是否呈现新模式？视觉专家是否可能因 routing 偏好而 "饿死"？
- **需要验证**: (1) 将 per-expert activation norm 监控扩展为 per-modality 分析; (2) 测试 activation clipping 是否需要 per-modality 阈值（视觉 vs 语言 FFN 激活分布不同）; (3) EP-level load balancing 在模态路由偏好下是否需要 per-modality ℒ_EP
- **相关论文**: [[2026-Step35Flash]], [[2026-GLM-5]], [[2026-Beyond-LM]]

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
- [Post-2c-1] 🔴 质量-多样性 Pareto 前沿
  - OpenMMReasoner 的 no-filtering 策略与常见的 aggressive curation 矛盾——何时过滤有益 vs 何时过滤有害？
  - 三种互补解释: (1) 噪声正则化——"非最优"推理路径提供 reasoning-level label smoothing; (2) 覆盖保留——过滤创建分布 gap（删除含 backtracking 的链导致不会自我纠错）; (3) 探索预调——messy 推理链降低 RL 策略自信度，促进探索
  - 需要形式化建模: 在什么数据质量阈值下，过滤开始损害多样性多于帮助质量？
  - 潜在思路: 信息论建模 H(Path_set) vs 质量分的 Pareto 曲线；A/B test 不同过滤阈值的下游 RL 效果
  - 相关论文: [[2025-OpenMMReasoner]]
- [Post-2d] 🔴 跨域迁移的泛化性边界
  - OpenMMReasoner 验证数学推理 → 多模态推理的正迁移（"textual reasoning transfers alongside strengthened multimodal reasoning"）
  - 边界未明: 需要共享推理模式（如分步求解、逻辑链构建）
  - 其他领域（代码推理、科学推理、常识推理）的迁移效果未知
  - 哪些领域组合存在负迁移？如何量化领域间的"推理模式相似度"？
  - 混合比例的最优值未知（OpenMMReasoner 未明说 We-Math2.0 占比）
  - 潜在思路: 构建多领域迁移矩阵（N×N）；基于任务表征的相似度度量；元学习框架自动发现可迁移模式；动态混合策略（训练中根据验证集调整领域权重）
  - 相关论文: [[2025-OpenMMReasoner]]
- [Post-2e] 🔴 教师-学生最优规模比与蒸馏效率
  - OpenMMReasoner 使用 235B→7B (~33×) 教师-学生规模比，教师比 baseline 在所有 benchmark 上平均提升 ≥4.5 分
  - 核心问题: (1) 是否存在最小有效比例（32B→7B 约 4.6× 是否足够）？(2) 规模比与 ×k 采样的交互——更大教师是否让更少采样即可达到相同多样性？(3) 不同领域对教师质量的敏感度不同
  - 潜在思路: 系统消融不同规模教师的蒸馏效果；信息论分析蒸馏效率与规模比的关系；per-domain 教师质量评估决定是否需要领域特化教师
  - 相关论文: [[2025-OpenMMReasoner]]

#### [Post-3] 🔴 DPO 类方法在多模态场景有效吗？
- [Post-3a] 🔴 偏好数据如何构造 (图文场景的偏好是什么？)
- [Post-3b] 🔴 Offline 方法的分布覆盖问题在多模态是否更严重

---

### [RL] RL 阶段

#### [RL-1] 🔴 多模态 Reward Signal 如何设计？
- [RL-1a] 🟡 图文场景的 reward model 该怎么训？
  - 已有尝试: Kimi K2.5 GRM (Generative Reward Model) 是 KB 中首个细粒度多维 reward model——用生成式模型输出 helpfulness/relevance/aesthetic quality 等多维度评估，超越 CLIP+ImageReward 的二元判断
  - 但 GRM 的架构细节、训练数据、评估准确率均未公开，可复制性存疑
  - 相关论文: [[2025-KimiK2.5]]
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
  - 已有尝试: MMaDA UniGRPO 结构化随机 mask ratio 替代 Monte Carlo 128-sample; LaViDa-R1 complementary masking (w=1)——该技术源自 LaViDa 的 Complementary Masking 训练方法，后被扩展为 RL likelihood estimator; Kimi K2.5 Toggle RL 从输出长度维度优化——交替预算约束/全长生成，输出 token 减少 25-30%（与 dLLM 特有技术正交的通用方案）; LFPO 通过 Theorem 3.1 完全绕过 likelihood 计算（无似然度方案），同时 flow matching 轨迹拉直效应减少推理步数（代码 -41.8 步，推理 -159 步），进一步降低采样成本; Step 3.5 Flash MIS-PO 用离散过滤（接受/拒绝）替代连续重要性加权——完全消除高方差权重项，在 token 和轨迹层面限制优化在信赖域内，为 off-policy 严重的长时推理 RL 提供第五种范式（vs 结构化 mask / complementary masking / shrinkage baseline / 无似然度），在 IMO 85.4% / LiveCodeBench 86.4% frontier 难度下验证
  - 相关论文: [[2025-LaViDa]], [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-KimiK2.5]], [[2026-LFPO]], [[2026-Step35Flash]]
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
- [RL-2g] 🔴 Velocity-based vs Likelihood-based dLLM RL 的系统对比
  - LFPO 的速度场修正（无似然度，精确梯度）和 UniGRPO/complementary masking 的似然度估计（近似，有方差）代表两条根本不同的优化路径
  - 理论上 LFPO 消除了似然度估计方差，但 Theorem 3.1 依赖连续时间极限假设，少步推理（8-32 步）场景下离散化误差可能系统性偏大
  - Token 级 credit assignment 被牺牲——LFPO 在时间步级别操作速度残差，关键推理步骤仅涉及 2-3 个 token 时信号被平均化；likelihood-based 方法可通过 token 级 log prob 做细粒度归因
  - 需要在相同基座（LLaDA 8B / DiffuCoder）、相同任务、相同 reward 下做严格对比——精度 vs 方差 vs 计算成本 vs 推理步数的全面 Pareto 分析
  - 潜在思路: 混合方案——LFPO 粗粒度方向 + complementary masking 细粒度 token 级修正；自适应切换（训练早期用 LFPO 快速对齐，后期切换至 likelihood-based 精调）
  - 相关论文: [[2026-LFPO]], [[2025-MMaDA]], [[2026-LaViDa-R1]]
- [RL-2h] 🔴 RL 后训练的轨迹拉直机制
  - LFPO 观察到 RL 训练后推理步数显著减少（代码 -41.8 步，推理 -159.0 步），论文归因于 flow matching 的轨迹拉直效应，但缺乏严格理论分析
  - 对比: AGRPO 在 MATH 上反而增加 +73.6 步——不同 RL 方法对推理步数的影响方向相反
  - 核心问题: 为什么对比式速度修正会导致更直的去噪路径？这是 flow matching 的固有性质还是对比式优化的特异效果？
  - 与 XDLM 预训练减步（混合噪声核）和 DiMOO ML-Cache 推理减步形成三层加速体系——预训练 / 后训练 / 推理各一层
  - 潜在思路: 分析对比式优化目标的隐式正则化效应；设计专门利用轨迹拉直效应的训练目标进一步放大减步效果；验证 LFPO + XDLM 双层减步的叠加效果
  - 相关论文: [[2026-LFPO]], [[2025-XDLM]], [[2025-Lumina-DiMOO]]
- [RL-2i] 🔴 IcePop 解决策略版本不匹配，但 dLLM 还有输入分布不匹配（P-Diff-04）——是否可将 IcePop 扩展为双维度修正？
- [RL-2j] 🔴 跨阶段蒸馏在异质模态任务（text NLL<2 vs image NLL>6）间的 advantage 归一化策略
- [RL-2k] 🔴 RL 训练中 sparse/dynamic 计算的确定性需求 vs dLLM 推理加速的动态性冲突
- [RL-2l] 🔴 MIS-PO 离散过滤 vs 连续重要性加权的理论统一
  - Step 3.5 Flash MIS-PO（离散接受/拒绝过滤）和 GLM-5 IcePop（连续 pop 函数校正）代表两种根本不同的 off-policy 处理范式
  - 离散过滤: 方差更低（完全消除高方差权重项），但信息损失（丢弃 off-policy 样本）
  - 连续校正: 信息保留更好（重加权而非丢弃），但方差仅降低未消除
  - 核心问题: 在什么 off-policy 严重程度下离散过滤优于连续校正？是否存在理论最优的离散化粒度？
  - 对 dLLM 的特殊意义: image token NLL>6 导致 off-policy 极端严重，MIS-PO 的激进过滤可能更适合 dLLM RL
  - 潜在思路: 自适应离散化（off-policy ratio 低时连续加权，高时离散过滤）；per-modality 阈值（text τ=0.9, image τ=0.7）
  - 相关论文: [[2026-Step35Flash]], [[2026-GLM-5]]
- [RL-2m] 🔴 MIS-PO 在异构 Reward 分布下的阈值收缩
  - Step 3.5 Flash 用固定阈值处理三类 reward（verifiable / non-verifiable / agent），但三类 reward 的分布差异极大
  - dLLM 面临更异构的 reward 分布: text 理解（连续评分）、image 生成（CLIP/GenEval）、grounding（IoU 二值）
  - 核心问题: 单一阈值是否适用于 reward 分布差异极大的多模态场景？是否需要 per-reward-type MIS-PO 阈值？
  - 潜在思路: uncertainty-weighted 阈值（根据各 reward 的置信区间动态调整）；compositional reward filtering（先一致性硬门 → 再 CLIP 过滤 → 最后 MIS-PO）
  - 相关论文: [[2026-Step35Flash]]

#### [RL-4] 🔴 轨迹级 RL vs Token 级 RL 的 Tradeoff
- MMaDA-Parallel ParaRL 采用轨迹级优化（沿整个去噪轨迹应用 CLIP-based alignment reward）
- MMaDA UniGRPO 采用 token 级 reward（masked tokens 的平均 log prob）
- 核心 tradeoff: (1) 轨迹级 RL 的 credit assignment 更困难（哪个时间步的哪个 token 导致了最终结果？），但可能更好地建模多步依赖；(2) Token 级 RL 信号更密集，但可能稀释关键步骤的重要性
- 开放问题: 轨迹级 reward 在中间步骤的有效性——MMaDA-Parallel 用 CLIP 评估噪声图像，但 CLIP 在 out-of-distribution 输入上的可靠性未验证
- 潜在思路: Process Reward Model for dLLM（评估中间去噪状态质量）；混合策略（关键步骤用轨迹 reward，其他用 token reward）
- 相关论文: [[2025-MMaDA-Parallel]], [[2025-MMaDA]]

- [RL-4a] 🔴 序列级 vs Token 级 Importance Ratio 的 Tradeoff
  - OpenMMReasoner GSPO 使用**序列级 importance ratio**（π(seq)/π_ref(seq) 单一标量），替代 GRPO 的**token 级乘积**（∏ᵢ π(aᵢ|sᵢ)/π_ref(aᵢ|sᵢ)），训练更稳定
  - Step 3.5 Flash MIS-PO 使用离散过滤（接受/拒绝）隐式在轨迹级操作——两者共同暗示序列/轨迹级方案在长序列场景优于 token 级
  - 核心 tradeoff: token 级 ratio 乘积方差随序列长度 L 指数增长（长 CoT 尤其严重），但提供精细信用分配；序列级 ratio 方差被平滑但丧失 token 级分辨率
  - 对 dLLM 的意义: image token NLL>6 导致 token 级 ratio 爆炸更严重——序列级方案可能天然更适合 dLLM RL
  - 需要验证: GSPO 的序列级 ratio 能否通过 complementary masking 适配到 masked diffusion（GSPO-MDM）？方差界的理论推导？
  - 相关论文: [[2025-OpenMMReasoner]], [[2026-Step35Flash]]

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
  - 已有尝试: MMaDA 对 GeoQA/CLEVR 用 correctness+CLIP reward, UniGRPO 提升 GSM8K +8.2; LaViDa-R1 在 MathVista +2.4; OpenMMReasoner GSPO 在 9 个 benchmark 平均 +11.6%; EBPO 在数学推理上 G=8 时 +11.28%（验证改进 advantage 估计对困难推理任务的显著效果，机制可迁移到多模态）; Kimi K2.5 视觉 RL 后文本基准微弱提升（MMLU-Pro +1.7%, GPQA-Diamond +2.1%），方向性证据但效应量可能在噪声范围内; LFPO 在 LLaDA 8B 上 GSM8K +9.9%、MATH +7.0%，同时推理步数显著减少（-159 步），验证无似然度范式在推理任务上的有效性
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-OpenMMReasoner]], [[2026-EBPO]], [[2025-KimiK2.5]], [[2026-LFPO]]
- [RL-3b] 🔴 RL 对 grounding 能力的影响
  - 相关论文: [[2026-LaViDa-R1]]
- [RL-3c] 🔴 传统 VLM RL vs dLLM RL 的方法论差异
  - OpenMMReasoner 在传统 VLM（Qwen2.5-VL-7B，AR 生成）上验证 RL（GSPO）有效性，与 dLLM RL 工作（MMaDA/LaViDa-R1）形成架构分野
  - Kimi K2.5 作为 1.04T AR-based MoE VLM，为 dLLM RL 提供性能天花板参考；GRM/Toggle RL/Cross-Modal RL 三个方法论组件可迁移到 dLLM
  - Qwen3-VL 成为 KB 中最强 AR baseline（MMMU 72.3% 32B, 68.9% 8B, beats GPT-4o），使 AR/dLLM RL 方法论对比研究更有参考价值。其 RLHF（200K 偏好对, PPO with KL penalty）是标准 AR RL 管线
  - 核心差异:
    - **Likelihood 估计**: 传统 VLM 直接用 AR 解码的 log p(y|x)；dLLM 需要特殊估计（complementary masking/随机 mask ratio）覆盖完整去噪时间步
    - **KL 正则化**: 传统 VLM 的 token 分布熵较低，KL estimator 方差小；dLLM 高熵分布下 KL 存在争议（LaViDa-R1 移除 KL，MMaDA/DiMOO 保留 KL）
    - **Rollout 策略**: 传统 VLM 直接 ×16 采样；dLLM 需要覆盖去噪时间步（complementary masking 或随机 mask ratio）
  - 开放问题: 两者的 RL 方法能否互相借鉴？GSPO 的稳定性优势能否迁移到 dLLM？GSPO 能否适配 masked diffusion？
  - 潜在思路: 抽象出与架构无关的 PG 组件（advantage 估计、reward shaping、正则化）；GSPO-MDM（将 GSPO 的 group-based advantage 适配到 masked diffusion）；统一 Policy Gradient 框架
  - **进展**: EBPO 的 shrinkage baseline 是首个与架构无关的 PG 改进组件——仅修改 advantage 估计中的 baseline 项，不依赖 AR 还是 masked diffusion 的 likelihood 估计方式，可直接移植到 GSPO/UniGRPO/Self-GRPO。LFPO 是第二个仅适用于 dLLM 的 RL 组件（继 answer-forcing 后）——Theorem 3.1 依赖离散 token 空间 + cross-entropy loss 特性，无法直接迁移到 AR 模型，进一步证实 dLLM RL 正在发展出独立于 AR RL 的方法论体系。Step 3.5 Flash MIS-PO 是第三个架构无关的 PG 改进组件——离散过滤（接受/拒绝）替代连续重要性加权，在 off-policy 严重时完全消除高方差权重项，与 EBPO shrinkage（advantage 方差）正交（MIS-PO 降低 importance weight 方差），两者可叠加。MIS-PO vs IcePop (GLM-5) 形成离散过滤 vs 连续校正两种 off-policy 处理范式
  - 相关论文: [[2025-OpenMMReasoner]], [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]], [[2026-EBPO]], [[2025-KimiK2.5]], [[2026-LFPO]], [[2025-Qwen3-VL]], [[2026-Step35Flash]], [[2026-GLM-5]]

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
  - VidLaDA 首次将 dLLM 扩展到视频理解: 长视频上 dLLM 优于 AR（LongVideoBench +3.2, MLVU +3.0），短视频时序任务 dLLM 弱于 AR（MVBench -10.2）。三重鲁棒性证据: 位置 variance <2% vs AR >10%，时间位置平稳 vs U 型，帧稀疏无损 vs 急剧下降。MARS-Cache 实现 12.5× 推理加速
  - XDLM 通过 stationary noise kernel 统一 MDLM (k=0) 和 UDLM (k=1)，证明纯 mask 噪声 (k=0) 不是离散扩散的最优选择——k=0.1 混合噪声在几乎不损失理解（54.110 vs MDLM 53.650）的前提下大幅提升少步生成质量（FID 54.1 vs MDLM 80.8）。在 LLaDA-8B 上 continual pretraining 验证（MBPP 32 步 15.0 vs 基线 6.8）。对 KB 中所有基于 MDLM 的工作有直接影响——k=0.1 kernel 可作为 drop-in 训练目标替换
  - 相关论文: [[2025-LaViDa]], [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]], [[2025-LaViDa-O]], [[2025-LLaDA-V]], [[2025-ReDiff]], [[2026-Beyond-LM]], [[2025-SDAR-VL]], [[2025-VidLaDA]], [[2025-XDLM]]
- [Diff-1b-1] 🔴 dLLM 信息容量上界悖论——理论与实验不一致
  - VidLaDA Proposition 3.2 证明 bidirectional decoding 信息容量上界**更低**（通过 Markov chain decomposition + Data Processing Inequality），但实验中 dLLM 性能更好
  - 悖论: 更低的信息容量上界意味着更少信息可用，逻辑上应是劣势而非优势。论文将其作为 dLLM 优势的理论依据存在逻辑断层
  - 正确解释可能是: dLLM 的优势来自"更均匀的信息分配"（每个 token 对等参与表征构建），而非"更高信息容量"
  - 核心需求: 建立区分"representation capacity"和"generation capacity"的理论框架——理解任务的性能取决于表征质量而非生成容量
  - 相关论文: [[2025-VidLaDA]]
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
  - VidLaDA MARS-Cache: 帧级 chunk attention（±1 帧局部 O(n·k) vs 全局 O(n²)）+ adaptive anchor token 搜索 + 模态异步刷新（视觉慢/文本快，深层快/浅层慢），12.5× 加速，KB 中最高的单一加速方案。与 Prefix-DLM、ML-Cache 正交可叠加（理论 15-20×）
  - XDLM 少步高质量生成: 通过 k=0.1 混合噪声核训练，8-32 步即可获得高质量结果（ImageNet-1K 16 步 FID 25.77 vs MDLM 80.8），是推理加速的第六个正交维度——减少步数 2-4× 与上述每步加速技术正交叠加，理论总加速 30-100×
  - LFPO RL 轨迹拉直: 对比式速度修正使去噪路径更直，代码平均减少 41.8 步、推理减少 159 步（对比 AGRPO 在 MATH 上增加 +73.6 步），是第七个正交维度——后训练层面减步，与 XDLM 的预训练层面减步机制不同，两者可叠加
  - 相关论文: [[2025-LaViDa]], [[2025-MMaDA]], [[2025-Lumina-DiMOO]], [[2025-LaViDa-O]], [[2025-ReDiff]], [[2025-Sparse-LaViDa]], [[2025-DiffusionVL]], [[2025-SDAR-VL]], [[2025-VidLaDA]], [[2025-XDLM]], [[2026-LFPO]]
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
  - Qwen3-VL-8B 作为同规模但更强的 AR 基座（MMMU 68.9% vs Qwen2.5-VL 更低），提供了检验"stronger AR base → better diffusion finetuning"的直接对比机会。用 Qwen3-VL-8B 重复 DiffusionVL 实验可验证 "5% 数据达 95% AR 性能" 的关系是否是常数
  - 相关论文: [[2025-DiffusionVL]], [[2025-Qwen3-VL]]
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
- [Diff-1j] 🔴 离散扩散噪声核的最优设计空间
  - XDLM 证明 k=0.1（10% uniform + 90% mask）打破 MDLM-UDLM Pareto 前沿，但 k=0.1 是纯经验值无理论推导
  - 核心问题: (1) k 的最优值是否跨任务/跨规模/跨词表稳定？(2) 是否需要 time-varying k（训练早期 k=0 快速收敛，后期 k→0.1 延续学习）？(3) 多模态场景下是否需要 modality-aware k（文本 k_text vs 图像 k_image）？
  - XDLM 的 performance crossover 发现（MDLM 早期强但饱和 vs XDLM/UDLM 后期持续提升）暗示 time-varying k 可能严格更优
  - 潜在思路: 系统性消融（task × scale × vocab × k grid search）；理论推导 k 与任务信息论特性的关系；自适应 k schedule 结合 SDAR-VL 的 PBNC 思想
  - 相关论文: [[2025-XDLM]], [[2025-SDAR-VL]]
- [Diff-1k] 🔴 MDLM 训练饱和的根因——训练目标 vs 训练-推理分布不匹配
  - XDLM 的 performance crossover 发现 MDLM 在 >200K 步后快速饱和，论文归因于"训练目标过简单"（mask-only 二分类）
  - 替代解释: 训练-推理分布不匹配在长训练后累积——模型过度拟合"干净 context + [MASK]"条件分布，推理时的"部分正确 context + [MASK]"越来越偏离训练分布。与 P-Diff-04 从不同角度描述同一现象
  - 区分两种归因很重要: 若是目标过简单 → XDLM 混合噪声是正确方案；若是分布不匹配 → ReDiff 精炼训练可能更直接
  - 潜在思路: 设计对照实验——(1) MDLM + ReDiff 精炼（仅解决分布不匹配）vs (2) XDLM k=0.1（同时改变目标和分布）vs (3) XDLM + ReDiff（双层方案）。若 (1) 也消除饱和，说明分布不匹配是主因
  - 相关论文: [[2025-XDLM]], [[2025-ReDiff]]

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
- [Uni-4a] 🟡 计算瓶颈: 视频 token 数爆炸
  - VidLaDA MARS-Cache 提供推理层面的部分解决: 帧级 chunk attention + anchor 复用 + 模态异步刷新实现 12.5× 加速，将视频推理从不可用（3.3 TPS）提升至可用（33.6 TPS），精度几乎无损（MLVU 50.7% vs 50.2%）
  - 但训练层面的 token 爆炸问题仍未解决（VidLaDA 使用 SigLIP2 + 2×2 spatial pooling 做 4× token 压缩，仍受限于帧数）
  - 相关论文: [[2025-VidLaDA]]
- [Uni-4b] 🔴 时序建模: AR 时序 + Diffusion 空间的混合
- [Uni-4c] 🔴 视频理解和生成的统一是否比图像更难？
- [Uni-4d] 🔴 世界模型视角: 视频生成 = 物理规律学习？

#### [Uni-5] 🟡 统一模型的 Post-training 和 RL
- [Uni-5a] 🟡 统一模型如何做 alignment？
  - 理解和生成分别对齐 vs 联合对齐
  - 已有方案: MMaDA 提供首个 diffusion-native 全链路（预训练→Mixed CoT SFT→UniGRPO RL）; LaViDa-R1 提供统一 PG 框架（SFT+GRPO+self-distillation）; DiMOO 提供四阶段管线 + Self-GRPO 联合 RL; EBPO 的 shrinkage baseline 可作为架构无关的 advantage 估计改进，直接嵌入任何 GRPO 变体; LFPO 提供无似然度的速度场修正范式，天然规避 KL 在高熵 image token 下的方差问题
  - dLLM RL 已形成四种范式: 似然度近似(UniGRPO) / 似然度降方差(complementary masking) / advantage 降方差(EBPO) / 无似然度(LFPO)
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]], [[2026-EBPO]], [[2026-LFPO]]
- [Uni-5b] 🟡 统一模型的 reward 如何设计？→ 连接到 [RL-1c]
  - 理解正确性 + 生成质量 + 一致性 的多目标 reward
  - 已有方案: MMaDA Diversified Reward (correctness/format/CLIP/ImageReward), LaViDa-R1 Multi-Reward (correctness/IoU/EditScore), DiMOO Self-GRPO (模型自身理解能力作为隐式 reward)
  - DiMOO 的自评估方案: 用 entity-relation-value 三元组自动生成理解问题评估 T2I 质量，无需外部 reward model
  - 已发现局限: CLIP reward 不支持 compositional reasoning → 连接到 [RL-1d]; Self-GRPO 有 bootstrapping 偏差风险
  - "一致性" 维度仍完全未被探索
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]]
- [Uni-5c] 🟡 RL 是否能促进理解-生成的协同？
  - 假设: RL 优化"看图生图"任务可能同时提升两种能力
  - 已有证据: MMaDA UniGRPO 同时提升 GSM8K (推理) 和 ImageReward (生成); LaViDa-R1 多任务 RL 各任务均有提升; DiMOO Self-GRPO 验证「一个 loss 同时提升 T2I 和理解」; Kimi K2.5 在 AR MoE (1.04T) 上观察到视觉 RL 后文本基准提升 (MMLU-Pro +1.7%, GPQA-Diamond +2.1%)，首次在非 dLLM 架构上定量观察跨模态协同，但效应量小且缺乏统计检验
  - DiMOO 的联合目标 L(θ) = -∑w(g)(ℓ_T2I + ℓ_MMU) + β·KL，明确将 T2I 和理解 loss 绑定在同一优化中
  - 但机制不清晰: 是共享表示的迁移还是任务间正则化？dLLM 中的联合优化 vs AR 中的顺序训练后迁移，机制可能不同
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]], [[2025-Lumina-DiMOO]], [[2025-KimiK2.5]]

#### [Uni-6] 🔴 Thinking-Aware Image Synthesis 的评估标准
- MMaDA-Parallel 提出 ParaBench（300 challenging prompts，6 维度评估：text quality, text alignment, image quality, image alignment, image consistency, cross-modal output alignment）
- 首次系统化评估"推理感知图像生成"——需要先推理后生成的任务
- 核心问题: 如何量化 CoT 推理对生成质量的因果贡献？faithfulness 如何测量？
- 现有 GenEval/COCO 等基准主要评估 compositional 能力，不评估推理链质量
- 开放问题: (1) 推理链的 faithfulness（扰乱推理是否影响生成质量？）；(2) 推理链的必要性（哪些任务真正需要推理？）；(3) 推理链的可解释性（能否从推理链预测生成结果？）
- 潜在思路: 因果干预实验（mask/扰乱推理链观察生成变化）；推理链质量与生成质量的相关性分析；构建需要多步推理的 T2I 基准
- 相关论文: [[2025-MMaDA-Parallel]]
