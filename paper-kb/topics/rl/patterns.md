# RL — 跨论文经验规律

> 从多篇论文中归纳出的经验规律。每条 pattern 需要至少 2 篇论文支撑。

## 格式

```
### [P-RL-xx] 规律名称
- **现象**:
- **支撑论文**:
- **可能解释**:
- **例外情况**:
- **启示**:
```

---

### [P-RL-01] CLIP-based Reward 不支持 Compositional Reasoning
- **现象**: CLIP score 作为 T2I reward 在简单 prompt 上可用，但对需要空间推理、属性绑定、多对象计数等 compositional 能力的 prompt 无效
- **支撑论文**: [[2025-MMaDA]]（承认 CLIP reward 无法评估需要世界知识的 T2I prompt，GenEval Position 仅 0.20）、[[2026-LaViDa-R1]]（废弃 PickScore/CLIP-based reward，明确列为 anti-pattern，改用 EditScore）
- **可能解释**: CLIP 的对比学习目标优化全局图文相似度而非细粒度语义对应；CLIP 的 zero-shot 特性不具备 compositional reasoning 所需的逻辑推理能力
- **例外情况**: 对简单 single-object T2I prompt（MMaDA GenEval Single Object 0.99）CLIP reward 有效
- **启示**: dLLM T2I RL 需要开发支持 compositional reasoning 的专用 reward model，或使用 VLM-as-judge（但有幻觉风险）

---

### [P-RL-02] dLLM RL 的 Log-Likelihood 估计需覆盖完整去噪时间步 + 对齐推理结构
- **现象**: 固定 mask ratio（如 d1 的 t=1 全 mask）劣于动态覆盖策略。MMaDA 的随机 mask ratio p∈[0,1] 优于 d1；LaViDa-R1 的 complementary masking (w=1) 在互补覆盖和方差降低上进一步改进。Complementary masking 技术源自 LaViDa 的训练效率优化（ScienceQA +67% on 200K subset），后被 LaViDa-R1 扩展为 RL likelihood estimator。**新增维度**: SPG 的 block-wise masking 揭示了**masking 结构**同样关键——按推理时的 semi-AR block 结构（前面 clean + 中间随机 mask + 后面全 mask）生成 MC 样本，在 Countdown 上比 random masking 提升 +23.9%，说明训练-推理分布对齐是被严重忽视的效率瓶颈
- **支撑论文**: [[2025-LaViDa]]（Complementary Masking 原创，训练场景的 antithetic 方差降低）、[[2025-MMaDA]]（结构化随机 mask ratio 优于 d1 的固定策略）、[[2026-LaViDa-R1]]（complementary masking 覆盖全部 token，w=1 避免权重不平衡）、[[2025-SPG]]（block-wise masking 对齐 semi-AR 推理结构，Countdown +23.9%）
- **可能解释**: (1) 完整覆盖时间步提供更准确的 ELBO 估计，减少 train-inference gap；(2) i.i.d. 采样有遗漏问题，互补（antithetic）采样降低方差约一半；(3) **masking 结构对齐**: LLaDA 推理时实际遇到的中间状态是结构化的（block-wise semi-AR 顺序），random masking 生成的中间状态与此严重不匹配。Block-wise masking 消除此分布偏移，使 MC 估计更准确
- **例外情况**: 当序列很短时，覆盖策略和结构的差异不显著；SPG 的 block-wise masking 训练后在 random unmasking 推理下仍泛化良好（Figure 6），说明结构对齐更多是改善训练质量而非限制推理策略
- **启示**: dLLM RL 的 likelihood 估计方案有三个维度需优化：(1) 时间步覆盖（complementary masking）；(2) masking 结构（block-wise masking）；(3) 上下界选择（SPG 的三明治策略）。三个维度正交可叠加

---

### [P-RL-03] Mixed SFT 冷启动是 dLLM RL 成功的必要前提
- **现象**: MMaDA 在 Mixed Long-CoT SFT 之后做 UniGRPO RL（GSM8K: 17.4→65.2→73.4，SFT 贡献远大于 RL）；LaViDa-R1 在 100K steps SFT 后做 5K steps RL。DiMOO 在 Stage III SFT (30M) 后做 Stage IV Self-GRPO。三者均采用先 SFT 再 RL 的两阶段策略。LaViDa 的 Stage 3a CoT 推理增强（从 VL-Rethinker-7B 蒸馏 19.2K 样本）直接成为 LaViDa-R1 SFT 冷启动的组成部分，验证了 CoT 蒸馏数据可作为 SFT 阶段的有效来源
- **支撑论文**: [[2025-LaViDa]]（Stage 3a CoT 蒸馏作为后续 RL 冷启动的数据基础）、[[2025-MMaDA]]（CoT SFT 带来 +47.8, RL 再 +8.2）、[[2026-LaViDa-R1]]（Stage 1 SFT 远大于 Stage 2 RL 步数）、[[2025-Lumina-DiMOO]]（Stage III SFT 30M → Stage IV Self-GRPO）
- **可能解释**: dLLM 从预训练到 RL 的 policy gap 过大，需要 SFT 提供合理的初始 policy 和统一 CoT 格式；dLLM 的并行去噪使随机探索效率更低，更依赖有意义的起点
- **例外情况**: 对极简单任务（格式遵循等）可能直接 RL；d1 直接对 LLaDA 做 RL 但效果有限
- **启示**: dLLM 后训练应为"大规模 CoT SFT → 有限步 RL 精炼"，SFT 数据质量和覆盖度是 RL 天花板的关键

---

### [P-RL-04] dLLM RL 中 KL 正则化的有效性与 token 分布特性相关（非二元争议）
- **现象**: MMaDA (UniGRPO) 和 DiMOO (Self-GRPO) 保留 KL 正则化，训练稳定。LaViDa-R1 发现 image token NLL>6 时 KL estimator 方差极大导致训练发散，移除 KL 改用 SFT loss 作为隐式正则化后训练更稳定。MMaDA-Parallel ParaRL 的 KL 策略未知，但基于 MMaDA 经验可能保留 KL
- **支撑论文**: [[2025-Lumina-DiMOO]]（保留 KL，Self-GRPO 在 1024² 下训练稳定）、[[2026-LaViDa-R1]]（移除 KL，NLL>6 导致发散）、[[2025-MMaDA]]（保留 KL，UniGRPO 稳定）、[[2025-MMaDA-Parallel]]（KL 策略未明确，可能保��）
- **可能解释**: KL 有效性不是二元选择（保留 vs 移除），而是连续谱，取决于多个因素：(1) **image token 占比**——DiMOO 在 4096+ image tokens + 较少 text tokens 场景下仍稳定，但其 aMUSEd-VQ (codebook 8192) 的 token 分布可能比 LaViDa-R1 使用的 tokenizer 更集中（熵更低）；(2) **序列总长度**——长序列中 KL 估计的方差累积更大；(3) **KL 系数 β**——DiMOO 可能使用较小的 β 值压制方差；(4) **codebook 特性**——不同 VQ tokenizer 的离散分布形态（均匀 vs 偏斜）直接影响 KL 估计质量。EBPO 的 shrinkage baseline 提供了第三条路径——不修改 KL 项，而是通过更好的 advantage baseline 降低整体训练方差
- **例外情况**: 纯文本 RL 中 KL 正则化普遍有效（token 分布熵较低）；当 image token 占比低或使用低熵 codebook 时 KL 可能仍然安全
- **启示**: dLLM RL 的正则化策略可能需要根据 token 分布特性（熵、NLL、codebook 大小、image token 占比）自适应选择——在低熵条件下保留 KL，高熵条件下用 SFT 正则化替代。理论上需要推导离散高熵分布下 KL estimator 的方差界

---

### [P-RL-05] 轨迹级 RL 提供密集监督但 Credit Assignment 更困难
- **现象**: MMaDA-Parallel ParaRL 沿整个去噪轨迹应用 CLIP-based alignment reward（在中间步骤也施加 reward），而 MMaDA UniGRPO 仅在最终输出评估 token 级 reward。ParaRL 在 ParaBench 上达到 59.8% output alignment，但 CLIP reward 在中间噪声步骤的可靠性未验证
- **支撑论文**: [[2025-MMaDA-Parallel]]（ParaRL 轨迹级 reward）、[[2025-MMaDA]]（UniGRPO token 级 reward）
- **可能解释**: (1) 轨迹级 reward 提供密集监督——每个去噪步骤都有 reward 信号，而非仅最终输出；(2) 与 diffusion 过程对齐——每步都应朝"更对齐"方向演化；(3) Timestep-dependent weighting（文本 1/t，图像 constant）平衡不同模态的去噪难度曲线
- **例外情况**: (1) CLIP reward 在中间噪声步骤的有效性存疑——噪声图像是 out-of-distribution 输入，CLIP 评分可靠性未验证；(2) Credit assignment 困难——哪个时间步的哪个 token 导致了最终结果？Timestep-dependent weighting 是粗粒度解决方案；(3) Sparse trajectory sampling 可能遗漏关键转折点
- **启示**: 轨迹级 vs token 级 RL 是 tradeoff——前者密集监督但 credit assignment 难，后者信号稀疏但更精确。未来方向：Process Reward Model for dLLM（评估中间去噪状态质量）；混合策略（关键步骤用轨迹 reward，其他用 token reward）；自适应 reward 频率（根据去噪阶段动态调整）

---

### [P-RL-06] Test-time Scaling 是训练不足的"后期补丁"而非能力突破
- **现象**: dMLLM-TTS 在三个基座模型上的 GenEval 提升呈现明显规律——基座越弱，收益越大（MMaDA +29.4% > Muddit +26.4% > DiMOO +17.9%）。DiMOO 作为最强基座（GenEval 0.78），推理时搜索仅提升 +17.9%；而 MMaDA 作为最弱基座（GenEval 0.51），推理时搜索提升高达 +29.4%
- **支撑论文**: [[2025-dMLLM-TTS]]（首个 dLLM test-time scaling 系统化研究）
- **可能解释**: (1) Test-time scaling 本质上是用推理时计算成本弥补训练不足——当基座模型训练充分时（如 DiMOO 的 110M 数据），推理时搜索的边际收益递减；(2) 强基座模型的生成质量分布更集中（top-k 候选差距小），分层搜索的筛选效果有限；(3) 弱基座模型的错误更明显，Self-Verified Feedback 的纠错空间更大
- **例外情况**: (1) 对特定任务（如 compositional generation），即使强基座也可能从 test-time scaling 受益；(2) 如果推理时搜索引入新的能力（如 LaViDa-R1 Tree Search 的中间状态探索），可能突破训练时上界
- **启示**: 能训练阶段搞定的就不在 test-time 做——test-time scaling 更像"后期补丁"而非根本性能力提升。优先级应该是：训练时优化（数据规模、RL、精炼训练）> 推理时搜索。Test-time scaling 的价值场景：(1) 基座模型已固定无法重新训练；(2) 需要动态调整质量-效率 tradeoff（低延迟场景用少步数，高质量场景用多轨迹搜索）；(3) 特定困难样本需要额外计算投入

---

### [P-RL-07] SFT 数据多样性比规模更重要（答案多样性 + 跨域混合）
- **现象**: OpenMMReasoner 通过 ×8 采样扩展答案多样性 + 跨域混合（数学推理数据），在 874K SFT 样本上实现 Qwen2.5-VL-7B 基座 11.6% 提升；明确指出过度过滤（aggressive filtering）损害多样性反而降低性能。MMaDA 的 Mixed Long-CoT SFT 跨领域混合（数学+代码+视觉推理）贡献 +47.8（远大于 RL 的 +8.2），其中多领域混合是关键设计
- **支撑论文**: [[2025-OpenMMReasoner]]（×8 答案多样性 + 跨域混合 > 数据规模；过度过滤损害多样性）、[[2025-MMaDA]]（Mixed Long-CoT SFT 跨领域混合贡献远大于 RL）
- **可能解释**: (1) 答案多样性让模型学习"推理空间的分布"而非"单点最优解"，避免过拟合单一推理模式；(2) 跨域混合利用推理能力的模态无关性——数学推理的逻辑链构建能力正迁移到视觉推理（OpenMMReasoner 验证 "textual reasoning transfers alongside strengthened multimodal reasoning"）；(3) 多样性提供隐式正则化，防止模型过拟合特定模态的表面特征
- **例外情况**: (1) 任务只有唯一正确答案且推理路径固定时（如简单分类），多样性扩展效果有限；(2) 教师模型在目标任务上表现差时，×8 采样可能放大错误；(3) 跨域混合比例不当可能导致负迁移（如混入过多文本数据导致模型忽视视觉信息）
- **启示**: SFT 数据构造应"优先多样性，其次规模"——先确保答案多样性（多条推理路径）和领域多样性（跨域混合），再考虑增加数据量。过度过滤是 anti-pattern。与 [P-RL-03] 互补——P-RL-03 说明 SFT 是 RL 的天花板，本 pattern 进一步指出 SFT 天花板的关键是多样性而非规模

---

### [P-RL-08] GRPO 的局部 Baseline 在小 Group Size + 高难度任务下系统性失效
- **现象**: GRPO 使用纯局部 group mean 作为 baseline。当 group size G≤8 且任务困难时（成功率 ~10%），P(group 内全部失败) = 0.9^G（G=4 时 65.6%，G=8 时 43.0%），导致 advantage 全为零、梯度消失，策略完全不更新。EBPO 用 James-Stein 收缩估计将局部 group mean 与全局策略成功率插值，在 G=8 时比 GRPO 平均 +11.28%（Qwen3-8B, AIME/MATH-500/OlympiadBench）。G=32 时差距消失（62.41% vs 62.91%），说明这是小样本校正而非根本性算法改进
- **支撑论文**: [[2026-EBPO]]（首次系统分析 GRPO saturated failure 问题，提出 shrinkage baseline 解决方案，G=8 时 +11.28%）
- **可能解释**: (1) 收缩将极端估计（全 0）"拉"向全局先验，在 saturated failure 下产生 -S_q × μ_glob 的非零负梯度信号；(2) James-Stein 类收缩估计在同时估计多个均值时严格优于 sample mean（MSE 更低），这在统计学上有成熟理论保证；(3) Topic-coherent sampling 进一步降低 Welford 在线估计器在异质分布间的震荡方差
- **例外情况**: (1) G≥32 时局部统计量已足够精确，收缩边际收益消失；(2) Gaussian 近似假设 θ_q ~ N(μ_glob, τ²) 在极度偏斜的 reward 分布（如 dLLM image token）下可能不最优；(3) Welford 在线估计器在训练早期策略快速变化时可能系统性滞后
- **启示**: 多模态 RL 的 group size 天然受限（图像/视频生成 rollout 成本高），saturated failure 率更高（多模态推理更困难），是 shrinkage baseline 的天然应用场景。三种应对 saturated failure 的机制可分层组合：(1) EBPO shrinkage baseline（统计层面，从全失败 group 提取负向梯度，架构无关）；(2) LaViDa-R1 answer-forcing（生成层面，通过 dLLM inpainting 注入正向样本，dLLM 专有）；(3) Complementary masking w=1（估计层面，降低 likelihood 方差，dLLM 专有）。三者正交可叠加——shrinkage 改进 baseline，answer-forcing 改进 sample 质量，complementary masking 改进 likelihood 估计。**注意**: 目前仅 1 篇论文支撑且仅在纯文本 LLM 上验证，需等 dLLM 实验验证后升级为成熟 pattern

---

### [P-RL-09] Off-Policy 处理存在离散过滤 vs 连续校正两种正交范式
- **现象**: Step 3.5 Flash MIS-PO 用离散过滤（接受/拒绝二元决策）替代连续重要性加权，在 token 和轨迹层面仅对信赖域内样本优化，完全消除高方差权重项。GLM-5 IcePop 用连续 pop 函数平滑校正训练-推理策略不匹配（β=2, 非对称 clipping ε_low=0.2/ε_high=0.28）。两种方法从不同角度解决异步 RL 的 off-policy 问题，均在 frontier 规模验证（Step 3.5 196B/11B, GLM-5 744B/40B）
- **支撑论文**: [[2026-Step35Flash]]（MIS-PO 离散过滤，IMO 85.4% / LiveCodeBench 86.4%）、[[2026-GLM-5]]（IcePop 连续校正，Intelligence Index 50 分开源 #1）
- **可能解释**: (1) 离散过滤更激进——完全丢弃 off-policy 样本消除方差，但牺牲信息（部分 off-policy 样本可能仍有价值）；(2) 连续校正更保守——保留所有样本但重加权，方差降低但不消除；(3) 两者的优劣取决于 off-policy 严重程度——极端 off-policy（长时 agent rollout）时离散过滤优势大，温和 off-policy（短文本生成）时连续校正信息利用率更高
- **例外情况**: (1) 数据稀缺时离散过滤可能丢弃过多样本导致训练信号饥饿（需足够大 rollout 批次）；(2) MIS-PO 固定阈值假设任务同质——异构任务（数学推理 vs 创意写作）策略漂移速度不同，单一阈值可能失效；(3) 训练冷启动阶段策略接近随机，off-policy ratio 极端，两种方法都面临初始信号不足
- **启示**: 对 dLLM RL，image token NLL>6 导致 off-policy 严重程度远超纯文本——MIS-PO 的离散过滤可能更适合 dLLM RL。两种范式与 EBPO shrinkage baseline 均正交（MIS-PO/IcePop 处理 importance weight 方差，EBPO 处理 advantage baseline 方差），三者可分层叠加。开放问题: 是否存在自适应方案（off-policy ratio 低时连续校正、高时离散过滤）？per-modality 阈值（text τ=0.9, image τ=0.7）是否可行？

---

### [P-RL-10] Expert Collapse ≠ Routing Collapse（MoE 训练稳定性的深层诊断）
- **现象**: Step 3.5 Flash 和 GLM-5 独立发现——MoE 训练中专家可以出现 activation norm 归零（功能性死亡），但 routing 统计（gate probability、dispatch counts）完全正常。传统监控 routing 统计无法发现此类 collapse，必须监控 per-expert activation norm（RMS/mean at FFN intermediate），其中 max-to-median ratio 是最可靠的 early indicator
- **支撑论文**: [[2026-Step35Flash]]（196B/11B MoE，288 routed experts，17.2T tokens 仅 1 次 loss spike）、[[2026-GLM-5]]（744B/40B MoE，256 experts，同样观察到 activation death 而非 routing collapse）
- **可能解释**: Expert Collapse 的因果链（Step 3.5 识别）：高频 bi-gram 触发专家专门化 → pre-norm 允许单专家主导 → SwiGLU 稀疏激活放大幅度 → Muon 低秩更新加速 collapse。Routing 统计正常是因为 gate 仅决定 dispatch 比例，不反映专家内部的激活健康状态。Activation clipping（FFN intermediate 逐元素裁剪）优于 weight clipping，因为直接切断了激活放大环节
- **例外情况**: (1) 小规模 MoE（10B 以下）可能不出现此不稳定模式；(2) 非 Muon 优化器的 collapse 根因链可能不同（Muon 特有的 Newton-Schulz 正交化是链条一环）；(3) 适度的专家专化是期望行为——"Expert Collapse"和"健康专化"的边界未被量化
- **启示**: (1) 所有大规模 MoE 训练应将 per-expert activation norm 监控（特别是 max-to-median ratio）纳入标准监控指标，不能仅依赖 routing 统计；(2) 对多模态 MoE（如 Beyond-LM per-modality shared experts），需要 per-modality activation norm 监控——51:1 数据不平衡下视觉专家可能优先饿死；(3) Activation clipping 可能需要 per-modality 阈值（视觉 vs 语言 FFN 激活分布不同）

---

### [P-RL-11] 序列级/轨迹级 Importance Ratio 比 Token 级更稳定（pre-pattern）
- **现象**: OpenMMReasoner GSPO 将 GRPO 的 token 级 importance ratio 乘积替换为序列级单一 ratio，训练"收敛更快、reward 更高、行为更稳定"。Step 3.5 Flash MIS-PO 使用离散过滤（接受/拒绝），隐式在轨迹级操作。两者共同暗示序列/轨迹级方案在长序列场景优于 token 级
- **支撑论文**: [[2025-OpenMMReasoner]]（GSPO 序列级 ratio > GRPO token 级 ratio，在 9 个多模态推理 benchmark 上验证）、[[2026-Step35Flash]]（MIS-PO 轨迹级离散过滤，在 IMO/LiveCodeBench frontier 难度验证）
- **可能解释**: Token 级 importance ratio 的乘积方差随序列长度 L 指数增长（∏ᵢ π/π_ref 的方差 ~ exp(L)），长 CoT 推理尤其严重。序列级 ratio 等价于 token 级的几何均值，个体波动被平滑。更小 ε clipping threshold 进一步限制更新幅度
- **例外情况**: (1) 序列级 ratio 牺牲 token 级信用分配精细度——所有 token 获得相同 importance weight，关键推理步骤（<10% tokens）的梯度被稀释；(2) 仅 2 篇论文，且来自非常不同的场景（VLM reasoning vs frontier LLM training），需更多验证
- **启示**: 对 dLLM RL 特别有意义——image token NLL>6 导致 token 级 ratio 爆炸更严重，序列级方案可能天然更适合。GSPO-MDM（将序列级 ratio 适配到 masked diffusion）是值得探索的方向。**注**: 当前为 pre-pattern，需 dLLM 上的实验验证后升级为成熟 pattern
