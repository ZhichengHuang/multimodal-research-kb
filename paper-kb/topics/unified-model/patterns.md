# 统一模型 — 跨论文经验规律

> 从多篇论文中归纳出的经验规律。每条 pattern 需要至少 2 篇论文支撑。

## 格式

```
### [P-Uni-xx] 规律名称
- **现象**:
- **支撑论文**:
- **可能解释**:
- **例外情况**:
- **启示**:
```

---

### [P-Uni-01] Masked Diffusion 统一框架下的跨模态训练产生正向协同
- **现象**: MMaDA Mixed Long-CoT SFT 阶段，文本推理、多模态理解、T2I 生成的所有指标同步提升（Figure 6），没有观察到显著 seesaw 效应。LaViDa-R1 的统一后训练在数学推理、grounding、image editing 上均取得提升。DiMOO Self-GRPO 联合优化 T2I+理解，验证「一个 loss 同时提升两种能力」
- **支撑论文**: [[2025-MMaDA]]（Figure 6 跨模态协同效应）、[[2026-LaViDa-R1]]（统一 PG 框架多任务 RL 正向转移）、[[2025-Lumina-DiMOO]]（Self-GRPO 联合优化 T2I+理解）
- **可能解释**: 统一 mask-predict 目标让文本/图像 token 共享同一语义恢复能力；CoT 格式的学习可迁移至图像生成的推理；多任务训练提供正则化防过拟合
- **例外情况**: (1) 任务比例严重失衡时某任务可能欠训练；(2) RL 阶段的梯度冲突可能削弱协同效应；(3) 仅在 8-10B 验证，更大规模行为未知
- **启示**: dLLM 统一训练是有基础的方向，不需要引入解耦来避免冲突，但需仔细设计任务采样比例和 reward scale

---

### [P-Uni-02] 在足够数据规模下，模态无关全共享架构可匹配非对称分支架构
- **现象**: DiMOO（模态无关全共享，~110M 训练数据）GenEval 88% ≈ LaViDa-O（非对称 Elastic-MoT 8B+2.4B）GenEval 89% (w/ reflection)。两者在理解上差距较大（MMMU 58.6% vs 45.1%），但 LaViDa-O 训练数据量远小于 DiMOO。Beyond-LM 的 Modality-Specific FFN（部分共享）在从零训练场景达到 competitive 性能，提供"全共享"和"完全分离"之间的新数据点。
- **支撑论文**: [[2025-Lumina-DiMOO]]（全共享 + 大规模数据 GenEval 88%）、[[2025-LaViDa-O]]（Elastic-MoT 非对称 GenEval 89%）、[[2026-Beyond-LM]]（部分共享 Modality-Specific FFN）
- **可能解释**: 在数据充足时，全共享架构的参数可通过梯度更新隐式学习模态专用表征，非对称分支的归纳偏置不再是必须的。「用数据换简洁性」vs「用架构换数据效率」是等效的 tradeoff。部分共享（共享 attention，分离 FFN）可能在两者之间提供平衡。
- **例外情况**: (1) 数据受限场景下非对称架构可能更优（LaViDa-O 用更少数据达到接近性能）；(2) 全共享在极端任务不对称时可能出现梯度冲突；(3) Modality-Specific FFN 相对全共享和完全分离的增量价值尚未被量化
- **启示**: 架构选择（全共享 vs 部分共享 vs 解耦）应结合数据可得性和计算预算决策；当前 dLLM regime 下数据规模可能比架构选择更重要。

---

### [P-Uni-03] MoE 架构可有效协调模态间容量需求不对称，缩小扩展律差距
- **现象**: Beyond-LM 发现 MoE 架构（per-modality shared experts, G=16）将视觉-语言扩展律指数差距从 0.10（dense）缩小到 0.05，在相同激活参数下性能匹配或超过 dense 模型。MoE 自然涌现模态专家分化，无需显式监督。
- **支撑论文**: [[2026-Beyond-LM]]（MoE 扩展律系统性研究）
- **可能解释**: (1) Per-modality shared experts 为每个模态提供基础能力保障，routing experts 实现任务级专业化；(2) MoE 的稀疏激活允许更大的总参数量，为数据饥渴的视觉模态提供更多容量；(3) 自然涌现的专家分化减少了跨模态干扰
- **例外情况**: (1) 仅在预训练阶段验证，后训练和 RL 阶段的 scaling law 未知；(2) 51:1 的视觉-语言数据不平衡可能导致 routing collapse；(3) 最优 granularity (G) 和 expert 数量尚无系统性指导
- **启示**: MoE 是 LaViDa-O Elastic-MoT 任务级路由的自动化替代方案，可作为统一模型扩展的默认架构选择。需要进一步研究 MoE 配置空间（G/E/shared expert 设计）在不同训练阶段的最优值。

---

### [P-Uni-04] RAE 统一视觉表示提供连续扩散路线的第三条路径
- **现象**: Beyond-LM 使用 RAE (Representation Autoencoder, SigLIP 2) 作为单一连续编码器同时服务理解和生成，避免 VQ 离散化信息瓶颈。这与 DiMOO 的纯 VQ 和 LaViDa-O 的 SigLIP+VQ 双路方案形成三种不同策略。
- **支撑论文**: [[2026-Beyond-LM]]（RAE 统一表示 + flow matching）、[[2025-Lumina-DiMOO]]（纯 VQ）、[[2025-LaViDa-O]]（SigLIP+VQ 双路）
- **可能解释**: (1) RAE 的语义级表示空间平衡���判别性和生成性；(2) 连续表示保留更多视觉细节，理论上 FID/美学质量更优；(3) 与 flow matching 连续扩散框架天然对齐
- **例外情况**: (1) RAE 在极高分辨率生成（2048²）时可能遇到信息瓶颈；(2) 低级视觉任务（super-resolution）表现可能弱于专用编码器；(3) 连续扩散的多步 ODE 求解无法享受 masked diffusion 的并行采样加速
- **启示**: 三种策略各有优劣——「用数据换简洁性」(DiMOO 纯 VQ) vs 「用架构换数据效率」(LaViDa-O 双路) vs 「用连续表示换统一性」(Beyond-LM RAE)。选择应基于数据规模、计算预算和任务需求。

---

### [P-Uni-05] 并行生成架构可消除 Sequential Pipeline 的错误传播
- **现象**: MMaDA-Parallel 发现 sequential reasoning-then-generation 存在错误传播问题——推理阶段错误污染生成阶段条件输入，导致性能退化。通过并行架构（interleaved token sequences + bidirectional attention）在 ParaBench 上达到 59.8% output alignment vs Bagel 52.9%（+6.9 pp）
- **支撑论文**: [[2025-MMaDA-Parallel]]（并行架构解决错误传播）、[[2025-ReDiff]]（精炼训练解决错误传播）
- **可能解释**: (1) Interleaved token sequence 使文本和图像在同一去噪过程中互相约束，错误可被双向纠正；(2) Unified mask prediction 使跨模态对齐在训练时隐式优化；(3) 局部性原理——相关的文本和图像 token 空间上相邻，缩短跨模态信息流路径
- **例外情况**: (1) 文本-图像因果依赖强的任务（如"先描述图像再根据描述生成新图像"）天然是 sequential 的，parallel 会破坏因果链；(2) 极端长度不对称（1024 文本 vs 256 图像 token）导致某模态梯度信号被稀释；(3) Interleaving 策略任意性——如何决定哪些 token 交错尚无系统方法
- **启示**: 并行生成是解决错误传播的架构层面方案，与 ReDiff 的训练层面方案（精炼训练）正交互补。未来方向：并行生成 + 主动精炼组合；基于语义对应的 interleaving 策略；混合 parallel-sequential 架构（根据任务类型动态选择）

---

### [P-Uni-06] Vision-first vs LLM-first 初始化路线的对称性 tradeoff
- **现象**: dLLM 统一模型存在两种对称的初始化路线——LLM-first（从预训练 LLM 出发扩展到视觉）vs Vision-first（从预训练 T2I 模型出发扩展到文本理解）。LLM-first 路线由 LaViDa 系列（8B + SigLIP + LLaVA-style）和 MMaDA/DiMOO（LLaDA + MAGVIT-v2/aMUSEd-VQ）代表，文本推理强但视觉生成质量受限于 VQ tokenizer。Vision-first 路线由 Muddit（1B Meissonic + CLIP text→MM-DiT 架构）代表，视觉生成质量高（FID 7.81）但文本能力受限（CLIP 77 token 硬限制，无 CoT 推理）
- **支撑论文**: [[2025-LaViDa]]（LLM-first，8B，ScienceQA 81.49 但 TextVQA 56.3）、[[2025-MMaDA]]（LLM-first，8B，GSM8K 73.4）、[[2025-Lumina-DiMOO]]（LLM-first，8B，MMMU 58.6%）、[[2025-Muddit]]（Vision-first，1B，FID 7.81 但 CLIP 77 token 限制）
- **可能解释**: (1) 预训练模型的初始化决定了模型的"能力天花板"——LLM 预训练提供强推理能力但弱视觉先验，T2I 预训练反之；(2) LLM-first 路线可利用大规模语言预训练的知识，Vision-first 路线利用高质量视觉先验；(3) 两种路线的最终收敛点可能相同（足够数据和计算下），但路径和效率不同
- **例外情况**: (1) 参数量不对称——LLM-first 均为 8-10B，Vision-first 仅 1B Muddit，公平对比缺失；(2) DiffusionVL 的 AR→Diffusion 转换提供了第三条路径（AR 初始化 + 扩散微调）；(3) Beyond-LM 从零训练（无 LLM/T2I 初始化）也达到 competitive 性能
- **启示**: 当前 LLM-first 路线在文本理解和推理上有明显优势，是主流选择。Vision-first 路线的价值在于视觉生成质量——未来可探索两种路线的融合（如 LLM 骨干 + T2I adapter）。选择应基于目标任务的模态偏好：理解为主选 LLM-first，生成为主选 Vision-first
