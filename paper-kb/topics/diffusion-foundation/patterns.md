# Diffusion 基础 — 跨论文经验规律

> 从多篇论文中归纳出的经验规律。每条 pattern 需要至少 2 篇论文支撑。

## 格式

```
### [P-Diff-xx] 规律名称
- **现象**:
- **支撑论文**:
- **可能解释**:
- **例外情况**:
- **启示**:
```

---

### [P-Diff-01] 8B 规模 Masked Diffusion 模型在理解和生成上可达 Competitive 甚至 SOTA 水平
- **现象**: MMaDA (LLaDA-8B) 在 GSM8K 达 73.4、T2I 超 SDXL/Janus、多模态理解超 Show-o/SEED-X。LaViDa (LLaDA-8B/Dream-7B) 首个 dLLM VLM 家族，在 LLaVa-1.6/Open-LLaVa-Next 上达 competitive，同时贡献 Complementary Masking 训练效率优化和 Prefix-DLM 推理加速。LaViDa-R1 (LaViDa-O-10.4B) 在 Lisa-Grounding 上超越 specialist 模型。DiMOO (LLaDA-8B + ~110M 数据) GenEval 88% 超越 FLUX.1-dev (82%) 和 GPT-4o (84%)，MMMU 58.6%。LaViDa-O (10.4B Elastic-MoT) GenEval 89% w/ reflection，1024² 高分辨率。LLaDA-V 在纯理解场景控制变量对比中 11/18 基准超越 LLaMA3-V，尤其在知识推理 (MMMU +3.2, MMMU-Pro +6.9) 和多图/视频理解上优势明显
- **支撑论文**: [[2025-LaViDa]]（首个 dLLM VLM 家族，理解端 competitive + 训练/推理效率优化）、[[2025-MMaDA]]（多任务 competitive 性能）、[[2026-LaViDa-R1]]（grounding specialist level）、[[2025-Lumina-DiMOO]]（GenEval SOTA 超越专用模型）、[[2025-LaViDa-O]]（1024² + 多任务 SOTA）、[[2025-LLaDA-V]]（纯理解场景控制变量对比 11/18 胜出）、[[2025-MMaDA-Parallel]]（ParaBench 59.8% output alignment，并行生成架构）
- **可能解释**: 离散扩散的 bidirectional attention 在理解任务上天然优于 AR causal attention；mask-predict 目标与 MLM 的相似性使其继承了 BERT 类模型的理解优势；大规模数据 (~110M) 可有效弥补 VQ token 语义不足
- **例外情况**: (1) 纯文本 sequential reasoning 仍弱于 AR（MATH500 36.0 vs Qwen2-7B 更高）；(2) 低级视觉任务（super-resolution, dehazing）表现弱——VQ 信息瓶颈；(3) FID/美学质量维度与 FLUX/SD3 仍有差距；(4) 图表/文档理解系统性弱于 AR (AI2D -3.3, DocVQA -2.3)——需要顺序解析的任务是 dLLM 的结构性劣势 (LLaDA-V 控制变量实验)
- **启示**: 离散扩散不仅是 competitive 的统一模型路线，在 compositional T2I 上已达 SOTA，在纯理解场景也可超越 AR 骨干。主要局限从「高分辨率生成差距」转向「低级视觉任务信息瓶颈」、「纯文本 sequential reasoning」和「结构化信息顺序解析」

---

### [P-Diff-02] dLLM 双向注意力在全局推理任务上系统性优于 AR，在顺序解析任务上系统性弱于 AR
- **现象**: LLaDA-V 在完全控制变量下（同 SigLIP2 + MLP + 训练数据）验证 dLLM 在 MMMU (+3.2), MMMU-Pro (+6.9), MuirBench (+0.9), MLVU (+2.0) 等需要全局推理/多源信息整合的任务上系统性超越 AR (LLaMA3-V)；但在 AI2D (-3.3), DocVQA (-2.3), RealworldQA (-2.8) 等需要顺序解析结构化信息的任务上系统性弱于 AR。LaViDa-R1 在 Lisa-Grounding (mIoU 60.0, P@0.5 66.7) 超越 AR specialist 模型，进一步验证 dLLM 在需要全局空间理解的任务上的优势。LaViDa 的 OCR/文档理解弱势（TextVQA 56.3, DocVQA 59.0）与 LLaDA-V 发现一致（间接证据）；而 LaViDa FIM 在约束诗歌补全上 1.00 vs AR 0.41/0.37，从正面验证 dLLM 在需要全局双向理解的生成任务（infilling）上有压倒性优势。Beyond-LM 的 Hybrid Attention Masking（帧内双向 + 跨序列因果）试图兼顾两种需求，但跨序列因果约束仍会破坏多图推理。
- **支撑论文**: [[2025-LLaDA-V]]（唯一的控制变量对比，18 个基准的 fine-grained 优劣势分布）、[[2026-LaViDa-R1]]（Lisa-Grounding 超越 AR specialist）、[[2025-LaViDa]]（FIM infilling 1.00 vs 0.41——全局双向优势的新维度证据；OCR/文档弱势的间接验证）、[[2025-MMaDA]]（GSM8K 73.4 接近 AR 模型，但 GenEval Position 仅 0.20 暴露空间推理弱点）、[[2026-Beyond-LM]]（Hybrid Attention 的混合尝试）
- **可能解释**: (1) bidirectional attention 允许每个 token 访问完整上下文，在需要综合多条信息的全局推理任务上有结构性优势；(2) mask-predict 训练目标迫使模型从部分信息重建完整语义，训练���更 robust 的表征；(3) 但图表/文档理解需要逐行逐区域的顺序解析，具有隐含拓扑结构依赖（行列标题→数据单元），AR 的 causal 生成自然编码了这种层次结构，而 dLLM 并行预测将所有位置视为等权，丢失结构先验
- **例外情况**: (1) 仅 LLaDA-V 一篇提供了严格控制变量对比，DiMOO/MMaDA 未做同等条件的 AR 基线；(2) 劣势可能部分归因于 LLaDA 预训练在文档类文本上不充分而非架构本身；(3) Hybrid Attention 的混合策略尚未被系统评估
- **启示**: dLLM 在多模态理解领域的竞争力应聚焦于推理密集型任务（知识推理、数学推理、多图综合推理），而非结构化信息解析任务。统一模型设计中可考虑混合 attention 策略（理解阶段 bidirectional + 解析阶段 block-wise causal）弥补此劣势。

---

### [P-Diff-03] 连续扩散（Flow Matching）vs 离散扩散（Masked Diffusion）在统一模型中的 tradeoff
- **现象**: Beyond-LM 是 KB 中唯一使用连续扩散（flow matching）的统一模型，开辟"AR 文本 + 连续扩散视觉"新路线。连续扩散避免 VQ 离散化信息瓶颈，理论上 FID/美学质量更优，但推理需要多步 ODE 求解（vs masked diffusion 的并行采样）。KB 中其他所有统一模型（MMaDA、LaViDa 系列、DiMOO、LaViDa-O）均使用离散 masked diffusion，在 1024² 分辨率 + GenEval 88-89% 已达 competitive/SOTA。
- **支撑论文**: [[2026-Beyond-LM]]（flow matching 连续扩散）、[[2025-MMaDA]]、[[2025-LaViDa]]、[[2026-LaViDa-R1]]、[[2025-Lumina-DiMOO]]、[[2025-LaViDa-O]]（masked diffusion 离散扩散）
- **可能解释**: (1) 连续扩散保留像素级细节，避免 VQ codebook 的信息瓶颈；(2) 离散扩散与 LLM 的 token 框架天然对齐，训练和推理更简单；(3) 离散扩散可享受 Prefix-DLM KV 缓存和 ML-Cache 等加速技术，连续扩散难以应用
- **例外情况**: (1) 缺乏同等条件下的性能对比（相同架构、数据、规模），无法判断哪个更优；(2) 连续扩散在统一模型中的最优实现方式尚未被充分探索；(3) 两种范式在不同任务类型（高分辨率生成 vs 低级视觉 vs 理解）上的优劣可能不同
- **启示**: 连续 vs 离散扩散代表统一模型的两条技术路线，需要系统对比生成质量、训练效率、推理速度、与 LLM 框架的兼容性。当前离散路线已有更多验证，但连续路线的潜力尚未被充分挖掘。

---

### [P-Diff-04] dLLM 训练-推理分布不匹配是幻觉和错误级联的根本原因
- **现象**: ReDiff 发现标准 masked diffusion 在干净数据上训练但从噪声中间输出生成，导致并行解码时早期 token 错误污染后续生成。通过两阶段精炼训练（合成错误 + 模型特定错误）显著减少幻觉（CLAIR +11.2）和错误级联（8 tokens/step 时 +21.06）。MMaDA-Parallel 从架构层面揭示 sequential reasoning-then-generation 的错误传播问题，通过并行生成（interleaved token sequences + bidirectional attention）在 ParaBench 上达到 59.8% output alignment vs Bagel 52.9%
- **支撑论文**: [[2025-ReDiff]]（首次系统性解决 dLLM 训练-推理 gap，两阶段精炼训练）、[[2025-MMaDA-Parallel]]（并行架构解决 sequential 错误传播，ParaBench 评估基准）
- **可能解释**: (1) 标准 masked diffusion 训练假设输入是干净的，但推理时中间状态包含模型自身产生的错误；(2) 并行生成的错误级联效应在训练时未被建模——早期错误通过注意力传播到语义相关 token；(3) 模型缺乏"纠错"能力，只有"去噪"能力；(4) Sequential pipeline 的错误传播是架构层面的问题，parallel 架构通过双向约束消除
- **例外情况**: (1) 合成错误分布与真实错误不匹配时效果受限——随机替换创建均匀错误分布，但真实错误是结构化的；(2) 过度训练纠错可能损害正常生成能力；(3) 并行架构在因果依赖强的任务上可能失效；(4) Interleaving 策略任意性——如何决定哪些 token 交错尚无系统方法
- **启示**: dLLM 错误传播有两条互补解决路线：训练层面（ReDiff 精炼训练）和架构层面（MMaDA-Parallel 并行生成）。两者正交可组合。未来方向：结构化错误注入训练（基于注意力的级联错误模拟）；并行生成 + 主动精炼组合；理论分析错误传播的图结构

---

### [P-Diff-05] 在线自我纠错学习优于通用合成错误训练
- **现象**: ReDiff Stage II 用模型自身草稿 + 专家修正的"草稿-精炼对"训练，效果优于 Stage I 的通用合成错误。DiMOO Self-GRPO 用模型自身理解能力作为隐式 reward 评估 T2I 质量，同样体现"模型学习修正自己的特定错误"优于"通用数据增强"。Sparse-LaViDa 的 register tokens 可视为一种"模型自身的压缩表示"——64 个可学习 token 作为被截断 mask token 的全局摘要
- **支撑论文**: [[2025-ReDiff]]（Stage II 在线自我纠错，专家模型 o4-mini 修正模型草稿）、[[2025-Lumina-DiMOO]]（Self-GRPO 自评估，用 entity-relation-value 三元组生成理解问题评估 T2I）、[[2025-Sparse-LaViDa]]（register tokens 作为模型内部的容量补偿机制）
- **可能解释**: (1) 通用合成错误无法覆盖模型的特定失败模式——模型的错误分布是分布依赖且训练中演化的；(2) 模型自身草稿与推理时的错误分布完美对齐；(3) 专家模型/自评估提供高质量修正目标，形成"自我改进"循环；(4) 在线学习能适应模型错误分布的动态变化；(5) 模型内部的可学习表示（如 register tokens）可以自适应地补偿推理时的信息损失
- **例外情况**: (1) 专家模型质量不足或有系统性偏差（如 LaViDa-R1 报告的 VLM-as-judge 幻觉问题）——偏差会传递到学生模型；(2) 模型草稿质量太差导致错误分布过于极端；(3) Self-GRPO 有 bootstrapping 偏差风险（模型评估自己的输出）；(4) 可能陷入局部最优（只修正某类错误）；(5) Register tokens 数量不足时无法充分补偿容量损失
- **启示**: dLLM 后训练应采用"模型特定错误驱动"策略，而非仅依赖通用数据增强。这为 [[P-RL-03]] Mixed SFT 冷启动提供了新维度——SFT 数据可包含模型自身草稿的修正版本。未来方向：多专家集成减少偏差传播、与 RL 结合（外部 reward 覆盖专家偏差）

---

### [P-Diff-06] Register Tokens 作为虚拟容量补偿是 MDM 稀疏化的关键
- **现象**: Sparse-LaViDa 引入 64 个可学习 register tokens 作为被截断 mask token 的"全局摘要"，补偿稀疏化导致的容量损失。实验显示 0 register 时 GenEval 0.76、FID 9.32，64 register 时 GenEval 0.78、FID 7.63，容量补偿效果显著
- **支撑论文**: [[2025-Sparse-LaViDa]]（首次在 MDM 推理加速中使用 register tokens 作为容量补偿机制）
- **可能解释**: (1) MDM 的 mask token 虽不携带语义信息但提供表征空间（M·d 维），截断后模型失去"虚拟工作记忆"；(2) Register tokens 通过可学习参数（64·d 维）重新引入容量，充当"虚拟 mask token"的角色；(3) 这是有损压缩——64 << M 时（如 M=256），register 必须学会将数百个潜在槽位的信息压缩到 64 个全局摘要中；(4) 类似于 ViT 的 [CLS] token、BERT 的 [SEP] token——特殊 token 作为全局信息的压缩表示
- **例外情况**: (1) 仅一篇论文支撑，需要在其他 MDM 模型（MMaDA、DiMOO）上独立验证；(2) Register 数量不足时无法充分补偿容量损失；(3) Register 数量过多时增加计算成本，抵消截断收益；(4) 训练不充分时 register 可能退化为无效 token
- **启示**: MDM 稀疏化优化不能简单截断 token，必须通过某种机制补偿容量损失。Register tokens 提供了一种通用的容量补偿范式，可能适用于其他稀疏化场景（如 sparse attention、token pruning）。未来方向：动态 register 数量（根据截断比例调整）；层次化 register（不同层使用不同数量）；在其他 MDM 模型上验证通用性

---

### [P-Diff-07] Step-Causal Attention Mask 可在保留双向上下文的前提下实现 KV 缓存
- **现象**: Sparse-LaViDa 的 step-causal attention mask（clean token 间双向 + mask token 仅看 clean token）实现 clean token KV 跨步缓存，质量几乎无损（GenEval 0.78 vs 0.77）。LaViDa Prefix-DLM 的前缀 causal mask 是 step-causal 的特例——仅前缀部分 causal，response 部分 bidirectional
- **支撑论文**: [[2025-Sparse-LaViDa]]（step-causal mask 实现 clean token KV 缓存，1.95-2.83× 加速）、[[2025-LaViDa]]（Prefix-DLM 前缀 causal mask，3.9× 加速）
- **可能解释**: (1) 双向 attention 和 KV 缓存并非完全对立——通过非对称 attention pattern（部分 token 双向 + 部分 token 因果）可以兼得；(2) Step-causal mask 打破全局依赖——clean token 的 KV 仅依赖于其他 clean token 和 register token（这些在多步中不变），因此可缓存；(3) 关键是这种因果约束不破坏 mask token 预测所需的双向信息流——mask token 仍能看到完整上下文（所有 clean token）；(4) 训练-推理一致性至关重要——训练时不使用 step-causal mask 但推理时使用会导致 GenEval 崩溃至 0.71（vs 0.78）
- **例外情况**: (1) 需要 mask token 间交互的任务（如协同生成）可能受因果约束影响；(2) 极端动态场景（每步 clean token 集合剧烈变化）缓存命中率低；(3) 多轮对话中历史轮次的表征可能无法根据新轮次内容更新
- **启示**: 为 dLLM 推理优化提供了新的设计空间——不必在"全双向"和"全因果"之间二选一，可以设计混合 attention pattern 实现加速-质量平衡。这为其他 dLLM 架构（如连续扩散、flow matching）的 KV 缓存优化提供了思路。未来方向：block-wise step-causal mask（局部双向 + 全局因果）；自适应 attention pattern（根据 token 稳定性动态调整）

---

### [P-Diff-08] MDM 推理加速的四个正交维度：前缀缓存 + mask 截断 + 稳定 token 缓存 + 块级并行
- **现象**: LaViDa Prefix-DLM（前缀 KV 缓存 3.9×）、Sparse-LaViDa（mask token 截断 1.95-2.83×）、DiMOO ML-Cache（稳定 token 缓存 2×）、DiffusionVL Block Diffusion（块间 KV 缓存 2×）四种加速技术作用于不同 token 子集（输入条件 vs 未确定 vs 已确定 vs 块级结构），理论上可叠加实现 10-15× 总加速
- **支撑论文**: [[2025-LaViDa]]（Prefix-DLM 前缀缓存）、[[2025-Sparse-LaViDa]]（mask token 截断 + step-causal mask）、[[2025-Lumina-DiMOO]]（ML-Cache 稳定 token 缓存）、[[2025-DiffusionVL]]（Block Diffusion 块级并行 + KV 缓存）
- **可能解释**: (1) 四种技术的加速来源完全不同——Prefix-DLM 消除跨步的冗余前缀计算，Sparse-LaViDa 消除冗余 mask token 计算，ML-Cache 消除已确定 token 的重复计算，Block Diffusion 通过块级并行减少串行步骤；(2) 作用对象正交——前缀（visual+prompt）vs 未确定 token（mask）vs 已确定 token（高 logit clean）vs 块级结构（跨块 KV 复用）；(3) 实现机制正交——causal mask（Prefix-DLM）vs 稀疏表示+register tokens（Sparse-LaViDa）vs 选择性跳过（ML-Cache）vs 混合 attention（Block Diffusion）
- **例外情况**: (1) 四篇论文独立验证各自技术，但尚无工作验证组合效果；(2) 四种缓存机制的 attention mask 协调复杂（前缀 causal + response step-causal + ML-Cache 选择性跳过 + Block Diffusion 块内双向/块间因果）；(3) 缓存对象可能重叠（已 unmask 的 token 通常也是高 logit 的），叠加收益可能不是乘法的；(4) 四重缓存的显存管理复杂，可能成为��瓶颈；(5) Block Diffusion 的块间信息衰减问题需要更多实验验证（长序列/多块场景）
- **启示**: 为 dLLM 推理效率追赶 AR 模型提供了清晰路线图——不是单一技术的突破，而是多个正交技术的系统组合。这也揭示了 MDM 推理的四个独立优化维度，未来工作可以在每个维度上继续深化（如自适应前缀范围、动态 register 数量、更精确的稳定性判断、自适应块大小）。最有价值的后续工作是验证四重加速的实际效果和工程可行性

---

### [P-Diff-09] AR-to-Diffusion 知识迁移的数据效率优势存在混淆变量，无法证明扩散训练本身的优越性
- **现象**: DiffusionVL 用 738K 样本（5%）扩散微调达到 35.1 MMMU-Pro，接近 AR 基线 Qwen2.5-VL-7B (36.7)，显著超越 LLaDA-V 从零训练 15M 样本的数据效率。A2D-VL 在相同数据量下性能相当（35.0 vs 35.1），但需要 annealing 策略。表面上看数据效率提升约 20×（15M/738K≈20）
- **支撑论文**: [[2025-DiffusionVL]]（738K 达 95% 性能）、[[2025-A2D-VL]]（同期工作验证）、[[2025-LLaDA-V]]（15M 从零训练基线）
- **批判性分析**:
  - **混淆变量**: DiffusionVL 的 base AR 模型（Qwen2.5-VL）本身用**数百万甚至上亿样本**预训练，这些知识被"免费"继承。真正的对比应该是（Qwen 预训练数据 + 738K 扩散微调）vs（15M 从零训练扩散），算上 Qwen 的预训练成本，DiffusionVL 的总数据量可能**远超** LLaDA-V
  - **AR 模型质量是隐藏变量**: 论文未测试"弱 AR 模型 + 扩散微调"的下界。如果 base AR 模型仅 30% 准确率，扩散微调后可能仅 28-32%——这说明性能主要来自 AR 基础，而非扩散训练的增益
  - **多模态 dLLM 训练数据太少**: LLaDA-V 的 15M 样本相比 AR-VLM 的训练数据（通常 50M-100M+）仍然是小规模。扩散模型可能需要**更多数据**才能从零学会视觉-语言对齐
  - **违反第一性原理**: 这篇论文**无法证明基于 diffusion 的模型能够完全依托自己的训练方式达到很好的性能**，因为它的成功可能主要来自 AR 模型的知识迁移，而非扩散训练本身的优越性
- **例外情况**: (1) 仅在 VLM 场景验证，纯语言或其他模态未知；(2) 基座 AR 模型质量差时无法弥补基础能力缺陷；(3) 微调数据分布与下游任务严重不匹配时效率下降；(4) 如果给扩散模型**同等规模**的训练数据（50M-100M），是否能超越 AR？这个问题 DiffusionVL 无法回答
- **启示**: AR-to-Diffusion 转换证明了"AR→Diffusion 转换是可行的"，但**未证明**"Diffusion 训练本身优于 AR 训练"。需要在**同等训练数据、同等计算资源**下对比从零训练的 AR vs Diffusion，才能判断哪个范式本质上更优。当前结论应谨慎表述为"AR 初始化可大幅降低扩散 VLM 训练成本"，而非"扩散模型数据效率高于 AR"

---

### [P-Diff-10] Block Diffusion（块扩散）提供 AR-Diffusion 融合的中间范式
- **现象**: DiffusionVL 的 Block Diffusion 将序列分为固定大小块（默认 block size=8），块内使用双向注意力并行去噪，块间保持因果依赖自回归生成。这实现了 KV-cache 跨块复用（2× 加速）和变长生成支持。SDAR-VL 采用类似的块状扩散策略，并通过 ABNS/EMRS/PBNC 解决训练稳定性问题。块大小 1-16 时性能差异 <1.1 分（MMMU-Pro），块大小 8 为最佳平衡点
- **支撑论文**: [[2025-DiffusionVL]]（Block Diffusion 策略，AR→Diffusion 转换 + KV-cache 复用 + 2× 加速）、[[2025-SDAR-VL]]（块状扩散 VLM，训练稳定性优化）
- **可能解释**: (1) Block Diffusion 是 "全序列扩散" 和 "纯自回归" 的中间点——块内并行获得扩散优势（去噪、infilling），块间因果保持 AR 优势（变长、KV-cache）；(2) 块内 token 共享噪声水平，可同时预测减少串行步骤；(3) 是 LaViDa Prefix-DLM 的泛化——从"前缀因果"扩展到"任意块边界因果"
- **例外情况**: (1) 块边界处可能出现不连贯（实验显示影响较小）；(2) 块大小固定不适应所有任务（短回复 vs 长文档）；(3) 块间信息衰减——第 N 块访问第 1 块信息需经过 N-1 次中继，长序列/多块场景下性能曲线未验证；(4) 与位置编码交互可能产生意外行为（需验证 RoPE 兼容性）
- **启示**: Block Diffusion 提供了 AR 和 Diffusion 融合的务实方案——不需要完全放弃 AR 的 KV-cache 和变长生成优势。与 P-Diff-08 中的四维加速互补——Block Diffusion 是第四个正交加速维度（块级并行 + KV 缓存）。未来方向：自适应块大小（根据任务/内容动态调整）；分层块结构（粗粒度块规划 + 细粒度块内填充）；扩展到视频（时间维度分块 + 空间维度扩散）

