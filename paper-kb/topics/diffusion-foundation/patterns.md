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
- **现象**: MMaDA (LLaDA-8B) 在 GSM8K 达 73.4、T2I 超 SDXL/Janus、多模态理解超 Show-o/SEED-X。LaViDa (LLaDA-8B/Dream-7B) 首个 dLLM VLM 家族，在 LLaVa-1.6/Open-LLaVa-Next 上达 competitive，同时贡献 Complementary Masking 训练效率优化和 Prefix-DLM 推理加速。LaViDa-R1 (LaViDa-O-10.4B) 在 Lisa-Grounding 上超越 specialist 模型。DiMOO (LLaDA-8B + ~110M 数据) GenEval 88% 超越 FLUX.1-dev (82%) 和 GPT-4o (84%)，MMMU 58.6%。LaViDa-O (10.4B Elastic-MoT) GenEval 89% w/ reflection，1024² 高分辨率。LLaDA-V 在纯理解场景控制变量对比中 11/18 基准超越 LLaMA3-V，尤其在知识推理 (MMMU +3.2, MMMU-Pro +6.9) 和多图/视频理解上优势明显。**RL 后训练新 SOTA**: StableDRL 在 LLaDA-8B-Instruct 上 GSM8K 84.2%（超越此前 MMaDA 73.4）、MATH500 41.8%，在 SDAR-8B 块状扩散上 AIME'24 16.7%（超越 AR Qwen3-8B 10.0%），首次实现 dLLM 全参数 RL 训练 >1000 步
- **支撑论文**: [[2025-LaViDa]]（首个 dLLM VLM 家族，理解端 competitive + 训练/推理效率优化）、[[2025-MMaDA]]（多任务 competitive 性能）、[[2026-LaViDa-R1]]（grounding specialist level）、[[2025-Lumina-DiMOO]]（GenEval SOTA 超越专用模型）、[[2025-LaViDa-O]]（1024² + 多任务 SOTA）、[[2025-LLaDA-V]]（纯理解场景控制变量对比 11/18 胜出）、[[2025-MMaDA-Parallel]]（ParaBench 59.8% output alignment，并行生成架构）、[[2025-VidLaDA]]（首个 dLLM 视频理解模型，LongVideoBench 61.4 > LLaVA-Video 58.2, MLVU 53.4 > LLaVA-Video 50.4）、[[2026-Omni-Diffusion]]（首个三模态 dLLM any-to-any 模型，text+image+speech 统一建模，ASR WER 7.05 超越 AnyGPT 8.50）、[[2026-StableDRL]]（首次全参数 dLLM RL 训练 >1000 步，GSM8K 84.2% / AIME'24 16.7% 超越 AR Qwen3-8B）
- **可能解释**: 离散扩散的 bidirectional attention 在理解任务上天然优于 AR causal attention；mask-predict 目标与 MLM 的相似性使其继承了 BERT 类模型的理解优势；大规模数据 (~110M) 可有效弥补 VQ token 语义不足
- **例外情况**: (1) 纯文本 sequential reasoning 仍弱于 AR（MATH500 36.0 vs Qwen2-7B 更高）；(2) 低级视觉任务（super-resolution, dehazing）表现弱——VQ 信息瓶颈；(3) FID/美学质量维度与 FLUX/SD3 仍有差距；(4) 图表/文档理解系统性弱于 AR (AI2D -3.3, DocVQA -2.3)——需要顺序解析的任务是 dLLM 的结构性劣势 (LLaDA-V 控制变量实验)；(5) 短视频时序任务弱于 AR（MVBench 59.4 vs Qwen2.5-VL 69.6, -10.2）——差距远超图像域（-2 至 -3），视频时序因果推理是 dLLM 更严重的结构性弱点 (VidLaDA)
- **启示**: 离散扩散不仅是 competitive 的统一模型路线，在 compositional T2I 上已达 SOTA，在纯理解场景也可超越 AR 骨干，在长视频理解上也展现优势。主要局限从「高分辨率生成差距」转向「低级视觉任务信息瓶颈」、「纯文本 sequential reasoning」、「结构化信息顺序解析」和「短视频时序因果推理」

---

### [P-Diff-02] dLLM 双向注意力在全局推理任务上系统性优于 AR，在顺序解析任务上系统性弱于 AR
- **现象**: LLaDA-V 在完全控制变量下（同 SigLIP2 + MLP + 训练数据）验证 dLLM 在 MMMU (+3.2), MMMU-Pro (+6.9), MuirBench (+0.9), MLVU (+2.0) 等需要全局推理/多源信息整合的任务上系统性超越 AR (LLaMA3-V)；但在 AI2D (-3.3), DocVQA (-2.3), RealworldQA (-2.8) 等需要顺序解析结构化信息的任务上系统性弱于 AR。LaViDa-R1 在 Lisa-Grounding (mIoU 60.0, P@0.5 66.7) 超越 AR specialist 模型，进一步验证 dLLM 在需要全局空间理解的任务上的优势。LaViDa 的 OCR/文档理解弱势（TextVQA 56.3, DocVQA 59.0）与 LLaDA-V 发现一致（间接证据）；而 LaViDa FIM 在约束诗歌补全上 1.00 vs AR 0.41/0.37，从正面验证 dLLM 在需要全局双向理解的生成任务（infilling）上有压倒性优势。Beyond-LM 的 Hybrid Attention Masking（帧内双向 + 跨序列因果）试图兼顾两种需求，但跨序列因果约束仍会破坏多图推理。
- **支撑论文**: [[2025-LLaDA-V]]（唯一的控制变量对比，18 个基准的 fine-grained 优劣势分布）、[[2026-LaViDa-R1]]（Lisa-Grounding 超越 AR specialist）、[[2025-LaViDa]]（FIM infilling 1.00 vs 0.41——全局双向优势的新维度证据；OCR/文档弱势的间接验证）、[[2025-MMaDA]]（GSM8K 73.4 接近 AR 模型，但 GenEval Position 仅 0.20 暴露空间推理弱点）、[[2026-Beyond-LM]]（Hybrid Attention 的混合尝试）、[[2025-VidLaDA]]（视频维度独立验证: 长视频全局推理 dLLM 优——LongVideoBench +3.2, MLVU +3.0；短视频时序任务 dLLM 弱——MVBench -10.2；三重鲁棒性: 位置 variance <2% vs >10%，时间位置平稳 vs U 型，帧稀疏无损 vs 急剧下降）
- **可能解释**: (1) bidirectional attention 允许每个 token 访问完整上下文，在需要综合多条信息的全局推理任务上有结构性优势；(2) mask-predict 训练目标迫使模型从部分信息重建完整语义，训练出更 robust 的表征；(3) 但图表/文档理解需要逐行逐区域的顺序解析，具有隐含拓扑结构依赖（行列标题→数据单元），AR 的 causal 生成自然编码了这种层次结构，而 dLLM 并行预测将所有位置视为等权，丢失结构先验；(4) 视频时空维度放大了双向 attention 优势——对象、关系、事件分布在时空中无固有 left-to-right 顺序，但短视频时序因果推理（动作顺序、因果关系、状态变化）是 dLLM 更严重的结构性弱点
- **例外情况**: (1) 仅 LLaDA-V 一篇提供了严格控制变量对比，DiMOO/MMaDA 未做同等条件的 AR 基线；(2) 劣势可能部分归因于 LLaDA 预训练在文档类文本上不充分而非架构本身；(3) Hybrid Attention 的混合策略尚未被系统评估；(4) bidirectional 优势**依赖足够长的时序范围**——短视频优势消失且差距更大（MVBench -10.2 远超图像域 -2 至 -3），说明 dLLM 在短时序因果推理能力不足（VidLaDA）
- **启示**: dLLM 在多模态理解领域的竞争力应聚焦于推理密集型任务（知识推理、数学推理、多图综合推理、长视频全局理解），而非结构化信息解析和短时序因果推理任务。统一模型设计中可考虑混合 attention 策略（理解阶段 bidirectional + 解析/时序阶段 block-wise causal）弥补此劣势。Qwen3-VL 的 MRoPE 消融（+4.2%）从 AR 侧证实了空间-时序位置编码的关键性——dLLM 的弱势可能不仅源于注意力方向（单向 vs 双向），也源于位置编码设计（无显式空间编码 vs MRoPE 三轴编码），详见 [[P-PT-04]]。

---

### [P-Diff-03] 连续扩散（Flow Matching）vs 离散扩散（Masked Diffusion）在统一模型中的 tradeoff
- **现象**: Beyond-LM 是 KB 中唯一使用连续扩散（flow matching）的统一模型，开辟"AR 文本 + 连续扩散视觉"新路线。连续扩散避免 VQ 离散化信息瓶颈，理论上 FID/美学质量更优，但推理需要多步 ODE 求解（vs masked diffusion 的并行采样）。KB 中其他所有统一模型（MMaDA、LaViDa 系列、DiMOO、LaViDa-O）均使用离散 masked diffusion，在 1024² 分辨率 + GenEval 88-89% 已达 competitive/SOTA。
- **支撑论文**: [[2026-Beyond-LM]]（flow matching 连续扩散）、[[2025-MMaDA]]、[[2025-LaViDa]]、[[2026-LaViDa-R1]]、[[2025-Lumina-DiMOO]]、[[2025-LaViDa-O]]（masked diffusion 离散扩散）
- **可能解释**: (1) 连续扩散保留像素级细节，避免 VQ codebook 的信息瓶颈；(2) 离散扩散与 LLM 的 token 框架天然对齐，训练和推理更简单；(3) 离散扩散可享受 Prefix-DLM KV 缓存和 ML-Cache 等加速技术，连续扩散难以应用
- **例外情况**: (1) 缺乏同等条件下的性能对比（相同架构、数据、规模），无法判断哪个更优；(2) 连续扩散在统一模型中的最优实现方式尚未被充分探索；(3) 两种范式在不同任务类型（高分辨率生成 vs 低级视觉 vs 理解）上的优劣可能不同
- **启示**: 连续 vs 离散扩散代表统一模型的两条技术路线，需要系统对比生成质量、训练效率、推理速度、与 LLM 框架的兼容性。当前离散路线已有更多验证，但连续路线的潜力尚未被充分挖掘。

---

### [P-Diff-04] dLLM 训练-推理分布不匹配是幻觉和错误级联的根本原因
- **现象**: ReDiff 发现标准 masked diffusion 在干净数据上训练但从噪声中间输出生成，导致并行解码时早期 token 错误污染后续生成。通过两阶段精炼训练（合成错误 + 模型特定错误）显著减少幻觉（CLAIR +11.2）和错误级联（8 tokens/step 时 +21.06）。MMaDA-Parallel 从架构层面揭示 sequential reasoning-then-generation 的错误传播问题，通过并行生成（interleaved token sequences + bidirectional attention）在 ParaBench 上达到 59.8% output alignment vs Bagel 52.9%。XDLM 的 k=0.1 混合噪声从噪声核设计角度隐式缓解此问题——训练时让模型暴露于"token 被随机替换"的情况，使推理时面对包含错误预测的中间状态更 robust。NAP 从数据层面提供第四条路径——并行推理轨迹数据使训练时模型暴露于"多个不完整推理流同时存在"的中间状态，与推理时的并行去噪状态更接近
- **支撑论文**: [[2025-ReDiff]]（首次系统性解决 dLLM 训练-推理 gap，两阶段精炼训练）、[[2025-MMaDA-Parallel]]（并行架构解决 sequential 错误传播，ParaBench 评估基准）、[[2025-XDLM]]（混合噪声核从噪声设计层面隐式缓解分布不匹配，performance crossover 发现暗示 MDLM 饱和可能源自此问题）、[[2026-NAP]]（并行推理轨迹数据重构从数据层面缓解训练-推理分布不匹配，GSM8K 256步 +14.4%）
- **可能解释**: (1) 标准 masked diffusion 训练假设输入是干净的，但推理时中间状态包含模型自身产生的错误；(2) 并行生成的错误级联效应在训练时未被建模——早期错误通过注意力传播到语义相关 token；(3) 模型缺乏"纠错"能力，只有"去噪"能力；(4) Sequential pipeline 的错误传播是架构层面的问题，parallel 架构通过双向约束消除；(5) XDLM 的 uniform 噪声使训练分布更接近推理时的"包含错误的中间状态"分布，是噪声核设计层面的隐式对齐；(6) NAP 的并行数据结构使训练分布更接近推理时的并行去噪状态——多条推理流同时进展，与推理时的并行 unmask 动态一致
- **例外情况**: (1) 合成错误分布与真实错误不匹配时效果受限——随机替换创建均匀错误分布，但真实错误是结构化的；(2) 过度训练纠错可能损害正常生成能力；(3) 并行架构在因果依赖强的任务上可能失效；(4) Interleaving 策略任意性——如何决定哪些 token 交错尚无系统方法；(5) XDLM 的缓解是隐式的，不如 ReDiff 的显式错误注入直接；(6) NAP 的并行数据重构需要教师模型能生成多条高质量独立推理路径，困难问题上难以满足，且序列长度 O(m) 膨胀带来 ~12× 训练成本增长
- **启示**: dLLM 错误传播现有四条互补解决路线：噪声核设计层面（XDLM 混合噪声核）、训练策略层面（ReDiff 精炼训练）、架构层面（MMaDA-Parallel 并行生成）和数据层面（NAP 并行推理轨迹）。四者正交可组合——XDLM 预训练 + NAP 并行数据后训练 + ReDiff 精炼可能是三层方案。未来方向：四层组合验证；结构化错误注入训练（基于注意力的级联错误模拟）；理论分析错误传播的图结构

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

### [P-Diff-08] MDM 推理优化的多个正交维度：前缀缓存 + mask 截断 + 稳定 token 缓存 + 块级并行 + 视频帧级加速 + 少步高质量生成 + RL 轨迹拉直 + 解码偏置校正 + 自适应序列长度
- **现象**: LaViDa Prefix-DLM（前缀 KV 缓存 3.9×）、Sparse-LaViDa（mask token 截断 1.95-2.83×）、DiMOO ML-Cache（稳定 token 缓存 2×）、DiffusionVL Block Diffusion（块间 KV 缓存 2×）、VidLaDA MARS-Cache（帧级 chunk attention + anchor 复用 + 模态异步刷新 12.5×）、XDLM 少步推理（混合噪声核使 8-32 步即可高质量，步数减少 2-4×）、LFPO RL 轨迹拉直（对比式速度修正使去噪路径更直，代码 -41.8 步、推理 -159.0 步）九种加速技术作用于不同维度，理论上可叠加实现 30-100× 总加速。NAP 的"并行数据+并行解码"是第十个正交维度——从数据层面消除 AR 解码收敛，在低步数（高并行度）下显著提升质量（GSM8K 256步 +14.4%）
- **支撑论文**: [[2025-LaViDa]]（Prefix-DLM 前缀缓存）、[[2025-Sparse-LaViDa]]（mask token 截断 + step-causal mask）、[[2025-Lumina-DiMOO]]（ML-Cache 稳定 token 缓存）、[[2025-DiffusionVL]]（Block Diffusion 块级并行 + KV 缓存）、[[2025-VidLaDA]]（MARS-Cache 帧级局部 attention + adaptive anchor + 模态异步刷新，12.5× 加速，KB 中最高的单一加速方案）、[[2025-XDLM]]（stationary noise kernel k=0.1 使 8-32 步即可高质量生成，ImageNet-1K 16 步 FID 25.77 vs MDLM 80.8）、[[2026-LFPO]]（对比式速度修正的 flow matching 轨迹拉直效应，代码平均减少 41.8 步、推理减少 159.0 步，对比 AGRPO 在 MATH 上增加 +73.6 步）、[[2026-Omni-Diffusion]]（Position Penalty 校正两端解码偏置避免图像重复 + Adaptive Token Length Assignment 根据模态长度相关性减少冗余 mask token）、[[2026-NAP]]（并行推理轨迹数据重构 + Parallel-Forced Decoding，GSM8K 256步 +14.4%，是第十个正交加速维度）
- **可能解释**: (1) 九种技术的加速来源完全不同——Prefix-DLM 消除跨步的冗余前缀计算，Sparse-LaViDa 消除冗余 mask token 计算，ML-Cache 消除已确定 token 的重复计算，Block Diffusion 通过块级并行减少串行步骤，MARS-Cache 利用视频帧间 attention 局部性和视觉 token 时间稳定性降低注意力计算量，XDLM 通过训练时混合噪声使模型 robust 于少步推理的残余噪声从而减少所需步数，LFPO 通过 RL 后训练的 flow matching 轨迹拉直效应使去噪路径更直从而减少所需步数��NAP 通过并行数据重构消除训练数据的隐式顺序性使模型学会真正的并行解码从而在低步数下提升质量；(2) 作用对象正交——前缀（visual+prompt）vs 未确定 token（mask）vs 已确定 token（高 logit clean）vs 块级结构（跨块 KV 复用）vs 帧级结构（帧间 attention 稀疏化 + 模态差异化刷新）vs 步数维度-预训练（训练目标层面减少所需推理步数）vs 步数维度-后训练（RL 优化使去噪轨迹更高效）vs 解码偏置校正（Position Penalty）vs 数据-解码协同（NAP 并行数据+并行解码）；(3) XDLM 和 LFPO 都减少步数但机制正交——XDLM 通过混合噪声核改变训练目标使模型 robust 于少步推理（预训练层面），LFPO 通过 RL 速度场修正使去噪路径更直（后训练层面），两者可叠加
- **例外情况**: (1) 九篇论文独立验证各自技术，但尚无工作验证组合效果；(2) 七种缓存/加速机制的 attention mask 协调复杂；(3) 缓存对象可能重叠（已 unmask 的 token 通常也是高 logit 的），叠加收益可能不是乘法的；(4) Block Diffusion 的块间信息衰减问题需要更多实验验证（长序列/多块场景）；(5) MARS-Cache 的帧级 chunk attention 仅适用于视频/多帧场景，单图理解不适用；(6) XDLM 的 k=0.1 在多模态统一场景未验证，ML-Cache 的 max logit 稳定性估计在混合噪声下需要检验；(7) LFPO 的轨迹拉直效应缺乏严格理论解释——是 flow matching 的固有性质还是对比式优化的特异效果尚不明确；AGRPO 反而增加步数说明并非所有 RL 方法都有此效果
- **启示**: 为 dLLM 推理效率追赶 AR 模型提供了清晰路线图——不是单一技术的突破，而是多个正交技术的系统组合。九维加速覆盖预训练（XDLM）、后训练（LFPO）、数据-解码协同（NAP）、推理（其余六种）四个阶段，形成完整的加速管线。最有价值的后续工作是验证多维加速的实际叠加效果（重点是 XDLM 预训练减步 × LFPO 后训练减步 × Prefix-DLM × ML-Cache 四维叠加）。Qwen3-VL 的 MoE 稀疏激活（40% latency reduction）为 dLLM 提供了第八个潜在维度——dLLM + MoE = 并行生成 + 稀疏计算双重效率提升（但 dLLM 中 MoE 的 routing 行为与 AR 可能不同，需验证）

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

---

### [P-Diff-11] 离散扩散混合噪声核 (k=0.1) 打破 MDLM-UDLM 理解-生成 Pareto 前沿
- **现象**: XDLM 通过 stationary noise kernel K = (k/N)J + μM 统一 MDLM (k=0, 纯 mask) 和 UDLM (k=1, 纯 uniform)，在 k=0.1 处同时保持接近 MDLM 的理解能力（零样本基准 54.110 vs MDLM 53.650）和远超 MDLM 的少步生成质量（ImageNet-1K 16 步 FID 25.77 vs 80.8）。在 LLaDA-8B 上 continual pretraining 验证（MBPP 32 步 15.0 vs 基线 6.8）。同时发现 performance crossover：MDLM 训练早期（<200K 步）有优势但快速饱和，XDLM/UDLM 后期持续提升
- **支撑论文**: [[2025-XDLM]]（stationary noise kernel 统一框架，k=0.1 sweet spot，performance crossover 发现，scalar formulation 计算效率优化）
- **可能解释**: (1) 少量 uniform 噪声 (k=0.1) 在训练时让模型暴露于"token 被随机替换"（vs 仅被替换为 [MASK]），使模型在少步推理时更 robust 于残余噪声——因为推理中间状态包含错误预测的 token（类似 uniform 噪声），而非仅有 [MASK]；(2) 90% mask 噪声保留了 MDLM 的理解优势——mask-predict 目标与 MLM 类似，在理解任务上天然有效；(3) performance crossover 可能源自 MDLM 的训练-推理分布不匹配在长训练后累积——XDLM 的 uniform 噪声分量隐式缓解了此问题（与 P-Diff-04 从不同角度描述同一现象）
- **例外情况**: (1) 仅一篇论文支撑，k=0.1 的最优性需要在其他 dLLM（DiMOO、LaViDa-O、MMaDA）上独立验证；(2) 仅在纯文本 (OWT) 和纯图像 (ImageNet-1K) 上分别验证，多模态统一场景（混合 token 序列）下最优 k 可能不同；(3) k=0.1 无理论推导，不同任务/规模/词表下最优值可能不同；(4) LLaDA-8B 验证为 continual pretraining 而非从零训练，与 P-Diff-09 类似的混淆变量
- **启示**: XDLM 对 KB 中所有基于 MDLM (k=0) 的工作有"底层训练目标升级"效应——k=0.1 kernel 可作为 drop-in 替换，仅修改 loss 函数即可能提升少步生成质量。最有价值的后续验证是在 DiMOO 四阶段训练管线中将 k 从 0 替换为 0.1。同时，XDLM 与 ReDiff 的精炼训练从不同层次（噪声核设计 vs 训练策略）解决训练-推理分布不匹配问题，两者正交可组合为双层方案

---

### [P-Diff-13] Pre-Infilling 是 MDM 独有的输出格式控制机制，AR 模型无法实现
- **现象**: Omni-Diffusion 的 Special Token Pre-Infilling 在初始全 mask 序列的 0.25L 位置预填 [BoS] token，引导模型前 25% 生成文本、后 75% 生成语音，使语音生成可显式 attend 到同步生成的文本内容。DiMOO 的 `<end-of-line>` token 在每行图像 token 末尾插入，保留 2D 结构信息。LaViDa 的 FIM（fill-in-the-middle）通过前中后重排实现任意位置 infilling（约束诗歌 1.00 vs AR 0.41）。LaViDa-O 的 planning token 在生成前规划输出结构。这四种技术均利用 MDM 初始序列可编辑的独有特性——AR 模型必须按序列顺序生成，无法在生成前预控制输出格式
- **支撑论文**: [[2026-Omni-Diffusion]]（Special Token Pre-Infilling 控制多模态输出比例和位置）、[[2025-Lumina-DiMOO]]（`<end-of-line>` 保留 2D 结构）、[[2025-LaViDa]]（FIM infilling 1.00 vs AR 0.41/0.37）、[[2025-LaViDa-O]]（planning token 规划输出结构）
- **可能解释**: (1) MDM 的初始序列是全 mask 的"画布"，可在任意位置预填 token 作为生成锚点；(2) 预填的特殊 token 将生成问题分解为多个子区域，每个子区域独立去噪但通过 bidirectional attention 共享全局信息；(3) AR 模型的因果生成使其只能"从左到右"依次填充，无法预设中间锚点
- **例外情况**: (1) 预填比例（如 25%/75%）是经验值，不同任务最优比例不同；(2) 纯语音或纯图像生成不需要 Pre-Infilling（单模态无需输出格式控制）；(3) 预填位置错误可能导致生成质量下降（如 [BoS] 位置偏差导致文本-语音不同步）
- **启示**: MDM 的"输出编排"能力是 AR 模型无法复制的独有优势，构成 MDM 独有的 capability 维度。将 Pre-Infilling 从特定任务（文本+语音同步）推广为通用输出控制框架——预填 [BoI]/[BoS]/[BoT] 在任意位置控制图像/语音/文本的输出位置、比例和交错模式——是 MDM 统一模型的重要研究方向

---

### [P-Diff-12] dLLM RL 的七种范式：似然度近似 / 似然度降方差 / 似然度 bounds / advantage 降方差 / 无似然度 / off-policy 离散过滤 / 优化公式鲁棒化
- **现象**: dLLM RL 已形成七条正交的优化路径：(1) UniGRPO（MMaDA）用结构化随机 mask ratio 近似 likelihood，方差可控但仍有近似误差；(2) Complementary Masking（LaViDa-R1）用互补 mask 和 w=1 权重降低 likelihood 估计方差，更精确但仍需 ODE 类计算；(3) **SPG 三明治 bounds**——根据 advantage 符号选择性使用 ELBO（正）或 EUBO（负）构成有效下界，首次解决"最小化 ELBO 不等价于最小化 log-likelihood"的数学缺陷；(4) EBPO 用 James-Stein shrinkage 改进 advantage baseline 估计，架构无关但不解决 likelihood 估计本身的问题；(5) LFPO 通过 Theorem 3.1 完全绕过 likelihood 计算，在 logit 空间做精确速度场修正，是唯一的无似然度方案；(6) **MIS-PO 离散过滤**（Step 3.5 Flash）——用接受/拒绝二元决策替代连续重要性加权，在 token 和轨迹层面仅对信赖域内样本优化，完全消除高方差权重项；(7) **StableDRL 优化公式鲁棒化**——不改善 likelihood 估计质量或 advantage 估计质量，而是修改 GRPO 更新公式本身以容忍噪声（无条件裁剪 + 自归一化），首次实现 dLLM 全参数 RL 训练 >1000 步
- **支撑论文**: [[2025-MMaDA]]（UniGRPO 似然度近似）、[[2026-LaViDa-R1]]（Complementary Masking 似然度降方差 + answer-forcing 训练信号恢复）、[[2025-SPG]]（三明治 ELBO+EUBO bounds + block-wise masking 对齐推理结构）、[[2026-EBPO]]（Shrinkage baseline advantage 降方差）、[[2026-LFPO]]（无似然度速度场修正）、[[2026-Step35Flash]]（MIS-PO 离散过滤，IMO 85.4% / LiveCodeBench 86.4%）、[[2026-StableDRL]]（无条件裁剪+自归一化，LLaDA-8B GSM8K 84.2%, SDAR-8B AIME'24 16.7%）
- **可能解释**: (1) 七种范式解决 dLLM RL 管线中的不同瓶颈——UniGRPO 和 Complementary Masking 解决 likelihood 估计精度/方差，SPG 解决 likelihood bounds 的方向性偏差（负 advantage 下 ELBO 无效），EBPO 解决 advantage baseline 估计，LFPO 从根本上消除 likelihood 估计需求，MIS-PO 消除 off-policy importance weight 方差，StableDRL 消除 GRPO 更新公式对噪声 ratio 的脆弱性；(2) 范式间的正交性使得组合成为可能——StableDRL + EBPO + answer-forcing 可叠加为三层 dLLM RL 稳定性栈（详见 [[stabledrl-complete-three-layer-stack]]），分别解决梯度尖峰/信号消失/探索失败三个正交病理。NAP 的并行推理轨迹 SFT 可作为 Layer 0（稳定的并行冷启动数据），为 RL 阶段提供更好的 policy 起点；(3) 不同范式有不同的适用场景——LFPO 在少步推理下离散化误差可能偏大，EBPO 在大 group size 时优势消失，SPG 的 EUBO 在高熵 image token 下可能有数值问题，answer-forcing 仅适用于 dLLM，StableDRL 仅在纯文本验证
- **例外情况**: (1) 七种范式的严格 Pareto 对比尚未在同等条件下进行——各论文使用不同基座（LLaDA vs DiffuCoder vs SDAR）、不同任务、不同 reward；(2) LFPO 的 Theorem 3.1 依赖连续时间极限，少步场景下精度未验证；(3) SPG 的 EUBO β-power 在 image token（NLL>6）下可能导致数值 underflow；(4) 多模态场景（高熵 image token）下各范式的表现可能与纯文本场景差异显著；(5) StableDRL 和 MIS-PO 均仅在纯文本/AR 场景验证，dLLM 多模态实验缺失
- **启示**: dLLM RL 已从单一方法（d1）发展为七范式竞争格局。最有价值的后续工作是在统一基准下对比七种范式的 Pareto 前沿（精度 × 方差 × 计算成本 × 推理步数），以及验证范式间的组合效果（尤其是三层栈 StableDRL + EBPO + answer-forcing、LFPO + EBPO 双层改进）
---

### [P-Diff-14] (候选) dLLM 的 AR 解码收敛是训练数据属性而非架构属性
- **现象**: NAP 通过 ARness 和 SeqDep 两个度量首次系统性诊断了 DLM 收敛为 AR-like 左到右解码的现象。实验发现: (1) 标准预训练数据（FineWeb）和 long-CoT 数据均具有强 SeqDep；(2) LLaDA 和 Dream 即使使用 confidence-based Arbitrary Order 解码，unmask 顺序仍呈严格线性对角线模式；(3) Long-CoT 后训练进一步加剧 ARness；(4) 通过并行推理轨迹数据重构（消除 privileged ordering）+ Parallel-Forced Decoding，在低步数下显著超越标准 long-CoT 训练（GSM8K 256步 +14.4%）
- **支撑论文**: [[2026-NAP]]（ARness/SeqDep 度量 + 并行数据消除 AR 收敛，首次系统性诊断和解决）
- **可能解释**: (1) 训练数据的隐式顺序结构（left-to-right 写作习惯、step-by-step 推理链）使模型学会了"优先从左到右解码"的 AR-like 策略；(2) 解码策略本身（如 confidence-based）无法解决此问题——根源在数据而非解码端；(3) 并行推理轨迹数据中多条独立路径之间没有因果依赖，模型无法学到 privileged ordering
- **例外情况**: (1) 仅单篇论文支撑，需独立验证；(2) XDLM k=0.1 通过改变噪声核（非数据）也部分缓解 AR 收敛——说明数据非唯一因素，噪声核设计也影响解码行为；(3) 仅在后训练（~100K 样本）验证，预训练阶段的并行化效果未知；(4) 性能提升中有多少来自 "m× ensemble 效应"（与 self-consistency 等效）vs "真正的并行解码能力提升"未分离
- **启示**: 如果此 pattern 成立，对 KB 中所有 dLLM 工作有深远影响——当前所有 dLLM 可能都在"名义上并行但��际上 AR"地运行。解决方案不仅在解码端，更需要从数据层面重新设计训练语料的结构。与 XDLM 的噪声核假说和 ReDiff 的训练-推理 gap 假说形成三种竞争解释，更可能的现实是三者共同作用
- **影响范围**: NAP 诊断 KB 中多个 Fast-DLM 方法（MMaDA block-wise, DiffusionVL Block Diffusion, SDAR-VL 块状扩散）本质上在放大而非消除 AR 行为——通过"先稳定前缀再生成后续"获得加速

---

### [P-Diff-15] 一步生成可通过训练时分布演化实现，无需推理时迭代
- **现象**: Drifting Model 提出全新生成范式——将迭代步骤从推理时转移到训练时。通过反对称漂移场 V_{p,q}(x) = V⁺_p(x) - V⁻_q(x)（真实数据吸引 + 生成样本排斥），在训练过程中演化 pushforward 分布，自然实现一步生成。ImageNet 256×256 达 1-NFE FID 1.54（latent）/ 1.61（pixel）SOTA
- **支撑论文**: [[2026-Drifting]]（反对称漂移场 + 训练时分布演化，1-NFE SOTA）、[[2026-Beyond-LM]]（flow matching 连续扩散，可作为 Drifting 的多步精修器）
- **可能解释**: (1) 深度学习训练本身的迭代性（SGD 更新）可替代推理时的 ODE/SDE 迭代——网络参数更新隐式驱动样本分布演化；(2) 反对称性是关键——V_{p,q} = -V_{q,p} 保证 p=q 时 V=0 达到平衡，破坏反对称性导致 FID 41-177（从 8.46）；(3) 不依赖 SDE/ODE 理论、不需要 adversarial 训练、不需要 distillation
- **例外情况**: (1) 仅在图像生成验证，文本/多模态场景未知；(2) 训练成本可能高于 diffusion（需要在线维护 pushforward 分布）；(3) 与 dLLM 的离散 token 空间兼容性未知——Drifting 在连续 latent 空间操作
- **启示**: 为 dLLM 统一模型提供潜在的一步生成路径（当前 dLLM 需 8-50 步）。如果 Drifting 可适配到离散 token 空间，将从根本上消除 dLLM 的推理步数瓶颈。VTP 的语义增强 tokenizer 提供的 latent 空间可作为 Drifting 的特征空间替代方案
