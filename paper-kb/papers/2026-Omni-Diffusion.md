---
title: "Omni-Diffusion: Unified Multimodal Understanding and Generation with Masked Discrete Diffusion"
authors: [Lijiang Li, Zuwei Long, Yunhang Shen, Heting Gao, Haoyu Cao, Xing Sun, Caifeng Shan, Ran He, Chaoyou Fu]
date: 2026-03
venue: arxiv
url: "https://arxiv.org/abs/2603.06577"
tags: [diffusion, unified-model, architecture, understanding, generation]
category: unified-model/diffusion-native
level: 2
status: read
importance: medium
problem_tree_nodes: [Uni-1a, Diff-1b, Diff-1c]
aliases: [Omni-Diffusion]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结
首个基于 mask-based discrete diffusion 的 any-to-any 多模态模型，通过统一 mask token 预测建模文本、图像、语音三种模态的联合分布，实现跨模态理解与生成。

## 核心 Insight
将多模态统一建模的范式从"LLM 文本骨干 + 外接模态解码器"推向"在离散 diffusion 框架下直接对多模态 token 联合分布建模"——模态对齐不再依赖适配器桥接，而是通过共享的 mask prediction 目标在统一语义空间中自然涌现。

## 与已有工作的关系
- **继承自**: [[2025-MMaDA]]（diffusion-native 统一模型路线）, Dream-7B（dLLM 骨干）
- **对比**: [[AnyGPT]]（AR any-to-any）, [[NExT-GPT]]（AR + 外接 diffusion 解码器）, [[NExT-Omni]]（discrete flow matching）
- **互补**: [[2025-Lumina-DiMOO]]（dLLM 统一模型但仅 text+image）, [[2025-LaViDa]]（dLLM VLM 但仅理解）

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 整体架构
- 骨干: Dream-7B（预训练 mask-based discrete diffusion LLM）
- 图像 tokenizer: MAGVIT-v2（下采样 16×, codebook 8192）
- 语音编码器: SenseVoiceSmall（memory-equipped self-attention + MLP adapter 投射到 diffusion model 隐空间）
- 语音解码器: GLM-4-Voice decoder（FSQ codebook 16384, token rate 12.5 Hz）
- 词表扩展: 在 Dream-7B 原有文本词表基础上新增 16384 speech tokens + 8192 image tokens

### 训练损失
统一的 mask token 预测交叉熵损失（式1），仅对被 mask 的 token 计算损失，无模态特定优化目标。mask ratio r 由均匀采样的 timestep t ∈ [0,1] 决定。

### 三阶段渐进训练
- **Stage 1 (Visual-Language Pre-Alignment)**: text-to-image + image captioning，对齐视觉模态和预训练语言模型语义空间
- **Stage 2 (Speech-Vision-Language Joint Alignment)**: 保留 Stage 1 数据 + 引入 ASR/TTS 数据，实现语音-文本对齐
- **Stage 3 (SDVI Capability Improvement)**: 微调 SDVI 数据集（spoken VQA + speech-to-image），提升跨模态统一对齐

### 推理策略
- Entropy-based decoding（继承自 Dream-Instruct-7B），按 token 置信度（负熵）排序，选 top-k 高置信 token 采样
- 支持 repetition penalty 和 classifier-free guidance

## Building Blocks（可复用组件）

### Block 1: 多模态 Mask Token 联合预测
- **做法**: 将 text/image/speech 三种模态 tokenize 为离散 token 序列，用 special tokens（|BoI|/|EoI|, |BoS|/|EoS|）包裹各模态 token，拼接为统一序列。对整个序列随机 mask，用单一模型预测被 mask 的 token
- **机制 (WHY it works)**: 所有模态共享同一个 mask prediction 目标函数和同一个 transformer 参数空间，迫使模型学习跨模态对齐的统一语义表示。与 AR 模型的"LLM 生成文本 → 外接模型转换其他模态"不同，这里模型直接在多模态 token 空间中建模联合分布
- **适用条件**: 各模态必须有高质量离散 tokenizer；模态间语义关联需要通过数据体现
- **什么时候会 break**: (1) 模态间 token 数量极度不平衡时（如 speech token 远多于 text），模型可能偏向高 token 数模态；(2) tokenizer 质量差时联合建模的信息瓶颈更严重
- **可组合方向**: 扩展到更多模态（video, 3D）；与 MoE 结合实现模态专化路由

### Block 2: 三阶段渐进训练管线
- **做法**: Stage 1 仅 text+image → Stage 2 加入 speech → Stage 3 引入跨模态交互数据（SDVI）。学习率 Stage 1-2: 1e-4, Stage 3: 1e-5
- **机制 (WHY it works)**: 渐进引入模态避免多分布数据同时训练的不稳定性。先锚定视觉-语言对齐（最成熟的数据源），再逐步引入语音，最后通过跨模态交互数据强化统一对齐
- **适用条件**: 各模态有不同量级的训练数据时效果最好；需要各模态 tokenizer 预训练完成
- **什么时候会 break**: (1) 后期 stage 引入新模态时可能遗忘早期 stage 学到的对齐（catastrophic forgetting）——论文通过在后续 stage 保留早期数据部分缓解；(2) 模态引入顺序的选择缺乏理论指导
- **可组合方向**: 与 LaViDa-O 的三阶段课程或 DiMOO 的四阶段管线对比/融合

### Block 3: Attenuated Tail-Pad Masking
- **做法**: 为支持变长生成，在序列末尾追加随机数量的 pad token。训练时 pad token 的 mask ratio 乘以衰减因子 γ=0.6，使 pad token 被 mask 的概率低于普通 token
- **机制 (WHY it works)**: 均匀 masking 下 pad token 过多导致模型过拟合于预测 pad，推理时生成过多无意义 pad。降低 pad token 的 mask ratio 使梯度主要由语义 token 驱动，减少 pad 过拟合
- **适用条件**: 所有 mask-based diffusion 模型在需要变长输出时均适用
- **什么时候会 break**: γ 过小时模型可能无法学会在正确位置终止生成；γ 过大时回到过拟合问题
- **可组合方向**: 与 Sparse-LaViDa 的 register tokens 结合；与 XDLM 的混合噪声核结合

### Block 4: Position Penalty（图像生成）
- **做法**: 推理早期阶段，将序列末尾 N^t 个 token 的 logits 乘以 γp=0.5（缩小），抑制模型同时从序列首尾解码，迫使从序列开头开始有序生成
- **机制 (WHY it works)**: MDM 倾向于同时从序列两端向中间解码（因两端 token 上下文信息最丰富）。由于连续时间步解码的 token 语义相似，两端同时解码会在图像顶部和底部产生重复模式。Position penalty 强制单向有序生成，避免语义重复
- **适用条件**: 图像 token 按 raster scan 排列的 MDM 模型
- **什么时候会 break**: (1) 非 raster scan 排列（如 2D position）时需要重新设计；(2) 对非图像模态（如文本）不适用（文本不存在空间重复问题）
- **可组合方向**: 与 LaViDa-O 的 Stratified Sampling 思路互补（LaViDa-O 空间分散，此处时序有序）；与 PC-Sampler 的位置校准类似但机制不同

### Block 5: Special Token Pre-Infilling（语音任务）
- **做法**: 在初始全 mask 序列中，将 0.25L 位置预填入 [begin-of-speech] token，引导模型在前 25% 生成文本回复、后 75% 生成对应语音
- **机制 (WHY it works)**: 利用 MDM 的独特优势——可修改初始 mask 序列控制输出格式。预填 [BoS] 使模型在生成语音时可显式 attend 到同步生成的文本内容，提升语音与文本的语义一致性
- **适用条件**: 需要文本和语音同步输出的对话任务；需要明确的文本-语音长度比例先验
- **什么时候会 break**: (1) 文本-语音长度比例偏离 1:3 时效果下降；(2) 纯语音生成（无需文本引导）时不适用
- **可组合方向**: 扩展到图文交错生成（预填 [BoI] 控制图像位置）；与 LaViDa-O 的 planning token 概念类似

### Block 6: Adaptive Token Length Assignment（语音任务）
- **做法**: 根据 ASR/TTS 任务中 text 和 speech 的长度相关性，自适应设置初始 mask 序列长度：TTS 为 3.5× text token 长度，ASR 为 0.2× speech token 长度
- **机制 (WHY it works)**: 减少不必要的 mask token 数量，加速采样过程且避免生成过多无用 token
- **适用条件**: 输入和输出模态间存在稳定长度比例关系的任务
- **什么时候会 break**: 长度比例不稳定的任务（如自由对话中语音长度变化大）
- **可组合方向**: 与 Attenuated Tail-Pad Masking 互补——前者控制初始长度，后者处理变长输出

### Block 7: SDVI 数据集构建
- **做法**: 从 LLaVA-OneVision 数据构造 spoken VQA（过滤数学/编程、改写选择题、限制答案长度 ≤100 词、用 CosyVoice2 voice cloning 1000 种声音转语音）；从 Blip3o-Pretrain-JourneyDB 构造 speech-to-image（同样 voice cloning）。各 30K 样本
- **机制 (WHY it works)**: 跨模态交互（speech↔image）的训练数据在自然场景中稀缺。通过 TTS 合成将成熟的 text-image 数据转换为 speech-image 数据，低成本获取跨模态对齐信号
- **适用条件**: 需要高质量 TTS 模型和丰富的文本-图像/VQA 数据源
- **什么时候会 break**: (1) TTS 合成的语音缺乏自然对话特征（韵律、停顿）；(2) 30K 样本可能不足以覆盖复杂跨模态交互场景
- **可组合方向**: 扩展到视频-语音交互数据；增加真实语音数据混合训练

## Anti-patterns / 已知失败模式
- MDM 两端同时解码导致图像重复模式（论文通过 Position Penalty 缓解）——PC-Sampler 从不同角度（位置校准解码偏置）解决同一问题
- 均匀 masking pad token 导致过拟合生成过多 pad（论文通过 Attenuated Tail-Pad Masking 缓解）——该问题在 KB 中其他 dLLM 论文（DiMOO, MMaDA）中以「block-wise + </answer> 早停」等不同方式处理
- 语音生成缺乏文本语义引导时逻辑/连贯性下降（论文通过 Special Token Pre-Infilling 缓解）
- **三模态 token 数量不平衡**: speech token（12.5 Hz → 数百个 token per 秒）远多于 text token，可能导致训练梯度被 speech token 主导——论文未讨论此问题
- **评估基准单薄**: 仅用 CLIP-T/CLIP-I 评估 T2I（KB 中 DiMOO 已达 GenEval 88%、DPG 86%），缺乏 GenEval/FID/DPG-Bench 等更全面的评估
- **缺乏 RL 后训练**: KB 中 MMaDA (UniGRPO)、DiMOO (Self-GRPO)、LaViDa-R1 (complementary masking RL) 均证明 RL 后训练可显著提升性能，Omni-Diffusion 仅有三阶段 SFT
- **训练-推理分布不匹配**: 论文未考虑 ReDiff 揭示的并行生成错误传播问题，也未采用 XDLM 的混合噪声核来缓解

## 实验关键发现
- **Speech**: ASR WER 7.05（超越 AnyGPT 8.50），TTS WER 3.07（超越 GLM-4-Voice 5.64，接近 CosyVoice 2.82）
- **Visual Understanding**: POPE 76.6, MME-P 1216.7, Seed-2-Plus 34.5——超越 AnyGPT 和 NExT-GPT
- **Text-to-Image**: CLIP-T 0.235, CLIP-I 0.667——文本-图像对齐超越其他 any-to-any 模型
- **Sampling Efficiency**: text-to-image 从 256 步减少到 10 步，CLIP-T/CLIP-I 仅从 0.235/0.667 降至 0.226/0.650；TTS 0.125L 步仍可用（WER 4.83）
- **Speech-to-Image**: CLIP-T 0.225 / CLIP-I 0.645，与 text-to-image 质量接近，验证跨模态对齐有效
- **Inpainting**: 无需额外训练即可支持（MDM mask 机制天然兼容）

## Relations (结构化)
- `extends` → [[2025-MMaDA]]: 从 text+image 扩展到 text+image+speech 的 diffusion-native 统一模型；同为模态无关全共享架构 + MAGVIT-v2 tokenizer
- `extends` → [[Dream-7B]]: 在 Dream-7B 骨干上扩展多模态能力（LaViDa-D 也使用 Dream-7B）
- `alternative_to` → [[AnyGPT]]: 同为 any-to-any 模型，Omni-Diffusion 在 ASR/TTS/VQA/T2I 均超越 AnyGPT
- `alternative_to` → [[NExT-GPT]]: 同为 any-to-any 模型，但用统一 diffusion 替代 LLM+外接 diffusion 解码器
- `alternative_to` → [[NExT-Omni]]: 同为非 AR any-to-any，但用 mask diffusion 替代 discrete flow matching；NExT-Omni 受限于 text-only backbone 需外接模型
- `combines_with` → [[2025-Lumina-DiMOO]]: DiMOO 的 ML-Cache 推理加速、Self-GRPO 联合 RL、`<end-of-line>` 任意分辨率均可应用于 Omni-Diffusion 提升性能
- `combines_with` → [[2025-LaViDa]]: LaViDa 的 Complementary Masking 可提升训练效率；Prefix-DLM 可加速推理；FIM 能力可扩展语音 infilling
- `combines_with` → [[2025-XDLM]]: XDLM 的 k=0.1 混合噪声核可替换 Omni-Diffusion 的纯 mask 噪声（k=0），在 10 步采样下可能显著提升质量
- `combines_with` → [[2025-Sparse-LaViDa]]: Sparse-LaViDa 的 mask token 截断 + register tokens 可加速 Omni-Diffusion 推理
- `combines_with` → [[2025-ReDiff]]: ReDiff 的精炼训练可解决 Omni-Diffusion 未考虑的并行生成错误传播问题
- `combines_with` → [[2025-VTP]]: VTP 的语义增强 tokenizer 可替换 MAGVIT-v2 提升生成质量上界
- `motivated_by` → [[VITA]]: 同一团队前作（AR 多模态模型），Omni-Diffusion 将 VITA 的 duplex 多模态交互思路从 AR 迁移到 dLLM

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- 首次证明 mask-based discrete diffusion 可统一三种模态（text+image+speech）的理解与生成，打破 any-to-any 模型必须依赖 AR 骨干的范式
- 提出并解决了 MDM 的三个推理问题：两端同时解码的图像重复（Position Penalty）、变长输出的 pad 过拟合（Attenuated Tail-Pad Masking）、语音生成的文本语义引导（Pre-Infilling）

### 未解决的问题
- 问题: 视觉生成质量与专用扩散模型和 KB 中其他 dLLM 统一模型差距大——缺乏 GenEval 评估，CLIP-I 0.667 不足以判断 compositional 能力
  - 为什么难: MAGVIT-v2 tokenizer 的信息瓶颈 + 模型容量分散在三种模态 + 缺乏 RL 后训练
  - 潜在思路: 升级到语义增强 tokenizer（VTP 方案）；引入 UniGRPO/Self-GRPO 等 RL 优化；增加训练数据规模（对比 DiMOO 的 ~110M）
- 问题: 三模态 token 数量不平衡——speech token (12.5 Hz) 在长语音中可达数千 token，远超 text/image
  - 为什么难: 统一 mask ratio 对三种模态等权，但信息密度差异巨大
  - 潜在思路: per-modality mask ratio；模态加权 loss；MoE 路由实现模态容量分配
- 问题: 语音-视觉交互数据仅 60K（TTS 合成），是否足以支撑复杂跨模态场景
  - 为什么难: 真实三模态交互数据稀缺；TTS 合成语音缺乏自然对话特征
  - 潜在思路: 扩大合成数据规模到百万级；引入真实语音-视觉数据混合训练
- 问题: 缺乏 RL 后训练——KB 中 MMaDA/DiMOO/LaViDa-R1 均证明 RL 可显著提升 dLLM 统一模型性能
  - 为什么难: 三模态 RL 的 reward 设计更复杂（需要覆盖 text/image/speech 三种输出的质量评估）
  - 潜在思路: 扩展 UniGRPO/Self-GRPO 到语音任务（WER 作为 verifiable reward）

### 对问题树的推进
- 推进了 [[problem-tree#统一架构路线之争]]: 新增 any-to-any dLLM（三模态）数据点，进一步验证 Diffusion 原生路线可 scale 到更多模态（text+image+speech），扩展了 MMaDA/DiMOO 仅 text+image 的范围
- 推进了 [[problem-tree#连续扩散 vs 离散扩散 (Masked Diffusion)]]: 证明 masked diffusion 不仅可做 text+image，还可扩展到 speech，模态数量不是 masked diffusion 的限制
- 推进了 [[problem-tree#采样效率]]: Position Penalty 和 Adaptive Token Length Assignment 是 MDM 推理优化的新技术，与 KB 中已有的七维加速技术正交
- 新增问题: [Uni-1a-1] 🔴 三模态（及更多模态）联合训练的模态竞争/遗忘问题——模态引入顺序、数据配比、token 数量平衡如何系统化设计
- 新增问题: [Diff-1c-3] 🔴 MDM 解码顺序偏置（两端向中间）对不同模态的影响——图像产生重复模式（已验证），文本和语音是否也存在类似偏置
- 新增问题: [Diff-1b-2] 🔴 Pre-Infilling 作为 MDM 通用输出格式控制机制的潜力——可否推广到图文交错、结构化输出等更广泛场景

## 个人深度评注

### [Critic] 增量贡献评估
| Block | 新颖性 | 理由 |
|-------|--------|------|
| Block 1: 多模态联合预测 | 低 | MMaDA/DiMOO 已在 text+image 上充分验证，扩展到 speech 是工程增量 |
| Block 2: 三阶段渐进训练 | 低中 | 与 MMaDA 三阶段和 DiMOO 四阶段同质，新增的 SDVI stage 有一定价值 |
| Block 3: Attenuated Tail-Pad Masking | 中 | 首次识别并解决 MDM 变长生成中 pad 过拟合问题，对所有 MDM 可复用 |
| Block 4: Position Penalty | 中 | 首次识别 MDM 两端同时解码导致的图像重复问题，与 PC-Sampler 独立发现类似现象 |
| Block 5: Special Token Pre-Infilling | 中高 | 利用 MDM 独有的初始序列可编辑特性控制多模态输出格式，与 LaViDa FIM 和 DiMOO Pre-Infilling 思路类似但首次应用于语音任务，泛化潜力大 |
| Block 6: Adaptive Token Length | 低 | 简单工程优化 |
| Block 7: SDVI 数据集 | 低 | 标准 TTS 合成数据工程 |

### [Critic] 关键隐含假设
1. **统一 mask ratio 对三种模态等效**: 论文用同一个 t 对 text/image/speech 统一 mask，但三者信息密度截然不同——mask 掉文本的否定词改变语义，mask 掉图像局部 patch 仅局部缺失，mask 掉语音 token 可能导致韵律断裂。与 MMaDA 的假设一致但在三模态下更严重
2. **Pre-Infilling 的 25%/75% 切分是最优的**: 文本:语音 = 1:3 的比例是经验值，不同回复长度和内容复杂度下最优值可能不同
3. **合成语音足够代表真实语音**: SDVI 数据通过 CosyVoice2 TTS 合成，假设合成语音的分布可迁移到真实语音场景。但合成语音缺乏自然对话中的犹豫、重复、语气变化等现象
4. **Position Penalty 仅图像需要**: 论文仅对图像模态应用位置惩罚，但语音 token 也存在时序结构（开头和结尾的 silence 可能产生类似的两端偏置）
5. **无需 RL 即可达到 competitive 性能**: 与 KB 中的主流经验相矛盾——MMaDA (UniGRPO +8.2 GSM8K)、DiMOO (Self-GRPO GenEval +X%)、LaViDa-R1 均证明 RL 后训练是性能提升的关键

### [Connector] 技术谱系定位
```
Dream-7B (masked diffusion LM)
  │
  ├── [路线 A: 统一模型, 模态无关全共享]
  │   ├── MMaDA (2025-05, text+image, LLaDA-8B) — 方法论开创
  │   ├── DiMOO (2025-10, text+image, LLaDA-8B) — 数据 scaling 实证
  │   └── Omni-Diffusion (2026-03, text+image+speech, Dream-7B) ← 本文
  │       └── 首次将 dLLM 统一模型扩展到三模态
  │
  ├── [路线 B: 非对称专用架构]
  │   └── LaViDa → LaViDa-O → LaViDa-R1
  │
  └── [路线 D: 纯理解]
      └── LLaDA-V (同骨干 Dream 变体: LaViDa-D)
```

Omni-Diffusion 在路线 A 中的定位: 横向扩展（更多模态），而非纵向深化（更强性能）。它在模态覆盖范围上超越 MMaDA/DiMOO（三模态 vs 双模态），但在每个模态的深度和 RL 后训练上不如后者。

### [Connector] 与 KB 核心论文的深度对比
| 维度 | Omni-Diffusion | MMaDA | DiMOO |
|---|---|---|---|
| 模态 | text+image+speech | text+image | text+image |
| 骨干 | Dream-7B | LLaDA-8B | LLaDA-8B |
| 图像 Tokenizer | MAGVIT-v2 | MAGVIT-v2 | aMUSEd-VQ |
| 语音编解码 | SenseVoiceSmall+GLM-4-Voice | N/A | N/A |
| 训练数据 | ~17M+（表 4 汇总） | ~数 M | ~110M |
| RL 后训练 | 无 | UniGRPO | Self-GRPO |
| T2I 评估 | CLIP-T/I only | GenEval 63%→84%(+RL) | GenEval 88% |
| 理解评估 | POPE 76.6, MME-P 1216.7 | GSM8K 73.4 | MMMU 58.6% |
| 核心贡献 | 三模态扩展 + 推理技巧 | RL 方法论(UniGRPO) | 数据工程 + 系统设计 |

### [Ideator] 潜在研究方向
1. **Pre-Infilling 泛化为通用 MDM 输出控制框架**: 将 Special Token Pre-Infilling 从"文本+语音同步生成"推广到任意多模态输出格式控制——预填 [BoI]/[BoS]/[BoT] 在任意位置控制图像/语音/文本的输出位置和比例。这与 LaViDa 的 FIM 和 DiMOO 的 `<end-of-line>` 思路互补，构成 MDM 独有的"输出编排"能力（AR 模型必须按序列顺序生成，无法预控制格式）
2. **Omni-Diffusion + RL 后训练**: 将 KB 中已有的 dLLM RL 方法（UniGRPO/Self-GRPO/LFPO）直接应用于 Omni-Diffusion。语音任务天然有 verifiable reward（WER 可精确计算），是 RL 最理想的场景之一。三模态联合 RL 可能在 P-Uni-01 跨模态协同上提供新的数据点
3. **三模态 MoE 容量分配**: 结合 Beyond-LM 的 per-modality shared experts 和 Qwen3-VL 的 MoE 设计，为 Omni-Diffusion 引入三模态 MoE 路由——文本/图像/语音各自有 shared experts + 共享 routing experts。解决三模态 token 数量不平衡下的容量分配问题
4. **MDM 解码顺序偏置的系统性研究**: Position Penalty 暴露了 MDM 的"两端向中间"解码偏置。这一偏置是否在文本（首尾 token 先解码导致中间缺乏连贯性）和语音（开头结尾先解码导致中间韵律断裂）中也存在？系统性研究可为所有 MDM 的推理策略设计提供指导
