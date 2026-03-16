---
title: "Generative Modeling via Drifting"
authors: [Mingyang Deng, He Li, Tianhong Li, Yilun Du, Kaiming He]
date: 2026-02
venue: arxiv
url: "https://arxiv.org/abs/2602.04770"
tags: [generation, architecture, diffusion]
category: diffusion-foundation/one-step-generation
level: 2
status: read
importance: high
problem_tree_nodes: [Diff-1a, Diff-1c]
aliases: [Drifting Model, Drifting]
---

<!-- ==================== Level 1: 快速索引 (必填, 5min) ==================== -->

## 一句话总结

Drifting Model 提出一种全新生成建模范式：不在推理时迭代演化分布（如 diffusion/flow），而是在训练时通过"漂移场"演化 pushforward 分布，自然实现一步生成，在 ImageNet 256×256 上达到 1-NFE FID 1.54（latent）/ 1.61（pixel）的 SOTA。

## 核心 Insight

生成建模的迭代步骤可以从推理时转移到训练时。diffusion/flow 模型在推理时通过 x_{i+1} = x_i + Δx_i 迭代演化样本分布；Drifting Model 利用深度学习训练本身的迭代性（SGD 更新），将分布演化嵌入训练过程。核心机制是定义一个**反对称漂移场** V_{p,q}(x) = V⁺_p(x) - V⁻_q(x)（真实数据吸引 + 生成样本排斥），当 p=q 时 V=0 达到平衡——训练 loss 就是最小化 ||V||²。这不依赖 SDE/ODE 理论，不需要 adversarial 训练，也不需要 distillation，是一种概念上全新的生成范式。

## 与已有工作的关系
- **继承自**: Mean-shift 方法（漂移场是 kernel-weighted mean-shift 的双源变体）、MMD/Moment Matching（共享 kernel 度量分布差异的思想，但优化方式不同）、SimSiam/BYOL（stop-gradient 训练范式）
- **对比**: [[iMeanFlow]]（flow-based 1-NFE distillation, FID 1.72）、[[iCT]]（consistency training 1-NFE, FID 34.24）、[[MeanFlow]]（flow-based 1-NFE, FID 3.43）、[[AdvFlow]]（adversarial flow 1-NFE, FID 2.38）——这些都依赖 SDE/ODE 框架或 adversarial training，Drifting 完全独立于此。GAN（同为单步生成器但需对抗训练）
- **互补**: [[2025-VTP]]（VTP 的语义增强 tokenizer 提供的 latent 空间可作为 Drifting 的特征空间替代方案）、[[2026-Beyond-LM]]（flow matching 连续扩散可作为 Drifting one-step 输出的可选多步精修器）

<!-- ==================== Level 2: 方法拆解 (重要论文, 15min) ==================== -->

## 方法细节

### 核心框架
1. **Pushforward 形式化**: 生成建模 = 学习 f 使得 f#(p_prior) ≈ p_data。训练产生 {f_i} 序列，对应 {q_i} 分布序列
2. **训练时漂移**: 网络更新使样本隐式漂移 x_{i+1} = x_i + Δx_i，其中 Δx_i = f_{i+1}(ε) - f_i(ε)
3. **漂移场**: V_{p,q}(x) 定义样本的漂移方向/幅度，核心性质：**反对称**（V_{p,q} = -V_{q,p}）保证 p=q 时 V=0
4. **训练目标**: L = E[||f_θ(ε) - stopgrad(f_θ(ε) + V(f_θ(ε)))||²]，即最小化漂移场的范数

### 漂移场设计（Kernel-based）
- 吸引力 V⁺_p(x) = (1/Z_p) E_p[k(x,y⁺)(y⁺ - x)]（向真实数据的加权均值漂移）
- 排斥力 V⁻_q(x) = (1/Z_q) E_q[k(x,y⁻)(y⁻ - x)]（向生成样本的加权均值漂移）
- 合并：V_{p,q}(x) = (1/Z_pZ_q) E_{p,q}[k(x,y⁺)k(x,y⁻)(y⁺ - y⁻)]
- 核函数：k(x,y) = exp(-||x-y||/τ)，用 softmax 实现归一化，类似 InfoNCE

### 特征空间漂移
- 在预训练 SSL 特征空间（MoCo/SimCLR/MAE）中计算漂移 loss
- 多尺度特征：ResNet 多 stage 的多分辨率特征图上分别计算漂移 loss
- 自定义 "latent-MAE"：在 SD-VAE latent space 上用 MAE 预训练 ResNet 特征编码器，避免推理时经过 VAE decoder

### CFG（训练时 guidance）
- 负样本 = (1-γ) 生成样本 + γ 不同类别真实样本
- 等价于生成 q_θ(·|c) ≈ α·p_data(·|c) - (α-1)·p_data(·|∅)
- CFG 是训练时行为，推理仍为一步

### 实现
- 生成器：DiT-like Transformer（B/2 或 L/2），输入 32×32×4 高斯噪声 → 输出 latent
- Tokenizer：标准 SD-VAE
- 批次结构：每类 N_pos 正样本 + N_neg 负样本，N_c 个类
- 训练 1280 epochs，Best FID 1.54 (L/2)

## Building Blocks（可复用组件）

### Block 1: 反对称漂移场（Anti-symmetric Drifting Field）
- **做法**: 定义 V_{p,q} = V⁺_p - V⁻_q 作为训练时的分布演化驱动力。V⁺ 为 data-driven 吸引，V⁻ 为 generated-sample 排斥，两者结构相同但符号相反，保证 p=q 时 V=0
- **机制 (WHY it works)**: 反对称性保证平衡态唯一对应 p=q。直觉上：如果 q 在某区域"多于" p，该区域生成样本被更多排斥；如果"少于" p，被更多吸引——形成自纠正动态。与 MMD 相关但不同：MMD 优化全局统计量，Drifting 通过局部漂移移动每个样本。与 GAN 不同：无对抗训练，不需要判别器
- **适用条件**: 需要高质量特征空间使 kernel 有效（raw pixel space 无法工作）；需要足够多的正/负样本估计 V（N_pos=64, N_neg=64 per class）；kernel 温度 τ 需要适当（过大则 kernel flat，过小则只关注最近邻）
- **什么时候会 break**: (1) 无特征编码器时 kernel 无法有效度量语义相似性，ImageNet 实验直接失败；(2) 反对称性被破坏（如不等权 attraction/repulsion）会导致平衡态偏移，FID 从 8.46 暴涨至 41-177；(3) 正负样本数量极少时 V 估计方差过大，FID 显著恶化（N_pos=1 → FID 20.43 vs N_pos=64 → 8.46）；(4) 理论上 V=0 ⇏ q=p 的逆命题不成立（仅有 heuristic 充分条件）
- **可组合方向**: 漂移场设计可扩展——其他 kernel、其他吸引-排斥形式、学习的漂移场；与 flow matching 的速度场概念有类比关系（漂移场是训练时速度场）；可应用于非图像领域（已验证 robotics control）

### Block 2: Feature-space Drifting Loss（特征空间漂移 Loss）
- **做法**: 在预训练 SSL 编码器（MoCo/SimCLR/MAE）的特征空间中计算漂移场和漂移 loss，使用多尺度 ResNet 特征（4 个 stage）的多位置特征向量分别计算
- **机制 (WHY it works)**: 高维像素空间中 kernel k(x,y) 对语义相似度不敏感——所有样本都"远离"彼此导致 kernel 近乎均匀（flat），漂移信号消失。SSL 特征空间中语义相似的样本距离更近，kernel 可有效区分相似/不相似样本，提供有意义的漂移梯度。多尺度特征提供从局部纹理到全局语义的多层次梯度信号
- **适用条件**: 需要高质量预训练 SSL 编码器（MAE > MoCo > SimCLR）；编码器需支持空间特征图（ResNet-style）；编码器宽度和预训练时长与生成质量强相关（width 640 + 1280 ep → FID 4.28 vs width 256 + 192 ep → 8.46）；分类微调进一步提升（4.28 → 3.36）
- **什么时候会 break**: (1) SSL 编码器质量不足时退化为低质量代理目标；(2) 纯 pixel space 完全失败——这是方法的核心依赖；(3) 编码器与生成域的 domain gap（如用 ImageNet 预训练的编码器处理 text-to-speech）
- **可组合方向**: 可用更强的 SSL 方法（DINOv2、MAE-V2）训练编码器；可替换为多模态编码器（CLIP）适配条件生成；latent-MAE 方案避免 VAE decoder 反向传播开销，高效可行

### Block 3: Training-time CFG（训练时 Classifier-Free Guidance）
- **做法**: 在负样本中混入 γ 比例的**不同类别真实数据**作为额外负样本（除生成样本外），等效于训练时实现 CFG。模型额外以 CFG scale α 为条件输入
- **机制 (WHY it works)**: 混入其他类别数据作为负样本，使排斥力不仅来自生成分布 q，还来自"非目标类别"的真实分布。这迫使模型学到的 q_θ(·|c) 不仅接近 p(·|c)，还远离 p(·|∅)——实现类似 CFG 的 sharpening 效果。训练时 α 随机采样，推理时可自由调整
- **适用条件**: 类别标签可用（class-conditional generation）；γ 和 α 的范围需要调优
- **什么时候会 break**: (1) 无类别标签的无条件生成场景不适用；(2) CFG scale 过大可能导致 mode dropping（与标准 diffusion CFG 类似的 FID-IS tradeoff）
- **可组合方向**: 可扩展到 text-conditional 生成（将类别换为 text embedding）；训练时 CFG 消除了推理时 double-pass 的计算开销（纯一步生成保持）

### Block 4: Latent-MAE 特征编码器
- **做法**: 在 SD-VAE latent space（32×32×4）上用 MAE 目标预训练 ResNet-style 编码器，直接在 latent 上操作，避免将 latent 解码回 pixel 再提取特征
- **机制 (WHY it works)**: 标准 SSL 编码器（MoCo/SimCLR）在 pixel space 操作，训练时需要将 generator 的 latent 输出通过 VAE decoder 解码到 pixel，再通过 SSL 编码器提取特征——VAE decoder 成为瓶颈（内存和计算）。Latent-MAE 直接在 latent space 上工作，消除 VAE decoder 的开销，同时通过足够的编码器宽度（640）和训练时长（1280 ep）达到甚至超越 pixel-space SSL 编码器的性能（FID 3.36 vs MoCo 8.41）
- **适用条件**: 需要足够大的 latent-MAE（width ≥ 384，training ≥ 192 ep）；latent space 需要有足够的语义结构（SD-VAE latent 具备此条件）
- **什么时候会 break**: (1) latent space 语义结构差时 MAE 重建目标不提供好的表征；(2) 编码器过小时性能不足
- **可组合方向**: 可替换为其他 latent-space SSL 方法（latent-DINOv2、latent-contrastive）；可用于其他需要 latent-space 特征的应用（latent-space GAN、latent-space consistency model）

## Anti-patterns / 已知失败模式
- **反对称性必须严格保持**: 1.5× attraction 导致 FID 从 8.46 暴涨至 41.05；2.0× repulsion 导致 FID 112.84；attraction-only 达 177.14。这说明漂移场不能有任何偏向
- **无特征编码器完全失败**: 在 ImageNet 上无法在 raw latent/pixel space 中工作，kernel 无法有效度量高维数据的语义相似性
- **正/负样本数不足时性能显著下降**: N_pos=1 时 FID 20.43 vs N_pos=64 时 8.46，估计漂移场需要足够样本
- **理论保证不完整**: V=0 ⇒ q=p 的逆命题仅在 heuristic 条件下成立，非严格理论保证

## 实验关键发现
- **SOTA 1-NFE 生成**: ImageNet 256×256 latent FID 1.54（L/2），超越所有之前的 1-NFE 方法（iMeanFlow 1.72, AdvFlow 2.38, MeanFlow 3.43），并接近多步方法（DiT-XL/2 2.27, SiT-XL/2 2.06）
- **Pixel-space 同样 SOTA**: 1-NFE FID 1.61（L/16），超越 StyleGAN-XL（2.30）且计算量仅 87G FLOPs（StyleGAN-XL 1574G）
- **反对称性是关键**: 破坏反对称性的 destructive ablation 导致 FID 从 8.46 到 41-177，验证了理论设计的必要性
- **特征编码器质量决定生成质量**: latent-MAE width 256→640 + 分类微调 → FID 从 8.46 降至 3.36，编码器是最大的 leverage point
- **Base 模型已接近 XL 性能**: B/2（133M）FID 1.75 接近 iMeanFlow-XL/2（610M）FID 1.72——参数效率极高
- **Robotics 控制上匹配/超越 Diffusion Policy**: 1-NFE Drifting Policy 在 6 个 robosuite 任务上整体匹配 100-NFE Diffusion Policy，证明泛化性
- **Best FID 在 CFG=1.0（无 guidance）时达到**: 不同于 diffusion 模型通常需要 CFG>1.0，暗示训练时 CFG 已充分吸收 guidance 效果

## Relations (结构化)
- `alternative_to` → [[iMeanFlow]]: 两种完全不同的 1-NFE 生成路线——iMeanFlow 从 flow matching 蒸馏单步映射（依赖 ODE 轨迹近似），Drifting 通过训练时分布演化天然一步生成（不依赖 SDE/ODE）。Drifting FID 1.54 vs iMeanFlow 1.72
- `alternative_to` → [[AdvFlow]]: AdvFlow 引入 adversarial loss 改善 flow-based 一步生成（FID 2.38），Drifting 完全不用对抗训练（用 kernel-based 漂移场替代判别器），更稳定且性能更好
- `alternative_to` → [[StyleGAN-XL]]: 同为一步 pixel-space 生成。Drifting pixel FID 1.61 (87G FLOPs) vs StyleGAN-XL 2.30 (1574G FLOPs)——无对抗训练、更高质量、计算量更少
- `motivated_by` → [[SimSiam/BYOL]]: stop-gradient 训练范式直接继承自 self-supervised learning 的 teacher-student 结构
- `motivated_by` → [[MoCo]]/[[SimCLR]]: 正负样本的 contrastive 结构（kernel softmax ≈ InfoNCE）和 SSL 编码器的使用直接受启发于对比学习
- `combines_with` → [[2025-VTP]]: VTP 的多目标语义 tokenizer (CLIP+DINOv2+reconstruction) 提供的 latent space 可替代 Drifting 的 SSL 编码器——消除外部编码器依赖的潜在方案
- `enables` → [[2026-Beyond-LM]]: Drifting 的 one-step 输出可作为 flow matching 精修的起点（比纯高斯噪声"更近"目标分布），实现质量-效率灵活 tradeoff

<!-- ==================== Level 3: 完整分析 (核心论文, 30min) ==================== -->

## 问题定位
### 解决的问题
- **一步生成的性能天花板**: 之前最强 1-NFE 方法（iMeanFlow 1.72）仍依赖 ODE 框架的单步近似，性能受限于轨迹近似误差。Drifting 提供了概念上全新的路径——通过训练时分布演化天然实现一步生成，FID 1.54 超越所有蒸馏方法
- **单步生成对 adversarial training 的依赖**: GAN 和 AdvFlow 都需要对抗训练来实现高质量单步生成。Drifting 用 kernel-based 漂移场替代判别器，消除了对抗训练的不稳定性

### 未解决的问题
- **Text-conditional 扩展性**:
  - 为什么难: 当前 batch 结构（N_c 类 × N_pos 正样本 × N_neg 负样本）严重依赖清晰的类别边界。Text prompt 空间连续无限，每个 prompt 通常只有 1 张对应图像，无法构造 N_pos=64 的正样本集。训练时 CFG 的负样本定义（"不同条件的真实数据"）在文本空间中模糊
  - 潜在思路: 用 CLIP text embedding 的语义邻域定义"同类"；retrieval-based 正/负样本构造；弱化 per-prompt 要求，在语义簇层面操作
- **SSL 编码器依赖的根本性限制**:
  - 为什么难: 编码器质量决定生成质量上界（width 256→640 + cls ft: FID 8.46→3.36，编码器改进占总 FID 改善的约一半）。编码器的训练目标（对比/重建）与生成任务不完全对齐——MoCo 的 augmentation invariance 可能抹去对生成重要的颜色/纹理细节。无编码器完全失败（kernel 在高维原始空间中 flat），这是方法论层面的硬约束
  - 潜在思路: VTP 语义增强 tokenizer 替代 SSL 编码器（消除外部依赖）；端到端联合训练编码器和生成器（但可能引入训练不稳定）；探索不同 SSL 目标（DINOv2、MAE-V2）对漂移场的影响
- **理论收敛保证缺失**:
  - 为什么难: V=0 ⇒ p=q 的逆命题不成立。当两个分布有相同的 kernel-weighted 一阶矩但高阶矩不同时，V 可接近 0 但 p≠q。论文仅提供 heuristic identifiability 条件（bilinear constraints under non-degeneracy），非严格理论
  - 潜在思路: 使用 characteristic kernel（如 Gaussian kernel 对 MMD 有完备性保证）可能提供更强的收敛保证；或从信息论角度推导漂移场优化与 f-divergence 最小化的关系
- **训练成本过高（1280 epochs ImageNet）**:
  - 为什么难: 分布演化嵌入 SGD 训练过程，演化速度受 learning rate 约束。不像 ODE 求解器可用自适应步长。粗估总样本处理量约 21 亿次，是标准 DiT 训练的 ~16 倍
  - 潜在思路: 自适应漂移步长（根据 ||V|| 动态调整 learning rate）；curriculum training（先粗粒度特征空间，后细粒度特征空间）；与蒸馏方法的 total cost（teacher training + distillation）做系统性对比

### 对问题树的推进
- 推进了:
  - [[problem-tree#[Diff-1a] Flow Matching vs DDPM: 在统一模型中哪个更优？]]: 引入"训练时分布演化 (Drifting)"作为超越 Flow Matching vs DDPM/Masked Diffusion 二元框架的第三条路线。Drifting 完全不依赖 SDE/ODE，不需要前向噪声过程。但目前仅验证 class-conditional ImageNet，text-conditional 和多模态统一场景尚未验证
  - [[problem-tree#[Diff-1c] 采样效率: 如何在统一模型中减少 diffusion 步数]]: Drifting 从根本上消解采样效率问题——推理天然一步，不需要任何加速技术。1-NFE FID 1.54 超越 250-NFE DiT-XL/2 (2.27)。但训练成本（1280 epochs）是推理效率的代价——将推理计算转移为训练计算
- 新增问题:
  - **[Diff-1l] 生成范式多元化——Drifting Model 的 text-conditional 扩展和多模态适用性**: Drifting 证明训练时分布演化是可行的一步生成路线，但核心依赖 class-conditional 设置和 SSL 编码器。能否扩展到 text-conditional / text-to-video / 多模态统一模型？SSL 编码器依赖是否限制了跨领域泛化？
  - **[Diff-1m] SSL 编码器作为生成质量瓶颈的理论分析**: Drifting 揭示了一种新型生成质量瓶颈——不是 tokenizer 信息损失、不是扩散步数不足，而是**特征空间的语义质量**。编码器无法区分的视觉模式，生成器也无法学会生成。这与 VTP 证明的"语义增强是生成 scaling 的必要条件"在方向上一致

## 个人深度评注
- **概念创新 vs 实用性 gap**: Drifting 的概念创新是真实的——将迭代从推理时转移到训练时，这是一个干净的视角转换。但方法的核心依赖（高质量 SSL 编码器 + class-conditional + 1280 epochs 训练）严重限制了实用性。与 KB 中关注的多模态统一模型方向相比，Drifting 目前更像是一个理论贡献而非可直接应用的工具
- **与 MMD 的关系被低估**: 论文将 Drifting 定位为"全新范式"，但底层数学工具（kernel-based 正负样本比较）与 MMD 生成模型高度相关。核心新颖性在于：(1) 逐点漂移场 vs 全局统计量，(2) stop-gradient 目标 vs 直接优化 MMD。这些是重要的设计选择但不构成"范式革命"
- **SSL 编码器依赖的深层含义**: "用预训练 SSL 编码器替代 GAN 判别器"——这是对 Drifting 最精准的一句话描述。好处是训练稳定（无对抗博弈），代价是失去自适应性（判别器可以发现生成器的新弱点，SSL 编码器不会）。最终生成质量被 SSL 编码器的感知能力限制——这是方法论层面的天花板
- **对 KB 的启示（训练时 CFG 最有迁移价值）**: 在 Drifting 的四个 Building Block 中，对 KB 中 dLLM 统一模型最有迁移价值的是 Block 3（训练时 CFG）。在 masked diffusion 训练中引入"不同条件的真实数据"作为负样本——这不需要 Drifting 的整个框架，可以作为独立的训练策略插入现有 dLLM 管线。与 DiMOO Self-GRPO（用模型自身理解评估生成）互补：Self-GRPO 是在线自反馈，训练时 CFG 是离线条件对比
- **潜在组合方向**: 最值得探索的是 Drifting + VTP 的组合——用 VTP 的语义增强 latent space 替代 SSL 编码器，既消除外部依赖，又利用 VTP 已验证的 gFID 1.11 生成适用性。这需要验证 VTP 的 64 维 latent 是否提供足够的多尺度信息用于漂移场计算
