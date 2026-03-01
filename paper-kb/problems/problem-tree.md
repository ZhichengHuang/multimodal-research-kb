# 研究问题树

> 每个节点格式: [ID] 问题描述
> 状态标记: 🔴 open | 🟡 partially-solved | 🟢 solved
> 连接标记: → 表示跨分支依赖

---

## [Root] 如何构建强大的多模态大模型？

---

### [PT] 预训练阶段

#### [PT-1] 🔴 视觉信息如何高效注入 LLM？
- [PT-1a] 🔴 Vision encoder 该冻结还是联合训练？
  - 冻结: 训练效率高，但 encoder 表征不一定对齐 LLM
  - 联合训练: 效果更好但成本高，scaling 行为不明确
  - 已有尝试:
  - 相关论文: [[2026-LaViDa-R1]]
- [PT-1b] 🔴 视觉 token 数量与性能的 tradeoff
  - 高分辨率需要大量 token → 计算成本
  - 压缩策略 (Perceiver, pooling) vs 保留细节
  - 已有尝试:
  - 相关论文:
- [PT-1c] 🔴 Connector 架构选择的本质区别是什么？
  - MLP vs Q-Former vs Perceiver vs cross-attention
  - 已有尝试:
  - 相关论文:

#### [PT-2] 🔴 多模态预训练数据如何配比？
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

#### [PT-3] 🔴 多模态 Scaling Law 是什么形态？
- [PT-3a] 🔴 Vision encoder 和 LLM 的最优 size 比
- [PT-3b] 🔴 数据量 vs 模型 size 的 tradeoff

---

### [Post] 后训练阶段

#### [Post-1] 🔴 多模态幻觉的根因是什么？如何系统解决？
- [Post-1a] 🔴 幻觉来源: 预训练数据偏差 vs 对齐不足 vs 解码策略
  - 已有尝试:
  - 相关论文:
- [Post-1b] 🔴 SFT 阶段能解决多少幻觉？上界在哪？
  - 已有尝试:
  - 相关论文:
- [Post-1c] 🔴 需要 RL 介入吗？→ 连接到 [RL-1]
  - 已有尝试:
  - 相关论文:

#### [Post-2] 🔴 多模态 SFT 数据如何高效构造？
- [Post-2a] 🔴 合成数据 vs 人工标注的质量对比
- [Post-2b] 🔴 多模态 instruction 的多样性与覆盖度

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
  - 已有尝试: MMaDA UniGRPO 结构化随机 mask ratio 替代 Monte Carlo 128-sample; LaViDa-R1 complementary masking
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]]
- [RL-2b] 🔴 Group 构造: 同一图不同回答 vs 不同图？
- [RL-2c] 🔴 高熵 token 分布下的 RL 正则化
  - MMaDA 保留 KL penalty, LaViDa-R1 发现 image token NLL>6 导致 KL estimator 方差极大、训练发散
  - LaViDa-R1 用 SFT loss 替代 KL 作为隐式正则化，有效但理论不完备
  - 根本问题: KL 约束对 dLLM 的高熵离散分布是否根本不合适？
  - 潜在思路: 分 token 类型加权 KL, entropy-aware clipping, 理论推导最优正则化形式
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]]

#### [RL-3] 🔴 RL 能提升多模态推理能力吗？
- [RL-3a] 🟡 视觉推理 (图表/几何) 的 verifiable reward
  - 已有尝试: MMaDA 对 GeoQA/CLEVR 用 correctness+CLIP reward, UniGRPO 提升 GSM8K +8.2; LaViDa-R1 在 MathVista +2.4
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]]
- [RL-3b] 🔴 RL 对 grounding 能力的影响
  - 相关论文: [[2026-LaViDa-R1]]

---

### [Diff] Diffusion 基础方法论

#### [Diff-1] 🔴 扩散范式在多模态统一模型中的选择
- [Diff-1a] 🔴 Flow Matching vs DDPM: 在统一模型中哪个更优？
  - Flow Matching 训练更稳定, 采样更快
  - 但与 AR 框架的兼容性分别如何？
  - 已有尝试:
  - 相关论文:
- [Diff-1b] 🟡 连续扩散 vs 离散扩散 (Masked Diffusion)
  - 离散扩散和 LLM 的 token 框架天然对齐
  - 连续扩散保留更多信息但需要架构改造
  - 已有证据: MMaDA 和 LaViDa 系列均验证离散 masked diffusion 在统一模型中可行且 competitive，连续扩散在统一场景中尚无可比工作
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]]
- [Diff-1c] 🔴 采样效率: 如何在统一模型中减少 diffusion 步数
  - Consistency model / distillation 是否适用
  - Few-step generation 对理解能力的影响
  - 已有尝试: MMaDA 用 50 步（vs 1024）图像生成保持强性能，文本/理解可用 1/2-1/4 步数
  - 相关论文: [[2025-MMaDA]]

#### [Diff-2] 🔴 Diffusion 骨干架构演进
- [Diff-2a] 🔴 DiT 成功的本质原因？Transformer 替代 UNet 的关键
- [Diff-2b] 🔴 MM-DiT (多模态 DiT) 的设计空间
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

#### [Tok-2] 🔴 理解和生成对 Tokenizer 的矛盾需求（核心瓶颈）
- [Tok-2a] 🔴 理解需要语义丰富表示 (CLIP-like) vs 生成需要像素级细节 (VQGAN-like)
  - 这是统一 tokenizer 的根本矛盾
  - 已有尝试:
  - 相关论文:
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
- [Tok-2d] 🔴 MAGVIT-v2 是否是当前 dLLM 统一模型生成质量的主要瓶颈？
  - MMaDA 和 LaViDa-O 都使用 MAGVIT-v2 (codebook 8192, 512×512), 两者 T2I 均与 FLUX/SD3 有明显差距
  - 需区分: 多少差距来自 tokenizer 信息损失 vs 模型容量/训练数据不足
  - 潜在思路: 在 MMaDA/LaViDa 框架中替换为更强 tokenizer (BSQ/LFQ) 做消融实验
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]]

#### [Tok-3] 🔴 Tokenizer 对下游训练的影响
- [Tok-3a] 🔴 Tokenizer 质量对生成质量的上界限制
- [Tok-3b] 🔴 Tokenizer 和主模型是否需要联合训练？

---

### [Uni] 理解生成一体化

#### [Uni-1] 🔴 统一架构路线之争
- [Uni-1a] 🔴 哪条路线最可能 scale？
  - AR+Diffusion (Transfusion, MetaMorph): 灵活, 各取所长, 但训练目标不统一
  - 纯AR离散化 (Chameleon, Emu3): 框架统一优雅, 但视觉离散化有损
  - 解耦编码 (Janus, SEED): 实用, 避免冲突, 但不够"统一"
  - Diffusion原生: MMaDA 首次在 NeurIPS 级别验证 8B 规模可行，证伪风险降低
  - 已有尝试: MMaDA (diffusion-native 三任务 competitive), LaViDa-R1 (grounding specialist level)
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]]
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

#### [Uni-2] 🔴 理解与生成的能力关系
- [Uni-2a] 🟡 共享参数下能力是否冲突？
  - 实验证据: MMaDA Figure 6 显示 Mixed CoT 训练中三项任务指标同步提升，无 seesaw 效应
  - 是否存在 mutual benefit？MMaDA 和 LaViDa-R1 均观察到跨模态正向协同
  - 但仅在 8-10B 规模验证，更大规模行为未知
  - → 连接到 [Diff-2c]
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]]
- [Uni-2b] 🔴 MoE / Routing 解耦是否可行？
  - 部分参数共享 + 部分专用
  - 路由策略如何设计
- [Uni-2c] 🔴 训练配比与课程学习
  - 理解数据 vs 生成数据 vs 交叉数据的配比
  - 先理解后生成 vs 联合训练 vs 课程策略
- [Uni-2d] 🔴 表示对齐: 理解和生成是否需要在同一语义空间？

#### [Uni-3] 🔴 生成质量追赶专用模型
- [Uni-3a] 🔴 当前差距分析: 统一模型 vs FLUX/SD3 级别
- [Uni-3b] 🔴 生成质量的瓶颈到底在哪？
  - Tokenizer? Diffusion 步数? 模型容量分配? 训练数据?
- [Uni-3c] 🔴 用 RL 优化生成质量 → 连接到 [RL-1c]
  - DPOK, DDPO 等方法在统一模型中的应用
- [Uni-3d] 🔴 图像编辑作为理解+生成的联合测试

#### [Uni-4] 🔴 扩展到视频
- [Uni-4a] 🔴 计算瓶颈: 视频 token 数爆炸
- [Uni-4b] 🔴 时序建模: AR 时序 + Diffusion 空间的混合
- [Uni-4c] 🔴 视频理解和生成的统一是否比图像更难？
- [Uni-4d] 🔴 世界模型视角: 视频生成 = 物理规律学习？

#### [Uni-5] 🟡 统一模型的 Post-training 和 RL
- [Uni-5a] 🟡 统一模型如何做 alignment？
  - 理解和生成分别对齐 vs 联合对齐
  - 已有方案: MMaDA 提供首个 diffusion-native 全链路（预训练→Mixed CoT SFT→UniGRPO RL）; LaViDa-R1 提供统一 PG 框架（SFT+GRPO+self-distillation）
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]]
- [Uni-5b] 🟡 统一模型的 reward 如何设计？→ 连接到 [RL-1c]
  - 理解正确性 + 生成质量 + 一致性 的多目标 reward
  - 已有方案: MMaDA Diversified Reward (correctness/format/CLIP/ImageReward), LaViDa-R1 Multi-Reward (correctness/IoU/EditScore)
  - 已发现局限: CLIP reward 不支持 compositional reasoning → 连接到 [RL-1d]
  - "一致性" 维度仍完全未被探索
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]]
- [Uni-5c] 🟡 RL 是否能促进理解-生成的协同？
  - 假设: RL 优化"看图生图"任务可能同时提升两种能力
  - 已有证据: MMaDA UniGRPO 同时提升 GSM8K (推理) 和 ImageReward (生成); LaViDa-R1 多任务 RL 各任务均有提升
  - 但机制不清晰: 是共享表示的迁移还是任务间正则化？
  - 相关论文: [[2025-MMaDA]], [[2026-LaViDa-R1]]
