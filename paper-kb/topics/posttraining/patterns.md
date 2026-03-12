# 后训练 — 跨论文经验规律

> 从多篇论文中归纳出的经验规律。每条 pattern 需要至少 2 篇论文支撑。

## 格式

```
### [P-Post-xx] 规律名称
- **现象**:
- **支撑论文**:
- **可能解释**:
- **例外情况**:
- **启示**:
```

---

### [P-Post-01] dLLM 统一模型中 SFT 的提升幅度通常远大于 RL
- **现象**: MMaDA 的 Mixed CoT SFT 带来 GSM8K +47.8（17.4→65.2），UniGRPO RL 仅再提 +8.2（65.2→73.4）。LaViDa-R1 的 SFT Stage 1 步数（100K）远大于 RL Stage 2（5K）。两者中 SFT 的绝对贡献均占主导
- **支撑论文**: [[2025-MMaDA]]（SFT 贡献 85%+ 的推理提升）、[[2026-LaViDa-R1]]（SFT stage 步数占比 95%+）
- **可能解释**: dLLM 预训练偏重图像重建，语言/推理能力被压制，SFT 阶段主要在恢复和激活已有能力而非学习新能力；RL 的作用更多是"精炼"（边际优化）而非"创建"（能力涌现）
- **例外情况**: LaViDa-R1 在 image editing 上 SFT 几乎饱和（+0.01），RL 反而贡献更大（+0.10）——说明当 SFT 数据覆盖不到的模式时，RL 的探索价值更高
- **启示**: (1) 投入更多资源在 SFT 数据质量上可能比 RL 算法创新更有效；(2) RL 的最大价值在 SFT 数据稀缺的任务上（如 editing）；(3) "先 SFT 再 RL" 的两阶段范式对 dLLM 似乎是必需的

---

### [P-Post-02] dLLM 后训练存在三条互补路线：SFT 冷启动 + RL 精炼 + Test-time Scaling
- **现象**: KB 中的 dLLM 后训练方案可归纳为三条互补路线，各有最适场景：(1) **SFT 冷启动**——MMaDA Mixed CoT SFT (+47.8 GSM8K)、LaViDa-R1 Stage 1 (100K steps)、DiMOO Stage III (30M)、LLaDA-V 推理增强训练，贡献主要性能提升；(2) **RL 精炼**——UniGRPO、统一 PG、Self-GRPO、GSPO (OpenMMReasoner)，边际优化但在 SFT 覆盖不到的任务上价值高；(3) **Test-time Scaling**——dMLLM-TTS HTS 搜索、LaViDa-R1 Tree Search，推理时搜索作为"后期补丁"
- **支撑论文**: [[2025-MMaDA]]（SFT >> RL）、[[2026-LaViDa-R1]]（SFT + RL 两阶段，tree search 推理时搜索）、[[2025-Lumina-DiMOO]]（四阶段管线含 SFT + Self-GRPO）、[[2025-dMLLM-TTS]]（test-time scaling，基座越弱收益越大）、[[2025-OpenMMReasoner]]（传统 VLM 上 SFT + GSPO RL）
- **可能解释**: (1) SFT 激活被预训练压制的能力（主要贡献），RL 在 SFT 覆盖不到的分布上探索（边际贡献），test-time 在固定模型上做搜索（补丁贡献）；(2) 三者的投入产出比递减——SFT 的人均收益最高，test-time 最低但最灵活
- **例外情况**: (1) LaViDa-R1 image editing 上 RL 贡献（+0.10）大于 SFT（+0.01），说明 SFT 数据覆盖不到时 RL 价值更高；(2) 基座很弱时 test-time scaling 收���很大（MMaDA +29.4%），但 P-RL-06 指出这是"训练不足的补丁"
- **启示**: 最优投入优先级：SFT 数据质量 > RL 算法 > 推理时搜索。但三者正交可组合——先 SFT 冷启动，再 RL 精炼，最后 test-time scaling 做最终提升。EBPO 的 shrinkage baseline 可直接改进 RL 阶段的 GRPO 训练效率

---

### [P-Post-03] 推理 SFT 数据应默认不过滤或最小过滤（pre-pattern）
- **现象**: OpenMMReasoner 明确采用 "no-filtering policy"，保留教师模型 ×8 采样的全部输出（包括"非最优但正确"的推理路径），性能反而优于 aggressive filtering。MMaDA 的 Mixed CoT SFT 使用跨领域广泛混合而非精细过滤，贡献 +47.8 GSM8K
- **支撑论文**: [[2025-OpenMMReasoner]]（显式 no-filtering 策略优于过滤变体）、[[2025-MMaDA]]（跨域广泛混合无激进过滤，SFT 贡献远大于 RL）
- **可能解释**: (1) **噪声正则化**: "非最优"推理路径提供 reasoning-level label smoothing，防止过拟合单一推理模式；(2) **覆盖保留**: 过滤创建系统性分布 gap——删除含 backtracking 的推理链导致模型不会从错误中恢复；(3) **探索预调**: messy 推理链降低 SFT 后策略的过度自信，为 RL 阶段保留探索空间
- **例外情况**: (1) 教师模型在特定领域严重不准确（accuracy <50%）时需过滤错误答案（非"路径不优"而是"答案错误"）；(2) 任务有唯一正确答案且推理路径固定时（如简单分类），多样性无增量信息；(3) 纯事实性错误必须过滤
- **启示**: 推理 SFT 数据构造应"先保留多样性，仅过滤事实性错误"。与 [P-RL-07] 互补——P-RL-07 是"多样性比规模重要"的宏观原则，P-Post-03 是"实现多样性的具体策略之一：不过滤"。**注**: 当前为 pre-pattern，MMaDA 未显式框架为 "no-filtering"，但其广泛混合做法方向一致

---

### [P-Post-04] 跨域推理迁移：文本推理能力正向迁移到多模态推理
- **现象**: OpenMMReasoner 在纯文本数学推理数据上训练后，多模态推理 benchmark 同步提升（"textual reasoning transfers alongside strengthened multimodal reasoning"）。MMaDA Mixed Long-CoT SFT 混合文本推理 + 多模态理解 + T2I 数据，所有指标同步提升（Figure 6），无显著 seesaw。Kimi K2.5 在 AR MoE 模型上观察到视觉 RL 提升文本基准（MMLU-Pro +1.7%, GPQA-Diamond +2.1%），将跨域迁移从 dLLM 推广到 AR 架构
- **支撑论文**: [[2025-OpenMMReasoner]]（文本推理→多模态推理正向迁移，9 个 benchmark 验证）、[[2025-MMaDA]]（跨域混合 SFT 无 seesaw）、[[2025-KimiK2.5]]（AR MoE 上视觉 RL→文本基准正向迁移）
- **可能解释**: (1) 推理能力的核心组件（逻辑链构建、证据整合、假设验证）是模态无关的，文本推理训练强化这些通用组件后自动惠及多模态推理；(2) CoT 格式的学习可跨模态迁移——文本 CoT 的结构化推理模式被多模态推理复用；(3) 多任务训练提供正则化，防止过拟合单一模态
- **例外情况**: (1) 迁移方向可能不对称——文本→视觉迁移效果可能强于视觉→文本（K2.5 的 +1.7% 效应量较小）；(2) 任务比例严重失衡时某模态可能欠训练；(3) 纯视觉感知任务（如 OCR、细粒度识别）可能不受益于文本推理迁移
- **启示**: 多模态推理训练应默认包含文本推理数据作为"推理能力增强剂"。数据配比优化应关注推理能力的跨域迁移效率，而非仅关注目标模态的数据量
