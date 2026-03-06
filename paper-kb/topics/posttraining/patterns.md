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
