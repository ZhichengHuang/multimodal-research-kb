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

### [P-RL-02] dLLM RL 的 Log-Likelihood 估计需覆盖完整去噪时间步
- **现象**: 固定 mask ratio（如 d1 的 t=1 全 mask）劣于动态覆盖策略。MMaDA 的随机 mask ratio p∈[0,1] 优于 d1；LaViDa-R1 的 complementary masking (w=1) 在互补覆盖和方差降低上进一步改进
- **支撑论文**: [[2025-MMaDA]]（结构化随机 mask ratio 优于 d1 的固定策略）、[[2026-LaViDa-R1]]（complementary masking 覆盖全部 token，w=1 避免权重不平衡）
- **可能解释**: 完整覆盖时间步提供更准确的 ELBO 估计，减少 train-inference gap；i.i.d. 采样有遗漏问题，互补（antithetic）采样降低方差约一半
- **例外情况**: 当序列很短时，覆盖策略的差异不显著
- **启示**: dLLM RL 的 likelihood 估计方案仍有改进空间——目前最优的 complementary masking 仍是经验选择，可能存在理论更优的方案（importance sampling / control variates）

---

### [P-RL-03] Mixed SFT 冷启动是 dLLM RL 成功的必要前提
- **现象**: MMaDA 在 Mixed Long-CoT SFT 之后做 UniGRPO RL（GSM8K: 17.4→65.2→73.4，SFT 贡献远大于 RL）；LaViDa-R1 在 100K steps SFT 后做 5K steps RL。两者均采用先 SFT 再 RL 的两阶段策略
- **支撑论文**: [[2025-MMaDA]]（CoT SFT 带来 +47.8, RL 再 +8.2）、[[2026-LaViDa-R1]]（Stage 1 SFT 远大于 Stage 2 RL 步数）
- **可能解释**: dLLM 从预训练到 RL 的 policy gap 过大，需要 SFT 提供合理的初始 policy 和统一 CoT 格式；dLLM 的并行去噪使随机探索效率更低，更依赖有意义的起点
- **例外情况**: 对极简单任务（格式遵循等）可能直接 RL；d1 直接对 LLaDA 做 RL 但效果有限
- **启示**: dLLM 后训练应为"大规模 CoT SFT → 有限步 RL 精炼"，SFT 数据质量和覆盖度是 RL 天花板的关键
