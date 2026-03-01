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

### [P-Diff-01] 8B 规模 Masked Diffusion 模型在理解和生成上可达 Competitive 水平
- **现象**: MMaDA (LLaDA-8B) 在 GSM8K 达 73.4、T2I 超 SDXL/Janus、多模态理解超 Show-o/SEED-X。LaViDa-R1 (LaViDa-O-10.4B) 在 Lisa-Grounding 上超越 specialist 模型
- **支撑论文**: [[2025-MMaDA]]（多任务 competitive 性能）、[[2026-LaViDa-R1]]（grounding specialist level）
- **可能解释**: 离散扩散的 bidirectional attention 在理解任务上天然优于 AR causal attention；mask-predict 目标与 MLM 的相似性使其继承了 BERT 类模型的理解优势
- **例外情况**: (1) 纯文本 sequential reasoning 仍弱于 AR（MATH500 36.0 vs Qwen2-7B 更高）；(2) 位置推理弱（GenEval Position 0.20）；(3) 高分辨率生成与 FLUX/SD3 有差距
- **启示**: 离散扩散是成熟的统一模型路线，主要局限在 sequential reasoning 和高分辨率生成——这些不是根本架构缺陷而是训练规模问题
