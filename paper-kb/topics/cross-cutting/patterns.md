# 跨方向 — 经验规律

> 跨越多个子方向的通用经验规律。

## 格式

```
### [P-CC-xx] 规律名称
- **现象**:
- **涉及方向**: []
- **支撑论文**:
- **可能解释**:
- **例外情况**:
- **启示**:
```

---

### [P-CC-01] dLLM 的 Infilling 是 AR 不可复制的结构性优势
- **现象**: dLLM 的双向注意力天然支持文本 infilling——给定前缀和后缀，填充中间部分。LaViDa FIM 在约束诗歌补全上达到 1.00 vs AR (LLaVa-1.6) 仅 0.41/0.37（句子级/样本级）。这一能力被 LaViDa-R1 转化为 answer-forcing——将 ground-truth answer 注入末尾利用 dLLM inpainting 反向填充推理链，解决了 RL 中"困难问题全部得零分导致训练信号消失"的核心难题。AR 模型需要特殊的 FIM 训练格式（prefix-suffix-middle 重排序），而 dLLM 只需将前后缀标记为 clean、中间部分标记为 [MASK]
- **涉及方向**: [diffusion-foundation, rl, unified-model]
- **支撑论文**: [[2025-LaViDa]]（FIM 能力验证，infilling 1.00 vs 0.41）、[[2026-LaViDa-R1]]（answer-forcing 利用 infilling 做 guided exploration，解决训练信号消失）
- **可能解释**: (1) dLLM 的 mask-predict 目标天然训练了"从已知上下文恢复缺失信息"的能力，infilling 是这一能力的直接应用；(2) 双向注意力使模型可同时看到前缀和后缀信息，提供更丰富的生成上下文；(3) AR 模型只能顺序生成，看不到后缀信息
- **例外情况**: (1) 约束非常长且复杂时，dLLM 的并行生成可能难以保证全局一致性；(2) answer-forcing 注入比例过高（50%+）会导致 collapse（扭曲 advantage 分布）；(3) infilling 区域很大、上下文很少时退化为无条件生成
- **启示**: Infilling 是 dLLM 相对 AR 的"杀手级"独有能力，应作为 dLLM 路线的核心卖点。未来方向：(1) 扩展到 code infilling（给定测试用例输出反填代码）；(2) 图像 inpainting 的类比应用；(3) 多轮对话中的 contextual generation

---

### [P-CC-02] dLLM 统一模型的自评估范式在训练和推理中均有效
- **现象**: 多篇论文独立发现利用 dLLM 统一模型自身的理解能力评估生成质量是可行的：(1) DiMOO Self-GRPO 用自身理解能力生成 entity-relation-value 三元组评估 T2I 生成，联合优化 T2I+理解；(2) dMLLM-TTS 的 Self-Verified Feedback 用自身理解能力在 test-time 筛选高质量候选（GenEval +17.9%）；(3) ReDiff Stage II 用模型自身草稿 + 专家修正做"在线自我纠错学习"
- **涉及方向**: [rl, unified-model, posttraining]
- **支撑论文**: [[2025-Lumina-DiMOO]]（Self-GRPO 自评估联合 RL）、[[2025-dMLLM-TTS]]（Self-Verified Feedback 推理时自评估）、[[2025-ReDiff]]（模型自身草稿做在线纠错训练）
- **可能解释**: (1) 统一模型同时具备理解和生成能力，使"生成→理解评估→反馈优化"的闭环成为可能；(2) 自评估比外部 reward model（如 CLIP）更贴合模型的实际生成分布——自己的理解偏差与生成偏差对齐；(3) 自评估免去了训练/部署独立 reward model 的成本
- **例外情况**: (1) Bootstrapping 偏差——理解能力有偏差时，reward signal 也有偏差，形成正反馈循环；(2) 自评估可能出现"自我接受偏差"（sycophancy），倾向于给自己的生成高分；(3) 理解能力弱的模型（如 MMaDA MMMU 30.2）可能不适合自评估
- **启示**: 自评估范式是统一模型的独有优势——AR 的纯生成/纯理解模型无法形成此闭环。但需要引入外部校准（anchor reward 或 VLM-as-judge）周期性修正 bootstrapping 偏差。最有价值的场景：已部署的统一模型不可重训练时，test-time 自评估提供免费的质量提升
