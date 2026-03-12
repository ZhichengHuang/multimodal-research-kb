# 视觉 Tokenizer — 跨论文经验规律

> 从多篇论文中归纳出的经验规律。每条 pattern 需要至少 2 篇论文支撑。

## 格式

```
### [P-Tok-xx] 规律名称
- **现象**:
- **支撑论文**:
- **可能解释**:
- **例外情况**:
- **启示**:
```

---

### [P-Tok-01] 纯重建 Tokenizer 存在 Anti-scaling 现象，语义增强是生成 Scaling 的必要条件
- **现象**: 纯重建目标训练的视觉 tokenizer 在 compute 增加时出现 anti-scaling——重建质量（rFID）持续改善但生成质量（gFID）反而恶化。VTP 量化了这一现象: rFID 2.0→0.5（改善）但 gFID 55.04→58.56（恶化）。引入语义目标（CLIP + DINOv2 SSL）后 gFID 持续改善至 1.11。Concurrent work RAE 在 S→L 规模时 gFID 从 3.50 恶化到 6.09，进一步验证了纯重建/单一语义目标的 scaling 局限性
- **支撑论文**: [[2025-VTP]]（系统性量化 anti-scaling，CLIP+SSL+重建三目标解决）、[[RAE]]（concurrent work，语义增强但 scaling 设计不当仍退化）、[[2025-Lumina-DiMOO]]（间接支撑——DiMOO 使用纯重建 aMUSEd-VQ，在低级视觉任务上表现弱，暗示重建 tokenizer 的语义不足）
- **可能解释**: (1) 纯重建目标在 compute 增加时将额外容量分配给对生成无益的高频像素细节（纹理/噪声），latent space 被"像素噪声污染"；(2) 语义目标（CLIP, SSL）迫使 latent space 编码 object identity、spatial relations、text-image alignment——这些恰好是 diffusion 生成模型最需要的条件信号；(3) 语义增强的 latent space 使 diffusion model 更容易学习 text→image 映射，因此 gFID 随 compute 持续改善而非饱和
- **例外情况**: (1) 极小 compute 预算下（<1/10 总 compute），重建目标尚未饱和，此时多目标训练的额外开销可能不值得；(2) 下游任务本身是像素级任务（super-resolution, inpainting）时，重建优化可能不会饱和；(3) VTP 仅在连续 latent 上验证，离散 VQ tokenizer 上多目标训练��否同样有效未知（codebook collapse 风险）
- **启示**: 任何 tokenizer 训练都应监控 rFID 和 gFID 的 scaling 曲线，一旦出现 anti-scaling 信号即引入语义目标。未来 dLLM 统一模型应优先使用语义增强 tokenizer（VTP 或类似方案），而非纯重建 tokenizer（aMUSEd-VQ, MAGVIT-v2）。"理解 vs 生成矛盾"是伪命题——语义增强反而促进生成

---

### [P-Tok-02] Tokenizer 质量上界由语义信息密度而非重建保真度决定
- **现象**: VTP 的 rFID 0.36（重建极好）+ gFID 1.11（生成极好）同时达到，而纯重建 tokenizer 的 rFID 0.5（重建也好）但 gFID 58.56（生成极差）。DiMOO 使用纯重建 aMUSEd-VQ 在低级视觉任务（super-res, dehazing）表现弱，但通过 ~110M 大规模数据在高级任务（GenEval 88%）上表现好——说明下游模型可部分弥补 tokenizer 语义不足，但有上界
- **支撑论文**: [[2025-VTP]]（anti-scaling 量化 + 语义增强解决）、[[2025-Lumina-DiMOO]]（纯 VQ tokenizer 的语义不足被大规模数据部分弥补，但低级视觉任务仍受限）
- **可能解释**: (1) Diffusion 生成模型在 latent space 上学习 text→image 映射，语义丰富的 latent space 提供更好的"地图"；(2) 重建保真度高但语义稀疏的 latent space 迫使 diffusion model 自行学习语义结构，增加学习难度；(3) 语义信息密度决定了 latent space 的"可导航性"——语义越丰富，diffusion 轨迹越容易找到正确方向
- **例外情况**: (1) 像素级任务（super-resolution）可能更依赖重建保真度而非语义密度；(2) 极大规模下游模型可能有足够容量自行学习语义（DiMOO 用 8B+110M 数据部分弥补），但这是次优策略
- **启示**: Tokenizer 设计应以"最大化语义信息密度"为首要目标，重建保真度作为约束而非目标。评估 tokenizer 质量时，gFID（生成质量）比 rFID（重建质量）更重要
