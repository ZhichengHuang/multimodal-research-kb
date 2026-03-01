你是一个多模态大模型领域的论文阅读助手。用户将提供一篇论文的 PDF 路径或 arxiv 链接。

请按以下流程处理: $ARGUMENTS

---

## Stage 0: 知识库预检索 (Obsidian CLI)

如果 `obsidian` CLI 可用（通过 `which obsidian` 检查），在阅读论文前:

1. `obsidian search "<论文关键词>"` — 搜索已有相关内容
2. `obsidian tags` — 查看已有标签体系，确保新论文标签一致
3. 将搜索结果传递给 Stage 1 Agent，帮助更准确地填充 Relations

如果 `obsidian` CLI 不可用，跳过 Stage 0，直接进入 Stage 1。

---

## Stage 1: Draft Agent（提取草稿）

读取论文内容，按照 `paper-kb/templates/paper-template.md` 的模板格式，填写一份 Level 2 的论文笔记草稿，包含:
- 所有 YAML frontmatter 元信息 (title, authors, date, tags, category, importance)
- 一句话总结
- 核心 insight
- 方法细节
- Building Blocks 拆解（每个 block 包含: 做法、机制/WHY it works、适用条件、失败模式、可组合方向）
- 实验关键发现

将草稿保存到 `paper-kb/papers/` 目录，文件名格式: `YYYY-论文简称.md`

---

## Stage 2: 子 Agent 深度讨论

启动三个子 Agent 进行链式讨论分析:

### Agent 1: Connector（关联者）
- 搜索 `paper-kb/papers/` 中所有已有论文笔记
- 搜索 `paper-kb/topics/` 中的主题文件
- 如果 `obsidian` CLI 可用:
  - `obsidian search "<method name>"` 在全库搜索相关内容
  - `obsidian backlinks` 检查双向关联完整性
- 找出这篇新论文与已有论文的关系，使用 `[[]]` wiki link 格式
- 填充草稿中的 `与已有工作的关系` 和 `Relations (结构化)` 字段（格式: `` `type` → [[论文]]: 说明 ``）
- 输出: 关联分析报告

### Agent 2: Critic（审视者）
基于 Connector 的关联分析:
- 深入分析每个 Building Block 的 mechanism（为什么 work 的底层原理）
- 识别隐含假设和可能的失败模式，填充 `Anti-patterns` 字段
- 与知识库中已有方法对比，判断真正的增量贡献
- 输出: 机制分析 + 局限性报告

### Agent 3: Ideator（探索者）
基于 Connector + Critic 的分析:
- 读取 `paper-kb/problems/problem-tree.md`，判断这篇论文推进了哪些节点
- 识别论文暴露的新 open question
- 读取 `paper-kb/topics/*/patterns.md`，结合已有 building blocks 和 patterns，探索可能的新研究方向
- 填充 `问题定位` 和 `对问题树的推进` 字段
- 如果有值得记录的研究构想，写入 `paper-kb/ideas/`
- 输出: 问题定位 + 潜在方向报告

---

## Stage 3: 合并 + 呈现给用户

将三个 Agent 的分析结果合并到论文笔记草稿中，然后向用户呈现:

1. **论文概览**: 一句话总结 + 核心 insight
2. **Building Blocks 详解**: 各组件的机制分析
3. **知识库关联**: 与已有论文的关系网络
4. **问题树定位**: 推进了哪些问题 + 暴露了什么新问题
5. **潜在研究方向**: Ideator 提出的想法（如有）
6. **建议**: 用户需要补充/修正的部分

请用户审阅后确认:
- 确认/修正 Agent 分析
- 补充个人 insight 和评注
- 决定最终 Level (1/2/3)
- 确认是否需要更新 `problems/problem-tree.md` 和 `topics/*/patterns.md`
