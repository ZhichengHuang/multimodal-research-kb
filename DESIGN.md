# 多模态大模型研究知识库 — 系统设计方案

## 1. 项目目标

把「读论文 → 提取知识 → 建立关联 → 发现机会」这条链路结构化，实现:
- **人可以高效阅读和积累**
- **Agent 可以检索、推理、组合出新的算法方案**

读论文服务于三个核心目的:
1. 这篇论文的方法和启发是什么
2. 和过往论文的关系是什么
3. 结合知识库，这篇论文解决了什么问题，还有什么问题没解决，有什么潜在思路

最终目标: 知识库积累到一定规模后，Agent 能基于其中的 Building Blocks、Patterns、问题树，自主设计可能的算法方案。

---

## 2. 研究方向

聚焦**多模态大模型**的四个维度:

| 维度 | 核心关注点 |
|------|-----------|
| 多模态预训练 | 视觉编码器、连接器、数据工程、训练策略、架构、Scaling Law |
| 多模态后训练 | SFT、对齐、DPO 变体、数据筛选、幻觉消除 |
| 多模态 RL | Reward 建模、策略优化 (GRPO/PPO)、RL for reasoning/hallucination |
| Diffusion 统一模型 | AR+Diffusion 混合、纯 AR 离散化、解耦编码、Diffusion 原生、视觉 Tokenizer |

---

## 3. 知识库目录结构

```
multimodal-research-kb/
├── CLAUDE.md                           # 项目说明 (Claude Code 自动加载)
├── DESIGN.md                           # 本设计文档
├── .gitignore
│
├── .claude/commands/                   # 自定义 slash commands
│   ├── read-paper.md                   # /read-paper: 三阶段论文阅读流程
│   ├── explore-ideas.md                # /explore-ideas: 探索研究方向
│   ├── research-question.md            # /research-question: 深度研究分析
│   └── update-kb.md                    # /update-kb: 知识库维护
│
└── paper-kb/                           # 知识库主体 (Obsidian vault 根目录为项目根)
    ├── OVERVIEW.md                     # 知识库总览 + 统计信息 + Dataview 动态查询
    ├── templates/
    │   └── paper-template.md           # 论文笔记模板
    ├── papers/                         # 逐篇论文结构化笔记
    ├── ideas/                          # 研究方案构想
    ├── problems/
    │   ├── problem-tree.md             # 树状研究问题图谱
    │   └── solved-log.md              # 已解决问题记录
    └── topics/                         # 按主题组织的综述 + 经验规律
        ├── pretraining/patterns.md
        ├── posttraining/patterns.md
        ├── rl/patterns.md
        ├── diffusion-foundation/patterns.md
        ├── visual-tokenizer/patterns.md
        ├── unified-model/patterns.md
        └── cross-cutting/patterns.md
```

---

## 4. 论文笔记分层记录设计

核心思想: 不是每篇论文都值得花同样时间记录，但每篇读过的论文都应留痕。

### Level 1: 快速索引卡（5 分钟）

适用于大量快速扫过的论文，占比约 60%。

填写内容:
- YAML frontmatter 元信息 (title, authors, date, tags, category)
- 一句话总结
- 核心 insight
- 与已有工作的简要关系

用途: Agent 做检索索引，你知道这篇论文的存在和大意。

### Level 2: 方法拆解（15 分钟）

适用于和研究方向直接相关的重要论文，占比约 30%。

在 Level 1 基础上增加:
- 方法细节 + 关键公式
- **Building Blocks 拆解**（核心价值）
  - 每个 block 包含: 做法、机制 (WHY it works)、适用条件、失败模式、可组合方向
- Anti-patterns / 已知失败模式
- 实验关键发现
- 结构化 Relations

用途: Agent 提取 building blocks 做组合创新。

### Level 3: 完整研究分析（30 分钟）

适用于 milestone 级论文，占比约 10%。

在 Level 2 基础上再增加:
- 问题定位（解决了什么 / 没解决什么 / 潜在思路）
- 对问题树的推进
- **个人深度评注**（最有价值的信息，Agent 无法替代）

用途: 产生新想法，为 Agent 提供方向约束和品味判断。

### Level 可升级

先用 Level 1 快速扫过，后续发现重要性时再补充到 Level 2/3。

---

## 5. Building Blocks 设计（知识库的核心资产）

Building Block 是论文方法拆解出的可复用零件，是 Agent 做组合创新的原材料。

### 每个 Block 必须包含的字段

```markdown
### Block: [组件名称]
- **做法**: 具体怎么做的
- **机制 (WHY it works)**: 底层原理，为什么这个方法有效
- **适用条件**: 在什么场景下可以用
- **什么时候会 break**: 失败模式和边界条件
- **可组合方向**: 和什么类型的 block 互补
```

### 为什么 mechanism 层描述至关重要

Agent 做组合创新时，需要判断两个 block 在机制层面是否兼容。

例如:
- GRPO 的 group relative baseline 本质是 **Monte Carlo estimation of baseline**
- Process Reward Model 本质是 **sparse reward → dense step-level signal**

这两个在机制层面互补（一个解决 baseline 估计，一个解决 reward 信号密度），所以可以组合。
如果只记录表面做法，Agent 无法自主发现这种互补性。

---

## 6. 让 Agent 有效探索研究点的四个关键设计

### 6.1 Mechanism 层描述
给 building block 加机制描述，使 Agent 能理解 block 间的兼容性。这是组合创新的基础。

### 6.2 失败记录 / Anti-patterns
记录什么方法在什么条件下不 work。Agent 不会提出已证伪的方案，保障方案质量下限。

```markdown
## Anti-patterns / 已知失败模式
- 直接把 offline DPO 的 loss 用到 online 采样上 → 训练不稳定
  (来源: paper-x, 原因: importance weight 方差爆炸)
```

### 6.3 跨论文经验规律 (Patterns)
从多篇论文中归纳的共同模式，存放在 `topics/*/patterns.md`。

```markdown
### [P-RL-01] Online > Offline for Reasoning
- **现象**: 在 math/code reasoning 上，online RL 一致优于 offline
- **支撑论文**: [paper-1, paper-2, paper-3]
- **可能解释**: reasoning 需要探索组合空间，offline 数据覆盖不足
- **例外情况**: 简单 reasoning 任务上差距缩小
- **启示**: 新方法如果想做 offline reasoning RL，需要解决分布覆盖问题
```

每条 pattern 需要至少 2 篇论文支撑。Agent 基于 pattern 做可行性推断。

### 6.4 问题树结构
用树状结构组织研究问题（而非扁平列表），标注子问题间的依赖关系和跨分支连接。

问题树六大分支:
- [PT] 预训练阶段
- [Post] 后训练阶段
- [RL] RL 阶段
- [Diff] Diffusion 基础方法论
- [Tok] 视觉 Tokenizer
- [Uni] 理解生成一体化

Agent 可以: 定位到具体子问题 → 理解依赖关系 → 避免提出被上游问题 block 的方案。

---

## 7. 论文间关系系统

### 关系类型（固定 6 种）

| 类型 | 含义 | 举例 |
|------|------|------|
| `extends` | 在 X 基础上改进 | GRPO extends PPO |
| `alternative_to` | 解决同一问题的不同方案 | DPO alternative_to PPO |
| `combines_with` | 可组合使用 | PRM combines_with GRPO |
| `motivated_by` | 被什么问题/现象驱动 | GRPO motivated_by critic-model-cost |
| `enables` | 使某方法成为可能 | DiT enables scalable diffusion |
| `conflicts_with` | 方法间存在冲突 | discrete-token conflicts_with fine-grained-generation |

### 结构化格式（Markdown + Wiki Links）

Relations 使用 markdown + `[[]]` wiki links 格式，取代之前的 YAML 代码块:

```markdown
## Relations (结构化)
<!-- type: extends | alternative_to | combines_with | motivated_by | enables | conflicts_with -->
- `extends` → [[论文文件名]]: 说明
- `alternative_to` → [[论文文件名]]: 说明
- `combines_with` → [[论文文件名]]: 说明
```

好处:
- Obsidian 识别 `[[]]` 为链接，自动建立反向链接 + 图谱显示
- Agent 仍可用正则 `` /`(\w+)` → \[\[(.+?)\]\]: (.+)/ `` 解析结构化信息
- 可读性更好

### 论文引用约定

所有论文引用（包括知识库内和尚未收录的外部论文）统一使用 `[[]]` wiki links:
- KB 内论文: `[[2026-LaViDa-R1]]` → Obsidian 可点击跳转
- 外部论文: `[[DeepSeek-R1]]` → Obsidian 显示为红色 unresolved link，作为天然的"待读提醒"
- 论文简称: 通过 frontmatter 的 `aliases` 字段支持，如 `aliases: [LaViDa-R1]`

Agent 可自动构建关系网络，回答"解决问题 X 有哪些路线？它们之间是什么关系？"

---

## 8. 三阶段论文阅读工作流

### Stage 1: Draft Agent（提取型，自动）

```
输入: PDF 文件 或 arxiv 链接
输出: Level 2 模板草稿

提取内容:
- 元信息 (title, authors, date, tags)
- 一句话总结
- 方法流程 + 关键公式
- Building Blocks 初步拆解
- 论文自己声称的 contribution 和 limitation
```

### Stage 2: 子 Agent 深度讨论（分析型，核心环节）

三个角色按链式依赖讨论:

```
Connector（关联者）先跑
  → 搜索知识库找关联，填充 relations 字段
  → 输出传给 Critic

Critic（审视者）
  → 基于关联上下文分析 mechanism 和局限性
  → 填充 mechanism、anti-patterns 字段
  → 输出传给 Ideator

Ideator（探索者）
  → 基于完整分析，定位问题树 + 探索研究方向
  → 更新 problem-tree，写入 ideas/
```

三个角色是有信息依赖的链式讨论，不是各自独立运行。

### Stage 3: 用户审阅（人类判断）

用户拿到完整分析后:
1. 确认/修正 Agent 对方法的理解
2. 补充个人 insight（最有价值的部分）
3. 调整重要性和 Level
4. 确认关联是否正确
5. 标注 Ideator 提出的方向哪些值得追

---

## 9. 自定义命令设计

### /read-paper [PDF路径或arxiv链接]
三阶段论文阅读流水线: Draft → Connector+Critic+Ideator 深度讨论 → 用户审阅。
输出: 结构化论文笔记 + 关联分析 + 问题定位 + 潜在方向。

### /explore-ideas [问题描述或节点ID]
基于知识库探索潜在研究方向。
流程: 问题定位 → 知识库检索 (papers + blocks + patterns + anti-patterns) → 生成 2-3 个候选方案（含 building blocks 组合、预期优势、风险、关键假设）→ 方案对比推荐。

### /research-question [具体研究问题]
深度研究分析。
流程: 问题拆解 → 全面文献检索 → 现有方案全景 + Gap 分析 → 新方案提出（含验证计划）→ 研究路线图。

### /update-kb [范围]
知识库维护。
任务: 更新论文间关联 → 发现新 patterns → 更新问题树状态 → 刷新统计信息。
所有变更先呈现给用户确认再执行。

---

## 10. Agent 方案设计的工作原理

当知识库积累足够多的论文后，Agent 做方案设计的思维链路:

```
1. 理解问题
   → 在 problem-tree.md 定位到具体子问题节点
   → 获取: 已有尝试 + 为什么难 + 上下游依赖

2. 检索相关信息
   → papers/: 找到相关论文和 building blocks
   → patterns.md: 找到跨论文经验规律
   → anti-patterns: 找到已知失败模式

3. 组合推理
   → 从多篇论文中提取 building blocks
   → 分析 mechanism 层面的兼容性
   → 排除与 anti-patterns 冲突的组合
   → 利用 patterns 做可行性判断

4. 生成方案
   → 2-3 个候选方案，每个包含:
      - 使用了哪些论文的哪些 block
      - 组合的机制理由
      - 预期优势和风险
      - 需要验证的关键假设

5. 评估排序
   → 新颖性、可行性、预期收益、风险、验证成本
```

### Agent 工作原则
- 所有方案必须基于知识库中的已有 Building Blocks，注明出处
- 明确区分"有论文支持的结论"和"推测性判断"
- 关注 anti-patterns 和失败模式，避免提出已被证伪的方案
- 组合 blocks 时需分析机制层面的兼容性
- 如果知识库信息不足，明确指出需要先阅读哪些论文填补空白

---

## 11. 知识库价值增长曲线

```
论文数量    Agent 能力
 0-10      基础检索，关联分析有限
10-30      开始出现有意义的 patterns，block 组合有初步价值
30-50      关系网络形成，Agent 能做跨论文的 gap 分析
 50+       Block 库和 pattern 库足够丰富，Agent 能生成有参考价值的方案
100+       知识库成为真正的研究加速器
```

关键: 知识库的价值不是线性增长，而是在 building blocks 和 patterns 达到一定密度后指数增长。

---

## 12. 使用方式

```bash
# 进入项目目录
cd ~/Project/multimodal-research-kb

# 启动 Claude Code (CLAUDE.md 自动加载)
claude

# 日常读论文
/read-paper https://arxiv.org/abs/xxxx
/read-paper /path/to/paper.pdf

# 探索研究方向
/explore-ideas [Tok-2a] 理解和生成对tokenizer的矛盾需求
/explore-ideas 如何让统一模型的生成质量追上专用模型

# 深度分析研究问题
/research-question 如何在统一模型中融合AR和Diffusion

# 定期维护知识库
/update-kb
/update-kb patterns
/update-kb problem-tree
```

---

## 13. Obsidian 集成

### Vault 配置
- **Vault 根目录**: 项目根目录 `multimodal-research-kb/`（而非 `paper-kb/` 子目录）
- 在 Obsidian 中 "Open folder as vault" 选择项目根目录
- `.obsidian/` 目录由 Obsidian 自动创建，无需手动管理

### Wiki Links 约定
- 所有论文引用使用 `[[]]` wiki link 格式
- KB 内论文: `[[2026-LaViDa-R1]]` — 可点击跳转
- KB 外论文: `[[DeepSeek-R1]]` — 显示为红色 unresolved link，作为"待读提醒"
- 论文简称: 通过 frontmatter `aliases` 字段定义，如 `aliases: [LaViDa-R1]`
- 问题树链接: `[[problem-tree#节点标题]]` 链接到特定节点
- Relations 格式: `` `type` → [[论文]]: 说明 ``

### Obsidian CLI (可选增强)
前提: 安装 Obsidian 并启用 CLI (Settings > General > CLI, 需 Obsidian v1.12+)。

所有 CLI 命令包裹在"如果可用"条件中，基础工作流不依赖 Obsidian:
- `/update-kb` — Task 0 使用 `obsidian orphans/unresolved/deadends/tags/backlinks` 做健康检查
- `/read-paper` — Stage 0 使用 `obsidian search/tags` 做预检索
- `/explore-ideas` 和 `/research-question` — Step 2 使用 `obsidian search/backlinks/links` 扩大检索

### 推荐社区插件
- **Dataview**: OVERVIEW.md 包含 Dataview 查询块，可动态展示论文列表、Tag 统计等
- **Graph View** (内置): 可视化论文关系网络

### Graph View 使用建议
- 过滤显示 `paper-kb/papers/` 目录，聚焦论文关系
- 按 tag 着色节点，直观展示研究方向分布
- 红色 unresolved links 指示待读论文
