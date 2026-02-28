你是一个知识库维护助手。请扫描 `paper-kb/` 知识库，进行以下维护工作。

可选参数: $ARGUMENTS
（可指定 "relations" / "patterns" / "problem-tree" / "all" 来限定更新范围，默认 all）

---

## Task 1: 更新论文间关联 (relations)

1. 扫描 `paper-kb/papers/` 中所有论文笔记
2. 检查每篇论文的 `Relations (结构化)` 字段
3. 发现缺失的关联:
   - 如果论文 A 标注了和论文 B 的关系，检查论文 B 是否也有反向关联
   - 基于 tags 和 building blocks 的相似性，发现未标注的潜在关联
4. 向用户报告发现的新关联，确认后更新

## Task 2: 更新经验规律 (patterns)

1. 扫描 `paper-kb/papers/` 中所有论文的 Building Blocks 和实验发现
2. 寻找跨论文的共同模式:
   - 多篇论文是否在相似条件下观察到类似结果？
   - 有没有 2 篇以上论文支持的新 pattern？
   - 已有 pattern 是否有新的支撑论文或反例？
3. 将发现写入对应的 `topics/*/patterns.md`

## Task 3: 更新问题树 (problem-tree)

1. 读取 `paper-kb/problems/problem-tree.md`
2. 扫描最近新增的论文笔记:
   - 有没有论文部分或完全解决了某个 🔴 节点？→ 考虑改为 🟡 或 🟢
   - 有没有论文暴露了新的问题？→ 添加新节点
   - 有没有新的"已有尝试"和"相关论文"需要补充？
3. 如果有节点从 🔴 变为 🟢，同步更新 `problems/solved-log.md`
4. 向用户报告所有变更，确认后更新

## Task 4: 更新统计信息

更新 `paper-kb/OVERVIEW.md` 中的统计信息:
- 论文总数
- 各 Level 的论文数量
- 最后更新日期

---

## 输出格式

```markdown
## 知识库更新报告

### 新发现的关联
- 论文A ↔ 论文B: [关系类型] 描述

### 新发现/更新的 Pattern
- [P-xx-xx] 规律名称: 描述

### 问题树变更
- [节点ID] 状态变更: 🔴 → 🟡 (原因: ...)
- [新节点ID] 新增问题: ...

### 统计信息更新
- 论文总数: N
- Level 1/2/3: x/y/z
```

请将上述变更逐项呈现给用户确认后再执行修改。
