你是一个知识库维护助手。请扫描 `paper-kb/` 知识库，进行以下维护工作。

可选参数: $ARGUMENTS
（可指定 "relations" / "patterns" / "problem-tree" / "all" 来限定更新范围，默认 all）

---

## Task 0: 知识库结构健康检查 (Obsidian CLI)

如果 `obsidian` CLI 可用（通过 `which obsidian` 检查），执行以下检查:

1. **孤立论文检测**: `obsidian orphans`
   → 找到没有任何链接的论文，建议添加关联

2. **未解析引用 → 智能 To-Read 列表**: `obsidian unresolved`
   → 找到 `[[引用]]` 指向不存在的文件，对每个 unresolved link 生成带优先级的 to-read 列表:

   优先级评估逻辑:
   - **P0 (必读)**: 被 3+ 篇论文引用，或被 problem-tree 直接提及，或作为 `extends`/`motivated_by` 的 target
   - **P1 (推荐)**: 被 2 篇论文引用，或被 `importance: high` 的论文在 Relations 中引用
   - **P2 (可选)**: 被 1 篇论文引用，且仅出现在 `combines_with` 或正文中

   输出格式:
   ```markdown
   ### 📚 To-Read 优先级列表

   #### P0 必读
   - [[论文名]] — 被 N 篇论文引用, 关联节点 [节点ID], 作为 type target
     - 引用者: [[引用者1]], [[引用者2]]

   #### P1 推荐
   - [[论文名]] — 被 N 篇高重要性论文引用, 作为 type target

   #### P2 可选
   - [[论文名]] — 被 1 篇论文在"互补"中提及
   ```

3. **死胡同检测**: `obsidian deadends`
   → 找到只有入链没有出链的论文，可能需要补充 relations

4. **Tag 覆盖度分析**: `obsidian tags --count`
   → 发现覆盖薄弱的研究方向

5. **问题树关联完整性**: `obsidian backlinks paper-kb/problems/problem-tree.md`
   → 检查有多少论文关联到问题树

如果 `obsidian` CLI 不可用，跳过 Task 0，直接进入 Task 1。

---

## Task 1: 更新论文间关联 (relations)

1. 扫描 `paper-kb/papers/` 中所有论文笔记
2. 检查每篇论文的 `Relations (结构化)` 字段（格式: `` `type` → [[论文]]: 说明 ``）
3. 发现缺失的关联:
   - 如果论文 A 标注了和论文 B 的关系，检查论文 B 是否也有反向关联
   - 基于 tags 和 building blocks 的相似性，发现未标注的潜在关联
   - 如果 `obsidian` CLI 可用:
     - 使用 `obsidian backlinks <paper>` 检查反向关联完整性
     - 使用 `obsidian links <paper>` 获取出链列表，交叉验证
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
