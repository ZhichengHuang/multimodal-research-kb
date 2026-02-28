# 多模态大模型研究知识库

## 研究方向总览

本知识库聚焦于多模态大模型的四个核心维度:

```
多模态预训练 × 多模态后训练 × 多模态RL × Diffusion统一架构
```

## 目录索引

### 论文笔记
- `papers/` — 逐篇论文的结构化笔记 (Level 1/2/3)

### 主题综述
- `topics/pretraining/` — 视觉编码器、连接器、数据工程、训练策略、架构、Scaling law
- `topics/posttraining/` — 多模态SFT、对齐、DPO变体、数据筛选
- `topics/rl/` — Reward建模、策略优化、RL减幻觉、RL提推理
- `topics/diffusion-foundation/` — 扩散范式(DDPM/Flow Matching)、潜空间扩散、离散扩散、DiT架构
- `topics/visual-tokenizer/` — 离散Tokenizer、量化策略、连续表示、理解/生成Tokenizer需求
- `topics/unified-model/` — AR+Diffusion混合、纯AR离散化、解耦编码、Diffusion原生、训练策略、视频统一、世界模型
- `topics/cross-cutting/` — 多模态Scaling law、数据效率、评估体系

### 问题追踪
- `problems/problem-tree.md` — 树状结构的研究问题图谱
- `problems/solved-log.md` — 已解决问题的记录

### 研究构想
- `ideas/` — Agent 和研究者共同产生的算法方案

## 知识库使用方式

### 日常读论文流程
1. `/read-paper [PDF路径或arxiv链接]` — Agent 自动生成草稿 + 子Agent深度讨论 + 你审阅
2. 审阅后确认，笔记存入 `papers/`，关联信息更新到 `topics/` 和 `problems/`

### 探索研究方向
1. `/explore-ideas [问题描述或问题树节点ID]` — Agent 搜索知识库，生成候选研究方案
2. `/research-question [具体研究问题]` — 深度分析，输出带依据的方案对比

### 维护知识库
1. `/update-kb` — 扫描知识库，更新跨论文关联、经验规律和问题树状态

## 统计信息
- 论文总数: 0
- Level 1: 0 | Level 2: 0 | Level 3: 0
- 最后更新: -
