# 多模态大模型研究知识库项目

## 项目概述
这是一个多模态大模型领域的研究知识库，用于系统化管理论文笔记、跟踪研究问题、发现研究方向。

## 知识库结构
- `paper-kb/papers/` — 逐篇论文的结构化笔记（Level 1/2/3）
- `paper-kb/topics/` — 按主题组织的综述和跨论文经验规律（patterns）
- `paper-kb/problems/problem-tree.md` — 树状研究问题图谱
- `paper-kb/problems/solved-log.md` — 已解决问题记录
- `paper-kb/ideas/` — 研究方案构想
- `paper-kb/templates/paper-template.md` — 论文笔记模板

## 研究方向
聚焦多模态大模型四个维度:
1. 多模态预训练（视觉编码器、连接器、数据工程、训练策略）
2. 多模态后训练（SFT、对齐、DPO变体）
3. 多模态RL（reward建模、策略优化、RL for reasoning/hallucination）
4. 基于Diffusion的理解生成一体化（AR+Diffusion混合、离散化统一、解耦编码、视觉Tokenizer）

## 自定义命令
- `/read-paper [PDF路径或arxiv链接]` — 三阶段论文阅读: Agent生成草稿 → 子Agent深度讨论 → 用户审阅
- `/explore-ideas [问题描述或节点ID]` — 基于知识库探索潜在研究方向，生成候选算法方案
- `/research-question [研究问题]` — 深度分析特定研究问题，输出全景分析+新方案+路线图
- `/update-kb [范围]` — 维护知识库: 更新关联、patterns、问题树状态

## 论文笔记规范
- 文件名: `YYYY-论文简称.md`
- Level 1 (5min): 索引卡，一句话总结+tags+核心insight
- Level 2 (15min): 方法拆解，Building Blocks含mechanism+适用条件+失败模式
- Level 3 (30min): 完整分析，问题定位+个人评注
- Building Block 必须包含 WHY it works（机制层描述）

## Agent 工作原则
- 生成方案时必须基于知识库中的已有 Building Blocks，注明出处
- 明确区分"有论文支持的结论"和"推测性判断"
- 关注 anti-patterns 和失败模式，避免提出已被证伪的方案
- 组合 blocks 时需分析机制层面的兼容性
