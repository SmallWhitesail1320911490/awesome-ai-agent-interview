# 第 3 章：RAG 检索增强生成

本章涉及 RAG 的工作流、向量数据库选型、分块策略与高级检索优化。

## 核心知识点总结
- RAG 原理与工作流程
- 向量数据库选型（Milvus, Pinecone, Chroma 等）
- Embedding 模型选择
- 文档分块策略（Chunking）
- 检索优化（混合检索、重排序）
- RAG 与 Agent 结合架构

## 高频面试题

### 1. 请简述 RAG (Retrieval-Augmented Generation) 的完整流程。
**参考答案**：
1. **数据准备**：将外部文档提取文本、分块（Chunking），使用 Embedding 模型转换为向量，存入向量数据库。
2. **检索阶段**：用户提出 query，将其向量化后去向量数据库中进行相似度检索（如 Cosine Similarity），召回 Top-K 相关的 Chunk。
3. **生成阶段**：将召回的内容与原 query 组合为 Prompt，交给 LLM 生成最终答案。

### 2. RAG 中的 Chunking 策略有哪些？如何选择 Chunk 大小？
**参考答案**：
常见的 Chunking 策略：固定大小切分、按段落/句子切分、按语义切分结构化切分。
Chunk 大小选择应平衡：
- 小 Chunk 召回更精确，但可能丢失上下文。
- 大 Chunk 保留上下文完整性，但带来更多噪声和成本（Token 开销）。通常设置 256-1024 Token 之间，带一定的 Overlap（重叠）。

*(更多面试题持续补充中...)*
