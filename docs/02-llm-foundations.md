# 第 2 章：LLM 基础知识

本章涵盖大语言模型的底层架构、注意力机制、微调技术等基础知识。

## 核心知识点总结
- Transformer 架构原理
- 注意力机制（Self-Attention, Multi-Head Attention）
- Tokenization 分词技术
- 预训练与微调（RLHF, SFT, PEFT/LoRA）
- 大模型幻觉问题与缓解

## 高频面试题

### 1. 简述 Transformer 的核心结构及其优势。
**参考答案**：
Transformer 由 Encoder 和 Decoder 组成，核心是 Self-Attention（自注意力机制）和 Feed-Forward Neural Network（前馈神经网络）。相对于 RNN/LSTM 它的优势在于：
- 可以并行计算，极大提升训练速度。
- 更好地捕捉长距离依赖关系。

### 2. 什么是幻觉（Hallucination），如何缓解？
**参考答案**：
幻觉是指 LLM 生成看似合理但与事实不符或在上下文中无根据的内容。
缓解方法：
1. 使用 RAG（检索增强生成）提供外部事实基础。
2. 给出更严谨的 Prompt（例如："如果不知道请回答不知道"）。
3. 调整 Temperature（降温以降低随机性）。
4. 对模型进行 RLHF 微调。

*(更多面试题持续补充中...)*
