# 第 9 章：部署与性能优化

本章涵盖 LLM 模型推理加速和底层部署知识。

## 核心知识点总结
- KV Cache / FlashAttention / PagedAttention
- 模型量化与蒸馏
- DeepSpeed / vLLM 推理优化
- Docker/K8s 云原生部署

## 高频面试题

### 1. 简述 vLLM 中 PagedAttention 的核心思想。
**参考答案**：
传统的 LLM 推理在管理 KV Cache 时会预留连续内存，导致大量的内存碎片和显存浪费。
PagedAttention 借鉴了操作系统的虚拟内存分页机制，将连续的 KV Cache 切分成固定大小的 Block（页），离散地分布在显存中。这极大减少了显存碎片（内存浪费降到了 4% 以下），从而在相同的硬件上容纳更多的并发请求（提高 Throughput）。

*(更多面试题持续补充中...)*
