# 第 9 章：部署与性能优化

> 大模型推理优化是面试中的深水区，考察对底层原理的理解。
> 内容来源：CSDN、博客园、vLLM 官方文档、腾讯云开发者等。

---

## 一、核心知识点总结

### 1.1 LLM 推理全景

```
用户请求 → Prefill（预填充/首次推理阶段）→ Decode（逐Token 自回归生成）→ 返回结果
```

- **Prefill 阶段**：处理整个 Prompt，compute-bound（计算密集）。
- **Decode 阶段**：逐步生成每个 Token，memory-bound（内存密集）。

### 1.2 核心优化技术

| 技术 | 优化目标 | 核心原理 |
|------|----------|----------|
| KV Cache | 避免重复计算 | 缓存已计算的 K、V 矩阵 |
| PagedAttention | 显存利用率 | 借鉴 OS 分页管理 KV Cache |
| FlashAttention | 计算效率 | IO-aware 的精确注意力算法 |
| GQA/MQA | 减少 KV Cache 大小 | 多个 Q 共享 K/V |
| 模型量化 | 减小模型体积 | FP16→INT8/INT4 |
| 投机解码 | 减少延迟 | 小模型预测 + 大模型验证 |
| Continuous Batching | 吞吐量 | 动态批处理请求 |

---

## 二、高频面试题

### Q1：什么是 KV Cache？为什么只缓存 K 和 V？（⭐⭐⭐⭐⭐）

**参考答案**：

**KV Cache 本质**：在自回归生成过程中，缓存之前所有 Token 的 Key 和 Value 矩阵，避免每生成一个新 Token 时重复计算之前所有 Token 的 K 和 V。

**为什么不缓存 Q**：
- 在 Decode 阶段，每次只有当前新 Token 的 Q 需要计算。
- Q 只用于当前 Token 的 Attention 计算，不会被后续 Token 复用。
- K 和 V 需要所有历史 Token 的值来计算 Attention 分数，所以必须保留。

**KV Cache 的代价**：
- 随序列长度线性增长，对长序列场景显存压力巨大。
- 例如：LLaMA-2-70B，4096 Token 序列，KV Cache 约需 2.5GB 显存。

### Q2：简述 PagedAttention 的核心思想。（⭐⭐⭐⭐⭐）

**参考答案**：

传统 KV Cache 需要连续的显存空间，导致：
- **内部碎片**：预分配过多，浪费显存。
- **外部碎片**：释放后留下不连续的小块。

**PagedAttention 的解决方案**：
1. 借鉴操作系统的**虚拟内存分页机制**。
2. 将 KV Cache 切分为固定大小的 **Block（页）**。
3. Block 可以**离散分布**在显存的任意位置。
4. 通过 **Block Table（页表）** 映射逻辑位置到物理位置。

**效果**：
- 显存浪费降低到 4% 以下（传统方式高达 60-80%）。
- 相同硬件可容纳更多并发请求。
- 支持 KV Cache 在不同请求间**共享**（如 Shared Prefix）。

### Q3：FlashAttention 的核心优化是什么？（⭐⭐⭐⭐⭐）

**参考答案**：

FlashAttention 是一个 **IO-aware** 的精确注意力算法（不是近似）：

**核心优化**：
1. **Tiling（分块计算）**：将 Q、K、V 矩阵分成小块，分批加载到 SRAM（片上缓存）。
2. **避免物化中间矩阵**：传统 Attention 需要写出 $QK^T$ 这个 $N \times N$ 矩阵到 HBM（显存），FlashAttention 在 SRAM 中完成所有计算。
3. **在线 Softmax**：使用数值稳定的在线 Softmax 算法，无需存储完整 Attention 矩阵。

**效果**：
- 速度提升 2-4 倍。
- 内存使用从 $O(N^2)$ 降低到 $O(N)$。
- 支持更长的序列。

### Q4：GQA 和 MQA 的区别？它们如何减少 KV Cache？（⭐⭐⭐⭐）

**参考答案**：

| 方法 | 说明 | KV Cache 大小 |
|------|------|---------------|
| **MHA** | 每个 Head 有独立的 Q、K、V | 最大 |
| **MQA** | 所有 Head 共享同一组 K、V | 最小（1/H） |
| **GQA** | 将 Head 分成 G 组，组内共享 K、V | 中等（G/H） |

GQA 是 MHA 和 MQA 的折中，在保持较好质量的同时显著减小 KV Cache。LLaMA-2-70B 等模型采用 GQA。

### Q5：模型量化的主要方法和权衡是什么？（⭐⭐⭐⭐）

**参考答案**：

**常见量化方法**：
1. **训练后量化（Post-Training Quantization, PTQ）**：
   - GPTQ：基于逐层 Hessian 近似的权重量化。
   - AWQ（Activation-aware Weight Quantization）：考虑激活值重要性的权重量化。
2. **量化感知训练（QAT）**：训练中模拟量化效果。
3. **KV Cache 量化**：将 KV Cache 从 FP16 量化为 FP8/INT8。

**权衡**：
| 量化精度 | 模型大小压缩 | 质量损失 | 推理速度提升 |
|----------|------------|----------|------------|
| FP16→INT8 | ~2x | 轻微 | ~1.5-2x |
| FP16→INT4 | ~4x | 明显 | ~2-3x |

### Q6：vLLM 的核心优势和优化技术有哪些？（⭐⭐⭐⭐⭐）

**参考答案**：

vLLM 是目前最流行的开源 LLM 推理引擎，核心优化包括：

1. **PagedAttention**：离散化 KV Cache 管理，大幅提升显存利用率。
2. **Continuous Batching**：动态批处理，请求完成立即释放资源让新请求加入。
3. **Tensor Parallelism**：多 GPU 分布式推理。
4. **投机解码（Speculative Decoding）**：小模型预测多个 Token，大模型并行验证。
5. **自定义 CUDA Kernels**：优化底层 GPU 运算。
6. **Prefix KV Cache Sharing**：共享相同系统提示的 KV Cache。

### Q7：什么是投机解码（Speculative Decoding）？（⭐⭐⭐⭐）

**参考答案**：

**核心思想**：
1. 用一个轻量级"草稿模型"（Draft Model）快速生成 K 个候选 Token。
2. 将这 K 个 Token 送入大模型进行**并行验证**。
3. 大模型接受或部分拒绝草稿 Token。

**优势**：
- 大模型的验证是并行的（K 个 Token 一次性验证），比自回归逐个生成快。
- 数学上保证输出分布与纯大模型完全一致（无质量损失）。

**前提**：草稿模型的接受率足够高才有加速效果。

### Q8：大模型部署中如何选择推理框架？（⭐⭐⭐⭐）

**参考答案**：

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| **vLLM** | PagedAttention、高吞吐 | 在线高并发服务 |
| **TGI (HuggingFace)** | 与 HF 生态集成好 | 快速部署 HF 模型 |
| **TensorRT-LLM** | NVIDIA 优化 | 追求极致性能 |
| **Ollama** | 简单易用 | 本地开发和测试 |
| **SGLang** | 结构化生成优化 | JSON/结构化输出场景 |

### Q9：Continuous Batching 和 Static Batching 的区别？（⭐⭐⭐⭐）

**参考答案**：

**Static Batching**：凑齐一批请求统一处理。早完成的请求必须等所有请求完成才能返回。

**Continuous Batching**：
- 请求完成后立即释放资源。
- 空出的位置立即让排队的新请求加入。
- GPU 利用率大幅提升。

效果：相比静态批处理，吞吐量可提升 2-10 倍。
