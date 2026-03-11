# 第 5 章：规划与推理

本章涵盖 Agent 是如何做复杂任务分解和自我反思的。

## 核心知识点总结
- ReAct 框架详解
- Chain-of-Thought / Tree-of-Thought
- 任务分解与规划策略
- 反思与自我纠错机制

## 高频面试题

### 1. 简述 ReAct（Reasoning and Acting）框架的执行流程。
**参考答案**：
ReAct 将推理（Reasoning）和行动（Acting）交替进行。标准流程为：
1. **Thought（思考）**：大模型根据当前状态思考下一步需要做什么。
2. **Action（动作）**：大模型选择工具并生成参数进行调用。
3. **Observation（观察）**：执行工具后获得环境反馈。
然后不断循环（Thought -> Action -> Observation），直到获得最终答案（Finish）。该框架使得模型在执行前能够有逻辑地推演，大大提升了解决复杂问题的成功率。

### 2. Chain-of-Thought (CoT) 的常见做法有哪些？
**参考答案**：
1. **Zero-shot CoT**：在 Prompt 最后加上“Let's think step by step”（让我们一步一步思考）。
2. **Few-shot CoT**：在系统提示词中提供几个包含推理过程的示例。

*(更多面试题持续补充中...)*
