# CUDA 对照结论

状态：VALID；PyTorch 2.6.0+cu126；设备：NVIDIA GeForce RTX 3050 Laptop GPU。

## 端到端结果

- Seed 9（每设备 3 轮）：总耗时中位数 CPU 13.966s，CUDA 13.504s，CUDA 加速 1.034×。
- Seed 14（每设备 3 轮）：总耗时中位数 CPU 42.025s，CUDA 37.418s，CUDA 加速 1.123×。
- 所有配对轮次的候选数、replay 特征数、transition、update 和最后 loss 一致：PASS。

## 推理交叉点

- 包含 NumPy→设备→CPU NumPy 往返时，CUDA 首次达到 CPU 的批量约为 1024 个候选。
- 当前 4096 候选/块的热身纯评分 CUDA 加速为 4.671×。
- 短训练的 CUDA NN 阶段仍可能更慢，因为首次 CUDA 初始化和 batch=1 反向传播尚未摊薄。

## 块大小

- Seed 14 的最小核心耗时出现在 chunk=65536：31.894s。
- 选择正式候选块大小 65536；4GB RTX 3050 已完成该负载且语义结果一致。

## 决策

后续训练使用 CUDA 与 chunk=65536，但只将其视为小幅端到端优化。候选枚举仍占核心耗时的大多数，CUDA 不能替代候选生成和样本效率优化。
