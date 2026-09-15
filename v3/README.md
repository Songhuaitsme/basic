# V3 Candidate Generator 优化

V3 只在本目录新增实现，不修改 `v1/`、`v2/` 或 `shared/`。它继续复用冻结的
V1 Task、Candidate、SLA、Feature、DQN 和 Scheduler 接口。

## 当前实现（无损）

1. 将 ReservationSnapshot 中的 CPU allocation 按 Node 分组。
2. 将 Node allocation 转换为 `start -> +power`、`end -> -power` 事件。
3. 使用 sweep line 合并负载、电价和绿电边界，建立六个累计积分前缀。
4. 积分索引覆盖任务完整 SLA 窗口，由 Anchor 与第二轮 Time Sampling 共享。
5. 保持原有 `CPU -> Path -> SLA -> 精确能源/成本` 过滤和计算顺序。

候选集合、成本/绿电公式和 DQN 特征 schema 不变，因此现有模型可以直接由
V3 evaluation 入口加载。动态补贴/碳税计费分支仍回退到冻结的逐候选算法，避免
在没有等价证明的情况下改变非线性计费语义。

## 入口

```powershell
py -m v3.train_v3 --steps 600000 --seed 7 --profile

py -m v3.evaluate_v3 `
  --policy candidate_dqn `
  --model-path artifacts/v2/formal/candidate_dqn_seed7_layered_pool_600000_full.pt
```

输出默认写入 `artifacts/v3/`。

## 后续有损实验（尚未默认启用）

- 便宜近似指标粗筛后，只对少量候选执行精确能源计算。
- 事件驱动 Time Candidate：保留 earliest、SLA、电价、绿电和 allocation 边界。

这两项会改变候选动作空间，应在 V3 内以显式实验模式实现，并重新训练和做固定
Task Trace 对照，不应混入当前无损路径。
