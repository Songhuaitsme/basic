# V5：保持候选语义的启发式生成加速

V5 继承 V4 的等待净收益门控、V3 的能源积分实现和现有 DQN。
保留 layered_pool 的节点排名、时间采样、分层名额、Pareto 支配规则、
候选顺序及并列处理；complete 模式继续走原有完整枚举。

这版实施第一阶段的等价优化，以 V4 的现有候选池为基准：

- Anchor 与第二轮采样共享候选记录，重复点不再做 CPU/路径检查或生成 ID。
- 同一生成上下文与计费器共享已物化的候选，重复点不再计算能源/成本。
- 节点总 CPU 容量确定不足时提前拒绝，采用覆盖原两种容差的保守条件。
- 成本、绿电、负载的归一化边界只计算一次。
- Pareto 比较以 NumPy 分块执行，保留原有严格支配关系及输入顺序。

缓存只活跃于单次 layered_pool 准备过程，正常返回或异常时均清理。
生成器的 `last_generation_stats` 提供最近一次记录/候选缓存命中及 CPU 提前拒绝计数。
现有 profiler 可查看精确指标计算和对象构造数量的变化。

未加入基于近似评分的 Top-K 截断或 Q 值上界剪枝；这类剪枝需要额外的可靠上界，
而且生成阶段还必须保持原节点排名和候选池成员。当前版本不宣称相对全枚举的全局最优，
也不预设端到端加速比例。

## 使用

```powershell
py -m v5.train_v5 --steps 600000 --seed 42 --profile

py -m v5.evaluate_v5 --policy candidate_dqn --model-path artifacts/v3/formal/candidate_dqn_v3_warmup2d_600000.pt --seed 42 --warmup-days 2 --arrival-cutoff 288
```

默认输出到 `artifacts/v5/`。模型和特征 schema 保持原样，可加载 V3/V4 模型做推理；
完整训练恢复使用 V5 自己生成的 resume checkpoint。

轻量验证：`py -m unittest v5.tests.test_candidate_generator`。
