# V4：主动等待净收益门控

V4 复用 V3 的候选生成、能耗核算、DQN 网络和特征 schema，只在策略完成候选选择后增加一层确定性的净收益门控。

## 决策规则

设策略候选为 `selected`，同一快照下最早可行候选为 `earliest`：

```text
objective_gain = ObjectiveScore(selected) - ObjectiveScore(earliest)
wait_penalty = SLA等待权重 × active_wait / 交通日时长
net_gain = objective_gain - wait_penalty
```

- `active_wait <= 1e-12`：不触发门控，保留策略候选。
- `net_gain > SLA最小收益门槛`：接受等待。
- 否则：回退到最早可行候选。

默认等待权重：Hard=`1.0`、Soft=`0.5`、Flexible=`0.25`；默认最小收益门槛均为 `0.0`。这些参数是保守初始值，正式对比前应在验证 seeds 上校准。

## 兼容性

- Candidate feature schema 和 DQN 网络结构不变，可直接加载 V3 checkpoint 做推理消融。
- V4 训练时，门控回退后的候选会同步写回 selection trace，保证实际提交动作与 replay 动作一致。
- V4 决策记录新增 proposed candidate、净收益、等待惩罚、门控结果与原因。

## 命令

```powershell
py -m v4.evaluate_v4 --policy candidate_dqn --model-path artifacts/v3/formal/candidate_dqn_v3_warmup2d_600000.pt --seed 42 --warmup-days 2 --arrival-cutoff 288

py -m v4.train_v4 --steps 600000 --seed 42
```
