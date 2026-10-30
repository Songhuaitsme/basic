# V7：静态路径查询与重试网格复用

V7 在 V6 能源核算、V5 候选池和 V4 等待门控基础上减少静态路径准备开销。
V6 保持为对照版本。候选、特征和模型 schema 保持原样，可使用现有 V6 模型推理。

## 优化内容

- 缓存源节点、目标节点和带宽可行图对应的路径，包括无路径结果，并复用不可变 PathSpec。
- 带宽不按近似值舍入，而按链路容量分界点划分：只有过滤后保留的链路完全一致时才共享缓存。
- 同一带宽可行图只复制、过滤一次；缓存均有 LRU 容量上限。
- 同一任务重试时复用路径与传输时长，每次仍根据当前决策时间重新计算 first/last 网格边界。
- K=1 且边权为严格正整数、路径和可精确表示时，使用单源 Dijkstra 的前驱关系证明最短路径唯一。
  只有唯一最短路径直接使用单源结果；并列、浮点或零边权、K>1 等情况沿用原 shortest_simple_paths。
- 缓存和重试网格不写入训练 checkpoint，恢复后按需重建；精确训练恢复使用 V7 checkpoint。

路径提供器持有独立拓扑快照。动态修改容量、距离、成本或连通关系后，必须调用
`provider.replace_topology(new_graph)` 切换版本并清空缓存；不要直接修改 provider.graph。
这种显式失效不需要每次查询都重新扫描或哈希整张图。

## 执行

在仓库根目录执行一次 V7 评估，固定 V6 模型、CPU、seed=42、预热两天和正式一天：

```powershell
py -m v7.evaluate_v7 --policy candidate_dqn --model-path artifacts/v6/logs/candidate_dqn.pt --device cpu --seed 42 --warmup-days 2 --arrival-cutoff 288 --output artifacts/v7/evaluation/candidate_dqn_seed42.json
```

推荐的 V6/V7 性能对照：串行交替运行各三次，默认不启用高开销的详细计时。

```powershell
py -m v7.compare_v6_v7
```

默认输出到 `artifacts/v7/evaluation/comparison_v6_v7_seed42/`：

- `trial_01_v6.json`、`trial_01_v7.json` 等六份原始报告。
- `comparison.csv`：累计耗时、候选链路耗时、决策均值/P95、缓存命中和实际搜索次数。
- `comparison.json`：中位耗时、下降比例、精确一致性验证和完整缓存计数。

比较器允许版本、代码和优化配置哈希不同，但同一版本各轮必须一致；任务、外生轨迹、
拓扑、依赖、模型等相同，全部 metrics 和除决策耗时外的所有决策字段必须一致。
不一致时停止汇总，不输出性能成功结论。已有 trial 文件时拒绝覆盖，可使用 `--output-dir` 换目录。

中断后继续运行：

```powershell
py -m v7.compare_v6_v7 --resume
```

续跑会先校验已有报告的模型、代码、配置、版本和实验条件，跳过有效报告，只执行剩余轮次。
若中断产生了不完整 JSON，原文件会备份为 `.interrupted-*` 后重跑该轮。代码、模型或配置
已经改变时拒绝复用旧结果，应指定新的输出目录；`--resume` 不会重新运行已验证完成的轮次。

只重新汇总已有结果：

```powershell
py -m v7.compare_v6_v7 --summarize-only
```

同一 V7 入口关闭本轮优化做消融：增加 `--disable-path-search-cache`，此时复用 V6 的路径和网格实现。

已有细分计时开关 `--candidate-profile-detail` 仍可使用；网格阶段额外细分为路径查询、
静态传输时长准备、当前时间边界计算。鉴于详细计时已有较大扰动，不用于性能对照。
低开销缓存统计始终输出到 `diagnostics.runtime_summary.path_search_cache`。

训练入口：

```powershell
py -m v7.train_v7 --steps 600000 --seed 42
```

## 轻量验证

```powershell
py -m unittest v7.tests.test_path_cache v7.tests.test_comparison
```

验证路径及顺序、并列回退、带宽边界、负缓存、LRU、拓扑失效、pickle、任务重试、
三类 SLA 的候选及门控决策一致性。端到端加速比例需以实际对照评估为准。
