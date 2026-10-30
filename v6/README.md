# V6：精确能源查询加速

V6 继承 V5 候选池、V3 sweep-line 前缀积分、V4 等待门控和现有 DQN。
本轮优化能源核算中的两个重复开销，保留候选集、采样、评分、特征和模型 schema。

- 候选能源查询缓存以快照和预测对象身份为键，不再计算整个资源分配快照的哈希。
- 缓存同时持有源对象，避免对象 ID 复用；新快照、新电价或绿电预测使用新的缓存项。
- 积分索引建立时一次性保存 float64 的只读边界、速率和前缀数组。
- 批量查询复用数组，原 sweep-line、前缀累加顺序、积分公式和边界容差保持一致。
- 缓存有容量上限，保存训练 checkpoint 时清空；恢复后按需重建。
- 非线性补贴与碳税计费继续走原有算法。

默认输出位于 `artifacts/v6/`。推理可以加载现有 V5 模型；完整训练恢复使用 V6 的
resume checkpoint。训练默认不启用 profile。评估沿用现有入口的基础耗时统计。

## 使用

训练（预热设置保持与 V5 默认一致）：

```powershell
py -m v6.train_v6 --steps 600000 --seed 42
```

用同一个 V5 模型评估，便于比较运行时改造，预热两天、正式评估一天：

```powershell
py -m v6.evaluate_v6 --policy candidate_dqn --model-path artifacts/v5/logs/candidate_dqn.pt --seed 42 --warmup-days 2 --arrival-cutoff 288 --device cpu
```

若评估新训练的 V6 模型，将模型路径改为 `artifacts/v6/logs/candidate_dqn.pt`。
固定模型比较更适合判断运行时优化收益；重新训练后的模型可能产生不同决策。

轻量验证：`py -m unittest v6.tests.test_energy`。
实际端到端加速比例尚未测量，不预设速度提升保证。

## 候选链路细分计时

开启 `--candidate-profile-detail` 后，新增分层阶段统计，默认关闭。可行性总计时的补齐
对基础统计也生效；候选语义不变。只支持 `layered_pool`。

单次细分评估（在仓库根目录执行）：

```powershell
py -m v6.evaluate_v6 --policy candidate_dqn --model-path artifacts/v6/logs/candidate_dqn.pt --device cpu --seed 42 --warmup-days 2 --arrival-cutoff 288 --candidate-profile-detail --output artifacts/v6/profiling/candidate_detail_seed42/detail.json
```

输出 `detail.json` 和 `detail.candidate_detail.csv`。JSON 细分位置为
`diagnostics.runtime_summary.candidate_detail`。CSV 按 `self_seconds` 降序排列，
包括外层未覆盖的剩余项，因而 `self_seconds` 的总和对应候选链路总耗时。
`inclusive_seconds` 包含子步骤，不能相加。`calls` 是计时区间次数，精确指标阶段可能
一次处理多个候选；实际候选数量、缓存命中、通过/拒绝等见 `counters`。

细分覆盖 Anchor、第二轮采样、最终候选流；候选流仅在消费批次内计时，不计入随后
的特征编码和 DQN 推理。使用现有叶子计时，避免重复计时和重复累计。
`outside_scopes_seconds`、`reconciliation_error_seconds`、`negative_self_count`
用于审计计时是否闭合；明显重叠会将细分 `status` 标记为 `INVALID`。

固定模型、CPU、seed=42、预热两天、正式一天，交替执行详细统计开关各三次：

```powershell
py -m v6.profile_candidates
```

此命令会启动 **6 次完整评估**，期间请保持模型文件和源码不变。基础 profiler 始终开启，
对照仅关闭新增的详细统计。默认目录为 `artifacts/v6/profiling/candidate_detail_seed42/`，
已有 trial 文件时拒绝覆盖；可用 `--output-dir` 指定新目录。输出：

- `trial_01_off.json` / `trial_01_on.json` 等原始报告及每次 on 对应的细分 CSV。
- `comparison.json` / `comparison.csv`：条件与决策一致性、每次耗时、中位数和额外统计开销。
- `candidate_detail.csv`：候选链路耗时处于中间位置的实际 on 运行，保持表内可加性。

比较器核对模型、代码、配置、轨迹、拓扑、依赖等哈希，并比较全部质量指标和除
`decision_wall_seconds` 外的全部决策字段。不一致或细分计时无效时停止汇总。
若额外计时开销超过 5%，结果会明确标记，不能把它当成无开销的性能基准。

只重新汇总已有报告：

```powershell
py -m v6.profile_candidates --summarize-only
```

若未归因比例仍超过 5%，可显式允许一次独立 cProfile 诊断（这次运行不参与开关对照）：

```powershell
py -m v6.profile_candidates --summarize-only --cprofile-if-needed
```

诊断输出 `diagnostic.pstats`、`diagnostic_functions.csv`（按函数自身耗时排序）与
`diagnostic_report.json`。细分统计的开销和历史计时不能混合；历史 428.19 秒无法回填。

轻量测试：

```powershell
py -m unittest v6.tests.test_candidate_profiling v6.tests.test_profile_candidates v5.tests.test_candidate_generator v5.tests.test_profiling v6.tests.test_energy
```
