# v1.0模型效果配对分析

状态：VALID_PAIRED_COMPARISON；配对种子数：1。

所有差值均为 treatment - baseline；置信区间为种子级配对Bootstrap。

| 指标 | 基线均值 | 模型均值 | 平均差值 | 95% CI | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| acceptance_rate | 0.99636217 | 0.99636217 | 0 | [0, 0] | NO_CHANGE |
| completion_rate | 0.99636217 | 0.99636217 | 0 | [0, 0] | NO_CHANGE |
| reservation_reliability | 1 | 1 | 0 | [0, 0] | NO_CHANGE |
| total_economic_cost_yuan | 49108690 | 47994389 | -1114300.9 | [-1114300.9, -1114300.9] | IMPROVED |
| cost_yuan_per_completed_cpu_hour | 12.790176 | 12.499961 | -0.29021553 | [-0.29021553, -0.29021553] | IMPROVED |
| completed_task_green_coverage | 0.7109507 | 0.70137327 | -0.0095774375 | [-0.0095774375, -0.0095774375] | DEGRADED |
| system_green_absorption_rate | 0.029144092 | 0.02896603 | -0.00017806215 | [-0.00017806215, -0.00017806215] | DEGRADED |
| expired_count | 9 | 9 | 0 | [0, 0] | NO_CHANGE |
| failed_count | 0 | 0 | 0 | [0, 0] | NO_CHANGE |
| active_wait_count | 1949 | 1263 | -686 | [-686, -686] | DIAGNOSTIC |
