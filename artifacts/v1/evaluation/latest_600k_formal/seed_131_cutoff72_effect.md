# v1.0模型效果配对分析

状态：VALID_PAIRED_COMPARISON；配对种子数：1。

所有差值均为 treatment - baseline；置信区间为种子级配对Bootstrap。

| 指标 | 基线均值 | 模型均值 | 平均差值 | 95% CI | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| acceptance_rate | 0.99220779 | 0.99220779 | 0 | [0, 0] | NO_CHANGE |
| completion_rate | 0.99220779 | 0.99220779 | 0 | [0, 0] | NO_CHANGE |
| reservation_reliability | 1 | 1 | 0 | [0, 0] | NO_CHANGE |
| total_economic_cost_yuan | 10594137 | 10637515 | 43378.038 | [43378.038, 43378.038] | DEGRADED |
| cost_yuan_per_completed_cpu_hour | 12.314651 | 12.365073 | 0.050422738 | [0.050422738, 0.050422738] | DEGRADED |
| completed_task_green_coverage | 0.61030276 | 0.69601151 | 0.085708755 | [0.085708755, 0.085708755] | IMPROVED |
| system_green_absorption_rate | 0.013079038 | 0.010800385 | -0.0022786523 | [-0.0022786523, -0.0022786523] | DEGRADED |
| expired_count | 3 | 3 | 0 | [0, 0] | NO_CHANGE |
| failed_count | 0 | 0 | 0 | [0, 0] | NO_CHANGE |
| active_wait_count | 283 | 81 | -202 | [-202, -202] | DIAGNOSTIC |
