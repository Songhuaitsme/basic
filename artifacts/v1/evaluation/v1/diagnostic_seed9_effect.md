# v1.0模型效果配对分析

状态：VALID_PAIRED_COMPARISON；配对种子数：1。

所有差值均为 treatment - baseline；置信区间为种子级配对Bootstrap。

| 指标 | 基线均值 | 模型均值 | 平均差值 | 95% CI | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| acceptance_rate | 1 | 1 | 0 | [0, 0] | NO_CHANGE |
| completion_rate | 1 | 1 | 0 | [0, 0] | NO_CHANGE |
| reservation_reliability | 1 | 1 | 0 | [0, 0] | NO_CHANGE |
| total_economic_cost_yuan | 1068.2833 | 1214.1151 | 145.83182 | [145.83182, 145.83182] | DEGRADED |
| cost_yuan_per_completed_cpu_hour | 7.3668488 | 8.3725006 | 1.0056517 | [1.0056517, 1.0056517] | DEGRADED |
| completed_task_green_coverage | 1 | 1 | 0 | [0, 0] | NO_CHANGE |
| system_green_absorption_rate | 0.00036199542 | 0.00010559866 | -0.00025639675 | [-0.00025639675, -0.00025639675] | DEGRADED |
| expired_count | 0 | 0 | 0 | [0, 0] | NO_CHANGE |
| failed_count | 0 | 0 | 0 | [0, 0] | NO_CHANGE |
| active_wait_count | 0 | 2 | 2 | [2, 2] | DIAGNOSTIC |
