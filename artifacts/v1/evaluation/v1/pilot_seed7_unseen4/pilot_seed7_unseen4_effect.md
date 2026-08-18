# v1.0模型效果配对分析

状态：VALID_PAIRED_COMPARISON；配对种子数：4。

所有差值均为 treatment - baseline；置信区间为种子级配对Bootstrap。

| 指标 | 基线均值 | 模型均值 | 平均差值 | 95% CI | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| acceptance_rate | 1 | 1 | 0 | [0, 0] | NO_CHANGE |
| completion_rate | 1 | 1 | 0 | [0, 0] | NO_CHANGE |
| reservation_reliability | 1 | 1 | 0 | [0, 0] | NO_CHANGE |
| total_economic_cost_yuan | 22389.364 | 22698.088 | 308.72341 | [-1091.8559, 2049.3337] | DEGRADED |
| cost_yuan_per_completed_cpu_hour | 8.0043112 | 8.0933982 | 0.089086936 | [-0.22565254, 0.4728878] | DEGRADED |
| completed_task_green_coverage | 0.94864381 | 1 | 0.051356187 | [-1.110223e-16, 0.15332217] | IMPROVED |
| system_green_absorption_rate | 0.0014133498 | 0.0016376231 | 0.00022427332 | [-0.00028590632, 0.00099572347] | IMPROVED |
| expired_count | 0 | 0 | 0 | [0, 0] | NO_CHANGE |
| failed_count | 0 | 0 | 0 | [0, 0] | NO_CHANGE |
| active_wait_count | 1 | 6.25 | 5.25 | [3.75, 6.5] | DIAGNOSTIC |
