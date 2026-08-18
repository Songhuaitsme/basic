# v1.0模型效果配对分析

状态：VALID_PAIRED_COMPARISON；配对种子数：1。

所有差值均为 treatment - baseline；置信区间为种子级配对Bootstrap。

| 指标 | 基线均值 | 模型均值 | 平均差值 | 95% CI | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| acceptance_rate | 1 | 1 | 0 | [0, 0] | NO_CHANGE |
| completion_rate | 1 | 1 | 0 | [0, 0] | NO_CHANGE |
| reservation_reliability | 1 | 1 | 0 | [0, 0] | NO_CHANGE |
| total_economic_cost_yuan | 46704.405 | 34493.338 | -12211.067 | [-12211.067, -12211.067] | IMPROVED |
| cost_yuan_per_completed_cpu_hour | 13.309078 | 9.8293626 | -3.4797155 | [-3.4797155, -3.4797155] | IMPROVED |
| completed_task_green_coverage | 0.95091823 | 1 | 0.049081771 | [0.049081771, 0.049081771] | IMPROVED |
| system_green_absorption_rate | 0.0005188801 | 0.0012043045 | 0.00068542443 | [0.00068542443, 0.00068542443] | IMPROVED |
| expired_count | 0 | 0 | 0 | [0, 0] | NO_CHANGE |
| failed_count | 0 | 0 | 0 | [0, 0] | NO_CHANGE |
| active_wait_count | 1 | 0 | -1 | [-1, -1] | DIAGNOSTIC |
