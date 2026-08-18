# v1.0模型效果配对分析

状态：VALID_PAIRED_COMPARISON；配对种子数：5。

所有差值均为 treatment - baseline；置信区间为种子级配对Bootstrap。

| 指标 | 基线均值 | 模型均值 | 平均差值 | 95% CI | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| acceptance_rate | 0.99770115 | 0.99770115 | 0 | [0, 0] | NO_CHANGE |
| completion_rate | 0.99770115 | 0.99770115 | 0 | [0, 0] | NO_CHANGE |
| reservation_reliability | 1 | 1 | 0 | [0, 0] | NO_CHANGE |
| total_economic_cost_yuan | 2035832.4 | 2001948.1 | -33884.261 | [-46196.367, -20685.149] | IMPROVED |
| cost_yuan_per_completed_cpu_hour | 12.018552 | 11.82333 | -0.19522232 | [-0.26123274, -0.14165422] | IMPROVED |
| completed_task_green_coverage | 0.59299114 | 0.6171194 | 0.024128265 | [-0.029123663, 0.11236817] | IMPROVED |
| system_green_absorption_rate | 0.0035575616 | 0.007029774 | 0.0034722124 | [0.0020555537, 0.0053842584] | IMPROVED |
| expired_count | 0.2 | 0.2 | 0 | [0, 0] | NO_CHANGE |
| failed_count | 0 | 0 | 0 | [0, 0] | NO_CHANGE |
| active_wait_count | 33.6 | 9.8 | -23.8 | [-28.4, -19.8] | DIAGNOSTIC |
