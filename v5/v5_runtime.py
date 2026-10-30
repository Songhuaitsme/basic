# 文件功能：在 V4 上替换候选生成器，接入决策内缓存及候选池优化配置。
# 主要内容：candidate_config_view、create_v5_runtime。
# 模块上游（导入关系）：v5、v5._entrypoint、v6.v6_runtime、v5.tests.test_candidate_generator。
# 模块下游（项目依赖）：v4.v4_runtime、v5.scheduler.candidate_generator。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""V5 runtime: V4 wait gate with lossless candidate-pool acceleration."""

from pathlib import Path

from v4.v4_runtime import create_v4_runtime

from .scheduler import V5CandidateGenerator


# 作用：返回 V5 候选池优化版本及功能配置。
# 上游（直接调用）：v5._entrypoint.use_v5_runtime.v5_training_config_view、v5.v5_runtime.create_v5_runtime。
def candidate_config_view():
    return {
        "V5_CANDIDATE_OPTIMIZATION_VERSION": V5CandidateGenerator.optimization_version,
        "V5_DECISION_LOCAL_CANDIDATE_CACHE": True,
        "V5_STATIC_CPU_EARLY_REJECTION": True,
        "V5_VECTORIZED_PARETO": True,
    }


# 作用：构建 V4 后替换候选生成器，登记候选池优化配置。
# 上游（直接调用）：v6.v6_runtime.create_v6_runtime。
# 调用上下文：由 V5 适配上下文注入 V1 入口；V6 工厂继续调用本函数替换能源核算。
# 下游调用：create_v4_runtime、candidate_config_view、V5CandidateGenerator、Path(__file__).with_name。
def create_v5_runtime(**kwargs):
    runtime = create_v4_runtime(**kwargs)
    old = runtime.scheduler.candidate_generator
    generator = V5CandidateGenerator(
        old.compute_nodes,
        old.scheduling_cycle_sim,
        old.path_provider,
        old.transmission_model,
        old.calendar,
        old.time_tolerance,
        candidate_mode=old.candidate_mode.value,
        active_wait_enabled=old.active_wait_enabled,
        pool_max_by_sla=old.pool_max_by_sla,
        pool_node_limit_by_sla=old.pool_node_limit_by_sla,
        pool_time_samples_by_sla=old.pool_time_samples_by_sla,
    )
    generator.profiler = old.profiler
    runtime.scheduler.candidate_generator = generator
    runtime.config_view_overrides.update(candidate_config_view())
    runtime.evaluation_code_files += (
        Path(__file__),
        Path(__file__).with_name("profiling.py"),
        Path(__file__).parent / "scheduler" / "candidate_generator.py",
    )
    return runtime
