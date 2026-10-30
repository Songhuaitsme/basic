# 文件功能：扩展训练性能统计字段，兼容 V5 候选优化和恢复后的计时状态。
# 主要内容：V5TrainingPerformanceProfiler。
# 模块上游（导入关系）：v5._entrypoint、v5.tests.test_profiling。
# 模块下游（项目依赖）：v1.profiling。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""V5 profiler compatibility with the training warm-up stage."""

from v1.profiling import TrainingPerformanceProfiler


class V5TrainingPerformanceProfiler(TrainingPerformanceProfiler):
    # 作用：保存依赖与配置，建立实例初始状态。
    # 调用约定：构造实例时由 Python 调用。
    # 下游调用：super().__init__。
    def __init__(self):
        super().__init__()
        self.counters["warmup_scheduler_cycle_count"] = 0

    # 作用：恢复序列化状态，补齐必要索引或默认字段。
    # 调用约定：由 pickle/检查点反序列化流程调用。
    # 下游调用：super().__setstate__。
    def __setstate__(self, state):
        super().__setstate__(state)
        self.counters.setdefault("warmup_scheduler_cycle_count", 0)
