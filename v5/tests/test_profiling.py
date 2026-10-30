# 文件功能：验证 profiling 的行为、边界条件和回归约束。
# 主要内容：ProfilingTest。
# 使用入口：由 unittest 发现并执行，构造依赖后验证断言。
# 模块下游（项目依赖）：v1.train_v1、v5._entrypoint、v5.profiling。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Regression check for profiled warm-up and entry-point restoration."""

import pickle
import unittest

import v1.train_v1 as training
from v5._entrypoint import use_v5_runtime
from v5.profiling import V5TrainingPerformanceProfiler


class ProfilingTest(unittest.TestCase):
    # 作用：验证 warmup_counter_and_legacy_state 场景，覆盖 profiling 的相关约束。
    # 调用约定：由 unittest 发现并调用，使用本文件的夹具和辅助函数。
    # 下游调用：use_v5_runtime、training.TrainingPerformanceProfiler、training._run_profiled_operation、profiler.summary、V5TrainingPerformanceProfiler.__new__ 等。
    def test_warmup_counter_and_legacy_state(self):
        original = training.TrainingPerformanceProfiler
        with use_v5_runtime():
            profiler = training.TrainingPerformanceProfiler()
            result = training._run_profiled_operation(
                profiler, "warmup_scheduler_cycle_count", lambda: "completed"
            )
            self.assertEqual(result, "completed")
            self.assertEqual(profiler.summary(1.0)["counters"]["warmup_scheduler_cycle_count"], 1)
            self.assertEqual(pickle.loads(pickle.dumps(profiler)).counters, profiler.counters)
            state = dict(profiler.__dict__)
            state["counters"] = dict(profiler.counters)
            del state["counters"]["warmup_scheduler_cycle_count"]
            restored = V5TrainingPerformanceProfiler.__new__(V5TrainingPerformanceProfiler)
            restored.__setstate__(state)
            self.assertEqual(restored.counters["warmup_scheduler_cycle_count"], 0)
        self.assertIs(training.TrainingPerformanceProfiler, original)


if __name__ == "__main__":
    unittest.main()
