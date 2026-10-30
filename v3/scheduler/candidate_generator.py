# 文件功能：提供继承 V1 实现的 V3 候选生成器类型，作为运行时替换入口。
# 主要内容：V3CandidateGenerator。
# 模块上游（导入关系）：v3.scheduler、v3.v3_runtime、v5.scheduler.candidate_generator、v5.tests.test_candidate_generator。
# 模块下游（项目依赖）：v1.scheduler.candidate_generator。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""V3 candidate generator using the frozen V1 candidate semantics."""

from v1.scheduler.candidate_generator import CandidateGenerator


class V3CandidateGenerator(CandidateGenerator):
    """Marker implementation for the lossless V3 candidate pipeline.

    Feasibility remains ordered as CPU, Path, SLA, then exact accounting.  The
    expensive accounting implementation is supplied by the V3 runtime, so no
    V1 candidate-generation source needs to be edited or copied.
    """

    optimization_version = "sweep-prefix-full-sla-v1"
