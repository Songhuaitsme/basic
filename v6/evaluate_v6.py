# 文件功能：在 V6 运行时适配上下文中调用 V1 冻结策略评估主流程，设置版本及输出参数。
# 主要内容：main。
# 使用入口：命令行执行，或由其他脚本导入后调用。
# 模块下游（项目依赖）：v1.evaluate_v1、v6._entrypoint。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""V6 evaluation entry point, compatible with existing V3/V4/V5 models."""

from v1.evaluate_v1 import main as _evaluation_main

from ._entrypoint import use_v6_runtime


# 作用：解析或转发命令行参数，组织本文件执行流程。
# 调用约定：本文件命令行入口或上层版本入口调用。
# 下游调用：use_v6_runtime、_evaluation_main。
def main():
    with use_v6_runtime():
        _evaluation_main(
            system_version="6.0",
            default_output="artifacts/v6/evaluation/report.json",
            default_audit_mode="periodic",
            default_report_mode="compact",
            default_candidate_chunk_size=65536,
        )


if __name__ == "__main__":
    main()
